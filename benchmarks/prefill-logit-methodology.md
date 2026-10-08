# Prefill-logit (KLD) validation gate

Why this file exists: on 2026-10-08, issue #113 showed that the existing same-seed coherence gate and
the MTP gate are **blind to a prefill-logit divergence**.  The BF16/WMMA chunked GDN path was shipping
as the default and shifted the prefill logits substantially (mean KLD 0.032 on gfx1201, 0.62 on
gfx1100) while decode was clean, the greedy text looked normal, and every op-level oracle
(`test-backend-ops -o GATED_DELTA_NET`, 46/46) passed.  Only a KL/perplexity comparison against a
known-good base saw it.  This file is the prefill-logit gate: the protocol, the canonical command, the
thresholds, and the scope of changes it must run for.

Status: **2026-10-08 (r29) — gate defined and mandatory.**  Run it before tagging a release whose block
set touches a prefill kernel (and after any change to the delivery's prefill dispatch).  The first
recorded run is the r29 issue-#113 example (`WORKLOG.md` 2026-10-08 r29).

**Automation.**  `scripts/gate-prefill-logits.sh` implements this protocol and is the canonical entry
point:

```bash
./scripts/gate-prefill-logits.sh --record                 # establish the known-good base (writes $BASE.meta)
./scripts/gate-prefill-logits.sh                          # compare the current build against the base
./scripts/gate-prefill-logits.sh --ab "GGML_CUDA_GDN_CHUNKED_BF16=0"   # exact-fallback A/B
```

`--record` writes a `$BASE.meta` sidecar (build ref, model, corpus sha256, ctx/chunks, base sha256, PPL)
and refuses a compare whose model/corpus/settings do not match the base unless `--allow-base-mismatch`
is given.  The manual commands below are exactly what the script runs; use them directly only if the
script's pins do not fit.

## What it catches

Any change that moves the prefill forward pass enough to shift the prompt logits, but not enough to
change greedy decode text or trip a `test-backend-ops` oracle: a new or changed prefill kernel family
(GDN/SSM chunked, flash attention, MMQ/MMVF/MMB prefill GEMMs, MoE prefill), a changed reduction
order, an accumulator or recurrent-state precision change, or a new default-on approximate path.

The coherence gate samples the decode distribution *starting from the post-prefill state*; a uniform
prefill shift is invisible to it.  The MTP gate exercises decode/verify shapes, not the prefill
kernels.  This gate is the missing third axis.

## The protocol

Record reference logits from a known-good build (the previous release, or the exact/sequential
fallback arm), then compare the candidate cell by cell.  `llama-perplexity` reports the mean KL
divergence, the same-top-p fraction and the per-chunk table.

### Pinned model + corpus + flags

| item | value |
|---|---|
| model (gfx1201, 32 GiB) | `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` |
| model (24 GiB cards, e.g. gfx1100) | `/llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf` |
| corpus | `/llm/models/wikitext-2-raw/wiki.test.raw` (a code corpus is a useful second arm; issue #113 reproduced on both) |
| flags | `-c 512 --chunks 40 -ngl 99 -fa on -t 8` |
| device pin | `HIP_VISIBLE_DEVICES=0` (single GPU, so the result is card-local) |

40 x 512 chunks is the r29 calibration: it separates mean KLD 0.03 from 0.0005 in well under a minute
of compute per arm.  Do not shrink it without re-calibrating; a shorter window is noisier.

### 1. Record the base (once per known-good build)

```bash
HIP_VISIBLE_DEVICES=0 ./build/bin/llama-perplexity -m <model> -f <corpus> \
  -c 512 --chunks 40 -ngl 99 -fa on -t 8 --kl-divergence-base /tmp/prefill-base.kld
```

The base file is large (multi-GB); it is a scratch artifact and is **not** committed.  Record its
sha256, the build id and the base PPL in the release's `WORKLOG.md` entry.

### 2. Run the candidate

```bash
HIP_VISIBLE_DEVICES=0 ./build/bin/llama-perplexity -m <model> -f <corpus> \
  -c 512 --chunks 40 -ngl 99 -fa on -t 8 \
  --kl-divergence-base /tmp/prefill-base.kld --kl-divergence
```

### 3. Acceptance

| metric | threshold |
|---|---|
| mean KLD | `<= 0.005` |
| same top p | `>= 98.0 %` |
| max KLD | inspect only; no hard limit (a single tokenizer/context edge outlier is not a regression) |

r29 calibration: the clean fp32 chunked GDN arm measured mean KLD **0.00054** / same-top-p **98.8 %**
(gfx1201, 27B Q8_0) and **0.000037** / **99.7 %** (gfx1100, Q4_K_M); the rejected bf16 arm measured
**0.0324** / **93.6 %** and **0.6228** / **79.4 %**.  The thresholds sit ~10x above the clean value and
well below the regression.

### 4. Exact-fallback A/B (when the change adds or edits an approximate kernel)

When the change introduces or modifies a kernel that has an exact or sequential fallback, additionally
compare the candidate's **default** against that **fallback** arm directly (for GDN: `GGML_CUDA_GDN_CHUNKED=0`)
and require the same thresholds.  This is the test that would have caught issue #113: the fallback arm
was clean, the default was not.  A change is only allowed to be a default-on approximate path if this
A/B passes; otherwise it stays opt-in.

## Scope: when the gate is required

Run it before a release (and after any change to `patches/`) whenever the block set touches:

- the GDN/SSM chunked prefill (`gated_delta_net*`);
- the flash-attention prefill kernels or their dispatch;
- MMQ / MMVF / MMB prefill GEMMs or a new quant path;
- MoE prefill, the shared-expert band, or the expert cache;
- any default-on approximate path (a precision change or a reduction-order change).

Not required for decode-only changes (the MTP gate covers those) or doc-only changes.

## Recording

Add the base build id/hash and the candidate mean KLD / same-top-p for each arm to the release's
`WORKLOG.md` entry.  If the gate fails, the change does not ship as a default; either fix it, make it
opt-in, or flip the existing default off (the r29 resolution).
