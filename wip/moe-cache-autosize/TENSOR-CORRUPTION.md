# `-sm tensor` corruption on a wide prefill ubatch — handover (open)

**Status: OPEN, CRITICAL (silent wrong output). Opened 2026-10-06 (r15 session). TODO #41.**
**Nothing here is in the delivery** (the `wip/` rule): the r15 release (`v16-a55e952b8-r15`) fixes the
`-sm tensor` CPU fallback + the slow cache (TODO #40), *not* this.

Read this first if you are picking up the `-sm tensor` prefill corruption. It records the exact repro,
the bisected threshold, everything already ruled out (with the commands, so you do not repeat them), the
false lead that cost most of the r15 session, and the hypotheses left.

---

## 1. TL;DR

A single prefill ubatch **wider than ~3600 tokens** under `-sm tensor` makes the **target model itself**
produce garbage (`////////////////////////////////`). `-sm layer` is fine at the same width, so it is specific
to the **Meta-split prefill path**. It is pre-existing (the r14 binary reproduces it), unrelated to the
expert cache, and it also destroys MTP acceptance — which is why it first looked like "MTP is broken".

It is a **silent corruption**: the model answers, it is just wrong. Treat it as a correctness bug, not a
perf bug, and gate any fix on output coherence (`////`=0) at a wide ubatch.

## 2. Exact repro

Model: Qwen3.8-Flash-Next **UD-IQ4_XS** (`/llm/models/Qwen3.8/Flash-Next/IQ4_XS/`,
`...-UD-IQ4_XS-00001-of-00003.gguf`) + its MTP head
(`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`). 2 GPU, cache auto.

```bash
# a ~3800-token prompt (build it by repeating prompts/code-python.txt; ~3.8 chars/token)
python3 - <<'PY'
base = open('/home/stew675/llama-cpp-rdna-boosts/prompts/code-python.txt').read()
open('/tmp/pl_3800.txt','w').write(base*266 + "\n\nWrite a short summary.")   # ~3800 tokens
PY

cd ~/llama-r13   # the r15 16-block chain
D=/llm/models/Qwen3.8/Flash-Next/IQ4_XS
M=$D/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
T=$D/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf

HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib \
./build/bin/llama-cli -m $M -md $T -ngl 99 -sm tensor -ncmoe 48 \
  -fa on -ctk q8_0 -ctv q8_0 -t 8 -c 32768 -b 4096 -ub 4096 \
  --lazy-mode off --load-mode auto --spec-type draft-mtp --spec-draft-n-max 3 \
  -f /tmp/pl_3800.txt -n 32 --seed 42 --temp 0 --reasoning off \
  --single-turn --no-display-prompt --no-warmup > /tmp/out.txt 2>/dev/null
grep -c '////' /tmp/out.txt     # 8 = CORRUPT;  0 = coherent
```

`--no-warmup` is not required but keeps the run short. The signal is `////` in stdout (the delivery's
own degenerate-output marker — the same one the `c32K/c128K` coherence gate uses).

## 3. The threshold (bisected)

`-sm tensor`, `-ub 4096`, one prefill ubatch, `-c 32768`:

| prompt tokens | `////` | note |
|---|---:|---|
| 6 | 0 | coherent (single tiny ubatch) |
| 400 – 2800 | 0 | coherent |
| 3000 – 3600 | 0 | coherent |
| **~3800** | **8** | **CORRUPT** |
| 4000, 5000, 6000, 7000, 16000 | 8 | CORRUPT |

The trigger is the **ubatch width**, not `-ub` and not the ubatch *count*:
* `-ub 4096` + a 3800-token prompt = **one** ubatch → corrupt.  So it is not the cross-ubatch bridge.
* `-ub 8192` + a 3800-token prompt → the ubatch is still 3800 → also corrupt (same threshold).
* `-ub 2048` + a 3800-token prompt = two ubatches (2048 + 1752) → **coherent**.
* `-b 8192 -ub 2048` → ubatch 2048 → coherent.  It is `-ub` (the ubatch), not `-b`.

Narrow it further first (the exact boundary is not known; ~3600 works, ~3800 breaks).  Candidates worth
checking against a plausible constant: 3584 (= 4096 − 512), 3640, 3712.

## 4. `-sm layer` is NOT affected (the decisive control)

Same prompt, same run flags, only `-sm tensor` → `-sm layer`:

| config | `////` |
|---|---:|
| `-sm tensor -ub 4096` | **8** |
| `-sm layer  -ub 4096` | **0** |
| `-sm layer  -ub 2048` | 0 |
| `-sm tensor -ub 2048` | 0 |

So the bug lives in **the Meta-split prefill path** (the tensor-split buffer/scheduler/FA arrangement),
not in the model, not in the kernels generically.

## 5. Ruled out (do not re-test these)

All at `-sm tensor -ub 4096`, 3800-token prompt:

| ruled out | command / evidence | result |
|---|---|---|
| the expert cache | `MOE_EXPERT_CACHE_MIB=0` | still `////` |
| a r15 regression | the r14 binary (`~/llama.cpp/build-rocm/bin/llama-cli`) | still `////` |
| the derived KQ mask | `LLAMA_KQ_MASK_DERIVED=0` | still `////` |
| QSA sparse FA | `LLAMA_QSA_SPARSE_FA=0` (and combined with the above) | still `////` |
| GDN chunked prefill | `GGML_CUDA_GDN_CHUNKED=0` | still `////` |
| the ubatch *count* / cross-ubatch bridge | a 4k prompt is one ubatch and still breaks | n/a |
| `-b` (batch) | `-b 8192 -ub 2048` is coherent | n/a |
| `-c` (context) | broken at `-c 32768` and at `-c 8192` | n/a |
| leftover processes / VRAM pressure | `pgrep -cf 'bin/llama'` = 0, `rocm-smi --showmemuse` = 0 %, 172 GB RAM free | n/a |
| a load-time OOM (a separate, also-open issue) | the failing loads here complete; the arena prints normally | n/a |

**Still un-ruled-out environment switches worth trying next** (all cheap): `GGML_SCHED_DEVGATHER`,
`LLAMA_MMAP_HOST_EXPERTS=0`, `GGML_SCHED_EVENTS=0`, `-fa off` if the model allows it, `-ctk/-ctv f16`
(to remove the q8_0 KV from the picture), and `--load-mode none` (to see whether the pageable path
changes it).

## 6. The false lead: "MTP never accepts" (read this so you do not re-chase it)

What the r15 session first saw was `#mean acc len = 1.00` / `draft_n_accepted = 0` of 3063 under
`-sm tensor` with a wide prompt, and it looked like an MTP-specific bug.  It is not:

1. The **target's output is already corrupt** (`////`) at the same width, so the target's hidden state is
   garbage and every draft is correctly rejected.  Fixing the corruption should restore acceptance.
2. **The draft side is provably fine.**  With a *narrow* prompt (6 tokens) at `-ub 4096`, the draft
   candidates are **byte-identical** to `-ub 2048`:
   `1144 ' need' (1.000)`, `1902 ' must'`, `16546 'need'`, `4087 ' answer' (0.781)`, `310 ' to'`,
   `5707 ' respond'`, and the acceptance is normal (3.50).
3. **Acceptance is not a verdict below `-n 2000`** (`mtp-adaptive-methodology.md` rule 0).  At `-n 32`
   *every* config, including working ones, reads `1.00`.  Do not gate this bug on acceptance; gate it on
   **`////`=0**.

## 7. Hypotheses left (ordered by how cheap they are to test)

1. **The wide-ubatch staging path.**  The block-06 op-offload H2D staging ring uploads the host expert
   weights for the prefill, has a width gate floored at **64 tokens**, and under `-sm tensor` the meta
   backend owns one **MIRRORED** ring per device that must serve **arbitrary byte ranges**.  The recorded
   invariant is "a **partially staged ubatch is never allowed**" — a width where the stage/serve split
   silently drops or truncates a range is the leading candidate.  Instrument the ring's per-split
   staged/total byte accounting at ~3600 vs ~3800 tokens and compare across the two `-sm` modes.
2. **The Meta split geometry at a wide query batch.**  `-sm layer` is immune, so the difference is the
   split.  Check whether the split's per-device slice boundaries are recomputed per ubatch width and
   whether some buffer is sized from the *first* (small) graph and then reused at the wide one
   (`ggml-backend-meta.cpp` buffer split / `stage_input`).
3. **FA with a wide query batch under a split.**  The FA launcher switches family at `Q->ne[1] > 8`
   (WMMA) and again at the mask boundary; a wide `n_q` with a split K/V head is where the QSA/derived
   mask work lives.  The derived-mask and QSA switches are ruled out, but the split *head* arrangement
   is not.
4. **Numerics, not logic.**  Compare the wide-ubatch logits under `-sm tensor` against `-sm layer` and
   against a `-ub 2048` chunked reference (same seed, greedy) to see whether the corruption is a
   truncated/NaN region (logic) or a drift (numerics).  `test-backend-ops` at the exact shapes is the
   oracle if a kernel is suspected.

## 8. Performance context (why this bug matters for the R9V comparison)

A large `-ub` is the **prefill lever** when the experts are host-resident (the staging ring amortizes
over the ubatch). 2 GPU IQ4_XS, 16k prompt, coherent-only numbers:

| config | prefill | decode (n=2048) | coherent |
|---|---:|---:|---|
| `-sm layer -ub 2048` | 379 | **67.1** | yes |
| `-sm layer -ub 8192` | **1040** | 42.8 | yes |
| `-sm tensor -ub 8192` | **1570** | — | **no** (this bug) |
| Reddit R9V reference (theirs is slower hardware) | 1166 | 57.3 | yes |

The decode drop at a large ubatch is the **same knob**: the wider ubatch permanently reserves a bigger
compute buffer (3810 → 11339 MiB), the auto cache arena yields (42130 → 26707 MiB under `-sm layer`;
16908 MiB under `-sm tensor`), and residency — hence decode — falls.  That tension is TODO #42; this
document is only about the corruption (TODO #41).

**Note on the comparison:** the R9V numbers come from the Reddit report on a 96 GB DDR4-3200 / 3900X
host, on the same UD-IQ4_XS, via an OpenAI server with 5 requests per context; ours are `llama-cli`
`[ Prompt | Generation ]` medians on much faster hardware over one run.  Treat the shape as indicative
and re-measure like-for-like before quoting a win.

## 9. Practical traps recorded this session (save yourself the time)

* **Leftover processes.**  A killed/timed-out `llama-cli` keeps ~24 GB VRAM per GPU and silently halves
  every subsequent number (an early `-ub 4096` read 269/19.5 instead of ~460/69).  Always
  `pgrep -af 'bin/llama'` and `rocm-smi --showmemuse` before trusting a measurement; `pkill -9 -f` on a
  pattern that also matches your own shell kills the shell (exit 137) — use `[l]lama-cli`.
* **`-p "$(cat bigfile)"` hits "Argument list too long"** above ~128 KB.  Use `-f <file>`.
* **`llama-cli -v` echoes the prompt**, so a naive "generated text" slice can be the prompt.  Slice with
  `sed -n '/^> /,/\[ Prompt:/p' file | sed '1d;$d'`, or run without `-v`.
* **Short-run acceptance is meaningless** (see §6.3).  For coherence use `////`; for acceptance use
  `-n >= 2000`.
* The r15 build of record is `~/llama-r13` (branch `rdna-r13`, tip `7e2dcd8f1`, tree `0e9273f8`);
  the r14 comparison build is `~/llama.cpp/build-rocm`.  Build with
  `ROCM_PATH=/opt/rocm-7.14.1-gfx120X cmake --build build --target llama-cli -j 16`.

## 10. Definition of done

* A minimal, deterministic reproducer at the widest ubatch that fails (ideally a `test-backend-ops`-style
  shape, not a 3800-token prompt).
* The point of corruption identified (which op/tensor/KV range goes wrong, and which `-sm tensor`
  difference causes it).
* A fix that keeps `-sm layer`, single-GPU and `-ncmoe 0` unchanged, gated on: `////`=0 at ~3600 and
  ~3800 and 16k tokens under `-sm tensor`; the `-ncmoe 0` 3-GPU Q4_K_M byte-identity; width purity
  (`none == n1 == n3 == n7`); and a long MTP acceptance run at `-n 3000` (>= 0.45), all at `-ub 8192`.
