# `-sm tensor` corruption on a wide prefill ubatch — handover (open)

**Status: OPEN, CRITICAL (silent wrong output). ROOT CAUSE LOCATED 2026-10-06 (r16 session) — the
block-06 H2D staging ring.**  TODO #41.  **Nothing here is in the delivery** (the `wip/` rule): the r15
release (`v16-a55e952b8-r15`) fixes the `-sm tensor` CPU fallback + the slow cache (TODO #40), *not*
this.

Read this first if you are picking up the `-sm tensor` prefill corruption. It records the exact repro,
the bisected threshold, the located root cause and its workaround, everything already ruled out (with
the commands, so you do not repeat them), the false lead that cost most of the r15 session, and the
hypotheses left.

---

## 0. ROOT CAUSE (2026-10-06): the block-06 H2D staging ring

The corruption is the **op-offload H2D staging ring**, and it is **not width-specific at all** — it is
simply corrupt **whenever it is active**.  The apparent "~3600-token threshold" is the point where the
**calibrated activation gate** flips staging on:

```
sched_stage_min_tokens: H2D staging calibration: 14.5 GB/s -> min_tokens=1542
```

`GGML_SCHED_STAGE_MIN_TOKENS` pins that gate, and pinning it settles the question completely:

| IQ3_XXS, `-sm tensor -ub 4096` | `////` | MTP acc | verdict |
|---|---:|---:|---|
| default (gate decides → staging **on**) | 1 | 1.00 | CORRUPT |
| `GGML_SCHED_STAGE_MIN_TOKENS=999999` (never stage) | **0** | **3.88** | **COHERENT** |
| `GGML_SCHED_STAGE_MIN_TOKENS=0` (always stage) | 1 | 1.00 | CORRUPT |

Same at `-ub 2048` (**forced** staging corrupts there too — so staging is broken at every width, the
default just leaves it off below the gate):

| IQ3_XXS, `-ub 2048` | staging forced on | staging forbidden |
|---|---|---|
| `-sm tensor` | `////`=1, acc 1.00 (corrupt) | `////`=0, acc 2.82 (**coherent**) |
| `-sm layer`  | **ABORTS** (exit 134) | `////`=0, acc 3.33 (coherent) |

**The `-sm layer` immunity is an artefact of the gate, not of the split.**  Under `-sm layer` the split
input is the whole layer table, so `sched_stage_min_tokens_for_bytes()` scales the threshold far above
the prompt width (the scaling is `base * bytes / 144 MiB`) and staging stays **off** — `-sm layer` is
coherent only because it never stages.  Under `-sm tensor` the meta split shards the table, `bytes` (the
largest host-weight input of the split) is smaller, the scaled threshold drops below the prompt width,
and staging turns **on** — hence occasional corruption only under `-sm tensor`.  Forcing staging on under
`-sm layer` is therefore the **control** that shows the ring is broken there too (it aborts instead of
silently corrupting, because the CUDA backend has `stage_input == nullptr` and takes the generic ring).

### The hardware/perf trade-off (why this matters)

Full model (UD-IQ4_XS, 2 GPU, `-ncmoe 48`, 16k prompt, `-n 1024`, coherent rows only):

| `-sm tensor` | prefill | decode | coherent | MTP acc |
|---|---:|---:|---|---:|
| `-ub 8192`, staging **on**  | **1557** | 29.4 | ❌ | 1.00 |
| `-ub 8192`, staging **off** | 888.6 | 49.5 | ✅ | 3.77 |
| `-ub 4096`, staging **on**  | 1058 | 29.4 | ❌ | 1.00 |
| `-ub 4096`, staging **off** | 663 | **65.2** | ✅ | 3.79 |

The staging ring is worth **+75 % prefill** (1557 vs 889) — which is exactly why the Reddit R9V
comparison was tempting — but every staged run is garbage.  Note the decode column is confounded here:
staging-on breaks MTP (acc 1.00), so its "decode" is really a 1-token-per-step rate, not a like-for-like
number against the staging-off rows.

### The failure signature under `-sm layer` (the abort gives the ring away)

`GGML_SCHED_STAGE_MIN_TOKENS=0`, `-sm layer`, `-ub 2048`, exit 134:

```
D sched_stage_issue: batch_tokens=2044 n_inputs=24 staged=1
E ROCm error: invalid argument
E   current device: 1, in function ggml_backend_cuda_buffer_set_tensor
E   hipMemcpyAsync((char *) tensor->data + offset, data, size, hipMemcpyHostToDevice, stream 2)
```

### Where to look in the code

* `ggml/src/ggml-backend.cpp` — `sched_stage_min_tokens()` (~2105, the link calibration),
  `sched_stage_min_tokens_for_bytes()` (~2180, the `base * bytes / 144 MiB` scaling that makes the two
  `-sm` modes differ), `sched_stage_issue()`/the staged-input consume loop (~2480) and the tripwire that
  aborts when a redirected input reaches a copy path.
* `ggml/src/ggml-backend-meta.cpp` — **`ggml_backend_meta_stage_input()` (~2182, the `-sm tensor`
  path)** and its consume side in `ggml_backend_meta_graph_compute()` (~2463), where each device's
  `simple_tensor->data` is repointed at `chunk.slot`; restored by `ggml_backend_meta_stage_guard`
  (~2376).  `struct stage_chunk` (~1917) carries `src/stride/n_copies` for the gather.  **This is the
  prime suspect**: the slot holds a *gathered/compacted* per-device slice, and the consuming kernels
  then index it through the simple tensor's own `nb` — any mismatch between the gather's pack layout
  and the tensor's strides is a silent corruption.  Also check the `stage_free_ev` timing in the guard
  destructor (it runs after the child graphs are *launched*, not completed).
* `ggml/src/ggml-cuda/ggml-cuda.cu` — `ggml_backend_cuda_stage_gather()` (~8092, note the
  `n_copies <= 4096` host-gather vs the scratch 2-D path switch), `h2d_pin_buffer`/`h2d_scratch`, and
  `ggml_backend_cuda_buffer_set_tensor` (line 836, the `-sm layer` abort).

### Workaround (works today, no code change)

`GGML_SCHED_STAGE_MIN_TOKENS=999999` (never stage) restores coherence **and** MTP at any `-ub` under
`-sm tensor`.  It costs the staging prefill win, so it is a stopgap, not the fix — but it is a clean
A/B and a safe thing to hand a user hitting the corruption.

### Is this a delivery regression?

Possibly pre-existing since block 06 (r12).  The r12/block-06 validation A/B
("Q4_K_M ub8192 staging 5283 vs gather 4035") was a **prefill-throughput** comparison; the staging-on
side of the trade-off appears never to have been **coherence-gated under `-sm tensor`**, which is how a
silent corruption could ride along a documented win.  Check `block/*` tags and the r12 records before
concluding; if it is pre-existing, the gate that shipped was throughput-only.

---

## 1. TL;DR

**Root cause found: the block-06 H2D staging ring is corrupt whenever it is active** (see §0).  The
original symptom — a single prefill ubatch **wider than ~3600 tokens** under `-sm tensor` makes the
**target model itself** produce garbage (`////`) — is real, but the ~3600 is not a width limit: it is
the point where the ring's calibrated activation gate turns staging **on**.  Staging off =
`GGML_SCHED_STAGE_MIN_TOKENS=999999` = coherent + working MTP at every `-ub`.

`-sm layer` looks immune only because the gate scales with the host table size, which is larger there,
so staging stays off.  Pre-existing (the r14 binary reproduces it), unrelated to the expert cache.  It
also destroys MTP acceptance — which is why it first looked like "MTP is broken".

It is a **silent corruption**: the model answers, it is just wrong.  Treat it as a correctness bug, not a
perf bug, and gate any fix on output coherence (`////`=0) with staging **on**.

## 2. Exact repro

Model: Qwen3.8-Flash-Next **UD-IQ4_XS** (`/llm/models/Qwen3.8/Flash-Next/IQ4_XS/`,
`...-UD-IQ4_XS-00001-of-00003.gguf`) + its MTP head
(`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`). 2 GPU, cache auto.

```bash
# a ~3800-token prompt (build it by repeating prompts/code-python.txt; ~3.8 chars/token).  NOTE: the
# base file is 2025 chars (~533 tokens), so it is `base*7`, NOT a large multiplier -- `base*266` is
# ~141 000 tokens and fails with "exceeds the available context size".
python3 - <<'PY'
base = open('/home/stew675/llama-cpp-rdna-boosts/prompts/code-python.txt').read()
open('/tmp/pl_3800.txt','w').write(base*7 + "\n\nWrite a short summary.")   # ~3731 tokens
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

**IQ3_XXS reproduces it identically** (2 GPU, 79 GB, same flags) — so it is **quant- and
model-independent**:

| IQ3_XXS | `-sm tensor` | `-sm layer` |
|---|---|---|
| `-ub 4096` | `////`=8, acc 1.00 ❌ | `////`=0, acc 3.75 ✅ |
| `-ub 2048` | `////`=0, acc 2.82 ✅ | `////`=0, acc 3.33 ✅ |

IQ3_XXS is also the **faster cycle** for this investigation (79 GB vs 96 GB, and it has its own
`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` head in its directory).

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

## 4. `-sm layer` is NOT affected — but only because it never stages

Same prompt, same run flags, only `-sm tensor` → `-sm layer`:

| config | `////` |
|---|---:|
| `-sm tensor -ub 4096` | **8** |
| `-sm layer  -ub 4096` | **0** |
| `-sm layer  -ub 2048` | 0 |
| `-sm tensor -ub 2048` | 0 |

This was originally read as "the bug lives in the Meta-split prefill path".  It is actually an artefact of
the **activation gate** (§0): the meta split shards the table, so the size-scaled threshold drops below
the prompt width and staging turns on; `-sm layer` sees the whole table, its threshold stays high, and it
never stages.  **Forcing** staging under `-sm layer` (`GGML_SCHED_STAGE_MIN_TOKENS=0`) makes it fail too —
with an abort rather than silence.  So `-sm layer` is a control for *the gate*, not for the split; the
real control is `GGML_SCHED_STAGE_MIN_TOKENS` at a fixed `-sm`.

## 5. Ruled out (do not re-test these)

**The cause is found — see §0.  The controls below establish what it is *not*, which is what narrowed it
to the staging ring; keep them as regression checks, but do not re-run them hoping to find the bug.**

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

Hypothesis 1 below was **the cause** and is confirmed (§0).  What remains is *why* the ring's data is
wrong, which is now a bounded code question rather than a hunt:

1. ~~**The wide-ubatch staging path.**~~  **CONFIRMED** — the block-06 op-offload H2D staging ring is
   corrupt whenever active (`GGML_SCHED_STAGE_MIN_TOKENS` A/B, §0).  It is **not** width-specific: the
   ~3600 tokens was only the calibrated activation gate.
2. **The gather's pack layout vs the consuming tensor's strides (prime suspect).**
   `ggml_backend_meta_stage_input()` gathers each device's slice to `chunk.slot` with
   `src/stride/n_copies` (`ggml_backend_cuda_stage_gather`), then `graph_compute` repoints
   `simple_tensor->data = chunk.slot`.  If the gather's compacted layout does not match the strides the
   kernels index the simple tensor with, every staged weight is misread — silently, and only while
   staging is on.  Compare `chunk.stride`/`chunk.size` against the simple tensor's `nb` for one MoE
   op and dump the slot's first/last bytes against the host table.
3. **`input` vs `input_cpy` geometry.**  The ring derives `size`/`chunk_size_full`/`n_chunks` from
   **`input_cpy`** but gathers from **`input->data`**.  Any shape/stride difference between them is a
   wrong gather geometry.  Cheap to check with an assert/log.
4. **`stage_free_ev` timing.**  `ggml_backend_meta_stage_guard`'s destructor records the free event
   after the child graphs are *enqueued* (not completed) and cycles only `stage_n_slots` (default 8)
   slots, so a slot can be re-gathered while its previous consumer is still reading it.  Verify the
   event is recorded on the stream the consumers actually run on, after their completion.
5. **`h2d_pin_buffer`/`h2d_scratch` sizing.**  The ring slot is reserved with `copy_bytes`
   (`n_chunks * chunk_size_j`) and the gather re-requests `h2d_stage_buffer(slot, total)`; a slot/pin
   sized for a *different* tensor (the meta context is shared across all staged uploads) would overrun.
   Check the resize/reallocation behaviour and whether two tensors can share a slot index.
6. **Numerics, not logic.**  Still worth one cheap check: compare wide-ubatch logits with staging on vs
   off (same seed, greedy) to see whether the corruption is a truncated/NaN region (logic) or a drift
   (numerics).

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

* The exact defect in the staging path identified (which of §7.2–§7.5), with the byte-level comparison
  that proves it (host table vs ring slot for one staged MoE op).
* A fix that keeps `-sm layer`, single-GPU and `-ncmoe 0` unchanged, gated on **staging left ON** at:
  `////`=0 and coherent text at 3800 and 16k tokens under `-sm tensor`, `-ub 4096` **and** `-ub 8192`;
  the `-ncmoe 0` 3-GPU Q4_K_M byte-identity; width purity (`none == n1 == n3 == n7`); and a long MTP
  acceptance run at `-n 3000` (>= 0.45).
* A decision recorded on the shipped gate: if the ring cannot be made correct quickly, the interim
  delivery should **raise/disable the `-sm tensor` staging gate** rather than ship a silent corruption —
  the current default mixes correct and corrupt runs by prompt width, which is the worst outcome.
* Regression: a gate that **coherence-checks the staging-on side** (the r12/block-06 win was validated on
  throughput only, §0 "Is this a delivery regression?").
