# `-sm tensor` corruption on a wide prefill ubatch — ROOT-CAUSED, FIX VALIDATED (not shipped)

**Status: ROOT-CAUSED AND FIXED 2026-10-06 — the fix is validated but NOT in the delivery** (the
`wip/` rule): `wip/moe-cache-autosize/stage-redirect-fix.patch` (2 files: `ggml-backend-meta.cpp` +25/-1, `ggml-backend.cpp` +8/-1).  TODO #41.  The r15
release (`v16-a55e952b8-r15`) fixes the `-sm tensor` CPU fallback + the slow cache (TODO #40), *not*
this.

**One line:** the meta staging consume repointed the device tensor at the ring slot and the stage guard
then restored the pointer immediately after enqueuing the child graphs — but the kernels read
`tensor->data` at *execution* time, so they saw the restored (stale) pointer and never read the staged
bytes.  Copying the slot into the real buffer instead fixes it, is equally fast, and keeps the staging
win (**+12 % prefill**: 999 vs 890 t/s at 16k / `-ub 8192`).

Read §0 first.  Everything else is the trail: the exact repro, the bisected threshold, the gate, the
ruled-out list (with commands), the false lead that cost the r15 session, the method trap that cost this
one, and what is still open (3 GPUs; the generic ring's own redirect).

---

## 0. RESOLVED: the staged data was fine; the *restore* clobbered the pointer before the kernels read it

**Root cause (2026-10-06, final):** the meta staging consume repoints the device tensor at the ring slot
(`simple_tensor->data = chunk.slot`), and `ggml_backend_meta_stage_guard` then restores the pointer
**immediately after the child graphs are enqueued**.  The kernels read `tensor->data` when they *execute*,
not when they are queued, so they see the restored (stale) pointer -- the staged bytes are never read and
the layer is fed whatever the graph-allocated `input_cpy` buffer still held.  Fix:
`wip/moe-cache-autosize/stage-redirect-fix.patch` (2 files: `ggml-backend-meta.cpp` +25/-1, `ggml-backend.cpp` +8/-1).

### The measurement that settled it

* `GGML_SCHED_STAGE_MIN_TOKENS=999999` (never stage) -> coherent, MTP acc 3.88.
* Instrumented the redirect (`data_now=<slot>`) against the address `MUL_MAT_ID` actually dispatches:
  **intersection = 0** -- 24 distinct slot pointers were handed to tensors, 360 distinct `src0->data`
  values reached the kernel, and they never overlap.  The op was reading the stale buffer.
* A/B on the guard alone (same binary, same redirect):

  | IQ3_XXS, `-sm tensor -ub 4096`, staging ON | `////` | acc |
  |---|---:|---:|
  | redirect + restore (upstream) | 1 | 1.00 |
  | **redirect, NO restore** | **0** | **3.88** |
  | `stage_d2d` into the real buffer (the fix) | 0 | 3.88 |

So the redirect was *fundamentally right*; only the restore timing was wrong.

### And the redirect does not buy any speed

| 16k prompt, `-ub 8192`, UD-IQ4_XS, `-n 1024` | prefill | decode | coherent | MTP acc |
|---|---:|---:|---|---:|
| staging OFF | 890 | 49.4 | yes | 3.77 |
| **staging + D2D (the shipped fix)** | **999** | 44.0 | yes | 3.77 |
| staging + redirect, no restore | 1002 | 44.0 | yes | 3.77 |
| staging + redirect upstream (restore) | 1572 | 29.2 | **no** | 1.00 |

The forbidden `stage_d2d` copy costs nothing measurable, so **the copy is the right fix** -- safer than a
deferred restore, and equally fast.  The corrupt 1572 t/s was reading stale memory and not doing the work,
the same artifact the block-06 record already documents for the device gather.

**Corollary:** with correct staging the coherent prefill wall at 16k / `-ub 8192` is **~1000 t/s**, not
1500.  The honest staging win over not staging is **+12 %** (999 vs 890).  At ~6.8 GB/s that is roughly
half of the calibrated 14.5 GB/s H2D link, so the wide-ubatch bottleneck is *not* the link -- see TODO #42.

### Consequences for the rest of the staging design

* **Eradicated (2026-10-06): the generic ring's redirect was the same defect, and it is now default off.**
  `sched->stage_mode` had defaulted to **1 (redirect)** with `sched_stage_restore` running at the top of
  the next `sched_stage_issue` -- i.e. on the same "the kernels have been launched" assumption that the
  meta path just disproved.  The patch now defaults `stage_mode` to **0** (stage-then-D2D); the redirect
  stays reachable only via `GGML_SCHED_STAGE_MODE=1` for A/B.  This was an *unvalidated default*, and it is
  very likely why forcing staging under `-sm layer` has never worked end to end.
* **Still open in the ring (unsupported config only):** forcing staging under `-sm layer`
  (`GGML_SCHED_STAGE_MIN_TOKENS=0`, i.e. below the designed 64-token floor) gives **mode 1 = abort**
  (`hipMemcpyAsync ... invalid argument` in `ggml_backend_cuda_buffer_set_tensor`) and **mode 0 = corrupt**
  (`////`) on a 3.7k-token prefill.  Not reachable by default -- the calibrated gate keeps staging off for
  `-sm layer` -- but it means `-sm layer` staging is not a working feature, only an untested path.
* **#43 is not the device count.**  A dense model (Qwen3.5-4B Q8_0) under `-sm tensor` is coherent on both
  2 and 3 devices (prefill 3874 / 3328 t/s), so the 3-device meta split is sound.  The 3-GPU corruption
  needs the **MoE / host-expert** path (`-ncmoe > 0`) and is independent of staging -- the remaining
  default-reachable corruption, and the next thing to fix.
* Any future redirect-based staging optimisation must defer the restore to the **free-event boundary**
  (the slots already carry `stage_free_ev` for exactly this ordering), never to the end of `graph_compute`.
* The MoE expert cache already carries a workaround for this class of pointer churn ("a meta graph
  rebuild can hand the op a different simple-tensor pointer than the one the hook saw", the
  `(layer, role, device)` semantic-key fallback in `moe_cache_get_table`) -- a useful precedent.


* **`sched->stage_mode` defaults to 1 (redirect) and has the same defect** in the generic ring
  (`sched_stage_restore` runs at the top of the next `sched_stage_issue`, i.e. still before the kernels
  execute).  That is very likely why forcing staging under `-sm layer` has never worked end-to-end.  It
  should default to `stage_mode = 0` (stage-then-D2D) or implement a deferred restore.
* Any future redirect-based staging optimisation must defer the restore to the **free-event boundary**
  (the slots already carry `stage_free_ev` for exactly this ordering), never to the end of `graph_compute`.
* The MoE expert cache already carries a workaround for this class of pointer churn ("a meta graph
  rebuild can hand the op a different simple-tensor pointer than the one the hook saw", the
  `(layer, role, device)` semantic-key fallback in `moe_cache_get_table`) -- a useful precedent.

### Method notes (both cost real time)

* `GGML_LOG_WARN` from ggml during graph compute is **filtered** by `llama-cli`'s default log level, so an
  instrumented build can look like "the hook is never called".  Use `fprintf(stderr, ...)` here.
* Do not trust `refs_in_child_graph`-style inference: the child-graph node arrays *did* reference the
  redirected object, and the op *still* never saw the slot.  Dump the address at both ends and intersect.

## 1. TL;DR

**The block-06 H2D staging ring corrupted every layer it staged, and the defect was the pointer, not the
data** (see §0).  The original symptom — a single prefill ubatch **wider than ~3600 tokens** under
`-sm tensor` makes the **target model itself** produce garbage (`////`) — is real, but ~3600 is not a
width limit: it is where the calibrated activation gate turns staging **on**.

* staging off (`GGML_SCHED_STAGE_MIN_TOKENS=999999`) -> coherent + working MTP at every `-ub`;
* **staging on with the fix** (`stage_d2d` instead of the redirect) -> coherent + working MTP, and faster
  than staging off (IQ4_XS 16k: prefill **1000 vs 875**, decode 43.6 vs 48.0);
* staging on upstream (redirect) -> `////`, MTP acc 1.00.

`-sm layer` looked immune only because the gate scales with the host table size, which is larger there,
so staging stays off; forcing it on aborts (a separate, unexplained `set_tensor` failure).  Pre-existing
(the r14 binary reproduces it), unrelated to the expert cache.

It is a **silent corruption**: the model answers, it is just wrong.  Gate any change here on output
coherence (`////`=0) **with staging on**, never on throughput.

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

---

## 11. TODO #43 — `-sm tensor` + 3 GPUs + host experts silently corrupts (IQ4_XS only)

**Open.  Deterministic, not allocator luck.**  Separate from the staging redirect above (still reproduces
with `GGML_SCHED_STAGE=0` and with `-ncmoe` small).

### Repro and the contrast matrix (2 x/3 x R9700, greedy `--seed 42`, 3.7k prompt, `////` = corrupt)

| model | `-sm` | GPUs | staging | result |
|---|---|---|---|---|
| UD-IQ4_XS | tensor | 3 | off (`MIN_TOKENS=999999`) | `////` at `-ub` **512, 2048 and 4096**, acc 1.00 |
| UD-IQ4_XS | tensor | 3 | on | `////` (996 t/s) |
| UD-IQ4_XS | **layer** | 3 | off | **coherent**, acc 3.54 |
| UD-IQ4_XS | tensor | 2 | off | **coherent**, acc 3.54 |
| UD-IQ3_XXS | tensor | 3 | off | **coherent** at 512 / 2048 / 4096 (acc 3.36-3.62) |
| Qwen3.5-4B Q8_0 (dense) | tensor | 3 | n/a | **coherent**, prefill 3328 t/s |

Invariant to `-c` (8192 / 16384 / 32768) and to `-ncmoe` (8 / 32 / 48), with
`MOE_EXPERT_CACHE_MIB=0` and `=4096`, and with `GGML_SCHED_STAGE=0`.

### What that rules out

* **Not the device count**: a dense model is coherent on 3 devices, so the 3-way meta split and its
  2-step all-reduce are sound.
* **Not the expert cache**: identical with the cache off, at a fixed 4096 MiB, and at auto.
* **Not the staging ring**: identical with the whole staging subsystem off.
* **Not the ubatch width**: broken at `-ub 512` too (so not the calibrated gate and not a wide-batch
  kernel).
* **Not `-sm tensor` alone**: 2 devices are coherent, and IQ3_XXS is coherent on 3.
* **Not allocator layout luck**: invariant across four different context sizes and three `-ncmoe` values.
* **Not the all-reduce path**: identical with the internal AR and with `GGML_CUDA_ALLREDUCE=nccl` (which
  uses a completely different collective), so the 2-step/delayed-AR MoE branch is not implicated.
* **Not the quant's *weights* alone**: IQ3_XXS on the same 3 GPUs is coherent, so the split geometry
  itself is representable -- it is the interaction of the *wider* IQ4_XS over-read with the 3-way slice.

**Next single measurement to make (do this before writing any fix):** add the `[INT] mmid ENTER` log back
to `ggml_cuda_mul_mat_id` (it prints `dst->ne[2]`, `src0->name`, `src0->data`, `src0->buffer`) and diff
the 3-GPU IQ4_XS prefill against the *2-GPU coherent* run: compare `src0->ne[]`/`nb[]` per device and
which kernel family is dispatched.  That is what distinguishes "the slice geometry is wrong" from "the
tail guard is missing" -- and a speculative pad written past a tensor allocation risks clobbering a
neighbour in the gallocr-packed buffer, so it must not be the first move.

### Leading hypothesis, and the evidence for it

The symptom is the documented **MMQ-NaN `////` signature**, and the repo's own gate script names the
mechanism precisely (`scripts/gate-qwen4exp-quant-coherence.sh`):

> the quantized `MUL_MAT_ID` MMQ's speculative read past a routed expert can land in stale (NaN) bytes
> and poison the tile -> the model emits a repeated `/`.  … The guard is a one-time zero of each expert
> slot's head; the host-resident upload path (`copy_experts`) guards with `min(expert_size, 512)`, but
> the gather hard-coded 64 bytes.  **64 is enough for IQ4_NL only - IQ4_XS (and any quant whose MMQ
> over-read is wider) still corrupts.**

That is exactly our quant split: **IQ4_XS needs the 512-byte guard, IQ3_XXS survives the narrower
over-read.**  And the whole-table paths do **not** write that guard: `copy_experts` pads each pruned run
except the final one (`padding_end = last_id < n_expert - 1 ? padding : 0`), while the full-tensor copy
(the `else` branch in `sched_compute_splits`) and the staged `stage_d2d` upload nothing past the table at
all.  The tail past a table's last expert is therefore whatever the destination buffer held.

**To confirm/refute:** zero `min(expert_size, 512)` bytes at `dst + ggml_nbytes(table)` in every
whole-table path (full-tensor copy, staged `stage_d2d`) and re-run the 3-GPU IQ4_XS repro.  The iface
`memset_tensor(buffer, tensor, value, offset, size)` must be called **directly** (`ggml_backend_tensor_memset`
asserts `offset + size <= ggml_nbytes(tensor)`).  Under `-sm tensor` the pad must be applied per **simple**
device buffer, i.e. through the meta backend, not once on the meta tensor.  If the pad makes it coherent,
the same pad belongs on the generic ring's `stage_d2d` -- and the pruned path's final run should get it
too, which is a likely latent bug of the same family in `copy_experts`.

**Alternative if the pad does not fix it:** instrument `ggml_cuda_mul_mat_id`'s `src0->data`,
`src0->ne`/`nb` and the dispatched kernel family for the 3-device IQ4_XS case and compare against the
2-device (coherent) case -- a structural difference in the per-device expert slice is then the target.

---

## 12. TODO #43, second session (2026-10-06): the guard hypothesis is DISPROVEN; the 3-way slice geometry is PROVEN correct

**Status: open.  The §11 leading hypothesis (missing `min(expert_size,512)` guard on the whole-table
paths) is wrong, and so is "the 3-way slice geometry is wrong".  Nothing is promoted.**  The build used
is `~/llama-r13` @ `4643be072` (r15 tip + the #41 stage-redirect fix) plus the throwaway diagnostics
saved here as `mmid-geometry-diag.diff` (an `[INT]` dump in `ggml_cuda_mul_mat_id`, an `[TAIL]` dump +
a `GGML_MMID_TAIL_GUARD` 512 B `get_alloc_size` tail, and `GGML_MMID_NO_MMQ`).  Every result below is
the 3731-token prompt (`/tmp/pl_3800.txt` = `code-python.txt` x7), `-sm tensor -ncmoe 48 -ub 512
-b 512 -c 8192 -t 8`, cache off (`MOE_EXPERT_CACHE_MIB=0`), staging off (`GGML_SCHED_STAGE=0`),
`--seed 42 --temp 0`; corrupt = `grep -c '////'` > 0.

### 12.1 The slice geometry is correct (measurement trumps the hypothesis)

`GGML_STAGE_INTROSPECT=1` on the 2-GPU and 3-GPU IQ4_XS runs dumps every `src0` the MoE consumer
receives.  Per device:

| tensor (blk.2 small case) | 2 GPU (`ne[1]`/`ne[0]`) | 3 GPU |
|---|---|---|
| `ffn_gate_exps` (iq4_xs, split axis 1) | 256 / 384 | 256 / 256 / 128 |
| `ffn_up_exps`   (iq4_xs, split axis 1) | 256 / 384 | 256 / 256 / 128 |
| `ffn_down_exps` (q8_0,  split axis 0) | 256 / 384 | 256 / 256 / 128 |

* the slices **sum to the full dimension** (gate/up `n_ff = 640`; down `n_embd = 2560` via `nb[1]`),
  every slice is a positive multiple of 128 (128/256/384), and every `nb[]` is exactly
  `simple_nb[1] * simple_ne[split]` (e.g. iq4_xs `nb[1]=1360`, `nb[2]=174080` for a 128-row slice).
* **`ne[2] = 512` (the expert axis) is never split** -- the split is on `n_ff`/`n_embd`, so every device
  holds all 512 experts for its row range.  No device sees a zero or block-misaligned expert slice.
* the split is a deliberate load-balancing partition that rotates the extra 128-row block between
  devices per layer (blk.0 = 128/256/256, blk.1 = 256/128/256, blk.10 = 256/256/128, ...), so it is not
  a uniform `-ts` fraction.
* **the 3-GPU IQ3_XXS run has byte-identical geometry** for blk.2 (256/256/128, all `iq3_s`/`iq4_nl`)
  and is coherent.  So the *only* variable between the coherent and corrupt runs is the **quant of the
  expert tensors**, not the split geometry.  (In the UD-IQ4_XS file the only `iq4_xs` expert tensors
  are `blk.2.ffn_gate_exps` and `blk.2.ffn_up_exps`; all other expert tensors are iq3_s / iq4_nl /
  q8_0.  The file name overstates the quant coverage.)

### 12.2 The guard is NOT the fix (tested directly, inside the allocation)

`ggml_cuda_mul_mat_q` already zeroes compute-buffer slack:
`if (usage == COMPUTE) { size_alloc = get_alloc_size(...); if (size_alloc > size_data) memset(data +
size_data, 0, size_alloc - size_data); }`.  For these expert slices `ne0 = 2560` is
`MATRIX_ROW_PADDING`-aligned, so the CUDA `get_alloc_size` returned exactly `ggml_nbytes` and the clear
never fired.  `GGML_MMID_TAIL_GUARD=1` made `ggml_backend_cuda_buffer_type_get_alloc_size` return
`ggml_nbytes + 512` for quantized `ne[2] > 1` tensors, and the `[TAIL]` log confirms the clear then ran
on **every** device slice, e.g. `blk.2.ffn_up_exps dev=2 nbytes=89128960 alloc=89129472` (512 B zeroed
inside the tensor's own allocation).  **The 3-GPU prefill still emitted `////////`.**  So the standard
"MMQ over-read past the last routed expert / slice" family is *not* this bug.

### 12.3 It is not the MMQ kernel

`GGML_CUDA_FORCE_CUBLAS=1` disables MMQ for the routed matmul and takes the exact dequant + hipBLAS
path (the `ggml_cuda_mul_mat_id_needs_sync` assert does **not** fire for this config, unlike the
`GGML_MMID_NO_MMQ` diagnostic, which aborts precisely because it skips the MMQ branch without making
`needs_sync` true).  **It still emitted `////////`** (Generation 6.8 t/s).  A kernel-family bug in the
MMQ tile math is therefore ruled out; the corrupt bytes are in the *inputs* / the *state* the exact
path reads too.

### 12.4 Also ruled out this session (do not repeat)

| switch | 3-GPU IQ4_XS result |
|---|---|
| staging forced (`GGML_SCHED_STAGE=1 GGML_SCHED_STAGE_MIN_TOKENS=64`) | `////////` |
| MTP removed (no `-md`, no `--spec-type`) | `////////` (so not the MTP/verify path) |
| tail guard active (12.2) | `////////` |
| exact math (`GGML_CUDA_FORCE_CUBLAS=1`, 12.3) | `////////` |
| `LLAMA_MMAP_HOST_EXPERTS=0` | `////////` |
| `GGML_SCHED_EVENTS=0` | `////////` |

### 12.5 What is left (the next measurement to make)

The geometry is right, the guard is not it, and the exact math path corrupts too -- so the bug is in the
**host-resident expert data path for `iq4_xs` specifically**: the host master → per-device `input_cpy`
copy, or the dequant/interpretation of those bytes.  The cheapest decisive probe is a byte compare, not
another kernel hunt:

1. For `blk.2.ffn_up_exps` on the 3-GPU run, dump the host master's first expert bytes and the
   corresponding bytes after the per-device copy (host readback immediately after the graph), and diff.
   `moe-expert-cache.cu` already has `moe_cache_read_check` for a comparable check under the cache.
   If the copy matches, diff the *routed-expert* bytes the kernel/hipBLAS actually consume.
2. Compare the **host master's `nb[]` / repack layout** against the device `input_cpy` layout for an
   `iq4_xs` table vs an `iq3_s` table -- `copy_experts` derives `expert_size = ggml_nbytes(input)/n_expert`
   and the meta `set_tensor_async` splices with `chunk_size_full = input_cpy->nb[2]`; any host-side
   repack row padding difference is a silent geometry mismatch that the *device* log cannot see.
3. If those are clean, bisect by `-ncmoe`: find whether a single host layer corrupts, and whether it is
   always the `blk.2` table (the only `iq4_xs` one) or any layer once `blk.2` is on the host.

---

## 13. Reader-side instrumentation (2026-10-06, third session): the guard is a *device-0 prefix*, and that is the systemic wrong assumption

**Status: open.  Instrumentation committed (scratch patch `mmid-reader-instrumentation.diff`, applied to
`~/llama-r13`).  Nothing promoted.**  The reader now reports its own bounds and the first NaN.

### 13.1 What the reader knows (and proves)

`ggml_cuda_mmq_load_tiles_iq4_xs` / `_q8_0` receive `kbx0`, `i_max`, and `stride`, so they know exactly
which rows/blocks exist.  Added detectors (`GGML_OVR*` device `printf`) show:

* **iq4_xs row clamp is exact.**  `[OVRFIRST] iq4_xs i=3 i_max=127 kbx0=24320 stride=10 I=64 J=16
  fallback=0`; the `[OVR] i > i_max` alarm never fires.  The weight rows the reader reads all exist.
* **Host pruning is complete.**  `[USEDCHK] blk.2.ffn_up_exps n_expert=512 used_distinct=356..415`; the
  kernel only reads id-referenced experts, and all of those are copied (`[COPYCHK] ... bad=0`).
* **The first NaN is `blk.2.ffn_down` (q8_0), with a clean activation.**  Ordered `[NANCHK]` for the
  3-GPU run at the 512 B guard: `src1 ffn_gate nan=0`, `dst ffn_gate nan=0`, `src1 ffn_up nan=0`,
  `dst ffn_up nan=0`, `src1 ffn_down nan=0`, **`dst ffn_down nan=84`**, then everything downstream.
  So the corrupt quant is not the `iq4_xs` gate/up at all -- blk.2's *down* is q8_0.
* **The q8_0 reader's structural reach is 4 blocks.**  `[OVRQ8] I=128 J=128 fallback=1 ... stride=320
  kbx=0 koff=0 second_block=4`: the loader unconditionally reads `bxi[MMQ_TILE_NE_K/QI8_0] = bxi[4]`
  (136 B).  For an expert slice whose K extent is short (the down split gives `ne0 = n_ff` = 128/256 ->
  `stride` = 4/8 q8_0 blocks), the last row's `bxi[4]` runs past the expert.

### 13.2 The systemic wrong assumption

Every guard in this repo is **a byte prefix copied once**, and under `-sm tensor` the scheduler
distributes that prefix through `ggml_backend_meta_buffer_set_tensor_async`, which splits along the
tensor-split axis.  Look at the tail branch:

```cpp
for (size_t j = 0; j < n_backends; j++) {
    ...
    if (rem != 0 && offset_j < rem) {           // <-- only while the running offset is < rem
        const size_t tail_len = std::min(rem - offset_j, chunk_size_j);
        ... copy `tail_len` guard bytes ...
    }
    offset_j += chunk_size_j;                   // dev 0 owns the whole guard; dev 1+ never see it
}
```

`chunk_size_dev0` (the first device's share of every expert) is ~175 KiB-348 KiB, while `rem` is 512 B.
So `offset_j >= rem` for `j >= 1` and **only device 0 receives the guard bytes**.  Devices 1 and 2's
"next expert" region is written by neither the aligned copy (that only covers *routed* experts) nor the
tail (only dev 0), so it keeps whatever the reused graph buffer held.  Whether that is finite is
allocation state -- which is exactly why every higher-level "fix" has produced a new corner case.

This also explains the device-count split cleanly: 2-GPU vs 3-GPU change which device is "last" and
what the reused buffer holds, and it explains why the guard had to be enlarged *and* still felt fragile.

### 13.3 Next measurement

Add `ctx.device` to the `[NANCHK] dst blk.2.ffn_down` line.  If the NaN is on device 1/2 (not 0), this
section is confirmed and the fix is: **make every device's speculative-tail bytes finite**, either by
distributing the guard across devices (each device needs its own `min(expert_size, reach)` prefix) or,
more robustly, by one-time zero-filling each device's `input_cpy` simple-tensor region (the same
`(buffer, geometry)`-keyed trick the cache gather already uses for its arena).  The reader-side clamp
(§13.1) is the complementary half: the reader can zero the SRAM lane it speculatively loaded, so no
NaN propagates even if a byte was never written.

### 13.4 CONFIRMED: the NaN is on device 2, and the reader-side clamp fixes it

`[NANCHK] dst dev=2 blk.2.ffn_down_exps nan=84` while `dev=0` and `dev=1` are clean.  For the down
(split on `n_ff`) the guard chunks are `dev0=272 B, dev1=272 B, dev2=136 B`; the prefix is distributed
`dev0 <- 272`, `dev1 <- 240`, `dev2 <- 0` at `rem=512`, and `dev2` only starts receiving guard bytes once
`rem > 544` -- which is why the empirical threshold was ~640/672 and why it looked like a magic number.
**The guard is a device-0 prefix; every device owns its own speculative tail.**

Reader-side fix, tested: in `ggml_cuda_mmq_load_tiles_q8_0`, compute `koff = kbx0 % stride` and store 0
for any block with `koff + kbx (+ MMQ_TILE_NE_K/QI8_0) >= stride`, and likewise zero the `x_df` scale for
out-of-range blocks.  Candidate patch: `mmid-reader-clamp.diff`.

| 3-GPU, `-ub 512`, 512 B guard | coherent | prefill |
|---|---:|---:|
| clamp off | **no** (`////`) | 372 t/s |
| clamp on  | **yes** | 151 t/s |

The clamp is correct but costs ~2.5x (the predicated/ternary store on the hot path).  Two zero-runtime-cost
alternatives, both systemic (fix the *shape* of the guard, not the byte count):

* **Per-device guard**: `ggml_backend_meta_buffer_set_tensor_async` should give *each* device its own
  `min(reach, chunk)` prefix of that device's slice of the next expert, instead of one contiguous prefix.
* **One-time finite-fill**: zero each `input_cpy` simple-tensor region once per `(buffer, geometry)` (the
  same keyed trick `moe_cache_gather_host` uses for its arena).  Then *any* speculative read, of any
  size, is finite by construction and no reach needs to be known at all.

The real invariant that was missing: **under tensor split, "the bytes past an expert are finite" must
hold per device, not once on the meta tensor.**
