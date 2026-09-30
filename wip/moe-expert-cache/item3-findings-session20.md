# Item 3 — findings (session 20, 2026-10-01)

Session 20 measured item 3 ("the user graph-input copies", README §0.3) with new instrumentation at tip
`f4b255041`.  **The brief's premise is wrong**: the cost is *not* the `GGML_TENSOR_FLAG_INPUT` branch.

## What was measured

Added `GGML_SCHED_SYNCDBG` counters for the input loop (`ITER` = whole loop incl. `continue` paths), the
USER (INPUT-flag) branch, the OTHER branch, the `stage_input` call and the stage drain — a temporary,
**uncommitted** patch to `ggml/src/ggml-backend.cpp` (26 lines; `git diff` in `~/llama-decode`).
Each run is `llama-bench -p 8192 -n 0 -ub 8192 -r 1` on 3× R9700 (warmup + 1 timed pass → halve the
totals for a per-pass figure).

### Q4_K_M 35B-A3B, `-ngl 99 -ncmoe 99 -sm tensor -ub 8192` (`MIB=16384`)

```
pp8192 1819.94 t/s                                   (pass = 4.50 s)
input_loop=976 8937.3ms                              (4.46 s/pass)
  USER branch:  56 copies   422.6ms  (7.55 ms/call)  <- the brief blamed THIS
  OTHER:        24 copies    58.4ms  (2.43 ms/call)
  STAGE_INPUT 960 8456.1ms   DRAIN 0 0.0ms           <- the real cost
SCHEDSYNC calls=456 total=986.4ms
```

`STAGE_INPUT` = `ggml_backend_meta_stage_input()`, called from the input loop for every host-weight
split input.  It is 95 % of the input loop; the USER branch is 5 %.

### Why `stage_input` blocks the host

`ggml_backend_meta_stage_input` (`ggml-backend-meta.cpp` ~2375) picks a per-device chunk path; for a
`-sm tensor` partial split it calls `stage_gather` →
`ggml_backend_cuda_stage_gather` (`ggml-cuda.cu` ~7959).  The **auto** mode (`GGML_META_GATHER_MODE` unset,
`n_copies <= 4096`) does the "host gather": a strided `memcpy` into a pinned slot, then one pinned H2D:

```
GATHERDBG calls=1920 wait=113.9ms host=7861.2ms issue=13.2ms   (Q4_K_M, auto)
GATHERDBG calls=3384 wait=2.2ms   host=24043.7ms issue=21.3ms  (IQ4,   auto)
```

So the **host `memcpy`** (7.9 s / 24.0 s) is the cost — not an event wait, not the H2D.

| Q4_K_M config | pp8192 | `STAGE_INPUT` | GATHERDBG host |
|---|---|---|---|
| auto (host gather) | 1822 | 8416 ms | 7861 ms |
| `GGML_META_GATHER_MODE=1` (force host gather) | 1354 | 12241 ms | 11584 ms |
| `GGML_META_GATHER_MODE=2` (scratch + device D2D) | 1844 | **1183 ms** | 0 (scratch path) |
| `GGML_SCHED_STAGE=0` (staging off) | 1478 | 0 | — |

Auto == cache-off == `DEVMAP=0` (both ~8.9 s), so the cost is **cache-independent**.

### IQ4 Flash-Next, `-ngl 99 -ncmoe 48 -sm tensor -ub 8192` (`MIB=16384`)

| config | pp8192 | `STAGE_INPUT` | notes |
|---|---|---|---|
| auto | 550.73 | 24887.5 ms | host=24043.7 ms |
| `GGML_META_GATHER_MODE=2` | **1881.47** | 101.1 ms | **`h2d_bytes=0`** — see below |
| `GGML_SCHED_STAGE=0` | 819.89 | 0 | `set_async=130` (B2 gather) |
| `GGML_SCHED_STAGE=0` + `GGML_SCHED_DEVGATHER=0` | 659.36 | 0 | `set_async=426386`, `get_async=788` (host path) |

## The two candidate conclusions (and the trap)

1. **Legitimate:** `GGML_META_GATHER_MODE=2` (device D2D compaction instead of the host gather) is
   **byte-identical and correct on Q4_K_M** (long-prompt `-ub 8192` greedy text identical, only the
   timing line differs) and drops `stage_input` 8.4 s → 1.2 s for **+1.2 %** (the host work is overlapped;
   only the last slice is on the critical path).  Worth landing as a default if the +1.2 % holds
   order-balanced.

2. **Trap — do NOT trust the IQ4 mode-2 number.**  On IQ4 `h2d_scratch(whole)` returns `nullptr` for
   every call, so `stage_gather` hits `if (scratch == nullptr) return false;` **before**
   `cudaEventRecord(done_ev)`.  `ggml_backend_meta_stage_input` **ignores `stage_gather`'s return value**,
   so it reports success while the done-event was never recorded and the slot was never filled — the
   consumer then reads an uninitialised ring slot.  The +241 % is therefore most likely **computing on
   garbage**, not a win.  (The two "identical output" checks so far did **not** exercise the staged path:
   the `-p "The capital of France is"` run is 5 tokens and the `-f`/`-ub 512` long-prompt runs sit below
   `sched_stage_min_tokens` ≈ 1536.  A staged-path byte-identity needs `-ub 8192` **and** a ≥2048-token
   prompt **and** a context/batch that fits — `-c 12288 -b 8192` OOMs on 3× 32 GB with `MIB=16384`.)

**Latent bug to fix regardless:** `ggml_backend_cuda_stage_gather` must record `done_ev` (or the caller
must check the return value) before it reports success — otherwise a ring-slot allocation failure is a
silent stale-data read, not a clean fallback.

## Session 20 continued — the clean single-variable result (2026-10-01)

Added `GGML_SCHED_GATHER_FIRST=1` so that **only the routed expert tables** are deferred to the device
gather (`sched_input_gatherable`, the predicate the ring path already uses) while every other host weight
keeps staging.  This removes the confound in `GGML_SCHED_STAGE=0`, which disables staging everywhere.

`llama-bench -p 8192 -n 0 -sm tensor -r 1`, `MOE_EXPERT_CACHE_MIB=16384`:

| model / ub | stage-all (auto) | gather-first | staging-off (ref) | winner |
|---|---|---|---|---|
| Q4_K_M `-ub 8192` | **1847.9** | 1493.0 | 1494.3 | staging **+24 %** |
| Q4_K_M `-ub 512` (forced) | 477.4 | **496.3** | — | gather **+4 %** |
| IQ4 `-ub 8192` | 564.0 | **829.6** | 821.3 | gather **+47 %** |
| IQ4 `-ub 2048` | 570.7 | **843.4** | — | gather **+48 %** |
| IQ4 `-ub 512` (forced) | 155.5 | **315.5** | — | gather **+103 %** |

`gather-first == staging-off` on both models, so **the non-expert host weights are never staged** — the
entire staging decision is "whole-shard staging vs routed gather for the expert tables".  The sign flips
by model *and* by ubatch width, i.e. exactly the `-sm tensor` expert-upload trade, and it is **not** a
property of the parameter count.

### Reading it

Per-pass staging host time vs the gather path's pass time (the "floor"):

| config | staging host/pass | gather pass | verdict |
|---|---|---|---|
| Q4_K_M `-ub 8192` | 4.2 s | 5.5 s | staging host < floor -> **stage** |
| Q4_K_M `-ub 512` | 16.0 s | 16.5 s | ~equal -> gather (marginally) |
| IQ4 `-ub 8192` | 12.2 s | 9.7 s | staging host > floor -> **gather** |
| IQ4 `-ub 512` | 48.5 s | 26.0 s | staging host >> floor -> **gather** |

So the operative rule is roughly **"stage iff the whole-shard host-memcpy time beats the gather's pass
number"**, and the inputs to it are: shard bytes per ubatch (model geometry x split), the routed-union
size (active experts x ubatch, which is why the crossover moves with `-ub`), the host-memcpy rate, and the
non-expert compute (architecture — IQ4's QSA sparse attention gives it a much larger compute budget per
token than Q4_K_M at the same expert bytes, which is why the gather's device kernel can hide its cost).

### Consequence for the design

A fixed policy cannot be right: the existing `sched_stage_min_tokens` gate stages for *wide* batches, which
is optimal for Q4_K_M and exactly wrong for IQ4 (and it is also wrong for Q4_K_M at narrow `-ub`).  The
choice needs to be **adaptive** — either a runtime probe (measure a ubatch each way, then lock in) or a
calibrated rule keyed on measured shard bytes vs the measured H2D/gather rates, in the same spirit as the
existing `sched_stage_min_tokens` bandwidth calibration.  `GGML_SCHED_GATHER_FIRST=1` (the experiment gate)
is the gather-always arm; it is a **+47..+103 % IQ4 win and a -19 % Q4_K_M loss**, so it must not become the
unconditional default.

### Also landed (independent of the above)

`ggml_backend_meta_stage_input` now **aborts the attempt when a device cannot stage its slice**
(`stage_gather` returning false — e.g. its whole-range scratch is bounded at 256 MiB and the range
exceeds it).  Before, the return value was ignored, the ring slot stayed unfilled and its done-event
unrecorded, and the consumer read whatever the slot last held — silently correct only while the weights
are static.  This is what made `GGML_META_GATHER_MODE=2` look like +241 % on IQ4; with the fix that arm
reproduces the staging-off number (819 t/s), as it should.  `GGML_META_SCRATCH_MB` additionally makes the
scratch bound tunable (default 256 MiB) so a large-shard model can be tested on the device-D2D arm.

## Session 20 concluded — 122B-A10B, gemma4 excluded, and the refined rule (2026-10-01)

### Qwen3.5-122B-A10B (Q4_K_XL, qwen35moe, 10B active, 73 GiB, 3 shards)

`llama-bench -p 8192 -n 0 -ub 8192 -r 1`, `MIB=16384`, 3x R9700:

| config | pp8192 | STAGE_INPUT | host gather |
|---|---|---|---|
| **baseline `-ngl 99` (all on GPU)** | **3378.8** | — | — |
| `-ncmoe 99 -sm tensor` stage-all | **675.5** | 24653 ms (12.3 s/pass) | 23402 ms |
| `-ncmoe 99` gather-first | 615.2 | 0 | — |
| `-ncmoe 99` staging-off | 618.0 | 0 | — |

So the 122B is a **staging-preferring** point (676 vs 616, staging +10 %), like Q4_K_M and unlike IQ4.

### The complete matrix (all same-model single-variable A/Bs)

| model | arch | active | ub | staging | gather | winner |
|---|---|---|---|---|---|---|
| Q4_K_M 35B-A3B | qwen35moe | 3B | 8192 | **1848** | 1493 | staging +24 % |
| Q4_K_M 35B-A3B | qwen35moe | 3B | 512 | 477 | **496** | gather +4 % |
| 122B-A10B | qwen35moe | 10B | 8192 | **675** | 616 | staging +10 % |
| IQ4 Flash-Next | qwen4exp | 6B | 8192 | 564 | **830** | gather +47 % |
| IQ4 Flash-Next | qwen4exp | 6B | 2048 | 571 | **843** | gather +48 % |
| IQ4 Flash-Next | qwen4exp | 6B | 512 | 156 | **316** | gather +103 % |
| gemma4 26B-A4B | gemma4 | 4B | — | (excluded from `-sm tensor`) | | |

**Active-set size is NOT the discriminator**: the 122B has 10B active (more than IQ4's 6B) yet prefers
staging.  **Architecture is**: both `qwen35moe` points prefer staging; both `qwen4exp` points prefer the
gather, by a wide margin.  The mechanism is the ratio below.

### The rule that fits every point

Stage iff the **whole-shard host-gather time per pass** is less than the **gather path's pass time**:

| model | staging host/pass | gather pass | verdict |
|---|---|---|---|
| Q4_K_M ub8192 | 4.15 s | 5.49 s | stage ✓ |
| Q4_K_M ub512 | 16.0 s | 16.5 s | gather (marginal) ✓ |
| 122B-A10B | 11.7 s | 13.3 s | stage ✓ |
| IQ4 ub8192 | 12.2 s | 9.7 s | gather ✓ |
| IQ4 ub512 | 48.5 s | 26.0 s | gather ✓ |

The host gather runs at a near-constant ~10-14 GB/s single-threaded, so its per-pass cost is set by the
shard bytes; the gather path's pass time is set by the model's compute.  `qwen4exp`'s QSA v3 sparse
attention gives it a much larger compute budget per token, so the device gather kernel hides its cost
there and the host gather does not.

### Consequence (unchanged, now well-supported)

A fixed policy cannot be right — `sched_stage_min_tokens` stages for *wide* batches, which is optimal for
the two `qwen35moe` models and wrong for `qwen4exp` (and marginal at narrow ub).  The gate must be
**adaptive**: a runtime probe (measure one ubatch each way, lock in) or a rule keyed on the measured
host-gather time vs the measured pass time.  `GGML_SCHED_GATHER_FIRST=1` is the gather-always experiment
knob.

### gemma4 — excluded from `-sm tensor` (2026-10-01)

`llm_arch_supports_sm_tensor()` now rejects `LLM_ARCH_GEMMA4`, so `-sm tensor` fails at load with
`LLAMA_SPLIT_MODE_TENSOR not implemented for architecture 'gemma4'` (covers `-ncmoe`, `--fit` and a low
`-ngl`, since it is arch-level).  Reason: gemma4's fused expert tensor has a *segmented* split layout and
the host-resident-MoE per-ubatch upload has no correct path — the async setter is contiguous-only, and the
MMQ tail pad breaks row alignment, so it asserted at the first expert upload; a partial fix (segmented
port + tail walk + 1-D copies) still hung non-deterministically.  **Parked follow-up**, not abandoned:
finish the segmented async upload (with a deterministic repro, e.g. `compute-sanitizer`) and then re-enable
`-sm tensor` for gemma4.  `-sm layer` works today (pp8192 758.7 t/s, ub 8192).

No regression from the exclusion: gemma4 `-sm layer` 758.7, Q4_K_M `-sm tensor` 1843, IQ4 556.

## Plan

1. **Revert the temporary diagnostics** (or move them behind a `GGML_SCHED_SYNCDBG=2` guard) so the tree is
   clean before any patch is cut.
2. **Fix the latent `stage_gather` bug**: DONE — `ggml_backend_meta_stage_input` aborts on a `stage_gather`
   failure instead of reporting success (see above).
3. **Verify correctness on the staged path**: the `stage_gather` abort fix removes the stale-slot read; the
   device-D2D arm on Q4_K_M was verified byte-identical with `-ub 8192`.
4. **Decide the real item-3 fix** — now two independent costs:
   a. the **USER-branch pipeline drain** (gemma4 `-sm layer`): async copy + in-stream wait + pinned source
      lifetime (the original premise, 21.2 s/pass there);
   b. the **expert-table staging/gather gate** (Q4_K_M/IQ4 `-sm tensor`): adaptive (see above), blocked on
      more data — Gemma4 is blocked by the pre-existing assert, the 122B/10B-active model is pending.
5. **Fix the pre-existing `ggml_backend_meta_set_tensor_async` assert** (`nr[0] != 1`): give the async path
   the same segmented handling `set_tensor` has (or fall back to the sync path).  Until then gemma4 cannot
   run `-sm tensor` with offloaded experts at all.
6. Gates + `expNN` + docs, as usual.

## Adaptive gate — design proven, implementation opened and handed over (2026-10-01)

**The probe is necessary, not a choice.** The 122B and IQ4 move the *same* shard bytes per pass (166 GB)
with opposite winners, and the host-staging share of the staged pass is **higher** for the staging winners
(Q4_K_M 4.15/4.42 = 94 %, 122B 11.7/12.1 = 96 %) than for IQ4 (12.2/14.5 = 84 %), so neither a static
formula nor a staged-pass decomposition can separate them.  Only the two arms' *own* pass times can.

**Design (implemented, then reverted):** `GGML_SCHED_STAGE_AUTO=1`; the scheduler times pass 0 with
staging, pass 1 with the gather (`sched->stage_gather_first = true`), then latches
`stage_gather_first = (gather_pass < staged_pass)` and stops timing.  Both arms are bit-identical, so the
choice is performance-only.  Wired into the existing `sched_gather_first(sched)` gate that
`sched_input_gatherable` already feeds.

**Why it was reverted:** the first version (single-pass, latch on
`host_input_us*100 >= pass_us` — the wrong metric) never latched.  Corrected to the two-pass form, it
latched *and* produced an impossible throughput — **Q4_K_M pp8192 5989 t/s** and **IQ4 1564 t/s** against
the known staged/gather numbers (1843 / 1493 and 556 / 830).  5989 is faster than the all-resident
baseline, i.e. **the expert upload was being skipped entirely** — a correctness failure, not just a perf
one.  Reverted to the known-good tree (`51b1f48be`), which reproduces Q4_K_M 1847 and IQ4 556.

**First thing to check next time:** the env arm `GGML_SCHED_GATHER_FIRST=1` is correct (Q4_K_M 1493,
IQ4 830) while the latched arm was not, so the bug is in the interaction of the latch with the rest of the
input loop — most likely `sched->stage_gather_first` being read before it is initialised (check every
`ggml_backend_sched_new` path and any sched the caller creates without it) or the deferred
promotion/take-over record being built while the expert input is skipped.  Add an assert that a routed
expert table input was either staged, gathered, taken over or copied on every pass, and run the
byte-identity gates before any perf number.

## Session 21 — r28 rebase + adaptive staging-vs-gather probe (2026-10-02)

### Rebase onto delivery r28

The campaign was rebased from delivery **r26** (`0d58404e1`) onto **r28** (`60361cb9f`).  The rebase
replayed all 41 campaign commits with **no conflicts**; the net campaign diff is unchanged (12 files,
5321 insertions before the probe).  The r28 delivery changed `ggml/src/ggml-cuda/ggml-cuda.cu` and
`mmvq.cu`, the two files the campaign also touches, but the hunks do not overlap.  New branch
**`wip-moe-devmap-r28`**, tip `171b7e18e`, patch `exp22`.  Verified on the rebased tree: 2-GPU
`-sm tensor` oracle **`de8be4d0c90c`** (byte-identical to r26), `test-backend-ops -o MUL_MAT_ID`
**929/929**.

### The first probe attempt, root-caused

Session 20 finished with the adaptive probe reverted: it latched correctly but then produced an
**impossible throughput** (Q4_K_M pp8192 5989 t/s) and was assumed to be skipping the expert upload.
This session reproduced it exactly — first implementation latched `gather` and reported **6553 t/s**
— and found the real mechanism:

**A staging pass leaves the whole expert table resident in the per-device staging ring.**  The
gather arm sampled immediately after a staging pass reads that residency instead of copying the
routed experts, so it measures **~40 % fast** (Q4_K_M `-ub 8192`: probe gather sample 1.24 s vs the
true cold gather 2.03 s; forced `GGML_SCHED_GATHER_FIRST=1` = 4035 t/s).  The output was **correct**
(the weights are static), which is why the byte-identity check alone did not flag it.

Three fixes make the probe faithful:

1. **Sample the gather arm FIRST** (one warm-up pass + one sample), then the staging arm.  A staging
   pass can no longer taint the gather measurement.
2. **Synchronize each probe pass.**  The measurement moved from `compute_splits` into
   `ggml_backend_sched_graph_compute_async`, which forces a `ggml_backend_sched_synchronize` on a probe
   pass before timing it — `compute_splits` alone only launches, so the GPU gather looked free.
3. **5 % hysteresis toward staging.**  The two arms measurement noise is ~10 %, so only switch to the
   gather when it wins by ≥ 5 %; a near-tie stays on the delivery-default staging arm.

Latch message: `sched_stage_auto_finish: stage-auto latched {staging|gather} (staging N us, gather M us)`
(needs `-v`).

### r28 measurements (3x / 2x R9700, warm `-r 3`)

| model | devices | ub | staging | gather | probe latch |
|---|---|---:|---:|---:|---|
| Q4_K_M 35B-A3B | 2 | 8192 | **5283** | 4035 | staging ✓ |
| Q4_K_M 35B-A3B | 2 | 512  | 704 (gather forced¹) | 714 | n/a (no decision) |
| IQ4 Flash-Next | 3 | 8192 | **1414** | 1233 | staging (5 % hysteresis) ✓ |

¹ Below `sched_stage_min_tokens` (~1536) staging is gated off, so the gather always runs and the
probe makes no decision.

**The r28 delivery changed the trade.**  On r26 the report measured IQ4 gather **+47 %** over staging
(830 vs 564) and Q4_K_M staging **+24 %** (1848 vs 1493).  On r28 staging is competitive or better at
`-ub 8192` on both models (the block-06 staging ring is the delivery’s work), and at `-ub 512` the two
arms are within noise because staging is gated off.  So the adaptive gate’s practical value on r28 is
small, and it is kept **opt-in** (`GGML_SCHED_STAGE_AUTO=1`) rather than defaulted.

### Gates

* Default path unchanged: 2-GPU `-sm tensor` `-ncmoe 0` oracle **`de8be4d0c90c`** (probe compiled in,
  `GGML_SCHED_STAGE_AUTO` unset); `test-backend-ops -o MUL_MAT_ID` **929/929**.
* Probe on: Q4_K_M latches staging; IQ4 latches staging; prefill output byte-identical across probe /
  staging / gather / staging-off (`c8b345cfafaa`).

### Patch / tip

Branch `wip-moe-devmap-r28` tip **`171b7e18e`**, patch
**`exp22-moe-expert-cache-r28-adaptive-stage.patch`** (`git diff 60361cb9f..HEAD`, clean-applies to r28).

### Halo (gfx1151) validation (2026-10-02)

Rebuilt the exact campaign tree on `halo` (gfx1151, ROCm 7.14): `~/llama.cpp` base + the r28 net patch
(`rdna-boosts-all.patch`, applied tree `dc2decae…` == r28) + `exp22`, giving tree `d5438fa9…` == the
local campaign tree.  Clean build (`~/bin/build-llama-rocm-714`); `test-backend-ops -o MUL_MAT_ID`
**929/929**.

The transparency gate **FAILED** and exposed a latent bug (README §0.C).  On gfx1151 `-ncmoe 99` does
not offload when the model fits — the log says `offloaded 42/42 layers to GPU` — so there are no
host-resident expert weights, the cache never arms (`tables=0`, `arena=0.0 MiB`, `takeover=0`), and
setting `MOE_EXPERT_CACHE_MIB` still changes the output because `ggml_cuda_cache_blocks_fusion()`
stands the cache-band fusions down whenever `!moe_cache_has_arena()`:

| run | hash |
|---|---|
| halo `-ncmoe 0` (oracle) | `15038c19ddc8` |
| halo `-ncmoe 99` (cache off) | `15038c19ddc8` (== oracle) |
| halo `-ncmoe 99` + `MOE_EXPERT_CACHE_MIB=8192` | `de8be4d0c90c` |
| local gfx1201 `-ncmoe 0` (no MIB) | `de8be4d0c90c` |
| local gfx1201 `-ncmoe 0` + `MIB=8192` | `15038c19ddc8` |

The divergence reproduces on gfx1201 (`-ncmoe 0`), so it is not gfx1151-specific: enabling the cache when
it can never take an input over still turns the cache-band fusions off.  The adaptive staging-vs-gather
probe is meta-only and does not fire on a single GPU, so it is inert on halo and unchanged by this.
