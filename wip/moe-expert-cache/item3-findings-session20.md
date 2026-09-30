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

## Plan

1. **Revert the temporary diagnostics** (or move them behind a `GGML_SCHED_SYNCDBG=2` guard) so the tree is
   clean before any patch is cut.
2. **Fix the latent `stage_gather` bug**: DONE — `ggml_backend_meta_stage_input` aborts on a `stage_gather`
   failure instead of reporting success (see above).
3. **Verify IQ4 correctness on the staged path** with `-ub 8192`, a ≥2048-token prompt and a fitting
   context (e.g. `-c 8192 -b 4096 -ub 4096`, `MIB` small or unset), auto vs mode 2; compare greedy text.
   Only after that is the IQ4 mode-2 number meaningful.
4. **Land the legitimate win**: prefer the device-D2D compaction over the host gather when the host gather
   is the expensive one (a size/`n_copies` heuristic, or simply default `MODE=2` when `h2d_scratch` can
   hold `whole`), gated behind an env var first, order-balanced A/B (`auto` then `mode2` then `mode2` then
   `auto`) on Q4_K_M and IQ4.
5. **Decide the real item-3 fix** based on (3): if IQ4's mode 2 *is* correct, the win is large and the fix
   is "avoid the host gather for big expert tables"; if it is garbage, the item becomes "make the meta
   `stage_input` defer routed expert tables to the B2 device gather" (the ring path already does this via
   `sched_input_gatherable`, but the meta `stage_input` path does not).
6. Gates + `expNN` + docs, as usual.
