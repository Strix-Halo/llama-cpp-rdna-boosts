# H2D staging ring: overlap op-offloaded MoE expert uploads with compute (issue #50)

Status: **OPEN — prototype implemented and measured (2026-09-26)**.  Not part of the delivery.  Origin:
community offer in [issue #50](https://github.com/stew675/llama-cpp-rdna-boosts/issues/50)
(briansp2020) of an opt-in "`GGML_CUDA_H2D_STAGING_SLOTS`" ring, plus
`GGML_SCHED_MOE_COPY_ALL_MIN_TOKENS` and `GGML_SCHED_EVENTS`.  This tree records the independent
investigation, the design the delivery would carry, and `h2d-stage.patch` — a working prototype that
overlaps the upload with compute and is byte-identical (see §5b).

## 1. The problem

With host-resident MoE experts (`--cpu-moe` / `-ncmoe N`), a prefill ubatch at/above
`GGML_OP_OFFLOAD_MIN_BATCH` (default 32) uploads each layer's expert tensors to the GPU through the
scheduler's op-offload path (`ggml_backend_sched_compute_splits`, `ggml-backend.cpp`): it copies a
layer's weights, computes the layer, copies the next, … .  The copy and the compute are back to back,
so on a bandwidth-limited link the prefill is **copy-bound**.

This is the single biggest lever for VRAM-poor users: it is what lets a 21 GiB MoE model prefill on a
card that cannot hold it, and right now that path throws away most of the link.

## 2. Measurements (this box)

Host: 1× R9700 gfx1201, ROCm 7.14, model `Qwen3.6-35B-A3B-UD-Q4_K_M` (21.10 GiB, 35.51 B params),
`llama-bench -ncmoe 99 -fa 1 -p 8192 -n 1`.

**PCIe is x4 on this box** (the cards report a 32 GT/s link but the BIOS allocates 4 lanes).  That is
important: it is *not* the reporter's x16 box, and it moves the optimum (see §4).  Measured H2D with
`h2d-bw.cpp`: **14.5 GB/s**, identical for pageable and pinned memory — i.e. the link, not the buffer,
is the limit.

| config | pp8192 t/s |
|---|---|
| `-ncmoe 0` (all experts on GPU, no upload) | **6502** |
| `-ncmoe 99`, `-b 8192 -ub 8192` | 3154 |
| `-ncmoe 99`, `-b 8192 -ub 4096` | 2247 |
| `-ncmoe 99`, `-b 8192 -ub 2048` | 1410 |
| `-ncmoe 99`, `-b 8192 -ub 1024` | 784 |
| `-ncmoe 99`, `-b 8192 -ub 512` | 473 |

So the upload costs **4.6x** at `ub 2048` (6502 vs 1410).

**Gotcha found while measuring:** `-ub` is capped by `-b` (default 2048).  A `-ub` sweep without
raising `-b` plateaus at 2048 and hides the effect — the first sweep looked like "nothing scales past
2048"; with `-b 8192` the volume scaling is clean.  Always pass `-b` with `-ub` for this work.

**Where the time goes** (temporary scheduler instrumentation, since removed):

* Each prefill ubatch copies **~15.5 GB** of used experts — essentially the whole expert set, and it is
  *independent of the ubatch size*.  At `ub 2048` the 4 ubatches copy ~62 GB; at `ub 8192` one ubatch
  copies ~15.5 GB.  This is why larger `-ub` wins: it is fewer whole-set passes, not less per pass.
* The scheduler's input loop takes **1.46 s per ubatch** (copy ~1.07 s at 14.5 GB/s + ~0.4 s of ids
  readback / gather / event waits) — i.e. **~100 % of the prefill time** at `ub 2048` (4 × 1.46 s = 5.8 s).
  At `ub 8192` it is 1.46 s of a ~2.6 s pass, so the copy is still ~40 % and the compute ~50 %.
* **Whole-tensor trial upload (`COPY_ALL`) is *slower* here**: 1219 vs 1408 t/s at effective `ub 2048`.
  On x4 the used-expert pruning saves more bytes than the ids readback costs, even when most experts are
  used.  So `GGML_SCHED_MOE_COPY_ALL_MIN_TOKENS` is an x16/large-batch optimisation, not a universal one
  — any port must make it conditional, not a straight lift.

## 3. Code map

* `ggml/src/ggml-backend.cpp`
  * `ggml_backend_sched_backend_id_from_cur` — picks the offloader (`1.off`, `ggml_backend_offload_op`).
  * `ggml_backend_sched_compute_splits` — the copy-then-compute loop and its `input_cpy` reuse waits.
    The MoE branch (ids readback + used-expert gather + `ggml_backend_tensor_set_async`) is here, as is
    the whole-tensor `else` branch.
* `ggml/src/ggml-cuda/ggml-cuda.cu`
  * `ggml_backend_cuda_set_tensor_async` / `_2d_async` — the H2D copy, on `cuda_ctx->stream()`.
  * `ggml_backend_cuda_cpy_tensor_async` — device-to-device only.
* `ggml/src/ggml-cuda/common.cuh`
  * `ggml_backend_cuda_context::streams[dev][GGML_CUDA_MAX_STREAMS]` (`GGML_CUDA_MAX_STREAMS` = 8) and
    `stream(device, i)` / `stream()` — a **dedicated copy stream is already available** (`stream(dev, 1)`);
    it is unused today.  `copy_event` exists but is for the old copy-thread path.
  * `ggml_cuda_stream_context` — "forked stream" support added for the concurrent MoE work.
* `ggml/src/ggml-cuda/fattn-common.cuh` — the block-15 staging arena
  (`fattn_stage_try_get`, bounded by `GGML_CUDA_FA_STAGE_MAX_MB`, null on failure) is the **precedent**
  for how a device scratch should be allocated here: outside the graph reserve, bounded, fall back on OOM.

## 4. The design space

**Why this is x4/x16-dependent.**  Overlap only buys what the compute can hide.  At `ub 8192` the copy
is ~1.07 s and the compute ~1.26 s, so perfect overlap takes the pass from ~2.6 s to ~1.26 s (~2x,
matching the reporter's +39-74 %).  But when the copy is the majority (`ub 2048`, copy 4.3 s vs compute
1.26 s) overlap can at best reach the *copy* time: **~4.3 s → +37 %, not 2x**.  So:
* **x16 (fast link, copy ~ compute):** the ring hides nearly the whole copy.
* **x4 (this box, copy ≫ compute):** the ring helps, but **reducing the volume** helps more.

The two levers, in order:

1. **Fewer whole-set passes** (volume).  Larger `-ub` does this but costs compute-buffer VRAM — exactly
   what a VRAM-poor user lacks.  A device **expert cache** (the separate draft PR #27861) is the real
   volume fix; it is orthogonal and complementary.
2. **Overlap** the remaining copy with compute (the ring).  This is the general, VRAM-neutral win and
   the one worth building here.

Sub-levers: pin the host source (measured no gain on this box — the link is the limit, so **do not**
spend effort there); remove the per-layer ids readback sync (useful mainly when the copy is cheap, i.e.
x16 — make it conditional).

**What residency tracking can and cannot do.**  The op-offload path streams each offloaded weight into
a reused compute-buffer slot; it has **no residency**, so the same expert tensor is re-uploaded once per
ubatch (~15.5 GB here, whole set).  Tracking "already resident" therefore only pays off via a **device
expert cache** — keep a tensor on the GPU and skip its copy on a hit.  That is a *different* feature
(the draft upstream PR #27861) and it is the **volume** lever, where the ring is the **latency** lever:

* A cache reduces the *number* of uploads (up to `n_ubatches×` when it holds the whole set); the ring
  reduces the *cost* of the uploads that remain.
* They **compose**: cache hits skip the copy, misses go through the ring.  A combined design would be
  cache → prune (when the batch is narrow) → ring-overlap (the rest).
* The cache **costs the very resource the feature exists to save**.  For the 35B here the expert set is
  ~16 GB; a 32 GB card can hold it (and would gain most from a cache), but a 16 GB card cannot, and
  there the ring is the only lever.  So the ring is the VRAM-neutral general case; the cache is the
  high-VRAM win.
* Residency alone does **not** rescue the current streaming path: there is nothing resident to track.
  And the ring's whole-tensor uploads *bypass* the ids pruning (you cannot prefetch a subset whose ids
  are only known after the router runs), which is why the prototype trades volume for overlap.
* Tracking *is* possible — the question is hit rate.  A ring of the last few whole tensors gets **zero**
  hits because the access is a repeated linear scan over ~144 distinct tensors.  A partial whole-tensor
  cache is also useless on that cyclic scan (LRU thrashes).  But the router selects a **subset** per
  layer, so a bounded **per-layer, per-expert** cache can hit even when the whole set does not fit —
  that is the draft expert-cache PR's design (~1.3 MB/expert here, so a few GB buys a real hot set).
  It composes with the ring for the misses, but it is a PR-sized feature with its own eviction/
  correctness surface.

**Options:**

* **(a) Scheduler-level prefetch ring (proposed).**  The scheduler knows the split order, so it can issue
  split S+1's host→device copies on a copy stream into a ring of device slots while split S computes.
  Bounded arena with a null/fallback path (block-15 precedent) so it cannot OOM.  Generic enough to sit
  in `ggml-backend.cpp` with a small CUDA hook.
* **(b) CUDA-backend-only ring** (the reporter's shape): simpler to land, but the scheduler still drives
  the copies, so some overlap logic has to live there anyway.
* **(c) Expert cache** (#27861): cuts the volume across ubatches but needs the VRAM it is trying to save.
* **(d) Bigger `-ub` only**: free of code but costs VRAM and plateaus on the compute.

## 5. Proposed design (delivery shape)

Scheduler-level, bounded, default-on with a kill-switch — matching the repo's conventions:

1. **Dedicated copy stream.**  Use `ggml_backend_cuda_context::stream(device, 1)` (or a named
   `copy_stream()`), and add a way to issue `set_tensor_async` on a chosen stream (the current one
   hardcodes `cuda_ctx->stream()`).
2. **Bounded staging ring.**  Allocate `N` slots (`GGML_SCHED_STAGE_SLOTS`, default ~2-3) in a dedicated
   device arena sized to the largest offloaded tensor, capped by `GGML_SCHED_STAGE_MAX_MB` (default
   ~1-2 GiB, so the feature is usable by VRAM-poor users).  `try_get` semantics: on `cudaMalloc`
   failure, disable and fall back to the in-order path (never abort).  The ring must be **outside** the
   compute-graph reserve so `--fit` sees it (block-15 issue #33 lesson).
3. **Pipeline.**  For split S: wait `copy_done[S]`; issue split S+1's copies into the next free slot on
   the copy stream; record `copy_done[S+1]`; run split S; record `free[slot]` after the last consumer.
   The consuming nodes read the staged tensor instead of the reused `input_cpy` (this is what removes
   the "wait for the previous split before overwriting the input" serialisation).
4. **Adaptive copy-all vs used-experts.**  Keep the ids pruning where it wins (x4: always measured); use
   whole-tensor only when the batch is wide *and* the ids sync is a measured fraction of the copy
   (x16 case).  Gate it, do not hardcode.
5. **Graphs.**  Skip CUDA/HIP graph capture for a compute that consumes staged uploads (prefill only),
   as the reporter found necessary.
6. **Bit-identity.**  The staged bytes are the same bytes; only their placement/timing changes, so the
   result must stay byte-identical.  That is the QA gate.

## 5b. Prototype (2026-09-26) — implemented and measured

`h2d-stage.patch` (against the r10 tree; `GGML_SCHED_STAGE=1` enables it) is a first-cut implementation
of §5.  It is **whole-tensor** by construction: the overlap needs the bytes known up front, so the
used-expert pruning (which needs a device-side ids readback) is bypassed.  That is the reporter's
`COPY_ALL` coupling, and it is why the gain is bandwidth-dependent.

* **Backend interface** (`ggml-backend-impl.h`): four optional hooks — `stage_buffer(slot,size)`,
  `stage_upload(dst,src,size,ev)` (H2D on a dedicated copy stream, `stream(dev,1)`, then record `ev`),
  `stage_wait(ev)` (make the copy stream wait for `ev`), `stage_d2d(dst,src,size)` — all NULL by default,
  so every other backend keeps the in-order path.  The meta backend lists them as NULL too.
* **CUDA backend** (`common.cuh`, `ggml-cuda.cu`): a 4-slot device ring (`h2d_stage_buffer` grows a slot
  on demand, returns null on `cudaMalloc` failure), the four hooks, and the ring freed in the context
  destructor.
* **Scheduler** (`ggml-backend.cpp`): at the top of each split the offloaded host weights are uploaded
  into the ring on the copy stream (overlapping the previous split's compute); the input loop then waits
  on the slot's event and D2Ds the slot into the real split input on the main stream, recording the
  slot's free event afterwards.  The D2D is a few % of the H2D and keeps the graph unchanged.

### Results

`llama-bench -ncmoe 99 -fa 1 -p 8192 -b 8192`, `GGML_SCHED_STAGE` 0 (serial + pruned) vs 1
(ring + whole-tensor).  Generated text is **byte-identical** (`sha=6541eadb9041`) on both boxes.

| box | link | ub | stage=0 | stage=1 | gain | `-ncmoe 0` |
|---|---|---|---|---|---|---|
| soar (gfx1201) | PCIe5 **x4** | 1024 | 783 | 744 | **−5.0 %** | 6502 |
| soar | x4 | 2048 | 1412 | 1479 | +4.7 % | 6502 |
| soar | x4 | 4096 | 2249 | 2933 | **+30 %** | 6502 |
| soar | x4 | 8192 | 3162 | 5490 | **+74 %** | 6502 |
| fingon (gfx1100) | PCIe4 **x16** | 2048 | 2507 | 3862 | **+54 %** | 6036 |
| fingon | x16 | 8192 | 4115 | 5592 | **+36 %** | 6036 |

On x4 there is a **crossover between `ub 1024` and `ub 2048`**: at `ub 1024` the whole-tensor ring
* loses* (−5 %), because the whole expert set is re-uploaded 8 times and the extra bytes (staging
disables the ids pruning) outweigh the overlap.  From `ub 2048` up the overlap wins, and steeply
(+74 % at `ub 8192`).  This is the adaptive-volume trade in miniature: the ring pairs naturally with a
*large* ubatch (few whole-set passes) or with a link fast enough that the extra bytes are cheap, so the
shippable version must pick per `(link, ub)` rather than enable unconditionally.

At fingon `ub 8192` the ring leaves only a **7 % gap** to the fully GPU-resident build (5592 vs 6036):
the upload is nearly fully hidden.  The §4 x4/x16 split is confirmed — on the wide link the ring wins at
every ubatch; on the narrow link it wins big only when the copy is small enough to hide.

### Bounded arena + adaptive gate (2026-09-26)

`GGML_SCHED_STAGE_MAX_MB` (default 1024) caps the ring; a growth past it returns null and the split
falls back to the in-order copy.  `GGML_SCHED_STAGE_MIN_TOKENS` gates staging on the batch width (read
from the `MUL_MAT_ID` expert-id tensor's `ne[1]`).

| box | ub | stage=0 | stage=1 no gate | stage=1 min=2048 | stage=1 max=32 MB |
|---|---|---|---|---|---|
| soar x4 | 1024 | 783 | 744 (−5 %) | 783 (gated) | 779 (budget) |
| soar x4 | 8192 | 3165 | 5475 | 5468 | 3163 (budget) |
| fingon x16 | 1024 | 1589 | **1958 (+23 %)** | 1597 (gated) | — |
| fingon x16 | 8192 | 4100 | 5597 | 5603 | — |

Both fallbacks are exact: the gate reproduces `stage=0` at a narrow batch, and a too-small budget
reproduces `stage=0` at any batch.

### Link calibration (2026-09-26)

The gate's threshold is now **measured, not hardcoded**: `stage_h2d_gbps` times a synchronous 512 MiB
H2D copy once per device (sized past the 64 MiB Infinity Cache, which otherwise makes a x4 link read
~25 GB/s), and the scheduler maps it to `min_tokens = 1536 − 132·(bw − 14.5)` (floored at 0), with
`GGML_SCHED_STAGE_MIN_TOKENS` as the explicit override.  Measured:

| box | measured H2D | min_tokens | ub 1024 | ub 2048 | ub 8192 |
|---|---|---|---|---|---|
| soar (PCIE5 x4) | 14.45 GB/s | 1536 | 779 (gated) | 1479 | 5479 |
| fingon (PCIe4 x16) | 28.6 GB/s | 0 | 1958 | 3858 | 5597 |

The same binary therefore self-tunes to the right crossover on both links — the x4 build gates
`ub 1024` (which regresses) while the x16 build stages it (+23 %).

**The threshold is link-dependent**: staging wins at `ub 1024` on x16 (+23 %) but *loses* on x4 (−5 %),
so a fixed constant cannot be right for both — hence the calibration.

### Known limitations / next steps

* whole-tensor only (pruning gives way to overlap) — the adaptive gate now keys on the **measured**
  link bandwidth (§ below); an adaptive volume rule could still do better;
* the ring is **bounded** (`GGML_SCHED_STAGE_MAX_MB`, default 1024 MiB) with an exact fallback;
* only the direct CUDA backend is exercised (single GPU); `-sm tensor` needs the meta backend to forward
  the hooks (next step);
* no graph-capture, concurrent-serving or deep-context gate yet.

## 6. Plan / next steps

1. ~~Re-measure on the other RDNA hosts~~ — **done** (fingon is x16; the x4/x16 prediction is
   confirmed).  `halo` (gfx1151) is deliberately excluded: it is unified-memory, so expert offload is
   pointless there.
2. ~~Prototype the minimal ring~~ — **done** (§5b), byte-identical, +36-74 % where the copy can hide.
3. **Make it shippable**: adaptive whole-tensor vs used-expert pruning; bound the ring and fall back
   (block-15 arena pattern); forward the hooks through the meta backend for `-sm tensor`; then the
   gates (same-seed text, `W=1..8`, `test-backend-ops`, MTP, deep-context, concurrent server) and the
   gfx1100/gfx1201 records.
4. **A/B against the reporter's patch** on the same box/model once their PR lands.
5. Then decide the delivery home (block 15 is the attention-memory campaign and already owns the staging
   arena precedent; block 11 is the CUDA prefill-graph skip — the graph interaction may argue for 11).

## 7. Risks / open questions

* **ROCm event semantics with pageable sources.**  The reporter notes an event after a pageable
  `hipMemcpyAsync` resolves against the copy stream's tail, so the design cannot run far ahead; our
  `h2d-bw.cpp` shows the copy *does* overlap a compute kernel, so 1-deep pipelining should still work,
  but the achievable lookahead depth must be measured.
* **VRAM cost** of the ring: cap it and fall back; do not make it a hard requirement.
* **Correctness of slot reuse**: the free event must be after *every* consumer of the slot, and views of
  the staged tensor must not outlive it.
* **`-ub`/`-b` interaction** in every benchmark (see §2).
* **Downstream of the pass**: the total per-prompt copy is `n_ubatches × whole expert set`; the ring does
  not change that, so on x4 the combination (ring + larger `-ub` where affordable) is what moves the
  number most.  Say so in any user-facing claim.

## 8. Artifacts

* `h2d-stage.patch` — the prototype (against the r10 tree): `GGML_SCHED_STAGE=1` enables the 4-slot
  staging ring.  Applies clean on top of `scripts/apply-all.sh`; build with `BUILD_DIR=build-rocm`.
* `h2d-bw.cpp` — standalone HIP H2D bandwidth + copy/compute overlap probe.  Build:
  `hipcc -O2 --offload-arch=<arch> -L/opt/rocm-7.14-gfx1201/lib -lamdhip64 -Wl,-rpath,/opt/rocm-7.14-gfx1201/lib h2d-bw.cpp -o h2d-bw`.
  On this box: 14.5 GB/s pageable and pinned; a 138 ms 2 GiB copy overlaps a 60 ms compute kernel
  (pageable total ~149 ms, not the ~198 ms serial sum).
