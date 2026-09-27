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

### Merged design: PR #51's redirect + our gate/arena (2026-09-26)

The A/B with the reporter's ring (PR #51) settled the two open questions, so the prototype is now the
**merged** patch:

| element | source | why |
|---|---|---|
| op reads the ring slot directly (`input_cpy->data` redirected, restored at the next issue) | **PR #51** | the D2D was the entire gap: at `ub 8192` on soar, redirect 5638 vs D2D 5433 t/s — the slot then moves once instead of twice |
| 8 slots, runtime-tunable (`GGML_SCHED_STAGE_SLOTS`) | merged | depth, not the D2D, was the second lever: 4 slots 5488, 6 slots 5451, **8 slots 5737**, 16 slots (budget-limited) collapse |
| `GGML_SCHED_EVENTS` | **PR #51** | theirs, ported and kept: it is what lets *their* ring overlap at all; our staged path already skips that synchronize |
| bounded arena (`GGML_SCHED_STAGE_MAX_MB`, default 2048 MiB) with a clean fallback | ours | a *partially* staged ubatch is worse than either (any input left on the pruned path does an ids readback + full device sync): measured 3047 vs 5745 t/s, so an over-large ring now disables staging instead of collapsing |
| link-calibrated gate, floored above the widest verify batch | ours | keeps the x4 `ub 1024` case gated (theirs regresses 5 % there) and can never stage a decode/verify batch |
| restore-at-issue, plus a tripwire assert | ours | the restore is structural (every split re-issues), and the assertion turns any future redirected-tensor write into an abort |

The redirect is safe here for two reasons the reporter's own audit confirms: input copies are
`ggml_dup_tensor_layout` duplicates whose `data` is `NULL` until the allocator sets it, and **prefill
graphs are never captured** (`ggml_cuda_graph_is_multi_token`), so no baked pointer can outlive a slot.

**Results** (merged, stage=0 → stage=1, `llama-bench -ncmoe 99 -fa 1 -p 8192 -n 1 -b 8192`):

| box | model | ub | off | merged | gain |
|---|---|---|---|---|---|
| soar (PCIe5 x4) | Q4_K_M | 1024 | 783 | 780 | 0 % (gated) |
| soar | Q4_K_M | 2048 | 1407 | 1476 | +4.9 % |
| soar | Q4_K_M | 4096 | 2251 | 2946 | +30.9 % |
| soar | Q4_K_M | 8192 | 3168 | **5739** | **+81 %** |
| fingon (PCIe4 x16) | Q4_K_M | 1024 | 1336 | 1473 | +10.3 % |
| fingon | Q4_K_M | 2048 | 2192 | 2916 | +33.0 % |
| fingon | Q4_K_M | 4096 | 3199 | **5687** | **+77.8 %** |
| fingon | Q4_K_M | 8192 | 3978 | 6196 | +55.8 % |
| fingon | Q3_K_M | 4096 | 3486 | 6284 | +80.3 % |
| fingon | Q3_K_M | 8192 | 4168 | 5877 | +41.0 % |

Head-to-head at soar `ub 8192` (Q4_K_M): r10 3168, our D2D 5490, PR #51 5743, **merged 5739**.

On the reporter's PCIe5 x16 box (55 GB/s, Q4_K_M) the D2D version reached 2708/4728/5378/5539 at
ub 1024/2048/4096/8192 against their 2804/5250/6206/5934 — on a fast link the serialized D2D costs
~10 %, which is why the redirect was taken. The `ub 4096 → 8192` dip on fingon is a real ring/VRAM
effect (the reporter sees the same shape on their box), not a gating artifact.

**Purity:** generated text is byte-identical between `GGML_SCHED_STAGE=0`, the redirect default, and the
D2D fallback (`GGML_SCHED_STAGE_MODE=0`) on both boxes and both models — soar Q4_K_M `sha=e7e29d5a470a`,
fingon Q3_K_M `sha=82fe7fa67e03`, fingon Q4_K_M `sha=9fd6b0048aec`.  Decode is untouched (the gate
floors above the widest verify batch, and the decode weight split carries no host-weight input);
`tg64` 21.39 vs 22.10 t/s on fingon (within noise), and the 22.66 GB Q4_K_M runs without OOM on the
24 GB 7900 XTX.

### `-sm tensor`: resolved 2026-09-27 — the ring was never the problem

The blocker was **not** that the meta backend could not stage.  It was that under `-sm tensor` there were
**no H2D weight uploads to overlap at all**: the meta device declared no `offload_op`, so
`ggml_backend_sched_backend_id_from_cur()` could never place an op-offloaded node on it and the whole
MoE silently executed on the **CPU**.

```c
// ggml_backend_sched_backend_id_from_cur()
if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
    for (int b = 0; b < src_backend_id; b++) {
        if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
            return b;                                    // <- never reached: meta's offload_op was nullptr
        }
    }
}
```

Evidence: `GGML_SCHED_DEBUG=2` showed the `MUL_MAT_ID` nodes on `CPU`, and forcing the same state on the
working build (`GGML_OP_OFFLOAD_MIN_BATCH=1000000`) reproduced the reported numbers exactly
(`-sm tensor -ncmoe 99`, pp2048/ub2048: **508.42** measured vs **498.73** in the handover).

Three fixes (all in `h2d-stage.patch`):

1. **`ggml_backend_meta_device_offload_op()`** — the meta device now declares offload support when
   *every* simple device does (the meta backend runs the node on all of them).  This is the enabling fix.
2. **Mirrored tensors serve arbitrary byte ranges** (`ggml_backend_meta_set/get_tensor_async`).  Enabling
   op-offload lit up the scheduler's used-expert pruning path, which uploads a range of the expert
   tensor at a non-zero offset and reads the router's ids as a **strided view's raw span**
   (`ffn_moe_topk-0`: i32 `[8,2048]`, `nb[1]=1024`) — the old `GGML_ASSERT(offset == 0)` /
   `GGML_ASSERT(ggml_is_contiguous(...))` pair aborted on both (reproduced as a core dump at
   `ggml-backend-meta.cpp:2083`).  A MIRRORED tensor needs no chunk arithmetic, so its range is simply
   forwarded; the partial axes keep their asserts.
3. **Per-device staging** (`stage_input`) — the hook described in §5/§5b, now implemented in the meta
   backend: each device's chunk lands in that device's own ring on its auxiliary copy stream, and
   `graph_compute` waits, redirects the simple tensors and frees the slots via an RAII guard.

Also fixed here: **15 backends' positional `ggml_backend_i` initializers** omitted the staging fields, so
adding them silently assigned each backend's `graph_optimize` to `stage_buffer` (and NULLed
`graph_optimize`) — metal/vulkan/hexagon/virtgpu lost their optimizer, and metal would have crashed under
`GGML_SCHED_STAGE=1`.

**Results** (`-sm tensor -ncmoe 99 -fa 1`, 2× R9700, pp8192, `llama-bench`, this box):

| ub | MoE on CPU (pre-fix) | op-offload, no staging | + staging |
|---|---|---|---|
| 1024 | ~340 | 337.17 | 345.20 (gated) |
| 2048 | 508.42 | 620.27 | **715.71** |
| 4096 | — | 1092.56 | **1413.94** |
| 8192 | 523.06 | 1823.12 | **2741.56** |

`-ncmoe 10 -sm tensor` (host **and** device MoE layers) works and stages: 1958.74 → **2281.82** (+16 %).
For reference on the same box `-sm layer -ncmoe 99` pp8192/ub8192 is 2708.21 → **4111.14** — tensor split
replicates the expert weights (their split state is `MIRRORED`), so it uploads the whole tensor to *every*
device and cannot match layer split; the fix removes the *CPU fallback*, it does not make tensor split the
fastest way to serve `-ncmoe`.

**Gates** (2 GPU, gfx1201): stage 0 vs stage 1 greedy text byte-identical
(`-sm tensor -ncmoe 99`, 48 tokens, sha `0936c8318533`); the narrow-ubatch crash repro (ub 1024, gate
closed → pruned path) now runs; `test-backend-ops -o FLASH_ATTN_EXT` 2/2 and `-o MUL_MAT_ID` 2/2.  `-sm
layer` is untouched by all three fixes (its `stage_input` is NULL and the MIRRORED change is meta-only).

The earlier handover's "enabling it needs per-device rings, a separate change" was half right: per-device
rings *were* needed, but they were the third fix, not the first.  `ggml_backend_sched_new` now accepts
either `stage_buffer` or `stage_input` as "a backend supports staging".

### Reporter confirmation (2026-09-27, briansp2020, 55 GB/s PCIe5 x16)

Their run of the merged patch (at `c1f43fc`) on a 55 GB/s box confirms the merge:

| ub | `STAGE=0` | old prototype | PR #51 | merged `STAGE=1` | merged `MODE=0` (D2D) | `+EVENTS` |
|---|---|---|---|---|---|---|
| 1024 | 1839 | 2711 | 2804 | **2796 (+52 %)** | 2755 | 2796 |
| 2048 | 3011 | 4728 | 5259 | **5460 (+81 %)** | 4879 | 5468 |
| 4096 | 3979 | 5381 | 6207 | **6193 (+56 %)** | 5632 | 6180 |
| 8192 | 4513 | 5534 | 5939 | **5909 (+31 %)** | 5657 | 5909 |

* merged vs PR #51: within 0.5 %, except ub 2048 where the merge is **+3.8 %** — our 8 slots vs their 3.
* **`EVENTS` is neutral on the merged path**, as measured here; **D2D costs −1.5 / −10.6 / −9.1 / −4.3 %**
  at ub 1024/2048/4096/8192, confirming the ~10 % estimate for that link (and the ~3.5 % here at x4) and
  the redirect default.  `tg32` flat in every arm (33.1–34.3 t/s).
* **Same-seed pair** (`llama-completion -f prompts/code-python.txt -n 64 --seed 42 --temp 0`, sha
  `6cd450472487`) identical for `STAGE=0`, `STAGE=1` and `STAGE=1 MODE=0`; server prefill **3667 → 5485
  t/s (+50 %)** with byte-identical greedy text in all three modes and **no tripwire asserts**.
* Two of their notes are now stale: `-sm tensor` is no longer inert (above) and the `MUL_MAT_ID` failure
  is fixed in r11.

**Their one actionable point** — the calibration line was invisible at default verbosity — is fixed:
`ggml`'s `GGML_LOG_INFO` maps to **TRACE** verbosity, which is below llama.cpp's default threshold, so
only `GGML_LOG_WARN` and above survive; and `llama-bench` installs `llama_null_log_callback` whenever
`-v` is absent (`tools/llama-bench/llama-bench.cpp`), discarding *every* level.  `sched_stage_min_tokens`
now also emits a one-line notice at WARN **when `GGML_SCHED_STAGE` is set explicitly** (so it stays
silent if staging ever becomes default-on).  Verified visible in `llama-server` at default verbosity:

```
0.02.647.314 W sched_stage_min_tokens: H2D staging: 14.5 GB/s link -> whole-weight uploads staged from 1542 tokens
```

`llama-bench` still needs `-v` (the tool silences all logs otherwise), or `-lv 4` for the INFO form.

### Gate battery (2026-09-26, gfx1201)

| gate | result |
|---|---|
| `test-backend-ops -o FLASH_ATTN_EXT` | OK (2/2) |
| `test-backend-ops -o MUL_MAT_ID` | was FAIL (pre-existing, identical on the r9 tree `b48fb3f68` with no staging patch) — **fixed in r11** as a block-13 amendment: `mul_mat_vec_q_moe_launch` sized its grid for a row tile of 3 then launched the RPB 2 kernel when `k == 3*qk`, leaving the last third of the rows unwritten.  Now **2/2 OK**.  See `WORKLOG.md` 2026-09-26 (r11) |
| width purity `W=1..8` (4B, f16 + q8_0) | PURE, and `GGML_SCHED_EVENTS=1` gives the *same* hashes (`bc8c5b7b0f24c937` f16, `3aa9cb89f496df8e` q8_0) — the events knob is purity-neutral |
| MTP `draft-mtp n3` (27B Q8_0, 2 GPU) | text byte-identical `ce64c8ed4974` stage 0/1, 44.1 t/s |
| deep-context prefill (~32k prompt, `-ncmoe 99`, ub 2048) | text byte-identical `ba3f67f221b9`; prompt 1128 → **1363 t/s (+21 %)** |
| concurrent server soak (3 × ~20k prompts × 6 rounds, `-ncmoe 99`, ub 4096) | 18/18 OK both stages, clean logs; wall clock 245 s → **187 s (−24 %)** |

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
