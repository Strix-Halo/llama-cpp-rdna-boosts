# UPSTREAM-PR: ggml-backend-meta: let the tensor split op-offload host weights (and serve byte ranges on mirrored tensors)

**Status:** prepared 2026-09-27; `git apply --check` clean on master **`84e76d8a2`**; **built and measured
standalone on pristine master + this patch** (gfx1201, 2× R9700, ROCm 7.14): `505.49 -> 1690.26 t/s`
(`llama-bench -sm tensor -ncmoe 99 -fa 1 -p 8192 -b 8192 -ub 8192`).  Not filed.

**Where it lives in the delivery:** **block 06** since r12 (2026-09-27) — `patches/0006`, from
`archive/work/h2d-staging-ring/h2d-stage.patch`, fixes 1 and 2 of three.  The same patch also carries an
op-offload H2D staging ring that overlaps these uploads (fix 3); this file is the standalone upstreamable
part.  If upstream takes it, drop it from block 06 at the next regeneration.

## The bug

`ggml_backend_meta_device_iface` leaves `offload_op` as `nullptr`, and
`ggml_backend_dev_offload_op()` returns false for a device with no hook.  The scheduler's op-offload
selection is:

```c
// ggml_backend_sched_backend_id_from_cur()
if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
    for (int b = 0; b < src_backend_id; b++) {
        if (ggml_backend_supports_op(sched->backends[b], tensor) &&
            ggml_backend_offload_op(sched->backends[b], tensor)) {
            return b;
        }
    }
}
```

so under `-sm tensor` the meta backend is **never** selected for a node whose weights live on the host,
and such a node is placed on the CPU instead.  With `-ncmoe`/`--cpu-moe` that means the whole MoE is
executed on the CPU even though every device it is split over is idle-capable of running it — the exact
opposite of what op-offload is for.  (`-sm layer` is unaffected: there the CUDA backend is in the
scheduler directly and has its own `offload_op`.)

Reproduce on master, `-sm tensor -ncmoe 99`:

* `GGML_SCHED_DEBUG=2` (needs `-v`) prints the `MUL_MAT_ID` nodes assigned to `CPU`;
* `llama-bench -m <MoE model> -ncmoe 99 -fa 1 -p 8192 -b 8192 -ub 8192 -sm tensor` reports ~505 t/s
  where `-sm layer` reports ~2700 t/s on the same box.

Fixing only the hook then exposes a second, independent gap: the scheduler's used-expert pruning (which
that branch enables) does not upload whole chunks.  It uploads a byte range of the expert tensor at a
non-zero offset, and it reads the router's expert ids as a **strided view's raw span**
(`blk.N.ffn_moe_topk`: i32 `[8, 2048]` with `nb[1] = 1024`), indexing that span with the view's own
strides.  `ggml_backend_meta_set_tensor_async()` / `_get_tensor_async()` asserted

```c
GGML_ASSERT(offset == 0);
GGML_ASSERT(ggml_is_contiguous(tensor));
```

before even looking at the split axis, so both calls **abort** (`ggml-backend-meta.cpp:2083` on the
pruning path's ids readback, with the delivery's trees).  The assertions are needed by the *partial*
split, which splices a tensor across devices as whole chunks — but they are not needed by the
*`MIRRORED`* split, where every device holds the same bytes.

## The fix

`ggml/src/ggml-backend-meta.cpp`, 48 insertions / 17 deletions:

1. Add `ggml_backend_meta_device_offload_op()`, true when *every* simple device would offload the op on
   its own (the meta backend runs the node on all of them, so it cannot accept a node only some can),
   and register it in `ggml_backend_meta_device_iface`.
2. Hoist the `MIRRORED` case of `set_tensor_async`/`get_tensor_async` **above** the chunk-arithmetic
   assertions: a mirrored tensor needs no chunk arithmetic, so an arbitrary `(offset, size)` is
   forwarded to every device (set) / read from device 0 (get).  The partial axes keep both assertions
   and are byte-for-byte unchanged.

## Validation

On **pristine master `84e76d8a2` + this patch only** (not the delivery), gfx1201, 2× R9700,
Qwen3.6-35B-A3B UD-Q4_K_M:

| `-sm tensor -ncmoe 99 -fa 1 -p 8192 -b 8192 -ub 8192` | t/s |
|---|---|
| MoE on CPU (`GGML_OP_OFFLOAD_MIN_BATCH=1000000`, i.e. the pre-patch behaviour) | 505.49 |
| with this patch (op-offload) | **1690.26** |

* `test-backend-ops -o MUL_MAT_ID` on the same build: **2/2 backends passed, OK** (this patch does not
  touch it; master's per-type mmvq caps keep `n = 16` off the MoE kernel that the delivery's raised band
  exposes a latent mis-launch in).
* `llama-cli -sm tensor -ncmoe 99` generates normally (no abort, coherent output).
* The pruning path itself — the part that needed fix 2 — is exercised by that run: it is the
  non-staged path taken whenever the expert tensor is not staged.

**What was NOT validated:** no NVIDIA/other-backend hardware was available, and the change is
backend-agnostic (scheduler + meta-backend device/MIRRORED handling only).  Multi-device counts above 2
were not re-run.  The performance claim is for a host-resident-MoE (`-ncmoe`) tensor split; a tensor
split with fully device-resident weights is unaffected (no host weight, so the branch is never taken).

## Test plan for a PR

`test-backend-ops` (unaffected), then a `-sm tensor -ncmoe N` MoE model compared against
`-sm layer -ncmoe N` and against the pre-patch build on the same box, plus a same-seed greedy
generation to confirm coherence.  A reviewer note: the *upload* of a host weight into a mirrored split
input means every device receives the whole tensor, so on a tensor split this path is for models whose
expert weights are host-resident; it does not make a tensor split faster than a layer split for them
(tensor split mirrors these weights — their split state comes from the scheduler's `GGML_OP_NONE`
input copy defaulting to `MIRRORED`, which is a separate question from this patch).
