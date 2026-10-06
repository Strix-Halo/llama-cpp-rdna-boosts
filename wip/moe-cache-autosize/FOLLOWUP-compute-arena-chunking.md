# FOLLOW-UP — compute-buffer chunking and VMM-backed buffers (TODO #42)

**Status:** investigation only, nothing here is implemented or promoted.  Written after the r20 work
(the arena fail-soft yield + the compute-buffer margin, see `ARENA-UB-TENSION.md` §13 and §14).
Read §1 first: it is the set of facts that makes both ideas concrete, and one of them is a surprise.

## 1. What the allocator actually does today (the facts)

**The compute buffer is already a chunked virtual buffer.**  `ggml-alloc.c`:

```c
#define GGML_VBUFFER_MAX_CHUNKS 16
// virtual buffer with contiguous memory range, split into multiple backend buffers (chunks)
struct vbuffer { ggml_backend_buffer_t chunks[GGML_VBUFFER_MAX_CHUNKS]; };
```

Each chunk is a separate `ggml_backend_buft_alloc_buffer` call, i.e. a separate `cudaMalloc`.  The
realloc trigger is evaluated **per chunk**:

```c
for (int c = 0; c < tal->n_chunks; c++) {
    if (ggml_dyn_tallocr_max_size(tal, c) > ggml_vbuffer_chunk_size(buf, c)) realloc = true;
}
```

**But the chunking is dormant.**  The chunk size comes from
`ggml_dyn_tallocr_new(alignment, ggml_backend_buft_get_max_size(buft))`, and `ggml_dyn_tallocr_new_chunk`
sets the new chunk's single free block to `MAX(min_size, alloc->max_chunk_size)`.  The CUDA buffer type
declares **`get_max_size = NULL` → `SIZE_MAX`** (ggml-cuda.cu:1043), so chunk 0 is unbounded, every
tensor lands in it, and the "chunked virtual buffer" collapses to **one ~6.5-11.8 GiB `cudaMalloc`**.
That single allocation is what must be physically contiguous, and it is what grows in place.

Other chunk facts:
* `ggml_dyn_tallocr_new_chunk` caps the last chunk (`n_chunks == MAX-1`) at `SIZE_MAX/2` -- i.e. the
  16th chunk is deliberately unbounded, so fine-grained unit sizes must fit in 15 chunks.
* `ggml_vbuffer_alloc` has **exactly one call site** (ggml-alloc.c, the realloc path) and it always
  passes `GGML_BACKEND_BUFFER_USAGE_COMPUTE`.  Model weights go through a different path
  (`ggml_backend_buft_alloc_buffer_n_plan`, ggml-backend.cpp:67).  **So the gallocr is compute-only** --
  any change here cannot affect weight buffers.
* `get_max_size` itself is NOT compute-only: `ggml-backend.cpp:67` (weights), the meta multi-GPU split
  (ggml-backend-meta.cpp:334) and the RPC path (ggml-rpc.cpp:1299) all read it.  Bounding it *globally*
  would chunk the weight buffers too -- do not do that.

**Where the margin landed (r20).**  `ggml_vbuffer_alloc` now allocates `layout_size * (1 + pct/100)`
with `GGML_COMPUTE_BUFFER_MARGIN_PCT` (default 10).  Because the trigger compares against the
*allocated* size, a growth inside the margin cannot realloc.  Measured: the layout delta was +3.3 %
(6564 -> 6780 MiB server; the cli failure was a 11765.52 MiB allocation), pct=0 aborts 3/3 and pct>=8
succeeds 3/3.  Cost at pct=10: 1275 MiB of arena residency (63.8 % -> 61.5 %, i.e. ~127 MiB per point,
~139 MiB/device for the last two points).

## 2. Idea A -- uniform, bounded compute chunks

**Goal:** make the compute buffer N allocations of one fixed unit size, so its physical units are the
same kind of object as the arena's table allocations and the workspace pool's blocks.  Uniform units
are what kill external fragmentation: any freed unit can serve any request.

**Sketch:** give the scheduler's gallocr a *compute-only* chunk budget instead of changing the global
`get_max_size` (which would chunk weight buffers -- see §1).  Options, cheapest first:

1. Add a field/parameter to `ggml_dyn_tallocr_new` (or a `ggml_gallocr_set_max_buffer_size`) and set it
   only for the gallocr the scheduler creates for compute.  The existing machinery then does the rest.
2. An env knob (`GGML_COMPUTE_BUFFER_CHUNK_MIB`) read in `ggml-alloc.c`, applied only when
   `usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE`.

**Caveats / open questions:**
* **The 16-chunk cap.** A 6.8 GiB buffer in 512 MiB units needs 14 chunks; in 256 MiB units, 27.  The
  16th chunk is unbounded, so a too-small unit silently reverts to one huge allocation for the tail.
  Either raise `GGML_VBUFFER_MAX_CHUNKS` (each chunk costs a `ggml_backend_buffer` object and a
  `cudaMalloc`) or choose a unit >= peak/15.
* **Does chunking actually reduce reallocs?** A *fixed* unit means every chunk's `max_size` is the unit,
  so the per-chunk comparison is against a constant -- a layout that grows by one tensor would want
  another unit-sized chunk rather than a bigger existing one.  That may be *better* (count changes, not
  sizes) but it is not obviously so.  **Measure it.**
* **The margin and the chunk size interact.**  A fixed unit plus a margin is `unit * 1.10`; decide
  whether the margin applies per chunk (the r20 behaviour) or only to the last chunk.
* **Fragmentation does not vanish on its own.**  Uniform *request* sizes only help if the physical
  allocator can satisfy them from a uniform free list; a device that is 40 % full of one 11 GiB
  allocation is still fragmented.  This is the argument for Idea B.

**Validation:** the cli repro (`-ub 8192 -b 8192`, 2 GPU, cache auto, 16k prompt) must stay clean; the
server concurrent repro must stay clean; record the number of chunks, the total compute-buffer size,
the arena residency, and the stand-down count.  A win is fewer/smaller stand-downs at equal residency.

## 3. Idea B -- a VMM-backed buffer type (chunked VA, non-contiguous physical)

**Goal:** keep the single contiguous *virtual* range the kernels need (`base + slot*stride`), but back it
with fixed-size physical chunks mapped in on demand.  This is the only way to relax the *physical*
contiguity without touching a kernel.

**The existing VMM pool is not usable as a general allocator** (ggml-cuda.cu:604):

```c
void * alloc(...)  // cuMemCreate(granularity-rounded chunk) + cuMemMap at the end of a 32 GB VA pool
void free(void * ptr, size_t size) {
    pool_used -= size;
    GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));   // LIFO only
}
```

* `free` **never unmaps** -- physical memory is never given back for the life of the pool.
* It **asserts LIFO**: all deallocations must be in reverse order of allocation.  That is fine for the
  per-op workspace pool; it is unusable for the arena, which must yield memory on demand.
* It reserves a fixed `CUDA_POOL_VMM_MAX_SIZE = 32 GB` of VA per device, and `pool_size` grows
  monotonically.

**And it is switched off for AMD anyway:** `GGML_HIP_NO_VMM` defaults to `ON` (ggml/CMakeLists.txt:219)
and is `ON` in this build, so `GGML_USE_VMM` is not defined, `device_vmm` stays 0, and
`new_pool_for_device` always returns `ggml_cuda_pool_leg` (raw `cudaMalloc`).  The code carries a HIP
workaround for `ROCR-Runtime#285` (unmap each mapping individually, ggml-cuda.cu:626) -- a hint that
mapping/unmapping is the risky part on ROCm and worth reading before turning it on.

**If pursued, the shape would be:** a new `ggml_backend_cuda_buffer_type` variant (or a mode of the
existing one) that reserves a VA range of the layout size and maps
`granularity`-sized chunks into it, plus a real `free` (unmap + release, non-LIFO, with a free list of
mapped chunks).  Then the compute buffer *and* the arena could draw from one pool of uniform chunks --
literally "swap an arena chunk for a compute buffer chunk".

**Risks / to measure before believing any of it:**
* VA cost and mapping cost; `cuMemMap`/`cuMemSetAccess` on every growth.
* Peer access: VMM memory is not implicitly peer-accessible, so P2P/RCCL needs explicit
  `cuMemSetAccess` (the existing pool already handles this -- see `use_peer_access` there).
* Why HIP defaults it off.  Start by reading `ROCR-Runtime#285`; then try it on gfx1201 and see.
* Cheap first experiment: just flip `GGML_HIP_NO_VMM=OFF` and re-run the cli repro **with
  `GGML_COMPUTE_BUFFER_MARGIN_PCT=0`**.  If the original abort disappears with no margin at all, VMM
  alone is doing real work and the rest is worth building; if it does not change, VMM does not address
  the failure and the time is better spent on Idea A.
* Note the VMM pool is a *workspace* pool; the compute buffer and the arena go through the *buffer type*.
  Enabling the pool today would not touch what actually failed.

## 4. Suggested order

1. **Idea B's cheap probe** (flip `GGML_HIP_NO_VMM=OFF`, margin off, run the cli repro).  One build,
   one answer to "does VMM matter here at all".  Do NOT ship it -- it changes every pool in the build.
2. **Idea A**, with the unit chosen from the measured peak (>= peak/15) and `GGML_VBUFFER_MAX_CHUNKS`
   raised if needed.  Validate as in §2.
3. **Idea B proper** only if (1) pays off.

Keep the r20 arena yield in place throughout: it is the backstop for everything the margin and the
chunk/vmm work do not cover, and it is already validated (6/6 concurrent-prefill runs clean, 0 full
releases, hit rate held at ~0.965-0.976).
