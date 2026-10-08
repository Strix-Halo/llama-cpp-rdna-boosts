# `host-expert-dio-cache` — a bounded, pinned, DIO-filled host tier for the MoE expert cache

Status: **OPEN / Phase 0** (design + mmap removal).  Nothing here is part of the delivery.

## Why

`--host-experts mmap` (folded into block 06 in r28) backs the `-ncmoe`/`-cmoe` host expert set with the
pageable `CPU_Mapped` model mapping, so the OS page cache acts as a reclaimable host tier and the
`ROCm_Host` pinned allocation disappears (peak RSS 40.7 -> 22.2 GB on the r28 1x-R9700 measurement).

Issue #116 shows why that cannot work on this hardware.  On a GPU without XNACK (gfx1201 here reports
`XNACK enabled: NO`; verified with a HIP probe) a kernel read of a pageable address is a fatal
"page not present" fault.  The cache reads the host master in place in three places:

| site | host read |
|---|---|
| cold read (`moe_cache_get_cold` -> `mul_mat_vec_q_moe`) | `t.host_dev` |
| device gather (`moe_cache_gather_kernel`) | `weight->data` directly |
| device-side policy / seed fills | `t.host_dev` |

and one copy site: the eager fill of an axis-0 `-sm tensor` split does a pageable 2-D H2D
(`cudaMemcpy2DAsync`), which is the ROCm 7.14 `copyBufferRectAligned` fault the r28 comment already
documents.

`mlock`/`mmap+mlock` do not help: mlock stops reclaim, it does not give the GPU a mapping.
`cudaHostRegister` would, but it pins the pages, so the "reclaimable" property is gone.

So the mmap host master is **dead on RDNA** and the only in-place-read-safe host master is pinned RAM.

## Goal

Replace the pageable mmap host master with a **bounded, pinned host tier** that is filled on demand from
the GGUF with DirectIO (`--load-mode dio`, O_DIRECT), with its own LRU eviction:

```
VRAM arena (L1)  ->  pinned host pool (L2, budgeted)  ->  GGUF on disk (L3)
```

* pinned, so GPU in-place reads and direct H2D are safe (the `ROCm_Host` property, but bounded);
* filled on demand, so the non-swappable RAM is a chosen budget (`MOE_HOST_POOL_MIB`), not the whole
  expert set (18.6 GB for the 35B-A3B Q4_K_M `-ncmoe 40`);
* DIO, so the fill does not pollute the page cache (no second copy) and works on a read-only GGUF.

`--host-experts mmap` and `LLAMA_MMAP_HOST_EXPERTS` are **dropped**: RDNA cannot use them and they are
the only consumer of the pageable host-master path.

## What already exists (reuse, do not rebuild)

* `moe_cache`'s per-expert addressing: `host_bytes`, `src_off`, `host_pitch`, `split_axis`, the device
  cold read and the arena fill already model a strided per-expert slice.
* `llama_file`'s DIO path (`read_aligned_chunk`, `O_DIRECT`, `LLAMA_DIRECT_IO_BUFFER_SIZE` bounce
  buffer) is the fill primitive; the loader already has `(llama_file*, offs)` per tensor.
* The VRAM arena, LFRU admission (`access_locked`), device-remap and policy kernels are unchanged.

## Design sketch

Per `moe_cache` host table, replace `t.host` (a pointer into the full master) with a **pool descriptor**:

```
host_pool_t {                       // one per (process, device) or per table; budgeted by MOE_HOST_POOL_MIB
    char * base;                    // pinned (cudaMallocHost) pool, n_slots * host_bytes
    uint32_t slots;                 // budget / host_bytes
    lru;                            // slot -> (table, expert), clock, protected bit
    ...
}
```

* A cold miss in the host tier does a host-side fill: `pread`/DIO of the expert's slice at
  `(fd, offs + e*stride + src_off, rows, pitch)` into the LRU slot's pinned buffer.  Because the GPU
  reads the slot only after the host fill, the fill is synchronous on the host (it is a CPU read +
  write into RAM; the later H2D/cold read is async on the stream).
* The VRAM arena fill then `cudaMemcpyAsync`s from the pinned slot, exactly as today.
* The device cannot distinguish a pooled slot from the old full-master alias, so the kernel cold read is
  unchanged.

### The hard constraint: no scheduler fallback

Today, when the cache declines, the scheduler uploads the experts from `tensor->data` (the full host
master).  A **bounded** host pool cannot satisfy that, so every `MUL_MAT_ID` on a pooled table must be
served by the cache (or the loader must keep a valid full master, which defeats the budget).  Options:

1. Pool only when the caller can guarantee no fallback: gate the pool behind `MOE_HOST_POOL_MIB` and
   make the scheduler path a hard error (or force a synchronous pool fill + `tensor->data` alias) when
   the cache declines.
2. Keep a small resident "shim" master that aliases the pool for the scheduler's whole-table copy (the
   scheduler would copy garbage for non-resident experts).  Not acceptable.
3. Implement the pool inside the cache only for the **cold read** (L2) while the loader still allocates
   the full master — a pure win only if the master is sparse, which it is not.

>(1) is the intended path; it is also the reason this is a campaign and not a patch.

## Phases

* **Phase 0 (this campaign): drop the pageable path.**  Remove `--host-experts mmap` /
  `LLAMA_MMAP_HOST_EXPERTS` and the loader downgrade, keep the #116 device-accessibility gate as a safety
  rail (all masters are pinned now), and land the `MOE_EXPERT_CACHE_MIB >= 2048` floor.  Delivery
  candidate for block 06.
* **Phase 1: plumbing.**  Export `(llama_file*, offs, host_bytes, src_off, host_pitch, split_axis)` per
  registered host table to ggml-cuda (a small C registry or a `llama_model_loader` callback).  Keep the
  full pinned master as the default; add a `MOE_HOST_POOL_MIB` gate (default 0 = off) that, when set,
  allocates the pool instead.
* **Phase 2: pool + LRU fill.**  Pinned pool, per-expert LRU, DIO fill; cold read and arena fill from the
  pool.  Validate against the full-pinned master (byte-identity of text, MUL_MAT_ID, prefill-logit KLD).
* **Phase 3: no-fallback contract.**  Make the pooled `MUL_MAT_ID` path self-sufficient (always served by
  the cache) and prove the scheduler never reads a non-resident pooled expert.
* **Phase 4: promote.**  Re-validate the combination, write the block-06 amendment, regenerate the set.

## Validation (per phase)

* coherence: same-seed text == the full-pinned-master build (dense 4B golden, qwen35moe greedy);
* `test-backend-ops -o MUL_MAT_ID` 931/931;
* prefill-logit KLD gate (`scripts/gate-prefill-logits.sh`) if a prefill path is touched;
* RSS/pinned accounting: `ROCm_Host` stays within `MOE_HOST_POOL_MIB` (+ arena);
* fault count: 0, on gfx1201 with a forced partial arena and with `-sm layer` / `-sm tensor`.

## Evidence so far (issue #116 session, 2026-10-08)

* `hipHostGetDevicePointer`: `hipHostMalloc` -> success (dev == host); mmap/malloc -> `hipErrorInvalidValue`.
  `hipPointerGetAttributes`: `Host(1)` vs `Unregistered(0)`.
* `XNACK enabled: NO` on gfx1201 (`rocminfo`).
* mmap + `-sm tensor` + partial arena faults in `mul_mat_vec_q_moe` (cold read) and, with the cold path
  gated, in the axis-0 eager fill (`MOE_EXPERT_CACHE_FILL=0` clears it); no kernel name on the copy fault.
* The pinned + `-sm tensor` + partial-arena fault reproduced at 1024 and 2048 MiB/device and is a
  **separate** host over-read (no pageable warning), parked here.
