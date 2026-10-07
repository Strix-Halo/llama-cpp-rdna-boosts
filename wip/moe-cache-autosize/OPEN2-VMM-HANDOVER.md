# OPEN 2 — the movable-boundary slab: cold-start handover

**READ THIS FIRST.**  The design is **implemented and both DoD gates pass**.  The dated sections below (from
"2026-10-06" down to "2026-10-06 (final)") are the trail and still carry the now-**superseded**
per-allocation VA-pool / unmapping story — read them as history.  The **current** state is this section plus
"2026-10-07: THE MOVABLE-BOUNDARY SLAB".

---

## Current status (2026-10-07) — start here

The slab is in the working tree behind **`GGML_CUDA_SLAB=1`** (default off while WIP) and fixes both gates.
The `~/llama.cpp` tree is **dirty** at the r21 tip with these changes uncommitted; the full diff is
`open2-ideaB-vmm-compute.diff` in this directory, and the build is `build-rocm-vmmc`.

| gate (`-ub 8192`, cache auto, 16k prompt) | r21 (delivered) | slab |
|---|---|---|
| cli `-n 2000` | decode 78.7 / prefill 1683 / MTP acc 0.9245 | decode **74.9** / prefill **1705** / acc **0.92448** |
| server, drop on, wide1 -> short -> wide2 | **ABORTS** (`cudaMalloc failed` -> `ggml-backend-meta.cpp:1817` assert) | **0 aborts**, all four responses coherent, arena restored to **35254 MiB** |

`////` = 0 everywhere.

### The design (the thing not to lose)

**ONE slab per device**, created at the first compute-buffer alloc:

* `cuMemAddressReserve` + exactly **ONE `cuMemMap`** of the allocatable VRAM (free VRAM minus
  `GGML_CUDA_SLAB_RESERVE_MIB`), rounded down to the boundary chunk.
* Split by a **BOUNDARY**: LOW `[0, boundary)` = the **work pool** (the compute buffer); HIGH
  `[boundary, size)` = the **MoE expert-cache arena**.
* Growing the work pool is a **boundary move inside the already-mapped slab**: the lowest arena chunks are
  reassigned to the work pool and the arena tables living in them are evicted (`moe_cache_evict_slab_range`
  -> `ggml_cuda_slab_arena_free`).  **HIP is never called at runtime** — only at slab creation and shutdown.
* The work region's base VA never moves, so a growing layout **keeps its tensor addresses** (the work buffer
  is a view whose reported size is the boundary).
* The boundary also moves **DOWN**: after the prefill -> decode drop the work layout is narrow (~1.9 GiB), so
  `ggml_cuda_slab_work_release` (from the compute buffer's destructor) plus the next, smaller `work_alloc`
  hand the slack back to the arena.

**Do not re-introduce unmapping.**  ROCm rejects a partial `hipMemUnmap` of a mapping (measured:
`vmmprobe2.cpp`), so a byte-granular "tail prune" is impossible when an allocation is one mapping, and unit
mapping broke an unrelated `hipMemcpy2DAsync` in-tree (`vmmprobe3/4.cpp`).  The slab needs neither.

### The two bugs that had to be fixed to make it work (both OURS, not the design's)

1. **The wholesale-fallback gate had to become PER TABLE.**  `moe_cache_has_arena_locked()` is all-or-nothing,
   but under the slab a **partial** cache is the NORMAL state (evicting the tables in the taken chunks is
   routine).  With the global gate, one boundary move disabled the whole cache -> the decode ran on the
   host-expert path: **8 CPU cores busy / GPUs ~30 % / decode 7-21 t/s**.  Now `moe_cache_take_over`,
   `moe_cache_update_host`, `moe_cache_get_table` and `moe_cache_get_slot` gate on **that table's**
   arena/slots and still agree per table; only the FUSION guard stays global (`moe_cache_has_arena()`).
2. **The arena's allocation unit is FINE (the VMM granularity, 2 MiB), not the 64 MiB boundary chunk.**
   Chunk-aligning each table's slab wasted up to 63 MiB per table (~9 GiB over 288 tables), so the sizing
   (which does not model the rounding) over-committed, most tables failed, and the decode collapsed to the
   host path again.  The slab is ONE mapping, so the sub-slab unit is pure bookkeeping.  `alloc_all_locked`
   holds back `n_tables * unit` and sizes against `ggml_cuda_slab_arena_total()` — NOT `cudaMemGetInfo`,
   which reads only the little left outside the slab.

### Environment

| var | default | meaning |
|---|---|---|
| `GGML_CUDA_SLAB` | `0` | **the gate**; `1` enables the slab (HIP-only) |
| `GGML_CUDA_SLAB_CHUNK_MIB` | `64` | the boundary-move chunk |
| `GGML_CUDA_SLAB_RESERVE_MIB` | `8192` | left OUTSIDE the slab — the GPU weights and the KV cache are allocated *after* the first compute buffer in this fork |
| `GGML_COMPUTE_BUFFER_CHUNK_MIB` | `256` | workspace allocation = `(ceil(need/C)+1)*C` (whole chunks + one spare; absorbed growth, rare re-alloc, the spare also guards over-reads). `0` -> `GGML_COMPUTE_BUFFER_MARGIN_PCT` (10) |
| `GGML_CUDA_COMPUTE_VMM` | `0` | the earlier per-allocation VMM pool (secondary; the slab replaces it) |
| `MOE_EXPERT_CACHE_VALIDATE` | `0` | `1` structural validator, `2` + arena-head finiteness (debug; no-op when unset) |
| `MOE_EXPERT_CACHE_YIELD_WHOLESALE` | `0` | A/B: release the whole arena instead of per-table on a VMM-pool yield |
| `MOE_EXPERT_CACHE_MIB` / `_RESERVE_MIB` | pre-existing | explicit arena budget / the reserve held out of the auto budget |

### Files changed (vs the r21 tip; full diff = `open2-ideaB-vmm-compute.diff`, ~1360 lines)

* `ggml/src/ggml-cuda/ggml-cuda.cu` — the slab (`ggml_cuda_slab_*`), the compute-buffer view, the
  `slab_view` flag/destructor, the older VMM pool, `get_compute_chunk_bytes`.
* `ggml/src/ggml-cuda/ggml-cuda-vmm.h` — the slab + VMM API.
* `ggml/src/ggml-cuda/moe-expert-cache.cu/.h` — the per-table gates, `moe_cache_evict_slab_range`, the
  slab-backed arena (`alloc_table_locked`), `moe_cache_rearm`, `moe_cache_prune` (written, parked),
  `moe_cache_validate`, `table_arena_bytes`/`free_arena_backing`.
* `ggml/src/ggml-alloc.c` — the chunk-quantized work allocation (`ggml_vbuffer_chunk_alloc_size`).
* `ggml/src/ggml-backend-impl.h`, `ggml-backend.cpp`, `ggml-backend-meta.cpp` — the appended buft capabilities
  `alloc_buffer_usage` and `get_compute_chunk_bytes` (threaded through the Meta buft), and the re-arm
  device-iface hook.
* `ggml/include/ggml-backend.h`, `src/llama-context.cpp` — `ggml_backend_dev_moe_cache_rearm`, called at the
  prefill -> decode drop site after the narrow layout is re-reserved.

### Repro / build

```bash
# build (the working dir; gate the slab at RUNTIME, not compile time)
cd ~/llama.cpp
cmake --build build-rocm-vmmc --target llama-cli llama-server -j 16

# cli DoD (note GGML_CUDA_SLAB=1)
BIN=./build-rocm-vmmc/bin/llama-cli wip/moe-cache-autosize/open2-harness-cli.sh slabdod 2000 GGML_CUDA_SLAB=1

# server safety (wide1 -> short -> wide2)
BIN=./build-rocm-vmmc/bin/llama-server wip/moe-cache-autosize/open2-harness-server.sh sl2 \
    GGML_CUDA_SLAB=1 LLAMA_DROP_COMPUTE_BUFFERS=1
```

The harnesses are copied into this directory; `BIN=` selects the build, `LOGDIR=` the log dir (default
`/tmp/item1`).  Server mode knobs: `SRV_SEQ=1` (sequential), `SRV_UB=<n>`, `SRV_NPRED=<n>`; do NOT pass `-np`
without `--kv-unified` (the 16k prompt gets a 400).  `llama-cli` always `--single-turn`; never run two
benchmarks at once.

### Remaining work ("polishing")

1. **Eviction policy.**  `moe_cache_evict_slab_range` currently evicts whatever tables sit in the taken
   chunks (address order).  Better: use the cache's own hotness (`count`/`ghost`/`last`) to choose, or
   allocate arena tables from the HIGH end downward so the boundary always takes the coldest.
2. **Sweep** `GGML_CUDA_SLAB_CHUNK_MIB` / `GGML_CUDA_SLAB_RESERVE_MIB` (the reserve bounds the arena; the
   weights+KV need it) and `GGML_COMPUTE_BUFFER_CHUNK_MIB`.
3. **Wider gates:** MTP `-n 3000` + `--reasoning on/off` (rule 0 of `mtp-adaptive-methodology.md`),
   3-GPU coherence, `-ncmoe 0` byte-identity, the `llama-batched-bench -npl 1,4,8` purity gate.
4. **Default-on decision + `ENVIRONMENT.md`** (the default-on policy wants beneficial, gated features on).
5. **Promotion decision (maintainer).**  Two correctness fixes are candidates for the delivery *now*, both
   only reachable through the slab today:
   * the per-table fallback gate (the old global gate + a partial cache corrupted the output — measured
     `////` and MTP acc 0.009);
   * `moe_cache_build_remap_kernel`'s `slot < n_res` clamp (a stale device slot map after an eviction).
   Per the WIP rule the maintainer decides; do not fold them into `patches/` unasked.
6. **Parked/dead code to remove or re-justify** if the slab becomes the only path: the older VMM pool
   (`GGML_CUDA_COMPUTE_VMM`), `ggml_cuda_vmm_shrink` (no-op), `moe_cache_prune` (never called).

---

## Historical trail (superseded — do not follow as a plan)

# OPEN 2 — VMM / movable-split slab: cold-start handover (2026-10-06)

**Read this first in a new session.**  It supersedes the `FOLLOWUP-compute-arena-chunking.md` plan for
Idea B.  The design is settled in outline; the next step is a **controlled probe**, then the pool.

## Where we are

* **r21 is shipped and pushed** (`v16-a55e952b8-r21`, fork tip `94c3eeb89`) — the OPEN 1 safety subset
  (arena `t.slots` fix, per-layer re-size, `MTP_DRAFT_N_UBATCH=512`, the wide-prefill drop default-on for
  `llama-cli` only).  Delivery repo `main` at `accd84a`; `validate-set.sh` green.
* The OPEN 1 cli DoD is met by default (`-ub 8192` cache auto 16k: decode **78.7** / prefill **1683**).
  The **`llama-server` half is open**.
* The `~/llama.cpp` working tree is **clean at the r21 tip** (the Idea A change was reverted; its diff is
  `open2-ideaA-chunking.diff`).

## What has been measured (so you do not redo it)

1. **The wide layout is a contiguity problem.**  A later wide prefill on a server needs one contiguous
   `cudaMalloc` back; freeing *all* 288 arena tables (40 GB) does not yield it.  `-ub 8192` and `-ub
   6144` abort; `-ub <= 5120` survives at n=16, but `-ub 4608` at `-n 2000` still faulted
   (`mul_mat_vec_q_moe`, the r20 §13 class) — so `-ub` only moves the cliff.
2. **Idea A (plain compute chunking) helps but is not enough.**  `GGML_COMPUTE_BUFFER_CHUNK_MIB` chunked
   the compute buffer -> the server **survived** a later wide prefill (0 aborts, partial stand-downs) and
   the cli still met the DoD (75.7), but the wide prefill still consumes the arena (65 % -> 26 %) and the
   one-shot sizing **never re-grows it**.  Detail: `OPEN1-FINDINGS.md` (last section).
3. **The `-ub` crossover is state-dependent** (it moved between runs as free VRAM drifted).

## The VMM probe — status: INCONCLUSIVE, needs a controlled re-run

Built `build-rocm-vmm` with `GGML_HIP_NO_VMM=OFF` (full command in §Environment).  The existing
`ggml_cuda_pool_vmm` (a bump allocator over a 32 GB VA reservation) then backs the **workspace** pool.
Results:

| run | result |
|---|---|
| cli `-ub 8192` cache auto, `GGML_COMPUTE_BUFFER_MARGIN_PCT=0`, VMM build, n=16 | exit 0, coherent |
| same, **non-VMM** r21 control, minutes later | **exit 0, coherent too** |
| server, drop ON + margin=0, VMM build, wide1->short->wide2 | **abort** (`11763 MiB cudaMalloc failed`, 288 stand-downs) |

So this probe **does not yet prove VMM does anything**: the margin=0 abort is free-VRAM/fragmentation
dependent, and both builds passed once the box had more free VRAM.  Two facts are nevertheless solid:

* ROCm 7.14's VMM primitives **work** here — the VMM build runs, and the pool's teardown does
  `cuMemUnmap` per mapping + `cuMemAddressFree` with no error.
* Enabling VMM for the **workspace pool only** cannot fix the server, because the failing object is the
  **compute-buffer `cudaMalloc`**, which is not VMM-backed (`ggml_backend_cuda_buffer_type_alloc_buffer`
  -> `ggml_cuda_device_malloc` -> `cudaMalloc`).

**Controlled probe to run first:** pin the box state (fresh boot, or record free VRAM before each run),
then A/B the cli margin=0 repro N=5 on `build-rocm-r16` vs `build-rocm-vmm`, and instrument *which*
allocator fails (`ggml_cuda_device_malloc` vs the pool vs the Q8_1 arena).  Also report the device VMM
capability line (`ggml_cuda_info().devices[d].vmm`, printed at load) to confirm VMM is actually on.

## The design (settled in outline)

**One per-device VMM reservation, split dynamically between the compute buffer and the MoE arena** — the
user's "massive contiguous slab, move the split point up/down":

* Reserve a large VA range per device (`cuMemAddressReserve`); map physical in fixed units
  (`cuMemCreate` + `cuMemMap`) as either side needs them.  Growing the compute buffer = mapping more
  units at its end (its VA is contiguous, so a realloc is a pointer move, not a copy).  Evicting arena
  entries = unmap their units and map them under the compute side.
* **Why VMM is mandatory:** plain `cudaMalloc` cannot split/extend/merge ranges, and the arena is
  stride-addressed (`(char *)t.arena + slot*t.expert_bytes` — 8 call sites) so a table's slab must be
  **one contiguous VA run**.  Units are fungible only if they can be remapped under stable VA.
* **Asymmetry to respect:** the compute buffer is already chunk-friendly (per-tensor chunks, any sizes,
  no straddling), while the arena is not.  So arena->compute is easy even without VMM (hand a whole
  table block over as a chunk); compute->arena is the direction that needs VMM.
* **Must not break:** decode CUDA-graph capture (addresses must stay valid -> stable VA); the arena
  head-pad / head-zero invariants (keyed on `(buffer, expert_bytes)`); the fused-op purity anchors; and
  no arena unit may be remapped while a kernel that carries its address is in flight
  (`moe_cache_sync_devices_locked`; the fused kernels take the arena address as a launch parameter).
* **Per device + peer access:** `cuMemSetAccess` for the P2P/AR/RCCL paths (`-sm tensor`, NCCL build);
  the existing pool already has this logic, mirror it.
* **The existing `ggml_cuda_pool_vmm` is not reusable as-is:** its `free` is LIFO and never unmaps, and
  it is a workspace bump allocator.  A new unit pool (free list, non-LIFO, real unmap, owner shared by
  the compute buffer and the arena) is the deliverable.

## ROCm

Use **7.14.1** (`/opt/rocm-7.14.1-gfx120X`) — its VMM works.  **ROCm 10.1.0** is installed at
`/opt/rocm-10.1.0-gfx120X/` purely as a fallback if 7.14 VMM turns out to be unreliable under the real
pool (map/unmap churn, not just teardown).  The maintainer flagged it on 2026-10-06; do not switch
speculatively.

## Environment / reproduction

```bash
# r21 build (current default; the promoted tree)
cd ~/llama.cpp
BUILD_DIR=build-rocm-r16 EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
cmake --build build-rocm-r16 --target llama-cli llama-server -j 16

# VMM build (GGML_USE_VMM on; workspace pool becomes ggml_cuda_pool_vmm)
BUILD_DIR=build-rocm-vmm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS= -DGGML_HIP_NO_VMM=OFF" ~/bin/build-llama-rocm-714

# harnesses (copied here; BIN= selects the build, LOGDIR= the log dir)
BIN=./build-rocm-vmm/bin/llama-cli    ./wip/moe-cache-autosize/open2-harness-cli.sh <tag> 16 [ENV=...]
BIN=./build-rocm-vmm/bin/llama-server ./wip/moe-cache-autosize/open2-harness-server.sh <tag> [ENV=...]
# server mode knobs: SRV_SEQ=1 (sequential), SRV_UB=<n>, SRV_NPRED=<n>; server default is 4 unified slots
# (do NOT pass -np without --kv-unified or the 16k prompt is rejected with 400).
```

The harnesses assume `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/...` + `/tmp/pl_16k.txt` (16k prompt) and
`HIP_VISIBLE_DEVICES=0,1`, `-sm tensor -ncmoe 48`.  `llama-cli` always `--single-turn`; never run two
benchmarks at once.

## Next steps (in order)

1. Controlled VMM probe (above) + identify the failing allocator.
2. Implement the **unit pool** (`cuMemCreate/Map/Unmap/AddressReserve/SetAccess`, per device, non-LIFO,
   free list, `GGML_COMPUTE_BUFFER_VMM` style env gate).
3. Back the **compute buffer** with it first (a VMM buffer type or a mode of the CUDA buft); confirm the
   wide prefill's realloc is a pointer move and the server survives.
4. Put the **arena** on it (tables as VA ranges of units) so units can flow back; then the arena can
   re-grow after a wide prefill and the server DoD should fall out.
5. Gates: cli DoD, server concurrent long+short + later wide prefill, 3-GPU coherence, `-ncmoe 0`
   byte-identity, MTP `-n 3000`.

---

## 2026-10-06 (late): VMM prototype — findings (supersedes the "inconclusive" probe above)

**The VMM works.**  A standalone probe (`/tmp/item1/vmmprobe.cpp`) on ROCm **7.14.1 / gfx1201**:
`hipDeviceAttributeVirtualMemoryManagementSupported` = 1, granularity 2 MiB, `hipMemCreate` -> `hipMemMap`
-> `hipMemSetAccess` -> device memset + host readback OK, and — the operation the movable split
actually needs — **`hipMemUnmap` then `hipMemMap` at the SAME VA works and reads back correctly**.  So
**ROCm 10.1.0 is not needed** (keep it only as a fallback).

**Bug 1 (real, in the delivery):** the VMM capability detection in `ggml_cuda_init` was inside
`#if defined(GGML_USE_VMM)`.  With the default `GGML_HIP_NO_VMM=ON`, `GGML_USE_VMM` is undefined, so
`devices[].vmm` was **always 0** — the existing `ggml_cuda_pool_vmm` never activated on this box, which
is why `GGML_HIP_NO_VMM=OFF` changed nothing and why the whole earlier "VMM probe" was meaningless.  Fix:
detect under `defined(GGML_USE_HIP)` too; `new_pool_for_device` keeps its own `GGML_USE_VMM` guard so the
workspace pool's behaviour is unchanged.

**Prototype (diff: `open2-ideaB-vmm-compute.diff`, 5 files, ~250 lines):**
* appended optional `alloc_buffer_usage(buft, size, usage)` to `ggml_backend_buffer_type_i` (defaults to
  `alloc_buffer`; only the graph allocator calls it, with the real usage) and threaded it through
  `ggml_vbuffer_alloc` and the **meta** buft (the graph buft under `-sm tensor` is `Meta()`);
* a per-device VMM pool in `ggml-cuda.cu` (64 GiB VA reservation, first-fit free list, coalescing,
  `cuMemCreate`/`Map`/`Unmap`/`SetAccess` per allocation), env `GGML_CUDA_COMPUTE_VMM` (HIP-only, 0/unset
  = off); the COMPUTE buffer uses it, model weights stay on `cudaMalloc`;
* on `cuMemCreate` OOM it yields the MoE arena (`moe_cache_shrink_step`, then `moe_cache_release_arena`)
  and retries — VMM needs **no contiguous block**, so a partial yield is enough.
* **Bug 2 (prototype):** the coalescing code reused `next` after `erase(next)` (dangling iterator ->
  SIGSEGV in `std::prev`); `erase` now feeds the previous-neighbour lookup.

**Results** (server/k/cli on 2x gfx1201, `-ub 8192`, cache auto, 16k prompt):

| run | result |
|---|---|
| cli, VMM on | exit 0, **coherent AND byte-identical generated text to r21**, arena 70.6 % |
| server, VMM on OR off, **drop off** | coherent (`<think>...`, no `////`) |
| server, **VMM off + drop ON** | **still aborts** (`GGML_ASSERT(bufs.back() != nullptr)`, meta:1817) |
| server, **VMM on + drop ON** | **survives** the fatal wide1->short->wide2 (0 abort signals) but output is garbage (`////`) |
| server, **VMM on + drop ON + cache DISABLED** | **coherent** (isolates the re-alloc from the culling) |

**Conclusion:** the VMM compute-buffer allocation and its free/re-alloc across requests are **correct**.
The garbage is the **arena culling**, not VMM: with the drop on, the first wide prefill stands down
~184-288 tables (`stand_down_table_locked` -> `cudaFree`), and that cull is what corrupts the run.  This
is the latent bug the abort used to hide.

**Next (the design's step 4):** put the **arena** on the same pool so a cull is an `Unmap` (VA stable,
physical returned) instead of a `cudaFree`, and a re-arm is a re-`Map` at the same VA.  Stable arena VA is
also what keeps captured decode graphs and the cache's fusion guard / `moe_cache_take_over` agreement
valid across a split move.  Then re-run the server drop-on gate.

---

## 2026-10-06 (later): ROOT CAUSE FOUND + FIXED — the wholesale-fallback invariant had a third consumer

**The corruption is a real bug in the delivered cache logic, independent of VMM** — VMM is only what
made the failing path reachable (before it, the same allocation always aborted).

`AGENTS.md` states: *"a partially-failed cache falls back wholesale (the fusion guard and
`moe_cache_take_over` must agree)"*.  Two consumers honoured the gate (`ggml_cuda_cache_blocks_fusion`
and `moe_cache_take_over` both check `moe_cache_has_arena_locked()`); a **third** did not:

* `moe_cache_get_table()` kept returning the arena for a **surviving** table, and
* the fill path in `moe_cache_update_host()` kept building that table's remap / filling its arena.

So after a partial stand-down the take-over hook had declined (the scheduler copied `input_cpy`), yet the
op still redirected to an arena/remap the fallback path had stopped maintaining -> the repeated-`/`
corruption.  **Fix:** gate both on `moe_cache_has_arena_locked()`.  In `moe_cache_update_host` the gate must
sit **after** the priming/sizing bookkeeping (placed before it, `alloc_all_locked` never latches and the
cache silently never sizes — measured: no `alloc_all_locked` line, `stood_down=0`).

**How it was found:** the `MOE_EXPERT_CACHE_VALIDATE` validator (new, this campaign) showed the cache
*structurally consistent with every arena head finite* after the cull — ruling out the arena and pointing
at the consumers.  A wholesale-vs-partial A/B (release the whole arena vs stand down single tables) then
isolated it: wholesale = coherent, partial = garbage.

**Also changed:** a compute-buffer yield now **releases the whole arena** by default
(`moe_cache_release_arena`).  A partial stand-down is now correct, but the surviving tables are bypassed
until the run ends, so their VRAM would be wasted — and VMM needs only *physical*, not a contiguous
block, so the whole-arena release frees it in one go.  `MOE_EXPERT_CACHE_YIELD_PARTIAL=1` restores the
partial path for A/B.  New debug knob: `MOE_EXPERT_CACHE_VALIDATE=1` (structural) / `=2` (+ arena
finiteness).

**Results (final build):**

| run | result |
|---|---|
| cli, VMM on, `-n 2000` (the DoD) | prefill **1674 t/s**, decode **78.5 t/s**, MTP acc **0.9245**, `////`=0, arena 40078 (70.6 %) |
| server, drop on + VMM on, cache AUTO | 0 aborts, coherent |
| server, drop on + VMM on, **forced cull** (`MOE_EXPERT_CACHE_MIB=20000`, stood_down=181) | 0 aborts, **coherent** (`<think>...`) |
| server, drop on + VMM on + cache DISABLED | coherent |
| server, VMM off + drop on | still aborts (meta:1817) — VMM is what makes the path reachable |

**Remaining for the server DoD (the design's split-move):** a cull now *works* but leaves the cache off for
the rest of the run, so a later decode is slow.  The **re-arm** is the missing piece: when the compute
buffer is dropped (context idle) re-size the arena to the auto target and re-enable the cache
(`g_sized` is a one-shot today).  With the arena on the pool that re-arm is a re-`Map` at stable VA, which
is also what keeps captured decode graphs valid.  Then sweep the split: wide prefill takes units, idle
returns them, decode gets the cache back.

---

## 2026-10-06 (latest): the unified VMM allocator — built, split_move works, second cycle still bad

**Built (this is the design the maintainer asked for):** one per-device VMM allocator with VA and physical
SEPARATED (`ggml-cuda-vmm.h` + `ggml_cuda_vmm_{reserve,release,map,unmap,alloc,free,owns}` in ggml-cuda.cu).
**Both** the compute buffer and the MoE arena now come from it:

* a table **reserves** its slab VA once and keeps it for life (stride addressing + captured graphs); a
  cull only **unmaps** the physical and a re-arm **maps** it back at the SAME VA;
* `alloc_table_locked`/`stand_down_table_locked`/`free_table_buffers_locked`/`release_arena` all route to
  the pool when `GGML_CUDA_COMPUTE_VMM=1` (legacy `cudaMalloc` otherwise), tracked by `arena_reserved`;
* the pool mutex protects only the VA free list -- `map`/`unmap` take no lock because the yield
  (`moe_cache_shrink_step` -> `unmap`) must not run under it (deadlock otherwise).

**This fixed the allocator-separation problem:** the cull is now an `Unmap` whose units return to the SAME
pool the workspace draws from, so a growth needs no contiguous block and no `cudaFree`/`cudaMalloc` churn.
Forced cull `MIB=20000`: `stood_down=131` (was 288 + abort), **0 aborts**, and the re-arm restored the
arena to 38580.8 MiB with the hit rate going 0.72 -> 0.96 (residents preserved -- no second arena, no copy).

**Still broken: the SECOND cull/re-arm cycle.** wide1 (cycle 1) is coherent; wide2 (cycle 2) is `////`
(a variant faults in `mul_mat_vec_q_moe` on an UNMAPPED arena page).  Sequential requests reproduce it, so
it is not a race.  `srv-vp` (arena on the pool, re-arm not yet effective) was fully coherent, so the trigger
is the re-arm mapping the arena again.  Guards now on all four consumers (`moe_cache_get_table`,
`moe_cache_update_host`, `moe_cache_get_slot`, plus `take_over`), and `stand_down` now clears the
DEVICE-side maps (`slot_dev`, `used_dev`) -- so the cause is NOT an unguarded consumer in the host
decision path.  Leading suspects, in order:

1. **a decision baked before the cull** -- the fused gate+up/mmvq kernel reads the arena from its launch
   parameters (`moe_cache_redirect_fused`), and a CUDA graph captured while the arena was mapped may be
   replayed after a cull/unmap with a stale fusion decision.  `ggml_cuda_cache_blocks_fusion` is evaluated at
   graph build, not at replay.  (Needs a way to disable HIP graph capture to confirm; `GGML_CUDA_GRAPH_OPT`
   is an unrelated switch.)
2. the **device-remap** path (`g_devmap_armed` stays armed across a cull; the re-armed tables'
   `slot_dev` is cleared but the survivors' is not, and the eager re-arm pass is not re-run).
3. the re-armed tables are fresh-physical + `cudaMemset`-zeroed, so a reader that believes they are
   resident reads zeros -- exactly the `////` pattern.

Next: instrument the path decision for the first decode token of each cycle (arena vs `input_cpy`), or find
and use a real "disable HIP graph capture" switch, to separate (1) from (2)/(3).  The fix likely needs the
cache to invalidate the scheduler's captured graphs / force a fusion re-evaluation on a cull+re-arm.

---

## 2026-10-06 (final this session): chunk-quantized workspace DONE; chunk PRUNE blocked by ROCm

The maintainer's design -- discrete uniform chunks that absorb growth ("minimum chunks + one") and a
chunk-level PRUNE instead of a cull/re-arm -- was built.  Status:

**DONE and working: chunk-quantized workspace allocation.**  A new appended buft capability
`get_compute_chunk_bytes` (CUDA: `GGML_COMPUTE_BUFFER_CHUNK_MIB`, MiB, default **256**, HIP-only; meta buft
aggregates it) makes `ggml_vbuffer_chunk_alloc_size` return `(ceil(need/C) + 1) * C` for the COMPUTE
buffer: whole chunks plus ONE spare.  The realloc trigger compares the *allocated* size, so growth inside
the spare is free, the re-alloc only fires past the next high-water mark, and the spare is the mapped
over-read guard at the buffer's end.  Measured: the decode/verify compute buffer went 1870.8 MiB ->
**2048.0 MiB** (8 x 256 MiB, exact), cli coherent, no regression.  `GGML_COMPUTE_BUFFER_CHUNK_MIB=0` falls
back to `GGML_COMPUTE_BUFFER_MARGIN_PCT`.

**ROCm constraint found (probes committed as `vmmprobe2/3/4.cpp`):**
* `hipMemUnmap` of a **sub-range** of a mapping fails (`hipErrorInvalidValue`) -- so a table's tail can only
  be given back if every granularity unit is its OWN mapping (`vmmprobe2`).
* Unit mapping itself works, including unmap + re-map at the same VA, and at full scale (6144 units / 12 GiB,
  no error) (`vmmprobe3`, `vmmprobe4`).

**BLOCKED: the chunk prune.**  The prune was implemented (`moe_cache_prune`, `ggml_cuda_vmm_shrink`, plus a
`slot < n_res` clamp in `moe_cache_build_remap_kernel` that makes a stale device slot map safe after a
prune).  But when the pool maps the 11776 MiB compute buffer as ~5900 unit mappings, an unrelated
`hipMemcpy2DAsync` (cpy.cu:479) starts failing with `hipErrorInvalidValue` at context init -- even though the
standalone 6144-unit map succeeds.  The single-mapping pool does not show it.  So the unit mapping is
**parked**: `ggml_cuda_vmm_map_phys`/`unmap` are back to one mapping, `ggml_cuda_vmm_shrink` is a no-op, and
`moe_cache_prune` is implemented but not called (the reclaim is still `moe_cache_shrink_step`).  The tree is
back to a working state (cli coherent, chunk quantization live).

**Next step to unblock:** find why a unit-mapped VMM range breaks `hipMemcpy2DAsync` in-tree when it works
standalone -- prime suspects: (a) HIP's 2-D copy needs a pitch/limit check that VMM ranges trip only when the
mapping count is high; (b) the copy's source/dest is a *view* whose pitch is fine but whose range the driver
cannot resolve; (c) it is not the mapping count at all but the chunk-quantized SIZE interacting with a
`ggml_cuda_cpy_as_memcpy_2d` path.  A minimal repro is to run the cli with the unit mapping and
`GGML_COMPUTE_BUFFER_CHUNK_MIB=0` (isolates chunk size from mapping count).  Once that is understood, the
prune is a small re-enable (the code is written).

---

## 2026-10-07: THE MOVABLE-BOUNDARY SLAB -- built, and BOTH gates pass

Superseded the whole unmapping story.  The maintainer's design is implemented and works.

**The design (as specified):** ONE slab per device, `cuMemAddressReserve`d and **mapped exactly once**, then
split by a BOUNDARY: the LOW region `[0, boundary)` is the work pool (the compute buffer) and the HIGH region
`[boundary, size)` is the MoE expert-cache arena.  Growing the work pool is a **boundary move inside the
already-mapped slab** -- the lowest arena chunks are reassigned to the work pool (evicting the arena tables
that live there) -- so **HIP is never called at runtime**: only at slab creation and at shutdown.  Because the
work region's base VA never moves, a growing layout keeps its tensor addresses.  This removes every wall the
per-allocation VA pool kept hitting: no partial `hipMemUnmap` (ROCm rejects it), no unit mapping, no
`cuMemCreate`/`cuMemMap` churn at runtime, and no "the HIP allocator won't give the physical back".

**The two bugs that had to be fixed to make it work (both were mine, not the design's):**

1. **The wholesale-fallback gate had to become PER TABLE.**  `moe_cache_has_arena_locked()` is all-or-nothing,
   and the slab makes a PARTIAL cache the NORMAL state (evicting the tables in the taken chunks is routine).
   With the global gate, one boundary move disabled the whole cache -> the decode ran on the host path
   (8 CPU cores busy, GPUs 30 %, decode 7-21 t/s).  Now `moe_cache_take_over`, `moe_cache_update_host`,
   `moe_cache_get_table` and `moe_cache_get_slot` gate on **this table's** arena/slots and still agree per
   table; only the FUSION guard stays global (`moe_cache_has_arena()`), so a partial cache loses the fusions
   but keeps serving its residents.
2. **The arena's allocation unit had to be FINE (the VMM granularity), not the 64 MiB boundary chunk.**
   Chunk-aligning each table's slab wasted up to 63 MiB per table (~9 GiB over 288 tables), so the sizing
   (which does not model the rounding) over-committed and most tables failed to allocate -> the same host-path
   collapse.  The slab is ONE mapping, so a sub-slab allocation unit is pure bookkeeping and can be 2 MiB.
   `alloc_all_locked` also holds back `n_tables * unit` for the rounding.

**Also:** `ggml_cuda_slab_work_release` (called from the compute buffer's destructor) lets the boundary move
**down** when the next work need is smaller (the narrow post-prefill layout), handing the slack back to the
arena; `alloc_all_locked` sizes against the slab's arena region (`ggml_cuda_slab_arena_total`), not
`cudaMemGetInfo` (which reads only the small amount left outside the slab); the slab leaves
`GGML_CUDA_SLAB_RESERVE_MIB` (default 8192) outside itself, because the GPU weights and the KV cache are
allocated AFTER the first compute buffer in this fork.

**Results (gate `GGML_CUDA_SLAB=1`, default off):**

| run | result |
|---|---|
| cli `-n 2000` (the DoD) | prefill **1705 t/s**, decode **74.9 t/s**, MTP acc **0.92448** (= r21 exactly), hit 0.9555, `////`=0 |
| server, drop on, wide1->short->wide2 | **0 aborts**, ALL FOUR responses coherent, arena restored to 35254 MiB |
| server boundary move | `evicted 10028+9994 MiB of arena tables from the taken slab chunks`, arena re-grew after the drop |

So the previously-fatal wide2 case is fixed, and the cli DoD holds.  The r21 DoD numbers were
prefill 1683 / decode 78.7; the slab gives 1705 / 74.9 with the same acceptance -- within noise, and with no
cull/re-arm and no runtime HIP calls.

**Next:** (a) sweep the chunk size / reserve; (b) decide the eviction policy (evicting by address is
arbitrary -- the cache's own hotness data could choose which chunks to give up, or the arena could allocate
tables from the HIGH end so the boundary takes the coldest); (c) MTP `-n 3000`, 3-GPU, `-ncmoe 0`
byte-identity; (d) default-on decision + ENVIRONMENT.md.

---

# Why this exists — the arena is a cache, the slab is what makes it affordable

*(Written to be read on its own; it is the "why" behind the "how" above.  Useful as the opening of a
write-up, an upstream framing, or the release notes for the feature.)*

## The problem upstream's `-ncmoe` has

`-ncmoe` keeps MoE expert weights on the host and brings the ones a token needs to the GPU for that
matmul.  It is a **transport**: it holds no state between tokens, so every ubatch pays the transfer again,
and its only lever is pruning to the experts the router actually picked this token.  The experts are the
bulk of the model in a modern MoE, so the transfer is the bulk of the per-token PCIe traffic — and the
GPU spends the decode stalled behind it (measured here: 8 CPU cores saturated doing the host-side gather
and staging while the GPUs sit at ~30 %).

## What the arena changes

The MoE expert-cache arena keeps the hot experts **resident on the GPU** and fetches only the misses.  It
is a **cache**, not a transport, and the difference is persistence: residency survives across tokens *and
across requests*, so the transfer is amortised rather than repeated.  The number that matters becomes the
**hit rate** — the same quantity this campaign has been grading on, measured at 0.9656 in the cli gate
(and 0.43 in the run where a bug starved the arena, which is precisely what made the GPUs idle).

Everything else in the arena is in service of that number: the slot remap and the resident/cold split, the
fusion guard, the admission policy, the identity fast path, the deferred promotion.  `GREEDY-PURITY.md`
§§19/21/24/25/27 exist because those are the invariants that keep a *cache* byte-identical to the
cache-less oracle — a transport never needed them.

## Why the cache needs the slab

A cache is only worth having when it is **large and long-lived**.  Those are exactly the two properties
that collide with a graph allocator that must also be large and growable: the compute buffer is sized from
a measure graph and can need more at runtime, the arena wants to own everything else, and both are asking
the same physical VRAM for a contiguous block.  That collision is the whole bug:

* on ROCm the driver does not coalesce: freeing the **entire** arena (288 tables, tens of GiB) still could
  not satisfy one 11.7 GiB `cudaMalloc` for a grown compute layout — the free space existed, never as one
  block.  Yielding the arena therefore did not rescue the allocation, and the run aborted.
* the obvious workaround — free more, retry, and re-fill cold afterwards — throws away exactly the
  residency the cache exists to provide, so it defeats the feature.

The **movable-boundary slab** dissolves the collision instead of arbitrating it: one VA reservation per
device, mapped once, split by a scalar boundary into a work region and an arena region.  Growing the work
region is a **boundary move inside memory that is already mapped** — the lowest arena chunks change owner
and the tables living there are evicted — so there is no second allocation, no contiguity requirement, no
fragmentation between the two consumers, and no driver call at runtime at all.  Because the work region's
base never moves, a growing layout keeps its tensor addresses too.

So the two halves are not independent improvements:

* **the arena supplies the value** — amortised expert residency instead of per-token transport;
* **the slab supplies the coexistence** — that residency and the compute layout can both be large, without
  either being able to starve the other, on a driver that would otherwise fragment them apart.

Either one alone only moves the problem.

## How to frame it upstream

**"A GPU-side cache for host-resident MoE experts, enabled by a movable-boundary VA slab."**  The cache is
the product; the slab is the mechanism.

* It is **additive and opt-in**: with the feature off, `-ncmoe` behaves exactly as it does today, and the
  slab falls back to the existing allocation path wherever VMM is unavailable (the `devices[].vmm`
  capability gate already exists for the workspace pool).
* The **small, generalisable part is small**: the slab is a couple of hundred lines of VA bookkeeping —
  reserve once, split by a boundary, move it, free at shutdown.  The heavy, fork-specific engineering is
  the arena (the purity invariants above), and it should be presented as such rather than lumped in.
* The honest headline is *"this fixes a problem upstream has not hit yet"*: upstream's compute buffer is
  reserved once and grows at most once, and its expert path is stateless, so neither consumer ever parks
  the free VRAM.  The slab becomes necessary the moment an expert cache does.
