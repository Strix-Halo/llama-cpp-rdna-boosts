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
