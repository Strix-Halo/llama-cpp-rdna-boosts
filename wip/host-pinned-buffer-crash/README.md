# Host-resident expert load: GPU page fault under `--load-mode none` (2 GPUs)

**Status: PROMOTED to the delivery in `v16-a55e952b8-r15` (2026-10-06) as part of blocks 06 + 13.**
The `-sm tensor` CPU-fallback and the inert/slow expert cache are fixed (see below); the
`--load-mode none` page fault itself is **still open** (TODO #38).

## Update 2026-10-06 (r15 session) -- the fix chain, and the resolution

Implemented the plan below as a WIP patch (`tensor-host-buft-sched-offload.patch`, this dir) and measured.
It is a **two-part, coupled** fix that makes the MoE expert cache engage under `-sm tensor`, but the
`-sm tensor` cache is currently a **net loss**, so nothing here should ship yet.

### Both changes are required

1. **Loader host buffer type** (`src/llama-model-loader.cpp`).  Under `-sm tensor` the repeating layer's
device is the **Meta** device, and the Meta device's `get_host_buffer_type` returns **null** when its
simple devices have *different* host buffer types.  Ours differ, because the F1-F4 fix made `ROCm_Host`
**per device** (`ggml_backend_cuda_host_buffer_type_dev`).  So the `-ncmoe` override fell through to the
pageable `CPU_REPACK` (`.is_host == nullptr`).  The patch prefers a real device's pinned host buft
(`LLAMA_TENSOR_HOST_BUFT`, default on in the WIP build).
2. **Scheduler Meta containment** (`ggml/src/ggml-backend.cpp`, the host-weight offload loop inside
`ggml_backend_sched_backend_id_from_cur`).  The F1-F4 device pin
(`if (src_buft_dev != dev) continue;`) skips the **Meta** backend under `-sm tensor`, so the host expert
op fell back to the **CPU**.  The patch accepts a Meta device that contains `src_buft_dev`
(`ggml_backend_meta_dev_n_devs` / `ggml_backend_meta_dev_simple_dev`).

Evidence (2 GPU IQ4_NL, `-ncmoe 48`, `-lv 5`, `GGML_SCHED_DEBUG=2`):

| | before | after (1+2) |
|---|---|---|
| `-sm tensor` expert buffer | `CPU_REPACK` (pageable, not `is_host`) | `ROCm_Host` (pinned, `is_host`) |
| `-sm tensor` `MUL_MAT_ID` backend | `CPU` | `Meta(ROCm0,ROCm1)` |
| `-sm tensor` cache | inert (h=0, arena 0) | engages (h=0.978, arena 45879 MiB, 70.8 %) |
| `-sm layer` expert buffer / backend | `ROCm_Host` / `ROCm0` | unchanged (57.8 vs 57.1 t/s) |

### The remaining blocker: `-sm tensor` + **MTP** (only)

MTP n3, c8192, `code-python.txt`, n=128, 2 GPU IQ4_NL `-ncmoe 48`, `--load-mode auto`, clean box
(an earlier measurement round was polluted by a leftover `llama-cli` holding ~24 GB/device):

| config | t/s |
|---|---:|
| MTP `-sm layer`, cache auto | **57.8** |
| MTP `-sm tensor`, cache off | 30.7 |
| MTP `-sm tensor`, cache auto | **20.3** |
| MTP `-sm tensor`, cache auto, `MOE_EXPERT_CACHE_DEVPOLICY=0` | 31.5 |
| MTP `-sm tensor`, cache auto, `MOE_EXPERT_CACHE_PREFILL_SEED=0` | 30.6 |
| MTP `-sm tensor`, cache auto, `DEVMAP=0` / `KSLOT=0` / `TOUCH=0` / `PROVISIONAL=0` / `DEVGATHER=1` / `PERIOD=1` | 20-23 |

No-MTP (`--spec-type none`) is FINE under both splits: `-sm tensor` 16.9 -> **35.4** (+110 %),
`-sm layer` 18.8 -> 37.6.

**The decisive new measurement is `llama-batched-bench`** (multi-token decode, no MTP),
`-sm tensor -ncmoe 48`, `-npp 64 -ntg 64`:

| npl | cache off | cache auto | `-sm layer` cache auto |
|---|---:|---:|---:|
| 1 | 17.5 | 21.2 | 10.2 |
| 4 | 40.0 | **127.3** | 95.5 |

So under `-sm tensor` the cache is *excellent* for plain 4-token batched decode (3.2x over cache-off, and
it BEATS `-sm layer`).  The bug is **specific to the MTP verify**: a 4-token `batched-bench` step costs
~31 ms with the cache, while an MTP step (draft + the same 4-token verify) costs ~172 ms.  Cache-off, both
are ~100-114 ms.  So the cache makes the MTP step 1.5x SLOWER while it makes an equivalent batched step
3x FASTER.

**Ruled out this session (all instrumented/measured, not guessed):**

* the CPU-side cache work -- `moe_cache_take_over` averages 0.0005-0.0012 ms/call (total 6 ms over the
  whole run under `-sm tensor`, vs 0.5 ms under `-sm layer`), and `moe_cache_update` is called < 200
  times; the deferred-promotion/sync block never fires (`moe_promote_recs` empty, no meta synchronize);
* CUDA-graph capture (`GGML_CUDA_DISABLE_GRAPHS=1` identical), warmup, draft op-offload,
  `DEVGATHER`, `KSLOT`, `TOUCH`, `PROVISIONAL`, `PERIOD`, `DEVMAP`;
* the verify path running on the CPU -- with the real 544-token prompt every graph's 144/147 expert ops
  are on `Meta(ROCm0,ROCm1)`.  (With a *short* prompt one 13-token graph does land on the CPU because
  `GGML_OP_OFFLOAD_MIN_BATCH` defaults to 32; forcing it to 1 made things worse, 5.9 t/s);
* the device policy is a real but only ~11 t/s component (20.3 -> 31.5); it does not explain the loss.

So the next step is a **kernel-level** comparison of the MTP verify's `MUL_MAT_ID` with and without the
cache under `-sm tensor` (rocprof on a short run; the model load makes a full run too slow to trace).
Until then `-sm tensor` + host experts + MTP + cache stays **off**; plain (non-MTP) decode benefits and
`-sm layer` (57.8) is the config to recommend for MTP on 2 GPUs.

### RESOLVED: the `-sm tensor` + MTP overhead was the device-side admission policy on split tables

The apparent "the cache makes `-sm tensor` + MTP slower" (20.3 t/s) was a short-run artifact plus one
real defect:

1. **A fixed startup cost, misread as a slow cache.**  `-sm tensor` + MTP + cache has a ~4.8 s one-time
   cost, so `-n 128` is dominated by it.  Sweeping the token count shows the steady state is fine:
   n128 20.3 -> n256 31.3 -> n512 45.5 t/s (and the cache-off baseline is a flat ~30).
2. **The device-side admission policy (and its prefill seed) must not run on a SPLIT table.**  Its policy
   kernel and seed are tuned for a whole, per-device expert; on a `-sm tensor` (Meta-split) slice they
   cost ~10 t/s.  **Fix:** `alloc_table_locked` (`moe-expert-cache.cu`) now arms the device policy only
   when `t.split_axis < 0`; `MOE_EXPERT_CACHE_DEVPOLICY_SPLIT=1` restores the old behaviour.  Split tables
   keep the cache (arena + slot remap) and use the host promotion instead.

Measured (2 GPU IQ4_NL, `-ncmoe 48`, MTP n3, c8192, `--load-mode auto`):

| `-sm tensor` MTP n3 | n128 | n512 | n1024 |
|---|---:|---:|---:|
| cache off | 28.5 | 30.5 | 30.8 |
| cache auto, before | 20.3 | 45.5 | 60.3 |
| **cache auto, after** | **31.2** | **55.3** | **70.8** |
| `-sm layer` cache auto (unchanged by the fix) | 57.8 | 67.0 | - |

So `-sm tensor` now **beats** `-sm layer` at long runs (70.8 vs 67.0) and is 2.3x its own cache-off
baseline.  Why the split differs: the Meta forward drives the device policy per simple device, and the
seed's bulk admission fills strided slices (an axis-0 split uses `cudaMemcpy2DAsync`), so machinery that
pays off for a whole per-device expert costs more than it saves once the expert is split in half.

Note the fix is general: `split_axis >= 0` is set by the Meta forward (the scheduler passes `-1`), so
this also covers any future split table, and `-sm layer` / single-GPU are untouched.

**Correctness -- all byte-identical** (sha256 of the generated text, `prompts/code-python.txt`, `-n 128`,
seed 42, temp 0): `-sm tensor` cache off == auto == MTP n1 == n3 == n7 == `-sm layer` =
`03c4c58e14742964`; `-ncmoe 0` 3-GPU Q4_K_M cache off == auto == `MIB=8192` = `49cadd794126ed66`.

**Still to run before shipping (r15):** long MTP acceptance at `-n 3000` (>= 0.45), `test-backend-ops -o
MUL_MAT_ID`, coherence c32K/c128K, 1-GPU check, and the batched-bench stock-relative gate.

### The page fault is NOT fixed by the loader change

8 x `--load-mode none -sm tensor -ncmoe 48` (`AMD_SERIALIZE_KERNEL=3`, cache off): **1/8 still faults**
(same `GPU node-3 ... Page not present`, at LOAD, stderr only 308 bytes -- before the buffer-size lines).
The fix lowers the rate at best.  The faulting call is still unidentified.

### Revised direction

* Ship 1+2 only as the *enabler* (they fix the CPU fallback and the inert cache), and only once the
  device-policy overhead is resolved.
* The fault needs its own investigation (a forced-failure `rocgdb`/`rocprof` run); the host buft was not it.
* Full WIP diff: `tensor-host-buft-sched-offload.patch` (+ the earlier `pinned-2d-upload.patch`).

---

## Update 2026-10-05 (r14 session)

**The original RLIMIT/`ROCm_Host` fallback root cause is DISPROVEN.**  `hipHostMalloc` succeeds well past
this box's 80 GiB `RLIMIT_MEMLOCK` (measured 90/92/93 GiB all `hipSuccess` with ~100 GiB free), and the
`--load-mode none` + `-sm tensor` config does **not** use `ROCm_Host` at all.  The real layout:

| config | host expert buffer | PLE | device |
|---|---|---|---|
| `-sm tensor -ncmoe 48` | **`CPU_REPACK` 64800 MiB (pageable)** | `CPU` 27807 (none) / `CPU_Mapped` 27466 (auto) | `Meta()` 1790 |
| `-sm layer  -ncmoe 48` | **`ROCm_Host` 61557 + 31050 (pinned, per device)** | same | `Meta()` |

So `-sm layer`'s host master is pinned per device while `-sm tensor`'s is a **single pageable
`CPU_REPACK` buffer** (the loader's `layer_host_buft` is null for the Meta device, so the `-ncmoe`
override falls back to `select_weight_buft(..., buft_list_cpu)` = `CPU_REPACK`).  A GPU access to that
pageable master faults on a page that is not resident -- which is why the crash is `-sm tensor`-only,
why it is memory-state dependent (rare: ~1/3..1/6 runs), and why the default `--load-mode auto`/`mmap`
(and `-sm layer`) are stable.  The fault is at **load** (stdout is only the loading spinner), on a host
VA, and the `--load-mode auto` vs `none` difference is only the PLE (`CPU_Mapped` vs `CPU`).

**Landed this session (WIP, not yet in the delivery):**

* `ggml_backend_cuda_buffer_set_tensor_2d` (the load-path splice) now gathers the source rows into a
  pinned staging buffer and copies from there instead of issuing a **pageable-source
  `cudaMemcpy2DAsync`** -- the pattern `ggml_backend_cuda_set_tensor_2d_async` already uses (exp32/33).
  This removes one known-faulting H2D 2-D path; the crash still reproduced ~1/3 after it, so it is not
  the whole cause.
* `common_init_result` now warns (not rejects) on `--load-mode none` + `-sm tensor` + host experts,
  naming the workaround (`--load-mode auto`/`mmap` or `-sm layer`).

**Still open:** the exact faulting call.  A coredump did not materialise (`HSA_ENABLE_COREDUMP=1`); the
next step is `rocgdb`/`rocprof` on a forced-failure run, or forcing the `-sm tensor` host master to the
pinned `ROCm_Host` buft (a loader change) and re-testing -- that is the leading fix direction.

### Same root cause: the MoE expert cache is INERT under `-sm tensor`

`CPU_REPACK`'s buffer type sets `.is_host = nullptr`, so `ggml_backend_buft_is_host(CPU_REPACK)` is
**false**.  The host experts under `-sm tensor` are therefore not recognised as host:

* the loader's `moe_host_expert_bytes` accumulation (gated on `ggml_backend_buft_is_host`) is empty, so
  the cache preflight and the `--fit` floor reservation never run;
* the cache's table registration never happens, so `alloc_all_locked` never sizes an arena.

Measured 2 GPU IQ4_NL `-ncmoe 48`, MTP n3, c8192, n=128, `--load-mode auto`:

| split | cache | t/s | h | arena |
|---|---|---:|---:|---:|
| `-sm layer` | off | 30.1 | - | - |
| `-sm layer` | auto | **57.1** | 0.983 | 50411 MiB (77.8 %) |
| `-sm tensor` | off | 31.1 | - | - |
| `-sm tensor` | auto | 30.6 | **0.0000** | **0 (never sized)** |

So `-sm tensor` is not intrinsically slow here (its cache-off rate equals `-sm layer`'s); the whole gap is
that its cache never engages.  The fix is the same one: put the `-ncmoe`/`--cpu-moe` host experts in a
**pinned host buft** (`ROCm_Host`, `is_host = true`) for the tensor split too, instead of the pageable
`CPU_REPACK` fallback (the loader's `layer_host_buft` is null because the Meta device has no host buft).
That should both stop the fault and let the cache engage.

---

## Implementation plan (r15 follow-up) -- the combined fix

**NEXT SESSION: read this section, then implement.  One loader change fixes BOTH the `--load-mode none`
page fault and the inert cache under `-sm tensor`.**

### The change (single site)

`src/llama-model-loader.cpp`, in `create_tensor`'s `buft_for_tensor` lambda, the `-ncmoe`/`--cpu-moe`
CPU-override branch (currently ~line 1259):

```cpp
if (op == GGML_OP_MUL_MAT_ID && buft_list_layer != nullptr && !buft_list_layer->empty()) {
    ggml_backend_dev_t layer_dev = buft_list_layer->front().first;
    ggml_backend_buffer_type_t layer_host_buft =
        layer_dev != nullptr ? ggml_backend_dev_host_buffer_type(layer_dev) : nullptr;
    if (layer_host_buft != nullptr && weight_buft_supported(hparams, t_meta, op, layer_host_buft, layer_dev)) {
        buft = layer_host_buft;                 // pinned ROCm_Host (is_host == true)
    }
}
if (buft == nullptr) {
    buft = select_weight_buft(hparams, t_meta, op, buft_list_cpu);   // -> pageable CPU_REPACK
}
```

Under `-sm tensor` the layer's device is the **Meta** device, which has no host buffer type, so
`layer_host_buft` is null and the fallback lands on pageable `CPU_REPACK` (its `.is_host` is null, so
`ggml_backend_buft_is_host(CPU_REPACK) == false`).  Before that fallback, for an `op == GGML_OP_MUL_MAT_ID`
expert prefer a **real device's** `ggml_backend_dev_host_buffer_type` (pinned, `is_host == true`):

```cpp
if (buft == nullptr && op == GGML_OP_MUL_MAT_ID) {
    // A host expert master a device may read must be pinned/device-accessible.  CPU_REPACK is pageable
    // AND not `is_host`, which faults on a UVA read and keeps the MoE expert cache inert.
    for (const auto & [dev, dev_buft] : *buft_list) {            // verify this holds the REAL GPUs
        ggml_backend_buffer_type_t hb = dev != nullptr ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (hb != nullptr && weight_buft_supported(hparams, t_meta, op, hb, dev)) { buft = hb; break; }
    }
}
```

**First verification step:** under `-sm tensor`, does the top-level `buft_list` contain the real GPUs or
only the Meta device?  Log both lists at DEBUG on the first run.  If `buft_list` has no real device host
type, take the pinned host buft that `make_cpu_buft_list` already seeds into `buft_list_cpu` (the first
device's `ROCm_Host`), or iterate the global `ggml_backend_reg` devices.

### Why it fixes both

* **Fault:** the `-sm tensor` host master becomes pinned/`is_host` -- the GPU's access (the split upload
  or any UVA read) is then to device-accessible memory.
* **Cache:** `ggml_backend_buft_is_host` becomes true, so the loader's `moe_host_expert_bytes`
  accumulates (the preflight and the `--fit` floor reservation finally run for `-sm tensor`) and the
  cache registers + sizes a table -> `alloc_all_locked` engages.

### Risks / unknowns

* The host buft carries a `.device`; under `-sm tensor` the master is shared across devices.  Confirm the
  Meta split still assembles correctly and the same-seed output is unchanged.
* `-sm layer` and single-GPU must be unchanged (their `layer_host_buft` is already a real device).
* If the Meta split rejects a per-device host buft (or `weight_buft_supported` is false for it), fall back
  to today's behaviour and record it -- then the fallback alternative is to teach the cache/preflight to
  treat `CPU_REPACK` as a host master (weaker: the fault root cause remains).

### Validation plan (gfx1201)

1. **Cache engages:** 2 GPU IQ4_NL `-sm tensor -ncmoe 48`, `--load-mode auto`, cache auto vs `MIB=0` --
   expect `h > 0.9`, an arena sized, and ~57 t/s (vs 30.6 today); cache-off unchanged (~31).
2. **Fault repro:** `--load-mode none -sm tensor -ncmoe 48`, **N >= 10** runs (`AMD_SERIALIZE_KERNEL=3`)
   -- expect 0 faults (today ~1/3); repeat `-ncmoe 32/40`.
3. **Gates:** `-ncmoe 0` byte-identity (3-GPU Q4_K_M, auto == off == `MIB=8192`); width purity
   `none == n1 == n3 == n7`; long MTP acceptance >= 0.45 at `-n 3000`; coherence c32K/c128K (`////`=0);
   single-GPU + 3-GPU `-sm layer` unchanged.
4. `test-backend-ops -o MUL_MAT_ID`.

### Delivery

* Land as **block 06** (the loader's host-buft selection already lives there), for **r15**, together with
  the already-WIP pinned-staging `set_tensor_2d` fix + the `--load-mode none` warning
  (`wip/host-pinned-buffer-crash/pinned-2d-upload.patch`, committed on `main` but not in r14).
* Add an env kill-switch (`LLAMA_TENSOR_HOST_BUFT=0`) so the A/B is cheap and the change is revertible
  (the WIP promotion rule).
* Regenerate + `validate-set.sh` + build + the gates above, then cut the release and refresh the fork's
  `rdna-boosts` branch (the mandatory release-time refresh in `AGENTS.md`).


## Sibling WIP campaigns (index relocated here)

There is no longer a `README.md` directly under `wip/`; every campaign is a self-contained handover under
its own `README.md`.  The current set, for cross-discovery:

| directory | what | status |
|---|---|---|
| `moe-cache-autosize/` | arm + auto-size `MOE_EXPERT_CACHE_MIB` when experts are host-resident | PROMOTED (r14, block 13) |
| `layer-split-host-experts/` | per-device host bufts so `-sm layer` spreads experts over the GPUs | PROMOTED (r14, block 06) |
| `host-pinned-buffer-crash/` (this) | `--load-mode none` GPU page fault, and the inert `-sm tensor` cache | OPEN (r15) |
| `moe-cpu-overlap/` | genuine CPU/GPU overlap for the expert misses (Strata shape) | OPEN / scoping |
| `strata-amd-kernels/` | compare Strata's AMD decode kernels against block-10/13/15 | OPEN / scoping |
| `nwarps/` | per-M `nwarps` MoE candidate -- the one deliberate width-purity impurity | ACTIVE (env-OFF) |
| `mmvq-verify-rows/` | faster multi-token mmvq on RDNA4 (bit-exact) | open |
| `host-memory-footprint/` | host-memory footprint of GPU-resident weights (gfx1100) | open |
| `fp8-support/` | native FP8 E4M3 for RDNA4 | PARKED |
| `moe-mmq-overread/` | resolved MoE MMQ tail over-read (kept as the record) | closed |

Resolution pattern (investigate -> fold into the owning block -> regenerate + validate -> record +
archive -> ship) is documented in `archive/work/lightning-indexer-fusion/README.md`; promotion rules are
in `AGENTS.md`.

## Original (2026-10-05) record

**Status: OPEN (2026-10-05). Nothing here is in the delivery.** Found during the auto-size M0 sweep
(`wip/moe-cache-autosize/`). Maintainer: "We really need to fix up all those crashes."

## Symptom

`llama-cli`, 2 x R9700 (gfx1201), `-sm tensor` + host-resident experts (`-ncmoe ~>=32`), a large model
(`/llm/models/Qwen3.8/Flash-Next/IQ4_NL/`, **93.16 GiB total / 63.28 GiB experts**):

```
Memory access fault by GPU node-3 (Agent handle: 0x...) on address 0x7f... Reason: Page not present or supervisor privilege.
Failed to write segment data to pipe: Broken pipe
GPU coredump: handler exited with error (status: 1)
```
Host exit code 134 (SIGABRT). The fault is on a **host virtual address**; the failing device is a
R9700 (rocminfo Agent 3), not the 4 GB card.

## The trigger is `--load-mode none` (with a big pinned footprint)

| load mode | other | result |
|---|---|---|
| `--load-mode none` | `--lazy-mode off` | **crash ~2/3 runs** (batch-dependent; one batch was 0/3, another 3/3) |
| `--load-mode none` | `--lazy-mode auto` | passed the batches run |
| `--load-mode auto` (default) | either | **pass 3/3** (even with `AMD_SERIALIZE_KERNEL=3`) |
| `--load-mode mmap` | `--lazy-mode off` | **pass 3/3** |

It is **not** the expert cache: `MOE_EXPERT_CACHE_MIB=0` (cache off) still crashes. `-sm layer` does not
crash (it just runs the known-bad MoE-consolidated decode, 10.3 t/s). `-ncmoe 24 -sm tensor` did not
crash in the runs done; `-ncmoe 32/40/48` did (once each at 32, always-ish at 40/48).

`AMD_SERIALIZE_KERNEL=3` makes it **deterministic in a session** (3/3), but the same command passed 3/3
later in a different memory state — so the crash is a function of how much pinned memory the box can
actually back at that moment, not a pure race.

## Root cause (leading, evidence-backed)

`--load-mode none` (`use_mmap = false`) puts **the whole model's host-resident weights into ONE
`ROCm_Host` buffer**:

* `--load-mode none`: `ROCm_Host model buffer size = 92606.97 MiB` (no `CPU_Mapped`).
* `--load-mode mmap`: `CPU_Mapped 27465.95 MiB` (the PLE) + `ROCm_Host 64800.00 MiB` (the experts).

`ROCm_Host` is `ggml_backend_cuda_host_buffer_type`; its allocator
(`ggml/src/ggml-cuda/ggml-cuda.cu:1782`) does:

```c
void * ptr = ggml_cuda_host_malloc(size);            // cudaMallocHost(size)
if (ptr == nullptr) {
    return ggml_backend_buft_alloc_buffer(cpu_buffer_type, size);  // pageable, but
}                                                                  // buft keeps the ROCm_Host NAME
```

Two problems combine:

1. **The request (92.6 GiB) exceeds this box's `RLIMIT_MEMLOCK` (80 GiB)** (`ulimit -l` = 83886080 KiB,
   hard limit the same). `--load-mode mmap` pins 64.8 GiB (< 80) and is stable; the full 92.6 GiB
   single request is not. When the pin cannot be fully backed, the buffer is not uniformly
   device-accessible and a GPU access faults.
2. **The fallback is silently mislabelled** — if `cudaMallocHost` fails, a pageable CPU buffer is
   returned under the `ROCm_Host` name, so `ggml_backend_buft_is_cuda_host()` and every
   device-accessibility assumption still pass while the memory is pageable.

`cudaMallocHost` here also lacks `cudaHostAllocPortable`/`Mapped` (the allreduce path uses
`hipHostMallocPortable | hipHostMallocMapped`), so cross-device access is not requested explicitly.

The fault is a **GPU access to a host page that is not mapped for that device**, which is exactly what
a partially-backed / pageable-fallback `ROCm_Host` buffer produces.

## Fix directions (not yet decided)

1. **Loader**: under `--load-mode none`, don't pin the whole model into one host buffer. Keep the
   device-access-critical tables (routed experts) pinned and route large non-expert host tensors (the
   26.8 GiB PLE) to a mapped/CPU buft — i.e. apply the same "host experts stay pinned" rule the
   `LLAMA_MMAP_HOST_EXPERTS` path already applies, and cap the total pin request.
2. **Host buft allocator**: never return a pageable buffer under the `ROCm_Host` name. Either return
   nullptr (clean load error: "unable to allocate ROCm_Host buffer") or return a buffer whose
   `get_name` is not the host type so downstream device-accessibility checks are correct.
3. **Portability**: use `cudaHostAllocPortable | cudaHostAllocMapped` (or `cudaHostRegisterPortable`)
   for the host buffer when more than one device may read it.
4. **Guard**: reject `--load-mode none` + `-sm tensor` + host-resident experts with a clean error that
   names `--load-mode mmap`/default, until 1-3 land.

A `RLIMIT_MEMLOCK` pre-check (compare the planned pinned bytes against `getrlimit(RLIMIT_MEMLOCK)` and
the device's pin capacity) would turn this class of failure into a warning at every pin site.

## Repro (minimal, as run here)

```bash
M=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib \
MOE_EXPERT_CACHE_MIB=0 AMD_SERIALIZE_KERNEL=3 \
./build-rocm/bin/llama-cli -m $M -ngl 99 -sm tensor -ncmoe 48 -fa 1 \
  -ctk q8_0 -ctv q8_0 -t 8 -c 8192 -b 2048 -ub 2048 \
  --lazy-mode off --load-mode none --spec-type none \
  -p "The capital of France is" -n 32 --seed 42 --temp 0 --single-turn --no-display-prompt
# AMD_SERIALIZE_KERNEL=3 for determinism; drop --load-mode none (use auto) to pass.
```

## Next steps

* N1: capture the failing GPU kernel (ROCm coredump via `HSA_ENABLE_COREDUMP=1`, or `rocprof`/`rocgdb`)
  to confirm which access faults.
* N2: verify the RLIMIT link — reduce the pinned request below 80 GiB without changing the load mode
  (e.g. pin only the experts) and confirm it passes; then reproduce with an artificially raised
  `RLIMIT_MEMLOCK` if the hard limit can be lifted.
* N3: pick one of fix directions 1-4, implement behind a fix, and gate on the 2-GPU IQ4_NL repro plus
  the existing `-sm tensor` oracles (`de8be4d0c90c`, `d7bef4c6fdc3`).
* N4: sweep the neighbouring load modes (`dio`, `mlock`, `mmap+mlock`) for the same fault.

## Related

* `wip/moe-cache-autosize/README.md` — the M0 sweep that surfaced this (2-GPU IQ4_NL).
* `patches/README.md` block 06 (staging / meta split), block 14 (`-sm tensor` gates).
* `src/llama-model-loader.cpp:1283-1297` — the `LLAMA_MMAP_HOST_EXPERTS` pinned-expert exception.
* `ggml/src/ggml-cuda/ggml-cuda.cu:1764-1795` — `ggml_cuda_host_malloc` and the silent fallback.
