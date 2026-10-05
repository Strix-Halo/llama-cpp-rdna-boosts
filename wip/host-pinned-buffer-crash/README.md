# Host-resident expert load: GPU page fault under `--load-mode none` (2 GPUs)

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
