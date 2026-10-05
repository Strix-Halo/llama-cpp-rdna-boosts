# Host-resident expert load: GPU page fault under `--load-mode none` (2 GPUs)

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
