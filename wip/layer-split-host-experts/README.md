# `-sm layer` + host experts: distribute the expert buffers per device

**Status: OPEN (2026-10-05). Nothing here is in the delivery.** Maintainer: "it's routing the experts
all to GPU0 instead of layering them evenly across the GPUs." Confirmed in the source; the fix is
scoped below.

## Symptom

2 x R9700, Qwen3.8-Flash-Next IQ4_NL (93 GiB), `-sm layer -ncmoe 48`, MTP n3, c8192, q8_0 KV,
cache on (`MOE_EXPERT_CACHE_MIB=24000`): **10.3 t/s** — vs `-sm tensor -ncmoe 24` cache off **43.2**
and `-sm layer` cache off (CPU MoE) **32.0**. The layer assignment itself is even (layers 0..23 on
ROCm0, 24..47 on ROCm1 — verified in the loader log); it is the **expert buffers** that all land on
device 0, so every MoE op runs there and GPU 1 idles on the expert half.

The delivery already knows the symptom: `WORKLOG.md` 2026-10-05 (r12) — *"With `-sm layer` on 2 GPUs
and host experts (`--n-cpu-moe` 29 and above) every MoE op of GPU 1's layers runs on GPU 0."* and
`COMMUNITY-CONFIG.md` — *"Multi-GPU: `tensor` beats `layer` at every offload level."*

## Root cause (source-confirmed)

Three facts compose:

1. **The CUDA host buffer type is a device-0 singleton** —
   `ggml_backend_cuda_host_buffer_type()` (`ggml/src/ggml-cuda/ggml-cuda.cu:1797`) has
   `/* .device = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0)`, and
   `ggml_backend_cuda_device_get_host_buffer_type(dev)` returns that same singleton for **every**
   device (`ggml-cuda.cu:8498-8501`). So there is no per-device host buft. **This is upstream code**
   (identical in `~/stock-9113` and `~/orig/llama.cpp`), not a fork regression.
2. **`make_cpu_buft_list()` adds only the first device's host buft** —
   `src/llama-model.cpp:1166-1195`: it loops the devices, adds the first non-null host buft, and
   `break`s. So `buft_list_cpu` contains exactly one host buft, whose `.device` is device 0 (fact 1).
3. **The `-ncmoe` CPU override selects from `buft_list_cpu`** — `create_tensor()` in
   `src/llama-model-loader.cpp:1250-1260`:
   ```cpp
   if (overrides->buft == ggml_backend_cpu_buffer_type()) {
       buft = select_weight_buft(hparams, t_meta, op, buft_list_cpu);   // -> device-0 host buft
   }
   ```
   Every host-resident expert tensor (all layers) therefore gets the device-0 host buft, regardless
   of the layer's assigned device. The scheduler places an op on the buffer's device, so all MoE ops
   run on ROCm0.

## Fix

**F1 — per-device CUDA host buffer types (`ggml-cuda.cu`).** Replace the singleton with a per-device
table:
```cpp
static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host[GGML_CUDA_MAX_DEVICES];
ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(int device);      // sets .device = reg_dev_get(reg, device)
ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(dev);  // -> the table entry for dev's index
```
Pinned host memory (`cudaMallocHost`) is device-agnostic/UVA, so the storage does not change — only
the `.device` the scheduler reads.

**F2 — the loader picks the host buft of the layer's device.** In `create_tensor()`, for the
CPU-override case, prefer the host buft of the device the layer is assigned to. The layer's device is
available as `buft_list_layer->front().first` (the GPU buft list passed for that repeating layer);
fall back to `select_weight_buft(..., buft_list_cpu)` when the layer is not offloaded. Gate to
`op == GGML_OP_MUL_MAT_ID` (host experts) so non-expert CPU overrides are unaffected.

**F3 (evaluate) — `make_cpu_buft_list` adding every device's host buft**, if F2 needs the list to
contain them; the ordering/matching must not change the single-GPU case.

## Why it matters

* It is the difference between `-sm layer` being a trap and a **viable multi-GPU split for
  oversized models** — and `-sm layer` is the split that does *not* have the `--load-mode none`
  crash (`wip/host-pinned-buffer-crash/`) and does not need the tensor-split host-expert staging.
* If it works, `-sm layer -ncmoe` may beat `-sm tensor -ncmoe` for Qwen3-Flash-Next-class models on
  2 GPUs, and it makes the auto-size campaign's multi-GPU target reachable without the crash.
* It is a clean, generic upstream PR candidate (`upstream/`): any `-sm layer` + `-ncmoe` user on
  multi-GPU is affected.

## Gates

* **Repro:** 2 x R9700, IQ4_NL, `-sm layer -ncmoe 48`, MTP n3, c8192 q8_0, cache on: expect the
  distribution to lift 10.3 t/s substantially (target >= the `-sm tensor -ncmoe 24` 43.2, or at least
  the `-sm layer` CPU 32.0), with **both** GPUs busy.
* Byte-identity: `-ncmoe 0` oracles `de8be4d0c90c` (2-GPU tensor) / `15038c19ddc8` (1-GPU layer);
  single-GPU behavior unchanged.
* The r12 race harness (`archive/work/2gpu-sched-fixes/VERIFICATION.md`) and the block-06
  cross-device event path still pass.
* `test-backend-ops`, width purity `W=1..8`, MTP acceptance, coherence.

## Open questions

1. Does the scheduler honour a host buffer's `.device` for op placement, or does it use the op's
   other inputs? (Read `ggml_backend_sched`'s split assignment for `MUL_MAT_ID` with a host src0.)
2. With per-device host bufts, does the block-06 staged upload (which is per device anyway) work
   unchanged?
3. Does upstream want this (device-agnostic host buffer types are a real design limitation)?
4. Interaction with `--no-host` / `buft_list_cpu` ordering.

## Record pointers

* `src/llama-model.cpp:1166-1195` (`make_cpu_buft_list`), `src/llama-model-loader.cpp:1067-1300`
  (`select_weight_buft`, `create_tensor` CPU override), `ggml/src/ggml-cuda/ggml-cuda.cu:1797-1815`
  and `:8498-8501` (host buft singleton).
* `WORKLOG.md` 2026-10-05 (r12); `COMMUNITY-CONFIG.md`; `wip/host-pinned-buffer-crash/`.
