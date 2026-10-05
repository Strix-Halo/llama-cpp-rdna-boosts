# UPSTREAM-PR: per-device host buffer types + host-expert op offload placement

**Patch:** `UPSTREAM-PR-per-device-host-buffers.patch` (4 files, +75/-21; re-cut against upstream
`a55e952b8`, applies clean with offsets).
**Delivery home:** **block 16** (`patches/0016-rdna-boosts-block-16-per-device-host-buffers-distrib.patch`).
**Status:** prepared 2026-10-05; validated on gfx1201 (2 x / 3 x R9700); no NVIDIA hardware available; not filed.

## What it fixes (upstream)

`-sm layer` + host-resident MoE experts (`-ncmoe`/`-cmoe`) on 2+ GPUs routes **every** expert op to
device 0, leaving the other GPUs idle on the expert half. Three upstream places compose:

1. **`ggml_backend_cuda_host_buffer_type()` is a device-0 singleton** — its `.device` is
   `ggml_backend_reg_dev_get(reg, 0)` for every device, so there is no per-device host buffer type.
2. **`llama_model_loader`'s `ctx_key` comparator compares buffer types by name** — all host buffer
   types are named `<BACKEND>_Host`, so distinct per-device host bufts collapse into one context and
   one buffer.
3. **`ggml_backend_sched_backend_id_from_cur`'s op-offload loop returns the first backend that can
   offload a host weight** — always device 0. The weight's buffer device is never consulted.

The CUDA loader (`src/llama-model-loader.cpp`, CPU-override branch) also always selects the device-0
host buffer for `MUL_MAT_ID` weights.

## The change

* `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_backend_cuda_host_buffer_type_dev(int device)` backed by a
  per-device table; `ggml_backend_cuda_device_get_host_buffer_type(dev)` returns the entry for
  `ctx->device`; the public `ggml_backend_cuda_host_buffer_type(void)` keeps returning device 0.
* `src/llama-model-loader.cpp`: for a `MUL_MAT_ID` weight overridden to a CPU buffer, prefer
  `ggml_backend_dev_host_buffer_type(layer device)`.
* `src/llama-model-loader.h`: the `ctx_key` comparator falls back to the device name when buffer-type
  names are equal (so per-device host bufts do not share a context).
* `ggml/src/ggml-backend.cpp`: the op-offload loop skips backends whose device differs from
  `ggml_backend_buft_get_device(src->buffer->buft)`.

Generic to every backend with host buffers; no AMD-specific code.

## Validation (gfx1201, 2 x / 3 x Radeon AI PRO R9700, ROCm 7.14.1)

* **2 x R9700, Qwen3.8-Flash-Next IQ4_NL (93 GiB, 63 GiB experts), `-sm layer -ncmoe 48`, cache
  24000 MiB, MTP n3, c8192 q8_0:** `10.3 -> 58.6 t/s` (same-session A/B; `-sm tensor -ncmoe 24`
  cache-off is 45.4).
* **3 x R9700, same model/config:** **81.0 t/s**.
* **1 GPU / single-device:** unchanged (48.7 t/s before and after).
* Output coherent (`////`=0), same-seed deterministic (`618b47905a2a`, two runs), MTP acceptance
  `0.88942`, `test-backend-ops -o MUL_MAT_ID` 931/931 and `-o FLASH_ATTN_QSA` 26/26.

## Not validated

* NVIDIA / other backends (same code path, but no hardware here).
* MUSA / SYCL host-buffer types (they define their own; the `ggml-backend.cpp` change is
  backend-agnostic).

## Notes for filing

Re-create the branch from current upstream master rather than applying the copy; confirm the
`ctx_key` comparator change does not merge any two intentionally-distinct same-name buffer types on
master (the device-name tiebreak is the safe order: name first, then device).
