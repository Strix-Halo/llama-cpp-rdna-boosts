# `-sm layer` + host experts: distribute the expert buffers per device

**Status: FIXED in the WIP fork, measured (2026-10-05). Nothing here is in the delivery.**
Maintainer: "it's routing the experts all to GPU0 instead of layering them evenly across the GPUs."
Confirmed in the source; the fix is implemented in `~/llama.cpp` (`fix.patch` beside this file) and
measured **2-GPU IQ4_NL `-sm layer -ncmoe 48`: 10.3 -> 55.4 t/s** in a same-session A/B.

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

## Fix (implemented — 4 parts; `fix.patch`)

Three facts compose as above; the fix needs all four because the buffer device is only half the story —
the **scheduler's op-offload loop** is the actual decider.

**F1 — per-device CUDA host buffer types (`ggml/src/ggml-cuda/ggml-cuda.cu`).** `ggml_backend_cuda_host_buffer_type()`
(public, now device 0) is split into `ggml_backend_cuda_host_buffer_type_dev(int device)` backed by a
per-device `static ggml_backend_buffer_type bufts[GGML_CUDA_MAX_DEVICES]`, and
`ggml_backend_cuda_device_get_host_buffer_type(dev)` returns the entry for `ctx->device`. Pinned storage
is device-agnostic (UVA); only `.device` differs.

**F2 — the loader picks the layer's device host buft (`src/llama-model-loader.cpp`).** In
`create_tensor`'s CPU-override branch, for `op == GGML_OP_MUL_MAT_ID` prefer
`ggml_backend_dev_host_buffer_type(buft_list_layer->front().first)` (the layer's assigned device),
falling back to today's `select_weight_buft(..., buft_list_cpu)`.

**F3 — `ctx_key` must not merge same-name bufts (`src/llama-model-loader.h`).** The comparator compared
buffer types by **name**; all per-device host bufts are named `ROCm_Host`, so they collapsed into one
context and one buffer (all on device 0). It now falls back to the device name when names are equal.

**F4 — the scheduler offload loop is the real decider (`ggml/src/ggml-backend.cpp`).**
`ggml_backend_sched_backend_id_from_cur`'s `sched->op_offload` loop returns the **first** backend that
can offload the host weight, which is always device 0. It now skips backends whose device differs from
`ggml_backend_buft_get_device(src->buffer->buft)`. Without F4, F1+F2 change the buffers but every MoE
op still runs on device 0.

## Measured (same session, 2 x R9700, IQ4_NL 93 GiB, MTP n3, c8192 q8_0, n=128)

| config | before | after |
|---|---:|---:|
| `-sm layer -ncmoe 48`, cache 24000 | 10.3 | **55.4** |
| `-sm tensor -ncmoe 24`, cache off | 43.2 | 45.4 (unchanged, environment drift) |
| 1 GPU IQ3_XXS `-ncmoe 48`, cache 20480 | 48.7 | 48.7 (unchanged) |
| 2 GPU `-sm layer`, cache off (CPU MoE) | 32.4 | 32.4 |

**`-sm layer` now beats `-sm tensor` for the oversized 2-GPU case** (55.4 vs 45.4) — and it is the split
that does not hit the `--load-mode none` crash (`wip/host-pinned-buffer-crash/`). Correctness: output
coherent (`////`=0), same-seed deterministic (`618b47905a2a`, two runs), 4B `-ncmoe 0` all-resident fine
(95.8 t/s). The single-GPU 52.1 first reported was environmental — the pre-change build measured 48.7 in
the same state, so the change is a no-op for one device.

The patch is at [`fix.patch`](fix.patch) (4 files, +75/-21). **Not folded into `patches/`** — that needs
the maintainer's go-ahead, plus the gates below and a block placement decision (06 generic scheduler +
loader, or a new block).

## Original fix write-up (for reference)

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
