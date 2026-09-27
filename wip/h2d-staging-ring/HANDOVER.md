# HANDOVER — issue #50: H2D staging ring (the `-sm tensor` blocker is resolved)

Updated 2026-09-27.  `README.md` in this directory is the full evidence record; `TODO.md` item 24 is the
same work in tracker form.

**Status: the blocker is fixed and measured.  What remains is the gate battery's remaining items and the
promotion decision.**

## 0. Summary

The ring is no longer inert under `-sm tensor`.  The reported symptom ("staging does nothing under tensor
split") was real, but its cause was not the ring: **the meta device never declared `offload_op`, so under
`-sm tensor` the scheduler never op-offloaded the MoE at all and the expert weights ran on the CPU.**  With
that fixed (plus two follow-on fixes it exposed), `-sm tensor -ncmoe 99` pp8192/ub8192 goes

**523 t/s (MoE on CPU) → 1823 t/s (op-offload) → 2742 t/s (op-offload + staging ring)**.

## 1. The three fixes (all in `h2d-stage.patch`, 1200 lines, 20 files, applies clean on r11)

1. **`ggml_backend_meta_device_offload_op()`** (`ggml/src/ggml-backend-meta.cpp`) — the meta device
   declares offload support when every simple device does.  `ggml_backend_dev_offload_op()` defaults to
   false for a device with a NULL hook, so `ggml_backend_sched_backend_id_from_cur()`'s op-offload branch
   could never select the meta backend.  **This is the enabling fix.**
2. **Mirrored tensors serve arbitrary byte ranges** (`ggml_backend_meta_set/get_tensor_async`) — enabling
   op-offload lit up the scheduler's used-expert pruning, which uploads at a non-zero byte offset and reads
   the router's ids as a strided view's raw span; the old `offset == 0` / `ggml_is_contiguous` assert pair
   aborted on both (core dump at `ggml-backend-meta.cpp:2083`).  A MIRRORED tensor needs no chunk
   arithmetic, so its range is forwarded as-is; the partial axes keep their asserts.
3. **Per-device staging** (`stage_input` hook + the meta-side ring) — what §3 of the previous handover
   described as the whole job.  It was needed, but it is the third fix, not the first.

Plus a latent bug: **15 backends' positional `ggml_backend_i` initializers** omitted the staging fields, so
adding them assigned each backend's `graph_optimize` to `stage_buffer` and NULLed `graph_optimize`
(metal/vulkan/hexagon/virtgpu lost their optimizer).  All 18 initializer lists are now complete.

## 2. Measured (`-sm tensor -ncmoe 99 -fa 1`, 2× R9700, pp8192, this box)

| ub | MoE on CPU (pre-fix) | op-offload, no staging | + staging |
|---|---|---|---|
| 1024 | ~340 | 337.17 | 345.20 (gated) |
| 2048 | 508.42 | 620.27 | **715.71** |
| 4096 | — | 1092.56 | **1413.94** |
| 8192 | 523.06 | 1823.12 | **2741.56** |

`-ncmoe 10 -sm tensor` (host **and** device MoE layers) works and stages: 1958.74 → **2281.82** (+16 %).
`-sm layer -ncmoe 99` pp8192/ub8192 on the same box: 2708.21 → **4111.14** for reference.

Tensor split replicates the expert weights (split state `MIRRORED`), so it uploads the whole tensor to
**every** device — it cannot match layer split.  The fix removes the CPU fallback; it does not make tensor
split the better way to serve `-ncmoe`.

## 3. Gates run

| gate | result |
|---|---|
| stage 0 vs stage 1 greedy text, `-sm tensor -ncmoe 99` | byte-identical, sha `0936c8318533` |
| narrow-ubatch crash repro (ub 1024 → gate closed → pruned path) | runs clean (was a core dump) |
| `test-backend-ops -o FLASH_ATTN_EXT` | 2/2 OK |
| `test-backend-ops -o MUL_MAT_ID` | 2/2 OK |
| `-sm layer` path | untouched by all three fixes (`stage_input` NULL; the MIRRORED change is meta-only) |

## 4. What remains before promotion

1. **Re-run the rest of the gate battery** on this build (the single-device items in README §"Gate
   battery" were green before the `-sm tensor` work and `-sm layer` is unaffected, but the scheduler did
   gain a hook and a flag — re-run items 1-6 to be safe, then 7 for `-sm tensor`).
2. **Decide the delivery home.**  The change is no longer one feature: it is (a) a meta-device capability
   (`offload_op`), (b) a meta-backend range fix, (c) the staging ring, (d) a 15-backend initializer fix.
   (d) is an independent latent-bug fix and probably wants its own block-00-style home; (a)+(b) are a
   `-sm tensor` op-offload fix that is valuable **without** the ring (523 → 1823 t/s) and could ship
   separately from (c).
3. **Regenerate the delivery set** from a canonical rebuild (`scripts/make-patches.sh` now also re-cuts
   `rdna-boosts-all.patch`), then `make-release.sh` + `validate-set.sh` + tag.
4. **The reporter's input on PR #51** is still outstanding (a 55 GB/s run was requested).

## 5. Environment, build, repro

* **soar** (local): 3× R9700 gfx1201, `~/llama.cpp` (r11 tree `080deacaa` + `h2d-stage.patch` +
  the uncommitted r10 test additions), build `build-rocm`.
  Runtime libs come from the binary's RUNPATH (`/opt/rocm-7.14.1-gfx102X/lib`); `/opt/rocm-7.14-gfx1201`
  is **not** the one used by this build.
* Model: `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21.10 GiB).
* **The `/tmp/canon-fix` worktree of `~/llama.cpp/.git` is the canonical r11 chain** (branch
  `r10-rebuild`, tip `080deacaa`, clean).  `git -C /tmp/canon-fix apply --check <patch>` verifies a patch
  applies on r11 without touching the working tree.
* Repro:
  ```bash
  cd ~/llama.cpp
  M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  for s in 0 1; do HIP_VISIBLE_DEVICES=0,1 GGML_SCHED_STAGE=$s ./build-rocm/bin/llama-bench \
      -m "$M" -ncmoe 99 -fa 1 -p 8192 -n 1 -b 8192 -ub 8192 -sm tensor -r 1; done
  ```
* **Build**: `CCACHE=0 BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714` (~6 min, 16 cores).  `CCACHE=0`
  because the HIP build emits no `.d` files and ccache's direct mode has silently reused stale objects
  after a header edit here — the script `rm -rf`s the build dir, so this is the safe path.
* **`GGML_SCHED_DEBUG=2` needs `-v`** to print the node→backend assignments; that is how the missing
  op-offload was found (grep `MUL_MAT_ID` in the split dump).
* **`GGML_LOG_INFO` needs `-v`**; `GGML_LOG_DEBUG` (the scheduler's `batch_tokens=... staged=...` line)
  apparently does not print even with `-v`, so instrument with `GGML_LOG_INFO` when debugging.
* A clean way to reproduce the pre-fix behaviour without a rebuild:
  `GGML_OP_OFFLOAD_MIN_BATCH=1000000` makes every device refuse op-offload, sending the MoE back to the
  CPU.

## 6. Reference

| thing | where |
|---|---|
| the WIP patch (r11 base) | `wip/h2d-staging-ring/h2d-stage.patch` — **1200 lines, 20 files** |
| PR #51 (the reporter's ring) | `wip/h2d-staging-ring/reporter-h2d-ring.patch` |
| evidence record | `wip/h2d-staging-ring/README.md` §"`-sm tensor`: resolved 2026-09-27" |
| tracker item | `TODO.md` item 24 |
| scheduler side | `ggml/src/ggml-backend.cpp` — `sched_stage_issue` (sets `stage_split_ok`), the `stage_input` call in the input loop, `sched_stage_min_tokens` |
| backend contract | `ggml/src/ggml-backend-impl.h` — `stage_buffer/upload/wait/d2d/h2d_gbps` + `stage_input` |
| meta side | `ggml/src/ggml-backend-meta.cpp` — `ggml_backend_meta_device_offload_op`, the MIRRORED early-outs in `set/get_tensor_async`, `ggml_backend_meta_stage_input`, `ggml_backend_meta_stage_guard`, `ggml_backend_meta_stage_h2d_gbps` |
| CUDA side | `ggml/src/ggml-cuda/common.cuh` (`h2d_stage[]`, `h2d_stage_buffer`, `copy_stream()`); `ggml-cuda.cu` (the `ggml_backend_cuda_stage_*` impls + the calibration) |
| knobs | `GGML_SCHED_STAGE`, `GGML_SCHED_STAGE_MODE`, `GGML_SCHED_STAGE_SLOTS` (8), `GGML_SCHED_STAGE_MAX_MB` (2048), `GGML_SCHED_STAGE_MIN_TOKENS`, `GGML_SCHED_EVENTS`, `GGML_OP_OFFLOAD_MIN_BATCH` |
