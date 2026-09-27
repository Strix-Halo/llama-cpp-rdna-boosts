# HANDOVER — issue #50: per-device H2D staging for `-sm tensor` (the promotion blocker)

Updated 2026-09-26, after the PR #51 merge and the r11 release.  This file is the **only** thing you
need to start; `README.md` in this directory is the full evidence record, and `TODO.md` item 24 is the
same work in tracker form.

**The merged op-offload H2D staging ring works and is validated, but it is inert under `-sm tensor`, and
the maintainer treats that as a blocker to promoting this WIP into the main patch set.**  That is the
next session's job.  Everything below is what was already established so it does not have to be
re-derived.

## 0. What changed since the last handover (do not redo)

* **The merge is done.** `h2d-stage.patch` (this directory) is PR #51's op-reads-the-slot redirect +
  their `GGML_SCHED_EVENTS` + their tripwire, with our bounded arena, link-calibrated gate and 8 slots.
  It **applies clean on r11** (`git apply --check` verified against a canonical r11 rebuild).
* **The gate battery ran** (gfx1201): `FLASH_ATTN_EXT` OK; `W=1..8` pure and `GGML_SCHED_EVENTS`
  purity-neutral; MTP text byte-identical; deep prefill pure at +21 % prompt; server soak 18/18 and
  −24 % wall clock.  Numbers and the merged matrix: `README.md` §"Merged design" and §"Gate battery".
* **r11 shipped** (`v16-84e76d8a2-r11`, tag pushed): the `MUL_MAT_ID` `rpb` mis-launch the battery turned
  up is fixed in block 13.  That is unrelated to this work but it moved the delivery tip, so regenerate
  from r11, not r10.

## 1. Objective

Make the op-offload H2D staging ring work under `-sm tensor` (2+ devices), so that
`llama-server --cpu-moe` / `-ncmoe` with a tensor-split model overlaps its expert uploads the way the
single-device path does.  Then re-run the gate battery and promote the WIP into the delivery.

## 2. The problem, measured

With `-sm tensor` the ring is **inert**: the scheduler logs
`GGML_SCHED_STAGE=1 but no backend supports it (e.g. -sm tensor); staging inactive` and the numbers are
identical to staging off.

| box | config | stage 0 | stage 1 |
|---|---|---|---|
| soar, 2× R9700, 35B-A3B Q4_K_M | `-sm tensor -ncmoe 99`, `pp8192`, `ub 2048` | 498.73 t/s | 499.14 t/s |

For scale, the same model/ubatch on **one** device with `-sm layer` is ~1400 t/s and stages to 1476.  So
tensor split plus op-offload is ~2.8× slower to begin with — worth knowing when prioritising, but the
maintainer wants it fixed rather than documented.

## 3. The meta backend, precisely (the anchors)

Under tensor split the scheduler's backend list is `Meta(ROCm0,ROCm1)` **+ `CPU`** (`src/llama-context.cpp`
fills `backend_ptrs` from `backends`; the meta device wraps the per-device backends).  The meta backend is:

* `ggml/src/ggml-backend-meta.cpp`.
* `ggml_backend_meta_i` at **line 2656**; the five staging hooks sit at **2672-2676** and are all
  `nullptr` — that is *why* the scheduler reports "no backend supports it".
* `ggml_backend_meta_set_tensor_async` at **line 1965**.  This is the interesting function.  One logical
  upload is **spliced across N devices**:
  * `ggml_backend_meta_get_split_state(tensor, false)` gives the axis; the real cases are
    `SPLIT_AXIS_0/1/2` (partial) and `SPLIT_AXIS_MIRRORED` (`ggml-backend.h:369`).
  * For a partial axis it walks the simple backends with
    `ggml_backend_tensor_set_2d_async(simple_backend, simple_tensor, data + offset_j, offset,
    chunk_size_j, i_stop - i_start, chunk_size_j, chunk_size_full)` — one chunk per device, at a
    per-device offset.
  * For `MIRRORED` it calls `ggml_backend_tensor_set_async` on **every** device with the same data.
* `ggml_backend_meta_buffer_simple_tensor(tensor, j)` at **line 476** maps a meta tensor to its
  per-device "simple" tensor through a **pointer-keyed container**:
  `buf_ctx->get_simple_tensor_container(tensor)` → `stc.simple_tensors.find(tensor)`.  It does **not**
  read `tensor->data`.
* `ggml_backend_meta_graph_compute` at **line 2119** builds/caches **per-device child graphs**:
  `bcj.nodes[i] = ggml_backend_meta_buffer_simple_tensor(node, j)` (**line 2178**) and runs them on each
  simple backend.  The cache is keyed on `cgraph->uid` (`needs_rebuild`), and the container is cleared
  and refilled on rebuild (`stc.simple_tensors.clear()` at **2158**, filled at **1391**).
* The per-device backends are the **raw CUDA backends** (`ggml_backend_meta_simple_backend(backend, j)`)
  — and those *do* have the staging hooks (`stage_buffer`, `stage_upload`, `stage_wait`, `stage_d2d`,
  `stage_h2d_gbps`), each with its own ring in its own `ggml_backend_cuda_context`.

## 4. The trap — read this before writing code

**Forwarding the five hooks to the meta backend does not work, and neither does the scheduler's
existing redirect.**

* The scheduler's redirect (`stage_mode == 1`) points **`input_cpy->data`** at the ring slot.  Under
  tensor split `input_cpy` is the *meta* tensor; the ops never read it — `graph_compute` reads the
  per-device simple tensors, located by the pointer-keyed container, not by `data`.  So the redirect
  would be a silent no-op **at best**, and at worst it desynchronises the meta mapping.
* A single ring slot on one device cannot hold a spliced upload anyway; the data must land on N devices
  at N offsets.

So: **the ring must live inside the meta backend, per device, and the redirect must be applied to the
per-device simple tensors.**  Do not return a non-NULL `stage_buffer` from the meta backend — the
scheduler would take the redirect path above and corrupt nothing visibly while doing nothing useful.

## 5. Recommended design

Meta-level ring, reusing the per-device CUDA rings that already exist:

1. **`ggml_backend_meta_set_tensor_async`** — for an eligible whole-tensor upload (see the gate, §6),
   instead of the direct splices: for each simple backend `j` with a non-zero chunk, call that simple
   backend's own hooks —
   `simple_j->iface.stage_buffer(simple_j, slot_j, chunk_size_j)`,
   `stage_wait(simple_j, free_ev_j)` (slot reuse),
   `stage_upload(simple_j, dst_j, src + offset_j, chunk_size_j, done_ev_j)`,
   then save `simple_j->data` and point it at `dst_j`.
   Keep a small meta-level table: `{simple_tensor, slot, orig_data, done_ev, free_ev}` per device.
   A `MIRRORED` axis stages the same whole tensor on every device (same size, N slots).
2. **`ggml_backend_meta_graph_compute`** — before running the child graphs, wait each device's stream on
   its slots' `done_ev`; after they are launched, record each `free_ev` on that device's main stream
   (same pattern as `wip/h2d-staging-ring/h2d-stage.patch`'s redirect branch, and the same reason the
   free is recorded *after* the compute).
3. **Restore** `simple_j->data = orig_data` before any write that is not a staged upload (the
   `set_tensor_async` entry, `graph_optimize`, teardown), and never leave a slot pointer live across a
   call.
4. **Gate** — the scheduler owns the width decision and the calibrated `min_tokens`
   (`sched_stage_min_tokens`).  `set_tensor_async` does not know the token count, so either
   (a) add a small scheduler→backend hook so the meta backend can accept/reject per batch (cleanest), or
   (b) give the meta backend a `stage_h2d_gbps` that delegates to simple backend 0 and let it gate on
   `size` only for now.  (a) is the recommended shape; decide it before writing the ring, because it
   changes the `ggml_backend_i` contract and the meta/meta-device designated initializers.
5. **Sanity note:** whole-tensor by construction — the ids are not known before the router, so the
   scheduler's whole-tensor staging path is the only one that can be overlapped (`README.md` §5b).

## 6. Risks / audit list for the next session

* **Child-graph caching.** `graph_compute` caches per-device child graphs keyed on `cgraph->uid`; the
  cached nodes hold *simple tensor pointers* whose `data` is read at launch, so a per-ubatch redirect
  works — **but only if the per-device backend does not capture a CUDA graph**.  Prefill (the staged
  case) never captures (`ggml_cuda_graph_is_multi_token`), and decode never stages, so the same argument
  that made the single-device redirect safe should hold per device.  Verify, and add the same tripwire
  assert (a redirected simple tensor reaching a copy path must abort).
* **`n_reduce_steps` / all-reduce.** A partial split inserts reductions between the child graphs; the
  wait/free ordering must be relative to the *device stream that reads the slot*, not to the reduce.
* **`MIRRORED` vs partial.** Both need staging; the chunk sizes differ (whole tensor vs per-device).
* **`n_copies > 1` (pipeline parallel).** Still unexercised by the single-device work; the meta path
  makes it more reachable.  Audit before assuming.
* **Memory.** N devices × 8 slots × (chunk size) — the per-device chunk is smaller than the whole
  tensor for a partial split, so this may be *cheaper* than the single-device case; measure it.
* **`GGML_SCHED_STAGE_MAX_MB`** is per CUDA context today; with N devices the effective budget is N×.
  Decide whether that is acceptable or whether the meta level must budget.

## 7. Environment, build, repro

* **soar** (local): 3× R9700 gfx1201, ROCm 7.14-gfx1201, `~/llama.cpp` (r9 tip + r10 + the WIP patch in
  the working tree), build `build-rocm`.  Model
  `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
* **fingon** (ssh, no GitHub SSH key — fetch over HTTPS, agent forwarding does not work):
  `~/llama-r10` (r10 applied) + `build-rocm`, gfx1100, PCIe4 x16.  **Sync it to r11 before using it.**
* **Repro of the blocker** (2 GPUs):
  ```bash
  export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
  M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  cd ~/llama.cpp
  for s in 0 1; do env HIP_VISIBLE_DEVICES=0,1 GGML_SCHED_STAGE=$s ./build-rocm/bin/llama-bench \
      -m $M -ncmoe 99 -fa 1 -p 8192 -n 1 -b 8192 -ub 2048 -sm tensor 2>&1 | sed -n 's/.*pp8192 *| *\([0-9.]*\).*/\1/p'; done
  # add -v to see the "staging inactive" warning
  ```
* **Build gotchas (cost hours last session — READ `README.md` §4 too):** the HIP build emits no `.d`
  files, so ccache silently reuses stale objects after a `common.cuh` edit; `rm` the specific
  `ggml-cuda/*.o` before rebuilding, and treat any `common.cuh` layout change as requiring a full CUDA
  rebuild.  `cudaEventCreate`/`cudaEventElapsedTime` are **not** aliased on this toolchain (use
  `…CreateWithFlags`, or time a synchronous copy).  `__CUDA_ARCH__` cannot be used to split the HIP
  host/device passes in `common.cuh`.  Build with
  `BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714` (it `rm -rf`s the dir; ccache makes that cheap).
  `GGML_LOG_INFO`/`WARN` are suppressed below `-v`.

## 8. Gate battery + promotion checklist

Re-run after the `-sm tensor` work (all were green on the single-device ring, so a diff against these is
the test):

1. `test-backend-ops -o FLASH_ATTN_EXT` (2/2) and `-o MUL_MAT_ID` (**2/2 — this is the r11 fix; if it
   fails, you are on a stale tree**).
2. `W=1..8` width purity (f16 + q8_0, 4B) — pure, and `GGML_SCHED_EVENTS` purity-neutral.  Tool:
   `archive/work/issue-30-mtp-decode-regression/tools/width-matrix.cpp`, built against
   `build-rocm/bin` (`-I include -I ggml/include -lllama -lggml -lggml-base`).
3. Greedy same-seed text, staging off vs on, on both boxes and both models.  Soar 4B/35B references:
   `sha=6541eadb9041` / `e7e29d5a470a`; fingon Q3_K_M `82fe7fa67e03`, Q4_K_M `9fd6b0048aec`.
4. MTP `draft-mtp n3` (27B Q8_0, 2 GPU) text byte-identical (`ce64c8ed4974`) + acceptance.
5. Deep-context prefill (~32k prompt, `-ncmoe 99`, ub 2048) — pure and faster.
6. Concurrent server soak (3 × ~20k prompts × 6 rounds, ub 4096) — 18/18, clean log.
7. **New:** the same as 1-6 with `-sm tensor` (2 GPUs) once the ring engages there, plus the tensor-split
   purity gate used elsewhere in this repo.
8. Promotion: fold into the owning delivery block — **block 15** owns the `fattn_stage` bounded-arena
   precedent and **block 11** the CUDA prefill-graph skip; the scheduler contract change (§5.4) is the
   part that decides.  Then regenerate from a canonical r11 rebuild:
   `scripts/make-patches.sh <fork> 84e76d8a2 <new-tip>` (it now also re-cuts `rdna-boosts-all.patch`),
   `scripts/make-release.sh`, `scripts/validate-set.sh`, tag.

## 9. Reference

| thing | where |
|---|---|
| the merged WIP patch | `wip/h2d-staging-ring/h2d-stage.patch` (applies on r11) |
| PR #51 (the reporter's ring) | `wip/h2d-staging-ring/reporter-h2d-ring.patch` |
| evidence record | `wip/h2d-staging-ring/README.md` |
| tracker item | `TODO.md` item 24 |
| scheduler side | `ggml/src/ggml-backend.cpp` — `sched_stage_issue`, `sched_stage_restore`, `sched_stage_min_tokens`, `stage_mode` |
| backend contract | `ggml/src/ggml-backend-impl.h` — the five `stage_*` hooks |
| CUDA side | `ggml/src/ggml-cuda/common.cuh` — `h2d_stage[]`, `h2d_stage_buffer`, `h2d_stage_budget`, `copy_stream()`; `ggml-cuda.cu` — the `ggml_backend_cuda_stage_*` implementations and the `stage_h2d_gbps` calibration |
| knobs | `GGML_SCHED_STAGE`, `GGML_SCHED_STAGE_MODE` (1 redirect default / 0 D2D), `GGML_SCHED_STAGE_SLOTS` (8), `GGML_SCHED_STAGE_MAX_MB` (2048), `GGML_SCHED_STAGE_MIN_TOKENS`, `GGML_SCHED_EVENTS` |
