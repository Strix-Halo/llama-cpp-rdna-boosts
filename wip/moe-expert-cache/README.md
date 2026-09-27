# Decode-side MoE expert caching: high-speed expert decode for models that do not fit

**Status (2026-09-27): NEW CAMPAIGN — reconnaissance landed, no code yet.**  The prefill sibling
(`archive/work/tensor-split-expert-split/`, delivery release `v16-84e76d8a2-r16`) is **closed**: its goal
("prefill wins under `-sm tensor` with host-resident experts") is delivered.  This campaign is the decode
half of the same story.

**Not a blocker for anything.**  An optimisation campaign; nothing in the delivery depends on it.

---

## 0. HANDOVER BRIEF — read this first (cold start)

### Goal

Make **decode (and speculative verify) fast for MoE models whose expert weights do not fit in VRAM**, by
keeping a **hot set of experts resident on the device** and handling the cold set cheaply — without giving
up the **tensor split** (`-sm tensor`) that the delivery now makes the best prefill configuration.

The end goal is **Qwen3.8-Flash-Next** (176 B params, 93 GiB at IQ4_NL, `qwen4exp`, 48 layers × 512
experts) running at useful decode speed on 2× R9700 — a model that does not fit 64 GiB and today decodes
slowly.  The iteration target is **Qwen3.6-35B-A3B Q8_0** (37.8 GiB, does not fit one 32 GiB card),
because it is faster to load and test.

### The gap (measured 2026-09-27, this box)

`llama-bench -p 0 -n 64 -fa 1`, Qwen3.6-35B-A3B **Q8_0**, gfx1201:

| config | `tg64` t/s |
|---|---:|
| 2 GPU `-sm layer -ncmoe 0` (all experts **resident**) | **79.7** |
| 1 GPU `-ncmoe 20` (half the expert layers on host) | 36.4 |
| **1 GPU `-ncmoe 99` (all experts host) — the campaign's baseline** | **24.2** |
| 2 GPU `-sm layer -ncmoe 99` | 22.5 |
| 2 GPU `-sm tensor -ncmoe 99` | 21.5 |

**Why all-host decode is slow: the MoE is not on the GPU at all.**  `ggml_backend_cuda_device_offload_op()`
(`ggml/src/ggml-cuda/ggml-cuda.cu`) only offloads an op when
`get_op_batch_size(op) >= op_offload_min_batch_size` (default **32**, `GGML_OP_OFFLOAD_MIN_BATCH`).  For
`GGML_OP_MUL_MAT_ID` that size is `op->ne[2]` = **`n_tokens`**, which is **1 at decode** — so the decode
graph's `ffn_moe_gate/up/down` are assigned to the **CPU**, in every split mode.  Prefill (ubatch ≥ 32)
offloads; decode does not.  So the 24.2 t/s is a **CPU MoE** number, and every extra resident layer buys
throughput (see the `-ncmoe 20` row) because it moves MoE work back to the GPU.

Note the shape of the prize: **a single card with a cache can beat two cards with none** (36.4 vs 22.5),
and the resident ceiling is 3.3× the baseline.  A working cache does not have to reach 79.7 to be a large
win; it has to move the decode MoE back onto the GPU for the experts that fit.

### Prior art — the baseline starting points (all read 2026-09-27)

Three implementations, summarised in the closed prefill campaign's
`archive/work/tensor-split-expert-split/README.md` §26-§28 (read those sections — they are the
requirements document, with exact file paths):

1. **GenerelSchwerz `llama.cpp` `moe-cache`** (`ggml/src/ggml-cuda/moe-cache.cu`, ~14k lines) — the mature
   **layer-split** design and the closest match to our stack.  Per-layer, per-owner (per-GPU) cache buffer
   with N expert slots; **grouped decode kernels** compute the resident experts on the GPU; misses are
   staged/uploaded; prefetch, replacement policy, an **early router**, and CUDA-graph capture.  Measured
   **Qwen3.6 35B decode 42.8 → 111.6 tok/s (2.6×)** at the same peak VRAM on a 16 GB RTX 5070 Ti.  Its host
   side (`moe-cache-host.cu`) is the mature version of our pinning/staging work: a `moe_host_source`, a
   bounded **pinned staging budget**, automatic expert-group registration, pageable fallback with pinned
   staging, and a dedicated **asynchronous copy worker thread**.  **It explicitly refuses `-sm tensor`**
   ("the tensor-mode meta backend cannot consume the cached buffer"), so it is a *layer-split* reference.
2. **R9V** (`github.com/Dyluhn/R9V`, local `~/R9V`) — **the prior art that DOES tensor-split offloaded
   experts**, on this exact 2× R9700 hardware, for Qwen3.8-Flash-Next.  It keeps
   `get_tensor_model_parallel_rank()`-sharded expert masters (each rank holds `1/tp` of **every** expert —
   i.e. exactly our `-sm tensor` split), a per-rank **hot/cold manifest** (`hot_experts_by_layer`, from
   routing calibration, `materialize_hot_expert_cache`), **cold shards in pinned UVA host memory read in
   place** (`allocate_uva_host_empty` + `parameter.data = get_accelerator_view_from_cpu_tensor(...)`), and a
   VRAM **slot cache** on top (`QWEN38_TIERED_EXPERT_CACHE_SLOTS` ≤ 128, `second_touch_rr`/`lru`,
   optional async fill).  **Key lesson: R9V does not win by making the upload fast — it avoids the upload**
   (resident hot set + UVA in-place cold reads).  Reference files:
   `runtimes/qwen38-flash-next-gfx1201-v1/retained-mtp4/python/vllm_gguf_plugin/quantization/tiered_experts.py`,
   `docs/config.md`, `crates/r9v-ir/src/plan.rs` (`ExpertPlacement::{Device, HostCompute, HostFetch}`).
3. **Strata** (`github.com/Niko1221/Strata`, local `~/Strata`) — single-card, **CPU computes the misses**
   concurrently with the GPU, plus a frequency-profiled VRAM cache (`src/core/expert_cache.cpp`).  Its
   bottleneck is the **CPU expert pool** (663.6 MB/token at ~40 GB/s), not the H2D — their fix is to stop
   reading bytes.  No split mode at all; least transferable.

The synthesis: **the proven decode lever is a hot-expert VRAM cache**; the open question is whether it can
be made to work under our **meta backend / `-sm tensor`**, which is exactly what R9V says is possible and
what nobody in the llama.cpp ecosystem has built (moe-cache refuses).

### The phased plan

Work strictly left to right; each phase must show a measured decode win over its own baseline before the
next starts.

* **Phase 0 — baseline & instrumentation** (partly done, above).  Sweep `tg` across `-ncmoe 0..40` for
  1 GPU and 2 GPU, `-sm layer` and `-sm tensor`; confirm the CPU-vs-GPU MoE assignment; get a repeatable
  decode harness (the delivery's decode protocol is `d16384`, but shallow is fine for iteration — record
  depth with every number, §"Measurement protocol").
* **Phase 1 — single GPU, Qwen3.6-35B-A3B Q8_0.**  Build a per-layer VRAM **expert slot cache** on one
  card and get `MUL_MAT_ID` decode back on the GPU.  Start with a **static** resident set (no routing
  profile) and a simple policy; measure how throughput scales with the resident fraction.  This is the
  iteration vehicle: one card, one model, fast builds, the smallest surface.
* **Phase 2 — 2 GPUs, `-sm layer`.**  The cache is naturally per-device (a layer split assigns whole layers
  to a device), so this is the moe-cache-shaped case and should be the easiest multi-GPU step.
* **Phase 3 — 2 GPUs, `-sm tensor`.**  The hard and novel one: each device holds a **slice** of every
  expert (axis-1 gate/up, axis-0 down), so a "resident expert" is a resident **per-device slice**, and the
  down projection still needs its cross-device partial reduce.  R9V is the map.  This is the phase that
  unlocks the end goal.
* **Phase 4 — Qwen3.8-Flash-Next IQ4_NL** (93 GiB, `qwen4exp`/QSA).  Re-validate everything on the real
  target; note its PLE tables are host-resident and lazy-loaded (`--lazy-mode auto`), and pre-warm matters.

### Design space (decide by measurement, in this order)

* **A. Hot set in VRAM + stream the misses** (moe-cache).  Reuses what the delivery already has — the
  pinned host expert source (`LLAMA_MMAP_HOST_EXPERTS`), the block-06 staging ring, the op-offload
  `copy_experts` pruning.  Natural first implementation, and it works for `-sm layer` immediately.
* **B. Hot set in VRAM + cold experts read in place over UVA** (R9V).  Pin the host shards once and map
  them into the device address space (`hipHostRegister` + `hipHostMallocMapped`/`cudaHostGetDevicePointer`),
  then have the expert kernel read cold experts straight from host memory — **no per-token H2D at all**.
  This is the end state that avoids the transfer rather than hiding it, and the one that composes with the
  tensor split.  llama.cpp's op-offload does not use UVA today.
* **C. CPU computes the misses** (Strata).  Useful as a fallback for boxes with no spare VRAM, but it
  fights the CPU-pool bandwidth limit and has no split story; keep it as a comparison, not the plan.

### Where the work lives (code hooks)

| concern | location |
|---|---|
| op-offload gate (why decode is on CPU) | `ggml_backend_cuda_device_offload_op()` + `get_op_batch_size()` in `ggml/src/ggml-cuda/ggml-cuda.cu` (default 32, `GGML_OP_OFFLOAD_MIN_BATCH`) |
| the MoE op | `GGML_OP_MUL_MAT_ID`; decode kernel `mul_mat_vec_q_moe` / `ggml_cuda_mul_mat_id` |
| used-expert pruning (host→device ranges) | `ggml_backend_sched_compute_splits`'s `copy_experts` in `ggml/src/ggml-backend.cpp` |
| host-expert pinning (r15) | `select_weight_buft` in `src/llama-model-loader.cpp` (`LLAMA_MMAP_HOST_EXPERTS`) |
| H2D staging ring (r16) | `sched_stage_*` (`ggml-backend.cpp`), `stage_*` hooks (`ggml-backend-impl.h`, `ggml-cuda.cu`, `ggml-cuda/common.cuh`) |
| the split upload (r16) | `ggml_backend_meta_stage_input` + `ggml_backend_meta_set_tensor_async` (`ggml/src/ggml-backend-meta.cpp`) |
| per-tensor split policy | `llama_meta_device_get_split_state` (`src/llama-model.cpp`) |
| existing MoE helpers | `ggml/src/ggml-cuda/topk-moe.cu`, `moe-weighted-reduction.cu`, `ggml/src/models/qwen4exp.cpp` |

### Open questions / risks

1. **Where does the cache live** — a new `ggml_cuda` buffer type (moe-cache) vs the meta backend's simple
   tensors (needed for `-sm tensor`).  The meta backend "cannot consume a cached buffer" today; that is the
   Phase-3 problem to solve.
2. **Residency decision** — static vs routing-level hot set.  The maintainer's earlier steer was "no
   explicit hot-weight tuning" for prefill; for decode a **measured** hot set is the proven 2.6× (moe-cache,
   R9V), so expect to need one.  Start static, then add a routing profile.
3. **Decode/verify purity** — the delivery's `W=1..8` width-purity rule (`GREEDY-PURITY.md`) means the
   cache must produce the *same* arithmetic at every verify width; a grouped/split-by-residency kernel is
   exactly the kind of change that breaks it.  Gate every step with the plain-vs-`draft-mtp` purity check.
4. **Speculative decode** — MTP drafts `n` tokens per step; the used-expert set and the cache hit rate must
   be evaluated at the **verify width**, not just `n_tokens = 1` (the delivery's `MMVQ_MAX_BATCH_SIZE`/
   `MMVF_MAX_BATCH_SIZE` banding).
5. **VRAM budget vs KV** — the cache competes with KV cache for VRAM; the delivery's `--fit` does not know
   about it (same trap as the r3 staging-arena OOM, issue #33).  Any cache allocation must fail soft.
6. **`-sm tensor` down projection** — axis-0 split, `nb[1]`-sized chunks (~524k tiny blocks); any per-token
   upload must be device-side or 2-D (the prefill campaign's §26.3/§29).  This is why UVA (B) is the
   promising `-sm tensor` answer.

### Follow-ups inherited from the prefill campaign (do these as part of this campaign)

These are the loose ends the prefill campaign did **not** close; they are now owned here, cross-referenced
from `archive/work/tensor-split-expert-split/README.md` (§30.5-§31):

1. **Prune the staged upload to the used experts.**  The r16 staged path uploads the **whole** expert
   tensor (all experts, half per device) regardless of ubatch, bypassing the used-expert pruning; at
   ub 512 that is ~10× the volume the pruning path moves.  The split's compute floor is ~7000 t/s and the
   staged path reaches ~5400 at ub 8192, so there is room.  (This also matters for decode if the staged
   mechanism is reused there.)
2. **`GGML_META_GATHER_MODE` device-count heuristic.**  The auto choice (host gather vs device D2D for the
   fine `ffn_down` slice) **inverts** with device count: at 3 GPUs the host gather wins at ub 8192
   (4905 vs 4694), while at 2 GPUs auto (D2D) wins.  Make the heuristic device-count-aware.
3. **3-GPU ub-8192 split loss.**  The split is −5.6 % vs mirrored at ub 8192 on 3 GPUs (the quant block
   limits the split to 2-of-3 devices; the loss is in the upload path).  Close it or document it.
4. **Strip the campaign's env-gated debug/A-B knobs from the delivery.**  The r16 promotion carries the
   debug instrumentation (`GGML_META_STAGEDBG`, `GGML_CUDA_GCDBG`, etc.) alongside the user-facing
   kill-switches; fold them out before any `upstream/` PR candidate is cut.  (Recorded in `WORKLOG.md`
   r16 and `patches/README.md`.)
5. **The §31 prefill residency idea is superseded** by this campaign's Phase 3/UVA design — do not build a
   separate prefill-only residency path; make the cache/UVA design serve both if it can.
6. **`-ncmoe` decode on the CPU** is the root observation of this campaign (see "The gap"); it is not a
   "follow-up" so much as the problem statement.

### Environment, build, and repro

* **Build tree for this campaign:** create a fresh worktree for the new work — the r16 delivery tree is
  `~/llama-promote` (branch `r16-tip`, tip `92b14a61`, tree `46a5a43d`), and a clean upstream is
  `~/llama-upstream` at `84e76d8a2`.  Recommended: `git -C ~/llama.cpp worktree add ~/llama-decode -b
  wip-moe-expert-cache <r16-tip>` (or off `~/llama-promote`'s branch), then build with
  `BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714` (~7 min cold,
  ccache warm after).
* **Iteration models:** `/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` (37.8 GiB — the
  Phase-1 vehicle) and `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/…UD-Q4_K_M.gguf` (21.1 GiB — fast smoke tests,
  fits one card).
* **End-goal model:** `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-0000N-of-00009.gguf`
  (93 GiB, `qwen4exp`); its shipping `runme` uses `--lazy-mode auto` + the shared MTP head, and **PLE
  pre-warm matters for its absolute numbers** — do not compare `llama-bench` absolutes for it without the
  server/lazy path.
* **Repro of the baseline gap:** `HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-bench -m <Q8_0> -ncmoe 99 -fa 1
  -p 0 -n 64 -sm tensor -r 3` → ~24 t/s; `GGML_OP_OFFLOAD_MIN_BATCH=0` puts the decode MoE back on the GPU
  for a *correctness* look (it will be slow while every expert streams per token).
* Runtime libs come from the binary's RUNPATH (`/opt/rocm-7.14.1-gfx102X/lib`); `~/llama-r12` is the old
  prefill campaign's tree — use `~/llama-promote`/a fresh worktree instead.
* **Never run parallel benches**; pin nothing; record depth with every number.

### Measurement protocol (do not skip)

* Decode must be measured **at depth** as well as shallow (`d0` and, for the delivery gate, `d16384`):
  "everything is fast at depth 0" is the delivery's standing rule.
* Always gate a decode/fusion change with the **MTP** methodology
  (`benchmarks/mtp-adaptive-methodology.md`) and the **width-purity** check
  (`--spec-type none` vs `--spec-type draft-mtp`, `W=1..8` bit-identical), per `GREEDY-PURITY.md`.
* Same-seed greedy text must be identical to the pre-change build; `test-backend-ops -o MUL_MAT_ID` green.

### Artifacts in this directory

* `README.md` — this brief (the live notebook from here on; append dated findings, never edit them away).
* `decode-baseline.csv` — the Phase-0 `tg` sweep to beat (added with the first measurement session).

### Immediate next steps (fresh session)

1. Build the tree (§Environment) and reproduce the gap table above (`-p 0 -n 64`, Q8_0, 1 GPU, `-ncmoe
   {0,20,32,40,99}`) as the campaign's `decode-baseline.csv`.
2. Read the three prior-art sections (`archive/work/tensor-split-expert-split/README.md` §26-§28) and pick
   the Phase-1 kernel/cache shape (recommended: A, reusing the pinned host source + staging ring).
3. Prototype **Phase 1** on one GPU: a static-resident expert slot cache + a decode path that keeps
   `MUL_MAT_ID` on the GPU for resident experts, with the misses on the existing host path.  Measure
   throughput vs resident fraction.  Fail soft on VRAM (issue #33 lesson).
4. Only then move to Phase 2 (`-sm layer`, 2 GPU) and Phase 3 (`-sm tensor`, 2 GPU = the R9V case).
