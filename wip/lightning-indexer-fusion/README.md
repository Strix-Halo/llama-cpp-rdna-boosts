# WIP — lightning-indexer fusion convergence (RDNA)

**Created:** 2026-10-05, after the `v16-a55e952b8-r1` re-base.
**For:** a follow-up session (fresh context).
**Delivery state:** `~/llama-cpp-rdna-boosts` `main` = release `v16-a55e952b8-r1`; fork
`~/llama.cpp` branch `rdna-boosts` tip `def454e4c`, base `a55e952b8`.
**Status:** **NOT part of the delivery.** This tree is investigation/integration only. Do not apply
it to `patches/` or the fork without a promotion decision.

---

## Why this WIP exists

The r1 re-base pulled in upstream's own qwen4exp work (`889edf43d` indexer-score memory,
`4e2713c16` mask construction, `66e0c17ee` fix qwen4exp, `c061df198` add MTP, `10f340d1a` re-enable
`-sm tensor`, `185103dcf` `llama_prefetch_rows`) next to the fork's RDNA-specific QSA pipeline. The
two indexer implementations now coexist in the tree:

| | ours (RDNA fast path) | upstream (general) |
|---|---|---|
| score | `ggml_indexer_score` / `ggml_indexer_fill` (`indexer-score.cu`); `ggml_lightning_indexer` for the prefill WMMA arm | `ggml_lightning_indexer` + `ggml_top_k` |
| selection | `ggml_indexer_top_k` (`indexer-topk.cu`, radix) | `ggml_top_k` |
| attention | `ggml_flash_attn_qsa` / `fattn-qsa3.cu` | generic sparse `build_attn_mha(..., n_kv_max)` |
| pool | derived block-vector cache (`set_input_qsa` / `get_pool`, `llama-memory-hybrid-idx.*`) | generic kpool (`set_input_kpool` / `kpool_access` / `build_qsa_sel`) |
| backends | CUDA (+ CPU oracle) | CUDA, CPU, Metal, Vulkan, SYCL |

Our implementation is the intended specialization for this repo — this WIP is about **taking
upstream's good ideas into it**, not about replacing it with upstream's graph. See
`wip/qwen4exp-qsa-convergence/` for the model-level counterpart.

## Where ours is already ahead — do not regress

- **4-head WMMA** (`supports_indexer4`, `ggml/src/ggml-cuda/lightning-indexer.cu`). Upstream's
  4-head support (`889edf43d`) is the **vec** kernel only ("too few for a wmma tile").
- **Fused radix `indexer_top_k`** vs upstream's generic `ggml_top_k` + `ggml_get_rows`.
- **Fused `indexer_score`/`indexer_fill` + derived block-vector cache** (decode re-pool
  elimination). Upstream re-pools through the kpool every step.
- **F16 pool mask** on the `ggml_lightning_indexer` call (`score_wmma_mask()` is already
  `GGML_TYPE_F16`). Upstream's F32→F16 change is a no-op for us.

## Items to investigate / integrate (ranked)

### 1. `llama_prefetch_rows` in the PLE n-gram path (highest value)

**What upstream did:** `185103dcf` added `llama_prefetch_rows(tensor, idx, n)` (mmap `madvise`
prefetch, routed through `llama-impl`/`llama-mmap`) and a `model.can_prefetch` set, and called it in
`llm_graph_input_qwen4exp_ple::set_input()` (and gemma4) **before**
`ggml_backend_tensor_set(rows, ...)`.

**Why it matters here:** the user-reported cost exactly: the lazy PLE reader re-reads
`per_layer_token_embd` rows from disk during prefill; benchmarks must use `-lm none` to avoid it.
Our fork has the managed `lazy_reader->gather()` path (`LLAMA_LAZY_IO_THREADS`, batched cold-page
fetch) plus a plain `ggml_backend_tensor_set` path, but **neither prefetches ahead**.

**Task:** call `llama_prefetch_rows(ple, idx.data(), idx.size())` in our PLE `set_input` (both the
managed and the non-managed branch, when `model.can_prefetch.count(ple)`), before the row upload.

**Acceptance:** `llama-bench` qwen4exp Flash-Next IQ4_NL 3-GPU `-lm auto` (or `-lzm on`) prefill
variance drops toward the `-lm none` result (`pp512 ~1448 ± 16`); no correctness change (same-seed
hash `359ff4337837`). A/B with the prefetch call removed.

**Pointers:** `src/models/qwen4exp.cpp` (`llm_graph_input_ple::set_input`, `build_inp_ple`);
upstream `git show 185103dcf`; `src/llama-mmap.cpp` (prefetch), `src/llama-model.h` (`can_prefetch`).

### 2. Register the score as `LLM_FUSED_OP_LIGHTNING_INDEXER`

**What upstream did:** `889edf43d` calls
`res->add_fused_node({LLM_FUSED_OP_LIGHTNING_INDEXER, score, il})` for the indexer score.

**Why it matters here:** `llama_context::resolve_fused_ops()` (`src/llama-context.cpp`) walks
`get_fused_nodes()` and **warns when a fused op lands on a different device than its layer** ("usually
due to missing support"). Our fork registers only `LLM_FUSED_OP_DSV4_HC_PRE/POST`, so we get no
diagnostic when the indexer (with its arch/type/head-count gates) silently falls off the layer's
device.

**Task:** register `LLM_FUSED_OP_LIGHTNING_INDEXER` for the score node in our
`build_qsa_top_k`/`build_qsa_store_k` path. (The `llm_fused_op_lid_probe` already exists in
`llama-context.cpp`.)

**Acceptance:** a forced-indexer-device-mismatch probe (e.g. run a qwen4exp layer on a device without
the 4-head arm) emits the "is assigned to device ... (usually due to missing support)" warning;
default runs are unchanged.

**Pointers:** `src/llama-graph.h` (`LLM_FUSED_OP_LIGHTNING_INDEXER`), `src/llama-context.cpp`
(`llm_fused_op_lid_probe`, `resolve_fused_ops`), `src/models/qwen4exp.cpp` (our `add_fused_node`
sites at the HC fusions).

### 3. Per-head accumulate-into-one-score in the non-WMMA prefill chain

**What upstream did:** `889edf43d` replaced the all-heads product + rectified copy with a loop that
computes each head's product, rectifies, and sums **into one `[n_pool, n_tokens]` f32 score** (the
allocator runs `ggml_relu`/`ggml_add` in place; the `_inplace` variants are not needed).

**Why it matters here:** our WMMA prefill arm uses `ggml_lightning_indexer` (one score) and the decode
path uses the fused `ggml_indexer_score` (one score) — both already at one buffer. But our **fallback
chain** (`build_qsa_top_k`'s non-WMMA branch, the gfx1100 issue #59/#60 path) still materializes the
all-heads product `[score_blocks, n_idx_h, n_query, n_stream]` + relu, then `head_sum`. That is the
path where peak score memory bites at deep context.

**Task:** rewrite the chain fallback to upstream's per-head accumulate-into-one-score; drop the
now-redundant chunked `score_mem` branch if the per-head loop subsumes its peak.

**Acceptance:** deep-context gfx1100 (or `GGML_CUDA_FA_WMMA_256=0`) prefill peak compute-buffer
reduced; same-seed text byte-identical (`LLAMA_QSA_SCORE_WMMA=0` A/B); `pp150000`/`pp98304` not
regressed. `test-backend-ops -o INDEXER_SCORE` / width probe green.

**Pointers:** `src/models/qwen4exp.cpp` (`build_qsa_top_k` chain branch, `head_sum`, `score_mem`);
upstream `git show 889edf43d -- src/models/qwen4exp.cpp`.

### 4. Seed-free mask construction

**What upstream did:** `4e2713c16` replaced the `cast(view(sel_idx, 1))` seed + fill/repeat with
`ggml_new_tensor_4d(...) + ggml_fill + ggml_repeat_4d`, and hoisted
`ggml_build_forward_expand(gf, sel_idx)` so the scatter storage frees earlier.

**Why it matters here:** our main `build_attn_qsa` mask path already uses a different scheme
(`ggml_fill(kq_mask, -INFINITY)` + `ggml_set_rows`), but our prefill block-mask build may still carry
a seed tensor. Removing it drops an allocation and a dependency edge.

**Task:** diff our prefill mask build against upstream's; adopt the seed-free form where we still use
a seed.

**Acceptance:** same-seed text byte-identical; `llama_get_memory_breakdown` compute-buffer peak not
worse.

**Pointers:** `src/models/qwen4exp.cpp` (`build_qsa_top_k` mask build); upstream
`git show 4e2713c16`.

### 5. Structural: feed the standard FA kernels an index list (biggest leverage)

**What upstream does:** upstream's sparse QSA calls `build_attn_mha(..., n_kv_max)` on the normal
tile/MMA/vec FA kernels, which accept a top-k cell list. Our `FLASH_ATTN_QSA` is a **parallel FA
implementation** (`fattn-qsa.cu`, `fattn-qsa3.cu`) with its own CPU oracle and its own bug history
(width purity, head-chunking, quant-type gates, issue #89).

**Why it matters here:** if our top-k selection fed the *existing* FA kernels, the sparse path would
inherit the entire RDNA FA investment for free — the RDNA4 GQA-6 decode/verify band, the V3 derived
kq mask, the V4/V5 native KV dequant, the per-arch head caps, the width-purity guarantees, and the
`FLASH_ATTN_EXT` (6358/6358) coverage — instead of maintaining and re-validating a second kernel.

**Task (large, separate campaign):** investigate presenting the `indexer_top_k` cell list to
`fattn-tile`/`fattn-mma-f16`/`fattn-vec` via the `n_kv_max`/`i_sup` sparse mechanism; keep the fused
`FLASH_ATTN_QSA` kernel as the RDNA-specialized fast path if it still wins, but make the standard
sparse path the correctness/coverage fallback (dense is already the fallback for non-native types).

**Acceptance:** `test-backend-ops -o FLASH_ATTN_EXT,FLASH_ATTN_QSA` green; qwen4exp same-seed hash
`359ff4337837` for both the fused and standard-sparse arms; a measured decision on which arm stays
default per arch.

**Pointers:** `ggml/src/ggml-cuda/fattn.cu` (chooser), `fattn-tile.cuh`, `fattn-mma-f16.cuh`,
`fattn-qsa.cu`; upstream `f11d642a2`/`36d7b0834` (FA tuning) and `66e0c17ee` (`build_attn_mha` with
`n_kv_max`).

## Non-goals

- Porting the QSA graph to Vulkan/Metal/SYCL (out of scope for this repo).
- Replacing our fused top-k/derived cache with upstream's generic ones (that is
  `wip/qwen4exp-qsa-convergence/`'s decision, and the default is to keep ours).

## Gates to run for any change here

```
test-backend-ops -o INDEXER_TOPK,INDEXER_SCORE,FLASH_ATTN_QSA,HC_MIX,FLASH_ATTN_EXT
4B same-seed: 1c5d32ac537d
qwen4exp Flash-Next IQ4_NL 3-GPU -sm tensor -lm none: 359ff4337837, pp512 ~1448
```
