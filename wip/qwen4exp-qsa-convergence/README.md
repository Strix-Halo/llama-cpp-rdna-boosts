# WIP — qwen4exp QSA model-level convergence

**Created:** 2026-10-05, after the `v16-a55e952b8-r1` re-base.
**For:** a follow-up session (fresh context).
**Delivery state:** `main` = `v16-a55e952b8-r1`; fork `rdna-boosts` tip `def454e4c`, base `a55e952b8`.
**Status:** **NOT part of the delivery.** Investigation/integration only.

---

## Why this WIP exists

During the r1 re-base, upstream's qwen4exp work and the fork's fused QSA pipeline were both kept in
the tree, but at the **model level the fork's implementation is wired and upstream's is dead code**:

- `src/models/qwen4exp.cpp` is the fork's r37 version (fused `build_qsa_top_k`, `build_qsa_store_k`,
  `llm_graph_input_qsa`/`_qsa_k`, `ggml_indexer_score`/`_fill`/`_top_k`, `ggml_flash_attn_qsa`,
  derived block-vector cache).
- `src/llama-memory-hybrid-idx.{h,cpp}` (keep-both merge) contains **both** our `set_input_qsa` /
  `get_pool` / `qsa_score_key_limits` **and** upstream's `set_input_kpool` / `kpool_access` /
  `gather_mla_rows`. Upstream's kpool is used by glm5-next but **not** by our qwen4exp.
- `src/models/models.h`'s `llama_model_qwen4exp::graph` declares both the fork's QSA methods and
  upstream's `llm_graph_input_kpool` / `build_inp_kpool` / `build_qsa_sel` (declared, some undefined
  in our qwen4exp.cpp).

That is the worst maintenance position: two complete indexer/pooling stacks, one live, one dead. This
WIP is the **decision + cleanup** counterpart to `wip/lightning-indexer-fusion/` (which is about
*improving* ours).

## Decision to make (lean: converge on one)

Options:

- **(A) Keep ours, delete upstream's qwen4exp kpool dead code.** Cheapest, best RDNA perf, but keeps
  diverging from upstream every re-base and throws away upstream's newer qwen4exp fixes.
- **(B) Adopt upstream's graph** (`ggml_lightning_indexer` + `ggml_top_k` + generic sparse FA +
  kpool), then re-add our fused ops as CUDA-side accelerations. Best long-term; a re-port.
- **(C) Hybrid:** keep our fused score/top-k/FA but move the **pool storage** to upstream's kpool
  (pooled key in the indexer cache row) to drop the separate derived-cache tensor + watermarks.

The working assumption for this repo (RDNA-first) is **(A) now + (C) as the simplification**, with
**(B)** revisited only if upstream's qwen4exp perf closes the gap. Record the decision here before
touching code.

## Items

### 1. Audit upstream qwen4exp fixes we are missing

Our r37 graph predates upstream's recent fixes. Diff the two and list anything load-bearing:

- `66e0c17ee` "llama: fix qwen4exp" — large `llama-memory-hybrid-idx.cpp` rework + `llama-hparams.h`
  additions (`indexer_kpool`, `indexer_kpool_by_order`, `indexer_kpool_select_tail`,
  `indexer_kpool_row`) + `keep kq_mask input the same shape`.
- `159c651f5` "qwen4exp: fix tests".
- `c061df198` "Qwen4Exp: add MTP" — upstream MTP; ours is independent. Check for shared-infra fixes
  (`llama-memory-recurrent.cpp` `state_clear`, `build_rs`, MTP-only file handling).
- `10f340d1a` "model: re-enable -sm tensor for qwen4exp" — **see item 2**.
- `4e2713c16` / `889edf43d` — covered in `wip/lightning-indexer-fusion/`.

**Acceptance:** a written list of upstream qwen4exp changes with a keep/drop decision each, plus a
test for any adopted one.

### 2. `-sm tensor` gate reconciliation

Our block 14 has `llm_arch_supports_sm_tensor(qwen4exp)` true only on HIP (`#ifdef GGML_USE_HIP`).
Upstream `10f340d1a` re-enabled `-sm tensor` for qwen4exp on **all** backends via a graph-placement
fix (expand `hc_init` right after it is built, so the PLE gather is a CPU split input). On ROCm the
practical behaviour is the same (ours allows it), but:

- confirm our HIP gate doesn't reject a case upstream now supports on ROCm;
- decide whether to drop the gate and take upstream's placement fix (the gate was added because the
  fused QSA/HC ops' CPU-fallback subgraphs broke the meta splitter on Vulkan/Metal — if upstream's
  fix removes that, the gate can go for qwen4exp on ROCm at least);
- `wip/qwen4exp-qsa-convergence/` vs the AGENTS note "gemma4 with `-sm tensor` is rejected".

**Acceptance:** 3-GPU `-sm tensor` qwen4exp run hash `359ff4337837`; `-sm tensor` still rejected where
the fused ops genuinely can't run; documented in `patches/README.md`.

### 3. Remove or wire the dead upstream kpool

Either:
- **(A)** delete `build_inp_kpool` / `build_qsa_sel` / the qwen4exp `llm_graph_input_kpool` class
  from `qwen4exp.cpp` and `models.h`, and prune the fork's `llama_memory_hybrid_idx` of our QSA
  methods if we move to (C); or
- **(C)** port our pooling into upstream's kpool layout and delete `get_pool`/`pool_layers`/`pool_wm`/
  `derived_enabled`/`qsa_derived_limits`.

**Acceptance:** no dead qwen4exp kpool symbols in the fork; `INDEXER_TOPK/SCORE/FILL` + width probe +
`359ff4337837` all green.

### 4. Memory: derived-cache tensor vs in-cache pooled slot

Our derived cache is a per-indexer-layer `F32 [idx_dim x ceil(n_kv/r) x n_stream]` tensor
(`llama_mem_pool_layer`) plus per-layer host watermarks. Upstream's kpool stores the pooled key in
the **indexer cache row** (`indexer_kpool_row = 2`, `raw key | pooled key`) — no extra tensor. Note
upstream's slot is per **cell** (n_kv) and ours is per **block** (n_kv/r), so the comparison is not
obvious; measure both (VRAM and prefill/decode) before choosing.

**Acceptance:** a VRAM + pp/tg table for baseline vs kpool-in-cache on the Flash-Next IQ4_NL 3-GPU
gate.

## Non-goals

- Re-implementing the fused kernels (that is `wip/lightning-indexer-fusion/`).
- Cross-backend QSA support.

## Pointers

- `src/models/qwen4exp.cpp`, `src/models/models.h` (`llama_model_qwen4exp`), `src/llama-memory-hybrid-idx.{h,cpp}`.
- `git show 66e0c17ee 10f340d1a 185103dcf 4e2713c16 889edf43d c061df198`.
- `WORKLOG.md` 2026-10-05 (r1) "Block 14, qwen4exp" conflict note.
