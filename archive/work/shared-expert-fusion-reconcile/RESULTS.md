# shared-expert fusion reconciliation — RESULTS (resolved in `v16-a55e952b8-r3`)

**Resolved:** 2026-10-05.  **Decision: keep all three fusions; they operate on disjoint graph
segments.**  Recorded in a code comment in block 13 and in `patches/README.md`.
**Original handover:** [`README.md`](README.md).
**Delivery change:** yes — one clarifying comment in `ggml/src/ggml-cuda/ggml-cuda.cu` (block 13);
no functional change.

## The three (four) shared-expert paths and their node sets

| # | path | matches | writes |
|---|------|---------|--------|
| 1 | upstream `bed0a8566` `ggml_cuda_match_shared_expert` | `{MUL_MAT_ID, MUL_MAT_ID, GLU, MUL_MAT, MUL_MAT, GLU}` — routed gate/up/GLU **and** shared gate/up/GLU | routed GLU (`routed`), shared GLU (`shared_dst`) |
| 2 | block 13 `ggml_cuda_op_shexp_down_gate` | `{MUL_MAT(down), MUL_MAT(gate_inp), SIGMOID, MUL, ADD, ADD}` — the shared-expert **down + gate_inp epilogue** | final `l_out` (`down(swiglu)*sigmoid(gate_inp) + moe_out + ffn_residual`) |
| 3 | MMB `LLAMA_HC_BLK16` MWR merge (`ggml_cuda_op_moe_weighted_reduction`) | the routed weighted reduction `MUL` and the `ADD(reduction, ffn_shexp_gated)` | the reduction output (BF16 block_out), folding the shared `ADD` |
| 4 | block 13 `ggml_cuda_mul_mat_q_pair` | two consecutive `MUL_MAT(_ID)` sharing src1/ids (routed gate+up, dense up+gate) | both matmul outputs (prefill) |

(1) and (2) are **disjoint**: (1) consumes the shared gate/up/GLU nodes, (2) consumes the shared
down and `gate_inp` nodes that come after them.  (3) is default-off and, when armed, deliberately
consumes the `ffn_out = ADD(moe_out, gated)` node that (2) needs, so it *replaces* (2) for the
routes MMB takes; it is not a competing default.  (4) is the prefill activation-quantize sharing
for the pair itself and never overlaps (1) or (2) (different node windows / dispatch bands).

## Evidence (gfx1201, 3× R9700, `GGML_CUDA_FUSE_TRACE` instrumentation, Qwen3.6-35B-A3B `Q8_0`)

A temporary one-shot trace on the two `try_fuse` arms and the `std::rotate` in
`ggml_backend_cuda_graph_optimize`:

* **`-sm layer`** (routed and shared widths both whole): `upstream_shared` **200×** and
  `shexp_down_gate` **160×** on one 2-token pass — both fire, on the same layers, disjoint node
  sets.  This is the direct non-overlap proof.
* **`-sm tensor`**: `upstream_shared` **0×**, `shexp_down_gate` still fires (600× over a 20-token
  pass).  The upstream arm is **dormant** here, not shadowed.
* Upstream's graph reorder (`std::rotate`) fires for every layer in both modes, so the matcher
  does engage at `graph_optimize` time; it is the *compute-time* shape precondition that fails
  under tensor split.

### Why (1) stands down under `-sm tensor`

The trace printed the failing check's operands:

```
up=ffn_moe_up-0  sup=ffn_up-0  w=blk.0.ffn_up_exps.weight  sw=blk.0.ffn_up_shexp.weight
w.ne=[2048,128,256]  sw.ne=[2048,512]
```

`ggml_cuda_match_shared_expert` requires `weight->ne[1] == shared_weight->ne[1]`.  Under
`-sm tensor` the routed expert weight is sharded along its FFN axis with the FFN granularity
`lcm(blk_size, 128)` (`get_split_granularity`, `src/llama-model.cpp`): for Q8_0 that is 128, so
device 0 sees a `[2048,128,256]` shard while the shared expert is the mirrored `[2048,512]`.
The widths differ and the fusion is correctly declined.  Under `-sm layer` both are whole
(`512`), so it engages.  This is a property of upstream's matcher, inherited with the base — not
something the fork's merge broke, and not something to "fix" (the fusion's `stride_col_dst`
sharing genuinely needs equal widths).

## Bit-identity / verify band

Nothing changed at runtime: the delivery ships only a comment.  The `shexp_down_gate` epilogue
stays token-generic and pinned to the single-token reduction order for the whole
`n_tokens <= MMVQ_MAX_BATCH_SIZE` band (its existing `-sm layer`/`-sm tensor` behaviour is
unchanged).  `upstream_shared` is `should_fuse_mul_mat_vec_q`-gated (MMVQ only) so it cannot
introduce a prefill/verify reduction split.

## Validation (this delivery)

* **No code change** other than the comment; `MUL_MAT_ID` **931/931**, `MUL_MAT,MUL_MAT_ID`
  **2235/2235** (both `GGML_CUDA_MMQ_PREC` values), `HC_MIX` **30/30**.
* Coherence unchanged: dense 4B `1c5d32ac537d`, qwen4exp Flash-Next IQ4_NL `359ff4337837`.
* The MWR (`LLAMA_HC_BLK16`) interaction is documented, not tested-armed here (default off; it
  needs MMB to fire, which this box's default config does not enable for these models).

## Delivery record

* `ggml/src/ggml-cuda/ggml-cuda.cu`, block 13: the "DISJOINT from upstream's fused shared-expert
  MMVQ" note at the `shexp_down_gate` matcher.
* `patches/README.md` block-13 notes.
* Release `v16-a55e952b8-r3`, tree `25a8e137a585cd9fc2907a74236998f881635b8e`.
