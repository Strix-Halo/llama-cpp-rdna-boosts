# WIP — shared-expert fusion reconciliation

**Created:** 2026-10-05, after the `v16-a55e952b8-r1` re-base.
**For:** a follow-up session.
**Status:** **NOT part of the delivery.** Investigation/validation only.

---

## Why this WIP exists

The r1 re-base landed upstream's `bed0a8566` "CUDA: fuse shared experts into MMVQ"
(`ggml_cuda_match_shared_expert` + a fused routed/shared gate+up MMVQ launch) **on top of** the
fork's own shared-expert fusions from block 13 and the MMB campaign. There are now **three**
shared-expert epilogues in `ggml/src/ggml-cuda/ggml-cuda.cu` that can match overlapping qwen4exp /
qwen35moe MoE graphs:

1. **Upstream `bed0a8566` — routed+shared gate/up pair → one MMVQ.**
   `ggml_cuda_match_shared_expert(cgraph, i, i+3)` matches
   `{MUL_MAT_ID, MUL_MAT_ID, GLU, MUL_MAT, MUL_MAT, GLU}` and calls
   `ggml_cuda_mul_mat_vec_q(..., &fusion)` with `fusion.shared_up` / `shared_gate` / `shared_dst`
   (`ggml-cuda.cu` ~line 4729; matcher at ~line 2425).
2. **Fork block 13 — shared-expert down+gate epilogue (`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE`).**
   Matches `{MUL_MAT, MUL_MAT, SIGMOID, MUL, ADD, ADD}` → `ggml_cuda_op_shexp_down_gate`
   (`ggml-cuda.cu` ~line 6013-6075) and writes
   `dst = down(swiglu) * sigmoid(gate(x)) + moe_out + ffn_residual`. Token-generic, pinned to the
   single-token reduction order so `W=1..8` agrees.
3. **MMB `LLAMA_HC_BLK16` MWR merge (default OFF) — `ggml_cuda_op_moe_weighted_reduction`**
   folding the shared-expert `ADD` after the weighted reduction (`ggml-cuda.cu` ~line 4926).
4. Also in the area: **block 13's `ggml_cuda_mul_mat_q_pair`** (shared activation quantize for two
   consecutive MMQ muls) and block-13's fused gate MMQ (`ggml_cuda_mul_mat_q_switch_type_gate`).

They are *probably* complementary (different graph segments), but that has not been verified, and
the precedence/`return N` consumption of the first match decides whether the later ones ever fire.
`MUL_MAT_ID` 931/931 passes, but there is no test that asserts *which* fusion fired for a real
qwen4exp MoE layer, and no gate that the three do not double-write or shadow each other.

## Objectives

1. **Map the qwen4exp / qwen35moe MoE graph** and determine, per layer shape, which of (1)-(4) fires
   and in what order. Instrument with `GGML_CUDA_GCDBG` / the existing fusion `GGML_LOG_INFO` lines
   and the `GGML_PAIR_2X` / `GGML_CUDA_DISABLE_*` kill switches.
2. **Prove non-overlap or fix it.** If (1) consumes the shared pair before (2) can see the down
   projection, confirm (2) still fires for the standalone (non-upstream-fused) shape; if (1) and (2)
   can both match the same down MUL_MAT, make the precedence explicit and documented.
3. **Bit-identity for the MoE verify band.** The block-13 note is emphatic: the fused epilogue does
   not reproduce the unfused chain's arithmetic, so it must stay token-generic and whole-band
   (`n_tokens <= MMVQ_MAX_BATCH_SIZE`). Confirm upstream's (1) does not reintroduce a width
   dependence (it is `shared_up`-gated and MMVQ only).
4. **Decide redundancy.** If (1) subsumes part of block 13's gate+up pair fusion
   (`ggml_cuda_mul_mat_q_pair` / `ggml_cuda_mul_mat_q_switch_type_gate`) for the shared lane, retire
   the duplicate.

## Acceptance

- A trace showing, for one real qwen4exp prefill and one decode, exactly one arm per MoE with no
  double computation.
- qwen4exp same-seed `359ff4337837` and 4B `1c5d32ac537d` unchanged.
- `test-backend-ops -o MUL_MAT_ID,HC_MIX` green; MoE MTP acceptance unchanged (methodology:
  `benchmarks/mtp-adaptive-methodology.md`).
- A recorded decision (keep both / retire one) with the measured or bit-identity justification.

## Pointers

- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_match_shared_expert` (~2425), the call (~4729),
  `ggml_cuda_op_shexp_down_gate` matcher (~6013), MWR merge (~4926),
  `ggml_cuda_mul_mat_q_pair` (mmq.cu).
- `ggml/src/ggml-cuda/mmvq.cu` / `mmvq.cuh` (`ggml_cuda_op_shexp_down_gate`, `shared_up` in
  `mul_mat_vec_q_moe`), `ggml/src/ggml-cuda/mmq.cu` (`ggml_cuda_mul_mat_q_switch_type_gate`).
- Env kill switches: `GGML_CUDA_DISABLE_SHEXP_DOWN_GATE`, `GGML_PAIR_2X`,
  `GGML_CUDA_DISABLE_FUSION`; `LLAMA_HC_BLK16`.
- `patches/README.md` block-13 notes; `GREEDY-PURITY.md` §24; upstream `git show bed0a8566`.
