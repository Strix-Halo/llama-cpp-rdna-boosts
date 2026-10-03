# WIP — MMQ `prec_src1` × fused-gate / W4A4 interaction

**Created:** 2026-10-05, after the `v16-a55e952b8-r1` re-base.
**For:** a follow-up session.
**Status:** **NOT part of the delivery.** Validation/hardening only.

---

## Why this WIP exists

The r1 re-base had to merge two features that edit the **same template parameter slot** of the whole
MMQ stack:

- upstream `e9f824d8c` "llama: add `llama_prec_policy` + model-driven W4A4 path" introduced
  `ggml_prec prec_src1` (native FP4 `Q4` activations on Blackwell, else `Q8_1`);
- the fork's block 13 introduced `bool has_gate` (the fused gate+up+GLU MMQ) in that slot.

The merge keeps both, with `prec_src1` **before** `has_gate`:

```cpp
template <ggml_type type, int J, bool fallback, bool fixup, ggml_prec prec_src1 = GGML_PREC_Q8, bool has_gate = false>
// mul_mat_q_process_tile, mul_mat_q, launch_mul_mat_q, mul_mat_q_switch_J
template <ggml_type type, ggml_prec prec_src1 = GGML_PREC_Q8, bool has_gate = false>
// mul_mat_q_case
```

and `DECL_MMQ_CASE_GATE` = `mul_mat_q_case<type, GGML_PREC_Q8, true>`, with `DECL_MMQ_CASE_W4A4`
restored (`mul_mat_q_case<type, GGML_PREC_Q4>`).

The RDNA validation passed (`MUL_MAT_ID` 931/931), but the two features are only jointly exercised on
the paths this box reaches. Open questions:

1. **The fused-gate dispatch hardcodes `GGML_PREC_Q8`**
   (`ggml_cuda_mul_mat_q_switch_type_gate` in `mmq.cu`). That is correct today because the gate types
   are Q3_K/Q4_K/Q5_K/Q6_K/Q8_0 (never FP4). If a fused-gate type is ever added for MXFP4/NVFP4, the
   hardcoded Q8 silently disables W4A4 there — assert or thread `prec_src1` through instead.
2. **`ggml_cuda_mul_mat_q_pair`** (block 13's shared activation quantize for two consecutive MMQ
   muls) computes `prec_i = ggml_cuda_mmq_get_prec_src1(src0_i, dst_i, cc)` per tensor and passes it
   to `ggml_cuda_mul_mat_q_switch_type`. Confirm this matches what the standalone
   `ggml_cuda_mul_mat_q` would have chosen for the same tensor (the pair's comment claims
   "non-native-fp4 types", but the code does not assert it).
3. **No FP4 hardware here** (the W4A4 path targets Blackwell / sm_120+). The repo's scope policy says
   non-AMD paths must stay *consistent* (no aborts, no uninstantiated pairs) even if unvalidated, so
   the minimum bar is: the FP4 template instantiations exist, `mmq` compiles for CUDA, and
   `test-backend-ops` does not abort on a CUDA/Blackwell box.

## Objectives

1. Add a `GGML_ASSERT(prec == GGML_PREC_Q8 || gate type is not FP4)` (or thread `prec_src1`) in
   `ggml_cuda_mul_mat_q_switch_type_gate` so the hardcoded Q8 cannot silently mismatch a future FP4
   gate type.
2. Assert (or document) the pair fusion's non-native-fp4 assumption, or make it handle FP4.
3. On a CUDA/Blackwell box (or CI), run `test-backend-ops -o MUL_MAT,MUL_MAT_ID` and the fused-gate
   case with `GGML_CUDA_MMQ_PREC=q4` and `q8` to prove both template arms instantiate and produce
   the expected precision.
4. Re-confirm on gfx1201 that `MUL_MAT_ID` 931/931 and the gate MMQ prefill paths are unchanged.

## Acceptance

- `grep` shows no place where `prec_src1` and `has_gate` are passed in the wrong order.
- `DECL_MMQ_CASE_W4A4` and `DECL_MMQ_CASE_GATE` both expand (the `generate_cu_files.py` instance
  files carry both).
- A note in `patches/README.md` block 13 recording the merged parameter order and the FP4 caveat.

## Pointers

- `ggml/src/ggml-cuda/mmq.cuh` (`mul_mat_q_process_tile`, `mul_mat_q`, `launch_mul_mat_q`,
  `mul_mat_q_switch_J`, `mul_mat_q_case`, `DECL_MMQ_CASE*`).
- `ggml/src/ggml-cuda/mmq.cu` (`ggml_cuda_mmq_get_prec_env`/`_src1`,
  `ggml_cuda_mul_mat_q_switch_type_gate`, `ggml_cuda_mul_mat_q_pair`).
- `ggml/src/ggml-cuda/template-instances/generate_cu_files.py` (`TYPES_MMQ_W4A4`, `MMQ_GATE_TYPES`).
- upstream `git show e9f824d8c`; WORKLOG 2026-10-05 (r1) "Block 13" note.
