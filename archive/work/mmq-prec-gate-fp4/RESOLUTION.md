# mmq-prec-gate-fp4 — RESOLVED (`v16-a55e952b8-r3`)

**Resolved:** 2026-10-05, folded into **block 13** (release `v16-a55e952b8-r3`).
**Original handover:** [`README.md`](README.md).
**Delivery change:** yes — `ggml/src/ggml-cuda/mmq.cu` only.

The r1 re-base merged upstream `e9f824d8c`'s `ggml_prec prec_src1` into the same MMQ template
slot the fork's block 13 used for `bool has_gate`, with `prec_src1` **before** `has_gate`
(`mul_mat_q_process_tile<type,J,fallback,fixup,prec_src1,has_gate>`, `mul_mat_q`,
`launch_mul_mat_q`, `mul_mat_q_switch_J`, `mul_mat_q_case<type,prec_src1,has_gate>`).  That
order is audited and correct; this record closes the FP4-consistency follow-up.

## What was done

1. **Fused-gate dispatch no longer hardcodes Q8 silently.**  `ggml_cuda_mul_mat_q_switch_type_gate`
   now takes `const ggml_prec prec_src1` (threaded from `ggml_cuda_mul_mat_q`, which already
   computed it) and asserts `prec_src1 == GGML_PREC_Q8`.  The gate types
   (`Q3_K/Q4_K/Q5_K/Q8_0/Q6_K`) and the `DECL_MMQ_CASE_GATE` instantiations are never FP4, so
   this is a behaviour-preserving no-op today.  If an FP4 fused-gate type is ever added, the
   assert fires and forces the author to give it its own Q4 arm — the failure mode the WIP
   called out (a hardcoded Q8 silently disabling the Blackwell W4A4 path) is gone.
2. **The pair fusion's non-FP4 assumption is now explicit.**  `ggml_cuda_mul_mat_q_pair`
   asserts neither weight is `NVFP4`/`MXFP4`.  The dispatch in `ggml_cuda_try_fuse` already
   excludes them (`src0->type != NVFP4 && src0->type != MXFP4 && src0_next->type != ...`), and
   the pair always quantizes to Q8_1 (it has no fp4 quantize + scale path), so the assert only
   documents/guards the existing contract.

No change to `mmq.cuh`, `generate_cu_files.py`, the template instances, or any runtime result.

## Validation (gfx1201 / ROCm 7.14)

* **Instantiations present.**  Both macros expand in the generated files:
  `mmq-instance-mxfp4.cu` / `mmq-instance-nvfp4.cu` carry `DECL_MMQ_CASE_W4A4`,
  `mmq-instance-{q3_k,q4_k,q5_k,q6_k,q8_0}.cu` carry `DECL_MMQ_CASE_GATE` (the r2 tree already
  had this; the check is recorded because the WIP made it an acceptance item).
* **`test-backend-ops -o MUL_MAT,MUL_MAT_ID`** (single R9700): **2235/2235** with
  `GGML_CUDA_MMQ_PREC=q8` and **2235/2235** with `GGML_CUDA_MMQ_PREC=q4`.  On gfx1201
  `blackwell_mma_available()` is false, so `q4` does not select the native arm — the run proves
  the env path does not abort and the Q8 fallback is intact, which is the AMD-reachable bar.
* **`MUL_MAT_ID`** 931/931, **`MUL_MAT`** in the 2235 above; gate MMQ prefill exercised end to
  end by the qwen35moe trace (see the sibling reconciliation record).
* **No-FP4 waiver.**  There is no Blackwell / sm_120+ hardware on this host, so the W4A4 arm
  itself is not executed here.  Per the repo's scope policy the non-AMD path is only required to
  stay *consistent* (both macros expand, no uninstantiated pairs), which it does.

## Delivery record

* `patches/0013-rdna-boosts-block-13-...patch` (block 13 amended).  The `DECL_MMQ_CASE_W4A4` /
  `DECL_MMQ_CASE_GATE` macro set and the `prec_src1`-before-`has_gate` order were established by
  the r1/r2 re-base merge; r3 adds the `prec_src1` thread-through/assert and the pair FP4 assert.
* `patches/README.md` block-13 notes: the merged parameter order + the FP4 caveat.
* Release `v16-a55e952b8-r3`, tree `25a8e137a585cd9fc2907a74236998f881635b8e`.
