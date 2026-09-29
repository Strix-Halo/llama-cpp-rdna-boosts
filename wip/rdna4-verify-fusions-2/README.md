# wip/rdna4-verify-fusions-2 — two more decode fusions extended to 2..8-token verify batches

A follow-up to `wip/rdna4-verify-fusions/` (#63, integrated in r21). Two `git am` patches for r21. Each applies on its own (and together with `wip/rdna4-fa-band-wide`),
each is bit-exact, and each has its own off switch.

## Why

A kernel trace of one 5-token verify pass on r21 (27B UD-Q4_K_XL, q8_0 KV, the width a DFlash2 / MTP
verify at n-max 4 runs at) showed 1,787 kernels vs 1,389 for one token. The GPU was idle 13.8 % of the
pass, almost all of it in ~3 us gaps between launches. Two single-token fusions accounted for 272 of
the extra launches:

| per 5-token pass | one token | 5 tokens |
|---|---|---|
| GDN gate/beta: `ssm_gate_beta_fused_q8_0` | 48 launches | 96 Q8_0 mmvq + 48 `add_softplus_mul` + 48 sigmoid |
| residual ADD before `rms_norm_q8_1` | folded into the mmvq epilogue | 128 `k_bin_bcast` |

## 0001 — `ssm_gate_beta_fused_q8_0` for `ncols` 1..8

- **Kernel:** the fused kernel gets an `ncols` template parameter. Each token keeps the single-token K
  order and cross-warp reduction. `nwarps` comes from `calc_nwarps_weight()` at the launch width, the
  same selector as the standalone mmvq launch it replaces, so the band matches decode bit for bit.
- **Matcher:** lifted from `ne[1] == 1` to `ne[1] <= MMVQ_MAX_BATCH_SIZE`. It now also requires
  contiguous outputs of `nrows * ncols` elements.
- **Off switch:** `GGML_CUDA_FUSE_GATE_BETA_VERIFY=0`.
- **Effect:** -144 launches per 5-token pass.

## 0002 — residual ADD folded into `rms_norm_q8_1`

- **Kernel:** at 2..8 tokens the ADD -> RMS_NORM -> MUL chain that feeds an mmvq matmul runs as one
  `rms_norm_q8_1_f32<.., has_add = true>`. It computes `x = a + b` (one IEEE add, as `k_bin_bcast`),
  writes `x` out for the residual stream, then normalizes and quantizes as before.
- **Scope:** one token is unaffected, since its ADD is still folded into the mmvq epilogue.
- **Off switch:** `GGML_CUDA_FUSE_ADD_RMS_Q8=0`.
- **Effect:** `k_bin_bcast` goes from 128 to 1 per 5-token pass.

## Results (gfx1201, R9700, ROCm 10.0)

`llama-bench`, 27B UD-Q4_K_XL, q8_0 KV, t/s. The rows are cumulative (off/on/off/on per patch; pp1
is unchanged):

| | pp5 | pp8 |
|---|---|---|
| r21 | 122.9 | 164.6 |
| + 0001 | 125.0 | 166.4 |
| + 0002 | 126.2 | 167.9 |

Server, same model, DFlash2 n-max 4, greedy, 768 tokens. ms per verify step, each arm run twice:

| | short code | short reasoning | doc 50k | code 97k |
|---|---|---|---|---|
| 0001 off -> on | 49.05 -> 48.48 | 48.82 -> 48.20 | 55.14 -> 54.48 | 60.26 -> 59.68 |
| 0002 off -> on | 48.35 -> 48.09 | 48.15 -> 47.82 | 54.49 -> 54.14 | 59.82 -> 59.25 |

Together that is about -1.7..-2 % per step. The texts were byte-identical in every arm.

## Gates

`test-logits-width-probe`: `width_purity=PASS` and per-W / row0-row1 hashes identical on vs off, for
each patch, on 7 runs:

- 27B UD-Q4_K_XL q8_0 (`RS=from_w`, `RS=0`) and f16
- 27B IQ4_XS q8_0
- 27B NVFP4-MTP f16
- 35B-A3B q8_0
- gemma-4-26B q8_0

## What is left at 2..8 tokens

- **48 conv-state concats** (`concat_transposed_src1_dim0`, ~0.3 ms per pass). Removing them would need
  `ssm_conv` to read the states and the qkv output directly and write the conv input itself.
- **The gate/up GLU split.** Lifting the mmvq epilogue fusions to `ncols > 1` was slower here: the
  `has_fusion` ksplit instance costs 2-3x per launch.
