# wip/rdna4-dense-swiglu-mmq — dense SWIGLU folded into the mmq down-projection quantize (prefill, bit-exact)

One `git am` patch for r25 (`ggml-cuda.cu`, `mmq.cu/.cuh`, `quantize.cu/.cuh`; +132/-6). Bit-exact, with an off
switch.

## Why

Prefill profile of Qwen3.8-27B UD-Q4_K_XL on r21 (gfx1201, q8_0 KV, ub 512): per 512-token ubatch the FFN's
`silu(gate) * up` runs as `unary_gated_op_kernel` (64 launches, ~6 ms of ~343 ms) and writes a
`17408 x 512` F32 tensor that `quantize_mmq_q8_1` reads straight back for the down projection.

Block 13 already folds this for MoE (`ggml_cuda_mul_mat_q_swiglu`), but only for the qwen3.6 / qwen4exp expert
shapes, Q8_0 / IQ4_NL weights (the D4 q8_1 layout) and RDNA3_5.

## The change

- **Quantize kernel:** `quantize_mmq_q8_1` gets a `glu` template variant. It computes `silu(gate) * up` on
  load, with the same `ggml_cuda_op_silu_single` and one multiply as `unary_gated_op_kernel<op_silu>`, so the
  values quantized are the floats the unfused kernel writes. Every ds layout (D4 / DS4 / D2S6), so Q4_K,
  Q5_K, Q6_K and IQ4_XS down projections qualify. It uses the same grid and chunking as
  `quantize_mmq_q8_1_cuda`.
- **Matcher:** `GLU(SWIGLU) -> MUL_MAT` with the GLU output consumed only by the matmul, split or single-tensor
  GLU, 2-D, 16-byte aligned rows. It fires only when the new `ggml_cuda_mul_mat_takes_mmq()` says
  `ggml_cuda_mul_mat()` would run that matmul through mmq (the same predicate chain in the same order: MMB,
  MMVF, MMF, the MMVQ band incl. the RDNA4/RDNA3_5 dense-band rule, then `should_use_mmq`). Decode and the verify
  band never reach it, and the kernel choice for the down projection does not change.
- **Existing path untouched:** the block-13 MoE fold keeps its own path.
- **Off switch:** `GGML_CUDA_FUSE_SWIGLU_MMQ=0`.

## Results (gfx1201, R9700, ROCm 10.0, measured on r21 + #64, re-checked on r25 under Gates)

Kernel trace, pp512:
- `unary_gated_op_kernel`: 128 -> 65 launches and 12.4 -> 2.4 ms. The other 64 are the GDN / attention gates;
  1 FFN layer did not match.
- `quantize_mmq_q8_1`: 9.0 -> 14.9 ms, since it now reads gate + up, at about the DRAM rate.
- Net: -4 ms per ubatch.

Server (production flags: DFlash2 n-max 4, 262k, q8_0 KV), prefill t/s, 2 rounds:

| | doc 50k | code 97k |
|---|---|---|
| off | 1043.9 | 885.5 |
| on | 1056.6 (+1.2 %) | 892.2 (+0.8 %) |

Decode ms/step, all four greedy texts and VRAM were identical.

## Gates

- **`test-logits-width-probe`:** `width_purity=PASS` and per-W / row0-row1 hashes identical on vs off, on 7 runs:
  27B UD-Q4_K_XL q8_0 (`RS=from_w`, `RS=0`) and f16, 27B IQ4_XS q8_0, 27B NVFP4-MTP f16, 35B-A3B q8_0,
  gemma-4-26B q8_0.
- **On r25 (tree `c7385cd5`) + this patch:** hashes identical on vs off on 6 of the 7 runs. The seventh, 27B NVFP4-MTP
  f16, is not repeatable run to run with the fusion off either (4 runs each: r23 split 3 / 1, r25 split 2 / 2 across two
  hashes), so on vs off can't be compared there. llama-bench off / on (2 rounds each, first run discarded): pp2048
  1337 / 1350 (+1.0 %).

## Tried, didn't help: RDNA4 mmq tile sweep

The prefill mmq on RDNA4 runs one config for Q4_K / Q5_K / Q6_K / IQ4_XS at J = 128 (256 threads, I 128,
occupancy 2, no stream-k). I compiled variants of that row and timed the 27B shapes (`test-backend-ops perf`,
n = 512, 20 shapes, 2 passes). Nothing beat the shipped row:

| variant | sum vs shipped (16.63 ms) |
|---|---|
| occupancy 1 | same |
| I 64 / 128 threads (occupancy 2 or 4) | +4 % |
| I 256 / 512 threads | +21 % |
| stream-k | +29 % |
| J capped at 96 | +26 % |
| J capped at 64 | +87 % |

Two things that may save someone time:

- **`K_vram = 512` looks 2x faster but is wrong.** The `mul_mat_q_process_tile` loop body always consumes two
  `MMQ_TILE_NE_K` sub-tiles per iteration but advances `kb0` by `K_vram / qk`, so any `K_vram` other than 256
  skips half of K. The table's `static_assert(K_vram % 256 == 0)` allows 512.
- **`GGML_CUDA_MMQ_J_MAX` does not pick J.** It only sizes the q8_1 buffer; `mul_mat_q_switch_J` chooses J by
  tile count.
