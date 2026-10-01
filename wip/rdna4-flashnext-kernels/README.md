# wip/rdna4-flashnext-kernels

Kernel speedups found while profiling Qwen3.8-Flash-Next (qwen4exp, Unsloth UD-Q2_K_XL) on gfx1201 (R9700), on
top of release r26 (tree `afbdc436`). Two `git am` patches:

- `0001-cuda-RDNA4-decode-prefill-kernel-speedups-...patch`: the kernel changes (r26 -> tree `0378f575`).
- `0002-scripts-rdna4-bitcheck-test-tools-...patch`: **optional**. It adds the two byte-identity checkers I used
  (`scripts/rdna4-bitcheck/`). They are development tools only, not part of the build, and 0001 does not depend
  on them. Skip 0002 if you don't want them in the tree.

Despite the model in the name, only part of 0001 is Flash-Next-specific. "UD-Q2_K_XL" is a mix label: its experts
are IQ2_XS / IQ3_XXS (gate/up) and IQ4_NL (down), and its dense weights are Q5_K / Q6_K / Q8_0. The Q2_K type itself
is not changed here.

## Testing status: more to do once the MoE expert cache lands

This is **kernel-level work only**. Full model-level testing is deliberately deferred until `archive/work/moe-expert-cache`
lands, because the cache changes where the expert matmuls run and what dominates decode. Once it lands I plan to:

- rebase onto it (0001 touches `mmvq.cu` in about 11 lines, which the cache patch also edits);
- repeat the greedy-identity runs (plain and MTP) and the kernel profiles on top of the cache;
- run end-to-end speed on Flash-Next;
- run a greedy-identity check on the other models the general changes touch (e.g. a dense 27B with K-quants),
  which has not been done yet.

## Bit-exactness (what has been checked so far)

Every change keeps each output's arithmetic: the same item-to-thread mapping, accumulation order, warp butterflies and
cross-warp sum order. What changes is unrolling, rows per block, which rows a warp visits (the skipped ones hold exactly
+0.0) and tile enumeration.

- On this r26 tree:
  - `mmbit` (60 MUL_MAT / MUL_MAT_ID cases at Flash-Next shapes, including a token-0 width-purity hash) is byte-identical
    to the unpatched build.
  - Both `hcbit` md5s (HC_MIX at nt 1..8) match the original kernel.
  - `test-backend-ops` on ROCm0 passes HC_MIX 20/20, MUL_MAT_ID 929/929 and MUL_MAT 1297/1297.
- On the r25 base: Flash-Next greedy output is identical to r25 with `--spec-type none` and with `draft-mtp` n-max 4
  (278/482 drafts accepted in both).

## Changes: qwen4exp only (GGML_OP_HC_MIX)

| Kernel | Change | Case | Before | After |
|---|---|---|---|---|
| hc_mix up (`hc_mix_up_silu_dot`) | new band kernel: all tokens per block, reduction only over the warp's own rows (partials pre-zeroed), used for nt 1..8 | nt 1 / 5 | 16.5 / 59.7 us | 9.1 / 20.6 us |
| hc_mix down tail | v = silu(lo/hc) quantized once, by the block that completes each q8_1 group (per-(token, group) counters), instead of in every up block; `GGML_CUDA_HC_MIX_PREQ=0` turns it off | up + down + memset, nt 1 / 5 | 19.8 / 51.9 us | 18.7 / 46.0 us |
| `hc_mix_rms_gamma_quant` | x / xn kept in registers (a thread owns the same columns in all three passes) | nt 1 / 5 | 3.9 / 3.4 us | 3.3 / 2.9 us |

## Changes: general (any model using these types or shapes)

| Kernel | Change | Scope | Case | Before | After |
|---|---|---|---|---|---|
| dense `mul_mat_vec_q_ksplit`, 1 token | 2 rows per block (`GGML_MMVQ_RDNA4_WEIGHT_RPB1`) | RDNA4, all dense weight types | Q5_K 10240 / 6144 rows; Q6_K; Q8_0 2560 (cold) | 38.5 / 26.2; 28.0; 14.2 us | 30.6 / 19.1; 26.1; 13.3 us |
| `mul_mat_vec_f` (F32, e.g. MoE router 2560 -> 512) | `#pragma unroll 4` on the K loop; one cross-warp exchange for all columns at nt > 1 | all backends | nt 1 / 5 (cold) | 17.9 / 19.0 us | 9.8 / 11.0 us |
| IQ2_XXS / IQ2_XS / IQ3_XXS sign unpack | `apply_ksigns`: nibble multiply instead of byte compare/subtract (grid bytes are 1..127, so `(g ^ 0xFF) + 1` cannot carry) | all backends, mmvq + mmq loaders | IQ2_XS MoE nt 1 / 5 | 15.7 / 61.2 us | 14.2 / 53.9 us |
| `mul_mat_vec_q_moe` | `#pragma unroll 2` on the item loop, IQ2_XS only (IQ3_XXS / IQ4_NL lose) | all backends | IQ2_XS nt 1 | 14.2 us | 12.85 us |
| mmq `q8_0_16` vec dot (IQ2_XS, Q3_K layout) | `#pragma unroll 1`: the default fully unrolled at J=16 and spilled (256 VGPRs + scratch) | AMD MMA path | IQ2_XS MoE nt 512 | 1709 us | 1677 us |
| mmq `q8_1_q8_1` vec dot (Q4_K / Q5_K / ...) | `#pragma unroll 1`: J <= 64 spilled | AMD MMA path | Q5_K 10240 nt=64; Q4_K 2560 nt=64 (cold) | 189.5; 57.8 us | 60.5; 25.0 us |
| routed-compact MoE mmq | IQ2_XS and IQ3_XXS take the IQ J bands and the compact path; this flips the `!use_compact(IQ2_XS, 48)` static_assert | RDNA3.5 / RDNA4 | IQ2_XS nt 512 / 2048; IQ3_XXS nt 512 / 2048 | 1674 / 2374; 1355 / 1887 us | 1467 / 2224; 869 / 1662 us |

Numbers are kernel medians from rocprofv3. "Cold" means a bench that cycles more than 256 MiB of weight copies, because
test-backend-ops perf is cache-hot: a Flash-Next layer's experts fit the 64 MiB Infinity Cache.

On Flash-Next (r25 base, same llama-bench arguments, experts of 22 layers on the CPU), rocprofv3 GPU kernel time per
pass is:

- decode: 15.05 -> 13.44 ms (-10.8 %);
- 5-token verify: 26.75 -> 22.08 ms (-17.4 %);
- pp2048 ubatch: 765.8 -> 737.1 ms (-3.8 %).

Roughly 40 % of the decode and verify gain is the qwen4exp-only hc_mix work. These are GPU times, not tokens/s: wall
time on this setup is dominated by the CPU experts until the cache lands.

`test-backend-ops` gains HC_MIX eval + perf cases, Flash-Next MUL_MAT / MUL_MAT_ID perf shapes and a 512-expert top-10
TOPK_MOE eval case.

## Tried and not kept

- LDS copy of `iq2xs_grid` for the mmq loaders: +18 %.
- IQ2_XS mmq tiles at 256 threads / I = 128: +24 %. Q5_K J=128 at 128 threads / I = 64: +3 %.
- 4 or 8 rows per block for long-K MoE mmvq: flat or worse.
- Tree argmax in `topk_moe`: 9.0 -> 9.4 us (it is shuffle-latency bound).
- A band `hc_mix_down_dots` (one block per row, all tokens): slower at every nt (fewer blocks; the weights were L2 hits).
- `#pragma unroll 1` in `q8_0_q8_1_mma`: removes the IQ3_XXS J=32 spill (-34 % at nt=32) but costs Q8_0 6-8 %; it would
  need a type-conditional loop.

## Notes that may be useful

- A compiler spill scan (`-Rpass-analysis=kernel-resource-usage`) of every mmq instance on gfx1201 found that Q2_K spills
  64-2156 B/lane at every J, and Q5_0 / Q5_1 / Q1_0 spill at J=128. These are not Flash-Next types, so they are not
  touched here.
- `test-backend-ops -p` is a regex (an unescaped `[` throws), and perf mode skips whole-graph tests such as TOPK_MOE.
- rocprofv3 PMC counters (FETCH_SIZE, GL2C_*, TCP_REQ) all read 0 on this gfx1201 setup, so the analysis used timing
  experiments and compiler resource reports instead.

Only gfx1201 on ROCm 10.0 was measured.
