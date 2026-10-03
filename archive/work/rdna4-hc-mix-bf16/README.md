# wip/rdna4-hc-mix-bf16

A BF16 variant of the fused `GGML_OP_HC_MIX` for qwen4exp models whose hyper-connection weights are BF16 (Flash-Next
GSQ-RCO). It is bit-identical to the unfused BF16 decode chain and gives about +5 % decode. One `git am` patch on r35:

- `0001-cuda-fused-hc_mix-for-BF16-hyper-connection-weights-.patch` (r35 tree `d08fbaf2` -> `41e4a733`; 4 files,
  +284 / -5: `ggml-cuda/hc-mix.cu`, `ggml-cuda/ggml-cuda.cu` (supports_op), `ggml.c` (type asserts),
  `src/models/qwen4exp.cpp` (gate)). It touches different files from PR #90, so the two apply in either order.

## Why

The ISTA-DASLab GSQ-RCO quants keep `hc_{attn,ffn}_{down,up,inject}` in BF16. The fused hc_mix in block 14 is
Q8_0-only, so `build_hc_mix` falls back to the unfused chain on these models:

`rms_norm*gamma` -> `mul_mat_vec_f` (down) -> scale+silu -> `mul_mat_vec_f_vb` (up) -> `dsv4_hc_pre` (gate + collapse)
-> `mul_mat_vec_f` (inject)

That's six dispatches per mixer and 96 mixers per token. Decode on this model is already dispatch-bound: about 16 ms of
kernels plus about 4 ms of gaps per token, with roughly 2,400 dependent dispatches at 2.6-3.2 µs each on gfx1201. So
dispatch count matters.

## How

The BF16 variant replays the chain in three dispatches, with the same arithmetic:

1. `hc_mix_rms_gamma_quant` without the quantize step (the xn it already produces for Q8_0);
2. the down rows and the inject rows in one grid, each row computed exactly like the
   `mul_mat_vec_f<bf16, float, nt, 256>` block (same per-thread K order, same warp butterfly, same butterfly over the
   per-warp sums);
3. the up rows of the 4 streams of one column in one block (one warp per stream, emulating the 160-thread
   `mul_mat_vec_f_vb` block), reading `silu(lo/hc)` as the scale+silu kernel computes it (precomputed once per block
   in shared memory), followed by the gated collapse as `dsv4_hc_pre_f32` computes it.

The block sizes match what the mmvf launcher picks for these K (it depends on K only), so every `nt` in the
decode/verify band replays the unfused arithmetic.

Gates:

- `hc == 4`, BF16 down/up/inject (CUDA `supports_op` mirrors this).
- The hc weights must be in a non-host buffer. Only the CUDA/HIP backend implements the BF16 variant (the CPU
  `HC_MIX` is Q8_0-only), so CPU-resident hc weights (CPU-only runs, `-ot ...hc_...=CPU`) keep the unfused chain.
- `LLAMA_HC_MIX_BF16=0` keeps the unfused chain for A/B.

## Validation

R9700 (gfx1201), Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, `-ncmoe 24 -ub 256 -c 114688 -ctk q8_0 -ctv q8_0`,
`MOE_EXPERT_CACHE_MIB=1024`, `LLAMA_INDEXER_NOBLOCK=1` (see #89/#90). Same binary, fused vs `LLAMA_HC_MIX_BF16=0`:

- **Greedy**, 6 short prompts x 400 tokens: identical.
- **Long temperature-0 chat prompts** (11k / 22k / 30k tokens, reasoning + answer hashed): 11k and 30k identical in
  every run. The 22k prompt is not run-to-run deterministic on this setup in either mode (unfused gave two different
  hashes in two runs; fused gave the unfused run's hash twice out of three), so that variation predates this patch.
  I haven't tracked down where it comes from yet.
- **test-backend-ops `-o HC_MIX`**: OK. Those are the existing Q8_0 cases; the CPU reference has no BF16 path, so
  BF16 isn't covered there. Its bit-identity rests on the model-level checks above.
- **hc weights forced to CPU** (`-ot 'blk\..*\.hc_.*=CPU'`): loads and runs, no assert. The chain then runs on CPU.
- Earlier, on r29 / r32 / r34: greedy and short-prompt SHAs identical. In production here since 2026-10-02.

**Speed** (`llama-bench -ncmoe 24 -ub 256 -b 2048 -p 512 -n 128 -r 3`, 2 interleaved passes):

| | tg128 | pp512 |
|---|---|---|
| unfused (`LLAMA_HC_MIX_BF16=0`) | 48.7 / 48.8 | 780 / 770 |
| fused | 51.2 / 51.1 (+4.9 %) | 797 / 773 |

Prefill doesn't change, since the op only serves the decode/verify band.
