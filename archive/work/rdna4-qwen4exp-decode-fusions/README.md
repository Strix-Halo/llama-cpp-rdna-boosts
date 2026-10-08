> **CLOSED / FOLDED (r27, 2026-10-07).**  Accepted as PR #114 (@briansp2020) and folded into delivery
> **block 15**.  Four bit-identical decode kernels/fusions for the qwen4exp (Qwen3.8-Flash-Next) graph,
> each with a default-on runtime kill-switch.  See `WORKLOG.md` 2026-10-07 (r27), the block-15 r27
> amendment in `patches/README.md`, and `ENVIRONMENT.md` §7 for the switches.

## Validation on this box (3x R9700 / gfx1201, ROCm 7.14.1)

The PR was measured on 2x R9700 / ROCm 10.0.  Reproduced here on the delivery's reference build; the
per-patch microbench is `test-backend-ops perf -o HC_MIX` (BF16, `n_embd=2560`, `hc=4`, `hc_lr=320`,
`inject=1`), whole op:

| n_tokens | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---:|---:|---:|---:|---:|---:|
| r26, us | 46.00 | 43.81 | 44.82 | 50.13 | 55.31 | 70.64 |
| r26 + the four patches, us | 43.96 | 41.82 | 41.86 | 44.11 | 45.83 | 55.61 |
| gain | +4.4 % | +4.5 % | +6.6 % | +12.0 % | +17.1 % | +21.3 % |

End to end (qwen4exp UD-IQ3_XXS + shared Q8_0 MTP head, adaptive `n-max 3`, `-sm tensor`, 3 GPU,
`-c 16384 -ub 2048`, prose prompt, `-n 1200`, `--reasoning off`, interleaved): **102.4 / 102.1 / 102.3
t/s -> 104.8 / 104.7 / 104.7 t/s (+2.4 %)**, generated text byte-identical in every run
(`sha=4efc5e295062`, 3632 chars).  A second `-n 800` pair gave `+2.1 %` with `sha=d876d027b30f`.  The
host-expert config showed no reliable win after warm-up (`~+1.7 %`; the first-load delta was an arena
warm-up artifact).  Oracles: `HC_MIX 30/30`, `GATED_DELTA_NET 46/46`; dense 4B same-seed
`1c5d32ac537d` unchanged; the build is warning-free.

The campaign text below is the original handover, unchanged.

---

# rdna4-qwen4exp-decode-fusions: four bit-identical decode kernels/fusions for the qwen4exp (Flash-Next) graph

Four format-patches on top of `v16-a55e952b8-r26` (applied tree `6c7dc021c`), applied after `patches/00*.patch` with
`git am`.  They touch only `ggml/src/ggml-cuda/` (`hc-mix.cu/.cuh`, `unary.cu/.cuh`, `gated_delta_net.cu/.cuh`,
`ggml-cuda.cu`).  Each one has a runtime environment switch (default on; `=0` restores the r26 path), and each is
bit-identical to the path it replaces.

| patch | what | switch | decode gain (measured alone) |
|---|---|---|---|
| [`0001`](0001-cuda-hc_mix-BF16-up-collapse-scheduled-for-latency-b.patch) | BF16 `hc_mix` up/collapse kernel scheduled for latency: each warp runs its block's rows back to back (next row's weights prefetched, no barrier between rows), then one barrier and all (row, token) collapses in parallel; the second butterfly is written as its closed form `((w0+w4)+w2)+(w1+w3)` (same XOR tree); 256 blocks | `GGML_HC_UP_V2=0` | +2.0 % |
| [`0002`](0002-cuda-fuse-HC_COMBINE-into-the-BF16-hc_mix-norm-that-.patch) | `HC_COMBINE` fused into the BF16 `HC_MIX` that reads it: one kernel computes the combine (same `w = 2*sigmoid(inject/hc)`), writes it, and feeds the grouped RMSNorm from registers | `GGML_CUDA_FUSE_HC_COMBINE_MIX=0` | +1.3 % |
| [`0003`](0003-cuda-fuse-UNARY-sigmoid-MUL-ADD-with-a-per-row-gate-.patch) | shared-expert gate `ffn_out = moe_out + ffn_shexp * sigmoid(gate)` (gate `[1, n_tokens]`) as one kernel instead of three | `GGML_CUDA_FUSE_SIGMOID_MUL_ADD=0` | +1.3 % |
| [`0004`](0004-cuda-fold-the-GDN-beta-sigmoid-into-the-sequential-g.patch) | the `beta` sigmoid folded into the sequential `gated_delta_net` kernel (decode/verify band only: one sequence, <= 16 tokens, never the chunked kernels); the GDN -> cpy cache fusion is kept | `GGML_CUDA_FUSE_GDN_BETA_SIGMOID=0` | +0.35 % |

## Notes on bit-identity

- **0002**: the first version was not bit-identical: `hc_combine_kernel` compiles `res + bo*w` as `v_mul` + `v_add`,
  and the fused kernel contracted it into `v_fmac`.  `__fmul_rn` does not prevent that on HIP; a
  `#pragma clang fp contract(off)` scope around the expression does (checked in the ISA).
- **0003**: the allocator runs the `ADD` in place, so `ggml_cuda_check_fusion_memory_ranges` rejected the chain and
  the fusion never fired.  The matcher instead allows `dst == x` or `dst == y` exactly (same row stride) and requires
  disjointness otherwise (the gate included).  The product is kept out of an fma, as above.
- **0004**: in the real decode graph a state `RESHAPE` sits between the sigmoid and the GDN, so the matcher looks past
  view/no-op nodes that do not read the sigmoid (up to 4 nodes), and requires a single consumer.
- I checked that each fusion actually fires with a kernel trace (kernel counts per token) before reading an A/B.

## Results (2 x R9700 / gfx1201, ROCm 10.0)

`test-backend-ops perf -o HC_MIX`, BF16, n_embd 2560, hc 4, hc_lr 320, inject on (GPU 1):

| n_tokens | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---|---|---|---|---|---|
| r26, us | 24.6 | 25.3 | 28.9 | 33.0 | 38.7 | 52.8 |
| r26 + these patches, us | 19.7 | 21.2 | 23.6 | 27.7 | 32.4 | 45.2 |

Server, Flash-Next GSQ-RCO IQ3_XXS + shared Q8_0 MTP head (n-max 3, p-min 0), `-sm tensor`, 256K, `-ub 2048`,
`MTP_DRAFT_N_UBATCH=256`, `LLAMA_KV_N_PAD_MIN=1024`, `GGML_COMPUTE_BUFFER_MARGIN_PCT=3`, r26 vs r26 + these patches,
same session, one round:

| t/s | r26 | + patches | |
|---|---|---|---|
| greedy 400-token decode | 120.3 | 124.1 | +3.2 % |
| A / B (new prompts) | 115.9 / 116.0 | 122.1 / 122.0 | +5.3 / +5.2 % |
| A / B (repeated) | 119.1 / 116.6 | 125.3 / 122.7 | +5.2 / +5.2 % |
| C (new prompt) | 107.7 | 113.5 | +5.4 % |

All experts on the host (`--n-cpu-moe 48`, PLE in VRAM, default slab; arena 37.6 GiB, 92 % residency), one run each:
greedy 80.1 -> 83.0 t/s (+3.6 %), prose 93-96 -> 95-102 t/s, code 104-111 -> 120-121 t/s (chat at temperature 0.7,
so noisier).

Greedy output is unchanged in every run (all-VRAM sha `45e68f1bdfeb`, host-expert sha `9f826c79de59`, each the
same with and without the patches), with the same MTP acceptance, and 0 GPU faults.  `test-backend-ops`: HC_MIX
30/30, GATED_DELTA_NET 46/46.

The per-patch gains in the table above are from earlier same-build A/Bs (switch 0 vs 1, two rounds each) on our r20
tensor-split build; the combined r26 numbers are consistent with their sum.  An earlier revision of 0001 carried
a private copy of the DPP butterfly; on r26 that duplicates `warp_reduce_sum`, so it is gone, and the HC_MIX timings
and greedy output of the final patches match that revision (within 0.5 %).

## Not measured

RDNA3 hardware; `-sm layer`; other models (the matchers are written for the qwen4exp graph patterns and fall through
when they do not match).
