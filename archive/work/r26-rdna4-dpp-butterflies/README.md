> **CLOSED / FOLDED (r26, 2026-10-07).**  Accepted as PR #110 (@briansp2020) and folded into delivery **block 15**, with the **build-time** gate `-DGGML_HIP_NO_DPP_XOR` only.  See `WORKLOG.md` 2026-10-07 (r26) and the block-15 r26 amendment in `patches/README.md`.

A runtime device-side gate was implemented and tested but reproducibly wedged the FA prefill at ~4096
tokens with MTP, so it was dropped.  The campaign text below is the original handover.

# rdna4-dpp-butterflies: DPP lane exchange for the wave32 warp reductions (RDNA3/RDNA4)

One format-patch on top of `v16-a55e952b8-r19` (applied tree `24bea75d`), applied after `patches/00*.patch` with
`git am`.  It changes only `ggml/src/ggml-cuda/common.cuh`.

| patch | what | switch |
|---|---|---|
| [`0001`](0001-ggml-cuda-DPP-lane-exchange-for-the-wave32-warp-butt.patch) | `warp_reduce_sum` (float, float2, int) and `warp_reduce_max` do their XOR butterfly with DPP (`permlanex16` for offset 16, `update_dpp` `row_xmask` for 8..1) instead of `__shfl_xor` | build with `-DGGML_HIP_NO_DPP_XOR` to get `__shfl_xor` back |

## Why

On gfx1201 every `__shfl_xor_sync` compiles to an LDS `ds_bpermute`.  A kernel that does many butterflies per row
is then bound by LDS permute throughput, not by memory.  The clearest case is the BF16 `hc_mix` up/collapse kernel
(Flash-Next GSQ-RCO): it replays the unfused mmvf arithmetic, so it runs `nt * 6` full 32-lane butterflies per row.
`test-backend-ops perf -o HC_MIX` (n_embd 2560, hc_lr 320, BF16) scales with the token count rather than with the
weight bytes, and a diagnostic build that skips only the butterflies took the 4-token op from 45 us to 27 us.

DPP moves the same value from the same lane without going through LDS.  Each step is still `x += partner(x)` with
the same partner at the same offset, in the same order (16, 8, 4, 2, 1), so every sum and max is bit-identical
to the `__shfl_xor` version.  It is compiled only for `RDNA3`/`RDNA4` (wave32), with a `static_assert(width <= 32)`
in the DPP branch; every other target keeps `__shfl_xor`.

## Results (2 x R9700, ROCm 10.0, r19 vs r19 + this patch, same session)

`test-backend-ops perf -o HC_MIX`, BF16, n_embd 2560, hc_lr 320, inject on (GPU 1):

| n_tokens | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---|---|---|---|---|---|
| r19, us | 24.3 | 29.7 | 36.4 | 45.0 | 53.0 | 88.6 |
| + DPP, us | 24.2 | 25.5 | 28.9 | 33.0 | 39.4 | 53.7 |

Bandwidth-bound matmuls do not move (`MUL_MAT` m=4096 k=14336, bf16 / q8_0 / q4_K / iq3_xxs at n=1 and n=4: all
within 0.5 %).

Server, Flash-Next GSQ-RCO IQ3_XXS + shared Q8_0 MTP head (n-max 3, p-min 0.5, draft experts on the CPU),
`-sm layer -ts 59,41 --n-cpu-moe 0`, 256K, `-ub 2048`, no other local changes, two alternating rounds each:

| t/s | r19 | r19 + DPP | |
|---|---|---|---|
| greedy 400-token decode | 77.3 / 80.9 | 80.5 / 84.6 | +4 % |
| 6 requests: A (mean of 6) / B (mean of 4) | 81.8 / 75.8 | 84.4 / 79.2 | +3 / +4.5 % |
| C (new prompt) | 79.2 / 79.9 | 82.7 / 81.2 | +3 % |
| 2 concurrent requests (sum) | 94.9 / 86.2 | 100.0 / 100.5 | +11 % |

Greedy output is unchanged (same sha in every run), 0 GPU faults.  Under `-sm tensor` with the #106 patches the
same change gave greedy 114.5 -> 117.8-118.6 t/s and 2 concurrent 140-144 -> 167-168 t/s, also with unchanged output.

`test-backend-ops` with the patch (GPU 1): MUL_MAT 1305, MUL_MAT_ID 932, FLASH_ATTN_EXT 6359, TOP_K 526, SOFT_MAX
215, ARGSORT 79, RMS_NORM 52, NORM 51, GATED_DELTA_NET 47, SSM_CONV 46, HC_MIX 31, L2_NORM 21, SUM_ROWS 11, MEAN 11,
ARGMAX 8, HC_COMBINE 1: all OK, 0 failures.

## Not measured

RDNA3 hardware (the code path is the same wave32 DPP, but I only have gfx1201); 3-GPU layouts; `-sm row`.
