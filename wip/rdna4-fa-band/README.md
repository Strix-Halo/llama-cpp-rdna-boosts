# wip/rdna4-fa-band — faster RDNA4 GQA-6 decode/verify FA band

Two `git am` patches for `ggml/src/ggml-cuda/fattn-mma-f16.cuh` (the only file touched):

| patch | what | bit-exact vs stock |
|---|---|---|
| `0001` | band-only MMA config row: K and V loaded in two 64-half2 batches, forward K order for 256-dim non-MLA heads | **yes** |
| `0002` | 8 warps per block for the band (optional) | no — rounding change, still W-pure |

Measured on r17 + #57 (gfx1201, R9700). `fattn-mma-f16.cuh` is identical in r17 and r19, and both patches
apply cleanly to r19 (`5f2fc93fa` + `git am`).

## Why

The band (`flash_attn_ext_f16<256, 256, 4, 8>`, RDNA4, `n_q <= 8`) is where single-token decode time goes at
depth on the dense 27B: ~25 % of GPU time per token at 97k. With a q8_0 cache it read that cache at ~310 GB/s,
against ~617 GB/s for the f16 vec kernel; it runs at 256 VGPRs with scratch spills and ~4 waves per SIMD, too
few to hide its synchronous K/V loads.

## 0001 — 64-wide K/V batches (bit-exact)

- **The row:** `ggml_cuda_fattn_mma_get_config_rdna()` gets a row for `is_rdna4 && ncols2 == 8 && DKQ == 256 &&
  DV == 256 && ncols == 32`: `fattn_mma_config(128, 2, 64, 64, 64, 64, 1, true)` (nbatch_K2 / nbatch_V2 64
  instead of 128). Only the band matches it.
- **Order:** the K batches are iterated in reverse for MLA's K→V tile reuse. Once K is split, that would change
  the head-dimension accumulation order. A 256-dim non-MLA head (`!V_is_K_view && DKQ == 256`) now iterates
  them forward, so the result is bit-identical to the single-batch row. MLA and the other head sizes keep
  today's order (and every other 256-dim config still loads K in one batch).
- **Plumbing:** the config lookup carries `ncols2`. The host helpers take it explicitly; the device helpers
  default it to 1. (A default on the host side collides with the device overloads inside `__launch_bounds__`:
  "attribute requires parameter to be an integer constant".)
- **Prefill is untouched:** the `ncols1_16/ncols2_2` instance compiles to byte-identical ISA, and
  `ncols1_32/ncols2_1` differs only in a `__LINE__` constant.

## 0002 — 8 warps per block (optional)

Same row with `nthreads 256, occupancy 1`. Each Q column's KV rows are split 4 ways instead of 2, so the
partial VKQ sums combine in a different order: not bit-identical to 0001. The split does not depend on `n_q`,
so W = 1..8 stay bit-identical to each other.

## Kernel time

q8_0 K/V, head 256, GQA 6, n_q 1..4, µs per call (`test-backend-ops perf` with local FLASH_ATTN_EXT cases):

| kv | stock r17 | + 0001 | + 0001 + 0002 |
|---|---|---|---|
| 51200 | 380 | 336 (−12 %) | 283 (−25 %) |
| 98304 | 685 | 587 (−14 %) | 487 (−29 %) |

## End to end

27B UD-Q4_K_XL, q8_0 KV, r17 + #57 (tg t/s):

| | stock | + 0001 | + 0001 + 0002 |
|---|---|---|---|
| llama-bench tg128 at d50000 | 24.88 | 25.65 (+3.1 %) | 26.00 (+4.5 %) |
| llama-bench tg128 at d97000 | 22.47 | 23.24 (+3.4 %) | 24.00 (+6.4 %) |
| DFlash2 n-max 6, 262k server, 97k prompt | 56.3 | 57.3 (+1.9 %) | 57.65 (+2.4 %) |

Short context is unchanged (tg128 at d0 28.95 → 28.89).

The + 0001 + 0002 column is from a separate A/B run whose own stock baseline was 24.88 / 22.55 / 56.3;
each percentage is against its own run's baseline.

## Gates

- **Full `test-backend-ops`:** 0 failures, with and without 0002.
- **`test-logits-width-probe`** (`prose-rdna-boosts.txt 1024 512`): PASS on 7 runs — 27B UD-Q4_K_XL (f16;
  q8_0 `RS=from_w`), 27B IQ4_XS, 27B NVFP4, 35B-A3B (f16; q8_0 `RS=from_w`), gemma-4-26B MXFP4 — worst maxdiff 0.
  With 0001 alone, per-W hashes are identical to stock everywhere. With 0002, the q8_0 hashes differ from stock
  but stay W-pure; the f16 hashes are unchanged (the f16 band uses a different row).
- **Decode-path perplexity** (`llama-perplexity -ub 1 -c 2048`, so the band runs): 9.9297 ± 0.61291 with and
  without 0002.
- **Greedy text** at 50k and 97k depth: identical to stock with 0001 (and with 0001 + 0002).

## Tried, didn't help

- aligned 4-byte q8_0 loads with `v_alignbyte` (2× slower);
- `Q_in_reg = false`;
- other K/V batch widths (32/32, 64/128, 128/64);
- an extra KV split on top of 0001 (≤ 4 %, and it changes rounding).

Only RDNA4 was measured; RDNA3 never takes the band row.
