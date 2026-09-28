# Maintainer verification for r21 (PR #62)

Integrated 2026-09-28 into the delivery as a **block-15 amendment** (release-in-progress
`v16-84e76d8a2-r21`).  Both patches were folded into the block-15 commit (`fattn-mma-f16.cuh`).

Toolchain: `~/bin/build-llama-rocm-714` (`/opt/rocm-7.14.1-gfx102X`), `GPU_TARGETS=gfx1201`, one R9700
(`HIP_VISIBLE_DEVICES=0`), model Qwen3.8-27B UD-Q4_K_XL.

## Gates

| gate | result |
|---|---|
| build | clean |
| `test-backend-ops` (full, incl. `FLASH_ATTN_EXT`) | **18905/18905** |
| same-seed greedy (`prompts/reasoning.txt`, seed 42, temp 0, 300 tok, q8_0 KV) | **identical to r20** (`017e51ea04b1`), `none == n1 == n3 == n7` |
| perplexity (`-ub 1 -c 2048 --chunks 8`, wikitext-2) | 5.0109 ± 0.1274 → **5.0187 ± 0.1277** (within noise) |
| `tg128 @ d50000` (q8_0 KV) | 24.42 → **25.52 t/s (+4.5 %)** |

Patch 1 is bit-exact by construction (band-only config row + forward K-batch order for 256-dim non-MLA
heads).  Patch 2 is a rounding change but W-pure, so both the within-build `none == draft-mtp` invariant
and the r20 greedy text are preserved (the rounding did not flip a token in these runs).

## One factual correction

The new config row matches `ncols == 32`, i.e. the **native-quantized** band arm (`ncols1 = 4`).  The
2-byte f16/bf16 arm selects `ncols1 = 2` (`ggml_cuda_fattn_band_wmma_ncols1`), so it compiles to
`ncols = 16` and does **not** take the new row:

| config | baseline | patched |
|---|---|---|
| f16 KV `tg128 @ d16384` | 27.50 t/s | 27.57 t/s |

So the README's "f16 3-4 %" is misattributed (likely measured before the block-15 r5 two-byte arm existed).
Extending the 64-wide row to the 2-byte arm is a follow-up.

`validate-set.sh` green (strict 16/16 `git am`).
