# Maintainer verification for r21 (PR #63)

Integrated 2026-09-28 into the delivery (release-in-progress `v16-84e76d8a2-r21`).  Patches 1, 2, 3 and 5
were folded into **block 08** (`ggml-cuda.cu`, `norm.cu/.cuh`, `unary.cu/.cuh`); the conv-input concat
patch 4 was folded into **block 14** (`concat.cu`).

Toolchain: `~/bin/build-llama-rocm-714` (`/opt/rocm-7.14.1-gfx102X`), `GPU_TARGETS=gfx1201`, one R9700
(`HIP_VISIBLE_DEVICES=0`).

## Gates

| gate | result |
|---|---|
| build | clean |
| `test-backend-ops` (full) | **18905/18905** |
| same-seed greedy (`Q4_K_XL`, q8_0 KV, seed 42, 300 tok) | **byte-identical to r20** (`017e51ea04b1`), `none == n1 == n3 == n7` |
| MoE MTP (35B-A3B Q4_K_M, `-ncmoe 99`, `n3`, 600 tok) | acceptance **identical** (0.77695), 55.1 → 56.9 t/s |
| combined r21 vs r20 `pp1,2,4,5,8 -n 0` (t/s) | 26.77/47.65/81.74/90.60/106.28 → 27.17/51.08/95.63/114.30/151.19 |

The multi-row `rms_norm_q8_1` path (patch 3) is only correct because patch 1 fixes the weight-row stride;
the byte-identical verify text confirms the fix (otherwise the verify rows would read past the one-row
weight).  `mmid_single` is preserved at the `rms_norm_q8_1` call site, so the multi-token `MUL_MAT_ID`
path keeps its single-token Q8_1-cache guard.

`validate-set.sh` green (strict 16/16 `git am`).
