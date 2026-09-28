# Maintainer verification for r21 (PR #57)

Integrated 2026-09-28 into the delivery as a **block-10 + block-13 amendment** (release-in-progress
`v16-84e76d8a2-r21`).  The contributor's three patches were folded into the fork block commits: the
multi-row launch into **block 13** (`mmvq.cu`), the `ggml_cuda_mul_small()`/`__mul24` changes into
**block 10** (`vecdotq.cuh`).

Toolchain: `~/bin/build-llama-rocm-714` (`/opt/rocm-7.14.1-gfx102X`), `GPU_TARGETS=gfx1201`, one
R9700 (`HIP_VISIBLE_DEVICES=0`).  The author's table was tuned on **ROCm 10.0**; this is the ROCm 7.14
re-check.

## Gates

| gate | result |
|---|---|
| build | clean |
| `test-backend-ops` (full) | **18905/18905** |
| same-seed greedy (`Q4_K_XL`, q8_0 KV, `prompts/reasoning.txt`, seed 42, temp 0, 300 tok) | **byte-identical to r20**, `none == n1 == n3 == n7 == 017e51ea04b1` |
| `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` (q8_0 KV) | B=1 28.24 -> 28.32, **B=4 78.67 -> 89.11 (+13 %)**, **B=8 93.53 -> 125.66 (+34 %)** |
| MTP (`-n 2000`, prose, q8_0 KV) | acceptance **identical** (n3 0.82849, n7 0.62347); `n3` 58.9 -> **65.7 t/s (+11.5 %)**, `n7` 53.6 -> **69.4 t/s (+29.5 %)** |

## Per-type table on ROCm 7.14 (`test-backend-ops perf`, m=4096 `k=14336`, 3 repeats)

Transfers cleanly for the main types (q3_K +7..+24 %, q4_K +8..+38 %, q6_K +3..+27 %, q2_K +12..+26 %,
plus **+14 % at n=1** from `__mul24`).  Two ROCm-7.14-specific dips vs the author's ROCm 10.0 sweep:

| type | n=2 | n=3 | n=4 | n=8 |
|---|---:|---:|---:|---:|
| q5_1 | -3.3 % | -4.2 % | -2.4 % | +12.2 % |
| iq4_xs | -2.6 % | -4.4 % | +1.3 % | +10.9 % |

The aggregate verify band is a large net win (the common MTP/DFlash widths are 4 and 8), so the table is
kept as tuned; a per-type retune for q5_1/iq4_xs on the 7.14 compiler is an open follow-up.  The earlier
single-run `q8_0 n=1 -29 %` was measurement noise (repeats: parity).

`validate-set.sh` green (strict 16/16 `git am`).
