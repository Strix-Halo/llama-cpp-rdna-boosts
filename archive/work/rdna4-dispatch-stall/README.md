# wip/rdna4-dispatch-stall

A follow-up to PR #75 for gfx1201 (R9700), on top of release r28 (tree `dc2decae`). One `git am` patch:

- `0001-cuda-RDNA4-mmvq-avoid-the-gfx1201-grid-size-dispatch.patch`: `mmvq.cu` only (r28 -> tree `f8a3da4c`).

It reworks PR #75's one-token two-rows-per-block default (`GGML_MMVQ_RDNA4_WEIGHT_RPB1`). While testing #75 on a dense
27B, I found that its biggest wins and losses were mostly not about rows per block. They came from a GPU-side slowdown
at certain grid sizes, which the two-row default happened to dodge for some shapes and walk into for others. This
patch keeps the two rows where they genuinely help and keeps the launch off the bad grid sizes.

## The stall

On gfx1201, a launch at certain grid sizes takes a fixed ~8 µs longer than its neighbours. Nothing else changes:
same kernel, same data (cache-hot), only `gridDim.x`. A standalone HIP repro with no ggml involved (one wave32 block
per row, each summing its row; 1760-byte rows, average of 400 launches):

| workgroups | µs / launch |
|---|---|
| 2032 | 5.30 |
| **2048** | **13.38** |
| 2064 | 5.41 |
| 4064 | 7.28 |
| **4096** | **16.13** |
| 4128 | 7.37 |
| 6112 | 8.46 |
| **6144** | **18.06** |

What I've been able to pin down:

- It tracks the **total wave count** near multiples of 2048: 1-wave × 2048 blocks, 4-wave × 512 and 8-wave × 256 all
  stall, and so do 2D grids such as 1024 × 2.
- It is **not memory**. It happens with hot data, and in the real kernel remapping which rows the blocks read changes
  nothing. Adding one empty block moves it.
- Plain launches and hipGraph launches behave the same. Which sizes stall depends on how long each block runs, so
  it can't be avoided by skipping a fixed list of sizes.

I couldn't find it reported anywhere, so I filed it with the repro as
[ROCm/TheRock#8634](https://github.com/ROCm/TheRock/issues/8634) (also
[ROCm/legacy-rocm-build#6689](https://github.com/ROCm/legacy-rocm-build/issues/6689)). No reply yet. The patch
doesn't depend on the cause.

Environment: kernel 7.0.0-34, amdgpu-dkms 7.1.3, HIP 7.15 (ROCm core 10.0), VBIOS 113-APM107573-100, CP firmware
MEC 3430 / PFP 3180 / ME 3080 (amdgpu-dkms-firmware 31.50). The stall is unchanged with the newest GC 12.0.1 firmware in
linux-firmware (commit `caf919cc`, MEC 3450 / PFP 3190 / ME 3090): same repro numbers within ±0.8 µs.

## What the patch does

All RDNA4-only (`MMVQ_PARAMETERS_RDNA4`); other GPUs compile and run as before.

**Row loop.** One-token dense decode for Q4_K / Q5_K / Q6_K / IQ4_XS with 2048..16384 rows: when the launch would
have more than 1792 blocks (`GGML_MMVQ_RDNA4_DECODE_GRID`), a balanced grid of at most 1792 blocks walks the rows
instead, block b taking row group b, b + grid, ... (a group is one or two rows). The loop wraps the existing kernel body
in place. Every other instantiation `break`s after one pass; I checked by disassembly (on r26) that their ISA is
unchanged.

**Rows per block, one token**, chosen from per-type row sweeps:

| Type | r28 | This patch |
|---|---|---|
| Q4_K, Q5_K | 2 | 2 (looping above 1792 blocks); 1 for long-K above 16384 rows |
| Q6_K | 2 | 1 (looping); 2 for long-K at 2176..3584 rows |
| IQ4_XS | 2 | short-K from 6144 rows: 2 (looping); otherwise 1 (looping) |
| Q8_0 short-K (8-warp blocks), Q3_K | 2 | 2 (no loop: Q3_K loses 8..35 % in it) |
| everything else | 2 | 1, as before #75 |

Two rows always need an even row count of at least 1024, as in r28.

**Verify, 4..8 tokens**, Q8_0 short-K (8-warp blocks): 2 rows instead of 1. This keeps #71's finding that 4 rows are
slow for that block. Two rows beat one from 4 tokens up; at 2..3 tokens they don't, so those keep one row.

## Bit-exactness

Every row keeps its own one-warp dot product; only which block computes it changes.

- `mmbit`: 60/60 cases byte-identical to r28. Also with the loop forced on for every row count and a 7-block grid,
  at one and two rows per block (a test-only build).
- `test-backend-ops` on ROCm0: MUL_MAT 1297/1297, MUL_MAT_ID 929/929.
- Qwen3.8-27B-UD-Q4_K_XL greedy output identical to r28, plain and with DFlash n-max 4 (276/485 drafts accepted in
  both; sha `50afec90ceba`).

## Numbers (gfx1201, r28 vs r28 + this patch)

**Kernel sweep, one-token decode** (mmbench, rocprofv3 cold-cache medians; 97 shapes: Q3_K / Q4_K / Q5_K / Q6_K /
IQ4_XS / IQ4_NL / IQ3_S, 1536..248320 rows, K 2560..17408). r28 was measured twice to show the noise:

| | sum | best shape | worst shape | shapes > +1 % |
|---|---|---|---|---|
| r28, second run | −0.02 % | −0.7 % | +0.9 % | 0 |
| patched | −2.09 % | −26.5 % | +1.2 % | 2 |

The two shapes over +1 % are IQ4_XS 7680×2560 (20.12 → 20.36 µs) and 4064×2560 (11.48 → 11.60 µs). The big wins are
the stall sizes, e.g. IQ4_XS 8192×2560 28.6 → 21.0 µs, Q5_K 4096×5120 33.0 → 24.6 µs, IQ4_XS 12288×2560 37.1 → 30.4 µs.

**Kernel sweep, verify**, Q8_0 at K 640 / 2560 / 2880, 512..10240 rows: −14.1 % / −17.1 % / −14.9 % / −3.2 % in sum at
4 / 5 / 6 / 8 tokens, ±0.2 % at 2..3 tokens. Worst shape +3.7 % (4096×640 at 8 tokens).

**Qwen3.8-27B-UD-Q4_K_XL, llama-bench, 2 rounds:**

| | r28 | patched |
|---|---|---|
| tg128 | 29.13 / 29.11 t/s | 29.46 / 29.47 t/s (+1.2 %) |
| pp512 | 1450 / 1420 t/s | 1424 / 1421 t/s |
| pp4096 | 1422 / 1402 t/s | 1401 / 1402 t/s |

Prefill doesn't use these kernels; the first r28 round ran first, from an idle GPU, and the second rounds match.

**Qwen3.8-Flash-Next UD-Q2_K_XL (22 layers of experts on the CPU)**, rocprofv3, runs interleaved r28 / patched /
r28 / patched. On this setup, total GPU time moves ~1 % between two runs of the same build, which is more than this
patch changes, so here are the dense mmvq kernels it touches (summed time of calls under 5× the median):

| Kernel | Phase | r28 (two runs) | patched (two runs) |
|---|---|---|---|
| `ksplit<Q5_K>` | tg64 | 630.9 / 630.2 ms | 615.5 / 615.8 ms (−2.4 %) |
| `ksplit<Q6_K>`, `<Q4_K>`, `<Q8_0>` | tg64 | 387.6 / 385.0 ms | 386.9 / 386.8 ms |
| `ksplit<Q8_0>` (5-token verify) | pp5 | 13.0 / 13.1 ms (12.60 µs median) | 10.4 / 10.4 ms (10.00 µs) |

That's about −0.6 % of Flash-Next's decode and verify GPU time. Its weights mostly avoid the stall sizes already, and
#75's two rows were a net win there; the patch keeps that and removes the losses #75 caused on other shapes, such as the
dense 27B. Prefill (mmq) isn't touched.

I'm happy to rerun any of this on top of the MoE cache, or on other shapes, once it lands.
