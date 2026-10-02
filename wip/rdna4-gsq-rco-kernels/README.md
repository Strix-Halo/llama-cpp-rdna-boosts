# wip/rdna4-gsq-rco-kernels

RDNA4 kernel work for the [ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
quants on gfx1201 (R9700). All changes are bit-exact. One `git am` patch:

- `0001-cuda-RDNA4-kernels-for-the-Flash-Next-GSQ-RCO-mixes-.patch`: `mmvf.cu`, `vecdotq.cuh`, `mmq-load-tiles.cuh`,
  `mmq.cuh`, `mmvq.cu`.

**It applies on top of `wip/rdna4-dispatch-stall` (PR #78)**: r28 (tree `dc2decae`) + #78's 0001 (tree `f8a3da4c`) +
this patch = tree `205b4ea8`. Only one hunk depends on #78 (IQ3_S / IQ2_S join its row-loop types); I can rebase it
onto plain r28 if you'd rather take this one alone.

## Why this model is different

GSQ-RCO picks a quant type per tensor, so the mix is unlike Unsloth's UD-Q2_K_XL that #75 was tuned on:

- The hyper-connection weights (`hc_attn_*`, `hc_ffn_*`) are **BF16**. `build_hc_mix` takes the fused `ggml_hc_mix`
  path only for Q8_0 weights, so here they run as the generic chain with plain BF16 `mul_mat_vec_f`. That was **38 %
  of decode GPU time**, almost all of it one shape: the 10240 x 320 up-projection, launched as one 160-thread block
  per row (51200 waves, which is also a grid-stall size from #78).
- Expert types IQ3_S / IQ2_S / IQ2_XXS (gate/up) and Q2_0 (down) are new to the RDNA4 tuning.

## Changes

| Change | Scope | Case | Before | After |
|---|---|---|---|---|
| `#pragma unroll 4` on the HIP BF16 K loop of `mul_mat_vec_f` (the F32 path has it) | HIP BF16 | router 512 x 2560 / hc down 320 x 10240 / inject 4 x 10240, 1 token | 14.8 / 15.1 / 6.4 us | 5.6 / 11.8 / 2.5 us |
| `mul_mat_vec_f_vb`: one warp per row reproducing the original block exactly (see below) | RDNA4, F32/BF16, 1 K iteration, >= 3072 rows | hc up 10240 x 320, 1 / 5 tokens | 23.6 / 36.7 us | 14.1 / 32.1 us |
| IQ2_S / IQ3_S sign decode: `apply_ksigns` instead of `__vcmpne4`/`__vsub4` (grid bytes are in [1, 127]) | mmvq vec dot + mmq tile loaders | MoE mat-vec 1 / 5 tok; MoE mmq 512 tok | IQ3_S 19.6 / 82.0; 1008 us | 17.7 / 71.8; 926 us |
| IQ3_S / IQ2_S join the #78 one-token row loop | RDNA4 | dense 2048..16384 rows, K 2560 | e.g. IQ3_S 4096 rows 21.0 us | 12.6 us (-5..-42 %, no losses) |
| IQ2_XXS / IQ2_S / Q2_0 take the routed-compact MoE mmq (RDNA4-only, like IQ2_XS / IQ3_XXS since #75) | RDNA4 | 256 / 512 / 2048 tokens | Q2_0 1201 / 1515 / 2463 us | 993 / 1012 / 1668 us |

**`mul_mat_vec_f_vb`.** For short rows each thread of the original block does a single K iteration, so most of the
block's time is the two-barrier cross-warp reduction. The new kernel gives a row one warp: lane `l` keeps the partial
sums of virtual threads `l, l + 32, ...` separately, in the same K order, then does the same warp butterfly per
virtual warp and the same final butterfly over the per-warp sums (zeros past them), so every output is bit-identical
to `mul_mat_vec_f<T, float, ncols, block_size>`. It runs 4 rows per block with at most 1792 warps (the #78 stall),
and only where a sweep (K 192..512 x 3072..10240 rows x 1..8 tokens) showed a win: `(block_size / 32) * ncols <= 36`.
Fewer rows lose up to 37 %, so they keep the old kernel.

The sign-decode change is why IQ3_S / IQ2_S join the row loop: the faster dot product moved 6144..16384-row dense
launches onto the stall sizes (+20..+46 % until they took the loop).

## Bit-exactness

- `mmbit2` (new, F32/BF16 `mul_mat_vec_f`, 16 shapes x 1..8 tokens + a broadcast case, 260 cases), `mmbit3`
  (IQ2_S / IQ3_S dense + MoE, 1..512 tokens, 60) and `mmbit4` (routed-compact IQ2_XXS / IQ2_S / Q2_0 at 512 experts,
  10 used, 16..2048 tokens, 18): all byte-identical to r28 + #78. The new path was also forced on for every shape
  with a 3- and 7-block grid: identical.
- `mmbit` (the #75 set) 60/60 identical; `test-backend-ops` MUL_MAT 1297/1297, MUL_MAT_ID 929/929.
- GSQ IQ3_XXS greedy output identical, plain and with the shared MTP drafter at n-max 4 (282/466 accepted both).
- Qwen3.8-27B-UD-Q4_K_XL unchanged: tg128 29.49 vs 29.49 t/s, greedy identical plain and DFlash.

## Numbers

GSQ IQ3_XXS with 24 layers of experts on the CPU (`-ncmoe 24`, `-ub 256`), rocprofv3, clean GPU time vs r28 + #78:

| Phase | before | after |
|---|---|---|
| decode (tg64) | 2878 ms | 2376 ms (-17.4 %) |
| 5-token verify | 443 ms | 396 ms (-10.7 %) |
| prefill (pp2048) | 3556 ms | 3267 ms (-8.1 %) |

With this many experts on the CPU, decode is mostly CPU time, so end-to-end tok/s moves less than GPU time (and is
+-5..8 % noisy run to run). The CPU side is a separate PR (`wip/x86-q2_0-avx2`).

Things I looked at and left alone: a MoE item-loop `unroll 2` for IQ3_S / IQ2_S / IQ2_XXS / Q2_0 (+-1 %), and a fused
BF16 `hc_mix` (the BF16 chain is now ~3.96 ms/token against ~3.22 for the fused Q8_0 path at half the bytes, so maybe
1..2 % of decode, and not bit-exact).

I'm happy to rerun any of this on other shapes or on top of the MoE cache.
