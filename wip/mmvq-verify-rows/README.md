# Faster multi-token mmvq on RDNA4 (bit-exact)

Three commits to `ggml/src/ggml-cuda/mmvq.cu` and `vecdotq.cuh` (+106 / −32 lines). They speed up the dense mmvq weight kernel (`mul_mat_vec_q_ksplit`) for 2–8-token batches, which are the speculative verify step and small multi-slot decodes. They also speed up the k-quant and i-quant dot products at every width.

Per-row arithmetic is unchanged: same thread mapping, K order and warp reduction. So `W = 1..8` stays one hash, and the logits are bit-identical to the unpatched build.

The patches `git am` cleanly on top of `scripts/apply-all.sh` for `v16-84e76d8a2-r17` (applied tree `dc7ce12a…`, matching `release.json`). Each commit builds on its own. Earlier versions of the first commit (a flat 4 rows) and of the multiply commits were developed and tested on r9 and r16; the final series was validated on r17.

## The commits

1. **Multi-row blocks for `ncols_dst` 2..8 on RDNA4** (`mmvq.cu`).
   - **The problem:** `calc_rows_per_block` falls through to 1 for every RDNA table, and RDNA4's nwarps is 1. So a 4-token verify re-read all four q8_1 activation columns once per weight row, about 8× the weight bytes, served from cache. At n = 4, every k-quant was issue-bound at ~400–490 GB/s-equivalent even when cache-resident. The Q6_K lm_head (248320 × 5120) streamed at 577 GB/s from DRAM at n = 1, but only 430 at n = 4.
   - **The fix:** more rows per block share the activation loads. The ISA shows the compiler CSEs them, and the Q4_K "sum of u" dp4a, across the rows.
   - **Rows per block, from a gfx1201 sweep** (1/2/4 rows × every quantized type × n = 2..8 × four shapes). The first flat-4 version was 41–62 % slower on NVFP4 and 10 % slower on Q1_0, which is why the rule is per type:

     | rows per block | types |
     |---|---|
     | 1 | NVFP4, Q1_0 |
     | 2 | IQ2_XXS, IQ2_XS, IQ3_XXS, Q2_0, Q5_1 |
     | 4 | every other quantized type (`GGML_MMVQ_RDNA4_WEIGHT_RPB`) |
     | Q2_K | 4, but 2 at 7..8 columns (the 4-row kernel exceeds 256 VGPRs and spilled: 160 → 277 µs at n = 8) |

     A per-(type, n) rule would add at most 2.5 % (IQ4_XS), so it stays per type.
   - **One-row fallback, chosen on the host and only instantiated for HIP at 2..8 columns:**
     - when `nrows % rows_per_block != 0`. A multi-row block reads every row it owns, so such a weight would read past its end. The results are discarded, but the read could fault at an allocation boundary. This is covered by the 16 odd-`m` multi-token k-quant cases in `test-backend-ops -o MUL_MAT`.
     - when the launch would have fewer than **512 blocks** (`GGML_MMVQ_RDNA4_WEIGHT_MIN_BLOCKS`). Short weights lose from multi-row blocks because the GPU is no longer filled: 48-row SSM `alpha`/`beta` projections were **26 % slower** at 4 rows. A sweep over 48/1024/2048/3072/17408 rows puts the cut at 512: 2048 rows keep +21 %, and 48 rows are back to parity.
2. **`__mul24` for the k-quant scale multiply (HIP only)** (`vecdotq.cuh`). The `dp4a sum × 6-bit scale` multiplies in the Q3_K–Q6_K mmvq dot products compiled to `v_mul_lo_u32`, which is quarter rate on RDNA: 64 of the 574 instructions in the Q4_K n = 4 loop. The operands are tiny, so `ggml_cuda_mul_small()` (`__mul24` → `v_mul_i32_i24`) is exact. CUDA keeps the plain `*`, since `__mul24` isn't faster there.
3. **The same for Q2_K and the i-quants** (`vecdotq.cuh`). Q2_K, IQ2_XXS/XS/S, IQ3_XXS/S and IQ4_XS had 16–80 `v_mul_lo_u32` per iteration (`sumi × ls`, operands < 2^19).

## Results on r17 (one R9700 gfx1201, ROCm 10.0, PCIe 5.0 x16; stock r17 vs r17 + this series)

**End-to-end decode**, `llama-server -c 131072…262144 -ctk q8_0 -ctv q8_0 --reasoning-effort low`, 768 greedy tokens, prefix cached so the timing is decode-only. Greedy text is **byte-identical in all 26 prompt pairs**.

| model / speculation | short code | short reasoning | 50k-token doc | 97k-token code |
|---|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL, DFlash2 n-max 3 (round 1) | 66.8 → 73.8 (+10.5 %) | 56.8 → 63.0 (+10.9 %) | 50.2 → 55.1 (+9.8 %) | 43.5 → 47.2 (+8.5 %) |
| same, round 2 | 66.5 → 73.8 (+11.0 %) | 56.7 → 63.0 (+11.1 %) | 50.2 → 55.1 (+9.8 %) | 43.5 → 47.0 (+8.0 %) |
| Qwen3.8-27B UD-Q4_K_XL, **draft-mtp n-max 7** (W = 8) | 54.9 → 71.3 (**+29.9 %**) | 38.5 → 50.1 (**+30.1 %**) | 55.0 → 69.4 (**+26.2 %**) | 41.8 → 51.1 (**+22.2 %**) |
| Qwen3.8-27B UD-IQ4_XS, DFlash2 n-max 3 | 72.6 → 79.5 (+9.5 %) | 62.3 → 68.3 (+9.6 %) | 56.8 → 61.7 (+8.6 %) | 48.4 → 52.0 (+7.4 %) |
| Qwen3.8-27B NVFP4 (MTP head), draft-mtp n-max 7 | 48.3 → 48.2 (−0.2 %) | 35.0 → 34.9 (−0.3 %) | 54.8 → 54.9 (+0.2 %) | 37.4 → 37.5 (+0.3 %) |
| Qwen3.6-35B-A3B Q4_K_M, no speculation | 93.7 → 93.7 | 93.7 → 93.7 | 80.1 → 80.2 | – |
| gemma-4-26B-A4B MXFP4, no speculation | 86.8 → 86.9 | 86.8 → 86.9 | 72.9 → 73.0 | – |

The gain grows with the verify width, which makes longer drafts relatively cheaper. NVFP4 is at parity by design (1 row). MoE without speculation doesn't change, because single-token decode and `mul_mat_vec_q_moe` take other paths; that kernel only shares the (exact) dot products.

**Kernel, all types** (`test-backend-ops perf -o MUL_MAT`, m = 4096, k = 14336), speedup vs stock r17:

| type | n=1 | n=2 | n=3 | n=4 | n=5 | n=8 |
|---|---|---|---|---|---|---|
| q4_0 | −0.6 | +59.6 | +58.7 | +56.7 | +77.5 | +109.8 |
| q4_1 | −0.5 | +18.9 | +47.9 | +42.5 | +67.5 | +66.8 |
| q5_0 | −0.8 | +16.0 | +27.1 | +28.9 | +49.5 | +70.9 |
| q5_1 | −0.8 | +12.1 | +11.5 | +16.4 | +18.9 | +34.0 |
| q8_0 | −0.2 | +24.8 | +17.9 | +38.2 | +41.9 | +55.7 |
| q2_K | +7.9 | +28.0 | +28.8 | +31.1 | +34.1 | +35.2 |
| q3_K | +5.5 | +16.5 | +23.0 | +31.2 | +35.1 | +48.5 |
| q4_K | +7.2 | +34.6 | +40.4 | +43.8 | +50.0 | +66.1 |
| q5_K | +7.1 | +31.8 | +35.3 | +42.4 | +45.7 | +58.6 |
| q6_K | +4.2 | +23.6 | +29.2 | +35.5 | +44.7 | +59.4 |
| iq1_s | −0.5 | +45.4 | +70.2 | +71.2 | +88.5 | +104.1 |
| iq1_m | −0.5 | +21.4 | +34.8 | +41.7 | +49.2 | +87.8 |
| iq2_xxs | +0.6 | +12.3 | +16.4 | +19.0 | +15.1 | +23.4 |
| iq2_xs | +1.2 | +13.7 | +15.8 | +22.3 | +31.9 | +28.6 |
| iq2_s | +1.4 | +9.0 | +13.1 | +31.8 | +47.1 | +31.9 |
| iq3_xxs | +0.6 | +14.7 | +16.8 | +26.5 | +25.3 | +21.7 |
| iq3_s | +0.5 | +11.6 | +9.8 | +16.6 | +28.7 | +43.0 |
| iq4_nl | −0.2 | +40.2 | +39.5 | +34.6 | +53.9 | +77.9 |
| iq4_xs | −0.2 | +28.6 | +36.3 | +41.4 | +53.7 | +62.1 |
| mxfp4 | −0.6 | +31.1 | +27.9 | +29.5 | +43.5 | +70.4 |
| q2_0 | −0.1 | +11.6 | +35.1 | +40.8 | +45.3 | +54.5 |
| nvfp4 | −0.5 | +0.0 | −0.2 | −0.1 | −0.2 | −0.2 |
| q1_0 | −0.9 | +0.1 | −0.1 | +0.1 | +0.1 | +0.1 |

Nothing is below −1 % (run-to-run noise); f16/bf16/f32 don't use this kernel and are within ±0.3 %. **llama-bench** pp512/tg128 is unchanged on 27B Q4_K_XL (1414/29.4 → 1407/29.4), 27B IQ4_XS (1393/33.6 → 1389/33.6) and 35B-A3B (5111/98.3 → 5095/98.7).

## Gates (all on r17, stock vs patched)

- `test-backend-ops test` (full suite): **0 failures on both**.
- `test-logits-width-probe <model> prompts/prose-rdna-boosts.txt 1024 512`: **`width_purity=PASS (worst maxdiff 0)` on all 7 runs, with per-W hashes identical between stock and patched.** Runs: 27B UD-Q4_K_XL, 27B UD-IQ4_XS, 27B NVFP4, 35B-A3B Q4_K_M and gemma-26B MXFP4 at `KV=f16 RS=0`, plus 27B Q4_K_XL and 35B-A3B at `KV=q8_0 RS=from_w`.
- Scratch spills: the list of mmvq kernels with a private segment is **identical to stock** (12 pre-existing Q1_0 fused instances). Checked from `-S --cuda-device-only` ISA of `mmvq.cu`.
- Not run: the full `GREEDY-PURITY.md` matrix, `-sm tensor` / multi-GPU, CPU-offloaded experts, other architectures.

## Scope and what we could not test

- **Rows per block is RDNA4-only** (`MMVQ_PARAMETERS_RDNA4`). RDNA3_0/3_5 would opt in through `calc_rows_per_block_weight`, but we have no hardware. Your gfx1100/gfx1151 sweeps show these knobs don't transfer blindly.
- **The multiply change covers all AMD** (`GGML_USE_HIP`) and is exact everywhere. It was measured only on gfx1201; `v_mul_lo_u32` is quarter rate on earlier AMD generations too, but we haven't measured those.
- The Q2_K spill cap and the per-type table are tuned against this ROCm 10.0 compiler. A different compiler could move a VGPR boundary. Re-running the spill-list check (a scratch segment on any `mul_mat_vec_q_ksplit` instance) would catch that.
