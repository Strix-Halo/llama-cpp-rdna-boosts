# wip/x86-q2_0-avx2

An AVX2 `ggml_vec_dot_q2_0_q8_0` for x86, bit-identical to the generic one, on top of release r28 (tree `dc2decae`).
One `git am` patch, CPU only (`ggml-cpu/arch/x86/quants.c`, `ggml-cpu/arch-fallback.h`):

- `0001-ggml-cpu-x86-AVX2-Q2_0-x-Q8_0-dot-product-bit-identi.patch` (r28 -> tree `119065e4`).

## Why

x86 has no Q2_0 dot product: `arch-fallback.h` aliases the scalar `ggml_vec_dot_q2_0_q8_0_generic` (only ARM has a
SIMD one). On Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS the down experts of 30 of the 48 layers are Q2_0, and on a 64 GB box
the experts of 24 layers run on the CPU. `perf` during decode (Ryzen 9 9900X):

| | share of all samples |
|---|---|
| libgomp (threads waiting) | 52.6 % |
| **`ggml_vec_dot_q2_0_q8_0`** | **28.4 %** |
| `ggml_vec_dot_iq2_xs_q8_K` / `iq3_s` / `iq2_xxs` | 3.7 / 3.5 / 3.0 % |

## How

Per 32-weight half block (8 packed bytes against one Q8_0 block):

- broadcast the 8 bytes; one in-lane `pshufb` gives each 128-bit lane its 4 bytes four times;
- `srlv_epi32` by 0 / 2 / 4 / 6 and `& 3`: element `e` of a lane now holds field `e` of each of its 4 bytes (a 4x4
  transpose of the 2-bit weights); one more `pshufb` gives the Q8_0 bytes the same transpose;
- `maddubs(q, y) - maddubs(1, y)` is the pairwise sum of `(q - 1) * y`, exact (|pair| <= 762), then `madd` and a
  horizontal add.

The integer sums are exact. The generic code compiles to a separate multiply and add for `sumi += d1 * sumi_block` and
`sumf += d0 * sumi` (no FMA), so the new code does them with `_mm_mul_ss` / `_mm_add_ss`, which GCC does not contract.
A first version with plain C there got FMAs and differed in the last bit, which is how I found this.

## Bit-exactness

- A CPU-backend byte check (Q2_0 dense 2560 x 640 and 1000 x 2560, and MoE with 512 experts / 10 used, 1..64 tokens,
  21 cases) is identical to r28.
- `test-backend-ops -b CPU -o MUL_MAT` passes for every q2_0 case.
- GSQ IQ3_XXS greedy output identical (plain and MTP-drafted).

## Numbers (GSQ IQ3_XXS, `-ncmoe 24`, three interleaved rounds)

| | r28 (+ GPU patches) | + this patch |
|---|---|---|
| decode tg64 | 27.3 / 26.6 / 28.7 t/s | 36.6 / 34.3 / 36.3 t/s (~+30 %) |
| 5-token batch | ~45 t/s | 55..66 t/s |
| greedy 400 tokens, plain / MTP n-max 4 | 26.3..28.7 / 30.3..31.5 t/s | 35.0 / 43.6 t/s |

Afterwards `ggml_vec_dot_q2_0_q8_0` is 7.9 % of samples. Threads: `-t 16` was a little ahead of the default 12 on this
12-core box (tg 37.6 vs 37.2, pp5 73 vs 64).

**On top of `wip/moe-expert-cache` (`exp23`).** In case it's useful for the cache's validation: with `exp23` applied
on r28, plus #78, the GSQ GPU patch and this one (all applied cleanly), a single R9700 running GSQ IQ3_XXS at
`-ncmoe 24` gives decode 37.4 -> 47.9 / 49.5 t/s at `MOE_EXPERT_CACHE_MIB=1024 / 2048` (llama-bench). In a 4-slot
llama-server it fits at 64k / 80k / 96k / 112k context with `MIB=1024` (the arena sizes itself from free VRAM), serves
near-limit prompts (110.7k tokens, 636 t/s prefill) and greedy output stays identical to the cache-off build. That's
what I'm running in production now.
