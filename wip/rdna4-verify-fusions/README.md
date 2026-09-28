# wip/rdna4-verify-fusions — small-kernel fusions for the speculative verify band

Five `git am` patches for r19 (`5f2fc93fa`). They touch `ggml-cuda.cu`, `norm.cu/.cuh`, `unary.cu/.cuh` and
`concat.cu`; `mmvq.cu` and the fattn files are untouched, so they are independent of #57 and #62.
Measured on r19 + #57 + #62 (gfx1201, R9700); the patched files are byte-identical with and without those
two series underneath.

| patch | what | bit-exact | kill switch (default on) |
|---|---|---|---|
| `0001` | `rms_norm_q8_1`: broadcast the single-row norm weight (latent multi-row bug) | yes (decode unchanged) | — |
| `0002` | `rms_norm` + `scale` (the GDN q/k l2 norm) in one kernel, every width | yes | `GGML_CUDA_FUSE_RMS_SCALE` |
| `0003` | norm → Q8_1 and gated unary → Q8_1 cache prefill for 2..8 tokens (RDNA4) | yes | `GGML_CUDA_FUSE_Q8_1_VERIFY` |
| `0004` | GDN conv-input concat: tiled transpose kernel for 2..8 tokens | yes (a copy) | `GGML_CUDA_CONCAT_VERIFY` |
| `0005` | GDN gate chain `add(dt)` → `softplus` → `mul(a)` in one kernel for 2..8 tokens (RDNA4) | yes | `GGML_CUDA_FUSE_GDN_GATE` |

## Why

A 5-token target ubatch (one DFlash2 n-max-4 verify pass) of the dense 27B launches 2,171 kernels, against
1,485 for one token: several decode fusions are gated to `ne[1] == 1`. On this card each launch costs ~1-3 us
of kernel time plus a sub-5 us dispatch gap, so the verify pass carries ~700 extra small launches.

## 0001 — `rms_norm_q8_1` weight stride

`ggml_cuda_op_rms_norm_q8_1` passes `weight->nb[1]` as the weight row stride, and the kernel offsets the weight
by `row * stride`. The norm weight is a single row, so rows >= 1 read past its end. Decode never reaches this
(the fusion is taken only for single-token matmuls, so `row == 0`); 0003 does. Stride 0 for a single-row
weight, as the unfused `rms_norm` + `mul` broadcast does. (Found because the probe hashes changed while
`width_purity` still said PASS: every width >= 2 was wrong in the same way.)

## 0002 — `rms_norm` + `scale`

`build_gdn_l2_norm` is `ggml_scale(ggml_rms_norm(x, eps/n), 1/sqrt(n))`. `rms_norm_scale_f32` rounds the norm
output as `rms_norm_f32` writes it and applies `s * v + b` as `scale_f32` does. 96 launches fewer per token
at every width (48 GDN layers × q, k).

## 0003 — Q8_1 prefill for the verify band

`rms_norm_q8_1` and `unary_mul_q8_1` are row-generic but gated by `ggml_cuda_should_fuse_mul_mat_vec_q(mm)`,
which requires a single-token matmul. They leave the matmul unchanged (they only pre-fill its quantize cache),
so a `verify_band` flag lets these two call sites take RDNA4 2..8-token matmuls. MUL_MAT_ID keeps its guards.
192 launches fewer per 5-token pass.

## 0004 — conv-input concat

`concat_transposed_src1_dim0` was taken only from 32 new tokens; 2..8 fell back to `concat_non_cont`, one
mostly idle 128-thread block per channel (~10k blocks, 13 us per GDN layer). The tiled kernel bounds-checks
the token index on load and store, so it is correct for any count; from 2 tokens it takes ~3 us.

## 0005 — GDN gate chain

Decode fuses the gate into `ssm_gate_beta`. For 2..8 tokens it ran as `k_bin_bcast(add)`, `softplus`,
`k_bin_bcast(mul)`: the existing unary+mul fusion needs equal shapes and `dt` / `ssm_a` are per-head vectors
broadcast over the tokens. `add_softplus_mul_bcast_f32` does `op_softplus(x + dt[h]) * a[h]` in one pass.
96 launches fewer per verify pass.

## Results (27B UD-Q4_K_XL, q8_0 KV)

Kernels per 5-token pass 2,171 → 1,787. `llama-bench -p N -n 0` (one ubatch of N tokens, t/s):

| | pp1 | pp2 | pp4 | pp5 | pp8 |
|---|---|---|---|---|---|
| r19 + #57 + #62 | 28.5 | 52.0 | 98.9 | 119.3 | 159.8 |
| + 0001..0005 | 28.7 | 53.7 | 101.8 | 122.1 | 163.2 |

Server, 262k, DFlash2 Q4_K_M n-max 4, greedy, 768 tokens, two rounds each (t/s):

| | short code | short reasoning | doc 50k | code 97k |
|---|---|---|---|---|
| all switches off | 84.9 | 70.0 | 61.65 | 56.55 |
| all on | 86.7 (+2.1 %) | 71.6 (+2.3 %) | 63.0 (+2.2 %) | 57.65 (+1.9 %) |

All four texts are byte-identical between the arms.

## Gates

- **`test-logits-width-probe`** (`prose-rdna-boosts.txt 1024 512`), 7 runs on 5 models (27B UD-Q4_K_XL q8_0
  `RS=from_w` / `RS=0` / f16, 27B IQ4_XS, 27B NVFP4, 35B-A3B, gemma-4-26B MXFP4): PASS, per-W and row0/row1
  hashes identical with all switches off vs on, and identical to our r17 + #57 + #62 production build.
- **Full `test-backend-ops`:** 18905/18905.

## Tried, didn't help

Lifting the `ncols_dst == 1` gate on the mmvq epilogue fusions (gate/up + GLU, `+ADD` residual, dual K/V
output) is bit-exact but makes the verify pass 12-20 % slower (pp5 119 → 95-105): at 2..8 columns the
ksplit kernel is issue-bound and the `has_fusion` instance is 2-3× slower per launch even for `+ADD` only
(q5_K, 5120 rows, 5 columns: 45 → 101 us with a compile-time gate-free instance). Fusions that stay out of
the mmvq kernel are the ones that pay.

A measurement note: `llama-bench -p N` zeroes the recurrent state every repetition, so its pp profiles show
96 `scale_f32` (`ggml_scale_inplace(state_zero, 0)`) per pass that a server verify pass never launches.

Only RDNA4 was measured. 0002 and 0004 are not RDNA4-gated (0002 is exact on any backend; 0004 is a copy).
