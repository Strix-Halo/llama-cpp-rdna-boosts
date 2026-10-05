# Strata's AMD decode kernels vs our MMB/HIP kernels

**Status: OPEN / scoping (2026-10-05).** Maintainer: "worthwhile if they're better than our MMB style
kernels we have today." This note lists the candidates and the comparison method; it is not yet a
measurement.

Related: `patches/README.md` block 10 (`mmb.cu` IQ dequant, `__mul24` exact scales), block 13
(`mmvq.cu` multi-row verify blocks, `mul_mat_vec_q_moe`), block 15 (`topk-moe`, MoE gate+up+GLU),
`wip/mmvq-verify-rows/`, `~Strata` `src/kernels/cuda/iq_kernels.cu`, `src/kernels/cpu/*`, and the
gfx1201 findings in Strata `docs/AMD_HIP.md`.

## Candidates (read from the current Strata tree)

| candidate | what | their measured effect | our counterpart |
|---|---|---|---|
| **#262 packed byte ops** | IQ dequant: byte subtract/compare on four lanes at once via `v_perm_b32`/packed ops instead of the shift/or form (and one `v_perm_b32` instead of a 2-op perm) | **R9700 decode +15 %** (46.0 -> 53.0 tok/s on a 4K prompt, 52.0 -> 60.5 warm); prompts unchanged; same tokens | block 10 `mmb.cu`: already has a `v_perm_b32` note ("one instruction where the …", "LUT held in registers … applied with `v_perm_b32`"); the IQ3 dequants may still be on the shift/or form — **read and diff** |
| **`router_top10` fast path** | MoE router: drop the serial FP64 sum and the block barriers; `hip_router_fast` bounds it vs FP64 | **R9700 decode 62.4 -> 70.0 tok/s** (+1-4 % in a user's A/B); same ids and weights bit for bit | block 15 `topk-moe.cu` (MoE gate+up+GLU) and `top-k.cu`; does our router carry an FP64 sum or barriers? |
| **`fused_gr` LDS carveout** | `cudaFuncAttributePreferredSharedMemoryCarveout` on HIP for the fused gated-residual kernel (`f166564`, #646) | not stated on AMD; small | block 15 fusions / `gdn`/`hc` paths — check whether the carveout hint is set on HIP |
| **arena transparent huge pages** | `6d51272 pinned: back the Linux expert arena with transparent huge pages` | not stated | host expert arena is `hipHostMalloc` (r15 `LLAMA_MMAP_HOST_EXPERTS`); no THP advice |
| **`#646` IQ-grid staging** | the single-matrix mmvq does **not** stage the IQ grids (10-20 % slower on RTX 5070 at 3-8 cols) — i.e. they chose *not* to stage | 10-20 % at verify widths | block 13 mmvq; check our IQ grid handling at 2-8 columns |
| **`verify` zero-doorbell + sub-warp packing** | `cfd3b72` (CUDA): verify-graph doorbells, sub-warp expert packing, shared-mem staging, batched PLE/MTP | CUDA only; principle may port | block 11 verify graphs; our verify fusions (r11) |

## Method

1. **Read both kernels side by side** and classify: instruction mix (perm/packed vs shift/or, FP64 vs
   FP32 router reduction), grid/block mapping, LDS usage, and whether the difference is already in our
   block 10/13/15 work. Only the ones with a real delta go forward.
2. **Microbench** the qwen4exp geometry (48 layers, 512 experts, top-10, `n_embd 2560`, `head_dim 256`)
   and the dense 27B geometry, on gfx1201 / ROCm 7.14.1, against the r12 build. A/B with env switches
   where the block already has them (`GGML_CUDA_*`).
3. **End-to-end**: single R9700, IQ3_XXS + MTP head, `MOE_EXPERT_CACHE_MIB=20480` (the auto-size
   campaign's target), `-ncmoe 48`, MTP n3, shallow + 32K. Also the fully-resident / multi-GPU case,
   where our decode is not PCIe-bound and a kernel win should translate more directly.
4. **Purity**: any change must keep `W = 1..8` bit-identical (band-uniform knobs, `GREEDY-PURITY.md`
   §19/§24/§25) and pass `llama-batched-bench -npl 1,4,8` stock-relative (`mtp-adaptive-methodology.md`
   rule 5). A kernel that changes the reduction order is not shippable here.

## Caveat on the end-to-end ceiling

At 43 % residency our single-card decode is partly **UVA-cold-read bound** (the other campaign's
finding), so a 15 % kernel win will not fully show end-to-end there. The honest targets are the
fully-resident and multi-GPU/short-K cases. Still worth it: these are HIP/RDNA4 kernels, exactly this
repo's scope, and several are shaped like upstream PRs (`upstream/`).

## Milestones

* **K0 — Kernel inventory + diff** (no build): the table above filled in with our source line-by-line,
  and a shortlist with expected deltas.
* **K1 — Microbench the shortlist** on gfx1201.
* **K2 — Port the winners** (RDNA-gated, per-change kill switch, bit-identical where the reduction order
  allows).
* **K3 — Gates + promotion decision** (delivery block amendment, and/or `upstream/` copy).

## Attribution

Strata is MIT. Any transcribed kernel should carry the upstream attribution the delivery already uses
for the i-quant dequantizers in `~Strata third_party/ggml/LICENSE`.
