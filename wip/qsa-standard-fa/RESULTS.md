# RESULTS — QSA on the standard FA kernels (Phase 0-2)

**Session:** 2026-10-05 (follows the `wip/qsa-standard-fa/README.md` handover, after
`v16-a55e952b8-r4`).
**Status:** **Phase 0 complete; Phase 1 adopted; Phase 2 partially adopted (valid-shape AMD
enablement); Phases 3-7 deferred** with concrete blockers.  Nothing promoted to `patches/` — this
tree is investigation only (WIP rule).  The code lives on the fork branch **`qsa-standard-fa`** off
`cd1485fd1`; the diff is `results/phase0/phase1-2-port.patch`.

## TL;DR

Upstream's `n_kv_max` sparse mechanism (mask compaction + MMA index gather) is now **ported to HIP
and validated on gfx1201**.  It is off the delivery path by default (only `FLASH_ATTN_EXT` cases with
`n_kv_max > 0` reach it), so the CHANGE IS INERT in the delivery.  Two findings decide the rest of
the campaign:

1. **The AMD WMMA/MFMA kernel only instantiates the sparse loader for tiles of `ncols1*ncols2 >= 16`
   with `ncols2 != 1`** (`flash_attn_ext_f16`'s `AMD_WMMA_AVAILABLE` guard).  Three of the four
   upstream sparse shapes — `(512,512,1,8)`, `(256,256,1,8)` — are therefore **invalid on AMD**;
   only `(576,512,1,16)` and `(256,256,8,8)` can be launched.  qwen4exp's **decode** width
   (`n_q <= 4`, gqa 12) has no NVIDA-shaped AMD instantiation at all.
2. The NVIDIA engagement bound `K->ne[1] >= max(4096, 2*n_gather)` rejects most shallow/narrow
   `n_kv_max` hints, so even the valid shapes only engage at long context.

Consequently the "route qwen4exp through the standard sparse path" (Option B) is **deferred**: it
needs (a) a new `(256,256,*,16)` AMD sparse instantiation for the decode width, and (b) a
model-level A/B against the fused `FLASH_ATTN_QSA` that this session did not establish.  Option A
(HIP sparse support) is a clean, independently valuable win and is the adopted part.

## Adopt / defer / drop

| task | verdict | evidence |
|---|---|---|
| **P0** baseline + harness | **done** | `FLASH_ATTN_QSA` 26/26, `FLASH_ATTN_EXT` 6358/6358 (ROCm0/1/2), oracle sweep 59/59 (pre- and post-change). |
| **P1** compact the mask on HIP | **adopt** (non-destructive) | `flash_attn_mask_to_sparse_indices` + `ggml_cuda_flash_attn_ext_compact_mask` unguarded for HIP; `__ballot_sync` wrapped for the 64-bit AMD member mask (`ggml_cuda_ballot`).  `FLASH_ATTN_EXT` 6358/6358 after. |
| **P1** harness that proves the path runs | **adopt** | `GGML_CUDA_FA_SPARSE_TRACE=1` prints the `shall_use_sparse` decision (`[FA_SPARSE_CHECK]`).  Prevents the "green = dense fallback" trap the README warns about. |
| **P2** AMD sparse predicate + kill-switch | **adopt** | `shall_use_sparse` is arch-aware (WMMA/MFMA on AMD), `GGML_CUDA_FA_SPARSE=0` disables.  AMD guard rejects invalid `ncols`. |
| **P2** instantiate the sparse MMA arm on AMD | **adopt for `(576,512,1,16)`, `(256,256,8,8)`** | both launch and pass; see trace below. |
| **P2** qwen4exp-shaped head-256 dispatch | **adopt (wide, n_q>4 only)** | `switch_ncols2<256,256>` now takes the `ncols1=8,ncols2=8` sparse arm when the RDNA band does not apply (gqa 5..8) and the predicate holds; the test `(256,256,2,{12,1},8192,64,…)` exercises it and passes. |
| **P2** `(512,512,1,8)` / `(256,256,1,8)` on AMD | **drop (invalid)** | `ncols=8 < 16` → `NO_DEVICE_CODE`; forcing it produced `HIP kernel flash_attn_ext_f16 has no device code compatible with HIP arch 1300` + an HSA exception.  Guarded off. |
| **P2** qwen4exp `n_q <= 4` decode sparse | **defer** | needs a new `(256,256,*,16)` sparse instantiation (valid on AMD) and a chooser entry; not done here. |
| **P2** AMD-specific engagement crossover | **defer** | kept the NVIDIA `2*n_gather` bound; no AMD tuning attempted. |
| **P3** native-KV gathers | **partly free** | the block-15 native loaders already thread `use_sparse` (`flash_attn_ext_f16_load_tile_native`) and the AMD path compiled them; no per-type test matrix was run. |
| **P4** graph wiring + per-arch default | **defer** | `build_attn_mha(...n_kv_max...)` is a one-arg change, but correctness (per-row finite bound vs the `ncols1` union) and width purity need the model-level A/B first. |
| **P5** tile index-gather fallback | **defer** | unchanged; not needed until P4 lands. |
| **P6/P7** cleanup + release | **n/a** | no promotion. |

## What changed (fork branch `qsa-standard-fa`)

`results/phase0/phase1-2-port.patch` (202 lines, 4 files):

* `ggml/src/ggml-cuda/common.cuh` — `ggml_cuda_ballot()` (CUDA 32-bit vs HIP 64-bit member mask).
* `ggml/src/ggml-cuda/fattn-common.cuh` — `ggml_cuda_fattn_sparse_enabled()` (`GGML_CUDA_FA_SPARSE`)
  + `ggml_cuda_fattn_sparse_trace()` (`GGML_CUDA_FA_SPARSE_TRACE`).
* `ggml/src/ggml-cuda/fattn.cu` — unguard the compaction for HIP (keep MUSA aborted); arch-aware
  `shall_use_sparse` with the AMD `ncols` guard; AMD head-256 wide-sparse dispatch.
* `ggml/src/ggml-cuda/fattn-mma-f16.cuh` — unguard the `use_sparse=true` instantiation for HIP
  (keep MUSA).

No block was amended; the delivery `patches/` are untouched.

## Evidence

### Gates (post-change, gfx1201 / ROCm 7.14, 3× R9700)

| gate | result |
|---|---|
| `test-backend-ops -o FLASH_ATTN_EXT` | **6358/6358** on ROCm0/1/2, 0 failures (same count as baseline) |
| `test-backend-ops -o INDEXER_TOPK,INDEXER_SCORE,FLASH_ATTN_QSA,HC_MIX` | **59/59** on ROCm0/1/2 |
| sparse-hint cases (`-p 'n_kv_max=…'`) | **21/21**, and the trace proves the sparse path was taken |

### Sparse engagement trace (correctness run)

```
[FA_SPARSE_CHECK] ok=1 … K->ne1=4096 need=4096 ncols1=1 ncols2=16   # (576,512,gqa16) decode
[FA_SPARSE_CHECK] ok=1 … K->ne1=8192 need=8192 ncols1=8 ncols2=8    # (256,256,gqa12) verify/prefill width
[FA_SPARSE_CHECK] ok=0 … ncols1=1 ncols2=8                          # (512/256, …,1,8): AMD ncols<16 → dense
```

The `(256,256,gqa12)` case **fails to launch (HSA exception) without the AMD guard** and passes with
it — the guard is a correctness fix, not just a policy choice.

### Performance — inconclusive, needs the model A/B

`test-backend-ops perf -p 'n_kv_max=…'` with `GGML_CUDA_FA_SPARSE=1` vs `=0` showed ratios ~1.00 for
the valid shapes (and `[FA_SPARSE_CHECK]` trace lines do **not** appear in `perf` mode — the perf
harness evidently does not reproduce the kernel-selection context, so these numbers cannot be used
to judge the sparse path).  **The fused-vs-standard decision (README open question 1) is not
resolved by this session.**  Any future P4 work must measure on the real qwen4exp model with
`LLAMA_QSA_STANDARD_FA`-style gating, `-lm none`, and the width probe.

## Path matrix (current, gfx1201)

| geometry | AMD dense | AMD sparse | default |
|---|---|---|---|
| head 576 / gqa16, `n_q<=4`, long KV | MMA | **`(576,512,1,16)`** | dense (sparse only if `n_kv_max>0` and `2*n_gather` ok) |
| head 256 / gqa12, `n_q>4` | MMA ncols2=4 | **`(256,256,8,8)`** | dense unless `n_kv_max>0` |
| head 256 / gqa12, `n_q<=4` | MMA/tile | **none (no valid instance)** | dense |
| head 512 / gqa8 | MMA/tile | **none (ncols=8 invalid on AMD)** | dense |
| qwen4exp sparse regime (`n_kv>width`) | — | `FLASH_ATTN_QSA` (fused) | fused |

## Next steps (handover to the next session)

1. **P2b:** add `(256,256,1,16)` (and, if the chooser needs it, `(256,256,4,16)`) to
   `may_use_sparse`, instantiate, and dispatch it for the gqa-12 `n_q<=4` decode band.  Verify the
   AMD WMMA `ncols=16` arm accepts it (it should — same shape class as the working 576 arm).
2. **P2c:** decide the AMD engagement crossover (the `2*n_gather` bound is NVIDIA-tuned).
3. **P4:** wire `n_kv_max` into `build_attn_qsa`'s standard-sparse arm behind
   `LLAMA_QSA_STANDARD_FA=0` (default keeps fused).  The per-query bound is `top_k->ne[0]` (the
   launcher multiplies by `ncols1` for the union); **do not** pass `width = top_k + r - 1`, which is
   the indexer selection width, not the per-row finite count.
4. Re-run the full gate catalogue, the `W=1..8` width probe and the plain-vs-`draft-mtp` text gate
   (P4 is the purity risk), then record adopt/defer/drop and consider promotion.

## Files

* `results/phase0/fa_qsa_baseline.txt` — pre-change `FLASH_ATTN_QSA` 26/26.
* `results/phase0/sparse_ext_trace.txt`, `phase1-2-port.patch`.
* `prompts`/`scripts` used are unmodified.
