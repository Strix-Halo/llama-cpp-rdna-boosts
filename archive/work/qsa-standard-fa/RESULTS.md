# RESULTS — QSA on the standard FA kernels (Phase 0-2)

**Session:** 2026-10-05 (follows the `wip/qsa-standard-fa/README.md` handover, after
`v16-a55e952b8-r4`).
**Status:** **Phase 0 complete; Phase 1 adopted; Phase 2 adopted for the AMD-legal shapes;
Option B/C dropped on measurement; Phases 3-7 closed as not worth pursuing.**  Nothing is promoted
to `patches/` — this tree is investigation only (WIP rule).  The code lives on the fork branch
**`qsa-standard-fa`** (`98af9d949` + `f1211f784`) off `cd1485fd1`; the net diff is
`results/phase0/phase1-2-port.patch` (184 lines, 4 files).

## TL;DR — the unification does not pay off

Upstream's `n_kv_max` sparse mechanism (dense-mask compaction + MMA index gather) is now **ported to
HIP and validated on gfx1201**, for the shape the AMD WMMA/MFMA loader can actually launch.  It is
off the delivery path (only `FLASH_ATTN_EXT` cases with `n_kv_max > 0` reach it), so the change is
**inert in the delivery**.

A direct op microbenchmark at qwen4exp's geometry (head 256, gqa 12, top-k = `n_kv_max` = 2048,
`n_q = 1`, gfx1201) settles the campaign's central question:

| context | fused `FLASH_ATTN_QSA` | standard sparse FA | dense FA |
|---|---|---|---|
| 4096 (f16) | — | 66.78 µs | **28.78 µs** |
| 16384 (f16) | **43.58 µs** | 75.71 µs | 100.75 µs |
| 32768 (f16) | — | 92.70 µs | 166.19 µs |
| 65536 (f16) | **43.22 µs** | 111.77 µs | 346.66 µs |
| 16384 (q8_0 KV) | **36.45 µs** | 85.09 µs | 110.98 µs |
| 16384 (`n_q = 4`) | **80.07 µs** | 99.70 µs | 272.10 µs |

**The fused kernel is ~1.7–2.6× faster and flat in context.**  The standard path grows with context
because `ggml_cuda_flash_attn_ext_compact_mask` scans the *entire* dense mask (O(n_kv)) on every
call, so its prepass alone costs more than the fused kernel's whole attention at long context.  At
shallow context it is slower than plain dense FA.  Therefore:

* **Option C (delete `FLASH_ATTN_QSA`) — dropped.**  The fused kernel is the faster path by a wide
  margin and stays the qwen4exp default.
* **Option B (route qwen4exp through the standard sparse path) — dropped.**  It would be a ~1.7×
  decode regression and would re-open the width-purity band for no gain.
* **Option A (HIP sparse support) — adopted** as a non-destructive capability (coverage/correctness,
  and it makes the false-green sparse `test-backend-ops` cases actually exercise the path on AMD).

### The two hard constraints found

1. **AMD needs `ncols1*ncols2 >= 16`.**  `flash_attn_ext_f16`'s `AMD_WMMA_AVAILABLE` guard rejects
   smaller tiles, so upstream's `(512,512,1,8)` and `(256,256,1,8)` sparse shapes are **invalid on
   AMD** (forcing one produced `HIP kernel … has no device code` + an HSA exception).  Only
   `(576,512,1,16)` and `(256,256,8,8)` are launchable.  qwen4exp's decode width has no NVIDIA-shaped
   AMD instantiation at all.
2. **The compaction prepass is O(n_kv).**  This is the reason the standard path cannot win at long
   context regardless of the attention kernel.

## Adopt / defer / drop

| task | verdict | evidence |
|---|---|---|
| **P0** baseline + harness | **done** | `FLASH_ATTN_QSA` 26/26, `FLASH_ATTN_EXT` 6358/6358 ×3, oracle sweep 59/59. |
| **P1** compact the mask on HIP | **adopt** (non-destructive) | compaction unguarded for HIP; `__ballot_sync` wrapped as `ggml_cuda_ballot` for AMD's 64-bit member mask. `FLASH_ATTN_EXT` 6358/6358 after. |
| **P1** prove the path runs | **adopt** | `GGML_CUDA_FA_SPARSE_TRACE=1` prints the decision (`[FA_SPARSE_CHECK]`); the existing sparse cases are otherwise false-green. |
| **P2** AMD predicate + kill-switch + instantiation | **adopt (AMD-legal shapes)** | arch-aware `shall_use_sparse`, `GGML_CUDA_FA_SPARSE=0` opt-out, `(576,512,1,16)` sparse verified; AMD `ncols` guard rejects the invalid shapes. |
| **P2** qwen4exp head-256 decode arm (`ncols2=16`) | **drop** | measured 1.7–2.6× slower than the fused kernel; costs a heavy WMMA instance TU and a chooser change. Reverted. |
| **P2** AMD engagement crossover | **defer/moot** | the standard path loses at every measured context; no tuning is worthwhile. |
| **P3** native-KV gathers | **free (compiled)** | the block-15 native loaders already thread `use_sparse`; no per-type matrix run. |
| **P4** graph wiring + per-arch default | **drop** | would route qwen4exp to a slower kernel and re-open width purity. |
| **P5** tile index-gather fallback | **drop** | unnecessary once P4 is dropped. |
| **P6/P7** cleanup + release | **n/a** | no promotion. |

## What changed (fork branch `qsa-standard-fa`)

`results/phase0/phase1-2-port.patch` (4 files):

* `ggml/src/ggml-cuda/common.cuh` — `ggml_cuda_ballot()` (CUDA 32-bit vs HIP 64-bit member mask).
* `ggml/src/ggml-cuda/fattn-common.cuh` — `ggml_cuda_fattn_sparse_enabled()` (`GGML_CUDA_FA_SPARSE`)
  + `ggml_cuda_fattn_sparse_trace()` (`GGML_CUDA_FA_SPARSE_TRACE`).
* `ggml/src/ggml-cuda/fattn.cu` — unguard the compaction for HIP (MUSA still aborts); arch-aware
  `shall_use_sparse` with the AMD `ncols >= 16` guard and the trace.
* `ggml/src/ggml-cuda/fattn-mma-f16.cuh` — unguard the `use_sparse=true` instantiation for HIP.

No block was amended; the delivery `patches/` are untouched.

## Evidence

### Gates (final minimal port, gfx1201 / ROCm 7.14, 3× R9700)

| gate | result |
|---|---|
| `test-backend-ops -o FLASH_ATTN_EXT` | **6358/6358** on ROCm0/1/2, 0 failures |
| `test-backend-ops -o INDEXER_TOPK,INDEXER_SCORE,FLASH_ATTN_QSA,HC_MIX` | **59/59** |
| sparse-hint subset (`-p 'n_kv_max=…'`) | **21/21**, trace proves `(576,512,gqa16)` sparse ran on AMD |

### Sparse engagement trace

```
[FA_SPARSE_CHECK] ok=1 … dkq=576 dv=512 gqa=16 K->ne1=4096 need=4096 ncols1=1 ncols2=16
[FA_SPARSE_CHECK] ok=0 … dkq=256 dv=256 gqa=12 … ncols1=1 ncols2=8   # ncols=8 < 16 -> dense
```

### Measurement method

Standard FA perf cases (`test_flash_attn_ext` with `n_kv_max`) and fused QSA perf cases
(`test_flash_attn_qsa`, same cache and selection width) were added to `make_test_cases_perf` on the
WIP branch only, then `test-backend-ops perf -b ROCm0`.  The measurement cases were reverted from the
final branch; the raw numbers are the table above.

## Recommendation

Keep `patches/` as-is.  Do **not** fold this campaign's changes into the delivery: Option A is inert
and Option B/C is a measured regression.  The only reason to promote the HIP sparse port would be to
give the AMD backend a correct (if slow) implementation of the standard sparse op for upstream
compatibility and test coverage; that is a low-priority, non-AMD-parity cleanup, not a qwen4exp win.

If upstream ever adds a sparse fast path to the **tile** kernel or makes `compact_mask` incremental,
re-measure — the O(n_kv) prepass is the specific blocker to revisit.

## Files

* `results/phase0/fa_qsa_baseline.txt` — pre-change `FLASH_ATTN_QSA` 26/26.
* `results/phase0/sparse_ext_trace.txt`, `phase1-2-port.patch` (net port).
