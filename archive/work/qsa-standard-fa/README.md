# WIP — QSA on the standard FA kernels (upstream sparse-index unification)

**Created:** 2026-10-05, after `v16-a55e952b8-r4`.
**For:** a follow-up session (fresh context).  This README is the handover; it is written so the
session can pick the campaign up cold, drive it to completion, resolve the WIP and cut the next
release without re-deriving the context.
**Status:** **RESOLVED 2026-10-05** — archived here.  Outcome: upstream's `n_kv_max` sparse
mechanism was ported to HIP (Option A, adopted as a non-destructive capability) but the measurement
shows the fused `FLASH_ATTN_QSA` is **1.7-2.6× faster** at the qwen4exp geometry, so Option B
(route qwen4exp through the standard path) and Option C (delete the fused kernel) are **dropped**.
No delivery change is recommended.  See [RESULTS.md](RESULTS.md) for the numbers, the AMD
`ncols >= 16` constraint and the O(n_kv) compaction-prepass blocker.  The WIP code is on the fork
branch `qsa-standard-fa` (net diff in `results/phase0/phase1-2-port.patch`), not in `patches/`.
**Delivery state:** `~/llama-cpp-rdna-boosts` `main` = release **`v16-a55e952b8-r4`**;
`release.json` `release=v16-a55e952b8-r4`, `base=a55e952b8`, `tip=cd1485fd1`,
`tree=714f94f050dfce08c987a8a14467f456fe6e9d60`.  Fork `~/llama.cpp` branch `rdna-boosts` tip
`cd1485fd1`, base `a55e952b8`.

## Session preamble

```
We recently rebased out 16 patch set against the `master` tip at ~/llama.cpp/ and the result is in the `rdna-boosts` branch there
llama.cpp is built with ~/bin/build-llama-rocm-714
Almost all of this system's model files are located at /llm/models/*
```

---

## How to resolve this tree — the WIP-resolution pattern

This campaign is the follow-up to **item 5** of the archived
[`../lightning-indexer-fusion/`](../lightning-indexer-fusion/README.md) handover
(its `RESULTS.md` is the sibling resolution record).  That handover documents the pattern this
campaign must also follow; the summary is:

1. **Investigate and decide, with evidence.**  For every task below state **adopt / defer / drop**
   with numbers (before/after, byte-identity, VRAM, launch counts, kernel times).  Follow the repo
   rules: **default-on for anything beneficial, env vars only disable**; **RDNA-first, non-AMD stays
   consistent**; **width purity `W=1..8`** for anything on the decode/verify band.  Where a change
   is accepted into the delivery, give it a kill-switch and run the A/B.
2. **Fold each adopted win into the block that owns it.**  The canonical chain is the fork's
   `rdna-boosts` branch (`git log --oneline a55e952b8..rdna-boosts` — 16 block commits).  Put a
   change in the **earliest block that owns the file**; the rebase recipe is in
   `../lightning-indexer-fusion/README.md` (§2).  The fold map for this
   campaign is in [Fold targets](#fold-targets).
3. **Regenerate and validate (no release yet).**

   ```bash
   cd ~/llama-cpp-rdna-boosts
   ./scripts/make-patches.sh  ../llama.cpp a55e952b8 <new-tip>
   ./scripts/make-release.sh  --tip <new-tip> \
       --tree "$(git -C ../llama.cpp rev-parse <new-tip>^{tree})"   # release name inherited (r4)
   ./scripts/validate-set.sh                                       # strict 16/16 git am + tree match
   ```

   Plus a clean warning-free `all`-target build (`~/bin/build-llama-rocm-714`) and the gates in
   [The gate catalogue](#the-gate-catalogue).
4. **Record, archive, update the docs.**  Move `wip/qsa-standard-fa/` →
   `archive/work/qsa-standard-fa/` and add `RESULTS.md` (adopt/defer/drop table, measurements,
   delivery record); leave this `README.md` in place as the handover.  Update the relevant
   `wip/*/README.md` (mark the campaign archived), `AGENTS.md`, `README.md`, `MANIFESTS.md`, `BASELINE.md`, `WORKLOG.md` and
   `patches/README.md`.
5. **Cut the next release** (`v16-a55e952b8-r5` or whatever the maintainer names it):

   ```bash
   ./scripts/make-release.sh --tip <new-tip> \
       --tree "$(git -C ../llama.cpp rev-parse <new-tip>^{tree})" --release v16-a55e952b8-r5
   git add -A && git commit -m "release v16-a55e952b8-r5: ..."
   git push origin main
   git tag -a v16-a55e952b8-r5 -m "..." && git push origin v16-a55e952b8-r5
   ```

   The tag is the release identity: it triggers `.github/workflows/validate.yml` and the
   `docker-ghcr.yml` GHCR/patch-set pipeline.  Confirm `release.json.release` equals the tag and
   that both `Validate delivery set` runs go green.

---

## Why this WIP exists

qwen4exp's sparse attention is a **second, parallel FA implementation**: `ggml_flash_attn_qsa`
(`ggml/src/ggml-cuda/fattn-qsa.cu`, `fattn-qsa3.cu`) with its own CPU oracle, its own
width-purity contract, its own per-arch/per-KV-type gates, and its own bug history (K/V-head
chunking, quant-type gates, issue #89).  Upstream's standard `ggml_flash_attn_ext` kernels already
have a **sparse mode** (`n_kv_max` + a compacted mask index list), and upstream's own qwen4exp uses
it.  If our top-k selection fed that mechanism, the sparse path would inherit the entire RDNA FA
investment — V3 derived mask, V4/V5 native KV dequant, the RDNA4 GQA-6 decode/verify WMMA band, the
per-arch head caps, the `FLASH_ATTN_EXT` (6358/6358) coverage — instead of maintaining a second
kernel.  This campaign is about **unifying onto upstream's framework**, which also makes future
upstream merges smoother.

## Goal / definition of done

A qwen4exp (and any future QSA model) sparse attention that:

* runs through `ggml_flash_attn_ext`'s `n_kv_max` sparse mechanism on **ROCm/gfx12 + gfx11**, for
  every KV cache type the fused `FLASH_ATTN_QSA` supports (`f16/bf16/q8_0/q4_0/q4_1/q5_0/q5_1/
  iq4_nl`),
* is **width-pure** (`W = 1..8` bit-identical) on the decode/verify band, and MTP-clean,
* is at least as fast as the fused kernel on the archs that matter (or the fused kernel is kept as
  a per-arch fast path with the standard path as the correctness/coverage fallback),
* passes the full [gate catalogue](#the-gate-catalogue), and
* is shipped as block amendments + a release, with a written `RESULTS.md`.

**Non-goals:** porting the QSA graph to Vulkan/Metal/SYCL; replacing our fused top-k / derived cache
with upstream's generic kpool (`archive/work/qwen4exp-qsa-convergence/DECISION.md` decided to keep
ours); touching non-AMD sparse performance.

---

## Current state of the world (verified on `cd1485fd1`)

### How the standard sparse mechanism works

* The op param is `ggml_flash_attn_ext_set_n_kv_max(op, n_kv_max)` (`ggml/include/ggml.h:2512`):
  *"Use finite mask entries as a sparse K/V set. Set 0 to disable. n_kv_max must bound the number of
  finite entries in every mask row."*
* `launch_fattn` (`ggml/src/ggml-cuda/fattn-common.cuh`) takes a `use_sparse` template/arg; when set
  it calls `ggml_cuda_flash_attn_ext_compact_mask` (`fattn.cu:106`), which scans the **dense F16
  mask** and writes, per (stream, query-tile) group, a compacted `indices[]` list of finite columns
  plus a `counts[]`, capped at `n_kv_max` (`fattn-common.cuh:2080-2089`).
* The AMD/NVIDIA MMA kernel's `use_sparse` template (`fattn-mma-f16.cuh`, grep `indices`) takes
  `i_sup` (the count) + `indices[]` and gathers K/V rows:
  `index = i < i_sup ? indices[k_VKQ_0+i] : -1;` then loads from `K + index*stride`.
* `n_kv = use_sparse ? n_kv_max : K->ne[1]` (`fattn-common.cuh:2200`), so the whole attention
  shrinks to the selected cells.
* Upstream qwen4exp uses it: `build_attn_mha(q, k, v, nullptr, sel_mask, nullptr, nullptr, n_sel,
  kq_scale, il)` (`a55e952b8:src/models/qwen4exp.cpp:930`), with `sel_mask` the selection mask
  reshaped to the kq-mask shape.

### The catch — it is NVIDIA-MMA-only today

* `flash_attn_mask_to_sparse_indices` (`fattn.cu:8`) and `ggml_cuda_flash_attn_ext_compact_mask`
  are inside `#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)`; the HIP `#else` is
  `GGML_ABORT("sparse flash attention is only supported on NVIDIA CUDA")` (`fattn.cu:106-125`).
* `ggml_cuda_flash_attn_ext_mma_f16_shall_use_sparse` (`fattn.cu:128`) returns `false` on HIP
  immediately; on NVIDIA it also requires `turing_mma_available(cc)`, no ALiBi/softcap,
  `mask->ne[0]==K->ne[1]`, `mask->ne[2]==1`, and `K->ne[1] >= max(4096, 2*n_gather)`.
* The `use_sparse=true` MMA instantiation is inside `#if !defined(GGML_USE_HIP)` at
  `fattn-mma-f16.cuh:~2538`, so **no AMD kernel is built with the gather arm**.
* `ggml_cuda_flash_attn_ext_mma_f16_may_use_sparse` (`fattn-mma-f16.cuh:2170`) admits only four
  shapes: `DKQ/DV 512/512`, `576/512`, `256/256` (two GQA variants).
* `ggml_cuda_flash_attn_ext_compact_mask` asserts `ncols1 ∈ {1, 8}`.
* The **tile** kernel has no index list at all: its `i_sup` is a contiguous-prefix bound
  (`!oob_check || i < i_sup ? KV + i*stride`); `fattn-tile.cuh` has 1 textual occurrence of
  `indices`.  So "the normal tile/MMA/vec kernels accept a top-k cell list" is really "the NVIDIA
  MMA kernel does".
* `test-backend-ops` registers sparse FA cases unconditionally
  (`tests/test-backend-ops.cpp:12298-12318`), so on this box they pass **through the dense
  fallback** (`shall_use_sparse` false → the sparse mask is treated as a dense mask).  There is no
  HIP sparse coverage even though the tests are green.

### What our fork already provides

* `build_attn_qsa` (`src/models/qwen4exp.cpp`) already builds `kq_mask_top_k` (fill `-INF`,
  `ggml_set_rows` the top-k cells, add the base mask) — exactly the "few finite entries" mask the
  standard compaction expects — and its dense fallback already calls
  `build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr, /*n_kv_max=*/0, kq_scale, il)`.
  Setting `n_kv_max = width` there is a one-argument change **on the graph**; the kernel is the gap.
* The fused kernel `ggml_flash_attn_qsa` reads 8 KV types natively and is width-pure / oracled
  (`FLASH_ATTN_QSA` 26/26).  It is the RDNA-specialized fast path this campaign must either replace
  or keep alongside the standard path.

---

## The design space

**Option A — port the sparse mechanism to HIP, keep `FLASH_ATTN_QSA` as the fast path.**  Smallest
risk: no default flip, the new path is validated against the existing oracle, and the fused kernel
keeps serving production.  Deliverable: HIP sparse support + `FLASH_ATTN_EXT` sparse cases that
actually exercise it on AMD.

**Option B — route qwen4exp through the standard sparse path by default when it wins.**  Requires A
plus width purity, native-KV gathers, and a per-arch perf decision.  This is the real unification.

**Option C — delete `FLASH_ATTN_QSA`.**  Only after B is width-pure, native-KV-complete and at least
as fast on gfx1201/gfx1150/gfx1100.  Do not attempt before B is proven.

**Recommended path:** A → B → (C only if the numbers justify it).  A is non-destructive and is
independently valuable.

---

## Work plan

Each phase lists tasks and the **gate subset** that must be green before moving on.  Full commands
and expected values are in [The gate catalogue](#the-gate-catalogue).

### Phase 0 — baseline and harness (do this first)

Tasks:
1. Rebuild the delivery tree from the fork tip and record the Phase-0 baseline (all hashes/perf
   numbers) so every later A/B has a same-box reference.  Store the raw logs/CSVs under
   `results/phase0/`.
2. Read, in order: `AGENTS.md` (rules + default-on policy), `GREEDY-PURITY.md` §§1-4, 11, 19-21, 27,
   29, 36, this file, `archive/work/lightning-indexer-fusion/RESULTS.md`,
   `archive/work/qwen4exp-qsa-convergence/DECISION.md`, `patches/README.md` block-15 sections.
3. Confirm the exact `FLASH_ATTN_QSA` oracle set and expected counts (`test-backend-ops
   -o FLASH_ATTN_QSA` → 26/26) and the sparse `FLASH_ATTN_EXT` cases that currently run dense.
4. Build a **sparse-vs-dense cross-check harness**: the `FLASH_ATTN_QSA` oracle is the reference for
   the new sparse path; add/extend a `test_flash_attn_ext` case with `n_kv_max > 0` that fails if
   the HIP sparse path silently falls back to dense (e.g. assert the launcher selects it, or a
   debug counter/trace) so "green" actually means "sparse exercised".

Gate subset: Phase-0 baseline recorded; the harness demonstrates the fallback-vs-sparse distinction.

### Phase 1 — compact the mask on HIP (non-destructive)

Tasks:
1. Unguard `flash_attn_mask_to_sparse_indices` and `ggml_cuda_flash_attn_ext_compact_mask` for HIP
   (`fattn.cu`).  Validate the warp-ballot offset logic, `ggml_cuda_pdl_sync/lc`, `__ballot_sync`,
   `__popc` on gfx11/gfx12.
2. Relax/extend the `ncols1 ∈ {1,8}` assertion to the `ncols1` qwen4exp actually uses (our QSA
   folds the GQA group; the standard chooser's `ncols1` is chosen by `fattn.cu`'s
   `switch_ncols1`, so decide whether to add sparse cases for the existing `ncols1` values or add a
   new `ncols1`).
3. Add a `test-backend-ops` sparse case (or a debug counter) that proves compaction runs on HIP.

Gate subset: compaction correctness vs the dense mask on random masks (bit-identical selected set);
`FLASH_ATTN_EXT` still 6358/6358 (the new path must not change the existing cases' fallback).

### Phase 2 — instantiate the sparse MMA arm on AMD

Tasks:
1. Instantiate `flash_attn_ext_f16<..., use_sparse=true>` for `AMD_WMMA_AVAILABLE` /
   `AMD_MFMA_AVAILABLE` shapes (gfx12 WMMA `256/256` etc.).  The `use_sparse` template already
   threads through `flash_attn_ext_f16_load_tile_native<..., use_sparse>`, but it has never been
   compiled on AMD; `cp_async` is `static_assert`ed off for sparse (`fattn-mma-f16.cuh:790`);
   re-check shared-memory sizing / `nstages`.
2. Make `shall_use_sparse` arch-aware (or add an AMD predicate) with a **kill-switch**
   (`GGML_CUDA_FA_SPARSE=0/1`) so the path is A/B-able and bisectable.
3. Handle the `K->ne[1] >= max(4096, 2*n_gather)` engagement bound and decide the AMD crossover.
4. Validate against `FLASH_ATTN_QSA` for **f16/bf16** first (the AMD sparse loaders most likely
   path).

Gate subset: `FLASH_ATTN_EXT` sparse cases exercise the AMD path and pass; oracle agreement vs
`FLASH_ATTN_QSA` on the same mask/top-k; `W=1..8` width purity on qwen4exp f16/bf16.

### Phase 3 — native KV gathers

Tasks:
1. Compose the index gather with the block-15 native-KV loaders for `q8_0`, `q4_0`, `q4_1`, `q5_0`,
   `q5_1`, `iq4_nl` (the fused kernel supports all 8; the NVIDIA sparse path is f16/bf16 only).
2. Extend `test_flash_attn_ext` sparse cases per KV type (mirror the `FLASH_ATTN_QSA` type loop at
   `tests/test-backend-ops.cpp:11924`).
3. Watch the issue-#47 typed-store and issue-#89 block-index traps (see
   `GREEDY-PURITY.md` and `patches/README.md` block-15 sections).

Gate subset: per-type `FLASH_ATTN_EXT` sparse green; `scripts/gate-qwen4exp-quant-coherence.sh`
(0 `////`, gather ON == OFF); per-type width purity.

### Phase 4 — graph wiring + per-arch default

Tasks:
1. In `build_attn_qsa` (`src/models/qwen4exp.cpp`), pass `n_kv_max = width` to `build_attn_mha` on
   the standard-sparse arm, gated by the standard path's engagement predicate; keep
   `ggml_flash_attn_qsa` behind a kill-switch (`LLAMA_QSA_STANDARD_FA=0` to force the fused kernel).
2. Decide the per-arch default with measurements (below): fused vs standard-sparse, for gfx1201 /
   gfx1150 / gfx1100.
3. Ensure the derived-visibility / dense-fallback interactions still work (the mask may be null on
   the derived path; the standard sparse path needs a mask, so it must fall back to dense/fused
   there).
4. Preserve the existing `-sm tensor` and QSA-vs-KV-type gates (they must compose).

Gate subset: full [correctness](#correctness--validation) + [quality](#quality--purity) sets,
quantity and text byte-identity, MTP.  **This is the phase that can break purity** — do not skip.

### Phase 5 — tile path / coverage fallback

Tasks:
1. Decide whether the tile kernel needs an index-gather path (gfx1151 head-512,
   `GGML_CUDA_FA_WMMA_256=0`, and other non-MMA shapes), or whether those keep `FLASH_ATTN_QSA`.
   Adding the gather to `fattn-tile.cuh` is a new loader (the tile `i_sup` is a prefix bound only).
2. If not added, document the per-arch/per-shape matrix (which path serves which shape) and keep the
   fused kernel for the uncovered cells.
3. Add `test-backend-ops` coverage that asserts the intended path per shape on each arch.

Gate subset: `FLASH_ATTN_EXT` + `FLASH_ATTN_QSA` green on the arch(s) available; the shape→path
matrix is written down.

### Phase 6 — cleanup and documentation

Tasks:
1. Remove dead code only if Option C cleared (fused kernel no longer default anywhere).
2. Strip/keep the kill-switches per the default-on policy (a beneficial default is ON; the env var
   only disables).
3. Write `RESULTS.md` with the adopt/defer/drop table, all raw numbers, the path matrix and the
   delivery record.
4. Update the docs listed in step 4 of the pattern.

### Phase 7 — release

Follow step 5 of the pattern.

---

## The gate catalogue

Run from `~/llama.cpp` (build `~/bin/build-llama-rocm-714`, GPU `HIP_VISIBLE_DEVICES=0,1,2`).
`llama-cli` **must** use `--single-turn`; wrap blocking commands in `timeout`.  Hash generation
text with `~/llama-cpp-rdna-boosts/scripts/extract-generated.py` — never `sed`/`grep` the log (the
CLI emits backspace corrections).

### Correctness / validation

| gate | command | expected |
|---|---|---|
| oracle sweep | `./build-rocm/bin/test-backend-ops -o INDEXER_TOPK,INDEXER_SCORE,FLASH_ATTN_QSA,HC_MIX` | **59/59** on ROCm0/1/2 |
| full FA | `./build-rocm/bin/test-backend-ops -o FLASH_ATTN_EXT` | **6358/6358** (must grow with the new sparse cases, not shrink) |
| indexer | `test-backend-ops -o LIGHTNING_INDEXER` / `-o TOPK_QSA` / `-o INDEXER_TOPK` | **225/225**, **4/4**, **3/3** |
| matmul | `test-backend-ops -o MUL_MAT_ID` / `-o MUL_MAT,MUL_MAT_ID` | **931/931** / **2235/2235** |
| other ops | `test-backend-ops -o HC_MIX,GATED_DELTA_NET,RMS_NORM,ARGSORT` | **30/30**, **46/46**, **51/51**, **78/78** |
| recurrent | `tests/test-recurrent-state-depth`, `tests/test-recurrent-state-rollback` | green (counts are config-dependent, pre-existing) |
| sampler | `tests/test-speculative-adaptive` | all OK |
| delivery | `~/llama-cpp-rdna-boosts/scripts/validate-set.sh` | strict 16/16, applied tree == `release.json.tree` |

### Quality / purity

| gate | command | expected |
|---|---|---|
| dense coherence | 4B `Qwen3.5/4B/Q8_0`, 3-GPU `-sm tensor`, `-p "The capital of France is" -n 20 --seed 42 --temp 0` | `1c5d32ac537d` |
| qwen4exp coherence | Flash-Next IQ4_NL, 3-GPU `-sm tensor -lm none --reasoning off -n 20` (and `-lm auto` identical) | `359ff4337837`; text `The capital of France is **Paris**.` |
| 27B coherence | `Qwen3.8/27B/Q8_0`, `-n 20 -lm none -lzm on` | `da2e2d192e21` |
| wider hashes | Flash-Next `-n 24` (`d73f9238f6d6`); 35B UD-Q3_K_M `-n 24` (`461ca8cd0e88`) | smoke |
| width purity | `./build-rocm/bin/test-logits-width-probe <model.gguf> <prompt.txt> 256 512`, sweep `W=1..8` | one hash per config per KV type (the instrument for the decode/verify band; `GREEDY-PURITY.md` §11) |
| plain vs spec | `--spec-type none` vs `draft-mtp --spec-draft-n-max 3` (and `n7` for qwen4exp) same seed | byte-identical text when in the pure range |
| quant coherence | `scripts/gate-qwen4exp-quant-coherence.sh` | 0 `////`, gather ON == OFF |
| perplexity | `llama-perplexity` fused/sparse vs the dense masked oracle `LLAMA_QSA_SPARSE_FA=0` | ratio within noise (the only quality metric that sees a QSA defect, `GREEDY-PURITY.md` §21) |
| MTP | `benchmarks/mtp-adaptive-methodology.md` Protocol A + the four axes with `prompts/` | acceptance >= ~0.45 at pos 1; MTP >= plain at default depth |
| causal leak | random-text continuation probe (`GREEDY-PURITY.md` §23) | no tokens from the masked-out suffix |

### Performance

| gate | command | expected |
|---|---|---|
| decode | `llama-bench -p 0 -n 128 -r 4 -lm none` | qwen4exp ~56-57 t/s (r4 baseline 57.1) |
| prefill | `llama-bench -p 512,4096,8192 -n 0 -r 4 -lm none` | r4 baseline pp512 ~1450-1540 |
| deep prefill | `llama-bench -p 150000 -n 0 -r 3` (arch permitting) | not regressed vs `FLASH_ATTN_QSA` |
| verify width | `llama-batched-bench -npl 1,4,8` (dense K-quant + quantized KV) | within noise at B=1, **no worse** at B=4/B=8 (`mtp-adaptive-methodology.md` rule 5) |
| MTP throughput | the four-axis run | no regression on any axis; phase-switching probe `prompts/code-reasoning-mixed.txt` |

**Rules:** interleave A/B runs on the same box/state; `-lm none` for qwen4exp benchmarks (the PLE
weights are host-resident); size `-t`/`--cpu-mask` so the GPU-IRQ cores (the top `NUM_GPUS`, i.e.
13/14/15 on this 3-GPU host) stay free (`AGENTS.md` "Thread sizing"); never run parallel benches.

---

## Environment / host notes

* gfx1201 (3× R9700), ROCm 7.14 at `/opt/rocm-7.14.1-gfx120X`; build via
  `~/bin/build-llama-rocm-714` (`BUILD_DIR=build-rocm`, `JOBS=16`, ccache on by default).
* Other validated targets: gfx1150 (Strix Halo) and gfx1100 (RX 7900 XTX) — use them if available;
  they are where the fused-vs-standard decision differs most.
* Models under `/llm/models/`: `Qwen3.5/4B/Q8_0` (dense gate), `Qwen3.8/27B/Q8_0`,
  `Qwen3.6/35B-A3B/Q8_0` (MoE), `Qwen3.8/Flash-Next/IQ4_NL` (qwen4exp, needs all 3 GPUs and
  `-lm none` for benchmarks; its `mtp-*.gguf` sidecar is the MTP head), `Gemma4/31B`
  (+`31B-QAT`) for the SWA/head-512 tile path.
* The qwen4exp `n_heads` is 16 (`ngram 3 × heads_per_ngram 8`); `idx_dim` 128; `n_idx_h` 4.
* Tool binaries live in `build-rocm/bin` (`llama-cli`, `llama-bench`, `llama-batched-bench`,
  `test-backend-ops`, `test-logits-width-probe`, `llama-perplexity`).

---

## Fold targets

| this campaign touches | owning block |
|---|---|
| `src/models/qwen4exp.cpp` (`build_attn_qsa` selection, `n_kv_max`, fallback chain) | **block 14** |
| `ggml/src/ggml-cuda/fattn.cu`, `fattn-common.cuh`, `fattn-mma-f16.cuh`, `fattn-tile.cuh`, `fattn-qsa*.cu[h]`, `ggml-cuda.cu` | **block 15** |
| `tests/test-backend-ops.cpp` | the block that owns the op under test (block 08/14/15) |
| `ggml/src/ggml-cuda/lightning-indexer.cu` / `indexer-*.cu` | **block 15** |
| `ggml/src/ggml-hip/CMakeLists.txt` (new instance files) | **block 15** |

The rebase recipe (amend the owning block, then `git rebase --onto <new-blk> <old-blk>
rdna-boosts`) is in `archive/work/lightning-indexer-fusion/README.md` §2.  After a rebase,
`git diff <old-tip> <new-tip> --stat` must show only the intended files.

---

## Risks and pitfalls

* **The sparse path does not exist on HIP.**  Do not assume "the kernels accept a list" — Phase 1-2
  are real ports, and every gate must be demonstrated to *exercise* the new path (the existing
  "green" sparse cases run dense on this box).  A silent dense fallback is the most likely way to
  ship a no-op.
* **Width purity is the hard part.**  Routing qwen4exp decode through `ggml_flash_attn_ext` re-opens
  the `n_q`/`ncols`/kernel-family boundaries that blocks 00/08/15 closed (`GREEDY-PURITY.md` §§11,
  14, 19, 36).  The `W=1..8` probe and the plain-vs-`draft-mtp` text gate are mandatory, not
  optional.
* **`n_kv_max` is a correctness contract.**  The compaction caps at `n_kv_max` and truncates the
  rest; if it under-counts the finite mask entries, attention silently drops cells.  Bound it by the
  top-k width, and add a debug assert / count check while bringing it up.
* **Derived-visibility path has no mask.**  When `qsa_derive_vis` is on, `kq_mask` is null; the
  standard sparse path needs a mask, so it must fall back (dense/fused) there.  Do not make the
  standard path unconditional.
* **Native-KV gather compose.**  The NVIDIA sparse loader is f16/bf16; the block-15 native loaders
  and issue-#47 typed stores and issue-#89 block indexing are easy to break (see their records).
* **`test-backend-ops` green ≠ sparse exercised.**  Add an assertion/trace that the sparse path was
  selected before trusting a pass.
* **Scope policy.**  Non-AMD must stay consistent (do not leave an uninstantiated/mismatched
  predicate); non-AMD performance is out of scope.
* **Default-on policy.**  A win goes in ON with a kill-switch, in the same change; a correctness-risk
  knob stays default-off.  State which kind each new env var is.
* **`-lm none` for qwen4exp benchmarks.**  Without it the PLE rows are re-read from disk per prefill
  and pp512 reads ~668 ± 265 on the first rep; the r4 record is ~1450-1540 warm.

---

## Straggling follow-up items (documented here so they are not lost)

These came out of the r1-r4 re-base work and are adjacent to this campaign.  Cover them in
`RESULTS.md` even if they are deferred:

1. **WIP-resolution pattern reference.**  The six r1 follow-ups are closed; keep this campaign's
   `RESULTS.md` consistent with `archive/work/*/RESULTS.md` / `RESOLUTION.md` so the next re-base's
   handover can cite it.
2. **`llama_prefetch_rows` regression.**  The generic PLE prefetch helper (upstream `185103dcf`) was
   measured ~15-20 % slower than the fork's per-row `madvise` loop and dropped in r4.  If upstream
   fixes the helper or the access pattern changes, re-measure; the record is
   `archive/work/lightning-indexer-fusion/RESULTS.md` item 1.
3. **`LLM_FUSED_OP_LIGHTNING_INDEXER` probe is dead.**  `cparams.auto_flid = false` unconditionally
   (`src/llama-context.cpp:343`), so the registration added in r4 is inert.  If enabling it for
   qwen4exp is ever wanted, it must be done without changing `fused_lid` resolution for
   DeepSeek/DSA models (they read it for the `"lid"` mask type) — enable per-arch/per-context, and
   re-run `FLASH_ATTN_EXT`.
4. **`qwen4exp-qsa-convergence` open trigger.**  The decision to keep the fused QSA graph
   (`archive/work/qwen4exp-qsa-convergence/DECISION.md`) has explicit triggers to revisit: upstream
   qwen4exp perf parity, a correctness/memory bug that upstream's graph fixes, or a re-base where
   the two graphs stop composing.  This campaign is the *kernel-level* counterpart; if it lands, the
   graph-level convergence may also become attractive.
5. **`test-backend-ops` sparse cases on HIP are false-green.**  Until Phase 1-2 land, the sparse
   `FLASH_ATTN_EXT` cases pass via the dense fallback.  Consider gating/skipping them on HIP with a
   clear message (or asserting the sparse selection) so a future reader is not misled.
6. **qwen4exp `--spec-draft-n-max` purity range.**  The guarantee is `<= 7` (the FA `n_q > 8` and
   matmul `ncols == 8` switches); the CLI warns for 8..15.  Any change to the FA chooser here must
   re-run the width probe and the MTP gate for the configured depths.
7. **Fused-kernel open items** (carry forward from the archived handovers): the `FLASH_ATTN_QSA`
   head-chunking (`min(QSA_MAX_HEADS, gqa_ratio)`), the `iq4_nl`/quant-type gates, issue #89's
   block-index fix, and `GGML_CUDA_QSA3` default policy.  Each is a reason the fused kernel is still
   needed on some cell of the shape matrix.
8. **Non-AMD consistency.**  The sparse mechanism is NVIDIA-only; the port must keep the CUDA/MUSA
   paths working (guards, instantiations, `test-backend-ops`).  Document any intentionally
   AMD-only half, per the scope policy.

---

## Pointer index

* Upstream sparse machinery: `ggml/src/ggml-cuda/fattn.cu` (`compact_mask`, `shall_use_sparse`,
  `may_use_sparse`, `switch_ncols1`), `fattn-common.cuh` (`launch_fattn`, `use_sparse`, `n_kv`),
  `fattn-mma-f16.cuh` (`use_sparse` template, `indices`/`i_sup`), `fattn-tile.cuh` (no index list).
* Op param: `ggml/include/ggml.h:2512`; graph caller: `src/llama-graph.cpp:2712-2771`
  (`build_attn_mha`).
* Our fused implementation: `ggml/src/ggml-cuda/fattn-qsa.cu`, `fattn-qsa3.cu`,
  `ggml/src/ggml.c:5646` (`ggml_flash_attn_qsa`); graph side `src/models/qwen4exp.cpp`
  (`build_qsa_top_k`, `build_attn_qsa`, `build_qsa_store_k`).
* Tests: `tests/test-backend-ops.cpp:11924` (`FLASH_ATTN_QSA` 26), `:12298` (sparse `FA_EXT`),
  `:12447` (indexer top-k), `:11649` (sparse mask).  Width probe:
  `tests/test-logits-width-probe.cpp`.
* Reference records: `archive/work/lightning-indexer-fusion/README.md` + `RESULTS.md`;
  `archive/work/qwen4exp-qsa-convergence/DECISION.md`; `patches/README.md` block-15 sections;
  `GREEDY-PURITY.md`; `benchmarks/mtp-adaptive-methodology.md`; `prompts/README.md`.

## Open questions (decide and record in `RESULTS.md`)

* Is the standard sparse MMA path ever faster than `FLASH_ATTN_QSA` on gfx1201 at the qwen4exp GQA
  geometry?  (Measure before committing to Option B/C.)
* Which shapes must keep the fused kernel (tile path, head-512, `GGML_CUDA_FA_WMMA_256=0`)?
* Does the compaction kernel's per-query-tile index list match the fused kernel's per-token list
  semantics closely enough to be bit-transparent, or is a re-baseline unavoidable?  (A re-baseline
  must be recorded against the dense masked oracle, not only against the fused hash.)
* Can the sparse selection engage at shallow context, or does the `K->ne[1] >= 4096` bound force a
  dual-path policy?
