# Promotion record — `wip/moe-expert-cache` into the 16 delivery blocks (BETA)

**Branch:** `promote-moe-caching` (this repo, pushed to `origin`).
**Release label:** `v16-84e76d8a2-r28-moe-cache-beta1` (**not a release** — no tag, no GHCR
image, no merge to `main`).
**Fold worktree:** `~/llama-fold`, branch `fold-moe`, built from `84e76d8a2`.
**Campaign source:** `~/llama-decode`, branch `wip-moe-devmap-r28`, tip `f5a79e6ab`
(base `60361cb9f` = delivery r28), net diff `git diff 60361cb9f..wip-moe-devmap-r28` = **12 files,
+5436/-48**.

| | |
|---|---|
| base | `84e76d8a2` (tree `5112eedbce0548ab9547d883e8aa54e993852e94`) |
| fold tip | `ce06f7add75ba02e11281f2e567ddd14b3f08c81` |
| net tree | `19221824972d040e4fc83dd245b4966919e1fa99` |
| campaign tree | `19221824972d040e4fc83dd245b4966919e1fa99` (byte-identical) |

## Mapping — campaign file → delivery block

| campaign file | block | why |
|---|---|---|
| `ggml/include/ggml-backend.h` | **06** | generic scheduler API (`ggml_backend_sched_set_moe_cpu_split`) |
| `ggml/src/ggml-backend-impl.h` | **06** | generic backend iface (`moe_cache_update` / `_take_over` / `_promote` / `_gather`) |
| `ggml/src/ggml-cpu/ggml-cpu.c` | **06** | opt-in CPU MoE routing profiler (`GGML_MOE_PROFILE`), independent |
| `src/llama-arch.cpp` | **14** | `llm_arch_supports_sm_tensor()` rejects `LLM_ARCH_GEMMA4` (segmented expert upload unfinished) |
| `ggml/src/ggml-cuda/moe-expert-cache.cu` | **15** | the cache engine; authored against the r28 tree |
| `ggml/src/ggml-cuda/moe-expert-cache.h` | **15** | engine header |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | **15** | CUDA consumer hooks + cache-band fusion stand-down; uses block-15 HC matcher / weighted-down context |
| `ggml/src/ggml-cuda/mmvq.cu` | **15** | in-kernel slot lookup in `mul_mat_vec_q_moe` |
| `ggml/src/ggml-backend.cpp` | **15** | scheduler takeover / promote / gather hooks; uses block-15 `GGML_ENV_STR` |
| `ggml/src/ggml-backend-meta.cpp` | **15** | Meta delegation; uses block-15 r16 `stage_gather` chunk geometry |
| `ggml/src/ggml-cuda/common.cuh` | **15** | `GGML_META_SCRATCH_MB` / `h2d_scratch_bound` (block-15 `h2d_scratch`) |
| `src/llama-graph.cpp` | **15** | opt-in CPU-computes-the-misses branch (calls the block-06 scheduler API) |

Blocks **00-05 and 07-13** change only in their `From <sha>` / `index` lines (bodies unchanged).

**Why block 15 carries the bulk.**  The campaign was authored on top of the full r28 tree, so
`ggml-cuda.cu`, `ggml-backend.cpp` and `ggml-backend-meta.cpp` are interleaved with block-14/15
features (`GGML_ENV_STR`, the r16 `stage_gather` chunk fields, `h2d_scratch`, the `hc_mix` fusion
matcher).  Folding them into block 06/13 would need explicit forward references to later-block code,
which the `archive/work/beta-integration` fold rule forbids (it prefers relocation).  Block 15 is the
repo's established home for subsystem/campaign folds — the r16 prefill sibling is there too — and the
campaign's own note recorded block 06 as the long-term home for the scheduler half.

## Method (reproducible)

Fresh linear rebuild from `84e76d8a2`.  For each block `i` in order:

1. `git cherry-pick <block_i>` (the r28 chain commit).
2. `git apply --3way --index /tmp/camp/<file>.patch` for every campaign file assigned to block `i`
   (per-file patches cut from `git diff 60361cb9f..wip-moe-devmap-r28 -- <file>`; `--3way` merges
   against the r28 blob recorded in the patch's `index` line).
3. `git add -A && git commit --amend` (block 06/14/15 keep the original subject and append the
   campaign-fold provenance to the body).

The fold tip tree is asserted equal to the campaign tree (`1922182…`).  The 16 amended block
commits on `fold-moe`:

```
05df27c0ab0c3d38d97560fa6a7a3e4fdd570968  block 00
3dec368817d4750bb98a954f115f269c5204fa29  block 01
43b37ff76a390a11cc90c4afac8b0a161667b9b6  block 02
36175c90240b9239ca546bff2786600b4bc7bfb0  block 03
7d0a8e5a99f3480de315586548eb4e59d0c0f2d1  block 04
2afc8141a689cdaee7649a2caca6eeb2cbb88953  block 05
5ac6c00843cd88fa81ab4e2498a3804733b6a2a5  block 06  (campaign: iface + CPU profiler)
09d6fde719d863daf482027611f8ba5d5c66c0a3  block 07
806dfab49cba087c62033e1a5c6aba9b8ee12c4f  block 08
17b7243140252cecaf6f2d7e493ccc908cfd582d  block 09
f96c2773845a71abb9581837b641ff3171679d12  block 10
1b5e2520c20c6ff4295cff73a2c3b0c692a78cba  block 11
dae010c39bf9e6ea78d4f0c009fb04e7b0d99d36  block 12
b30ba56fc2061003070dd60cf39f498154b42ec3  block 13
394d994d92655392276fabe9f3bbb6386aed0e76  block 14  (campaign: gemma4 guard)
ce06f7add75ba02e11281f2e567ddd14b3f08c81  block 15  (campaign: cache engine + consumers)
```

Regenerate with `scripts/make-patches.sh ~/llama-fold 84e76d8a2 ce06f7add…`, then
`scripts/make-release.sh --base 84e76d8a2 --base-tree 5112eedb… --tip ce06f7add… --tree 1922182…`.

## Verification (this fold)

* `scripts/validate-set.sh` **green** — artifact checksums, strict 16/16 `git am` on a fresh
  `84e76d8a2` codeload tarball, applied tree == `release.json.tree`.
* Clean `gfx1201` / ROCm 7.14 build (`~/bin/build-llama-rocm-714`) **exit 0**.
* `test-backend-ops -o MUL_MAT_ID` **929/929**.
* The tree is byte-identical to the campaign tree, so the campaign's own gates (byte-identity to the
  `-ncmoe 0` oracle, width purity `none == n1 == n3 == n7`, MTP acceptance, deep coherence, and the
  gfx1201 / gfx1151 / gfx1100 records in `README.md` / `WORKLOG.md`) carry over unchanged.

## Known beta follow-ups

* **Three new compiler warnings** vs r28 (deliberately not fixed in the fold, to keep the tree
  byte-identical to the validated campaign tree):
  * `ggml/src/ggml-cpu/ggml-cpu.cpp` and `ggml/src/ggml-rpc/ggml-rpc.cpp`:
    `-Wmissing-field-initializers` for `moe_cache_update` (the new iface fields are not named in
    those vtables).
  * `ggml/src/ggml-cpu/ggml-cpu.c`: tautological `dst->name != NULL` in `ggml_moe_prof_record`.
  Fix them with the next amendment and re-run the gates.
* **Env-gated debug/A-B knobs** ride along (`GGML_MOE_PROFILE`, `GGML_SCHED_SYNCDBG`,
  `GGML_SCHED_INPUTDBG`, `GGML_META_NOSYNC`, `MOE_EXPERT_CACHE_FORCE_DEVMAP`, …).  The r16 promotion
  left the same follow-up; strip them before an `upstream/` PR candidate is cut.
* **gemma4 `-sm tensor`** stays rejected until the segmented host-resident-expert async upload is
  finished (parked; `TODO.md`).
* **A future re-cut** could move the scheduler/interface half to block 06 and the MoE kernels to block
  13 once the campaign is stable and the block-14/15 entanglements can be resolved deliberately.
