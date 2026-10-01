# Promotion record — `wip/moe-expert-cache` into the 16 delivery blocks (BETA)

**Branch:** `promote-moe-caching` (this repo, pushed to `origin`).
**Release label:** `v16-84e76d8a2-r28-moe-cache-beta2` (**not a release** — no tag, no GHCR
image, no merge to `main`).
**Fold worktree:** `~/llama-fold`, branch `fold-moe-clean`, built from `84e76d8a2`.
**Campaign source:** `~/llama-decode`, branch `wip-moe-devmap-r28`, tip `f5a79e6ab`
(base `60361cb9f` = delivery r28), net diff `git diff 60361cb9f..wip-moe-devmap-r28` = **12 files,
+5436/-48**.

| | |
|---|---|
| base | `84e76d8a2` (tree `5112eedbce0548ab9547d883e8aa54e993852e94`) |
| fold tip | `0f77c32d1473147e811d445119e47ea28561be18` |
| net tree | `0fe985fbe28085f6e57d29802a1a016f5bf82c4c` |

**beta2 pass (this revision).**  beta1 (`ce06f7add`, tree `1922182…`) kept the tree byte-identical to the
campaign's validated tree and carried its instrumentation.  beta2 fixes the three new compiler warnings
(the CPU routing profiler removed; the CPU/RPC iface vtables name the new fields) and strips the
campaign's env-gated debug/A-B instrumentation (the CPU-computes split, the adaptive staging probe, the
`GGML_SCHED_*DBG` / `GGML_META_NOSYNC` / `GGML_META_SCRATCH_MB` / `GGML_META_GATHER_NOPAD` knobs and the
engine's bring-up knobs).  The default path is unchanged; the admission gates re-ran green
(`validate-set.sh`, warning-free build, `MUL_MAT_ID` 929/929, byte-identity `de8be4d0c90c` 2-GPU tensor,
width purity `none == n1 == n3 == n7 == 15038c19ddc8` 1-GPU layer, MTP `n3` 0.729, deep coherence rc=0
13 sections).  See `WORKLOG.md` 2026-10-02 (moe-cache beta2).

## Mapping — campaign file → delivery block

| campaign file | block | why |
|---|---|---|
| `ggml/src/ggml-backend-impl.h` | **06** | generic backend iface (`moe_cache_update` / `_take_over` / `_promote` / `_gather`) |
| `ggml/src/ggml-cpu/ggml-cpu.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp` | **06** | name the four new iface fields (NULL) so the build stays warning-free |
| `src/llama-arch.cpp` | **14** | `llm_arch_supports_sm_tensor()` rejects `LLM_ARCH_GEMMA4` (segmented expert upload unfinished) |
| `ggml/src/ggml-cuda/moe-expert-cache.cu` | **15** | the cache engine; authored against the r28 tree |
| `ggml/src/ggml-cuda/moe-expert-cache.h` | **15** | engine header |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | **15** | CUDA consumer hooks + cache-band fusion stand-down; uses block-15 HC matcher / weighted-down context |
| `ggml/src/ggml-cuda/mmvq.cu` | **15** | in-kernel slot lookup in `mul_mat_vec_q_moe` |
| `ggml/src/ggml-backend.cpp` | **15** | scheduler takeover / promote / gather hooks; uses block-15 `GGML_ENV_STR` |
| `ggml/src/ggml-backend-meta.cpp` | **15** | Meta delegation; uses block-15 r16 `stage_gather` chunk geometry |
| `ggml/src/ggml-cuda/common.cuh` | **15** | block-15 `h2d_scratch` (the `GGML_META_SCRATCH_MB` bound is removed in beta2) |

Blocks **00-05 and 07-13** change only in their `From <sha>` / `index` lines (bodies unchanged).
`ggml/include/ggml-backend.h` and `src/llama-graph.cpp` dropped out of the campaign delta in beta2
(the `ggml_backend_sched_set_moe_cpu_split` API and the CPU-computes split were removed), as did
`ggml/src/ggml-cpu/ggml-cpu.c` (the routing profiler).

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

The beta2 chain is a fresh linear rebuild from `84e76d8a2` (the beta1 `fold-moe` chain plus the
cleanup patch, folded into blocks 06 and 15 and with the two amended commit messages).  Its final tree is
**`0fe985fbe28085f6e57d29802a1a016f5bf82c4c`**.  The 16 amended block commits:

```
9514e82525a3  block 00
829f7b147d71  block 01
4012f449df76  block 02
b3d1c11ab100  block 03
0ea38dcdacea  block 04
d806a41a0309  block 05
9a56767ef336  block 06  (campaign: iface + warning-clean other-backend vtables)
cf5365368d61  block 07
52312ebf77c9  block 08
4c0936b6a397  block 09
33ceb41cc5fc  block 10
cecb1bd274d4  block 11
4dd1832a7f24  block 12
c7d95e20f59f  block 13
61e4b0534724  block 14  (campaign: gemma4 guard)
0f77c32d1473  block 15  (campaign: cache engine + consumers)
```

Regenerate with `scripts/make-patches.sh /tmp/rebuild 84e76d8a2 0f77c32d…`, then
`scripts/make-release.sh --base 84e76d8a2 --base-tree 5112eedb… --tip 0f77c32d… --tree 0fe985fb…`.

## Verification (this fold)

* `scripts/validate-set.sh` **green** — artifact checksums, strict 16/16 `git am` on a fresh
  `84e76d8a2` codeload tarball, applied tree == `release.json.tree` (`0fe985fb…`).
* Clean `gfx1201` / ROCm 7.14 build **warning-free**, exit 0.
* `test-backend-ops -o MUL_MAT_ID` **929/929**.
* Byte-identity to the `-ncmoe 0` oracle **`de8be4d0c90c`** (2-GPU `-sm tensor`, 300 tok); width purity
  `none == n1 == n3 == n7 == 15038c19ddc8` (1-GPU `-sm layer`); MTP `n3` acceptance **0.729**; deep
  coherence rc=0, 13 sections, `## Conclusion`.

## Known beta follow-ups

* ~~Three new compiler warnings~~ **FIXED in beta2**: the CPU routing profiler is removed (its
  `dst->name != NULL` tautology was the warning), and the CPU/RPC iface vtables name the four new
  `moe_cache_*` fields (NULL).
* ~~Env-gated debug/A-B knobs ride along~~ **FIXED in beta2**: the CPU-computes split, the adaptive
  staging probe (`GGML_SCHED_STAGE_AUTO`/`GGML_SCHED_GATHER_FIRST`), the `GGML_SCHED_*DBG`
  input-loop instrumentation, `GGML_META_NOSYNC`, `GGML_META_SCRATCH_MB`, `GGML_META_GATHER_NOPAD`,
  `GGML_CUDA_CACHEDBG`/`GGML_CUDA_FUSE_LOG` and the engine's bring-up knobs (`_DEBUG`, `_VERIFY`,
  `_SELFTEST`, `_ASSERT`, `_FAIL_ALLOC`, `_PROGRESS`, `_TIMING`, `_SKIP_ROLE`, `_FORCE_COPY`,
  `_FORCE_DEVMAP`, `_NOEVICT`, `_ADMIT`, `_COLD`, `_WARMUP_TOKENS`, `_PREFILL_LOAD`, `_TABLES`) are
  stripped.  The functional default-on kill switches (`_MIB`, `_SLOTS`, `_PERIOD`, `_TOUCH`, `_FILL`,
  `_RESERVE_MIB`, `_DEVMAP`, `_DEVPOLICY`, `_KSLOT`, `_PREFILL_SEED`, `_PREFILL_SEED_N`,
  `_PROVISIONAL`) remain.
* **gemma4 `-sm tensor`** stays rejected until the segmented host-resident-expert async upload is
  finished (parked; `TODO.md`).
* **A future re-cut** could move the scheduler/interface half to block 06 and the MoE kernels to block
  13.  **Investigated in beta2:** the iface declaration is already in block 06; the move is blocked by
  the campaign's scheduler code overlapping (and using) r28's block-15 `GGML_ENV_STR` macro and the
  meta `stage_gather` staging.  `git apply --3way` of the cleaned `ggml-backend.cpp` at r28's block-06
  tree gives **one** conflict (the `wait_before_overwrite` refactor); `ggml-backend-meta.cpp` gives
  **two** (both in `stage_gather`).  A correct re-cut must first relocate those r28 block-15 pieces
  into block 06 and resolve the block-15 cherry-pick duplicates; `ggml-cuda.cu`/`mmvq.cu` interleave
  with block-15's HC fusion matcher.  Parked with the exact conflict points in `WORKLOG.md` 2026-10-02
  (moe-cache beta2).
