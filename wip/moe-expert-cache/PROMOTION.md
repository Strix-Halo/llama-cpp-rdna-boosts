# Promotion record — `wip/moe-expert-cache` into the 16 delivery blocks (BETA)

**Branch:** `promote-moe-caching` (this repo, pushed to `origin`).
**Release label:** `v16-84e76d8a2-r28-moe-cache-beta3` (**not a release** — no tag, no GHCR
image, no merge to `main`).
**Fold worktree:** `~/llama-fold` (the beta3 reorg chain is branch `reorg-moe`, built from
`84e76d8a2`).
**Campaign source:** `~/llama-decode`, branch `wip-moe-devmap-r28`, tip `f5a79e6ab`
(base `60361cb9f` = delivery r28), net diff `git diff 60361cb9f..wip-moe-devmap-r28` = **12 files,
+5436/-48**; the cleaned delta is `git diff 60361cb9f..0f77c32d` (beta2 tip) = **11 files**.

| | |
|---|---|
| base | `84e76d8a2` (tree `5112eedbce0548ab9547d883e8aa54e993852e94`) |
| fold tip | `5bbba5d64be7a711261df8c185c5e10150f7801c` |
| net tree | `0fe985fbe28085f6e57d29802a1a016f5bf82c4c` (byte-identical to beta2) |

**beta3 (this revision).**  Re-partitions the campaign to the blocks whose code it extends (see the
mapping below).  The net tree is **unchanged** from beta2, so every beta2/campaign gate carries over;
the admission gates were re-run on the reorged chain and reproduce the hashes.

**beta2.**  Fixed the three new compiler warnings (the CPU routing profiler removed; the CPU/RPC iface
vtables name the new fields) and stripped the campaign's env-gated debug/A-B instrumentation (the
CPU-computes split, the adaptive staging probe, the `GGML_SCHED_*DBG` / `GGML_META_NOSYNC` /
`GGML_META_SCRATCH_MB` / `GGML_META_GATHER_NOPAD` knobs and the engine's bring-up knobs).

**beta1** (`ce06f7add`, tree `1922182…`) kept the tree byte-identical to the campaign's validated tree
and carried its instrumentation.

See `WORKLOG.md` 2026-10-02 (moe-cache beta3) and (moe-cache beta2).

## Mapping — campaign file → delivery block

| campaign file | block | why |
|---|---|---|
| `ggml/src/ggml-backend-impl.h` | **06** | generic backend iface (`moe_cache_update` / `_take_over` / `_promote` / `_gather`) |
| `ggml/src/ggml-cpu/ggml-cpu.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp` | **06** | name the four new iface fields (NULL) so the build stays warning-free |
| `ggml/src/ggml-backend.cpp` | **06** | scheduler half: routed-expert rebalance, merged per-layer MoE split, device gather + staging deferral, input takeover + deferred promotion; `GGML_ENV_STR` is relocated here from block 15 |
| `ggml/src/ggml-backend-meta.cpp` | **06** | the four `moe_cache_*` Meta delegations (pure additions); the `stage_gather` return-value guard stays in block 15 |
| `ggml/src/ggml-cuda/moe-expert-cache.cu` | **13** | the cache engine (self-contained new file) |
| `ggml/src/ggml-cuda/moe-expert-cache.h` | **13** | engine header |
| `ggml/src/ggml-cuda/mmvq.cu` | **13** | in-kernel slot lookup in `mul_mat_vec_q_moe` |
| `src/llama-arch.cpp` | **14** | `llm_arch_supports_sm_tensor()` rejects `LLM_ARCH_GEMMA4` (segmented expert upload unfinished) |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | **15** | CUDA consumer hooks + cache-band fusion stand-down; interleaves with block-15 HC fusion / weighted-down code |
| `ggml/src/ggml-cuda/common.cuh` | **15** | `h2d_pin`/`h2d_scratch` helpers the gather uses (interleaves with block-15 staging) |
| `ggml/src/ggml-backend-meta.cpp` (`stage_input` `stage_gather` hunk) | **15** | modifies r28 block-15's `stage_gather` staging |

Blocks **00-05, 07-12** change only in their `From <sha>` / `index` lines (bodies unchanged).
`ggml/include/ggml-backend.h` and `src/llama-graph.cpp` dropped out of the campaign delta in beta2
(the `ggml_backend_sched_set_moe_cpu_split` API and the CPU-computes split were removed), as did
`ggml/src/ggml-cpu/ggml-cpu.c` (the routing profiler).

**Why block 15 keeps `ggml-cuda.cu`, `common.cuh` and the `stage_input` hunk.**  Those campaign pieces
are interleaved with r28 block-15 code: `ggml-cuda.cu`'s hooks sit inside `ggml_cuda_try_fuse` next to
the block-15 `ggml_cuda_match_hc_mix` / qwen4exp weighted-down functions and
`ggml_cuda_cache_blocks_fusion`'s insertion site; `common.cuh`'s 4-line change is a duplicate comment
inside block 15's `h2d_scratch` block; the `stage_input` hunk edits the `stage_gather` call that r28
introduces in block 15.  Moving them earlier would require relocating those block-15 functions into
block 13 and inverting the block boundaries, so they stay with the memory campaign.  Block 15's tree
is set to the target tree, so it carries exactly the remaining r28 + campaign content.

## Method (reproducible)

Fresh linear rebuild from `84e76d8a2`.  For each block `i` in order:

1. `git cherry-pick <block_i>` (the r28 chain commit: `ec8d3fe33`…`60361cb9f`).
2. `git apply --3way --index /tmp/camp/<file>.patch` for every campaign file assigned to block `i`
   (per-file patches cut from `git diff 60361cb9f..0f77c32d -- <file>`, i.e. the cleaned beta2 delta;
   `--3way` merges against the r28 blob recorded in the patch's `index` line).
3. `git add -A && git commit --amend` (block 06/13/14 keep the original subject and append the
   campaign-fold provenance to the body).
4. For **block 15**, set the tree to the target (`git read-tree -u --reset 0f77c32d^{tree}`) and
   `git commit -C 60361cb9f`, so it carries exactly the remaining content.

The reorg resolved only these `--3way` conflicts: `ggml-backend.cpp` at block 06 (**one**, the
`wait_before_overwrite` refactor; resolved to the campaign form and the `GGML_ENV_STR` macro relocated
into block 06), `ggml-backend-meta.cpp` at block 06 (**two**: the `stage_input` hunk left for block 15,
and the meta iface initializer merging `.graph_optimize = nullptr` + the four `moe_cache_*` fields),
and `ggml-backend-meta.cpp` at block 14 (**one**: the iface initializer, resolved to
`.graph_optimize = ggml_backend_meta_graph_optimize` + the four fields).

The beta3 chain final tree is **`0fe985fbe28085f6e57d29802a1a016f5bf82c4c`**.  The 16 block commits:

```
d94fdf7426..  block 00
bcfcd3b46..  block 01
b091df4f9..  block 02
fc9f64c97..  block 03
f98727886..  block 04
2569fa971..  block 05
141fcfe69..  block 06  (campaign: iface + scheduler/Meta + warning-clean vtables + GGML_ENV_STR)
8b39526ca..  block 07
8ee91ddf5..  block 08
cdd2a6b08..  block 09
35f9b3daa..  block 10
06f7c0b19..  block 11
313be1050..  block 12
81f72f5d6..  block 13  (campaign: engine + mmvq slot lookup)
8301304ad..  block 14  (campaign: gemma4 guard)
5bbba5d64..  block 15  (campaign glue: ggml-cuda.cu + common.cuh + stage_input guard)
```

Regenerate with `scripts/make-patches.sh <worktree> 84e76d8a2 5bbba5d64…`, then
`scripts/make-release.sh --base 84e76d8a2 --base-tree 5112eedb… --tip 5bbba5d64… --tree 0fe985fb… --release v16-84e76d8a2-r28-moe-cache-beta3`.

## Verification (this fold)

* `scripts/validate-set.sh` **green** — artifact checksums, strict 16/16 `git am` on a fresh
  `84e76d8a2` codeload tarball, applied tree == `release.json.tree` (`0fe985fb…`).
* Clean `gfx1201` / ROCm 7.14 build **warning-free**, exit 0.
* `test-backend-ops -o MUL_MAT_ID` **929/929**.
* Byte-identity to the `-ncmoe 0` oracle **`de8be4d0c90c`** (2-GPU `-sm tensor`, 300 tok); width purity
  `none == n1 == n3 == n7 == 15038c19ddc8` (1-GPU `-sm layer`); MTP `n3` acceptance **0.75273**; deep
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
* ~~A future re-cut could move the scheduler/interface half to block 06 and the MoE kernels to block
  13~~ **DONE in beta3**: block 06 carries the iface + scheduler/Meta half, block 13 the engine + the
  `mmvq.cu` slot lookup, block 14 the arch guard.  `ggml-cuda.cu`, `common.cuh` and the `stage_input`
  guard stay in block 15 because they interleave with its fusion/staging code (see the mapping).
* **gemma4 `-sm tensor`** stays rejected until the segmented host-resident-expert async upload is
  finished (parked; `TODO.md`).
* Optional future cosmetic cleanup: the duplicate `h2d_pin`/gather comment that beta2's delta carries
  in `common.cuh` (a 4-line duplication inside the block-15 `h2d_scratch` block) could be dropped when
  block 15 is next touched.
