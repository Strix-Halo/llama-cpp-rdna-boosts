# Promotion record — `wip/moe-expert-cache` into the 16 delivery blocks (BETA)

**Branch:** `promote-moe-caching` (this repo, pushed to `origin`).
**Release label:** `v16-84e76d8a2-r28-moe-cache-beta5` (**not a release** — no tag, no GHCR
image, no merge to `main`).
**Fold worktree:** `~/llama-fold` (the current beta5 chain is branch `beta5-fold`, built from
`84e76d8a2`).
**Campaign source:** `~/llama-decode`, branch `wip-moe-devmap-r28`, tip `f5a79e6ab`
(base `60361cb9f` = delivery r28), net diff `git diff 60361cb9f..wip-moe-devmap-r28` = **12 files,
+5436/-48**; the cleaned delta is `git diff 60361cb9f..0f77c32d` (beta2 tip) = **11 files**.

| | |
|---|---|
| base | `84e76d8a2` (tree `5112eedbce0548ab9547d883e8aa54e993852e94`) |
| fold tip | `8e16c882ad8ebe6d7f3498e5940758f2d8802611` |
| net tree | `65276106fc5a6f62e1d81f4975c4816012ea4fc4` (== the validated `beta5-clean` tree) |

**beta5 (this revision).**  Three things over beta4: (1) the WIP one-time expert-head zero was launched
**after** the gather, so it zeroed the first 64 bytes of the routed experts the gather had just written
— the byte-identity regression; it now launches **before** the gather (both splits reproduce their
`-ncmoe 0` oracles, `de8be4d0c90c` / `15038c19ddc8`, throughput-neutral).  (2) The device gather
**registers its expert table** on the gather path too (it was registered only on the host-upload hook,
so a qwen4exp prefill registered just the last layer and the deferred arena sizing latched on 3 tables,
collapsing decode below the uncached path).  (3) The gather gate is **model-aware**: an expert table
`>= 224 MiB` always gathers (unconfounded, `--lazy-mode off`): qwen4exp ub8192 1403 -> 3065, Q8_0
neutral, Q4_K_M keeps the width gate.  Also: the qwen4exp PLE mmap warm-up is documented as the reason
every prior qwen4exp prefill comparison needs `-lzm off`, and the 128K/q8_0 Q8_0 target is tuned and
coherence-verified.  Folded into **block 06** (`ggml-backend.cpp` gate) and **block 13**
(`moe-expert-cache.cu`); `release.json` regenerated; `scripts/validate-set.sh` green (strict 16/16
`git am`, applied tree `65276106f…`).  See `WORKLOG.md` 2026-10-02 (moe-cache beta5 fold) and
(moe-cache beta5 validation).

**beta4 (this revision).**  Fixes two prefill regressions the campaign's always-on scheduler changes
introduced, found by a same-box A/B against `main` (r28): the device gather is width-gated to the
below-`sched_stage_min_tokens` band (above it whole-shard staging wins) and the routed-expert rebalance
is gated to the decode/verify band (`MUL_MAT_ID` `ne[2] <= 8`).  Prefill then matches/beats r28 at
every cell while the below-gate gather wins and the cache decode wins are kept.  See `WORKLOG.md`
2026-10-02 (moe-cache beta4).

**beta3.**  Re-partitioned the campaign to the blocks whose code it extends (see the mapping below).

**beta2.**  Fixed the three new compiler warnings (the CPU routing profiler removed; the CPU/RPC iface
vtables name the new fields) and stripped the campaign's env-gated debug/A-B instrumentation (the
CPU-computes split, the adaptive staging probe, the `GGML_SCHED_*DBG` / `GGML_META_NOSYNC` /
`GGML_META_SCRATCH_MB` / `GGML_META_GATHER_NOPAD` knobs and the engine's bring-up knobs).

**beta1** (`ce06f7add`, tree `1922182…`) kept the tree byte-identical to the campaign's validated tree
and carried its instrumentation.

See `WORKLOG.md` 2026-10-02 (moe-cache beta5 fold), (moe-cache beta5 validation), (moe-cache beta4),
(moe-cache beta3) and (moe-cache beta2).

## Mapping — campaign file → delivery block

| campaign file | block | why |
|---|---|---|
| `ggml/src/ggml-backend-impl.h` | **06** | generic backend iface (`moe_cache_update` / `_take_over` / `_promote` / `_gather`) |
| `ggml/src/ggml-cpu/ggml-cpu.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp` | **06** | name the four new iface fields (NULL) so the build stays warning-free |
| `ggml/src/ggml-backend.cpp` | **06** | scheduler half: routed-expert rebalance (now decode-band only), merged per-layer MoE split, device gather + staging deferral (now below-gate only), input takeover + deferred promotion; `GGML_ENV_STR` is relocated here from block 15 |
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

The beta4 chain final tree is **`a4c8564963f9f16f760f3a4b1805516f323b8cab`**.  The 16 block commits:

```
d94fdf7426..  block 00
bcfcd3b46..  block 01
b091df4f9..  block 02
fc9f64c97..  block 03
f98727886..  block 04
2569fa971..  block 05
54279601d..  block 06  (campaign: iface + scheduler/Meta + warning-clean vtables + GGML_ENV_STR; beta4 prefill fix)
09288e6bc..  block 07
adbf7f720..  block 08
24ea78867..  block 09
10bfe2f03..  block 10
73080068c..  block 11
7aacc647d..  block 12
88fd3e8f8..  block 13  (campaign: engine + mmvq slot lookup)
08f049b25..  block 14  (campaign: gemma4 guard)
f29745280..  block 15  (campaign glue: ggml-cuda.cu + common.cuh + stage_input guard)
```

Regenerate with `scripts/make-patches.sh <worktree> 84e76d8a2 f29745280…`, then
`scripts/make-release.sh --base 84e76d8a2 --base-tree 5112eedb… --tip f29745280… --tree a4c85649… --release v16-84e76d8a2-r28-moe-cache-beta4`.

## Verification (this fold)

* `scripts/validate-set.sh` **green** — artifact checksums, strict 16/16 `git am` on a fresh
  `84e76d8a2` codeload tarball, applied tree == `release.json.tree` (`a4c85649…`).
* Clean `gfx1201` / ROCm 7.14 build **warning-free**, exit 0.
* `test-backend-ops -o MUL_MAT_ID` **929/929**.
* Byte-identity to the `-ncmoe 0` oracle **`de8be4d0c90c`** (2-GPU `-sm tensor`, 300 tok); width purity
  `none == n1 == n3 == n7 == 15038c19ddc8` (1-GPU `-sm layer`); MTP `n3` acceptance **0.75273**; deep
  coherence rc=0, 13 sections, `## Conclusion`.
* **Prefill A/B vs `main` (r28), `pp8192` ub8192** — beta4 matches/beats r28 at every cell
  (2gpu tensor ncmoe 0/16/32/40 = 7305/6320/5596/5259 vs 7269/6306/5606/5257; 2gpu layer 16/40 =
  5551/4531 vs 5512/4511; 1gpu 16/40 = 5611/5405 vs 5593/5398), where beta3 lost 15-56 %.  Below-gate
  gather wins kept (ub512 708 vs r28 603; ub1024 1211 vs 1016); gather gate `pp2048 -ub 512` **723.6**.
  Cache decode unchanged (1gpu layer 39.7 -> 74.2 -> 81.2 t/s; 2gpu tensor 32.9 -> 78.6).

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
* ~~The campaign's scheduler changes must not regress prefill~~ **FIXED in beta4**: the device gather is
  width-gated to the below-`sched_stage_min_tokens` band and the routed-expert rebalance is gated to
  the decode/verify band (`MUL_MAT_ID` `ne[2] <= 8`).  Prefill matches/beats r28 at every measured
  cell; the below-gate gather wins and the cache decode wins are kept.  See `WORKLOG.md` 2026-10-02
  (moe-cache beta4).
* **gemma4 `-sm tensor`** stays rejected until the segmented host-resident-expert async upload is
  finished (parked; `TODO.md`).
* Optional future cosmetic cleanup: the duplicate `h2d_pin`/gather comment that beta2's delta carries
  in `common.cuh` (a 4-line duplication inside the block-15 `h2d_scratch` block) could be dropped when
  block 15 is next touched.
