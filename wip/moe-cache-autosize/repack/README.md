# Repack: move the MoE expert cache + arena subsystem into block 06 (DONE, awaiting release)

**Goal (maintainer):** the MoE cache + arena/slab is *structural* llama.cpp work, independent of the
RDNA-specific kernel blocks. Put all of it in **block 06** and rebase blocks 07-15 onto it.

**Result: the new chain is built and VERIFIED — 16 blocks, final tree byte-identical to r24's.**
Nothing is released yet; `~/llama.cpp` still holds the shipped layout (block 15 owns it) plus the
uncommitted-to-a-release r24 fix.

## Where things live

| | |
|---|---|
| sandbox worktree | `~/llama-repack` (branch `repack06`), built from block 05 |
| the new chain | `a55e952b8..a7b8afdd2`, 16 blocks, tree `1a580f937447949e27f4f822b19714c1c8ebb826` |
| T_final (target) | `1a580f937447949e27f4f822b19714c1c8ebb826` (= the r24 tree in `~/llama.cpp`) |
| scripts | `repack.sh` (the whole run), `extract_v4.py` (hunk extraction), `resolve_conflicts.py` (additive merge) |
| the extraction | `subsystem-extract.diff` (76 hunks -> 68 after filtering) |

## The method (and why it is safe)

1. **06'** = block 06 + the subsystem, extracted as *keyword-filtered hunks* from
   `git diff cd98790ec <T_final>` (keywords: `moe_cache`, `ggml_cuda_slab`, `ggml_cuda_vmm`,
   `alloc_buffer_usage`, `drop_compute_buffers`, `moe_host_expert_bytes`, ...).  Applied **per file**
   (`git apply` is atomic: one refused hunk rolls back everything — that bit me once).
   * `moe-expert-cache.{cu,h}` + `ggml-cuda-vmm.h` land as the **final** files (byte-identical to T_final).
   * `mmvq.cu` is **excluded**: its 8 `moe_cache_*` refs are the RDNA-specific cold seam and stay with
     block 13's kernel work.  The module itself references **no** RDNA symbols (checked), so it stands alone.
   * block 09's `compute_headroom 16 -> 128` shares a hunk with the `alloc_buffer_usage` split, so it is
     **line-filtered out** of the extraction and stays in block 09.
2. **07..14 replayed with `git cherry-pick`**, conflicts resolved by **additive merge** (keep both sides,
   dedup) — 08:13, 13:37, 14:5 conflicts, all additive.
3. **15' = the remainder**: `git read-tree --reset <T_final>` + `checkout-index` + commit, so the tip is
   *by construction* T_final.
4. **Acceptance: the final tree equals T_final exactly** -> zero code change, proven.

## Result (per-block, vs the original)

| block | original | new | moved |
|---|---|---|---|
| 06 | 172 files, +2851 | 184 files, **+10025** | the whole subsystem in |
| 07 | +25 | +25 | identical |
| 08 | +5310/-138 | +5051/-131 | -259 (cache-adjacent) |
| 09 | +6/-1 | **+6/-1** | identical (preserved by the line filter) |
| 10, 11, 12 | — | **identical** | untouched |
| 13 | 29 files, +5933 | 21 files, +2142 | -3791 (the module creation) |
| 14 | 57 files, +7848 | 56 files, +7622 | -226 |
| 15 | +13848 | 207 files, +11043 | -2805 (the subsystem's evolution) |

## Remaining steps before it can ship

1. Spot-check that no block adds subsystem code that a later block then reverts (the collector's noise);
   if block 13/14 still touch `moe-expert-cache.*`, re-resolve those files with **take-ours** (the module is
   already final in 06') instead of the additive merge.
2. Build the new tip (same tree -> ccache hits) and run the gates: coherence, MTP rule-0, the server
   wide1 -> short -> wide2 scenario.
3. Regenerate `patches/` + `release.json` from the new chain, `validate-set.sh` (strict `git am` on a fresh
   tarball must reproduce the tree), whitespace-clean.
4. Docs: `patches/README.md` (the block-06/13/15 rows), README/MANIFESTS/BASELINE headers, WORKLOG, TODO.
5. Release r24 (this repack + the `MIN_MIB` fix), tag, push the delivery repo, refresh the fork branch.

**Note:** the r24 code fix (the early MoE-cache floor: `llama-model.cpp` preflight walks the real device
map, `common.cpp` calls it before the context, `moe-expert-cache.cu` never disables late) is already
committed in `~/llama.cpp` as block 15 and is part of T_final.
