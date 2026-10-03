# `archive/work/issue-89` - indexer top-k block-path fix (promoted)

**This fix is in the delivery: release `v16-84e76d8a2-r36` (2026-10-05).**  The PR #90 record below is the
campaign history; `pr90-block-path-fix.patch` is the author's original one-file patch.  The delivery folded
it into **block 15** (`ggml/src/ggml-cuda/indexer-topk.cu`) together with a new backend-op regression test.

## Where this is now (2026-10-05, promoted to r36)

* Folded into **block 15** on top of r35; the recurrence is amended (only block 15 changes in content).
* Fork tip **`9b8b6f10815d285cd7f8828ab431686937873085`**, tree
  **`c595f29253ad70d693793d010f5e5399dadf57ae`**.
* `patches/` (16 patches), `release.json` and `rdna-boosts-all.patch` are regenerated; release string
  **`v16-84e76d8a2-r36`**.
* `scripts/validate-set.sh` strict **16/16** on a fresh `84e76d8a2` tarball, applied tree == recorded.
* Added `tests/test-backend-ops.cpp` case `test_indexer_topk_block` (registered as `INDEXER_TOPK`): three
  shapes, including a one-sequence cell offset and a two-stream (unified KV) cell map.  The two
  mismatching shapes **fail** on the unfixed r35 build and pass on r36, so the block/cell partition
  contract now has an oracle the way the rest of the qwen4exp ops do.
* Gates on gfx1201 / ROCm 7.14: `INDEXER_TOPK` 3/3, `TOPK_QSA` 4/4, `LIGHTNING_INDEXER` 225/225,
  `FLASH_ATTN_QSA` 26/26, `MUL_MAT_ID` 929/929; 4B `7386359e5dac` and 35B-A3B `cf7f8b23f404`
  unchanged from r35.
* **End-to-end reproduction on the reporter's model.**  The local Qwen3.8-Flash-Next UD-IQ3_XXS copy
  (77 GB), one R9700, `-ngl 99 -sm layer -c 65536 --n-cpu-moe 24 -ub 256 -ctk q8_0 -ctv q8_0 -cram
  2048`, `MOE_EXPERT_CACHE_MIB=1024`, `GGML_SCHED_SYNC_GRAPH_INPUTS=1`, two concurrent about-8k-token
  requests (second starts 13 s later).  The unfixed r35 kernel **crashes** with
  `Memory Fault Error ... kernel: void flash_attn_qsa<256, (ggml_type)8, false>(...)` and `Memory access
  fault ... Reason: Page not present or supervisor privilege`; the fixed r36 kernel runs the same
  scenario clean over three replays.  A sequential two-slot run is not sensitive (with `-kvu` an idle
  slot is saved and cleared when the next request arrives, so the sequences do not coexist), and
  `-np N` explicitly disables the auto unified KV, so `-kvu` is required to reproduce.

The rest of this file is the PR author's original write-up (from PR #90).

---

A fix for issue #89: the block fast path of the fused indexer top-k (`indexer_topk_radix_cuda_blocks`, block 15) can
leave output entries unwritten. One `git am` patch, one file (`ggml/src/ggml-cuda/indexer-topk.cu`, +82 lines):

- `pr90-block-path-fix.patch` (r35 tree `d08fbaf2` -> `40bf7aa8`; it also applies
  cleanly on r34 `3c07e1f6` -> `54a9f319`; `indexer-topk.cu` is the same in both).

## What goes wrong

The block path runs radix pass 1 at cell level, passes 2-4 at block level, then the gather
(`indexer_topk_write_blocks_grouped`) at cell level. Each hist-block `h` owns:

- in the block passes: blocks `[h*bchunk, (h+1)*bchunk)`;
- in the gather: cells `[h*bchunk*r, (h+1)*bchunk*r)`.

The gather's per-range bases come from `g_cnt`/`e_cnt`, which `indexer_topk_hist_accum` builds from those histograms.
The two partitions only agree when block `b` occupies cells `[b*r, b*r+r)`, i.e. one sequence laid out from cell 0.
Two common cases break that:

- **a unified KV holding several sequences** (default `-np`): blocks are keyed by (sequence, position bucket), so a
  block's cells sit wherever that sequence's cells are;
- **a single sequence after the first request**: the KV head is no longer at cell 0, so the cells are offset.

When the per-range counts are off, some ranges are given too few output slots and others too many. Entries are
left unwritten, holding whatever the buffer held before, or overwritten. If the stale bytes are out of range,
`flash_attn_qsa` gathers K/V at a garbage address (GPU page fault, the server hangs). If they happen to be in range,
nothing faults, but the attention silently uses the wrong cells.

The per-row radix totals don't depend on the partition, so the selection threshold (`states[row]`) is correct. Only
the per-range bases are wrong.

## The fix

After the last radix pass, a new kernel `indexer_topk_count_cells_grouped` recounts `g_cnt`/`e_cnt` over exactly
the cell ranges the gather walks, with the gather's own key logic (block key, `cell_pos`/`q_pos` visibility). Then
`indexer_topk_base_scan` and the gather run unchanged. Passes 2-4 stay block-level. The cost is one cell-level
counting pass. For a single sequence laid out from cell 0, the new counts equal the old ones, so that output does not
change.

## Validation

R9700 (gfx1201), Qwen3.8-Flash-Next GSQ IQ3_XXS, `llama-server -c 114688 --n-cpu-moe 24 -ub 256 -ctk q8_0 -ctv q8_0`,
`MOE_EXPERT_CACHE_MIB=1024`, `GGML_SCHED_SYNC_GRAPH_INPUTS=1`. Tested on r34 plus a local fused-hc_mix commit
that does not touch this file.

**The #89 repro** (default `-np`, two chat requests where the second starts while the first's prefill is in its last
chunk):

| build | result |
|---|---|
| r34 | `HSA_STATUS_ERROR_MEMORY_FAULT` on turn 1, 3 of 3 |
| r34 + a local index check before each `flash_attn_qsa` launch | caught: 12 of 49224 indices outside `[0, 8704)`, stale float bits |
| r34 + `LLAMA_INDEXER_NOBLOCK=1` | clean, 2 of 2 (12 turns each) |
| **r34 + this patch**, with the index check | **clean, 2 of 2 (24 turns), 0 out-of-range indices** |
| **r34 + this patch** (release image) | **clean, 12 turns** |

**Single sequence** (`-np 1`, three long temperature-0 chat prompts in turn, reasoning + answer hashed):

| build | prompt 1 (11k) | prompt 2 (22k) | prompt 3 (30k) |
|---|---|---|---|
| r34 | `c518bed18388` | `419bb91a7539` | `96703574d9c3` |
| r34 + `LLAMA_INDEXER_NOBLOCK=1` (the independent grouped cell path) | `c518bed18388` | `c6b26d7d68e6` | `ded218287bcd` |
| **r34 + this patch** | `c518bed18388` | `c6b26d7d68e6` | `ded218287bcd` |

The first prompt (fresh cache, cells from 0) agrees everywhere. The later ones match the independent path only with
the patch. Six short greedy prompts are identical on all three builds.

**Speed** (`llama-bench -ncmoe 24 -ub 256 -b 2048 -p 2048 -n 64 -d 0,30000 -r 2`):

| build | pp2048 | pp2048 @ d30000 | tg64 | tg64 @ d30000 |
|---|---|---|---|---|
| r34 | 749 ± 89 | 691 | 49.3 | 46.4 |
| r34 + `LLAMA_INDEXER_NOBLOCK=1` | 813 | 691 | 48.9 | 46.4 |
| r34 + this patch | 810 | 688 | 49.1 | 46.2 |

All three are equal within noise. In an earlier run `NOBLOCK` looked 2-9 % slower at depth, which was noise, so for
anyone who doesn't want to patch, `LLAMA_INDEXER_NOBLOCK=1` is a fine workaround until this or another fix lands.
