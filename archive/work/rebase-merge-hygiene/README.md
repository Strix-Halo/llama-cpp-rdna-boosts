# WIP — re-base merge hygiene (distribute the block-15 build fixes)

**Created:** 2026-10-05, after the `v16-a55e952b8-r1` re-base.
**For:** a follow-up session.
**Status:** delivery-quality task (not an experiment). It touches the canonical block chain.

---

## Why this WIP exists

The `v16-a55e952b8-r1` re-base replayed the 16 block commits with `git rebase --onto`, resolved the
conflicts, and then — after the replay — found a number of **build fixes** in conflict regions that
had been resolved with a missing closing brace or an unmigrated API. Following the r1 re-base's
precedent ("Post-rebase compile fix (folded into block 15)"), all of them were folded into **block
15**. Result: the final tree builds and `validate-set.sh` is green, but the intermediate blocks do
**not** each build/bisect:

- **block 13** committed `mmq.cu` / `mmq.cuh` / `mmvq.cu` with the `prec_src1`×`has_gate` merge
  unfinished (missing `}`, `ggml_prec`/`bool` mix-ups, a hardcoded gate prec that only block 15
  finalizes).
- **block 14** committed the *hybrid* `src/models/models.h` (`llama_model_qwen4exp::graph` with both
  the fork's QSA methods and upstream's dead kpool) and the matching hybrid `qwen4exp.cpp`; block 15
  replaces both with the fork's r37 versions.
- **block 01** committed `common/speculative.cpp` / `tests/test-recurrent-state-depth.cpp` with old
  batch-API calls that block 15 migrates.
- **block 08** committed `argsort.cu` / `norm.cu` / `unary.cu` / `ggml-cuda.cu` with braces that
  block 15 closes.

This is cosmetic for `scripts/apply-all.sh` (all 16 are applied, final tree validated) but it hurts
`git bisect`, per-block review, and future re-bases (a rebase of a broken intermediate commit
re-conflicts).

## Objectives

Redistribute the fixes so **every block commit builds on its own** (at least the default
`ggml-hip` + `llama` targets), without changing the final tree:

1. Inventory the fix hunks and assign each to its originating block:
   - block 01: `common/speculative.cpp`, `tests/test-recurrent-state-depth.cpp`;
   - block 08: `ggml/src/ggml-cuda/argsort.cu`, `norm.cu`, `unary.cu`, `ggml-cuda.cu`;
   - block 13: `ggml/src/ggml-cuda/mmq.cu`, `mmq.cuh`, `mmvq.cu`;
   - block 14: `src/models/models.h`, `src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.*`,
     `src/llama-model.cpp` (the `n_rs_batch` glm5-next ctor arg);
   - block 15: `common/sampling.cpp`, `src/llama-context.cpp`, `tests/test-backend-ops.cpp`, the
     remaining `ggml-cuda.cu` bits.
2. `git rebase -i a55e952b8` with `edit` stops on 01/08/13/14; at each stop, restore the relevant
   files from the current block-15 tree (`git checkout <r38-tip> -- <files>`), `git commit --amend`.
3. Leave block 15 with only its own genuine content.

## Acceptance

- `for c in $(git rev-list a55e952b8..HEAD); do git checkout -q $c; cmake --build <dir> --target llama; done`
  is green at every one of the 16 commits (or a documented per-block waiver).
- Final tree **unchanged**: `git rev-parse HEAD^{tree}` still equals
  `6a44aa2904772db02dbc88960397efe8138498df` (or the new tree if the distribution legitimately
  differs — then regenerate).
- `scripts/make-patches.sh` + `scripts/make-release.sh` + `scripts/validate-set.sh` re-run green
  (the tip/tree change, so `release.json` must be regenerated).

## Risk / notes

- Redistributing **rewrites the block commit SHAs**, so `release.json.tip`/`tree` change; the release
  tag `v16-a55e952b8-r1` already points at the old chain. Decide whether to keep the tag (and cut a
  `-r2`) or leave the tag as-is and only clean the chain for the next re-base.
- Do not run this while a container build from the tag is in flight without coordinating.

## Pointers

- `WORKLOG.md` 2026-10-05 (r1) "Conflicts resolved" and "Follow-ups".
- `git show def454e4c --stat` (block 15) shows the files it now carries beyond its own scope.
- `scripts/make-patches.sh`, `scripts/make-release.sh`, `scripts/validate-set.sh`.
