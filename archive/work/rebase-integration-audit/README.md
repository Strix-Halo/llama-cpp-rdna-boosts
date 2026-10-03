# WIP — re-base integration audit (paths that compiled but were not exercised)

**Created:** 2026-10-05, after the `v16-a55e952b8-r1` re-base.
**For:** a follow-up session.
**Status:** **NOT part of the delivery** (validation/hardening only).

---

## Why this WIP exists

The r1 re-base migrated several fork paths onto upstream APIs and merged a few same-slot changes.
They **compile** and the standard gates pass (`MUL_MAT_ID` 931/931, `FLASH_ATTN_EXT` 6358/6358,
`HC_MIX` 30/30, `FLASH_ATTN_QSA` 26/26, `INDEXER_TOPK` 3/3, `GATED_DELTA_NET` 46/46, `RMS_NORM`
51/51, 4B `1c5d32ac537d`, qwen4exp `359ff4337837`) — but the gates this box can run do not exercise
the specific migrated paths. Each item below is a real, untested touch-point from the re-base.

## Items

### 1. LF / DFlash device path (`common/speculative.cpp`, `GGML_LF_DFLASH_DEV=1`)

The block-15 conflict added `llama_batch_free(batch)` / `llama_batch_free(batch_inject)` (now
`common_batch`, no such free) and `batch_in.pos[row]` (now `batch_in.tokens[row].pos[0]`). The
**F1 device-to-device layer-input path** in the DFlash drafter
(`common_speculative_impl_draft_dflash::process`) was migrated blind. It is opt-in and this box had no
DFlash drafter when it was written; there is now `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`.

**Task:** run the DFlash drafter with `GGML_LF_DFLASH_DEV=1` and `=0` and confirm byte-identical
greedy output and no host/device race; check `is_mrope` (`--reasoning`) and multi-ubatch prefill.

### 2. `llama_context::extract_layer_inputs` bool + F1 device path

Upstream `185103dcf` made `extract_layer_inputs` return `bool` (was `void` in r37). The merge kept our
F1 device branch and made it `return true`; the packed host branch returns `extracted`. Both the
`lf_layer_dev` allocation and the `lf_consumed_ev` event path are untested after the merge.

**Task:** exercise with the DFlash drafter (item 1); add a minimal assertion that the caller's
`extract_all_idxs` use (upstream's new return-value consumer in `llama-context.cpp`) is honoured.

### 3. `common_sampler_clone` merge (S2 + upstream `rng`)

Block 15's S2 optimization leaves `cur` / `cur_p` empty in a clone ("scratch, rebuilt by the next
sample"), while upstream `1fb7ef3e3` copies `.rng` for probabilistic/rejection sampling. The merged
initializer now has `.cur = {}`, `.cur_p = {}`, `.rng = gsmpl->rng`, plus our `t_total_us`/`fast_*`.

**Task:** confirm no code path reads a clone's `cur` / `cur_p` before the next sample — in particular
the rejection-sampling path (`dp.result_q`, `common_sampler_sample_and_accept_n_rejection`) and MTP
`--draft-probabilistic`/`--spec-type draft-mtp`. A same-seed run with probabilistic drafting on vs
off, and `common_sampler_copy` vs `common_sampler_clone` equivalence, are the gates.

### 4. `n_rs_batch` in upstream's glm5-next hybrid constructor

The fork threads `n_rs_batch` into `llama_memory_hybrid_idx`'s ctor. Upstream's new GLM5-Next case
(`llama-model.cpp`) did not pass it; the merge added `/* n_rs_batch */ cparams.n_rs_batch`. There is
no GLM5-Next model on this box, so it is compile-only.

**Task:** add an `LLM_ARCH_GLM5_NEXT` fixture (or assert the ctor's `n_rs_seq`/`n_rs_batch` relation)
so a future upstream rename of the recurrent ctor is caught; run `test-llama-archs` if a GLM5-Next
GGUF becomes available.

### 5. Argsort tie-break in the multi-column `bitonic_step`

Upstream `6a2743f02` made the bitonic argsort handle rows wider than one block (several columns per
thread). Our deterministic index tie-break was moved inside `bitonic_step`, so it should apply to
every owned column — but only single-block rows are covered by the existing `test-backend-ops`
`ARGSORT` matrix here.

**Task:** add/extend a `test-backend-ops -o ARGSORT` case with `ncols > 1024` (and duplicate values)
and assert the tie-break order matches the stable (CUB) result; this backs the fused MoE router's
determinism (`GREEDY-PURITY.md` §20/§31).

### 6. `tests/test-recurrent-state-depth.cpp` batch-API migration

Migrated from `common_batch_add`/`llama_batch`/`llama_decode` to `common_batch.add`/`llama_process`.
It is a test binary, so the re-base only checks that it compiles.

**Task:** run it against `Qwen3.6-35B-A3B` (Q4_K_M/Q8_0 as in `AGENTS.md`) for the `n_rs_seq`/rollback
sweep and compare to the pre-rebase record (the r1 note records **153 pre-existing Phase-B failures**
at large `n_rs_batch`; confirm the same count and max diffs, i.e. no new regression).

### 7. Block-15 S2 fast top-k vs upstream's sampling changes

`common/sampling.cpp`: ours keeps `fast_k`/`fast_bias`/`fast_bias_ids` (S2 exact top-k fast path,
`GGML_LF_FAST_TOPK`) and drops `cur`/`cur_p` in the clone; upstream added `rng` and the probabilistic
sampler. Validate that the S2 path and the probabilistic path do not interact (the fast path must
stand down whenever the chain is not "logit-bias only + top-k").

**Task:** run `tests/test-speculative-adaptive` + a greedy/probabilistic sampling A/B with
`GGML_LF_FAST_TOPK=0/1` and `--temp 0`/`--temp 1`.

## Acceptance

- Each item has a runnable gate added to the appropriate test or `benchmarks/` protocol, not just a
  one-off assertion.
- Any behaviour difference is either fixed or recorded as an accepted limitation with the measured
  impact.

## Pointers

- `common/speculative.cpp` (`common_speculative_impl_draft_dflash`), `src/llama-context.cpp`
  (`extract_layer_inputs`), `common/sampling.cpp` (`common_sampler_clone`/`_copy`),
  `src/llama-model.cpp` (`LLM_ARCH_GLM5_NEXT`), `ggml/src/ggml-cuda/argsort.cu` (`bitonic_step`),
  `tests/test-recurrent-state-depth.cpp`, `tests/test-speculative-adaptive.cpp`.
- upstream commits: `f1ea20621`, `60e9cf7a7`, `1fb7ef3e3`, `185103dcf`, `6a2743f02`.
- `benchmarks/mtp-adaptive-methodology.md`; `WORKLOG.md` 2026-10-05 (r1).
