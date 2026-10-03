# lightning-indexer fusion convergence — RESULTS (resolved in `v16-a55e952b8-r4`)

**Resolved:** 2026-10-05.  **Decision: adopt upstream's `LLM_FUSED_OP_LIGHTNING_INDEXER`
registration; drop the `llama_prefetch_rows` prefetch (it is a measured pp512 regression vs the
fork's existing per-row `madvise` loop); drop/defer the score-accumulation, seed-free-mask and
standard-FA-index-list items.**  The one delivery change is the registration (blocks 14 + 15);
no functional/behaviour change.
**Original handover:** [`README.md`](README.md).
**Delivery:** `main` release `v16-a55e952b8-r4`, fork tip **`cd1485fd1`**, net tree
**`714f94f050dfce08c987a8a14467f456fe6e9d60`** (`validate-set.sh` green, strict 16/16 `git am`).

This is the last of the six r1 re-base follow-up WIPs; the other five are recorded in
`../qwen4exp-qsa-convergence/`, `../shared-expert-fusion-reconcile/`, `../mmq-prec-gate-fp4/`,
`../rebase-merge-hygiene/` and `../rebase-integration-audit/`.

---

## Item-by-item resolution

### 1. `llama_prefetch_rows` in the PLE n-gram path — **DROP** (measured regression)

The handover read upstream `185103dcf` as "our PLE path does not prefetch ahead".  That is not the
state of the fork: `llm_graph_input_ple::set_input()` already queues every gathered row's page with
`MADV_WILLNEED` in its host-resident branch (block 14, the managed-PLE campaign), and the managed
reader (`llama_lazy_reader::gather`, when `LLAMA_LAZY_BUF_MB` arms it) prefetches its cold pages
with `posix_fadvise(WILLNEED)` plus an I/O thread pool.

Replacing that loop with the generic `llama_prefetch_rows(per_layer_tok_embd, idx, n)` (which sorts
and merges the row ranges into batched `madvise` calls) is **byte-identical but slower** on this
box.  `llama-bench -p 512 -n 0 -r 4 -ngl 99 -sm tensor -lm none`, Qwen3.8-Flash-Next IQ4_NL, 3× R9700
(interleaved, same build script `~/bin/build-llama-rocm-714`):

| build | pp512 t/s (run 1 / run 2) |
|---|---|
| pristine fork (`-lm none`) | 1542 / 1530 |
| + registration (item 2) only | 1450 / 1507 |
| + registration **and** `llama_prefetch_rows` | 1220 / 1208 |

The helper costs ~15–20 % pp512 and is *not* needed for correctness; the fork's loop is the faster
path for this scattered, tiny-row (90 B/row, 16 heads) access pattern.  **Keep the fork's loop.**
The `-lm auto` / `-lm none` distinction makes no difference here — the PLE tensor is lazy
(`lazy_mode=auto`, 28.8 GiB > 4 GiB) under both, so it is mmap-backed either way.

### 2. Register the score as `LLM_FUSED_OP_LIGHTNING_INDEXER` — **ADOPT**

Adopted upstream `889edf43d`'s registration for both fused indexer-score nodes in our graph:

* `src/models/qwen4exp.cpp`, `build_qsa_top_k` fused-decode branch (`ggml_indexer_score`) —
  **block 14** (the branch exists there).
* the prefill WMMA arm (`ggml_lightning_indexer`) — **block 15** (block 15 introduces that arm).

`resolve_fused_ops()`'s Lightning Indexer probe could then report a layer/device mismatch instead
of the indexer silently falling off the layer's device.  **Caveat (inert today):** the base sets
`cparams.auto_flid = false` unconditionally (`src/llama-context.cpp`), so the `llm_fused_op_lid_probe`
never runs and the registration has no effect yet.  It is kept because it is exactly what upstream
does and it is a zero-risk alignment (a `push_back`); it costs nothing and takes effect if the base
ever enables the probe.  The handover's "the probe already exists" was read as "the probe already
runs" — it exists but is disabled.

### 3. Per-head accumulate-into-one-score in the non-WMMA chain — **DROP/DEFER**

The handover described upstream's intermediate PR step.  The **final** upstream change
(`889edf43d`) replaces the all-heads product with `ggml_lightning_indexer` (all-ones weights), not
a per-head loop.  Our fork already uses `ggml_lightning_indexer` in the prefill WMMA band
(`n_tps >= 128`, `idx_dim == 128`, `n_idx_h == 4`); the remaining per-op chain is the **deliberate**
gfx1100 issue-#59/#60 path where the generic-vec fused op measured ~7 % slower, and its peak is
already bounded by `GGML_QSA_SCORE_MEM` (default on): `relu_inplace` removes the score's double
buffer and the >128 MiB branch chunks the assembly.  Validating the further memory claim needs a
gfx1100 (not present here), and re-baselining the chain onto the fused op would change prefill
numerics on the wrong side of issue #60.  **No change.**

### 4. Seed-free mask construction — **DROP** (already done)

Upstream `4e2713c16` rewrites the **kpool** mask build (`build_qsa_sel`), which our graph does not
use.  Our `build_attn_qsa` mask already has no seed tensor: it does
`ggml_fill(kq_mask, -INFINITY)` in place on the input, and `zeros` is a fresh
`ggml_new_tensor_4d` + `ggml_fill`.  Nothing to adopt.

### 5. Feed the standard FA kernels an index list — **DEFER** (large separate campaign)

Presenting the `indexer_top_k` cell list to `fattn-tile`/`fattn-mma-f16`/`fattn-vec` via the
sparse/`n_kv_max` mechanism is a multi-session re-port with its own correctness and perf gates; the
handover itself calls it the biggest-leverage but separate campaign.  The fused `FLASH_ATTN_QSA`
kernel stays the RDNA fast path.  Not started; no code changed.

## Validation (gfx1201 / ROCm 7.14, `~/bin/build-llama-rocm-714`)

* Clean `all`-target build from the final tip, **warning-free**.
* `test-backend-ops`: `INDEXER_TOPK,INDEXER_SCORE,FLASH_ATTN_QSA,HC_MIX` **59/59** on each of
  ROCm0/1/2; `FLASH_ATTN_EXT` **6358/6358** on each.
* Coherence unchanged: dense 4B 3-GPU `-sm tensor` **`1c5d32ac537d`**; qwen4exp Flash-Next IQ4_NL
  3-GPU `-sm tensor -lm none --reasoning off -n 20` **`359ff4337837`** (and `-lm auto` identical).
* `tg128` (`-p 0 -n 128 -r 4 -lm none`): **56.4** t/s vs pristine **57.1** — within noise (the
  registration is not on any compute path).
* `scripts/validate-set.sh` green: strict 16/16 `git am` on a fresh `a55e952b8` tarball, applied tree
  `714f94f050dfce08c987a8a14467f456fe6e9d60`.

## Delivery record

* **Block 14** amended (`b9a4c4814`): the decode fused-score registration.
* **Block 15** re-based onto it and amended (`cd1485fd1`): the WMMA prefill registration.  The
  rebase applied with no conflicts; the 14→15 delta is only those lines (verified against the old
  tip: `git diff 159dcf2ea cd1485fd1 -- src/models/qwen4exp.cpp`).
* `patches/0014`/`0015` regenerated; `release.json` tip/tree/hashes updated; **release name stays
  `v16-a55e952b8-r3` in `release.json` until the r4 cut** (this ships in r4 together with the
  `qwen4exp-qsa-convergence` amendment already on `main`).
* `patches/README.md` block-14/15 notes + the `0014`/`0015` table rows; `WORKLOG.md` 2026-10-05
  (lightning-indexer-fusion).
