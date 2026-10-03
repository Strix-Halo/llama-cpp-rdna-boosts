# qwen4exp QSA convergence — DECISION + AUDIT

**Resolved:** 2026-10-05, as an **unreleased block-14 amendment on `main`** (tip `b6529d088`, tree
`c77aeb55c91972257e228adca4cbcaa30d649be5`).  The r3 tag is unchanged; this change ships in **r4**
together with the `lightning-indexer-fusion` fold (see `wip/lightning-indexer-fusion/README.md`).
**Original handover:** [`README.md`](README.md).

---

## Decision

**(A) Keep the fork's fused QSA graph, and adopt the one upstream qwen4exp-local fix that applies
(`10f340d1a`'s `hc_init` graph-split placement).  Defer (B) adopted-upstream-graph and (C)
kpool-pooling, and defer item 4, with the triggers below.**

Rationale: the fork's graph is the validated RDNA performance path (qwen4exp prefill 1448 t/s /
decode 55.8 t/s on 3× R9700, r1 numbers) and there is no measured upstream perf parity to justify a
re-port.  (B)/(C) each require re-porting the fused score/top-k/QSA/FA ops onto upstream's
kpool/lightning-indexer graph or moving the pool storage, with no proven benefit and real risk to the
`359ff4337837` behaviour.  Defer until a concrete driver appears.

**Triggers to revisit:** upstream qwen4exp prefill/decode closes the gap on this box; a
correctness/memory bug in the fork's QSA path that upstream's graph already fixes; or a re-base where
the two graphs stop composing.

## Item-by-item resolution

### Item 1 — audit upstream qwen4exp fixes (DONE)

All of these are in the re-base fork point `a55e952b8`; the fork's block 14 keeps its own
`qwen4exp.cpp` and thereby drops the upstream **qwen4exp.cpp-local** changes.  The shared-infra
halves (memory/recurrent/model/gguf) remain in the tree because only `qwen4exp.cpp`,
`models.h`, `llama-memory-hybrid-idx.*` were replaced.

| commit | what | verdict |
|---|---|---|
| `66e0c17ee` fix qwen4exp | replace `build_qsa_top_k`/`llm_graph_input_qsa` with the kpool `build_qsa_sel`/`build_inp_kpool`; add `indexer_kpool*` hparams | **drop** — this is the fork-vs-upstream graph fork; `indexer_kpool*` hparams stay in `llama-hparams.h` and serve glm5-next |
| `159c651f5` fix tests | kpool scatter "each dead slot into its own dump row" + `set_input_kpool` pad-cell fix | **drop** the kpool half; the `llama-memory-hybrid-idx.cpp` half is already in the tree (base) and serves glm5-next |
| `c061df198` MTP | upstream MTP + shared-infra (`llama-memory-recurrent.cpp` `state_clear`, `build_rs`, `llama-model.*`, `common/speculative.cpp`) | **already in the tree** (shared files, base); the fork's MTP is independent and validated (r37) |
| `10f340d1a` `-sm tensor` | `ggml_build_forward_expand(gf, res_hc)` after `hc_init`; un-gate `sm_tensor` for qwen4exp | **ADOPT the one-liner; keep the gate** (see item 2) |
| `4e2713c16` mask construction | kpool mask seed/repeat simplification | **drop** (kpool); the RDNA analogue is `wip/lightning-indexer-fusion/` |
| `889edf43d` indexer score memory | per-head in-place score + `ggml_lightning_indexer` | **drop** (kpool); RDNA analogue is `wip/lightning-indexer-fusion/` |

Adopted change (`src/models/qwen4exp.cpp`, block 14, exactly upstream's fix):
```cpp
cb(res_hc, "hc_init", -1);
// make sure hc_init is in the same graph split as the first layer (-sm tensor).  Upstream
// 10f340d1a; without it the REPEAT stays on the CPU when the PLE gather is a CPU node and the
// meta splitter views a host-resident reshaped node.
ggml_build_forward_expand(gf, res_hc);
```
The graph already carried the analogous `ggml_build_forward_expand(gf, ple_emb)` line for the PLE
input, so this is the same idiom.  **Test:** byte-identical on both delivery coherence gates
(see Evidence).

### Item 2 — `-sm tensor` gate (RESOLVED: keep the HIP gate)

Our block 14 has `llm_arch_supports_sm_tensor(qwen4exp)` return true only under `GGML_USE_HIP`.  On
ROCm this already allows the split, so upstream's un-gating is a no-op here; the 3-GPU `-sm tensor`
`359ff4337837` gate passes.  The gate's `#else` is kept because our *fused* QSA/HC ops fall back to
the CPU on Vulkan/Metal/SYCL and the meta splitter cannot reconcile the mirrored-vs-split operand
states — a different problem from the PLE/`hc_init` placement that upstream fixed.  The adopted
`hc_init` fix removes that half of the hazard and keeps us aligned with upstream if the gate is ever
lifted after NVIDIA validation.

### Item 3 — dead upstream kpool (ALREADY RESOLVED by r2)

There is **no dead qwen4exp kpool code**.  `qwen4exp.cpp` contains no kpool symbol; the
`llm_graph_input_kpool` / `build_inp_kpool` / `build_kpool_select` declarations in `models.h` are
inside `llama_model_glm5_next`.  The two pooling stacks in the shared memory class are both live:

| stack | symbols | model |
|---|---|---|
| fork QSA | `set_input_qsa`, `get_pool`, `qsa_score_key_limits`, `build_qsa_top_k`, `build_qsa_store_k` | qwen4exp |
| upstream kpool | `set_input_kpool`, `kpool_access`, `gather_mla_rows`, `build_inp_kpool`, `build_kpool_select` | glm5-next |

Deleting either would break the other model.  r2's qwen4exp-hybrid replacement had already removed
the spliced kpool declarations.

### Item 4 — derived cache vs in-cache pooled slot (DEFERRED)

Only meaningful under (C): our derived cache is a separate per-indexer-layer `F32 [idx_dim ×
⌈n_kv/r⌉ × n_stream]` tensor + host watermarks; upstream stores the pooled key in the indexer cache
row.  No measurement is warranted while we keep our graph.  Revisit with (C).

## Evidence (gfx1201, 3× R9700)

* `ggml_build_forward_expand(gf, res_hc)` applied and rebuilt (incremental `llama-cli`):
  * dense 4B `-sm tensor` same-seed 20-token greedy: **`1c5d32ac537d`** (unchanged)
  * qwen4exp Flash-Next IQ4_NL 3-GPU `-sm tensor -lm none --reasoning off -n 20`:
    **`359ff4337837`** (unchanged)
* `scripts/validate-set.sh` green: strict 16/16 `git am` on a fresh `a55e952b8` tarball, applied tree
  `c77aeb55c91972257e228adca4cbcaa30d649be5`.

## Delivery record

* Block 14 amended (`a12853e20`); the tip becomes `b6529d088` (tree
  `c77aeb55c91972257e228adca4cbcaa30d649be5`).  Blocks 15 re-based onto it (bodies unchanged).
* `patches/0014-...patch` regenerated; `release.json` tip/tree/hashes updated, **release name kept
  `v16-a55e952b8-r3`** (no tag; this ships in r4).
* `patches/README.md` block-14 amendment note + the `0014` table row; `WORKLOG.md` 2026-10-05
  (qwen4exp-qsa-convergence).
