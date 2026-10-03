# `wip/` — active exploration and session handovers

**Nothing in this tree is part of the delivery.** It is never applied to `patches/`, the fork, or a
llama.cpp checkout unless the maintainer explicitly asks for a specific item (see the WIP rule in
`AGENTS.md`). Each directory is a self-contained handover so a follow-up session can pick it up
without re-deriving context.

## Re-base follow-ups (created 2026-10-05, after `v16-a55e952b8-r1`)

These came out of the 2026-10-05 re-base onto upstream master `a55e952b8`. Ordered by value:

| directory | what | type |
|---|---|---|
| [`lightning-indexer-fusion/`](lightning-indexer-fusion/README.md) | take upstream's good qwen4exp indexer ideas into our RDNA fused implementation: PLE `llama_prefetch_rows`, `LLM_FUSED_OP_LIGHTNING_INDEXER` registration, per-head score accumulation in the fallback chain, seed-free mask, and (large) feeding the standard FA kernels an index list | integration |
| [`qwen4exp-qsa-convergence/`](qwen4exp-qsa-convergence/README.md) | two live pooling stacks (our `set_input_qsa`/derived cache vs upstream's `set_input_kpool`); audit missing upstream fixes; resolve the `-sm tensor` gate; decide converge vs delete dead code | decision + cleanup |
| [`rebase-merge-hygiene/`](rebase-merge-hygiene/README.md) | distribute the block-15 build fixes back into blocks 01/08/13/14 so every block bisects/builds; the final tree is correct, the intermediates are not | delivery quality |
| [`shared-expert-fusion-reconcile/`](shared-expert-fusion-reconcile/README.md) | upstream `bed0a8566` MMVQ shared-expert fusion vs block-13 `shexp_down_gate` vs `LLAMA_HC_BLK16` MWR merge: precedence, overlap, bit-identity for the verify band | validation |
| [`rebase-integration-audit/`](rebase-integration-audit/README.md) | paths that compiled but were not exercised: LF/DFlash device path + `extract_layer_inputs`, `common_sampler_clone` S2/rng, `n_rs_batch` in the glm5-next ctor, wide-row argsort tie-break, `test-recurrent-state-depth` | validation |
| [`mmq-prec-gate-fp4/`](mmq-prec-gate-fp4/README.md) | `ggml_prec prec_src1` × `has_gate` merged in the same MMQ template slot; hardcoded Q8 in the fused-gate dispatch; W4A4/FP4 consistency | validation |

## Open experiments (older)

| directory | what | status |
|---|---|---|
| [`nwarps/`](nwarps/README.md) | per-M `nwarps` MoE candidate — the one deliberate width-purity impurity | ACTIVE (2026-09-21) |
| [`mmvq-verify-rows/`](mmvq-verify-rows/README.md) | faster multi-token mmvq on RDNA4 (bit-exact, +106/−32) | open |
| [`host-memory-footprint/`](host-memory-footprint/README.md) | host-memory footprint of GPU-resident weights (gfx1100) | open (2026-10-02) |
| [`fp8-support/`](fp8-support/README.md) | native FP8 E4M3 for RDNA4 | PARKED (2026-09-26) |
| `moe-mmq-overread/` | resolved MoE MMQ tail over-read (`HANDOVER.md`, `RESOLUTION.md`) | closed (kept as record) |

## Promotion (the way out of `wip/`)

A validated win gets an environment kill-switch so it can be A/B tested and bisected, the *combination*
is re-validated (individual validations do not carry over), and only then is a delivery block amended
with the maintainer's go-ahead. Anything applicable to unadulterated upstream gets a copy under
[`../upstream/`](../upstream/).
