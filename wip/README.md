# `wip/` — active exploration and session handovers

**Nothing in this tree is part of the delivery.** It is never applied to `patches/`, the fork, or a
llama.cpp checkout unless the maintainer explicitly asks for a specific item (see the WIP rule in
`AGENTS.md`). Each directory is a self-contained handover so a follow-up session can pick it up
without re-deriving context.

## Re-base follow-ups (created 2026-10-05, after `v16-a55e952b8-r1`)

These came out of the 2026-10-05 re-base onto upstream master `a55e952b8`. Ordered by value.
**`rebase-merge-hygiene/` and `rebase-integration-audit/` were resolved in `v16-a55e952b8-r2` and
archived under [`../archive/work/`](../archive/work/) (`RESOLUTION.md` / `RESULTS.md`).**
**`shared-expert-fusion-reconcile/` and `mmq-prec-gate-fp4/` were resolved in `v16-a55e952b8-r3`
and archived under [`../archive/work/`](../archive/work/) (`RESULTS.md` / `RESOLUTION.md`).**

| directory | what | type |
|---|---|---|
| [`lightning-indexer-fusion/`](lightning-indexer-fusion/README.md) | take upstream's good qwen4exp indexer ideas into our RDNA fused implementation: PLE `llama_prefetch_rows`, `LLM_FUSED_OP_LIGHTNING_INDEXER` registration, per-head score accumulation in the fallback chain, seed-free mask, and (large) feeding the standard FA kernels an index list | integration |
| [`qwen4exp-qsa-convergence/`](qwen4exp-qsa-convergence/README.md) | two live pooling stacks (our `set_input_qsa`/derived cache vs upstream's `set_input_kpool`); audit missing upstream fixes; resolve the `-sm tensor` gate; decide converge vs delete dead code | decision + cleanup |
| ~~`shared-expert-fusion-reconcile/`~~ | **resolved r3** — the three shared-expert arms are disjoint (upstream's arm is dormant under `-sm tensor`); see `../archive/work/shared-expert-fusion-reconcile/RESULTS.md` | archived |
| ~~`mmq-prec-gate-fp4/`~~ | **resolved r3** — `prec_src1` threaded + asserted in the fused-gate MMQ; pair FP4 exclusion asserted; see `../archive/work/mmq-prec-gate-fp4/RESOLUTION.md` | archived |
| ~~`rebase-merge-hygiene/`~~ | **resolved r2** — every block builds; see `../archive/work/rebase-merge-hygiene/RESOLUTION.md` | archived |
| ~~`rebase-integration-audit/`~~ | **resolved r2** — argsort tie-break finding + fixes; see `../archive/work/rebase-integration-audit/RESULTS.md` | archived |

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
