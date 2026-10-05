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
**`qwen4exp-qsa-convergence/` was resolved on `main` after r3 (an unreleased block-14 amendment,
shipping in r4) and archived there (`DECISION.md`).**
**`lightning-indexer-fusion/` was resolved in `v16-a55e952b8-r4` (the registration was adopted;
the prefetch was dropped as a measured regression) and archived under `../archive/work/`
(`RESULTS.md`).**  That closes the six r1 re-base follow-up WIPs.  The one deferred sub-item of that
series - item 5 of the archived handover (run QSA on the standard FA kernels) - now has its own
campaign handover: [`qsa-standard-fa/`](qsa-standard-fa/README.md).

The **resolution pattern** used for all six (investigate → fold the win into the owning block →
regenerate + validate → record + archive → ship) is documented in the archived handover
[`../archive/work/lightning-indexer-fusion/README.md`](../archive/work/lightning-indexer-fusion/README.md).

| directory | what | type |
|---|---|---|
| ~~`qsa-standard-fa/`~~ | **resolved 2026-10-05** — ported upstream's `n_kv_max` sparse mechanism to HIP (Option A, adopted as a non-destructive capability); **dropped** Option B/C because the fused `FLASH_ATTN_QSA` is 1.7-2.6× faster at the qwen4exp geometry and the standard path's compaction prepass is O(n_kv); see `../archive/work/qsa-standard-fa/RESULTS.md` | archived |
| ~~`lightning-indexer-fusion/`~~ | **resolved r4** — adopted the `LLM_FUSED_OP_LIGHTNING_INDEXER` registration; dropped `llama_prefetch_rows` (measured pp512 regression); deferred the standard-FA index-list campaign; see `../archive/work/lightning-indexer-fusion/RESULTS.md` | archived |
| ~~`qwen4exp-qsa-convergence/`~~ | **resolved after r3** — decision (A): keep our fused QSA graph, adopt upstream's `hc_init` split fix, defer kpool convergence; see `../archive/work/qwen4exp-qsa-convergence/DECISION.md` | archived |
| ~~`shared-expert-fusion-reconcile/`~~ | **resolved r3** — the three shared-expert arms are disjoint (upstream's arm is dormant under `-sm tensor`); see `../archive/work/shared-expert-fusion-reconcile/RESULTS.md` | archived |
| ~~`mmq-prec-gate-fp4/`~~ | **resolved r3** — `prec_src1` threaded + asserted in the fused-gate MMQ; pair FP4 exclusion asserted; see `../archive/work/mmq-prec-gate-fp4/RESOLUTION.md` | archived |
| ~~`rebase-merge-hygiene/`~~ | **resolved r2** — every block builds; see `../archive/work/rebase-merge-hygiene/RESOLUTION.md` | archived |
| ~~`rebase-integration-audit/`~~ | **resolved r2** — argsort tie-break finding + fixes; see `../archive/work/rebase-integration-audit/RESULTS.md` | archived |

## Strata-derived campaigns (created 2026-10-05)

From re-reading `~/Strata` (current at 2026-10-04) after the maintainer noted it has moved on since our
2026-09-27 read. The trigger was a report of 50-60 t/s with MTP on a single 32 GB card with a 3-bit
Qwen3.8-Flash-Next; the same card here measures **52.1 t/s** once `MOE_EXPERT_CACHE_MIB` is armed
(34.2 t/s unset). The three campaigns below cover the parts that are still genuinely theirs.

| directory | what | status |
|---|---|---|
| [`moe-cache-autosize/`](moe-cache-autosize/README.md) | **the core focus**: arm `MOE_EXPERT_CACHE_MIB` automatically when expert weights are host-resident (`-ncmoe`/`-cmoe`) and derive the size from free VRAM (reserve + floor + multi-GPU + `--fit` + draft), so the user never sizes it by hand | OPEN (2026-10-05) |
| [`moe-cpu-overlap/`](moe-cpu-overlap/README.md) | genuine CPU/GPU overlap for the expert misses (Strata's custom pipeline); our CPU-computes-misses arm was correct but lost to scheduler serialization above a ~1.2 GiB arena | OPEN / scoping (2026-10-05) |
| [`host-pinned-buffer-crash/`](host-pinned-buffer-crash/README.md) | **blocker found in the auto-size M0 sweep**: 2-GPU IQ4_NL + `-sm tensor` + host experts + `--load-mode none` gives a GPU page fault (pinned `ROCm_Host` request 92.6 GiB > 80 GiB `RLIMIT_MEMLOCK`; the host-buft allocator silently falls back to a pageable buffer under the host name). Default `--load-mode auto`/`mmap` is stable | OPEN (2026-10-05) |
| [`strata-amd-kernels/`](strata-amd-kernels/README.md) | compare Strata's AMD decode kernels (#262 packed-byte IQ dequant, `router_top10`, `fused_gr` carveout, arena THP, #646 IQ-grid staging) against our block-10/13/15 kernels; port the ones that win | OPEN / scoping (2026-10-05) |

## Open experiments (older)

| directory | what | status |
|---|---|---|
| ~~[`sched-moe-restage/`](sched-moe-restage/README.md)~~ | **resolved 2026-10-05, promoted in `v16-a55e952b8-r5`** - the block-06 scheduler re-stage fix for a second `MUL_MAT_ID` consumer (PR #96); see `../archive/work/sched-moe-restage/RESULTS.md` | archived |
| ~~[`qsa-standard-fa/`](qsa-standard-fa/README.md)~~ | **resolved 2026-10-05** — HIP sparse port adopted as a non-destructive capability; Option B/C dropped (fused `FLASH_ATTN_QSA` is 1.7-2.6× faster, compaction prepass is O(n_kv)); see `../archive/work/qsa-standard-fa/RESULTS.md` | archived |
| ~~[`ar-non-rdna4/`](../archive/work/issue-86/README.md)~~ | **PROMOTED in `v16-a55e952b8-r9`, resolved 2026-10-04** - the internal/hybrid HIP all-reduce is no longer RDNA4-only: default-on non-RDNA4 (`GGML_CUDA_AR_ALLOW_NON_RDNA4=0` opts out) + first-call NCCL-to-internal failover (issue #86); see `../archive/work/issue-86/` | archived |
| ~~(issue #99, no `wip/` tree)~~ | **RESOLVED in `v16-a55e952b8-r9`** - the gemma4 `-sm tensor` gate is relaxed for all-resident / `-ngl`-offloaded loads; only host-resident experts (`-ncmoe`/`-cmoe`) and the MTP head (`gemma4-assistant`) are rejected cleanly (issue #99); see `../archive/work/issue-99/` | resolved |
| ~~[`issue-93-ring/`](../archive/work/issue-93-ring/README.md)~~ | **PROMOTED in `v16-a55e952b8-r7`** - auto-size the op-offload H2D staging ring budget + table-size-scaled gate + graceful per-split fallback (issue #93; the fixed 2048 MiB default silently disabled staging for 450 MiB expert tables). +51 % `pp8192 @ -ub 8192` on gfx1201 x4, byte-identical output, gather untouched; see `../archive/work/issue-93-ring/` | promoted |
| ~~[`issue-97-vram/`](../archive/work/issue-97/README.md)~~ | **PROMOTED in `v16-a55e952b8-r8`, resolved 2026-10-04** - the one-off H2D staging bandwidth calibration (512 MiB `cudaMalloc` probe) no longer runs for a split with no host-resident weight (issue #97, Windows VRAM not returned after `cudaFree`). Reporter confirmed on Windows (16.4 -> 43.2 t/s, byte-identical output); the directory keeps the r36/r37 backport patch. See `../archive/work/issue-97/README.md` | archived |
| ~~[`mtp-draft-op-offload/`](../archive/work/mtp-draft-op-offload/README.md)~~ | **PROMOTED in `v16-a55e952b8-r10` (block 01)** - PR #98 (`LLAMA_MTP_DRAFT_OP_OFFLOAD=0`, keep the MTP draft context's host ops on the host). The first review's "no gain" was a measurement error (the expert cache was not armed). With `MOE_EXPERT_CACHE_MIB=4096` the switch frees **610 MiB** here (draft ROCm2 compute 2054 -> 1444 MiB) and **~1.1 GB** on the reporter's box; opt-in, no default change. Prefill A/B over 3 full prefills shows no win (549 vs 541 t/s, noise), so the benefit is memory-only. See `../archive/work/mtp-draft-op-offload/` | archived |
| ~~[`rdna4-verify-fusions-lf/`](../archive/work/rdna4-verify-fusions-lf/README.md)~~ | **PROMOTED in `v16-a55e952b8-r11` (block 15), resolved 2026-10-04** - PR #102 superseded #100, addressing all review points (check removed, per-context GLU mark, `GGML_CUDA_FUSE_*` names, RDNA4 gates) and adding the `vf_sweep` geometry sweep.  Bit-identical across on == off == stock r10 over 16,130 cases, all three fusions confirmed firing, and an alternating A/B measured +1.31..+1.56 % with byte-identical output.  See `../archive/work/rdna4-verify-fusions-lf/VERIFICATION-r10.md` | archived |
| ~~[`2gpu-sched-fixes/`](../archive/work/2gpu-sched-fixes/VERIFICATION.md)~~ | **PROMOTED in `v16-a55e952b8-r12` (blocks 06 + 13), resolved 2026-10-05** - issue #103 / PR #104: order a cross-device split-input copy after the destination's queued work (block 06) and guard the MoE expert-cache alias lookup by device and layer (block 13). The reporter's stock repro does not fire on x16 links, so a scratch prefill-rebalance harness was used to get the FAIL -> PASS (`!!!!` -> correct) on 2 x R9700; patch 0001's page fault was not reproduced and is retained as a correct guard. See `../archive/work/2gpu-sched-fixes/VERIFICATION.md` | archived |
| [`nwarps/`](nwarps/README.md) | per-M `nwarps` MoE candidate — the one deliberate width-purity impurity | ACTIVE (2026-09-21) |
| [`mmvq-verify-rows/`](mmvq-verify-rows/README.md) | faster multi-token mmvq on RDNA4 (bit-exact, +106/−32) | open |
| [`host-memory-footprint/`](host-memory-footprint/README.md) | host-memory footprint of GPU-resident weights (gfx1100) | open (2026-10-02) |
| ~~`nonuniform-slot-alloc/`~~ | **closed NEGATIVE, archived 2026-10-06** — a non-uniform per-layer slot budget buys ≤1 % in-sample and ~0 held-out at any arena size worth using (greedy water-filling on the campaign routing profiles); no code, no build. See [`archive/work/nonuniform-slot-alloc/RESULTS.md`](../archive/work/nonuniform-slot-alloc/RESULTS.md) | archived |
| ~~`gather-mode/`~~ | **shelved, archived 2026-10-06** — the §30.6 "device-count-aware gather-mode heuristic" premise is disproven: the `stage_gather` host-vs-compact optimum tracks **physical ubatch width** (2-GPU == 3-GPU; crossover ~6500–7000 t), and a device-count rule would regress ub ≤6144 by 9–33 %. Dropped under the standing "anything gather-based is too risky" caution (this is the *staging* gather, distinct from the corrupt `GGML_SCHED_DEVGATHER`, but the win was +5 % at one width and unverified for purity). See [`archive/work/gather-mode/README.md`](../archive/work/gather-mode/README.md) | archived |
| [`fp8-support/`](fp8-support/README.md) | native FP8 E4M3 for RDNA4 | PARKED (2026-09-26) |
| `moe-mmq-overread/` | resolved MoE MMQ tail over-read (`HANDOVER.md`, `RESOLUTION.md`) | closed (kept as record) |

## Promotion (the way out of `wip/`)

A validated win gets an environment kill-switch so it can be A/B tested and bisected, the *combination*
is re-validated (individual validations do not carry over), and only then is a delivery block amended
with the maintainer's go-ahead. Anything applicable to unadulterated upstream gets a copy under
[`../upstream/`](../upstream/).
