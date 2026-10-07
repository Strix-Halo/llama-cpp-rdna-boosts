# Open WIP campaigns (index)

This is the index `AGENTS.md` refers to.  `wip/` holds **active exploration only** — nothing here is part
of the delivery, and nothing here may be applied to the fork without the maintainer's explicit go-ahead
(see the WIP and promotion rules in `AGENTS.md`).  Each campaign is a self-contained handover under its own
`README.md`.

| directory | what | status |
|---|---|---|
| [`fit-slab-accounting/`](fit-slab-accounting/README.md) | bring the MoE arena budget and slab headroom into `--fit` (G1+G2) | **PARKED** — Phase 1 attempted; auto floor under `-sm tensor` corrupts ([`PHASE1-ATTEMPT.md`](fit-slab-accounting/PHASE1-ATTEMPT.md)) |
| [`fp8-support/`](fp8-support/README.md) | native FP8 E4M3 for RDNA4 | PARKED |
| [`host-memory-footprint/`](host-memory-footprint/README.md) | host-memory footprint of GPU-resident weights (gfx1100) | open |
| [`mmvq-verify-rows/`](mmvq-verify-rows/README.md) | faster multi-token mmvq on RDNA4 (bit-exact) | open |
| [`moe-cpu-overlap/`](moe-cpu-overlap/README.md) | genuine CPU/GPU overlap for the expert misses (Strata shape) | OPEN / scoping |
| [`nwarps/`](nwarps/README.md) | per-M `nwarps` MoE candidate — the one deliberate width-purity impurity | ACTIVE (env-OFF) |
| [`strata-amd-kernels/`](strata-amd-kernels/README.md) | compare Strata's AMD decode kernels against block-10/13/15 | OPEN / scoping |

**Closed campaigns** live in `archive/work/`; each left a redirect stub here so historical pointers resolve.
Recently closed:

* [`archive/work/r26-rdna4-dpp-butterflies/`](../archive/work/r26-rdna4-dpp-butterflies/README.md) - PR #110's DPP wave32 warp butterflies, **folded into block 15 in r26** behind the build-time `-DGGML_HIP_NO_DPP_XOR` gate (a runtime device-side gate wedged the FA prefill, so it was dropped).
* [`archive/work/r26-dflash-dev-default-on/`](../archive/work/r26-dflash-dev-default-on/README.md) - PR #107's DFlash F1 warn-and-fall-back + default-on, **folded into block 15 in r26** (TODO #30 closed).
* [`archive/work/moe-cache-autosize/`](../archive/work/moe-cache-autosize/README.md) - the expert-cache /
  arena campaign (arm + auto-size, the movable-boundary slab).
* [`archive/work/expert-cache-split/`](../archive/work/expert-cache-split/README.md) — closed with its
  "mirrored experts" premise **refuted** (the weights are already split per device).
* [`archive/work/host-pinned-buffer-crash/`](../archive/work/host-pinned-buffer-crash/README.md) — the
  `--load-mode none` host-expert page fault; no longer reproduces (14/14 clean on r24).  Only the stale
  `common/common.cpp` warning removal remains (`TODO.md` #45).
* [`archive/work/layer-split-host-experts/`](../archive/work/layer-split-host-experts/README.md) —
  per-device host bufts so `-sm layer` spreads host experts over the GPUs (**delivered in r14**, block 06).
* [`archive/work/moe-mmq-overread/`](../archive/work/moe-mmq-overread/README.md) — the MoE MMQ expert-table
  over-read (**resolved 2026-10-03**; the host gather's prefill "win" was the corruption).

The resolution pattern (investigate -> fold into the owning block -> regenerate + validate -> record +
archive -> ship) is documented in `archive/work/lightning-indexer-fusion/README.md`; promotion rules are in
`AGENTS.md`.
