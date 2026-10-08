# Open WIP campaigns (index)

This is the index `AGENTS.md` refers to.  `wip/` holds **active exploration only** — nothing here is part
of the delivery, and nothing here may be applied to the fork without the maintainer's explicit go-ahead
(see the WIP and promotion rules in `AGENTS.md`).  Each campaign is a self-contained handover under its own
`README.md`.

| directory | what | status |
|---|---|---|
| [`fit-slab-accounting/`](fit-slab-accounting/README.md) | bring the MoE arena budget and slab headroom into `--fit` (G1+G2) | **PARKED** — Phase 1 attempted; auto floor under `-sm tensor` corrupts ([`PHASE1-ATTEMPT.md`](fit-slab-accounting/PHASE1-ATTEMPT.md)) |
| [`host-expert-dio-cache/`](host-expert-dio-cache/README.md) | drop the pageable `--host-experts mmap` host master and replace it with a bounded, pinned, DIO-filled host tier (issue #116) | **OPEN / Phase 0** |
| [`fp8-support/`](fp8-support/README.md) | native FP8 E4M3 for RDNA4 | PARKED |
| [`host-memory-footprint/`](host-memory-footprint/README.md) | host-memory footprint of GPU-resident weights (gfx1100) | open |
| [`mmvq-verify-rows/`](mmvq-verify-rows/README.md) | faster multi-token mmvq on RDNA4 (bit-exact) | open |
| [`moe-cpu-overlap/`](moe-cpu-overlap/README.md) | genuine CPU/GPU overlap for the expert misses (Strata shape) | OPEN / scoping |
| [`nwarps/`](nwarps/README.md) | per-M `nwarps` MoE candidate — the one deliberate width-purity impurity | ACTIVE (env-OFF) |
| [`strata-amd-kernels/`](strata-amd-kernels/README.md) | compare Strata's AMD decode kernels against block-10/13/15 | OPEN / scoping |

**Closed campaigns** live in `archive/work/`; each left a redirect stub here so historical pointers resolve.
Recently closed:

* [`archive/work/moe-cache-band16/`](../archive/work/moe-cache-band16/README.md) - PR #115's expert-cache band widening (the decode/verify band now follows the routed-expert MMVQ band, 16 tokens on RDNA4, via a single-owner `moe_cache_band` hook) plus the device-side admission fill-list fix, **folded into block 06 in r31** behind `GGML_MOE_CACHE_MAX_TOK=8`.  On qwen35moe `-ncmoe 40`: the 16-token MoE batch 123.7 -> 702.5 t/s, `draft-mtp n-max 12` 33.9 -> 110.7 t/s, `n_max <= 7` byte-identical and `n-max 8` cache text bit-identical to all-VRAM.

* [`archive/work/gfx12-gdn-accuracy/`](../archive/work/gfx12-gdn-accuracy/README.md) - the gfx1201 (RDNA4) bf16 chunked-GDN accuracy-parity campaign.  **Refuted:** the gfx12 kernel has per-op parity (equal op NMSE, identical layer-0 real-data error, marginally tighter WMMA), is tiling-invariant, and an fp16-operand variant that is 65x more accurate per op still moves the model KLD only 1.5x.  The residual gfx1201-vs-gfx11 KLD is model-level numerical sensitivity, not a GDN defect.  Only artifact: a **test coverage fix folded into block 02** (cache-fusion bf16 gate + the model's exact op shape), pending the next release; no kernel change.

* [`archive/work/gdn-bf16-audit/`](../archive/work/gdn-bf16-audit/README.md) - issue #113's BF16 chunked-GDN prefill KLD: the r29 divergence was an `A_sc` stride aliasing bug for `n_seqs > 1` (not bf16 precision).  **Fixed in r30** (block 02), the bf16 default is back on, and the op test's gates are now realistic with a 1e-4 bf16 tolerance.
* [`archive/work/rdna4-qwen4exp-decode-fusions/`](../archive/work/rdna4-qwen4exp-decode-fusions/README.md) - PR #114's four bit-identical qwen4exp decode fusions (latency-scheduled BF16 `hc_mix` up/collapse, `HC_COMBINE` folded into that norm, the shared-expert `sigmoid`-`mul`-`add` gate, and the GDN `beta` sigmoid), **folded into block 15 in r27** behind four default-on `=0` switches (`GGML_HC_UP_V2`, `GGML_CUDA_FUSE_HC_COMBINE_MIX`, `GGML_CUDA_FUSE_SIGMOID_MUL_ADD`, `GGML_CUDA_FUSE_GDN_BETA_SIGMOID`); +2.4 % decode with byte-identical text.
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
