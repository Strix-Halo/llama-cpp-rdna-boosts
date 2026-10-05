# rdna-boosts TODO / follow-up tracker

Cross-project tracker so important state survives context compaction.  **Forward-looking only**: this
file lists what is still open (active work, waiting items, accepted limitations, parked ideas);
closed and retired work lives in `WORKLOG.md` and the dated records it points to.  Details never
live here — they live in `AGENTS.md`, `patches/README.md`, `MANIFESTS.md`, `WORKLOG.md`,
`GREEDY-PURITY.md`, `wip/*` and `benchmarks/`.

**Current state (release `v16-a55e952b8-r10`, 2026-10-04):** the delivery is the **16-patch set**
against fork point **`a55e952b8`**, canonical tip `b86854900`, net tree
**`dab5186bc0527508156507fd323a9109924cb03e`** (`validate-set.sh` green).  See `AGENTS.md` and
`release.json` for the current state and `WORKLOG.md` for the dated records; the release history
before r1 (on the previous base `84e76d8a2`) is in `WORKLOG.md` and `archive/docs/`.  This tracker is
**forward-looking only**; resolved work has moved to `WORKLOG.md`.

## Active (kept compact: only what this repo will work on next)

### 39. `-sm layer` + host experts routes every MoE op to GPU 0 (per-device host bufts)

**Opened 2026-10-05; FIXED in the WIP fork and measured — awaits the maintainer's go-ahead to fold into
`patches/`.**  2 x R9700, IQ4_NL, `-sm layer -ncmoe 48`, cache on: **10.3 -> 55.4 t/s** in a same-session
A/B (patch `wip/layer-split-host-experts/fix.patch`, 4 files, +75/-21).  `-sm layer` now beats `-sm
tensor` for the oversized 2-GPU case (55.4 vs 45.4) and does not hit the `--load-mode none` crash.
The chain: `ggml_backend_cuda_host_buffer_type()` was a device-0 singleton (upstream); the `ctx_key`
comparator merges same-name bufts; and the scheduler's op-offload loop returns the *first* capable
backend.  Fix = per-device host bufts (F1) + layer-device host-buft choice in `create_tensor` (F2) +
comparator device tiebreak (F3) + offload-loop device filter (F4).  Single-GPU is unchanged (48.7 both
before and after; the earlier 52.1 was environmental).  **Next:** decide block placement and run the
full gates (`-ncmoe 0` oracles, `W=1..8` purity, MTP acceptance, the r12 race harness) before folding
in; also the `upstream/` copy (this is generic `-sm layer` + `-ncmoe` multi-GPU).  Record:
`wip/layer-split-host-experts/`.

### 38. Host-resident expert load: GPU page fault under `--load-mode none` (2 GPUs)

**Opened 2026-10-05; blocker for the 2-GPU auto-size target, no code yet.**  2 x R9700,
`-sm tensor` + `-ncmoe >= 32` on Qwen3.8-Flash-Next IQ4_NL (93 GiB), `--load-mode none`: GPU page fault
(`Page not present`), host SIGABRT, **nondeterministic** (memory-state dependent; deterministic under
`AMD_SERIALIZE_KERNEL=3` in a session).  Not the cache (`MOE_EXPERT_CACHE_MIB=0` still fires);
`--load-mode auto` (default) and `--load-mode mmap` are stable, `-sm layer` does not fire.  Evidence
points at the single 92.6 GiB `ROCm_Host` pinned buffer `--load-mode none` creates (`RLIMIT_MEMLOCK` is
80 GiB here) plus `ggml_backend_cuda_host_buffer_type_alloc_buffer`'s silent pageable fallback under
the host-buffer name (`ggml/src/ggml-cuda/ggml-cuda.cu:1782`).  **Next:** N1 capture the failing kernel
(ROCm coredump / rocprof); N2 confirm the RLIMIT link; N3 pick a fix (loader pin budget / loud
fallback / portable pin / clean reject) behind the 2-GPU repro + the `-sm tensor` oracles.  Record:
`wip/host-pinned-buffer-crash/`.

### 37. `MOE_EXPERT_CACHE_MIB` auto-enable + auto-size (the 34 -> 52 t/s hole)

**Opened 2026-10-05; auto mode IMPLEMENTED + measured in the WIP fork (session 2) -- awaits the
maintainer's go-ahead to promote.**  The decode-side MoE expert cache is opt-in and
`MOE_EXPERT_CACHE_MIB` unset means off, so a `-ncmoe` user silently runs the CPU expert path: measured
on one R9700 / gfx1201 with Qwen3.8-Flash-Next UD-IQ3_XXS + MTP, **34.2 t/s unset vs 52.1 t/s at
`MIB=20480`** (details and the full sweep in `wip/moe-cache-autosize/README.md`).  q8_0 KV is fine at
depth (48.1 @32K, 42.7 @128K when the arena is sized around it), so no 4-bit KV is needed.  Direction:
unset == auto (enabled, sized from `free - reserve` in `alloc_all_locked`), `0` == off, `>0` == fixed;
add a floor below which the arena is declined and a reserve that covers the MTP draft staging copies
and prefill compute growth.  **Next:** M0 (peak-VRAM grid -> reserve formula).  Record:
`wip/moe-cache-autosize/`.  (This is the user-facing half of the Strata comparison.)

**Session 2 (2026-10-05) update.**  `unset == auto` is implemented in `~/llama-r13`
(`wip/moe-cache-autosize/auto-mode.patch`, `ggml/src/ggml-cuda/moe-expert-cache.cu` only, +106/-22) and
measured with the block-16 layer fix + `--load-mode auto`: 1G IQ3_XXS auto **48.6** (manual best 48.7);
2G IQ4_NL `-sm layer` auto **57.8** (62.3 at n=256, 77.8 % residency) vs cache-off 31.3; 3G **81.3**;
2G/3G `-sm tensor` auto ~= off (46.0 vs 45.3); `-ncmoe 0` byte-identical to cache-off.  The old 2-GPU
tensor cache regression did not reproduce.  **Blocker for the floor:** a small arena is far *worse* than
the CPU path under MTP (IQ3_XXS crossover ~8192 MiB / ~18 % residency; 2048 MiB = 18.9 vs off 32.0),
and "disable below the floor" does not restore cache-off speed under MTP (9.5 vs 33.1 -- a stale MTP
draft context created while the cache was enabled; byte-identical text, so correctness holds).  Default
floor is therefore 0 (always arm) + a `< 20 %` residency warning; the real fix is an early enable/disable
decision before the draft context exists.  Full tables + next steps in `wip/moe-cache-autosize/README.md`.

**Session 3 (2026-10-05) update.**  The **early floor decision is implemented and validated**: an
optional `moe_cache_preflight` device iface, driven by `llama_model_moe_cache_preflight(model)` from
`common_init_result` after the target context and before the MTP draft context, disables an auto cache
whose projected arena is below `max(MOE_EXPERT_CACHE_MIN_MIB, MOE_EXPERT_CACHE_MIN_RES_PCT% of the host
experts)` (default 18 %).  Below-floor MTP now matches cache-off (**34.7 vs 34.6**; was 9.5), auto stays
48.6 (1 GPU) / 57.8 (2 GPU), explicit `MOE_EXPERT_CACHE_MIB` is skipped, fully-resident is inert.  The
patch is 9 files / +223-22 (`wip/moe-cache-autosize/auto-mode.patch`).  Also measured the `--fit` vs
arena interaction (mild on Flash-Next: `--fit` keeps 33-44 % residency) and wrote up the three
policies; **needs the maintainer's call** on whether `--fit` should reserve an arena floor (recommended),
prefer context (preflight then disables), or a hybrid.  See the README's "Session 3" section.

**Session 3 continued -- policy (c) implemented.**  `common/fit.cpp` now reserves
`floor = max(MOE_EXPERT_CACHE_MIN_MIB, MOE_EXPERT_CACHE_MIN_RES_PCT% x host_expert_bytes)` per device in
`--fit`'s margin (auto + host experts only; explicit `MIB` and fully-resident runs untouched), so
`--fit` sizes the context around the floor and the arena then takes any remaining free VRAM down to
`MOE_EXPERT_CACHE_RESERVE_MIB`.  Host-expert bytes are accumulated in the loader
(`create_tensor` -> `llama_model::moe_host_expert_bytes`, works under `--fit`'s `no_alloc`) and exposed
via `llama_model_moe_host_expert_bytes`.  Two WARNs now state the reserved floor and the actual arena
size/residency (the latter also with `--fit` off).  **Logging (maintainer decision):** the notices stay
at WARN; llama-cli keeps upstream's `LOG_LEVEL_ERROR` default and so does not announce them (as with any
upstream warning), while **llama-server's default shows them** (`-lv 3` reveals them in llama-cli).  No
CLI log-level change.  Patch now 13 files / +295-25.

### 36. Genuine CPU/GPU overlap for the MoE misses (Strata's pipeline shape)

**Opened 2026-10-05; scoping only.**  The delivered cache-on path computes every expert on the GPU
(resident from VRAM, misses via UVA over PCIe); the cache-off path computes them on the CPU but
**serialised** with the GPU splits.  Our own CPU-computes-misses arm was correct and negative above a
~1.2 GiB arena (the loss is ~600 scheduler dispatches + ~3 cross-backend copies per layer/token, not
CPU compute; see `archive/work/moe-expert-cache/WORKLOG.md`).  Strata's custom engine runs the CPU pool
*because* it is not under llama.cpp's serial scheduler.  **Next:** O0 - measure how much of a 52 t/s
token is the UVA cold fraction at 43 % residency; if it is <20 %, park it.  Design space and gates:
`wip/moe-cpu-overlap/README.md`.

### 35. Audit Strata's AMD decode kernels against our block-10/13/15 kernels

**Opened 2026-10-05; scoping only.**  Candidates: #262 packed-byte IQ dequant (Strata measured R9700
decode +15 %), `router_top10` fast FP32 router (62.4 -> 70), `fused_gr` LDS carveout, arena transparent
huge pages, and the #646 IQ-grid-staging choice.  Our decode is partly UVA-bound at 43 % residency, so
the honest end-to-end targets are the fully-resident and multi-GPU cases; the kernels are still in
scope (RDNA4 HIP) and several are upstream-PR shaped.  **Next:** K0 - line-by-line kernel diff and a
shortlist with expected deltas.  Record: `wip/strata-amd-kernels/README.md`.

### 33. PR #104 follow-ups: cross-backend event wait and per-change kill-switches

**Opened 2026-10-05; both landed in `v16-a55e952b8-r12`, non-blocking.**  The block-06 fix adds a
`ggml_backend_event_wait(input_backend, event)` where the event was created on `split_backend`, the
scheduler's first cross-backend event wait.  Several backends implement `event_wait` assuming their
own event object (`ggml_backend_sycl_event_wait` does a `static_cast<sycl::event*>`, Vulkan casts to
`vk_event*`, Metal to `ggml_metal_event_t`); on a scheduler that mixes device backends this is at
best a host block and at worst an abort.  The delivery is CUDA/meta/CPU only today, so it is not hit
now.  **Next:** add a same-family guard (or a per-backend foreign-event capability).  Separately,
neither block-06 nor block-13 change has a per-change env kill-switch (the WIP promotion rule);
`GGML_SCHED_EVENTS=0` disables the block-06 fix only by turning off all per-split events, at a
measured prefill cost.  Record: `archive/work/2gpu-sched-fixes/VERIFICATION.md`.

### 34. Patch 0001 (MoE expert-cache alias guard) still needs its own FAIL -> PASS

**Opened 2026-10-05.**  Block 13's `alias_find_checked` (PR #104 patch 0001) was not exercised on
this box: no `moe_cache_tally_kernel` page fault and no stale-alias warning fired, even with the
scratch prefill-rebalance harness and `MOE_EXPERT_CACHE_MIB=6144` (the harness reproduced the
block-06 race instead).  The guard is correct by construction (a table whose arena lives on another
device must never be used by the calling device's op), so it shipped with the fix.  **Next:** have
the reporter rerun the `MOE_EXPERT_CACHE_MIB=6144` case against a tree that also has patch 0002, so
the alias check is exercised without the race masking it.  Record:
`archive/work/2gpu-sched-fixes/VERIFICATION.md`.

### 32. `gdn-conv.cu` device idiom (block 15; found in the PR #102 review)

**Opened 2026-10-04; cosmetic, no correctness impact.**  `gdn_conv_check` gates the 2..255-token arm
on `ggml_cuda_info().devices[ggml_cuda_get_device()].cc`, while the batched-copy path added by PR #102
uses `ggml_cuda_info().devices[ctx.device].cc`.  They agree on every current graph because the device
is set before `ggml_cuda_try_fuse` runs, so it is two idioms for the same thing.  **Next:** pick one
(pass the context device into the check, or use `ggml_cuda_get_device()` consistently) at the next
block-15 touch.  Record: `archive/work/rdna4-verify-fusions-lf/VERIFICATION-r10.md` (review notes).

### 31. PR #100: three verify-step fusions (GLU -> Q8_1, GDN VCONV, batched CPY)

**Opened 2026-10-04; accepted as a WIP record, NOT promoted.**  Contributor PR
[#100](https://github.com/stew675/llama-cpp-rdna-boosts/pull/100) (@overdoingism) adds three
default-on verify-step fusions, each with a kill switch, in `archive/work/rdna4-verify-fusions-lf/`:
`GGML_CUDA_FUSE_GLU_Q8_1` (a verify-step GLU also writes the next mmvq's Q8_1), `GGML_CUDA_FUSE_GDN_CONV_VERIFY`
(the fused GDN concat + conv for 2..255-token batches) and `GGML_CUDA_FUSE_CPY_BATCH` (consecutive
same-layout f32 CPY nodes in one launch).  (The #100 revision used the `GGML_LF_*` names.)

Review on r10 (test worktree `/home/stew675/llama-pr100`): applies cleanly and builds warning-free;
output is byte-identical on 35B-A3B Q8_0 MTP n3 (`8d733a56c740`) and 27B Q4_K_XL MTP n3
(`1acb04bd9104`, the recorded 27B gate) for fusions on == off == r10 baseline, with and without
verify graphs; the batched-copy fusion fires and runs (150/528 runs).  Measured throughput is ~flat
(27B 53.6 vs 52.9 t/s, 35B 143.2 vs 141.5 on vs off), so the win is unproven.

**Before promotion:** (1) the `GGML_LF_GLU_Q8_CHECK=1` self-check is broken: it does a host sync
under graph capture (`operation not permitted when stream is capturing`) and, with verify graphs
disabled, perturbs the output (22079 vs 113 chars), so it is not a valid oracle; (2) the GLU fusion
mark is file-static cross-node state passed from `ggml_cuda_try_fuse` to the launcher (fragile,
single-threaded); (3) the `lf-*` / "Llama-Frankenstein" naming and AI-authored comments need
normalising; (4) the PR body has no measurements; (5) VCONV relaxes a kernel precondition
(`T >= 256` -> `T >= 2`) that needs a geometry sweep; (6) no arch gate despite the RDNA4 title;
(7) the overlap with the existing fusion machinery needs review as a whole.  The author has been
offered the chance to refine the PR first; otherwise it is a lower-priority item to pick up later.
Record: `archive/work/rdna4-verify-fusions-lf/`.

**Follow-up:** contributor [PR #102](https://github.com/stew675/llama-cpp-rdna-boosts/pull/102)
(opened 2026-10-04) supersedes the #100 patch with a revised `verify-fusions.patch` built on r10,
plus a geometry sweep (`vf_sweep.cpp`) and a rewritten README.  It removes the broken `_CHECK`
self-check, moves the GLU mark into the CUDA context, normalises the naming and adds RDNA4 gates.
**Verified 2026-10-04** against r10 (worktree `/home/stew675/llama-fix97`, branch `pr102-review`,
build `build-pr102`): applies cleanly (tree `38ebce2f`), builds warning-free, and the 16,130-case
`vf_sweep` is bit-identical across fusions on, fusions off and stock r10, with all three fusions
confirmed firing (rocprof counts match the PR).  The end-to-end DFlash gate is byte-identical
on == off == stock r10 (`55824e640aa0`).  All #100 review points are addressed.

**A/B 2026-10-04 (the effect size is now settled):** Qwen3.8-27B-UD-Q4_K_XL + DFlash2, q8_0 KV,
`-ub 512`, `-n 512`, WikiText-2 prefixes at 8.3K / 35.2K / 110.4K tokens, alternating on/off runs on
the same binary.  Every on run beat every off run: **+1.51 % (5/5)**, **+1.56 % (5/5)**,
**+1.31 % (3/3)**; generated text byte-identical within each depth.  So the win is real and larger
than the earlier single-run spread suggested.  Under the default-on policy the patch is
promotion-ready (block 15 is the natural home); promotion itself is the maintainer's call.
Record: `archive/work/rdna4-verify-fusions-lf/VERIFICATION-r10.md`.

### 30. Default-flip the DFlash device-resident layer features (PR #73)

**Opened 2026-09-30** with r27.  PR #73's device path (`GGML_LF_DFLASH_DEV=1`) is now validated on
gfx1201 as output-identical to the host path and faster (`Qwen3.8-27B-DFlash2-Q4_K_M`, 27B
UD-Q4_K_XL, q8_0 KV, 5246-token prompt: `GGML_LF_DFLASH_DEV=0` and `=1` both give
`487 chars sha=dad22c4270ab`; prefill 1144.2 -> 1232.5 t/s, generation 63.8 -> 65.4 t/s).  It stays
opt-in because the device buffer allocation (`llama_context::extract_layer_inputs`) is a hard
`GGML_ASSERT` on failure: a nearly-full card aborts instead of falling back.  **Next:** give the
allocation the same warn-once-and-fall-back treatment as the FA staging arena (issue #33), then flip
the default on with `GGML_LF_DFLASH_DEV=0` as the kill switch.  Record: `WORKLOG.md` 2026-09-30 (r27).

**Maintainer note 2026-10-04:** DFlash2 was compared against the delivery's adaptive MTP across the
broad matrix and adaptive MTP came out ahead, so this is a **support** item (get the DFlash2 path
working correctly, without the hard assert), not a performance priority.  The device-resident default
flip is therefore lower value than the original r27 note implies; keep it for completeness.

### 29. The address-selected `ROPE -> VIEW -> SET_ROWS` fusion decides the W=1 decode logits (issue #58 item D)

**Opened 2026-09-29** while analysing issue #58 item D (cross-start greedy nondeterminism, @DanoPTT,
gfx1201 / Windows / ROCm 10).  `ggml_cuda_check_fusion_memory_ranges()` selects the fusion by buffer-address
overlap; the fused path is not proven bit-transparent, so the decode logits are a function of the allocation
plan.  Logits-level bisect on gfx1201 / ROCm 7.14 (`Qwen3.8-27B-Q6_K`, `test-logits-width-probe`, P=256,
W=1 row-0): default `f6d62323d9339541`; `GGML_CUDA_DISABLE_FUSION=1`, all-address-gated-fusions-off, and
`GGML_CUDA_DISABLE_ROPE_SET_ROWS=1` all give `3ab223a4f08afd6e`; `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1`,
the other per-fusion switches, graphs and `FA_KV_NATIVE` do not move it.  20/20 fresh Linux starts agree
(the layout is stable here).  Not yet separated: whether the fused `rope_multi` kernel itself differs, or
its elided F32 buffer re-addresses a neighbouring fusion.  **Next:** isolate the kernel (a focused
`rope -> view -> set_rows` backend test or a forced-fire probe), then either make the fused path bit-exact
or stop the guard deciding by raw address.  Switches shipped default-off in `v16-84e76d8a2-r24` (block 15).
Record: `WORKLOG.md` 2026-09-29 (r24), `GREEDY-PURITY.md` §41.

### 28. Retune the RDNA4 dense mmvq rows-per-block table for `q5_1` / `iq4_xs` on ROCm 7.14

**Opened 2026-09-28** while integrating PR #57 (now in `v16-84e76d8a2-r21`, blocks 10+13).  The per-type
`calc_rows_per_block_weight` table was swept by the author on **ROCm 10.0**; on the maintainer's **ROCm
7.14** build it transfers for the main types (q3_K +7..+24 %, q4_K +8..+38 %, q6_K +3..+27 %, q2_K +14 %
at n=1) but carries two small dips: **`q5_1` -3..-4 %** and **`iq4_xs` -3..-4 %** at n=2/3 -- both
**+10..+12 %** at n=8, and the common MTP/DFlash verify widths are 4 and 8.  Kept as tuned (the aggregate
band is a large net win); a per-type retune (e.g. 1 row for these two at 2..3 columns) needs its own
verify-width A/B.  Record: `wip/mmvq-verify-rows/VERIFICATION-r21.md`, `WORKLOG.md` 2026-09-28 (r21,
PR #57).

### 27. Extend the RDNA4 GQA-6 FA band's 64-wide K/V row to the 2-byte (f16/bf16) arm

**Opened 2026-09-28** while integrating PR #62 (now in `v16-84e76d8a2-r21`, block 15).  The new band-only
MMA config row (`fattn-mma-f16.cuh`: `is_rdna4 && ncols2 == 8 && DKQ == DV == 256 && ncols == 32`) matches
only the **native-quantized** arm (`ncols1 = 4`).  The 2-byte f16/bf16 arm uses `ncols1 = 2`
(`ggml_cuda_fattn_band_wmma_ncols1`) -> `ncols = 16`, so it keeps the old 128-half2 row.  The PR README's
"f16 3-4 %" is therefore **misattributed** (measured f16 KV `tg128 @ d16384` 27.50 -> 27.57, noise).  A
64-wide row for the 2-byte arm is worth a sweep: the arm has no dequantisation to hide the unused
columns, and r6 made native bf16 default-on so the arm is live.  Record:
`wip/rdna4-fa-band/VERIFICATION-r21.md`, `WORKLOG.md` 2026-09-28 (r21, PR #62).

## Waiting on others (not actionable in this repo)

### 6. Cross-arch / gfx1100 validation (the gfx1201 port + its Phase 2.5 probe are DONE — see `WORKLOG.md`)
- **Still open, needs other hardware:**
  * gfx1100 (`fingon`, 24 GiB): the §4.2 remainder with *no* gfx1100 data yet — the GDN gfx11 NW16 scan
    retune (~106K VGPR/CU vs a possible 64K classic), `split_j`/config rows, the quantize chunk,
    routed-compact, the hc/PLE fusions, and the two block-13 MTP regression fixes under RDNA3
    (acceptance gate).
  * gfx1100 q8_0 native-arm prefill trade (2026-09-18, r5): the block-15 q8_0 native arm costs ~5 %
    of gemma-4-26B-A4B head-512 deep prefill on gfx1100 (773 vs 813 t/s `pp2048 @ d98304` with
    `GGML_CUDA_FA_KV_NATIVE=0`, stock 810) but buys +44 % decode at d65536, so it stays on.  The
    dense head-256 model is unaffected (1405.9 vs 1404.4).  Candidate fix: keep native decode but
    restore node-scratch F16 staging for prefill on RDNA3_0.  See `WORKLOG.md` 2026-09-18 (r5).
  * gfx1151 (`halo`): Phase 3's cross-arch fingerprint check (gfx1201 == gfx1151 numerics) — a
    verification goal, not a port; also item 7's MTP crossover re-measure (item 4 is closed).
- **Tracker hygiene:** the plan's own open checkboxes are **stale** (Phase 1 is complete and the doc
  predates qwen4exp's promotion to block 14); read the banner at the top of
  `archive/work/qwen4exp/gfx1201-porting.md` before trusting them.

### 8. Dual 7900XTX (gfx1100, community): block-12 validation
- Hybrid HIP all-reduce on RDNA3 **pairs** is being validated by a community member on their dual-7900XTX
  box (hybrid-dispatch matrix internal/nccl/none + the bounded-spin path at depth-16384).  The block-12
  arch gate stays RDNA4-only until then.  Volunteer env: `GGML_CUDA_ALLREDUCE=internal`.
- The block-13 gfx1100 leg is DONE (single-GPU 7900 XTX; see the `WORKLOG.md` entry); what remains here is block 12, which
  is N/A on a single-GPU box.  Where: `patches/0012` + the block-12 notes in `patches/README.md`.

### 12. Upstream: file the staged PR candidates
- `upstream/README.md` — five are written up and evidence-verified on pristine master `9cf3bf256`:
  the ggml-alloc unused-view release, the sched probe, the keys-only indexer cache (A1), the `attn_k`
  null-mask guard (A2), plus `UPSTREAM-PR-fa-decode-verify-kernel-family.{md,patch}` (the F1 chooser fix,
  whose NVIDIA/Ada half the fork deliberately does not land — see the AGENTS.md scope policy).
- Filing is the maintainer's call.

## Documented, deliberately NOT fixed (accepted limitations — do not re-report)


- **The `launch-ledger` remainder (item 5(c)) — measured, not pursued (2026-09-12 (8)).**  The small-pp
  remainder (+38 `scale_f32`/eval, an `rms_norm<256,true>` count diff) is sub-0.2 %, root-cause-only.
- **The mmq mma `sum[]` accumulator-overflow latent defect (item 5(d)) — accepted, no upstream report
  (2026-09-12 (8)).**  `process_tile` sizes the per-thread accumulator as `J*I/(nwarps*32)` while the
  AMD-WMMA vec_dot indexes up to `J/2-1`, so any config with `I < nwarps*16` silently corrupts
  (deterministic for J=128, racy for J=48/24).  Root cause + full evidence matrix:
  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-mmq-j128-latent-defect.md`.  Upstream
  ships **no** violating config (every `mmq-config-*.cuh` row keeps `I >= nwarps*16`) and the block-13
  rows never violate it either, so there is no upstream reproducer to file.
- **V3 prefill cost is arch-dependent (item 5(g)) — accepted.**  gfx1151 measured −3.2 % at pp20480
  (4B, q8_0) vs the RDNA4 reference −1.3 %, decode flat; still a large net win (−799 MiB compute +
  −799 MiB host) and on by default.
- **Upstream monitor: ROCm unaligned-width split-load (item 13) — standing, no action (2026-09-12 (8)).**
  Fixed locally in block 13; no PR planned (upstream is busy with its own qwen4exp work).  It resolves
  naturally as a re-base conflict if upstream fixes it; nothing to track.

- **Mixed K/V cache types fall off the GPU attention path.**  Any mixed pair (`bf16`+`q8_0`, `f16`+`q8_0`)
  gives `graph splits = 18`, a ~1.5 GiB host compute buffer and pp2048 7924 → 640–1049 t/s on the 4B.
  Maintainer policy (2026-09-11): **reject differing K/V types** — every mixed pair is 1.7–3.6× slower
  and never smaller; upstream already enforces same-K/V for DeepSeek V4 (#25871).  **Decided and
  implemented 2026-09-11 (12): hard-rejected at context creation** — `params.type_k != params.type_v`
  now fails `llama_init_from_model` with a message naming both types and telling the user to set
  `--cache-type-v` to match (block-14 amendment; upstream's MLA/DeepSeek4-only condition is dropped).
  Both types default to f16, so only an explicit `--cache-type-k`/`-v` can trigger it.
- **gemma-4-E4B-it + 3-GPU `-sm tensor`** aborts in the meta splitter (`ggml-backend-meta.cpp:1177`)
  because its 2 KV heads are fewer than the 3 devices (one device gets a zero-extent share).  Works on
  1/2 GPUs and on 3 GPUs with `-sm layer`; maintainer's call: no fix.  Every other model is unaffected
  (a future block or upstream report could make the splitter tolerate a zero-extent share).
- **`src/llama-kv-cache.h:274` `-Wunused-private-field` for `v_enabled`** on a full build (the field *is*
  used, in `llama-kv-cache.cpp:232`; clang's per-TU analysis fires).  A `[[maybe_unused]]` one-liner
  silences it; left alone to keep the V5 amendment scoped to the FA kernels.
- **Not worth pursuing** (measured, no win): the decode fq-inline-quantize port (wash-to-negative — the
  tree already launches fewer kernels/step and sits at wall parity) and the GDN +72-launch 2-kernel split
  (cosmetic).

## Parked (not planned now)
- ~~**`--fit` for `-sm tensor` (raised 2026-09-18, issue #38 investigation).**~~  **DONE 2026-09-21, promoted
  as the block-06 r12 amendment** from `beta/tensor-fit-fix/` (now `archive/work/tensor-fit-fix/`).  The
  work existed as a beta patch all along; it was re-validated against r11 (the fit now has to size for the
  reachable packed kq mask), promoted into **block 06** (the delivery's general system-operations bucket,
  since the change is dependency-free), and the campaign record archived.  `--fit` is no longer a no-op
  under tensor split: per-device targets, proportional or honoured `-ts`, then auto-`n_ctx` and an `-ngl`
  binary search, with an explicit `-c` never overridden.  Gates: the default fit cases reproduce the
  2026-09-18 record exactly (auto-ctx 27B: 59899 -> `n_ctx 43264`), the `-ngl`-reduction cases are more
  conservative than the beta (the fit sizes for the mask), seven end-to-end loads generate with zero
  out-of-memory and zero compute-buffer growth (dense/MoE, 2 and 3 GPU, embedded and separate/adaptive
  MTP), and the same-seed gate is byte-identical.  See `WORKLOG.md` 2026-09-21 (r12) and the block-06 (r12)
  amendment in `patches/README.md`.  **Still worth an `upstream/UPSTREAM-PR-*` candidate** - the change is
  generic llama.cpp; only the placement (block 15) is delivery-specific.
- ~~**The `fattn-mma-f16` instance-set build cost (raised 2026-09-15, r5).**~~  **PARTLY DONE in r6
  (2026-09-18).**  Candidate (a) is delivered: `generate_cu_files.py` now emits one MMA TU per
  `(ncols1, ncols2, head size)` and the head-512 instances are listed **first** in the backend source
  order (the order is the actual fix — clean `ggml-hip -j16` **323.4 -> 236.0 s, -27 %**), and the tile
  instances are split per `(head size, KV type)`.  Build-time only: identical instantiations and
  linked-library symbols.  **Candidate (b) was tried and REJECTED (2026-09-18):** a runtime KV-type
  dispatch made the build *slower* (236 -> 304 s), and `__noinline__` on the loader cut it to 136 s
  but cost a universal 1.5-2.5 % prefill (f16 KV included), because the optimiser's cross-inlining of
  the force-inlined native loaders is what makes them fast.  The loaders stay force-inlined and the
  build-speed answer is **ccache** (the script's wiped rebuild went 282 -> 4.2 s; the script enables
  it when `ccache` is on PATH).  Evidence, the TU-timing tool and the reproduction recipe:
  `archive/work/build-time-regression/`; the r6 records are in `patches/README.md` and `WORKLOG.md`
  (2026-09-18 (r6) and (build process)).
- **`rdna-boosts-all.patch` hygiene (raised 2026-09-15).**  The single-file net patch is a documented
  delivery artifact (1.35 MiB) that is regenerated on every release, so each revision adds ~1.3 MiB of
  history — the dominant `.git` cost (the raw logs trimmed 2026-09-15 compressed to only ~1.07 MiB total,
  so history is otherwise compact).  Options when someone picks this up: (a) keep as-is (it is derivable
  from `patches/` + `scripts/apply-all.sh`, so it is pure convenience); (b) stop tracking it and generate
  it on demand in the release pipeline / for the GitHub Release asset (the layout table and
  `docker-ghcr.yml` both reference it, so those pointers move); (c) a history rewrite
  (`git filter-repo` + force-push + re-tagging every `v16-*`) — measure the real recovery first: the
  patch is already close to incompressible text, so the win is bounded and the cost is a force-push to a
  published repo (see the Pushing policy).  **Not today; no work started.**
- **Restore the block-13 RDNA3_5 single-token fusion perf (item 16).**  The ~0.9 % `tg128` the purity
  skip costs; the proposed "pin `nwarps`/`rps`/item-split" fix is **invalid** (the two arms are already
  launch-identical).  Live candidates: codegen (`has_fusion` register pressure / FMA contraction) and the
  Q8_1 cache.  Low priority — the skip is the accepted purity trade.  `archive/work/strix-halo/rdna35-mmvq-fusion-purity/README.md` §9.
- **`-Wshadow` cleanup for `src/` (item 15).**  Audited: 128 warnings / 27 files, 46 in the risky
  "shadows a local variable" class.  Wants a dedicated cleanup commit (~20 upstream files) to avoid
  colliding on every re-base.  `archive/work/shadow-warnings/RECORD-2026-09-12-shadow-audit.md`.
- **MoE topk fusion adoption (item 5(a)).**  ~0.5 % prefill for a ~188 MB arena cost and a numerics fork
  (fused top1 logit 18.424 -> 18.690 vs the unfused reference), so it needs a quality gate before it can
  be trusted.  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-launch-overhead-topk.md`.
- **`ssm_alpha` + `ssm_beta` single-walk fusion (item 5(b)).**  ~0.3-0.6 % prefill, blocked by the graph
  expansion order (the two MMs are non-adjacent) -> needs a graph restructure or load-time stacked
  weights.  `archive/work/wip-archive/qwen4exp/discovery/2026-09-06-strix-halo-gfx1151-cijk-dense-gemm.md`.

### LFRU host→GPU slow hot-weight migration
- Survivor of the expert-tiering experiment (dropped 2026-09-05 — most of its aims are already covered by
  current llama.cpp options).  The one idea left: an LFRU-style very slow migration of hot weights from
  host to GPU (persistent GPU slot cache + CPU-computed cold tail).  Design notes:
  `archive/work/qwen4exp/LRU_EXPERTS.md`, `PHASE0_ROUTING.md`, `HANDOVER-2026-09-04-tiering.md`.

## Where the current lists live

- Remaining gfx1151 work + the 2026-09-12 TODO audit: `archive/work/strix-halo/HANDOVER-2026-09-12-remaining-gfx1151.md`.
- QSA sparse-regime width purity (items 4/7 disposition): `archive/work/strix-halo/RECORD-2026-09-12-qsa-sparse-width.md`.
- Environment, instruments, reference hashes and the landing procedure for KV/FA work:
  `archive/work/kv-quant-purity-followups/HANDOVER-2026-09-11-remaining-work.md` (its §0 status and its items 1/5
  and F3 are **superseded** — see `WORKLOG.md`).
- qwen4exp carried-forward open items: `beta/qwen4exp/README.md` ("Open items (carried forward from WIP)").
- Delivery verification contract + dated records: `MANIFESTS.md`, `patches/README.md`, `WORKLOG.md`,
  `AGENTS.md` headers.
- Purity/invariant analysis and the instrument rules: `GREEDY-PURITY.md`.
- Benchmarks + gates: `benchmarks/` (the adaptive-MTP baseline gate: `mtp-adaptive-methodology.md`).
- Memory campaign (wins, V3/V4 plans, Block 15 record — now promoted to `patches/0015`): `archive/work/block-15-campaign-wins/HANDOVER.md`,
  `README.md`, `BETA-TESTING.md`; upstream PR candidates: `upstream/README.md`.

## Parked: finish the meta segmented expert upload (host-resident experts under `-sm tensor`)

`llm_arch_supports_sm_tensor()` no longer rejects `LLM_ARCH_GEMMA4` outright (r9, issue #99): an
all-resident or `-ngl`-offloaded gemma4 splits under `-sm tensor`, and only a host-resident expert
table (`-ncmoe`/`-cmoe`, detected in `llama_model_create`) and the MTP head (`gemma4-assistant`) are
rejected.  The remaining work is the segmented per-ubatch upload for a host-resident expert table:
gemma4's fused expert tensor (`ffn_gate_up_exps`) has a segmented split layout (`n_segments`/`nr`),
and `ggml_backend_meta_set_tensor_async` only handles a contiguous slice, so the MMQ tail pad that
`copy_experts` appends makes the range non-row-aligned.

A partial fix exists but is **not** landed (it hung non-deterministically, one GPU spinning): port
`ggml_backend_meta_buffer_set_tensor`'s segmented handling into the async setter, copy the
row-aligned part through the segments and the <=512-byte pad as a flat per-device range, and use
per-row 1-D async sets instead of the 2-D pageable `hipMemcpy2DAsync` (which faults on this shape).
Give it a deterministic repro (e.g. `compute-sanitizer`, or the `GGML_CUDA_SPLICE_GATHER` /
`GGML_META_PINHOST` force-modes) before retrying.  All of it is in
`archive/work/moe-expert-cache/item3-findings-session20.md`; the issue-#99 gate change is in
`archive/work/issue-99/README.md`.
