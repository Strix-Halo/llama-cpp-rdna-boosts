# MoE expert cache — README (current state + fresh-session handover)

`wip/` campaign.  Decode-side MoE expert caching for models whose experts do **not** fit in VRAM: keep
a hot subset resident on the device and serve the rest cheaply, so `-ncmoe` decode approaches the
full-residency `-ncmoe 0` throughput as the cache grows — without giving up `-sm tensor`.

**This file is the live handover.**  The verbose, dated records and the original design briefs are in
**`WORKLOG.md`**; the "Completed work" index at the bottom points at them.  Do not edit the WORKLOG
records in place — append a new dated entry and add a one-liner to the index.

> Status: **not part of the delivery.**  Everything here is `wip/`, applies only to the campaign
> worktree `~/llama-decode`, and is default-OFF (`MOE_EXPERT_CACHE_MIB` unset) until it passes a
> promotion gate.  Nothing in the delivery `patches/` is touched by this campaign.

---

## 0. Status — all campaign items CLOSED

> **Status (2026-10-02, session 21).**  The campaign is **rebased onto delivery r28** (`60361cb9f`) —
> no conflicts; the net campaign diff is unchanged.  Branch **`wip-moe-devmap-r28`**, tip
> **`f5a79e6ab`**, patches **`exp23`** (current, clean-applies to r28).  B1, item 1, B3 and the
> `DEVMAP` default remain **DONE** (§4 and `WORKLOG.md`); every campaign default is ON
> (`MOE_EXPERT_CACHE_MIB` arms the cache; `DEVMAP` / `DEVPOLICY` / `KSLOT` ride along).
>
> **Validated on gfx1100 (`fingon`, RX 7900 XTX, 24 GiB, session 21e):** a 29.3 GB Q6_K 35B-A3B that
> cannot fit the card runs at **32.8 -> 80.3 t/s** tg128 (warm ~94, cache h=0.9312) with `-ncmoe 99` +
> `MOE_EXPERT_CACHE_MIB=14000`, and Q4_K_M `-ncmoe 0` == cache-on (`359ff4337837`).
>
> **The item-3 adaptive staging-vs-gather gate is CLOSED as a no-op on r28** (`GGML_SCHED_STAGE_AUTO`
> kept as a dormant opt-in).  The first attempt's "impossible throughput" was root-caused: a staging pass
> leaves the whole expert table resident in the per-device staging ring, so a gather sampled right after
> it measures ~40 % fast.  The probe now samples gather-first (warm) then staging, synchronizes each
> probe pass, and has a 5 % hysteresis toward staging.  But on r28 staging wins or ties above the width
> gate on every measured case (Q4_K_M ub8192 5283 vs 4035; ub2048 1988.6 vs 1995.4; IQ4 ub8192 1414 vs
> 1233; ub2048 841.3 vs 845.5) and the gather already runs below it, so the fixed policy is optimal and
> the probe would always latch staging.  See §0.B, `item3-findings-session20.md` ("Session 21"/"21d")
> and `WORKLOG.md`.
>
> Everything here is `wip/`: it applies only to the campaign worktree `~/llama-decode`, **never** to the
> delivery `patches/`, and `~/llama-decode` is never pushed.  Delivery-repo docs may be pushed to
> `origin/main`.

### Work items (all closed)

**(A) The user graph-input copies (item 3's original premise) — CLOSED, NEGATIVE (session 21c).**  The
`GGML_TENSOR_FLAG_INPUT` branch is negligible in every measured case: gemma4 26B-A4B `-sm layer`
`-ub 8192` **0.8 ms** total, Q4_K_M `-sm tensor` `-ub 8192` 230 ms and `-ub 2048` 514 ms — all ≤7 % of
the input loop, which is dominated by `stage_input` (94 %).  The gemma4 `-sm layer` cost that the item
was blamed for is the normal end-of-pass synchronize waiting for the prefill compute (`SCHEDSYNC`: 2
large + 4 tiny calls per pass), not the user input.  The archived
`archive/work/closing-the-gap/2026-09-24-gfx1151-input-copy-cost.md` had already measured the gfx1151
per-ubatch host-input copy at **0.03–0.05 % prefill / 0.5–0.9 % decode** and the maintainer decided
**do not port the input ring**.  No fix warranted; the real `-sm tensor` input-loop cost is the
whole-shard staging gather (item B).

**(B) The adaptive staging-vs-gather gate — CLOSED as a no-op on r28 (session 21d).**  The gate only
decides above `sched_stage_min_tokens` (the H2D-bandwidth-calibrated width gate, ~1536 tokens at
14.5 GB/s); below it staging is disabled and the gather runs unconditionally.  Above the gate, staging
wins or ties everywhere measured on r28 (`-sm tensor -p 8192`, warm `-r 3`): Q4_K_M ub8192 5283 vs
4035 (+31 %), ub2048 1988.6 vs 1995.4 (tie), ub1024 gate-forced-open 1212.0 vs 1207.5 (tie); IQ4 ub8192
1414 vs 1233 (slight staging), ub2048 841.3 vs 845.5 (tie).  The r26 qwen4exp gather wins (+47 %/+48 %)
no longer reproduce — the r28 staging/compute work erased the deficit (IQ4 ub2048 staging 571 -> 841) —
so the probe would latch **staging** in every active case and the fixed width-gated policy (gather below
the gate, staging above) is already optimal.  `GGML_SCHED_STAGE_AUTO` is kept as a **dormant opt-in** A/B
knob (the implementation — gather-first ordering, per-pass synchronize, 5 % hysteresis — stays in
`ggml-backend.cpp`), alongside `GGML_SCHED_GATHER_FIRST`.  See `item3-findings-session20.md`
"Session 21d".

**(C) The cache-enabled-but-unserviceable transparency bug — FIXED (2026-10-02, session 21b).**  If
`MOE_EXPERT_CACHE_MIB` is set but there are **no host-resident routed expert weights** (so the cache can
never take an input over), `ggml_cuda_cache_blocks_fusion()` stood the cache-band fusions down because
`moe_cache_enabled() && !moe_cache_has_arena()`, which changed the arithmetic vs the cache-less run.  The
fix adds `moe_cache_has_tables()`: an **empty** cache now behaves exactly as if disabled (allow the
fusions); only a cache that has a routed expert table it might take over but cannot serve (the r8
item-23 garbage case) stands them down.  Before: `-ncmoe 0` + `MIB=8192` = `15038c19ddc8` vs `-ncmoe 0` =
`de8be4d0c90c`.  After: identical on gfx1201 (2-GPU tensor `de8be4d0c90c`, 1-GPU layer `15038c19ddc8`)
and gfx1151 (`15038c19ddc8`), and `-ncmoe 99 MIB=8192`/`MIB=1` still match the oracle.  Patch `exp23`.

**Read `item3-findings-session20.md` first — it is the complete, self-contained record.**  Summary:

Whole-shard H2D staging (`ggml_backend_meta_stage_input`) uploads the device's entire expert shard every
ubatch through a single-threaded host memcpy gather; the B2 device gather moves only the routed experts
through a GPU kernel.  Neither wins universally, and **no static signal separates them** (the 122B and
IQ4 move the *same* shard bytes per pass with opposite winners, and the host-staging share of the staged
pass is *higher* for the staging winners, so it misleads).  Four models, clean single-variable A/Bs
(`GGML_SCHED_GATHER_FIRST=1` defers **only** the routed expert tables to the gather):

| model | arch | active | ub | staging | gather | winner |
|---|---|---|---|---|---|---|
| Q4_K_M 35B-A3B | qwen35moe | 3B | 8192 | **1848** | 1493 | staging |
| 122B-A10B | qwen35moe | 10B | 8192 | **675** | 616 | staging |
| IQ4 Flash-Next | qwen4exp | 6B | 8192 | 564 | **830** | gather |
| IQ4 Flash-Next | qwen4exp | 6B | 2048 | 571 | **843** | gather |
| IQ4 Flash-Next | qwen4exp | 6B | 512 | 156 | **316** | gather |
| Q4_K_M 35B-A3B | qwen35moe | 3B | 512 | 477 | **496** | gather (marginal) |

⇒ **A runtime probe is required.**  The design is `GGML_SCHED_STAGE_AUTO=1`: time pass 0 with staging,
pass 1 with the gather, then latch the faster arm (both arms are bit-identical, so the choice is
performance-only).  A first implementation **was reverted** — it latched correctly but produced an
impossible throughput (Q4_K_M **5989** t/s vs the known 1843), i.e. the expert upload was being
**skipped entirely** — a correctness failure, not a perf one.  The known-good tree is `51b1f48be`
(it reproduces Q4_K_M 1847 / IQ4 556).

**Start here:**
1. The env arm `GGML_SCHED_GATHER_FIRST=1` is **correct**; the probe's first latched arm was not.  The
   root cause is **staging-ring residency**, not the gather path — a gather sampled right after a
   staging pass reads the table the ring still holds.  The probe therefore samples **gather first**.
2. The probe forces a `synchronize` on each probe pass (`ggml_backend_sched_graph_compute_async`), so
   the GPU gather cost is measured; a plain `compute_splits` return only launches.
3. The latch has a 5 % hysteresis toward **staging** (the delivery default): the arms' measurements
   carry ~10 % noise and on r28 they are often near-tied.
4. A routed expert table is handled by exactly one of stage / gather / take-over / copy on every pass;
   the `ggml_backend_meta_stage_input` abort-on-`stage_gather`-failure fix (session 20) is what makes a
   silent stale-slot read impossible.  Run the byte-identity gates before trusting any perf number.

### Parked: gemma4 `-sm tensor` support

`llm_arch_supports_sm_tensor()` now rejects `LLM_ARCH_GEMMA4` (clean error at load: `not implemented for
architecture 'gemma4'`; arch-level, so it covers `-ncmoe`, `--fit` and a low `-ngl`).  gemma4's fused
expert tensor has a **segmented** split layout and the host-resident-MoE per-ubatch upload has no correct
path (the async setter is contiguous-only, and `copy_experts`' MMQ tail pad breaks row alignment).  A
partial fix (segmented port + tail walk + per-row 1-D copies instead of the faulting 2-D pageable blit)
still hung non-deterministically and is reverted.  **Follow-up** (also in the delivery `TODO.md`): finish
the segmented async upload with a deterministic repro (`compute-sanitizer`, or the
`GGML_CUDA_SPLICE_GATHER` / `GGML_META_PINHOST` force-modes) and re-enable `-sm tensor`.  `-sm layer`
works (pp8192 758.7).  Detail in `item3-findings-session20.md`.

### Gates (unchanged — see "The gates every change must pass" below)

Byte-identity `15038c19ddc8` (1-GPU `-sm layer`), `de8be4d0c90c` (2-GPU `-sm tensor`), `d7bef4c6fdc3`
(3-GPU `-sm tensor`); width purity `none == n1 == n3 == n7`; `MUL_MAT_ID` OK; MTP n3 acceptance
> ~0.45; deep coherence; and the `-sm tensor` gather gate (`pp2048 -ub 512` ~724 t/s, do not regress).

### Files touched in session 20 (for diff review)

`ggml/src/ggml-backend.cpp` — input-loop diagnostics (`GGML_SCHED_SYNCDBG`, `GGML_SCHED_INPUTDBG`) and
the `GGML_SCHED_GATHER_FIRST` experiment gate; `ggml/src/ggml-backend-meta.cpp` — **only** the safe
`stage_gather` abort fix; `ggml/src/ggml-cuda/common.cuh` — `GGML_META_SCRATCH_MB`;
`src/llama-arch.cpp` — the gemma4 `-sm tensor` exclusion.

### Code map

| concern | location |
|---|---|
| staging ring issue/drain | `ggml/src/ggml-backend.cpp`: `sched_stage_issue`, `sched_input_gatherable`, `sched_stage_ev`, the input-loop host-weight branch (`copy_experts`, the devgather call) |
| gather kernel + slot maps | `ggml/src/ggml-cuda/moe-expert-cache.cu`: `moe_cache_gather_kernel`, `moe_cache_gather_host`, `moe_cache_take_over` / `moe_cache_get_table` / `slot_dev` |
| meta slice geometry (item 1, done) | `ggml/src/ggml-backend-meta.cpp`: `ggml_backend_meta_moe_cache_gather` (re-enabled) — the same offset formula as `ggml_backend_meta_moe_cache_update`; the CUDA side pads the MMQ tail in `moe_cache_gather_kernel` |
| cache iface / adapters | `ggml/src/ggml-backend-impl.h`, `ggml-cuda.cu` (`ggml_backend_cuda_moe_cache_*`), `ggml-backend-meta.cpp` |
| MoE ids read (item 2) | `ggml/src/ggml-cuda/mmvq.cu` (`mul_mat_vec_q_moe*`), the fused gate+up and down-fold call sites in `ggml-cuda.cu` |
| user graph-input copy (item 3) | `ggml/src/ggml-backend.cpp`: the `GGML_TENSOR_FLAG_INPUT` branch of `ggml_backend_sched_compute_splits` |

### Quick start (copy/paste)

```sh
cd ~/llama-decode && git switch wip-moe-devmap-r28 && git log -1   # expect f5a79e6ab (r28 base + campaign + adaptive probe + item-C fix)
cmake --build build-rocm --target llama-cli llama-bench -j 16

IQ4=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
IQ3=/llm/models/Qwen3.8/Flash-Next/IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
MTP=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib

# B1 prefill (now ~1044/1627/1761 t/s at ub 512/2048/8192), 1 GPU:
MOE_EXPERT_CACHE_MIB=24576 MOE_EXPERT_CACHE_DEVMAP=1 ./build-rocm/bin/llama-bench \
  -m $IQ4 -ngl 99 -ncmoe 48 -sm layer -fa 1 -lzm auto -lm none -t 8 -p 8192 -n 0 -b 8192 -ub 2048 -r 3 -o jsonl

# item 1 (fixed): 3-device tensor-split gather must equal the gather-OFF text on a >8-token prompt
P=$(cat ~/llama-cpp-rdna-boosts/prompts/reasoning.txt)
for G in 0 1; do HIP_VISIBLE_DEVICES=0,1,2 GGML_SCHED_DEVGATHER=$G \
  ./build-rocm/bin/llama-cli -m $IQ4 -ngl 99 -ncmoe 48 -sm tensor -fa 1 -lzm auto -lm none -t 8 -c 4096 \
  --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt -p "$P" -n 8 > /tmp/g$G.out 2>/dev/null; \
  python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/g$G.out; done   # both hashes equal

# 3-device transparency oracle (IQ3_XXS fits -ncmoe 0): cache-on == cache-off -ncmoe 0
```

### Traps (do not re-derive)

* The **tensor-split gather is ENABLED** (`.moe_cache_gather` on the Meta iface; `GGML_SCHED_DEVGATHER=0` opts out).
  The **real** qwen4exp corruption was the missing MMQ expert-table **tail pad**, not the slice geometry — the
  pruned gather must fill the next expert's first `min(expert_bytes,512)` bytes (`GGML_META_GATHER_NOPAD=1`
  restores the bug).  Do not "fix" the `off += nb[axis+1]` accumulation; it is correct.
* The gather is **not** the bottleneck for bytes — it runs at ~link speed; the gap is volume + residency.
* `llama-cli` needs `--single-turn --no-display-prompt`; never `-v` for the hash runs; pin `-t 8`; never
  run parallel benches.
* Cache-on vs cache-off at `-ncmoe` differ because cache-off runs the MoE on the **CPU** and cache-on on
  the **GPU** — the transparency oracle is `-ncmoe 0`, not `-ncmoe 48/99`.
* The campaign patch base is now **r26** (`0d58404e1`); `exp17` is r25-based and stale.

### Status recap (what is done)

* **B1** prefill residency + upload overlap — **DONE**; the scheduler half is **delivery r26**.
* **B2** device gather — **DONE**: unsplit AND tensor-split arms default ON (item 1 fixed the tensor-split
  corruption — the missing MMQ expert-table tail pad — and re-enabled the Meta delegate; `+18.9 %`
  `pp2048 -ub 512`).
* **B4** qwen4exp / Qwen3.8-Flash-Next — **complete** (see `b4-single-gpu-iq4nl.md`).
* **B3** in-kernel slot lookup — **DONE (session 19, `exp20`)**: default-ON whenever `DEVMAP=1`; 0 remap launches; byte-identical 1/2/3 GPU + width purity + MTP + coherence; ~+1.3 % Q4_K_M.
* **`DEVMAP` default flipped ON (session 19b, `exp21`)** — the promotion gates were re-run green; default vs eager `MIB=9216` **+9.4 %**; `DEVMAP=0` opts out.
* **B5** closed; **B6** prompt-routing seed done.
* Current patch sequence: `exp17` (r25) -> `exp18` (r26 rebase) -> `exp19` (session 18: item 1) -> `exp20` (session 19: B3) -> **`exp21`** (session 19b: DEVMAP default ON).

---

## 1. Where things are

| | |
|---|---|
| **Worktree** | `~/llama-decode`, branch **`wip-moe-devmap-r28`**, tip **`f5a79e6ab`** (= **r28** (`60361cb9f`) + the campaign commits through session 21b / the adaptive probe + item-C fix).  The pre-rebase branch `wip-moe-devmap-v2` (tip `51b1f48be`, = **r26** (`0d58404e1`) + the campaign through session 20) is kept for reference.  Prior tips: `2376ac6cf` (session 18 / item 1, backed up as `backup/wip-moe-devmap-v2-r26-item1`), `6ca5c1c77` (session 17 / B1, r26 base; backed up as `backup/wip-moe-devmap-v2-r26-b1`), `6140bba76` (r25-based; backed up as **`backup/wip-moe-devmap-v2-r25`**), `c83899985` (session 15), `ceea0cfb6` (session 11), `6d3e26e0d` (session 10); the pre-rebase r21 tips are backed up as `backup/wip-moe-devmap-v2-r21` (`a1d0fa985`) and `backup/wip-moe-expert-cache-r21` (`c7dd40a23`).  The eager path without devmap is branch `wip-moe-expert-cache` (`7e6c4cf66`); both build the same `build-rocm`. |
| **Build** | `cd ~/llama-decode && cmake --build build-rocm --target llama-cli llama-bench -j 16` (~1-2 min incremental with ccache).  Full rebuild: `BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714` (~7 min cold). |
| **Full patch** | **`exp23-moe-expert-cache-r28-itemC-fusion-guard.patch`** (`git diff 60361cb9f..wip-moe-devmap-r28`, clean-applies to **r28** `60361cb9f`) = the r28 rebase + the adaptive `GGML_SCHED_STAGE_AUTO` probe + the item-C `moe_cache_has_tables()` fusion-guard fix.  **`exp22-moe-expert-cache-r28-adaptive-stage.patch`** = the session-21 snapshot (rebase + probe, before item C).  **`exp19-moe-expert-cache-r26-b2-tensorpad.patch`** (`git diff 0d58404e1..wip-moe-devmap-v2`, clean-applies to **r26** `0d58404e1`) = `exp18` + item 1 (the Meta gather re-enable, the tensor-split default, and the MMQ expert-table tail pad in `moe_cache_gather_kernel`).  **`exp18-moe-expert-cache-r26-b1-rebase.patch`** = the session-17 tip (B1 only).  **`exp17-moe-expert-cache-r25-b1-prefill-gather.patch`** (r25 base) is **stale/superseded** — kept for history.  `exp16-moe-expert-cache-r25-b4-gather-off.patch` = the session-16 tip; `exp15-moe-expert-cache-r25-b2-devgather.patch` = the session-15 tip (gather default on); `exp14-moe-expert-cache-r25-b6-prefill-seed.patch` = the session-14 tip;  `exp13-moe-expert-cache-r25-devpolicy.patch` = the session-13 tip;  `exp12-moe-expert-cache-r25.patch` = the session-11 tip on r25 (`ceea0cfb6`); `exp11-moe-expert-cache-devmap-pipelined.patch` = the session-10 tip on r21 (`a1d0fa985`); `exp10` = the item-3 tip (`56f015057`); `exp9` = the items-1+2 tip (`c7dd40a23`); `exp8` = the session-7 cold-workaround snapshot; `exp7`/`exp6` older. |
| **Parked branch** | **`wip-moe-devmap-v2`** (tip `c5bbb7ee2`, the live branch); `wip-moe-expert-cache` (`7e6c4cf66`, the eager path); `wip-moe-devmap` (`6b8a7ed06`, the BROKEN first cut).  Pre-rebase SHAs are in the `backup/*-r21` refs. |
| **Iteration model** | `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21 GiB, fits 1 card; the fast smoke model). |
| **End-goal model** | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` (100 GiB `qwen4exp`, 48x512 experts) — Phase 4; its lazy/PLE path makes `llama-bench` absolutes non-comparable.  **Transparency oracle:** `/llm/models/Qwen3.8/Flash-Next/IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf` (77 GiB, fits `-ncmoe 0`; `-ncmoe 0` == cache-on `-ncmoe 99` == `77c6f546460d`).  Shared MTP head: `.../IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`. |
| **Hardware** | 3x R9700 (gfx1201); use `HIP_VISIBLE_DEVICES=0[,1]`.  Pin `-t 8` (the GPU IRQs live on the top cores). |
| **Delivery** | `~/llama-cpp-rdna-boosts` `main`; the campaign README/WORKLOG/patches live in `wip/moe-expert-cache/`.  The `~/llama-decode` checkout is **never pushed**. |
| **Drop-off baseline** | `decode-arena-sweep.md` — the warm decode `tg` vs arena-size sweep (depth 0 + depth 16384); the session-10 pipelined devmap already flattened it (see its Postscript 2).  Session 12 added the **device-side admission policy** (a depth-0 win at every measured arena) and the `_PROGRESS` log that shows the cache is still warming long past the bench length; the depth-16384 sweep and the promotion gates remain. |

### How to run (throughput / purity / MTP / coherence)

```sh
MQ4=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

# throughput (per-rep, warm = rep >= 2; rep 1 pays the one-time cold fill)
HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=10240 ./build-rocm/bin/llama-bench \
  -m $MQ4 -ncmoe 99 -ngl 99 -fa 1 -sm tensor -t 8 -p 0 -n 1024 -r 6 -o jsonl

# NEW (item 3b) devmap path: add MOE_EXPERT_CACHE_DEVMAP=1.  The sweep is MIB=1024/2048/3072/4096/6144/8192/9216
# plus the identity endpoint MIB=9344 (slots == 256).
HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=8192 MOE_EXPERT_CACHE_DEVMAP=1 ./build-rocm/bin/llama-bench \
  -m $MQ4 -ncmoe 99 -ngl 99 -fa 1 -sm tensor -t 8 -p 0 -n 1024 -r 4 -o jsonl

# promotion breakdown + expert traffic (use -v; grep the exit report)
HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=9216 MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_TIMING=1 \
  ./build-rocm/bin/llama-bench -v -m $MQ4 -ncmoe 99 -ngl 99 -fa 1 -sm tensor -t 8 -p 0 -n 512 -r 2 -o jsonl \
  2>&1 | grep moe_cache_report

# machinery A/B at h=1 (same arena): identity vs forced devmap (one run each)
#   MOE_EXPERT_CACHE_MIB=9344 MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_FORCE_DEVMAP=0   # identity 94.0
#   MOE_EXPERT_CACHE_MIB=9344 MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_FORCE_DEVMAP=1   # devmap   84.9

# NEW (session 12) device-side admission policy + progress log (needs DEVMAP=1).  -v is required for
# the WARN progress lines.  Compare DEVPOLICY 0 vs 1 at FORCE_DEVMAP=1 MIB=9344 (machinery, h=1) and at
# a small MIB (fill-heavy).
HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=9344 MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_FORCE_DEVMAP=1 \
  MOE_EXPERT_CACHE_DEVPOLICY=1 MOE_EXPERT_CACHE_PROGRESS=1024 \
  ./build-rocm/bin/llama-bench -v -m $MQ4 -ncmoe 99 -ngl 99 -fa 1 -sm tensor -t 8 -p 0 -n 1024 -r 4 -o jsonl \
  2>&1 | grep moe_cache_progress

# kernel-count ground truth (the 7.14 /usr/bin/rocprofv3 hangs on this workload; the ROCm 10 one works):
#   MOE_EXPERT_CACHE_MIB=9344 MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_FORCE_DEVMAP=1 \
#   /opt/rocm-10.0.0-gfx120X/bin/rocprofv3 --kernel-trace -o /tmp/prof -- ./build-rocm/bin/llama-bench ... -n 32 -r 1
#   then: sqlite3 /tmp/prof_results.db  -> rocpd_kernel_dispatch joined to rocpd_info_kernel_symbol
#   expects `moe_cache_build_remap_kernel` = exactly 240 * decode_tokens dispatches @ ~1.57 us.

# byte-identity oracle (fusions ON): full-table GPU, `-ncmoe 0`
HIP_VISIBLE_DEVICES=0,1 ./build-rocm/bin/llama-cli -m $MQ4 -ngl 99 -ncmoe 0 -fa 1 -sm tensor \
  -t 8 -c 8192 --seed 42 --temp 0 --reasoning off --ignore-eos --single-turn --no-display-prompt \
  -p "$(cat ~/llama-cpp-rdna-boosts/prompts/reasoning.txt)" -n 300 > /tmp/oracle.out
python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/oracle.out     # hash WITHOUT -v
```

`llama-cli`/`llama-bench` raise the log level to ERROR unless `-v`, so every cache diagnostic (including
`FULLY RESIDENT`, the sizing line and the exit `moe_cache_report`) is invisible in a normal run — use
`-v` for diagnostics and a *separate* run for the hash (`-v` corrupts the text extractor).

### Env knobs (`moe-expert-cache.h`)

| knob | default | meaning |
|---|---|---|
| `MOE_EXPERT_CACHE_MIB` | 0 (inert) | per-device VRAM budget.  Full residency (`slots == n_experts`) turns the identity path on; on this model that is ≈9.3 GiB (`MIB≈9344`, slots 256/256) for the 2-GPU `-sm tensor` Q4_K_M iteration model. |
| `_SLOTS` | 0 | explicit uniform slots/table (forces immediate sizing). |
| `_COLD` | **`uva`** | `uva` = in-place pinned-host cold reads (the default now that the fused gate+up+GLU / down-fold kernels serve cold ids).  `off` restores the pre-1b fill-every-miss policy. |
| `_ADMIT` | `touch` | `always` / `value` = rejected admission rules. |
| `_PERIOD` / `_TOUCH` | 32 / 2 | LFRU decay period / touch threshold. |
| `_RESERVE_MIB` | 1024 | VRAM held back from the arena (arena is sized from *free* memory, so `--fit` need not know). |
| `_FAIL_ALLOC` | 0 | induced-allocation-failure fail-soft test. |
| `_ASSERT` / `_ASSERT_SABOTAGE` | 0 | structural invariant / gate-liveness self-test (CPU-split). |
| `_SELFTEST`, `_VERIFY`, `_REPORT`, `_DEBUG`, `_SKIP_ROLE`, `_FORCE_COPY`, `_NOEVICT`, `_CPUSPLIT` | — | bring-up / A-B knobs. |
| `_DEVMAP` | **1 (ON)** since session 19; `0` = off | device-side remap (item 3 / the "gentle curve"): build the slot remap on the device + deferred post-graph promotion, instead of the eager per-op host routing readback + full device sync.  Session 10 made the promotion **pipelined** (async double-buffered readback) and **slot-dirty-skipped** (**+22 % over eager** at partial residency: 85 vs 70 t/s at h~0.9).  The session-10 blocker was the promotion gates — byte-identity / width purity / MTP / coherence — which were **re-run green in session 19**, so it is now **default ON**.  At h=1 the identity fast path wins the lookup first (a no-op at full residency); measured default vs `DEVMAP=0` at `MIB=9216`: **70.7 -> 77.3 t/s (+9.4 %)**.  `DEVMAP=0` restores the eager host path. |
| `_FORCE_DEVMAP` | 0 | keep the devmap path even at `h=1` (suppresses the identity fast path).  A/B knob: at the same arena it isolates the devmap *machinery* cost (identity 94.0 vs forced-devmap 84.9 = 1.14 ms/token). |
| `_DEVPOLICY` | **1 (ON)** when `_DEVMAP=1`; `0` = off | run the LFRU admission + eviction + fill on the GPU: one batched kernel per device per token replaces the per-table host promotion (used-list D2H + host policy + slot-map H2D) and copies admitted experts from the pinned host alias into the arena in the same launch.  **Defaulted ON in session 13** after the self-test/width/MTP/concurrency gates; kill switch `=0`. |
| `_KSLOT` | **1 (ON)** when `_DEVMAP=1`; `0` = off | **B3 (session 19)**: resolve the slot map in the MoE ids consumer — `mul_mat_vec_q_moe` does `channel = slot[ids[i]]` (cold encoding `n_res+e`) and writes its own `used_dev`/`used_gate_dev`, so the per-table remap kernels are gone (160 capture launches -> **0**).  Byte-identical on 1/2/3 GPU (layer+tensor) incl. the cold path; width purity `none==n1==n3==n7==15038c19ddc8`; MTP n3 0.753; deep coherence.  Order-balanced **+1.3 % Q4_K_M**, ~+0.6 % Q8_0. |
| `_PREFILL_SEED` | **1 (ON)**; `0` = off | **the prompt-routing seed (session 14, B6, works).**  Tally the prefill routing on the device (a kernel from `ggml_cuda_mul_mat_id`, so `-sm tensor`'s block-06 staging cannot bypass it) and bulk-admit the hottest experts as provisional slots at the first decode-band policy flush.  Default ON (it passed the byte-identity / width-purity / `MUL_MAT_ID` / MTP / coherence gates); `=0` is the kill switch.  Needs `DEVMAP=1` (with `DEVPOLICY` default-on); inert with `DEVMAP=0`/`DEVPOLICY=0`, and the tally is gated on `g_devmap` so an unused seed costs nothing.  `llama-cli -n 300` `-sm layer`: `MIB=8192` 52.3 -> 57.9, `MIB=16384` 60.9 -> 80.8 t/s.  See `WORKLOG.md` 2026-09-29 (session 14). |
| `_PREFILL_SEED_N` | 0 | cap the seeded/loaded experts per table (`0` = all `slots`); the traffic knob. |
| `_PREFILL_LOAD` | **0** (off) | fill experts `0..slots-1` into the arena at load.  With `_PROVISIONAL=1` it is the **fastest** config (session 12i); without it, neutral-to-(-1.5 %) because a full arena gates every miss through `touch`. |
| `_PROVISIONAL` | **1 (ON)**; `0` = off | treat a pre-filled/seeded expert as an **empty slot for admission** until its first hit (per-slot `slot_prov`, cleared on hit/admission): the doorkeeper is bypassed for it while it still serves hits.  Default ON (session 14): it is the only way the prompt seed's not-yet-hit entries are reclaimed, and it is inert until something is pre-filled/seeded (a no-op for an empty arena).  `PREFILL_LOAD + PROVISIONAL` = 58.59 t/s vs empty 58.09 vs prefill-no-prov 57.22 (session 12i); seed `MIB=16384` 79.8 -> 80.6 with it (session 14).  Byte-identical. |
| `_WARMUP_TOKENS` | 0 (off) | **measured negative (session 12h)**: force first-touch admission (`always`) for the first N decode tokens to convert a pre-filled/seed arena quickly.  Raises `h` but lowers throughput (55.5 vs 58.1 t/s) - hit rate is not the objective, PCIe traffic is. |
| `_PROGRESS` | 0 (off) | log cumulative admissions (fills) vs evictions, resident/slots and `h` every N decode tokens per device (needs `-v` for llama-bench/cli).  Shows cache warm-up vs steady state; the log proves the default `touch` policy is still `WARMING` at 18k tokens. |
| `_TIMING` | 0 | print the deferred-promotion breakdown and the expert-access/traffic accounting at exit. |
| `GGML_SCHED_STAGE_AUTO` | 0 (off) | **dormant opt-in** item-3 adaptive staging-vs-gather probe (needs `GGML_SCHED_DEVGATHER`; disabled by `GGML_SCHED_GATHER_FIRST`).  Samples a warm gather pass then a staging pass (gather first, each probe pass synchronized), latches the faster with a 5 % hysteresis toward staging.  **Not promoted**: on r28 the fixed width-gated policy is already optimal (§0.B). |
| `GGML_SCHED_GATHER_FIRST` | 0 (off) | force the routed expert tables onto the device gather instead of whole-shard staging (the probe's gather-always arm; also the A/B knob). |
| `GGML_CUDA_CACHEDBG`, `GGML_CUDA_FUSE_LOG`, `GGML_SCHED_SYNCDBG`, `GGML_CUDA_GCDBG` | — | diagnostics. |

### The gates every change must pass

1. **Byte-identity to the `-ncmoe 0` oracle** (full-table GPU, fusions ON): `15038c19ddc8` at 300 tok
   (Q4_K_M, 2-GPU `-sm tensor`, `prompts/reasoning.txt`).
2. **Width purity**: `none == n1 == n3 == n7`.
3. **`-sm layer` regressions**: 1-GPU and 2-GPU, fusions off, `MIB=8192` → `ad30da7b5a3a`.
4. **`test-backend-ops -o MUL_MAT_ID`** green (929/929).
5. **MTP** `n3`: acceptance > ~0.45 at pos-1 and `draft-mtp >= plain` at depth 3.
6. **Deep coherence** (maintainer gate): a 12-section ~7k-word essay + MTP completes naturally, no
   repetition collapse.  Prompt: `coherence-essay-prompt.txt` in this directory (sha256
   `5e9a8ab0…`); run with `-n 12000 -c 16384 --spec-type draft-mtp --spec-draft-n-max 3`, no
   `--ignore-eos`, and check rc=0 + a `## Conclusion`.

---

## 2. Remaining work items

> Items 1, 2, 1a and 3b-II are **DONE** — their narratives were moved to `WORKLOG.md` ("§2 completed work items").  **Item 3 (the user graph-input copies) is closed NEGATIVE** (§0.A) — the `GGML_TENSOR_FLAG_INPUT` branch is negligible and the gfx1151 host-input ring was already rejected by the maintainer.  Item 4 remains here.

### Item 4 — Lower priority / deferred

* **Route (2), graph-level arena redirect**: once a table is sized, point the weight's `data` at the
  arena and stop the op being host/offloaded, with the remap as a persistent graph input.  Removes the
  `input_cpy`/offload entirely (the asymptotic shape), but needs a re-schedule after sizing.
* **Prefill cache-aware**: with `-ncmoe`, prefill uploads the used experts per ubatch and pays the
  op-offload sync; attribute the prefill gap and make prefill approach `-ncmoe 0` as experts load.
  Also inherited: **prune the staged prefill upload to the used experts** (r16 stages the whole tensor).
* **Static block-pin analyser** (no-code warm start; `-ncmoe` cannot express a non-uniform hot subset).
* **2-level VRAM cache / prefetch** — measured within ~0.01 of LFRU (SLRU) or superseded by it; revisit
  only on a measured miss-cost breakdown.
* **Tensor-split cold admission** is item 1 (done); the **static `NOEVICT` set** is measured worse, do not
  re-try as the mechanism.

---

## 3. The decode arena vs the prefill `-ncmoe` system (how they fit)

They are **two different axes that currently run independently** — not one system with two names, and
not really competing, but they are not yet unified and the VRAM accounting is split between them.

**Prefill (`-ncmoe N` + block-06 op-offload staging).**  `N` is a **static placement** decision: `N`
MoE layers' experts live in the host pool.  There is **no persistence** — each prefill ubatch uploads
the *used* experts (pruned) through the H2D staging ring, overlapped with compute, then the device copy
is reused for the next ubatch.  The deliverable's win was making that upload asynchronous/overlapped, so
the `-ncmoe` sweep is nearly flat (6450→5794 t/s over 0→40 on 1 GPU).  Prefill is **band-excluded from
the cache** (`n_tokens > 8`): the cache does not touch it.

**Decode (`MOE_EXPERT_CACHE_MIB`).**  The arena is a **persistent per-device VRAM cache** of a hot
subset (LFRU), decode/verify band only, over the same host pool `-ncmoe` created.  At `h=1` (slots ==
n_experts) the identity path makes it a plain device copy of the whole table; below that, misses are
served in place from the pinned host (UVA cold, item 1) or filled.  Its x-axis is therefore the
**resident fraction `h`**, not `-ncmoe`.

**Where they touch / the friction.**
* **Same bytes, two mechanisms.**  `-ncmoe N` puts experts in the host pool; the arena caches some of
them back onto the device.  Set `MIB` large enough and the arena *is* `-ncmoe 0` for decode while prefill
still streams from the host — that is the intended, desirable overlap, not a conflict.
* **VRAM accounting is independent.**  `--fit` sizes the KV cache + compute reserve and knows nothing
  about the arena (the arena is deliberately sized from *free* VRAM after `--fit`, minus
  `MOE_EXPERT_CACHE_RESERVE_MIB`), and the prefill staging arena is likewise outside the compute-graph
  reserve (issue #33).  So `MIB` is a third, self-managed VRAM consumer.  A big arena shrinks the free
  headroom a deep ubatch's staging/`--fit` growth might want, and the fail-soft path (arena alloc fails
  → the cache stands its fusions down, item 1a) is the safety valve.
* **No cross-regime reuse yet.**  The arena is not consulted by prefill, and prefill's staged upload is
  not left resident for decode.  On a mixed server the same experts are uploaded per prefill ubatch and
  separately cached for decode — correct, but two copies of the machinery.

**The unification is item 4's route (2)** ("graph-level arena redirect"): once a table is sized, point
the weight's `data` at the arena and stop the op being host/offloaded, with the remap as a persistent
graph input.  Then the arena *is* the persistent device-resident set, prefill only stages the misses, and
`-ncmoe`/the scheduler treat arena-resident experts as device-resident.  That is the asymptotic shape and
the natural long-term home; it needs a re-schedule after sizing, which is why it is deferred.

**Practical reading for the gentle-curve work.**  Keep the two axes separate for now: the item-3/item-3b
deliverable is a table over **arena size** (resident fraction), with `-ncmoe 99` fixed and the
`-ncmoe 0` / cache-off lines as endpoints.  Item 3 landed byte-correct, session 10 made it a win, and the
next task (item 3b-II) is to remove the per-table remap kernels and then, optionally, the remaining host
promotion (the original item-3b device-side admission policy).  A separate
follow-up should characterize the *mixed* case (`-ncmoe N` with an arena) once 3b lands, because that is
where the prefill and decode systems actually meet.

---

## 4. Completed work — index (one line each, detail in `WORKLOG.md`)

| # | done | detail |
|---|---|---|
| 1 | Phase 0 baseline + gap table (2-GPU resident 79.7/90.5, all-host CPU MoE 24.2 t/s) | WORKLOG: *HANDOVER BRIEF* / *The gap* |
| 2 | Routing profiling + admission policy: per-layer slot ranges + **LFRU** beat LRU/LFU/second-touch | WORKLOG: *FINDINGS* §1.6, §2.2 |
| 3 | Seam decision: `device_alias()`/base-pointer table at the `MUL_MAT_ID` weight source, **not** a buffer type | WORKLOG: *FINDINGS* §2.3 |
| 4 | Phase 1a: per-layer arena + LFRU + slot-remap consumer, driven from the scheduler hook | WORKLOG: *The hardening tasks* (Phase 1a status), *EVICTION BUG* |
| 5 | Eviction wrong-output bug root-caused (CUDA graph capture vs the per-token host decision) and fixed | WORKLOG: *EVICTION BUG* |
| 6 | H1 targeted fusion guard (only cache-band routed fusions stand down) | WORKLOG: *The hardening tasks* H1 |
| 7 | H2 width purity + transparency (byte-identical to the oracle across `none`/`n3`/`n7`) | WORKLOG: *H2 RECORD* |
| 8 | H3 budget/slot allocator (uniform, adaptive, aligns gate/up/down residency) | WORKLOG: *The hardening tasks* H3 |
| 9 | **Delivery fix promoted:** the qwen35moe `ssm_gate_beta` fusion was width-impure → release `v16-84e76d8a2-r18` | WORKLOG: *FUSION WIDTH-UNIFORMITY* |
| 10 | Phase 1b UVA cold read + the `touch` admission policy (later found to conflict with fused kernels — see item 2) | WORKLOG: *PHASE 1B RECORD*, *PHASE 1B POLICY RECORD* |
| 11 | Phase 1d fail-soft + `--fit` (arena sized from free VRAM; induced-failure liveness) | WORKLOG: *1D RECORD* |
| 12 | CPU-computes-the-misses arm: built, correct, **negative above a ~1.2 GiB arena** (per-op dispatch, not CPU compute) | WORKLOG: *CPU-COMPUTES … DESIGN/TEST PLAN*, *… RESULT* |
| 13 | Phase 2 `-sm layer`: "only 1 GPU active" bug fixed (pass-3.5 rebalance) + per-device arenas/MIB | WORKLOG: *EARLIER HANDOVER (session 3)* |
| 14 | Phase 3 `-sm tensor`: per-device slice arenas + Meta delegator; byte-identical | WORKLOG: *PHASE 3 RECORD* |
| 15 | Phase 3 route (1): one layer's routed ops in one split, consumer-driven hook, deferred per-input sync → +93 % over CPU MoE | WORKLOG: *PHASE 3 FINDING*, *PHASE 3 RESULT* |
| 16 | Session 7 attribution: the gap is the per-layer host round-trip (~42 input loops + ~83 ids readbacks/token), not fusion (down fold ~0.2 %) | WORKLOG: *SESSION 7 RESULT* |
| 17 | **Session 7 identity fast path** (the gate): warm `tg1024` 70.3 → 94.2 vs the `-ncmoe 0` 96.4 oracle (97.7 %) | WORKLOG: *SESSION 7 RESULT* |
| 18 | `-sm layer` / 1-GPU curves with identity: 1-GPU 90.0/94.7 (95 %), 2-GPU 72.2/82.9 (87 %) | WORKLOG: *SESSION 7 RESULT* |
| 19 | Cold + fused-kernel hang found; **workaround** `COLD=off` (+ the cold stand-down); deep coherence gate passes | WORKLOG: *SESSION 7 CORRECTNESS FIX + DEEP COHERENCE GATE* |
| 20 | Device-side remap first cut (**broken**, parked) | branch `wip-moe-devmap`; WORKLOG: *SESSION 7 CORRECTNESS FIX* (last paragraph) |
| 21 | **Session 8 item 1**: per-expert-stride tensor-split UVA cold path (`cold_channel_stride`/`cold_row_stride`, host geometry in `moe_cache_get_cold`); axis-0 `ffn_down` verified | WORKLOG: *SESSION 8 ITEMS 1+2* |
| 22 | **Session 8 item 2**: cold-aware fused gate+up+GLU (independent gate-lane cold geometry); `COLD=uva` default restored, cold stand-down removed | WORKLOG: *SESSION 8 ITEMS 1+2* |
| 23 | **Session 8 item 1a**: a cache that cannot serve stands its cache-band fusions down wholesale (fail-soft + tiny-MIB byte-identity bug fixed) | WORKLOG: *SESSION 8 ITEMS 1+2* |
| 24 | **Session 9 item 3**: device-side remap v2 — byte-correct (transition arming + persistent `used_dev` used-list + admission parity), but **10-15 % slower than eager at every residency**, so it does not flatten the cliff; kept opt-in | WORKLOG: *ITEM 3 DEVICE-SIDE REMAP*; branch `wip-moe-devmap-v2`; `exp10-…-devmap.patch` |
| 25 | **Session 10 item 3b-I**: pipelined double-buffered used-list readback + slot-dirty skip — promote **25.1 -> 1.8 us/call, 6.2 -> 0.43 ms/token**; devmap **60 -> 85.6 t/s at h~0.98** (+23 % over eager), plateau gone, cliff halved to 85.6 -> 94.1; byte-identical | WORKLOG: *ITEM 3B PIPELINED PROMOTION*; `exp11-…-pipelined.patch` |
| 26 | **Session 10 attribution**: same-arena `FORCE_DEVMAP` A/B + `rocprofv3` -> residual 1.14 ms/token = ~48 % 240 per-table remap kernels (240/token, 1.57 us) / ~39 % host promotion / ~11 % PCIe; high-`h` cycling is compulsory only (evictions ~0, 4.4 MiB/token) | WORKLOG: *ITEM 3B PIPELINED PROMOTION* |
| 27 | **Session 10 next target (open)**: item 3b-II — eliminate/batch the per-table remap kernels (fold the redundant gate-lane remap; batch; or fuse the slot lookup into the MoE ids read) | README: *Item 3b-II* |
| 28 | **Session 11 r25 rebase**: replayed the 16 campaign commits onto delivery r25 (`81fda69c8`); 2 conflicts (`ssm_gate_beta` kill-switch vs r18/r23, `wait_before_overwrite` vs r22 `GGML_ENV_STR`); new tips `ecac6360c` / `7e6c4cf66`, pre-rebase SHAs in `backup/*-r21`; the 2-GPU `-sm tensor` oracle moved to `de8be4d0c90c`, `-sm layer` unchanged at `15038c19ddc8`, MUL_MAT_ID 929/929 | WORKLOG: *ITEM 3B-II OPTION 1 + r25 REBASE*; `exp12-moe-expert-cache-r25.patch` |
| 29 | **Session 11 item 3b-II option 1**: fold the gate-lane `used_dev` write into the up-lane remap kernel — **240 -> 160 remap launches/token** (captures 960 -> 640), byte-identical at `MIB=1024/9216` + `-sm layer` purity, throughput +~1 % | WORKLOG: *ITEM 3B-II OPTION 1 + r25 REBASE* |
| 30 | **Session 11 item 3b-II option 2**: build the layer's routed `down` remap in the gate+up kernel (from the down's own slot map) and skip the down launch (`remap_fresh`) — **160 -> 80 launches/token** (captures 640 -> 320), byte-identical + width-pure; **and the `rocprofv3` finding that the remap kernels are not the critical path** — the ~0.36 ms/token host promotion (d2h + policy) is, and skipping it collapses to all-cold | WORKLOG: *ITEM 3B-II OPTION 2* |
| 31 | **Session 11b**: the **dirty-table** promotion filter was implemented and **rejected** — the policy cost is the *dirty* tables' fills, not the clean tables' lookups (policy only 734.7 -> 644.1 ms; evictions 0 -> 102) | WORKLOG: *ITEM 3B-II OPTION 2* (the tried negative result) |
| 32 | **Session 12 item-3b**: the **device-side admission policy** (`MOE_EXPERT_CACHE_DEVPOLICY=1`, default off) — one batched LFRU kernel per device per token + in-kernel host->arena fill copy, no used-list D2H / slot-map H2D / fill-list readback.  Win: `forced-devmap MIB=9344` 86.35 -> **89.74**, `MIB=1024` 46.73 -> **50.87**, `MIB=4096` flat; byte-identical, width-pure, MUL_MAT_ID, deep coherence | WORKLOG: *DEVICE-SIDE ADMISSION POLICY + PROGRESS LOG*; branch tip `2632f6011`; `exp13-moe-expert-cache-r25-devpolicy.patch` |
| 33 | **Session 12 instrumentation**: `MOE_EXPERT_CACHE_PROGRESS=N` / `_MS=T` log cumulative admissions/evictions (+ resident/slots/h and per-interval rates) per device — the wall-clock view shows the admit **rate** decays 4500/s -> ~12/s with ~0 evictions, i.e. a hot-set grower, not a churner | WORKLOG: *DEVICE-SIDE ADMISSION POLICY + PROGRESS LOG*, *PROGRESS-LOG CORRECTION* |
| 34 | **Session 12c prefill seed (scaffold)**: `_PREFILL_SEED=1` tallies the prefill routing and bulk-admits the hottest at sizing — **gated off and currently inert**; the prefill upload is intercepted by the block-06 staging before the cache hook, and sizing runs at a load-time/`--fit` warmup before the prompt | WORKLOG: *PREFILL-SEED PROTOTYPE* |
| 35 | **Session 13 (group A)**: device-policy vs host **self-test PASS**; **two cache-enabled pessimisation fixes** (forced `MUL_MAT_ID` offload scoped to the decode band; graph-capture gate scoped to graphs with a cache-band op) — `npl=16` 137->52 regression fixed; **`DEVPOLICY` defaulted ON**; concurrency in-band `npl` 1/2/4/8 = 49/113/172/204 t/s vs 36/50/80/100 no-cache | WORKLOG: *GROUP A PROMOTION GATES* |
| 36 | **Session 14 (B6)**: the **prompt-routing seed** — device prefill tally from `ggml_cuda_mul_mat_id` + `seed_prefill_lazy_locked` bulk-admits the hottest experts as provisional slots at the first decode-band flush.  **`PREFILL_SEED` and `PROVISIONAL` defaulted ON** (kill switch `=0`).  Byte-identical (`15038c19ddc8` / `de8be4d0c90c`), width-pure, `MUL_MAT_ID` 929/929, MTP `n3` 0.79989, coherence green; `-sm layer` `-n 300` `MIB=16384` 60.9 -> **80.8**, `MIB=8192` 52.3 -> **57.9** t/s (beats arbitrary `PREFILL_LOAD`+prov by up to +7 % at mid residency) | WORKLOG: *B6 PROMPT-ROUTING SEED*; patch `exp14-…-b6-prefill-seed.patch` |
| 37 | **Session 15 (B2 first cut)**: **device-side expert gather** for small-ubatch `-ncmoe` prefill (`moe_cache_gather_kernel` + `GGML_SCHED_DEVGATHER`, prefill band; CUDA + Meta) — replaces the per-op routing readback + device sync, on the compute stream so the overwrite/input syncs drop too.  Byte-identical (N=300/1000), `MUL_MAT_ID` 929/929; `-ub 512` prefill `pp2048` 1-GPU +4.5 % / 2-GPU tensor +22.6 %, `pp4096` +6.7 %/+18.5 %, no regression at `-ub 8192`.  **Superseded on the "pageable UVA" point by row 38; default flipped OFF in row 39** | WORKLOG: *B2 FIRST CUT*; patch `exp15-…-b2-devgather.patch` |
| 38 | **Session 16 (B2 measured)**: the gather is **already at PCIe link speed** — `MOE_CACHE_GATHER_NOCOPY` gives 5376 t/s vs 720 at `pp2048 -ub 512` (on-device 5403), and instrumented traffic is 51.2 MiB / 168.6 experts per call, ~49 GB/pass = **~20 GB/s aggregate**, faster than the staged path's DMA (13.3 GB/s).  The gap is each of the 4 ubatches re-uploading its ~65 % subset; **the recorded DMA+compact route is a dead end**; the fix is residency (B1) | WORKLOG: *B2 MEASURED* |
| 39 | **Session 16 (B4)**: the cache **arms and runs on Qwen3.8-Flash-Next** (3x R9700 `-sm tensor`).  IQ4_NL (100 GiB, `-ncmoe 48`): 432 tables, 388/512 slots/table (75.8 %), arena 49 GiB, device-remap armed; coherent + deterministic, decode **14.9 -> 29.0 t/s (+95 %)**.  **Transparency PASS** on IQ3_XXS (fits `-ncmoe 0`): oracle `77c6f546460d` == cache-on `-ncmoe 99` `77c6f546460d`, cache-off (CPU MoE) `32576231856e`; 16.5 -> 29.0 t/s (+76 %).  **Width purity PASS**: `plain == draft-mtp n_max 1 == 3 == 7`.  Also **found the B2 gather corrupts qwen4exp prefill** -> **defaulted OFF** (commit `c5bbb7ee2`).  `PREFILL_SEED`/`PROVISIONAL` default-ON retained | WORKLOG: *B4 VALIDATION*; patch `exp16-…-b4-gather-off.patch` |
| 40 | **Session 17 (B1)**: single R9700 8K prefill `-ub 512/1024/2048/8192` `~990/1474/1519/1461` t/s (was `220/331/448/581`).  Merged routed-MoE bands (31 inputs) were never staged, so the 450 MiB expert weight uploaded serially behind host event syncs; now `sched_stage_issue` counts host-weight inputs and defers routed tables to the **device gather (default ON, unsplit only)**, `GGML_SCHED_EVENTS` defaults ON, and host->device split inputs copy asynchronously.  Gates: `15038c19ddc8` / `de8be4d0c90c`, width purity `2ede4fe056cc`, `MUL_MAT_ID` 929/929, 3-device qwen4exp coherent (`359ff4337837`) | WORKLOG: *B1*; patch `exp17-…-b1-prefill-gather.patch` |
| 41 | **Session 17 addendum (r26 rebase)**: the campaign is rebased onto delivery **r26** (`0d58404e1`); the scheduler half of B1 is delivery block-06 now, so the duplicated hunks are dropped.  Tip `6ca5c1c77`, patch `exp18`.  Rebased 8K prefill `-ub 512/2048/8192` `~1044/1627/1761` t/s; gates `15038c19ddc8` / `de8be4d0c90c` / `MUL_MAT_ID` 929/929 green | WORKLOG: *B1*; patch `exp18-…-r26-b1-rebase.patch` |
| 42 | **Session 18 (item 1)**: the session-16 gather attribution was **wrong** — the slice geometry was correct all along (D2H of every routed expert on all 3 devices matched the host byte-for-byte); the qwen4exp prefill corruption was the pruned gather missing the host path's MMQ **expert-table tail pad** (`min(expert_size,512)` past each routed group's last expert).  The gather now pads the next expert's slice; `.moe_cache_gather` re-enabled, tensor-split arm default ON (`GGML_META_GATHER_NOPAD=1` / `GGML_SCHED_DEVGATHER=0` kill switches).  Gates: `de8be4d0c90c` / `15038c19ddc8`, 3-device IQ3 cache-on == `-ncmoe 0`, deep coherence (12k essay rc=0, 13 sections, `## Conclusion`), `MUL_MAT_ID` OK, `pp2048 -ub 512` 2-GPU tensor prefill **609 -> 724 t/s (+18.9 %)**.  Tip `2376ac6cf`, patch `exp19` | WORKLOG: *ITEM 1 FIXED*; patch `exp19-moe-expert-cache-r26-b2-tensorpad.patch` |
| 43 | **Session 19 (B3)**: resolve the MoE slot map in the ids consumer (`mul_mat_vec_q_moe` does `slot[ids[i]]` + writes its own used-list), so the per-table remap kernels vanish (160 -> **0**); `moe_cache_get_slot`/`moe_cache_kslot_active`; **default ON with `DEVMAP`** (`KSLOT=0` opt-out).  Byte-identical 1/2/3 GPU (layer+tensor) incl. the cold encoding; width purity `none==n1==n3==n7==15038c19ddc8`; MTP n3 0.753; `MUL_MAT_ID` OK; deep coherence rc=0 13 sections.  Order-balanced **+1.3 %** Q4_K_M (Q8_0 ~+0.6 %, i.e. smaller not larger) | WORKLOG: *B3 DONE*; tip `a8b493184`, patch `exp20-moe-expert-cache-r26-b3-kslot.patch` |
| 44 | **Session 19b**: **`MOE_EXPERT_CACHE_DEVMAP` default flipped ON** (`DEVMAP=0` opts out) — the device-remap path's promotion gates (byte-identity 1/2/3 GPU, width purity, MTP, coherence, `MUL_MAT_ID`) were re-run green; default vs eager `MIB=9216` **+9.4 %** (`DEVPOLICY`/`KSLOT` ride along).  Tip `f4b255041`, patch `exp21` | WORKLOG: *DEVMAP DEFAULT FLIP*; patch `exp21-moe-expert-cache-r26-devmap-default.patch` |
| 45 | **Session 21**: rebase the campaign onto delivery **r28** (`60361cb9f`) + the adaptive staging-vs-gather probe (`GGML_SCHED_STAGE_AUTO`, opt-in; gather-first ordering + probe-pass sync + 5 % hysteresis); root-caused the first attempt's impossible throughput as **staging-ring residency** | `item3-findings-session20.md` "Session 21"; patch `exp22-moe-expert-cache-r28-adaptive-stage.patch` |
| 46 | **Session 21b (item C)**: fixed the **cache-enabled-but-unserviceable** transparency divergence — `moe_cache_has_tables()` distinguishes "no routed expert table" (behave as if disabled) from "tables exist but cannot serve" (r8 item-23 stand-down).  Validated on gfx1201 and gfx1151 | `item3-findings-session20.md` "Session 21b"; patch `exp23-moe-expert-cache-r28-itemC-fusion-guard.patch` |
| 47 | **Session 21d**: the adaptive staging-vs-gather gate (B) closed as a **no-op on r28** — staging wins/ties above the width gate on Q4_K_M and IQ4; the r26 qwen4exp gather wins no longer reproduce.  `GGML_SCHED_STAGE_AUTO` kept as a dormant opt-in | `item3-findings-session20.md` "Session 21d" |
| 48 | **Session 21e (gfx1100)**: campaign validated on `fingon` (RX 7900 XTX, 24 GiB) — a 29.3 GB Q6_K 35B-A3B that cannot fit runs at **32.8 -> 80.3 t/s** tg128 (warm ~94, h=0.9312) with the cache; Q4_K_M transparency `359ff4337837`; `MUL_MAT_ID` 929/929 | `WORKLOG.md` session 21e |
| — | **NO OPEN ITEMS**: B1/B2/B3/B4/B6, item 1 and the DEVMAP default are DONE; item 3 is NEGATIVE; item C is fixed; the adaptive gate (B) is closed as a no-op on r28. | README §0; WORKLOG |

| 44 | **DEVMAP default ON** (`MOE_EXPERT_CACHE_DEVMAP=1`); `DEVPOLICY`/`KSLOT` ride along; default vs eager +9.4 % at MIB=9216 | WORKLOG: *DEVMAP DEFAULT FLIP* |
| 45 | **Adaptive staging/gather probe — OPEN, handed over.** The decision is model+ubatch dependent and needs a runtime probe; the first implementation was reverted (it skipped the expert upload). Start at `item3-findings-session20.md` | this README §0; `item3-findings-session20.md` |
| 46 | **gemma4 `-sm tensor` excluded** (segmented expert upload unfinished, parked); `-sm layer` works | `item3-findings-session20.md`; delivery `TODO.md` |
| 47 | Item 3 diagnosis complete: the cost is the meta `stage_input` host gather (not the INPUT branch); 4-model A/B matrix + the rule "stage iff host-gather/pass < gather-pass" | `item3-findings-session20.md` |
| 48 | `stage_gather` silent-failure fixed (aborts instead of leaving an unfilled ring slot with no done-event) | `item3-findings-session20.md` |

### Reference tables (kept in the WORKLOG, not duplicated here)

* 2-GPU `-sm layer` quick-reference harness + traps — WORKLOG: *2-GPU `-sm layer` quick reference*.
* State-in-one-screen, build/run, and the H1/H2/H3 detail — WORKLOG: *State in one screen* onward.
* Prior art (GenerelSchwerz `moe-cache`, R9V, Strata) and the phased plan — WORKLOG sections 0-3.
* Traps learned (do not re-derive) — WORKLOG: *Traps learned this session*.
