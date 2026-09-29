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

## 0. One-screen status (2026-09-28, session 8)

**Items 1 and 2 are DONE.  The `-sm tensor` full-residency gate is met and the partial-residency curve
is now byte-identical down to a zero-VRAM arena.**  The per-expert-stride tensor-split UVA cold path
(item 1) and the cold-aware fused gate+up+GLU / down-fold kernels (item 2) are implemented, and UVA
cold is the default again now that the fused kernels serve cold ids.  Every residency from a 1 MiB
budget (no arena at all) through the identity path is byte-identical to the `-ncmoe 0` oracle
`15038c19ddc8` at 300 tokens: `MIB = 1/64/256/1024/1536/2048/4096/8192/10240`, width-pure
`none == n1 == n3 == n7`, `-sm layer` 1/2-GPU `ad30da7b5a3a`, `test-backend-ops -o MUL_MAT_ID`
929/929, MTP `n3` acceptance 0.78855, and the deep coherence essay (12 sections + `## Conclusion`,
7012 words, rc=0).

The default flip also **fixes a pre-existing byte-identity bug the campaign had not hit**: with
`COLD=off` (or any config where the cache declined), the *cache-band gate+up fusion* read the
scheduler's op-offload `input_cpy` through a fused call site that is only correct when the scheduler
itself staged the copy, so a 4-token batch (used experts > slots) produced garbage logits from
`MIB <= 1024`.  `ggml_cuda_cache_blocks_fusion` now stands the cache-band fusions down wholesale
whenever `moe_cache_has_arena()` is false (priming, any failed allocation, a budget below one expert),
and the hook declines when a table has no arena, so a cache that cannot serve falls back to the
per-op path.  The old `MOE_EXPERT_CACHE_COLD=off` small-arena decline is therefore also correct now,
and the induced/partial-allocation fail-soft path is byte-identical.

**The one remaining headline task is item 3 (device-side remap / the "gentle curve" at `h<1`).**
A partial arena is byte-correct and uses the `touch` + in-place-UVA policy at every size, but the
per-layer host round-trip still caps `h<1` throughput (the identity path removes it only at `h==1`).

---

## 1. Where things are

| | |
|---|---|
| **Worktree** | `~/llama-decode`, branch **`wip-moe-expert-cache`**, tip **`c7dd40a23`** = r21 `feefecfbc` + 13 wip commits (10 files, +2858/-37), clean tree. |
| **Build** | `cd ~/llama-decode && cmake --build build-rocm --target llama-cli llama-bench -j 16` (~1-2 min incremental with ccache).  Full rebuild: `BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714` (~7 min cold). |
| **Full patch** | `exp9-moe-expert-cache-session8-coldpath.patch` (3329 lines, `git diff feefecfbc..wip-moe-expert-cache`, clean-applies to r21).  `exp8` = the session-7 cold workaround snapshot; `exp7` = pre-cold-fix session 7; `exp6` = session 6. |
| **Parked branch** | **`wip-moe-devmap`** (`6b8a7ed06`, **marked BROKEN**) — the first-cut device-side remap (gentle curve). |
| **Iteration model** | `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21 GiB, fits 1 card; the fast smoke model). |
| **End-goal model** | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/…` (93 GiB `qwen4exp`, 48x512 experts) — Phase 4; its lazy/PLE path makes `llama-bench` absolutes non-comparable. |
| **Hardware** | 3x R9700 (gfx1201); use `HIP_VISIBLE_DEVICES=0[,1]`.  Pin `-t 8` (the GPU IRQs live on the top cores). |
| **Delivery** | `~/llama-cpp-rdna-boosts` `main`; the campaign README/WORKLOG/patches live in `wip/moe-expert-cache/`.  The `~/llama-decode` checkout is **never pushed**. |

### How to run (throughput / purity / MTP / coherence)

```sh
MQ4=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

# throughput (per-rep, warm = reps >= 3; rep 1 pays the one-time identity fill)
HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=10240 ./build-rocm/bin/llama-bench \
  -m $MQ4 -ncmoe 99 -ngl 99 -fa 1 -sm tensor -t 8 -p 0 -n 1024 -r 6 -o jsonl

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
| `MOE_EXPERT_CACHE_MIB` | 0 (inert) | per-device VRAM budget.  Full residency (`slots == n_experts`) turns the identity path on; reached at ~`MIB=10240` for the 2-GPU `-sm tensor` Q4_K_M iteration model. |
| `_SLOTS` | 0 | explicit uniform slots/table (forces immediate sizing). |
| `_COLD` | **`uva`** | `uva` = in-place pinned-host cold reads (the default now that the fused gate+up+GLU / down-fold kernels serve cold ids).  `off` restores the pre-1b fill-every-miss policy. |
| `_FUSED_COLD_OK` | — | (removed: the fused kernels now do cold reads, so there is no stand-down to override.) |
| `_ADMIT` | `touch` | `always` / `value` = rejected admission rules. |
| `_PERIOD` / `_TOUCH` | 32 / 2 | LFRU decay period / touch threshold. |
| `_RESERVE_MIB` | 1024 | VRAM held back from the arena (arena is sized from *free* memory, so `--fit` need not know). |
| `_FAIL_ALLOC` | 0 | induced-allocation-failure fail-soft test. |
| `_ASSERT` / `_ASSERT_SABOTAGE` | 0 | structural invariant / gate-liveness self-test (CPU-split). |
| `_SELFTEST`, `_VERIFY`, `_REPORT`, `_DEBUG`, `_SKIP_ROLE`, `_FORCE_COPY`, `_NOEVICT`, `_CPUSPLIT` | — | bring-up / A-B knobs. |
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

### Item 1 — Per-expert-stride tensor-split UVA cold path  ✅ **DONE (session 8)**

`moe_cache_get_cold` now returns the host master's device alias (with the per-device slice offset
folded in), the resident slot count, and the host geometry — the per-expert byte stride and the per-row
byte pitch (0 meaning "the arena's own row stride").  `cold_safe` is now "the host geometry is
representable" (any contiguous per-expert blob, or equal-width rows for the strided axis-0 `ffn_down`
slice), not "unsplit", and `mul_mat_vec_q_moe` takes `cold_channel_stride`/`cold_row_stride` and uses
`cold ? cold_base + cold_channel_stride*(id-n_res) + cold_row_stride*row` for a cold id.  The resident
path is untouched (`ne[2]`/`nb` stay at the full expert count, so the dispatcher heuristics are
unchanged).  Verified byte-identical to the oracle at every partial residency down to `MIB=1` (no arena),
including the axis-0 `ffn_down` slice (the tricky one): a forced-cold A/B at `MIB=1024` (h=0.55,
~217k cold reaches) is `15038c19ddc8`.  See `exp9-…-session8-coldpath.patch` and the WORKLOG
session-8 entry.

### Item 2 — Make the fused cache-band MoE kernels cold-aware  ✅ **DONE (session 8)**

The gate+up+GLU fusion reaches `mul_mat_vec_q_moe_launch` (the `has_ids` mmvq path), so the fix was to
thread the item-1 geometry through it AND to resolve the **gate** lane's cold geometry independently
(the gate lane is a separate table with its own pinned host master and slot count, even though it shares
the up lane's remap).  The fused gate+up and the `[MUL_MAT_ID, MUL]` down fold now serve cold ids, so:
`MOE_EXPERT_CACHE_COLD` defaults to `uva`; the `ggml_cuda_cache_blocks_fusion` cold stand-down, the
`MOE_EXPERT_CACHE_FUSED_COLD_OK` footgun and its warning are gone; and the `touch` + in-place-UVA policy
is the default again.  Along the way the change exposed and fixed a pre-existing byte-identity bug — see
"Remaining work items → Item 1a" below.

### Item 1a — A cache that cannot serve must fall back wholesale  ✅ **DONE (session 8)**

With `COLD=off`, or any config where the hook declined (a 4-token batch with `used > slots`, an arena
alloc failure, a budget below one expert), the **cache-band gate+up fusion** read the scheduler's
op-offload `input_cpy` through a fused call site that is only correct when the scheduler itself staged
the copy — so `MIB <= 1024` (and the induced/partial fail-soft path) emitted garbage logits from a
near-tie ~120 tokens in.  `ggml_cuda_cache_blocks_fusion` now stands the cache-band MoE fusions down
whenever `moe_cache_has_arena()` is false (false until sizing, on any failed arena, on an empty cache),
and `moe_cache_update_host` declines when a table has no arena.  Verified: `MIB=1/64/256/1024`,
`COLD=off` at `MIB=1`, `FAIL_ALLOC=1/3/7` (all/partial induced failures) are all `15038c19ddc8`, and
the `COLD=off` `MIB=1024` kill-switch is correct again.

### Item 3 — Device-side remap ("gentle curve")  ⟵ **start here**

**Goal.**  At `h<1` the cache still falls to ~70 t/s because the scheduler reads the routing back to the
host per layer (for the used-expert pruning *and* the cache's remap decision) and does a full device
synchronize per layer.  The identity path removes that only at `h==1`.  The maintainer's target is a
**gentle** approach as the resident fraction rises.

**Design.**  Build the slot remap **on the device** from the routing `ids` and a device
`expert -> slot` map (`slot_dev` int32[n_experts]) with a tiny capture-safe prepass kernel; the
scheduler takes the input over **before** the ids readback and records the routing for a **deferred
post-graph promotion pass** (one sync per token, not one per layer).  Fills then lag one token; the
current token's misses must be servable cold — item 1 (the per-expert-stride cold path) is now DONE,
so a split table is servable cold and the blocker is gone.

**State.**  A first cut is on branch **`wip-moe-devmap`** (`6b8a7ed06`, BROKEN): the kernel, the
`moe_cache_devmap` struct, `moe_cache_get_table`'s devmap branch, `moe_cache_take_over`/`_promote_host`,
the CUDA/Meta adapters, the scheduler takeover + deferred post-pass, and the fused redirect already
accept a `stream`.  It faulted at `h<1` on `-sm tensor` because split tables were `cold_safe=false`
(item 1); that prerequisite has landed, so restart from it and re-test — the identity path and the cold
path are cleanly separated on `wip-moe-expert-cache`, so cherry-pick/rebuild rather than merging the
broken tip.

**Extra care.**  The `slot_dev` H2D and the fill copies are enqueued on the compute stream after the
graph, so the next token's remap kernel and MoE kernel are ordered after them; the capture must not
bake a per-token decision (the known graph/eviction class — keep decisions constant per shape).

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

## 3. Completed work — index (one line each, detail in `WORKLOG.md`)

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

### Reference tables (kept in the WORKLOG, not duplicated here)

* 2-GPU `-sm layer` quick-reference harness + traps — WORKLOG: *2-GPU `-sm layer` quick reference*.
* State-in-one-screen, build/run, and the H1/H2/H3 detail — WORKLOG: *State in one screen* onward.
* Prior art (GenerelSchwerz `moe-cache`, R9V, Strata) and the phased plan — WORKLOG sections 0-3.
* Traps learned (do not re-derive) — WORKLOG: *Traps learned this session*.
