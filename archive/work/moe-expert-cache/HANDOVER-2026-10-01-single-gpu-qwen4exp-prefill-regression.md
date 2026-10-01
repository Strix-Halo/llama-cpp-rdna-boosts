# HANDOVER — "halo" single-GPU Qwen3.8-Flash-Next IQ4_NL: **prefill + decode together**

**Opened:** 2026-10-01 (supersedes the earlier single-sided version of this note)
**Status:** **SOLVED 2026-10-01** — two independent bugs found and fixed; see "## SOLVED" below.
**Priority:** HIGH (was) — this was the combined prefill + decode campaign's headline result.
**Bisection start:** `r26-b1 6ca5c1c77` (good prefill).

## SOLVED (2026-10-01) — two bugs, both fixed

**Bug 1 — decode collapse (gather-path registration).** The MoE cache registered a table only on the
host upload hook (`moe_cache_update_host`); the **device gather** (`moe_cache_gather_host`) uploaded
without registering.  A prefill that used the gather registered only the layer that fell off it (the
last one), the deferred arena sizing latched on those 3 tables (`sized 512 slots/table from 3 tables`),
the other 47 layers registered afterwards with 0 slots and declined, and decode fell off the cache —
worse than uncached (8.54 t/s vs 20) because the partial arena stood the cache-band fusions down
globally.  *Fix:* one `moe_cache_table(...)` call in `moe_cache_gather_host`.  Proof:
`MOE_EXPERT_CACHE_SLOTS=97` (explicit slots) restored 36.5; `GGML_SCHED_DEVGATHER=0` sized 144 tables.

**Bug 2 — prefill regression (the session-18 MMQ tail pad).** Bisected to campaign commit `d76e18efd`
("item 1"): the pad written after every routed expert to keep the MMQ's 512-byte expert-table over-read
finite cost **~3x** in *every* form tried (in-kernel trailing copy, folded into the last chunk, separate
kernel, each block padding its own head, zeros vs host bytes — all ~660 t/s vs 2130+).  It is **not** a
stride/misalignment (the layout is untouched and **zeros fix it**, so the over-read only needs *finite*
bytes) and not bandwidth (a 512 x 64-byte write = 32 KB).  The correctness threshold is exactly **64
bytes** (one cache line; 48 still corrupts, 64 is fine) and the perf cliff is ~54, so the fast and
correct bands do not overlap for a per-gather pad.  *Fix:* **zero the first 64 bytes of every expert
slot ONCE per `input_cpy` buffer** (`moe_cache_gather_zero_heads_kernel`, tracked in `g_heads_zeroed`).
Only the non-routed heads ever rely on it — the gather overwrites routed heads with real data — so the
invariant holds for the buffer's life with no per-gather cost.  (Note: the later "§0.6 separate second
regression" observation was a measurement error — a bad env-var name `MIN_TOKENS` instead of
`GGML_SCHED_STAGE_MIN_TOKENS`, which reverted the run to the staging path.  There is only one prefill
regression: this pad.)

**Verified (single R9700, Qwen3.8-Flash-Next IQ4_NL, `-ncmoe 48 -sm layer -fa 1`, MIB=12288,
`-b 2048 -ub 2048`, gather on):**

| metric | before | after |
|---|---:|---:|
| `pp8192` | 641 (staging) / 648 (gather+pad) | **2401** |
| `tg1024` (post-prefill, 8K KV) | 8.5 (collapse) / 40 | **36.5** |
| coherence (`coherence-essay-prompt.txt`, `-n 12000 -c 16384 --reasoning off`) | `[Start thinking] //////` | **PASS**: 8457 words, 12 numbered sections + `### Conclusion`, fluent prose, 0 `////`, no repetition |

`-p 8192 -n 1024` in one process: **pp 2401 + tg 36.5** — the prefill+decode middle ground.

## Transfer to Qwen3.6-35B-A3B Q8_0 (`qwen35moe`, 37.8 GB, single GPU)

Same combined test, swapped model (`-ncmoe N -sm layer -fa 1`, MIB as noted, `-b 2048 -ub 2048`,
`-p 8192 -n 1024 -r 2`):

| config | beta4 pre-fix (pp / tg) | beta5 fixed (pp / tg) |
|---|---:|---:|
| `-ncmoe 8`  MIB=4096 | — | **2644 / 72.5** |
| `-ncmoe 12` MIB=8192 | — | 2081 / 80.5 |
| `-ncmoe 16` MIB=8192 | 1746 / 80.1 | 1743 / **81.5** |
| `-ncmoe 16` MIB=16384 | — | 1751 / 81.6 |
| `-ncmoe 24` MIB=12288 | — | 1320 / 78.9 |
| `-ncmoe 40` MIB=12288 | 903 / 61.4 | 901 / 61.6 |

**The fix is neutral for qwen35moe** — it never had either bug (the registration collapse is
qwen4exp-specific; the tail pad did not cost qwen35moe the ~3x it cost qwen4exp).  But the model is
*already* fast on one oversized card: **up to 2644 t/s prefill + 72.5 t/s decode** at `-ncmoe 8`, or
**1750 / 81.6** at `-ncmoe 16` (the best decode).  `-ncmoe 8` now fits at `-ub 2048` (it OOM'd at
`-ub 4096`).  Coherence clean (`slashline=0`, proper thinking trace).  So no regression from the fix,
and a good single-GPU middle ground either way.

Patch: `archive/work/moe-expert-cache/beta5-fix-registration-and-head-zero.patch` (also `~/llama-fold` branch
`beta4` = `21de1b20b`).  **Still to do:** the default gather/staging gate sends qwen4exp `-ub 2048` to
*staging* (641); the gather (2401) needs `GGML_SCHED_STAGE_MIN_TOKENS=999999` or a model-aware gate.
And the beta5 tree still carries temporary A/B knobs (`GGML_META_GATHER_*`) to strip before promotion.
**Repo:** `~/llama-cpp-rdna-boosts` (branch `promote-moe-caching`); campaign tips in `~/llama-fold`.

---

## 0. The goal (the actual "halo")

**One R9700 (32 GiB)**, **Qwen3.8-Flash-Next IQ4_NL** (96 GiB — does not fit), **good prefill and >35 t/s
decode at the same time**, no MTP.

`MOE_EXPERT_CACHE_MIB=12288` is the working hypothesis for the middle ground: it leaves VRAM for the
static weights, the KV cache and the prefill workspace while still holding a useful expert arena.

**Focal target: `-b 2048 -ub 2048`.**  Fall back to `-ub 1024` only if 2048 cannot fit.

**The single command to judge any candidate (measures both in one process, so the cache warms from the
prefill and the decode runs over the real 8K KV):**

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
M=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=12288 MOE_EXPERT_CACHE_DEVMAP=1 \
  <BIN> -m $M -ngl 99 -ncmoe 48 -sm layer -fa 1 --lazy-mode auto --load-mode none -t 8 \
        -p 8192 -n 1024 -b 2048 -ub 2048 -r 2
```

**Success:** `pp8192` high (≥ ~1400) **and** `tg1024` ≥ ~35 in the same run.

## 0.5 ROOT CAUSE FOUND + VERIFIED FIX (2026-10-01)

**The prefill→decode collapse is the deferred arena sizing latching on an incomplete
`g_tables` set.**  Evidence on `r26-b1 6ca5c1c77` (single GPU, MIB=12288):

* `-p 0` (no prefill): `alloc_all_locked: sized 97 slots/table from 144 tables`, `takeover=144`, decode **36**.
* `-p 512`: `alloc_all_locked: sized 512 slots/table from 3 tables` (layer 47 only!), `takeover=1`, decode **8.54**.
* `MOE_EXPERT_CACHE_SLOTS=97` (explicit slots, deferred sizing skipped): `-p 512` decode **36.5**.
* `GGML_SCHED_DEVGATHER=0`: 144 tables sized, decode **32.6** (but prefill collapses to 162).

**Mechanism.**  A table is registered by the host upload hook (`moe_cache_update_host`).  The
**device gather** (`moe_cache_gather_host`, scheduler branch `ids_tensor->ne[1] > 8`) uploads *without
registering*.  So with the gather on, a prefill registers only the layers that fall off the gather —
in practice just the last layer (`layer=47`).  `alloc_all_locked()` then fires on the first table's
second hooked call and **latches `g_sized=true`**, sizing the arena from the 3 tables registered so
far (layer 47 → `512/512` fully resident, 1350 MiB).  Layers 0-46 register afterwards, find no
`layer_slots` entry, get **0 slots**, and decline: the decode falls off the cache, and because the
arena is partial the cache-band fusions stand down globally — **8.54 t/s, below the uncached 20**.

**Fix (one call).**  Register the table on the gather path too: a `moe_cache_table(...)` call in
`moe_cache_gather_host`, mirroring `moe_cache_update_host`'s geometry (`host_bytes`, `expert_bytes`,
`host_pitch`, `slice_off`, `split_axis`).  Registration is a cheap map insert; with deferred sizing it
does not allocate.  Upload path must never decide registration.

**Result (r26-b1 + fix, single GPU, `-b 2048 -ub 2048`, MIB=12288):**

| `-p` | sizing | `pp8192` (for p=8192) | `tg1024` |
|---|---:|---:|---:|
| 0 | 97 slots / 144 tables | — | 35.9 |
| 512 | 97 slots / 144 tables | 1232 | 36.3 |
| 2048 | 97 slots / 144 tables | 2145 | 35.9 |
| 8192 | 97 slots / 144 tables | 2197 | 35.4 |

**This is the middle ground: ~2.1-2.2k prefill t/s at `-ub 2048` AND ~35-36 decode t/s in the same run.**
The fix removes the prefill/decode anti-correlation by decoupling registration from the upload path.

Still to decide: the gather kernel itself is model-dependent (fast for qwen4exp, measured slower for
Q4_K_M `-sm layer` at large ub) — see §5.  With registration fixed, the gather can run at every width,
so the question is only whether the model should *choose* gather vs staging.

## 0.6 SEPARATE PREFILL REGRESSION — still open (2026-10-01)

With the §0.5 registration fix applied to **beta4**, the decode is fixed (38.7 t/s, 144 tables sized) but
the **prefill does not recover**: beta4+fix `pp8192` = 615-648, whereas b26 (r26-b1) = **2145** at the
same `-b 2048 -ub 2048`/MIB=12288.  So the prefill regression is *independent* of the decode collapse.

Facts gathered:

* Lifting the gate does nothing: beta4+fix `GGML_SCHED_STAGE_MIN_TOKENS=999999` (gather everywhere)
  = 648; `GGML_SCHED_DEVGATHER=0` = 591; default = 615.  So the gather/staging gate is **not** the
  qwen4exp prefill lever on beta4.
* On b26 the gather **is** the lever: `DEVGATHER=1` 1234 vs `=0` 600 (same `-p 8192 -n 32 -r 1`).  So the
  gather's ~2x benefit is present in b26 and ~absent in beta4.
* The gather kernel + `moe_cache_gather_host` are **byte-identical between b26 and beta4 except the
  session-18 MMQ tail guard** (`pad = min(expert_bytes,512)`, 512 bytes written after each routed
  expert only when `s==0`).  That is far too small to cost 2x, so the gather is not *slow* in beta4 —
  it is *ineffective* (something around it changed).
* B1's win was **three** things: (1) the device gather + deferring routed tables from whole-tensor H2D
  staging, (2) **per-split scheduler events ON** (kills thousands of full device syncs), (3) **async
  host->device split-input copies**.  The commit chain b26 -> session21 includes
  `c135d71ca` *"exclude gemma4 from -sm tensor; park the segmented async upload"* and
  `171b7e18e` *"rebase onto r28 + adaptive staging-vs-gather probe (GGML_SCHED_STAGE_AUTO)"*.
  The most likely prime suspect is that (2)/(3) were parked/disabled and beta2 removed the AUTO probe.

### Bisection chain (campaign, `~/llama-fold`)

```
b26 6ca5c1c77        r26 + B1 (session 17)            FAST 2145   (fix applied)
 d76e18efd           item1 tail-pad
 0cdcf89d9           B3 devmap
 f5078dc64           default DEVMAP ON
 ed54ac913           item3 diag + stage_gather + meta segmented set_tensor_async
 b760ba152           meta segmented async tail pad
 c135d71ca           exclude gemma4; PARK the segmented async upload
 171b7e18e           rebase onto r28 + AUTO probe
 f5a79e6ab           item C (session 21)              SLOW ~600
```

Build each in a worktree and run the §0 combined command at `-p 8192 -n 32 -r 1` (fast) to find the
first slow tip.  Likely the regression lands on or before `171b7e18e` (r28 rebase) or at `c135d71ca`
(parked async upload).  Note the two branches diverge: `6ca5c1c77` is the **r26** B1, `0ffb3b13f` is
the **r25** B1; the chain to `f5a79e6ab` runs through `0ffb3b13f`.

### Lowest-risk delivery fix candidate

b26 is `r26 + B1`; beta4 is `r28 + campaign + beta`.  The middle ground exists on **b26 with the §0.5
registration fix**.  Restoring it in the delivery means finding which of B1's three halves was lost in
sessions 18-21 (or the r28 rebase) and restoring it — the gather kernel itself is unchanged.

## 1. The key finding: the two sides are anti-correlated

Combined run (`-p 8192 -n 1024 -b 2048 -ub 2048`, `MIB=12288`, single GPU):

| tree | date | `pp8192` | `tg1024` (post-prefill) | `tg1024` standalone |
|---|---|---:|---:|---:|
| **B1 `6140bba76` (r25+campaign)** | 09-30 02:16 | **1933-1995** | **8.53** | ~42 |
| **r26-b1 `6ca5c1c77`** | 09-30 06:17 | **1865** | **8.53** | ~42 |
| campaign session21 `f5a79e6ab` | 09-30 18:15 | ~600 | — | — |
| **beta4 `f29745280`** | 10-01 04:50 | **601-617** | **38.9** | ~39 |

So **B1/r26-b1 have great prefill but collapse to 8.53 t/s decode once the prefill has run**, and
**beta4 has good decode but ~3× worse prefill**.  Neither is the middle ground.  This is exactly the
"each side was optimised independently" diagnosis.

**The decode collapse is NOT the expert cache.**  B1 `-v` report after the prefill:

```
moe_cache_report: h=1.0000 (10/10 reaches), fills=1536 evictions=0, tables=3, slots=1536, arena=1350.0 MiB
moe_cache_report: takeover=1 decline_all=0 ... expert access: hits=10 fills=0 colds=0 (hit=100 %)
moe_cache_report: VRAM budget: requested=12288 MiB/device, free-clamped=0, arena alloc failures=0
```

Fully resident, 100 % hits, no cold reads, no allocation failures — yet decode is 8.53.  So the collapse
is a **long-context / post-prefill state** effect (KV/attention, graph allocation, residency of the
non-expert weights, or the prefill's staging arena lingering), **not** cache misses.  (Note also: only
**3 tables** show as sized — layer 47 — with a 1350 MiB arena on a 12288 MiB budget; that itself may be a
bug or a deferred-sizing artifact worth checking.)

Also note the standalone-vs-combined gap: B1 standalone decode is ~42, combined is 8.53.  The collapse
appears only once the 8K KV exists.

## 2. Timeline / bisection tips (all in `~/llama-fold`)

```
r25 delivery       81fda69c8   2026-09-29 13:20   prefill ub2048 ~?
B1  (r25+campaign) 6140bba76   2026-09-30 02:16   pp ub2048 1715, tg standalone ~42, combined tg 8.53
r26 delivery       0d58404e1   2026-09-30 05:55   pp ub2048  567  (no campaign)
r26-b1             6ca5c1c77   2026-09-30 06:17   pp ub2048 1865, combined tg 8.53   <-- START HERE
session 18         2376ac6cf   (gather tail-pad)
session 20         51b1f48be   2026-09-30 11:21
session 21         f5a79e6ab   2026-09-30 18:15   pp ~588
r28 delivery       60361cb9f
beta1              ce06f7add   2026-09-30 21:34
beta2              0f77c32d   2026-09-30 21:59
beta3              5bbba5d64   2026-09-30 22:16
beta4              f29745280   2026-10-01 04:50   pp 557, combined tg 38.9
```

Build a tip into its own worktree (ccache makes a re-build of an already-built tree seconds):

```bash
git -C ~/llama-fold worktree add /tmp/bisect-<tip> <tip>
cd /tmp/bisect-<tip>
BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
```

Already built during this session: `/tmp/b1` (`6140bba76`), `/tmp/b26` (`6ca5c1c77`), `/tmp/reorg` (beta4).
Remove with `git -C ~/llama-fold worktree remove /tmp/<name> --force` when done.

## 3. Plan

1. **Start from `6ca5c1c77`** (r26-b1 — good prefill).  Reproduce the combined run and diagnose **why
   decode collapses after the prefill**:
   - Is it prefill-specific or long-context-specific?  Compare `-p 8192 -n 1024` vs `-p 0 -n 1024` vs
     `-p 512 -n 1024` at the same MIB; sweep `-p` 512/2048/8192.
   - Run `rocm-smi` (VRAM used) + `llama-server`/`-v` around the boundary; is VRAM exhausted when the KV
     is large (forcing the non-expert weights or the arena to spill)?
   - Try `MIB=8192/12288/16384`, `GGML_CUDA_FA_KV_NATIVE=0`, `-fa 0`, smaller `-c` (does the KV budget
     drive it?), `GGML_SCHED_STAGE=0` (does the prefill's staging arena linger?).
   - Check the graph (`-v` `sched_reserve`): does the decode graph reallocate after the prefill?
   - Check whether the collapse is in the MoE op or in attention (op timing / `GGML_CUDA_OP_TIMING`).
2. **Then bisect the prefill regression** from `6ca5c1c77` → `51b1f48be` → `f5a79e6ab` → `0d58404e1`/r28,
   using the combined command at `-b 2048 -ub 2048`.  The prime suspect is the r26→r28 delivery staging
   rework + the campaign session-18..21 changes (see §4).
3. **Find the middle ground** and land it (beta5), re-running the combined command and the standard gates.

## 4. Code map + envs

* `ggml/src/ggml-backend.cpp` — `sched_stage_issue`, `sched_input_gatherable`, `sched_stage_min_tokens`,
  `sched_stage_batch_tokens`, the input-loop host-weight branch (`copy_experts`, the device gather).
* `ggml/src/ggml-cuda/moe-expert-cache.cu` — the cache engine: `moe_cache_gather_kernel`,
  `moe_cache_gather_host`, `alloc_all_locked` (sizing), `moe_cache_update_host`, the UVA cold path.
* `ggml/src/ggml-cuda/ggml-cuda.cu` — the CUDA cache iface, `ggml_cuda_cache_blocks_fusion`, `get_op_batch_size`.
* `ggml/src/ggml-backend-meta.cpp` — Meta delegation (`-sm tensor` only).
* `archive/work/tensor-split-expert-split/README.md` — the r16 staging design.
* `archive/work/moe-expert-cache/WORKLOG.md` — B1 (session 17), the r26 rebase addendum, the session-21d
  staging-vs-gather A/B, the CPU-computes-the-misses arm.

A/B envs: `GGML_SCHED_STAGE=0/1`, `GGML_SCHED_EVENTS=0/1`, `GGML_SCHED_DEVGATHER=0/1`,
`GGML_SCHED_STAGE_MIN_TOKENS=<n>`, `GGML_SCHED_STAGE_SLOTS=<n>`, `GGML_SCHED_STAGE_MODE=<0/1>`,
`LLAMA_MMAP_HOST_EXPERTS=0/1`, `GGML_CUDA_SPLICE_GATHER=0/1`, `GGML_CUDA_FA_KV_NATIVE=0/1`,
`MOE_EXPERT_CACHE_*`.  `GGML_SCHED_GATHER_FIRST` / `GGML_SCHED_STAGE_AUTO` were **removed in beta2**;
their originals are in `~/llama-decode` and can be restored if the model-dependent gather choice is needed.

## 5. Do-not-break list

* The **Q4_K_M** prefill gates (the beta4 fix): 2gpu tensor ncmoe 40 ≥ 5257, 2gpu layer ≥ 4511,
  1gpu ≥ 5398 at `pp8192 ub8192`.  The ungated gather cost Q4_K_M `-sm layer` `-ncmoe 40` **-42 %**; the
  fix must be model/probe-dependent (or arch-aware), **not** a blanket revert.
* Admission gates: `validate-set.sh`, byte-identity `de8be4d0c90c`, width purity `15038c19ddc8`,
  `MUL_MAT_ID` 929/929, MTP `n3`, deep coherence.

---

### Raw measurements (2026-10-01, single R9700, model as §0)

**Prefill only** (`-p 8192 -n 0 -b 8192`):

| build | ub512 | ub1024 | ub2048 | ub8192 |
|---|---:|---:|---:|---:|
| B1 `6140bba76` | 1043 | 1406 | 1715 | 1887 |
| r26-b1 `6ca5c1c77` | — | — | 1865 | — |
| session21 `f5a79e6ab` | 238 | — | 588 | 1116 |
| beta4 `f29745280` | 238 | 367 | 557 | 1064 |

**Combined** (`-p 8192 -n 1024 -b 2048 -ub 2048 -r 2`, `MIB=12288 DEVMAP=1`):

| build | pp8192 | tg1024 |
|---|---:|---:|
| B1 `6140bba76` | 1933 / 1963 (seed/no-seed) | 8.54 / 8.53 |
| B1 MIB=8192 / 16384 | 1983 / 1995 | 8.53 / 8.53 |
| r26-b1 `6ca5c1c77` | 1865 | 8.53 |
| beta4 `f29745280` | 607 / 617 | 38.9 / 38.9 |

**Decode only** (`-p 0 -n 1024`): B1 `MIB=8192/16384/24576` = 32.0 / 38.6 / 42.2 (seed+prov: 42.6);
beta4 `MIB=24576` = 39.2.  cache off = ~19-21 both.

B1 cache report (after the combined run, MIB=12288): `h=1.0000`, `arena=1350.0 MiB`,
`tables=3 slots=1536`, `free-clamped=0`, `alloc failures=0`, `hits=10 fills=0 colds=0`.
