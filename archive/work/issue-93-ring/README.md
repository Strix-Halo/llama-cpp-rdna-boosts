# Issue #93: auto-size the H2D staging ring budget (+ table-size-scaled gate)

Status: **PROMOTED (2026-10-05, release `v16-a55e952b8-r7`)**.  Folded into **block 06** (the
scheduler half, `ggml/src/ggml-backend.cpp`) and **block 15** (the ring + accounting) of the
canonical chain, new tip `27b6254e7b00ec7036ba7bf893a3ea5def7c22c2`, net tree
`77ee997c9fad231ea64ffb3a4247a1d158819b48` (`validate-set.sh` green, strict 16/16 `git am`).
Historical fork branch `wip/issue-93-ring` (tip `74b05d550`, on top of the r6 block-15 tip
`1d10390a8`); patch [`issue-93-ring.patch`](issue-93-ring.patch) (sha256
`a56871b51e7eaf31f282580fe498055140e2ad307524dfcca33f73ca03147502`).

## The problem (issue #93, @briansp2020)

With host-resident MoE experts (`-ncmoe`), qwen4exp's expert tables are **450 MiB**.  The
op-offload H2D staging ring was capped at the fixed `GGML_SCHED_STAGE_MAX_MB` default of
**2048 MiB** with 8 slots.  Eight 450 MiB slots need ~3.6 GiB, so the fifth growth tripped the
budget, `h2d_stage_buffer()` returned NULL, and the scheduler set `stage_enabled = false` **for
the rest of the run** — every remaining MoE split fell back to the serial pruned host-copy path
(0 % copy/compute overlap; the reporter's rocprof trace: 30 % H2D, 0 % overlap).  A bigger ring
(`GGML_SCHED_STAGE_MAX_MB=8192 GGML_SCHED_STAGE_SLOTS=16`) recovered **+42 %** at `-ub 2048` on
their PCIe5 x16 box.

Two bugs interact:

1. **Budget default too small for large tables.**  The default was a fixed 2048 MiB, unrelated to
   the table size.  A large table silently disabled the ring.
2. **The staging width gate is table-size-independent.**  `sched_stage_min_tokens()` is calibrated
   from the link bandwidth against a **144 MiB** reference table (the campaign's Q4_K_M tables).
   For a 450 MiB table the whole-table copy is ~3x the bytes, so the true "staging beats pruning"
   crossover is at a **wider** batch.  On a slow link (x4) staging a 450 MiB table at `-ub 2048`
   *loses* even when the full ring fits.

## The solution

`wip/issue-93-ring.patch` (6 files, +221/-31), against r6 (`1d10390a8`):

1. **`ggml/src/ggml-cuda/common.cuh` — auto-sized ring budget.**
   `GGML_SCHED_STAGE_MAX_MB` now defaults to **auto**: when unset,
   `h2d_stage_budget()` returns `h2d_stage_slots() x (largest slot ever requested + 512)`, so the
   ring can always hold a full depth of whatever table it is feeding.  An explicit
   `GGML_SCHED_STAGE_MAX_MB` still wins (A/B / VRAM control).  `h2d_stage_slots()` is the shared
   `GGML_SCHED_STAGE_SLOTS` parse (default 8, clamped to `H2D_STAGE_SLOTS`).

2. **`ggml/src/ggml-backend.cpp` — table-size-scaled gate + graceful fallback.**
   `sched_stage_min_tokens_for_bytes()` scales the link-calibrated base threshold by
   `host_table_bytes / 144 MiB` (`GGML_SCHED_STAGE_TABLE_REF_MB` overrides the reference; `0`
   disables the scaling and reproduces the old width-only gate).  `sched_stage_issue()` now plans
   the **whole split** before issuing any upload: if any slot cannot be allocated it skips the
   split (never partially staged, which the campaign measured as worse than either path) but
   **does not disable the ring** — the next split tries a different slot, so a VRAM- or
   budget-limited ring degrades to a partial pipeline instead of the old "off for the rest of the
   run" cliff.

3. **`--fit` / memory-breakdown accounting.**  The ring is a raw device allocation outside the
   compute-graph reserve, and auto-sizing can grow it to several GiB.  `h2d_stage_bytes()` /
   `h2d_stage_bound()` are exposed through the CUDA reg (`llama_model::max_host_weight_tensor_bytes()`
   supplies the bound), and `llama_context::memory_breakdown()` counts the ring in both the live
   and the `no_alloc` (fit) paths.

**The gather is not touched and stays default-off.**  The `////` corruption was the persistent
*device gather* (`moe_cache_gather_host`), not the staging ring; `GGML_SCHED_DEVGATHER` is
unchanged (off unless explicitly set).  The ring is the host-copy path (`copy_experts`), which
copies the guard pad every pass.

## Validation (soar, gfx1201 / ROCm 7.14, 1x R9700, PCIe5 x4)

Model: `Qwen3.8-Flash-Next-UD-IQ3_XXS` (450 MiB expert tables), `-ncmoe 48 -sm layer -fa 1
--load-mode none`.  **All numbers with `-lzm off`** — the managed lazy reader re-faults the PLE
weights during prefill and makes the run-to-run noise larger than the effect.

`llama-bench -p 8192 -n 0 -b 8192 -ub <U> -r 3 -t 8`:

| `-ub` | serial (`GGML_SCHED_STAGE=0`) | staged (`=1`, auto ring) | default (auto) | gate |
|---:|---:|---:|---:|---|
| 2048 | 785 | 789 | 791 | gated (serial) — tie |
| 4096 | 1142 | 1071 | 1065 | gated (serial) — staging would lose 6 % |
| 8192 | 1570 | **2374** | **2374** | staged — **+51 %** |

The r6 default (fixed 2048 MiB) was the **serial** column at every width (the ring disabled
itself), so the fix is ~+51 % at `-ub 8192` on this box and leaves `-ub 2048/4096` alone.

**Purity.**  Same-seed greedy, 5246-token prefill + 64 tokens, `--seed 42 --temp 0`:
`GGML_SCHED_STAGE=0` and `=1` are **byte-identical** (`9b38c3005063`, 0 `////` runs).  No decode
corruption.

**Regression gates.**  `test-backend-ops -o MUL_MAT_ID` **OK** (ROCm0); dense 4B same-seed
`1c5d32ac537d` (matches r6).

**Reference model (144 MiB tables, where the r6 ring already fit).**  Qwen3.6-35B-A3B UD-Q4_K_M,
`-ncmoe 99 -sm layer`: `-ub 8192` 3059 -> 4002 staged (the pre-existing ring win); `-ub 2048`
1444 -> 1504 (the scaled gate for a 144 MiB table equals the unscaled base, so behavior is
unchanged).  No regression.

**`-sm tensor` (meta per-device ring), 2x R9700.**  qwen4exp IQ3_XXS `-ncmoe 48`: `-ub 2048`
1006 -> 1006; `-ub 8192` 1887 -> **2268** (+20 %, meta staging is noisier).

**Escapes verified.**  `GGML_SCHED_STAGE_TABLE_REF_MB=0` reproduces the old width-only gate
(`-ub 2048` stages: 525 vs 653 gated); an explicit small `GGML_SCHED_STAGE_MAX_MB` degrades
gracefully instead of disabling the ring.

## Reporter confirmation (2026-10-04, @briansp2020, R9700 on PCIe 5.0 x16)

Built r7 from the repo patches onto `a55e952b8` (tree `77ee997c`, matching `release.json`).  Model
Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, `-ncmoe 48 -sm layer -fa 1 -lzm off`, `llama-bench -p 8192
-n 0 -r 3`, two interleaved passes, no `MOE_EXPERT_CACHE_MIB` and no ring env vars (auto budget).
The link calibrates at 55.4 GB/s.  They could not use `-lm none` (the 40.9 GB of host experts plus
the 26.8 GB embedding table exceed the box's 59 GiB of RAM), so the runs use the default load mode.

| `-ub` | serial (`GGML_SCHED_STAGE=0`) | staged (auto) | change |
|---:|---:|---:|---:|
| 2048 | 1579 / 1575 | 2133 / 2131 | **+35 %** |
| 4096 | 1909 / 1917 | 2370 / 2370 | **+24 %** |
| 8192 | 2206 / 2204 | 2375 / 2373 | **+8 %** |

Slot sweep (`-ub 2048`, auto budget): 4 slots 2117 / 2130, 8 slots 2128 / 2131, 16 slots
2137 / 2139 (depth is within noise once the ring holds a full table per slot).

Output (`llama-server -c 65536 -np 1 -ub 2048`, no MTP, `-lzm off`, temperature 0): greedy output
identical staged vs serial; three long prompts of 11k / 22k / 34k tokens byte-identical staged vs
serial (`79adea679b5c`, `d2320d242c1b`, `71e7c37e26ac`); prefill 1052-1280 -> 1508-1724 t/s; no
"staging disabled" lines.

**Gather note.**  The reporter tried zeroing the 512-byte expert heads on every gather instead of
once per buffer (a 14-line `moe-expert-cache.cu` change, offered as a follow-up) and with it
`GGML_SCHED_DEVGATHER=1` reproduces the staged outputs exactly; corrected, the gather is ~35 %
slower than staging for long prefill on this link.  This confirms the 3052 vs 1403 figure cited in
`SCHED_GATHER_TABLE_MIN_BYTES` is from the corrupted pass; the gather stays default-off.

## Open follow-ups

* Take the reporter's default-off device-gather correctness change when offered, and correct the
  stale corrupt-pass figure in the `SCHED_GATHER_TABLE_MIN_BYTES` comment (no default changes).
* The auto budget is deliberately not free-VRAM-capped (the WIP author found a cap re-disabled
  staging); the `--fit` accounting is the guard instead.  A backend-reported effective slot count
  (cycle only the slots that fit) would be a cleaner follow-up.
