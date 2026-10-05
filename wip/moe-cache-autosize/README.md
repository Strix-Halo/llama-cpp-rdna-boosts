# MoE expert cache: auto-enable + auto-size (`MOE_EXPERT_CACHE_MIB`)

**Status: PROMOTED to the delivery as part of block 13 in `v16-a55e952b8-r14` (2026-10-05).**  The
campaign record below is kept as provenance.  Maintainer direction: if `-ncmoe > 0`
and the user did not set `MOE_EXPERT_CACHE_MIB`, the cache should arm itself and size itself from the
hardware, so the "administrative" load leaves the user. This is the core focus of the campaign.

Related: `COMMUNITY-CONFIG.md` (the manual config guide), `archive/work/moe-expert-cache/` (the
campaign that built the cache), `GREEDY-PURITY.md` §19/§24/§25, `patches/README.md`
(block 06/13/15 notes), `wip/moe-cpu-overlap/` (the other half: making the miss path overlap).

## Session 4 (2026-10-05): preflight refinement + reserve grid + promotion gates

### Preflight refinement: aux-context reserve (DONE)

The preflight ran before the MTP draft context existed, so its `free - reserve` projection over-shot the
real arena by the draft's own memory.  Measured gap (single R9700 IQ3_XXS auto): **3655 MiB @c8192,
3661 @c32768, 3956 @c131072** -- the draft's compute + KV + head, weakly context-dependent.  The
preflight now subtracts a per-device `aux_reserve_bytes`, supplied by `common` (4096 MiB when an MTP/draft
context is configured, else 0; `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` overrides).  Projected vs actual:
**19666 vs 20107 MiB** -- now conservative (the safe direction).  Plumbed through the `moe_cache_preflight`
device iface + `llama_model_moe_cache_preflight(model, aux_reserve_bytes)`, so it works whether or not
`--fit` ran.  (An exact per-device figure would need the draft to exist first -- exactly what the early
decision avoids -- so a fixed reserve is the pragmatic answer.)

### Reserve grid (DONE -- no OOM, arena yields)

`wip/moe-cache-autosize/reserve-grid.sh` + `reserve-grid.log`, single R9700 IQ3_XXS, auto mode:

| ctx | ub | MTP | draft-offload | preflight | arena | t/s |
|---|---:|---|---|---:|---:|---:|
| 8192 | 2048 | on | 1 | 19666 | 20108 | 32.8 (n=32; 48.6 at n=128) |
| 32768 | 2048 | on | 1 | 19310 | 19746 | 32.2 |
| 131072 | 2048 | on | 1 | 17884 | 18025 | 30.1 |
| 32768 | 4096 | on | 1 | 19310 | 19746 | 32.2 |
| 32768 | 8192 | on | 1 | 19310 | 19746 | 32.2 |
| 32768 | 2048 | on | 0 | 19310 | 19746 | 32.1 |
| 32768 | 2048 | **off** | 1 | 23744 | 23640 | 39.4 |
| 131072 | 8192 | on | 1 | 17884 | 18025 | 30.1 |

Every run rc=0, **no OOM** (the only "abort" text is `--fit` declining to move `-ngl 99` layers).  The
arena tracks `free - reserve - aux` closely and yields to the context (`-ub` does not change it because
the prefill compute buffer is allocated before the arena).  The n=32 `t/s` is warm-up-inclusive.

### Promotion gates (all green on the WIP build)

| gate | result |
|---|---|
| **`-ncmoe 0` byte-identity** (3-GPU `-sm tensor`, Q4_K_M, fully resident) | auto == off == `MIB=8192`, **byte-identical response** (23c9baad289a) |
| **width purity** (1-GPU `-sm layer -ncmoe 48`, IQ3_XXS) | `none == n1 == n3 == n7` (1024196129a5) |
| **long MTP acceptance** (1 GPU, `-n 3000`, `code-reasoning-mixed`) | **50.1 t/s**, `////`=0, 2017/2944 = **0.685** >= 0.45 |
| **long MTP acceptance** (2 GPU IQ4_NL, `-n 3000`) | **65.1 t/s**, `////`=0, 1802/2481 = **0.726** >= 0.45 |
| **coherence** (1 GPU, c32K / c128K, n=64) | `////`=0 / 0 (39.5 / 37.0 t/s) |
| 3-GPU IQ4_NL `-sm layer` (n=256) | 81.3 t/s, `////`=0 |

Gotcha for text comparison: `llama-cli`'s stdout carries the *variable-length loading spinner*, so a raw
`sha256` of the output differs between runs even for identical text.  Extract the response with
`sed -n '/^> /,/\[ Prompt:/p' | sed '1d;$d'` before hashing/diffing.  (A `--spec-type none` run must not
also pass `-md`; that combination exits the server.)

### Conclusion

Auto mode (`unset == auto`, `0 == off`, `>0 == fixed`) is **promotion-ready on the gates above**: a large
consistent win on the oversized-MoE `-sm layer` + host-experts cases, never a regression on the
fully-resident / explicit-MIB paths, and neither small-arena nor below-floor MTP trap remains.  Before it
leaves `wip/`: the maintainer's go-ahead, a delivery block and/or the `upstream/` copy (the change is
generic), and a server-concurrency + `--fit` matrix re-run.

## Session 3 (2026-10-05): early floor decision implemented + `--fit`/arena analysis

### Early floor decision (DONE, validated)

`auto-mode.patch` now also adds an **early preflight** so the auto floor decision happens before the
MTP draft context exists:

* new optional device iface `moe_cache_preflight(dev, host_expert_bytes)` (`ggml-backend-impl.h`,
  `ggml_backend_dev_moe_cache_preflight`);
* `llama_model_moe_cache_preflight(model)` (`src/llama-model.cpp`) sums the host-resident `*_exps`
  tensor bytes per device from `tensors_by_name` and calls the iface;
* `common/common.cpp` calls it right after the target context is created in `common_init_result` -- i.e.
  **after** `--fit`, the KV and the compute reserve are paid, and **before** `common_speculative_init`
  creates the draft context;
* `moe_cache_preflight` (cache) latches the first call: in auto mode, if the projected arena
  (`free - reserve`) is below `max(MOE_EXPERT_CACHE_MIN_MIB, MOE_EXPERT_CACHE_MIN_RES_PCT% of the host
  expert bytes)` it sets `g_enabled = false`; an explicit `MOE_EXPERT_CACHE_MIB` is left verbatim.

`MOE_EXPERT_CACHE_MIN_RES_PCT` defaults to **18** (the measured MTP crossover) and `..._MIN_MIB` to 0.

**Measured (single R9700 IQ3_XXS `-ncmoe 48`, MTP n3, c8192, n=128):**

| case | before (post-hoc disable) | **with preflight** | cache-off |
|---|---:|---:|---:|
| forced below-floor (`MOE_EXPERT_CACHE_MIN_MIB=999999`) | 9.5 | **34.7** | 34.6 |
| auto (arena 20107 MiB, 43 % res) | 48.6 | **48.6** | 33.9 |
| explicit `MIB=20480` (preflight skipped) | 48.6 | **48.6** | - |
| 2G IQ4_NL `-sm layer` auto | 57.8 | **57.8** | 31.3 |
| fully-resident 4B `-ncmoe 0` (no host table) | 95.1 | **94.5** | 94.6 |

The preflight-fixed below-floor path now matches cache-off, so the auto floor no longer has the MTP
regression.  The projected arena at preflight (23762 MiB here) still over-estimates the final one
(20107) by the draft context's own memory; the 18 % floor absorbs that, but a larger draft (or a very
small margin) could still overshoot -- a future refinement is to subtract the draft's measured memory.

### `--fit` vs the arena (measured)

Single R9700, IQ3_XXS, `-ncmoe 48`, MTP n3, auto, `--fit` **on** (default):

| config | n_ctx chosen | auto arena | residency |
|---|---:|---:|---:|
| no `-c` (fit picks) | **4096** (its min) | 20198 MiB | 43.6 % |
| `-c 32768` | 32768 | 19746 MiB | 42.6 % |
| `--fit off` | 262144 | 15488 MiB | 33.4 % |
| `--fit-target 8192` | 4096 | 20198 MiB | 43.6 % |

So on these Flash-Next models the conflict is **mild** -- the KV is small (QSA/sparse; ~18 KiB/token)
next to the 46 GiB host expert set, so the arena keeps 33-44 % even at the full 256K context.  The
feared "`--fit` eats the arena" does not happen here; `--fit` actually picks the *minimum* context
(4096).  It can happen for a large-KV model.

### `--fit` + arena: the judgment call (needs a maintainer decision)

The arena is sized **after** `--fit` from `free - reserve`, so `--fit` has no idea it exists.  Three
policies:

* **(a) Reserve a slice for the arena in `--fit`'s margin** (`fit_params_target`): add
  `MOE_EXPERT_CACHE_FIT_MIB` (or a fraction) per device when `-ncmoe > 0` and the cache is auto, so
  `--fit` sizes context around it.  Predictable arena, but silently takes context -- and `--fit` is on
  by default, so it would change every `--fit` run.
* **(b) Prefer context; let the preflight disable the cache when starved** (implemented, zero extra
  code).  `--fit` keeps its contract (max context that fits), and a tight GPU simply gets the CPU
  expert path -- which the preflight now proves is as fast as cache-off, not the old 9.5 t/s trap.
* **(c) Hybrid**: reserve only the *floor* (e.g. `MIN_RES_PCT` of the host experts, or `MIN_MIB`) in the
  `--fit` margin, so an arena that fits is always useful, and let the preflight disable below it.

**Chosen: (c), implemented (2026-10-05).**  `common/fit.cpp` reserves `floor = max(MIN_MIB,
MIN_RES_PCT% x host_expert_bytes[d])` per device in `--fit`'s margin (only when `MOE_EXPERT_CACHE_MIB`
is unset and host experts exist); `--fit` then sizes the context around it, and the arena takes any
remaining free device memory down to `MOE_EXPERT_CACHE_RESERVE_MIB`.  The host-expert bytes come from
the loader (`create_tensor` accumulates host expert weights per device into
`llama_model::moe_host_expert_bytes`), so they are correct even under `--fit`'s `no_alloc`
measurement; `llama_model_moe_host_expert_bytes(model, dev)` exposes them.  An explicit
`MOE_EXPERT_CACHE_MIB` skips the reservation (verbatim), and a fully resident model is untouched.

**Measured (single R9700, IQ3_XXS, `-ncmoe 48`, MTP n3, auto):** floor 8348 MiB reserved; `--fit`
still picks n_ctx 4096 (its min) and the arena is 20198 MiB (43.6 %).  With `--fit off` the arena is
15488 MiB (33.4 %).  2 GPU IQ4_NL: floor 11664 MiB (18 % of 64800), arena 50538 MiB (78 %), 57.6 t/s.

**Warnings (at WARN, unchanged).**  `common_params_fit_impl` logs a WARN naming the reserved floor
(and the host-expert total and the reserve); `alloc_all_locked` logs a WARN with the actual arena size
and residency, also when `--fit` is off (so the arena size is always stated).  **Decision (maintainer,
2026-10-05):** keep them at WARN and leave llama-cli's default verbosity untouched (upstream's
`LOG_LEVEL_ERROR`) -- llama-cli does not announce warnings at its default verbosity (the same as every
other warning upstream), while **llama-server's default (`LOG_LEVEL_INFO`) shows them**; `-lv 3` in
llama-cli reveals them too.  No CLI log-level change.  A side effect to watch: `--fit`'s margin also
drives its layer placement, so the reservation can move a few dense layers to the CPU as well as shrink
the context.

### Remaining work

1. Refine the preflight projection to account for the draft context's memory.
2. The reserve grid (ctx x ub x MTP x draft-offload) and the full promotion gates.

## Session 2 (2026-10-05): auto mode implemented + M0 re-baselined on r13

**Auto mode is implemented** in `~/llama-r13` (WIP, uncommitted): `ggml/src/ggml-cuda/moe-expert-cache.cu`
only, +106/-22, patch `auto-mode.patch` beside this file. Semantics exactly the locked ones:

* `MOE_EXPERT_CACHE_MIB` set (incl. `0`) -> verbatim; `0` -> off (kill switch); `>0` -> fixed MiB/device.
* unset -> **auto**: `g_enabled = true`, `g_budget = SIZE_MAX`; each device is sized at
  `alloc_all_locked` from its own `free - reserve` (the budget is already per device).
* fully-resident (`-ncmoe 0`) -> no host table registers -> the cache stays inert; `moe_cache_ready()`
  now returns true when there are no tables so CUDA-graph capture is not held off.
* `MOE_EXPERT_CACHE_MIN_MIB` (default **0**) is an optional floor; see "the floor is a trap" below.

**Measured, `--load-mode auto` (the default; avoids the `--load-mode none` crash):**

| config (MTP n3, c8192 q8_0, n=128 unless noted) | cache OFF | manual best | **AUTO** |
|---|---:|---:|---:|
| 1G IQ3_XXS `-ncmoe 48` | 32.6 | 48.7 (MIB=20480) | **48.6** (20107 MiB, 43 % res) |
| 2G IQ4_NL `-sm layer -ncmoe 48` | 31.3 | 55.5 (MIB=24000) | **57.8**; **62.3** at n=256 (77.8 % res) |
| 2G IQ4_NL `-sm tensor -ncmoe 24` | 45.3 | 46.5 (MIB=8192) | **46.0** |
| 3G IQ4_NL `-sm layer -ncmoe 48` | 30.4 | 81.1 (MIB=24000) | **81.3** |
| 3G IQ4_NL `-sm tensor -ncmoe 24` | - | - | 39.7 |

The old 2-GPU tensor cache **regression (34.6 vs 43.2)** did **not** reproduce with the layer fix +
`--load-mode auto`: tensor auto 46.0 vs off 45.3 (a small win). Correctness: 2G auto coherent
(`////`=0), accepted 185/208 at n=256 (same as manual), **`-ncmoe 0` is byte-identical to cache-off**
(4B Q8_0, 95.0 vs 95.1 t/s, identical text).

**M0 re-baseline note:** the earlier M0 numbers used `--load-mode none` and the pre-fix `-sm layer`
(tensors all on GPU 0, 10.3 t/s). With block 16, `-sm layer` + a distributed arena is the 2-/3-GPU
winner. `-sm tensor` with host experts is essentially flat (auto ~= off) here.

### The floor is a trap (top open issue)

Under **MTP**, a small arena is far **worse** than the CPU path, and a bigger floor is needed than the
old ~1.2 GiB claim. Single R9700, IQ3_XXS, `-ncmoe 48`, MTP n3, cache-off = 32.0:

| MIB | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 6144 | 8192 | 12288 | 16384 | 20480 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| t/s | 4.4 | 14.4 | 14.7 | 15.5 | 16.3 | 18.9 | 23.7 | 27.0 | 30.6 | 36.9 | 42.7 | 48.7 |

The cache crosses **cache-off at ~8192 MiB (~18 % residency)** for this model under MTP. So "arm
whenever `-ncmoe > 0`" is only safe because auto sizes to `free - reserve` (usually a high residency);
a small auto arena (a nearly-full GPU) would regress. Hence the default floor is still contentious.

**Why the floor cannot simply "disable below the floor":** `moe_cache_enabled()` drops to false but
the *slow state is already baked in*. Measured `-ncmoe 48` IQ3_XXS, MTP: enabled-with-0-slot-arena
(`MIB=64`) = 4.4; enable-at-init-then-disable (`MIN_MIB=999999`, which logs "disabling the cache") =
**9.5 vs cache-off 33.1** (byte-identical text, so correctness holds; it is purely a scheduling cost).
Plain decode is almost fine (decline 21.1 vs off 22.2). It is **MTP-specific**: a stale/differently
sized MTP draft context (compute buffer 3222 vs 2972 MiB) is created while the cache is enabled and the
later disable does not undo it. Not CUDA graphs (`GGML_CUDA_DISABLE_GRAPHS=1` unchanged), not DEVMAP,
not `LLAMA_MTP_DRAFT_OP_OFFLOAD`.

**Consequence:** the shipped default is `MOE_EXPERT_CACHE_MIN_MIB=0` (always arm) plus a **low-residency
warning** (`< 20 %`), so a nearly-full GPU is loud rather than silently slow. The floor/disable path
needs the auto decision to happen **before the draft context is created** (a llama.cpp-side hook, or an
early sizing at the target's `sched_reserve`) -- see the next-steps list.

### Next steps (revised)

1. **Early floor decision**: decide enable/disable before the MTP draft context exists (either a
   `llama.cpp` hook after model load that tells the backend the host-expert byte count, or trigger
   `alloc_all_locked` at the target's `sched_reserve`). Gate: below-floor MTP == cache-off t/s.
2. **Residency-exponentiated floor**: the crossover is a residency fraction (~18 % here), not an
   absolute MiB; express `MOE_EXPERT_CACHE_MIN_MIB` (or a new `..._MIN_RES`) as a fraction.
3. **Cache-vs-MTP quant dependence** (still open): single-GPU IQ4_NL MTP is flat (33.6 -> 33.4) while
   IQ3_XXS MTP is +49 % (32.6 -> 48.7). Both have the same arena MiB; IQ4_NL experts are bigger
   (0.88 vs 0.63 MiB) so its residency is lower. See `IQ4_NL` absent from block 13's multi-row table.
4. **The reserve grid** (ctx x ub x MTP x draft-offload) -- not done this session; needed before a
   large default arena can be trusted at `-ub 8192` / 128K.
5. Promote only after `-ncmoe 0` oracles, `W=1..8` purity, MTP acceptance >= 0.45 at `-n 3000`, the
   coherence grid, and the 3-GPU check. Work stays in `wip/` with `MOE_EXPERT_CACHE_MIB=0` as the kill
   switch.

## Handover (2026-10-05, end of session) — read this first

**Where the campaign stands:** the auto-enable/auto-size design is agreed but **not implemented**.  M0
measurements were taken before the `-sm layer` fix below, so a large part must be re-measured.

### What changed this session (affects the campaign)

* **A multi-GPU `-sm layer` + host-expert bug was found and FIXED in the delivery as block 16**
  (`v16-a55e952b8-r13`, `patches/0016-*`, `WORKLOG.md` 2026-10-05 (r13),
  `wip/layer-split-host-experts/`).  Every host-expert MoE op was routed to device 0 because the CUDA
  host buffer type was a device-0 singleton, the loader's `ctx_key` merged same-name bufts, and the
  scheduler's op-offload loop returned the first capable backend.  After the fix, 2-GPU IQ4_NL
  `-sm layer -ncmoe 48` is **58.6 t/s** (was 10.3) and 3-GPU is **81.0**.
* **The `-sm tensor` + `-ncmoe` + `--load-mode none` crash (TODO #38) is STILL OPEN** (3/3 on r13);
  default `--load-mode auto` or `-sm layer` avoids it.  `wip/host-pinned-buffer-crash/`.

### The layer-vs-tensor puzzle (maintainer question, unresolved)

Why is `-sm layer` now **faster** than `-sm tensor` for an oversized MoE with host experts
(58.6 vs 45.4 at 2 GPUs), when for **dense GPU-resident** models tensor split is consistently faster on
multiple GPUs?

Current hypothesis (to test):

* With **host experts**, the tensor split shards every expert on axis 1 (gate/up) / axis 0 (down), so
  (a) the per-ubatch host-expert upload geometry is fine-grained -- the campaign already measured
  `ffn_down_exps` axis-0 splitting as **~524288 tiny host copies per device per layer**
  (`archive/work/tensor-split-expert-split/README.md` §26.3), and (b) the down projection needs a
  cross-device partial reduce per MoE op.  The layer split keeps each expert whole on one device: one
  contiguous upload, no cross-device reduce.
* For **dense GPU-resident** weights there is no host upload and no per-expert sharding, so tensor's
  better per-layer load balance wins (the r16 record).
* The `-sm tensor` host path is also the crash-prone one, and the meta backend's staged upload is the
  suspect.

**Test plan:** profile the 2-GPU tensor vs layer path (upload bytes/copies per token, cross-device
reduces, per-device busy time) at the same residency; if the hypothesis holds, the recommendation for
oversized MoE on multi-GPU is **`-sm layer` + cache**, and the auto-size guide should say so.  This may
also show the tensor path can be repaired (coarser upload geometry), which is a separate win.

### Next steps (in order)

1. **Re-baseline M0 on the r13 build with `-sm layer`** -- the earlier numbers used `-sm tensor`
   (`-ncmoe 24`, 43.2/34.6).  Re-run the single/2-/3-GPU IQ3_XXS + IQ4_NL cache sweeps with the fix.
2. **Re-check the cache-vs-MTP quant dependence** (IQ3_XXS +52 %, IQ4_NL -2 % at 20 GiB, single GPU):
   is it the MTP verify path, the expert size, or the GPU MoE kernel (IQ4_NL is not in block 13's
   multi-row table)?  This gates whether auto-enable is safe under MTP.
3. **Reserve/floor measurement grid** (M0 proper): peak non-arena VRAM across ctx x ub x MTP x
   draft-offload, and cache-on-vs-off at 256/512/1024 MiB, to derive `MOE_EXPERT_CACHE_RESERVE_MIB`
   and `MOE_EXPERT_CACHE_MIN_MIB`.
4. **Then implement auto mode** (unset == auto sized from free VRAM, `0` == off, `>0` == fixed).

### Locked semantics (do not re-litigate)

`MOE_EXPERT_CACHE_MIB` set (any value, incl. `0`) -> use it verbatim; `0` -> off; `-ncmoe > 0` (or
`-cmoe`) and the variable unset -> auto; no host-expert tables -> inert.  `-cmoe` is `-ncmoe <all>`.

## Why (the user-visible hole)

The cache is **opt-in** and today `MOE_EXPERT_CACHE_MIB` unset == cache off. With it off, `-ncmoe`
makes the expert op run on the **CPU**, which is ~1.5x slower than the cache-on GPU path even when
20 GiB of the card sits idle. Measured on a single R9700 / gfx1201 (32 GiB), ROCm 7.14.1, `build-rocm`,
Qwen3.8-Flash-Next UD-IQ3_XXS (45.29 GiB of routed experts) + `mtp-…shared-Q8_0.gguf`,
`-ngl 99 -sm layer -fa 1 -t 8 -b 2048 -ub 2048 --lazy-mode off --load-mode none
--spec-type draft-mtp --spec-draft-n-max 3`, prompt `prompts/code-python.txt`, temp 0, seed 42:

| cache | context (KV) | tg t/s |
|---|---:|---:|
| off, `-ncmoe 48` | 8K (q8_0) | **34.2** |
| `MIB=12288` | 8K (q8_0) | 39.8 |
| `MIB=20480` | 8K (q8_0) | **52.1** |
| `MIB=24576` (clamped to 20176) | 8K (q8_0) | 52.0 |
| `MIB=28672` | 8K (q8_0) | 51.9 |
| `MIB=20480`, **no MTP** | 8K (q8_0) | 33.2 |
| `MIB=20480` | 32K (q8_0) | 48.1 |
| `MIB=16384` | 128K (q8_0) | 42.7 |
| `MIB=12288` | 128K (q8_0) | 37.0 |
| `MIB=21504`, `RESERVE_MIB=256` | 8K (q8_0) | 52.9 |

So the missing cache is the whole 34 -> 52 gap, and MTP only pays with the cache armed (+57 %: 33.2 ->
52.1). The arena is clamped to `free − MOE_EXPERT_CACHE_RESERVE_MIB`; the verbose run showed
`MIB=24576` -> `clamping to 20176 MiB (free=21200 − 1024)`, i.e. **222 slots/table x 48 = 10,656 of
24,576 = 43 %** max residency here.

**q8_0 KV is fine.** The 2-4 t/s that a 4-bit KV buys (52.1 vs 48.1 at 32K) is not worth the context
rot; the maintainer wants q4_0 KV avoided. Auto-sizing must therefore budget around q8_0 KV.

## Current mechanism (read from `~/llama.cpp`, after r12)

* `parse_env()` (`ggml/src/ggml-cuda/moe-expert-cache.cu`) runs at `moe_cache_init()`, which is called
  from `ggml_backend_cuda_init` — **before** the model loads. `g_enabled = (MOE_EXPERT_CACHE_MIB set
  and > 0)`; `g_budget = that value`.
* `-ncmoe` lands later, as a host buffer-type override on the expert tensors
  (`src/llama-model.cpp`); the CUDA backend only learns about it when the scheduler calls
  `moe_cache_update` for each host-resident expert table.
* Sizing is **deferred**: the first decode pass is a *priming pass* (`t.primed = true`, returns false),
  the second pass calls `alloc_all_locked()`, which sizes every table uniformly from
  `cudaMemGetInfo(free) − g_reserve_mib` (default 1024 MiB). Fail-soft: a failed allocation disables
  the arena, it does not abort.
* An explicit `MOE_EXPERT_CACHE_SLOTS` bypasses all of this (immediate uniform allocation).
* If `g_enabled` is true but there are no host tables, the cache is inert — the r29 item-C fix made an
  empty cache behave exactly as disabled (fusions allowed). This is the property the auto mode needs.

## Target behaviour (LOCKED — maintainer, 2026-10-05)

Priority order, highest first:

1. **`MOE_EXPERT_CACHE_MIB` is set** (any value, including `0`) ⇒ **always use it verbatim**; no
auto-derivation, no surprise.
2. **`MOE_EXPERT_CACHE_MIB=0`** ⇒ **off** (the kill switch; restores today's CPU `-ncmoe` path).
3. **`-ncmoe > 0` (or `-cmoe`) and the variable unset** ⇒ **auto**: enabled, budget derived from the
   hardware at `alloc_all_locked`.
4. **Variable unset and no host-resident expert tables** (`-ncmoe 0`) ⇒ inert (no arena).

`-cmoe` is `llm_ffn_exps_cpu_override()`, i.e. the same condition as `-ncmoe <n_layer>`; it is included.
The trigger is the *host-resident expert table*, not the variable — see D1 for why the decision is
made from the table registration rather than from the CLI flag.

## Design

### D1 — Enable early, decide late

`g_enabled` must be true at `parse_env` time because `moe_cache_enabled()` gates the graph hooks and the
table registration that feeds `alloc_all_locked`. So: treat **unset as auto** (enable, `g_budget =
huge`), and let the existing clamp derive the real value in `alloc_all_locked` (already
"free − reserve"). No llama.cpp-side hook is needed; the empty-cache inertness covers a fully-resident
model (`-ncmoe 0`), which must stay byte-identical.

### D2 — The reserve is the whole problem

`alloc_all_locked` runs after model + KV + compute reserve are allocated, so the KV for the *requested*
context is already paid for. The reserve must still cover things that grow or were not sized there:

1. **MTP draft context.** With the cache armed, the draft context can stage full-size device copies of
   its host expert tables (`common/speculative.cpp` block-01 note: ~610 MiB on this qwen4exp head,
   ~1.1 GB on the reporter's). Either the reserve covers it, or auto mode defaults
   `LLAMA_MTP_DRAFT_OP_OFFLOAD=0` (freeing it, no throughput cost per r10).
2. **Compute/prefill growth.** A larger `-ub` prefill needs a larger compute buffer; the fixed
   `MOE_EXPERT_CACHE_RESERVE_MIB=1024` is a guess. `-ncmoe` + big arena + `-ub 4096/8192` must not OOM
   (cf. `COMMUNITY-CONFIG.md` gotcha: `-ncmoe 8` fits `-ub 2048` but OOMs `-ub 4096`).
3. **Staging ring** (block 06) is fail-soft (allocates outside the compute reserve; failure disables
   staging), so it is not a hard reserve item, but a starved ring costs prefill.
4. **Prompt-cache / server growth headroom** (the context is fixed at load, but `--fit`/server paths
   may size differently).

**Unknown to measure (M0):** peak non-arena VRAM across `{ctx 8K,32K,128K} x {ub 2048,4096,8192} x
{MTP on,off} x {draft-offload on,off}` on gfx1201 + IQ3_XXS, and the resulting minimum safe reserve.
Derive a formula (e.g. `reserve = fixed_headroom + k·compute_bytes + draft_bytes`) rather than a magic
1088.

### D3 — Floor (when to stay off)

Below some arena size the cache's fixed per-op cost outweighs the saved CPU bytes; the campaign's
CPU-split crossover was ~1.2-1.5 GiB, and `COMMUNITY-CONFIG.md` names a ~2 GiB maintainer floor. Add
`MOE_EXPERT_CACHE_MIN_MIB` (default ~1024-2048): if the auto-derived arena is below it, decline the
arena and take the CPU path. **Unknown (M1):** cache-on vs cache-off at 256/512/1024 MiB on
qwen4exp IQ3_XXS — measure before fixing the floor.

### D4 — Multi-GPU and `-sm tensor`

`alloc_all_locked` already sizes per device and takes the min slot count across a layer's devices.
Auto must not change that; each device's `free` decides its own arena. gemma4 `-sm tensor` is rejected
until the segmented host-expert upload lands (`TODO.md`), so scope here is `-sm layer` (all backends)
and qwen4exp `-sm tensor`. `MOE_EXPERT_CACHE_SLOTS` remains an override and should win over auto.

**The 2-GPU target (maintainer, 2026-10-05):** `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/`
(`Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` + the shared Q8_0 MTP head). Measured
geometry: **93.16 GiB total, 63.28 GiB of routed experts**, 26.82 GiB PLE, 48 layers, 512 experts.
It does **not** fit 2 x 32 GiB (64 GiB) with the PLE/KV/compute, so it is the canonical
`-sm layer`/`-sm tensor` + `-ncmoe` + auto-arena case. Acceptance for M0/M3: the two cards' arenas
size themselves from their own free VRAM, the model loads and runs with the variable unset, no OOM at
32K q8_0 KV, and decode beats the no-cache 2-GPU baseline.

### D5 — `--fit` / server

`--fit` does not know about the arena, and the arena is sized after `--fit`, so auto is compatible by
construction. Validate `--fit` with and without auto (the arena must yield, not fight). Server mode
adds the concurrency band gate (`n_tokens <= 8`); re-run the concurrent-session coherence check.

## Gates (all must be green before promotion)

* **`-ncmoe 0` byte-identity:** auto-armed but empty cache == cache-off. Oracles `de8be4d0c90c`
  (1-GPU layer), `15038c19ddc8` (2-GPU tensor), `d7bef4c6fdc3` (3-GPU tensor).
* **Cache-on arithmetic oracle:** `-ncmoe 99 MIB=auto` vs `MIB=1` must match (the trap: cache-on GPU
  MoE != cache-off CPU MoE by construction; do not use `-ncmoe 48` as the oracle).
* **Width purity:** `none == n1 == n3 == n7` on qwen4exp (and the existing `W=1..8` matrix on a
  qwen35moe model).
* **MTP:** acceptance >= ~0.45, MTP >= plain, on `prompts/{prose,code,recall,reasoning}` at `-n 3000`
  (`benchmarks/mtp-adaptive-methodology.md`).
* **Coherence:** deep-context `llama-cli` diff / no `////`, 32K and 128K, q8_0 KV.
* **No OOM:** ctx x ub x MTP grid above, on 1 GPU and `-sm layer` 2/3 GPU.
* **Performance:** single R9700 IQ3_XXS auto >= manual best (>= 52 @8K, >= 48 @32K, >= 42 @128K,
  q8_0 KV); `-ncmoe 0` unchanged; prefill unchanged.

## M0 results so far (2026-10-05, single R9700 + 2 x R9700, gfx1201)

### Single GPU, `-ncmoe 48`, c8192, q8_0 KV, n=128, `prompts/code-python.txt`

| model (routed experts) | spec | cache off | cache on (20 GiB) | delta |
|---|---|---:|---:|---:|
| IQ3_XXS (45.29 GiB) | none | 20.8 | **39.4** | +89 % |
| IQ3_XXS | MTP n3 | 34.2 | **52.1** | +52 % |
| IQ4_NL (63.28 GiB) | none | 19.7 | **28.6** | +45 % |
| IQ4_NL | MTP n3 | 34.2 | 33.5 | **-2 %** |

**Rule:** the cache always helps *plain* decode. Under **MTP** (the default usage) the gain is
quant/residency-dependent: IQ3_XXS wins big, IQ4_NL is flat. So "arm it whenever `-ncmoe > 0`" is not
safe on its own under MTP — the MTP verify path decides. Investigation needed into why IQ4_NL's
GPU-cache MTP gain is small (IQ4_NL is not in block 13's multi-row table; expert 0.88 MiB/table vs
IQ3_XXS 0.63).

### 2 GPUs, IQ4_NL (93.16 GiB / 63.28 GiB experts), c8192, q8_0 KV, MTP n3, n=128

| split | `-ncmoe` | cache | tg t/s | note |
|---|---:|---|---:|---|
| `layer` | 48 | off | 32.0 | |
| `layer` | 48 | on 24000 | 10.3 | known-bad: MoE consolidates on GPU 0 |
| `tensor` | 24 | off | **43.2** | the working fast baseline |
| `tensor` | 24 | on 8192 | 34.6 | **cache regression** |
| `tensor` | 24 | on 4096/16384/32768 | 36.4/34.7/34.6 | regression at every size |
| `tensor` | 24 | off, **plain** | 24.4 | |
| `tensor` | 24 | on 8192, **plain** | **38.7** | +59 % |
| `tensor` | 32 | off | 40.1 | crashed once |
| `tensor` | >= 40 | either | **crash** | GPU page fault, see below |

So on 2 GPUs the same plain-vs-MTP split appears, and **`-sm layer` + host experts is unusable**
(10.3 t/s — the MoE-consolidation mode). The `-sm tensor -ncmoe 24` baseline needs no cache at all
(43.2), and the cache makes it worse.

### Blocker: `-sm tensor` + host experts + `--load-mode none` crashes

A GPU page fault (nondeterministic, memory-state dependent) at load with
`--load-mode none` + big model + `-ncmoe >= 32` on 2 GPUs. Reproduced and root-caused to the pinned
`ROCm_Host` footprint (92.6 GiB for `-lm none` vs an 80 GiB `RLIMIT_MEMLOCK`); `--load-mode auto`
(the default) and `--load-mode mmap` are stable. Full record: `wip/host-pinned-buffer-crash/`.
**This must be fixed before auto-enable can be safe on a 2-GPU oversized model.**

### Consequences for the design

* The auto-enable decision cannot be "host tables exist". It needs a *net-win* gate (or the MTP verify
  gain fixed), plus a multi-GPU guard until the load crash is fixed.
* The `-ncmoe` choice itself matters: `-sm layer` + host experts is a trap; auto should warn (and the
  `-sm tensor` path should be the recommended one).
* q8_0 KV is confirmed fine at depth on the single-GPU case (48.1 @32K, 42.7 @128K); no 4-bit KV.

## Milestones

* **M0 — Reserve measurement + formula.** Peak-VRAM instrumentation, the grid above, a derived reserve
  and a first `MOE_EXPERT_CACHE_MIN_MIB`. Includes the **2-GPU IQ4_NL** case (63.28 GiB of experts on
  2 x 32 GiB) for `-sm layer` and `-sm tensor`. Deliverable: numbers + the formula, no code.
* **M1 — Auto mode.** `unset == auto` (`g_budget = SIZE_MAX`), `0 == off`, positive == fixed; the floor.
  A/B vs the manual best on the R9700; the `-ncmoe 0` oracles.
* **M2 — Draft reserve.** Pick reserve-vs-`LLAMA_MTP_DRAFT_OP_OFFLOAD=0`; re-run the MTP gates.
* **M3 — Multi-GPU / `-sm tensor` / `--fit` / server.**
* **M4 — Gates + docs.** Update `COMMUNITY-CONFIG.md` (auto is now the default), `patches/README.md`
  (block 13 amendment), `AGENTS.md` "default-on" note, `WORKLOG.md`.

## Open questions

1. Is "all free − reserve" always optimal, or does a very large arena cost prefill (compute headroom)?
   (Sweep the arena size against prefill + decode.)
2. Does auto-on help a small `-ncmoe` (2-8 host layers), or only a mostly-host-resident model?
3. Should auto default `LLAMA_MTP_DRAFT_OP_OFFLOAD=0`, or reserve for it?
4. Interaction with `MOE_EXPERT_CACHE_PREFILL_SEED` (default on) — the seed fills every slot from the
   prompt's routing; with an auto arena that is ~43 % resident, is the seed still the right warm start?
5. Does the change belong in the delivery at all, or as an upstream-PR candidate (`upstream/`)? It is
   generic to any MoE + `-ncmoe`, which argues for upstream.

## Risks

* **Default behaviour change for every `-ncmoe` user** (GPU MoE vs CPU MoE arithmetic). Intended under
  the default-on policy; `MOE_EXPERT_CACHE_MIB=0` is the kill switch. Must be called out in the release
  notes.
* **VRAM taken from KV/context.** The reserve + floor are the guards; an over-eager arena that starves
  a 128K context would be a regression, so the context must never shrink for the cache (measure at the
  requested context, `benchmarks/mtp-adaptive-methodology.md` rule 0 style: the real depth).
* **`--fit` users** who relied on the arena being off.
