# MoE expert cache: auto-enable + auto-size (`MOE_EXPERT_CACHE_MIB`)

**Status: OPEN (2026-10-05). Nothing here is in the delivery.** Maintainer direction: if `-ncmoe > 0`
and the user did not set `MOE_EXPERT_CACHE_MIB`, the cache should arm itself and size itself from the
hardware, so the "administrative" load leaves the user. This is the core focus of the campaign.

Related: `COMMUNITY-CONFIG.md` (the manual config guide), `archive/work/moe-expert-cache/` (the
campaign that built the cache), `GREEDY-PURITY.md` §19/§24/§25, `patches/README.md`
(block 06/13/15 notes), `wip/moe-cpu-overlap/` (the other half: making the miss path overlap).

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
