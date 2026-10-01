# HANDOVER — beta5 validation, scaffolding strip, and the qwen35moe opportunity

**Created:** 2026-10-02
**For:** the next session (fresh context).
**Branch:** `promote-moe-caching` (delivery repo `~/llama-cpp-rdna-boosts`); the fix lives on
`~/llama-fold` branch `beta4` at **`21de1b20b`**, and as a self-contained patch at
`wip/moe-expert-cache/beta5-fix-registration-and-head-zero.patch`.

---

## RESULT (2026-10-02, same day) - do not re-derive

The objectives below are done.  The beta5 code is on `~/llama-fold` branch **`beta5-clean`** (based on
`21de1b20b`), patch `wip/moe-expert-cache/beta5-clean-gates-and-reorder.patch`, built at `/tmp/reorg`
(`build-rocm-b3`).  Three findings changed the picture:

1. **The WIP one-time zero was launched AFTER the gather**, so it zeroed the routed experts' own first
   64 bytes on each buffer's first gather.  That was the byte-identity regression.  **Moving the zero
   before the gather** makes both splits byte-identical (`de8be4d0c90c` / `15038c19ddc8`), keeps the
   fast zero, and is throughput-neutral.  A host-reading *fill* also restores byte-identity but costs
   ~3x prefill - not the fix.
2. **The qwen4exp "~1600 vs ~2400" prefill gap is the PLE mmap warm-up**, not a regression.  The
   `per_layer_token_embd.weight` (27465 MiB) is lazy by default and warms each pass.  `--lazy-mode off`
   loads it into host RAM: qwen4exp prefill is then **flat ~2650** t/s.  **Every prior qwen4exp prefill
   comparison in the campaign (r16 tensor-split, item-3, beta4) is confounded and must be re-measured
   with `-lzm off`.**
3. **The gather gate is now model-aware**: a table `>= 224 MiB` always gathers.  Unconfounded, qwen4exp
   ub8192 is gather 3052 vs staging 1403; Q8_0 3595 vs 3552; Q4_K_M 4291 vs **5624** (keeps the width
   gate).  qwen4exp ub8192 goes 1403 -> 3065 with no config change; Q4_K_M unchanged.

All admission gates are green (MUL_MAT_ID 929/929, byte-identity/width-purity, MTP 0.75273, Q4_K_M
matrix, coherence 8770/6793 words on qwen4exp/qwen35moe).  See `WORKLOG.md` "2026-10-02 (moe-cache beta5
validation)" and `wip/moe-expert-cache/COMMUNITY-CONFIG.md`.  The scaffolding of section 2 below is
stripped; the two section-0 fixes remain.

---

## 0. What beta5 is (read this first)

beta5 = **beta4 + two fixes** in `ggml/src/ggml-cuda/moe-expert-cache.cu` (block 13):

1. **Gather-path registration** (fixes the decode collapse).  The MoE cache registered an expert table
   only in the host upload hook (`moe_cache_update_host`); the **device gather**
   (`moe_cache_gather_host`, the whole prefill path) uploaded without registering.  A prefill therefore
   registered only the layer that fell off the gather (the last one), the deferred arena sizing latched
   on those 3 tables, the other 47 layers registered afterwards with 0 slots and declined, and decode
   fell off the cache — *slower* than uncached (8.54 vs 20 t/s) because the partial arena stood the
   cache-band fusions down globally.  **Fix: one `moe_cache_table(...)` call in
   `moe_cache_gather_host`** (name_layer/name_role + the same geometry the host hook uses).

2. **One-time expert-head zero** (fixes the prefill regression).  The quantized `MUL_MAT_ID` load
   speculatively reads the **first ~64 bytes of the NEXT expert slot**; the pruned gather left them as
   the reused `input_cpy`'s stale/NaN bytes (NaN×0 = NaN poisons the tile), which is what the
   session-18 MMQ tail pad (`d76e18efd`) was added for.  That pad cost **~3×** in *every* form tried
   (in-kernel trailing copy, folded into the last chunk, separate kernel, own-head pad, zeros vs host
   bytes).  It is **not** a stride/misalignment (the layout is untouched and zeros fix it) and not
   bandwidth (a 512×64 B write = 32 KB).  The correctness threshold is exactly **64 bytes** (48 still
   corrupts, 64 is fine) and the perf cliff is ~54, so **no per-gather pad is both fast and correct**.
   **Fix: zero the first `head_bytes` of every expert slot ONCE per `input_cpy` buffer**
   (`moe_cache_gather_zero_heads_kernel`, tracked in `g_heads_zeroed`); the gather overwrites routed
   heads with real data, so only non-routed heads rely on the zero, and the invariant holds for the
   buffer's life.

**Verified numbers (single R9700, gfx1201, ROCm 7.14.1):**

| model / config | pre-fix (beta4) | beta5 |
|---|---|---|
| **Qwen3.8-Flash-Next IQ4_NL** qwen4exp, `-ncmoe 48 -sm layer`, MIB=12288, `-b2048 -ub2048`, `-p8192 -n1024` | pp 641 / tg 8.5 (or 648) | **pp 2401 / tg 36.5** |
| same, coherence essay `-n12000 -c16384 --reasoning off` | `[Start thinking] //////` | **PASS**: 8457 words, 12 sections + `### Conclusion`, fluent, 0 `////` |
| **Qwen3.6-35B-A3B Q8_0** qwen35moe, `-ncmoe 8` MIB=4096, `-b2048 -ub2048` | — | **pp 2644 / tg 72.5** |
| qwen35moe `-ncmoe 16` MIB=8192 | pp 1746 / tg 80.1 | **pp 1743 / tg 81.5** |
| qwen35moe `-ncmoe 40` MIB=12288 | pp 903 / tg 61.4 | pp 901 / tg 61.6 |

**The fix is neutral for qwen35moe** (it had neither bug) and a large win for qwen4exp.

---

## 1. Objective of the next session

1. **Strip the diagnostic scaffolding** (see §2) so beta5 becomes a clean, promotable block-13 delta.
2. **Run the full validation gate set** (§4) on the cleaned tree.
3. **Resolve the gather-vs-staging gate** so qwen4exp gets the gather at `-ub 2048` **by default**
   without regressing qwen35moe / Q4_K_M (the beta4 width gate currently sends qwen4exp to staging:
   pp 641 vs the gather's 2401).
4. **Explore the qwen35moe opportunity** (§5) — whether the qwen4exp-style residency/gather win can be
   brought to qwen35moe, and publish the best single-GPU "oversized Q8_0 MoE" config (§6).

Do **not** tag, push a GHCR image, or merge to `main` (see AGENTS.md "Pushing policy" / "Default-on
policy").  Commit work to `promote-moe-caching` and push only to this repo's `origin`.

---

## 2. Precise scaffolding to strip

Everything below is in `ggml/src/ggml-cuda/moe-expert-cache.cu` **except** the last item.  All of it is
diagnostic; only the two fixes of §0 stay.  (Line numbers are from the beta5 tree in `/tmp/reorg`,
`git diff f29745280 -- ggml/src/ggml-cuda/moe-expert-cache.cu`.)

**Env knobs to remove (3):**

| knob | where | why it existed | action |
|---|---|---|---|
| `GGML_META_GATHER_NOPAD` | ~line 2027 | disable the pad to A/B the corruption | **remove** (the fix is correctness, not a feature) |
| `GGML_META_GATHER_HEAD_BYTES` | ~line 2026 | sweep the pad size while diagnosing | **remove**; hard-code `head_bytes = min(expert_bytes, 512)` (matches the host path's `min(expert_size, 512)`; the one-time zero is free at any size, so use the full 512) |
| `GGML_META_GATHER_PAD_SEP` | ~line 2051 | run the pad as a separate kernel while diagnosing | **remove** |

(`GGML_META_GATHER_PAD_BYTES` and `GGML_META_GATHER_PAD_ZERO` were already dropped during the session;
if any remain, remove them too.  Verify with `grep -n GGML_META_GATHER ggml/src/ggml-cuda/moe-expert-cache.cu`
once edited — the only surviving `GGML_META_GATHER*` string should be none.)

**Code to remove:**

* `moe_cache_gather_pad_kernel` — the whole `static __global__` function (~line 830), a diagnostic
  alternative to the in-kernel pad.
* Its launch — the `if (pad > 0 && getenv("GGML_META_GATHER_PAD_SEP") != nullptr) { ... }` block
  (~line 2051).
* The **own-head MMQ tail guard** block in `moe_cache_gather_kernel` (~lines 789-810), the
  `// MMQ tail guard: write our OWN slot's first \`pad\` bytes ...` comment and body.
* The `int64_t pad` parameter of `moe_cache_gather_kernel` (and all `pad` uses inside it).
* The `int zero_fill` parameter (only the pad kernel used it).
* In `moe_cache_gather_host`: `const int64_t pad = 0;`, `const int zero_fill = 0;`, and the stale
  `// MMQ tail guard (the session-16 ...)` comment above the old pad code (~line 2011).

**Keep:**

* The `moe_cache_table(...)` registration call in `moe_cache_gather_host` (§0.1).
* `moe_cache_gather_zero_heads_kernel`, `g_heads_zeroed`, the `#include <unordered_set>`, and the
  one-time zero launch (with `head_bytes` hard-coded per the table above).
* `moe_cache_policy_copy` (used by the main gather and the seed-fill kernels).

**Where it lands:** block 13 (`moe-expert-cache.{cu,h}` + the `mmvq.cu` slot lookup).  The beta3 reorg
put the cache engine in block 13 and the scheduler/interface in block 06; this diff is engine-only, so
it belongs in block 13.  Regenerate the patch set (`scripts/make-patches.sh` + `scripts/make-release.sh`)
only after the strip and a green gate run; until then keep it on the WIP branch.

---

## 3. Environment (do not re-derive)

* **Box:** 3× R9700 gfx1201, 184 GiB RAM; ROCm `/opt/rocm-7.14.1-gfx120X`; single-GPU work uses
  `HIP_VISIBLE_DEVICES=0`.
* **Build:** `BUILD_DIR=build-rocm-b3 EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`
  (ccache makes a one-file rebuild ~5 s).  `LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib`.
* **Worktrees:** `~/llama-fold` = the fork.  `beta4` branch = the beta5 fix (`21de1b20b`).  `/tmp/reorg`
  = the built beta5 tree (`build-rocm-b3`).  `/tmp/prefix` = pre-fix `f29745280` (`build-rocm`).  Scratch
  trees `/tmp/b26` (r26-b1), `/tmp/b1` (r6140bba76), `/tmp/bs-*` (campaign bisection tips).
  Remove stale ones with `git -C ~/llama-fold worktree prune`.
* **Models:**
  * qwen4exp: `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf`
    (9 parts, ~93 GiB; the `mtp-…-shared-Q8_0.gguf` sidecar sits next to it).
  * qwen35moe: `/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` (37.8 GB).
  * others under `/llm/models/` (`Qwen3.6/35B-A3B/Q4_K_M`, `Qwen3.8/27B/IQ4_NL`, `Gemma4/`).
* **`llama-cli` MUST use `--single-turn --no-display-prompt`.**  Always `-t 8` (GPU IRQs pin to cores
  13-15).  Never run benches in parallel.
* **Prompt for the coherence gate:** `wip/moe-expert-cache/coherence-essay-prompt.txt`
  (sha256 `5e9a8ab0…`, 170 words).  Run `--reasoning off`, `-n 12000 -c 16384`; the essay should be
  ~8000 words / 12 numbered `###` sections + `### Conclusion`, fluent, **zero `////`** and no repeated
  lines.

---

## 4. Validation gates to run (all of them)

Run on the **cleaned** tree, in this order; record every number in `WORKLOG.md` and
`wip/moe-expert-cache/`.

1. **Clean build, warning-free:** `BUILD_DIR=… ~/bin/build-llama-rocm-714`; `grep -icE "warning:|error:"`
   must be **0** (beta2 removed the four `moe_cache_*` NULL-field warnings and the two `ggml-cpu.c`
   ones; if any reappear, the strip re-introduced an initializer gap).
2. **`scripts/validate-set.sh`** green (16/16 `git am`, checksums, applied tree == `release.json.tree`).
   Note: this needs the patch set regenerated first if the tree changed; otherwise run it against the
   current `release.json` to confirm the delivery still applies.
3. **`test-backend-ops -o MUL_MAT_ID`** 929/929 (the op the cache sits under).
4. **Byte-identity gate:** 2-GPU `-sm tensor` 300-token greedy hash must be **`de8be4d0c90c`**.
5. **Width-purity gate:** 1-GPU `-sm layer` `W=1..8` hash **`15038c19ddc8`**.
6. **Cache decode sweep (1-GPU qwen4exp, `-sm layer`):** MIB 0/8192/16384 with the gather on — the
   earlier gate recorded ~39.7/74.2/81.2 t/s; re-record with the fix (expect ≥ that).
7. **Coherence:** the essay gate of §3 on **both** models (qwen4exp and qwen35moe), `slashline == 0`.
8. **MTP `n3`** acceptance (2-GPU qwen4exp): record vs the beta4 baseline **0.75273**.
9. **The two single-GPU "middle ground" sweeps** with the gather forced
   (`GGML_SCHED_STAGE_MIN_TOKENS=999999`) and with the default gate:
   * qwen4exp: `-ncmoe 48 -sm layer -fa 1 MIB=12288 -b2048 -ub2048 -p8192 -n1024 -r 2` → must show
     **pp ≥ 2000, tg ≥ 35** with the gather.
   * qwen35moe: `-ncmoe 8 MIB=4096` and `-ncmoe 16 MIB=8192`, same shape → record pp/tg.
10. **The Q4_K_M prefill matrix (the beta4 do-not-break list):** `pp8192 ub8192 -r 3` (gfx1201,
    2-GPU tensor/layer + 1-GPU layer, `-ncmoe 0/16/32/40`) must stay ≥ beta4:
    `7269/6306/5606/5257` (tensor), `5512/4511` (layer), `5593/5398` (1gpu).  The ungated gather cost
    Q4_K_M `-sm layer -ncmoe 40` **-42%**; the gate must still protect it.
11. **Multi-GPU (`-sm tensor`) must beat single GPU** at every offload level for the oversized models —
    measure the 2-GPU equivalent of the §4.9 sweeps and confirm `tensor` > 1-GPU (and, where the r16
    record exists, `tensor` > `layer`).

---

## 5. The qwen4exp → qwen35moe opportunity (open question)

beta5's two fixes were **qwen4exp-triggered** and are **neutral for qwen35moe**:

* the registration collapse happened because the **qwen4exp gather graph registers only the last
  layer**; qwen35moe registers all tables, so its arena always sized correctly;
* the tail-pad 3× did not reproduce on qwen35moe (its prefill is identical with the gather and with
  staging).

**Investigate** whether qwen35moe can be given the same class of win — i.e. is there a way to make its
single-GPU oversized decode/prefill meaningfully faster, or is it already at the ceiling?  Concretely:

* **Why does qwen35moe not collapse after a prefill?**  Instrument `moe_cache_table` registration order
  and `alloc_all_locked`'s table count for qwen35moe vs qwen4exp (`MOE_EXPERT_CACHE_DEBUG=1`).  If
  qwen35moe's prefill registers all tables, why (different graph structure, staging instead of gather,
  a different `n_tok` band)?  The answer decides whether other MoE archs are at risk of the same bug.
* **Is the arena actually being used for the decode?**  Check the `moe_cache_report` `h`, `takeover`,
  `hits/colds` for qwen35moe at `-ncmoe 8/16`.  qwen35moe is dense-routed (A3B), so its hot set may be
  broad; is the cache helping or is the decode GPU-bound already?
* **The gather for qwen35moe:** the beta4 width gate assumes the gather only wins for qwen4exp.  Measure
  qwen35moe prefill gather vs staging at `-ub 512/1024/2048/4096/8192` (both `-sm layer` and
  `-sm tensor`).  The Q4_K_M gate says the gather loses at large ub; check whether that is specific to
  the small-expert-byte Q4_K_M or holds for Q8_0 too.
* **Residency sizing for dense routing:** if qwen35moe's hot set is broad, a larger `MIB` may pay; sweep
  `MIB` against `tg` at fixed `-ncmoe`.  (The `-ncmoe 8` MIB=4096 → 72.5 vs `-ncmoe 16` MIB=8192 → 81.6
  result suggests the decode is capped by residency, not by the gather.)
* If nothing transfers, say so explicitly and record why — a documented "qwen35moe is already at its
  single-GPU ceiling; the fix is qwen4exp-specific" is a valid, useful outcome.

**Constraint:** any change must keep the qwen4exp numbers and the Q4_K_M matrix, and stay
byte-identical on the admission gates.

---

## 6. Community angle — an "ideal single-GPU config for oversized Q8_0 MoE"

The user wants the community to be able to converge on a good config for running an **oversized Q8_0
MoE** (model + cache > VRAM) on one card.  Deliverable for the follow-up: a short, reproducible tuning
guide + a recorded config table, e.g. under `wip/moe-expert-cache/` and (once promoted) `README.md`:

* State the axes: `-ncmoe` (how many layers offloaded), `MOE_EXPERT_CACHE_MIB` (arena size),
  `-ub/-b` (ubatch), `-sm layer` vs `-sm tensor`, and the gather/staging choice.
* The known-good qwen35moe Q8_0 single-R9700 points (`-b2048 -ub2048 -fa1`):
  `-ncmoe 8 MIB=4096` = **2644 / 72.5**, `-ncmoe 12 MIB=8192` = 2081 / 80.5,
  `-ncmoe 16 MIB=8192` = 1750 / **81.6**, `-ncmoe 24 MIB=12288` = 1320 / 78.9,
  `-ncmoe 40 MIB=12288` = 903 / 61.5.  Recommend the trade explicitly (max prefill vs max decode — the
  user asked for "both good at once", which here is the `-ncmoe 8..16` band).
* Publish the **method** (the one-process `-p8192 -n1024` combined measurement; warm the model; `-t 8`;
  single GPU) so others reproduce rather than compare single-sided runs.
* Note the qwen4exp point: `-ncmoe 48 -sm layer MIB=12288 -b2048 -ub2048` = **2401 / 36.5**.
* Flag that `-ncmoe 8` only fits at `-ub 2048` (it OOM'd at `-ub 4096`), and that `--fit`/`-c` interact.

---

## 7. Definition of done

* The scaffolding of §2 is gone; the tree is warning-free; the two §0 fixes remain.
* §4 gates all green (or every deviation explained and recorded).
* The gather-vs-staging gate is model-aware (qwen4exp gather at `-ub 2048` by default; Q4_K_M/qwen35moe
  protected) **or** the decision and its measurement are recorded as the accepted trade.
* §5 has a concrete answer (a win, or a documented "already at ceiling" with evidence).
* §6 exists as a reproducible guide with the config table.
* `WORKLOG.md` (new dated entry, top) + this handover updated; commit and push to
  `origin/promote-moe-caching`.
