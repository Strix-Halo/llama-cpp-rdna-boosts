# OPEN 2 — the movable-boundary slab: cold-start handover

**READ THIS FIRST.**  The design is **implemented, released as r22, and both DoD gates pass**.  The dated
sections below are the trail; read the "Current status", "OPEN ITEMS + the r22 fold plan" and "SESSION 8"
sections as the current state.

---

## Current status (r22, 2026-10-07) — start here

The slab is **SHIPPED**: **`v16-a55e952b8-r22`**, canonical block-15 tip **`562e06f81`**, net tree
**`c0927a3ea887588564f7fa1b354d773871a20b6f`**, still **16 blocks** (folded into block 15 in place, no
rebase).  `validation-set.sh` is green (strict `git am` on a fresh tarball reproduces the tree).  The fork's
`rdna-boosts` branch is refreshed to `562e06f81`; the delivery repo's `main` is `c73003a` with the tag.
**`~/llama.cpp` is now CLEAN at `562e06f81`** — the working-tree diff that this document was written
against (`open2-ideaB-vmm-compute.diff`) is IN the delivery, so do not look for it as uncommitted WIP; build
with `BUILD_DIR=build-rocm-vmmc` (or any) as usual.  The variable is **default ON**
(`GGML_CUDA_SLAB=0` is the kill switch).

| gate | r21 | r22 (shipped) |
|---|---|---|
| cli `-n 2000`, `-ub 8192`, cache auto, 16k, default | decode 78.7 / prefill 1683 / acc 0.9245 | decode **74.2** / prefill **1712.9** / acc **0.92448** / hit 0.9555 |
| cli `-n 3000 --reasoning on` (rule 0) | acc 0.53519 (1848/3453) / 64.3 t/s | acc **0.53519 (1848/3453 — bit-identical)** / 61.5 t/s |
| server default (wide1 -> short -> wide2) | **ABORTS** (`cudaMalloc failed` -> `ggml-backend-meta.cpp:1817` assert) | **0 aborts**, both wide responses coherent, arena 38026.8 MiB (67.0 %) |
| 3-GPU (`HIP_VISIBLE_DEVICES=0,1,2`) | — | coherent, arena **99.9 % residency** (56725 of 56762 MiB) |
| `llama-batched-bench -npl 1,4,8` (rule 5, same-binary A/B) | — | **identical** to slab-off at B=1/4/8 |
| dense model, slab on vs off | — | **byte-identical** generated text |

`////` = 0 everywhere.  **Cost:** the slab's arena is smaller than the plain path's, because the plain path
sizes the arena against free VRAM *after* the weights are resident and effectively double-books the compute
buffer's space.  Measured on the cli: arena 36474 vs 38859 MiB (same binary) -> decode 61.5 vs 63.8 t/s in
the `--reasoning on` config (**~3.6 %**).  The drop being default-on and the reserve being right-sized more
than repay it (the r22 DoD decode matches r21 within noise while the server gained ~22 GiB of arena).

### OPEN ITEMS + the r22 fold plan (read this before anything else)

**Blocking candidate (RESOLVED as "not viable" -- see SESSION 8).**  `ggml_cuda_slab_extend` reclaims the
reserve the model did not need, which leaves the cards with **~150-350 MB free** at steady state (measured by
the maintainer on a live 16k-context server; I measured 348 MiB).  Any allocation larger than that which
cannot be redirected to the slab **aborts the process** (CUDA_CHECK).  The transient-into-slab work was
implemented TWICE and is now REVERTED: it *does* work for llama.cpp's own transients (the pool and the
buffer path are served from the slab -- "served a 16.82/33.65 MiB workspace transient"), but it does NOT
make a thin headroom safe, because a third consumer is out of reach: **the ROCm BLAS stack (hipBLASLt)
allocates its own Tensile code objects and workspace behind the application's back** (llama.cpp never calls
`cublasLtMatmul`; the build routes GEMMs through it), and that allocation cannot be satisfied from the slab.
Measured at `GGML_CUDA_SLAB_HEADROOM_MIB=2048`: `hipModuleLoad failed` for the Tensile `.co` files (6x, and
0x in every successful run) then `Hip error: 'out of memory' at hipblaslt.cpp:164` -> abort.  With
`ROCBLAS_USE_HIPBLASLT=0` at 2048 the server SURVIVES but the output is **CORRUPT** (`////`), and at 4096
with hipBLASLt off a 16k request returned 1 token then EOS -- so **disabling hipBLASLt is NOT a safe
workaround** and must not be recommended.

**Conclusion for r22: the headroom must cover the RUNTIME's own allocations, not just llama.cpp's.**  Keep
`GGML_CUDA_SLAB_HEADROOM_MIB=4096` (verified across the gates and on the 163860 config) and do NOT ship the
transient fallbacks; they are a margin, never a licence to run thin.  Re-landing them needs the corruption at
a thin headroom understood first (see the open item below).

**FIELD-VALIDATED (SESSION 9, 2026-10-07i).**  On the maintainer's real server config (2 GPU
`-sm tensor -ncmoe 48`, `-ub 6144 -c 204800`, cache auto, `GGML_CUDA_ALLREDUCE=ce`) the slab ran a
30k-token prompt straight after a 45k-token generation: **0 aborts**, prefill **1445 t/s**, decode
**68-71 t/s** (r21 stayed at 35-40), and the drop's `moe_cache_rearm` restored the arena to
**38854.7 MiB** (88 tables re-armed).  Logs: `/tmp/s22-logs`.

**Observation (DOWNGRADED -- maintainer: it happened with hipBLASLt DISABLED, i.e. in the unsupported BLAS
configuration, so it is not worth chasing).**  During one of the thin-headroom runs the cards sat at ~31 GB,
an allocation failed, the used VRAM fell to ~13 GB each and the server continued -- "it's like it dropped the
whole SLAB cache".  The run in question had `ROCBLAS_USE_HIPBLASLT=0`, which also produced the corrupt
output and the 1-token/EOS 16k request, so it is treated as a property of that configuration rather than of
the slab.  Recorded for completeness only; if it ever appears with hipBLASLt ENABLED, reproduce it with
continuous VRAM monitoring plus a log of every cudaFree/arena-free, because nothing in this design unmaps
client memory (`ggml_cuda_slab_extend` only maps; freeing a table or a pool block returns the range to the
slab's free list and leaves the physical mapped).

**Refinement on the corruption evidence (Session 8).**  The `////` corruption at headroom 2048 was observed
ONLY with `ROCBLAS_USE_HIPBLASLT=0`, so it does NOT condemn the transient fallbacks -- but it also does not
vindicate them: with hipBLASLt ENABLED at 2048 the failure is the library's own allocation
(`hipModuleLoad failed` / `hipblaslt.cpp:164`), which no fallback can redirect, so the fallbacks cannot make a
thin headroom viable either way.  Their only possible value now is margin at a given headroom, and at the
shipped default (4096) they were INERT in every run (0 slab-served transients), so they stay out.

**Why the cards sit at ~200-350 MB free -- EMERGENT, not a chosen number.**  `GGML_CUDA_SLAB_HEADROOM_MIB`
(4096) is what stays free *at the moment the slab is extended*; the steady state is that amount MINUS
everything allocated afterwards.  Measured consumption (~3.7 GiB of the 4 GiB): the MTP draft buffer
(**762 MiB**, measured), the FA-QSA workspace (**762 MiB**, measured), hipBLASLt's Tensile code objects +
internal workspace, and the workspace pool's high-water mark.  So the residual free VRAM is a leftover, not a
target.  Raising the headroom raises it one-for-one (8192 -> ~4.3 GiB free, at the cost of ~4 GiB of arena);
lowering it is what fails (2048 aborts inside hipBLASLt).  4096 is the verified floor for this config.
**Other open items** (full detail in TODO.md #42 and the session sections below): the parked
unit-mapping/tail-prune path; the pre-existing `MOE_EXPERT_CACHE_MIN_MIB` late-disable corruption (defaults
to 0); the STALE `build-rocm-r16` reference build; the `--fit` interaction (a fit that starts with less free
VRAM than expected iterates 7 rounds instead of 2 and can trip a PRE-EXISTING Meta-backend assert at
`ggml-backend-meta.cpp:519` -- reproducible with a rapid restart, gone with a 15 s gap or `-fit off`;
`archive/work/host-pinned-buffer-crash` territory).  **The three SESSION 9 diagnostics were FIXED in r23** (see SESSION
10): the `MMB_*` prints are gated on `GGML_CUDA_MMB_LOG`, `ggml_cuda_slab_extend` reports post-mapping free
VRAM, and the stale-alias count is surfaced; the **`Meta()` teardown size warning is fixed too** (the drop
refreshes `backend_buf_exp_size`; only a genuine growth past the reservation warns now).  **New unfixed finding
from that work: `atexit(moe_cache_report)` never prints** (`if (!g_enabled) return;` -- any real run releases
the arena or disables the cache first), so re-enabling it needs its own gate.  Next battle named by the
maintainer: **under `-sm tensor -ncmoe` the expert weights are mirrored rather than split across the GPUs**
(TODO #44).

**r22 fold plan — DONE, via block 15 (route B).**  The slab shipped in **block 15** (amended in place: block
15 IS the tip, so no rebase and no conflicts; `n_blocks` stayed 16 and the delivered code is identical to what
folding into 06 would produce).  **Folding into block 06 remains optional pure repackaging** — it changes no
code, only which patch carries it, and it needs the rebase of blocks 07-15 (conflicts concentrate in
13/14/15, which touch the same functions).  Do it only if the block-06 attribution is wanted; the delivery is
correct without it.  When it IS done, the mandatory gates are: `scripts/validate-set.sh` (16/16), a build,
the coherence gate, the MTP rule-0 gate, whitespace-clean, and the fork `rdna-boosts` refresh
(`--force-with-lease`, personal fork only).

### The design (the thing not to lose)

**ONE slab per device**, created at the first compute-buffer alloc:

* `cuMemAddressReserve` + exactly **ONE `cuMemMap`** of the allocatable VRAM (free VRAM minus
  `GGML_CUDA_SLAB_RESERVE_MIB`), rounded down to the boundary chunk.
* Split by a **BOUNDARY**: LOW `[0, boundary)` = the **work pool** (the compute buffer); HIGH
  `[boundary, size)` = the **MoE expert-cache arena**.
* Growing the work pool is a **boundary move inside the already-mapped slab**: the lowest arena chunks are
  reassigned to the work pool and the arena tables living in them are evicted (`moe_cache_evict_slab_range`
  -> `ggml_cuda_slab_arena_free`).  **HIP is never called at runtime** — only at slab creation and shutdown.
* The work region's base VA never moves, so a growing layout **keeps its tensor addresses** (the work buffer
  is a view whose reported size is the boundary).
* The boundary also moves **DOWN**: after the prefill -> decode drop the work layout is narrow (~1.9 GiB), so
  `ggml_cuda_slab_work_release` (from the compute buffer's destructor) plus the next, smaller `work_alloc`
  hand the slack back to the arena.

**Do not re-introduce unmapping.**  ROCm rejects a partial `hipMemUnmap` of a mapping (measured:
`vmmprobe2.cpp`), so a byte-granular "tail prune" is impossible when an allocation is one mapping, and unit
mapping broke an unrelated `hipMemcpy2DAsync` in-tree (`vmmprobe3/4.cpp`).  The slab needs neither.

### The three bugs that had to be fixed to make it work (all OURS, not the design's)

1. **The wholesale-fallback gate had to become PER TABLE.**  `moe_cache_has_arena_locked()` is all-or-nothing,
   but under the slab a **partial** cache is the NORMAL state (evicting the tables in the taken chunks is
   routine).  With the global gate, one boundary move disabled the whole cache -> the decode ran on the
   host-expert path: **8 CPU cores busy / GPUs ~30 % / decode 7-21 t/s**.  Now `moe_cache_take_over`,
   `moe_cache_update_host`, `moe_cache_get_table` and `moe_cache_get_slot` gate on **that table's**
   arena/slots and still agree per table; only the FUSION guard stays global (`moe_cache_has_arena()`).
2. **The arena's allocation unit is FINE (the VMM granularity, 2 MiB), not the 64 MiB boundary chunk.**
   Chunk-aligning each table's slab wasted up to 63 MiB per table (~9 GiB over 288 tables), so the sizing
   (which does not model the rounding) over-committed, most tables failed, and the decode collapsed to the
   host path again.  The slab is ONE mapping, so the sub-slab unit is pure bookkeeping.  `alloc_all_locked`
   holds back `n_tables * unit` and sizes against `ggml_cuda_slab_arena_total()` — NOT `cudaMemGetInfo`,
   which reads only the little left outside the slab.
3. **`work_live` had to be a MULTISET OF REQUESTED SIZES, not a bool** (found and fixed this session — see
   below).  This one silently corrupted output; it is the most important of the three.

### SESSION 10 (2026-10-07j): r23 -- the SESSION 9 nits fixed, and the atexit report found dead

**Shipped** `v16-a55e952b8-r23` (block 15 amended in place; tip `ef49781df`, tree `f652d71c`; 16/16,
`validate-set.sh` green).  No behaviour change: DoD **1695.0 / 74.4** with MTP acceptance **0.92448**
(bit-identical to r22) and the rule-0 gate **0.53519 = 1848/3453** (bit-identical to r21/r22).

* the six `MMB_*` prints -> gated on `GGML_CUDA_MMB_LOG` via a new `mmb_dbg()`.  A/B on an IQ4_NL prefill
  (`T=6144`, the field's shape): **0 lines by default, 83 with the gate on** -- so the gate is live, not a
  dead path.
* `ggml_cuda_slab_extend` -> `4.03 GiB left free AFTER this mapping, which is GGML_CUDA_SLAB_HEADROOM_MIB
  -- not extra headroom`.
* the stale-alias guard -> `...; 1 refused so far` (reproduced on the wide -> short -> wide server run).
* the teardown size check -> the drop refreshes `backend_buf_exp_size` to the re-reserved layout, and the
  check only WARNs when the current size EXCEEDS the expectation (a legitimate shrink is DEBUG).  The field's
  two big mismatches (Meta 2048 vs 11776, CPU 0.52 vs 241.2 MiB) are gone; **the server teardown is silent**.
* **NEW (deliberately unfixed): `atexit(moe_cache_report)` never prints.**  Its first line is
  `if (!g_enabled) return;`, and any real run releases the arena or disables the cache before exit.  Enabling
  it means exercising a dormant shutdown path (it re-reads per-device policy counters after the context is
  destroyed), so it is not a nit -- it needs its own gate.  Consequence: the alias count lives in the WARN.
* **Correction to our own methodology:** in this CLI the generated text goes to **stderr**, so the earlier
  dense "slab on vs off byte-identical" checks that captured stdout only were comparing timings.  Redone on
  both streams (the `> prompt` .. `[ Prompt:` region): byte-identical (142 chars each).

### SESSION 9 (2026-10-07i): r22 FIELD VALIDATION on the real server config -- stable, fast, and the drop re-arms

**Verdict: the slab holds on the very config that produced the original crash.**  Full server logs:
`/tmp/s22-logs` (2 GPU `HIP_VISIBLE_DEVICES=0,1 GGML_CUDA_ALLREDUCE=ce`, `-sm tensor --n-cpu-moe 48`,
`-ub 6144 -b 6144 -c 204800 --no-kv-unified`, `-ctk/-ctv q8_0`, `--fit off`, `--spec-type draft-mtp
--spec-draft-n-max 3`, IQ4_NL 9-shard qwen4exp: n_embd 2560, 48 layers, 512 experts / 10 used).

**First, retire the earlier "observations".**  The `ce_tmp2` OOM crash and the 35-40 t/s "as if the wide
work pool was never relinquished" both came from `build-rocm`, which was an **r21 build with no slab in it**
(0 `ggml_cuda_slab` symbols -- `build-rocm` was rebuilt from r22 for this test).  They were r21 behaviour,
and both are precisely what r22 fixes.

| what the log shows | number |
|---|---|
| slab per device at init | 19.94 GiB = work 8.25 + arena 11.69 (8.00 GiB reserve, 4.00 GiB VA spare) |
| `ggml_cuda_slab_extend` after the first prompt | +2.12 / +1.62 GiB -> 22.06 / 21.56 GiB (work **2.25** GiB + arena 19.81 / 19.31 GiB) |
| arena auto-sizing | 39424.2 MiB of 64800.0 MiB host experts (**60.8 %**), restored to **38854.7** MiB |
| the 30k-token prefill at `-ub 6144` | **1242 -> 1534 t/s**, 29592 tokens in 20.47 s -- no abort (this is the shape that crashed r21) |
| its boundary move | `moe_cache_evict_slab_range` evicted 5670.0 then 6156.0 MiB (arena 39424 -> 27036 MiB) |
| the following drop | `moe_cache_rearm: re-armed 88 stood-down expert-cache tables ... arena now 38854.7 MiB` |
| decode after the 30k prefill | 35 -> 59 -> **68-71 t/s** (r21 stayed at 35-40) |
| the 45k-token generation | 68.4 t/s mean, MTP acceptance 0.6455, arena hit 0.9633 |
| the next generation | **70.9 t/s** mean, acceptance **0.95131**, arena hit 0.9545 |

0 aborts, no corruption, full-speed decode **after** a wide prefill, and the arena fully restored: the
design's central claim, on the maintainer's real workload.  (The `ce_tmp2` allocations this path makes are
~90 MiB/device at `-ub 6144` for this model -- `need = n_dev*ceil(ne/n_dev)*2 B` x3 buffers -- and the
steady-state headroom covers them.)

**Three log findings (diagnostic/cosmetic; none affects the result).**

1. **Stray debug prints ship in the delivery.**  `mmb.cu:2122` (`MMB_TALL`), `:2137` (`MMB_BLK16`),
   `:2354` (`MMB_DOWN16`), `:2457` (`MMB_GLU`), `:2499` (`MMB_SHADOW`, which *keeps* firing every 50th
   hit) are **unconditional** `fprintf(stderr, ...)`.  The file's own convention is to gate these behind
   `GGML_CUDA_MMB_LOG` (`lg`), which the neighbours at `:1598`, `:1946` and `:2069` do.  The user's log
   carries two `MMB_TALL(wide 384x64)` lines.  **Proposal: gate them behind `GGML_CUDA_MMB_LOG`.**
2. **`ggml_cuda_slab_extend`'s message reports free VRAM measured BEFORE the mapping.**  It logs
   "now 22.06 GiB: ... 6.18 GiB left free", but 6.18 GiB is `free_b` read *before* the 2.12 GiB mapping;
   the true steady state is ~4.06 GiB = the headroom, i.e. the design working as intended.  Easy to
   misread (this session did).  **Proposal: log `free_b - add`.**
3. **`alias_find_checked`'s stale-alias count is never surfaced.**  The guard fired once (table layer 15
   vs op layer 11, same device) and correctly refused the alias; `g_alias_stale` counts, but nothing ever
   reports it, so "happened once" is indistinguishable from "warned once, happened 100k times".
   **Proposal: report it at teardown (or under `MOE_EXPERT_CACHE_VALIDATE`).**  Not a correctness issue --
   the guard is what protects `moe_cache_tally_kernel`.

**Explained -- retire the item:** the teardown `Meta()/CPU compute buffer size of X does not match
expectation of Y` warnings (512 vs 768 and 2304 vs 8448 MiB).  `backend_buf_exp_size` is captured **once**,
in `llama_context::sched_reserve()` (the first, widest layout), and `~llama_context` compares it against the
*current* sched buffer size -- which under the now-default drop is the narrow re-reserve.  The mismatch is
therefore structural whenever the layout changed during the run.  **Proposal: refresh
`backend_buf_exp_size` in the drop's narrow re-reserve** (it also feeds the memory report at
`llama_context.cpp:4050`), or downgrade the message once a drop has occurred.

**Still open from earlier sessions** (unchanged): the `MOE_EXPERT_CACHE_MIN_MIB` late-disable corruption
(defaults to 0); the stale `build-rocm-r16` reference build; the `--fit`/Meta-assert interaction; the parked
unit-map/tail-prune.  And the next battle the maintainer named: under `-sm tensor -ncmoe` the **expert
weights are mirrored across the GPUs and should be split** (tracked as TODO #44).

### SESSION 8 (2026-10-07h): the transient-into-slab work -- implemented, ENGAGED, REVERTED (and why)

**What was built (2nd attempt, better design).**  Instead of looping `moe_cache_shrink_step` (which frees
whichever table is largest, i.e. SCATTERED ranges that may never coalesce into one run), the fallback got a
new slab primitive: `ggml_cuda_slab_arena_alloc_transient(device, size)` takes the top band
`[mapped - want, mapped)` and evicts it through `moe_cache_evict_slab_range` -- the SAME range eviction a
boundary move uses -- so the freed space is contiguous BY CONSTRUCTION.  Both consumers then use it:
`ggml_cuda_pool_leg::alloc` (with a `slab_ptrs` set + `release_block`, so a slab block returns to the slab's
free list and never to `cudaFree`) and `ggml_backend_cuda_buffer_type_alloc_buffer` (with
`ctx->slab_backed`/`slab_size`, same on the destructor path).

**It works.**  Logged, live: `served a 16.82 MiB / 16.82 MiB / 33.65 MiB workspace transient from the
movable-boundary slab`, alongside the boundary moves (`evicted 4728.0 / 5118.0 MiB of arena tables`).  So
llama.cpp's OWN transients -- including the 762 MiB FA-QSA workspace and the 762 MiB MTP-draft buffer that
motivated all of this -- can now be satisfied from the arena.

**But it does NOT make a thin headroom safe, and the experiment that showed that is the reason it was
reverted.**  At `GGML_CUDA_SLAB_HEADROOM_MIB=2048` (`-ub 4096 -c 163860`, arena 74.0 %):

| configuration | outcome |
|---|---|
| headroom 2048, hipBLASLt on | `hipModuleLoad failed` for the Tensile `.co` files (6x; 0x in EVERY successful run) then `Hip error: 'out of memory' at hipblaslt.cpp:164` -> ABORT |
| headroom 2048, `ROCBLAS_USE_HIPBLASLT=0` | server survives, but the output is **CORRUPT** (`////` on 3 of 4 requests) |
| headroom 4096, `ROCBLAS_USE_HIPBLASLT=0` | no corruption, but a 16k request returned 1 token then EOS |
| headroom 4096, hipBLASLt on (the shipped default) | all coherent: short 23.5, 16k 23.2, short 43.9, 16k 27.2 t/s; arena 37861.6 MiB (66.7 %); 348 MiB free |

**Two conclusions.**
1. **The ROCm BLAS stack (hipBLASLt) is a third consumer that cannot be redirected.**  llama.cpp never calls
   `cublasLtMatmul` (grep: no call site outside `vendors/`); this build routes GEMMs through hipBLASLt, which
   loads its own Tensile code objects and allocates its own workspace behind the application's back.  Those
   need REAL free VRAM.  So the headroom must cover the runtime's allocations, not just llama.cpp's -- the
   design cannot make it zero.  **Disabling hipBLASLt is not a workaround** (it corrupted output at a thin
   headroom and stunted a request at a generous one), which is worth recording because it is widely
   recommended elsewhere.
2. Because of (1), the fallbacks cannot be justified as "this lets the headroom shrink", and their only
   remaining effect -- silent corruption instead of a clean abort at a thin headroom -- is strictly worse.
   **REVERTED** (`git checkout -- .` + re-apply `open2-ideaB-vmm-compute.diff`, the verified session-7 state;
   0 fallback markers left in the tree).  The headroom default stays **4096**.

**Also recorded from this session (maintainer observation, open):** during a thin-headroom run the cards sat
at ~31 GB, an allocation failed, and the used VRAM fell to ~13 GB each while the server continued.  Nothing in
this design unmaps client memory, so that needs a reproduction with continuous monitoring -- see the open item
in "OPEN ITEMS" above.

### SESSION 7 (2026-10-07g): THE RESERVE RIGHT-SIZING (landed) + headroom default 4096

**The complaint:** a short prompt must not hold the maximum reserve, because the arena is sized against
that boundary -- so a chat of short prompts got a small cache and a slow decode.

**Root cause.**  The compute reserve is created for the WIDEST graph the parameters allow, and the work
region holds it until a drop releases it.  The drop condition was `n_tokens_prev > 8 && ubatch.n_tokens <= 8`
-- a WIDE pass followed by a narrow one -- so a workload that never prefills never dropped, and the reserve
was held for the whole run.  (The MoE sizing then sized against it: see Session 6.)

**Fix.**  Under the slab a drop is cheap and safe (the layout comes back via a boundary move), so the
condition became "the live work region is materially wider than the narrow layout we last settled on":
self-calibrating, no history guess, and it fires on the FIRST narrow pass.  New plumbing, mirroring the
existing rearm hook: `ggml_backend_dev_slab_work_size(dev)` (0 = no slab) via the device iface, plus a
per-device record in `llama_context` (`slab_narrow_boundary`) written AFTER each drop's narrow re-reserve.
A 512 MiB slack keeps a 1-token vs `n_rs_batch`-token verify difference from thrashing the layout every
token.  **Without a slab the classic wide-pass condition is kept**, because there reclaiming the wide layout
is exactly what used to abort.

**Results** (`-ub 4096 -c 163860`, 2 GPU, SHORT prompt FIRST -- the case that used to keep the reserve):

| headroom | arena before | arena after | 16k request |
|---|---|---|---|
| 2048 | 32263.1 MiB (56.8 %) | **42018.8 MiB (74.0 %)** | ABORTS (FA workspace) |
| 4096 | ~28167 MiB | **37861.6 MiB (66.7 %)** | OK, 16k coherent, warm decode 44.3 t/s |

So **+9.7 GiB total (+4.9 GiB/card)** at equal headroom, and the log now shows the intent:
`work 2.00 GiB + arena 21.06 GiB` at sizing.

**Headroom default raised 2048 -> 4096.**  Not slack: it must cover the largest TRANSIENT workspace, and
those abort (CUDA_CHECK).  The transient grows with the CONTEXT, so a fixed 2048 is not safe for all
configs: at `-ub 4096 -c 163860` a 16k prompt aborts inside `ggml_cuda_flash_attn_qsa3`'s workspace at 2048
(while the arena is 74 %), and 4096 is reliable.  At `-c 32768`/`-ub 8192` 2048 was fine.  Getting back to
2048 -- worth ~4.2 GiB more arena here -- needs the Session 5 work (let the transients draw on the slab's
yieldable arena), which is now the top remaining item.

**Verified with the new defaults:** short (23.5 t/s), 16k (15995 tokens, coherent, `////`=0), then short
again at **44.3 t/s** (warm cache), arena 37861.6 MiB (66.7 %), server alive.

### SESSION 6 (2026-10-07f): THE ARENA SIZING FIX (landed)

**Root cause of the size-timing wart.**  `alloc_all_locked` sizes the arena against `mapped - boundary`, and
the sizing fires on the SECOND pass over a table (the first only primes).  During a PREFILL that second pass
is the second PREFILL ubatch -- before the drop releases the wide work layout -- so the arena was sized
against a boundary that still carried the wide reserve.  Measured on `-ub 4096 -c 163860`:
`work 6.75 GiB + arena 12.31 GiB` instead of `work 2.00 GiB + arena 17.38 GiB`.

**Fix (moe-expert-cache.cu, the sizing trigger).**  Size only on a DECODE-BAND pass (`n_tok <= 8`), matching
the drop's own condition, so the first sizing after a prefill happens after the drop.  With the drop disabled
the boundary is still wide at that point, which is correct there -- the work region really is occupied.

**Results** (`-ub 4096 -c 163860`, 2 GPU, long prompt FIRST, 16k coherent, `////`=0):

| headroom | arena BEFORE | arena AFTER | per card |
|---|---|---|---|
| 2048 (default) | 32263.1 MiB (56.8 %) | **34645.7 MiB (61.0 %)** | 16.9 GiB |
| 6144 | 24059.2 MiB (42.4 %) | **33258.9 MiB (58.6 %)** | 16.2 GiB |

The extension log after the fix confirms the intent: `work 2.00 GiB + arena 17.38 GiB`.  The 16k request also
survives at the DEFAULT headroom now (it did not before).

**REMAINING ORDERING SENSITIVITY (next step): a SHORT request arriving FIRST still sizes early.**  It is a
decode-band pass (6 tokens <= 8), no drop has happened, and the work region still holds the LOAD-TIME reserve
(the widest graph, ~6.75 GiB at `-ub 4096`), which is only ever released by a wide-prefill -> decode drop.
So a short-first workload keeps the wide reserve and gets the smaller arena (32263.1 MiB at headroom 2048).
The reserve really does occupy that memory, so the arena is not wrong -- the RESERVE is: it is sized for the
widest possible graph and is never right-sized to the observed workload.  The fix is to let the drop fire on
a wide -> narrow transition in general (not only after a wide prefill), i.e. right-size the reserve to what
the workload actually needs; that makes the arena order-independent.  Worth re-doing the transient-into-slab
work (Session 5) AFTER it, since with a correctly sized arena the pressure that triggered those fallbacks may
not arise at all.


### SESSION 5 (2026-10-07e): transients inside the slab -- attempted, REVERTED, and the real blocker isolated

**The question.** `--fit` with a 2 GiB target works; can we grab 1 GiB back after the fit and fall back if it
fails?  **No, and the reason matters:** the demand that actually fails is not a load-time need.  From the
headroom-1024 run, with the server already listening and a 16k request in flight:

```
0.16.07  listening                        <- fit and load already DONE
0.29.65  allocating 762.01 MiB ... cudaMalloc failed   (ggml_backend_cuda_buffer_type_alloc_buffer)
0.35.51  allocating  32.06 MiB ... cudaMalloc failed
0.35.51  ROCm error: out of memory  ->  ggml_cuda_pool_leg::alloc (flash_attn_qsa3)
```

So the consumers are (a) the **workspace pool** (`ggml_cuda_pool_leg`, a 762 MiB FA-QSA workspace) and
(b) a **762 MiB buffer** allocated by the MTP draft machinery (`common_speculative_impl_draft_mtp::process`)
plus small 32/64 MiB bits.  They arrive PER REQUEST at long context, so anything taken after the fit is taken
permanently (ROCm will not unmap a sub-range of the slab's one mapping) and the next long request aborts
inside `CUDA_CHECK` -- there is nothing to "fall back" to.  ~1 GiB free also failed a **762 MiB** request,
so the 2 GiB is not slack: it is the workspace high-water mark.

**The right shape** (maintainer's own reading, and it is correct): those transients belong *inside* the slab
when the slab is on, because the arena is the only YIELDABLE consumer -- freeing a table returns its range to
the slab's free list, so the arena can serve them, whereas a `cudaMalloc` outside the slab can never be
rescued by the arena.  Two fallbacks were implemented and DID engage:

* `ggml_cuda_pool_leg::alloc`: on `cudaMalloc` failure, yield tables (`moe_cache_shrink_step`, until it fits)
  and serve the block from `ggml_cuda_slab_arena_alloc`; the block is tracked in `slab_ptrs` so `free` /
  `clear_pool` return it via `ggml_cuda_slab_arena_free` (never `cudaFree`).
* `ggml_backend_cuda_buffer_type_alloc_buffer`: the same, with `ctx->slab_backed` / `slab_size` so
  `free_buffer` returns the range to the slab.  This is the path the MTP draft 762 MiB buffer takes.

Evidence they work: `served a 225.03 / 225.03 / 64.10 MiB workspace transient from the movable-boundary slab`.
**But the 16k request still failed** -- at headroom 512 and at the default 2048 -- so they are NOT a fix on
their own, and a fixed 4-table yield cap was too small (4 x ~66 MiB cannot cover 762 MiB).  **The changes were
REVERTED** (`git checkout -- .` + re-apply `open2-ideaB-vmm-diff` at the verified commit) to leave the tree at
the last validated state, and the design is recorded here rather than shipped unvalidated.

**THE REAL BLOCKER, isolated by the revert: the arena's size depends on WHEN the one-shot sizing lands
relative to the drop, and that decides whether a 16k request fits at all.**  Same binary, same args:

| sizing moment | sizing log | 16k request |
|---|---|---|
| after the drop | `work 2.00 GiB + arena 18.00 GiB` | **OK** (arena 35920.8 MiB) |
| before the drop | `work 11.50 GiB + arena 12.69 GiB` | **FAILS** (out of memory in the FA workspace) |

The sizing is a one-shot at the first decode and takes `mapped - boundary`; if the wide view is still live it
under-sizes the arena by ~5 GiB/device.  Fix this FIRST: either size the arena from the NARROW (post-drop)
need, or re-size once after the first re-arm, or defer the sizing until the boundary is narrow.  The
transient-into-the-slab work above is worth redoing AFTER that, since the two effects are confounded
(a bigger arena means the pressure that triggered the fallbacks may not arise at all).

**Verified at this commit** (`d0daadc`): cli DoD arena 39357.2 MiB / acc 0.92448 / hit 0.9639 / `////`=0;
server default (drop on, extension on) 0 aborts and coherent wide1+wide2; and a live `llama-server`
(4 slots, `--kv-unified`) is serving 16k prompts coherently.  **NOT verified:** any configuration with a
headroom below 2048 MiB -- do not claim ~31 GiB used works yet.

### SESSION 4 (2026-10-07d): drop-on default, VRAM reclaim, and a FALSE ALARM corrected


**Drop-on is now the DEFAULT for every tool** (`common/common.h`, so also `llama-server`; `cli.cpp` still sets
it explicitly).  The old "cli only" split existed because a server could not RECLAIM a wide layout after
releasing it; under the slab that reclaim is the boundary move, and it is verified (wide1 -> short -> wide2
at `-ub 8192`: 0 aborts, coherent, evicted 9786+9792 MiB then re-armed to 37915.6 MiB).  Effect on the
server: arena **15908.8 -> 38026.8 MiB** (28.0 % -> 67.0 % residency).  `LLAMA_DROP_COMPUTE_BUFFERS=0` is
the kill switch.

**VRAM reclaim: `ggml_cuda_slab_extend`.**  The slab is created during the FITTING PROBE, so
`GGML_CUDA_SLAB_RESERVE_MIB` is a guess (8 GiB) and ~1.9 GiB/card of it sat idle afterwards.  The slab now
reserves VA for `free - headroom` but MAPS only `free - reserve` at creation, and once the weights / KV /
draft are resident (the cache's sizing is the first such moment) it maps the remainder down to
`GGML_CUDA_SLAB_HEADROOM_MIB`.  This is the design's second and last HIP touch: it maps a NEW range above
everything in use, so no address moves and nothing is unmapped.  Measured (cli DoD, headroom 1024): arena
**36584.8 -> 39357.2 MiB** (64.5 % -> 69.3 % residency), hit 0.9555 -> 0.9639, decode **75.5 -> 78.3 t/s**,
acceptance unchanged at 0.92448, `////`=0.

**PUSHBACK, with measurements: ~31 GiB used is a CLIFF, not headroom.**  The space outside the slab must
cover the largest TRANSIENT workspace, not the steady-state weights/KV, and those are allocated by
`ggml_cuda_pool_leg` with `CUDA_CHECK` -- a failure ABORTS.  With the headroom at 1024 MiB the run aborted
inside `ggml_cuda_flash_attn_qsa3`'s workspace allocation (VRAM was at 30.99/31.00 GiB, i.e. exactly the
target); at 1536 MiB a 16k request failed; at 2048 MiB it is reliable.  So the default is **2048** (cards
sit at ~30.0 GiB, i.e. what they did before) and `GGML_CUDA_SLAB_HEADROOM_MIB=1024` is the opt-in for
~31 GiB with that risk.  The cliff scales with context length.

**CORRECTION -- the server crash I reported is a TEST-HARNESS ARTIFACT, not a slab regression.**

| configuration | starts | crashed |
|---|---|---|
| r21 build (`build-rocm-r16`, no slab) | 5 | 0 |
| this build, `GGML_CUDA_SLAB=0` | 5 | 0 |
| this build, slab ON, 2 s between starts | 21 | **9** |
| this build, slab ON, **15 s** between starts | 5 | **0** |
| this build, slab ON, interleaved 6x | 6 | 0 |
| this build, slab ON, `-fit off` | 5 | 0 |

The crash is `ggml-backend-meta.cpp:519 GGML_ASSERT(ggml_backend_buffer_is_meta(meta_buf))` during
`--fit`, reached via `ggml_backend_meta_buffer_n_bufs`.  It needs the fit to START with less free VRAM than
it expects: the failing runs always show **7 fitting rounds instead of 2**, and the 2 s restart loop trips
that because the previous process's ~20 GiB of slab is not released yet.  With a 15 s gap, or with
`-fit off`, it never reproduces -- and `-fit off` starts in 12 s vs 18-24 s.  So this is a PRE-EXISTING
fit/Meta robustness bug (the `archive/work/host-pinned-buffer-crash` campaign's territory), made visible by a rapid
restart, not by the slab.  Also fixed while chasing it: the slab view now REPORTS the size that was
requested rather than the boundary (`ggml_vbuffer_size()` feeds both the graph allocator's accounting and
the fit's; a 512 MiB probe buffer claiming 11.5 GiB was wrong on its own terms).

**OPEN FRAGILITY (needs a decision): the arena size depends on WHEN the sizing lands relative to the
drop.**  The sizing is a one-shot at the first decode, and the boundary at that instant sets the arena:
`work 2.00 GiB + arena 18.00 GiB` when the drop had already fired, vs `work 11.50 GiB + arena 12.69 GiB`
when the wide view was still live -- both seen on otherwise identical starts (and the latter also failed a
16k request).  Either size the arena from the NARROW need, or re-size once after the first re-arm.

**Verified in the final state:** cli DoD 39357.2 MiB / acc 0.92448 / hit 0.9639 / `////`=0; server default
(drop on) 0 aborts, wide1+wide2 coherent; a live `llama-server` (4 slots, `--kv-unified`) served a 16k
prompt coherently at arena 35920.8 MiB.

### SESSION 3 (2026-10-07c): the hard cache floor, the allocator clean-up, and a LATENT correctness bug


**Hard cache floor + streaming fallback (maintainer's rule).**  The slab is created only if it can fit
`estimated max work buffer + 2048 MiB` (`GGML_CUDA_SLAB_MIN_ARENA_MIB`) after the reserve; otherwise it
logs an ERROR naming the numbers and the cache **STREAMS** the experts from the host (the stock `-ncmoe`
path), which is byte-identical to `MOE_EXPERT_CACHE_MIB=0`.  `work` is the first work need — the widest
layout the fitting probe reserves — so it IS the estimate; a later graph needing more moves the boundary
again and eats into the floor, which is what `MOE_EXPERT_CACHE_MIN_MIB` (below) is the second guard for.
Verified: `GGML_CUDA_SLAB_MIN_ARENA_MIB=20000` declines, and the run gives MTP acceptance **0.89506
(145/162) — bit-identical to `MOE_EXPERT_CACHE_MIB=0`** at 26.9 t/s, `////`=0.

**THE DISABLE MUST HAPPEN BEFORE THE FIRST CACHE-CONSULTING GRAPH.**  The first cut put the new check in
`alloc_all_locked` (the sizing, ~0.26 s, after the first prefill's graph) and it produced MTP acceptance
**0.00342 (2/585), 7.3 t/s** instead of 0.89506 — coherent-looking text, silently broken draft.  Reason:
`MOE_EXPERT_CACHE_MIB=0` is inert because `moe_cache_init` returns *before registering any table*; a late
`g_enabled = false` instead leaves the registered tables and the graphs already built against them, and
the draft and target then disagree.  The disable now happens from the slab's own decision at the first
compute-buffer allocation (~0.15 s, before any MoE graph) via `moe_cache_disable_streaming()` (called with
no slab lock held), and `alloc_all_locked` gained an `if (!g_enabled) return;` so a cache disabled that
early never sizes or allocates an arena.

**LATENT BUG FOUND (pre-existing, in the shipped set): `MOE_EXPERT_CACHE_MIN_MIB`'s auto floor has the
same flaw.**  `alloc_all_locked` flips `g_enabled = false` at sizing — the same late flip — so setting
`MOE_EXPERT_CACHE_MIN_MIB` (e.g. to decline a small arena) yields **acceptance 0.00342 / 7.7 t/s** rather
than the streaming path (measured directly: `MOE_EXPERT_CACHE_MIN_MIB=100000`).  Severity: the floor
DEFAULTS TO 0 (disabled), so this is only reachable when the knob is set explicitly — not on a default
run.  Not yet fixed: the auto floor cannot decide before the first graph (it needs the registered table
set), so the fix needs its own investigation (options: make the consumers consult a live "usable" answer,
or un-register the tables on a late disable).  **Promotion candidate; the maintainer decides.**

**Dead-allocator clean-up (maintainer asked).**  REMOVED: the per-allocation VMM pool in its entirety
(`GGML_CUDA_COMPUTE_VMM`, `ggml_cuda_vmm_{enabled,reserve,release,map,unmap,unmap_units,alloc,free,owns,
map_with_yield,shrink}`, the pool struct/state, the buft's pool branch, the destructor's `vmm_owns` branch,
and every `use_vmm` branch in `moe_expert_cache.cu` ~300 lines); the parked `moe_cache_prune`; and the
`MOE_EXPERT_CACHE_YIELD_WHOLESALE` branch.  KEPT, deliberately: (a) `ggml_cuda_pool_vmm` — that is
UPSTREAM's workspace pool, not ours, and it is compiled out unless `GGML_USE_VMM` is set for it (verified:
`new_pool_for_device` keeps its own guard, so fixing the VMM capability detection did not switch it on);
(b) `moe_cache_shrink_step` / `moe_cache_release_arena` / the OOM yield — the plain-path fail-soft, which
is live on every non-HIP backend, because the slab is HIP-scoped by design.  `alloc_table_locked` now has
exactly two backings: the slab range, or `cudaMalloc`.

**Verified after the clean-up:** cli DoD unchanged (prefill 1714 / decode 75.5 / acc 0.92448 / hit 0.9555 /
`////`=0); `GGML_CUDA_SLAB=0` (the cudaMalloc path whose VMM branches were deleted) coherent at acceptance
0.81977; the min-arena fallback bit-identical to streaming; and a `llama-server` (4 slots, `--kv-unified`,
16k prompt -> short -> 16k, plus `/v1/chat/completions` with `cache_prompt: true`) served all of them
coherent with 0 aborts (arena 15908.8 MiB, hits 0.73-0.85 — the drop-off default, so the wide work region
holds its space).

**Note on the placement claim:** top-down arena placement does NOT make the *hottest* tables the last to be
evicted when the arena is full (the steady state here) — with the region fully used, the eviction order
follows the allocation order (layer order either way).  The real win is that a boundary move walks through
EMPTY chunks while the arena has slack (a capped/`MIB`-limited arena, or the re-arm after a drop).
Hotness-last needs hotness-based *placement* (admit the coldest into the low band), which we do not have.

### SESSION 2 (2026-10-07b): the corruption bug, the placement policy, default-on

**BUG (silent corruption): `work_live` was a `bool`.**  Two compute buffers are live at once (the main
context and the MTP draft context each own one), so releasing either cleared the flag and made the slab
believe the work region was idle.  The next `work_alloc` then **shrank the boundary under a still-live
view** and the arena re-took chunks that view's tensors were using.  Symptom: `draft acceptance = 0.01047`
and `////` in the output, with the arena inflated to 39578 MiB.  Fix: a `std::multiset<size_t> work_needs`
holding every live view's **requested** size; the boundary may shrink only down to `max(work_needs)`.

Why *requested* and not *reported*: the view's tensor layout is bounded by the size the graph allocator
asked for (`ggml_vbuffer_alloc` passes the tallocr's `max_size`, rounded up plus one spare unit), and the
allocator keeps its offsets inside it.  The buffer's REPORTED size is only the boundary we advertise.  Using
reported would pin the floor at the widest boundary any view was ever handed (the MTP draft context
allocates while the boundary is wide) and the shrink would never fire again — measured: arena 17128 MiB,
decode 48.5 t/s instead of 36584 MiB / 74.2 t/s.

**This is why the gate matrix has to include the DROP-OFF and 3-GPU/serve paths.**  Before the fix the
drop-on paths were all green; only the default server config (drop OFF) corrupted.

**Arena placement is the eviction policy.**  A boundary move can only take a CONTIGUOUS range of chunks at
the bottom of the arena, so there is no "evict the coldest table" decision at move time — the only lever is
where a table SITS.  `ggml_cuda_slab_arena_alloc` now allocates from the **highest free run, at its top**, so
the band just above the boundary stays free while the arena has slack and a move walks through empty chunks.
(A bottom-up fill put the first table hard against the boundary and made every growth cost a table.)

**`GGML_CUDA_SLAB` flipped to DEFAULT ON.**  Rationale: r21 (the delivered set) ABORTS in the server DoD
config, which is worse than a ~4 % decode cost; the env var is now a kill switch.  Init is fail-soft so this
is safe on unvalidated devices: it declines unless `devices[].vmm` is set, no driver call can `GGML_ABORT`
(`cuMemAddressReserve`/`cuMemMap`/`cuMemSetAccess` are all checked and cleaned up), the chunk must be a
power of two, and any failure falls through to the VMM pool / `cudaMalloc`.  **The maintainer should confirm
this default before it is promoted into a block.**

**The reserve is a HARD constraint of the one-mapping design.**  `GGML_CUDA_SLAB_RESERVE_MIB` (default
`max(8192, 25 % of the device)`) is what stays outside the slab for the weights / KV / workspaces.  Because
the slab is created during the FITTING PROBE (before the weights exist) and ROCm will not unmap a sub-range
of its single mapping, a too-small reserve cannot be recovered from: freeing an arena table returns bytes to
the slab's free list, never to the driver.  Measured: 4096 MiB -> a failed 32 MiB hipBLASLt workspace
allocation, exit 134.  The generic OOM path in `ggml_cuda_device_malloc` therefore SKIPS the arena churn when
the slab is live and names the reserve instead.

**CAUTION about the r21 reference build.**  `build-rocm-r16` is STALE (`llama-batched-bench` is from 05:53,
before the r21 tip).  It is 2x slower than a same-config run on the new build in `llama-batched-bench`
(`npp 16 ntg 32`: B=1 12.4 vs 27.1 t/s) while matching on `llama-bench tg64` (27.3 both) and on the MoE cli
(64.3 vs 63.8).  Do NOT read that 2x as a win.  Rebuild a clean r21 worktree before using it as a reference.

### Environment

| var | default | meaning |
|---|---|---|
| `GGML_CUDA_SLAB` | `1` | **the kill switch**; `0` restores the plain allocation path (and the server abort) |
| `GGML_CUDA_SLAB_CHUNK_MIB` | `64` | the boundary-move chunk (must be a power of two) |
| `GGML_CUDA_SLAB_RESERVE_MIB` | `max(8192, 25 % of the device)` | left OUTSIDE the slab — the weights / KV / draft / workspaces must fit here, and a too-small value ABORTS the run |
| `GGML_CUDA_SLAB_MIN_ARENA_MIB` | `2048` | **the hard cache floor**: the slab is created only if it fits `estimated work buffer + this`.  Otherwise it declines, logs an error and the cache STREAMS from the host |
| `GGML_CUDA_SLAB_HEADROOM_MIB` | `4096` | left free OUTSIDE the slab after `ggml_cuda_slab_extend` reclaims the unused reserve.  **Not slack: it must cover the largest transient workspace** (those abort). `1024` gives ~31 GiB used but aborts in the FA-QSA workspace; `0` disables the extension |
| `GGML_COMPUTE_BUFFER_CHUNK_MIB` | `256` | workspace allocation = `(ceil(need/C)+1)*C` (whole chunks + one spare; absorbed growth, rare re-alloc, the spare also guards over-reads). `0` -> `GGML_COMPUTE_BUFFER_MARGIN_PCT` (10) |
| `GGML_CUDA_COMPUTE_VMM` | — | **REMOVED** (the per-allocation VMM pool was retired; see below) |
| `MOE_EXPERT_CACHE_VALIDATE` | `0` | `1` structural validator, `2` + arena-head finiteness (debug; no-op when unset) |
| `MOE_EXPERT_CACHE_YIELD_WHOLESALE` | — | **REMOVED** (the wholesale yield branch is gone; the slab evicts per table) |
| `MOE_EXPERT_CACHE_MIB` / `_RESERVE_MIB` | pre-existing | explicit arena budget / the reserve held out of the auto budget |

### Files changed (vs the r21 tip; full diff = `open2-ideaB-vmm-compute.diff`, ~1360 lines)

* `ggml/src/ggml-cuda/ggml-cuda.cu` — the slab (`ggml_cuda_slab_*`), the compute-buffer view, the
  `slab_view` flag/destructor, the older VMM pool, `get_compute_chunk_bytes`.
* `ggml/src/ggml-cuda/ggml-cuda-vmm.h` — the slab + VMM API.
* `ggml/src/ggml-cuda/moe-expert-cache.cu/.h` — the per-table gates, `moe_cache_evict_slab_range`, the
  slab-backed arena (`alloc_table_locked`), `moe_cache_rearm`, `moe_cache_prune` (written, parked),
  `moe_cache_validate`, `table_arena_bytes`/`free_arena_backing`.
* `ggml/src/ggml-alloc.c` — the chunk-quantized work allocation (`ggml_vbuffer_chunk_alloc_size`).
* `ggml/src/ggml-backend-impl.h`, `ggml-backend.cpp`, `ggml-backend-meta.cpp` — the appended buft capabilities
  `alloc_buffer_usage` and `get_compute_chunk_bytes` (threaded through the Meta buft), and the re-arm
  device-iface hook.
* `ggml/include/ggml-backend.h`, `src/llama-context.cpp` — `ggml_backend_dev_moe_cache_rearm`, called at the
  prefill -> decode drop site after the narrow layout is re-reserved.

### Repro / build

```bash
# build (the working dir; gate the slab at RUNTIME, not compile time)
cd ~/llama.cpp
cmake --build build-rocm-vmmc --target llama-cli llama-server -j 16

# cli DoD (note GGML_CUDA_SLAB=1)
BIN=./build-rocm-vmmc/bin/llama-cli archive/work/moe-cache-autosize/open2-harness-cli.sh slabdod 2000 GGML_CUDA_SLAB=1

# server safety (wide1 -> short -> wide2)
BIN=./build-rocm-vmmc/bin/llama-server archive/work/moe-cache-autosize/open2-harness-server.sh sl2 \
    GGML_CUDA_SLAB=1 LLAMA_DROP_COMPUTE_BUFFERS=1
```

The harnesses are copied into this directory; `BIN=` selects the build, `LOGDIR=` the log dir (default
`/tmp/item1`).  Server mode knobs: `SRV_SEQ=1` (sequential), `SRV_UB=<n>`, `SRV_NPRED=<n>`; do NOT pass `-np`
without `--kv-unified` (the 16k prompt gets a 400).  `llama-cli` always `--single-turn`; never run two
benchmarks at once.

### Remaining work ("polishing") -- SUPERSEDED by "OPEN ITEMS + the r22 fold plan" above; kept as the session-2 record

1. ~~**Eviction policy.**~~  Done: arena placement is top-down (highest run, its top), which is the only
   lever a positional boundary move has; the move now walks through empty chunks whenever the arena has
   slack.  A hotness-aware *placement* (admit cold tables low) is still available if it ever matters — but
   with a full arena, which is the steady state here, the evicted set is fixed by the position, so the
   remaining idea is to separate the "evict now" cost from residency (e.g. relocate a hot table upward when
   the arena has slack) rather than to choose at move time.
2. **Sweep** `GGML_CUDA_SLAB_CHUNK_MIB` (64) and `GGML_CUDA_COMPUTE_BUFFER_CHUNK_MIB` (256).  The reserve is
   NOT sweepable downward on this box: 8192 is what the weights+KV+draft actually need (4096 aborts), so on a
   32 GiB card there is no slack to recover.
3. **Wider gates:** done this session for 3-GPU, the dense transparency check, the rule-0 MTP gate and the
   rule-5 `llama-batched-bench` A/B.  Still open: a *clean* r21 rebuild as the stock reference (the current
   `build-rocm-r16` is stale), `-ncmoe 0` byte-identity on a model that fits, and the 4-axis adaptive-MTP
   axes + `prompts/code-reasoning-mixed.txt`.
4. ~~**Default-on decision + `ENVIRONMENT.md`.**~~  Default flipped ON (see above) — **needs the maintainer's
   confirmation**, since it trades ~3.6 % decode for the r21 abort.  `ENVIRONMENT.md` is the *delivery's* doc
   and must not list these until the slab is promoted into a block: add `GGML_CUDA_SLAB` (kill switch),
   `GGML_CUDA_SLAB_CHUNK_MIB` and `GGML_CUDA_SLAB_RESERVE_MIB` at promotion time.
5. **Promotion decision (maintainer).**  Three correctness fixes are candidates, all reachable only through
   the slab today:
   * the per-table fallback gate (the old global gate + a partial cache corrupted the output);
   * `moe_cache_build_remap_kernel`'s `slot < n_res` clamp;
   * **`ggml_cuda_slab_work_release`'s live-view floor** — this is a NEW field on the slab struct, so it
     travels with the slab, but the *class* of bug (a release shrinking a shared region under a live view) is
     worth a note wherever the compute buffer is shared.
   Per the WIP rule the maintainer decides; do not fold them into `patches/` unasked.
6. **Parked/dead code to remove or re-justify** if the slab becomes the only path: the older VMM pool
   (`GGML_CUDA_COMPUTE_VMM`), `ggml_cuda_vmm_shrink` (no-op), `moe_cache_prune` (never called).  Also consider
   whether the compute buffer still needs its `+1` spare chunk under the slab (the whole slab is mapped, so
   the over-read guard is automatic there).
7. **The `Meta()` "does not match expectation" warnings** at context teardown are pre-existing (the reported
   size is the boundary) but they are now noisier under the slab; consider reporting the *expected* size when
   the buffer is a slab view.

---

## Historical trail (superseded — do not follow as a plan)

# OPEN 2 — VMM / movable-split slab: cold-start handover (2026-10-06)

**Read this first in a new session.**  It supersedes the `FOLLOWUP-compute-arena-chunking.md` plan for
Idea B.  The design is settled in outline; the next step is a **controlled probe**, then the pool.

## Where we are

* **r21 is shipped and pushed** (`v16-a55e952b8-r21`, fork tip `94c3eeb89`) — the OPEN 1 safety subset
  (arena `t.slots` fix, per-layer re-size, `MTP_DRAFT_N_UBATCH=512`, the wide-prefill drop default-on for
  `llama-cli` only).  Delivery repo `main` at `accd84a`; `validate-set.sh` green.
* The OPEN 1 cli DoD is met by default (`-ub 8192` cache auto 16k: decode **78.7** / prefill **1683**).
  The **`llama-server` half is open**.
* The `~/llama.cpp` working tree is **clean at the r21 tip** (the Idea A change was reverted; its diff is
  `open2-ideaA-chunking.diff`).

## What has been measured (so you do not redo it)

1. **The wide layout is a contiguity problem.**  A later wide prefill on a server needs one contiguous
   `cudaMalloc` back; freeing *all* 288 arena tables (40 GB) does not yield it.  `-ub 8192` and `-ub
   6144` abort; `-ub <= 5120` survives at n=16, but `-ub 4608` at `-n 2000` still faulted
   (`mul_mat_vec_q_moe`, the r20 §13 class) — so `-ub` only moves the cliff.
2. **Idea A (plain compute chunking) helps but is not enough.**  `GGML_COMPUTE_BUFFER_CHUNK_MIB` chunked
   the compute buffer -> the server **survived** a later wide prefill (0 aborts, partial stand-downs) and
   the cli still met the DoD (75.7), but the wide prefill still consumes the arena (65 % -> 26 %) and the
   one-shot sizing **never re-grows it**.  Detail: `OPEN1-FINDINGS.md` (last section).
3. **The `-ub` crossover is state-dependent** (it moved between runs as free VRAM drifted).

## The VMM probe — status: INCONCLUSIVE, needs a controlled re-run

Built `build-rocm-vmm` with `GGML_HIP_NO_VMM=OFF` (full command in §Environment).  The existing
`ggml_cuda_pool_vmm` (a bump allocator over a 32 GB VA reservation) then backs the **workspace** pool.
Results:

| run | result |
|---|---|
| cli `-ub 8192` cache auto, `GGML_COMPUTE_BUFFER_MARGIN_PCT=0`, VMM build, n=16 | exit 0, coherent |
| same, **non-VMM** r21 control, minutes later | **exit 0, coherent too** |
| server, drop ON + margin=0, VMM build, wide1->short->wide2 | **abort** (`11763 MiB cudaMalloc failed`, 288 stand-downs) |

So this probe **does not yet prove VMM does anything**: the margin=0 abort is free-VRAM/fragmentation
dependent, and both builds passed once the box had more free VRAM.  Two facts are nevertheless solid:

* ROCm 7.14's VMM primitives **work** here — the VMM build runs, and the pool's teardown does
  `cuMemUnmap` per mapping + `cuMemAddressFree` with no error.
* Enabling VMM for the **workspace pool only** cannot fix the server, because the failing object is the
  **compute-buffer `cudaMalloc`**, which is not VMM-backed (`ggml_backend_cuda_buffer_type_alloc_buffer`
  -> `ggml_cuda_device_malloc` -> `cudaMalloc`).

**Controlled probe to run first:** pin the box state (fresh boot, or record free VRAM before each run),
then A/B the cli margin=0 repro N=5 on `build-rocm-r16` vs `build-rocm-vmm`, and instrument *which*
allocator fails (`ggml_cuda_device_malloc` vs the pool vs the Q8_1 arena).  Also report the device VMM
capability line (`ggml_cuda_info().devices[d].vmm`, printed at load) to confirm VMM is actually on.

## The design (settled in outline)

**One per-device VMM reservation, split dynamically between the compute buffer and the MoE arena** — the
user's "massive contiguous slab, move the split point up/down":

* Reserve a large VA range per device (`cuMemAddressReserve`); map physical in fixed units
  (`cuMemCreate` + `cuMemMap`) as either side needs them.  Growing the compute buffer = mapping more
  units at its end (its VA is contiguous, so a realloc is a pointer move, not a copy).  Evicting arena
  entries = unmap their units and map them under the compute side.
* **Why VMM is mandatory:** plain `cudaMalloc` cannot split/extend/merge ranges, and the arena is
  stride-addressed (`(char *)t.arena + slot*t.expert_bytes` — 8 call sites) so a table's slab must be
  **one contiguous VA run**.  Units are fungible only if they can be remapped under stable VA.
* **Asymmetry to respect:** the compute buffer is already chunk-friendly (per-tensor chunks, any sizes,
  no straddling), while the arena is not.  So arena->compute is easy even without VMM (hand a whole
  table block over as a chunk); compute->arena is the direction that needs VMM.
* **Must not break:** decode CUDA-graph capture (addresses must stay valid -> stable VA); the arena
  head-pad / head-zero invariants (keyed on `(buffer, expert_bytes)`); the fused-op purity anchors; and
  no arena unit may be remapped while a kernel that carries its address is in flight
  (`moe_cache_sync_devices_locked`; the fused kernels take the arena address as a launch parameter).
* **Per device + peer access:** `cuMemSetAccess` for the P2P/AR/RCCL paths (`-sm tensor`, NCCL build);
  the existing pool already has this logic, mirror it.
* **The existing `ggml_cuda_pool_vmm` is not reusable as-is:** its `free` is LIFO and never unmaps, and
  it is a workspace bump allocator.  A new unit pool (free list, non-LIFO, real unmap, owner shared by
  the compute buffer and the arena) is the deliverable.

## ROCm

Use **7.14.1** (`/opt/rocm-7.14.1-gfx120X`) — its VMM works.  **ROCm 10.1.0** is installed at
`/opt/rocm-10.1.0-gfx120X/` purely as a fallback if 7.14 VMM turns out to be unreliable under the real
pool (map/unmap churn, not just teardown).  The maintainer flagged it on 2026-10-06; do not switch
speculatively.

## Environment / reproduction

```bash
# r21 build (current default; the promoted tree)
cd ~/llama.cpp
BUILD_DIR=build-rocm-r16 EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
cmake --build build-rocm-r16 --target llama-cli llama-server -j 16

# VMM build (GGML_USE_VMM on; workspace pool becomes ggml_cuda_pool_vmm)
BUILD_DIR=build-rocm-vmm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS= -DGGML_HIP_NO_VMM=OFF" ~/bin/build-llama-rocm-714

# harnesses (copied here; BIN= selects the build, LOGDIR= the log dir)
BIN=./build-rocm-vmm/bin/llama-cli    ./archive/work/moe-cache-autosize/open2-harness-cli.sh <tag> 16 [ENV=...]
BIN=./build-rocm-vmm/bin/llama-server ./archive/work/moe-cache-autosize/open2-harness-server.sh <tag> [ENV=...]
# server mode knobs: SRV_SEQ=1 (sequential), SRV_UB=<n>, SRV_NPRED=<n>; server default is 4 unified slots
# (do NOT pass -np without --kv-unified or the 16k prompt is rejected with 400).
```

The harnesses assume `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/...` + `/tmp/pl_16k.txt` (16k prompt) and
`HIP_VISIBLE_DEVICES=0,1`, `-sm tensor -ncmoe 48`.  `llama-cli` always `--single-turn`; never run two
benchmarks at once.

## Next steps (in order)

1. Controlled VMM probe (above) + identify the failing allocator.
2. Implement the **unit pool** (`cuMemCreate/Map/Unmap/AddressReserve/SetAccess`, per device, non-LIFO,
   free list, `GGML_COMPUTE_BUFFER_VMM` style env gate).
3. Back the **compute buffer** with it first (a VMM buffer type or a mode of the CUDA buft); confirm the
   wide prefill's realloc is a pointer move and the server survives.
4. Put the **arena** on it (tables as VA ranges of units) so units can flow back; then the arena can
   re-grow after a wide prefill and the server DoD should fall out.
5. Gates: cli DoD, server concurrent long+short + later wide prefill, 3-GPU coherence, `-ncmoe 0`
   byte-identity, MTP `-n 3000`.

---

## 2026-10-06 (late): VMM prototype — findings (supersedes the "inconclusive" probe above)

**The VMM works.**  A standalone probe (`/tmp/item1/vmmprobe.cpp`) on ROCm **7.14.1 / gfx1201**:
`hipDeviceAttributeVirtualMemoryManagementSupported` = 1, granularity 2 MiB, `hipMemCreate` -> `hipMemMap`
-> `hipMemSetAccess` -> device memset + host readback OK, and — the operation the movable split
actually needs — **`hipMemUnmap` then `hipMemMap` at the SAME VA works and reads back correctly**.  So
**ROCm 10.1.0 is not needed** (keep it only as a fallback).

**Bug 1 (real, in the delivery):** the VMM capability detection in `ggml_cuda_init` was inside
`#if defined(GGML_USE_VMM)`.  With the default `GGML_HIP_NO_VMM=ON`, `GGML_USE_VMM` is undefined, so
`devices[].vmm` was **always 0** — the existing `ggml_cuda_pool_vmm` never activated on this box, which
is why `GGML_HIP_NO_VMM=OFF` changed nothing and why the whole earlier "VMM probe" was meaningless.  Fix:
detect under `defined(GGML_USE_HIP)` too; `new_pool_for_device` keeps its own `GGML_USE_VMM` guard so the
workspace pool's behaviour is unchanged.

**Prototype (diff: `open2-ideaB-vmm-compute.diff`, 5 files, ~250 lines):**
* appended optional `alloc_buffer_usage(buft, size, usage)` to `ggml_backend_buffer_type_i` (defaults to
  `alloc_buffer`; only the graph allocator calls it, with the real usage) and threaded it through
  `ggml_vbuffer_alloc` and the **meta** buft (the graph buft under `-sm tensor` is `Meta()`);
* a per-device VMM pool in `ggml-cuda.cu` (64 GiB VA reservation, first-fit free list, coalescing,
  `cuMemCreate`/`Map`/`Unmap`/`SetAccess` per allocation), env `GGML_CUDA_COMPUTE_VMM` (HIP-only, 0/unset
  = off); the COMPUTE buffer uses it, model weights stay on `cudaMalloc`;
* on `cuMemCreate` OOM it yields the MoE arena (`moe_cache_shrink_step`, then `moe_cache_release_arena`)
  and retries — VMM needs **no contiguous block**, so a partial yield is enough.
* **Bug 2 (prototype):** the coalescing code reused `next` after `erase(next)` (dangling iterator ->
  SIGSEGV in `std::prev`); `erase` now feeds the previous-neighbour lookup.

**Results** (server/k/cli on 2x gfx1201, `-ub 8192`, cache auto, 16k prompt):

| run | result |
|---|---|
| cli, VMM on | exit 0, **coherent AND byte-identical generated text to r21**, arena 70.6 % |
| server, VMM on OR off, **drop off** | coherent (`<think>...`, no `////`) |
| server, **VMM off + drop ON** | **still aborts** (`GGML_ASSERT(bufs.back() != nullptr)`, meta:1817) |
| server, **VMM on + drop ON** | **survives** the fatal wide1->short->wide2 (0 abort signals) but output is garbage (`////`) |
| server, **VMM on + drop ON + cache DISABLED** | **coherent** (isolates the re-alloc from the culling) |

**Conclusion:** the VMM compute-buffer allocation and its free/re-alloc across requests are **correct**.
The garbage is the **arena culling**, not VMM: with the drop on, the first wide prefill stands down
~184-288 tables (`stand_down_table_locked` -> `cudaFree`), and that cull is what corrupts the run.  This
is the latent bug the abort used to hide.

**Next (the design's step 4):** put the **arena** on the same pool so a cull is an `Unmap` (VA stable,
physical returned) instead of a `cudaFree`, and a re-arm is a re-`Map` at the same VA.  Stable arena VA is
also what keeps captured decode graphs and the cache's fusion guard / `moe_cache_take_over` agreement
valid across a split move.  Then re-run the server drop-on gate.

---

## 2026-10-06 (later): ROOT CAUSE FOUND + FIXED — the wholesale-fallback invariant had a third consumer

**The corruption is a real bug in the delivered cache logic, independent of VMM** — VMM is only what
made the failing path reachable (before it, the same allocation always aborted).

`AGENTS.md` states: *"a partially-failed cache falls back wholesale (the fusion guard and
`moe_cache_take_over` must agree)"*.  Two consumers honoured the gate (`ggml_cuda_cache_blocks_fusion`
and `moe_cache_take_over` both check `moe_cache_has_arena_locked()`); a **third** did not:

* `moe_cache_get_table()` kept returning the arena for a **surviving** table, and
* the fill path in `moe_cache_update_host()` kept building that table's remap / filling its arena.

So after a partial stand-down the take-over hook had declined (the scheduler copied `input_cpy`), yet the
op still redirected to an arena/remap the fallback path had stopped maintaining -> the repeated-`/`
corruption.  **Fix:** gate both on `moe_cache_has_arena_locked()`.  In `moe_cache_update_host` the gate must
sit **after** the priming/sizing bookkeeping (placed before it, `alloc_all_locked` never latches and the
cache silently never sizes — measured: no `alloc_all_locked` line, `stood_down=0`).

**How it was found:** the `MOE_EXPERT_CACHE_VALIDATE` validator (new, this campaign) showed the cache
*structurally consistent with every arena head finite* after the cull — ruling out the arena and pointing
at the consumers.  A wholesale-vs-partial A/B (release the whole arena vs stand down single tables) then
isolated it: wholesale = coherent, partial = garbage.

**Also changed:** a compute-buffer yield now **releases the whole arena** by default
(`moe_cache_release_arena`).  A partial stand-down is now correct, but the surviving tables are bypassed
until the run ends, so their VRAM would be wasted — and VMM needs only *physical*, not a contiguous
block, so the whole-arena release frees it in one go.  `MOE_EXPERT_CACHE_YIELD_PARTIAL=1` restores the
partial path for A/B.  New debug knob: `MOE_EXPERT_CACHE_VALIDATE=1` (structural) / `=2` (+ arena
finiteness).

**Results (final build):**

| run | result |
|---|---|
| cli, VMM on, `-n 2000` (the DoD) | prefill **1674 t/s**, decode **78.5 t/s**, MTP acc **0.9245**, `////`=0, arena 40078 (70.6 %) |
| server, drop on + VMM on, cache AUTO | 0 aborts, coherent |
| server, drop on + VMM on, **forced cull** (`MOE_EXPERT_CACHE_MIB=20000`, stood_down=181) | 0 aborts, **coherent** (`<think>...`) |
| server, drop on + VMM on + cache DISABLED | coherent |
| server, VMM off + drop on | still aborts (meta:1817) — VMM is what makes the path reachable |

**Remaining for the server DoD (the design's split-move):** a cull now *works* but leaves the cache off for
the rest of the run, so a later decode is slow.  The **re-arm** is the missing piece: when the compute
buffer is dropped (context idle) re-size the arena to the auto target and re-enable the cache
(`g_sized` is a one-shot today).  With the arena on the pool that re-arm is a re-`Map` at stable VA, which
is also what keeps captured decode graphs valid.  Then sweep the split: wide prefill takes units, idle
returns them, decode gets the cache back.

---

## 2026-10-06 (latest): the unified VMM allocator — built, split_move works, second cycle still bad

**Built (this is the design the maintainer asked for):** one per-device VMM allocator with VA and physical
SEPARATED (`ggml-cuda-vmm.h` + `ggml_cuda_vmm_{reserve,release,map,unmap,alloc,free,owns}` in ggml-cuda.cu).
**Both** the compute buffer and the MoE arena now come from it:

* a table **reserves** its slab VA once and keeps it for life (stride addressing + captured graphs); a
  cull only **unmaps** the physical and a re-arm **maps** it back at the SAME VA;
* `alloc_table_locked`/`stand_down_table_locked`/`free_table_buffers_locked`/`release_arena` all route to
  the pool when `GGML_CUDA_COMPUTE_VMM=1` (legacy `cudaMalloc` otherwise), tracked by `arena_reserved`;
* the pool mutex protects only the VA free list -- `map`/`unmap` take no lock because the yield
  (`moe_cache_shrink_step` -> `unmap`) must not run under it (deadlock otherwise).

**This fixed the allocator-separation problem:** the cull is now an `Unmap` whose units return to the SAME
pool the workspace draws from, so a growth needs no contiguous block and no `cudaFree`/`cudaMalloc` churn.
Forced cull `MIB=20000`: `stood_down=131` (was 288 + abort), **0 aborts**, and the re-arm restored the
arena to 38580.8 MiB with the hit rate going 0.72 -> 0.96 (residents preserved -- no second arena, no copy).

**Still broken: the SECOND cull/re-arm cycle.** wide1 (cycle 1) is coherent; wide2 (cycle 2) is `////`
(a variant faults in `mul_mat_vec_q_moe` on an UNMAPPED arena page).  Sequential requests reproduce it, so
it is not a race.  `srv-vp` (arena on the pool, re-arm not yet effective) was fully coherent, so the trigger
is the re-arm mapping the arena again.  Guards now on all four consumers (`moe_cache_get_table`,
`moe_cache_update_host`, `moe_cache_get_slot`, plus `take_over`), and `stand_down` now clears the
DEVICE-side maps (`slot_dev`, `used_dev`) -- so the cause is NOT an unguarded consumer in the host
decision path.  Leading suspects, in order:

1. **a decision baked before the cull** -- the fused gate+up/mmvq kernel reads the arena from its launch
   parameters (`moe_cache_redirect_fused`), and a CUDA graph captured while the arena was mapped may be
   replayed after a cull/unmap with a stale fusion decision.  `ggml_cuda_cache_blocks_fusion` is evaluated at
   graph build, not at replay.  (Needs a way to disable HIP graph capture to confirm; `GGML_CUDA_GRAPH_OPT`
   is an unrelated switch.)
2. the **device-remap** path (`g_devmap_armed` stays armed across a cull; the re-armed tables'
   `slot_dev` is cleared but the survivors' is not, and the eager re-arm pass is not re-run).
3. the re-armed tables are fresh-physical + `cudaMemset`-zeroed, so a reader that believes they are
   resident reads zeros -- exactly the `////` pattern.

Next: instrument the path decision for the first decode token of each cycle (arena vs `input_cpy`), or find
and use a real "disable HIP graph capture" switch, to separate (1) from (2)/(3).  The fix likely needs the
cache to invalidate the scheduler's captured graphs / force a fusion re-evaluation on a cull+re-arm.

---

## 2026-10-06 (final this session): chunk-quantized workspace DONE; chunk PRUNE blocked by ROCm

The maintainer's design -- discrete uniform chunks that absorb growth ("minimum chunks + one") and a
chunk-level PRUNE instead of a cull/re-arm -- was built.  Status:

**DONE and working: chunk-quantized workspace allocation.**  A new appended buft capability
`get_compute_chunk_bytes` (CUDA: `GGML_COMPUTE_BUFFER_CHUNK_MIB`, MiB, default **256**, HIP-only; meta buft
aggregates it) makes `ggml_vbuffer_chunk_alloc_size` return `(ceil(need/C) + 1) * C` for the COMPUTE
buffer: whole chunks plus ONE spare.  The realloc trigger compares the *allocated* size, so growth inside
the spare is free, the re-alloc only fires past the next high-water mark, and the spare is the mapped
over-read guard at the buffer's end.  Measured: the decode/verify compute buffer went 1870.8 MiB ->
**2048.0 MiB** (8 x 256 MiB, exact), cli coherent, no regression.  `GGML_COMPUTE_BUFFER_CHUNK_MIB=0` falls
back to `GGML_COMPUTE_BUFFER_MARGIN_PCT`.

**ROCm constraint found (probes committed as `vmmprobe2/3/4.cpp`):**
* `hipMemUnmap` of a **sub-range** of a mapping fails (`hipErrorInvalidValue`) -- so a table's tail can only
  be given back if every granularity unit is its OWN mapping (`vmmprobe2`).
* Unit mapping itself works, including unmap + re-map at the same VA, and at full scale (6144 units / 12 GiB,
  no error) (`vmmprobe3`, `vmmprobe4`).

**BLOCKED: the chunk prune.**  The prune was implemented (`moe_cache_prune`, `ggml_cuda_vmm_shrink`, plus a
`slot < n_res` clamp in `moe_cache_build_remap_kernel` that makes a stale device slot map safe after a
prune).  But when the pool maps the 11776 MiB compute buffer as ~5900 unit mappings, an unrelated
`hipMemcpy2DAsync` (cpy.cu:479) starts failing with `hipErrorInvalidValue` at context init -- even though the
standalone 6144-unit map succeeds.  The single-mapping pool does not show it.  So the unit mapping is
**parked**: `ggml_cuda_vmm_map_phys`/`unmap` are back to one mapping, `ggml_cuda_vmm_shrink` is a no-op, and
`moe_cache_prune` is implemented but not called (the reclaim is still `moe_cache_shrink_step`).  The tree is
back to a working state (cli coherent, chunk quantization live).

**Next step to unblock:** find why a unit-mapped VMM range breaks `hipMemcpy2DAsync` in-tree when it works
standalone -- prime suspects: (a) HIP's 2-D copy needs a pitch/limit check that VMM ranges trip only when the
mapping count is high; (b) the copy's source/dest is a *view* whose pitch is fine but whose range the driver
cannot resolve; (c) it is not the mapping count at all but the chunk-quantized SIZE interacting with a
`ggml_cuda_cpy_as_memcpy_2d` path.  A minimal repro is to run the cli with the unit mapping and
`GGML_COMPUTE_BUFFER_CHUNK_MIB=0` (isolates chunk size from mapping count).  Once that is understood, the
prune is a small re-enable (the code is written).

---

## 2026-10-07: THE MOVABLE-BOUNDARY SLAB -- built, and BOTH gates pass

Superseded the whole unmapping story.  The maintainer's design is implemented and works.

**The design (as specified):** ONE slab per device, `cuMemAddressReserve`d and **mapped exactly once**, then
split by a BOUNDARY: the LOW region `[0, boundary)` is the work pool (the compute buffer) and the HIGH region
`[boundary, size)` is the MoE expert-cache arena.  Growing the work pool is a **boundary move inside the
already-mapped slab** -- the lowest arena chunks are reassigned to the work pool (evicting the arena tables
that live there) -- so **HIP is never called at runtime**: only at slab creation and at shutdown.  Because the
work region's base VA never moves, a growing layout keeps its tensor addresses.  This removes every wall the
per-allocation VA pool kept hitting: no partial `hipMemUnmap` (ROCm rejects it), no unit mapping, no
`cuMemCreate`/`cuMemMap` churn at runtime, and no "the HIP allocator won't give the physical back".

**The two bugs that had to be fixed to make it work (both were mine, not the design's):**

1. **The wholesale-fallback gate had to become PER TABLE.**  `moe_cache_has_arena_locked()` is all-or-nothing,
   and the slab makes a PARTIAL cache the NORMAL state (evicting the tables in the taken chunks is routine).
   With the global gate, one boundary move disabled the whole cache -> the decode ran on the host path
   (8 CPU cores busy, GPUs 30 %, decode 7-21 t/s).  Now `moe_cache_take_over`, `moe_cache_update_host`,
   `moe_cache_get_table` and `moe_cache_get_slot` gate on **this table's** arena/slots and still agree per
   table; only the FUSION guard stays global (`moe_cache_has_arena()`), so a partial cache loses the fusions
   but keeps serving its residents.
2. **The arena's allocation unit had to be FINE (the VMM granularity), not the 64 MiB boundary chunk.**
   Chunk-aligning each table's slab wasted up to 63 MiB per table (~9 GiB over 288 tables), so the sizing
   (which does not model the rounding) over-committed and most tables failed to allocate -> the same host-path
   collapse.  The slab is ONE mapping, so a sub-slab allocation unit is pure bookkeeping and can be 2 MiB.
   `alloc_all_locked` also holds back `n_tables * unit` for the rounding.

**Also:** `ggml_cuda_slab_work_release` (called from the compute buffer's destructor) lets the boundary move
**down** when the next work need is smaller (the narrow post-prefill layout), handing the slack back to the
arena; `alloc_all_locked` sizes against the slab's arena region (`ggml_cuda_slab_arena_total`), not
`cudaMemGetInfo` (which reads only the small amount left outside the slab); the slab leaves
`GGML_CUDA_SLAB_RESERVE_MIB` (default 8192) outside itself, because the GPU weights and the KV cache are
allocated AFTER the first compute buffer in this fork.

**Results (gate `GGML_CUDA_SLAB=1`, default off):**

| run | result |
|---|---|
| cli `-n 2000` (the DoD) | prefill **1705 t/s**, decode **74.9 t/s**, MTP acc **0.92448** (= r21 exactly), hit 0.9555, `////`=0 |
| server, drop on, wide1->short->wide2 | **0 aborts**, ALL FOUR responses coherent, arena restored to 35254 MiB |
| server boundary move | `evicted 10028+9994 MiB of arena tables from the taken slab chunks`, arena re-grew after the drop |

So the previously-fatal wide2 case is fixed, and the cli DoD holds.  The r21 DoD numbers were
prefill 1683 / decode 78.7; the slab gives 1705 / 74.9 with the same acceptance -- within noise, and with no
cull/re-arm and no runtime HIP calls.

**Next:** (a) sweep the chunk size / reserve; (b) decide the eviction policy (evicting by address is
arbitrary -- the cache's own hotness data could choose which chunks to give up, or the arena could allocate
tables from the HIGH end so the boundary takes the coldest); (c) MTP `-n 3000`, 3-GPU, `-ncmoe 0`
byte-identity; (d) default-on decision + ENVIRONMENT.md.

---

# Why this exists — the arena is a cache, the slab is what makes it affordable

*(Written to be read on its own; it is the "why" behind the "how" above.  Useful as the opening of a
write-up, an upstream framing, or the release notes for the feature.)*

## The problem upstream's `-ncmoe` has

`-ncmoe` keeps MoE expert weights on the host and brings the ones a token needs to the GPU for that
matmul.  It is a **transport**: it holds no state between tokens, so every ubatch pays the transfer again,
and its only lever is pruning to the experts the router actually picked this token.  The experts are the
bulk of the model in a modern MoE, so the transfer is the bulk of the per-token PCIe traffic — and the
GPU spends the decode stalled behind it (measured here: 8 CPU cores saturated doing the host-side gather
and staging while the GPUs sit at ~30 %).

## What the arena changes

The MoE expert-cache arena keeps the hot experts **resident on the GPU** and fetches only the misses.  It
is a **cache**, not a transport, and the difference is persistence: residency survives across tokens *and
across requests*, so the transfer is amortised rather than repeated.  The number that matters becomes the
**hit rate** — the same quantity this campaign has been grading on, measured at 0.9656 in the cli gate
(and 0.43 in the run where a bug starved the arena, which is precisely what made the GPUs idle).

Everything else in the arena is in service of that number: the slot remap and the resident/cold split, the
fusion guard, the admission policy, the identity fast path, the deferred promotion.  `GREEDY-PURITY.md`
§§19/21/24/25/27 exist because those are the invariants that keep a *cache* byte-identical to the
cache-less oracle — a transport never needed them.

## Why the cache needs the slab

A cache is only worth having when it is **large and long-lived**.  Those are exactly the two properties
that collide with a graph allocator that must also be large and growable: the compute buffer is sized from
a measure graph and can need more at runtime, the arena wants to own everything else, and both are asking
the same physical VRAM for a contiguous block.  That collision is the whole bug:

* on ROCm the driver does not coalesce: freeing the **entire** arena (288 tables, tens of GiB) still could
  not satisfy one 11.7 GiB `cudaMalloc` for a grown compute layout — the free space existed, never as one
  block.  Yielding the arena therefore did not rescue the allocation, and the run aborted.
* the obvious workaround — free more, retry, and re-fill cold afterwards — throws away exactly the
  residency the cache exists to provide, so it defeats the feature.

The **movable-boundary slab** dissolves the collision instead of arbitrating it: one VA reservation per
device, mapped once, split by a scalar boundary into a work region and an arena region.  Growing the work
region is a **boundary move inside memory that is already mapped** — the lowest arena chunks change owner
and the tables living there are evicted — so there is no second allocation, no contiguity requirement, no
fragmentation between the two consumers, and no driver call at runtime at all.  Because the work region's
base never moves, a growing layout keeps its tensor addresses too.

So the two halves are not independent improvements:

* **the arena supplies the value** — amortised expert residency instead of per-token transport;
* **the slab supplies the coexistence** — that residency and the compute layout can both be large, without
  either being able to starve the other, on a driver that would otherwise fragment them apart.

Either one alone only moves the problem.

## How to frame it upstream

**"A GPU-side cache for host-resident MoE experts, enabled by a movable-boundary VA slab."**  The cache is
the product; the slab is the mechanism.

* It is **additive and opt-in**: with the feature off, `-ncmoe` behaves exactly as it does today, and the
  slab falls back to the existing allocation path wherever VMM is unavailable (the `devices[].vmm`
  capability gate already exists for the workspace pool).
* The **small, generalisable part is small**: the slab is a couple of hundred lines of VA bookkeeping —
  reserve once, split by a boundary, move it, free at shutdown.  The heavy, fork-specific engineering is
  the arena (the purity invariants above), and it should be presented as such rather than lumped in.
* The honest headline is *"this fixes a problem upstream has not hit yet"*: upstream's compute buffer is
  reserved once and grows at most once, and its expert path is stateless, so neither consumer ever parks
  the free VRAM.  The slab becomes necessary the moment an expert cache does.
