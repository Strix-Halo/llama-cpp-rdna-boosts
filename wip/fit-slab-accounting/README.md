# `--fit` × slab × MoE arena — accounting campaign

**Status:** OPEN / scoping done, Phase 1 (G1+G2) approved to aim for a release.
**Goal:** make `--fit` the **single VRAM planner** for the RDNA/ROCm expert-cache system: it must reserve
the MoE arena budget and the movable-boundary slab's headroom, instead of the arena being an emergent
"whatever is free afterwards" third consumer.
**Owner handover:** read §0 → §2 → §4.  §2 is the constraint that shapes everything.
**Related:** `archive/work/moe-cache-autosize/` (the slab + auto-sizing campaign, CLOSED),
`wip/moe-cpu-overlap/` (TODO #36, a different miss-path question), TODO #47.

---

## 0. TL;DR

Two concrete gaps, both fixable without touching the cache engine:

| # | gap | fix |
|---|---|---|
| **G1** | An **explicit `MOE_EXPERT_CACHE_MIB` is invisible to `--fit`** (`common/fit.cpp:305` only reserves when the var is *unset*). `--fit` sizes the context as if the arena were not there. | add the explicit per-device budget to the fit margin |
| **G2** | The slab's **`GGML_CUDA_SLAB_HEADROOM_MIB` is a hard floor that `--fit` does not model**, and the `--fit` default target (1 GiB) is *smaller* than it (4096 MiB). A fully-fitted ROCm run can leave less free VRAM than hipBLASLt needs, and the next wide prefill aborts (`exit 134`). | add a slab-headroom floor to the fit margin, queried from the device (single source of truth) |

**Ship Phase 1 = G1 + G2.** Phase 2 (single-source-of-truth getters, de-duplicate the floor policy) and
Phase 3 (arena-first budgeting / 2-pass auto) are scoped in §7 but not required for the release.

---

## 1. The two gaps, in code

### G1 — explicit `MOE_EXPERT_CACHE_MIB` is not reserved

`common/fit.cpp` (~line 300-333), "policy c":

```cpp
const char * mib_env    = getenv("MOE_EXPERT_CACHE_MIB");
const bool   cache_auto = (mib_env == nullptr || mib_env[0] == '\0');
...
if (cache_auto && host_total > 0) {       // <-- explicit MIB falls through, no reservation
    ... margins[id] += floor_b;           // floor = max(MIN_MIB, MIN_RES_PCT% * host_expert_bytes)
}
```

So `MOE_EXPERT_CACHE_MIB=16384 --fit on` gets **zero** reservation: `--fit` picks an `n_ctx` as if the
16 GiB did not exist, then the runtime arena claims it and the requested residency is either not honoured
or the slab declines.  The runtime side is also explicit about this: `moe_cache_preflight()`
(`moe-expert-cache.cu:2085`) returns immediately with "an explicit `MOE_EXPERT_CACHE_MIB` is used
verbatim; no early floor", so **nothing** reconciles the two.

### G2 — the slab headroom is not modelled

The slab is ONE mapping per device split by a movable boundary: work pool below, MoE arena above
(`ENVIRONMENT.md` §1.3).  Two things stay **outside** the slab:

* `GGML_CUDA_SLAB_RESERVE_MIB` = `max(8192, 25% of device)` — the weights / KV / draft / workspaces
  region.  Too small → *run ends* (`ENVIRONMENT.md`: 4096 → failed hipBLASLt workspace, exit 134).
* `GGML_CUDA_SLAB_HEADROOM_MIB` = `4096` — free VRAM kept outside after `slab_extend` reclaims the
  reserve the model did not need.  This is **not slack**; it is the runtime's allocation budget.

`--fit`'s per-device target free space is `fit_params_target[dev]` = **1024 MiB** by default
(`common/common.h:496`).  On a ROCm run that fits a context tightly, free-after-fit can be ~1 GiB, the
slab reclaims its unused reserve, and hipBLASLt then has < 4 GiB to work with on the next wide prefill —
the abort class the maintainer has been bitten by several times.

Note the headroom requirement **grows with prefill width/context** (measured: 2048 MiB was fine at
`-c 32768 -ub 8192`; the same 2048 **aborted** at `-c 163860 -ub 4096`, and 4096 is the verified floor
there).  A static `--fit` constant is therefore not enough — see §2.

---

## 2. The hipBLASLt constraint (READ THIS FIRST — it shapes both fixes)

**The ROCm BLAS stack allocates VRAM behind the application's back and that allocation CANNOT be routed
into the slab.**  This is the gotcha that has stung this project repeatedly, and it is the reason G2 is a
*hard floor*, not a heuristic:

* llama.cpp never calls `cublasLtMatmul`; the build routes GEMMs through hipBLASLt, which lazily loads
  **Tensile code objects** (`.co`) and allocates its **own workspace**, neither of which the slab can
  serve.  They need REAL free VRAM.
* Measured at `GGML_CUDA_SLAB_HEADROOM_MIB=2048` (`-ub 4096 -c 163860`):

  | config | outcome |
  |---|---|
  | headroom 2048, hipBLASLt on | `hipModuleLoad failed` for the Tensile `.co` files (6x) then `Hip error: 'out of memory' at hipblaslt.cpp:164` → **ABORT** |
  | headroom 2048, `ROCBLAS_USE_HIPBLASLT=0` | server survives, output **CORRUPT** (`////`) |
  | headroom 4096, `ROCBLAS_USE_HIPBLASLT=0` | no corruption, but a 16k request returned 1 token then EOS |
  | headroom 4096, hipBLASLt on (shipped) | coherent; arena 37861 MiB (66.7 %); 348 MiB free |

* **Disabling hipBLASLt is NOT a workaround** (corrupt at a thin headroom, stunted request at a generous
  one).  Do not recommend it, and do not build anything that depends on it.
* The steady-state free VRAM after extend is therefore **emergent** (~150-350 MiB on the maintainer's
  server): the 4096 MiB headroom is consumed by the MTP draft buffer (762 MiB), the FA-QSA workspace
  (762 MiB), hipBLASLt's Tensile objects + workspace, and the workspace pool high-water mark.

**Consequence for `--fit`:** the fit's `no_alloc` probe **cannot measure** the hipBLASLt allocation (it
runs no wide prefill, and the allocation is invisible to `llama_memory_breakdown`).  So the fit must
**reserve the configured headroom** rather than discover it — and that headroom must be the *same value
the slab will use*, which is why the getter (Phase 2 / G2) reads the device, never a duplicated default.

Detail: `archive/work/moe-cache-autosize/OPEN2-VMM-HANDOVER.md` §"But it does NOT make a thin headroom
safe", and `ENVIRONMENT.md` §1.3.

---

## 3. Current code map

| concern | location |
|---|---|
| fit's per-device margin + "policy c" cache floor | `common/fit.cpp:300-333` |
| fit's memory probe (`no_alloc`, returns `host_expert_bytes`) | `common/fit.cpp:30-160` |
| `fit_params_target` default (1 GiB) | `common/common.h:491-496` |
| fit entry point + `margins` plumbing | `common/fit.cpp:1133`, `common/common.cpp:1221-1250` |
| cache preflight (latch, floor, disable) | `ggml/src/ggml-cuda/moe-expert-cache.cu:2085` |
| preflight call site (after model load, before draft ctx) | `common/common.cpp:1344-1353` |
| slab reserve / headroom / min-arena constants | `ggml/src/ggml-cuda/ggml-cuda.cu:598`, `:633`, `:662` |
| slab work-size getter + iface registration | `ggml-cuda.cu:739`, `:9953`, `:9981` |
| existing device iface seam | `ggml/src/ggml-backend-impl.h:364-371`, wrapper `ggml-backend.cpp:300`, public decl `ggml/include/ggml-backend.h:68` |
| drop / rearm on the prefill→decode transition | `src/llama-context.cpp:1700-1790` |

The iface already carries `slab_work_size`; adding `slab_headroom_bytes` (and, for Phase 2,
`slab_reserve_bytes` / `slab_min_arena_bytes`) is the same pattern: append the field, add a public
wrapper, register in CUDA, and add a `nullptr` in the CPU / Meta / RPC initializers (the r26
warning-free-build rule).

---

## 4. Phase 1 (the release): G1 + G2

### 4.1 The rule

`--fit`'s margin is the free VRAM to leave after fitting.  Two needs must fit in that free space:
the hipBLASLt headroom **and** the arena budget.  The arena lives *inside* the slab, so
`arena = free_after_fit - reserve - headroom` — i.e. the headroom and the arena are both drawn from the
same free VRAM.  The correct reservation is therefore:

```
margin[id] = max( margin[id], slab_headroom_bytes(dev) ) + cache_reservation_bytes(dev)
```

* `cache_reservation_bytes` = explicit `MOE_EXPERT_CACHE_MIB` if set, else the existing floor
  (`max(MIN_MIB, MIN_RES_PCT% * host_expert_bytes)`); **0** when the cache is off (`MOE_EXPERT_CACHE_MIB=0`).
* `slab_headroom_bytes` = the slab's own value, queried from the device (0 on non-slab backends, so the
  change is inert on CPU/CUDA/MUSA/SYCL/Vulkan).

`max(..., headroom)` rather than `+= headroom` avoids double-counting when the user's `--fit-target`
already exceeds the headroom; the cache budget is additive because the arena is a distinct consumer.

### 4.2 Implementation sketch

**a. Device iface** (`ggml/src/ggml-backend-impl.h`, append after `slab_work_size`):

```c
// (optional) the VRAM this backend's allocator must keep free OUTSIDE the slab for allocations
// it cannot route (BLAS/Tensile code objects + workspace).  0 when there is no such consumer.
size_t (*slab_headroom_bytes)(ggml_backend_dev_t dev);
```

Public wrapper in `ggml-backend.cpp` (same shape as `ggml_backend_dev_slab_work_size`) and decl in
`ggml/include/ggml-backend.h`; register `ggml_backend_cuda_device_slab_headroom_bytes` in `ggml-cuda.cu`
returning `ggml_cuda_slab_enabled() ? ggml_cuda_slab_headroom_bytes() : 0`; add `nullptr` to the CPU,
Meta and RPC initializers.

**b. `common/fit.cpp`** — replace the `if (cache_auto && host_total > 0)` block with:

```cpp
if (nd > 0 && host_total > 0) {
    const char * mib_env  = getenv("MOE_EXPERT_CACHE_MIB");
    const bool   mib_set  = (mib_env != nullptr && mib_env[0] != '\0');
    const int64_t mib_val = mib_set ? atoll(mib_env) : 0;
    const bool   cache_on = !mib_set || mib_val > 0;

    for (size_t id = 0; id < nd; id++) {
        // G2: the slab headroom is a hard floor on every slab-capable device.
        int64_t floor_b = (int64_t) ggml_backend_dev_slab_headroom_bytes(devs[id]);
        if (floor_b > margins[id]) margins[id] = floor_b;

        // G1: reserve the arena -- explicit value verbatim, else the auto floor.
        if (cache_on && host_expert_bytes[id] > 0) {
            int64_t arena_b = mib_set
                ? mib_val * MiB
                : std::max<int64_t>(min_mib * MiB,
                                    (int64_t) host_expert_bytes[id] * min_res_pct / 100);
            if (arena_b > 0) margins[id] += arena_b;
        }
    }
    LOG_WRN("... reserving headroom + arena ...");
}
```

Keep the `min_mib` / `min_res_pct` reads, but read them **once** and note the duplication with
`moe_cache_preflight` as Phase 2 work.

**c. Decide the G2 gate.**  Two options, for the maintainer:

* **(b1) only when host experts exist** (`host_total > 0`) — conservative, tiny blast radius, fixes the
  offload configs the maintainer runs.
* **(b2) on every slab-capable device** — more correct (hipBLASLt aborts on dense large prefills too),
  but changes fitted contexts for existing non-MoE ROCm `--fit` users.  Recommend b2 *after* a measured
  A/B; ship b1 if the release window is tight.

### 4.3 Ordering / invariants to preserve

* **The fit's reservation must agree with `moe_cache_preflight`.**  The preflight latches *before* the MTP
  draft context exists (`common/common.cpp:1344-1353`); the fit runs earlier still.  If they disagree, the
  r24 bug class returns (a late `g_enabled=false` plans the draft and target with different kernels).
  Phase 1 reduces the disagreement (the explicit MIB is now reserved) but does not yet unify the policy.
* **Never leave less than `reserve + headroom` free.**  The slab's `reserve` too small is a *clean abort*
  (exit 134), not a degrade — treat it as a hard constraint.
* **The headroom value must be single-sourced.**  `fit.cpp` reads it from the device; it must never
  re-declare `4096`.  If a user raises `GGML_CUDA_SLAB_HEADROOM_MIB` for a huge `-ub`, both the slab and
  `--fit` must see it.
* The slab is the **lowest-priority** consumer (it yields via `moe_cache_release_arena` / `_shrink`).  The
  fit must not rely on that as the primary plan.

### 4.4 What Phase 1 deliberately does NOT change

* Auto mode still uses the **floor + remainder** policy (context-priority): `--fit` reserves the floor,
  the arena takes the rest.  Arena-first / target-residency is Phase 3.
* The runtime arena is still sized from `free - reserve - aux`.  Phase 1 only stops `--fit` from
  over-committing; it does not make the runtime size to a pre-declared budget.
* No change to the cache engine, the slab allocator, or the drop/rearm path.

---

## 5. Validation gates (Phase 1)

1. **No-abort matrix.** `--fit on` × `-ncmoe {0, 48}` × `MOE_EXPERT_CACHE_MIB {unset, 0, 4096, 16384}` ×
   `-sm {layer, tensor}` × MTP {on, off} × `-c {auto, 32768, 204800}`.  Assert exit 0 and no
   `hipModuleLoad failed` / `hipblaslt.cpp:164` in the log.  Include the wide-prefill case
   (`-ub 4096/8192` with a 16k+ prompt) that reproduces the G2 abort today.
2. **Reservation is honoured.** With an explicit `MOE_EXPERT_CACHE_MIB`, the built arena
   (`slot print_timing` / `moe_cache_report`) is ≥ the requested value, and the fitted `n_ctx` is
   monotonically non-increasing as the requested budget grows.
3. **Dense / non-MoE unaffected** (if b2): a dense ROCm `--fit` model still fits and runs; and on a
   non-slab backend (`GGML_CUDA_SLAB=0`, or a non-ROCm arm) the margin is unchanged.
4. **Standing gates** (unchanged): byte-identity to the `-ncmoe 0` oracle, width purity
   `none == n1 == n3 == n7`, MTP acceptance, deep coherence, `test-backend-ops -o MUL_MAT_ID`.
5. **Field A/B:** on the maintainer's server config (`-sm tensor -ncmoe 48`, `-ub 6144 -c 204800`, cache
   auto, `GGML_CUDA_ALLREDUCE=ce`), compare the fitted `n_ctx` and the resulting arena with/without the
   change; confirm the 30k-prefill-then-decode path still reaches 68-71 t/s and 0 aborts.

---

## 6. Risks

* **Context shrink.**  Reserving the headroom + an explicit arena can reduce the fitted `n_ctx`.  That is
  the intended trade (guaranteed residency) but must be surfaced in the `--fit` log, and the default
  (auto floor + headroom) must not shrink contexts on existing configs by much — measure.
* **Probe inaccuracy.**  The fit's `used` is a projection (`no_alloc`); the #42 record already found the
  reserve graph under-bounds a real first prefill by ~426 MiB.  The headroom floor is the safety margin;
  do not treat a passing fit as proof.
* **iface change.**  Appending a device-iface field touches every backend initializer; keep the build
  warning-free (the r26 rule) and ensure the CPU/Meta/RPC fields are `nullptr` so non-RDNA is inert.
* **Duplicated floor policy** (fit vs preflight) remains until Phase 2 — a latent inconsistency, not a
  regression.

---

## 7. Phase 2 / 3 (not required for the release)

**Phase 2 — single source of truth.**  Expose `slab_reserve_bytes`, `slab_min_arena_bytes` on the iface;
move the floor policy into one helper both `fit.cpp` and `moe_cache_preflight` call; delete the duplicated
`getenv` logic.  Makes future changes one-line.

**Phase 3 — arena-first budgeting (upstream's model).**  Let the user (or an auto policy) declare the
per-device arena budget *before* fitting; `--fit` reserves it and the runtime sizes to exactly that budget
instead of `free - reserve`.  Circularity for auto:
* 3a: arena-priority for an explicit MIB / target fraction; floor+remainder for auto.
* 3b: **two-pass fit** — fit the context, predict the arena, re-fit with it reserved, verify convergence.
  The probe is `no_alloc`, so an extra pass is comparatively cheap; oscillation is the risk to measure
  (same family as the `Meta()` 7-round-iteration finding).

This is the "graph-level arena redirect" end goal in `archive/work/moe-expert-cache/README.md` §3, and it
is what makes `--fit` a true single planner.

---

## 8. Open questions for the maintainer

1. **G2 gate:** b1 (host experts only) or b2 (every slab device)?  Recommend b2 after an A/B.
2. **Auto semantics:** should `MIN_RES_PCT` become a *target* (reserve and size to it) rather than just a
   floor?  That would make the arena predictable at the cost of context size.
3. **`--fit-target` default (1 GiB) vs slab headroom (4 GiB):** this looks like a bug independent of the
   cache (any near-full ROCm `--fit` run).  Fold the fix into G2, or file it separately?

---

## 9. References

* `ENVIRONMENT.md` §1.3 (the slab), §2 (scheduler/staging) — the variable table and the hipBLASLt warning.
* `archive/work/moe-cache-autosize/OPEN2-VMM-HANDOVER.md` — the thin-headroom experiment and the
  hipBLASLt evidence; `ARENA-UB-TENSION.md` §13/§14; `OPEN1-FINDINGS.md` §3 (the drop extra reserve).
* `archive/work/moe-expert-cache/README.md` §3 — arena vs `-ncmoe` (the two axes) and the unification goal.
* `TODO.md` #42 (closed slab work, residual items), #47 (this), #36 (`moe-cpu-overlap`).
* `AGENTS.md` — WIP/promotion rules (this directory is **not** delivery), default-on policy, scope policy.
