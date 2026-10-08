# `--fit` × slab × MoE arena — accounting campaign

**Status:** **PARKED (2026-10-07).**  Phase 1 (G1+G2 + the tensor-split plumbing it needs) was
implemented, built warning-free and tested; it is not shipped and `patches/` is untouched.  G2 (slab
headroom) and G1 (explicit `MOE_EXPERT_CACHE_MIB`) are validated safe, but newly enabling the **auto
floor** under `-sm tensor` reproducibly corrupts.  Full record + the patch: [`PHASE1-ATTEMPT.md`](PHASE1-ATTEMPT.md).

**Update 2026-10-08 (field data, no code change).**  A third gap **G3** was identified from the
discussion #108 field report (briansp2020: partial offload silently disables the MoE expert cache,
decode 40-47 t/s) and confirmed with a 2x R9700 `-ncmoe` sweep.  G3 is the slab **reserve**
(`GGML_CUDA_SLAB_RESERVE_MIB`), one level below G1/G2's fit *margin*: the flat default is charged
against the post-weights free VRAM, so partial offload trips the slab's decline and drops the run to
host streaming.  The same sweep also reproduced the parked `////` corruption at the arena boundary
(2/2), so the corruption stays quarantined as its own item.  See **G3** in §0/§1.3/§4.5 and the full
procedure + numbers in **§10**.  G3 can land with the option-A safe subset (it never enables the auto
floor).

**Goal:** make `--fit` the **single VRAM planner** for the RDNA/ROCm expert-cache system: it must reserve
the MoE arena budget and the movable-boundary slab's headroom, instead of the arena being an emergent
"whatever is free afterwards" third consumer.
**Owner handover:** read §0 → §2 → §4.  §2 is the constraint that shapes everything.
**Related:** `archive/work/moe-cache-autosize/` (the slab + auto-sizing campaign, CLOSED),
`wip/moe-cpu-overlap/` (TODO #36, a different miss-path question), TODO #47.

---

## 0. TL;DR

Three concrete gaps, all fixable without touching the cache engine:

| # | gap | fix |
|---|---|---|
| **G1** | An **explicit `MOE_EXPERT_CACHE_MIB` is invisible to `--fit`** (`common/fit.cpp:305` only reserves when the var is *unset*). `--fit` sizes the context as if the arena were not there. | add the explicit per-device budget to the fit margin |
| **G2** | The slab's **`GGML_CUDA_SLAB_HEADROOM_MIB` is a hard floor that `--fit` does not model**, and the `--fit` default target (1 GiB) is *smaller* than it (4096 MiB). A fully-fitted ROCm run can leave less free VRAM than hipBLASLt needs, and the next wide prefill aborts (`exit 134`). | add a slab-headroom floor to the fit margin, queried from the device (single source of truth) |
| **G3** | The slab's **`GGML_CUDA_SLAB_RESERVE_MIB`** is a flat `max(8192, 25 %)` that is never planned, and it is subtracted from `free_b` at slab creation. With partial offload `free_b - 8192 < work + 2048`, the slab **declines** (`cache_unusable`) and the whole run silently streams from the host. Too small is the exit-134 abort, too large is this decline, so it must be planned, not guessed. | compute the reserve from the actual post-slab need (KV + draft + workspace) and hand it to the slab (Phase 2's `slab_reserve_bytes` gains a setter).  **Not** "lower the default" -- see §10.4. |

**Ship Phase 1 = G1 + G2.** G3 was added 2026-10-08; it is independent of the auto floor and can land
with the option-A safe subset, or roll into Phase 2 if it grows.  Phase 2 (single-source-of-truth
getters/setters, de-duplicate the floor policy) and Phase 3 (arena-first budgeting / 2-pass auto) are
scoped in §7 but not required for the release.

---

## 1. The gaps, in code

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

### G3 — the slab *reserve* is a flat guess, and it is what declines under partial offload

G1/G2 are about the fit **margin**.  G3 is one level down: the amount the slab keeps **outside** itself
for the allocations it cannot serve (`ggml_cuda_slab_reserve_bytes`, `ggml-cuda.cu:598`):

```c
reserve_mib = max(8192, total_b / 4 / MiB);     // 8192 MiB on a 32 GiB card
...
const size_t want = (free_b - reserve) / chunk * chunk;   // what is left for (work + arena)
if (want < up(work_min, chunk) + ggml_cuda_slab_min_arena_bytes()) {   // floor 2048 MiB
    s.cache_unusable = true;                    // -> moe_cache_disable_streaming()
    return false;                               // the cache streams from the host for the whole run
}
```

`free_b` is measured at slab creation, **already net of the GPU-resident weights / `-ot` overrides**.  So
the same flat 8 GiB is charged to every config: an all-VRAM run barely notices, a heavily offloaded run
loses the cache even though it is the config that needs it.  The decline is not the model failing to
fit, it is the reserve not fitting:

| config (2x R9700, IQ3_XXS, `-sm tensor -c 32768`) | `free_b` | `want` | `work + 2048` | outcome | decode |
|---|---:|---:|---:|---|---:|
| `-ncmoe 48` | large | large | — | slab + 37 GiB arena | 105.9 t/s (warm) |
| `-ncmoe 18` | 16.4 GiB | 8.4 GiB | ~3.5 GiB | fits | 65.9 t/s |
| `-ncmoe 12` | 11.05 GiB | 2.82 GiB | 3.28 GiB | **declines** -> streams | 60.5 t/s |
| `-ncmoe 12` + `GGML_CUDA_SLAB_RESERVE_MIB=4608` | 11.05 GiB | 6.4 GiB | 3.28 GiB | fits, but **corrupts 2/2** (§10.4) | 30.4 t/s |
| `-ncmoe 8` | ~7.0 GiB | free ≤ reserve | — | **no slab** (early return) -> arena 0 | 56.1 t/s |
| `-ncmoe 8` + `...RESERVE_MIB=4608` | ~7.0 GiB | ~2.5 GiB | 3.28 GiB | fits (6168 MiB arena) | **86.0 t/s** |

This is **Finding A** of discussion #108: "with partial offload the default `GGML_CUDA_SLAB_RESERVE_MIB`
(8192) leaves too little for the slab ... decode drops to ~40-47 t/s, against ~88 on r20", worked around
with `=4608`.  The reporter's `-ncmoe 18` log (`free 11072`, `want 2880`, floor 2048) is the same
arithmetic, and our `-ncmoe 12` reproduced the decline message verbatim.  §2 explains why the reserve
cannot simply be lowered: a too-small reserve is the `hipBLASLt` exit-134 abort, and §10.4 shows the
boundary between "decline" and "fits" also corrupts.

**The fix is to plan the reserve, not shrink it.**  The fit is the only place that knows the post-slab
need (KV + draft + workspaces), so G3 = the fit computes the reserve and the slab consumes it.
`slab_reserve_bytes` (Phase 2, §7) becomes a **settable** value rather than a flat default.  G3 is
independent of the **auto floor** and therefore does not touch the parked corruption.

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

### 2.1 Can hipBLASLt be routed through the slab? (investigated 2026-10-07)

**Verdict: the code-object class cannot be; the workspace class can, but only by leaving rocBLAS.**

Call path in this build: `ggml_cuda_mul_mat_cublas_impl` (`ggml-cuda.cu:2799`) -> `hipblasSgemm` /
`hipblasGemmEx` / `hipblasGemmStridedBatchedEx` (the `cublas*` aliases in
`ggml/src/ggml-cuda/vendors/hip.h`) -> `librocblas` -> internally (`rocblas_gemm_hipblaslt_backend` /
`hipblaslt_host.cpp`, symbols confirmed inside `librocblas.so`) -> `libhipblaslt`.  So the allocations
happen two library levels below ggml, and neither library exposes an allocator hook to us.

| class | how allocated | routable through the slab? |
|---|---|---|
| **Tensile/RocRoller code objects** (the `hipModuleLoad failed` in the 2048 abort) | loaded by the runtime inside hipBLASLt | **No.**  `hipModuleLoad` owns the backing memory; no public API accepts a caller buffer. |
| **Matmul workspace** | `hipblasLtMatmul(..., void * workspace, size_t workspaceSizeInBytes, ...)` — caller-provided | **Only if we call hipBLASLt ourselves.**  rocBLAS allocates it internally and ROCm 7.14 has no `rocblas_set_workspace` / user-driven-memory API (grep of `rocblas.h`: absent). |

Consequences:

1. **The observed abort at headroom 2048 was the code-object class** (`hipModuleLoad failed`), so
   routing the workspace would **not** have fixed it.  The G2 headroom floor stays necessary
   regardless of any workspace routing.
2. Routing the workspace means replacing the hipBLAS GEMM path with direct hipBLASLt calls
   (`hipblasLtMatmulAlgoGetHeuristic` + `hipblasLtMatrixLayout` descriptors + the
   `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` / `hipblaslt_ext::GemmPreference::setMaxWorkspaceBytes`
   preference) passing a slab-backed workspace.  That is a `vendors/hip` partial rewrite (algo
   selection, transposes, batched/strided-batched, compute types, epilogue) and it removes only the
   *workspace* term.  Scope it as its own item; do **not** block G1/G2 on it.
3. **`HIPBLASLT_PRELOAD_KERNELS` is a NO-OP for this workload (measured 2026-10-07).**  With
   `HIPBLASLT_PRELOAD_KERNELS=1`, hipBLASLt still `initialize`s at the first wide GEMM (the request), not
   at model init: the hipBLASLt log is the identical 5933 lines with the same timestamps, post-init idle
   VRAM is identical (28656 MiB/device), the request peak is identical (32460 vs 32278 MiB), and prefill
   is within noise (858 vs 862 t/s).  So it does **not** move the code objects to init and cannot make the
   allocation pre-arena.  (`HIPBLASLT_TENSILE_LIBPATH` / `HIPBLASLT_USE_ROCROLLER` /
   `HIPBLASLT_ROCROLLER_NO_CUSTOM_KERNEL` select *which* kernels, not *when* they load.)

### 2.2 Measured: a thin headroom corrupts on the maintainer's config (2026-10-07)

Reproduced at `GGML_CUDA_SLAB_HEADROOM_MIB=2048` on the exact server config (`--fit off`, `-sm tensor
-ncmoe 48`, `-ub 6144 -c 204800`, IQ4_XS, 6388-token wide prefill):

| headroom | output | prefill | notes |
|---|---|---:|---|
| 4096 (base) | coherent ("# A scheduling puzzle ...") | 861 / 863 t/s | idle 28656 MiB, peak 32278 |
| 4096 + `HIPBLASLT_PRELOAD_KERNELS=1` | coherent | 858 t/s | identical to base |
| 2048 | **`!!!!!!!!`** (corrupt) | **475 t/s** | `moe_cache_evict_slab_range` + `moe_cache_rearm` mid-request, a stale MoE-cache alias, VRAM collapsed 32.5 -> 13.0 GiB |

The failure mode here is **silent corruption plus thrash**, not the `hipModuleLoad` abort the older
`-ub 4096 -c 163860` record saw — i.e. a thin headroom is not even reliably loud.  That is the strongest
argument for G2: the fit must guarantee the headroom, not hope the runtime degrades safely.
4. `HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` (default `0` = no workspace) is the cap, but it is only
   settable through the hipBLASLt API — i.e. again only from a direct-call path.

**Net:** G1+G2 are the fix; the slab-routing idea is a separate, larger optimisation whose ceiling is a
reduced *workspace* floor, never a zero floor.

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

## 4. Phase 1 (the release): G1 + G2 (+ G3)

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

### 4.5 G3 — reserve planning (scope addition, 2026-10-08)

The §4.1 rule reserves the headroom + arena in the **fit margin**, but the slab still splits `free_b`
with its own flat `GGML_CUDA_SLAB_RESERVE_MIB`.  G3 closes that gap:

* Add a **setter** alongside the getter: `slab_set_reserve_bytes(dev, n)` (or pass the reserve into the
  slab at creation).  The CUDA arm applies it to the slab's reserve; CPU/Meta/RPC are inert.  This is
  the same iface-append pattern as G2's `slab_headroom_bytes`.
* The fit computes `reserve = max(headroom, planned_post_slab_bytes)` and reserves
  `max(target, headroom) + reserve + arena`, so the slab it is about to create can always fit
  `work + min_arena`.
* Keep the flat default as the fallback when `--fit` is off or the fit aborts (`-ngl` set by the user),
  so non-fitted runs are unchanged.
* **Do not** implement G3 as "lower the default": §10.4 shows that reaching into the declined band with
  `GGML_CUDA_SLAB_RESERVE_MIB=4608` corrupts at the boundary.  G3 must move the reserve and the headroom
  together (G2 is what makes a smaller reserve safe).
* Sequencing: G3 develops alongside G1+G2 (option A) because it never enables the auto floor under
  `-sm tensor`.  If it turns out to need a declared runtime arena budget, it merges with Phase 3 (§7).

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
6. **G3 transitions (added 2026-10-08).**  For the `-ncmoe` values where the slab flips from decline to
   fits (here ~12-16), assert coherent text (`////` = 0), MTP acceptance, and width purity.  §10.4 shows
   the boundary itself corrupts, so a reserve change that "unlocks" a declined band must be gated on
   coherence, not throughput.

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

### Separately — hipBLASLt workspace routing (§2.1)

Out of scope for G1+G2 and not a substitute for the headroom floor: only hipBLASLt's *matmul workspace*
is routable, and only via a `vendors/hip` rewrite to call hipBLASLt directly; the *code objects* that
caused the 2048 abort are not routable.  Keep it as its own candidate with its own gates.

### Separately — force hipBLASLt's init to model-load time

Since the code objects load lazily at the first GEMM (§2.1/§2.2), a **warm-up GEMM at init** (before the
slab/arena is sized) would make the footprint measurable and pre-arena, removing the guess from the G2
headroom.  That is a change in the ggml-cuda reserve path, not an env knob — scope it only if the guessed
headroom proves insufficient in the field.

---

## 8. Open questions for the maintainer

1. **G2 gate:** b1 (host experts only) or b2 (every slab device)?  Recommend b2 after an A/B.
2. **Auto semantics:** should `MIN_RES_PCT` become a *target* (reserve and size to it) rather than just a
   floor?  That would make the arena predictable at the cost of context size.
3. **`--fit-target` default (1 GiB) vs slab headroom (4 GiB):** this looks like a bug independent of the
   cache (any near-full ROCm `--fit` run).  Fold the fix into G2, or file it separately?
4. **G3 reserve source:** should the fit publish an absolute per-device reserve (getter/setter), or
   should the slab derive it from a fit-declared post-slab need and keep `max(8192, 25 %)` only as the
   non-fitted fallback?  The former is smaller; the latter keeps one number.

---

## 9. References

* `ENVIRONMENT.md` §1.3 (the slab), §2 (scheduler/staging) — the variable table and the hipBLASLt warning.
* `archive/work/moe-cache-autosize/OPEN2-VMM-HANDOVER.md` — the thin-headroom experiment and the
  hipBLASLt evidence; `ARENA-UB-TENSION.md` §13/§14; `OPEN1-FINDINGS.md` §3 (the drop extra reserve).
* `archive/work/moe-expert-cache/README.md` §3 — arena vs `-ncmoe` (the two axes) and the unification goal.
* `TODO.md` #42 (closed slab work, residual items), #47 (this), #36 (`moe-cpu-overlap`).
* `AGENTS.md` — WIP/promotion rules (this directory is **not** delivery), default-on policy, scope policy.
* Discussion #108 comment 18801936 (briansp2020, 2026-10-07) — the partial-offload field report that G3
  (and §10) resolves; his N=18/N=14 `GGML_CUDA_SLAB_RESERVE_MIB=4608` workaround.

---

## 10. Field data (2026-10-08): the `-ncmoe` sweep, the G3 cliff, and the boundary corruption

Recorded for the revival; no code changed.  Source: discussion #108 comment 18801936 plus a
reproduction on the maintainer's box.  Transient artifacts: `/tmp/ub2048/sweep.sh`, `sweepw.sh`, logs
`sw-*` / `sww-*`.

### 10.1 Setup / procedure

* **Box:** gfx1201, 2x R9700 (32 GiB), ROCm 7.14.1, 184 GiB host, 16 cores.
* **Model:** `Qwen3.8-Flash-Next UD-IQ3_XXS` (3 shard) + `mtp-...-shared-Q8_0.gguf`.
* **Flags:** `HIP_VISIBLE_DEVICES=0,1`, `-sm tensor -ncmoe N -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -t 8`,
  `-c 32768 -b 2048 -ub 2048`, `--spec-type draft-mtp --spec-draft-n-max 3 --reasoning off`, greedy.
  `--fit` default (it aborts early on a user-set `-ngl`, so it does not resize here).
* **PLE placement:** host-resident (`CPU_Mapped` ~= 27.5 GiB) — **not** the reporter's regime (his PLE is
  in VRAM), so his fit floor and degenerate threshold sit higher.  The shape transfers; the absolute
  `-ncmoe` where the cliff lands does not.
* **Prefill item:** the repo's `prompts/prose-rdna-boosts.txt` (16,074 B) concatenated 4x = **64,296 B =
  20,984 tokens** (the server's count) — about a third of the reporter's ~61K prompt.
* **Procedure (matters):** a **warm pass** (the 20,984-token prefill + 768 decode) then the **measured**
  pass (same prefill + 256 decode).  A cold-arena measurement badly understates the cache configs:
  `-ncmoe 48` reads **43.5 t/s cold vs 105.9 t/s warm** (arena hit 0.82 vs 0.998).  Always warm first.
* **Sweep:** `sweepw.sh <tag> 48,40,32,24,16,8,0`; the cold run is `sweep.sh <tag> <list>`.

### 10.2 Warm-arena results (the comparison that matters)

| `-ncmoe` | arena MiB | prefill t/s | decode t/s | arena hit |
|---:|---:|---:|---:|---:|
| 48 | 37181 | 832 | 105.9 | 0.998 |
| 40 | 29610 | 929 | 106.3 | 0.995 |
| 32 | 22051 | 1048 | 104.2 | 0.984 |
| 24 | 14342 | 1210 | 95.9 | 0.940 |
| 16 | 6768 | 1426 | 80.1 | 0.785 |
| 8 | **0** | 1716 | 56.1 | — |
| 0 | — (no host experts) | 2245 | 119.7 | — |

* **Prefill is monotonic** in the host-expert bytes (832 -> 2245 t/s, `-ncmoe 48 -> 0`): every ubatch
  stages the host shards, so fewer host experts is always faster.
* **Decode is a plateau** at high `-ncmoe` while the arena holds ~99 % of the host set (48/40/32 within
  noise); it falls only once the arena shrinks (24 -> 95.9, 16 -> 80.1, 8 -> 56.1 with a 0 MiB arena).
* **`-ncmoe 48` is not an optimum.**  `40` and `32` strictly dominate it (decode tied, prefill +12 % /
  +26 %); below ~24 you trade decode for prefill.  `-ncmoe 0` is best of all but only fits here because
  the PLE is host-resident.
* Two practical takeaways for the delivery docs: don't tune `-ncmoe` *down* for decode (the warm arena
  is already at the plateau), but `40`/`32` are free prefill wins; and every "the cache is slow"
  measurement must state its arena hit/warmth or it is not comparable.

### 10.3 Cold-arena results (for contrast — do not quote as steady state)

| `-ncmoe` | 48 | 44 | 40 | 36 | 32 | 28 | 24 | 20 | 16 | 12 | 8 | 4 | 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| arena MiB | 42027 | 38201 | 34293 | 30500 | 26740 | 22932 | 19137 | 15356 | 11574 | **0** | **0** | **0** | — |
| prefill | 777 | 815 | 862 | 909 | 970 | 1029 | 1106 | 1184 | 1287 | 1357 | 1535 | 1713 | 1983 |
| decode | 43.5 | 45.2 | 47.3 | 49.6 | 52.5 | 56.1 | 60.6 | 65.4 | 71.1 | 60.5 | 56.5 | 71.4 | 112.2 |

The decode column is warm-up-limited (hit ~0.82); the 12/8 dip is the G3 decline below.

### 10.4 The G3 cliff is also a corruption cliff

* `-ncmoe 12`, default reserve: slab **declines** (`only 2816 MiB is available after the 8192 MiB
  reserve`, `work 1280 + floor 2048`), cache disabled -> streaming.  Decode 60.5 t/s, output
  **correct**.
* `-ncmoe 8` + `GGML_CUDA_SLAB_RESERVE_MIB=4608`: slab fits (6168 MiB arena, 78.3 %), coherent, decode
  **56.5 -> 86.0** — the decline really was G3.
* `-ncmoe 12` + `GGML_CUDA_SLAB_RESERVE_MIB=4608`: slab fits (9962 MiB, 85 %), but **`////` output and
  draft acceptance 0.0000 (0/471), reproduced 2/2**, with `alias_find_checked: ignoring a stale MoE-cache
  alias (table device 0 layer 4, op device 0 layer 3)` — the same signature as
  [`PHASE1-ATTEMPT.md`](PHASE1-ATTEMPT.md) §2.  So the declined band cannot be unlocked by lowering the
  reserve alone.

**Consequences for the revival:** (1) G3 must plan the reserve *with* G2's headroom rather than shrink
it; (2) the boundary corruption stays quarantined as its own root-cause item and must not be "fixed"
by a reserve default change; (3) any G3 change needs the §5 no-abort matrix plus a coherence / width-
purity gate at the specific `-ncmoe` values where the slab transitions from decline to fits.

