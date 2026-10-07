# Phase 1 implementation attempt — PARKED (2026-10-07)

**Status: PARKED.**  The code was implemented, built clean, and tested.  It is **not** shipped and
`patches/` is untouched.  The blocker is a repro corruption described below; the decision was to document
and park rather than keep pushing, because the interaction is fragile.

**Artifact:** [`phase1-fit-slab-accounting-WIP.patch`](phase1-fit-slab-accounting-WIP.patch)
(19,766 B, 10 files; sha256 `71480cdfa9728d3ca31641ff42167631f4bb3d9ad86f5f2fb177d42ef4c8d094`).
It applies to the r26 block-15 tree (`release.json` base `a55e952b8`, tip tree
`6c7dc021cd03cd5e367af79c29a5cba88092bab5`) with `git apply` in `~/llama.cpp`.

---

## 1. What the patch does

It is the full Phase 1 (G1 + G2) **plus** the tensor-split plumbing that G1/G2 need (see the parent
`README.md` §4 and §7 of this directory):

* **G2 — slab headroom in `--fit`.**  New device-iface `slab_headroom_bytes` (decl + wrapper + CUDA getter
  + `nullptr` in CPU/Meta/RPC), returning `ggml_cuda_slab_enabled() ? ggml_cuda_slab_headroom_bytes() : 0`.
  `--fit` reserves it as a hard floor: `max(base_margin, headroom) + arena`.
* **G1 — explicit `MOE_EXPERT_CACHE_MIB` reserved verbatim**, instead of ignored.
* **Tensor-split plumbing** (the pre-existing bug): the fit's cache reservation was keyed on
  `model->devices`, which under `-sm tensor` is the Meta wrapper, so `host_expert_bytes` was 0 and the
  reservation also wrote to `margins` while the tensor branch consumes `margins_s`.  The patch adds
  `llama_model_moe_host_expert_dev_count/_dev` (a map-keyed real-device enumerator), captures the
  per-real-device bytes in the fit probe, and applies the reservation to **both** `margins`
  (layer/single-GPU) and `ttarget` (tensor split).

Files: `common/fit.cpp`, `ggml/include/ggml-backend.h`, `ggml/src/ggml-backend-impl.h`,
`ggml/src/ggml-backend.cpp`, `ggml/src/ggml-backend-meta.cpp`, `ggml/src/ggml-cpu/ggml-cpu.cpp`,
`ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-rpc/ggml-rpc.cpp`, `src/llama-ext.h`,
`src/llama-model.cpp`.

**Build:** clean and **warning-free** on gfx1201 / ROCm 7.14.1 (incremental `cmake --build build-rocm`,
`rc=0`, zero `warning:`/`error:` lines).  No runtime crash; the failure is silent corruption.

---

## 2. Why it is parked — the auto floor under `-sm tensor` corrupts

Test: the maintainer's server model/config — `Qwen3.8-Flash-Next UD-IQ4_XS`, `-sm tensor -ncmoe 48`,
`--fit on` with `-ngl` **unset** (so the fit actually runs), a 6388-token prompt, `--spec-type none`,
greedy, 2×R9700.

| arm | fit margin | output | prefill t/s | arena MiB |
|---|---|---:|---:|---:|
| cache **off** (`MOE_EXPERT_CACHE_MIB=0`) | — | coherent | 821 | — |
| auto, `MOE_EXPERT_CACHE_MIN_RES_PCT=0` (headroom only) | 4096/dev | coherent | 856 | 42128 |
| auto, **`MIN_RES_PCT=18` (default)** | ~9204/dev | **`!!!!!!!!`** | 578 | 37179 → 28713 |
| auto, **`MIN_RES_PCT=18`** (re-run, 15 s gap) | ~9204/dev | **`!!!!!!!!`** | 530 | 37179 → 28713 |
| explicit **`MOE_EXPERT_CACHE_MIB=8192`** | 12288/dev | coherent | 844 | 16297 |

* The corruption is **reproducible 2/2 even with the recorded 15 s rapid-restart gap** (the records flag
  `--fit on` as restart-sensitive; that is not what this is).
* It is **not reservation size**: the explicit 8192 arm reserves *more* (12288/dev) and is coherent; the
  18 % auto floor reserves ~9204/dev and corrupts.  It is the specific layout the auto floor's
  per-device margin makes the fit choose.
* The corrupt runs log `moe_cache_evict_slab_range` + `moe_cache_rearm` mid-request and an
  `alias_find_checked: ignoring a stale MoE-cache alias (table device 0 layer 9, op device 0 layer 6)`.
  **But the alias warning also fires in the coherent arms** and the code comment calls a stale
  cross-graph address "routine", so it is **not** established as the cause.
* `G2` (headroom) and `G1` (explicit MIB) are **validated safe** by the matrix.  Only the *auto floor*
  newly enabled under `-sm tensor` is unsafe.

**Key point:** this layout never occurred before the patch.  The auto floor was **dead** under
`-sm tensor` (the `host_expert_bytes == 0` + `margins`/`margins_s` mismatch documented in the parent
README §1), so the patch is what exposes a latent, layout-dependent corruption in the slab/cache
carve-out path.  Whether the latent bug is pre-existing is not yet established; what is established is
that enabling the auto floor under tensor split reaches it reliably.

---

## 3. Open question to resolve before revisiting

Where does the `-sm tensor` + auto-floor layout diverge into corruption?  Candidates, in order:

1. **The cache's table aliasing** (`alias_find_checked`, `moe-expert-cache.cu`): tables are keyed by
   scheduler tensor address with a `(key, device, layer)` guard.  A different split/`ngl` changes the
   allocator's address reuse across graphs.  The guard refuses the alias, but that only means "not
   cache-managed" — trace whether the op then takes the host path or a stale one.
2. **The slab boundary move** (`moe_cache_evict_slab_range` → `moe_cache_rearm`) under the specific
   arena size the auto floor yields.
3. **`n_gpu_layers` chosen differently** by the fit (LOG_TRC-only output): the resulting layer placement
   may put an expert table on a device the cache does not expect.

A good first step is to log the fit's chosen `n_gpu_layers`/tensor split at LOG_WRN for the two arms
(auto vs explicit) and diff them, then bisect on that.

---

## 4. Options if it is picked up again

* **A — ship the safe subset.**  Keep **G2 (headroom)** + **G1 (explicit MIB)** + the tensor-split
  plumbing, but do **not** newly enable the auto floor under `-sm tensor` (preserve the pre-change
  behaviour there; it stays active on `-sm layer`/single-GPU).  Delivers the stated goal with no new
  corruption, and quarantines the latent bug.
* **B — root-cause first.**  Treat the corruption as its own campaign, fix it, then ship the auto floor
  too.
* **C — fix the alias fallback.**  Make a refused alias force an explicit host-path fallback (it should
  already) and re-test.  Speculative.

Recommendation on file was **A**; the maintainer chose to **park** the whole item instead.

---

## 5. Re-applying / reverting

```sh
cd ~/llama.cpp                       # must be the block-15 tree (tree 6c7dc021...)
git apply /path/to/wip/fit-slab-accounting/phase1-fit-slab-accounting-WIP.patch
BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
# ... or the fast loop:
cmake --build build-rocm --target llama-server llama-cli -j 16
```

To revert: `git checkout -- .` in `~/llama.cpp` (the patch is uncommitted work only).

The `~/llama.cpp` checkout was restored to the unmodified r26 block-15 tree and the
`backup-pre-fit-slab` branch was deleted after this attempt; `build-rocm` was rebuilt to match.

---

## 6. Record

* Build: `cmake --build build-rocm` `rc=0`, warning-free (2026-10-07).
* Server A/B logs: `/tmp/bench/hblt-fit-{auto2,auto3,nocache,headroom,mib}.server.log` (transient).
* Decision: **documented and parked**, `patches/` untouched, TODO #47 updated.
