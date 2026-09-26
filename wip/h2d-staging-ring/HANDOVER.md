# HANDOVER — issue #50: merge the reporter's ring (PR #51) with our `h2d-stage.patch`

Written 2026-09-26, end of the session that built the prototype and the link calibration.  Read this
first; `README.md` in this directory is the full evidence record and is not required to start.

## 1. Objective

Produce **one** shippable op-offload H2D staging patch that takes the best of both:

* **PR #51** — the reporter's CUDA-backend ring (`wip/h2d-staging-ring/reporter-patch/` once the PR is
  merged, else GitHub PR
  [#51](https://github.com/stew675/llama-cpp-rdna-boosts/pull/51), branch from `briansp2020`); and
* **`h2d-stage.patch`** — our scheduler-level ring prototype (this directory).

Then run the validation gates and decide the delivery home (block 15 has the staging-arena precedent;
block 11 is the CUDA prefill-graph skip).

## 2. Current state

* Our prototype: **`h2d-stage.patch`** (481 lines, applies on top of `scripts/apply-all.sh` at r10).
  Byte-identical output, env-gated by `GGML_SCHED_STAGE=1`, bounded arena, link-calibrated gate.
  Validated on soar (gfx1201) and fingon (gfx1100).  Record: `README.md` §5b.
* The reporter's patch is **reviewed and commented** in PR #51 (review posted 2026-09-26).  It is not
  merged; the files are `wip/h2d-staging-ring/reporter-patch/0001-*.patch` + `README.md` on that branch.
* The in-tree `~/llama.cpp` (soar) working tree carries **r10 + `h2d-stage.patch`** (uncommitted);
  `git diff -- ggml/src/ggml-backend.cpp ggml/src/ggml-backend-impl.h ggml/src/ggml-backend-meta.cpp
  ggml/src/ggml-cuda/common.cuh ggml/src/ggml-cuda/ggml-cuda.cu` regenerates our patch.

## 3. Design comparison and cherry-pick matrix

| aspect | PR #51 (theirs) | `h2d-stage.patch` (ours) | take |
|---|---|---|---|
| where the ring lives | CUDA backend (`set_tensor_async`) | scheduler + 5 backend hooks | **theirs** for the redirect, **ours** for the hooks/bounding |
| how the op reads staged data | `tensor->data` redirected to the slot — **one** H2D move | stage slot, then **D2D** into the split input on the main stream — two moves | theirs (see §6 risk) |
| arena bound / OOM | unbounded, `cudaMalloc` via `CUDA_CHECK` → abort | `GGML_SCHED_STAGE_MAX_MB` (default 1024), null + fall back | **ours** |
| gate | fixed `COPY_ALL_MIN_TOKENS=1024` | calibrated from measured H2D: `1536 − 132·(bw − 14.5)` | **ours** |
| per-split `ggml_backend_synchronize` | fixed by `GGML_SCHED_EVENTS=1` | only avoided on the all-staged path | **theirs** (`EVENTS`) |
| CUDA/HIP graphs | disabled when a staged upload is consumed | not checked — verify | **theirs** |
| slot free timing | `graph_compute` frees **all** pending slots per split (blocks prefetch-ahead) | frees a slot right after its D2D (allows prefetch) | reconcile — free per-slot after the consumer |
| whole-tensor requirement | yes (needs `COPY_ALL`) | yes (structural: ids aren't known before the router) | both — keep the calibrated gate |

**Target shape:** their `tensor->data` redirect + graph skip, their `GGML_SCHED_EVENTS`, our bounded
arena and link-calibrated gate, and a per-slot free that does not block a deeper prefetch.

## 4. Build environment — READ FIRST (this cost most of the session)

* **HIP `.cu` TUs emit no `.d` dependency files**, and ccache's direct mode then does not invalidate on
  `common.cuh` changes.  Symptom: a header change appears to build but the object is **stale** (our
  calibration silently read 0 for several iterations).  Workaround used all session:
  `rm -f build-rocm/.../ggml-cuda.cu.o && cmake --build build-rocm --target <target> -j16`.
  Proper fix for the delivery: add `-MD`/`.d` generation or `CCACHE_DEPEND=1` to the build.  Also, when
  a `.cu` *is* touched, the whole FA instance group recompiles — batch `.cuh`/`.cu` edits.
* **`cudaEventCreate` and `cudaEventElapsedTime` are not aliased** on this ROCm 7.14 toolchain (only
  `cudaEventCreateWithFlags`, `cudaEventRecord`, `cudaStreamWaitEvent`).  Use `…CreateWithFlags`, or
  time a synchronous copy with `ggml_time_us()` (what the calibration does).
* **`__CUDA_ARCH__` cannot be used to split the HIP host/device passes** in `common.cuh` here — keep
  host-only code out of `common.cuh` or it breaks the device pass.  The calibration lives in
  `ggml-cuda.cu` as a plain host static for that reason.
* Build: `BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714` on each host (soar `gfx1201`,
  fingon `gfx1100`); runtime `LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib` (soar) /
  `/opt/rocm-7.14-gfx1100/lib` (fingon).

## 5. Repos and hosts

* Delivery repo (this one) on `main`; latest commits: `5684022` (link calibration),
  `c4c2403` (bounded arena + gate).  `wip/h2d-staging-ring/` holds the patch, the probe and the record.
* **soar** (local): `~/llama.cpp` on `rdna-boosts` + working-tree prototype; 3× R9700, PCIe5 **x4**
  (BIOS), measured H2D 14.45 GB/s.
* **fingon** (ssh, no GitHub SSH key — fetch over HTTPS; agent forwarding does not work): r10 applied in
  `~/llama-r10` (worktree, branch `rdna-r10`, tree `b59faadd…`), built in `~/llama-r10/build-rocm`;
  1× RX 7900 XTX, PCIe4 **x16**, measured H2D 28.6 GB/s.  Patch transfer:
  `cd ~/llama.cpp && git diff -- <5 files> > /tmp/h2d-stage.patch && scp /tmp/h2d-stage.patch fingon:/tmp/`
  then on fingon `git -C ~/llama-r10 checkout -- <5 files> && git -C ~/llama-r10 apply /tmp/h2d-stage.patch`.
* `halo` (gfx1151) is **out of scope** — unified memory makes `-ncmoe` pointless there.

## 6. The one real risk to resolve

The reporter redirects `tensor->data` to the slot so the consuming op reads it directly, restoring it
via `h2d_orig` + `h2d_slot_ptrs` + a `GGML_ASSERT` before any non-staged upload.  That breaks the
elsewhere-safe assumption that a split input's `data` is its allocation.  Before adopting it, either
(i) prove no other reader observes it (views, non-staged copies, other backends, allocator bookkeeping,
`--n-cpu-moe` mixed paths), or (ii) keep our D2D (slower by a few percent, keeps the invariant).  Do
**not** merge the redirect without one of those.

## 7. Validation done / to do

Done (ours): byte-identical same-seed text on soar + fingon (`sha=6541eadb9041`); budget and gate
fallbacks exact; PP sweep `ub 512…8192` on both hosts; link calibration on both hosts.

To do before either lands:
1. Re-run the above on the **merged** patch, both hosts.
2. `test-backend-ops` (it should be untouched, but confirm), the `W=1..8` width probe
   (`GREEDY-PURITY.md` §§10-11), MTP (`draft-mtp n3` + adaptive, 27B `draft-mtp` acceptance/text), and a
   deep-context prefill + server soak (the reporter's 81/81 shape).
3. `-sm tensor`: the meta backend's stage hooks are `nullptr`, so staging is inert under tensor split.
   Forward them (or gate staging off there).  The reporter's patch is direct-backend only too.
4. Graph capture: confirm staged prefill skips capture (theirs does; ours needs checking).
5. `scripts/validate-set.sh` + patch/`release.json` regen if this becomes a delivery block.

## 8. A/B plan (agreed with the reporter)

* Run **their** patch on soar (x4) and fingon (x16), same model, `ub 2048/4096/8192`, knobs
  `SLOTS=3 COPY_ALL=1024 EVENTS=1`, versus r10 and versus our `GGML_SCHED_STAGE=1`.
* They will run **our** `h2d-stage.patch` on their PCIe5 **x16** box (53 GB/s DMA) — the one link
  neither of us has; it separates the D2D cost from the calibration.
* Post the combined table on PR #51.

## 9. Known numbers (for the record)

`llama-bench -ncmoe 99 -fa 1 -p 8192 -n 1 -b 8192`, host-resident experts, ring off → on:

| box | link | ub | off | on (ours) | `-ncmoe 0` |
|---|---|---|---|---|---|
| soar | PCIe5 x4 | 1024 | 783 | 744 (ours gates it to 779) | 6502 |
| soar | x4 | 2048 | 1412 | 1479 | 6502 |
| soar | x4 | 4096 | 2249 | 2933 | 6502 |
| soar | x4 | 8192 | 3162 | 5490 | 6502 |
| fingon | PCIe4 x16 | 1024 | 1589 | 1958 | 6036 |
| fingon | x16 | 2048 | 2507 | 3858 | 6036 |
| fingon | x16 | 8192 | 4100 | 5597 | 6036 |

(The reporter's PCIe5 x16 box: +74 % at ub 2048 / +57 % at ub 4096 on the 35B Q4_K_M — see PR #51.)
Latent prior art, decode-only and complementary (not part of this task): the expert cache measured in
issue #47's comment (per-layer expert slots, ~3 MiB/slot, 72–78 % hit rate, decode +36 %, prefill −17 %
that this ring recovers).

## 10. First commands for the next session

```bash
cd ~/llama-cpp-rdna-boosts && git pull            # main has the patch + records
gh pr view 51 && gh pr diff 51 > /tmp/pr51.diff   # the reporter's patch
cd ~/llama.cpp && git status --short              # r10 + h2d-stage.patch in the tree
# build after any .cuh/.cu edit (see §4):
rm -f build-rocm/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/ggml-cuda.cu.o
cmake --build build-rocm --target llama-bench llama-server -j 16
```
