# `archive/work/issue-86` - run the internal (host-staged) AR on non-RDNA4 (issue #86)

**Status:** **PROMOTED in `v16-a55e952b8-r9` (block 12), default-ON.**  The arch gate is bypassed on
every HIP arch by default (`GGML_CUDA_AR_ALLOW_NON_RDNA4=0` opts out) and the first non-internal
`try_allreduce` failure is re-served through the internal pipeline.  The opt-in patch below is the
pre-promotion record; the delivery no longer needs it.  The reporter has not confirmed on their 2x
gfx1100 yet; promotion rests on the design, the pre-existing 2x-RDNA3-behind-x4 deployment record
and the gfx1201 no-op (dense 4B `-sm tensor` `1c5d32ac537d` with the default and `=0`).
**Issue:** https://github.com/stew675/llama-cpp-rdna-boosts/issues/86
**Patch (pre-promotion, opt-in variant):** [`ar-non-rdna4.patch`](ar-non-rdna4.patch)

## Symptom

`-sm tensor` on 2 GPUs hangs after:

```
W ggml_cuda_ar_pipeline_init: internal all-reduce is RDNA4-only (gfx1200/gfx1201);
  device 0 reports gfx1100 -- falling back to the default path
W NCCL AllReduce failed (unhandled cuda error ...) - not retrying NCCL,
  falling back to butterfly AllReduce for the rest of this run.
```

`-sm layer` works because it needs no AllReduce.

## Diagnosis

Tensor parallelism needs at least one working AllReduce. The block-12 `hybrid`
algorithm is NCCL/RCCL for large (prefill) tensors plus the internal
host-staged pipeline for small (decode/verify) tensors. On gfx1100 the internal
half never initialises because `ggml_cuda_ar_pipeline_init` is gated to RDNA4,
so `hybrid` degenerates to NCCL-only. The reporter's RCCL refuses to dispatch
its kernels (`hipErrorIllegalState`) because the PCIe root port has no 32/64-bit
AtomicOp completer (ROCm/ROCm#6520). The runtime fallback catches that and
re-routes to the meta backend's generic butterfly, which is what appears to
hang on this topology.

## Why the internal pipeline is the answer

The internal AR was built for exactly this: it stages every tensor through
mapped pinned host memory over PCIe and never needs peer access or system
atomics (`hipHostMallocPortable | hipHostMallocMapped` +
`hipHostGetDevicePointer`, `__threadfence_system` + volatile arrival ring,
`__builtin_amdgcn_s_sleep(1)`). It was only gated to RDNA4 because that is
where it was validated. A 2x RDNA3-behind-x4 user independently arrived at the
same "force internal AR" fix (recorded in
`archive/work/wip-archive/hybrid-allreduce/12-hybrid-allreduce-hip.md`).

The copy-engine (`ce`) arm is a separate transport that uses
`cudaMemcpyPeerAsync` + cross-device events and no RCCL kernels. It has **no
arch gate** (`ggml_backend_cuda_comm_init_ce` has no gfx check), so it is
already open on RDNA3; it only needs `cudaDeviceEnablePeerAccess` to succeed.

## The patch

Two changes, both inert on RDNA4:

1. `allreduce-hip.cu`: the RDNA4-only gate is bypassed when
   `GGML_CUDA_AR_ALLOW_NON_RDNA4=1` (with a warning that the path is
   unverified on that arch).
2. `ggml-cuda.cu`: when the selected AllReduce function is not the internal
   pipeline and it fails at runtime, the current call is retried on the
   internal pipeline (when one exists) instead of falling through to the
   butterfly. On a RCCL-broken host this is the difference between the first
   prefill AllReduce hanging and completing.

## Options for the reporter

| want | env | needs the patch? |
|---|---|---|
| internal for everything (no RCCL) | `GGML_CUDA_AR_ALLOW_NON_RDNA4=1 GGML_CUDA_ALLREDUCE=internal` | yes |
| hybrid: small -> internal, large -> RCCL, fail over to internal | `GGML_CUDA_AR_ALLOW_NON_RDNA4=1` | yes (the failover retry is what makes it safe) |
| copy-engine for everything (P2P, no RCCL) | `GGML_CUDA_ALLREDUCE=ce` | no, already open on RDNA3 |
| copy-engine + internal for small | `GGML_CUDA_ALLREDUCE=ce GGML_CUDA_AR_ALLOW_NON_RDNA4=1` | yes (only for the internal half) |

`GGML_CUDA_ALLREDUCE=internal` with the patch is the most robust on a root port
without AtomicOp (`atomic` in `dmesg` and `ROCm/ROCm#6520`). `ce` is the
fastest if P2P works but silently degrades to the broken RCCL path if
`cudaDeviceEnablePeerAccess` fails.

## How to try it (source build)

```bash
# fresh delivery tree
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout a55e952b8
bash <this-repo>/scripts/apply-all.sh .
git apply <this-repo>/wip/ar-non-rdna4/ar-non-rdna4.patch
# configure with the RDNA3 target so the gfx1100 code objects are built
cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=ON -DAMDGPU_TARGETS="gfx1100" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Then run any of the option rows above.

## Local validation (gfx1201, 2x R9700, ROCm 7.14.1)

- Clean build of `llama-cli`, `llama-bench`, `test-backend-ops`.
- `-sm tensor` same-seed 4B Q8_0 `llama-cli`: `hybrid`, `internal`, `ce` and
  `GGML_CUDA_AR_ALLOW_NON_RDNA4=1 hybrid` all give `1c5d32ac537d`, the
  delivery's recorded dense 4B hash.
- `llama-bench` 4B Q8_0 `-sm tensor` (pp512 / tg128): hybrid 8508.9 / 105.2,
  internal 6949.1 / 108.4, ce 8713.9 / 105.2.
- The gate env is a no-op on RDNA4 (no extra warning, identical output).

**Not validated on gfx1100 here** (no RDNA3 pair available). The internal path
itself is generic HIP, and the design doc records a 2x RDNA3-behind-x4
deployment that used it. This patch exists so the reporter can confirm.

## Promotion (done 2026-10-04, r9)

Both changes were folded into **block 12** with the non-RDNA4 internal path **enabled by default**
and `GGML_CUDA_AR_ALLOW_NON_RDNA4=0` as the opt-out.  `scripts/validate-set.sh` is green (strict
16/16 `git am`, applied tree == `release.json.tree`).  Validated on gfx1201 (default == `=0` == the
recorded dense 4B `-sm tensor` hash `1c5d32ac537d`); the gfx1100 hardware gate is still open (no
RDNA3 pair on this box) and the reporter has not re-run.  See `WORKLOG.md` 2026-10-04 (r9).
