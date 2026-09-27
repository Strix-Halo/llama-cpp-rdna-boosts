# UPSTREAM-PR: keep host-resident MoE expert weights pinned (don't downgrade `MUL_MAT_ID` to the mmap)

**The change.** `src/llama-model-loader.cpp`'s `select_weight_buft` has an "avoid using a host buffer when
using mmap" guard that replaces the pinned host buffer type with the CPU (mmap'd) buffer type for every
upstream buffer type when the model is memory-mapped.  That is right for tensors the CPU reads, but a
`MUL_MAT_ID` **weight** is the one tensor the scheduler's op-offload (`-ncmoe` / `--cpu-moe`) H2D-uploads
to the GPU every ubatch.  Downgrading it makes those uploads read the pageable model mapping, which on
ROCm is a measurable loss and, for the meta backend's 2-D spliced upload, a **fault**.  The guard now
skips the downgrade for `MUL_MAT_ID` weights only.

This is arguably restoring the loader's own stated intent: `make_cpu_buft_list` adds the device host
buffer type precisely because storing CPU tensors there "reduces the time spent on data transfers" when
batches are offloaded to a GPU, and the mmap downgrade then defeats that for the weights that matter most.

**Where it lives in the delivery.**  Block 06 (the general system-operations bucket), release
`v16-84e76d8a2-r15`.  Kill-switch: `LLAMA_MMAP_HOST_EXPERTS=0` restores the old mmap behaviour.

**Patch.**  `UPSTREAM-PR-moe-host-expert-pinning.patch`.  Applies clean to upstream master `84e76d8a2`
(`git apply --check` verified).  It is **not** to be applied on top of the full delivery set (it would
double-apply).

**Why it is a good standalone PR.**  It is 3 functional lines, one condition, one `#include`-free
`getenv`-gated static, touches one loader file, changes no arithmetic, and is trivially A/B-able.  The
cost is explicit and the user controls it (RAM pinning).  The failure mode it fixes (pageable MoE weight
uploads) is generic to any backend whose host-to-device copies are synchronous from pageable memory; ROCm
is where it was measured, but the loader logic is backend-independent.

**Validation (delivery side, gfx1201, 2x R9700, ROCm 7.14.1).**

* Same-seed greedy text **byte-identical** with `LLAMA_MMAP_HOST_EXPERTS=1` and `=0`
  (`llama-cli -m Qwen3.6-35B-A3B-UD-Q4_K_M -ngl 99 -ncmoe 99 -sm tensor -fa 1
  -p "The capital of France is" -n 20 --seed 42 --temp 0`, `sha=359ff4337837` both ways) — the change
  moves only the buffer type, never the arithmetic.
* **+83 %** prefill: `llama-bench -ncmoe 99 -fa 1 -p 8192 -ub 8192 -sm tensor` -> **5104 t/s pinned vs
  2794 t/s unpinned**.
* Independently corroborated: GenerelSchwerz's `moe-cache` llama.cpp fork ships pinned host staging for
  pageable MoE sources (its tip commit is "stage automatically pageable MoE legacy sources"); see
  `wip/tensor-split-expert-split/README.md` §26.

**What was NOT validated.**  No NVIDIA/CUDA hardware was available; the CUDA path's benefit is inferred
from the same mechanism (a pinned source makes `cudaMemcpyAsync` truly asynchronous, and CUDA's 2-D copy
does not need the ROCm workaround that the delivery's fast path already carries).  The 2-D-upload fault
is ROCm-specific.  Other backends (Vulkan/Metal/CPU) are unaffected either way, since the guard only
changes which host buffer type a `MUL_MAT_ID` weight lands in and those loads are not op-offloaded the
same way.
