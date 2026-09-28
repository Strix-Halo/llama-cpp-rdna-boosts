# Phase 2 (`-sm layer`, multi-GPU): the device bug is FOUND, and one of my fixes INTRODUCED a regression

**Date: 2026-09-28.  Author: this session.  Status: UNFINISHED - the worktree is left with UNCOMMITTED
Phase-2 changes that are KNOWN-BROKEN.  `git checkout .` in `~/llama-decode` returns it to the last good
state (`922098442`, the 1d commit).  The diff is preserved as `phase2-sm-layer-WIP.patch` next to this file.**

## 1. The bug the maintainer reported: only 1 GPU active ("Item 2 exists to catch this sort of thing")

**Root cause, confirmed by instrumentation.**  `ggml_backend_sched_split_graph` chooses the device for an op
with a host-resident weight in pass 1:

```c
if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
    for (int b = 0; b < src_backend_id; b++) {          // <-- FIRST match wins
        if (ggml_backend_supports_op(...) && ggml_backend_offload_op(...)) return b;
    }
}
```

With `-ncmoe` the expert weight is host memory, so the op is runnable on ANY GPU and the loop always returns
the **lowest-index** one.  Measured under 2 GPUs + `-sm layer`: **all 120 offloaded MoE ops on CUDA device 0,
device 1 idle**, and for every layer-1 op two cross-device copies (activation in, result out).  That is why
the cached 2-GPU path measured *slower* than the 1-GPU one (Q4_K_M 44.62 vs 56.21; Q8_0 42.39 vs 52.06).

The layer split itself is fine - it was verified independently: `layer 0..20 -> ROCm0, 21..41 -> ROCm1`.

**The fix that works: a pass-3.5 rebalance** (inserted after pass 3, before pass 4), which moves each
host-weight op onto the device that OWNS its layer.  The owning device cannot be read from the op's own data
inputs there - they are still `-1` at that point (probe output:

```
moe probe2: ffn_moe_gate-0 node_id=0 src[1]=attn_post_norm-0 (reshaped) w=0 bid=-1
moe probe2: ffn_moe_gate-0 node_id=0 src[2]=ffn_moe_topk-0               w=0 bid=-1
```

) and pass 4 would only drag them onto whatever device pass 1 picked.  It is read from the layer's
**device-resident** weights instead (`layer_dev[L]` built from the non-host WEIGHTS buffers, layer parsed
from the `blk.<N>.` name).  Verified working:

```
moe rebalance: MUL_MAT_ID ffn_moe_gate-21 -> ROCm1 (layer 21)     ... 399 moves
moe hook: offloaded MUL_MAT_ID expert upload handled by CUDA device 0
moe hook: offloaded MUL_MAT_ID expert upload handled by CUDA device 1     <-- both GPUs now active
```

Two dumb bugs cost time on the way there and are worth recording: my first guard was `*node_backend_id <= 0`,
which treats **device 0** as "unassigned" (`0` is a valid GPU id; unassigned is `-1`) and skipped exactly the
ops it was meant to move; and an early attempt put the preference in pass 1, where it can never fire.

## 2. A second, real bug the fix EXPOSED

Once layer 21+ MoE ran on device 1, it **faulted**:

```
Memory Fault Error [GPU index: 1, faulting addr: 0x7f48ef14c000,
  kernel: void mul_mat_vec_q_moe<(ggml_type)12, 2, false>(...)]
```

Cause: the arenas were `cudaMalloc`ed with whatever device was *current* when sizing ran - i.e. all on
device 0 - and `cudaHostGetDevicePointer` was likewise resolved on one device.  So a device-1 kernel read a
device-0 pointer.  Fixed by making the cache per-device: `table_t.device`, a `device_guard` around
`alloc_table_locked`, the UVA alias resolved with that device current, and the arena allocated **lazily at
first hooked use** (the sizing pass runs with one device current, so allocating every table there puts every
arena on that device).  After that: no fault.

## 3. What is PROVEN

* Both GPUs handle the offloaded MoE under `-sm layer` (the reported bug).
* The 2-GPU path no longer faults.
* **The 2-GPU cache is deterministically byte-identical to the 1-GPU cache**: `4968c937e7c9` for
  2gpu-cache-a, 2gpu-cache-b and 1gpu-cache (Q4_K_M, fusions off, 300 tokens).  So the layer split and the
  rebalance preserve the arithmetic exactly - which is what the rebalance was supposed to guarantee.
* Throughput, 2 GPUs, stable over 3 repeats: no cache 37.30, cache 16 GiB 42.46/42.60/43.27 (+15 %).
  1 GPU cache 16 GiB is 61.34, i.e. **for a model that FITS one card, forcing `-sm layer` costs a lot and the
  cache does not recover it** - expected, because a layer split is a serial pipeline that adds transfer cost
  without adding decode parallelism.  **Phase 2's value is therefore CAPACITY (VRAM to cache experts of a
  model that does not fit), not throughput.**  That also means `MOE_EXPERT_CACHE_MIB` should probably become
  **per device**, since today it is a global total that gets split across devices and so buys no extra
  capacity with the second card.

## 4. THE REGRESSION I INTRODUCED - read this before reusing the patch

**The deferred-sizing path (`MOE_EXPERT_CACHE_MIB`) stopped being transparent.**  Measured 1 GPU, fusions
off, 300 tokens, oracle `ad30da7b5a3a`:

| knob | hash | chars |
|---|---|---:|
| `MOE_EXPERT_CACHE_SLOTS=64`  | `ad30da7b5a3a` | 1386 | (pure) |
| `MOE_EXPERT_CACHE_SLOTS=128` | `ad30da7b5a3a` | 1386 | (pure) |
| `MOE_EXPERT_CACHE_MIB=2048`  | `64d5cb99b395` | 806  | **impure, and short output** |
| `MOE_EXPERT_CACHE_MIB=8192`  | `ee68cad3a202` | 1046 | **impure** |
| `MOE_EXPERT_CACHE_MIB=10240` | `86868e8cdfa4` | 1047 | **impure** |

Before this session's per-device restructuring, `MIB=8192` (fusions off) WAS `ad30da7b5a3a` - see the 1D
RECORD.  So this is a regression from the restructuring, not a pre-existing defect.

The `MIB=2048` row is the alarming one: 806 chars for a `-n 300` run means the model stopped early, i.e. it
degraded.  **Do not use the deferred path from this patch.**

Decisive clue for whoever picks this up: `MIB=10240` resolves to the *same effective slot count* as the pure
`SLOTS=128` (both cap at `n_experts`), yet their hashes differ.  So the defect is **not the slot count** - it
is the path.  The prime suspect is the restructuring of the sizing block: it was

```c
if (g_slots_hint <= 0) { priming; alloc_all_locked(); arena check; }
```

and became

```c
if (!t.primed) { priming; return false; }        // now applies to the EXPLICIT path too
if (!g_sized) alloc_all_locked();                // no longer allocates; computes g_uniform_slots only
if (!t.allocated) { t.device = device; alloc_table_locked(t, g_uniform_slots); }
```

The explicit path still passes, the deferred path does not, so the bug is most likely in *when*
`g_uniform_slots` is computed relative to the priming pass (the sizing now happens on the first table's
SECOND call rather than on a separate allocation sweep) - check whether `g_total_expert_bytes` is complete at
that moment, and whether `g_uniform_slots`/`g_total_one_expert_bytes` are consistent between the two paths.

## 5. Recommended next steps

1. **Revert to `922098442`** (`git checkout .` in `~/llama-decode`) - the 1d state, where the deferred path is
   pure.  Re-apply the rebalance pass alone first (it is self-contained and clearly correct) and re-verify
   `MIB=8192` fusions-off still gives `ad30da7b5a3a` on 1 GPU.
2. **Then** add the per-device arena allocation, and re-run the same 1-GPU gate plus the 2-GPU pair
   (`4968c937e7c9` twice, and equality with the 1-GPU cache).
3. Fix the deferred-path regression before anything else - a cache that changes the output is not shippable,
   whatever the throughput says.
4. Consider `MOE_EXPERT_CACHE_MIB` **per device** (see section 3) - otherwise the second card buys no cache
   capacity, which is the whole point of Phase 2.
5. The rebalance pass is a **general** fix (it also affects the delivered prefill MoE offload on any
   multi-GPU `-sm layer` config, which has the same lowest-index-wins behaviour), so it is a good
   `upstream/` candidate and a candidate for a delivery block - but it needs its own gates first.
