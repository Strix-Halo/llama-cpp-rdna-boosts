# Phase 2 (`-sm layer`, multi-GPU): the device bug is FOUND, and one of my fixes INTRODUCED a regression

**Date: 2026-09-28.  Author: this session.  Status: RESOLVED (2026-09-28, session 3).**  Section 1's device
bug is fixed (commit `d5868bb5a`), section 4's regression was reverted and never returned, and step 3 -
the per-device arenas - is landed as commit `44ebd14b6` (see section 6).  The 2-GPU `-sm layer` path now
runs, is byte-identical to the 1-GPU path and to the full-table GPU oracle, and `MOE_EXPERT_CACHE_MIB` is
per device (README NEXT STEP item 4).  Sections 1-4 remain the history of how the bug was found (and the
broken `phase2-sm-layer-WIP.patch` stays reference-only).

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

## 4b. STATUS AFTER THE REVERT + RE-APPLY (verified)

Executed, in this order:

1. **Reverted** `~/llama-decode` to `922098442` (1d).  Baseline re-verified: `MIB=8192`, `MIB=2048`,
   `SLOTS=128`, `FAIL_ALLOC=1` all `ad30da7b5a3a` (1386 chars) - **the deferred path is pure again**, so the
   regression was entirely this session's per-device restructuring.
2. **Re-applied the rebalance pass alone** (commit `d5868bb5a`, +96 lines across
   `ggml/src/ggml-backend.cpp` and `ggml/src/ggml-cuda/ggml-cuda.cu`; no cache-module changes at all).
   **Verified pure on 1 GPU**: `MIB=8192` / `MIB=2048` / `SLOTS=128` all `ad30da7b5a3a`, 1386 chars.  So the
   rebalance is a no-op on 1 GPU and does NOT touch the deferred path - which confirms the regression's
   cause was the allocation restructuring, not the rebalance.
3. **2 GPUs**: both devices now handle the offloaded MoE (device 0 and device 1, 399 moves) - the reported
   bug is fixed.  The 2-GPU *run* still aborts, as expected at this step: the arenas are still allocated on
   device 0, so a device-1 kernel reads a device-0 pointer and faults.  **Step 3 is required before the 2-GPU
   path is usable.**

## 5. Recommended next steps

1. ~~Revert to `922098442`, re-apply the rebalance pass alone, re-verify `MIB=8192` fusions-off gives
   `ad30da7b5a3a` on 1 GPU.~~  **DONE 2026-09-28** - see section 4b.  The rebalance alone is at `d5868bb5a`.
2. **Add the per-device arena allocation - and do it WITHOUT restructuring when tables are allocated.**
   That restructuring (moving the allocation out of `alloc_all_locked()` into a lazy per-table call in the
   hook) is what regressed the deferred path.  **The safe route: leave the sizing/allocation structure
   exactly as it is** (every table allocated in one sweep, on the then-current device = device 0) and then,
   in the hook, on the first use of a table whose `t.device != device`, **migrate that one table's arena**
   (free on the old device, allocate on the right one) under a device guard.  Add `table_t.device` and
   resolve the UVA alias per device at that point.  This keeps `g_uniform_slots`/`g_total_expert_bytes` and
   the priming-pass timing untouched, which is the whole point.
   Then re-run the 1-GPU gate above (`MIB=8192` and `MIB=2048` must both stay `ad30da7b5a3a`) **and** the 2
   GPUs (no abort; the 2-GPU cache equals the 1-GPU cache).
3. Fix the deferred-path regression before anything else - a cache that changes the output is not shippable,
   whatever the throughput says.
4. Consider `MOE_EXPERT_CACHE_MIB` **per device** (see section 3) - otherwise the second card buys no cache
   capacity, which is the whole point of Phase 2.
5. The rebalance pass is a **general** fix (it also affects the delivered prefill MoE offload on any
   multi-GPU `-sm layer` config, which has the same lowest-index-wins behaviour), so it is a good
   `upstream/` candidate and a candidate for a delivery block - but it needs its own gates first.

## 6. RESOLUTION (2026-09-28, session 3): per-device arenas + per-device MIB - 2-GPU `-sm layer` WORKS

Commit `44ebd14b6` on `wip-moe-expert-cache` (on top of the rebalance commit `d5868bb5a`).  The whole Phase
2 blocker list (sections 5.2 and 5.4) is done.

**Ingredient 1 - per-device arenas (step 3).**  `table_t` carries `int device`; `moe_cache_table` and
`moe_cache_update_host` take the CUDA ordinal (the backend adapter passes `ctx->device`, NOT the scheduler
iface).  A `device_guard` RAII makes a table's owner device current for every `cudaMalloc`/`cudaFree`/
`cudaHostGetDevicePointer`, and `alloc_table_locked` runs under it.  Crucially, **when tables are allocated
is unchanged**: the priming pass records `t.device` while it registers every table, and the one-sweep
`alloc_all_locked()` then allocates each table on its own device.  That is the tightrope the previous
session fell off - it moved allocation into the hook and regressed the deferred path; this keeps
`g_sized`/priming/`g_uniform_slots` timing byte-for-byte identical (gate (a) proves it).  A migration
backstop in the hook (free on the old device, realloc on the new, clear the residency map) covers a late
device discovery; in practice it never fires, because a table's owner is learned on pass 1, before any
capture.

**Ingredient 2 - `MOE_EXPERT_CACHE_MIB` is now PER DEVICE (README NEXT STEP item 4).**  `alloc_all_locked`
groups the unallocated tables by `t.device`, clamps `g_budget` against each device's own free memory (minus
`_RESERVE_MIB`), and gives every table on a device the same `budget_d / one_expert_bytes_d` slots.  Uniform
within a device (gate/up/down stay aligned); a second card now buys a second arena instead of splitting one
budget.  A single-device run is unchanged.  The exit report gained a per-device breakdown.

### Gates (all in this session; Q4_K_M, `-ngl 99 -ncmoe 99 -fa 1 -sm layer -t 8 -c 8192`, 300 tokens,
seed 42, fusions off, `prompts/reasoning.txt`, `GGML_OP_OFFLOAD_MIN_BATCH=0`)

| gate | config | result |
|---|---|---|
| (a) 1-GPU regression | `MIB=8192` / `MIB=2048` / `SLOTS=128` / no-cache | **all `ad30da7b5a3a`** (1386 chars) |
| (b) 2-GPU purity | `MIB=8192` / `MIB=2048` / no-cache | **all `ad30da7b5a3a`** |
| (b) width purity | 2 GPU `none` / `n1` / `n3` / `n7` | **all `ad30da7b5a3a`** |
| (b) fail-soft | `FAIL_ALLOC=1`, 1 GPU and 2 GPU | `ad30da7b5a3a` |
| (c) both devices active | `-v`, `MIB=2048` | device 0 **and** device 1 in the hook WARN; 912 rebalance moves |

The 2-GPU cache hash equals the 1-GPU cache hash **and** the full-table GPU oracle: the layer split, the
rebalance and the per-device arenas change no arithmetic.

**Per-device sizing is real** (`MIB=2048`, 2 GPU, `-v`):

```
alloc_all_locked: device 0: sized 53 slots/table from 63 tables / 38.1 MiB per expert (budget 2048.0 MiB/device)
alloc_all_locked: device 1: sized 58 slots/table from 57 tables / 34.8 MiB per expert (budget 2048.0 MiB/device)
moe_cache_report: device 0: tables=63 slots/table(sum)=3339 arena=2017.3 MiB
moe_cache_report: device 1: tables=57 slots/table(sum)=3306 arena=2020.5 MiB
```

At `MIB=2048` the all-roles hit rate is now **h=0.6090** (was 0.4682 when the global budget was split in
half), i.e. the second card buys ~2x the cached expert coverage for the same MIB.

### Throughput (2x R9700 gfx1201, Q4_K_M, `-t 8`, `llama-bench -n 512 -r 3`, fusions on)

| config | tg512 t/s |
|---|---:|
| 1 GPU no-cache | 38.65 |
| 1 GPU cache 8192 | **57.64** |
| 2 GPU no-cache | 37.67 |
| 2 GPU cache `MIB=2048` | 35.77 |
| 2 GPU cache `MIB=8192` | **43.02** (+14.2 % vs 2-GPU no-cache) |

This confirms section 3's conclusion: for a model that FITS one card, forcing `-sm layer` costs throughput
and the cache does not recover it (1 GPU 57.64 vs 2 GPU 43.02).  Phase 2's value is **CAPACITY** for a
model that does not fit; the throughput pair is recorded only as the step-3 gate.  `MIB=2048` being
*slower* than no-cache is the same shape - at a low hit rate the UVA cold reads cost more than the r19
threaded CPU MoE saves; the cache pays off from ~4-8 GiB/device on this model.

### MTP health (2 GPU, cache `MIB=8192`, fusions ON, `-lv 4 -n 2000`, `prompts/reasoning.txt`)

| config | tg t/s | acceptance | mean len |
|---|---:|---:|---:|
| `none` | 42.50 | - | - |
| `n3` | **86.12** | 0.82753 (1425/1722) | 3.48 |

`draft-mtp` beats plain by +103 % and pos-1 acceptance (0.83) is far above the 0.45 floor.

### Still open (not blockers)

* The migration backstop is untested in anger (it should never fire).  Leave it, or delete it if the
  explicit-slots path is retired.
* Phase 3 (`-sm tensor`): the end-goal geometry.  The per-device plumbing here is a prerequisite, but the
  tensor-split case has per-device *slices* of an expert and the cold path MUST be UVA (see README section
  3.5), so the arena shape and the `device_alias()` seam need a second look there.
* Item 5 (rebalance as a general/`upstream/` candidate) is untouched by this session.
