# 2gpu-sched-fixes (issue #103): two fixes for `-sm layer` on 2 GPUs with host-resident experts

Two independent, small fixes found on 2 × R9700 (gfx1201, PCIe 5.0 x8 each) with qwen4exp and
`--n-cpu-moe`.  Both are format-patches on top of `v16-a55e952b8-r11` (tree `38ebce2f`), applied
after `patches/00*.patch` with `git am`.  Full report, repro and the bisect table: issue
[#103](https://github.com/stew675/llama-cpp-rdna-boosts/issues/103).

| patch | file | what |
|---|---|---|
| [`0001-...alias-.patch`](0001-moe-expert-cache-validate-device-and-layer-on-alias-.patch) | `ggml/src/ggml-cuda/moe-expert-cache.cu` | trust an expert-cache alias only if its table is on the calling device (and in the op's layer) - fixes a GPU page fault in `moe_cache_tally_kernel` |
| [`0002-...split-input-copies-a.patch`](0002-ggml-backend-order-cross-device-split-input-copies-a.patch) | `ggml/src/ggml-backend.cpp` | order a cross-device split-input copy after the destination backend's queued work - fixes garbage prefill output with the default-on scheduler events |

## 0002: cross-device split-input copy race (garbage output)

**Symptom.** `-sm layer` on 2 GPUs, `--n-cpu-moe >= 29` (GPU 1's layers have host experts),
`-ub 2048`, a 36k-token prompt: the reasoning comes out as `!!!!…`.  `GGML_SCHED_EVENTS=0`,
`GGML_SCHED_STAGE=0` or `HIP_LAUNCH_BLOCKING=1` each hide it.

**Cause.**
- A device-to-device split input is copied by `ggml_backend_cuda_cpy_tensor_async` on the
  **source** backend's stream.  `wait_before_overwrite()` makes only the split backend's own stream
  wait, so the copy is not ordered after work already queued on the split backend.
- With host experts every MoE op of GPU 1's layers runs on GPU 0.  GPU 0 computes split k, and its
  output is copied to GPU 1 by an outbound copy issued on GPU 0's stream **after** split k's event was
  recorded (when the next GPU 1 split collects its inputs).
- Split k+1 then copies its input from GPU 1 into GPU 0's buffer on GPU 1's stream.  When the
  allocator has reused split k's output region for that input, the incoming copy can land before the
  outbound copy has read it.
- A source-side wait on `sched->events[split_backend_id]` is not enough: that event marks the end of
  split k, not the outbound copy queued after it.

**Fix.** Right before such a copy, record a fresh event on the split backend and make the source
backend wait on it.  Host-sourced inputs are excluded (they take the existing paths).  Re-recording
the per-split event at a later point only makes the existing waits on it stricter.

## 0001: device-blind expert-cache alias lookup (GPU page fault)

**Symptom.** The same setup with `MOE_EXPERT_CACHE_MIB=6144` crashes on the first long prompt:
`Memory Fault Error [GPU index: 0, ... kernel: moe_cache_tally_kernel(...)]`, "page not present".

**Cause.** `moe_cache_tally_prefill` (fallback) and `moe_cache_get_table` (first) resolve
`g_alias_to_id` without checking the table's device.  Aliases are keyed by scheduler tensor
addresses, which the allocator reuses across graphs, so on 2 GPUs a lookup can return the other
device's table, and its `prefill_count_dev` / arena is then used from the wrong GPU.

**Fix.** `alias_find_checked()` trusts an alias only when the table is on the calling device and,
when the op is known, in the op's layer; otherwise it falls through to the existing semantic
`(layer, role, device)` lookup.  A one-time warning reports an ignored alias.  On its first run it
also caught a same-device, wrong-layer alias (table layer 7, op layer 3), so it may matter on one GPU.

## Validation

2 × R9700, ASUS ProArt X870E-Creator, kernel 7.0.0-38, ROCm 10.0 container, ISTA-DASLab
Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS + shared Q8_0 MTP head, `-c 262144 -sm layer --n-cpu-moe 48
-ub 2048 -ctk q8_0 -ctv q8_0`, ring 4 slots / 2048 MB, greedy needles ("how many buildings").

| build | `--n-cpu-moe 48`, 36k / 97k needle | `+ MOE_EXPERT_CACHE_MIB=6144` |
|---|---|---|
| r10 / r11 stock | `!!!!` | GPU page fault in `moe_cache_tally_kernel` |
| r10 + 0001 + 0002 | correct (7 checks over 5 runs, `--n-cpu-moe` 29 and 48) | not run |
| r10 + 0001, `GGML_SCHED_EVENTS=0` | (not the point of this run) | correct, no faults |
| r11 + 0001 + 0002 | correct / correct | correct, no faults, 1 stale alias ignored |

Speed (r10, events on), second fresh prompt of each size after load:

| build | decode | 37k prefill | 155k prefill |
|---|---|---|---|
| r10 stock (garbage) | 35.6 t/s | 1,114 t/s | 1,120 t/s |
| r10 `GGML_SCHED_EVENTS=0` (workaround) | 35.8 | 1,088 | 1,027 |
| r10 + 0001 + 0002 | 34.9 | 1,146 | 1,111 |

All-experts-in-VRAM on 2 GPUs (`--n-cpu-moe 0`) and 1-GPU setups are not affected by the bug.  I have
not benchmarked 0002 in those layouts; there, events on vs off made no measurable difference (r10:
37k 1,830 vs 1,818 and 1,194 vs 1,194 t/s), so the extra wait should not matter either.

Not changed: the separate observation that every host-weight split runs on backend 0 (the
"rebalance onto the owning device" pass does not take effect in this layout) - noted in #103.
