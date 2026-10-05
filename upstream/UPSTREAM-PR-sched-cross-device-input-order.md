# UPSTREAM-PR: ggml-backend: order a cross-device split-input copy after the destination's queued work

**Status:** candidate, written against upstream master `a55e952b8` and re-cut from the fork's block-06
fix (`v16-a55e952b8-r12`, issue #103 / PR #104).  The **fork** form is validated (FAIL -> PASS on
2 x R9700 with a prefill-rebalance harness); the upstream form below is a straight port of the same
hunk and has not yet been built or measured on pristine upstream.  Not filed.

**File:** `ggml/src/ggml-backend.cpp` — `ggml_backend_sched_compute_splits()`, one insertion, +15 lines
(6 code + 9 comment).

## The bug

A device-to-device split input is copied by the destination backend's `cpy_tensor_async`, but the CUDA
implementation runs the copy on the **source** backend's stream:

```c
// ggml_backend_cuda_cpy_tensor_async(), cross-device branch
CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, nbytes, cuda_ctx_src->stream()));
```

`ggml_backend_sched_compute_splits()` only makes the **destination** stream wait for its own previous
work:

```c
// wait for the split backend to finish using the input before overwriting it
if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
} else {
    ggml_backend_synchronize(split_backend);
}
```

so the incoming copy is not ordered against work already queued on the destination backend.  The
dangerous case is the **outbound** copy of an earlier split's output: it is issued on the destination
backend's stream when the next split collects its inputs, i.e. *after* the earlier split's event was
recorded.  If the allocator reuses that output's region for the new input, the incoming copy on the
source stream can land before the outbound copy has read it.

This shows up with host-resident MoE experts under `-sm layer` on 2 GPUs: the offloaded-expert ops of
both GPUs' layers all execute on device 0, so every layer boundary crosses devices and the allocator
recycles the activation regions.  The visible symptom is garbage prefill output (or a hung decode, as
the corrupted activations feed the sampler).  Making the source wait on the *existing* per-split event
is not enough: that event marks the end of the earlier split, not the outbound copy queued after it.

## The fix

Right before the copy paths, record a fresh event on the split (destination) backend and make the
source backend wait on it.  The fresh record is ordered after the outbound copy, so the source stream
cannot start the incoming copy until the outbound read is done.  Host-sourced inputs are excluded
(they are enqueued on the split backend's stream and already ordered); backends without `event_wait`
are excluded; when events are disabled the existing full-synchronize path already covers it.

## What was validated

Fork-side (delivery block 06, `v16-a55e952b8-r12`), on 2 x R9700 / ROCm 10.0.0 / qwen4exp
UD-IQ3_XXS + shared Q8_0 MTP, greedy, the reporter's flags (`-sm layer --n-cpu-moe 48 -ub 2048
-b 2048`, q8_0 K/V, `--spec-type draft-mtp`, `GGML_SCHED_SYNC_GRAPH_INPUTS=1`,
`LLAMA_MTP_DRAFT_OP_OFFLOAD=0`, `LLAMA_MTP_SPARSE=0`).  On this box (x16 links) the stock tree does
not fail by itself; the race was opened with a scratch harness that forces the block-06 prefill
rebalance, after which stock gives `!!!!` (2/2), the same tree with `GGML_SCHED_EVENTS=0` is correct,
and this fix (patch 0002) is correct (3/3 at 580 lines, 1/1 at 900 lines).  Same-seed greedy on
Qwen3.5-4B-Q8_0 with `-sm layer` and `-sm tensor` on 2 GPUs is byte-identical to stock.  Full record:
`archive/work/2gpu-sched-fixes/VERIFICATION.md`.

## What was NOT validated

- The upstream form itself has not been built or benchmarked; re-cut on the PR branch and run an
  equivalent multi-GPU host-expert case, or a deterministic harness, before filing.
- The new wait is the scheduler's first **cross-backend** `ggml_backend_event_wait` (an event created
  on `split_backend`, waited on by `input_backend`).  CUDA events are device-generic, but several
  backends' `event_wait` assume their own event object (`ggml_backend_sycl_event_wait` does a
  `static_cast<sycl::event*>`, Vulkan casts to `vk_event*`, Metal to `ggml_metal_event_t`).  A mixed
  device-backend scheduler (for example CUDA plus SYCL/Vulkan) would need a same-family guard before
  this can be filed broadly.  The delivery is CUDA/meta/CPU only, so it is not hit there.
- No per-change kill-switch; `GGML_SCHED_EVENTS=0` disables it only by turning off all per-split
  events, which has a measured prefill cost.
