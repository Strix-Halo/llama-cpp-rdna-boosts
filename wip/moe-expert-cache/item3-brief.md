# Item 3 brief — the user graph-input copies (~5 s/pass headroom)

**Status: OPEN.**  Campaign tip **`f4b255041`**, patch **`exp21-moe-expert-cache-r26-devmap-default.patch`**
(r26 base).  Worktree `~/llama-decode`, branch `wip-moe-devmap-v2`, build `build-rocm`.
Backup ref: `backup/wip-moe-devmap-v2-r26-devmap-default` (`f4b255041`).

This is the self-contained handover for README §0 item 3.  Line numbers are for tip `f4b255041`; re-grep
before editing.  It is the **only remaining campaign item** (B1/B2/B3/B4/B6, item 1 and the `DEVMAP`
default are done).

## Goal

Remove (or hide) the per-split **synchronous H2D of the user graph inputs** (`GGML_TENSOR_FLAG_INPUT`,
e.g. `inp_pos`, `attn_inp_k_idxs`) on a multi-device `-sm tensor` merged routed-MoE band, so the
input-loop host block shrinks.

**Measurement to reproduce (3-device IQ4, `-sm tensor -ncmoe 48 -p 8192 -ub 8192`,
`GGML_SCHED_SYNCDBG=1`):**

```
SCHEDUPLOAD set_async=0 calls 0.0ms | get_async=8 calls ... | input_loop=290 7395.9ms
                                                          against an 18.8 s pass (434.8 t/s)
SCHEDSYNC calls=5964 total=1.04ms   (the per-split full-meta synchronizes)
```

So ~7.4 s of an 18.8 s pass is the input loop: **290 iterations**, each doing an
`event_synchronize` plus a synchronous `ggml_backend_tensor_copy`; the expert copies are no longer the
cost (the B1/B2 gather covers them — `set_async=0`, `get_async=8`).

## Why it is synchronous (the exact code)

* **The user-input branch** — `ggml/src/ggml-backend.cpp` (~**2173-2181**), inside the `split->n_inputs`
  loop of `ggml_backend_sched_compute_splits`:

  ```c
  if (input->flags & GGML_TENSOR_FLAG_INPUT) {
      // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
      if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
          ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
      } else {
          ggml_backend_synchronize(split_backend);
      }
      ggml_backend_tensor_copy(input, input_cpy);   // SYNCHRONOUS
  } else { ... }
  ```

  It is unconditional and is *not* covered by the r26 amendment.

* **The r26 non-INPUT async path** (delivery block-06 amendment) — `ggml-backend.cpp` (~**2484-2499**):
  a host→dev split input is enqueued with `ggml_backend_tensor_set_async` after an in-stream
  `ggml_backend_event_wait`, **only when**:

  ```c
  if (host_src && dev_dst && split_backend->iface.set_tensor_async != NULL &&
      split_backend->iface.event_wait != NULL) { ... }
  ```

* **Why that gate fails under `-sm tensor`** — the Meta backend (`ggml-backend-meta.cpp`, iface tail
  ~**3508-3520**) has `set_tensor_async = ggml_backend_meta_set_tensor_async` (~**2129**) but
  **`event_record = nullptr` and `event_wait = nullptr`**.  So `event_wait != NULL` is false and the
  scheduler keeps the synchronous copy.  (The meta `set_tensor_async` itself *does* handle MIRRORED and
  partial splits — arbitrary ranges for MIRRORED, chunk-aligned for partial — so it is usable; only the
  event wiring is missing.)

Net: **two things must change** — the INPUT branch must get an async treatment, and the Meta backend
needs event wiring (or the async path needs a meta-aware ordering alternative) before the r26 path can
apply to `-sm tensor`.

## The session-17 crash (do NOT repeat it)

A naive `set_tensor_async` on the INPUT branch crashed at `-ub 2048` with
**`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`**: a `-ub 2048` prefill splits into **4 ubatches**, each of
which runs `set_inputs` + `graph_compute` and **overwrites the same input buffer**, so the source bytes
changed between the enqueue and the copy.  The fix must pin the source lifetime to the copy lifetime.

## Options (easiest first)

1. **Event wiring for the Meta backend.**  Give `ggml_backend_meta_i` an `event_record`/`event_wait` that
   records on / waits for **all** simple backends (or add a meta-aware ordering hook).  Then the existing
   r26 async path covers `-sm tensor`.  This alone fixes the non-INPUT split inputs; the INPUT branch still
   needs (2) or (3).  Watch: `ggml_backend_event` is currently per-device; a meta event must own one event
   per simple device and wait on all.
2. **Per-ubatch staging buffer (the robust fix).**  On the first split that needs a user input, copy it
   **once** into a scheduler/pinned staging buffer (host→host, cheap), then have every split `set_tensor`
   / `set_tensor_async` from the staging buffer; reuse of the staging buffer across ubatches is
   event-guarded (wait for its pending copies before the next ubatch overwrites it).  Turns N-device ×
   splits synchronous H2D into one small host copy + async H2D per split.
3. **Source-stability gate (safe subset).**  Take the async path only when the source is provably stable
   for the whole graph (e.g. `n_ubatch == 1` / a single decode ubatch).  Correct and cheap to reason
   about, but it does **not** help the `-ub 8192` (4-ubatch) case the measurement targets.

Option 2 is the one that matches "match the source lifetime to the copy lifetime"; option 1 is a
prerequisite for the non-INPUT path on meta but does not by itself fix the INPUT branch.

## Gates (all six; use the campaign's current defaults: DEVMAP on, KSLOT on)

* **Byte-identity:** 1-GPU `-sm layer` Q4_K_M `15038c19ddc8` / Q8_0 `ba0b9b47c2d1`; 2-GPU `-sm tensor`
  `de8be4d0c90c`; 3-GPU `-sm tensor` `d7bef4c6fdc3`.  Also with the cache **off** (`MIB` unset) and with
  `DEVMAP=0`, so the scheduler change is proven orthogonal to the cache.
* **Width purity** `none == n1 == n3 == n7` (35B-A3B Q4_K_M, `MIB=8192`, MTP).
* **`test-backend-ops -o MUL_MAT_ID`** OK; **deep coherence** (`coherence-essay-prompt.txt`,
  `-n 12000 -c 16384 --spec-type draft-mtp --spec-draft-n-max 3`, rc=0 + `## Conclusion`).
* **The `-sm tensor` gather gate** must stay: 2-GPU tensor Q4_K_M `-ncmoe 40` `pp2048 -ub 512`
  `llama-bench` ~**724 t/s** (was 609 with the gather off — do not regress item 1).
* **Measurement:** `GGML_SCHED_SYNCDBG=1` `input_loop=N Tms` and `SCHEDSYNC` at `-ub 8192` **and**
  `-ub 2048`; the `-sm layer` 1-GPU path must not regress.

## Files

| concern | location |
|---|---|
| user-input sync branch | `ggml/src/ggml-backend.cpp` ~2173-2181 (`GGML_TENSOR_FLAG_INPUT`) |
| r26 non-INPUT async path + `wait_before_overwrite` | `ggml/src/ggml-backend.cpp` ~2239, ~2484-2499 |
| staging ring issue / drain | `ggml/src/ggml-backend.cpp`: `sched_stage_issue` ~2035, `sched_input_gatherable`, the consumed-slot branch ~2176-2210 |
| instrumentation | `ggml/src/ggml-backend.cpp` ~275 (`SCHEDUPLOAD` / `input_loop`), ~450 (`SCHEDSYNC`), `ggml_backend_event_synchronize` ~602 |
| meta async + iface | `ggml/src/ggml-backend-meta.cpp`: `ggml_backend_meta_set_tensor_async` ~2129, `stage_input` ~2375, iface tail ~3508 (`event_record`/`event_wait` NULL) |
| cuda set_tensor_async | `ggml/src/ggml-cuda/ggml-cuda.cu` ~3209 |

## Reproduce / measure (copy-paste)

```sh
cd ~/llama-decode && git switch wip-moe-devmap-v2 && git log -1        # expect f4b255041
cmake --build build-rocm --target llama-cli llama-bench -j 16

IQ4=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
HIP_VISIBLE_DEVICES=0,1,2 MOE_EXPERT_CACHE_MIB=16384 GGML_SCHED_SYNCDBG=1 \
  ./build-rocm/bin/llama-cli -m $IQ4 -ngl 99 -ncmoe 48 -sm tensor -fa 1 -lzm auto -lm none -t 8 -c 8192 \
  --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt -p "Hello" -n 8 -ub 8192 2>&1 | tail -4
# watch SCHEDUPLOAD input_loop / SCHEDSYNC
```

## Rules

* `wip/` only; applies only to `~/llama-decode`; never to the delivery `patches/`; `~/llama-decode` is
  never pushed.  Delivery-repo docs may be committed+pushed to `origin/main`.
* Pin `-t 8`; `llama-cli` always `--single-turn --no-display-prompt`; never `-v` for hash runs; never run
  parallel benches; on this host the GPU IRQs sit on the top cores (see AGENTS.md).
* Opt-in first, gate, then flip; a negative result with attribution is a valid outcome.
* When done: append a dated `WORKLOG.md` entry + README update, cut the next `expNN`
  (`git diff 0d58404e1..wip-moe-devmap-v2`), and commit/push the delivery-repo docs.
