# RESOLUTION — two MMQ `MUL_MAT_ID` tail over-read holes in the host-resident path (r31, 2026-10-03)

Status: **resolved and landed as `v16-84e76d8a2-r31`.**  This closes the investigation `HANDOVER.md`
opened.  It found **two independent** holes, not one, and the fix is two small guards.

> **Scope.**  Both holes need a *partial / pruned* expert buffer, which only exists when experts are
> **host-resident** (`-ncmoe`).  With all MoE weights device-resident, `MUL_MAT_ID` reads the model's
> weight tensor directly: the tail over-read reads adjacent experts (finite) and, past the last one, the
> backend's **zeroed allocation padding** (`ggml_backend_cuda_buffer_init_tensor`:
> `cudaMemset(data + original_size, 0, padded_size - original_size)` — "initialize padding to 0 to
> avoid possible NaN values").  The gather/staging/arena are all host-expert machinery
> (`sched_input_gatherable` requires a host weight), so an all-resident model is unaffected.

## The shared root cause

The quantized `MUL_MAT_ID` MMQ weight-tile loader reads a full K tile and does **not** clamp its read to
the row for the fast (non-fallback) path.  The last row of an expert therefore over-reads into the
**head of the next slot**.  Upstream covers this on the host copy path (`copy_experts` copies
`expert_size_copy + min(expert_size, 512)` bytes).  Every pruned/partial device buffer must provide the
same guarantee — and two did not:

* **the routed-expert gather destination** (owned by the graph allocator), and
* **the decode-cache slot arena** (owned by the cache).

`NaN * 0 = NaN`, so a single stale/uninitialized head poisons the whole tile → repeated `/`.

## Hole A — the cache arena (the abort corruption)

`alloc_table_locked` (block 13, `moe-expert-cache.cu`):

```c
cudaMalloc(&t.arena, (size_t) arena_slots * t.expert_bytes);   // no zero, no tail pad
```

`cudaMalloc` does not zero.  An **empty (never-filled) slot's head** was uninitialized memory — usually
NaN — and the bytes **past the last slot** were out of the allocation.  The r30 one-time zero guarded
only the *gather* destination, never the arena.  This is what the abort reproduced: an aborted stream
perturbs the slot population enough to expose an empty slot's head, and it is **persistent** for the
process because the arena lives on.

Fix (one-time, no per-token cost):

```c
const size_t head_pad   = t.expert_bytes < 512 ? t.expert_bytes : 512;
const size_t arena_size = (size_t) arena_slots * t.expert_bytes + head_pad;   // + tail
cudaMalloc(&t.arena, arena_size);
cudaMemset(t.arena, 0, arena_size);
```

Every slot head starts finite (zero); filled slots are overwritten with real expert data; evicted slots
keep finite bytes.  Confirmed live: a long prompt → abort reasoning mid-stream → second prompt now
**survives multiple interruptions**.

## Hole B — the gather destination (the prefill-benchmark corruption)

The gather's one-time head zero is keyed on `(input_cpy->data, expert_bytes)` and never re-arms; the
graph allocator re-uses the destination region between ubatches, so the zero does not survive a
**multi-ubatch prefill**.  NaN MoE routing then *skips expert work*, so the gather benchmarked
2–4× faster while doing less.

Evidence (qwen4exp IQ4_NL, 1 R9700, `-sm layer -ncmoe 48 -fa 1 --lazy-mode off --load-mode none
-t 8 -b/ub 2048 -p 8192 -n 0`, `MIB=12288`):

| build | gather | pp8192 |
|---|---|---:|
| pre-r31 | on (r30 once-only zero) | 2674 |
| **r31** (arena fixed) | on (`GGML_SCHED_DEVGATHER=1`) | **2683** |
| pre-r31 | off (staging) | 655 |
| **r31** (arena fixed) | off (default, staging) | **657** |

The arena fix does **not** move the gather number, so Hole B is independent.  A bandwidth check proves
it: the prefill reads ~68 GB from host RAM per ubatch, so 2683–3086 t/s at `-ub 8192` needs ~25 GB/s —
above this box's PCIe — while 657/1396 t/s needs ~11.6 GB/s.

Fix: **disable the gather by default** (`ggml-backend.cpp`, `sched->devgather_enabled`); the
staging/host-copy path copies the guard pad *every* pass into a backend-owned ring the graph allocator
cannot clobber.  `GGML_SCHED_DEVGATHER=1` re-enables it for A/B.

## Re-established baseline (r31 default = staging/host path)

qwen4exp IQ4_NL (100 GB), 1 R9700 (32 GiB), `-sm layer -ncmoe 48 -fa 1 --lazy-mode off --load-mode none`
`-t 8 MIB=12288`:

| `-b/-ub` | pp8192 | tg1024 |
|---:|---:|---:|
| 2048 | 659 | 39.7 |
| 4096 | 1018 | 40.3 |
| 8192 | 1396 | 35.9 |

2-GPU `-sm tensor -ctk q8_0 -ctv q8_0 MIB=8192 -ub 2048 -p 1024`: **pp1024 562**.

Inputs are PCIe-bound (~11–12 GB/s effective).  The docs' old "staging 657–1403" reproduces exactly,
confirming the baseline.  **The gather's old 2650–3060 was never real.**

## What was tried and rejected (kept as evidence)

* **r30's one-time head zero (once per buffer)** — insufficient: does not survive a multi-ubatch
  prefill.  Superseded by the arena `memset` (Hole A) and by defaulting the gather off (Hole B).
* **In-kernel MMQ guard / in-gather pad copy / persistent gather destination** — each measured
  682–1532, i.e. *the correct speed*; the "4× cost of the guard" was the difference between correct and
  NaN work, not a guard cost.  (The persistent destination was additionally broken under `-sm tensor`,
  where mutating the simple tensor's `data` does not reach the op.)
* **Single-buffer persistent destination / shared across tables** — correct on 1 GPU but not under the
  meta path; not needed once staging is the default.

## Follow-ups

1. **Re-measure on a PCIe 5.0 ×16 link.**  Two of the three R9700s will be removed so the remaining card
   runs at ×16 instead of ×4; the prefill is PCIe-bound, so this should move the ~1396 number
   materially.  Re-baseline before any optimisation.
2. **Then the real gap.**  vLLM reaches ~3500 t/s on one card with a wide ubatch; the honest baseline is
   ~1400 at `-ub 8192`.  Prime suspect: the per-ubatch expert re-upload (only residency removes it).
3. **Re-evaluate the gather** after the arena fix and the ×16 re-measure: if its destination is made
   persistent/never-reused *and* the `-sm tensor` redirect is fixed (via the `moe_cache_get_table`
   consumer seam), and it is genuinely correct, it may pull ahead of staging.  Until then it is off.
4. **Widen the gate.**  `scripts/gate-qwen4exp-quant-coherence.sh` runs a short prompt and cannot see a
   multi-ubatch prefill corruption (the once-only zero survives one ubatch).  Add a multi-ubatch /
   post-abort case and a **bandwidth-plausibility assertion** (compare against the bytes actually
   copied) so an inflated "win" can never be recorded as real again.
