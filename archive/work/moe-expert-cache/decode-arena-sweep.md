# Decode arena-size sweep — the baseline drop-off curve (2026-09-28, session 8)

This is the **decode-side analogue of the prefill `-ncmoe` gentle-drop-off table** posted in
[discussion #54](https://github.com/stew675/llama-cpp-rdna-boosts/discussions/54).  It is the baseline
the next item (the device-side remap, `README.md` item 3) must flatten.  It is **not** a final gate
record: the granularity below is what a development pass needs to see the shape.

## Method

* Model: `Qwen3.6-35B-A3B UD-Q4_K_M` (21.1 GiB, 256 experts, top-8, 40 MoE layers), 2×R9700 (gfx1201).
* `HIP_VISIBLE_DEVICES=0,1`, `-ncmoe 99 -ngl 99 -fa 1 -sm tensor -t 8`, fusions on (default).
* Depth-0 table: `llama-bench -p 0 -n 1024 -r 6`; depth-16384 table: `-n 512 -d 16384 -r 4`.
* **Warm mean** excludes rep 1, which pays the one-time cold fill (`llama-bench -o jsonl` `samples_ts`;
  the first sample is listed too so the fill penalty is visible).  This is the "be careful of the
  initial cold-fill penalty" rule.
* x-axis: `MOE_EXPERT_CACHE_MIB` (per-device arena budget) and the derived resident slots/table
  (`h = slots/256`).  `MIB ≈ 36.6 × slots` on this model; the arena is clamped to free VRAM − 1 GiB
  reserve, so the exact slots can differ by ±1.

## Depth 0 — warm `tg1024`

| MIB | slots/256 | h | warm `tg1024` | reps (1..6) |
|---:|---:|---:|---:|---|
| 1024 | 28 | 0.109 | 42.9 | 41.7, 44.4, 44.3, 41.7, 41.5, 42.5 |
| 2048 | 56 | 0.219 | 55.4 | 49.1, 56.3, 55.9, 54.9, 55.6, 54.1 |
| 3072 | 84 | 0.328 | 62.3 | 51.6, 61.5, 62.5, 62.2, 62.7, 62.8 |
| 4096 | 112 | 0.438 | 66.1 | 52.2, 65.2, 66.4, 66.3, 66.3, 66.4 |
| 5120 | 140 | 0.547 | 68.5 | 52.3, 68.5, 68.1, 68.3, 68.9, 68.8 |
| 6144 | 168 | 0.656 | 69.8 | 50.8, 69.7, 69.9, 69.6, 69.7, 70.0 |
| 7168 | 196 | 0.766 | 69.8 | 50.5, 68.3, 69.7, 70.4, 70.4, 70.4 |
| 8192 | 224 | 0.875 | 69.7 | 50.9, 68.8, 69.6, 69.7, 69.9, 70.4 |
| 9216 | 252 | 0.984 | 69.8 | 50.9, 68.3, 69.8, 69.9, 70.4, 70.4 |
| **9280** | **254** | **0.992** | **69.4** | 50.7, 68.9, 69.5, 69.8 |
| **9344** | **256** | **1.000** | **94.1** | 32.4, 93.8, 94.2, 94.3 |
| 10240 | 256 | 1.000 | 93.9 | 32.4, 93.5, 93.9, 94.0, 94.0, 94.0 |

Endpoints (same harness, `-ncmoe` only): **`-ncmoe 0`** (everything on device) ≈ **96** t/s warm;
**cache off** (`-ncmoe 99`, delivered CPU MoE, no `MIB`) ≈ **32.9** t/s warm (28.6 with fusions off).

## Depth 16384 — warm `tg512 @ d16384`

| config | warm `tg512 @ d16384` | reps (1..4) |
|---|---:|---|
| cache off (`-ncmoe 99`, CPU MoE) | 31.4 | 30.5, 31.3, 31.5, 31.3 |
| MIB=2048 (h=0.22) | 52.4 | 43.1, 52.7, 52.2, 52.2 |
| MIB=4096 (h=0.44) | 62.4 | 42.9, 61.7, 62.6, 62.9 |
| MIB=6144 (h=0.66) | 64.1 | 42.9, 62.7, 65.0, 64.8 |
| MIB=8192 (h=0.88) | 63.9 | 43.0, 60.8, 65.7, 65.1 |
| MIB=9216 (h=0.98) | 64.1 | 43.7, 62.9, 65.0, 64.5 |
| MIB=10240 (**identity, h=1**) | 87.3 | 19.4, 87.3, 87.3, 87.4 |
| `-ncmoe 0` (oracle) | 89.9 | 87.2, 89.4, 90.1, 90.1 |

## The shape, and the two separate effects

1. **A gentle rise below h≈0.5**: 42.9 → 55.4 → 62.3 → 66.1 as 28→112 slots.  This is the part that
   already behaves like the prefill table: more resident experts, fewer misses, higher throughput.
2. **A flat plateau at ≈70 from h≈0.66 to h≈0.99**: 69.8 … 69.4 while 168→254 slots.  Adding resident
   experts buys **nothing** here.
3. **A cliff at exactly h=1 (slots 254→256, +64 MiB)**: 69.4 → 94.1.  The last two experts are worth
   +36 %.

The plateau and the cliff are the identity fast path's all-or-nothing threshold: below h=1 the decode
path takes the scheduler's per-layer host round-trip; at h=1 the identity path removes it.  The
round-trip is per layer per token, so it is a **constant** cost once you are off the identity path —
which is why the plateau is flat rather than gently rising.

## It is NOT the fusions (verified)

The session-7 record already attributed the h<1 gap to the round-trip, but the plateau number ≈ the
fully-cached-**fusions-off** number, so that was re-checked directly:

| config | fusions ON | fusions OFF | fusion gain |
|---|---:|---:|---:|
| MIB=8192 (h=0.875) | 69.2 | 56.5 | +12.7 |
| MIB=10240 (identity) | 94.0 | 71.9 | +22.1 |
| cache off (CPU MoE) | 32.9 | 28.6 | +4.3 |

So the cache-band fusions **do** fire at h<1 (confirmed with `GGML_CUDA_FUSE_LOG`: at MIB=8192 a 20-token
run has 320 `ffn_moe_gate` gate+up+GLU, 320 `ffn_moe_down` folds and 160 weighted folds — **identical** to
MIB=10240).  The coincidence `h<1 fusions-on (69.2) ≈ h=1 fusions-off (71.9)` is exactly that: the
round-trip cost at h<1 is roughly equal to the fusion benefit at h=1.

## The round-trip, quantified

`GGML_SCHED_SYNCDBG=1`, 20-token decode + prefill (same prompt):

| config | `SCHEDSYNC calls` | `get_async` | `input_loop` |
|---|---:|---:|---:|
| MIB=8192 (h=0.875) | 4701 | 1906 | 1046 |
| MIB=10240 (identity) | 2184 | **228** | 1046 |

The identity path removes ~1678 of the 1906 `get_async` calls (the per-layer ids readback) and halves
the syncs.  That is the whole plateau/cliff.

## What the next item must produce

With the device-side remap (item 3) removing the ids readback at **every** h, the expected shape is a
monotone rise from ≈42 (h≈0.1) through ≈70 (h≈0.5) toward the identity number, with no plateau and no
cliff — i.e. the decode analogue of the prefill table.  The acceptance gate is this sweep, warm reps,
depth 0 **and** depth 16384, with the h→1 endpoint within a few percent of the identity path.

---

## Postscript 2 (2026-09-29, session 10): the devmap cliff is halved, and the residual is management overhead

Session 10 found why the item-3 device-side remap lost: the deferred promotion did a **synchronous
per-table D2H** of the used-list on 240 tables/token (`25.1 us/call`, **81 %** of the 6.2 ms/token pass -
the session-9 "copies were never the cost" note was wrong).  A **double-buffered pipelined readback**
(pinned `cudaMemcpyAsync`, policy applied to the previous token's buffer) plus a **slot-dirty skip** took
the pass to **0.43 ms/token** and produced the new curve:

| `MIB` | slots/256 | h | old (eager) | **new (devmap)** |
|---:|---:|---:|---:|---:|
| 1024 | 28 | 0.11 | 42.9 | **45.5** |
| 2048 | 56 | 0.22 | 55.4 | **62.2** |
| 3072 | 84 | 0.33 | 62.3 | **74.5** |
| 4096 | 112 | 0.44 | 66.1 | **80.8** |
| 6144 | 168 | 0.66 | 69.8 | **86.1** |
| 8192 | 224 | 0.88 | 69.7 | **85.3** |
| 9216 | 252 | 0.98 | 69.8 | **85.6** |
| **9344** | **256** | **1.00** | **94.1** | **94.1** |

Depth 16384: `MIB=2048` **60.9** (old 52.4), `MIB=4096` **75.0** (old 62.4), `MIB=8192` **76.6** (old 63.9),
identity **87.5** (old 87.3).  The plateau is gone and the cliff
is **85.6 -> 94.1 (+10 %)**, down from 69.4 -> 94.1 (+36 %); h~0.98 is at **91 % of identity** (was 74 %).

**Attribution of the remaining 1.14 ms/token at h=1** (same-arena `MOE_EXPERT_CACHE_FORCE_DEVMAP` A/B,
identity 94.0 vs forced-devmap 84.9): host promotion **0.44 ms** (d2h 0.16 / host LFRU 0.20 / slot 0.03),
**240 per-table `moe_cache_build_remap_kernel` dispatches ~0.55 ms** (`rocprofv3`: exactly 240/token,
1.57 us each = 0.375 ms of raw GPU + graph-node gaps), expert fill H2D **0.13 ms** (3.85 MiB/token at the
aggregate ~30 GB/s).  So the residual is **management overhead, not PCIe**: at `MIB=8192` over 4095 tokens
the cache has **99.25 % access hit and only 234 evictions total** (4.4 MiB/token); PCIe only becomes a real
limiter below h~0.9 (`MIB=2048`: 15.6 evictions/token, 84.6 MiB/token ≈ 2.96 ms of the ~6 ms gap).  The next
target is the remap kernels - see `README.md` Item 3b-II.

## Postscript (2026-09-29, session 9): the pre-pipelining device-side remap did NOT flatten the curve

The item-3 device-side remap is now byte-correct (`WORKLOG.md` 2026-09-29; branch `wip-moe-devmap-v2`,
`exp10-…-devmap.patch`), but its warm `tg1024` curve is **monotonically worse than this baseline at every
`MIB`** (see the table in the WORKLOG entry), so this sweep still stands as the delivered shape.  The
cliff is open; the deferred host promotion costs more than the per-layer readback it removes, and closing
it needs a device-side admission (LRU) policy rather than a host promotion pass.
