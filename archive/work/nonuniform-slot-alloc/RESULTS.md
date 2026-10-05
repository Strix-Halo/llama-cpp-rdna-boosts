# Non-uniform per-layer slot allocation in the MoE expert cache — NEGATIVE (2026-10-06)

**Status:** closed negative, no code and no build. Recorded here so the idea is not re-derived.

**Hypothesis.** The delivery's decode-band expert cache (`MOE_EXPERT_CACHE_MIB`) allocates a
**uniform** slot count per layer (`layer_slots[l] = budget / total_expert_bytes`, see
`alloc_all_locked()` in `ggml/src/ggml-cuda/moe-expert-cache.cu`). Per-layer routing concentration
varies a lot (see below), so a **non-uniform, hotness-ranked per-layer slot budget** should raise the
aggregate hit rate at a fixed VRAM budget — the same lesson Strata found one level up ("a global pool
measures ~3 % at the same budget; per-layer ranges turn it into 21–70 %").

**Result.** At every arena size worth using, the optimal non-uniform allocation is worth **≤1 %
in-sample and ~0 held-out**. It only moves the needle in the sub-1 GiB arena regime, where the cache is
barely useful anyway. The delivery's uniform per-layer ranges already capture the real (Strata) win;
skewing the counts on top adds nothing.

## Method

The only per-layer expert-frequency data in the repo is the moe-expert-cache campaign profiler
(`archive/work/moe-expert-cache/prof-*.csv`, decode band `n_tokens <= 8`, Qwen3.6-35B-A3B Q8_0,
2000-token greedy generations on five prompts). Format: `expert,<layer>,<expert>,<reach_count>`.

For a fixed total slot budget `K`, the optimal per-layer allocation of a separable concave coverage
objective `sum_l sum_{i<S_l} count_l[i]` under `sum_l S_l <= K` is obtained by greedy water-filling:
repeatedly give the next slot to the layer whose next-ranked expert has the highest reach count. That
greedy is optimal for this objective. Compare it to the uniform `K / n_layers`.

Reproduce with `waterfill.py` (same directory).

`h(S=64)` per layer (top-64 experts' share of that layer's touches) confirms the concentration spread
is real and **transfers at the layer level** (L0–L4 diffuse in every prompt; L20 and the L25–L31 band
concentrated in every prompt) — unlike the per-expert book, which the campaign showed does not transfer
(held-out 0.24–0.38, and it collapses mid-run 0.83 -> 0.55):

| layer | prose | mixed | python | reasoning | recall |
|---|---:|---:|---:|---:|---:|
| L0 | 0.51 | 0.46 | 0.50 | 0.51 | 0.51 |
| L2 | 0.49 | 0.41 | 0.44 | 0.55 | 0.47 |
| L20 | 0.85 | 0.85 | 0.86 | 0.92 | — |
| L26 | 0.84 | — | — | 0.87 | 0.94 |
| L31 | 0.80 | 0.81 | 0.79 | — | — |

Per-layer *touch counts* are uniform (every layer ~5370 touches), so only concentration differs.

## Result table

| slots/layer (arena) | in-sample optimal gain vs uniform | held-out (alloc from `prose`, scored on the other 4) |
|---|---:|---:|
| 4 (~0.5 GiB) | +5.3 … +9.2 % | +1.9 … +4.0 % |
| 8 (~1 GiB) | +2.8 … +5.2 % | +1.9 … +3.3 % |
| 16 (~2 GiB) | +0.9 … +2.7 % | +0.2 … +1.4 % |
| **32 (~4 GiB)** | **+0.2 … +0.8 %** | **−0.3 … +0.1 %** |
| **64 (~8 GiB)** | **+0.2 … +0.7 %** | **−0.1 … 0.0 %** |
| 96 | +0.5 … +1.8 % | +0.5 … +1.0 % |
| 128 | +0.9 … +2.0 % | +0.7 … +1.4 % |

The optimal allocation never zeroes a layer and is only mildly skewed (at 64/layer the range is 44–93).
Aggregate coverage at uniform 64/layer is 0.72, matching the campaign's in-sample `h(64)` of 0.63–0.81,
so the model is faithful.

## Why

Coverage is **concave** in `S`: the marginal value of a slot is the reach count of the next-ranked
expert, and that marginal is *similar across layers* — concentrated layers have already saturated their
heads, diffuse layers keep a steep tail, and water-filling equalizes them back to roughly uniform. The
layer concentration spread is real but maps to almost no capacity-planning advantage.

## What is NOT implied

- This does **not** contradict the per-layer *ranges* win (Strata). Uniform per-layer ranges are the
  big step; non-uniform counts on top are a wash.
- It does not bear on the **dynamic** policy (LFRU admission/eviction), which is what handles the
  non-transferable per-expert book. Only the static per-layer *budget shape* is measured here.
- Prefill is untouched: the decode-band profiler excludes it precisely because a prefill ubatch routes
  nearly every expert, so residency has no small hot subset to hold there.

## Conclusion

Do not build non-uniform per-layer slot allocation. The residency headroom is in the policy / cold-path
cost and in prefill volume + overlap, not in the budget shape.
