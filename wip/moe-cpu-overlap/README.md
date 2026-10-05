# Overlapping the CPU expert pool with the GPU (the Strata shape)

**Status: OPEN / scoping (2026-10-05).** Maintainer: yes, keen — Strata demonstrates it works. The
previous statement that it "cannot be done effectively" was about the shape we tried (an interleaved
CPU branch inside llama.cpp's serial scheduler). Strata does not use that shape; it has a custom
pipeline that runs the CPU miss compute *beside* the GPU resident compute. This note scopes doing that
in the fork.

Related: `wip/moe-cache-autosize/` (the other half), `archive/work/moe-expert-cache/WORKLOG.md`
§"CPU-COMPUTES-THE-MISSES ARM" (the prior negative result), `archive/work/tensor-split-expert-split/README.md`
§26 (the Strata read), `~Strata` `src/core/expert_source.cpp`, `src/program/generate.cpp`,
`src/kernels/cpu/pool.cpp`.

## The prior result (why the first attempt failed)

Built, correct, and **negative above a ~1.2 GiB arena**. Q4_K_M, 1 GPU, `-ncmoe 99 -fa 1 -sm layer -t 8`,
`tg1024`:

| arena | `h` | no split | C=2 | C=4 | C=6 |
|---|---:|---:|---:|---:|---:|
| 8192 MiB | .95 | **58.27** | 40.57 | 36.96 | - |
| 2048 MiB | .73 | **39.10** | 36.04 | 35.53 | - |
| 1024 MiB | .55 | 31.17 | 30.68 | **32.00** | 29.38 |
| 512 MiB | .35 | 26.37 | - | **28.55** (+8.3 %) | - |

Mechanism (the finding worth keeping): the loss is **identical at `-t 1` and `-t 8`**, so it is not
CPU compute and not the thread-pool wake. It is that the interleaved CPU branch is **serialised with
the GPU splits**: ~15 ops x 40 layers = ~600 scheduler dispatches plus ~3 cross-backend copies per
layer per token = ~8-10 ms/token, an order of magnitude larger than the CPU compute it enables. The
modelled advantage was real (UVA PCIe ~79 us/cold expert vs CPU ~27 us/expert; ~2.9x at the time,
1.5-2.2x when re-measured on r19), and the routing is genuinely concentrated (profile `h_expert ≈
0.645` at 4,105 slots, ten-fold leave-one-out). The campaign's own verdict:
> "The real prize is **overlap** - cached runs leave ~14 of 16 cores idle (measured: cache 1.90 cores
> busy vs 8.94 for the delivered CPU MoE), and concurrent execution would give +17 %/+40 % instead of
> +12 %/+29 %. That is a scheduler change."

## What Strata actually does (current source, 2026-10-04)

* Experts pinned in host RAM; a **VRAM `ExpertCache`** (`src/core/expert_cache.hpp`) holds the hot
  `(layer, expert)` slots (static profile + adaptive, per-layer ranges).
* `ExpertDispatch` splits **by router index**: of the K routed experts, the resident ones run on the
  **GPU**, the rest on the **CPU** (`moe_combine` sums by router index, so the halves rejoin without a
  scatter).
* The engine's own pipeline (`session_loop` -> `ExpertPool::run`) runs the CPU pool **at the same time
  as the GPU works on the cached experts** (`docs/DETAILS.md` "How it works"). `--shared-late` is an
  A/B for whether the shared expert runs after the CPU pool or *overlapped with it* — i.e. the overlap
  is explicit and tuned.
* Their measured context: CPU expert pool 663.6 MB/token at ~40 GB/s = 16.2 ms of a ~53 ms token,
  with only 1.055 ms of 19.035 hidden by the old pipeline (96 % exposed) — their fix was a custom
  overlapped loop plus the cache, not a faster copy.

## Design space for the fork

The moat is that llama.cpp's `ggml_backend_sched_compute_splits` runs splits one at a time. Options,
cheapest first:

1. **Fused multi-backend MoE op.** One graph node owns the whole MoE: it launches the GPU kernel for
   the resident experts and, in the same node, has a host-side CPU worker pool compute the cold experts
   on a side thread, then joins and combines. Avoids the ~600 dispatches entirely (one node per layer).
   Requires a new op + a CPU implementation reachable from the CUDA backend, and a way to hold the
   router ids host-side. This is closest to Strata's shape.
2. **Dedicated CPU miss pool + per-layer ring.** Keep the graph as-is, but instead of a CPU *split* per
   layer, enqueue the cold work to a persistent pool thread that runs on pinned host memory while the
   GPU graph runs, with an event/ring handoff. Needs the scheduler to expose "don't wait for this
   backend" for that op.
3. **Split-graph concurrency.** Split the decode graph into a GPU subgraph and a CPU subgraph with
   `GGML_SCHED_PRIO_HIGH`/`LOW`, and teach the scheduler to overlap a CPU split with a GPU split on
   different sides of a dependency. Highest leverage if it works, but touches core scheduling, the
   CUDA-graph capture path (block 11 keeps verify widths captured), and the block-06 events.
4. **Hybrid of the delivered paths.** Cache-on currently sends **all** cold misses to the GPU over UVA;
   cache-off computes all experts on the CPU. A partial split (e.g. CPU takes the experts whose blobs
   are cheapest for it) is option 1/2 with a policy.

**Do not** re-build the old interleaved per-op CPU branch (option 2 of the prior campaign): measured
negative, and its cost is structural.

## Why it is hard here (and not for Strata)

* **Scheduling**: the ~600 dispatches are llama.cpp's; a custom engine has none.
* **Purity**: the delivery's `W = 1..8` width purity and the `none == draft-mtp` guarantee; mixing CPU
  and GPU arithmetic for the same layer is not bit-identical (the prior arm documented this), so the
  gate has to be acceptance + long-horizon coherence, not a hash.
* **CUDA graphs**: decode/verify widths are captured (block 11); a side host thread that must
  synchronise per layer can break capture or serialise.
* **Cross-backend consistency**: the delivery keeps CUDA/HIP/MUSA/SYCL predicates consistent; a new op
  needs a CPU path on every backend that can host experts.
* **The win is bounded by residency.** With a 20 GiB arena on a 32 GiB card we already hold ~43 % of
  the experts, so the overlap only helps the ~57 % cold fraction. The bigger prize is on small cards
  (Strata's 12 GB / 62 t/s case, <10 % resident), which are *less* interesting to this repo's RDNA4
  validation boxes. Frame the campaign accordingly: a 32 GiB card gains less than a 12 GiB card.

## Milestones

* **O0 — Re-measure the ceiling.** On gfx1201 + qwen4exp IQ3_XXS: how much of a 52 t/s token is GPU,
  how much is the UVA cold read, how many cores are idle (`STRATA_DECODE_TIMING`-style split; our
  equivalent is `GGML_CUDA_*` profiling / `nsys`). Decide whether the cold fraction is even the
  bottleneck at 43 % residency. **Gate:** if the cold read is <20 % of the token, park this.
* **O1 — Pick the shape** (1 vs 3) with a written design and a correctness argument. Prototype on one
  dense-ish MoE model first if possible.
* **O2 — Prototype** on a single GPU, `-sm layer`, default-off env kill switch.
* **O3 — Gates**: acceptance, long-horizon coherence at 32K/128K q8_0, no `////`, the
  `-sm layer`/`-sm tensor` matrix, CUDA-graph capture intact, and the `-ncmoe 0` oracles unchanged.
* **O4 — Decide**: promote / park. Promotion needs a same-change kill switch and a re-validated
  combination (WIP promotion rule).

## Open questions

1. Can `ggml_backend_sched` overlap a CPU compute with a GPU compute at all today? (Read
   `ggml_backend_sched_compute_splits` and the `GGML_SCHED_PRIO_*` handling.)
2. Is a fused MoE op acceptable to the delivery's purity rules, or does the mix force it into a WIP-only
   env-gated mode?
3. Does the overlap help prefill (no — prefill uses every expert) or only decode?
4. Upstream applicability: a generic "CPU computes MoE misses while the GPU runs" op is broadly useful
   (any `-ncmoe`), so an `upstream/` copy is likely warranted.
