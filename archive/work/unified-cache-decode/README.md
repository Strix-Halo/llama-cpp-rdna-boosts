# Unified-cache **decode** fully-masked KV-group skip (issue #48 follow-up)

Status: **CLOSED (2026-09-26) — measured, not worth pursuing; archived under `archive/work/unified-cache-decode/`.**
Option (b) was implemented, validated and measured; it is break-even because the per-layer prepass costs
about what the skip saves.  The decode penalty is ~1 %, not the large one §1 assumed.  See §0.

Parent work (already shipped): release `v16-84e76d8a2-r10`, block-15 amendment
[issue #48](https://github.com/stew675/llama-cpp-rdna-boosts/issues/48) —
`WORKLOG.md` 2026-09-26 (r10), `patches/README.md`, `GREEDY-PURITY.md` §37.

## 0. Session findings (2026-09-26) — read this first

**The unified-cache decode penalty is ~1 %, not the large one §1 assumed.**  The mechanism is real —
unified decode passes `K->ne[1]` = the whole shared cache with `Q->ne[3] = 1`, vs a per-stream
`K->ne[1]` when not unified — but the FA is a small fraction of the decode step (the weights dominate
at `Q->ne[1] <= 8`), so the extra masked cells cost little:

| config | decode TG `-no-kvu` | decode TG `-kvu` | penalty |
|---|---|---|---|
| 4B 1-GPU, `llama-batched-bench` npp 4096 B=4 | 249.00 | 242.00 | −2.8 % |
| 4B 1-GPU, npp 32768 B=2 | 124.29 | 123.62 | −0.5 % |
| 4B 1-GPU, npp 65536 B=2 (131k shared) | 102.54 | 101.04 | −1.5 % |
| 27B 2-GPU `-sm tensor`, npp 4096 B=2 | 52.57 | 52.44 | −0.25 % |
| 4B 1-GPU server `-np 2 --kv-unified`, 2×15.7k, 128 tok | 51.43 | 50.94 | −1.0 % |

**Option (b) was implemented and measured: break-even.**  Removing the `Q->ne[1] > 8` gate and
threading `kq_blocks_tile` through the WMMA band was **correct** (`test-backend-ops -o FLASH_ATTN_EXT`
**6354/6354** on both MMA and forced-tile; the decode skip fires on the tile `parallel_blocks` path and
the WMMA band), but the win is nil (27B 2-GPU `-sm tensor`, `-kvu`):

| npp | B | skip=0 | skip=1 |
|---|---|---|---|
| 4096 | 1 (no foreign cells) | 25.16 | 24.92 (−0.95 %) |
| 4096 | 2 | 52.18 | 52.13 |
| 16384 | 1 | 24.50 | 24.35 (−0.6 %) |
| 16384 | 2 | 48.71 | 48.83 (+0.25 %) |

The B=1 rows isolate the **prepass overhead** (no foreign cells → nothing skippable): the per-layer
memset + prepass launch costs ~0.6-1 % per step, which cancels the ~1 % FA saving.  **Option (a)**
(per-slot ownership cached in `llama_kv_cache`, computed once per ubatch instead of per layer) would
remove that overhead and net the ~1 % FA saving — still marginal for the plumbing.

**Recommendation:** close the decode follow-up as *measured, not worth pursuing*.  Revisit only if a
user reports a large decode regression with many long-lived slots on a small model at very deep
context, where the FA fraction is larger.  The prefill case was different because `Q->ne[1]` is in the
hundreds/thousands there, so the FA dominates and the waste is 20-45 %; r10 is unaffected and is the
real win.

### `-sm tensor` 2-device validation (the user's explicit ask)

* r10 is **pure** under `-sm tensor` 2-GPU 27B: single-sequence greedy text `7a7430617465` identical
  with `GGML_CUDA_FA_MASK_SKIP=0/1`.  (An apparent difference was only the CLI spinner and the timing
  line — always hash with `scripts/extract-generated.py`, never `diff` raw stdout.)
* The derived tensors are host-resident (`GGML_ASSERT(ggml_backend_buffer_is_host(...))` in
  `set_input_kq_derived`), so the launcher's host-side `tok_lo`/`tok_hi` read is valid under tensor
  split.
* Concurrent serving is **nondeterministic run-to-run even with the skip disabled, and even on a
  single GPU** (no `-sm tensor`, no all-reduce): the same request yields a different greedy token when
  batched with a different neighbour, because the batch composition changes the kernel/reduction order.
  Sequential requests are fully deterministic.  A concurrent-server text hash is therefore not a purity
  gate; use `test-backend-ops` (bit-identical vs the CPU reference) for the packed path.  This is
  inherent to continuous batching and not introduced by the delivery.
* The band's `flash_attn_stream_k_fixup_uniform` and the `parallel_blocks` `flash_attn_combine_results`
  both treat a zero-rowsum / `-FLT_MAX/2` partial as an exact no-op, so a skipped band block is safe.

---

The rest of this document is the original handoff and stays for reference / re-evaluation.


## 1. The remaining surface

The r10 fix skips **fully-masked interior KV groups during prefill**. A unified KV cache
(`--kv-unified`) also makes every **decode/verify** step attend over the whole shared range, i.e.
the other active slots' cells (all `-INF` for the current sequence) plus the empty tail. Those groups
are still processed. The cost is `O(n_kv)` per layer per step for the query row(s), so it grows with
the *other* slots' context, and it is the decode-side analogue of the ~45 % prefill drop the reporter
measured.

The reporter's issue was TTFT/prefill and that is fixed; **decode TTG with several long-lived slots is
the open half.**

## 2. Current state (r10) — what is and is not covered

| path | shape | r10 coverage |
|---|---|---|
| derived mask | single-sequence **prefill** (`n_tokens > 8`) | **skip on** (host batch-wide bitmap) |
| packed mask | multi-sequence **prefill** (`Q->ne[1] > 8`) | **skip on** (GPU per-tile prepass) |
| packed mask | **decode/verify** `Q->ne[1] <= 8` (band) | **skip off** — deliberately |
| derived mask | verify `n_q = 9..16` (`n_max 8..15`) | **skip on** (it is the prefill kernel) |

Kill-switch: `GGML_CUDA_FA_MASK_SKIP=0` (default on).

The decode/verify gate is explicit in `launch_fattn` (`ggml/src/ggml-cuda/fattn-common.cuh`):

```c
const bool kq_block_skip_packed =
    kq_skip_ok && cell_pos == nullptr && mask != nullptr && Q->ne[1] > 8 && ...;
```

and `kq_mask_derivable()` (`src/llama-kv-cache.cpp`) rejects `ubatch.n_tokens <= 8` with the comment
"decode has to stay on the packed path" — a *policy*, not an intrinsic limit.

## 3. Why it was left out of r10 (read before changing the gate)

1. **The band is the width-purity anchor.** Decode/verify runs the round-robin KV split
   (`kb0_step = P`, `gridDim.y = P`), which is what makes `W = 1..8` agree bit-wise (issue #25,
   `GREEDY-PURITY.md` §§10-11). Skipping a fully-masked group is exact, so in principle it cannot move
   `W = 1..8`; but the band writes a **neutral partial** for a block that owns no iteration, and the
   `flash_attn_stream_k_fixup_uniform` combine reads it. A skipped block must land in exactly the same
   neutral-partial state as an owned-none block. **Verify that before enabling the band skip.**
2. **Prepass cost shape changes.** For `n_q = 1` a packed-mask prepass is `O(n_kv)` — the same order as
   the FA itself — so a per-step, per-layer prepass is a real tax. For `n_q = 2..8` it is `~1/head` of
   the FA. The derived host bitmap is `O(n_kv)` too but runs on the CPU; the GPU prepass runs
   `n_layers` times per step.
3. **`kq_mask_derivable()` gates on `n_tokens <= 8`** for a reason that no longer fully holds: the
   original comment predates the r10 bitmap and says decode must keep the packed mask for purity. That
   decision needs re-deriving: with a bitmap, the derived form could serve decode too.

## 4. Design options (in rough order of payoff)

**(a) Cache the per-slot group visibility in `llama_kv_cache` (best).**  A group is invisible to
sequence `s` iff it contains no live cell of `s` (the causal/window part is then a small per-token
correction). That ownership only changes when the cache is mutated (`seq_rm`, `seq_cp`, cell
allocation/reuse), *not* every step. Compute `own[stream][group]` once per mutation, store it next to
the KV tensor, and have the launcher combine it with the causal window and hand it to the kernel as
the existing `kq_blocks`. This removes the per-step `O(n_kv)` scan entirely and is the only option that
scales to many slots. Needs: a new tiny tensor/op input (the r10 `kq_blocks` op argument already
exists, so the kernel side is done), an invalidation hook on every cache mutation, and a correctness
argument for SWA/ALiBi/M-RoPE.

**(b) Extend the r10 packed prepass to the band.**  Drop the `Q->ne[1] > 8` gate and make
`flash_attn_mask_to_KV_blocks` produce a per-(Q stream, query tile, group) bitmap for decode too. The
band's query tile is `ncols1` (often 1), so the prepass is `O(n_q * n_kv)`. Measure the prepass cost
against the FA saving; likely a win at depth/with other slots, a small loss when alone.

**(c) Lower the derived `n_tokens <= 8` gate.**  Then a single-sequence decode computes the batch-wide
bitmap on the host from `cell_pos` (no GPU prepass, no packed mask). But the derived form is
implemented by the **MMA and tile** kernels only (not vec), and the CPU reference / op-test coverage
must follow. It also interacts with the band: the derived path always takes the prefill-shaped kernel
selection, which for `n_q <= 8` is not the band. Check the chooser (`ggml_cuda_get_best_fattn_kernel`)
and whether the derived bitmap can be consumed by the band path at all.

The likely landing zone is **(a)**, with **(b)** as the quick A/B to size the prize.

## 5. Where the code is

* `ggml/src/ggml-cuda/fattn-common.cuh`
  * `ggml_cuda_fattn_kq_block_skip_enabled()` — the `GGML_CUDA_FA_MASK_SKIP` gate.
  * `fattn_kq_group_masked()` — the per-block bit test.
  * `flash_attn_kq_derived_blocks` — host-derived batch-wide bitmap.
  * `flash_attn_mask_to_KV_blocks` — GPU prepass over the packed mask, per (stream, tile, group).
  * `launch_fattn` — the `kq_skip_ok` / `kq_block_skip_derived` / `kq_block_skip_packed` block and the
    `kq_blocks` kernel argument. **The `Q->ne[1] > 8` gate lives here.**
* `ggml/src/ggml-cuda/fattn-mma-f16.cuh` — `flash_attn_ext_f16` + `flash_attn_ext_f16_process_tile`
  (`kq_blocks_tile`, the `if (kq_blocks != nullptr)` skip loops; the band path passes `nullptr`).
* `ggml/src/ggml-cuda/fattn-tile.cuh` — the same skip in the tile KV loops.
* `ggml/src/ggml-cuda/fattn-vec.cuh` — signature only; the vec kernel never skips (not selected on AMD).
* `src/llama-kv-cache.cpp` — `kq_mask_derivable()`, `set_input_kq_derived()`, `set_input_kq_mask()`.
* `src/llama-graph.cpp` — `build_attn_inp_kq_mask` / `llm_graph_input_kq_derived`.
* `tests/test-backend-ops.cpp` — `test_flash_attn_ext` with `derived_hole` / `mask_hole`; **add
  decode-width (`nb 1..8`) and multi-sequence variants here.**

## 6. Repro / measurement harness (as used for r10)

Models on this host:

* small/fast: `/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf`
* MTP (dense): `/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf` (built-in `nextn` head)
* MTP (separate head): `/llm/models/Qwen3.8/27B/EfficientThink/mtp-Qwen3.8-27B-Q4_0.gguf`
* MTP (qwen4exp / Flash-Next): `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/` (+ `mtp-*-shared-Q8_0.gguf`)

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
# build: BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714   (ccache makes rebuilds seconds)

# multi-sequence prefill (r10 reference): S_PP should be flat vs -no-kvu after the fix
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-batched-bench -m ~/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -fa on -c 40960 -b 4096 -ub 512 -npp 8192 -ntg 1 -npl 1,4 -kvu

# packed/unified decode A/B (the target): TG at B=1,4,8 at depth, -kvu vs -no-kvu
HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-batched-bench -m ~/Qwen3.5-4B-Q8_0.gguf \
  -ngl 99 -fa on -c 40960 -b 512 -ub 512 -npp 16 -ntg 256 -npl 1,4,8 -kvu
```

For the concurrent server A/B, the r10 harness pattern is a Python script that starts
`llama-server -np 2 --kv-unified --no-cache-idle-slots -b 1024 -ub 512` with
`GGML_CUDA_DISABLE_GRAPHS=1`, fires two `/completion` requests as threads, and compares
`timings.prompt_ms` / total wall over several repetitions (the server is nondeterministic run-to-run,
so use a median and never a single run). For decode, compare `timings.predicted_*` / `predicted_n` at a
fixed depth with long-lived slots.

## 7. Gates (do not ship without all of these)

1. `test-backend-ops -o FLASH_ATTN_EXT` on **MMA and tile** (`GGML_CUDA_FA_WMMA_256=0`) — must stay
   green with new **decode-width** packed-hole cases and multi-sequence cases.
2. The `W = 1..8` width-purity probe (`GREEDY-PURITY.md` §§10-11): the band skip must not move any
   width's logits. Re-run the raw-logit probe, not just text.
3. `llama-batched-bench -npp 16 -ntg 32 -npl 1,4,8` **stock-relative** verify-width gate
   (`benchmarks/mtp-adaptive-methodology.md` rule 5) — no worse at B=4/B=8.
4. **MTP** on the 27B Q8_0: `draft-mtp n3` and `draft-mtp-adaptive n7`, `-n 1000`, acceptance and t/s
   and generated text identical with `GGML_CUDA_FA_MASK_SKIP` on/off. r10 baseline:
   acceptance 0.59331 / 0.58129, text `sha=832aed3d869d`.
5. Single-stream decode (`llama-bench tg128`) and single-stream prefill unchanged.
6. Concurrent-server decode TTG at depth with N long-lived slots: must improve, and must not regress
   the single-slot case.
7. `scripts/validate-set.sh` green; patch + `release.json` regenerated; docs (`WORKLOG.md`,
   `patches/README.md`, `GREEDY-PURITY.md`) updated; new tag `v16-84e76d8a2-r11`.

## 8. Traps

* **Band fixup:** a skipped band block must write the same neutral partial as an owned-none block
  (`max -FLT_MAX/2`, rowsum 0, VKQ 0); the combine must be bit-identical with and without the skip.
* **Cache mutation:** if you cache `own[stream][group]`, invalidate on every `seq_rm`/`seq_cp`/cell
  reuse — a stale bitmap silently drops visible cells. The recurrent/GDN cache has its own rollback
  semantics; keep this to `llama_kv_cache`.
* **`n_rs_batch`/rollback** is orthogonal but shares the verify width; do not let the two changes
  interact untested.
* **`get_n_kv` pads to 256**, so the group math is exact; do not assume `n_kv % 256 == 0` anywhere else.
* **Do not touch the NVIDIA path.** The bitmap is AMD-gated (`GGML_CUDA_CC_IS_AMD`); a decode skip must
  keep that gate or the `kq_blocks` slot would be misread as `KV_max` on non-AMD.
* **Arch scope:** the change is generic across gfx1201/gfx1100/gfx1151 (shared kernels, no per-arch
  code). Validate on gfx1100 (`fingon`, RX 7900 XTX) and gfx1151 (`halo`, Strix Halo) before the
  promotion — the delivery's precedent validates all three RDNA families.

## 9. Done criteria

Decode/verify with `--kv-unified` no longer pays `O(other slots' cells)` per step; `W = 1..8` and MTP
are bit-identical; single-stream is unregressed; the win is measured on gfx1201 **and** gfx1100 /
gfx1151; the delivery carries it as a block-15 amendment with a dated `WORKLOG.md` entry.
