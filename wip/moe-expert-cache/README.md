# Decode-side MoE expert caching: high-speed expert decode for models that do not fit

**Status (2026-09-28): H1 (targeted fusion) and H3 (uniform adaptive arena sizing) are done.  For a
*non-evicting* cache the output is bit-identical to the full-table GPU oracle and +50 to +78 % over the
delivered CPU decode path.  There is a CRITICAL open correctness bug for a small (evicting) cache: it
produces deterministically wrong output.  See "CRITICAL BUG" in the handover before running any gate.**  The prefill sibling
(`archive/work/tensor-split-expert-split/`, delivery release `v16-84e76d8a2-r16`) is **closed**: its goal
("prefill wins under `-sm tensor` with host-resident experts") is delivered.  This campaign is the decode
half of the same story.

**Not a blocker for anything.**  An optimisation campaign; nothing in the delivery depends on it.  All of
this is `wip/` and applies only to `~/llama-decode`.

---

## CURRENT HANDOVER (2026-09-28): next session continues Phase 1a hardening

The cold-start brief is section 0 below; sections 1 to 3 are the design history, the policy measurement,
and the revised plan.  Read this block first, then jump to whichever section it cites.

### State in one screen

Phase 1a (single GPU, per-layer compact VRAM slot cache, LFRU, slot-remap consumer) works, and **H1 is
done**.  It is one patch, `exp3-moe-expert-cache-phase1a.patch` (879 lines, forward-applies to clean r17).
The working tree is `~/llama-decode` on branch `wip-moe-expert-cache` (r17 `20b0efc5b` + the `exp2`
profiler in `ggml-cpu.c` + the `exp3` cache).  Nothing in the delivery or in `patches/` is touched.

Measured (1 GPU, `-ncmoe 99 -fa 1 -sm layer`; cache 8 GiB / 64 slots; `tg1024`, real steady state):

| config | delivered CPU MoE | cache (H1+H3, fusions on) |
|---|---:|---:|
| Q4_K_M, d0 | 29.4 | **52.3** (+78 %) |
| Q8_0, d0 | 24.4 | **37.3** (+53 %) |
| Q8_0, d16384 | 23.7 | **33.9** (+43 %) |
| Q4_K_M, d16384 | 28.2 | **48.8** (+73 %) |

H1's targeted fusion guard, versus the earlier blanket `GGML_CUDA_DISABLE_FUSION=1` run on Q8_0 d0
`tg64`: 30.7 -> **33.6** (+9.5 %).  All three roles (gate/up/down) are consumed, steady-state
`h` = 0.86 (Q8_0) / 0.94 (Q4_K_M).  **These numbers are only valid because the 64-slot cache did not hit
the CRITICAL BUG below** (its output matches the full-table GPU oracle); a 16-slot cache is wrong.
Short-run text is byte-identical across the GPU paths; `test-backend-ops -o MUL_MAT_ID` is 4/4.

### Build and run

```sh
cd ~/llama-decode && cmake --build build-rocm --target llama-cli llama-bench -j 16
MQ=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
# cache on (8 GiB):
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=8192 MOE_EXPERT_CACHE_TABLES=120 \
  ./build-rocm/bin/llama-bench -m "$MQ" -ncmoe 99 -ngl 99 -fa 1 -sm layer -p 0 -n 64 -r 3
# correctness A/B (GPU kernels identical, expert source differs):
#   off: GGML_OP_OFFLOAD_MIN_BATCH=0 GGML_CUDA_DISABLE_FUSION=1
#   on : MOE_EXPERT_CACHE_MIB=8192 GGML_OP_OFFLOAD_MIN_BATCH=0
```

Env knobs (all in `moe-expert-cache.h`): `MOE_EXPERT_CACHE_MIB` (0/unset = inert), `_TABLES`,
`_SLOTS`, `_PERIOD`, `_FILL` (default 1), `_VERIFY`, `_REPORT`, `_SELFTEST`, `_DEBUG`.  Key code:
`ggml/src/ggml-cuda/moe-expert-cache.{h,cu}`, the scheduler hook in `ggml/src/ggml-backend.cpp`
(`copy_experts`, iface `moe_cache_update`), the iface field in `ggml/src/ggml-backend-impl.h`, and the
consumer + offload/fusion gates in `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_mul_mat_id`,
`ggml_backend_cuda_device_offload_op`, `ggml_cuda_cache_blocks_fusion` called from `ggml_cuda_try_fuse`).

### The hardening tasks (H1 done; H3 done; H2 blocked on the critical bug)

**H1. Targeted fusion. DONE (2026-09-28).**  `ggml_cuda_try_fuse` no longer stands every fusion down
when the cache is on.  A new `ggml_cuda_cache_blocks_fusion(cgraph, i)` returns true only when the cache
is active **and** the node begins a fusion that reads a routed expert table in the cache band: a
`MUL_MAT_ID` with `ne[2] <= MOE_EXPERT_CACHE_MAX_TOK` (the gate+up+GLU triple, the routed pair, and the
qwen4exp weighted-down all start at the routed matmul), or a `GLU` whose next node is a cache-band
`MUL_MAT_ID` (the swiglu to routed-down fold).  Every other fusion (router/topk, GDN, QSA, rope, norms)
returns.  Prefill (`n_tokens > 8`) keeps its MoE fusions, because the cache does not take prefill inputs
over and their `input_cpy` is fully copied.  The band constant `MOE_EXPERT_CACHE_MAX_TOK` (8) lives in the
header and is shared by the hook and the guard.  Result (Q8_0 d0): blanket-off 30.7 -> 33.6 t/s, i.e.
+9.5 %, with byte-identical output.  Option (b) (a cache-aware *fused* MoE that reads the arena/remap, so
even the decode MoE fuses) remains an optional follow-up; (a) is sufficient and loses only the fused-MoE
decode arithmetic, which the cache replaces with the same kernel it uses for the per-op path.

**H3. Budget/slot allocator. DONE (2026-09-28).**  The arena is no longer split by a fixed
`MOE_EXPERT_CACHE_TABLES` hint.  The first decode pass only *registers* every table and runs uncached (the
scheduler copies the full experts); the second pass sizes **every** table to the SAME slot count
`budget / total_expert_bytes` and starts filling.  This adapts to the real table count and to mixed expert
sizes, and it aligns gate/up/down residency (an expert resident for gate is resident for up and down too).
On Q4_K_M the old scheme gave gate/up 121 slots vs down 99; the new one gives all 120 tables **112**, and
steady-state `h` rose 0.79 to 0.83.  `h` is also much higher over 1024 tokens than over 64 (Q8_0 0.86 vs
0.78, Q4_K_M 0.94 vs 0.83), so measure the cache over long generations.  `MOE_EXPERT_CACHE_SLOTS` still
forces an immediate uniform count (the self-test uses it).  A one-token uncached prologue is the cost of
knowing the true table set.

**H2. W=1..8 verify-width purity and MTP.  BLOCKED on the critical bug below** (a width gate is meaningless
while the byte source can be wrong).  Only W=1 is validated so far.  With the cache on and off,
run `--spec-type none` vs `--spec-type draft-mtp` same-seed text, per-W hashes, and the MTP methodology
(`benchmarks/mtp-adaptive-methodology.md`: acceptance above ~0.45 at pos 1, MTP at least equal to plain at
depth 3, plus `llama-batched-bench -npl 1,4,8`).  The consumer already handles `n_tok <= 8`; the hook
skips `n_tok > 8`.

**H3. Budget/slot allocator.**  `moe_cache_table()` splits `MOE_EXPERT_CACHE_MIB` across
`MOE_EXPERT_CACHE_TABLES` assuming equal `expert_bytes`; the `ffn_down` slice is larger (720896 vs 589824
bytes on Q8_0), so it gets proportionally fewer slots.  Make the split byte-aware and fail soft (the
existing report already warns on over-subscription).

### CRITICAL BUG (2026-09-28): wrong output when the cache must evict

**The cache is only correct while it never evicts.**  With a small arena the output diverges
deterministically from the full-table GPU oracle, and the wrong result changes with the eviction policy
(different `MOE_EXPERT_CACHE_PERIOD` values give different wrong hashes), which proves a residency-dependent
wrong read.  This is the top priority: it blocks H2 and any promotion, and it is why the feature is still
default-OFF.

Reproduction (Q8_0, 1 GPU, `-ncmoe 99 -fa 1 -sm layer`, fusions off, 256 tokens, seed 42,
`prompts/reasoning.txt`):

| byte source | slots/table | output |
|---|---|---|
| full-table GPU oracle (`-ngl 99`, no `-ncmoe`) | n/a | correct (Q4_K_M `0e940e23c27f`) |
| no-cache offload (`GGML_OP_OFFLOAD_MIN_BATCH=0`) | n/a | correct (`9b8aaf992518`) |
| cache | 64 | correct (`9b8aaf992518`) |
| cache | 32 | correct (`9b8aaf992518`) |
| cache | 16 | **wrong** (`2ee274cc7003`) |
| cache | 8 | **wrong** (`0845859be8d9`) |

The onset is NOT the first eviction: 16 slots matches 64 slots through token ~96 and first diverges at
token ~128.  All three roles are affected (`MOE_EXPERT_CACHE_SKIP_ROLE` = `ffn_gate_exps` / `ffn_up_exps` /
`ffn_down_exps` each leaves a different wrong output), so the defect is in the shared machinery.

**Ruled out** (checked directly):
* Not a race / run-ahead: `MOE_EXPERT_CACHE_VERIFY=1` (stream-synchronize after every fill and remap) and
  `GGML_CUDA_DISABLE_GRAPHS=1` both give the same wrong hash.
* Not a stale remap: a read-time check that copies the op's own device `ids` (routing matches the hook,
  `ROUTE=0`) and compares the device remap to the live map and the arena bytes to the master finds **0**
  remap or byte mismatches.
* Not the mmvq compute-buffer padding clear (the redirected tensor now carries a null `buffer` and the clear
  is guarded; no output change, so it was inert and is kept as hygiene).
* Not the `weight_cpy -> table` alias (now overwrites instead of `emplace`; no output change).

**Three fixes landed while chasing it** (all defensible, none the root cause): the remap is uploaded
`cudaMemcpyAsync` on the **compute** stream with a persistent host buffer (a synchronous copy on the legacy
stream is not ordered with the non-blocking compute stream); the redirected tensor's `buffer` is nulled
with a guard in the mmvq padding clear; the `weight_cpy` alias overwrites.  `MOE_EXPERT_CACHE_SKIP_ROLE`
and `moe_cache_read_check()` (graphs disabled only) are kept as diagnostics.

**Next step to root-cause:** dump one op's output tensor for the first diverging token (96..128 on Q8_0)
and compare cache-16 vs cache-32 layer by layer.  Because the arena and remap verify correct at read time,
the leading hypothesis is a per-(layer, role) **table-identity** error: the op resolving to another
table's remap/arena through the raw-pointer `g_src0_to_id` map.  Keying the lookup by something stronger
than the device pointer (or carrying the table id into the op) would confirm or eliminate it.

### Purity protocol correction (2026-09-28)

The cache moves the decode MoE from the CPU to the GPU, and the delivered CPU MoE path and the GPU MoE
path are **pre-existing**-different over long generations (verified with the cache inert: `-ncmoe 99` CPU
`abac78fdb0a1` vs `GGML_OP_OFFLOAD_MIN_BATCH=0` GPU `8f7a6c226f4d`, 1024 tokens, Q8_0).  So "same-seed
text cache-on vs cache-off" must be evaluated **GPU-vs-GPU with the same fusion state**, and the trust-
worthy oracle is the **full-table GPU** run (`-ngl 99`, no `-ncmoe`) where the model fits.  CUDA fusions
also change the long-run text vs fusions off even with the cache inert (router fusion not the cause), so
hold the fusion state fixed on both sides.

### Traps learned this session (do not re-derive)

- The top-k `ids` is a **strided view**.  Index it with `ids_tensor->nb[0]`/`nb[1]`, never
  `tok*n_used+j`.  (The scheduler's own `ids` vector is the raw linear bytes, not a compacted block.)
- Keep `src0`'s `ne[2]`/`nb` at the full expert count in the consumer: shrinking `ne[2]` to `slots` changes
  the dispatcher's kernel-family heuristics and breaks bit-identity.
- The scheduler hook runs **outside** CUDA graph capture: the tiny remap upload is synchronous (its source
  is a transient host vector), but the slot **fill** must be `cudaMemcpyAsync` on `ctx->stream()` (a
  synchronous fill inside capture aborts the graph).
- The host master and the op's `src0` are **different tensors** (the scheduler redirects `src0` to
  `input_cpy`).  Drive the policy from the scheduler hook with `weight` (master) and alias `weight_cpy`
  for the op lookup (`moe_cache_get_table`).
- `moe_cache_observe` is kept in the module but no longer called (it double-registered the device copy).
- **Any fusion that reads a routed expert table in the cache band must stand down**, or it reads the
  redirected `input_cpy` that the scheduler did not populate.  That is a *wrong output*, not a slowdown.
  The guard is `ggml_cuda_cache_blocks_fusion` (H1); every new decode-band `MUL_MAT_ID` fusion must go
  through it.  Prefill fusions are safe (the cache does not take prefill inputs over).
- The band is one constant, `MOE_EXPERT_CACHE_MAX_TOK` (8), shared by the hook, the offload relaxation,
  and the fusion guard.  Change it in one place only.

### Fusion policy (answer: yes, enable them; H1 did)

Fusions are bit-identical and beneficial, so the blanket stand-down was a prototype expedient, not the
plan.  H1 restores them now: every non-MoE fusion is on, and only the cache-band routed-expert fusions
stand down (the cache consumer replaces them).  Prefill MoE fusions are untouched.  The remaining option
is a cache-aware fused MoE that reads the arena/remap, which would restore the fused-MoE decode arithmetic
too; it is not needed for correctness or for the current win.

### After hardening

Phase 1b (UVA cold reads, section 3.3) replaces the per-miss fill with in-place pinned-host reads;
Phase 3 (section 3.5) is the `-sm tensor` per-device-slice geometry.

---

## 0. HANDOVER BRIEF — read this first (cold start)

### Goal

Make **decode (and speculative verify) fast for MoE models whose expert weights do not fit in VRAM**, by
keeping a **hot set of experts resident on the device** and handling the cold set cheaply — without giving
up the **tensor split** (`-sm tensor`) that the delivery now makes the best prefill configuration.

The end goal is **Qwen3.8-Flash-Next** (176 B params, 93 GiB at IQ4_NL, `qwen4exp`, 48 layers × 512
experts) running at useful decode speed on 2× R9700 — a model that does not fit 64 GiB and today decodes
slowly.  The iteration target is **Qwen3.6-35B-A3B Q8_0** (37.8 GiB, does not fit one 32 GiB card),
because it is faster to load and test.

### The gap (measured 2026-09-27, this box)

`llama-bench -p 0 -n 64 -fa 1`, Qwen3.6-35B-A3B **Q8_0**, gfx1201:

| config | `tg64` t/s |
|---|---:|
| 2 GPU `-sm layer -ncmoe 0` (all experts **resident**) | **79.7** |
| 2 GPU `-sm tensor -ncmoe 0` (all experts **resident**) | **90.5** |
| 1 GPU `-ncmoe 20` (half the expert layers on host) | 36.4 |
| **1 GPU `-ncmoe 99` (all experts host) — the campaign's baseline** | **24.2** |
| 2 GPU `-sm layer -ncmoe 99` | 22.5 |
| 2 GPU `-sm tensor -ncmoe 99` | 21.5 |

> These are the **r17** numbers (full CSV: `decode-baseline.csv`).  r13–r16 regressed this path to ~13.8 t/s
> (the r13 tiny-CPU-graph heuristic ran the offloaded decode MoE multi-threaded); the **r17** block-06
> amendment exempted `MUL_MAT_ID` src0 and restored the r12 baseline, so this campaign starts from a
> correct floor.  See `WORKLOG.md` 2026-09-27 (r17).

**Why all-host decode is slow: the MoE is not on the GPU at all.**  `ggml_backend_cuda_device_offload_op()`
(`ggml/src/ggml-cuda/ggml-cuda.cu`) only offloads an op when
`get_op_batch_size(op) >= op_offload_min_batch_size` (default **32**, `GGML_OP_OFFLOAD_MIN_BATCH`).  For
`GGML_OP_MUL_MAT_ID` that size is `op->ne[2]` = **`n_tokens`**, which is **1 at decode** — so the decode
graph's `ffn_moe_gate/up/down` are assigned to the **CPU**, in every split mode.  Prefill (ubatch ≥ 32)
offloads; decode does not.  So the 24.2 t/s is a **CPU MoE** number, and every extra resident layer buys
throughput (see the `-ncmoe 20` row) because it moves MoE work back to the GPU.

Note the shape of the prize: **a single card with a cache can beat two cards with none** (36.4 vs 22.5),
and the resident ceiling is 3.3× the baseline.  A working cache does not have to reach 79.7 to be a large
win; it has to move the decode MoE back onto the GPU for the experts that fit.

### Prior art — the baseline starting points (all read 2026-09-27)

Three implementations, summarised in the closed prefill campaign's
`archive/work/tensor-split-expert-split/README.md` §26-§28 (read those sections — they are the
requirements document, with exact file paths):

1. **GenerelSchwerz `llama.cpp` `moe-cache`** (`ggml/src/ggml-cuda/moe-cache.cu`, ~14k lines) — the mature
   **layer-split** design and the closest match to our stack.  Per-layer, per-owner (per-GPU) cache buffer
   with N expert slots; **grouped decode kernels** compute the resident experts on the GPU; misses are
   staged/uploaded; prefetch, replacement policy, an **early router**, and CUDA-graph capture.  Measured
   **Qwen3.6 35B decode 42.8 → 111.6 tok/s (2.6×)** at the same peak VRAM on a 16 GB RTX 5070 Ti.  Its host
   side (`moe-cache-host.cu`) is the mature version of our pinning/staging work: a `moe_host_source`, a
   bounded **pinned staging budget**, automatic expert-group registration, pageable fallback with pinned
   staging, and a dedicated **asynchronous copy worker thread**.  **It explicitly refuses `-sm tensor`**
   ("the tensor-mode meta backend cannot consume the cached buffer"), so it is a *layer-split* reference.
2. **R9V** (`github.com/Dyluhn/R9V`, local `~/R9V`) — **the prior art that DOES tensor-split offloaded
   experts**, on this exact 2× R9700 hardware, for Qwen3.8-Flash-Next.  It keeps
   `get_tensor_model_parallel_rank()`-sharded expert masters (each rank holds `1/tp` of **every** expert —
   i.e. exactly our `-sm tensor` split), a per-rank **hot/cold manifest** (`hot_experts_by_layer`, from
   routing calibration, `materialize_hot_expert_cache`), **cold shards in pinned UVA host memory read in
   place** (`allocate_uva_host_empty` + `parameter.data = get_accelerator_view_from_cpu_tensor(...)`), and a
   VRAM **slot cache** on top (`QWEN38_TIERED_EXPERT_CACHE_SLOTS` ≤ 128, `second_touch_rr`/`lru`,
   optional async fill).  **Key lesson: R9V does not win by making the upload fast — it avoids the upload**
   (resident hot set + UVA in-place cold reads).  Reference files:
   `runtimes/qwen38-flash-next-gfx1201-v1/retained-mtp4/python/vllm_gguf_plugin/quantization/tiered_experts.py`,
   `docs/config.md`, `crates/r9v-ir/src/plan.rs` (`ExpertPlacement::{Device, HostCompute, HostFetch}`).
3. **Strata** (`github.com/Niko1221/Strata`, local `~/Strata`) — single-card, **CPU computes the misses**
   concurrently with the GPU, plus a frequency-profiled VRAM cache (`src/core/expert_cache.cpp`).  Its
   bottleneck is the **CPU expert pool** (663.6 MB/token at ~40 GB/s), not the H2D — their fix is to stop
   reading bytes.  No split mode at all; least transferable.

The synthesis: **the proven decode lever is a hot-expert VRAM cache**; the open question is whether it can
be made to work under our **meta backend / `-sm tensor`**, which is exactly what R9V says is possible and
what nobody in the llama.cpp ecosystem has built (moe-cache refuses).

### The phased plan

Work strictly left to right; each phase must show a measured decode win over its own baseline before the
next starts.  **(Superseded 2026-09-28 by §3, which keeps this ordering but fixes the policy, the seam and
the `-sm tensor`-first constraint; read §3.)**

* **Phase 0 — baseline & instrumentation** (partly done, above).  Sweep `tg` across `-ncmoe 0..40` for
  1 GPU and 2 GPU, `-sm layer` and `-sm tensor`; confirm the CPU-vs-GPU MoE assignment; get a repeatable
  decode harness (the delivery's decode protocol is `d16384`, but shallow is fine for iteration — record
  depth with every number, §"Measurement protocol").
* **Phase 1 — single GPU, Qwen3.6-35B-A3B Q8_0.**  Build a per-layer VRAM **expert slot cache** on one
  card and get `MUL_MAT_ID` decode back on the GPU.  Start with a **static** resident set (no routing
  profile) and a simple policy; measure how throughput scales with the resident fraction.  This is the
  iteration vehicle: one card, one model, fast builds, the smallest surface.
* **Phase 2 — 2 GPUs, `-sm layer`.**  The cache is naturally per-device (a layer split assigns whole layers
  to a device), so this is the moe-cache-shaped case and should be the easiest multi-GPU step.
* **Phase 3 — 2 GPUs, `-sm tensor`.**  The hard and novel one: each device holds a **slice** of every
  expert (axis-1 gate/up, axis-0 down), so a "resident expert" is a resident **per-device slice**, and the
  down projection still needs its cross-device partial reduce.  R9V is the map.  This is the phase that
  unlocks the end goal.
* **Phase 4 — Qwen3.8-Flash-Next IQ4_NL** (93 GiB, `qwen4exp`/QSA).  Re-validate everything on the real
  target; note its PLE tables are host-resident and lazy-loaded (`--lazy-mode auto`), and pre-warm matters.

### Design space (decide by measurement, in this order)

* **A. Hot set in VRAM + stream the misses** (moe-cache).  Reuses what the delivery already has — the
  pinned host expert source (`LLAMA_MMAP_HOST_EXPERTS`), the block-06 staging ring, the op-offload
  `copy_experts` pruning.  Natural first implementation, and it works for `-sm layer` immediately.
* **B. Hot set in VRAM + cold experts read in place over UVA** (R9V).  Pin the host shards once and map
  them into the device address space (`hipHostRegister` + `hipHostMallocMapped`/`cudaHostGetDevicePointer`),
  then have the expert kernel read cold experts straight from host memory — **no per-token H2D at all**.
  This is the end state that avoids the transfer rather than hiding it, and the one that composes with the
  tensor split.  llama.cpp's op-offload does not use UVA today.
* **C. CPU computes the misses** (Strata).  Useful as a fallback for boxes with no spare VRAM, but it
  fights the CPU-pool bandwidth limit and has no split story; keep it as a comparison, not the plan.

### Where the work lives (code hooks)

| concern | location |
|---|---|
| op-offload gate (why decode is on CPU) | `ggml_backend_cuda_device_offload_op()` + `get_op_batch_size()` in `ggml/src/ggml-cuda/ggml-cuda.cu` (default 32, `GGML_OP_OFFLOAD_MIN_BATCH`) |
| the MoE op | `GGML_OP_MUL_MAT_ID`; decode kernel `mul_mat_vec_q_moe` / `ggml_cuda_mul_mat_id` |
| used-expert pruning (host→device ranges) | `ggml_backend_sched_compute_splits`'s `copy_experts` in `ggml/src/ggml-backend.cpp` |
| host-expert pinning (r15) | `select_weight_buft` in `src/llama-model-loader.cpp` (`LLAMA_MMAP_HOST_EXPERTS`) |
| H2D staging ring (r16) | `sched_stage_*` (`ggml-backend.cpp`), `stage_*` hooks (`ggml-backend-impl.h`, `ggml-cuda.cu`, `ggml-cuda/common.cuh`) |
| the split upload (r16) | `ggml_backend_meta_stage_input` + `ggml_backend_meta_set_tensor_async` (`ggml/src/ggml-backend-meta.cpp`) |
| per-tensor split policy | `llama_meta_device_get_split_state` (`src/llama-model.cpp`) |
| existing MoE helpers | `ggml/src/ggml-cuda/topk-moe.cu`, `moe-weighted-reduction.cu`, `ggml/src/models/qwen4exp.cpp` |

### Open questions / risks

1. **Where does the cache live** — a new `ggml_cuda` buffer type (moe-cache) vs the meta backend's simple
   tensors (needed for `-sm tensor`).  The meta backend "cannot consume a cached buffer" today; that is the
   Phase-3 problem to solve.
2. **Residency decision** — static vs routing-level hot set.  The maintainer's earlier steer was "no
   explicit hot-weight tuning" for prefill; for decode a **measured** hot set is the proven 2.6× (moe-cache,
   R9V), so expect to need one.  Start static, then add a routing profile.
3. **Decode/verify purity** — the delivery's `W=1..8` width-purity rule (`GREEDY-PURITY.md`) means the
   cache must produce the *same* arithmetic at every verify width; a grouped/split-by-residency kernel is
   exactly the kind of change that breaks it.  Gate every step with the plain-vs-`draft-mtp` purity check.
4. **Speculative decode** — MTP drafts `n` tokens per step; the used-expert set and the cache hit rate must
   be evaluated at the **verify width**, not just `n_tokens = 1` (the delivery's `MMVQ_MAX_BATCH_SIZE`/
   `MMVF_MAX_BATCH_SIZE` banding).
5. **VRAM budget vs KV** — the cache competes with KV cache for VRAM; the delivery's `--fit` does not know
   about it (same trap as the r3 staging-arena OOM, issue #33).  Any cache allocation must fail soft.
6. **`-sm tensor` down projection** — axis-0 split, `nb[1]`-sized chunks (~524k tiny blocks); any per-token
   upload must be device-side or 2-D (the prefill campaign's §26.3/§29).  This is why UVA (B) is the
   promising `-sm tensor` answer.

### Follow-ups inherited from the prefill campaign (do these as part of this campaign)

These are the loose ends the prefill campaign did **not** close; they are now owned here, cross-referenced
from `archive/work/tensor-split-expert-split/README.md` (§30.5-§31):

1. **Prune the staged upload to the used experts.**  The r16 staged path uploads the **whole** expert
   tensor (all experts, half per device) regardless of ubatch, bypassing the used-expert pruning; at
   ub 512 that is ~10× the volume the pruning path moves.  The split's compute floor is ~7000 t/s and the
   staged path reaches ~5400 at ub 8192, so there is room.  (This also matters for decode if the staged
   mechanism is reused there.)
2. **`GGML_META_GATHER_MODE` device-count heuristic.**  The auto choice (host gather vs device D2D for the
   fine `ffn_down` slice) **inverts** with device count: at 3 GPUs the host gather wins at ub 8192
   (4905 vs 4694), while at 2 GPUs auto (D2D) wins.  Make the heuristic device-count-aware.
3. **3-GPU ub-8192 split loss.**  The split is −5.6 % vs mirrored at ub 8192 on 3 GPUs (the quant block
   limits the split to 2-of-3 devices; the loss is in the upload path).  Close it or document it.
4. **Strip the campaign's env-gated debug/A-B knobs from the delivery.**  The r16 promotion carries the
   debug instrumentation (`GGML_META_STAGEDBG`, `GGML_CUDA_GCDBG`, etc.) alongside the user-facing
   kill-switches; fold them out before any `upstream/` PR candidate is cut.  (Recorded in `WORKLOG.md`
   r16 and `patches/README.md`.)
5. **The §31 prefill residency idea is superseded** by this campaign's Phase 3/UVA design — do not build a
   separate prefill-only residency path; make the cache/UVA design serve both if it can.
6. **`-ncmoe` decode on the CPU** is the root observation of this campaign (see "The gap"); it is not a
   "follow-up" so much as the problem statement.

### Environment, build, and repro

* **Build tree for this campaign: already created and built** — `~/llama-decode`, branch
  `wip-moe-expert-cache`, at the **r17** delivery tip `20b0efc5b273b26f6892012edb07d81e08b44d30` (tree
  `dc7ce12a6af627b0f140b9743e62bc0204f11b10`), with `build-rocm/{bin/llama-bench,bin/llama-cli}` ready.
  Iterate with `cmake --build build-rocm --target llama-bench llama-cli -j 16` (the meta/scheduler TUs
  rebuild in under a minute; a full build via `BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS="
  ~/bin/build-llama-rocm-714` is ~7 min cold).  A clean upstream reference is `~/llama-upstream` at
  `84e76d8a2`.  (The r16 prefill tree is `~/llama-promote`; the canonical r17 chain is `~/llama-fix`
  branch `r17-fix`.)
* **Iteration models:** `/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` (37.8 GiB — the
  Phase-1 vehicle) and `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/…UD-Q4_K_M.gguf` (21.1 GiB — fast smoke tests,
  fits one card).
* **End-goal model:** `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-0000N-of-00009.gguf`
  (93 GiB, `qwen4exp`); its shipping `runme` uses `--lazy-mode auto` + the shared MTP head, and **PLE
  pre-warm matters for its absolute numbers** — do not compare `llama-bench` absolutes for it without the
  server/lazy path.
* **Repro of the baseline gap:** `HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-bench -m <Q8_0> -ncmoe 99 -fa 1
  -p 0 -n 64 -sm tensor -r 3` → ~24 t/s; `GGML_OP_OFFLOAD_MIN_BATCH=0` puts the decode MoE back on the GPU
  for a *correctness* look (it will be slow while every expert streams per token).
* Runtime libs come from the binary's RUNPATH (`/opt/rocm-7.14.1-gfx102X/lib`); `~/llama-r12` is the old
  prefill campaign's tree — use `~/llama-promote`/a fresh worktree instead.
* **Never run parallel benches**; pin nothing; record depth with every number.

### Measurement protocol (do not skip)

* Decode must be measured **at depth** as well as shallow (`d0` and, for the delivery gate, `d16384`):
  "everything is fast at depth 0" is the delivery's standing rule.
* Always gate a decode/fusion change with the **MTP** methodology
  (`benchmarks/mtp-adaptive-methodology.md`) and the **width-purity** check
  (`--spec-type none` vs `--spec-type draft-mtp`, `W=1..8` bit-identical), per `GREEDY-PURITY.md`.
* Same-seed greedy text must be identical to the pre-change build; `test-backend-ops -o MUL_MAT_ID` green.

### Artifacts in this directory

* `README.md` — this brief (the live notebook from here on; append dated findings, never edit them away).
* `decode-baseline.csv` — the Phase-0 `tg` sweep to beat (added with the first measurement session).

### Immediate next steps (fresh session)

1. ~~Build the tree and reproduce the gap table.~~ **DONE** — `~/llama-decode` is built at r17 and
   `decode-baseline.csv` records the `tg64` baseline (all-resident 79.7, all-host 24.2).
2. Read the three prior-art sections (`archive/work/tensor-split-expert-split/README.md` §26-§28) and pick
   the Phase-1 kernel/cache shape (recommended: A, reusing the pinned host source + staging ring).
3. Prototype **Phase 1** on one GPU: a static-resident expert slot cache + a decode path that keeps
   `MUL_MAT_ID` on the GPU for resident experts, with the misses on the existing host path.  Measure
   throughput vs resident fraction.  Fail soft on VRAM (issue #33 lesson).
4. Only then move to Phase 2 (`-sm layer`, 2 GPU) and Phase 3 (`-sm tensor`, 2 GPU = the R9V case).

---

## 1. Phase-1 design — port Strata's expert cache (single GPU), keeping the R9V end state in view

**Maintainer direction (2026-09-27):** start from `~/Strata` — its shape is the most compatible with a
single GPU, and it is the first working milestone before `-sm tensor`.  **Constraint:** the design must be
adaptable to the `-sm tensor` end state (`~/R9V`, TP-sharded experts + hot set + UVA cold reads) without a
rewrite — see §1.2 first, because it decides the data layout.

### 1.1 What Strata actually is (read from source, 2026-09-27)

Three objects, each already verified upstream; the third is unfinished there:

1. **`PinnedArena`** (`include/strata/core/pinned.hpp`, `src/platform/memory.cpp`) — all expert weights in
   **page-locked host memory** (2 MiB pages / `cuMemHostRegister`, with a normal-page fallback that is
   *reported*, not silently taken), with optional per-slice registration and a **device alias** for UVA.
   → **We already have most of this:** r15's `LLAMA_MMAP_HOST_EXPERTS` puts the host-resident `MUL_MAT_ID`
   weights in the `ROCm_Host` buft (`hipHostMalloc`).  Missing: large-page backing, one arena instead of N
   pinned tensors, and the **device alias** (`hipHostGetDevicePointer`) the UVA cold path needs.
2. **`ExpertCache`** (`include/strata/core/expert_cache.hpp`, `src/core/expert_cache.cpp`) — a **VRAM slot
   arena** plus a `(layer, expert) → slot or -1` residency table, filled from the pinned arena (async DMA or
   blocking at startup), with `verify_slot()` reading a slot back and byte-comparing.  **No eviction**
   (deliberate — eviction policy is the measured question, `R4.1`).  The load-bearing finding is
   **per-layer admission**: a global free-slot counter fills the cache inside the first position it ever sees
   (measured 2.97 % hit at 256 slots); giving each layer its own range `q = slots/n_layers` turns the same
   budget into 21 % (8 slots/layer) / 70 % (64).  The per-layer range is what we take.
3. **`ExpertDispatch`** (`include/strata/core/expert_source.hpp`) — **the split is by router index**: of the
   K routed experts, the resident ones are computed on the **GPU** and the rest on the **CPU**; the CPU zeroes
   the hit rows of `parts` and each hit carries its routed index (`dst`) to the GPU kernel; `moe_combine`
   sums by router index, so the two halves rejoin without a scatter.  **Strata has not finished this** —
   `moe_hit_grouped_s2` does not exist, so with the cache on "the engine is slower by the fill cost and faster
   by nothing".  That GPU hit kernel is the piece we must build.

Its measured context (a different, larger model) is worth keeping: the CPU expert pool costs **663.6 MB/token
at ~40 GB/s = 16.2 ms of a ~53 ms token**, and the pipeline hides only 1.055 ms of it — the CPU work is 96 %
exposed.  The lever is to stop the CPU reading those bytes, not to make the CPU read them faster.

### 1.2 The R9V constraint (decide the layout now, not later)

`-sm tensor` decode means each device holds a **slice** of every expert (axis-1 gate/up, axis-0 down) and the
down projection carries a cross-device partial reduce.  So:

* **The residency is keyed by `(layer, expert)`; the slot holds *this device's slice* of that expert.**  Under
  1 GPU / `-sm layer` the slice is the whole expert; under `-sm tensor` it is the axis slice.  The table and
  the dispatch are identical — only `blob_bytes` and the kernel geometry change.  **Do not build the cache
  around whole-expert blobs**; build it around "the bytes this device would upload for this expert".
* **The GPU hit path computes a per-device partial result**, exactly as the existing on-device split does;
  the meta backend's partial reduce is unchanged.  Nothing in the cache may assume the output is complete.
* **The cold path must not be a per-token H2D of the misses.**  The cache makes `(1-h)` of the experts cold;
  if those still cross PCIe every token the streaming bound below kills it.  R9V's answer is **UVA** — map the
  pinned host shards into the device address space and read the cold slices **in place** — which is also the
  only sane shape for the axis-0 `ffn_down` slice (the prefill campaign's §26.3: a host gather there is
  ~524k tiny copies).  **So the single-GPU prototype should still route cold reads through a `device_alias()`
  hop even when the alias is the same device**, so the `-sm tensor` port is a geometry change, not a rewrite.

### 1.3 Measured bounds (2026-09-27, r17, gfx1201)

Qwen3.6-35B-A3B, `tg64`, `-fa 1`:

| shape | 1 GPU | 2 GPU |
|---|---:|---:|
| **CPU MoE** (today's `-ncmoe` decode), Q8_0 | **24.3** | 21.3 (tensor) / 22.8 (layer) |
| **GPU MoE, stream every miss** (`GGML_OP_OFFLOAD_MIN_BATCH=0`), Q8_0 | **7.6** | — |
| GPU MoE, **all resident**, Q4_K_M | 91.8 | — |
| CPU MoE, all host, Q4_K_M | 28.9 | — |
| CPU MoE, half resident, Q4_K_M | 41.9 | — |
| GPU MoE, all resident, Q8_0 | (does not fit) | **90.5** `tensor` / 80.5 `layer` |

Two conclusions the design must respect:

* **PCIe is slower than the CPU's own pinned-memory reads.**  14.45 GB/s over the link versus ~40 GB/s for
  the CPU pool means *"put the MoE on the GPU and stream the misses"* is **3× worse than the CPU** (7.6 vs
  24.3).  A cache is not a nice-to-have; without it the GPU path loses.  The cold path must be UVA (in place)
  or CPU-compute, not a per-token H2D.
* **The prize is large and bounded by resident fraction.**  On one card, all-resident is 3.2× the CPU path
  (91.8 vs 28.9 on Q4_K_M).  A cache that holds a fraction `h` should land between them; the hit rate is the
  number to measure, and per-layer admission is what makes a small budget useful.

### 1.4 Cache budget (Q8_0, one 32 GiB card)

A Q8_0 expert (gate+up+down) is **3.00 MiB**, so with ~4–10 GiB spare on the card:

| cache | slots/layer (of 256) | resident |
|---:|---:|---:|
| 4 GiB | 34 | 13 % |
| 6 GiB | 51 | 20 % |
| 8 GiB | 68 | 27 % |
| 10 GiB | 85 | 33 % |

Strata's profile hit rate is `h_expert ≈ 0.645` at 4,105 slots for a 48×512 model (0.61–0.68 ten-fold
leave-one-out); a smaller per-layer budget will be lower, and the top-8 routing concentrates the popular
experts, so 20–30 % resident is plausibly useful.  **Measure `h` on real routing before sizing anything.**

### 1.5 Phase-1 implementation plan (single GPU, layer split)

> **Superseded 2026-09-28 by §3.**  The steps below are the starting sketch; §3 is the go-forward plan.  It
takes the same Strata shape but pins the policy (LFRU, §2.2) and the seam (`device_alias()` pointer table,
§2.3), and demotes the static profile to a warm start / optional hard-pin analyser.

1. **Residency plumbing.**  A VRAM slot arena + `(layer, expert) → slot` table (per-layer ranges), fed by a
   static/profile hot set, on top of the r15 pinned host source.  No eviction.  Gate it off by default
   (`GGML_MOE_EXPERT_CACHE_MIB`, 0 until measured) with a startup `h`/`fills`/`verify` report like Strata's.
2. **The GPU hit path.**  Compute the resident experts on the GPU from the cache and the misses on the CPU
   (or UVA), split **by router index**, combined in router order.  Build it so the "device address" of a blob
   is already an abstraction (the `device_alias()` hop of §1.2), even when it points at the same device.
3. **Purity gate.**  The delivery's `W=1..8` width-purity rule applies: the cache must produce the same
   arithmetic at every verify width and with the cache on/off (`--spec-type none` vs `draft-mtp`, same-seed
   text).  The split-by-residency path is exactly the kind of change that breaks this.
4. **Measure** `h`, `tg` vs resident fraction, and the cold-path cost, on Q8_0 single GPU; only then move to
   2-GPU layer split, then `-sm tensor` (the R9V case).

### 1.6 The gating measurement (2000-token generations, held-out): the routing is concentrated, but a STATIC profile does not transfer

**Instrument:** `exp1-moe-routing-profiler.patch` — an env-gated (`GGML_MOE_PROFILE`) counter in the CPU
`ggml_compute_forward_mul_mat_id` (where decode's MoE runs under `-ncmoe`).  It records `(layer, expert)`
usage for the **decode/verify band only** (`n_tokens <= 8`; a prefill ubatch routes nearly every expert and
would swamp the distribution) and dumps `expert,<layer>,<expert>,<count>` at exit.  `hitrate.py` scores it.

**Runs:** Qwen3.6-35B-A3B Q8_0, 1 GPU, `-ncmoe 99 -sm layer`, **2000 greedy tokens each** on the delivery's
five prompts (`prose`, `code-python`, `reasoning`, `recall`, `code-reasoning-mixed`).  (A 256-token run was
done first and over-states everything — longer generations keep touching new experts, so the same cache
covers less of the distribution: in-sample h at S=64 was 0.800 at 256 tokens, **0.724 at 2000**.)

`h(S)` = fraction of routed experts a **per-layer static hot set of `S`** would have resident, book = the
prompt's own frequency (in-sample) or a specified book (held-out):

| S/layer | resident | cache | in-sample (range over 5 prompts) | held-out, book = **one** prompt (prose) | held-out, book = **all 5** combined |
|---:|---:|---:|---:|---:|---:|
| 16 | 6 % | 1.9 GiB | 0.36–0.42 | — | — |
| 32 | 12 % | 3.8 GiB | 0.52–0.61 | — | 0.13–0.44 |
| 48 | 19 % | 5.6 GiB | 0.63–0.73 | — | 0.21–0.53 |
| **64** | **25 %** | **7.5 GiB** | **0.63–0.81** | **0.24–0.38** | **0.30–0.61** |
| 128 | 50 % | 15.0 GiB | 0.90–0.96 | 0.54–0.64 | 0.68–0.85 |

**Read — this changes the design, and it is why the measurement was worth doing:**

1. **The routing is concentrated and a cache is clearly worth it *if the book matches the workload*.**
   In-sample, 25 % resident buys 63–81 % of the reaches; 50 % buys 90–96 %.  On one 32 GiB card a ~7.5 GiB
   cache is realistic and would leave the cold path ~1/5 of the experts instead of all of them.
2. **A static hot set does NOT transfer across workloads.**  A book built from `prose` scores only
   **0.24–0.38 at S=64** when the run is `code`/`reasoning`/`recall` (vs 0.63–0.81 in-sample).  Even a book
   built from all five prompts only reaches **0.30–0.61**, and `recall` (verbatim continuation) is the worst
   transfer of all (0.30).  So a **static profile is a warm start, not the mechanism** — it must be paired
   with a **dynamic admission policy** that adapts to the running workload (Strata's `second_touch_rr`,
   R9V's slot cache, moe-cache's replacement) and/or a per-workload profile.
3. **Budget the dynamic policy, not the profile.**  The meaningful number to design against is the
   *steady-state* hit rate of the chosen policy (fill-on-second-touch / LRU / LFU-decay) measured over a
   long run, not the profile's in-sample score.  The static numbers above are the upper bound of the warm
   start.

**Follow-ups the profiling leaves:** (a) score on a *later* slice of the same long run (warm the cache on
tokens 0..1000, score on 1000..2000) — the closest proxy for a dynamic policy; (b) measure a real
second-touch/LRU policy, not a rank list; (c) re-check on `qwen4exp`/Flash-Next (48×512), where Strata's
held-out profile was 0.645 — a much larger expert count, so the hit rate may behave differently.

Artifacts: `profile-prose-256tok.csv` (the first, short run, kept to document why short runs mislead) and the
five 2000-token dumps (regenerable via the commands in §1.7; not all committed for size).  `hitrate.py`
reproduces every table above.

---

## 2. FINDINGS (2026-09-28): the dynamic admission policy measured; the cache integration branch point settled

This session did the top task of the handover: **settle the policy and the integration seam before writing
any cache code** (the campaign's "measure h first" rule).  Three results: the instrument now captures an
ordered, token-indexed trace; the policy is measured on all five workloads at 2000 tokens; and the
`ggml_cuda`-buffer-vs-meta-simple-tensor question (open question #1) is answered.

### 2.1 Instrument: `exp2-moe-routing-profiler.patch` (supersedes `exp1`; do not apply both)

The old `exp1` profiler counted per `MUL_MAT_ID` op and aggregated `(layer, expert)` counts.  It had two
flaws for a policy measurement:

* **it triple-counted.**  The MoE graph issues one `MUL_MAT_ID` per gate/up/down and they all share the same
  `ids`, so the old `layer` line was `3 x tokens` (the committed `prof-*.csv` show 5370/6021 for 1790/2007
  tokens).  The extension coalesces consecutive ops of one `(layer, token)` — exactly the granularity a cache
  looks an expert up at.
* **it had no order.**  LFRU/LRU/LFU/second-touch are all order-dependent, so an aggregate book cannot score
  them.  The extension writes an ordered, token-indexed trace: `route,<tok>,<layer>,<expert>` (one line per
  reach) to `<path>.trace`.

It also adds the requested `GGML_MOE_PROFILE_SPLIT=N`, which writes two aggregate books for the **same run**:
`<path>.pre` (tokens `< N`) and `<path>.post` (tokens `>= N`) — the held-out proxy §1.6 asked for — and a
`split,<N>` marker in the trace.  `GGML_MOE_PROFILE_MAXTOK=1` records pure W=1 decode (a prefill *tail*
chunk of `<=8` tokens is otherwise indistinguishable from a verify batch and pollutes the trace; even the
2000-token runs had a 7-token tail in the `reasoning` prompt before this).  Inert without
`GGML_MOE_PROFILE` (one `getenv`).

Traces for this section (1 GPU, Qwen3.6-35B-A3B Q8_0, `-ncmoe 99 -fa 1 -sm layer`, `-n 2000 --ignore-eos`,
`GGML_MOE_PROFILE_MAXTOK=1 GGML_MOE_PROFILE_SPLIT=1000`, one run per prompt):

```sh
MQ=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
for p in prose-rdna-boosts code-python reasoning recall code-reasoning-mixed; do
  env HIP_VISIBLE_DEVICES=0 GGML_MOE_PROFILE=/tmp/moecache-traces/$p.csv \
      GGML_MOE_PROFILE_SPLIT=1000 GGML_MOE_PROFILE_MAXTOK=1 \
      ./build-rocm/bin/llama-cli -m "$MQ" -ncmoe 99 -fa 1 -sm layer \
      -f ~/llama-cpp-rdna-boosts/prompts/$p.txt -n 2000 --ignore-eos \
      --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt
  # -> /tmp/moecache-traces/$p.csv{,.pre,.post,.trace}
done
# policy simulator (no GPU needed, ~1 min for 8 policies x 5 S x 5 prompts)
python3 policy_sim.py /tmp/moecache-traces --windows /tmp/moecache-traces/policy-windows.csv
```

Each trace is 1999 decode steps (8 experts x 40 layers = 639,680 reaches; the first sampled token's MoE runs
inside the prompt batch, so 2000 generated tokens give 1999 profiled steps) at d0.  Runtime libs come from
the binary RUNPATH; the campaign build tree is `~/llama-decode` (`build-rocm`).  `--ignore-eos` is required
to compare all five at equal length: `recall` emits EOS at 505 tokens on its own (§1.6's dump), the other
four ran to ~1800-2000, so the earlier numbers were not length-matched.

**Same-run held-out warm/cold (item 1's deliverable).**  Rank by tokens 0..1000, score on 1000..2000 — the
same workload, so this removes §1.6's cross-workload transfer error:

| prompt | S=8 | S=16 | S=32 | S=48 | S=64 |
|---|---:|---:|---:|---:|---:|
| code-python | 0.248 | 0.368 | 0.496 | 0.589 | 0.659 |
| code-reasoning-mixed | 0.204 | 0.326 | 0.458 | 0.555 | 0.634 |
| prose-rdna-boosts | 0.188 | 0.292 | 0.435 | 0.554 | 0.642 |
| reasoning | 0.222 | 0.343 | 0.504 | 0.614 | 0.695 |
| recall | 0.219 | 0.351 | 0.544 | 0.679 | 0.770 |

At 25% resident (S=64, 7.5 GiB) the warm start holds **0.63-0.77** within a workload — the static profile is
far better than §1.6's cross-workload 0.24-0.38, but it is still a warm start, not the steady state (below).

### 2.2 Measured: a tuned LFRU dominates LRU, LFU and second-touch on the robustness metric

`policy_sim.py` replays the trace per layer (per-layer slot ranges, Strata's load-bearing finding) and
implements the prior-art policies (`static`, `warm_lru`, `lru`, `second_touch`, `lfu_decay`) plus combined
ones (`lfru_decay`, `slru`, `lru_window`).  `h_steady` is the hit rate over the **last 1000 of 1999** steps;
"worst" is the minimum across the five prompts (= the robustness number the campaign asked for, not the best
rank list).  Q8_0, 1 GPU, `-ncmoe 99 -sm layer`, d0:

| S/layer | cache | policy | `h_steady` mean | worst prompt | worst 200-tok window |
|---:|---:|---|---:|---:|---:|
| 64 (25%) | 7.5 GiB | static (warm 0..1000) | 0.680 | 0.635 | 0.553 |
| 64 | 7.5 GiB | `lru` | 0.763 | 0.737 | 0.700 |
| 64 | 7.5 GiB | `second_touch` | 0.752 | 0.729 | 0.699 |
| 64 | 7.5 GiB | `lfu_decay:32` | 0.766 | 0.743 | 0.718 |
| 64 | 7.5 GiB | `slru:0.5` | 0.775 | 0.753 | 0.726 |
| 64 | 7.5 GiB | **`lfru_decay:32`** | **0.783** | **0.764** | **0.736** |
| 32 (12%) | 3.7 GiB | `lru` | 0.583 | 0.542 | 0.505 |
| 32 | 3.7 GiB | `second_touch` | 0.561 | 0.517 | 0.460 |
| 32 | 3.7 GiB | `lfu_decay:32` | 0.586 | 0.562 | 0.527 |
| 32 | 3.7 GiB | **`lfru_decay:16`** | **0.611** | **0.576** | **0.547** |
| 16 (6%) | 1.9 GiB | `lru` | 0.425 | 0.360 | 0.339 |
| 16 | 1.9 GiB | **`lfru_decay:16`** | **0.438** | **0.398** | **0.377** |
| 8 (3%) | 0.9 GiB | `lru` | 0.229 | 0.175 | 0.163 |
| 8 | 0.9 GiB | **`lfu_decay:32`** | **0.288** | **0.258** | **0.244** |

`lfru_decay` = admit every miss, evict the resident with the smallest hit counter, break ties by oldest use,
halve all counters every 16-32 decode steps.  Full grid: `policy-sim.csv`; per-window rates:
`policy-windows.csv`.

**What it says.**

1. **A dynamic policy beats the static warm start at every budget** (+0.05 at S=8, +0.10 at S=64, mean).  The
   static book is a useful *accelerator* (it removes the cold first ~100 steps) but it is not the mechanism —
   exactly what §1.6 predicted, now confirmed within a single workload.
2. **The tuned hybrid is the winner on the metric that matters: the worst workload and the worst window.**
   `lfru_decay` (period ~16-32) has the highest worst-prompt `h_steady` at S=16 and S>=32 and is within
   0.008 of the best at S=8; `lru` is beaten on its own worst case by +0.027 (S=64) to +0.034 (S=32).  The
   decay period is a broad plateau (16/24/32 are within ~0.01); period 32 is the robust default, and at a
   **tiny** budget (S<=8) the pure-LFU admission filter (`lfu_decay`) is marginally better than admit-all.
3. **`second_touch` is dominated.**  R9V's default and moe-cache's shape measure the *lowest* worst-case
   dynamic policy at every S: the ghost counter delays admission, so it reacts slower to a phase shift than
   plain LRU and cannot match LFRU's frequency protection.  Corroborated, but not the policy to port.
4. **The collapse is real, and only the dynamic policies survive it.**  The phase-switching prompt
   (`code-reasoning-mixed`, S=64) shows the static book decaying mid-run — windows 0-9:

   | policy | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | static (warm) | 0.83 | 0.78 | 0.72 | 0.72 | 0.78 | **0.55** | **0.57** | 0.72 | 0.66 | 0.67 |
   | `lru` | 0.83 | 0.83 | 0.81 | 0.81 | 0.82 | 0.82 | 0.80 | 0.82 | 0.83 | 0.81 |
   | `lfru_decay:32` | 0.84 | 0.81 | 0.79 | 0.80 | 0.80 | 0.84 | 0.81 | 0.81 | 0.82 | 0.78 |

   `prose` is the same shape (static 0.75→0.58 at window 8, `lfru_decay:32` 0.80→0.75).  The static book is
   `0.83` while the code phase runs and `0.55` after the phase change, because it was warmed on the wrong
   half.  A design that fills the cache by the profile alone and never evicts would inherit that cliff.

**Decision:** the cache's admission/replacement policy is **LFRU — admit every miss, evict the resident with
  the smallest decaying hit count (LRU tie-break), counters halved every ~32 decode steps** — with per-layer
  slot ranges.  The static profile is a warm-start hint only; `slru:0.5` is the close, counter-free second
  (two ordered lists, no decay), worth keeping as the low-overhead A/B.

**Depth check (house rule: measure at depth as well as shallow).**  The same 2000-step trace from a ~16k
prompt (the prose prompt tripled, 15.7k tokens, `-c 20480`) reproduces the ranking.  S=64: `lfru_decay:32`
h_steady `0.779` / worst window `0.750`, `slru:0.5` `0.769`/`0.730`, `lfu_decay:32` `0.759`/`0.741`, `lru`
`0.755`/`0.693`, `second_touch` `0.738`/`0.708`, `static` `0.681`/`0.636`.  The choice is not a shallow-context
artifact; the five-prompt set already spans prompt depths ~0.3k-5.2k and this adds ~16k.

### 2.3 Recommendation: hook the cache at the `MUL_MAT_ID` weight-source seam, not a buffer type

This settles §1.4 open question #1.  **Recommendation: do not add a `ggml_cuda` cache buffer type to the
delivery.  Build the residency as a per-device `(layer, expert) -> device_alias()` manager at the CUDA
backend's `MUL_MAT_ID` weight-source seam, and use a custom CUDA arena only as the Phase-1 single-GPU
scaffold behind that same `device_alias()` seam.**

Why not a buffer type, from the code:

* **A buffer type is static storage; residency is dynamic.**  `ggml_backend_meta_buffer_type()` builds the
  meta tensor's split from its simple devices' *existing* buffer types, and `ggml_backend_meta_device_supports_buft()`
  (`ggml/src/ggml-backend-meta.cpp`) only accepts a buft owned by one of its simple devices.  A cache buft
  therefore cannot back the split expert tensor; the split tensor stays in the ordinary CUDA buft and the
  cache would have to be a second, side allocation the kernel consults.  This is precisely why moe-cache
  refuses `-sm tensor` (§26.2: "the tensor-mode meta backend cannot consume the cached buffer").
* **The kernel needs a base pointer per expert, not a new allocation.**  Today `src0` is one contiguous expert
  table and the MMQ/mmvq path offsets rows by expert id (the ids compaction is in `ggml/src/ggml-cuda/mmid.cu`;
  the same `ids` drive gate/up/down).  A cache makes the used experts' storage discontiguous — a resident
  VRAM slot or a cold alias — so the interface the kernel needs is `(layer, expert) -> base_ptr`.  That is
  the *same* interface for 1 GPU, `-sm layer` and `-sm tensor`; only `blob_bytes` (the device's slice) and
  the pointer's source change.  A `device_alias()` hop now makes the `-sm tensor` port a geometry change, not
  a rewrite (the §1.2 constraint).
* **The scheduler seam already exists.**  `ggml_backend_sched_compute_splits`'s `copy_experts`
  (`ggml/src/ggml-backend.cpp`, the `used_ids` bitmap + `ggml_backend_tensor_set_async` into the per-op
  `input_cpy`) already copies only the used experts to the device.  A cache turns "copy every used expert"
  into "copy only the misses", and the `input_cpy`/`set_tensor_async` path is the natural hook.  The
  op-offload gate (`get_op_batch_size()` in `ggml/src/ggml-cuda/ggml-cuda.cu`: `MUL_MAT_ID` -> `op->ne[2]` =
  `n_tokens` = 1 at decode, vs the default 32) has to be relaxed for a cache-active MoE, since the resident
  experts *are* on the device.  Under `-sm tensor` the meta device's `offload_op` is `all_of(simple_devs)`,
  so the same relaxation is needed there.
* **`-sm tensor` fixes the cold path to UVA.**  Each device's slot holds *its* slice (axis-1 gate/up, axis-0
  down); `ffn_down`'s split is `nb[1] = 176` x `524288` chunks (§26.3), so a per-token H2D of the misses is
  half a million tiny copies per layer — the only sane cold path is R9V's: pin the host slice and map it into
  the device address space (`hipHostRegister` + `hipHostGetDevicePointer`), then read cold experts in place.
  The meta backend's per-device partial reduce is untouched.

The trade, stated plainly:

* **Custom `ggml_cuda` arena (moe-cache's shape):** fastest route to a Phase-1 number, reuses `input_cpy`, no
  scheduler/meta changes — but it is `-sm layer`-only, is thrown away for the end goal, and puts a
  `--fit`-invisible allocation inside the CUDA backend (the issue #33 trap; any VRAM allocation must fail
  soft).
* **`device_alias()` pointer table on the meta simple tensors:** more work (a residency manager, a
  pointer-indirection in the `mul_mat_id` dispatch, pinned host slices + UVA registration, and a scheduler
  that can ask for "resident slot or cold alias") — but it is the only geometry that reaches the 93 GiB
  qwen4exp/`-sm tensor` end goal, and it carries forward the r15 pinning + r16 staging/op-offload work.

**Decision:** branch at `device_alias()`.  Phase 1 may wrap a plain CUDA slot arena to get a number, but the
residency table and the kernel-facing pointer must be written once, in the shape both splits need.

**Two quantitative bounds to carry into Phase 1 (Q8_0, 1 GPU, from §1.3):** a token touches
`8 experts x 3 MiB x 40 layers = 960 MiB`; at the measured 14.45 GB/s link a 22% cold fraction (`h=0.78`, S=64)
is ~211 MiB/token = ~14.6 ms/token, i.e. a ~68 t/s UVA cold-read ceiling — still 2.8x today's 24.2 t/s but
short of the 91.8 t/s all-resident.  A **CPU-computes-the-misses** variant reads the same 211 MiB at ~40 GB/s
= ~5.3 ms, which fits *under* the GPU's all-resident compute (10.9 ms/token at 91.8 t/s); Strata's design
(design space C) is therefore not just a fallback — at our hit rates it could hide the cold set entirely,
which is why it should be kept as the Phase-1 comparison, not discarded.  Both are bounded by getting the
resident set computed on the GPU first.

### 2.4 What is still open (for the next session)

1. **Verify-width traces.**  §2.2 is W=1.  The campaign's open question #4 wants the same policy scored at
   the MTP verify width (`n_tokens` 1..8), where the used-expert set per step is wider and the hit rate per
   *reach* differs.  The instrument already records the band (`GGML_MOE_PROFILE_MAXTOK=8`); run a
   `draft-mtp` trace and re-score.
2. **Phase 1 build.**  A single-GPU slot arena (per-layer, LFRU) behind `device_alias()`, with the resident
   experts computed on the GPU and the cold set on the existing CPU path (the quickest measurable win), then
   the UVA/pointer-table variant.  Gate with the delivery's W=1..8 width-purity check and same-seed text.
3. **`qwen4exp` re-check.**  §1.6's Strata held-out profile was 0.645 at 48x512; re-run the trace on
   Flash-Next IQ4_NL when its lazy/PLE path allows a stable 2000-token decode (its `llama-bench` absolutes
   are not comparable — see §0).
4. **Profile source.**  The warm-start book should come from a per-workload profile or the delivery's own
   prompt set; the same-run `.pre`/`.post` split is the offline proxy, not a runtime dependency.

Artifacts added this session: `exp2-moe-routing-profiler.patch`, `policy_sim.py`, `policy-sim.csv`,
`policy-windows.csv`.  The 5 raw traces (~10 MiB each) and the ~16k-depth trace are regenerable with the
command in §2.1 and are **not** committed; the old `prof-*.csv` are the exp1 (triple-counted, order-free)
historical record and are left untouched.  None of this touches the delivery: it is `wip/`, env-gated, and
applies to `~/llama-decode` only.

---

## 3. REVISED IMPLEMENTATION PLAN (2026-09-28) — go-forward reference

**This supersedes §1.5 and the §0 phased plan.**  Maintainer sign-off 2026-09-28: proceed with the
`device_alias()` seam, keep the `-sm tensor` end state as the primary constraint, and treat the static
hard-pin as a lower-priority, no-code accelerator.

### 3.0 What changed from §1.5

| item | was (§1.5) | now (§2/§3) |
|---|---|---|
| policy | "a static/profile hot set, no eviction" | **LFRU** — admit every miss, evict min decaying count (LRU tie-break), halve counts every ~32 decode steps, per-layer slot ranges; static book is a warm start only |
| seam | a `ggml_cuda` slot arena fed from the pinned host source | **`(layer, expert) -> device_alias()` pointer table** at the `MUL_MAT_ID` weight-source seam; a CUDA arena is only the Phase-1 scaffold *behind that seam* |
| hard-pin | not in the plan | a **lower-priority session analyser** that emits a per-layer pin table for existing `-ot`/`-ncmoe` config (a warm start / no-code win, **not** the mechanism) |
| `-sm tensor` | "adapt later" | **the design target from the first line of code** |

### 3.1 North star / non-negotiable invariants (the "do not back into a corner" list)

1. **One abstraction, three geometries.**  The deliverable's decode MoE path must be the same code for
   1 GPU, `-sm layer` and `-sm tensor`; the only differences are `blob_bytes` (the *device's slice* of the
   expert) and the alias source.  If a change needs whole-expert-only blobs or a non-meta consumer, stop and
   redesign.
2. **Every blob address is a `device_alias()`.**  The kernel reads a per-`(layer, expert)` base pointer and
   never assumes the expert table is contiguous.  Under `-sm tensor` that alias may be a resident VRAM slot
   or a UVA view of the pinned host slice (this device's slice in both cases).
3. **Residency is keyed by `(layer, expert)`; the slot holds this device's slice** (axis-1 gate/up, axis-0
   down).  Never build a whole-expert-blob cache.
4. **Per-layer slot ranges** (Strata: a global pool measures ~3% at the same budget; per-layer turns it into
   21-70%).  LFRU as above; the static profile only seeds it.
5. **VRAM allocation fails soft and is visible.**  The arena is outside the compute reserve; a failed alloc
   disables the cache and warns (issue #33), and `--fit` must either count it or document the exclusion.
6. **Width purity.**  `W = 1..8` must agree and same-seed text must be identical cache-on vs cache-off.  The
   resident/cold split must rejoin in router order and zero the complementary half deterministically.
7. **The cold path is never a per-token H2D of the `ffn_down` slice** (176-byte rows x 524288).

### 3.2 Phase 0.5 (lower priority): static block-pin analyser

A no-code win for users, and the same profile loader the cache needs for its warm start.  It does **not**
block the cache.

* Use `exp2` to collect routing for a workload (or the delivery prompt set), rank layers (and experts, for
  the cache) by reach frequency, and emit a config table.
* **Granularity is the whole expert tensor / layer, not the expert.**  `-ot` matches a tensor-name regex and
  `-ncmoe N` expands to per-block overrides for blocks `0..N-1` only (`llm_add_n_cpu_ffn_overrides`,
  `common/common.h`), so today's options cannot pin an individual expert.  The analyser's value over `-ncmoe`
  is that it can emit a **non-uniform, hotness-ranked subset of blocks** (pin the hottest layers, leave the
  rest), which `-ncmoe`'s first-N rule cannot express.
* Be explicit about the ceiling: §1.6 says a static set does not transfer across workloads (held-out
  0.24-0.38), and §2.2 confirms that even within one workload it collapses mid-run (0.83 -> 0.55 on the
  phase-switching prompt).  This is a warm start / no-code option, not the mechanism.  Ship it after the
  cache seam; the profile format is shared.

### 3.3 Phase 1 — single GPU, Qwen3.6-35B-A3B Q8_0 (the iteration vehicle)

> **Current status and the next three tasks are in the CURRENT HANDOVER at the top of this file.**  The
> slot-remap consumer is built and measured; H1 to H3 are the hardening steps.

Goal: keep `MUL_MAT_ID` decode on the GPU for the resident fraction, with the seam Phase 3 needs, and a
measured `tg` vs `h` curve.  Build order is 1a -> 1b -> 1c -> 1d; each is a separate commit + measurement.

**1a — seam + LFRU residency + split-by-router-index dispatch (cold on the CPU).**
* Per-device residency manager: per-layer slot arena (`S = MOE_EXPERT_CACHE_MIB / (n_layers x 3 MiB)` for
  Q8_0), `(layer, expert) -> slot`, LFRU (period ~32), optional static warm start.
* The `mul_mat_id` dispatch consumes a base-pointer table (resident slot vs cold alias) rather than assuming
  contiguity.  For 1a the cold alias is the existing CPU expert source, and the op is split **by router
  index** (Strata's `ExpertDispatch`): zero the resident rows of the CPU `parts`, compute the resident rows on
  the GPU, and combine by router index (the shared `mul_mat_id` combine already sums by routed `dst`).  This
  is the only shape that makes the resident fraction help without a per-token H2D.
* Relax the op-offload gate for a cache-active MoE: `get_op_batch_size(MUL_MAT_ID) = op->ne[2] = 1` at decode
  keeps the whole op on the CPU, but a resident expert *is* on the device.  The scheduler must be taught to
  offload the resident half and leave the cold half on the CPU — this is the main correctness surface.
* Gate: Strata-style `h`/fills/verify report; `test-backend-ops -o MUL_MAT_ID` green; same-seed text ==
  cache-off at W=1; `tg` vs `h` (and vs the 24.2 t/s CPU and 91.8 t/s all-resident bounds) at d0 and d16384.

**Phase 1a status (2026-09-28, `exp3-moe-expert-cache-phase1a.patch`, built + proven on `~/llama-decode`):**

* Delivered: `ggml/src/ggml-cuda/moe-expert-cache.{h,cu}` — the `device_alias()` seam
  (`moe_cache_alias_get`), the per-layer compact slot arena, LFRU (period 32, LRU tie-break), fail-soft
  `cudaMalloc` (a failed alloc disables that table and warns), lazy per-`src0` table registration, and an
  exit `h`/fills/evictions report.  Gated by `MOE_EXPERT_CACHE_MIB` (default 0 = inert, bit-identical).
* Driven from the **scheduler's op-offload seam**, not the CUDA op: a new backend iface hook
  `moe_cache_update(backend, weight, ids, n_used, n_tok)` (`ggml-backend-impl.h`) is called by
  `ggml_backend_sched_compute_splits`'s `copy_experts` with the **true host master** `input` and the
  host-readable routing.  That is the only place both exist (the op-offload redirect makes the op's
  `src0->data` point at the device `input_cpy`), and it runs **before** the backend's MoE fusion, so it
  covers the delivered fused path (the earlier per-op observer did not — the decode MoE is fused via
  `ggml_cuda_try_fuse`'s `mul_mat_id_glu_ops`, so `ggml_cuda_mul_mat_id` is skipped by default).  The
  CUDA backend implements it as `moe_cache_update_host()` and passes `ctx->stream()`, so the fill is
  `cudaMemcpyAsync` and **capture-safe** (a synchronous fill aborted CUDA graph capture; the scheduler's
  own expert copies are async for the same reason).
* Proven: `MOE_EXPERT_CACHE_SELFTEST=1` byte-verifies the resident slots — **PASS**
  (`slots=4 resident=4 hits=3 misses=13 fills=13 evictions=9`).  Live on the real path
  (`-ncmoe 99 GGML_OP_OFFLOAD_MIN_BATCH=0`, fusion default, `MOE_EXPERT_CACHE_FILL=1`): **120 tables
  (= 40 layers x gate/up/down), 6736 slots, 4047 MiB <= the 4096 MiB budget, `h=0.712` (22558/31680
  reaches) over 32 decode steps**, and multiple CUDA-graph captures complete cleanly.
* **Slot-remap consumer: DONE (2026-09-28).**  The op now reads the arena.  `moe_cache_update_host()`
  stages the slot-remapped ids (indexing the routing with the ids tensor's own strides, which matters: the
  top-k `ids` is a strided view, not contiguous), the scheduler skips its own expert copy when the backend
  takes the input over, and `ggml_cuda_mul_mat_id` re-dispatches through shallow `src0`/`ids` copies whose
  `data` points at the arena and the remap buffer.  `src0`'s `ne[2]`/`nb` stay at the full expert count so
  the dispatcher's kernel-family heuristics (and therefore the arithmetic) are unchanged; only the base
  pointer moves, and the remapped ids stay below `slots`.  Prototype detail: all backend fusions are stood
  down while the cache is active (the consumer is the unfused per-op path), so the A/B runs the cache-off
  side with `GGML_CUDA_DISABLE_FUSION=1` too.

  Measured (1 GPU, `-ncmoe 99 -fa 1 -sm layer`, `tg64`, one or three reps; cache 8 GiB unless noted):

  | config | delivered CPU MoE | no-cache GPU stream | cache |
  |---|---:|---:|---:|
  | Q4_K_M, d0 | 29.1 | 9.7 | **35.5** (+22 %) |
  | Q8_0, d0 | 24.5 | - | **30.8** (+26 %); 23.6 at 4 GiB |
  | Q8_0, d16384 | 23.7 | - | **29.7** (+25 %) |

  Live `h = 0.7895` (49266/62400 reaches) with the 8 GiB budget (8160 MiB arena, 7680 slots, 120 tables).
  Same-seed greedy text is **byte-identical** cache-on vs the no-cache GPU path (short gate
  `359ff4337837`, 573-char decode `a371535187a5`), and `test-backend-ops -o MUL_MAT_ID` is 4/4.  Q8_0 is
  break-even at 4 GiB: its expert slices are ~1 MiB, so ~22 slots/table at 8 GiB is where `h` is high
  enough to beat the CPU.

* **What is left for Phase 1a:** (a) replace the blanket fusion stand-down with a **cache-aware fused MoE**
  (or a targeted MoE-only disable) so the non-MoE fusions stay on; (b) the W=1..8 verify-width purity and
  MTP gates (the consumer handles `n_tok <= 8`, but acceptance and width purity are not yet re-measured);
  (c) a real budget/slot allocator (the current per-table split uses a fixed `MOE_EXPERT_CACHE_TABLES`
  estimate, and the larger `ffn_down` slice gets fewer slots than gate/up); (d) the `-sm tensor`
  per-device-slice geometry (Phase 3).  Then Phase 1b (UVA cold reads) removes the per-miss fill.

**1b — UVA cold reads (the parallel-offload hypothesis).**
* r15 already puts the host experts in pinned memory (`ROCm_Host`/`hipHostMalloc`).  Register the slice and
  take a device alias (`hipHostRegister` + `hipHostGetDevicePointer` / `cudaHostGetDevicePointer`);
  `device_alias()` returns the resident slot when resident, else the UVA alias, and the kernel reads cold
  experts in place — no per-token H2D, no CPU cold compute.
* **Maintainer hypothesis to test:** split-wise expert offload uses more of the PCIe lanes in parallel, the
  way the r16 prefill split did (2-GPU `-sm tensor -ncmoe` beats upstream's only 2-GPU option at every
  offload level, +91% to +148%).  Measure achieved aggregate PCIe bandwidth, not just t/s.
* A/B the two cold policies at equal `h`: **CPU-computes-the-misses** (1a) vs **UVA in-place** (1b).  §2.3
  predicts CPU-compute wins at our hit rates because it uses ~40 GB/s of host DRAM instead of ~14.5 GB/s of
  PCIe, but this must be measured, not assumed.  Keep the loser as the fallback.

**1c — verify width + purity.**
* Score the policy at `W = 1..8` with `GGML_MOE_PROFILE_MAXTOK=8` (the MTP verify band) — the used-expert set
  per step is wider, so `h` per reach differs from §2.2's W=1 number.  Ensure the resident/cold split is
  bit-identical at every width.
* Gate: `--spec-type none` vs `--spec-type draft-mtp` same-seed text; the
  `benchmarks/mtp-adaptive-methodology.md` protocol (acceptance > ~0.45 at pos 1, MTP >= plain at depth 3).

**1d — fail-soft + `--fit`.**  Kill-switch `MOE_EXPERT_CACHE_MIB=0` (default off until measured).  A failed
arena alloc disables the cache and warns once; decide whether `--fit` learns about the arena or the exclusion
is documented.

### 3.4 Phase 2 — 2 GPUs, `-sm layer`

Whole MoE layers are assigned to one device, so the cache is per-device and a slot holds a whole expert.  This
is the moe-cache-shaped case and should be the first multi-GPU checkpoint: it validates the same dispatch
seam with the simplest geometry.  Gate: per-device `h`/`tg`, the cross-device AR unchanged
(`GGML_CUDA_ALLREDUCE=hybrid`), same-seed text vs `-ncmoe`-only.

### 3.5 Phase 3 — 2 GPUs, `-sm tensor` (the end-goal geometry)

* Each device holds **its slice** of every expert; `blob_bytes` is the axis slice and the meta backend's
  per-device partial reduce is untouched.
* The residency manager is per-device; the pointer table feeds the meta backend's simple expert tensors.
* The cold path **must** be UVA: the `ffn_down` split is `nb[1] = 176` x `524288` chunks, so a per-token H2D
  is half a million tiny copies per layer.  This is R9V's mechanism (TP-sharded masters + pinned UVA cold +
  a slot cache) with our LFRU instead of `second_touch`.
* Checkpoint: the r15 pinning and r16 staging/op-offload prefill wins must not regress; measure prefill and
  decode together.

### 3.6 Phase 4 — Qwen3.8-Flash-Next IQ4_NL (93 GiB, `qwen4exp`)

Re-validate the whole thing on the real target, including the 48x512 geometry (re-measure `h` and re-tune the
LFRU decay period there), the QSA arm gates, and the lazy/PLE path (`--lazy-mode auto`; its `llama-bench`
absolutes are not comparable without the server/pre-warm path — see §0).

### 3.7 Deferred / later (explicitly not now)

* **2-level VRAM expert cache** (protected hot tier + warm tier): a good idea, but SLRU (the two-level
  recency variant) measured within ~0.01 of LFRU at every budget, and LFRU's decaying count is already a
  soft multi-level ranking.  Revisit only if a measured miss-cost breakdown shows a distinct hot set that one
  level cannot hold; the policy simulator is ready to tune it.
* **Prefetch / early router** (moe-cache): LFRU's short decay already captures the consecutive-token reuse;
  add prefetch only if a trace shows a predictable next-step set that admission latency misses.
* **Prune the staged prefill upload to used experts** (inherited follow-up 1) — prefill, not this campaign's
  decode path; leave it with the prefill work.

### 3.8 Explicitly NOT doing (corner-avoidance)

* No cache `ggml_backend_buffer_type` in the delivery interface (see §2.3).
* No whole-expert-blob-only cache shape.
* No per-token H2D of the `ffn_down` slice.
* No `--fit`-invisible allocation and no abort on arena OOM.
* No policy port from R9V/moe-cache that §2.2 did not measure as best (`second_touch` is dominated).

**First commit of Phase 1a:** the `device_alias()`/base-pointer plumbing + the LFRU residency manager on the
current CPU cold path, single GPU, behind `MOE_EXPERT_CACHE_MIB` (default 0).  Nothing else changes until that
is measured.
