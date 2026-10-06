# ENVIRONMENT.md — environment variables that affect this repo

Scope: everything **this repo adds or repurposes**, plus the upstream variables and the non-ggml
runtime variables (HIP/ROCm) that materially affect how this delivery behaves.  Defaults are taken from
the `rdna-boosts` tree at release `v16-a55e952b8-r20`; the table was generated from the code, not from
memory (see [Maintaining this file](#maintaining-this-file)).

## How to read this

| class | meaning |
|---|---|
| **Kill-switch** | the feature is **ON by default** because it passed its gate; the variable exists to disable it for A/B, bisection or a workaround. Setting it to `0` restores the old behaviour. |
| **Opt-in** | OFF by default; the feature is not gated or not for production. |
| **Tuning** | takes a number/string; the default is the validated value. |
| **Diagnostic** | no effect on the default path — it only logs, dumps, or gates a debug helper. Never set in production. |

Two idioms are used in the code and they are **not** interchangeable:

* *presence-only*: `getenv("X") != nullptr` — **any** value (including `0`) enables it.
* *value-gated*: `getenv("X") != nullptr && atoi(...)` — `1` enables, `0` leaves it off.

The tables below say which one each variable uses, because guessing wrong is a classic time sink.

## The short answer: what you actually need

**Nothing.** Every feature the delivery ships is default-on, and this is deliberate policy (see
`AGENTS.md`, "Default-on policy"). You do **not** need to set anything for the recommended server or
`llama-cli` invocation. The variables below exist to *disable* things, to bisect a regression, or to
tune a value that has a validated default.

The only ones worth knowing by heart:

| variable | default | why you would touch it |
|---|---|---|
| `GGML_COMPUTE_BUFFER_MARGIN_PCT` | `10` (HIP only) | `0` disables the compute-buffer slack — the fastest way to reproduce the old TODO #42 abort |
| `MOE_EXPERT_CACHE_MIB` | unset = auto | `0` disables the MoE expert cache entirely (CPU expert path) |
| `GGML_CUDA_ALLREDUCE` | `hybrid` | force `internal` \| `nccl` \| `ce` for multi-GPU A/B |
| `LLAMA_DROP_COMPUTE_BUFFERS` | `llama-cli`: on / `llama-server`: off | release the wide-prefill layout for a larger arena; a server must not drop. Override either way for A/B |
| `GGML_CUDA_OP_TIMING` | off | per-op GPU timing/profiling |

---

## 1. Compute buffers and the MoE arena

### 1.1 `GGML_COMPUTE_BUFFER_MARGIN_PCT` — **tuning**, default `10`, HIP/ROCm only

Pads every **compute** buffer allocation by that percentage (`ggml-alloc.c`, via the buffer type's
`get_compute_margin_pct`).  The graph allocator sizes the compute buffer from a *measure* graph, but a
runtime graph can carry a different live-tensor set (host-expert staging, MTP taps) and need a little
more — measured +3.3 %.  Growing it is a free-then-allocate-larger, so it needs a contiguous block
**bigger than the one just released**, which fails once the free VRAM belongs to the MoE arena.  Taking
the slack up front moves the cost to before the arena is sized.

* `0` disables it — and **reproduces the original abort**, so it is the natural A/B control.
* Cost, measured on 2× R9700 (`-sm tensor -ncmoe 48`): 1275 MiB of arena residency (63.8 % → 61.5 %),
  ≈127 MiB per percentage point.
* Only the RDNA/ROCm path opts in (`#if defined(GGML_USE_HIP)`); every other backend's allocation sizes
  are unchanged.  Read once per process.
* Compute-only: model weights, the KV cache and the arena are untouched.

### 1.2 MoE expert-cache arena

| variable | default | class | notes |
|---|---|---|---|
| `MOE_EXPERT_CACHE_MIB` | unset = **auto** | tuning | sizes each device from its free VRAM. `0` disables the cache **completely** (CPU expert path) — prefer this over a tiny arena, which stands the cache-aware MoE fusions down and is ~8× slower than the cache-less path. |
| `MOE_EXPERT_CACHE_RESERVE_MIB` | `1024` | tuning | VRAM held back from the arena for the compute side / KV cache.  The old server workaround raised this to `8192`; **that is no longer needed** (the r20 slack + the fail-soft yield cover it). |
| `MOE_EXPERT_CACHE_MIN_MIB` | `0` | tuning | auto-sizing floor; below it the cache disables itself instead of building a useless arena. |
| `MOE_EXPERT_CACHE_MIN_RES_PCT` | `18` when auto | tuning | minimum residency worth keeping. |
| `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` | unset | tuning | extra reserve for auxiliary buffers. |
| `MOE_EXPERT_CACHE_SLOTS` | `0` | tuning | explicit slot count hint (0 = derive from the budget). |
| `MOE_EXPERT_CACHE_PERIOD` | `32` | tuning | promotion/rebalance period, in tokens. |
| `MOE_EXPERT_CACHE_FILL` | `1` | kill-switch | fill-on-miss. |
| `MOE_EXPERT_CACHE_TOUCH` | `2` | tuning | `admit=touch`: uses since last resident before a fill is allowed. |
| `MOE_EXPERT_CACHE_DEVMAP` | `1` | kill-switch | device-side remap (builds the slot remap on the device). |
| `MOE_EXPERT_CACHE_DEVPOLICY` | `1` | kill-switch | device-side admission policy. |
| `MOE_EXPERT_CACHE_DEVPOLICY_SPLIT` | unset | tuning | split-policy variant. |
| `MOE_EXPERT_CACHE_KSLOT` | `1` | kill-switch | resolve the slot map in the MoE ids consumer instead of a materialized remap buffer. |
| `MOE_EXPERT_CACHE_PROVISIONAL` | `1` | kill-switch | provisional eviction during a pass. |
| `MOE_EXPERT_CACHE_PREFILL_SEED` | `1` | kill-switch | seed the cache during prefill. |
| `MOE_EXPERT_CACHE_PREFILL_SEED_N` | `0` | tuning | how many prefill seeds (0 = default). |
| `GGML_MOE_GATHER_ONCE` | off | diagnostic | gather each expert once per pass. |

> The arena is the **lowest-priority** VRAM consumer: any device allocation that runs short frees the
> largest table (then the whole arena) and retries.  There is no variable for that, and it should stay
> that way.  See `ARENA-UB-TENSION.md` §13/§14.

---

## 2. Scheduler and H2D staging (block 06)

The whole-shard H2D staging path for host-resident MoE expert tables.  Defaults below are the validated
configuration.

| variable | default | class | notes |
|---|---|---|---|
| `GGML_SCHED_STAGE` | on | kill-switch (value) | master switch for the staging path. |
| `GGML_SCHED_STAGE_MODE` | unset | tuning | staging mode; the delivery uses the `stage_d2d` form (redirect is corruption-prone). |
| `GGML_SCHED_STAGE_MIN_TOKENS` | `64` | tuning | width gate floor.  Below it the copy loop's gather branch serves the input instead. |
| `GGML_SCHED_STAGE_SLOTS` | unset | tuning | ring slot count. |
| `GGML_SCHED_STAGE_MAX_MB` | unset | tuning | ring budget cap. |
| `GGML_SCHED_STAGE_TABLE_REF_MB` | `0` | tuning | `0` = width-only gate (r18 disabled the stale r7 table-size scaling; a positive value restores it for A/B). |
| `GGML_SCHED_EVENTS` | on | kill-switch | set `0` to drop the cross-device event ordering. This was the pre-r12 workaround for the `-sm layer` + host-expert garbage-prefill bug; it is **fixed** in block 06, so leave it on. |
| `GGML_SCHED_DEVGATHER` | on | kill-switch | device-gather path. |
| `GGML_SCHED_SYNC_GRAPH_INPUTS` | on | kill-switch | extra graph-input sync. |
| `GGML_SCHED_DEBUG` | unset | diagnostic | scheduler debug output. |
| `GGML_SCHED_DEBUG_REALLOC` | unset | diagnostic | log buffer reallocations. |
| `GGML_META_SPLIT_COPY` | unset | tuning | meta-backend split copy handling. |
| `GGML_STAGE_META_REDIRECT` | off | diagnostic | meta redirection instead of the explicit d2d completion. |
| `GGML_STAGE_NO_RESTORE` | off | diagnostic | skip the redirect restore (debug only — unsafe). |
| `GGML_STAGE_GATHER_SCRATCH` | off | diagnostic | force the gather scratch path. |

---

## 3. MTP / speculative decoding

| variable | default | class | notes |
|---|---|---|---|
| `MTP_DRAFT_N_UBATCH` | `512` | tuning | caps the MTP draft's encoder-injection chunk.  The draft only drafts `n_max+1` tokens, so a 512-token chunk is plenty, and the ~1.6 GiB/device the wide chunk held at `-ub 8192` goes to the MoE arena.  `0` restores the target's `-ub` (no cap); a tiny cap (e.g. 4) costs prefill speed. |
| `LLAMA_SPEC_DRAFT_N_MAX_CLAMP` | `1` = clamp on | kill-switch | clamps `--spec-draft-n-max` to 15 (the recurrent snapshot bound). `0` escapes the clamp; **purity is only promised to 7**. |
| `LLAMA_MTP_DRAFT_OP_OFFLOAD` | unset | tuning | op-offload behaviour for the draft model. |
| `LLAMA_MTP_SPARSE` / `LLAMA_MTP_SPARSE_DECODE` | unset | tuning | sparse MTP variants. |
| `GGML_LF_DFLASH_DEV` | unset | tuning | dflash speculative backend device. |

The tuned adaptive-MTP controller is selected with **`--spec-type draft-mtp-adaptive`** (a CLI flag, not
an environment variable).  Plain `--spec-type draft-mtp` does not use it.

---

## 4. Model loading and context

| variable | default | class | notes |
|---|---|---|---|
| `LLAMA_MMAP_HOST_EXPERTS` | on | kill-switch | mmap the host-resident expert tensors. |
| `LLAMA_TENSOR_HOST_BUFT` | unset | tuning | host buffer type for overridden tensors. |
| `LLAMA_DEVICE_INPUT` | off | **opt-in** | device-side input handling. |
| `LLAMA_DROP_COMPUTE_BUFFERS` | `llama-cli`: on / `llama-server`: off | tuning / kill-switch | drop the wide-prefill compute layout at the prefill→decode transition so a wide `-ub` and a large arena coexist (`-ub 8192` cache-auto cli decode 45.6 → 78.7 t/s).  Follows `common_params::drop_compute_buffers`; set the env to `0`/`1` to override either default.  **`llama-server` must leave it off**: a later wide prefill needs a contiguous ~12.4-12.9 GB layout back and the arena cannot yield one (`-ub 4096` reclaims, `-ub 8192` aborts). |
| `LLAMA_DROP_EXTRA_RESERVE_MIB` | `0` | tuning | when the drop fires, VRAM held out of the arena per device for a later compute growth.  Default `0` — the compute-buffer margin covers the small post-drop growth and the arena's layer-uniform re-size absorbs fragmentation, so a reserve only costs arena. |
| `LLAMA_LAZY_BUF_MB` / `LLAMA_LAZY_IO_THREADS` / `LLAMA_LAZY_READER_STATS` | unset | tuning / diagnostic | lazy-mode buffer size, reader thread count, reader statistics. |

---

## 5. CPU MoE offload threading

| variable | default | class | notes |
|---|---|---|---|
| `GGML_CPU_MOE_OFFLOAD_THREADS` | derived (block-06 cap) | tuning | overrides the thread cap for `-ncmoe` offloaded-MoE decode.  The cap only covers that case. |
| `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD` | off | diagnostic | disables the single-thread path for tiny graphs. |

Thread sizing is **not** an environment variable and is not optional: `/usr/local/bin/pin_gpu_irqs.sh`
puts the GPU IRQs on the top `NUM_GPUS` cores, and the default `-t` puts the worker pool there and
starves the GPU.  Size `-t` (or `--cpu-mask` with `-t` inside the mask) to leave those cores free.
See `AGENTS.md`.

---

## 6. Multi-GPU comms

| variable | default | class | notes |
|---|---|---|---|
| `GGML_CUDA_ALLREDUCE` | `hybrid` on Linux | tuning | `hybrid` (NCCL for large, internal for small) \| `internal` \| `nccl` \| `ce` (SDMA copy-engine, 2 GPU).  Unrecognised values warn and fall through to the platform default.  `nccl` is **not** bit-identical under `-sm tensor` (the internal AR BF16-round-trips), so use it as a smoke comparison only. |
| `GGML_CUDA_P2P` | off | tuning | enables peer access.  Note VMM allocations need explicit `cuMemSetAccess` for P2P (see `FOLLOWUP-compute-arena-chunking.md`). |

---

## 7. Fusion A/B switches

Each of these disables exactly one optimization so it can be bisected.  **The feature is on by default;
the variable turns it off.**  Read the idiom column carefully — setting a presence-only switch to `0`
still disables the feature.

### 7.1 `GGML_CUDA_FUSE_*` — presence of the variable with a value, `0` disables

`GGML_CUDA_FUSE_ADD_RMS_Q8`, `GGML_CUDA_FUSE_CPY_BATCH`, `GGML_CUDA_FUSE_GATE_BETA_VERIFY`,
`GGML_CUDA_FUSE_GDN_CONV_VERIFY`, `GGML_CUDA_FUSE_GDN_GATE`, `GGML_CUDA_FUSE_GLU_Q8_1`,
`GGML_CUDA_FUSE_Q8_1_VERIFY`, `GGML_CUDA_FUSE_SWIGLU_MMQ`.
These use `getenv(...) == nullptr || atoi(...) != 0`, so unset = **on** and `0` = off.
(Git history note: these were originally *opt-in* `=1` switches; they are now default-on kill-switches.)

The same idiom — and therefore also default-on despite the name — covers
`GGML_CUDA_DISABLE_IDX_RELU_SUM` and `GGML_CUDA_CONCAT_VERIFY`.

### 7.2 `GGML_CUDA_DISABLE_*` — value-gated (`=1` disables, `0` keeps it on)

`GGML_CUDA_DISABLE_FUSED_ADDMUL`, `GGML_CUDA_DISABLE_FUSION`, `GGML_CUDA_DISABLE_GDN_CPY`,
`GGML_CUDA_DISABLE_HC_FUSION`, `GGML_CUDA_DISABLE_L2_NORM_PAIR`, `GGML_CUDA_DISABLE_MOE_DOWN_FOLD`,
`GGML_CUDA_DISABLE_MOE_MMQ_FUSION`, `GGML_CUDA_DISABLE_MWR`, `GGML_CUDA_DISABLE_NORM_Q8_1`,
`GGML_CUDA_DISABLE_NORM_ROWS`, `GGML_CUDA_DISABLE_ROPE_SETROWS`, `GGML_CUDA_DISABLE_SCALE_UNARY`,
`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE`, `GGML_CUDA_DISABLE_SNAKE`, `GGML_CUDA_DISABLE_SSM_CONV_IN`,
`GGML_CUDA_DISABLE_SSM_PRESCAN`, `GGML_CUDA_DISABLE_WEIGHTED_DOWN`.

`GGML_CUDA_DISABLE_SHEXP_DOWN_GATE=1` is the A/B switch for the byte-identical fused shared-expert band
(`1 <= nt <= 8`).

### 7.3 `GGML_CUDA_DISABLE_*` — **presence-only** (any value, including `0`, disables)

`GGML_CUDA_DISABLE_CONV_FUSION`, `GGML_CUDA_DISABLE_MMID_512`, `GGML_CUDA_DISABLE_MMQ_ROUTED`,
`GGML_CUDA_DISABLE_MMVQ_DENSE_BAND`, `GGML_CUDA_DISABLE_MMVQ_MOE_BAND`,
`GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE`, `GGML_CUDA_DISABLE_ROPE_SET_ROWS`,
`GGML_CUDA_DISABLE_TOPK_MOE_FUSION`.

### 7.4 Other kernel tuning

| variable | default | notes |
|---|---|---|
| `GGML_CUDA_MMQ_J_MAX` | tuning | MMQ `J` tile width cap. |
| `GGML_Q6_COMPACT_J` | tuning | Q6 compact-`J` variant. |
| `GGML_CUDA_ENABLE_RDNA3_5_SINGLE_TOKEN_FUSIONS` | off (**opt-in**, RDNA3.5 only) | the qwen4exp weighted-down single-token chain. |
| `GGML_PAIR_OFF` / `GGML_PAIR_DENSE_OFF` / `GGML_PAIR_2X` | off | paired-kernel variants. |
| `GGML_CUDA_SPLICE_GATHER`, `GGML_CUDA_SCALE_UNARY`, `GGML_CUDA_HC_MIX_BAND`, `GGML_CUDA_HC_MIX_PREQ`, `GGML_CUDA_DISABLE_HC_COMB`, `GGML_CUDA_DISABLE_HC_MIX`, `LLAMA_FUSED_HC_MIX`, `LLAMA_FUSED_HC_COMBINE`, `LLAMA_FUSED_DSV4_HC_PRE/POST`, `LLAMA_HC_MIX_BF16`, `GGML_CUDA_LIGHTNING_INDEXER4_GFX1100/GFX1201`, `GGML_LF_FAST_TOPK`, `LLAMA_INDEXER_NOBLOCK`, `LLAMA_INDEXER_NOGROUP` | off / default-on | model-specific fused chains. A/B only. |
| `GGML_CUDA_DISABLE_VERIFY_GRAPHS`, `GGML_HIP_GRAPH_FORCE_UPDATE` | off | HIP graph-capture controls. |
| `GGML_FORCE_NO_INTEGRATED` | off | ignore integrated GPUs when picking devices. |

---

## 8. Flash attention / QSA (qwen4exp)

| variable | default | class | notes |
|---|---|---|---|
| `GGML_HIP_FA_BAND_WIDE` | on | kill-switch | `0` disables the wide band. |
| `GGML_HIP_FA_BAND_WMMA` / `GGML_HIP_FA_BAND_WMMA_SPLIT` | unset | tuning | band kernel selection. |
| `GGML_CUDA_FA_WMMA_256` / `GGML_CUDA_FA_WMMA_MAX_HEAD` | unset | tuning | WMMA tile / head caps. |
| `GGML_CUDA_FA_KV_NATIVE` / `GGML_CUDA_FA_MASK_SKIP` | unset | tuning | native-KV and mask-skip paths. |
| `GGML_CUDA_FA_STAGE_MAX_MB` | unset | tuning | FA staging budget. |
| `LLAMA_QSA_OFF` | off | kill-switch | disables QSA. |
| `LLAMA_QSA_SPARSE_FA` | on | kill-switch | `0` = the dense masked path. **This is the QSA oracle**: the `W=1..8` matrix is blind to a width-uniform corruption, so measure the perplexity ratio against `LLAMA_QSA_SPARSE_FA=0` when touching the QSA op. |
| `LLAMA_QSA_KEYS_ONLY`, `LLAMA_QSA_PLE_HOSTGATHER`, `LLAMA_QSA_DENSE_SHORTCUT`, `LLAMA_QSA_SCORE_BOUNDS`, `LLAMA_QSA_SCORE_STRIP`, `LLAMA_QSA_SCORE_WMMA`, `LLAMA_QSA_SCORE_WMMA_MB`, `GGML_CUDA_QSA_IDENTITY`, `GGML_CUDA_QSA_SLICES`, `GGML_CUDA_QSA3`, `GGML_CUDA_QSA_INDEXER_CACHE`, `GGML_CUDA_QSA_INDEXER_SCORE`, `GGML_QSA_DERIVED_BIAS`, `GGML_QSA_DERIVED_VIS`, `GGML_QSA_SCORE_MEM`, `LLAMA_KQ_MASK_DERIVED` | off / unset | tuning | QSA variants and indexer tuning. |

The QSA op has a CPU oracle: `test-backend-ops -o FLASH_ATTN_QSA` (18/18 minimum).

---

## 9. GDN / SSM (block 02)

| variable | default | class | notes |
|---|---|---|---|
| `GGML_CUDA_GDN_CHUNKED` | on | kill-switch | compared as a **string** against `"0"` — literally `GGML_CUDA_GDN_CHUNKED=0` forces the sequential kernel (correct, bit-identical, slow). The chunked prefill is K-independent with threshold `max(K>16?K:16, n_rs_batch)`. |
| `GGML_CUDA_GDN_CHUNKED_BF16` | unset | tuning | BF16 chunked variant. |
| `GDN_DBG_SKIP_KKT`, `GDN_DBG_SKIP_SCAN` | off | diagnostic | skip stages (debug). |

---

## 10. MMB (block 08)

The `mmb` campaign is part of the delivery.  `GGML_CUDA_MMB` defaults to `1`; the rest are tuning knobs
for tile/shape selection and cache sizing.  Treat them as internal: the validated configuration is the
default, and they exist to reproduce measurements from `benchmarks/`.

`GGML_CUDA_MMB`, `GGML_CUDA_MMB_BF16W`, `GGML_CUDA_MMB_CACHE`, `GGML_CUDA_MMB_CFG`,
`GGML_CUDA_MMB_DENSE`, `GGML_CUDA_MMB_DENSE_TYPES`, `GGML_CUDA_MMB_DOWN16`, `GGML_CUDA_MMB_F32SPLIT`,
`GGML_CUDA_MMB_F32SPLIT_MIN_K`, `GGML_CUDA_MMB_F32SPLIT_MIN_M`, `GGML_CUDA_MMB_GLU`,
`GGML_CUDA_MMB_GLU_THRESH`, `GGML_CUDA_MMB_HC16`, `GGML_CUDA_MMB_IQ3XXS`, `GGML_CUDA_MMB_MIN_T`,
`GGML_CUDA_MMB_RDNA3`, `GGML_CUDA_MMB_ROUTED`, `GGML_CUDA_MMB_ROUTED_THRESH`, `GGML_CUDA_MMB_SHADOW`,
`GGML_CUDA_MMB_SHADOW_MB`, `GGML_CUDA_MMB_TALL`, `GGML_CUDA_MMB_TALL_MIN_M`, `GGML_CUDA_MMB_TILE`,
`GGML_CUDA_MMB_TINY_M`, `GGML_CUDA_MMB_TINY_TT`, `GGML_CUDA_MMB_TYPES`, `GGML_CUDA_MMB_LOG`,
`GGML_CUDA_MMB_MARK_LOG`, `LLAMA_MMB_CVT_LOG`, `LLAMA_MMB_F32SPLIT`, `LLAMA_HC_BLK16`,
`LLAMA_HC_GATEMIX`, `LLAMA_HC_RES16`.

---

## 11. Diagnostics and profiling

None of these change the computed result on the default path.

| variable | notes |
|---|---|
| `GGML_CUDA_OP_TIMING` | per-op GPU timing. |
| `GGML_CUDA_GCDBG` | CUDA-graph debug. |
| `GGML_STREAMDBG` | stream debug. |
| `GGML_ALLOCATOR_DEBUG` | build-flag companion for allocator tracing. |
| `GGML_CUDA_MWR_DEBUG`, `GGML_CUDA_HC_BLK16_DEBUG`, `GGML_CUDA_IDX_RELU_SUM_LOG`, `GGML_CUDA_FUSE_CPY_BATCH_DEBUG`, `GGML_CUDA_CONV_DEBUG`, `GGML_CUDA_CONV_FUSION_MULTI`, `LLAMA_HC_CN_DEBUG`, `LLAMA_HC_GATEMIX_DEBUG`, `LLAMA_QSA_SCORE_STRIP_DEBUG`, `GGML_CUDA_QSA_DEBUG` | feature-specific debug output. |
| `FA`, `KV`, `RS`, `GDN_DBG_A`, `GDN_DBG_P`, `GDN_DBG_ONES`, `GDN_DBG_DUMP`, `GDN_DBG_SAVE` | **internal probes.** Single/double-letter or `_DBG_` names used by debug dump blocks. Never set these in production — two of them take a value that selects a dump *mode*. |

---

## 12. Non-ggml environment (HIP/ROCm runtime)

These are not llama.cpp variables but they control how the delivery runs.

| variable | notes |
|---|---|
| `HIP_VISIBLE_DEVICES` | which GPUs are visible, in order. Part of every validated command (`0,1` or `0,1,2`). |
| `LD_LIBRARY_PATH` | must include the matching ROCm runtime libs, e.g. `/opt/rocm-7.14.1-gfx120X/lib`, or the build will not start. |
| `ROCR_VISIBLE_DEVICES` | lower-level alternative to `HIP_VISIBLE_DEVICES`; prefer `HIP_VISIBLE_DEVICES`. |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | forces managed allocations; not used in this repo's validated configurations. |

There is **no** environment variable for thread pinning; use `-t` / `--cpu-mask` as described in §5.

---

## 13. Retired / superseded

| variable | status |
|---|---|
| `MOE_EXPERT_CACHE_RESERVE_MIB=8192` | the pre-r20 server workaround. **No longer needed** — the default `1024` plus the r20 compute-buffer slack and the fail-soft arena yield cover it. |
| `MOE_ARENA_HEADROOM_MIB` | tried and **removed**. A flat arena headroom changes nothing: the failure is a realloc needing a block bigger than the one just freed, so the slack has to be on the *allocation*, not the arena. |
| `MTP_DRAFT_N_UBATCH=4` | was a workaround for the same failure; a cap is now the default (`512`), so this is just an over-tight value. |
| `GGML_HIP_NO_VMM` (build option, not runtime) | HIP VMM is **off by default** (`ON` = do not use VMM).  Whether to enable it is an open investigation — see `wip/moe-cache-autosize/FOLLOWUP-compute-arena-chunking.md`. |

---

## Maintaining this file

The tables were derived from the code, so re-derive rather than trust them after a block change:

```bash
# every variable the delivery adds or repoints, with the idiom used
for f in ~/llama-cpp-rdna-boosts/patches/*.patch; do
  grep -hoE 'getenv *\( *"[A-Za-z0-9_]+"|GGML_ENV_STR *\( *"[A-Za-z0-9_]+"|env_int *\( *"[A-Za-z0-9_]+"' "$f"
done | grep -oE '"[A-Za-z0-9_]+"' | tr -d '"' | sort -u
```

Then, for each name, grep the tree for its usage.  **Do not** classify by name alone: `*_DISABLE_*` is
not consistently value-gated, and `*_FUSE_*` semantics changed from opt-in to kill-switch.  A variable
that is genuinely new should follow the repo's conventions: default-on if it helps (with the variable as
a *kill-switch* that restores the old behaviour), default-off only for a known-risky correctness
workaround, and a first line in the relevant block's `patches/README.md` note.
