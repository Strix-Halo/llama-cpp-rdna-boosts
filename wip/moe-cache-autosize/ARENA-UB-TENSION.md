# TODO #42 -- the arena-vs-`-ub` tension, and the 2-GPU `-ub 8192` OOM
**Status (r19, 2026-10-06): the prefill work shipped in r17/r18; the Stage-2 arena-safety work
shipped in r19 as a fail-soft guard (see SS12.8) -- the partial shrink is the one open item, with a
dedicated handover in `wip/moe-cache-autosize/HANDOVER-unredirect.md`.**
**Status: prefill SHIPPED (items 1-3, r17/r18); Stage 2 (the cache-auto `-ub 8192` OOM + arena
shrink) has a VALIDATED candidate in §12 -- decode 75.5 t/s, no OOM.  Remaining: the reserve fix.**
**Read the HANDOVER BRIEF below first.**

## HANDOVER BRIEF (cold start)

**Scope:** TODO #42.  Stage-1 prefill is done (items 1-3 shipped): the remaining work is the **Stage-1
base-gate follow-up** (§11.4: the width-only gate may itself be too high for big tables) and **Stage 2**
(the 2-GPU cache-auto `-ub 8192` OOM + the arena shrink/reclaim, decode).  Root cause and history are
§0-§4 / §8-§10; the Stage-1 tracker is §11.

**Current state (release `v16-a55e952b8-r18`, 2026-10-06):**
* Delivery tree `1df33ad45fa5f8474269c590a6ead50369af12c3` (16 blocks).  **Items 1-3 are all IN the
delivery**: item 2 (pinned 2-D H2D for the split staging slice) + r16's blocks 16/17 folded into block
15, item 1 (per-pass gather guard) into block 13, and item 3 (width-only staging gate; the stale table-
size scaling disabled) into block 06.  The staging **redirect stays off** (`stage_d2d`).  The
`stage1-item*.patch` files here are superseded by `patches/` and kept as provenance.
* Scratch checkout `~/llama.cpp` on branch `rdna-boosts`; binary `build-rocm-r16/bin/llama-cli`.
  Build: `cd ~/llama.cpp && BUILD_DIR=build-rocm-r16 EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`;
  fast loop `cmake --build build-rocm-r16 --target llama-cli -j 16`.
* Box: 2 x R9700 (gfx1201, 32 GB), plus a third for the 3-GPU gates.  Model:
  `/llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf` + the shared
  MTP head `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`.  16k prompt `/tmp/pl_16k.txt` (rebuild:
  `prompts/code-python.txt` x30 + a short instruction); 32k = x60.  Flags unless noted:
  `-sm tensor -ncmoe 48 -fa on -ctk q8_0 -ctv q8_0 -t 8 -c 32768 -b/-ub 8192 --lazy-mode off
  --load-mode auto --spec-type draft-mtp --spec-draft-n-max 3 --temp 0 --seed 42 --reasoning off
  --single-turn --no-display-prompt`, `MOE_EXPERT_CACHE_MIB=0` for the prefill work (cache auto OOMs at
  `-ub 8192` until Stage 2 lands).

**Task 1 -- Stage-1 item 3 (width gate, prefill).**  Goal: recover the ~**+16 %** that forcing every
split gives: at `-ub 8192`/16k/cache-off the split prefill is **1530 t/s default** vs **1775 t/s forced**
(`GGML_SCHED_STAGE=1 GGML_SCHED_STAGE_MIN_TOKENS=1`).  The gate is `sched_stage_min_tokens_for_bytes`
in `ggml-backend.cpp`: `base * table_bytes / SCHED_STAGE_TABLE_REF_BYTES(144 MiB)`, which overshoots for
the large IQ4_XS tables (so some splits never stage).  A/B it against the forced/gate-off runs; gates:
coherence `////`=0 on a **long multi-ubatch** prefill (32k), byte-identity vs the old path, 3-GPU, and
the `-ncmoe 0` gate model.

**Task 2 -- Stage 2 (decode): DONE in §12 (candidate).**  Goal (the original DoD): 2 GPU, **cache auto**,
`-ub 8192`, 16k prefill then 2000 decode, **no OOM**, decode **>= 68.9**.  Met: **prefill 1639.9 /
decode 75.5 t/s**, arena 38686 MiB (68.2 %), MTP acc 0.9245, coherent.  Design (patch
`stage2-arena-shrink.patch`): drop the wide-prefill compute layout at the prefill -> decode transition,
re-reserve the verify width (`cparams.n_rs_batch`), hold that layout out of the arena
(`moe_cache_set_extra_reserve`), and cap the MTP draft's `n_ubatch` to 512 (`MTP_DRAFT_N_UBATCH`).
**Root cause of the OOM is a reserve miscalculation** (reserve 11339 vs first-prefill 11765, a
fragmentation-sensitive in-place realloc); the follow-up is to make the reserve a true upper bound
(then the draft cap can go) + the arena-reclaim retry.  Full detail: §12.

**Env knobs:** `GGML_SCHED_STAGE`, `GGML_SCHED_STAGE_MIN_TOKENS`, `GGML_STAGE_GATHER_SCRATCH` (force the
old scratch path), `GGML_MOE_GATHER_ONCE` (restore the once-only gather guard), `GGML_SCHED_DEVGATHER`,
`GGML_META_SPLIT_COPY` (`0` = mirrored), `MOE_EXPERT_CACHE_MIB`, `MOE_EXPERT_CACHE_RESERVE_MIB`,
`MTP_DRAFT_N_UBATCH` (MTP draft encoder-injection chunk, default 512),
`GGML_ALLOCATOR_DEBUG` (gallocr per-chunk `max_size` tensor dump -- the tool that named the SS0 root cause).

**Hard rules:** `llama-cli` always `--single-turn`; size `-t` to leave the GPU IRQ cores (13/14/15 on
3 GPUs) free -> use `-t 8`; **never run benches in parallel** (`pgrep -af '[b]in/llama'` first); every
speed claim must be coherence-gated (`////`=0) **and** MTP-gated (`-n 2000` floor, `benchmarks/
mtp-adaptive-methodology.md`); surface findings to the maintainer before any delivery commit/release.

## 0. Headline: the reported OOM is NOT the arena

Reproduced with the task's exact `llama-cli` command (`-n 2000` and `-n 4` both crash identically, at
the **first prefill ubatch**, ~0.24 s, before any MoE arena exists):

```
ggml_gallocr_reserve_n_impl: reallocating Meta() buffer from size 11339.14 MiB to 11765.52 MiB
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 11765.52 MiB on device 0: cudaMalloc failed:
    out of memory (free=10538.0 total=32624.0 MiB)
ggml-backend-meta.cpp:1799: GGML_ASSERT(bufs.back() != nullptr) failed
```

There is **no `alloc_all_locked` / arena line before the crash** (with `-lv 5`, logged) -- the arena
is sized later, on the second decode-band MoE access, and the run never reaches it.  The task's
"after the arena is allocated" framing does not hold for this `llama-cli` repro; the OOM is the
target context's own **grow-only compute buffer** needing +426 MiB at the first prefill and the device
having only 10538 MiB free after the old 11339 MiB buffer is freed.

The arena's role is the *decode* half (SS2), not the crash.

## 1. Why the reserve is 426 MiB short (cache-on only)

`cache off` (`MOE_EXPERT_CACHE_MIB=0`) runs clean: the target's runtime peak is **exactly the
reserved 11339.14 MiB**.  `cache auto` has the *same* reserve (11339.14) but a runtime peak of
**11765.52 MiB**.

Instrumented the gallocr (`GGML_ALLOCATOR_DEBUG`, now reverted) and diffed the reserve event against
the runtime peak event.  The extra tensors are the **last layers**: the runtime peak carries
`attn_output-47`, `ffn_moe_*-47`, `hc_mixed-47`, `Meta(...)#blk.47.ffn_down_exps.weight#0` (850 MB),
`linear_attn_out-46`, ...; the reserve's peak was reached earlier (layer 45) and packed the tail into
freed space.  So the reserve graph (`memory->init_full()` + a synthetic ubatch) and the real first
prefill graph lay the tail out differently; cache-on shifts the tail past the reserve bound.

Ruled out: the derived kq mask (`resolve_fused_ops` disables it for this model -- "derived kq mask
flash attention was not used in the probe graph" -- so both reserve and runtime use the packed mask);
the MoE cache fusions (prefill `ne[2] > band` keeps them on in both); the arena (absent).

Net: **the reserved pp graph is not a true upper bound for the first runtime prefill** on this
hybrid-memory model, by ~426 MiB, and only the cache-on layout exposes it.

## 2. The wide-prefill layout permanently starves the arena (the real #42 tension)

Once the run survives (SS3), the arena is sized from `free - reserve` at the second decode-band
access, i.e. around the already-grown 11765 MiB compute buffer:

| 2 GPU, cache auto, 16k prefill | compute (MiB) | arena (MiB) | residency | decode (t/s) |
|---|---:|---:|---:|---:|
| `-ub 4096` (baseline) | 6782 | 30820 | 55.1 % | **68.9** |
| `-ub 8192` (after SS3 fix) | 11765 | 21175 | 38.0 % | **41.0** |

The ~5 GiB/device of extra prefill layout is the whole arena difference.  The compute buffer is
grow-only and never returns the prefill-only portion, so a wide `-ub` and a large arena cannot
coexist -- exactly TODO #42.

(Secondary: cache-on 16k `-ub 8192` prefill measured **871-878 t/s** vs **1094 t/s** cache-off at the
same point -- 16-20 % slower.  Not investigated this session; worth a follow-up.)

## 3. Validated remedy for the OOM: cap the MTP draft context's `n_ubatch`

The MTP draft context was reserving its compute buffers for the target's full `-ub`: **1696.95 MiB on
each device** (it only drafts `n_max+1` tokens per step; the encoder injection is already fed in
chunks).  That resident 1.7 GiB is what makes the target's 11765 MiB growth miss.

`wip/moe-cache-autosize/arena-ub-tension.diff` (scoped to `spec_mtp`):

```cpp
if (spec_mtp) {
    cparams.n_ubatch = std::max(1, params.speculative.draft.n_max + 1);
}
```

Measured (full `-n 2000` run, `-lv 4`):

| config | prefill | decode | coherent | MTP acc | mean len |
|---|---:|---:|---|---:|---:|
| 2 GPU, cache auto, `-ub 8192`, this fix | **871.1** | **41.0** | `////`=0 | 0.917 | 3.75 |

The target's 11339 -> 11765 growth now succeeds; the run is coherent and MTP-healthy.  **It fixes the
OOM but not the DoD decode target** (41.0 < 68.9), because of SS2.

## 4. Partial remedy for the decode half, and the wall it hits

Added a scheduler "drop the grow-only compute buffers" primitive
(`ggml_gallocr_drop_buffers` / `ggml_backend_sched_drop_buffers`) and called it on the target's
prefill -> decode transition (`n_tokens_prev > 8 && ubatch.n_tokens <= 8`).  The first decode then
re-reserves a tiny layout and the arena grows:

| 2 GPU, cache auto, `-ub 8192` | arena | residency |
|---|---:|---:|
| without the drop | 21175 MiB | 38.0 % |
| **with the drop** | **41352 MiB** | **73.4 %** |

But the full `-n 2000` run then **aborts at the first MTP verify**: the target's verify graph needs a
**1700.73 MiB** compute layout the single-token decode did not, and the arena already took the freed
space:

```
reallocating Meta() buffer from size 1700.73 MiB to 1700.73 MiB
cudaMalloc failed: out of memory (free=868.0 total=32624.0 MiB)
```

So the shrink, as hooked, reserves the *decode* layout while the arena sizes before the *verify*
layout is known.  A server would hit the same growth on any later prefill (the buffer is grow-only and
the arena is a first-come permanent reservation).

### What the full DoD needs (plan; not implemented)

1. **Shrink to `max(decode, verify)` before the arena is sized.**  Reserve the target's post-prefill
   compute layout for `n_max+1` tokens (the verify batch), not the single-token tg graph, then let
   `alloc_all_locked` size the arena around it.  ~1700 MiB/device compute + a >=55 % arena is then
   possible in one config.
2. **Make the arena yield on any later compute growth** (fail-soft ordering, TODO #38): give
   `ggml_backend_cuda_buffer_type_alloc_buffer` a retry path that calls a new
   `moe_cache_release_arena()` (free every arena + disable the cache for the run, with a WARN) before
   giving up.  The arena is the lowest-priority consumer; a smaller/absent arena is strictly better
   than a crash.  This is also the server-safe guard for (1).
3. Re-run the DoD + 3-GPU and `-ncmoe 0` byte-identity regressions after (1)+(2).

## 5. Method notes

- The OOM is at the first *prefill*, so `-n 4` and `-n 2000` reproduce it identically and fast; no
  need to run a full decode to see it.
- The gallocr's per-chunk `max_size` debug (`GGML_ALLOCATOR_DEBUG`) is the useful instrument: it names
  exactly which tensors are live at the peak.  It is verbose -- redirect and grep.
- `GGML_MTP_DRAFT_OP_OFFLOAD=0` (the existing `wip/mtp-draft-op-offload/` knob) does **not** change the
  draft reserve here (still 1696.95 MiB); it is not a workaround for this OOM.
- `free=` in the alloc-failure path (instrumented, reverted) is the number of bytes the allocator
  believes are free after the old buffer was freed; fragmentation matters (a later run failed a
  11765 MiB malloc with `free=21972`).

## 6. Files

- `arena-ub-tension.diff` -- the session's candidate changes (MTP `n_ubatch` cap + draft lazy drop +
  `ggml_gallocr_drop_buffers` / `ggml_backend_sched_drop_buffers` + the transition hook + the
  `llama_context_drop_compute_buffers` API).  **Experimental; not promoted.**
- `TENSOR-CORRUPTION.md` SS8 and `PREFILL-WALL.md` SS3 are the prior context for this tension.

## 7. Follow-up: `-sm layer` on the same box (2 GPU, cache auto, 16k prefill + `-n 2000`)

Measured on the same clean r16 build, same prompt/seed, `-t 8`, all coherent (`////`=0):

| config | prefill t/s | decode t/s | arena (MiB) | residency | hit | MTP acc (mean len) |
|---|---:|---:|---:|---:|---:|---:|
| `-sm tensor -ub 4096` | 711.6 | **69.1** | 31265 | 55.1 % | 0.936 | 0.916 (3.74) |
| `-sm layer  -ub 4096` | 735.5 | 56.7 | 37180 | 65.5 % | 0.971 | 0.926 (3.78) |
| `-sm layer  -ub 8192` | **1409.4** | 43.9 | 27311 | 48.1 % | 0.924 | 0.918 (3.75) |
| `-sm tensor -ub 8192` (stock) | OOM | -- | -- | -- | -- | crash |

Reading:

* `-sm layer` is **not** affected by the `-sm tensor` OOM (no Meta-backend compute buffer), and at
  `-ub 8192` it is now the fastest prefill measured on this box (**1409 t/s**, vs the r13 record's
  1040).  The corruption work did not regress it; `-ub 4096` is healthy (acc 0.926).
* It has a real **decode penalty**: at `-ub 4096` it has a bigger arena (37180 vs 31265) and a higher
  hit rate (0.971 vs 0.936) yet decodes 18 % slower (56.7 vs 69.1).  The layer-split pipeline cost
  outweighs the cache advantage.  `-sm tensor -ub 4096` remains the decode champion.
* The arena-vs-`-ub` tension is identical under layer split (48.1 % at `-ub 8192` vs 65.5 % at
  `-ub 4096`); layer split only removes the crash.  If a 2-GPU production config can be `-sm layer`,
  the wide-prefill config is already available with **no code change** (1409 / 43.9), and the SS3
  `-sm tensor` fix matters only for tensor-split users.

Single-run numbers; repeat before quoting.

## 8. The staging ring / redirect, re-tested on r16 (is the r12 prefill trick recoverable?)

Prompt: match/beat the `-sm layer` prefill (1409 t/s at `-ub 8192`) with `-sm tensor`; the hypothesis
was that block 16 disabled the redirect (`stage_d2d` default) because of a misdiagnosed corruption, and
that block 17's padding/guard fix had actually been the cause.  Clean r16 build, 2 GPU, `-sm tensor
-ncmoe 48`, cache off (to dodge the #42 OOM), 16k prompt, `-n 4`:

| staging config | prefill t/s | `////` |
|---|---:|---|
| staging off (`GGML_SCHED_STAGE=0`) | 997.0 | 0 |
| default gate (no env) | 1097.7 | 0 |
| D2D forced (`STAGE=1 MIN_TOKENS=1`) | 1171.0 / 1175.9 | 0 |
| redirect, default gate (`GGML_STAGE_META_REDIRECT=1`) | 1560.2 | **1** |
| redirect, forced | 1715.5 | **1** |
| **redirect, forced, NO restore (`GGML_STAGE_NO_RESTORE=1`)** | **1182.9** | **0** |

Reading:

* The redirect's 1560-1715 is the **stale-pointer less-work artifact**: once the restore is removed
  (the only coherent redirect), it is **1182.9**, i.e. identical to the D2D copy.  So re-enabling the
  redirect buys **no real prefill speed**; the restore timing is the defect, and the D2D copy is the
  correct form (as the r12 notes already measured: `MODE=0` ~3.5 % on our box).
* The block-17 padding fix is the **separate** #43 pruned-upload guard; it does not make the redirect
  safe (the redirect path never writes that guard).  The two were shipped together in r16, which is
  probably why they got conflated.
* Forcing the width gate on (`MIN_TOKENS=1`) is worth ~+7 % over the default gate (1176 vs 1098);
  staging off is 997.  So the real staging win at `-ub 8192` is ~**+18 %** (997 -> 1176).
* The coherent tensor ceiling (~1180) is still below `-sm layer`'s 1409 (which does **not** stage --
  layer staging aborts).  That gap is the split mode (mirrored expert upload + Meta partition +
  all-reduce), matching the r12 record (35B Q4_K_M pp8192/ub8192: tensor 2742 vs layer 4111).  **The
  redirect is not the lever** for matching layer prefill.

Open (stage-1) candidates, none measured yet: (a) whether the cache-on prefill penalty (871 cache-on
vs ~1094 cache-off, possibly the WIP MTP `n_ubatch` cap chunking the draft encoder injection rather
than the cache itself) is recoverable; (b) the default width gate's per-split scaling (forcing all
splits is +7 %); (c) a structurally cheaper mirrored-expert prefill.

### 8.1 Decision (maintainer, this session): staging mode = `stage_d2d` (keep the default)

The two coherent ~1176 t/s options are `stage_d2d` and `redirect + GGML_STAGE_NO_RESTORE=1`.  They use
the **same** ring (auto-sized `GGML_SCHED_STAGE_MAX_MB`) and the same scheduler `input_cpy` reservation;
the redirect only skips the copy, it does not free a buffer, so there is **no VRAM difference**.  With
no memory advantage, the tie-break is corruption history: the redirect is the more error-prone path
(and `NO_RESTORE` is a diagnostic, not a correct cleanup).  **Keep `stage_d2d`** (the block-16 default);
do **not** re-enable the redirect, and drop it from the #42 plan.

Corollaries for the next step:

* The prefill lever is **not** the staging mode.  Work the `stage_d2d` path.
* The default width gate leaves real staging wins un-staged for the large IQ4_XS tables (~+7 % at
  `-ub 8192` when every split is forced: 1176 vs 1098) -- the gate scales by `table_bytes / 144 MiB`,
  which for these tables overshoots the batch and turns staging off exactly where the whole-table
  copy amortizes best.  Fixing the gate is a cheap, redirect-free stage-1 win.
* The dominant tensor-vs-layer prefill gap remains structural (mirrored expert upload + Meta partition
  + all-reduce): coherent tensor ~1176-1183 vs `-sm layer` 1409.  The cache-on penalty (871 vs ~1094,
  with the WIP MTP cap in play) is the other candidate; both belong in a separate stage-1 session.

## 9. Stage 1: split vs mirrored expert copies (the "why are we mirroring" correction)

**Correction to §8:** the current tree does **not** mirror the MoE experts.  `GGML_META_SPLIT_COPY`
defaults to **1** (split; `0` = the old r12 mirrored behaviour) -- `src/llama-model.cpp` strips the
`<backend>#<name>#<c>` wrapper so the offload copy inherits the weight's split state (gate/up axis 1,
down axis 0).  The `tensor-split-expert-split` campaign (closed 2026-09-27) shipped this as the prefill
fast path.  §8's "mirrored expert upload" was stale.

**Measured on the current build** (2 GPU, `-sm tensor -ncmoe 48`, 16k prompt, `-ub 8192`, cache **off**,
`-n 4`, all coherent):

| config | staging off | default gate | staging forced |
|---|---:|---:|---:|
| split (default, `GGML_META_SPLIT_COPY=1`) | 964.9 | 1038.9 | 1172.0 |
| mirrored (`GGML_META_SPLIT_COPY=0`) | 1049.6 | -- | **1405.2** |
| `-sm layer` (cache auto, from §7) | -- | 1409.4 | -- |

So on *this* model both **mirrored (1405)** and **layer (1409)** beat the default **split (1039-1172)**.

### Why mirroring can win even though it uploads more

Instrumented `ggml_backend_cuda_stage_gather` (`GGML_STAGE_GATHER_DEBUG`, reverted) on the split run.
The per-tensor gathers are:

| tensor | width/stride/`n_copies` | total (device slice) | whole (range) | path |
|---|---|---|---:|---:|---|
| `ffn_gate_exps` | 281600 / 704000 / **512** | 137.5 MB | 343.8 MB | **host gather** (slice only) |
| `ffn_up_exps` | 422400 / 704000 / **512** | 206.2 MB | 343.8 MB | **host gather** (slice only) |
| `ffn_down_exps` | 408 / 680 / **1 310 720** | 510 MB | **850 MB** | **scratch + D2D** (whole range) |

* `ffn_{gate,up}_exps` split on axis 1 gives one block per **expert** (512 <= the 4096 host-gather
  bound), so the device H2Ds only its slice -- a real ~half saving.
* `ffn_down_exps` split on axis 0 gives **1.31 M** tiny blocks (> 4096, ~500 k+ per layer as the
  block-06 r13 note recorded), so `stage_gather` falls back to: H2D the **entire contiguous range**
  (850 MB, i.e. the same volume as the mirrored copy) into device scratch, then a D2D 2-D compaction
  down to the slice (510 MB).  So **the largest tensor saves nothing** and adds a D2D.
* The split's down output is `PARTIAL` -> an extra cross-device reduction per MoE layer.

Net: "mirroring sends more H2D" holds only for gate/up; for the dominant down projection the split
uploads the same whole-table volume **plus** a compaction, and it pays a per-layer partial reduce.  On
a 2-GPU box where prefill is upload-bound that is why mirrored (single contiguous async 1-D H2D,
no reduce) wins.

### Stage-1 candidates (redirect-free, `stage_d2d`)

1. **Width-gate scaling** (cheap): split default 1039 -> forced 1172 (+13 %); mirror likewise leaves
   staging un-staged.  Correct the `table_bytes / 144 MiB` scaling for the large IQ4_XS tables.
2. **Make the down split upload only its slice**: the source weight is **pinned** (block-06 pinning), so
   a *pinned* `cudaMemcpy2DAsync` H2D straight into the slot (width/stride_src) would move the slice
   only, avoiding both the 1.31 M-block host gather and the whole-range scratch.  (The campaign's
   pageable 2-D H2D was the pathological case; pinned was not tried here.)
3. **Reconcile with the campaign**: on the 35B Q4_K_M (current build) mirrored also edges split
   (3551.9 vs 3426.2), whereas the campaign measured split >> mirrored.  The staged split fast path
   looks regressed after the re-base, or geometry-dependent -- a separate investigation.

## 10. Is `GGML_SCHED_DEVGATHER=1` coherent now that r16 fixed the corruptions?

**No -- it is a different corruption.**  The device gather was not disabled for the r16 bugs (block 16
staging redirect, block 17 per-device input copy).  It was disabled by the **r31** work
(`wip/moe-mmq-overread/RESOLUTION.md`, `v16-84e76d8a2-r31`): its *one-time* expert-head zero
(`moe_cache_get`, keyed on `(weight_cpy->data, expert_bytes)`) does **not survive a multi-ubatch
prefill** -- the graph allocator reuses the `input_cpy` region between ubatches/tables, so the MMQ tail
over-read reads stale NaN and NaN*0 poisons the tile.  The apparent 2-4x prefill win (qwen4exp
2650-3060 t/s) was routing skipping expert work, not speed.  r31 fixed only the cache-arena head zero
(Hole A, present here: `alloc_table_locked` `+head_pad`/`cudaMemset`) and **defaulted the gather off**
(Hole B); `GGML_SCHED_DEVGATHER=1` is an A/B switch.

Current tree: `sched->devgather_enabled` defaults off; `sched_input_gatherable` still lets the gather run
**above** the staging gate for a table >= 224 MiB, so IQ4_XS's 850 MB down table is eligible.

Measured on the current build (2 GPU, `-sm tensor -ncmoe 48`, 16k = 2 ubatches, cache off, `-n 4`):

| config | prefill t/s | `////` |
|---|---:|---|
| staging (default) | 1098.3 | 0 |
| `GGML_SCHED_DEVGATHER=1` | 1174.6 | 0 |

So it *happened* to be coherent in this 2-ubatch run and **+7 %**, but that is not proof: Hole B is
`input_cpy`-reuse dependent and the guard is still once-only.  r31's own conclusion is that the gather
"is not reliable for a multi-ubatch prefill until its destination is made persistent/never-reused", and
that the *correct* gather loses to staging on a fast link (~35 %, reporter 2026-10-04); on our x4 box
the r31 correct-speed range was 682-1532 vs staging 655-1398.

**Why this matters for Stage 1:** the device gather copies only the **routed** experts (pruned), which is
exactly what would fix the split down-projection's whole-range scratch upload (§9).  So "make the gather
correct" is a strong Stage-1 candidate: the fix is to re-arm the finite-head guard per gather (or persist
the destination so it is never reused), not the r16 regressions.  Validate on a long multi-ubatch prefill
before trusting it.

## 11. STAGE 1 PLAN (prefill, `stage_d2d`, tracker)

Goal: make coherent `-sm tensor -ub 8192` prefill match/beat `-sm layer` (1409 t/s) **without**
re-enabling the redirect.  Baseline: split default 1039-1172, mirrored 1405, layer 1409 (SS7/SS9).

| # | item | rationale | status |
|---|---|---|---|
| **1** | **Correct + persistent-guard device gather** | the gather copies only the **routed** experts, so it avoids the split's whole-range down scratch (SS9) *and* the mirrored full-table copy.  Blocker is r31 Hole B: the finite-head zero in `moe_cache_get` is once-only keyed on `(weight_cpy->data, expert_bytes)`, so a reused `input_cpy` loses it.  Fix = re-arm the guard per gather (or make the destination never-reused), then validate on a long multi-ubatch prefill. | **DONE -- shipped in r17** (SS11.1) |
| **2** | Pinned 2-D H2D for the down slice | the down split slice is `width`/`stride_src` strided; the source is pinned, so a pinned `cudaMemcpy2DAsync` straight into the slot would move only the slice (no 1.31 M-block host gather, no whole-range scratch). | **DONE -- shipped in r17** (SS11.2) |
| **3** | Width-gate scaling fix | the r7 `table_bytes / 144 MiB` size scaling is stale (fitted while the ring-budget bug was still disabling the ring).  Re-validated: staging beats the serial path at every `-ub` 1024..8192 for both 450 MiB and 850 MiB tables, so the scaling is a pure loss (up to +59 % recovered by disabling it). | **DONE -- shipped in r18** (SS11.4) |

All three keep `stage_d2d` (redirect stays off, SS8.1).  Gate for every item: same-seed coherence
(`////`=0) **on a long multi-ubatch prefill**, and the 3-GPU + `-ncmoe 0` byte-identity regressions.

### 11.1 Item 1 result: the gather guard is now re-armed per pass (free), but the gather is not the prefill lever

Implemented (`ggml/src/ggml-cuda/moe-expert-cache.cu`): the finite-head zero in `moe_cache_get` is now
run on **every** gather instead of once per `(weight_cpy->data, expert_bytes)`; `GGML_MOE_GATHER_ONCE=1`
restores the old arm for A/B.  Kept in the tree as the item-1 candidate.

Measured on the current build (2 GPU, `-sm tensor -ncmoe 48`, cache off):

| config | 16k (2 ub) | 32k (4 ub) | `////` |
|---|---:|---:|---|
| staged split (gather off) | 1089-1098 | -- | 0 |
| gather, once-only guard | 1174.2 | 1255.1 | 0 |
| **gather, always-zero guard** | **1175.0** | **1255.2** | 0 |
| mirrored, staged | 1405.2 | -- | 0 |

* The per-pass guard is **free** (1175.0 vs 1174.2; 1255.2 vs 1255.1), so the "3x cost" in the old
  comment was a different (per-routed-expert copy) form, not this `n_experts x head_pad` memset.
* **Hole B did not reproduce here**: the once-only arm was already coherent at both 16k and 32k, on the
  `-sm tensor` big-table gather.  The r31 repro was qwen4exp `-sm layer -ncmoe 48` on one R9700; until
  that (or another) repro is found on this base, the re-arm is **correct-by-construction / defensive**,
  not a demonstrated fix.  Kept anyway: it removes a reuse-dependent silent-corruption class for free.
* **The gather is not the prefill lever.**  Re-armed gather 1175 ~= staged split 1172, both well below
  mirrored 1405.  So a correct pruned upload does **not** beat the mirrored whole-table 1-D H2D at
  `-ub 8192` on this box (matches r31: correct gather ~4-10 % over staging at best).

Conclusion: item 1 is done (correct + free) but delivers no prefill win.  **Item 2 (pinned 2-D H2D for
the split down slice) is the one that can halve the H2D** -- mirror per device gates+up+down =
344+344+850 = 1538 MB vs a correct split slice 137+206+510 = 853 MB -- so it is the item most likely to
push a coherent split past mirrored (1405) and toward layer (1409).  Recommend item 2 next; item 3
(+13 %) as the cheap warm-up.

### 11.2 Item 2 result: pinned 2-D H2D for the split slice -- BIG WIN (beats mirrored and layer)

Implemented (`ggml-backend-impl.h`, `ggml-cuda.cu`, `ggml-backend-meta.cpp`): `stage_gather` gains a
`src_pinned` flag; when the source weight is pinned (a GPU host buft, not the pageable mmap) and
`GGML_STAGE_GATHER_SCRATCH` is unset, the compacted slice is copied with **one `cudaMemcpy2DAsync`
H2D** (`dst,width <- src+offset,stride_src, width x n_copies`), moving only `total = width*n_copies`
bytes instead of the whole contiguous range + a device 2-D compaction.  `GGML_STAGE_GATHER_SCRATCH=1`
forces the old path.  `src_pinned` is computed in `ggml_backend_meta_stage_input` from the input
buffer's buft device (GPU buft => pinned; CPU/mmap buft => pageable).

Measured (2 GPU, `-sm tensor -ncmoe 48`, cache off, `stage_d2d`, coherent `////`=0):

| config | prefill t/s | vs before item 2 |
|---|---:|---:|
| split, **default gate** | **1530.1** | 1038.9 -> +47 % |
| split, staging forced | **1774.8** (16k) / **1961.7** (32k) | 1172 -> +51 / +67 % |
| split, forced scratch (old path) | 1169.6 (16k) / 1311.2 (32k) | baseline |
| mirrored, staged | 1405.2 | -- |
| `-sm layer`, cache auto | 1409.4 | -- |

* **The generated text is byte-identical** between the pinned 2-D H2D, the host/scratch path and the
  no-gather staging path (only the timing line differs), at 16k and 32k.
* 3 GPU split forced: **1718.7 t/s**, coherent.  (`-ncmoe 0` on IQ4_XS aborts with a 11230 MiB
  `cudaMalloc` OOM -- the 96 GB model does not fit 3 x 32 GB with `-ub 8192`; the item-2 path is inert
  without host experts, so use the smaller `-ncmoe 0` gate model.)
* The win grows with prefill length (16k 1775 -> 32k 1962), i.e. it is the H2D cut it should be:
  the split's down slice is now `width*n_copies` instead of the whole 850 MB range.

**Stage-1 prefill goal met and exceeded:** split default 1530 / forced 1775-1962, above mirrored 1405
and layer 1409, redirect still off.  Next: item 3 (width-gate, +? now) as a cheap follow-up, then fold
item 2 into the decode-stage work (the cache-auto `-ub 8192` OOM is still the separate #42 item).

### 11.3 PROMOTED (2026-10-06, `v16-a55e952b8-r17`)

Item 2 (pinned 2-D H2D for the split slice) and the defensive per-pass gather guard (item 1) are
**promoted to the delivery**, folded into the existing blocks (the staging win into block 15, the guard
into block 13) so the set is back to **16 patches** (`0000`-`0015`).  r16's blocks 16 + 17 are folded
into block 15 as well.  Release tree `04764deb8322d77029060ff37d265d1dbc7a799f`; `validate-set.sh` green
(16/16 strict `git am` on a fresh tarball).  The two `wip/moe-cache-autosize/stage1-item*.patch` files
here are superseded by `patches/` and kept only as provenance.  Remaining: item 3 (width gate) and
stage 2 (cache-auto `-ub 8192` OOM + arena shrink/reclaim).

### 11.4 Item 3 result: the r7 table-size scaling is stale -- DISABLE it (candidate; beats old default at every width)

**Finding: the `table_bytes / 144 MiB` width-gate scaling (r7, issue #93) never helps on this box and
costs up to +59 % prefill.**  r7 shipped the size scaling in the *same change* as the ring auto-budget
fix, while the old fixed 2048 MiB budget was still disabling the ring mid-run: the "staging loses"
data point (450 MiB, `-ub 4096`, 1071 vs 1142) it was fitted to was therefore confounded by a
partially/fully disabled ring.  Re-validated on gfx1201 x4 with the ring fix, the pinned 2-D H2D
(item 2) and the per-pass gather guard (item 1) all in place.

Method: 2 GPU (`HIP_VISIBLE_DEVICES=0,1`, device count asserted from the log), `-ncmoe 48`, cache off
(`MOE_EXPERT_CACHE_MIB=0`), 16k `/tmp/pl_16k.txt`, `-n 4`, `--seed 42 --temp 0 --reasoning off`, `-t 8`,
`--lazy-mode off`, `draft-mtp n3`; all runs `////`=0.  `off` = `GGML_SCHED_STAGE=0`; `forced` =
`GGML_SCHED_STAGE=1 GGML_SCHED_STAGE_MIN_TOKENS=1`; `default` = no env (scaled gate, base 1542 + scale).

| model / host table | `-ub` | off | forced (staged) | old default (scaled) | new default (unscaled, `REF_MB=0`) |
|---|---:|---:|---:|---:|---:|
| Flash-Next IQ4_XS, 850 MiB, `-sm tensor` | 1024 | 299.8 | 377.9 | 296.3 | 296.3 |
| " | 2048 | 487.5 | 701.9 | 480.2 | **686.6** |
| " | 4096 | 729.0 | 1215.7 | ~729 | **1160.7** |
| " | 8192 | 997.0 | 1777.1 | 1540.7 | **1663.0** |
| Flash-Next IQ3_XXS, 450 MiB, `-sm layer` | 1024 | 292.9 | 289.0 | -- | -- |
| " | 2048 | 494.1 | 552.4 | ~494 | -- |
| " | 4096 | 748.2 | 999.4 | 757.1 | **977.8** |
| " | 8192 | 1062.1 | 1617.0 | 1551.4 | -- |

* **Staging wins at every width from 1024 to 8192 on both table sizes** (the only non-win is the
  450 MiB table at 1024: 289 vs 293, a tie).  The scaled gate turned a win into a loss for every table
  > 144 MiB (the reference), i.e. it is exactly backwards on this box.
* Disabling the scaling reproduces the width-only gate and recovers the win: IQ4_XS 2048 480 -> **687**
  (+43 %), 4096 729 -> **1161** (+59 %), 8192 1541 -> **1663** (+8 %); IQ3_XXS 4096 757 -> **978** (+29 %).
* **The generated text is byte-identical** to the old default (only the load spinner differs), 2 GPU, 16k;
  the 3-GPU default (`-ub 8192`) is coherent (1529.6 t/s).
* **No regression on the 144 MiB reference**: for a table <= the reference, the scaling was already a
  factor of 1, so the default behaviour there is unchanged (the r7 `-ub 8192` 35B-A3B numbers stand).
* Change: `SCHED_STAGE_TABLE_REF_BYTES` default `144 MiB -> 0` (width-only gate).  The env
  `GGML_SCHED_STAGE_TABLE_REF_MB=<MiB>` still restores the scaling for A/B.  **PROMOTED to the delivery
  in `v16-a55e952b8-r18` (block 06)**; the `stage1-item3-width-gate.patch` file here is kept as provenance.

**Separate, still open: the base width gate itself may be too high for big tables.**  At `-ub 1024`
staging *also* wins for the 850 MiB table (377.9 vs 299.8, +26 %) yet the bandwidth-calibrated base
(1542 tokens, 14.5 GB/s x4) leaves it un-staged.  The crossover appears to *decrease* with table size
(larger table -> the serial path's per-split device-sync overhead dominates sooner), the opposite of
the r7 scaling.  A re-derivation (possibly an *inverted* scaling) is a further stage-1 item -- it
needs the 144 MiB reference re-measured at 1024/2048 first, since that is the only data point that
justified a base of ~1536.

**The H2D calibration probe is a one-off ~0.4 s at the first prefill**, not a per-token cost (it is 5 %
of a 16k prefill but 2.7 % of a 32k one).  For the `llama-server` long-uptime case it is irrelevant;
do not chase it.

## 12. STAGE 2 RESULT: the cache-auto `-ub 8192` OOM is fixed and the decode DoD is met (candidate)

**2 GPU, `-sm tensor -ncmoe 48`, cache auto, `-ub 8192`, 16k `/tmp/pl_16k.txt` + `-n 2000`, `-t 8`,
coherent (`////`=0), device count asserted:** first prefill no longer OOMs; **prefill 1639.9 t/s,
decode 75.5 t/s** (DoD `>= 68.9`), arena 38686 MiB (68.2 %), hit 0.958, MTP acc **0.9245** (mean 3.77).
The doc's older §3 remedy (draft `n_ubatch = n_max+1`) gave 871/41 with a 21175 MiB arena; this is
better on both axes.

| config (2 GPU unless noted) | prefill | decode | arena | notes |
|---|---:|---:|---|---|
| cache auto `-ub 8192` (**DoD**) | **1639.9** | **75.5** | 38686 (68.2 %) | no OOM |
| cache auto `-ub 4096` | 1174.4 | 76.8 | 39794 (70.1 %) | was 69.2 decode at r18 |
| cache off `-ub 8192` | 1625.3 | -- | -- | -2 % vs the uncapped 1663.9 |
| 3 GPU cache auto `-ub 8192` | 926.0 | 95.3 | 100 % | coherent |
| 3 GPU cache off `-ub 8192` | 1539.1 | -- | -- | -1.6 % vs uncapped 1564 |

### The design (candidate patch `stage2-arena-shrink.patch`, 12 files, +185)

1. **Drop the wide-prefill compute layout at the prefill -> decode transition** and re-reserve the
   widest post-prefill (verify) layout.  In `llama_context::process_ubatch`, when
   `n_tokens_prev > 8 && ubatch.n_tokens <= 8` and the model has host-resident experts (and this is not
   the MTP draft context): `ggml_backend_sched_drop_buffers()`, then `graph_reserve(cparams.n_rs_batch,
   1, cparams.n_rs_batch, mctx)`.  The compute buffers are otherwise grow-only, so the 11765 MiB
   prefill layout would stay resident and the arena (sized on the first full decode pass) gets only
   20621 MiB (36 %).  New primitives: `ggml_gallocr_drop_buffers` / `ggml_backend_sched_drop_buffers`
   (free every distinct compute buffer + invalidate the layout) and
   `llama_context::drop_compute_buffers` / `llama_context_drop_compute_buffers`.
2. **Hold the post-prefill layout out of the arena.**  After the re-reserve, read the scheduler's
   per-backend buffer size (the 1700.7 MiB verify layout) and pass it to the MoE cache via a new
   `ggml_backend_dev_moe_cache_set_reserve(dev, bytes)` -> `moe_cache_set_extra_reserve(device, bytes)`,
   which `alloc_all_locked` adds to its reserve.  Without it the arena takes the freed VRAM and the
   verify's grow-in-place realloc (`reallocating Meta() 1700.73 -> 1700.73`) fails next to the arena.
   Under `-sm tensor` the scheduler's backends are Meta wrappers, so the setter is driven from the
   model's real (host-expert-owning) devices.
3. **The MTP draft's `n_ubatch` is capped to 512** (`common/speculative.cpp`, `MTP_DRAFT_N_UBATCH`
   overrides).  The draft is not decode-only: during the target's first prefill it consumes the encoder
   injection (the target's hidden states) in `n_ubatch` chunks, so its compute buffer is sized for the
   chunk.  At the target's full `-ub` that is ~1.6-1.7 GiB/device, which is what makes the target's own
   growth miss.  A small chunk is all the draft needs (it drafts `n_max+1` tokens); 512 keeps the
   injection fast (2 GPU cache-off prefill 1634 vs 1290 at `n_max+1`).

### Root cause of the OOM is a reserve miscalculation (TODO #42 follow-up)

The target's pp reserve is **11339 MiB but its own first prefill needs 11765** (+426 MiB, a cache-on
tail-layout difference; §1).  That growth is a *free-then-alloc of a contiguous 11.8 GiB block*, so it
is **fragmentation-sensitive**: with the draft capped, 256/512/1024/3072/4096/6144 all pass while
**2048 fails** even though the draft's buffer was only 420 MiB there.  **The correct fix is to make the
reserve a true upper bound for the first prefill** (then no in-place realloc, no fragmentation
sensitivity, and the draft cap can be lifted entirely).  That, plus the arena-reclaim retry (fail-soft,
TODO #38) on any later growth, is the remaining Stage-2 work.

### Measurement notes

* The llama-cli `Prompt:` aggregate **includes the MTP draft's encoder injection**, so it is not the
  target's prompt eval.  Same run, 3 GPU: slot timing `prompt processing ... 1410.56 t/s` vs the
  reported `Prompt: 717.5 t/s`; 2 GPU: slot 1613.71 vs reported 1289.9.  Quote the slot timing for the
  target's prefill.
* The arena is sized on the first *decode* pass, i.e. **after** the prefill (log: prefill 0.34.5,
  `alloc_all_locked` 0.34.8), so the auto-cache sizing is not a prefill cost.  On 3 GPU the auto sizer
  fills to 100 % because all 56 GiB of host experts fit; the 1024 + 1700 MiB reserve is what is left for
  a later growth.

### 12.1 The drop is NOT server-safe (found by a live server test, 2026-10-06) -- now opt-in

The §12 drop was default-on and **aborted a `llama-server`** on the second prompt.  Repro: 2 GPU,
`-ub 4096 -c 102400` cache auto, MTP n3; turn 1 generated ~41,840 tokens fine (arena sized 44285 MiB,
78 %); a new chat window (a ~360-token prompt) then needed a **6780.42 MiB** compute layout and

```
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 6780.42 MiB on device 0: cudaMalloc failed
ggml-backend-meta.cpp:1799: GGML_ASSERT(bufs.back() != nullptr) failed   (ggml_abort)
```

**Why:** the drop releases the wide-prefill compute layout so the arena can be large, but a *later*
prompt needs that layout back, and the arena now holds the VRAM.  This is exactly the server-safe
ordering the §4 plan flagged (step 2, the arena-reclaim retry): a smaller/absent arena is strictly
better than an abort.

**Decision for now:** the drop (and its arena extra-reserve) is gated behind
`LLAMA_DROP_COMPUTE_BUFFERS=1` and **defaults OFF**, so the delivery candidate is r18-safe (the
wide-prefill layout stays resident and every later prompt reuses it).  Consequences:

* The §12 DoD numbers (decode 75.5) hold only with `LLAMA_DROP_COMPUTE_BUFFERS=1`, i.e. single-shot
  `llama-cli` workloads.
* For a server, the safe cache-auto arena is the no-drop sizing: `-ub 4096 -c 102400` sizes to
  36198 MiB (63.8 %, helped by the §12 draft cap) and decodes at the r18 rate; `-ub 8192` sizes to
  ~20621 MiB (36 %) and decodes ~53.  A later prompt reuses the resident layout and never aborts.

**To make the drop default-on it needs the arena-reclaim retry** (`moe_cache_release_arena()` + a retry
in `ggml_backend_cuda_buffer_type_alloc_buffer`, fail-soft, TODO #38): on a compute-alloc failure free
the arena, warn once, and retry.  That turns the abort into "the cache yields for the rest of the run"
-- safe, but the decode win is then single-shot anyway, so the reclaim is a crash guard rather than a
server win.  The better long-term direction is to shrink the *prefill* compute layout itself, so a wide
`-ub` and a large arena both fit without dropping.

### 12.2 The core #42 bug on a *server*: the auto arena starves a later compute growth

With the drop off (r18-safe), a `llama-server` (2 GPU, `-ub 4096 -c 102400`, cache auto, MTP n3) still
aborted, on the **second concurrent session**:

```
allocating 6771.52 MiB on device 0: cudaMalloc failed   (turn 1 only)
allocating  154.88 MiB on device 1: cudaMalloc failed   (turn 2, after a small fix)
ggml-backend-meta.cpp:1799: GGML_ASSERT(bufs.back() != nullptr) failed
```

**Root cause.** The auto arena sizes as `free - (MOE_EXPERT_CACHE_RESERVE_MIB + headroom)` at the first
full decode pass.  In a *server* the first prompt is short, so the target's compute buffer is not yet at
its runtime maximum; a later, wider batch (a second slot; the batched prefill) needs the full `-ub 4096`
layout and the arena now owns that VRAM.  Measured: the pp reserve is **6564.4 MiB** but the runtime
first prefill needs **6771.5 MiB** -- the reserve is genuinely **~207 MiB short** (the §1 tail-layout
artifact; ~426 MiB for the 850 MB-table `-ub 8192` case).  Device 1 then needed only ~105-155 MiB, i.e.
the arena had taken *everything* there.

**The arena's per-device accounting is the practical problem.** `alloc_all_locked` computes
`avail = free - reserve - extra` per device, but:
* the model's device list under `-sm tensor` is a single Meta wrapper, so a "set the reserve on the
  model's devices" loop misses device 1 (observed: the reserve only ever reached device 0);
* making the reserve device-global (implemented) still did not stop the device-1 abort, so the per-device
  projection/`alloc_table_locked` accounting needs its own review.

**Working configuration (validated).**  `MOE_EXPERT_CACHE_RESERVE_MIB=8192` -- the arena then leaves
8 GiB and a two-session concurrent run (a 1500-token prompt generating 2000 tokens, plus a second
prompt mid-generation) completed with **0 crashes**.  Cost: the arena drops (measured 14745-21897 MiB,
26-39 % residency), so decode is slower than the no-reserve case; the arena is the lowest-priority
consumer, so this is the correct trade until the headroom is computed automatically.

**TODO (the real #42 fix).**  Make the arena leave `pp_reserve_runtime - current_compute + margin` per
device *automatically* (the `moe_cache_set_extra_reserve` plumbing exists, but the reservation must be
applied per device and verified against the actual `alloc_table_locked` consumption).  Until then, a
server needs an explicit large `MOE_EXPERT_CACHE_RESERVE_MIB`.

### 12.3 THE FIX: reserve a true-upper-bound compute buffer (large arena, any `-np`, no OOM)

**Root cause of the whole #42 family: the compute reserve was not a true upper bound.**  `sched_reserve`
builds a *measure* graph for `min(n_ctx, n_ubatch)` tokens, but the runtime graph marks additional
tensors live (the MTP layer-input taps, `llama_context::extract_layer_inputs`) so its layout is a few %
larger: measured **6564.4 MiB reserved vs 6771.5 MiB needed** (`-ub 4096`), and 11339 vs 11765
(`-ub 8192`).  The compute buffer is grow-only, so it then grows *after* the MoE expert-cache arena has
taken the free VRAM -> `cudaMalloc` abort (llama-cli start-up OOM; llama-server second/concurrent
prompt).

**Fix (block-level, `llama_context::sched_reserve`):** reserve a **synthetic batch slightly wider than
any batch the context can run**,
`n_tokens = min(n_ctx, n_ubatch + max(256, n_ubatch/16))`.  No batch wider than `n_ubatch` is ever
actually run -- the extra tokens only make the measure layout an upper bound (the slack is real memory:
~360 MiB at `-ub 4096`, ~800 MiB at `-ub 8192`).  Consequences:

* the compute buffer never grows mid-run => **no `cudaMalloc` abort, for a server of any `-np`**;
* the auto arena can stay **large and shared** -- it is one per-device cache, shared by every slot (it
  was never per-session);
* the **MTP draft `n_ubatch` cap is no longer needed** (default removed; `MTP_DRAFT_N_UBATCH` kept as an
  A/B knob -- the draft's encoder injection runs in chunks of `n_ubatch`, so capping it costs prefill);
* the **`-np`-conditional reserve is no longer needed** (removed);
* the compute drop remains a **single-shot opt-in** (`LLAMA_DROP_COMPUTE_BUFFERS=1`) for maximum decode.

**Measured** (2 GPU, `-sm tensor -ncmoe 48`, cache auto, 16k + `-n 2000`, coherent `////`=0, **0 OOM**):

| config | prefill | decode | arena | hit |
|---|---:|---:|---|---:|
| `-ub 8192` | 1720.3 | 45.6 | 15688 MiB (27.6 %) | 0.814 |
| `-ub 4096` | 1191.1 | 66.7 | 29435 MiB (51.9 %) | 0.929 |
| `llama-server` `-np 4`, two concurrent sessions | -- | -- | 29767 MiB (52.4 %) | 0 crashes |

**Trade.**  The bigger compute reserve costs arena (and decode) versus the crash-prone no-margin
sizing -- 63.8 % -> 52.4 % on the server, decode 69.2 -> 66.7 at `-ub 4096`.  The arena is the
lowest-priority VRAM consumer, so a smaller arena is strictly better than an abort.  The remaining knob
is the slack (`max(256, n_ubatch/16)`; the measured shortfall is 207-426 MiB), and the arena-reclaim
retry (TODO #38) is still the right belt-and-braces guard for any residual shortfall.

### 12.4 Per-turn arena hit rate (server log), and what `-np` does to the arena

**Per-turn stats (added).**  The server now logs the decode arena hit rate for each turn, right next to
the MTP acceptance line (`server_slot::print_timings`, server INFO):

```
0.26.615.686 I slot print_timing: id  3 | task 0 | draft acceptance = 0.62319 (  129 accepted /   207 generated), mean len =  2.84
0.26.615.706 I slot print_timing: id  3 | task 0 |   MoE arena = 0.9115 (727032 hit / 797580 reaches this turn), arena 36141.7 MiB
```

Plumbing: `moe_cache_get_stats()` (aggregates the same counters `moe_cache_report` prints, including the
device-side admission-policy D2H) -> `moe_cache_stats` device iface ->
`ggml_backend_dev_moe_cache_stats` -> `llama_moe_cache_stats(model, ...)`.  The server keeps the previous
cumulatives per slot and logs the delta.  The counters are process-global (the arena is one shared
per-device cache), so with concurrent slots the delta is whole-process activity during the turn, not
strictly one session's -- it is a diagnostic, not a per-session metric.  Only emitted when the arena is
active (host-resident experts with the cache enabled).

**`-np` and the arena (measured, same build/config):** `-np 4` -> arena **36531 MiB (64.4 %)**;
`-np 1` -> **37418 MiB (65.9 %)**.  So a smaller `-np` does give a larger arena, but only ~0.9 GiB
(~1.5 pp), because the compute reserve is dominated by the **prefill (pp) graph**, which is sized from
`min(n_ctx, n_ubatch)` -- `-ub`/`-b` are the levers (`-ub 4096` -> ~29-37 GiB arena, `-ub 8192` ->
~15.7 GiB) -- while `-np` only enters through the decode (tg) graph's `n_seqs` and a few per-sequence
tensors in the pp graph.  With the §12.3 reserve margin the arena is safe for any `-np`, so `-np` is now
purely a scheduling/throughput choice, not a memory one.

### 12.5 Instrumented: the reserve is ~7x short (944 MiB vs 6771 MiB) -- the measure graph omits the host-expert staging

Added a temporary `WARN` in `sched_reserve` printing the computed reserve, and ran the server:

```
W sched_reserve: WIP reserve: n_tokens=4352 n_slack=256 n_seqs=4 no_alloc=0 sizes_MiB=[944 214]
```

So the **whole** Meta compute reserve is **944 MiB**, while the runtime graph needs **6771-6773 MiB**.
The measured shortfall for `-ub 4096` is therefore **~5.8 GiB, not ~200 MiB** -- the extra 256-token
slack is irrelevant.  The measure graph does **not contain the host-expert staging tensors**
(`Meta(...)#blk.N.ffn_down_exps.weight#0`, the 850 MB staged expert table the §1 peak carries), so it
is not a usable upper bound for a `-ncmoe` run at all.

That is the real root cause of the whole #42 family:
* the compute buffer is *supposed* to be pre-sized for the widest graph, but for host-resident experts
  the pre-sizing omits the dominant tensors;
* the arena (sized from the leftover free VRAM) then owns the ~6 GiB the runtime staging later needs;
* the growth is a free-then-alloc of a contiguous ~6.7 GiB block, so it is also fragmentation-sensitive.

**The real fix is therefore: make the reserve include the host-expert staging** (reserve with the
host-expert/MoE-offload path active -- the reserve runs before the scheduler's offload decision for the
staging inputs).  Once the reserve is an actual upper bound, the arena cannot starve the compute and no
reclaim/shrink is needed at all.  The `n_slack` margin stays as a small safety net.

**Fail-soft guard (added anyway).**  `moe_cache_release_arena()`, called from the CUDA alloc path on a
failed `cudaMalloc`, frees the arena once and retries (cache disabled for the run) -- so even an
unforeseen shortfall degrades instead of aborting.  The better version of that guard is the requested
one: **shrink the arena in steps** (free the largest tables on the failing device until the failed size
is available, retry) so most of the cache survives, rather than dropping it all.  The arena's *own*
slot-allocation soft-fail should likewise retry at half the slots before settling for 0.

### 12.6 Correction to §12.5: the 944 MiB was the MTP draft; the target's reserve is ~6.4 GiB

`sched_reserve` runs **twice** -- once for the target context, once for the MTP draft.  The two lines:

```
W sched_reserve: WIP reserve: n_tokens=4352 n_seqs=4 no_alloc=0 sizes_MiB=[6429 129]   <- target
W sched_reserve: WIP reserve: n_tokens=4352 n_seqs=4 no_alloc=0 sizes_MiB=[944  214]   <- MTP draft
```

So the **target's** reserve is **6429 MiB** (server) / **6636 MiB** (cli) and the runtime needs **6780**
(`reallocating Meta() buffer from size 6636.95 MiB to 6780.17 MiB`).  The gap is **~150-350 MiB**, not
5.8 GiB -- §12.5 misattributed the draft's line.

Crucially the gap is a **peak-tensor-set / layout difference, not a size one** (the peak is set by
*which* tensors are live at the peak, incl. the 850 MB staged expert table), which is why the 256-token
reserve slack did **not** close it (it added ~360 MiB of layout but the peak barely moved: the peak is a
single large tensor, not a sum).  So:

* the token-slack is **dropped** (it costs arena and does not fix the gap);
* the fix is a **headroom held out of the arena** -- device-global, `max(512 MiB, compute/16)` -- which
  covers the layout gap so the compute buffer can still reach its runtime peak after the arena is sized;
* the reclaim guard (§12.5) remains the fail-soft backstop.

(And the proper long-term fix is still to make the measure graph carry the same live-tensor set as the
runtime -- the host-expert staging / MTP taps -- so the reserve is an exact upper bound and no headroom
is needed.)

### 12.7 A headroom cannot fix it: the failure is the grow-in-place realloc, not free VRAM

Swept the arena headroom (`MOE_ARENA_HEADROOM_MIB` = 512 / 1024 / 2048 / 4096) against the concurrent
long+short prefill.  **Every value behaves identically**: no abort (the reclaim guard fires), and the
guard is needed at every value:

```
0.20.216 W process_ubatch: WIP arena headroom 4096.0 MiB (compute 6564.4 MiB)
0.20.233 W alloc_all_locked: arena 28050.3 MiB (49.4 %)          <- arena shrunk by ~8 GiB (per-device)
0.25.809 W moe_cache_release_arena: released ... (27605.2 MiB)
0.25.809 W ggml_backend_cuda_buffer_type_alloc_buffer: 6780.30 MiB allocation failed on device 0
```

So the headroom does **not** prevent the failure even at 4 GiB.  Reason: the growth is a
**grow-in-place realloc** -- `ggml_gallocr_reserve_n_impl` does `ggml_vbuffer_free(old)` then
`ggml_vbuffer_alloc(new)`, i.e. it frees **6564 MiB and immediately asks for 6780 MiB**.  So it needs a
*contiguous* block 216 MiB **larger than the one it just released**; extra free VRAM elsewhere does not
help, which is why 4096 MiB of headroom changes nothing.  (This also explains the earlier
fragmentation-sensitivity: 2048 passing/failing at random.)

**Consequences for the design:**
* **Drop the headroom** -- it costs arena (28050 vs 36198 MiB, 49 % vs 64 %) for no benefit.
* The real fix is still to make the reserve an **exact upper bound** so the buffer never needs to
  reallocate at all (the measured gap is only ~216 MiB, but it must be *inside the reserve layout*, not
  beside it).  Reserving with the runtime's live-tensor set (host-expert staging + MTP taps) is the way.
* The reclaim guard is therefore the *primary* safety net, not a backstop, and should be the **shrink**
  form (free the largest tables on the failing device until the failed size is contiguous, retry) so
  most of the cache survives instead of all of it.

### 12.8 (1)+(2) done: headroom dropped; slot-count retry in; full release stays (partial shrink is unsafe yet)

**(1) Headroom dropped.**  The `MOE_ARENA_HEADROOM_MIB` extra-reserve and the token-slack are gone (§12.6
/§12.7 showed neither closes the gap, and the headroom only cost arena).  The arena is back to 36198 MiB
(63.8 %) on the server.

**(2b) Arena slot-count retry (in).**  `alloc_table_locked` no longer drops a table to 0 slots on a
failed `cudaMalloc`.  It now tries the requested count, then the **exact** largest count the free VRAM
can hold (`cudaMemGetInfo` -- one call, lands *at* the max), then a **0.95 geometric descent** for the
fragmentation case (a failed `cudaMalloc` is cheap, so the attempts are free).  The table keeps the
largest slot count that actually allocates, with a WARN.  (Exact accounting > 0.95 walk > halving, and a
bisect would be the next refinement if a tighter fit is ever needed.)

**(2a) The partial shrink is NOT safe yet.**  Implemented `moe_cache_shrink_step()` (free the largest
table arena, retry, repeat -- the shortfall is only ~216 MiB, so one or two tables should suffice) and
wired it into the failed-allocation path.  **It crashed the server** (client exit 52, no `GGML_ASSERT`,
no cudaMalloc error -- a silent death) after freeing 10 tables: freeing individual arenas while the
cache stays enabled leaves the **device-side remap / admission-policy descriptors** referencing the
freed arena.  So the guard is back to `moe_cache_release_arena()` (full release, proven: concurrent
long+short prefills complete, 0 aborts) -- the cache is lost for the run when it fires, but the run
survives.

**To get the cache-surviving shrink there is one prerequisite:** a **per-table stand-down** -- when a
table's arena is freed mid-run, invalidate its remap (`remap_n_used`), its `g_alias_to_id` entry and its
device-policy descriptor (or rebuild the policy array) so no in-flight or subsequent consumer reads the
freed pointer.  Until that exists, free-all is the only safe arena release.

**Net state of the Stage-2 candidate:** reserve = the plain pp/tg reserve (no slack); the compute buffer
may still grow ~216 MiB into a later, wider graph; the MoE arena is sized from the leftover VRAM and the
compute growth is satisfied by the fail-soft guard (full release).  The server is crash-free under
concurrent prefills; the requested *minimal* shrink and the exact reserve remain open.

### 12.9 The partial shrink needs an un-redirect, not just a table teardown

Implemented the per-table stand-down (`stand_down_table_locked`) mirroring the proven device-migration
teardown -- free the arena **and** the remap, zero `remap_n_used`/`remap_cap`, reset the residency maps,
update `g_arena_bytes`, and erase the table's `g_alias_to_id` entries -- and re-wired the shrink loop
(`moe_cache_shrink_step()` per release, retry after each).  **It still dies silently** (client exit 52,
no `GGML_ASSERT`, no cudaMalloc error) after ~28 tables:

```
W moe_cache_shrink_step: stood down the largest table (layer=14 role=blk.14.ffn_down_exps.weight, 170.3 MiB) ...
```

**Why.**  A table teardown is not enough: the scheduler has **already repointed the graph's `input_cpy`
tensor at the arena** (`moe_cache_take_over`), so as soon as the arena is freed the in-flight graph
holds a dangling `weight_cpy->data`.  This is the *same class* as the block-16 staging-redirect
corruption: the redirect is only safe while the pointed-to storage lives.

**The missing piece** is therefore an **un-redirect**: before freeing any table's arena, restore the
redirected tensors' `data` to their real `input_cpy` buffer (exactly what `sched_stage_restore` does for
the staging redirect), or defer the shrink to a graph boundary where no redirect is live.  `moe_cache_take_over`
already records the original pointer for its own restore path, so the data is available -- it just has to
be applied for the shrunk tables too.

**Until then the guard is the full release** (`moe_cache_release_arena`), which is safe because it also
sets `g_enabled = false`: the scheduler then takes the non-redirect path and the same in-flight
`alloc_graph` retry re-establishes `input_cpy`.  Verified: concurrent long+short prefills complete,
0 aborts, arena back to 36198 MiB (63.8 %) -- the cache is lost for the run when the guard fires.

**So the design is one well-understood step from complete:** add the un-redirect to
`stand_down_table_locked` (or shrink at a graph boundary), and the guard becomes "free a few hundred MiB
of the least-needed tables" instead of "lose the arena".
