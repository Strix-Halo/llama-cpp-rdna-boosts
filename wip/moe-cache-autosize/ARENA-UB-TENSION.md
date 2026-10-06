# TODO #42 -- the arena-vs-`-ub` tension, and the 2-GPU `-ub 8192` OOM

**Status: OOM root-caused; one validated remedy; the full DoD (wide prefill + fast decode in one
config) needs a compute-buffer shrink whose sequencing is not yet safe.  Nothing promoted.**

Session build: `~/llama.cpp` @ block-17 tip `60280296f`, binary `build-rocm-r16/bin/llama-cli`,
scratch patch `arena-ub-tension.diff` (this dir, 8 files / +107).  Box: 2 x R9700 (gfx1201, 32 GB),
`Qwen3.8-Flash-Next-UD-IQ4_XS` + its shared Q8_0 MTP head, `-sm tensor -ncmoe 48 -fa on -ctk q8_0
-ctv q8_0 -t 8 -c 32768 -b 8192 -ub 8192`, 16k prompt `/tmp/pl_16k.txt`, `--temp 0 --seed 42
--reasoning off`, cache auto unless noted.

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
| **1** | **Correct + persistent-guard device gather** | the gather copies only the **routed** experts, so it avoids the split's whole-range down scratch (SS9) *and* the mirrored full-table copy.  Blocker is r31 Hole B: the finite-head zero in `moe_cache_get` is once-only keyed on `(weight_cpy->data, expert_bytes)`, so a reused `input_cpy` loses it.  Fix = re-arm the guard per gather (or make the destination never-reused), then validate on a long multi-ubatch prefill. | **IN PROGRESS** |
| **2** | Pinned 2-D H2D for the down slice | the down split slice is `width`/`stride_src` strided; the source is pinned, so a pinned `cudaMemcpy2DAsync` straight into the slot would move only the slice (no 1.31 M-block host gather, no whole-range scratch). | pending |
| **3** | Width-gate scaling fix | default gate 1039 -> forced 1172 (+13 %); the `table_bytes / 144 MiB` scaling overshoots for the large IQ4_XS tables and leaves staging off where it amortizes best. | pending |

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
