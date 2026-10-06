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
