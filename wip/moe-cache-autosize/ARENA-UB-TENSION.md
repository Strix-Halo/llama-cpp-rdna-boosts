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
