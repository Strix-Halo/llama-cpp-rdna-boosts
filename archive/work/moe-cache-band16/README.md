> **CLOSED / FOLDED (r31, 2026-10-08).**  Accepted as PR #115 (@briansp2020) and folded into delivery
> **block 06**.  The MoE expert cache's decode/verify band now follows the routed-expert MMVQ band
> (`get_mmvq_mmid_max_batch`, 16 tokens on RDNA4) instead of a literal 8, through a new
> `moe_cache_band` backend iface hook that gives the band a single owner (the meta backend reports the
> narrowest over its devices), plus a fix for the device-side admission kernel's fill list (an expert
> could be marked resident without its fill once the 64-entry list was full; now 256 and admission
> stops when full).  `GGML_MOE_CACHE_MAX_TOK=8` restores the pre-r31 band.  RDNA3 (where the 16-wide
> routed MMVQ band is incorrect/floored) and NVIDIA stay at 8 by construction.  See `WORKLOG.md`
> 2026-10-08 (r31), the block-06 r31 amendment in `patches/README.md`, and `ENVIRONMENT.md` §1.2.

## Validation on this box (3x R9700 / gfx1201, ROCm 7.14.1)

The PR was measured on 2x R9700 / ROCm 10.0.  Reproduced here on the delivery's reference build with
`Qwen3.6-35B-A3B UD-Q4_K_M` (qwen35moe, nextn MTP head, `-ncmoe 40` host experts, arena 37324 MiB at
100 % residency), which is the delivery's MoE reference model.

**Rule 5 (`llama-batched-bench`), `-ngl 99 -sm tensor -ncmoe 40 -fa 1 -ctk q8_0 -ctv q8_0
-npp 16 -ntg 32 -npl 1,4,8,16 -b 2048 -ub 2048 -c 16384`, S_TG t/s (higher is better):**

| tokens / MoE step | r31-pending (band 8) | PR, `GGML_MOE_CACHE_MAX_TOK=8` | PR (band 16) | all-VRAM (`-ncmoe 0`) |
|---|---:|---:|---:|---:|
| 4 (B=4) | 160.4 | 158.3 | 150.8 | 182.2 |
| 8 (B=8) | 296.0 | 292.4 | 290.1 | 328.8 |
| **16 (B=16)** | **123.7** | **132.1** | **702.5** | **720.2** |

B<=8 is unchanged within noise (the band does not reach them).  B=16 is **5.7x** the old band and
matches the all-VRAM build (2.5 % under).  The old code comment that warned "offload above 8 took
`-npl 16` 137 -> 52 t/s" was measured before the extended routed MMVQ band existed; with the band at
the kernel's own width the takeover is the fast path.

**Protocol A (`llama-cli`, prose prompt `sha256 fabdec65…`, seed 42, temp 0, bf16 KV, `--reasoning
off`, `-n 400`, same model/config):**

| config | r31-pending | PR (band 16) |
|---|---:|---:|
| `--spec-type none` | 81.6 t/s | 81.4 t/s |
| `draft-mtp --spec-draft-n-max 12 --spec-draft-n-start 12` (13-token verify) | **33.9 t/s**, acc 0.331 | **110.7 t/s**, acc 0.336 |

**Text purity (generated-text sha256, via `scripts/extract-generated.py`):**

* `--spec-type none`: `92daa37ab115`, identical patched/pre-patch.
* `draft-mtp n-max 7`: `92daa37ab115`, identical patched/pre-patch **and** to `none` (the `n_max <= 7`
  purity guarantee holds).
* `draft-mtp n-max 8` (9-token verify, on the widened band): patched `b1a0ddf528c7`; pre-patch
  `6124e50891c5`; patched with `GGML_MOE_CACHE_MAX_TOK=8` `6124e50891c5` (the kill-switch restores the
  old band exactly); patched **all-VRAM** (`-ncmoe 0`) `b1a0ddf528c7`, so the band-16 cache path is
  **bit-identical to the all-VRAM build** at the width it newly serves.

**Other gates:** prefill-logit KLD (dense Qwen3.8-27B-Q8_0 base) mean KLD **0.000707** / same-top-p
**98.755 %** (PASS); `test-backend-ops -o MUL_MAT_ID` **931/931** on gfx1201; build warning-free;
`validate-set.sh` green (strict 16/16, tree `4f259e104f24dea2cf578b4cdfe5edd3ad2cec25`).

**Pre-existing, orthogonal (present in both the patched and the r31-pending build, not from this PR):**
Both were fixed in **r32** (`WORKLOG.md` 2026-10-08 (r32): #48 the post-prefill re-reserve stream count,
block 06; #49 the meta split-state stack overflow, block 15).

* `llama-batched-bench` with `-npl 4` as the **first** row aborts in `ggml_reshape_3d`
  (`build_layer_attn_linear`) on qwen35moe `-sm tensor -ncmoe 40`; with a B=1 row first, B=4/8/16 run.
* `llama-cli --spec-type draft-mtp` at `--spec-draft-n-max 3 --spec-draft-n-start 3` (and the default
  adaptive controller) segfaults on the same model/config; `n_max` 7/8/12 run.

**Not measured:** RDNA3 and NVIDIA hardware (the band stays 8 by construction); models with more than
16 experts per token (admissions past 256 in one pass stay cold, correct but fewer fills).

---

# moe-cache-band16: widen the MoE expert-cache band to the routed-expert MMVQ band (16 on RDNA4)

One format-patch on top of `v16-a55e952b8-r28` (applied tree `ae5aa3e06`), applied after `patches/00*.patch` with
`git am`.  8 files, +81 / -11.  `GGML_MOE_CACHE_MAX_TOK=8` restores the previous band.

## The problem: a concurrency cliff on the host-expert path

The expert cache takes over a routed `MUL_MAT_ID` only when the batch has <= 8 tokens.  With MTP every active stream
verifies `n_max + 1` tokens per step, so at n-max 3 three or more concurrent streams give 12-16-token MoE batches.
Those fall to the scheduler's host path (ids readback + full device synchronize + H2D copy of every used expert into
`input_cpy`), and the arena is not used at all.

2 x R9700 (gfx1201, ROCm 10.0), Flash-Next GSQ-RCO IQ3_XXS + shared Q8_0 MTP head (n-max 3, p-min 0), `-sm tensor
--n-cpu-moe 48`, 256K, default slab (arena 37.6 GiB, 92 % residency); N identical greedy 300-token requests, decode t/s:

| concurrent streams | tokens per MoE step | r26 (band 8), total | all experts in VRAM (`-ncmoe 0`), total |
|---|---|---|---|
| 1 | 4 | 107.7 | 121.2 |
| 2 | 8 | 163.6 | 184.5 |
| 3 | 12 | **45.8** (15.3 each) | 192.7 |
| 4 | 16 | **53.0** (13.2 each) | 200.6 |

The same cliff applies to a single stream with n-max >= 8 and to 9-16-token prompt chunks.

## The change

The cache serves a batch only through the routed-expert MMVQ kernel (it reads the arena through the slot remap;
MMQ / MMF do not), and the arena redirect already runs at the top of `ggml_cuda_mul_mat_id`, before kernel selection.
On RDNA4 that kernel already serves 16 columns (`MMVQ_MOE_MAX_BATCH_SIZE`), so the band can follow it:

- `MOE_EXPERT_CACHE_MAX_TOK` becomes `moe_cache_max_tok()` / `moe_cache_max_tok_dev(device)`: the narrowest
  `get_mmvq_mmid_max_batch` over the quantized types on the device, never below the historical 8.  That is 16 on
  RDNA4, CDNA, GCN and RDNA1/2, and stays 8 on RDNA3 (where the 16-wide MoE MMVQ band is floored to 8 because it is
  incorrect on gfx1100) and on NVIDIA.  `GGML_MOE_CACHE_MAX_TOK` can lower it.
- A new backend iface hook `moe_cache_band`, so the band has a single owner: the CUDA backend reports its device's
  band, the meta backend the narrowest over its devices, and a backend without the hook keeps 8.  The scheduler's four
  literal 8s (the take-over before the ids readback, the per-layer split grouping, the decode-band rebalance, the
  device-gather gate) and the meta backend's `moe_cache_update` ask the backend instead.  CPU / RPC get explicit
  `NULL` entries (warning-free build).
- Left at 8 on purpose: `alloc_all_locked()`'s `n_tok <= 8` in `moe-expert-cache.cu`, which is tied to
  `llama-context`'s compute-buffer drop condition rather than to the cache band.

## A separate, pre-existing finding fixed in the same patch: the admission fill list

`moe_cache_policy_kernel` stages a pass's admitted experts in a fixed shared-memory list (`MOE_CACHE_POLICY_MAX_FILL`,
64).  Once the list was full, the loop still set `t.slot[e]` (the expert counts as resident) but did not stage its fill,
so the next pass could read a stale arena slot.  64 is already below `n_used * 8 = 80` for a 10-expert model like
Flash-Next at the old band; at high residency the misses per pass stay small, which may be why it has not shown up.
The list is now 256 (16 tokens x 16 experts) and the loop stops admitting when it is full (the expert stays cold and
is served from the host alias).

## Results

Same setup, one run each:

| concurrent streams | band 8, total | band 16 (r26 + patch), total | band 16 (r28 + patch), total | all-VRAM, total |
|---|---|---|---|---|
| 1 | 107.7 | 107.3 | - | 121.2 |
| 2 | 163.6 | 157.7 | - | 184.5 |
| 3 | 45.8 | 182.5 | 179.1 | 192.7 |
| 4 | 53.0 | 198.3 | 197.7 | 200.6 |

`-sm layer -ts 59,41 --n-cpu-moe 48`, 3 streams: 45.3 t/s total with `GGML_MOE_CACHE_MAX_TOK=8`, 123.0 with the patch.

Output:

- Single-request greedy now equals the all-VRAM output: IQ3_XXS `45e68f1bdfeb` (with band 8 the host-expert path gave
  `9f826c79de59`), IQ3_S `c8528dd75a50` (78 % residency).  The 14-token test prompt now takes the same MoE MMVQ
  kernel the all-VRAM build uses at that width.  Plain decode == draft-mtp.
- Four identical concurrent greedy requests (IQ3_XXS) give the same four outputs as the all-VRAM build with no cache
  (`9d079b784a63`, `4db112aa75fe`, `f0989d10464d`, `0b16668b23ea`), so the band-16 cache path is bit-identical to
  all-VRAM under 4-way concurrency.  (They differ from the single-request output only through batching, identically
  in both builds.)
- 0 GPU faults in every run.

## Not measured

RDNA3 and NVIDIA hardware (the band stays 8 there by construction, since it follows `get_mmvq_mmid_max_batch`);
models with more than 16 experts per token (admissions past 256 in a pass stay cold - correct, fewer fills); the
`nwarps` campaign (env-off), whose per-width MoE choice would now also see 9-16-token batches on the cache path.

## Next

Batches above 16 tokens still bypass the arena: 17-63 tokens take the host used-expert copy, and >= 64 tokens the
whole-table staging, which re-uploads experts that are already resident.  I'll keep working on that side and send it
separately if it works out.
