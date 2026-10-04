# issue #99 - the gemma4 `-sm tensor` guard can be relaxed (upstream fixes)

**Status:** **RESOLVED in `v16-a55e952b8-r9` (block 14).**  All-resident and `-ngl`-offloaded gemma4
now split under `-sm tensor`; only a host-resident expert table (`-ncmoe`/`-cmoe`) and the gemma4
MTP head (`gemma4-assistant`) are rejected, both with a clean message.
**Issue:** https://github.com/stew675/llama-cpp-rdna-boosts/issues/99

## Report

The block-14 `llm_arch_supports_sm_tensor()` gate (`src/llama-arch.cpp`) rejected `LLM_ARCH_GEMMA4`
outright, because gemma4's fused expert tensor (`ffn_gate_up_exps`) has a SEGMENTED split layout
(`n_segments`/`nr`) and the per-ubatch upload of a host-resident MoE had no correct tensor-split
path.  The guard's own comment said *"the all-resident case is fine"*, but the arch-level rejection
made that configuration unreachable.  The reporter (WillWorker) ran a fully GPU-resident
gemma-4-31B with the single `case LLM_ARCH_GEMMA4:` line removed and it loaded, built its graph and
served cleanly in tensor mode.

## Why the guard existed, and why the core of it is gone

The original reject came from two upstream fused-QKV split bugs, **both now fixed in the
`a55e952b8` base**:

* **#28965** (`fb27a525d`) - *TP: fix split state and granularity for fused QKV gemma4, qwen35*:
  gemma-4-31B has `n_embd` 5376 but Q 8192, so the old handler tripped
  `GGML_ASSERT(tensor->ne[axis] == n_embd + 2*n_embd_gqa)`; the split is now computed from
  `n_head * n_embd_head_k`.
* **#29294** (`f805c57a2`) - *fix tensor split for fused qkv with uneven K/V head sizes*: gemma4 is
  32 Q / 16 KV heads (GQA 2:1).

The remaining hard crash was never the attention split; it is the **host-resident expert upload**
of the segmented `ffn_gate_up_exps` (see below).

## Investigation (gfx1201, 3x R9700, ROCm 7.14.1)

Built the `a55e952b8` delivery tree with the gemma4 line removed and exercised the paths with
`--seed 42 --temp 0`:

| config | result |
|---|---|
| gemma-4-26B-A4B Q8_0, `-ngl 99 -sm tensor`, no MTP | **works**; greedy output byte-identical to `-sm layer` |
| gemma-4-31B Q6_K, `-ngl 99 -sm tensor`, no MTP | **works**; greedy output byte-identical to `-sm layer` |
| gemma-4-26B-A4B Q8_0, `-ngl 40 -sm tensor` (whole layers on CPU) | works (exit 0) |
| gemma-4-26B-A4B Q8_0, `-ncmoe 40 -sm tensor -ub 128`, 600-token prefill | **aborts**: `GGML_ASSERT(split_state.nr[0] == 1)` at `ggml-backend-meta.cpp` |
| gemma-4-31B Q6_K + `-md gemma-4-31B-it-Q8_0-MTP.gguf --spec-type draft-mtp`, `-sm tensor` | **aborts**: meta ratio assert at `ggml-backend-meta.cpp:1212` |

So the relaxation must keep two rejections:

1. **Host-resident experts** - `-ncmoe`/`-cmoe` pushes a CPU-buffer override for the segmented
   expert tensor; `ggml_backend_meta_set_tensor_async` only understands a contiguous
   single-segment slice, so `GGML_ASSERT(split_state.nr[0] == 1)` fires at the first upload.  A low
   `-ngl`/`--fit` offload is *not* this case (whole layers sit on the CPU backend with no per-ubatch
   upload) and is fine.
2. **The MTP head** - `gemma4-assistant` shares the target's KV (`is_mem_shared`), so it must use the
   target's split.  The draft is a single layer pinned to rotation 0 while it reads the target's
   rotating KV caches, so the meta ratio check asserts.  `-sm layer` is the working MTP config.

## The fix (block 14)

* `src/llama-arch.cpp`: `LLM_ARCH_GEMMA4` removed from the false-list; `LLM_ARCH_GEMMA4_ASSISTANT`
  added to it (the MTP head fails cleanly with `not implemented for architecture 'gemma4-assistant'`).
* `src/llama-model.cpp`: `llama_model_create()` rejects a gemma4 tensor split only when
  `llm_params_have_host_expert_override(params)` - it tests the `tensor_buft_overrides` patterns
  against a gemma4 expert weight name (`blk.0.ffn_gate_up_exps.weight` / `ffn_down_exps.weight`) and
  checks the target buffer is not a GPU/IGPU - with a clear "use -sm layer" message.

No delivery behaviour changes for non-gemma4 arches.

## Validation

* gemma-4-26B-A4B Q8_0 and gemma-4-31B Q6_K, `-ngl 99 -sm tensor`: byte-identical to `-sm layer`.
* `-ngl 40 -sm tensor`: still runs.
* `-ncmoe 40 -sm tensor`: clean rejection (was an abort).
* gemma4 MTP + `-sm tensor`: clean rejection on `gemma4-assistant` (was an abort); MTP + `-sm layer`
  still works.
* Dense 4B 3-GPU `-sm tensor` gate `1c5d32ac537d` unchanged; `scripts/validate-set.sh` green.

The full assert reproductions are recorded in `WORKLOG.md` 2026-10-04 (r9).

## Open follow-ups

* Validate on gfx1100 / gfx1151 (3-device RDNA3) gemma4 all-resident tensor split.
* The real fix for the host-resident case is a segmented `ggml_backend_meta_set_tensor_async`
  (tracked with the MoE-expert-cache work); once it lands, the `-ncmoe` rejection can go too.
* The MTP head's rotation-0 mismatch would need the draft to compute on the target's rotated KV
  partition (or a rotation-aware split state) - out of scope for this gate relaxation.
