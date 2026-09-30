# wip/issue72 — GDN state arrives MIRRORED to a mixed (AXIS_1) meta handler

**Status: candidate fix, NOT validated against the reported crash.**  Opened 2026-09-30 as the
investigation of issue #72 (@WillWorker): a 3-GPU `-sm tensor` qwen4exp run with per-slot ctx above the
model's native 262144 aborts in `handle_gated_delta_net` because the GDN recurrent state (`state_predelta`)
arrives `MIRRORED` while the q/k/v/g/beta inputs arrive `AXIS_1` (head-split).

This is an **upstream meta-backend gap**, not boosts-specific: `handle_gated_delta_net` is identical in
upstream master (`ggml/src/ggml-backend-meta.cpp`, master lines 813-827).  It is kept here rather than
folded into the delivery because the reporter's crash could not be reproduced on the maintainer's box (see
"Reproduction" below), so the candidate patch has not passed a gate.

## Mechanism (static analysis)

1. The GDN op (`GGML_OP_GATED_DELTA_NET`) reads the recurrent state as `src[5]`, shape
   `[S_v, S_v, H, n_seqs]`.  The handler requires a head split there (axis 0/1/2 of the 4-D state).
2. When the graph is a single scheduler split (the normal case), `state_predelta` is a computed activation,
   so `ggml_backend_meta_get_split_state` derives its split state from the `cache_s` producer.  The
   reshape places the head-aligned split on **axis 2**, and the assert passes.
3. When the graph splits into two or more meta splits, the scheduler creates a boundary copy of the state:
   `Meta(ROCm0,ROCm1,ROCm2)#state_predelta-<il>#<c>` (`ggml_backend_sched_split_graph`,
   `ggml_dup_tensor_layout` + `ggml_format_name`).  The copy has `view_src == nullptr`, `op == GGML_OP_NONE`
   and lives in the COMPUTE buffer, so it takes the **per-tensor policy** path.
4. The policy (`llama_meta_device_get_split_state`, `src/llama-model.cpp`) strips the
   `<backend>#<name>#<n>` wrapper to `state_predelta-<il>` and matches it against the model-tensor regexes.
   It matches none of them, so it falls through to the "everything else" arm and returns
   `GGML_BACKEND_SPLIT_AXIS_MIRRORED`.
5. `handle_gated_delta_net` has an all-six-mirrored arm and a head-split-input arm, but no mixed
   "AXIS_1 inputs + MIRRORED state" arm, so it aborts on the `src_ss[5]` assert.

The reporter's threshold (stable at per-slot 131072, abort at >= 171994) is consistent with a memory-driven
graph split: at the larger per-slot ctx the KV cache leaves less per-device compute headroom, so the
scheduler splits the graph and the boundary copy appears.  The reporter's "the allocator mirrors the state"
hypothesis is half right: the state is not re-allocated, it is a **name-policy default** on the scheduler
copy.

## Reproduction

Not reproduced here.  3x R9700 / gfx1201, ROCm 7.14, `Qwen3.8-Flash-Next-UD-IQ4_XS` (89 GB, the nearest
local model), the reporter's server config (`-sm tensor --parallel 2 -fa on -b 128 -ub 128 -ctk bf16
-ctv bf16 --kv-unified-per-slot 172032 -c 344064`), a 126640-token prompt: prefill completes at
~447 t/s with no abort.  `rocm-smi` showed ~3.5 GB free per device after the run versus the reporter's
~1-2 GB, and an external 5 GB/device VRAM hog (total ~33 GB/34 GB used) still did not produce the abort,
just a slower prefill.  The reporter's model loads at 97.7 GB and is the likely reason their graph splits
while this one does not.  Next: run the reporter's exact `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF` IQ4_XS
or find a forced-split repro.

## Candidate fix

`issue72-candidate-state-predelta-split.patch` (one hunk in `src/llama-model.cpp`): classify the
`state_predelta-<il>` policy copies with `GGML_BACKEND_SPLIT_AXIS_2`, parsing the layer index from the name
so the rotation matches `get_il_eff(il)` like the `cache_s` config.  This is fix family 1 from the issue
(keep the state head-split); the state is ~6 MB/slot so mirroring it saves nothing, and the CUDA GDN kernel
reads `src_state->data` at the per-device head offset, so a head-split copy is both correct and cheap.

Family 2 (accept MIRRORED + head-offset views) is larger and touches the kernel's state write-back, so it
was not attempted.

Verified so far: the fork builds warning-free with the patch and the 4B `-sm tensor` coherence hash is
unchanged (`1c5d32ac537d`).  The branch never fires in the single-split case, so the reported crash path
still needs a reproduction to confirm.
