# upstream/ — standalone upstream-PR candidates

This directory holds candidates to be **upstreamed to ggml-org/llama.cpp as pull requests**: clean,
self-contained changes that could stand on their own.  Two kinds:

* **already inside a delivery patch** (`patches/` 0001-0014 and `beta/qwen4exp/...`) — here as a
  reference and as the starting point for a PR branch;
* **campaign wins staged in the beta Block 15** (`archive/work/block-15-campaign-wins/`) — here because they are upstream-applicable in
  their own right (if upstream takes one, Block 0015 must drop it on the next regeneration).

They are NOT additional deliverables: applying one of these on top of the full delivery
would double-apply the hunks.  Treat each as:

1. a reference for the exact upstreamable change (with its own README/notes), and
2. the starting point for a PR branch against upstream master (re-created from the
   file's current state in the fork, not blindly applied).

| file | what it is | where it lives in the delivery | status |
|---|---|---|---|---|
| `UPSTREAM-PR-fa-kv-split-width.patch` + `.md` | ggml-cuda: make the flash-attention KV-split (`parallel_blocks`) heuristic query-width independent for small batches — decode and every speculative verify width then reduce identically, so greedy output no longer changes with the MTP draft length | **delivery block 00** (structural and architecture fixes); block 00 also carries the Vulkan masked-V fixes | prepared 2026-09-10; validated on gfx1201 against the delivery (issue #25: `--spec-draft-n-max 2` == `4` on 2/3-GPU, adaptive == both, plain decode byte-identical, acceptance gate holds); applies clean to `9113cc188`; not filed |
| `UPSTREAM-PR-fa-decode-verify-kernel-family.patch` + `.md` | ggml-cuda: keep the decode/verify band on one flash-attention kernel family — a quantized K/V cache took VEC for `n_q <= 2` and TILE from `n_q = 3`, so greedy decode disagreed with spec-draft verify (F1) | **inside block 08** (2026-09-11 amendment) | written, validated (3x gfx1201); NVIDIA/Ada arm included |
| `UPSTREAM-PR-ggml-sched-probe.patch` + `.md` | a debug probe for the ggml backend scheduler (per-op fallback / sync tracking), core-ggml only | inside `beta/qwen4exp/qwen4exp-support.patch` (the sched-fallback-sync hunk) | prepared, not filed; re-verified 2026-09-10 to still apply clean to current master (`9cf3bf256`) |
| `UPSTREAM-PR-ggml-alloc-unused-view.patch` + `.md` | ggml-alloc: release view sources whose views are never consumed (a real leak; repro included) | **staged in beta Block 15** (`../archive/work/block-15-campaign-wins/`, win W4); A/B via `../archive/work/block-15-campaign-wins/ab/w4-revert.patch` | prepared 2026-09-10, applies clean to `9cf3bf256`, repro + `test-alloc`/`test-batch-alloc` verified **on master**; upstream-drop check 2026-09-10: still absent upstream; not filed |
| `UPSTREAM-PR-kv-cache-keys-only.patch` + `.md` | llama: keys-only KV caches -- `llama_kv_cache` gains `v_enabled` (no V tensor, no V-side op); the qwen4exp indexer store passes `false` (its V is dead: the indexer scores keys) | **staged in beta Block 15** (`../archive/work/block-15-campaign-wins/`, win W3) | prepared 2026-09-10; verified on **master** (`9cf3bf256`, CPU): applies clean, compiles, indexer KV **72.00 → 24.00 MiB** at ctx 8192 (K 24 / V 48 → K 24 / no V), same-seed text byte-identical, `test-alloc`/`test-batch-alloc` 0 failures; not filed |
| `UPSTREAM-PR-attn-k-null-mask-guard.patch` + `.md` | llama: `llm_graph_input_attn_k` tolerates an absent kq mask (guard the fill like the sibling `attn_kv` class does; let `can_reuse_impl` accept a null mask) | **staged in beta Block 15** (`../archive/work/block-15-campaign-wins/`, part of win W2) | prepared 2026-09-10; verified on **master** (`9cf3bf256`, CPU): applies clean, compiles, same-seed text byte-identical; **hardening only** -- no reachable null-mask path on master today (every construction site builds a mask); not filed |
| `UPSTREAM-PR-get-rows-iq4-nl-sub-qk-k.patch` + `.md` | ggml-cuda: `GET_ROWS` on an `iq4_nl` row that is not a whole `QK_K` super-block — the support predicate sends it to the CPU; add the sub-block `get_rows_cuda_q<QK4_NL, QR4_NL, dequantize_q4_nl>` path and accept `ne00 % QK4_NL == 0` | **inside block 08** (2026-09-13 sixth amendment, TODO item 3) | prepared 2026-09-13; applies clean to `790cf51aa`; validated on gfx1201 (GPU split count 142 -> 22, prefill +25-36 %, `GET_ROWS` 215/215 -> 219/219, op bit-exact vs CPU); no NVIDIA hardware was available; not filed (see the `.md` "what was NOT validated") |
| `UPSTREAM-PR-meta-offload-op.patch` + `.md` | ggml-backend-meta: let the tensor split op-offload host weights — the meta device left `offload_op` NULL, so under `-sm tensor` the scheduler never selected it for an op-offloaded node and `-ncmoe`/`--cpu-moe` executed the whole MoE on the CPU.  Also lets a `MIRRORED` tensor serve an arbitrary byte range (the used-expert pruning uploads at a non-zero offset and reads the router ids as a strided view's raw span, both of which the old `offset == 0` / `ggml_is_contiguous` asserts rejected) | **delivery block 06** (r12, 2026-09-27) — source tree `archive/work/h2d-staging-ring/`, patch `h2d-stage.patch` (fixes 1+2 of three) | prepared 2026-09-27; applies clean to master `84e76d8a2`; **built and measured standalone on pristine master + this patch**: `-sm tensor -ncmoe 99 -fa 1 pp8192 ub8192` **505.49 -> 1690.26 t/s**, `test-backend-ops -o MUL_MAT_ID` 2/2 OK, generation coherent; no NVIDIA hardware available; not filed |

| `UPSTREAM-PR-moe-host-expert-pinning.patch` + `.md` | llama: keep host-resident MoE expert weights pinned — `select_weight_buft`'s "avoid using a host buffer when using mmap" guard now skips the downgrade for `MUL_MAT_ID` weights, the tensors op-offload (`-ncmoe`) uploads every ubatch, so those uploads read pinned memory instead of the pageable model mapping | **delivery block 06** (r15, 2026-09-27) | prepared 2026-09-27; applies clean to master `84e76d8a2`; validated on gfx1201 2x R9700: **+83 %** `-sm tensor -ncmoe 99` pp8192 (2794 -> 5104 t/s) and bit-identical same-seed output; same finding ships in GenerelSchwerz's `moe-cache` fork; no NVIDIA hardware available; not filed |
| `UPSTREAM-PR-sched-cross-device-input-order.patch` + `.md` | ggml-backend: order a cross-device split-input copy after the destination backend's queued work — the copy runs on the source backend's stream, so with the allocator reusing an earlier split's output region it could overwrite that output before its outbound copy read it (garbage prefill output with `-sm layer` on 2 GPUs and host experts) | **delivery block 06** (r12, 2026-10-05, issue #103 / PR #104) | prepared 2026-10-05; the fork form is validated (FAIL -> PASS on 2 x R9700 with a prefill-rebalance harness, byte-identical elsewhere); the upstream form is re-cut against `a55e952b8` and applies clean; not built or filed against upstream yet |
| `UPSTREAM-PR-per-device-host-buffers.patch` + `.md` | ggml-cuda + llama + ggml-backend: per-device host buffer types and host-expert offload placement — `-sm layer` + `-ncmoe` sent every MoE op to device 0 (the host buft was a device-0 singleton, the `ctx_key` comparator merged same-name bufts, and the op-offload loop returned the first backend) | **delivery block 06** (r14, 2026-10-05; folded in from r13's block 16) | prepared 2026-10-05; applies clean to `a55e952b8`; validated on gfx1201 2 x / 3 x R9700 (`-sm layer -ncmoe 48` 10.3 -> 58.6 t/s at 2 GPUs, 81.0 at 3; single-device unchanged; MUL_MAT_ID 931/931, QSA 26/26, acceptance 0.889); no NVIDIA hardware available; not filed |

When filing, re-create the branch from upstream master and re-run the file's own Validation section —
do not blindly apply the copy, and never apply one of these on top of the full delivery (that would
double-apply the hunks).

If a hunk here is ever accepted upstream, it should be dropped from the delivery patch
set on the next regeneration (the delivery then carries only the fork-local remainder).

**Backlog:** the FA KV-split width fix (above) is written up and validated against the delivery.  A
second candidate — the masked-V / freed-cell kernel fixes (HIP `fattn-tile.cuh`, HIP
`fattn-mma-f16.cuh`, Vulkan `flash_attn.comp`/`flash_attn_cm1.comp`, today folded into block 14) — is
already validated on gfx1151 + Vulkan but has **not** been re-cut against unadulterated master yet
(the HIP halves sit on the fork's native-bf16/WMMA FA path, so the upstream form needs a fresh port).
The four older candidates (the sched probe, the allocator view-release, the keys-only indexer cache,
and the `attn_k` null-mask guard) are written up above, each with its own `.md` evidence and an
explicit "what was not validated" section.  Re-run that section on the PR branch before filing; a hunk
accepted upstream is dropped from the delivery at the next regeneration.
