# Tensor-split expert parallelism: split the MoE expert weights instead of mirroring them

**Status:** opened 2026-09-27, immediately after the op-offload H2D staging work was promoted into the
delivery as the block-06 r12 amendment (`v16-84e76d8a2-r12`).  **Nothing implemented yet** — this file
records the identified root cause, the required work and the cheap first experiment, so it is not
re-derived.

**Not a blocker for anything.**  TODO item 24 (the staging ring's inertness under `-sm tensor`) is
closed: the ring engages there now.  This is a *further* optimisation of that path, and its payoff has
to be measured rather than assumed.

## 1. The problem

Under `-sm tensor` + `-ncmoe N` (host-resident MoE experts, op-offloaded) every device receives the
**whole** expert tensor — their meta split state is `MIRRORED` — so both devices compute the full MoE
and the upload is duplicated.  Measured on gfx1201 with 2× R9700, 35B-A3B UD-Q4_K_M:

| config (pp8192, `-fa 1`) | ub 8192 | ub 2048 |
|---|---|---|
| `-sm tensor -ncmoe 99`, staging on | 2742 t/s | 716 t/s |
| `-sm layer -ncmoe 99`, staging on | **4111 t/s** | — |
| `-sm tensor -ncmoe 0` (experts on the device) | 8022 t/s | — |

A layer split wins because each device holds half the experts; a tensor split holds all of them twice.

## 2. Root cause (identified, three parts)

1. **The split state of the offload input copy is `MIRRORED` by default.**  The scheduler creates the
   split input with `tensor_copy()` → `ggml_dup_tensor_layout()`, which has `op == GGML_OP_NONE`, in a
   **COMPUTE** buffer.  In `ggml-backend-meta.cpp`'s `calculate_split_state()`, the weights branch asks
   the per-tensor policy callback, but the `COMPUTE` branch propagates from the op — and
   `case GGML_OP_NONE` returns `{GGML_BACKEND_SPLIT_AXIS_MIRRORED, …}` unconditionally.  **llama.cpp's
   per-tensor policy never sees the offload copy at all.**
2. **That policy does ask for a real split.**  `llama_meta_device_get_split_state()`
   (`src/llama-model.cpp`) returns axis **1** for `blk.N.ffn_{up,gate}_exps.weight` and axis **0** for
   `blk.N.ffn_down_exps.weight` (axis 0 of `[n_ff_exp, n_embd, n_expert]` is the **contraction** dim).
   So the intent exists; the copy just does not inherit it.
3. **The split-state machine has no rule for it.**  `handle_mul_mat()` covers
   `(MIRRORED, MIRRORED)`, `(axis1, MIRRORED)`, `(MIRRORED, axis1)`, `(axis0, axis0)`,
   `(axis≥2 == axis≥2)`, `(axis≥2, MIRRORED)` — and then **aborts**.  There is no
   `(axis0, MIRRORED)` rule, and a contraction-dim split needs a `SPLIT_AXIS_PARTIAL` output plus a real
   all-reduce rather than a plain split.

Useful anchors: `calculate_split_state()` / `case GGML_OP_NONE` / `handle_mul_mat()` in
`ggml/src/ggml-backend-meta.cpp`; `llama_meta_device_get_split_state()` in `src/llama-model.cpp`;
`ggml_backend_sched_backend_id_from_cur()` + the input copy in `ggml/src/ggml-backend.cpp`.

## 3. The cheap first experiment

Make the policy callback answer for the offload copies and see where the split-state machine stops.
The copy is named `<backend>#<src name>#<copy>` (`ggml_format_name(tensor_copy, "%s#%s#%d", …)` in
`ggml_backend_sched_alloc_splits`), so the meta side can strip the `<backend>#` prefix and the trailing
`#N` and reach the same regexes the policy already uses.  Two candidate shapes:

* **meta-side:** in `calculate_split_state()`, for `op == GGML_OP_NONE` with no sources and a `COMPUTE`
  buffer, consult `dev_ctx->get_split_state()` with the normalised name instead of returning `MIRRORED`;
* **llama.cpp-side:** strip the prefix/suffix inside `llama_meta_device_get_split_state()`.

Activations also arrive as `GGML_OP_NONE` copies, so the fallback must stay `MIRRORED` — the policy
returns that for anything it does not match, which makes this safe, but it must be checked explicitly
(a wrongly-split activation is a silent wrong answer, not a crash).

The first `GGML_ABORT` after that names the `handle_mul_mat()` rule to write.  That is the whole point
of doing it this way: the split-state machine is `GGML_ASSERT`-heavy, so it *tells* you what it needs
instead of failing quietly.

## 4. Full work list

1. Seed the offload copy's split state from the weight it receives (§3).
2. Add the missing `handle_mul_mat()` `MUL_MAT_ID` rules: contraction-dim split → `SPLIT_AXIS_PARTIAL`
   + the meta backend's all-reduce; output-dim split → a plain split; expert-dim split (axis 2) →
   `PARTIAL` + all-reduce.
3. Fix the per-device granularity so the split respects the quantised block size and the expert
   boundaries (`get_split_granularity()` in the policy).
4. Validate: this path has **no test-backend-ops coverage** (`MUL_MAT_ID` there is the dense case), so
   it needs same-seed greedy coherence, a perplexity ratio against the mirrored build, and the `-ncmoe`
   prefill/decode numbers at several `ub`.

## 5. Payoff — measure, do not assume

Splitting halves each device's upload (~144 → ~72 MiB per expert tensor per ubatch) but adds a
cross-device reduction of the MoE output per layer.  On the x4 link here the upload dominates, so it
should win; on a fast link (the reporter's 55 GB/s PCIe5 x16) it could be a wash or a loss.  The one
number that decides it is `-sm tensor -ncmoe 99` prefill t/s at ub 2048-8192 versus the current 716 /
1414 / 2742, measured against `-sm layer`'s 4111.

## 6. Related

* `archive/work/h2d-staging-ring/` — the promoted staging work (block-06 r12) this builds on; its
  HANDOVER is the previous brief.
* `upstream/UPSTREAM-PR-meta-offload-op.patch` — the upstream-standalone part of that fix.
* `TODO.md` item 26 — the tracker entry.

## 7. Environment, build, repro (start here next session)

* **Build tree: `~/llama-r12`** — the r12 delivery chain (tip `de71ddd58`), a worktree of
  `~/llama.cpp/.git`, on branch `wip-tensor-split`, with `build-rocm/` already built.  **Do not use
  `~/llama.cpp`** (its working tree is the older r11 + the pre-fold WIP patch, now stale) or
  `/tmp/canon-fix` (the r11 reference chain).
* Build: `cd ~/llama-r12 && CCACHE=0 BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714` (~6 min, 16 cores).
  `CCACHE=0` because the HIP build emits no `.d` files and ccache's direct mode has silently reused
  stale objects after a header edit on this tree.
* Runtime libs come from the binary's RUNPATH (`/opt/rocm-7.14.1-gfx102X/lib`); the similarly-named
  `/opt/rocm-7.14-gfx1201` is **not** the tree this build uses.
* Models: `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21.10 GiB, the `-ncmoe`
  target), `~/Qwen3.5-4B-Q8_0.gguf` (width probe), `prompts/prose-rdna-boosts.txt`.
* **Repro of the current (mirrored) behaviour** — the §1 numbers, 2 GPUs:
  ```bash
  cd ~/llama-r12
  M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  for ub in 2048 8192; do
    for sm in tensor layer; do
      echo -n "-sm $sm ub $ub : "
      HIP_VISIBLE_DEVICES=0,1 GGML_SCHED_STAGE=0 ./build-rocm/bin/llama-bench -m "$M" -ncmoe 99 -fa 1 \
          -p $ub -n 1 -b $ub -ub $ub -sm $sm -r 1 2>&1 | sed -n 's/.*pp'"$ub"' *| *\([0-9.]*\).*/\1/p'
    done
  done
  ```
* **Reproduce the pre-op-offload baseline** without a rebuild — make every device refuse offload:
  `GGML_OP_OFFLOAD_MIN_BATCH=1000000` sends the MoE back to the CPU (the 523 / 508 t/s state).
* **See the split states and the gate**: `GGML_SCHED_DEBUG=2 ... -v 2>&1 | grep MUL_MAT_ID` (that is how
  the missing op-offload was found).  `GGML_LOG_INFO` needs `-v` in the tools, and `llama-bench` installs
  `llama_null_log_callback` whenever `-v` is absent, discarding *every* level — use `llama-cli` or
  `llama-server` when you need log output.
* **Text hashes: run `llama-cli` WITHOUT `-lv 4`.**  `scripts/extract-generated.py` already warns about
  this; `-lv` interleaves timing lines into the extracted text, so the hash becomes run-dependent.  It
  cost a full false "purity failure" during the r12 gate — read the script's docstring first.
* `halo` (gfx1151) is deliberately excluded from `-ncmoe` work (unified memory).
* **The r12 gate baseline to diff against** (all on the frozen r12 tree, gfx1201): `FLASH_ATTN_EXT` 2/2,
  `MUL_MAT_ID` 2/2, `W=1..8` f16 pure / q8_0 at the documented `{W=1}` edge, `-sm tensor` greedy text
  `0936c8318533` and `-sm layer` `f90525c438c4`, MoE MTP `2a7439c54eb7` (acceptance 0.70612),
  `-sm tensor` PPL 14.4657, 32k prefill 1273 t/s, server soak 187 s / 18-18.
