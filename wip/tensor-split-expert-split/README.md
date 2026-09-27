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

## 8. Findings, first probe (2026-09-26/27) — the premise needed correcting

The §3 probe was run on the r12 tree (`~/llama-r12`). It **reached the split-state machine and was
stopped by an assert** (which is what §3 wanted), but the measurements taken on the way there change how
this campaign should be framed. Patches and instruments: `exp1-split-copy-abort.patch` (4 files, +81).

**1. The upload is *not* the whole tensor.**  §1's "every device receives the whole expert tensor" is
wrong in an important way: the scheduler's **used-expert pruning** is active (the same `copy_experts`
path the staging comment in `ggml-backend.cpp` refers to). It uploads only the *used* experts, as
**consecutive groups**, plus a trailing `min(expert_size, 512)`-byte pad for MMQ's over-read. Measured
first group, `blk.0.ffn_gate_exps.weight`, ub 2048: **25 of 256 experts + 512 bytes**.

Device-side H2D volume for `-p 2048 -ub 2048 -r 1` (2× R9700, instrumented at the CUDA H2D funnel):

| | device 0 | device 1 |
|---|---|---|
| `-sm tensor -ncmoe 99` | 30.0 GiB | **30.0 GiB** |
| `-sm layer  -ncmoe 99` | 30.7 GiB | 1.1 GiB |

So the duplication is real (each device gets the same used-expert set), but the *per-device* volume is
the pruned set, not the full weights.

**2. The 2× gap is compute, not bytes.**  Staging overlaps the upload and is the cheapest way to price
it: at ub 2048 it moved `-sm tensor` 650.4 → 715.5 t/s (**+10 %**), and `-sm layer` 1262 → 1432
(+13 %). So the upload is ~10 % of the time in both modes — it cannot explain 715 vs 1432. The reason
tensor mode is slower is that the weight **copy** is `MIRRORED`, so **each device executes the whole MoE
(all 256 experts, every token)**. Compare `-sm tensor -ncmoe 0`, where the same policy splits the
weights on-device (axis 1 / axis 0, verified below): **8022 t/s** at the same ub. The offload path is
~11× off the on-device split, and only ~2× off `-sm layer`.

**3. `-sm layer` runs the whole MoE on ONE device.**  `ggml_backend_sched_backend_id_from_cur()`'s
op-offload loop returns the *first* eligible backend (`for b = 0; b < src_backend_id; b++` → always
`ROCm0`), which the device-side table above confirms (30.7 GiB on device 0, 1.1 GiB on device 1). So
"a layer split halves the expert work per device" is false — it does *all* of it on one device and still
wins by 2×. The tensor path must therefore be losing more than duplication, and the split itself is not
obviously worth more than the difference to `-sm layer`.

**4. Decode runs the MoE on the CPU in both modes.**  With `-ncmoe 99`, `MUL_MAT_ID` is refused for
op-offload below the 32-batch threshold (`get_op_batch_size()` is `op->ne[2]`, which is `n_tokens` = 1 at
decode), so the decode graph's `ffn_moe_gate/up/down` are assigned to **CPU** — in `-sm layer` too, so it
is not a tensor-split regression. Only prefill offloads. Worth its own look later.

**5. The blocker, exactly.**  With the copy inheriting the policy's axis (1 for
`ffn_gate/up_exps.weight`), the first expert-weight upload aborts:

```
ggml-backend-meta.cpp:2082: GGML_ASSERT(size % chunk_size_full == 0) failed
name=Meta(ROCm0,ROCm1)#blk.0.ffn_gate_exps.weight#0 axis=1 offset=0
size=14746112 chunk_full=589824 nbytes=150994944 ne=[2048,512,256,1]
```

`14746112 = 25 × 589824 + 512` — the 512-byte MMQ pad is not a whole `nb[axis+1]` chunk. The assert is
correct: the spliced path can only express whole chunks.

**Good news from the probe:** the policy *is* consulted and *does* answer for the copy (axis 1 for
`ffn_*_ups/gate_exps`, axis 0 for `down_exps`), the split states then propagate through `swiglu` →
`MUL_MAT_ID` exactly as they do on-device (`-ncmoe 0`: `ffn_moe_gate` axis 0 → `swiglu` axis 0 →
`ffn_moe_down` axis 0 → `(axis0, axis0)` → resolved MIRRORED after the partial sum), and **no abort
happens anywhere else** — the whole activation-copy set (398 op-NONE compute tensors) is unaffected, so
the blast radius of the enablement is just the weight copies.

## 9. Revised plan

The goal is now **halving the compute** (each device computing its slice of every expert), with the
upload reduction as a bonus — not the other way round. Order of work:

1. **Price the split before building it.** The gap to `-sm layer` is 2× and the gap to the on-device
   split (`-ncmoe 0`) is 11×. Measure how much of the `-sm tensor` cost is the *pruning machinery*
   (ids readback + a full `ggml_backend_synchronize`, which on the meta device syncs both GPUs) rather
   than duplication: run with the pruning bypassed (whole-tensor staging, `GGML_SCHED_STAGE=1
   GGML_SCHED_STAGE_MIN_TOKENS=0`) and also instrument `stage_upload` so the ring's coverage is
   visible. If bypassing the pruning recovers most of the gap, the split is not the lever.
2. **Make the split upload tolerate the padding.** Round the byte range to whole chunks (clamped at
   `nbytes`) or copy the pad separately, and handle the tail group (a group ending at the last expert
   reaches exactly `nbytes`, so rounding up would read past the source). The pad exists so MMQ's
   over-read is not NaN, so a split copy must leave each device's slice with non-NaN slack too — that
   is the correctness risk to think about first.
3. **Then** consider the expert-parallel split (axis 2), which needs new split-state machinery
   (`MUL_MAT_ID`'s output axis 2 is `n_tokens`, not experts) but needs no reduction at all.
