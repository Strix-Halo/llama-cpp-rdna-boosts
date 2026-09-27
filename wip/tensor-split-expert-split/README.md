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

## 10. Second probe (2026-09-26/27): the split is implemented and propagates; the partial reduce faults

Patch: `exp2-split-copy-implemented.patch` (`~/llama-r12`, branch `wip-tensor-split`). Three parts:

1. **`calculate_split_state` consults the policy for copies** (§8 item 5) — `policy_tensor` also covers
   `op == NONE` with no view source in a COMPUTE buffer. No other tensor is affected (the 398 activation
   copies still resolve to mirrored).
2. **The policy strips the copy wrapper** (`<backend>#<name>#<n>`) so a copy inherits its weight's split.
   Gated by `GGML_META_SPLIT_COPY`: `0` = r12 behaviour (mirrored), `1` = all weight copies (default),
   `2` = axis-1 copies only (the bisection).
3. **The spliced upload tolerates the scheduler's pruned ranges.** `copy_experts` copies whole experts
   plus a trailing `min(expert_size, 512)`-byte MMQ pad, one range per group of *consecutive* used experts,
   so `offset` is chunk-aligned but non-zero and `size` need not be a whole number of chunks. Two facts
   had to be got right, and both were wrong in the first two attempts:
   * `data` is passed **already advanced** to the range start (`input->data + expert_offset`), so the
     source must not add `i_start` again — it is a *destination* offset only. (That silent ambiguity is
     why the original code asserted `offset == 0`.)
   * the destination must be rebased per device: chunk `k` is at `k*chunk_size_full` in the meta tensor
     but `k*chunk_size_j` in device `j`'s simple tensor.
   Verified geometry: meta `[2048,512,256,1]` chunk 589824; simple `[2048,256,256,1]` chunk 294912 × 2 =
   589824 ✓. With `offset` implied as destination-only, the change is a strict no-op for aligned uploads.

**Result.** `GGML_META_SPLIT_COPY=0` reproduces the r12 baseline exactly (716.59 t/s, ub 2048, no fault).
`=2` (axis-1 copies only) now aborts on a **missing split-state rule**, and that is a finding in itself:

```
unsupported mul_mat split states: node=ffn_moe_down-0
  src0=Meta(ROCm0,ROCm1)#blk.0.ffn_down_exps.weight#0  axis=10 (MIRRORED)
  src1=ffn_moe_swiglu-0                                axis=0
```

`handle_mul_mat` has `(axis0, MIRRORED)` (a split contraction weight with a mirrored activation) but **not
the reverse, `(MIRRORED, axis0)`** — so a *mirrored* down weight cannot consume a split activation. So the
axis-1 and axis-0 halves cannot be enabled independently: either both copies split or neither.

`=1` (all copies) passes every split-state check — **no assert fires anywhere** — and then takes a
**memory access fault** (device 1) on the first forward pass, at both `-ub 2048` and `-ub 8192` (so it is
not the pruned-range rebasing: at 8192 the range is one full-tensor offset-0 copy). That isolates the
remaining blocker to the **runtime partial reduction** for a host-resident (copied) expert weight — the
`(axis0, axis0) → PARTIAL` + all-reduce path. Prime suspect: the reduce machinery's per-backend temp
buffers / subgraph split are sized and sequenced from the *static* split container, and a split-state that
comes from a COMPUTE-buffer copy may not get them set up. Next step is to instrument
`ggml_backend_meta_graph_compute`'s reduce step (n_reduce_steps, `max_tmp_size`, `node_red`) for the
copied weight rather than guess.

**Debugging rule learned here (do not repeat):** never run this path with `HIP_LAUNCH_BLOCKING=1`. The
meta backend's cross-device `hipStreamWaitEvent` progress needs asynchronous launches; blocking launches
deadlock it (one GPU busy, the other idle) — which is a diagnostic artifact, not a property of the code.
`AMD_SERIALIZE_KERNEL` alone is no better. And clean up with `pgrep -x <binary>`, not `pkill -f "<name>"`,
whose pattern matches the shell running it.

## 11. The ring is not the problem, and it is already per-card (2026-09-27)

Two corrections and one measured fact.

**Correction: the PCIe links are independent per card, not one shared x4.**  `soar` gives each R9700 its
own 4 lanes (the BIOS caps each slot at x4); the lanes are not shared between cards, and the 14.45 GB/s
calibration is a **per-device** number (which is how `h2d_gbps` is measured — min over simple backends, so
the block-06 gate math is unaffected).  A duplicated upload therefore runs at ~14.45 GB/s *per card in
parallel*, so the duplication is close to free in wall time once overlapped — earlier notes here that read
the 2-GPU exposed-upload figure (1.43 s vs 1.15 s for 1 GPU) as shared-link contention were wrong; that
ratio is what independent lanes predict.

**Do we need one ring per card?  No.**  There is one logical slot index, and **each simple backend owns its
own slot arena** (`h2d_stage[]` is per CUDA context, allocated through `stage_buffer(simple_backend, slot,
size)`), with per-device done/free events (`stage_done_ev[device][slot]`).  Slot `k` therefore means "slot
`k` on every device" — the right shape for one logical upload that fans out to N cards.  Two rings would
only add two slot counters to keep in sync.

**Measured (`GGML_RING_STATS`, ub 8192, `-sm tensor -ncmoe 99`, 2 GPUs):** 480 upload waits and
**545 µs total** blocked on the upload — the ring hides it essentially perfectly.

| ub 8192, `-sm tensor` | t/s | per pass | upload exposure |
|---|---|---|---|
| stage 0 | 1909 | 4.29 s | exposed (the 1.29 s the ring saves) |
| stage 1 | 2735 | 3.00 s | 0.5 ms — hidden |
| 1 GPU, identical work | 5740 | 1.43 s | hidden |

So with the ring doing its job, **the two-device mirrored compute is 2.1× the single-device compute of the
same work** — that is where the gap lives, and it is exactly what halving each device's work would remove.
It also confirms the campaign's premise (the lever is the compute) in the cleanest way available.

**Consequence for the split, to design for before building it:** `ggml_backend_meta_stage_input` accepts only
the **flat** case (`chunk_size_full == size`), which holds for a `MIRRORED` copy but is false for a split one
(`nb[axis+1] != nbytes`).  So enabling the split would silently drop the expert uploads off the ring and
re-expose them, giving back the 1.29 s the ring currently saves at ub 8192.  The spliced upload therefore
needs a ring-compatible form too (one chunk per device, as the arena is already per device), or the split
must be proven to win by more than that exposure.

## 12. Third probe: the fault is localized, and a new structural cost is measured (2026-09-27)

Patch: `exp3-fault-localized.patch`. Instruments: `GGML_META_EXECDBG` (subgraph/device/node dumps),
`GGML_META_BUFDBG` (simple-tensor data offsets vs device buffer sizes), `GGML_META_GCDBG`
(host time inside `ggml_backend_meta_graph_compute`).

**Determinism (asked, and replicated).**  The copy is not hand-split: it gets its state from the *same*
policy function (`llama_meta_device_get_split_state`) with the same inputs (stripped name, layer index,
`tensor_split`, device count), so `ne[]`, `nr[]`, the per-layer `rotation`, and the granularity come out
identical to the `-ncmoe 0` weight.  That is *required*, not incidental: the gate/up split (axis 1) and the
down split (axis 0) must yield **matching per-device segment sizes** for the partial reduce to line up,
which is what `handle_mul_mat`'s `split_states_equal` checks.  Verified on gate_exps: the static weight's
`{256x1, 256x1}` and the copy's simple tensors `[2048,256,256]` (256/device) agree.

**Fault localized to `MUL_MAT_ID` on the split copy.**  `GGML_META_EXECDBG` shows the last child graph
launched before the fault is a **one-node** graph, `ffn_moe_gate-3(MUL_MAT_ID)`, on both devices.  So the
`mul_mat_id` kernel faults when fed the split weight copy.

**Hypothesis disproved.**  The simple tensor's data pointer is `base(simple_buf) + (tensor->data -
base(tensor->buffer))` — the same offset within each device's buffer.  `GGML_META_BUFDBG` shows every
expert copy is comfortably in range (`off + nbytes(t_ij)` ~115 MB against a 1.65 GB per-device compute
buffer), so this is *not* an out-of-bounds device buffer.  Geometry, strides (`nb[2]` scaled by the device
fraction — 294912 × 2 = 589824 ✓), split state and range are all verified correct.

Prime remaining suspect: the meta **compute-container lifecycle** for a *split* tensor.  The compute
container is double-buffered and cleared per `graph_compute`, and with op-offload it is cleared ~120 times
per pass (see below), whereas a `-ncmoe 0` weight lives in the *static* container that is built once.  A
stale/re-used simple tensor across those rebuilds is exactly what would surface as a device fault here.

**New structural finding: the op-offload path hands the meta backend one graph per op.**  `EXECDBG`:
`n_subgraphs=1 n_backends=2 reduce_steps=1` then a 1-node subgraph.  So `ggml_backend_meta_graph_compute`
runs ~120 times per pass (3 ops × 40 layers), each doing the container swap+clear, subgraph analysis,
per-node simple-tensor lookup, fusion checks, temp-buffer checks and event wiring for a single op.

**Measured cost of that (ub 8192, 2 GPUs, stage 1):**

| mode | t/s | per pass | METAGC |
|---|---|---|---|
| `-sm tensor` | 2736 | 2.99 s | **324 calls, 1194.9 ms host, 3.69 ms each (~0.44 s/pass, ~15 %)** |
| `-sm layer` | 4113 | 1.99 s | none — the meta backend is not used |

So ~15 % of the tensor pass is host-side meta setup that exists in no other mode, and it is *not* compute.
This also explains part of why 1 GPU (5740, no meta device at all) beats both split modes.

**Why this matters for the plan.**  The remaining gap is larger than this overhead (the 2-GPU mirrored
pass is 2.99 s against 1.43 s for one card), so this is not the whole story — but it is a real, separately
fixable inefficiency, and it suggests looking at *how the offload dispatches work* (one graph per op)
before building more split machinery.  The two candidate levers are now: (a) reduce the number of
op-offload graphs / the meta per-graph cost, and (b) the split (halving each device's work), which still
needs the compute-container fault fixed.

**Debugging rules re-confirmed:** `AMD_SERIALIZE_KERNEL=3` deadlocks this path exactly like
`HIP_LAUNCH_BLOCKING=1` (one GPU pegged at 100 %, the other idle) — never use either; and a crashed tool's
buffered stdout is lost, so run with `stdbuf -o0 -e0` when you need to see how far it got.

## 13. Fourth probe: the tensor path's penalty is a FIXED per-node dispatch cost (2026-09-27)

Patch: `exp4-dispatch-overhead.patch`. Instruments: `GGML_META_GCDBG` gained a per-phase breakdown, and
`GGML_META_SSDBG` / `GGML_META_SS_FIX` A/B the split-state cache.

**Why this matters (the maintainer's framing, confirmed by measurement).**  `-sm tensor` is the *better*
mode in general: with the experts on-device it beats `-sm layer` at **prefill** (7732 vs 6422, ub 8192) and
at **decode** (`tg64 @ d16384` 76.87 vs 74.83).  For dense models the lead is large, and the maintainer's
rule of thumb holds here: **the greater the active-parameter count, the wider tensor split's lead**, because
what the tensor path pays is a *fixed* cost while compute scales with active params.  So the only place
`-sm tensor` loses today is the MoE + host-resident-expert (op-offload) case — and that is the case worth
fixing, because a working `-sm tensor -ncmoe` is the single configuration that is best at both.

**The cost, measured (ub sweep, `-sm tensor -ncmoe 99`, 2 GPUs, stage 1, one pass each):**

| ub | t/s | per pass | METAGC | share |
|---|---|---|---|---|
| 64 | 43.1 | 1.48 s | **1078 ms** | 73 % |
| 512 | 187.5 | 2.73 s | 1101 ms | 40 % |
| 2048 | 715.5 | 2.86 s | 1126 ms | 39 % |
| 8192 | 2732.9 | 3.00 s | 1219 ms | 41 % |

324 calls (~120/pass, = 3 offloaded ops x 40 layers) at ~3.4 ms each — and the total is **essentially
independent of `ub`**.  At ub 64 the MoE compute is negligible yet the cost is unchanged, so this is **not
device wait**: it is host-side dispatch, a **fixed ~1.1 s per pass**.

**What it is NOT** (each ruled out by measurement):

* **not the meta backend's own bookkeeping** — a per-phase breakdown inside `graph_compute` sums to **17.5 ms**
  (body 0.0, map 0.7, stage 2.1, analysis 11.9, dispatch-marker 2.8) against a 1195 ms total, so ~98.5 % is
  in the child-graph dispatch loop itself;
* **not the split-state cache** — `GGML_META_SSDBG` shows `calc=17722 hit=53263 clears=23` (a 75 % hit rate);
  A/B-ing the whole-cache clear for a single-entry erase changes nothing (2729 -> 2736 t/s, 1212 -> 1201 ms);
* **not fusion or CUDA-graph options** — `GGML_CUDA_DISABLE_FUSION=1`, `GGML_CUDA_GRAPH_OPT=0/1` and
  `GGML_SCHED_STAGE=0` all leave METAGC at ~1085 ms;
* **not the op-offload itself** — with `GGML_OP_OFFLOAD_MIN_BATCH=1000` (MoE back on the CPU) the meta path
  still spends **1024 ms** in 164 calls at 6.2 ms each.  The product (calls x per-call) is ~1.0 s either way,
  so the cost is proportional to the **total node count**, not to how the graph is split.

**Consequence.**  The penalty is a *fixed per-node dispatch cost in the tensor-split path*, paid twice for
mirrored work (hence ~2x a single card).  It is therefore structural and independent of expert splitting:
**halving the experts cannot remove it.**  Two candidate levers, in order of expected value:

1. ~~Cut the number of dispatches.~~  **RULED OUT by the same measurement.**  Batching the offloaded ops into
   fewer splits would cut `calls` but not `nodes`, and the two configurations above show the *product* is
   ~1.0 s either way (324 x 3.3 ms with offload, 164 x 6.2 ms without) — so the cost is **per node**, not per
   split.  Merging the splits would gain nothing.  (Worth knowing before anyone spends a day on the scheduler:
   the 120-graphs-per-pass structure is untidy, but it is not what costs the second.)
2. **Cut the per-call cost** inside the child-graph dispatch — needs `ggml_backend_cuda_graph_compute`
   instrumented (a large TU; not yet done).

Either way the target is now the *dispatch structure*, not the split, and not the ring.

## 14. Fifth probe: the per-node cost is in the CUDA backend, and it is mode-independent (2026-09-27)

Patch: `exp5-cuda-per-node-cost.patch` (`GGML_CUDA_GCDBG`, a host timer + node counter around
`ggml_backend_cuda_graph_compute`).

**Where the §13 dispatch time goes.**  Re-deriving the §13 phase markers (my labels were off by one slot):
the child-graph construction **and** the MMB `graph_optimize(mark_params)` pass together are only **17.5 ms**
— the whole ~1178 ms is the *dispatch loop* itself, i.e. the `ggml_backend_graph_compute_async` calls into
the simple backends.  So the cost is inside the CUDA backend's `graph_compute`, not in the meta backend.

**Measured (ub 8192 unless noted; host time only — `graph_compute` returns after enqueueing, there is no
stream synchronize in it):**

| mode | calls | nodes | host_total | per_node | per_call | t/s |
|---|---|---|---|---|---|---|
| **1 GPU** | 324 | 14824 | **851-861 ms** | 57-58 µs | 2627-2657 µs | 5729-5738 |
| 2 GPU `-sm layer` | 366 | 14824 | 1002 ms | 68 µs | 2738 µs | 4112 |
| 2 GPU `-sm tensor` | 968 | 29648 | 1170 ms | 40 µs | 1209 µs | 2735 |

**The two things that matter here:**

1. **~40-68 µs of host time per node is very high** (a bare kernel launch is a few µs).  That is a real,
   separately valuable inefficiency — and it is paid in **every** mode, including the fastest (1 GPU), where
   ~0.3 s of each 1.43 s pass is this host work.  In fact this is the first candidate found that is *not*
   specific to tensor split, so it is worth its own look regardless of how this campaign ends.
2. **It does not explain the tensor-vs-1-GPU gap.**  Host cost is ~0.85-1.17 s in all three modes (only
   1.37x between 1 GPU and tensor, while the pass time differs by 2.1x), and calls/nodes are *identical*
   across `ub` and identical with the offload disabled (`GGML_OP_OFFLOAD_MIN_BATCH=1000`: still 324 calls /
   14824 nodes) — because the graph's node count does not depend on batch size and the MoE is only ~120 of
   ~14824 node-visits.  So the remaining tensor penalty is GPU-side, or the *same* host cost is **exposed**
   in the tensor path (its per-subgraph, cross-device event waits can stop the host running ahead) while the
   single-device path **hides** it.  Distinguishing those two is the next measurement: it needs a device-side
   attribution, not another host timer.

Caveats to keep: the numbers are per *run* (llama-bench does several graph computations), so ~7 graphs x
~2000 nodes x ~46 splits; and the host cost does not vary with `ub` at all (795 ms at ub 64 vs 861 ms at
ub 8192), which is expected for issue-side work but means it must be compared against *pass* time, not
against a per-token rate.
