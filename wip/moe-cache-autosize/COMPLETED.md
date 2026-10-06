# COMPLETED — the `moe-cache-autosize` campaign (closed work)

This is the ledger of what is **finished**.  The live work — and the only thing a new session needs to
read — is [`README.md`](README.md).  Deep records, for when a closed item needs re-litigating:

| record | what |
|---|---|
| [`ARENA-UB-TENSION.md`](ARENA-UB-TENSION.md) | the #42 design record (§0-§11 history, §12-§14 the r19/r20 work) |
| [`../../archive/docs/TENSOR-CORRUPTION.md`](../../archive/docs/TENSOR-CORRUPTION.md) | the #41/#43 trail (45 KB of bisection detail) |
| [`../../archive/docs/HANDOVER-unredirect.md`](../../archive/docs/HANDOVER-unredirect.md) | the mis-framed "un-redirect" investigation and what it actually turned out to be |

Artifacts that used to sit in this directory (patches, diffs, measurement scripts) are **deleted** —
their content is either in the delivery (`patches/`, so it is recoverable from any release) or
superseded, and each is listed at the end of this file.

---

## 1. The campaign's own goal — DONE (r14, block 13; r15 follow-up; r19/r20)

**Goal (maintainer, 2026-10-05):** if `-ncmoe > 0` and the user did not set `MOE_EXPERT_CACHE_MIB`, the
cache should arm itself and **size itself from the hardware**, so the administrative load leaves the
user.

Shipped:

* **Auto-arm + auto-size.**  `g_auto` is true when `MOE_EXPERT_CACHE_MIB` is unset; the arena is sized
  per device from `free - reserve`.  Design constraints D1-D5 (enable early / decide late, the reserve is
  the whole problem, the floor, multi-GPU + `-sm tensor`, `--fit`/server) were all implemented.
* **Early floor decision (DONE, validated).**  Below `MOE_EXPERT_CACHE_MIN_MIB` the cache **disables
  itself** rather than registering a zero-slot arena — a registered-but-empty arena stands the
  cache-band MoE fusions down and measures ~8× slower than the cache-less path (64 MiB → 4.3 t/s vs
  `MIB=0` 32.0 t/s).
* **Aux-context reserve (DONE).**  The preflight runs before the MTP draft context exists, so its
  `free - reserve` projection over-shot the real arena by the draft's own memory (measured gap 3655 MiB
  @c8192 … 3956 @c131072).  It now subtracts a per-device `aux_reserve_bytes` (4096 MiB when a draft
  context is configured, else 0; `MOE_EXPERT_CACHE_AUX_RESERVE_MIB` overrides).  Projected vs actual
  **19666 vs 20107 MiB** — conservative, the safe direction.
* **Reserve grid (DONE).**  `reserve-grid` swept ctx × ub × MTP × draft-offload on a single R9700
  (IQ3_XXS, auto): no OOM, the arena yields as designed.  The grid script itself was build-specific and
  is deleted; the results are in the session record.
* **Promotion gates were green on the WIP build**, then promoted as **block 13 in `v16-a55e952b8-r14`**.
  (This is tracker item **#37**, "the 34 → 52 t/s hole": one R9700 / gfx1201, IQ3_XXS + MTP measured
  **34.2 t/s unset vs 52.1 at `MIB=20480`**.)

### 1b. TODO #40 — `-sm tensor` host experts ran on the CPU, and the split-table device policy — DONE (r15)

**Opened 2026-10-06 (r15 session); promoted into blocks 06 + 13 in `v16-a55e952b8-r15`.**  Under
`-sm tensor` + `-ncmoe` **every host-resident expert op ran on the CPU**: r14's own per-device host bufts
made the Meta device's `get_host_buffer_type` return null (its simple devices' host bufts differ), so the
loader's `-ncmoe` override fell back to the pageable `CPU_REPACK` buffer (`.is_host == nullptr`) and the
scheduler's op-offload device pin then skipped the Meta backend.  With that fixed the cache engaged, but
looked slow with MTP — the real cause was the **device-side admission policy and its prefill seed**, tuned
for a whole, per-device expert and costing ~10 t/s on a Meta-split slice (plus a ~2-5 s one-time startup
that dominated a short `-n 128` run).

Fix: **block 06** prefers a real device's pinned host buft when the layer device has none
(`LLAMA_TENSOR_HOST_BUFT=0` restores `CPU_REPACK`) plus `meta_dev_contains` in the offload loop;
**block 13** arms the device policy only when `t.split_axis < 0`
(`MOE_EXPERT_CACHE_DEVPOLICY_SPLIT=1` restores it).  Measured, 2 GPU IQ4_NL `-sm tensor -ncmoe 48` MTP n3
`-n 3000`: **30.3 → 88.0 t/s** (vs `-sm layer` 76.2); 3 GPU `-sm tensor` 99.1 (vs 84.8); byte-identical.

This is what closes the "the cache is inert under `-sm tensor`" gap noted above.

**The one gap that followed it ("under `-sm tensor` the cache is inert") is CLOSED.**  The claim was
that `-sm tensor`'s host experts land in the pageable `CPU_REPACK` buffer, so registration/preflight
never fired (measured h=0, arena 0, 30.6 t/s vs `-sm layer` 57.1).  At r20 the cache is armed and
working under `-sm tensor`: the validated server run reports an arena of **25.9-34.9 GiB** (it depends on
free VRAM at sizing time) with a per-turn hit rate of **0.965-0.976**.  The fix came via the
pinned-host-buffer work, not via a loader change here.

---

## 2. TODO #41 — the `-sm tensor` staging redirect — DONE (r16, blocks 16+17; folded into block 15 in r17)

**Symptom:** a wide prefill (`-ub >= ~3000`) under `-sm tensor` made the **target** emit `////` and MTP
accept **0/3063** drafts (the acceptance was a *symptom*).

**Root cause:** the `~3600` was never a width limit — it is where `sched_stage_min_tokens()` turns the
op-offload H2D staging ring **on** (`-sm tensor`'s meta split shards the table, so the size-scaled
threshold drops below the prompt; `-sm layer`'s whole-table threshold stays above, which is why it
looked immune).  The staged bytes were **correct**; the consume path repointed the device tensor at the
ring slot (`simple_tensor->data = chunk.slot`) and the stage guard then restored the pointer
*immediately after enqueuing the child graphs* — but the kernels read `tensor->data` at **execution**
time, so they always saw the restored, stale pointer.

**Proof and fix:** the intersection of the 24 slot pointers handed to tensors and the 360 `src0->data`
values seen at `MUL_MAT_ID` dispatch was **0**.  Copying the slot into the real buffer instead gives
coherent output and MTP acc 3.88, and costs nothing: `+12 %` prefill (**999 vs 890 t/s**, 16k / `-ub
8192`).  The corrupt redirect's 1557-1572 t/s was reading stale memory, not a speedup.

**In the delivery** (`ggml-backend-meta.cpp`): the consume path calls `stage_d2d` into the real buffer;
`GGML_STAGE_META_REDIRECT=1` restores the (incorrect) redirect for A/B only.

**Sub-item also closed:** the generic ring's `stage_mode` used to default to `1` (the redirect) with the
same restore-timing defect.  The default is now **`0`** — stage, then D2D into the split input
(`ggml-backend.cpp`: `sched->stage_mode = mode_env != NULL ? atoi(mode_env) : 0;`).
`GGML_SCHED_STAGE_MODE=1` re-enables the redirect for A/B.

---

## 3. TODO #43 — `-sm tensor` + 3 GPUs + host experts, IQ4_XS only — DONE (r16; folded into block 15 in r17)

**Symptom:** deterministic `////` + acc 1.00 for UD-IQ4_XS under `-sm tensor` on 3 GPUs, at every `-ub`.
It was **not** the device count (dense is fine on 3), **not** the cache, **not** the staging subsystem
(it survived `GGML_SCHED_STAGE=0`), **not** the ubatch width, and **not** allocator luck.  It needed MoE
**host-resident experts** (`-ncmoe > 0`) **and** a wide-over-read quant (IQ4_XS, not IQ3_XXS).

| model | `-sm` | GPUs | result (3.7k prompt, greedy) |
|---|---|---|---|
| UD-IQ4_XS | tensor | 3 | `////`, acc 1.00 at every `-ub` |
| UD-IQ4_XS | layer | 3 | coherent, acc 3.54 |
| UD-IQ4_XS | tensor | 2 | coherent, acc 3.54 |
| UD-IQ3_XXS | tensor | 3 | coherent (acc 3.36-3.62) |
| Qwen3.5-4B Q8_0 dense | tensor | 3 | coherent, 3328 t/s prefill |

**Disproven along the way:** the slice geometry was proven correct (per-device slices sum to the full
dim, 128-aligned, never zero — the 3-GPU IQ3_XXS run has byte-identical blk.2 geometry and is coherent),
the guard-pad hypothesis was disproven (a 512 B tail on every quantized expert tensor's own allocation
still emitted `////`), and it was **not** the MMQ kernel (`GGML_CUDA_FORCE_CUBLAS=1` still emitted it).

**Fix that was promoted:** the per-device guard prefix in the tensor-split upload (the guard had been
distributed as a contiguous prefix, which only reaches device 0).  The reader-side clamp
(`mmid-reader-clamp.diff`) and the one-time finite fill were explored and **not** used.

---

## 4. Stage 1 (prefill) — items 1-3 — DONE (r17 / r18)

* **Item 1 — the gather guard** re-armed per pass (free); the device gather is *not* the prefill lever.
* **Item 2 — pinned 2-D H2D for the `-sm tensor` split staging slice: the big prefill win.**
  `-ub 8192` split prefill **1775 t/s (from 1172)** at 16k and **1962 (from 1311)** at 32k, byte-identical
  and coherent, and *above* both `-sm layer` (1409) and mirrored (1405).  Promoted in
  **`v16-a55e952b8-r17`**.
* **Item 3 — the r7 table-size scaling on the staging width gate is stale: disabled.**
  `SCHED_STAGE_TABLE_REF_BYTES = 0` (width-only gate).  Re-validated on 2 GPU / cache off / 16k: staging
  beats the serial path at **every** `-ub` 1024-8192 for both a 450 MiB and an 850 MiB table —
  IQ4_XS `-ub 2048` **480 → 687**, `-ub 4096` **729 → 1161**, `-ub 8192` 1541 → 1663 (32k 1911);
  IQ3_XXS `-ub 4096` **757 → 978**.  Byte-identical to the old default; no change for tables ≤ 144 MiB.
  Promoted in **`v16-a55e952b8-r18`**.  `GGML_SCHED_STAGE_TABLE_REF_MB` restores the scaling for A/B.

Stage-1's design decision that still holds: **staging mode = `stage_d2d`** (redirect off) — no VRAM
difference vs the redirect, and the redirect is corruption-prone (§2).

---

## 5. Stage 2 (#42's crash half) — DONE (r19, r20)

The 2-GPU cache-auto `-ub 8192` **OOM is fixed**, and so is the server variant of it.

* **The failure was a grow-in-place realloc, not the arena.**  `sched_reserve` sizes the compute buffer
  from a *measure* graph (6636 MiB), but the runtime graph needs **6780 MiB** — a ~216 MiB
  peak-tensor-set difference.  `ggml_gallocr_reserve_n_impl` frees the old buffer and allocates a larger
  one, so it needs a *contiguous* block **bigger than the one it just released**; free VRAM elsewhere
  (including a smaller arena) does **not** substitute.  This is why an arena headroom (swept
  512/1024/2048/4096 MiB) changed nothing, and why the failure was fragmentation-sensitive.
* **r19 — fail-soft arena release.**  A failed compute `cudaMalloc` frees the whole arena, warns, and
  retries; the run survives with the cache disabled instead of aborting.  Plus an **arena slot-count
  retry** (try the requested count → the exact max from `cudaMemGetInfo` → a 0.95 geometric descent) and
  a **per-turn arena hit rate** in the server log next to the MTP acceptance line.
* **r20 — the actual fix.**  Three defects, found by getting a real backtrace first:
  1. **A 10 % compute-buffer slack** (`GGML_COMPUTE_BUFFER_MARGIN_PCT`, HIP-only, opt-in per buffer type
     via `get_compute_margin_pct`).  Padding the *allocation* (not the layout) means the per-chunk
     realloc trigger does not fire for a growth inside the slack.  Measured: the cli repro is **3/3
     aborts at `pct=0` → 3/3 clean at `pct>=8`**; cost **1275 MiB** of arena residency
     (63.8 % → 61.5 %, ~127 MiB per point) at the default 10.
  2. **The "falls back wholesale" invariant is enforced.**  `moe_cache_has_arena()` documents it and the
     fusion guard acts on it, but the *scheduler* hook `moe_cache_take_over` did not agree — it still
     took the input over, so the scheduler skipped populating `input_cpy` and the fallback op read a
     tensor that was never filled, whose stale `data` pointed into the freed arena.  That was the GPU VM
     fault (ROCr turns one into `abort()`, which is why the earlier session saw a "silent death").
  3. **No arena is freed under an in-flight kernel** (`moe_cache_sync_devices_locked()`), because the
     fused MoE kernels carry the arena address in their *launch parameters*.
  Plus the arena yield moved to the single allocation choke point (`ggml_cuda_device_malloc`, with the
  Q8_1 cache arena routed through it), which is what makes the workspace pool and the Q8_1 arena survive.

**Evidence (r20):** cli repro 3/3 clean, and the kill-switch `=0` still reproduces the abort (so the
default is on *and* effective); server concurrent long+short prefills **7/7 `A=0 B=0 alive`** with 22
*partial* stand-downs, **0** full releases, 0 faults/OOMs, 0 reallocations; hit rate 0.9651 → 0.9741;
dense 3-GPU coherence gate clean.

---

## 6. The "un-redirect" investigation — DONE (resolved by §5)

`HANDOVER-unredirect.md` hypothesised that a persistent graph tensor held an arena pointer that needed
restoring.  **That framing was wrong:** the redirects (`moe_cache_redirect_fused`) write into *stack-local*
tensor copies (`ggml_tensor src0_c, ids_c, gate_c;` at the call sites) immediately before launch, so no
host pointer dangles.  The real defects were §5's (1)-(3).  The durable lesson is the handover's own
first instruction — **get a real backtrace before theorising**: `coredumpctl`/gdb gave the SIGABRT and
the faulting kernel in one run, after a session of reasoning from a false "silent segfault" premise.

---

## 7. Superseded artifacts (deleted from this directory)

| file | why it is gone |
|---|---|
| `auto-mode.patch` | shipped as the auto-size campaign (r14, block 13) |
| `stage1-item1-gather-guard.patch` | shipped (r17) |
| `stage1-item2-pinned-2d-h2d.patch` | shipped (r17) |
| `stage1-item3-width-gate.patch` | shipped (r18) |
| `stage-redirect-fix.patch` | shipped — in `ggml-backend-meta.cpp` (§2) |
| `stage2-arena-shrink.patch` | superseded — r19/r20 moved the yield to one choke point and added the margin; the stand-down now lives in block 15 |
| `arena-ub-tension.diff` | superseded by §5 |
| `source-guard-fix.patch` | shipped with #43 (§3) |
| `mmid-geometry-diag.diff`, `mmid-reader-instrumentation.diff`, `mmid-reader-clamp.diff` | #43 diagnostics and an *explored* clamp that was not used (§3) |
| `m0-run.sh`, `m0-r13.log`, `reserve-grid.sh`, `reserve-grid.log` | build-specific measurement tooling for finished experiments (they hard-code `~/llama-r13`); results are in the session records |

**Moved (not deleted):**

| was | now |
|---|---|
| `wip/moe-cache-autosize/TENSOR-CORRUPTION.md` | [`../../archive/docs/TENSOR-CORRUPTION.md`](../../archive/docs/TENSOR-CORRUPTION.md) (closed; §2/§3 above) |
| `wip/moe-cache-autosize/HANDOVER-unredirect.md` | [`../../archive/docs/HANDOVER-unredirect.md`](../../archive/docs/HANDOVER-unredirect.md) (resolved; §6 above) |

Dated delivery records (`WORKLOG.md` release entries, the `patches/README.md` release header blocks,
`MANIFESTS.md`, the delivery `README.md` patch table) still name the old paths — they are dated statements
and were deliberately left alone.  This table is the resolver.

If one of these is needed again, recover it from git history (`git log --diff-filter=D -- <path>`) or
from the delivery patches.
