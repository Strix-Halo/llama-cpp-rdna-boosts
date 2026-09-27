# Tensor-split expert parallelism: split the MoE expert weights instead of mirroring them

**Status (2026-09-27, later session):** **PROMOTED — the whole fast path is delivery release
`v16-84e76d8a2-r16` (folded into block 15); the win is announced in GitHub Discussions #54.**  The
§29 blocker was NOT the staged path at all: `stage_input` was silently returning `false` because its
device-slice sum was checked against the whole-tensor `size` instead of `chunk_size_full` (§30).  With that one-line
fix the staged split reaches **1652 t/s @ ub 2048 / 5454 @ ub 8192** (was 288), bit-identical to mirrored;
an opt-in-to-default pinned splice gather for the *pruning* path then fixes small ub too, so the default
`-sm tensor -ncmoe` with the split beats mirrored at **every** ub measured (512..8192, +5..+20 %).  The
source-pinning fix remains delivered as release **`v16-84e76d8a2-r15`** (block 06, +83 % pp8192).
**Default-on:** the fast path is now the *no-env-var* path (§30.7) — staging, split copies, the pinned
splice gather and the pinned expert source all self-select from `-sm`/`-ncmoe`, each with a kill-switch;
the real default at pp8192 is **5446** vs **3271** for the old opt-in behaviour.
Read §0 first, then §1-7 (the original plan), then §§21-29 (findings), then **§30 (the fix + results +
default-on)** and **§31 (the next phase: partial VRAM expert residency for prefill)**.

**Not a blocker for anything.**  This is an optimisation campaign; nothing in the delivery depends on it.

## 0. HANDOVER BRIEF — read this first (cold start)

### Goal

Make **`-sm tensor` + `-ncmoe`** (host-resident MoE experts) the best configuration for *both* prefill and
decode.  It is the only configuration that can be: with the experts **on** the device, `-sm tensor` already
beats `-sm layer` at prefill (7732 vs 6422 t/s, ub 8192) **and** decode (`tg64 @ d16384` 76.87 vs 74.83),
and the maintainer's rule holds — the greater the active-parameter count, the wider tensor's lead, because
the tensor path's overhead is *fixed* while compute scales with the active params.  A MoE is therefore the
worst case for it, and the only place it loses today.

The maintainer's framing, which the measurements support: **the expert split is the core lever.**  It
divides *both* the per-card upload volume and the per-card compute, and the mirroring is why every effect
in §§8-§16 was full-size on every card.

### How to read this file

§8 onwards is a chronological lab notebook (dated, evidence-first, including mistakes).  **Where sections
disagree, the later one wins**: §16b supersedes §16, §16 supersedes §11, §17 supersedes §12's guess, §19
supersedes §17/§18's *location*, §22 root-causes the fault, §23 finds the pinning lever, §25 settles the
queueing question, §27/§28 compare the external prior art, **§29 is the current open work**.  §0 is the
distilled state; trust it over any older section (except §29, which is newer than this brief's prose in
places — when they disagree, §29 wins).

> **`GGML_META_SPLIT_COPY` defaults to 1** (only `0` is the r12 mirrored behaviour).  Every "mirrored"
> measurement MUST pass `GGML_META_SPLIT_COPY=0`; without it you are measuring the split.  See §25.1.

### What the 2026-09-27 (second) session did

1. **Promoted the pinning fix** to the delivery as release **`v16-84e76d8a2-r15`** (block 06) — validated
   (strict 16/16 apply, build clean, coherence bit-identical, +83 % prefill) and documented (WORKLOG,
   `patches/README.md`, README/AGENTS/BASELINE/MANIFESTS headers, `upstream/UPSTREAM-PR-moe-host-expert-pinning.*`).
2. **Settled the queueing question (§25):** whole-tensor staging is worth **+34 %** to the mirrored path;
   the split cannot stage at all (meta has no events → full host sync per upload); the split's exposed cost
   is the H2D; its compute floor is **7037** vs mirrored **5107**.
3. **Compared the external prior art (§26-§28):** Strata (single-card, CPU-computes-misses) and
   GenerelSchwerz `moe-cache` (layer-split VRAM cache) both independently do the pinning and neither does
   `-sm tensor`; the vLLM setups (dual-r9700, radiance) are all **VRAM-resident**; **R9V is the one that
   tensor-splits offloaded experts** — hot manifest in VRAM + cold shards in pinned UVA host memory.
4. **Implemented the staged split upload and hit a bug (§29):** it lands at **~290 t/s** (the plain-2D
   number) instead of the **~1497** the pruning + `PINHOST` path gets.  Fix that first.

### Status in one screen (end of the 2026-09-27 session)

* **SHIPPED — the pinning fix is the delivery's win (r15).**  `select_weight_buft`'s "avoid using a host
  buffer when using mmap" downgrade is now skipped for `MUL_MAT_ID` weights, so `-ncmoe` op-offload uploads
  read pinned memory.  `-sm tensor -ncmoe 99` pp8192 **2794 -> 5104 t/s (+83 %)** on 2x R9700,
  bit-identical; delivery release **`v16-84e76d8a2-r15`** (block 06), `validate-set.sh` green.  Kill-switch
  `LLAMA_MMAP_HOST_EXPERTS=0`.  §23 has the finding, WORKLOG / `patches/README.md` the promotion, §28 the
  independent corroboration.
* **The split is implemented and numerically correct, and it is the only config that beats mirroring once
  the upload is hidden.**  `GGML_META_SPLIT_COPY=1` splits the expert uploads (gate/up axis 1, down axis 0;
  `ffn_moe_down` becomes `PARTIAL`).  Its **compute floor is 7037 t/s vs mirrored's 5107** — halving each
  device's work is worth ~**+38 %** *if the transfer is hidden* (§25.4).  The campaign is now entirely
  about hiding that upload.
* **The blocker is transfer EXPOSURE, quantified (§25).**  The meta backend has no
  `event_record`/`event_wait`, so the scheduler falls back to `ggml_backend_synchronize(meta)` — a full
  host sync of **both** devices — before every expert upload; and `stage_input` rejects a split `input_cpy`,
  so the split never reaches the copy stream.  Hence mirrored (2x the bytes, hidden) beats the split (half
  the bytes, exposed).  A `GGML_META_NOSYNC`+copy-stream hack reaches 4927 (parity, unsafe).
* **THE OPEN WORK (§29): the staged split upload is implemented but lands at ~290 t/s**, exactly the
  plain-2D number, instead of the ~1497 the pruning + `PINHOST` path gets — and it is structural to the
  staged path, not the transfer shape (all `GGML_META_GATHER_MODE` values equal; the D2D compaction probes
  at 0.47 ms).  Three checks listed in §29.
* **The prior art says the winning move may be to stop copying (§28, R9V):** TP-sharded experts (our
  `-sm tensor`) + a measured hot-expert set resident in a VRAM slot cache + **cold shards in pinned UVA
  host memory the GPU reads in place** (no per-ubatch H2D at all).

### RESOLVED this session — see §30

The §29 blocker was not the staged machinery: **`ggml_backend_meta_stage_input` compared its accumulated
per-device chunk-size sum against the whole-tensor `size` instead of `chunk_size_full`, so it always
returned `false` for a split tensor and the run silently fell back to the slow per-device 2-D splice.**
One-line fix (`offset_j != chunk_size_full`); the staged split then reaches **1652 t/s @ ub 2048** (was
288) and **5454 @ ub 8192**, bit-identical output.  Making the compact splice's pinned gather the default
(`GGML_CUDA_SPLICE_GATHER`, opt-out) fixes the sub-gate ubs too, so the **default** split now beats
mirrored at every ub tested (512..8192, +5..+20 %).  §30 has the table and the verification; **§31 is the
next phase** the maintainer asked for: a partially VRAM-resident expert set for prefill (R9V-style), with
the decode slot-cache left to the phase after.

The build tree carries all the instrumentation (`exp15` + the §30 fix + the splice-gather default; the
working diff is exported as `exp16-staged-split-fixed.patch`); the environment and repro are just below.

### THE FAULT (historical — root-caused in §22; the r15 pinning fix makes the pageable case moot).  Kept for the record.

The short version: the fault is the splice's pageable `hipMemcpy2DAsync`, 2 GPUs only, and it is **not** a
race between the upload and the consumer — 1-D copies on the same addresses are clean.  Everything below is
the lead-up, kept for the record.

**Repro (fails in ~30 s at this size):**

```bash
cd ~/llama-r12
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
HIP_VISIBLE_DEVICES=0,1 GGML_META_SPLIT_COPY=1 GGML_SCHED_STAGE=1 \
  ./build-rocm/bin/llama-bench -m "$M" -ncmoe 99 -fa 1 -p 128 -ub 128 -n 1 -b 128 -sm tensor -r 1
# -> "Memory access fault by GPU node-2 ... Page not present or supervisor privilege" (rc=141)
# with GGML_META_SPLIT_COPY=0 the same command completes: rc=0, pp128 = 57.93 t/s
#   (re-measured 2026-09-27; the number is small because ub 128 is tiny).
```

**Symptom.**  A device memory-access fault, at a *varying* point in the first prefill (layer 0's gate in
some runs, layer 1 in others; the fault address differs every run) — it is asynchronous, so where it is
*reported* is not where it *happens*.

**Where it is (§19, bisected by forcing a sync after every launch, `GGML_META_SYNCEACH=1`).**  With the
sync, `SPLIT_COPY=0` completes and `SPLIT_COPY=1` prints both `ffn_moe_gate-1` syncs and then dies — so the
guilty launch is the **next `graph_compute`: the 2-node `ffn_moe_up-0 + ffn_moe_swiglu-0` subgraph.**  Its
dump shows the interesting thing:

```
ffn_moe_up-0 (MUL_MAT_ID)     data=0x…83e62480  ne=[256,8,128]
ffn_moe_swiglu-0 (GLU)        data=0x…7ac62480  ne=[256,8,128]
  src0 ffn_moe_gate-0         data=0x…83c62480   <- computed in the PREVIOUS graph_compute
  src1 ffn_moe_up-0           data=0x…83e62480   <- computed in this one
```

**So it dies on the first consumer of a split MoE intermediate that crosses a `graph_compute` boundary.**
Under `-ncmoe` the scheduler's op-offload hands the meta backend **one op per `graph_compute`**, and the
meta compute container is **double-buffered and cleared per `graph_compute`** — the gate's output from
generation A must survive the builds of B, C, …  Every pointer in the dump is sane and symmetric with the
working mirrored run, which is why this reads as **lifetime/generation, not arithmetic** (§12's
compute-container hypothesis, now with a named victim).

**THE NEXT EXPERIMENT (VOID — kept for the record).**  §0's original suggestion was to force the gate and
up into the **same** `graph_compute` with the delivery's fused gate+up+GLU arm (`GGML_CUDA_MMB_GLU=1`).
**That experiment cannot work by construction (measured 2026-09-27):** with `-ncmoe` the scheduler's
offload always dispatches `ffn_moe_gate-N` as its **own** `graph_compute` and `ffn_moe_up-N +
ffn_moe_swiglu-N` as the next one (`GGML_META_EXECDBG`), so a CUDA-backend-level gate+up+GLU fusion can
never see the three nodes together.  `GGML_CUDA_MMB_GLU=1` therefore changes nothing and the split run
still faults.  See §21.

**Already ruled out for this fault (all measured — do not re-derive):** the MMQ `MUL_MAT_ID` kernel args
(every field derives from the simple tensor; `s02 = nb02/ts` is 2048 blocks split vs 4096 mirrored — §17);
buffer bounds (`GGML_META_BUFDBG` "ok" for every expert copy); the MMB layer and the delivery's
routed-compact kernel (`GGML_CUDA_MMB=0` / `MMB_ROUTED=0` / `MMB_GLU=0` / `GGML_CUDA_DISABLE_MMQ_ROUTED=1`
all still fault); the partial-reduce machinery (drained — §21); garbage expert ids (`min=0 max=255
out_of_range=0` when readable); the ids view's parent pointer (§18/§19, and §21's `VIEWDBG` again); the
simple-tensor **container lifetime** (§21: `GGML_META_CNDBG` shows **zero** lookups that miss the current
double-buffered container); the **stream** (§21: `GGML_STREAMDBG` shows the upload and the consumer use the
*same* per-device stream); and the **H2D staging ring** (§21: `GGML_SCHED_STAGE=0` still faults).
`-sm layer` "working" proves nothing — `GGML_META_SPLITDBG` shows **zero** split expert copies there
(layer split never consults the policy).

### SECOND LEVER (§20, tracked, complementary): the upload stalls

Measured: the mirrored expert upload is **932 MiB per device per run** (~466 MiB per prefill pass), and the
transfer of that at the link's 14.4 GB/s is ~65 ms — but `hipMemcpyAsync` **from the pageable mmap blocks
the host for the whole call** (2724 calls, 6107 ms total, 483 of them >10 ms, mean 2.2 ms; a standalone
144 MiB probe: **10.461 ms blocked vs 0.001 ms pinned**).  Because the host is *inside* the call it can
only issue one device's copy at a time, so the two devices' DMAs cannot overlap.  Prize: `-ncmoe 0` (no
uploads) is **7945 t/s** vs **713** for `-ncmoe 99` at ub 2048 — 11× — so it is the **stalls**, not the
bytes.  Two parts, both the maintainer's suggestion to "queue the results while continuing to process":
**(1) pin the source** (`cudaHostRegister` / `ggml_backend_cuda_host_buffer_type`) so the copies are truly
asynchronous; **(2) overlap** layer N+1's uploads with layer N's compute (what the §11 staging ring was
built for — it hides the *wait*, the transfer is still on the critical path).  The split halves the
per-card volume; pinning removes the serialisation.

### Other things NOT to re-litigate (measured, each with its section)

* **The ring is not the problem and is already per-card** (§11) — one logical slot index, but each simple
  backend owns its own slot arena with per-device events.
* **The PCIe lanes are independent per card** (14.45 GB/s per device, not a shared x4) — §11.
* **The penalty is not host dispatch cost** (§13/§14/§15, corrected by §16/§16b): the CUDA backend's
  "host time" and the 5.25 ms/call cuBLAS figure are the host waiting on a stream; a standalone probe times
  `hipblasSgemm`/`rocblas_sgemm` at **2 µs**.  `GGML_CUDA_GCDBG` host time must never be read as work.
* **The devices are idle during the prefill, not busy** (§16): kernel-busy is ~5-7% (461 ms of kernels in
  an 11.6 s run, median kernel 11 µs).  `rocm-smi`'s `gpu_busy_percent` counts the copy engines — do not
  use it as a compute signal.  **Per-device sampling is mandatory**, and note `HIP_VISIBLE_DEVICES=0` is
  **physical GPU[1]** on this box (a sampler watching GPU[0] reads zeros and will mislead you).
* **The used-expert pruning works** (§8, §16b) — the uploads are pruned ranges, not full tensors
  ("make the pruning prune" is **retracted**).
* **The trailing `min(expert_size,512)` MMQ pad is deliberate upstream code** and the split path places it
  correctly per device (verified; a first draft claimed otherwise — retracted, §17).

### Environment

* **Build tree: `~/llama-r12`**, branch **`wip-tensor-split`**, at the r12 tip `de71ddd58`, with the
  experiment instrumentation applied in the working tree (**`exp15-staged-split-upload.patch` == the
  current diff**).  The delivery's promoted pinning fix is release `v16-84e76d8a2-r15` (block 06, tree
  `d609d34d1`), built and validated separately in `/tmp/deliver` (a worktree of `~/llama.cpp`).
  **Do not use `~/llama.cpp`** (its `rdna-boosts` branch is a different 16-block chain) or `/tmp/canon-fix`
  (the r11 chain; the `/tmp/deliver` and `/tmp/upcand` worktrees were scratch and are removable).
* **Full build:** `cd ~/llama-r12 && CCACHE=0 BUILD_DIR=build-rocm ~/bin/build-llama-rocm-714` (~6 min,
  16 cores).  `CCACHE=0` because the HIP build emits no `.d` files and ccache's direct mode has silently
  reused stale objects after a header edit.
* **Fast loop (use this for instrument edits):** `cmake --build build-rocm --target llama-bench -j 16` —
  the meta/scheduler TUs (`ggml-backend-meta.cpp`, `ggml-backend.cpp`) rebuild in well under a minute and
  even `ggml-cuda.cu` is only a few minutes.  No need for the ~6 min full script unless CMake changes.
* Runtime libs come from the binary's RUNPATH (`/opt/rocm-7.14.1-gfx102X/lib`); `/opt/rocm-7.14-gfx1201`
  is **not** the tree this build uses.
* Model: `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21.10 GiB, gfx1201 ×2 on
  `soar`).  Width/MTP probes: `~/Qwen3.5-4B-Q8_0.gguf`, `/llm/models/Qwen3.8/27B/Q8_0/…`.

### Instruments already built into `~/llama-r12`'s binary (all env-gated)

| env var | what it does |
|---|---|
| `GGML_META_SPLIT_COPY` | `0` = r12 behaviour (mirrored, works), `1` = all weight copies split (faults), `2` = axis-1 only (aborts on a missing rule, §10) |
| `GGML_META_SYNCEACH` | **the fault bisector**: sync + print `SYNCEACH sub/dev/nodes/first/last` after every launch |
| `GGML_META_EXECDBG` | per-subgraph node + pointer + geometry dump (MUL_MAT / MUL_MAT_ID / GLU) |
| `GGML_META_REDDBG` | reduce-path accounting (`max_tmp_size`, PARTIAL nodes, `set_tmp_data`, and which copies get split) |
| `GGML_META_VIEWDBG` | the view branch of `init_tensor_impl` (parent identity, buffer/buft, both pointers) |
| `GGML_META_READIDS` | read the `MUL_MAT_ID` ids back in process (with `READEIDS begin/end` brackets) |
| `GGML_META_BUFDBG` | simple-tensor offset vs device buffer size (`ok`/`OVERFLOW`) |
| `GGML_META_SPLITDBG` / `GGML_META_GCDBG` / `GGML_META_SSDBG` / `GGML_META_SS_FIX` | split geometry / meta per-phase host time / split-state cache |
| `GGML_SET_BYTES` / `GGML_SET_NAMES` / `GGML_RING_STATS` | H2D volume per device / ring wait-gap stats |
| `GGML_SCHED_SYNCDBG` | scheduler `synchronize` / `event_synchronize` / `set_async` / input-loop accounting |
| `GGML_CUDA_GCDBG` | CUDA backend host time per call/node/op/branch (back-pressure caveat above) |
| `GGML_CUDA_DISABLE_MMQ_ROUTED`, `GGML_CUDA_MMB*` | A/B the routed-compact and MMB paths |
| `LLAMA_MMAP_HOST_EXPERTS` | the §23 pinning fix: default on (keep `MUL_MAT_ID` host weights pinned); `=0` restores the mmap downgrade |
| `GGML_SCHED_BUFTDBG` | print each op-offloaded weight's buffer type (`ROCm_Host` vs `CPU_Mapped`) |
| `GGML_META_NO_2D` | the §22 A/B: replace the splice's 2-D copy with 1-D copies (slow, but clean) |
| `GGML_META_PINHOST` / `GGML_META_PINRING` / `GGML_META_D2DSPLICE` | §23/§25 diagnostics: host gather into a per-device pinned ring + 1-D H2D.  `PINHOST=1` same stream, `=2` queued on the copy stream (§25.6), `=4` gather-only floor, `=5` pure-compute floor; `PINRING` = ring depth; `D2DSPLICE` = whole-range 1-D H2D + D2D compact |
| `GGML_META_NOSYNC` | §25.3: skip the scheduler's pre-upload `ggml_backend_synchronize(meta)` (**unsafe**, WAR guard) — measures the queueing ceiling |
| `GGML_META_GATHER_MODE` | §29: force the staged split-upload shape — `0` auto (host gather for coarse, whole-range+D2D for fine), `1` host gather only, `2` whole-range+D2D only |
| `GGML_META_UPLOADDBG`, `GGML_META_TOUCHSRC`, `GGML_META_PINSRC`, `GGML_META_TAILDBG` | the §22/§23 upload-geometry / cold-page / `hipHostRegister` / tail diagnostics |

External tools used: `rocprofv3 --kernel-trace/--memory-copy-trace/--hip-trace -f csv -o DIR -d DIR -- cmd`
(all three were decisive in §16).

### Debugging rules learned the hard way (do not repeat)

* **Never** run this path with `HIP_LAUNCH_BLOCKING=1` **or** `AMD_SERIALIZE_KERNEL=3` — both deadlock it
  (one GPU pegged, the other idle); the meta backend's cross-device `hipStreamWaitEvent` progress needs
  asynchronous launches.  A *targeted* `ggml_backend_synchronize` (`GGML_META_SYNCEACH`) is safe and is the
  right bisect tool.
* A device fault is **asynchronous**: "it faulted here" always means "it surfaced here".  Bisect with
  `GGML_META_SYNCEACH` before believing any location.
* A crashed tool loses block-buffered stdout — always `stdbuf -o0 -e0`, and `-v` is needed for
  `GGML_LOG_*` in the tools (`llama-bench` installs a null log callback otherwise).
* Clean up with `pgrep -x <binary>` / `pkill -x`, **never** `pkill -f "<name>"` (it matches the shell).
* `GGML_META_DEBUG=1` (split-state dump) needs `-v` on the tool.
* Measure volume and time **in the same run** before concluding "X is slow" (§16's lesson).

### Artifacts in this directory

* **`0000-source-pinning-llama-mmap-host-experts.patch`** — **the delivery win**, now **promoted as release
  `v16-84e76d8a2-r15`** (block 06; §24.1).  Kept here as the standalone reference.  Kill-switch
  `LLAMA_MMAP_HOST_EXPERTS=0`.
* `exp1..exp15-*.patch` — cumulative working-tree diffs (`exp15` == the current instrumentation; the touched
  files are `ggml-backend-meta.cpp`, `ggml-backend.cpp`, `ggml-backend-impl.h`, `ggml-cuda/ggml-cuda.cu`,
  `ggml-cuda/common.cuh`, `src/llama-model.cpp`, `src/llama-model-loader.cpp`).  Apply with `git apply` in
  `~/llama-r12` if the tree is ever lost.  `exp11` added `GGML_META_CNDBG`, the post-reduce `SYNCEACH
  REDUCE` drain, `GGML_META_SYNCUPLOAD` modes 1/2/3, `GGML_STREAMDBG`, `GGML_META_SPLIT_COPY=3`, and the
  `VIEWDBG`/`BUFDBG`/`SPLITDBG` re-runs; `exp12` adds the splice **tail fix**, `GGML_META_NO_2D` (the
  decisive 1-D-vs-2-D A/B), `GGML_META_UPLOADDBG`, `GGML_META_TOUCHSRC`, `GGML_META_PINSRC`,
  `GGML_META_TAILDBG`; `exp13` adds the **source-pinning loader fix** (`LLAMA_MMAP_HOST_EXPERTS`),
  `GGML_META_PINHOST`/`GGML_META_PINRING` (the pinned-ring gather+1D splice), `GGML_META_D2DSPLICE`, and
  `GGML_SCHED_BUFTDBG`; `exp14` adds `GGML_META_PINHOST` modes 2/4/5, the per-device pin rings, and
  `GGML_META_NOSYNC`; **`exp15` adds the staged split upload** (`stage_gather` in the iface + CUDA, the
  `h2d_scratch` device scratch, the meta `stage_input` split branch) and `GGML_META_GATHER_MODE`.

  The clean candidate is `0000-…` (promoted as r15); `exp15` is that change plus every diagnostic.
* `overlapprobe.cpp` — standalone ROCm probe (§25.4) proving an H2D on one stream overlaps compute on
  another on this box (serial 46.3 ms vs two-stream 37.1 ms).
* `d2dcompactprobe.cpp` — standalone ROCm probe (§29) timing the split's device-side D2D compaction
  (`ffn_down`: `176 x 524288`, pitch 352 -> 176): **0.47 ms** (~587 GB/s); gate shape 0.29 ms.
* `tool-blasprobe.cpp`, `h2dprobe2.cpp`, `mmapprobe.cpp`, `bigprobe.cpp`, `asyncprobe.cpp`,
  `h2d2dprobe.cpp`, `h2d2dcold.cpp`, `h2d2dalign.cpp`, `d2d2dprobe.cpp` — the standalone ROCm probes that
  settled §15/§16 and §22 (the last four all PASS, i.e. they do **not** reproduce the §22 fault — see §22.4).
  Compile with
  `hipcc -O2 -I/opt/rocm-7.14.1-gfx102X/include -L…/lib --offload-arch=gfx1201 X.cpp -o X -lamdhip64`.
* §7 has the environment/repro detail; §2 the design; §10 the split implementation; §12/§19 the fault.

## 15. Sixth probe (2026-09-27): THE CORRECTION — the host cost is device back-pressure, not host work

> **Superseded by §16/§16b.** This section correctly kills §13/§14's "per-node dispatch cost" by
> showing the "slow" call is a 2 µs BLAS call — but its own conclusion ("the device is at 100%, so it is
> back-pressure") was then overturned by §16's kernel trace: the devices are **~5-7 % kernel-busy** during
> the prefill and `rocm-smi` counts the *copy engines*. The real mechanism is the blocking pageable
> `hipMemcpyAsync` (§16/§16b). Read §16b next.

Patch: `exp6-dispatch-attribution.patch` (`GGML_CUDA_GCDBG` extended: per-op-type, per-mul_mat-branch,
per-cublas-section, and pool counters).  Tool: `tool-blasprobe.cpp`.  **This section supersedes the
conclusions of §13 and §14.**  The measurements there are correct; the interpretation was wrong, and the
correction was prompted by the maintainer looking at the box: *"GPU1 was at 100%"*.

**1. The chain of attribution (ub 8192, 1 GPU, `-ncmoe 99`, `-sm tensor`).**  §14's ~850 ms "host time in
`ggml_backend_cuda_graph_compute`" decomposes as:

```
MUL_MAT=690.5ms/844(818us)   <- one op type is 81% of it
  of which  cublas=629.6ms/120 calls (5.25ms per call)   <- and that is 92% of MUL_MAT
    of which  the gemm call itself = 631ms/120
    src0 alloc + dequant = 0.0ms/120, src1 = 0.0ms/120, dst_temp = 0.0ms/120, to_fp32 = 0.0ms/120
    pool: 7 mallocs total, 1.1ms, no OOM, no frees
```

So 100% of it is the BLAS call — and the call is **not** expensive: the standalone probe
(`tool-blasprobe.cpp`, `hipcc -lhipblas -lrocblas`) times `hipblasSgemm` *and* `rocblas_sgemm` at the exact
shapes (A[256x256] F32, N = 1/64/8192, op(A)=T) on both a default and a **non-default stream** (llama.cpp
uses one) at **2 us per call**, shape-independent.

**2. The answers to the obvious follow-ups, each measured:**

| question | measurement | answer |
|---|---|---|
| is it the batch size (GPU work)? | ub 64/512/2048/8192 -> gemm total 640.8/643.8/633.2/633.9 ms | **no** — constant total, so not kernel time |
| is it our MMB layer? | `GGML_CUDA_MMB=0`, `MMB_SHADOW=0`, `MMB_CACHE=0` | no — 690/729/696/701 ms, unchanged |
| is it the op-offload path? | `-ncmoe 0` (experts on device) | no — 636.8ms/120, same |
| is it Qwen3.6's SSM/GDN layers only? | the dense 4B Q8_0: **zero** cublas calls | yes — 120 F32-weight GEMMs, 3 per layer x 40 |
| why cuBLAS? | `ggml_cuda_should_use_mmf` rejects dense F32 at `src1_ncols > 16` | by design (upstream rule), not a regression |
| is the pool involved? | pool malloc 7 calls / 1.1 ms total | no |

**3. The correction.**  If a 2 us call measures 5.25 ms of wall time, the host thread is **blocked**, not
working — and the maintainer's observation says on what: **the device is at 100%**.  This is ordinary
producer/consumer back-pressure: when the device's queue is full, whichever host call comes next blocks
until the device drains.  So:

* **`ggml_backend_cuda_graph_compute`'s host time is not a cost — it is a measurement of device
  saturation.**  A saturated device names some innocent call as the "slow" one; that is why the figure
  tracks total node count and why it is the same in every mode, including the *fastest* one.
* **Therefore §13's lever 2 and §14's "attack the 57 us per node" are VOID.**  There is no host-side win
  there: the numbers were the host waiting for a GPU that was already 100% busy.  §13's lever 1 (merge
  the splits) was already ruled out; both levers are now dead ends, and so is the whole "dispatch
  structure" line.  Nothing in the meta backend's dispatch needs fixing.
* **The earlier inference "the tensor path's penalty is structural and cannot be removed by halving the
  experts" is WRONG** — it rested on the penalty being host dispatch cost.

**4. What *is* happening in the two-device modes (per-device `rocm-smi` sampling, 1 s cadence).**  Physical
GPU[1] is HIP device 0 on this box (a sampler watching GPU[0] sees zeros — that is how the correction was
missed the first time; per-device sampling is mandatory).

| config (`-ncmoe 99`, ub 8192) | GPU[1] (HIP 0) | GPU[2] (HIP 1) | t/s | vs 1 GPU |
|---|---|---|---|---|
| **1 GPU** | **99-100%** | — | **5736** | 1.00 |
| 2 GPU `-sm layer` | **98-100%** | 17-44% | 4092 | **0.71** |
| 2 GPU `-sm tensor` | 66-71% | 78-87% | 2736 | **0.48** |

Neither two-device mode beats a single card, and **the single card is the most efficient configuration
there is**.  `-sm layer` runs its bottleneck device at 100% *and still* loses 29% against the same device
running alone — so the loss is not "work not shared", it is **cross-device latency**: every layer boundary
in a split graph is a `hipStreamWaitEvent` hop, and the devices spend their time waiting on each other
(which is exactly what "both devices busy but neither saturated, at 78% and 0.48x the single-card rate"
looks like).  Under `-sm tensor` with mirrored experts both devices do the full MoE *and* stall on the
chain, so it is the worst of both.

**5. What this restores, and what the next step should be.**  The campaign's original premise is intact
and now better supported than before: `-sm tensor` with mirrored host-resident experts cannot win
(two devices doing the same work, stalling on each other), while with the experts **on** the device it
does win (`-ncmoe 0`: tensor 7732 > 1 GPU 6542 > layer 6422) **because each device's work is genuinely
halved**.  So the expert split is still the lever — and the interesting new question is the other side of
the same coin:

* **(A) the split** (the campaign's goal) — halve each device's work.  Blocked by the §12 partial-reduce
  fault, unchanged (re-confirmed here: a run without `GGML_META_SPLIT_COPY=0` still faults).
* **(B) the cross-device stall** — *why* are both devices only ~78% busy?  A single card manages 100% and
  5736 t/s; if the two-device tensor path could run its devices at 100% it would gain ~1.5-2x on **every**
  tensor-split workload (dense as well as MoE), which may be worth more than the split itself, and it is
  the same investigation from the other end.  Prime suspects: the meta backend's per-subgraph event chain
  (a wait per subgraph rather than per dependency), and the all-reduce per layer.

## 16b. CORRECTION to §16 (same session, after measuring volume and time in the same run)

§16 read the "implied size" of the >10 ms copies as the *transfer*, which gave ~148 MiB per copy and led
to the claim that **the pruning is inactive and full 144 MiB expert tensors are uploaded**.  Measuring
volume (`GGML_SET_BYTES`) and the HIP trace **in the same run** shows that is wrong:

```
H2D device=0 bytes=977375072 (932.1 MiB)      <- total for the WHOLE run, per device
H2D device=1 bytes=977375072 (932.1 MiB)
hipMemcpyAsync: n=2724  total=6107.3 ms  (>10 ms: 483 calls, 5779.6 ms)
   of which H2D            n=1634  6083.8 ms
             D2H            n=8        0.1 ms
             no copy row   n=1082     23.4 ms   (allocations etc.)
```

232 MiB per pass per device is ~466 MiB (for the run's ~2 prefill passes) = **~3.9 MiB per expert tensor**,
i.e. ~7 experts of 0.56 MiB - **the pruning IS working**, and it is in line with §8's measured 25-of-256
ranges.  The transfer of 932 MiB at the link's 14.4 GB/s is **65 ms**; the measured time is **6.1 s**.
**So ~99% of the blocking is WAIT, not transfer.**

**The corrected mechanism:** `hipMemcpyAsync` from a pageable source is implemented **synchronously** - the
driver drains the stream (waits for the queued compute) and *then* stages the copy - so each of the 2724
calls blocks the host for ~2.2 ms (483 of them for >10 ms) regardless of how small the copy is.  The
`bigprobe` numbers still explain it: at 144 MiB the *transfer* itself blocks 10.46 ms, and the same call
returns in 0.001 ms when the source is pinned - i.e. **pageable = synchronous, pinned = asynchronous**, and
the host-block in llama.cpp is the stream drain plus a small staging copy.

**Consequences for the two levers:**

* **(B1) pin the source - unchanged and still the recommendation.** The probe is unambiguous (0.001 ms vs
  10.46 ms per call) and it removes both the drain and the serialization, letting the two devices' copies
  overlap with each other and with compute.  **Prize, measured: `-ncmoe 0` (no uploads at all) is 7945 t/s
  at ub 2048 against 713 t/s for `-ncmoe 99` - 11x - so the upload *stalls*, not the upload bytes, are what
  `-ncmoe` costs.**
* **(B2) "make the pruning prune" is RETRACTED.**  The pruning is already working; the volume is not the
  problem.  There is no ~10x of volume on the table here - what is on the table is the *stall*.

Also worth noting from the same trace: `hipFuncGetAttributes` n=180 total=807 ms **max 453 ms** - this
platform produces long host-side waits inside HIP API calls generally, so "host time in an API call" must
never be read as work without checking the volume it moved.

## 16. Seventh probe (2026-09-27): THE ROOT CAUSE — the expert uploads are pageable copies that BLOCK the host

**This section supersedes the mechanism proposed in §11, §13, §14 and §15.**  Tools: `rocprofv3`
(kernel / memory-copy / HIP-API traces), `tool-blasprobe.cpp`, `h2dprobe.cpp`, `h2dprobe2.cpp`,
`mmapprobe.cpp`, `bigprobe.cpp`, `asyncprobe.cpp`.  Instrument: `exp7-scheduler-upload-drains.patch`
(`GGML_SCHED_SYNCDBG`: synchronize / event-synchronize / set_async / get_async / input-loop accounting).

**1. The devices are idle, not busy.**  A kernel trace of the tensor prefill (`-p 2048 -ub 2048 -r 1`)
shows **~5-7% kernel busy per device** (461 ms of kernels across an 11.6 s run; median kernel 11 us,
only 23 kernels over 1 ms).  So §15's read of the maintainer's "GPU1 is at 100%" — device back-pressure —
was wrong too: `rocm-smi`'s `gpu_busy_percent` counts the DMA/copy engines as well.  **The prefill is not
compute-bound; the GPUs are waiting for data.**

**2. The HIP API trace names the blocking call.**  Whole tensor run:

```
hipMemcpyAsync   n=2724   total=6095.4 ms   max=19.4 ms   mean=2.238 ms      <- the entire run
hipFuncGetAttributes  n=180  total=807 ms      (kernel attribute queries, once per kernel)
hipLaunchKernel  n=13472 total=529 ms          (13k launches, 39 us each - fine)
hipStreamSynchronize n=3154 total=105 ms       (the scheduler's drains are cheap)
```

by duration bucket:

| bucket | calls | total | share of time | implied size @14.4 GB/s |
|---|---|---|---|---|
| <50 us | 1538 | 5.9 ms | 0% | - |
| 50-500 us | 328 | 86.8 ms | 1% | ~3 MiB |
| 0.5-2 ms | 372 | 215.4 ms | 4% | ~7 MiB |
| **>10 ms** | **483** | **5772.7 ms** | **95%** | **~148 MiB** |

**483 copies of ~148 MiB.**  `150994944 B` is exactly `blk.0.ffn_gate_exps.weight`'s `nbytes` (§8's own
assert prints it) — so these are **FULL expert tensors, ~120 per pass (3 per layer x 40 layers) =
~17 GiB per pass per device.**  The used-expert pruning is *not* pruning in this configuration.

**3. Why it is 2x on two devices: the copy blocks the host, so the two DMAs cannot overlap.**  Progressive
probes (`h2dprobe2.cpp` -> `mmapprobe.cpp` -> `bigprobe.cpp`) show the rule is a *size* threshold, not
"pageable" in general: 512 KiB pageable copies are async (host-block 7 us), but at the expert size the
driver's staging path saturates and the call becomes **synchronous**:

| 144 MiB `hipMemcpyAsync` source | host-block inside the call |
|---|---|
| malloc (pageable) | **10.461 ms** (= the whole transfer, 14.4 GB/s) |
| **mmap model file (pageable - what `-ncmoe` actually uses)** | **10.459 ms** |
| **pinned (`hipHostMalloc`)** | **0.001 ms** (returns immediately, truly async) |

Because the host thread is *inside* the call for the full 10.5 ms it can issue exactly one device's copy at
a time.  That is the tensor penalty, end to end:

* 1 GPU: 120 copies x 10.5 ms = **1.26 s** of non-overlappable transfer per pass; measured pass **1.39 s**.
* 2 GPU (mirrored): the same 120 copies per device, but serialized on the host -> **2.5 s**; measured **2.87 s**.
* `-ncmoe 0` (experts on device, no uploads): **0.26 s**.  The 3x is the transfer.

It also finally explains §13's unexplained facts: the cost is **per layer, not per token** (hence
ub-independent), and the "1.1 s fixed cost" is 120 x ~10 ms of blocked copies.

**4. What the scheduler accounting rules out** (`GGML_SCHED_SYNCDBG`, ub 2048, per run):

| | 1 GPU | 2 GPU tensor | 2 GPU layer |
|---|---|---|---|
| synchronize | 380 calls, **31 ms** | 1143 calls, **54 ms** | 586 calls, **117 ms** |
| event_synchronize | **never called** | **never called** | **never called** |
| `ggml_backend_tensor_set_async` | **0 calls** | **0 calls** | **0 calls** |
| `ggml_backend_tensor_get_async` | 4 calls | 12 calls | 4 calls |
| **`input_loop` total** | **33.6 ms** | **5794.0 ms** | **120.3 ms** |

So the meta backend's event chain, the synchronizes and the ids readback are all *cheap* — the input loop's
5.8 s in the tensor case is **`hipMemcpyAsync` itself**, reached through `ggml_backend_tensor_copy` /
`ggml_backend_tensor_set` (not the `*_async` wrappers, which is why instrumenting those showed 0).
(§13's "cross-device event latency" hypothesis for (B) is therefore also closed.)

**5. The two levers, both now measured rather than guessed.**

* **(A) Pin the source** so the uploads are truly asynchronous and two devices' DMAs overlap.  The probe
  says pinned returns in 0.001 ms instead of blocking 10.5 ms, so the mirrored upload stops being a 2x
  serialization.  **Expected: the tensor prefill approaches the 1-GPU time (~1.4 s vs 2.87 s, i.e. ~2x).**
  In llama.cpp the source is the mmap'd model file (pageable) or a malloc'd host buffer; a
  `cudaHostRegister`'d host buffer (`ggml_backend_cuda_host_buffer_type`) or pinning the expert tensors
  is the shape of the change.
* **(B) Make the pruning actually prune.**  The uploads are FULL 144 MiB tensors, yet §8 measured the
  used-expert set as *25 of 256* at ub 2048 - a ~10x smaller range.  If the full-tensor copies are
  avoidable, the transfer drops from ~17 GiB to ~1.7 GiB per pass and **the bottleneck moves back to
  compute** (~0.26 s), which would be far more than 2x.  This is the highest-value item found in the whole
  campaign and it helps **every** `-ncmoe` configuration, 1 GPU included.

**6. So the campaign's original premise was right and the detours are closed.**  §1 said the tensor path
duplicates the expert upload; §11 concluded the ring hides it. **The ring hides the *wait*, not the
transfer**: a `GGML_RING_STATS` wait of 545 us is consistent with a 1.26 s of *transfer* on the critical
path.  The duplication is real and it is the cost.  Splitting the experts (§12's plan) would halve the
volume per device - which is exactly lever (B) - so the campaign's target and this new finding are the same
thing, and (B) is available *without* the split machinery (and therefore without the partial-reduce fault
that blocks §12).

## 17. Eighth probe (2026-09-27): the split-copy fault, localized; and what it is not

Patch: `exp8-fault-localization.patch` (`GGML_META_EXECDBG` extended to dump the failing subgraph's
pointers/geometry per device).  Repro: `-sm tensor -ncmoe 99` + `GGML_META_SPLIT_COPY=1`, any `ub`
(`-p 128 -ub 128` fails fastest).

**Symptoms.**  A device memory-access fault (`Memory access fault by GPU node-2 ... Page not present`)
at the **first offloaded MoE op**, `ffn_moe_gate-0 (MUL_MAT_ID)`, on dev 0.  **It is non-deterministic**:
`-ncmoe 1` completed in one run (917 t/s) and faulted in the next; the fault address differs every run
(0x7f24a3a10000, 0x7f8f8da00000, 0x7fe510e02000, ...).  A race, not a geometry error.

**The geometry is exactly right.**  `EXECDBG` for the failing op vs the working mirrored run:

| | mirrored (works) | split (faults) |
|---|---|---|
| dst | `ne=[512,8,128]` nb=[4,2048,16384,2097152] | `ne=[256,8,128]` nb=[4,1024,8192,1048576] |
| src0 weight | `ne=[2048,512,256]` nb=[144,1152,**589824**,150994944] | `ne=[2048,256,256]` nb=[144,1152,**294912**,75497472] |
| src1 activation | `[2048,1,128]` | **identical** |
| src2 ids (VIEW) | `[8,128]` nb=[4,1024,...] | **identical** |

So every field is the correct half-version, `nb[2] == ne[1]*nb[1]` holds on both sides, and the activation
and ids are byte-identical to the case that works.

**What is ruled out, each by measurement:**

* **MBM/MMB** — `GGML_CUDA_MMB=0`, `GGML_CUDA_MMB_ROUTED=0`, `GGML_CUDA_MMB_GLU=0` (and combinations) all
  still fault.
* **the routed-compact MoE kernel** (delivery kernel, `GGML_CUDA_DISABLE_MMQ_ROUTED=1`) — still faults, so
  it is the *plain* MMQ `MUL_MAT_ID` path.
* **the `mmq_args`** — every field is derived from the simple tensor (`GGML_TENSOR_BINARY_OP_LOCALS` →
  `s02 = nb02/ts_src0`, `nchannels_x = ne02`), so `stride_channel_x` is 294912/144 = 2048 blocks for the
  split against 4096 mirrored, i.e. correct; `ne02` (the expert count) is unchanged by an axis-1 split.
* **buffer bounds** — `GGML_META_BUFDBG` prints `ok` for every expert copy (`need ~116 MiB` vs
  `have 229 MiB` for the worst case).
* **the CPU-vs-layer hypothesis** — `-sm layer` "works", but `GGML_META_SPLITDBG` shows **zero** split
  expert copies there (layer split never consults the policy for them), so that datapoint proves nothing.

**What is now the leading suspect: the `PARTIAL` reduce path.**  The split states the meta assigns
(`GGML_META_DEBUG=1`, needs `-v`) are:

```
Meta(...)#blk.0.ffn_gate_exps.weight#0 [NONE, 1, {256x1, 256x1}]
  -> ffn_moe_gate-0 [MUL_MAT_ID, 0,    {256x1, 256x1}]      <- plain split, no reduce
  -> ffn_moe_swiglu-0 [GLU, 0, ...]
Meta(...)#blk.0.ffn_down_exps.weight#0 [NONE, 0, {256x1, 256x1}]
  -> ffn_moe_down-0 [MUL_MAT_ID, PARTIAL, {0x1, 0x1}]       <- the ONLY new state vs any working config
```

The gate/up split needs no reduction (axis 0 output split), so `PARTIAL` on the down op is the only
machinery this change newly exercises, and it is exactly what §12 suspected.  Note the fault is *reported*
at the gate launch but launches are asynchronous, so the crashing kernel is not necessarily the gate's.
**RETRACTED (maintainer question, checked):** §17's first draft called the trailing `min(expert_size,512)`
MMQ pad placement a data-corruption bug.  It is not.  The pad is deliberate upstream code ("copy a bit
extra ... so there are no NaNs in the padding of the last expert ... necessary for MMQ in the CUDA
backend"), and the split path reproduces it correctly per device: the pad's source is
`i_stop*chunk_size_full + j*chunk_size_j` and its destination is `i_stop*chunk_size_j`, i.e. exactly the
first bytes of the next expert's slice *on that device* - which is what the mirrored path writes too.
No bug, nothing to fix.

**The next experiment (do this first - it is small and decisive).**  Instrument the reduce step inside
`ggml_backend_meta_graph_compute` (`n_reduce_steps`, `max_tmp_size`, `node_red`, and the
`PARTIAL` subgraph split) and dump it for the split run, as §12 already proposed.  The specific question:
for a `PARTIAL` MUL_MAT_ID whose src0 is a **COMPUTE-buffer copy** rather than a static weight, is
`max_tmp_size`/`node_red` set up at all?  The prime hypothesis remains the compute-container lifecycle
(§12): the COMPUTE container is double-buffered and cleared per `graph_compute`, which is not the
container the `-ncmoe 0` split (which works) uses.

## 18. Ninth probe (2026-09-27): the fault is the **ids view's device pointer**, not the kernel or the reduce

Instrument in `exp9-ids-view-probe.patch` (extends `METAEXEC` to read back the `MUL_MAT_ID` `src2` ids
*in process*, before the launch, using the same strides the kernel uses).

**The measurement.**  In the working (`SPLIT_COPY=0`) run the readback succeeds and shows the ids are
perfect:

```
METAEXEC  ids n=1024 n_expert=256 min=0 max=255 out_of_range=0 firstbad=-1
```

In the splitting (`SPLIT_COPY=1`) run, the log dumps the gate graph - including
`src2 ffn_moe_topk-0 op=VIEW data=0x7f9a1ce40080 ne=[8,128,1,1] nb=[4,1024,131072,131072]` - and then
**faults before printing any ids**, with the fault address ~7 GB beyond every tensor again
(`0x7f9bc1a00000`).

**Bracketed, so this is measured and not inferred** (a device fault is asynchronous, so "the fault is
in my readback" had to be proven): the readback prints `READEIDS begin <name>` before and
`READEIDS end <name>` after.

| run | begin | end |
|---|---|---|
| `SPLIT_COPY=0` | 480 | **480** |
| `SPLIT_COPY=1` | **1** | **0** |

The first device read of the ids' parent (`ffn_moe_topk-0`, layer 0) begins and never completes.  So the
crashing access is a plain device read of `view_src->data + view_offs` - and the kernel would fault on
exactly the same pointer.  That reorders the diagnosis completely:

* it is **not** the MMQ `MUL_MAT_ID` kernel (the fault precedes it),
* it is **not** the partial reduce (that graph is `n_subgraphs=1`, so no reduce runs at all),
* it is **not** garbage ids (`min=0 max=255`, and the same readback is clean in the mirrored run),
* it is the **ids view's device pointer** - `ffn_moe_topk-0`'s `data`/`view_src` resolves to something
  ~7 GB outside the device buffers, and the kernel would have faulted on exactly the same address.

**Why this is the right shape of answer.**  `ggml_backend_meta_buffer_init_tensor_impl`'s view branch is
the only place that can produce a pointer like this:

```c
if (t_ij->view_src != nullptr) {
    t_ij->data = (char *) t_ij->view_src->data + t_ij->view_offs;
    ...
}
```

It remaps `t_ij->view_src` to the *simple* tensor **only when the parent's buffer is a meta buffer**
(`ggml_backend_buffer_is_meta(t_ij->view_src->buffer)`).  If the parent of the ids view is not in a meta
buffer - e.g. it is the scheduler's ids copy sitting in a host/CPU buffer, or a simple tensor from an
earlier **compute-container generation** - then `view_src->data` is a host pointer (or a stale one) and
the GPU reads host memory: the fault address is ~7 GB into the host mapping, which is exactly what every
fault in this campaign has shown.  That is §12's compute-container-lifecycle suspicion, now with a
concrete, single-place mechanism.

**The next experiment (very small).**  In the view branch above, print for the ids view: the parent's
name, `view_src->buffer` (pointer + `buft` name via `ggml_backend_buffer_name`), whether
`ggml_backend_buffer_is_meta(view_src->buffer)` is true, `t_ij->view_src->data`, `t_ij->view_offs`, and
`t_ij->data` - in **both** `SPLIT_COPY=0` and `=1` runs, and diff them.  If the parent is not a meta
buffer in the split case (or the generation differs), the fix is to make the ids copy's simple tensor a
proper meta-backed slice in that path rather than inheriting a raw pointer.

## 19. Tenth probe (2026-09-27): bisected by forced sync - the guilty subgraph is `ffn_moe_up + swiglu`

The fault is asynchronous, so every earlier "it faulted here" really meant "it *surfaced* here".  A
per-launch sync (`GGML_META_SYNCEACH`, in `exp10-sync-bisect.patch`) fixes that: after every
`graph_compute_async` the meta backend synchronizes and prints `SYNCEACH sub=.. dev=.. nodes=.. first=..
last=..`, so the last print before the death names the guilty launch.

* §18's readback was a red herring for *location*: with it **gated off** (`GGML_META_READIDS`) the split
  run still faults, just later.  A `tensor_get` synchronizes the stream, so the read only moved where the
  already-pending fault was drained.
* With `GGML_META_SYNCEACH=1`: `SPLIT_COPY=0` completes (61 t/s - the syncs dominate); `SPLIT_COPY=1`
  prints both `sub=0 dev=0/1 nodes=1 first=ffn_moe_gate-1` and then dies, i.e. the **next**
  `graph_compute` is the guilty one.

That next one is the **2-node `ffn_moe_up-0 + ffn_moe_swiglu-0` subgraph**, and its dump shows why it is
the interesting one:

```
METAEXEC   ffn_moe_up-0  data=0x7f5583e62480 ne=[256,8,128] nb=[4,1024,8192,1048576]
METAEXEC   ffn_moe_swiglu-0 (GLU)  data=0x7f557ac62480 ne=[256,8,128]
METAEXEC     src0 ffn_moe_gate-0  data=0x7f5583c62480    <- computed in the PREVIOUS graph_compute
METAEXEC     src1 ffn_moe_up-0    data=0x7f5583e62480    <- computed in this one
```

So it dies on the first consumer of a **split MoE intermediate that crosses a `graph_compute` boundary**.
Under `-ncmoe` the scheduler's op-offload hands the meta backend *one op per graph_compute*, and the meta
compute container is **double-buffered and cleared per `graph_compute`** - so the gate output from
generation A must survive the build of generations B, C, ...  Every pointer in the dump is sane and
symmetric with the working mirrored run, which is why this reads as a **lifetime/generation** bug rather
than arithmetic: §12's hypothesis, now with a named victim.

**Next experiment (cheap, and it separates the two mechanisms).**  Put the gate and the up into the *same*
`graph_compute` so the intermediate does not cross a boundary - the delivery's **fused gate+up+GLU** path
does exactly that (`GGML_CUDA_MMB_GLU=1`, or the paired/fused-GLU arm).  If the fault disappears, the
container generation is confirmed and the fix belongs in the container lifetime (keep a split
intermediate's simple tensor alive across generations), not in the split arithmetic.

## 20. TRACKED FOLLOW-UP (do not lose): asynchronous/pinned uploads - lever (B1), the maintainer's "queue the results"

Raised by the maintainer and deliberately left open while the split is finished.  §16/§16b measured it:
`hipMemcpyAsync` from a pageable source (the mmap'd model file) **blocks the host for the whole call**
(10.461 ms at 144 MiB against 0.001 ms pinned), so the two devices' uploads cannot overlap and the
mirrored upload is paid twice.  The prize is measured: `-ncmoe 0` (no uploads at all) **7945 t/s** vs
**713 t/s** for `-ncmoe 99` at ub 2048 - **11x** - so it is the *stalls*, not the bytes.

1. **Pin the source** (`cudaHostRegister` / `ggml_backend_cuda_host_buffer_type`) so the copies are truly
   asynchronous and both devices' DMAs overlap with compute.
2. **Queue/overlap the transfer with compute**: with pinned sources the host can run ahead and issue
   layer N+1's expert uploads while layer N computes - which is what the §11 staging ring was built for.
   The ring hides the *wait*; the transfer is still on the critical path, so the two combine.

This lever and the split are complementary: the split halves the per-card volume, pinning removes the
serialisation.


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

## 21. Eleventh probe (2026-09-27 evening): the fault needs 2 GPUs, is a probabilistic race, and the split path is functionally CORRECT

Patch: `exp11-fault-needs-2gpu.patch` (cumulative; exp10 + the new instruments below).  Tree: `~/llama-r12`
(branch `wip-tensor-split`, tip `de71ddd58`); verified at session start that the working diff == `exp10`
even though the `rdna-boosts` branch had been recreated in between (the `wip-tensor-split` branch is
independent, so the reboot did not touch it).

**This section supersedes §19's container/generation hypothesis and §18's ids-view suspicion.  Neither is
the cause.**  What it establishes, all by measurement:

### 1. The split path is CORRECT on one GPU, and `-sm layer` is still the wrong control

* **1 GPU, `SPLIT=1` (`-ncmoe 99 -sm tensor -p 128 -ub 128`): rc=0, pp128 157.5 t/s** vs 155.5 for
  `SPLIT=0`.  The split-state machinery, the axis-1/axis-0 propagation and the MMQ `MUL_MAT_ID` consumer
  are all fine there (the splice degenerates to one full-size segment, so this does not exercise the
  multi-device splice — but it does prove nothing else in the path is broken).
* **The 2-GPU split path is bit-identical to mirrored when it runs.**  With `SPLIT=1 SYNCUPLOAD=1`, the
  same-seed greedy text of `llama-cli -ncmoe 99 -sm tensor -p "The capital of France is" -n 32 --seed 42
  --temp 0 --reasoning off` is `sha=359ff4337837` — **identical to `SPLIT=0`** (35 chars, both).  So the
  split does not change the arithmetic result on this gate.
* **2 GPUs, `SPLIT=1`: fault, every run (~12/12), always reported on `GPU node-2` (device 1).**
* **2 GPUs, `SPLIT=0` (mirrored): rc=0.**  The §0 control.
* `-sm layer` remains a useless control (`SPLITDBG` shows zero split expert copies there — layer split
  never consults the policy).

### 2. The fault scales to a ONE-LAYER repro, so it is cheap to iterate on

`-ncmoe 1` (a single layer's experts on the host, so **one** split copy per device) faults the same way:
`Memory access fault by GPU node-2 ... Page not present or supervisor privilege`, rc=141.  `-ncmoe 3` and
`-ncmoe 8` do too.  Use `-ncmoe 1` from now on — same fault, a fraction of the setup.

### 3. Ruled out this session (each measured, each a real experiment — do not re-derive)

| hypothesis | instrument / A/B | result |
|---|---|---|
| H2D staging ring | `GGML_SCHED_STAGE=0` | still faults |
| all-reduce (internal / RCCL / copy-engine) | `GGML_CUDA_ALLREDUCE=none\|nccl\|internal` | still faults on all three (so the butterfly fallback too) |
| the PARTIAL reduce's temp/subgraph machinery | added `SYNCEACH REDUCE` (draned the reduce after each subgraph) | every reduce prints `ok`; fault is in the *next* graph |
| the ids view's device pointer (§18) | `GGML_META_VIEWDBG` | parent is meta-backed, `data` is a sane device pointer, identical to the mirrored run |
| the double-buffered compute **container lifetime** (§19) | `GGML_META_CNDBG` (log any lookup that misses the current container but is found in the other) | **zero** misses — every simple-tensor lookup hit the current container |
| a wrong/absent `ggml_cuda_set_device` in the async copy | added `ggml_cuda_set_device(cuda_ctx->device)` to `set_tensor_async`/`set_tensor_2d_async` | still faults (reverted — not the fix) |
| upload↔consumer **stream** mismatch | `GGML_STREAMDBG` | the split upload and the consumer use the **same** per-device stream (`dev0 0x…1150`, `dev1 0x…2f70`) |
| geometry of the splice | `GGML_META_SPLITDBG` + `GGML_META_BUFDBG` | ranges are chunk-aligned, per-device `chunk_j`/`nb` are right, no `OVERFLOW`, volume matches the pruned used-expert groups (ub 8192: `blk.0.ffn_gate_exps` = 205 experts + 512 pad, then 50 experts) |

### 4. What the fault actually looks like: a probabilistic race

`GGML_META_SYNCUPLOAD=<mode>` adds a host `ggml_backend_synchronize` after the split H2D copies:

| mode | where it syncs | ub 128 | ub 8192 |
|---|---|---|---|
| — (off) | — | faults (~10/10) | faults (1/1) |
| 1 | both devices, after every split upload | **passes (4/4)** | **faults (3/3)** |
| 2 | device 1 only, after every upload | faults | — |
| 3 | both, only the first two uploads | faults | — |

So a full barrier after every split upload **narrows but does not close** the race: it zeroed the fault
at ub 128 (4/4) and still lived with it at ub 8192 (0/3).  That is the signature of a timing-dependent
race, not a deterministic geometry error.  It also means the earlier "the sync fixes it, so it is an
ordering bug" reading is only half true — the barrier removes *a* window, and ub 8192 opens another.

`SYNCUPLOAD=1` at ub 2048 measures **334 t/s** (`-ncmoe 99 -sm tensor`) against **716** for the mirrored
control, i.e. the barrier itself is the dominant cost at small ub, so this is **not** yet a
measurement of the split's value (ub 8192 faults before it can be timed).

### 5. A real (benign) splice bug found on the way — the tail source for device `j>0`

In `ggml_backend_meta_set_tensor_async`'s splice branch the short tail is issued as

```c
set_async(simple_tensor_j, data + offset_j + (i_stop - i_start)*chunk_size_full,
          dst_base + (i_stop - i_start)*chunk_size_j, rem);
```

The destination is right, but the **source** is not: `rem = (offset+size) % chunk_size_full` is the *whole*
range's remainder and can be smaller than `offset_j` (device 1's sub-block offset).  For `ffn_down_exps`
(Q5_K, `chunk_full = 352`, `chunk_j = 176`, `rem = 512 % 352 = 160`) device 1's tail reads
`[range_end - 32, range_end + 128)`, i.e. **past the range** (and past the tensor for the last expert).
For `ffn_gate_exps` (`chunk_full = 589824`, `rem = 512`) the reads still land inside the next expert, so it
is invisible.  The tail only ever carries the 512-byte MMQ pad, so wrong content is harmless — but it is a
real out-of-range read that should be fixed (distribute the `rem` bytes to the device whose sub-block they
fall in, or use a single per-device source offset that accounts for it).

### 6. Where to go next — DONE, see §22

1. **Get the faulting kernel** — done: it is the splice's pageable `hipMemcpy2DAsync`
   (`__amd_rocclr_copyBufferRectAligned`), §22.2.
2. **Fix the §5 tail bug** — done (§22.1); it was not the fault.
3. **Pin the source** — tried, does not fix it (§22.4).
4. **The efficient fix** is §22.5: 1-D H2D the whole range to a device staging slot, then compact it
   with a D2D 2-D copy (or a small kernel).  Then measure the §5 payoff.

## 22. Twelfth probe (2026-09-27, later): the fault is the splice's pageable `hipMemcpy2DAsync` — ROOT CAUSE

Patch: `exp12-2d-copy-root-cause.patch` (exp11 + the tail fix + `GGML_META_NO_2D` / `GGML_META_TOUCHSRC` /
`GGML_META_UPLOADDBG` / `GGML_META_PINSRC`).  Probes: `h2d2dprobe.cpp`, `h2d2dcold.cpp`, `h2d2dalign.cpp`,
`d2d2dprobe.cpp`.

### 1. Step #1 done: the tail bug is fixed (and was NOT the fault)

`ggml_backend_meta_set_tensor_async`'s splice now issues the short tail only to the device that owns it:

```c
if (rem != 0 && offset_j < rem) {
    const size_t tail_len = std::min(rem - offset_j, chunk_size_j);
    ... copy tail_len bytes from data + offset_j + (i_stop-i_start)*chunk_size_full ...
}
```

`GGML_META_TAILDBG` confirms it (gate/up `rem=512` → dev1 `tail_len=0`; down `rem=160`, `off_j=176` → dev1
`tail_len=0`), same-seed greedy text is still **bit-identical** (`sha=359ff4337837`), and **the fault is
unchanged** — as predicted, those over-reads stayed inside the host tensor and the destination writes were
in bounds.

### 2. The faulting kernel is the copy, not the MMQ consumer

`HSA_ENABLE_SDMA=0` makes the ROCm runtime name it:

```
Memory Fault Error [host: soar, GPU index: 0, faulting addr: 0x…, kernel: __amd_rocclr_copyBufferRectAligned]
```

`__amd_rocclr_copyBufferRectAligned` is the ROCclr implementation of `hipMemcpy2DAsync`.  The
`GGML_META_UPLOADDBG` dump of the last copy before the death shows sane arguments whose fault address lands
**inside the source range** of the previous (device-0) copy:

```
UP2D …#blk.0.ffn_gate_exps.weight#0 axis=1 dev=0 i=[28,38) cf=589824 cj=294912 nc=10
     src=[0x7fd0a01f9720, 0x7fd0a0751720) dst=[0x7fcefdc42480, 0x7fcefdf12480)
     buf_base=0x7fcefd200000 buf_size=240522368 nbytes_simple=75497472
UP2D … dev=1  src=[0x7fd0a0241720, 0x7fd0a0799720) …
faulting addr: 0x7fd0a0215000   (inside the dev-0 src range)
```

### 3. THE DECISIVE A/B: 1-D copies on the same addresses do not fault

`GGML_META_NO_2D=1` replaces the single `ggml_backend_tensor_set_2d_async` call with a loop of the 1-D
`ggml_backend_tensor_set_async` (exactly the generic `set_tensor_2d_async` fallback, and exactly what the
mirrored path already uses):

| ub 8192, `-ncmoe 99 -sm tensor`, no sync | result |
|---|---|
| 2-D splice (`NO_2D` unset) | fault (many/many) |
| `NO_2D=1` (1-D loop) | **rc=0, pp8192 65.4-67.7 t/s (3/3)** |
| `NO_2D=1` at ub 128 | **rc=0, pp128 2.75 t/s** |

Same source addresses, same destination addresses, same stream, same ordering — only the API differs.  So
the fault is **entirely in `hipMemcpy2DAsync` from this pageable (model-mmap) source**, and the split
logic, the splice geometry and the MMQ consumer are all correct.

The `NO_2D` numbers are a *diagnostic*, not a fix: the 1-D loop issues one copy per 176/294912-byte chunk
(thousands per weight), so it is ~40x slower than the mirrored 2736 t/s.

### 4. What I could NOT establish, and what was ruled out

* **The standalone probes do not reproduce it.**  `h2d2dprobe`/`h2d2dcold`/`h2d2dalign` run the same
  shapes (`cf=589824/cj=294912/nc=10` and `cf=352/cj=176/nc=2048`) from pinned, `malloc`, cold read-only
  file mmap, at every destination alignment (0/16/128/256/4096) and on both devices — all PASS.  The real
  trigger therefore involves some state of the live workload (the 7.8 GB compute buffer, the actual model
  mmap, the interleaving with compute) that the isolated probe does not capture.  Do not conclude from the
  probe passing that the call is safe.
* **Cold source pages are not it**: `GGML_META_TOUCHSRC=1` (touch every page on the CPU first) still faults.
* **Pinning the source is not a fix**: `GGML_META_PINSRC=1` (`hipHostRegister` on the pageable range; the
  first range returns "no error", i.e. the model mmap was not already pinned) still faults.
* `HSA_ENABLE_SDMA` does not matter: the fault happens on both the SDMA and the blit-kernel paths.
* D2D 2-D copies are **fine** (`d2d2dprobe`: H2D and D2D both PASS standalone).

### 5. Recommended fix (efficient, avoids the broken call)

Do the compaction in two steps, neither of which is a pageable H2D 2-D copy:

1. **1-D H2D the whole (pruned) range** into a per-device device staging slot — the path the mirrored
   upload and the block-06 staging ring already use, and which is proven correct here; then
2. **D2D 2-D `hipMemcpy2DAsync`** (or a tiny device kernel) to compact the staged range into the
   per-device split weight tensor.  D2D 2-D is not implicated by the fault and passes the probe.

The block-06 staging ring already owns per-device device slots and the redirect plumbing, so the natural
home is a `stage_input` variant that stages a *split* copy and then compacts on device.  Note the H2D
volume becomes the full pruned range per device (the mirrored volume), not the half; §16b measured that the
mirrored upload is ~99 % *wait*, not bytes, and §9/§11 concluded the lever is the **compute**, so that is
an acceptable trade — and it may compose with the §20 `pin the source` work rather than replace it.

Until then: the fault is fully worked around for *correctness* experiments with `GGML_META_NO_2D=1` (slow)
or `GGML_META_SYNCUPLOAD=1` at small ub only (and it does not survive ub 8192).  **The campaign's payoff
number (the §5 table) still needs the efficient fix.**

## 23. Thirteenth probe (2026-09-27, latest): SOURCE PINNING — the real lever, and it redirects the campaign

Follows §22.  Two things were asked: (a) widen the pinned ring, (b) implement source pinning for the
`-ncmoe` expert weights.  Both are done; (b) turned out to be much bigger than the split itself.

### (a) Ring depth does not help — ~3.4 t/s at ub 8192 is the honest number

`GGML_META_PINRING=<n>` swept over the gather+1D splice at ub 8192, pageable source:

| ring | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| pp8192 | ROCm error | 3309 | 3265 | 3346 | 3157 | 2613 | 2653 |

**Deeper is slower, not faster** — the host wait is not the limiter; the number of distinct pinned buffers
is (one hot buffer stays in cache; many cold ones do not).  So §22's 4047 t/s was only reachable by
*not waiting at all*, which is exactly the data race that faulted 1-in-3; **3436 is the correct figure**
(+26 % over the 2732 mirrored baseline).  Do not chase 4047 by widening the ring.

### (b) SOURCE PINNING: the `-ncmoe` experts are pageable mmap, and that is the whole bug

The loader already *offers* a pinned host buffer type for CPU tensors (`make_cpu_buft_list` adds
`ggml_backend_dev_host_buffer_type(dev)`, and its own comment says that storing CPU tensors there "reduces
the time spent on data transfers" when batches are offloaded to a GPU).  But `llama-model-loader.cpp`'s
`select_weight_buft` then **throws it away** when `use_mmap`:

```c
// avoid using a host buffer when using mmap
if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev)) {
    buft = ggml_backend_dev_buffer_type(cpu_dev);   // -> CPU_Mapped, i.e. the pageable model mmap
}
```

`GGML_SCHED_BUFTDBG` confirms it: `blk.0.ffn_gate_exps.weight buft=CPU_Mapped`, while small tensors get
`ROCm_Host`.  So under `-ncmoe` every expert upload reads the pageable model mapping — which is why
`hipMemcpy2DAsync` faults (§22), and why the mirrored uploads stall the host (§16b).

**Fix (implemented, `exp13`):** skip that downgrade for `MUL_MAT_ID` weights — precisely the weights the
scheduler's op-offload uploads every ubatch.  It is a one-condition change, default on, with
`LLAMA_MMAP_HOST_EXPERTS=0` to restore the old mmap behaviour:

```c
if (use_mmap && buft_dev && buft == ggml_backend_dev_host_buffer_type(buft_dev) &&
        !(host_experts && op == GGML_OP_MUL_MAT_ID)) { ... downgrade ... }
```

It only affects CPU-resident `MUL_MAT_ID` weights, i.e. `-ncmoe` (on-device experts keep their GPU buft;
pure-CPU users get the CPU buft, so nothing changes there).  Cost: the expert set is pinned, not mmap'd —
~17 GiB non-swappable for the 35B-A3B.

### The payoff table (`-ncmoe 99`, gfx1201 x2, pp t/s)

| ub | mirrored mmap | **mirrored pinned** | split+gather pageable | split+gather pinned | split plain-2D pinned |
|---|---|---|---|---|---|
| 128 | 63 | 177 | 210 | 219 | — |
| 2048 | 715 | 1396 | 1427 | **1780** | — |
| 4096 | 1413 | **2706** | — | 2451 | — |
| 8192 | 2732 | **5066** | 3436 | 3448 | 948 |

(`-lm none`/`-lm mlock` reproduce the "pinned" column without the loader fix; the loader fix does it with
the normal mmap load mode.  Same-seed greedy text is **bit-identical** in every cell, `sha=359ff4337837`.)

The **split's crossover is between ub 2048 and 4096**: it wins at 2048 (1780 vs 1396) and loses from
4096 on (2451 vs 2706).  The tensor-vs-layer gap also collapses with ub: pinned `-sm tensor` beats
pinned `-sm layer` by +24 % at 8192 but only +3.5 % at 4096 (2613), and pinned `-sm layer` does not move
with pinning at all (2599 -> 2613, it is compute-bound on its single device).  For reference, with the
experts **on** the device (`-ncmoe 0`, the 21 GiB model fits in 2x32 GiB) `-sm tensor` is 7695 at ub 8192
and 8244 at ub 4096, i.e. still far ahead of any host-resident-expert config.  Note also that the
pinned-tensor pass time is nearly flat (~1.5 s) from ub 2048 to 8192, i.e. a **fixed per-pass cost
dominates** there (see §13), which is why its t/s scales almost linearly with ub.

### What this means for the campaign — read this before doing any more split work

1. **Pinning the source is the campaign's biggest single lever, it is simple, and it helps the MIRRORED
   path**: `-sm tensor -ncmoe 99` goes 2732 → **5066 t/s** at ub 8192 (**+85 %**), 715 → 1396 at ub 2048
   (+95 %), 63 → 177 at ub 128 (+181 %).  It also removes the §16b host-blocked pageable copies and the
   §22 fault (the split then runs fault-free even with the plain 2-D copy).
2. **The expert split is now only a small-ub win, and the crossover is between ub 2048 and 4096.**
   With a pinned source the split+gather is 219 vs 177 (ub 128) and 1780 vs 1396 (ub 2048) — a real win —
   but 2451 vs 2706 (ub 4096) and 3448 vs **5066** (ub 8192), losses, because its host gather grows with
   `ub` while the mirrored upload is a single async 1-D copy per group.  The campaign's premise ("the
   split is the core lever") no longer holds once the source is pinned.
3. **If the split is to matter at large ub, the host gather must go.**  The remaining candidate is §22.5
   done properly: 1-D H2D the range to a device staging slot (async, from pinned) and compact it with a
   **small device kernel** (not `hipMemcpy2DAsync`, which is slow even from pinned — 948 t/s).  That is a
   real implementation, and is only worth it if it can beat 5066 — which needs the split's compute
   halving (plus its down-projection all-reduce) to exceed the mirrored path's pinned upload.

**Recommended next action:** ship the pinning as the delivery win (it is 3 lines and +85 %), and re-open
the split as a separate question only if a device-side compaction is built.

## 24. Handover: the delivery candidate, the extrapolation, and the open "result queueing" question

### The delivery candidate (NOT promoted)

`0000-source-pinning-llama-mmap-host-experts.patch` — a clean, self-contained 3-line change to
`src/llama-model-loader.cpp` (the loader exception in `select_weight_buft`; §23.(b)).  Apply with
`git apply` on a checkout at the delivery base, or fold it into a block when promoting (block 06, the
delivery's general system-operations bucket, is the natural home).  Kill-switch:
`LLAMA_MMAP_HOST_EXPERTS=0`.

It is deliberately **not** in `patches/` yet.  Per the campaign's promotion rule the combination has to be
re-validated, and the maintainer wants the **result-queueing** question answered first (§24.3).

### What is proven (recap of §23, all measured, gfx1201 x2)

| `-ncmoe 99`, pp t/s | ub 128 | ub 2048 | ub 4096 | ub 8192 |
|---|---|---|---|---|
| mirrored, pageable | 63 | 715 | 1413 | 2732 |
| **mirrored, pinned** | **177** | **1396** | **2706** | **5066** |
| split+gather, pinned | 219 | 1780 | 2451 | 3448 |
| `-sm layer`, pinned | — | — | 2613 | 4127 |
| `-sm tensor -ncmoe 0` (VRAM) | — | — | 8244 | 7695 |

* Pinning the source: **+85 %** at ub 8192, +95 % at 2048, +181 % at 128; bit-identical output.
* Tensor-mirrored + pinned now beats `-sm layer` at every ub measured.
* The split is a small-ub-only win (crossover 2048-4096); its host gather is what caps it.
* Decode is untouched (under `-ncmoe` it runs the MoE on the CPU — §8.4), so the change is prefill-shaped.

### The extrapolation (maintainer's framing — endorsed by the mechanism)

**This platform is the pessimistic case, and the numbers above are therefore a lower bound.**  `soar`
gives each R9700 only **PCIe4 x4** (§11, ~14.45 GB/s per card), on a 2026-era host.  The win from pinning
is a *host-stall removal* (§16b: a pageable `hipMemcpyAsync` blocks the host for the whole call, so the
two cards' DMAs cannot overlap; a pinned copy returns immediately), so it scales with how much of the pass
the upload was: on a **PCIe5 x8** pair (~4x the per-card bandwidth) the same pinned upload costs far less
wall time, and on **faster host memory** the gather/copy side is cheaper too.  Neither narrows the gap to
VRAM residency — the weights still have to cross the link every ubatch — so **parity with `-ncmoe 0` is not
a realistic target**, but the point of the exercise is not parity: it is to make a MoE whose experts do not
fit in VRAM *usable* instead of pathological, and §23 shows that is achieved (the config goes from the
worst place to the best host-resident place, and past `-sm layer`).  The safe expectation to carry into a
faster box is: **same conclusions, larger absolute numbers** — and, if anything, pinning becomes *more*
valuable relative to the 2-D/gather workarounds, because those exist only to route around a pageable
source that a fast host would make cheap to keep resident.

### 24.3 The open question: result/upload queueing (next session, to its very end)

**Hypothesis (maintainer):** the split scenario is held back by the absence of result queueing — i.e. the
upload of split N+1 is not overlapped with the compute of split N.

**Where that lives today.**  In `ggml_backend_sched_compute_splits` the scheduler waits for the split
backend, uploads the split's inputs, then launches the split's graph — all on the same device stream, so
upload and compute are serialized per split.  The block-06 staging ring (`stage_input` / `stage_upload` /
`stage_wait`) was built to decouple exactly this (it uploads on an auxiliary copy stream and makes the
compute stream wait on a per-slot event), but `ggml_backend_meta_stage_input` **rejects a split
`input_cpy`** (`chunk_size_full != size`), so the split path never benefits from it.  The mirrored path
does use the ring, but the measured mirrored win came from pinning, not from the ring, so it is not yet
known how much queueing would add on top for either.

**What to build/measure next (both paths, split and mirrored):**

1. **Queue layer N+1's expert upload while layer N computes.**  Either extend `stage_input` to a split
   `input_cpy` (the splice would then run from the slot instead of the source) or add a plain per-split
   double-buffered upload with copy-stream events.  Success metric: the upload disappears from the
   critical path — i.e. `GGML_RING_STATS` shows ~0 exposed wait at every ub, and prefill rises.
2. **Remove the split's host gather** (its large-ub bottleneck): 1-D H2D the pruned range to a device
   slot, then compact it with a **small device kernel** (not `hipMemcpy2DAsync`, which is slow from pinned
   — 948 t/s — and faults from pageable).  Only this can let the split stay ahead past ub 2048.
3. **Measure the full grid**: `{mirrored, split} x {queued, not} x {pinned} x ub {128,2048,4096,8192}`,
   plus decode, against the §23 table.  Decide promotion (pinning and/or split) only after that.

If queueing closes the split's gap, the right end state may be "pinning + queueing + split"; if it does
not, the right end state is just "pinning" (3 lines) and the split is closed as a small-ub-only curiosity.

## 25. Fourteenth probe (2026-09-27, continuation after compaction): the QUEUEING question is answered — the split loses on EXPOSURE, not volume

Follows §24.3.  The question was: does result/upload queueing help the split or the mirrored path?  **Yes
for both, and it is the whole story for the split.**  Instrument: `exp14-gather-queue.patch`
(`GGML_META_PINHOST=2` = queued H2D on the copy stream, `=4` = gather-only floor, `=5` = pure-compute
floor, per-device pin rings; `GGML_META_NOSYNC=1` = skip the scheduler's pre-upload sync — unsafe, for
measurement only).

### 25.1 Correction first: `GGML_META_SPLIT_COPY` defaults to **1**

The experiment gate in `src/llama-model.cpp` returns `1` when the env var is unset (only `0` restores the
r12 mirrored behaviour).  Every "mirrored" run in this session must therefore pass
`GGML_META_SPLIT_COPY=0` explicitly; without it you are measuring the split (933 t/s at ub 8192, and the
§22 fault from pageable).  That was the whole of the first hour's confusion — record it, and set the
default back to 0 in any promoted build.

### 25.2 Queueing is worth +34 % to the MIRRORED path

`-sm tensor -ncmoe 99`, pinned (loader fix), ub 8192 (`-b 8192`):

| `GGML_SCHED_STAGE` | pp8192 | note |
|---|---|---|
| 1 (ring: H2D on copy stream, issued early) | **4504** | queued |
| 0 (no ring: H2D on the compute stream) | 3370 | exposed |

So the block-06 ring's whole-tensor staging is not a memory trick — it is what lets the transfer overlap
compute.  Pinning removes the host *stall* (§16b); staging removes the *exposure*.  They are different
levers and both are needed.

### 25.3 The split could not queue at all

`GGML_SCHED_STAGE` makes **no** difference to the split (3460 vs 3532 at ub 8192): `stage_input` rejects a
split `input_cpy` (`chunk_size_full != size`, §11), so the split falls to the pruning path, whose H2D runs
on the compute stream.  The pruning path also does, per expert weight:

```c
if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
    ggml_backend_event_wait(split_backend, ...);
} else {
    ggml_backend_synchronize(split_backend);   // <-- taken for the meta backend
}
```

The meta backend sets `event_record`/`event_wait` to **nullptr** (§25.4), so `sched->events[meta]` is null
and the else runs: `ggml_backend_synchronize(meta)` is a **full host synchronize of both devices before
every upload**.  That is the serializer: the host cannot run ahead, so the next split's gather *and* H2D
are issued only after the previous split has finished.

### 25.4 The decomposition — the gather is NOT the bottleneck, the H2D is

Split pinned, ub 8192, `GGML_META_SPLIT_COPY=1 GGML_SCHED_STAGE=0`:

| variant | pp8192 | per pass | delta |
|---|---|---|---|
| `PINHOST=1` (gather + H2D) | 3586 | 2.28 s | — |
| `PINHOST=4` (gather, **no H2D**) | 6245 | 1.31 s | gather = **0.15 s** |
| `PINHOST=5` (no gather, no H2D) | 7037 | 1.16 s | H2D = **0.97 s** |
| mirrored `STAGE=1` | 5107 | 1.61 s | |

**The host gather is already overlapped (0.15 s exposed).  The exposed cost is the H2D transfer (0.97 s).**
The split's compute floor is **7037 t/s** — well above mirrored's 5107 — which is the *half the compute*
working as designed.  Hiding the transfer is worth ~2x on the split.

(A `hipcc` probe, `overlapprobe.cpp`, confirms an H2D on one stream *can* overlap compute on another on
this box: serial 46.3 ms vs two-stream 37.1 ms for a 512 MiB copy against ~13 ms of compute.)

### 25.5 Why mirrored wins at depth despite moving MORE bytes — the user's paradox

At ub 8192 the split transfers **half** the bytes of mirrored (one device's slice instead of the whole
tensor per device), yet mirrored is faster (5107 vs 3586).  It is not the bytes:

* **mirrored, staged**: H2D issued on the copy stream, before the scheduler's wait → **hidden**; the pass
  is the compute.
* **split, pruning path**: full host sync, then H2D on the compute stream → **fully exposed**; the pass is
  compute + transfer.

Moving *more* bytes efficiently beats moving *fewer* bytes slowly.  That is the entire §24.3 hypothesis,
confirmed from the other direction.

### 25.6 The ceiling, and the safe fix

With the sync removed **and** the transfer on the copy stream
(`GGML_META_PINHOST=2 GGML_META_NOSYNC=1`, ub 8192) the split reaches **4927** — parity with mirrored
(5107), from 3209.  `NOSYNC` is *unsafe* (it removes the WAR guard on a reused input buffer), so the real
fix is the one the maintainer described: **queue the gather results** so the upload is issued early into a
slot the current compute does not use, and the consumer waits on a per-slot event — i.e. teach the meta
`stage_input` the split case (§24.3.1), which also gives it `event_record`/`event_wait`.

Design for the next step (the split half is already sketched in `ggml_backend_meta_stage_input`'s
non-mirrored branch, but is gated out because it stages one *chunk*, not the whole tensor):

1. Give the meta backend `event_record`/`event_wait` (today they are `nullptr` and force the host sync).
2. Let `stage_input` accept a whole split tensor and stage each device's **compacted** slice into its ring
   slot on the copy stream — either a host gather into the slot's pinned staging (the `PINHOST` gather,
   already shown to be hidden) or a whole-range 1-D H2D plus a device-side D2D compact.
3. Point the consumer's simple tensor at the slot (the mirrored drain already does this, no D2D), so the
   split gets the same overlap mirrored has.

Expected: ~5000-7000 t/s at ub 8192 (vs mirrored 5107), and it should preserve the small-ub split win.

## 26. REASSESSMENT (2026-09-27): two external implementations, and what they say about this campaign

The maintainer asked for a stop-and-compare against two repos that claim to have solved host-resident MoE:
`~/Strata` (Niko1221/Strata) and GenerelSchwerz/llama.cpp `moe-cache`.  Both were read; both converge on
our §23 finding, and **neither implements `-sm tensor`**.

### 26.1 Strata (github.com/Niko1221/Strata) — CPU computes the misses, VRAM caches the hits

A from-scratch, **single-card NVIDIA** engine for Qwen3.8-Flash-Next (fixed geometry, no split mode at all
— `grep -i 'split_mode|tensor_split|multi-gpu' src include` is empty).

* Experts live in RAM (pinned); the **CPU computes the non-resident experts in place, concurrently with the
  GPU**; a frequency-profiled **VRAM expert cache** (`src/core/expert_cache.cpp`, `h_expert` 0.6447
  leave-one-out over 4105 slots) holds the hottest experts so the CPU reads only the misses.
* Their measured bottleneck is the **CPU expert pool** (663.6 MB of expert bytes per token at ~40 GB/s =
  16.2 ms of a ~53 ms token; "the pipeline hides 1.055 ms of 19.035" — 96 % exposed because the residual
  chain is strictly serial), **not** an H2D transfer.  Their fix is to stop reading bytes, not to move them
  faster.
* The cache is only half-wired today ("`moe_hit_grouped_s2` does not exist … the engine is slower by the
  fill cost and faster by nothing"), so even its own claim is not yet demonstrated.
* **Nothing here transfers to `-sm tensor`**: no second device, no split, CPU execution.

### 26.2 moe-cache (GenerelSchwerz/llama.cpp `moe-cache`) — a real VRAM expert cache, layer-split only

A maintained llama.cpp fork, **NVIDIA CUDA only**, opt-in (`--moe-expert-cache-mib`), with a large,
well-documented implementation (`ggml/src/ggml-cuda/moe-cache.cu` is ~14k lines).

* Per-layer, per-owner (per-GPU) **cache buffer type** holding N expert slots; **grouped decode kernels**
  compute the resident experts on the GPU; misses are staged/uploaded; prefetch, replacement, an early
  router, and CUDA-graph capture.  Measured **Qwen3.6 35B decode 42.8 -> 111.6 tok/s (2.6x)** at the same
  peak VRAM on a 16 GB RTX 5070 Ti.
* The host side (`moe-cache-host.cu/.cuh`) is the mature version of our §23: a `moe_host_source`
  (data/size/expert_stride), a bounded **pinned staging budget**, automatic expert-group registration,
  pageable fallback with pinned staging, and a dedicated **asynchronous copy worker thread**
  (`moe_host_copy_worker`).  Its tip commit is literally *"cuda: stage automatically pageable MoE legacy
  sources"* — pinned staging when an automatic registration is rejected.  That is §22/§23, independently.
* **It explicitly rejects tensor split**: *"Tensor split with a nonzero expert cache is rejected before
  weight allocation because the tensor-mode meta backend cannot consume the cached buffer.  Use layer
  split or set the expert cache size to zero for tensor split."*  Its multi-GPU work is layer-split
  "owner groups" only ("does not add tensor or expert parallelism").

### 26.3 What this means for the campaign

1. **Our pinning fix is corroborated by two independent implementations.**  Both keep the expert source
   pinned/registered and stage the pageable case through pinned host memory.  `0000-source-pinning-…` is a
   real, correct, externally-validated finding — promote it.
2. **Nobody has solved `-sm tensor` + host experts.**  moe-cache deliberately refuses it; Strata has no
   split.  There is no prior art to borrow, and no evidence the tensor-split prefill problem is where the
   value is.
3. **The proven, big lever is a VRAM expert cache for *decode* (2.6x)** — and it is inherently layer-split,
   because it is a per-owner CUDA buffer.  For **prefill** (every expert used) a cache cannot help.  This
   campaign is prefill-shaped.
4. **Our §25 "queueing" finding still stands**, and the reference design for it is moe-cache's copy
   worker: a dedicated thread doing the pinned copies, not a host `memcpy` loop on the scheduler's thread.
5. **New: the split's down projection is pathologically fine-grained to upload.**  `ffn_down_exps` splits on
   axis 0 (the contraction dim), so `chunk_size_full = nb[1] = 352` and `n_chunks = 524288`; the gathered
   staging therefore does **524288 tiny host copies per device per layer** — which is exactly why the
   §25.6 gather staging measured 970 t/s (worse than the exposed 3586).  Any split upload for the down
   projection needs a 2-D or device-side gather, not a host loop.

### 26.4 Recommended next steps

1. **Promote the pinning fix now** (block-06 amendment): 3 lines, +85 % prefill, externally corroborated,
   independent of the split question.  Get it gated and out.
2. **Do not invest further in the tensor-split download until (a) is shipped.**  The split is unique but
   (i) its down-projection upload geometry fights every transfer path, (ii) it only pays on a fast link at
   large ub, and (iii) the industry's answer (a decode expert cache) is layer-split.
3. **Consider re-aiming the campaign at decode**, where the proven 2.6x lives: a layer-split VRAM expert
   cache for `-sm layer -ncmoe` (our §0 note that decode "runs the MoE on the CPU" under `-ncmoe` is the
   same CPU-pool cost Strata fights).  That is the lever with evidence behind it.

## 27. vLLM prior art (2026-09-27): all of it is VRAM-resident — none host-streams experts

The maintainer pointed at two vLLM setups that serve large models at speed on 2x R9700, to argue the
tensor split is achievable.  Both were fetched and read.  The conclusion is not that the split is wrong —
it is that **the prior art does not exercise the case this campaign is about**.

### 27.1 `bkvargyas/dual-r9700-vllm-proxmox`

`docker-compose.yml`: `Qwen/Qwen3.8-27B-FP8` (or `amd/…-Quark-AWQ-MXFP4`), `--tensor-parallel-size=2`,
DFlash2 speculative decoding, 200k context, P2P one-shot all-reduce over an emulated PCIe switch.
Reported: ~160 t/s single-stream decode, 5153 t/s prefill @16k, 922k-token KV.

It is a **dense 27B** (no experts) and it is **fully resident**: FP8 27B ≈ 27 GB, MXFP4 ≈ 14 GB, against
64 GB of VRAM.  `grep -i "moe|expert|cpu.offload"` finds nothing but the AITER MoE toggle.  There is no
host-to-device expert streaming anywhere in it.

### 27.2 `StillDeadcode/vllm-radiance` (the serving image/fork)

The RDNA4 vLLM stack behind that image: pruned ROCm, gfx1201 correctness patches, R4D attention,
`libr4d` (paged attention, fused GDN prefill, P2P all-reduce, skinny bf16 GEMM), and **RDNA4-tuned fused-MoE
Triton configs** (`moe-configs/E=256,N=256,…,fp8_w8a8,block_shape=[128,128].json`).  It explicitly supports
**Qwen3.6-35B-A3B-FP8 (256 experts / top-8) at TP=2** — the same architecture family as this campaign's
model.

It is also **fully resident**: 35B-A3B in FP8 is ~35 GB, against 64 GB.  `grep -i "cpu.offload|offload.gb"`
is empty; the only offload knob mentioned is a host **KV** tier (`--kv-offloading-size`), and the guide
says to omit it.

### 27.3 What this means

* **Every cited setup runs the weights in VRAM.** vLLM's MoE (tensor-parallel *or* expert-parallel) is
  resident too; its answer to "the model does not fit" is **quantize until it does** (FP8/MXFP4), not
  stream experts from host.  There is no vLLM path that H2D-uploads the selected experts per ubatch.
* **The analog of these setups in our delivery is `-ncmoe 0 -sm tensor`** (all experts VRAM-resident, tensor
  split).  We already measure that at **7695 t/s (ub 8192) / 8244 (ub 4096)** prefill — at or above their
  5153 @16k on a different model and quant.  So "tensor split at speed" is not something we are failing to
  reach; we reach it whenever the experts fit.
* **The campaign's `-ncmoe` case is the genuinely unsolved one.** When the experts do *not* fit, the
  industry's answer is a smaller quant or more VRAM; the split's PCIe-bandwidth argument (each card uploads
  1/N instead of the whole tensor) is exactly right and exactly unexplored.
* **Our own data already says the split is viable**: its compute floor is **7037 t/s** vs mirrored's
  **5107**, so halving each device's work is worth ~+38 % *if the transfer is hidden*.  The blocker is the
  exposure (the host sync, §25.3) and the down-projection's fine-grained gather (§26.3), not the split.

**Next experiment (the one this all points at).**  Make the split's upload actually hide:

1. **Kill the host sync** by driving the split upload through `stage_input` (§25.3/§26), so the H2D is issued
   on the copy stream before the scheduler's wait.
2. **Fix the down gather.**  `ffn_down_exps` splits on the innermost dim, so a per-block host gather is
   524288 tiny copies per layer — the 970 t/s regression in §25.6.  The right shape is either a **device-side
   compaction** (1-D H2D the contiguous `size` into the slot, then a `cudaMemcpy2DAsync` D2D or a small
   gather kernel into the simple tensor) or a vectorised host gather.  Note the trade: the whole-range H2D is
   the *mirrored* volume, which may not hide behind the split's (halved) compute — so the gather that reads
   only the device's half is preferable if it can be made cheap.
3. Measure the full grid again; the target is the §25.4 floor (≈7000 t/s), not parity.

## 28. R9V (`github.com/Dyluhn/R9V`, local `~/R9V`): the prior art that DOES tensor-split offloaded experts

`~/R9V` is a full runtime for **Qwen3.8-Flash-Next on 2x R9700** that runs with the experts *not all
fitting*.  It is the case this campaign is about, and it works.  Its mechanism (points at
`runtimes/qwen38-flash-next-gfx1201-v1/retained-mtp4/python/vllm_gguf_plugin/quantization/tiered_experts.py`
and its `docs/config.md`) is:

* **TP-sharded experts.**  The vLLM fork keeps `get_tensor_model_parallel_rank()`-sharded expert masters —
  each rank holds `1/tp_degree` of **every** expert.  That is exactly what our `-sm tensor` split is, and it
  is the load-bearing confirmation the maintainer was after: **the tensor split itself is not the problem.**
* **A measured hot/cold manifest.**  A per-rank `hot_experts_by_layer` list (48 layers x 512 experts,
  `_validate_hot_lists`) is built from routing calibration and copied to the accelerator on load
  (`materialize_hot_expert_cache`).  The README: keeping rank 1's 400 most-used experts per layer resident
  drops its host expert copy from 55.4 to 40.3 GiB.
* **Cold shards live in pinned UVA host memory, not in a copy queue.**  `allocate_tiered_cold_host_empty` /
  `allocate_uva_host_empty`, then `parameter.data = get_accelerator_view_from_cpu_tensor(cold_owner)` and
  `_vllm_is_uva_offloaded = True`: the cold experts' "device" pointer is a **GPU view of host pinned
  memory**, so the kernel reads them in place over PCIe.  There is no per-ubatch H2D copy for a cold expert.
* **A VRAM slot cache on top.**  `QWEN38_TIERED_EXPERT_CACHE_SLOTS` (<=128, 160/192 under admitted
  synchronous LRU), policies `second_touch_rr` (default) or `lru`, optional async fill
  (`QWEN38_TIERED_EXPERT_CACHE_ASYNC`), fill batch 1/2/4.  A second touch promotes a cold expert into a
  resident slot.
* **A staging ring and an I/O queue.**  `docs/config.md`: `io.chunk_mb` (default 16), `io.queue_depth`
  (default 8), `host.pinned_budget` (`auto = min(free-4GB, need)`) — the same shape as our block-06 ring and
  the §23 pinning.
* It also has `PlanStrategy::{Tp, Ep}` and `ExpertPlacement::{Device, HostCompute, HostFetch}` in
  `crates/r9v-ir/src/plan.rs`, i.e. a per-(layer, expert) placement decision rather than a per-tensor one.

### What this means for the campaign

1. **The split is validated by prior art.**  TP-sharded experts + host offload runs at speed.  The
   maintainer's premise holds, and §27's "nobody does this" is corrected: R9V does, on this exact hardware.
2. **But R9V does *not* win by making the upload fast.**  It wins by (a) keeping the hot experts resident and
   (b) reading the cold ones **in place over PCIe (UVA)** instead of copying them into VRAM per ubatch.
   The upload is avoided, not accelerated.  That is orthogonal to our op-offload path, which always copies
   the selected experts.
3. **Our §25/§27 numbers still bound the copy path**: the split's compute floor is 7037 t/s vs mirrored's
   5107, so hiding the transfer is worth ~+38 %.  If we can hide it, the copy path is competitive; if we
   cannot, R9V's answer (a hot cache + UVA cold reads) is the one with evidence behind it.
4. **The direct experiment R9V suggests** for llama.cpp: a **hot-expert cache** (a small per-layer VRAM slot
   pool fed by a routing profile) is the decode lever, and the cold reads could plausibly go through a
   pinned host buffer the same way our §23 pinning already makes the host tensor — CUDA/HIP can map pinned
   host memory into the device address space (`cudaHostAllocMapped`), which llama.cpp's op-offload does not
   currently use.

## 29. Fifteenth probe (2026-09-27): the split upload through `stage_input` — implemented, and an open bug

Instrument: `exp15-staged-split-upload.patch`.  It gives the meta backend a working split staging path and
an override to attribute it, and it is **not yet correct/fast** — this section records exactly where it
stands so the next session does not re-derive it.

**What was built.**

* `stage_gather` (backend iface) now takes `(slot, src, offset, width, stride, n, ev)` and assembles a
  split device's strided slice into the ring slot on the copy stream, in two shapes:
  * **host gather** for a coarse slice (e.g. `ffn_gate/up`: 256 blocks of 294912 bytes) — gather into the
    pinned ring slot, one 1-D H2D;
  * **whole-range H2D + device D2D compaction** for a fine slice (`ffn_down`: `width = 176`,
    `stride = 352`, `n = 524288`) — a per-block host gather there is half a million `memcpy` calls, so H2D
    the contiguous range into a per-device device scratch (`h2d_scratch`, 256 MiB cap) and let
    `cudaMemcpy2DAsync` (`cudaMemcpyDeviceToDevice`, copy stream) do the compaction.
* `GGML_META_GATHER_MODE` forces the shape (0 auto, 1 host, 2 compact) for attribution.

**What is measured.**

* The D2D 2-D compaction is **not** the problem: a standalone probe of the exact down shape
  (`176 x 524288`, spitch 352 -> dpitch 176) runs in **0.47 ms** (~587 GB/s); the gate shape 0.29 ms.
* The split's split-upload shapes are irrelevant to the current result: `GATHER_MODE` 0/1/2 all give
  **~290 t/s** at ub 2048.
* The pruning + `GGML_META_PINHOST` path (no staging) gives **~1497 t/s** at ub 2048; the plain 2-D splice
  **~289**; `GGML_META_NO_2D` (1-D loop) **~22**; mirrored staging **~1395**.

**The open bug.**  The staged split lands at **~290 t/s** — i.e. exactly the plain-2-D number, not the
~1500 the fast host gather achieves in the pruning path.  Since all three `GATHER_MODE` values are the
same, the cost is *not* the transfer shape: it is structural to the staged path itself (the redirect
`simple_tensor->data = chunk.slot`, the `stage_input` hand-off, or the drain), or `stage_gather` is
silently failing and the consumer is reading an unwritten slot.  Next session's first three checks:
1. confirm the staged output is **correct** (`llama-cli` same-seed vs the pinned build) — if it is wrong,
   `stage_gather` is failing and the timing is a red herring;
2. time the drain's `ggml_backend_event_wait` and the `stage_input` issue separately (`GGML_RING_STATS`
   printed nothing for this path);
3. A/B the redirect (point `simple_tensor->data` at the slot vs D2D it back into the real tensor and read
   that) — the mirrored path redirects and is fast, but the mirrored slice is contiguous while the split's
   is compacted, so the child graph's access pattern differs.

The instrument is preserved so the next session starts from the build, not from scratch.

## 30. Sixteenth probe (2026-09-27, later session): §29's bug was a WRONG GUARD — the split now wins at every ub

Follows §29.  The open bug was not in the staged path's mechanics at all: **`ggml_backend_meta_stage_input`
never ran.**  Its final reservation check compared the accumulated per-device sub-block offsets against
the *whole tensor* size instead of one full chunk:

```c
// ggml/src/ggml-backend-meta.cpp, before:
if (entry.chunks.empty() || (!mirrored && offset_j != size)) { return false; }
// after:
if (entry.chunks.empty() || (!mirrored && offset_j != chunk_size_full)) { return false; }
```

`offset_j` is the running sum of each device's `chunk_size_j` (0 for dev 0, `chunk_size_j` for dev 1, ...),
so for an axis-1 split it ends at `chunk_size_full = 589824`, **not** `size = 150994944`.  The guard
therefore returned `false` for every split tensor, silently: the scheduler fell through to the pruning
path and the plain per-device `hipMemcpy2DAsync` splice — which is exactly the ~290 t/s the §29 note took
for a staged-path cost, and exactly why all three `GGML_META_GATHER_MODE` values looked identical (they
were never reached).  `GGML_META_GATHERDBG` (added here) confirms the new path: `calls=480
wait=116.8ms host=1988.9ms issue=2.9ms host_bytes=23040MiB h2d_bytes=51608MiB`.  The splice's own loop
ends with the correct `GGML_ASSERT(offset_j == chunk_size_full)` — the staged path just had the wrong
comparison.

### 30.1 The staged split after the fix (prefill, `-sm tensor -ncmoe 99`, 2× R9700, ub = p, `-r 3`)

| pp t/s | ub 2048 | ub 8192 |
|---|---|---|
| mirrored + staged | 1392-1394 | 5124-5126 |
| **split + staged (fixed)** | **1652-1688** | **5333-5454** |
| previously ("§29") | 288 | — |

Bit-identical greedy output (`sha=359ff4337837` on the 35-char gate; `sha=4b623b02d8c1` on a 200-token
generation over `prompts/prose-rdna-boosts.txt` — same hash for mirrored and split).

### 30.2 Small ub: promote the pinned splice gather (it was a diagnostic)

Above the staging gate (~1542 tokens at this link's 14.5 GB/s) the whole-tensor staged path is used.  Below
it the scheduler prunes to the used experts and uploads through the splice — whose `hipMemcpy2DAsync` is
**5-7× slower than a pinned host gather + one queued 1-D H2D** (and faults from a pageable source, §22).
The gather already existed as the `GGML_META_PINHOST` diagnostic; it is now the default for a compacted
strided upload (`n_copies > 1 && stride_tensor == size`), kill-switch **`GGML_CUDA_SPLICE_GATHER=0`**:

| pp t/s | ub 512 | ub 1024 | ub 1536 |
|---|---|---|---|
| mirrored (default) | 493-504 | 815-826 | 1109-1122 |
| split, splice gather (**new default**) | **592-615** | **936-952** | **1224-1256** |
| split, `GGML_CUDA_SPLICE_GATHER=0` | 85 | 163 | 224 |

### 30.3 The default split beats mirrored at every ub

`GGML_META_SPLIT_COPY=1` (split copies) vs `=0` (r12 mirrored), both with staging on, `-r 3`:

| ub | mirrored (default) | split (default) | delta |
|---|---|---|---|
| 512 | 493.4 | 592.3 | **+20.1 %** |
| 1024 | 816.7 | 936.0 | **+14.6 %** |
| 1536 | 1109.2 | 1224.3 | **+10.4 %** |
| 2048 | 1394.1 | 1667.6 | **+19.6 %** |
| 4096 | 2683.9 | 3147.0 | **+17.3 %** |
| 8192 | 5097.9 | 5363.5 | **+5.2 %** |

Decode is **flat** (the MoE runs on the CPU under `-ncmoe` in both modes, §8.4): `tg64` 24.85 vs 25.49 at
depth 0, 25.58 vs 25.20 at 8192, 25.35 vs 24.88 at 16384.  So this is a pure prefill win.

### 30.4 Verification

* `test-backend-ops -o MUL_MAT_ID` 3/3 backends OK; `-o FLASH_ATTN_EXT` 3/3 OK.
* Same-seed greedy text bit-identical: short gate `359ff4337837`, 200-token prose `4b623b02d8c1`.
* GATHERDBG confirms the gather volume is the expected per-device half (23 GiB host gather / 50 GiB H2D
  across the 4 bench passes ≈ 5.6 / 12.6 GiB per pass per device).

### 30.5 What is left, and what it says about §25.4's floor

The split is now ahead, but the staged path still uploads the **whole** expert tensor (all 256 experts,
half per device) regardless of ub, i.e. it bypasses the used-expert pruning — at ub 512 that is ~10× the
volume the pruning path moves.  The split's §25.4 compute floor is ~7000 t/s, and 5363 @ 8192 is still
short of it, so there is room.  Two follow-ups remain: (a) prune the staged upload to the used experts, and
(b) the §31 residency idea, which reduces the host-resident half instead.  The delivery promotion (a block
amendment carrying the two one-line fixes + the splice-gather default, all with kill-switches) is now
justified on the numbers; it has not been cut yet.

### 30.6 Three GPUs (asked 2026-09-27): correct, and a bigger win at 2048/4096, parity at 8192

3× R9700 (`HIP_VISIBLE_DEVICES=0,1,2`), `-sm tensor -ncmoe 99 -r 3`:

| ub | 3-GPU mirrored | 3-GPU split | delta | 2-GPU split (for reference) |
|---|---|---|---|---|
| 512 | 497.2 | 597.0 | **+20.1 %** | 592 |
| 1024 | 798.9 | 885.6 | **+10.8 %** | 936 |
| 2048 | 1397.3 | 1840.5 | **+31.7 %** | 1668 |
| 4096 | 2666.4 | 3560.9 | **+33.5 %** | 3147 |
| 8192 | 4972.3 | 4692.6 | **-5.6 %** | 5364 |

* **It works: same-seed greedy text is bit-identical on 3 GPUs** (`sha=359ff4337837`).
* **The split is quant-block-bound, so it never becomes 3-way.**  This model has 40 MoE layers
(`block_count=41, n_layer=40`), but the layer count is not the constraint: `-sm tensor` splits **within**
each layer's expert tensor, along the FFN intermediate axis.  `ffn_gate_exps`/`up_exps` are `[2048, 512,
256]` (axis 1 = 512, Q4_K, block 256) and `ffn_down_exps` is `[512, 2048, 256]` (axis 0 = 512, Q5_K, block
256), and `get_split_granularity` gives FFN weights `lcm(blck_size, 128) = 256`; the splitter then snaps
`ne_s*(j+1)/n_devices` down to a multiple of that.  `512 / 256` is **two** blocks, so for any `n_devices >= 2`
the segments are at most `[0, 256, 256]` (3 GPUs) or `[0, 256, 0, 256]` (4 GPUs) — a quant block cannot be
cut in half.  `tc.rotation = il % n_devices` just rotates which device sits idle per layer (`STAGEDBG`:
layer 0 → dev 1,2; layer 1 → dev 0,2; …), balancing load without ever making a third device compute.
**The way to a real 3-way split is a finer-block expert quant**: for Q8_0 (block 32) the granularity is
`lcm(32,128) = 128` and `512/128` gives `[128, 128, 256]`, i.e. all three participate (unevenly).  This is
algorithmic/granularity, not a hardware limit and not specific to this campaign.
* **The 8192 loss is the upload path, not compute or AR.**  With the experts on-device
(`-ncmoe 0`) 3-GPU mirrored/split are identical (7443 vs 7434), and the AR choice does not matter
(hybrid 4684 vs nccl 4692; `internal` 1857).  At 8192 the split's host path is the cost, and the
`GGML_META_GATHER_MODE` auto-heuristic inverts with device count: auto (device-D2D for the fine
`ffn_down` slice) 4694, host gather (`GATHER_MODE=1`) **4905** (parity with mirrored 4972), device-D2D
forced 4326 — whereas at 2 GPUs auto beats host gather at both 2048 (1750 vs 1371) and 8192 (5444 vs
5010).  A device-count-aware heuristic is the obvious follow-up; not yet done.

## 30.7 Default-on: the fast path is the NO-env-var path (2026-09-27, per maintainer request)

The rule is AGENTS.md's default-on policy: a validated win is on unless a kill-switch turns it off, and the
features must **self-select** from `-sm`/`-ncmoe` rather than needing the user to know them.  This session
flipped the one remaining opt-in and confirmed the rest already scope themselves:

| feature | default | kill-switch | where it fires |
|---|---|---|---|
| op-offload H2D staging | **on** (was opt-in) | `GGML_SCHED_STAGE=0` | any host-resident weight upload to a stage-capable backend (meta under `-sm tensor`, plain CUDA under `-sm layer`/1 GPU) |
| split expert copies | **on** | `GGML_META_SPLIT_COPY=0` | only the meta backend's offloaded `MUL_MAT_ID` copy (`-sm tensor`) |
| pinned splice gather | **on** | `GGML_CUDA_SPLICE_GATHER=0` | only a compact strided host->device upload (the split splice) |
| pinned expert source | **on** (r15) | `LLAMA_MMAP_HOST_EXPERTS=0` | only CPU-resident `MUL_MAT_ID` weights (`-ncmoe`) |

The features only *act* on the paths they belong to, so "intelligent selection" is structural, not a
mode switch.  Measured with **no env vars at all** (the real default), `-sm tensor -ncmoe 99`:

| ub | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| **default (no env)** | **612** | **979** | **1742** | **3279** | **5446** |
| was: `SCHED_STAGE=0` | 597 | — | 1561 | — | 3271 |

And staging is a broad win, not a split-only one (it was the opt-in in block 06):

| config, ub 8192 | default | `GGML_SCHED_STAGE=0` | effect |
|---|---|---|---|
| 2 GPU `-sm tensor -ncmoe 99` | 5446 | 3271 | **+66 %** |
| 1 GPU `-ncmoe 99` | 5852 | 3203 | **+83 %** |
| 2 GPU `-sm layer -ncmoe 99` | 4116 | 2744 | **+50 %** |
| 2 GPU `-sm tensor -ncmoe 0` | 7672 | 7663 | neutral (no host weights) |
| dense 4B Q8_0, no offload | 10449 | 10424 | neutral (inert) |

Correctness: pure default and the original r12 behaviour (`SCHED_STAGE=0 SPLIT_COPY=0`) produce
**bit-identical** greedy output on 1/2/3 GPUs and both split modes (`sha=359ff4337837`).  The staging
capability check turns it off silently on a backend that cannot stage (`stage_buffer`/`stage_input` both
null), so other backends are untouched; the adaptive width gate still keeps decode/verify batches
(< ~1542 tokens) off the whole-tensor path.

**The delivery promotion is now fully specified**: block 06's staging enable flips to opt-out, plus the
one-line `stage_input` guard fix, the `GGML_CUDA_SPLICE_GATHER` default, and the split-copy default — all
with the kill-switches above.  Not yet cut into `patches/`.

## 30.8 A genuinely over-VRAM model: Qwen3.8-Flash-Next IQ4_NL (93 GiB) on 2× R9700

Requested 2026-09-27.  `qwen4exp` arch (block 14), 176.9 B params, 93.16 GiB across 9 shards, so the
weights do not fit 64 GiB of VRAM and `-ncmoe` is mandatory.  `-sm tensor -ncmoe 99 -r 3`:

| ub | default (no env) | `SPLIT_COPY=0` | `SCHED_STAGE=0` |
|---|---|---|---|
| 2048 | 517 | 476 | 526 |
| 4096 | 540 | — | 532 |
| 8192 | **1468** | 1452 | **624** |

What it says:

* **The default fast path works on a 93 GiB model with no env vars**, and staging is the dominant lever
  at large ub (**+135 %** at 8192: 624 -> 1468).  The split is +9 % at 2048 and neutral at 8192 here.
* **The sub-8192 numbers are inside this model's noise.**  `GGML_SCHED_STAGE_MIN_TOKENS=8192` and
  `SCHED_STAGE=0` should be identical (both gate staging off at 4096) but measured 574 vs 532 — ~8 %
  run-to-run variance, presumably PLE/page-cache state (see below).  Do not read the 2048/4096 deltas.
* **The staging gate is ub-only, and this model's crossover is higher.**  The adaptive gate
  (`sched_stage_min_tokens`, anchored 1536 tokens at 14.5 GB/s) does not know the expert-tensor size;
  here each expert tensor is **450 MiB** (vs 144 MiB on the 35B), so the pruning-vs-staging crossover
  moves up.  A size-aware gate is a follow-up; not done.
* **The split fires and is well-formed**: `STAGEDBG` shows axis 1, `chunk_full = 921600`, 512 chunks,
  and a **256:384 (2:3)** rotation across the two equal GPUs — the granularity snap (`g = lcm(32,128) =
  128` for IQ4_NL) on this arch's 640-wide split axis, not device imbalance (both cards show 60 MiB used
  idle).  `nr = 1`, so `stage_input`'s `offset_j == chunk_size_full` guard holds for uneven sizes too.
* **No staging self-disable**: the PLE tables (`blk.N.ple_key/value.weight`) are host-resident and are
  `WEIGHTS`-usage host buffers, which `sched_stage_is_host_weight` would stage — but no
  "H2D staging disabled: ... does not fit" fired, so none exceeds the 2 GiB ring budget.  Worth
  re-checking if a future model has a single >2 GiB host weight used as a split input.
* **PLE caveat (maintainer, 2026-09-27):** this model benefits massively from the PLE objects being
  pre-warmed, which `llama-bench` does not do (the shipping `runme` uses `--lazy-mode auto` and the
  managed lazy reader).  So the absolute t/s above is **not** the model's real throughput; only the
  default-vs-opt-out ratios are meaningful.  A server-path measurement with PLE warm is the follow-up.

## 31. NEXT PHASE (maintainer direction, 2026-09-27): partial VRAM expert residency for prefill

**Maintainer's framing:** hold **half of the experts in VRAM**, let the other half migrate from the host —
*for prefill*; the decode-time hot-expert slot cache is the phase after this one.  This is the R9V
direction (§28) minus the routing-profile hot-set tuning: no calibration, just a split of the expert set
between resident and streamed.  It is also exactly what `-ncmoe` already does at **layer** granularity
(`-ncmoe N` keeps N layers' experts host-side), so the first reconnaissance is the residency curve at layer
granularity, then the gap to per-expert residency.

### 31.1 Reconnaissance: layer-granularity residency is a CLIFF, not a slope

`-sm tensor -ncmoe N` on 2× R9700, split default (`GGML_META_SPLIT_COPY=1`, staging on), `-r 3`:

| N host layers | pp2048 | pp8192 |
|---|---|---|
| 0 (all resident) | **7930** | **7718** |
| 8 | 4797 | 7046 |
| 16 | 3338 | 6534 |
| 20 (**half**) | 2916 | 6352 |
| 24 | 2583 | 6148 |
| 32 | 2096 | 5798 |
| 40 (all host) | 1752 | 5455 |

**At ub 2048 moving just 8 of 40 layers to the host costs 40 %**; at ub 8192 the same move costs 9 %.
The half-resident point is 37 % of all-resident at ub 2048 but 82 % at 8192.  Two things this tells the
design:

1. **The cost is not the upload volume.**  At ub 2048 the used-expert pruning makes the 8 host layers move
   only ~350 MiB/pass/device, yet they cost 3.1 s/pass-worth of throughput; at ub 8192 the same 8 layers
   move ~10x the volume for a quarter of the relative cost.  The cliff is the *offload pipeline itself*:
   with any host experts the scheduler's op-offload hands the meta backend one `graph_compute` per op
   (SS12-14: ~3.4 ms each, plus a device synchronize per upload, S25.3), and the graph is now mixed
   host/device.  A per-expert resident set that still routes the non-resident experts through the same
   per-op op-offload would inherit the cliff.
2. **Layer granularity is the wrong shape anyway.**  A host layer uploads its whole expert set and a device
   layer uploads nothing, so (for a fixed fraction) the upload volume is the same as per-expert residency
   -- but the compute placement is all-or-nothing per layer, and the offload-path cost is paid by every
   host layer.  The R9V answer (S28) is per-expert: a VRAM slot cache for the resident experts + the cold
   ones read *in place* over UVA, with **no per-ubatch H2D** and therefore no offload-path cliff.
3. **The prefill bound is the expert touch fraction.**  Prefill uses ~all 256 experts (top-8 x 8192 tokens
   covers the set at ub 8192; even at ub 2048 it is ~25/256 per the S8 measurement), so a resident set can
   only save the upload of the experts it holds -- it cannot make prefill faster than `-ncmoe 0` (7718 /
   7930).  The realistic target is therefore **between the all-host split (5364/1668) and all-resident
   (7718/7930)**, moving along the fraction resident.  The layer curve above is a *floor* on that (it
   pays the offload cliff); a per-expert design should beat it for the same resident bytes.

### 31.2 The design sketch to evaluate next

* **Resident set:** `R` expert slots per (layer, device) in VRAM, filled at load time by a **static** rule
  (e.g. the first `R` experts, or a round-robin) -- no routing calibration, per the maintainer's ask.  A
  routing-aware fill is the *later* refinement (R9V's `hot_experts_by_layer`).
* **Cold set:** host-resident, and the key choice is R9V's: map the pinned host shards into the device
  address space (`hipHostRegister` + `cudaHostGetDevicePointer`, `cudaHostAllocMapped`) so the expert
  kernel reads them in place over PCIe.  That removes the per-ubatch H2D *and* the offload pipeline;
  llama.cpp's op-offload does not use UVA today, and the decode-time cost of in-place PCIe reads is the
  reason moe-cache stages instead -- which is exactly why this belongs to *prefill* (bandwidth-bound, and
  the resident bytes are the model's own weights).
* **First measurement to make:** the pure *volume* bound.  Before any kernel work, price the split's
  staged upload against the resident fraction by forcing a fraction of the experts to be staged and the
  rest skipped (a `GGML_META_RESIDENT_LAYERS`-style diagnostic), and compare with the `-ncmoe` curve
  above.  That says whether per-expert residency is worth a kernel change at all.
* **Decode is a separate phase** (the maintainer's instruction): there the hot set *is* routing-driven and
  a slot cache with prefetch is the proven 2.6x lever (moe-cache, S26.2).  Do not conflate the two.

*(This section is the plan + reconnaissance; it is deliberately unfinished -- the implementation is the
next session.  The numbers above are the anchors to beat.)*

## 32. PROMOTED (2026-09-27, r16): the whole fast path is in the delivery

**Release `v16-84e76d8a2-r16`** (tip `92b14a6131905dc6efcd4500dcf4f1dc5a28531b`, tree
`46a5a43d49c8fa4dfa7a4120805d69c0132b4906`; `validate-set.sh` green, strict 16/16 `git am`).  Folded
into **block 15** (the last block; re-basing the chain around block 06 silently dropped 166 lines of
`fattn-mma-f16.cuh`, so the reliable fold was an amendment of the final block — the natural long-term
home is block 06).  The announcement is GitHub Discussions #54; the full report is
`REPORT-ncmoe-prefill.md`; the raw sweep is `sweep-full.csv` (and the harness `sweep-ncmoe.sh`).

Everything is **default-on and self-selecting**; a stock `llama-server … -sm tensor -ncmoe N` needs no env
vars.  Kill-switches: `GGML_SCHED_STAGE=0` (staging), `GGML_META_SPLIT_COPY=0` (mirrored experts),
`GGML_CUDA_SPLICE_GATHER=0` (plain 2-D splice), `LLAMA_MMAP_HOST_EXPERTS=0` (pageable host experts).

### The headline table — pp8192, Qwen3.6-35B-A3B UD-Q4_K_M, every `-ncmoe` level

| `-ncmoe` | ours 1 GPU | ours 2 GPU `tensor` | ours 2 GPU `layer` | upstream 1 GPU | upstream 2 GPU `layer` | tensor/layer | ours-tensor vs upstr-layer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6450 | 7639 | 6308 | 3882 | 3994 | +21 % | +91 % |
| 4 | 6046 | 6959 | 5955 | 3640 | 3649 | +17 % | +91 % |
| 8 | 6148 | 7014 | 6080 | 3408 | 3446 | +15 % | +104 % |
| 12 | 6106 | 6740 | 6038 | 3237 | 3265 | +12 % | +106 % |
| 16 | 6051 | 6541 | 5986 | 3108 | 3114 | +9 % | +110 % |
| 20 | 6005 | 6324 | 5943 | 2978 | 2974 | +6 % | +113 % |
| 24 | 5964 | 6137 | 5557 | 2845 | 2783 | +10 % | +120 % |
| 28 | 5919 | 5971 | 5121 | 2732 | 2601 | +17 % | +130 % |
| 32 | 5878 | 5832 | 4741 | 2625 | 2448 | +23 % | +138 % |
| 36 | 5767 | 5590 | 4390 | 2537 | 2306 | +27 % | +142 % |
| 40 | 5794 | 5445 | 4107 | 2448 | 2193 | +33 % | +148 % |

(every integer level 0..40 is in `sweep-full.csv`; evaluated at `-r 1`.)

* Offload cost on the delivery is nearly flat (1 GPU 6450 → 5794 over the whole range, −10 %); upstream
  loses 3882 → 2448 (−37 %), and upstream's 2-GPU layer split ends *slower than a single card*.
* `-sm tensor` beats `-sm layer` at every level and the margin *grows* with offload (+21 % → +33 %).
* Same-seed greedy output is bit-identical across the defaults and every opt-out; `MUL_MAT_ID` and
  `FLASH_ATTN_EXT` backend-op tests are green.

### On the PCIe5 x4 caveat

This host gives each R9700 only **PCIe 5.0 ×4** (slot cap).  The win is host-stall removal, so it should
**scale up** with link width — the announcement explicitly invites anyone with ×8/×16 per GPU to
reproduce the sweep.  Until such data arrives, treat the absolute numbers as a **lower bound**.

### Next phase (unchanged)

§31 (partial VRAM expert residency for prefill) and the decode-time routing-driven VRAM expert cache.
