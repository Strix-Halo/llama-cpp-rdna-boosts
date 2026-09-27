# Tensor-split expert parallelism: split the MoE expert weights instead of mirroring them

**Status:** opened 2026-09-27, immediately after the op-offload H2D staging work was promoted into the
delivery as the block-06 r12 amendment (`v16-84e76d8a2-r12`).  **The split itself is now implemented and
propagates correctly; the campaign's framing has changed twice under measurement** — read §0 first, then
§1-7 (the original plan, still the map), then §8-14 (the findings; each supersedes part of what precedes it).

**Not a blocker for anything.**  TODO item 24 (the staging ring's inertness under `-sm tensor`) is
closed: the ring engages there now.  This is a *further* optimisation of that path, and its payoff has
to be measured rather than assumed.

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
disagree, the later one wins**: §16b supersedes §16, §16 supersedes §11, §17 supersedes §12's guess, and
§19 supersedes §17/§18's *location*.  §0 is the distilled state; trust it over any older section.

### Status in one screen

* **The split is implemented and propagates correctly.**  `GGML_META_SPLIT_COPY=0` reproduces the r12
  baseline exactly; `=1` splits the expert weight copies (gate/up on axis 1, down on axis 0) and the meta
  split states come out exactly right (`ffn_moe_gate [MUL_MAT_ID, axis 0]`, `ffn_moe_down [MUL_MAT_ID,
  PARTIAL]`).  §10 has the implementation; §2 the design.
* **`=1` then faults** (non-deterministically) part-way into the first prefill.  **That fault is the only
  thing standing between the campaign and a measurement of the win.**  Everything else below is either
  done or ruled out.
* **Why it is worth finishing:** the mirroring makes every card pay full size.  Reference numbers at
  ub 2048 (`llama-bench`, gfx1201 ×2): `-ncmoe 99` + `-sm tensor` **713 t/s**, `-sm layer` 1432, **1 GPU
  1470**, and with the experts on the device (`-ncmoe 0`) **7945 t/s**.  The split is what closes the gap.

### THE FAULT — the only blocker. Start here.

**Repro (fails in ~30 s at this size, deterministic enough to iterate on):**

```bash
cd ~/llama-r12
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
HIP_VISIBLE_DEVICES=0,1 GGML_META_SPLIT_COPY=1 GGML_SCHED_STAGE=1 \
  ./build-rocm/bin/llama-bench -m "$M" -ncmoe 99 -fa 1 -p 128 -ub 128 -n 1 -b 128 -sm tensor -r 1
# -> "Memory access fault by GPU node-2 ... Page not present or supervisor privilege" (rc=141)
# with GGML_META_SPLIT_COPY=0 the same command completes (~150 t/s at this ub).
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

**THE NEXT EXPERIMENT (cheap, one run each — do this first).**  Force the gate and the up into the **same**
`graph_compute` so the intermediate never crosses a boundary; the delivery's fused gate+up+GLU arm does
exactly that:

```bash
# hypothesis: with no cross-graph_compute split intermediate, the fault disappears
HIP_VISIBLE_DEVICES=0,1 GGML_META_SPLIT_COPY=1 GGML_CUDA_MMB_GLU=1 GGML_SCHED_STAGE=1 \
  ./build-rocm/bin/llama-bench -m "$M" -ncmoe 99 -fa 1 -p 128 -ub 128 -n 1 -b 128 -sm tensor -r 1
```
If it passes, the container generation is confirmed and the fix belongs in the **simple-tensor lifetime**
(keep a split intermediate's simple tensor alive across generations), *not* in the split upload arithmetic.
If it still faults, re-run with `GGML_META_SYNCEACH=1 GGML_META_EXECDBG=1` and bisect the next launch;
`GGML_META_EXECDBG` prints every subgraph's nodes and pointers, so the guilty one is the last printed.

**Already ruled out for this fault (all measured — do not re-derive):** the MMQ `MUL_MAT_ID` kernel args
(every field derives from the simple tensor; `s02 = nb02/ts` is 2048 blocks split vs 4096 mirrored — §17);
buffer bounds (`GGML_META_BUFDBG` "ok" for every expert copy); the MMB layer and the delivery's
routed-compact kernel (`GGML_CUDA_MMB=0` / `MMB_ROUTED=0` / `MMB_GLU=0` / `GGML_CUDA_DISABLE_MMQ_ROUTED=1`
all still fault); the partial-reduce machinery (the faulting graph is `n_subgraphs=1`, so no reduce runs;
`set_tmp_data` is never called); garbage expert ids (`min=0 max=255 out_of_range=0` when readable); and
the ids view's parent pointer (§18/§19: identical in both runs).  `-sm layer` "working" proves nothing —
`GGML_META_SPLITDBG` shows **zero** split expert copies there (layer split never consults the policy).

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
  experiment instrumentation applied in the working tree (`exp10-sync-bisect.patch` == the current diff).
  **Do not use `~/llama.cpp`** (stale: r11 + the pre-fold WIP patch) or `/tmp/canon-fix` (the r11 chain).
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

* `exp1..exp10-*.patch` — cumulative working-tree diffs (`exp10` == the current instrumentation; each
  contains all four touched files: `ggml-backend-meta.cpp`, `ggml-backend.cpp`, `ggml-cuda.cu`,
  `src/llama-model.cpp`).  Apply with `git apply` in `~/llama-r12` if the tree is ever lost.
* `tool-blasprobe.cpp`, `h2dprobe2.cpp`, `mmapprobe.cpp`, `bigprobe.cpp`, `asyncprobe.cpp` — the standalone
  ROCm probes that settled §15/§16 (compile with
  `hipcc -O2 -I/opt/rocm-7.14.1-gfx102X/include -L…/lib --offload-arch=gfx1201 X.cpp -o X -lhipblas -lrocblas`).
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
