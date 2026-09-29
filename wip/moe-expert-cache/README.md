# Decode-side MoE expert caching: high-speed expert decode for models that do not fit

**Status (2026-09-28): Phase 1a is complete, and Phase 1b's UVA cold read is implemented, byte-correct and
measured.  H1 (targeted fusion), H2 (W=1..8 width purity + MTP) and H3 (uniform adaptive arena sizing) are
all done, and the eviction wrong-output bug is root-caused (CUDA graph capture vs the per-token host
takeover decision) and fixed.  The cache is byte-identical to the full-table GPU oracle across the whole
verify band (`none`/`n3`/`n7`).  H2 also turned up, and this campaign fixed, a **delivery-wide
width-impurity**, **promoted and tagged 2026-09-28 as release `v16-84e76d8a2-r18`** (block-13 amendment):
the qwen35moe `ssm_gate_beta` fusion reduced K with a different warp count than the standalone mmvq
launch, so W=1 and W>=2 disagreed; the delivered `-ncmoe` path is now byte-pure with fusions ON
(`none == n1 == n3 == n7`).  **Phase 1b is complete: the fill-vs-cold policy is settled - second-touch
admission (`touch`) with in-place UVA cold reads, both now the defaults - and the cache is +97 % over the
delivered CPU path at a 12 GiB arena (48.95 vs 24.79 t/s), with every gate green (byte-identity,
`none == n3 == n7`, MTP acceptance 0.744 and MTP +17.7 % over plain).  See "### PHASE 1B POLICY RECORD";
the cache itself stays default-OFF until promoted.  The CPU-computes-the-misses comparison arm
(section 2.3) is **built, correct, and a NEGATIVE RESULT** (2026-09-28): it wins only below a ~1.2 GiB
arena and loses -30 % at 8 GiB, so it is not promoted - the mechanism is fixed per-op dispatch overhead,
not CPU compute.  **Phase 1 (single GPU) is now CLOSED: 1a (negative result, parked as a follow-up), 1b,
1c and 1d (fail-soft + `--fit`, PASS) are all done.  **Phase 2 (2 GPUs, `-sm layer`) is DONE** (2026-09-28): the
reported "only 1 GPU active" bug is root-caused and FIXED (the scheduler's offload loop always picked the
lowest-index GPU; a pass-3.5 rebalance now puts each host-weight op on its layer's own device), and the
per-device arena blocker is FIXED too - each table's arena is allocated on its owner device and
`MOE_EXPERT_CACHE_MIB` is now a **per-device** budget.  The 2-GPU path is byte-identical to the 1-GPU path
and to the full-table GPU oracle (`ad30da7b5a3a` at `none`/`n1`/`n3`/`n7`, fusions off, `MIB=8192`/`2048`),
MTP `n3` acceptance is 0.82753, and 2-GPU cache `MIB=8192` is 43.02 t/s vs 37.67 no-cache (+14 %).  See the
CURRENT HANDOVER block below and `phase2-sm-layer-record.md` section 6.  Phase 3 (`-sm tensor`) follows.**  The prefill sibling
(`archive/work/tensor-split-expert-split/`, delivery release `v16-84e76d8a2-r16`) is **closed**: its goal
("prefill wins under `-sm tensor` with host-resident experts") is delivered.  This campaign is the decode
half of the same story.

**Not a blocker for anything.**  An optimisation campaign; nothing in the delivery depends on it.  All of
this is `wip/` and applies only to `~/llama-decode`.

---

## CURRENT HANDOVER (2026-09-28, session 3): **Phase 2 - 2 GPUs, `-sm layer` - DONE (device bug fixed,
per-device arenas landed, 2-GPU path byte-pure)**

The cold-start brief is section 0 below; sections 1 to 3 are the design history, the policy measurement,
and the revised plan.  Read this block first, then jump to whichever section it cites.  **Phase 1
(single GPU) is CLOSED - 1a (negative result, parked), 1b, 1c, 1d all done; do not revisit it.**  **Phase 2
(`-sm layer`, 2 GPUs) is done** - device selection, per-device arenas and per-device `MIB` are all landed
and verified (below).  The next phase is **Phase 3 (`-sm tensor`)**.  Phase 2's full record is
`phase2-sm-layer-record.md` (sections 1-5 = the history, section 6 = the resolution).

**Worktree state (rebased onto r21, 2026-09-28).**  Campaign `~/llama-decode`, branch
`wip-moe-expert-cache`, tip **`ce5c9731e`** = the **r21** delivery tip `feefecfbc` + the 9 wip commits
(the Phase-1a/1b work, the 1d fail-soft, the CPU-split arm, the Phase-2 rebalance and the per-device
arenas/`MOE_EXPERT_CACHE_MIB`).  It was rebased cleanly onto r21 (`git rebase --onto feefecfbc 16977e9d1`),
the net campaign diff is byte-for-byte the old one (9 files, +2193/-16), and it **builds clean** on
ROCm 7.14 / gfx1201 with the campaign smoke green (`MUL_MAT_ID` 929/929; `MOE_EXPERT_CACHE_MIB=2048` ->
28 slots/table, 2041 MiB arena, `tg64` 35.0 t/s on 35B-A3B Q4_K_M).  Build with `cd ~/llama-decode &&
cmake --build build-rocm --target llama-cli llama-bench -j 16` (or a full `BUILD_DIR=build-rocm
EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`).  The full campaign patch is now
**`exp6-moe-expert-cache-phase3.patch`** (2739 lines, verified to apply clean to r21 `feefecfbc`);
the previous `exp5-moe-expert-cache-r21.patch` (2529 lines) is the Phase-2 snapshot;
`exp4-moe-expert-cache-phase2.patch` applies to r19, `exp3-moe-expert-cache-phase1a.patch` is the Phase-1
snapshot, and `phase2-sm-layer-WIP.patch` is the broken allocation-restructuring attempt (**reference
only, do not apply**).  Delivery repo `~/llama-cpp-rdna-boosts` `main` is now at **`v16-84e76d8a2-r21`**
(tagged); the campaign rebase touches nothing in `patches/` or the delivery.

**The bug the maintainer reported ("only 1 GPU active even when 2 are meant to be") is ROOT-CAUSED AND
FIXED.**  `ggml_backend_sched_split_graph` chose the device for an op with a host-resident weight in pass 1
with a first-match-wins loop, so with `-ncmoe` (expert weights in host memory, runnable on ANY GPU) it
always picked the **lowest-index** device.  Measured: **all 120 offloaded MoE ops on CUDA device 0**, device
1 idle, plus two cross-device copies per layer-1 op.  The fix is a **pass-3.5 rebalance** that moves each
host-weight op onto the device that owns its layer, read from the layer's **device-resident** weights (the
op's own data inputs are still `-1` at that point, and pass 4 would only drag them onto whatever pass 1
picked).  Verified: 912 moves over a run, and the hook reports **both** CUDA device 0 and device 1.

**Per-device arenas and per-device `MIB` are landed.**  The 2-GPU abort (a device-1 kernel reading a
device-0 arena pointer) is gone: `table_t.device` is learned on the priming pass, `alloc_all_locked()`
sizes and allocates each table on its own device under a device guard, and `MOE_EXPERT_CACHE_MIB` is now a
**per-device** budget so the second card buys a second arena instead of splitting one budget in half.  A
single-device run is unchanged.  **Gates (Q4_K_M, `-ncmoe 99 -fa 1 -sm layer`, fusions off, 300 tokens):**
1 GPU `MIB=8192`/`MIB=2048`/`SLOTS=128` and 2 GPUs `MIB=8192`/`MIB=2048`/`none`/`n1`/`n3`/`n7` are **all
`ad30da7b5a3a`**, the same hash as the full-table GPU oracle - the layer split, the rebalance and the
per-device arenas change no arithmetic.  `FAIL_ALLOC=1` is still clean.  2-GPU `MIB=2048` h=0.6090 (was
0.4682 with the split budget); 2-GPU cache `MIB=8192` 43.02 t/s vs no-cache 37.67 (+14 %); MTP `n3`
acceptance 0.82753 at 86.12 t/s vs plain 42.50.

### NEXT STEP (the resume pointer)

1. ~~Build the CPU-computes split.~~ **DONE - NEGATIVE RESULT.**  Wins only below a ~1.2-1.5 GiB arena
   (-30 % at 8 GiB, -9 % at 2 GiB, +3 % at 1 GiB) and the maintainer's floor is ~2 GiB, so the paying regime
   is ruled out.  Mechanism: **fixed per-op dispatch overhead** (~8-10 ms/token of ~600 serialised
   CPU-branch dispatches), not CPU compute and not thread-pool wake (proved by `-t 1` losing the same
   -35 % as `-t 8`).  Kept default-OFF as `MOE_EXPERT_CACHE_CPUSPLIT=0`.  **Do not re-tune the small-arena
   regime.**
2. ~~1d fail-soft + `--fit`.~~ **DONE - PASS** ("### 1D RECORD"): the arena is the lowest-priority VRAM
   consumer, so `--fit` need not learn about it; induced all-fail / partial-fail allocations are
   byte-identical with no abort, and a 64 GiB / 1 TiB request clamps to what is free.  **Phase 1 CLOSED.**
3. ~~Per-device arenas (Phase 2, step 3).~~ **DONE (2026-09-28, session 3)** - see
   `phase2-sm-layer-record.md` section 6.  The 2-GPU abort is gone, and the 2-GPU cache hash equals the
   1-GPU cache hash and the full-table GPU oracle (`ad30da7b5a3a`) at `MIB=8192`/`MIB=2048`, fusions off.

   **The lesson, kept:** do **NOT** restructure *when* tables are allocated.  Moving the allocation out of
   `alloc_all_locked()` into a lazy per-table call inside the hook is what REGRESSED the deferred-sizing
   path in the previous session (`MOE_EXPERT_CACHE_MIB` stopped being transparent: `8192` ->
   `ee68cad3a202`, `2048` -> a degraded 806-char output).  The landed fix keeps the one-sweep
   `alloc_all_locked()` and only records `t.device` on the priming pass, so the timing (and the deferred
   path's purity) is byte-for-byte unchanged.  `phase2-sm-layer-WIP.patch` is the broken split attempt -
   **reference only, do not apply**.  The hook now carries a migration backstop for a late-learned owner,
   but it never fires in normal operation.
4. ~~`MOE_EXPERT_CACHE_MIB` should become per device.~~ **DONE (2026-09-28, session 3).**  It is now a
   **per-device** budget: `alloc_all_locked()` groups tables by owner device, clamps against each device's
   own free memory (minus `_RESERVE_MIB`), and gives all of a device's tables the same slot count.  At
   `MIB=2048` on 2 GPUs the all-roles hit rate rose 0.4682 -> **0.6090** for the same MIB.  A single-device
   run is unchanged.  **Phase 2's value is still CAPACITY, not throughput** - for a model that FITS one
   card, forcing `-sm layer` costs throughput (1 GPU 57.64 vs 2 GPU 43.02 t/s at `MIB=8192`), so do not
   frame Phase 2 as a throughput win.
5. **The rebalance pass is a GENERAL fix and a good `upstream/` candidate.**  The delivered prefill MoE
   offload under multi-GPU `-sm layer` has the same lowest-index-wins behaviour, so it is also a candidate
   for a delivery block (block 06 is the general system-operations bucket) - but it needs its own gates on
   the delivered path first, and the maintainer's go-ahead.
6. ~~**NEXT: Phase 3 (`-sm tensor`)**, where the cold path MUST be UVA because the `ffn_down` split
   is `nb[1] = 176` x 524288 chunks per layer, and each device holds a *slice* of every expert (so the
   arena shape and the `device_alias()` seam need a fresh look).~~  **FIRST CUT DONE (2026-09-28, session 4) — see
   "### PHASE 3 RECORD" below.**  The `-sm tensor` cache now runs: the Meta backend delegates the
   scheduler's hook to each simple CUDA backend with a per-device slice descriptor, each device owns a
   compact arena of *its slice* of the resident experts, and the fill is slice-aware (axis-1 contiguous,
   axis-0 2-D).  It is **byte-identical to the full-table GPU oracle**, width-pure `none == n1 == n3 ==
   n7`, MTP acceptance 0.831, and **+48 % over the delivered CPU MoE** at 8 GiB/device (48.7 vs 32.9
   t/s).  The **cold path is deliberately NOT wired for a split table yet** (a host slice is not a
   contiguous per-expert blob), so split tables fill every miss and `touch` degenerates to `always`.
   The UVA cold read with a per-expert stride is the next task.  (There is no `-sm split` mode in this
   tree: NONE/LAYER/ROW/TENSOR; `-sm row` is deprecated upstream and out of scope.)

   **Phase 3 is framed as a GAIN, not a restoration (maintainer, 2026-09-28).**  The `-sm layer` decode
   loss is structural, not a cache defect: whole layers form a **serial pipeline** across devices, so the
   second card adds a cross-device hop per layer but **no decode parallelism** - the cache can only
   recover capacity, never the lost parallel width.  Under `-sm tensor` the second card adds real compute
   **width** (each device computes its slice of every expert, the meta backend's per-device partial
   reduce combines them), so decode should improve outright, with the expert cache adding capacity on top.
   Measure Phase 3 against the **1-GPU** decode number, not against the 2-GPU `-sm layer` one.

### PHASE 3 RECORD (2026-09-28, session 4): `-sm tensor` cache — per-device slice arenas, byte-identical,
+48 % over the delivered CPU MoE

**Status: first cut DONE.**  The cache now works under `-sm tensor` (the end-goal geometry) for the fill
path.  The tensor-split **cold** path (UVA in place) is the remaining task; split tables are marked
`cold_safe = false` and fill every miss, so the `touch` doorkeeper degenerates to `always` (correct, just
higher churn).

**Why it did not work before.**  Under `-sm tensor` the scheduler's split backend is the **Meta** backend,
not a CUDA backend - and the Meta backend's `moe_cache_update` iface was NULL, so the cache was completely
inert on that path (the decode MoE ran on the CPU).

**What was built** (`exp6-moe-expert-cache-phase3.patch`, on top of the Phase-2 tip):
* **A slice descriptor through the iface.**  `moe_cache_update` gained `slice_off`/`split_axis`, and
  `moe_cache_table` gained `host_bytes`/`src_off`/`host_pitch`/`split_axis`: `expert_bytes` is *this
  device's slice*, `host_bytes` is the full host-master expert stride, and an axis-0 slice additionally
  carries the host row pitch for a 2-D fill.
* **A Meta delegator** (`ggml_backend_meta_moe_cache_update`): derives the split state of the scheduler's
  `input_cpy`, then forwards to each simple backend's own hook with that device's simple tensor (the exact
  tensor the CUDA `MUL_MAT_ID` consumer reads) and the running `offset_j` (the same accumulation the
  prefill splice uses: `simple_tensor->nb[axis+1]`).  Returns true only if **every** device took over.
* **Per-device registration keyed by `(host master, device)`**, with a separate consumer alias map and a
  semantic `(layer, role, device)` fallback (a Meta graph rebuild can hand the op a different simple-tensor
  pointer than the hook saw).
* **Layer-uniform slots across devices**: `alloc_all_locked` now takes the min slot count over the devices
  that own a layer, because device 0 and device 1 must agree on the residency or a remap id names
  different experts on each.  (`-sm layer` is unchanged: one device per layer.)
* **Slice-aware fill**: axis-1 (gate/up) is one 1-D `cudaMemcpyAsync`; axis-0 (down) is one
  `cudaMemcpy2DAsync` (`rows = host_bytes/host_pitch`, `row = expert_bytes/rows`).

**The geometry, confirmed on Qwen3.6-35B-A3B Q4_K_M** (2 GPUs, `-ncmoe 99 -sm tensor`, `-v`):

| tensor | host expert | device slice | axis | src_off | pitch |
|---|---:|---:|---:|---:|---:|
| `ffn_gate_exps` / `ffn_up_exps` | 589824 B | 294912 B | 1 (contig.) | 0 / 294912 | 0 |
| `ffn_down_exps` | 720896 B | 360448 B | 0 (strided) | 0 / 176 | 352 |

240 tables register (120 per device), each device sizes 56 slots/table at 2 GiB/device, 224 at 8 GiB, and
both devices report in the hook.

**Gates (all green, 2026-09-28 session 4):**
* **Full-table GPU oracle, fusions OFF.**  `-sm tensor -ncmoe 0` (all experts resident, split, no cache) =
  `bde521b0305e` (910 chars), and `-sm tensor -ncmoe 99 MOE_EXPERT_CACHE_MIB=8192` = **`bde521b0305e`** -
  byte-identical.  So the per-device slice fill and the arena read reproduce the full-table arithmetic.
* **Width purity, fusions ON.**  `none == n1 == n3 == n7 == 15038c19ddc8` (300 tokens, cache 8 GiB).
* **MTP health.**  `draft-mtp n3`, `-n 1000`: acceptance **0.83100** (713/858), mean len 3.49, acc/pos
  `(0.920, 0.815, 0.759)` - comparable to the `-sm layer` 2-GPU 0.82753.
* **1-GPU `-sm layer` regression.**  My keying/lookup refactor still gives `ad30da7b5a3a` (fusions off,
  `MIB=8192`), so the extraction of the consumer lookup did not disturb Phase 1/2.

**Throughput** (Q4_K_M, 2×R9700, `-ncmoe 99 -fa 1 -sm tensor -t 8`, `tg512`):

| config | t/s |
|---|---:|
| delivered CPU MoE (cache inert) | **32.85 ± 0.06** |
| cache 2 GiB/device | 22.9 |
| cache 8 GiB/device | **43-49** (h=0.93) |
| cache 12 GiB/device | 42.8 |

The 8 GiB point is **+31-48 % over the CPU MoE** and byte-identical.

**The ceiling, and why the cache does NOT approach `-ncmoe 0` even at h≈1** (the obvious question).  On
this model `-ncmoe 0` keeps every expert device-resident (split, no cache):

| config (`tg1024`) | fusions ON | fusions OFF |
|---|---:|---:|
| `-ncmoe 0` all-resident split | **96.15 ± 0.91** | **73.08 ± 0.54** |
| cache `MIB=10240` (h=0.995) | 49.26 ± 5.84 | 42.39 ± 5.10 |

So with all experts resident the cache is still **~40-49 % below the equivalent all-resident path**.  The
gap is **not the fills** (h=0.995, `evictions=0`, `fills` already amortised over the 2048-token run) and
not correctness; it is **the cache machinery plus the fusion stand-down**:
1. **H1 stands the decode MoE fusions down** (gate+up+GLU MMQ) because a fused MoE would read the
   redirected `input_cpy` instead of the arena.  The fusions alone are worth **+32 %** on the resident path
   (96.15 vs 73.08).
2. **The MoE still runs through the host-weight op-offload path every token** even when nothing needs
   filling: 120 hook calls per device per token (mutex + LFRU + a per-role `cudaMemcpyAsync` remap upload)
   and 120 consumer shallow-copy re-dispatches, versus a direct device read of a resident table.
The realistic next levers are therefore (a) a **cache-aware fused MoE** that reads the arena/remap (README
option (b), recovers the MoE fusion) and (b) trimming the per-token hook/remap overhead - not more arena.
For a model that **does not fit** (`-ncmoe 0` impossible) the comparison is CPU MoE vs cache, and the
+48 % / 49 vs 33 t/s stands.  **Open:** the cache runs also show run-to-run variance (stddev 5-11 t/s)
absent from the `-ncmoe 0` runs.  Note this model *fits one card*, so the `-sm tensor` cache is slower
than the 1-GPU `-sm layer` cache (57.6 t/s); Phase 3's value is **capacity + compute width for a model
that does not fit**, judged against 1-GPU decode, per the handover.

**Next for Phase 3:** (1) the tensor-split **cold** path - a per-expert-stride UVA / cold read so split
tables get `touch` admission and stop filling every miss (this is what the handover flagged: the axis-0
host slice is strided, so the current contiguous `id - n_res` cold encoding cannot serve it); (2) chase the
variance; (3) re-validate on a model that genuinely needs the split (Qwen3.8-Flash-Next IQ4_NL).

7. ~~Housekeeping: release r19 tag.~~ **SUPERSEDED (2026-09-28):** `v16-84e76d8a2-r21` is the current
   release, committed, tagged and pushed (r19 was never tagged and is now superseded); this campaign
   worktree is rebased onto r21.  No action needed.

### 2-GPU `-sm layer` quick reference (so the next session does not re-derive any of it)

**Harness** (hardware: 3x R9700 gfx1201, 32 GiB each; use `HIP_VISIBLE_DEVICES=0,1`):

```sh
MQ4=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
MQ8=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
# throughput
env HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=16384 ./build-rocm/bin/llama-bench \
  -m $MQ4 -ncmoe 99 -ngl 99 -fa 1 -sm layer -t 8 -p 0 -n 512 -r 3
# purity (the oracle is ad30da7b5a3a; fusions off is the proven-transparent config)
env HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=16384 \
  GGML_CUDA_DISABLE_FUSION=1 GGML_OP_OFFLOAD_MIN_BATCH=0 ./build-rocm/bin/llama-cli \
  -m $MQ4 -ngl 99 -ncmoe 99 -fa 1 -sm layer -t 8 -c 8192 --seed 42 --temp 0 --reasoning off --ignore-eos \
  --single-turn --no-display-prompt -p "$(cat prompts/reasoning.txt)" -n 300
python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py <log>      # hash WITHOUT -v
```

**Traps, all hit this session:**
* **`-v` corrupts the text extractor** (the log grows to ~2 M chars and the extractor slices it, so the
  "hash" is noise - two identical runs gave different hashes).  **Take the hash WITHOUT `-v` and the
  diagnostics WITH `-v`, in separate runs.**  This nearly became a false "the cache is impure" finding.
* **`llama-bench` sets the log threshold to ERROR unless `-v`** (`tools/llama-bench/llama-bench.cpp:2321`),
  and `llama-cli` does the same - so every cache diagnostic is invisible in a normal run.
* **Pin `-c 8192`**: llama-cli's default context is the model's training length (262144) and thrashes one
  card, reporting ~14 t/s instead of ~35 - looks exactly like a cache regression, is not.
* **Never use `GGML_OP_OFFLOAD_MIN_BATCH=0` for a t/s number** - it relaxes the op-offload gate for EVERY
  host-weight op and destroys cache throughput.  It is for the byte-identity A/B only.
* **A gate must be shown able to fail before it is trusted** (`MOE_EXPERT_CACHE_ASSERT_SABOTAGE=1` and
  `MOE_EXPERT_CACHE_FAIL_ALLOC=N` exist for exactly that).
* Two of my own bugs, worth not repeating: a guard written `*node_backend_id <= 0` treats **device 0** as
  "unassigned" (`0` is a valid GPU id; unassigned is `-1`), which skipped exactly the ops the pass was for;
  and attempting the preference in **pass 1** can never work because the op's data inputs are unassigned
  there.

**Env knobs**: `MOE_EXPERT_CACHE_MIB` (**per-device** budget - under `-sm layer` each device owns a
disjoint set of layers, so it is allocated on every device that owns cache tables), `_SLOTS`
(explicit, forces immediate sizing - the pure path this session), `_FAIL_ALLOC` (fail-soft liveness test),
`_RESERVE_MIB` (VRAM held back from the arena, default 1024), `_COLD`, `_ADMIT`, `_TOUCH`, `_VERIFY`,
`_ASSERT`, `_ASSERT_SABOTAGE`, `_CPUSPLIT` (0 = off, the negative-result arm).

**Files**: `phase2-sm-layer-record.md` (the full Phase-2 record, sections 1-5) and
`phase2-sm-layer-WIP.patch` (the BROKEN per-device attempt - **reference only, do not apply**).  Code:
`ggml/src/ggml-backend.cpp` (`ggml_backend_sched_split_graph`, the pass-3.5 rebalance; `moe_name_layer()`),
`ggml/src/ggml-cuda/ggml-cuda.cu` (the adapter diagnostic), and the cache module
`ggml/src/ggml-cuda/moe-expert-cache.{h,cu}` (the per-device work goes here).
4. ~~Housekeeping: release r19 tag.~~ **SUPERSEDED (2026-09-28):** `v16-84e76d8a2-r21` is the current
   release, committed, tagged and pushed (r19 was never tagged and is now superseded); this campaign
   worktree is rebased onto r21.  No action needed.

**Worktree state (rebased onto r21, 2026-09-28):** campaign worktree `~/llama-decode` clean at the
**r21 tip `feefecfbc` + 9 wip commits**, new tip **`ce5c9731e`** (`54a43d886` Phase 1a + H1/H2/H3,
`16992aab7` Phase 1b UVA, `bdc394ec1` Phase 1b policy, `c1d311596`/`570b240c7` the CPU-split arm,
`2e7fcfe32` 1d, `48f2306a0` the Phase-2 rebalance, `2d303b69e` per-device arenas + per-device `MIB`,
`ce5c9731e` Phase 3 slice arenas) + the
`exp2` profiler in `ggml-cpu.c`.  The rebase (`git rebase --onto feefecfbc 16977e9d1`) was clean and the
net campaign diff is byte-for-byte the old one; the build and smoke gates are green (see the Worktree
state note at the top).  **`exp6-moe-expert-cache-phase3.patch`** (2739 lines) is now the full campaign patch
and is verified to apply clean to r21; `exp4-moe-expert-cache-phase2.patch` applies to r19 and
`exp3-moe-expert-cache-phase1a.patch` (1695 lines) is the Phase-1 snapshot.  The old `~/llama-fix`
r18/r19 canonical fork is superseded by the r21 chain.

### State in one screen

Phase 1a (single GPU, per-layer compact VRAM slot cache, LFRU, slot-remap consumer) works, and Phase 1b is
complete: the UVA cold read plus the second-touch admission policy, both default (`COLD=off` /
`ADMIT=always` are the kill-switches).  **H1, H2 and H3 are all done and the eviction wrong-output bug is
fixed, so the cache is byte-identical to the full-table GPU oracle across the whole verify band
(`none`/`n3`/`n7`) with CUDA graphs on.**  **Phase 2 (`-sm layer`, 2 GPUs) is also done** (device rebalance +
per-device arenas + per-device `MIB`), byte-pure across the band.  The current full campaign patch is
**`exp6-moe-expert-cache-phase3.patch`**, applies to **clean r21** (`feefecfbc`).  Nothing in the
delivery or in `patches/` is touched (r21 IS the delivery, tagged 2026-09-28).

Measured (1 GPU, `-ncmoe 99 -fa 1 -sm layer`; cache 8 GiB / 64 slots; `tg1024`, real steady state;
CUDA graphs on):

| config | delivered CPU MoE | cache (H1+H3, fusions on) |
|---|---:|---:|
| Q4_K_M, d0 | 29.3 | **47.0** (+60 %) |
| Q8_0, d0 | 24.4 | **38.3** (+57 %) |
| Q8_0, d16384 | 23.7 | **35.2** (+49 %) |
| Q4_K_M, d16384 | 28.5 | **43.0** (+51 %) |

With the Phase 1b defaults (`touch` + UVA cold) the same Q8_0 d0 point is **42.01** t/s at an 8 GiB arena
and **48.95** at 12 GiB - i.e. **+97 % over the delivered CPU path**; the rows above are the Phase 1a
(fill-every-miss) policy, now the `ADMIT=always` kill-switch.

H1's targeted fusion guard, versus the earlier blanket `GGML_CUDA_DISABLE_FUSION=1` run on Q8_0 d0
`tg64`: 30.7 -> **33.6** (+9.5 %).  All three roles (gate/up/down) are consumed, steady-state
`h` = 0.86 (Q8_0) / 0.94 (Q4_K_M).  Every arena size is now byte-identical to the full-table GPU oracle
(`s8`/`s16`/`s64`/deferred), and `test-backend-ops -o MUL_MAT_ID` is 4/4.  See the baseline caveat below
for why "vs the delivered CPU" overstates the win.

### Build and run

```sh
cd ~/llama-decode && cmake --build build-rocm --target llama-cli llama-bench -j 16
MQ=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
# cache on (8 GiB; -n 1024 so the hit rate is the steady state, not the 64-token transient):
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=8192 \
  ./build-rocm/bin/llama-bench -m "$MQ" -ncmoe 99 -ngl 99 -fa 1 -sm layer -p 0 -n 1024 -r 2
# correctness A/B (GPU kernels identical, expert source differs):
#   off: GGML_OP_OFFLOAD_MIN_BATCH=0 GGML_CUDA_DISABLE_FUSION=1
#   on : MOE_EXPERT_CACHE_MIB=8192 GGML_OP_OFFLOAD_MIN_BATCH=0
# NOTE: GGML_OP_OFFLOAD_MIN_BATCH=0 is for the byte-identity A/B only - it relaxes the op-offload gate
#   for EVERY host-weight op, not just MUL_MAT_ID, and drops cache throughput 38.6 -> 15.2 t/s.
# trustworthy oracle where the model fits one card (Q4_K_M): -ngl 99 with NO -ncmoe
# Phase 1b defaults (in-place cold reads + second-touch admission) need NO extra env:
#   MOE_EXPERT_CACHE_MIB=8192                   # the shipped default policy
#   ... MOE_EXPERT_CACHE_COLD=off               # opt out: fill every miss (Phase 1a)
#   ... MOE_EXPERT_CACHE_ADMIT=always|value     # opt out: the rejected admission rules
#   ... MOE_EXPERT_CACHE_NOEVICT=1              # the static-resident A/B (measured clearly worse)
```

Env knobs (all in `moe-expert-cache.h`): `MOE_EXPERT_CACHE_MIB` (0/unset = inert), `_SLOTS` (explicit
uniform count, also forces immediate sizing), `_PERIOD` (LFRU decay, default 32), `_FILL` (default 1),
`_VERIFY`, `_REPORT`, `_SELFTEST`, `_DEBUG`, `_NOEVICT` (Strata-style never-evict A/B), `_ASSERT`
(map/slot invariant), `_SKIP_ROLE` (bypass one role), `_FORCE_COPY` (debug: stage the arena *and* let the
scheduler copy `input_cpy`, to prove no fusion reads the un-staged table).  Phase 1b knobs, both defaults
with kill-switches: `_COLD` (`uva` by default; `=off` restores fill-every-miss) and `_ADMIT` (`touch` by
default; `=always` / `=value` select the rejected rules), plus `_TOUCH` (the touch threshold, default 2).
`_TABLES` is now **advisory only** (legacy).  VRAM/fail-soft (1d): `_RESERVE_MIB` (default 1024, the VRAM
held back from the arena - the arena sizes itself from what is actually FREE, so `--fit` never needs to
know about it) and `_FAIL_ALLOC` (0 = off; the induced-allocation-failure liveness test for the fail-soft
path, same idea as `_ASSERT_SABOTAGE`).  Key code:
`ggml/src/ggml-cuda/moe-expert-cache.{h,cu}`, the scheduler hook in `ggml/src/ggml-backend.cpp`
(`copy_experts`, iface `moe_cache_update`), the iface field in `ggml/src/ggml-backend-impl.h`, and the
consumer + offload/fusion gates in `ggml/src/ggml-cuda/ggml-cuda.cu` (`ggml_cuda_mul_mat_id`,
`ggml_backend_cuda_device_offload_op`, `ggml_cuda_cache_blocks_fusion` called from `ggml_cuda_try_fuse`).

### The hardening tasks (H1, H2 and H3 all done)

**H1. Targeted fusion. DONE (2026-09-28).**  `ggml_cuda_try_fuse` no longer stands every fusion down
when the cache is on.  A new `ggml_cuda_cache_blocks_fusion(cgraph, i)` returns true only when the cache
is active **and** the node begins a fusion that reads a routed expert table in the cache band: a
`MUL_MAT_ID` with `ne[2] <= MOE_EXPERT_CACHE_MAX_TOK` (the gate+up+GLU triple, the routed pair, and the
qwen4exp weighted-down all start at the routed matmul), or a `GLU` whose next node is a cache-band
`MUL_MAT_ID` (the swiglu to routed-down fold).  Every other fusion (router/topk, GDN, QSA, rope, norms)
returns.  Prefill (`n_tokens > 8`) keeps its MoE fusions, because the cache does not take prefill inputs
over and their `input_cpy` is fully copied.  The band constant `MOE_EXPERT_CACHE_MAX_TOK` (8) lives in the
header and is shared by the hook and the guard.  Result (Q8_0 d0): blanket-off 30.7 -> 33.6 t/s, i.e.
+9.5 %, with byte-identical output.  Option (b) (a cache-aware *fused* MoE that reads the arena/remap, so
even the decode MoE fuses) remains an optional follow-up; (a) is sufficient and loses only the fused-MoE
decode arithmetic, which the cache replaces with the same kernel it uses for the per-op path.

**H3. Budget/slot allocator. DONE (2026-09-28).**  The arena is no longer split by a fixed
`MOE_EXPERT_CACHE_TABLES` hint.  The first decode pass only *registers* every table and runs uncached (the
scheduler copies the full experts); the second pass sizes **every** table to the SAME slot count
`budget / total_expert_bytes` and starts filling.  This adapts to the real table count and to mixed expert
sizes, and it aligns gate/up/down residency (an expert resident for gate is resident for up and down too).
On Q4_K_M the old scheme gave gate/up 121 slots vs down 99; the new one gives all 120 tables **112**, and
steady-state `h` rose 0.79 to 0.83.  `h` is also much higher over 1024 tokens than over 64 (Q8_0 0.86 vs
0.78, Q4_K_M 0.94 vs 0.83), so measure the cache over long generations.  `MOE_EXPERT_CACHE_SLOTS` still
forces an immediate uniform count (the self-test uses it).  A one-token uncached prologue is the cost of
knowing the true table set.

**H2. W=1..8 verify-width purity and MTP.  DONE (2026-09-28) - PASS.**  Full evidence in "### H2 RECORD"
below.  Result: the cache is byte-identical to the GPU no-cache path across the whole band and to the
**full-table GPU oracle** on a model that fits one card (`none`/`n3`/`n7` all `ad30da7b5a3a`), and MTP is
healthy and accelerating.  One caveat: with CUDA fusions ON (which H1 restored) `none != draft-mtp` on
MoE, but that is a **late paraphrastic near-tie**, present on the delivered CPU path too, and a
force-copy A/B proves it is **not** a stale read of the un-staged `input_cpy`.

### H2 RECORD (2026-09-28): width purity, cache transparency, and MTP - PASS

Config: 1 GPU, `-ncmoe 99 -ngl 99 -fa 1 -sm layer -c 8192`, seed 42, temp 0, `prompts/reasoning.txt`
(sha256 `242f5e6f2ba2...`), 300 tokens, CUDA graphs on, cache 8 GiB (`MOE_EXPERT_CACHE_MIB=8192`;
deferred sizing gives 64 slots/table).  **Pin `-c`:** llama-cli's default 262144-token context thrashes a
single card and reports 13.9 t/s instead of ~35, which looks like a cache regression but is not.

**1. Cache transparency, Q8_0, fusions OFF.**  Controlled A/B: only `MOE_EXPERT_CACHE_MIB` differs.

| spec | GPU no-cache (1) | cache (2) |
|---|---|---|
| none | `6b5dfe0de946` | `6b5dfe0de946` |
| n1 / n2 / n3 / n4 / n5 / n6 / n7 | `6b5dfe0de946` | `6b5dfe0de946` |

(1) `GGML_OP_OFFLOAD_MIN_BATCH=0 GGML_CUDA_DISABLE_FUSION=1`   (2) that + `MOE_EXPERT_CACHE_MIB=8192`.
All 16 runs are one hash: the cache changes nothing, and `none == draft-mtp N` for every `N` (TILE/MMVQ
band purity holds with the cache on).

**2. Full-table GPU oracle, Q4_K_M (fits one 32 GiB card), fusions OFF.**  The true reference: the whole
model resident on the GPU, no `-ncmoe`.

| spec | full table (`-ngl 99`, no `-ncmoe`) | cache (`-ncmoe 99` + MIB) |
|---|---|---|
| none | `ad30da7b5a3a` | `ad30da7b5a3a` |
| n3 | `ad30da7b5a3a` | `ad30da7b5a3a` |
| n7 | `ad30da7b5a3a` | `ad30da7b5a3a` |

The cache reproduces the oracle byte-for-byte at W = 1 / 4 / 8, and `none == n3 == n7`.

**3. Is the fusions-ON impurity a stale read?  NO - force-copy A/B.**  With `MOE_EXPERT_CACHE_FORCE_COPY=1`
the hook still stages the arena and remaps the ids but returns false, so the scheduler *also* performs the
full `input_cpy` copy.  If any fusion read the un-staged table, the output would change.  It does not:

| config | none | n3 |
|---|---|---|
| cache, fusions ON (copy skipped) | `430f3ea79621` | `899bd6f1be73` |
| cache, fusions ON, FORCE_COPY (copy done) | `430f3ea79621` | `899bd6f1be73` |

So `ggml_cuda_cache_blocks_fusion` (H1) is **complete** - no decode-band fusion reads the expert table -
and the fusions-ON `none != draft-mtp` is arithmetic drift.  The drift is a **late paraphrastic
near-tie**: the first difference is at ~200-264 tokens, on a rewording ("not the first engineer and not
the last" vs "not first and not last among the engineers"), and the **delivery shows the same** with the
cache inert (CPU fusions-ON `none = 6744006631df` / `n3 = 431bbf3a1605`, first diff ~198 tokens; CPU
fusions-OFF is pure at `899bd6f1be73`).  MoE byte-identity is explicitly not required
(`benchmarks/mtp-adaptive-methodology.md` rule 3).  Consequence: the H1 "+9.5 %, byte-identical output"
holds at short length but not over a long generation - record both.

**4. MTP acceptance and throughput.**  `-lv 4`, `-n 2000` (methodology rule 0 floor), fusions ON:

| config | acceptance | acc per pos | mean len | tg t/s |
|---|---|---|---|---|
| cache `none` | - | - | - | 27.7 |
| cache `n3` | **0.74457** (137/184) | (0.855, 0.742, 0.613) | 3.21 | **34.3** |
| delivered CPU `n3` | 0.77654 (139/179) | (0.883, 0.783, 0.650) | 3.32 | 24.0 |

Healthy: pos-1 well above 0.45 and `draft-mtp` beats plain on the same build (+24 %).  The small delta vs
the CPU path is the different expert arithmetic, not a defect.

**5. Verify-width scaling** (`llama-batched-bench -npp 0 -ntg 64 -npl 1,4,8`, fusions ON):

| B | cache t/s | delivered CPU t/s |
|---|---|---|
| 1 | 30.83 | 24.21 |
| 4 | 56.17 | 32.71 |
| 8 | 70.61 | 35.59 |

The cache scales *better* with batch (expert reads amortize); no verify-width regression.

**6. Fusions-ON caveat + one pre-existing observation.**  With fusions ON the cache's decode MoE stands
down (H1) and uses the per-op kernel; the surviving non-MoE fusions drift the long-run text, as above.
Separately, the **unsupported** no-cache config that forces the decode MoE onto the GPU with fusions ON
(`GGML_OP_OFFLOAD_MIN_BATCH=0`, *no* cache) generates garbage (`/////`), and none of `GGML_PAIR_OFF` /
`GGML_CUDA_DISABLE_MOE_MMQ_FUSION` / `_NORM_ROWS` / `_TOPK_MOE_FUSION` fixes it.  Not the shipped path
(shipped decode offloads the MoE to the CPU) and the cache path avoids it via H1; noted for whoever
revisits the fused-MoE option (b).

**Bottom line: PASS.**  Byte-identical to the oracle across the band, MTP healthy and accelerating,
verify-width scaling good.  The cache is ready for Phase 1b (since implemented and measured - see
"### PHASE 1B RECORD").

### FUSION WIDTH-UNIFORMITY: FOUND AND FIXED (2026-09-28)

Why this was opened: H2 showed `none != draft-mtp` under fusions ON, and `GGML_CUDA_DISABLE_FUSION=1`
makes it pure.  The maintainer's concern is correct - a width-dependent fusion silently confounds every
future coherence check (`plain` vs `draft-mtp` is the delivery's text-purity gate).  A force-copy A/B
(above) had already proved it is **not** the cache reading the wrong bytes; this section answers *which
fusion*.

Method: on the cache path (which stands its own MoE-band fusions down), toggle fusion kill-switches and
test `none == n3` at 300 tokens, seed 42, Q8_0.

| configuration | `none` | `n3` | pure? |
|---|---|---|---|
| reference: `GGML_CUDA_DISABLE_FUSION=1` | `6b5dfe0de946` | `6b5dfe0de946` | **yes** |
| baseline (fusions on) | `430f3ea79621` | `899bd6f1be73` | no |
| MWR off | `ceaa8e6a83ed` | `899bd6f1be73` | no |
| SHEXP off | `899bd6f1be73` | `6b5dfe0de946` | no |
| ROPE_SETROWS off | `29c730c2d1c2` | `0be2094d1eb1` | no |
| all 17 switches (MWR, SHEXP, TOPK_MOE, ROPE_SETROWS, NORM_ROWS, NORM_Q8_1, FUSED_ADDMUL, SSM_CONV_IN, SSM_PRESCAN, MOE_MMQ, MMVQ_DENSE_BAND, WEIGHTED_DOWN, CONV, PAIR_OFF, PAIR_DENSE, HC, GDN_CPY, SNAKE) | `945b57010ff5` | `6b5dfe0de946` | no |

**The culprit is the qwen35moe SSM gate/beta fusion (`ssm_gate_beta`).**  `ggml_cuda_try_fuse` fuses the
two Q8_0 alpha/beta projections plus their softplus/sigmoid gating chain into one kernel, but only when
`alpha_w->src[1]->ne[1] == 1` (**"decode only"**).  So W=1 runs the fused kernel and W>=2 the unfused
chain.  The fused kernel chose `calc_nwarps(GGML_TYPE_Q8_0, 1, ...)` (1 warp on RDNA4) while the
standalone dense mmvq **weight** launch uses `calc_nwarps_weight(...)` - and for Q8_0 with `K < 4096`
(this model's `n_embd` = 2048) that is the wide block (**8 warps**, the 2026-09-12 (18) rule).  Different
warp count = different K-reduction order, so the decode and verify paths disagree by ULPs, which flips a
near-tie after ~200 tokens.

This is why no kill-switch bisect converged: the other switches shift the arithmetic *uniformly* (both
widths), so they never equalise `none` and `n3`; only removing the one `decode only` fusion does.
`GGML_CUDA_DISABLE_SSM_GATE_BETA=1` alone already made it pure (`none == n3 == 899bd6f1be73`).

**The fix** (`ggml/src/ggml-cuda/mmvq.cu`, `ggml_cuda_op_ssm_gate_beta`): pick `nwarps` with
`calc_nwarps_weight(GGML_TYPE_Q8_0, 1, table_id, long_k)`, `long_k = src1->ne[0] >= 4096`, exactly as
the standalone launch does.  The fused W=1 path is then bit-identical to the unfused verify chain, so the
fusion is width-uniform and **stays on**.

**Validation (fusions ON, no kill switches, Q8_0, cache 8 GiB, seed 42, `prompts/reasoning.txt`):**

| spec | before fix | after fix |
|---|---|---|
| `none` | `430f3ea79621` | `899bd6f1be73` |
| `n1` | - | `899bd6f1be73` |
| `n3` | `899bd6f1be73` | `899bd6f1be73` |
| `n7` | - | `899bd6f1be73` |
| `none` @ n=1000 | - | `8f105d1fbf84` |
| `n3` @ n=1000 | - | `8f105d1fbf84` |

`none == n1 == n3 == n7` at 300 tokens, and `none == n3` over a 1000-token run: MoE is **width-pure with
fusions on**.  No regressions: MTP acceptance unchanged (**0.74457**, pos-1 0.855), the fusions-off
transparency/oracle path byte-identical (`6b5dfe0de946`), `test-backend-ops -o MUL_MAT_ID` 2/2, and
`tg1024` within noise (Q8_0 d0 37.99 vs 38.3, d16384 35.54 vs 35.2; Q4_K_M d0 57.02 vs 47.0 - a possible
gain, not yet A/B'd carefully).

**Scope: this is a DELIVERY fix, not a cache fix.**  It lives in `mmvq.cu` (block 13's file) and applies
to every qwen35moe/qwen3.5-style GDN model; it invalidated the methodology's "MoE byte-identity is not
required" exemption (`benchmarks/mtp-adaptive-methodology.md` rule 3) for this path.  **PROMOTED
2026-09-28 as release `v16-84e76d8a2-r18`** (block-13 amendment, canonical tip `135ce8b73`, tree
`df3f6ec94`, `validate-set.sh` green; see `WORKLOG.md` 2026-09-28 r18 and `GREEDY-PURITY.md` §39).  It is
shown above from the working tree because the cache campaign owned the checkout when it was found; the
promoted patch carries the same 9 lines.

The kill switches added during the hunt are kept as diagnostics (default off, no behaviour change):
`GGML_CUDA_DISABLE_MWR`, `_NORM_Q8_1`, `_GDN_CPY`, `_FUSED_ADDMUL`, `_SSM_CONV_IN`, `_SSM_PRESCAN`,
`_SSM_GATE_BETA`, `_L2_NORM_PAIR`, `_ROPE_SETROWS`, `_SNAKE`, plus `GGML_CUDA_FUSE_LOG=1` (logs every
matched fusion with its op and dims at the `ggml_cuda_try_fuse` call site).

### PHASE 1B RECORD (2026-09-28): UVA cold reads - implemented, byte-correct, and measured

**What was built** (`MOE_EXPERT_CACHE_COLD=uva`, default off):
* `table_t.host_dev` - the device-accessible alias of the pinned host slice, obtained once per table with
  `cudaHostGetDevicePointer` (the `ROCm_Host`/`cudaMallocHost` buffers are pinned, so this always resolves;
  the code falls back to the host pointer itself under UVA).
* A **cold id region** in the slot-remap: an id `< slots` is a VRAM slot; `id >= slots` is host expert
  `id - slots`.  The hook encodes a non-resident expert that way instead of filling, so the same
  `src0 + slot*expert_bytes` addressing works for both.
* `moe_cache_get_cold(arena, &cold_base, &n_res)` - looked up **inside `mul_mat_vec_q_moe_launch`** by the
  arena base (`vx`), so no consumer/dispatcher plumbing was needed.
* `mul_mat_vec_q_moe` gained `(cold_base, n_res_cold)` and selects the base per routed id
  (`vx_use = id >= n_res ? cold_base : vx`, `channel_x -= n_res`).  Default `(nullptr, 0)` = the 1a
  resident-only contract, a dead branch.
* Admission: a miss is filled only while a free slot exists or when it beats the eviction victim's
  decaying value; otherwise it is served cold.  `MOE_EXPERT_CACHE_NOEVICT=1` + `COLD=uva` gives the
  fill-once-then-cold (static resident set) variant.

**Correctness: byte-identical to the fill path.**  `uva == fill == 6b5dfe0de946` for `none`, `n3` and `n7`
(cache 8 GiB, fusions off, `prompts/reasoning.txt`, 300 tokens, seed 42): reading cold experts from the
pinned host alias gives exactly the bytes the H2D fill would have put in the arena.

**Measured (Q8_0, 1 GPU, `-ncmoe 99 -ngl 99 -fa 1 -sm layer`, `tg1024`, fusions on):**

| policy | arena | h | cold reaches | t/s |
|---|---:|---:|---:|---:|
| 1a fill + LFRU evict | 8160 MiB | 0.8605 | 0 | **37.96** |
| 1b UVA, static (no evict) | 8160 MiB | 0.7543 | 475341 (24 %) | 30.48 |
| 1a fill + evict | 4080 MiB | 0.7100 | 0 | 26.49 |
| 1a fill + evict | 3060 MiB | 0.6507 | 0 | 23.64 |
| 1a fill + evict | 2040 MiB | 0.5684 | 0 | 20.56 |

**Reading**: at **equal memory** the fill+evict policy wins decisively (37.96 vs 30.48, +25 %, and reaches a
higher h); at **equal h** (~0.75) they are break-even (1a ~29.8 t/s interpolated vs 1b 30.48).  So the H2D
fill is **not** the bottleneck on one GPU - the bulk copy's higher effective bandwidth plus the higher h an
eviction policy reaches beat scattered per-access PCIe reads.  This confirms section 2.3's *direction* (the
host-DRAM/bulk path is competitive or better at our hit rates); the UVA mechanism's real value is the
**`-sm tensor` geometry** (section 2.3: a per-device fill there is hundreds of thousands of tiny copies, so
zero-copy in-place reads are the only sane cold path), not a single-GPU speedup.

**Two traps found while measuring:** (1) `GGML_OP_OFFLOAD_MIN_BATCH=0` - the env the H2 correctness A/B
used - **cripples the cache path** for throughput (38.6 -> 15.2 t/s) because it relaxes the op-offload
gate for *every* host-weight op, not just `MUL_MAT_ID`; it is fine for the byte-identity A/B but must not
be used for a t/s number.  (2) The cold admission's value test degenerates under the LFRU decay (a
decayed victim often has count 0, so any incoming beats it) - a real admission policy needs a stronger
predicate (second-touch, or a sampled-frequency victim comparison).

### PHASE 1B POLICY RECORD (2026-09-28): the fill-vs-cold policy is SETTLED - second-touch admission

**The answer: `MOE_EXPERT_CACHE_ADMIT=touch` is the default, with the UVA cold transport also default.**

**The cost model first, because it is what decides this** (Q8_0, 1 GPU, `tg1024`, both transports linear
in the miss fraction, so the whole question is "which curve, and which `h`"):

| transport | fit (ms/token) | all-resident bound | effective PCIe |
|---|---|---:|---:|
| fill + LFRU evict | `ms = 15.94 + 75.6 (1-h)` | 62.7 t/s | 13.5 GB/s |
| UVA read in place | `ms = 14.80 + 72.9 (1-h)` | 67.6 t/s | 14.0 GB/s |

UVA is ~5 % faster **at equal `h`**, but the fill/evict policy reaches a given `h` with ~35-40 % less
memory (`h=0.75`: 4.9 GiB vs 8.1 GiB) because a filled expert is not re-transferred on every use.  So
neither transport wins outright: the transport is worth ~5 %, and `h` is worth everything.

**The policy therefore has to buy `h`, and the doorkeeper does.**  Three admission rules at a fixed arena
(`COLD=uva`, so a rejected miss is served in place), `tg1024`:

| arena | `always` (fill every miss) | `value` | **`touch`** (admit from the 2nd use) |
|---:|---:|---:|---:|
| 8192 MiB | 38.56 (h 0.8657) | 37.63 (h 0.8568) | **41.49 (h 0.8810)** |
| 4096 MiB | 26.57 (h 0.7111) | 26.75 (h 0.7146) | **30.83 (h 0.7607)** |
| 2048 MiB | 20.71 (h 0.5737) | 21.30 (h 0.5736) | **23.29 (h 0.6142)** |

`touch` wins at **every** arena - and it wins `h` too, which is the point.  The mechanism is slot churn:
at 4096 MiB `always` performs **278469 evictions** (a one-shot expert repeatedly evicting a proven one) for
`h=0.7128`, while `touch` performs **15840** and reaches `h=0.7630`; its 213306 first-touch misses are
served cold and cost exactly what the model above predicts (30.83 measured vs 31.0 predicted).  The
doorkeeper makes the arena hold *proven* experts, so it raises `h` **and** cuts transferred bytes.

`value` (the rule the earlier session guessed at) is the loser: comparing the incoming's decaying demand
against the victim's `count` is almost always true, so it degenerates to `always` (849 rejections at
4096 MiB vs `touch`'s 434595).  `touch` needs a per-expert decaying **ghost** counter (touches while
non-resident, reset on admission, halved with the same period as `count`); `MOE_EXPERT_CACHE_TOUCH=N`
moves the threshold - `N=2` is optimal anywhere the arena is not tiny, `N=3` is ~3 % better only at a
2 GiB arena.

**New defaults end to end** (`tg1024`, only `MOE_EXPERT_CACHE_MIB` set; `COLD=uva` + `ADMIT=touch` are now
the defaults, with `COLD=off` / `ADMIT=always` as the kill-switches):

| arena | delivered CPU | old default (fill) | **new default (touch + UVA)** |
|---:|---:|---:|---:|
| 2048 MiB | 24.8 | 20.71 | **23.44** |
| 4096 MiB | 24.8 | 26.57 | **30.62** |
| 8192 MiB | 24.8 | 38.44 | **42.01** |
| 12288 MiB | 24.8 | 47.32 | **48.95** |

At a 12 GiB arena that is **+97 % over the delivered CPU path** (48.95 vs 24.79).  The win is largest
where the arena is small (2048 MiB: 23.44 vs 20.71) because that is where the churn `touch` removes was
the whole cost; at 12 GiB little churn is left to remove, so the residual is the UVA transport's ~5 %.

**Gates (all green, 2026-09-28):** fusions-off byte-identity `uva+touch == 6b5dfe0de946` at 8 GiB **and**
at a 1 GiB arena (heavy cold path); the delivered `-ncmoe` path untouched (`431bbf3a1605`); the cache
still width-pure with fusions on (`none == n3 == n7 == 899bd6f1be73`); MTP acceptance **0.74394**, acc per
pos `(0.876, 0.751, 0.603)` (pos-1 0.876 > the 0.45 floor) and **MTP `n3` 35.69 t/s vs plain 30.32
t/s (+17.7 %)** at `-n 2000`.

**Decision.**  `touch` + UVA cold is the Phase-1b policy: it both raises `h` and uses the cheaper
transport, and it is the shape `-sm tensor` needs (the cold path is already an in-place read from a
per-device pinned slice).  `always`/`value` and `COLD=off` stay as kill-switches.  The **static** resident
set with UVA cold (`NOEVICT`) is clearly worse (h 0.754 vs 0.881 at 8 GiB), so Strata's no-evict design
stays rejected.

**Still open (not this piece):** the **CPU-computes-the-misses** arm of section 2.3 (its own numbers say a
CPU cold set at ~40 GB/s can hide under the GPU's resident compute; it needs a split-by-router-index
dispatch, and the delivered CPU MoE is thread-starved to ~2/16 cores, so it must be pinned to a fixed
thread count to be a fair comparison), and Phase 3 (`-sm tensor`).

**Also observed (a delivery note, not a cache result):** the delivered `-ncmoe` MTP acceptance is now
**0.85757** (the H2 record measured 0.77654 pre-r18).  The r18 width-uniformity fix is the only change on
that path, so it appears to have raised the delivered acceptance by ~+0.08 on top of making W=1..8 agree.

**Status: Phase 1b DONE (mechanism + policy), gates green.**

### DELIVERY BASELINE CHANGE (2026-09-28, r19): the CPU MoE is no longer thread-starved

The delivered `-ncmoe` CPU MoE decode used to run on ~2 cores: r17's tiny-CPU-graph heuristic serialised
every offloaded-MoE decode graph on one thread.  Release **`v16-84e76d8a2-r19`** (block-06 amendment;
`WORKLOG.md` 2026-09-28 r19) replaces that with a capped multi-threaded rule
(`max(1, hardware_concurrency()/2)`, overridable with `GGML_CPU_MOE_OFFLOAD_THREADS`), because the r17
measurement behind the heuristic was confounded by this host pinning its GPU IRQs to cores 13-15.  **The
CPU baseline this campaign measures against therefore changed materially:**

| model, `-ncmoe 99 -t 16` | before (r18) | after (r19) |
|---|---:|---:|
| Qwen3.6-35B-A3B Q4_K_M d0 | 29.11 | **38.01** |
| Qwen3.6-35B-A3B Q4_K_M d16384 | 28.11 | **37.03** |
| Qwen3.6-35B-A3B Q8_0 d0 | 24.80 | **29.44** |
| gemma-4-26B-A4B Q4_K_XL d0 | 21.99 | **37.44** |

The per-reach CPU cost is now ~27 us (Q4_K_M, 960 reaches/token), against ~79 us per cold reach for a UVA
PCIe read (1.0625 MiB at the measured ~13.5 GB/s).  **The CPU is therefore ~2.9x cheaper per cold expert
than the PCIe transfer** - which is the ratio section 2.3 uses to argue the CPU-computes-the-misses arm
should win, and it is now measured rather than assumed.  (The worktree was rebased onto r19 on 2026-09-28;
the r19 re-baseline is below.)

### CPU-COMPUTES-THE-MISSES ARM: DESIGN + TEST PLAN (2026-09-28, pinned before implementation)

**Status: design pinned, implementation NOT started.**  This section is the go-forward spec.

#### The payoff is smaller than section 2.3 assumed (measured on r19)

Fit `tg1024` (d0, `-t 16`) for both arms; the ``1-h`` slope is the per-token cost of the misses, and the
CPU's slope is its per-reach cost times 960 reaches:

| model | UVA PCIe slope (ms / unit of `1-h`) | CPU slope | CPU advantage |
|---|---:|---:|---:|
| Qwen3.6-35B-A3B Q4_K_M | 40 | 26.3 | 1.5x |
| Qwen3.6-35B-A3B Q8_0 | 72.9 | 33.2 | 2.2x |

Serialised (llama.cpp runs splits one at a time, so the CPU branch's time *adds*):

| `h` | Q8_0 | Q4_K_M |
|---|---:|---:|
| 0.95 (8 GiB) | +12 % | +4 % |
| 0.86 (4 GiB) | +29 % | +12 % |

Section 2.3's 3x advantage only holds for the large Q8_0 expert slices; for Q4_K_M's half-size experts
PCIe is already cheap.  **The real prize is overlap** - cached runs leave ~14 of 16 cores idle (measured:
cache 1.90 cores busy vs 8.94 for the delivered CPU MoE), and concurrent execution would give +17 %/+40 %
instead of +12 %/+29 %.  That is a scheduler change, not this arm, and it is the higher-leverage of the two.

#### Design: no new ops are needed

Because the MoE's final combine is a **sum over the `j` axis** and `sum_rows` is order-independent, the two
branches do not need a concat and do not need a permuted id axis:

* `w_gpu = get_rows(probs, selected_experts) * gpu_mask`  (reuses the existing weights op, one masked mul)
* CPU branch: `ids_cpu` (host-written `[C, t]` i32, the cold experts first, padded) ->
  `mm_id(gate_up_exps, cur, ids_cpu)` -> GLU -> `mm_id(down_exps, ..., ids_cpu)` ->
  `w_cpu = get_rows(probs, ids_cpu) * cpu_mask` -> `sum_rows(mul(...))`
* `out = add(out_gpu, out_cpu)`
* `gpu_mask`, `cpu_mask` and `ids_cpu` are host-written graph inputs, exactly like the campaign's existing
  remap write.  Tokens with fewer than `C` cold experts are handled by `cpu_mask`, **not** by a zero expert.
* The `weights = get_rows(probs, selected_experts)` derivation (llama-graph.cpp) is why no weight plumbing
  is needed: the weights are derived from the ids, so each branch can derive its own from its own ids.

The hook owns the partition.  Any expert the CPU covers is masked out of the GPU branch (its arena id
becomes the zero slot, so the GPU contributes 0 for it).  **Any cold expert the CPU does NOT cover
(`c_t > C`) must fall back to the GPU's UVA alias - never be dropped.**

#### The structural invariant is the primary gate

This mode's failure mode is not a crash and not a bad hash: it is an expert counted **twice** or **dropped**,
which leaves the output plausible and rots it slowly.  So the primary gate is structural.  With
`MOE_CPU_SPLIT_ASSERT=1` (mirroring the campaign's `MOE_EXPERT_CACHE_ASSERT`), verify for every (token, op)
that the multiset of experts summed across the two branches equals the routed set `{id_0..id_{k-1}}`
**exactly once each**, and that the `gpu_mask` + `cpu_mask` covering is exactly 1 per routed slot.

#### Test plan

1. **Byte-identical endpoints.**  `C=0` must equal the GPU/UVA path and `C=k` must equal the delivered
   `-ncmoe` CPU path, byte for byte.  These two anchor the plumbing (ids, masks, weights, adds) exactly,
   and they are the only bit-exact assertions the mode admits.
2. **Structural assert** over a long run (>= 2000 tokens) so every variable-`c_t` path is exercised,
   including `c_t > C` (the fallback to the UVA alias).
3. **Perplexity within noise** against the pure paths on a fixed text (the campaign's established quality
   metric; the QSA work used the same ratio form because a pure hash cannot see a width-uniform defect).
4. **Long-horizon coherence, not a hash.**  The mode mixes CPU and GPU arithmetic for the same layer *by
   design*, so same-seed hashes are meaningless here.  Gate on a long generation (>= 4000 tokens, and at
   d16384 as well) checked for degeneration - repetition collapse, topic loss, reasoning breakdown - plus
   a checkable-answer task (recall/code) to catch a slow quality slide.
5. **No NaN/Inf** in the logits across the `C` sweep: a dropped expert will not NaN, but a mis-masked one
   feeds NaNs in from the padding.

#### Purity contract (state this prominently wherever the mode is documented)

**This mode mixes CPU and GPU arithmetic for the same layer and is therefore NOT bit-identical to any other
path.  The delivered `431bbf3a1605` gate does not apply to it and its failure must not be reported as a
regression.**  What is required instead is (1) the structural invariant above, (2) no drop / no
double-count, (3) coherence and perplexity at parity with the pure paths.  The mode is opt-in (`-ncmoe` plus
an env gate) and off by default.

### 1D RECORD (2026-09-28): fail-soft + `--fit` - PASS

**The `--fit` decision: the arena is the lowest-priority VRAM consumer, so `--fit` does not need to learn
about it.**  `alloc_all_locked` now clamps its budget to `cudaMemGetInfo()` at sizing time - i.e. after
`--fit`, the compute-graph reserve and the KV cache have all taken theirs - minus
`MOE_EXPERT_CACHE_RESERVE_MIB` (default 1024 MiB, for later growth: a bigger ubatch, more context, a draft
pipeline).  Consequences, deliberately chosen:

* The arena can never over-commit, so it cannot starve the compute reserve or the KV cache - which is the
  whole of the issue-#33 trap that the r3 staging arena fell into.
* A user who asks for more than is free **gets less, warned** - not an abort and not a silent steal.  This
  is the same policy the delivery already uses for the r3/r16 staging arena ("the arena is a *speed*
  buffer, so a failed cudaMalloc returns null and the launcher reads the raw cache natively instead of
  aborting").
* `--fit` and `llama_get_memory_breakdown()` stay unaware of the cache, which is fine **because** of the
  above: the fitter sizes for what it can see, and the arena yields.  The residual a user must know is that
  `MOE_EXPERT_CACHE_MIB` is a *cap*, not a reservation - it does not reduce the context `--fit` will choose.

**Evidence (Q4_K_M, 1 GPU, `-ncmoe 99 -fa 1 -sm layer -t 8 -c 8192`, seed 42, `prompts/reasoning.txt`,
300 tokens, fusions ON, no `-v`).**  Every run is `rc=0`; the cache is transparent, so the correct
invariant for all of them is *byte-identical output*:

| config | rc | hash |
|---|---:|---|
| `MOE_EXPERT_CACHE_MIB=8192` (working) | 0 | `883011516483` |
| `... FAIL_ALLOC=1` (all 120 arenas induced-fail) | 0 | `883011516483` |
| `... FAIL_ALLOC=7` (18 of 120 fail) | 0 | `883011516483` |
| `MOE_EXPERT_CACHE_MIB=65536` (64 GiB on a 32 GiB card) | 0 | `883011516483` |
| `MOE_EXPERT_CACHE_MIB=1048576` (1 TiB) | 0 | `883011516483` |
| fusions OFF, working / all-fail / partial-fail | 0 | `ad30da7b5a3a` (all three) |

So an induced arena failure changes **nothing** in the output, warns once per affected table, disables the
cache for just those tables, and never aborts.  The fusions-OFF trio also reproduces the H2-recorded oracle
hash exactly, which ties this test to the earlier transparency evidence.

The clamp is real, not a no-op (from the `-v` run - note the hash is meaningless under `-v`, see the trap
below, but the counters and log lines are valid):

```
MOE_EXPERT_CACHE_MIB=65536 MiB exceeds the 28804 MiB free on the device
  (free=29828 MiB - 1024 MiB reserve); clamping to 28804 MiB
sized 395 slots/table ... (budget 65536.0 MiB, arena 18662.0 MiB)     # vs 112 slots at 8192 MiB
arena alloc failed for layer=0 role=blk.0.ffn_gate_exps.weight (112 slots x 589824 B)
  [induced by MOE_EXPERT_CACHE_FAIL_ALLOC]; cache disabled for this table     # FAIL_ALLOC sample
```

**Second trap, same family as the `-v` one: `-v` corrupts the text extractor.**  With `-v` the log grows
to ~2 M chars of verbose output and `scripts/extract-generated.py` slices it, so the resulting "hash" is
harness noise (`3a2c29444aaf` vs `77879bfe6772` for two *identical* runs, and `230084e44646` for the
delivered path that is `b360afe820c8` without `-v`).  **Always take the hash WITHOUT `-v` and the
diagnostics WITH `-v`, in separate runs.**  This nearly became a false "the cache is impure" finding.

**Minor reporting TODO (not a correctness issue):** at the clamped budget the exit report prints
`arena 18662.0 MiB` where `slots x sum(expert_bytes)` would suggest ~28.8 GiB, so `g_arena_bytes`
under-reports in that configuration (it over-counts nothing and every allocation succeeded, so the real
usage is somewhere in between).  Worth reconciling before the number is quoted, because the clamp makes
"how much did I actually get?" a question users will ask.

Env: `MOE_EXPERT_CACHE_RESERVE_MIB=N` (default 1024; the VRAM held back from the arena) and
`MOE_EXPERT_CACHE_FAIL_ALLOC=N` (0 = off; 1 = fail every table's arena alloc; N>1 = fail every Nth table -
the fail-soft liveness test, the same idea as `..._ASSERT_SABOTAGE`).

### CPU-COMPUTES-THE-MISSES ARM: RESULT - **NEGATIVE above a ~1.2 GiB arena** (2026-09-28)

**Verdict: the arm is built, correct and safe, and it does not pay off in any arena size that matters.**
The maintainer's floor is ~2 GiB (below that the model is too large for the card in the first place), and
the arm only wins BELOW that floor.  **Do not promote it, and do not spend more time on the small-arena
regime.**  The code stays in the campaign worktree as an env-gated, default-OFF experiment.

#### Economics (Q4_K_M, 1 GPU, `-ncmoe 99 -fa 1 -sm layer -t 8`, `tg1024`, r2)

| arena | `h` | no split | C=2 | C=4 | C=6 |
|---|---:|---:|---:|---:|---:|
| 8192 MiB | .95 | **58.27** | 40.57 | 36.96 | - |
| 2048 MiB | .73 | **39.10** | 36.04 | 35.53 | - |
| 1024 MiB | .55 | 31.17 | 30.68 | **32.00** | 29.38 |
| 512 MiB | .35 | 26.37 | - | **28.55** (+8.3 %) | - |

So the crossover is ~1.2-1.5 GiB: **-30 % at 8 GiB, -9 % at 2 GiB, +3 % at 1 GiB, +8 % at 0.5 GiB.**
Depth behaves the same (-28 % at d16384: 50.37 -> 36.30).  `C=4` is the sweet spot; `C=6` already loses.
Every one of those wins is inside a regime the maintainer has rejected.

#### The mechanism: fixed per-op dispatch overhead, NOT CPU compute and NOT thread-pool wake

This is the finding worth keeping.  A thread sweep isolates it:

| | no split | C=2 | delta |
|---|---:|---:|---:|
| `-t 1` | 56.39 | 36.38 | **-35 %** |
| `-t 2` | 56.57 | 37.06 | -35 % |
| `-t 8` | 56.32 | 39.31 | **-30 %** |

The loss is essentially **identical at one CPU thread and at eight**.  If the CPU branch's compute or its
thread-pool wake were the cost, `-t 1` would have been dramatically better.  It is not, so the cost is
structural: the interleaved CPU branch is SERIALISED with the GPU splits, and it adds ~15 ops x 40 layers =
~600 scheduler dispatches plus ~3 cross-backend copies per layer per token.  That is ~8-10 ms/token of
fixed overhead, which is the whole loss.

The CPU's own expert compute really is as cheap as the model said: the `C=4 -> C=6` marginal at 1024 MiB
costs (1/29.38 - 1/32.00) s over 80 extra experts = **~35 us per expert**, close to the all-CPU path's
27 us.  The problem is that the fixed overhead is an order of magnitude larger than the compute it enables.

**What would actually change the economics**, i.e. what the next person should NOT re-derive:
1. **Genuine CPU/GPU overlap.** The interface is a pipeline and llama.cpp runs splits one at a time; the
   fixed overhead above is the serialisation made visible.  This was always the higher-leverage change and
   it would also help the delivered `-ncmoe` path (cached runs leave ~14 of 16 cores idle).
2. **Far fewer CPU-branch ops** - one fused CPU op per layer instead of ~15.  Only 6 of the 8 cold experts
   per token are ever useful at a 2 GiB arena, so the arm's ceiling at any arena size a user would choose
   is small.

#### Correctness: PASS, and the gates are proven live

The mode is correct - the design and its invariant are sound, which is why the negative result is about
ECONOMICS only:

* **The structural invariant is live and passes.** With `MOE_EXPERT_CACHE_ASSERT_SABOTAGE=1` (injects a
  duplicate) the checker reported **880 failures**; with it off, **0 failures** with 4 `INVARIANT ok`
  summaries (one per 256 checks).  A gate that never fires proves nothing, so this liveness test is part
  of the gate.
* **d0 coherence, 2500 tokens, C=2, invariant on: fully coherent.** Still working the same scheduling
  puzzle, re-evaluating cases and checking constraints at the end - no repetition collapse, no topic loss,
  no degeneration.  (Log `/tmp/coh-c2.log`; hash `07b3a81a47d5`.)
* **d16384, 256 tokens, invariant on: 240 `ok` summaries, 0 failures.**
* One real bug was found and fixed on the way: `ggml_div(cpu_w_3d, moe_w_sum_2d)` tripped
  `GGML_ASSERT(ggml_can_repeat(b, a))`; the CPU branch now mirrors the GPU branch's proven 2-D norm form
  (`reshape_2d` -> `div` -> `reshape_3d`).
* Another real bug, and the one that made the first measurements meaningless: `remap_dev` was allocated
  LAZILY inside the hook, so when a layer's first role ran, its sibling roles still had a null remap ->
  the layer-wide readiness check failed -> the gate turned the CPU branch off -> it computed and
  contributed **zero**, i.e. pure overhead that looked exactly like a legitimate -2x split.  It is now
  allocated at sizing time, for every table at once.

#### Two traps for the next person

* **`llama-bench` sets the log threshold to ERROR unless `-v`** (`tools/llama-bench/llama-bench.cpp:2321`),
  and `llama-cli` suppresses INFO/WARN the same way.  Every cache and CPU-split diagnostic is therefore
  INVISIBLE in a normal run - including `alloc_all_locked: sized N slots/table`, the `CPU SPLIT diag` line
  and the `CPU split C=N: experts handed to the CPU=...` counters.  Always add `-v` when reading cache
  behaviour, and prefer `GGML_LOG_WARN` to `GGML_LOG_INFO` for one-shot diagnostics.
* **Verify a gate can fail before trusting it.** The invariant's liveness test (`..._ASSERT_SABOTAGE=1`) is
  what turned "0 failures" from a hopeful silence into evidence; the same instinct is what caught the
  lazy-`remap_dev` bug, because a WARN was added to both the graph and the hook side.

Env: `MOE_EXPERT_CACHE_CPUSPLIT=C` (default 0 = off; the graph side reads the same variable name),
`MOE_EXPERT_CACHE_ASSERT=1` (the structural invariant), `MOE_EXPERT_CACHE_ASSERT_SABOTAGE=1` (gate
liveness self-test).

### EVICTION BUG: ROOT-CAUSED AND FIXED (2026-09-28)

**Symptom.**  With a small arena (<= 16 slots) the cache produced deterministically wrong output while a
large arena (>= 32 slots) was exact.  `MOE_EXPERT_CACHE_PERIOD=1` was correct and `period=32` was wrong,
which looked like an eviction-count bug; it was not.

**Root cause: CUDA graph capture.**  The slot-remap redirect is a *host* decision made inside
`ggml_cuda_mul_mat_id`.  A CUDA graph executes that function **only during capture**, so the captured graph
bakes in, per op, whether it reads the compact arena (+remap) or the full `input_cpy`.  The scheduler's
per-token hook kept deciding independently and, on takeover, **skips the `input_cpy` copy**.  When the two
disagreed - a replay whose captured choice was `input_cpy`, but whose token the hook took over - the op
read a buffer nobody refilled: stale experts, wrong output.  `period=1` made the hook's decision constant
(always takeover), so the capture and every replay agreed and it *looked* correct; `period=32` made `all`
flip, and the mismatch appeared.  See `make graph-safe` below for the fix.

Three real bugs contributed and are fixed:
1. **Inline remap.**  The hook read `expert_slot[e]` as it admitted each expert, but a later admission in
   the same token could evict an earlier one (its freshly-decayed count is the smallest), leaving an
   earlier position pointing at a slot a later fill had refilled.  Fixed with a **two-pass** remap: admit
   all, then read the map.
2. **Remap upload not stream-ordered.**  It was a synchronous `cudaMemcpy` on the legacy stream, which is
   not ordered with the non-blocking compute stream.  Fixed to `cudaMemcpyAsync` on the compute stream with
   a persistent host buffer.
3. **Declined hook left the old remap live.**  On `!all` the hook returned false without invalidating, so
   the consumer could still redirect using the previous token's remap.  Fixed by zeroing `remap_n_used` at
   the top of every hook call.

**The graph-safe fix** (`ggml_cuda_graph_check_compability`):
* **Protection.**  A used expert is never chosen as an eviction victim while its own token is being staged,
  so the hook always leaves every used expert resident and its success is a **constant** per shape.
* **Deterministic decline.**  A table with `slots < n_used * n_tok` declines before touching the map, also
  a constant per shape.
* **Capture after ready.**  `ggml_cuda_graph_check_compability` returns false while
  `moe_cache_enabled() && !moe_cache_ready()` (the arena not yet sized), so the first captured graph is
  taken with the cache ready and redirects.  With an explicit `MOE_EXPERT_CACHE_SLOTS` the cache is ready
  immediately; the deferred path captures from the second token.

**Result** (Q8_0, 1 GPU, `-ncmoe 99 -fa 1 -sm layer`, fusions off, `prompts/reasoning.txt`): every arena
size now matches the full-table GPU oracle with **CUDA graphs enabled** - `s8`/`s16`/`s64` and the 2048 MiB
deferred path all reproduce `eee6ad499b02` (128), `9b8aaf992518` (256), `efa4c3f4e8fc` (1024); Q4_K_M at
`s16` reproduces the full-table oracle `0e940e23c27f` (256).  Short gate `359ff4337837` identical;
`test-backend-ops -o MUL_MAT_ID` 4/4.  Graphs turn out to be nearly neutral for this workload anyway
(Q4_K_M d0 `tg1024` 46.99 with graphs vs 46.06 with `GGML_CUDA_DISABLE_GRAPHS=1`), so the earlier "graphs
cost 12 %" was the buggy build reading a stale buffer faster.

Diagnostics kept in the module: `MOE_EXPERT_CACHE_SKIP_ROLE` (bypass one role), `MOE_EXPERT_CACHE_ASSERT=1`
(map/slot invariant), `moe_cache_read_check()` (read-time routing/remap/byte check; graphs off only), and
the `takeover` / `decline_all` / `get_ok` counters in the report.

### Baseline caveat: the delivered CPU MoE is thread-starved (2026-09-28, reported by the maintainer)

The delivered `-ncmoe 99` decode MoE runs on only ~2 of 16 cores: Q4_K_M d0 `tg256` is flat **29.26 / 29.31
/ 29.39 / 29.22 t/s at `-t 2 / 4 / 8 / 16`**.  So the "cache vs delivered CPU" figure overstates the win -
the CPU side is leaving most of the machine idle.  The controlled comparison for the cache is **GPU cache
vs GPU no-cache** (the streaming path, ~7-10 t/s), and the full-table GPU oracle for correctness.  Worth a
separate look in the delivery: a fully-threaded CPU MoE would raise the baseline this campaign is measured
against.

### Purity protocol correction (2026-09-28)

The cache moves the decode MoE from the CPU to the GPU, and the delivered CPU MoE path and the GPU MoE
path are **pre-existing**-different over long generations (verified with the cache inert: `-ncmoe 99` CPU
`abac78fdb0a1` vs `GGML_OP_OFFLOAD_MIN_BATCH=0` GPU `8f7a6c226f4d`, 1024 tokens, Q8_0).  So "same-seed
text cache-on vs cache-off" must be evaluated **GPU-vs-GPU with the same fusion state**, and the trust-
worthy oracle is the **full-table GPU** run (`-ngl 99`, no `-ncmoe`) where the model fits.  CUDA fusions
also change the long-run text vs fusions off even with the cache inert (router fusion not the cause), so
hold the fusion state fixed on both sides.

### Traps learned this session (do not re-derive)

- **`llama-cli` needs an explicit small `-c` for a cache A/B.**  Its default context is the model's
  training length (262144 here); with the cache's arena plus that KV cache a single 32 GiB card thrashes
  and `Generation` reads 13.9 t/s instead of ~35 (deterministic - the text hash is unchanged).  Pin
  `-c 8192` and compare with `llama-bench`/`llama-batched-bench` for throughput.
- **The fusions-ON MoE text is not byte-pure even on the delivered path** (`none != draft-mtp`, first diff
  ~200 tokens, a paraphrase); only fusions-OFF is.  Do not treat a fusions-ON long-run text difference as
  a cache bug - run the `MOE_EXPERT_CACHE_FORCE_COPY=1` A/B first (equal output = no stale read).
- **A CUDA graph captures the HOST-side per-op decision.**  The consumer
  (`ggml_cuda_mul_mat_id`) runs only at capture, so the captured graph bakes in "read the arena" vs
  "read `input_cpy`" per op.  Any per-token decision that can flip (takeover vs decline) makes the
  capture disagree with a replay and the op reads a buffer nobody refilled.  This was the eviction bug.
  Keep the decision a **constant per shape** (protection + deterministic decline) and do not capture
  until `moe_cache_ready()`.  When seeing a residency/period-dependent wrong output, first re-run with
  `GGML_CUDA_DISABLE_GRAPHS=1` - if it goes away, it is this class.
- The top-k `ids` is a **strided view**.  Index it with `ids_tensor->nb[0]`/`nb[1]`, never
  `tok*n_used+j`.  (The scheduler's own `ids` vector is the raw linear bytes, not a compacted block.)
- Keep `src0`'s `ne[2]`/`nb` at the full expert count in the consumer: shrinking `ne[2]` to `slots` changes
  the dispatcher's kernel-family heuristics and breaks bit-identity.
- The scheduler hook runs **outside** CUDA graph capture: the tiny remap upload is synchronous (its source
  is a transient host vector), but the slot **fill** must be `cudaMemcpyAsync` on `ctx->stream()` (a
  synchronous fill inside capture aborts the graph).
- The host master and the op's `src0` are **different tensors** (the scheduler redirects `src0` to
  `input_cpy`).  Drive the policy from the scheduler hook with `weight` (master) and alias `weight_cpy`
  for the op lookup (`moe_cache_get_table`).
- `moe_cache_observe` is kept in the module but no longer called (it double-registered the device copy).
- **Any fusion that reads a routed expert table in the cache band must stand down**, or it reads the
  redirected `input_cpy` that the scheduler did not populate.  That is a *wrong output*, not a slowdown.
  The guard is `ggml_cuda_cache_blocks_fusion` (H1); every new decode-band `MUL_MAT_ID` fusion must go
  through it.  Prefill fusions are safe (the cache does not take prefill inputs over).
- The band is one constant, `MOE_EXPERT_CACHE_MAX_TOK` (8), shared by the hook, the offload relaxation,
  and the fusion guard.  Change it in one place only.

### Fusion policy (answer: yes, enable them; H1 did)

Fusions are bit-identical and beneficial, so the blanket stand-down was a prototype expedient, not the
plan.  H1 restores them now: every non-MoE fusion is on, and only the cache-band routed-expert fusions
stand down (the cache consumer replaces them).  Prefill MoE fusions are untouched.  The remaining option
is a cache-aware fused MoE that reads the arena/remap, which would restore the fused-MoE decode arithmetic
too; it is not needed for correctness or for the current win.

### After hardening

Phase 1b (UVA cold reads, section 3.3) replaces the per-miss fill with in-place pinned-host reads;
Phase 3 (section 3.5) is the `-sm tensor` per-device-slice geometry.

---

## 0. HANDOVER BRIEF — read this first (cold start)

### Goal

Make **decode (and speculative verify) fast for MoE models whose expert weights do not fit in VRAM**, by
keeping a **hot set of experts resident on the device** and handling the cold set cheaply — without giving
up the **tensor split** (`-sm tensor`) that the delivery now makes the best prefill configuration.

The end goal is **Qwen3.8-Flash-Next** (176 B params, 93 GiB at IQ4_NL, `qwen4exp`, 48 layers × 512
experts) running at useful decode speed on 2× R9700 — a model that does not fit 64 GiB and today decodes
slowly.  The iteration target is **Qwen3.6-35B-A3B Q8_0** (37.8 GiB, does not fit one 32 GiB card),
because it is faster to load and test.

### The gap (measured 2026-09-27, this box)

`llama-bench -p 0 -n 64 -fa 1`, Qwen3.6-35B-A3B **Q8_0**, gfx1201:

| config | `tg64` t/s |
|---|---:|
| 2 GPU `-sm layer -ncmoe 0` (all experts **resident**) | **79.7** |
| 2 GPU `-sm tensor -ncmoe 0` (all experts **resident**) | **90.5** |
| 1 GPU `-ncmoe 20` (half the expert layers on host) | 36.4 |
| **1 GPU `-ncmoe 99` (all experts host) — the campaign's baseline** | **24.2** |
| 2 GPU `-sm layer -ncmoe 99` | 22.5 |
| 2 GPU `-sm tensor -ncmoe 99` | 21.5 |

> These are the **r17** numbers (full CSV: `decode-baseline.csv`).  r13–r16 regressed this path to ~13.8 t/s
> (the r13 tiny-CPU-graph heuristic ran the offloaded decode MoE multi-threaded); the **r17** block-06
> amendment exempted `MUL_MAT_ID` src0 and restored the r12 baseline, so this campaign starts from a
> correct floor.  See `WORKLOG.md` 2026-09-27 (r17).

**Why all-host decode is slow: the MoE is not on the GPU at all.**  `ggml_backend_cuda_device_offload_op()`
(`ggml/src/ggml-cuda/ggml-cuda.cu`) only offloads an op when
`get_op_batch_size(op) >= op_offload_min_batch_size` (default **32**, `GGML_OP_OFFLOAD_MIN_BATCH`).  For
`GGML_OP_MUL_MAT_ID` that size is `op->ne[2]` = **`n_tokens`**, which is **1 at decode** — so the decode
graph's `ffn_moe_gate/up/down` are assigned to the **CPU**, in every split mode.  Prefill (ubatch ≥ 32)
offloads; decode does not.  So the 24.2 t/s is a **CPU MoE** number, and every extra resident layer buys
throughput (see the `-ncmoe 20` row) because it moves MoE work back to the GPU.

Note the shape of the prize: **a single card with a cache can beat two cards with none** (36.4 vs 22.5),
and the resident ceiling is 3.3× the baseline.  A working cache does not have to reach 79.7 to be a large
win; it has to move the decode MoE back onto the GPU for the experts that fit.

### Prior art — the baseline starting points (all read 2026-09-27)

Three implementations, summarised in the closed prefill campaign's
`archive/work/tensor-split-expert-split/README.md` §26-§28 (read those sections — they are the
requirements document, with exact file paths):

1. **GenerelSchwerz `llama.cpp` `moe-cache`** (`ggml/src/ggml-cuda/moe-cache.cu`, ~14k lines) — the mature
   **layer-split** design and the closest match to our stack.  Per-layer, per-owner (per-GPU) cache buffer
   with N expert slots; **grouped decode kernels** compute the resident experts on the GPU; misses are
   staged/uploaded; prefetch, replacement policy, an **early router**, and CUDA-graph capture.  Measured
   **Qwen3.6 35B decode 42.8 → 111.6 tok/s (2.6×)** at the same peak VRAM on a 16 GB RTX 5070 Ti.  Its host
   side (`moe-cache-host.cu`) is the mature version of our pinning/staging work: a `moe_host_source`, a
   bounded **pinned staging budget**, automatic expert-group registration, pageable fallback with pinned
   staging, and a dedicated **asynchronous copy worker thread**.  **It explicitly refuses `-sm tensor`**
   ("the tensor-mode meta backend cannot consume the cached buffer"), so it is a *layer-split* reference.
2. **R9V** (`github.com/Dyluhn/R9V`, local `~/R9V`) — **the prior art that DOES tensor-split offloaded
   experts**, on this exact 2× R9700 hardware, for Qwen3.8-Flash-Next.  It keeps
   `get_tensor_model_parallel_rank()`-sharded expert masters (each rank holds `1/tp` of **every** expert —
   i.e. exactly our `-sm tensor` split), a per-rank **hot/cold manifest** (`hot_experts_by_layer`, from
   routing calibration, `materialize_hot_expert_cache`), **cold shards in pinned UVA host memory read in
   place** (`allocate_uva_host_empty` + `parameter.data = get_accelerator_view_from_cpu_tensor(...)`), and a
   VRAM **slot cache** on top (`QWEN38_TIERED_EXPERT_CACHE_SLOTS` ≤ 128, `second_touch_rr`/`lru`,
   optional async fill).  **Key lesson: R9V does not win by making the upload fast — it avoids the upload**
   (resident hot set + UVA in-place cold reads).  Reference files:
   `runtimes/qwen38-flash-next-gfx1201-v1/retained-mtp4/python/vllm_gguf_plugin/quantization/tiered_experts.py`,
   `docs/config.md`, `crates/r9v-ir/src/plan.rs` (`ExpertPlacement::{Device, HostCompute, HostFetch}`).
3. **Strata** (`github.com/Niko1221/Strata`, local `~/Strata`) — single-card, **CPU computes the misses**
   concurrently with the GPU, plus a frequency-profiled VRAM cache (`src/core/expert_cache.cpp`).  Its
   bottleneck is the **CPU expert pool** (663.6 MB/token at ~40 GB/s), not the H2D — their fix is to stop
   reading bytes.  No split mode at all; least transferable.

The synthesis: **the proven decode lever is a hot-expert VRAM cache**; the open question is whether it can
be made to work under our **meta backend / `-sm tensor`**, which is exactly what R9V says is possible and
what nobody in the llama.cpp ecosystem has built (moe-cache refuses).

### The phased plan

Work strictly left to right; each phase must show a measured decode win over its own baseline before the
next starts.  **(Superseded 2026-09-28 by §3, which keeps this ordering but fixes the policy, the seam and
the `-sm tensor`-first constraint; read §3.)**

* **Phase 0 — baseline & instrumentation** (partly done, above).  Sweep `tg` across `-ncmoe 0..40` for
  1 GPU and 2 GPU, `-sm layer` and `-sm tensor`; confirm the CPU-vs-GPU MoE assignment; get a repeatable
  decode harness (the delivery's decode protocol is `d16384`, but shallow is fine for iteration — record
  depth with every number, §"Measurement protocol").
* **Phase 1 — single GPU, Qwen3.6-35B-A3B Q8_0.**  Build a per-layer VRAM **expert slot cache** on one
  card and get `MUL_MAT_ID` decode back on the GPU.  Start with a **static** resident set (no routing
  profile) and a simple policy; measure how throughput scales with the resident fraction.  This is the
  iteration vehicle: one card, one model, fast builds, the smallest surface.
* **Phase 2 — 2 GPUs, `-sm layer`.**  The cache is naturally per-device (a layer split assigns whole layers
  to a device), so this is the moe-cache-shaped case and should be the easiest multi-GPU step.
* **Phase 3 — 2 GPUs, `-sm tensor`.**  The hard and novel one: each device holds a **slice** of every
  expert (axis-1 gate/up, axis-0 down), so a "resident expert" is a resident **per-device slice**, and the
  down projection still needs its cross-device partial reduce.  R9V is the map.  This is the phase that
  unlocks the end goal.
* **Phase 4 — Qwen3.8-Flash-Next IQ4_NL** (93 GiB, `qwen4exp`/QSA).  Re-validate everything on the real
  target; note its PLE tables are host-resident and lazy-loaded (`--lazy-mode auto`), and pre-warm matters.

### Design space (decide by measurement, in this order)

* **A. Hot set in VRAM + stream the misses** (moe-cache).  Reuses what the delivery already has — the
  pinned host expert source (`LLAMA_MMAP_HOST_EXPERTS`), the block-06 staging ring, the op-offload
  `copy_experts` pruning.  Natural first implementation, and it works for `-sm layer` immediately.
* **B. Hot set in VRAM + cold experts read in place over UVA** (R9V).  Pin the host shards once and map
  them into the device address space (`hipHostRegister` + `hipHostMallocMapped`/`cudaHostGetDevicePointer`),
  then have the expert kernel read cold experts straight from host memory — **no per-token H2D at all**.
  This is the end state that avoids the transfer rather than hiding it, and the one that composes with the
  tensor split.  llama.cpp's op-offload does not use UVA today.
* **C. CPU computes the misses** (Strata).  Useful as a fallback for boxes with no spare VRAM, but it
  fights the CPU-pool bandwidth limit and has no split story; keep it as a comparison, not the plan.

### Where the work lives (code hooks)

| concern | location |
|---|---|
| op-offload gate (why decode is on CPU) | `ggml_backend_cuda_device_offload_op()` + `get_op_batch_size()` in `ggml/src/ggml-cuda/ggml-cuda.cu` (default 32, `GGML_OP_OFFLOAD_MIN_BATCH`) |
| the MoE op | `GGML_OP_MUL_MAT_ID`; decode kernel `mul_mat_vec_q_moe` / `ggml_cuda_mul_mat_id` |
| used-expert pruning (host→device ranges) | `ggml_backend_sched_compute_splits`'s `copy_experts` in `ggml/src/ggml-backend.cpp` |
| host-expert pinning (r15) | `select_weight_buft` in `src/llama-model-loader.cpp` (`LLAMA_MMAP_HOST_EXPERTS`) |
| H2D staging ring (r16) | `sched_stage_*` (`ggml-backend.cpp`), `stage_*` hooks (`ggml-backend-impl.h`, `ggml-cuda.cu`, `ggml-cuda/common.cuh`) |
| the split upload (r16) | `ggml_backend_meta_stage_input` + `ggml_backend_meta_set_tensor_async` (`ggml/src/ggml-backend-meta.cpp`) |
| per-tensor split policy | `llama_meta_device_get_split_state` (`src/llama-model.cpp`) |
| existing MoE helpers | `ggml/src/ggml-cuda/topk-moe.cu`, `moe-weighted-reduction.cu`, `ggml/src/models/qwen4exp.cpp` |

### Open questions / risks

1. **Where does the cache live** — a new `ggml_cuda` buffer type (moe-cache) vs the meta backend's simple
   tensors (needed for `-sm tensor`).  The meta backend "cannot consume a cached buffer" today; that is the
   Phase-3 problem to solve.
2. **Residency decision** — static vs routing-level hot set.  The maintainer's earlier steer was "no
   explicit hot-weight tuning" for prefill; for decode a **measured** hot set is the proven 2.6× (moe-cache,
   R9V), so expect to need one.  Start static, then add a routing profile.
3. **Decode/verify purity** — the delivery's `W=1..8` width-purity rule (`GREEDY-PURITY.md`) means the
   cache must produce the *same* arithmetic at every verify width; a grouped/split-by-residency kernel is
   exactly the kind of change that breaks it.  Gate every step with the plain-vs-`draft-mtp` purity check.
4. **Speculative decode** — MTP drafts `n` tokens per step; the used-expert set and the cache hit rate must
   be evaluated at the **verify width**, not just `n_tokens = 1` (the delivery's `MMVQ_MAX_BATCH_SIZE`/
   `MMVF_MAX_BATCH_SIZE` banding).
5. **VRAM budget vs KV** — the cache competes with KV cache for VRAM; the delivery's `--fit` does not know
   about it (same trap as the r3 staging-arena OOM, issue #33).  Any cache allocation must fail soft.
6. **`-sm tensor` down projection** — axis-0 split, `nb[1]`-sized chunks (~524k tiny blocks); any per-token
   upload must be device-side or 2-D (the prefill campaign's §26.3/§29).  This is why UVA (B) is the
   promising `-sm tensor` answer.

### Follow-ups inherited from the prefill campaign (do these as part of this campaign)

These are the loose ends the prefill campaign did **not** close; they are now owned here, cross-referenced
from `archive/work/tensor-split-expert-split/README.md` (§30.5-§31):

1. **Prune the staged upload to the used experts.**  The r16 staged path uploads the **whole** expert
   tensor (all experts, half per device) regardless of ubatch, bypassing the used-expert pruning; at
   ub 512 that is ~10× the volume the pruning path moves.  The split's compute floor is ~7000 t/s and the
   staged path reaches ~5400 at ub 8192, so there is room.  (This also matters for decode if the staged
   mechanism is reused there.)
2. **`GGML_META_GATHER_MODE` device-count heuristic.**  The auto choice (host gather vs device D2D for the
   fine `ffn_down` slice) **inverts** with device count: at 3 GPUs the host gather wins at ub 8192
   (4905 vs 4694), while at 2 GPUs auto (D2D) wins.  Make the heuristic device-count-aware.
3. **3-GPU ub-8192 split loss.**  The split is −5.6 % vs mirrored at ub 8192 on 3 GPUs (the quant block
   limits the split to 2-of-3 devices; the loss is in the upload path).  Close it or document it.
4. **Strip the campaign's env-gated debug/A-B knobs from the delivery.**  The r16 promotion carries the
   debug instrumentation (`GGML_META_STAGEDBG`, `GGML_CUDA_GCDBG`, etc.) alongside the user-facing
   kill-switches; fold them out before any `upstream/` PR candidate is cut.  (Recorded in `WORKLOG.md`
   r16 and `patches/README.md`.)
5. **The §31 prefill residency idea is superseded** by this campaign's Phase 3/UVA design — do not build a
   separate prefill-only residency path; make the cache/UVA design serve both if it can.
6. **`-ncmoe` decode on the CPU** is the root observation of this campaign (see "The gap"); it is not a
   "follow-up" so much as the problem statement.

### Environment, build, and repro

* **Build tree for this campaign: created, rebased onto r21 and built** — `~/llama-decode`, branch
  `wip-moe-expert-cache`, which is the **r21 tip `feefecfbc`** (tree
  `9975a333d3d785da662dfcc9b601c442d3be8104`) **+ 9 wip commits** (tip `ce5c9731e`), with
  `build-rocm/{bin/llama-bench,bin/llama-cli}` ready.  Iterate with `cmake --build build-rocm --target
  llama-bench llama-cli -j 16` (the cache/mmvq TUs rebuild in a minute or two with ccache; a full build via
  `BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714` is ~7 min cold).
  A clean upstream reference is `~/llama-upstream` at `84e76d8a2`; the canonical r21 chain is the
  `~/llama-r21` worktree (branch `rdna-r21`, tip `feefecfbc`; the delivery `main` is `v16-84e76d8a2-r21`).
* **Iteration models:** `/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf` (37.8 GiB — the
  Phase-1 vehicle) and `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/…UD-Q4_K_M.gguf` (21.1 GiB — fast smoke tests,
  fits one card).
* **End-goal model:** `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-0000N-of-00009.gguf`
  (93 GiB, `qwen4exp`); its shipping `runme` uses `--lazy-mode auto` + the shared MTP head, and **PLE
  pre-warm matters for its absolute numbers** — do not compare `llama-bench` absolutes for it without the
  server/lazy path.
* **Repro of the baseline gap:** `HIP_VISIBLE_DEVICES=0 ./build-rocm/bin/llama-bench -m <Q8_0> -ncmoe 99 -fa 1
  -p 0 -n 64 -sm tensor -r 3` → ~24 t/s; `GGML_OP_OFFLOAD_MIN_BATCH=0` puts the decode MoE back on the GPU
  for a *correctness* look (it will be slow while every expert streams per token).
* Runtime libs come from the binary's RUNPATH (`/opt/rocm-7.14.1-gfx102X/lib`); `~/llama-r12` is the old
  prefill campaign's tree — use `~/llama-promote`/a fresh worktree instead.
* **Never run parallel benches**; pin nothing; record depth with every number.

### Measurement protocol (do not skip)

* Decode must be measured **at depth** as well as shallow (`d0` and, for the delivery gate, `d16384`):
  "everything is fast at depth 0" is the delivery's standing rule.
* Always gate a decode/fusion change with the **MTP** methodology
  (`benchmarks/mtp-adaptive-methodology.md`) and the **width-purity** check
  (`--spec-type none` vs `--spec-type draft-mtp`, `W=1..8` bit-identical), per `GREEDY-PURITY.md`.
* Same-seed greedy text must be identical to the pre-change build; `test-backend-ops -o MUL_MAT_ID` green.

### Artifacts in this directory

* `README.md` — this brief (the live notebook from here on; append dated findings, never edit them away).
* `decode-baseline.csv` — the Phase-0 `tg` sweep to beat (added with the first measurement session).

### Immediate next steps (fresh session)

1. ~~Build the tree and reproduce the gap table.~~ **DONE** — `~/llama-decode` is built at r17 and
   `decode-baseline.csv` records the `tg64` baseline (all-resident 79.7, all-host 24.2).
2. Read the three prior-art sections (`archive/work/tensor-split-expert-split/README.md` §26-§28) and pick
   the Phase-1 kernel/cache shape (recommended: A, reusing the pinned host source + staging ring).
3. Prototype **Phase 1** on one GPU: a static-resident expert slot cache + a decode path that keeps
   `MUL_MAT_ID` on the GPU for resident experts, with the misses on the existing host path.  Measure
   throughput vs resident fraction.  Fail soft on VRAM (issue #33 lesson).
4. Only then move to Phase 2 (`-sm layer`, 2 GPU) and Phase 3 (`-sm tensor`, 2 GPU = the R9V case).

---

## 1. Phase-1 design — port Strata's expert cache (single GPU), keeping the R9V end state in view

**Maintainer direction (2026-09-27):** start from `~/Strata` — its shape is the most compatible with a
single GPU, and it is the first working milestone before `-sm tensor`.  **Constraint:** the design must be
adaptable to the `-sm tensor` end state (`~/R9V`, TP-sharded experts + hot set + UVA cold reads) without a
rewrite — see §1.2 first, because it decides the data layout.

### 1.1 What Strata actually is (read from source, 2026-09-27)

Three objects, each already verified upstream; the third is unfinished there:

1. **`PinnedArena`** (`include/strata/core/pinned.hpp`, `src/platform/memory.cpp`) — all expert weights in
   **page-locked host memory** (2 MiB pages / `cuMemHostRegister`, with a normal-page fallback that is
   *reported*, not silently taken), with optional per-slice registration and a **device alias** for UVA.
   → **We already have most of this:** r15's `LLAMA_MMAP_HOST_EXPERTS` puts the host-resident `MUL_MAT_ID`
   weights in the `ROCm_Host` buft (`hipHostMalloc`).  Missing: large-page backing, one arena instead of N
   pinned tensors, and the **device alias** (`hipHostGetDevicePointer`) the UVA cold path needs.
2. **`ExpertCache`** (`include/strata/core/expert_cache.hpp`, `src/core/expert_cache.cpp`) — a **VRAM slot
   arena** plus a `(layer, expert) → slot or -1` residency table, filled from the pinned arena (async DMA or
   blocking at startup), with `verify_slot()` reading a slot back and byte-comparing.  **No eviction**
   (deliberate — eviction policy is the measured question, `R4.1`).  The load-bearing finding is
   **per-layer admission**: a global free-slot counter fills the cache inside the first position it ever sees
   (measured 2.97 % hit at 256 slots); giving each layer its own range `q = slots/n_layers` turns the same
   budget into 21 % (8 slots/layer) / 70 % (64).  The per-layer range is what we take.
3. **`ExpertDispatch`** (`include/strata/core/expert_source.hpp`) — **the split is by router index**: of the
   K routed experts, the resident ones are computed on the **GPU** and the rest on the **CPU**; the CPU zeroes
   the hit rows of `parts` and each hit carries its routed index (`dst`) to the GPU kernel; `moe_combine`
   sums by router index, so the two halves rejoin without a scatter.  **Strata has not finished this** —
   `moe_hit_grouped_s2` does not exist, so with the cache on "the engine is slower by the fill cost and faster
   by nothing".  That GPU hit kernel is the piece we must build.

Its measured context (a different, larger model) is worth keeping: the CPU expert pool costs **663.6 MB/token
at ~40 GB/s = 16.2 ms of a ~53 ms token**, and the pipeline hides only 1.055 ms of it — the CPU work is 96 %
exposed.  The lever is to stop the CPU reading those bytes, not to make the CPU read them faster.

### 1.2 The R9V constraint (decide the layout now, not later)

`-sm tensor` decode means each device holds a **slice** of every expert (axis-1 gate/up, axis-0 down) and the
down projection carries a cross-device partial reduce.  So:

* **The residency is keyed by `(layer, expert)`; the slot holds *this device's slice* of that expert.**  Under
  1 GPU / `-sm layer` the slice is the whole expert; under `-sm tensor` it is the axis slice.  The table and
  the dispatch are identical — only `blob_bytes` and the kernel geometry change.  **Do not build the cache
  around whole-expert blobs**; build it around "the bytes this device would upload for this expert".
* **The GPU hit path computes a per-device partial result**, exactly as the existing on-device split does;
  the meta backend's partial reduce is unchanged.  Nothing in the cache may assume the output is complete.
* **The cold path must not be a per-token H2D of the misses.**  The cache makes `(1-h)` of the experts cold;
  if those still cross PCIe every token the streaming bound below kills it.  R9V's answer is **UVA** — map the
  pinned host shards into the device address space and read the cold slices **in place** — which is also the
  only sane shape for the axis-0 `ffn_down` slice (the prefill campaign's §26.3: a host gather there is
  ~524k tiny copies).  **So the single-GPU prototype should still route cold reads through a `device_alias()`
  hop even when the alias is the same device**, so the `-sm tensor` port is a geometry change, not a rewrite.

### 1.3 Measured bounds (2026-09-27, r17, gfx1201)

Qwen3.6-35B-A3B, `tg64`, `-fa 1`:

| shape | 1 GPU | 2 GPU |
|---|---:|---:|
| **CPU MoE** (today's `-ncmoe` decode), Q8_0 | **24.3** | 21.3 (tensor) / 22.8 (layer) |
| **GPU MoE, stream every miss** (`GGML_OP_OFFLOAD_MIN_BATCH=0`), Q8_0 | **7.6** | — |
| GPU MoE, **all resident**, Q4_K_M | 91.8 | — |
| CPU MoE, all host, Q4_K_M | 28.9 | — |
| CPU MoE, half resident, Q4_K_M | 41.9 | — |
| GPU MoE, all resident, Q8_0 | (does not fit) | **90.5** `tensor` / 80.5 `layer` |

Two conclusions the design must respect:

* **PCIe is slower than the CPU's own pinned-memory reads.**  14.45 GB/s over the link versus ~40 GB/s for
  the CPU pool means *"put the MoE on the GPU and stream the misses"* is **3× worse than the CPU** (7.6 vs
  24.3).  A cache is not a nice-to-have; without it the GPU path loses.  The cold path must be UVA (in place)
  or CPU-compute, not a per-token H2D.
* **The prize is large and bounded by resident fraction.**  On one card, all-resident is 3.2× the CPU path
  (91.8 vs 28.9 on Q4_K_M).  A cache that holds a fraction `h` should land between them; the hit rate is the
  number to measure, and per-layer admission is what makes a small budget useful.

### 1.4 Cache budget (Q8_0, one 32 GiB card)

A Q8_0 expert (gate+up+down) is **3.00 MiB**, so with ~4–10 GiB spare on the card:

| cache | slots/layer (of 256) | resident |
|---:|---:|---:|
| 4 GiB | 34 | 13 % |
| 6 GiB | 51 | 20 % |
| 8 GiB | 68 | 27 % |
| 10 GiB | 85 | 33 % |

Strata's profile hit rate is `h_expert ≈ 0.645` at 4,105 slots for a 48×512 model (0.61–0.68 ten-fold
leave-one-out); a smaller per-layer budget will be lower, and the top-8 routing concentrates the popular
experts, so 20–30 % resident is plausibly useful.  **Measure `h` on real routing before sizing anything.**

### 1.5 Phase-1 implementation plan (single GPU, layer split)

> **Superseded 2026-09-28 by §3.**  The steps below are the starting sketch; §3 is the go-forward plan.  It
takes the same Strata shape but pins the policy (LFRU, §2.2) and the seam (`device_alias()` pointer table,
§2.3), and demotes the static profile to a warm start / optional hard-pin analyser.

1. **Residency plumbing.**  A VRAM slot arena + `(layer, expert) → slot` table (per-layer ranges), fed by a
   static/profile hot set, on top of the r15 pinned host source.  No eviction.  Gate it off by default
   (`GGML_MOE_EXPERT_CACHE_MIB`, 0 until measured) with a startup `h`/`fills`/`verify` report like Strata's.
2. **The GPU hit path.**  Compute the resident experts on the GPU from the cache and the misses on the CPU
   (or UVA), split **by router index**, combined in router order.  Build it so the "device address" of a blob
   is already an abstraction (the `device_alias()` hop of §1.2), even when it points at the same device.
3. **Purity gate.**  The delivery's `W=1..8` width-purity rule applies: the cache must produce the same
   arithmetic at every verify width and with the cache on/off (`--spec-type none` vs `draft-mtp`, same-seed
   text).  The split-by-residency path is exactly the kind of change that breaks this.
4. **Measure** `h`, `tg` vs resident fraction, and the cold-path cost, on Q8_0 single GPU; only then move to
   2-GPU layer split, then `-sm tensor` (the R9V case).

### 1.6 The gating measurement (2000-token generations, held-out): the routing is concentrated, but a STATIC profile does not transfer

**Instrument:** `exp1-moe-routing-profiler.patch` — an env-gated (`GGML_MOE_PROFILE`) counter in the CPU
`ggml_compute_forward_mul_mat_id` (where decode's MoE runs under `-ncmoe`).  It records `(layer, expert)`
usage for the **decode/verify band only** (`n_tokens <= 8`; a prefill ubatch routes nearly every expert and
would swamp the distribution) and dumps `expert,<layer>,<expert>,<count>` at exit.  `hitrate.py` scores it.

**Runs:** Qwen3.6-35B-A3B Q8_0, 1 GPU, `-ncmoe 99 -sm layer`, **2000 greedy tokens each** on the delivery's
five prompts (`prose`, `code-python`, `reasoning`, `recall`, `code-reasoning-mixed`).  (A 256-token run was
done first and over-states everything — longer generations keep touching new experts, so the same cache
covers less of the distribution: in-sample h at S=64 was 0.800 at 256 tokens, **0.724 at 2000**.)

`h(S)` = fraction of routed experts a **per-layer static hot set of `S`** would have resident, book = the
prompt's own frequency (in-sample) or a specified book (held-out):

| S/layer | resident | cache | in-sample (range over 5 prompts) | held-out, book = **one** prompt (prose) | held-out, book = **all 5** combined |
|---:|---:|---:|---:|---:|---:|
| 16 | 6 % | 1.9 GiB | 0.36–0.42 | — | — |
| 32 | 12 % | 3.8 GiB | 0.52–0.61 | — | 0.13–0.44 |
| 48 | 19 % | 5.6 GiB | 0.63–0.73 | — | 0.21–0.53 |
| **64** | **25 %** | **7.5 GiB** | **0.63–0.81** | **0.24–0.38** | **0.30–0.61** |
| 128 | 50 % | 15.0 GiB | 0.90–0.96 | 0.54–0.64 | 0.68–0.85 |

**Read — this changes the design, and it is why the measurement was worth doing:**

1. **The routing is concentrated and a cache is clearly worth it *if the book matches the workload*.**
   In-sample, 25 % resident buys 63–81 % of the reaches; 50 % buys 90–96 %.  On one 32 GiB card a ~7.5 GiB
   cache is realistic and would leave the cold path ~1/5 of the experts instead of all of them.
2. **A static hot set does NOT transfer across workloads.**  A book built from `prose` scores only
   **0.24–0.38 at S=64** when the run is `code`/`reasoning`/`recall` (vs 0.63–0.81 in-sample).  Even a book
   built from all five prompts only reaches **0.30–0.61**, and `recall` (verbatim continuation) is the worst
   transfer of all (0.30).  So a **static profile is a warm start, not the mechanism** — it must be paired
   with a **dynamic admission policy** that adapts to the running workload (Strata's `second_touch_rr`,
   R9V's slot cache, moe-cache's replacement) and/or a per-workload profile.
3. **Budget the dynamic policy, not the profile.**  The meaningful number to design against is the
   *steady-state* hit rate of the chosen policy (fill-on-second-touch / LRU / LFU-decay) measured over a
   long run, not the profile's in-sample score.  The static numbers above are the upper bound of the warm
   start.

**Follow-ups the profiling leaves:** (a) score on a *later* slice of the same long run (warm the cache on
tokens 0..1000, score on 1000..2000) — the closest proxy for a dynamic policy; (b) measure a real
second-touch/LRU policy, not a rank list; (c) re-check on `qwen4exp`/Flash-Next (48×512), where Strata's
held-out profile was 0.645 — a much larger expert count, so the hit rate may behave differently.

Artifacts: `profile-prose-256tok.csv` (the first, short run, kept to document why short runs mislead) and the
five 2000-token dumps (regenerable via the commands in §1.7; not all committed for size).  `hitrate.py`
reproduces every table above.

---

## 2. FINDINGS (2026-09-28): the dynamic admission policy measured; the cache integration branch point settled

This session did the top task of the handover: **settle the policy and the integration seam before writing
any cache code** (the campaign's "measure h first" rule).  Three results: the instrument now captures an
ordered, token-indexed trace; the policy is measured on all five workloads at 2000 tokens; and the
`ggml_cuda`-buffer-vs-meta-simple-tensor question (open question #1) is answered.

### 2.1 Instrument: `exp2-moe-routing-profiler.patch` (supersedes `exp1`; do not apply both)

The old `exp1` profiler counted per `MUL_MAT_ID` op and aggregated `(layer, expert)` counts.  It had two
flaws for a policy measurement:

* **it triple-counted.**  The MoE graph issues one `MUL_MAT_ID` per gate/up/down and they all share the same
  `ids`, so the old `layer` line was `3 x tokens` (the committed `prof-*.csv` show 5370/6021 for 1790/2007
  tokens).  The extension coalesces consecutive ops of one `(layer, token)` — exactly the granularity a cache
  looks an expert up at.
* **it had no order.**  LFRU/LRU/LFU/second-touch are all order-dependent, so an aggregate book cannot score
  them.  The extension writes an ordered, token-indexed trace: `route,<tok>,<layer>,<expert>` (one line per
  reach) to `<path>.trace`.

It also adds the requested `GGML_MOE_PROFILE_SPLIT=N`, which writes two aggregate books for the **same run**:
`<path>.pre` (tokens `< N`) and `<path>.post` (tokens `>= N`) — the held-out proxy §1.6 asked for — and a
`split,<N>` marker in the trace.  `GGML_MOE_PROFILE_MAXTOK=1` records pure W=1 decode (a prefill *tail*
chunk of `<=8` tokens is otherwise indistinguishable from a verify batch and pollutes the trace; even the
2000-token runs had a 7-token tail in the `reasoning` prompt before this).  Inert without
`GGML_MOE_PROFILE` (one `getenv`).

Traces for this section (1 GPU, Qwen3.6-35B-A3B Q8_0, `-ncmoe 99 -fa 1 -sm layer`, `-n 2000 --ignore-eos`,
`GGML_MOE_PROFILE_MAXTOK=1 GGML_MOE_PROFILE_SPLIT=1000`, one run per prompt):

```sh
MQ=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
for p in prose-rdna-boosts code-python reasoning recall code-reasoning-mixed; do
  env HIP_VISIBLE_DEVICES=0 GGML_MOE_PROFILE=/tmp/moecache-traces/$p.csv \
      GGML_MOE_PROFILE_SPLIT=1000 GGML_MOE_PROFILE_MAXTOK=1 \
      ./build-rocm/bin/llama-cli -m "$MQ" -ncmoe 99 -fa 1 -sm layer \
      -f ~/llama-cpp-rdna-boosts/prompts/$p.txt -n 2000 --ignore-eos \
      --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt
  # -> /tmp/moecache-traces/$p.csv{,.pre,.post,.trace}
done
# policy simulator (no GPU needed, ~1 min for 8 policies x 5 S x 5 prompts)
python3 policy_sim.py /tmp/moecache-traces --windows /tmp/moecache-traces/policy-windows.csv
```

Each trace is 1999 decode steps (8 experts x 40 layers = 639,680 reaches; the first sampled token's MoE runs
inside the prompt batch, so 2000 generated tokens give 1999 profiled steps) at d0.  Runtime libs come from
the binary RUNPATH; the campaign build tree is `~/llama-decode` (`build-rocm`).  `--ignore-eos` is required
to compare all five at equal length: `recall` emits EOS at 505 tokens on its own (§1.6's dump), the other
four ran to ~1800-2000, so the earlier numbers were not length-matched.

**Same-run held-out warm/cold (item 1's deliverable).**  Rank by tokens 0..1000, score on 1000..2000 — the
same workload, so this removes §1.6's cross-workload transfer error:

| prompt | S=8 | S=16 | S=32 | S=48 | S=64 |
|---|---:|---:|---:|---:|---:|
| code-python | 0.248 | 0.368 | 0.496 | 0.589 | 0.659 |
| code-reasoning-mixed | 0.204 | 0.326 | 0.458 | 0.555 | 0.634 |
| prose-rdna-boosts | 0.188 | 0.292 | 0.435 | 0.554 | 0.642 |
| reasoning | 0.222 | 0.343 | 0.504 | 0.614 | 0.695 |
| recall | 0.219 | 0.351 | 0.544 | 0.679 | 0.770 |

At 25% resident (S=64, 7.5 GiB) the warm start holds **0.63-0.77** within a workload — the static profile is
far better than §1.6's cross-workload 0.24-0.38, but it is still a warm start, not the steady state (below).

### 2.2 Measured: a tuned LFRU dominates LRU, LFU and second-touch on the robustness metric

`policy_sim.py` replays the trace per layer (per-layer slot ranges, Strata's load-bearing finding) and
implements the prior-art policies (`static`, `warm_lru`, `lru`, `second_touch`, `lfu_decay`) plus combined
ones (`lfru_decay`, `slru`, `lru_window`).  `h_steady` is the hit rate over the **last 1000 of 1999** steps;
"worst" is the minimum across the five prompts (= the robustness number the campaign asked for, not the best
rank list).  Q8_0, 1 GPU, `-ncmoe 99 -sm layer`, d0:

| S/layer | cache | policy | `h_steady` mean | worst prompt | worst 200-tok window |
|---:|---:|---|---:|---:|---:|
| 64 (25%) | 7.5 GiB | static (warm 0..1000) | 0.680 | 0.635 | 0.553 |
| 64 | 7.5 GiB | `lru` | 0.763 | 0.737 | 0.700 |
| 64 | 7.5 GiB | `second_touch` | 0.752 | 0.729 | 0.699 |
| 64 | 7.5 GiB | `lfu_decay:32` | 0.766 | 0.743 | 0.718 |
| 64 | 7.5 GiB | `slru:0.5` | 0.775 | 0.753 | 0.726 |
| 64 | 7.5 GiB | **`lfru_decay:32`** | **0.783** | **0.764** | **0.736** |
| 32 (12%) | 3.7 GiB | `lru` | 0.583 | 0.542 | 0.505 |
| 32 | 3.7 GiB | `second_touch` | 0.561 | 0.517 | 0.460 |
| 32 | 3.7 GiB | `lfu_decay:32` | 0.586 | 0.562 | 0.527 |
| 32 | 3.7 GiB | **`lfru_decay:16`** | **0.611** | **0.576** | **0.547** |
| 16 (6%) | 1.9 GiB | `lru` | 0.425 | 0.360 | 0.339 |
| 16 | 1.9 GiB | **`lfru_decay:16`** | **0.438** | **0.398** | **0.377** |
| 8 (3%) | 0.9 GiB | `lru` | 0.229 | 0.175 | 0.163 |
| 8 | 0.9 GiB | **`lfu_decay:32`** | **0.288** | **0.258** | **0.244** |

`lfru_decay` = admit every miss, evict the resident with the smallest hit counter, break ties by oldest use,
halve all counters every 16-32 decode steps.  Full grid: `policy-sim.csv`; per-window rates:
`policy-windows.csv`.

**What it says.**

1. **A dynamic policy beats the static warm start at every budget** (+0.05 at S=8, +0.10 at S=64, mean).  The
   static book is a useful *accelerator* (it removes the cold first ~100 steps) but it is not the mechanism —
   exactly what §1.6 predicted, now confirmed within a single workload.
2. **The tuned hybrid is the winner on the metric that matters: the worst workload and the worst window.**
   `lfru_decay` (period ~16-32) has the highest worst-prompt `h_steady` at S=16 and S>=32 and is within
   0.008 of the best at S=8; `lru` is beaten on its own worst case by +0.027 (S=64) to +0.034 (S=32).  The
   decay period is a broad plateau (16/24/32 are within ~0.01); period 32 is the robust default, and at a
   **tiny** budget (S<=8) the pure-LFU admission filter (`lfu_decay`) is marginally better than admit-all.
3. **`second_touch` is dominated.**  R9V's default and moe-cache's shape measure the *lowest* worst-case
   dynamic policy at every S: the ghost counter delays admission, so it reacts slower to a phase shift than
   plain LRU and cannot match LFRU's frequency protection.  Corroborated, but not the policy to port.
4. **The collapse is real, and only the dynamic policies survive it.**  The phase-switching prompt
   (`code-reasoning-mixed`, S=64) shows the static book decaying mid-run — windows 0-9:

   | policy | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
   |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
   | static (warm) | 0.83 | 0.78 | 0.72 | 0.72 | 0.78 | **0.55** | **0.57** | 0.72 | 0.66 | 0.67 |
   | `lru` | 0.83 | 0.83 | 0.81 | 0.81 | 0.82 | 0.82 | 0.80 | 0.82 | 0.83 | 0.81 |
   | `lfru_decay:32` | 0.84 | 0.81 | 0.79 | 0.80 | 0.80 | 0.84 | 0.81 | 0.81 | 0.82 | 0.78 |

   `prose` is the same shape (static 0.75→0.58 at window 8, `lfru_decay:32` 0.80→0.75).  The static book is
   `0.83` while the code phase runs and `0.55` after the phase change, because it was warmed on the wrong
   half.  A design that fills the cache by the profile alone and never evicts would inherit that cliff.

**Decision:** the cache's admission/replacement policy is **LFRU — admit every miss, evict the resident with
  the smallest decaying hit count (LRU tie-break), counters halved every ~32 decode steps** — with per-layer
  slot ranges.  The static profile is a warm-start hint only; `slru:0.5` is the close, counter-free second
  (two ordered lists, no decay), worth keeping as the low-overhead A/B.

**Depth check (house rule: measure at depth as well as shallow).**  The same 2000-step trace from a ~16k
prompt (the prose prompt tripled, 15.7k tokens, `-c 20480`) reproduces the ranking.  S=64: `lfru_decay:32`
h_steady `0.779` / worst window `0.750`, `slru:0.5` `0.769`/`0.730`, `lfu_decay:32` `0.759`/`0.741`, `lru`
`0.755`/`0.693`, `second_touch` `0.738`/`0.708`, `static` `0.681`/`0.636`.  The choice is not a shallow-context
artifact; the five-prompt set already spans prompt depths ~0.3k-5.2k and this adds ~16k.

### 2.3 Recommendation: hook the cache at the `MUL_MAT_ID` weight-source seam, not a buffer type

This settles §1.4 open question #1.  **Recommendation: do not add a `ggml_cuda` cache buffer type to the
delivery.  Build the residency as a per-device `(layer, expert) -> device_alias()` manager at the CUDA
backend's `MUL_MAT_ID` weight-source seam, and use a custom CUDA arena only as the Phase-1 single-GPU
scaffold behind that same `device_alias()` seam.**

Why not a buffer type, from the code:

* **A buffer type is static storage; residency is dynamic.**  `ggml_backend_meta_buffer_type()` builds the
  meta tensor's split from its simple devices' *existing* buffer types, and `ggml_backend_meta_device_supports_buft()`
  (`ggml/src/ggml-backend-meta.cpp`) only accepts a buft owned by one of its simple devices.  A cache buft
  therefore cannot back the split expert tensor; the split tensor stays in the ordinary CUDA buft and the
  cache would have to be a second, side allocation the kernel consults.  This is precisely why moe-cache
  refuses `-sm tensor` (§26.2: "the tensor-mode meta backend cannot consume the cached buffer").
* **The kernel needs a base pointer per expert, not a new allocation.**  Today `src0` is one contiguous expert
  table and the MMQ/mmvq path offsets rows by expert id (the ids compaction is in `ggml/src/ggml-cuda/mmid.cu`;
  the same `ids` drive gate/up/down).  A cache makes the used experts' storage discontiguous — a resident
  VRAM slot or a cold alias — so the interface the kernel needs is `(layer, expert) -> base_ptr`.  That is
  the *same* interface for 1 GPU, `-sm layer` and `-sm tensor`; only `blob_bytes` (the device's slice) and
  the pointer's source change.  A `device_alias()` hop now makes the `-sm tensor` port a geometry change, not
  a rewrite (the §1.2 constraint).
* **The scheduler seam already exists.**  `ggml_backend_sched_compute_splits`'s `copy_experts`
  (`ggml/src/ggml-backend.cpp`, the `used_ids` bitmap + `ggml_backend_tensor_set_async` into the per-op
  `input_cpy`) already copies only the used experts to the device.  A cache turns "copy every used expert"
  into "copy only the misses", and the `input_cpy`/`set_tensor_async` path is the natural hook.  The
  op-offload gate (`get_op_batch_size()` in `ggml/src/ggml-cuda/ggml-cuda.cu`: `MUL_MAT_ID` -> `op->ne[2]` =
  `n_tokens` = 1 at decode, vs the default 32) has to be relaxed for a cache-active MoE, since the resident
  experts *are* on the device.  Under `-sm tensor` the meta device's `offload_op` is `all_of(simple_devs)`,
  so the same relaxation is needed there.
* **`-sm tensor` fixes the cold path to UVA.**  Each device's slot holds *its* slice (axis-1 gate/up, axis-0
  down); `ffn_down`'s split is `nb[1] = 176` x `524288` chunks (§26.3), so a per-token H2D of the misses is
  half a million tiny copies per layer — the only sane cold path is R9V's: pin the host slice and map it into
  the device address space (`hipHostRegister` + `hipHostGetDevicePointer`), then read cold experts in place.
  The meta backend's per-device partial reduce is untouched.

The trade, stated plainly:

* **Custom `ggml_cuda` arena (moe-cache's shape):** fastest route to a Phase-1 number, reuses `input_cpy`, no
  scheduler/meta changes — but it is `-sm layer`-only, is thrown away for the end goal, and puts a
  `--fit`-invisible allocation inside the CUDA backend (the issue #33 trap; any VRAM allocation must fail
  soft).
* **`device_alias()` pointer table on the meta simple tensors:** more work (a residency manager, a
  pointer-indirection in the `mul_mat_id` dispatch, pinned host slices + UVA registration, and a scheduler
  that can ask for "resident slot or cold alias") — but it is the only geometry that reaches the 93 GiB
  qwen4exp/`-sm tensor` end goal, and it carries forward the r15 pinning + r16 staging/op-offload work.

**Decision:** branch at `device_alias()`.  Phase 1 may wrap a plain CUDA slot arena to get a number, but the
residency table and the kernel-facing pointer must be written once, in the shape both splits need.

**Two quantitative bounds to carry into Phase 1 (Q8_0, 1 GPU, from §1.3):** a token touches
`8 experts x 3 MiB x 40 layers = 960 MiB`; at the measured 14.45 GB/s link a 22% cold fraction (`h=0.78`, S=64)
is ~211 MiB/token = ~14.6 ms/token, i.e. a ~68 t/s UVA cold-read ceiling — still 2.8x today's 24.2 t/s but
short of the 91.8 t/s all-resident.  A **CPU-computes-the-misses** variant reads the same 211 MiB at ~40 GB/s
= ~5.3 ms, which fits *under* the GPU's all-resident compute (10.9 ms/token at 91.8 t/s); Strata's design
(design space C) is therefore not just a fallback — at our hit rates it could hide the cold set entirely,
which is why it should be kept as the Phase-1 comparison, not discarded.  Both are bounded by getting the
resident set computed on the GPU first.

### 2.4 What is still open (for the next session)

1. **Verify-width traces.**  §2.2 is W=1.  The campaign's open question #4 wants the same policy scored at
   the MTP verify width (`n_tokens` 1..8), where the used-expert set per step is wider and the hit rate per
   *reach* differs.  The instrument already records the band (`GGML_MOE_PROFILE_MAXTOK=8`); run a
   `draft-mtp` trace and re-score.
2. **Phase 1 build.**  A single-GPU slot arena (per-layer, LFRU) behind `device_alias()`, with the resident
   experts computed on the GPU and the cold set on the existing CPU path (the quickest measurable win), then
   the UVA/pointer-table variant.  Gate with the delivery's W=1..8 width-purity check and same-seed text.
3. **`qwen4exp` re-check.**  §1.6's Strata held-out profile was 0.645 at 48x512; re-run the trace on
   Flash-Next IQ4_NL when its lazy/PLE path allows a stable 2000-token decode (its `llama-bench` absolutes
   are not comparable — see §0).
4. **Profile source.**  The warm-start book should come from a per-workload profile or the delivery's own
   prompt set; the same-run `.pre`/`.post` split is the offline proxy, not a runtime dependency.

Artifacts added this session: `exp2-moe-routing-profiler.patch`, `policy_sim.py`, `policy-sim.csv`,
`policy-windows.csv`.  The 5 raw traces (~10 MiB each) and the ~16k-depth trace are regenerable with the
command in §2.1 and are **not** committed; the old `prof-*.csv` are the exp1 (triple-counted, order-free)
historical record and are left untouched.  None of this touches the delivery: it is `wip/`, env-gated, and
applies to `~/llama-decode` only.

---

## 3. REVISED IMPLEMENTATION PLAN (2026-09-28) — go-forward reference

**This supersedes §1.5 and the §0 phased plan.**  Maintainer sign-off 2026-09-28: proceed with the
`device_alias()` seam, keep the `-sm tensor` end state as the primary constraint, and treat the static
hard-pin as a lower-priority, no-code accelerator.

### 3.0 What changed from §1.5

| item | was (§1.5) | now (§2/§3) |
|---|---|---|
| policy | "a static/profile hot set, no eviction" | **LFRU** — admit every miss, evict min decaying count (LRU tie-break), halve counts every ~32 decode steps, per-layer slot ranges; static book is a warm start only |
| seam | a `ggml_cuda` slot arena fed from the pinned host source | **`(layer, expert) -> device_alias()` pointer table** at the `MUL_MAT_ID` weight-source seam; a CUDA arena is only the Phase-1 scaffold *behind that seam* |
| hard-pin | not in the plan | a **lower-priority session analyser** that emits a per-layer pin table for existing `-ot`/`-ncmoe` config (a warm start / no-code win, **not** the mechanism) |
| `-sm tensor` | "adapt later" | **the design target from the first line of code** |

### 3.1 North star / non-negotiable invariants (the "do not back into a corner" list)

1. **One abstraction, three geometries.**  The deliverable's decode MoE path must be the same code for
   1 GPU, `-sm layer` and `-sm tensor`; the only differences are `blob_bytes` (the *device's slice* of the
   expert) and the alias source.  If a change needs whole-expert-only blobs or a non-meta consumer, stop and
   redesign.
2. **Every blob address is a `device_alias()`.**  The kernel reads a per-`(layer, expert)` base pointer and
   never assumes the expert table is contiguous.  Under `-sm tensor` that alias may be a resident VRAM slot
   or a UVA view of the pinned host slice (this device's slice in both cases).
3. **Residency is keyed by `(layer, expert)`; the slot holds this device's slice** (axis-1 gate/up, axis-0
   down).  Never build a whole-expert-blob cache.
4. **Per-layer slot ranges** (Strata: a global pool measures ~3% at the same budget; per-layer turns it into
   21-70%).  LFRU as above; the static profile only seeds it.
5. **VRAM allocation fails soft and is visible.**  The arena is outside the compute reserve; a failed alloc
   disables the cache and warns (issue #33), and `--fit` must either count it or document the exclusion.
   **RESOLVED 2026-09-28 (1d): the arena is the lowest-priority consumer** - it is sized from what is
   actually free at sizing time (after `--fit`/KV/compute, minus a reserve), so it cannot over-commit and
   `--fit` does not need to know about it; fail-soft is the backstop.  See "### 1D RECORD".
6. **Width purity.**  `W = 1..8` must agree and same-seed text must be identical cache-on vs cache-off.  The
   resident/cold split must rejoin in router order and zero the complementary half deterministically.
7. **The cold path is never a per-token H2D of the `ffn_down` slice** (176-byte rows x 524288).

### 3.2 Phase 0.5 (lower priority): static block-pin analyser

A no-code win for users, and the same profile loader the cache needs for its warm start.  It does **not**
block the cache.

* Use `exp2` to collect routing for a workload (or the delivery prompt set), rank layers (and experts, for
  the cache) by reach frequency, and emit a config table.
* **Granularity is the whole expert tensor / layer, not the expert.**  `-ot` matches a tensor-name regex and
  `-ncmoe N` expands to per-block overrides for blocks `0..N-1` only (`llm_add_n_cpu_ffn_overrides`,
  `common/common.h`), so today's options cannot pin an individual expert.  The analyser's value over `-ncmoe`
  is that it can emit a **non-uniform, hotness-ranked subset of blocks** (pin the hottest layers, leave the
  rest), which `-ncmoe`'s first-N rule cannot express.
* Be explicit about the ceiling: §1.6 says a static set does not transfer across workloads (held-out
  0.24-0.38), and §2.2 confirms that even within one workload it collapses mid-run (0.83 -> 0.55 on the
  phase-switching prompt).  This is a warm start / no-code option, not the mechanism.  Ship it after the
  cache seam; the profile format is shared.

### 3.3 Phase 1 — single GPU, Qwen3.6-35B-A3B Q8_0 (the iteration vehicle)

> **Current status and the next three tasks are in the CURRENT HANDOVER at the top of this file.**  The
> slot-remap consumer is built and measured; H1 to H3 are the hardening steps.

Goal: keep `MUL_MAT_ID` decode on the GPU for the resident fraction, with the seam Phase 3 needs, and a
measured `tg` vs `h` curve.  Build order is 1a -> 1b -> 1c -> 1d; each is a separate commit + measurement.

**1a — seam + LFRU residency + split-by-router-index dispatch (cold on the CPU).**
* Per-device residency manager: per-layer slot arena (`S = MOE_EXPERT_CACHE_MIB / (n_layers x 3 MiB)` for
  Q8_0), `(layer, expert) -> slot`, LFRU (period ~32), optional static warm start.
* The `mul_mat_id` dispatch consumes a base-pointer table (resident slot vs cold alias) rather than assuming
  contiguity.  For 1a the cold alias is the existing CPU expert source, and the op is split **by router
  index** (Strata's `ExpertDispatch`): zero the resident rows of the CPU `parts`, compute the resident rows on
  the GPU, and combine by router index (the shared `mul_mat_id` combine already sums by routed `dst`).  This
  is the only shape that makes the resident fraction help without a per-token H2D.
* Relax the op-offload gate for a cache-active MoE: `get_op_batch_size(MUL_MAT_ID) = op->ne[2] = 1` at decode
  keeps the whole op on the CPU, but a resident expert *is* on the device.  The scheduler must be taught to
  offload the resident half and leave the cold half on the CPU — this is the main correctness surface.
* Gate: Strata-style `h`/fills/verify report; `test-backend-ops -o MUL_MAT_ID` green; same-seed text ==
  cache-off at W=1; `tg` vs `h` (and vs the 24.2 t/s CPU and 91.8 t/s all-resident bounds) at d0 and d16384.

**Phase 1a status (2026-09-28, `exp3-moe-expert-cache-phase1a.patch`, built + proven on `~/llama-decode`):**

* Delivered: `ggml/src/ggml-cuda/moe-expert-cache.{h,cu}` — the `device_alias()` seam
  (`moe_cache_alias_get`), the per-layer compact slot arena, LFRU (period 32, LRU tie-break), fail-soft
  `cudaMalloc` (a failed alloc disables that table and warns), lazy per-`src0` table registration, and an
  exit `h`/fills/evictions report.  Gated by `MOE_EXPERT_CACHE_MIB` (default 0 = inert, bit-identical).
* Driven from the **scheduler's op-offload seam**, not the CUDA op: a new backend iface hook
  `moe_cache_update(backend, weight, ids, n_used, n_tok)` (`ggml-backend-impl.h`) is called by
  `ggml_backend_sched_compute_splits`'s `copy_experts` with the **true host master** `input` and the
  host-readable routing.  That is the only place both exist (the op-offload redirect makes the op's
  `src0->data` point at the device `input_cpy`), and it runs **before** the backend's MoE fusion, so it
  covers the delivered fused path (the earlier per-op observer did not — the decode MoE is fused via
  `ggml_cuda_try_fuse`'s `mul_mat_id_glu_ops`, so `ggml_cuda_mul_mat_id` is skipped by default).  The
  CUDA backend implements it as `moe_cache_update_host()` and passes `ctx->stream()`, so the fill is
  `cudaMemcpyAsync` and **capture-safe** (a synchronous fill aborted CUDA graph capture; the scheduler's
  own expert copies are async for the same reason).
* Proven: `MOE_EXPERT_CACHE_SELFTEST=1` byte-verifies the resident slots — **PASS**
  (`slots=4 resident=4 hits=3 misses=13 fills=13 evictions=9`).  Live on the real path
  (`-ncmoe 99 GGML_OP_OFFLOAD_MIN_BATCH=0`, fusion default, `MOE_EXPERT_CACHE_FILL=1`): **120 tables
  (= 40 layers x gate/up/down), 6736 slots, 4047 MiB <= the 4096 MiB budget, `h=0.712` (22558/31680
  reaches) over 32 decode steps**, and multiple CUDA-graph captures complete cleanly.
* **Slot-remap consumer: DONE (2026-09-28).**  The op now reads the arena.  `moe_cache_update_host()`
  stages the slot-remapped ids (indexing the routing with the ids tensor's own strides, which matters: the
  top-k `ids` is a strided view, not contiguous), the scheduler skips its own expert copy when the backend
  takes the input over, and `ggml_cuda_mul_mat_id` re-dispatches through shallow `src0`/`ids` copies whose
  `data` points at the arena and the remap buffer.  `src0`'s `ne[2]`/`nb` stay at the full expert count so
  the dispatcher's kernel-family heuristics (and therefore the arithmetic) are unchanged; only the base
  pointer moves, and the remapped ids stay below `slots`.  Prototype detail: all backend fusions are stood
  down while the cache is active (the consumer is the unfused per-op path), so the A/B runs the cache-off
  side with `GGML_CUDA_DISABLE_FUSION=1` too.

  Measured (1 GPU, `-ncmoe 99 -fa 1 -sm layer`, `tg64`, one or three reps; cache 8 GiB unless noted):

  | config | delivered CPU MoE | no-cache GPU stream | cache |
  |---|---:|---:|---:|
  | Q4_K_M, d0 | 29.1 | 9.7 | **35.5** (+22 %) |
  | Q8_0, d0 | 24.5 | - | **30.8** (+26 %); 23.6 at 4 GiB |
  | Q8_0, d16384 | 23.7 | - | **29.7** (+25 %) |

  Live `h = 0.7895` (49266/62400 reaches) with the 8 GiB budget (8160 MiB arena, 7680 slots, 120 tables).
  Same-seed greedy text is **byte-identical** cache-on vs the no-cache GPU path (short gate
  `359ff4337837`, 573-char decode `a371535187a5`), and `test-backend-ops -o MUL_MAT_ID` is 4/4.  Q8_0 is
  break-even at 4 GiB: its expert slices are ~1 MiB, so ~22 slots/table at 8 GiB is where `h` is high
  enough to beat the CPU.

* **What is left for Phase 1a:** (a) replace the blanket fusion stand-down with a **cache-aware fused MoE**
  (or a targeted MoE-only disable) so the non-MoE fusions stay on; (b) the W=1..8 verify-width purity and
  MTP gates (the consumer handles `n_tok <= 8`, but acceptance and width purity are not yet re-measured);
  (c) a real budget/slot allocator (the current per-table split uses a fixed `MOE_EXPERT_CACHE_TABLES`
  estimate, and the larger `ffn_down` slice gets fewer slots than gate/up); (d) the `-sm tensor`
  per-device-slice geometry (Phase 3).  Then Phase 1b (UVA cold reads) removes the per-miss fill.

  **Phase 1b status (2026-09-28): DONE - mechanism and policy, gates green.**  See "### PHASE 1B RECORD"
  and "### PHASE 1B POLICY RECORD" at the top of this file.  `MOE_EXPERT_CACHE_COLD=uva` serves a
  non-resident expert in place from the pinned host alias (a cold id region + a `(cold_base, n_res_cold)`
  pair in `mul_mat_vec_q_moe`), and `MOE_EXPERT_CACHE_ADMIT=touch` serves a first-touch miss cold instead
  of evicting a proven resident.  Both are now the defaults.  Result: `tg1024` Q8_0 d0 **42.01** t/s at an
  8 GiB arena and **48.95** at 12 GiB, vs 24.79 for the delivered CPU path (+97 %); the doorkeeper also
  raises `h` (0.7128 -> 0.7630 at 4 GiB) by removing slot churn (278469 -> 15840 evictions).  A static
  resident set (`NOEVICT`) is measured clearly worse.

**1b — UVA cold reads (the parallel-offload hypothesis).**
* r15 already puts the host experts in pinned memory (`ROCm_Host`/`hipHostMalloc`).  Register the slice and
  take a device alias (`hipHostRegister` + `hipHostGetDevicePointer` / `cudaHostGetDevicePointer`);
  `device_alias()` returns the resident slot when resident, else the UVA alias, and the kernel reads cold
  experts in place — no per-token H2D, no CPU cold compute.
* **Maintainer hypothesis to test:** split-wise expert offload uses more of the PCIe lanes in parallel, the
  way the r16 prefill split did (2-GPU `-sm tensor -ncmoe` beats upstream's only 2-GPU option at every
  offload level, +91% to +148%).  Measure achieved aggregate PCIe bandwidth, not just t/s.
* A/B the two cold policies at equal `h`: **CPU-computes-the-misses** (1a) vs **UVA in-place** (1b).  §2.3
  predicts CPU-compute wins at our hit rates because it uses ~40 GB/s of host DRAM instead of ~14.5 GB/s of
  PCIe, but this must be measured, not assumed.  Keep the loser as the fallback.

**1c — verify width + purity.**
* Score the policy at `W = 1..8` with `GGML_MOE_PROFILE_MAXTOK=8` (the MTP verify band) — the used-expert set
  per step is wider, so `h` per reach differs from §2.2's W=1 number.  Ensure the resident/cold split is
  bit-identical at every width.
* Gate: `--spec-type none` vs `--spec-type draft-mtp` same-seed text; the
  `benchmarks/mtp-adaptive-methodology.md` protocol (acceptance > ~0.45 at pos 1, MTP >= plain at depth 3).

**1d - fail-soft + `--fit`.  DONE (2026-09-28) - PASS.**  Full evidence in "### 1D RECORD" below.
**Decision on `--fit`: the arena does NOT need to learn about it, because the arena is the
LOWEST-priority VRAM consumer** - it is sized from `cudaMemGetInfo()` at sizing time, i.e. after `--fit`,
the compute reserve and the KV cache have taken theirs, minus `MOE_EXPERT_CACHE_RESERVE_MIB` (default
1024 MiB).  So it can never over-commit, and a user who asks for more than is free simply gets less
(warned).  Proven: a 64 GiB and a 1 TiB request both clamped and produced byte-identical output.  A
failed allocation still fails soft as the backstop, and that is proven too by an induced-failure knob.

### 3.4 Phase 2 — 2 GPUs, `-sm layer`

Whole MoE layers are assigned to one device, so the cache is per-device and a slot holds a whole expert.  This
is the moe-cache-shaped case and should be the first multi-GPU checkpoint: it validates the same dispatch
seam with the simplest geometry.  Gate: per-device `h`/`tg`, the cross-device AR unchanged
(`GGML_CUDA_ALLREDUCE=hybrid`), same-seed text vs `-ncmoe`-only.

### 3.5 Phase 3 — 2 GPUs, `-sm tensor` (the end-goal geometry)

* Each device holds **its slice** of every expert; `blob_bytes` is the axis slice and the meta backend's
  per-device partial reduce is untouched.
* The residency manager is per-device; the pointer table feeds the meta backend's simple expert tensors.
* The cold path **must** be UVA: the `ffn_down` split is `nb[1] = 176` x `524288` chunks, so a per-token H2D
  is half a million tiny copies per layer.  This is R9V's mechanism (TP-sharded masters + pinned UVA cold +
  a slot cache) with our LFRU instead of `second_touch`.
* Checkpoint: the r15 pinning and r16 staging/op-offload prefill wins must not regress; measure prefill and
  decode together.

### 3.6 Phase 4 — Qwen3.8-Flash-Next IQ4_NL (93 GiB, `qwen4exp`)

Re-validate the whole thing on the real target, including the 48x512 geometry (re-measure `h` and re-tune the
LFRU decay period there), the QSA arm gates, and the lazy/PLE path (`--lazy-mode auto`; its `llama-bench`
absolutes are not comparable without the server/pre-warm path — see §0).

### 3.7 Deferred / later (explicitly not now)

* **2-level VRAM expert cache** (protected hot tier + warm tier): a good idea, but SLRU (the two-level
  recency variant) measured within ~0.01 of LFRU at every budget, and LFRU's decaying count is already a
  soft multi-level ranking.  Revisit only if a measured miss-cost breakdown shows a distinct hot set that one
  level cannot hold; the policy simulator is ready to tune it.
* **Prefetch / early router** (moe-cache): LFRU's short decay already captures the consecutive-token reuse;
  add prefetch only if a trace shows a predictable next-step set that admission latency misses.
* **Prune the staged prefill upload to used experts** (inherited follow-up 1) — prefill, not this campaign's
  decode path; leave it with the prefill work.

### 3.8 Explicitly NOT doing (corner-avoidance)

* No cache `ggml_backend_buffer_type` in the delivery interface (see §2.3).
* No whole-expert-blob-only cache shape.
* No per-token H2D of the `ffn_down` slice.
* No `--fit`-invisible allocation and no abort on arena OOM.
* No policy port from R9V/moe-cache that §2.2 did not measure as best (`second_touch` is dominated).

**First commit of Phase 1a:** the `device_alias()`/base-pointer plumbing + the LFRU residency manager on the
current CPU cold path, single GPU, behind `MOE_EXPERT_CACHE_MIB` (default 0).  Nothing else changes until that
is measured.
