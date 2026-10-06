# MoE expert cache / arena — campaign handover

**State at `v16-a55e952b8-r20` (2026-10-06).**  The cache arms and sizes itself (r14), the `-sm tensor`
staging corruptions are fixed (r16), the prefill staging work is in (r17/r18), and #42's **crash** half
is fixed (r19/r20: a 10 % HIP-only compute-buffer slack + a fail-soft arena yield at one allocation
choke point).  **Everything closed is in [`COMPLETED.md`](COMPLETED.md)** — read that instead of the
history if you need to know whether an old claim still holds.

**One item is open and it is a DoD miss, not a bug: TODO #42's second half.**  Start below.

---

# OPEN 1 (top of the list) — TODO #42(1): decouple the wide-`-ub` prefill layout from the arena

> **Session findings (2026-10-06): [`OPEN1-FINDINGS.md`](OPEN1-FINDINGS.md).**  The DoD is met on
> `llama-cli` by default (decode **78.7** / prefill **1683**; the drop is now cli-only, option B).  The
> r20 shrink bug is fixed (a shrunk arena table advertised the requested slot count -> over-read), the
> layer is re-sized as a unit, the draft cap defaults to 512, and the drop extra reserve defaults to 0.
> **Still open: a solid `llama-server` solution.**  The drop must stay off for a server -- a later wide
> prefill at `-ub 8192` needs a contiguous ~12.4-12.9 GB layout back and freeing the whole arena
> (288 tables, 40 GB) does not yield one (`-ub 4096` reclaims fine).  The server DoD needs the compute
> buffer chunked/VMM (OPEN 2).  Read that file first; candidate code `open1-candidate.diff`.

## The goal (the DoD, unchanged)

On this box (2× R9700, `-sm tensor -ncmoe 48`, cache auto, 16k prompt) we can have a **big prefill** or a
**big arena** but not both:

| config | prefill | decode | arena |
|---|---:|---:|---|
| `-ub 8192` | **1040-1080 t/s** (staging on) | 41-45 t/s | small |
| `-ub 4096` | 711-735 | **68.9-69.1 t/s** | large |
| Reddit R9V reference (slower HW) | 1166 | 57.3 | — |

**DoD: `-ub 8192`'s prefill *and* decode ≥ 68.9 t/s at 16k.**  Part (2) of the plan is done; **(1) is
not.**

## What is done (do not redo)

* **(2) The arena-reclaim retry is in** — r19/r20.  Any device allocation that runs short now frees the
  largest arena table (then the whole arena) and retries, so a later compute growth survives.  Validated:
  server concurrent prefills 7/7 clean, 0 full releases.
* **The crash is gone** — the `-ub 8192` cache-auto OOM (a `11765.52 MiB` allocation failing next to the
  arena) is fixed by the 10 % compute-buffer slack.  See `COMPLETED.md` §5.

## What is NOT done — the piece to build

**(1) Shrink the compute buffer to `max(decode, verify)` *before* the arena is sized.**

The mechanism, in one paragraph: `sched_reserve` sizes the compute buffer from a **measure** graph, and
the arena is then sized from `free - reserve`.  A wide `-ub` reserves a permanently large compute buffer
(3810 MiB at `-ub 2048` → **11339 MiB at `-ub 8192`**), so the arena gets what is left and decode falls
with it.  But the wide layout is only needed **during prefill**: the verify/decode graphs need much less.
If the wide layout is given back before the arena is sized, the arena can be large *and* the wide
prefill stays fast.

Why the obvious version is not enough (already tried, r19/r20):

* `LLAMA_DROP_COMPUTE_BUFFERS=1` (opt-in, single-shot) drops the wide-prefill layout at the
  prefill→decode transition and re-reserves the verify width.  `-ub 8192` decode **45.6 → 75.5 t/s** —
  so the mechanism works.  It is **not server-safe**: a *later* prompt needs the layout back, and with
  the arena holding the VRAM the allocation aborts.
* Doing it *after* the arena is sized does not help: the arena already owns the memory.
* A flat arena headroom does **not** help either — the failure is a realloc needing a block *bigger than
  the one just released* (swept 512/1024/2048/4096 MiB, no change).

## Design sketch (the maintainer's proposal, worth starting here)

A **deferred reallocation**, i.e. a `needs_reallocation` / "shrink pending" flag rather than acting
inside an allocation:

1. Do not free anything mid-allocation.  Record the need (and the target size) and let the current work
   finish.
2. At a **graph boundary**, with the device provably idle, free/re-reserve once, and resume.

Where it fits: the *between-graphs* case (the compute-buffer growth in `sched_alloc_graph`) is clean —
fail that graph's alloc, reclaim at the boundary, retry the alloc; this also removes the per-stand-down
`cudaDeviceSynchronize` the current yield has to do.  The *mid-graph* case (the workspace pool and the
Q8_1 cache arena fail inside a launch) cannot be deferred without aborting a half-launched graph, so the
choke-point yield must stay as the backstop.  This is all worked through in
[`FOLLOWUP-compute-arena-chunking.md`](FOLLOWUP-compute-arena-chunking.md) §6 of the old handover and in
[`ARENA-UB-TENSION.md`](ARENA-UB-TENSION.md) §12.5-§12.9.

## Read this before you change the margin

**r20's 10 % slack pulls *against* this DoD.**  A bigger compute buffer means a *smaller* arena
(measured: 63.8 % → 61.5 % residency).  It buys crash-safety; it costs decode headroom.  Once the wide
layout can be given back before the arena is sized, the margin should be **re-reduced** (or removed) —
otherwise we pay twice.  Treat the two as one design, not two independent knobs.

## Environment to reproduce

```bash
# build (ccache on; CMake >= 4.3 needs the quoted empty HIP flags)
cd ~/llama.cpp
BUILD_DIR=build-rocm-r16 EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
cmake --build build-rocm-r16 --target llama-cli llama-server -j 16     # fast loop

# the wide-ub decode measurement (2 GPU, cache auto, 16k prompt = prompts/code-python.txt x30)
HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib ./build-rocm-r16/bin/llama-cli \
  -m /llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  -md /llm/models/Qwen3.8/Flash-Next/IQ4_XS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  -sm tensor -ncmoe 48 -fa on -ctk q8_0 -ctv q8_0 -t 8 -c 32768 -b 8192 -ub 8192 \
  --spec-type draft-mtp --spec-draft-n-max 3 --reasoning off -n 2000 -f /tmp/pl_16k.txt \
  --seed 42 --temp 0 --no-display-prompt --single-turn
```

* `-n 2000` is the floor for a decode number; `-n 16` is fine for a *correctness* check.
* The A/B control for the r20 slack is `GGML_COMPUTE_BUFFER_MARGIN_PCT=0` — it reproduces the old abort.
* The single-shot drop is `LLAMA_DROP_COMPUTE_BUFFERS=1` (it is the thing being made server-safe).
* Never run benches in parallel; `-t 8` (the GPU IRQs own the top cores).  See `AGENTS.md`.

## Acceptance

* `-ub 8192`, cache auto, 16k: decode ≥ 68.9 t/s **and** prefill ≥ 1040 t/s, coherent (`////` = 0,
  same-seed greedy) **and** MTP-gated (`-n 2000`+; acceptance ~0.92).
* Server-safe: the concurrent long+short prefill test completes with 0 aborts, and a **later** request
  after a wide prefill does not abort.
* The arena yield should fire **rarely or never** in that configuration (it is the backstop, not the
  mechanism).

---

# OPEN 2 — compute-buffer chunking and VMM-backed buffers

Investigation only, nothing implemented: [`FOLLOWUP-compute-arena-chunking.md`](FOLLOWUP-compute-arena-chunking.md).
Two findings make it concrete:

* The compute buffer is **already** a chunked virtual buffer (`GGML_VBUFFER_MAX_CHUNKS = 16`, and the
  realloc trigger is per chunk) — but the CUDA buffer type reports `get_max_size = SIZE_MAX`, so it
  collapses to **one ~11.8 GiB `cudaMalloc`**.  Uniform chunk sizes would make arena and compute units
  interchangeable.  Caveats: the 16-chunk cap (the last chunk is unbounded), and `get_max_size` also
  feeds the *weight* buffers, so this must be a **compute-only** budget.
* HIP VMM is **off by default** (`GGML_HIP_NO_VMM=ON`) and the existing VMM pool cannot serve as a
  general allocator (its `free` never unmaps and it asserts LIFO).  The file opens with a **cheap
  one-build probe** — flip `GGML_HIP_NO_VMM=OFF` with the margin off and re-run the cli repro — that
  decides whether VMM is worth building at all before any of it is written.

---

# OPEN 3 — the prefill I/O wall

[`PREFILL-WALL.md`](PREFILL-WALL.md) (measurement only; keep it as-is).  Staging is +12 %, but the wall
is the **per-ubatch host-expert upload**: fitting `t = U + C·T` from the 3.7k/8k single-ubatch points
gives ~4300 t/s of compute and ~7.2 s/ubatch of upload, i.e. an effective H2D of **~7.6 GB/s of the
14.5 GB/s link, only partly overlapped**.  That makes the upload overlap — not the kernels — the next
prefill lever, and it is the natural companion to OPEN 1 (both are about the same per-ubatch cost).

---

# Parked — the meta segmented expert upload

For gemma4's fused `ffn_gate_up_exps` (segmented split layout, `n_segments`/`nr`):
`ggml_backend_meta_set_tensor_async` only handles a contiguous slice, so the MMQ tail pad that
`copy_experts` appends makes the range non-row-aligned.  A partial fix exists but hung
non-deterministically (one GPU spinning), so it needs a **deterministic repro first**
(`compute-sanitizer`, or the `GGML_CUDA_SPLICE_GATHER` / `GGML_META_PINHOST` force-modes).  Detail:
`archive/work/moe-expert-cache/item3-findings-session20.md`.

---

# Locked semantics (do not re-litigate)

These were settled by measurement and/or the maintainer; re-opening them costs a session:

1. **Staging mode = `stage_d2d`** (redirect off).  No VRAM difference vs the redirect, and the redirect
   is corruption-prone (`COMPLETED.md` §2).
2. **The arena is the lowest-priority VRAM consumer** — a smaller arena beats an abort, always.
3. **`-sm layer` has no Meta compute buffer** and therefore never hits the `-ub 8192` OOM; it is not the
   path to "both axes" (its whole-table staging threshold keeps staging off at wide `-ub`).
4. **Mixed K/V cache types stay hard-rejected**; `-ctk`/`-ctv` must match.
5. **A feature that helps is default-on**, with the env var as a *kill-switch*, not an opt-in
   (`AGENTS.md`).
6. **The arena sizing is one-shot** — it happens at the first full decode and is not re-evaluated.

---

# Directory contents

| file | what |
|---|---|
| `README.md` | this handover — the live work |
| [`COMPLETED.md`](COMPLETED.md) | the closed-work ledger (the campaign's own goal, #41, #43, Stage 1 item 1-3, Stage 2, the un-redirect resolution) + the list of deleted artifacts |
| [`ARENA-UB-TENSION.md`](ARENA-UB-TENSION.md) | the #42 design record — §0-§11 history, **§12-§14 = the r19/r20 work** (read §12.5-§12.9 and §13/§14 for OPEN 1) |
| [`FOLLOWUP-compute-arena-chunking.md`](FOLLOWUP-compute-arena-chunking.md) | OPEN 2 |
| [`PREFILL-WALL.md`](PREFILL-WALL.md) | OPEN 3 (measurement only) |
| `../../archive/docs/TENSOR-CORRUPTION.md` | the #41/#43 trail (closed) |
| `../../archive/docs/HANDOVER-unredirect.md` | the mis-framed un-redirect investigation (closed; the lesson is in `COMPLETED.md` §6) |

Related: `../host-pinned-buffer-crash/README.md` (the WIP index), `../moe-cpu-overlap/` (the other half —
making the *miss* path overlap), `../../GREEDY-PURITY.md` §19/§24/§25, `../../patches/README.md`
(block 06/13/15 notes), `../../COMMUNITY-CONFIG.md` (the manual config guide).
