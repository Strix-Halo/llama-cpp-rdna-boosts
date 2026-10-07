# HANDOVER — fix the MoE MMQ expert-table over-read **in the kernel**, not on the host

Status: **RESOLVED 2026-10-03 — see `RESOLUTION.md`; do not act on the plan below.**  The
investigation concluded that the gather's prefill "win" over staging was itself the corruption
(NaN routing skips work), and that the correct prefill is ~4× lower and about equal to staging.
The default was switched to the staging/host path.  The sections below are the original
investigation record and are retained for history.
Base: `v16-84e76d8a2-r30`. Tree `/home/stew675/llama.cpp` is **clean r30** (no local diffs).
Author of the diagnosis: session 2026-10-02 (r30, repeated-`/` follow-up).

> **One-line objective.** The routed-expert `MUL_MAT_ID` MMQ speculatively over-reads past each
> expert into the next expert's bytes. The host-resident-expert **gather** prunes the experts, so
> those bytes are stale/NaN and poison the tile (repeated `/`). r30 guards this from the host by
> zeroing every expert slot head — which is correct but costs ~4x prefill. **Fix it at the source:
> make the fast MMQ loader clamp (or zero) its out-of-range read when it is operating on a MoE
> expert table.** That removes the need for any host-side guard/pad/re-arm, preserves the fast
> config, and is the correct side of the problem.

---

## 0. Current state (read this first)

| item | value |
|---|---|
| release | `v16-84e76d8a2-r30` (`release.json.tip` `6bba985363599e8dd92290ca32a1fb15876bbaf2`) |
| fork checkout | `/home/stew675/llama.cpp`, branch `rdna-boosts`, HEAD `ff2e867cfed28d193d342088ec1cdb4a6f292557` |
| fork tree | `0fe48395051775079fb18041142e3f22dbf82a72` == `release.json.tree` (**clean, no diffs**) |
| build | `/home/stew675/llama.cpp/build-rocm` (gfx1201, ROCm 7.14.1, ccache) |
| running server | r30 binary on `:8033` (2-GPU `-sm tensor`, q8_0 KV — see §6 command) |
| loader script | `/tmp/r30repro/live.sh` (created this session; may be gone next session — recreate from §6) |
| repro prompt | `/tmp/r30repro/prompt.txt` (the HTML/Three.js prompt) |
| delivery repo | `/home/stew675/llama-cpp-rdna-boosts` (this file lives under `archive/work/moe-mmq-overread/`) |

The local build is the *only* thing that must stay in sync for this work; the delivery
`patches/0013` still ships the host-side guard. Any landed change here becomes a block-13 (and
possibly upstream) patch — see the delivery process in `AGENTS.md`.

---

## 1. The symptom and the reliable reproduction

**Symptom.** qwen4exp (`Qwen3.8-Flash-Next`) under `-ncmoe` + `MOE_EXPERT_CACHE_MIB` +
`-sm tensor` degenerates to a solid run of `/` (e.g. `reasoning_content` = hundreds of `////`
runs). Once it happens it is **persistent for the process**: afterwards *every* request — even
`"Say hi."` — comes back as `/`, regardless of seed or prompt.

**What actually triggers it: an aborted streaming request** (the WebUI navigating / re-sending
mid-stream). The prefill completes, decode streams, the client disconnects mid-decode; the next
request's prefill then over-reads stale `input_cpy` bytes.

**Deterministic repro** (server must be up, `:8033`):

```python
# 1) abort a streaming request after ~25 chunks
body = {"model":"Qwen3.8-Flash-Next-IQ4_NL",
        "messages":[{"role":"user","content":open('/tmp/r30repro/prompt.txt').read()}],
        "temperature":1.0,"max_tokens":4000,"stream":True,"seed":S,
        "chat_template_kwargs":{"reasoning_effort":"medium"}}
r = urllib.request.urlopen(Request(base, data=json.dumps(body).encode(), ...), timeout=20)
for i, line in enumerate(r):
    if i >= 25: break
r.close()
# 2) then send "Say hi." (non-streaming) and count '////'
```

Measured:

| build | corrupt count after abort |
|---|---|
| r30 | **24/24** (3 rounds x 8 probes), and persistent |
| r30 + host guard reproduced per gather / per graph / per request | **0/24** |

A single aborted stream poisons the process; it is not seed- or prompt-specific after that.

---

## 2. Root cause: why there is an over-read at all

The quantized `MUL_MAT_ID` for a host-resident expert table runs the **MMQ** kernel. The MMQ
weight-tile loader reads a full tile and, in the **fast path**, does **not** clamp the read to the
expert's bounds:

- `ggml/src/ggml-cuda/mmq-load-tiles.cuh` — every `ggml_cuda_mmq_load_tiles_<type>` reads
  `bxi = x + kbx0 + i*stride + kbx` and only clamps when `fallback` is true, e.g.
  `ggml_cuda_mmq_load_tiles_iq4_nl` (~line 1489):
  ```cpp
  for (int i0 = 0; i0 < I; i0 += nrows*nwarps) {
      int i = i0 + (...);
      if (fallback) { i = min(i, i_max); }          // <-- the only bounds guard
      const block_iq4_nl * bxi = (const block_iq4_nl *) x + kbx0 + i*stride + kbx;
      ...
  }
  ```
- `ggml/src/ggml-cuda/mmq.cuh` — the K loop calls it without a tail bound (~line 923, gate at 962):
  ```cpp
  for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
      load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
  ```
- `ggml/src/ggml-cuda/mmq.cuh` — which path is chosen (`mul_mat_q_case`, line 1924):
  ```cpp
  void mul_mat_q_case(...) {
      if (args.nrows_x % 128 == 0) { constexpr bool fallback = false; ... }   // fast, unclamped
      else                         { constexpr bool fallback = true;  ... }   // clamped
  }
  ```

So whenever the expert output dim is a multiple of 128 (qwen4exp is), the **fast unclamped** path
is used and the last tile reads past the expert. This is *deliberate*: for a normal weight the read
past the end lands in the tensor's **zero-padded buffer tail**
(`ggml_backend_cuda_buffer_init_tensor` clears the padding), so it is harmless and free. For
`MUL_MAT_ID`, `x` is one expert **inside** the table, so the read lands in the **next expert's
first bytes**.

Upstream accommodates exactly this on the host side. In `ggml/src/ggml-backend.cpp` `copy_experts`
(~line 2374):

```cpp
const size_t padding = std::min<size_t>(expert_size, 512);
const size_t padding_end = last_id < n_expert - 1 ? padding : 0;
// "copy a bit extra ... to ensure there are no NaNs in the padding of the last expert;
//  this is necessary for MMQ in the CUDA backend"
```

The **gather** (`moe-expert-cache.cu`, `moe_cache_gather_host`, line 1876) copies only the routed
experts, at their original offsets, and leaves every other slot as the graph allocator's reused
`input_cpy` bytes (often NaN). Hence the stale-head read and the `/`.

---

## 3. Why the host-side guard is the **wrong side** (with numbers)

r30's guard (`g_heads_zeroed` + `moe_cache_gather_zero_heads_kernel`, lines 185/1864/1942-1948)
zeroes the first `min(expert_bytes, 512)` bytes of **every** expert slot in `input_cpy`, once per
process. It is correct but:
- it is keyed on `(buffer, expert_bytes)` and never re-armed, so it does **not survive** the graph
  allocator reusing `input_cpy` for other tensors — the abort case.
- re-running it is expensive: the write is **512 bytes into each of the table's slots**, which are
  ~`expert_bytes` (MBs) apart over a ~450 MB table — a **scattered write over the whole table**.

Measurements (gfx1201, single GPU `-sm layer`, qwen4exp IQ4_NL, `-p 8192 -n 1024 -b 2048 -ub 2048`):

| variant | pp8192 t/s |
|---|---:|
| **r30** (guard once, never re-armed) | **2668** |
| guard re-run per gather (in-kernel own-slot zero) | 414 |
| guard re-run per graph (separate all-slots zero kernel) | 690 |
| guard re-run **once per request** (decode→prefill reset) | 674 |
| bitmap-tail (few writes) **+** old all-slots zero | 682 |
| guard removed entirely | ~2670 |
| **the same zero kernel launched 94x with `nbytes = 0` (no writes)** | **2643** |

**Conclusion:** the cost is the guard's **scattered writes**, not its launches and not its
instruction count. The zero kernel itself is 0.006 ms host / 0.01 ms device; yet its presence drags
the subsequent gather from ~5.5 ms to ~16.7 ms. r30 only survives because it runs the guard **once**,
in the unmeasured warmup, and never again.

2-GPU `-sm tensor` gather-path `pp1024` (`MOE_EXPERT_CACHE_MIB=8192 ... -ctk q8_0 -ctv q8_0`):
r30 = **697**, guard re-run = ~530.

Everything tried on the host side is a workaround for the fact that the kernel reads where it
should not. Padding on the host means every routed expert — and, to be safe, every slot — must be
padded on every path, i.e. a permanent tax for a read that the kernel should not issue.

---

## 4. What was tried this session and why each was rejected

1. **Re-run r30's all-slots zero per gather / per graph / per request** — correctness **0/24**, but
   prefill **~4x slower** (numbers above). The decode→prefill reset fires exactly once (`DBGRESET
   n=1 zeros_since_last=94`), so it is not "still firing"; the writes themselves are the cost.
2. **In-kernel own-slot conditional zero** (`prev_used && !used`, combined scan) — correctness ok,
   prefill 682. The combined scan (removing `break`) and/or the added code hurt the (occupancy- or
   launch-latency-bound) gather.
3. **Separate bitmap + tail kernels, few writes, after the gather** — prefill 682. Extra kernels in
   the gather path are themselves costly here.
4. **Force the clamped loader (`fallback = true`) for MoE** (`mul_mat_q_case`, condition
   `args.ids_dst != nullptr`) — **this is NOT a correctness fix.** With the guard removed it still
   gave **24/24**; it only clamps the **row** index, not the K tail. It *was* a free perf win
   (1-GPU pp8192 2637 vs 2668; 2-GPU gather pp1024 **1902** vs 697), but `fallback` changes the
   loader **and** `vec_dot` **and** `write_back` **and** the whole config
   (`ggml_cuda_mmq_get_util_funcs` / `ggml_cuda_mmq_get_config`), is only validated for this one
   shape, and is a broad change to every MoE model. **Reverted.** The unclamped fast path exists for
   a reason and should not be blanket-disabled.
5. **Nuking a clobbered arena with NaN to test** — the arena is fully seeded before the decode band,
   so it is not the issue; the guard (prefill `input_cpy`) was.

Do not repeat 1-4 as "fixes"; treat them as evidence for the direction.

---

## 5. The correct fix: make the fast loader pad itself **when flagged**

Keep the fast config/`vec_dot`/`write_back`; change only the **loader's out-of-range read** for the
MoE case.

**Design options (prefer the last one if it is viable):**

- **(a) Clamp the address to the expert's last valid block.**
  ```
  off  = kbx0 + i*stride + kbx;
  off  = min(off, (int64_t) nrows_x * stride - 1);   // last block of THIS expert
  bxi  = (const block_<t> *) x + off;
  ```
  `nrows_x` is recoverable from `i_max` already passed to the loader.
- **(b) Zero the out-of-range read instead** (no memory access, pure register):
  ```
  if (off > last) { v = 0; }         // and the scale/df load likewise
  ```
  Preferable if the compiler likes it; it touches no memory.
- **Flag plumbing:** add a compile-time parameter to the loader variants only (e.g.
  `template <ggml_type type, int J, bool fallback, bool moe_pad>`), **not** to `vec_dot`/`write_back`.
  Select `moe_pad = true` only on the MoE dispatch. The kernel already has `ids_dst != nullptr`
  / `expert_bounds != nullptr` at runtime, so the dispatch can choose the clamping funcs while
  keeping the fast arithmetic. Watch the build-time cost: the loaders are instantiated per
  `(type, J, fallback)` across the `template-instances/*.cu` files; a new axis multiplies them.
  If the build cost is unacceptable, a runtime `load_tiles` selection inside
  `ggml_cuda_mmq_get_util_funcs`'s returned struct is an alternative (two function pointers, pick on
  `args.ids_dst`).

**Why this should be cheap.** `load_tiles` runs inside `#pragma unroll` loops and the out-of-range
case only ever occurs on the **last** K/row tile, so the clamp can be confined to that iteration
(or a single `min`/predicate per load). This is *not* the same as the `fallback` config swap.

**What it buys:** no host guard, no `min(expert_size, 512)` pad dependency for the gather, no
re-arm, no clobber sensitivity. The scheduler's host `copy_experts` pad can stay (harmless, and still
needed for the non-gather host path), but the gather becomes padding-free.

**If the kernel change is judged too broad to land**, the runner-up is a **cache-owned, padded,
never-reused gather destination** (zero the inter-expert gaps once at allocation and redirect the op
to it, like the decode arena). Do **not** fall back to a host re-arm.

---

## 6. Environment and exact commands

Host: 3x AMD Radeon AI PRO R9700 (gfx1201), ROCm 7.14.1, 16-core Ryzen 9950X3D.
`rocminfo`: **`XNACK enabled: NO`** (so a non-pinned host pointer read from the GPU **faults**, it
does not silently corrupt). `/usr/local/bin/pin_gpu_irqs.sh` pins GPU IRQs to the top N cores.

Model: `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf`
(a symlink tree; IQ4_NL/IQ4_XS/IQ3_XXS are all available).

Server (the user's config; `live.sh`):

```bash
export HIP_VISIBLE_DEVICES=1,2
export MOE_EXPERT_CACHE_MIB=8192
export MOE_EXPERT_CACHE_DEVMAP=1
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib
exec /home/stew675/llama.cpp/build-rocm/bin/llama-server \
  -m /llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf \
  --alias Qwen3.8-Flash-Next-IQ4_NL --fit off --top-k 20 --port 8033 --threads 7 --parallel 1 \
  --top-p 0.95 --min-p 0.001 --verbosity 3 --host 0.0.0.0 --cpu-strict 1 --predict 98304 \
  --cpu-range 1-7 --threads-http 4 --load-mode none --cache-ram 16384 --ctx-size 204800 \
  --flash-attn auto --temperature 1.0 --batch-size 2048 --ubatch-size 2048 --n-gpu-layers all \
  --no-kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --ctx-checkpoints 64 \
  --image-min-tokens 1024 --cache-idle-slots --reasoning-budget 65536 --reasoning-preserve \
  --checkpoint-min-step 4096 --lazy-mode off --n-cpu-moe 48 --split-mode tensor \
  --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

Prefill A/B (single GPU, matches `COMMUNITY-CONFIG.md`; r30 baseline **2668**):

```bash
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=12288 MOE_EXPERT_CACHE_DEVMAP=1 \
  ./build-rocm/bin/llama-bench -m "$IQ4" -ngl 99 -sm layer -ncmoe 48 -fa 1 -lzm off -lm none \
  -t 8 -p 8192 -n 1024 -b 2048 -ub 2048 -r 2
```

2-GPU gather path (r30 baseline **697**):

```bash
HIP_VISIBLE_DEVICES=1,2 MOE_EXPERT_CACHE_MIB=8192 MOE_EXPERT_CACHE_DEVMAP=1 \
  ./build-rocm/bin/llama-bench -m "$IQ4" -ngl 99 -sm tensor -ncmoe 48 -fa 1 -lzm off -lm none \
  -t 7 -p 1024 -n 0 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -r 10
```

Build: incremental `cmake --build build-rocm --target llama-server llama-bench -j 16`.
Note: editing `mmq.cuh` recompiles the per-type `template-instances/mmq-instance-*.cu` (~70 s);
editing `moe-expert-cache.cu` is a few seconds.

There is a **second, unrelated** issue observed this session: the load sometimes aborts with
`Memory access fault by GPU node-N ... Page not present` in the warmup
`mul_mat_vec_q_moe<IQ4_NL,...>`. That is the cache's UVA cold read when the host-expert buffer is
not actually pinned: `ggml_backend_cuda_host_buffer_type_alloc_buffer` silently falls back to a
pageable CPU buffer when `cudaMallocHost` fails, and `bind_host_dev_locked`
(`moe-expert-cache.cu:289`) then does `t.host_dev = t.host`. With `XNACK=NO` that faults. It is
flaky and independent of the corruption; file it separately. (`-lm mlock` does **not** help — mode 2
implies `use_mmap=false`, so it mlocks nothing.)

---

## 7. Validation plan for the kernel-side fix

Correctness gates, all on the clean r30 + the kernel change:
1. **Abort repro** (§1) → must be **0/24** (this is the whole point). Also re-run a few hours of
   normal WebUI use.
2. **No guard, no pad**: temporarily remove r30's `g_heads_zeroed` write (keep code, set
   `nbytes=0`) and confirm 0/24. This proves the kernel fix replaces the host guard.
3. **Byte-identity**: 2-GPU `-sm tensor -ncmoe 0` == cache-on `-ncmoe 99 MIB=8192` == `de8be4d0c90c`;
   1-GPU `-sm layer -ncmoe 99` == `15038c19ddc8`. A clamp/zero of out-of-range reads must not change
   any *in-range* result, so these must hold bit-for-bit.
4. `scripts/gate-qwen4exp-quant-coherence.sh` (IQ3_XXS/IQ4_NL/IQ4_XS/Q4_K_M; 0 `////`, gather ON ==
   OFF).
5. `test-backend-ops -o MUL_MAT_ID` (929/929) and `-o MUL_MAT` (1297/1297).
6. **Perf**: 1-GPU pp8192 and 2-GPU gather pp1024 must stay at/above the r30 baselines (2668 / 697).
   If the 2-GPU figure keeps the ~1902 from the reverted `fallback` experiment, even better — but do
   not chase it at the cost of correctness.
7. **Breadth** (because MMQ is shared): qwen35moe (`Qwen3.6-35B-A3B` Q8_0 / Q4_K_M) prefill and a
   dense model, per `benchmarks/mtp-adaptive-methodology.md`. The `fallback` experiment regressed
   nothing here, but the fast path was kept for a reason — measure.
8. Warning-free clean build.

---

## 8. Key files / lines (r30)

| file | symbol | line |
|---|---|---|
| `ggml/src/ggml-cuda/mmq.cuh` | `mul_mat_q_case` (fast/clamped selection) | 1924 |
| `ggml/src/ggml-cuda/mmq.cuh` | `load_tiles(x, ..., offset_x + kb0, ...)` K loop | 923 (gate: 962) |
| `ggml/src/ggml-cuda/mmq.cuh` | `ggml_cuda_mmq_get_util_funcs` (loader/vec_dot/write_back per `fallback`) | ~556, ~1523 |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | `ggml_cuda_mmq_load_tiles_iq4_nl`, `if (fallback) i = min(i, i_max)` | 1489, 1526, 1547 |
| `ggml/src/ggml-backend.cpp` | host pad `min(expert_size, 512)` in `copy_experts` | 2374-2385 |
| `ggml/src/ggml-backend.cpp` | `sched_input_gatherable` / `SCHED_GATHER_TABLE_MIN_BYTES` | 1976 / 1974 |
| `ggml/src/ggml-cuda/moe-expert-cache.cu` | `g_heads_zeroed` / zero kernel / `moe_cache_gather_host` | 185 / 1864 / 1876-1948 |
| `ggml/src/ggml-cuda/moe-expert-cache.cu` | `moe_cache_gather_kernel` (destination = `e*expert_bytes`) | ~769 |
| `ggml/src/ggml-cuda/moe-expert-cache.cu` | `bind_host_dev_locked` (UVA fallback, separate crash) | 289 |

## 9. Postscript — the principle to carry forward

The over-read is a property of the kernel, so the fix belongs in the kernel. Padding from the host
is a permanent, unconditional tax to compensate for a read the kernel should not issue, and it is
fragile (it only works if *every* writer of the buffer re-establishes it). Make the fast path
self-aware: when it is reading a MoE expert table it must not read another expert's bytes. Keep the
arithmetic fast; fix the load.
