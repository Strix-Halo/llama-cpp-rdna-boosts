# HANDOVER — the un-redirect fix for a safe MoE-arena *partial* shrink (TODO #42)

**Audience:** a fresh session.  Read this top to bottom before touching anything; §1-§3 are the
situation, §4-§6 are the investigation, §7-§9 are the mechanics.

**Status of the code at handover:** release **`v16-a55e952b8-r19`** (2026-10-06) is cut and pushed.  The
**fail-soft full-release guard** ships and is validated; the **partial shrink is implemented but
UNSAFE** and is *not wired* into the release (it is dead code plus this document).  The mission is to
make the partial shrink safe and then wire it.

> **RESOLVED (same day) -- see `ARENA-UB-TENSION.md` SS13.**  The partial shrink **is now safe**: three
defects were found and fixed, and none of them was the "un-redirect" this document hypothesised (that
framing was wrong -- the redirects are *launch-local stack copies*, so there is no persistent dangling
pointer to restore).  Continue reading only for the investigation method; the conclusions below are
superseded.  The brief version: (1) there is no single culprit allocator -- the compute buffer, the mmq
workspace pool and the Q8_1 cache arena each failed in turn, so the yield policy was moved to the one
choke point `ggml_cuda_device_malloc`; (2) `moe_cache_take_over` did not honour the "a partially-failed
cache falls back wholesale" invariant that the fusion guard assumes, so a fallback op read an
`input_cpy` the scheduler never populated; (3) the fused kernels hold the arena in their *launch
parameters*, so releasing it under an in-flight kernel is a GPU VM fault that ROCr turns into a
`SIGABRT` -- the shrink now synchronizes first.  Verified: 6/6 concurrent-prefill runs clean, 0 full
releases, 0 faults, hit rate holds ~0.965-0.976, dense 3-GPU coherence clean.

---

## 1. Where everything lives

| thing | path |
|---|---|
| delivery repo (this) | `~/llama-cpp-rdna-boosts` |
| fork checkout (build/test) | `~/llama.cpp`, branch `rdna-boosts`, build dir `build-rocm-r16` |
| release record | `release.json` (`v16-a55e952b8-r19`, tip `659080b4c`, tree `24bea75d`) |
| the campaign record | `wip/moe-cache-autosize/ARENA-UB-TENSION.md` (§0-§12.9 — read §12.5-§12.9 especially) |
| the MoE cache | `ggml/src/ggml-cuda/moe-expert-cache.{cu,h}` |
| the alloc guard | `ggml/src/ggml-cuda/ggml-cuda.cu` → `ggml_backend_cuda_buffer_type_alloc_buffer` |
| the scheduler/allocator | `ggml/src/ggml-alloc.c`, `ggml/src/ggml-backend.cpp` |
| the context transition hook | `src/llama-context.cpp` → `llama_context::process_ubatch` |

## 2. The problem being solved

TODO #42 is the "arena vs `-ub` tension".  Concretely, on a `llama-server` (2 GPU, `-sm tensor
-ncmoe 48 -ub 4096 -c 102400`, cache auto, MTP n3) the **second / concurrent prompt aborts**:

```
allocating 6780.30 MiB on device 0: cudaMalloc failed: out of memory
ggml-backend-meta.cpp:1799: GGML_ASSERT(bufs.back() != nullptr) failed
```

The chain (all measured, see §12.5-§12.7):

* `llama_context::sched_reserve` pre-sizes the compute buffer from a **measure graph**: target reserve
  **6564 MiB** (server) / 6636 MiB (cli).  The **runtime** prefill graph then needs **6780 MiB** — the
  measure graph carries a different *live-tensor set* (a "peak-tensor-set" difference, dominated by the
  850 MB staged expert table), so the reserve is ~**216 MiB short**.  It is **not** a size effect: a
  synthetic wider batch (`n_ubatch + 256` tokens) added ~360 MiB of layout and did **not** close it.
* The MoE expert-cache **arena** is sized from the leftover VRAM at the first full decode pass
  (`alloc_all_locked`: `avail = free - reserve`).  On a server the first prompt is short, so the compute
  buffer has not yet reached its runtime peak; a later/concurrent wider batch grows it **after** the
  arena owns the space.
* The growth is a **grow-in-place realloc**: `ggml_gallocr_reserve_n_impl` runs
  `ggml_vbuffer_free(old)` then `ggml_vbuffer_alloc(new)`, so it needs a *contiguous* block **216 MiB
  larger than the one it just freed**.  Extra free VRAM elsewhere does not help — which is why an arena
  "headroom" of 512/1024/2048/4096 MiB changed **nothing** (swept, §12.7), and why the failure is
  fragmentation-sensitive (2048 passes/fails at random).

## 3. What r19 already ships (do not regress it)

1. **Fail-soft full-release guard** — `moe_cache_release_arena()` called from a failed compute
   `cudaMalloc`: free the whole arena, warn, retry.  The run survives with the expert cache disabled for
   the rest of the run.  Validated (two concurrent long+short prefills, 0 aborts).  **This is the current
   behaviour and it is safe but wasteful — it gives up the entire cache.**
2. **Arena slot-count retry** — `alloc_table_locked` no longer drops a table to 0 slots on a failed
   `cudaMalloc`: it tries the requested count, then the exact max from `cudaMemGetInfo`, then a 0.95
   geometric descent.
3. **Per-turn arena hit rate** in the server log, next to the MTP acceptance line:
   `MoE arena = 0.9115 (727032 hit / 797580 reaches this turn), arena 36141.7 MiB`.
   Plumbing: `moe_cache_get_stats` → `ggml_backend_dev_moe_cache_stats` → `llama_moe_cache_stats`.
4. **Opt-in single-shot drop** — `LLAMA_DROP_COMPUTE_BUFFERS=1` (default OFF): drop the wide-prefill
   compute layout at the prefill→decode transition and re-reserve the verify width, so a wide `-ub` and a
   large arena coexist (`-ub 8192` cache-auto decode 45.6 → 75.5 t/s).  Not server-safe.

## 4. The mission: make the *partial* shrink safe

Goal: when a compute allocation cannot fit next to the arena, **stand down the few largest tables
only** (freeing a few hundred MiB — the shortfall is ~216 MiB), retry, and keep the rest of the cache —
instead of releasing all 36 GiB.

Implemented-but-unwired pieces:

* `stand_down_table_locked(table_t & t, int idx)` — mirrors the **proven** device-migration teardown:
  frees `t.arena` **and** `t.remap_dev`, zeroes `remap_cap`/`remap_n_used`/`remap_n_tok`, clears
  `slot_expert`/`expert_slot`/`slot_dev_host`, updates `g_arena_bytes`, sets `slots = 0`,
  `slot_dirty = true`, and **erases the table's `g_alias_to_id` entries**.
* `moe_cache_shrink_step()` — free the single largest table (by `slots * expert_bytes`) and return true;
  returns false and sets `g_enabled = false` when nothing is left.
* `moe_cache_shrink_arena(need_bytes)` — the batch version (unused; kept for reference).

Wiring (the change to make once safe) is in `ggml_backend_cuda_buffer_type_alloc_buffer`:

```c
if (err != cudaSuccess) {
    (void) cudaGetLastError();
    while (moe_cache_shrink_step()) {          // stand down the largest table ...
        err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
        if (err == cudaSuccess) break;         // ... and stop as soon as it fits
        (void) cudaGetLastError();
    }
}
if (err != cudaSuccess) { /* fall back to moe_cache_release_arena() */ }
```

## 5. The failure (what happens today with that wiring on)

With the shrink loop enabled, the server **dies silently** — client exit 52, **no `GGML_ASSERT`, no
`cudaMalloc` error, no ggml abort** — after ~28 stand-downs:

```
W moe_cache_shrink_step: stood down the largest table (layer=14 role=blk.14.ffn_down_exps.weight, 170.3 MiB) ...
```

So it is a plain segfault/illegal access, not an assertion.  The full release does **not** do this,
because it also sets `g_enabled = false`.

## 6. What we know about the redirect (SUPERSEDED -- see SS13 in ARENA-UB-TENSION.md)

> **This section's leading hypothesis was wrong.**  `src0_c`/`ids_c`/`gate_c` are launch-local *stack
> copies*; the arena address reaches the kernel as a launch parameter, and no persistent graph tensor
> holds it.  The real defects were (1) three different allocators each being the one that ran when the
> arena held the last free VRAM, (2) `moe_cache_take_over` not honouring the wholesale-fallback
> invariant, and (3) freeing the arena under an in-flight kernel.  The steps below are kept only as a
> record of the method that found them -- the *first* one (get a real backtrace) is what actually
> cracked it.

The scheduler has **already committed** to the arena before the failed allocation:

1. During graph build the scheduler calls `moe_cache_take_over(weight, weight_cpy, ...)`.  On true it
   sets `g_alias_to_id[weight_cpy] = id` and **skips the expert copy for that graph** (the consumer is
   supposed to read the arena instead).
2. The **op** (`ggml_cuda_mul_mat_id` and the fused gate/up/down kernels) resolves `weight_cpy` through
   `g_alias_to_id` **at launch** and reads `t.arena` (+ `t.remap_dev`, `t.slot_dev`, or the identity
   path) rather than `weight_cpy`'s own data.
3. The allocation failure happens later, inside `ggml_backend_sched_alloc_graph`.

**Leading hypothesis:** freeing `t.arena` in that window leaves something the *already-built / in-flight*
graph will dereference.  The stand-down erases `g_alias_to_id` for the table, so the op should fall back
— but the fallback reads `weight_cpy`, whose contents were never copied this round (the scheduler skipped
it), so this is at best stale, at worst a stale pointer.

**Checklist to confirm (in order):**

1. **Get a real backtrace first.**  This is the single most valuable step.  Run under gdb (see §7) and
   get the faulting frame — it will say immediately whether it is the op, the promotion, the policy
   flush, or the remap kernel.
2. **Bisect the stand-down.**  Comment out the `g_alias_to_id` erase, then the `remap_dev` free, then the
   `arena` free — with a build between each — to find which free causes the death.  A single run per
   variant costs ~1 minute.
3. **Check the device-side admission policy.**  `g_policy_dev[]` holds per-device arrays
   (`pd.ids`/`pd.desc`) that a kernel walks; a stood-down table's descriptor may still carry a
   `slot_dev`/`slot` pointer into the freed arena.  `moe_cache_progress_locked` /
   `moe_cache_policy_flush` and the `slot_dev`/`used_dev` fields of `table_t` are the sites.  If so, the
   stand-down must also drop the table from the policy arrays (or rebuild them).
4. **Check the remap/kernel path.**  `moe_cache_kslot_apply` guards with `t.arena != arena || t.slots <= 0`
   (line ~2650), but `moe_cache_build_remap_kernel` / the `devmap` path may not.
5. **Consider the simplest correct fix instead of a teardown: shrink at a graph boundary.**
   The failure window exists only because the shrink runs *inside* `sched_alloc_graph`.  Recording a
   "shrink request" and applying it at the **next** `process_ubatch` boundary (where
   `LLAMA_DROP_COMPUTE_BUFFERS` already does its `ggml_backend_sched_drop_buffers`), before the next
   graph is built, removes the whole class.  The catch: the very allocation that failed still has to
   succeed *this* graph — so you would need to grow the compute buffer *before* the arena is sized
   instead (i.e. the "defer the arena sizing until the compute buffer has reached its peak" variant).
6. **The principled alternative (arguably the real fix):** make the reserve an actual upper bound by
   reserving with the runtime's live-tensor set (the host-expert staging + MTP taps) — then nothing ever
   grows after the arena is sized and no shrink is needed at all.  See §12.6; the reserve is built in
   `llama_context::sched_reserve` via `graph_reserve(...)`, and the staging `input_cpy` tensors are
   produced by the scheduler's split path (`ggml_backend_sched_split_graph`), so the measure graph must
   go through the same offload decision.

## 7. Build, run, and reproduce

**Build** (the fork is already configured; `ccache` is on):

```bash
cd ~/llama.cpp
BUILD_DIR=build-rocm-r16 EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
# fast loop:
cmake --build build-rocm-r16 --target llama-cli llama-server -j 16
```

**Run the server** (2 GPU; the helper `/tmp/r42/serve.sh` may still exist, else inline):

```bash
HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib \
  ./build-rocm-r16/bin/llama-server \
  -m /llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  -md /llm/models/Qwen3.8/Flash-Next/IQ4_XS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  -sm tensor -ncmoe 48 -fa on -ctk q8_0 -ctv q8_0 -t 8 -c 102400 -b 4096 -ub 4096 \
  --lazy-mode off --load-mode auto --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 0.0.0.0 --port 8033
```

**Reproduce the crash** (the short prompt must transition to decode while the long prefill is still
running).  `/tmp/r42/conc.sh` does exactly this:

```bash
P1=$(head -c 40000 /tmp/pl_16k.txt | python3 -c "import sys,json;print(json.dumps(sys.stdin.read()))")
P2=$(head -c  2000 /tmp/pl_16k.txt | python3 -c "import sys,json;print(json.dumps(sys.stdin.read()))")
curl -s -m 900 .../v1/chat/completions -d "{\"messages\":[{\"role\":\"user\",\"content\":$P1}],\"max_tokens\":300}" -o /tmp/A.json &
sleep 4
curl -s -m 900 .../v1/chat/completions -d "{\"messages\":[{\"role\":\"user\",\"content\":$P2}],\"max_tokens\":100}" -o /tmp/B.json
wait
```

Success = `A=0 B=0`, `curl .../health` returns `{"status":"ok"}`, and the log has no `GGML_ASSERT`.

**Get a backtrace** (the death is silent, so this is the key tool):

```bash
# enable core dumps, or run the server under gdb:
gdb --args ./build-rocm-r16/bin/llama-server <the flags above>
(gdb) run
# ... reproduce ...
(gdb) bt full
```

## 8. Acceptance criteria for the fix

* The concurrent long+short repro completes with **0 aborts**; the server stays alive and coherent.
* The guard frees **a few hundred MiB**, not the whole arena — check the log for
  `stood down the largest table ... (170.3 MiB)` a handful of times rather than
  `released the MoE expert cache arena (35809.2 MiB)`.
* The **per-turn `MoE arena =` hit rate after the event stays high** (it should be ~0.95+, not ~0).
* `-ub 4096` cache-auto still runs (prefill/decode unchanged within noise); the dense gate
  (`llama-cli` 4B Q8_0, same-seed, `////`=0) is coherent; a `-n 2000` MTP run has acceptance ~0.92.
* Time-box the investigation; if the teardown proves deep, prefer §6.5/§6.6 (graph-boundary shrink or an
  exact reserve) over a large refactor of the device-policy/remap state.

## 9. Housekeeping

* **Never push upstream** (`ggml-org/llama.cpp`).  Only this repo and the personal fork
  (`git@github.com:stew675/llama.cpp.git`, branch `rdna-boosts`).  See `AGENTS.md`.
* WIP code stays out of `patches/` until it is validated and the maintainer approves a promotion.
* If you cut a release, follow `scripts/make-patches.sh` → `scripts/make-release.sh` →
  `scripts/validate-set.sh`, then tag + push, and refresh the fork's `rdna-boosts` branch
  (`apply-all.sh` on a clean base; tree must equal `release.json.tree`).
* Report findings to the maintainer **before** any delivery-affecting commit (AGENTS.md).
