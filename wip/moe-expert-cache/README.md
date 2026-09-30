# MoE expert cache — README (current state + fresh-session handover)

`wip/` campaign.  Decode-side MoE expert caching for models whose experts do **not** fit in VRAM: keep
a hot subset resident on the device and serve the rest cheaply, so `-ncmoe` decode approaches the
full-residency `-ncmoe 0` throughput as the cache grows — without giving up `-sm tensor`.

**This file is the live handover.**  The verbose, dated records and the original design briefs are in
**`WORKLOG.md`**; the "Completed work" index at the bottom points at them.  Do not edit the WORKLOG
records in place — append a new dated entry and add a one-liner to the index.

> Status: **not part of the delivery.**  Everything here is `wip/`, applies only to the campaign
> worktree `~/llama-decode`, and is default-OFF (`MOE_EXPERT_CACHE_MIB` unset) until it passes a
> promotion gate.  Nothing in the delivery `patches/` is touched by this campaign.

---

## 0. NEXT SESSION — remaining work (B1 and item 1 DONE; the scheduler half of B1 is delivery r26)

> **Handover state (2026-09-30, session 18).**  **B1 and item 1 are DONE; the campaign is on the delivery
> `v16-84e76d8a2-r26`** (tip `0d58404e1`).  Branch **`wip-moe-devmap-v2`**; full patch
> **`exp19-moe-expert-cache-r26-b2-tensorpad.patch`** (clean-applies to r26; supersedes `exp18`).  The
> scheduler half of B1 — the per-split events default, the host-weight staging gate and the async
> split-input copy — is now the **delivery block-06 amendment in r26**, so the campaign no longer carries
> it.  Rebased single-card 8K prefill on Qwen3.8-Flash-Next IQ4_NL, `MIB=24576 DEVMAP=1`:
> `-ub 512/1024/2048/8192` = **`~1044/1627/1761`** t/s (r25 delivery alone: `~233/362/425/870`).  Gates
> green: `15038c19ddc8` (1-GPU `-sm layer`), `de8be4d0c90c` (2-GPU `-sm tensor`), `MUL_MAT_ID` 929/929.
>
> **Item 1 (session 18): the tensor-split gather is re-enabled and correct.**  The session-16 attribution
> was wrong — the slice geometry was always right; the qwen4exp `////////` was the pruned gather leaving
> the MMQ's 512-byte expert-table **tail over-read** with the reused `input_cpy`'s stale/NaN bytes.  The
> gather now reproduces the host path's tail pad.  Verified: IQ4/IQ3 oracle byte-identity, the restored
> `-sm tensor` small-ub prefill win (`pp2048 -ub 512` 609 -> **724 t/s, +18.9 %**), and the full 12k-token
> coherence gate on the previously-corrupt model (rc=0, 13 sections, `## Conclusion`).  Detail:
> `WORKLOG.md` 2026-09-30 (session 18).
>
> **The two remaining items are below.**  Neither is a correctness bug; each is measured headroom.  Follow
> the campaign rules: opt-in first, keep the host path as the A/B reference, run the byte-identity + width
> purity + `MUL_MAT_ID` + MTP + coherence gates before any default flip, and a negative result with
> attribution is a valid outcome.

### 1. Fix the tensor-split gather slice geometry — ✅ DONE (session 18)

**The geometry was never the bug.**  `off += simple_tensor->nb[ss.axis + 1]` is the correct cumulative
per-device source offset for the single-segment `nr[0]==1` split the delegate enforces, for both the axis-1
gate/up (`nb[2]`) and the axis-0 `ffn_down` (`nb[1]`) layouts.  A D2H of every routed expert on every
device of the 3-device qwen4exp split matched the host master byte-for-byte (`bad=0`).

**The real cause was the missing MMQ expert-table tail pad.**  The scheduler host path pads
`min(expert_size, 512)` bytes past each routed group's last expert ("necessary for MMQ in the CUDA
backend"); the pruned gather did not, so the MMQ over-read saw the reused `input_cpy`'s stale/NaN tail
(NaN on qwen4exp).  The fix fills the first `pad` bytes of the next expert's slice after every routed
expert (`exp19`); `GGML_META_GATHER_NOPAD=1` restores the buggy copy for A/B.  `.moe_cache_gather` is
re-enabled on the Meta iface and default-ON (`GGML_SCHED_DEVGATHER=0` opts out).  The unsplit arm is
unchanged (the pad writes the same bytes it would anyway).

**Gates.**  `de8be4d0c90c` / `15038c19ddc8`; 3-device IQ3_XXS cache-on == `-ncmoe 0` oracle; deep coherence
(12k-token essay, rc=0, 13 sections, `## Conclusion`); `MUL_MAT_ID` OK; small-ub `-sm tensor` prefill win
back (**+18.9 %**).

### 2. B3 — fuse the slot lookup into the MoE ids read (~2-3 % decode)

**Self-contained work brief: `B3-brief.md`** (golden code map with tip-`2376ac6cf` line numbers, the exact
change, the traps, and the gates).  Summary:

Removing the remap kernels means the hot mmvq / fused kernels must take `slot_dev` + `n_res` + the raw
`ids` and do `slot[ids[i]]` in-kernel, **and** the `used_dev` routing record (which the batched device
policy replays) must move into those kernels too — the gate/up/down tables each need their own used list.
That is a template/dispatch change across the mmvq + fused paths plus the used-list accounting, not just a
lookup.  Session 11b measured the remap kernels at `0.27 ms/token` with the GPU ~29 % busy and found
halving the launches did not move throughput, so the reward is **~2-3 % decode** — weigh it against the
risk.  Gate it and A/B at a fixed `h`.

### 3. The user graph-input copies still hold a host `event_synchronize` (~5 s/pass headroom)

The r26 amendment made the **generic** host->device split-input copy asynchronous, but the
`GGML_TENSOR_FLAG_INPUT` branch still does `ggml_backend_event_synchronize(sched->events[...])` plus a
synchronous `ggml_backend_tensor_copy`.  On a merged routed-MoE band those graph inputs (`inp_pos`,
`attn_inp_k_idxs`, ...) are re-copied on every split, and the host block measured **~5 s/pass at `-ub 8192`**.
**Re-measured (session 18, before any attempt):** 3-device IQ4 `-sm tensor -ncmoe 48 -p 8192 -ub 8192`,
`GGML_SCHED_SYNCDBG=1`: `SCHEDUPLOAD input_loop=290 7395.9ms` against an 18.8 s pass (434.8 t/s), with
`set_async=0` and `get_async=8` — i.e. the expert copies are no longer the cost (the gather covers them),
the ~7.4 s is the 290 per-split `ggml_backend_synchronize(split_backend)` (full 3-device drain; 5964 syncs,
1.04 s) plus the per-split synchronous mirrored user-input copies through
`ggml_backend_meta_buffer_set_tensor` (3 H2D per input per split).  The fix is a scheduler/backend change
(match the source lifetime to the copy lifetime), not a kernel tweak.
They are **semantically distinct per split** (12 `input_cpy` allocations per pass), so the "copy once" set
tried in session 17 could not apply.  A safe async conversion needs the source lifetime pinned until the
copy completes — the naive async `set_tensor_async` on the user branch **crashed** in session 17
(`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` at `-ub 2048`, because the 4 ubatches overwrite the source
between the enqueue and the copy).  Options: pin the graph-input buffers once; or add a per-input staging
buffer with an event-guarded lifetime; or prove the source is stable per ubatch and gate the async copy.

### Code map

| concern | location |
|---|---|
| staging ring issue/drain | `ggml/src/ggml-backend.cpp`: `sched_stage_issue`, `sched_input_gatherable`, `sched_stage_ev`, the input-loop host-weight branch (`copy_experts`, the devgather call) |
| gather kernel + slot maps | `ggml/src/ggml-cuda/moe-expert-cache.cu`: `moe_cache_gather_kernel`, `moe_cache_gather_host`, `moe_cache_take_over` / `moe_cache_get_table` / `slot_dev` |
| meta slice geometry (item 1, done) | `ggml/src/ggml-backend-meta.cpp`: `ggml_backend_meta_moe_cache_gather` (re-enabled) — the same offset formula as `ggml_backend_meta_moe_cache_update`; the CUDA side pads the MMQ tail in `moe_cache_gather_kernel` |
| cache iface / adapters | `ggml/src/ggml-backend-impl.h`, `ggml-cuda.cu` (`ggml_backend_cuda_moe_cache_*`), `ggml-backend-meta.cpp` |
| MoE ids read (item 2) | `ggml/src/ggml-cuda/mmvq.cu` (`mul_mat_vec_q_moe*`), the fused gate+up and down-fold call sites in `ggml-cuda.cu` |
| user graph-input copy (item 3) | `ggml/src/ggml-backend.cpp`: the `GGML_TENSOR_FLAG_INPUT` branch of `ggml_backend_sched_compute_splits` |

### Quick start (copy/paste)

```sh
cd ~/llama-decode && git switch wip-moe-devmap-v2 && git log -1        # expect 2376ac6cf (r26 base + B1 + item 1)
cmake --build build-rocm --target llama-cli llama-bench -j 16

IQ4=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
IQ3=/llm/models/Qwen3.8/Flash-Next/IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
MTP=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
export HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib

# B1 prefill (now ~1044/1627/1761 t/s at ub 512/2048/8192), 1 GPU:
MOE_EXPERT_CACHE_MIB=24576 MOE_EXPERT_CACHE_DEVMAP=1 ./build-rocm/bin/llama-bench \
  -m $IQ4 -ngl 99 -ncmoe 48 -sm layer -fa 1 -lzm auto -lm none -t 8 -p 8192 -n 0 -b 8192 -ub 2048 -r 3 -o jsonl

# item 1 (fixed): 3-device tensor-split gather must equal the gather-OFF text on a >8-token prompt
P=$(cat ~/llama-cpp-rdna-boosts/prompts/reasoning.txt)
for G in 0 1; do HIP_VISIBLE_DEVICES=0,1,2 GGML_SCHED_DEVGATHER=$G \
  ./build-rocm/bin/llama-cli -m $IQ4 -ngl 99 -ncmoe 48 -sm tensor -fa 1 -lzm auto -lm none -t 8 -c 4096 \
  --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt -p "$P" -n 8 > /tmp/g$G.out 2>/dev/null; \
  python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/g$G.out; done   # both hashes equal

# 3-device transparency oracle (IQ3_XXS fits -ncmoe 0): cache-on == cache-off -ncmoe 0
```

### Traps (do not re-derive)

* The **tensor-split gather is ENABLED** (`.moe_cache_gather` on the Meta iface; `GGML_SCHED_DEVGATHER=0` opts out).
  The **real** qwen4exp corruption was the missing MMQ expert-table **tail pad**, not the slice geometry — the
  pruned gather must fill the next expert's first `min(expert_bytes,512)` bytes (`GGML_META_GATHER_NOPAD=1`
  restores the bug).  Do not "fix" the `off += nb[axis+1]` accumulation; it is correct.
* The gather is **not** the bottleneck for bytes — it runs at ~link speed; the gap is volume + residency.
* `llama-cli` needs `--single-turn --no-display-prompt`; never `-v` for the hash runs; pin `-t 8`; never
  run parallel benches.
* Cache-on vs cache-off at `-ncmoe` differ because cache-off runs the MoE on the **CPU** and cache-on on
  the **GPU** — the transparency oracle is `-ncmoe 0`, not `-ncmoe 48/99`.
* The campaign patch base is now **r26** (`0d58404e1`); `exp17` is r25-based and stale.

### Status recap (what is done)

* **B1** prefill residency + upload overlap — **DONE**; the scheduler half is **delivery r26**.
* **B2** device gather — **DONE**: unsplit AND tensor-split arms default ON (item 1 fixed the tensor-split
  corruption — the missing MMQ expert-table tail pad — and re-enabled the Meta delegate; `+18.9 %`
  `pp2048 -ub 512`).
* **B4** qwen4exp / Qwen3.8-Flash-Next — **complete** (see `b4-single-gpu-iq4nl.md`).
* **B5** closed; **B6** prompt-routing seed done.
* Current patch sequence: `exp17` (r25) -> `exp18` (r26 rebase) -> **`exp19`** (session 18: item 1, the tensor-split gather tail pad + re-enable).

---

## Device-side admission policy — IMPLEMENTED (session 12; keep the brief for the design rationale)

> **Status (2026-09-29, session 12): DONE as an opt-in win.**  `MOE_EXPERT_CACHE_DEVPOLICY=1` implements
> exactly the design below (device LFRU + in-kernel copy; the one deviation is that the fill is performed
> by the policy kernel from the pinned host alias rather than by a host-driven fill list, which removes the
> fill-list readback and the per-token sync).  Measured: `forced-devmap MIB=9344` 86.35 -> **89.74**,
> `MIB=1024` 46.73 -> **50.87**, `MIB=4096` flat; byte-identity/width-purity/MUL_MAT_ID/coherence all green.
> Branch tip `2632f6011`, patch `exp13-moe-expert-cache-r25-devpolicy.patch`.  **Still open:** the staged
> self-test (step 2 below - synthetic sequence, kernel vs `access_locked` victim parity), the depth-16384
> arena sweep, the MTP/default-flip promotion gates, and option 3 (fuse the slot lookup into the MoE ids
> read).  The text below is the original brief; the staged plan's steps 1-3 and 5 are done, step 2 (the
> self-test) is not, and step 4 (the batched launch) was folded into step 3 (the descriptor array).
> This is the one open performance item.  Everything below is self-contained; line numbers are for branch
> `wip-moe-devmap-v2` tip **`ceea0cfb6`** (delivery r25).  The user's own note: they believe we are close
> to the limit given **4 lanes of PCIe bandwidth** — so the goal is to *close this last point cleanly*, then
> move to the qwen4exp end-goal model.  Do it **opt-in first** (`MOE_EXPERT_CACHE_DEVPOLICY=1`, default off),
> keep the host path as the A/B reference and the fail-soft fallback, and flip the default only if the arena
> sweep wins.

### Goal

Move the **LFRU admission + eviction + fill decision** from the host (`access_locked`) to the GPU, and read
back only a small **fill list** instead of the per-table used lists (240 D2H/token) and the host policy
(240 x 8 `expert_slot` lookups/token).  Target: remove most of the **~0.41 ms/token** promotion cost at
`MIB=9216`, i.e. close a meaningful part of the 87 -> 94.5 t/s gap to identity.

### Why this and not something else (measured, 2x R9700, r25)

* `MIB=9216` devmap warm `tg1024` **~87.4** vs identity (`MIB=9344`) **94.5**; `-ncmoe 0` oracle ~96.
* `MOE_EXPERT_CACHE_TIMING=1`, `-n 1024 -r 4` (4095 tok), `MIB=9216`: promote **240 calls/token, total
  0.411 ms/token** = **d2h 0.154** (240 async `cudaMemcpyAsync`) + **policy 0.179** (`access_locked` x8
  per table) + **slot_h2d 0.024**.  h = 0.9934.
* The promotion is **real work, not overhead**: a diagnostic that skips the policy body collapses the arena
  to all-cold UVA and **25 t/s**.
* A **dirty-table** filter (skip tables whose used experts are all resident) was implemented and **rejected**
  (session 11b): the policy cost is the *dirty* tables' **fills**, not the clean tables' lookups — policy
  only 734.7 -> 644.1 ms — and the stale LRU cadence added evictions (0 -> 102).  See `WORKLOG.md` 11b.
* The remap kernels (80/token after options 1+2, 0.27 ms/token GPU) are byte-relevant but only ~1 %
  throughput and **not** the target.

### The current host flow (what you are replacing)

Per table, per token, in the **deferred pass** after the graph (all in `moe-expert-cache.cu` unless noted):

1. `ggml-backend.cpp:2493-2512` syncs each promote backend, then calls
   `r.promote_backend->iface.moe_cache_promote(...)` once per `moe_promote_rec` (240/token).
2. `moe_cache_promote_host` (:1677): D2H the used list `t.used_dev` -> pinned `used_host`/`used_host2`
   (**pipelined**, one token of lag — read the previous buffer); for each used expert call
   `access_locked`; if `t.slot_dirty`, rebuild `slot_pin` and H2D `slot_dev`.
3. `access_locked` (:391) is the LFRU policy: hit -> `count[e]++`, `last[e]=clock`; miss ->
   `ghost[e]++`, pick a victim (min `count`, tie oldest `last`, never one in the token's `protect` list),
   optionally reject (`admit=touch` needs `ghost[e] >= g_touch`; `value` needs `ghost[e] > count[victim]`),
   else evict + fill (H2D) + `slot_dirty=true`.  `decay_locked` (:348) halves `count`/`ghost` every
   `g_period` steps.
4. Table state lives in `table_t` (:~24): `slot_expert`, `expert_slot`, `count`, `ghost`, `last`,
   `clock`, `last_decay`, `slot_dirty`, and the device `slot_dev` / `used_dev` / `slot_pin` / `slot_dev_host`.
5. The consumer side: `moe_cache_get_table` (:1562) fills `moe_cache_devmap`
   (`moe-expert-cache.h:66`: `slot_dev`, `used_dev`, `n_experts`, `n_res`, `remap_fresh`); the remap kernel
   `moe_cache_build_remap_kernel` (:851) + launcher `moe_cache_launch_remap` (:899) build `remap_dev` from
   `slot_dev` and write `used_dev`; `moe_cache_redirect_fused` (:1821) is called from the fused gate+up
   (`ggml-cuda.cu:5631`) and the down fold (`ggml-cuda.cu:5831`); the per-op consumer is `ggml-cuda.cu:2593/2598`.
6. iface plumbing: `ggml-backend-impl.h:240` (`moe_cache_take_over`) and `:245` (`moe_cache_promote`);
   CUDA impl `ggml-cuda.cu:8068/8076`; Meta delegation `ggml-backend-meta.cpp:3326/3355`
   (`take_over` ORs `need_promote` over devices; `promote` forwards to each simple backend).  The takeover
   records `moe_promote_rec` in `ggml-backend.cpp:2250-2260`.

### Proposed design — device LFRU + fill-list readback

**Device state, allocated in `alloc_table_locked` (:547) next to `slot_dev`/`used_dev` (all raw
`cudaMalloc` on `t.device`), behind `MOE_EXPERT_CACHE_DEVPOLICY`:**

* `slot_expert_dev[slots]` int32 — slot -> expert, -1 empty (device mirror of host `slot_expert`).
* `count_dev[n_experts]` int32, `last_dev[n_experts]` int32 (or int64), `ghost_dev[n_experts]` int32.
* per-table `clock_dev`/`decay_dev` int32 (keep the clock on the device so no host sync is needed).
* **per device:** a descriptor array `{slot_dev, slot_expert_dev, count_dev, last_dev, ghost_dev, used_dev,
  n_experts, slots, n_used, n_tok, fill_head}` and a **fill-list** buffer
  `{int32 count, {int32 table_id, int32 slot, int32 expert} x cap}`.  Build the descriptors once at arm
  time (tables are fixed after sizing), so they are CUDA-graph-stable.
* Admission params (`g_admit`, `g_touch`, `g_period`, `g_noevict`) as kernel arguments.

**Policy kernel (one launch per device per token, enqueued after the graph on the compute stream):** one
block/warp per table; read `used_dev[n_used*n_tok]`; for each used expert `e`:

* hit (`slot_dev[e] >= 0`): `count_dev[e]++`, `last_dev[e] = clock`.
* miss: find an empty slot, else the min-`count` victim **not in the used list** (intra-token protect);
  apply admission (`touch`: reject if `ghost_dev[e] < g_touch`; `value`: reject if
  `ghost_dev[e] <= count_dev[victim]`); on admit: `slot_dev[victim] = -1`, `slot_expert_dev[slot] = e`,
  `slot_dev[e] = slot`, `count_dev[e] = 1`, `ghost_dev[e] = 0`, `last_dev[e] = clock`, append
  `(table_id, slot, e)` to the fill list; on reject: `ghost_dev[e]++`.
* decay every `g_period` (halve `count_dev`/`ghost_dev`), then `clock++`.

**Host side (new `moe_cache_promote_device`):** launch the policy kernel (in the deferred pass, after the
backend sync), D2H the fill-list **count** (1 read) and then the **entries** (1 read), and for each entry
issue the H2D fill using the existing host geometry (`host`, `host_bytes`, `src_off`, `host_pitch`,
`split_axis` — the code in `access_locked`'s fill branch is the reference).  **No per-table used-list D2H,
no host policy, no `slot_dev` H2D** (the kernel already updated `slot_dev`).

### Ordering / correctness requirements (read before coding)

1. **Lag is fine, ordering is not optional.**  The policy applies token T's routing and takes effect for
   T+1.  Token T's remap kernel used the pre-policy `slot_dev`; the policy updates it after.  Enqueue the
   policy kernel **after** the whole graph on the **same stream** (its input `used_dev` is written by remap
   kernels inside the graph), then enqueue the fills after it.  The next token's graph is enqueued after,
   so both are ordered — no extra host sync beyond the one the deferred pass already does.
2. **Keep the policy kernel out of the CUDA graph.**  It depends on the *whole* graph's `used_dev` writes,
   so launch it per token from the deferred pass.  The captured decode graph still holds the remap + MoE
   kernels.  The descriptor array must be fixed after sizing (it is).
3. **Port the admission EXACTLY.**  Same `touch`/`value`/`always`, same `g_touch`, same `g_period` decay,
   same victim tie-break (min `count`, then oldest `last`), same `NOEVICT`.  Add a self-test (mirror
   `MOE_EXPERT_CACHE_SELFTEST`) that replays a synthetic routing sequence and asserts the kernel and
   `access_locked` choose the same slot/victim.  A different *victim* still computes correct output, so
   byte-identity will NOT catch a policy bug — only the hit-rate `h` and the curve will.
4. **Intra-token protect.**  Never evict an expert used earlier in the same token (host does this with the
   `protect` array).  The kernel must scan the used list when choosing a victim.
5. **`-sm tensor` / Meta.**  Each simple backend has its own tables, descriptors and fill list; the
   per-device policy kernel runs independently.  `ggml_backend_meta_moe_cache_promote` is where the
   delegation for the new entry point goes.
6. **Fail-soft.**  If a device array/descriptor alloc fails, that table falls back to the host path.
7. **CPU-computes split** (`g_cpu_split > 0`) needs the host `used` list and must stay host-side.
8. **Identity tables** (`slots == n_experts`) have no promotion; the device policy only applies to devmap.
9. **Reporting / `moe_cache_read_check`.**  Host `count`/`ghost`/`slot_expert`/`expert_slot` go stale; either
   D2H the device arrays once at exit for `moe_cache_report`, or keep the host mirrors only under
   `MOE_EXPERT_CACHE_VERIFY`.
10. **Pipelining goes away** for the device path (`used_host`/`used_host2` become host-fallback only).

### Staged plan (each step buildable + verifiable)

1. **Device state + mirror.**  Allocate the arrays, seed from the host state at arm time, behind
   `MOE_EXPERT_CACHE_DEVPOLICY=1`.  Host path still runs; bit-identical, no behaviour change.
2. **Policy kernel + self-test.**  Implement the kernel as a pure function over
   `(used, slot_dev, slot_expert_dev, count/last/ghost)` -> `(updated state, fill list)`; self-test it
   against `access_locked` on synthetic sequences.
3. **Wire the fill list.**  Run the kernel in the deferred pass, read the fill list, issue the H2D fills,
   skip the host policy + `slot_dev` H2D.
4. **Batch.**  One launch per device per token via the descriptor array.
5. **Measure + decide.**  Arena sweep (depth 0 + 16384), six gates, promote timing.  Flip the default only
   if it wins; otherwise record the negative result and move to qwen4exp.

### Gates

* The **six gates in §1**: byte-identity to the r25 oracle (`de8be4d0c90c` 2-GPU `-sm tensor`,
  `15038c19ddc8` 1-GPU `-sm layer`), width purity `none == n3 == n7`, `-sm layer` regression,
  `test-backend-ops -o MUL_MAT_ID` 929/929, MTP, deep coherence.
* The **`decode-arena-sweep.md` curve** (warm reps, depth 0 **and** 16384): the device policy must match
  or beat the host policy at every `MIB`.
* `MOE_EXPERT_CACHE_TIMING=1` promote breakdown: d2h + policy should collapse to ~1 fill-list read + fills;
  the reported `h` must be unchanged (same admission decisions).
* The **self-test** (step 2) must pass with the kernel and host policy agreeing.

### Reference numbers at `ceea0cfb6` (r25, 2x R9700)

| | |
|---|---|
| devmap `MIB=9216` warm `tg1024` | ~87.4 |
| identity `MIB=9344` | 94.5 |
| `-ncmoe 0` oracle | ~96 |
| promote at `MIB=9216` | 240 calls/token, 0.411 ms/token (d2h 0.154 / policy 0.179 / slot 0.024) |
| remap launches (options 1+2) | 80/token, 0.27 ms/token GPU |

---

## 1. Where things are

| | |
|---|---|
| **Worktree** | `~/llama-decode`, branch **`wip-moe-devmap-v2`**, tip **`2376ac6cf`** (= **r26** (`0d58404e1`) + the campaign commits through session 18 / item 1).  Prior tips: `6ca5c1c77` (session 17 / B1, r26 base; backed up as `backup/wip-moe-devmap-v2-r26-b1`), `6140bba76` (r25-based; backed up as **`backup/wip-moe-devmap-v2-r25`**), `c83899985` (session 15), `ceea0cfb6` (session 11), `6d3e26e0d` (session 10); the pre-rebase r21 tips are backed up as `backup/wip-moe-devmap-v2-r21` (`a1d0fa985`) and `backup/wip-moe-expert-cache-r21` (`c7dd40a23`).  The eager path without devmap is branch `wip-moe-expert-cache` (`7e6c4cf66`); both build the same `build-rocm`. |
| **Build** | `cd ~/llama-decode && cmake --build build-rocm --target llama-cli llama-bench -j 16` (~1-2 min incremental with ccache).  Full rebuild: `BUILD_DIR=build-rocm EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714` (~7 min cold). |
| **Full patch** | **`exp19-moe-expert-cache-r26-b2-tensorpad.patch`** (`git diff 0d58404e1..wip-moe-devmap-v2`, clean-applies to **r26** `0d58404e1`) = `exp18` + item 1 (the Meta gather re-enable, the tensor-split default, and the MMQ expert-table tail pad in `moe_cache_gather_kernel`).  **`exp18-moe-expert-cache-r26-b1-rebase.patch`** = the session-17 tip (B1 only).  **`exp17-moe-expert-cache-r25-b1-prefill-gather.patch`** (r25 base) is **stale/superseded** — kept for history.  `exp16-moe-expert-cache-r25-b4-gather-off.patch` = the session-16 tip; `exp15-moe-expert-cache-r25-b2-devgather.patch` = the session-15 tip (gather default on); `exp14-moe-expert-cache-r25-b6-prefill-seed.patch` = the session-14 tip;  `exp13-moe-expert-cache-r25-devpolicy.patch` = the session-13 tip;  `exp12-moe-expert-cache-r25.patch` = the session-11 tip on r25 (`ceea0cfb6`); `exp11-moe-expert-cache-devmap-pipelined.patch` = the session-10 tip on r21 (`a1d0fa985`); `exp10` = the item-3 tip (`56f015057`); `exp9` = the items-1+2 tip (`c7dd40a23`); `exp8` = the session-7 cold-workaround snapshot; `exp7`/`exp6` older. |
| **Parked branch** | **`wip-moe-devmap-v2`** (tip `c5bbb7ee2`, the live branch); `wip-moe-expert-cache` (`7e6c4cf66`, the eager path); `wip-moe-devmap` (`6b8a7ed06`, the BROKEN first cut).  Pre-rebase SHAs are in the `backup/*-r21` refs. |
| **Iteration model** | `/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (21 GiB, fits 1 card; the fast smoke model). |
| **End-goal model** | `/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf` (100 GiB `qwen4exp`, 48x512 experts) — Phase 4; its lazy/PLE path makes `llama-bench` absolutes non-comparable.  **Transparency oracle:** `/llm/models/Qwen3.8/Flash-Next/IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf` (77 GiB, fits `-ncmoe 0`; `-ncmoe 0` == cache-on `-ncmoe 99` == `77c6f546460d`).  Shared MTP head: `.../IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`. |
| **Hardware** | 3x R9700 (gfx1201); use `HIP_VISIBLE_DEVICES=0[,1]`.  Pin `-t 8` (the GPU IRQs live on the top cores). |
| **Delivery** | `~/llama-cpp-rdna-boosts` `main`; the campaign README/WORKLOG/patches live in `wip/moe-expert-cache/`.  The `~/llama-decode` checkout is **never pushed**. |
| **Drop-off baseline** | `decode-arena-sweep.md` — the warm decode `tg` vs arena-size sweep (depth 0 + depth 16384); the session-10 pipelined devmap already flattened it (see its Postscript 2).  Session 12 added the **device-side admission policy** (a depth-0 win at every measured arena) and the `_PROGRESS` log that shows the cache is still warming long past the bench length; the depth-16384 sweep and the promotion gates remain. |

### How to run (throughput / purity / MTP / coherence)

```sh
MQ4=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

# throughput (per-rep, warm = rep >= 2; rep 1 pays the one-time cold fill)
HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=10240 ./build-rocm/bin/llama-bench \
  -m $MQ4 -ncmoe 99 -ngl 99 -fa 1 -sm tensor -t 8 -p 0 -n 1024 -r 6 -o jsonl

# NEW (item 3b) devmap path: add MOE_EXPERT_CACHE_DEVMAP=1.  The sweep is MIB=1024/2048/3072/4096/6144/8192/9216
# plus the identity endpoint MIB=9344 (slots == 256).
HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=8192 MOE_EXPERT_CACHE_DEVMAP=1 ./build-rocm/bin/llama-bench \
  -m $MQ4 -ncmoe 99 -ngl 99 -fa 1 -sm tensor -t 8 -p 0 -n 1024 -r 4 -o jsonl

# promotion breakdown + expert traffic (use -v; grep the exit report)
HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=9216 MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_TIMING=1 \
  ./build-rocm/bin/llama-bench -v -m $MQ4 -ncmoe 99 -ngl 99 -fa 1 -sm tensor -t 8 -p 0 -n 512 -r 2 -o jsonl \
  2>&1 | grep moe_cache_report

# machinery A/B at h=1 (same arena): identity vs forced devmap (one run each)
#   MOE_EXPERT_CACHE_MIB=9344 MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_FORCE_DEVMAP=0   # identity 94.0
#   MOE_EXPERT_CACHE_MIB=9344 MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_FORCE_DEVMAP=1   # devmap   84.9

# NEW (session 12) device-side admission policy + progress log (needs DEVMAP=1).  -v is required for
# the WARN progress lines.  Compare DEVPOLICY 0 vs 1 at FORCE_DEVMAP=1 MIB=9344 (machinery, h=1) and at
# a small MIB (fill-heavy).
HIP_VISIBLE_DEVICES=0,1 MOE_EXPERT_CACHE_MIB=9344 MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_FORCE_DEVMAP=1 \
  MOE_EXPERT_CACHE_DEVPOLICY=1 MOE_EXPERT_CACHE_PROGRESS=1024 \
  ./build-rocm/bin/llama-bench -v -m $MQ4 -ncmoe 99 -ngl 99 -fa 1 -sm tensor -t 8 -p 0 -n 1024 -r 4 -o jsonl \
  2>&1 | grep moe_cache_progress

# kernel-count ground truth (the 7.14 /usr/bin/rocprofv3 hangs on this workload; the ROCm 10 one works):
#   MOE_EXPERT_CACHE_MIB=9344 MOE_EXPERT_CACHE_DEVMAP=1 MOE_EXPERT_CACHE_FORCE_DEVMAP=1 \
#   /opt/rocm-10.0.0-gfx120X/bin/rocprofv3 --kernel-trace -o /tmp/prof -- ./build-rocm/bin/llama-bench ... -n 32 -r 1
#   then: sqlite3 /tmp/prof_results.db  -> rocpd_kernel_dispatch joined to rocpd_info_kernel_symbol
#   expects `moe_cache_build_remap_kernel` = exactly 240 * decode_tokens dispatches @ ~1.57 us.

# byte-identity oracle (fusions ON): full-table GPU, `-ncmoe 0`
HIP_VISIBLE_DEVICES=0,1 ./build-rocm/bin/llama-cli -m $MQ4 -ngl 99 -ncmoe 0 -fa 1 -sm tensor \
  -t 8 -c 8192 --seed 42 --temp 0 --reasoning off --ignore-eos --single-turn --no-display-prompt \
  -p "$(cat ~/llama-cpp-rdna-boosts/prompts/reasoning.txt)" -n 300 > /tmp/oracle.out
python3 ~/llama-cpp-rdna-boosts/scripts/extract-generated.py /tmp/oracle.out     # hash WITHOUT -v
```

`llama-cli`/`llama-bench` raise the log level to ERROR unless `-v`, so every cache diagnostic (including
`FULLY RESIDENT`, the sizing line and the exit `moe_cache_report`) is invisible in a normal run — use
`-v` for diagnostics and a *separate* run for the hash (`-v` corrupts the text extractor).

### Env knobs (`moe-expert-cache.h`)

| knob | default | meaning |
|---|---|---|
| `MOE_EXPERT_CACHE_MIB` | 0 (inert) | per-device VRAM budget.  Full residency (`slots == n_experts`) turns the identity path on; on this model that is ≈9.3 GiB (`MIB≈9344`, slots 256/256) for the 2-GPU `-sm tensor` Q4_K_M iteration model. |
| `_SLOTS` | 0 | explicit uniform slots/table (forces immediate sizing). |
| `_COLD` | **`uva`** | `uva` = in-place pinned-host cold reads (the default now that the fused gate+up+GLU / down-fold kernels serve cold ids).  `off` restores the pre-1b fill-every-miss policy. |
| `_ADMIT` | `touch` | `always` / `value` = rejected admission rules. |
| `_PERIOD` / `_TOUCH` | 32 / 2 | LFRU decay period / touch threshold. |
| `_RESERVE_MIB` | 1024 | VRAM held back from the arena (arena is sized from *free* memory, so `--fit` need not know). |
| `_FAIL_ALLOC` | 0 | induced-allocation-failure fail-soft test. |
| `_ASSERT` / `_ASSERT_SABOTAGE` | 0 | structural invariant / gate-liveness self-test (CPU-split). |
| `_SELFTEST`, `_VERIFY`, `_REPORT`, `_DEBUG`, `_SKIP_ROLE`, `_FORCE_COPY`, `_NOEVICT`, `_CPUSPLIT` | — | bring-up / A-B knobs. |
| `_DEVMAP` | **0** (off) | device-side remap (item 3): build the slot remap on the device + deferred post-graph promotion.  Since session 10 the promotion is **pipelined** (async double-buffered readback) and **slot-dirty-skipped**, so it is a +22 % win over eager at partial residency (85 t/s at h~0.9 vs 70).  Width purity was re-confirmed in session 10 (`none == n3 == n7 == 15038c19ddc8`); the full MTP/coherence gate re-run and the default flip are the remaining promotion steps — see `WORKLOG.md` 2026-09-29 (session 10). |
| `_FORCE_DEVMAP` | 0 | keep the devmap path even at `h=1` (suppresses the identity fast path).  A/B knob: at the same arena it isolates the devmap *machinery* cost (identity 94.0 vs forced-devmap 84.9 = 1.14 ms/token). |
| `_DEVPOLICY` | **1 (ON)** when `_DEVMAP=1`; `0` = off | run the LFRU admission + eviction + fill on the GPU: one batched kernel per device per token replaces the per-table host promotion (used-list D2H + host policy + slot-map H2D) and copies admitted experts from the pinned host alias into the arena in the same launch.  **Defaulted ON in session 13** after the self-test/width/MTP/concurrency gates; kill switch `=0`. |
| `_PREFILL_SEED` | **1 (ON)**; `0` = off | **the prompt-routing seed (session 14, B6, works).**  Tally the prefill routing on the device (a kernel from `ggml_cuda_mul_mat_id`, so `-sm tensor`'s block-06 staging cannot bypass it) and bulk-admit the hottest experts as provisional slots at the first decode-band policy flush.  Default ON (it passed the byte-identity / width-purity / `MUL_MAT_ID` / MTP / coherence gates); `=0` is the kill switch.  Needs `DEVMAP=1` (with `DEVPOLICY` default-on); inert with `DEVMAP=0`/`DEVPOLICY=0`, and the tally is gated on `g_devmap` so an unused seed costs nothing.  `llama-cli -n 300` `-sm layer`: `MIB=8192` 52.3 -> 57.9, `MIB=16384` 60.9 -> 80.8 t/s.  See `WORKLOG.md` 2026-09-29 (session 14). |
| `_PREFILL_SEED_N` | 0 | cap the seeded/loaded experts per table (`0` = all `slots`); the traffic knob. |
| `_PREFILL_LOAD` | **0** (off) | fill experts `0..slots-1` into the arena at load.  With `_PROVISIONAL=1` it is the **fastest** config (session 12i); without it, neutral-to-(-1.5 %) because a full arena gates every miss through `touch`. |
| `_PROVISIONAL` | **1 (ON)**; `0` = off | treat a pre-filled/seeded expert as an **empty slot for admission** until its first hit (per-slot `slot_prov`, cleared on hit/admission): the doorkeeper is bypassed for it while it still serves hits.  Default ON (session 14): it is the only way the prompt seed's not-yet-hit entries are reclaimed, and it is inert until something is pre-filled/seeded (a no-op for an empty arena).  `PREFILL_LOAD + PROVISIONAL` = 58.59 t/s vs empty 58.09 vs prefill-no-prov 57.22 (session 12i); seed `MIB=16384` 79.8 -> 80.6 with it (session 14).  Byte-identical. |
| `_WARMUP_TOKENS` | 0 (off) | **measured negative (session 12h)**: force first-touch admission (`always`) for the first N decode tokens to convert a pre-filled/seed arena quickly.  Raises `h` but lowers throughput (55.5 vs 58.1 t/s) - hit rate is not the objective, PCIe traffic is. |
| `_PROGRESS` | 0 (off) | log cumulative admissions (fills) vs evictions, resident/slots and `h` every N decode tokens per device (needs `-v` for llama-bench/cli).  Shows cache warm-up vs steady state; the log proves the default `touch` policy is still `WARMING` at 18k tokens. |
| `_TIMING` | 0 | print the deferred-promotion breakdown and the expert-access/traffic accounting at exit. |
| `GGML_CUDA_CACHEDBG`, `GGML_CUDA_FUSE_LOG`, `GGML_SCHED_SYNCDBG`, `GGML_CUDA_GCDBG` | — | diagnostics. |

### The gates every change must pass

1. **Byte-identity to the `-ncmoe 0` oracle** (full-table GPU, fusions ON): `15038c19ddc8` at 300 tok
   (Q4_K_M, 2-GPU `-sm tensor`, `prompts/reasoning.txt`).
2. **Width purity**: `none == n1 == n3 == n7`.
3. **`-sm layer` regressions**: 1-GPU and 2-GPU, fusions off, `MIB=8192` → `ad30da7b5a3a`.
4. **`test-backend-ops -o MUL_MAT_ID`** green (929/929).
5. **MTP** `n3`: acceptance > ~0.45 at pos-1 and `draft-mtp >= plain` at depth 3.
6. **Deep coherence** (maintainer gate): a 12-section ~7k-word essay + MTP completes naturally, no
   repetition collapse.  Prompt: `coherence-essay-prompt.txt` in this directory (sha256
   `5e9a8ab0…`); run with `-n 12000 -c 16384 --spec-type draft-mtp --spec-draft-n-max 3`, no
   `--ignore-eos`, and check rc=0 + a `## Conclusion`.

---

## 2. Remaining work items

### Item 1 — Per-expert-stride tensor-split UVA cold path  ✅ **DONE (session 8)**

`moe_cache_get_cold` now returns the host master's device alias (with the per-device slice offset
folded in), the resident slot count, and the host geometry — the per-expert byte stride and the per-row
byte pitch (0 meaning "the arena's own row stride").  `cold_safe` is now "the host geometry is
representable" (any contiguous per-expert blob, or equal-width rows for the strided axis-0 `ffn_down`
slice), not "unsplit", and `mul_mat_vec_q_moe` takes `cold_channel_stride`/`cold_row_stride` and uses
`cold ? cold_base + cold_channel_stride*(id-n_res) + cold_row_stride*row` for a cold id.  The resident
path is untouched (`ne[2]`/`nb` stay at the full expert count, so the dispatcher heuristics are
unchanged).  Verified byte-identical to the oracle at every partial residency down to `MIB=1` (no arena),
including the axis-0 `ffn_down` slice (the tricky one): a forced-cold A/B at `MIB=1024` (h=0.55,
~217k cold reaches) is `15038c19ddc8`.  See `exp9-…-session8-coldpath.patch` and the WORKLOG
session-8 entry.

### Item 2 — Make the fused cache-band MoE kernels cold-aware  ✅ **DONE (session 8)**

The gate+up+GLU fusion reaches `mul_mat_vec_q_moe_launch` (the `has_ids` mmvq path), so the fix was to
thread the item-1 geometry through it AND to resolve the **gate** lane's cold geometry independently
(the gate lane is a separate table with its own pinned host master and slot count, even though it shares
the up lane's remap).  The fused gate+up and the `[MUL_MAT_ID, MUL]` down fold now serve cold ids, so:
`MOE_EXPERT_CACHE_COLD` defaults to `uva`; the `ggml_cuda_cache_blocks_fusion` cold stand-down, the
`MOE_EXPERT_CACHE_FUSED_COLD_OK` footgun and its warning are gone; and the `touch` + in-place-UVA policy
is the default again.  Along the way the change exposed and fixed a pre-existing byte-identity bug — see
"Remaining work items → Item 1a" below.

### Item 1a — A cache that cannot serve must fall back wholesale  ✅ **DONE (session 8)**

With `COLD=off`, or any config where the hook declined (a 4-token batch with `used > slots`, an arena
alloc failure, a budget below one expert), the **cache-band gate+up fusion** read the scheduler's
op-offload `input_cpy` through a fused call site that is only correct when the scheduler itself staged
the copy — so `MIB <= 1024` (and the induced/partial fail-soft path) emitted garbage logits from a
near-tie ~120 tokens in.  `ggml_cuda_cache_blocks_fusion` now stands the cache-band MoE fusions down
whenever `moe_cache_has_arena()` is false (false until sizing, on any failed arena, on an empty cache),
and `moe_cache_update_host` declines when a table has no arena.  Verified: `MIB=1/64/256/1024`,
`COLD=off` at `MIB=1`, `FAIL_ALLOC=1/3/7` (all/partial induced failures) are all `15038c19ddc8`, and
the `COLD=off` `MIB=1024` kill-switch is correct again.

### Item 3 — Device-side remap ("gentle curve")  ✅ **DONE — NEGATIVE, then FIXED by item 3b-I (session 10)**

**Outcome (2026-09-29, session 9): the device-side remap was byte-correct but uniformly 10-15 % SLOWER
than the eager partial-residency path.  Session 10 root-caused it (a *synchronous* per-table used-list D2H
was 81 % of the promotion) and fixed it, so the curve is now plateau-free and +22 % over eager at every
partial residency — see Item 3b-I.  `MOE_EXPERT_CACHE_DEVMAP=1` stays opt-in pending the width-purity/MTP
gates.  Detail: `WORKLOG.md` 2026-09-29 (session 9) and (session 10).**

**Goal.**  At `h<1` the cache still falls to ~70 t/s because the scheduler reads the routing back to the
host per layer (for the used-expert pruning *and* the cache's remap decision) and does a full device
synchronize per layer.  The identity path removes that only at `h==1`.  The maintainer's target is a
**gentle** approach as the resident fraction rises — the decode analogue of the prefill `-ncmoe`
drop-off table in [discussion #54](https://github.com/stew675/llama-cpp-rdna-boosts/discussions/54).

**Measured baseline to flatten (session 8, `decode-arena-sweep.md`, warm, 2×R9700) — superseded by the session-10 table below:**

| h | slots/256 | warm `tg1024` | warm `tg512@d16384` |
|---:|---:|---:|---:|
| 0.11 (`MIB` 1024) | 28 | 42.9 | — |
| 0.22 (`MIB` 2048) | 56 | 55.4 | 52.4 |
| 0.44 (`MIB` 4096) | 112 | 66.1 | 62.4 |
| 0.66 (`MIB` 6144) | 168 | 69.8 | 64.1 |
| 0.88 (`MIB` 8192) | 224 | 69.7 | 63.9 |
| 0.99 (`MIB` 9280) | 254 | 69.4 | — |
| **1.00 (`MIB` 9344+, identity)** | **256** | **94.1** | **87.3** |
| — (`-ncmoe 0` oracle) | — | ≈96 | 89.9 |

**Session-10 result (devmap, pipelined promotion + slot-dirty skip; warm `tg1024`, `-n 1024 -r 4`):**

| `MIB` | slots/256 | h | old (eager) | **new (devmap)** | delta |
|---:|---:|---:|---:|---:|---:|
| 1024 | 28 | 0.11 | 42.9 | **45.5** | +6 % |
| 2048 | 56 | 0.22 | 55.4 | **62.2** | +12 % |
| 3072 | 84 | 0.33 | 62.3 | **74.5** | +20 % |
| 4096 | 112 | 0.44 | 66.1 | **80.8** | +22 % |
| 6144 | 168 | 0.66 | 69.8 | **86.1** | +23 % |
| 8192 | 224 | 0.88 | 69.7 | **85.3** | +22 % |
| 9216 | 252 | 0.98 | 69.8 | **85.6** | +23 % |
| **9344** | **256** | **1.00** | **94.1** (identity) | **94.1** | — |

Depth 16384 (`-n 512 -d 16384 -r 3`): `MIB=2048` **60.9** (old 52.4), `MIB=4096` **75.0** (old 62.4),
`MIB=8192` **76.6** (old 63.9), identity **87.5** (old 87.3).  The
plateau (69.4-69.8 from h=0.66) is **gone**; the residual cliff is 9216→9344 = 85.6→94.1 (**+10 %**, was
**+36 %**), and h~0.98 is at **91 % of identity** (was 74 %).  The remaining 1.14 ms/token at h=1 splits
~48 % GPU remap-kernel dispatch / ~39 % host promotion / ~11 % expert PCIe — **not** bandwidth (see
Item 3b-I).

The plateau (`≈70` from h≈0.66 to h≈0.99) and the cliff (254→256 slots = +64 MiB buys +36 %) are the
identity path's all-or-nothing threshold, **not** fusions: the cache-band fusions fire identically at
every h (verified with `GGML_CUDA_FUSE_LOG`), and `h<1 fusions-on (69.2) ≈ h=1 fusions-off (71.9)` is a
coincidence — the round-trip costs about what the fusions gain.  The round-trip is quantified in
`decode-arena-sweep.md` (`get_async` 1906 at h=0.88 vs 228 at h=1).

**Acceptance gate (the deliverable).**  Warm reps (rep 1 pays the cold fill), depth 0 **and**
depth 16384, over the same `MIB` grid plus the `-ncmoe 0` / cache-off endpoints: the curve must be
**monotone and cliff-free**, and the h→1 endpoint must stay within a few percent of the identity path.
The full prefill-style table (every `MIB`, both depths, `-r 5`+) is the final artifact — the sweep above
is the development-resolution version.

**Design.**  Build the slot remap **on the device** from the routing `ids` and a device
`expert -> slot` map (`slot_dev` int32[n_experts]) with a tiny capture-safe prepass kernel; the
scheduler takes the input over **before** the ids readback and records the routing for a **deferred
post-graph promotion pass** (one sync per token, not one per layer).  Fills then lag one token; the
current token's misses must be servable cold — item 1 (the per-expert-stride cold path) is now DONE,
so a split table is servable cold and the blocker is gone.

**State.**  Implemented on branch **`wip-moe-devmap-v2`** (`56f015057`; full patch
`exp10-moe-expert-cache-devmap.patch`): the kernel, the `moe_cache_devmap` struct,
`moe_cache_get_table`'s devmap branch, `moe_cache_take_over`/`_promote_host`, the CUDA/Meta adapters, the
scheduler takeover + deferred post-pass, and the fused redirect accept a `stream`.  Three correctness
fixes were required beyond the old broken first cut (`wip-moe-devmap`, `6b8a7ed06`): (1) **transition
arming** - devmap takeover stays off until one uniform eager pass has filled every devmap table, or the
sizing token mixes eager/devmap roles and diverges gate/up maps (the fused gate+up kernel indexes the
gate lane with the UP table's remap); (2) the **deferred readback must not touch the graph's routing
tensor** - its storage is recycled once the graph completes, so the remap kernel now writes the routing
into a cache-owned persistent `used_dev` buffer and the promotion reads that (cold fraction 97.6 % ->
6.8 %); (3) the promotion must use the **same admission policy** as the eager hook.  All gates are green
with devmap ON (byte-identity `15038c19ddc8` at `MIB=1024/4096/8192`, width purity `none == n3 == n7`,
1-GPU `-sm layer` `883011516483`), but the measured warm `tg1024` curve was **monotonically worse** than
eager at every `MIB` (1024: 38.6 vs 42.3; 4096: 58.0 vs 66.0; 8192: 59.5 vs 69.4; 9216: 59.5 vs 69.4),
with the identity endpoint still ~94.  **Session 10 corrected the attribution**: it was not the fills - it was
the **synchronous per-table `cudaMemcpy` D2H** of the used-list on 240 tables/token (25.1 us/call, 5.0 s of
the 6.2 s pass; the session-9 "copies were never the cost" note was wrong).  The pipelined + slot-dirty fix
below turns the loss into **85.6 t/s at h~0.98** (+23 % over eager).  The remaining gap is now ~48 % GPU
remap-kernel dispatch / ~39 % host promotion / ~11 % PCIe.

**Item 3b-I — pipelined promotion (DONE, session 10, branch `wip-moe-devmap-v2` tip `a1d0fa985`).**  Two
changes in `moe_cache_promote_host` (no interface change): a **double-buffered pipelined readback**
(`cudaMemcpyAsync` this token into the other pinned buffer, apply the policy to the previous call's buffer,
whose copy the inter-token backend synchronize has completed - one extra token of admission lag) and a
**slot-dirty skip** for the `slot_dev` rebuild/upload.  Promote cost **25.1 -> 1.8 us/call, 6.2 -> 0.43
ms/token**; byte-identity preserved at `MIB=1024/4096/9216`.  Full patch
`exp11-moe-expert-cache-devmap-pipelined.patch`.  Instrumentation: `MOE_EXPERT_CACHE_TIMING=1` prints the
promote breakdown + access/traffic accounting, and `MOE_EXPERT_CACHE_FORCE_DEVMAP=1` keeps devmap at `h=1` for
the machinery A/B.

**The clean h=1 attribution** (`MIB=9344`, identity 94.0 vs forced-devmap 84.9 = **1.14 ms/token**): host
promotion 0.44 (d2h 0.16 / policy 0.20 / slot 0.03), the **240 per-table `moe_cache_build_remap_kernel`
dispatches ~0.55** (`rocprofv3`: exactly 240/token, 1.57 us each = 0.375 ms GPU + gaps), expert fill H2D
0.13 (3.85 MiB/token at ~30 GB/s aggregate).  Non-resident cycling at high `h` is compulsory only: `MIB=8192`
over 4095 tokens = 99.25 % access hit, **234 evictions total**, 4.4 MiB/token; PCIe only becomes a real
limiter below h~0.9 (`MIB=2048`: 15.6 evictions/token, 84.6 MiB/token ≈ 2.96 ms of ~6 ms).

**Extra care.**  The `slot_dev` H2D and the fill copies are enqueued on the compute stream after the
graph, so the next token's remap kernel and MoE kernel are ordered after them; the capture must not
bake a per-token decision (the known graph/eviction class — keep decisions constant per shape).

### Item 3b-II — eliminate / batch the per-table remap kernels

**Options 1 and 2 are DONE (session 11, tip `ceea0cfb6`).**  Option 1 folds the gate-lane `used_dev`
write into the up kernel.  Option 2 builds the layer's routed `down` remap in the same kernel from the
down's **own** slot map and makes the down fold skip its launch (`remap_fresh`, cleared by that table's
promotion after the graph, so eager/capture/replay all agree).  Dispatches **240 -> 80 per token**
(captures 960 -> 320), byte-identity and width purity preserved.  See `WORKLOG.md` 2026-09-29 (session
11 / 11b).

**The finding: the remap kernels are not the critical path.**  `rocprofv3` (the ROCm-10 tool) measured
`moe_cache_build_remap_kernel` at 0.27 ms/token (160 launches) with the GPU only ~29 % busy; halving the
launches (option 1 then 2) does not move warm `tg1024` outside the ~2 % bench noise.  The residual is the
**per-token host promotion** - 240 calls/token, `d2h 0.154 + policy 0.179 + slot 0.024 = ~0.36 ms/token`
at `MIB=9216` - and it is real admission work (skipping it collapses to all-cold UVA and 25 t/s).

**Goal (remaining).**  Move the promotion's used-list read and admission decision off the host so the
per-token synchronize + D2H + policy stop sitting on the critical path; the target is `h→1` within a few
percent of identity, and the acceptance gate is still the `decode-arena-sweep.md` curve (warm reps,
depth 0 **and** 16384) plus the six gates in §1.

**Why (and the tried negative result).**  A **dirty-table** filter was implemented and rejected in session
11b: the remap kernel tagged non-resident used experts with a `MOE_CACHE_COLD_BIT` and the promotion
skipped the policy for tables with no such bit.  It barely helped (policy only 734.7 -> 644.1 ms at
`MIB=9216`) because the policy cost is the *dirty* tables' fills, not the clean tables' lookups; worse,
the stale LRU cadence introduced evictions (0 -> 102).  So the residual is mostly the compulsory fill
churn of a sub-full arena, not removable host overhead.  The one structural fix left is the full
**device-side admission policy** (LFRU + fill-list built on the GPU), which removes the host D2H + policy
entirely - **see the *Device-side admission policy* fresh-session brief above for the design, code
pointers, ordering requirements, staged plan and gates**.

**Implementation pointers (tip `ceea0cfb6`).**
* kernel + launcher: `ggml/src/ggml-cuda/moe-expert-cache.cu` — `moe_cache_build_remap_kernel` (~848),
  `moe_cache_launch_remap` (~872), `moe_cache_redirect_fused` (~1758), `moe_cache_promote_host` (~1648).
* call sites: `ggml/src/ggml-cuda/ggml-cuda.cu` — fused gate+up `moe_cache_redirect_fused(...)` (~5578),
  the `[MUL_MAT_ID, MUL]` down fold (~5778), the per-op consumer in `ggml_cuda_mul_mat_id` (~2579).
* the fused kernel reads the **UP** table's remap for **both** lanes; `gate_cpy->data` points at the gate
  *arena* (filled per the UP map), so the gate remap output is never read — only its `used_dev` side effect.
* the deferred pass lives in `ggml/src/ggml-backend.cpp` (~2486-2510); the Meta/CUDA iface adapters are
  `ggml-backend-meta.cpp` (~3320) and `ggml-cuda.cu` (~8007).

**Fresh-session first steps.**  (1) `cd ~/llama-decode && git switch wip-moe-devmap-v2 && git log -1`
(expect `ecac6360c`), then the incremental build.  (2) Reproduce the endpoint: `MIB=9216 DEVMAP=1` warm
`tg1024` ~87 and byte-identity to the r25 `-sm tensor` oracle `de8be4d0c90c` (`MIB=1024/9216`, DEVMAP
on/off).  (3) Measure any further change with `FORCE_DEVMAP=1` at `MIB=9344` so the residency is taken out
of the comparison (machinery A/B: identity 94.5 vs forced-devmap ~87.0), and read the `remap-kernel
launches` line of `MOE_EXPERT_CACHE_TIMING=1` as the deterministic counter.

**Options, easiest first.**
1. **Fold the gate lane in.**  ✅ **DONE (session 11)** — see the top of this section.  One remap kernel
   writes both tables' `used_dev`; 80 of the 240 launches are gone.
2. **Batch the launches.**  A remap for every table could be one kernel over a packed descriptor array
   (slot map + remap ptr + used ptr per table), scheduled once.  The routing for layer L is only ready
   after L's router, so batching must respect the graph order - feasible as a small number of per-layer
   or per-band batches.
3. **Fuse the slot lookup into the MoE ids read.**  Pass `slot_dev` instead of a remap buffer to the
   `mul_mat_vec_q_moe` / MMQ paths and do `slot[ids[i]]` in-kernel; this removes the remap kernels
   entirely (and the `used_dev` write could move to an atomic in the router).  Highest reward, touches the
   hot kernels and the cold/zero-slot encodings - gate it and A/B carefully.

**The remaining host promotion (0.20 ms policy + 0.16 d2h).**  **DONE (session 12):** the device-side
admission policy now runs the LFRU + fill on the GPU behind `MOE_EXPERT_CACHE_DEVPOLICY=1`, removing the
used-list D2H, the host policy and the slot-map H2D (the fill is done by the same kernel from the pinned
host alias).  It is a measured win (`forced-devmap MIB=9344` 86.35 -> 89.74; `MIB=1024` 46.73 -> 50.87).
The gates/self-test and the default flip remain.

**Trap list (learned the hard way).**  (1) Do not read the graph's routing tensor after the graph — use
`used_dev`.  (2) Keep every role of a layer on identical maps (the fused gate+up kernel reads the UP remap
for the GATE lane).  (3) Arm the fast path only after one full uniform eager pass, or the sizing token mixes
paths.  (4) Keep the graph-time decision constant per shape, or CUDA-graph replay will bake a stale choice.
(5) The promote uses a **pipelined** readback: never read the buffer you are about to overwrite (alternate
`used_host`/`used_host2`), and remember the backend must synchronize between tokens or the read races.

### Item 4 — Lower priority / deferred

* **Route (2), graph-level arena redirect**: once a table is sized, point the weight's `data` at the
  arena and stop the op being host/offloaded, with the remap as a persistent graph input.  Removes the
  `input_cpy`/offload entirely (the asymptotic shape), but needs a re-schedule after sizing.
* **Prefill cache-aware**: with `-ncmoe`, prefill uploads the used experts per ubatch and pays the
  op-offload sync; attribute the prefill gap and make prefill approach `-ncmoe 0` as experts load.
  Also inherited: **prune the staged prefill upload to the used experts** (r16 stages the whole tensor).
* **Static block-pin analyser** (no-code warm start; `-ncmoe` cannot express a non-uniform hot subset).
* **2-level VRAM cache / prefetch** — measured within ~0.01 of LFRU (SLRU) or superseded by it; revisit
  only on a measured miss-cost breakdown.
* **Tensor-split cold admission** is item 1 (done); the **static `NOEVICT` set** is measured worse, do not
  re-try as the mechanism.

---

## 3. The decode arena vs the prefill `-ncmoe` system (how they fit)

They are **two different axes that currently run independently** — not one system with two names, and
not really competing, but they are not yet unified and the VRAM accounting is split between them.

**Prefill (`-ncmoe N` + block-06 op-offload staging).**  `N` is a **static placement** decision: `N`
MoE layers' experts live in the host pool.  There is **no persistence** — each prefill ubatch uploads
the *used* experts (pruned) through the H2D staging ring, overlapped with compute, then the device copy
is reused for the next ubatch.  The deliverable's win was making that upload asynchronous/overlapped, so
the `-ncmoe` sweep is nearly flat (6450→5794 t/s over 0→40 on 1 GPU).  Prefill is **band-excluded from
the cache** (`n_tokens > 8`): the cache does not touch it.

**Decode (`MOE_EXPERT_CACHE_MIB`).**  The arena is a **persistent per-device VRAM cache** of a hot
subset (LFRU), decode/verify band only, over the same host pool `-ncmoe` created.  At `h=1` (slots ==
n_experts) the identity path makes it a plain device copy of the whole table; below that, misses are
served in place from the pinned host (UVA cold, item 1) or filled.  Its x-axis is therefore the
**resident fraction `h`**, not `-ncmoe`.

**Where they touch / the friction.**
* **Same bytes, two mechanisms.**  `-ncmoe N` puts experts in the host pool; the arena caches some of
them back onto the device.  Set `MIB` large enough and the arena *is* `-ncmoe 0` for decode while prefill
still streams from the host — that is the intended, desirable overlap, not a conflict.
* **VRAM accounting is independent.**  `--fit` sizes the KV cache + compute reserve and knows nothing
  about the arena (the arena is deliberately sized from *free* VRAM after `--fit`, minus
  `MOE_EXPERT_CACHE_RESERVE_MIB`), and the prefill staging arena is likewise outside the compute-graph
  reserve (issue #33).  So `MIB` is a third, self-managed VRAM consumer.  A big arena shrinks the free
  headroom a deep ubatch's staging/`--fit` growth might want, and the fail-soft path (arena alloc fails
  → the cache stands its fusions down, item 1a) is the safety valve.
* **No cross-regime reuse yet.**  The arena is not consulted by prefill, and prefill's staged upload is
  not left resident for decode.  On a mixed server the same experts are uploaded per prefill ubatch and
  separately cached for decode — correct, but two copies of the machinery.

**The unification is item 4's route (2)** ("graph-level arena redirect"): once a table is sized, point
the weight's `data` at the arena and stop the op being host/offloaded, with the remap as a persistent
graph input.  Then the arena *is* the persistent device-resident set, prefill only stages the misses, and
`-ncmoe`/the scheduler treat arena-resident experts as device-resident.  That is the asymptotic shape and
the natural long-term home; it needs a re-schedule after sizing, which is why it is deferred.

**Practical reading for the gentle-curve work.**  Keep the two axes separate for now: the item-3/item-3b
deliverable is a table over **arena size** (resident fraction), with `-ncmoe 99` fixed and the
`-ncmoe 0` / cache-off lines as endpoints.  Item 3 landed byte-correct, session 10 made it a win, and the
next task (item 3b-II) is to remove the per-table remap kernels and then, optionally, the remaining host
promotion (the original item-3b device-side admission policy).  A separate
follow-up should characterize the *mixed* case (`-ncmoe N` with an arena) once 3b lands, because that is
where the prefill and decode systems actually meet.

---

## 4. Completed work — index (one line each, detail in `WORKLOG.md`)

| # | done | detail |
|---|---|---|
| 1 | Phase 0 baseline + gap table (2-GPU resident 79.7/90.5, all-host CPU MoE 24.2 t/s) | WORKLOG: *HANDOVER BRIEF* / *The gap* |
| 2 | Routing profiling + admission policy: per-layer slot ranges + **LFRU** beat LRU/LFU/second-touch | WORKLOG: *FINDINGS* §1.6, §2.2 |
| 3 | Seam decision: `device_alias()`/base-pointer table at the `MUL_MAT_ID` weight source, **not** a buffer type | WORKLOG: *FINDINGS* §2.3 |
| 4 | Phase 1a: per-layer arena + LFRU + slot-remap consumer, driven from the scheduler hook | WORKLOG: *The hardening tasks* (Phase 1a status), *EVICTION BUG* |
| 5 | Eviction wrong-output bug root-caused (CUDA graph capture vs the per-token host decision) and fixed | WORKLOG: *EVICTION BUG* |
| 6 | H1 targeted fusion guard (only cache-band routed fusions stand down) | WORKLOG: *The hardening tasks* H1 |
| 7 | H2 width purity + transparency (byte-identical to the oracle across `none`/`n3`/`n7`) | WORKLOG: *H2 RECORD* |
| 8 | H3 budget/slot allocator (uniform, adaptive, aligns gate/up/down residency) | WORKLOG: *The hardening tasks* H3 |
| 9 | **Delivery fix promoted:** the qwen35moe `ssm_gate_beta` fusion was width-impure → release `v16-84e76d8a2-r18` | WORKLOG: *FUSION WIDTH-UNIFORMITY* |
| 10 | Phase 1b UVA cold read + the `touch` admission policy (later found to conflict with fused kernels — see item 2) | WORKLOG: *PHASE 1B RECORD*, *PHASE 1B POLICY RECORD* |
| 11 | Phase 1d fail-soft + `--fit` (arena sized from free VRAM; induced-failure liveness) | WORKLOG: *1D RECORD* |
| 12 | CPU-computes-the-misses arm: built, correct, **negative above a ~1.2 GiB arena** (per-op dispatch, not CPU compute) | WORKLOG: *CPU-COMPUTES … DESIGN/TEST PLAN*, *… RESULT* |
| 13 | Phase 2 `-sm layer`: "only 1 GPU active" bug fixed (pass-3.5 rebalance) + per-device arenas/MIB | WORKLOG: *EARLIER HANDOVER (session 3)* |
| 14 | Phase 3 `-sm tensor`: per-device slice arenas + Meta delegator; byte-identical | WORKLOG: *PHASE 3 RECORD* |
| 15 | Phase 3 route (1): one layer's routed ops in one split, consumer-driven hook, deferred per-input sync → +93 % over CPU MoE | WORKLOG: *PHASE 3 FINDING*, *PHASE 3 RESULT* |
| 16 | Session 7 attribution: the gap is the per-layer host round-trip (~42 input loops + ~83 ids readbacks/token), not fusion (down fold ~0.2 %) | WORKLOG: *SESSION 7 RESULT* |
| 17 | **Session 7 identity fast path** (the gate): warm `tg1024` 70.3 → 94.2 vs the `-ncmoe 0` 96.4 oracle (97.7 %) | WORKLOG: *SESSION 7 RESULT* |
| 18 | `-sm layer` / 1-GPU curves with identity: 1-GPU 90.0/94.7 (95 %), 2-GPU 72.2/82.9 (87 %) | WORKLOG: *SESSION 7 RESULT* |
| 19 | Cold + fused-kernel hang found; **workaround** `COLD=off` (+ the cold stand-down); deep coherence gate passes | WORKLOG: *SESSION 7 CORRECTNESS FIX + DEEP COHERENCE GATE* |
| 20 | Device-side remap first cut (**broken**, parked) | branch `wip-moe-devmap`; WORKLOG: *SESSION 7 CORRECTNESS FIX* (last paragraph) |
| 21 | **Session 8 item 1**: per-expert-stride tensor-split UVA cold path (`cold_channel_stride`/`cold_row_stride`, host geometry in `moe_cache_get_cold`); axis-0 `ffn_down` verified | WORKLOG: *SESSION 8 ITEMS 1+2* |
| 22 | **Session 8 item 2**: cold-aware fused gate+up+GLU (independent gate-lane cold geometry); `COLD=uva` default restored, cold stand-down removed | WORKLOG: *SESSION 8 ITEMS 1+2* |
| 23 | **Session 8 item 1a**: a cache that cannot serve stands its cache-band fusions down wholesale (fail-soft + tiny-MIB byte-identity bug fixed) | WORKLOG: *SESSION 8 ITEMS 1+2* |
| 24 | **Session 9 item 3**: device-side remap v2 — byte-correct (transition arming + persistent `used_dev` used-list + admission parity), but **10-15 % slower than eager at every residency**, so it does not flatten the cliff; kept opt-in | WORKLOG: *ITEM 3 DEVICE-SIDE REMAP*; branch `wip-moe-devmap-v2`; `exp10-…-devmap.patch` |
| 25 | **Session 10 item 3b-I**: pipelined double-buffered used-list readback + slot-dirty skip — promote **25.1 -> 1.8 us/call, 6.2 -> 0.43 ms/token**; devmap **60 -> 85.6 t/s at h~0.98** (+23 % over eager), plateau gone, cliff halved to 85.6 -> 94.1; byte-identical | WORKLOG: *ITEM 3B PIPELINED PROMOTION*; `exp11-…-pipelined.patch` |
| 26 | **Session 10 attribution**: same-arena `FORCE_DEVMAP` A/B + `rocprofv3` -> residual 1.14 ms/token = ~48 % 240 per-table remap kernels (240/token, 1.57 us) / ~39 % host promotion / ~11 % PCIe; high-`h` cycling is compulsory only (evictions ~0, 4.4 MiB/token) | WORKLOG: *ITEM 3B PIPELINED PROMOTION* |
| 27 | **Session 10 next target (open)**: item 3b-II — eliminate/batch the per-table remap kernels (fold the redundant gate-lane remap; batch; or fuse the slot lookup into the MoE ids read) | README: *Item 3b-II* |
| 28 | **Session 11 r25 rebase**: replayed the 16 campaign commits onto delivery r25 (`81fda69c8`); 2 conflicts (`ssm_gate_beta` kill-switch vs r18/r23, `wait_before_overwrite` vs r22 `GGML_ENV_STR`); new tips `ecac6360c` / `7e6c4cf66`, pre-rebase SHAs in `backup/*-r21`; the 2-GPU `-sm tensor` oracle moved to `de8be4d0c90c`, `-sm layer` unchanged at `15038c19ddc8`, MUL_MAT_ID 929/929 | WORKLOG: *ITEM 3B-II OPTION 1 + r25 REBASE*; `exp12-moe-expert-cache-r25.patch` |
| 29 | **Session 11 item 3b-II option 1**: fold the gate-lane `used_dev` write into the up-lane remap kernel — **240 -> 160 remap launches/token** (captures 960 -> 640), byte-identical at `MIB=1024/9216` + `-sm layer` purity, throughput +~1 % | WORKLOG: *ITEM 3B-II OPTION 1 + r25 REBASE* |
| 30 | **Session 11 item 3b-II option 2**: build the layer's routed `down` remap in the gate+up kernel (from the down's own slot map) and skip the down launch (`remap_fresh`) — **160 -> 80 launches/token** (captures 640 -> 320), byte-identical + width-pure; **and the `rocprofv3` finding that the remap kernels are not the critical path** — the ~0.36 ms/token host promotion (d2h + policy) is, and skipping it collapses to all-cold | WORKLOG: *ITEM 3B-II OPTION 2* |
| 31 | **Session 11b**: the **dirty-table** promotion filter was implemented and **rejected** — the policy cost is the *dirty* tables' fills, not the clean tables' lookups (policy only 734.7 -> 644.1 ms; evictions 0 -> 102) | WORKLOG: *ITEM 3B-II OPTION 2* (the tried negative result) |
| 32 | **Session 12 item-3b**: the **device-side admission policy** (`MOE_EXPERT_CACHE_DEVPOLICY=1`, default off) — one batched LFRU kernel per device per token + in-kernel host->arena fill copy, no used-list D2H / slot-map H2D / fill-list readback.  Win: `forced-devmap MIB=9344` 86.35 -> **89.74**, `MIB=1024` 46.73 -> **50.87**, `MIB=4096` flat; byte-identical, width-pure, MUL_MAT_ID, deep coherence | WORKLOG: *DEVICE-SIDE ADMISSION POLICY + PROGRESS LOG*; branch tip `2632f6011`; `exp13-moe-expert-cache-r25-devpolicy.patch` |
| 33 | **Session 12 instrumentation**: `MOE_EXPERT_CACHE_PROGRESS=N` / `_MS=T` log cumulative admissions/evictions (+ resident/slots/h and per-interval rates) per device — the wall-clock view shows the admit **rate** decays 4500/s -> ~12/s with ~0 evictions, i.e. a hot-set grower, not a churner | WORKLOG: *DEVICE-SIDE ADMISSION POLICY + PROGRESS LOG*, *PROGRESS-LOG CORRECTION* |
| 34 | **Session 12c prefill seed (scaffold)**: `_PREFILL_SEED=1` tallies the prefill routing and bulk-admits the hottest at sizing — **gated off and currently inert**; the prefill upload is intercepted by the block-06 staging before the cache hook, and sizing runs at a load-time/`--fit` warmup before the prompt | WORKLOG: *PREFILL-SEED PROTOTYPE* |
| 35 | **Session 13 (group A)**: device-policy vs host **self-test PASS**; **two cache-enabled pessimisation fixes** (forced `MUL_MAT_ID` offload scoped to the decode band; graph-capture gate scoped to graphs with a cache-band op) — `npl=16` 137->52 regression fixed; **`DEVPOLICY` defaulted ON**; concurrency in-band `npl` 1/2/4/8 = 49/113/172/204 t/s vs 36/50/80/100 no-cache | WORKLOG: *GROUP A PROMOTION GATES* |
| 36 | **Session 14 (B6)**: the **prompt-routing seed** — device prefill tally from `ggml_cuda_mul_mat_id` + `seed_prefill_lazy_locked` bulk-admits the hottest experts as provisional slots at the first decode-band flush.  **`PREFILL_SEED` and `PROVISIONAL` defaulted ON** (kill switch `=0`).  Byte-identical (`15038c19ddc8` / `de8be4d0c90c`), width-pure, `MUL_MAT_ID` 929/929, MTP `n3` 0.79989, coherence green; `-sm layer` `-n 300` `MIB=16384` 60.9 -> **80.8**, `MIB=8192` 52.3 -> **57.9** t/s (beats arbitrary `PREFILL_LOAD`+prov by up to +7 % at mid residency) | WORKLOG: *B6 PROMPT-ROUTING SEED*; patch `exp14-…-b6-prefill-seed.patch` |
| 37 | **Session 15 (B2 first cut)**: **device-side expert gather** for small-ubatch `-ncmoe` prefill (`moe_cache_gather_kernel` + `GGML_SCHED_DEVGATHER`, prefill band; CUDA + Meta) — replaces the per-op routing readback + device sync, on the compute stream so the overwrite/input syncs drop too.  Byte-identical (N=300/1000), `MUL_MAT_ID` 929/929; `-ub 512` prefill `pp2048` 1-GPU +4.5 % / 2-GPU tensor +22.6 %, `pp4096` +6.7 %/+18.5 %, no regression at `-ub 8192`.  **Superseded on the "pageable UVA" point by row 38; default flipped OFF in row 39** | WORKLOG: *B2 FIRST CUT*; patch `exp15-…-b2-devgather.patch` |
| 38 | **Session 16 (B2 measured)**: the gather is **already at PCIe link speed** — `MOE_CACHE_GATHER_NOCOPY` gives 5376 t/s vs 720 at `pp2048 -ub 512` (on-device 5403), and instrumented traffic is 51.2 MiB / 168.6 experts per call, ~49 GB/pass = **~20 GB/s aggregate**, faster than the staged path's DMA (13.3 GB/s).  The gap is each of the 4 ubatches re-uploading its ~65 % subset; **the recorded DMA+compact route is a dead end**; the fix is residency (B1) | WORKLOG: *B2 MEASURED* |
| 39 | **Session 16 (B4)**: the cache **arms and runs on Qwen3.8-Flash-Next** (3x R9700 `-sm tensor`).  IQ4_NL (100 GiB, `-ncmoe 48`): 432 tables, 388/512 slots/table (75.8 %), arena 49 GiB, device-remap armed; coherent + deterministic, decode **14.9 -> 29.0 t/s (+95 %)**.  **Transparency PASS** on IQ3_XXS (fits `-ncmoe 0`): oracle `77c6f546460d` == cache-on `-ncmoe 99` `77c6f546460d`, cache-off (CPU MoE) `32576231856e`; 16.5 -> 29.0 t/s (+76 %).  **Width purity PASS**: `plain == draft-mtp n_max 1 == 3 == 7`.  Also **found the B2 gather corrupts qwen4exp prefill** -> **defaulted OFF** (commit `c5bbb7ee2`).  `PREFILL_SEED`/`PROVISIONAL` default-ON retained | WORKLOG: *B4 VALIDATION*; patch `exp16-…-b4-gather-off.patch` |
| 40 | **Session 17 (B1)**: single R9700 8K prefill `-ub 512/1024/2048/8192` `~990/1474/1519/1461` t/s (was `220/331/448/581`).  Merged routed-MoE bands (31 inputs) were never staged, so the 450 MiB expert weight uploaded serially behind host event syncs; now `sched_stage_issue` counts host-weight inputs and defers routed tables to the **device gather (default ON, unsplit only)**, `GGML_SCHED_EVENTS` defaults ON, and host->device split inputs copy asynchronously.  Gates: `15038c19ddc8` / `de8be4d0c90c`, width purity `2ede4fe056cc`, `MUL_MAT_ID` 929/929, 3-device qwen4exp coherent (`359ff4337837`) | WORKLOG: *B1*; patch `exp17-…-b1-prefill-gather.patch` |
| 41 | **Session 17 addendum (r26 rebase)**: the campaign is rebased onto delivery **r26** (`0d58404e1`); the scheduler half of B1 is delivery block-06 now, so the duplicated hunks are dropped.  Tip `6ca5c1c77`, patch `exp18`.  Rebased 8K prefill `-ub 512/2048/8192` `~1044/1627/1761` t/s; gates `15038c19ddc8` / `de8be4d0c90c` / `MUL_MAT_ID` 929/929 green | WORKLOG: *B1*; patch `exp18-…-r26-b1-rebase.patch` |
| 42 | **Session 18 (item 1)**: the session-16 gather attribution was **wrong** — the slice geometry was correct all along (D2H of every routed expert on all 3 devices matched the host byte-for-byte); the qwen4exp prefill corruption was the pruned gather missing the host path's MMQ **expert-table tail pad** (`min(expert_size,512)` past each routed group's last expert).  The gather now pads the next expert's slice; `.moe_cache_gather` re-enabled, tensor-split arm default ON (`GGML_META_GATHER_NOPAD=1` / `GGML_SCHED_DEVGATHER=0` kill switches).  Gates: `de8be4d0c90c` / `15038c19ddc8`, 3-device IQ3 cache-on == `-ncmoe 0`, deep coherence (12k essay rc=0, 13 sections, `## Conclusion`), `MUL_MAT_ID` OK, `pp2048 -ub 512` 2-GPU tensor prefill **609 -> 724 t/s (+18.9 %)**.  Tip `2376ac6cf`, patch `exp19` | WORKLOG: *ITEM 1 FIXED*; patch `exp19-moe-expert-cache-r26-b2-tensorpad.patch` |
| — | **NEXT (open)**: **1.** **B3** (fuse the slot lookup into the MoE ids read, ~2-3 % decode), **2.** the user graph-input copies (~5 s/pass headroom).  **B1 DONE, B2 both arms default-ON, B4 COMPLETE, B6 done, item 1 DONE** | README: §0 (NEXT SESSION); WORKLOG: *ITEM 1 FIXED*, *B1*, *B2 MEASURED*, *B4 VALIDATION* |

### Reference tables (kept in the WORKLOG, not duplicated here)

* 2-GPU `-sm layer` quick-reference harness + traps — WORKLOG: *2-GPU `-sm layer` quick reference*.
* State-in-one-screen, build/run, and the H1/H2/H3 detail — WORKLOG: *State in one screen* onward.
* Prior art (GenerelSchwerz `moe-cache`, R9V, Strata) and the phased plan — WORKLOG sections 0-3.
* Traps learned (do not re-derive) — WORKLOG: *Traps learned this session*.
