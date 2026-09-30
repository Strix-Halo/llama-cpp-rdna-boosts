# B3 brief — fuse the slot lookup into the MoE ids read

**Status: OPEN.**  Campaign tip **`2376ac6cf`**, patch **`exp19-moe-expert-cache-r26-b2-tensorpad.patch`**
(r26 base).  Worktree `~/llama-decode`, branch `wip-moe-devmap-v2`, build `build-rocm`.
Backup refs: `backup/wip-moe-devmap-v2-r26-item1` (`2376ac6cf`), `backup/wip-moe-devmap-v2-r26-b1`
(`6ca5c1c77`).

This is the self-contained handover for README §0 item 2.  Line numbers are for tip `2376ac6cf`; re-grep
before editing.

## Goal

Remove the per-table `moe_cache_build_remap_kernel` launches (80/token after options 1+2) by making the
hot mmvq / fused MoE kernels take `slot_dev` + `n_res` + the raw device `ids` and do `slot[ids[i]]`
**in-kernel**.  The `used_dev` routing record that the deferred promotion / batched device policy replays
must also move into those kernels, because each table (gate / up / down) needs its own used list.

**Reward: ~2-3 % decode** (session 11b: the remap kernels are `0.27 ms/token` with the GPU ~29 % busy;
halving the launches did not move throughput).  **Risk:** a template/dispatch change across the mmvq +
fused paths plus the used-list accounting.  Gate it and A/B at a fixed `h`; a negative result with
attribution is a valid outcome.

## Current mechanism (what B3 replaces)

State lives in the `table_t` (`moe-expert-cache.cu` ~line 106): `slot_dev` (device `expert -> slot`, -1 =
cold) and `used_dev` (device `int32[n_experts*8]`, the raw routing the remap kernel last read).  The
public view handed to consumers is `struct moe_cache_devmap` (`moe-expert-cache.h` ~line 66):
`{ slot_dev, used_dev, n_experts, n_res, remap_fresh }`.  `n_res` is the first **cold-encoded** id ==
resident slots + the CPU zero slot.

Flow per layer per token (devmap = partial residency):

1. `moe_cache_redirect_fused()` (`moe-expert-cache.cu` **:3093**) resolves the up table and (for the fused
   gate+up) the gate table via `moe_cache_get_table()`; sets `src0_cpy->data = arena`.
2. `moe_cache_launch_remap()` (**the target**) launches `moe_cache_build_remap_kernel` (**`:1517`**): one
   thread per `(tok, j)`, reads `e = ids[i]`, computes `s = slot[e]`, writes
   `remap[i] = (s >= 0) ? s : (n_res + (e >= 0 ? e : 0))` (the **cold encoding**), and writes
   `used[i] = e`.  In the same launch it can fold the gate lane's `used2` write, and a sibling `down`
   table's `remap2`/`used3` (option 2).
3. `redirect_fused` points `ids_cpy->data` at `remap` with strides `{4, nu*4}`.  The consumer kernels index
   the compact arena with those already-remapped ids.
4. `remap_fresh` (cleared by that table's promotion after the graph) skips the down's redundant launch.
5. `used_dev` is consumed by `moe_cache_promote_host()` (deferred promotion, ~:2740-2830) and by the
   batched device policy (`moe_cache_policy_flush` **:3007**, descriptor `d.used = t.used_dev` at :2871).

Call sites / consumers:

* `ggml/src/ggml-cuda/ggml-cuda.cu`: fused gate+up+GLU `moe_cache_redirect_fused(glu, src0, gate, ids, …)`
  at **:5651**; the `[MUL_MAT_ID, MUL]` down fold at **:5851**; the per-op consumer in
  `ggml_cuda_mul_mat_id` at **:2604**.
* `ggml/src/ggml-cuda/mmvq.cu`: **`mul_mat_vec_q_moe`** (**`:1315`**) — the one-warp-per-token MoE mmvq
  kernel; `ids[channel_dst + token_idx * ids_stride]` is the channel (`:1377`).  Launchers
  `mul_mat_vec_q_moe_launch` (`:1571`) + the `has_ids` arms (`:1506`/`:1539`).  The dense
  `mul_mat_vec_q` ids path (`:732`/`:1018`) also exists but is not the cache consumer.
* MMQ/prefill is **out of scope**: the cache serves the decode/verify band only (`n_tok <= 8`).

## The change

1. Pass `slot_dev` (and `n_res`) to the mmvq/fused kernels **instead of** a materialized `remap_dev`, and
   resolve the channel in-kernel: `channel = (s >= 0) ? s : (n_res + e)`, with `s = slot_dev[e]`; a null
   `slot_dev` remains the identity fast path (raw ids already index the arena).
2. Move the `used_dev` write into the same kernels: the up/gate consumer writes the up + gate used lists;
   the down consumer writes its own.  Preserve the exact raw-`e` values (`used_dev` is the routing, not
   the slot).
3. Retire the remap kernel + the `remap` plumbing; `remap_fresh` and the option-1/option-2 folding become
   unnecessary.
4. Keep `moe_cache_promote_host`'s used-list readback as the **A/B reference** until the device path is
   proven; do not delete it in the same change.

## Subtleties / traps (do not re-derive)

* **The fused gate+up kernel reads the UP table's remap for BOTH lanes.**  So the in-kernel lookup must
  use the up `slot_dev` for both lanes, and write both the up and gate `used_dev` lists.  The gate lane's
  remap *value* is unused — only its used list.
* **The down remap is built from the down's OWN slot map** (`slot2`), not the up map.  With B3 the down
  consumer does its own in-kernel lookup from the down's `slot_dev`; the `used3` write moves there.
* **Cold encoding must be byte-exact:** `remap[i] = s >= 0 ? s : n_res + e`.  `mul_mat_vec_q_moe` indexes
  cold channels at `n_res` onward (item 1's per-expert-stride UVA cold path) — a wrong encoding reads the
  wrong expert silently.
* **`used_dev` has one writer today** (the remap kernel).  With multiple consumer kernels it must not be
  double-written or raced; the fused gate+up writes two tables, the down fold one.
* **Capture safety:** the devmap decision must stay constant per graph shape, and `slot_dev` is a stable
  pointer after sizing.  The current kernel-per-table design is capture-safe; preserve that.
* **Identity vs devmap:** `remap == nullptr` (and `slot_dev == nullptr`) is the identity path used when
  `slots == n_experts`; do not force a slot lookup there.
* `used_dev` capacity is `n_experts * 8` (the decode-band `n_used * n_tok` bound) — keep it.

## Gates (all six, README §1) and the A/B

* Byte-identity: `15038c19ddc8` (1-GPU `-sm layer`), `de8be4d0c90c` (2-GPU `-sm tensor`); the 3-device
  IQ3_XXS transparency oracle.
* Width purity `none == n1 == n3 == n7`; `-sm layer` regression; `test-backend-ops -o MUL_MAT_ID` 929/929;
  MTP `n3`; deep coherence (`coherence-essay-prompt.txt`, `-n 12000 -c 16384`).
* **Deterministic counter:** `MOE_EXPERT_CACHE_TIMING=1` prints the remap-launch count — it must reach
  **0/token** after B3.  Compare at a fixed `h` (`MOE_EXPERT_CACHE_FORCE_DEVMAP=1 MOE_EXPERT_CACHE_MIB=9344`
  isolates the devmap machinery at `h=1`).

## Files

| concern | location |
|---|---|
| remap kernel / launcher / devmap struct / redirect / promotion | `ggml/src/ggml-cuda/moe-expert-cache.cu` (:1517, :1565, :3093, :2740, :3007), `moe-expert-cache.h` (~:66) |
| mmvq MoE consumer | `ggml/src/ggml-cuda/mmvq.cu` (`mul_mat_vec_q_moe` :1315, launchers :1506/:1539/:1571) |
| fused / per-op call sites | `ggml/src/ggml-cuda/ggml-cuda.cu` (:5651 gate+up, :5851 down fold, :2604 per-op) |
