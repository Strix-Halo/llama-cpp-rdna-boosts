# HANDOVER — host 2nd-level expert cache (pinned, DIO-filled)

For a fresh session picking up `wip/host-expert-dio-cache/`.  Read
[`README.md`](README.md) first for the motivation and design; this file is the
"where the code is and what to do next" layer.

## State when this was written (2026-10-08, r33)

* **Delivery r33 is shipped** (`v16-a55e952b8-r33`, tag pushed, delivery `main`
  pushed, personal fork `rdna-boosts` force-pushed to `6e567349c`).  It removes
  `--host-experts mmap` / `LLAMA_MMAP_HOST_EXPERTS` and makes host experts always
  pinned, plus the cache safety rails and the `MOE_EXPERT_CACHE_MIB >= 2048` floor.
  See `WORKLOG.md` "2026-10-08 (r33)".
* **Nothing of the pool exists yet** — this campaign is Phase 0 only (the removal).
* Canonical worktree used to cut r33: `/tmp/llama-canon` (branch `canon-r32`,
  16 blocks, tip `6e567349c`).  Ephemeral; recreate with
  `git worktree add /tmp/llama-canon a55e952b8` + `RDNA_BRANCH=... scripts/apply-all.sh`.
* Build tree: `~/llama.cpp/build-rocm-hybrid` (the `~/llama.cpp` working tree is
  byte-identical to the r33 tip, so its binaries ARE the r33 binaries).  Build:
  `cd ~/llama.cpp && BUILD_DIR=build-rocm-hybrid EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714`
  (ccache makes a re-configure cheap; the script `rm -rf`s the build dir).

## The problem in one paragraph

The expert cache keeps `-ncmoe`/`-cmoe` experts in a per-device pinned
(`ROCm_Host`) master and a VRAM arena.  The full pinned master costs ~18.6 GB
non-swappable RAM for the 35B-A3B Q4_K_M `-ncmoe 40`.  The r28 "use the pageable
model mmap so the OS page cache is the host tier" idea is dead on RDNA (no XNACK
-> a kernel cannot read a pageable address; issue #116).  The replacement is a
**bounded** pinned host pool that is filled on demand from the GGUF with DirectIO
(O_DIRECT, no page-cache copy), with LRU eviction.  That bounds the non-swappable
RAM at a chosen budget.

## The hard constraint (read this before designing)

Today, whenever the cache **declines** for a table the scheduler uploads the used
experts from `tensor->data` (the whole host master) via
`ggml_backend_sched`'s host-input copy path.  A bounded pool cannot back that full
pointer, so **the pool only works if the cache owns every expert access for a
pooled table** — including the prefill/tall-batch path that currently declines or
goes through `moe_cache_gather`.  The plausible route is to force
`moe_cache_gather` for pooled tables (`sched_input_gatherable()` must return true
for them; see `ggml/src/ggml-backend.cpp`) and have the gather source the pool
(loading from disk on a miss).  That is a re-architecture of block 06's expert
path, not a delta — hence the campaign.

A second, simpler alternative is acceptable if the above proves too invasive:
keep the full pinned master for the scheduler fallback, and let the pool be an
*additional* bounded VRAM-adjacent staging tier.  That does not reduce the pinned
RAM, so it does not meet the goal — record it as a rejected option, not a plan.

## Where the seams are (all paths relative to the repo root)

### Cache (`ggml/src/ggml-cuda/moe-expert-cache.cu`)

| piece | function | notes |
|---|---|---|
| registration | `moe_cache_table()` | binds `t.host`, `t.host_dev`; stores `host_bytes`, `src_off`, `host_pitch`, `split_axis`, `n_experts`, `expert_bytes`, `device` |
| device-alias resolution | `bind_host_dev_locked()` | now only sets `host_dev` when `cudaHostGetDevicePointer` succeeds (r33) |
| residency + fill | `access_locked()` | fills from `t.host` with `cudaMemcpyAsync` (1-D) or `cudaMemcpy2DAsync` (axis-0); this is the natural hook for a pool fill |
| eager hook | `moe_cache_update_host()` | builds the remap; **r33 added an early decline for a partial pageable axis-0 table** — the pool changes this decision |
| device gather | `moe_cache_gather_host()` | reads `weight->data` in a kernel; the pooled path should read a pool slot instead |
| cold geometry | `moe_cache_get_cold()` | hands the mmvq MoE kernel `cold_base + e*stride`; a pool slot has the same geometry, so the kernel is unchanged |
| full-residency fast path | `alloc_table_locked()` | the identity fill copies the whole table with `cudaMemcpyAsync`/`2DAsync`; a pooled table has no identity path |
| prefill seed | `apply_prefill_seed_locked()` | reads `host_dev` in `moe_cache_seed_fill_kernel`; pool-aware or disabled for pooled tables |
| device policy | `moe_cache_policy_kernel` / `build_policy_descs_locked()` | reads `d.host_dev`; pool-aware or off for pooled tables |
| env parse | `parse_env()` | add `MOE_HOST_POOL_MIB` here (default 0 = off; the full pinned master stays the default until the pool is validated) |

### Loader (`src/`)

| piece | where | notes |
|---|---|---|
| per-tensor file+offset | `llama_tensor_weight { idx, file, offs, tensor }` in `llama-model-loader.h`; `load_data_for()` in `llama-model-loader.cpp` does `file->seek(offs); file->read_raw(data, n_size)` | the `(file, offs, n_bytes)` triple the pool needs |
| DIO read primitive | `llama_file::read_raw()` -> `read_aligned_chunk()` in `llama-mmap.cpp` (`O_DIRECT`, `LLAMA_DIRECT_IO_BUFFER_SIZE` 64 MiB bounce, `-DIO` aligned) | reuse for a pool fill; needs `--load-mode dio` or an explicit O_DIRECT fd |
| host buft | `ggml_backend_cuda_host_buffer_type_dev()` in `ggml/src/ggml-cuda/ggml-cuda.cu` (`cudaMallocHost`) | a pooled allocator could live here |
| file lifetime | `llama_model_loader::files` are destroyed after load | the pool must keep an fd open (its own `open(O_RDONLY|O_DIRECT)`) or store the path(s) |

**Plumbing recommendation (re-use, minimal coupling):** add a
`moe_cache_set_host_source(const void * tensor_data, const char * path, size_t offs,
size_t expert_bytes, size_t src_off, size_t host_pitch, int split_axis, int n_experts)`
in ggml-cuda, called from the loader for each host `*_exps` tensor after
`load_data_for`.  The cache keys it by `tensor_data` (== `t.host`) at
registration.  Store the **path + offset**, and open the pool's own fd lazily, so
the loader's `llama_file` lifetime is irrelevant.  Keep the current full pinned
allocation until `MOE_HOST_POOL_MIB` is set.

## Suggested Phase order (do not skip validation)

1. **Plumbing, inert.**  Add the registry + loader call + `MOE_HOST_POOL_MIB`
   parse.  With the env unset, behaviour must be byte-identical (no pool alloc,
   no new fd).  Validate: dense 4B coherence + `validate-set.sh` (the code is a
   WIP patch, so this is the campaign's own check, not the delivery's).
2. **Pool + cold-read fill.**  `MOE_HOST_POOL_MIB=N` allocates an `N` MiB/device
   pinned pool keyed by `(table, expert)`; `moe_cache_get_cold()` and the arena
   fill source a pool slot; a pool miss does the DIO read synchronously before the
   consuming kernel is enqueued.  Keep the cache's decline paths disabled while
   pooled (see the constraint).  Validate: greedy text == the full-pinned build,
   `test-backend-ops -o MUL_MAT_ID`, prefill-logit KLD, RSS within budget.
3. **No-fallback.**  Make the scheduler route every pooled-table expert upload
   through `moe_cache_gather` (or prove it never touches `tensor->data` for a
   pooled table).  This is where the real risk is.
4. **Prefill.**  A pooled table's prefill either goes through the pool-aware
   gather or the loader keeps a full master for prefill only (document which).
5. **Promote** with the maintainer's go-ahead: fold into block 06, regenerate,
   re-run the full gate matrix.

## Validation commands that matter here

```sh
# coherence (dense, golden 1c5d32ac537d)
HIP_VISIBLE_DEVICES=0,1,2 ~/llama.cpp/build-rocm-hybrid/bin/llama-cli \
  -m /llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf -ngl 99 -sm tensor -mg 0 \
  -p "The capital of France is" -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn

# MUL_MAT_ID
HIP_VISIBLE_DEVICES=0 ~/llama.cpp/build-rocm-hybrid/bin/test-backend-ops -o MUL_MAT_ID

# prefill-logit KLD gate (base already recorded; Qwen3.8-27B Q8_0)
bash scripts/gate-prefill-logits.sh

# MoE MTP smoke (35B-A3B Q4_K_M, prose prompt fabdec65...; full gate is -n 3000)
#   --spec-type none  vs  --spec-type draft-mtp, --reasoning off, -n 2000, bf16 KV
```

Thread sizing is not optional: `/usr/local/bin/pin_gpu_irqs.sh` puts the GPU IRQs
on cores 13-15; size `-t` to leave them free (`-t 8` used here).  Never run
parallel benches.

## Open bug parked here (not #116)

`--host-experts pinned` + `-sm tensor` + a **partial** arena still faults at a
host address with **no** pageable warning: reproduced at `MOE_EXPERT_CACHE_MIB=1024`
and `=2048` (on 3x R9700, Q4_K_M `-ncmoe 40`, `-sm tensor`).  It is a host
over-read (the reporter's issue #115 note describes MMQ reading one K tile past an
expert's last row), and the cache's host masters are the suspicious side.  It
should be root-caused before the pool lands, because the pool makes every host
read a controlled path.  Evidence in this session's notes / `WORKLOG.md` r33
"Follow-on".

## Environment pitfalls seen this session

* A GPU fault wedges the box: subsequent runs hang at random points (often before
  cache registration) even though a plain HIP probe still succeeds.  Recover with
  `sudo amd-smi reset -G -g all` (or `/sys/kernel/debug/dri/*/amdgpu_gpu_recover`)
  before trusting a hang as a real signal.
* `MOE_EXPERT_CACHE_MIB` below 2048 now aborts (r33) — do not use 1024 for A/B.
* `--host-experts mmap` now errors (r33) — do not expect it to run.
* The 27B Q8_0 (29 GB) fits one 32 GiB R9700 with `-ngl 99` for the KLD gate.
