# HANDOVER — issue #50: op-offload H2D staging ring (merged)

Updated 2026-09-26, after the PR #51 merge.  `README.md` is the evidence record; this is the state.

## 1. State: the merge is done

`h2d-stage.patch` (521 lines, this directory) is now the **merged** patch.  It applies cleanly on top of
`scripts/apply-all.sh` at r10 and takes the best of both designs:

* **PR #51's redirect** — the consuming op reads the ring slot directly (`input_cpy->data` pointed at the
  slot, restored at the next issue), so the bytes move once instead of twice.  The D2D was the whole
  gap on a fast link (~10 % on the reporter's 55 GB/s box).
* **PR #51's `GGML_SCHED_EVENTS`** — ported; it is what lets the redirect overlap at all in their design.
* **Our bounded arena** (`GGML_SCHED_STAGE_MAX_MB`, default 2048 MiB) with a clean fallback: a *partially*
  staged ubatch is worse than either (pruned-path inputs do an ids readback + device sync), so an
  over-large ring disables staging rather than collapsing.
* **Our link-calibrated gate** (floored above the widest verify batch, so decode/verify can never stage).
* **8 slots** by default, runtime-tunable (`GGML_SCHED_STAGE_SLOTS`) — the second lever after the D2D.
* **Our restore-at-issue** plus a **tripwire assert** (the reporter's suggestion) that aborts if a
  redirected split input ever reaches a copy path.

Validated: purity byte-identical on gfx1201 and gfx1100 (both models); decode untouched; the 22.66 GB
Q4_K_M model runs on the 24 GB 7900 XTX without OOM.  Numbers and the full matrix: `README.md` §"Merged
design".

## 2. Files

| file | what |
|---|---|
| `h2d-stage.patch` | **the merged patch** (r10 base, 5 files: `ggml-backend-impl.h`, `ggml-backend-meta.cpp`, `ggml-backend.cpp`, `ggml-cuda/common.cuh`, `ggml-cuda/ggml-cuda.cu`) |
| `reporter-h2d-ring.patch` | snapshot of PR #51's patch (unchanged since review) |
| `README.md` | evidence record: measurements, design, the merge section |
| `h2d-bw.cpp` | standalone H2D probe |

Env knobs: `GGML_SCHED_STAGE` (enable), `GGML_SCHED_STAGE_MODE` (1 redirect default / 0 D2D),
`GGML_SCHED_STAGE_SLOTS` (8), `GGML_SCHED_STAGE_MAX_MB` (2048), `GGML_SCHED_STAGE_MIN_TOKENS` (override
the calibrated gate), `GGML_SCHED_EVENTS` (event wait instead of a full synchronize with one copy).

## 3. What remains

1. **`-sm tensor` / meta backend.** The meta backend lists the stage hooks as `NULL`, so staging is
   **inert** under tensor split (the scheduler finds no `stage_h2d_gbps` and no `stage_buffer`).  Decide:
   forward the hooks (the meta backend must pick the underlying device for the split) or keep it inert and
   document.  Neither our patch nor PR #51 has been exercised there.
2. **The gate battery** (the handover's original list): `test-backend-ops`, the `W=1..8` width probe
   (`GREEDY-PURITY.md` §§10-11), MTP (`draft-mtp n3` + adaptive, acceptance + text), a deep-context
   prefill, and a concurrent-server soak.  Only the greedy-text purity gate has run.
3. **Have the reporter run the merged patch** on their PCIe5 x16 box (55 GB/s) — they offered, and it is
   the one link neither of us can measure locally.
4. **Delivery decision**: block 15 owns the `fattn_stage` arena precedent, block 11 the CUDA prefill-graph
   skip.  Then `scripts/validate-set.sh` + a patch/`release.json` regen.
5. **Build fix (unrelated but active)**: the HIP build emits no `.d` files and ccache then misses header
   changes, giving silently stale objects (`README.md` §4).  Until it is fixed, `rm` the specific
   `ggml-cuda/*.o` after any `common.cuh` edit — a `common.cuh` layout change also requires a full CUDA
   rebuild for ABI consistency.
6. Latent prior art that composes: the decode-only expert cache in issue #47's comment (upstream PR
   #27861) — the ring is the latency lever, the cache the volume lever.

## 4. Open design note

The redirect holds *structurally*, not by invariant (the reporter's audit: input copies are
`ggml_dup_tensor_layout` duplicates with `data == NULL` until the allocator sets it; `n_copies > 1` and
the meta backend are unexercised).  Our restore-at-issue makes the restore structural for the scheduler's
own paths, and the tripwire assert catches a future redirected write.  If someone later needs
`n_copies > 1` or tensor split, re-audit before assuming it still holds.

## 5. Commands

```bash
# apply + build (soar; fingon is ~/llama-r10 with the same recipe)
cd ~/llama.cpp && git diff -- ggml/src/ggml-backend-impl.h ggml/src/ggml-backend-meta.cpp \
  ggml/src/ggml-backend.cpp ggml/src/ggml-cuda/common.cuh ggml/src/ggml-cuda/ggml-cuda.cu > /tmp/h2d.patch
cd ~/llama.cpp && git apply /tmp/h2d.patch
rm -f build-rocm/ggml/src/ggml-hip/CMakeFiles/ggml-hip.dir/__/ggml-cuda/*.o
cmake --build build-rocm --target llama-bench llama-cli -j 16

# sweep (soar)
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1201/lib HIP_VISIBLE_DEVICES=0
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
GGML_SCHED_STAGE=1 ./build-rocm/bin/llama-bench -m $M -ncmoe 99 -fa 1 -p 8192 -n 1 -b 8192 -ub 4096
```
