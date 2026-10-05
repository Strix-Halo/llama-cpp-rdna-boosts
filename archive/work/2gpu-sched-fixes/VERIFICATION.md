# 2gpu-sched-fixes (issue #103, PR #104): PROMOTED in `v16-a55e952b8-r12`

Status: **PROMOTED (2026-10-05, release `v16-a55e952b8-r12`) into block 06 (patch 0002) and
block 13 (patch 0001).**  Contributor [PR #104](https://github.com/stew675/llama-cpp-rdna-boosts/pull/104)
by @briansp2020, fixing the two failures in issue
[#103](https://github.com/stew675/llama-cpp-rdna-boosts/issues/103).

| patch | file | owning block | what |
|---|---|---|---|
| `0002-ggml-backend-order-cross-device-split-input-copies-a.patch` | `ggml/src/ggml-backend.cpp` | block 06 | order a cross-device split-input copy after the destination backend's queued work |
| `0001-moe-expert-cache-validate-device-and-layer-on-alias-.patch` | `ggml/src/ggml-cuda/moe-expert-cache.cu` | block 13 | trust an expert-cache alias only if its table is on the calling device (and in the op's layer) |

The two author patches are kept verbatim in this directory; `prefill-rebalance-harness.patch` is the
scratch A/B knob used to reproduce the race on this box (NOT part of the delivery).

## Why the reporter's stock repro did not fire here

On the maintainer's 3x R9700 box, stock r11 was correct for every configuration the reporter listed
(ROCm 7.14.1 and ROCm 10.0.0, `llama-cli` and `llama-server`, `--n-cpu-moe` 29 and 48, `-t 8` and
`-t 16`, `GGML_SCHED_STAGE_MODE=0`, `GGML_SCHED_STAGE_SLOTS=4`, `MOE_EXPERT_CACHE_MIB=6144`, both
GPU pairs).  The reporter's cards sit on PCIe 5.0 **x8** slots; this box reports all three at
**x16**, so the peer copies are roughly twice as fast and the window is smaller.

## The harness that reproduces it

`prefill-rebalance-harness.patch` adds `GGML_SCHED_REBALANCE_PREFILL`, which lifts block 06's
`ne[2] > 8` guard and so rebalances prefill-width `MUL_MAT_ID` host-weight ops onto the owning
device.  That is a legitimate scheduling change (not a bug injection); it just moves the cross-device
copies and reopens the reporter's window on this hardware.  It is a re-validation harness, **never
part of the delivery** (it is the same pass that block 06 keeps off for prefill because it costs
42 percent at `-ub 8192`).

## Reproduced FAIL -> PASS (ROCm 10.0.0, 2x R9700, qwen4exp UD-IQ3_XXS + shared Q8_0 MTP)

The reporter's exact flags (`-sm layer --n-cpu-moe 48 -ub 2048 -b 2048 -ctk/-ctv q8_0`,
`--spec-type draft-mtp -md mtp-...-Q8_0.gguf -ngld 99 -otd exps=CPU --spec-draft-n-max 3
--spec-draft-p-min 0.5 -ctkd/-ctvd q8_0`, `GGML_SCHED_SYNC_GRAPH_INPUTS=1`,
`LLAMA_MTP_DRAFT_OP_OFFLOAD=0`, `LLAMA_MTP_SPARSE=0`), greedy, 1 / 580 / 900-line needles.

| build (all with the harness knob) | events | prompt | result |
|---|---|---|---|
| stock r11 | on | 580-line | `!!!!` (2/2), generation 8.2-8.4 t/s |
| stock r11 | on | 900-line | `!!!!`, generation 7.8 t/s |
| stock r11 | `GGML_SCHED_EVENTS=0` | 580-line | correct ("580 buildings"), generation 24.4 t/s |
| r11 + patch 0002 only | on | 580-line | correct, generation 24.8 t/s |
| r11 + patches 0001 + 0002 | on | 580-line | correct (3/3), generation 24.1-25.5 t/s |
| r11 + patches 0001 + 0002 | on | 900-line | correct ("899 buildings"), generation 23.2 t/s |

This is exactly the reporter's symptom (`!!!!`), their workaround (events off is correct), and the
PR's fix (patch 0002 makes events-on correct).  Patch 0001 is inert without `MOE_EXPERT_CACHE_MIB`.

## What this does NOT validate

**Patch 0001's page fault was not reproduced here.**  No `moe_cache_tally_kernel` fault and no
stale-alias warning fired on this box even with the harness and `MOE_EXPERT_CACHE_MIB=6144` (the
harness run produced the race `!!!!`, not the fault).  The device/layer check is retained as a
clearly-correct guard: a table whose arena lives on another device must never be used by the calling
device's op.  It needs a reporter-side rerun against a tree that also has 0002 to be exercised on
its own.

## Delivery verification

- The amended chain is `a55e952b8` + 16 blocks, new tip `66ecd1d2558523a924dad380f575c51d713e6f3c`,
  net tree `1cd1d27e9467a1508f4b43eb98c18350055585bd`.
- `git diff ea8658864..66ecd1d2` is exactly the two author patches (40 added lines), i.e. the net
  tree is r11 plus these fixes and nothing else.
- `scripts/validate-set.sh` green: per-file sha256, strict 16/16 `git am` on a fresh `a55e952b8`
  tarball, applied tree matches `release.json`.
- Behavior preservation: same-seed greedy on Qwen3.5-4B-Q8_0, 2 GPUs, `-sm layer` and `-sm tensor`,
  stock r11 versus r11 + patches gives byte-identical generated text (timing line aside).

## Open follow-ups (see `TODO.md`)

1. The new wait is the scheduler's first cross-backend `ggml_backend_event_wait`; several backends'
   `event_wait` assume their own event type, so a mixed CUDA+SYCL/Vulkan scheduler would need a
   same-family guard.
2. Neither change has a per-change env kill-switch (promotion rule).  `GGML_SCHED_EVENTS=0` disables
   0002 only by turning off all per-split events, at a measured prefill cost.
3. Patch 0001 still needs a reporter-side FAIL -> PASS.
