# Gather-mode heuristic: §30.6's device-count premise is DISPROVEN — the driver is ubatch width (2026-10-06)

**Status:** open — the §30.6 follow-up ("device-count-aware gather-mode heuristic") is wrong on the
current base. The optimum tracks the **physical ubatch width**, not the device count, and a naive
device-count rule would badly regress. No delivery change has been made; this tree holds the A/B
instrument and the data.

## Background

`ggml_backend_cuda_stage_gather()` (`ggml/src/ggml-cuda/ggml-cuda.cu`) stages one device's strided
expert slice when the split-path staging is active (`-sm tensor -ncmoe`). It picks between two shapes:

- **host gather** — `n_copies` host `memcpy`s into the pinned ring slot, then one 1-D H2D (moves the
  compacted slice, 1x volume, but the gather is on the CPU);
- **whole-range + D2D compaction** — one whole-range H2D (2x volume for `ffn_down`: `stride/width = 2`)
  then a `cudaMemcpy2DAsync` D2D to compact.

The existing heuristic is `host_gather = n_copies <= 4096`, i.e. shape-only. For Q4_K_M 35B the coarse
`ffn_gate/up` slices have `n_copies = 256` (host gather) and the fine `ffn_down` slice has
`n_copies = 524288` (compact). `archive/work/tensor-split-expert-split/README.md` §30.6 measured the 3-GPU
ub-8192 case losing to host gather and proposed a **device-count-aware** heuristic ("auto (device-D2D for
the fine `ffn_down` slice) 4694, host gather 4905 … a device-count-aware heuristic is the obvious
follow-up").

## Method

`ab-instrument.patch` adds `GGML_CUDA_GATHER_MODE` (0/auto, 1/force host, 2/force compact) to
`ggml_backend_cuda_stage_gather`. Applied to the delivery tip `ea8658864` in the worktree
`~/llama-gather-ab` (branch `wip/gather-mode`); build with `~/bin/build-llama-rocm-714`.

Harness (3x R9700 gfx1201, Qwen3.6-35B-A3B UD-Q4_K_M, one ubatch per pass unless noted):

```
HIP_VISIBLE_DEVICES=0,1,2 GGML_CUDA_GATHER_MODE=$M ./build-rocm/bin/llama-bench \
  -m .../Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ngl 99 -ncmoe 99 -fa 1 -sm tensor -t 8 \
  -p $N -n 0 -b $N -ub $N -r 3
```

## Result — 3 GPUs, single ubatch (`-p N -b N -ub N`)

| ub | auto (0) | host gather (1) | compact (2) | winner |
|---:|---:|---:|---:|---|
| 2048 | 1772.24 | 1328.50 | — | auto, **+33.4 %** |
| 4096 | 3423.90 | 2605.75 | — | auto, **+31.4 %** |
| 5120 | 4022.18 | 3244.66 | — | auto, **+23.9 %** |
| 6144 | 4234.55 | 3893.18 | — | auto, **+8.8 %** |
| 7168 | 4277.46 | 4417.49 | — | host, +3.3 % |
| 8192 | 4463.92 | 4698.37 | 4256.51 | host, **+5.25 %** |

Auto's advantage falls monotonically with width and flips between 6144 and 7168.

## Result — 2 GPUs (same base, same model)

| ub | auto (0) | host gather (1) | winner |
|---:|---:|---:|---|
| 2048 | 1701.66 | 1371.49 | auto, +24.1 % |
| 8192 | 4393.59 | 4761.10 | host, +8.4 % |

**Identical behaviour to 3 GPUs at the same width.** Device count is not the driver.

## Result — logical vs physical width (3 GPUs, `-p 8192`)

| `-b` | `-ub` | ubatches | auto (0) | host gather (1) | winner |
|---:|---:|---:|---:|---:|---|
| 8192 | 2048 | 4 | 1766.97 | 1354.50 | auto, +30.5 % |
| 8192 | 4096 | 2 | 3404.17 | 2608.98 | auto, +30.5 % |
| 6144 | 6144 | 2 | 3080.93 | 2518.23 | auto, +22.3 % |

The **physical** ubatch width (`-ub`) decides; the logical batch does not.

## What this means

1. **§30.6's device-count-aware heuristic is wrong for the current base.** A rule like "host gather at
   >= 3 devices" would regress ub 2048/4096/5120/6144 by **9–33 %** to buy +5 % at 8192.
2. **The real driver is the compute/upload ratio**, which the ubatch width sets: at small widths the
   upload is exposed and the host gather's 524288 CPU `memcpy`s cost more than the compact path's 2x
   engine transfer; at large widths the compute hides the transfer and the host gather's 1x volume wins.
3. **The crossover (~6500–7000 tokens here) is model- and link-dependent**, like the existing staging
   width gate, which is calibrated (`sched_stage_min_tokens`) rather than fixed.

## Candidate fixes (not yet built)

- **Width-aware threshold.** Plumb the split's token count into `stage_gather` and pick host gather for
  the fine slice only above a calibrated width. Captures the +3–5 % at ub >= 7168; needs calibration and
  is only worth it if the target regime is `-ub 8192`.
- **Remove the tradeoff: device-side / 2-D-pinned compaction.** A 2-D H2D from the pinned host master
  (`cudaMemcpy2DAsync(dst, width, src, stride_src, width, n_copies, H2D)`) or a small device gather kernel
  would move 1x volume **and** keep the gather off the CPU, potentially winning at every width. This is
  the principled fix (§24.3 item 2), and the campaign already measured a fine-slice 2-D D2D compaction at
  0.47 ms; the open question is the 2-D **host->device** transfer from pinned (§22 found the *pageable*
  2-D copy both slow and the source of a fault; the r15 pinned source may change that).
- **Do nothing.** The win is +5 % at one width; the risk of a misfiring threshold is a 30 % regression.
