# `wip/` — host-memory footprint of GPU-resident weights (`fingon`, gfx1100)

**Status: OPEN — new investigation, seeded 2026-10-02.**  This is a `wip/` tree; nothing here is part of
the delivery (`patches/`).  It applies only to `~/llama-fingon` (campaign) / `~/llama.cpp` (delivery) on
`fingon`, which is never pushed.

## The phenomenon

On `fingon` (RX 7900 XTX, **24 GiB** VRAM, **~30.5 GiB** RAM, 131 GiB swap, gfx1100, ROCm 7.14),
`kswapd0` runs hot and the box is memory-starved during `-ncmoe` prefill even though the model is only
22.7 GB.  The suspicion: **weights that are uploaded to and resident on the GPU still keep host memory**,
and/or the host-resident expert pool is unreclaimable, so the kernel cannot make progress.

## Seed measurement (2026-10-02)

Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.66 GB, 256 experts, top-8, ~40 MoE layers).
`HIP_VISIBLE_DEVICES=0`, `-ngl 99 -fa 1 -t 8 -p 256 -ub 256`, monitor `/proc/<pid>/status` +
`/proc/meminfo` every 2 s (`memwatch.sh`, peak shown):

| config | peak `VmRSS` | `RssShmem` | `RssFile` | `RssAnon` | `VmLck` | `MemFree` |
|---|---:|---:|---:|---:|---:|---:|
| `-ncmoe 0` (all GPU), @2 s | 16.8 GB | ~0.02 GB | 16.4 GB | 0.2 GB | 0 | — |
| `-ncmoe 40` (all experts host) | **29.5 GB** | **18.68 GB** | ~10.7 GB | ~0.1 GB | 0 | **~0.25 GB** |

Observations:

* The offloaded expert pool shows as **`RssShmem` ≈ 18.7 GB** — a *shared-memory* (GPU-visible /
  `hipHostMalloc`-style) allocation, **not** the mmap.  That is the `-ncmoe` host-expert buffer.
* `RssFile` (~10.7 GB) is the model **mmap page cache**; the all-GPU case is dominated by it (16.4 GB).
* `Mlocked: 0` and `VmLck: 0` everywhere — nothing is OS-`mlock`ed, yet `MemFree` falls to ~0.25 GB and
  kswapd churns.  So either the shmem pages are being re-dirtied every pass, or the reclaim path is the
  problem, not pinning.
* The reported problem is **not** model size alone (22.7 GB fits in 30.5 GB): the resident set is ~30 GB
  because host pool (18.7) + touched mmap (10.7) ≈ 29.4 GB, leaving no headroom.

> Note: an earlier seed run accidentally omitted `HIP_VISIBLE_DEVICES=0` and ran `-sm layer` across the
> gfx1100 **and** the gfx1036 iGPU — re-run every measurement pinned to GPU 0.

## Hypotheses to test (in order)

1. **GPU-uploaded weights keep their mmap pages resident.**  After the loader copies a weight to VRAM,
   the backing mmap pages stay in page cache (`RssFile`) and are only reclaimed under pressure.  A
   `madvise(MADV_DONTNEED)` / `posix_fadvise(POSIX_FADV_DONTNEED)` on GPU-resident ranges after upload
   could free ~10 GB.
2. **The `-ncmoe` host-expert pool is a GPU-visible shmem buffer that cannot be reclaimed cheaply.**
   Identify its backing (`/proc/<pid>/maps` + `/proc/<pid>/fd`; `hipHostMalloc`, memfd, tmpfs?) and
   whether it is dirtied/re-read each pass (kswapd) or pinned implicitly.
3. **Host/GPU duplication of specific weights.**  Are `token_embd` / `output` / router weights also held
   host-side when they are GPU-resident (as `RssAnon` or a second buffer)?  The current data says
   `RssAnon` is tiny (~0.1–0.2 GB), so this looks *unlikely*, but confirm per-tensor.
4. **`--no-mmap` / `--mlock` push everything to `RssAnon`** (unreclaimable) — verify the runme variants.
5. **`LLAMA_MMAP_HOST_EXPERTS` pinning cost** (delivery r15): `=0` uses the pageable mmap instead of the
   pinned shmem host buffer; measure both the memory footprint and the prefill cost (r15 measured ~2.7k
   vs ~5.1k t/s pinned at pp8192).

## Method / repro

```sh
cd ~/llama-fingon     # or ~/llama.cpp for the delivery build
export LD_LIBRARY_PATH=/opt/rocm-7.14-gfx1100/lib HIP_VISIBLE_DEVICES=0
MQ4=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

# monitored load: samples /proc/<pid>/status + /proc/meminfo, dumps the largest smaps region at peak
./memwatch.sh 60 ./build-rocm/bin/llama-bench -m "$MQ4" -ngl 99 -ncmoe 40 -fa 1 -t 8 \
    -p 256 -n 0 -b 256 -ub 256 -r 1 -o jsonl
```

Compare, all pinned to GPU 0:

| axis | values |
|---|---|
| `-ncmoe` | 0 (all GPU, small ctx), 12 (largest that still fits at ub 8192), 40 (all host) |
| mmap | default (mmap) vs `--no-mmap` |
| locking | default vs `--mlock` |
| host experts | `LLAMA_MMAP_HOST_EXPERTS` unset (pinned) vs `=0` (pageable) |
| cache | `MOE_EXPERT_CACHE_MIB` unset vs armed (does the arena change the host pool?) |

Record per configuration: peak `VmRSS` / `RssAnon` / `RssFile` / `RssShmem` / `VmSwap` / `VmLck`,
`MemFree`/`Cached` at peak, kswapd CPU (`pidstat -p $(pgrep kswapd0) 1`), and the largest
`/proc/<pid>/maps` regions with their backing file.

## Likely mitigations to evaluate (after the root cause)

* `madvise(MADV_DONTNEED)` (or `posix_fadvise(POSIX_FADV_DONTNEED)`) the mmap ranges of weights that are
  now GPU-resident, once loading is complete — the delivery could do this in the model loader / at the
  end of `llama_model_load`.
* Avoid a *pinned* host pool for the whole `-ncmoe` expert set; keep a small pinned **staging ring** and
  read the master from the pageable mmap (`LLAMA_MMAP_HOST_EXPERTS=0` is the existing A/B).
* Ensure the host-expert pool is backed by reclaimable memory (or advise the kernel it is cold between
  passes), if hypothesis 2 holds.
* A `--cache-ram` / `--no-mmap` accounting fix if `RssAnon` is unexpectedly large.

## Tooling

* `memwatch.sh <seconds> <cmd...>` — samples the child's `/proc/<pid>/status` + `/proc/meminfo` every
  2 s, and copies `/proc/<pid>/smaps` at peak RSS so the largest regions can be inspected.

## State of the two builds on `fingon`

| path | what |
|---|---|
| `~/llama.cpp` | delivery r28 (16 patches), tree `dc2decae…` |
| `~/llama-fingon` | r28 + campaign `exp23`, tree `1922182…` |
| `~/bin/build-llama-rocm-714` | builds `./build-rocm` for gfx1100 against `/opt/rocm-7.14-gfx1100` |
| `~/llama-cpp-rdna-boosts` | this delivery repo (docs/patches) |
