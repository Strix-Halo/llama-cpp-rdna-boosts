# Running an oversized Q8_0 MoE on a single card — a reproducible config guide

**Audience:** someone with one RDNA4 card (this was measured on a single R9700 / gfx1201, ROCm 7.14.1)
and a model + KV cache that does **not** fit in VRAM, so some MoE layers must be offloaded.

**Scope:** `wip/moe-expert-cache` (the decode-side MoE expert cache).  `MOE_EXPERT_CACHE_MIB` arms a
persistent per-device VRAM cache of the hot experts; `-ncmoe N` decides how many MoE layers' experts
live in the host pool.  All numbers below are single-GPU unless stated.

## The method (read this before comparing numbers)

* **One process, both sides.**  Measure prefill and decode in the **same** `llama-bench` invocation
  (`-p 8192 -n 1024`).  A warm cache from the prefill is part of the result; a standalone `-p 0 -n 1024`
  run reports a different decode.
* **Warm the model.**  Repeat (`-r 2+`) and quote the **last** rep.  qwen4exp has a 26.8 GiB
  `per_layer_token_embd.weight` (PLE) that is mmap-lazy by default; it warms with every pass.
* **`--lazy-mode off` for qwen4exp.**  This loads the PLE into host RAM and removes the warm-up: prefill
  becomes **flat** (~2650 t/s at `-p8192 -ub2048` instead of a cold ~1600 climbing to ~1790).  Every
  qwen4exp prefill comparison must use it.  qwen35moe/Q4_K_M have no lazy tensor.
  (`LLAMA_LAZY_BUF_MB` sizes a managed RAM buffer instead.)
* **Pin threads:** `-t 8`.  The GPU IRQs on this host live on the top cores; a default `-t` starves the
  cards.
* **Never run two benches at once.**
* **`-f`/`--single-turn --no-display-prompt`** for `llama-cli`; `-v` only for diagnostics (it corrupts
  the hash extractor).

## The axes

| axis | what it does | notes |
|---|---|---|
| `-ncmoe N` | host-resident MoE layers | higher N = more experts in RAM; prefill falls, decode is capped by residency |
| `MOE_EXPERT_CACHE_MIB=<MiB>` | per-device VRAM cache budget | the decode win; `≥ slots×expert_bytes` = full residency (`h=1`) |
| `-b` / `-ub` | batch / ubatch | `-ub 2048` is the sweet spot here; `-ub 8192` raises prefill but costs VRAM |
| `-sm layer` vs `-sm tensor` | split | 1 GPU: `layer`.  Multi-GPU: `tensor` beats `layer` at every offload level (the r16 record) |
| gather vs staging | how the used experts are uploaded | **automatic**: a large expert table (≥ 224 MiB/op) always gathers, else the width gate decides |

## The config table — Qwen3.6-35B-A3B Q8_0 (37.8 GB) on one R9700 (32 GiB)

`-ngl 99 -ncmoe N -sm layer -fa 1 -b 2048 -ub 2048 -t 8 -p 8192 -n 1024 -r 2`, `MOE_EXPERT_CACHE_DEVMAP=1`:

| `-ncmoe` | `MIB` | pp8192 (t/s) | tg1024 (t/s) | note |
|---:|---:|---:|---:|---|
| 8 | 4096 | **2624** | 72.9 | max prefill; `-ub 4096` OOMs here |
| 12 | 8192 | 2081 | 80.5 | |
| 16 | 8192 | 1740 | **80.0** | best decode |
| 24 | 12288 | 1320 | 78.9 | |
| 40 | 12288 | 901 | 61.5 | |

There is no single setting that maximizes both: **`-ncmoe 8..16` is the "both good at once" band**
(≈2620 t/s prefill at `-ncmoe 8`, ≈80 t/s decode at `-ncmoe 16`).  Past `-ncmoe 16` the decode is capped
by how much of the hot set fits in the arena.

## The qwen4exp point — Qwen3.8-Flash-Next IQ4_NL (100 GB) on one R9700

`-ngl 99 -ncmoe 48 -sm layer -fa 1 --lazy-mode off --load-mode none -b 2048 -ub 2048 -t 8`, `MIB=12288`:

| config | pp8192 | tg1024 |
|---|---:|---:|
| gather (automatic now: 450 MiB table) | **~2650-3060** | ~37 |
| staging (the old default) | 657-1403 | ~39 |

The gather is a 2-4x prefill win and is now selected automatically.  `--lazy-mode off` is required for
comparable numbers.  This model needs `-c`/`--fit` headroom: it is 100 GB on a 32 GiB card.

## Repeat it yourself

```bash
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
BIN=./build-rocm-b3/bin/llama-bench
MQ8=/llm/models/Qwen3.6/35B-A3B/Q8_0/Qwen3.6-35B-A3B-Q8_0.gguf
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=8192 MOE_EXPERT_CACHE_DEVMAP=1 \
  $BIN -m "$MQ8" -ngl 99 -ncmoe 16 -sm layer -fa 1 -t 8 \
       -p 8192 -n 1024 -b 2048 -ub 2048 -r 2

# qwen4exp (PLE fully in RAM, gather automatic):
IQ4=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=12288 MOE_EXPERT_CACHE_DEVMAP=1 \
  $BIN -m "$IQ4" -ngl 99 -ncmoe 48 -sm layer -fa 1 --lazy-mode off --load-mode none \
       -t 8 -p 8192 -n 1024 -b 2048 -ub 2048 -r 2
```

## Gotchas

* `-ncmoe 8` on the Q8_0 35B fits at `-ub 2048`; it OOMs at `-ub 4096`.
* `-ncmoe` and `MOE_EXPERT_CACHE_MIB` are independent: `-ncmoe` is a static placement, the arena is a
  persistent decode cache over the same host pool.  Set `MIB` large enough and the arena *is* `-ncmoe 0`
  for decode while prefill still streams from host.
* `--fit` knows nothing about the arena (it is allocated from free VRAM after `--fit`), so `-c`/`--fit`
  and a large `MIB` can both claim the same headroom.
