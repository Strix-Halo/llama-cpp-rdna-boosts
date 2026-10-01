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
| `-b` / `-ub` | batch / ubatch | `-ub 4096` is the sweet spot here; `-ub 8192` is usually too big (VRAM/OOM) |
| `-sm layer` vs `-sm tensor` | split | 1 GPU: `layer`.  Multi-GPU: `tensor` beats `layer` at every offload level (the r16 record) |
| gather vs staging | how the used experts are uploaded | **automatic**: a large expert table (≥ 224 MiB/op) always gathers, else the width gate decides |

## The headline config — Qwen3.6-35B-A3B Q8_0 (37.8 GB) on one R9700 (32 GiB)

This is the representative case: the weights **do not fit** the 32 GiB card, so `-ncmoe` is mandatory and
the expert cache is what makes decode usable.  `-ngl 99 -sm layer -fa 1 -t 8 -p 8192 -n 1024 -r 2`,
`MOE_EXPERT_CACHE_DEVMAP=1`, **`-ub 4096`** (the old `-ub 2048` understates prefill badly):

| `-ncmoe` | `MIB` | pp8192 (t/s) | tg1024 (t/s) | note |
|---:|---:|---:|---:|---|
| 8 | 4096 | - | - | **OOM at `-ub 4096`** (not enough VRAM left) |
| 12 | 8192 | **3122** | 79.9 | max prefill while keeping ~80 decode |
| **16** | **8192** | **2706** | **81.1** | **recommended balanced default** |
| 20 | 12288 | 2404 | **81.8** | best decode |
| 24 | 12288 | 2139 | 78.6 | |
| 32 | 12288 | 1786 | 70.5 | |
| 40 | 12288 | 1554 | 61.8 | |

`-ub 4096` lifts prefill a lot versus `-ub 2048` at the same `-ncmoe`/`MIB` (e.g. `-ncmoe 16 MIB=8192`:
2706 vs 1751 t/s) with the same decode, so it is the setting to tune around.  The gather is automatic and
neutral for Q8_0 (its 272 MiB tables are above the 224 MiB threshold): ub 512/1024/2048/4096/8192 =
607/1042/1755/2707/3584 t/s, within noise of staging at every width.

**Recommendation:** `-ncmoe 16 MOE_EXPERT_CACHE_MIB=8192 -sm layer -fa 1 -b 4096 -ub 4096 -t 8` gives
**2706 prefill / 81.1 decode** in one run.  Go to `-ncmoe 12 MIB=8192` if prefill matters more (3122 /
79.9), or `-ncmoe 20 MIB=12288` for the last drop of decode (2404 / 81.8).  There is no setting that
maximizes both: the decode plateaus around `-ncmoe 16-20` and falls past it.

### ub 2048 reference (for comparison with older records)

| `-ncmoe` | `MIB` | pp8192 | tg1024 |
|---:|---:|---:|---:|
| 8 | 4096 | 2635 | 72.1 |
| 16 | 8192 | 1751 | 81.0 |
| 16 | 16384 | 1753 | 81.8 |
| 24 | 12288 | 1326 | 78.1 |

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
       -p 8192 -n 1024 -b 4096 -ub 4096 -r 2

# qwen4exp (PLE fully in RAM, gather automatic):
IQ4=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=12288 MOE_EXPERT_CACHE_DEVMAP=1 \
  $BIN -m "$IQ4" -ngl 99 -ncmoe 48 -sm layer -fa 1 --lazy-mode off --load-mode none \
       -t 8 -p 8192 -n 1024 -b 2048 -ub 2048 -r 2
```

## Gotchas

* `-ncmoe 8` on the Q8_0 35B fits at `-ub 2048` but **OOMs at `-ub 4096`** - fewer offloaded layers
  means more expert bytes on the card.  Use `-ncmoe >= 12` with `-ub 4096`, or `-ub 2048`.
* `-ncmoe` and `MOE_EXPERT_CACHE_MIB` are independent: `-ncmoe` is a static placement, the arena is a
  persistent decode cache over the same host pool.  Set `MIB` large enough and the arena *is* `-ncmoe 0`
  for decode while prefill still streams from host.
* `--fit` knows nothing about the arena (it is allocated from free VRAM after `--fit`), so `-c`/`--fit`
  and a large `MIB` can both claim the same headroom.
