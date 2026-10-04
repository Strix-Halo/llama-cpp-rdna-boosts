# mtp-draft-op-offload (PR #98): a real ~610 MiB draft-context saving when the expert cache is armed

Status: **RE-MEASURED (2026-10-04): the PR is correct and does save memory when the MTP draft
context also serves device expert copies; promotion decision pending.**  PR
[#98](https://github.com/stew675/llama-cpp-rdna-boosts/pull/98) by @briansp2020 adds
`LLAMA_MTP_DRAFT_OP_OFFLOAD=0`, which builds the MTP draft context with `op_offload = false` so the
drafter's host-resident ops (its experts under `-otd exps=CPU`) run on the host instead of being
offloaded.  The change is 9 lines in `common/speculative.cpp`
([`pr98-mtp-draft-op-offload.patch`](pr98-mtp-draft-op-offload.patch)).  It is correct and the switch
works.  Our first review reported "no measurable gain"; that was a **measurement error**: we did not
arm `MOE_EXPERT_CACHE_MIB`, and the saving only appears when the draft context reserves full-size
device expert copies (which the expert cache causes).  Re-measured with `MOE_EXPERT_CACHE_MIB=4096`,
the switch frees **610 MiB** on our box and the reporter measured **~1.1 GB** on theirs.

## What the PR claims (r6)

With `--spec-type draft-mtp -otd exps=CPU`, the qwen4exp Q8_0 MTP head's experts
(`blk.48.ffn_{gate,up,down}_exps`, 850 MiB each) live in host memory, and op offload stages
full-size device copies of them into the draft context's compute buffer (2,550 MiB = 3 x 850).
`LLAMA_MTP_DRAFT_OP_OFFLOAD=0` makes those ops run on the host instead, freeing the reservation;
the author measured 16.2 -> 15.1 GB right after load with no throughput cost.

## What we measured on r7 (`77ee997c`, the issue-#93 delivery)

The switch is confirmed active (`op_offload = 0` in the draft context, verified with a temporary
log).  Model: `Qwen3.8-Flash-Next-UD-IQ3_XXS` + shared Q8_0 MTP head, `-ncmoe 48 -sm layer -fa 1
-lm none -lzm off`, `--spec-draft-n-max 3`, seed 42 / temp 0.

**The draft context's compute buffer does not hold the expert copies on r7, with the switch either
way.**  Verbose reserve:

| context | `-c 1024 -ub 256` | `-c 32768` (author's flags) | `-c 131072` |
|---|---:|---:|---:|
| target (`-ncmoe 48`) | 764.79 MiB | 2984.30 MiB | 2984 MiB |
| draft (offload on, unset) | **53.01 MiB** | **548.06 MiB** | **436.65 MiB** |
| draft (offload off, `=0`) | **53.01 MiB** | **548.06 MiB** | **436.65 MiB** |

(author's flags = `LLAMA_MTP_SPARSE=0 GGML_SCHED_STAGE_SLOTS=16 GGML_SCHED_STAGE_MAX_MB=8192 -ctk q8_0
-ctv q8_0 --spec-draft-p-min 0.5).

The draft's `blk.48.ffn_{gate,up,down}_exps` (850 MiB each) are loaded on `ROCm_Host` and the draft
`op_offload` is 1 by default, yet the draft reserve never grows to include them, so there is nothing
for the switch to remove.  The scheduler debug confirms the draft **prefill** `MUL_MAT_ID`
(2048-token catch-up) is placed on ROCm0 with a `ROCm0#blk.48...` weight copy, while the decode
`MUL_MAT_ID`s stay on CPU (`ne[2] < 32`); the prefill copy does not appear in the draft reserve on
r7.

**Post-load VRAM at the author's 256K config is identical:** `-c 262144 -b 2048 -ub 2048`, MTP n3,
`-otd exps=CPU`, `--fit off`, after load and before any request: **18065 MiB** with the switch on
and **18065 MiB** with it off.

**Peak VRAM and throughput during a run are identical, and the greedy output is byte-identical
(`5120581a3300` in every cell):**

| config | `LLAMA_MTP_DRAFT_OP_OFFLOAD` | peak VRAM | prompt t/s | gen t/s |
|---|---|---:|---:|---:|
| default gate, 2k prompt | unset | 9322 MiB | 407.1 | 28.8 |
| default gate, 2k prompt | `=0` | 9322 MiB | 406.6 | 29.1 |
| `GGML_SCHED_STAGE_TABLE_REF_MB=0` (force the draft to stage) | unset | 12974 MiB | 352.7 | 19.8 |
| `GGML_SCHED_STAGE_TABLE_REF_MB=0` | `=0` | 13147 MiB | 352.5 | 20.4 |
| `GGML_SCHED_STAGE_MIN_TOKENS=0` (stage every batch) | unset | 12930 MiB | 267.0 | 20.5 |
| `GGML_SCHED_STAGE_MIN_TOKENS=0` | `=0` | 12930 MiB | 267.3 | 19.4 |

The forced-staging row is the strongest test: it removes the table-size gate so the draft's 850 MiB
tables stage at `-ub 2048`, and disabling op offload still does not lower VRAM (the 173 MiB delta is
noise, in the wrong direction).

## Re-measured with the expert cache armed (2026-10-04)

The condition missed in the first review is `MOE_EXPERT_CACHE_MIB`.  With it set, the draft context
reserves full-size device copies of its host expert tables, and that is what the switch removes.
Draft context at `-c 262144`, author's flags, `MOE_EXPERT_CACHE_MIB=4096`, same UD-IQ3_XXS model and
shared Q8_0 head (r7 + #98 binary):

| switch | draft ROCm2 compute | draft ROCm_Host compute | post-load VRAM |
|---|---:|---:|---:|
| unset (op offload on) | 2054.25 MiB | 1124.07 MiB | 20027 MiB |
| `LLAMA_MTP_DRAFT_OP_OFFLOAD=0` | 1444.06 MiB | 1124.07 MiB | 19417 MiB |

Delta: **610.19 MiB** of draft device compute buffer, and 610 MiB of post-load VRAM.  The `=0`
floor (1444.06 MiB) matches the reporter's floor (1444.33 MiB) almost exactly, which shows the floor
is model-independent and the unset value scales with the head's expert sizes (their 2550 MiB vs our
2054 MiB).  The 610 MiB vs their ~1.1 GB is therefore the same effect on different head files.
Without the cache armed the draft reserve is 548 MiB either way (the earlier result) and there is
nothing for the switch to remove.

## Recommendation

The PR is correct and the switch does what it says once the expert cache is armed: 610 MiB here,
~1.1 GB on the reporter's box (their head's expert tables are larger).  It is an opt-in switch
(`=0` disables op offload; unset keeps the current behaviour), so promoting it changes no default.
Two things remain before promoting: whether the memory win is worth shipping a knob whose benefit
only appears under `MOE_EXPERT_CACHE_MIB`, and whether the reporter's single-run prefill hint
(1246 t/s unset vs 1863 t/s `=0` on a 37k prompt with the cache armed) reproduces, since if it does
this is a prefill win too and not only a memory win.  The reporter will correct the PR title and
README to the 1.1 GB figure and the cache condition.
