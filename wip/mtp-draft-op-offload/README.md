# mtp-draft-op-offload (PR #98): reviewed, not a measurable gain on r7

Status: **REVIEWED, NOT PROMOTED (2026-10-05)**.  PR
[#98](https://github.com/stew675/llama-cpp-rdna-boosts/pull/98) by @briansp2020 adds
`LLAMA_MTP_DRAFT_OP_OFFLOAD=0`, which builds the MTP draft context with `op_offload = false` so the
drafter's host-resident ops (its experts under `-otd exps=CPU`) run on the host instead of being
offloaded.  The change is 9 lines in `common/speculative.cpp`
([`pr98-mtp-draft-op-offload.patch`](pr98-mtp-draft-op-offload.patch)).  It is correct and the switch
works, but on the r7 delivery it shows **no measurable VRAM, prefill or decode gain**, so the
recommendation is **do not promote**.  The author's reported gain was measured on the r6 tree
(`2b57533c`), before the issue-#93 staging fix.

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

## Why the r6 result does not reproduce on r7

The r7 draft context does not reserve the expert copies at all, so there is nothing for the switch to
free here.  The PR's 2,550 MiB figure is the size of the draft's host expert tables (3 x 850, the
draft's `ROCm_Host model buffer`), which matches the PR's own measured VRAM delta being about 1.1 GB
rather than 2.55 GB.  On r7 the draft reserve is 548 MiB (author's flags) and 436 MiB (128k) with the
switch either way, and the draft's prefill weight copy does not land in that reserve.  We could not
find a config, including forced staging at every batch, where disabling the draft's op offload
changed device memory.  The compute-buffer reserve code is unchanged r6 -> r7 (the issue-#93 amendment
touches only `memory_breakdown` accounting in `llama-context.cpp` and the runtime staging gate), so
the reservation the switch targets does not appear to exist on either tree on this hardware.

## Recommendation

Do not promote #98 as-is.  It is a correct, low-risk opt-in knob, so it is harmless to keep in the
toolbox if @briansp2020 still sees a benefit for the 256K + `MOE_EXPERT_CACHE_MIB=4096` case on a
rebuild at r7 or later, but the delivery should not ship it: it adds a context-behaviour switch with
no measured benefit on the current tree, and the repo's default-on policy means it would have to be
justified as either a default-on win (it is not) or a documented correctness workaround (it is not).

Worth asking the author to re-run their 256K table on `v16-a55e952b8-r7` (or later) before we close
the PR, so the answer is against the current delivery rather than the r6 tree.
