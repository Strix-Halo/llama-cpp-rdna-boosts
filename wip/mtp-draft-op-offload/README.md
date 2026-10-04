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
-lm none -lzm off`, `--spec-draft-n-max 3 -otd exps=CPU`, seed 42 / temp 0.

**The draft context's compute buffer is 436.65 MiB in both modes**, not 2,550 MiB.  The r7 verbose
load reserve:

```
sched_reserve:      ROCm0 compute buffer size =   436.65 MiB      (draft, offload on)
sched_reserve:  ROCm_Host compute buffer size =   100.63 MiB
```

**Peak VRAM (sum of the three dGPUs) and throughput are identical, and the greedy output is
byte-identical (`5120581a3300` in every cell):**

| config | `LLAMA_MTP_DRAFT_OP_OFFLOAD` | peak VRAM | prompt t/s | gen t/s |
|---|---|---:|---:|---:|
| default gate, 2k prompt | unset | 9322 MiB | 407.1 | 28.8 |
| default gate, 2k prompt | `=0` | 9322 MiB | 406.6 | 29.1 |
| `GGML_SCHED_STAGE_TABLE_REF_MB=0` (force the draft to stage) | unset | 12974 MiB | 352.7 | 19.8 |
| `GGML_SCHED_STAGE_TABLE_REF_MB=0` | `=0` | 13147 MiB | 352.5 | 20.4 |

The forced-staging row is the strongest test: it removes the table-size gate so the draft's 850 MiB
tables stage at `-ub 2048`, and disabling op offload still does not lower VRAM (the 173 MiB delta is
noise, in the wrong direction).

## Why the r6 result does not carry over

The headline 2,550 MiB does not reproduce on r7: the draft context's **compute buffer** is 436.65 MiB
with the switch either way, and 2,550 MiB is exactly the size of the draft context's
`ROCm_Host model buffer` (the host expert weights) in the same verbose load, so the r6 figure may
have been a different line of the load report.  More importantly, the compute-buffer reserve code is
identical between r6 and r7 (the issue-#93 amendment only touches the runtime staging gate/budget
and the `--fit` accounting), so the reservation the switch is meant to remove was not present on r7
to begin with.  Whatever the r6 baseline measured, on the current delivery there is nothing left for
the switch to save.

## Recommendation

Do not promote #98 as-is.  It is a correct, low-risk opt-in knob, so it is harmless to keep in the
toolbox if @briansp2020 still sees a benefit for the 256K + `MOE_EXPERT_CACHE_MIB=4096` case on a
rebuild at r7 or later, but the delivery should not ship it: it adds a context-behaviour switch with
no measured benefit on the current tree, and the repo's default-on policy means it would have to be
justified as either a default-on win (it is not) or a documented correctness workaround (it is not).

Worth asking the author to re-run their 256K table on `v16-a55e952b8-r7` (or later) before we close
the PR, so the answer is against the current delivery rather than the r6 tree.
