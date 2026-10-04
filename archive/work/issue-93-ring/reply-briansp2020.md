Thanks for the detailed report and the rocprof trace. It pointed straight at the problem.

We reproduced it here. The root cause is two things that were stacked on top of each other:

1. `GGML_SCHED_STAGE_MAX_MB` was a fixed 2048 MiB, unrelated to the table size. qwen4exp's expert
   tables are 450 MiB, so a full 8 slot ring needs about 3.6 GiB. The fifth slot growth tripped the
   budget, `h2d_stage_buffer()` returned NULL, and the scheduler set `stage_enabled = false` for the
   rest of the run. That is exactly the 30 percent H2D and 0 percent overlap you saw.

2. The staging width gate was calibrated against a 144 MiB reference table, so a 450 MiB table was
   staged at widths where the whole table copy actually loses to the pruned serial path. On our x4
   box `-ub 2048` was 653 t/s serial versus 525 t/s staged, so fixing only (1) would have made it
   slower.

The fix is in the candidate release `v16-a55e952b8-r7`:

- The ring budget is auto sized from the largest slot when `GGML_SCHED_STAGE_MAX_MB` is unset, so a
  full ring always fits the table it feeds. An explicit value still wins.
- The width gate is scaled by `host_table_bytes / 144 MiB`, so a large table is only staged at widths
  where the full table copy really wins. `GGML_SCHED_STAGE_TABLE_REF_MB=0` disables the scaling.
- A shortfall now skips that one split instead of disabling the ring for the whole run.
- The raw ring is counted in `llama_get_memory_breakdown` and `--fit`.

The device gather is untouched and stays default off. The `////` corruption was the gather, not the
ring, so this does not go near it.

On our gfx1201 PCIe5 x4 box with qwen4exp UD-IQ3_XXS, `-ncmoe 48 -sm layer -fa 1 -lm none -lzm off`
(the `-lzm off` matters, the managed lazy reader otherwise makes the run to run noise larger than the
effect):

| `-ub` | serial | staged (auto) |
|------:|-------:|--------------:|
| 2048  | 785    | 789 (gated)   |
| 4096  | 1142   | 1071 (gated)  |
| 8192  | 1570   | 2374 (+51 %)  |

Output is byte identical between staged and serial, 0 `////`, and the usual gates (`MUL_MAT_ID`,
dense 4B coherence) are green.

If you have a moment, could you try the candidate on your x16 box? The case we most want confirmed is
your original `-ub 2048`, since that is where the budget fix should recover most of your +42 percent,
and it would be good to see the 4/8/16 slot sweep again on the real x16 link. The patch is in the
repo at `archive/work/issue-93-ring/issue-93-ring.patch` (it is also folded into the r7 patch set),
and a quick A/B is `GGML_SCHED_STAGE=0` versus leaving it unset. Please run it with `-lzm off` so the
numbers are directly comparable.

Thanks again for the trace and the numbers. They made this one straightforward to pin down.
