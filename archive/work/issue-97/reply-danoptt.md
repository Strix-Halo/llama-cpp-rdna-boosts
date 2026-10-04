# Reply draft for issue #97 (@DanoPTT)

Posted 2026-10-05.  Kept as the record; the GitHub comment is the live copy.  The reporter
confirmed the fix on Windows 2026-10-04 and the issue was closed — the confirmation data is in
`README.md` ("Reporter confirmation").  The comment's backport link was updated to the archived
path after this directory moved from `wip/issue-97-vram/` to `archive/work/issue-97/`.

---

Thanks for the detailed report, and for narrowing it down to `sched_stage_issue()` yourself. Your
diagnosis is exactly right.

The root cause is the ordering in the scheduler. `sched_stage_issue()` called
`sched_stage_min_tokens()` (the adaptive staging width gate, which runs the one-off H2D calibration)
before its loop over `sched_stage_is_host_weight()` inputs. A dense full-offload graph still has
splits with inputs, so the first one reached the gate, calibrated, and paid for the 512 MiB probe.
Nothing would ever have been staged.

The fix is to only reach the calibration when the split actually has a host-resident weight. On the
current delivery (`a55e952b8` base) that is a small change in `sched_stage_min_tokens_for()`: it now
returns 0 as soon as `sched_stage_host_weight_bytes(split) == 0`, so `sched_stage_min_tokens()` is
never called for a split with nothing to stage. A split that does carry a host weight calibrates
exactly as before, and `GGML_SCHED_STAGE_MIN_TOKENS` / `GGML_SCHED_STAGE=0` keep working.

This is promoted as `v16-a55e952b8-r8` (block 06). Validation here on gfx1201 (1x R9700): a dense
`Qwen3.5-4B-Q8_0` full-offload run at `-lv 4` no longer logs `H2D bandwidth calibration` /
`H2D staging calibration`; a `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL` `-ncmoe 99` prefill still logs both
(14.5 GB/s -> min_tokens 1542, now at the first host-weight split). Same-seed greedy is
byte-identical to r7, and the delivery's dense 3-GPU `-sm tensor` 4B gate is unchanged.
`validate-set.sh` is green.

You are on `v16-84e76d8a2-r36`, which is the previous base. That tree does not have the r7 per-split
helper, so here is the equivalent one-hunk backport:

https://github.com/stew675/llama-cpp-rdna-boosts/blob/main/archive/work/issue-97/fix-issue-97-stage-calibration-r36-r37.patch

It applies with `git apply` or `patch -p1` to a clean r36/r37 checkout and is the same guard,
inline in `sched_stage_issue()`. A rebuild is all that is needed. If you would rather move to the
current base, `v16-a55e952b8-r8` carries the fix in block 06.

When you get a chance, could you confirm on your box: the per-process GPU memory at startup
(dedicated + shared), the ~100k decode, and your greedy purity checks. For the dense config the
calibration line should be gone even at `-lv 4`, and the `-ncmoe` config should still show it.

One note on the probe itself: it is still a plain `cudaMalloc` + `cudaFree`. I did not change that,
because the behaviour you are seeing (freed memory not returned to the process counters) is the
Windows driver's accounting, not something the scheduler can see. With the fix a dense run never
reaches it, which is the case that was costing you VRAM.
