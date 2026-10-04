# issue #97 - the H2D staging calibration allocates a 512 MiB probe for a split with no host weight

**Status:** the fix is **promoted into the delivery as `v16-a55e952b8-r8`** (block 06).  This
directory only carries the backport for reporters still on the `84e76d8a2` base (r36/r37), because
that tree does not have the r7 per-split helper (`sched_stage_min_tokens_for`).

## Root cause

`ggml_backend_cuda_stage_h2d_gbps()` (block 06, `ggml/src/ggml-cuda/ggml-cuda.cu`) runs a one-off
H2D bandwidth calibration: a 512 MiB `cudaMalloc`, a warm-up copy plus three timed copies, then
`cudaFree`.  The scheduler reaches it through `sched_stage_min_tokens()`, the adaptive staging width
gate.  `sched_stage_issue()` called that gate **before** its host-weight loop, so every split with
any input - including the splits of a dense, fully offloaded model - triggered the calibration even
though nothing would ever be staged.

On the reporter's Windows 11 / ROCm 10.0 (TheRock) box the `cudaFree` does not return the 512 MiB to
the per-process GPU memory counters, so the allocation is stranded for the rest of the session.  On
a config already at the 32 GB boundary the spill goes to shared (system) memory and decode
collapses: a 27B Q5 at 161K ctx measured 46.8 t/s with the calibration avoided vs 16.4 t/s with it
(same greedy acceptance).  `GGML_SCHED_STAGE_MIN_TOKENS=<any>` and `GGML_SCHED_STAGE=0` both skip
the calibration, which is how the reporter isolated it.

## Fix (r8, `a55e952b8` base)

`ggml-backend.cpp`: `sched_stage_min_tokens_for()` computes `sched_stage_host_weight_bytes(split)`
first and returns 0 when it is zero, so `sched_stage_min_tokens()` (and the calibration) is never
reached for a split that has nothing to stage.  Splits with a host weight calibrate exactly as
before.  Net diff (old tip `27b6254e7` -> new tip `05bbd56e0`):

```diff
 static int64_t sched_stage_min_tokens_for(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split) {
-    return sched_stage_min_tokens_for_bytes(sched, sched_stage_host_weight_bytes(split));
+    const size_t bytes = sched_stage_host_weight_bytes(split);
+    if (bytes == 0) {
+        return 0;
+    }
+    return sched_stage_min_tokens_for_bytes(sched, bytes);
 }
```

## Backport for the `84e76d8a2` base (r36/r37)

`fix-issue-97-stage-calibration-r36-r37.patch` is the same guard, inline in `sched_stage_issue()`
(that tree has no `sched_stage_min_tokens_for` helper).  Apply to a clean r36/r37 checkout:

```bash
git apply fix-issue-97-stage-calibration-r36-r37.patch
# or: patch -p1 < fix-issue-97-stage-calibration-r36-r37.patch
```

Then rebuild.  The check is identical: a dense full-offload run must no longer log
`H2D bandwidth calibration` / `H2D staging calibration` even at `-lv 4`, while a `-ncmoe` run must
still log them.  `GGML_SCHED_STAGE_MIN_TOKENS` continues to work as the explicit override.

## Validation (r8, gfx1201, 1x R9700)

* Dense `Qwen3.5-4B-Q8_0`, `-ngl 99`: no calibration log at `-lv 4` (r7 logs both lines).
* `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL`, `-sm layer -ncmoe 99`: both lines still logged
  (14.5 GB/s -> min_tokens=1542), now at the first host-weight split.
* Same-seed greedy (`--seed 42 --temp 0`) byte-identical to r7: dense 4B `c10fd88999c5`,
  `-ncmoe` MoE `d4313ad642b8`; dense 3-GPU `-sm tensor` 4B gate `1c5d32ac537d` unchanged.
* `scripts/validate-set.sh` green (strict 16/16 `git am`, applied tree == `release.json.tree`).
