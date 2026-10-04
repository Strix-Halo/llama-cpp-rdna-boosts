# issue #97 - the H2D staging calibration allocates a 512 MiB probe for a split with no host weight

**Status:** **resolved** — the fix is **promoted into the delivery as `v16-a55e952b8-r8`** (block 06)
and the reporter confirmed it on their Windows box 2026-10-04 (`v16-84e76d8a2-r36` + the backport
below); issue #97 is closed (`fixed` / `merged`).  This directory also carries the backport for
reporters still on the `84e76d8a2` base (r36/r37), because that tree does not have the r7 per-split
helper (`sched_stage_min_tokens_for`).

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

## Reporter confirmation (Windows 11, 1x R9700 gfx1201, ROCm 10.0.0, 2026-10-04)

Source: [issue #97 comment](https://github.com/stew675/llama-cpp-rdna-boosts/issues/97#issuecomment-5977763673)
(@DanoPTT).  Config: Swift-Qwen3.8-27B Q5_K_M (dense), `-ngl 999`, ctx 172032, `-np 2
--kv-unified`, `--spec-type draft-mtp --spec-draft-n-max 3`, f16 KV, `-lv 4`; memory from the
Windows `GPU Process Memory` counters.  The reporter applied
`fix-issue-97-stage-calibration-r36-r37.patch` to a clean r36 tree and rebuilt.

| build | env | dedicated after start | shared after start | calibration lines | ~100k prompt pp | decode tg | shared after request |
|---|---|---|---|---|---|---|---|
| r36 | – | 31936 MiB | 1150 MiB | 2 | 815.7 t/s | 16.37 t/s | 1138 MiB |
| r36 | `GGML_SCHED_STAGE=0` | 32082 MiB | 476 MiB | 0 | 815.0 t/s | 43.29 t/s | 464 MiB |
| r36 + fix | – | 32082 MiB | 476 MiB | 0 | 799.6 t/s | **43.21 t/s** | 464 MiB |

* With the fix and no env var the dense run never logs `H2D bandwidth calibration` /
  `H2D staging calibration`, even at `-lv 4`, and the footprint matches the `GGML_SCHED_STAGE=0`
  workaround to the MiB.  Decode is back (16.4 -> 43.2 t/s; draft acceptance 87/119 in all runs).
* Greedy (temp 0, top_k 1, `n_predict 128`, same seed) on the ~100k prompt is **byte-identical
  across all three builds**.
* A host-weight config still calibrates: same model `-ngl 40` (partial offload), ~2.8k prompt — with
  the fix both lines are still logged (12.5 GB/s -> min_tokens 1798), same as without it
  (1800).
* The reporter's history-dependence check (one request alone after a long unrelated request on the
  other slot, `-np 2 --kv-unified`, 6 cases) is clean on both builds, every output byte-identical
  between r36 and r36 + fix.
* The single pp reading with the fix is ~2 % lower, which the reporter read as run-to-run noise (one
  run each); the fix path never runs the probe, so there is no plausible prefill cost.

The reporter has dropped `GGML_SCHED_STAGE=0` from their launcher.  Closed; the backport below stays
available for anyone else on the r36/r37 base.
