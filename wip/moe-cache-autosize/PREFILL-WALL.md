# The wide-`-ub` prefill wall with host-resident experts (~16k, 2 GPU, UD-IQ4_XS)

**Status: measurement only, 2026-10-06.  Not a delivery change; nothing promoted.**  Build
`~/llama-r13` @ `4643be072` = r15 tip + the #41 stage-redirect fix (`stage_d2d`) + inert env-gated
diagnostics (`mmid-geometry-diag.diff`, none of them set here).

**Config:** 2 x R9700 (`HIP_VISIBLE_DEVICES=0,1`), `Qwen3.8-Flash-Next-UD-IQ4_XS` + its shared Q8_0 MTP
head, `-sm tensor -ncmoe 48 -fa on -ctk q8_0 -ctv q8_0 -t 8 -c 32768 -b/-ub` as noted,
`--lazy-mode off --load-mode auto --spec-type draft-mtp --spec-draft-n-max 3 -n 8 --temp 0
--no-warmup`.  Prompt `/tmp/pl_16k.txt` = `prompts/code-python.txt` x30 (~15980 tok; `x15` = ~8k,
`x7` = the 3.7k prompt used in the corruption work).  All runs coherent (`////` = 0).

## Measurements

| config | Prompt t/s |
|---|---:|
| `-ub 8192`, staging **on** (the #41 D2D fix) | **1080 / 1074 / 1054** (three runs) |
| `-ub 8192`, staging **off** (`GGML_SCHED_STAGE=0`) | 853 |
| `-ub 4096`, staging on | 704 |
| `-ub 8192`, staging on, 8k prompt (1 ubatch) | 895 |
| `-ub 8192`, staging on, 3.7k prompt (1 ubatch) | 460 |

`-ncmoe` sweep to isolate upload from compute **could not run** on 2 x 32 GB: `-ncmoe 49` == `-ncmoe 48`
== 1080 (layer 48 is the dense output head, not MoE); `-ncmoe 46` / `42` abort with
`ggml_backend_cuda_buffer_type_alloc_buffer: allocating 11427.44 MiB on device 0: cudaMalloc failed:
out of memory`.  So the separation here is the token scaling and the staging A/B instead.

## Reading

Fit `t = U + C*T` from the two single-ubatch points (3.7k, 8k):

* `C = (8192/895 - 3731/460) / (8192 - 3731) = 2.33e-4 s/tok` -> a compute-only ceiling of **~4300 t/s**;
* `U = 3731/460 - C*3731 = 7.24 s` **fixed per ubatch**.

The per-ubatch host-expert upload is ~55 GiB (the whole host table per MoE split), so `U` implies an
**effective H2D of ~7.6 GB/s -- ~52 % of the 14.5 GB/s calibrated link**.  The 16k point (2 ubatches,
`2U + C*16k = 18.3 s` predicted vs `15.25 s` measured) shows the second ubatch's upload partly
pipelines behind the first's compute, so `U` shrinks a little run-to-run, but the shape is clear:

* **the wall at `-ub 8192` is the per-ubatch host-expert upload, not compute** (compute is ~4x headroom);
* the upload runs at only about half the calibrated link rate, and is only partly overlapped;
* `-ub 4096` is *slower* (704 vs 1080) because the same fixed per-ubatch upload is amortized over half
  the tokens -- the wider ubatch is purely an upload-amortization win.

## Next

1. Instrument the ring: per-slot staged bytes vs the wall clock (or a rocprof/rocTracer timeline) to see
   where the ~2x effective-rate loss is -- the H2D itself, the `stage_d2d` copy, the child-graph wait, or
   the slot free/reuse dependency.  The #41 fix replaced the (corrupt) pointer redirect with a real D2D
   copy; confirm the D2D is overlapped with the next H2D and is not serializing the ring.
2. Per-op timing at `-ub 8192` (the repo's scheduler/`GGML_SCHED_DEBUG` paths or rocprof) to confirm the
   compute share really is minor, i.e. that `C` above is not hiding a serialized reduction.
3. TODO #42: the `-ub 8192` compute buffer permanently reserves ~11339 MiB, which shrinks the decode
   arena (42130 -> 26707 MiB under `-sm layer`).  A prefill-only ubatch sizing (or a separate max-prefill
   allocation freed before decode) would let a big prefill ubatch and a big arena coexist -- the likely
   way to get both the 1080 t/s prefill and the ~67 t/s decode in one config.
