# wip/rdna4-verify-fusions-lf — three bit-identical speculative-verify fusions

One `git am` patch made on `v16-a55e952b8-r8` (applied tree `af02d2d4…`). It also applies to r9 (applied tree `6cf4f532…`, `git apply --check`, one hunk offset by 13 lines); it has not been rebuilt on r9.

| fusion | what it does | off switch |
|---|---|---|
| GLU -> Q8_1 | the verify-step SwiGLU writes the Q8_1 activation for the next mmvq directly (r8 does this only for prefill) | `GGML_LF_GLU_Q8=0` |
| VCONV | the fused GDN concat + conv also for 2..255-token batches | `GGML_LF_VCONV=0` |
| batched copy | consecutive same-layout f32 CPY nodes (the GDN conv-state snapshots) in one launch | `GGML_LF_CPY_BATCH=0` |

All three are bit-identical and default-on. Measurements and gates are in the PR description.

*The patch and this note were written by Claude (AI).*
