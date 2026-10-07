> **CLOSED / FOLDED (r26, 2026-10-07).**  Accepted as PR #107 (@overdoingism) and folded into delivery **block 15**.  See `WORKLOG.md` 2026-10-07 (r26) and the block-15 r26 amendment in `patches/README.md`.

The host path is always available, so the fallback needs no extra memory, and F1 is now default-on
for single-sequence DFlash (`GGML_LF_DFLASH_DEV=0` disables).  The campaign text below is the original handover.

# wip/dflash-dev-default-on — F1 falls back instead of asserting, then defaults on (TODO item 30)

One `git am` patch made on `v16-a55e952b8-r10` (canonical tip `b86854900`, net tree `dab5186b…`). It also applies cleanly to `v16-a55e952b8-r12` (net tree `1cd1d27e…`); the two files it touches are unchanged between r10 and r12. It has not been rebuilt on r12.

- **The F1 allocation no longer aborts.** If the target context cannot allocate its device layer-input buffers (`llama_context::extract_layer_inputs`), it logs one warning and stays on the host path for the rest of the context. The host buffers are always reserved, so this needs no extra memory. The draft sees `llama_lf_get_layer_inp_dev()` return null and gathers the features from the host buffers instead, so the output does not change.
- **F1 is now default-on for single-sequence DFlash.** `GGML_LF_DFLASH_DEV=0` turns it off. With more than one sequence, F1 stays on the host path as before; the warning about that now appears only when `GGML_LF_DFLASH_DEV` is set explicitly.

*The patch and this note were written by Claude (AI).*
