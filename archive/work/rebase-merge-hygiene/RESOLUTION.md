# RESOLUTION — re-base merge hygiene (distribute the block-15 build fixes)

**Resolved:** 2026-10-05 (release `v16-a55e952b8-r2`).
**Status:** DONE — every block commit builds (`all` target), final tree valid, `validate-set.sh` green.

---

## Outcome in one line

The r1 chain had **eight** independent intermediate-build breaks, not the four/five the handover
inventoried.  All were assigned to the earliest block that owns the broken code, the block commits
were rebuilt with the same messages/authors (new SHAs, same per-block content plus the fixes), and
**every one of the 16 commits now builds the `all` target** with `~/bin/build-llama-rocm-714`
(gfx1201 / ROCm 7.14, fresh configure per block).

* new canonical tip **`dbe88ea6e3afd86da26ce766ae8b71d2b26b67ac`**
* merge-hygiene tree **`6a44aa2904772db02dbc88960397efe8138498df`** (unchanged from r1)
* r2 tree **`c38ba8f2066f01c3a1f69207a7e0860e5026ef17`** (the two hardening changes below)

## The complete break inventory (found by clean-configuring every block)

The first sweep was misleading: the reused `build-rocm`/`build-r38` cache kept the tip's
`GGML_CUDA_FA_QUANTS`, so blocks 00-07 died in *configure* with a stale `iq4_nl` complaint.  A fresh
`rm -rf` + configure + `cmake --build --target all` per commit produced the real list.  The
handover's 11-file "build fixes" commit (`95d3b11c8`) only fixes what the *final* tree needed; it does
not cover block 02/03/04, nor the second half of the block-14 hybrid.

| owning block | file | break | fix |
|---|---|---|---|
| 01 | `tests/test-recurrent-state-depth.cpp` | old `common_batch_add`/`llama_decode` API | migrate to `common_batch.add`/`llama_process` |
| 01 | `include/llama.h`, `src/llama-cparams.h`, `src/llama-context.cpp` | the test sets `llama_context_params::n_rs_batch`, a block-02 field | move the field + default-init to block 01 |
| 02 | `src/llama-model.cpp` | general `llama_memory_hybrid_idx` ctor call missing the `n_rs_batch` block-02 added | add `/* n_rs_batch */ cparams.n_rs_batch,` |
| 03 | `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | gfx11 signed-zero guard says `!swz_V`; the variable is `swz` (upstream `1884824fd` collapsed `swz_K`/`swz_V`) | `!swz_V` → `!swz` |
| 04 | `ggml/src/ggml-cuda/fattn.cu` | an extra `}` after the WMMA `logit_softcap` block closes the kernel chooser early | drop the extra brace |
| 04 | `src/llama-context.cpp` | the tensor-split FA hint calls `ggml_backend_dev_is_cuda` before its (block-15) definition | move the helper definition to block 04 |
| 08 | `ggml/src/ggml-cuda/ggml-cuda.cu` | missing `}` after `ggml_cuda_op_rms_norm_scale_fused(...); return 1;` nests the rest of `ggml_cuda_try_fuse` | close the `if` |
| 08 | `ggml/src/ggml-cpu/ops.cpp` | **(new hardening)** CPU `cmp_argsort` compares values only, so its tie order is unstable while the CUDA bitonic/CUB path is index-stable | make it a total order with an index tie-break |
| 08 | `tests/test-backend-ops.cpp` | **(new hardening)** the `ARGSORT` matrix only used unique values; ties were untested | add a `ties` variant + `{2048,8,1,1}` and `{4096,2,1,1}` cases |
| 13 | `ggml/src/ggml-cuda/mmq.cu` | missing `}` closing `ggml_cuda_mmq_get_prec_src1` | add `}` |
| 13 | `ggml/src/ggml-cuda/mmq.cuh` | `DECL_MMQ_CASE_W4A4` used but never defined | restore the `#define` |
| 13 | `ggml/src/ggml-cuda/mmvq.cu` | `mul_mat_vec_q_ksplit`'s `stride_col_dst` is `const` but assigned in the shared-expert lane | drop `const` |
| 14 | `src/llama-memory-hybrid-idx.cpp` | missing `}` closing `set_input_kpool` | add `}` |
| 14 | `src/models/qwen4exp.cpp`, `src/models/models.h` | the re-based hybrid splices a QSA class header (`llm_graph_input_qsa`) onto a kpool class body (`llm_graph_input_kpool` ctor, `pool_cells` members) *and* upstream's kpool graph routing, and models.h lost `build_qsa_sel`'s parameter list | restore the coherent pre-block-15 QSA `qwen4exp.cpp` from the **r37 block-14** commit and the tip's matching `models.h` |

Block 15 keeps the fix-commit's genuine **self**-fixes: `common/speculative.cpp` (DFlash `llama_batch_free`/
`batch.tokens[row].pos[0]`), `qwen4exp.cpp` (`s_copy_extra` → `s_copy_tail`), `test-backend-ops.cpp`
(one `}`) and the two `ggml-cuda.cu` shared-expert braces.  They were already correct in the tip.

## Why the block-14 hybrid was restored from r37, not "repaired"

The block-14 `qwen4exp.cpp` is byte-for-byte a 3-way-merge collision: the class is named
`llm_graph_input_qsa` but its constructor/destructor are `llm_graph_input_kpool(...)`, and the file then
defines upstream's `build_inp_kpool`/`build_qsa_sel` whose declarations are absent from `models.h`.
No local renaming fixes it because the graph routing and the class members disagree in both
directions.  The coherent pre-block-15 content is the **r37 block-14** (`11ad48b15`) file; the r1
merge had simply failed to produce it.  Block 15 still carries the genuine r37 block-15 delta, so the
`block 14 → 15` patch is still non-empty (the QSA memory-campaign changes).

## Verification (gfx1201 / ROCm 7.14, `~/bin/build-llama-rocm-714`)

* **`all` target green at every one of the 16 commits** (fresh `rm -rf build-check` + configure each
  time, `-j16`, ccache).  The handover's acceptance is at least `ggml-hip` + `llama`; we ran `all`.
* `scripts/validate-set.sh` green: base `a55e952b8`, strict **16/16** `git am`, applied tree
  `c38ba8f2066f01c3a1f69207a7e0860e5026ef17` == `release.json.tree`.
* `test-backend-ops`: `ARGSORT` **78/78** (was 74/78 before the CPU tie-break fix), `MUL_MAT_ID`
  **931/931**, `RMS_NORM` **51/51**, `INDEXER_TOPK` **3/3**, `HC_MIX` **30/30**,
  `GATED_DELTA_NET` **46/46**.
* Coherence unchanged: dense 4B 3-GPU `-sm tensor` `1c5d32ac537d`; qwen4exp Flash-Next IQ4_NL
  3-GPU `359ff4337837`.

## Reproduce

```bash
# per-block build check (fresh configure each commit)
for c in $(git -C ~/llama.cpp rev-list --reverse a55e952b8..dbe88ea6e); do
    git -C ~/llama.cpp checkout -q $c
    rm -rf ~/llama.cpp/build-check
    HIPCXX=/opt/rocm-7.14.1-gfx120X/lib/llvm/bin/clang HIP_PATH=/opt/rocm-7.14.1-gfx120X \
      cmake -S ~/llama.cpp -B ~/llama.cpp/build-check <same flags as ~/bin/build-llama-rocm-714> >/dev/null
    cmake --build ~/llama.cpp/build-check --target all -j16 -- -k || echo "FAIL $c"
done
```
