# VERIFICATION — PR #102 revision 2, r10 / gfx1201 (2026-10-04)

Independent maintainer re-run of `verify-fusions.patch` (PR #102).  WIP record only, not promoted.

## Apply and build

* Worktree `/home/stew675/llama-fix97` at r10 canonical tip `b86854900` (tree `dab5186b…`).
* `git am wip/rdna4-verify-fusions-lf/verify-fusions.patch` applies cleanly; applied tree
  **`38ebce2f738f9486a5fc1a95d26ca5a523bac902`**, matches the PR.
* `BUILD_DIR=build-pr102 ~/bin/build-llama-rocm-714`: **clean, 0 warnings / 0 errors** (6m21s, ccache).

## Geometry sweep (`vf_sweep all`, 16,130 cases)

Built against `build-pr102/bin` and against the stock r10 `build-rocm-r10/bin`.

| run | hashes |
|---|---|
| fusions on (default) | 16,130 lines |
| fusions off (`GGML_CUDA_FUSE_GLU_Q8_1=0 GGML_CUDA_FUSE_GDN_CONV_VERIFY=0 GGML_CUDA_FUSE_CPY_BATCH=0`) | identical to on |
| stock r10 | identical to on |

`diff` empty for both pairs, **0 FAIL** lines.  Every case is bit-identical.

## Fusions actually fire (rocprofv3 kernel counts, r10)

| kernel | count | PR claim |
|---|---:|---|
| `cpy_batch` (debug counter, conv subset) | 9,112 | 9,112 |
| `unary_gated_q8_1_op_kernel` (glu subset) | 3,640 | 3,640 |
| `quantize_q8_1` (glu subset) | 520 (down from 4,160) | 520 |
| `gdn_conv_direct_kernel` (conv subset) | 8,400 | 8,400 |

So the bit-identity is not from the fusions failing to engage.

## End-to-end (Qwen3.8-27B-UD-Q4_K_XL + Qwen3.8-27B-DFlash2, `--spec-type draft-dflash`, q8_0 KV, 1 GPU)

```
llama-cli -m Qwen3.8-27B-UD-Q4_K_XL.gguf --spec-type draft-dflash \
  --spec-draft-model Qwen3.8-27B-DFlash2-Q4_K_M.gguf -ngl 99 -ctk q8_0 -ctv q8_0 \
  -c 8192 -b 2048 -ub 512 -fa auto --seed 42 --temp 0 \
  -p "The capital of France is" -n 64 --no-display-prompt --single-turn
```

| build | generated hash |
|---|---|
| patched, fusions on | `138 chars sha=55824e640aa0` |
| patched, fusions off | `138 chars sha=55824e640aa0` |
| stock r10 | `138 chars sha=55824e640aa0` |

Byte-identical across all three.  (The absolute hash differs from the recorded `1acb04bd9104` because
this run used a different GPU/ubatch config; the on/off/stock identity is the gate.)

## Code review notes

* The GLU to Q8_1 cache key and size match the `mmvq.cu` quantize path exactly
  (`ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1`, same `ne`/`nb` key), so the matmul reads
  the block the GLU wrote.  The mark lives in two fields of `ggml_backend_cuda_context`, is reset on
  every `ggml_cuda_try_fuse` call and cleared by the GLU launcher whether or not it fired.
* `cpy-batch` only merges consecutive `GGML_OP_CPY` nodes with identical source and destination
  layouts, checks destination/source span independence, skips no-op view nodes, and is gated off for
  concurrent streams and on non-RDNA4.
* `gdn-conv.cu` gates the 2..255-token arm on RDNA4 via
  `ggml_cuda_info().devices[ggml_cuda_get_device()].cc`, the same device idiom
  `ggml_cuda_should_fuse_mul_mat_vec_q` already uses.

## Verify-step A/B (throughput), 2026-10-04

Setup: `Qwen3.8-27B-UD-Q4_K_XL` (qwen35, hybrid SSM) + `Qwen3.8-27B-DFlash2-Q4_K_M`,
`--spec-type draft-dflash`, q8_0 KV, `-ub 512`, `-t 7`, greedy (`--temp 0 --seed 42`), `-n 512`,
three WikiText-2 prefixes (8.3K / 35.2K / 110.4K tokens), one GPU.  Alternating on/off runs, same
binary, all three kill switches flipped for the off arm.

Metric: the repo's reported generation t/s.  This is equivalent to ms/verify-step here because the
generated streams are byte-identical (below), so the verify-round count is fixed.

| depth | on (mean) | off (mean) | on/off | delta | on faster |
|---|---:|---:|---:|---:|---:|
| 8.3K | 21.520 | 21.200 | 1.0151 | +1.51 % | 5/5 |
| 35.2K | 64.980 | 63.980 | 1.0156 | +1.56 % | 5/5 |
| 110.4K | 64.533 | 63.700 | 1.0131 | +1.31 % | 3/3 |

Every on run beat every off run at every depth, with no overlap.  Generated text is byte-identical
within each depth, on == off: `e61f057b27d7` (8.3K), `8653b91f666b` (35.2K), `8f3c4553bbee` (110.4K).

So the ~1 to 1.5 % claim holds up under an alternating A/B with more repetitions than the PR ran, and
it is a bit larger than the PR's own single-run spread suggested (closer to +1.3 to +1.6 % here).

## Verdict

Correctness, fusion firing and every #100 review point pass.  The win is real and consistently
about +1.3 to +1.6 % end-to-end generation with DFlash at these depths.  The promotion decision is a
policy call: is that worth the new `cpy-batch` kernel, the relaxed GDN precondition and the GLU mark
mechanism?  With the delivery's default-on policy (a beneficial, gated, correctness-preserving
feature is on), the evidence supports promotion.

