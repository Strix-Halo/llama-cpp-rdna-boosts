# Independent validation — gfx1201, 2026-10-03

The repo maintainer re-ran the PR's core claim on the delivery box (3x R9700 / gfx1201,
ROCm 7.14.1) against the **r36** tree + this PR's feature patch (applied clean on r36,
even though the PR was cut on r35). Result: **the BF16 fused op is bit-identical to the
unfused BF16 chain**, the fused path does engage, and it is **~+3.8 % decode** even on the
small 4-layer test model (see *Speed* below).

## Setup

The only local qwen4exp model is `Qwen3.8-Flash-Next-Q4_K_M` (`/llm/models/Qwen3.8/Flash-Next/Q4_K_M/`),
whose `hc_{attn,ffn}_{down,up,inject}` tensors are **Q8_0** (the standard quant), so the
BF16 path cannot be reached on it. A synthetic BF16-hc model was built from it:

```bash
# keep layers 0..3 (layer 3 is the compressed/indexer layer, so the hybrid-indexer
# memory layer still creates its cache) and put the 6 hc weight tensors per layer in BF16
./build-rocm/bin/llama-quantize --allow-requantize \
    --prune-layers 4,5,...,47 \
    --tensor-type-file hc_bf16.txt \
    <flash-next q4_k_m -00001-of-00005.gguf> hcbf16-4l.gguf Q4_K_M 16
# hc_bf16.txt:
#   hc_(attn|ffn)_(down|up|inject)\.weight=bf16
#   hc_(attn|ffn)_norm\.weight=f32
```

`--prune-layers` shrinks `block_count` to 4 but leaves the layer-length metadata arrays at 48,
so the output was patched in place (without moving the tensor data) to set
`qwen4exp.attention.compress_ratios` = `[0,0,0,4]` (4 entries) and bump `GGUF.kv_count` by one;
a dummy KV fills the freed metadata bytes so the data section offset is unchanged. Geometry is
the reporter's: `n_embd=2560`, `hc=4`, `hc_lr=320`. The model has no standalone `hc_head`
(the local Q4_K_M keeps `output_hc_*` Q8_0), but every per-layer mixer runs the BF16 op.

## Result

`llama-cli -ngl 99 -c 512 -ub N -b N -p <36-word prompt> -n 16 --seed 42 --temp 0 --single-turn --no-display-prompt`:

| ubatch (HC_MIX `nt`) | fused (default) | `LLAMA_HC_MIX_BF16=0` |
|---|---|---|
| 8 (prefill `nt=8`, decode `nt=1`) | `8ada57e4b2bd7522` | `8ada57e4b2bd7522` |
| 5 (prefill `nt=5`, decode `nt=1`) | `8ada57e4b2bd7522` | `8ada57e4b2bd7522` |
| 3 (prefill `nt=3`, decode `nt=1`) | `8ada57e4b2bd7522` | `8ada57e4b2bd7522` |

(sha256 of the extracted generated text.) A short-prompt run additionally exercised `nt=8`
and `nt=1` and matched (`06bf26849190be5a`). Two runs per mode were identical, so the
agreement is run-to-run stable and not a coincidence. A temporary instrumentation in
`ggml_cuda_op_hc_mix` confirmed the fused kernel is actually launched (3 calls logged per
fused run; 0 with the kill-switch). `test-backend-ops -o HC_MIX` stays **20/20** on all three
ROCm devices (those are the existing Q8_0 cases; the Q8_0 path is untouched — the BF16 arm is
type-gated and the shared `hc_mix_rms_gamma_quant` change only adds a `y == nullptr` early
return that the Q8_0 launcher never hits).

## Speed

`llama-bench -m hcbf16-4l.gguf -ngl 99 -p 0 -n 256 -r 5 -t 8`, three interleaved iterations
(fused = default, unfused = `LLAMA_HC_MIX_BF16=0`; the only graph difference is the mixer):

| iteration | fused tg256 | unfused tg256 |
|---|---|---|
| 1 | 293.26 | 282.11 |
| 2 | 292.53 | 282.66 |
| 3 | 292.29 | 281.07 |
| **mean** | **292.69** | **281.95** |

The fused path is **+3.8 %** on this model. The mechanism is exact: each mixer goes from 6
dispatches (rms×gamma, down mmvf, scale+silu, up mmvf, dsv4_hc_pre, inject mmvf) to 3. The
reporter's full model runs 96 mixers/token (48 layers × 2) vs this test's 8 (4 × 2), so
+3.8 % here is a diluted lower bound and the reported **+4.9 %** on the full GSQ-RCO model is
consistent. This is an end-to-end throughput number from `llama-bench`, not an op-level
extrapolation.

## Open before promotion into the delivery blocks

1. **Beta window / go-ahead.** The contributor's production use started 2026-10-02 (~1 day);
   the promotion rule wants the ~4-5 day window and the maintainer's go-ahead.
2. **CPU path.** `ggml/src/ggml-cpu/ops.cpp:ggml_compute_forward_hc_mix_f32` still asserts
   Q8_0 for down/up, while this PR relaxes the `ggml_hc_mix()` asserts to accept BF16. The
   model gate keeps the op off the CPU (it requires hc weights in a non-host buffer), but a
   BF16 op that *is* scheduled on CPU aborts. Adding the BF16 arm to the CPU reference is
   cheap and lets `test-backend-ops -o HC_MIX` cover the BF16 case.
3. **Backend consistency.** The gate tests `!ggml_backend_buffer_is_host()`, not CUDA. Vulkan
   and Metal declare no `HC_MIX` support at all, so a BF16-hc qwen4exp on those backends would
   build the op, find no supporting device, and fall back to CPU (item 2) — whereas the
   existing Q8_0 path falls back to the CPU reference and runs. Either make the gate
   CUDA-specific or land item 2 first.
4. **Block home.** `hc-mix.cu`, the `supports_op` arm, the `ggml.c` asserts and the
   `qwen4exp.cpp` gate all belong to block 14; folding means amending the block-14 commit and
   rebasing block 15 on it, then regenerating `patches/`, `rdna-boosts-all.patch` and
   `release.json`.
