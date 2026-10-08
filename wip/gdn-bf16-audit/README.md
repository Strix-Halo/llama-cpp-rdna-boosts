# BF16/WMMA chunked GDN: find and fix the prefill-logit divergence (issue #113)

**Status: OPEN / mid-investigation (2026-10-08).**  This campaign is the follow-up to issue #113.  The
r29 delivery already ships the safety fix (the bf16 path is opt-in, `GGML_CUDA_GDN_CHUNKED_BF16=0` by
default).  This campaign exists to decide whether the bf16 prefill win (~4-8 % end-to-end on gfx1201 /
gfx1100) can be recovered legitimately without the KLD penalty, or whether it should be left off.

**Nothing here is part of the delivery.**  Do not apply anything under `wip/` to the fork without the
maintainer's explicit go-ahead (see the WIP and promotion rules in `AGENTS.md`).  Every candidate must
pass `scripts/gate-prefill-logits.sh` (mean KLD <= 0.005, same-top-p >= 98 %) and then be validated on
gfx1201 + gfx1100 + gfx1151 before it can be promoted.

## Context

- The reader: `benchmarks/prefill-logit-methodology.md` (the gate protocol) and `WORKLOG.md` 2026-10-08
  (r29) (the finding and the default flip).
- The kernels: `ggml/src/ggml-cuda/gated_delta_net_chunked_bf16.cu` (gfx12/RDNA4, ~1000 lines) and
  `..._gfx11.cu` (gfx11/RDNA3/RDNA3.5, a near-copy).  The fp32 reference is
  `gated_delta_net_chunked.cu`.
- The hypothesis to test (maintainer): the bf16 *numerics* may be near-lossless, and the divergence may
  be an implementation/formulation issue in the kernel rather than the mantissa width.

## Environment

| host | arch | build | notes |
|---|---|---|---|
| `soar` (local) | gfx1201 | `~/llama.cpp/build-rocm-hybrid` (r29, tree `7061a481`) | 3 x R9700 |
| `fingon` (`ssh`) | gfx1100 | `~/llama.cpp/build-rocm` (r29) | 1 x W7800-class, 24 GiB |
| `halo` (`ssh`) | gfx1151 | `~/llama.cpp/build-rocm` (r29) | Strix Halo |

The recorded known-good gate base on soar is
`~/.cache/rdna-boosts/gates/prefill-logit-gfx1201-Qwen3.8-27B-Q8_0.kld` (fp32 chunked default, 40 x 512
wikitext-2).  `scripts/gate-prefill-logits.sh` is the entry point.

## What was measured (2026-10-08, gfx1201, 27B Q8_0, 40 x 512 wikitext-2)

Mean KLD vs the recorded fp32-chunked default base:

| build / arm | mean KLD | same top p |
|---|---:|---:|
| fp32 chunked (reference) | 0.00054 | 98.8 % |
| **bf16 kernel (the one in question)** | **0.032** | **93.6 %** |
| bf16 kernel with the KKT inverse computed fp32 ("option 1") | 0.038 | 93.4 % |
| fp32 kernel, only the KKT inverse *operand* rounded to bf16 | 0.00054 | 98.8 % |
| fp32 kernel, only the recurrent state *operand* rounded to bf16 | 0.00047 | 99.0 % |
| fp32 kernel, `Vb`+`U`+`QS`+`KQ`+state rounded to bf16 | 0.00059 | 98.8 % |
| fp32 kernel, also `K`+`Q` rounded to bf16 | 0.00058 | 98.8 % |

Op-level NMSE vs the `test-backend-ops` CPU oracle (temporary tight gate, `-o GATED_DELTA_NET`):

| kernel | NMSE on the `head_size=128, n_seq_tokens>1` cases |
|---|---|
| fp32 chunked | **0.000000000** (bit-exact with the CPU oracle) |
| bf16 kernel | **~9e-6** (7 cases; 8.6e-6 .. 1.6e-5) |
| fp32 kernel with *every* operand rounded to bf16 | **~6e-6** |

## The two facts that do not yet reconcile

1. **At the op level, the bf16 kernel is exactly as accurate as bf16 operand rounding predicts**
   (~9e-6 vs the ~6e-6 of a bf16-rounded fp32 algorithm).  So, on the synthetic op-test inputs, the
   bf16 kernel is not "buggy" — it is doing bf16.
2. **At the model level, the bf16 kernel's KLD (0.032) is ~50x the bf16-rounded fp32 proxy's (0.0006),
   despite the two op NMSEs being within ~1.5x of each other.**

That discrepancy is the crux.  It means the model KLD is not a function of the op NMSE alone: either the
bf16 kernel's error is *data-dependent* (much larger on the model's learned gates / real state
magnitudes than on the synthetic test gates), or the model exercises a path the op test does not (most
suspect: the fused GDN->cache prefill path, `state_d_ext`), or the error is spatially/structurally
correlated in a way that the aggregate NMSE hides and the model amplifies.

## Ruled out

- **Option 1 (fp32 KKT inverse)**: no gain (0.038 vs 0.032); the inverse operand rounded to bf16 costs
  4e-6 mean KLD in an otherwise-fp32 pipeline.  The inverse is not the source.
- **Option 2 (split the state operand)**: the state operand rounded to bf16 costs ~0 in the fp32
  algorithm.  Also mechanically blocked: the scan kernel already uses ~61 KB of the 64 KB LDS limit at
  BV=64, so a second bf16 state buffer (or an fp32 state) does not fit without a BV=32 retile.
- **Option 3 (fp16 operands)**: the reasoning that motivated it (operand width) is weakened by fact 1 —
  at the op level the bf16 kernel already matches bf16 rounding, so 3 extra mantissa bits may not touch
  the model-level divergence.  A full fp16 port (both arch files) is not justified until fact 2 is
  explained.  Range (fp16 max 65504) on the recurrent state is a secondary risk.
- **Options 4/5 (fp32-FMA state products / Ozaki bf16x2)**: precision-only, so fact 2 applies to them
  too; 5 is additionally the slowest and most work.

## Next steps (in order)

1. **Dump real GDN inputs and compare the two kernels on them.**  Add a temporary hook that writes the
   q/k/v/g/beta/state for the first GDN invocation of a real prefill (one ubatch) to a file, then run
   both kernels on that dump and diff `attn_out`/`state_out`.  This will show whether the model's data
   drives a bf16 error far above the synthetic 9e-6.

## Second session (2026-10-08) -- real-data differential

The dump hook was built (temporary, reverted; see the git history of `gated_delta_net_chunked.cu` /
`..._bf16.cu`): every chunked GDN invocation is written to `<path>.NNN` (attn) and `<path>.NNN.state`.
On `llama-cli -p <wikitext>` with the 27B Q8_0, the model processes the prompt in **42-token chunks**
(192 chunked invocations = 16 GDN layers x 12 chunks), and each invocation runs a single 64-token
kernel chunk.  Comparing clean fp32 vs bf16 invocation-by-invocation, layer 0:

| chunk | attn relRMS | state relRMS |
|---|---:|---:|
| 0 | 7.5e-5 | 3.4e-3 |
| 1 | 4.0e-6 | 7.9e-3 |
| 2 | 4.8e-6 | 2.4e-2 |
| 4 | 1.3e-2 | 9.4e-3 |
| 5 | 2.0e-2 | 2.2e-2 |
| 11 | 1.9e-2 | 4.5e-2 |

So the **error compounds across prefill chunks**: the per-chunk state error starts at the bf16 level
(3.4e-3) but the accumulated divergence reaches 1-4e-2 in both the state and the attn output by the last
chunk.  This is the source of the end-to-end KLD: the model KLD is dominated by the later chunks.

The op-level NMSE had hidden this because a single op (`n_seq_tokens` up to 1024) is compared against a
CPU oracle in isolation; the *cross-invocation* carry is where the bf16 kernel degrades.

### Negative result that reframes it

Rounding the bf16 kernel's carried fp32 state to bf16 after every chunk (matching the bf16 operand
precision, which the fp32-operator proxy does) did **not** help: mean KLD **0.0329** vs 0.0324.  So the
compounding is **not** the carried-state mantissa; it is injected by the per-chunk state *update*
(`S' = e^{g_last} S + (e^{g_last-g_t} K^T) @ V'`), whose bf16 operands (double-rounded `bf16(K)*el` and
bf16 `V'`) inject a per-chunk error that the recurrence carries forward.  The fp32-operator proxy does
not compound (KLD 0.0006) even though it also rounds `V'`, because it uses fp32 `K*el` in that update.

### Reframed hypothesis (next experiment)

The state-update **`K` operand** is the leading suspect: the bf16 kernel feeds `bf16(bf16(K)*el)` into
the update WMMA, while the proxy feeds fp32 `K*el` and stays near-lossless.  Test it directly by
rounding the proxy's state-update `K` to bf16; if the proxy then reproduces ~0.03, the fix is to keep
the state-update K (or the whole `el*K^T` operand) at higher precision (e.g. a split bf16x2 for that one
operand, or fp32 FMA for the state update only).

### Original next-steps list (kept for reference)

1. Dump real GDN inputs and compare the two kernels on them (done above).
2. **Check the fused-cache path.**  The op test's `test_gated_delta_net_cache_fusion` runs at the default
   gate, not the tightened one; the model's prefill writes the state through `state_d_ext`.  Tighten that
   test too and run it with `GGML_CUDA_GDN_CHUNKED_BF16=1`.
3. **Characterize the real gate spans / state magnitudes.**  The bf16 kernel's `ge = __expf(gcs)` can
   underflow fp32 to 0 on a 64-token chunk with large negative gates (~e^-88), and the state can grow
   with v.  Dump `max|gcs|`, `max|S|`, `max|V|` per chunk on real data and check for underflow/large
   magnitudes.  (Both kernels use fp32 for `ge`, so this is only a lead if the bf16 kernel handles the
   underflowed values differently.)
4. Only once the source is identified: implement the fix in **both** arch files, then run the gate and
   `llama-bench` on all three boxes.

## Reproduction commands

```bash
# Op-level NMSE (temporarily set test_gated_delta_net::max_nmse_err to 1e-12 to print it):
cmake --build ~/llama.cpp/build-rocm-hybrid --target test-backend-ops -j 16
HIP_VISIBLE_DEVICES=0 ~/llama.cpp/build-rocm-hybrid/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET
GGML_CUDA_GDN_CHUNKED_BF16=1 HIP_VISIBLE_DEVICES=0 \
  ~/llama.cpp/build-rocm-hybrid/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET

# End-to-end gate (soar):
~/llama-cpp-rdna-boosts/scripts/gate-prefill-logits.sh          # default (fp32) PASS
~/llama-cpp-rdna-boosts/scripts/gate-prefill-logits.sh --ab "GGML_CUDA_GDN_CHUNKED_BF16=1"  # records bf16 base
```

## Do-not

- Do not promote any precision-split variant that has not been re-validated as a whole against the gate.
- Do not duplicate the change only in the gfx12 file; gfx11 is a separate kernel and moves independently.
- Do not forget `LLAMA_SPEC_DRAFT_N_MAX_CLAMP`-style gates for any new env knob; default OFF while the
  gate result is uncertain.
