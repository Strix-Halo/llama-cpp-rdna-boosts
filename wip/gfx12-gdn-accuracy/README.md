# gfx1201 (RDNA4) bf16 chunked-GDN accuracy: match gfx11xx without losing the speed

**Status: OPEN / scoping (2026-10-08).**  Successor to `archive/work/gdn-bf16-audit/` (the issue-#113
investigation, which is resolved).  This campaign is **accuracy refinement, not a correctness bug**: the
r30 delivery is correct and validated on all three arches.  The goal is to close the residual gap between
the RDNA4 (gfx12) and RDNA3/RDNA3.5 (gfx11) bf16 chunked-GDN kernels without giving back the prefill win.

**Nothing here is part of the delivery.**  Do not apply anything under `wip/` to the fork without the
maintainer's explicit go-ahead.  Any candidate must keep `GGML_CUDA_GDN_CHUNKED_BF16` default-on, pass
`scripts/gate-prefill-logits.sh` (mean KLD <= 0.005, same-top-p >= 98 %, **with `n_seq > 1`**), and be
validated on gfx1201 + gfx1100 + gfx1151.

## Goal / success criteria

Make the gfx12 kernel's bf16 error distribution match the gfx11 kernel's as closely as possible, while
keeping the gfx1201 bf16 prefill win (~+7.7 % over the fp32 chunked kernel).  Concretely, target the
"baseline" numbers below (median KLD ~3e-5, top-1 miss ~0.5 %) at no perf cost.  If a change cannot do
that, document why (e.g. hardware) and keep the current tuned gfx12 kernel.

## Current state (delivery r30)

- `GGML_CUDA_GDN_CHUNKED_BF16` is **default-on** for `S_v == 128` on RDNA3/RDNA4 (`=0` opts out).  The
  r29 default-off was the `A_sc` `n_seqs > 1` stride-aliasing workaround, fixed in r30 (block 02).
- Issue #113 is closed on the correctness side; this is now a quality/parity project.
- The gfx12 kernel is `ggml/src/ggml-cuda/gated_delta_net_chunked_bf16.cu`; the gfx11 kernel is
  `..._bf16_gfx11.cu`; the fp32 chunked reference is `gated_delta_net_chunked.cu`.

## The evidence

### Clean same-model cross-arch comparison (4B Q8_0, identical file, `-c 512 --chunks 40 -fa on`)

| metric | gfx1201 (gfx12) | gfx1100 (gfx11) | gfx1151 (gfx11) |
|---|---:|---:|---:|
| mean KLD | 0.001037 | 0.000219 | 0.000170 |
| **median KLD** | **0.000377** | **0.000029** | **0.000030** |
| 95% KLD | 0.001947 | 0.000177 | 0.000174 |
| 99% KLD | 0.004709 | 0.000487 | 0.000541 |
| 99.9% KLD | 0.018394 | 0.004199 | 0.003230 |
| top-1 miss | 1.343 % | 0.461 % | 0.490 % |

The two gfx11 arches are indistinguishable, so the gfx11 kernel is the **baseline target** and is stable
across RDNA3 and RDNA3.5.  The gfx12 typical error is ~13x larger.

### 27B cross-check

- gfx1201 27B Q8_0: mean KLD 0.000707, same-top-p 98.755 %, median KLD 0.000227, 99 % KLD 0.00510.
- gfx1151 27B Q8_0: mean KLD 0.000150, same-top-p 99.667 %, median KLD 0.000018.
- gfx1100 27B Q4_K_M: mean KLD 0.000052 (confounded by the quant; the 4B test above supersedes it).

### Op-level accuracy is equal

`test-backend-ops -o GATED_DELTA_NET` (realistic gates `-0.5 .. -1e-4`, tight gate) gives ~1.2-1.4e-5
NMSE on **all three** arches, for both `n_seqs=1` and `n_seqs=2`.  So the single-op accuracy is the same;
the difference is only visible at the model level (error distribution / correlation across chunks).

## What is ruled out

- **Tiling / reduction order.**  `NTV`/`SVT` change how many independent outputs a wave computes, not the
  summation order of any one output.  Both kernels reduce the k-dimension as 8 `sk` steps of one 16-wide
  WMMA and the state update over the same 4 steps.  So the difference is not the tiling.
- **Operand mantissa width / state / inverse / K / Q / V** (from the issue-#113 audit): a bf16-rounded
  fp32 algorithm stays near-lossless.  Not re-opened here.
- **The `A_sc` stride aliasing** (r30 fix): resolved and validated.
- **Shape / padding bugs**: `n_seq_tokens = 42/63/65` are on par with the other sizes.
- **Model / quant confounds**: the 4B test uses one identical model file on all three arches.

## Hypotheses (ranked)

1. **The gfx12 WMMA instruction's internal reduction / fragment layout is coarser than gfx11's.**
   `wmma_f32_16x16x16_bf16_w32_gfx12` uses 8 bf16/lane ("two runs of four"); the gfx11
   `wmma_f32_16x16x16_bf16_w32` uses 16 bf16/lane (full row).  If RDNA4 combines the 16-term reduction
   differently (or at lower intermediate precision), this is hardware and no surrounding code can fix it.
2. **A gfx12-kernel staging difference.**  e.g. the gfx12 `gdn_fragP` reads 4 `uint2` (16 shorts) into a
   union whose `gdn_v8bf` is only 8 elements; check whether the read/use layout is right and whether the
   conversion path does an extra round-trip the gfx11 kernel avoids.
3. **Cross-chunk accumulation/correlation.**  Same per-op NMSE but a broader model distribution could be a
   more correlated per-chunk error.  The replay below distinguishes it.

## Next steps (in order)

1. **bf16 WMMA microbenchmark (smallest, decides hypothesis 1).**  One `wmma_f32_16x16x16_bf16` with known
   random bf16 operands, summed over a fixed k, compared against an fp64 CPU reference, run on gfx1201
   and gfx11xx.  If gfx12's per-instruction error is ~10x worse, it is the instruction and the campaign is
   a hardware note, not a code change.
2. **Real-data replay (decides 2 vs 3).**  Add a temporary dump of the GDN op *inputs* (q/k/v/g/beta/state)
   for one real model invocation, then run both kernels on the identical tensors and compare the output
   error and bias.  The issue-#113 campaign already built the *output* dump
   (`archive/work/gdn-bf16-audit/README.md` has the helper and the invocation-by-invocation growth table
   for gfx1201); extend it to inputs and port it to the gfx11 kernel.
3. **Only if 2 points at code**, fix the gfx12 staging and re-measure KLD + `llama-bench` on all three.
4. **If 1/2 say hardware**, evaluate whether a different RDNA4 path is both more accurate and still faster
   than fp32: e.g. fp16 operands (10-bit mantissa) or a split/compensated bf16 WMMA for the sensitive
   operands.  Measure KLD against the gate and prefill against the fp32 chunked arm.
5. **Do not** retile gfx12 to the gfx11 shape blindly: the op NMSE is already equal, so it is unlikely to
   move the model KLD, and it risks the RDNA4 prefill win (the tuned `NTV=2` amortises K/Q fragment loads).

## Environment / repro

Boxes: `soar` (gfx1201, local), `fingon` (gfx1100), `halo` (gfx1151).  The fork's `rdna-boosts` branch is
at the **r30** tip `e14c10fbe` (tree `f832fb68`).  The remotes currently carry a mix of scp'd r30 kernel
files and an r29 dispatch, so **refresh them first**:

```bash
# on fingon / halo
cd ~/llama.cpp && git fetch origin && git stash -u 2>/dev/null; git checkout -f rdna-boosts && git reset --hard origin/rdna-boosts
cmake --build build-rocm -j 16 --target llama-perplexity llama-bench test-backend-ops
```

The same-model cross-arch repro (4B Q8_0), run on each box:

```bash
M=/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf
W=/llm/models/wikitext-2-raw/wiki.test.raw
B=/tmp/4b-compare.kld
# fp32 base (explicit on r30, the default is bf16)
GGML_CUDA_GDN_CHUNKED_BF16=0 <ppl> -m $M -f $W -c 512 --chunks 40 -ngl 99 -fa on -t 8 --kl-divergence-base $B
# bf16 candidate
<ppl> -m $M -f $W -c 512 --chunks 40 -ngl 99 -fa on -t 8 --kl-divergence-base $B --kl-divergence
```

Capture `Mean/Median/95 %/99 %/99.9 % KLD` and `Same top p`.  The gate is
`scripts/gate-prefill-logits.sh` (it uses the `-c 512` default batch, i.e. `n_seq = 4`).

Op NMSE (temporarily set `test_gated_delta_net::max_nmse_err` to `1e-12` to print it, then revert):

```bash
HIP_VISIBLE_DEVICES=0 <build>/bin/test-backend-ops -b ROCm0 -o GATED_DELTA_NET
```

## Pointers

- Correctness fix + investigation: `archive/work/gdn-bf16-audit/README.md`, `WORKLOG.md` 2026-10-08
  (r29/r30), `patches/README.md` block 02.
- Gate: `benchmarks/prefill-logit-methodology.md`, `scripts/gate-prefill-logits.sh`.
- Kernels: `ggml/src/ggml-cuda/gated_delta_net_chunked_bf16.cu` (gfx12),
  `..._bf16_gfx11.cu` (gfx11), `gated_delta_net_chunked.cu` (fp32 reference).
