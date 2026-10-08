# gfx1201 (RDNA4) bf16 chunked-GDN accuracy: match gfx11xx without losing the speed

**Status: OPEN / mid-investigation (2026-10-08, session 2).**  Successor to `archive/work/gdn-bf16-audit/` (the issue-#113
investigation, which is resolved).  This campaign is **accuracy refinement, not a correctness bug**: the
r30 delivery is correct and validated on all three arches.  The goal is to close the residual gap between
the RDNA4 (gfx12) and RDNA3/RDNA3.5 (gfx11) bf16 chunked-GDN kernels without giving back the prefill win.

**Session-2 headline:** the gfx12 bf16 kernel is **not less accurate per operation**.  Its WMMA is
marginally *tighter* than gfx11's; the two kernels round the same operands at the same points; the
synthetic op NMSE is equal (incl. the model's exact shape); and the layer-0 real-data error is identical.
The 13x model KLD comes from **cross-layer error compounding**: from layer 1 onward the gfx1201 within-arch
error grows ~2.6x faster (1.1e-2 vs 3.8e-3 at layer 23).  So the lever is the error's *structure*, not its
magnitude.  See "Session 2 findings" below.

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

## Hypotheses (ranked) -- original list, session 1

1. ~~The gfx12 WMMA instruction's internal reduction / fragment layout is coarser than gfx11's.~~
   **REFUTED (session 2, item 1).**
2. ~~A gfx12-kernel staging difference (e.g. the `gdn_fragP` union).~~  **REFUTED:** the r30 file has
   `union { uint2 w[2]; }` and the diff is layout-only (session 2, item 2).
3. Cross-chunk / cross-layer accumulation: **CONFIRMED as the mechanism** (session 2, item 5): equal
   layer-0 error, ~2.6x faster compounding on gfx1201 from layer 1.


## Session 2 findings (2026-10-08)

Instrumentation: `wmma-bf16-accuracy.cpp` (this directory); op-test additions in `_gfx12`
(only in the local checkout, see "Repro"); a temporary per-invocation GDN output dump in the fp32 and both
bf16 kernels (`GDN_DUMP_DST`, removed from the delivery tree; kept in the local checkout while the campaign
runs).  `gdn_dump_dst` writes the first 200 chunked invocations to `<path>.NNN`.

### 1. The WMMA instruction is not the cause (hypothesis 1 refuted)

Single 16x16x16 bf16 WMMA vs an fp64 reference on identical inputs:

| | 1 dot | 8 chained dots |
|---|---:|---:|
| gfx1201 (`_gfx12`) | 2.43e-8 | 1.03e-7 |
| gfx1100 (gfx11) | 2.95e-8 | 1.34e-7 |
| gfx1151 (gfx11) | 2.95e-8 | 1.34e-7 |

Both accumulate in full fp32 (`~2^-24`); gfx12 is marginally tighter.  Build/run:
`hipcc -O3 -DGFX12 --offload-arch=gfx1201 wmma-bf16-accuracy.cpp -o /tmp/wmma-gfx12`.

### 2. The two kernels are algorithmically identical

`diff` of the two files is layout-only: the fragment helpers, the C/state store layouts,
`GDN_ACC_M`/`gdn_store_acc8_b16`, `gdn_swrite`, the `SROW`/`KD` indexing, and the `NW`/`NTV`/`SVT` tiling.
The `gdn_f2bf`/`gdn_f2bf4` conversion sites match 1:1 (the gfx11 helper wrappers expand to the same set),
so **both kernels round the same operands to bf16 at the same points**.  The `s.gv` decay scaling is
correct for each fragment layout (gfx12 `4*hi+{0..3,8..11}` vs gfx11 `{0..15}`).

### 3. Synthetic op NMSE is equal, including the model's exact shape

`test-backend-ops -o GATED_DELTA_NET` with the bf16 gate forced to `1e-12` (temporary):

| case | gfx1201 | gfx1100 |
|---|---:|---:|
| `(16,128,256,2,2)` -- the model's real prefill op | 1.354e-5 | 1.364e-5 |
| `(16,128,1024,2,3)` | 1.353e-5 | 1.344e-5 |
| `(16,128,128/256/512/1024,1,3)` | 1.36-1.38e-5 | 1.33-1.40e-5 |

Signed bias on gfx1201 is tiny (`signed_mean ~ +3e-7`, `rms ~ 5e-4` for `(16,128,256,2,2)`), so no bias
difference either.

### 4. The model fuses the recurrent cache; the fused path (n_seqs=2) is fine

A one-time dispatch log (`GDN_OP_SHAPE`) on the 4B Qwen3.5 perplexity run shows the real op is
`S_v=128 H=32 (H_k=16, v_repeat=2) n_tokens=256 n_seqs=2 kda=0 cache=1 K=1` (plus a 2-token sequential
decode).  The fused-cache (`state_d_ext`) path is therefore live.  A new 128-wide cache-fusion case
(`(16,128,256,2,K=2)`) passes on both arches and both fp32/bf16; the `(16,128,256,1,K=2)` case fails on
**both** fp32 and bf16 (a pre-existing n_seqs=1 fused-cache issue, unrelated to bf16, worth a separate
look).

### 5. Real-data layer-0 error is identical, but it compounds 2.6x faster on gfx1201

`llama-perplexity` 4B Q8_0, wikitext-2, `-c 512 --chunks 1 -fa on`; within-arch bf16-vs-fp32 attn relRMS
per chunked invocation (the first 24; layer 0 is invocation 0):

| invocation | gfx1201 | gfx1100 | ratio |
|---|---:|---:|---:|
| 000 (layer 0) | 1.346e-3 | 1.336e-3 | 1.01 |
| 001 | 3.645e-3 | 1.693e-3 | 2.15 |
| 002 | 4.958e-3 | 2.029e-3 | 2.44 |
| 013 | 1.398e-2 | 4.414e-3 | 3.17 |
| 023 | 1.088e-2 | 3.785e-3 | 2.87 |
| mean ratio | | | **2.63** |

The layer-0 magnitude is identical; the divergence begins at layer 1 and stabilises at ~2.6x.

### 6. The fp32 base itself differs across arches

The fp32 (reference) layer-0 attn output differs across arches by **1.19e-3** relative (comparable to the
bf16 error itself), i.e. the upstream kernels (embedding/conv/attention/matmul/l2_norm) already round
differently on gfx12 vs gfx11.  So the model's inputs to GDN are not bit-identical across arches.

### Interpretation

The gfx12 kernel's per-op error has the same magnitude but a different fine **direction** than gfx11's
(equal RMS, different rounding reduction/accumulation order).  In this 24-GDN-layer recurrent model that
direction difference is amplified ~2.6x per layer, turning an equal per-op error into a ~13x model KLD.
Note the amplification is a property of the model's sensitivity, not of a specific bug: the layer-0 error
is already the same size.  This is why the op test (isolated, one op) shows nothing and why the model
gate does.

### Consequences for the goal

- "Make gfx1201 match gfx11" cannot be done by fixing an op-level inaccuracy -- there is none to fix.
- The available levers are (a) reduce the kernel's absolute error so the amplified result shrinks toward
the fp32 reference, or (b) reproduce the gfx11 error direction bit-for-bit (not possible across different
hardware/upstream codegen).
- (a) is a real option: the growth is proportional to the seed error, so fp16 operands (3 more mantissa
bits) or a compensated/fp32 state update would cut the compounded error roughly proportionally.  That is
exactly the "option 3" the issue-#113 audit deferred; the equal per-op NMSE was read as "fp16 will not
help", but the compounding result says it will shrink the chain even though it cannot zero it.

## Session 2 follow-ups (2026-10-08, later)

### Tiling experiment: NEGATIVE -- the tiling is not the lever

The gfx12 kernel was rebuilt with the gfx11 tiling (`NW=16`, `NTV=1`, `SVT=1`, the 4x4 wave map)
while keeping the gfx12 WMMA and fragment layout.  The within-arch bf16 error per layer is
**bit-identical** to the original gfx12 kernel at every invocation:

| layer | gfx1201 orig | gfx1201 retiled | gfx1100 |
|---|---:|---:|---:|
| 0 | 1.3460e-3 | 1.3460e-3 | 1.3361e-3 |
| 1 | 3.6446e-3 | 3.6446e-3 | 1.6933e-3 |
| 23 | 1.0878e-2 | 1.0878e-2 | 3.7847e-3 |

So the fp32 accumulation order (the tiling) does not enter the error at all -- each output's sum is
over the same k-values in the same order regardless of which wave owns the tile.  The gfx12 kernel's
numerics are fixed by the RDNA4 fragment layout / WMMA instruction and the input, and cannot be moved
by retiling.  (The retile was reverted.)

### The reported fused-cache failure is a TEST assertion, not a kernel bug

`(16,128,256,1,K=2)` fused-cache dumped and compared per snapshot slot: slot 0 is correct
(relRMS 2.95e-3 = the bf16 level), slot 1 is stale (relRMS 3.59).  The chunked kernel writes only
snapshot slot 0; the older slots are "caller-owned" (see the sequential kernel's `keep_rs_t` write:
"slot 0 = most recent state ... when n_tokens < K only slots 0..n_tokens-1 are written; older slots
are caller-owned").  A long prefill is never rolled back into (`GDN_CHUNKED_MIN_TOKENS = max(K, 16,
n_rs_batch)`; every verify batch is sequential and restores snapshots it wrote itself --
`common/common.cpp:1399-1401`), so those slots are never read.  The test was asserting them against
the CPU oracle.

Fix (test only):
- `test_gated_delta_net_cache_fusion` gets a `max_nmse_err()` override granting the bf16 chunked cases
the same `1e-4` gate as `test_gated_delta_net` (it previously had no bf16 relaxation at all).
- The bf16-eligible cache-fusion coverage is `(16,128,256,2,K=1)` and `(16,128,256,1,K=1)` (the chunked
  case the fused kernel writes in full) plus `(16,128,8,1,K=2)` (a sequential batch for the snapshot
  path).  The old `(...,K=2)` long-prefill rows were asserting caller-owned slots and were dropped.
- `test_gated_delta_net` also gains the model's exact shape `(16,128,256,2,2)` and `(16,128,256,1,2)`.

`GATED_DELTA_NET` is 48/48 and `GATED_DELTA_NET_CACHE_FUSION` 8/8 on gfx1201 (bf16 + fp32) and 48/48 on
gfx1100 and gfx1151 (bf16 + fp32).  The change is test-only; it belongs with block 02.  NOT yet folded
into `patches/` (awaiting the maintainer's go-ahead to cut the next release).

### Where that leaves the campaign

Per-op accuracy is equal and the tiling is a dead end; the only lever left is the **absolute** error
magnitude.  fp16 operands (10-bit mantissa) or a compensated/fp32 state update would shrink the seed
error, and the compounded result shrinks roughly in proportion -- it cannot make gfx1201 bit-match
gfx11, but it can pull gfx1201's KLD toward gfx11's.  That is the "option 3" the issue-#113 audit
deferred; the equal per-op NMSE was read as "fp16 will not help", but the compounding result says it
will scale the chain even though it cannot zero it.  Cost: a full fp16 port of both arch files, and
fp16 range (65504) on the recurrent state.

## Session 3 (2026-10-08): the KLD is model-sensitivity, not a GDN defect

### The four/six-way KLD table

Same `GGML_CUDA_GDN_CHUNKED_BF16` A/B (bf16 vs fp32 chunked) on the same 4B wikitext-2 gate,
4B Q8_0 and its F16 sibling (F16 = the Q8_0 dequantized via `llama-quantize --allow-requantize`, so
the **weights are identical**; only the matmul implementation differs):

| model / arch | mean KLD | median KLD | same top p |
|---|---:|---:|---:|
| Q8_0 / gfx1201 | 0.001037 | 0.000377 | 98.66 % |
| Q8_0 / gfx1100 | 0.000219 | 0.000029 | 99.54 % |
| Q8_0 / gfx1151 | 0.000170 | 0.000030 | 99.51 % |
| F16 / gfx1201 | **0.000019** | 0.000003 | 99.78 % |
| F16 / gfx1100 | 0.002087 | 0.001447 | 97.29 % |
| F16 / gfx1151 | 0.002534 | 0.001522 | 97.05 % |

The identical bf16 GDN kernel gives a model-level KLD spanning >100x, and the arch ranking **flips**
between the two model precisions (gfx1201 worst on Q8_0, best on F16).  A kernel defect cannot do that;
the KLD is measuring how a fixed ~1.3e-3 per-op bf16 perturbation propagates through a specific
quantized/precision pipeline, and that amplification is chaotic.

### The GDN inputs differ upstream, at the quantization scale

Layer-0 GDN inputs, gfx1201 vs gfx1100 (both Q8_0): q 3.3e-3, k 4.4e-3, v 1.6e-3, g 1.2e-3,
beta 1.4e-3 (q maxabs 7.9e-3 against mean |q| 4.3e-2).  No attention runs before layer 0, so this is the
embedding -> RMS norm -> Q8_0 projections chain.  The cross-arch difference vector is **91-93 %
cosine-correlated** with the same-arch Q8_0-vs-F16 difference, and both are the same magnitude, so the
cross-arch divergence is at the Q8_0 activation-quantization scale, not a GDN input difference.

`ggml_cuda_should_use_mmq` returns true for Q8_0 on both RDNA3 and RDNA4, so both arches take MMQ (the
MMQ-vs-hipBLAS path is not the difference); the remaining candidate is the arch-tuned MMQ tile/schedule,
or the precision of the surrounding kernels.

### The tiling is not the lever (session-2 result, restated)

Retiling gfx12 to the gfx11 shape (`NW=16`, `NTV=1`, `SVT=1`) is **bit-identical** to the original, so the
fp32 accumulation order does not enter the error at all.

### Session-3 conclusion

The gfx12 bf16 GDN kernel is not less accurate than the gfx11 one.  There is no GDN defect to fix, and
"make gfx1201 match gfx11" is not reachable by changing the GDN kernel.  The only GDN-side lever is to
shrink the absolute bf16 error (fp16 operands / compensated state update), which shrinks the amplified
result roughly proportionally but cannot reproduce gfx11's error direction.

### New lead outside the GDN: the F16 model is ~100x worse on gfx11

The F16 (dequantized) model's bf16-GDN KLD is 0.000019 on gfx1201 but 0.0021-0.0025 on gfx1100/gfx1151.
That is the reverse of the Q8_0 ranking and a much larger ratio, so the gfx11 F16 matmul path (or the
gfx11 kernels around it) may itself be less accurate / more amplification-prone in the F16 regime.  That
is a separate investigation from this campaign, but it is the more promising one if the goal is a real
accuracy win, and it is where the next session should look.

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
