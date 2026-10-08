# MOVED — this campaign is archived and its premise was refuted

`wip/gfx12-gdn-accuracy/` was **closed on 2026-10-08 and moved to
[`archive/work/gfx12-gdn-accuracy/`](../../archive/work/gfx12-gdn-accuracy/README.md)**.

The premise was that the gfx1201 (RDNA4) bf16 chunked-GDN kernel was less accurate than the gfx11 one.
**Measurement refuted that:** the two kernels have equal op NMSE and identical layer-0 real-data error,
and the gfx12 WMMA is marginally *tighter*.  Retiling gfx12 to the gfx11 shape is bit-identical, and an
fp16-operand variant that is **65x more accurate per op** (2e-7 vs 1.35e-5 NMSE) still moves the model KLD
only 1.5x.  The residual gfx1201-vs-gfx11 KLD is model-level numerical sensitivity, not a GDN defect.

The only delivery artifact is a **test-only coverage fix folded into block 02** (the cache-fusion bf16 gate
+ the model's exact op shape), which rides along with the next release.  No GDN kernel code changed.

This stub exists so historical pointers to `wip/gfx12-gdn-accuracy/...` still resolve.
