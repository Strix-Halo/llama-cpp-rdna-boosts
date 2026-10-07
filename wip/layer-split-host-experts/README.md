# MOVED — this campaign is archived

`wip/layer-split-host-experts/` was **closed on 2026-10-07 and moved to
[`archive/work/layer-split-host-experts/`](../../archive/work/layer-split-host-experts/README.md)** (the
record and `fix.patch`, kept verbatim).

The fix was **PROMOTED to the delivery as part of block 06** in `v16-a55e952b8-r14` (2026-10-05; r13
carried it as a separate block 16): per-device host bufts + a layer-device host-buft choice, so `-sm layer`
with host-resident experts spreads the expert ops over the GPUs instead of routing them all to device 0
(2 x R9700, IQ4_NL `-sm layer -ncmoe 48`: **10.3 -> 55.4 t/s**).

This stub exists so the historical pointers to `wip/layer-split-host-experts/...` still resolve.
