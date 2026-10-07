# MOVED — this campaign is archived

`wip/host-pinned-buffer-crash/` was **closed on 2026-10-07 and moved to
[`archive/work/host-pinned-buffer-crash/`](../../archive/work/host-pinned-buffer-crash/README.md)**.

The `-sm tensor` host-expert CPU fallback and the inert cache shipped in r15 (blocks 06 + 13).  The last
open half — the `--load-mode none` GPU page fault — was re-tested on the r24 delivery and **no longer
reproduces** (14/14 clean; the routed experts are pinned `ROCm_Host` and the host-gathered PLE sits on
pageable CPU).  The only residual is removing the now-stale `common/common.cpp` warning (`TODO.md` #45).

The open-campaign index that used to live in this directory's README is now at
[`wip/CAMPAIGNS.md`](../CAMPAIGNS.md).

This stub exists so the historical pointers to `wip/host-pinned-buffer-crash/...` still resolve.
