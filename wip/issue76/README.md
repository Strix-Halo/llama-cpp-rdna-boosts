# wip/issue76 — launch_fattn: declare `kq_blocks` in allocation order

Patch for issue #76. One `git am` patch against `v16-84e76d8a2-r27` (applied tree `7427f424…`, matching `release.json`). It only moves the `kq_blocks` pool buffer declaration so the buffers are declared in the order they are allocated; no computation changes.

Validated on r14 (R9700, gfx1201, ROCm 10, VMM pool build): the aborting request completes after the fix. The r27 patch applies to r27 and was checked by reading the source; it has not been rebuilt on r27. Details are in issue #76.

*The patch and this note were written by Claude (AI).*
