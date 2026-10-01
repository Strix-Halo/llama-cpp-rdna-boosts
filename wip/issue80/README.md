# wip/issue80 — common_sampler: exact top-k fast path, sampler clone without the candidate copy

Patch for issue #80. One `git am` patch, made on `v16-84e76d8a2-r14`. It only touches `common/sampling.cpp`, which no rdna-boosts block changes, so it applies unchanged to r27.

Validated on an r14-based build (R9700, gfx1201, ROCm 10; Linux and Windows): byte-identical output with the fast path on and off. Measurements are in issue #80.

*The patch and this note were written by Claude (AI).*
