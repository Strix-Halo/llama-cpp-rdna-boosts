# MoE MMQ expert-table over-read — ARCHIVED (resolved 2026-10-03)

The host-resident-expert gather's claimed prefill "win" over staging was the corruption itself (NaN
routing skipped work); the gather default was switched back to the staging/host path, and the kernel-side
fix is the finite-head guard (the once-only expert-head zero no longer survives a reused `input_cpy`, so it
now runs on every gather).  The delivery behaviour is described in `patches/README.md` (blocks 06 / 13) and
`WORKLOG.md` 2026-10-03.

Records: [`HANDOVER.md`](HANDOVER.md) — the original investigation (the plan below its banner is superseded
and retained only for history); [`RESOLUTION.md`](RESOLUTION.md) — the conclusion.
