# MOVED — this campaign is archived and its premise was refuted

`wip/expert-cache-split/` was **closed on 2026-10-07 and moved to
[`archive/work/expert-cache-split/`](../../archive/work/expert-cache-split/README.md)**.

`TODO.md` #44 claimed that `-sm tensor` + host experts **mirrors** the expert weights, so that splitting
them would double cache residency and halve the host copy.  **Measurement refuted that premise:** the
expert weights are already split per device (each cache table holds `expert_bytes = host_bytes / 2`), and
the host master is a single buffer.  No code was changed.  The evidence — the field-log arithmetic
(`arena 39424.2 MiB of 64800.0 MiB host experts`, where 64800 is exactly the model's full expert set) and
an instrumented per-table geometry dump — is in the archived README.

This stub exists so the historical pointers to `wip/expert-cache-split/...` — including the comments in
`TODO.md` #44 and the `wip/moe-cache-autosize/README.md` stub — still resolve.
