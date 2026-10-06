# OPEN 1 — session findings (r20 base, 2026-10-06)

**Status: the sibling (layer) re-size is implemented and validated; the MTP draft cap is now a default;
the DoD is met on `llama-cli`.  The extra-reserve investigation produced one blocking finding: the drop
is NOT server-safe at `-ub 8192`, and no extra-reserve value can make it so (item 4 is answered).**

Code candidate: `open1-candidate.diff` (applied in the `~/llama.cpp` scratch tree, build
`build-rocm-r16`).  Nothing is promoted.

Harness: `build-rocm-r16/bin/llama-cli` / `llama-server`, 2 GPU (`HIP_VISIBLE_DEVICES=0,1`), `-sm tensor
-ncmoe 48 -fa on -ctk q8_0 -ctv q8_0 -t 8 -c 32768 -b/-ub 8192`, `--spec-type draft-mtp
--spec-draft-n-max 3 --reasoning off -f /tmp/pl_16k.txt --seed 42 --temp 0 --no-display-prompt
--single-turn`, coherent (`////` = 0).  `-n 2000` for decode; server = 2 concurrent requests then a later
wide request.

---

## 1. Implemented (candidate diff)

* **`t.slots = got_slots`** in `alloc_table_locked` — the r20 slot-count retry shrank the arena but left
  `t.slots` at the requested count, so a shrunk table advertised more slots than its arena held (a device
  over-read: `////` + MTP acc 0.007).  Also fixed the `g_arena_bytes` accounting to the achieved count.
* **Layer-uniform re-size** (`alloc_all_locked`) — allocate each layer as a unit, largest expert slice
  first; if a table falls short, free the layer and retry at the achieved minimum (monotone decreasing).
  A small `free_table_buffers_locked` releases every device/pinned buffer a table owns.  A final
  non-uniform check stays as a WARN+disable safety net.
* **`MTP_DRAFT_N_UBATCH` default 512** (`common/speculative.cpp`); `=0` restores the target's `-ub`.
* **Drop default = llama-cli only (option B).**  `common_params::drop_compute_buffers` (default false)
  -> `llama_context_params::drop_compute_buffers` -> `llama_cparams`; `tools/cli/cli.cpp` sets it true,
  `llama-server` leaves it false.  `LLAMA_DROP_COMPUTE_BUFFERS` still overrides for A/B.  The drop's
  extra reserve now defaults to **0** (`LLAMA_DROP_EXTRA_RESERVE_MIB=N` raises it).

Validated (fully-default config, no env):

| | arena | resid. | prefill | decode avg |
|---|---:|---:|---:|---:|
| llama-cli (drop on by default) | 40078 | 70.6 % | 1683 | **78.7** |
| llama-cli `LLAMA_DROP_COMPUTE_BUFFERS=0` | 18848 | 33.2 % | 1711 | -- |
| llama-server (drop off by default), wide1->short->wide2 | 26110 | 46.0 % | -- | 0 aborts, all exit 0 |

The over-fill case that previously corrupted still degrades gracefully (layer 47 -> 3 uniform slots,
coherent, 72.2 t/s).

## 2. DoD on `llama-cli` (2 GPU, cache auto, 16k)

| config | arena | resid. | prefill | decode avg |
|---|---:|---:|---:|---:|
| default (margin on, drop off) | 18848 | 33.2 % | 1717 | 41.0 (earlier state) |
| drop + draft cap 512 + reserve 0 + extra 1024 | 40078 | 70.6 % | **1696** | **70.5** |
| drop + draft cap 512 + reserve 0 + extra 0 (re-size fires) | 42129 | 74.2 % | 1616 | **72.2** |

DoD (decode >= 68.9, prefill >= 1040) met; steady-state tg ~76-79.  `////` = 0 and MTP acc 0.92448 in
 every DoD run.

Run-to-run variance on the 2000-token average is ~±1.5 t/s (extra=1024 repeats: 70.5, 68.8), and the
arena moves with the box's free VRAM (40078-42129 MiB across the session).  **extra=0 is the stronger
choice now that the re-size exists**: it sizes the largest arena and still runs coherent at 72.2, the
re-size absorbing any fragmentation by degrading a single layer.

## 3. Item 4 — what the drop extra reserve buys (measured)

Two distinct roles, and they need very different amounts:

1. **Arena fragmentation slack.**  The auto budget is `free - reserve - extra` and the arena is allocated
   table-by-table, so filling 100 % of free VRAM makes a late, large table shrink.  Sweep (n=150, drop +
   draft cap + reserve 0):

   | extra MiB/device | arena MiB | resid. | re-size |
   |---:|---:|---:|---|
   | 0 | 42129 | 74.2 % | 1 layer -> 3 slots |
   | 256 | 41574 | 73.2 % | none |
   | 512 | 41076 | 72.4 % | none |
   | 768 | 40577 | 71.5 % | none |
   | 1024 | 40078 | 70.6 % | none |

   So ~256 MiB/device removes the shrink, and the new re-size makes anything below that a graceful
   per-layer degrade instead of corruption.  The arena slack the resize reports is a diagnostic; with
   the resize in place even extra=0 is coherent and *fast* (72.2).

2. **Compute growth headroom for a later wide prefill — this is where it fails.**  A later `-ub 8192`
   prefill needs a **contiguous 12939 MiB** compute layout back on device 0.  With `-np 4` and the drop
   on, the server stood down **all 288 arena tables (39992 MiB freed)** and the `cudaMalloc(12939 MiB)`
   *still failed* (`ggml_backend_meta.cpp:1813 GGML_ASSERT`), for the default extra reserve
   (`verify_bytes` ~1870 MiB), for extra=0, and for the concurrent-short variant.  The gallocr frees the
   old buffer before the new alloc (`ggml-alloc.c:1026-1027`), so this is a **contiguity** limit: freeing
   N arena blocks does not produce one ~12.6 GB contiguous block.  The smaller `-ub 4096` layout
   (~6.5 GB) *does* reclaim (server survives, 0 aborts), which brackets the limit.

   Reserving the wide layout out of the arena *does* make it safe — extra=13000 MiB/device: 0 aborts —
   but the arena collapses to 14801 MiB (26.1 %), **worse than the no-drop baseline** (31.0 %).  So the
   extra reserve cannot buy server-safety without giving up the DoD.

## 4. Server-safety verdict (the OPEN 1 acceptance clause)

| server config (`-ub 8192`, wide1 -> wide2) | result |
|---|---|
| no drop (r20) | clean, arena 31.0 % |
| drop + draft cap, extra default / 0 | **abort** (12939 MiB alloc fails after freeing the whole arena) |
| drop + extra 13000 MiB/device | clean, arena 26.1 % (no better than no drop) |
| **drop + `-np 1` + extra 0 (sequential)** | **abort** — 288 tables stood down (40892 MiB freed), `cudaMalloc(12409 MiB)` on device 0 still fails |

`llama-server` defaults to `-np 4`; the failure also reproduces sequentially (`SRV_SEQ=1`) and with
`-np 1` (the `-np 1` wide layout is 12409 MiB vs 12939 for `-np 4`).  So **`-np 1` does not make the
drop server-safe**: a second wide prefill aborts the same way.  The drop is a *single-wide-prefill*
win (cli / one-shot); any workload that can issue another wide prefill after the arena is sized is at
risk at `-ub 8192`.  (`-ub 4096`, a ~6.5 GB layout, reclaims fine -- the limit is the size of one
contiguous block, not the total freed.)

**Conclusion for OPEN 1:** the DoD is reachable for `llama-cli` (and any single-wide-prefill workload),
but "server-safe at `-ub 8192`" needs the compute buffer to be **chunked/VMM** (OPEN 2) — then a wide
prefill can be satisfied from uniform chunks the arena can yield.  Until then the drop cannot be
default-on for servers.

## 5. Open decisions

* Default-on the drop for `-np 1` / `llama-cli` only, keep it opt-in for a server?  Or hold the whole
  drop until OPEN 2?
* Extra reserve: with the re-size in place, the DoD is best at extra=0 (72.2); ~256-1024 MiB/device is the
  conservative arena choice.  Recommend a small value (or 0) and let the re-size absorb fragmentation.
* Promote the `t.slots` fix + the layer re-size as a standalone correctness fix (independent of OPEN 1)?
