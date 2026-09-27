# block 06 tiny-CPU-graph heuristic vs CPU-offloaded FFN — 2026-09-27 (r13, issue #52)

**Single RX 9700 / gfx1201** (`AMD Radeon AI PRO R9700`, 32 GiB), **Ryzen 9 9950X3D2** (16 cores,
SMT disabled in firmware), ROCm `7.14.1-gfx102X`, Release build from
`~/bin/build-llama-rocm-714` (`-DGGML_HIP=ON -DGPU_TARGETS=gfx1201 -DGGML_HIP_RCCL=1
-DGGML_HIP_GRAPHS=ON -DGGML_NATIVE=1`).  `HIP_VISIBLE_DEVICES=0`.  No other work on the GPU/CPU;
all runs sequential.

Model `Qwen3.8-27B-IQ4_NL.gguf` (16.23 GiB, 27.32 B), 39 FFN blocks overridden to the CPU so the
decode path is bounded by host weight reads:

```
llama-bench -m Qwen3.8-27B-IQ4_NL.gguf -p 32 -n 64 -r 1 -ngl 999 \
  -ot 'blk\.([0-9]|[1-2][0-9]|3[0-8])\.ffn_.*=CPU' -fa on -ctk q8_0 -ctv q8_0
```

## Result (tg64, t/s)

| build | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| r12 (shipped — output-byte heuristic) | 3.22 | — | — |
| **r13 (fix)** | **4.76** | **4.88** | **5.02** |
| r13 + `GGML_CPU_DISABLE_TINY_GRAPH_SINGLE_THREAD=1` (full threads, = pre-0028) | 4.87 | 4.95 | 4.82 |

The fix restores full-thread behavior within run-to-run noise; the absolute gap to the reporter's
8.66 t/s is this host's memory bandwidth (its DDR5 is populated for capacity, not speed), not the
fix.

## Mechanism

`ggml_backend_cpu_graph_n_threads()` (`ggml/src/ggml-cpu/ggml-cpu.cpp`, added by the folded
`beta/mmb-general` 0028) decided "tiny" from `sum(ggml_nbytes(node))` over the graph's **outputs**.
A CPU-offloaded FFN chunk is four `MUL_MAT` nodes with ~16 KiB activation outputs that each read
tens of MiB of weights, so it passed the `<= 32` node / `<= 16 MiB` bounds and ran on **one** thread
(`user` time in the buggy run is ~25 s vs ~200 s full-thread — the single thread spins on memory).
An instrumented build of the fix shows the only graphs still classified tiny on this workload are
`GET_ROWS`: 1 node, 20-164 KiB of gathered output (the embedding/logit tables, of which only the
gathered rows are read — exempting `src0` is what keeps the heuristic alive for the
host-resident-embedding spec-decode case it was written for).  The FFN chunks report `nodes=4`,
~50 MB of inputs and keep the full pool.

## Coherence

Greedy `llama-cli` on the same offloaded-FFN config (`--seed 42 --temp 0 --single-turn`):
byte-identical with the heuristic on and off — the only `diff` line is the printed timing.  Expected:
the ops whose thread count changes (`MUL_MAT`) are row-parallel, and the `GET_ROWS`/copy graphs keep
their single thread.

## Reproduction on the reporter's box

RX 9070 16 GB / Ryzen 9800X3D, Qwen3.8-27B IQ4_XS: 8.66 t/s (r12 without 0028 / r12 with the
kill-switch) vs 1.74 t/s (r12 with 0028).  `scripts/validate-set.sh` green on the r13 set (strict
16/16 `git am`, applied tree `b1a3bf1a…`).
