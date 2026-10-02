# wip/rdna4-fa-band-wide — one pass over the KV cache for 5..8-token verify batches

One `git am` patch for r21 (`fattn-mma-f16.cuh` + `fattn.cu`, +15/-1). Bit-exact.

## Why

The band folds GQA 6 into `ncols2 = 8` and runs `ncols1 = 4` (32 columns) per block. A 5..8-token
verify batch therefore launches two column tiles, and each streams the whole KV cache. With q8_0 KV
the band is ~70 % slower at `n_q = 5` than at `n_q = 4` (gfx1201, head 256, 4 KV heads, us per call):

| kv | n_q 1..4 | n_q 5..8 |
|---|---|---|
| 51200 | 287-294 | 476-484 |
| 98304 | 490-500 | 852-862 |

A DFlash2 / MTP verify at n-max 4 is `n_q = 5`, so this is the width it runs at on every step.

## The change

- **Row:** `ggml_cuda_fattn_mma_get_config_rdna()` gets a band row for `ncols == 64`:
  `fattn_mma_config(512, 1, 64, 64, 64, 32, 1, true)`. That is the 32-column band row with 16 warps
  instead of 8, so each Q column group still has `np = 4` warps. Every column gets the same KV-row
  split and the same combine order as in the 32-column block, so the result is bit-identical.
  `nbatch_combine` is 32 to keep the 16-warp combine buffer within 64 KiB of LDS. It sums the same
  partials in the same order, in narrower DV chunks.
- **Dispatch:** the band sends `n_q > 4` to the `<8, 8>` instance when the band runs 4-column tiles
  (the q8_0-style types) and the KV length is at least 4096. Below ~2k the 16-warp block is mostly
  idle and slower (kv 1024: +5..8 %). Both paths give the same result, so the gate cannot change an
  output. The 2-column f16/bf16 mode is untouched.
- **Kill switch:** `GGML_HIP_FA_BAND_WIDE=0` restores the two-tile launch.

## Results (gfx1201, R9700, ROCm 10.0, q8_0 KV)

`test-backend-ops perf`, us per call, `n_q` 5..8 (two-tile -> one block):

| kv | 4096 | 8192 | 16384 | 32768 | 51200 | 98304 |
|---|---|---|---|---|---|---|
| change | -11..-12 % | -15 % | -10 % | -11 % | -12 % | -13 % |

`n_q` 1..4 and f16 are unchanged.

Server, 27B UD-Q4_K_XL, q8_0 KV, DFlash2 n-max 4, greedy, 768 tokens, off/on/off/on (ms per verify
step, repeats within 0.1 ms):

| | doc 50k | code 97k |
|---|---|---|
| two tiles | 55.9 | 61.95 |
| one block | 55.1 (-1.4 %) | 60.3 (-2.6 %) |

All texts were byte-identical between the arms.

## Gates

- **`FLASH_ATTN_EXT`:** 6354/6354.
- **`test-logits-width-probe`:** PASS, with per-W and row0/row1 hashes identical on vs off, on 6
  runs: 27B UD-Q4_K_XL q8_0 (`RS=from_w`, `RS=0`), 27B f16, 27B IQ4_XS q8_0, 35B-A3B q8_0, and
  gemma-4-26B q8_0. These were run before the KV gate was added, so the wide path was exercised.

## Tried, didn't help

Two things I tried on top of the wide block did not pay off.

**Register-staged prefetch.** I fetched the next K/V batch's raw q8_0 bytes into registers during
the current batch's MMA. It was bit-exact but slower: kv 98304 `n_q = 1` went +7 %. The kernel is
already at 256 VGPRs, so the prefetch registers raised the 32-column block's scratch from 408 to
648 B per thread.

**LDS DMA.** A register-free copy (`__builtin_amdgcn_global_load_lds`) is not available on gfx1201.

The `GGML_HIP_FA_BAND_WMMA_SPLIT` values 48, 64 and 96 were all slower than the default of 32, with
the wide block too.
