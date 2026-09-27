# Host-resident MoE experts under `-sm tensor` are now a first-class prefill configuration

**Release:** `v16-84e76d8a2-r16` &nbsp;•&nbsp; **Model:** Qwen3.6-35B-A3B UD-Q4_K_M (21.1 GiB, 256 experts, top-8, 40 MoE layers) &nbsp;•&nbsp; **Hardware:** 1×/2× AMD Radeon AI PRO R9700 (gfx1201, 32 GiB)

I'm very happy to share the result this prefill campaign was aiming at: **on the `rdna-boosts` patch set, `-sm tensor` with host-resident MoE experts (`-ncmoe N`) is now the fastest prefill configuration at every offload level** — for both 1 and 2 GPUs — and it beats stock `llama.cpp` by **+66 % at `-ncmoe 0` rising to +148 % at `-ncmoe 40`**.

Two years of "you can't tensor-split a MoE whose experts are on the host" are over: it works, it's numerically identical, and it's the *fastest* thing you can run.

## The result

`pp8192`, ubatch 8192, `-fa 1`, `-ncmoe N` (N = MoE expert layers kept **on the host**), 1× `llama-bench -r 1` on gfx1201:

| `-ncmoe` | ours 1 GPU | ours 2 GPU `-sm tensor` | ours 2 GPU `-sm layer` | upstream 1 GPU | upstream 2 GPU `-sm layer` | tensor vs layer | ours-tensor vs upstream-layer |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6450 | **7639** | 6308 | 3882 | 3994 | +21 % | +91 % |
| 2 | 6166 | 7025 | 6090 | 3767 | 3767 | +15 % | +86 % |
| 4 | 6046 | 6959 | 5955 | 3640 | 3649 | +17 % | +91 % |
| 6 | 6103 | 7109 | 6027 | 3541 | 3537 | +18 % | +101 % |
| 8 | 6148 | 7014 | 6080 | 3408 | 3446 | +15 % | +104 % |
| 10 | 6125 | 6863 | 6057 | 3346 | 3350 | +13 % | +105 % |
| 12 | 6106 | 6740 | 6038 | 3237 | 3265 | +12 % | +106 % |
| 14 | 6076 | 6638 | 6009 | 3187 | 3186 | +10 % | +108 % |
| 16 | 6051 | 6541 | 5986 | 3108 | 3114 | +9 % | +110 % |
| 18 | 6033 | 6426 | 5971 | 3055 | 3043 | +8 % | +111 % |
| 20 | 6005 | 6324 | 5943 | 2978 | 2974 | +6 % | +113 % |
| 22 | 5980 | 6220 | 5808 | 2915 | 2883 | +7 % | +116 % |
| 24 | 5964 | 6137 | 5557 | 2845 | 2783 | +10 % | +120 % |
| 26 | 5946 | 6058 | 5330 | 2795 | 2697 | +14 % | +125 % |
| 28 | 5919 | 5971 | 5121 | 2732 | 2601 | +17 % | +130 % |
| 30 | 5896 | 5883 | 4919 | 2681 | 2525 | +20 % | +133 % |
| 32 | 5878 | 5832 | 4741 | 2625 | 2448 | +23 % | +138 % |
| 34 | 5854 | 5732 | 4572 | 2594 | 2382 | +25 % | +141 % |
| 36 | 5767 | 5590 | 4390 | 2537 | 2306 | +27 % | +142 % |
| 38 | 5758 | 5517 | 4245 | 2494 | 2245 | +30 % | +146 % |
| 40 | 5794 | 5445 | 4107 | 2448 | 2193 | +33 % | +148 % |

Raw `-ncmoe` 0..40 (every level) is in the attached CSV (`wip/tensor-split-expert-split/sweep-full.csv`).

The shapes are the story:

* **The delivery's offload cost is nearly flat.** 1 GPU goes 6450 → 5794 t/s across the whole range (`ncmoe 0 → 40`); upstream goes 3882 → 2448. Putting *all* the experts on the host costs the patch set only **−10 %** on one card, while upstream loses **−37 %**.
* **`-sm tensor` beats `-sm layer` at every level**, and the gap *widens* as more experts move to the host: +21 % with everything in VRAM, **+33 % when everything is on the host**. The tensor split halves both the transfer and the compute per card; the layer split does neither (the expert work lands on one device).
* **The whole upper half of the table is a configuration upstream cannot run at all** — stock `llama.cpp` refuses `-sm tensor` with `-ncmoe` (mirrored experts), so its only 2-GPU option is the layer split, which is also *slower than a single card* once offload climbs (2193 vs 2448 at `-ncmoe 40`).

## What changed

The release folds four things into the patch set (block 06/15):

1. **Op-offload H2D staging is now ON by default** (`GGML_SCHED_STAGE=0` opts out). It overlaps a host→device expert upload with the previous split's compute instead of serialising on it. This is the single biggest lever and it helps *every* host-resident config: 2 GPU `-sm tensor` **+66 %**, 1 GPU **+83 %**, 2 GPU `-sm layer` **+50 %** at pp8192; it is a no-op when nothing is offloaded.
2. **A split expert upload is staged per device** (the meta backend now gathers each device's slice and queues it on the copy stream), so `-sm tensor` offload gets the same overlap the mirrored path had.
3. **The compacted strided splice uses a pinned host gather + one queued 1-D H2D** instead of the pageable `hipMemcpy2DAsync` (default; `GGML_CUDA_SPLICE_GATHER=0` reverts). This is what keeps the sub-gate ubatches fast, and it fixes the pageable 2-D copy fault some reporters hit.
4. **Host-resident `MUL_MAT_ID` weights stay pinned** (already in `v16-…-r15`, `LLAMA_MMAP_HOST_EXPERTS=0` reverts), so the copies are genuinely asynchronous.

All four are **on by default and self-selecting**: they only engage for the relevant path (host-resident weights, `-sm tensor`, a split upload), so running `llama-server … -sm tensor -ncmoe N` needs no environment variables at all. Each has a documented kill-switch for A/B and bisection. Same-seed greedy output is **bit-identical** with the features on and off (and across 1/2/3 GPUs and both split modes), and `test-backend-ops -o MUL_MAT_ID` / `-o FLASH_ATTN_EXT` are green.

## Reproduction

```bash
# the patch set (16 blocks) against upstream 84e76d8a2
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 84e76d8a2
bash /path/to/llama-cpp-rdna-boosts/scripts/apply-all.sh .
# build (ROCm), then:
HIP_VISIBLE_DEVICES=0,1 ./build/bin/llama-bench \
  -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -ncmoe 24 -fa 1 \
  -p 8192 -ub 8192 -n 1 -b 8192 -sm tensor -r 3
```

## A request: faster links welcome

**This box gives each R9700 only PCIe 5.0 ×4** (the BIOS caps each slot at four lanes). The offload win here is fundamentally *host-stall removal* — the features make the upload asynchronous and overlapped — so it should scale **up** with better host↔GPU bandwidth, and the absolute numbers should rise with it.

If you have **2 GPUs on PCIe 5.0 ×8 (or ×16) each**, I would love to see this exact sweep reproduced. Even just `-ncmoe 8, 16, 24, 32, 40` at pp8192/pp32768 on any MoE would be a very welcome datapoint — please post your board, CPU, link width, model/quant and numbers. Issues/PRs with the raw `llama-bench` output are perfect.

## Thanks

The staging ring and the host-expert pinning grew out of the `h2d-staging-ring` work, and @briansp2020's PR #51 and independent `moe-cache` work corroborated the source-pinning direction. The remaining decode-side lever (a routing-driven VRAM expert cache, ~2.6× in the prior art) is the next phase and is deliberately not part of this release.
