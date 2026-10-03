# llama-cpp-rdna-boosts

A patch collection that brings **AMD RDNA-specific performance work** to
[llama.cpp](https://github.com/ggml-org/llama.cpp): MTP decode, chunked
gated-delta-net prefill, BF16 KV and WMMA flash-attention, fused MoE and
k-quant decode paths, a hybrid all-reduce, qwen4exp (Qwen3.8-Flash-Next)
support, and an attention-memory campaign that frees several GiB of VRAM.

It ships as **16 patches** (block 00 + blocks 01-15) for a clean llama.cpp
checkout at the fork point **`a55e952b8`** (upstream master, 2026-10-03
re-base).  Each block is a self-contained `git am` commit, so you can apply
the whole set or pick the ones you want.  The **`mmb` (bf16-WMMA weight GEMM) / QSA / indexer
campaign**, formerly the 28-patch opt-in `archive/work/mmb-general/` set, is now **folded into the delivery
blocks** — the `mmb` core into block 08, the catch-all system-operations fixes into block 06, and the
qwen4exp/QSA/HC/indexer work into block 15 — so the **16 patches alone reproduce the full campaign
tree**.  `archive/work/mmb-general/` is retained only as the historical verification record;
see [The `mmb` campaign is in the delivery](#the-mmb-campaign-is-in-the-delivery).  The current release is
**`v16-a55e952b8-r6`** — the r1 re-base onto upstream `a55e952b8` (203 commits), the `r2`-`r5`
follow-up folds, and the issue-#95 dynamic-backend `-sm tensor` fix — see [Current state](#current-state).

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout a55e952b8
bash <path-to-this-repo>/scripts/apply-all.sh .   # creates branch rdna-boosts
```

- One-line summary of each block: [The 16 blocks](#the-16-blocks)
- The folded `mmb`/QSA campaign: [The `mmb` campaign is in the delivery](#the-mmb-campaign-is-in-the-delivery)
- Apply details, env knobs, server config: [`patches/README.md`](patches/README.md)
- **Running a model bigger than your VRAM on one card** (`-ncmoe` + the MoE expert cache, with the
  measured config tables and the measurement method): [`COMMUNITY-CONFIG.md`](COMMUNITY-CONFIG.md)
- What changed recently: [`WORKLOG.md`](WORKLOG.md)
- Current status and validation: [Current state](#current-state)

> **The `mmb`/QSA/indexer campaign is folded into the delivery (2026-09-25, release `r8`).**  The 16
> delivery patches now absorb the 28 `archive/work/mmb-general/` patches; the applied tree is `24bb0f5acb…`
> and the build is clean on gfx1201.  The working plan, per-patch mapping and validation record are
> in [`archive/work/beta-integration/integration.md`](archive/work/beta-integration/integration.md).

## Releases

Frozen deliveries are published as GitHub Releases and tagged in this repo
(the tag is the release identity: `v16-<fork-point>-r<N>`, e.g.
**`v16-84e76d8a2-r7`**, where `r1` is the re-base, `r2` the block-10 MoE-VDR arch-scope fix, `r3` the
block-14 `hc_combine` CPU-reference fix (issue #44) + the beta re-base, `r4` the block-15 RDNA4
GQA-6 decode/verify flash-attention band (issue #45), `r5` the block-15 f16/bf16 band coverage
(issue #45 follow-up), `r6` the block-15 bf16 native default flip, `r7` the block-14 Meta-tensor-split
scheduler race fix, `r8` the **`archive/work/mmb-general` fold into the 16 blocks** (the campaign is now
part of the delivery, no separate beta apply step), `r9` the block-15 typed non-swizzled K/V store
fix for the MMA FA prefill loader (issue #47), `r10` the block-15 fully-masked KV-group skip that
stops a `--kv-unified` concurrent prefill from paying for the other slots' cells (issue #48), `r11` the
block-13 MoE MMVQ `rpb` mis-launch fix behind the `MUL_MAT_ID` backend-ops failure, `r12` the block-06
op-offload **H2D staging ring** (issue #50, merging PR #51 by @briansp2020) with the `-sm tensor`
op-offload fix it needed, plus the block-06 rename to `general system-operations bucket`, `r13` the
block-06 tiny-CPU-graph heuristic fix that counts the tensors a node reads, so a CPU-offloaded FFN chunk
is no longer serialized on one thread (issue #52), `r14` the block-15 fix that computes the derived
kq-mask window on the device instead of reading the device copy of `tok_lo`/`tok_hi` from the host (the
Windows `0xC0000005` on the first prefill ubatch, issue #53), `r15` the block-06 fix that keeps
host-resident MoE expert weights (`MUL_MAT_ID`) pinned instead of downgrading them to the pageable mmap,
`r16`-`r28` the amendments recorded in [`WORKLOG.md`](WORKLOG.md) (the host-resident-expert prefill fast
path, the r19 offloaded-MoE thread cap, the r20-r23 verify-band wins, the r24/r25 rope-fusion fixes, the
r26 staging-ring overlap fix, the r27 contributor-PR collection and the r28 VMM pool free-order fix), and
**`r29` the decode-side MoE expert cache** (`MOE_EXPERT_CACHE_MIB`, opt-in; see
[`COMMUNITY-CONFIG.md`](COMMUNITY-CONFIG.md)), **`r30` the block-13 gather head-pad fix**, **`r31` the
two MMQ `MUL_MAT_ID` tail over-read fixes**, **`r32` the four contributor PRs folded into block 15**, **`r33`
the default-off block-06 async graph-input race candidate (issue #87)**, **`r34` the issues #59/#60
qwen4exp fix folded into block 15**, **`r35` the block-06 issue-#87 async graph-input fix flipped
default-on** and **`r36` the block-15 issue-#89 indexer top-k block-path fix (PR #90)**, **`r37` the
block-15 BF16 hyper-connection mixer fusion (PR #91)**, and on the `a55e952b8` base **`r1` the 203-commit
re-base**, **`r2` the merge-hygiene + integration-audit fold**, **`r3` the `mmq-prec-gate-fp4` +
`shared-expert-fusion-reconcile` fold**, **`r4` the `qwen4exp-qsa-convergence` +
`lightning-indexer-fusion` fold**, **`r5` the block-06 scheduler re-stage fix (PR #96)** and **`r6`
the issue-#95 `GGML_USE_HIP` propagation fix for dynamic-backend builds**;
each later release on the same base
increments `N`).  `release.json.release` must equal the tag — CI
checks it — and only a tag push cuts a release.  Each release carries
`rdna-boosts-all.patch`, `patches.tar.gz`, `release.json`
and `SHA256SUMS`, so a consumer can pin a tag and verify the artifacts instead
of tracking a moving `main`.

`release.json` is the delivery's single source of truth (fork point, canonical
tip/tree, block count, per-artifact sha256); `scripts/apply-all.sh`,
`scripts/validate-set.sh` and CI all read it.  The container pipeline is
**tag-driven**, so ordinary commits to `main` (docs / benchmarks / `WORKLOG.md`)
only run the cheap patch validation — see [`CONTAINERS.md`](CONTAINERS.md) for
the release process and the prebuilt ROCm images.

## Supported architectures

The set targets the **RDNA3 / RDNA3.5 / RDNA4** GPU families:

| family | arches | example parts |
|--------|--------|---------------|
| RDNA 3 | `gfx1100` | RX 7900 XTX/XT, RX 7800 XT, ... |
| RDNA 3.5 | `gfx1150`/`gfx1151` | Strix Point / Strix Halo APUs |
| RDNA 4 | `gfx1200`/`gfx1201` | RX 9060 XT; RX 9070 / 9070 XT |

> **The prebuilt images do not include `gfx1150` (Strix Point).**  The GHCR
> containers carry code objects for `gfx1100;gfx1151;gfx1200;gfx1201` only, so
> a Strix Point iGPU (`gfx1150`, e.g. Ryzen AI 9 HX 370 / Radeon 890M) fails
> on the first kernel launch (`ROCm error: device kernel image is invalid`).
> Build from source with `gfx1150` in the target list, or set
> `HSA_OVERRIDE_GFX_VERSION=11.5.1` at runtime to select the `gfx1151` code
> objects.  See [Strix Point (gfx1150) and prebuilt
> images](CONTAINERS.md#strix-point-gfx1150-and-prebuilt-images) for both paths.

**RDNA4 (gfx120x) sees the most benefit** — the WMMA flash-attn path, the
chunked-GDN kernel, the k-quant VDR boosts and block 12's internal
all-reduce were all first built and validated there. As much of that work
as possible is back-ported to the RDNA3/3.5 families instead of being
gated off:

- block 02's **chunked gated-delta-net** bf16/WMMA prefill ships as two
  arch-segregated kernels: a dedicated first-gen WMMA port for gfx11
  (`gated_delta_net_chunked_bf16_gfx11.cu`) next to the RDNA4 kernel;
- block 04's **WMMA flash-attn** is *not* RDNA4-only despite the block
  name — RDNA3.0 runs it with the same 576-head limit as RDNA4, RDNA3.5
  with a tuned 320-head limit;
- block 10 adds a **dedicated RDNA3.5 mmvq parameter table** (previously
  folded into the RDNA2 fallback) on top of the RDNA4 k-quant boosts.

Arch selection is **runtime** everywhere in the set (device `cc` /
`gcnArchName`; there is no compile-time arch gating), so a multi-arch
build such as `GPU_TARGETS="gfx1100;gfx1151;gfx1201"` yields one binary
that picks the right path on whichever of these it runs on. The one
genuine exception is **block 12** — its internal all-reduce is RDNA4-only
(gfx1200/gfx1201) and falls back to RCCL elsewhere (see
`patches/README.md` for the gate and env knobs).  Block 13's fused
MoE MMQ gate now covers RDNA4 + RDNA3_5 + RDNA3_0 (gfx1151 validated
2026-09-05, gfx1100 validated 2026-09-05 — see
[Current state](#current-state)).

## Layout

```
├── README.md              # this file: overview + consumer workflow
├── AGENTS.md              # working guide for LLM agents in this repo
├── MANIFESTS.md           # apply order, per-block verification, validation history
├── BASELINE.md            # fork point, patch provenance, drift policy
├── GREEDY-PURITY.md       # purity rulebook: index, invariants, per-finding claims (read before shipping)
│                          #   narratives/evidence for the closed cases: archive/docs/GREEDY-PURITY-FINDINGS.md
├── WORKLOG.md             # dated delivery records (newest first; README points here)
├── rdna-boosts-all.patch  # convenience: the entire 16-patch net as ONE patch
├── patches/               # the delivery set: 0000-0015
│   └── README.md          # apply instructions + block-12 env knobs + server config
├── scripts/
│   ├── apply-all.sh       # the verified apply flow (git am; automatic -3 fallback on drift)
│   └── make-patches.sh    # regenerates the set from the fork (~/llama.cpp)
├── benchmarks/            # benchy methodology + v1/v2 results + graphs (dated records)
├── prompts/               # versioned, hash-stable test prompts (sha256-recorded; never edited in place)
├── wiki/                  # source for the GitHub wiki (Home, MTP & Adaptive MTP, Quick Reference); see wiki/README.md
├── wip/                   # ACTIVE exploration docs / handoffs (currently only: nwarps/)
├── upstream/              # upstream-PR candidates (UPSTREAM-PR-*.md + .patch) + their index
└── archive/               # the rest: archive/work/ (closed experiments + the archived wip/ trees) + archive/docs/ (history)
```

> **History:** the `baseline/<sha>` branches, `block/01-…11` tags, and all
> dated validation records belong to the old pre-block-12 structure and live
> in `archive/docs/` (see also `archive/work/` for the closed experiments).
> Do not mix them with the current `patches/` files.

## The 16 blocks

| patch | what |
|-------|------|
| `0000` | **structural and architecture fixes** — FA small-batch KV-split width invariance (issue #25) + Vulkan masked-V/freed-cell fixes (dead columns never read V). The base every later block applies on top of. |
| `0001` | adaptive MTP draft depth (`--draft-mtp-adaptive`) |
| `0002` | fused chunked gated-delta-net prefill kernel (bf16/WMMA, arch-segregated gfx12/gfx11) |
| `0003` | BF16 KV cache + native-BF16 flash-attn (+ the HIP masked-V/freed-cell fixes since 2026-09-10) |
| `0004` | RDNA4 WMMA flash-attn + Q6_K mmq prefill perf (WMMA path also runs on RDNA3.0/3.5, tuned head limits) |
| `0005` | CPU bit-identical decode/verify batches |
| `0006` | **general system-operations bucket** — the delivery's **catch-all** for changes that fit no other block: the FA instance build-time work, `--fit` under `-sm tensor`, the host-buffer input layer, the tiny-CPU-split single-thread fix and (r12) the op-offload **H2D staging ring** + tensor-split op-offload.  Named for what it is since r12; the original host-buffer revert content is long gone (upstream reverted #24233 in #28604).  Amended r13 (issue #52): the tiny-CPU-split heuristic now counts what a graph *reads* — not only its node outputs — so a CPU-offloaded FFN chunk keeps the full thread pool |
| `0007` | meta device-wrapper skip |
| `0008` | fused-core prefill kernels + GPU bit-identical results (needs blocks 03+04; amended 2026-09-07 with the mul_mat+add through-view shape guard, PR #15). **Now folds the `mmb` (bf16-WMMA dequant weight GEMM) core, the RDNA4 fragment port / per-arch tuning, and the GDN/PLE conv1d + narrow-row RMS-norm prefill fusions** (absorbed from the former `archive/work/mmb-general` campaign). |
| `0009` | meta-buffer compute-container headroom |
| `0010` | k-quant-boosts: Q4_K/Q5_K/Q6_K/Q8_0 mmvq VDR (+ q8_1 quantize-cache fusions; adds a dedicated RDNA3.5 mmvq table) |
| `0011` | skip CUDA graphs for multi-token PRE-FILL (decode keeps graph replay) |
| `0012` | **hybrid HIP all-reduce** — custom internal AR for the small-tensor decode path, per-size hybrid dispatch vs RCCL, RDNA4-only gate (bounded in-kernel spin since 2026-08-30 fix round; builds without RCCL) |
| `0013` | **fused MoE gate+up+GLU MMQ + mmvq short-K item-split** — prefill fused expert MMQ (RDNA4 + RDNA3_5 + RDNA3_0, Q3_K/Q4_K/Q5_K/Q8_0/Q6_K, env opt-out `GGML_CUDA_DISABLE_MOE_MMQ_FUSION`) + decode item-split (rpb 2/4/8) merged with the upstream has_fusion mmvq path |
| `0014` | **qwen4exp / Qwen3.8-Flash-Next support** — QSA sparse FA (default) + fused indexer top-k, HC_MIX/HC_COMBINE fused decode ops, managed lazy reader, MTP draft-head, WS4 hyperconn prefill fusions, QSA decode campaign + per-arch dense/QSA decode policy (promoted from `beta/qwen4exp`; see `patches/README.md` block-14 notes). The masked-V/freed-cell fixes it once carried now live in blocks 00 (Vulkan) and 03 (HIP). |
| `0015` | **attention-memory wins (block 15)** — promoted 2026-09-12 from `archive/work/block-15-campaign-wins/`: **V3** derived kq mask (`LLAMA_KQ_MASK_DERIVED`, on by default), **V4** native q8_0 + **V5** native bf16 K/V in the FA kernels (both behind `GGML_CUDA_FA_KV_NATIVE`, opt-in default 0), **W1** QSA score-chain memory (`GGML_QSA_SCORE_MEM`), **W2** derived QSA per-block bias + visibility (`GGML_QSA_DERIVED_BIAS`/`GGML_QSA_DERIVED_VIS`), **W3** keys-only QSA indexer cache (`LLAMA_QSA_KEYS_ONLY`), **W4** ggml-alloc unused-view release (no gate; A/B revert in `archive/work/block-15-campaign-wins/ab/`).  ~3.4 GiB/GPU + ~1.2 GiB host saved on qwen4exp, ~800 MiB/GPU + ~800 MiB host on dense models, at ~1.3 % prefill / ~0.3 % decode. **Now also folds the qwen4exp/QSA/HC/indexer campaign** (`qsa3` packed-block WMMA attention, fused indexer top-k + prefill score fusions, HC16 native-BF16 producers, `hc_gate_mix`, sparse MTP-draft attention, the sparse-QSA/derived-indexer defaults, and the host-buffer/CPU/meta fixes) — the former `archive/work/mmb-general` work. |

> **Block 15 (attention-memory wins) is part of the delivery since
> 2026-09-12** (`patches/0015`, promoted from
> `archive/work/block-15-campaign-wins/`; a fresh set is now **16 patches**,
> blocks 00-15).

> **Greedy-purity note (read before shipping):** on the K-split decode
> paths, block 10 (`0010`) is the only patch that changes decode numerics on
> ANY architecture — its VDR kernels reorder the fp32 reduction. Compute
> outputs are not bit-identical to a build without it (max logit diff 0.184
> vs 0.203 for flash-attn on/off; greedy streams are deterministic within a
> build but can flip across configs). This is a different rounding path, not
> a correctness change. If you require 100% greedy purity across builds, do
> not install `0010-…k-quant-boosts…patch` — it is one line to drop from
> `scripts/apply-all.sh`. Full discussion:
> [`GREEDY-PURITY.md`](GREEDY-PURITY.md). **Block-13 caveat (2026-09-02):**
> block 13 rewrites the small-batch mmvq decode kernel and is a second
> decode-numerics source on the rows that run it (short-K K<4096 ncols==1
> rows, MoE projections; ncols 2..8 and long-K rows were restored to the
> pre-block-13 K-split kernel by the 2026-09-02 fix). Excluding block 10 no
> longer reproduces stock bits exactly on those rows — see
> GREEDY-PURITY.md §9.

## Consumer workflow

```
# 1. fresh clone of llama.cpp, at the fork point recorded in release.json
BASE=$(jq -r .base release.json)          # from this repo
FORK=https://github.com/ggml-org/llama.cpp
git clone $FORK && cd llama.cpp
git checkout "$BASE"

# 2. apply the set (automated; strict 16/16 git am on the recorded base)
bash <path-to-this-repo>/scripts/apply-all.sh .
#    = git am patches/0000…0015  (one commit per block on a fresh `rdna-boosts` branch)

# 3. build + verify (trim -DGPU_TARGETS to your GPU arch for a faster build)
cmake -B build -DGGML_HIP=ON -DGGML_HIP_RCCL=1 -DGPU_TARGETS="gfx1100;gfx1151;gfx1201" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
# coherence gate (same-seed output must match a known-good build):
./build/bin/llama-cli -m <model> -ngl 99 -sm tensor -mg 0 -p "The capital of France is" \
  -n 20 --seed 42 --temp 0 --no-display-prompt --single-turn
```

> **Speed up rebuilds with ccache.**  The set's flash-attention template instances are the build's
> critical path, and their native-KV loader arms are deliberately force-inlined — the optimiser's
> cross-inlining is what makes them fast at runtime *and* slow to compile.  With `ccache` on PATH,
> a wiped rebuild of *unchanged* sources is a full cache hit: measured **282 s -> 4.2 s** on a
> 16-core gfx1201 box, **321.8 -> 5.2 s** on gfx1151 and **383.9 -> 4.8 s** on gfx1100 (657/657
> compile steps hit on each).  Add
> `-DCMAKE_HIP_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache`
> (the launcher form works with ROCm clang HIP device compilation; ccache 4.12.3 tested, on CMake
> 4.3).  ccache
> replays the compiler's own objects, so the cached build is the same code — verified with
> same-seed greedy text (identical hash on every host before and after enabling it),
> `llama-bench` (within noise) and `test-backend-ops`.  Any header change (e.g. `fattn-mma-f16.cuh`)
> invalidates its dependents, i.e. the whole FA group.
>
> **Do not use `git apply` on the concatenated 1-11 series** — it silently
> drops hunks (30 files / 2483 lines vs the correct 35 / 6094, verified
> 2026-08-29). `git am` (or `scripts/apply-all.sh`) is the required flow.

### Manual equivalent

```bash
git am patches/000[1-9]-*.patch patches/001[0-5]-*.patch   # blocks 01-15
git add -A && git commit -m "rdna-boosts: block 15: campaign memory wins"
```

### The `mmb` campaign is in the delivery

On the **`beta-integration`** branch the `mmb` (bf16-WMMA dequant weight GEMM) / QSA / indexer
campaign — the former 28-patch `archive/work/mmb-general/` set — is **folded directly into the 16 delivery
blocks**, so the normal workflow above is all there is to apply.  There is **no separate beta layer
any more**:

* **block 08** absorbs the `mmb` core, the RDNA4 fragment port and per-arch tuning, and the
  GDN/PLE conv1d + narrow-row RMS-norm prefill fusions;
* **block 06** (the catch-all system-operations bucket) absorbs the host-buffer input layer and the tiny-CPU-split
  single-thread fix;
* **blocks 13/14** absorb the `mmb` fusion stand-downs and the extended MMVQ routed band;
* **block 15** absorbs `qsa3`, the fused indexer top-k + prefill score fusions, HC16, `hc_gate_mix`,
  sparse MTP-draft attention, the sparse-QSA/derived-indexer defaults, and the meta/CPU backend fixes.

Applying the 16 patches to `84e76d8a2` therefore reproduces the **full campaign tree** (r8's
`24bb0f5acb3e866abd4cad8c0de1bad45a20cb47`, plus r9's issue-#47 MMA-FA typed-store fix -> current
`a3dc4bbb680bf9dd8bcb5949ec833dec2a892aeb`) in one pass:

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 84e76d8a2
bash <path-to-this-repo>/scripts/apply-all.sh .   # 16/16 strict, tree a3dc4bbb…
```

The per-patch fold mapping and validation record are in
[`archive/work/beta-integration/integration.md`](archive/work/beta-integration/integration.md).  Much of the MMB
kernel work is heavily adapted from **[pwilkin](https://github.com/pwilkin)**'s
[`strix-halo` fork](https://github.com/pwilkin/llama.cpp/commits/strix-halo/), with thanks.

> **Historical record only.**  [`archive/work/mmb-general/`](archive/work/mmb-general/) (the 28 patch files, the
> `mmb-general.patch`, `BETA-TESTING.md`, the gfx1201/gfx1100 records) is kept as the campaign's
> verification record; its patches are **no longer applied separately** and the `apply-beta.sh`
> helper has been **removed** (the delivery itself now contains the campaign).  Its gfx1151
> beta-window re-validation was **GREEN**
> (2026-09-25 — the four gates + the recurrent rollback; see `BETA-TESTING.md` §8), which is what
> the fold relies on.

## Recommended configuration — adaptive MTP + `ngram-mod`

The best general-purpose speculative-decoding configuration measured on this delivery combines the
**adaptive MTP controller** with the draftless **`ngram-mod`** speculator:

```bash
--spec-type draft-mtp-adaptive,ngram-mod \
  --spec-ngram-mod-n-match 45 \
  --spec-draft-n-max 9 --spec-draft-n-start 9
```

`ngram-mod` supplies the long verbatim-recall drafts the MTP head cannot match, while the adaptive
controller keeps the MTP depth right for everything else.  Measured against plain `draft-mtp-adaptive`
on the Q8_0 2-GPU reference cell (`-n 3000`, a 4-prompts-per-axis corpus): **recall +67.5 %,
code +0.8 %, prose +0.5 %, reasoning −1.9 %, overall +13.6 %**; on the dense 1-card cell it is
code/prose-neutral with the same recall win.  The small reasoning cost is the price of the deeper
`n-max 9`; a workload with no verbatim recall is marginally better served by plain
`draft-mtp-adaptive`.

Guidance:

* **Cap.**  `9` is the general-purpose pick and the optimum on a **single card** (a cap of `6` costs
  11-14 % on code there).  On a **multi-card tensor split** `6-7` is ~1 % better.  A lower cap saves
  only a small verify-batch scratch, not the model/KV memory.
* **`n_match`.**  Keep it `>= 40` **and an integer multiple of the cap** — `9/45`, `8/48`, `6/42`.
  Too short (`nm24`) makes ngram fire on incidental code repeats and lose code throughput; a
  non-multiple (e.g. `nm42` at cap 12) degrades acceptance.
* The combo changes the draft strategy, so it is **opt-in** — the controller default stays plain
  `--spec-type draft-mtp-adaptive`.

Full derivation (the 4×4 corpus, all four cells, and the rejected controller alternatives):
**[`archive/work/mtp-journey-2026-09-17/SUMMARY.md`](archive/work/mtp-journey-2026-09-17/SUMMARY.md)** (narrative in its
[`README.md`](archive/work/mtp-journey-2026-09-17/README.md)); the dated controller records are in
[`benchmarks/`](benchmarks/README.md), newest `2026-09-15-adaptive-mtp-tuning.md`.

## VRAM vs prefill — the derived KQ mask (`LLAMA_KQ_MASK_DERIVED`)

**On by default, deliberately.**  The delivery derives the attention mask inside the FA kernel from
compact per-cell state instead of materialising the `n_kv x n_q` f16 mask.  That removes
`n_ubatch x n_ctx x 2` bytes of compute-buffer VRAM **plus the same again on the host** — measured on
a 9B at `-c 98304` / ub 512: **184.0 -> 88.4 MiB** device and **112.0 -> 16.4 MiB** host.  The saving
scales linearly with the ubatch, which is the point: a deep-context **MoE** or **qwen4exp /
Qwen3.8-Flash-Next** workload wants a large ubatch, and that is exactly the configuration where the
mask is biggest (~800 MiB/GPU at ub 2048 / 196k) and where the VRAM the feature frees is the
difference between fitting the context and not.

The cost is **prefill only** — decode is untouched, because the derived path only fires for batches
larger than 8 tokens (speculative verify keeps the packed mask, so `n_max <= 7` stays bit-identical).
Measured PP512, mask on vs off, `-r 3` ("+" = the mask helps):

| config | d0 | 32k | 64k | 98k |
|---|---|---|---|---|
| gfx1201 9B dense 1 GPU | — | +1.9 % | — | **+3.5 %** |
| gfx1201 27B 2 GPU **tensor** | −1.3 % | −0.4 % | **+1.3 %** | **+1.7 %** |
| gfx1201 27B 2 GPU layer | — | — | — | −1.6 % |
| gfx1201 27B 3 GPU tensor | −3.4 % | −0.6 % | — | **+2.0 %** |
| gfx1151 9B dense 1 GPU | +0.4 % | −0.2 % | −0.9 % | −1.8 % |
| gfx1151 35B-A3B MoE 1 GPU | −0.3 % | −0.3 % | −0.8 % | −1.6 % |
| gfx1100 9B dense 1 GPU | −0.4 % | −0.5 % | −0.4 % | **−0.2 %** |

The tensor-split shape is the one to understand: the packed mask grows with `n_kv`, so on a
**tensor split at shallow depth the mask is a small loss (−1.3 % at d0, crossing zero near 48k) and
becomes a win by 64k+**; on the maintainer's 3-GPU tensor serving setup it is a win at depth.  gfx1100
and gfx1151 pay a depth-growing ~1–2 % (they did not recover as much from the r7 kernel fix as
gfx1201 — the iGPU shares host bandwidth and the 7900 XTX has more of its own).  gfx1100 on a
dual-card **`-sm tensor`** split is the one cell we still cannot measure here (only a single 7900 XTX
is available); a community report on 2x RX 7900 XTX is pending.

**Turning it off.**  `LLAMA_KQ_MASK_DERIVED=0` restores the packed mask (upstream's behaviour).
Worth doing if you are on **gfx1100/gfx1151** and want the last ~1–2 % of deep prefill, or on a
**tensor split at shallow depth** and prefill latency matters more than the VRAM.  For a deep-context
MoE / qwen4exp workload the default is the right side of the trade.

**Not a correctness knob:** same-seed output is byte-identical either way (the derived mask produces
the same values; only the memory layout and prefill cost differ).

**It works on both prefill kernels (r9).**  The derived mask is implemented by the **MMA** and the
**tile** flash-attention kernels, so it is no longer tied to the chooser picking MMA: a head above the
per-arch WMMA cap (RDNA4 576, RDNA3_5 320, RDNA3_0 256) used to lose the mask entirely, and that is
every **Gemma4** (head 512) on gfx1100/gfx1151 — the two arches that live on the tile kernel.  There
the mask is a *win*, not a tax (PP512, mask on vs off):

| config (tile kernel, natural selection) | cell | delta |
|---|---|---|
| gfx1100 Gemma4 12B, q8_0 KV | @ 16k / @ 32k | **+1.2 %** / **+0.9 %** |
| gfx1151 Gemma4 12B, q8_0 KV | @ 16k / @ 32k | **+1.6 %** / +0.6 % |
| gfx1201 Gemma4 E4B (tile forced) | @ d0 | **+2.3 %** |

and **decode pays nothing for it.**  Decode and the spec verify batch always take the tile kernel (the
chooser's WMMA branch requires `ne[1] > 8`), so the derived branch there is a cost on every arch; it is
hoisted out of the KV loop so the packed path's code generation is unchanged.  Measured r8 vs r9 at
that kernel: `tg128` deltas of **+0.01 %** (gfx1201 9B @ d16384), **+0.02 %** (gfx1100 9B @ d16384),
and flat on gfx1151 — the earlier per-iteration form cost −0.5..−0.8 % at depth before the hoist.

**The vec kernel has no derived arm**, but it is decode/verify-only (`n_tps <= 2`) while the derived
form only exists for prefill-shaped batches (`kq_mask_derivable()` rejects `n_tokens <= 8`), so it
cannot be selected for one.  If the launch log *does* print `derived kq mask flash attention not
supported, set to disabled` on a CUDA/HIP backend, the FA node did not reach the GPU at all — check
which kernel serves that head (the log adds a note pointing there).

One performance caveat with nothing to do with this knob: forcing the **tile** kernel for a head it
would not normally serve (a stale `GGML_CUDA_FA_WMMA_256=0`, a fixed env in the September qwen4exp
gates, is the usual cause) makes a head-256 model on gfx1201 **~3x slower at deep prefill** (9B,
`-d 98304`: 2104 -> 710 t/s).  That env is worth removing regardless of the derived mask.  One
exception worth knowing: **qwen4exp / Qwen3.8-Flash-Next gets its deep-context mask elision from the
QSA path's own derived visibility** (`GGML_QSA_DERIVED_VIS`, the code's "-800 MiB win"), which is
independent of this knob; `LLAMA_KQ_MASK_DERIVED` only serves that model's dense shortcut
(`n_kv <= 2051`), where the mask is tiny.

Full matrix, raw CSVs and the A/B harness: [`archive/work/kq-mask-derived-ab/`](archive/work/kq-mask-derived-ab/); the
2026-09-19 block-15 (r7) amendment in [`patches/README.md`](patches/README.md).

## Cross-start determinism on ROCm (issue #67)

Issue [#67](https://github.com/stew675/llama-cpp-rdna-boosts/issues/67) (from #58 item D, reported by
[@DanoPTT](https://github.com/DanoPTT) on Windows / ROCm 10 / gfx1201) tracked Q6_K greedy output that
intermittently swapped one near-tie across fresh `llama-server` starts while staying deterministic within
a start.  It is **closed as an external ROCm issue — nothing to fix in this patch set.**  Two separate
defects were in play:

* **A real bit-transparency bug in the fused rope path, fixed in r25.**  The address-gated
  `ROPE -> VIEW -> SET_ROWS` fusion did not reproduce the unfused chain bit-for-bit: clang contracted the
  fused `<float,__half>` and unfused `<float,float>` template instantiations differently, and one element
  of the prefill K-cache write landed on opposite sides of an f16 rounding boundary.  r25's
  `#pragma clang fp contract(off)` in `rope.cu` makes every instantiation round identically, verified at
  the logits level on Linux (`W=1` hash `60e77916673db071` with the fusion on and off).  See
  [Current state](#current-state) for the r25 record.

* **The residual per-start flip is hipBLASLt solution selection in ROCm, not ggml** — and it reproduces
  on a stock upstream build with none of these patches.  The F32 `mul_mat` for `ssm_alpha`/`ssm_beta`
  (5120 -> 48) at 512 prefill columns misses the mmvf/mmf kernels, goes to `hipblasSgemm`, and rocBLAS
  routes it to hipBLASLt (`rocblaslt_matmul`, T,N, m=48 n=512 k=5120).  `HIPBLASLT_LOG_MASK=160` shows a
  per-process pick between a bit-identical reference solution pair (`140231`/`140232`) and a deviating
  pair (`140216`/`140217`); it is random from process to process, so layer 0's prefill output — and with
  it the whole run — differs at startup.  This is
  [ROCm/rocm-libraries#12126](https://github.com/ROCm/rocm-libraries/issues/12126).

**Workaround: `ROCBLAS_USE_HIPBLASLT=0`.**  On the reporter's setup, 150 fresh starts per arm: default
**5/150** deviate, `ROCBLAS_USE_HIPBLASLT=0` **0/150** (bit-identical to the reference class), with no
measurable prefill/decode cost on the current build (the earlier text-hash runs measured higher flip
rates, up to ~14 %).  `ROCBLAS_DEFAULT_ATOMICS_MODE=0` does not help.  The selection is **prefill-only**
(decode at n=1 uses mmvf), so decode throughput is unaffected.

**Linux / ROCm 7.14 does not reproduce it** — the reporter's box (Windows / ROCm 10) is where the
per-process selection actually varies.  The r24 rope-fusion kill switches
(`GGML_CUDA_DISABLE_ROPE_SET_ROWS=1`, `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1`) and r25's fix remain
shipped for bisection and for the real defect above.

## When upstream master moves

The patches are static against the fork point in `release.json.base`. When upstream
drifts and hunks no longer apply, re-base the block commits (the fork checkout carries
them), regenerate the whole set with `scripts/make-patches.sh`, then refresh
`release.json` (`scripts/make-release.sh --base … --tip … --tree …`) and update the
current-state headers. The old `baseline/<sha>`-branch-per-upstream-range workflow
was retired when the delivery moved to the flat 16-patch set on `main`.

## Upstreaming

Some blocks are candidates for upstream contribution to
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp); others are
expected to stay fork-local. Block 12's internal all-reduce is gated to
RDNA4 pending community verification on RDNA3 pairs. See `MANIFESTS.md`
for per-block verification and `BASELINE.md` for provenance.

## Current state

- **Release `v16-a55e952b8-r6` (2026-10-05): issue #95 - dynamic-backend (Docker) builds now allow
  `-sm tensor` for qwen4exp.**  Same fork point `a55e952b8`; new canonical block-15 tip `1d10390a8`,
  net tree `2b57533c8002d11bd047c75a3323b30229f7f526`; strict 16/16 `git am` (`validate-set.sh`
  green).  `ggml_add_backend()` publishes `GGML_USE_<backend>` on the `ggml` target only when
  `GGML_BACKEND_DL=OFF`; the published containers build with `-DGGML_BACKEND_DL=ON` (for
  `GGML_CPU_ALL_VARIANTS`), so `src/llama-arch.cpp`'s `#ifdef GGML_USE_HIP` compiled to the
  HIP-absent branch there and the qwen4exp tensor-split gate rejected `-sm tensor` even though the
  HIP backend was built.  `ggml/src/ggml-hip/CMakeLists.txt` now also does
  `target_compile_definitions(ggml PUBLIC GGML_USE_HIP)`, so the macro reaches the main libraries in
  both build modes; the static build already had it and is unchanged.  Verified on a local
  `GGML_BACKEND_DL=ON` configure matching the Dockerfile: before, the `llama` target had no
  `GGML_USE_HIP`; after, the `LLM_ARCH_QWEN4EXP` case preprocesses to `return true`.  See
  `WORKLOG.md` 2026-10-05 (r6) and issue #95.
- **Previous release `v16-a55e952b8-r5` (2026-10-05): contributor PR #96, the block-06 scheduler re-stage fix
  for the MoE expert cache, is folded into the delivery.**  Same fork point `a55e952b8` (tree
  `3550faf840a88ae652e5ff8d32067f28a836d87b`); new canonical block-15 tip `b5ca42a92`, net tree
  `c5c716e796b29d770902ff2aecfeaba42e80f487`; strict 16/16 `git am` (`validate-set.sh` green).
  `ggml_backend_sched_split_graph` registered a weight as a split input only when it first created
  the copy, so when the same host-resident expert weights fed `MUL_MAT_ID` in two splits the later
  split reused the earlier split's copy.  With `MOE_EXPERT_CACHE_MIB` armed that copy was taken over
  by the 1-row decode-band consumer (`moe_cache_take_over` aliases it and the op never fills it), so
  the later wide consumer read stale bytes.  qwen4exp's unmasked MTP export (a full-row last-layer
  FFN next to the gathered logits tail) is exactly that second consumer: the wide op produced NaN
  in `t_h_nextn`, which reached the drafter's KV and collapsed MTP draft acceptance on every slot
  while target output stayed correct.  The fix registers the weights as an input of the current
  split as well (once per split) when the copy already exists and the node is a `MUL_MAT_ID` reading
  WEIGHTS through `src[0]`, so they are staged for this split's routing; single-consumer graphs are
  unchanged.  End-to-end gate (R9700, Flash-Next UD-IQ3_XXS + shared Q8_0 MTP head, `-ncmoe 48`,
  `MOE_EXPERT_CACHE_MIB=2048`, `-ctk/-ctv q8_0`, MTP n3): a 446-token prefill collapses the probe
  from 141/149 to **0/591** on r4, and the fix holds 141/149; the same sequence on IQ4_NL is
  142/156 -> **0/591** on r4 and 142/156 with the fix; target output identical throughout.  Gates:
  clean warning-free build, `MUL_MAT_ID` 931/931, dense 4B `1c5d32ac537d`, qwen4exp Flash-Next
  Q4_K_M (no MTP head) `622da9ec8ec2`.  See `WORKLOG.md` 2026-10-05 (r5) and
  `archive/work/sched-moe-restage/`.
- **Previous release `v16-a55e952b8-r4` (2026-10-05): the last two r1 follow-ups — `qwen4exp-qsa-convergence`
  and `lightning-indexer-fusion` — are resolved (blocks 14 and 15).**  Same fork point `a55e952b8`;
  new canonical block-15 tip `cd1485fd1`, net tree `714f94f050dfce08c987a8a14467f456fe6e9d60`;
  strict 16/16 `git am` (`validate-set.sh` green).  **qwen4exp-qsa-convergence:** keep the fork's
  fused QSA graph (A) and adopt upstream `10f340d1a`'s `ggml_build_forward_expand(gf, res_hc)` after
  `cb(res_hc, "hc_init", -1)` (block 14); byte-identical on the gates, inert while the `-sm tensor`
  gate stays HIP-only; defer upstream's kpool graph/pooling
  (`archive/work/qwen4exp-qsa-convergence/DECISION.md`).  **lightning-indexer-fusion:** register the
  fused indexer-score nodes as `LLM_FUSED_OP_LIGHTNING_INDEXER` (upstream `889edf43d`) — the decode
  op (`ggml_indexer_score`, block 14) and the prefill WMMA arm (`ggml_lightning_indexer`, block 15)
  — so `resolve_fused_ops()`'s Lightning Indexer probe can report a layer/device mismatch; inert
  today (`cparams.auto_flid = false`).  The handover's `llama_prefetch_rows` PLE item was **dropped**
  as a measured ~15–20 % pp512 regression vs the fork's existing per-row `madvise` loop (1220 vs
  1450–1542 t/s), the score/mask items were dropped, and the standard-FA index-list campaign is
  deferred (`archive/work/lightning-indexer-fusion/RESULTS.md`).  Gates: clean warning-free `all`
  build, `INDEXER_TOPK,INDEXER_SCORE,FLASH_ATTN_QSA,HC_MIX` 59/59, `FLASH_ATTN_EXT` 6358/6358, dense
  4B `1c5d32ac537d`, qwen4exp IQ4_NL `359ff4337837` (`-lm none` and `-lm auto`), `tg128` 56.4 vs
  57.1.  See `WORKLOG.md` 2026-10-05 (r4).
- **Previous release `v16-a55e952b8-r3` (2026-10-05): the `mmq-prec-gate-fp4` and
  `shared-expert-fusion-reconcile` r1 follow-ups are resolved (block 13 only).**  Same fork point
  `a55e952b8`; new canonical block-15 tip `3d1cd47f2`, net tree
  `25a8e137a585cd9fc2907a74236998f881635b8e`; strict 16/16 `git am` (`validate-set.sh` green).
  **mmq-prec-gate-fp4:** `ggml_cuda_mul_mat_q_switch_type_gate` takes `prec_src1` and asserts
  `GGML_PREC_Q8`, and `ggml_cuda_mul_mat_q_pair` asserts no `NVFP4`/`MXFP4` weight — no-ops today
  (the gate types are Q3_K/Q4_K/Q5_K/Q8_0/Q6_K) that stop a future FP4 gate type from silently
  hardcoding Q8 and disabling Blackwell W4A4; `MUL_MAT,MUL_MAT_ID` 2235/2235 at
  `GGML_CUDA_MMQ_PREC=q8` and `=q4`.  **shared-expert-fusion-reconcile:** upstream's base
  `bed0a8566` fused shared-expert MMVQ and block 13's `shexp_down_gate` epilogue are **disjoint**
  (both fire under `-sm layer`; upstream is dormant under `-sm tensor` because the routed expert is
  sharded to a 128-multiple FFN while the shared expert is mirrored); no functional change, the
  precedence is now documented in the code.  Gates unchanged: `MUL_MAT_ID` 931/931, `HC_MIX` 30/30,
  `RMS_NORM` 51/51, `ARGSORT` 78/78, `INDEXER_TOPK` 3/3, `GATED_DELTA_NET` 46/46, dense 4B
  `1c5d32ac537d`, 3-GPU `-sm tensor` Flash-Next IQ4_NL `359ff4337837`.  See `WORKLOG.md` 2026-10-05
  (r3), `archive/work/mmq-prec-gate-fp4/RESOLUTION.md` and
  `archive/work/shared-expert-fusion-reconcile/RESULTS.md`.
- **Previous release `v16-a55e952b8-r2` (2026-10-05): the first two r1 re-base follow-ups are resolved.**  Same fork
  point `a55e952b8` (tree `3550faf840a88ae652e5ff8d32067f28a836d87b`); new canonical block-15 tip
  `dbe88ea6e3afd86da26ce766ae8b71d2b26b67ac`, net tree `c38ba8f2066f01c3a1f69207a7e0860e5026ef17`;
  strict 16/16 `git am` on a fresh base tarball (`validate-set.sh` green).  **Merge hygiene:** every
  block commit now builds the `all` target (the eight intermediate breaks are redistributed to the
  blocks that own them; see `archive/work/rebase-merge-hygiene/RESOLUTION.md`).  **Integration
  audit:** DFlash `GGML_LF_DFLASH_DEV=1` vs `=0` byte-identical, `common_sampler_clone`/probabilistic
  MTP clean, and a new duplicate-value `ARGSORT` case exposed a real CPU-oracle bug (the CPU
  `cmp_argsort` compared values only, so its tie order was unstable while the CUDA bitonic path is
  index-stable); its comparator is now a total order (`ARGSORT` 78/78).  Gates: `MUL_MAT_ID` 931/931,
  `RMS_NORM` 51/51, `INDEXER_TOPK` 3/3, `HC_MIX` 30/30, `GATED_DELTA_NET` 46/46, dense 4B
  `1c5d32ac537d`, 3-GPU `-sm tensor` Flash-Next IQ4_NL `359ff4337837`.  See `WORKLOG.md` 2026-10-05
  (r2).
- **Previous release `v16-a55e952b8-r1` (2026-10-05): the 16-block set re-based onto upstream master
  `a55e952b8` (203 upstream commits).**  Fork point `a55e952b8` (tree
  `3550faf840a88ae652e5ff8d32067f28a836d87b`); canonical block-15 tip `def454e4c`, net tree
  `6a44aa2904772db02dbc88960397efe8138498df`; strict 16/16 `git am` on a fresh base tarball
  (`validate-set.sh` green).  The re-base folded in upstream's batch-API migration
  (`common_batch.add`/`llama_process`), the probabilistic-draft-sampling commit, the bitonic-argsort
  refactor, the `rms_norm`+`scale` fusion, the fused shared-expert MMVQ launch, the BF16 `ggml_cuda_cast`
  generalization, the `ggml_prec prec_src1` MMQ parameter (merged with our `has_gate`), and upstream's
  own qwen4exp MTP/kpool/mask/indexer/`-sm tensor` work.  Our qwen4exp fused-op implementation is
  kept as the model graph; upstream's kpool machinery coexists (and serves glm5-next).  Block 08's
  standalone `rms_norm_scale_f32`/`GGML_CUDA_FUSE_RMS_SCALE` fusion was **retired** (upstream
  `1ab7e5ad2` subsumes it).  Gates: clean build, `MUL_MAT_ID` 931/931, `FLASH_ATTN_EXT` 6358/6358,
  `HC_MIX` 30/30, `FLASH_ATTN_QSA` 26/26, `INDEXER_TOPK` 3/3, `GATED_DELTA_NET` 46/46, `RMS_NORM`
  51/51, dense 4B `1c5d32ac537d`, 3-GPU `-sm tensor` Flash-Next IQ4_NL `359ff4337837`.
  See `WORKLOG.md` 2026-10-05 (r1).
- **Release `v16-84e76d8a2-r37` (2026-10-05): BF16 hyper-connection mixer fusion folded into block
  15.**  Integrates contributor PR #91 (@briansp2020).  The ISTA-DASLab GSQ-RCO quants keep
  `hc_{attn,ffn}_{down,up,inject}` in BF16, and block 14's fused `GGML_OP_HC_MIX` was Q8_0-only, so
  those models ran the six-dispatch unfused chain (rms×gamma → down mmvf → scale+silu → up mmvf →
  `dsv4_hc_pre` → inject mmvf), 96 mixers per token on a dispatch-bound decode.  The new BF16 arm
  replays that chain in three dispatches with the same per-thread K order and reductions
  (bit-identical): `hc_mix_rms_gamma_quant` without the quantize; the down rows and the inject rows in
  one grid (`mul_mat_vec_f<bf16,float,nt,256>`); and the up rows of the four hc streams of one column
  plus the gated collapse in one block (`mul_mat_vec_f_vb<bf16,160,4>` + `dsv4_hc_pre_f32`).  It
  engages only for GPU-resident BF16 hc weights at `hc_lr == 320` (the emulated mmvf block) and
  `hc == 4`; `LLAMA_HC_MIX_BF16=0` keeps the chain.  The CPU `HC_MIX` reference gained the matching
  BF16 arm, and the new `test-backend-ops` BF16 cases (`HC_MIX` 30/30 = 20 Q8_0 + 10 BF16 across
  `nt` 1/2/3/5/8) exposed and fixed a **pre-existing** no-inject dst-stride bug at `nt > 1` (the
  reference hard-coded the `n_embd + hc` row stride; with no inject tail it is `n_embd`).  Output-
  preserving: fused == unfused byte-identical for `nt` 1/3/5/8 on a BF16-hc Flash-Next fixture
  (`-ub 3/5/8`, all `8ada57e4b2bd7522`), +3.8 % `tg256` on that 4-layer fixture vs the contributor's
  full-model `tg128` +4.9 %; strict 16/16 `git am`, `validate-set.sh` green (tip `f3994517`, tree
  `ea5f8012f30d1aef94f1b3057ae58897fff0d61a`).  See `WORKLOG.md` 2026-10-05 (r37) and
  `archive/work/rdna4-hc-mix-bf16/`.
- **Release `v16-84e76d8a2-r36` (2026-10-05): issue #89's indexer top-k block-path fix folded into
  block 15.**  Integrates contributor PR #90 (@briansp2020).  The fused indexer top-k's block fast
  path (`indexer_topk_radix_cuda_blocks`) partitioned its block-level radix passes by block range while
  the gather (`indexer_topk_write_blocks_grouped`) partitions by cell range; the histogram-derived
  per-range `g_cnt`/`e_cnt` bases only match when block `b`'s cells are `[b*r, b*r+r)`.  A unified KV
  holding several sequences (blocks keyed by (sequence, position bucket)), or a single sequence whose
  KV head has moved past cell 0, broke that: output entries were left unwritten or overwritten, and
  `flash_attn_qsa` then gathered K/V at stale out-of-range indices (GPU page fault, server hang) or
  silently attended to the wrong cells.  A new `indexer_topk_count_cells_grouped` kernel recounts
  `g_cnt`/`e_cnt` over exactly the cell ranges the gather walks after the last radix pass, with the
  gather's key logic; passes 2-4 stay block-level and the from-cell-0 output is unchanged.  A new
  `INDEXER_TOPK` backend-op case (`test_indexer_topk_block`; a from-cell-0 control, a one-sequence
  offset map and a two-stream unified-KV map) fails on the unfixed r35 build, and the reporter's
  concurrent unified-KV scenario on Qwen3.8-Flash-Next UD-IQ3_XXS crashes the unfixed r35 kernel in
  `flash_attn_qsa<256, Q8_0>` while running clean on r36.  Output-preserving: 4B
  `7386359e5dac` and 35B-A3B `cf7f8b23f404`; `INDEXER_TOPK` 3/3, `TOPK_QSA` 4/4, `LIGHTNING_INDEXER`
  225/225, `FLASH_ATTN_QSA` 26/26, `MUL_MAT_ID` 929/929; strict 16/16 `git am`, `validate-set.sh` green
  (tip `9b8b6f108`, tree `c595f29253ad70d693793d010f5e5399dadf57ae`).  See `WORKLOG.md` 2026-10-05 (r36)
  and `archive/work/issue-89/`.
- **Release `v16-84e76d8a2-r35` (2026-10-04): block-06 issue-#87 fix is default-on.**  The r33 A/B
  candidate is promoted to the default after the reporter confirmed it clears their crash.  The r26 async
  split-input H2D path (`ggml_backend_sched_compute_splits`) copies straight from the host pointer, and
  the recurrent-state copy `rs_s_copy` is always consumed through views (`s_copy_main` / `s_copy_extra`)
  that lose `GGML_TENSOR_FLAG_INPUT`, so the copy races the host overwrite on the next ubatch.
  `ggml_backend_sched_graph_input()` resolves the view chain and the copy loop takes the synchronous
  user-input branch for (views of) graph inputs, while host-weight uploads keep the async/staged path.
  `GGML_SCHED_SYNC_GRAPH_INPUTS` is now an opt-out: unset = enabled, `=0` restores the r26 behaviour
  (A/B / bisect only).  Output-preserving: 4B `Qwen3.5-4B-Q8_0` `-sm tensor` `7386359e5dac` with unset
  and `=0`, and 35B-A3B `Qwen3.6-35B-A3B-Q8_0` `-sm layer` `cf7f8b23f404`; `MUL_MAT_ID` 929/929; strict
  16/16 `git am`, `validate-set.sh` green (tip `b01620f2d`, tree
  `d08fbaf2ca842ea3c3ce044ac45c0a8d0f11c597`).  Blocks 07-15 are rebased onto the amended block 06.  See
  `WORKLOG.md` 2026-10-04 (r35).
- **Release `v16-84e76d8a2-r34` (2026-10-04): the issues #59/#60 qwen4exp fix folded into block 15.**  Promotes
  the `archive/work/issues-59-60` gfx1100 candidate (built on r27) onto r33, before the reporter's external 196K
  confirmation returned.  **#59 (closed):** `ggml_cuda_flash_attn_qsa3_supported` is false on RDNA3_0
  unless `GGML_CUDA_QSA3=1` (`=0` force-off everywhere), and a packed QSA op's support equals the qsa3
  predicate; the qwen4exp graph probes the packed op through `ggml_backend_dev_supports_op`, so on gfx1100
  the two natural-F16 K/V packs are not materialised.  RDNA3_5/RDNA4 defaults unchanged.  **#60:** the
  4-head lightning-indexer prefill score is supported on RDNA3_0 again, and `build_qsa_top_k`'s `use_wmma`
  takes the fused op once the score exceeds `LLAMA_QSA_SCORE_WMMA_MB` MiB (default 64, `0` = always fused)
  and keeps the faster chain below it (`n_kv` about 64K for the reporter's `r=4` geometry, so their d30K /
  d64K points keep the chain and the 196K prefill fits).  The unfused chain's `mul_mat+relu` and chunked
  form now use `ggml_relu_inplace` (bit-identical; removes the 2x-score reserve peak).
  `LLAMA_QSA_SCORE_WMMA=0/1` forces chain/fused.  Also counts the FA prefill staging arena in
  `llama_get_memory_breakdown`/`--fit` (issue #33 follow-up).  Candidate gates on gfx1100: clean build,
  `FLASH_ATTN_QSA` 23/23 (26/26 with `GGML_CUDA_QSA3=1`), `LIGHTNING_INDEXER` 225/225, `TOPK_QSA` 4/4,
  `FLASH_ATTN_EXT` 6354/6354, dense 27B same-seed `1acb04bd9104` identical to r20; rebased onto r33 with no
  conflicts; strict 16/16 `git am`, `validate-set.sh` green (tip `33a8c30db`, tree
  `3c07e1f6e303efa59a92d0d63d2acf5e30666cb2`).  See `WORKLOG.md` 2026-10-04 (r34) and `archive/work/issues-59-60/`.
- **Release `v16-84e76d8a2-r33` (2026-10-04): block-06 default-off candidate for issue #87.**  The r26
  async split-input H2D path (`ggml_backend_sched_compute_splits`) copies straight from the host pointer,
  and the recurrent-state copy `rs_s_copy` is always consumed through views (`s_copy_main` /
  `s_copy_extra`) that lose `GGML_TENSOR_FLAG_INPUT`, so the copy races the host overwrite on the next
  ubatch.  `GGML_SCHED_SYNC_GRAPH_INPUTS=1` resolves the view chain with
  `ggml_backend_sched_graph_input()` and takes the synchronous user-input branch for (views of) graph
  inputs, while host-weight uploads keep the async/staged path.  Unset keeps the r26 behaviour.  This is
  an A/B candidate: the default flips to on only if the issue #87 reporter confirms it fixes the crash.
  Output-preserving: 4B `Qwen3.5-4B-Q8_0` `-sm tensor` `7386359e5dac` and 35B-A3B
  `Qwen3.6-35B-A3B-Q8_0` `-sm layer` `cf7f8b23f404` with the variable unset and `=1`; `MUL_MAT_ID`
  929/929; strict 16/16 `git am`, `validate-set.sh` green (tip `13a3b1353`, tree
  `14444e869d75871514d2aa99924264386014d55a`).  See `WORKLOG.md` 2026-10-04 (r33).
- **Release `v16-84e76d8a2-r32` (2026-10-04): four contributor PRs folded into block 15.**  The set
  accepts and integrates **PR #78** (RDNA4 mmvq workaround for a gfx1201 grid-size dispatch stall:
  single-token Q4_K/Q5_K/Q6_K/IQ4_XS decode row-loops a grid capped at 1792 blocks, per-type rows per
  block from the row sweeps, and 2 rows at 4..8 verify tokens for Q8_0 short-K; repro
  [ROCm/TheRock#8634](https://github.com/ROCm/TheRock/issues/8634)), **PR #83** (the GSQ-RCO RDNA4
  kernels on top of #78: BF16 `mul_mat_vec_f` unroll and warp-per-row `mul_mat_vec_f_vb`, IQ2_S/IQ3_S
  `apply_ksigns`, IQ3_S/IQ2_S in the row loop, and IQ2_XXS/IQ2_S/Q2_0 routed-compact MoE mmq),
  **PR #84** (x86 AVX2 `ggml_vec_dot_q2_0_q8_0`, bit-identical to the generic scalar code) and
  **PR #81** (issue #80: an exact top-k fast path in `common_sampler_sample()` plus a clone that does
  not copy the candidate array; `GGML_LF_FAST_TOPK=0` opts out).  All four are bit-exact or
  output-identical.  One integration note: #78's `nrows_loop` signature hunk has a context identical
  to the earlier `mul_mat_vec_q_switch_fusion`, so `git am` placed it on the wrong function; the fold
  moves it to `mul_mat_vec_q_switch_fusion_ksplit`.  Gates: clean warning-free build, `MUL_MAT_ID`
  929/929, `MUL_MAT` 1297/1297, CPU `MUL_MAT` 1323/1323 (80 `q2_0`), 4B `-sm tensor` `7386359e5dac` and
  35B-A3B `-sm layer` `cf7f8b23f404` identical to r31, sampling on == off `c118179c57ec`, strict 16/16
  `git am`, `validate-set.sh` green (tip `9d46b0966`, tree `b090750760c58cc4c2271cbf4d260fe0413c52a3`).
  See `WORKLOG.md` 2026-10-04 (r32).
- **Release `v16-84e76d8a2-r31` (2026-10-03): two MMQ `MUL_MAT_ID` tail over-read holes in the
  host-resident expert path.**  The quantized `MUL_MAT_ID` MMQ loader reads a full K tile without
  clamping the fast path to the row, so a partial expert buffer's last row over-reads into the **head of
  the next slot** (`NaN * 0 = NaN` poisons the tile -> a repeated `/`).  The host copy path covers this
  (`copy_experts` copies `+ min(expert_size, 512)` bytes); two pruned device buffers did not.
  **Hole A** (block 13, `moe-expert-cache.cu`): the decode-cache slot arena was `cudaMalloc`'d with **no
  zero and no tail pad**, so an empty slot's head was uninitialized (NaN) memory and the bytes past the
  last slot were out of the allocation — the corruption an aborted stream exposed, persistent for the
  process.  It is now `slots*expert_bytes + min(expert_bytes,512)` and `cudaMemset` once.  **Hole B**
  (block 06, `ggml-backend.cpp`): the device gather's one-time head zero does not survive a multi-ubatch
  prefill (the graph allocator re-uses `input_cpy`), so NaN MoE routing skipped expert work and the
  gather benchmarked 2-4x fast; its old 2650-3060 t/s is **above this box's PCIe bandwidth** for the
  bytes copied, so it was never real.  `sched->devgather_enabled` now defaults to **false** (staging /
  host-copy, correct by construction); `GGML_SCHED_DEVGATHER=1` re-enables the gather for A/B only.  The
  two holes are independent (the arena fix does not move the gather number), so both fixes are needed,
  and both need a **partial** expert buffer — `-ncmoe 0` / device-resident MoE is unaffected.
  **Re-baselined** (qwen4exp IQ4_NL, 1 R9700, `MIB=12288`): `pp8192` **659 / 1018 / 1396** and `tg1024`
  **39.7 / 40.3 / 35.9** at `-ub 2048/4096/8192`; 2-GPU `-sm tensor` `pp1024` **562**.  Strict 16/16
  `git am`, `validate-set.sh` green, warning-free build, `MUL_MAT_ID` 929/929.  Full record:
  [`wip/moe-mmq-overread/RESOLUTION.md`](wip/moe-mmq-overread/RESOLUTION.md), `WORKLOG.md` 2026-10-03
  (r31).
- **Release `v16-84e76d8a2-r30` (2026-10-02): block-13 amendment - the expert-gather head pad**
  (superseded by r31's broader over-read fix).
  r29's always-on host-resident-expert **device gather** hard-coded its one-time expert-head zero to 64
  bytes (the *IQ4_NL* threshold the beta5 session measured, not a quant-independent one), so **IQ4_XS**
  (and any quant whose MMQ over-read is wider) corrupted into a repeated `/` - on one GPU as well as
  multi-GPU; the beta5 gate set only used IQ4_NL / Q8_0 / Q4_K_M.  The gather now uses the host path's
  own guard value, `min(expert_bytes, 512)`, and keys the one-time zero on `(input_cpy buffer,
  expert_bytes)` because the allocator reuses one `input_cpy` across tables with different geometry.
  Prefill is unchanged (`pp8192` 3439.15 -> 3438.81 t/s).  The new per-quant regression gate is
  [`scripts/gate-qwen4exp-quant-coherence.sh`](scripts/gate-qwen4exp-quant-coherence.sh) (0 `////` and
  gather ON == OFF per quant; passes on r30, hard-fails on r29).  See `WORKLOG.md` 2026-10-02 (r30).
- **Release `v16-84e76d8a2-r29` (2026-10-01): the decode-side MoE expert cache is in the delivery** -
  superseded by r30's gather head-pad fix.
  The `archive/work/moe-expert-cache` campaign (promoted through PR #82) is folded into the 16 blocks:
  **block 06** takes the generic backend expert-cache interface (`moe_cache_update`/`_take_over`/
  `_promote`/`_gather`; the CPU and RPC iface vtables name the new fields `NULL`, so every backend still
  builds warning-clean) and the scheduler half (`ggml-backend.cpp`, `ggml-backend-meta.cpp`), **block 13**
  the engine itself (`ggml/src/ggml-cuda/moe-expert-cache.{cu,h}` — a per-device VRAM slot arena over the
  pinned host expert pool, LFRU admission, UVA cold reads, the full-residency identity fast path, the
  device-side remap with pipelined promotion, the on-GPU admission policy and the prompt-routing seed)
  plus the in-kernel slot lookup in `mmvq.cu`, **block 14** the gemma4 `-sm tensor` guard, and **block 15**
  the CUDA consumer glue that interleaves with its own fusion/staging code (`ggml-cuda.cu`, `common.cuh`,
  the `stage_input` `stage_gather` guard).  **The cache is opt-in: `MOE_EXPERT_CACHE_MIB=<MiB>` arms a
  per-device VRAM arena; unset, every entry point is a no-op and the build is bit-identical to r28.**
  The two always-on parts are the device-side expert **gather** (an expert table `>= 224 MiB` always
  gathers instead of staging the whole shard) and the decode-band gate on the routed-expert rebalance;
  **r31 demoted the gather** — its old "1403 -> 3065 t/s" was the prefill over-read corruption, not a
  real win.  Measured on one R9700 with Qwen3.6-35B-A3B `Q8_0`
  (37.8 GB on a 32 GiB card) at the real target — 128K context, `q8_0` KV — `-ncmoe 20
  MOE_EXPERT_CACHE_MIB=8192` gives **884 pp8192 / 57.1 tg@128k**; the arena alone takes 1-GPU `-sm layer`
  decode 39.7 -> 74.2 -> 81.2 t/s and 2-GPU `-sm tensor` 32.9 -> 78.6, and prefill matches or beats r28 at
  every measured `-ncmoe` cell.  **Config guide with the measured tables: [`COMMUNITY-CONFIG.md`](COMMUNITY-CONFIG.md).**
  Gates (gfx1201 / ROCm 7.14): `validate-set.sh` green (strict 16/16 `git am`, applied tree ==
  `release.json.tree` `65276106f…`), warning-free build, `test-backend-ops -o MUL_MAT_ID` 929/929,
  byte-identity to the `-ncmoe 0` oracles (`de8be4d0c90c` 2-GPU `-sm tensor`, `15038c19ddc8` 1-GPU
  `-sm layer`), width purity `none == n1 == n3 == n7`, MTP `n3` acceptance 0.75273, deep coherence with 13
  sections + `## Conclusion`.  **One behaviour change for users: gemma4 with `-sm tensor` is now rejected
  with a clean error** (its fused `ffn_gate_up_exps` segmented split has no correct host-resident-expert
  async upload path, so `-ncmoe` asserted at the first expert upload — use `-sm layer`; tracked in
  `archive/work/moe-expert-cache`), and `--fit` does not know about the arena (it claims free VRAM after `--fit`
  has sized), so a large `MOE_EXPERT_CACHE_MIB` and a `-c`/`--fit` target compete for the same headroom.
  See `WORKLOG.md` 2026-10-02 (moe-cache beta5 fold / beta5 validation) and
  `archive/work/moe-expert-cache/PROMOTION.md` for the fold mapping and the full gate record.  The entries below
  describe r28 and earlier.
- **16-patch set** (block 00 + blocks 01-15) for llama.cpp at the fork point
  **`84e76d8a2`** (upstream master "metal : fix graph capture and handle empty graphs", 2026-09-24 re-base).
- Canonical 16-block chain on **`main`**: tip
  **`dc7d4772cf9f8a3a4b1c9b57e0b1e5b5f2b4b6f0`**, net tree
  **`1f83d4d36e4c5ef19143e90e180e6a9ede1e86df`**  (r8 campaign tree + the issue-#47 store fix + the r10
  mask skip + the r11 `rpb` mis-launch fix + the r12 staging ring + the r13 tiny-graph fix + the r14
  derived-mask device-window fix + the r15 host-expert pinning fix + the r16 host-resident-expert prefill
  fast path + the r17 decode regression fix + the r18 `ssm_gate_beta` width-uniformity fix + the r19
  offloaded-MoE thread cap + the r20 dense-Q6_K `VDR=2` and spec-verify HIP-graph fixes + the r21 three
  contributor PRs + the r22 `getenv` hot-path caching amendment + the r23 PR #64 three verify-band wins +
  the r24 address-gated rope-fusion kill switches + the r25 rope-fusion bit-transparency fix + the r26
  block-06 staging-ring overlap fix + the r27 four-PR collection and issue-#71 rows fix + the r28 VMM
  pool free-order fix + the r29 decode-side MoE expert cache + the r30 expert-gather head-pad fix + the
  r31 two MMQ `MUL_MAT_ID` tail over-read fixes + the r32 four-PR collection + the r33 default-off
  block-06 async graph-input race candidate + the r34 issues #59/#60 qwen4exp fix); release
  **`v16-84e76d8a2-r34`**.
- **Previously: VMM pool free-order abort fixed (block 15, r28, 2026-09-30, issue #76, PR #77 by overdoingism).**
  `ggml_cuda_pool_vmm` is a stack: `free()` must run in the reverse of the allocation order, and
  `ggml_cuda_pool_alloc` destroys in reverse declaration order, so a pool buffer has to be declared in the
  order it is allocated.  `kq_blocks` (the issue-#48 fully-masked-group bitmap) was declared after
  `dst_tmp`/`dst_tmp_meta` but allocated before them, so a batch that also needed `dst_tmp_meta`
  (fractional stream-k tiles, or `parallel_blocks > 1`) freed `kq_blocks` while it was not on top of the
  stack and aborted in `ggml_cuda_pool_vmm::free` right after prompt processing.  Only a VMM-enabled build
  (`GGML_HIP_NO_VMM=OFF`) shows it, because the legacy pool (the HIP default) does not check the order;
  `GGML_CUDA_FA_MASK_SKIP=0` was the workaround.  The fix only moves the declaration.  Reproduced and fixed
  on gfx1201 / ROCm 7.14 with `-DGGML_HIP_NO_VMM=OFF`: `test-backend-ops -o FLASH_ATTN_EXT` aborts in
  `launch_fattn` on its first case before the fix and passes **6354/6354** after it; the default build's 4B
  `-sm tensor` same-seed text is unchanged and the build is warning-free; strict 16/16 `git am`,
  `validate-set.sh` green.  Full record: `WORKLOG.md` 2026-09-30 (r28).
- **Previously: four contributor PRs + the issue-#71 rows fix (block 15, r27, 2026-09-30).**  **PR #68** (briansp2020)
  folds the dense FFN `silu(gate) * up` into the mmq down-projection quantize (`quantize_mmq_q8_1` `glu`
  variant; bit-exact; `GGML_CUDA_FUSE_SWIGLU_MMQ=0` off).  **PR #73** (overdoingism) keeps the DFlash
  target's layer features on the device for a single sequence (`GGML_LF_DFLASH_DEV=1`; kept opt-in because
  this box has no DFlash drafter, so the default-flip gate has not run).  **PR #74** (overdoingism)
  replaces the ksplit mmvq verify epilogue's per-output butterflies with a template-recursive halving
  reduce (bit-identical; one-wave blocks only).  **PR #75** (briansp2020) adds the qwen4exp `HC_MIX`
  band-up / prequantized-down-tail / register-`rms_gamma` kernels plus general RDNA4 1-token 2-row dense
  mmvq, F32 mmvf, `apply_ksigns`, an IQ2_XS MoE unroll and the IQ2_XS/IQ3_XXS routed-compact mmq bands (the
  latter gated to RDNA4, so gfx1151 keeps its source-of-record plain path).  **Issue #71** returns 1 row for
  `ncols_dst >= 2 && nwarps > 1`, so multi-row blocks are only used by one-wave blocks, while PR #75's
  single-token `RPB1` stays.  Gates: `test-backend-ops` MUL_MAT_ID 929/929, MUL_MAT 1297/1297, HC_MIX
  20/20, GATED_DELTA_NET 46/46; 4B coherence `1c5d32ac537d` and qwen4exp `359ff4337837` identical to r26;
  27B q8_0 width probe byte-identical with `width_purity=PASS`; strict 16/16 `git am`, `validate-set.sh`
  green.  Full record: `WORKLOG.md` 2026-09-30 (r27).
- **Previously: the op-offload prefill upload no longer serialises (block 06, r26, 2026-09-30, issue #50 staging ring).**
  The r12 staging ring was not overlapping a host-resident expert upload because `GGML_SCHED_EVENTS`
  defaulted **OFF** (so `wait_before_overwrite()` became a full device synchronize - measured 1858 calls /
  5.2 s in one single-R9700 8K prefill pass at `-ub 8192`) and the `split->n_inputs > stage_n_slots` gate
  skipped merged routed-MoE bands (31 inputs: one 450 MiB expert weight plus ~30 tiny view/ids inputs), so
  the weight took the serial host path.  The amendment counts host-weight inputs for that gate, defaults
  the events ON (`GGML_SCHED_EVENTS=0` opts out) and enqueues a host->device split-input copy asynchronously
  (`event_wait != NULL` gate, so the Meta backend keeps its buffer copy).  Delivery-only single R9700,
  Qwen3.8-Flash-Next IQ4_NL 8K prefill `-ub 512/1024/2048/8192` `~233/362/567/1090` t/s vs r25's
  `~233/362/425/870`; output-preserving (qwen4exp `359ff4337837` at default / `GGML_SCHED_EVENTS=0` /
  staging-off), `MUL_MAT_ID` 929/929, 2-GPU `-sm tensor` `-ncmoe 0` == `-ncmoe 40` == `359ff4337837`.  Full
  record: `WORKLOG.md` 2026-09-30 (r26), `patches/README.md` block-06 amendment.
- **Previously: the address-gated `ROPE -> VIEW -> SET_ROWS` fusion is bit-transparent (block 15, r25, 2026-09-29,
  issue #67, from #58 item D).**  The cross-start Q6_K greedy flip was not the allocator: a canonicalised
  per-graph allocation-plan dump is byte-identical with the fusion on vs off, so the `add_alloc_deps` pass
  needs no rope entry.  It was clang contracting the fused `<float,__half>` and unfused `<float,float>`
  template instantiations differently - both compute the f16-midpoint float `beb67000` for one element of
  the 256x1024 prefill K-cache write, yet stored -0.3562 vs -0.3564.  `#pragma clang fp contract(off)` at
  the top of `ggml/src/ggml-cuda/rope.cu` makes every rope instantiation round identically.  Verified:
  default and `GGML_CUDA_DISABLE_ROPE_SET_ROWS=1` now both give W=1 hash `60e77916673db071`,
  `width_purity=PASS`, 4B same-seed coherence unchanged (`1c5d32ac537d`), `validate-set.sh` green.  The
  residual cross-start flip the reporter still saw on Windows was later traced to hipBLASLt solution
  selection, not this stack — see [Cross-start determinism on ROCm (issue #67)](#cross-start-determinism-on-rocm-issue-67).
  Full record: `WORKLOG.md` 2026-09-29 (r25), `GREEDY-PURITY.md` §41.
- **Address-gated rope-fusion kill switches (block 15, r24, 2026-09-29, issue #58 item D).**
  `GGML_CUDA_DISABLE_ROPE_SET_ROWS=1` and `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1` bisect the cross-start
  greedy nondeterminism the reporter sees on Q6_K.  At the logits level on gfx1201 / ROCm 7.14 the W=1
  decode hash moves only with the address-overlap-selected fusion subset, and the upstream
  `ROPE -> VIEW -> SET_ROWS` fusion is the trigger (`GGML_CUDA_DISABLE_FUSION=1` and all-address-fusions-off
  both give `3ab223a4f08afd6e` against the default `f6d62323d9339541`; the other per-fusion switches, graphs
  and `FA_KV_NATIVE` do not move it).  Both switches default off, so the shipped path is unchanged.  Full
  record: `WORKLOG.md` 2026-09-29 (r24), `GREEDY-PURITY.md` §41.
- **Three RDNA4 verify-band wins (block 15, r23, 2026-09-29, PR #64 by @briansp2020).**  (1) A wide FA-band
  block (`ncols = 64`, 512 threads) computes query widths 5..8 in one pass over the KV cache instead of two
  32-column tiles that each stream the whole cache: gfx1201 q8_0 `n_q` 5..8 -11..-15 % from 4k KV rows,
  `n_q` 1..4 and f16 unchanged, `GGML_HIP_FA_BAND_WIDE=0` off.  (2) `ssm_gate_beta_fused_q8_0` takes an
  `ncols` template so the GDN gate/beta fusion serves 2..8 tokens (-144 launches per 5-token pass,
  `GGML_CUDA_FUSE_GATE_BETA_VERIFY=0` off).  (3) The residual ADD is folded into `rms_norm_q8_1` for 2..8
  tokens (-127 launches per pass, `GGML_CUDA_FUSE_ADD_RMS_Q8=0` off).  Contributor numbers: pp5 122.9 ->
  126.2 t/s, -1.7..-2 % ms per server verify step (ROCm 10.0).  Re-verified here on ROCm 7.14 / gfx1201:
  `FLASH_ATTN_EXT` 6354/6354, width-probe PASS with every W hash byte-identical with the fusions and the
  band on vs off, 4B coherence unchanged.  Full record: `WORKLOG.md` 2026-09-29 (r23).
- **`getenv()` on the fusion/staging hot paths is cached (block 15, r22, 2026-09-29, issue #65).**
  `ggml_can_fuse_subgraph_ext()` read `LLAMA_HC_CN_DEBUG` on **every** candidate fusion window (millions of
  calls per pass) and `gdn_conv_enabled()`/`ple_conv_enabled()` read `GGML_CUDA_DISABLE_CONV_FUSION` on
  every conv launch.  `getenv` is cheap on Linux, but on Windows it locks and rescans the environment block:
  the reporter measured 2,431,189 + 31,204 calls per 256 tokens and ~15 ms of extra host time per graph,
  costing ~10 % decode (40 vs 45 t/s).  Each flag is now resolved once; the scan also caches the other
  per-op/per-graph debug gates (`GGML_CUDA_GCDBG`, `GGML_CUDA_OP_TIMING`, `GGML_STREAMDBG`, the
  `GGML_META_*`/`GGML_SCHED_*` gates, `GGML_CUDA_MMQ_J_MAX`, `GGML_Q6_COMPACT_J`,
  `GGML_CUDA_DISABLE_MMID_512`, `GGML_PAIR_2X`, `GGML_CUDA_GDN_CHUNKED(_BF16)`, `GGML_CUDA_FA_WMMA_*`,
  `GGML_CUDA_QSA_*`, `GGML_CPU_MOE_OFFLOAD_THREADS`).  Same-seed text is byte-identical to the
  pre-amendment build (`83eec5e9b4f0`); the clean-build warnings are fixed too.  Full record:
  `WORKLOG.md` 2026-09-29 (r22).
- **Three contributor PRs by @briansp2020** (r21, 2026-09-28), each independently re-verified on ROCm
  7.14 / gfx1201: **#57** (blocks 10+13) RDNA4 multi-row mmvq verify blocks + exact `__mul24` --
  byte-identical output, `llama-batched-bench -npl 1,4,8` B=4 +13 %/B=8 +34 %, MTP n3 +11.5 %/n7 +29.5 %;
  **#62** (block 15) the RDNA4 GQA-6 FA band gets 64-wide K/V batches (bit-exact) + 8 warps (W-pure) --
  `tg128 @ d50000` 24.42 -> **25.52** t/s, text/perplexity unchanged; **#63** (blocks 08+14) five
  bit-exact verify-band fusions incl. the `rms_norm_q8_1` weight-stride fix -- text and MoE MTP
  acceptance identical.  Full records: `WORKLOG.md` 2026-09-28 (r21) and the three
  `wip/*/VERIFICATION-r21.md` notes.
- **Dense Q6_K `VDR=2` restored and spec-verify batches keep HIP graphs** (blocks 10 + 11, r20, 2026-09-28,
  issue #58): the 2026-09-12 verify-regression revert took Q6_K's `vdr2` (16 elements/call) down with
  Q4_K/Q5_K's `vdr4` (32/call); dense Q6_K now uses `_vdr2` again, scoped to RDNA4/RDNA3_0 and band-uniform
  across `W = 1..8` (Q6_K `B=8` 74.29 -> **86.61** t/s on the reporter's model; `Swift-Q5_K_M` 83.68 ->
  **85.89**).  Separately, the graph-cache predicate no longer lumps the fixed-shape spec-verify widths
  (2..8) in with prefill, and the cache is keyed per `(first node, n_tokens)`, so verify batches replay HIP
  graphs (Windows/ROCm 10 +16-19 %; ~+1 % on Linux/ROCm 7.14) with `GGML_CUDA_DISABLE_VERIFY_GRAPHS=1` as
  the kill switch.  Greedy `plain == draft-mtp` is byte-identical (`581aca110917`).
- **The offloaded-MoE decode runs multi-threaded, capped, instead of serialised** (block 06, r19,
  2026-09-28): r17's tiny-CPU-graph heuristic exempted `MUL_MAT_ID`'s src0 (the whole expert table) from
  its byte count, so every `-ncmoe` decode graph measured "tiny" and ran on one thread.  The r17
  measurement behind that (`tg64` 13.8 multi-threaded vs 24.4 single) was taken with the worker pool on
  the cores this host pins its GPU IRQs to (`pin_gpu_irqs.sh` puts them on the top `NUM_GPUS` cores --
  13/14/15 with 3x R9700): the same thread count on one CCD gives 36.0 t/s and `--poll 0` changes nothing, so the
  loss was the collision, not a thread-pool re-arm cost.  A graph whose `MUL_MAT_ID` weights are
  host-resident **but not in the CPU backend's buffer type** (the `-ncmoe` signature) is not tiny in work
  and now runs multi-threaded, **capped at `max(1, hardware_concurrency()/2)`**; the cap exists because a
  CPU+GPU split is a pipeline and the CPU side must not own every core.  `GGML_CPU_MOE_OFFLOAD_THREADS=N`
  overrides it (`0` = uncapped; an override above the default warns once), so users keep control.  Net at
  `-ncmoe 99 -t 16` (capped vs the one-thread behaviour): d0 Q8_0 24.80 -> **29.44**, Q4_K_M 29.11 ->
  **38.01**, gemma-4-26B-A4B Q4_K_XL 21.99 -> **37.44**; d16384 +23 %/+32 %/+67 %; MTP `n3` acceptance
  unchanged (0.79268) at **+81.7 %** t/s; prefill and the non-MoE path unmoved, and same-seed output is
  byte-identical (`431bbf3a1605`) at every thread count and override.  **Consequence for benchmarking on
  this host: size `-t` (or `--cpu-mask`, with `-t` inside the mask) to leave the GPU IRQ cores free -- one
  CCD's worth of threads is the robust choice.**
- **The qwen35moe SSM gate/beta fusion is width-uniform** (block 13, r18, 2026-09-28): block 08's
  decode-only `ggml_cuda_op_ssm_gate_beta` pinned plain `calc_nwarps()` (1 warp) while block 13's standalone
  dense mmvq weight rule `calc_nwarps_weight()` gives the launch it replaces 8 warps for Q8_0 with
  `K < 4096`, so a W=1 decode and a W>=2 verify reduced K differently and `--spec-type none` vs
  `--spec-type draft-mtp` diverged after ~200 tokens on Qwen3.6-35B-A3B (`n_embd` 2048).  The fusion now
  uses `calc_nwarps_weight(..., long_k = ne[0] >= 4096)` (a no-op for long K), so it is bit-identical to
  the unfused chain and stays ON: the delivered `-ncmoe` path is byte-pure (`none == n1 == n3 == n7`),
  MTP acceptance is unchanged (0.77654) and `MUL_MAT_ID`/`GATED_DELTA_NET`/`SSM_CONV` backend ops are 2/2.
- **Host-resident MoE decode** (block 06, r17 2026-09-27, **superseded by r19**): r17 exempted
  **`MUL_MAT_ID`'s src0**, the whole expert weight table, from the tiny-CPU-graph byte count, so the
  offloaded decode MoE graph (~120 one-token `MUL_MAT_ID` graphs per pass under `-ncmoe`) was classified
  "tiny" and serialised on one thread (Qwen3.6-35B-A3B Q8_0 `-ncmoe 99` `tg64` 13.8 -> 24.4 t/s).  r19
  found the measurement that justified it was confounded by the host's GPU-IRQ core pinning and replaced
  the exemption with the capped multi-threaded rule above.
- **Host-resident MoE experts (`-ncmoe`) under `-sm tensor` are now a first-class prefill configuration**
  (block 15, r16, 2026-09-27): the op-offload H2D staging ring is **on by default** (`GGML_SCHED_STAGE=0`
  opts out), the meta `stage_input` gained a **split branch** that gathers each device's slice into its
  ring slot on the copy stream (its slice-sum guard had compared against the whole-tensor `size` instead
  of `chunk_size_full`, so `-sm tensor` offload had silently fallen back to the slow splice), the
  compacted strided splice uses a **pinned gather + queued 1-D H2D** (`GGML_CUDA_SPLICE_GATHER=0` reverts)
  instead of the pageable `hipMemcpy2DAsync`, and split expert copies are on (`GGML_META_SPLIT_COPY=0`
  reverts).  All self-select from `-sm`/`-ncmoe`, so a stock `llama-server … -sm tensor -ncmoe N` needs no
  env vars.  On Qwen3.6-35B-A3B Q4_K_M (gfx1201 x1/x2, pp8192) the default 2-GPU `-sm tensor -ncmoe`
  beats upstream `84e76d8a2` at **every** offload level — **+91 %** at `-ncmoe 0` rising to **+148 %** at
  `-ncmoe 40` (all experts host) against upstream's only 2-GPU option (`-sm layer`) — and `tensor` beats
  `layer` by **+21 % → +33 %**.  Bit-identical output; `MUL_MAT_ID` / `FLASH_ATTN_EXT` green.
- **Host-resident MoE experts now stay pinned** (block 06, r15, 2026-09-27): with `-ncmoe` the scheduler
  op-offloads the used experts every ubatch, but `select_weight_buft`'s "avoid using a host buffer when
  using mmap" downgrade sent those uploads through the **pageable** model mapping — which on ROCm blocks
  the host inside `hipMemcpyAsync` (so the two cards' DMAs cannot overlap) and makes the meta backend's
  2-D spliced upload fault in `hipMemcpy2DAsync`.  The downgrade is now skipped for `MUL_MAT_ID` weights
  (default on, `LLAMA_MMAP_HOST_EXPERTS=0` restores it), which is **+83 %** on `-sm tensor -ncmoe 99`
  pp8192 (2794 -> 5104 t/s on 2x R9700) and **bit-identical** output.  Cost: the expert set is pinned,
  non-swappable RAM.  Same finding, independently, in GenerelSchwerz's `moe-cache` fork.
- **The derived kq-mask inputs are no longer read on the host** (block 15, r14, 2026-09-27, issue #53):
  the r10 fully-masked KV-group skip built its batch-wide bitmap by dereferencing the derived
  `tok_lo`/`tok_hi` from the CPU in `launch_fattn`, but the backend scheduler copies those host graph
  inputs to the compute backend, so the launcher sees *device* copies — a host read of device memory is
  `0xC0000005` in `ggml-hip.dll` on the first prefill ubatch wherever the allocation is not CPU-mapped
  (Windows/WDDM; Linux masks it) and a race against the in-flight copy elsewhere.  The window is now
  reduced cooperatively inside `flash_attn_kq_derived_blocks`; the bitmap is unchanged, so the skip is
  still exact.  The same fix replaces the direct `t->data` writes in the `derived`/`mask_hole`/
  `FLASH_ATTN_QSA` `test-backend-ops` initializers with `ggml_backend_tensor_set` (the first `derived=1`
  case segfaulted on Windows before any derived case ran).  `FLASH_ATTN_EXT` 6354/6354 and same-seed text
  identical across `{skip on, skip=0, derived=0}`; interleaved prefill A/B within noise.  See `WORKLOG.md`
  (2026-09-27 r14) and `patches/README.md`.
- **The tiny-CPU-graph single-thread heuristic no longer serializes CPU-offloaded FFN decode** (block 06,
  r13, 2026-09-27, issue #52): the heuristic added in r8 summed only the nodes' **output** activations
  when deciding "tiny", so a CPU-offloaded FFN chunk (four `MUL_MAT` nodes with ~16 KiB outputs that
  read tens of MiB of weights each token) ran on a single thread — 8.66 -> 1.74 t/s on the reporter's
  box, and 3.22 -> ~4.9 t/s on the gfx1201 + 9950X3D reproduction.  It now counts each node's non-view
  input tensors too, with `GET_ROWS` `src0` exempt (the embedding table is read only where gathered),
  so the host-resident-embedding spec-decode case the heuristic exists for keeps its single thread.
  Greedy same-seed output is byte-identical with the heuristic on or off.  See `WORKLOG.md` (2026-09-27
  r13) and `patches/README.md`.
- **The op-offload H2D staging ring is delivered** (block 06, r12, 2026-09-27, issue #50 / PR #51 by
  **@briansp2020**, whose redirect design the merge adopts): a prefill's host-resident MoE expert upload
  is issued on a per-device copy stream into a bounded slot ring and overlapped with the previous split's
  compute, with a link-calibrated width gate and a fallback that disables staging rather than degrading
  (measured `pp8192` +81 % on the x4 box, +31-81 % on his 55 GB/s box; same-seed text, MTP, `W=1..8` and
  perplexity all byte-identical with staging on or off).  The same block fixes the **`-sm tensor`
  op-offload path** — the meta device declared no `offload_op`, so `-ncmoe` had been executing the whole
  MoE on the CPU (523 -> 1823 -> 2742 t/s at `pp8192`/`ub8192`) — and completes 15 backends'
  `ggml_backend_i` initializer lists, a latent bug that had NULLed `graph_optimize` for
  metal/vulkan/hexagon/virtgpu.  **Block 06 is renamed** to `general system-operations bucket`.
  See `WORKLOG.md` (2026-09-27 r12) and `patches/README.md`.
- **The MMA FA prefill K/V store is typed again on the non-swizzled (AMD) path** (block 15, r9,
  2026-09-26, issue #47): upstream `1884824fd`'s swizzle refactor left the generic loader storing
  through `(char *) tile_KV + swizzle_bytes<…>`, which is address-identical but drops the `half2`
  alignment, so HIP split the 16-byte shared store.  Restoring the typed store under `if constexpr
  (!swz)` recovers +2-7 % on `hsk=256` prefill shapes and **+4.4 %** on the reporter's 27B
  UD-Q4_K_XL q8_0 pp4096 @ d40000 (872.6 -> 911.3 t/s; reporter 885.5 -> 922.0 = +4.1 %), decode flat,
  output byte-identical.  See `patches/README.md` (2026-09-26 block-15 r9) and `WORKLOG.md`.
- **The `mmb`/QSA/indexer campaign is folded into the delivery** (2026-09-25, release `r8`):
  the former 28-patch opt-in `archive/work/mmb-general/` set is now part of the 16 block patches — the
  `mmb` (bf16-WMMA dequant weight GEMM) core, the RDNA4 fragment port / per-arch tuning and the
  GDN/PLE/RMS prefill fusions in **block 08**; the catch-all system-operations fixes in **block 06**;
  `qsa3`, the fused indexer, HC16, `hc_gate_mix`, sparse MTP-draft and the MMVQ band in **block 15**
  (with block 14's pair stand-down and block 13's GLU stand-down).  Strict `git am` 16/16
  reproduces the full campaign tree `24bb0f5acb…` and the gfx1201 build is clean.
  `archive/work/mmb-general/` is kept as the historical verification record and the `apply-beta.sh` helper
  has been removed.  See [`archive/work/beta-integration/integration.md`](archive/work/beta-integration/integration.md).
- **The RDNA4 GQA-6 decode/verify FA band covers f16 (and, through its native arm, bf16) too
  (block 15, r5, 2026-09-25, issue #45 follow-up, reported by
  [@DanoPTT](https://github.com/DanoPTT)):** the
  head-256 GQA-6 `n_q <= 8` band no longer pays the tile kernel's 3x K/V re-fetch/dequantization;
  the whole band runs the WMMA kernel with the GQA group folded into one block (`ncols2 = 8`) and the
  KV split round-robin over a fixed `P` (independent of `n_q` and the KV length), so decode and every
  verify width reduce identically.  It started quantized-only (r4, reported by
  [@overdoingism](https://github.com/overdoingism)) and r5 extends it to the 2-byte types with a
  per-element-size config: native-quantized keeps `ncols1 = 4` / `P = nsm`, the 2-byte types take
  `ncols1 = 2` / `P = max(2, 3*nsm/4)`.  f16 kv 102400 verify widths 2.1-3.2x faster; 27B
  `draft-mtp n3` at ~30k +13 % (f16) / +14 % (bf16 native); plain f16 decode -2.5..-4.2 %.  **r6
  (2026-09-25) flips the bf16 native K/V arm to default ON** (`GGML_CUDA_FA_KV_NATIVE` unset now
  enables it, `=0` disables every native arm), since r5 made native bf16 the path to the band and the
  incoming beta prefill boosts outweigh its small prefill cost.  Default on
  (`GGML_HIP_FA_BAND_WMMA=0` opts out); prefill untouched.  See `patches/README.md`
  (2026-09-25 block-15 r4/r5/r6) and `WORKLOG.md`.
- **The qwen4exp CPU `hc_combine` reference is correct** (block 14, r3, 2026-09-25, issue #44):
  `ggml_compute_forward_hc_combine_f32` read `block_out` with a `t*ne[1]` row stride and `inject`
  with `t*hc`, but the model hands both over as multi-token tensors whose own `nb[1]` differs
  (`block_out` is a contiguous `[n_embd, nt]`, `inject` a view into the mix output with row stride
  `n_embd+hc`).  Every fused multi-token ubatch therefore read the wrong rows and a CPU-resident
  qwen4exp decoder layer emitted EOS as its first generated token; nt == 1 was accidentally correct.
  The reference now uses each tensor's own `nb[1]` (0 for a broadcast `ne[1] == 1`), mirroring the
  CUDA kernel, and is bit-identical at nt == 1.  Validated with an op-level CPU-vs-HIP oracle that
  fails 7/8 multi-token cases pre-fix and passes all 8 post-fix (gfx1100).
- **The wide-VDR MoE expert path is RDNA4/RDNA3_0-only** (block 10, r2, 2026-09-25): the
  `VDR_Q4_K/Q5_K/Q6_K_Q8_1_MMVQ_MOE` entry points were unconditional while only Q8_0 was arch-gated,
  so RDNA3_5 (gfx115x) ran the Q4_K/Q6_K experts — the Q4_K_M expert types — with the wide chunk the
  block-10 comment reserved for RDNA4/RDNA3_0.  `get_vec_dot_q_cuda()`/`get_vdr_mmvq()` now ignore
  `moe` on every other target in one place; base-16 MoE `draft-mtp n3` 0.73967 → 0.76484, 87.5 → 89.6 t/s.
- **Shared-NextN MTP heads are usable** (block 00, r13, 2026-09-22): a head with
  `nextn_shared_target_tensors` (no `token_embd`/`output` of its own, e.g. the qwen4exp
  `mtp-…-shared-Q8_0.gguf` sidecar) died every draft round on the M-RoPE `X < Y` check because the
  MTP driver inferred KV sharing from `ctx_other` alone.  `is_mem_shared` is now gated on the
  `gemma4-assistant` arch; it is an upstream bug (`04eb4c446`, #23398) folded into the block-00 base.
- **`--fit` works under `-sm tensor`** (block 6, r12, promoted from the former `beta/tensor-fit-fix/`, now `archive/work/tensor-fit-fix/`): upstream
  threw `not implemented for SPLIT_MODE_TENSOR` and swallowed it, so the default-**on** `--fit` was a
  silent no-op under tensor split.  The Meta device's accessors are now exposed and `common/fit.cpp`
  has a dedicated tensor path (per-device targets from `--fit-target`, a proportional split or an
  honoured `-ts`, then auto-`n_ctx` reduction and an `-ngl` binary search); an explicit `-c` is never
  overridden.  Re-validated on r11 before promotion (fit decisions, 7 end-to-end loads with zero
  out-of-memory and zero compute-buffer growth, byte-identical same-seed gate).
- **The compute reserve accounts for the reachable (packed) kq mask** (issue #42, block 15,
  2026-09-20): V3's derived kq mask is a per-*batch* optimization, so a 2-D M-RoPE image/audio batch or a
  multi-sequence batch allocates the packed mask (`n_kv*n_tokens*2` bytes), which the reserve — measured
  with the derived form on — did not contain.  At depth that mask is hundreds of MiB, so a deep-context
  image batch grew the compute buffer mid-run; under the default `--fit-target 256` that growth failed
  (`cudaMalloc failed: out of memory`, `failed to process mtmd chunk`) and the next request asserted.
  `sched_reserve()` now measures with the packed mask **when such a batch is reachable** (the new
  `kq_mask_packed_reachable()`: M-RoPE or `n_seq_max > 1`), so `--fit` counts it exactly where it can
  happen.  Same-seed output is byte-identical and throughput is unchanged; the reporter's M-RoPE model
  pays 8960 tokens / -4.4 % of fitted context, while a non-M-RoPE single-sequence model keeps V3's
  reserve untouched.  A failed buffer allocation now also invalidates the allocator's layout instead of
  asserting on a later graph.
- **Block 11 replays HIP graphs for split-MoE decode again** (issue #41, 2026-09-20): the pre-fill
  test keyed off `nodes[0]->ne[1]`, which is `n_expert_used` (10) on the expert tensor a one-token
  decode split starts with under `-ncmoe`, so every decode split was skipped as multi-token.  A new
  `ggml_cuda_graph_is_multi_token()` reads the real token count from `MUL_MAT_ID`'s `ne[2]` / a weight
  `MUL_MAT`'s `src1->ne[1]` (0 -> 50 warmups / 0 -> 687 replays, `tg` 10.6 -> 12.8 t/s on
  Qwen3.8-Flash-Next UD-Q4_K_XL, output bit-identical), and on HIP the exec is now
  destroyed/re-instantiated instead of updated, avoiding the ROCm <= 10.0 `hipGraphExecUpdate` leak
  (`GGML_HIP_GRAPH_FORCE_UPDATE=1` opt-out).
- **FA instance build-time fix** (blocks 06/13/15, 2026-09-18): the MMA instances are generated per
  `(ncols1, ncols2, head size)` and the head-512 ones are listed first in the backend source order,
  the tile instances per `(head size, KV type)`, and the fused-gate MMQ instances moved out of
  `mmq.cu`.  Clean `ggml-hip -j16` **323.4 -> 236.0 s (-27 %)**, identical instantiations and symbols,
  no runtime change; the order, not the split, is what delivers it.
- **gfx1100 (RDNA3_0) WMMA FA is capped at head 256** (r5, block 04, issue #30): the 2026-09-14
  RDNA4 #28102 config transfer shipped RDNA4-tuned rows *and* a lifted head cap to gfx1100, so head
  512 took WMMA where stock takes tile and lost up to 23 % of deep prefill (gemma-4-26B-A4B
  `pp2048 @ d98304` q8_0 661 -> 773 t/s, bf16 656 -> 851); head 256 keeps WMMA, a +44-52 %
  deep-prefill win.  RDNA4 (576) / RDNA3_5 (320) are unchanged.
- **gfx1100 (RDNA3_0) tensor split keeps the stock AMD FA `ncols2` rule** (block 04, issue #30):
  the 2026-09-14 split-aware hint (wider generic `ncols2` for tensor-split attention) was RDNA4-tuned
  and cost RDNA3_0 deep prefill (`pp100K` 667.5 -> 779.4 t/s on 2× RX 7900 XTX, stock 805.0; decode
  unchanged).  A single gfx1100 card is unaffected (it already took the AMD rule).
- `--fit` no longer SIGSEGVs with `--spec-type draft-mtp-adaptive` and a minimal per-tier MTP
  head (issue #38; block 01, one line in `common/common.cpp`).
- A clean HIP build no longer prints the ~10k FA "loop not unrolled" warnings
  (`-Wno-pass-failed`, block 15; no codegen change).
- Patches `patches/0000-…0015-…` apply with **strict 16/16 `git am`** (no 3-way
  fallback, whitespace-clean) via `scripts/apply-all.sh`.  `scripts/validate-set.sh`
  re-checks the artifact hashes, the strict apply and the applied tree against `release.json`.
- **`hybrid` is the default all-reduce**; `GGML_CUDA_ALLREDUCE=ce` selects the opt-in
  copy-engine (SDMA) 2-GPU mode and `=nccl` forces RCCL.
- **Greedy purity**: plain decode == `draft-mtp` verify for `--spec-draft-n-max <= 7`
  across the supported KV types.  Depths 8..15 are allowed with a visible notice (a
  verify wider than 8 rows switches kernel family); `> 15` is clamped (the recurrent
  rollback snapshot bound).
- Last full `test-backend-ops` on this cut (gfx1201): **18083/18083**, with
  `FLASH_ATTN_EXT` **5952/5952** and `FLASH_ATTN_QSA` **22/22**.

The **dated record of every change** (re-bases, block amendments, issue fixes,
measurements) is [`WORKLOG.md`](WORKLOG.md), newest first.  Per-block notes, env knobs
and server configuration live in [`patches/README.md`](patches/README.md); apply order
and the verification contract in [`MANIFESTS.md`](MANIFESTS.md); fork point and drift
policy in [`BASELINE.md`](BASELINE.md); the purity rulebook in
[`GREEDY-PURITY.md`](GREEDY-PURITY.md).

## Community Acknowledgements

This work is becoming a community effort and I'd like to offer special thanks to the
following users for the assistance in finding issues and offering solutions!

- https://github.com/1337hero
- https://github.com/bakon11
- https://github.com/briansp2020  (block-13 moe_weighted_reduction float4 remainder fix + block-14 MUL_MAT_ID pair-fusion layout gate, issues #19 and #18)
- https://github.com/eoprede
- https://github.com/overdoingism  (issue #45: the RDNA4 head-256 GQA-6 decode/verify flash-attention band, reported with the diagnosis, op-level data, the round-robin KV split idea and a working opt-in patch; the r4 block-15 band is built on that submission)
- https://github.com/pwilkin  (the `strix-halo` fork at https://github.com/pwilkin/llama.cpp/commits/strix-halo/, heavily adapted for the MMB bf16-WMMA dequant-weight GEMM work, now folded into the delivery — formerly `archive/work/mmb-general/`)
- https://github.com/tungel
- https://github.com/DanoPTT  (block-08 mul_mat+add through-view shape guard, PR #15; and issue #45 follow-up: the f16 verify-width diagnosis / the f16 + bf16 band coverage folded into block 15 in r5/r6, measured on their R9700)

I, and everyone else who benefits from this work, really appreciate you!

## Inspirational Works

While most of the work in this repository are original works of my own, there are
some significant portions, most notably around the prefill tuning, inspired by the
excellent work performed by the community of: https://github.com/halo-box/strix-llama.cpp

Thank you to all the maintainers of the Strix Halo Llama.cpp project

Of course none of this would be possible without the baseline that all of this rests
on, and that is the huge community over at https://github.com/ggml-org/llama.cpp

Many thanks to the llama.cpp team

## License

Same as llama.cpp (MIT).
