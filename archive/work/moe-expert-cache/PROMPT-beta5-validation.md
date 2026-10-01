# Kick-off prompt — beta5 validation + strip + qwen35moe opportunity

Paste the block below as the first message of a fresh session.

---

We are in `~/llama-cpp-rdna-boosts` on branch `promote-moe-caching`.  Read
`archive/work/moe-expert-cache/HANDOVER-2026-10-02-beta5-validation-gates.md` in full, then carry out its §1
objective.  It supersedes nothing in AGENTS.md — the Pushing policy, the Default-on policy, the
`llama-cli --single-turn` rule and the `-t 8`/no-parallel-benches rules all still apply.

Summary of where we are: beta5 (`~/llama-fold` branch `beta4` = `21de1b20b`, patch at
`archive/work/moe-expert-cache/beta5-fix-registration-and-head-zero.patch`, built at `/tmp/reorg`) is beta4 plus
two fixes in `ggml/src/ggml-cuda/moe-expert-cache.cu` (block 13): (1) register the expert table on the
device-gather path too, which stops the qwen4exp post-prefill decode collapse; (2) replace the ~3x
per-gather MMQ tail pad with a one-time zero of each expert slot's first bytes.  On a single R9700 the
qwen4exp middle ground is now `pp8192 2401 + tg1024 36.5` at `-b 2048 -ub 2048 -ncmoe 48 -sm layer
MIB=12288` (was 641/8.5).  It is neutral for qwen35moe.

Do these, in order, and keep me posted with evidence (not summaries):

1. **Strip the scaffolding** exactly as listed in the handover §2 (the three `GGML_META_GATHER_*` env
   knobs, `moe_cache_gather_pad_kernel` + its launch, the own-head tail-guard block, the `pad` and
   `zero_fill` kernel parameters, the stale pad comments) — keeping only the two fixes.  Verify with
   `grep -rn GGML_META_GATHER ggml/src/ggml-cuda/` (should be empty) and a warning-free build.
2. **Run the full gate set** in handover §4 on the cleaned tree, recording every number.  Do not skip
   the Q4_K_M prefill matrix (the do-not-break list) or the multi-GPU `-sm tensor` "must beat single
   GPU" check.
3. **Resolve the gather-vs-staging gate** (handover §4.3) so qwen4exp uses the gather by default at
   `-ub 2048` (today the beta4 width gate sends it to staging, 641 vs 2401) without regressing
   qwen35moe/Q4_K_M.  A model-aware or probe-based choice is fine; a blanket ungating is not.
4. **Investigate the qwen35moe opportunity** (handover §5): find out *why* qwen35moe never collapsed
   (instrument the registration order vs qwen4exp), whether the gather/arena can be made to help it,
   and whether it is simply already at its single-GPU ceiling.  A documented "qwen4exp-specific fix"
   with evidence is an acceptable answer.
5. **Produce the community config guide** of handover §6 — the reproducible method plus the measured
   single-GPU oversized-Q8_0-MoE config table — so the community can converge on a good setting.

Do not tag, publish a GHCR image, or merge to `main`.  Commit to `promote-moe-caching` and push to this
repo's `origin` only; back up early and often (storms/outages are a real risk this week).

---

## One-liners you will need

```bash
# build the working tree (ccache: ~5 s for a one-file change)
cd /tmp/reorg && BUILD_DIR=build-rocm-b3 EXTRA_CMAKE_FLAGS="-DCMAKE_HIP_FLAGS=" ~/bin/build-llama-rocm-714
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib

# the combined single-GPU middle-ground measurement (one process; gather forced for qwen4exp)
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=12288 MOE_EXPERT_CACHE_DEVMAP=1 \
  GGML_SCHED_STAGE_MIN_TOKENS=999999 \
  /tmp/reorg/build-rocm-b3/bin/llama-bench \
  -m /llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf \
  -ngl 99 -ncmoe 48 -sm layer -fa 1 --lazy-mode auto --load-mode none -t 8 \
  -p 8192 -n 1024 -b 2048 -ub 2048 -r 2

# Coherence (NEVER omit --single-turn --no-display-prompt)
HIP_VISIBLE_DEVICES=0 MOE_EXPERT_CACHE_MIB=12288 MOE_EXPERT_CACHE_DEVMAP=1 \
  GGML_SCHED_STAGE_MIN_TOKENS=999999 \
  /tmp/reorg/build-rocm-b3/bin/llama-cli \
  -m /llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf \
  -ngl 99 -ncmoe 48 -sm layer -fa 1 --lazy-mode auto --load-mode none -t 8 \
  -f archive/work/moe-expert-cache/coherence-essay-prompt.txt -n 12000 -c 16384 --reasoning off \
  --seed 42 --temp 0 --single-turn --no-display-prompt > /tmp/coh.txt 2>&1
grep -c '////' /tmp/coh.txt   # must be 0
```

Build the pre-fix comparison tree if needed with
`git -C ~/llama-fold worktree add /tmp/prefix f29745280` then the same build line with
`BUILD_DIR=build-rocm`.
