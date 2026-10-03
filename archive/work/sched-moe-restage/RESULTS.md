# sched-moe-restage - RESOLVED (promoted in v16-a55e952b8-r5)

Promoted from contributor PR #96 (@briansp2020) directly into **block 06** (the scheduler half of the
MoE expert cache) as release `v16-a55e952b8-r5`.  The WIP record (`README.md` + the `git am` patch)
is kept beside this file.

## The bug

`ggml_backend_sched_split_graph` registered a weight as a split input only when it first created the
copy.  When the same host-resident expert weights fed `MUL_MAT_ID` in two splits, the later split hit
the existing-copy branch and was never listed as an input.  With `MOE_EXPERT_CACHE_MIB` armed, the
earlier 1-row decode-band consumer took that copy over (`moe_cache_take_over` aliases `weight_cpy`
for the device-side lookup and leaves `input_cpy->data` unfilled).  The wide consumer is outside the
cache band (`op->ne[2] > MOE_EXPERT_CACHE_MAX_TOK`), so `moe_cache_get_table` declines it and the op
reads the stale `input_cpy`.  qwen4exp's unmasked MTP export is exactly that wide second consumer; the
NaN it produced reached `t_h_nextn`, the drafter's KV and every draft slot, collapsing MTP acceptance
while target output stayed correct.

## The fix

In pass 5, when the copy already exists and the node is a `MUL_MAT_ID` reading WEIGHTS through
`src[0]`, register the weights as an input of the current split as well (once per split,
deduplicated).  The split then stages them for its own routing before it runs.  Single-consumer
graphs are unchanged.

## Independent validation on the maintainer's host (gfx1201 / ROCm 7.14)

R9700, Flash-Next UD-IQ3_XXS + shared `Q8_0` MTP head, `-ncmoe 48 -ub 2048 -b 2048 -ctk q8_0
-ctv q8_0`, `MOE_EXPERT_CACHE_MIB=2048`, `GGML_SCHED_STAGE_SLOTS=16 GGML_SCHED_STAGE_MAX_MB=8192`,
`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.5`.  Probe = 18-token prompt, 200
tokens, temp 0; the report's trigger is a 446-token prefill between the two probes.

| build | probe before | probe after the prefill |
|---|---|---|
| r4 `cd1485fd1` | 141/149 | **0/591** |
| r5 `b5ca42a92` | 141/149 | **141/149** |

The same sequence on IQ4_NL is 142/156 -> **0/591** on r4 and 142/156 on r5.  The probe's target text
is identical in every run.  Standard gates: warning-free build, `test-backend-ops -o MUL_MAT_ID`
931/931, dense 4B `1c5d32ac537d`, qwen4exp Flash-Next Q4_K_M (no MTP head) `622da9ec8ec2`;
`validate-set.sh` strict 16/16 with applied tree `c5c716e796b29d770902ff2aecfeaba42e80f487`.

## Promotion record

- block: **06** (`ggml/src/ggml-backend.cpp`), the scheduler half of the expert cache.  The block-06
  commit message gained the r5 amendment note; blocks 07-15 were re-based onto it.
- `patches/`, `rdna-boosts-all.patch` and `release.json` regenerated as `v16-a55e952b8-r5`.
- docs updated: `AGENTS.md`, `README.md`, `MANIFESTS.md`, `BASELINE.md`, `patches/README.md`,
  `WORKLOG.md`.
