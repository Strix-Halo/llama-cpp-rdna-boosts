Hi Brian, thanks for this one and for the earlier MTP work.

We applied #98 on top of the r7 tree (`v16-a55e952b8-r7`, the release that folds the issue #93
staging fix) and spent a while trying to reproduce the MTP draft context memory saving. The switch
itself works: with `LLAMA_MTP_DRAFT_OP_OFFLOAD=0` the draft context is created with `op_offload = 0`,
and the target context is untouched. But on our R9700 we cannot get the draft context to reserve the
2,550 MiB, with or without the switch, so we would like you to re-measure on r7 and tell us what you
see.

Here is exactly what we tried and how we measured it.

**Draft context reserve.** We read the `sched_reserve: ROCm0 compute buffer size = ...` lines from a
verbose load (`llama-cli --verbosity 4`), which prints one pair per context (target prefill and
decode, then draft prefill and decode). With your flags
(`LLAMA_MTP_SPARSE=0 GGML_SCHED_STAGE_SLOTS=16 GGML_SCHED_STAGE_MAX_MB=8192 -ctk q8_0 -ctv q8_0
--spec-draft-p-min 0.5`), the target context reserves 2984.30 MiB and the MTP draft context reserves
548.06 MiB. That 548.06 is identical with `LLAMA_MTP_DRAFT_OP_OFFLOAD` unset and with it set to 0.

**Post-load VRAM.** We started `llama-server` at 256K (`-c 262144 -b 2048 -ub 2048`, MTP n3,
`-otd exps=CPU`, `--fit off`), waited for `/health` to return ok, and summed
`/sys/class/drm/card{1,2,3}/device/mem_info_vram_used` before sending any request. Both
configurations read 18065 MiB.

**Peak VRAM during a run.** We sampled the same sysfs sum every 0.3 s across a full prompt plus
generation, and kept the maximum. Default gating: 9322 MiB both ways.
`GGML_SCHED_STAGE_TABLE_REF_MB=0` to remove the table size scaling: 12974 MiB with offload on versus
13147 MiB with it off. `GGML_SCHED_STAGE_MIN_TOKENS=0` to stage every batch including decode: 12930
MiB both ways.

**Where the op goes.** With `GGML_SCHED_DEBUG=2 -lv 6` the draft's prefill `MUL_MAT_ID` is placed on
ROCm0 (`ffn_moe_gate-48 [ROCm0]`, with a `ROCm0#blk.48.ffn_gate_exps` weight copy), and its decode
`MUL_MAT_ID`s stay on CPU because `ne[2] < 32`. So the prefill op is being offloaded. What we cannot
find on r7 is that copy showing up in the draft context's reserved compute buffer, which is why the
switch has nothing to remove on our side.

Output is byte identical in every configuration (`6145e62c557a` on a ten token greedy run), so
nothing here is unsafe, it just does not move memory for us. One difference worth noting: we tested
with the UD-IQ3_XXS model and its shared Q8_0 head, while your table used the GSQ-RCO IQ3_XXS build.
We would be surprised if that mattered for the draft reserve, but it is the one thing that is not
identical.

Could you re-run your 256K table on r7 (or later) and paste the draft context's `compute buffer size`
line with the switch on and off, plus the post-load VRAM number? We would like to know whether your
r6 number was a tree difference or a setup difference. If it still saves on your box we are happy to
take it, since 2.5 GiB with no speed cost is worth having. If it does not move for you either, we
will keep it on the shelf.

For reference, the two numbers we are comparing:

| MTP draft context | offload on | `LLAMA_MTP_DRAFT_OP_OFFLOAD=0` |
|---|---:|---:|
| compute buffer, your flags | 548.06 MiB | 548.06 MiB |
| post-load VRAM at 256K | 18065 MiB | 18065 MiB |

Thanks again, and no rush.
