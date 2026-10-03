# wip/sched-moe-restage

A scheduler fix for MTP draft collapse on qwen4exp with the MoE expert cache. The expert weights get re-staged when a
later split has a second `MUL_MAT_ID` consumer of them. One `git am` patch, one file (`ggml/src/ggml-backend.cpp`, +23):

- `0001-sched-re-stage-expert-weights-for-a-MUL_MAT_ID-consu.patch` (r4 tree `714f94f0` -> `c5c716e7`; it also
  applies cleanly on r36 `c595f292` -> `21dfaf5c`).

## What goes wrong

**Symptom.** With `--spec-type draft-mtp` on Flash-Next (`-ncmoe 48`, `MOE_EXPERT_CACHE_MIB` set, default slots),
draft acceptance drops to about 0 for every slot after certain requests, and stays there:

- two concurrent chats, every time;
- single prompts of 278, 548 or 1406+ tokens.

A later prompt can bring it back; 1055 tokens did in our runs. Target output is never affected.

**Where the NaN comes from.** An eval callback on the draft context (local debug only) shows the drafter's input,
`mtp_h_input`, carrying NaN from row 1 of the catch-up decode. So the target's `t_h_nextn` already holds NaN.

**Why the target's export does.** For a prefill chunk, the unmasked MTP export (`mtp_export_defer` in `qwen4exp.cpp`)
recomputes the last layer's FFN on every row, next to the gathered logits tail. So `blk.N.ffn_{gate,up,down}_exps`
have two `MUL_MAT_ID` consumers. A `GGML_SCHED_DEBUG=2 -v` dump of a 381-token prefill:

```
SPLIT #143: ROCm0 # 3 inputs: [blk.47.ffn_gate_exps.weight] [blk.47.ffn_up_exps.weight] [blk.47.ffn_down_exps.weight]
  node #6985 (MUL_MAT_ID): ffn_moe_gate-47 (0K)   <- gathered tail, 1 row (cache decode band)
  ...
SPLIT #144: ROCm0 # 0 inputs
  node #7052 (MUL_MAT_ID): ffn_moe_gate-47 (9M)   <- export tail, 381 rows
  ...
```

**Why split #144 reads unstaged experts.** `ggml_backend_sched_split_graph` registers a weight as a split input only
when it first creates the copy. So split #144 has no inputs and reads the copy that #143 prepared for its own
routing. The expert cache took that copy over for the 1-row decode-band consumer and never filled it. The 381-row
consumer therefore reads experts that were never staged for its routing, or a stale arena remap. The result can be
NaN.

**Why the state carries over.** The NaN goes into the drafter's input and its indexer KV. That is why the collapse
carries across requests and slots, and why the prompt that clears it depends on cell layout.

**Controls.**

- Unsetting `MOE_EXPERT_CACHE_MIB`, or setting `GGML_OP_OFFLOAD_MIN_BATCH` large, fixes it.
- These don't change it: `LLAMA_MTP_SPARSE=0`, `LLAMA_QSA_OFF=1`, `LLAMA_GRAPH_REUSE_DISABLE=1`,
  `GGML_CUDA_DISABLE_GRAPHS=1`, `LLAMA_INDEXER_NOBLOCK=1`, `LLAMA_QSA_DENSE_SHORTCUT=0`, `GGML_CUDA_FA_MASK_SKIP=0`,
  `LLAMA_KQ_MASK_DERIVED=1`, `LLAMA_HC_MIX_BF16=0`.
- Upstream master has the same register-once logic, but neither the export tail nor the cache, and doesn't show it.

## The fix

In pass 5, when the copy already exists and the node is a `MUL_MAT_ID` reading weights through `src[0]`, the weights
are also registered as an input of the current split (once per split). That split then stages them for its own
routing before it runs. The routing comes from the previous split, so the ids are ready.

Graphs where each expert weight has a single consumer get no extra input, so plain decode and prefill are unchanged.
The cost is one extra staging of one layer's experts per MTP prefill chunk.

An earlier attempt picked the widest consumer within the split instead. It aborts at `GGML_ASSERT(id >= 0 && id <
n_expert)`, because in that case the wide consumer's ids are not computed yet when the split's inputs are staged.
Fixing it in the graph instead, for example by keeping both tails in one consumer, would also work, but would change
the logits path that the export tail was built to keep bit-identical.

## Validation

R9700 (gfx1201), Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS, shared Q8_0 MTP head, `-ncmoe 48 -ub 2048 -b 2048 -ctk q8_0
-ctv q8_0`, `GGML_SCHED_STAGE_SLOTS=16 GGML_SCHED_STAGE_MAX_MB=8192`.

The repro runs on a fresh server:

1. a probe: `/completion`, 18-token prompt, 200 tokens, temperature 0;
2. 278-, 548- and 1055-token prompts, each followed by the probe;
3. a concurrent pair of chats, then the probe again.

On r4 (`v16-a55e952b8-r4`, tree `714f94f0`), 64K context, MTP n3 `--spec-draft-p-min 0.5`, `MOE_EXPERT_CACHE_MIB=2048`:

| | r4 | r4 + fix |
|---|---|---|
| 278 / 548 / 1055 prompts, accepted / drafted | 0/87, 0/87, 21/27 | 20/28, 20/26, 20/26 |
| probe after each | 0/591 (19 t/s), 0/591, 131/177 | 130-131/176-177 (48-49 t/s) |
| concurrent pair, accepted / drafted | 1/1413 + 0/1791 (22 t/s total) | 290/442 + 399/529 (45.6 t/s total) |
| probe after the pair | 69/363 (28 t/s) | 131/176 (48.6 t/s) |
| probe output hash | 359abe1e9f09 | 359abe1e9f09 |

r4 + fix at 256K without MTP (`MOE_EXPERT_CACHE_MIB=4096`): greedy `c39d78416cd1`, chat 42.8 t/s, 259.6k-token prompt at
1732 t/s with the correct needle answer, and VRAM peak 31.8 GB.

On r36 (+ #91), before r37 / the rebase:

| | r36 | r36 + fix |
|---|---|---|
| NaN draft steps (debug build) | hundreds | 0 / 1985 |
| 278 / 548 / 1055 prompts, accepted / drafted | 0/87, 0/87, 21/27 | 20/28, 20/26, 20/26 |
| probe after each | 0/591 (18 t/s), 0/591, 130/175 | 130/173-175 (47-48 t/s) |
| concurrent pair | 0 accepted, ~22 t/s total | 299/421 + 427/479, 47.7 t/s total |
| probe output hash | 359abe1e9f09 | 359abe1e9f09 |

256K, without MTP (r36 + fix, `MOE_EXPERT_CACHE_MIB=4096`): greedy `c39d78416cd1`, chat 42.4 t/s, 259.6k-token prefill
at 1710 t/s and VRAM peak 31.8 GB. All are the same as without the patch.

256K, with MTP n3 `--spec-draft-p-min 0.5` (`MIB=2048`): greedy identical at 48.3 t/s (no-MTP 42.2), code 48-53
t/s, and drafting stays normal after the 32k and 259k prompts.
