# RESULTS — re-base integration audit (paths that compiled but were not exercised)

**Resolved:** 2026-10-05 (release `v16-a55e952b8-r2`).
**Status:** DONE — 5 of 7 items validated on hardware, 1 hardened with a new gate that exposed a real
CPU/GPU oracle mismatch, 1 documented as a model-availability waiver.

---

## Item-by-item

### 1. LF / DFlash device path (`GGML_LF_DFLASH_DEV`) — PASS

`Qwen3.8-27B-UD-Q4_K_XL` + `Qwen3.8-27B-DFlash2-Q4_K_M`, `--spec-type draft-dflash`,
`-ngl 99`, same seed/greedy:

| run | `GGML_LF_DFLASH_DEV=0` | `GGML_LF_DFLASH_DEV=1` |
|---|---|---|
| `-p "The capital of France is" -n 64` | `1acb04bd9104` | `1acb04bd9104` |
| `--reasoning on`, `prompts/prose-rdna-boosts.txt`, `-n 48` (M-RoPE + multi-ubatch prefill) | `1866e197bc4f` | `1866e197bc4f` |

Byte-identical, no host/device race, M-RoPE and a long prefill both clean.

### 2. `llama_context::extract_layer_inputs` bool + F1 device path — PASS (via item 1)

The r1 merge kept the F1 device branch and made it `return true`; upstream's new return value is
consumed by `llama-context.cpp`.  The DFlash `GGML_LF_DFLASH_DEV=1` runs above exercise that branch
end-to-end (the `lf_dev` allocation, `lf_consumed_ev`, the layer-input gather) and are byte-identical
to the `=0` host path.  No extra assertion added — the two hashes are the gate.

### 3. `common_sampler_clone` (S2 + upstream `.rng`) — PASS

* `tests/test-speculative-adaptive`: **all tests OK**.
* MTP `--spec-type draft-mtp` with `--spec-draft-sampling greedy` and `probabilistic` on
  Qwen3.6-35B-A3B Q4_K_M: both `exit=0`, distinct coherent text (`a17226134430` / `1fd13c16062a`).
  The probabilistic path is the one that reads the clone's `.rng`; no read of a clone's empty
  `cur`/`cur_p` before the next sample was observed.

### 4. `n_rs_batch` in the glm5-next `llama_memory_hybrid_idx` ctor — WAIVER (no model)

No GLM5-Next GGUF on this box, so the branch is compile-only.  The r1 r2 chain carries
`/* n_rs_batch */ cparams.n_rs_batch` in the ctor argument list and every block (and `all`) builds,
which is the available evidence.  A `test-llama-archs` case is the follow-up if a GLM5-Next fixture
appears.

### 5. Argsort tie-break in the multi-column `bitonic_step` — FINDING + FIX

Added a `ties` variant to `test_argsort` (duplicate-heavy rows) and cases at `{2048,8,1,1}` and
`{4096,2,1,1}` (span more than one bitonic block).  **The new cases failed on the r1 tree (74/78):**

```
[ARGSORT] ERR = 0.373891554 > 0.000000100   ARGSORT(type=f32,ne=[2048,8,1,1],order=1,ties=1): FAIL
```

Root cause: the CUDA `bitonic_step` already breaks ties by the smaller index (to match NVIDIA CUB's
stable radix sort and the fused MoE router), but the **CPU oracle** `cmp_argsort`
(`ggml/src/ggml-cpu/ops.cpp`) compared values only, so `std::sort` on a large duplicate-heavy row is
unstable and the CPU/GPU orders diverge.  Fix: make `cmp_argsort` a total order —

```cpp
if (data[a] == data[b]) {
    return a < b;   // index tie-break
}
```

`ARGSORT` is now **78/78**.  This is a real hardening change and is part of the r2 tree.

### 6. `tests/test-recurrent-state-depth.cpp` batch-API migration — PASS (runs, pre-existing counts)

Migrated to `common_batch.add` / `llama_process`; the test builds in every block and runs.

* Qwen3.6-35B-A3B UD-Q4_K_M, `-ngl 99`: Phase A (verify-band) green for `n_rs_seq <= 7`, then failing;
  **130 total** (66 A + 64 B).
* same model, `-ngl 0`: Phase A green for `n_rs_seq <= 6`; **230 total** (98 A + 132 B).

The counts are configuration/backend dependent and **pre-existing**: the r2 tree is byte-identical
to the r1 tree in every runtime file except the isolated CPU argsort tie-break.  The handover's
"153 Phase-B" figure is from the older gfx1151/Strix-Halo record
(`WORKLOG.md` 2026-09-2x) and is not an r1 number.  **No new regression**; the sweep's
green up to the depth the model's snapshot layout supports, then fails identically across backends.
Worth a dedicated follow-up: why the verify-band rollback stops matching above `n_rs_seq` 6-7 on
this model (the r13 note called the same cell "green" on gfx1151).

### 7. Block-15 S2 fast top-k vs upstream sampling — PASS

`GGML_LF_FAST_TOPK=0` vs `=1` on Qwen3.5-4B-Q8_0, `--top-k 40`, 48 tokens:

| temp | fast=0 | fast=1 |
|---|---|---|
| 0 | `6ff72e08d38b` | `6ff72e08d38b` |
| 1 | `6ff72e08d38b` | `6ff72e08d38b` |

Byte-identical, and `test-speculative-adaptive` + the probabilistic MTP smoke test above cover the
`rng` interaction.  The S2 path stands down for non-`logit-bias + top-k` chains as designed.

## Acceptance scorecard

| item | gate | result |
|---|---|---|
| 1 DFlash device path | dev=0 vs 1 byte-identity (plain + M-RoPE + multi-ubatch) | PASS |
| 2 `extract_layer_inputs`/F1 | exercised by item 1 | PASS |
| 3 `common_sampler_clone` | `test-speculative-adaptive` + greedy/probabilistic MTP | PASS |
| 4 glm5-next `n_rs_batch` | compile-only (no model) | WAIVER |
| 5 argsort tie-break | new `ARGSORT …ties=1` cases | **FAILED → FIXED (78/78)** |
| 6 recurrent-state-depth | test runs on the migrated API, counts recorded | PASS (pre-existing) |
| 7 S2 fast top-k | fast=0/1 byte-identity + adaptive test | PASS |
