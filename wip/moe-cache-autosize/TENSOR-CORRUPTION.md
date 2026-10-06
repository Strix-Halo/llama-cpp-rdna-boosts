# `-sm tensor` corruption on a wide prefill ubatch — ROOT-CAUSED, FIX VALIDATED (not shipped)

**Status: ROOT-CAUSED AND FIXED 2026-10-06 — the fix is validated but NOT in the delivery** (the `wip/`
rule): `wip/moe-cache-autosize/stage-redirect-fix.patch` (1 file, +15/-1).  TODO #41.  The r15 release
(`v16-a55e952b8-r15`) fixes the `-sm tensor` CPU fallback + the slow cache (TODO #40), *not* this.

**One line:** the op-offload H2D staging ring's `-sm tensor` consume path repointed the device tensor at
the ring slot instead of copying the slot into the tensor's real buffer; the host-offloaded expert
consumer does not read through `simple_tensor->data`, so the stale buffer was used and the layer was
silently corrupted.  Copy instead of repoint -> coherent, MTP works, staging win retained (+14 % prefill).

Read §0 first.  Everything else is the trail: the exact repro, the bisected threshold, the gate, the
ruled-out list (with commands), the false lead that cost the r15 session, the method trap that cost this
one, and what is still open (3 GPUs; the generic ring's own redirect).

---

## 0. RESOLVED (2026-10-06): the staged slot was fine, the *redirect* was not

**Root cause: the op-offload H2D staging ring's `-sm tensor` consume path repointed the device tensor
at the ring slot (`simple_tensor->data = chunk.slot`) instead of copying the slot into the tensor's real
buffer.  The consumer of a host-offloaded expert table does not read it through `simple_tensor->data`, so
the slot was ignored and the *stale real buffer* was used for the whole layer.**  Fix = copy the slot
into the real buffer (`stage_d2d`), one line of policy, in `wip/moe-cache-autosize/stage-redirect-fix.patch`.

### The chain that got us there

1. The activation gate, not a width limit.  `sched_stage_min_tokens()` calibrates the staging crossover
   from the measured H2D link and `sched_stage_min_tokens_for_bytes()` scales it by the host table size
   (`base * bytes / 144 MiB`).  Under `-sm tensor` the meta split *shards* the table, the scaled
   threshold drops below the prompt width, and staging turns **on**; under `-sm layer` the whole table
   keeps it above, so `-sm layer` never stages.  That is the entire reason `-sm layer` looked immune.
   Pinning the gate proves staging is the trigger:

   | IQ3_XXS, `-sm tensor -ub 4096` | `////` | MTP acc |
   |---|---:|---:|
   | default (gate -> staging **on**) | 1 | 1.00 |
   | `GGML_SCHED_STAGE_MIN_TOKENS=999999` (never stage) | 0 | 3.88 |
   | `GGML_SCHED_STAGE_MIN_TOKENS=0` (always stage, at `-ub 2048` too) | 1 | 1.00 |

2. Staging under `-sm tensor` is the **meta backend's `stage_input`**, not the ring.  Instrumented
   (`GGML_STAGE_INTROSPECT=1`): 96 `stage_input ENTER` for the prefill, all `gather=1`,
   `n_chunks=512`, `chunk_size_full=524800`, and 192 redirect / 192 restore with `pending=1` (no slot
   accumulation).  The geometry checks out exactly: device 0 = dim-1 rows `[0,256)`, device 1 =
   `[256,640)`, of each 512-expert block (per-device bytes 209920 / 314880 summing to 524800).

3. So the **staged bytes were correct**, and the defect had to be the redirect.  The decisive A/B
   (same binary, same flags, only the consume changed):

   | IQ3_XXS, `-sm tensor -ub 4096`, staging ON | `////` | acc | verdict |
   |---|---:|---:|---|
   | redirect (upstream) | 1 | 1.00 | CORRUPT |
   | **copy slot -> real buffer (the fix)** | **0** | **3.88** | **COHERENT** |
   | staging off (reference) | 0 | 3.88 | coherent |

   Copying the slot into the real buffer gives byte-identical behaviour to not staging at all, with MTP
   working.  That is conclusive: the gather is right, the *pointer* was the problem.

### Why the redirect is silently wrong

Some consumer of the split expert weight does not obtain its address from `simple_tensor->data`.  The
stage-guard comment in `ggml-backend-meta.cpp` explains its own design around a different assumption
("the split's consumers read the per-device simple tensors"), and for the **dense** path they do -- which
is why the redirect was never caught.  For the **host-offloaded `MUL_MAT_ID`** table something else
resolves the address (a prepared/cached descriptor or a cache-table keyed on the *original* buffer; the
`moe_cache_*` hooks are keyed on `(input_cpy buffer, expert_bytes)`, but `MOE_EXPERT_CACHE_MIB=0` still
reproduces, so the cache is not it).  **Identifying that consumer is the remaining open question** -- it
would let the redirect be restored safely (it is worth one D2D copy per ubatch); until then, copy.

### The fix

`wip/moe-cache-autosize/stage-redirect-fix.patch` (1 file, +15/-1, `ggml-backend-meta.cpp`): in the
staged-input consume loop, `stage_d2d()` the slot into `simple_tensor->data` and leave the pointer alone.
`GGML_STAGE_META_REDIRECT=1` restores the old redirect **for A/B only**.

Validated (2 x R9700, gfx1201, greedy `--seed 42`, coherence = no `////`):

| run | `////` | MTP acc | prefill | decode |
|---|---:|---:|---:|---:|
| IQ3_XXS 16k `-ub 8192` `-n 1024` | 0 | 3.77 | 1073 | 56.1 |
| IQ3_XXS 3.7k `-ub 4096` `-n 32` | 0 | 3.88 | -- | -- |
| IQ4_XS 16k `-ub 8192` `-n 1024` | 0 | 3.77 | 1000 | 43.6 |
| IQ4_XS 16k `-ub 4096` `-n 1024` | 0 | 3.79 | 676 | 62.0 |
| IQ4_XS 16k `-ub 8192`, staging OFF (for contrast) | 0 | 3.77 | 875 | 48.0 |
| `MOE_EXPERT_CACHE_MIB=0` + staging + fix, IQ3_XXS `-ub 4096` | 0 | 3.10 | -- | -- |
| `-sm layer` `-ub 8192` (must be untouched) | 0 | 3.70 | 1159 | 50.8 |

**The staging win is +14 % prefill, not +75 %.**  The old redirect's 1557-1572 t/s prefill was itself a
corruption artifact (it was not doing the work) -- exactly the trap the block-06 record already documents
for the device gather ("the gather's apparent 2-4x prefill win over staging was an artifact of the
corruption").  **A throughput A/B on this path is not evidence unless the staging-on side is
coherence-gated.**

### Still open after the fix

* **3 GPUs: `-sm tensor` prefill is corrupt even with staging OFF.**  IQ4_XS 3 x R9700 `-ub 4096`
  16k: staging off -> `////` x1, acc 1.00, 1594 t/s; staging on + fix -> `////` x1, acc 1.00, 996 t/s.
  A *different* bug, not touched here, and not previously recorded.  (`-ub 8192` on 3 GPUs first OOMs at
  context creation: `Meta() buffer 11257 -> 11429 MiB`, `cudaMalloc failed` -- see TODO #42.)
* **The generic ring's redirect is equally unvalidated.**  `sched->stage_mode` defaults to **1**
  (redirect `input_cpy->data`) and applies to `-sm layer`; forcing staging there aborts
  (`hipMemcpyAsync ... invalid argument` in `ggml_backend_cuda_buffer_set_tensor`), so it has never run
  end-to-end either.  Either validate it coherence-wise or default it to `stage_mode = 0`.
* **Which consumer ignores `simple_tensor->data`** (above).
* **The `-sm layer` abort** (the `set_tensor` invalid-argument) is still unexplained.

### Method note (this cost the most time)

`GGML_LOG_WARN` from ggml during graph compute is **filtered** by `llama-cli`'s default level, so an
instrumented build can look like "the hook is never called".  Use `fprintf(stderr, ...)` in this path (the
warnings seen at startup come from before `llama_log_set` installs the filter).

## 1. TL;DR

**The block-06 H2D staging ring corrupted every layer it staged, and the defect was the pointer, not the
data** (see §0).  The original symptom — a single prefill ubatch **wider than ~3600 tokens** under
`-sm tensor` makes the **target model itself** produce garbage (`////`) — is real, but ~3600 is not a
width limit: it is where the calibrated activation gate turns staging **on**.

* staging off (`GGML_SCHED_STAGE_MIN_TOKENS=999999`) -> coherent + working MTP at every `-ub`;
* **staging on with the fix** (`stage_d2d` instead of the redirect) -> coherent + working MTP, and faster
  than staging off (IQ4_XS 16k: prefill **1000 vs 875**, decode 43.6 vs 48.0);
* staging on upstream (redirect) -> `////`, MTP acc 1.00.

`-sm layer` looked immune only because the gate scales with the host table size, which is larger there,
so staging stays off; forcing it on aborts (a separate, unexplained `set_tensor` failure).  Pre-existing
(the r14 binary reproduces it), unrelated to the expert cache.

It is a **silent corruption**: the model answers, it is just wrong.  Gate any change here on output
coherence (`////`=0) **with staging on**, never on throughput.

## 2. Exact repro

Model: Qwen3.8-Flash-Next **UD-IQ4_XS** (`/llm/models/Qwen3.8/Flash-Next/IQ4_XS/`,
`...-UD-IQ4_XS-00001-of-00003.gguf`) + its MTP head
(`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`). 2 GPU, cache auto.

```bash
# a ~3800-token prompt (build it by repeating prompts/code-python.txt; ~3.8 chars/token).  NOTE: the
# base file is 2025 chars (~533 tokens), so it is `base*7`, NOT a large multiplier -- `base*266` is
# ~141 000 tokens and fails with "exceeds the available context size".
python3 - <<'PY'
base = open('/home/stew675/llama-cpp-rdna-boosts/prompts/code-python.txt').read()
open('/tmp/pl_3800.txt','w').write(base*7 + "\n\nWrite a short summary.")   # ~3731 tokens
PY

cd ~/llama-r13   # the r15 16-block chain
D=/llm/models/Qwen3.8/Flash-Next/IQ4_XS
M=$D/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
T=$D/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf

HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib \
./build/bin/llama-cli -m $M -md $T -ngl 99 -sm tensor -ncmoe 48 \
  -fa on -ctk q8_0 -ctv q8_0 -t 8 -c 32768 -b 4096 -ub 4096 \
  --lazy-mode off --load-mode auto --spec-type draft-mtp --spec-draft-n-max 3 \
  -f /tmp/pl_3800.txt -n 32 --seed 42 --temp 0 --reasoning off \
  --single-turn --no-display-prompt --no-warmup > /tmp/out.txt 2>/dev/null
grep -c '////' /tmp/out.txt     # 8 = CORRUPT;  0 = coherent
```

`--no-warmup` is not required but keeps the run short. The signal is `////` in stdout (the delivery's
own degenerate-output marker — the same one the `c32K/c128K` coherence gate uses).

**IQ3_XXS reproduces it identically** (2 GPU, 79 GB, same flags) — so it is **quant- and
model-independent**:

| IQ3_XXS | `-sm tensor` | `-sm layer` |
|---|---|---|
| `-ub 4096` | `////`=8, acc 1.00 ❌ | `////`=0, acc 3.75 ✅ |
| `-ub 2048` | `////`=0, acc 2.82 ✅ | `////`=0, acc 3.33 ✅ |

IQ3_XXS is also the **faster cycle** for this investigation (79 GB vs 96 GB, and it has its own
`mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` head in its directory).

## 3. The threshold (bisected)

`-sm tensor`, `-ub 4096`, one prefill ubatch, `-c 32768`:

| prompt tokens | `////` | note |
|---|---:|---|
| 6 | 0 | coherent (single tiny ubatch) |
| 400 – 2800 | 0 | coherent |
| 3000 – 3600 | 0 | coherent |
| **~3800** | **8** | **CORRUPT** |
| 4000, 5000, 6000, 7000, 16000 | 8 | CORRUPT |

The trigger is the **ubatch width**, not `-ub` and not the ubatch *count*:
* `-ub 4096` + a 3800-token prompt = **one** ubatch → corrupt.  So it is not the cross-ubatch bridge.
* `-ub 8192` + a 3800-token prompt → the ubatch is still 3800 → also corrupt (same threshold).
* `-ub 2048` + a 3800-token prompt = two ubatches (2048 + 1752) → **coherent**.
* `-b 8192 -ub 2048` → ubatch 2048 → coherent.  It is `-ub` (the ubatch), not `-b`.

Narrow it further first (the exact boundary is not known; ~3600 works, ~3800 breaks).  Candidates worth
checking against a plausible constant: 3584 (= 4096 − 512), 3640, 3712.

## 4. `-sm layer` is NOT affected — but only because it never stages

Same prompt, same run flags, only `-sm tensor` → `-sm layer`:

| config | `////` |
|---|---:|
| `-sm tensor -ub 4096` | **8** |
| `-sm layer  -ub 4096` | **0** |
| `-sm layer  -ub 2048` | 0 |
| `-sm tensor -ub 2048` | 0 |

This was originally read as "the bug lives in the Meta-split prefill path".  It is actually an artefact of
the **activation gate** (§0): the meta split shards the table, so the size-scaled threshold drops below
the prompt width and staging turns on; `-sm layer` sees the whole table, its threshold stays high, and it
never stages.  **Forcing** staging under `-sm layer` (`GGML_SCHED_STAGE_MIN_TOKENS=0`) makes it fail too —
with an abort rather than silence.  So `-sm layer` is a control for *the gate*, not for the split; the
real control is `GGML_SCHED_STAGE_MIN_TOKENS` at a fixed `-sm`.

## 5. Ruled out (do not re-test these)

**The cause is found — see §0.  The controls below establish what it is *not*, which is what narrowed it
to the staging ring; keep them as regression checks, but do not re-run them hoping to find the bug.**

All at `-sm tensor -ub 4096`, 3800-token prompt:

| ruled out | command / evidence | result |
|---|---|---|
| the expert cache | `MOE_EXPERT_CACHE_MIB=0` | still `////` |
| a r15 regression | the r14 binary (`~/llama.cpp/build-rocm/bin/llama-cli`) | still `////` |
| the derived KQ mask | `LLAMA_KQ_MASK_DERIVED=0` | still `////` |
| QSA sparse FA | `LLAMA_QSA_SPARSE_FA=0` (and combined with the above) | still `////` |
| GDN chunked prefill | `GGML_CUDA_GDN_CHUNKED=0` | still `////` |
| the ubatch *count* / cross-ubatch bridge | a 4k prompt is one ubatch and still breaks | n/a |
| `-b` (batch) | `-b 8192 -ub 2048` is coherent | n/a |
| `-c` (context) | broken at `-c 32768` and at `-c 8192` | n/a |
| leftover processes / VRAM pressure | `pgrep -cf 'bin/llama'` = 0, `rocm-smi --showmemuse` = 0 %, 172 GB RAM free | n/a |
| a load-time OOM (a separate, also-open issue) | the failing loads here complete; the arena prints normally | n/a |

**Still un-ruled-out environment switches worth trying next** (all cheap): `GGML_SCHED_DEVGATHER`,
`LLAMA_MMAP_HOST_EXPERTS=0`, `GGML_SCHED_EVENTS=0`, `-fa off` if the model allows it, `-ctk/-ctv f16`
(to remove the q8_0 KV from the picture), and `--load-mode none` (to see whether the pageable path
changes it).

## 6. The false lead: "MTP never accepts" (read this so you do not re-chase it)

What the r15 session first saw was `#mean acc len = 1.00` / `draft_n_accepted = 0` of 3063 under
`-sm tensor` with a wide prompt, and it looked like an MTP-specific bug.  It is not:

1. The **target's output is already corrupt** (`////`) at the same width, so the target's hidden state is
   garbage and every draft is correctly rejected.  Fixing the corruption should restore acceptance.
2. **The draft side is provably fine.**  With a *narrow* prompt (6 tokens) at `-ub 4096`, the draft
   candidates are **byte-identical** to `-ub 2048`:
   `1144 ' need' (1.000)`, `1902 ' must'`, `16546 'need'`, `4087 ' answer' (0.781)`, `310 ' to'`,
   `5707 ' respond'`, and the acceptance is normal (3.50).
3. **Acceptance is not a verdict below `-n 2000`** (`mtp-adaptive-methodology.md` rule 0).  At `-n 32`
   *every* config, including working ones, reads `1.00`.  Do not gate this bug on acceptance; gate it on
   **`////`=0**.

## 7. Hypotheses left (ordered by how cheap they are to test)

Hypothesis 1 below was **the cause** and is confirmed (§0).  What remains is *why* the ring's data is
wrong, which is now a bounded code question rather than a hunt:

1. ~~**The wide-ubatch staging path.**~~  **CONFIRMED** — the block-06 op-offload H2D staging ring is
   corrupt whenever active (`GGML_SCHED_STAGE_MIN_TOKENS` A/B, §0).  It is **not** width-specific: the
   ~3600 tokens was only the calibrated activation gate.
2. **The gather's pack layout vs the consuming tensor's strides (prime suspect).**
   `ggml_backend_meta_stage_input()` gathers each device's slice to `chunk.slot` with
   `src/stride/n_copies` (`ggml_backend_cuda_stage_gather`), then `graph_compute` repoints
   `simple_tensor->data = chunk.slot`.  If the gather's compacted layout does not match the strides the
   kernels index the simple tensor with, every staged weight is misread — silently, and only while
   staging is on.  Compare `chunk.stride`/`chunk.size` against the simple tensor's `nb` for one MoE
   op and dump the slot's first/last bytes against the host table.
3. **`input` vs `input_cpy` geometry.**  The ring derives `size`/`chunk_size_full`/`n_chunks` from
   **`input_cpy`** but gathers from **`input->data`**.  Any shape/stride difference between them is a
   wrong gather geometry.  Cheap to check with an assert/log.
4. **`stage_free_ev` timing.**  `ggml_backend_meta_stage_guard`'s destructor records the free event
   after the child graphs are *enqueued* (not completed) and cycles only `stage_n_slots` (default 8)
   slots, so a slot can be re-gathered while its previous consumer is still reading it.  Verify the
   event is recorded on the stream the consumers actually run on, after their completion.
5. **`h2d_pin_buffer`/`h2d_scratch` sizing.**  The ring slot is reserved with `copy_bytes`
   (`n_chunks * chunk_size_j`) and the gather re-requests `h2d_stage_buffer(slot, total)`; a slot/pin
   sized for a *different* tensor (the meta context is shared across all staged uploads) would overrun.
   Check the resize/reallocation behaviour and whether two tensors can share a slot index.
6. **Numerics, not logic.**  Still worth one cheap check: compare wide-ubatch logits with staging on vs
   off (same seed, greedy) to see whether the corruption is a truncated/NaN region (logic) or a drift
   (numerics).

## 8. Performance context (why this bug matters for the R9V comparison)

A large `-ub` is the **prefill lever** when the experts are host-resident (the staging ring amortizes
over the ubatch). 2 GPU IQ4_XS, 16k prompt, coherent-only numbers:

| config | prefill | decode (n=2048) | coherent |
|---|---:|---:|---|
| `-sm layer -ub 2048` | 379 | **67.1** | yes |
| `-sm layer -ub 8192` | **1040** | 42.8 | yes |
| `-sm tensor -ub 8192` | **1570** | — | **no** (this bug) |
| Reddit R9V reference (theirs is slower hardware) | 1166 | 57.3 | yes |

The decode drop at a large ubatch is the **same knob**: the wider ubatch permanently reserves a bigger
compute buffer (3810 → 11339 MiB), the auto cache arena yields (42130 → 26707 MiB under `-sm layer`;
16908 MiB under `-sm tensor`), and residency — hence decode — falls.  That tension is TODO #42; this
document is only about the corruption (TODO #41).

**Note on the comparison:** the R9V numbers come from the Reddit report on a 96 GB DDR4-3200 / 3900X
host, on the same UD-IQ4_XS, via an OpenAI server with 5 requests per context; ours are `llama-cli`
`[ Prompt | Generation ]` medians on much faster hardware over one run.  Treat the shape as indicative
and re-measure like-for-like before quoting a win.

## 9. Practical traps recorded this session (save yourself the time)

* **Leftover processes.**  A killed/timed-out `llama-cli` keeps ~24 GB VRAM per GPU and silently halves
  every subsequent number (an early `-ub 4096` read 269/19.5 instead of ~460/69).  Always
  `pgrep -af 'bin/llama'` and `rocm-smi --showmemuse` before trusting a measurement; `pkill -9 -f` on a
  pattern that also matches your own shell kills the shell (exit 137) — use `[l]lama-cli`.
* **`-p "$(cat bigfile)"` hits "Argument list too long"** above ~128 KB.  Use `-f <file>`.
* **`llama-cli -v` echoes the prompt**, so a naive "generated text" slice can be the prompt.  Slice with
  `sed -n '/^> /,/\[ Prompt:/p' file | sed '1d;$d'`, or run without `-v`.
* **Short-run acceptance is meaningless** (see §6.3).  For coherence use `////`; for acceptance use
  `-n >= 2000`.
* The r15 build of record is `~/llama-r13` (branch `rdna-r13`, tip `7e2dcd8f1`, tree `0e9273f8`);
  the r14 comparison build is `~/llama.cpp/build-rocm`.  Build with
  `ROCM_PATH=/opt/rocm-7.14.1-gfx120X cmake --build build --target llama-cli -j 16`.

## 10. Definition of done

* The exact defect in the staging path identified (which of §7.2–§7.5), with the byte-level comparison
  that proves it (host table vs ring slot for one staged MoE op).
* A fix that keeps `-sm layer`, single-GPU and `-ncmoe 0` unchanged, gated on **staging left ON** at:
  `////`=0 and coherent text at 3800 and 16k tokens under `-sm tensor`, `-ub 4096` **and** `-ub 8192`;
  the `-ncmoe 0` 3-GPU Q4_K_M byte-identity; width purity (`none == n1 == n3 == n7`); and a long MTP
  acceptance run at `-n 3000` (>= 0.45).
* A decision recorded on the shipped gate: if the ring cannot be made correct quickly, the interim
  delivery should **raise/disable the `-sm tensor` staging gate** rather than ship a silent corruption —
  the current default mixes correct and corrupt runs by prompt width, which is the worst outcome.
* Regression: a gate that **coherence-checks the staging-on side** (the r12/block-06 win was validated on
  throughput only, §0 "Is this a delivery regression?").
