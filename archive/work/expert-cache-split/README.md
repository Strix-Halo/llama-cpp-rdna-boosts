> ## ARCHIVED 2026-10-07 — campaign CLOSED, premise REFUTED (no code change)
>
> `TODO.md` #44 claimed that under `-sm tensor --n-cpu-moe N` each GPU caches a **mirrored** copy of the
> expert weights, so the cache residency would double and the host copy would halve if they were split.
> **That premise is wrong: the expert weights are already split per device, and were never mirrored on
> the current delivery.**  The whole handoff below is therefore kept as a *historical record of a false
> premise*; read the refutation first, and do not act on §1-§7 (its goal, arithmetic and hypotheses all
> assume mirroring).
>
> ### What was measured (2026-10-07)
>
> **1. The field log's own arithmetic already refutes mirroring.**  `/tmp/s22-logs` (2× R9700,
> `-sm tensor --n-cpu-moe 48`, IQ4_NL qwen4exp) line 63 reads `arena 39424.2 MiB of 64800.0 MiB host
> experts (60.8% residency)` with `288 tables`.  The model's **full** expert set (the `ffn_(gate|up|down)_exps`
> tensors of the 9-shard GGUF, 48 layers × 512 experts) is exactly **64800.0 MiB**.  `alloc_all_locked`
> computes the denominator as `host_total = Σ_tables n_experts × expert_bytes` over *all devices*:
>
> | layout | tables | `expert_bytes` | host_total |
> |---|---|---|---|
> | mirrored | 288 | full expert | **129600 MiB** (2× the set) |
> | split | 288 | the device's half | **64800 MiB** (the set) |
>
> The log's 64800 is the split value.  The `60.8%` is likewise `arena / full-set`, which a mirrored run
> could not produce (it would report ~30.4%).
>
> **2. A direct runtime dump of the cache tables.**  With a transient diagnostic in `moe_cache_table`
> ([`geom-diagnostic.patch`](geom-diagnostic.patch)), on Qwen3.6-35B-A3B Q4_K_M, 2× R9700,
> `-sm tensor -ncmoe 41`, every registered table reports `expert_bytes = 0.500 × host_bytes` with a
> symmetric per-device `src_off` — gate/up (axis 1) `0` vs `294912`, down (axis 0) `0` vs `176/210`
> (with `host_pitch`).  The table sum is 18662.0 MiB, exactly the denominator in the run's
> `MoE expert cache (auto)` line.  Forced mirroring (`GGML_META_SPLIT_COPY=0`) registers **zero** cache
> tables, because `ggml_backend_meta_moe_cache_update` requires a single-segment axis split.
>
> **3. The host master is a single buffer, not a per-device copy.**  Under `-sm tensor` the meta device's
> host buffer type is `nullptr` (its simple devices have distinct per-device `ROCm_Host` types), so the
> loader's fallback (`llama-model-loader.cpp:1283-1297`) places the host expert master in **one** device's
> pinned host buft; `model.moe_host_expert_bytes` is therefore a **single** entry holding the whole set.
> There is no duplicated host copy to halve.
>
> ### Why the "mirrored" inference was made — and why it is invalid
>
> The cache keys `g_sem_to_id` on `(layer, role, device)` and `g_key_to_id` on `(src0, device)`, so a
> **split** also yields one table per (layer, role) **per device** = `48 × 3 × 2 = 288` tables.  The table
> *count* is identical under mirroring and splitting; only `expert_bytes` vs `host_bytes` (and the
> `host_total` denominator) can distinguish them.  `TODO.md` #44's "a split would be 144" inverted the
> test: the split is 288, and the field log's 288 tables with a 64800 denominator is the *split* signature.
>
> ### Consequence
>
> * Residency is already `arena / full_expert_set`; each device already caches only its own slice of every
>   expert.  The arena — not mirroring — is the binding constraint (60.8% on the field config).  There is
>   no "residency doubling" and no "host copy halving" available.
> * No delivery code changed; `wip/expert-cache-split/` is a redirect stub.  Residual observations that
>   are **not** this campaign (recorded here for whoever needs them): the field config runs
>   `--load-mode none`, which this delivery itself warns is *known to fault intermittently during the
>   split upload*; and the single `moe_host_expert_bytes` entry means the post-prefill drop decision only
>   walks the owner device's slab.
>
> The historical handoff follows unchanged (paths were NOT rewritten because it never had artifacts; it
> contains no relative markdown links into the repo).

---

# Expert-cache residency under `-sm tensor` + host experts: stop MIRRORING the cache tables

> **Note on this historical text: its central premise is refuted (see the ARCHIVED block above).**

**Cold start — read this file top to bottom; it is self-contained.**  Opened 2026-10-07 by the maintainer
(`TODO.md` #44).  **Status at archive time: CLOSED — the "mirrored" premise was disproven by measurement.**

---

## 1. The goal

Under `-sm tensor --n-cpu-moe N` each GPU currently registers a MoE expert-cache table for **every**
layer-role: the field log proves the cache's table set is **mirrored across devices** —

```
MoE expert cache (auto): arena 39424.2 MiB of 64800.0 MiB host experts (60.8% residency)
moe_cache_update_host: device-remap fast path armed after a uniform eager pass over 288 tables
288 tables = 48 layers x 3 roles x 2 devices      <- every device holds a table for EVERY layer-role
```

If each device instead cached only **its own slice** of a split expert set, then:

* **cache residency roughly doubles** — with the same per-device arena we would go from ~61 % of the *whole*
  expert set to ~100 % of the device's *half* (see the arithmetic in §2);
* **host RAM halves** — the host-resident expert copy stops being duplicated per device;
* the per-card upload volume and per-card compute already halve: the *scheduler-side* split of the uploads
  and compute **is implemented and default-on** (see §3), so this campaign is about the **cache/host copy**,
  not about the upload path.

Maintainer's framing: *"Ideally they should be split, but one battle at a time"* (2026-10-07), and *"this
should improve cache residency overall if we're not mirroring weights"*.

## 2. Why it matters: the field arithmetic (2026-10-07, logs `/tmp/s22-logs`)

Config: 2× R9700, `-sm tensor --n-cpu-moe 48`, `-ub 6144 -c 204800`, IQ4_NL qwen4exp (48 layers, 512 experts),
cache auto, `GGML_CUDA_ALLREDUCE=ce`.

| quantity | measured | if the expert set were split per device |
|---|---|---|
| host expert bytes (both devices) | **64800 MiB** | 32400 MiB (each device its own half) |
| cache arena (both devices) | **39424.2 MiB** (60.8 % residency) | same arena |
| per device | 32400 MiB of "host experts" vs 19712 MiB of arena | 16200 MiB of experts vs 19712 MiB of arena |
| → residency per device | **~61 %** | **~100 %** (everything resident) |

So the expected win is **host footprint ~halved and the decode arena fully resident** on the 2-GPU field
config — a decode-speed and host-RAM win, not a prefill one (prefill residency is the *closed* prior
campaign's topic, §3).

## 3. What is already known — do not redo

**The upload/compute split exists and is default-on.**  `llama_meta_device_get_split_state`
(`src/llama-model.cpp`, ~line 445) strips the scheduler's `<backend>#<tensor>#<copy>` wrapper so a split
input copy **inherits its weight's split state**; the gate is `GGML_META_SPLIT_COPY` with default **1**
(`0` = the old mirrored behaviour, `2`/`3` = the bisection modes for the fault that was hit).  The prior
campaign's rule: **every "mirrored" measurement must pass `GGML_META_SPLIT_COPY=0`**, otherwise you are
measuring the split.

**Prior art in this repo:**

| where | what it settles |
|---|---|
| `archive/work/tensor-split-expert-split/README.md` | the upload/compute split: implemented, numerically correct, `GGML_META_SPLIT_COPY` gate, the **memory fault** it hit and the bisection that localized it (§29/§30), the compute floor (7 037 vs 5 107 t/s at pp8192), and **§31 = "partial VRAM expert residency for prefill"** (the R9V direction: per-expert hot slots + cold shards read in place over UVA).  CLOSED; the *cache* (decode) half was handed on. |
| `archive/work/moe-expert-cache/` | the cache's own history (why it exists, the sizing/eviction model). |
| `archive/work/moe-cache-autosize/` | the arena/slab work: the movable-boundary slab, the drop/re-arm, the auto sizing that consumes `moe_host_expert_bytes`. |

**Code pointers that define today's behaviour:**

* the cache keys a table on `(layer, role, **device**)` —
  `g_sem_to_id.emplace(std::make_tuple(layer, sem_role(src0), device), id)` (`moe-expert-cache.cu`, in
  `moe_cache_table`) — so mirrored host experts necessarily yield duplicate per-device tables.
* host-expert bytes are accumulated **per device** in the loader:
  `moe_host_expert_bytes[ggml_backend_buft_get_device(buft)] += ggml_nbytes(&t_meta)` (`llama-model-loader.cpp`,
  lines ~1399 and ~1441), copied to the model at `llama-model.cpp:1910`, and consumed by
  `llama_model_moe_host_expert_bytes` (used by the cache auto-sizing, the drop decision and the preflight).
* the cache's auto-sizing sizes each device's arena against `ggml_cuda_slab_arena_total(d)` and holds back
  `n_tables * ggml_cuda_slab_arena_unit(d)` (`alloc_all_locked`) — **fewer tables per device also holds back
  less**, a second-order win.
* `-sm layer` already routes host experts per device (the r14 work, `TODO.md` #39) and is *faster* than
  `-sm tensor -ncmoe` today (`AGENTS.md`); closing that gap is this campaign's motivation.
* the drop/re-arm path (`src/llama-context.cpp`, `slab_narrow_boundary`) and the slab's boundary moves are
  the pieces that make a *smaller* per-device table set safe — they must both be re-run (§6).

## 4. What to measure FIRST (in this order — no code changes yet)

1. **Is the host-resident expert tensor one mirrored buffer per device, or one tensor every device reads?**
   Instrument or read the placement: at load, log `(tensor name, buffer, device, nbytes)` for the
   `ffn_*_exps` tensors (`llama-model-loader.cpp` / `llama_model::load_tensors`).  The field log's
   `moe_host_expert_bytes` had **one** entry of ~56.7 GiB — that is consistent with one buffer holding the
   whole set, but confirm *which device's* buft owns it and whether the second device gets its own copy.
2. **Table inventory per device** — `MOE_EXPERT_CACHE_VALIDATE=1` (or the per-turn arena line) on the field
   config: how many tables per device, how many bytes, and what the per-table slot counts are.  Expect
   144 tables/device today; the target is 72.
3. **The counterfactual residency** — with the same arena, what would a split give?  (`arena_bytes /
   (host_expert_bytes/2)` vs `arena_bytes / host_expert_bytes`.)  This sets the expected payoff and is
   needed to judge any implementation attempt.
4. **Host RSS** at steady state (the field config's host footprint) — the "halved host copy" claim.
5. Only after 1-4: pick the implementation point (§5).

## 5. Hypotheses, in expected-payoff order

* **H1 — the fix is in the split-state provider.**  The *weight* is host-resident, so the scheduler's split
  state for it defaults to mirrored; the copy-inherits-the-weight rule (`GGML_META_SPLIT_COPY=1`) then
  propagates mirroring to both the upload and the cache's view.  Fix: give a host-resident expert tensor a
  proper split state (gate/up axis 1, down axis 0 — the same axes the prior campaign used), so every
  consumer (upload copy, cache table extent, `moe_host_expert_bytes`) sees a slice.
  *Cheapest to try; verify with measurement 2 (table count) and 3 (residency).*
* **H2 — the fix is in the cache's table extent.**  Keep the host tensor as-is but register a table per
  device for only that device's slice (`src0 + offset`, `expert_bytes` slice extent) — i.e. make the cache's
  per-device view match what the device actually computes.  Requires the loader/model to know each device's
  slice extent → likely needs H1's information anyway.
* **H3 — the fix is in the loader's host-buft choice.**  Route the host expert tensor to a *per-device*
  pinned buft whose tensor is the device's slice (the r14 per-device-host-buft machinery already exists,
  `ggml_backend_cuda_host_buffer_type_dev`).  Biggest change; only if H1/H2 hit a wall.

## 6. Risks and invariants — do not break these

* **The cache's correctness invariants** (all from the r19-r24 work, `GREEDY-PURITY.md` §19/§24/§25 and
  `TODO.md` #42): the **per-table** fallback gate must stay consistent with the **global** fusion gate; the
  **wholesale-fallback** rule (a partially-failed cache falls back wholesale); the layer-uniform slot count
  (`t.slots` must be the *achieved* count); no arena freed while a kernel carries its address.
* **The slab** (`block 06`): the arena region, the movable boundary, `ggml_cuda_slab_extend`, the
  drop + `moe_cache_rearm`.  A split table set changes how much arena is needed and how many tables are
  evicted per boundary move — re-run the field config *and* `wide1 -> short -> wide2`.
* **The fault that was hit before**: split copies interacting with the gather path faulted; the
  `GGML_META_SPLIT_COPY=2/3` bisections exist to isolate it.  Keep that gate working and bisectable.
* **`-sm layer` must not regress** (it is the current best for host experts), non-AMD backends stay
  consistent, and per-block buildability is *not* required (the delivery applies as a whole).
* Any change that alters **numerics** must follow `GREEDY-PURITY.md` §36 (plain == draft-mtp for
  f16/bf16/q8_0; coarse quants logits-level).  A placement change should be bit-identical — prove it.

## 7. Acceptance criteria

1. **Residency**: measured before/after on the field config (per-device table count, arena residency), and
   the host footprint (halved, or the shortfall explained).
2. **Coherence**: `////` = 0 same-seed; and for a placement change, the same-seed text must be
   **byte-identical** to the current build (it is a placement change, not an arithmetic one).
3. **MTP rule 0**: `-n 3000 --reasoning on`, `--reasoning on` for R, long run — acceptance must stay
   **0.53519 = 1848/3453** (bit-identical) unless the change is deliberately numeric.
4. **DoD** (`-ub 8192` cache-auto 16k): decode **≥ 68.9** / prefill **≥ 1040 t/s**, and the server
   `wide1 -> short -> wide2` scenario **0 aborts** with the arena restored.
5. **`-sm layer`** unchanged; `scripts/validate-set.sh` green; a dated `WORKLOG.md` entry; `AGENTS.md` /
   `patches/README.md` / `release.json` updated if anything ships.

## 8. Environment and harness (copy-paste)

```bash
# build (ccache makes a wiped rebuild fast)
cd ~/llama.cpp && BUILD_DIR=build-<tag> ~/bin/build-llama-rocm-714
#   fast loop: cmake --build build-<tag> --target llama-cli llama-server -j 16

# the field configuration (2 GPU, host experts, 204800 ctx)
HIP_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib \
  build-<tag>/bin/llama-server -m /llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf \
  -md /llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  -fa on -ctk q8_0 -ctv q8_0 -sm tensor --n-cpu-moe 48 -ub 6144 -b 6144 -c 204800 --no-kv-unified \
  --spec-type draft-mtp --spec-draft-n-max 3 --cache-idle-slots --ctx-checkpoints 64 --fit off \
  -t 7 --threads-http 4 -lv 3 --host 127.0.0.1 --port 8033

# the cli + server regression harnesses (never run benchmarks in parallel)
BIN=./build-<tag>/bin/llama-cli   /home/stew675/llama-cpp-rdna-boosts/archive/work/moe-cache-autosize/open2-harness-cli.sh <tag> 2000
BIN=./build-<tag>/bin/llama-server SRV_UB=8192 SRV_NPRED=48 /home/stew675/llama-cpp-rdna-boosts/archive/work/moe-cache-autosize/open2-harness-server.sh <tag>
```
*(The harnesses live with the archived campaign: `archive/work/moe-cache-autosize/open2-harness-*.sh`;
`scripts/gate-qwen4exp-quant-coherence.sh` is the other gate.)*

**Log lines that matter:** `MoE expert cache (auto): arena X MiB of Y MiB host experts (Z% residency)`,
`device-remap fast path armed after a uniform eager pass over N tables`, `moe_cache_evict_slab_range`,
`moe_cache_rearm`, `ggml_cuda_slab_extend`, and the per-turn `MoE arena = h (hits/reaches), arena X MiB`.

**Hard rules:** `llama-cli` needs `--single-turn` (`--no-display-prompt` for scripted output); size `-t` so
the GPU-IRQ cores (13,14,15 with 3 GPUs) stay free; never run parallel benches; the MTP gates must be long
(`-n 3000`) and reasoning-pinned.
