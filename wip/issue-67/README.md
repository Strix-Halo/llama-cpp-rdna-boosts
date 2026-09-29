# wip/issue-67 - the address-gated `ROPE -> VIEW -> SET_ROWS` fusion decides the W=1 decode logits

**START HERE.**  This directory tracks issue **#67** (from #58 item D, @DanoPTT, gfx1201 / Windows / ROCm 10):
greedy output on Q6_K is not reproducible across fresh `llama-server` starts.

Current release is **`v16-84e76d8a2-r24`** (commit `6d88c45` on `main`), which adds two default-off kill
switches: `GGML_CUDA_DISABLE_ROPE_SET_ROWS=1` and `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1`.

## State of the investigation

The observable is the W=1 row-0 logits hash from `test-logits-width-probe` on the reporter's model family
(`/llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf`, `prompts/recall.txt`, P=256, ubatch=512):

| config | W=1 row-0 hash |
|---|---|
| default | `f6d62323d9339541` |
| `GGML_CUDA_DISABLE_ROPE_SET_ROWS=1` | `3ab223a4f08afd6e` |
| `GGML_CUDA_DISABLE_FUSION=1` | `3ab223a4f08afd6e` |
| `GGML_CUDA_DISABLE_GRAPHS=1` + `..._ROPE_SET_ROWS=1` | `3ab223a4f08afd6e` |
| `GGML_CUDA_DISABLE_RMS_NORM_MUL_ROPE=1` | `f6d62323d9339541` (no change) |
| `GGML_CUDA_DISABLE_GRAPHS=1` / `GGML_CUDA_FA_KV_NATIVE=0` | `f6d62323d9339541` (no change) |
| the other per-fusion switches | `f6d62323d9339541` (no change) |
| 20 fresh default starts (Linux) | `f6d62323d9339541` (20/20) |

- The upstream `ROPE -> VIEW -> SET_ROWS` fusion (`ggml_cuda_should_fuse_rope_set_rows`) is the only lever.
  A temporary probe that forced the memory-range check to `false` for every address-gated fusion **except**
  this one still returned `f6d6`, so the rope fusion alone decides the hash.
- The difference is a rounding-path difference, not a semantic bug: `KL(fused||unfused)=1.7e-4`, top-1 token
  matches, top-10 overlap 10/10.  The sequence state differs; the first differing half is a 1-ULP delta that
  amplifies over the layers.

## What has been ruled out

- **The fused rope kernel arithmetic.**  `wip/issue-67/rope_fuse_test.cpp` builds the exact
  `rope -> view -> set_rows` chain and compares the final F16 cache with the fusion on vs off.  It is
  **bit-identical** for NORMAL / NEOX / MROPE / IMROPE, full and partial rotation, and the model's exact
  params (head 256, `n_rot=64`, sections `[11,11,10,0]`, `freq_base=1e7`).  So `rope_multi<float,half>` ==
  `rope_multi<float,float>` + `k_set_rows<float,long,half>`.
- **Graph capture.**  The difference persists with `GGML_CUDA_DISABLE_GRAPHS=1`.
- **HIP graphs, `FA_KV_NATIVE`.**  Neither moves the hash.
- **A different address-gated fusion.**  Forcing all address-gated fusions off except this one does not
  change the hash.
- **Two contraction sites** (the rotated `x0*cos - x1*sin` and the YaRN `theta_interp*(1-ramp_mix) +
  theta_extrap*ramp_mix`): forcing both to explicit `__f*_rn` changed neither hash.
- **A kernel-config flip elsewhere.**  A full `rocprofv3 --kernel-trace` diff (default vs
  `DISABLE_ROPE_SET_ROWS=1`) shows the **only** name change is `rope_multi<...,float,__half>` appearing in the
  fused run (plus the expected `k_set_rows` count).  No downstream kernel changes config.

## The open question

The rope kernel is transparent and the kernel set is identical, yet the model output differs when the fusion
is toggled.  That leaves the **graph/allocation side effect of eliding the F32 rope buffer**:

- Hypothesis A (leading): the elided rope output buffer re-addresses the compute buffer, and some kernel
  takes an **address/alignment-dependent internal branch** (same kernel name) that rounds differently, or
  there is a **missing alloc dependency** for this fusion that lets the allocator reuse a live buffer.
  The delivery's `add_alloc_deps` pass (`ggml-cuda.cu`, "performance positive fusions") does **not** cover
  the rope fusion - see the TODO comment there.
- Hypothesis B: the scheduler places a node on a different backend between the two configs.  `rocprofv3`
  shows the rope and set_rows kernels on the GPU in both, so this is unlikely for those nodes, but a
  `GGML_SCHED_DEBUG=2` per-node dump (with `--verbose`) has not been captured.

## Next steps (fresh session)

1. Compare the **allocation plans** between default and `GGML_CUDA_DISABLE_ROPE_SET_ROWS=1` for the same
   graph: tensor addresses (`GGML_SET_BYTES` / `GGML_SET_NAMES`), buffer sizes, and whether any tensor
   aliases the elided rope output.  Look for an address that moves and feeds a kernel with an
   alignment-dependent branch.
2. Capture `GGML_SCHED_DEBUG=2 --verbose` per-node backend for both configs and diff (do not trust the
   suppressed probe logs; run `llama-cli ... --single-turn --no-display-prompt`).
3. If it is a missing alloc dep, add it (mirroring the conv/norm-gated entries) and re-run the hash gate.
4. Only then decide the fix: make the fused path bit-transparent (preferred, §31) or take the raw allocator
   address out of the selection.

## Reproduce

```bash
# model + probe (GPU-vs-GPU, same binary, only the env differs)
cd ~/llama.cpp
export LD_LIBRARY_PATH=$PWD/build-rocm/bin:/opt/rocm-7.14.1-gfx120X/lib
export HIP_VISIBLE_DEVICES=0
M=/llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf
P=~/llama-cpp-rdna-boosts/prompts/recall.txt
./build-rocm/bin/test-logits-width-probe $M $P 256 512                       # default
GGML_CUDA_DISABLE_ROPE_SET_ROWS=1 ./build-rocm/bin/test-logits-width-probe $M $P 256 512

# isolated rope fusion test
g++ -O2 -std=c++17 rope_fuse_test.cpp -I~/llama.cpp/ggml/include -I~/llama.cpp/ggml/src \
    -L~/llama.cpp/build-rocm/bin -lggml -lggml-base -o /tmp/rope_fuse_test
GGML_BACKEND_PATH=~/llama.cpp/build-rocm/bin LD_LIBRARY_PATH=... /tmp/rope_fuse_test imrope
# add NDIMS=<n> to override n_dims; GGML_CUDA_DISABLE_ROPE_SET_ROWS=1 to disable the fusion
```

**Methodology note:** always compare **GPU vs GPU** (same binary, same backend, only the env/kill switch
differs).  GPU rope and CPU rope are not bit-exact, so the CPU backend must not be used as a purity
reference.  `llama-cli` must always be run with `--single-turn` (and `--no-display-prompt` for scripted
output) or it drops into chat mode and hangs.

## Repro of the current model's rope params

```
n_embd_head_k = 256, n_head_kv = 4, n_rot = 64, rope type = 40 (IMROPE),
mrope sections = [11,11,10,0], n_ctx_orig_yarn = 262144, rope.freq_base = 1e7
```
