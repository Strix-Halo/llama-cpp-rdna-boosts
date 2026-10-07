#!/bin/bash
# Item-1 repro harness.  Usage: run.sh <tag> <n_tokens> [env assignments...]
set -u
TAG="${1:?tag}"; N="${2:?n}"; shift 2
cd ~/llama.cpp
export HIP_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
LOG=/tmp/item1/$TAG.log
: > "$LOG"
echo "=== $TAG  n=$N  env=$*  $(date +%F_%T) ==="
START=$(date +%s)
env "$@" ${BIN:-./build-rocm-r16/bin/llama-cli} \
  ${EXTRA_ARGS:-} \
  -m /llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  -md /llm/models/Qwen3.8/Flash-Next/IQ4_XS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  -sm tensor -ncmoe 48 -fa on -ctk q8_0 -ctv q8_0 -t 8 -c 32768 -b 8192 -ub 8192 \
  --spec-type draft-mtp --spec-draft-n-max 3 --reasoning off -n "$N" -f /tmp/pl_16k.txt \
  --seed 42 --temp 0 --no-display-prompt --single-turn -lv 3 > "$LOG" 2>&1
RC=$?
END=$(date +%s)
echo "exit=$RC elapsed=$((END-START))s"
echo "--- metrics ---"
grep -aE "compute buffer size|MoE expert cache|auto arena|residency|alloc_all_locked|reallocating|arena total|extra arena reserve|sched_reserve: reserve|graph:|prompt processing|Prompt:|n_decoded|slot print_timing|draft acceptance|MoE arena =|drop_compute|ggml_backend_cuda_buffer_type_alloc_buffer.*failed|release_arena|shrink_step|stood down" "$LOG" | tail -60
echo "--- coherence ---"
grep -ac '////' "$LOG" | sed 's/^/slash4_lines=/'
echo "--- tail ---"
tail -3 "$LOG"
