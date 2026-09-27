#!/usr/bin/env bash
# wip/tensor-split-expert-split: the ub x {split,mirrored} x {staged,prune} prefill grid.
# Usage: bench-grid.sh [ub ...]
set -u
cd ~/llama-r12
M=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
BIN=./build-rocm/bin/llama-bench
UBS=${*:-"128 2048 4096 8192"}
REPS=${REPS:-3}

pp() { # $1 label $2 env
  local out
  out=$(env HIP_VISIBLE_DEVICES=0,1 $2 timeout 900 $BIN -m "$M" -ncmoe 99 -fa 1 \
        -p "$UB" -ub "$UB" -n 1 -b "$UB" -sm tensor -r "$REPS" 2>/dev/null | grep -E '\| *pp[0-9]+ *\|')
  printf '%-34s ub=%-5s %s\n' "$1" "$UB" "$(echo "$out" | sed 's/.*| *\(pp[0-9]*\) *| *\([0-9.]*\).*/\1 \2/')"
}

for UB in $UBS; do
  pp "mirrored+staged"   "GGML_META_SPLIT_COPY=0 GGML_SCHED_STAGE=1"
  pp "split+staged"      "GGML_META_SPLIT_COPY=1 GGML_SCHED_STAGE=1"
  pp "mirrored+prune+PIN" "GGML_META_SPLIT_COPY=0 GGML_SCHED_STAGE=0 GGML_META_PINHOST=1"
  pp "split+prune+PIN"    "GGML_META_SPLIT_COPY=1 GGML_SCHED_STAGE=0 GGML_META_PINHOST=1"
done
