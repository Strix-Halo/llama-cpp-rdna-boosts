#!/usr/bin/env bash
# Prefill sweep across every MoE offload level (-ncmoe 0..40), 1 and 2 GPUs,
# -sm tensor / -sm layer, delivery vs upstream.  Emits CSV: config,ncmoe,pp8192
# Usage: sweep-ncmoe.sh OUTFILE [start] [step]
set -u
MODEL=/llm/models/Qwen3.6/35B-A3B/Q4_K_M/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
OURS=~/llama-promote/build-rocm/bin/llama-bench
UP=~/llama-upstream/build-rocm/bin/llama-bench
OUT="${1:-/tmp/sweep.csv}"
START="${2:-0}"
STEP="${3:-1}"
REPS="${REPS:-1}"

echo "config,ncmoe,pp8192" > "$OUT"

pp() { # bin devices sm ncmoe
  local out
  out=$(env HIP_VISIBLE_DEVICES="$2" timeout 600 "$1" -m "$MODEL" -ncmoe "$4" -fa 1 \
        -p 8192 -ub 8192 -n 1 -b 8192 -sm "$3" -r "$REPS" 2>/dev/null)
  echo "$out" | grep -E '\| *pp8192' | sed 's/.*| *\(pp8192\) *| *\([0-9.]*\).*/\2/'
}

emit() { # label bin devices sm ncmoe
  local v
  v=$(pp "$2" "$3" "$4" "$5")
  if [ -z "$v" ]; then v="NA"; fi
  printf '%s,%s,%s\n' "$1" "$5" "$v" | tee -a "$OUT"
}

for n in $(seq "$START" "$STEP" 40); do
  emit "ours-1gpu"            "$OURS" 0   layer  "$n"
  emit "ours-2gpu-tensor"     "$OURS" 0,1 tensor "$n"
  emit "ours-2gpu-layer"      "$OURS" 0,1 layer  "$n"
  emit "upstream-1gpu"        "$UP"   0   layer  "$n"
  emit "upstream-2gpu-layer"  "$UP"   0,1 layer  "$n"
done
echo "wrote $OUT ($(wc -l < "$OUT") rows)"
