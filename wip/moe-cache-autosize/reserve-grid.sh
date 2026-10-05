#!/bin/bash
# Reserve grid: peak-VRAM / no-OOM validation of auto mode across ctx x ub x MTP x draft-offload.
# Sequential; never parallel (AGENTS.md).  Single R9700, IQ3_XXS.
set -u
BIN=/home/stew675/llama-r13/build/bin/llama-cli
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
P=/home/stew675/llama-cpp-rdna-boosts/prompts/code-python.txt
Q=/llm/models/Qwen3.8/Flash-Next/IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
TQ=/llm/models/Qwen3.8/Flash-Next/IQ3_XXS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf

# run <ctx> <ub> <spec:mtp|none> <draft_offload:1|0> [mib]
run() {
  local ctx="$1" ub="$2" spec="$3" doff="$4" mib="${5:-auto}"
  local -a envs=(env HIP_VISIBLE_DEVICES=0 LLAMA_MTP_DRAFT_OP_OFFLOAD="$doff")
  [ "$mib" != "auto" ] && envs+=(MOE_EXPERT_CACHE_MIB="$mib")
  local -a cmd=("${envs[@]}" "$BIN" -m "$Q" -ngl 99 -sm layer -ncmoe 48 -fa 1 -ctk q8_0 -ctv q8_0
      -t 8 -c "$ctx" -b 2048 -ub "$ub" --lazy-mode off --load-mode auto
      -p "$(cat $P)" -n 32 --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt -v)
  if [ "$spec" = mtp ]; then cmd+=(-md "$TQ" --spec-type draft-mtp --spec-draft-n-max 3); else cmd+=(--spec-type none); fi
  local out rc
  out=$(timeout 1800 "${cmd[@]}" 2>/tmp/rg.err); rc=$?
  local pf ar g oom
  pf=$(grep -oE "projected arena [0-9.]+ MiB" /tmp/rg.err | grep -oE "[0-9.]+" | head -1)
  ar=$(grep -oE "arena [0-9.]+ MiB of" /tmp/rg.err | grep -oE "[0-9.]+" | head -1)
  g=$(echo "$out" | grep -oE "Generation: [0-9.]+" | tail -1 | grep -oE "[0-9.]+")
  oom=$(grep -icE "out of memory|failed to allocate|GGML_ASSERT|abort" /tmp/rg.err)
  printf "ctx=%-7s ub=%-5s %-5s doff=%s | preflight=%-8s arena=%-8s t/s=%-7s oom=%s rc=%s\n" \
    "$ctx" "$ub" "$spec" "$doff" "${pf:-?}" "${ar:-0/off}" "${g:-FAIL}" "$oom" "$rc"
}

echo "### reserve grid $(date +%H:%M:%S)"
# ctx sweep at ub=2048, MTP on
run 8192   2048 mtp 1
run 32768  2048 mtp 1
run 131072 2048 mtp 1
# ub sweep at ctx=32768, MTP on
run 32768  4096 mtp 1
run 32768  8192 mtp 1
# draft-offload off, and MTP off
run 32768  2048 mtp 0
run 32768  2048 none 1
# worst case: max ctx + max ub
run 131072 8192 mtp 1
