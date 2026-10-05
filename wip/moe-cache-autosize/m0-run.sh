#!/bin/bash
# M0 re-baseline on the r13 build (block 16).  Sequential; never parallel (AGENTS.md).
# Usage: m0-run.sh <set>   set = core | single | sweep | big
set -u
BIN=/home/stew675/llama-r13/build/bin/llama-cli
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
P=/home/stew675/llama-cpp-rdna-boosts/prompts/code-python.txt
Q=/llm/models/Qwen3.8/Flash-Next/IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf
TQ=/llm/models/Qwen3.8/Flash-Next/IQ3_XXS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf
M=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-PROJFIX-00001-of-00009.gguf
T=/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf

# run <label> <devs> <model> <md|-> <split> <ncmoe> <mib> <spec> <n> [ctx]
run() {
  local label="$1" devs="$2" model="$3" md="$4" sm="$5" ncmoe="$6" mib="$7" spec="$8" n="$9" ctx="${10:-8192}"
  local -a envs=(env HIP_VISIBLE_DEVICES="$devs")
  [ "$mib" != "auto" ] && envs+=(MOE_EXPERT_CACHE_MIB="$mib")
  local -a cmd=("${envs[@]}" "$BIN"
      -m "$model" -ngl 99 -sm "$sm" -ncmoe "$ncmoe" -fa 1 -ctk q8_0 -ctv q8_0 -t 8
      -c "$ctx" -b 2048 -ub 2048 --lazy-mode off --load-mode auto
      -p "$(cat $P)" -n "$n" --seed 42 --temp 0 --reasoning off --single-turn --no-display-prompt)
  [ "$md" != "-" ] && cmd+=(-md "$md")
  if [ "$spec" = "mtp" ]; then cmd+=(--spec-type draft-mtp --spec-draft-n-max 3); else cmd+=(--spec-type none); fi
  local out t acc slots
  out=$(timeout 2400 "${cmd[@]}" 2>/tmp/m0.err)
  t=$(echo "$out" | grep -oE "Generation: [0-9.]+" | tail -1)
  acc=$(echo "$out" | grep -oE "accept[^ ]* *=[^ ]*" | tail -1)
  slots=$(grep -oE "sized [0-9]+ slots/table" /tmp/m0.err | head -1)
  printf "%-52s %-16s %s %s\n" "$label" "${t:-FAIL}" "$slots" "$acc"
}

case "${1:-core}" in
core)
  # 2-GPU IQ4_NL: layer (fixed) vs tensor, cache off/on, MTP
  run "2G LAYER  ncmoe48 MIB=0     mtp"  0,1 $M $T layer  48 0     mtp 128
  run "2G LAYER  ncmoe48 MIB=24000 mtp"  0,1 $M $T layer  48 24000 mtp 128
  run "2G LAYER  ncmoe48 MIB=24000 plain" 0,1 $M - layer 48 24000 none 128
  run "2G TENSOR ncmoe24 MIB=0     mtp"  0,1 $M $T tensor 24 0     mtp 128
  run "2G TENSOR ncmoe24 MIB=8192  mtp"  0,1 $M $T tensor 24 8192  mtp 128
  run "3G LAYER  ncmoe48 MIB=0     mtp"  0,1,2 $M $T layer 48 0     mtp 128
  run "3G LAYER  ncmoe48 MIB=24000 mtp"  0,1,2 $M $T layer 48 24000 mtp 128
  ;;
single)
  run "1G IQ3_XXS ncmoe48 MIB=0     plain" 0 $Q - layer 48 0     none 128
  run "1G IQ3_XXS ncmoe48 MIB=20480 plain" 0 $Q - layer 48 20480 none 128
  run "1G IQ3_XXS ncmoe48 MIB=0     mtp"   0 $Q $TQ layer 48 0     mtp 128
  run "1G IQ3_XXS ncmoe48 MIB=20480 mtp"   0 $Q $TQ layer 48 20480 mtp 128
  run "1G IQ4_NL  ncmoe48 MIB=0     plain" 0 $M - layer 48 0     none 128
  run "1G IQ4_NL  ncmoe48 MIB=20480 plain" 0 $M - layer 48 20480 none 128
  run "1G IQ4_NL  ncmoe48 MIB=0     mtp"   0 $M $T layer 48 0     mtp 128
  run "1G IQ4_NL  ncmoe48 MIB=20480 mtp"   0 $M $T layer 48 20480 mtp 128
  ;;
sweep)
  for mib in 0 4096 8192 16384 24000 28672; do
    run "2G LAYER ncmoe48 MIB=$mib mtp" 0,1 $M $T layer 48 $mib mtp 128
  done
  ;;
big)
  run "2G TENSOR ncmoe32 MIB=0    mtp" 0,1 $M $T tensor 32 0    mtp 128
  run "2G TENSOR ncmoe40 MIB=0    mtp" 0,1 $M $T tensor 40 0    mtp 128
  run "2G LAYER  ncmoe40 MIB=24000 mtp" 0,1 $M $T layer 40 24000 mtp 128
  run "2G LAYER  ncmoe99 MIB=24000 mtp" 0,1 $M $T layer 99 24000 mtp 128
  run "1G IQ4_XS  ncmoe48 MIB=20480 mtp" 0 /llm/models/Qwen3.8/Flash-Next/IQ4_XS/*00001-of-0000*.gguf $T layer 48 20480 mtp 128
  ;;
auto)
  run "2G LAYER  ncmoe48 AUTO       mtp"  0,1 $M $T layer  48 auto mtp 128
  run "2G TENSOR ncmoe24 AUTO       mtp"  0,1 $M $T tensor 24 auto mtp 128
  run "3G LAYER  ncmoe48 AUTO       mtp"  0,1,2 $M $T layer 48 auto mtp 128
  run "3G TENSOR ncmoe24 AUTO       mtp"  0,1,2 $M $T tensor 24 auto mtp 128
  ;;
floor)
  for mib in 0 64 128 256 512 1024 2048; do
    run "1G IQ3 ncmoe48 MIB=$mib mtp" 0 $Q $TQ layer 48 $mib mtp 128
  done
  run "1G IQ3 ncmoe1  MIB=0    mtp" 0 $Q $TQ layer 1 0    mtp 128
  run "1G IQ3 ncmoe1  MIB=1024 mtp" 0 $Q $TQ layer 1 1024 mtp 128
  run "1G IQ3 ncmoe1  AUTO     mtp" 0 $Q $TQ layer 1 auto mtp 128
  run "1G IQ3 ncmoe4  MIB=0    mtp" 0 $Q $TQ layer 4 0    mtp 128
  run "1G IQ3 ncmoe4  AUTO     mtp" 0 $Q $TQ layer 4 auto mtp 128
  ;;
*) echo "unknown set: $1" >&2; exit 2;;
esac
