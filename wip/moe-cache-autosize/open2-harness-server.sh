#!/bin/bash
# Server-safety test: wide prefill -> concurrent short, then a LATER wide prefill (the case that needs
# the compute buffer back, and where the arena must yield).  Usage: srv.sh <tag> [env...]
set -u
TAG="${1:?tag}"; shift
cd ~/llama.cpp
export HIP_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH=/opt/rocm-7.14.1-gfx120X/lib
LOG=/tmp/item1/srv-$TAG.log
PORT=$(( 8100 + RANDOM % 400 ))
: > "$LOG"
echo "=== srv $TAG  env=$*  port=$PORT  $(date +%F_%T) ==="
env "$@" ${BIN:-./build-rocm-r16/bin/llama-server} \
  -m /llm/models/Qwen3.8/Flash-Next/IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  -md /llm/models/Qwen3.8/Flash-Next/IQ4_XS/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  -sm tensor -ncmoe 48 -fa on -ctk q8_0 -ctv q8_0 -t 8 -c 32768 -b ${SRV_UB:-8192} -ub ${SRV_UB:-8192} \
  --spec-type draft-mtp --spec-draft-n-max 3 --reasoning off \
  --host 127.0.0.1 --port "$PORT" -lv 3 > "$LOG" 2>&1 &
PID=$!
ready=0
for i in $(seq 1 180); do
  if ! kill -0 "$PID" 2>/dev/null; then echo "server died during load"; break; fi
  if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; then ready=1; break; fi
  sleep 1
done
if [ "$ready" != 1 ]; then echo "NOT READY (log tail):"; tail -5 "$LOG"; kill "$PID" 2>/dev/null; exit 1; fi
echo "ready after ${i}s"

req() { # req <tag> <prompt-file-or-string> <n_predict>
  local tag="$1" pfile="$2" n="$3"
  python3 - "$pfile" "$n" <<'PY' > /tmp/item1/srv-body.json
import json,sys
p=open(sys.argv[1]).read() if sys.argv[1].startswith('/') else sys.argv[1]
print(json.dumps({"prompt":p,"n_predict":int(sys.argv[2]),"temperature":0,"seed":42,"cache_prompt":False}))
PY
  curl -s -m 600 -X POST "http://127.0.0.1:$PORT/completion" \
       -H 'Content-Type: application/json' --data-binary @/tmp/item1/srv-body.json \
       > "/tmp/item1/srv-$TAG-$tag.out" 2>&1
  echo "  req $tag exit=$?"
}

# turn 1 + a concurrent short prompt
if [ "${SRV_SEQ:-0}" = 1 ]; then
  req wide1 /tmp/pl_16k.txt ${SRV_NPRED:-16}
else
  req wide1 /tmp/pl_16k.txt ${SRV_NPRED:-16} &
  W1=$!
  sleep 2
  req short "Write one sentence about rain." ${SRV_NPRED:-16} &
  S=$!
  wait $W1 $S
fi

# a LATER wide request (needs the wide compute layout back)
req wide2 /tmp/pl_16k.txt ${SRV_NPRED:-16}
req short2 "Write one sentence about snow." ${SRV_NPRED:-16}

if kill -0 "$PID" 2>/dev/null; then ALIVE=yes; else ALIVE=no; fi
echo "server_alive=$ALIVE"
kill "$PID" 2>/dev/null; wait "$PID" 2>/dev/null
echo "--- log signals ---"
grep -acE "GGML_ASSERT|cudaMalloc failed|out of memory|abort" "$LOG" | sed 's/^/abort_signals=/'
grep -aE "arena [0-9]|re-sized|cannot hold|stood down|release_arena|shrink_step|allocating .* cudaMalloc failed|slots requested" "$LOG" | sed 's/^/  /' | tail -25
echo "--- /completion coherence ---"
for f in wide1 wide2; do printf "  %s slash4=" "$f"; grep -ac '////' "/tmp/item1/srv-$TAG-$f.out"; done
