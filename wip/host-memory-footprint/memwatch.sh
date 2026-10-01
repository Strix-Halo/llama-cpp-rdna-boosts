#!/bin/bash
# memwatch.sh <seconds> <cmd...>
#
# Sample a child process's resident-memory breakdown and the system memory every 2 s, and copy
# /proc/<pid>/smaps at the peak-RSS moment so the largest regions can be inspected afterwards.
#
#   RssAnon   anonymous allocations (host buffers that are not shmem)
#   RssFile   file-backed mappings  -> the model mmap page cache
#   RssShmem  shared-memory mappings -> GPU-visible / hipHostMalloc-style host buffers
#
# Use HIP_VISIBLE_DEVICES=0 so only the gfx1100 is used (the gfx1036 iGPU otherwise joins a layer
# split and changes the footprint).
#
#   ./memwatch.sh 60 ./build-rocm/bin/llama-bench -m model.gguf -ngl 99 -ncmoe 40 -p 256 -n 0 -r 1
#
set -u
DUR=${1:-60}; shift
if [ $# -eq 0 ]; then echo "usage: $0 <seconds> <cmd...>" >&2; exit 2; fi

"$@" & PID=$!
echo "child pid=$PID : $*"

PEAK=0
for i in $(seq 1 "$DUR"); do
    sleep 2
    if ! kill -0 "$PID" 2>/dev/null; then echo "child exited at t=$((i*2))s"; break; fi
    RSS=$(awk  '/^VmRSS:/{print $2}' /proc/$PID/status 2>/dev/null)
    ANON=$(awk '/^RssAnon:/{print $2}' /proc/$PID/status 2>/dev/null)
    FILE=$(awk '/^RssFile:/{print $2}' /proc/$PID/status 2>/dev/null)
    SHM=$(awk  '/^RssShmem:/{print $2}' /proc/$PID/status 2>/dev/null)
    SW=$(awk   '/^VmSwap:/{print $2}' /proc/$PID/status 2>/dev/null)
    LK=$(awk   '/^VmLck:/{print $2}' /proc/$PID/status 2>/dev/null)
    MF=$(awk   '/^MemFree:/{print $2}' /proc/meminfo)
    MC=$(awk   '/^Cached:/{print $2}' /proc/meminfo)
    MLK=$(awk  '/^Mlocked:/{print $2}' /proc/meminfo)
    printf 't=%3ds rss=%6.0fM anon=%6.0fM file=%6.0fM shmem=%6.0fM swap=%5.0fM vmlck=%5.0fM | free=%6.0fM cached=%6.0fM mlocked=%6.0fM\n' \
        $((i*2)) $((RSS/1024)) $((ANON/1024)) $((FILE/1024)) $((SHM/1024)) $((SW/1024)) $((LK/1024)) \
        $((MF/1024)) $((MC/1024)) $((MLK/1024))
    if [ "${RSS:-0}" -gt "$PEAK" ]; then
        PEAK=$RSS
        cp /proc/$PID/smaps /tmp/memwatch_peak_smaps.txt 2>/dev/null
        cp /proc/$PID/maps  /tmp/memwatch_peak_maps.txt  2>/dev/null
    fi
done
wait "$PID" 2>/dev/null

if [ -f /tmp/memwatch_peak_smaps.txt ]; then
    echo "--- largest RSS regions at peak (kB, then the /proc/<pid>/maps line) ---"
    awk '
        /^[0-9a-f]+-[0-9a-f]+ /{addr=$1}
        /^Size:/{sz=$2}
        /^Rss:/ {if ($2+0 > 20000) print $2, addr, sz}
    ' /tmp/memwatch_peak_smaps.txt 2>/dev/null | sort -rn | head -12
    echo "--- mappings for those addresses (maps) ---"
    for a in $(awk '/^[0-9a-f]+-[0-9a-f]+ /{addr=$1} /^Size:/{sz=$2} /^Rss:/{if ($2+0>20000) print addr}' /tmp/memwatch_peak_smaps.txt | head -12); do
        grep -m1 "^$a " /tmp/memwatch_peak_maps.txt || true
    done
fi
