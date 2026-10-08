#!/usr/bin/env bash
# Prefill-logit KLD gate (issue #113).
#
# Automation of `benchmarks/prefill-logit-methodology.md`: compare the current build's
# prefill logits against a recorded known-good base with `llama-perplexity
# --kl-divergence`, and fail unless the divergence is rounding-level
# (mean KLD <= MAX_KLD, same-top-p >= MIN_TOPP).
#
# WHY THIS EXISTS:
#   Same-seed coherence and the MTP gate sample the decode distribution *from the
#   post-prefill state*; a uniform prefill shift is invisible to both.  The
#   default-on BF16/WMMA chunked GDN shipped that way in the 16-block set (issue
#   #113: mean KLD 0.032 on gfx1201, 0.62 on gfx1100) and passed every existing
#   gate.  This is the instrument that sees it.
#
# MODES:
#   (default)        compare the current build against $BASE (must already exist)
#   --record         (re)record $BASE from the current build, then compare it to
#                    itself (a self-test of the harness; mean KLD ~ 0)
#   --ab "K=V ..."   exact-fallback A/B: record $BASE with the given env applied
#                    (the known-good/fallback arm), then compare the *default*
#                    arm against it.  This is the check that catches issue #113:
#                    --ab "GGML_CUDA_GDN_CHUNKED_BF16=0" on a build whose default
#                    is the bf16 kernel.
#
# Usage:
#   ./gate-prefill-logits.sh [options]
#
# Options / environment (the environment variable is the default for the flag):
#   --base FILE       BASE           base logits file (default:
#                                    $BASE_DIR/prefill-logit-<arch>-<modelstem>.kld)
#   --base-dir DIR    BASE_DIR       default base directory
#                                    (default: ~/.cache/rdna-boosts/gates)
#   --model FILE      MODEL          model gguf (default: auto: Q8_0 if the largest
#                                    single card has >= 28 GiB, else the first
#                                    available 24-GiB-class quant)
#   --corpus FILE     CORPUS         default /llm/models/wikitext-2-raw/wiki.test.raw
#   --ppl FILE        LLAMA_PPL      default ~/llama.cpp/build-rocm-hybrid/bin/llama-perplexity
#   --ctx N           CTX            default 512
#   --chunks N        CHUNKS         default 40
#   --ngl N           NGL            default 99
#   --fa on|off|auto  FA             default on
#   --threads N       THREADS        default 8
#   --max-kld F       MAX_KLD        default 0.005   (fail if mean KLD > F)
#   --min-top-p F     MIN_TOPP       default 98.0    (fail if same-top-p < F)
#   --device N        DEVICE         HIP_VISIBLE_DEVICES (default 0; single card)
#   --timeout N       TIMEOUT        per-run seconds (default 1800)
#   --keep            KEEP=1         keep the per-run logs
#   --allow-base-mismatch            skip the base-meta compatibility check
#   --record, --ab SPEC
#
# Exit: 0 = PASS, 1 = FAIL (gate), 2 = usage/prerequisite error.
#
# This is a pre-release gate, not CI: it needs a ROCm GPU, the pinned model and a
# multi-GB base file.  Record the base from the previous release's build once, then
# run the candidate (default, and --ab for any new approximate kernel) before
# tagging.  Record the result in the release's WORKLOG.md entry.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LLAMA_PPL="${LLAMA_PPL:-$HOME/llama.cpp/build-rocm-hybrid/bin/llama-perplexity}"
CORPUS="${CORPUS:-/llm/models/wikitext-2-raw/wiki.test.raw}"
BASE_DIR="${BASE_DIR:-$HOME/.cache/rdna-boosts/gates}"
CTX="${CTX:-512}"
CHUNKS="${CHUNKS:-40}"
NGL="${NGL:-99}"
FA="${FA:-on}"
THREADS="${THREADS:-8}"
MAX_KLD="${MAX_KLD:-0.005}"
MIN_TOPP="${MIN_TOPP:-98.0}"
DEVICE="${DEVICE:-0}"
TIMEOUT="${TIMEOUT:-1800}"
KEEP="${KEEP:-0}"
ALLOW_BASE_MISMATCH=0

RECORD=0
RECORD_ENV=""
BASE="${BASE:-}"
MODEL="${MODEL:-}"

usage() { sed -n '2,57p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --base)                 BASE="$2"; shift 2 ;;
        --base-dir)             BASE_DIR="$2"; shift 2 ;;
        --model)                MODEL="$2"; shift 2 ;;
        --corpus)               CORPUS="$2"; shift 2 ;;
        --ppl)                  LLAMA_PPL="$2"; shift 2 ;;
        --ctx)                  CTX="$2"; shift 2 ;;
        --chunks)               CHUNKS="$2"; shift 2 ;;
        --ngl)                  NGL="$2"; shift 2 ;;
        --fa)                   FA="$2"; shift 2 ;;
        --threads)              THREADS="$2"; shift 2 ;;
        --max-kld)              MAX_KLD="$2"; shift 2 ;;
        --min-top-p)            MIN_TOPP="$2"; shift 2 ;;
        --device)               DEVICE="$2"; shift 2 ;;
        --timeout)              TIMEOUT="$2"; shift 2 ;;
        --keep)                 KEEP=1; shift ;;
        --allow-base-mismatch)  ALLOW_BASE_MISMATCH=1; shift ;;
        --record)               RECORD=1; RECORD_ENV=""; shift ;;
        --ab)                   RECORD=1; RECORD_ENV="${2:-}"; shift 2 ;;
        -h|--help)              usage 0 ;;
        *) echo "unknown argument: $1" >&2; usage 2 ;;
    esac
done

fail() { echo "FAIL: $*" >&2; KEEP=1; exit 1; }
die()  { echo "ERROR: $*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# prerequisites
# ---------------------------------------------------------------------------
[ -x "$LLAMA_PPL" ] || die "llama-perplexity not executable: $LLAMA_PPL (set LLAMA_PPL)"
[ -f "$CORPUS" ]    || die "corpus not found: $CORPUS"
command -v awk >/dev/null || die "awk required"

ARCH="$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+' || true)"
ARCH="${ARCH:-gpu}"

vram_gib() {
    local best=0 f b gib
    for f in /sys/class/drm/card*/device/mem_info_vram_total; do
        [ -r "$f" ] || continue
        b="$(cat "$f" 2>/dev/null)" || continue
        case "$b" in ''|*[!0-9]*) continue ;; esac
        gib=$((b / 1073741824))
        [ "$gib" -gt "$best" ] && best="$gib"
    done
    echo "$best"
}

# Auto model: Q8_0 when a single card can hold it, else the first available
# 24-GiB-class 27B quant (the same lineage the methodology pins for gfx1100).
resolve_model() {
    local q8=/llm/models/Qwen3.8/27B/Q8_0/Qwen3.8-27B-Q8_0.gguf
    local cands=(
        /llm/models/Qwen3.8/27B/Q4_K_M/Qwen3.8-27B-UD-Q4_K_M.gguf
        /llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf
        /llm/models/Qwen3.8/27B/Q5_K/Qwen3.8-27B-UD-Q5_K_M.gguf
        /llm/models/Qwen3.8/27B/Q6_K/Qwen3.8-27B-Q6_K.gguf
    )
    if [ -n "$MODEL" ]; then return 0; fi
    local gib; gib="$(vram_gib)"
    if [ "$gib" -ge 28 ] && [ -f "$q8" ]; then
        MODEL="$q8"
        return 0
    fi
    local c
    for c in "${cands[@]}"; do
        if [ -f "$c" ]; then MODEL="$c"; return 0; fi
    done
    die "no auto model found (set MODEL); biggest single card = ${gib} GiB"
}
resolve_model
[ -f "$MODEL" ] || die "model not found: $MODEL"

MODEL_STEM="$(basename "$MODEL" .gguf)"
if [ -z "$BASE" ]; then
    mkdir -p "$BASE_DIR" || die "cannot create base dir: $BASE_DIR"
    BASE="$BASE_DIR/prefill-logit-${ARCH}-${MODEL_STEM}.kld"
fi
META="$BASE.meta"

CORPUS_SHA="$(sha256sum "$CORPUS" | awk '{print $1}')"

if [ "$RECORD" -eq 0 ] && [ ! -f "$BASE" ]; then
    die "base not found: $BASE
       record it first from a known-good build with: $0 --record
       (or point --base at an existing file)"
fi

LOGDIR="${LOGDIR:-$(mktemp -d)}"
mkdir -p "$LOGDIR"
cleanup() { [ "$KEEP" -eq 1 ] || rm -rf "$LOGDIR"; }
trap cleanup EXIT

echo "==> prefill-logit KLD gate (issue #113)"
echo "    ppl=$LLAMA_PPL"
echo "    model=$MODEL"
echo "    corpus=$CORPUS  (sha256=${CORPUS_SHA:0:16}...)"
echo "    arch=$ARCH  device=$DEVICE  ctx=$CTX chunks=$CHUNKS fa=$FA ngl=$NGL threads=$THREADS"
echo "    base=$BASE"
echo "    thresholds: mean KLD <= $MAX_KLD, same-top-p >= ${MIN_TOPP}%"
[ "$RECORD" -eq 1 ] && echo "    mode=record${RECORD_ENV:+ (base arm: $RECORD_ENV)}"
[ -n "$RECORD_ENV" ] && echo "    --ab: base arm = '$RECORD_ENV', candidate arm = default"
echo "    logs=$LOGDIR"

# ---------------------------------------------------------------------------
# run helper
# ---------------------------------------------------------------------------
run_ppl() {
    local extra="$1" out="$2"; shift 2
    # shellcheck disable=SC2086  # $extra is an intended word-split K=V list
    timeout "$TIMEOUT" env HIP_VISIBLE_DEVICES="$DEVICE" ${extra:-} \
        "$LLAMA_PPL" -m "$MODEL" -f "$CORPUS" \
        -c "$CTX" --chunks "$CHUNKS" -ngl "$NGL" -fa "$FA" -t "$THREADS" \
        "$@" >"$out" 2>&1
}

mean_kld()  { sed -n 's/^Mean[[:space:]]*KLD:[[:space:]]*\([-0-9.][0-9.]*\).*/\1/p' "$1" | head -1; }
same_topp() { sed -n 's/^Same top p:[[:space:]]*\([0-9.][0-9.]*\).*/\1/p' "$1" | head -1; }
final_ppl() { sed -n 's/.*Final estimate: *PPL *= *\([0-9.][0-9.]*\).*/\1/p' "$1" | head -1; }
# A human-readable build reference: the fork checkout's HEAD when one encloses the
# binary, else the wrapper's own hash (llama-perplexity prints no build line).
build_ref() {
    local dir
    dir="$(cd "$(dirname "$LLAMA_PPL")/../.." 2>/dev/null && pwd)"
    if [ -n "$dir" ] && git -C "$dir" rev-parse --short=12 HEAD >/dev/null 2>&1; then
        git -C "$dir" rev-parse --short=12 HEAD
    else
        sha256sum "$LLAMA_PPL" | awk '{print substr($1,1,12)}'
    fi
}
le() { awk -v a="$1" -v b="$2" 'BEGIN{ exit !(a+0 <= b+0) }'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{ exit !(a+0 >= b+0) }'; }

# ---------------------------------------------------------------------------
# base compatibility (model/corpus/ctx/chunks must match to be a valid compare)
# ---------------------------------------------------------------------------
base_meta_field() { [ -f "$META" ] && sed -n "s/^$1=//p" "$META" | head -1 || true; }

check_base_compat() {
    [ -f "$META" ] || { echo "    WARN: no base metadata ($META); cannot verify the base matches this model/corpus"; return 0; }
    local problems=0 k v want
    for k in model corpus_sha256 ctx chunks; do
        v="$(base_meta_field "$k")"
        case "$k" in
            model)         want="$MODEL" ;;
            corpus_sha256) want="$CORPUS_SHA" ;;
            ctx)           want="$CTX" ;;
            chunks)        want="$CHUNKS" ;;
        esac
        if [ -n "$v" ] && [ "$v" != "$want" ]; then
            echo "    base meta mismatch: $k='$v' != current '$want'" >&2
            problems=1
        fi
    done
    if [ "$problems" -eq 1 ]; then
        if [ "$ALLOW_BASE_MISMATCH" -eq 1 ]; then
            echo "    WARN: base does not match the current model/corpus/settings (--allow-base-mismatch)"
        else
            fail "base $BASE does not match the current model/corpus/settings; re-record it (--record) or pass --allow-base-mismatch"
        fi
    fi
}

# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------
if [ "$RECORD" -eq 1 ]; then
    rec_log="$LOGDIR/base-record.log"
    echo
    echo "==> recording base with arm '${RECORD_ENV:-<default>}' (may take a few minutes)"
    [ -f "$BASE" ] && echo "    WARN: overwriting existing base $BASE"
    if ! run_ppl "$RECORD_ENV" "$rec_log" --kl-divergence-base "$BASE"; then
        echo "--- tail of $rec_log ---" >&2; tail -20 "$rec_log" >&2
        fail "base recording run failed"
    fi
    [ -f "$BASE" ] || fail "base file was not written: $BASE"
    grep -q "Final estimate" "$rec_log" || { tail -20 "$rec_log" >&2; fail "base recording did not complete (no 'Final estimate')"; }

    b_ppl="$(final_ppl "$rec_log")"
    b_ref="$(build_ref)"
    b_sha=""
    if [ "${SKIP_BASE_HASH:-0}" != "1" ]; then
        b_sha="$(sha256sum "$BASE" | awk '{print $1}')"
    fi
    {
        echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "host=$(hostname)"
        echo "arch=$ARCH"
        echo "build=$b_ref"
        echo "model=$MODEL"
        echo "corpus=$CORPUS"
        echo "corpus_sha256=$CORPUS_SHA"
        echo "ctx=$CTX"
        echo "chunks=$CHUNKS"
        echo "ngl=$NGL"
        echo "fa=$FA"
        echo "arm=${RECORD_ENV:-<default>}"
        echo "base_ppl=$b_ppl"
        echo "base_sha256=$b_sha"
    } > "$META"
    echo "    base PPL=$b_ppl build=${b_ref:-?} sha256=${b_sha:-<skipped>}"
    echo "    wrote $META"
    # If we recorded with an extra env arm, comparing the default against it is
    # the whole point; if we recorded the default, the compare is a self-test.
fi

if [ "$RECORD" -eq 0 ]; then
    check_base_compat
fi

# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------
cmp_log="$LOGDIR/candidate-compare.log"
echo
echo "==> comparing candidate (default${RECORD_ENV:+ vs base arm '$RECORD_ENV'})"
if ! run_ppl "" "$cmp_log" --kl-divergence-base "$BASE" --kl-divergence; then
    echo "--- tail of $cmp_log ---" >&2; tail -20 "$cmp_log" >&2
    fail "candidate run failed (see $cmp_log)"
fi
grep -qE '^Mean[[:space:]]*KLD:' "$cmp_log" || { tail -20 "$cmp_log" >&2; fail "candidate run did not complete (no mean KLD)"; }

KLD="$(mean_kld "$cmp_log")"
TOPP="$(same_topp "$cmp_log")"
CAND_PPL="$(final_ppl "$cmp_log")"
CAND_REF="$(build_ref)"
[ -n "$KLD" ]  || { tail -30 "$cmp_log" >&2; fail "could not parse mean KLD from $cmp_log"; }
[ -n "$TOPP" ] || { tail -30 "$cmp_log" >&2; fail "could not parse same-top-p from $cmp_log"; }

echo
[ -n "$CAND_PPL" ] && ppl_note=" PPL=$CAND_PPL" || ppl_note=""
echo "    candidate: build=${CAND_REF:-?}${ppl_note}"
echo "    mean KLD = $KLD   (limit $MAX_KLD)"
echo "    same top p = $TOPP %   (floor $MIN_TOPP %)"

rc=0
if ! le "$KLD" "$MAX_KLD"; then
    echo "FAIL: mean KLD $KLD > $MAX_KLD" >&2; rc=1
fi
if ! ge "$TOPP" "$MIN_TOPP"; then
    echo "FAIL: same top p $TOPP % < $MIN_TOPP %" >&2; rc=1
fi

echo
if [ "$rc" -eq 0 ]; then
    echo "==> PASS: prefill logits within rounding tolerance of the base"
    echo "    base=$BASE"
    [ "$KEEP" -eq 1 ] && echo "    logs kept in $LOGDIR"
    exit 0
fi
echo "==> FAIL: prefill-logit divergence (issue #113 class)" >&2
echo "    logs in $LOGDIR (set KEEP=1 to retain on success too)" >&2
KEEP=1
exit 1
