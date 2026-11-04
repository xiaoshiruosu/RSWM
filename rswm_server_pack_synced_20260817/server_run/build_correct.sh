#!/usr/bin/env bash
# Canonical, offline-only RSRoll paper dataset build entry.
#
# This is the only supported dataset-build entry. It deliberately delegates to
# lora_remotesense/build_dataset_from_benchmark.py and never to new/ builders.
set -Eeuo pipefail
umask 077

ROOT="${ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
BUILDER="$ROOT/lora_remotesense/build_dataset_from_benchmark.py"
VALIDATOR="$ROOT/server_run/check_dataset.py"
PY="${PY:-${RSWM_PYTHON:-python3}}"

N="${N:-100}"
TIER="${TIER:-aid}"
OUT="${OUT:-$ROOT/data/train_${N}_aid.h5}"
TMP="${TMP:-$ROOT/tmp_bench_${N}}"

case "$OUT" in /*) ;; *) OUT="$ROOT/$OUT" ;; esac
case "$TMP" in /*) ;; *) TMP="$ROOT/$TMP" ;; esac

fatal() {
    echo "FATAL: $*" >&2
    exit 2
}

[[ "$N" =~ ^[1-9][0-9]*$ ]] || fatal "N must be a positive integer, got $N"
(( N % 4 == 0 )) || fatal "N=$N must be divisible by four for the frozen scene balance"
[[ "$TIER" == "aid" ]] || fatal "paper dataset tier is fixed to AID, got TIER=$TIER"
[[ -f "$BUILDER" && -f "$VALIDATOR" ]] || \
    fatal "canonical builder/validator is missing under $ROOT"
for dependency in \
    "$ROOT/wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py" \
    "$ROOT/lora_training/camera_trajectory.py"; do
    [[ -f "$dependency" ]] || fatal "package dependency is missing: $dependency"
done
command -v "$PY" >/dev/null 2>&1 || [[ -x "$PY" ]] || \
    fatal "Python is unavailable: $PY"

# Fail closed on the frozen paper protocol. The grep guard catches accidental
# replacement of the canonical module before a many-hour build is started.
grep -q "DEFAULT_STEPS_PER_SEGMENT = 9" "$BUILDER" || \
    fatal "$BUILDER is not the steps=9 canonical builder"
grep -q "DEFAULT_WINDOW_HEIGHT = 128" "$BUILDER" || \
    fatal "$BUILDER is not the 128x222 canonical builder"
grep -q "DEFAULT_WINDOW_WIDTH = 222" "$BUILDER" || \
    fatal "$BUILDER is not the 128x222 canonical builder"

# SemanticGate is mandatory and must resolve to a staged local CLIP snapshot.
[[ "${RSWM_SEMGATE:-1}" == "1" ]] || \
    fatal "RSWM_SEMGATE must be exactly 1; paper data may not disable SemanticGate"
export RSWM_SEMGATE=1
[[ -n "${RSWM_SEMGATE_MODEL:-}" ]] || \
    fatal "RSWM_SEMGATE_MODEL must point to a local CLIP snapshot"
case "$RSWM_SEMGATE_MODEL" in /*) ;; *) fatal "RSWM_SEMGATE_MODEL must be an absolute local path" ;; esac
[[ -d "$RSWM_SEMGATE_MODEL" ]] || \
    fatal "local CLIP snapshot does not exist: $RSWM_SEMGATE_MODEL"
[[ -r "$RSWM_SEMGATE_MODEL/config.json" \
   && -r "$RSWM_SEMGATE_MODEL/preprocessor_config.json" ]] || \
    fatal "local CLIP snapshot is missing config/preprocessor files: $RSWM_SEMGATE_MODEL"
[[ -r "$RSWM_SEMGATE_MODEL/model.safetensors" \
   || -r "$RSWM_SEMGATE_MODEL/pytorch_model.bin" ]] || \
    fatal "local CLIP snapshot is missing model weights: $RSWM_SEMGATE_MODEL"

if [[ -n "${LOCAL_AID_DIR:-}" ]]; then
    case "$LOCAL_AID_DIR" in /*) ;; *) fatal "LOCAL_AID_DIR must be an absolute path" ;; esac
    [[ -d "$LOCAL_AID_DIR" ]] || fatal "LOCAL_AID_DIR does not exist: $LOCAL_AID_DIR"
fi

# All Hugging Face consumers run fail-closed in offline mode. AID must already
# be in HF_HOME or be supplied through LOCAL_AID_DIR.
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

echo "=================================================================="
echo " canonical offline build  N=$N tier=AID"
echo " ROOT=$ROOT"
echo " OUT=$OUT"
echo " TMP=$TMP"
echo " CLIP=$RSWM_SEMGATE_MODEL"
echo " semgate_min_frac=${RSWM_SEMGATE_MIN_FRAC:-<frozen-default>}"
echo " semgate_min_frac_water=${RSWM_SEMGATE_MIN_FRAC_WATER:-<inherit>}"
echo " semgate_min_frac_urban=${RSWM_SEMGATE_MIN_FRAC_URBAN:-<inherit>}"
echo " semgate_min_frac_rural=${RSWM_SEMGATE_MIN_FRAC_RURAL:-<inherit>}"
echo " protocol=window 128x222, target 480x832, steps 9, frames 17/25/33"
echo " overlap=[0.0,0.1) SemanticGate=enabled"
echo "=================================================================="

PARTIAL="$OUT.partial"

validate_canonical_h5() {
    "$PY" "$VALIDATOR" "$1" --validate --expect-n "$N"
}

quarantine_invalid_h5() {
    local source="$1" reason="$2" stamp destination
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    destination="${source}.invalid.${stamp}.$$"
    [[ ! -e "$destination" ]] || fatal "quarantine destination already exists: $destination"
    mv -- "$source" "$destination"
    echo "[quarantine] $reason"
    echo "[quarantine] moved $source -> $destination"
}

FINAL_READY=0
[[ ! -e "$OUT" || -f "$OUT" ]] || fatal "OUT exists but is not a regular file: $OUT"
if [[ -f "$OUT" ]]; then
    echo "[reuse] final dataset already exists; validating before reuse"
    if validate_canonical_h5 "$OUT"; then
        FINAL_READY=1
        echo "[reuse] existing final dataset passed canonical validation"
    else
        quarantine_invalid_h5 "$OUT" "final dataset failed canonical validation; rebuilding"
    fi
fi

if (( FINAL_READY == 0 )); then
    PARTIAL_READY=0
    [[ ! -e "$PARTIAL" || -f "$PARTIAL" ]] || \
        fatal "partial output exists but is not a regular file: $PARTIAL"
    if [[ -f "$PARTIAL" ]]; then
        echo "[recover] partial output exists; validating before deciding whether to resume publication"
        if validate_canonical_h5 "$PARTIAL"; then
            PARTIAL_READY=1
            echo "[recover] partial output is complete and valid; publishing without rebuilding"
        else
            quarantine_invalid_h5 "$PARTIAL" "partial dataset failed canonical validation; rebuilding"
        fi
    fi

    if (( PARTIAL_READY == 0 )); then
        mkdir -p -- "$(dirname -- "$OUT")" "$TMP"
        BUILD_COMMAND=(
            "$PY" "$BUILDER"
            -o "$PARTIAL" -n "$N"
            --dataset-tier aid
            --window-height 128 --window-width 222
            --steps 9
            --target-height 480 --target-width 832
            --min-overlap-ratio 0.0 --max-overlap-ratio 0.1
            --temp-dir "$TMP"
        )
        [[ -n "${LOCAL_AID_DIR:-}" ]] && \
            BUILD_COMMAND+=(--local-aid-dir "$LOCAL_AID_DIR")
        echo "[build] starting canonical builder; partial output=$PARTIAL"
        (cd -- "$ROOT" && "${BUILD_COMMAND[@]}")
        validate_canonical_h5 "$PARTIAL"
    fi

    # PARTIAL and OUT are in the same directory, so rename is atomic and a
    # consumer can never mistake an open HDF5 for a completed dataset.
    mv -- "$PARTIAL" "$OUT"
    echo "[publish] atomically published $OUT"
fi

echo "FINAL_RESULT=PASS"
echo "dataset=$OUT"
