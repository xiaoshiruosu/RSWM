#!/bin/bash
# Safe shared-server smoke launcher. It never picks or occupies all four GPUs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

: "${CUDA_VISIBLE_DEVICES:?Set exactly one GPU allocated to you, e.g. CUDA_VISIBLE_DEVICES=2}"
case "$CUDA_VISIBLE_DEVICES" in
    *,*) echo "FATAL: smoke requires one visible GPU, got $CUDA_VISIBLE_DEVICES" >&2; exit 2 ;;
    *[!0-9]*) echo "FATAL: use one numeric physical GPU index, got $CUDA_VISIBLE_DEVICES" >&2; exit 2 ;;
esac
# Smoke always exercises the stable A800 path even though deployment defaults
# to OFFLOAD=auto and throughput diagnostics may explicitly request 0.
export OFFLOAD=1
# shellcheck disable=SC1091
source "$HERE/a800_env.sh"

python "$HERE/preflight_a800.py" --strict --require-single-visible \
    --min-free-gib "${MIN_FREE_GIB:-70}"
python "$HERE/a800_smoke.py" --level "${SMOKE_LEVEL:-3}"

if [ "${REAL_MODEL_SMOKE:-0}" = "1" ]; then
    : "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR for REAL_MODEL_SMOKE=1}"
    : "${DATA_PATH:?Set DATA_PATH for REAL_MODEL_SMOKE=1}"
    OUTDIR="${OUTDIR:-$RSWM_BASE/work/a800_real_smoke_$(date +%Y%m%d_%H%M%S)}"
    POSE_MODE="${POSE_MODE:-native}"
    echo "Starting explicitly requested real 33-frame one-sample smoke: $OUTDIR"
    (cd "$ROOT/lora_remotesense/new" && python train.py \
        --checkpoint_dir "$CHECKPOINT_DIR" --data_path "$DATA_PATH" \
        --output_dir "$OUTDIR" --hardware_profile a800_80g \
        --offload 1 \
        --attention_backend auto --batch_size 1 --gradient_accumulation_steps 1 \
        --max_frames 33 --max_train_samples 1 --max_val_samples 1 --num_epochs 1 \
        --num_workers 0 --no_8bit_adam --pose_mode "$POSE_MODE")
fi
