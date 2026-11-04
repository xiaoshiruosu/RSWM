#!/bin/bash
# =============================================================================
# 验证集评估（SSIM/PSNR/LPIPS/FID/FVD）
#
# 用法：
#   LORA_CKPT=/path/best DATA_PATH=/path/train.h5 SPLIT_FILE=/path/train_val_split.json \
#       bash run_evaluate.sh
#
#   # 仅算指标（已有推理 npy）：
#   METRICS_ONLY=1 OUTDIR=/path/eval_output bash run_evaluate.sh
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$HERE/wan" ]; then
    PROJECT_ROOT="$HERE"
else
    PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
fi

STORAGE_ROOT="${RSWM_BASE:-$PROJECT_ROOT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${STORAGE_ROOT}/models/lingbot-world-base-cam}"
LORA_CKPT="${LORA_CKPT:-}"
DATA_PATH="${DATA_PATH:-${STORAGE_ROOT}/data/remotesense/train.h5}"
SPLIT_FILE="${SPLIT_FILE:-}"
OUTDIR="${OUTDIR:-${STORAGE_ROOT}/work/eval_$(date +%Y%m%d_%H%M%S)}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"   # 0 = 全量
METRICS_ONLY="${METRICS_ONLY:-0}"

mkdir -p "$OUTDIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARGS=(--checkpoint_dir "$CHECKPOINT_DIR" --output_dir "$OUTDIR")
ARGS+=(--device "${DEVICE:-cuda}")
if [ "$METRICS_ONLY" = "1" ]; then
    ARGS+=(--metrics_only)
else
    [ -n "$LORA_CKPT" ] || { echo "ERROR: 需 LORA_CKPT"; exit 1; }
    [ -n "$SPLIT_FILE" ] || { echo "ERROR: 需 SPLIT_FILE (由训练生成的 train_val_split.json)"; exit 1; }
    ARGS+=(--lora_path "$LORA_CKPT" --data_path "$DATA_PATH" --split_file "$SPLIT_FILE")
    [ "$MAX_SAMPLES" != "0" ] && ARGS+=(--max_samples "$MAX_SAMPLES")
fi

echo "评估: ${ARGS[*]}"
python "$HERE/evaluate.py" "${ARGS[@]}"

echo "[Done] $OUTDIR"
