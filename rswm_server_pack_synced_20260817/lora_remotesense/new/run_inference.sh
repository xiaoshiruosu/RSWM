#!/bin/bash
# =============================================================================
# 单图推理（用训练好的 LoRA 生成视频，验证训练效果）
#
# 用法：
#   LORA_CKPT=/path/to/best_checkpoint IMAGE=/path/x.jpg bash run_inference.sh
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
LORA_CKPT="${LORA_CKPT:?ERROR: 请设置 LORA_CKPT=/path/to/best_checkpoint}"
IMAGE="${IMAGE:?ERROR: 请设置 IMAGE=/path/to/image.jpg}"
PROMPT="${PROMPT:-A video sequence of a remote sensing scene from satellite view.}"
OUTPUT="${OUTPUT:-${STORAGE_ROOT}/work/inference_$(date +%Y%m%d_%H%M%S).mp4}"
FRAMES="${FRAMES:-33}"
STEPS="${STEPS:-70}"
SIZE="${SIZE:-480*832}"

mkdir -p "$(dirname "$OUTPUT")"

echo "推理: image=$IMAGE  lora=$LORA_CKPT  out=$OUTPUT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python "$HERE/inference.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --lora_path "$LORA_CKPT" \
    --image "$IMAGE" \
    --prompt "$PROMPT" \
    --output "$OUTPUT" \
    --frame_num "$FRAMES" \
    --sampling_steps "$STEPS" \
    --size "$SIZE" \
    --device "${DEVICE:-cuda}"

echo "[Done] $OUTPUT"
