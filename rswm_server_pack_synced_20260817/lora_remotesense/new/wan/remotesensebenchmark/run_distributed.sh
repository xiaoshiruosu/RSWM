#!/bin/bash
# Explicit multi-GPU launcher. It never chooses GPUs and never overwrites
# CUDA_VISIBLE_DEVICES; the user/scheduler owns physical GPU allocation.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NUM_GPUS="${NUM_GPUS:?Set NUM_GPUS explicitly, e.g. NUM_GPUS=2}"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to GPUs allocated to this job}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR under the project storage root}"
TRAIN_ENTRY="${TRAIN_ENTRY:-$HERE/train_lora.py}"
MASTER_PORT="${MASTER_PORT:-29500}"

[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_GPUS must be a positive integer" >&2; exit 2; }
VISIBLE_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [ "$NUM_GPUS" -gt "$VISIBLE_COUNT" ]; then
    echo "Requested NUM_GPUS=$NUM_GPUS but torch sees only $VISIBLE_COUNT devices inside CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"

echo "Explicit DDP launch: visible=$CUDA_VISIBLE_DEVICES nproc=$NUM_GPUS"
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
    "$TRAIN_ENTRY" \
    --model_config ti2v-5B \
    --checkpoint_dir "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --lora_rank "${LORA_RANK:-64}" \
    --lora_alpha "${LORA_ALPHA:-64}" \
    --train_ratio "${TRAIN_RATIO:-0.9}" \
    --val_ratio "${VAL_RATIO:-0.1}" \
    --save_interval 1 \
    --num_workers "${NUM_WORKERS:-0}"
