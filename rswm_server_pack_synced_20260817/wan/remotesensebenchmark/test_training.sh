#!/bin/bash
# Quick Test Script for LoRA Training
# 
# This script runs a quick test with minimal data to verify the training pipeline works

set -e  # Exit on error

echo "=========================================="
echo "LoRA Training Quick Test"
echo "=========================================="

# Configuration
MODEL_CONFIG="ti2v-5B"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR explicitly}"
DATA_PATH="${DATA_PATH:-multi_direction_benchmark_v2.h5}"
OUTPUT_DIR="${OUTPUT_DIR:-./lora_test_checkpoints}"
DEVICE_ID="${DEVICE_ID:-${LOCAL_RANK:-0}}"  # logical index inside CUDA_VISIBLE_DEVICES

# Test parameters (minimal for quick testing)
BATCH_SIZE=1
NUM_EPOCHS=1  # 只训练 1 个 epoch 用于测试
LORA_RANK=64
LEARNING_RATE=1e-4

echo ""
echo "Test Configuration:"
echo "  Model: $MODEL_CONFIG"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  Data: $DATA_PATH"
echo "  Epochs: $NUM_EPOCHS (test mode)"
echo "  Batch Size: $BATCH_SIZE"
echo "  LoRA Rank: $LORA_RANK"
echo "=========================================="

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    echo "Please generate the benchmark dataset first:"
    echo "  python remotesensebenchmark/build_multi_direction_benchmark_v2.py --output $DATA_PATH --instances 10"
    exit 1
fi

# Check if checkpoint directory exists
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "Please download the model first or update CHECKPOINT_DIR in this script"
    exit 1
fi

echo ""
echo "Starting training test..."
echo "=========================================="

# Run training with test parameters
python remotesensebenchmark/train_lora.py \
    --model_config $MODEL_CONFIG \
    --checkpoint_dir $CHECKPOINT_DIR \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lora_rank $LORA_RANK \
    --learning_rate $LEARNING_RATE \
    --t5_cpu \
    --save_interval 1 \
    --log_interval 5 \
    --device_id "$DEVICE_ID"

echo ""
echo "=========================================="
echo "Test completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check the output directory: $OUTPUT_DIR"
echo "  2. Verify checkpoint files were created"
echo "  3. Run full training with more epochs"
echo ""
echo "Full training command:"
echo "  python remotesensebenchmark/train_lora.py \\"
echo "    --model_config $MODEL_CONFIG \\"
echo "    --checkpoint_dir $CHECKPOINT_DIR \\"
echo "    --data_path $DATA_PATH \\"
echo "    --output_dir ./lora_checkpoints \\"
echo "    --num_epochs 10 \\"
echo "    --batch_size 1 \\"
echo "    --lora_rank 64 \\"
echo "    --learning_rate 1e-4 \\"
echo "    --t5_cpu"
echo ""
