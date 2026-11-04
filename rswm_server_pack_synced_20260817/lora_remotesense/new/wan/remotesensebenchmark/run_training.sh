#!/bin/bash
# LoRA Training Script for Wan2.2 with Multi-Direction Benchmark
# 
# This script provides example commands for training with different configurations

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Model and data paths
MODEL_CONFIG="ti2v-5B"  # Options: ti2v-5B, t2v-A14B, i2v-A14B
CHECKPOINT_DIR="./Wan2.2-TI2V-5B"
DATA_PATH="./multi_direction_benchmark_v2.h5"
OUTPUT_DIR="./lora_checkpoints"

# Training hyperparameters
BATCH_SIZE=1
NUM_EPOCHS=10
LEARNING_RATE=1e-4
LORA_RANK=128
LORA_ALPHA=128

# Device configuration
DEVICE_ID=0
NUM_WORKERS=4

# Logging
USE_WANDB=false
WANDB_PROJECT="wan2.2-lora-finetuning"

# ============================================================================
# Training Commands
# ============================================================================

echo "=========================================="
echo "Wan2.2 LoRA Fine-tuning"
echo "=========================================="
echo "Model: $MODEL_CONFIG"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Basic training command
python remotesensebenchmark/train_lora.py \
    --model_config $MODEL_CONFIG \
    --checkpoint_dir $CHECKPOINT_DIR \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --device_id $DEVICE_ID \
    --num_workers $NUM_WORKERS \
    --save_interval 1 \
    --log_interval 10

# ============================================================================
# Alternative Configurations
# ============================================================================

# Uncomment the configuration you want to use:

# --- RTX 4090 (24GB) Configuration ---
# python remotesensebenchmark/train_lora.py \
#     --model_config ti2v-5B \
#     --checkpoint_dir ./Wan2.2-TI2V-5B \
#     --data_path ./multi_direction_benchmark_v2.h5 \
#     --output_dir ./lora_checkpoints_4090 \
#     --batch_size 1 \
#     --num_epochs 10 \
#     --learning_rate 1e-4 \
#     --lora_rank 64 \
#     --lora_alpha 64 \
#     --device_id 0

# --- A100 (40GB) Configuration ---
# python remotesensebenchmark/train_lora.py \
#     --model_config ti2v-5B \
#     --checkpoint_dir ./Wan2.2-TI2V-5B \
#     --data_path ./multi_direction_benchmark_v2.h5 \
#     --output_dir ./lora_checkpoints_a100 \
#     --batch_size 2 \
#     --num_epochs 10 \
#     --learning_rate 1e-4 \
#     --lora_rank 128 \
#     --lora_alpha 128 \
#     --device_id 0

# --- A100 (80GB) with Large Model ---
# python remotesensebenchmark/train_lora.py \
#     --model_config t2v-A14B \
#     --checkpoint_dir ./Wan2.2-T2V-A14B \
#     --data_path ./multi_direction_benchmark_v2.h5 \
#     --output_dir ./lora_checkpoints_a100_80g \
#     --batch_size 2 \
#     --num_epochs 10 \
#     --learning_rate 5e-5 \
#     --lora_rank 128 \
#     --lora_alpha 128 \
#     --device_id 0

# --- With Weights & Biases Logging ---
# python remotesensebenchmark/train_lora.py \
#     --model_config ti2v-5B \
#     --checkpoint_dir ./Wan2.2-TI2V-5B \
#     --data_path ./multi_direction_benchmark_v2.h5 \
#     --output_dir ./lora_checkpoints \
#     --batch_size 1 \
#     --num_epochs 10 \
#     --learning_rate 1e-4 \
#     --lora_rank 128 \
#     --lora_alpha 128 \
#     --use_wandb \
#     --wandb_project wan2.2-lora \
#     --wandb_run_name experiment-1

# --- Training on Specific Scene Categories ---
# python remotesensebenchmark/train_lora.py \
#     --model_config ti2v-5B \
#     --checkpoint_dir ./Wan2.2-TI2V-5B \
#     --data_path ./multi_direction_benchmark_v2.h5 \
#     --output_dir ./lora_checkpoints_urban \
#     --scene_categories general urban \
#     --batch_size 1 \
#     --num_epochs 10 \
#     --learning_rate 1e-4

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=========================================="

