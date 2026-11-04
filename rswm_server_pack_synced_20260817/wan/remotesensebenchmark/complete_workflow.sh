#!/bin/bash
# Complete Workflow: From Data Generation to LoRA Training to Inference
# 
# This script demonstrates the complete pipeline for fine-tuning Wan2.2
# with the Multi-Direction Sliding Window Benchmark

set -e  # Exit on error

echo "=========================================="
echo "Wan2.2 LoRA Fine-tuning Complete Workflow"
echo "=========================================="

# ============================================================================
# Step 1: Generate Benchmark Dataset
# ============================================================================

echo ""
echo "Step 1: Generating Multi-Direction Benchmark Dataset..."
echo "=========================================="

python remotesensebenchmark/build_multi_direction_benchmark_v2.py \
    --output multi_direction_benchmark_v2.h5 \
    --instances 100 \
    --min-segments 2 \
    --max-segments 4 \
    --steps 8 \
    --seed 42

echo "✓ Dataset generated: multi_direction_benchmark_v2.h5"

# ============================================================================
# Step 2: Visualize Sample Data (Optional)
# ============================================================================

echo ""
echo "Step 2: Visualizing sample data..."
echo "=========================================="

python remotesensebenchmark/visualize_multi_direction_mp4.py \
    --sample 0 \
    --fps 8 \
    --output sample_visualization.mp4

echo "✓ Visualization saved: sample_visualization.mp4"

# ============================================================================
# Step 3: Download Pretrained Model (if not already downloaded)
# ============================================================================

echo ""
echo "Step 3: Checking for pretrained model..."
echo "=========================================="

MODEL_DIR="./Wan2.2-TI2V-5B"

if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading Wan2.2-TI2V-5B model..."
    
    # Using huggingface-cli
    pip install -q "huggingface_hub[cli]"
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir $MODEL_DIR
    
    echo "✓ Model downloaded to: $MODEL_DIR"
else
    echo "✓ Model already exists at: $MODEL_DIR"
fi

# ============================================================================
# Step 4: Train LoRA Model
# ============================================================================

echo ""
echo "Step 4: Training LoRA model..."
echo "=========================================="

python remotesensebenchmark/train_lora.py \
    --model_config ti2v-5B \
    --checkpoint_dir $MODEL_DIR \
    --data_path multi_direction_benchmark_v2.h5 \
    --output_dir ./lora_checkpoints \
    --batch_size 1 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --lora_rank 128 \
    --lora_alpha 128 \
    --device_id 0 \
    --save_interval 2 \
    --log_interval 10

echo "✓ Training completed! Checkpoints saved to: ./lora_checkpoints"

# ============================================================================
# Step 5: Test Inference with LoRA
# ============================================================================

echo ""
echo "Step 5: Testing inference with trained LoRA..."
echo "=========================================="

# Find the latest checkpoint
LATEST_CHECKPOINT=$(ls -td ./lora_checkpoints/checkpoint-epoch-* | head -1)

echo "Using checkpoint: $LATEST_CHECKPOINT"

python remotesensebenchmark/inference_with_lora.py \
    --model_config ti2v-5B \
    --checkpoint_dir $MODEL_DIR \
    --lora_checkpoint $LATEST_CHECKPOINT \
    --prompt "A satellite view moving northeast across urban landscape, then turning south over rural areas" \
    --size 1280*704 \
    --frame_num 33 \
    --output output_lora_test.mp4 \
    --seed 42

echo "✓ Inference completed! Video saved to: output_lora_test.mp4"

# ============================================================================
# Step 6: Compare with Base Model (Optional)
# ============================================================================

echo ""
echo "Step 6: Generating comparison video with base model..."
echo "=========================================="

python generate.py \
    --task ti2v-5B \
    --size 1280*704 \
    --ckpt_dir $MODEL_DIR \
    --offload_model True \
    --convert_model_dtype \
    --t5_cpu \
    --prompt "A satellite view moving northeast across urban landscape, then turning south over rural areas" \
    --frame_num 33 \
    --base_seed 42

echo "✓ Base model video generated for comparison"

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "=========================================="
echo "Workflow Completed Successfully!"
echo "=========================================="
echo ""
echo "Generated Files:"
echo "  - Dataset: multi_direction_benchmark_v2.h5"
echo "  - Visualization: sample_visualization.mp4"
echo "  - LoRA Checkpoints: ./lora_checkpoints/"
echo "  - LoRA Output: output_lora_test.mp4"
echo ""
echo "Next Steps:"
echo "  1. Review the training logs and metrics"
echo "  2. Compare LoRA output with base model output"
echo "  3. Fine-tune hyperparameters if needed"
echo "  4. Train on more data or epochs for better results"
echo ""
echo "=========================================="
