#!/bin/bash
# =============================================================================
# A800/A100 80GB LoRA 微调启动脚本（裸机 / AutoDL）
#
# 用法：
#   bash run_train.sh                           # OFFLOAD=auto（A800 解析为安全路径）
#   OFFLOAD=1 bash run_train.sh                 # 明确启用 offload（smoke）
#   OFFLOAD=0 bash run_train.sh                 # 显式吞吐/OOM 边界测试
#   NUM_EPOCHS=5 bash run_train.sh              # 自定义 epoch 数
#   DATA_PATH=/x/y.h5 OUTDIR=/x/out bash run_train.sh
#
# 显存预算（A800/A100 80GB, B=1）:
#   1× 14B DiT (bf16)             ~28 GB
#   LoRA grads + torch Adam state 依 rank/targets 而变
#   Activations (GC on)           依 max_frames 而变
#   Buffers                       ~3-5 GB
#   ----------------------------  ---------
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 独立打包：wan/ 与本脚本同级 -> 工程根就是 HERE
# 在原仓库内：HERE = lingbot/lora_remotesense/new -> 工程根上推两层
if [ -d "$HERE/wan" ]; then
    PROJECT_ROOT="$HERE"
else
    PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
fi

# === 路径配置（按需修改） ===
STORAGE_ROOT="${RSWM_BASE:-$PROJECT_ROOT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${STORAGE_ROOT}/models/lingbot-world-base-cam}"
DATA_PATH="${DATA_PATH:-${STORAGE_ROOT}/data/remotesense/train.h5}"
OUTDIR="${OUTDIR:-${STORAGE_ROOT}/work/rs_lora_80g_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${STORAGE_ROOT}/logs}"

# === 训练超参（与原版一致，不影响实验对比） ===
NUM_EPOCHS="${NUM_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LR="${LR:-1e-4}"
MAX_FRAMES="${MAX_FRAMES:-33}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
HARDWARE_PROFILE="${HARDWARE_PROFILE:-a800_80g}"

# === 显存策略 ===
# INACTIVE_MODEL_CPU remains a backward-compatible alias when OFFLOAD is unset.
OFFLOAD="${OFFLOAD:-${INACTIVE_MODEL_CPU:-auto}}"
case "$OFFLOAD" in
    auto|0|1) ;;
    *) echo "ERROR: OFFLOAD=$OFFLOAD 非法（应为 auto、0 或 1）" >&2; exit 1 ;;
esac

# === Resume ===
RESUME_FROM="${RESUME_FROM:-}"

# === 显存分配优化 ===
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR" "$OUTDIR"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

echo "============================================="
echo "RSWM LoRA 训练 (A800/A100 80GB 适配版)"
echo "  CKPT      : $CHECKPOINT_DIR"
echo "  DATA      : $DATA_PATH"
echo "  OUTDIR    : $OUTDIR"
echo "  LOG       : $LOG_FILE"
echo "  Epochs    : $NUM_EPOCHS"
echo "  LoRA rank : $LORA_RANK / alpha=$LORA_ALPHA"
echo "  Max frames: $MAX_FRAMES   Accum: $ACCUM_STEPS"
echo "  Hardware profile: $HARDWARE_PROFILE"
echo "  Offload request: $OFFLOAD (auto on A800 resolves to 1)"
echo "============================================="

# === 前置检查 ===
[ -d "$CHECKPOINT_DIR" ] || { echo "ERROR: CHECKPOINT_DIR 不存在: $CHECKPOINT_DIR"; exit 1; }
[ -f "$DATA_PATH" ] || { echo "ERROR: DATA_PATH 不存在: $DATA_PATH"; echo "先跑 bash run_build_dataset.sh"; exit 1; }

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

EXTRA_ARGS=()
[ -n "$RESUME_FROM" ] && EXTRA_ARGS+=(--resume_from "$RESUME_FROM")
[ -n "${NUM_WORKERS:-}" ] && EXTRA_ARGS+=(--num_workers "$NUM_WORKERS")

# tee 让日志同时落盘
python "$HERE/train.py" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTDIR" \
    --num_epochs "$NUM_EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --learning_rate "$LR" \
    --max_frames "$MAX_FRAMES" \
    --gradient_accumulation_steps "$ACCUM_STEPS" \
    --hardware_profile "$HARDWARE_PROFILE" \
    --offload "$OFFLOAD" \
    --attention_backend "${ATTENTION_BACKEND:-auto}" \
    --no_8bit_adam \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

echo
echo "[Done] 输出: $OUTDIR"
echo "日志: $LOG_FILE"
