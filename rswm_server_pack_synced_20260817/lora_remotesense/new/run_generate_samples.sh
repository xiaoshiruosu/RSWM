#!/bin/bash
# =============================================================================
# 训练完批量生成视频样例（mp4）
#
# 默认从训练数据 HDF5 里随机抽 NUM_SAMPLES 个，依次生成。
# 同时落 GT 视频在 gt/ 子目录用来肉眼对比。
#
# 用法：
#   LORA_CKPT=/path/best NUM_SAMPLES=8 bash run_generate_samples.sh
#
#   # 按场景过滤
#   LORA_CKPT=/path/best SCENE=urban NUM_SAMPLES=4 bash run_generate_samples.sh
#
#   # 指定 sample key
#   LORA_CKPT=/path/best KEYS="sample_00012 sample_00037" bash run_generate_samples.sh
#
#   # 同一个样本换 noise seed 出多个变体
#   LORA_CKPT=/path/best KEYS="sample_00012" GEN_SEED=123 \
#       OUTDIR=samples_seed123 bash run_generate_samples.sh
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$HERE/wan" ]; then
    PROJECT_ROOT="$HERE"
else
    PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
fi

# === 路径 ===
STORAGE_ROOT="${RSWM_BASE:-$PROJECT_ROOT}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${STORAGE_ROOT}/models/lingbot-world-base-cam}"
LORA_CKPT="${LORA_CKPT:?ERROR: 请设置 LORA_CKPT=/path/to/best_checkpoint}"
DATA_PATH="${DATA_PATH:-${STORAGE_ROOT}/data/remotesense/train.h5}"
OUTDIR="${OUTDIR:-${STORAGE_ROOT}/work/samples_$(date +%Y%m%d_%H%M%S)}"

# === 选样 ===
NUM_SAMPLES="${NUM_SAMPLES:-8}"
SCENE="${SCENE:-}"              # general / flood / urban / rural / 留空=不过滤
KEYS="${KEYS:-}"                # 留空=随机；非空例: "sample_00012 sample_00037"
SEED="${SEED:-42}"

# === 推理参数 ===
FRAMES="${FRAMES:-33}"
SIZE="${SIZE:-480*832}"
STEPS="${STEPS:-70}"
SHIFT="${SHIFT:-10.0}"
GUIDE_SCALE="${GUIDE_SCALE:-5.0}"
GEN_SEED="${GEN_SEED:-42}"
DEVICE="${DEVICE:-cuda}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

mkdir -p "$OUTDIR"

echo "============================================="
echo "生成视频样例"
echo "  ckpt    : $CHECKPOINT_DIR"
echo "  lora    : $LORA_CKPT"
echo "  data    : $DATA_PATH"
echo "  output  : $OUTDIR"
echo "  N       : $NUM_SAMPLES   scene=${SCENE:-(any)}   keys=${KEYS:-(random)}"
echo "  frames  : $FRAMES   steps=$STEPS   shift=$SHIFT   guide=$GUIDE_SCALE"
echo "  seeds   : pick_seed=$SEED   gen_seed=$GEN_SEED"
echo "============================================="

[ -d "$CHECKPOINT_DIR" ] || { echo "ERROR: CHECKPOINT_DIR 不存在: $CHECKPOINT_DIR"; exit 1; }
[ -d "$LORA_CKPT" ]      || { echo "ERROR: LORA_CKPT 不存在: $LORA_CKPT"; exit 1; }
[ -f "$DATA_PATH" ]      || { echo "ERROR: DATA_PATH 不存在: $DATA_PATH"; exit 1; }

ARGS=(
    --checkpoint_dir "$CHECKPOINT_DIR"
    --lora_path "$LORA_CKPT"
    --data_path "$DATA_PATH"
    --output_dir "$OUTDIR"
    --num_samples "$NUM_SAMPLES"
    --seed "$SEED"
    --gen_seed "$GEN_SEED"
    --frame_num "$FRAMES"
    --size "$SIZE"
    --sampling_steps "$STEPS"
    --shift "$SHIFT"
    --guide_scale "$GUIDE_SCALE"
    --device "$DEVICE"
)

[ -n "$SCENE" ] && ARGS+=(--scene "$SCENE")
[ -n "$KEYS" ]  && ARGS+=(--keys $KEYS)   # 不加引号，让 shell 拆分

python "$HERE/generate_samples.py" "${ARGS[@]}"

echo
echo "[Done] 样例输出: $OUTDIR"
echo "  generated/<key>.mp4   <- LoRA 生成结果"
echo "  gt/<key>.mp4          <- ground truth 对比"
echo "  meta/<key>.json + index.json"
