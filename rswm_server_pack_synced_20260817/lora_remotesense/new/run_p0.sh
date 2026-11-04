#!/bin/bash
# =============================================================================
# run_p0.sh — RSWM P0 全链一键复现（头条 claim: FVD -60% / Urban SSIM +51%）
#
# 串起 5 步: build_dataset -> train -> baseline_eval -> lora_eval -> eval_aligned
#
#   step 1  build   构建 HDF5 数据集（train_500 + 100 val）
#   step 2  train   LoRA 训练（默认 r=128 头条配置, 5 epochs, 80G offload）
#   step 3  base    Base 模型评估（无 LoRA，同一 val split）
#   step 4  lora    LoRA 模型评估
#   step 5  aligned 对齐到公共样本子集，打印 LoRA vs Base 对比表
#
# ---------------------------------------------------------------------------
# 用法：
#   # 0) 先跑一遍 smoke（20 样本 / 1 epoch / 每次评估 4 样本，~1h，验证端到端不 OOM）
#   SMOKE=1 CHECKPOINT_DIR=/nix/rsroll_25828/models/lingbot-world-base-cam bash run_p0.sh
#
#   # 1) 正式全量（500 样本 / 5 epoch / 全 val，单卡 A100-80G ~1.5 天）
#   CHECKPOINT_DIR=/nix/rsroll_25828/models/lingbot-world-base-cam bash run_p0.sh
#
#   # 只跑其中几步（断点续跑）：
#   STEPS="base lora aligned" WORK=/nix/rsroll_25828/work/p0_20260703_xxxx \
#       CHECKPOINT_DIR=... bash run_p0.sh
#
# 关键环境变量（含默认）：
#   CHECKPOINT_DIR  (必填) 基模权重目录
#   SMOKE           0      1=烟囱测试小配置
#   STEPS           全部    空格分隔的步骤子集: build train base lora aligned
#   WORK            自动    本次运行的根输出目录（续跑时手动指定同一个）
#   LORA_RANK/ALPHA 128/64  头条=128；跑 rank 消融时改这里
#   NUM_EPOCHS      5       论文配置（注意 run_train.sh 自身默认是 10！）
#   RS_INSTANCES    500     数据集样本数
#   MAX_SAMPLES     0       评估样本上限，0=全量
#   FRAME_NUM       33      评估生成帧数（与训练正式协议一致）
#   OFFLOAD         auto   auto/0/1；A800 auto→1，显式 0 仅用于吞吐/OOM 边界测试
#   DEVICE          cuda（物理卡由 CUDA_VISIBLE_DEVICES 选择）
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$HERE/wan" ]; then
    PROJECT_ROOT="$HERE"
else
    PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

# ---------- 配置 ----------
: "${CHECKPOINT_DIR:?必填: 基模权重目录, 例 CHECKPOINT_DIR=/nix/rsroll_25828/models/lingbot-world-base-cam}"
SMOKE="${SMOKE:-0}"
STEPS="${STEPS:-build train base lora aligned}"
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-64}"
OFFLOAD="${OFFLOAD:-${INACTIVE_MODEL_CPU:-auto}}"
case "$OFFLOAD" in auto|0|1) ;; *) echo "FATAL: OFFLOAD=$OFFLOAD 非法" >&2; exit 1 ;; esac
FRAME_NUM="${FRAME_NUM:-33}"
DEVICE="${DEVICE:-cuda}"

if [ "$SMOKE" = "1" ]; then
    RS_INSTANCES="${RS_INSTANCES:-20}"
    NUM_EPOCHS="${NUM_EPOCHS:-1}"
    MAX_SAMPLES="${MAX_SAMPLES:-4}"
    TAG="smoke"
else
    RS_INSTANCES="${RS_INSTANCES:-500}"
    NUM_EPOCHS="${NUM_EPOCHS:-5}"
    MAX_SAMPLES="${MAX_SAMPLES:-0}"
    TAG="full"
fi

# WORK 目录：续跑时务必手动传同一个 WORK
WORK="${WORK:-${PROJECT_ROOT}/outputs/p0_${TAG}_$(date +%Y%m%d_%H%M%S)}"
DATA_PATH="${DATA_PATH:-${WORK}/data/train.h5}"
TRAIN_OUT="${WORK}/train"
BASE_OUT="${WORK}/eval_base"
LORA_OUT="${WORK}/eval_lora"
ALIGNED_OUT="${WORK}/eval_aligned"
SPLIT_FILE="${TRAIN_OUT}/train_val_split.json"

mkdir -p "$WORK"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

has_step() { [[ " $STEPS " == *" $1 "* ]]; }

echo "==================================================================="
echo " RSWM P0 复现  [$TAG]"
echo "   WORK        : $WORK"
echo "   STEPS       : $STEPS"
echo "   CKPT(base)  : $CHECKPOINT_DIR"
echo "   LoRA        : r=$LORA_RANK alpha=$LORA_ALPHA  epochs=$NUM_EPOCHS"
echo "   dataset     : $RS_INSTANCES 样本 -> $DATA_PATH"
echo "   eval        : frame_num=$FRAME_NUM  max_samples=$MAX_SAMPLES  offload=$OFFLOAD"
echo "==================================================================="
[ -d "$CHECKPOINT_DIR" ] || { echo "ERROR: CHECKPOINT_DIR 不存在: $CHECKPOINT_DIR"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Evaluation keeps the safe path for auto; only an explicit 0 disables it.
INACTIVE_FLAG=(); [ "$OFFLOAD" = "0" ] && INACTIVE_FLAG=(--no_inactive_model_cpu)

# ---------- step 1: build ----------
if has_step build; then
    echo; echo ">>> [1/5] build dataset ($RS_INSTANCES 样本)"
    if [ -f "$DATA_PATH" ]; then
        echo "    $DATA_PATH 已存在，跳过构建（如需重建先删除该文件）。"
    else
        mkdir -p "$(dirname "$DATA_PATH")"
        python "$HERE/build_from_remotesense.py" \
            --instances "$RS_INSTANCES" \
            --output "$DATA_PATH" \
            --target-height 480 --target-width 832
        python "$HERE/verify_remotesense_h5.py" "$DATA_PATH"
    fi
fi

# ---------- step 2: train ----------
if has_step train; then
    echo; echo ">>> [2/5] LoRA 训练 (r=$LORA_RANK, $NUM_EPOCHS epochs)"
    [ -f "$DATA_PATH" ] || { echo "ERROR: 缺数据 $DATA_PATH，先跑 build"; exit 1; }
    CHECKPOINT_DIR="$CHECKPOINT_DIR" DATA_PATH="$DATA_PATH" OUTDIR="$TRAIN_OUT" \
    NUM_EPOCHS="$NUM_EPOCHS" LORA_RANK="$LORA_RANK" LORA_ALPHA="$LORA_ALPHA" \
    MAX_FRAMES="${MAX_FRAMES:-33}" ACCUM_STEPS="${ACCUM_STEPS:-4}" \
    OFFLOAD="$OFFLOAD" \
        bash "$HERE/run_train.sh"
fi

# best_checkpoint 与 split 定位
LORA_CKPT="${LORA_CKPT:-${TRAIN_OUT}/best_checkpoint}"

# ---------- step 3: baseline eval ----------
if has_step base; then
    echo; echo ">>> [3/5] Baseline 评估 (无 LoRA)"
    [ -f "$SPLIT_FILE" ] || { echo "ERROR: 缺 $SPLIT_FILE，需先完成 train"; exit 1; }
    BASE_ARGS=(--checkpoint_dir "$CHECKPOINT_DIR" --data_path "$DATA_PATH"
               --split_file "$SPLIT_FILE" --output_dir "$BASE_OUT"
               --frame_num "$FRAME_NUM" --device "$DEVICE" "${INACTIVE_FLAG[@]}")
    [ "$MAX_SAMPLES" != "0" ] && BASE_ARGS+=(--max_samples "$MAX_SAMPLES")
    python "$HERE/evaluate_baseline.py" "${BASE_ARGS[@]}" 2>&1 | tee "$WORK/log_base.txt"
fi

# ---------- step 4: lora eval ----------
if has_step lora; then
    echo; echo ">>> [4/5] LoRA 评估"
    [ -d "$LORA_CKPT" ] || { echo "ERROR: 缺 LoRA checkpoint $LORA_CKPT"; exit 1; }
    [ -f "$SPLIT_FILE" ] || { echo "ERROR: 缺 $SPLIT_FILE"; exit 1; }
    LORA_ARGS=(--checkpoint_dir "$CHECKPOINT_DIR" --lora_path "$LORA_CKPT"
               --data_path "$DATA_PATH" --split_file "$SPLIT_FILE"
               --output_dir "$LORA_OUT" --frame_num "$FRAME_NUM"
               --device "$DEVICE" "${INACTIVE_FLAG[@]}")
    [ "$MAX_SAMPLES" != "0" ] && LORA_ARGS+=(--max_samples "$MAX_SAMPLES")
    python "$HERE/evaluate.py" "${LORA_ARGS[@]}" 2>&1 | tee "$WORK/log_lora.txt"
fi

# ---------- step 5: aligned 对比 ----------
if has_step aligned; then
    echo; echo ">>> [5/5] eval_aligned 对齐子集对比"
    python "$HERE/eval_aligned.py" \
        --lora_dir "$LORA_OUT" --base_dir "$BASE_OUT" \
        --output_dir "$ALIGNED_OUT" --device "$DEVICE" 2>&1 | tee "$WORK/log_aligned.txt"
fi

echo
echo "==================================================================="
echo " P0 完成。产物："
echo "   数据集   : $DATA_PATH"
echo "   LoRA ckpt: $LORA_CKPT"
echo "   Base 评估: $BASE_OUT"
echo "   LoRA 评估: $LORA_OUT"
echo "   对齐对比 : $ALIGNED_OUT/comparison.json  <- 头条数字在这"
echo "==================================================================="
echo "后续（不在本包内，需从父仓库拿 rebin_overlap_metrics.py）："
echo "   - Overlap V2 re-bin: 复用 $LORA_OUT 的 npy 重新分箱算 FVD"
