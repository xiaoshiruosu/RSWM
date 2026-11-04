#!/bin/bash
# =============================================================================
# 从遥感原始数据重新构建 HDF5 数据集
#
# 用法：
#   bash run_build_dataset.sh                     # 100 样本（论文 main exp 配置）
#   RS_INSTANCES=20 bash run_build_dataset.sh     # 自定义样本数
#   DATA_PATH=/path/train.h5 bash run_build_dataset.sh   # 自定义输出路径
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$HERE/wan" ]; then
    PROJECT_ROOT="$HERE"
else
    PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
fi

# === 配置 ===
RS_INSTANCES="${RS_INSTANCES:-100}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/remotesense/train.h5}"
TARGET_H="${TARGET_H:-480}"
TARGET_W="${TARGET_W:-832}"

mkdir -p "$(dirname "$DATA_PATH")"

echo "============================================="
echo "Build dataset"
echo "  instances : $RS_INSTANCES"
echo "  output    : $DATA_PATH"
echo "  resolution: ${TARGET_H} x ${TARGET_W}"
echo "============================================="

if [ -f "$DATA_PATH" ]; then
    echo "[WARN] $DATA_PATH 已存在。"
    read -r -p "覆盖? [y/N] " ans
    [ "${ans:-N}" = "y" ] || { echo "取消。"; exit 0; }
fi

cd "$PROJECT_ROOT"

# =============================================================================
# 硬守卫：本脚本调用的 build_from_remotesense.py 不是论文管道
# (96×96/steps8/eurosat → 16/24/32 帧,拉伸糊,与论文不可比)。
# 论文数据请在包根执行:  bash server_run/build_correct.sh
# 仅调试可 FORCE_WRONG_BUILDER=1 bash run_build_dataset.sh(产物勿用于论文)。
# =============================================================================
if [ "${FORCE_WRONG_BUILDER:-0}" != "1" ]; then
    echo "=================================================================="
    echo "[REFUSED] run_build_dataset.sh 走的是错构建器(32帧糊图),已拦截。"
    echo "          正确建库: cd <包根 rswm_server_pack/> && bash server_run/build_correct.sh"
    echo "=================================================================="
    exit 2
fi

FORCE_WRONG_BUILDER=1 python "$HERE/build_from_remotesense.py" \
    --instances "$RS_INSTANCES" \
    --output "$DATA_PATH" \
    --target-height "$TARGET_H" \
    --target-width "$TARGET_W"

echo
echo "[Verify] 检查 HDF5 结构..."
python "$HERE/verify_remotesense_h5.py" "$DATA_PATH"

echo
echo "[Done] 数据集就绪: $DATA_PATH"
echo "下一步: bash $HERE/run_train.sh"
