#!/bin/bash
# =============================================================================
# P3 — 方向路由对比(~21-26h):几何通道(waypoint Plücker 位姿)训练 + 全 val 评估
#
# 论文的 decisive analysis:
#   语义通道 = P1(prompt 方向 + identity 位姿)        -> E_MAIN 的 Angular_Error
#   几何通道 = 本阶段(同 prompt + waypoint Plücker 位姿) -> E_GEO 的 Angular_Error
# 数据同源(GEO_H5 与 SEM_H5 样本/prompt 完全一致,位姿是唯一变量),
# 训练器同为 new/train.py(framewise=True) —— 干净的单变量对比,
# 避开了旧 plucker_cumulative 的 wrong-gradient + unfreeze 混淆。
# =============================================================================
# [v2.9.4] 产物角色 —— 进 run_manifest,并参与 headline 资格判定
export RSWM_PROFILE=legacy
export RSWM_RUN_ROLE=legacy
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -f "$GEO_H5" ] || { echo "FATAL: 缺 $GEO_H5,先跑 p1_main.sh"; exit 1; }

if ! done_p p3_train; then
    train_lora r16_geo 16 16 "$GEO_H5" "$T_GEO"   # [v2.7.2 ②] r128→r16 对齐主线
    stamp p3_train
fi

if ! done_p p3_eval; then
    # 全 val 评估:方向统计需要 n=100 的角度分布
    eval_lora r16_geo "$T_GEO/best_checkpoint" "$GEO_H5" \
              "$T_GEO/train_val_split.json" "$E_GEO" 0
    stamp p3_eval
fi

if ! done_p p3_aligned; then
    aligned geo_vs_base "$E_GEO" "$E_BASE" "$WORK/aligned_geo"
    stamp p3_aligned
fi

echo
echo "=========================================================="
echo " P3 完成 ✓  方向路由对比:"
echo "   语义通道 Angular_Error -> $E_MAIN/eval_results.json"
echo "   几何通道 Angular_Error -> $E_GEO/eval_results.json"
echo "   (随机基准 = 90°;两者都 ~90° 即为『结构性局限』结论)"
echo " 下一步: bash server_run/p4_overlap.sh"
echo "=========================================================="
