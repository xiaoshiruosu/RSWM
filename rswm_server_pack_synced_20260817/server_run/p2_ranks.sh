#!/bin/bash
# =============================================================================
# P2 — Rank 消融(~50-60h):r∈{16,32,64} 训练 + 各评 EVAL_N_ABL(默认50)
# baseline 不再评 —— 复用 P1 的 E_BASE(同 seed42 同数据 → val split 相同)
# alpha=rank(与 ablation_rank{16,32}.pbs 一致;r64 与主实验 64/64 一致)
# =============================================================================
# [v2.9] 本脚本属【已退役的 legacy 链】(SEM_H5 = identity 中和位姿)。
# 声明 legacy profile 以保持 SEM_ARM_MODE=identity,复现旧数字;
# 正式 ICLR 矩阵走 official profile(SEM_ARM_MODE=drop),见 common.sh。
export RSWM_PROFILE=legacy
# [v2.9.4] 产物角色 —— 进 run_manifest,并参与 headline 资格判定
export RSWM_RUN_ROLE=legacy
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -f "$T_MAIN/train_val_split.json" ] || { echo "FATAL: 先跑 p1_main.sh"; exit 1; }
[ -d "$E_BASE" ] || { echo "FATAL: 缺 baseline 评估 $E_BASE(P1.3)"; exit 1; }

for R in 32 64 128 ; do
    TDIR="$WORK/train_r${R}_sem"
    EDIR="$WORK/eval_r${R}_sem"

    if ! done_p "p2_train_r${R}"; then
        train_lora "r${R}_sem" "$R" "$R" "$SEM_H5" "$TDIR"
        stamp "p2_train_r${R}"
    fi

    if ! done_p "p2_eval_r${R}"; then
        # 各 rank 用自己的 split 文件(内容与 P1 相同:同 seed/同数据)
        eval_lora "r${R}_sem" "$TDIR/best_checkpoint" "$SEM_H5" \
                  "$TDIR/train_val_split.json" "$EDIR" "$EVAL_N_ABL"
        stamp "p2_eval_r${R}"
    fi

    if ! done_p "p2_aligned_r${R}"; then
        aligned "r${R}" "$EDIR" "$E_BASE" "$WORK/aligned_r${R}"
        stamp "p2_aligned_r${R}"
    fi
done

echo
echo "=========================================================="
echo " P2 完成 ✓  rank 表数字:"
echo "   r16(主线) -> $E_MAIN/eval_results.json (P1)"
for R in 32 64 128; do echo "   r${R}  -> $WORK/eval_r${R}_sem/eval_results.json"; done
echo " 下一步: bash server_run/p3_direction.sh"
echo "=========================================================="
