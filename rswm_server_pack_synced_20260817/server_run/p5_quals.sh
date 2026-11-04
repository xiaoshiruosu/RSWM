#!/bin/bash
# =============================================================================
# P5 — 定性素材(~2h):批量生成 LoRA mp4(带 GT 对照)
# baseline 的定性帧不用另跑 —— P1 的 $E_BASE/generated/*.npy 就是。
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

[ -d "$T_MAIN/best_checkpoint" ] || { echo "FATAL: 先跑 p1_main.sh"; exit 1; }

if ! done_p p5_quals; then
    # 每场景 4 个样本,FRAMES=33（与训练正式协议一致）
    for SC in urban rural general water; do
        LORA_CKPT="$T_MAIN/best_checkpoint" \
        DATA_PATH="$SEM_H5" \
        CHECKPOINT_DIR="$CHECKPOINT_DIR" \
        SCENE="$SC" NUM_SAMPLES=4 FRAMES="$MAX_FRAMES" \
        OUTDIR="$WORK/quals/$SC" \
            bash "$NEW/run_generate_samples.sh" 2>&1 | tee "$LOGS/p5_$SC.log"
    done
    stamp p5_quals
fi

echo
echo "=========================================================="
echo " P5 完成 ✓"
echo "   LoRA mp4 + GT 对照 -> $WORK/quals/<scene>/"
echo "   baseline 定性帧    -> $E_BASE/generated/*.npy (P1 已产出)"
echo " 最后: python server_run/collect_results.py $WORK"
echo "=========================================================="
