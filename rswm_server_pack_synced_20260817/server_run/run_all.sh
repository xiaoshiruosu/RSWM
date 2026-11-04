#!/bin/bash
# =============================================================================
# run_all.sh — 单卡 A800 全流程串行执行(P0→P5),断点续跑
#
# 用法(tmux 里跑):
#   export CHECKPOINT_DIR=/nix/rsroll_25828/models/lingbot-world-base-cam
#   bash server_run/run_all.sh              # 全流程(先 smoke,通过即继续)
#   SKIP_SMOKE=1 bash server_run/run_all.sh # 跳过 smoke
#
# 断点续跑:所有阶段以 $WORK/stamps/*.done 打点,重跑本脚本自动跳过已完成段。
# 想强制重跑某段:删对应 stamps/xxx.done 即可。
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "############ RSWM 从头复跑 — 全流程 ############"
date

if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    bash "$HERE/p0_smoke.sh"
    echo
    echo ">>> SMOKE 已通过。10 秒后进入正式流程(Ctrl-C 可中止先人工核对 smoke 产物)..."
    sleep 10
fi

bash "$HERE/p1_main.sh"
bash "$HERE/p2_ranks.sh"
bash "$HERE/p3_direction.sh"
bash "$HERE/p4_overlap.sh"
bash "$HERE/p5_quals.sh"

python "$HERE/collect_results.py" "${WORK:-$HERE/work}"

echo
echo "############ 全部完成 ✓ ############"
echo "结果总表: ${WORK:-$HERE/work}/summary.md"
date
