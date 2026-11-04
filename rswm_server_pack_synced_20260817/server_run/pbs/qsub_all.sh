#!/bin/bash
# =============================================================================
# qsub_all.sh — [LEGACY] 01-19 旧实验链(PBS afterok 串联)
#
# 当前冻结流程只用 qsub_corrected_rerun.sh(20-33);本脚本默认拒绝执行。
#
# 仅调试/复现旧结果时(从包根,或任意目录):
#   export CHECKPOINT_DIR=$HOME/RSWM/lingbot-world-base-cam
#   export CONDA_ENV=rswm            # 或绝对路径 /home/user/RSWM/envs/rswm
#   DRYRUN=1 FORCE_LEGACY=1 bash server_run/pbs/qsub_all.sh
#   FORCE_LEGACY=1 bash server_run/pbs/qsub_all.sh
#   WITH_P6=1 FORCE_LEGACY=1 bash server_run/pbs/qsub_all.sh
#
# 依赖图:
#   01_build ─┬─ 02_train_r128_sem ─┬─ 03_eval_base ─┐
#             │                      ├─ 04_eval_r128  │
#             │                      ├─ 13/14/15/16_overlap ─ 17_rebin
#             │                      └─ 18_quals                     │
#             ├─ 05/06/07_train_r{16,32,64} ─ 08/09/10_eval_r{...}   │
#             └─ 11_train_geo ─ 12_eval_geo                          │
#   19_collect  depend afterok: 03 04 08 09 10 12 17  (跑对齐+总表)
# =============================================================================
# =============================================================================
# [v2.7.4 废弃] 本编排器属 legacy 链(02-19),已被 qsub_corrected_rerun.sh 的
# 20-33 corrected 链取代(D3 拍板:主线=r16 三种子;拍板③:overlap 13-17 删除)。
# 三套目录规范混用会让汇总器出"正文主表缺失、aligned_main 却展示 r128"的
# 不一致(外部审查四轮确认)。冻结范围只含 corrected 链。
# 确要跑(仅调试/复现旧结果):FORCE_LEGACY=1 bash server_run/pbs/qsub_all.sh
# =============================================================================
if [ "${FORCE_LEGACY:-0}" != "1" ]; then
    echo "[REFUSED] qsub_all.sh 是 legacy 链,已废弃 —— 请用 qsub_corrected_rerun.sh(20-33)。"
    echo "          调试强制: FORCE_LEGACY=1 bash server_run/pbs/qsub_all.sh"
    exit 2
fi

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK_ROOT="$(cd "$HERE/../.." && pwd)"
DRYRUN="${DRYRUN:-0}"
command -v qsub >/dev/null || { echo "FATAL: 没有 qsub(要在 PBS 登录节点上跑)"; [ "$DRYRUN" = 1 ] || exit 1; }

# _pbs_env.sh 里 Gadi 路径已是默认值;这里只把「用户显式覆盖的」透传下去(未设就用默认)。
# CODE_ROOT 显式传,避免依赖 PBS_O_WORKDIR(更稳)。CHECKPOINT_DIR 非 Gadi 默认时才需传。
VLIST="CODE_ROOT=$PACK_ROOT"
for v in CHECKPOINT_DIR ASSET_ROOT CONDA_ENV CONDA_SH OFFLOAD MAX_FRAMES; do
    [ -n "${!v:-}" ] && VLIST="$VLIST,$v=${!v}"
done

sub() {  # sub FILE [dep_jobids...] -> 打印并返回 jobid
    local f="$1"; shift
    local depargs=()
    [ "$#" -gt 0 ] && depargs=(-W "depend=afterok$(printf ':%s' "$@")")
    if [ "$DRYRUN" = 1 ]; then
        echo "qsub -v $VLIST ${depargs[*]:-} $HERE/$f" >&2
        echo "DRY.$f"
    else
        qsub -v "$VLIST" "${depargs[@]}" "$HERE/$f"
    fi
}

echo "提交 RSWM PBS 作业链  CODE_ROOT=$PACK_ROOT  DRYRUN=$DRYRUN"
echo "  透传 -v: $VLIST"
echo "  (Gadi 默认: 项目 rx40 / 队列 gpuhopper / conda+路径见 _pbs_env.sh;换机器改那里或 export 覆盖)"
echo

j01=$(sub 01_build_data.pbs);                   echo "01_build           -> $j01"
j02=$(sub 02_train_r128_sem.pbs "$j01");         echo "02_train_r128_sem  -> $j02  (after $j01)"
j03=$(sub 03_eval_base.pbs "$j02");              echo "03_eval_base       -> $j03  (after $j02)"
j04=$(sub 04_eval_r128_sem.pbs "$j02");          echo "04_eval_r128_sem   -> $j04  (after $j02)"
j05=$(sub 05_train_r16.pbs "$j01");              echo "05_train_r16       -> $j05  (after $j01)"
j06=$(sub 06_train_r32.pbs "$j01");              echo "06_train_r32       -> $j06  (after $j01)"
j07=$(sub 07_train_r64.pbs "$j01");              echo "07_train_r64       -> $j07  (after $j01)"
j08=$(sub 08_eval_r16.pbs "$j05");               echo "08_eval_r16        -> $j08  (after $j05)"
j09=$(sub 09_eval_r32.pbs "$j06");               echo "09_eval_r32        -> $j09  (after $j06)"
j10=$(sub 10_eval_r64.pbs "$j07");               echo "10_eval_r64        -> $j10  (after $j07)"
j11=$(sub 11_train_geo.pbs "$j01");              echo "11_train_geo       -> $j11  (after $j01)"
j12=$(sub 12_eval_geo.pbs "$j11");               echo "12_eval_geo        -> $j12  (after $j11)"
j13=$(sub 13_overlap_trivial.pbs "$j02");        echo "13_overlap_trivial -> $j13  (after $j02)"
j14=$(sub 14_overlap_easy.pbs "$j02");           echo "14_overlap_easy    -> $j14  (after $j02)"
j15=$(sub 15_overlap_medium.pbs "$j02");         echo "15_overlap_medium  -> $j15  (after $j02)"
j16=$(sub 16_overlap_hard.pbs "$j02");           echo "16_overlap_hard    -> $j16  (after $j02)"
j17=$(sub 17_overlap_rebin.pbs "$j13" "$j14" "$j15" "$j16"); echo "17_overlap_rebin   -> $j17  (after 13-16)"
j18=$(sub 18_quals.pbs "$j02");                  echo "18_quals           -> $j18  (after $j02)"
j19=$(sub 19_collect.pbs "$j03" "$j04" "$j08" "$j09" "$j10" "$j12" "$j17"); echo "19_collect         -> $j19  (after 03,04,08,09,10,12,17)"

if [ "${WITH_P6:-0}" = 1 ]; then
    echo "[warn] P6(MVSplat360)需先在登录节点建独立 torch2.2.2 环境 + gated SVD,断网集群慎用。"
    echo "       没有现成 p6.pbs;参照 server_run/p6_mvsplat.sh 自行封装,depend afterok:$j02。"
fi

echo
echo "全部提交 ✓  查看: qstat -u \$USER   日志: 各 rswm_NN_*.o<jobid>"
echo "某作业超时失败 → 依赖它的会被 PBS 取消(Eqw/held);修/等资源后重投该作业及其下游即可"
echo "(stamp 断点 + 训练 --resume_from 自动续,已完成的不重跑)。"
