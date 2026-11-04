#!/bin/bash
# =============================================================================
# qsub_resisc45.sh — 第二数据集(RESISC45)泛化实验 编排(advisor point 5)
#
# 设计: **TEST-ONLY 泛化** —— 不在 RESISC45 上训练。建 RESISC45 benchmark,
#       然后拿**已训好的 AID checkpoint**去评它。训练集固定 AID、测试集换成
#       RESISC45,掉点 = 泛化间隙。这是最便宜、也最站得住的做法(不需要新训练)。
#
# 安全设计(照抄 qsub_corrected_rerun.sh):
#   - 独立 WORK_RESISC45,拒绝等于旧 AID work
#   - 容量/inode 预检(qk73 容量紧、rx40 inode 紧)
#   - -v WORK=... 传给每个作业;依赖关系显式打印
#   - DRYRUN=1 只打印不提交;任何失败非零退出
#   ⚠ 旧 AID work/(H5、checkpoint、eval 输出、summary.md)一律只读、不触碰。
#
# 依赖链(默认拆两个作业,build 与 eval 串 afterok):
#   40(PHASE=build) 建 RESISC45 GEO/SEM H5 + 校验 + split
#     → 40(PHASE=eval) 用 --lora-ckpt 评估(afterok:build)
#
# 用法(包根):
#   DRYRUN=1 bash server_run/pbs/qsub_resisc45.sh --lora-ckpt <路径>   # 只打印
#   bash server_run/pbs/qsub_resisc45.sh --lora-ckpt <路径>            # 真提交
#   bash server_run/pbs/qsub_resisc45.sh --lora-ckpt <路径> --channel geo
#   WORK_RESISC45=/scratch/rx40/$USER/rswm_resisc45 bash ... --lora-ckpt <路径>
#   FUSED=1 bash ... --lora-ckpt <路径>        # build+eval 合成一个作业
#
# --lora-ckpt 举例(旧 AID 主结果的 r16 语义通道):
#   server_run/work_corrected/train_r16_sem_s42/best_checkpoint
#   ⚠ 只读取该 checkpoint,不修改它。
#
# 前置: 先在登录节点跑 `python server_run/check_resisc45.py --check-offline
#       --check-source` 确认 RESISC45 离线缓存就绪(不做的话建库会静默产出合成数据)。
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS="$HERE"
ROOT="$(cd "$PBS/../.." && pwd)"
DRYRUN="${DRYRUN:-0}"
FUSED="${FUSED:-0}"

# ---- 参数 ----
LORA_CKPT=""
CHANNEL="sem"
TIER="${TIER:-resisc45}"   # 泛化数据集: resisc45 | patternnet | whurs19(--tier 覆盖)
N_RESISC45="${N_RESISC45:-200}"
EVAL_N_RESISC45="${EVAL_N_RESISC45:-50}"

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}"
    echo
    echo "必需: --lora-ckpt PATH   已训好的 AID LoRA best_checkpoint 目录"
    echo "可选: --tier resisc45|patternnet|whurs19 (默认 resisc45,泛化数据集选择)"
    echo "      --channel sem|geo  (默认 sem) | --n N (默认 $N_RESISC45) | --eval-n N (默认 $EVAL_N_RESISC45)"
}
while [ $# -gt 0 ]; do
    case "$1" in
        --lora-ckpt) LORA_CKPT="${2:?--lora-ckpt 需要参数}"; shift 2 ;;
        --tier)      TIER="${2:?--tier 需要参数}"; shift 2 ;;
        --channel)   CHANNEL="${2:?--channel 需要参数}"; shift 2 ;;
        --n)         N_RESISC45="${2:?--n 需要参数}"; shift 2 ;;
        --eval-n)    EVAL_N_RESISC45="${2:?--eval-n 需要参数}"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "FATAL: 未知参数 '$1'" >&2; usage >&2; exit 1 ;;
    esac
done

# ---- 必需参数校验 ----
case "$TIER" in resisc45|patternnet|whurs19) ;; *)
    echo "FATAL: --tier 必须是 resisc45|patternnet|whurs19,收到 '$TIER'" >&2; exit 1 ;;
esac

if [ -z "$LORA_CKPT" ]; then
    echo "FATAL: --lora-ckpt 是必需参数(要评哪个 AID 训好的 checkpoint)。" >&2
    usage >&2; exit 1
fi
if [ ! -d "$LORA_CKPT" ]; then
    echo "FATAL: --lora-ckpt '$LORA_CKPT' 不存在或不是目录。" >&2
    echo "       典型路径: <WORK_CORRECTED>/train_r16_sem_s42/best_checkpoint" >&2
    exit 1
fi
case "$CHANNEL" in
    sem|geo) ;;
    *) echo "FATAL: --channel 只能是 sem|geo(给的是 '$CHANNEL')" >&2; exit 1 ;;
esac
_abs() { python3 -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$1" 2>/dev/null || echo "$1"; }
LORA_CKPT="$(_abs "$LORA_CKPT")"

# ---- 独立 WORK:绝不复用旧 AID work ----
OLD_WORK="$ROOT/server_run/work"
# WORK 按 tier 派生独立目录(WORK_RESISC45 env 仍可显式覆盖,向后兼容)
WORK_RESISC45="${WORK_RESISC45:-$ROOT/server_run/work_${TIER}}"
if [ "$(_abs "$WORK_RESISC45")" = "$(_abs "$OLD_WORK")" ]; then
    echo "FATAL: WORK_RESISC45 指向旧 AID work ($OLD_WORK) —— 拒绝执行(旧结果不可覆盖)。" >&2
    echo "       请设 WORK_RESISC45 为一个独立目录再跑。" >&2
    exit 1
fi
# checkpoint 若正好落在新 WORK 里,建库会把它连坐删/覆盖的风险 → 也挡掉
case "$LORA_CKPT/" in
    "$(_abs "$WORK_RESISC45")"/*)
        echo "FATAL: --lora-ckpt 位于 WORK_RESISC45 ($WORK_RESISC45) 内部 —— 拒绝执行。" >&2
        echo "       待评 checkpoint 必须在本实验 WORK 之外(只读引用)。" >&2
        exit 1 ;;
esac
export WORK="$WORK_RESISC45"

echo "=================================================================="
echo "== 第二数据集泛化(TEST-ONLY,不训练 RESISC45)"
echo "== 独立 WORK        : $WORK"
echo "== 旧 AID work(只读): $OLD_WORK"
echo "== 待评 checkpoint  : $LORA_CKPT"
echo "== 通道 / 样本数     : CHANNEL=$CHANNEL  建库 N=$N_RESISC45  评估 n=$EVAL_N_RESISC45"
echo "=================================================================="

# ---- 依赖文件检查 ----
for f in "$PBS/40_build_resisc45.pbs" "$PBS/_pbs_env.sh" \
         "$ROOT/server_run/check_resisc45.py" "$ROOT/server_run/common.sh" \
         "$ROOT/server_run/neutralize_poses.py" \
         "$ROOT/lora_remotesense/build_dataset_from_benchmark.py"; do
    [ -f "$f" ] || { echo "FATAL: 缺依赖文件 $f" >&2; exit 1; }
done
echo "== 依赖文件齐全 ✓"

# ---- 容量 / inode 预检 [v2.4.4:共享 lquota 库,df 判决已废(P0-1)] ----
# 泛化 tier 足迹较小:GEO+SEM H5(N=200)+中间产物+1 份 eval ≈ 阈值 60GiB/20k。
source "$PBS/_quota_preflight.sh"
if [ "$DRYRUN" != "1" ]; then
    quota_preflight "$WORK" 60 20000 || exit 1
else
    echo "== [DRYRUN] 跳过配额预检"
fi

Q() { if [ "$DRYRUN" = "1" ]; then echo "qsub $*" >&2; echo "jobid_$RANDOM"; else qsub "$@"; fi; }

# 所有作业继承:独立 WORK + tier + 规模
VBASE="WORK=$WORK,DATASET_TIER=$TIER,N_RESISC45=$N_RESISC45,EVAL_N_RESISC45=$EVAL_N_RESISC45"

echo
if [ "$FUSED" = "1" ]; then
    echo "===== 提交 40(PHASE=all:建库+评估 单作业)====="
    j=$(Q -v "$VBASE,PHASE=all,CHANNEL=$CHANNEL,LORA_CKPT=$LORA_CKPT" \
           "$PBS/40_build_resisc45.pbs")
    echo "  40_resisc45_all = $j"
    echo
    echo "== 作业依赖: 无(单个融合作业;建库阶段 GPU 闲置约 1-2h)"
    JOBS="$j"
else
    echo "===== 提交 40(PHASE=build:建 RESISC45 GEO/SEM H5 + 校验 + split)====="
    j_build=$(Q -v "$VBASE,PHASE=build" "$PBS/40_build_resisc45.pbs")
    echo "  40_resisc45_build = $j_build"

    echo "===== 提交 40(PHASE=eval:AID checkpoint 评 RESISC45,afterok:build)====="
    j_eval=$(Q -v "$VBASE,PHASE=eval,CHANNEL=$CHANNEL,LORA_CKPT=$LORA_CKPT" \
                -W depend=afterok:"$j_build" "$PBS/40_build_resisc45.pbs")
    echo "  40_resisc45_eval  = $j_eval"
    echo
    echo "== 作业依赖: 40_resisc45_eval($j_eval)  afterok  40_resisc45_build($j_build)"
    JOBS="$j_build $j_eval"
fi

cat <<EOF

==================================================================
提交完毕(独立 WORK=$WORK;旧 AID work 未被触碰)。
作业: $JOBS

产物:
  $WORK/data/train_geo.h5            RESISC45 几何通道(waypoint Plücker)
  $WORK/data/train_semantic.h5       RESISC45 语义通道(identity 位姿)
  $WORK/data/resisc45_val_split.json test-only split(全量 val,场景分层交错)
  $WORK/resisc45_source_check.json   建库前源数据集校验报告
  $WORK/resisc45_h5_check.json       建库后 H5 完整性报告
  $WORK/eval_resisc45_${CHANNEL}_*/eval_results.json   泛化评估结果

跑完后的对比(AID → RESISC45 泛化间隙):
  # 同一个 checkpoint 在 AID 上的数字(旧结果,只读):
  #   <WORK_CORRECTED>/eval_r16_${CHANNEL}_s42/eval_results.json
  # 在 RESISC45 上的数字(本次):
  #   $WORK/eval_resisc45_${CHANNEL}_*/eval_results.json
  两者的 SSIM/PSNR/LPIPS/FID/FVD/Angular_Error_Mean 逐项对比即泛化间隙。
  ⚠ 报告时必须披露:RESISC45(256×256)是 4 张互不相关的源图拼成 512 场景
    (build_dataset_from_benchmark.py:310-340),AID(600×600)是单图裁剪 ——
    掉点混杂了域偏移 + 分辨率/GSD 偏移 + 拼接接缝,不是纯粹的「换数据集」。
    详见 server_run/README_RESISC45.md。

另一个通道(可选,评完 sem 再评 geo):
  bash server_run/pbs/qsub_resisc45.sh --lora-ckpt <geo 通道 ckpt> --channel geo
  (build 已 stamp,不会重建;直接复用同一份 RESISC45 H5)
==================================================================
EOF
