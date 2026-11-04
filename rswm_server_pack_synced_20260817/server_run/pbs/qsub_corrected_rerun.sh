#!/bin/bash
# =============================================================================
# qsub_corrected_rerun.sh — 档2 修正-prompt 全量重跑 编排
#
# ⚠️ 安全改版(2026-07-11):旧 AID 结果不可覆盖/删除。
#    本脚本**不再删除任何旧 H5/stamp**,而是强制写入一个【独立 WORK 根目录】:
#        WORK_CORRECTED (默认 $ROOT/server_run/work_corrected)
#    旧 AID 的 work/ (H5、checkpoint、eval 输出、summary.md) 一律只读、不触碰。
#
# 依赖链:
#   01 建库(新 builder = 修正 prompt + waypoints/segment_angles) → 独立 WORK
#     → 20-30 + 32/33 训练+评估(13 配置,全并行,afterok:01)
#     → 31 baseline 评估(afterok:20,取其 split)
#   之后在【登录节点】跑纯 CPU 分析(见末尾打印),不占 GPU。
#
# 用法(包根):
#   DRYRUN=1 bash server_run/pbs/qsub_corrected_rerun.sh          # 只打印,不提交
#   bash server_run/pbs/qsub_corrected_rerun.sh                   # 真提交
#   WORK_CORRECTED=/scratch/rx40/$USER/rswm_corrected bash ...    # 自定独立 WORK
#
# ⚠️ 前置:先在登录节点跑 第0步 校准 + 两个模块自测,全绿再提交(见末尾)。
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS="$HERE"
ROOT="$(cd "$PBS/../.." && pwd)"
DRYRUN="${DRYRUN:-0}"

# ---- 独立 WORK:绝不复用旧 AID work ----
OLD_WORK="$ROOT/server_run/work"
WORK_CORRECTED="${WORK_CORRECTED:-$ROOT/server_run/work_corrected}"

_abs() { python3 -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$1" 2>/dev/null || echo "$1"; }
if [ "$(_abs "$WORK_CORRECTED")" = "$(_abs "$OLD_WORK")" ]; then
  echo "FATAL: WORK_CORRECTED 指向旧 AID work ($OLD_WORK) —— 拒绝执行(旧结果不可覆盖)。" >&2
  echo "       请设 WORK_CORRECTED 为一个独立目录再跑。" >&2
  exit 1
fi
export WORK="$WORK_CORRECTED"

echo "== 独立 WORK: $WORK"
echo "== 旧 AID work(只读,不触碰): $OLD_WORK"

# ---- 容量 / inode 预检 [v2.4.4:共享 lquota 库,df 判决已废(P0-1)] ----
# 阈值 env 可覆盖:MIN_QUOTA_GIB(默认 300)/MIN_QUOTA_INODES(默认 20000)。
source "$PBS/_quota_preflight.sh"
if [ "$DRYRUN" != "1" ]; then
  quota_preflight "$WORK" 300 20000 || exit 1
else
  echo "== [DRYRUN] 跳过配额预检"
fi

Q() { if [ "$DRYRUN" = "1" ]; then echo "qsub $*" >&2; echo "jobid_$RANDOM"; else qsub "$@"; fi; }
# 所有作业继承独立 WORK
V="WORK=$WORK"
# 01 生成预注册 official_splits；训练作业从共享 WORK 自动读取。
# 若用户显式传了别的 TRAIN_SPLIT_FILE，也必须原样转进 PBS 环境。
if [ -n "${TRAIN_SPLIT_FILE:-}" ]; then
  V="$V,TRAIN_SPLIT_FILE=$TRAIN_SPLIT_FILE"
fi

# 主线三种子(20-25)+ 单种子消融(26-30)+ 单种子归因(32/33,拍板④)
# + [v2.8] 35 逐段富文本臂 / 36 真 no-pose 臂(审查致命伤 F2/②的对照)
TRAIN=(20_r16_sem_s42 21_r16_sem_s43 22_r16_sem_s44 \
       23_r16_geo_s42 24_r16_geo_s43 25_r16_geo_s44 \
       26_r128_sem_s42 27_r128_geo_s42 \
       28_r16_sem_attncam 29_r16_geo_attncam 30_r8_sem_s42 \
       32_r16_shufpose_s42 33_r16_geostrip_s42 \
       35_r16_richtext_s42 36_r16_nopose_s42)

for f in 01_build_data 34a_learnability_drop 34b_learnability_native "${TRAIN[@]}" 31_eval_base; do
  [ -f "$PBS/${f}.pbs" ] || { echo "FATAL: 缺 PBS $PBS/${f}.pbs" >&2; exit 1; }
done

echo "===== 提交 01 建库(→ 独立 WORK,不删任何旧文件)====="
j01=$(Q -v "$V" "$PBS/01_build_data.pbs")
echo "01_build = $j01"

# =============================================================================
# [v2.8 拍板 2026-07-29] 34 可学习性探针是【判门作业】,横在 01 与主矩阵之间。
# 理由:论文 headline 是 null("方向控制不迁移"),而"范式不迁移"与"这套配方
# 教不会任何东西"必须可区分 —— 34 用最简单的方向任务(单段/四方向/高重叠)
# 给出正对照。它 FAIL 就说明问题在管线不在 backbone,此时提交 20-33/35/36
# 只是在烧 GPU 产生无法解释的数字。全部训练作业因此 afterok:34。
# 34 是独立 profile(豁免 overlap<0.10,独立子目录),不与主表混库。
# =============================================================================
# [v2.8.1] 01 之后必须有【人工批准闸】,不能一路自动冲进 GPU。
# 理由:01 用的是 v2.8 新时间轴(源头去重 17/25/33 + overlap enforce + 语义门),
# 这是第一次跑;语义门是 report-only 口径,它的 attempts/audit 得有人看过、
# oracle 标定产物得在,才谈得上"数据可信"。此前 34 直接 afterok:01 自动起,
# 等于用没人看过的数据去做判门,判门本身就不可信了。
#
# 放行方式(二选一):
#   * 人工核完 01 产物后:  touch $WORK/APPROVED_01_DATA
#   * 明确不要这道闸:      SKIP_DATA_APPROVAL=1 bash .../qsub_corrected_rerun.sh
if [ "${SKIP_DATA_APPROVAL:-0}" != "1" ] && [ "$DRYRUN" != "1" ]; then
  _need=("$WORK/APPROVED_01_DATA")
  # [v2.9.2] 批准必须绑定【这一批数据】。旧版只看空文件是否存在,数据重建后
  # 旧批准仍然放行 —— "批准过一次就永远批准"。现在要求批准文件的内容等于
  # 01 落的 approval_token(它是 GEO/探针 H5 + 语义报告 + overlap 审计 +
  # oracle 标定 的联合摘要),任一环节重建 token 就变,旧批准自动失效。
  _tgt="$WORK/approval_target.json"
  if [ -f "$WORK/APPROVED_01_DATA" ] && [ -f "$_tgt" ]; then
    # [v2.9.3] 本脚本不 source common.sh,$PY 在这里【没有定义】——
    # set -u 下会 'PY: unbound variable' 直接杀掉编排器,而且是在 01 已经提交
    # 出去之后。更阴的是 DRYRUN=1 和 SKIP_DATA_APPROVAL=1 都绕过这段,于是
    # 文档让人先跑的干跑全绿、唯一能跑通的真提交路径恰好是把批准闸关掉那条。
    _PYBIN="${PY:-$(command -v python3 || command -v python || true)}"
    if [ -z "$_PYBIN" ]; then
        echo "FATAL: 找不到 python3/python,无法校验批准凭据" >&2; exit 1
    fi
    _want=$("$_PYBIN" -c "import json,sys;print(json.load(open(sys.argv[1]))['approval_token'])" "$_tgt" 2>/dev/null || echo "")
    _got=$(tr -d '[:space:]' < "$WORK/APPROVED_01_DATA")
    if [ -z "$_want" ] || [ "$_want" != "$_got" ]; then
      echo "FATAL: 批准凭据与当前数据不匹配 —— 数据在批准之后被重建过。" >&2
      echo "       期望 token: ${_want:-<读不出 approval_target.json>}" >&2
      echo "       批准文件里: ${_got:-<空>}" >&2
      echo "       请重新人工核对 01 产物,然后:" >&2
      echo "         echo $_want > $WORK/APPROVED_01_DATA" >&2
      exit 2
    fi
    echo "[编排] 批准凭据与当前数据匹配(token ${_want:0:12}…)"
  fi
  if [ ! -f "$WORK/APPROVED_01_DATA" ]; then
    cat >&2 <<EOM
[编排] 暂停:01 产物尚未人工批准。
       34 判门会拿 01 的数据去回答"这条管线到底能不能学会方向",
       所以它依赖的数据必须先被人看过 —— 否则判门自身不可信。

       请核对(01 跑完后在 \$WORK 下):
         semgate_attempts.json   语义门 report-only 的逐样本判定
         overlap_audit.json      overlap 重采审计(拒样数/换源图/接受分布)
         oracle 标定产物         光流量程标定(否则角误差地板不可解读)
         人工抽检若干样本帧       确认场景语义没被轨迹带跑(P0 人工检查那次)

       核完后:  echo <approval_token> > $WORK/APPROVED_01_DATA
                (token 见 $WORK/approval_target.json,01 跑完会打印)
       然后重跑本编排(01 已 stamp,不会重算)。
       确实要跳过:SKIP_DATA_APPROVAL=1 bash \$0
EOM
    exit 2
  fi
  echo "[编排] 01 产物已人工批准($WORK/APPROVED_01_DATA)"
fi

# [v2.9 拍板] 正式矩阵只许在 official profile 下提交。
# headline = drop(20-22) ↔ native(23-25);identity 只是 36 的诊断臂。
# 谁要用 legacy/identity 跑主矩阵,必须显式承担"产物不得作为 headline"。
_prof="${RSWM_PROFILE:-official}"
_sam="${SEM_ARM_MODE:-}"
if [ "$_prof" != "official" ] || { [ -n "$_sam" ] && [ "$_sam" != "drop" ]; }; then
  echo "FATAL: 正式矩阵要求 RSWM_PROFILE=official 且 SEM_ARM_MODE=drop(或不设)," >&2
  echo "       当前 RSWM_PROFILE=$_prof SEM_ARM_MODE=${_sam:-<未设,将取 drop>}。" >&2
  echo "       identity 只能作为 36 号静止相机诊断臂,不得占语义主线。" >&2
  echo "       确要跑 legacy 口径请用 legacy 链(02-12),别用本编排。" >&2
  exit 1
fi
echo "[编排] profile=official  语义主线=drop(真·text-only)  几何主线=native"
echo "[编排] source-disjoint split=$WORK/official_splits.json（01 生成，后续训练复用）"

# [v2.9.1 拍板] 判门拆成两个通路,共享 01 建的同一份探针库,**并行**跑。
# 两个都 PASS 才放行主矩阵 —— 只让 drop PASS 就解锁全部 native 实验,
# 证据链不完整:native 主线失败时仍可被归因为"pose 表示学不会"。
echo "===== 提交 34a/34b 可学习性判门(并行,afterok:01)====="
j34a=$(Q -v "$V" -W depend=afterok:$j01 "$PBS/34a_learnability_drop.pbs")
j34b=$(Q -v "$V" -W depend=afterok:$j01 "$PBS/34b_learnability_native.pbs")
echo "34a_learnability_drop   = $j34a  ← 文本方向可学习性"
echo "34b_learnability_native = $j34b  ← Plücker 方向可学习性"
echo "                            两个都 PASS 才放行 20-30/35/36"
j34="$j34a:$j34b"

echo "===== 提交 20-30 + 32/33 + 35/36 训练+评估(afterok:34a AND 34b)====="
declare -A JID
for t in "${TRAIN[@]}"; do
  JID[$t]=$(Q -v "$V" -W depend=afterok:$j34 "$PBS/${t}.pbs")
  echo "  $t = ${JID[$t]}"
done

echo "===== 提交 31 baseline 评估(afterok:20)====="
j31=$(Q -v "$V" -W depend=afterok:${JID[20_r16_sem_s42]} "$PBS/31_eval_base.pbs")
echo "31_eval_base = $j31"

cat <<EOF

===================================================================
全部提交完毕(独立 WORK=$WORK;旧 AID work 未被触碰)。
跑完后在【登录节点】跑纯 CPU 分析:

# 1) 方向指标校准
python server_run/validate_direction_metric.py \\
    --gt-h5 $WORK/data/train_semantic.h5 --split $WORK/train_r16_sem_s42/train_val_split.json \\
    --eval-dirs textonly=$WORK/eval_r16_sem_s42 textpluecker=$WORK/eval_r16_geo_s42 base=$WORK/eval_base \\
    -o $WORK/direction_validation.json

# 2) Track B 完整轨迹控制 + 诊断基线
#    [v2.6] :text/:geo 后缀 = 显式路由声明(commanded 语义;v2.6 evaluate 产出的
#    目录带 meta 标记可省略,归因目录 shufpose/geostrip 分析时务必显式 :geo)
python server_run/run_trajectory_eval.py \\
    --gt-h5 $WORK/data/train_geo.h5 --split $WORK/train_r16_sem_s42/train_val_split.json \\
    --eval-dirs textonly=$WORK/eval_r16_sem_s42:text textpluecker=$WORK/eval_r16_geo_s42:geo \\
                geostrip=$WORK/eval_r16_geostrip_s42:geo \\
    --with-baselines -o $WORK/trajectory_eval.json

# 3) 多种子聚合(缺 seed 会 fail-loud;默认含 Angular_Error_WellPosed_Mean,
#    聚合 v2.6 之前的旧目录时用 --metrics 去掉它)
python server_run/aggregate_seeds.py --seeds 42 43 44 \\
    --eval-dirs r16_sem=$WORK/eval_r16_sem_s{42,43,44} r16_geo=$WORK/eval_r16_geo_s{42,43,44} \\
    --pair r16_sem r16_geo --out-prefix $WORK/seed_summary

# 4) 参数效率汇总(含 r128 参数量上界)
python server_run/parameter_efficiency_summary.py --exp-dirs $WORK/train_r16_* $WORK/train_r8_* $WORK/train_r128_* \\
    --out-prefix $WORK/param_efficiency

# 5) [v2.6] 标准 I3D 口径 FVD 离线重算(文献可比;权重 offline_prefetch I3D=1 预取)
python server_run/recompute_fvd_i3d.py --i3d \$ASSET_ROOT/i3d/i3d_torchscript.pt \\
    --eval-dirs textonly=$WORK/eval_r16_sem_s42 textpluecker=$WORK/eval_r16_geo_s42 \\
                base=$WORK/eval_base -o $WORK/fvd_i3d.json --device cpu

# 6) 统一结果总表(corrected 目录优先,run_all 目录回退)
python server_run/collect_results.py $WORK
===================================================================
EOF
