#!/bin/bash
# =============================================================================
# common.sh — 全阶段共享配置(被 p*.sh source,不单独执行)
#
# 代码裁决(全部按仓库真实代码钉死,勿改):
#   构建器 : lora_remotesense/build_dataset_from_benchmark.py  (canonical:
#            AID / window 128x222 / steps 9 / target 480x832 / seed 42)
#            ―― 绝不使用 new/build_from_remotesense.py(96x96/eurosat,非论文管道)
#   训练器 : lora_remotesense/new/train.py
#            ―― 已核实含逐样本 low/high 路由修复(train.py:587-595)
#            ―― A800 80G:bf16/GC/T5+VAE CPU/单专家 offload/torch.AdamW
#   评估器 : new/evaluate.py + new/evaluate_baseline.py
#            ―― metrics.py 相对论文数字版本仅差 v2.7.1 的 sqrtm 调用签名
#               (F7 SciPy 兼容)与 well-posed 双栏;md5 已不同,属预期
#   语义通道: 位姿中和为 identity(neutralize_poses.py),等价原主实验 regime
#   几何通道: 构建器原生 waypoint Plücker 位姿(BUG-4 修复版)
#   超参   : MAX_FRAMES=33(=train_500.pbs)  epochs=5  lr=1e-4  seed=42
#            ACCUM=4(=ablation_rank128.pbs)  alpha=rank(=各 ablation pbs)
# =============================================================================

set -euo pipefail


# =============================================================================
# [v2.9 拍板 2026-07-29] 运行 profile —— 决定"语义臂"到底是什么口径
#
# ICLR benchmark 的 headline 必须是干净的 no-pose ↔ native 对比。identity
# (中和位姿仍走 Plücker 通路)只能当【静止相机命令诊断臂】,不得再占语义主线,
# 也不得再被称作 text-only。
#
#   official(默认)—— 正式矩阵:
#       20-22  drop     + GEO_H5    三种子 真·text-only 主线(headline)
#       23-25  native   + GEO_H5    三种子 Plücker 主线(headline)
#       26/28/30 drop   + GEO_H5    Rank/target 消融跟随主线口径
#       35     drop     + rich-text 逐段文本信息量臂
#       36     identity + SEM_H5    静止相机命令诊断臂(显式,不走自动推断)
#   legacy —— 已退役的 02-10 / p1_main / p2_ranks 链,保持 identity 以复现旧数字。
#
# **不依赖用户临时 export**:官方值就写在这里,legacy 入口自己声明
# RSWM_PROFILE=legacy。命令行仍可覆盖(用于诊断),但会大声打印。
# =============================================================================
export RSWM_PROFILE="${RSWM_PROFILE:-official}"
case "$RSWM_PROFILE" in
    official) _RSWM_DEFAULT_SEM_ARM=drop ;;
    legacy)   _RSWM_DEFAULT_SEM_ARM=identity ;;
    *) echo "FATAL: RSWM_PROFILE=$RSWM_PROFILE 非法(official|legacy)" >&2; exit 1 ;;
esac
if [ -n "${SEM_ARM_MODE:-}" ] && [ "$SEM_ARM_MODE" != "$_RSWM_DEFAULT_SEM_ARM" ]; then
    echo "[profile] ⚠️ SEM_ARM_MODE=$SEM_ARM_MODE 覆盖了 $RSWM_PROFILE profile 的" \
         "默认值 $_RSWM_DEFAULT_SEM_ARM —— 产物不得作为 headline 数字。" >&2
fi
export SEM_ARM_MODE="${SEM_ARM_MODE:-$_RSWM_DEFAULT_SEM_ARM}"
case "$SEM_ARM_MODE" in
    drop|identity|native) : ;;
    *) echo "FATAL: SEM_ARM_MODE=$SEM_ARM_MODE 非法(drop|identity|native)" >&2; exit 1 ;;
esac
# [v2.9.1 拍板] headline 资格是【机器可读的标记】,不是一句 warning。
# 只有 official profile + 未被覆盖的官方口径才配进主表;任何偏离(legacy 链、
# 手动把语义臂改回 identity、诊断性覆盖)都置 0,由 collect_results 自动拒收 ——
# 打印警告靠人看见,标记靠代码执行。
# ⚠️ 这里只算【profile 级】资格。真正的资格还要看单作业的 pose_mode ——
# 36 号显式 POSE_MODE=identity(静止相机诊断臂),但 POSE_MODE 是在各 PBS
# source 完 common.sh 【之后】才 export 的,这里看不到它。所以每次实际调用
# train/eval 时都用 _headline_eligible_for 重算,而不是继承这个基值。
if [ "$RSWM_PROFILE" = "official" ] && [ "$SEM_ARM_MODE" = "$_RSWM_DEFAULT_SEM_ARM" ]; then
    export RSWM_PROFILE_ELIGIBLE=1
else
    export RSWM_PROFILE_ELIGIBLE=0
fi
export RSWM_HEADLINE_ELIGIBLE=0   # 缺省不合格;由 _headline_eligible_for 逐次覆写
export RSWM_HEADLINE_INELIGIBLE_REASON=""
if [ "$RSWM_PROFILE_ELIGIBLE" = "0" ]; then
    if [ "$RSWM_PROFILE" != "official" ]; then
        export RSWM_HEADLINE_INELIGIBLE_REASON="RSWM_PROFILE=$RSWM_PROFILE(非 official 链)"
    else
        export RSWM_HEADLINE_INELIGIBLE_REASON=\
"SEM_ARM_MODE=$SEM_ARM_MODE 覆盖了 official 官方值 $_RSWM_DEFAULT_SEM_ARM"
    fi
fi
echo "[profile] RSWM_PROFILE=$RSWM_PROFILE  SEM_ARM_MODE=$SEM_ARM_MODE  " \
     "profile_eligible=$RSWM_PROFILE_ELIGIBLE${RSWM_HEADLINE_INELIGIBLE_REASON:+ ($RSWM_HEADLINE_INELIGIBLE_REASON)}"

# ---- 路径 ----
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # lingbot 仓库根
NEW="$ROOT/lora_remotesense/new"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?必须设置基模路径, 例: export CHECKPOINT_DIR=/nix/rsroll_25828/models/lingbot-world-base-cam}"
if [ -z "${WORK:-}" ]; then
    if [ -n "${RSWM_BASE:-}" ]; then
        WORK="$RSWM_BASE/work/rsroll_official"
    else
        WORK="$ROOT/server_run/work"
    fi
fi
DATA_DIR="$WORK/data"
STAMPS="$WORK/stamps"
LOGS="$WORK/logs"
mkdir -p "$DATA_DIR" "$STAMPS" "$LOGS"

# ---- canonical 超参(与产出论文数字的 pbs 一致) ----
SEED="${SEED:-42}"               # 训练种子;多种子实验 env 覆盖(SEED=43 qsub …),论文主数字=42
MAX_FRAMES="${MAX_FRAMES:-33}"   # 论文值 33 勿改;仅 80GB 卡 OOM 排障可临时降(如 17),正式数字必须 33
EPOCHS="${EPOCHS:-5}"
ACCUM="${ACCUM:-4}"
LR="1e-4"
# 硬件 profile 只决定显存/优化器工程路径，不改 rank、帧数、LR、数据协议。
# auto 会按作业内可见 GPU 显存选择；共享 A800 服务器由 a800_env.sh 显式设置。
HARDWARE_PROFILE="${HARDWARE_PROFILE:-auto}"
case "$HARDWARE_PROFILE" in
    auto|a800_80g|h200_141g) ;;
    *) echo "FATAL: HARDWARE_PROFILE=$HARDWARE_PROFILE 非法(auto/a800_80g/h200_141g)" >&2; exit 1 ;;
esac
export HARDWARE_PROFILE RSWM_HARDWARE_PROFILE="$HARDWARE_PROFILE"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-${RSWM_ATTENTION_BACKEND:-auto}}"
case "$ATTENTION_BACKEND" in
    auto|flash_attn|sdpa) ;;
    *) echo "FATAL: ATTENTION_BACKEND=$ATTENTION_BACKEND 非法(auto|flash_attn|sdpa)" >&2; exit 1 ;;
esac
export ATTENTION_BACKEND RSWM_ATTENTION_BACKEND="$ATTENTION_BACKEND"
DEVICE="${DEVICE:-cuda}"  # 逻辑设备；由用户的 CUDA_VISIBLE_DEVICES 决定物理卡
export DEVICE
# 训练成本硬门。0=不截断；正整数会被 train_lora 原样转成
# train.py --max_train_samples，并进入 checkpoint/run_manifest 身份。
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
if ! [[ "$MAX_TRAIN_SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "FATAL: MAX_TRAIN_SAMPLES=$MAX_TRAIN_SAMPLES 非法（应为 0 或正整数）" >&2
    exit 1
fi

# ---- 规模档位 ----
# =============================================================================
# [v2.10] 评估规模分层 —— 建库便宜、推理昂贵,两者必须解耦。
#
# 问题:旧版 N_MAIN=500 × train_ratio=0.8 → 评估恒为 100 条,而 FVD 的常规做法是
# 2048+,FID 在 N=100 上基本是噪声(这也解释了"FVD 大降但 FID 反升")。
# headline 完全建立在 N=100 上,审稿人一眼就看到。
#
# 但直接提到 2048 不可行:推理 ~0.11 FPS × 33 帧 ≈ 5 min/条,
# 1000 条 = 83 GPU-hour/臂,6 个 headline 臂 ≈ 500 GPU-hour。
#
# 分层方案:
#   * 建库 N_MAIN 提到 4000(纯 CPU,benchmark 作为**数据集**的规模由它决定)
#   * headline 臂评 N_EVAL_HEADLINE 条(默认 400)
#   * 消融/诊断臂评 N_EVAL_ABLATION 条(默认 100),且用 headline 的**前缀子集**
#     —— 同一批样本,消融的数字与 headline 直接可比,不是两个不相干的 100 条
#   * 另配 metric_sample_size_study.py:在**已生成**的结果上做子采样,
#     给出 FID/FVD 随 N 的收敛曲线与 bootstrap CI。零 GPU 成本,
#     直接回答"你这个 N 够不够"。
#
# ⚠️ 提高 N_MAIN 会让 01 建库变慢并占更多磁盘/inode —— 先跑配额预检。
# =============================================================================
N_MAIN="${N_MAIN:-4000}"         # 主数据集样本数(建库,CPU)
N_EVAL_HEADLINE="${N_EVAL_HEADLINE:-400}"   # headline 臂的评估条数
N_EVAL_ABLATION="${N_EVAL_ABLATION:-100}"   # 消融/诊断臂的评估条数(headline 的前缀子集)
export N_EVAL_HEADLINE N_EVAL_ABLATION
EVAL_N_ABL="${EVAL_N_ABL:-50}"   # 消融评估样本数(主表全量)
OVERLAP_N="${OVERLAP_N:-50}"     # overlap 每档评估样本数(原版 100,可 OVERLAP_N=100)

# ---- 数据集来源(泛化到第二数据集用)----
# aid(默认,论文主结果)/ resisc45(第二数据集,generalization)/ mixed。
# 跑第二数据集: DATASET_TIER=resisc45 WORK=.../work_resisc45 (WORK 换目录隔离产物)。
DATASET_TIER="${DATASET_TIER:-aid}"

# ---- 数据文件 ----
GEO_H5="$DATA_DIR/train_geo.h5"        # 构建器原生输出: waypoint Plücker 位姿(几何通道)
SEM_H5="$DATA_DIR/train_semantic.h5"   # 位姿中和 identity(语义通道 = 论文主实验 regime)

# ---- 产物目录 ----
# [v2.7.1 F10 修复] 目录名与内容物曾相悖:T_MAIN 叫 r16 而 p1_main 实际训
# r128 → parameter_efficiency 表 tag 错配、eval 目录找不着、p2 还会再训一个
# 真 r16 出拿错 checkpoint 风险。按 D3 拍板(主线=r16)统一:p1 主训改 r16,
# r128 移入 p2 的 rank 消融环。
T_MAIN="$WORK/train_r16_sem_s42"       # 主结果: r16 text-only seed42(split/主结果来源)
E_BASE="$WORK/eval_base"               # baseline 评估(全 sweep 复用)
E_MAIN="$WORK/eval_r16_sem_s42"   # [F10] 随主训 r16 更名
T_GEO="$WORK/train_r16_geo"            # [F10] P3 几何通道训练(r16 对齐主线)
E_GEO="$WORK/eval_r16_geo"
OVBASE="$WORK/overlap"                 # P4 overlap 分层

# ---- 运行环境 ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
# [v2.9.1 C9] python 不存在的机器(只有 python3)会让一切 $PY 调用静默失败。
# 显式探测,两个都没有就当场死,而不是把失败伪装成别的症状。
if [ -z "${PY:-}" ]; then
    if command -v python >/dev/null 2>&1; then PY=python
    elif command -v python3 >/dev/null 2>&1; then PY=python3
    else echo "FATAL: 既没有 python 也没有 python3(conda 没激活?)" >&2; exit 1; fi
fi
PY="$PY"

# ---- 显存 offload / optimizer 开关 ----
# OFFLOAD=auto(默认):只在当前可见逻辑 GPU 同时满足 total>=97000MiB 且
# free>=90000MiB 时关闭 offload；A800-80GB 自动解析为 1，空闲 H200 解析为 0。
# OFFLOAD=1:按 timestep 把闲置 14B DiT 移到 CPU；grad + AdamW state 同步迁移。
# OFFLOAD=0:双专家常驻。A800 上只允许用户显式做吞吐/OOM 边界测试，不自动选择。
OFFLOAD_REQUESTED="${OFFLOAD:-auto}"
case "$OFFLOAD_REQUESTED" in
    auto|0|1) ;;
    *) echo "FATAL: OFFLOAD=$OFFLOAD_REQUESTED 非法(只能是 auto、0 或 1)" >&2; exit 1 ;;
esac

_VRAM_TOTAL_MB=""; _VRAM_FREE_MB=""
if [ "$OFFLOAD_REQUESTED" = "auto" ]; then
    # Both calls use logical CUDA device 0 and therefore respect the user's
    # CUDA_VISIBLE_DEVICES instead of inspecting physical GPU 0.
    _MEM_PAIR="$($PY -c 'import torch
if torch.cuda.is_available():
 p=torch.cuda.get_device_properties(0); f,_=torch.cuda.mem_get_info(0)
 print(int(p.total_memory/1024**2), int(f/1024**2))' 2>/dev/null || true)"
    read -r _VRAM_TOTAL_MB _VRAM_FREE_MB <<<"${_MEM_PAIR:-}"
    if [[ "${_VRAM_TOTAL_MB:-}" =~ ^[0-9]+$ ]] && \
       [[ "${_VRAM_FREE_MB:-}" =~ ^[0-9]+$ ]] && \
       [ "$_VRAM_TOTAL_MB" -ge 97000 ] && [ "$_VRAM_FREE_MB" -ge 90000 ]; then
        OFFLOAD_RESOLVED=0
    else
        OFFLOAD_RESOLVED=1
    fi
else
    OFFLOAD_RESOLVED="$OFFLOAD_REQUESTED"
fi
export OFFLOAD="$OFFLOAD_REQUESTED" RSWM_OFFLOAD_RESOLVED="$OFFLOAD_RESOLVED"
echo "[common] OFFLOAD=$OFFLOAD_REQUESTED → $OFFLOAD_RESOLVED " \
     "(visible total=${_VRAM_TOTAL_MB:-unknown}MiB free=${_VRAM_FREE_MB:-unknown}MiB)"

if [ "$HARDWARE_PROFILE" = "a800_80g" ] && [ "$OFFLOAD_RESOLVED" = "0" ]; then
    echo "[common] WARNING: A800 OFFLOAD=0 仅用于显式吞吐/OOM 边界测试；" \
         "真实 33 帧 smoke 通过前不得进入正式矩阵。" >&2
fi
if [ "$HARDWARE_PROFILE" = "h200_141g" ] && [ "$OFFLOAD_RESOLVED" = "1" ]; then
    echo "[common] WARNING: H200 当前选择 offload（可能空闲显存不足或用户显式要求）。" >&2
fi
TRAIN_OFFLOAD_FLAG=(); EVAL_OFFLOAD_FLAG=(); TRAIN_OPTIMIZER_FLAG=()
TRAIN_OFFLOAD_FLAG=(--offload "$OFFLOAD_RESOLVED")
if [ "$OFFLOAD_RESOLVED" = "0" ]; then
    EVAL_OFFLOAD_FLAG=(--no_inactive_model_cpu)
    if [ "$HARDWARE_PROFILE" = "h200_141g" ] || \
       { [ "$HARDWARE_PROFILE" = "auto" ] && \
         [[ "${_VRAM_TOTAL_MB:-}" =~ ^[0-9]+$ ]] && [ "$_VRAM_TOTAL_MB" -ge 97000 ]; }; then
        TRAIN_OPTIMIZER_FLAG=(--use_8bit_adam)
    else
        TRAIN_OPTIMIZER_FLAG=(--no_8bit_adam)
    fi
    echo "[common] resolved OFFLOAD=0 —— 双专家常驻；A800 路径仍使用 torch.AdamW"
else
    TRAIN_OPTIMIZER_FLAG=(--no_8bit_adam)
    echo "[common] resolved OFFLOAD=1 —— 单专家 GPU + torch.AdamW state 同步换入换出"
fi
TRAIN_HARDWARE_FLAG=(--hardware_profile "$HARDWARE_PROFILE")
echo "[common] HARDWARE_PROFILE=$HARDWARE_PROFILE  ATTENTION=$ATTENTION_BACKEND  DEVICE=$DEVICE  MAX_FRAMES=$MAX_FRAMES  BATCH=1  ACCUM=$ACCUM"

# ---- 断点续跑工具 ----
stamp()  { touch "$STAMPS/$1.done"; echo "[stamp] $1 @ $(date '+%F %T')"; }
done_p() { [ -f "$STAMPS/$1.done" ]; }

# ---- [v2.4.3] 内容绑定 stamp:门/衍生物必须跟当前 H5 内容走 ----
# 布尔 stamp 的旁路:强制重建(删 p1_build.done 或改 builder)后,旧的
# p1_dupgate.done/p1_neutralize.done 仍在 → 新 H5 不再过门、SEM 不再同步。
# sha 绑定后:H5 内容一变,门与 neutralize 必然重跑。
# 用法: GEO_SHA=$(h5_sha "$GEO_H5")
#       if ! sha_stamp_ok "$STAMPS/p1_dupgate.sha256" "$GEO_SHA"; then ... fi
h5_sha()       { sha256sum "$1" | awk '{print $1}'; }
sha_stamp_ok() { [ -f "$1" ] && [ "$(cat "$1")" = "$2" ]; }
sha_stamp()    { printf '%s\n' "$2" > "$1"; echo "[stamp] $(basename "$1") ← ${2:0:12}…"; }

# ---- 训练封装: train_lora TAG RANK ALPHA DATA_H5 OUTDIR ----
# ---- [v2.8.2 P0/C3] pose_mode 缺省推断:显式 POSE_MODE 优先,否则读 H5 ----
# v2.8 给 train.py 加了 pose_mode↔H5 一致性硬门,而默认值是 native。于是所有
# "用 SEM_H5(identity_neutralized)却没设 POSE_MODE"的老入口全部启动即失败:
# p0_smoke.sh(DEPLOY 明写"一步都不能跳"的 canary)、p1_main.sh、p2_ranks.sh、
# pbs/02 与 04-10。v2.8.1 只补了 corrected 链的 20-33,漏了这一大批。
# 让默认值永不与数据矛盾,比逐个脚本补 export 更不容易再漏。
# ---- [v2.9.2 P0-3] 单作业级 headline 资格:profile 资格 AND pose_mode ≠ identity ----
# identity 按拍板定义是【静止相机命令诊断臂】,永远不入 headline ——
# 36 号与 identity baseline 都归此类。旧写法只看 profile+SEM_ARM_MODE,
# 于是 36 号虽然显式 POSE_MODE=identity,仍继承了 headline_eligible=true。
# 与 run_manifest.py 的 _NEVER_HEADLINE_ROLES 保持同一名单(改一处必须改两处;
# test_pose_timeline.py 有一条用例钉住二者一致)。
_RSWM_NEVER_HEADLINE_ROLES="diagnostic_stationary probe_gate legacy variant_attribution ladder trivial_baseline metric_calibration submission_scoring"
_role_never_headline() {
    case " $_RSWM_NEVER_HEADLINE_ROLES " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}
_headline_eligible_for() {
    local _pm="$1" _rl="${2:-}"
    if [ -n "$_rl" ] && _role_never_headline "$_rl"; then echo 0; return; fi
    if [ "${RSWM_PROFILE:-unset}" = "official" ] && \
       { [ "${RSWM_PROMPT_SCHEMA:-}" != "net_continuous" ] || \
         [ "${RSWM_PATH_ESTIMATOR:-}" != "registration" ]; }; then
        echo 0; return
    fi
    if [ "${RSWM_PROFILE_ELIGIBLE:-0}" = "1" ] && [ "$_pm" != "identity" ]; then
        echo 1
    else
        echo 0
    fi
}
_headline_reason_for() {
    local _pm="$1" _rl="${2:-}"
    if [ -n "$_rl" ] && _role_never_headline "$_rl"; then
        echo "role=$_rl 属诊断/判门/legacy,不入 headline"; return
    fi
    if [ "${RSWM_PROFILE:-unset}" = "official" ] && \
       { [ "${RSWM_PROMPT_SCHEMA:-}" != "net_continuous" ] || \
         [ "${RSWM_PATH_ESTIMATOR:-}" != "registration" ]; }; then
        echo "official 主通路要求 prompt_schema=net_continuous 且 path_estimator=registration"; return
    fi
    if [ "${RSWM_PROFILE_ELIGIBLE:-0}" != "1" ]; then
        echo "${RSWM_HEADLINE_INELIGIBLE_REASON:-profile 不合格}"
    elif [ "$_pm" = "identity" ]; then
        echo "pose_mode=identity 是静止相机命令诊断臂,按拍板定义永远不入 headline"
    else
        echo ""
    fi
}

_infer_pose_mode() {
    local _h5="$1"
    if [ -n "${POSE_MODE:-}" ]; then echo "$POSE_MODE"; return; fi
    # [v2.9.1 C9] 区分"读到了但不是 identity"与"根本没读成"。
    # 旧写法把 $PY 不存在 / h5py 缺失 / 文件损坏一律吞成 native ——
    # 而 common.sh 的 PY 默认是 `python`(很多机器只有 python3),
    # conda 没激活成功时 SEM_H5 会被静默判成 native,报错文案还会指向
    # "数据是中和 H5"而不是"推断器挂了"。
    local _out _rc
    _out="$($PY - "$_h5" <<'PYPM'
import sys
import h5py
with h5py.File(sys.argv[1], "r") as f:
    _v = f["metadata"].attrs.get("pose_injection", "") if "metadata" in f else ""
    _v = _v.decode() if isinstance(_v, bytes) else str(_v)
print("identity" if "identity" in _v else "native")
PYPM
)" ; _rc=$?
    if [ "$_rc" -ne 0 ] || [ -z "$_out" ]; then
        echo "FATAL: 无法从 $_h5 推断 pose_mode(\$PY=$PY,退出码 $_rc)。" >&2
        echo "       这不是'数据不是中和 H5',而是推断器本身没跑起来 ——" >&2
        echo "       检查 conda 是否激活、h5py 是否可用、文件是否存在。" >&2
        echo "       确要指定请显式 export POSE_MODE=native|identity|drop。" >&2
        exit 1
    fi
    echo "$_out"
}

train_lora() {
    local tag=$1 rank=$2 alpha=$3 data=$4 out=$5
    local _TRAIN_PM; _TRAIN_PM="$(_infer_pose_mode "$data")"
    # [v2.9.3] role 不 export —— export 会泄漏到后续命令,于是同一脚本里
    # train_lora 之后的 eval_lora 会拿到 role="train";更糟的是 34a/34b
    # 从没设过 role → 默认 train/eval → probe_gate 这条"判门不入 headline"
    # 的规则成了死代码,判门产物会被标成合格 headline。改 per-command 传。
    local _role="${RSWM_RUN_ROLE:-train}"
    # [v2.9.2 P0-3] 资格逐次重算 —— 36 号的 identity 不得继承 profile 级的 1
    export RSWM_HEADLINE_ELIGIBLE="$(_headline_eligible_for "$_TRAIN_PM" "$_role")"
    export RSWM_HEADLINE_INELIGIBLE_REASON="$(_headline_reason_for "$_TRAIN_PM" "$_role")"
    echo "===== [train:$tag] role=$_role r=$rank a=$alpha pose_mode=$_TRAIN_PM " \
         "headline=$RSWM_HEADLINE_ELIGIBLE data=$(basename "$data") ====="
    mkdir -p "$out"
    # 集群墙钟超时/抢占后自动续训(PBS walltime / Slurm 通用):若已存在 epoch
    # checkpoint(带 training_state.pt),从最新一个 --resume_from 恢复(train.py:977 支持),
    # 避免重投作业时从 epoch 1 重跑。关闭自动续训: NO_AUTO_RESUME=1。
    local resume_flag=() latest_ckpt=""
    if [ "${NO_AUTO_RESUME:-0}" != "1" ]; then
        # 从最新往旧找第一个「带 training_state.pt」的 checkpoint。
        # 关键:不能只看最新目录 —— 若作业在写 epoch-N 的目录时被杀(train.py 最后才存
        # training_state.pt),最新目录会缺 state,必须回退到上一个完整 checkpoint,否则从头重跑。
        while IFS= read -r d; do
            [ -n "$d" ] || continue
            if [ -f "$d/training_state.pt" ]; then latest_ckpt="$d"; break; fi
        done < <(ls -dt "$out"/checkpoint-epoch-* 2>/dev/null || true)
        if [ -n "$latest_ckpt" ]; then
            resume_flag=(--resume_from "$latest_ckpt")
            echo "[train:$tag] 检测到断点 $(basename "$latest_ckpt") → 自动 --resume_from 续训"
        fi
    fi
    local train_cap_flag=()
    if [ "$MAX_TRAIN_SAMPLES" -gt 0 ]; then
        train_cap_flag=(--max_train_samples "$MAX_TRAIN_SAMPLES")
        echo "[train:$tag] MAX_TRAIN_SAMPLES=$MAX_TRAIN_SAMPLES（确定性成本硬门）"
    fi
    local split_file_flag=()
    if [ -n "${TRAIN_SPLIT_FILE:-}" ]; then
        [ -f "$TRAIN_SPLIT_FILE" ] || {
            echo "FATAL: TRAIN_SPLIT_FILE 不存在: $TRAIN_SPLIT_FILE" >&2; exit 1;
        }
        split_file_flag=(--split_file "$TRAIN_SPLIT_FILE")
        echo "[train:$tag] 使用预注册 split: $TRAIN_SPLIT_FILE"
    fi
    ( cd "$NEW" && RSWM_RUN_ROLE="$_role" $PY train.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --data_path "$data" \
        --output_dir "$out" \
        --num_epochs "$EPOCHS" --batch_size 1 \
        --lora_rank "$rank" --lora_alpha "$alpha" \
        --learning_rate "$LR" --max_frames "$MAX_FRAMES" \
        --gradient_accumulation_steps "$ACCUM" --seed "$SEED" \
        --attention_backend "$ATTENTION_BACKEND" \
        --split_seed "${SPLIT_SEED:-42}" \
        --lora_targets "${LORA_TARGETS:-all}" \
        --pose_mode "$_TRAIN_PM" \
        "${train_cap_flag[@]}" \
        "${split_file_flag[@]}" \
        "${resume_flag[@]}" \
        "${TRAIN_HARDWARE_FLAG[@]}" \
        "${TRAIN_OPTIMIZER_FLAG[@]}" \
        "${TRAIN_OFFLOAD_FLAG[@]}" \
    ) 2>&1 | tee -a "$LOGS/train_${tag}.log"
    [ -d "$out/best_checkpoint" ] || { echo "FATAL: $out/best_checkpoint 不存在,训练失败"; exit 1; }
}

# ---- LoRA 评估封装: eval_lora TAG LORA_CKPT DATA_H5 SPLIT OUTDIR MAX_SAMPLES(0=全量) ----
eval_lora() {
    local tag=$1 ckpt=$2 data=$3 split=$4 out=$5 n=$6
    # [v2.8 ④] 评估必须与训练同 pose_mode —— 三臂唯一区分点。checkpoint 里
    # 记了 pose_mode,这里交叉校验:不一致直接死,不许"训 drop 评 native"。
    local _pm="${POSE_MODE:-native}"
    local _ck_pm
    _ck_pm=$($PY - "$ckpt" <<'PYCK'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "training_config.json"
try:
    print(json.loads(p.read_text()).get("data_split", {}).get("pose_mode")
          or json.loads(p.read_text()).get("pose_mode") or "")
except Exception:
    print("")
PYCK
)
    # [v2.8.2] POSE_MODE 未显式设置时,以 checkpoint 记录为准 —— 它是"这个权重
    # 到底按什么口径训出来的"的唯一真源。旧写法把未设等同于 native,于是老入口
    # "train(自动推断 identity) → eval(未设 → native)"会自己撞自己的硬门。
    # 显式设置了才做冲突校验(仍然不许"训 drop 评 native")。
    # [v2.9.1 C4] 采纳 checkpoint 值之前,必须先确认它与【本次评估数据】相容。
    # 13-16 overlap 档用 02 号在 SEM_H5 上训的 checkpoint(记 identity),
    # 但 DATA_H5 是 native 位姿的 overlap 库 —— 盲目采纳 identity 会让
    # evaluate 的双向门在加载完两个 14B 之后第一条样本就 RuntimeError;
    # 就算不崩,LoRA 臂(identity)与 baseline 臂(native)也在两个口径上,Δ 无意义。
    # 数据说了算,checkpoint 只在与数据相容时用来补全未声明的模式。
    if [ -z "${POSE_MODE:-}" ] && [ -n "$_ck_pm" ]; then
        local _data_pm; _data_pm="$(_infer_pose_mode "$data")"
        if [ "$_ck_pm" = "$_data_pm" ] || [ "$_ck_pm" = "drop" ]; then
            _pm="$_ck_pm"
            echo "[eval:$tag] POSE_MODE 未设,采用 checkpoint 的 pose_mode=$_pm"
        else
            _pm="$_data_pm"
            echo "[eval:$tag] ⚠️ POSE_MODE 未设;checkpoint 记的是 $_ck_pm,但本次" \
                 "数据 $data 的口径是 $_data_pm —— 以【数据】为准取 $_pm。" \
                 "(训练数据与评估数据口径不同,Δ 的可比性自行负责)"
        fi
    elif [ -n "$_ck_pm" ] && [ "$_ck_pm" != "$_pm" ]; then
        echo "FATAL: pose_mode 冲突 —— checkpoint=$_ck_pm 评估请求=$_pm。"
        echo "       三臂对比要求训评同模式;设 POSE_MODE=$_ck_pm 重跑。"
        exit 1
    fi
    # [v2.8.2 N5] 用 per-command 环境而非 export —— export 会留在整个 shell,
    # 一旦有人把 eval_lora(drop) → eval_base 排在同一脚本里,baseline 会静默
    # 继承 drop 通路,污染对照臂而且完全无症状。
    local _role="${RSWM_RUN_ROLE:-eval}"
    export RSWM_HEADLINE_ELIGIBLE="$(_headline_eligible_for "$_pm" "$_role")"
    export RSWM_HEADLINE_INELIGIBLE_REASON="$(_headline_reason_for "$_pm" "$_role")"
    echo "===== [eval:$tag] n=$n pose_mode=$_pm role=$_role headline=$RSWM_HEADLINE_ELIGIBLE ====="
    mkdir -p "$out"
    local extra=(); [ "$n" != "0" ] && extra=(--max_samples "$n")
    ( cd "$NEW" && RSWM_POSE_MODE="$_pm" RSWM_RUN_ROLE="$_role" $PY evaluate.py \
        --checkpoint_dir "$CHECKPOINT_DIR" --lora_path "$ckpt" \
        --data_path "$data" --split_file "$split" \
        --output_dir "$out" --frame_num "$MAX_FRAMES" --device "$DEVICE" \
        --gen_seed "${GEN_SEED:-42}" \
        "${extra[@]}" \
        "${EVAL_OFFLOAD_FLAG[@]}" \
    ) 2>&1 | tee "$LOGS/eval_${tag}.log"
}

# ---- baseline 评估封装: eval_base TAG DATA_H5 SPLIT OUTDIR MAX_SAMPLES ----
eval_base() {
    local tag=$1 data=$2 split=$3 out=$4 n=$5
    # [v2.8.2 P0] baseline 必须与被比较的臂同 pose 口径。
    # eval_lora 会 export RSWM_POSE_MODE,eval_base 此前完全没有这一步 ——
    # 后果有两面,都很贵:
    #   ① identity 臂:31 号 export POSE_MODE=identity 对 eval_base 是空操作,
    #      evaluate.py 的运行期硬门读到默认 native + SEM_H5 的 identity 位姿
    #      → 第一个样本就 RuntimeError,baseline 一张图都不出,主表所有 Δ 落空;
    #   ② drop 臂:baseline 静默走 native 通路拿到 identity Plücker 条件,
    #      而 35/36 存在的全部意义就是单变量对照 —— 差值里混进了"位姿通路开/关"。
    local _pm; _pm="$(_infer_pose_mode "$data")"
    local _role="${RSWM_RUN_ROLE:-baseline}"
    export RSWM_HEADLINE_ELIGIBLE="$(_headline_eligible_for "$_pm" "$_role")"
    export RSWM_HEADLINE_INELIGIBLE_REASON="$(_headline_reason_for "$_pm" "$_role")"
    echo "===== [eval-base:$tag] n=$n pose_mode=$_pm role=$_role headline=$RSWM_HEADLINE_ELIGIBLE ====="
    mkdir -p "$out"
    local extra=(); [ "$n" != "0" ] && extra=(--max_samples "$n")
    ( cd "$NEW" && RSWM_POSE_MODE="$_pm" RSWM_RUN_ROLE="$_role" $PY evaluate_baseline.py \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --data_path "$data" --split_file "$split" \
        --output_dir "$out" --frame_num "$MAX_FRAMES" --device "$DEVICE" \
        --gen_seed "${GEN_SEED:-42}" \
        "${extra[@]}" \
        "${EVAL_OFFLOAD_FLAG[@]}" \
    ) 2>&1 | tee "$LOGS/evalbase_${tag}.log"
}

# ---- 对齐对比封装: aligned TAG LORA_DIR BASE_DIR OUTDIR ----
aligned() {
    local tag=$1 lora=$2 base=$3 out=$4
    ( cd "$NEW" && $PY eval_aligned.py \
        --lora_dir "$lora" --base_dir "$base" \
        --output_dir "$out" --device "$DEVICE" \
    ) 2>&1 | tee "$LOGS/aligned_${tag}.log"
}

# =============================================================================
# [RSRoll-Bench] 三个新通路开关 —— 在这里集中 export，并**大声打印**。
#
# 为什么必须在 common.sh 里 export：PBS 作业不继承提交端 shell 的环境
# （除非 qsub -V）。若只在交互 shell 里 export，作业会静默跑成旧口径：
# prompt 仍是 4 个词、Track B 仍走光流 —— 而输出看起来完全正常。
# 这类失配唯一的补救是 run_manifest 的身份字段（它会记下实际生效的值，
# 并让两批产物无法混用），但那是**事后**发现，代价是一整批 GPU 时间。
#
# 默认值全部 = 旧行为，改动是显式的。
# =============================================================================
# 正式 ICLR 通路默认连续角 prompt + registration；legacy 复现实验保留
# 四方向词 + flow。这样正式作业不再依赖 qsub -V 或交互 shell 的临时 export。
if [ "$RSWM_PROFILE" = "official" ]; then
    _RSWM_DEFAULT_PROMPT_SCHEMA=net_continuous
    _RSWM_DEFAULT_PATH_ESTIMATOR=registration
else
    _RSWM_DEFAULT_PROMPT_SCHEMA=net
    _RSWM_DEFAULT_PATH_ESTIMATOR=flow
fi
export RSWM_PROMPT_SCHEMA="${RSWM_PROMPT_SCHEMA:-$_RSWM_DEFAULT_PROMPT_SCHEMA}"
export RSWM_PATH_ESTIMATOR="${RSWM_PATH_ESTIMATOR:-$_RSWM_DEFAULT_PATH_ESTIMATOR}"
export RSWM_BACKBONE="${RSWM_BACKBONE:-wan-inline}"
case "$RSWM_PROMPT_SCHEMA" in
    net|net_continuous|per_segment_continuous|stripped) ;;
    *) echo "FATAL: RSWM_PROMPT_SCHEMA=$RSWM_PROMPT_SCHEMA 非法" >&2; exit 1 ;;
esac
case "$RSWM_PATH_ESTIMATOR" in
    flow|registration) ;;
    *) echo "FATAL: RSWM_PATH_ESTIMATOR=$RSWM_PATH_ESTIMATOR 非法(flow|registration)" >&2; exit 1 ;;
esac

echo "[common] ROOT=$ROOT"
echo "[common] WORK=$WORK  CKPT=$CHECKPOINT_DIR"
echo "[rsroll] prompt_schema=$RSWM_PROMPT_SCHEMA  path_estimator=$RSWM_PATH_ESTIMATOR  backbone=$RSWM_BACKBONE"
if [ "$RSWM_PROFILE" = "official" ] && { \
        [ "$RSWM_PROMPT_SCHEMA" != "net_continuous" ] || \
        [ "$RSWM_PATH_ESTIMATOR" != "registration" ]; }; then
    export RSWM_HEADLINE_ELIGIBLE=0
    export RSWM_HEADLINE_INELIGIBLE_REASON="official 主通路要求 prompt=net_continuous 且 estimator=registration"
    echo "[rsroll] ⚠️ official 口径被覆盖：产物不得作为 headline。" >&2
fi
if [ "$RSWM_PROMPT_SCHEMA" = "net" ] && [ "$RSWM_PATH_ESTIMATOR" = "flow" ]; then
    echo "[rsroll] ⚠️ 当前是**旧口径**(4 词方向 + 光流估计器)。"
    echo "[rsroll]    要跑本文主通路：export RSWM_PROMPT_SCHEMA=net_continuous RSWM_PATH_ESTIMATOR=registration"
fi
