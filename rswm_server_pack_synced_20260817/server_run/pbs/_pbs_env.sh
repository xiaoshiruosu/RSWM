#!/bin/bash
# =============================================================================
# _pbs_env.sh — 被各 NN_*.pbs 用 `source` 引入的公共环境(Gadi/NCI 版)。不单独 qsub。
#   职责:激活 conda、设 Gadi 路径 + 离线开关、cd 到包根、source common.sh
#         (拿到 canonical 配置 + train_lora/eval_lora/eval_base/aligned + stamp)。
#   下面默认值 = 用户在 Gadi 上跑通 P0 的实测路径;换机器改这里或用 -v 覆盖。
# =============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

# ---- conda(Gadi 实测)----
CONDA_SH="${CONDA_SH:-/scratch/qk73/zz1230/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/scratch/rx40/zz1230/envs/rswm}"

# ---- 路径(ASSET_ROOT 放基模/缓存;CODE_ROOT=解压的包根)----
ASSET_ROOT="${ASSET_ROOT:-/scratch/qk73/zz1230/RSWM}"
export CODE_ROOT="${CODE_ROOT:-${PBS_O_WORKDIR:-$ASSET_ROOT/rswm_server_pack}}"
export ROOT="$CODE_ROOT"
export PACK_ROOT="$CODE_ROOT"
export WORK="${WORK:-$CODE_ROOT/server_run/work}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$ASSET_ROOT/lingbot-world-base-cam}"

# ---- PBS/Gadi 保持硬件中立；A800 共享服务器用 server_run/a800_env.sh ----
# auto 按作业内可见 GPU 显存选择 offload，保留原 H200 路径。若 PBS 也调度
# A800，可用 -v HARDWARE_PROFILE=a800_80g,OFFLOAD=1 显式覆盖。
export HARDWARE_PROFILE="${HARDWARE_PROFILE:-auto}"
export RSWM_HARDWARE_PROFILE="$HARDWARE_PROFILE"
export OFFLOAD="${OFFLOAD:-}"
export ATTENTION_BACKEND="${ATTENTION_BACKEND:-auto}"
export RSWM_ATTENTION_BACKEND="$ATTENTION_BACKEND"
export MAX_FRAMES="${MAX_FRAMES:-33}"
export ACCUM="${ACCUM:-4}"

# ---- 缓存全部指向 ASSET_ROOT(计算节点断网,预取好的权重/数据都在这)----
export HF_HOME="${HF_HOME:-$ASSET_ROOT/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$ASSET_ROOT/torch_cache}"   # FID/FVD/LPIPS 权重从这读
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ASSET_ROOT/xdg_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

echo "[pbs] node=$(hostname)  code=$CODE_ROOT  ckpt=$CHECKPOINT_DIR  profile=$HARDWARE_PROFILE  OFFLOAD=${OFFLOAD:-auto}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "[pbs] 无 nvidia-smi(纯 CPU 作业?)"

# conda activate 在 set -u 下会踩未定义变量,临时关掉
set +u
# shellcheck disable=SC1091
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
python -c "import torch;print('[pbs] torch',torch.__version__,'cuda?',torch.cuda.is_available())"

cd "$CODE_ROOT"
[ -f server_run/common.sh ] || { echo "FATAL: $CODE_ROOT 不是包根(缺 server_run/common.sh)"; exit 1; }
# shellcheck disable=SC1091
source "$CODE_ROOT/server_run/common.sh"

# corrected/official 链必须复用 01 预注册的 source-disjoint 三路划分。
# legacy 复现实验不强行切换，以免改变历史数字。这里仅设置默认路径，
# 真正调用 train_lora 时才检查文件：01/40/50 等建库作业在 split 产生前也会
# source 本文件，不能被训练专属前置条件误伤。
if [ "$RSWM_PROFILE" = "official" ]; then
    export TRAIN_SPLIT_FILE="${TRAIN_SPLIT_FILE:-$WORK/official_splits.json}"
fi

# =============================================================================
# [v2.9] 语义臂口径由 common.sh 的 RSWM_PROFILE 统一裁决(official→drop /
# legacy→identity),这里不再自己设默认值 —— 两处各设一份默认必然会漂移。
# 见 common.sh 顶部的 profile 段。
# =============================================================================
