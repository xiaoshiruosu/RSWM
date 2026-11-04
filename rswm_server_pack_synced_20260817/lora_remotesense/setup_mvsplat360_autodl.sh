#!/bin/bash
# =========================================================================
# MVSplat360 环境配置 — AutoDL
#
# 推荐镜像: PyTorch 2.2.0 + CUDA 12.1 (或更高)
# 推荐GPU: A100 80G / A800 80G
#
# Usage:
#   bash lora_remotesense/setup_mvsplat360_autodl.sh
#
# 总耗时约 15-20 分钟
# =========================================================================

set -euo pipefail

RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
INSTALL_DIR="${INSTALL_DIR:-$RSWM_BASE/baselines/mvsplat360/src}"
CKPT_DIR="${INSTALL_DIR}/checkpoints"
LINGBOT_DIR="${LINGBOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$RSWM_BASE/cache/pip-mvsplat}"
export TMPDIR="${TMPDIR:-$RSWM_BASE/tmp/mvsplat}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
mkdir -p "$(dirname "$INSTALL_DIR")" "$PIP_CACHE_DIR" "$TMPDIR"

PY="${PY:-python}"
if "$PY" -c 'import flash_attn' >/dev/null 2>&1; then
    echo "FATAL: 检测到 RSWM 主训练环境。MVSplat360 必须使用独立 torch2.2.2 环境，不能改动训练环境。" >&2
    exit 2
fi

echo "============================================"
echo "MVSplat360 Setup (AutoDL)"
echo "Start: $(date)"
echo "============================================"

# ------------------------------------------------------------------
# Step 1: 系统依赖
# ------------------------------------------------------------------
echo ""
echo "[Step 1/5] 独立环境检查（不升级 pip）..."
"$PY" -m pip --version

# ------------------------------------------------------------------
# Step 2: Clone MVSplat360
# ------------------------------------------------------------------
echo ""
echo "[Step 2/5] Clone MVSplat360..."

if [ -d "$INSTALL_DIR" ]; then
    echo "  已存在, pull latest..."
    cd "$INSTALL_DIR" && git pull && cd -
else
    git clone https://github.com/donydchen/mvsplat360.git "$INSTALL_DIR"
fi

# ------------------------------------------------------------------
# Step 3: 安装依赖
# ------------------------------------------------------------------
echo ""
echo "[Step 3/5] 安装 Python 依赖..."

# 如果 AutoDL 镜像自带的 PyTorch 版本不匹配, 先装 PyTorch
TORCH_VER=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
echo "  当前 PyTorch: $TORCH_VER"

if [[ "$TORCH_VER" == "none" ]] || [[ "$TORCH_VER" != 2.2* ]]; then
    echo "  安装 PyTorch 2.2.2..."
    "$PY" -m pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cu121
fi

# xformers
"$PY" -m pip install xformers==0.0.25.post1 --index-url https://download.pytorch.org/whl/cu121 2>/dev/null \
    || "$PY" -m pip install xformers

# MVSplat360 requirements
cd "$INSTALL_DIR"
"$PY" -m pip install -r requirements.txt

# 评估所需额外依赖
"$PY" -m pip install h5py scipy scikit-image lpips opencv-python-headless

echo "  依赖安装完成"

# ------------------------------------------------------------------
# Step 4: 下载权重
# ------------------------------------------------------------------
echo ""
echo "[Step 4/5] 下载权重..."

mkdir -p "$CKPT_DIR"
HF311_ENV="${HF311_ENV:-$RSWM_BASE/envs/hf311}"
HF_PY="$HF311_ENV/bin/python"
if [ -x "$HF311_ENV/bin/hf" ]; then
    HF_CLI="$HF311_ENV/bin/hf"
elif [ -x "$HF311_ENV/bin/huggingface-cli" ]; then
    HF_CLI="$HF311_ENV/bin/huggingface-cli"
else
    echo "FATAL: 缺少管理员下载环境 $HF311_ENV；不要在 MVSplat 或训练环境里安装 huggingface_hub。" >&2
    exit 2
fi
"$HF_PY" -c 'import sys,huggingface_hub as h; assert sys.version.split()[0]=="3.11.15"; assert h.__version__=="0.35.3"' || {
    echo "FATAL: 下载环境必须是 Python 3.11.15 + huggingface_hub 0.35.3。" >&2
    exit 2
}

# 4a. re10k.ckpt (~48MB, 无需认证)
if [ -f "${CKPT_DIR}/re10k.ckpt" ]; then
    echo "  [OK] re10k.ckpt"
else
    echo "  下载 re10k.ckpt (MVSplat encoder, ~48MB)..."
    "$HF_CLI" download donydchen/mvsplat re10k.ckpt --local-dir "$CKPT_DIR"
fi

# 4b. dl3dv_480p.ckpt (MVSplat360 主模型, 无需认证)
if [ -f "${CKPT_DIR}/dl3dv_480p.ckpt" ]; then
    echo "  [OK] dl3dv_480p.ckpt"
else
    echo "  下载 dl3dv_480p.ckpt (MVSplat360 full model)..."
    "$HF_CLI" download donydchen/mvsplat360 dl3dv_480p.ckpt --local-dir "$CKPT_DIR"
fi

# 4c. svd.safetensors (~9.5GB, gated model 需要 HF 登录)
if [ -f "${CKPT_DIR}/svd.safetensors" ]; then
    echo "  [OK] svd.safetensors"
else
    echo "  下载 svd.safetensors (SVD, ~9.5GB)..."
    echo "  ⚠ 这是 gated model, 如果失败请先:"
    echo "    1. 浏览器打开 https://huggingface.co/stabilityai/stable-video-diffusion-img2vid"
    echo "    2. 点击 'Agree and access repository'"
    echo "    3. 运行: huggingface-cli login"
    "$HF_CLI" download stabilityai/stable-video-diffusion-img2vid \
        svd.safetensors --local-dir "$CKPT_DIR" \
        || echo "  [FAIL] svd.safetensors 下载失败, 请手动处理 (见上方说明)"
fi

# ------------------------------------------------------------------
# Step 5: 上传 lingbot 评估代码
# ------------------------------------------------------------------
echo ""
echo "[Step 5/5] 检查 lingbot 代码..."

if [ -d "${LINGBOT_DIR}/lora_remotesense" ]; then
    echo "  lingbot 代码已存在: ${LINGBOT_DIR}"
else
    echo "  ⚠ 请上传 lingbot 项目到 ${LINGBOT_DIR}"
    echo "  需要的文件:"
    echo "    - lora_remotesense/evaluate_mvsplat360.py"
    echo "    - lora_remotesense/evaluate.py"
    echo "    - lora_remotesense/metrics.py"
    echo "    - 数据文件: train_hd_500.h5 + train_val_split.json"
fi

# ------------------------------------------------------------------
# 验证
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo "验证"
echo "============================================"

"$PY" -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else '')
"

echo ""
echo "权重状态:"
for f in re10k.ckpt svd.safetensors dl3dv_480p.ckpt; do
    if [ -f "${CKPT_DIR}/$f" ]; then
        SIZE=$(du -h "${CKPT_DIR}/$f" | cut -f1)
        echo "  [OK]   $f ($SIZE)"
    else
        echo "  [MISS] $f"
    fi
done

echo ""
echo "============================================"
echo "Setup 完成! $(date)"
echo ""
echo "运行评估:"
echo "  cd ${LINGBOT_DIR}"
echo "  PYTHONPATH=${LINGBOT_DIR}:\$PYTHONPATH python lora_remotesense/evaluate_mvsplat360.py \\"
echo "    --mvsplat360_dir ${INSTALL_DIR} \\"
echo "    --checkpoint ${CKPT_DIR}/dl3dv_480p.ckpt \\"
echo "    --data_path <你的 eval_data.h5 路径> \\"
echo "    --split_file <你的 split.json 路径> \\"
echo "    --output_dir ./eval_mvsplat360 \\"
echo "    --max_samples 20"
echo "============================================"
