#!/bin/bash
# =============================================================================
# offline_prefetch.sh — 在【能联网的机器】上一键下载断网计算节点所需的全部内容
#
# 产出一个 STAGE 目录,整体拷到计算节点后,P0-P5 全流程零联网。
#
# 用法(联网机上):
#   bash server_run/offline_prefetch.sh                  # 基模+AID+评估权重(~60GB)
#   WHEELS=1 bash server_run/offline_prefetch.sh         # 追加 pip wheelhouse(节点连 pypi 都不通时)
#   STAGE=/data/stage bash server_run/offline_prefetch.sh
#   国内: export HF_ENDPOINT=https://hf-mirror.com 后再跑
#
# 下载清单:
#   1) 基模 robbyant/lingbot-world-base-cam  ~56GB (含 DiT×2 / T5 / VAE / tokenizer,训练+评估唯一模型源)
#   2) AID 数据集 blanchon/AID (HF hub 缓存格式) ~3GB (canonical 建库 + check_aid_pools 都从它读)
#   3) 评估预训练权重 ~0.5GB: inception_v3(FID) + r3d_18(FVD代理) + alexnet(LPIPS backbone)
#   4) [可选 WHEELS=1] torch 2.4.0+cu121 / requirements / flash-attn wheel ~4GB
# =============================================================================
set -euo pipefail

RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
STAGE="${STAGE:-$RSWM_BASE/downloads/offline_stage}"
mkdir -p "$STAGE"
export HF_HOME="${HF_HOME:-$RSWM_BASE/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export TMPDIR="${TMPDIR:-$RSWM_BASE/tmp}"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export TORCH_HOME="${TORCH_HOME:-$RSWM_BASE/cache/torch}"
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Downloads intentionally use the administrator-approved Python 3.11 env.
# Never upgrade huggingface_hub in the training environment from this script.
HF311_ENV="${HF311_ENV:-$RSWM_BASE/envs/hf311}"
HF_PY="${HF_PY:-$HF311_ENV/bin/python}"
if [ -x "$HF311_ENV/bin/hf" ]; then
    HF_CLI="${HF_CLI:-$HF311_ENV/bin/hf}"
elif [ -x "$HF311_ENV/bin/huggingface-cli" ]; then
    HF_CLI="${HF_CLI:-$HF311_ENV/bin/huggingface-cli}"
else
    echo "FATAL: 找不到 $HF311_ENV/bin/{hf,huggingface-cli}；不要在训练 env 里 pip -U，先恢复专用 hf311 环境。" >&2
    exit 2
fi
"$HF_PY" -c 'import sys,huggingface_hub as h; py=sys.version.split()[0]; hv=h.__version__; print("[hf-download]",py,hv); assert py=="3.11.15", f"expected Python 3.11.15, got {py}"; assert hv=="0.35.3", f"expected huggingface_hub 0.35.3, got {hv}"' || {
    echo "FATAL: 下载环境必须是管理员提供的 Python 3.11.15 + huggingface_hub 0.35.3；不会修改训练环境。" >&2
    exit 2
}

echo "=================================================================="
echo " RSWM 离线预取  STAGE=$STAGE"
echo " HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co (默认;国内建议 hf-mirror.com)}"
echo "=================================================================="

# ---- 1. 基模(~56GB,断点续传,重跑会跳过已完成分片) ----
echo; echo "[1/4] 基模 robbyant/lingbot-world-base-cam → $STAGE/lingbot-world-base-cam"
"$HF_CLI" download robbyant/lingbot-world-base-cam \
    --local-dir "$STAGE/lingbot-world-base-cam"

# ---- 2. AID 数据集 → HF hub 缓存(节点上 load_dataset 离线直读) ----
echo; echo "[2/4] AID 数据集 blanchon/AID → $HF_HOME/hub/"
"$HF_CLI" download blanchon/AID --repo-type dataset

# 第二数据集(泛化实验用):设 TIER2=1 时预取 RESISC45 + PatternNet。
# 只跑 AID 论文主结果不用管;要跑 DATASET_TIER=resisc45 才需要。
if [ "${TIER2:-0}" = "1" ]; then
    echo "  [TIER2] RESISC45 泛化数据集 → $HF_HOME/hub/"
    "$HF_CLI" download blanchon/RESISC45   --repo-type dataset
    "$HF_CLI" download blanchon/PatternNet --repo-type dataset
    # WHU-RS19:19 类 1005 张(1001×600² + 4 个异形尺寸),零拼接,~100MB,CC-BY-4.0
    # [v2.5 拍板⑤] WHU-RS19 = 泛化【主】集;RESISC45/PatternNet 为辅助
    # (2×2 拼接引入接缝伪影这一额外域偏移,不可单独归因跨数据集泛化差距)。
    "$HF_CLI" download jonathan-roberts1/WHU-RS19 --repo-type dataset
fi

# ---- 2b. 外部 baseline 权重(v2.5 拍板②;按需开关,不进默认 ~60GB 路径)----
# BASELINE_A=1 : 平移+outpaint 平凡基线(档 A 必做)。SD-2-inpainting ~5GB,
#                非 gated,diffusers-native,主 env 直用。
if [ "${BASELINE_A:-0}" = "1" ]; then
    echo; echo "  [BASELINE_A] SD-2-inpainting(outpaint 基线)→ $HF_HOME/hub/"
    "$HF_CLI" download stabilityai/stable-diffusion-2-inpainting
fi

# MVSPLAT=1 : MVSplat360 对照(档 B,先冒烟 gate)。三权重 + 代码 + 第二
#             wheelhouse(torch2.2.2,与主 env 的 torch2.4+flash-attn 不兼容)。
#             ⚠️ svd.safetensors 是 GATED:须先在浏览器 Agree 并 huggingface-cli login。
if [ "${MVSPLAT:-0}" = "1" ]; then
    echo; echo "  [MVSPLAT] MVSplat360 权重/代码/wheelhouse → $STAGE/mvsplat360/"
    MVDIR="$STAGE/mvsplat360"; mkdir -p "$MVDIR/checkpoints"
    "$HF_CLI" download donydchen/mvsplat    re10k.ckpt       --local-dir "$MVDIR/checkpoints"
    "$HF_CLI" download donydchen/mvsplat360 dl3dv_480p.ckpt  --local-dir "$MVDIR/checkpoints"
    echo "  [MVSPLAT] svd.safetensors 为 GATED —— 未登录会失败(先 Agree + login)"
    "$HF_CLI" download stabilityai/stable-video-diffusion-img2vid \
        svd.safetensors --local-dir "$MVDIR/checkpoints" || {
        echo "  [WARN] SVD 下载失败(多半是未接受 gated 条款);MVSplat360 冒烟前必须补上"; }
    [ -d "$MVDIR/src" ] || git clone --depth 1 https://github.com/donydchen/mvsplat360.git "$MVDIR/src" || \
        echo "  [WARN] mvsplat360 代码 clone 失败(断网节点无法补,请在联网机重试)"
    WH2="$STAGE/wheelhouse_mvsplat"; mkdir -p "$WH2"
    pip download -d "$WH2" torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
        --index-url https://download.pytorch.org/whl/cu121 || echo "  [WARN] torch2.2.2 wheel 下载失败"
    pip download -d "$WH2" xformers==0.0.25.post1 \
        --index-url https://download.pytorch.org/whl/cu121 || true
    pip download -d "$WH2" h5py scipy scikit-image lpips opencv-python-headless || true
    [ -f "$MVDIR/src/requirements.txt" ] && pip download -d "$WH2" -r "$MVDIR/src/requirements.txt" || true
fi

# COGVIDEOX=1 : 跨 backbone 零样本 I2V(档 C,不阻塞冻结;camera-ready 加强项)。
if [ "${COGVIDEOX:-0}" = "1" ]; then
    echo; echo "  [COGVIDEOX] CogVideoX-5b-I2V(第二 backbone 零样本)→ $HF_HOME/hub/"
    "$HF_CLI" download THUDM/CogVideoX-5b-I2V
fi

# ---- 2c. [v2.7] 语义门 CLIP 权重(默认预取,~600MB)----
# P0 人检失败(裁剪后语义丢失)的修复依赖它:建库期拒绝采样门 + 第四道
# 复核门都用 CLIP 零样本评分。缺权重时建库【开工前】就硬错。
echo; echo "[2c] 语义门 CLIP openai/clip-vit-base-patch32 → $HF_HOME/hub/"
"$HF_CLI" download openai/clip-vit-base-patch32 ||     echo "  [WARN] CLIP 下载失败 —— 语义门无法运行,建库会硬错;换源重试"

# ---- 3. 评估预训练权重(直链,无需 torch) ----
echo; echo "[3/4] 评估权重 → $TORCH_HOME/hub/checkpoints/"
TC="$TORCH_HOME/hub/checkpoints"; mkdir -p "$TC"
for f in inception_v3_google-0cc3c7bd.pth r3d_18-b3b3357e.pth alexnet-owt-7be5be79.pth; do
    if [ -f "$TC/$f" ]; then echo "  [skip] $f"; continue; fi
    echo "  下载 $f"
    curl -fL --retry 3 "https://download.pytorch.org/models/$f" -o "$TC/$f.part"
    mv "$TC/$f.part" "$TC/$f"
done

# ---- 3b. 标准 I3D(FVD_I3D 离线重算;v2.6,审稿人小组必做项;I3D=1 开启) ----
# StyleGAN-V 发行的 i3d_torchscript.pt(~28MB,Kinetics-400 I3D,400-d logits):
# server_run/recompute_fvd_i3d.py 用它算与文献同口径的 FVD_I3D。
# [v2.6.1 审查修复 F2] 补上 I3D=1 开关 —— 文档四处写的都是 I3D=1,本段却曾
# 无条件执行(文档-代码漂移)。与 TIER2/BASELINE_A/MVSPLAT 同款按需开关。
# ⚠️ 两个候选 URL 在打包沙箱不可达(代理拦截),【未能实测 200】——首跑请在
#    联网机确认;都失败时用 I3D_URL=<直链> 覆盖。下载失败只 WARN 不中断
#    (训练不依赖它;recompute_fvd_i3d.py 缺文件会自行硬错)。
I3D_DIR="$STAGE/i3d"
if [ "${I3D:-0}" != "1" ]; then
    echo; echo "  [skip] I3D 权重(FVD_I3D 重算需要时: I3D=1 bash server_run/offline_prefetch.sh)"
elif mkdir -p "$I3D_DIR" && [ -f "$I3D_DIR/i3d_torchscript.pt" ]; then
    echo "  [skip] i3d_torchscript.pt"
else
    _I3D_OK=0
    for u in "${I3D_URL:-}" \
             "https://www.dropbox.com/s/ge9s5ujmvfslx3n/i3d_torchscript.pt?dl=1" \
             "https://huggingface.co/camenduru/FVD/resolve/main/i3d_torchscript.pt"; do
        [ -n "$u" ] || continue
        echo "  尝试 i3d_torchscript.pt ← $u"
        if curl -fL --retry 3 "$u" -o "$I3D_DIR/i3d_torchscript.pt.part"; then
            mv "$I3D_DIR/i3d_torchscript.pt.part" "$I3D_DIR/i3d_torchscript.pt"
            echo "  [ok] i3d_torchscript.pt  sha256=$(sha256sum "$I3D_DIR/i3d_torchscript.pt" | cut -c1-16)…  $(du -h "$I3D_DIR/i3d_torchscript.pt" | cut -f1)"
            echo "       (~28MB 量级;明显偏小说明拿到的是 HTML 错误页,删掉重试)"
            _I3D_OK=1; break
        fi
    done
    [ "$_I3D_OK" = "1" ] || echo "  [WARN] i3d_torchscript.pt 全部候选 URL 失败 —— FVD_I3D 重算前必须手动补上(I3D_URL=<直链> 重跑本脚本)"
fi

# ---- 4. 可选:pip wheelhouse(节点完全无 pypi 时) ----
if [ "${WHEELS:-0}" = "1" ]; then
    echo; echo "[4/4] wheelhouse → $STAGE/wheelhouse"
    WHEEL_PY="${WHEEL_PY:-python3}"
    PYV="$($WHEEL_PY -c 'import sys; print(sys.version.split()[0])')"
    if [ "$PYV" != "3.10.20" ]; then
        echo "FATAL: WHEELS=1 必须由 Python 3.10.20 下载训练环境 wheel；当前 $WHEEL_PY=$PYV。" >&2
        echo "       请设置 WHEEL_PY=<已验证训练环境>/bin/python 后重试。" >&2
        exit 2
    fi
    W="$STAGE/wheelhouse"; mkdir -p "$W"
    "$WHEEL_PY" -m pip download -d "$W" "torch==2.4.0" "torchvision==0.19.0" \
        --index-url https://download.pytorch.org/whl/cu121
    "$WHEEL_PY" -m pip download -d "$W" -r "$HERE/../lora_remotesense/new/requirements.txt"
    "$WHEEL_PY" -m pip download -d "$W" pip wheel setuptools
    FA_WHL="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
    if [ ! -f "$W/$FA_WHL" ]; then
        curl -fL --retry 3 -o "$W/$FA_WHL" \
          "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/$FA_WHL"
    fi
else
    echo; echo "[4/4] 跳过 wheelhouse(节点能连 pypi 时不需要;需要则 WHEELS=1 重跑本脚本,已下项自动跳过)"
fi

# ---- 汇总 + 节点侧指引 ----
echo
echo "=================================================================="
echo " 预取完成 ✓  目录大小:"
du -sh "$STAGE"/* 2>/dev/null
cat <<'EOF'

── 拷到计算节点(scp/rsync/移动硬盘均可)──────────────────────────
  STAGE/lingbot-world-base-cam/  →  /nix/rsroll_25828/models/lingbot-world-base-cam/
  HF_HOME                         = /nix/rsroll_25828/cache/huggingface/
  TORCH_HOME                      = /nix/rsroll_25828/cache/torch/
  STAGE/i3d/                     →  节点 $ASSET_ROOT/i3d/  (FVD_I3D 重算权重)
  STAGE/wheelhouse/              →  /nix/rsroll_25828/wheelhouse/ (若有)

── 计算节点上,每次跑之前(建议写进 ~/.bashrc)────────────────────
  export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  source server_run/a800_env.sh
  export CHECKPOINT_DIR=/nix/rsroll_25828/models/lingbot-world-base-cam

── 节点完全无 pypi 时,离线装环境───────────────────────────────
  export PIP_NO_INDEX=1 PIP_FIND_LINKS=/nix/rsroll_25828/wheelhouse
  cp /nix/rsroll_25828/wheelhouse/flash_attn-*.whl <包根>/lora_remotesense/new/
  cd <包根>/lora_remotesense/new && bash setup_env.sh

之后照常: cd <包根> && bash server_run/run_all.sh
外部 baseline 预取开关(v2.5 拍板②,默认关;按需单独跑):
  BASELINE_A=1 bash server_run/offline_prefetch.sh   # SD-2-inpainting ~5GB(必做)
  MVSPLAT=1    bash server_run/offline_prefetch.sh   # MVSplat360 权重+代码+torch2.2.2 wheelhouse
  COGVIDEOX=1  bash server_run/offline_prefetch.sh   # CogVideoX-5b-I2V ~11-20GB(camera-ready)
  拷贝:STAGE/mvsplat360/ → 节点 $MVSPLAT_DIR;wheelhouse_mvsplat/ 用
       PIP_NO_INDEX=1 PIP_FIND_LINKS=... 装进独立 torch2.2.2 conda env。
(MVSplat360 的 svd.safetensors 是 GATED:联网机先浏览器 Agree + huggingface-cli login)
EOF
