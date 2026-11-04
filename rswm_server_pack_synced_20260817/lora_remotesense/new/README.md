# RSWM A800/A100 80GB 训练工作指南（中文）

> **Primary validated hardware：NVIDIA A800-SXM4-80GB**（sm80、BF16、FA2→SDPA）。
> H200 141GB 是保留的 legacy/reference 路径（sm90、可选 FA3），本轮未做 H200 实机回归。

> **2026-08-14 共享 A800 更新**：服务器运行的权威步骤见
> `../../server_run/A800_80G.md`。当前 `setup_env.sh` 默认只核实固定环境，绝不自动升级；
> 下载使用独立的 `/nix/rsroll_25828/envs/hf311`。下文保留了独立打包说明，但凡与该
> A800 文档冲突，以 A800 文档、硬件 profile 和 preflight 为准。

> 本目录 (`new/`) 是一个**独立打包**，可直接 zip 上传到新服务器解压使用，
> 不需要再 clone 原仓库。`wan/` 基模代码与 `flash_attn` 预编译 wheel
> 都已经放在里面。

---

## 0. 这个包是用来做什么的

- **目标**：在 **80GB 显存**单卡 A800/A100 上跑 LingBot-World 双 14B 专家 + LoRA 微调。
- **正式协议**保持 `max_frames=33`；rank/alpha 由具体实验矩阵决定，硬件 profile 不改实验协议。
  只做工程层面的省显存改造，所以训练结果数字可以和论文已收的对齐。
- **包含**：训练 / 推理 / 评估 / 数据构建 / 基模代码 / 预编译 flash-attn wheel / 一键装环境脚本。
- **不包含**：基模权重（要从 HuggingFace 拉）、训练数据本身（用脚本现场重建）。

---

## 1. 工作流总览（一条路走完）

```
┌──────────────────────────────────────────────────────────────────────┐
│  step 0  zip 上传 new/ → 解压                                          │
│  step 1  bash setup_env.sh        装 PyTorch + 全部依赖 + flash-attn   │
│  step 2  huggingface-cli download  下载 14B 基模 checkpoint           │
│  step 3  bash run_build_dataset.sh 从 RemoteSense 现场生成 HDF5 数据    │
│  step 4  bash run_train.sh         LoRA 训练（默认低显存预设）          │
│  step 5  bash run_inference.sh     单图→视频，快速看一眼                │
│  step 6  bash run_generate_samples.sh 批量出 N 个 mp4 样例（含 GT 对照） │
│  step 7  bash run_evaluate.sh      LoRA / Base 评估 + 指标对齐         │
└──────────────────────────────────────────────────────────────────────┘
```

后面每步都有具体命令。如果你只想最快跑通，复制各步骤里第一个命令块就行。

---

## 2. 前置条件检查

跑之前先看一遍：

| 项目 | 要求 | 自查命令 |
| --- | --- | --- |
| Python | 3.10.20（训练环境） | `python -V` |
| CUDA | torch cu121；已验证 Driver 535.261.03（nvidia-smi 显示 CUDA 12.2） | `nvidia-smi` 与 `python -c "import torch; print(torch.__version__, torch.version.cuda)"` |
| GPU 显存 | Primary：A800-SXM4-80GB；Legacy/reference：H200 141GB | `nvidia-smi --query-gpu=memory.total --format=csv` |
| 系统内存 | A800 建议 ≥128GB；当前服务器为 1TiB | `free -h` |
| 磁盘 | ≥ 80GB（基模 ~56GB + 数据集 + checkpoint） | `df -h` |
| 网络 | 能访问 pypi / huggingface | `curl -I https://huggingface.co` |

> **AutoDL 用户**：选 80GB A100 / A800 卡，CUDA 12.1 镜像，Python 3.10 conda 环境。

---

## 3. 详细步骤

### Step 0 — 打包 / 上传 / 解压

**在你本地打包**（在 `lora_remotesense/` 上一层执行）：

```bash
cd <本地工程根>/lora_remotesense
zip -r new.zip new/ -x "*/__pycache__/*" "*.pyc"
ls -lh new.zip   # 应该 ~184MB（主要是 flash_attn .whl）
```

**上传**（scp / AutoDL 文件中转 / rsync 任选）：

```bash
scp new.zip user@<新服务器>:~/
```

**远端解压**：

```bash
mkdir -p ~/rswm && cd ~/rswm
unzip ~/new.zip
cd new
ls wan/                  # 应该看到 configs/ modules/ utils/ ...
ls flash_attn-*.whl      # 应该看到 .whl 文件
```

> 解压后 **当前目录 (`new/`) 就是工程根**，所有脚本通过 `sys.path` 在 `new/` 与 `new/wan/` 中自动找模块，
> 不需要 `pip install -e .`、不需要再 clone 原仓库。

---

### Step 1 — 装环境

**默认：核实已验证的 Python 3.10.20 训练环境（不安装、不升级）**

```bash
cd <包根>/lora_remotesense/new
bash setup_env.sh
```

只有明确需要重建环境时，才允许显式开启写操作：

```bash
RSWM_ALLOW_INSTALL=1 CREATE_CONDA=1 ENV_PREFIX=/nix/rsroll_25828/envs/rswm-a800 bash setup_env.sh
```

脚本会核实 Python、torch/torchvision CUDA build、flash-attn 和 HF/PEFT 栈，并运行
`pip check`。它不会静默编译 flash-attn；若 2.7.3 wheel 缺失或 ABI 不匹配会直接失败。

**成功的标志**（末尾 5 行类似）：

```
python : 3.10.x
torch  : 2.4.0+cu121 cuda? True
device : NVIDIA H100 96GB
vram   : 96.0 GB
flash_attn: 2.7.3
bitsandbytes: 0.43.x
wan import: OK, configs = ['i2v-A14B', ...]
dataset_loader import: OK
```

若有任意一行报 `FAILED`，跳到 §7 故障排查。

---

### Step 2 — 下载基模 checkpoint

```bash
# 推荐：huggingface CLI（要先有 HF 账户、模型权限）
huggingface-cli login                  # 第一次需要，粘贴 token
huggingface-cli download robbyant/lingbot-world-base-cam \
    --local-dir ~/rswm/lingbot-world-base-cam

# 记下路径，后面所有命令都要它
export CHECKPOINT_DIR=/nix/rsroll_25828/models/lingbot-world-base-cam
```

下载 ~56GB，慢的话可以挂 `--resume-download`。

**确认目录结构**：

```bash
ls $CHECKPOINT_DIR
# 应该看到:
#   low_noise_model/
#   high_noise_model/
#   vae/...  或  Wan2.1_VAE.pth
#   t5/...   或  models_t5_umt5-xxl-enc-bf16.pth
#   google/umt5-xxl/   (tokenizer)
```

---

### Step 3 — 重建数据集

```bash
cd ~/rswm/new

# 默认：100 样本，落到 data/remotesense/train.h5
bash run_build_dataset.sh

# 自定义：500 样本，输出位置自选
RS_INSTANCES=500 DATA_PATH=/nix/rsroll_25828/data/train_500.h5 bash run_build_dataset.sh
```

脚本会：

1. 调 `build_from_remotesense.py`，从 HuggingFace 拉 RS 数据（首次会下载，后续走缓存）
2. 生成 RemoteSense benchmark.h5（临时）
3. 转成 LingBot HDF5 格式（包含 image / video_frames / intrinsics / poses / prompt）
4. 自动调 `verify_remotesense_h5.py` 检查结构

**离线生成（无网，用合成数据）**：

```bash
python build_from_remotesense.py --no-hf --instances 100 -o data/train.h5
```

---

### Step 4 — 训练

```bash
cd ~/rswm/new

# 默认（推荐）：OFFLOAD=auto；A800 80GB 自动选择单专家 offload + torch.AdamW
CHECKPOINT_DIR=/nix/rsroll_25828/models/lingbot-world-base-cam \
DATA_PATH=/nix/rsroll_25828/data/remotesense/train.h5 \
    bash run_train.sh
```

默认配置已经把 A800/A100 80GB 必要的省显存开关都打开了：

- `--hardware_profile a800_80g`：核实 sm80 + 80GB，并使用 BF16 安全路径
- `--low_mem`：T5/VAE 默认在 CPU，调用时再上 GPU；GC ON；正式入口 grad_accum=4
- `OFFLOAD=auto`：A800 自动解析为 `1`，同一时刻只让 1 份 14B DiT 驻 GPU
- `--no_8bit_adam`：offload 路径固定 `torch.AdamW`；LoRA 梯度/state 随专家同步换入换出
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 默认开
- `compute_loss` 内主动 `del + empty_cache`

**显存大就关掉 offload 加速**（≥96GB H100/H200 用户）：

```bash
HARDWARE_PROFILE=h200_141g OFFLOAD=auto \
CHECKPOINT_DIR=$CHECKPOINT_DIR DATA_PATH=$DATA_PATH \
    bash run_train.sh
```

关掉后两份 14B 同时驻 GPU，省去 CPU↔GPU 来回搬运，约快 30–50%。

**自定义超参**：

```bash
NUM_EPOCHS=5 ACCUM_STEPS=4 \
LORA_RANK=64 LORA_ALPHA=64 MAX_FRAMES=33 \
OUTDIR=/nix/rsroll_25828/work/outputs/exp1 \
DATA_PATH=/nix/rsroll_25828/data/train.h5 \
CHECKPOINT_DIR=$CHECKPOINT_DIR \
    bash run_train.sh
```

**输出**：

```
$OUTDIR/
├── train_val_split.json     # 验证集 indices，评估时要用
├── checkpoint_epoch_*/
├── best_checkpoint/         # val loss 最低的
└── training.log
```

**显存组成（A800 80GB，默认 `OFFLOAD=auto→1`, B=1）**：

| 组件 | 占用 |
| --- | --- |
| 1 份 14B DiT (bf16, 当前 timestep 对应那份驻 GPU) | ~28 GB |
| LoRA grads + torch AdamW state | 随 rank/targets 变化，必须实测 |
| Activations (GC on, T=33) | 随分辨率/后端变化，必须实测 |
| 临时缓冲（VAE/T5 swap） | ~3–5 GB |
| **合计** | **由 Level-3 与真实 33 帧 smoke 记录，不预先声称** |

真实峰值尚未在本机验证，必须由 A800 Level-3 与真实 33 帧 smoke 记录；不要把组成项预算当成实测值。

> **显式吞吐测试**：传 `OFFLOAD=0` 可尝试双专家常驻。A800 80GB 很可能 OOM；
> 只有真实 smoke 通过后才能讨论吞吐，不能直接用于正式矩阵。H200 `auto` 在空闲显存足够时会解析为 0。

---

### Step 5 — 单图推理（看效果）

适合手头有一张自己的图想快速试一下。A800 默认使用 T5 CPU 与安全 offload；峰值必须实测。

```bash
LORA_CKPT=~/outputs/exp1/best_checkpoint \
IMAGE=~/rswm/new/examples/00/image.jpg \
PROMPT="A satellite video of a water-dominated remote-sensing scene. The camera follows a 2-segment trajectory with net displacement toward the east." \
CHECKPOINT_DIR=$CHECKPOINT_DIR \
    bash run_inference.sh
```

输出 mp4 / npy 在 `outputs/inference/` 下。

> 显存大就关掉 offload 加速：`python inference.py ... --no_inactive_model_cpu --no_t5_cpu`

---

### Step 6 — 批量生成视频样例（推荐）

训练完最常用的一步：从训练数据 HDF5 里抽 N 个样本一次性都生成 mp4，
**同时落 GT 视频在旁边对照**。模型只加载一次，比循环跑 `run_inference.sh` 快 N 倍。

**最常见用法 — 随机抽 8 个验证样本**：

```bash
LORA_CKPT=~/outputs/exp1/best_checkpoint \
DATA_PATH=/nix/rsroll_25828/data/train.h5 \
CHECKPOINT_DIR=$CHECKPOINT_DIR \
    bash run_generate_samples.sh
```

**按场景过滤**（论文 per-scene 对比时用）：

```bash
LORA_CKPT=~/outputs/exp1/best_checkpoint \
    SCENE=urban NUM_SAMPLES=4 \
    bash run_generate_samples.sh
# SCENE 可选: general / flood / urban / rural
```

**指定特定样本**（复现某个 case）：

```bash
LORA_CKPT=~/outputs/exp1/best_checkpoint \
    KEYS="sample_00012 sample_00037 sample_00091" \
    bash run_generate_samples.sh
```

**同一样本多 seed 出变体**（看生成多样性）：

```bash
for s in 42 123 999; do
    LORA_CKPT=~/outputs/exp1/best_checkpoint \
    KEYS="sample_00012" GEN_SEED=$s \
    OUTDIR=/nix/rsroll_25828/work/outputs/seed_$s bash run_generate_samples.sh
done
```

**输出结构**：

```
samples_<时间戳>/
├── generated/<key>.mp4     # LoRA 生成
├── gt/<key>.mp4            # ground truth 对照
├── meta/<key>.json         # 每个样本: prompt / scene / seed / 耗时
└── index.json              # 总索引
```

**断点续生成**：mp4 已存在自动跳过，半路 OOM 重启不重做。

**支持的环境变量**：`LORA_CKPT` `DATA_PATH` `OUTDIR` `NUM_SAMPLES`(默认 8) `SCENE` `KEYS` `SEED`(抽样种子) `GEN_SEED`(噪声种子) `FRAMES`(默认 33) `STEPS`(默认 70) `SHIFT` `GUIDE_SCALE` `DEVICE`。

---

### Step 7 — 评估

**LoRA 模型**：

```bash
LORA_CKPT=~/outputs/exp1/best_checkpoint \
DATA_PATH=/nix/rsroll_25828/data/train.h5 \
SPLIT_FILE=~/outputs/exp1/train_val_split.json \
OUTDIR=/nix/rsroll_25828/work/outputs/eval_lora \
CHECKPOINT_DIR=$CHECKPOINT_DIR \
    bash run_evaluate.sh
```

**Base 对照（无 LoRA）**：

```bash
python evaluate_baseline.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --data_path ~/data/train.h5 \
    --split_file ~/outputs/exp1/train_val_split.json \
    --output_dir ~/outputs/eval_base \
    --max_samples 100
```

**对齐到相同样本子集，重算指标**：

```bash
python eval_aligned.py \
    --lora_dir ~/outputs/eval_lora \
    --base_dir ~/outputs/eval_base \
    --output_dir ~/outputs/eval_aligned \
    --device cuda
```

输出：

```
eval_aligned/
├── aligned_keys.json
├── comparison.json
├── lora/{generated,gt,meta}/
└── base/{generated,gt,meta}/
```

控制台会打 SSIM / PSNR / LPIPS / FID / FVD / Angular_Error 的 LoRA vs Base 差异表。

**仅重算指标（npy 已存在）**：

```bash
METRICS_ONLY=1 OUTDIR=/nix/rsroll_25828/work/outputs/eval_lora bash run_evaluate.sh
```

---

## 4. 目录结构（zip 解压后）

```
new/
├── README.md                    # 你正在看的这份
├── setup_env.sh                 # 一键装环境
├── requirements.txt             # pip 依赖名单
├── flash_attn-2.7.3+...whl      # 预编译 flash-attn（CUDA 12.1 / torch 2.4 / py3.10）
│
├── train.py                     # 训练入口（80GB A100 适配）
├── run_train.sh                 # 训练 wrapper
├── pbs_train_96g.sh             # （PBS 集群用，已改为 80GB 默认；AutoDL 可忽略）
│
├── inference.py                 # 单图→视频推理
├── run_inference.sh
├── generate_samples.py          # 批量生成视频样例（训练完肉眼看效果用）
├── run_generate_samples.sh
│
├── evaluate.py                  # LoRA 评估 pipeline
├── evaluate_baseline.py         # Base 模型评估
├── eval_aligned.py              # 对齐子集重算指标
├── analyze_per_direction.py     # 按方向拆分
├── metrics.py                   # 指标实现
├── run_evaluate.sh
│
├── build_from_remotesense.py    # 从 RS 生成 HDF5
├── run_build_dataset.sh
├── verify_remotesense_h5.py     # HDF5 结构校验
├── check_dataset.py             # 数据统计/预览
├── extract_eval_subset.py       # 抽评估子集
│
├── dataset_loader.py            # LingBotDataset
├── camera_trajectory.py         # 相机轨迹生成
│
└── wan/                         # 基模 Python 包（必须随包带，不要删）
    ├── __init__.py
    ├── image2video.py
    ├── configs/
    ├── modules/                 # model / vae / t5 / attention / tokenizers
    ├── utils/
    ├── distributed/
    └── remotesensebenchmark/    # v5 benchmark builder
```

---

## 5. 环境变量速查（所有 run_*.sh 支持）

| 变量 | 默认 | 含义 |
| --- | --- | --- |
| `CHECKPOINT_DIR` | `./lingbot-world-base-cam` | 基模权重路径 |
| `DATA_PATH` | `data/remotesense/train.h5` | 训练 HDF5 |
| `OUTDIR` | `outputs/rs_lora_96g_<时间戳>` | 输出目录 |
| `NUM_EPOCHS` | 3 | epoch 数 |
| `ACCUM_STEPS` | 4 | 梯度累积 |
| `LORA_RANK` / `LORA_ALPHA` | 64 / 64 | LoRA 超参（**别动**，会偏离论文数字） |
| `MAX_FRAMES` | 33 | 正式序列长度（**别动**） |
| `OFFLOAD` | `auto` | `auto/0/1`；A800 auto→1，smoke 固定 1，显式 0 仅用于吞吐/OOM 边界测试 |
| `RS_INSTANCES` | 100 | 数据集样本数 |
| `LORA_CKPT` | — | 推理/评估时的 LoRA checkpoint |
| `SPLIT_FILE` | — | 评估时的 train_val_split.json |
| `METRICS_ONLY` | 0 | 设 1 → 跳过推理只算指标 |
| `T5_CPU` | 1 | T5 编码器是否在 CPU 上跑（省 ~10GB） |
| **批量生成专属** ↓ | | （`run_generate_samples.sh`） |
| `NUM_SAMPLES` | 8 | 从 HDF5 随机抽几个样本 |
| `SCENE` | — | 按场景过滤：general / flood / urban / rural |
| `KEYS` | — | 直接指定 sample key，如 `"sample_00012 sample_00037"` |
| `SEED` | 42 | 抽样种子（决定抽哪些 key） |
| `GEN_SEED` | 42 | 生成时的噪声种子（换它出多个变体） |
| `STEPS` | 70 | 采样步数 |
| `SHIFT` / `GUIDE_SCALE` | 10.0 / 5.0 | 调度器 shift 和 CFG 强度 |

---

## 6. 依赖名单一览

`requirements.txt` 涵盖以下分组（PyTorch 单独装）：

| 分组 | 包 |
| --- | --- |
| 基模运行 | diffusers / transformers / tokenizers / accelerate / safetensors / huggingface-hub / sentencepiece / protobuf / peft |
| 张量数值 | numpy<2 / scipy / einops |
| 图像视频 IO | Pillow / opencv-python / imageio / imageio-ffmpeg / scikit-image |
| 数据集 | h5py / datasets |
| 评估指标 | lpips / torchmetrics / pytorch-fid / scikit-learn |
| 训练辅助 | tqdm / ftfy / regex / easydict / matplotlib |
| 显存优化 | bitsandbytes |
| 注意力 | flash-attn 2.7.3（单独装，见 §3 Step 1） |

---

## 7. 故障排查

| 现象 | 处理 |
| --- | --- |
| `ModuleNotFoundError: bitsandbytes` | A800 固定环境使用 `torch.AdamW`，不要为此升级或改动已验证依赖；仅 legacy H200 8-bit 路径需要单独核实 bnb |
| `flash_attn` wheel 安装失败 | 停止并恢复已验证的 torch2.4/cu121、Python3.10、flash-attn2.7.3 wheel；不要在训练环境里临时升级或隐式源码编译 |
| `from wan.configs import WAN_CONFIGS` 失败 | 必须在 `new/` 目录下执行所有脚本；cd 出去后 `sys.path` 找不到本地 `wan/` |
| `huggingface-cli` 401/403 | 先 `huggingface-cli login`；模型若是私有，要在 HF 网站申请权限 |
| 训练 OOM | `OFFLOAD=auto` 在 A800 会启用安全路径；检查当前分配卡的空闲显存和 batch_size，不要终止他人进程 |
| 训练 CPU OOM（默认就开 inactive_cpu） | 单份 14B bf16 在 CPU ~28GB，另一份训练时也在 CPU；系统内存至少 80GB |
| 数据构建第一次很慢 | 下载使用独立 hf311 环境；缓存固定在 `/nix/rsroll_25828/cache/huggingface` |
| 评估时 `inception` / lpips 第一次下权重 | 默认从 HF/torch.hub 拉，离线环境提前 `pip install lpips && python -c "import lpips; lpips.LPIPS(net='alex')"` |
| `RuntimeError: CUDA out of memory ... 0 bytes free` | 极端碎片，重启 python 进程 + `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

---

## 8. 与原 143GB H200 版本的兼容性

- **checkpoint 格式一致**：`low_noise/`, `high_noise/`, `training_state.pt` 双向 resume 没问题
- **算法/超参未动**：训练数字应与原版对齐
- **`--inactive_model_cpu` 不改变 loss 计算路径**，只是把权重位置在 CPU/GPU 间来回挪

---

## 9. 环境重建（仅在明确授权时）

`new/` 目录下执行：

```bash
RSWM_ALLOW_INSTALL=1 bash setup_env.sh
# flash-attn 必须使用已经验证、与 torch2.4/cu121/Python3.10 ABI 一致的 2.7.3 wheel。
```

验证：

```bash
python -c "
import torch, flash_attn
from wan.configs import WAN_CONFIGS
from dataset_loader import LingBotDataset
print('OK')
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('flash_attn:', flash_attn.__version__)
print('configs:', list(WAN_CONFIGS.keys()))
"
```

---

## 10. 下一步建议

1. **首跑先用 demo 数据 + 1 epoch 验证端到端**：
   ```bash
   RS_INSTANCES=20 DATA_PATH=data/demo.h5 bash run_build_dataset.sh
   NUM_EPOCHS=1 ACCUM_STEPS=4 DATA_PATH=data/demo.h5 bash run_train.sh
   ```
   跑通后再换大数据集长跑。
2. **第一轮训练时另开终端 `watch -n 5 nvidia-smi`**，正常应在 55–65GB；若长期 >75GB 检查 batch 是否被改成 >1。
3. **训练完先 evaluate base + LoRA**，再 `eval_aligned.py` 对齐子集出对比表，是论文里要用的数。
