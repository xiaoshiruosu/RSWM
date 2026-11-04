# RSVideoWise 指南

> **部署硬件状态**：Primary validated hardware = **NVIDIA A800-SXM4-80GB**
>（sm80、BF16、FA2→SDPA）；H200 141GB 为 legacy/reference，sm90/FA3 路径保留但本轮未实机回归。

遥感视频外推基准数据集构建工具，扩展自 RemoteBAGEL 论文中的 RSWISE，专注于任意方向连续外推任务。本目录同时提供 **Wan2.2 LoRA 微调与推理** 脚本，支持遥感等垂直领域适配。

## 📋 项目概述

RSVideoWise Benchmark 用于构建遥感视频外推任务的基准数据集，核心特点：

- **中心序列 (V_c)**: 从中心网格提取的 8-16 帧时序序列
- **方向嵌入 (D)**: 支持任意角度 (0-360°) 的连续 sin/cos 编码
- **目标序列 (V_t)**: 沿指定方向的 8-16 帧序列，与中心序列重叠率 <10%
- **场景类别**: general (通用)、flood (洪水)、urban (城市)、rural (农村)
- **数据规模**: 共 2000 个实例 (每类场景 500 个)

**LoRA 微调与推理**（同目录）：

- **训练**: `train_lora.py` + `train_config.yaml`，支持 Multi-Direction Benchmark（v2/v5 数据格式兼容）、单卡与多卡 DDP、BF16
- **多卡训练**: 使用 `run_distributed.sh` 或 `torchrun --nproc_per_node=N train_lora.py ...`，脚本会自动根据 `LOCAL_RANK` 分配 GPU
- **推理**: `inference_with_lora.py` — **精度**：模型与 VAE 统一 BF16，避免黑色菱形网格；**权重比例**：`--lora_scale`（默认 0.8）调节 LoRA 强度、缓解过拟合噪点
- **遥感专项**: 详见 [遥感垂直领域微调指南 (REMOTE_SENSING_LORA_GUIDE.md)](./REMOTE_SENSING_LORA_GUIDE.md)

## 🛠️ 安装依赖

```bash
# 基础依赖
pip install numpy h5py tqdm

# 可选依赖 (用于 HuggingFace 数据源)
pip install datasets torch opencv-python scipy
```

## 📡 支持的 HuggingFace 数据集

以下数据集已验证可用 (2024年):

| 数据集 | 描述 | 样本数 |
|--------|------|--------|
| `timm/eurosat-rgb` | EuroSAT RGB Sentinel-2 卫星图像 | 27,000 |
| `blanchon/EuroSAT_RGB` | EuroSAT RGB 备用源 | 27,000 |
| `blanchon/UC_Merced` | UC Merced 土地利用数据集 | 2,100 |
| `GFM-Bench/BigEarthNet` | BigEarthNet Sentinel-1/2 多光谱 | 519,284 |
| `AdaptLLM/remote-sensing-visual-instructions` | 遥感视觉指令数据 | 36,400 |

> **注意**: 如果 HuggingFace 数据集不可用，系统会自动回退到合成数据生成模式。

## 📁 项目结构

```
remotesensebenchmark/
├── README.md                          # 本指南
├── build_rsvideowise_benchmark.py      # RSVideoWise 独立运行脚本 (推荐)
├── build_multi_direction_benchmark_*.py # Multi-Direction Benchmark 构建
├── train_lora.py                      # LoRA 微调脚本 (单卡/多卡 DDP)
├── run_distributed.sh                 # 多卡训练启动脚本 (需配置路径与 NUM_GPUS)
├── dataset_loader.py                  # 数据加载 (支持 v2/v5 benchmark 格式)
├── train_config.yaml                  # 训练配置 (含 noise_offset、遥感推荐)
├── inference_with_lora.py             # LoRA 推理脚本 (含 --lora_scale)
├── REMOTE_SENSING_LORA_GUIDE.md       # 遥感垂直领域微调指南
├── LORA_TRAINING_README.md            # LoRA 训练详细说明
├── QUICKSTART_LORA.md                 # LoRA 快速开始
├── rsvideowise_benchmark.h5           # 生成的数据集文件
└── benchmark/                         # 模块化实现
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── grid_partition.py
    ├── direction_embed.py
    ├── target_extract.py
    ├── validation.py
    ├── builder.py
    └── run_benchmark.py
```

## 🚀 快速开始

### 方法一：使用独立脚本 (推荐)

```bash
# 完整运行 (2000 实例，每场景 500 个)
python build_rsvideowise_benchmark.py

# 快速测试 (200 实例，每场景 50 个)
python build_rsvideowise_benchmark.py --instances 50

# 自定义输出路径
python build_rsvideowise_benchmark.py --output my_benchmark.h5

# 设置随机种子
python build_rsvideowise_benchmark.py --seed 123
```

### 方法二：使用模块化接口

```bash
python -m benchmark.run_benchmark --instances 500 --output benchmark_output/rsvideowise_benchmark.h5
```

### 方法三：Python API 调用

```python
from build_rsvideowise_benchmark import Config, RSVideoWiseBenchmarkBuilder

# 创建配置
config = Config()
config.instances_per_scene = 100  # 每场景实例数
config.output_path = "my_benchmark.h5"

# 构建基准数据集
builder = RSVideoWiseBenchmarkBuilder(config)
stats = builder.build()

print(f"生成实例数: {stats['total_instances']}")
print(f"平均重叠率: {np.mean(stats['overlap_ratios']):.2%}")
```

## 🔧 LoRA 微调与推理

### 训练：单卡与多卡

**单卡训练**（在项目根目录或本目录执行）：

```bash
# 在项目根目录
python remotesensebenchmark/train_lora.py \
  --model_config ti2v-5B \
  --checkpoint_dir /path/to/Wan2.2-TI2V-5B \
  --data_path /path/to/benchmark.h5 \
  --output_dir ./lora_checkpoints \
  --num_epochs 5

# 在本目录 (remotesensebenchmark/)
python train_lora.py --model_config ti2v-5B --checkpoint_dir ... --data_path ... --output_dir ...
```

**多卡训练（DDP）**：使用 `run_distributed.sh` 或直接 `torchrun`。请根据机器修改脚本内 `DATA_PATH`、`MODEL_PATH`、`OUTPUT_DIR` 和 `NUM_GPUS`。

```bash
# 使用脚本（需先编辑脚本中的路径与 GPU 数量）
chmod +x remotesensebenchmark/run_distributed.sh
./remotesensebenchmark/run_distributed.sh

# 或直接使用 torchrun（示例：3 卡）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=3 --master_port=29500 \
  remotesensebenchmark/train_lora.py \
  --model_config ti2v-5B \
  --checkpoint_dir /path/to/Wan2.2-TI2V-5B \
  --data_path /path/to/benchmark.h5 \
  --output_dir ./lora_checkpoints \
  --num_epochs 10 --num_workers 0
```

- 数据格式：`dataset_loader` 同时支持 **v2**（`start_frame`/`end_frame`）与 **v5**（`center_frame`/`target_frame`）生成的 benchmark.h5。
- 更多训练参数与说明见 [LORA_TRAINING_README.md](./LORA_TRAINING_README.md)。

### 文档入口

| 文档 | 说明 |
|------|------|
| [QUICKSTART_LORA.md](./QUICKSTART_LORA.md) | 5 分钟快速开始：数据生成 → 训练 → 推理 |
| [LORA_TRAINING_README.md](./LORA_TRAINING_README.md) | LoRA 训练详解：环境、数据、配置、checkpoint |
| [REMOTE_SENSING_LORA_GUIDE.md](./REMOTE_SENSING_LORA_GUIDE.md) | 遥感垂直领域：MoE 双专家、Rank/Alpha、noise_offset、VAE、8 像素对齐 |

### 推理说明：精度与权重比例

本仓库的推理管线（`inference_with_lora.py` 及底层 Wan pipeline）已做两项与画质相关的设置，无需额外配置即可生效：

| 项目 | 说明 |
|------|------|
| **精度调整** | 推理时**模型与 VAE 统一使用 bfloat16 (BF16)**，避免 FP16 数值溢出导致的**黑色菱形网格 (Checkerboard Artifacts)**。VAE 在加载时先设精度再上设备，与 `generate.py` 行为一致。 |
| **LoRA 权重比例** | 通过 **`--lora_scale`**（默认 `0.8`）调节 LoRA 强度，减轻过拟合带来的噪点；详见下方「推理脚本：LoRA 权重缩放」。 |

### 推理脚本：LoRA 权重缩放

使用 `inference_with_lora.py` 时，可通过 **`--lora_scale`** 调节 LoRA 强度，缓解过拟合导致的噪点（默认 `0.8`）：

```bash
# 默认缩放 0.8（减弱过拟合噪点）
python remotesensebenchmark/inference_with_lora.py \
  --checkpoint_dir ./Wan2.2-TI2V-5B \
  --lora_checkpoint ./lora_checkpoints/best_checkpoint \
  --prompt "your prompt" --output out.mp4

# 自定义缩放（如 0.6 进一步减弱）
python remotesensebenchmark/inference_with_lora.py \
  --lora_scale 0.6 \
  --checkpoint_dir ./Wan2.2-TI2V-5B \
  --lora_checkpoint ./lora_checkpoints/best_checkpoint \
  --prompt "your prompt" --output out.mp4

# 不缩放（保持原始 LoRA 强度）
python remotesensebenchmark/inference_with_lora.py --lora_scale 1.0 ...
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--lora_scale` | float | 0.8 | LoRA 权重缩放因子，用于减轻过拟合噪点 |

更多推理参数见脚本内 `--help`；训练参数与遥感推荐见 [LORA_TRAINING_README.md](./LORA_TRAINING_README.md) 与 [REMOTE_SENSING_LORA_GUIDE.md](./REMOTE_SENSING_LORA_GUIDE.md)。

---

## ⚙️ 命令行参数（RSVideoWise 构建）

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | `-o` | `rsvideowise_benchmark.h5` | 输出文件路径 |
| `--instances` | `-n` | `500` | 每个场景类别的实例数 |
| `--seed` | `-s` | `42` | 随机种子 |

## 📊 输出格式 (HDF5)

```
rsvideowise_benchmark.h5
├── metadata/                    # 元数据
│   ├── version                 # 版本号
│   ├── total_instances         # 总实例数
│   ├── scene_categories        # 场景类别列表
│   ├── max_overlap_ratio       # 最大重叠率阈值
│   └── creation_time           # 创建时间
├── data/                       # 数据
│   ├── sample_00000/
│   │   ├── center_sequence     # 中心序列 (T, H, W, C)
│   │   ├── target_sequence     # 目标序列 (T, H, W, C)
│   │   ├── direction_embedding # 方向嵌入 (64,)
│   │   └── [attrs]            # 属性 (角度、重叠率等)
│   ├── sample_00001/
│   └── ...
└── index/                      # 场景索引
    ├── general                 # 通用场景样本索引
    ├── flood                   # 洪水场景样本索引
    ├── urban                   # 城市场景样本索引
    └── rural                   # 农村场景样本索引
```

## 📖 读取数据示例

```python
import h5py
import numpy as np

with h5py.File('rsvideowise_benchmark.h5', 'r') as f:
    # 读取元数据
    total = f['metadata'].attrs['total_instances']
    print(f"总实例数: {total}")
    
    # 读取单个样本
    sample = f['data/sample_00000']
    center_seq = sample['center_sequence'][:]      # (T, H, W, C)
    target_seq = sample['target_sequence'][:]      # (T, H, W, C)
    direction_embed = sample['direction_embedding'][:]  # (64,)
    
    angle = sample.attrs['direction_angle']
    overlap = sample.attrs['overlap_ratio']
    scene = sample.attrs['scene_category']
    
    print(f"场景: {scene}, 角度: {angle:.1f}°, 重叠率: {overlap:.2%}")
    
    # 按场景类别筛选
    flood_indices = f['index/flood'][:]
    print(f"洪水场景样本数: {len(flood_indices)}")
```

## 📈 预期输出

运行完成后会显示统计信息：

```
======================================================================
RSVideoWise Benchmark Statistics
======================================================================
Total instances: 2000
Valid instances (<10% overlap): 2000
Invalid instances: 0

Overlap Statistics:
  - Average: 4.52%
  - Min: 0.00%
  - Max: 9.87%

Instances per scene:
  - general: 500
  - flood: 500
  - urban: 500
  - rural: 500

Processing time: 45.32 seconds
======================================================================

Benchmark built successfully!
Output: rsvideowise_benchmark.h5
```

## 🔧 配置说明

主要配置参数 (在 `Config` 类中):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `instances_per_scene` | 500 | 每场景实例数 |
| `min_frames` | 8 | 最小帧数 |
| `max_frames` | 16 | 最大帧数 |
| `image_size` | (512, 512) | 图像尺寸 |
| `embed_dim` | 64 | 方向嵌入维度 |
| `max_overlap_ratio` | 0.10 | 最大重叠率 |

## 📝 License

MIT License
