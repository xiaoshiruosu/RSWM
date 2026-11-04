# LoRA Fine-tuning for Wan2.2 with Multi-Direction Benchmark

本文档介绍如何使用 Multi-Direction Sliding Window Benchmark 数据集对 Wan2.2 视频生成模型进行 LoRA 微调。

## 📋 目录

- [概述](#概述)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [训练配置](#训练配置)
- [高级用法](#高级用法)
- [常见问题](#常见问题)

## 🎯 概述

### 什么是 LoRA？

LoRA (Low-Rank Adaptation) 是一种参数高效的微调方法，通过在预训练模型中添加低秩矩阵来适应新任务，而无需更新所有模型参数。

**优势：**
- ✅ 显著减少可训练参数（通常 <1% 的原始参数）
- ✅ 降低显存需求
- ✅ 加快训练速度
- ✅ 易于部署和共享

### 训练目标

使用 Multi-Direction Sliding Window Benchmark 数据集训练 Wan2.2 模型，使其能够：
1. 理解和生成多方向连续运动的视频
2. 根据方向嵌入控制视频中的运动方向
3. 生成具有复杂轨迹的遥感视频序列

## 🛠️ 环境准备

### 1. 安装依赖

```bash
# 基础依赖（如果已安装 Wan2.2，可跳过）
pip install -r requirements.txt

# LoRA 训练额外依赖
pip install peft wandb accelerate
```

### 2. 硬件要求

| 模型 | 最小显存 | 推荐显存 | 批次大小 |
|------|---------|---------|---------|
| TI2V-5B | 24GB (RTX 4090) | 40GB (A100) | 1-2 |
| T2V-A14B | 40GB (A100) | 80GB (A100) | 1-2 |
| I2V-A14B | 40GB (A100) | 80GB (A100) | 1-2 |

> 💡 **提示**: 使用 TI2V-5B 模型可以在消费级显卡（如 RTX 4090）上进行训练。

## 📊 数据准备

### 1. 生成 Benchmark 数据集

首先使用提供的脚本生成训练数据：

```bash
# 生成 Multi-Direction Benchmark 数据集
python remotesensebenchmark/build_multi_direction_benchmark_v2.py \
    --output multi_direction_benchmark_v2.h5 \
    --instances 100 \
    --min-segments 2 \
    --max-segments 4 \
    --steps 8
```

这将生成一个包含多方向滑动窗口序列的 HDF5 文件。

### 2. 验证数据集

```python
import h5py

with h5py.File('multi_direction_benchmark_v2.h5', 'r') as f:
    print(f"总样本数: {f['metadata'].attrs['total_instances']}")
    print(f"场景类别: {f['metadata'].attrs['scene_categories']}")
    print(f"窗口大小: {f['metadata'].attrs['window_size']}")
```

### 3. 训练/验证集划分

训练脚本会自动将数据集划分为训练集和验证集：

- **默认划分比例**: 80% 训练集，20% 验证集
- **可配置参数**: 
  - `--train_ratio`: 训练集比例（默认 0.8）
  - `--val_ratio`: 验证集比例（默认 0.2）
  - `--stratify_by`: 按元数据键进行分层划分（如 "scene_category"）

**示例**：
```bash
# 使用默认 80/20 划分
python train_lora.py --data_path ./data.h5 ...

# 自定义划分比例（70% 训练，30% 验证）
python train_lora.py \
    --train_ratio 0.7 \
    --val_ratio 0.3 \
    ...

# 按场景类别分层划分（确保每个场景的训练/验证比例一致）
python train_lora.py \
    --stratify_by scene_category \
    ...
```

## 🚀 快速开始

### 方法 1: 使用命令行参数

```bash
python remotesensebenchmark/train_lora.py \
    --model_config ti2v-5B \
    --checkpoint_dir ./Wan2.2-TI2V-5B \
    --data_path ./multi_direction_benchmark_v2.h5 \
    --output_dir ./lora_checkpoints \
    --batch_size 1 \
    --num_epochs 10 \
    --learning_rate 1e-4 \
    --lora_rank 128 \
    --lora_alpha 128
```

### 方法 2: 使用配置文件

1. 编辑 `train_config.yaml`:

```yaml
# 模型配置
model_config: "ti2v-5B"
checkpoint_dir: "./Wan2.2-TI2V-5B"

# 数据配置
data_path: "./multi_direction_benchmark_v2.h5"
train_ratio: 0.8  # 训练集比例
val_ratio: 0.2    # 验证集比例
stratify_by: null # 可选：按场景类别分层划分

# 训练超参数
batch_size: 1
num_epochs: 10
learning_rate: 1.0e-4
lora_rank: 128
lora_alpha: 128

# 输出配置
output_dir: "./lora_checkpoints"
save_interval: 1  # 每个 epoch 都保存
```

2. 运行训练（需要修改脚本以支持 YAML 配置）:

```bash
python remotesensebenchmark/train_lora.py --config train_config.yaml
```

> **注意**: 目前脚本使用命令行参数，YAML 配置支持需要额外实现。建议直接使用命令行参数。

### 方法 3: 使用 Python API

```python
from remotesensebenchmark.train_lora import LoRATrainer
import argparse

# 创建参数
args = argparse.Namespace(
    model_config='ti2v-5B',
    checkpoint_dir='./Wan2.2-TI2V-5B',
    data_path='./multi_direction_benchmark_v2.h5',
    output_dir='./lora_checkpoints',
    batch_size=1,
    num_epochs=10,
    learning_rate=1e-4,
    # ... 其他参数
)

# 初始化并训练
trainer = LoRATrainer(args)
trainer.train()
```

## ⚙️ 训练配置

### 核心参数说明

#### 模型配置
- `--model_config`: 选择模型类型
  - `ti2v-5B`: 5B 参数的文本/图像到视频模型（推荐用于入门）
  - `t2v-A14B`: 14B 参数的文本到视频 MoE 模型
  - `i2v-A14B`: 14B 参数的图像到视频 MoE 模型

#### LoRA 配置
- `--lora_rank`: LoRA 秩（默认 128）
  - 较小值（32-64）：参数更少，训练更快，但表达能力较弱
  - 较大值（128-256）：表达能力更强，但需要更多显存
- `--lora_alpha`: LoRA 缩放因子（通常设置为与 rank 相同）

#### 训练超参数
- `--batch_size`: 批次大小（根据显存调整）
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率（推荐 1e-4 到 1e-5）
- `--weight_decay`: 权重衰减（默认 0.01）

#### 数据集划分参数
- `--train_ratio`: 训练集比例（默认 0.8）
- `--val_ratio`: 验证集比例（默认 0.2）
- `--stratify_by`: 分层划分的元数据键（如 "scene_category"，默认 None）

### 推荐配置

#### RTX 4090 (24GB)
```bash
python train_lora.py \
    --model_config ti2v-5B \
    --batch_size 1 \
    --lora_rank 64 \
    --learning_rate 1e-4
```

#### A100 (40GB)
```bash
python train_lora.py \
    --model_config ti2v-5B \
    --batch_size 2 \
    --lora_rank 128 \
    --learning_rate 1e-4
```

#### A100 (80GB)
```bash
python train_lora.py \
    --model_config t2v-A14B \
    --batch_size 2 \
    --lora_rank 128 \
    --learning_rate 5e-5
```

## 🔬 高级用法

### 1. 使用 Weights & Biases 监控训练

```bash
python train_lora.py \
    --use_wandb \
    --wandb_project wan2.2-lora \
    --wandb_run_name my-experiment \
    ... # 其他参数
```

### 2. 筛选特定场景类别

```bash
python train_lora.py \
    --scene_categories general urban \
    ... # 其他参数
```

### 3. 多 GPU 训练（DDP）

`train_lora.py` 支持通过 `torchrun` 启动多卡 DDP 训练，会自动根据环境变量 `LOCAL_RANK` 分配 GPU，使用 `DistributedSampler` 划分数据，仅 rank 0 保存 checkpoint 和日志。

**方式一：使用脚本**（需先编辑脚本中的 `DATA_PATH`、`MODEL_PATH`、`OUTPUT_DIR`、`NUM_GPUS`）：

```bash
chmod +x remotesensebenchmark/run_distributed.sh
./remotesensebenchmark/run_distributed.sh
```

**方式二：直接使用 torchrun**（示例：3 卡）：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node=3 --master_port=29500 \
  remotesensebenchmark/train_lora.py \
  --model_config ti2v-5B \
  --checkpoint_dir /path/to/Wan2.2-TI2V-5B \
  --data_path /path/to/benchmark.h5 \
  --output_dir ./lora_checkpoints \
  --batch_size 1 \
  --num_epochs 10 \
  --learning_rate 5e-5 \
  --lora_rank 64 \
  --lora_alpha 64 \
  --train_ratio 0.9 \
  --val_ratio 0.1 \
  --save_interval 1 \
  --num_workers 0
```

- 多卡时建议 `--num_workers 0`，避免 DDP + h5py 多进程读同一文件可能引起的问题。
- 数据格式：支持 **v2**（`start_frame`/`end_frame`）与 **v5**（`center_frame`/`target_frame`）生成的 benchmark.h5。

### 4. 恢复训练（待实现）

```bash
python train_lora.py \
    --resume_from ./lora_checkpoints/checkpoint-epoch-5 \
    ... # 其他参数
```

### 5. 自定义训练/验证集划分

```bash
# 使用 70/30 划分
python train_lora.py \
    --train_ratio 0.7 \
    --val_ratio 0.3 \
    ...

# 按场景类别分层划分（确保每个场景类别在训练和验证集中都有相同的比例）
python train_lora.py \
    --stratify_by scene_category \
    --train_ratio 0.8 \
    --val_ratio 0.2 \
    ...
```

## 📈 监控训练进度

训练过程中会输出以下信息：

```
Epoch 1/10: 100%|██████████| 400/400 [12:34<00:00, 1.88s/it, loss=0.1234, avg_loss=0.1456, lr=1.00e-04]
Validation 1: 100%|██████████| 100/100 [02:15<00:00, val_loss=0.1567, avg_val_loss=0.1623]
Epoch 1/10 - Train Loss: 0.1456, Val Loss: 0.1623
Checkpoint saved to ./lora_checkpoints/checkpoint-epoch-1_20250127_143022
New best model saved (val_loss: 0.1623)
```

关键指标：
- **loss**: 当前批次的训练损失
- **avg_loss**: 当前 epoch 的平均训练损失
- **val_loss**: 验证集损失
- **lr**: 当前学习率

### 验证集评估

每个训练 epoch 结束后，模型会在验证集上进行评估：
- 自动计算验证损失
- 跟踪最佳模型（基于验证损失）
- 将验证指标记录到 wandb（如启用）

## 💾 Checkpoint 保存机制

训练过程中，每个 epoch 都会保存完整的 checkpoint，包含：

### Checkpoint 目录结构

```
lora_checkpoints/
├── checkpoint-epoch-1_20250127_143022/
│   ├── adapter_model.bin              # LoRA 权重
│   ├── adapter_config.json            # LoRA 配置
│   ├── training_state.pt              # 训练状态（optimizer, scheduler等）
│   ├── training_config.json           # 完整训练配置和参数
│   └── original_train_config.yaml     # 原始配置文件副本
├── checkpoint-epoch-2_20250127_150315/
│   └── ...
├── best_checkpoint/                   # 最佳模型（验证损失最低）
│   └── ...
└── training_summary.json              # 训练总结
```

### 保存的内容

1. **LoRA 权重** (`adapter_model.bin`): 可直接用于推理
2. **训练状态** (`training_state.pt`): 包含
   - 当前 epoch
   - 训练和验证指标
   - Optimizer 状态
   - Scheduler 状态
   - 模型配置信息

3. **完整配置** (`training_config.json`): 包含
   - 所有训练超参数（学习率、批次大小等）
   - LoRA 配置（rank, alpha）
   - 数据配置（训练/验证集大小、划分比例等）
   - 模型配置
   - 设备配置
   - 训练指标历史

4. **训练总结** (`training_summary.json`): 包含
   - 最佳 epoch 和验证损失
   - 总训练时间
   - 最终训练/验证损失

### 查看训练配置

```python
import json

# 查看某个 checkpoint 的完整配置
with open('./lora_checkpoints/checkpoint-epoch-5_*/training_config.json', 'r') as f:
    config = json.load(f)
    print(f"训练样本数: {config['data_config']['train_samples']}")
    print(f"验证样本数: {config['data_config']['val_samples']}")
    print(f"学习率: {config['training_hyperparameters']['learning_rate']}")
    print(f"训练损失: {config['train_metrics']['loss']}")
    print(f"验证损失: {config['val_metrics']['loss']}")

# 查看训练总结
with open('./lora_checkpoints/training_summary.json', 'r') as f:
    summary = json.load(f)
    print(f"最佳 epoch: {summary['best_epoch']}")
    print(f"最佳验证损失: {summary['best_val_loss']}")
    print(f"总训练时间: {summary['total_training_time']:.2f} 秒")
```

## 🎬 使用训练好的 LoRA 模型

训练完成后，可以这样使用 LoRA 权重：

```python
from wan.textimage2video import WanTI2V
from peft import PeftModel
import torch

# 加载基础模型
wan_ti2v = WanTI2V(
    config=cfg,
    checkpoint_dir='./Wan2.2-TI2V-5B',
    device_id=0
)

# 加载 LoRA 权重（使用最佳模型）
lora_path = './lora_checkpoints/best_checkpoint'
wan_ti2v.model = PeftModel.from_pretrained(
    wan_ti2v.model,
    lora_path
)

# 或者加载特定 epoch 的模型
# lora_path = './lora_checkpoints/checkpoint-epoch-10_20250127_150315'

# 生成视频
video = wan_ti2v.generate(
    prompt="A satellite view moving northeast then turning south",
    size=(1280, 704),
    frame_num=81
)
```

## ❓ 常见问题

### Q1: 显存不足怎么办？

**A**: 尝试以下方法：
1. 减小 `batch_size` 到 1
2. 降低 `lora_rank` (如 64 或 32)
3. 使用更小的模型 (ti2v-5B)
4. 启用梯度检查点（需要修改代码）

### Q2: 训练损失不下降？

**A**: 检查：
1. 学习率是否过大或过小（尝试 1e-4 到 1e-5）
2. 数据是否正确加载
3. 是否需要更多训练轮数
4. LoRA rank 是否足够

### Q3: 如何选择 LoRA rank？

**A**: 
- 简单任务（微调风格）: rank 32-64
- 中等任务（新场景适应）: rank 64-128
- 复杂任务（新能力学习）: rank 128-256

### Q4: 训练需要多长时间？

**A**: 取决于：
- 数据集大小（100-1000 样本）
- 硬件配置（RTX 4090 vs A100）
- 批次大小和 epoch 数

典型时间：
- RTX 4090 + 400 样本 + 10 epochs ≈ 2-4 小时
- A100 + 400 样本 + 10 epochs ≈ 1-2 小时

## 📚 参考资料

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [PEFT 库文档](https://huggingface.co/docs/peft)
- [Wan2.2 官方文档](https://github.com/Wan-Video/Wan2.2)

## 📝 License

MIT License

