# LoRA 微调文件总结

本文档列出了为 Wan2.2 LoRA 微调创建的所有文件及其用途。

## 📁 创建的文件列表

### 核心训练文件

1. **`dataset_loader.py`** - 数据集加载器
   - PyTorch Dataset 类，用于加载 Multi-Direction Benchmark HDF5 数据
   - 支持场景类别筛选、方向段数筛选
   - 自动归一化和数据转换
   - 约 150 行代码

2. **`train_lora.py`** - 主训练脚本
   - 完整的 LoRA 训练流程
   - 包含模型初始化、数据加载、训练循环
   - 支持 Weights & Biases 日志记录
   - 自动保存检查点
   - 约 500 行代码

3. **`train_config.yaml`** - 训练配置文件
   - 所有训练超参数的配置
   - 包含不同硬件的推荐配置
   - 详细的配置说明和提示

### 推理和测试文件

4. **`inference_with_lora.py`** - LoRA 推理脚本
   - 加载训练好的 LoRA 权重
   - 使用 LoRA 模型生成视频
   - 支持文本到视频和图像到视频
   - 约 200 行代码

### 自动化脚本

5. **`run_training.sh`** - 训练启动脚本
   - 提供多种训练配置示例
   - 包含 RTX 4090、A100 等不同硬件的配置
   - 可直接运行或作为参考

6. **`complete_workflow.sh`** - 完整工作流脚本
   - 从数据生成到训练到推理的完整流程
   - 一键运行所有步骤
   - 包含进度提示和错误处理

### 文档文件

7. **`LORA_TRAINING_README.md`** - 完整训练文档
   - 详细的使用说明
   - 环境准备、数据准备、训练配置
   - 常见问题解答
   - 约 300 行文档

8. **`QUICKSTART_LORA.md`** - 快速入门指南
   - 5 分钟快速上手
   - 一键运行命令
   - 常用配置示例

9. **`LORA_FILES_SUMMARY.md`** - 本文件
   - 所有文件的总结和说明

## 🗂️ 文件结构

```
remotesensebenchmark/
├── dataset_loader.py              # 数据集加载器
├── train_lora.py                  # 主训练脚本
├── train_config.yaml              # 训练配置
├── inference_with_lora.py         # LoRA 推理脚本
├── run_training.sh                # 训练启动脚本
├── complete_workflow.sh           # 完整工作流
├── LORA_TRAINING_README.md        # 完整文档
├── QUICKSTART_LORA.md             # 快速入门
├── LORA_FILES_SUMMARY.md          # 本文件
│
├── build_multi_direction_benchmark_v2.py  # 数据生成（已存在）
├── visualize_multi_direction_mp4.py       # 可视化（已存在）
└── ... (其他已存在的文件)
```

## 🚀 使用流程

### 方式 1: 一键运行（推荐新手）

```bash
chmod +x remotesensebenchmark/complete_workflow.sh
./remotesensebenchmark/complete_workflow.sh
```

### 方式 2: 分步执行

```bash
# 1. 生成数据
python remotesensebenchmark/build_multi_direction_benchmark_v2.py \
    --output multi_direction_benchmark_v2.h5 --instances 100

# 2. 训练 LoRA
python remotesensebenchmark/train_lora.py \
    --model_config ti2v-5B \
    --checkpoint_dir ./Wan2.2-TI2V-5B \
    --data_path multi_direction_benchmark_v2.h5 \
    --num_epochs 10

# 3. 推理测试
python remotesensebenchmark/inference_with_lora.py \
    --model_config ti2v-5B \
    --checkpoint_dir ./Wan2.2-TI2V-5B \
    --lora_checkpoint ./lora_checkpoints/checkpoint-epoch-10 \
    --prompt "Your prompt here"
```

### 方式 3: 使用配置脚本

```bash
chmod +x remotesensebenchmark/run_training.sh
./remotesensebenchmark/run_training.sh
```

## 📊 关键特性

### dataset_loader.py
- ✅ 自动加载 HDF5 数据
- ✅ 支持数据筛选（场景、方向段数）
- ✅ 自动归一化到 [-1, 1]
- ✅ PyTorch DataLoader 兼容

### train_lora.py
- ✅ 完整的训练循环
- ✅ 自动保存检查点
- ✅ 梯度裁剪
- ✅ 学习率调度
- ✅ W&B 集成
- ✅ 进度条显示

### inference_with_lora.py
- ✅ 加载 LoRA 权重
- ✅ 支持 T2V/I2V
- ✅ 可配置生成参数

## 🎯 核心功能

1. **数据加载**: `MultiDirectionDataset` 类
   - 从 HDF5 文件加载视频序列
   - 加载方向嵌入和元数据
   - 支持数据增强（可扩展）

2. **LoRA 训练**: `LoRATrainer` 类
   - 初始化 VAE、T5、Wan 模型
   - 配置 LoRA 适配器
   - 实现训练循环和损失计算
   - 保存和恢复检查点

3. **推理**: `load_model_with_lora` 函数
   - 加载基础模型
   - 应用 LoRA 权重
   - 生成视频

## 💾 输出文件

训练过程会生成：

```
lora_checkpoints/
├── checkpoint-epoch-1/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── training_state.pt
├── checkpoint-epoch-2/
│   └── ...
└── checkpoint-epoch-10/
    └── ...
```

## 📈 预期性能

| 硬件 | 模型 | 批次大小 | 训练时间 (10 epochs, 400 samples) |
|------|------|---------|----------------------------------|
| RTX 4090 | TI2V-5B | 1 | ~2-4 小时 |
| A100 40GB | TI2V-5B | 2 | ~1-2 小时 |
| A100 80GB | T2V-A14B | 2 | ~3-5 小时 |

## 🔧 自定义和扩展

### 添加新的数据增强

在 `dataset_loader.py` 中修改 `__getitem__` 方法：

```python
if self.transform is not None:
    result = self.transform(result)
```

### 修改损失函数

在 `train_lora.py` 中修改 `compute_loss` 方法：

```python
def compute_loss(self, batch: Dict) -> torch.Tensor:
    # 自定义损失计算
    pass
```

### 添加验证集

在 `LoRATrainer` 类中添加 `validate` 方法。

## 📚 相关文档

- [完整训练文档](LORA_TRAINING_README.md) - 详细使用说明
- [快速入门](QUICKSTART_LORA.md) - 5 分钟上手
- [数据集文档](SLIDING_BENCHMARK_README.md) - Benchmark 数据集说明
- [Wan2.2 主文档](../README.md) - Wan2.2 模型文档

## 🤝 贡献

如果你有改进建议或发现问题，欢迎提交 Issue 或 Pull Request！

## 📝 License

MIT License


核心训练文件
remotesensebenchmark/dataset_loader.py - PyTorch 数据集加载器
加载 Multi-Direction Benchmark HDF5 数据
支持场景筛选、数据归一化
remotesensebenchmark/train_lora.py - 主训练脚本（~500 行）
完整的 LoRA 训练流程
支持 VAE、T5、Wan 模型初始化
训练循环、损失计算、检查点保存
W&B 日志集成
remotesensebenchmark/train_config.yaml - 训练配置文件
所有超参数配置
不同硬件的推荐设置
推理和测试
remotesensebenchmark/inference_with_lora.py - LoRA 推理脚本
加载训练好的 LoRA 权重
生成视频
自动化脚本
remotesensebenchmark/run_training.sh - 训练启动脚本
多种配置示例（RTX 4090、A100 等）
remotesensebenchmark/complete_workflow.sh - 完整工作流
一键运行：数据生成 → 训练 → 推理
文档
remotesensebenchmark/LORA_TRAINING_README.md - 完整文档（~300 行）
详细使用说明、环境准备、常见问题
remotesensebenchmark/QUICKSTART_LORA.md - 快速入门
5 分钟上手指南
remotesensebenchmark/LORA_FILES_SUMMARY.md - 文件总结
所有文件的说明和使用流程
🚀 快速开始
方式 1: 一键运行（最简单）
方式 2: 分步执行
✨ 主要特性
✅ 完整的训练流程: 数据加载、模型初始化、训练循环、检查点保存
✅ LoRA 高效微调: 使用 PEFT 库，只训练 <1% 的参数
✅ 灵活配置: 支持多种模型（TI2V-5B、T2V-A14B、I2V-A14B）
✅ 硬件适配: RTX 4090、A100 等不同配置
✅ 监控和日志: 进度条、W&B 集成
✅ 详细文档: 完整的使用说明和快速入门
📊 预期性能
硬件	模型	训练时间 (10 epochs, 400 samples)
RTX 4090	TI2V-5B	~2-4 小时
A100 40GB	TI2V-5B	~1-2 小时
A100 80GB	T2V-A14B	~3-5 小时