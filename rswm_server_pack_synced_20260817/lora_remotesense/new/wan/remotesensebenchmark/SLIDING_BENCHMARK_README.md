# RSVideoWise Sliding Window Benchmark

遥感视频滑动窗口外推基准数据集构建工具，支持单向滑动和多方向连续外推任务。

## 📋 项目概述

Sliding Window Benchmark 用于构建遥感视频滑动窗口外推任务的基准数据集，包含两个核心工具：

### 1. Sliding Window Benchmark (单向滑动)
- **滑动窗口视频**: 展示窗口从中心位置沿指定方向滑动到目标位置的完整过程
- **方向嵌入 (D)**: 支持任意角度 (0-360°) 的连续 sin/cos 编码
- **重叠约束**: 起始与终止位置重叠率 <10%

### 2. Multi-Direction Benchmark (多方向连续外推)
- **多段方向嵌入**: 支持 2-4 段（可配置）连续不同方向
- **航点路径**: Start → Waypoint1 → Waypoint2 → ... → End
- **复杂轨迹**: 支持折线/曲线路径规划

## 📡 支持的 HuggingFace 数据集

以下数据集已验证可用 (2024年):

| 数据集 | 描述 | 样本数 |
|--------|------|--------|
| `timm/eurosat-rgb` | EuroSAT RGB Sentinel-2 卫星图像 | 27,000 |
| `blanchon/EuroSAT_RGB` | EuroSAT RGB 备用源 | 27,000 |
| `blanchon/UC_Merced` | UC Merced 土地利用数据集 | 2,100 |

> **注意**: 如果 HuggingFace 数据集不可用，系统会自动回退到合成数据生成模式。

## 🛠️ 安装依赖

```bash
# 基础依赖
pip install numpy h5py

# HuggingFace 数据源 (推荐)
pip install datasets Pillow

# 可视化依赖
pip install imageio

# 可选依赖 (用于更好的图像处理)
pip install opencv-python
```

## 📁 项目结构

```
remotesensebenchmark/
├── README.md                           # 主项目文档
├── SLIDING_BENCHMARK_README.md         # 本文件
├── build_sliding_window_benchmark.py   # 单向滑动 benchmark 构建脚本
├── build_multi_direction_benchmark.py  # 多方向 benchmark 构建脚本
├── visualize_sliding_window_gif.py     # 单向滑动 GIF 可视化
├── visualize_multi_direction_gif.py    # 多方向 GIF 可视化
├── sliding_window_benchmark.h5         # 生成的单向滑动数据集
└── multi_direction_benchmark.h5        # 生成的多方向数据集
```

## 🚀 快速开始

### Sliding Window Benchmark (单向滑动)

```bash
# 使用 HuggingFace 数据集 (默认)
python build_sliding_window_benchmark.py

# 快速测试 (40 实例，每场景 10 个)
python build_sliding_window_benchmark.py --instances 10

# 使用合成数据 (离线模式)
python build_sliding_window_benchmark.py --synthetic

# 自定义参数
python build_sliding_window_benchmark.py --output my_benchmark.h5 --instances 50 --steps 24

# 生成 GIF 可视化
python visualize_sliding_window_gif.py
```

### Multi-Direction Benchmark (多方向连续外推)

```bash
# 使用 HuggingFace 数据集 (默认)
python build_multi_direction_benchmark.py

# 快速测试
python build_multi_direction_benchmark.py --instances 10

# 使用合成数据 (离线模式)
python build_multi_direction_benchmark.py --synthetic

# 自定义方向段数
python build_multi_direction_benchmark.py --min-segments 3 --max-segments 6 --steps 10

# 生成 GIF 可视化
python visualize_multi_direction_gif.py
```

### Python API 调用

```python
from build_sliding_window_benchmark import SlidingWindowConfig, SlidingWindowBenchmarkBuilder
from build_multi_direction_benchmark import MultiDirectionConfig, MultiDirectionBenchmarkBuilder

# === 单向滑动 ===
config = SlidingWindowConfig()
config.instances_per_scene = 50
config.num_sliding_steps = 16
config.output_path = "my_sliding_benchmark.h5"

builder = SlidingWindowBenchmarkBuilder(config, use_hf=True)
stats = builder.build()

# === 多方向连续外推 ===
config = MultiDirectionConfig()
config.instances_per_scene = 50
config.min_segments = 2
config.max_segments = 4
config.output_path = "my_multi_benchmark.h5"

builder = MultiDirectionBenchmarkBuilder(config, use_hf=True)
stats = builder.build()
```

## ⚙️ 命令行参数

### build_sliding_window_benchmark.py

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | `-o` | `sliding_window_benchmark.h5` | 输出文件路径 |
| `--instances` | `-n` | `100` | 每个场景类别的实例数 |
| `--seed` | `-s` | `42` | 随机种子 |
| `--steps` | - | `16` | 滑动步数 |
| `--synthetic` | - | `false` | 使用合成数据代替 HuggingFace |

### build_multi_direction_benchmark.py

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | `-o` | `multi_direction_benchmark.h5` | 输出文件路径 |
| `--instances` | `-n` | `100` | 每个场景类别的实例数 |
| `--seed` | `-s` | `42` | 随机种子 |
| `--min-segments` | - | `2` | 最小方向段数 |
| `--max-segments` | - | `4` | 最大方向段数 |
| `--steps` | - | `8` | 每段滑动步数 |
| `--synthetic` | - | `false` | 使用合成数据代替 HuggingFace |

## 📊 输出格式 (HDF5)

### Sliding Window Benchmark

```
sliding_window_benchmark.h5
├── metadata/                      # 元数据
│   ├── version                   # 版本号
│   ├── total_instances           # 总实例数
│   ├── scene_categories          # 场景类别列表
│   ├── full_scene_size           # 完整场景尺寸 (512, 512)
│   ├── window_size               # 滑动窗口尺寸 (128, 128)
│   ├── num_sliding_steps         # 滑动步数
│   └── creation_time             # 创建时间
├── data/                         # 数据
│   ├── sample_00000/
│   │   ├── sliding_frames        # 滑动窗口序列 (N, H, W, C)
│   │   ├── center_frame          # 起始帧 (H, W, C)
│   │   ├── target_frame          # 目标帧 (H, W, C)
│   │   ├── full_scene            # 完整场景 (H, W, C)
│   │   ├── direction_embedding   # 方向嵌入 (64,)
│   │   ├── trajectory            # 轨迹坐标 (N, 2)
│   │   └── [attrs]               # 属性 (角度、重叠率等)
│   └── ...
└── index/                        # 场景索引
    ├── general
    ├── flood
    ├── urban
    └── rural
```

### Multi-Direction Benchmark

```
multi_direction_benchmark.h5
├── metadata/
│   ├── min_segments              # 最小方向段数
│   ├── max_segments              # 最大方向段数
│   └── steps_per_segment         # 每段步数
├── data/
│   ├── sample_00000/
│   │   ├── sliding_frames        # 完整滑动序列 (TotalSteps, H, W, C)
│   │   ├── segment_embeddings    # 每段方向嵌入 (NumSegments, 64)
│   │   ├── segment_angles        # 每段方向角度 (NumSegments,)
│   │   ├── waypoints             # 航点坐标 (NumSegments+1, 2)
│   │   ├── start_frame           # 起始帧
│   │   ├── end_frame             # 结束帧
│   │   └── full_scene            # 完整场景
│   └── ...
└── index_by_segments/            # 按段数索引
    ├── segments_2
    ├── segments_3
    └── segments_4
```

## 📖 读取数据示例

### 读取 Sliding Window Benchmark

```python
import h5py
import numpy as np

with h5py.File('sliding_window_benchmark.h5', 'r') as f:
    # 读取元数据
    total = f['metadata'].attrs['total_instances']
    window_size = f['metadata'].attrs['window_size']
    num_steps = f['metadata'].attrs['num_sliding_steps']
    print(f"总实例数: {total}, 窗口尺寸: {window_size}, 滑动步数: {num_steps}")
    
    # 读取单个样本
    sample = f['data/sample_00000']
    sliding_frames = sample['sliding_frames'][:]    # (N, H, W, C)
    center_frame = sample['center_frame'][:]        # (H, W, C)
    target_frame = sample['target_frame'][:]        # (H, W, C)
    direction_embed = sample['direction_embedding'][:]  # (64,)
    trajectory = sample['trajectory'][:]            # (N, 2)
    
    angle = sample.attrs['direction_angle']
    overlap = sample.attrs['overlap_ratio']
    scene = sample.attrs['scene_category']
    
    print(f"场景: {scene}, 角度: {angle:.1f}°, 重叠率: {overlap:.2%}")
    print(f"滑动序列形状: {sliding_frames.shape}")
```

### 读取 Multi-Direction Benchmark

```python
import h5py
import numpy as np

with h5py.File('multi_direction_benchmark.h5', 'r') as f:
    sample = f['data/sample_00000']
    
    # 读取多段方向嵌入
    segment_embeddings = sample['segment_embeddings'][:]  # (NumSeg, 64)
    segment_angles = sample['segment_angles'][:]          # (NumSeg,)
    waypoints = sample['waypoints'][:]                    # (NumSeg+1, 2)
    sliding_frames = sample['sliding_frames'][:]          # (TotalSteps, H, W, C)
    
    num_segments = sample.attrs['num_segments']
    steps_per_segment = sample.attrs['steps_per_segment']
    
    print(f"方向段数: {num_segments}")
    print(f"每段步数: {steps_per_segment}")
    print(f"总帧数: {len(sliding_frames)}")
    print(f"方向角度序列: {segment_angles}")
    
    # 按段提取帧
    for seg_idx in range(num_segments):
        start_idx = seg_idx * steps_per_segment
        end_idx = start_idx + steps_per_segment
        seg_frames = sliding_frames[start_idx:end_idx]
        seg_angle = segment_angles[seg_idx]
        print(f"段 {seg_idx+1}: 角度={seg_angle:.0f}°, 帧数={len(seg_frames)}")
```

## 🎬 GIF 可视化说明

### 单向滑动 GIF

```bash
python visualize_sliding_window_gif.py --sample 0 --fps 5
```

生成的 GIF 展示：
- **左侧**: 完整场景俯视图，标注起始(绿)、目标(红)、当前(黄)窗口位置
- **右侧**: 当前滑动窗口所看到的区域

### 多方向 GIF

```bash
python visualize_multi_direction_gif.py --sample 0 --fps 5
```

生成的 GIF 展示：
- 完整场景俯视图 + 彩色路径轨迹
- 每段用不同颜色标注
- 当前窗口位置实时显示

## 📈 预期输出

### Sliding Window Benchmark

```
======================================================================
Sliding Window Benchmark Builder
======================================================================

Configuration:
  - Output: sliding_window_benchmark.h5
  - Instances per scene: 100
  - Scene categories: ['general', 'flood', 'urban', 'rural']
  - Total instances: 400
  - Full scene size: (512, 512)
  - Window size: (128, 128)
  - Sliding steps: 16
  - Data source: HuggingFace

Loading HuggingFace datasets...
  general: Loading timm/eurosat-rgb...
  general: Loaded 27000 samples from timm/eurosat-rgb
  flood: Using cached timm/eurosat-rgb
  urban: Loading blanchon/UC_Merced...
  urban: Loaded 2100 samples from blanchon/UC_Merced
  rural: Using cached timm/eurosat-rgb

Processing scene: general
  Generated 100 samples
...

======================================================================
Sliding Window Benchmark Statistics
======================================================================
Total instances: 400
Valid instances (<10% overlap): 400
Invalid instances: 0

Overlap Statistics:
  - Average: 0.00%
  - Min: 0.00%
  - Max: 5.23%

Processing time: 45.32 seconds
======================================================================
```

### Multi-Direction Benchmark

```
======================================================================
Multi-Direction Benchmark Statistics
======================================================================
Total instances: 400
Valid instances (<10% overlap): 400

Overlap Statistics:
  - Average: 0.00%
  - Max: 3.15%

Segments Distribution:
  - 2 segments: 130 samples
  - 3 segments: 145 samples
  - 4 segments: 125 samples

Processing time: 38.21 seconds
======================================================================
```

## 🔧 配置说明

### SlidingWindowConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `instances_per_scene` | 100 | 每场景实例数 |
| `full_scene_size` | (512, 512) | 完整场景尺寸 |
| `window_size` | (128, 128) | 滑动窗口尺寸 |
| `num_sliding_steps` | 16 | 滑动步数 |
| `embed_dim` | 64 | 方向嵌入维度 |
| `max_overlap_ratio` | 0.10 | 最大重叠率 |

### MultiDirectionConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `instances_per_scene` | 100 | 每场景实例数 |
| `full_scene_size` | (512, 512) | 完整场景尺寸 |
| `window_size` | (96, 96) | 滑动窗口尺寸 |
| `min_segments` | 2 | 最小方向段数 |
| `max_segments` | 4 | 最大方向段数 |
| `steps_per_segment` | 8 | 每段滑动步数 |
| `embed_dim` | 64 | 方向嵌入维度 |

## 🔄 三种 Benchmark 对比

| 特性 | RSVideoWise | Sliding Window | Multi-Direction |
|------|-------------|----------------|-----------------|
| 核心数据 | 中心序列 + 目标序列 | 滑动窗口视频序列 | 多段滑动窗口序列 |
| 方向嵌入 | 单一方向 | 单一方向 | 多段连续方向 |
| 路径类型 | 直线 | 直线 | 折线/曲线 |
| 航点数 | 2 (起止) | 2 (起止) | N+1 (N段) |
| 方向嵌入维度 | (64,) | (64,) | (N, 64) |
| 时序信息 | 时间维度帧序列 | 空间滑动帧序列 | 空间滑动帧序列 |
| 数据源 | HuggingFace | HuggingFace | HuggingFace |
| 适用场景 | 视频外推预测 | 空间外推可视化 | 复杂路径规划/导航 |

## 📝 License

MIT License
