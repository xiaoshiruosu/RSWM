# Sliding Window Benchmark

遥感视频滑动窗口外推基准数据集构建工具，展示从中心区域沿方向嵌入方向滑动到目标区域的完整过程。

## 📋 项目概述

Sliding Window Benchmark 用于构建遥感视频滑动窗口外推任务的基准数据集，核心特点：

- **滑动窗口视频 (sliding_frames)**: 展示窗口从中心位置滑动到目标位置的完整过程
- **中心帧 (center_frame)**: 滑动起始位置的窗口视图
- **目标帧 (target_frame)**: 滑动结束位置的窗口视图
- **方向嵌入 (D)**: 支持任意角度 (0-360°) 的连续 sin/cos 编码
- **轨迹信息 (trajectory)**: 记录每一步窗口中心的位置坐标
- **场景类别**: general (通用)、flood (洪水)、urban (城市)、rural (农村)
- **重叠约束**: 起始与终止位置重叠率 <10%

## 🎯 核心概念

```
┌─────────────────────────────────────┐
│           Full Scene                │
│                                     │
│    ┌─────┐                          │
│    │Start│ ──────────────→ ┌─────┐  │
│    │(V_c)│   Direction D   │ End │  │
│    └─────┘                 │(V_t)│  │
│      ↑                     └─────┘  │
│   Center                   Target   │
│                                     │
└─────────────────────────────────────┘

滑动过程：窗口沿 D 方向从 Start 移动到 End
每一步都记录窗口内的图像，形成滑动视频序列
```

## 🛠️ 安装依赖

```bash
# 基础依赖
pip install numpy h5py

# 可视化依赖
pip install imageio Pillow

# 可选依赖 (用于更好的图像处理)
pip install opencv-python
```

## 📁 项目结构

```
remotesensebenchmark/
├── README.md                          # 主项目文档
├── SLIDING_WINDOW_README.md           # 本文件
├── build_sliding_window_benchmark.py  # 滑动窗口 benchmark 构建脚本
├── visualize_sliding_window_gif.py    # GIF 可视化脚本
├── sliding_window_benchmark.h5        # 生成的数据集文件
└── sliding_window_demo.gif            # 可视化演示 GIF
```

## 🚀 快速开始

### 步骤一：构建 Benchmark 数据集

```bash
# 完整运行 (400 实例，每场景 100 个)
python build_sliding_window_benchmark.py

# 快速测试 (40 实例，每场景 10 个)
python build_sliding_window_benchmark.py --instances 10

# 自定义参数
python build_sliding_window_benchmark.py --output my_benchmark.h5 --instances 50 --steps 24
```

### 步骤二：生成 GIF 可视化

```bash
# 可视化第一个样本 (带完整场景视图)
python visualize_sliding_window_gif.py

# 可视化指定样本
python visualize_sliding_window_gif.py --sample 5

# 简化版 (仅滑动窗口)
python visualize_sliding_window_gif.py --simple

# 起止对比动画
python visualize_sliding_window_gif.py --comparison
```

### 步骤三：Python API 调用

```python
from build_sliding_window_benchmark import SlidingWindowConfig, SlidingWindowBenchmarkBuilder

# 创建配置
config = SlidingWindowConfig()
config.instances_per_scene = 50       # 每场景实例数
config.num_sliding_steps = 16         # 滑动步数
config.window_size = (128, 128)       # 窗口尺寸
config.output_path = "my_benchmark.h5"

# 构建基准数据集
builder = SlidingWindowBenchmarkBuilder(config)
stats = builder.build()

print(f"生成实例数: {stats['total_instances']}")
print(f"平均重叠率: {np.mean(stats['overlap_ratios']):.2%}")
```

## ⚙️ 命令行参数

### build_sliding_window_benchmark.py

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--output` | `-o` | `sliding_window_benchmark.h5` | 输出文件路径 |
| `--instances` | `-n` | `100` | 每个场景类别的实例数 |
| `--seed` | `-s` | `42` | 随机种子 |
| `--steps` | - | `16` | 滑动步数 |

### visualize_sliding_window_gif.py

| 参数 | 缩写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | `sliding_window_benchmark.h5` | 输入 HDF5 文件 |
| `--output` | `-o` | `sliding_window_demo.gif` | 输出 GIF 文件 |
| `--sample` | `-s` | `0` | 要可视化的样本索引 |
| `--fps` | - | `5` | GIF 帧率 |
| `--simple` | - | `false` | 仅显示滑动窗口 |
| `--comparison` | - | `false` | 创建起止对比动画 |

## 📊 输出格式 (HDF5)

```
sliding_window_benchmark.h5
├── metadata/                      # 元数据
│   ├── version                   # 版本号
│   ├── total_instances           # 总实例数
│   ├── scene_categories          # 场景类别列表
│   ├── full_scene_size           # 完整场景尺寸 (512, 512)
│   ├── window_size               # 滑动窗口尺寸 (128, 128)
│   ├── num_sliding_steps         # 滑动步数
│   ├── max_overlap_ratio         # 最大重叠率阈值
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
│   ├── sample_00001/
│   └── ...
└── index/                        # 场景索引
    ├── general                   # 通用场景样本索引
    ├── flood                     # 洪水场景样本索引
    ├── urban                     # 城市场景样本索引
    └── rural                     # 农村场景样本索引
```

## 📖 读取数据示例

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
    sliding_frames = sample['sliding_frames'][:]    # (N, H, W, C) 滑动序列
    center_frame = sample['center_frame'][:]        # (H, W, C) 起始帧
    target_frame = sample['target_frame'][:]        # (H, W, C) 目标帧
    full_scene = sample['full_scene'][:]            # (H, W, C) 完整场景
    direction_embed = sample['direction_embedding'][:]  # (64,)
    trajectory = sample['trajectory'][:]            # (N, 2) 轨迹
    
    # 读取属性
    angle = sample.attrs['direction_angle']
    overlap = sample.attrs['overlap_ratio']
    scene = sample.attrs['scene_category']
    center_bounds = sample.attrs['center_bounds']
    target_bounds = sample.attrs['target_bounds']
    
    print(f"场景: {scene}, 角度: {angle:.1f}°, 重叠率: {overlap:.2%}")
    print(f"滑动序列形状: {sliding_frames.shape}")
    print(f"轨迹: 从 {trajectory[0]} 到 {trajectory[-1]}")
    
    # 按场景类别筛选
    flood_indices = f['index/flood'][:]
    print(f"洪水场景样本数: {len(flood_indices)}")
```

## 🎬 GIF 可视化说明

生成的 GIF 动画展示：

```
┌───────────────────────────────────────────────────┐
│ Step 8/16 | Angle: 45 deg | Progress: 50%         │
├─────────────────────────┬───┬─────────────────────┤
│                         │   │                     │
│     Full Scene          │ | │   Current Window    │
│     with overlays       │ | │   View              │
│                         │   │                     │
│  [Green] = Start        │   │                     │
│  [Red]   = Target       │   │                     │
│  [Yellow]= Current      │   │                     │
│                         │   │                     │
├─────────────────────────┴───┴─────────────────────┤
│ Green: Start | Red: Target | Yellow: Current      │
└───────────────────────────────────────────────────┘
```

- **左侧**: 完整场景俯视图，标注起始(绿)、目标(红)、当前(黄)窗口位置
- **右侧**: 当前滑动窗口所看到的区域
- **顶部**: 进度信息（步数、角度、完成百分比）

## 📈 预期输出

运行完成后会显示统计信息：

```
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

Instances per scene:
  - general: 100
  - flood: 100
  - urban: 100
  - rural: 100

Processing time: 12.45 seconds
======================================================================

Benchmark built successfully!
Output: sliding_window_benchmark.h5
```

## 🔧 配置说明

主要配置参数 (在 `SlidingWindowConfig` 类中):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `instances_per_scene` | 100 | 每场景实例数 |
| `full_scene_size` | (512, 512) | 完整场景尺寸 |
| `window_size` | (128, 128) | 滑动窗口尺寸 |
| `num_sliding_steps` | 16 | 滑动步数 |
| `embed_dim` | 64 | 方向嵌入维度 |
| `max_overlap_ratio` | 0.10 | 最大重叠率 |
| `num_channels` | 3 | 图像通道数 |

## 🔄 与 RSVideoWise Benchmark 的区别

| 特性 | RSVideoWise | Sliding Window |
|------|-------------|----------------|
| 核心数据 | 中心序列 + 目标序列 | 滑动窗口视频序列 |
| 时序信息 | 时间维度上的帧序列 | 空间滑动过程的帧序列 |
| 用途 | 视频外推预测 | 空间滑动过程可视化/预测 |
| 输出格式 | (T, H, W, C) | (Steps, H, W, C) |

---

# 🔀 Multi-Direction Benchmark (多方向连续外推)

支持**多步连续不同方向嵌入**的视频外推，即从起点到终点可以经过多个中间航点，每段使用不同的方向嵌入。

## 核心概念

```
┌─────────────────────────────────────────────────────┐
│                   Full Scene                        │
│                                                     │
│    ┌─────┐                                          │
│    │Start│ ────[D1: 45°]────→ ○ Waypoint1           │
│    └─────┘                     │                    │
│                                │ [D2: 135°]         │
│                                ↓                    │
│                              ○ Waypoint2            │
│                                │                    │
│                                │ [D3: 270°]         │
│                                ↓                    │
│                              ┌─────┐                │
│                              │ End │                │
│                              └─────┘                │
└─────────────────────────────────────────────────────┘

路径: Start → [方向1] → Waypoint1 → [方向2] → Waypoint2 → [方向3] → End
每段都有独立的方向嵌入，支持任意角度连续变化
```

## 快速使用

```bash
# 构建多方向 benchmark
python build_multi_direction_benchmark.py --instances 50

# 可视化第一个样本
python visualize_multi_direction_gif.py

# 自定义方向段数
python build_multi_direction_benchmark.py --min-segments 3 --max-segments 6 --steps 10
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output` | `multi_direction_benchmark.h5` | 输出文件路径 |
| `--instances` | `100` | 每场景实例数 |
| `--min-segments` | `2` | 最小方向段数 |
| `--max-segments` | `4` | 最大方向段数 |
| `--steps` | `8` | 每段滑动步数 |

## HDF5 数据结构

```
multi_direction_benchmark.h5
├── metadata/
│   ├── min_segments              # 最小方向段数
│   ├── max_segments              # 最大方向段数
│   └── steps_per_segment         # 每段步数
├── data/
│   ├── sample_00000/
│   │   ├── sliding_frames        # 完整滑动序列 (TotalSteps, H, W, C)
│   │   ├── segment_embeddings    # 每段方向嵌入 (NumSegments, EmbedDim)
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

## 读取数据示例

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
        seg_embed = segment_embeddings[seg_idx]
        print(f"段 {seg_idx+1}: 角度={seg_angle:.0f}°, 帧={seg_frames.shape}")
```

## 三种 Benchmark 对比

| 特性 | RSVideoWise | Sliding Window | Multi-Direction |
|------|-------------|----------------|-----------------|
| 方向嵌入 | 单一方向 | 单一方向 | 多段连续方向 |
| 路径类型 | 直线 | 直线 | 折线/曲线 |
| 航点数 | 2 (起止) | 2 (起止) | N+1 (N段) |
| 方向嵌入维度 | (64,) | (64,) | (N, 64) |
| 适用场景 | 简单外推 | 简单外推 | 复杂路径规划/导航 |

## 📝 License

MIT License
