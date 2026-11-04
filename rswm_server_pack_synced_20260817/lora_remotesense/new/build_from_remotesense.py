#!/usr/bin/env python3
"""
LingBot-World LoRA 训练数据集构建器 - 从 RemoteSense Benchmark 转换

本脚本从 wan/remotesensebenchmark 的 Multi-Direction Benchmark V5 生成数据，
并转换为 LingBot I2V LoRA 训练所需的 HDF5 格式。

【数据源】
- wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py
- 生成遥感滑动窗口序列（sliding window）+ 方向嵌入（direction embeddings）

【字段映射】RemoteSense -> LingBot HDF5
┌─────────────────────────────────┬──────────────────────────────────┐
│ RemoteSense 字段                 │ LingBot HDF5 字段                 │
├─────────────────────────────────┼──────────────────────────────────┤
│ center_frame (H,W,3)            │ image (H,W,3)                    │
│ sliding_frames (T,H,W,3)        │ video_frames (T,H,W,3)           │
│ N/A (遥感无相机参数)             │ intrinsics (T,4) [DUMMY]         │
│ N/A (遥感无相机参数)             │ poses (T,4,4) [DUMMY]            │
│ scene_category + direction      │ prompt (str)                     │
│ total_frames                    │ num_frames (int)                 │
└─────────────────────────────────┴──────────────────────────────────┘

【Dummy 相机参数说明】
由于遥感数据不包含真实相机内参/外参，本脚本生成 dummy 值：
- intrinsics: fx=fy=500, cx=W/2, cy=H/2（基于帧宽高）
- poses: 单位矩阵 4x4（identity），所有帧复用

如需替换为真实相机参数：
1. 修改 _make_dummy_intrinsics() 和 _make_dummy_poses() 函数
2. 或在训练时使用 --ignore_camera_control 跳过相机条件（需修改 train_lora.py）

【使用方法】
# 生成 demo 数据集（20 样本，用于快速测试）
python build_from_remotesense.py --demo --instances 20 -o data/remotesense/demo.h5

# 生成完整数据集（100 样本）
python build_from_remotesense.py --instances 100 -o data/remotesense/train.h5

# 使用合成数据（不依赖 HuggingFace datasets）
python build_from_remotesense.py --demo --no-hf --instances 20

【训练验证】
python train_lora.py \
  --checkpoint_dir ./lingbot-world-base-cam \
  --data_path data/remotesense/demo.h5 \
  --output_dir ./test_remotesense_lora \
  --num_epochs 1 \
  --batch_size 1 \
  --max_frames 33
"""

import os
import sys
import argparse
import time
import math
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import h5py

# new/ 自身 + 项目根 加入 sys.path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if (HERE / "wan").is_dir():
    PROJECT_ROOT = HERE
else:
    PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from camera_trajectory import (
    direction_angle_from_waypoints,
    make_poses_from_waypoints,
)


# ============================================================================
# 动态导入 remotesensebenchmark 模块（避免触发 wan/__init__.py）
# ============================================================================

def _load_remotesense_module():
    """
    使用 importlib 动态加载 remotesensebenchmark/build_multi_direction_benchmark_v5.py

    避免使用 `from wan.remotesensebenchmark import ...`，因为会触发 wan/__init__.py
    导入整个 wan 包，导致依赖 diffusers/easydict 等无关库。

    Returns:
        module: 加载的模块对象，包含 MultiDirectionConfig 和 MultiDirectionBenchmarkBuilder
    """
    v5_path = PROJECT_ROOT / "wan" / "remotesensebenchmark" / "build_multi_direction_benchmark_v5.py"

    if not v5_path.exists():
        raise FileNotFoundError(
            f"未找到 remotesensebenchmark v5 脚本: {v5_path}\n"
            f"请确认 wan/remotesensebenchmark/ 目录存在"
        )

    # 使用 importlib 按文件路径动态加载模块
    spec = importlib.util.spec_from_file_location(
        "remotesense_v5_module",  # 模块名（任意，不会污染 sys.modules）
        v5_path
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"无法创建模块 spec: {v5_path}")

    module = importlib.util.module_from_spec(spec)

    # 执行模块代码（加载类和函数定义）
    spec.loader.exec_module(module)

    return module


# 动态加载 remotesensebenchmark v5 模块
_rs_module = _load_remotesense_module()
MultiDirectionConfig = _rs_module.MultiDirectionConfig
MultiDirectionBenchmarkBuilder = _rs_module.MultiDirectionBenchmarkBuilder


# ============================================================================
# Dummy 相机参数生成
# ============================================================================

def _make_dummy_intrinsics(num_frames: int, width: int, height: int) -> np.ndarray:
    """
    生成 dummy 相机内参 [num_frames, 4]，格式：[fx, fy, cx, cy]

    遥感数据通常无真实相机参数，这里使用合理的默认值：
    - fx = fy = 500（假设焦距）
    - cx = width / 2（主点在图像中心）
    - cy = height / 2

    所有帧使用相同内参（静态相机假设）。
    """
    fx = fy = 500.0
    cx = width / 2.0
    cy = height / 2.0
    intr = np.array([fx, fy, cx, cy], dtype=np.float32)
    intrinsics = np.repeat(intr[None, :], repeats=num_frames, axis=0)
    return intrinsics


def _make_dummy_poses(num_frames: int) -> np.ndarray:
    """
    生成 dummy 相机位姿 [num_frames, 4, 4]，格式：4x4 变换矩阵

    遥感数据无相机轨迹，这里使用单位矩阵（identity）表示静态相机。
    所有帧位姿相同（无平移、无旋转）。
    """
    identity = np.eye(4, dtype=np.float32)
    poses = np.repeat(identity[None, :, :], repeats=num_frames, axis=0)
    return poses


def _generate_prompt_from_metadata(
    scene_category: str,
    num_segments: int,
    segment_angles: np.ndarray,
    waypoints: np.ndarray = None,
) -> str:
    """
    从 RemoteSense 元数据生成文本 prompt。

    方向由 waypoints 起终点推断（与 pose 注入用同一套逻辑，保证训练信号一致）。
    θ 约定（benchmark v5 line 483）:
        0°=east, 90°=north, 180°=west, 270°=south, 范围 [0, 360)。

    旧实现用 mean(segment_angles) + 翻转的映射，在 77.6% 样本上和 pose 方向相反，
    这是 pilot 实验方向控制完全失败的根因之一。
    """
    scene_desc = {
        "general": "a general remote sensing scene",
        "flood": "a flooded area with water bodies",
        "urban": "an urban area with buildings and roads",
        "rural": "a rural area with fields and vegetation",
    }

    desc = scene_desc.get(scene_category, "a remote sensing scene")

    direction = ""
    if waypoints is not None and len(waypoints) >= 2:
        theta = direction_angle_from_waypoints(waypoints)
    elif len(segment_angles) > 0:
        theta = float(segment_angles[0]) % 360.0
    else:
        theta = None

    if theta is not None:
        if theta < 45.0 or theta >= 315.0:
            direction = "moving eastward"
        elif theta < 135.0:
            direction = "moving northward"
        elif theta < 225.0:
            direction = "moving westward"
        else:
            direction = "moving southward"

    if direction:
        return f"A video sequence of {desc}, {direction} with {num_segments} direction changes."
    return f"A video sequence of {desc}."


# ============================================================================
# RemoteSense -> LingBot HDF5 转换器
# ============================================================================

class RemoteSenseToLingBotConverter:
    """将 RemoteSense Benchmark 数据转换为 LingBot HDF5 格式"""

    def __init__(self, output_path: str, target_size: Tuple[int, int] = (480, 832)):
        """
        Args:
            output_path: 输出 HDF5 文件路径
            target_size: 目标分辨率 (H, W)，默认 480x832
        """
        self.output_path = Path(output_path)
        self.target_size = target_size
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert_from_remotesense_h5(self, remotesense_h5_path: str, max_samples: int = None) -> Dict:
        """
        从 RemoteSense 的 HDF5 文件读取并转换为 LingBot 格式

        Args:
            remotesense_h5_path: RemoteSense benchmark.h5 路径
            max_samples: 最多转换的样本数（None = 全部）

        Returns:
            转换统计信息
        """
        print(f"\n{'='*70}")
        print("RemoteSense -> LingBot HDF5 转换器")
        print(f"{'='*70}")
        print(f"输入: {remotesense_h5_path}")
        print(f"输出: {self.output_path}")
        print(f"目标分辨率: {self.target_size[0]}x{self.target_size[1]}")
        print()

        start_time = time.time()

        with h5py.File(remotesense_h5_path, 'r') as src_f:
            metadata = dict(src_f['metadata'].attrs)
            total_instances = metadata['total_instances']

            if max_samples is not None:
                total_instances = min(total_instances, max_samples)

            # Window size drives the Plücker translation scale (1 window = 1 unit).
            # Fall back to a square 96 window if the benchmark didn't record it.
            win_raw = metadata.get('window_size', (96, 96))
            try:
                win_h, win_w = int(win_raw[0]), int(win_raw[1])
            except (TypeError, IndexError):
                win_h = win_w = int(win_raw)
            full_scene_raw = metadata.get('full_scene_size', (512, 512))
            try:
                scene_h, scene_w = int(full_scene_raw[0]), int(full_scene_raw[1])
            except (TypeError, IndexError):
                scene_h = scene_w = int(full_scene_raw)

            print(f"RemoteSense 元数据:")
            print(f"  - 总样本数: {metadata['total_instances']} (使用: {total_instances})")
            print(f"  - 场景类别: {metadata.get('scene_categories', 'N/A')}")
            print(f"  - 窗口大小: {metadata.get('window_size', 'N/A')}")
            print(f"  - 方向段数: {metadata.get('min_segments', '?')}-{metadata.get('max_segments', '?')}")
            print()

            with h5py.File(self.output_path, 'w') as dst_f:
                # 写入 LingBot 元数据
                meta = dst_f.create_group('metadata')
                meta.attrs['total_instances'] = total_instances
                meta.attrs['target_size'] = self.target_size
                meta.attrs['max_frames'] = 0  # 将在转换过程中更新
                meta.attrs['build_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                meta.attrs['source'] = 'remotesensebenchmark_v5'
                meta.attrs['source_file'] = str(remotesense_h5_path)
                meta.attrs['pose_injection'] = 'plucker_from_waypoints'
                meta.attrs['pose_window_size'] = (win_h, win_w)

                data_grp = dst_f.create_group('data')

                converted_count = 0
                max_frames_seen = 0

                for idx in range(total_instances):
                    sample_key = f'sample_{idx:05d}'

                    if sample_key not in src_f['data']:
                        print(f"警告: 跳过缺失样本 {sample_key}")
                        continue

                    src_sample = src_f['data'][sample_key]

                    # 读取 RemoteSense 数据
                    sliding_frames = src_sample['sliding_frames'][:]  # (T, H, W, 3)
                    center_frame = src_sample['center_frame'][:]      # (H, W, 3)
                    scene_category = src_sample.attrs['scene_category']
                    num_segments = src_sample.attrs['num_segments']
                    segment_angles = src_sample['segment_angles'][:]
                    # Waypoints are (N, 2) pixel coords (cy, cx). Required for
                    # directional Plücker pose injection (replaces dummy I_4).
                    waypoints = src_sample['waypoints'][:] if 'waypoints' in src_sample else None

                    num_frames = sliding_frames.shape[0]
                    max_frames_seen = max(max_frames_seen, num_frames)

                    # 转换为 LingBot 格式
                    image = self._resize_frame(center_frame, self.target_size)
                    video_frames = np.stack([
                        self._resize_frame(sliding_frames[i], self.target_size)
                        for i in range(num_frames)
                    ], axis=0)

                    H, W = self.target_size
                    intrinsics = _make_dummy_intrinsics(num_frames, W, H)
                    # Directional c2w poses derived from benchmark waypoints.
                    # Falls back to identity only when waypoints are absent
                    # (legacy benchmark versions), which preserves training
                    # behavior for old data.
                    if waypoints is not None and len(waypoints) >= 2:
                        poses = make_poses_from_waypoints(
                            num_frames=num_frames,
                            waypoints=waypoints,
                            scene_size=(scene_h, scene_w),
                            window_size=(win_h, win_w),
                        )
                    else:
                        poses = _make_dummy_poses(num_frames)
                    prompt = _generate_prompt_from_metadata(
                        scene_category, num_segments, segment_angles, waypoints
                    )

                    pose_source = "plucker_from_waypoints" if (
                        waypoints is not None and len(waypoints) >= 2
                    ) else "dummy_identity"

                    # 写入 LingBot HDF5
                    dst_sample = data_grp.create_group(sample_key)
                    dst_sample.create_dataset('image', data=image, compression='gzip')
                    dst_sample.create_dataset('video_frames', data=video_frames, compression='gzip')
                    dst_sample.create_dataset('intrinsics', data=intrinsics)
                    dst_sample.create_dataset('poses', data=poses)
                    dst_sample.attrs['pose_source'] = pose_source
                    # num_frames 同时存为 dataset 和 attrs（保持兼容性）
                    dst_sample.create_dataset('num_frames', data=np.array(num_frames, dtype=np.int32))
                    dst_sample.attrs['prompt'] = prompt
                    dst_sample.attrs['num_frames'] = num_frames  # dataset_loader.py 从这里读取
                    dst_sample.attrs['height'] = H
                    dst_sample.attrs['width'] = W
                    dst_sample.attrs['source_scene_category'] = scene_category
                    dst_sample.attrs['source_num_segments'] = num_segments

                    converted_count += 1

                    if (converted_count) % 10 == 0:
                        print(f"  进度: {converted_count}/{total_instances}")

                # 更新元数据
                meta.attrs['max_frames'] = max_frames_seen

        elapsed = time.time() - start_time

        stats = {
            'total_instances': converted_count,
            'target_size': self.target_size,
            'max_frames': max_frames_seen,
            'output_path': str(self.output_path),
            'processing_time': elapsed,
        }

        print(f"\n转换完成!")
        print(f"  - 转换样本数: {converted_count}")
        print(f"  - 最大帧数: {max_frames_seen}")
        print(f"  - 输出文件: {self.output_path}")
        print(f"  - 处理时间: {elapsed:.2f}s")
        print()

        return stats

    def _resize_frame(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize 单帧到目标分辨率

        Args:
            frame: (H, W, 3) float32
            target_size: (H, W)

        Returns:
            Resized frame (H, W, 3) float32
        """
        try:
            import cv2
            H, W = target_size
            return cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            # Fallback: numpy 简单插值
            H, W = target_size
            h, w = frame.shape[:2]
            h_idx = np.linspace(0, h - 1, H).astype(int)
            w_idx = np.linspace(0, w - 1, W).astype(int)
            return frame[np.ix_(h_idx, w_idx)]


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从 RemoteSense Benchmark 生成 LingBot LoRA 训练数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demo 模式：生成小规模测试数据集（默认 20 样本）"
    )
    parser.add_argument(
        "--instances",
        "-n",
        type=int,
        default=None,
        help="生成的样本数量（demo 模式默认 20，否则默认 100）"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出 HDF5 路径（默认: data/remotesense/demo.h5 或 train.h5）"
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=480,
        help="目标高度（默认 480）"
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=832,
        help="目标宽度（默认 832）"
    )
    parser.add_argument(
        "--no-hf",
        action="store_true",
        help="不使用 HuggingFace datasets，使用合成数据"
    )
    parser.add_argument(
        "--temp-dir",
        type=str,
        default="data/remotesense/temp",
        help="RemoteSense benchmark 临时输出目录"
    )

    args = parser.parse_args()

    # 设置默认值
    if args.demo:
        instances = args.instances if args.instances is not None else 20
        output_path = args.output if args.output else "data/remotesense/demo.h5"
    else:
        instances = args.instances if args.instances is not None else 100
        output_path = args.output if args.output else "data/remotesense/train.h5"

    target_size = (args.target_height, args.target_width)

    print(f"\n{'='*70}")
    print("步骤 1: 生成 RemoteSense Benchmark 数据")
    print(f"{'='*70}\n")

    # 配置 RemoteSense Benchmark Builder
    # 每个场景生成 instances/4 个样本（4 个场景类别）
    instances_per_scene = max(1, instances // 4)

    config = MultiDirectionConfig(
        output_dir=args.temp_dir,
        instances_per_scene=instances_per_scene,
        scene_categories=["general", "flood", "urban", "rural"],
        full_scene_size=(512, 512),
        window_size=(96, 96),
        steps_per_segment=8,
        min_segments=2,
        max_segments=4,
        random_seed=42,
    )

    # 构建 RemoteSense benchmark
    builder = MultiDirectionBenchmarkBuilder(config, use_hf=not args.no_hf)
    stats = builder.build()

    remotesense_h5 = Path(args.temp_dir) / "benchmark.h5"

    if not remotesense_h5.exists():
        print(f"\n错误: RemoteSense benchmark 生成失败，未找到 {remotesense_h5}")
        return 1

    print(f"\n{'='*70}")
    print("步骤 2: 转换为 LingBot HDF5 格式")
    print(f"{'='*70}\n")

    # 转换为 LingBot 格式
    converter = RemoteSenseToLingBotConverter(output_path, target_size=target_size)
    convert_stats = converter.convert_from_remotesense_h5(
        str(remotesense_h5),
        max_samples=instances
    )

    print(f"\n{'='*70}")
    print("完成!")
    print(f"{'='*70}")
    print(f"\nLingBot 数据集已保存: {output_path}")
    print(f"样本数: {convert_stats['total_instances']}")
    print(f"最大帧数: {convert_stats['max_frames']}")
    print(f"\n【验证训练命令】")
    print(f"python train_lora.py \\")
    print(f"  --checkpoint_dir ./lingbot-world-base-cam \\")
    print(f"  --data_path {output_path} \\")
    print(f"  --output_dir ./test_remotesense_lora \\")
    print(f"  --num_epochs 1 \\")
    print(f"  --batch_size 1 \\")
    print(f"  --max_frames {min(33, convert_stats['max_frames'])}")
    print()

    return 0


if __name__ == "__main__":
    # =========================================================================
    # 硬守卫：这不是论文的数据构建器！
    #   本脚本 = 96×96 窗 / steps=8 / eurosat合成 → 16/24/32 帧,横向拉伸糊,
    #   与论文数字【不可比】。正确的构建器是:
    #       lora_remotesense/build_dataset_from_benchmark.py  (128×222/steps9/AID)
    #   一键正确建库(在包根 rswm_server_pack/ 下执行):
    #       bash server_run/build_correct.sh
    #   如确实要跑本脚本(仅调试用,产物不能用于论文),设 FORCE_WRONG_BUILDER=1。
    # =========================================================================
    if os.environ.get("FORCE_WRONG_BUILDER") != "1":
        print("=" * 70)
        print("[REFUSED] build_from_remotesense.py 不是论文管道(96x96/steps8 → 32帧糊图)。")
        print("          正确建库: bash server_run/build_correct.sh  (在包根执行)")
        print("          调试强制: FORCE_WRONG_BUILDER=1 python build_from_remotesense.py ...")
        print("=" * 70)
        sys.exit(2)
    sys.exit(main())
