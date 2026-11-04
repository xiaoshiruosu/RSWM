#!/usr/bin/env python3
"""
Multi-Direction Sliding Window Benchmark Builder V5

V5 Features:
- Random start position (not fixed at center)
- Random target position (anywhere in valid bounds)
- Single image resize (no mosaic tiling)
- Saves sliding window sequences as MP4 video files
- Maintains all required fields for video generation model training:
  - sliding_frames (N, H, W, C)
  - center_frame (H, W, C) - renamed but kept for compatibility (actually start_frame)
  - target_frame (H, W, C)
  - full_scene (H, W, C)
  - direction_embedding (64,)
  - trajectory (N, 2)
  - attrs (angle, overlap ratio, etc.)

Key Difference from V3:
- Start position can be anywhere in the image (not just center)
- Target position can also be anywhere (not constrained by fixed direction from center)
- More diverse training data for video generation models
"""

import os

# Keep native CPU libraries bounded when this file is executed standalone.
# The canonical wrapper applies the same ceiling before importing this module.
_NATIVE_THREAD_CAP = 8
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    try:
        _thread_count = int(os.environ.get(_thread_env, str(_NATIVE_THREAD_CAP)))
    except ValueError:
        _thread_count = _NATIVE_THREAD_CAP
    os.environ[_thread_env] = str(max(1, min(_thread_count, _NATIVE_THREAD_CAP)))

import sys
import json
import math
import time
import argparse
import numpy as np
import h5py
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PILImage = None
    PIL_AVAILABLE = False

try:
    from datasets import load_dataset as hf_load_dataset
    HF_AVAILABLE = True
except ImportError:
    hf_load_dataset = None
    HF_AVAILABLE = False


# ============================================================================
# HuggingFace Dataset Configuration
# ============================================================================

HF_DATASET_CONFIG = {
    "general": {
        "datasets": [
            {"name": "timm/eurosat-rgb", "split": "train", "feature": "image"},
            {"name": "blanchon/EuroSAT_RGB", "split": "train", "feature": "image"},
        ]
    },
    "water": {  # [v2.5] flood 更名 water
        "datasets": [
            {"name": "timm/eurosat-rgb", "split": "train", "feature": "image"},
        ]
    },
    "urban": {
        "datasets": [
            {"name": "blanchon/UC_Merced", "split": "train", "feature": "image"},
            {"name": "timm/eurosat-rgb", "split": "train", "feature": "image"},
        ]
    },
    "rural": {
        "datasets": [
            {"name": "timm/eurosat-rgb", "split": "train", "feature": "image"},
        ]
    },
}

HF_DATASETS_CACHE: Dict[str, object] = {}


# ============================================================================
# Configuration
# ============================================================================

def _envbool(name: str, default: bool = True) -> bool:
    """[v2.8.2] 统一的布尔环境变量解析。未设 → default;
    "0"/"false"/"no"/"off"/"" → False;其余 → True。"""
    v = os.environ.get(name)
    if v is None:
        return default
    t = v.strip().lower()
    if t == "":
        # [v2.9.1 N7] 空串按 default 处理,不当成 False —— `export X=` 静默关掉
        # 硬约束与"禁止静默放宽"的基调相反。
        return default
    return t not in ("0", "false", "no", "off")


MAX_RETRY_PER_SAMPLE = 20


@dataclass
class MultiDirectionConfig:
    """Configuration for multi-direction benchmark V5 (random positions)"""
    output_dir: str = "multi_direction_benchmark_v5"
    instances_per_scene: int = 500  # 每场景样本数，总样本数 = 此值 × 4
    scene_categories: List[str] = field(default_factory=lambda: ["general", "water", "urban", "rural"])
    
    full_scene_size: Tuple[int, int] = (512, 512)
    num_channels: int = 3
    
    window_size: Tuple[int, int] = (96, 96)
    steps_per_segment: int = 9   # [v2.8.1 P1] 8 会产出 T=15/22/29(非 4n+1),训练侧拒收
    
    min_segments: int = 2
    max_segments: int = 4
    
    embed_dim: int = 64
    num_freq_bands: int = 16
    
    # [v2.9] overlap 现在是【区间】而不是单侧上界:accept ⟺ min ≤ ov < max。
    # 主 benchmark = [0.00, 0.10)(拍板的硬约束);34 可学习性探针 = 预注册的
    # 高重叠带 [0.50, 0.95) —— 探针"简单"的另一半理由就是高重叠(窗口大幅重合
    # → 接近"把已见内容往一个方向延一点"),此前它只是【豁免】上界,实际 overlap
    # 完全随机(实测 0.51/0.72 都出现过),等于没有证据说明任务确实降到了 easy。
    min_overlap_ratio: float = 0.00
    max_overlap_ratio: float = 0.10
    # [v2.8.2] 0=连续角(canonical);4=只走 E/N/W/S(34 号可学习性判门用)
    angle_quantize: int = 0
    random_seed: int = 42
    
    # Video settings
    video_fps: int = 10
    video_codec: str = "mp4v"
    
    # V5: Position randomization settings
    position_margin: float = 0.1  # Margin from edge as fraction of scene size
    min_segment_length: float = 0.15  # Minimum segment length as fraction of scene size


# ============================================================================
# Video Writer
# ============================================================================

class VideoWriter:
    """Write frames to MP4 video file"""
    
    def __init__(self, output_path: str, fps: int = 10, codec: str = "mp4v"):
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
    
    def write_frames(self, frames: np.ndarray) -> bool:
        if not CV2_AVAILABLE:
            return False
        
        try:
            N, H, W, C = frames.shape
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (W, H))
            
            if not writer.isOpened():
                return False
            
            for i in range(N):
                frame = frames[i]
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.clip(0, 255).astype(np.uint8)
                
                if C == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                writer.write(frame_bgr)
            
            writer.release()
            return True
            
        except Exception as e:
            print(f"Warning: Failed to write video {self.output_path}: {e}")
            return False


# ============================================================================
# Scene Generator
# ============================================================================

class SceneGenerator:
    """Generate scenes from HuggingFace datasets using single image resize"""
    
    def __init__(self, config: MultiDirectionConfig, use_hf: bool = True):
        self.config = config
        # ``use_hf=True`` is the official-data contract.  It must never be
        # silently weakened to procedural scenes when the package, cache, or
        # an individual source image cannot be loaded.  ``use_hf=False``
        # remains the explicit standalone-demo path.
        self.use_hf = bool(use_hf)
        self.hf_datasets: Dict[str, object] = {}
        self.hf_indices: Dict[str, int] = {}
        np.random.seed(config.random_seed)
        
        if self.use_hf:
            print("[official] HF dataset required, synthetic fallback disabled")
            self._load_hf_datasets()
            missing = [
                scene for scene in self.config.scene_categories
                if scene not in self.hf_datasets
            ]
            if missing:
                raise RuntimeError(
                    "[official] HuggingFace dataset loading failed for scenes "
                    f"{missing}; synthetic fallback disabled")
    
    def _load_hf_datasets(self) -> None:
        global HF_DATASETS_CACHE

        if hf_load_dataset is None:
            raise RuntimeError(
                "[official] HuggingFace datasets package is unavailable; "
                "synthetic fallback disabled")
        
        print("\nLoading HuggingFace datasets...")
        
        for scene_category in self.config.scene_categories:
            if scene_category not in HF_DATASET_CONFIG:
                continue
            
            scene_config = HF_DATASET_CONFIG[scene_category]
            
            for ds_info in scene_config["datasets"]:
                ds_name = ds_info["name"]
                ds_split = ds_info.get("split", "train")
                cache_key = f"{ds_name}:{ds_split}"
                
                if cache_key in HF_DATASETS_CACHE:
                    self.hf_datasets[scene_category] = {
                        "dataset": HF_DATASETS_CACHE[cache_key],
                        "feature": ds_info["feature"],
                        "name": ds_name,
                    }
                    self.hf_indices[scene_category] = 0
                    print(f"  {scene_category}: Using cached {ds_name}")
                    break

                try:
                    print(f"  {scene_category}: Loading {ds_name}...")
                    ds = hf_load_dataset(ds_name, split=ds_split, trust_remote_code=True)
                    HF_DATASETS_CACHE[cache_key] = ds
                    self.hf_datasets[scene_category] = {
                        "dataset": ds,
                        "feature": ds_info["feature"],
                        "name": ds_name,
                    }
                    self.hf_indices[scene_category] = 0
                    print(f"  {scene_category}: Loaded {len(ds)} samples")
                    break
                except Exception as e:
                    print(f"  {scene_category}: Failed to load {ds_name}: {e}")
        
        print()
    
    def _to_numpy_image(self, x) -> np.ndarray:
        if PIL_AVAILABLE and PILImage is not None and isinstance(x, PILImage.Image):
            arr = np.asarray(x)
        else:
            arr = np.asarray(x)
        
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] > 3:
            arr = arr[:, :, :3]
        
        return arr.astype(np.float32)
    
    def _resize_image(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        H, W = target_size
        h, w = img.shape[:2]
        
        if h == H and w == W:
            return img
        
        if CV2_AVAILABLE:
            interpolation = cv2.INTER_LANCZOS4 if (H > h or W > w) else cv2.INTER_AREA
            return cv2.resize(img, (W, H), interpolation=interpolation)
        else:
            h_idx = np.linspace(0, h - 1, H).astype(int)
            w_idx = np.linspace(0, w - 1, W).astype(int)
            return img[np.ix_(h_idx, w_idx)]
    
    def get_dataset_source(self, scene_category: str) -> str:
        if self.use_hf:
            if scene_category not in self.hf_datasets:
                raise RuntimeError(
                    "[official] no HuggingFace source loaded for "
                    f"scene={scene_category!r}; synthetic fallback disabled")
            source = str(self.hf_datasets[scene_category].get("name", "")).strip()
            if not source:
                raise RuntimeError(
                    "[official] loaded HuggingFace source has no dataset name for "
                    f"scene={scene_category!r}")
            return source
        return "standalone_demo"

    def generate_scene(self, scene_category: str, scene_id: int) -> np.ndarray:
        # [v2.3 修复 2026-07-23] 旧行为 np.random.seed(random_seed + scene_id)
        # 是跨场景重复样本的根因:scene_id 是【块内局部序号】,四个场景块的
        # 同偏移样本把全局 RNG 重置到同一状态 → num_segments/waypoints/裁剪
        # 偏移全部成为 scene_id 的确定函数、与场景无关 → 同偏移样本轨迹逐位
        # 相同(叠加取图池共享时连视频帧都相同;Track B 尸检 top-15 的
        # 121.50°/0.458/0.098 三重组即此指纹)。修复 = 种子同时吃场景名:
        # 每 (场景, 序号) 一条独立确定性流。⚠️ 用旧行为构建的一切数据集
        # (含旧 AID 与 D2 修正跑)都带跨块重复,用
        # server_run/check_benchmark_duplicates.py 判决。
        import zlib
        np.random.seed(
            (int(self.config.random_seed) * 1000003
             + zlib.crc32(f"{scene_category}:{int(scene_id)}".encode()))
            % (2 ** 32)
        )
        H, W = self.config.full_scene_size
        C = self.config.num_channels
        
        if self.use_hf:
            if scene_category not in self.hf_datasets:
                raise RuntimeError(
                    "[official] no HuggingFace source loaded for "
                    f"scene={scene_category!r}; synthetic fallback disabled")
            scene = self._load_scene_from_hf(scene_category, scene_id, H, W)
            if scene is not None:
                return scene
            raise RuntimeError(
                "[official] HuggingFace sample loading returned no image for "
                f"scene={scene_category!r}, scene_id={scene_id}; "
                "synthetic fallback disabled")
        
        if scene_category in ("flood", "water"):  # [v2.5] flood 更名 water
            scene = self._generate_flood_scene(H, W, C)
        elif scene_category == "urban":
            scene = self._generate_urban_scene(H, W, C)
        elif scene_category == "rural":
            scene = self._generate_rural_scene(H, W, C)
        else:
            scene = self._generate_general_scene(H, W, C)
        
        return scene.astype(np.float32)
    
    def _load_scene_from_hf(self, scene_category: str, scene_id: int, H: int, W: int) -> Optional[np.ndarray]:
        try:
            ds_info = self.hf_datasets[scene_category]
            dataset = ds_info["dataset"]
            feature_name = ds_info["feature"]
            
            n = len(dataset)
            if n == 0:
                return None
            
            idx = scene_id % n
            example = dataset[int(idx)]
            img = self._to_numpy_image(example[feature_name])
            scene = self._resize_image(img, (H, W))
            
            if scene.max() <= 1.0:
                scene = scene * 255.0
            
            return scene.clip(0, 255).astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Failed to load HF scene: {e}")
            return None
    
    def _generate_perlin_like(self, H: int, W: int) -> np.ndarray:
        result = np.zeros((H, W))
        for octave in range(4):
            freq = 2 ** octave
            amp = 1 / freq
            small_h = max(8, H // (32 // freq))
            small_w = max(8, W // (32 // freq))
            noise = np.random.rand(small_h, small_w)
            
            x = np.linspace(0, small_w - 1, W)
            y = np.linspace(0, small_h - 1, H)
            xi, yi = np.meshgrid(x, y)
            xi_floor = np.floor(xi).astype(int)
            yi_floor = np.floor(yi).astype(int)
            xi_ceil = np.minimum(xi_floor + 1, small_w - 1)
            yi_ceil = np.minimum(yi_floor + 1, small_h - 1)
            
            dx = xi - xi_floor
            dy = yi - yi_floor
            
            upscaled = (
                noise[yi_floor, xi_floor] * (1 - dx) * (1 - dy) +
                noise[yi_floor, xi_ceil] * dx * (1 - dy) +
                noise[yi_ceil, xi_floor] * (1 - dx) * dy +
                noise[yi_ceil, xi_ceil] * dx * dy
            )
            result += upscaled * amp
        
        return (result - result.min()) / (result.max() - result.min() + 1e-8)
    
    def _generate_general_scene(self, H, W, C) -> np.ndarray:
        texture = self._generate_perlin_like(H, W)
        texture2 = self._generate_perlin_like(H, W)
        combined = texture * 0.7 + texture2 * 0.3
        scene = np.stack([combined * 180 + 40, combined * 160 + 50, combined * 140 + 30], axis=-1)
        return scene.clip(0, 255)
    
    def _generate_flood_scene(self, H, W, C) -> np.ndarray:
        texture = self._generate_perlin_like(H, W)
        water_mask = texture < 0.4
        scene = np.stack([texture * 120 + 60, texture * 140 + 80, texture * 80 + 40], axis=-1)
        scene[water_mask, 0] *= 0.3
        scene[water_mask, 1] *= 0.5
        scene[water_mask, 2] = np.minimum(scene[water_mask, 2] + 100, 255)
        return scene.clip(0, 255)
    
    def _generate_urban_scene(self, H, W, C) -> np.ndarray:
        x = np.linspace(0, 16 * np.pi, W)
        y = np.linspace(0, 16 * np.pi, H)
        xx, yy = np.meshgrid(x, y)
        texture = (np.sin(xx) * np.sin(yy) + 1) / 2
        noise = np.random.rand(H, W) * 0.2
        combined = texture * 0.8 + noise
        scene = np.stack([combined * 150 + 50, combined * 140 + 50, combined * 160 + 40], axis=-1)
        return scene.clip(0, 255)
    
    def _generate_rural_scene(self, H, W, C) -> np.ndarray:
        texture = self._generate_perlin_like(H, W)
        x = np.linspace(0, 32 * np.pi, W)
        stripes = (np.sin(x) + 1) / 2
        stripes = np.tile(stripes, (H, 1))
        combined = texture * 0.5 + stripes * 0.5
        scene = np.stack([combined * 80 + 40, combined * 180 + 60, combined * 60 + 30], axis=-1)
        return scene.clip(0, 255)


# ============================================================================
# Direction Embedding
# ============================================================================

class DirectionEmbedder:
    def __init__(self, embed_dim: int = 64, num_freq_bands: int = 16):
        self.embed_dim = embed_dim
        self.num_freq_bands = num_freq_bands
        self.freq_scales = np.logspace(0, np.log10(1000), num_freq_bands)
        
        base_dim = num_freq_bands * 2 + num_freq_bands + 2
        np.random.seed(42)
        self.projection = np.random.randn(base_dim, embed_dim) / np.sqrt(base_dim)
    
    def encode(self, angle_degrees: float, magnitude: float = 1.0) -> Dict:
        angle_rad = np.deg2rad(angle_degrees % 360)
        
        angle_enc = []
        for freq in self.freq_scales:
            angle_enc.extend([np.sin(angle_rad * freq), np.cos(angle_rad * freq)])
        
        mag_norm = np.clip(magnitude / 10.0, 0, 1)
        mag_enc = [np.sin(mag_norm * np.pi * freq) for freq in self.freq_scales]
        
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        base = np.array(angle_enc + mag_enc + [dx, dy], dtype=np.float32)
        embedding = np.dot(base, self.projection).astype(np.float32)
        
        return {
            "embedding": embedding,
            "angle_degrees": angle_degrees % 360,
            "angle_radians": angle_rad,
            "direction_vector": (float(dx), float(dy))
        }


# ============================================================================
# Random Position Path Generator (V5 Key Feature)
# ============================================================================

class RandomPositionPathGenerator:
    """
    Generate multi-directional paths with random start and waypoint positions.
    Unlike V3, the start position is NOT fixed at the center.
    """
    
    def __init__(self, config: MultiDirectionConfig):
        self.config = config
    
    def _get_valid_bounds(self) -> Tuple[int, int, int, int]:
        """Get valid bounds for window center positions"""
        H, W = self.config.full_scene_size
        win_h, win_w = self.config.window_size
        margin_h = int(H * self.config.position_margin)
        margin_w = int(W * self.config.position_margin)
        
        # Ensure window stays fully within scene
        min_cy = max(win_h // 2, margin_h)
        max_cy = min(H - win_h // 2, H - margin_h)
        min_cx = max(win_w // 2, margin_w)
        max_cx = min(W - win_w // 2, W - margin_w)
        
        return min_cy, max_cy, min_cx, max_cx
    
    def _random_position(self) -> Tuple[int, int]:
        """Generate a random valid position"""
        min_cy, max_cy, min_cx, max_cx = self._get_valid_bounds()
        cy = np.random.randint(min_cy, max_cy + 1)
        cx = np.random.randint(min_cx, max_cx + 1)
        return (cy, cx)
    
    def _generate_next_waypoint(self, current_pos: Tuple[int, int], prev_angle: Optional[float] = None) -> Tuple[Tuple[int, int], float]:
        """Generate next waypoint with sufficient distance from current position"""
        H, W = self.config.full_scene_size
        min_cy, max_cy, min_cx, max_cx = self._get_valid_bounds()
        
        min_distance = max(H, W) * self.config.min_segment_length
        _q = int(getattr(self.config, "angle_quantize", 0) or 0)

        if _q == 4:
            # =================================================================
            # [v2.9.1 P0] 四方向 + overlap 带:位移【从带反解】,不靠拒绝采样。
            #
            # 轴向运动下 overlap = 1 − Δ/窗口该轴边长(_calculate_overlap 是
            # intersection/area1,一个轴无位移则该因子为 1)。于是
            #     ov ∈ [lo, hi)  ⟺  Δ ∈ ((1−hi)·extent, (1−lo)·extent]
            # 而 extent 对 E/W 是 win_w、对 N/S 是 win_h。
            #
            # 之前靠"随机角 + 随机距离 + 落带就收"的拒绝采样,在 128×222 窗口 +
            # 带 [0.50,0.95) 下与 min_segment_length=0.15 直接互斥:
            # N/S 需要 Δy ≤ 64,而 min_distance = 0.15×512 = 76.8(接受下限
            # 61.44)—— 可行窗口只有 3 个像素。实测逐方向接受率
            # E 0.396 / W 0.426 / N 0.013 / S 0.009,20 条样本里 N+S 期望 0.53 条,
            # 判门的"四方向各 ≥2"必然 FAIL → 整个矩阵一条都跑不了。
            #
            # 反解 + 轮转(而非纯随机)同时解决可行性与方向均衡。
            # min_segment_length 是给主 benchmark 的通用下限,在这里不适用:
            # 探针要的恰恰是"小位移 = 高重叠 = 简单任务"。
            # =================================================================
            win_h, win_w = self.config.window_size
            lo = float(getattr(self.config, "min_overlap_ratio", 0.0) or 0.0)
            hi = float(getattr(self.config, "max_overlap_ratio", 1.0) or 1.0)
            if not hasattr(self, "_dir_rr"):
                self._dir_rr = 0
            # 轮转起点错开,保证跨样本四方向均衡
            order = [(self._dir_rr + k) % 4 for k in range(4)]
            self._dir_rr = (self._dir_rr + 1) % 4
            AXIS = {0: (0, 1, win_w), 90: (-1, 0, win_h),
                    180: (0, -1, win_w), 270: (1, 0, win_h)}
            for _k in order:
                ang = float(_k * 90)
                if prev_angle is not None and abs(
                        ((ang - prev_angle + 180.0) % 360.0) - 180.0) < 1e-9:
                    continue                      # 禁止与上一段同向
                dy, dx, extent = AXIS[int(ang)]
                d_lo = (1.0 - hi) * extent        # 开区间下界(ov < hi)
                d_hi = (1.0 - lo) * extent        # 闭区间上界(ov >= lo)
                # 该方向在合法中心范围内还剩多少余量
                room = ((max_cy - current_pos[0]) if dy > 0 else
                        (current_pos[0] - min_cy) if dy < 0 else
                        (max_cx - current_pos[1]) if dx > 0 else
                        (current_pos[1] - min_cx))
                d_hi = min(d_hi, float(room))
                # 取整后仍需 d_lo < Δ <= d_hi 且 Δ >= 1
                lo_i, hi_i = int(np.floor(d_lo)) + 1, int(np.floor(d_hi))
                lo_i = max(lo_i, 1)
                if lo_i > hi_i:
                    continue                      # 该方向余量不足,换下一个
                dist = int(np.random.randint(lo_i, hi_i + 1))
                new_cy = int(current_pos[0] + dy * dist)
                new_cx = int(current_pos[1] + dx * dist)
                return (new_cy, new_cx), ang
            # 四个方向都放不下(起点贴角 + 带很窄)→ 落到下面的通用兜底

        # Try to generate a valid waypoint
        for _ in range(50):  # Max attempts
            # Generate random angle (avoid exact reversal if prev_angle exists)
            if prev_angle is not None:
                angle_offset = np.random.uniform(30, 330)
                angle = (prev_angle + angle_offset) % 360
            else:
                angle = np.random.uniform(0, 360)
            # [v2.8.2] angle_quantize=4 → 只走 E/N/W/S 四个基本方向。
            # 34 号判门的 PBS 头注释宣称探针是"单段直线 + 四个基本方向"的
            # 【最简单可能的方向任务】,但此前角度一直是 uniform(0,360) 连续采样,
            # 探针跑的其实是和主矩阵同分布的任务 —— 判门因此既证明不了
            # "可学"(PASS 时论文那句"平凡难度正对照"没有实验支撑),也解释不了
            # "不可学"(FAIL 时会把人指向管线 bug,而真因是难度从没降过)。
            if _q > 0:
                _step = 360.0 / _q
                angle = (round(angle / _step) % _q) * _step
                if prev_angle is not None and abs(
                        ((angle - prev_angle + 180.0) % 360.0) - 180.0) < 1e-9:
                    angle = (angle + _step) % 360.0   # 量化后撞回原方向则转一格
            
            # Generate random distance
            max_distance = max(H, W) * 0.4
            distance = np.random.uniform(min_distance, max_distance)
            
            # Calculate new position
            angle_rad = np.deg2rad(angle)
            dx = np.cos(angle_rad)
            dy = -np.sin(angle_rad)  # Negative because y increases downward

            # [v2.8.2] 量化模式下把"数学上为零"的分量精确置零。
            # 否则 sin(180°)=-1.22e-16 → current_cy + dy*distance = 256 - 1.2e-14,
            # int() 向零截断给 255 —— 1e-14 的浮点残差被放大成整整 1 像素偏移,
            # 重算出来的 actual_angle 变成 179.13° 而不是 180°,四方向约束形同虚设。
            if _q > 0:
                if abs(dx) < 1e-9:
                    dx = 0.0
                if abs(dy) < 1e-9:
                    dy = 0.0

            new_cx = int(current_pos[1] + dx * distance)
            new_cy = int(current_pos[0] + dy * distance)
            
            # Clamp to valid bounds
            new_cy = np.clip(new_cy, min_cy, max_cy)
            new_cx = np.clip(new_cx, min_cx, max_cx)
            
            # Check actual distance after clamping
            actual_distance = np.sqrt((new_cy - current_pos[0])**2 + (new_cx - current_pos[1])**2)
            if actual_distance >= min_distance * 0.8:
                # Recalculate actual angle based on final positions
                actual_angle = np.degrees(np.arctan2(-(new_cy - current_pos[0]), new_cx - current_pos[1])) % 360
                return (new_cy, new_cx), actual_angle
        
        # Fallback: random position if no valid waypoint found
        if _q > 0:
            # [v2.8.2] 量化模式的兜底也必须轴向,否则 50 次尝试全失败时会漏出
            # 一个任意角度,而这恰好发生在贴边的起点上(最需要约束的场合)。
            # 选四个基本方向里余量最大的那个,直接走到该方向的合法边界。
            _cands = []
            for _a, (_dy, _dx) in ((0.0, (0, 1)), (90.0, (-1, 0)),
                                   (180.0, (0, -1)), (270.0, (1, 0))):
                _ty = max_cy if _dy > 0 else (min_cy if _dy < 0 else current_pos[0])
                _tx = max_cx if _dx > 0 else (min_cx if _dx < 0 else current_pos[1])
                _room = abs(_ty - current_pos[0]) + abs(_tx - current_pos[1])
                _cands.append((_room, _a, (int(_ty), int(_tx))))
            _room, _a, _pos = max(_cands, key=lambda z: z[0])
            return _pos, _a
        new_pos = self._random_position()
        actual_angle = np.degrees(np.arctan2(-(new_pos[0] - current_pos[0]), new_pos[1] - current_pos[1])) % 360
        return new_pos, actual_angle
    
    def generate_path(self, num_segments: int) -> Dict:
        """
        Generate a multi-directional path with random start position.
        
        Args:
            num_segments: Number of direction segments
        
        Returns:
            Dictionary containing waypoints, angles, etc.
        """
        # Random start position (V5 key feature)
        start_pos = self._random_position()
        
        waypoints = [start_pos]
        segment_angles = []
        current_pos = start_pos
        prev_angle = None
        
        for seg_idx in range(num_segments):
            next_pos, angle = self._generate_next_waypoint(current_pos, prev_angle)
            waypoints.append(next_pos)
            segment_angles.append(angle)
            current_pos = next_pos
            prev_angle = angle
        
        return {
            "waypoints": waypoints,
            "segment_angles": segment_angles,
            "num_segments": num_segments,
            "steps_per_segment": self.config.steps_per_segment,
            "start_position": start_pos,
            "end_position": waypoints[-1],
        }


# ============================================================================
# Multi-Direction Extractor
# ============================================================================

class MultiDirectionExtractor:
    def __init__(self, config: MultiDirectionConfig):
        self.config = config
        self.direction_embedder = DirectionEmbedder(
            embed_dim=config.embed_dim,
            num_freq_bands=config.num_freq_bands
        )
    
    def extract_multi_direction_sequence(self, full_scene: np.ndarray, path_info: Dict) -> Dict:
        H, W, C = full_scene.shape
        win_h, win_w = self.config.window_size
        half_h, half_w = win_h // 2, win_w // 2
        steps_per_segment = path_info["steps_per_segment"]
        
        waypoints = path_info["waypoints"]
        segment_angles = path_info["segment_angles"]
        num_segments = path_info["num_segments"]
        
        all_frames = []
        trajectory = []
        segment_embeddings = []
        
        for seg_idx in range(num_segments):
            start_wp = waypoints[seg_idx]
            end_wp = waypoints[seg_idx + 1]
            angle = segment_angles[seg_idx]
            
            dir_result = self.direction_embedder.encode(angle)
            segment_embeddings.append(dir_result["embedding"])
            
            # [v2.8 P0 ②] 段界去重:除第一段外跳过 step=0 —— 它与上一段的
            # 末帧是同一窗口中心(同一张图)。旧行为每段都含首尾 → T=18/27/36
            # 里有 K-1 个重复帧,训练侧 4n+1 对齐再取【前】17/25/33 帧,等于
            # 保留重复帧却丢掉真实终点;评分端又按去重口径打分 —— 训练/生成/
            # 评分三条时间轴互不一致。现在源头就产 17/25/33:无重复帧、含真实
            # 终点、且恰好满足 4n+1(17=4·4+1, 25=4·6+1, 33=4·8+1),训练无需
            # 再截断。
            # [v2.8 ②] 4n+1 契约:去重后 T = steps + (K-1)(steps-1)。
            # 只有 steps=9 时 T ∈ {17,25,33} 恰好满足 4n+1(VAE 时间下采样 4×,
            # 训练侧 _collate_fn 会把非 4n+1 的帧数【截断】—— 那正是本轮修掉的
            # 尾截断 bug 的入口)。steps=8(dataclass 默认)去重后是 15/22/29,
            # 只有 29 合规 → 15/22 会被截成 13/21 丢尾。canonical 一律 --steps 9。
            if seg_idx == 0 and not getattr(self, "_steps_4n1_checked", False):
                self._steps_4n1_checked = True
                for _K in range(self.config.min_segments, self.config.max_segments + 1):
                    _T = steps_per_segment + (_K - 1) * (steps_per_segment - 1)
                    if (_T - 1) % 4 != 0:
                        # [v2.8.2 N6] 由 warn 改 raise。warn-only 与本轮"禁止静默
                        # 放宽阈值"的基调不一致:非 4n+1 落盘后,训练侧 _collate_fn
                        # 要么截尾(15→13、22→21,正是 v2.8 要修的那类 bug 的入口)
                        # 要么直接拒收整个数据集 —— 无论哪种,这批数据都是白建的。
                        _msg = (f"steps_per_segment={steps_per_segment} 下 K={_K} 的"
                                f"帧数 T={_T} 不满足 4n+1(VAE 时间下采样 4×)。"
                                f"canonical 请用 --steps 9(K=1..4 → T=9/17/25/33)。")
                        if _envbool("RSWM_ALLOW_NON_4N1", False):
                            print(f"    [WARN] {_msg} —— RSWM_ALLOW_NON_4N1 已放行,"
                                  f"产物不得用于训练。")
                        else:
                            raise ValueError(
                                f"[4n+1] {_msg} 逃生口 RSWM_ALLOW_NON_4N1=1(仅调试)。")
            _step0 = 0 if seg_idx == 0 else 1
            for step in range(_step0, steps_per_segment):
                t = step / (steps_per_segment - 1) if steps_per_segment > 1 else 0
                
                cy = int(start_wp[0] + t * (end_wp[0] - start_wp[0]))
                cx = int(start_wp[1] + t * (end_wp[1] - start_wp[1]))
                
                h_start = max(0, cy - half_h)
                h_end = min(H, cy + half_h)
                w_start = max(0, cx - half_w)
                w_end = min(W, cx + half_w)
                
                window = full_scene[h_start:h_end, w_start:w_end, :]
                
                if window.shape[0] != win_h or window.shape[1] != win_w:
                    window = self._resize_window(window, (win_h, win_w))
                
                all_frames.append(window)
                trajectory.append((cy, cx))
        
        all_frames = np.stack(all_frames, axis=0)
        segment_embeddings = np.stack(segment_embeddings, axis=0)
        
        # Combined direction embedding
        direction_embedding = segment_embeddings.mean(axis=0)
        
        start_bounds = self._get_bounds(waypoints[0], (win_h, win_w), (H, W))
        end_bounds = self._get_bounds(waypoints[-1], (win_h, win_w), (H, W))
        overlap_ratio = self._calculate_overlap(start_bounds, end_bounds)

        # [RSRoll-Bench P0#7] 逐帧 step-overlap 审计。
        #
        # 为什么必须加：上面这个 overlap_ratio 只比较**第一个和最后一个**窗口，
        # 于是"低重叠"这句话只对首尾成立。模型实际消费的是相邻帧构成的链，
        # 而按真实段长分布（L ~ U[0.15,0.40]×512、steps=9）复算，
        # 相邻帧重叠约 0.87、每步约 17.6 px（p95 24.8 px）——是一条**稠密高重叠链**。
        # 不把它记录下来，就无法回答"模型到底依赖多大的帧间重叠"这个核心问题；
        # 也无法解释为什么 Farnebäck 估计器在这里还能用、换到低重叠档就崩。
        step_ov = []
        for _i in range(1, len(trajectory)):
            _b0 = self._get_bounds(trajectory[_i - 1], (win_h, win_w), (H, W))
            _b1 = self._get_bounds(trajectory[_i], (win_h, win_w), (H, W))
            step_ov.append(self._calculate_overlap(_b0, _b1))
        if step_ov:
            _arr = np.asarray(step_ov, dtype=np.float64)
            step_overlap_stats = {
                "min": float(_arr.min()), "p50": float(np.median(_arr)),
                "mean": float(_arr.mean()), "max": float(_arr.max()),
            }
        else:
            step_overlap_stats = {"min": None, "p50": None, "mean": None, "max": None}
        
        return {
            # Required fields for video generation training
            "sliding_frames": all_frames,
            "center_frame": all_frames[0],  # First frame (start position)
            "target_frame": all_frames[-1],  # Last frame (end position)
            "full_scene": full_scene,
            "direction_embedding": direction_embedding,
            "trajectory": np.array(trajectory),
            
            # Additional fields
            "segment_embeddings": segment_embeddings,
            "segment_angles": np.array(segment_angles, dtype=np.float32),
            "waypoints": waypoints,
            "num_segments": num_segments,
            "total_frames": len(all_frames),
            "start_bounds": start_bounds,
            "end_bounds": end_bounds,
            "overlap_ratio": overlap_ratio,
            # [RSRoll-Bench P0#7] 首尾重叠之外，把**帧间**重叠也落盘：
            # 逐帧序列进 H5（可事后重扫），统计量进 overlap_audit.json（可直接读）。
            "step_overlaps": np.asarray(step_ov, dtype=np.float32),
            "step_overlap_stats": step_overlap_stats,
            "start_position": path_info["start_position"],
            "end_position": path_info["end_position"],
        }
    
    def _resize_window(self, window: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        if CV2_AVAILABLE:
            return cv2.resize(window, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        else:
            h, w = window.shape[:2]
            h_idx = np.linspace(0, h - 1, target_size[0]).astype(int)
            w_idx = np.linspace(0, w - 1, target_size[1]).astype(int)
            return window[np.ix_(h_idx, w_idx)]
    
    def _get_bounds(self, center: Tuple[int, int], window_size: Tuple[int, int], scene_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        cy, cx = center
        win_h, win_w = window_size
        H, W = scene_size
        return (max(0, cy - win_h // 2), min(H, cy + win_h // 2), max(0, cx - win_w // 2), min(W, cx + win_w // 2))
    
    def _calculate_overlap(self, bounds1: Tuple[int, int, int, int], bounds2: Tuple[int, int, int, int]) -> float:
        h1_s, h1_e, w1_s, w1_e = bounds1
        h2_s, h2_e, w2_s, w2_e = bounds2
        
        h_inter_s = max(h1_s, h2_s)
        h_inter_e = min(h1_e, h2_e)
        w_inter_s = max(w1_s, w2_s)
        w_inter_e = min(w1_e, w2_e)
        
        if h_inter_s >= h_inter_e or w_inter_s >= w_inter_e:
            return 0.0
        
        intersection = (h_inter_e - h_inter_s) * (w_inter_e - w_inter_s)
        area1 = (h1_e - h1_s) * (w1_e - w1_s)
        return intersection / area1 if area1 > 0 else 0.0


# ============================================================================
# Benchmark Builder V5
# ============================================================================

class MultiDirectionBenchmarkBuilder:
    """Build benchmark with random start/end positions and MP4 output"""
    
    def __init__(self, config: MultiDirectionConfig, use_hf: bool = True):
        self.config = config
        self.use_hf = use_hf
        self.scene_generator = SceneGenerator(config, use_hf=use_hf)
        self.path_generator = RandomPositionPathGenerator(config)  # V5: Random positions
        self.extractor = MultiDirectionExtractor(config)
        
        self.stats = {
            "total_instances": 0,
            "valid_instances": 0,
            "instances_per_scene": {},
            "overlap_ratios": [],
            "step_overlap_min": [],      # [RSRoll-Bench P0#7] 帧间重叠汇总
            "step_overlap_p50": [],
            "num_segments_distribution": [],
            "videos_saved": 0,
            "processing_time": 0.0
        }
    
    def build(self) -> Dict:
        start_time = time.time()
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        videos_dir = output_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        print("\n" + "=" * 70)
        print("Multi-Direction Benchmark Builder V5 (Random Start/End Positions)")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  - Output directory: {self.config.output_dir}")
        print(f"  - Instances per scene: {self.config.instances_per_scene}")
        print(f"  - Total instances: {self.config.instances_per_scene * len(self.config.scene_categories)}")
        print(f"  - Direction segments: {self.config.min_segments}-{self.config.max_segments}")
        print(f"  - Position margin: {self.config.position_margin:.0%}")
        print(f"  - Min segment length: {self.config.min_segment_length:.0%}")
        print(f"  - Data source: {'HuggingFace' if self.use_hf else 'Synthetic'}")
        print()
        
        np.random.seed(self.config.random_seed)
        global_idx = 0
        all_scene_cats = []   # (idx, scene_category, num_segments) for index building
        total_written = 0
        valid_count = 0

        h5_path = output_dir / "benchmark.h5"
        print(f"\nStreaming samples to {h5_path}")

        # [v2.8.2 N2] try/finally 保证异常路径上也落盘 overlap 审计 ——
        # 构建崩掉的那一次恰恰是最需要审计证据的一次。
        try:
          with h5py.File(h5_path, 'w') as f:
            # Write metadata header
            meta = f.create_group('metadata')
            meta.attrs['version'] = '5.0.0'
            meta.attrs['benchmark_type'] = 'multi_direction_v5_random_position'
            meta.attrs['scene_categories'] = self.config.scene_categories
            meta.attrs['full_scene_size'] = self.config.full_scene_size
            meta.attrs['window_size'] = self.config.window_size
            meta.attrs['embed_dim'] = self.config.embed_dim
            meta.attrs['min_segments'] = self.config.min_segments
            meta.attrs['max_segments'] = self.config.max_segments
            meta.attrs['steps_per_segment'] = self.config.steps_per_segment
            # [v2.9.2] 生成口径必须随产物走 —— 否则事后无法证明"这批数据确实是
            # 预注册区间/四方向"(34 判门的三条腿有两条在这里)。
            meta.attrs['min_overlap_ratio'] = float(
                getattr(self.config, 'min_overlap_ratio', 0.0) or 0.0)
            meta.attrs['max_overlap_ratio'] = float(self.config.max_overlap_ratio)
            meta.attrs['angle_quantize'] = int(
                getattr(self.config, 'angle_quantize', 0) or 0)
            meta.attrs['min_segment_length'] = float(self.config.min_segment_length)
            meta.attrs['position_margin'] = self.config.position_margin
            meta.attrs['min_segment_length'] = self.config.min_segment_length
            meta.attrs['video_fps'] = self.config.video_fps
            meta.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            data = f.create_group('data')

            for scene in self.config.scene_categories:
                print(f"Processing scene: {scene}")
                samples, global_idx = self._process_scene(scene, videos_dir, global_idx)
                self.stats["instances_per_scene"][scene] = len(samples)
                print(f"  Generated {len(samples)} samples")

                # Stream write: flush each sample immediately
                for sample in samples:
                    idx = total_written
                    self._write_sample_to_hdf5(data, idx, sample)
                    all_scene_cats.append((idx, sample['scene_category'], sample['num_segments']))
                    if sample['overlap_ratio'] < self.config.max_overlap_ratio:
                        valid_count += 1
                    total_written += 1
                    if total_written % 20 == 0:
                        f.flush()
                        print(f"    Progress: {total_written}")

                # Release memory after each scene
                del samples

            # Build indices
            index = f.create_group('index')
            for sc in self.config.scene_categories:
                indices = [i for i, s, _ in all_scene_cats if s == sc]
                index.create_dataset(sc, data=np.array(indices))

            seg_index = f.create_group('index_by_segments')
            for n_seg in range(self.config.min_segments, self.config.max_segments + 1):
                indices = [i for i, _, ns in all_scene_cats if ns == n_seg]
                if indices:
                    seg_index.create_dataset(f'segments_{n_seg}', data=np.array(indices))

            meta.attrs['total_instances'] = total_written

        except BaseException:
            # 失败路径:只写审计(不做最终断言 —— 断言的前提是构建走完),
            # 审计本身写失败也不许掩盖原始异常。
            try:
                self._write_overlap_audit(output_dir, h5_path=None, assert_final=False)
            except Exception as _e:
                print(f"  [overlap] ⚠ 失败路径上写审计也出错: {_e}")
            # [v2.8.2 N3] 半截产物改名标废。build_benchmark 下游有
            # glob("*.h5") 兜底,留着 benchmark.h5 会被当成正常产物捡走;
            # 直接删又丢掉排查线索 —— 改名两头都占。
            try:
                if h5_path.is_file():
                    _bad = h5_path.with_suffix(".h5.FAILED")
                    if _bad.exists():
                        _bad.unlink()
                    h5_path.rename(_bad)
                    print(f"  [build] 构建失败,半截产物已标废 → {_bad}")
            except Exception as _e:
                print(f"  [build] ⚠ 标废半截产物失败: {_e}")
            raise

        print(f"Saved HDF5 to {h5_path}")
        self._save_metadata_json(output_dir / "metadata.json")

        self._write_overlap_audit(output_dir, h5_path)

        self.stats["total_instances"] = total_written
        self.stats["valid_instances"] = valid_count
        self.stats["processing_time"] = time.time() - start_time
        
        return self.stats
    
    def _write_overlap_audit(self, output_dir, h5_path=None, assert_final=True):
        """[v2.8 ⑤ / v2.8.2 N2·N3] overlap 审计落盘 + 最终硬断言。

        N2:必须在【异常路径上也写】。overlap 耗尽 / 语义门耗尽这两种失败恰恰
            是最需要这份审计来定位的,旧代码却因为异常从 _process_scene 抛出、
            审计写在 build() 尾部而什么都拿不到(磁盘上只剩半截 h5 + videos/)。
        N3:判废时把 h5 删掉。旧代码断言在产物已完整落盘之后才炸,磁盘上留着
            一个结构完整、metadata.json 齐全的 benchmark.h5 —— 断点续跑/人工
            复用时它看起来完全正常。
        """
        from pathlib import Path as _P
        # [v2.9.1 N9] 用副本,不原地改 self._overlap_audit —— 正常路径与异常
        # 路径若都调到(将来很容易),二次写入会把 accept_rate 之类算重。
        import copy as _copy
        _au = _copy.deepcopy(getattr(self, "_overlap_audit", None))
        if _au is not None:
            ov = np.asarray(_au["overlap_accepted"], dtype=np.float64)
            _au["n_accepted"] = int(ov.size)
            if ov.size:
                _au["overlap_stats"] = {
                    "min": float(ov.min()),
                    "max": float(ov.max()), "p50": float(np.percentile(ov, 50)),
                    "p90": float(np.percentile(ov, 90)),
                    "p99": float(np.percentile(ov, 99)), "mean": float(ov.mean())}
            for _sc, _v in _au["per_scene"].items():
                # [v2.8.2 N1] accept_rate 只用【源图级】计数。旧分母把
                # reject_overlap(轨迹级,每张源图最多 traj_max_tries 条)和
                # accepted/reject_semantic(源图级)混在一起 —— 实测 8 个样本
                # 全在第一张源图上接受(源图级 100%),accept_rate 却输出 0.47,
                # 会被读成"一半源图不合格"。而这份 JSON 正是拍板要的验收证据。
                # A best-candidate fallback was first counted as a semantic
                # reject and is then also the accepted source; subtract that
                # overlap once so the source-image denominator stays exact.
                _src_tot = (_v["accepted"] + _v["reject_semantic"]
                            + _v.get("source_exhausted", 0)
                            - _v.get("semantic_fallback", 0))
                _v["accept_rate"] = (_v["accepted"] / _src_tot) if _src_tot else None
                _v["overlap_traj_rejects"] = _v["reject_overlap"]   # 轨迹级,另列
                _v["accept_rate_denominator"] = "source_images"
            _ap = output_dir / "overlap_audit.json"
            _ap.write_text(json.dumps(_au, indent=2, ensure_ascii=False))
            print(f"  [overlap] 审计 → {_ap}")
            print(f"  [overlap] 尝试 {_au['n_attempts']} / 拒(overlap) "
                  f"{_au['n_reject_overlap']} / 拒(语义) {_au['n_reject_semantic']} / "
                  f"源图耗尽 {_au['n_source_exhausted']}")
            if ov.size:
                print(f"  [overlap] 接受样本 overlap: max={ov.max():.4f} "
                      f"p50={np.percentile(ov,50):.4f} p90={np.percentile(ov,90):.4f} "
                      f"p99={np.percentile(ov,99):.4f}")
            # 最终硬断言:enforce 时 max 必须 < 阈值,否则整个构建判废
            _lo = float(_au.get("min_overlap_ratio", 0.0) or 0.0)
            if assert_final and _au["enforce"] and ov.size and (
                    ov.max() >= _au["max_overlap_ratio"] or ov.min() < _lo):
                if h5_path is not None and _P(h5_path).is_file():
                    _P(h5_path).unlink()
                    print(f"  [overlap] 判废,已删除 {h5_path}(避免被当成正常产物复用)")
                raise RuntimeError(
                    f"[overlap] 最终断言失败:overlap ∈ [{ov.min():.4f}, {ov.max():.4f}] "
                    f"超出预注册区间 [{_lo}, {_au['max_overlap_ratio']}) —— "
                    f"硬约束未被 enforce,构建判废。")


    def _process_scene(self, scene_category: str, videos_dir: Path, start_idx: int) -> Tuple[List[Dict], int]:
        samples = []
        target_count = self.config.instances_per_scene
        global_idx = start_idx
        
        dataset_source = self.scene_generator.get_dataset_source(scene_category)

        # [v2.7 语义门] 拒绝采样:裁剪后逐帧场景语义不达标 → 换下一张源图重采。
        # 背景:AID 标签是整图属性,滑窗离开类别主体后 prompt 语义变假监督
        # (P0 人检:water 样本 12 帧 10 帧纯草地)。游标确定性推进:每次尝试
        # (无论接受与否)消耗池里下一个索引,(场景,索引) 种子链保持不变,
        # 池耗尽由 HD 加载器硬错(继承 v2.4 无回绕语义)。
        # gate 由包装器注入(builder.semantic_gate);独立跑本脚本 = 无门旧行为。
        gate = getattr(self, "semantic_gate", None)
        gate_max_tries = int(getattr(self, "semantic_gate_max_tries", 6)) if gate else 1
        if gate_max_tries < 1:
            raise ValueError(
                f"semantic_gate_max_tries 必须 >=1,实得 {gate_max_tries}")
        if not hasattr(self, "_semgate_cursors"):
            self._semgate_cursors = {}
        self._semgate_cursors.setdefault(scene_category, 0)

        # [v2.8 ⑤ overlap enforce] 拍板 2026-07-29:主 benchmark 正式 enforce
        # overlap_ratio < 0.10(此前 max_overlap_ratio 只计数不拒样本,C1 的
        # "硬约束"claim 与代码不符)。执行约束:
        #   * 超阈轨迹在【写入前】拒绝,并对【同一源图】重采轨迹 —— 一条坏
        #     轨迹不许消耗一张源图;
        #   * 顺序 = 便宜的 overlap 检查 → 裁帧 → 语义门 → 原子写入
        #     (overlap 只用 waypoints 端点算,无需裁帧,先筛最省算力);
        #   * 每源图有限重采(RSWM_TRAJ_MAX_TRIES,默认 12),耗尽才换源图;
        #     全部轨迹候选再共享单样本硬上限 20,不再做乘法级超长重采;
        #   * 语义预算耗尽可选择几何合法的最高分候选并显式告警/审计;
        #     overlap/轨迹/source identity 永不放宽,禁止重用已入库源图。
        # [v2.8 自查修] 开关双通道:注入属性(测试)优先,否则读环境变量
        # RSWM_ENFORCE_OVERLAP —— 34 号判门作业正是用环境变量把 enforce 关掉
        # (它的高重叠探针数据按设计就该超阈);旧代码只认属性,34 会被 enforce
        # 全拒 → 源池疯狂消耗直至硬错,判门作业根本跑不起来。
        if hasattr(self, "enforce_overlap"):
            enforce_overlap = bool(self.enforce_overlap)
        else:
            # [v2.8.2 N4] 只认字面 "0" 会让 false/no/off/空串全部**打开** enforce,
            # 一个手滑就把 34 的独立 profile 变成主口径(高重叠数据在 enforce 下
            # 拒绝率约 0.70,会疯狂消耗源池直到硬错)。
            enforce_overlap = _envbool("RSWM_ENFORCE_OVERLAP", True)
        if not enforce_overlap:
            print(f"    [overlap] ⚠️ enforce 已关闭(RSWM_ENFORCE_OVERLAP=0)——"
                  f" 仅限 34 号可学习性探针等独立 profile,产物不得进主 benchmark")
        max_ov = float(self.config.max_overlap_ratio)
        min_ov = float(getattr(self.config, "min_overlap_ratio", 0.0) or 0.0)
        if min_ov >= max_ov:
            raise ValueError(f"overlap 区间非法: [{min_ov}, {max_ov})")
        traj_max_tries = int(os.environ.get("RSWM_TRAJ_MAX_TRIES", "12"))
        if traj_max_tries < 1:
            raise ValueError(
                f"RSWM_TRAJ_MAX_TRIES 必须 >=1,实得 {traj_max_tries}")
        # [v2.8.2 CONFIRMED-2] 换源图预算必须与语义门解耦。
        # 旧写法 gate_max_tries = ... if gate else 1 —— 无语义门时预算恒为 1,
        # 于是第一次 overlap 耗尽就 continue 出循环 → 整个构建当场崩,而屏幕上
        # 刚打印完"换源图"。canonical 几何下 overlap 拒绝率实测 0.55,连续 12 次
        # 被拒 P≈7.8e-4/样本,500 样本的无门构建 ≈32% 概率整跑硬崩。
        src_retry_budget = int(os.environ.get("RSWM_SRC_MAX_TRIES", "6"))
        if src_retry_budget < 1:
            raise ValueError(
                f"RSWM_SRC_MAX_TRIES 必须 >=1,实得 {src_retry_budget}")
        src_max_tries = max(int(gate_max_tries), src_retry_budget)

        # A single output slot used to have a multiplicative worst case of
        # src_max_tries × traj_max_tries.  Server retry overrides could turn
        # that into hundreds of trajectory generations and CLIP calls.  Count
        # every generated trajectory candidate against one hard per-sample
        # budget; tests may lower the budget, but production cannot raise it
        # above 20.
        requested_retry_cap = int(getattr(
            self, "max_retry_per_sample",
            os.environ.get("RSWM_MAX_RETRY_PER_SAMPLE", str(MAX_RETRY_PER_SAMPLE))))
        if requested_retry_cap < 1:
            raise ValueError(
                f"RSWM_MAX_RETRY_PER_SAMPLE 必须 >=1,实得 {requested_retry_cap}")
        max_retry_per_sample = min(requested_retry_cap, MAX_RETRY_PER_SAMPLE)
        if requested_retry_cap > MAX_RETRY_PER_SAMPLE:
            print(f"    [sampling][WARN] 请求的单样本重试上限 "
                  f"{requested_retry_cap} 超过硬上限 {MAX_RETRY_PER_SAMPLE};已截断")
        source_max_tries = min(src_max_tries, max_retry_per_sample)
        print(f"    [sampling] scene={scene_category} "
              f"max_retry_per_sample={max_retry_per_sample} "
              f"source_max_tries={source_max_tries} "
              f"traj_max_tries={traj_max_tries}")
        audit = self._overlap_audit = getattr(self, "_overlap_audit", {
            "n_attempts": 0, "n_reject_overlap": 0, "n_reject_semantic": 0,
            "n_source_exhausted": 0, "n_semantic_fallback": 0,
            "semantic_fallbacks": [], "overlap_accepted": [], "per_scene": {},
            "enforce": bool(enforce_overlap), "max_overlap_ratio": max_ov,
            "min_overlap_ratio": min_ov,
            "traj_max_tries": traj_max_tries,
            "max_retry_per_sample": max_retry_per_sample})
        audit.setdefault("n_semantic_fallback", 0)
        audit.setdefault("semantic_fallbacks", [])
        audit["max_retry_per_sample"] = max_retry_per_sample
        _sc_a = audit["per_scene"].setdefault(
            scene_category, {"accepted": 0, "reject_overlap": 0,
                             "reject_semantic": 0, "semantic_fallback": 0,
                             "source_used": 0, "source_exhausted": 0})
        _sc_a.setdefault("semantic_fallback", 0)

        scene_semantic_rejects = 0
        scene_retry_count = 0
        scene_fallbacks = 0
        for i in range(target_count):
            accepted = None
            best_semantic = None
            best_rank = (-math.inf, -math.inf)
            best_score_seen = None
            n_exhaust_overlap = 0
            candidate_count = 0
            semantic_attempts = 0
            retry_debug_printed = False
            _side_lo = _side_hi = 0   # overlap 拒绝落在下界侧/上界侧的次数
            for _attempt in range(source_max_tries):
                if candidate_count >= max_retry_per_sample:
                    break
                sid = self._semgate_cursors[scene_category]
                self._semgate_cursors[scene_category] = sid + 1
                scene = self.scene_generator.generate_scene(scene_category, sid)
                _sc_a["source_used"] += 1

                # --- 同一源图内重采轨迹,直到 overlap 达标(不消耗源图)---
                path_info = None
                _traj_attempts = 0
                remaining = max_retry_per_sample - candidate_count
                for _t in range(min(traj_max_tries, remaining)):
                    _traj_attempts += 1
                    candidate_count += 1
                    audit["n_attempts"] += 1
                    num_segments = np.random.randint(
                        self.config.min_segments,
                        self.config.max_segments + 1
                    )
                    cand = self.path_generator.generate_path(num_segments)
                    if not enforce_overlap:
                        path_info = cand
                        break
                    _wp = cand["waypoints"]
                    # 用【实际 scene 形状】而非 config.full_scene_size:
                    # extract 内部是 `H, W, C = full_scene.shape`,一旦 HF 源图
                    # 尺寸与 config 不符,预检放行的轨迹会在最终硬断言处炸掉
                    # 整个构建(几小时白跑)。同源测量,分歧从根上消除。
                    _hw = (int(scene.shape[0]), int(scene.shape[1]))
                    _b0 = self.extractor._get_bounds(
                        _wp[0], self.config.window_size, _hw)
                    _b1 = self.extractor._get_bounds(
                        _wp[-1], self.config.window_size, _hw)
                    _ov = self.extractor._calculate_overlap(_b0, _b1)
                    if min_ov <= _ov < max_ov:
                        path_info = cand
                        break
                    audit["n_reject_overlap"] += 1
                    _sc_a["reject_overlap"] += 1
                    if _ov < min_ov:
                        _side_lo += 1
                    else:
                        _side_hi += 1
                if path_info is None:
                    if (candidate_count - 1 > 10) and not retry_debug_printed:
                        _best = ("none" if best_score_seen is None
                                 else f"{best_score_seen:.4f}")
                        print(f"    [sampling][debug] scene={scene_category} "
                              f"sample_id={global_idx} candidate_count={candidate_count} "
                              f"best_score={_best}")
                        retry_debug_printed = True
                    # This source never reached SemanticGate, so always record
                    # it as overlap-exhausted (including a partial final source
                    # stopped by the global cap).  The build report uses this
                    # list to avoid misclassifying it as a semantic rejection.
                    audit["n_source_exhausted"] += 1
                    n_exhaust_overlap += 1
                    _sc_a["source_exhausted"] += 1
                    audit.setdefault("overlap_exhausted_sids", []).append(
                        f"{scene_category}|{int(sid)}")
                    _reason = ("per-source trajectory budget exhausted"
                               if _traj_attempts >= traj_max_tries
                               else "global sample retry cap reached")
                    print(f"    [overlap] {scene_category}#{i}: 源图 {sid} 的 "
                          f"{_traj_attempts} 条轨迹 overlap ∉ [{min_ov}, {max_ov});"
                          f"{_reason}")
                    if candidate_count >= max_retry_per_sample:
                        break
                    continue

                num_segments = path_info["num_segments"]
                extraction = self.extractor.extract_multi_direction_sequence(scene, path_info)
                if gate is None:
                    verdict = None
                    accepted = (sid, num_segments, path_info, extraction, verdict)
                    if (candidate_count - 1 > 10) and not retry_debug_printed:
                        print(f"    [sampling][debug] scene={scene_category} "
                              f"sample_id={global_idx} candidate_count={candidate_count} "
                              "best_score=none")
                        retry_debug_printed = True
                    break
                # [v2.7.7/v2.7.11] 分发按【签名判定】(审查方定稿设计):
                #   * 显式新参数或 **kwargs 代理 → 传 dataset/full_scene;
                #   * 旧签名桩 → 只传旧参数;
                #   * gate 内部真 TypeError 原样外抛(绝不二次调用);
                #   * 签名不可解析 → 硬错,不猜接口。
                # [A3-N2] 签名每 gate 只解析一次(缓存到 self),不在尝试循环里重算。
                if getattr(self, "_semgate_sig_cache_for", None) is not gate:
                    import inspect as _insp
                    try:
                        _params = _insp.signature(gate.check_frames).parameters
                    except (TypeError, ValueError) as _exc:
                        raise RuntimeError(
                            "无法判定 semantic gate 的 check_frames 接口签名 —— "
                            "包装器丢失签名时不许靠捕获内部异常猜接口") from _exc
                    _has_var_kw = any(pp.kind is _insp.Parameter.VAR_KEYWORD
                                      for pp in _params.values())
                    self._semgate_sig_cache_for = gate
                    self._semgate_sig_params = _params
                    self._semgate_sig_var_kw = _has_var_kw
                _params = self._semgate_sig_params
                _has_var_kw = self._semgate_sig_var_kw
                _kw = {}
                for _name, _val in (("dataset", dataset_source),
                                    ("full_scene", extraction["full_scene"])):
                    if _name in _params or _has_var_kw:
                        _kw[_name] = _val
                verdict = gate.check_frames(
                    extraction["sliding_frames"], scene_category, **_kw)
                semantic_attempts += 1
                if verdict.frac is not None:
                    score = float(verdict.frac)
                    if not math.isfinite(score):
                        raise RuntimeError(
                            f"[semgate] {scene_category}#{i} 返回非有限分数 "
                            f"{verdict.frac!r}")
                    best_score_seen = (score if best_score_seen is None
                                       else max(best_score_seen, score))
                if verdict.passed or verdict.skipped:
                    accepted = (sid, num_segments, path_info, extraction, verdict)
                    if (candidate_count - 1 > 10) and not retry_debug_printed:
                        _best = ("none" if best_score_seen is None
                                 else f"{best_score_seen:.4f}")
                        print(f"    [sampling][debug] scene={scene_category} "
                              f"sample_id={global_idx} candidate_count={candidate_count} "
                              f"best_score={_best}")
                        retry_debug_printed = True
                    break
                if verdict.frac is None:
                    raise RuntimeError(
                        f"[semgate] {scene_category}#{i} 拒绝结果缺少 frac,"
                        "无法选择 best candidate")
                margin = getattr(verdict, "mean_margin", None)
                rank = (float(verdict.frac),
                        float(margin) if margin is not None and math.isfinite(float(margin))
                        else -math.inf)
                if best_semantic is None or rank > best_rank:
                    best_rank = rank
                    best_semantic = (
                        sid, num_segments, path_info, extraction, verdict)
                audit["n_reject_semantic"] += 1
                _sc_a["reject_semantic"] += 1
                scene_semantic_rejects += 1
                print(f"    [semgate] {scene_category}#{i}: 源图索引 {sid} 被拒 "
                      f"(同意帧 {verdict.frac:.2f} < {verdict.threshold:.2f}),换图重采")
                if (candidate_count - 1 > 10) and not retry_debug_printed:
                    print(f"    [sampling][debug] scene={scene_category} "
                          f"sample_id={global_idx} candidate_count={candidate_count} "
                          f"best_score={best_score_seen:.4f}")
                    retry_debug_printed = True

            if accepted is None and best_semantic is not None:
                # Semantic filtering is the only soft constraint.  The saved
                # best candidate has already passed trajectory, overlap and
                # source-identity checks.  Lower only this sample's effective
                # rejection threshold, retain its real score, and make the
                # fallback explicit in warnings and overlap_audit.json.
                sid, num_segments, path_info, extraction, verdict = best_semantic
                original_threshold = (float(verdict.threshold)
                                      if verdict.threshold is not None
                                      else float(verdict.frac))
                effective_threshold = min(original_threshold, float(verdict.frac))
                verdict.passed = True
                verdict.threshold = effective_threshold
                accepted = (sid, num_segments, path_info, extraction, verdict)
                audit["n_semantic_fallback"] += 1
                _sc_a["semantic_fallback"] += 1
                scene_fallbacks += 1
                fallback_record = {
                    "scene": scene_category,
                    "sample_id": int(global_idx),
                    "scene_slot": int(i),
                    "source_scene_id": int(sid),
                    "candidate_count": int(candidate_count),
                    "semantic_attempts": int(semantic_attempts),
                    "best_score": float(verdict.frac),
                    "original_threshold": original_threshold,
                    "effective_threshold": effective_threshold,
                }
                audit["semantic_fallbacks"].append(fallback_record)
                print(f"    [semgate][WARNING] scene={scene_category} "
                      f"sample_id={global_idx} 重试预算耗尽;接受满足 overlap/轨迹/"
                      f"source identity 的 best candidate: candidate_count="
                      f"{candidate_count} best_score={verdict.frac:.4f} "
                      f"threshold={original_threshold:.4f}->{effective_threshold:.4f}")

            if accepted is None:
                # No geometry-valid candidate ever reached SemanticGate.  The
                # hard overlap/trajectory contract is never relaxed.
                _hint = ("窗口相对场景过大,端点几乎必然重叠 → 调小窗口或"
                         "提高 max_overlap_ratio" if _side_hi > _side_lo else
                         "位移下限(min_segment_length)与 overlap 下界互斥,"
                         "端点离得太远 → 调低 min_overlap_ratio,或用 "
                         "--angle-quantize 4 走带反解路径")
                raise RuntimeError(
                    f"[sampling] {scene_category} 第 {i} 个样本达到严格上限 "
                    f"{max_retry_per_sample},仍无满足 overlap [{min_ov}, {max_ov}) "
                    f"与合法轨迹的 candidate(偏低 {_side_lo} / 偏高 {_side_hi};"
                    f"完整耗尽源图 {n_exhaust_overlap})。{_hint}。"
                    "硬约束不允许 fallback。")

            sid, num_segments, path_info, extraction, verdict = accepted
            audit["overlap_accepted"].append(float(extraction["overlap_ratio"]))
            _sc_a["accepted"] += 1
            
            # Save video
            video_path = videos_dir / f"sample_{global_idx:05d}.mp4"
            video_writer = VideoWriter(str(video_path), fps=self.config.video_fps, codec=self.config.video_codec)
            if video_writer.write_frames(extraction["sliding_frames"]):
                self.stats["videos_saved"] += 1
            
            sample = {
                "sample_id": global_idx,
                "scene_category": scene_category,
                "dataset_source": dataset_source,
                # [v2.7] 实际消耗的池索引(语义门重采后 ≠ 块内序号 i)。
                # source registry 与 _annotate_source_identity 按它对账,
                # 否则源身份会错配到被拒尝试的源图上。
                "source_scene_id": int(sid),
                "video_path": str(video_path.relative_to(Path(self.config.output_dir))),
                
                # Required fields
                "sliding_frames": extraction["sliding_frames"],
                "center_frame": extraction["center_frame"],
                "target_frame": extraction["target_frame"],
                "full_scene": extraction["full_scene"],
                "direction_embedding": extraction["direction_embedding"],
                "trajectory": extraction["trajectory"],
                "semgate": (None if verdict is None or verdict.skipped else {
                    "frac": verdict.frac, "threshold": verdict.threshold,
                    "n_scored": verdict.n_scored,
                    "tries": max(1, semantic_attempts),
                    # [v2.7.11] 保真比建库期已算,落盘留证据(校准/复核对账用)
                    "mean_margin": getattr(verdict, "mean_margin", None),
                    "pres_ratio": getattr(verdict, "pres_ratio", None),
                    "pres_ratio_raw": getattr(verdict, "pres_ratio_raw", None),
                    "scene_pool_mass": getattr(verdict, "scene_pool_mass", None)}),
                
                # Additional fields
                "segment_embeddings": extraction["segment_embeddings"],
                "segment_angles": extraction["segment_angles"],
                "waypoints": np.array(extraction["waypoints"]),
                "num_segments": extraction["num_segments"],
                "total_frames": extraction["total_frames"],
                "overlap_ratio": extraction["overlap_ratio"],
                "start_bounds": extraction["start_bounds"],
                "end_bounds": extraction["end_bounds"],
                "start_position": extraction["start_position"],
                "end_position": extraction["end_position"],
                "steps_per_segment": self.config.steps_per_segment
            }
            
            samples.append(sample)
            self.stats["overlap_ratios"].append(extraction["overlap_ratio"])
            # [RSRoll-Bench P0#7] 帧间重叠同步入册（None 安全：K/steps 退化时统计量为 None）
            _sos = extraction.get("step_overlap_stats") or {}
            if _sos.get("min") is not None:
                self.stats["step_overlap_min"].append(float(_sos["min"]))
                self.stats["step_overlap_p50"].append(float(_sos["p50"]))
            self.stats["num_segments_distribution"].append(num_segments)
            global_idx += 1

            scene_retry_count += max(0, candidate_count - 1)
            accepted_count = i + 1
            if accepted_count % 20 == 0 or accepted_count == target_count:
                average_retry = scene_retry_count / accepted_count
                print(f"    [scene progress] scene={scene_category} "
                      f"accepted={accepted_count}/{target_count} "
                      f"semantic_rejects={scene_semantic_rejects} "
                      f"retry_count={scene_retry_count} "
                      f"average_retry={average_retry:.2f} "
                      f"fallbacks={scene_fallbacks}")
        
        return samples, global_idx
    
    def _write_sample_to_hdf5(self, data_group, idx: int, sample: dict) -> None:
        """Write a single sample into an open HDF5 data group."""
        grp = data_group.create_group(f'sample_{idx:05d}')
        grp.create_dataset('sliding_frames', data=sample['sliding_frames'], compression='gzip')
        grp.create_dataset('center_frame', data=sample['center_frame'], compression='gzip')
        grp.create_dataset('target_frame', data=sample['target_frame'], compression='gzip')
        grp.create_dataset('full_scene', data=sample['full_scene'], compression='gzip')
        grp.create_dataset('direction_embedding', data=sample['direction_embedding'])
        grp.create_dataset('trajectory', data=sample['trajectory'])
        grp.create_dataset('segment_embeddings', data=sample['segment_embeddings'])
        grp.create_dataset('segment_angles', data=sample['segment_angles'])
        grp.create_dataset('waypoints', data=sample['waypoints'])
        grp.attrs['sample_id'] = sample['sample_id']
        grp.attrs['scene_category'] = sample['scene_category']
        # [v2.7] 语义门溯源:实际池索引 + 门评分(转换器透传最终 H5)
        if 'source_scene_id' in sample:
            grp.attrs['source_scene_id'] = int(sample['source_scene_id'])
        if sample.get('semgate'):
            grp.attrs['semgate_frac'] = float(sample['semgate']['frac'])
            grp.attrs['semgate_threshold'] = float(sample['semgate']['threshold'])
            grp.attrs['semgate_n_scored'] = int(sample['semgate']['n_scored'])
            grp.attrs['semgate_tries'] = int(sample['semgate']['tries'])
            for _pk in ('mean_margin', 'pres_ratio', 'pres_ratio_raw',
                        'scene_pool_mass'):
                if sample['semgate'].get(_pk) is not None:
                    grp.attrs[f'semgate_{_pk}'] = float(sample['semgate'][_pk])
        grp.attrs['dataset_source'] = sample.get('dataset_source', 'unknown')
        grp.attrs['video_path'] = sample['video_path']
        grp.attrs['num_segments'] = sample['num_segments']
        grp.attrs['total_frames'] = sample['total_frames']
        grp.attrs['overlap_ratio'] = sample['overlap_ratio']
        grp.attrs['start_bounds'] = sample['start_bounds']
        grp.attrs['end_bounds'] = sample['end_bounds']
        grp.attrs['start_position'] = sample['start_position']
        grp.attrs['end_position'] = sample['end_position']
        grp.attrs['steps_per_segment'] = sample['steps_per_segment']

    # [v2.7.11] 死方法 _save_to_hdf5 已删除:它不写 source_scene_id/semgate_* attrs,
    # 若被复活,_annotate_source_identity 会回退块内序号对账 —— 而被拒尝试也在
    # registry 里 → 源身份静默错配且不触发 miss 硬错。活路径 = _write_sample_to_hdf5。

    def _save_metadata_json(self, output_path: Path) -> None:
        metadata = {
            "version": "5.0.0",
            "benchmark_type": "multi_direction_v5_random_position",
            "total_instances": self.stats["total_instances"],
            "videos_saved": self.stats["videos_saved"],
            "config": {
                "full_scene_size": list(self.config.full_scene_size),
                "window_size": list(self.config.window_size),
                "embed_dim": self.config.embed_dim,
                "min_segments": self.config.min_segments,
                "max_segments": self.config.max_segments,
                "steps_per_segment": self.config.steps_per_segment,
                "position_margin": self.config.position_margin,
                "min_segment_length": self.config.min_segment_length,
                "video_fps": self.config.video_fps,
                "scene_categories": self.config.scene_categories,
            },
            "creation_time": time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to {output_path}")


# ============================================================================
# Statistics
# ============================================================================

def print_stats(stats: Dict) -> None:
    print("\n" + "=" * 70)
    print("Multi-Direction Benchmark V5 Statistics (Random Positions)")
    print("=" * 70)
    print(f"Total instances: {stats['total_instances']}")
    print(f"Videos saved: {stats['videos_saved']}")
    print(f"Valid instances (<10% overlap): {stats['valid_instances']}")
    
    avg_overlap = np.mean(stats['overlap_ratios']) if stats['overlap_ratios'] else 0
    print(f"\nOverlap Statistics:")
    print(f"  - Average: {avg_overlap:.2%}")
    
    print(f"\nSegments Distribution:")
    seg_counts = {}
    for n in stats['num_segments_distribution']:
        seg_counts[n] = seg_counts.get(n, 0) + 1
    for n, count in sorted(seg_counts.items()):
        print(f"  - {n} segments: {count} samples")
    
    print(f"\nProcessing time: {stats['processing_time']:.2f} seconds")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build Multi-Direction Benchmark V5 (Random Positions)")
    parser.add_argument("--output", "-o", default="multi_direction_benchmark_v5", help="Output directory")
    parser.add_argument("--instances", "-n", type=int, default=500, help="Instances per scene (total = n × 4)")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--min-segments", type=int, default=2, help="Minimum direction segments")
    parser.add_argument("--max-segments", type=int, default=4, help="Maximum direction segments")
    parser.add_argument("--steps", type=int, default=9, help="Steps per segment")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--margin", type=float, default=0.1, help="Position margin from edge (0-0.5)")
    parser.add_argument("--min-seg-len", type=float, default=0.15, help="Min segment length fraction")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()
    
    config = MultiDirectionConfig()
    config.output_dir = args.output
    config.instances_per_scene = args.instances
    config.random_seed = args.seed
    config.min_segments = args.min_segments
    config.max_segments = args.max_segments
    config.steps_per_segment = args.steps
    config.video_fps = args.fps
    config.position_margin = args.margin
    config.min_segment_length = args.min_seg_len
    
    use_hf = not args.synthetic
    
    if not HF_AVAILABLE and use_hf:
        print("[official] HF dataset required, synthetic fallback disabled", file=sys.stderr)
        print("ERROR: install the datasets package or pass --synthetic only for "
              "the standalone demo.", file=sys.stderr)
        return 2
    
    if not CV2_AVAILABLE:
        print("Warning: OpenCV not available. Videos will not be saved.")
        print("Install with: pip install opencv-python\n")
    
    builder = MultiDirectionBenchmarkBuilder(config, use_hf=use_hf)
    stats = builder.build()
    
    print_stats(stats)
    print(f"\nBenchmark built successfully!")
    print(f"Output directory: {config.output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
