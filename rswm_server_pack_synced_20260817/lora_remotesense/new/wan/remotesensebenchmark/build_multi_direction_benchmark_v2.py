#!/usr/bin/env python3
"""
Multi-Direction Sliding Window Benchmark Builder V2

This version uses single image resize instead of mosaic tiling.
Each scene is created by resizing a single HuggingFace image directly to 512x512.

Key Changes from V1:
- No mosaic tiling: single image resize to full scene size
- Better preservation of image content and structure
- Each sample uses one unique source image

Key Features:
- Multiple direction segments (e.g., go right → then up → then left)
- Each segment has its own direction embedding
- Supports arbitrary number of waypoints
- Records the complete multi-directional trajectory
"""

import os
import sys
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
    "flood": {
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

@dataclass
class MultiDirectionConfig:
    """Configuration for multi-direction benchmark"""
    output_path: str = "multi_direction_benchmark_v2.h5"
    instances_per_scene: int = 100
    scene_categories: List[str] = field(default_factory=lambda: ["general", "flood", "urban", "rural"])
    
    full_scene_size: Tuple[int, int] = (512, 512)
    num_channels: int = 3
    
    window_size: Tuple[int, int] = (96, 96)
    steps_per_segment: int = 8
    
    min_segments: int = 2
    max_segments: int = 4
    
    embed_dim: int = 64
    num_freq_bands: int = 16
    
    max_overlap_ratio: float = 0.10
    random_seed: int = 42


# ============================================================================
# Scene Generator (Single Image Resize - No Mosaic)
# ============================================================================

class SceneGenerator:
    """
    Generate scenes from HuggingFace remote sensing datasets.
    Uses single image resize instead of mosaic tiling.
    Falls back to synthetic data if HF datasets are unavailable.
    """
    
    def __init__(self, config: MultiDirectionConfig, use_hf: bool = True):
        self.config = config
        self.use_hf = use_hf and HF_AVAILABLE
        self.hf_datasets: Dict[str, object] = {}
        self.hf_indices: Dict[str, int] = {}
        np.random.seed(config.random_seed)
        
        if self.use_hf:
            self._load_hf_datasets()
    
    def _load_hf_datasets(self) -> None:
        """Load HuggingFace datasets for each scene category"""
        global HF_DATASETS_CACHE
        
        print("\nLoading HuggingFace datasets...")
        
        for scene_category in self.config.scene_categories:
            if scene_category not in HF_DATASET_CONFIG:
                print(f"  Warning: No HF config for scene '{scene_category}', will use synthetic")
                continue
            
            scene_config = HF_DATASET_CONFIG[scene_category]
            dataset_loaded = False
            
            for ds_info in scene_config["datasets"]:
                ds_name = ds_info["name"]
                ds_split = ds_info.get("split", "train")
                
                cache_key = f"{ds_name}:{ds_split}"
                if cache_key in HF_DATASETS_CACHE:
                    self.hf_datasets[scene_category] = {
                        "dataset": HF_DATASETS_CACHE[cache_key],
                        "feature": ds_info["feature"]
                    }
                    self.hf_indices[scene_category] = 0
                    print(f"  {scene_category}: Using cached {ds_name}")
                    dataset_loaded = True
                    break
                
                try:
                    print(f"  {scene_category}: Loading {ds_name}...")
                    ds = hf_load_dataset(ds_name, split=ds_split, trust_remote_code=True)
                    HF_DATASETS_CACHE[cache_key] = ds
                    self.hf_datasets[scene_category] = {
                        "dataset": ds,
                        "feature": ds_info["feature"]
                    }
                    self.hf_indices[scene_category] = 0
                    print(f"  {scene_category}: Loaded {len(ds)} samples from {ds_name}")
                    dataset_loaded = True
                    break
                except Exception as e:
                    print(f"  {scene_category}: Failed to load {ds_name}: {e}")
                    continue
            
            if not dataset_loaded:
                print(f"  {scene_category}: No HF dataset available, will use synthetic")
        
        print()
    
    def _to_numpy_image(self, x) -> np.ndarray:
        """Convert PIL Image or array to numpy array"""
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
        """Resize image to target size using high-quality interpolation"""
        H, W = target_size
        h, w = img.shape[:2]
        
        if h == H and w == W:
            return img
        
        if CV2_AVAILABLE:
            # Use INTER_LANCZOS4 for better quality when upscaling
            if H > h or W > w:
                interpolation = cv2.INTER_LANCZOS4
            else:
                interpolation = cv2.INTER_AREA
            return cv2.resize(img, (W, H), interpolation=interpolation)
        else:
            # Bilinear interpolation using numpy
            h_scale = np.linspace(0, h - 1, H)
            w_scale = np.linspace(0, w - 1, W)
            
            h_floor = np.floor(h_scale).astype(int)
            w_floor = np.floor(w_scale).astype(int)
            h_ceil = np.minimum(h_floor + 1, h - 1)
            w_ceil = np.minimum(w_floor + 1, w - 1)
            
            h_weight = h_scale - h_floor
            w_weight = w_scale - w_floor
            
            result = np.zeros((H, W, img.shape[2]), dtype=img.dtype)
            for c in range(img.shape[2]):
                for i in range(H):
                    for j in range(W):
                        hf, hc = h_floor[i], h_ceil[i]
                        wf, wc = w_floor[j], w_ceil[j]
                        hw, ww = h_weight[i], w_weight[j]
                        
                        result[i, j, c] = (
                            img[hf, wf, c] * (1 - hw) * (1 - ww) +
                            img[hf, wc, c] * (1 - hw) * ww +
                            img[hc, wf, c] * hw * (1 - ww) +
                            img[hc, wc, c] * hw * ww
                        )
            return result
    
    def generate_scene(self, scene_category: str, scene_id: int) -> np.ndarray:
        """Generate a full scene image from HF dataset or synthetic"""
        np.random.seed(self.config.random_seed + scene_id)
        
        H, W = self.config.full_scene_size
        C = self.config.num_channels
        
        if self.use_hf and scene_category in self.hf_datasets:
            scene = self._load_scene_from_hf(scene_category, scene_id, H, W)
            if scene is not None:
                return scene
        
        if scene_category == "flood":
            scene = self._generate_flood_scene(H, W, C)
        elif scene_category == "urban":
            scene = self._generate_urban_scene(H, W, C)
        elif scene_category == "rural":
            scene = self._generate_rural_scene(H, W, C)
        else:
            scene = self._generate_general_scene(H, W, C)
        
        return scene.astype(np.float32)
    
    def _load_scene_from_hf(self, scene_category: str, scene_id: int, H: int, W: int) -> Optional[np.ndarray]:
        """
        Load and process a scene from HuggingFace dataset.
        Uses single image resize instead of mosaic tiling.
        """
        try:
            ds_info = self.hf_datasets[scene_category]
            dataset = ds_info["dataset"]
            feature_name = ds_info["feature"]
            
            n = len(dataset)
            if n == 0:
                return None
            
            # Get single image by index
            idx = scene_id % n
            example = dataset[int(idx)]
            
            img_data = example[feature_name]
            img = self._to_numpy_image(img_data)
            
            # Resize single image directly to target size (no mosaic)
            scene = self._resize_image(img, (H, W))
            
            # Normalize to 0-255 range
            if scene.max() <= 1.0:
                scene = scene * 255.0
            
            scene = scene.clip(0, 255).astype(np.float32)
            return scene
            
        except Exception as e:
            print(f"Warning: Failed to load HF scene for {scene_category}: {e}")
            return None
    
    def _generate_perlin_like(self, H: int, W: int) -> np.ndarray:
        """Generate Perlin-like noise texture (fallback)"""
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
    """Generate continuous direction embeddings"""
    
    def __init__(self, embed_dim: int = 64, num_freq_bands: int = 16):
        self.embed_dim = embed_dim
        self.num_freq_bands = num_freq_bands
        self.freq_scales = np.logspace(0, np.log10(1000), num_freq_bands)
        
        base_dim = num_freq_bands * 2 + num_freq_bands + 2
        np.random.seed(42)
        self.projection = np.random.randn(base_dim, embed_dim) / np.sqrt(base_dim)
    
    def encode(self, angle_degrees: float, magnitude: float = 1.0) -> Dict:
        """Generate direction embedding for angle"""
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
    
    def encode_sequence(self, angles: List[float]) -> Dict:
        """Encode a sequence of direction angles"""
        embeddings = []
        direction_vectors = []
        
        for angle in angles:
            result = self.encode(angle)
            embeddings.append(result["embedding"])
            direction_vectors.append(result["direction_vector"])
        
        return {
            "embeddings": np.stack(embeddings, axis=0),
            "angles": np.array(angles, dtype=np.float32),
            "direction_vectors": np.array(direction_vectors, dtype=np.float32)
        }


# ============================================================================
# Multi-Direction Path Generator
# ============================================================================

class MultiDirectionPathGenerator:
    """Generate multi-directional paths through the scene"""
    
    def __init__(self, config: MultiDirectionConfig):
        self.config = config
    
    def generate_path(
        self,
        num_segments: int,
        start_position: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """Generate a multi-directional path with specified number of segments."""
        H, W = self.config.full_scene_size
        win_h, win_w = self.config.window_size
        half_h, half_w = win_h // 2, win_w // 2
        
        if start_position is None:
            start_cy = H // 2
            start_cx = W // 2
        else:
            start_cy, start_cx = start_position
        
        waypoints = [(start_cy, start_cx)]
        segment_angles = []
        
        current_cy, current_cx = start_cy, start_cx
        
        for seg_idx in range(num_segments):
            if seg_idx > 0:
                prev_angle = segment_angles[-1]
                angle_offset = np.random.uniform(45, 315)
                angle = (prev_angle + angle_offset) % 360
            else:
                angle = np.random.uniform(0, 360)
            
            segment_angles.append(angle)
            
            segment_length = max(win_h, win_w) * 0.8
            
            angle_rad = np.deg2rad(angle)
            dx = np.cos(angle_rad)
            dy = -np.sin(angle_rad)
            
            next_cx = current_cx + dx * segment_length
            next_cy = current_cy + dy * segment_length
            
            next_cx = np.clip(next_cx, half_w, W - half_w)
            next_cy = np.clip(next_cy, half_h, H - half_h)
            
            waypoints.append((int(next_cy), int(next_cx)))
            current_cy, current_cx = int(next_cy), int(next_cx)
        
        return {
            "waypoints": waypoints,
            "segment_angles": segment_angles,
            "num_segments": num_segments,
            "steps_per_segment": self.config.steps_per_segment
        }


# ============================================================================
# Multi-Direction Sliding Window Extractor
# ============================================================================

class MultiDirectionExtractor:
    """Extract multi-directional sliding window sequences"""
    
    def __init__(self, config: MultiDirectionConfig):
        self.config = config
        self.direction_embedder = DirectionEmbedder(
            embed_dim=config.embed_dim,
            num_freq_bands=config.num_freq_bands
        )
    
    def extract_multi_direction_sequence(
        self,
        full_scene: np.ndarray,
        path_info: Dict
    ) -> Dict:
        """Extract sliding window sequence following multi-directional path."""
        H, W, C = full_scene.shape
        win_h, win_w = self.config.window_size
        half_h, half_w = win_h // 2, win_w // 2
        steps_per_segment = path_info["steps_per_segment"]
        
        waypoints = path_info["waypoints"]
        segment_angles = path_info["segment_angles"]
        num_segments = path_info["num_segments"]
        
        all_frames = []
        all_trajectory = []
        segment_frames_list = []
        segment_embeddings = []
        segment_info = []
        
        for seg_idx in range(num_segments):
            start_wp = waypoints[seg_idx]
            end_wp = waypoints[seg_idx + 1]
            angle = segment_angles[seg_idx]
            
            dir_result = self.direction_embedder.encode(angle)
            segment_embeddings.append(dir_result["embedding"])
            
            segment_frames = []
            segment_trajectory = []
            
            for step in range(steps_per_segment):
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
                
                segment_frames.append(window)
                segment_trajectory.append((cy, cx))
                
                all_frames.append(window)
                all_trajectory.append((cy, cx, seg_idx, angle))
            
            segment_frames_list.append(np.stack(segment_frames, axis=0))
            
            segment_info.append({
                "segment_index": seg_idx,
                "start_waypoint": start_wp,
                "end_waypoint": end_wp,
                "direction_angle": angle,
                "direction_vector": dir_result["direction_vector"],
                "num_frames": steps_per_segment
            })
        
        all_frames = np.stack(all_frames, axis=0)
        segment_embeddings = np.stack(segment_embeddings, axis=0)
        
        start_bounds = self._get_bounds(waypoints[0], (win_h, win_w), (H, W))
        end_bounds = self._get_bounds(waypoints[-1], (win_h, win_w), (H, W))
        overlap_ratio = self._calculate_overlap(start_bounds, end_bounds)
        
        return {
            "sliding_frames": all_frames,
            "segment_frames": segment_frames_list,
            "segment_embeddings": segment_embeddings,
            "segment_angles": np.array(segment_angles, dtype=np.float32),
            "trajectory": all_trajectory,
            "waypoints": waypoints,
            "segment_info": segment_info,
            "num_segments": num_segments,
            "total_frames": len(all_frames),
            "start_frame": all_frames[0],
            "end_frame": all_frames[-1],
            "start_bounds": start_bounds,
            "end_bounds": end_bounds,
            "overlap_ratio": overlap_ratio,
            "full_scene": full_scene
        }
    
    def _resize_window(self, window: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize window to target size"""
        if CV2_AVAILABLE:
            return cv2.resize(window, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
        else:
            h, w = window.shape[:2]
            h_idx = np.linspace(0, h - 1, target_size[0]).astype(int)
            w_idx = np.linspace(0, w - 1, target_size[1]).astype(int)
            return window[np.ix_(h_idx, w_idx)]
    
    def _get_bounds(
        self,
        center: Tuple[int, int],
        window_size: Tuple[int, int],
        scene_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """Get window bounds from center position"""
        cy, cx = center
        win_h, win_w = window_size
        H, W = scene_size
        
        h_start = max(0, cy - win_h // 2)
        h_end = min(H, cy + win_h // 2)
        w_start = max(0, cx - win_w // 2)
        w_end = min(W, cx + win_w // 2)
        
        return (h_start, h_end, w_start, w_end)
    
    def _calculate_overlap(
        self,
        bounds1: Tuple[int, int, int, int],
        bounds2: Tuple[int, int, int, int]
    ) -> float:
        """Calculate overlap ratio between two regions"""
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
# Benchmark Builder
# ============================================================================

class MultiDirectionBenchmarkBuilder:
    """Build multi-direction sliding window benchmark dataset"""
    
    def __init__(self, config: MultiDirectionConfig, use_hf: bool = True):
        self.config = config
        self.use_hf = use_hf
        self.scene_generator = SceneGenerator(config, use_hf=use_hf)
        self.path_generator = MultiDirectionPathGenerator(config)
        self.extractor = MultiDirectionExtractor(config)
        
        self.stats = {
            "total_instances": 0,
            "valid_instances": 0,
            "instances_per_scene": {},
            "overlap_ratios": [],
            "num_segments_distribution": [],
            "processing_time": 0.0
        }
    
    def build(self) -> Dict:
        """Build the complete benchmark"""
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("Multi-Direction Benchmark Builder V2 (Single Image Resize)")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  - Output: {self.config.output_path}")
        print(f"  - Instances per scene: {self.config.instances_per_scene}")
        print(f"  - Scene categories: {self.config.scene_categories}")
        print(f"  - Total instances: {self.config.instances_per_scene * len(self.config.scene_categories)}")
        print(f"  - Direction segments: {self.config.min_segments}-{self.config.max_segments}")
        print(f"  - Steps per segment: {self.config.steps_per_segment}")
        print(f"  - Window size: {self.config.window_size}")
        print(f"  - Data source: {'HuggingFace (Single Resize)' if self.use_hf else 'Synthetic'}")
        print()
        
        np.random.seed(self.config.random_seed)
        all_samples = []
        
        for scene in self.config.scene_categories:
            print(f"Processing scene: {scene}")
            samples = self._process_scene(scene)
            all_samples.extend(samples)
            self.stats["instances_per_scene"][scene] = len(samples)
            print(f"  Generated {len(samples)} samples")
        
        self._save_to_hdf5(all_samples)
        
        self.stats["total_instances"] = len(all_samples)
        self.stats["valid_instances"] = sum(
            1 for s in all_samples
            if s["overlap_ratio"] < self.config.max_overlap_ratio
        )
        self.stats["processing_time"] = time.time() - start_time
        
        return self.stats
    
    def _process_scene(self, scene_category: str) -> List[Dict]:
        """Process a single scene category"""
        samples = []
        target_count = self.config.instances_per_scene
        
        for i in range(target_count):
            scene = self.scene_generator.generate_scene(scene_category, i)
            
            num_segments = np.random.randint(
                self.config.min_segments,
                self.config.max_segments + 1
            )
            
            path_info = self.path_generator.generate_path(num_segments)
            extraction = self.extractor.extract_multi_direction_sequence(scene, path_info)
            
            sample = {
                "sample_id": len(samples),
                "scene_category": scene_category,
                "sliding_frames": extraction["sliding_frames"],
                "segment_embeddings": extraction["segment_embeddings"],
                "segment_angles": extraction["segment_angles"],
                "waypoints": np.array(extraction["waypoints"]),
                "num_segments": extraction["num_segments"],
                "total_frames": extraction["total_frames"],
                "start_frame": extraction["start_frame"],
                "end_frame": extraction["end_frame"],
                "start_bounds": extraction["start_bounds"],
                "end_bounds": extraction["end_bounds"],
                "overlap_ratio": extraction["overlap_ratio"],
                "full_scene": extraction["full_scene"],
                "steps_per_segment": self.config.steps_per_segment
            }
            
            samples.append(sample)
            self.stats["overlap_ratios"].append(extraction["overlap_ratio"])
            self.stats["num_segments_distribution"].append(num_segments)
            
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i + 1}/{target_count}")
        
        return samples
    
    def _save_to_hdf5(self, samples: List[Dict]) -> None:
        """Save all samples to HDF5 file"""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving {len(samples)} samples to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            meta = f.create_group('metadata')
            meta.attrs['version'] = '2.0.0'
            meta.attrs['benchmark_type'] = 'multi_direction_v2'
            meta.attrs['scene_mode'] = 'single_resize'  # New: indicates no mosaic
            meta.attrs['total_instances'] = len(samples)
            meta.attrs['scene_categories'] = self.config.scene_categories
            meta.attrs['max_overlap_ratio'] = self.config.max_overlap_ratio
            meta.attrs['embed_dim'] = self.config.embed_dim
            meta.attrs['full_scene_size'] = self.config.full_scene_size
            meta.attrs['window_size'] = self.config.window_size
            meta.attrs['min_segments'] = self.config.min_segments
            meta.attrs['max_segments'] = self.config.max_segments
            meta.attrs['steps_per_segment'] = self.config.steps_per_segment
            meta.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            data = f.create_group('data')
            
            for idx, sample in enumerate(samples):
                grp = data.create_group(f'sample_{idx:05d}')
                
                grp.create_dataset('sliding_frames', data=sample['sliding_frames'], compression='gzip')
                grp.create_dataset('segment_embeddings', data=sample['segment_embeddings'])
                grp.create_dataset('segment_angles', data=sample['segment_angles'])
                grp.create_dataset('waypoints', data=sample['waypoints'])
                grp.create_dataset('start_frame', data=sample['start_frame'], compression='gzip')
                grp.create_dataset('end_frame', data=sample['end_frame'], compression='gzip')
                grp.create_dataset('full_scene', data=sample['full_scene'], compression='gzip')
                
                grp.attrs['sample_id'] = sample['sample_id']
                grp.attrs['scene_category'] = sample['scene_category']
                grp.attrs['num_segments'] = sample['num_segments']
                grp.attrs['total_frames'] = sample['total_frames']
                grp.attrs['overlap_ratio'] = sample['overlap_ratio']
                grp.attrs['start_bounds'] = sample['start_bounds']
                grp.attrs['end_bounds'] = sample['end_bounds']
                grp.attrs['steps_per_segment'] = sample['steps_per_segment']
            
            index = f.create_group('index')
            for scene in self.config.scene_categories:
                indices = [i for i, s in enumerate(samples) if s['scene_category'] == scene]
                index.create_dataset(scene, data=np.array(indices))
            
            seg_index = f.create_group('index_by_segments')
            for n_seg in range(self.config.min_segments, self.config.max_segments + 1):
                indices = [i for i, s in enumerate(samples) if s['num_segments'] == n_seg]
                if indices:
                    seg_index.create_dataset(f'segments_{n_seg}', data=np.array(indices))
        
        print(f"Saved successfully to {output_path}")


# ============================================================================
# Statistics
# ============================================================================

def print_stats(stats: Dict) -> None:
    """Print benchmark statistics"""
    print("\n" + "=" * 70)
    print("Multi-Direction Benchmark V2 Statistics")
    print("=" * 70)
    print(f"Total instances: {stats['total_instances']}")
    print(f"Valid instances (<10% overlap): {stats['valid_instances']}")
    
    avg_overlap = np.mean(stats['overlap_ratios']) if stats['overlap_ratios'] else 0
    max_overlap = np.max(stats['overlap_ratios']) if stats['overlap_ratios'] else 0
    
    print(f"\nOverlap Statistics:")
    print(f"  - Average: {avg_overlap:.2%}")
    print(f"  - Max: {max_overlap:.2%}")
    
    print(f"\nSegments Distribution:")
    seg_counts = {}
    for n in stats['num_segments_distribution']:
        seg_counts[n] = seg_counts.get(n, 0) + 1
    for n, count in sorted(seg_counts.items()):
        print(f"  - {n} segments: {count} samples")
    
    print(f"\nInstances per scene:")
    for scene, count in stats['instances_per_scene'].items():
        print(f"  - {scene}: {count}")
    
    print(f"\nProcessing time: {stats['processing_time']:.2f} seconds")
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build Multi-Direction Benchmark V2 (Single Image Resize)")
    parser.add_argument("--output", "-o", default="multi_direction_benchmark_v2.h5", help="Output file path")
    parser.add_argument("--instances", "-n", type=int, default=100, help="Instances per scene")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--min-segments", type=int, default=2, help="Minimum direction segments")
    parser.add_argument("--max-segments", type=int, default=4, help="Maximum direction segments")
    parser.add_argument("--steps", type=int, default=8, help="Steps per segment")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of HuggingFace")
    args = parser.parse_args()
    
    config = MultiDirectionConfig()
    config.output_path = args.output
    config.instances_per_scene = args.instances
    config.random_seed = args.seed
    config.min_segments = args.min_segments
    config.max_segments = args.max_segments
    config.steps_per_segment = args.steps
    
    use_hf = not args.synthetic
    
    if not HF_AVAILABLE and use_hf:
        print("Warning: HuggingFace datasets library not available.")
        print("Install with: pip install datasets")
        print("Falling back to synthetic data.\n")
        use_hf = False
    
    builder = MultiDirectionBenchmarkBuilder(config, use_hf=use_hf)
    stats = builder.build()
    
    print_stats(stats)
    
    print("\nBenchmark built successfully!")
    print(f"Output: {config.output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
