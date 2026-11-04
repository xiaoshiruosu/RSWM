#!/usr/bin/env python3
"""
Sliding Window Benchmark Builder for RSVideoWise

This script generates a benchmark dataset that captures the sliding window
transition from center sequence to target sequence along the direction embedding.

Key Features:
- Generates full scene sequences
- Creates sliding window videos showing the transition from center to target
- The window slides along the direction specified by the direction embedding
- Outputs HDF5 format with sliding window frames
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

# Try to import optional dependencies
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

# Supported HuggingFace datasets for each scene category
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

# Global cache for loaded datasets
HF_DATASETS_CACHE: Dict[str, object] = {}


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window benchmark"""
    output_path: str = "sliding_window_benchmark.h5"
    instances_per_scene: int = 100
    scene_categories: List[str] = field(default_factory=lambda: ["general", "flood", "urban", "rural"])
    
    # Full scene parameters
    full_scene_size: Tuple[int, int] = (512, 512)  # Full scene H, W
    num_channels: int = 3
    
    # Sliding window parameters
    window_size: Tuple[int, int] = (128, 128)  # Window H, W
    num_sliding_steps: int = 16  # Number of steps from center to target
    
    # Direction embedding parameters
    embed_dim: int = 64
    num_freq_bands: int = 16
    
    # Validation
    max_overlap_ratio: float = 0.10
    
    # Random seed
    random_seed: int = 42


# ============================================================================
# Data Generation
# ============================================================================

class SceneGenerator:
    """
    Generate scenes from HuggingFace remote sensing datasets.
    Falls back to synthetic data if HF datasets are unavailable.
    """
    
    def __init__(self, config: SlidingWindowConfig, use_hf: bool = True):
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
        """Resize image to target size"""
        H, W = target_size
        h, w = img.shape[:2]
        
        if h == H and w == W:
            return img
        
        if CV2_AVAILABLE:
            return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            h_idx = np.linspace(0, h - 1, H).astype(int)
            w_idx = np.linspace(0, w - 1, W).astype(int)
            return img[np.ix_(h_idx, w_idx)]
    
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
        """Load and process a scene from HuggingFace dataset"""
        try:
            ds_info = self.hf_datasets[scene_category]
            dataset = ds_info["dataset"]
            feature_name = ds_info["feature"]
            
            n = len(dataset)
            if n == 0:
                return None
            
            num_tiles_h = 2
            num_tiles_w = 2
            tile_h = H // num_tiles_h
            tile_w = W // num_tiles_w
            
            scene = np.zeros((H, W, 3), dtype=np.float32)
            
            for th in range(num_tiles_h):
                for tw in range(num_tiles_w):
                    idx = (scene_id * 4 + th * num_tiles_w + tw) % n
                    example = dataset[int(idx)]
                    
                    img_data = example[feature_name]
                    img = self._to_numpy_image(img_data)
                    img_resized = self._resize_image(img, (tile_h, tile_w))
                    
                    y0 = th * tile_h
                    x0 = tw * tile_w
                    scene[y0:y0+tile_h, x0:x0+tile_w] = img_resized
            
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
        
        # Projection matrix
        base_dim = num_freq_bands * 2 + num_freq_bands + 2
        np.random.seed(42)
        self.projection = np.random.randn(base_dim, embed_dim) / np.sqrt(base_dim)
    
    def encode(self, angle_degrees: float, magnitude: float = 1.0) -> Dict:
        """Generate direction embedding for angle"""
        angle_rad = np.deg2rad(angle_degrees % 360)
        
        # Angle encoding (sin/cos at multiple frequencies)
        angle_enc = []
        for freq in self.freq_scales:
            angle_enc.extend([np.sin(angle_rad * freq), np.cos(angle_rad * freq)])
        
        # Magnitude encoding
        mag_norm = np.clip(magnitude / 10.0, 0, 1)
        mag_enc = [np.sin(mag_norm * np.pi * freq) for freq in self.freq_scales]
        
        # Direction vector
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        # Combine and project
        base = np.array(angle_enc + mag_enc + [dx, dy], dtype=np.float32)
        embedding = np.dot(base, self.projection).astype(np.float32)
        
        return {
            "embedding": embedding,
            "angle_degrees": angle_degrees % 360,
            "angle_radians": angle_rad,
            "direction_vector": (float(dx), float(dy))
        }


# ============================================================================
# Sliding Window Extractor
# ============================================================================

class SlidingWindowExtractor:
    """
    Extract sliding window sequence from center to target region.
    
    The sliding window moves from the center position to the target position
    along the direction specified by the direction embedding.
    """
    
    def __init__(self, config: SlidingWindowConfig):
        self.config = config
    
    def extract_sliding_sequence(
        self,
        full_scene: np.ndarray,
        direction_angle: float
    ) -> Dict:
        """
        Extract sliding window sequence from center to target.
        
        Args:
            full_scene: Full scene image (H, W, C)
            direction_angle: Direction angle in degrees (0-360)
        
        Returns:
            Dictionary containing:
            - sliding_frames: Array of frames showing window sliding (N, H, W, C)
            - center_frame: Starting frame (window at center)
            - target_frame: Ending frame (window at target)
            - center_bounds: (h_start, h_end, w_start, w_end)
            - target_bounds: (h_start, h_end, w_start, w_end)
            - trajectory: List of (cy, cx) positions
        """
        H, W, C = full_scene.shape
        win_h, win_w = self.config.window_size
        num_steps = self.config.num_sliding_steps
        
        # Center position
        center_cy = H // 2
        center_cx = W // 2
        
        # Calculate direction vector
        angle_rad = np.deg2rad(direction_angle)
        dx = np.cos(angle_rad)
        dy = -np.sin(angle_rad)  # Negative because y increases downward in image
        
        # Calculate maximum slide distance while staying in bounds
        # Target should be far enough to have <10% overlap
        slide_distance = max(win_h, win_w) * 0.95  # 95% of window size for <10% overlap
        
        # Calculate target position
        target_cx = center_cx + dx * slide_distance
        target_cy = center_cy + dy * slide_distance
        
        # Clamp target position to stay within image bounds
        half_h = win_h // 2
        half_w = win_w // 2
        target_cy = np.clip(target_cy, half_h, H - half_h)
        target_cx = np.clip(target_cx, half_w, W - half_w)
        
        # Generate trajectory from center to target
        trajectory = []
        sliding_frames = []
        
        for step in range(num_steps):
            t = step / (num_steps - 1) if num_steps > 1 else 0
            
            # Linear interpolation from center to target
            cy = int(center_cy + t * (target_cy - center_cy))
            cx = int(center_cx + t * (target_cx - center_cx))
            
            # Calculate window bounds
            h_start = cy - half_h
            h_end = cy + half_h
            w_start = cx - half_w
            w_end = cx + half_w
            
            # Clamp to image bounds
            h_start = max(0, h_start)
            h_end = min(H, h_end)
            w_start = max(0, w_start)
            w_end = min(W, w_end)
            
            # Extract window
            window = full_scene[h_start:h_end, w_start:w_end, :]
            
            # Resize if needed (edge case)
            if window.shape[0] != win_h or window.shape[1] != win_w:
                if CV2_AVAILABLE:
                    window = cv2.resize(window, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
                else:
                    # Simple resize using numpy
                    window = self._simple_resize(window, (win_h, win_w))
            
            sliding_frames.append(window)
            trajectory.append((cy, cx))
        
        sliding_frames = np.stack(sliding_frames, axis=0)
        
        # Get center and target bounds
        center_bounds = (
            center_cy - half_h,
            center_cy + half_h,
            center_cx - half_w,
            center_cx + half_w
        )
        target_bounds = (
            int(target_cy) - half_h,
            int(target_cy) + half_h,
            int(target_cx) - half_w,
            int(target_cx) + half_w
        )
        
        # Calculate overlap ratio
        overlap_ratio = self._calculate_overlap(center_bounds, target_bounds)
        
        return {
            "sliding_frames": sliding_frames,
            "center_frame": sliding_frames[0],
            "target_frame": sliding_frames[-1],
            "center_bounds": center_bounds,
            "target_bounds": target_bounds,
            "trajectory": trajectory,
            "overlap_ratio": overlap_ratio,
            "full_scene": full_scene
        }
    
    def _simple_resize(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Simple resize using nearest neighbor when cv2 not available"""
        h, w = img.shape[:2]
        target_h, target_w = target_size
        
        h_idx = np.linspace(0, h - 1, target_h).astype(int)
        w_idx = np.linspace(0, w - 1, target_w).astype(int)
        
        return img[np.ix_(h_idx, w_idx)]
    
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

class SlidingWindowBenchmarkBuilder:
    """Build sliding window benchmark dataset"""
    
    def __init__(self, config: SlidingWindowConfig, use_hf: bool = True):
        self.config = config
        self.use_hf = use_hf
        self.scene_generator = SceneGenerator(config, use_hf=use_hf)
        self.direction_embedder = DirectionEmbedder(
            embed_dim=config.embed_dim,
            num_freq_bands=config.num_freq_bands
        )
        self.extractor = SlidingWindowExtractor(config)
        
        self.stats = {
            "total_instances": 0,
            "valid_instances": 0,
            "instances_per_scene": {},
            "overlap_ratios": [],
            "direction_angles": [],
            "processing_time": 0.0
        }
    
    def build(self) -> Dict:
        """Build the complete benchmark"""
        start_time = time.time()
        
        print("\n" + "=" * 70)
        print("Sliding Window Benchmark Builder")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  - Output: {self.config.output_path}")
        print(f"  - Instances per scene: {self.config.instances_per_scene}")
        print(f"  - Scene categories: {self.config.scene_categories}")
        print(f"  - Total instances: {self.config.instances_per_scene * len(self.config.scene_categories)}")
        print(f"  - Full scene size: {self.config.full_scene_size}")
        print(f"  - Window size: {self.config.window_size}")
        print(f"  - Sliding steps: {self.config.num_sliding_steps}")
        print(f"  - Data source: {'HuggingFace' if self.use_hf else 'Synthetic'}")
        print()
        
        np.random.seed(self.config.random_seed)
        all_samples = []
        
        for scene in self.config.scene_categories:
            print(f"Processing scene: {scene}")
            samples = self._process_scene(scene)
            all_samples.extend(samples)
            self.stats["instances_per_scene"][scene] = len(samples)
            print(f"  Generated {len(samples)} samples")
        
        # Save to HDF5
        self._save_to_hdf5(all_samples)
        
        # Finalize stats
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
            # Generate full scene
            scene = self.scene_generator.generate_scene(scene_category, i)
            
            # Random direction angle
            direction_angle = np.random.uniform(0, 360)
            
            # Generate direction embedding
            dir_result = self.direction_embedder.encode(direction_angle)
            
            # Extract sliding window sequence
            extraction = self.extractor.extract_sliding_sequence(scene, direction_angle)
            
            # Build sample
            sample = {
                "sample_id": len(samples),
                "scene_category": scene_category,
                "sliding_frames": extraction["sliding_frames"],
                "center_frame": extraction["center_frame"],
                "target_frame": extraction["target_frame"],
                "full_scene": extraction["full_scene"],
                "direction_embedding": dir_result["embedding"],
                "direction_angle": dir_result["angle_degrees"],
                "direction_vector": dir_result["direction_vector"],
                "overlap_ratio": extraction["overlap_ratio"],
                "center_bounds": extraction["center_bounds"],
                "target_bounds": extraction["target_bounds"],
                "trajectory": extraction["trajectory"],
                "num_sliding_steps": self.config.num_sliding_steps
            }
            
            samples.append(sample)
            self.stats["overlap_ratios"].append(extraction["overlap_ratio"])
            self.stats["direction_angles"].append(direction_angle)
            
            # Progress
            if (i + 1) % 20 == 0:
                print(f"    Progress: {i + 1}/{target_count}")
        
        return samples
    
    def _save_to_hdf5(self, samples: List[Dict]) -> None:
        """Save all samples to HDF5 file"""
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving {len(samples)} samples to {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            # Metadata
            meta = f.create_group('metadata')
            meta.attrs['version'] = '1.0.0'
            meta.attrs['total_instances'] = len(samples)
            meta.attrs['scene_categories'] = self.config.scene_categories
            meta.attrs['max_overlap_ratio'] = self.config.max_overlap_ratio
            meta.attrs['embed_dim'] = self.config.embed_dim
            meta.attrs['full_scene_size'] = self.config.full_scene_size
            meta.attrs['window_size'] = self.config.window_size
            meta.attrs['num_sliding_steps'] = self.config.num_sliding_steps
            meta.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Data
            data = f.create_group('data')
            
            for idx, sample in enumerate(samples):
                grp = data.create_group(f'sample_{idx:05d}')
                
                # Store sliding frames (key data)
                grp.create_dataset('sliding_frames', data=sample['sliding_frames'], compression='gzip')
                grp.create_dataset('center_frame', data=sample['center_frame'], compression='gzip')
                grp.create_dataset('target_frame', data=sample['target_frame'], compression='gzip')
                grp.create_dataset('full_scene', data=sample['full_scene'], compression='gzip')
                grp.create_dataset('direction_embedding', data=sample['direction_embedding'])
                grp.create_dataset('trajectory', data=np.array(sample['trajectory']))
                
                grp.attrs['sample_id'] = sample['sample_id']
                grp.attrs['scene_category'] = sample['scene_category']
                grp.attrs['direction_angle'] = sample['direction_angle']
                grp.attrs['overlap_ratio'] = sample['overlap_ratio']
                grp.attrs['direction_vector'] = sample['direction_vector']
                grp.attrs['center_bounds'] = sample['center_bounds']
                grp.attrs['target_bounds'] = sample['target_bounds']
                grp.attrs['num_sliding_steps'] = sample['num_sliding_steps']
            
            # Index by scene
            index = f.create_group('index')
            for scene in self.config.scene_categories:
                indices = [i for i, s in enumerate(samples) if s['scene_category'] == scene]
                index.create_dataset(scene, data=np.array(indices))
        
        print(f"Saved successfully to {output_path}")


# ============================================================================
# Statistics
# ============================================================================

def print_stats(stats: Dict) -> None:
    """Print benchmark statistics"""
    print("\n" + "=" * 70)
    print("Sliding Window Benchmark Statistics")
    print("=" * 70)
    print(f"Total instances: {stats['total_instances']}")
    print(f"Valid instances (<10% overlap): {stats['valid_instances']}")
    print(f"Invalid instances: {stats['total_instances'] - stats['valid_instances']}")
    
    avg_overlap = np.mean(stats['overlap_ratios']) if stats['overlap_ratios'] else 0
    max_overlap = np.max(stats['overlap_ratios']) if stats['overlap_ratios'] else 0
    min_overlap = np.min(stats['overlap_ratios']) if stats['overlap_ratios'] else 0
    
    print(f"\nOverlap Statistics:")
    print(f"  - Average: {avg_overlap:.2%}")
    print(f"  - Min: {min_overlap:.2%}")
    print(f"  - Max: {max_overlap:.2%}")
    
    print(f"\nInstances per scene:")
    for scene, count in stats['instances_per_scene'].items():
        print(f"  - {scene}: {count}")
    
    print(f"\nProcessing time: {stats['processing_time']:.2f} seconds")
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main function to build the sliding window benchmark"""
    parser = argparse.ArgumentParser(description="Build Sliding Window Benchmark")
    parser.add_argument("--output", "-o", default="sliding_window_benchmark.h5", help="Output file path")
    parser.add_argument("--instances", "-n", type=int, default=100, help="Instances per scene")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=16, help="Number of sliding steps")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of HuggingFace")
    args = parser.parse_args()
    
    # Create config
    config = SlidingWindowConfig()
    config.output_path = args.output
    config.instances_per_scene = args.instances
    config.random_seed = args.seed
    config.num_sliding_steps = args.steps
    
    use_hf = not args.synthetic
    
    if not HF_AVAILABLE and use_hf:
        print("Warning: HuggingFace datasets library not available.")
        print("Install with: pip install datasets")
        print("Falling back to synthetic data.\n")
        use_hf = False
    
    # Build benchmark
    builder = SlidingWindowBenchmarkBuilder(config, use_hf=use_hf)
    stats = builder.build()
    
    # Print statistics
    print_stats(stats)
    
    print("\nBenchmark built successfully!")
    print(f"Output: {config.output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
