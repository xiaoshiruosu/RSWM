#!/usr/bin/env python3
"""
RSVideoWise Benchmark Builder - Standalone Script

Builds the RSVideoWise benchmark for remote sensing video extrapolation.
This benchmark extends RSWISE from RemoteBAGEL to video sequences.

Key Components:
- Center sequence (V_c): 8-16 frame temporal sequence from center grid
- Direction embed (D): Continuous sin/cos embedding for arbitrary angles (0-360°)
- Target sequence (V_t): Subsequent 8-16 frame sequence along D, with <10% overlap

Handles 4 scene categories: general, flood, urban, rural
Generates 2000 total instances (500 per scene)

Output: HDF5 file with all benchmark data
"""

import os
import sys

# Add benchmark module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import h5py
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Benchmark configuration"""
    output_path: str = "rsvideowise_benchmark.h5"
    instances_per_scene: int = 500
    scene_categories: List[str] = field(default_factory=lambda: ["general", "flood", "urban", "rural"])
    
    # Sequence parameters
    min_frames: int = 8
    max_frames: int = 16
    image_size: Tuple[int, int] = (512, 512)
    num_channels: int = 3
    temporal_length: int = 24
    
    # Embedding parameters
    embed_dim: int = 64
    num_freq_bands: int = 16
    
    # Validation
    max_overlap_ratio: float = 0.10
    
    # Random seed
    random_seed: int = 42


# ============================================================================
# Data Generation
# ============================================================================

class SyntheticDataGenerator:
    """Generate synthetic remote sensing temporal sequences"""
    
    def __init__(self, config: Config):
        self.config = config
        np.random.seed(config.random_seed)
    
    def generate_sequence(self, scene_category: str, seq_id: int) -> np.ndarray:
        """Generate synthetic temporal sequence for scene category"""
        np.random.seed(self.config.random_seed + seq_id)
        
        H, W = self.config.image_size
        T = self.config.temporal_length
        C = self.config.num_channels
        
        # Generate base texture based on scene type
        if scene_category == "flood":
            sequence = self._generate_flood_scene(T, H, W, C)
        elif scene_category == "urban":
            sequence = self._generate_urban_scene(T, H, W, C)
        elif scene_category == "rural":
            sequence = self._generate_rural_scene(T, H, W, C)
        else:  # general
            sequence = self._generate_general_scene(T, H, W, C)
        
        return sequence.astype(np.float32)
    
    def _generate_perlin_like(self, H: int, W: int) -> np.ndarray:
        """Generate Perlin-like noise texture"""
        result = np.zeros((H, W))
        for octave in range(4):
            freq = 2 ** octave
            amp = 1 / freq
            small_h = max(8, H // (32 // freq))
            small_w = max(8, W // (32 // freq))
            noise = np.random.rand(small_h, small_w)
            
            # Upscale using linear interpolation
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
    
    def _generate_general_scene(self, T, H, W, C) -> np.ndarray:
        """Generate general terrain scene"""
        texture = self._generate_perlin_like(H, W)
        frames = []
        for t in range(T):
            factor = 0.8 + 0.4 * np.sin(2 * np.pi * t / T)
            frame = np.stack([
                texture * 180 * factor + 40,
                texture * 160 * factor + 50,
                texture * 140 * factor + 30
            ], axis=-1)
            frames.append(frame.clip(0, 255))
        return np.stack(frames, axis=0)

    def _generate_flood_scene(self, T, H, W, C) -> np.ndarray:
        """Generate flood scene with water dynamics"""
        texture = self._generate_perlin_like(H, W)
        frames = []
        for t in range(T):
            progress = t / (T - 1)
            water_level = int(H * (0.3 + 0.4 * progress))

            frame = np.stack([
                texture * 120 + 60,
                texture * 140 + 80,
                texture * 80 + 40
            ], axis=-1)

            # Add water effect
            frame[water_level:, :, :2] *= 0.3
            frame[water_level:, :, 2] = np.minimum(frame[water_level:, :, 2] + 80, 255)
            frames.append(frame.clip(0, 255))
        return np.stack(frames, axis=0)

    def _generate_urban_scene(self, T, H, W, C) -> np.ndarray:
        """Generate urban scene with geometric patterns"""
        # Grid pattern for buildings
        x = np.linspace(0, 8 * np.pi, W)
        y = np.linspace(0, 8 * np.pi, H)
        xx, yy = np.meshgrid(x, y)
        texture = (np.sin(xx) * np.sin(yy) + 1) / 2
        noise = np.random.rand(H, W) * 0.3
        combined = texture * 0.7 + noise

        frames = []
        for t in range(T):
            factor = 0.9 + 0.2 * np.sin(2 * np.pi * t / T)
            frame = np.stack([
                combined * 150 * factor + 50,
                combined * 140 * factor + 50,
                combined * 160 * factor + 40
            ], axis=-1)
            frames.append(frame.clip(0, 255))
        return np.stack(frames, axis=0)

    def _generate_rural_scene(self, T, H, W, C) -> np.ndarray:
        """Generate rural agricultural scene"""
        texture = self._generate_perlin_like(H, W)

        # Agricultural stripes
        x = np.linspace(0, 16 * np.pi, W)
        stripes = (np.sin(x) + 1) / 2
        stripes = np.tile(stripes, (H, 1))
        combined = texture * 0.6 + stripes * 0.4

        frames = []
        for t in range(T):
            green_factor = 0.5 + 0.5 * np.sin(2 * np.pi * t / T)
            frame = np.stack([
                combined * 80 + 40,
                combined * 180 * green_factor + 60,
                combined * 60 + 30
            ], axis=-1)
            frames.append(frame.clip(0, 255))
        return np.stack(frames, axis=0)


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
# Sequence Extraction
# ============================================================================

class SequenceExtractor:
    """Extract center and target sequences with minimal overlap"""

    def __init__(self, config: Config):
        self.config = config

    def extract(
        self,
        full_sequence: np.ndarray,
        direction_angle: float,
        num_frames: int
    ) -> Dict:
        """Extract center and target sequences"""
        T, H, W, C = full_sequence.shape

        # Center region (middle third)
        h_start = H // 3
        h_end = 2 * H // 3
        w_start = W // 3
        w_end = 2 * W // 3
        center_h = h_end - h_start
        center_w = w_end - w_start

        # Temporal window
        num_frames = min(num_frames, T)
        t_start = (T - num_frames) // 2
        t_end = t_start + num_frames

        # Extract center sequence
        center_seq = full_sequence[t_start:t_end, h_start:h_end, w_start:w_end, :]
        center_bounds = (h_start, h_end, w_start, w_end)

        # Calculate target region based on direction
        angle_rad = np.deg2rad(direction_angle)
        dx = np.cos(angle_rad)
        dy = -np.sin(angle_rad)  # Negative because row increases downward

        # Shift by 95% of region size for <10% overlap
        shift_h = int(dy * center_h * 0.95)
        shift_w = int(dx * center_w * 0.95)

        # New target bounds
        new_h_start = max(0, min(H - center_h, h_start + shift_h))
        new_h_end = new_h_start + center_h
        new_w_start = max(0, min(W - center_w, w_start + shift_w))
        new_w_end = new_w_start + center_w

        target_bounds = (new_h_start, new_h_end, new_w_start, new_w_end)

        # Extract target sequence
        target_seq = full_sequence[t_start:t_end, new_h_start:new_h_end, new_w_start:new_w_end, :]

        # Calculate overlap
        overlap = self._calculate_overlap(center_bounds, target_bounds)

        return {
            "center_sequence": center_seq,
            "target_sequence": target_seq,
            "center_bounds": center_bounds,
            "target_bounds": target_bounds,
            "overlap_ratio": overlap,
            "num_frames": num_frames,
            "temporal_range": (t_start, t_end)
        }

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

class RSVideoWiseBenchmarkBuilder:
    """Main benchmark builder class"""

    def __init__(self, config: Config):
        self.config = config
        self.data_generator = SyntheticDataGenerator(config)
        self.direction_embedder = DirectionEmbedder(
            embed_dim=config.embed_dim,
            num_freq_bands=config.num_freq_bands
        )
        self.extractor = SequenceExtractor(config)

        # Statistics
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
        print("RSVideoWise Benchmark Builder")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  - Output: {self.config.output_path}")
        print(f"  - Instances per scene: {self.config.instances_per_scene}")
        print(f"  - Scene categories: {self.config.scene_categories}")
        print(f"  - Total instances: {self.config.instances_per_scene * len(self.config.scene_categories)}")
        print(f"  - Max overlap ratio: {self.config.max_overlap_ratio:.0%}")
        print(f"  - Frame range: {self.config.min_frames}-{self.config.max_frames}")
        print(f"  - Image size: {self.config.image_size}")
        print(f"  - Embedding dim: {self.config.embed_dim}")
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
            # Generate sequence
            sequence = self.data_generator.generate_sequence(scene_category, i)

            # Random parameters
            num_frames = np.random.randint(
                self.config.min_frames,
                self.config.max_frames + 1
            )
            direction_angle = np.random.uniform(0, 360)

            # Generate direction embedding
            dir_result = self.direction_embedder.encode(direction_angle)

            # Extract sequences
            extraction = self.extractor.extract(sequence, direction_angle, num_frames)

            # Build sample
            sample = {
                "sample_id": len(samples),
                "scene_category": scene_category,
                "source_dataset": "synthetic",
                "center_sequence": extraction["center_sequence"],
                "target_sequence": extraction["target_sequence"],
                "direction_embedding": dir_result["embedding"],
                "direction_angle": dir_result["angle_degrees"],
                "direction_vector": dir_result["direction_vector"],
                "overlap_ratio": extraction["overlap_ratio"],
                "num_frames": extraction["num_frames"],
                "center_bounds": extraction["center_bounds"],
                "target_bounds": extraction["target_bounds"],
            }

            samples.append(sample)
            self.stats["overlap_ratios"].append(extraction["overlap_ratio"])
            self.stats["direction_angles"].append(direction_angle)

            # Progress
            if (i + 1) % 100 == 0:
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
            meta.attrs['min_frames'] = self.config.min_frames
            meta.attrs['max_frames'] = self.config.max_frames
            meta.attrs['image_size'] = self.config.image_size
            meta.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

            # Data
            data = f.create_group('data')

            for idx, sample in enumerate(samples):
                grp = data.create_group(f'sample_{idx:05d}')

                grp.create_dataset('center_sequence', data=sample['center_sequence'], compression='gzip')
                grp.create_dataset('target_sequence', data=sample['target_sequence'], compression='gzip')
                grp.create_dataset('direction_embedding', data=sample['direction_embedding'])

                grp.attrs['sample_id'] = sample['sample_id']
                grp.attrs['scene_category'] = sample['scene_category']
                grp.attrs['source_dataset'] = sample['source_dataset']
                grp.attrs['direction_angle'] = sample['direction_angle']
                grp.attrs['overlap_ratio'] = sample['overlap_ratio']
                grp.attrs['num_frames'] = sample['num_frames']
                grp.attrs['direction_vector'] = sample['direction_vector']
                grp.attrs['center_bounds'] = sample['center_bounds']
                grp.attrs['target_bounds'] = sample['target_bounds']

            # Index by scene
            index = f.create_group('index')
            for scene in self.config.scene_categories:
                indices = [i for i, s in enumerate(samples) if s['scene_category'] == scene]
                index.create_dataset(scene, data=np.array(indices))

        print(f"Saved successfully to {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def print_stats(stats: Dict) -> None:
    """Print benchmark statistics"""
    print("\n" + "=" * 70)
    print("RSVideoWise Benchmark Statistics")
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


def main():
    """Main function to build the benchmark"""
    import argparse

    parser = argparse.ArgumentParser(description="Build RSVideoWise Benchmark")
    parser.add_argument("--output", "-o", default="rsvideowise_benchmark.h5", help="Output file path")
    parser.add_argument("--instances", "-n", type=int, default=500, help="Instances per scene")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Create config
    config = Config()
    config.output_path = args.output
    config.instances_per_scene = args.instances
    config.random_seed = args.seed

    # Build benchmark
    builder = RSVideoWiseBenchmarkBuilder(config)
    stats = builder.build()

    # Print statistics
    print_stats(stats)

    print("\nBenchmark built successfully!")
    print(f"Output: {config.output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

