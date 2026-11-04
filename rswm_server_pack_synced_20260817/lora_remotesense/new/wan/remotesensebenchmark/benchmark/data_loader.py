"""
Data Loader Module for RSVideoWise Benchmark

Handles downloading remote sensing data from Hugging Face and 
generates synthetic fallback data when needed.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Generator
from pathlib import Path
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RemoteSensingSequence:
    """Container for a remote sensing temporal sequence"""
    data: np.ndarray  # Shape: (T, H, W, C) or (T, C, H, W)
    scene_category: str
    source_dataset: str
    metadata: Dict
    
    @property
    def num_frames(self) -> int:
        return self.data.shape[0]
    
    @property
    def spatial_size(self) -> Tuple[int, int]:
        # Assuming (T, H, W, C) format
        return self.data.shape[1], self.data.shape[2]
    
    @property
    def num_channels(self) -> int:
        return self.data.shape[-1]


class HuggingFaceDataLoader:
    """
    Load remote sensing data from Hugging Face datasets.

    Verified and supported datasets:
    - timm/eurosat-rgb: EuroSAT RGB Sentinel-2 imagery (27k samples)
    - blanchon/EuroSAT_RGB: EuroSAT RGB alternate source
    - blanchon/UC_Merced: UC Merced Land Use dataset
    - GFM-Bench/BigEarthNet: BigEarthNet Sentinel-1/2 patches
    - AdaptLLM/remote-sensing-visual-instructions: RS visual instructions
    """
    
    def __init__(
        self,
        dataset_names: List[str],
        cache_dir: Path,
        temporal_length: int = 24,
        image_size: Tuple[int, int] = (512, 512)
    ):
        self.dataset_names = dataset_names
        self.cache_dir = cache_dir
        self.temporal_length = temporal_length
        self.image_size = image_size
        self.datasets = {}
        self._hf_available = False
        
    def initialize(self) -> bool:
        """Initialize and download datasets from Hugging Face"""
        try:
            from datasets import load_dataset
            self._hf_available = True
            logger.info("Hugging Face datasets library available")
            
            for dataset_name in self.dataset_names:
                try:
                    logger.info(f"Loading dataset: {dataset_name}")
                    # Try streaming mode first to avoid large downloads
                    ds = load_dataset(
                        dataset_name, 
                        split="train",
                        streaming=True,
                        trust_remote_code=True
                    )
                    self.datasets[dataset_name] = ds
                    logger.info(f"Successfully loaded: {dataset_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {dataset_name}: {e}")
                    continue
                    
            return len(self.datasets) > 0
            
        except ImportError:
            logger.warning("Hugging Face datasets not available, will use synthetic data")
            self._hf_available = False
            return False
    
    def _create_temporal_sequence(
        self,
        images: List[np.ndarray],
        target_length: int
    ) -> np.ndarray:
        """Create temporal sequence from list of images"""
        if len(images) >= target_length:
            return np.stack(images[:target_length], axis=0)

        # Interpolate to create more frames
        sequence = []
        for i in range(target_length):
            idx = i * (len(images) - 1) / (target_length - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, len(images) - 1)
            alpha = idx - idx_low

            interpolated = (1 - alpha) * images[idx_low] + alpha * images[idx_high]
            sequence.append(interpolated.astype(np.float32))

        return np.stack(sequence, axis=0)

    def load_sequences(
        self,
        scene_category: str,
        num_sequences: int = 100
    ) -> Generator[RemoteSensingSequence, None, None]:
        """Load sequences for a specific scene category"""
        if not self._hf_available or not self.datasets:
            logger.warning("No HF datasets available")
            return

        # Map scene categories to dataset labels
        category_mapping = {
            "general": ["natural", "vegetation", "land"],
            "flood": ["water", "wetland", "river"],
            "urban": ["urban", "city", "residential", "industrial"],
            "rural": ["agricultural", "farm", "pasture", "crop"]
        }

        keywords = category_mapping.get(scene_category, ["general"])
        count = 0

        for dataset_name, dataset in self.datasets.items():
            if count >= num_sequences:
                break

            try:
                for sample in dataset:
                    if count >= num_sequences:
                        break

                    # Extract image from sample
                    image = self._extract_image_from_sample(sample)
                    if image is None:
                        continue

                    # Create synthetic temporal sequence from single image
                    sequence = self._create_synthetic_temporal_from_image(
                        image, self.temporal_length
                    )

                    yield RemoteSensingSequence(
                        data=sequence,
                        scene_category=scene_category,
                        source_dataset=dataset_name,
                        metadata={"sample_idx": count}
                    )
                    count += 1

            except Exception as e:
                logger.warning(f"Error loading from {dataset_name}: {e}")
                continue

    def _extract_image_from_sample(self, sample: Dict) -> Optional[np.ndarray]:
        """Extract image array from HuggingFace sample"""
        try:
            # Try common image keys
            for key in ["image", "img", "pixel_values", "input"]:
                if key in sample:
                    img = sample[key]
                    if hasattr(img, "numpy"):
                        return np.array(img)
                    elif isinstance(img, np.ndarray):
                        return img
                    elif hasattr(img, "convert"):
                        # PIL Image
                        return np.array(img.convert("RGB"))
            return None
        except Exception:
            return None

    def _create_synthetic_temporal_from_image(
        self,
        base_image: np.ndarray,
        num_frames: int
    ) -> np.ndarray:
        """Create temporal sequence with synthetic motion from base image"""
        H, W = base_image.shape[:2]
        C = base_image.shape[2] if len(base_image.shape) > 2 else 1

        # Resize if needed
        if H != self.image_size[0] or W != self.image_size[1]:
            try:
                import cv2
                base_image = cv2.resize(base_image, self.image_size)
            except ImportError:
                # Simple resize using numpy
                from scipy.ndimage import zoom
                factors = (self.image_size[0]/H, self.image_size[1]/W, 1)
                base_image = zoom(base_image, factors[:len(base_image.shape)])

        frames = []
        for t in range(num_frames):
            # Add temporal variation (brightness, slight shift)
            variation = 1.0 + 0.1 * np.sin(2 * np.pi * t / num_frames)
            frame = (base_image * variation).clip(0, 255).astype(np.float32)
            frames.append(frame)

        return np.stack(frames, axis=0)


class SyntheticDataGenerator:
    """
    Generate synthetic remote sensing temporal sequences for testing.

    Creates realistic-looking patterns including:
    - Gradient patterns (simulating terrain elevation)
    - Perlin noise (natural textures)
    - Geometric patterns (urban structures)
    - Color variations (vegetation changes)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        num_channels: int = 3,
        temporal_length: int = 24,
        random_seed: Optional[int] = None
    ):
        self.image_size = image_size
        self.num_channels = num_channels
        self.temporal_length = temporal_length

        if random_seed is not None:
            np.random.seed(random_seed)

    def generate_sequence(
        self,
        scene_category: str,
        sequence_id: int = 0
    ) -> RemoteSensingSequence:
        """Generate a synthetic temporal sequence for given scene category"""

        generators = {
            "general": self._generate_general_scene,
            "flood": self._generate_flood_scene,
            "urban": self._generate_urban_scene,
            "rural": self._generate_rural_scene
        }

        generator = generators.get(scene_category, self._generate_general_scene)
        data = generator(sequence_id)

        return RemoteSensingSequence(
            data=data,
            scene_category=scene_category,
            source_dataset="synthetic",
            metadata={"sequence_id": sequence_id, "generator": scene_category}
        )

    def _generate_base_texture(
        self,
        pattern_type: str = "gradient"
    ) -> np.ndarray:
        """Generate base texture pattern"""
        H, W = self.image_size

        if pattern_type == "gradient":
            # Gradient pattern
            x = np.linspace(0, 1, W)
            y = np.linspace(0, 1, H)
            xx, yy = np.meshgrid(x, y)
            texture = (xx + yy) / 2

        elif pattern_type == "perlin":
            # Simplified Perlin-like noise
            texture = self._generate_perlin_like(H, W)

        elif pattern_type == "geometric":
            # Geometric grid pattern
            x = np.linspace(0, 8 * np.pi, W)
            y = np.linspace(0, 8 * np.pi, H)
            xx, yy = np.meshgrid(x, y)
            texture = (np.sin(xx) * np.sin(yy) + 1) / 2

        elif pattern_type == "random":
            texture = np.random.rand(H, W)

        else:
            texture = np.zeros((H, W))

        return texture.astype(np.float32)

    def _generate_perlin_like(self, H: int, W: int, scale: int = 64) -> np.ndarray:
        """Generate Perlin-like noise using octaves"""
        result = np.zeros((H, W))

        for octave in range(4):
            freq = 2 ** octave
            amp = 1 / freq

            # Generate low-res noise and upscale
            small_h = max(4, H // (scale // freq))
            small_w = max(4, W // (scale // freq))
            noise = np.random.rand(small_h, small_w)

            # Bilinear upscale
            from scipy.ndimage import zoom
            upscaled = zoom(noise, (H / small_h, W / small_w), order=1)
            result += upscaled[:H, :W] * amp

        return (result - result.min()) / (result.max() - result.min() + 1e-8)

    def _apply_temporal_evolution(
        self,
        base_frame: np.ndarray,
        num_frames: int,
        evolution_type: str = "smooth"
    ) -> np.ndarray:
        """Apply temporal evolution to create sequence"""
        H, W, C = base_frame.shape
        frames = []

        for t in range(num_frames):
            progress = t / (num_frames - 1)

            if evolution_type == "smooth":
                # Smooth brightness variation
                factor = 0.8 + 0.4 * np.sin(2 * np.pi * progress)
                frame = base_frame * factor

            elif evolution_type == "flood":
                # Rising water level simulation
                water_level = int(H * (0.3 + 0.4 * progress))
                frame = base_frame.copy()
                frame[water_level:, :, :] *= 0.3  # Darken flooded areas
                frame[water_level:, :, 2] = np.minimum(
                    frame[water_level:, :, 2] + 50, 255
                )  # Add blue tint

            elif evolution_type == "urban":
                # Construction/change simulation
                change_mask = np.random.rand(H, W) < (0.1 * progress)
                frame = base_frame.copy()
                for c in range(C):
                    frame[:, :, c] = np.where(
                        change_mask,
                        np.random.rand(H, W) * 128 + 64,
                        frame[:, :, c]
                    )

            elif evolution_type == "rural":
                # Seasonal vegetation changes
                green_factor = 0.5 + 0.5 * np.sin(2 * np.pi * progress)
                frame = base_frame.copy()
                frame[:, :, 1] *= green_factor  # Vary green channel

            else:
                frame = base_frame

            frames.append(frame.clip(0, 255).astype(np.float32))

        return np.stack(frames, axis=0)

    def _generate_general_scene(self, seed: int) -> np.ndarray:
        """Generate general terrain scene"""
        np.random.seed(seed)

        texture = self._generate_base_texture("perlin")
        base_frame = np.stack([
            texture * 180 + 40,  # R
            texture * 160 + 50,  # G
            texture * 140 + 30   # B
        ], axis=-1)

        return self._apply_temporal_evolution(
            base_frame, self.temporal_length, "smooth"
        )

    def _generate_flood_scene(self, seed: int) -> np.ndarray:
        """Generate flood scene with water dynamics"""
        np.random.seed(seed)

        # Base terrain
        texture = self._generate_base_texture("gradient")
        base_frame = np.stack([
            texture * 120 + 60,   # R - earth tones
            texture * 140 + 80,   # G - vegetation
            texture * 80 + 40     # B - low blue initially
        ], axis=-1)

        return self._apply_temporal_evolution(
            base_frame, self.temporal_length, "flood"
        )

    def _generate_urban_scene(self, seed: int) -> np.ndarray:
        """Generate urban scene with geometric patterns"""
        np.random.seed(seed)

        texture = self._generate_base_texture("geometric")
        noise = self._generate_base_texture("random") * 0.3
        combined = texture * 0.7 + noise

        base_frame = np.stack([
            combined * 150 + 50,   # R - concrete gray
            combined * 140 + 50,   # G
            combined * 160 + 40    # B - slight blue tint
        ], axis=-1)

        return self._apply_temporal_evolution(
            base_frame, self.temporal_length, "urban"
        )

    def _generate_rural_scene(self, seed: int) -> np.ndarray:
        """Generate rural agricultural scene"""
        np.random.seed(seed)

        texture = self._generate_base_texture("perlin")

        # Agricultural patterns (stripes)
        H, W = self.image_size
        x = np.linspace(0, 16 * np.pi, W)
        stripes = (np.sin(x) + 1) / 2
        stripes = np.tile(stripes, (H, 1))

        combined = texture * 0.6 + stripes * 0.4

        base_frame = np.stack([
            combined * 80 + 40,    # R - low red
            combined * 180 + 60,   # G - high green (vegetation)
            combined * 60 + 30     # B - low blue
        ], axis=-1)

        return self._apply_temporal_evolution(
            base_frame, self.temporal_length, "rural"
        )

