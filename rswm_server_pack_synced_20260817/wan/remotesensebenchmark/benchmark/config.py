"""
Configuration settings for RSVideoWise Benchmark
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading configuration"""
    # Hugging Face dataset sources for remote sensing
    # All datasets verified to exist on HuggingFace as of 2024
    hf_datasets: List[str] = field(default_factory=lambda: [
        "timm/eurosat-rgb",                           # EuroSAT RGB - Sentinel-2 land use (27k images)
        "blanchon/EuroSAT_RGB",                       # EuroSAT RGB alternate source
        "blanchon/UC_Merced",                         # UC Merced Land Use dataset
        "GFM-Bench/BigEarthNet",                      # BigEarthNet - Sentinel-1/2 patches
        "AdaptLLM/remote-sensing-visual-instructions", # RS visual instructions (36k samples)
    ])
    
    # Fallback synthetic data settings
    use_synthetic_fallback: bool = True
    synthetic_image_size: Tuple[int, int] = (512, 512)
    synthetic_channels: int = 3
    synthetic_temporal_length: int = 24  # Total frames available
    
    # Scene categories
    scene_categories: List[str] = field(default_factory=lambda: [
        "general", "flood", "urban", "rural"
    ])
    
    # Data cache directory
    cache_dir: Path = field(default_factory=lambda: Path("./data_cache"))


@dataclass
class SequenceConfig:
    """Sequence generation configuration"""
    # Center sequence parameters
    center_min_frames: int = 8
    center_max_frames: int = 16
    
    # Target sequence parameters  
    target_min_frames: int = 8
    target_max_frames: int = 16
    
    # Overlap constraint
    max_overlap_ratio: float = 0.10  # <10% overlap
    
    # Grid configuration
    grid_size: Tuple[int, int] = (3, 3)  # 3x3 spatial grid
    
    # Patch size for each grid cell
    patch_size: Tuple[int, int] = (128, 128)


@dataclass
class EmbeddingConfig:
    """Direction embedding configuration"""
    # Embedding dimensions
    embed_dim: int = 64
    
    # Number of frequency bands for positional encoding
    num_freq_bands: int = 16
    
    # Angle range (degrees)
    min_angle: float = 0.0
    max_angle: float = 360.0
    
    # Include magnitude information
    include_magnitude: bool = True


@dataclass
class BenchmarkConfig:
    """Main benchmark configuration"""
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./benchmark_output"))
    output_filename: str = "rsvideowise_benchmark.h5"
    
    # Instance counts per scene category
    instances_per_scene: int = 500
    total_instances: int = 2000  # 4 scenes x 500 instances
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    
    # Processing settings
    num_workers: int = 0
    batch_size: int = 32
    
    # Validation settings
    validate_overlap: bool = True
    validate_direction: bool = True
    
    def __post_init__(self):
        """Ensure directories exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = BenchmarkConfig()
