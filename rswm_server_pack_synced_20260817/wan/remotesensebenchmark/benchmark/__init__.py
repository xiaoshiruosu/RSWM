"""
RSVideoWise Benchmark - Remote Sensing Video Extrapolation Benchmark

This package implements the RSVideoWise benchmark for remote sensing video 
extrapolation, extending RSWISE from the RemoteBAGEL paper to video sequences.

Key Components:
- Center sequence (V_c): 8-16 frame temporal sequence from center grid
- Direction embed (D): Continuous sin/cos embedding for arbitrary angles (0-360°)
- Target sequence (V_t): Subsequent 8-16 frame sequence along D, with <10% overlap

Modules:
- data_loader: Download and load remote sensing data from Hugging Face
- grid_partition: Spatiotemporal grid partitioning (3x3xT)
- direction_embed: Continuous direction embedding generation
- target_extract: Target sequence extraction with interpolation
- validation: Overlap and direction accuracy validation
- builder: Main benchmark building pipeline
"""

__version__ = "1.0.0"
__author__ = "RSVideoWise Team"

from .data_loader import HuggingFaceDataLoader, SyntheticDataGenerator
from .grid_partition import SpatioTemporalGridPartitioner
from .direction_embed import DirectionEmbedding
from .target_extract import TargetSequenceExtractor
from .validation import OverlapValidator, DirectionValidator
from .builder import RSVideoWiseBenchmarkBuilder

__all__ = [
    "HuggingFaceDataLoader",
    "SyntheticDataGenerator", 
    "SpatioTemporalGridPartitioner",
    "DirectionEmbedding",
    "TargetSequenceExtractor",
    "OverlapValidator",
    "DirectionValidator",
    "RSVideoWiseBenchmarkBuilder",
]

