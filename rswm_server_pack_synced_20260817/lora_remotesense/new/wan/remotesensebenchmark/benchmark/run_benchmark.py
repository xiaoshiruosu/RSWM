#!/usr/bin/env python3
"""
RSVideoWise Benchmark Builder - Main Entry Point

This script builds the complete RSVideoWise benchmark for remote sensing 
video extrapolation. It generates 2000 instances (500 per scene category)
with center sequences, direction embeddings, and target sequences.

Usage:
    python -m benchmark.run_benchmark [options]

Options:
    --output: Output file path (default: benchmark_output/rsvideowise_benchmark.h5)
    --instances: Instances per scene (default: 500)
    --seed: Random seed (default: 42)
    --synthetic-only: Force synthetic data generation
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for benchmark building"""
    parser = argparse.ArgumentParser(
        description="Build RSVideoWise Benchmark for Remote Sensing Video Extrapolation"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_output/rsvideowise_benchmark.h5",
        help="Output file path (HDF5 format)"
    )
    parser.add_argument(
        "--instances", "-n",
        type=int,
        default=500,
        help="Number of instances per scene category"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Use only synthetic data (skip HuggingFace download)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import here to avoid circular imports
    from .config import BenchmarkConfig
    from .builder import RSVideoWiseBenchmarkBuilder
    
    print("=" * 70)
    print("RSVideoWise Benchmark Builder")
    print("Remote Sensing Video Extrapolation Benchmark")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  - Output: {args.output}")
    print(f"  - Instances per scene: {args.instances}")
    print(f"  - Total instances: {args.instances * 4}")
    print(f"  - Scene categories: general, flood, urban, rural")
    print(f"  - Random seed: {args.seed}")
    print(f"  - Data source: {'synthetic only' if args.synthetic_only else 'HuggingFace + fallback'}")
    print()
    
    # Configure
    config = BenchmarkConfig()
    config.instances_per_scene = args.instances
    config.total_instances = args.instances * len(config.data.scene_categories)
    config.random_seed = args.seed
    
    if args.synthetic_only:
        config.data.hf_datasets = []  # Disable HuggingFace
    
    # Build
    try:
        builder = RSVideoWiseBenchmarkBuilder(config)
        stats = builder.build(Path(args.output))
        
        # Print results
        print()
        print(stats.summary())
        print()
        print("Benchmark built successfully!")
        print(f"Output saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark building failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

