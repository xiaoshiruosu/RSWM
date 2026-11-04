"""
RSVideoWise Benchmark Builder

Main orchestrator for building the complete RSVideoWise benchmark.
Generates 2000 instances (500 per scene category) with:
- Center sequences (V_c)
- Direction embeddings (D)
- Target sequences (V_t)
"""

import numpy as np
import h5py
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
import logging
import time
from tqdm import tqdm

from .config import BenchmarkConfig, DEFAULT_CONFIG
from .data_loader import HuggingFaceDataLoader, SyntheticDataGenerator, RemoteSensingSequence
from .grid_partition import SpatioTemporalGridPartitioner
from .direction_embed import DirectionEmbedding
from .target_extract import TargetSequenceExtractor
from .validation import OverlapValidator, DirectionValidator, validate_sample

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkStats:
    """Statistics for the benchmark building process"""
    total_instances: int = 0
    valid_instances: int = 0
    invalid_instances: int = 0
    instances_per_scene: Dict[str, int] = field(default_factory=dict)
    overlap_ratios: List[float] = field(default_factory=list)
    direction_angles: List[float] = field(default_factory=list)
    processing_time: float = 0.0
    data_source: str = "unknown"
    
    def summary(self) -> str:
        """Generate summary string"""
        avg_overlap = np.mean(self.overlap_ratios) if self.overlap_ratios else 0
        lines = [
            "=" * 60,
            "RSVideoWise Benchmark Statistics",
            "=" * 60,
            f"Total instances: {self.total_instances}",
            f"Valid instances: {self.valid_instances}",
            f"Invalid instances: {self.invalid_instances}",
            f"Data source: {self.data_source}",
            f"Average overlap ratio: {avg_overlap:.2%}",
            f"Processing time: {self.processing_time:.2f} seconds",
            "",
            "Instances per scene:",
        ]
        for scene, count in self.instances_per_scene.items():
            lines.append(f"  - {scene}: {count}")
        lines.append("=" * 60)
        return "\n".join(lines)


class RSVideoWiseBenchmarkBuilder:
    """
    Main builder class for RSVideoWise benchmark.
    
    Workflow:
    1. Load/generate remote sensing temporal sequences
    2. Partition into 3x3xT spatiotemporal grids
    3. Extract center sequences (V_c)
    4. Generate direction embeddings (D)
    5. Extract target sequences (V_t) with <10% overlap
    6. Validate and save to HDF5
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        """
        Initialize benchmark builder.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self.partitioner = SpatioTemporalGridPartitioner(
            grid_size=self.config.sequence.grid_size,
            max_overlap=0.05
        )
        
        self.direction_embedder = DirectionEmbedding(
            embed_dim=self.config.embedding.embed_dim,
            num_freq_bands=self.config.embedding.num_freq_bands,
            include_magnitude=self.config.embedding.include_magnitude
        )
        
        self.target_extractor = TargetSequenceExtractor(
            max_overlap_ratio=self.config.sequence.max_overlap_ratio,
            min_frames=self.config.sequence.target_min_frames,
            max_frames=self.config.sequence.target_max_frames
        )
        
        self.overlap_validator = OverlapValidator(
            max_overlap_ratio=self.config.sequence.max_overlap_ratio
        )
        
        self.direction_validator = DirectionValidator()
        
        # Data source (will be initialized in build())
        self.data_loader = None
        self.synthetic_generator = None
        
        # Statistics
        self.stats = BenchmarkStats()
    
    def _initialize_data_source(self) -> str:
        """Initialize data source (HuggingFace or synthetic)"""
        # Try HuggingFace first
        self.data_loader = HuggingFaceDataLoader(
            dataset_names=self.config.data.hf_datasets,
            cache_dir=self.config.data.cache_dir,
            temporal_length=self.config.data.synthetic_temporal_length,
            image_size=self.config.data.synthetic_image_size
        )
        
        if self.data_loader.initialize():
            logger.info("Using HuggingFace datasets")
            return "huggingface"
        
        # Fallback to synthetic data
        if self.config.data.use_synthetic_fallback:
            logger.info("Falling back to synthetic data generation")
            self.synthetic_generator = SyntheticDataGenerator(
                image_size=self.config.data.synthetic_image_size,
                num_channels=self.config.data.synthetic_channels,
                temporal_length=self.config.data.synthetic_temporal_length,
                random_seed=self.config.random_seed
            )
            return "synthetic"
        
        raise RuntimeError("No data source available")
    
    def _generate_sample(
        self,
        sequence: RemoteSensingSequence,
        sample_idx: int
    ) -> Optional[Dict]:
        """Generate a single benchmark sample"""
        # Random number of frames
        num_frames = np.random.randint(
            self.config.sequence.center_min_frames,
            self.config.sequence.center_max_frames + 1
        )
        
        # Random direction angle
        direction_angle = np.random.uniform(0, 360)
        
        # Get sequence data
        full_seq = sequence.data
        T, H, W, C = full_seq.shape

        # Extract center sequence
        center_seq, center_metadata = self.partitioner.extract_center_sequence(
            full_seq, num_frames
        )
        center_bounds = center_metadata["spatial_bounds"]

        # Generate direction embedding
        direction_result = self.direction_embedder.generate(direction_angle)

        # Extract target sequence
        extraction_result = self.target_extractor.extract(
            full_seq,
            center_bounds,
            direction_angle,
            num_frames
        )

        # Validate
        sample_dict = {
            "center_sequence": extraction_result.center_sequence,
            "target_sequence": extraction_result.target_sequence,
            "center_bounds": center_bounds,
            "target_bounds": extraction_result.metadata.get("target_bounds"),
            "direction_angle": direction_angle,
            "direction_embedding": direction_result.to_numpy(),
        }

        validation = validate_sample(
            sample_dict,
            max_overlap=self.config.sequence.max_overlap_ratio
        )

        if not validation.is_valid and self.config.validate_overlap:
            logger.debug(f"Sample {sample_idx} invalid: {validation.errors}")
            return None

        # Build final sample
        return {
            "sample_id": sample_idx,
            "scene_category": sequence.scene_category,
            "source_dataset": sequence.source_dataset,
            "center_sequence": extraction_result.center_sequence,
            "target_sequence": extraction_result.target_sequence,
            "direction_embedding": direction_result.to_numpy(),
            "direction_angle": direction_angle,
            "direction_vector": direction_result.direction_vector,
            "overlap_ratio": extraction_result.overlap_ratio,
            "num_frames": num_frames,
            "center_bounds": center_bounds,
            "target_bounds": extraction_result.metadata.get("target_bounds"),
            "is_valid": validation.is_valid,
        }

    def build(self, output_path: Optional[Path] = None) -> BenchmarkStats:
        """
        Build the complete RSVideoWise benchmark.

        Args:
            output_path: Custom output path (uses config default if None)

        Returns:
            BenchmarkStats with generation statistics
        """
        start_time = time.time()

        if output_path is None:
            output_path = self.config.output_dir / self.config.output_filename

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize data source
        data_source = self._initialize_data_source()
        self.stats.data_source = data_source

        logger.info(f"Building RSVideoWise benchmark with {data_source} data")
        logger.info(f"Target: {self.config.total_instances} instances")

        # Set random seed
        np.random.seed(self.config.random_seed)

        all_samples = []

        # Process each scene category
        for scene_category in self.config.data.scene_categories:
            logger.info(f"Processing scene: {scene_category}")
            scene_samples = self._process_scene(
                scene_category,
                self.config.instances_per_scene
            )
            all_samples.extend(scene_samples)
            self.stats.instances_per_scene[scene_category] = len(scene_samples)

        # Save to HDF5
        self._save_to_hdf5(all_samples, output_path)

        # Finalize stats
        self.stats.total_instances = len(all_samples)
        self.stats.valid_instances = sum(1 for s in all_samples if s.get("is_valid", True))
        self.stats.invalid_instances = self.stats.total_instances - self.stats.valid_instances
        self.stats.processing_time = time.time() - start_time

        logger.info(f"Benchmark built successfully: {output_path}")

        return self.stats

    def _process_scene(
        self,
        scene_category: str,
        target_count: int
    ) -> List[Dict]:
        """Process a single scene category"""
        samples = []
        attempts = 0
        max_attempts = target_count * 3  # Allow some failures

        pbar = tqdm(total=target_count, desc=f"  {scene_category}")

        while len(samples) < target_count and attempts < max_attempts:
            attempts += 1

            # Generate sequence
            if self.synthetic_generator:
                sequence = self.synthetic_generator.generate_sequence(
                    scene_category,
                    sequence_id=attempts
                )
            else:
                # This would be from HuggingFace - simplified here
                sequence = self._get_hf_sequence(scene_category, attempts)
                if sequence is None:
                    # Fallback to synthetic if HF fails
                    if self.synthetic_generator is None:
                        self.synthetic_generator = SyntheticDataGenerator(
                            random_seed=self.config.random_seed
                        )
                    sequence = self.synthetic_generator.generate_sequence(
                        scene_category,
                        sequence_id=attempts
                    )

            # Generate sample
            sample = self._generate_sample(sequence, len(samples))

            if sample is not None:
                samples.append(sample)
                self.stats.overlap_ratios.append(sample["overlap_ratio"])
                self.stats.direction_angles.append(sample["direction_angle"])
                pbar.update(1)

        pbar.close()
        return samples

    def _get_hf_sequence(
        self,
        scene_category: str,
        idx: int
    ) -> Optional[RemoteSensingSequence]:
        """Get sequence from HuggingFace loader"""
        if self.data_loader is None:
            return None

        try:
            for seq in self.data_loader.load_sequences(scene_category, num_sequences=1):
                return seq
        except Exception as e:
            logger.debug(f"HF load failed: {e}")
        return None

    def _save_to_hdf5(
        self,
        samples: List[Dict],
        output_path: Path
    ) -> None:
        """Save all samples to HDF5 file"""
        logger.info(f"Saving {len(samples)} samples to {output_path}")

        with h5py.File(output_path, 'w') as f:
            # Create metadata group
            meta = f.create_group('metadata')
            meta.attrs['version'] = '1.0.0'
            meta.attrs['total_instances'] = len(samples)
            meta.attrs['scene_categories'] = self.config.data.scene_categories
            meta.attrs['max_overlap_ratio'] = self.config.sequence.max_overlap_ratio
            meta.attrs['embed_dim'] = self.config.embedding.embed_dim
            meta.attrs['creation_time'] = time.strftime('%Y-%m-%d %H:%M:%S')

            # Create data group
            data = f.create_group('data')

            # Store each sample
            for idx, sample in enumerate(tqdm(samples, desc="Saving")):
                sample_grp = data.create_group(f'sample_{idx:05d}')

                # Store arrays
                sample_grp.create_dataset(
                    'center_sequence',
                    data=sample['center_sequence'],
                    compression='gzip'
                )
                sample_grp.create_dataset(
                    'target_sequence',
                    data=sample['target_sequence'],
                    compression='gzip'
                )
                sample_grp.create_dataset(
                    'direction_embedding',
                    data=sample['direction_embedding']
                )

                # Store attributes
                sample_grp.attrs['sample_id'] = sample['sample_id']
                sample_grp.attrs['scene_category'] = sample['scene_category']
                sample_grp.attrs['source_dataset'] = sample['source_dataset']
                sample_grp.attrs['direction_angle'] = sample['direction_angle']
                sample_grp.attrs['overlap_ratio'] = sample['overlap_ratio']
                sample_grp.attrs['num_frames'] = sample['num_frames']
                sample_grp.attrs['direction_vector'] = sample['direction_vector']
                sample_grp.attrs['center_bounds'] = sample['center_bounds']
                if sample['target_bounds']:
                    sample_grp.attrs['target_bounds'] = sample['target_bounds']

            # Create index by scene category
            index = f.create_group('index')
            for scene in self.config.data.scene_categories:
                scene_indices = [
                    i for i, s in enumerate(samples)
                    if s['scene_category'] == scene
                ]
                index.create_dataset(scene, data=np.array(scene_indices))

        logger.info(f"Saved to {output_path}")


def build_benchmark(
    output_path: Optional[str] = None,
    instances_per_scene: int = 500,
    random_seed: int = 42
) -> BenchmarkStats:
    """
    Convenience function to build the benchmark.

    Args:
        output_path: Output file path
        instances_per_scene: Number of instances per scene category
        random_seed: Random seed for reproducibility

    Returns:
        BenchmarkStats with generation statistics
    """
    config = BenchmarkConfig()
    config.instances_per_scene = instances_per_scene
    config.total_instances = instances_per_scene * len(config.data.scene_categories)
    config.random_seed = random_seed

    builder = RSVideoWiseBenchmarkBuilder(config)

    if output_path:
        return builder.build(Path(output_path))
    return builder.build()

