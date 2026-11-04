#!/usr/bin/env python3
"""
Multi-Direction Benchmark Dataset Loader for Wan2.2 LoRA Fine-tuning

This module provides PyTorch Dataset classes for loading the Multi-Direction
Sliding Window Benchmark data for training video generation models.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import random


class MultiDirectionDataset(Dataset):
    """
    PyTorch Dataset for Multi-Direction Sliding Window Benchmark.
    
    Loads video sequences with multi-directional motion patterns and their
    corresponding direction embeddings for training video generation models.
    """
    
    def __init__(
        self,
        h5_path: str,
        scene_categories: Optional[List[str]] = None,
        num_segments_filter: Optional[Tuple[int, int]] = None,
        transform=None,
        load_full_scene: bool = False,
        normalize: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            h5_path: Path to the HDF5 benchmark file
            scene_categories: List of scene categories to include (None = all)
            num_segments_filter: Tuple of (min, max) segments to filter (None = all)
            transform: Optional transform to apply to frames
            load_full_scene: Whether to load full scene images
            normalize: Whether to normalize frames to [-1, 1]
        """
        self.h5_path = Path(h5_path)
        self.transform = transform
        self.load_full_scene = load_full_scene
        self.normalize = normalize
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {h5_path}")
        
        # Load metadata and build index
        with h5py.File(self.h5_path, 'r') as f:
            self.metadata = dict(f['metadata'].attrs)
            self.total_samples = self.metadata['total_instances']
            
            # Build sample indices based on filters
            self.sample_indices = []
            for idx in range(self.total_samples):
                sample_key = f'sample_{idx:05d}'
                if sample_key not in f['data']:
                    continue
                
                sample = f['data'][sample_key]
                
                # Filter by scene category
                if scene_categories is not None:
                    scene = sample.attrs['scene_category']
                    if scene not in scene_categories:
                        continue
                
                # Filter by number of segments
                if num_segments_filter is not None:
                    num_seg = sample.attrs['num_segments']
                    min_seg, max_seg = num_segments_filter
                    if num_seg < min_seg or num_seg > max_seg:
                        continue
                
                self.sample_indices.append(idx)
        
        print(f"Loaded dataset: {len(self.sample_indices)} samples from {self.h5_path}")
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Returns:
            Dictionary containing:
                - sliding_frames: Video frames (T, C, H, W)
                - segment_embeddings: Direction embeddings (NumSeg, D)
                - segment_angles: Direction angles (NumSeg,)
                - start_frame: First frame (C, H, W)
                - end_frame: Last frame (C, H, W)
                - full_scene: Full scene image (C, H, W) [optional]
                - metadata: Additional metadata
        """
        sample_idx = self.sample_indices[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            sample_key = f'sample_{sample_idx:05d}'
            sample = f['data'][sample_key]
            
            # Load frames (T, H, W, C) -> (T, C, H, W)
            # V5 benchmark uses center_frame/target_frame; v2 uses start_frame/end_frame
            sliding_frames = sample['sliding_frames'][:]
            if 'start_frame' in sample and 'end_frame' in sample:
                start_frame = sample['start_frame'][:]
                end_frame = sample['end_frame'][:]
            elif 'center_frame' in sample and 'target_frame' in sample:
                start_frame = sample['center_frame'][:]
                end_frame = sample['target_frame'][:]
            else:
                raise KeyError(
                    "Benchmark must have either (start_frame, end_frame) or (center_frame, target_frame). "
                    f"Found keys: {list(sample.keys())}"
                )
            
            # Load direction embeddings
            segment_embeddings = sample['segment_embeddings'][:]
            segment_angles = sample['segment_angles'][:]
            waypoints = sample['waypoints'][:]
            
            # Load metadata
            metadata = {
                'sample_id': sample.attrs['sample_id'],
                'scene_category': sample.attrs['scene_category'],
                'num_segments': sample.attrs['num_segments'],
                'total_frames': sample.attrs['total_frames'],
                'overlap_ratio': sample.attrs['overlap_ratio'],
                'steps_per_segment': sample.attrs['steps_per_segment']
            }
            
            # Optionally load full scene
            full_scene = None
            if self.load_full_scene:
                full_scene = sample['full_scene'][:]
        
        # Normalize frames from [0, 255] to [-1, 1] if requested
        if self.normalize:
            sliding_frames = (sliding_frames / 127.5) - 1.0
            start_frame = (start_frame / 127.5) - 1.0
            end_frame = (end_frame / 127.5) - 1.0
            if full_scene is not None:
                full_scene = (full_scene / 127.5) - 1.0
        
        # Convert to torch tensors and transpose to (C, H, W) format
        result = {
            'sliding_frames': torch.from_numpy(sliding_frames).permute(0, 3, 1, 2).float(),
            'segment_embeddings': torch.from_numpy(segment_embeddings).float(),
            'segment_angles': torch.from_numpy(segment_angles).float(),
            'waypoints': torch.from_numpy(waypoints).float(),
            'start_frame': torch.from_numpy(start_frame).permute(2, 0, 1).float(),
            'end_frame': torch.from_numpy(end_frame).permute(2, 0, 1).float(),
            'metadata': metadata
        }
        
        if full_scene is not None:
            result['full_scene'] = torch.from_numpy(full_scene).permute(2, 0, 1).float()
        
        # Apply transforms if provided
        if self.transform is not None:
            result = self.transform(result)
        
        return result


def split_dataset(
    dataset: MultiDirectionDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
    stratify_by: Optional[str] = None
) -> Tuple[Subset, Subset]:
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: MultiDirectionDataset instance
        train_ratio: Ratio of training samples (default: 0.8)
        val_ratio: Ratio of validation samples (default: 0.2)
        seed: Random seed for reproducibility
        stratify_by: Optional metadata key to stratify by (e.g., 'scene_category')
    
    Returns:
        Tuple of (train_subset, val_subset)
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "train_ratio + val_ratio must equal 1.0"
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    total_size = len(dataset)
    indices = list(range(total_size))
    
    # Stratified split if requested
    if stratify_by is not None:
        # Group indices by stratification key
        groups = {}
        for idx in indices:
            sample_idx = dataset.sample_indices[idx]
            with h5py.File(dataset.h5_path, 'r') as f:
                sample_key = f'sample_{sample_idx:05d}'
                if sample_key in f['data']:
                    sample = f['data'][sample_key]
                    key_value = sample.attrs.get(stratify_by, 'unknown')
                    if key_value not in groups:
                        groups[key_value] = []
                    groups[key_value].append(idx)
        
        # Split each group proportionally
        train_indices = []
        val_indices = []
        for group_indices in groups.values():
            random.shuffle(group_indices)
            split_idx = int(len(group_indices) * train_ratio)
            train_indices.extend(group_indices[:split_idx])
            val_indices.extend(group_indices[split_idx:])
    else:
        # Simple random split
        random.shuffle(indices)
        split_idx = int(total_size * train_ratio)
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    print(f"Dataset split: {len(train_subset)} training samples, {len(val_subset)} validation samples")
    
    return train_subset, val_subset

