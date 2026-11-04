#!/usr/bin/env python3
"""
LingBot-World LoRA 训练数据集加载器

加载 HDF5 格式的 LingBot I2V 训练数据，支持与 Wan2.2 类似的数据划分和筛选。
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import random


class LingBotDataset(Dataset):
    """
    PyTorch Dataset 用于 LingBot-World I2V LoRA 训练。
    
    加载包含图像、视频帧、文本提示和相机参数的 HDF5 数据。
    """
    
    def __init__(
        self,
        h5_path: str,
        transform=None,
        normalize: bool = True,
        max_frames: Optional[int] = None,
    ):
        """
        Args:
            h5_path: HDF5 数据集路径
            transform: 可选的数据增强
            normalize: 是否将图像归一化到 [-1, 1]
            max_frames: 最大帧数限制 (None 表示使用数据中的全部)
        """
        self.h5_path = Path(h5_path)
        self.transform = transform
        self.normalize = normalize
        self.max_frames = max_frames
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {h5_path}")
        
        with h5py.File(self.h5_path, 'r') as f:
            if 'metadata' not in f:
                raise ValueError(f"HDF5 格式错误: 缺少 'metadata' 组，请确认由 build_dataset.py 生成: {self.h5_path}")
            self.metadata = dict(f['metadata'].attrs)
            if 'total_instances' not in self.metadata:
                raise ValueError(f"HDF5 格式错误: metadata 中缺少 'total_instances'，请确认由 build_dataset.py 生成: {self.h5_path}")
            self.total_samples = int(self.metadata['total_instances'])
            if 'data' not in f:
                raise ValueError(f"HDF5 格式错误: 缺少 'data' 组: {self.h5_path}")
            self.sample_indices = list(range(self.total_samples))
        
        print(f"加载数据集: {len(self.sample_indices)} 个样本 from {self.h5_path}")
    
    def __len__(self) -> int:
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_idx = self.sample_indices[idx]
        sample_key = f'sample_{sample_idx:05d}'
        
        with h5py.File(self.h5_path, 'r') as f:
            if sample_key not in f['data']:
                raise KeyError(f"HDF5 中缺少样本 '{sample_key}'，请确认数据集由 build_dataset.py 生成且未损坏: {self.h5_path}")
            sample = f['data'][sample_key]
            
            image = sample['image'][:]  # (H, W, 3)
            video_frames = sample['video_frames'][:]  # (T, H, W, 3)
            intrinsics = sample['intrinsics'][:]
            poses = sample['poses'][:]
            prompt = sample.attrs['prompt']
            num_frames = sample.attrs['num_frames']
            
            if self.max_frames is not None and video_frames.shape[0] > self.max_frames:
                n = ((self.max_frames - 1) // 4) * 4 + 1
                n = min(n, video_frames.shape[0])
                video_frames = video_frames[:n]
                intrinsics = intrinsics[:n]
                poses = poses[:n]
                num_frames = n
        
        if self.normalize:
            image = (image / 127.5) - 1.0
            video_frames = (video_frames / 127.5) - 1.0
        
        result = {
            'image': torch.from_numpy(image).permute(2, 0, 1).float(),
            'video_frames': torch.from_numpy(video_frames).permute(0, 3, 1, 2).float(),
            'intrinsics': torch.from_numpy(intrinsics).float(),
            'poses': torch.from_numpy(poses).float(),
            'prompt': prompt,
            'num_frames': num_frames,
        }
        
        if self.transform is not None:
            result = self.transform(result)
        
        return result


def split_dataset(
    dataset: LingBotDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset, List[int], List[int]]:
    """
    将数据集划分为训练集和验证集。
    返回 (train_subset, val_subset, train_indices, val_indices)。
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6
    
    random.seed(seed)
    np.random.seed(seed)
    
    total_size = len(dataset)
    indices = list(range(total_size))
    random.shuffle(indices)
    
    split_idx = max(1, int(total_size * train_ratio))  # 至少保留 1 个训练样本
    split_idx = min(split_idx, total_size)  # 单样本时全部用于训练
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    print(f"数据划分: {len(train_subset)} 训练, {len(val_subset)} 验证 (seed={seed})")
    
    return train_subset, val_subset, train_indices, val_indices
