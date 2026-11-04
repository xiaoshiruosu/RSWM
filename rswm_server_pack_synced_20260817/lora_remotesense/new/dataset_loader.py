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
import json


def _truncation_allowed() -> bool:
    """[v2.9.1 C7] 帧数截断的逃生口。默认关闭 —— 截断是静默污染源。"""
    import os as _os
    return _os.environ.get("RSWM_ALLOW_FRAME_TRUNCATION", "0").strip().lower() \
        not in ("0", "false", "no", "off", "")


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
                # =============================================================
                # [v2.9.1 C7] 真正的截断在这里,不在 train.py 的 _collate_fn。
                #
                # v2.8.1 给 collate 加了"非 4n+1 / 批内混长度即硬错"的门,但
                # 那道门永远看不到这次截断:本处先把 33 → 17(仍是 4n+1、批内
                # 长度一致),collate 拿到的已经是"干净"的 17 帧。
                #
                # 而这次截断恰恰是最危险的一种:prompt 的净方向句、waypoints
                # 真值、语义门判定全按【完整】轨迹算,视频却只走了前半程 ——
                # K=4 的样本被当 K=2 训,模型被要求学一个数据里不存在的对应
                # 关系,症状完全静默(loss 照降、数字照出)。
                # canonical MAX_FRAMES=33 不触发;一旦为 OOM 排障降到 17 就中招。
                # =============================================================
                n = ((self.max_frames - 1) // 4) * 4 + 1
                n = min(n, video_frames.shape[0])
                if not _truncation_allowed():
                    raise ValueError(
                        f"样本 {key if 'key' in dir() else ''} 有 "
                        f"{video_frames.shape[0]} 帧 > max_frames={self.max_frames},"
                        f"截到 {n} 帧会让视频只走完轨迹的前 "
                        f"{n / video_frames.shape[0]:.0%},而 prompt 净方向 / "
                        f"waypoints 真值 / 语义门判定仍是全程口径 —— 训练目标与"
                        f"数据不符,且完全静默。\n"
                        f"canonical 用 MAX_FRAMES=33(覆盖 K=2..4 的 17/25/33)。"
                        f"确要为 OOM 排障降帧数,请显式 "
                        f"RSWM_ALLOW_FRAME_TRUNCATION=1,并且【不得】把该次产物"
                        f"用于任何 headline 数字。")
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


def split_dataset_from_file(
    dataset: LingBotDataset,
    split_file: str,
) -> Tuple[Subset, Subset, List[int], List[int]]:
    """从预注册划分构造子集；official test 永不进入训练器。"""
    p = Path(split_file)
    if not p.is_file():
        raise FileNotFoundError(f"划分文件不存在: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if "train_indices" in doc and "val_indices" in doc:
        train_indices = [int(i) for i in doc["train_indices"]]
        val_indices = [int(i) for i in doc["val_indices"]]
    elif isinstance(doc.get("splits"), dict):
        try:
            train_indices = [int(i) for i in doc["splits"]["train"]["indices"]]
            val_indices = [int(i) for i in doc["splits"]["val"]["indices"]]
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"{p}: official split 缺 splits.train/val.indices") from e
    else:
        raise ValueError(
            f"{p}: 不支持的 split schema；需要 train_indices/val_indices，"
            f"或 splits.train|val.indices")

    total = len(dataset)
    for name, values in (("train", train_indices), ("val", val_indices)):
        if len(values) != len(set(values)):
            raise ValueError(f"{p}: {name} split 内有重复索引")
        bad = [i for i in values if i < 0 or i >= total]
        if bad:
            raise IndexError(
                f"{p}: {name} 有 {len(bad)} 个越界索引，示例 {bad[:5]}；"
                f"数据集长度={total}")
    leak = sorted(set(train_indices) & set(val_indices))
    if leak:
        raise ValueError(f"{p}: train/val 交集非空，示例 {leak[:5]}")
    if not train_indices or not val_indices:
        raise ValueError(f"{p}: train/val 都必须非空")

    print(f"数据划分(预注册): {len(train_indices)} 训练, {len(val_indices)} 验证 "
          f"from {p}")
    return (Subset(dataset, train_indices), Subset(dataset, val_indices),
            train_indices, val_indices)
