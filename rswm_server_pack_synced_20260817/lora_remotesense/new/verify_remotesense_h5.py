#!/usr/bin/env python3
"""
验证 RemoteSense 生成的 HDF5 是否与 LingBotDataset 兼容
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if (HERE / "wan").is_dir():
    PROJECT_ROOT = HERE
else:
    PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import numpy as np

def verify_hdf5_structure(h5_path: str):
    """验证 HDF5 文件结构是否符合 LingBotDataset 要求"""

    print(f"验证文件: {h5_path}")
    print("=" * 70)

    try:
        with h5py.File(h5_path, 'r') as f:
            # 检查 metadata
            if 'metadata' not in f:
                print("❌ 缺少 'metadata' 组")
                return False

            metadata = f['metadata']
            required_meta = ['total_instances', 'target_size', 'max_frames']
            for key in required_meta:
                if key not in metadata.attrs:
                    print(f"❌ metadata 缺少属性: {key}")
                    return False

            print(f"✅ metadata 完整")
            print(f"   - total_instances: {metadata.attrs['total_instances']}")
            print(f"   - target_size: {metadata.attrs['target_size']}")
            print(f"   - max_frames: {metadata.attrs['max_frames']}")

            # 检查 data
            if 'data' not in f:
                print("❌ 缺少 'data' 组")
                return False

            data = f['data']
            total_instances = metadata.attrs['total_instances']

            print(f"\n检查样本结构 (共 {total_instances} 个):")

            for idx in range(min(total_instances, 3)):  # 检查前 3 个样本
                sample_key = f'sample_{idx:05d}'

                if sample_key not in data:
                    print(f"❌ 缺少样本: {sample_key}")
                    return False

                sample = data[sample_key]
                print(f"\n样本 {idx}:")

                # 检查必需的 datasets
                required_datasets = ['image', 'video_frames', 'intrinsics', 'poses']
                for ds_name in required_datasets:
                    if ds_name not in sample:
                        print(f"  ❌ 缺少 dataset: {ds_name}")
                        return False
                    print(f"  ✅ {ds_name}: {sample[ds_name].shape}")

                # 检查 num_frames（dataset 或 attrs）
                has_num_frames_dataset = 'num_frames' in sample
                has_num_frames_attr = 'num_frames' in sample.attrs

                if has_num_frames_dataset:
                    num_frames_ds = sample['num_frames'][()]
                    print(f"  ✅ num_frames (dataset): {num_frames_ds}")

                if has_num_frames_attr:
                    num_frames_attr = sample.attrs['num_frames']
                    print(f"  ✅ num_frames (attrs): {num_frames_attr}")

                if not (has_num_frames_dataset or has_num_frames_attr):
                    print(f"  ❌ 缺少 num_frames (dataset 或 attrs)")
                    return False

                # 检查 prompt
                if 'prompt' not in sample.attrs:
                    print(f"  ❌ 缺少 attrs: prompt")
                    return False
                print(f"  ✅ prompt: {sample.attrs['prompt'][:50]}...")

                # 验证数据一致性
                T = sample['video_frames'].shape[0]
                if has_num_frames_attr and sample.attrs['num_frames'] != T:
                    print(f"  ⚠️  num_frames ({sample.attrs['num_frames']}) != video_frames.shape[0] ({T})")

                if sample['intrinsics'].shape[0] != T:
                    print(f"  ❌ intrinsics.shape[0] ({sample['intrinsics'].shape[0]}) != T ({T})")
                    return False

                if sample['poses'].shape[0] != T:
                    print(f"  ❌ poses.shape[0] ({sample['poses'].shape[0]}) != T ({T})")
                    return False

            print("\n" + "=" * 70)
            print("✅ HDF5 结构验证通过！")
            return True

    except Exception as e:
        print(f"\n❌ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_dataset_loader(h5_path: str):
    """使用 LingBotDataset 加载测试"""

    print("\n" + "=" * 70)
    print("测试 LingBotDataset 加载")
    print("=" * 70)

    try:
        from dataset_loader import LingBotDataset

        dataset = LingBotDataset(h5_path, normalize=False)
        print(f"✅ 数据集加载成功，样本数: {len(dataset)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n样本 0 字段:")
            for key, value in sample.items():
                if isinstance(value, str):
                    print(f"  - {key}: {value[:50]}...")
                else:
                    print(f"  - {key}: {value.shape if hasattr(value, 'shape') else type(value)}")

            print("\n✅ LingBotDataset 兼容性测试通过！")
            return True

    except Exception as e:
        print(f"\n❌ LingBotDataset 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python verify_remotesense_h5.py <h5_path>")
        print("示例: python verify_remotesense_h5.py data/remotesense/demo.h5")
        sys.exit(1)

    h5_path = sys.argv[1]

    if not Path(h5_path).exists():
        print(f"❌ 文件不存在: {h5_path}")
        sys.exit(1)

    # 验证结构
    structure_ok = verify_hdf5_structure(h5_path)

    # 测试加载
    loader_ok = test_with_dataset_loader(h5_path)

    # ---- canonical 帧数校验(论文可比性,不只是结构自洽) ----
    #   正确构建器 build_dataset_from_benchmark.py (steps=9) → 帧数 ∈ {17, 25, 33}
#   [v2.8.2] v2.8 起源头做了段界去重(每段首帧只在第一段保留),canonical
#   恰为 17/25/33 且天然 4n+1。**旧值 18/27/36 是 v2.8 之前的**,留着会把
#   正确产物判成"错构建器废品"并给出自信而错误的重建指令。
    #   错构建器   build_from_remotesense.py       (steps=8) → 16/24/32,与论文不可比
    canonical_ok = True
    import os
    try:
        with h5py.File(h5_path, "r") as _f:
            _frames = sorted({int(_f["data"][k]["video_frames"].shape[0]) for k in _f["data"]})
    except Exception as _e:
        print(f"\n❌ 帧数校验无法读取文件({_e})")
        _frames = []
        canonical_ok = False
    if _frames and not set(_frames) <= {17, 25, 33}:
        canonical_ok = False
        print("\n" + "=" * 70)
        print(f"❌ 帧数分布 {_frames} 不是 canonical 的 {{17, 25, 33}} —— ")
        print("   这不是 canonical 产物(v2.8 前旧构建器→18/27/36;错构建器 steps=8→16/24/32),与论文【不可比】。")
        print("   重建: cd <包根 rswm_server_pack/> && bash server_run/build_correct.sh")
        print("   (仅调试想放行: ALLOW_NONCANONICAL=1 重跑本校验)")
        print("=" * 70)
        if os.environ.get("ALLOW_NONCANONICAL") == "1":
            canonical_ok = True
            print("[warn] ALLOW_NONCANONICAL=1 → 放行,但此数据不能用于论文实验。")
    elif _frames:
        print(f"\n✅ 帧数分布 {_frames} = canonical(steps9),与论文管道一致")

    if structure_ok and loader_ok and canonical_ok:
        print("\n" + "=" * 70)
        print("🎉 所有验证通过！数据集可用于训练")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ 验证失败，请检查数据集")
        print("=" * 70)
        sys.exit(1)
