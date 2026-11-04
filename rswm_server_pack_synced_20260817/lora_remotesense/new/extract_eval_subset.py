#!/usr/bin/env python3
"""
从 train_hd_500.h5 提取 N 个 eval 样本, 生成最小 h5 + split.json
用于上传 AutoDL 运行 MVSplat360 baseline

用法 (在 Gadi 上运行):
  python extract_eval_subset.py \
    --src ./data/remotesense/train.h5 \
    --split ./outputs/train_val_split.json \
    --dst eval_subset_20.h5 \
    --n 20
"""

import argparse
import json
import h5py
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="源 h5 文件")
    parser.add_argument("--split", required=True, help="train_val_split.json")
    parser.add_argument("--dst", default="eval_subset_20.h5", help="输出 h5")
    parser.add_argument("--n", type=int, default=20, help="提取样本数")
    args = parser.parse_args()

    with open(args.split) as f:
        split = json.load(f)
    val_indices = split["val_indices"][:args.n]

    with h5py.File(args.src, "r") as src:
        keys = sorted(src["data"].keys())
        total_keys = len(keys)

        with h5py.File(args.dst, "w") as dst:
            meta = dst.create_group("metadata")
            for attr_name in src["metadata"].attrs:
                meta.attrs[attr_name] = src["metadata"].attrs[attr_name]
            meta.attrs["total_instances"] = len(val_indices)

            data = dst.create_group("data")
            written = 0
            for idx in val_indices:
                if idx >= total_keys:
                    continue
                key = keys[idx]
                src.copy(src["data"][key], data, name=key)
                written += 1
                if written % 5 == 0:
                    print(f"  {written}/{len(val_indices)}")

    new_split = {
        "train_indices": [],
        "val_indices": list(range(len(val_indices))),
        "seed": 42,
    }
    split_path = Path(args.dst).with_suffix(".json")
    with open(split_path, "w") as f:
        json.dump(new_split, f)

    import os
    size_mb = os.path.getsize(args.dst) / 1024 / 1024
    print(f"\n完成: {written} 样本 → {args.dst} ({size_mb:.0f} MB)")
    print(f"Split: {split_path}")
    print(f"\n上传到 AutoDL 的文件:")
    print(f"  {args.dst}")
    print(f"  {split_path}")


if __name__ == "__main__":
    main()
