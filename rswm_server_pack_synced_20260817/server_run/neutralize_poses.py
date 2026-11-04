#!/usr/bin/env python
"""neutralize_poses.py — 复制 LingBot HDF5 并把所有 poses 置为 identity。

用途:当前 build_dataset_from_benchmark.py(BUG-4 修复版)一律注入 waypoint
Plücker 方向位姿;而论文主实验(语义通道)的 regime 是 prompt 文本方向 +
identity 位姿。本脚本从几何版数据复制出语义版,保证两版样本/prompt 完全一致,
位姿是唯一变量 —— 语义 vs 几何通道的干净单变量对比。

用法: python neutralize_poses.py SRC.h5 DST.h5
"""
import shutil
import sys

import h5py
import numpy as np


def main(src: str, dst: str) -> None:
    print(f"[copy] {src} -> {dst} (大文件,~分钟级)")
    shutil.copyfile(src, dst)
    n = 0
    with h5py.File(dst, "r+") as f:
        for key in f["data"]:
            g = f["data"][key]
            poses = g["poses"]
            t = poses.shape[0]
            eye = np.tile(np.eye(4, dtype=poses.dtype), (t, 1, 1))
            poses[...] = eye
            n += 1
        f["metadata"].attrs["pose_injection"] = "identity_neutralized"
    print(f"[done] {n} 个样本位姿已置 identity: {dst}")
    # 抽查
    with h5py.File(dst, "r") as f:
        k = sorted(f["data"].keys())[0]
        p = f["data"][k]["poses"][:]
        assert np.allclose(p[0], np.eye(4)), "位姿中和校验失败"
    print("[verify] 抽查通过")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
