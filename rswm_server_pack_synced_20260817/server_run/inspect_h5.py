#!/usr/bin/env python
"""inspect_h5.py — 从 LingBot HDF5 正确抽帧存 jpg + 打印诊断，排查"数据模糊"。

区分三种情况：
  1) 转换 bug（float 没 ×255 / 通道错）→ 看 dtype/min/max
  2) 帧本身无结构（纯色/退化）→ 看 std
  3) 128×222 窗口上采样到 480×832 的固有软化（正常，与论文一致）

用法:
  python server_run/inspect_h5.py <train.h5> [--key sample_00000] [--out ./inspect_h5] [--bgr]
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def to_uint8_rgb(arr, bgr=False):
    a = np.asarray(arr)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    if a.dtype != np.uint8:
        lo, hi = float(a.min()), float(a.max())
        if lo >= -1.01 and hi <= 1.01 and lo < -0.01:   # [-1,1]
            a = (a + 1.0) * 127.5
        elif hi <= 1.01:                                 # [0,1]
            a = a * 255.0
        # 否则假定已是 [0,255] 区间的 float
        a = np.clip(a, 0, 255).astype(np.uint8)
    a = a[..., :3]
    if bgr:
        a = a[..., ::-1]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("h5")
    ap.add_argument("--key", default=None, help="默认取第一个样本")
    ap.add_argument("--out", default="./inspect_h5")
    ap.add_argument("--bgr", action="store_true", help="若颜色发蓝把 B/R 换回来（cv2 存的是 BGR）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        keys = sorted(f["data"].keys())
        if not keys:
            sys.exit("FATAL: h5 里没有 data/ 样本")
        key = args.key or keys[0]
        g = f["data"][key]
        print(f"[h5] {args.h5}")
        print(f"     样本数={len(keys)}  查看 key={key}")

        for name in ("image", "video_frames", "poses"):
            if name in g:
                d = g[name]
                arr = d[:] if name != "poses" else d[:]
                print(f"  {name:13s} shape={tuple(arr.shape)} dtype={arr.dtype} "
                      f"min={float(arr.min()):.3f} max={float(arr.max()):.3f} "
                      f"mean={float(arr.mean()):.3f} std={float(arr.std()):.3f}")

        vf = g["video_frames"][:]                     # (T,H,W,C)
        if vf.ndim != 4:
            print(f"  [warn] video_frames 维度异常: {vf.shape}")
        T = vf.shape[0]
        H, W = vf.shape[1], vf.shape[2]
        idxs = sorted(set([0, T // 2, T - 1]))
        for i in idxs:
            Image.fromarray(to_uint8_rgb(vf[i], args.bgr)).save(
                out / f"{key}_frame{i:02d}.jpg", quality=95)

        cols = min(T, 6)
        rows = (T + cols - 1) // cols
        canvas = np.zeros((rows * H, cols * W, 3), np.uint8)
        for i in range(T):
            r, c = divmod(i, cols)
            canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = to_uint8_rgb(vf[i], args.bgr)
        Image.fromarray(canvas).save(out / f"{key}_montage.jpg", quality=90)

        std = float(vf.astype(np.float32).std())
        print(f"\n[out] {out}/{key}_frame*.jpg + {key}_montage.jpg  (帧尺寸 {H}x{W}, {T} 帧)")
        print("诊断：")
        if vf.dtype != np.uint8:
            print("  ⚠ dtype 非 uint8 → 你之前转 jpg 若没做归一化(×255)，就会一片灰糊（假模糊）")
        if std < 10:
            print(f"  ⚠ std={std:.1f} 很小 → 帧接近纯色/无结构，疑似转换 bug 或窗口退化")
        else:
            print(f"  ✓ std={std:.1f} 有内容。若图有结构但发软：128×222 窗口→480×832 上采样(~3.75×)"
                  "的固有软化，与论文数据一致，正常。")
        print("  颜色发蓝/发红 → 加 --bgr 重跑（cv2 存的是 BGR）。")


if __name__ == "__main__":
    main()
