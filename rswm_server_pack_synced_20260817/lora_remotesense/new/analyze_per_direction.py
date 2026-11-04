#!/usr/bin/env python3
"""
Per-direction (E/N/W/S) 指标分析 — 本地 CPU 版本
计算 SSIM / PSNR，按方向和方向×场景交叉分组

用法:
  python analyze_per_direction.py \
    --eval_dir eval_output_165035647.gadi-pbs/full \
    --output_dir eval_per_direction_results
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from skimage.metrics import structural_similarity


def compute_ssim(img1, img2):
    if img1.dtype != np.uint8:
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2, 0, 255).astype(np.uint8)
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return structural_similarity(img1[:h, :w], img2[:h, :w], channel_axis=2, data_range=255)


def compute_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    mse = np.mean((img1[:h, :w] - img2[:h, :w]) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def video_ssim(gen, gt):
    T = min(len(gen), len(gt))
    return float(np.mean([compute_ssim(gen[t], gt[t]) for t in range(T)]))


def video_psnr(gen, gt):
    T = min(len(gen), len(gt))
    return float(np.mean([compute_psnr(gen[t], gt[t]) for t in range(T)]))


def parse_direction(prompt):
    pl = prompt.lower()
    if "eastward" in pl: return "E"
    if "northward" in pl: return "N"
    if "westward" in pl: return "W"
    if "southward" in pl: return "S"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_per_direction_results")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_dir = eval_dir / "generated"
    gt_dir = eval_dir / "gt"
    meta_dir = eval_dir / "meta"

    # 收集所有样本并按方向分组
    by_direction = defaultdict(list)
    by_dir_scene = defaultdict(list)

    gen_files = sorted(gen_dir.glob("*.npy"))
    print(f"找到 {len(gen_files)} 个样本")

    for i, gf in enumerate(gen_files):
        key = gf.stem
        gt_file = gt_dir / f"{key}.npy"
        meta_file = meta_dir / f"{key}.json"

        if not gt_file.exists() or not meta_file.exists():
            continue

        with open(meta_file) as f:
            meta = json.load(f)

        direction = parse_direction(meta.get("prompt", ""))
        scene = meta.get("scene", "general")

        if direction is None:
            continue

        print(f"  [{i+1}/{len(gen_files)}] {key}: {direction} / {scene} ...", end="", flush=True)

        gen = np.load(str(gf))
        gt = np.load(str(gt_file))
        if gen.dtype != np.uint8:
            gen = np.clip(gen, 0, 255).astype(np.uint8)
        if gt.dtype != np.uint8:
            gt = np.clip(gt, 0, 255).astype(np.uint8)

        ssim = video_ssim(gen, gt)
        psnr = video_psnr(gen, gt)

        record = {
            "key": key, "direction": direction, "scene": scene,
            "ssim": ssim, "psnr": psnr,
            "num_frames_gen": gen.shape[0], "num_frames_gt": gt.shape[0],
        }
        by_direction[direction].append(record)
        by_dir_scene[(direction, scene)].append(record)
        print(f" SSIM={ssim:.4f} PSNR={psnr:.2f}")

    # 汇总 per-direction
    print(f"\n{'='*70}")
    print("Per-Direction Results")
    print(f"{'='*70}")

    results = {"per_direction": {}, "per_direction_scene": {}}

    for d in ["E", "N", "W", "S"]:
        records = by_direction.get(d, [])
        if not records:
            continue
        ssims = [r["ssim"] for r in records]
        psnrs = [r["psnr"] for r in records]
        results["per_direction"][d] = {
            "num_samples": len(records),
            "SSIM_mean": float(np.mean(ssims)),
            "SSIM_std": float(np.std(ssims)),
            "PSNR_mean": float(np.mean(psnrs)),
            "PSNR_std": float(np.std(psnrs)),
        }
        r = results["per_direction"][d]
        print(f"\n{d} (n={r['num_samples']}):")
        print(f"  SSIM: {r['SSIM_mean']:.4f} +/- {r['SSIM_std']:.4f}")
        print(f"  PSNR: {r['PSNR_mean']:.2f} +/- {r['PSNR_std']:.2f}")

    # 汇总 direction x scene
    print(f"\n{'='*70}")
    print("Direction x Scene Cross Analysis")
    print(f"{'='*70}")

    # 表头
    scenes = sorted({r["scene"] for recs in by_direction.values() for r in recs})
    header = f"{'Dir':>5s}"
    for sc in scenes:
        header += f" | {sc:>18s}"
    print(header)
    print("-" * len(header))

    for d in ["E", "N", "W", "S"]:
        row_ssim = f"{d:>5s}"
        row_psnr = f"{'':>5s}"
        for sc in scenes:
            records = by_dir_scene.get((d, sc), [])
            if records:
                s_mean = np.mean([r["ssim"] for r in records])
                p_mean = np.mean([r["psnr"] for r in records])
                row_ssim += f" | SSIM {s_mean:.3f} (n={len(records):>2d})"
                row_psnr += f" | PSNR {p_mean:.2f}      "
                results["per_direction_scene"][f"{d}_{sc}"] = {
                    "num_samples": len(records),
                    "SSIM_mean": float(s_mean),
                    "PSNR_mean": float(p_mean),
                }
            else:
                row_ssim += f" | {'N/A':>18s}"
                row_psnr += f" | {'':>18s}"
        print(row_ssim)
        print(row_psnr)

    # 保存
    json_path = out_dir / "per_direction_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {json_path}")

    # CSV
    csv_path = out_dir / "per_direction_table.csv"
    with open(csv_path, "w") as f:
        f.write("Direction,num_samples,SSIM_mean,SSIM_std,PSNR_mean,PSNR_std\n")
        for d in ["E", "N", "W", "S"]:
            r = results["per_direction"].get(d, {})
            if r:
                f.write(f"{d},{r['num_samples']},{r['SSIM_mean']:.4f},{r['SSIM_std']:.4f},{r['PSNR_mean']:.4f},{r['PSNR_std']:.4f}\n")

        f.write("\nDirection,Scene,num_samples,SSIM_mean,PSNR_mean\n")
        for d in ["E", "N", "W", "S"]:
            for sc in scenes:
                r = results["per_direction_scene"].get(f"{d}_{sc}", {})
                if r:
                    f.write(f"{d},{sc},{r['num_samples']},{r['SSIM_mean']:.4f},{r['PSNR_mean']:.4f}\n")
    print(f"CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
