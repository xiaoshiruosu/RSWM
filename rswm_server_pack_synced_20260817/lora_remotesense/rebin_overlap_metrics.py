#!/usr/bin/env python3
"""
Re-bin overlap ablation samples by actual overlap_ratio values and recompute metrics.

Pools all 400 samples (4 original levels x 100) from existing eval directories,
re-groups them by new overlap bins, and recomputes metrics per bin.
No re-inference needed — only reads existing .npy files.

Usage:
  # Quartile-based bins (equal sample count, recommended)
  python lora_remotesense/rebin_overlap_metrics.py \
      --base-dir <WORK>/ablation/overlap_v2 \
      --method lora \
      --output-dir /scratch/.../overlap_v2/rebin_lora \
      --strategy quartile

  # Equal-width bins (20pp each)
  python lora_remotesense/rebin_overlap_metrics.py \
      --base-dir <WORK>/ablation/overlap_v2 \
      --method lora \
      --output-dir /scratch/.../overlap_v2/rebin_lora \
      --strategy equal_width

  # Two-level split
  python lora_remotesense/rebin_overlap_metrics.py \
      --base-dir <WORK>/ablation/overlap_v2 \
      --method lora \
      --output-dir /scratch/.../overlap_v2/rebin_lora \
      --strategy two_level --split-point 0.25
"""

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

ORIGINAL_LEVELS = ["trivial", "easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Step 1: Collect all samples with overlap_ratio from benchmark.h5
# ---------------------------------------------------------------------------

def collect_samples(base_dir: Path, method: str) -> List[Dict]:
    """Read all samples from the 4 original benchmark.h5 files.

    Returns list of dicts with keys:
      original_level, key, overlap_ratio, scene_category, direction,
      direction_angle, gen_path, gt_path, meta_path
    """
    eval_subdir = f"eval_{method}"
    all_samples = []

    for level in ORIGINAL_LEVELS:
        h5_path = base_dir / level / "benchmark" / "benchmark.h5"
        eval_dir = base_dir / level / eval_subdir

        if not h5_path.exists():
            logger.warning(f"跳过 {level}: benchmark.h5 不存在 ({h5_path})")
            continue

        gen_dir = eval_dir / "generated"
        gt_dir = eval_dir / "gt"
        meta_dir = eval_dir / "meta"

        if not gen_dir.exists():
            logger.warning(f"跳过 {level}: eval 目录不存在 ({eval_dir})")
            continue

        with h5py.File(str(h5_path), "r") as f:
            total = int(f["metadata"].attrs.get("total_instances", 0))
            window_size = tuple(f["metadata"].attrs.get("window_size", (128, 222)))
            scene_size = tuple(f["metadata"].attrs.get("full_scene_size", (512, 512)))

            for idx in range(total):
                key = f"sample_{idx:05d}"
                if key not in f["data"]:
                    continue

                s = f["data"][key]
                overlap = float(s.attrs.get("overlap_ratio", -1))
                scene = str(s.attrs.get("scene_category", "general"))
                waypoints = s["waypoints"][:] if "waypoints" in s else None

                direction = None
                if waypoints is not None and len(waypoints) >= 2:
                    direction = _classify_direction(waypoints)

                gen_path = gen_dir / f"{key}.npy"
                gt_path = gt_dir / f"{key}.npy"
                meta_path = meta_dir / f"{key}.json"

                if not gen_path.exists() or not gt_path.exists():
                    logger.warning(f"跳过 {level}/{key}: .npy 文件缺失")
                    continue

                direction_angle = None
                if meta_path.exists():
                    with open(str(meta_path)) as mf:
                        meta = json.load(mf)
                    direction_angle = meta.get("direction_angle")
                    if direction is None:
                        prompt = meta.get("prompt", "").lower()
                        if "eastward" in prompt: direction = "E"
                        elif "northward" in prompt: direction = "N"
                        elif "westward" in prompt: direction = "W"
                        elif "southward" in prompt: direction = "S"

                all_samples.append({
                    "original_level": level,
                    "key": key,
                    "overlap_ratio": overlap,
                    "scene_category": scene,
                    "direction": direction,
                    "direction_angle": direction_angle,
                    "gen_path": str(gen_path),
                    "gt_path": str(gt_path),
                })

        logger.info(f"{level}: 收集 {sum(1 for s in all_samples if s['original_level'] == level)} 个样本")

    logger.info(f"共收集 {len(all_samples)} 个样本")
    return all_samples


def _classify_direction(waypoints) -> str:
    wp = np.asarray(waypoints, dtype=float)
    dy = wp[-1, 0] - wp[0, 0]
    dx = wp[-1, 1] - wp[0, 1]
    theta = np.degrees(np.arctan2(-dy, dx)) % 360
    if theta < 45 or theta >= 315:
        return "E"
    if theta < 135:
        return "N"
    if theta < 225:
        return "W"
    return "S"


# ---------------------------------------------------------------------------
# Step 2: Re-bin strategies
# ---------------------------------------------------------------------------

def rebin_quartile(samples: List[Dict]) -> Dict[str, List[Dict]]:
    """Split into 4 equal-count bins by overlap_ratio."""
    sorted_samples = sorted(samples, key=lambda s: s["overlap_ratio"])
    n = len(sorted_samples)
    q = n // 4

    bins = {}
    for i, label in enumerate(["level_I", "level_II", "level_III", "level_IV"]):
        start = i * q
        end = (i + 1) * q if i < 3 else n
        bins[label] = sorted_samples[start:end]

    for label, group in bins.items():
        overlaps = [s["overlap_ratio"] for s in group]
        logger.info(
            f"  {label}: n={len(group)}, "
            f"overlap=[{min(overlaps):.3f}, {max(overlaps):.3f}], "
            f"mean={np.mean(overlaps):.3f}"
        )

    return bins


def rebin_equal_width(samples: List[Dict], width: float = 0.20) -> Dict[str, List[Dict]]:
    """Split into equal-width bins."""
    bins = defaultdict(list)
    boundaries = np.arange(0, 0.80 + width, width)

    for s in samples:
        ov = s["overlap_ratio"]
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= ov < boundaries[i + 1]:
                label = f"ov_{boundaries[i]:.0%}-{boundaries[i+1]:.0%}".replace("%", "pct")
                bins[label].append(s)
                break
        else:
            if ov >= boundaries[-1]:
                label = f"ov_{boundaries[-2]:.0%}-{boundaries[-1]:.0%}".replace("%", "pct")
                bins[label].append(s)

    for label in sorted(bins.keys()):
        group = bins[label]
        overlaps = [s["overlap_ratio"] for s in group]
        logger.info(
            f"  {label}: n={len(group)}, "
            f"overlap=[{min(overlaps):.3f}, {max(overlaps):.3f}], "
            f"mean={np.mean(overlaps):.3f}"
        )

    return dict(sorted(bins.items()))


def rebin_two_level(samples: List[Dict], split_point: float = 0.25) -> Dict[str, List[Dict]]:
    """Split into High/Low overlap groups."""
    bins = {"low_overlap": [], "high_overlap": []}
    for s in samples:
        if s["overlap_ratio"] < split_point:
            bins["low_overlap"].append(s)
        else:
            bins["high_overlap"].append(s)

    for label, group in bins.items():
        if not group:
            continue
        overlaps = [s["overlap_ratio"] for s in group]
        logger.info(
            f"  {label}: n={len(group)}, "
            f"overlap=[{min(overlaps):.3f}, {max(overlaps):.3f}], "
            f"mean={np.mean(overlaps):.3f}"
        )

    return bins


# ---------------------------------------------------------------------------
# Step 3: Compute metrics per bin
# ---------------------------------------------------------------------------

def compute_bin_metrics(
    bin_samples: List[Dict],
    device: str = "cuda",
) -> Dict[str, Dict]:
    """Load videos and compute metrics for one bin, with scene/direction breakdowns."""
    # [v2.5 审查修复] 改用 new/metrics(带 flow_size/step);旧 lora_remotesense/
    # metrics.py 的角误差是量程外口径(GT 地板 82°),与主表不可比。
    # ⚠️ overlap 分层已按用户拍板③整节删除、不重跑;此修复仅防误跑时出错数。
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent / "new"))
    from metrics import compute_all_metrics  # noqa: E402

    gen_videos, gt_videos = [], []
    scenes, directions, target_angles = [], [], []

    for s in bin_samples:
        gen = np.load(s["gen_path"])
        gt = np.load(s["gt_path"])
        if gen.dtype != np.uint8:
            gen = np.clip(gen, 0, 255).astype(np.uint8)
        if gt.dtype != np.uint8:
            gt = np.clip(gt, 0, 255).astype(np.uint8)

        gen_videos.append(gen)
        gt_videos.append(gt)
        scenes.append(s["scene_category"])
        directions.append(s["direction"])
        target_angles.append(s["direction_angle"])

    if not gen_videos:
        return {}

    valid_angles = [a for a in target_angles if a is not None]

    results = {}

    # Overall
    logger.info(f"  计算 overall 指标 ({len(gen_videos)} 条)...")
    results["overall"] = compute_all_metrics(
        gen_videos, gt_videos,
        target_angles=valid_angles if valid_angles else None,
        device=device, flow_size=(128, 222), step=1,
    )
    results["overall"]["num_samples"] = len(gen_videos)

    # Per scene
    for scene in sorted(set(scenes)):
        idx = [i for i, sc in enumerate(scenes) if sc == scene]
        if len(idx) < 2:
            continue
        s_gen = [gen_videos[i] for i in idx]
        s_gt = [gt_videos[i] for i in idx]
        s_angles = [target_angles[i] for i in idx]
        s_valid = [a for a in s_angles if a is not None]

        logger.info(f"  计算 {scene} 指标 ({len(idx)} 条)...")
        results[scene] = compute_all_metrics(
            s_gen, s_gt,
            target_angles=s_valid if s_valid else None,
            device=device, flow_size=(128, 222), step=1,
        )
        results[scene]["num_samples"] = len(idx)

    # Per direction
    dir_set = sorted([d for d in set(directions) if d is not None])
    for d in dir_set:
        idx = [i for i, dd in enumerate(directions) if dd == d]
        if len(idx) < 2:
            continue
        d_gen = [gen_videos[i] for i in idx]
        d_gt = [gt_videos[i] for i in idx]
        d_angles = [target_angles[i] for i in idx]
        d_valid = [a for a in d_angles if a is not None]

        logger.info(f"  计算方向 {d} 指标 ({len(idx)} 条)...")
        results[f"dir_{d}"] = compute_all_metrics(
            d_gen, d_gt,
            target_angles=d_valid if d_valid else None,
            device=device, flow_size=(128, 222), step=1,
        )
        results[f"dir_{d}"]["num_samples"] = len(idx)

    return results


# ---------------------------------------------------------------------------
# Step 4: Save results
# ---------------------------------------------------------------------------

def save_bin_results(results: Dict, output_dir: Path, bin_name: str):
    """Save eval_results.json and CSV for one bin."""
    out = output_dir / bin_name
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "eval_results.json"
    with open(str(json_path), "w") as f:
        json.dump(results, f, indent=2, default=str)

    csv_path = out / "eval_results_table.csv"
    metrics_cols = ["SSIM", "PSNR", "LPIPS", "FID", "FVD",
                    "Angular_Error_Mean", "num_samples"]
    with open(str(csv_path), "w") as f:
        f.write("Scene," + ",".join(metrics_cols) + "\n")
        for scene, vals in results.items():
            row = [scene]
            for m in metrics_cols:
                v = vals.get(m, "")
                if isinstance(v, float):
                    row.append(f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}")
                else:
                    row.append(str(v))
            f.write(",".join(row) + "\n")

    logger.info(f"  保存: {json_path}")


def save_manifest(all_samples: List[Dict], bins: Dict[str, List[Dict]], output_dir: Path):
    """Save sample manifest with bin assignments."""
    key_to_bin = {}
    for bin_name, group in bins.items():
        for s in group:
            uid = f"{s['original_level']}/{s['key']}"
            key_to_bin[uid] = bin_name

    manifest = []
    for s in sorted(all_samples, key=lambda x: x["overlap_ratio"]):
        uid = f"{s['original_level']}/{s['key']}"
        manifest.append({
            "original_level": s["original_level"],
            "key": s["key"],
            "overlap_ratio": round(s["overlap_ratio"], 4),
            "scene": s["scene_category"],
            "direction": s["direction"],
            "new_bin": key_to_bin.get(uid, "unassigned"),
        })

    path = output_dir / "sample_manifest.json"
    with open(str(path), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest 保存: {path} ({len(manifest)} 条)")


def print_summary(bins: Dict[str, List[Dict]], all_results: Dict[str, Dict]):
    """Print comparison table for paper."""
    print(f"\n{'='*80}")
    print("OVERLAP RE-BIN SUMMARY")
    print(f"{'='*80}")

    header = f"{'Bin':<20s} {'N':>4s} {'Overlap':>12s} {'SSIM':>7s} {'PSNR':>7s} {'LPIPS':>7s} {'FID':>8s} {'FVD':>8s} {'Ang.Err':>8s}"
    print(header)
    print("-" * len(header))

    csv_lines = ["Bin,N,Overlap_Mean,Overlap_Min,Overlap_Max,SSIM,PSNR,LPIPS,FID,FVD,Angular_Error_Mean"]

    for bin_name in bins:
        group = bins[bin_name]
        overlaps = [s["overlap_ratio"] for s in group]
        ov_str = f"{np.mean(overlaps):.2f} [{min(overlaps):.2f},{max(overlaps):.2f}]"

        r = all_results.get(bin_name, {}).get("overall", {})
        ssim = r.get("SSIM", float("nan"))
        psnr = r.get("PSNR", float("nan"))
        lpips = r.get("LPIPS", float("nan"))
        fid = r.get("FID", float("nan"))
        fvd = r.get("FVD", float("nan"))
        ang = r.get("Angular_Error_Mean", float("nan"))

        print(
            f"{bin_name:<20s} {len(group):>4d} "
            f"{ov_str:>12s} "
            f"{ssim:>7.4f} {psnr:>7.2f} {lpips:>7.4f} {fid:>8.2f} {fvd:>8.2f} {ang:>8.2f}"
        )

        csv_lines.append(
            f"{bin_name},{len(group)},{np.mean(overlaps):.4f},"
            f"{min(overlaps):.4f},{max(overlaps):.4f},"
            f"{ssim:.4f},{psnr:.4f},{lpips:.4f},{fid:.4f},{fvd:.4f},{ang:.4f}"
        )

    print(f"{'='*80}\n")

    return "\n".join(csv_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-bin overlap ablation samples and recompute metrics",
    )
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Base dir containing trivial/easy/medium/hard subdirs")
    parser.add_argument("--method", type=str, default="lora", choices=["lora", "baseline"],
                        help="Which eval to use: eval_lora or eval_baseline")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for re-binned results")
    parser.add_argument("--strategy", type=str, default="quartile",
                        choices=["quartile", "equal_width", "two_level"],
                        help="Binning strategy")
    parser.add_argument("--split-point", type=float, default=0.25,
                        help="Split point for two_level strategy")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Step 1: Collect
    print(f"\n{'='*60}")
    print(f"Overlap Re-bin — {args.strategy} — {args.method}")
    print(f"Start: {time.strftime('%c')}")
    print(f"{'='*60}\n")

    logger.info("[Step 1/4] 收集样本...")
    all_samples = collect_samples(base_dir, args.method)
    if not all_samples:
        logger.error("没有找到样本，退出")
        sys.exit(1)

    overlaps = [s["overlap_ratio"] for s in all_samples]
    logger.info(
        f"  overlap 分布: mean={np.mean(overlaps):.3f}, "
        f"median={np.median(overlaps):.3f}, "
        f"range=[{min(overlaps):.3f}, {max(overlaps):.3f}]"
    )

    # Step 2: Re-bin
    logger.info(f"[Step 2/4] Re-bin (strategy={args.strategy})...")
    if args.strategy == "quartile":
        bins = rebin_quartile(all_samples)
    elif args.strategy == "equal_width":
        bins = rebin_equal_width(all_samples)
    elif args.strategy == "two_level":
        bins = rebin_two_level(all_samples, args.split_point)

    # Save manifest
    save_manifest(all_samples, bins, output_dir)

    # Step 3: Compute metrics per bin
    logger.info("[Step 3/4] 计算指标...")
    all_results = {}
    for bin_name, group in bins.items():
        if not group:
            logger.warning(f"  跳过空 bin: {bin_name}")
            continue
        logger.info(f"  === {bin_name} (n={len(group)}) ===")
        results = compute_bin_metrics(group, device=args.device)
        all_results[bin_name] = results
        save_bin_results(results, output_dir, bin_name)

    # Step 4: Summary
    logger.info("[Step 4/4] 输出汇总...")
    csv_text = print_summary(bins, all_results)

    summary_path = output_dir / "summary.csv"
    with open(str(summary_path), "w") as f:
        f.write(csv_text)
    logger.info(f"Summary CSV: {summary_path}")

    elapsed = time.time() - t_start
    print(f"\n完成! 耗时 {elapsed / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
