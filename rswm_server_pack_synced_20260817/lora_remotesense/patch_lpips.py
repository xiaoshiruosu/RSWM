#!/usr/bin/env python3
"""
补算 LPIPS 并 patch 已有 eval_results.json / CSV。

用法:
  python lora_remotesense/patch_lpips.py \
    --eval-dirs /path/to/eval_lora /path/to/eval_baseline \
    --device cuda

前置: 登录节点先运行一次缓存权重:
  python -c "import lpips; lpips.LPIPS(net='alex'); print('OK')"
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_lpips_model(device):
    import lpips
    return lpips.LPIPS(net="alex").to(device).eval()


def compute_video_lpips(model, gen, gt, device):
    T = min(len(gen), len(gt))
    scores = []
    for t in range(T):
        g = torch.from_numpy(gen[t]).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
        r = torch.from_numpy(gt[t]).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
        h = min(g.shape[2], r.shape[2])
        w = min(g.shape[3], r.shape[3])
        g = g[:, :, :h, :w].to(device)
        r = r[:, :, :h, :w].to(device)
        with torch.no_grad():
            scores.append(model(g, r).item())
    return float(np.mean(scores))


def patch_one_dir(eval_dir: Path, model, device: str):
    gen_dir = eval_dir / "generated"
    gt_dir = eval_dir / "gt"
    meta_dir = eval_dir / "meta"
    json_path = eval_dir / "eval_results.json"

    if not gen_dir.exists():
        print(f"SKIP {eval_dir}: no generated/ directory")
        return

    gen_files = sorted(gen_dir.glob("*.npy"))
    print(f"\n{'='*60}")
    print(f"Patching: {eval_dir}")
    print(f"  {len(gen_files)} samples")

    per_sample = []
    for i, gf in enumerate(gen_files):
        key = gf.stem
        gt_file = gt_dir / f"{key}.npy"
        meta_file = meta_dir / f"{key}.json"
        if not gt_file.exists():
            continue

        gen = np.load(str(gf))
        gt = np.load(str(gt_file))
        if gen.dtype != np.uint8:
            gen = np.clip(gen, 0, 255).astype(np.uint8)
        if gt.dtype != np.uint8:
            gt = np.clip(gt, 0, 255).astype(np.uint8)

        lp = compute_video_lpips(model, gen, gt, device)

        scene = "general"
        direction = None
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            scene = meta.get("scene", "general")
            prompt = meta.get("prompt", "").lower()
            for d, kw in [("E", "eastward"), ("N", "northward"),
                          ("W", "westward"), ("S", "southward")]:
                if kw in prompt:
                    direction = d
                    break

        per_sample.append({"key": key, "lpips": lp, "scene": scene, "direction": direction})
        if (i + 1) % 20 == 0 or i + 1 == len(gen_files):
            print(f"  [{i+1}/{len(gen_files)}] LPIPS={lp:.4f}")

    if not per_sample:
        print("  No valid samples")
        return

    overall_lpips = float(np.mean([s["lpips"] for s in per_sample]))
    print(f"  Overall LPIPS = {overall_lpips:.4f}")

    group_lpips = defaultdict(list)
    group_lpips["overall"] = [s["lpips"] for s in per_sample]
    for s in per_sample:
        group_lpips[s["scene"]].append(s["lpips"])
        if s["direction"]:
            group_lpips[f"dir_{s['direction']}"].append(s["lpips"])
            group_lpips[f"dir_{s['direction']}_{s['scene']}"].append(s["lpips"])

    if json_path.exists():
        with open(json_path) as f:
            results = json.load(f)
    else:
        results = {}

    for group_name, values in group_lpips.items():
        if group_name in results:
            results[group_name]["LPIPS"] = float(np.mean(values))
        else:
            results[group_name] = {"LPIPS": float(np.mean(values))}

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON patched: {json_path}")

    csv_path = eval_dir / "eval_results_table.csv"
    metrics = ["SSIM", "PSNR", "LPIPS", "FID", "FVD",
               "Angular_Error_Mean", "num_samples"]
    with open(csv_path, "w") as f:
        f.write("Scene," + ",".join(metrics) + "\n")
        for scene, vals in results.items():
            row = [scene]
            for m in metrics:
                v = vals.get(m, "")
                if isinstance(v, float):
                    row.append(f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}")
                else:
                    row.append(str(v))
            f.write(",".join(row) + "\n")
    print(f"  CSV patched: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dirs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("Loading LPIPS model...")
    model = load_lpips_model(args.device)
    print("LPIPS model ready")

    for d in args.eval_dirs:
        patch_one_dir(Path(d), model, args.device)

    print("\nDone!")


if __name__ == "__main__":
    main()
