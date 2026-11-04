#!/usr/bin/env python3
"""
对齐样本后重算 metrics

用法:
  python eval_aligned.py \
    --lora_dir eval_output_lora \
    --base_dir eval_output_baseline \
    --output_dir eval_output_aligned \
    --device cuda

功能:
  1. 找出 lora_dir 和 base_dir 中都存在的样本 (交集)
  2. 将交集样本软链接/复制到 output_dir 下的 lora/ 和 base/ 子目录
  3. 分别计算 metrics 并保存结果
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if (HERE / "wan").is_dir():
    PROJECT_ROOT = HERE
else:
    PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluate import compute_metrics, save_results

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)


def find_common_keys(dir_a: Path, dir_b: Path) -> list:
    """找出两个目录下 generated/ 中都存在的样本 key"""
    keys_a = {f.stem for f in (dir_a / "generated").glob("*.npy")}
    keys_b = {f.stem for f in (dir_b / "generated").glob("*.npy")}
    common = sorted(keys_a & keys_b)
    logger.info(f"LoRA 样本数: {len(keys_a)}, Base 样本数: {len(keys_b)}, 交集: {len(common)}")
    only_lora = sorted(keys_a - keys_b)
    only_base = sorted(keys_b - keys_a)
    if only_lora:
        logger.info(f"仅 LoRA 有: {only_lora}")
    if only_base:
        logger.info(f"仅 Base 有: {only_base}")
    return common


def copy_aligned_samples(src_dir: Path, dst_dir: Path, keys: list):
    """将交集样本从 src 复制到 dst"""
    for sub in ["generated", "gt", "meta"]:
        # [v2.7.2 ④/P2 修复] 复制前清空:同一输出目录重跑更小/不同的交集时,
        # 旧样本残留会静默混进指标计算(对齐比较的样本集悄悄变大)。
        sub_dir = dst_dir / sub
        if sub_dir.exists():
            shutil.rmtree(sub_dir)
        sub_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for key in keys:
        for sub, ext in [("generated", ".npy"), ("gt", ".npy"), ("meta", ".json")]:
            src = src_dir / sub / f"{key}{ext}"
            dst = dst_dir / sub / f"{key}{ext}"
            if src.exists():
                shutil.copy2(str(src), str(dst))
                copied += 1
    logger.info(f"复制 {copied} 个文件到 {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="对齐样本后重算 metrics")
    parser.add_argument("--lora_dir", type=str, required=True, help="LoRA 评估输出目录")
    parser.add_argument("--base_dir", type=str, required=True, help="Base 评估输出目录")
    parser.add_argument("--output_dir", type=str, default="eval_output_aligned")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    base_dir = Path(args.base_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 找交集
    common_keys = find_common_keys(lora_dir, base_dir)
    if not common_keys:
        logger.error("没有共同样本，无法对齐")
        sys.exit(1)   # [v2.7.1 F1/P0] 无共同样本=失败,不许 exit 0

    # 保存交集 key 列表
    keys_file = out_dir / "aligned_keys.json"
    with open(str(keys_file), "w") as f:
        json.dump({"common_keys": common_keys, "num_samples": len(common_keys)}, f, indent=2)
    logger.info(f"交集 key 已保存: {keys_file}")

    # Step 2: 复制交集样本
    lora_aligned = out_dir / "lora"
    base_aligned = out_dir / "base"
    copy_aligned_samples(lora_dir, lora_aligned, common_keys)
    copy_aligned_samples(base_dir, base_aligned, common_keys)

    # Step 3: 分别计算 metrics
    logger.info("=" * 50)
    logger.info("计算 LoRA (aligned) metrics...")
    lora_results = compute_metrics(str(lora_aligned), device=args.device)
    save_results(lora_results, str(lora_aligned))

    logger.info("=" * 50)
    logger.info("计算 Base (aligned) metrics...")
    base_results = compute_metrics(str(base_aligned), device=args.device)
    save_results(base_results, str(base_aligned))

    # Step 4: 生成对比表
    comparison = {}
    for scene in lora_results:
        if scene not in base_results:
            continue
        comparison[scene] = {}
        for metric in ["SSIM", "PSNR", "LPIPS", "FID", "FVD",
                        "Angular_Error_Mean", "num_samples"]:
            lv = lora_results[scene].get(metric)
            bv = base_results[scene].get(metric)
            comparison[scene][metric] = {
                "lora": lv,
                "base": bv,
                "diff": round(lv - bv, 4) if isinstance(lv, (int, float)) and isinstance(bv, (int, float)) else None,
            }

    comp_file = out_dir / "comparison.json"
    with open(str(comp_file), "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    logger.info(f"对比结果已保存: {comp_file}")

    # 打印对比表
    print(f"\n{'='*80}")
    print("ALIGNED COMPARISON (LoRA vs Base)")
    print(f"{'='*80}")
    for scene, metrics in comparison.items():
        n = metrics.get("num_samples", {}).get("lora", "?")
        print(f"\n--- {scene} (n={n}) ---")
        print(f"  {'Metric':25s} {'Base':>10s} {'LoRA':>10s} {'Diff':>10s}")
        for m, vals in metrics.items():
            if m == "num_samples":
                continue
            bv = f"{vals['base']:.4f}" if isinstance(vals['base'], float) else str(vals['base'])
            lv = f"{vals['lora']:.4f}" if isinstance(vals['lora'], float) else str(vals['lora'])
            dv = f"{vals['diff']:+.4f}" if vals['diff'] is not None else "N/A"
            print(f"  {m:25s} {bv:>10s} {lv:>10s} {dv:>10s}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
