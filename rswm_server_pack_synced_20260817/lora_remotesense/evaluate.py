#!/usr/bin/env python3
"""
RSWM 评估 Pipeline

用法:
  # 快速测试 (5 条)
  python lora_remotesense/evaluate.py \
    --checkpoint_dir <CKPT_DIR>/lingbot-world-base-cam \
    --lora_path <WORK>/train_.../best_checkpoint \
    --data_path <WORK>/data/train_semantic.h5 \
    --split_file <WORK>/train_.../train_val_split.json \
    --output_dir eval_output \
    --max_samples 5

  # 全量评估 (100 条验证集)
  python lora_remotesense/evaluate.py \
    --checkpoint_dir ... --lora_path ... --data_path ... --split_file ... \
    --output_dir eval_output

  # 仅计算指标 (已有推理结果)
  python lora_remotesense/evaluate.py --metrics_only --output_dir eval_output
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import h5py
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据加载
# ============================================================================

def load_val_indices(split_file: str) -> List[int]:
    """从 train_val_split.json 加载验证集 indices"""
    with open(split_file, "r") as f:
        split = json.load(f)
    return split["val_indices"]


def parse_scene_from_prompt(prompt: str) -> str:
    """从 prompt 中解析场景类别"""
    prompt_lower = prompt.lower()
    if "flood" in prompt_lower:
        return "flood"
    elif "urban" in prompt_lower:
        return "urban"
    elif "rural" in prompt_lower:
        return "rural"
    return "general"


def parse_direction_angle_from_prompt(prompt: str) -> Optional[float]:
    """从 prompt 中解析方向角度 (度)"""
    prompt_lower = prompt.lower()
    if "eastward" in prompt_lower:
        return 0.0
    elif "northward" in prompt_lower:
        return 90.0
    elif "westward" in prompt_lower:
        return 180.0
    elif "southward" in prompt_lower:
        return 270.0
    return None


def load_val_samples(data_path: str, val_indices: List[int], max_samples: Optional[int] = None):
    """加载验证集样本"""
    samples = []
    with h5py.File(data_path, "r") as f:
        keys = sorted(f["data"].keys())
        for idx in val_indices:
            if max_samples and len(samples) >= max_samples:
                break
            if idx >= len(keys):
                continue
            key = keys[idx]
            s = f["data"][key]
            sample = {
                "key": key,
                "index": idx,
                "image": s["image"][:],              # (H, W, 3)
                "video_frames": s["video_frames"][:], # (T, H, W, 3)
                "poses": s["poses"][:],               # (T, 4, 4)
                "intrinsics": s["intrinsics"][:],     # (T, 4)
                "prompt": s.attrs.get("prompt", "A satellite video of a remote sensing scene."),
                "num_frames": int(s.attrs.get("num_frames", s["video_frames"].shape[0])),
            }
            sample["scene"] = parse_scene_from_prompt(sample["prompt"])
            sample["direction_angle"] = parse_direction_angle_from_prompt(sample["prompt"])
            samples.append(sample)
    logger.info(f"加载 {len(samples)} 条验证样本")
    return samples


# ============================================================================
# 推理
# ============================================================================

def run_inference_batch(
    samples: List[Dict],
    checkpoint_dir: str,
    lora_path: str,
    output_dir: str,
    device: str = "cuda",
    frame_num: int = 33,
    size: str = "480*832",
    t5_cpu: bool = True,
    framewise: bool = True,
):
    """批量推理，保存生成视频和 GT"""
    from lora_remotesense.inference import load_lora_model, generate_video
    from wan.configs import WAN_CONFIGS
    from PIL import Image
    import tempfile

    gen_dir = Path(output_dir) / "generated"
    gt_dir = Path(output_dir) / "gt"
    meta_dir = Path(output_dir) / "meta"
    gen_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device)
    config = WAN_CONFIGS["i2v-A14B"]

    # 加载模型 (只加载一次)
    logger.info("加载模型...")
    text_encoder, vae, low_model, high_model = load_lora_model(
        checkpoint_dir=checkpoint_dir,
        lora_path=lora_path,
        device=device_obj,
        config=config,
        t5_cpu=t5_cpu,
    )
    logger.info("模型加载完成")

    skipped = 0
    for i, sample in enumerate(samples):
        # 断点续推: 跳过已完成的样本
        existing_gen = gen_dir / f"{sample['key']}.npy"
        if existing_gen.exists():
            skipped += 1
            logger.info(f"[{i+1}/{len(samples)}] 跳过 {sample['key']} (已存在)")
            continue

        logger.info(f"[{i+1}/{len(samples)}] 推理 {sample['key']}...")
        t0 = time.time()

        # 保存首帧为临时图片
        img_arr = sample["image"]
        if img_arr.dtype != np.uint8:
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_arr)

        # 保存 poses/intrinsics 到临时目录
        action_dir = tempfile.mkdtemp()
        np.save(os.path.join(action_dir, "poses.npy"), sample["poses"])
        np.save(os.path.join(action_dir, "intrinsics.npy"), sample["intrinsics"])

        # 推理
        try:
            video_tensor = generate_video(
                text_encoder=text_encoder,
                vae=vae,
                low_noise_model=low_model,
                high_noise_model=high_model,
                config=config,
                prompt=sample["prompt"],
                image=pil_img,
                device=device_obj,
                frame_num=min(frame_num, sample["num_frames"]),
                size=size,
                action_path=action_dir,
                t5_cpu=t5_cpu,
                framewise=framewise,
            )
            # video_tensor: (C, T, H, W) float, [-1, 1]
            # 转为 (T, H, W, C) uint8
            gen_video = video_tensor.clamp(-1, 1).permute(1, 2, 3, 0)
            gen_video = ((gen_video + 1) * 127.5).byte().cpu().numpy()
        except Exception as e:
            logger.error(f"推理失败 {sample['key']}: {e}")
            continue

        # 保存
        np.save(str(gen_dir / f"{sample['key']}.npy"), gen_video)

        gt_video = sample["video_frames"]
        if gt_video.dtype != np.uint8:
            gt_video = np.clip(gt_video, 0, 255).astype(np.uint8)
        np.save(str(gt_dir / f"{sample['key']}.npy"), gt_video)

        # 元数据
        meta = {
            "key": sample["key"],
            "prompt": sample["prompt"],
            "scene": sample["scene"],
            "direction_angle": sample["direction_angle"],
            "num_frames_gt": sample["num_frames"],
            "num_frames_gen": gen_video.shape[0],
        }
        with open(str(meta_dir / f"{sample['key']}.json"), "w") as f:
            json.dump(meta, f, indent=2)

        elapsed = time.time() - t0
        logger.info(f"  完成, {elapsed:.1f}s, gen shape={gen_video.shape}")

        # 清理显存
        torch.cuda.empty_cache()

    if skipped > 0:
        logger.info(f"跳过 {skipped} 个已完成样本, 新推理 {len(samples) - skipped} 个")
    logger.info("推理全部完成")


# ============================================================================
# 指标计算
# ============================================================================

def compute_metrics(output_dir: str, device: str = "cuda"):
    """从保存的 npy 文件计算所有指标"""
    from lora_remotesense.metrics import compute_all_metrics

    gen_dir = Path(output_dir) / "generated"
    gt_dir = Path(output_dir) / "gt"
    meta_dir = Path(output_dir) / "meta"

    # 收集所有样本
    gen_files = sorted(gen_dir.glob("*.npy"))
    logger.info(f"找到 {len(gen_files)} 个生成结果")

    gen_videos, gt_videos, target_angles, scenes, directions = [], [], [], [], []

    for gf in gen_files:
        key = gf.stem
        gt_file = gt_dir / f"{key}.npy"
        meta_file = meta_dir / f"{key}.json"

        if not gt_file.exists():
            logger.warning(f"缺少 GT: {key}")
            continue

        gen = np.load(str(gf))
        gt = np.load(str(gt_file))

        if gen.dtype != np.uint8:
            gen = np.clip(gen, 0, 255).astype(np.uint8)
        if gt.dtype != np.uint8:
            gt = np.clip(gt, 0, 255).astype(np.uint8)

        gen_videos.append(gen)
        gt_videos.append(gt)

        # 元数据
        if meta_file.exists():
            with open(str(meta_file)) as f:
                meta = json.load(f)
            scenes.append(meta.get("scene", "general"))
            angle = meta.get("direction_angle")
            target_angles.append(angle)
            # 解析方向
            prompt = meta.get("prompt", "").lower()
            if "eastward" in prompt: directions.append("E")
            elif "northward" in prompt: directions.append("N")
            elif "westward" in prompt: directions.append("W")
            elif "southward" in prompt: directions.append("S")
            else: directions.append(None)
        else:
            scenes.append("general")
            target_angles.append(None)
            directions.append(None)

    if not gen_videos:
        logger.error("没有可评估的样本")
        return {}

    # 全局指标
    logger.info("计算全局指标...")
    valid_angles = [a for a in target_angles if a is not None]
    valid_angle_videos = [v for v, a in zip(gen_videos, target_angles) if a is not None]

    results = {"overall": compute_all_metrics(
        gen_videos, gt_videos,
        target_angles=valid_angles if valid_angles else None,
        device=device,
    )}
    results["overall"]["num_samples"] = len(gen_videos)

    # 按场景分项
    scene_set = sorted(set(scenes))
    for scene in scene_set:
        idx = [i for i, s in enumerate(scenes) if s == scene]
        if len(idx) < 2:
            continue
        s_gen = [gen_videos[i] for i in idx]
        s_gt = [gt_videos[i] for i in idx]
        s_angles = [target_angles[i] for i in idx]
        s_valid_angles = [a for a in s_angles if a is not None]
        s_valid_vids = [v for v, a in zip(s_gen, s_angles) if a is not None]

        logger.info(f"计算 {scene} 场景指标 ({len(idx)} 条)...")
        results[scene] = compute_all_metrics(
            s_gen, s_gt,
            target_angles=s_valid_angles if s_valid_angles else None,
            device=device,
        )
        results[scene]["num_samples"] = len(idx)

    # 按方向分项
    dir_set = sorted([d for d in set(directions) if d is not None])
    for d in dir_set:
        idx = [i for i, dd in enumerate(directions) if dd == d]
        if len(idx) < 2:
            continue
        d_gen = [gen_videos[i] for i in idx]
        d_gt = [gt_videos[i] for i in idx]
        d_angles = [target_angles[i] for i in idx]
        d_valid_angles = [a for a in d_angles if a is not None]

        logger.info(f"计算方向 {d} 指标 ({len(idx)} 条)...")
        results[f"dir_{d}"] = compute_all_metrics(
            d_gen, d_gt,
            target_angles=d_valid_angles if d_valid_angles else None,
            device=device,
        )
        results[f"dir_{d}"]["num_samples"] = len(idx)

    # 方向 x 场景交叉分项
    for d in dir_set:
        for scene in scene_set:
            idx = [i for i in range(len(gen_videos))
                   if directions[i] == d and scenes[i] == scene]
            if len(idx) < 2:
                continue
            x_gen = [gen_videos[i] for i in idx]
            x_gt = [gt_videos[i] for i in idx]
            x_angles = [target_angles[i] for i in idx]
            x_valid_angles = [a for a in x_angles if a is not None]

            logger.info(f"计算 {d}x{scene} 指标 ({len(idx)} 条)...")
            results[f"dir_{d}_{scene}"] = compute_all_metrics(
                x_gen, x_gt,
                target_angles=x_valid_angles if x_valid_angles else None,
                device=device,
            )
            results[f"dir_{d}_{scene}"]["num_samples"] = len(idx)

    return results


def save_results(results: Dict, output_dir: str):
    """保存结果为 JSON 和 CSV"""
    out = Path(output_dir)

    # JSON
    json_path = out / "eval_results.json"
    with open(str(json_path), "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"结果已保存: {json_path}")

    # CSV (方便粘贴到论文)
    csv_path = out / "eval_results_table.csv"
    with open(str(csv_path), "w") as f:
        # 表头
        metrics = ["SSIM", "PSNR", "LPIPS", "FID", "FVD",
                    "Angular_Error_Mean", "num_samples"]
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
    logger.info(f"CSV 已保存: {csv_path}")

    # 终端打印
    print(f"\n{'='*70}")
    print("RSWM 评估结果")
    print(f"{'='*70}")
    for scene, vals in results.items():
        print(f"\n--- {scene} (n={vals.get('num_samples', '?')}) ---")
        for k, v in vals.items():
            if k == "num_samples":
                continue
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.4f}")
            else:
                print(f"  {k:25s}: {v}")
    print(f"{'='*70}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RSWM 评估 Pipeline")

    # 推理参数
    parser.add_argument("--checkpoint_dir", type=str, help="基础模型目录")
    parser.add_argument("--lora_path", type=str, help="LoRA checkpoint 目录")
    parser.add_argument("--data_path", type=str, help="HDF5 数据集路径")
    parser.add_argument("--split_file", type=str, help="train_val_split.json 路径")
    parser.add_argument("--output_dir", type=str, default="eval_output", help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None, help="最大评估样本数")
    parser.add_argument("--frame_num", type=int, default=33, help="生成帧数")
    parser.add_argument("--size", type=str, default="480*832", help="视频尺寸")
    parser.add_argument("--t5_cpu", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no_framewise", action="store_true", help="Use cumulative poses instead of framewise for Plücker embeddings")

    # 仅计算指标
    parser.add_argument("--metrics_only", action="store_true",
                        help="跳过推理，仅从已有结果计算指标")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.metrics_only:
        # Step 1: 推理
        assert args.checkpoint_dir, "--checkpoint_dir 必须指定"
        assert args.lora_path, "--lora_path 必须指定"
        assert args.data_path, "--data_path 必须指定"
        assert args.split_file, "--split_file 必须指定"

        val_indices = load_val_indices(args.split_file)
        logger.info(f"验证集 indices: {len(val_indices)} 条")

        samples = load_val_samples(args.data_path, val_indices, args.max_samples)

        run_inference_batch(
            samples=samples,
            checkpoint_dir=args.checkpoint_dir,
            lora_path=args.lora_path,
            output_dir=args.output_dir,
            device=args.device,
            frame_num=args.frame_num,
            size=args.size,
            t5_cpu=args.t5_cpu,
            framewise=not args.no_framewise,
        )

    # Step 2: 计算指标
    logger.info("开始计算指标...")
    results = compute_metrics(args.output_dir, device=args.device)
    save_results(results, args.output_dir)

    logger.info("评估完成!")


if __name__ == "__main__":
    main()
