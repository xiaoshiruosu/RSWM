#!/usr/bin/env python3
"""
训练完批量生成视频样例 (mp4)

和 evaluate.py 的区别:
  - 不算 metrics，专门出 mp4 给人看
  - 不需要 split_file；可以指定 indices / 随机 N 个 / 按 scene 过滤
  - 同时落 GT mp4 在旁边，方便肉眼对比

用法:
  # 随机 8 个验证样例
  python generate_samples.py \\
      --checkpoint_dir ./lingbot-world-base-cam \\
      --lora_path ./outputs/exp1/best_checkpoint \\
      --data_path ./data/remotesense/train.h5 \\
      --output_dir ./samples \\
      --num_samples 8

  # 指定 sample key
  python generate_samples.py ... --keys sample_00012 sample_00037 sample_00091

  # 按场景过滤 + 第一张图自己换
  python generate_samples.py ... --scene urban --num_samples 4
  python generate_samples.py ... --image ./my_photo.jpg --prompt "..."
"""

import argparse
import logging
import os
import sys
import time
import json
import random
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np

_SCENE_ALIASES = {"flood": "water"}  # [v2.5] 读旧 H5 时归一


def _canon(x):
    return _SCENE_ALIASES.get(str(x), str(x))
import h5py
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if (HERE / "wan").is_dir():
    PROJECT_ROOT = HERE
else:
    PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference import load_lora_model, generate_video
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)


def pick_keys(
    data_path: str,
    num_samples: int,
    keys: Optional[List[str]] = None,
    scene: Optional[str] = None,
    seed: int = 42,
) -> List[str]:
    """从 HDF5 选要生成的 sample key 列表"""
    with h5py.File(data_path, "r") as f:
        all_keys = sorted(f["data"].keys())

        if keys:
            missing = [k for k in keys if k not in all_keys]
            if missing:
                raise KeyError(f"以下 key 不在数据集中: {missing}")
            return keys

        if scene:
            filtered = []
            for k in all_keys:
                cat = _canon(f["data"][k].attrs.get("source_scene_category", ""))
                if cat == _canon(scene):   # [v2.5] flood/water 别名归一
                    filtered.append(k)
            all_keys = filtered
            logger.info(f"过滤 scene='{scene}' 后剩 {len(all_keys)} 个 key")

        if not all_keys:
            raise RuntimeError("没有匹配的样本")

        rng = random.Random(seed)
        pick = rng.sample(all_keys, min(num_samples, len(all_keys)))
        return sorted(pick)


def load_sample(data_path: str, key: str) -> dict:
    with h5py.File(data_path, "r") as f:
        s = f["data"][key]
        return {
            "key": key,
            "image": s["image"][:],
            "video_frames": s["video_frames"][:],
            "poses": s["poses"][:],
            "intrinsics": s["intrinsics"][:],
            "prompt": str(s.attrs.get("prompt", "A satellite video of a remote sensing scene.")),
            "num_frames": int(s.attrs.get("num_frames", s["video_frames"].shape[0])),
            "scene": str(s.attrs.get("source_scene_category", "unknown")),
        }


def video_array_to_mp4(arr_thwc_uint8: np.ndarray, save_path: Path, fps: int):
    """(T,H,W,3) uint8 → mp4，复用 wan.utils.save_video"""
    # save_video 要 (C,T,H,W) float [-1,1]
    t = torch.from_numpy(arr_thwc_uint8).float() / 127.5 - 1.0
    t = t.permute(3, 0, 1, 2)  # (3,T,H,W)
    save_video(
        tensor=t[None],
        save_file=str(save_path),
        fps=fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True, help="基模目录")
    p.add_argument("--lora_path", required=True, help="LoRA checkpoint 目录")
    p.add_argument("--data_path", required=True, help="HDF5 数据集（含 image/video_frames/poses/...）")
    p.add_argument("--output_dir", default=None, help="输出目录（默认 samples_<时间戳>/）")

    # 选样本的三种方式（互斥）
    p.add_argument("--num_samples", type=int, default=8, help="随机抽几个")
    p.add_argument("--keys", nargs="*", default=None, help="直接指定 sample key 列表")
    p.add_argument("--scene", default=None,
                   choices=[None, "general", "water", "urban", "rural", "flood"],  # [v2.5] flood=旧数据别名
                   help="按场景过滤")
    p.add_argument("--seed", type=int, default=42, help="随机抽样种子")

    # 推理参数
    p.add_argument("--frame_num", type=int, default=33)
    p.add_argument("--size", default="480*832")
    p.add_argument("--sampling_steps", type=int, default=70)
    p.add_argument("--shift", type=float, default=10.0)
    p.add_argument("--guide_scale", type=float, default=5.0)
    p.add_argument("--gen_seed", type=int, default=42,
                   help="生成时的 noise seed（同一 key 改这个可以出多个变体）")
    p.add_argument("--t5_cpu", action="store_true", default=True)
    p.add_argument("--no_t5_cpu", dest="t5_cpu", action="store_false")
    p.add_argument("--inactive_model_cpu", action="store_true", default=True,
                   help="按 timestep 切换 14B（80GB 默认 ON）")
    p.add_argument("--no_inactive_model_cpu", dest="inactive_model_cpu",
                   action="store_false")
    p.add_argument("--no_framewise", action="store_true")
    p.add_argument("--no_gt", action="store_true", help="不导出 GT 视频")
    p.add_argument("--device", default="cuda")

    args = p.parse_args()

    if args.output_dir is None:
        args.output_dir = f"samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir)
    (out_dir / "generated").mkdir(parents=True, exist_ok=True)
    if not args.no_gt:
        (out_dir / "gt").mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)

    # 1) 选 key
    keys = pick_keys(args.data_path, args.num_samples, args.keys, args.scene, args.seed)
    logger.info(f"将生成 {len(keys)} 个样本: {keys}")

    # 2) 加载模型（一次）
    device = torch.device(args.device)
    config = WAN_CONFIGS["i2v-A14B"]
    logger.info("加载模型 + LoRA ...")
    text_encoder, vae, low_model, high_model = load_lora_model(
        checkpoint_dir=args.checkpoint_dir,
        lora_path=args.lora_path,
        device=device,
        config=config,
        t5_cpu=args.t5_cpu,
        inactive_model_cpu=args.inactive_model_cpu,
    )
    logger.info("模型加载完成")

    # 3) 循环生成
    index_records = []
    for i, key in enumerate(keys, 1):
        gen_mp4 = out_dir / "generated" / f"{key}.mp4"
        if gen_mp4.exists():
            logger.info(f"[{i}/{len(keys)}] 跳过 {key}（已存在 {gen_mp4.name}）")
            continue

        logger.info(f"[{i}/{len(keys)}] 生成 {key} ...")
        t0 = time.time()
        sample = load_sample(args.data_path, key)

        img_arr = sample["image"]
        if img_arr.dtype != np.uint8:
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_arr)

        action_dir = tempfile.mkdtemp()
        np.save(os.path.join(action_dir, "poses.npy"), sample["poses"])
        np.save(os.path.join(action_dir, "intrinsics.npy"), sample["intrinsics"])

        try:
            video_tensor = generate_video(
                text_encoder=text_encoder,
                vae=vae,
                low_noise_model=low_model,
                high_noise_model=high_model,
                config=config,
                prompt=sample["prompt"],
                image=pil_img,
                device=device,
                frame_num=min(args.frame_num, sample["num_frames"]),
                size=args.size,
                action_path=action_dir,
                seed=args.gen_seed,
                sampling_steps=args.sampling_steps,
                shift=args.shift,
                guide_scale=(args.guide_scale, args.guide_scale),
                t5_cpu=args.t5_cpu,
                framewise=not args.no_framewise,
                inactive_model_cpu=args.inactive_model_cpu,
            )
        except Exception as e:
            logger.error(f"生成失败 {key}: {e}")
            continue

        # save gen mp4
        save_video(
            tensor=video_tensor[None],
            save_file=str(gen_mp4),
            fps=config.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )

        # save gt mp4
        if not args.no_gt:
            gt_mp4 = out_dir / "gt" / f"{key}.mp4"
            gt = sample["video_frames"]
            if gt.dtype != np.uint8:
                gt = np.clip(gt, 0, 255).astype(np.uint8)
            video_array_to_mp4(gt, gt_mp4, config.sample_fps)

        # meta
        meta = {
            "key": key,
            "scene": sample["scene"],
            "prompt": sample["prompt"],
            "num_frames_gt": sample["num_frames"],
            "num_frames_gen": int(video_tensor.shape[1]),
            "gen_seed": args.gen_seed,
            "sampling_steps": args.sampling_steps,
            "shift": args.shift,
            "guide_scale": args.guide_scale,
            "elapsed_sec": round(time.time() - t0, 1),
        }
        with open(out_dir / "meta" / f"{key}.json", "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        index_records.append(meta)

        logger.info(f"  {key}: {meta['elapsed_sec']}s -> {gen_mp4.name}")
        torch.cuda.empty_cache()

    # 4) 总索引
    with open(out_dir / "index.json", "w", encoding="utf-8") as fp:
        json.dump({
            "lora_path": args.lora_path,
            "data_path": args.data_path,
            "samples": index_records,
        }, fp, ensure_ascii=False, indent=2)

    logger.info(f"完成。共 {len(index_records)} 个样例落到 {out_dir}/")
    logger.info(f"  - generated/<key>.mp4   生成结果")
    if not args.no_gt:
        logger.info(f"  - gt/<key>.mp4          ground truth 对照")
    logger.info(f"  - meta/<key>.json       每个样本的元数据")
    logger.info(f"  - index.json            汇总")


if __name__ == "__main__":
    main()
