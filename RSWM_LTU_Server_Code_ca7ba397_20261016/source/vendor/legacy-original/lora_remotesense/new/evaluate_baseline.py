#!/usr/bin/env python3
"""
RSWM Baseline 评估 Pipeline（无 LoRA，原始预训练模型）

用法:
  # 推理 + 指标
  python evaluate_baseline.py \
    --checkpoint_dir ./lingbot-world-base-cam \
    --data_path ./data/remotesense/train.h5 \
    --split_file ./outputs/train_val_split.json \
    --output_dir eval_output_baseline \
    --max_samples 100

  # 仅计算指标（已有推理结果）
  python evaluate_baseline.py --metrics_only --output_dir eval_output_baseline
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if (HERE / "wan").is_dir():
    PROJECT_ROOT = HERE
else:
    PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据加载（复用 evaluate.py 的逻辑）
# ============================================================================

from evaluate import (
    load_val_indices,
    load_val_samples,
    compute_metrics,
    save_results,
)


# ============================================================================
# 原始模型加载（不注入 LoRA）
# ============================================================================

def load_base_model(
    checkpoint_dir: str,
    device: torch.device,
    config,
    t5_cpu: bool = True,
    inactive_model_cpu: bool = True,
):
    """加载原始预训练模型，不注入任何 LoRA 权重

    80GB A100 默认开 inactive_model_cpu + t5_cpu，预算与 load_lora_model 相同。
    """
    from wan.modules.model import WanModel
    from wan.modules.vae2_1 import Wan2_1_VAE
    from wan.modules.t5 import T5EncoderModel

    # T5
    logger.info("加载 T5 编码器...")
    t5_device = torch.device("cpu") if t5_cpu else device
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=t5_device,
        checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
        tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
    )
    text_encoder.model.eval().requires_grad_(False)

    # VAE
    logger.info("加载 VAE...")
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
        device=device,
    )

    # DiT（原始权重，无 LoRA）
    logger.info("加载 DiT 模型（原始权重，无 LoRA）...")
    low_noise_model = WanModel.from_pretrained(
        checkpoint_dir,
        subfolder=config.low_noise_checkpoint,
        torch_dtype=torch.bfloat16,
    )
    high_noise_model = WanModel.from_pretrained(
        checkpoint_dir,
        subfolder=config.high_noise_checkpoint,
        torch_dtype=torch.bfloat16,
    )

    low_noise_model.eval().requires_grad_(False)
    high_noise_model.eval().requires_grad_(False)
    if inactive_model_cpu:
        low_noise_model.cpu()
        high_noise_model.cpu()
        logger.info("DiT 模型加载完成（原始权重；两份均驻 CPU，采样时按 timestep 切换）")
    else:
        low_noise_model.to(device)
        high_noise_model.to(device)
        logger.info("DiT 模型加载完成（原始权重；两份均驻 GPU）")

    return text_encoder, vae, low_noise_model, high_noise_model


# ============================================================================
# 推理
# ============================================================================

def run_baseline_inference(
    samples: List[Dict],
    checkpoint_dir: str,
    output_dir: str,
    device: str = "cuda",
    frame_num: int = 33,
    size: str = "480*832",
    t5_cpu: bool = True,
    inactive_model_cpu: bool = True,
    gen_seed: int = 42,
):
    """用原始模型批量推理"""
    from inference import generate_video
    from wan.configs import WAN_CONFIGS
    from PIL import Image
    import shutil
    import tempfile

    gen_dir = Path(output_dir) / "generated"
    gt_dir = Path(output_dir) / "gt"
    meta_dir = Path(output_dir) / "meta"
    gen_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # [v2.9.2 P0-2] baseline 同样要身份门:三份 baseline(drop/native/identity)
    # 目录不同但复用逻辑一样,少了这道门就能拿 identity 的旧 baseline 去减
    # drop 臂,而 Δ 表看不出任何异样。
    from run_manifest import (build_manifest, read_manifest,
                              write_manifest, assert_compatible, summarize)
    _cur_manifest = build_manifest(
        role=os.environ.get("RSWM_RUN_ROLE", "baseline"),
        data_h5=os.environ.get("RSWM_MANIFEST_DATA_H5") or None,
        split_file=os.environ.get("RSWM_MANIFEST_SPLIT") or None,
        checkpoint_id="base",
        prompt_schema=(samples[0].get("prompt_schema") if samples else None),
        frame_num=frame_num,
        gen_seed=gen_seed,
        backbone=os.environ.get("RSWM_BACKBONE", "wan-inline"),
        backbone_adaptation="frozen",
        path_estimator=os.environ.get("RSWM_PATH_ESTIMATOR", "flow"))
    if any(meta_dir.glob("*.json")):
        assert_compatible(read_manifest(output_dir), _cur_manifest,
                          ctx="baseline 断点续推")
    write_manifest(output_dir, _cur_manifest)

    device_obj = torch.device(device)
    config = WAN_CONFIGS["i2v-A14B"]

    # 加载原始模型（无 LoRA）
    logger.info("加载原始模型（无 LoRA）...")
    text_encoder, vae, low_model, high_model = load_base_model(
        checkpoint_dir=checkpoint_dir,
        device=device_obj,
        config=config,
        t5_cpu=t5_cpu,
        inactive_model_cpu=inactive_model_cpu,
    )

    failed = 0   # [v2.7.1 F1/P0]

    for i, sample in enumerate(samples):
        logger.info(f"[{i+1}/{len(samples)}] 推理 {sample['key']}...")
        t0 = time.time()

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
                device=device_obj,
                frame_num=min(frame_num, sample["num_frames"]),
                size=size,
                action_path=action_dir,
                seed=gen_seed,
                t5_cpu=t5_cpu,
                inactive_model_cpu=inactive_model_cpu,
            )
            gen_video = video_tensor.clamp(-1, 1).permute(1, 2, 3, 0)
            gen_video = ((gen_video + 1) * 127.5).byte().cpu().numpy()
        except Exception as e:
            logger.error(f"推理失败 {sample['key']}: {e}")
            failed += 1
            continue
        finally:
            # [v2.7.1 F9/P2] mkdtemp 此前从不清理
            shutil.rmtree(action_dir, ignore_errors=True)

        np.save(str(gen_dir / f"{sample['key']}.npy"), gen_video)

        gt_video = sample["video_frames"]
        if gt_video.dtype != np.uint8:
            gt_video = np.clip(gt_video, 0, 255).astype(np.uint8)
        np.save(str(gt_dir / f"{sample['key']}.npy"), gt_video)

        meta = {
            "key": sample["key"],
            "prompt": sample["prompt"],
            "scene": sample["scene"],
            "direction_angle": sample["direction_angle"],
            "num_frames_gt": sample["num_frames"],
            "num_frames_gen": gen_video.shape[0],

            # [v2.5] 角误差口径与方向来源 —— compute_metrics 靠 meta 里的

            # pose_window_size 决定 flow_size;不写会静默回退 (128,222),

            # 换窗口重建时 baseline 臂与 LoRA 臂会落在两个口径上。

            "pose_window_size": list(sample.get("pose_window_size", (128, 222))),

            "direction_target_source": sample.get("direction_target_source"),

            "direction_word": sample.get("direction_word"),

            # [v2.6/O11] 位姿口径 —— baseline 在 SEM_H5 上评时是 identity(text
            # 路由)。旧目录名启发式对 base=… 恰会静默判成 geo,正是要根治的形态。
            "pose_conditioning": sample.get("pose_conditioning"),
            # [v2.9.1 C5] baseline 的 meta 此前不写 pose_mode/prompt_schema,
            # 于是 v2.9 拆出的三份 baseline(eval_base_{drop,native,identity})
            # 的 overall.pose_mode 全是 null —— 聚合层仍然只能靠目录名区分,
            # 正是这条改动声称要根治的形态。三个字段与 evaluate.py 逐字同源。
            "pose_mode": os.environ.get("RSWM_POSE_MODE", "native"),
            "prompt_schema": sample.get("prompt_schema", "net"),
            "rswm_profile": os.environ.get("RSWM_PROFILE", "unset"),
            "sem_arm_mode": os.environ.get("SEM_ARM_MODE", "unset"),
            "headline_eligible": os.environ.get("RSWM_HEADLINE_ELIGIBLE", "0") == "1",

            # [v2.6] well-posed 双栏依据(与 LoRA 臂同源同口径)。
            "gt_net_displacement_ratio": sample.get("gt_net_displacement_ratio"),
        }
        with open(str(meta_dir / f"{sample['key']}.json"), "w") as f:
            json.dump(meta, f, indent=2)

        elapsed = time.time() - t0
        logger.info(f"  完成, {elapsed:.1f}s, gen shape={gen_video.shape}")
        torch.cuda.empty_cache()

    logger.info("Baseline 推理全部完成")


# ============================================================================
# Main
# ============================================================================
    return failed   # [v2.7.2 ①/P0] main 的失败门要用;上一版这里没返回 → NameError


def main():
    parser = argparse.ArgumentParser(description="RSWM Baseline 评估（无 LoRA）")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="预训练模型路径")
    parser.add_argument("--data_path", type=str, default="", help="HDF5 数据路径")
    parser.add_argument("--split_file", type=str, default="", help="train_val_split.json")
    parser.add_argument("--output_dir", type=str, default="eval_output_baseline")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--frame_num", type=int, default=33)
    parser.add_argument("--size", type=str, default="480*832")
    parser.add_argument("--t5_cpu", action="store_true", default=True)
    parser.add_argument("--no_t5_cpu", dest="t5_cpu", action="store_false")
    parser.add_argument("--inactive_model_cpu", action="store_true", default=True,
                        help="按 timestep 切换 14B（80GB 默认 ON）")
    parser.add_argument("--no_inactive_model_cpu", dest="inactive_model_cpu",
                        action="store_false")
    parser.add_argument("--device", type=str, default="cuda",
                        help="逻辑设备；物理卡由 CUDA_VISIBLE_DEVICES 选择")
    parser.add_argument("--gen_seed", type=int, default=42,
                        help="扩散采样种子(baseline 无训练随机性,多采样种子评估用)")
    parser.add_argument("--metrics_only", action="store_true",
                        help="跳过推理，仅从已有结果计算指标")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # [v2.7.2 ①/P0 修复] 上一版失败门直接读 run_baseline_inference 的局部变量
    # failed(以及 metrics_only 分支外的 samples)→ 正常模式跑完昂贵推理后
    # NameError、--metrics_only 当场 UnboundLocalError。初始化 + 显式返回值。
    samples, failed = [], 0
    if args.metrics_only:
        # [v2.9.3 P0] baseline 也必须过这道门 —— 它是【所有 Δ 的分母】。
        # v2.9.2 只给 evaluate.py 加了,这边的门写在 run_baseline_inference 内部,
        # 而 --metrics_only 恰好把整个函数跳过 → 一次重算就能把 legacy/identity
        # 的旧 baseline 标成 official + headline_eligible=true,且不写 manifest,
        # 下游无法归因。与本轮 P0-2 逐字同型。
        from run_manifest import assert_metrics_only_ok, summarize
        _m = assert_metrics_only_ok(args.output_dir,
                                    ctx="evaluate_baseline --metrics_only")
        logger.info(f"[manifest] --metrics_only 口径一致 — {summarize(_m)}")

    if not args.metrics_only:
        assert args.checkpoint_dir, "--checkpoint_dir 必须指定"
        assert args.data_path, "--data_path 必须指定"
        assert args.split_file, "--split_file 必须指定"

        val_indices = load_val_indices(args.split_file)
        logger.info(f"验证集 indices: {len(val_indices)} 条")
        samples = load_val_samples(args.data_path, val_indices, args.max_samples)
        os.environ["RSWM_MANIFEST_DATA_H5"] = str(args.data_path)
        os.environ["RSWM_MANIFEST_SPLIT"] = str(args.split_file)

        failed = run_baseline_inference(
            samples=samples,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            device=args.device,
            frame_num=args.frame_num,
            size=args.size,
            t5_cpu=args.t5_cpu,
            inactive_model_cpu=args.inactive_model_cpu,
            gen_seed=args.gen_seed,
        )

    logger.info("开始计算指标...")
    # [v2.7.1 F1/P0] 全失败/高失败率硬错;空结果非零退出
    if samples and failed == len(samples):
        raise RuntimeError(f"全部 {len(samples)} 个 baseline 样本推理失败")
    if samples and failed > len(samples) * 0.2:
        raise RuntimeError(f"{failed}/{len(samples)} baseline 样本失败,超 20% 门")
    results = compute_metrics(args.output_dir, device=args.device)
    if not results:
        logger.error("baseline 评估产出为空 —— 非零退出")
        sys.exit(1)
    save_results(results, args.output_dir)
    logger.info("Baseline 评估完成!")


if __name__ == "__main__":
    main()
