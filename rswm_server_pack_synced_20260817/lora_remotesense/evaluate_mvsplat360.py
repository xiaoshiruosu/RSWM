#!/usr/bin/env python3
"""
MVSplat360 Baseline 评估 Pipeline

给 MVSplat360 更多信息（5 context views + camera poses）的前提下，
测试通用 NVS 方法在遥感数据上的表现，与 domain-adapted LoRA 对比。

用法:
  # 推理 + 指标
  python lora_remotesense/evaluate_mvsplat360.py \
    --mvsplat360_dir <MVSPLAT_DIR> \
    --checkpoint <MVSPLAT_DIR>/checkpoints/dl3dv_480p.ckpt \
    --data_path <WORK>/data/train_semantic.h5 \
    --split_file <WORK>/train_.../train_val_split.json \
    --output_dir eval_output_mvsplat360 \
    --max_samples 20

  # 仅计算指标（已有推理结果）
  python lora_remotesense/evaluate_mvsplat360.py --metrics_only --output_dir eval_output_mvsplat360
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import h5py
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)

# [v2.6] 改用 canonical 评估器(lora_remotesense/new/evaluate.py)。
# 旧版 import 的是顶层 legacy lora_remotesense/evaluate.py —— 那份没有 v2.5
# 生产口径修复(角误差在生成分辨率 + step=len//8 上算 = 量程外的 82° 地板那套),
# 也没有 v2.6 的 pose_conditioning / well-posed 字段。用它出的 eval_results.json
# 与 LoRA 臂不同口径,禁止入表。
_NEW = ROOT / "lora_remotesense" / "new"
sys.path.insert(0, str(_NEW))
from evaluate import (          # noqa: E402  ← lora_remotesense/new/evaluate.py
    load_val_indices,
    load_val_samples,
    compute_metrics,
    save_results,
)


# ============================================================================
# 数据格式转换: RSWM → MVSplat360
# ============================================================================

def select_context_indices(num_frames: int, num_context: int = 5) -> List[int]:
    """均匀选取 context frame indices"""
    if num_frames <= num_context:
        return list(range(num_frames))
    return [int(round(i * (num_frames - 1) / (num_context - 1)))
            for i in range(num_context)]


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    """将 c2w (4x4) 转换为 w2c (4x4)"""
    w2c = np.eye(4, dtype=np.float32)
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return w2c


def build_intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """构建 3x3 intrinsics matrix"""
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    return K


def normalize_intrinsics(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """归一化 intrinsics (按图像尺寸)"""
    K_norm = K.copy()
    K_norm[0, :] /= width
    K_norm[1, :] /= height
    return K_norm


def prepare_mvsplat360_batch(
    sample: Dict,
    num_context: int = 5,
    target_size: Tuple[int, int] = (480, 832),
) -> Dict:
    """将 RSWM 样本转换为 MVSplat360 输入格式

    Returns dict with:
        context_images: (V, 3, H, W) float32 [0, 1]
        context_extrinsics: (V, 4, 4) float32 w2c
        context_intrinsics: (V, 3, 3) float32 normalized
        target_extrinsics: (T, 4, 4) float32 w2c
        target_intrinsics: (T, 3, 3) float32 normalized
        target_indices: list of int (indices in original video)
        context_indices: list of int
    """
    video = sample["video_frames"]  # (T, H, W, 3) uint8
    poses = sample["poses"]          # (T, 4, 4) c2w
    intrinsics = sample["intrinsics"]  # (T, 4) [fx, fy, cx, cy]
    T = video.shape[0]
    H, W = target_size

    ctx_idx = select_context_indices(T, num_context)
    tgt_idx = [i for i in range(T) if i not in ctx_idx]

    if not tgt_idx:
        tgt_idx = list(range(T))

    def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
        """(N, H, W, 3) uint8 → (N, 3, H, W) float32 [0, 1]"""
        t = torch.from_numpy(frames).float() / 255.0
        t = t.permute(0, 3, 1, 2)  # (N, 3, H, W)
        if t.shape[2] != H or t.shape[3] != W:
            t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        return t

    def convert_poses(indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """转换指定 indices 的 poses 和 intrinsics"""
        extr_list = []
        intr_list = []
        for i in indices:
            w2c = c2w_to_w2c(poses[i])
            extr_list.append(w2c)

            fx, fy, cx, cy = intrinsics[i]
            K = build_intrinsics_matrix(fx, fy, cx, cy)
            K_norm = normalize_intrinsics(K, W, H)
            intr_list.append(K_norm)

        return (
            torch.from_numpy(np.stack(extr_list)),
            torch.from_numpy(np.stack(intr_list)),
        )

    ctx_frames = video[ctx_idx]
    ctx_images = frames_to_tensor(ctx_frames)
    ctx_extr, ctx_intr = convert_poses(ctx_idx)

    tgt_extr, tgt_intr = convert_poses(tgt_idx)

    return {
        "context_images": ctx_images,
        "context_extrinsics": ctx_extr,
        "context_intrinsics": ctx_intr,
        "target_extrinsics": tgt_extr,
        "target_intrinsics": tgt_intr,
        "target_indices": tgt_idx,
        "context_indices": ctx_idx,
    }


# ============================================================================
# MVSplat360 模型加载与推理
# ============================================================================

def load_mvsplat360_model(
    mvsplat360_dir: str,
    checkpoint_path: str,
    device: torch.device,
) -> object:
    """加载 MVSplat360 模型

    通过动态导入 MVSplat360 的模块来加载模型。
    """
    mvsplat360_dir = Path(mvsplat360_dir)
    sys.path.insert(0, str(mvsplat360_dir))

    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        config_dir = str(mvsplat360_dir / "config")

        with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
            cfg = compose(
                config_name="main",
                overrides=[
                    "+experiment=dl3dv_mvsplat360",
                    "mode=test",
                    f"checkpointing.load={checkpoint_path}",
                ],
            )

        from src.model.model_wrapper import ModelWrapper
        model = ModelWrapper(cfg)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)

        model = model.to(device).eval()
        logger.info(f"MVSplat360 模型加载完成: {checkpoint_path}")
        return model

    except Exception as e:
        logger.warning(f"Hydra 方式加载失败: {e}")
        logger.info("尝试直接加载模型...")
        return _load_model_direct(mvsplat360_dir, checkpoint_path, device)


def _load_model_direct(
    mvsplat360_dir: Path,
    checkpoint_path: str,
    device: torch.device,
) -> object:
    """直接加载方式（不依赖 Hydra config）"""
    try:
        from src.model.encoder import Encoder
        from src.model.decoder import Decoder

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        cfg = ckpt.get("hyper_parameters", {}).get("cfg", None)

        if cfg is not None:
            from src.model.model_wrapper import ModelWrapper
            model = ModelWrapper(cfg)
            model.load_state_dict(ckpt["state_dict"], strict=False)
            model = model.to(device).eval()
            logger.info("通过 checkpoint hyper_parameters 加载成功")
            return model
    except Exception as e:
        logger.warning(f"直接加载也失败: {e}")

    raise RuntimeError(
        f"无法加载 MVSplat360 模型。请确认:\n"
        f"  1. {mvsplat360_dir} 包含完整的 MVSplat360 代码\n"
        f"  2. {checkpoint_path} 是有效的 checkpoint\n"
        f"  3. conda 环境包含所有依赖"
    )


@torch.no_grad()
def run_mvsplat360_single(
    model,
    batch: Dict,
    device: torch.device,
    target_size: Tuple[int, int] = (480, 832),
) -> Optional[np.ndarray]:
    """对单个样本运行 MVSplat360 推理

    Returns:
        generated frames: (T_target, H, W, 3) uint8, or None on failure
    """
    H, W = target_size

    try:
        ctx_images = batch["context_images"].unsqueeze(0).to(device)
        ctx_extr = batch["context_extrinsics"].unsqueeze(0).to(device)
        ctx_intr = batch["context_intrinsics"].unsqueeze(0).to(device)
        tgt_extr = batch["target_extrinsics"].unsqueeze(0).to(device)
        tgt_intr = batch["target_intrinsics"].unsqueeze(0).to(device)

        model_input = {
            "context": {
                "image": ctx_images,
                "extrinsics": ctx_extr,
                "intrinsics": ctx_intr,
            },
            "target": {
                "extrinsics": tgt_extr,
                "intrinsics": tgt_intr,
            },
        }

        output = model(model_input)

        if isinstance(output, dict):
            rendered = output.get("color", output.get("video", output.get("images")))
        elif isinstance(output, (list, tuple)):
            rendered = output[0]
        else:
            rendered = output

        if rendered is None:
            logger.warning("模型输出为 None")
            return None

        # rendered: expected (B, T, C, H_out, W_out) or (B, C, T, H_out, W_out)
        if rendered.dim() == 5:
            if rendered.shape[2] == 3:  # (B, T, 3, H, W)
                rendered = rendered[0]  # (T, 3, H, W)
            elif rendered.shape[1] == 3:  # (B, 3, T, H, W)
                rendered = rendered[0].permute(1, 0, 2, 3)  # (T, 3, H, W)
            else:
                rendered = rendered[0]
        elif rendered.dim() == 4:
            if rendered.shape[1] == 3:  # (T, 3, H, W)
                pass
            else:  # (T, H, W, 3) already
                rendered = rendered.permute(0, 3, 1, 2)

        # Resize to target
        T_out = rendered.shape[0]
        if rendered.shape[2] != H or rendered.shape[3] != W:
            rendered = F.interpolate(
                rendered, size=(H, W), mode="bilinear", align_corners=False
            )

        # → (T, H, W, 3) uint8
        rendered = rendered.permute(0, 2, 3, 1)  # (T, H, W, 3)
        rendered = rendered.clamp(0, 1) * 255.0
        frames = rendered.byte().cpu().numpy()

        if np.isnan(frames.astype(float)).any():
            logger.warning("输出包含 NaN")
            return None

        return frames

    except Exception as e:
        logger.error(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# 组装完整视频（context + target）
# ============================================================================

def assemble_full_video(
    context_frames: np.ndarray,
    context_indices: List[int],
    generated_frames: Optional[np.ndarray],
    target_indices: List[int],
    total_frames: int,
    target_size: Tuple[int, int] = (480, 832),
) -> np.ndarray:
    """将 context 帧和生成的 target 帧合并为完整视频

    context 位置用原始帧填充，target 位置用生成帧填充。
    如果生成失败，target 位置用零帧（黑色）填充。
    """
    H, W = target_size
    video = np.zeros((total_frames, H, W, 3), dtype=np.uint8)

    for i, idx in enumerate(context_indices):
        if idx < total_frames and i < len(context_frames):
            frame = context_frames[i]
            if frame.shape[0] != H or frame.shape[1] != W:
                import cv2
                frame = cv2.resize(frame, (W, H))
            video[idx] = frame

    if generated_frames is not None:
        for i, idx in enumerate(target_indices):
            if idx < total_frames and i < len(generated_frames):
                video[idx] = generated_frames[i]

    return video


# ============================================================================
# 主推理循环
# ============================================================================

def run_mvsplat360_evaluation(
    samples: List[Dict],
    mvsplat360_dir: str,
    checkpoint_path: str,
    output_dir: str,
    num_context: int = 5,
    device: str = "cuda",
    target_size: Tuple[int, int] = (480, 832),
    gen_seed: int = 42,
):
    """MVSplat360 批量推理"""
    gen_dir = Path(output_dir) / "generated"
    gt_dir = Path(output_dir) / "gt"
    meta_dir = Path(output_dir) / "meta"
    gen_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    device_obj = torch.device(device)

    # [v2.6] 位姿健全性守卫:MVSplat360 是【位姿条件】的 NVS —— 喂 identity
    # 位姿(SEM_H5 被 neutralize 过)时 5 个 context 相机全部重合,渲染必然
    # 退化。旧 p6_mvsplat.sh 恰好用的是 $SEM_H5(既存 bug,已同步改 GEO)。
    # 这里再兜一道:全部样本 poses 都是 identity → 当场 FATAL,别烧 GPU。
    _n_ident = sum(
        1 for s in samples
        if np.allclose(np.asarray(s["poses"], dtype=np.float64), np.eye(4), atol=1e-6))
    if samples and _n_ident == len(samples):
        raise RuntimeError(
            "全部样本的 poses 都是 identity(位姿被中和的 SEM_H5?)。"
            "MVSplat360 需要真实相机位姿 —— 请改用 GEO_H5(原生 waypoint 位姿)。"
            "若由 p6_mvsplat.sh / mvsplat_smoke.sh 驱动且配置已是 GEO,多半是"
            "陈旧子集文件在作祟:删 $STAMPS 下 p6_subset.sha256 / "
            "mvsplat_smoke_subset.sha256 与对应子集 H5 重抽(v2.6.1 起 sha 绑定"
            "已自动处理,这种情况只出现在手工搬运旧文件时)。")

    logger.info(f"加载 MVSplat360 模型...")
    model = load_mvsplat360_model(mvsplat360_dir, checkpoint_path, device_obj)
    logger.info(f"模型加载完成, {num_context} context views 模式")

    succeeded, failed = 0, 0

    for i, sample in enumerate(samples):
        key = sample["key"]

        existing = gen_dir / f"{key}.npy"
        if existing.exists():
            logger.info(f"[{i+1}/{len(samples)}] 跳过 {key} (已存在)")
            succeeded += 1
            continue

        logger.info(f"[{i+1}/{len(samples)}] 推理 {key}...")
        t0 = time.time()

        T = sample["num_frames"]

        batch = prepare_mvsplat360_batch(
            sample,
            num_context=num_context,
            target_size=target_size,
        )

        # [v2.6] 采样种子:SVD refinement 是随机采样;不固定种子 = 不可复现。
        # 逐样本重置(而非全局一次),使断点续推与全量跑逐位一致。
        torch.manual_seed(int(gen_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(gen_seed))

        gen_error = None
        gen_target_frames = run_mvsplat360_single(
            model, batch, device_obj, target_size
        )
        if gen_target_frames is None:
            gen_error = "run_mvsplat360_single 返回 None(推理异常/输出 NaN,见日志)"

        ctx_frames = sample["video_frames"][batch["context_indices"]]
        full_video = assemble_full_video(
            context_frames=ctx_frames,
            context_indices=batch["context_indices"],
            generated_frames=gen_target_frames,
            target_indices=batch["target_indices"],
            total_frames=T,
            target_size=target_size,
        )

        np.save(str(gen_dir / f"{key}.npy"), full_video)

        gt_video = sample["video_frames"]
        if gt_video.dtype != np.uint8:
            gt_video = np.clip(gt_video, 0, 255).astype(np.uint8)
        np.save(str(gt_dir / f"{key}.npy"), gt_video)

        elapsed = time.time() - t0
        meta = {
            "key": key,
            "prompt": sample["prompt"],
            "scene": sample["scene"],
            "direction_angle": sample["direction_angle"],
            "num_frames_gt": T,
            "num_frames_gen": full_video.shape[0],
            "method": "mvsplat360",
            "num_context": num_context,
            "context_indices": batch["context_indices"],
            "target_indices": batch["target_indices"],
            "generation_success": gen_target_frames is not None,
            # [v2.6] 失败原因落盘 —— p6 黑帧自检的提示语一直让人"查 meta 的
            # error 字段",但旧版从没写过这个字段。
            "error": gen_error,
            # [v2.6] 复现性与口径标记(桥接层 bridge/记录契约必读字段):
            "gen_seed": int(gen_seed),
            "infer_seconds": round(float(elapsed), 3),
            # v2.5/v2.6 口径标记(new/evaluate.compute_metrics 的版本门要求;
            # 值由 canonical load_val_samples 提供,与 LoRA 臂逐位同源)
            "pose_window_size": list(sample.get("pose_window_size", (128, 222))),
            "direction_target_source": sample.get("direction_target_source"),
            "direction_word": sample.get("direction_word"),
            "pose_conditioning": sample.get("pose_conditioning"),
            "gt_net_displacement_ratio": sample.get("gt_net_displacement_ratio"),
        }
        with open(str(meta_dir / f"{key}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if gen_target_frames is not None:
            succeeded += 1
            logger.info(f"  成功, {elapsed:.1f}s, shape={full_video.shape}")
        else:
            failed += 1
            logger.warning(f"  失败 (黑帧填充), {elapsed:.1f}s")

        torch.cuda.empty_cache()

    logger.info(f"推理完成: {succeeded} 成功, {failed} 失败 / {len(samples)} 总计")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MVSplat360 Baseline 评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # [v2.4.3] 旧默认值指向已退役的 mv44 项目(不可复用)—— 改为必填,
    # 不给任何默认路径,防止无参调用静默打向已关闭的项目。
    parser.add_argument("--mvsplat360_dir", type=str, required=True,
                        help="MVSplat360 代码目录(必填;旧 mv44 默认已退役)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="MVSplat360 checkpoint 路径(必填;旧 mv44 默认已退役)")
    parser.add_argument("--data_path", type=str, default="", help="HDF5 数据路径")
    parser.add_argument("--split_file", type=str, default="", help="train_val_split.json")
    parser.add_argument("--output_dir", type=str, default="eval_output_mvsplat360")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--num_context", type=int, default=5,
                        help="Context views 数量 (default: 5)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--metrics_only", action="store_true",
                        help="跳过推理，仅计算指标")
    parser.add_argument("--no_metrics", action="store_true",
                        help="[v2.6] 只推理不算指标(冒烟走 baseline record 契约打分,"
                             "见 server_run/mvsplat_smoke.sh)")
    parser.add_argument("--gen_seed", type=int, default=42,
                        help="[v2.6] 逐样本采样种子(SVD refinement 随机;固定以复现)")

    args = parser.parse_args()
    if args.metrics_only and args.no_metrics:
        parser.error("--metrics_only 与 --no_metrics 互斥")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.metrics_only:
        assert args.data_path, "--data_path 必须指定"
        assert args.split_file, "--split_file 必须指定"

        val_indices = load_val_indices(args.split_file)
        logger.info(f"验证集 indices: {len(val_indices)} 条")
        samples = load_val_samples(args.data_path, val_indices, args.max_samples)

        run_mvsplat360_evaluation(
            samples=samples,
            mvsplat360_dir=args.mvsplat360_dir,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            num_context=args.num_context,
            device=args.device,
            gen_seed=args.gen_seed,
        )

    if args.no_metrics:
        logger.info("--no_metrics:跳过指标(用 baseline record 契约打分,"
                    "见 server_run/baselines/README_BASELINES.md 与 mvsplat_smoke.sh)")
        return
    logger.info("开始计算指标...")
    results = compute_metrics(args.output_dir, device=args.device)
    save_results(results, args.output_dir)

    # 打印方法信息
    print(f"\n{'='*70}")
    print(f"MVSplat360 Baseline (num_context={args.num_context})")
    print(f"{'='*70}")

    # 统计成功/失败
    meta_dir = Path(args.output_dir) / "meta"
    if meta_dir.exists():
        success = sum(1 for f in meta_dir.glob("*.json")
                     if json.load(open(f)).get("generation_success", True))
        total = len(list(meta_dir.glob("*.json")))
        print(f"Generation: {success}/{total} succeeded")

    logger.info("MVSplat360 评估完成!")


if __name__ == "__main__":
    main()
