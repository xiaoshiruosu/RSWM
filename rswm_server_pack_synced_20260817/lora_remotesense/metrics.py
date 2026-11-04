#!/usr/bin/env python3
"""
RSWM 评估指标模块

指标:
  - SSIM / PSNR: 逐帧结构与信噪比
  - LPIPS: 感知相似度
  - FID: Fréchet Inception Distance (帧级分布)
  - FVD: Fréchet Video Distance (视频级分布)
  - Angular Error: 方向可控性 (光流主方向 vs 条件方向)
"""
"""[v2.5 口径警告] 本文件是 **旧** metrics:estimate_motion_direction 没有
flow_size/step 形参,角误差在生成分辨率 + step=len//8 下计算 —— 即已知的量程外
口径(GT 地板实测 82°,v2.5 起一律禁用于论文数字)。

新代码一律用 lora_remotesense/new/metrics.py(带 flow_size/step)。
仍 import 本文件的历史脚本(rebin_overlap_metrics.py 等)产出的 Angular_Error_*
与主表不可比;overlap 分层已按用户拍板③整节删除,不再重跑。
"""
import warnings as _warnings
_warnings.warn(
    "lora_remotesense/metrics.py 是旧口径(无 flow_size/step);角误差不可与 "
    "v2.5 主表并列。新代码请用 lora_remotesense/new/metrics.py。",
    RuntimeWarning, stacklevel=2)



import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

# ============================================================================
# SSIM / PSNR
# ============================================================================

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张图的 SSIM (uint8 或 float32 [0,255])"""
    from skimage.metrics import structural_similarity
    if img1.dtype != np.uint8:
        img1 = np.clip(img1, 0, 255).astype(np.uint8)
    if img2.dtype != np.uint8:
        img2 = np.clip(img2, 0, 255).astype(np.uint8)
    # 确保尺寸一致
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    return structural_similarity(img1, img2, channel_axis=2, data_range=255)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算两张图的 PSNR"""
    if img1.dtype != np.float64:
        img1 = img1.astype(np.float64)
    if img2.dtype != np.float64:
        img2 = img2.astype(np.float64)
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = img1[:h, :w]
    img2 = img2[:h, :w]
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)

def compute_video_ssim(gen_frames: np.ndarray, gt_frames: np.ndarray) -> float:
    """逐帧计算 SSIM 取平均, 输入 (T, H, W, C)"""
    T = min(len(gen_frames), len(gt_frames))
    scores = [compute_ssim(gen_frames[t], gt_frames[t]) for t in range(T)]
    return float(np.mean(scores))


def compute_video_psnr(gen_frames: np.ndarray, gt_frames: np.ndarray) -> float:
    """逐帧计算 PSNR 取平均"""
    T = min(len(gen_frames), len(gt_frames))
    scores = [compute_psnr(gen_frames[t], gt_frames[t]) for t in range(T)]
    return float(np.mean(scores))


# ============================================================================
# LPIPS
# ============================================================================

_lpips_model = None
_lpips_unavailable = False


def _get_lpips_model(device="cuda"):
    global _lpips_model, _lpips_unavailable
    if _lpips_unavailable:
        return None
    if _lpips_model is None:
        try:
            import lpips
            _lpips_model = lpips.LPIPS(net="alex").to(device).eval()
        except Exception as e:
            print(f"Warning: LPIPS 初始化失败，跳过后续 LPIPS 计算: {e}")
            _lpips_unavailable = True
            return None
    return _lpips_model


def compute_video_lpips(
    gen_frames: np.ndarray,
    gt_frames: np.ndarray,
    device: str = "cuda",
) -> float:
    """逐帧计算 LPIPS 取平均, 输入 (T, H, W, C) uint8"""
    model = _get_lpips_model(device)
    if model is None:
        raise RuntimeError("LPIPS model unavailable")
    T = min(len(gen_frames), len(gt_frames))
    scores = []
    for t in range(T):
        g = torch.from_numpy(gen_frames[t]).permute(2, 0, 1).float() / 127.5 - 1.0
        r = torch.from_numpy(gt_frames[t]).permute(2, 0, 1).float() / 127.5 - 1.0
        # 对齐尺寸 (取较小的 H, W)
        h = min(g.shape[1], r.shape[1])
        w = min(g.shape[2], r.shape[2])
        g = g[:, :h, :w].unsqueeze(0).to(device)
        r = r[:, :h, :w].unsqueeze(0).to(device)
        with torch.no_grad():
            d = model(g, r).item()
        scores.append(d)
    return float(np.mean(scores))


# ============================================================================
# FID (帧级)
# ============================================================================

def compute_fid(
    gen_frames_list: List[np.ndarray],
    gt_frames_list: List[np.ndarray],
    device: str = "cuda",
    batch_size: int = 64,
) -> float:
    """
    计算 FID。输入为帧列表，每个元素 (H, W, C) uint8。
    使用 InceptionV3 提取特征后计算 Fréchet distance。
    """
    from torchvision.models import inception_v3
    from scipy.linalg import sqrtm

    model = inception_v3(pretrained=True, transform_input=False).to(device).eval()
    model.fc = torch.nn.Identity()  # 取 2048-d 特征

    def _extract_features(frames: List[np.ndarray]) -> np.ndarray:
        feats = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            imgs = []
            for f in batch:
                img = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                img = torch.nn.functional.interpolate(
                    img.unsqueeze(0), size=(299, 299), mode="bilinear", align_corners=False
                )
                imgs.append(img)
            imgs = torch.cat(imgs, dim=0).to(device)
            with torch.no_grad():
                feat = model(imgs).cpu().numpy()
            feats.append(feat)
        return np.concatenate(feats, axis=0)

    feat_gen = _extract_features(gen_frames_list)
    feat_gt = _extract_features(gt_frames_list)

    mu_gen, sigma_gen = feat_gen.mean(axis=0), np.cov(feat_gen, rowvar=False)
    mu_gt, sigma_gt = feat_gt.mean(axis=0), np.cov(feat_gt, rowvar=False)

    diff = mu_gen - mu_gt
    covmean = sqrtm(sigma_gen @ sigma_gt)   # [v2.7.1 F7] SciPy 1.18 删除 disp 参数
    if isinstance(covmean, tuple):              # 旧 scipy 返回 (sqrtm, errest) 的防御
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_gen + sigma_gt - 2 * covmean)
    return float(fid)


# ============================================================================
# FVD (视频级)
# ============================================================================

def _load_i3d_model(device: str = "cuda"):
    """加载 I3D 模型用于 FVD 特征提取"""
    try:
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=True).to(device).eval()
        model.fc = torch.nn.Identity()
        return model, "r3d_18"
    except Exception:
        return None, None


def _extract_video_features(
    videos: List[np.ndarray],
    model,
    device: str = "cuda",
) -> np.ndarray:
    """从视频列表提取 I3D 特征, 每个视频 (T, H, W, C) uint8"""
    feats = []
    for video in videos:
        T, H, W, C = video.shape
        # 采样 16 帧
        if T >= 16:
            indices = np.linspace(0, T - 1, 16, dtype=int)
        else:
            indices = np.arange(T)
            pad = np.repeat([T - 1], 16 - T)
            indices = np.concatenate([indices, pad])
        clip = video[indices]  # (16, H, W, C)
        # 转 tensor: (C, T, H, W)
        t = torch.from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
        t = torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(16, 112, 112), mode="trilinear", align_corners=False
        ).to(device)
        with torch.no_grad():
            feat = model(t).cpu().numpy().flatten()
        feats.append(feat)
    return np.stack(feats, axis=0)


def compute_fvd(
    gen_videos: List[np.ndarray],
    gt_videos: List[np.ndarray],
    device: str = "cuda",
) -> float:
    """计算 FVD, 输入为视频列表, 每个 (T, H, W, C) uint8"""
    from scipy.linalg import sqrtm

    model, name = _load_i3d_model(device)
    if model is None:
        print("Warning: 无法加载视频特征模型, 跳过 FVD")
        return float("nan")

    feat_gen = _extract_video_features(gen_videos, model, device)
    feat_gt = _extract_video_features(gt_videos, model, device)

    mu_gen, sigma_gen = feat_gen.mean(axis=0), np.cov(feat_gen, rowvar=False)
    mu_gt, sigma_gt = feat_gt.mean(axis=0), np.cov(feat_gt, rowvar=False)

    diff = mu_gen - mu_gt
    covmean = sqrtm(sigma_gen @ sigma_gt)   # [v2.7.1 F7] SciPy 1.18 删除 disp 参数
    if isinstance(covmean, tuple):              # 旧 scipy 返回 (sqrtm, errest) 的防御
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fvd = diff @ diff + np.trace(sigma_gen + sigma_gt - 2 * covmean)
    return float(fvd)


# ============================================================================
# 方向角度误差 (Angular Error)
# ============================================================================

def estimate_motion_direction(frames: np.ndarray) -> Optional[float]:
    """
    用光流估计视频的主运动方向 (度数, 0-360)。
    输入 (T, H, W, C) uint8。返回 None 表示无法估计。

    使用光流幅度加权，避免微弱运动区域的噪声主导方向估计。
    """
    try:
        import cv2
    except ImportError:
        print("Warning: OpenCV 不可用, 跳过方向估计")
        return None

    if len(frames) < 2:
        return None

    dx_total, dy_total = 0.0, 0.0
    weight_total = 0.0

    # 均匀采样帧对
    step = max(1, len(frames) // 8)
    for i in range(0, len(frames) - step, step):
        f1 = frames[i]
        f2 = frames[i + step]
        if f1.dtype != np.uint8:
            f1 = np.clip(f1, 0, 255).astype(np.uint8)
        if f2.dtype != np.uint8:
            f2 = np.clip(f2, 0, 255).astype(np.uint8)
        gray1 = cv2.cvtColor(f1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        # 用光流幅度加权，抑制静止区域噪声
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        dx_total += (flow[..., 0] * mag).sum()
        dy_total += (flow[..., 1] * mag).sum()
        weight_total += mag.sum()

    # 累积幅度太小说明几乎没有运动，方向估计不可靠
    if weight_total < 1.0:
        return None

    angle = np.degrees(np.arctan2(-dy_total, dx_total)) % 360
    return float(angle)


def compute_angular_error(
    estimated_angle: float,
    target_angle: float,
) -> float:
    """计算两个角度之间的最小差 (0-180 度)"""
    diff = abs(estimated_angle - target_angle) % 360
    if diff > 180:
        diff = 360 - diff
    return float(diff)


def compute_batch_angular_error(
    gen_videos: List[np.ndarray],
    target_angles: List[float],
) -> Dict[str, float]:
    """
    批量计算方向角度误差。
    返回 mean / median / std。
    """
    errors = []
    for video, target in zip(gen_videos, target_angles):
        # [v3.0] 显式 step=1 —— 默认 len//8 落在光流混叠区(24px→128° 误差)。
        # 本文件非 canonical 路径(canonical 是 new/metrics.py),但不留雷。
        est = estimate_motion_direction(video, step=1)
        if est is not None:
            err = compute_angular_error(est, target)
            errors.append(err)

    if not errors:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "count": 0}

    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "count": len(errors),
    }


# ============================================================================
# 汇总
# ============================================================================

def compute_all_metrics(
    gen_videos: List[np.ndarray],
    gt_videos: List[np.ndarray],
    target_angles: Optional[List[float]] = None,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    计算所有指标。
    gen_videos / gt_videos: 列表, 每个 (T, H, W, C) uint8
    target_angles: 条件方向角度列表 (度)
    """
    results = {}

    # 逐样本指标
    ssim_scores, psnr_scores, lpips_scores = [], [], []
    for gen, gt in zip(gen_videos, gt_videos):
        if gen.dtype != np.uint8:
            gen = np.clip(gen, 0, 255).astype(np.uint8)
        if gt.dtype != np.uint8:
            gt = np.clip(gt, 0, 255).astype(np.uint8)
        ssim_scores.append(compute_video_ssim(gen, gt))
        psnr_scores.append(compute_video_psnr(gen, gt))
        try:
            lpips_scores.append(compute_video_lpips(gen, gt, device))
        except Exception as e:
            print(f"Warning: LPIPS 计算失败: {e}")

    results["SSIM"] = float(np.mean(ssim_scores))
    results["PSNR"] = float(np.mean(psnr_scores))
    if lpips_scores:
        results["LPIPS"] = float(np.mean(lpips_scores))

    # FID (取每个视频的中间帧)
    gen_mid = [v[len(v) // 2] for v in gen_videos]
    gt_mid = [v[len(v) // 2] for v in gt_videos]
    try:
        results["FID"] = compute_fid(gen_mid, gt_mid, device)
    except Exception as e:
        print(f"Warning: FID 计算失败: {e}")
        results["FID"] = float("nan")

    # FVD
    try:
        results["FVD"] = compute_fvd(gen_videos, gt_videos, device)
    except Exception as e:
        print(f"Warning: FVD 计算失败: {e}")
        results["FVD"] = float("nan")

    # 方向角度误差
    if target_angles is not None:
        angular = compute_batch_angular_error(gen_videos, target_angles)
        results["Angular_Error_Mean"] = angular["mean"]
        results["Angular_Error_Median"] = angular["median"]
        results["Angular_Error_Std"] = angular["std"]
        results["Angular_Error_Count"] = angular["count"]

    return results
