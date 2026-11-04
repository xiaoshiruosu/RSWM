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

import numpy as np

# [v3.0 P0] 不可作弊契约。下面每一处 min() 都曾是作弊入口:
# 少交帧 / 只交局部图都能拿到 PSNR=inf(外部审查 2026-07-30 动态复现)。
# 默认严格:形状不一致即抛。宽松只能靠 RSWM_LENIENT_METRICS=1,且会污染结果。
from submission_contract import (assert_same_shape, strip_condition_frames,
                                 assert_enough_for_distribution,
                                 contract_summary, CONDITION_FRAMES,
                                 MIN_SAMPLES_FOR_DISTRIBUTION)

# [v3.1] torch 改**惰性导入**。
#
# 原本是模块级 `import torch`,而 torch 只被 4 个深度特征函数用到
# (compute_video_lpips / compute_fid / _load_video_feature_model /
#  _extract_video_features)—— SSIM/PSNR/角误差与整个**反作弊契约**都是
# 纯 numpy/cv2/skimage。
#
# 后果很实际:CI 刻意不装 torch(CPU-only、无网),于是
# test_submission_contract.py 与 test_metrics_known_answer.py **在 CI 里根本跑不起来**
# —— 而这两个恰恰覆盖最重要的东西(七种作弊、条件帧剔除、known-answer 度量)。
# 惰性导入后 CI 能覆盖它们,而 torch 可用时行为逐字不变。
def _torch():
    """需要 torch 的地方调它。缺 torch 时给出能照做的错误,而不是 ImportError 堆栈。"""
    try:
        import torch as _t
        return _t
    except ImportError as e:
        raise RuntimeError(
            "本指标需要 PyTorch(LPIPS / FID / FVD 都在深度特征那条链上),"
            "当前环境没有 torch。\n"
            "  纯 CPU 无 torch 环境下可用的指标:SSIM / PSNR / 角误差 / "
            "以及全部提交契约校验。\n"
            "  要算 LPIPS/FID/FVD 请在 rswm 环境里跑(见 requirements-lock.txt)。"
        ) from e
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
    assert_same_shape(img1, img2, what="SSIM 的输入图像")
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
    assert_same_shape(img1, img2, what="PSNR 的输入图像")
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
    assert_same_shape(gen_frames, gt_frames, what="视频")
    T = min(len(gen_frames), len(gt_frames))
    scores = [compute_ssim(gen_frames[t], gt_frames[t]) for t in range(T)]
    return float(np.mean(scores))


def compute_video_psnr(gen_frames: np.ndarray, gt_frames: np.ndarray) -> float:
    """逐帧计算 PSNR 取平均"""
    assert_same_shape(gen_frames, gt_frames, what="视频")
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
    assert_same_shape(gen_frames, gt_frames, what="视频")
    T = min(len(gen_frames), len(gt_frames))
    scores = []
    for t in range(T):
        g = _torch().from_numpy(gen_frames[t]).permute(2, 0, 1).float() / 127.5 - 1.0
        r = _torch().from_numpy(gt_frames[t]).permute(2, 0, 1).float() / 127.5 - 1.0
        # 对齐尺寸 (取较小的 H, W)
        assert_same_shape(g, r, what="LPIPS 的输入视频")
        h = min(g.shape[1], r.shape[1])
        w = min(g.shape[2], r.shape[2])
        g = g[:, :h, :w].unsqueeze(0).to(device)
        r = r[:, :h, :w].unsqueeze(0).to(device)
        with _torch().no_grad():
            d = model(g, r).item()
        scores.append(d)
    return float(np.mean(scores))


# ============================================================================
# FID (帧级)
# ============================================================================

def _load_frame_feature_model(device: str = "cuda"):
    """加载 FID 的 InceptionV3 特征器；失败返回 ``(None, None)``。

    单独暴露加载/抽特征入口，是为了让隐藏测试评分器逐样本流式抽特征，
    避免把几百条 480×832 的全部预测帧同时留在内存里。
    """
    try:
        from torchvision.models import inception_v3
        model = inception_v3(pretrained=True, transform_input=False).to(device).eval()
        model.fc = _torch().nn.Identity()
        return model, "inception_v3_pool3"
    except Exception:
        return None, None


def _extract_frame_features(
    frames: List[np.ndarray], model, device: str = "cuda", batch_size: int = 64
) -> np.ndarray:
    """从帧列表抽 Inception 特征；模型由 `_load_frame_feature_model` 提供。"""
    feats = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        imgs = []
        for f in batch:
            img = _torch().from_numpy(f).permute(2, 0, 1).float() / 255.0
            img = _torch().nn.functional.interpolate(
                img.unsqueeze(0), size=(299, 299), mode="bilinear",
                align_corners=False)
            imgs.append(img)
        imgs = _torch().cat(imgs, dim=0).to(device)
        with _torch().no_grad():
            feat = model(imgs).cpu().numpy()
        feats.append(feat)
    if not feats:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(feats, axis=0)

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
    from scipy.linalg import sqrtm
    model, _ = _load_frame_feature_model(device)
    if model is None:
        raise RuntimeError("无法加载 InceptionV3 FID 特征器（离线权重缺失？）")
    feat_gen = _extract_frame_features(gen_frames_list, model, device, batch_size)
    feat_gt = _extract_frame_features(gt_frames_list, model, device, batch_size)

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

FVD_EXTRACTOR_NAME = "r3d_18_proxy"   # [v2.6] 随 FVD 一起落盘的提取器身份标记


def _load_video_feature_model(device: str = "cuda"):
    """[v2.6 更名] 加载 FVD 特征提取器 —— 是 torchvision r3d_18【代理】,不是 I3D。

    旧名 `_load_i3d_model` + docstring「加载 I3D 模型」是误导(REVIEWER_STYLE
    §3-O3/§8-B):实际加载 Kinetics-400 预训练 ResNet3D-18(512-d,16 帧
    112×112),与标准 FVD 的 I3D(400-d logits,224×224)**不在同一特征空间**。
    因此本模块的 "FVD" 只在本仓各配置间内部可比,**不可与文献 I3D-FVD 并列**。
    要文献可比的数字,用 server_run/recompute_fvd_i3d.py 离线补算 FVD_I3D
    (标准 i3d_torchscript 口径)。
    ⚠️ 数值行为与旧函数逐字节一致 —— 只改名字与说明,不动任何计算。
    """
    try:
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=True).to(device).eval()
        model.fc = _torch().nn.Identity()
        return model, "r3d_18"
    except Exception:
        return None, None


# 向后兼容别名(外部脚本可能按旧名引用;新代码一律用新名)
_load_i3d_model = _load_video_feature_model


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
        t = _torch().from_numpy(clip).permute(3, 0, 1, 2).float() / 255.0
        t = _torch().nn.functional.interpolate(
            t.unsqueeze(0), size=(16, 112, 112), mode="trilinear", align_corners=False
        ).to(device)
        with _torch().no_grad():
            feat = model(t).cpu().numpy().flatten()
        feats.append(feat)
    return np.stack(feats, axis=0)


def compute_fvd(
    gen_videos: List[np.ndarray],
    gt_videos: List[np.ndarray],
    device: str = "cuda",
) -> float:
    """计算 FVD(r3d_18 代理口径,仅内部可比 —— 见 _load_video_feature_model),
    输入为视频列表, 每个 (T, H, W, C) uint8"""
    from scipy.linalg import sqrtm

    model, name = _load_video_feature_model(device)
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

# [RSRoll-Bench P0#2] 近静止拒答门的阈值：**平均**幅度（px / 像素 / 帧对）。
# 取值依据 instrument_floor_report.py D5 的处方原文（"如 ≥0.1 px/帧对"）。
MIN_MEAN_FLOW_PX = 0.1


def estimate_motion_direction(
    frames: np.ndarray,
    flow_size: Optional[Tuple[int, int]] = None,
    step: Optional[int] = None,
) -> Optional[float]:
    """
    用光流估计视频的主运动方向 (度数, 0-360)。
    输入 (T, H, W, C) uint8。返回 None 表示无法估计。

    使用光流幅度加权，避免微弱运动区域的噪声主导方向估计。

    flow_size=(h, w): 光流前先把帧 INTER_AREA 降到该分辨率(只降不升)。
    Farnebäck (winsize=15, levels=3) 的可靠位移带约 ≤10-15 px/对;
    480x832 生成分辨率下 GT 速度级内容位移 ~34 px/帧,叠加下方 step 采样
    (27 帧 → step=3)后每对 ~102 px,彻底超量程 —— 真实 GT 地板实测 82.2°
    (2026-07-23,与 Track B v1 的 80.67° 同一事故)。校准口径应传原生
    窗口分辨率(H5 metadata.attrs['pose_window_size'],例 128x222)。
    step: 帧对采样间隔;默认 None = len//8(历史行为)。窗口分辨率下
    GT 逐帧 ~9 px,step=3 时每对 ~27 px 已达量程边缘,标定敏感场景可传
    step=1。两个参数都不传时行为与旧版逐字节一致。
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
    n_px_total = 0

    # 均匀采样帧对
    if step is None:
        # [v3.0 核实修正] 默认从 len//8 改成 1。
        # instrument_floor_report 的合成对照实测(窗口 128×222,平滑纹理):
        #     ≤16 px/帧对 → 平均角误差 ≤5°;24 px → **128.77°**、带符号 −113°
        #     (时间混叠,估计器报的是近乎反向);34 px → 75°
        # 而 canonical 在窗口分辨率下 GT 逐帧约 9 px,旧默认 step=T//8 给出
        # T=17→18px / T=25→27px / T=33→36px —— 全部在灾难区。
        # evaluate.py 一直显式传 step=1 所以主链安全,但任何**不传**的调用方
        # (trajectory_metrics 的一处、旧 lora_remotesense/metrics.py)都会静默
        # 拿到 ~128° 的误差。默认值本身就该是安全的那个。
        step = 1
    else:
        step = max(1, int(step))
    for i in range(0, len(frames) - step, step):
        f1 = frames[i]
        f2 = frames[i + step]
        if f1.dtype != np.uint8:
            f1 = np.clip(f1, 0, 255).astype(np.uint8)
        if f2.dtype != np.uint8:
            f2 = np.clip(f2, 0, 255).astype(np.uint8)
        if flow_size is not None:
            fh, fw = int(flow_size[0]), int(flow_size[1])
            if fh <= 0 or fw <= 0:
                raise ValueError(f"flow_size 必须为正: {flow_size!r}")
            # [v2.5 审查修复] ①守卫改 AND:旧 OR 会在单轴超限时把另一轴【上采样】,
            # 与"只降不升"的声明相反;②宽高比硬校验:角度是在 resize【之后】用
            # arctan2 算的,各向异性缩放会直接旋转读数(如 (96,96) 对 480×832,
            # 45° 被读成 60°,~15° 系统性偏置,与 22.5° 量化地板同量级)。
            src_ar = f1.shape[1] / f1.shape[0]
            dst_ar = fw / fh
            if abs(dst_ar - src_ar) / src_ar > 0.05:
                raise ValueError(
                    f"flow_size={flow_size} 与帧尺寸 {f1.shape[:2]} 宽高比不符 "
                    f"({dst_ar:.3f} vs {src_ar:.3f}) —— 各向异性缩放会系统性扭曲"
                    f"角度读数,拒绝计算。请让窗口与生成分辨率同比。")
            if f1.shape[0] > fh and f1.shape[1] > fw:  # 只降不升(两轴都超才降)
                f1 = cv2.resize(f1, (fw, fh), interpolation=cv2.INTER_AREA)
                f2 = cv2.resize(f2, (fw, fh), interpolation=cv2.INTER_AREA)
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
        n_px_total += mag.size

    # [RSRoll-Bench P0#2] 近静止拒答门：**平均**幅度，不是幅度之和。
    # 旧写法 `weight_total < 1.0` 看的是所有像素所有帧对的幅度**之和**，
    # 阈值随像素数增长：128×222 窗口 × 8 对 = 227328 个样本时，
    # 它等价于平均幅度 4.4e-6 px —— 任何一点像素噪声都能越过。
    # 后果（instrument_floor_report.py D5 实测）：逐字相同的帧会被拒答，
    # 但只要加 σ=1 灰阶噪声就**不再拒答**，persistence 类退化生成会拿到
    # 一个纯噪声方向照常进 mean，而不是 None。
    # 本修复按该报告的处方改为平均幅度门（≥0.1 px/帧对）。
    # ⚠️ 这会改变既有数字，必须在任何新一轮生成**之前**落地。
    if n_px_total <= 0:
        return None
    mean_mag = weight_total / n_px_total
    if mean_mag < MIN_MEAN_FLOW_PX:
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
    flow_size: Optional[Tuple[int, int]] = None,
    step: Optional[int] = None,
) -> Dict[str, float]:
    """
    批量计算方向角度误差。
    返回 mean / median / std。
    """
    if len(gen_videos) != len(target_angles):
        # 曾经的 bug: 调用方传全量 videos + 过滤后的 angles, zip 按短的截断 →
        # 视频与目标角整体错位, 角误差全错且不报错。现在硬失败。
        raise ValueError(
            f"compute_batch_angular_error: videos({len(gen_videos)}) 与 "
            f"target_angles({len(target_angles)}) 长度不等 —— 两者必须逐一对齐 "
            f"(无方向的样本请传 None 占位, 本函数会跳过)。"
        )
    errors = []
    per_sample = []   # 与 gen_videos 等长; 无目标角或光流不可靠的样本记 None
    for video, target in zip(gen_videos, target_angles):
        if target is None:          # 该样本没有方向词 (fallback prompt) → 不参与
            per_sample.append(None)
            continue
        est = estimate_motion_direction(video, flow_size=flow_size, step=step)
        if est is not None:
            err = compute_angular_error(est, target)
            errors.append(err)
            per_sample.append(err)
        else:
            per_sample.append(None)

    if not errors:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"),
                "count": 0, "per_sample": per_sample}

    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "count": len(errors),
        # 逐样本值 —— 逐轨迹 bootstrap CI 需要它 (训练种子方差与测试轨迹方差是
        # 两种不同的不确定性, 必须都报, 不能互相替代)。聚合口径不受影响:
        # mean/median/std/count 的计算与含义与之前逐字节一致。
        "per_sample": per_sample,
    }


# ============================================================================
# 汇总
# ============================================================================

def compute_all_metrics(
    gen_videos: List[np.ndarray],
    gt_videos: List[np.ndarray],
    target_angles: Optional[List[float]] = None,
    device: str = "cuda",
    per_sample_out: Optional[List[dict]] = None,
    flow_size: Optional[Tuple[int, int]] = None,
    step: Optional[int] = None,) -> Dict[str, float]:
    """
    计算所有指标。
    gen_videos / gt_videos: 列表, 每个 (T, H, W, C) uint8
    target_angles: 条件方向角度列表 (度)
    per_sample_out: **可选出参**。传入一个 list 时, 就地填入逐样本指标
        [{"SSIM":..., "PSNR":..., "LPIPS":..., "Angular_Error":...}, ...],
        与 gen_videos 等长、同序。用途: 逐轨迹 bootstrap CI (测试轨迹方差,
        与训练种子方差是两回事)。不传则行为与既有调用方完全一致。
        注意: FID/FVD 是**分布级**指标, 定义上没有逐样本值, 故不出现在这里。
    """
    results = {}

    # =====================================================================
    # [v3.0 P0-4 修正位置] 条件帧必须在**一切保真指标之前**剔除。
    #
    # 上一版只在 FID/FVD 之前剔,而 SSIM/PSNR/LPIPS 的逐帧循环在那之前 ——
    # 修了一半。而 PSNR 那一半恰恰是最严重的:
    #   * 首帧是 i2v 的输入,gen[0] 与 gt[0] **逐位相同**是结构性必然;
    #   * compute_video_psnr 是逐帧 PSNR 取平均,MSE=0 那帧给出 inf;
    #   * 于是 mean([inf, 有限, ...]) = inf。
    # 实测:gen 后 16 帧全是噪声、只有首帧对,整段 PSNR = **inf**,
    # 剔除首帧后 = 7.77。也就是说旧口径下**每一份提交**的 PSNR 都是 inf,
    # 比"少交帧拿 inf"更彻底 —— 那条至少要主动作弊,这条是自动发生的。
    # =====================================================================
    _pairs_all = [strip_condition_frames(g, t) for g, t in zip(gen_videos, gt_videos)]
    gen_videos = [g for g, _ in _pairs_all]
    gt_videos = [t for _, t in _pairs_all]
    results["Condition_Frames_Excluded"] = int(CONDITION_FRAMES)
    results["Fidelity_Frames_Per_Sample"] = (
        int(len(gen_videos[0])) if gen_videos else None)

    # 逐样本指标
    ssim_scores, psnr_scores, lpips_scores = [], [], []
    lpips_per_sample = []   # 与 gen_videos 等长; 失败记 None (不能用 append 错位)
    for gen, gt in zip(gen_videos, gt_videos):
        if gen.dtype != np.uint8:
            gen = np.clip(gen, 0, 255).astype(np.uint8)
        if gt.dtype != np.uint8:
            gt = np.clip(gt, 0, 255).astype(np.uint8)
        ssim_scores.append(compute_video_ssim(gen, gt))
        psnr_scores.append(compute_video_psnr(gen, gt))
        try:
            v = compute_video_lpips(gen, gt, device)
            lpips_scores.append(v)
            lpips_per_sample.append(v)
        except Exception as e:
            print(f"Warning: LPIPS 计算失败: {e}")
            lpips_per_sample.append(None)

    results["SSIM"] = float(np.mean(ssim_scores))
    results["PSNR"] = float(np.mean(psnr_scores))
    if lpips_scores:
        results["LPIPS"] = float(np.mean(lpips_scores))

    # =====================================================================
    # [v3.0 P0-4] FID/FVD 的三处口径修正
    #
    # ① **排除条件帧**:首帧是 i2v 的输入,不是模型的预测。旧代码把它算进
    #    所有保真指标 —— 每份提交白拿一帧满分,而且抬高幅度随总帧数变化
    #    (17 帧时 1/17≈5.9%,33 帧时 3.0%),让不同 K 的档互相不可比。
    # ② **FID 不再只取中间帧**:单帧 FID 是 N 个样本的 N 个点,而 FID 需要
    #    足够多的图像才能估协方差。改用**全部预测帧**(排除条件帧后),
    #    样本量从 N 变成 N×(T-1)。
    # ③ **小分组不算分布指标**:旧代码在 len(idx)>=2 的方向×场景交叉块上
    #    也算 FID/FVD —— 那是纯噪声。下限见 MIN_SAMPLES_FOR_DISTRIBUTION。
    # =====================================================================
    # gen_videos/gt_videos 在函数开头已剔除条件帧,这里直接用,不能再剔一次
    # (再剔会把第 2 帧也丢掉)。
    _gen_pred, _gt_pred = gen_videos, gt_videos

    if assert_enough_for_distribution(len(gen_videos), metric="FID"):
        gen_fr = [f for v in _gen_pred for f in v]
        gt_fr = [f for v in _gt_pred for f in v]
        try:
            results["FID"] = compute_fid(gen_fr, gt_fr, device)
            results["FID_Frames"] = len(gen_fr)
            results["FID_Frame_Selection"] = "all_predicted_frames"
        except Exception as e:
            print(f"Warning: FID 计算失败: {e}")
            results["FID"] = float("nan")
    else:
        results["FID"] = None
        results["FID_Skipped_Reason"] = (
            f"n={len(gen_videos)} < {MIN_SAMPLES_FOR_DISTRIBUTION}"
            f" —— 分布型指标在该样本量上是噪声")

    if assert_enough_for_distribution(len(gen_videos), metric="FVD"):
        try:
            results["FVD"] = compute_fvd(_gen_pred, _gt_pred, device)
        except Exception as e:
            print(f"Warning: FVD 计算失败: {e}")
            results["FVD"] = float("nan")
    else:
        results["FVD"] = None
        results["FVD_Skipped_Reason"] = (
            f"n={len(gen_videos)} < {MIN_SAMPLES_FOR_DISTRIBUTION}"
            f" —— 分布型指标在该样本量上是噪声")
    # [v2.6] 提取器身份随数字落盘:此 FVD 是 r3d_18 代理口径,仅内部可比;
    # 文献可比口径见 server_run/recompute_fvd_i3d.py 的 FVD_I3D。
    results["FVD_Extractor"] = FVD_EXTRACTOR_NAME
    # 契约状态随数字落盘:下游据此判断这批数字是不是严格评出来的
    results["Contract"] = contract_summary()

    # 方向角度误差
    angular_per_sample = None
    if target_angles is not None:
        angular = compute_batch_angular_error(gen_videos, target_angles, flow_size=flow_size, step=step)
        results["Angular_Error_Mean"] = angular["mean"]
        results["Angular_Error_Median"] = angular["median"]
        results["Angular_Error_Std"] = angular["std"]
        results["Angular_Error_Count"] = angular["count"]
        angular_per_sample = angular.get("per_sample")

    # 可选出参: 逐样本指标 (逐轨迹 bootstrap 用)。
    # ⚠️ 索引对齐: 调用方若把 target_angles 过滤成"仅有效角"的子集, 传进来的
    # gen_videos 也必须是**同一子集**(evaluate.py:339-341 就是这么做的),
    # 否则角误差会错位。这里只做长度一致性断言, 不猜测调用方意图。
    if per_sample_out is not None:
        n = len(gen_videos)
        if angular_per_sample is not None and len(angular_per_sample) != n:
            raise RuntimeError(
                f"compute_all_metrics: angular per_sample 长度 {len(angular_per_sample)} "
                f"!= 视频数 {n} —— 调用方把 target_angles 与 gen_videos 传成了不同子集。"
            )
        for i in range(n):
            per_sample_out.append({
                "SSIM": float(ssim_scores[i]),
                "PSNR": float(psnr_scores[i]),
                "LPIPS": (float(lpips_per_sample[i])
                          if lpips_per_sample[i] is not None else None),
                "Angular_Error": (angular_per_sample[i]
                                  if angular_per_sample is not None else None),
            })

    return results
