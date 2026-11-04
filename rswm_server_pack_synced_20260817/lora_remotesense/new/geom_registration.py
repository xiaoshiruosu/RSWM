#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""geom_registration.py — 基于平面配准的轨迹恢复与几何一致性度量。

存在的理由（一句话）：现有 Track B 的全部轨迹指标都由 **Farnebäck 稠密光流**
积分得到，而稠密光流只在**相邻帧高度重叠**时有效。旧 benchmark 相邻帧位移
约 9.6 px / 222 px 窗口（≈95% 帧间重叠），所以光流能用；但新任务的整个卖点是
**低重叠**，一旦帧间重叠下降，光流估计器先崩，度量就不再是在测模型而是在测仪器。
本模块用**特征配准（ORB/SIFT + RANSAC 相似变换）**替换估计器，并额外给出三件
光流原理上给不出的量：**绝对尺度**（走了多远，不是走了什么方向）、**尺度漂移**
（模型有没有偷偷缩放）、**绝对定位**（生成帧能否在源 mosaic 中被定位）。

设计约束（与既有工程一致，勿破坏）：
  * **⚠️ 既有工程里有两套列序，接线时必须显式转换，否则是静默的 90°/180° 错误。**
    实测确认（2026-07-30）：
      - H5 的 `waypoints` = **(cy, cx) 相机**路径（`_camera_net_angle` 与之逐字一致）；
      - `trajectory_metrics.recover_content_path` 返回 **(x, y) 内容**路径
        （x=右/东正，y=下/南正，见 `_step_vector` / `_path_from_flows` 的 docstring），
        且 `segment_wise_angular_error` 内部按 `_vec_angle(d[0], d[1]) = _vec_angle(dx, dy)` 消费它。
    本模块的**原生输出是 (cy, cx) 相机路径**（与 waypoints 同序，便于和 GT 直接比）。
    要喂给 Track B 的任何"recovered_path"参数，**必须先过 `camera_path_to_content_xy()`**。
    直接把相机路径喂进去，实测逐段角误差会变成 90°/154° 这类似是而非的值（见自检 [9]）。
  * 角度约定：θ = degrees(atan2(-dy, dx)) % 360，即 east=0 / north=90 / west=180 / south=270；
    **内容流 = 相机 + 180°**。
  * **fail-loud，绝不用 0 冒充完美**：置信度不足一律返回 None 并置 degenerate 标志。
    这条是既有代码的硬规范（见 metrics.py 的 `weight_total < 1.0 -> None` 与
    trajectory_metrics 静止视频"全部字段为 None"的自检）。
  * 纯 numpy + opencv，不依赖 torch / 网络 / COLMAP / 深度模型 → 断网节点可跑，CPU 可跑。

自检：`python3 code/geom_registration.py --self-test`（含故障注入：注入错误实现后断言必须变红）
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from pack_path import resolve_or_exit_code  # noqa: E402

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise SystemExit("geom_registration 需要 opencv-python(-headless)：%s" % e)


# =============================================================================
# 配准结果容器
# =============================================================================

MIN_INLIERS = 12          # RANSAC 内点下限；低于此视为配准失败（不是 0 位移）
MIN_INLIER_RATIO = 0.15   # 内点率下限
MAX_SCALE_JUMP = 1.60     # 单步尺度比上限；超出视为退化（真实平移应 ≈1）
AMBIG_INLIER_RATIO = 0.55  # 第二假设/第一假设 的内点比阈值；超过即判"周期性歧义"
AMBIG_MIN_SEP_PX = 8.0     # 两个假设的平移差超过它才算"不同的解"（否则是同一解的抖动）


@dataclass
class PairRegistration:
    """一对图像之间的相似变换（平移 + 旋转 + 各向同性尺度）。

    dy, dx: 后帧相对前帧的**内容位移**（像素，图像坐标，y 向下）。
    scale : 后帧相对前帧的尺度比（>1 表示放大 / 视场变窄）。
    rot_deg: 面内旋转（度，逆时针为正）。
    ok    : 是否可信；False 时 dy/dx/scale/rot_deg 全部为 None。
    """

    ok: bool
    dy: Optional[float] = None
    dx: Optional[float] = None
    scale: Optional[float] = None
    rot_deg: Optional[float] = None
    n_inliers: int = 0
    n_matches: int = 0
    method: str = ""
    reason: str = ""
    M: Optional[np.ndarray] = None   # 2x3 相似变换（a→b），供复合累积用

    @property
    def inlier_ratio(self) -> float:
        return (self.n_inliers / self.n_matches) if self.n_matches else 0.0


@dataclass
class PathRecovery:
    """一段视频恢复出的窗口轨迹与几何诊断。"""

    path: Optional[np.ndarray]              # (T,2) float64, (cy,cx)，首帧锚定在 (0,0)
    per_step: List[PairRegistration] = field(default_factory=list)
    n_failed_steps: int = 0
    failed_indices: List[int] = field(default_factory=list)   # 哪几步失败（下游需排除，勿当静止）
    scale_drift: Optional[float] = None     # 末帧相对首帧的累计尺度 − 1（0 = 无漂移）
    rot_drift_deg: Optional[float] = None   # 末帧相对首帧的累计面内旋转（度）
    scale_curve: Optional[np.ndarray] = None  # (T,) 逐帧累计尺度
    median_step_px: Optional[float] = None    # 逐步位移中位数（存活门依据）
    rot_drift_wrapped_deg: Optional[float] = None  # 复合矩阵读数（模 180，仅交叉校验）
    valid: bool = False
    reason: str = ""


# =============================================================================
# 1. 成对配准
# =============================================================================

def _to_gray_u8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        if a.shape[2] == 3:
            a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        elif a.shape[2] == 1:
            a = a[:, :, 0]
        else:
            raise ValueError("不支持的通道数 %d" % a.shape[2])
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


SELF_SIM_THRESH = 0.80    # 自相似次峰得分阈；超过即判"周期性内容，本质歧义"


def self_similarity_score(img: np.ndarray, *, exclude_px: int = 12) -> float:
    """图像自相似度：中心块在自身内的**次峰**归一化相关得分（0~1，越高越歧义）。

    为什么需要它：完全周期的内容（农田垄行、停车场、光伏阵列）上，**所有**特征匹配
    都一致地支持同一个"错了整数个周期"的解，于是"第二假设"守卫也失效——
    压根不存在互相竞争的假设。这时唯一诚实的做法是识别**成因**：内容本身是周期的，
    位移在原理上不可辨识，任何估计器给出的数都不该被采信。
    """
    g = _to_gray_u8(img).astype(np.float32)
    h, w = g.shape
    th, tw = max(16, h // 3), max(16, w // 3)
    ty, tx = (h - th) // 2, (w - tw) // 2
    tmpl = g[ty:ty + th, tx:tx + tw]
    if tmpl.std() < 1e-3:
        return 1.0                      # 纯色：完全不可辨识
    res = cv2.matchTemplate(g, tmpl, cv2.TM_CCOEFF_NORMED)
    res = res.copy()
    y0 = min(max(0, ty), res.shape[0] - 1)
    x0 = min(max(0, tx), res.shape[1] - 1)
    ys, ye = max(0, y0 - exclude_px), min(res.shape[0], y0 + exclude_px + 1)
    xs, xe = max(0, x0 - exclude_px), min(res.shape[1], x0 + exclude_px + 1)
    res[ys:ye, xs:xe] = -1.0            # 屏蔽真峰邻域，只看次峰
    return float(res.max())


def _make_detector(kind: str, nfeatures: int):
    if kind == "orb":
        return cv2.ORB_create(nfeatures=nfeatures, fastThreshold=5, edgeThreshold=8)
    if kind == "sift":
        return cv2.SIFT_create(nfeatures=nfeatures)
    raise ValueError("未知 detector: %r（只支持 orb / sift）" % kind)


def register_pair(
    img_a: np.ndarray,
    img_b: np.ndarray,
    *,
    detector: str = "orb",
    nfeatures: int = 4000,
    ratio: float = 0.80,
    ransac_thresh: float = 3.0,
    min_inliers: int = MIN_INLIERS,
    ambiguity_guard: bool = True,
) -> PairRegistration:
    """估计 img_b 相对 img_a 的相似变换。失败返回 ok=False（不返回零位移）。

    用 estimateAffinePartial2D 而非 findHomography：正射影像上的滑窗只有
    平移+尺度（+可能的面内旋转），4 自由度模型比 8 自由度更稳、更不会把噪声
    拟合成透视畸变；同时**保留尺度**，这正是光流积分丢掉的那一维。

    **周期性歧义守卫**（`ambiguity_guard`，2026-07-30 审查后新增）：
    在农田垄行、停车场、集装箱堆场、光伏阵列这类**周期性结构**上，配准会以
    **满内点 + 尺度 1.0 的高置信度**返回一个"错了整数个周期"的解——
    内点数、内点率、尺度这三道守卫**按定义都抓不到它**（混叠解的内点是完美的）。
    守卫做法：把第一次 RANSAC 的外点再拟合一次；若第二个解拥有可比的支持度
    （内点 ≥ AMBIG_INLIER_RATIO × 第一解）且平移相差 > AMBIG_MIN_SEP_PX，
    则判定该图对**本质歧义**，返回 ok=False，而不是在两个解里瞎猜一个。
    宁可少给一个数，也不能给一个"看着自信的错数"。
    """
    ga, gb = _to_gray_u8(img_a), _to_gray_u8(img_b)
    if ambiguity_guard:
        ss = max(self_similarity_score(ga), self_similarity_score(gb))
        if ss > SELF_SIM_THRESH:
            return PairRegistration(
                False, method=detector,
                reason="内容自相似度 %.3f > %.2f（周期性结构），位移原理上不可辨识，拒绝给数" % (
                    ss, SELF_SIM_THRESH))
    det = _make_detector(detector, nfeatures)
    ka, da = det.detectAndCompute(ga, None)
    kb, db = det.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < min_inliers or len(kb) < min_inliers:
        return PairRegistration(False, method=detector, reason="特征点不足")

    norm = cv2.NORM_HAMMING if detector == "orb" else cv2.NORM_L2
    bf = cv2.BFMatcher(norm)
    knn = bf.knnMatch(da, db, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < min_inliers:
        return PairRegistration(False, n_matches=len(good), method=detector,
                                reason="Lowe ratio 后匹配不足")

    pa = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pb = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(
        pa, pb, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh,
        maxIters=5000, confidence=0.999,
    )
    if M is None or inl is None:
        return PairRegistration(False, n_matches=len(good), method=detector,
                                reason="RANSAC 未收敛")
    n_in = int(inl.sum())
    reg = PairRegistration(True, n_inliers=n_in, n_matches=len(good), method=detector)
    if n_in < min_inliers or reg.inlier_ratio < MIN_INLIER_RATIO:
        return PairRegistration(False, n_inliers=n_in, n_matches=len(good),
                                method=detector, reason="内点数/内点率不足")

    # M 把 a 的点映到 b 的点：p_b = M[:, :2] @ p_a + M[:, 2]
    a11, a21 = M[0, 0], M[1, 0]
    scale = float(math.hypot(a11, a21))
    rot = float(math.degrees(math.atan2(a21, a11)))
    if not (1.0 / MAX_SCALE_JUMP) <= scale <= MAX_SCALE_JUMP:
        return PairRegistration(False, n_inliers=n_in, n_matches=len(good),
                                method=detector, reason="尺度跳变 %.3f 超界" % scale)

    if ambiguity_guard:
        keep = (inl.ravel() == 0)
        if int(keep.sum()) >= min_inliers:
            M2, inl2 = cv2.estimateAffinePartial2D(
                pa[keep], pb[keep], method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thresh, maxIters=5000, confidence=0.999)
            if M2 is not None and inl2 is not None:
                n2 = int(inl2.sum())
                sep = float(math.hypot(M2[0, 2] - M[0, 2], M2[1, 2] - M[1, 2]))
                if n2 >= AMBIG_INLIER_RATIO * n_in and sep > AMBIG_MIN_SEP_PX:
                    return PairRegistration(
                        False, n_inliers=n_in, n_matches=len(good), method=detector,
                        reason=("周期性歧义：第二解内点 %d/%d、平移相差 %.1f px —— "
                                "疑似整周期混叠，拒绝给数" % (n2, n_in, sep)))

    # 内容位移取图像中心的位移：同一地物在 b 中相对 a 的移动量。
    h, w = ga.shape[:2]
    c = np.array([w / 2.0, h / 2.0])
    c_b = M[:, :2] @ c + M[:, 2]
    reg.dx = float(c_b[0] - c[0])
    reg.dy = float(c_b[1] - c[1])
    reg.scale, reg.rot_deg = scale, rot
    reg.M = M.astype(np.float64)
    return reg


def _invert_similarity(M: np.ndarray) -> np.ndarray:
    """求 2x3 相似变换的逆。M: p' = A p + t  →  M⁻¹: p = A⁻¹ p' − A⁻¹ t。"""
    A = np.asarray(M, dtype=np.float64)[:, :2]
    t = np.asarray(M, dtype=np.float64)[:, 2]
    Ai = np.linalg.inv(A)
    out = np.empty((2, 3), dtype=np.float64)
    out[:, :2] = Ai
    out[:, 2] = -Ai @ t
    return out


def _compose(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """复合两个 2x3 变换：先 M2 再 M1，即 x ↦ M1(M2(x))。"""
    A1, t1 = M1[:, :2], M1[:, 2]
    A2, t2 = M2[:, :2], M2[:, 2]
    out = np.empty((2, 3), dtype=np.float64)
    out[:, :2] = A1 @ A2
    out[:, 2] = A1 @ t2 + t1
    return out


# =============================================================================
# 2. 由帧序列恢复窗口轨迹（替换光流积分）
# =============================================================================

MIN_STEP_PX = 1.0   # 存活门：逐步位移中位数低于此视为近静止（对齐既有 flow 侧的幅值门控）


def recover_path_by_registration(
    frames: Sequence[np.ndarray],
    *,
    detector: str = "orb",
    max_failed_ratio: float = 0.10,
    min_step_px: float = MIN_STEP_PX,
    **kw,
) -> PathRecovery:
    """逐帧配准并**复合变换**，得到 (T,2) 的 (cy,cx) **相机**窗口轨迹。

    符号：配准给出的是**内容**位移（地物在画面里往哪跑），相机位移 = −内容位移。
    与 trajectory_metrics 的约定一致（该模块内 `gt_content_ang=(seg_ang+180)%360`）。

    **为什么是复合而不是相加**（2026-07-30 修）：逐帧位移是在**各自帧的坐标系**里量的。
    一旦序列存在面内旋转或尺度漂移——而这正是我们要测的生成视频漂移——把 dy/dx 直接
    累加就错了：后面帧的"一个像素"和首帧的"一个像素"既不同向也不同长。
    正确做法是维护累计变换 `H_t: frame_t → frame_0`（`H_t = H_{t-1} ∘ M_t⁻¹`），
    把每帧中心映回首帧坐标系再取位移。纯平移序列下两者等价，有旋转/缩放时才见分晓
    （自检 [10] 用一段带旋转的序列固定了这一差异）。

    失败步不插值、不填 0：位置保持上一帧，但记入 `failed_indices` 供下游排除；
    整段失败步占比超过 max_failed_ratio 即判 invalid。

    **`valid=True` 不等于"这条路径每一步都可信"**：只要 `n_failed_steps > 0`，
    路径里就有"位置被冻结"的假步，任何**路径积分量**（路径长、逐步长、端点误差、
    漂移曲线、闭环）都被污染。下游必须查 `failed_indices`——
    `absolute_step_error` / `rollout_drift_curve` / `loop_closure_error` 都接受该参数并
    在受污染时返回 None + 原因。（2026-07-30 审查实测：1 帧损坏即可产出
    path_len_rel_err=0.321、漂移 31px/帧 这类"看着合理"的数，全被记到模型头上。）
    """
    frames = list(frames)
    T = len(frames)
    if T < 2:
        return PathRecovery(None, reason="帧数 < 2")

    h, w = _to_gray_u8(frames[0]).shape[:2]
    c0 = np.array([w / 2.0, h / 2.0], dtype=np.float64)

    regs: List[PairRegistration] = []
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)  # frame_t → frame_0
    path = [(0.0, 0.0)]
    scales = [1.0]
    n_failed = 0
    failed: List[int] = []
    for t in range(1, T):
        r = register_pair(frames[t - 1], frames[t], detector=detector, **kw)
        regs.append(r)
        if r.ok and r.M is not None:
            H = _compose(H, _invert_similarity(r.M))   # H_t = H_{t-1} ∘ M_t⁻¹
        else:
            n_failed += 1
            failed.append(t)                            # 保持上一帧的 H（不假装静止）
        # frame_t 的中心映回首帧坐标系。注意：H 已是**逆**变换（frame_t→frame_0），
        # 内容/相机的反向已经被它吃掉，所以 d 直接就是**相机**位移，不可再取负
        # （2026-07-30：这里先写成再取负一次，导致整条路径 180° 反向，被自检 [9] 抓到）。
        c_in0 = H[:, :2] @ c0 + H[:, 2]
        d = c_in0 - c0
        path.append((float(d[1]), float(d[0])))         # (cy, cx) = (dy, dx)
        scales.append(float(math.hypot(H[0, 0], H[1, 0])))

    out = PathRecovery(np.asarray(path, dtype=np.float64), per_step=regs,
                       n_failed_steps=n_failed, failed_indices=failed)
    # H_T 把末帧映回首帧；末帧相对首帧的尺度是它的逆，故取倒数。
    s_end = float(math.hypot(H[0, 0], H[1, 0]))
    out.scale_drift = float(1.0 / s_end - 1.0) if s_end > 1e-9 else None

    # 旋转漂移**必须逐步解缠累加**，不能从复合矩阵读（2026-07-30 审查发现）：
    # atan2 的值域是 (−180,180]，累计旋转一旦超过半圈就回绕。
    # 实测：7 帧 × 40°/帧（真值 240°）从复合矩阵读出 −119.89° —— 大漂移被读成小漂移，
    # 而 GC-3 的卖点恰恰是"零旋转的解析空值"，这种回绕是**朝着有利于模型的方向**说谎。
    # 33 帧的 rollout 只要 >5.5°/帧就会回绕。
    # 逐步旋转是小增量，直接求和即为解缠累加（每步 |rot| 远小于 180°，由 MAX_SCALE_JUMP
    # 与内点门共同保证；单步若真超过半圈，该步会先因内点不足被判失败）。
    step_rots = [r.rot_deg for r in regs if r.ok and r.rot_deg is not None]
    out.rot_drift_deg = float(-sum(step_rots)) if step_rots else 0.0
    # 复合矩阵的读数只留作模 180 的交叉校验，不作为对外指标
    out.rot_drift_wrapped_deg = float(-math.degrees(math.atan2(H[1, 0], H[0, 0])))
    out.scale_curve = np.asarray([1.0 / s if s > 1e-9 else float("nan") for s in scales],
                                 dtype=np.float64)
    frac = n_failed / max(1, T - 1)
    if frac > max_failed_ratio:
        out.valid = False
        out.reason = "配准失败步占比 %.2f > %.2f（低重叠或内容崩坏）" % (frac, max_failed_ratio)
        return out

    # 存活门（2026-07-30 补）：**完全相同的两帧配准得完美**——内点满、位移零、失败步为 0，
    # 于是 persistence（输出冻结帧）这类退化输出会被判成 valid 而混进表里。
    # flow 侧靠 `weight_total < 1.0` 拦，配准侧必须显式拦：用逐步位移中位数。
    steps = np.hypot(np.diff(out.path[:, 0]), np.diff(out.path[:, 1]))
    med_step = float(np.median(steps)) if len(steps) else 0.0
    out.median_step_px = med_step
    # 「活着」= 平移 or 旋转 or 尺度 任一显著。只看平移会把**原地旋转/缩放**误判成静止
    # （2026-07-30：实测 40°/帧 的纯旋转序列被判"近静止"，而配准其实每步都完美恢复 40.0°）。
    moving = (med_step >= min_step_px
              or abs(out.rot_drift_deg or 0.0) >= 2.0
              or abs(out.scale_drift or 0.0) >= 0.05)
    if not moving:
        out.valid = False
        out.reason = ("近静止：逐步位移中位数 %.3f px < %.2f px，且无显著旋转/尺度变化"
                      "（persistence 类退化输出）" % (med_step, min_step_px))
        return out

    out.valid = True
    return out


# =============================================================================
# 3. 绝对定位：生成帧能否在源 mosaic 中被定位
# =============================================================================

def localize_in_mosaic(
    frame: np.ndarray,
    mosaic: np.ndarray,
    *,
    detector: str = "orb",
    nfeatures: int = 8000,
    min_inliers: int = MIN_INLIERS,
) -> Optional[Tuple[float, float, float]]:
    """把一帧定位到 mosaic 中，返回 (cy, cx, scale)；无法定位返回 None。

    这是光流路线原理上给不出的量：它是**绝对**的（相对源影像），因此可以回答
    "模型是不是真的走了指令要求的那么远"，而不只是"方向对不对"。
    mosaic 是**私有参照**（不进入模型输入），所以它不构成泄漏。
    """
    gf, gm = _to_gray_u8(frame), _to_gray_u8(mosaic)
    det = _make_detector(detector, nfeatures)
    kf, df = det.detectAndCompute(gf, None)
    km, dm = det.detectAndCompute(gm, None)
    if df is None or dm is None or len(kf) < min_inliers or len(km) < min_inliers:
        return None
    norm = cv2.NORM_HAMMING if detector == "orb" else cv2.NORM_L2
    knn = cv2.BFMatcher(norm).knnMatch(df, dm, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.80 * n.distance]
    if len(good) < min_inliers:
        return None
    pf = np.float32([kf[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pm = np.float32([km[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inl = cv2.estimateAffinePartial2D(pf, pm, method=cv2.RANSAC,
                                         ransacReprojThreshold=3.0, maxIters=5000,
                                         confidence=0.999)
    if M is None or inl is None or int(inl.sum()) < min_inliers:
        return None
    h, w = gf.shape[:2]
    c = np.array([w / 2.0, h / 2.0])
    c_m = M[:, :2] @ c + M[:, 2]
    scale = float(math.hypot(M[0, 0], M[1, 0]))
    return (float(c_m[1]), float(c_m[0]), scale)   # (cy, cx, scale)


# =============================================================================
# 4. Track G 指标：绝对量 + 漂移 + 闭环
# =============================================================================

def _contaminated(failed_indices: Optional[Sequence[int]]) -> bool:
    return bool(failed_indices)


_CONTAM_REASON = ("路径含配准失败步（位置被冻结），路径积分量不可信 —— "
                  "返回 None 而非把仪器故障记到模型头上")


def absolute_step_error(
    rec_path: np.ndarray,
    gt_path: np.ndarray,
    *,
    gsd_m_per_px: Optional[float] = None,
    failed_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Optional[float]]:
    """**不做弧长归一化**的逐步位移误差 —— 修掉 Track B 的尺度盲。

    Track B 的 path_shape_error / waypoint_error 都先做单位弧长归一化，
    因此"方向全对但走了 2 倍远"得分为 0 误差。本函数保留绝对尺度：
    走 2 倍远 → step_len_rel_err ≈ 1.0。
    """
    rec = np.asarray(rec_path, dtype=np.float64)
    gt = np.asarray(gt_path, dtype=np.float64)
    if rec.shape != gt.shape or rec.ndim != 2 or rec.shape[1] != 2:
        raise ValueError("rec/gt 形状必须一致且为 (T,2)，收到 %s / %s" % (rec.shape, gt.shape))
    if _contaminated(failed_indices):
        return {"path_len_rel_err": None, "endpoint_abs_err_px": None,
                "endpoint_abs_err_m": None, "step_len_rel_err_mean": None,
                "reason": _CONTAM_REASON}
    d_rec = np.diff(rec, axis=0)
    d_gt = np.diff(gt, axis=0)
    L_rec = float(np.hypot(d_rec[:, 0], d_rec[:, 1]).sum())
    L_gt = float(np.hypot(d_gt[:, 0], d_gt[:, 1]).sum())
    if L_gt < 1e-9:
        return {"path_len_rel_err": None, "endpoint_abs_err_px": None,
                "endpoint_abs_err_m": None, "step_len_rel_err_mean": None}
    per_step_rel = []
    for k in range(len(d_gt)):
        g = math.hypot(d_gt[k, 0], d_gt[k, 1])
        if g < 1e-6:
            continue
        r = math.hypot(d_rec[k, 0], d_rec[k, 1])
        per_step_rel.append(abs(r - g) / g)
    ep = float(np.hypot(*(rec[-1] - gt[-1])))
    return {
        "path_len_rel_err": abs(L_rec - L_gt) / L_gt,
        "endpoint_abs_err_px": ep,
        "endpoint_abs_err_m": (ep * gsd_m_per_px) if gsd_m_per_px else None,
        "step_len_rel_err_mean": float(np.mean(per_step_rel)) if per_step_rel else None,
    }


def rollout_drift_curve(rec_path: np.ndarray, gt_path: np.ndarray, *,
                        failed_indices: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """逐帧绝对定位误差曲线 + 线性漂移率（px/帧）。

    RSWISE 只在定性图里承认"多步会漂"，从不给数字。这条曲线就是那个数字。
    """
    rec = np.asarray(rec_path, dtype=np.float64)
    gt = np.asarray(gt_path, dtype=np.float64)
    if rec.shape != gt.shape:
        raise ValueError("形状不一致：%s vs %s" % (rec.shape, gt.shape))
    if _contaminated(failed_indices):
        return {"drift_curve_px": None, "drift_rate_px_per_frame": None,
                "final_drift_px": None, "reason": _CONTAM_REASON}
    err = np.hypot(rec[:, 0] - gt[:, 0], rec[:, 1] - gt[:, 1])
    t = np.arange(len(err), dtype=np.float64)
    slope = float(np.polyfit(t, err, 1)[0]) if len(err) >= 3 else None
    return {"drift_curve_px": err.tolist(),
            "drift_rate_px_per_frame": slope,
            "final_drift_px": float(err[-1])}


def loop_closure_error(rec_path: np.ndarray, *,
                       failed_indices: Optional[Sequence[int]] = None
                       ) -> Dict[str, Optional[float]]:
    """闭环误差：指令要求回到起点时，末帧与首帧的残余距离（px 及占路径长比例）。

    这是"闭环 rollout"这一支的判据，也是纯 2D 平移数据上仍然成立的几何自洽检验。
    """
    rec = np.asarray(rec_path, dtype=np.float64)
    if rec.ndim != 2 or rec.shape[0] < 3:
        return {"loop_gap_px": None, "loop_gap_rel": None}
    if _contaminated(failed_indices):
        return {"loop_gap_px": None, "loop_gap_rel": None, "reason": _CONTAM_REASON}
    gap = float(np.hypot(*(rec[-1] - rec[0])))
    d = np.diff(rec, axis=0)
    L = float(np.hypot(d[:, 0], d[:, 1]).sum())
    return {"loop_gap_px": gap, "loop_gap_rel": (gap / L) if L > 1e-9 else None}


def camera_path_to_content_xy(camera_path: np.ndarray) -> np.ndarray:
    """(T,2) **(cy,cx) 相机**路径 → (T,2) **(x,y) 内容**路径，即 Track B 的 recovered_path 口径。

    两步换算，缺一不可：
      1. 内容 = −相机（内容流与相机运动反向）；
      2. 列序 (cy,cx) → (x,y)：x 取 −cx，y 取 −cy。

    这是本模块与 `trajectory_metrics.py` 之间**唯一**合法的接线方式。
    实测：跳过本函数直接喂相机路径，`segment_wise_angular_error` 会给出 90°/154° 这类
    看似合理实则完全错误的值（自检 [9] 用故障注入固定了这一点）。
    """
    p = np.asarray(camera_path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("camera_path 必须是 (T,2)，收到 %s" % (p.shape,))
    out = np.empty_like(p)
    out[:, 0] = -p[:, 1]      # x = -cx
    out[:, 1] = -p[:, 0]      # y = -cy
    return out


def content_xy_to_camera_path(content_xy: np.ndarray) -> np.ndarray:
    """`camera_path_to_content_xy` 的逆变换（自反）。"""
    p = np.asarray(content_xy, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("content_xy 必须是 (T,2)，收到 %s" % (p.shape,))
    out = np.empty_like(p)
    out[:, 0] = -p[:, 1]      # cy = -y
    out[:, 1] = -p[:, 0]      # cx = -x
    return out


def bearing_from_path(path: np.ndarray, i: int = 0, j: int = -1) -> Optional[float]:
    """路径两点间的**连续**方位角（度，east=0/north=90），与既有约定一致。

    这是连续方向的打分靶：不做任何 90° 分箱，因此量化地板 = 0。
    """
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2:
        return None
    dy = p[j, 0] - p[i, 0]
    dx = p[j, 1] - p[i, 1]
    if math.hypot(dy, dx) < 1e-6:
        return None
    return float(math.degrees(math.atan2(-dy, dx)) % 360.0)


def angular_diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """两个方位角的最小夹角（0-180）。任一为 None 则返回 None（不填 0）。"""
    if a is None or b is None:
        return None
    d = abs((a - b + 180.0) % 360.0 - 180.0)
    return float(d)


# =============================================================================
# 5. 自检（含故障注入）
# =============================================================================

def _synth_mosaic(h: int = 1600, w: int = 1600, seed: int = 0) -> np.ndarray:
    """生成富纹理 mosaic：多尺度斑块 + 高频噪声，保证特征点充足且非周期。"""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, size=(h // 16, w // 16, 3), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    mid = cv2.resize(rng.integers(0, 255, size=(h // 4, w // 4, 3), dtype=np.uint8),
                     (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img = 0.55 * img + 0.30 * mid + 0.15 * rng.integers(0, 255, size=(h, w, 3)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _crop(mosaic: np.ndarray, cy: float, cx: float, wh: Tuple[int, int]) -> np.ndarray:
    """以 (cy,cx) 为中心裁 (h,w)；越界抛错（不做静默 wrap，与既有 builder 同规范）。"""
    h, w = wh
    y0, x0 = int(round(cy - h / 2)), int(round(cx - w / 2))
    if y0 < 0 or x0 < 0 or y0 + h > mosaic.shape[0] or x0 + w > mosaic.shape[1]:
        raise ValueError("裁剪越界：中心 (%.1f,%.1f) 窗口 %s" % (cy, cx, wh))
    return mosaic[y0:y0 + h, x0:x0 + w].copy()


def _polyline(start: Tuple[float, float], segs: Sequence[Tuple[float, float]],
              steps_per_seg: int) -> np.ndarray:
    """按 (bearing_deg, length_px) 段列表生成 (T,2) 相机路径 (cy,cx)。"""
    pts = [np.array(start, dtype=np.float64)]
    for bearing, length in segs:
        th = math.radians(bearing)
        dy, dx = -math.sin(th) * length, math.cos(th) * length   # north=90 → dy<0
        p0 = pts[-1]
        for s in range(1, steps_per_seg + 1):
            f = s / steps_per_seg
            pts.append(p0 + np.array([dy * f, dx * f]))
    return np.asarray(pts, dtype=np.float64)


def _self_test(allow_skip: bool = False) -> int:  # noqa: C901
    fails: List[str] = []
    skip_code = 0

    def check(name: str, cond: bool, info: str = "") -> None:
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, (" — " + info) if info else ""))
        if not cond:
            fails.append(name)

    WH = (128, 222)   # 与既有 pose_window_size 一致
    mos = _synth_mosaic(1600, 1600, seed=7)

    print("\n[1] 成对配准：已知平移必须被精确恢复")
    a = _crop(mos, 800, 800, WH)
    for dy, dx in [(0, 30), (-25, 0), (18, -40)]:
        b = _crop(mos, 800 + dy, 800 + dx, WH)
        r = register_pair(a, b)
        # 相机移动 (dy,dx) → 内容移动 (-dy,-dx)
        ok = r.ok and abs(r.dy - (-dy)) < 1.0 and abs(r.dx - (-dx)) < 1.0
        check("平移 (%+d,%+d) 恢复误差 <1px" % (dy, dx), ok,
              "ok=%s dy=%s dx=%s inl=%d" % (r.ok, r.dy, r.dx, r.n_inliers))
    check("平移对的 scale≈1", abs(register_pair(a, _crop(mos, 800, 830, WH)).scale - 1.0) < 0.02)

    print("\n[2] 尺度：Track B 测不出的那一维")
    z = cv2.resize(_crop(mos, 800, 800, (int(WH[0] * 1.25), int(WH[1] * 1.25))),
                   (WH[1], WH[0]), interpolation=cv2.INTER_AREA)
    rz = register_pair(a, z)
    check("放大 1.25x 被检出（scale 显著 <1 或 >1）", rz.ok and abs(rz.scale - 1.0) > 0.10,
          "scale=%s" % (None if not rz.ok else round(rz.scale, 3)))

    print("\n[3] 低重叠：光流会崩，配准应活下来")
    # 帧间重叠 = 1 - dx/222。dx=200 → 重叠 ~10%
    a2 = _crop(mos, 800, 700, WH)
    for dx, tag in [(20, "~91%"), (110, "~50%"), (178, "~20%"), (200, "~10%")]:
        b2 = _crop(mos, 800, 700 + dx, WH)
        rr = register_pair(a2, b2)
        flow = cv2.calcOpticalFlowFarneback(_to_gray_u8(a2), _to_gray_u8(b2), None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        fx = float(np.average(flow[..., 0], weights=np.hypot(*[flow[..., i] for i in (0, 1)]) + 1e-9))
        reg_err = abs(rr.dx - (-dx)) if rr.ok else float("inf")
        flow_err = abs(fx - (-dx))
        print("     重叠 %-5s: 配准误差 %7.2f px | Farneback 误差 %7.2f px" % (tag, reg_err, flow_err))
        if dx >= 178:
            check("重叠 %s 下配准仍成功且误差 <3px" % tag, reg_err < 3.0, "err=%.2f" % reg_err)
            check("重叠 %s 下 Farneback 误差 > 配准误差×3（证明估计器是瓶颈）" % tag,
                  flow_err > max(3.0 * reg_err, 5.0), "flow=%.1f reg=%.2f" % (flow_err, reg_err))

    print("\n[4] 整段轨迹恢复：多段折线，低重叠")
    gt = _polyline((800, 500), [(0.0, 540.0), (58.0, 300.0)], steps_per_seg=4)  # bearing 0° 后 58°
    frames = [_crop(mos, p[0], p[1], WH) for p in gt]
    rec = recover_path_by_registration(frames)
    check("valid=True", rec.valid, rec.reason)
    gt_anchored = gt - gt[0]
    max_err = float(np.max(np.hypot(rec.path[:, 0] - gt_anchored[:, 0],
                                    rec.path[:, 1] - gt_anchored[:, 1]))) if rec.valid else float("inf")
    check("逐帧绝对定位误差 <3px", max_err < 3.0, "max=%.2f px" % max_err)
    b_rec, b_gt = bearing_from_path(rec.path), bearing_from_path(gt_anchored)
    check("净方位角误差 <1.5°（连续靶，无 90° 分箱）",
          (angular_diff(b_rec, b_gt) or 99) < 1.5,
          "rec=%.2f gt=%.2f" % (b_rec, b_gt))
    check("尺度漂移 ≈0（纯平移序列）", abs(rec.scale_drift) < 0.05, "%.4f" % rec.scale_drift)

    print("\n[5] 绝对量指标：修掉 Track B 的尺度盲")
    ase = absolute_step_error(rec.path, gt_anchored, gsd_m_per_px=0.3)
    check("路径长相对误差 <2%", ase["path_len_rel_err"] < 0.02, "%.4f" % ase["path_len_rel_err"])
    check("端点误差换算成米已给出", ase["endpoint_abs_err_m"] is not None,
          "%.3f m" % ase["endpoint_abs_err_m"])
    over = gt_anchored * 2.0     # 方向全对、距离 2 倍
    ase2 = absolute_step_error(over, gt_anchored)
    check("走 2 倍远 → 路径长相对误差 ≈1.0（Track B 归一化后为 0）",
          abs(ase2["path_len_rel_err"] - 1.0) < 0.02, "%.3f" % ase2["path_len_rel_err"])
    check("走 2 倍远 → 逐步长相对误差 ≈1.0", abs(ase2["step_len_rel_err_mean"] - 1.0) < 0.02,
          "%.3f" % ase2["step_len_rel_err_mean"])

    print("\n[6] 漂移曲线与闭环")
    drift = rollout_drift_curve(rec.path, gt_anchored)
    check("漂移率已给出且 ≈0", drift["drift_rate_px_per_frame"] is not None
          and abs(drift["drift_rate_px_per_frame"]) < 0.5,
          "%.4f px/frame" % drift["drift_rate_px_per_frame"])
    synth_drift = gt_anchored + np.stack([np.linspace(0, 20, len(gt_anchored)),
                                          np.zeros(len(gt_anchored))], axis=1)
    d2 = rollout_drift_curve(synth_drift, gt_anchored)
    check("注入线性漂移 20px/全程 → 漂移率被检出为正",
          d2["drift_rate_px_per_frame"] > 1.0, "%.3f px/frame" % d2["drift_rate_px_per_frame"])
    loop = _polyline((800, 700), [(0.0, 200.0), (90.0, 200.0), (180.0, 200.0), (270.0, 200.0)], 3)
    lc = loop_closure_error(loop - loop[0])
    check("闭合方形轨迹 loop_gap≈0", lc["loop_gap_px"] < 1e-6, "%.2e px" % lc["loop_gap_px"])
    open_path = _polyline((800, 700), [(0.0, 200.0), (90.0, 200.0)], 3)
    lc2 = loop_closure_error(open_path - open_path[0])
    check("开放轨迹 loop_gap 显著 >0", lc2["loop_gap_px"] > 50, "%.1f px" % lc2["loop_gap_px"])

    print("\n[7] fail-loud：退化输入必须返回 None / False，不得填 0")
    flat = np.full((WH[0], WH[1], 3), 128, np.uint8)
    check("纯色对 → ok=False", register_pair(flat, flat).ok is False)
    r_flat = register_pair(flat, flat)
    check("失败时 dy/dx/scale 全为 None（不是 0）",
          r_flat.dy is None and r_flat.dx is None and r_flat.scale is None)
    rec_flat = recover_path_by_registration([flat] * 5)
    check("全纯色序列 → valid=False 且给出原因", (not rec_flat.valid) and bool(rec_flat.reason),
          rec_flat.reason)
    check("bearing_from_path 对静止路径返回 None",
          bearing_from_path(np.zeros((5, 2))) is None)
    frozen = [_crop(mos, 800, 800, WH)] * 6          # 富纹理但完全冻结：配准会"完美成功"
    rec_frozen = recover_path_by_registration(frozen)
    check("冻结帧序列（persistence）必须被存活门拦住，而不是 valid",
          (not rec_frozen.valid) and "近静止" in rec_frozen.reason, rec_frozen.reason)
    check("冻结序列的失败步为 0（证明拦它的确实是存活门而非配准失败）",
          rec_frozen.n_failed_steps == 0, "failed=%d" % rec_frozen.n_failed_steps)
    check("存活门可关闭（min_step_px=0 时不再拦）",
          recover_path_by_registration(frozen, min_step_px=0.0).valid is True)
    check("angular_diff 任一为 None → None", angular_diff(None, 10.0) is None)
    try:
        absolute_step_error(np.zeros((5, 2)), np.zeros((4, 2)))
        check("形状不一致必须抛错", False)
    except ValueError:
        check("形状不一致必须抛错", True)

    print("\n[8] 故障注入：把估计器改坏，上面的断言必须变红")
    真 = register_pair
    try:
        def broken(img_a, img_b, **kw):          # 恒返回零位移的"成功"
            return PairRegistration(True, dy=0.0, dx=0.0, scale=1.0, rot_deg=0.0,
                                    n_inliers=99, n_matches=99, method="broken")
        globals()["register_pair"] = broken
        bad = recover_path_by_registration(frames)
        bad_err = float(np.max(np.hypot(bad.path[:, 0] - gt_anchored[:, 0],
                                        bad.path[:, 1] - gt_anchored[:, 1])))
        check("坏估计器下 §4 的 <3px 断言确实会失败（断言有咬合力）", bad_err >= 3.0,
              "max=%.1f px" % bad_err)
        check("坏估计器下净方位角变得无意义（返回 None）",
              bearing_from_path(bad.path) is None)
    finally:
        globals()["register_pair"] = 真

    print("\n[12] 周期性纹理：混叠解必须被拒绝，而不是以高置信度给错数")

    def _lattice(h=900, w=900, period=48, seed=1):
        """无噪声周期栅格：农田垄行 / 停车场 / 集装箱堆场 / 光伏阵列的抽象。"""
        yy, xx = np.mgrid[0:h, 0:w]
        g = (((yy % period) < period // 2) ^ ((xx % period) < period // 2)).astype(np.uint8) * 200 + 20
        return np.repeat(g[:, :, None], 3, axis=2)

    lat = _lattice()
    n_wrong_guard = n_ok_guard = n_rej = 0
    n_wrong_noguard = 0
    for dx in [111, 178, 200]:                      # 50% / 20% / 10% 重叠
        for det in ("orb", "sift"):
            a = _crop(lat, 450, 300, WH)
            b = _crop(lat, 450, 300 + dx, WH)
            rg = register_pair(a, b, detector=det)                      # 带守卫
            rn = register_pair(a, b, detector=det, ambiguity_guard=False)  # 不带守卫
            if rg.ok:
                (n_wrong_guard if abs(rg.dx - (-dx)) > 5 else n_ok_guard).__int__()
                if abs(rg.dx - (-dx)) > 5:
                    n_wrong_guard += 1
                else:
                    n_ok_guard += 1
            else:
                n_rej += 1
            if rn.ok and abs(rn.dx - (-dx)) > 5:
                n_wrong_noguard += 1
    print("     守卫开：正确 %d / 拒绝 %d / **错误 %d**；守卫关：错误 %d"
          % (n_ok_guard, n_rej, n_wrong_guard, n_wrong_noguard))
    check("周期栅格上：开守卫后不再输出错误解", n_wrong_guard == 0,
          "仍有 %d 个错误解" % n_wrong_guard)
    check("故障注入：关掉守卫确实会产生高置信度错误解（证明守卫有用）",
          n_wrong_noguard > 0, "关守卫时错误 %d 个" % n_wrong_noguard)
    check("守卫不误伤：富纹理平移仍然成功",
          register_pair(_crop(mos, 800, 700, WH), _crop(mos, 800, 878, WH)).ok is True)

    print("\n[11] 绝对定位：把一帧定位回源 mosaic（C2 论点的支撑原语）")
    for cy, cx in [(800, 800), (400, 1100), (1200, 300)]:
        patch = _crop(mos, cy, cx, WH)
        loc = localize_in_mosaic(patch, mos)
        ok = loc is not None and abs(loc[0] - cy) < 2.0 and abs(loc[1] - cx) < 2.0
        check("定位 (%d,%d) 误差 <2px" % (cy, cx), ok,
              "got=%s" % (None if loc is None else str((round(loc[0], 2), round(loc[1], 2)))))
    check("定位同时给出尺度≈1", abs(localize_in_mosaic(_crop(mos, 800, 800, WH), mos)[2] - 1.0) < 0.05)
    zoomed = cv2.resize(_crop(mos, 700, 900, (int(WH[0] * 1.5), int(WH[1] * 1.5))),
                        (WH[1], WH[0]), interpolation=cv2.INTER_AREA)
    lz = localize_in_mosaic(zoomed, mos)
    # 注意语义：zoomed 是「裁 1.5 倍大的区域再缩回窗口尺寸」= 视场放宽(zoom OUT)，
    # 因此 frame→mosaic 的尺度应 ≈1.5（而不是 <1）。register_pair 是 a→b 方向，
    # 同一情形下给 ≈1/1.25（见 [2]）—— 两个函数方向相反，别把它们的尺度混为一谈。
    check("视场放宽 1.5x 的帧仍能定位，且尺度被检出 ≈1.5",
          lz is not None and abs(lz[0] - 700) < 6 and abs(lz[1] - 900) < 6
          and abs(lz[2] - 1.5) < 0.08,
          "got=%s" % (None if lz is None else str(tuple(round(v, 2) for v in lz))))
    check("纯色块无法定位 → None（不得返回 (0,0) 之类的假答案）",
          localize_in_mosaic(np.full((WH[0], WH[1], 3), 128, np.uint8), mos) is None)
    other = _synth_mosaic(800, 800, seed=99)     # 与 mos 无关的另一张图
    lo = localize_in_mosaic(np.full((WH[0], WH[1], 3), 200, np.uint8), other)
    check("异源纯色块也返回 None", lo is None)

    print("\n[10] 旋转/缩放序列：必须复合变换，逐帧相加会错")
    # 构造：相机向东走，同时画面每帧转 2°、整体缩放 1.02（模拟生成视频的漂移）
    THETA_PER_FRAME, S = 2.0, 1.02
    cam_gt = [(800.0, 600.0 + 60.0 * t) for t in range(7)]
    rot_frames = []
    for t, (cyt, cxt) in enumerate(cam_gt):
        th = math.radians(THETA_PER_FRAME * t)
        s_t = S ** t
        A = s_t * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        tvec = np.array([WH[1] / 2.0, WH[0] / 2.0]) - A @ np.array([cxt, cyt])
        M = np.hstack([A, tvec.reshape(2, 1)])
        rot_frames.append(cv2.warpAffine(mos, M, (WH[1], WH[0]), flags=cv2.INTER_LINEAR))
    rrot = recover_path_by_registration(rot_frames)
    check("旋转序列：valid=True", rrot.valid, rrot.reason)
    gt_rot = np.asarray(cam_gt) - np.asarray(cam_gt[0])
    comp_err = float(np.max(np.hypot(rrot.path[:, 0] - gt_rot[:, 0],
                                     rrot.path[:, 1] - gt_rot[:, 1]))) if rrot.valid else 1e9
    check("复合变换：相机轨迹误差 <5px", comp_err < 5.0, "max=%.2f px" % comp_err)
    # 逐帧相加（旧写法）作为故障注入对照
    naive = [(0.0, 0.0)]
    cy_n = cx_n = 0.0
    for r in rrot.per_step:
        if r.ok:
            cy_n += -r.dy
            cx_n += -r.dx
        naive.append((cy_n, cx_n))
    naive = np.asarray(naive)
    naive_err = float(np.max(np.hypot(naive[:, 0] - gt_rot[:, 0], naive[:, 1] - gt_rot[:, 1])))
    check("故障注入：逐帧相加在旋转下显著更差（>3×复合）",
          naive_err > max(3.0 * comp_err, 5.0),
          "naive=%.1f px vs composed=%.2f px" % (naive_err, comp_err))
    check("尺度漂移被检出（每帧 ×1.02，6 帧 ≈ +12.6%%）",
          rrot.scale_drift is not None and abs(rrot.scale_drift - (S ** 6 - 1)) < 0.03,
          "scale_drift=%.4f 期望≈%.4f" % (rrot.scale_drift, S ** 6 - 1))
    check("旋转漂移被检出（≈12°，容差 2°）",
          rrot.rot_drift_deg is not None and abs(abs(rrot.rot_drift_deg) - 12.0) < 2.0,
          "rot_drift=%.2f°" % rrot.rot_drift_deg)
    # 大角度：复合矩阵读数在 ±180 回绕，解缠累加不回绕
    big = []
    for t in range(7):
        th = math.radians(40.0 * t)
        A = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        tv = np.array([WH[1] / 2.0, WH[0] / 2.0]) - A @ np.array([800.0, 800.0])
        big.append(cv2.warpAffine(mos, np.hstack([A, tv.reshape(2, 1)]), (WH[1], WH[0]),
                                  flags=cv2.INTER_LINEAR))
    rbig = recover_path_by_registration(big)
    check("大旋转 240°：解缠累加不回绕（容差 5°）",
          rbig.valid and abs(abs(rbig.rot_drift_deg) - 240.0) < 5.0,
          "unwrapped=%.2f°" % (rbig.rot_drift_deg if rbig.valid else float("nan")))
    check("故障注入：复合矩阵读数确实回绕（证明上条修复必要）",
          rbig.valid and abs(abs(rbig.rot_drift_wrapped_deg) - 240.0) > 60.0,
          "wrapped=%.2f°" % (rbig.rot_drift_wrapped_deg if rbig.valid else float("nan")))
    check("scale_curve 长度 = 帧数", rrot.scale_curve is not None and len(rrot.scale_curve) == 7)

    print("\n[9] 与既有 trajectory_metrics 的接口互操作（列序陷阱）")
    cam = rec.path
    xy = camera_path_to_content_xy(cam)
    check("转换是自反的（往返还原）",
          np.allclose(content_xy_to_camera_path(xy), cam, atol=1e-9))
    check("转换后列序确为 (x,y) 且方向取反",
          abs(xy[4, 0] - (-cam[4, 1])) < 1e-9 and abs(xy[4, 1] - (-cam[4, 0])) < 1e-9)
    _root, _skip_code = resolve_or_exit_code(allow_skip, "与真实 trajectory_metrics 的接口互操作 [9]")
    _TM = None
    if _root is not None:
        try:
            sys.path.insert(0, _root + "/lora_remotesense/new")
            import trajectory_metrics as _TM   # noqa: E402
        except Exception as e:
            print("  [FAIL] 找到工程根但导入 trajectory_metrics 失败：%r" % (e,))
            fails.append("导入 trajectory_metrics")
    if _TM is not None:
        wps = np.array([gt[0], gt[4], gt[8]], dtype=np.float64)   # (cy,cx) 相机结点
        seg_ang = [0.0, 58.0]
        knots = [0, 4, 8]
        good = _TM.segment_wise_angular_error(xy, wps, seg_ang, knot_frames=knots)
        check("正确接线：逐段角误差 <2°", (good["mean"] or 99) < 2.0,
              "per=%s mean=%.2f" % ([round(e, 2) for e in good["per_segment"]], good["mean"]))
        bad = _TM.segment_wise_angular_error(cam, wps, seg_ang, knot_frames=knots)
        check("故障注入：直接喂相机路径必须显著变差（>30°）", (bad["mean"] or 0) > 30.0,
              "mean=%.2f" % bad["mean"])
        gtp = _TM.gt_content_path(wps, T=len(xy))
        shp = _TM.path_shape_error(xy, gtp)
        check("正确接线：Fréchet 形状误差 <0.05", shp["frechet"] < 0.05,
              "frechet=%.4f dtw=%.4f" % (shp["frechet"], shp["dtw"]))
        we = _TM.waypoint_error(xy, gtp, knots)
        check("正确接线：waypoint_error <0.05", (we if we is not None else 9) < 0.05,
              "we=%.5f" % we)

    print("\n" + "=" * 66)
    if fails:
        print("失败 %d 项：%s" % (len(fails), fails))
        return 1
    if _skip_code:
        print("本地断言全通过，但跨模块断言被跳过 —— 退出码 %d（跳过不算通过）。" % _skip_code)
        return _skip_code
    print("全部通过。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="平面配准轨迹恢复与几何一致性度量")
    ap.add_argument("--self-test", action="store_true", help="跑已知答案自检（含故障注入）")
    ap.add_argument("--allow-skip", action="store_true",
                    help="没有既有工程时允许跳过跨模块断言并仍返回 0（默认返回 2）")
    a = ap.parse_args()
    if a.self_test:
        return _self_test(allow_skip=a.allow_skip)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
