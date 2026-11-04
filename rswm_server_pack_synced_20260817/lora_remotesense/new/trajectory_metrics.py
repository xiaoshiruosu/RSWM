#!/usr/bin/env python3
"""
RSWM Track B —— 全轨迹可控性指标 (full-trajectory-control metrics)
================================================================

动机
----
Track A (metrics.py) 只回答"净方向对不对"(estimate_motion_direction ->
compute_angular_error),把整段视频折叠成一个方向角。当条件里带 Plücker 轨迹
(text + Plücker,完整多段折线路径)时,我们需要问更细的问题:生成视频的
内容路径是不是**逐段**、**逐拐点**地跟着 GT 轨迹走。本模块通过对逐帧光流做
积分,恢复生成视频的 CONTENT 路径,再与 GT 轨迹做形状/分段/拐点比对。

v2 —— 服务器标定后的四项修正 (根因来自一次真实 gt_oracle 标定运行)
--------------------------------------------------------------------
标定发现:拿 **真值视频** 喂 v1 Track B,得到 net_direction_error=80.67 deg、
valid_pct=80%、turn_recall=0.25 —— 指标本身失效,不是模型失效。四个根因与修正:

1. **分辨率 (native-res flow)**:GT/生成视频是 128x222 原生窗口的 480x832 上采样,
   内容位移 ~34 px/frame,达到/超过 Farneback (winsize=15, levels=3) 的可靠量程,
   光流系统性低估并噪声化。把帧先降采样回原生窗口分辨率 (H5
   f['metadata'].attrs['pose_window_size']) 后位移 ~9 px/frame,GT 逐段误差落回
   ~1-10 deg;synthetic_translation_check 在小位移 (~6 px) 下 8 方向恢复误差 0.020 deg。
   => score_trajectory 新增 window_size 参数,**所有** Track B 光流一律在窗口分辨率
   上计算 (一次共享降采样,单次 _compute_flows);min_mag 默认值改为窗口分辨率口径
   (1.5 px/frame)。waypoints 仍留在 scene 像素:形状比较先做单位弧长归一化+起点
   锚定,天然尺度不变;逐步向量只要求 **自身一致** (全部来自同一分辨率的光流)。
2. **净方向 (endpoint-based)**:v1 逐字复用 Track A estimate_motion_direction
   (T//8 步长采样 + 幅度加权全局求和),对多段折线不稳定。v2 用
   net_direction_from_path:恢复路径首尾连线方向 vs GT content 净方向。
   [v2.1] **门控由 GT 命令轨迹决定**:polyline_net_ratio(waypoints) <
   min_net_ratio (默认 0.2) 时"净方向"这个问题本身无意义 -> None + gated,
   与恢复质量无关(v2 曾用恢复路径比例门控:GT 近闭合样本被错误放行、
   GT 明确样本被错误门控,见服务器样本 00349/00331)。恢复路径自身比例
   另存 recovered_net_displacement_ratio 仅诊断。旧口径保留为
   track_a_net_direction_error / net_direction_error_flow 仅供对照。
3. **拐点 (windowed turn detection)**:v1 比较相邻单步朝向,会被重复帧/光流噪声/
   跨多帧的缓转漏掉真实拐点。v2 对 [t-W..t) 与 [t..t+W) 的 **步向量求均值** (不是
   平均角度) 再取夹角,阈值 + NMS (±win 内只留局部最大)。
4. **结点 (real knots)**:v1 用 round(linspace(0,T-1,K+1)) 均匀结点,但真实结点
   [!] [v2.8.2 起注意] 以下关于"真实结点 [0,9,17] vs 均匀 [0,8,17]"的论述描述的是 **v2.8 之前**的 18/27/36 帧数据(段界有重复帧)。v2.8 起源头做了段界去重 → K=2 是 17 帧、真实结点 [0,8,16],**均匀 linspace 现在恰好正确**。函数本身两种数据都成立(_knots_core 从逐帧位置经验提取、跳零步),但别据此反推"均匀假设是错的"这个结论 —— 对新数据它不再成立。

   不同 (v5 每段独立生成 steps_per_segment 帧,段边界处有重复帧:2 段 18 帧的真实
   结点是 [0,9,17] 而非均匀的 [0,8,17];4 段的第 3 个内部结点是 27 而非 26)。v2 从
   逐帧真实位置 (v5 源 benchmark 的 'trajectory' 数据集,
   build_multi_direction_benchmark_v5.py:826-834) 或 poses 平移列
   (poses[:,:3,3],camera_trajectory.py:97-127) 提取 **真实结点** =
   真实路径朝向变化帧;均匀 linspace 仅作显式回退并在输出里用
   knots_source='uniform_fallback' 标注。
   注意:规范转换器 (build_dataset_from_benchmark.py:765-810) 目前 **没有** 把
   trajectory/full_scene 拷进最终 H5 —— 在它补上之前,调用方可用 poses 或均匀回退。

坐标与符号约定 (务必与 metrics.py / camera_trajectory.py 一致)
------------------------------------------------------------
* 存储的 waypoints 为 [K+1, 2] 像素坐标 (cy, cx) = 窗口中心 = **CAMERA 路径**
  (build_from_remotesense.py:305 "Waypoints are (N,2) pixel coords (cy, cx)")。
* segment_angles 为 [K] 的 **CAMERA 方向角**(度),east=0/north=90/west=180/
  south=270,由 atan2(-dy, dx)%360 得到 (build_dataset_from_benchmark.py:580-590,
  camera_trajectory.py:59-72 direction_angle_from_waypoints)。
* CONTENT 光流 = 减去相机运动 (窗口东移 -> 画面内容西移)。因此
  **CONTENT 路径 = -CAMERA 路径**,CONTENT 方向 = CAMERA 方向旋转 180 度。
* 本模块统一在 **图像像素帧** (x = 向右/东为正, y = 向下/南为正) 里表达所有
  2D 路径:
    - 恢复路径:对逐帧幅度加权平均光流 (fx, fy) 做累积和,fx=右正=东正,
      fy=下正=南正 (与 metrics.estimate_motion_direction 的光流取向一致)。
    - GT content 路径:content_x = -(cx - cx0),content_y = -(cy - cy0)。
  两条路径同帧、同朝向,只差绝对尺度 -> 比形状前各自做单位总弧长归一化。
* 角度换算沿用 metrics 的 atan2(-dy, dx):把 y-down 翻到数学 y-up,得到与
  camera 方向词同一套 [east=0,north=90,west=180,south=270] 的角度。

哪些指标适用于哪种条件 (comment per spec)
-----------------------------------------
* text + Plücker (给了完整轨迹): **全部 Track B 指标都成立且被命令(commanded)** —
  segment_wise_angular_error / turn_recall_and_timing / waypoint_error /
  path_shape_error 都在评估"是否忠实跟随所给折线轨迹"。
* text-only (只有一句方向词,没有折线): 只有 **net_direction_error 与
  flow_magnitude** 是"被命令的"、有意义的可控性指标;segment/turn/waypoint/path
  仍然可以计算并报告,但它们比对的 GT 折线并未作为条件给模型 ->
  在 score_trajectory 输出里用 `commanded=False`(text_only 分支)标注,
  当作诊断量而非达标量看待。

依赖
----
* 光流:cv2.calcOpticalFlowFarneback,参数与 metrics.estimate_motion_direction
  完全一致 (metrics.py:268-272)。cv2 不可用时,依赖光流的函数返回 None,
  纯几何函数 (DTW/Fréchet/分段角度) 仍可用。
* 复用 metrics.estimate_motion_direction / metrics.compute_angular_error。

引用: metrics.py:238-295 (光流 API + 角度折叠), camera_trajectory.py:59-127
(waypoints->方向/平移约定), build_from_remotesense.py:304-307 (H5 字段形状)。
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

# 复用 Track A 的光流方向估计与角度折叠 (同目录 metrics.py)。
from metrics import estimate_motion_direction, compute_angular_error

# Farneback 参数,逐字对齐 metrics.estimate_motion_direction (metrics.py:268-272)。
_FARNEBACK_KW = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
)
_EPS = 1e-8


# ============================================================================
# 内部工具:光流 -> 逐步向量 / 路径 / 幅度
# ============================================================================

def _to_gray_uint8(frame: np.ndarray):
    """把一帧 (H,W,3) 转成 uint8 灰度图 (cv2 需要)。cv2 缺失时抛 ImportError。"""
    import cv2
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def _compute_flows(frames: np.ndarray) -> Optional[List[np.ndarray]]:
    """逐帧 (step=1) 计算稠密光流,返回 T-1 个 (H,W,2) float 数组。

    回答:"每一步画面内容往哪挪、挪多少"(逐帧稠密位移场)。
    与 metrics.estimate_motion_direction 不同,这里 **不做帧间下采样**、
    也 **不折叠成单一角度**,保留逐步向量场以便后续积分成路径。
    cv2 不可用或帧数 < 2 时返回 None。
    """
    try:
        import cv2
    except ImportError:
        return None
    if len(frames) < 2:
        return None
    flows: List[np.ndarray] = []
    prev = _to_gray_uint8(frames[0])
    for i in range(1, len(frames)):
        cur = _to_gray_uint8(frames[i])
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, **_FARNEBACK_KW)
        flows.append(flow)
        prev = cur
    return flows


def _maybe_downsample(frames: np.ndarray, window_size) -> np.ndarray:
    """(v2) 把帧降采样回原生窗口分辨率 (win_h, win_w),供 **全部** 光流共享。

    动机 (服务器标定):480x832 的生成/GT 视频是 128x222 窗口的上采样,内容位移
    ~34 px/frame,超过 Farneback 可靠量程 -> gt_oracle 都拿不到分 (80.67 deg)。
    降回窗口分辨率后 ~9 px/frame,GT 逐段误差 ~1-10 deg。

    行为
    ----
    * window_size=None -> 原样返回 (v1 行为,向后兼容)。
    * 帧高宽均 <= 窗口尺寸 -> 原样返回 (只降不升,绝不上采样放大噪声)。
    * 否则用 cv2.INTER_AREA 逐帧 resize 到 (win_w, win_h),**只做这一次**,
      之后 flow_magnitude / recover_content_path 等共享同一批降采样帧与光流。
    * cv2 缺失 -> 原样返回;下游 _compute_flows 同样因缺 cv2 返回 None ->
      valid=False,与既有"缺 cv2"路径一致,不会在错误分辨率上静默打分。
    """
    if window_size is None:
        return frames
    win_h, win_w = int(window_size[0]), int(window_size[1])
    if win_h <= 0 or win_w <= 0:
        raise ValueError(f"window_size 必须为正: {window_size!r}")
    frames = np.asarray(frames)
    if frames.ndim != 4 or len(frames) == 0:
        return frames
    H, W = frames.shape[1], frames.shape[2]
    if H <= win_h and W <= win_w:
        return frames
    try:
        import cv2
    except ImportError:
        return frames  # 下游 _compute_flows 会因缺 cv2 返回 None -> valid=False
    out = np.stack(
        [cv2.resize(f, (win_w, win_h), interpolation=cv2.INTER_AREA) for f in frames],
        axis=0,
    )
    return out


def _step_vector(flow: np.ndarray) -> tuple:
    """把一帧稠密光流场压成一个"主内容位移向量" (vx, vy, weight)。

    幅度加权平均 (与 metrics.py:273-277 抑制静止区噪声的思路一致),但这里保留
    向量而非折叠成角度。weight = 幅度总和,可作可靠性门控。
    坐标:vx = 右/东正,vy = 下/南正 (光流原生像素约定)。
    """
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    w = float(mag.sum())
    if w < 1.0:  # 累计幅度过小 -> 该步几乎无运动,视为零位移 (对齐 metrics 门控)
        return 0.0, 0.0, w
    vx = float((fx * mag).sum() / w)
    vy = float((fy * mag).sum() / w)
    return vx, vy, w


def _path_from_flows(flows: List[np.ndarray]) -> np.ndarray:
    """把逐步主位移向量累积成 (T,2) 内容路径,path[0] = (0,0)。

    坐标 x=东正, y=南正 (图像帧)。这是"生成视频的内容轨迹长什么样"。
    """
    steps = np.array([_step_vector(f)[:2] for f in flows], dtype=np.float64)  # (T-1,2)
    T = len(flows) + 1
    path = np.zeros((T, 2), dtype=np.float64)
    if len(steps):
        path[1:] = np.cumsum(steps, axis=0)
    return path


def _mean_magnitude_from_flows(flows: List[np.ndarray]) -> float:
    """逐帧稠密光流幅度的均值 (px/frame)。回答:"画面到底动没动、动多快"。"""
    if not flows:
        return 0.0
    per_step = [float(np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2).mean()) for f in flows]
    return float(np.mean(per_step))


# ============================================================================
# 内部工具:纯几何 (numpy-only)
# ============================================================================

def _path_length(path: np.ndarray) -> float:
    """折线总弧长 (相邻点欧氏距离之和)。"""
    if len(path) < 2:
        return 0.0
    d = np.diff(path, axis=0)
    return float(np.sqrt((d * d).sum(axis=1)).sum())


def _length_normalize(path: np.ndarray) -> np.ndarray:
    """按总弧长归一化到单位长度 (保持起点在原点、朝向不变)。

    这样 DTW / Fréchet 只比较 **形状**,不受绝对像素尺度影响。
    对已归一化的路径再次调用是幂等的 (总长=1)。
    """
    path = np.asarray(path, dtype=np.float64)
    L = _path_length(path)
    if L < _EPS:
        return path.copy()
    return path / L


def _anchor_to_origin(path: np.ndarray) -> np.ndarray:
    """把路径平移到起点为原点 (path - path[0])。

    形状比对必须是 **平移不变** 的:recover_content_path 的输出是逐步位移的累积和,
    起点天生为 (0,0);gt_content_path 也显式减掉了 cx_0/cy_0。所以对流水线里的
    真实输入本函数是恒等操作 (no-op)。它存在是为了让 path_shape_error 对
    **任意** 外部传入的路径 (例如带全局偏移的合成路径 / 自测用例) 也只比形状,
    不把一个纯平移误判成形状错误。总弧长平移不变,故与 _length_normalize 可交换。
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) == 0:
        return path.copy()
    return path - path[0]


def _knot_distances(
    recovered_path: np.ndarray,
    gt_path: np.ndarray,
    gt_knot_frames: Sequence[int],
) -> List[tuple]:
    """两条 (各自单位弧长归一化后的) 路径在 GT 结点帧处的欧氏距离。

    返回 [(frame, dist), ...],按 frame 升序,只含同时落在两条路径帧范围内的结点。
    供 waypoint_error (取均值) 与 waypoint_endpoint_error (取末项) 共用,
    保证两者的归一化与越界规则 **完全一致**。
    """
    rec = _length_normalize(np.asarray(recovered_path, dtype=np.float64))
    gt = _length_normalize(np.asarray(gt_path, dtype=np.float64))
    Tr, Tg = len(rec), len(gt)
    out: List[tuple] = []
    for f in gt_knot_frames:
        fi = int(f)
        if 0 <= fi < Tr and 0 <= fi < Tg:
            d = rec[fi] - gt[fi]
            out.append((fi, float(math.hypot(d[0], d[1]))))
    out.sort(key=lambda x: x[0])
    return out


def _pairwise_dist(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """成对欧氏距离矩阵 (n,m),numpy 广播。"""
    diff = P[:, None, :] - Q[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def _vec_angle(dx: float, dy: float) -> float:
    """位移向量 (dx=东正, dy=南正) -> 方向角 (度, [0,360))。

    atan2(-dy, dx): 把 y-down 翻到数学 y-up,得到与 camera 方向词同一套约定
    (east=0/north=90/west=180/south=270),对齐 metrics.py:283。
    """
    return math.degrees(math.atan2(-dy, dx)) % 360.0


def _camera_net_angle(waypoints: np.ndarray) -> float:
    """从 waypoints (cy,cx) 起终点求 CAMERA 净方向角 (度)。

    等价于 camera_trajectory.direction_angle_from_waypoints (camera_trajectory.py:59)。
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    dy = wp[-1, 0] - wp[0, 0]  # cy (下正)
    dx = wp[-1, 1] - wp[0, 1]  # cx (右正)
    return math.degrees(math.atan2(-dy, dx)) % 360.0


def _harmonic_mean(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """调和平均 2ab/(a+b) (F1 用)。任一侧 None -> None;两者皆 0 -> 0.0 (不除零)。"""
    if a is None or b is None:
        return None
    s = float(a) + float(b)
    if s <= _EPS:
        return 0.0
    return float(2.0 * float(a) * float(b) / s)


def _knot_frames(num_waypoints: int, T: int) -> np.ndarray:
    """waypoints 在 T 帧插值下落在的整数帧索引 (K+1 个结点)。

    与 camera_trajectory.interpolate_pixel_positions 的 linspace 插值一致
    (camera_trajectory.py:90-91): 结点均匀分布在 [0, T-1]。
    """
    if num_waypoints < 2:
        return np.array([0], dtype=int)
    return np.round(np.linspace(0, T - 1, num_waypoints)).astype(int)


# ============================================================================
# 公开 (v2):真实结点 —— 从逐帧位置 / poses 提取,替代均匀 linspace 假设
# ============================================================================

def _knots_core(positions: np.ndarray, angle_tol_deg: float) -> tuple:
    """真实结点提取核心:逐帧 2D 位置 -> (knot_frames, per_frame_segment_id)。

    定义:结点 = 真实路径 **朝向变化** 的帧 (纯几何,精确,不碰光流)。
    步向量 d_i = pos[i+1] - pos[i];朝向只对 |d_i| > _EPS 的"有效步"定义
    (v5 逐段生成会在段边界写出 **重复帧** -> 零步,必须跳过而非当噪声)。
    对相邻有效步 (a, b):折叠角差 > angle_tol_deg -> 在 b 处标记一次变化。
    相邻 (在有效步序列里紧邻) 的变化标记聚成一簇 (linspace 插值的拐角会被
    一个"混合步"劈成两个相邻标记),每簇取变化最大的标记为结点帧
    (并列取更早者)。angle_tol_deg 吸收整数像素舍入抖动 (~9 px 步长的
    逐步朝向抖动 <= ~13 deg < 默认 20)。

    返回 (np.ndarray[int] 结点帧 [0,...,T-1], np.ndarray[int] (T,) 每帧段号)。
    T=0 -> (空, 空);T=1 -> ([0], [0])。全静止 -> ([0, T-1], 全 0)。
    """
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"positions 必须是 (T,2),got {pos.shape}")
    T = len(pos)
    if T == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if T == 1:
        return np.array([0], dtype=int), np.array([0], dtype=int)

    steps = np.diff(pos, axis=0)                       # (T-1,2) = (d0, d1) 逐步向量
    mags = np.hypot(steps[:, 0], steps[:, 1])
    defined = np.where(mags > _EPS)[0]                 # 有效步 (跳过重复帧的零步)

    # 变化标记: (有效步序列中的序号 rank, 步索引 b, 折叠角差)
    marks: List[tuple] = []
    for r in range(1, len(defined)):
        a, b = int(defined[r - 1]), int(defined[r])
        # positions 是 (cy,cx) 或任何一致的 2D 帧;朝向用 _vec_angle(d_col1, d_col0)
        # = atan2(-d_row, d_col),与 waypoints (cy,cx) 的 camera 方向词约定一致。
        ha = _vec_angle(float(steps[a, 1]), float(steps[a, 0]))
        hb = _vec_angle(float(steps[b, 1]), float(steps[b, 0]))
        ch = compute_angular_error(hb, ha)
        if ch > float(angle_tol_deg):
            marks.append((r, b, ch))

    # 聚簇: 有效步序列里紧邻 (rank 差 1) 的标记属于同一物理拐角
    knot_list: List[int] = [0]
    if marks:
        cluster = [marks[0]]
        clusters = []
        for m in marks[1:]:
            if m[0] - cluster[-1][0] == 1:
                cluster.append(m)
            else:
                clusters.append(cluster)
                cluster = [m]
        clusters.append(cluster)
        for cl in clusters:
            # 每簇取角变化最大的步索引 b 作为结点帧;并列取更早者。
            best = max(cl, key=lambda m: (m[2], -m[1]))
            knot_list.append(int(best[1]))
    knot_list.append(T - 1)
    knots = np.array(sorted(set(knot_list)), dtype=int)

    seg_id = np.zeros(T, dtype=int)
    for k in range(len(knots) - 1):
        seg_id[knots[k]:knots[k + 1]] = k
    seg_id[T - 1] = max(0, len(knots) - 2)
    return knots, seg_id


def knots_from_positions(frame_positions, angle_tol_deg: float = 20.0) -> tuple:
    """(v2) 从 **真实逐帧窗口中心** 提取结点帧 (回答:每段真正始于哪一帧)。

    数学定义
    --------
    见 _knots_core:结点 = 真实路径朝向变化的帧,零步 (重复帧) 跳过,
    相邻变化聚簇后每簇一个结点;首尾帧 0 / T-1 恒为结点。

    输入
    ----
    frame_positions: (T,2) 逐帧窗口中心 (cy, cx),scene 像素 —— 即 v5 源
        benchmark 的 'trajectory' 数据集 (build_multi_direction_benchmark_v5.py:
        579 逐帧 append (cy,cx),826-834 写入 H5)。
    angle_tol_deg: 判定"朝向变了"的折叠角阈值 (度)。默认 20:吸收整数像素
        舍入抖动 (~9px 步长时逐步朝向抖动 <= ~13 deg),又远小于 benchmark
        相邻段的最小夹角 (45 deg)。

    输出 / 单位
    -----------
    (knot_frames, per_frame_segment_id):
      knot_frames: np.ndarray[int],K+1 个升序帧索引,含 0 与 T-1。单位 = 帧。
      per_frame_segment_id: np.ndarray[int] (T,),每帧所属段号 (0..K-1),
        帧 t 属于段 k 当且仅当 knot_frames[k] <= t < knot_frames[k+1]
        (末帧归最后一段)。
    为什么不用 round(linspace):v5 每段独立生成 steps_per_segment 帧且段边界
    重复一帧 -> 2 段 18 帧的真实结点是 [0,9,17],均匀假设给 [0,8,17];4 段的
    (以上是 v2.8 前的 18/27/36 数据;去重后 K=2→17 帧、结点 [0,8,16],均匀恰好对)
    第 3 个内部结点是 27,均匀给 26 (服务器标定实测)。

    方向性
    ------
    非达标量 (结构量)。供 segment_wise_angular_error / turn_recall_and_timing /
    waypoint_error 定位真实段边界。

    invalid 光流的处理
    ------------------
    不适用 —— 纯几何,不碰光流,cv2 缺失时照常工作。

    空 / 静止 / 段数不足
    --------------------
    T=0 -> (空, 空);T=1 -> ([0],[0]);全静止/单段直线 -> ([0, T-1], 全 0)。
    相邻段角度差 <= angle_tol_deg 时两段会被合并 (结点数少于 K+1) ——
    score_trajectory 检测到数目不匹配会回退均匀结点并标 'uniform_fallback'。
    """
    return _knots_core(frame_positions, angle_tol_deg)


def knots_from_poses(poses, angle_tol_deg: float = 20.0) -> tuple:
    """(v2) 从 c2w poses 的平移列提取结点帧 (positions 缺失时的次选)。

    数学定义
    --------
    poses (T,4,4) 的平移列 t = poses[:, :3, 3] = (tx, ty, tz),约定
    (camera_trajectory.py:97-127 pixel_to_world_translation):
        tx = (cx - cx0) / win_w   (东正),  ty = (cy - cy0) / win_h   (南正),  tz = 0。
    取伪像素位置 (cy', cx') = (ty, tx) 喂给 _knots_core。**尺度到不了真像素**
    (差 win_w / win_h 两个各向异性系数),但结点 = 朝向 **变化帧**:可逆对角
    线性映射把直线段映成直线段,朝向恒定 <=> 像下仍恒定,因此变化帧不变 ——
    up-to-scale 恢复对结点提取是充分的。朝向 **数值** 会被各向异性拉偏,
    所以本函数只用于定结点,**不** 用于重建 GT 路径形状。

    输入
    ----
    poses: (T,4,4) c2w。angle_tol_deg 同 knots_from_positions。

    输出 / 单位
    -----------
    同 knots_from_positions:(knot_frames, per_frame_segment_id)。

    方向性 / invalid 光流 / 退化
    ---------------------------
    同 knots_from_positions (纯几何)。注意:规范转换器的 poses 由
    interpolate_pixel_positions(waypoints, T) 线性插值而来 —— 拐角落在两帧
    之间时会产生一个"混合步",_knots_core 的聚簇逻辑把它并进同一物理拐角。
    """
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (4, 4):
        raise ValueError(f"poses 必须是 (T,4,4),got {p.shape}")
    trans = p[:, :3, 3]
    pos = np.stack([trans[:, 1], trans[:, 0]], axis=-1)  # (ty, tx) ≙ (cy', cx')
    return _knots_core(pos, angle_tol_deg)


# ============================================================================
# 公开:路径恢复与 GT 路径
# ============================================================================

def recover_content_path(
    frames: np.ndarray,
    flows: Optional[List[np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """恢复生成视频的 CONTENT 路径 (回答:生成内容实际走了什么轨迹)。

    数学定义
    --------
    对第 t 步稠密光流场 F_t (H,W,2),取幅度加权平均位移
        v_t = (sum_p F_t(p) * |F_t(p)|) / (sum_p |F_t(p)|),  若 sum_p |F_t(p)| >= 1
        v_t = (0, 0),                                        否则 (该步视为无运动)
    路径为其前缀和:  path[0] = (0,0);  path[t] = sum_{s<t} v_s。

    输入
    ----
    frames: (T,H,W,3) uint8 (非 uint8 会被 clip 到 [0,255] 再转)。
    flows:  可选的预计算稠密光流列表 (T-1 个 (H,W,2)),由 _compute_flows 产生。
            传入即复用,避免在同一段视频上重复跑 Farneback。传 None 则内部计算。
            调用方有责任保证 flows 确实来自 frames。

    输出 / 单位
    -----------
    (T,2) float64 位置序列,单位 = 像素 (px),x=右/东正, y=下/南正 (图像像素帧)。
    起点恒为原点 (0,0) —— 路径是相对位移的积分,不含绝对位置。

    方向性
    ------
    非达标量 (既非 higher- 也非 lower-is-better):这是一个中间表示,
    供 path_shape_error / segment_wise_angular_error / turn_detection 消费。

    invalid 光流的处理
    ------------------
    cv2 不可用 (ImportError) 或 T < 2 -> 返回 None (绝不静默退化成零路径)。
    单步累计幅度 < 1.0 (近静止/纹理不足) -> 该步计 (0,0),路径在该步保持不动;
    整段都如此时得到一条退化的全零路径,由 is_valid / flow_magnitude 门控识别。

    空 / 静止 / 段数不足
    --------------------
    T < 2 -> None。全静止视频 -> 全零 (T,2) 路径 (合法返回值,但 is_valid=False,
    score_trajectory 不会用它算方向)。本函数与 K 无关,不感知段数。
    """
    if flows is None:
        flows = _compute_flows(frames)
    if flows is None:
        return None
    return _path_from_flows(flows)


def gt_content_path(waypoints: Sequence, T: int, normalize: bool = True) -> np.ndarray:
    """由 CAMERA waypoints 生成 GT 的 CONTENT 路径 (回答:理想内容轨迹是什么)。

    数学定义
    --------
      1. 把 waypoints (cy,cx) [K+1,2] 按 linspace 线性插值到 T 帧,得到 (cy_t, cx_t)
         (对齐 camera_trajectory.interpolate_pixel_positions, camera_trajectory.py:90-94)。
      2. content = -camera 并把起点挪到原点:
             content_x[t] = -(cx_t - cx_0)      (x 东正)
             content_y[t] = -(cy_t - cy_0)      (y 南正)
         负号来自"窗口东移 => 画面内容西移"。
      3. normalize=True 时除以总弧长 L = sum_t |p_{t+1}-p_t|,使总弧长=1。

    输入
    ----
    waypoints: [K+1,2] CAMERA 像素坐标 (cy, cx) —— H5 里存的就是这个。
    T:         目标帧数 (通常 = len(gen_frames),保证结点与恢复路径帧对齐)。
    normalize: True -> 单位弧长 (只留形状);False -> 保留像素尺度。

    输出 / 单位
    -----------
    (T,2) float64。normalize=False 时单位 = 像素 (px);normalize=True 时无量纲
    (以总弧长为单位),起点恒为 (0,0)。

    方向性
    ------
    非达标量:这是 GT 参考路径本身,是 waypoint_error / path_shape_error 的比较基准。

    invalid 光流的处理
    ------------------
    不适用 —— 本函数纯几何,只吃 waypoints,不碰光流,cv2 缺失时照常工作。

    空 / 静止 / 段数不足
    --------------------
    waypoints 非 2D 或为空 -> 返回全零 (T,2) (退化为"静止"参考路径)。
    仅 1 个 waypoint (K=0,无段) -> 各帧重复该点 => content 路径恒为 (0,0);
    此时总弧长=0,_length_normalize 幂等返回原路径 (不会除零)。
    静止 GT (起终点重合,如闭合回路) 同理:弧长可能非零,归一化仍良定义。
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.ndim != 2 or wp.shape[0] < 1:
        return np.zeros((T, 2), dtype=np.float64)
    cy = wp[:, 0]
    cx = wp[:, 1]
    if len(wp) == 1:
        cy_i = np.repeat(cy, T)
        cx_i = np.repeat(cx, T)
    else:
        src = np.linspace(0.0, 1.0, len(wp))
        tgt = np.linspace(0.0, 1.0, T)
        cy_i = np.interp(tgt, src, cy)
        cx_i = np.interp(tgt, src, cx)
    # content = -camera,并把起点挪到原点
    content_x = -(cx_i - cx_i[0])   # x 东正
    content_y = -(cy_i - cy_i[0])   # y 南正
    path = np.stack([content_x, content_y], axis=-1)
    if normalize:
        path = _length_normalize(path)
    return path


# ============================================================================
# 公开:分段角度 / 拐点 / 结点误差 / 形状
# ============================================================================

def segment_wise_angular_error(
    recovered_path: np.ndarray,
    waypoints: Sequence,
    segment_angles: Sequence,
    knot_frames: Optional[Sequence[int]] = None,
) -> Dict[str, object]:
    """逐段方向误差 (回答:每一段是否朝对的方向走)。

    数学定义
    --------
    结点帧 knots = knot_frames (v2: 调用方传入 **真实结点**,来自
    knots_from_positions / knots_from_poses);缺省 (None) 时回退均匀假设
    round(linspace(0, T-1, K+1)) —— v1 行为,真实结点与之不同
    (2 段 18 帧: [0,9,17] vs [0,8,17]),错一帧就把段位移算劈。
    (以上是 v2.8 前的 18/27/36 数据;去重后 K=2→17 帧、结点 [0,8,16],均匀恰好对)
    对第 k 段 (帧跨度 [knots_k, knots_{k+1}]):
        d_k       = path[knots_{k+1}] - path[knots_k]          (该段净位移)
        rec_k     = atan2(-d_k.y, d_k.x) mod 360               (恢复的 content 段方向)
        gt_k      = (segment_angles[k] + 180) mod 360          (GT content 方向 = -camera)
        err_k     = 折叠角差 |rec_k - gt_k| -> [0,180]         (compute_angular_error)
    mean = err_k 在 **有效段** 上的算术均值。
    注意用的是段的 **净位移**,段内的绕行不影响本指标 —— 那是 path_shape_error 的职责。

    输入
    ----
    recovered_path: (T,2) content 路径 (recover_content_path 输出,或合成路径)。
    waypoints:      [K+1,2] CAMERA (cy,cx) —— 仅用于形状对齐语义,当前实现只用
                    segment_angles 与 K 定结点,故 waypoints 不参与数值计算。
    segment_angles: [K] CAMERA 段方向角 (度)。
    knot_frames:    (v2) 可选 [K+1] 真实结点帧。传入时必须恰有 K+1 个,
                    否则与 segment_angles 无法逐段配对 -> 直接 ValueError
                    (fail-loud,绝不静默错配);None -> 均匀回退。

    输出 / 单位
    -----------
    {"per_segment": [K 个 float|None], "mean": float|None}。单位 = 度 (deg),范围 [0,180]。
    None 表示该段无定义 (见下)。

    方向性
    ------
    lower-is-better (0 度 = 完全跟随;180 度 = 完全反向)。随机方向的期望值 = 90 度。

    invalid 光流的处理
    ------------------
    本函数不碰光流。若上游 recover_content_path 因 cv2 缺失返回 None,调用方
    (score_trajectory) 根本不会调用本函数,相关字段保持 None。若某段光流全被
    幅度门控清零,则该段净位移 < _EPS -> 该段记 None 并被排除出 mean 的分母
    (绝不用 0 或 90 度填充)。

    空 / 静止 / 段数不足
    --------------------
    K = 0 (segment_angles 为空) -> {"per_segment": [], "mean": None}。
    段跨度塌陷 (i1 <= i0,发生在 T-1 < K,即帧数少于段数) -> 该段 None。
    全静止路径 -> 每段位移均 < _EPS -> per_segment 全 None, mean = None。
    """
    path = np.asarray(recovered_path, dtype=np.float64)
    seg_ang = np.asarray(segment_angles, dtype=np.float64).ravel()
    K = len(seg_ang)
    T = len(path)
    if knot_frames is None:
        knots = _knot_frames(K + 1, T)
    else:
        knots = np.asarray(knot_frames, dtype=int).ravel()
        if len(knots) != K + 1:
            raise ValueError(
                f"knot_frames 必须恰有 K+1={K + 1} 个结点与 segment_angles 配对,"
                f"got {len(knots)}"
            )
    per_segment: List[Optional[float]] = []
    for k in range(K):
        i0, i1 = int(knots[k]), int(knots[k + 1])
        if i1 <= i0:
            per_segment.append(None)
            continue
        d = path[i1] - path[i0]
        if math.hypot(d[0], d[1]) < _EPS:
            per_segment.append(None)
            continue
        rec_ang = _vec_angle(float(d[0]), float(d[1]))
        gt_content_ang = (float(seg_ang[k]) + 180.0) % 360.0
        per_segment.append(compute_angular_error(rec_ang, gt_content_ang))
    valid = [e for e in per_segment if e is not None]
    mean = float(np.mean(valid)) if valid else None
    return {"per_segment": per_segment, "mean": mean}


def segment_success_rate(
    per_segment_errors: Optional[Sequence],
    thresh_deg: float = 45.0,
) -> Optional[float]:
    """分段成功率 (回答:有多大比例的段"走对了",而不只是"平均差多少度")。

    动机:segment_angular_mean 是均值,会被单个灾难性段拖垮,也会把"3 段都差 40 度"
    和"2 段完美 + 1 段反向 120 度"混为一谈 (均值都是 40)。成功率给出正交的一面:
    达标段的占比。

    数学定义
    --------
    S = { k : per_segment_errors[k] is not None }            (有效段集合)
    segment_success_rate = |{ k in S : err_k < thresh_deg }| / |S|
    注意是 **严格小于** thresh_deg。分母只数有效段。

    输入
    ----
    per_segment_errors: segment_wise_angular_error()["per_segment"],即 [K] 的
                        float|None 列表 (度)。整体传 None 也可 (视作无有效段)。
    thresh_deg:         达标阈值 (度)。score_trajectory 同时报 30 与 45 两档。

    输出 / 单位
    -----------
    float in [0,1],无量纲 (fraction)。无有效段时 None。

    方向性
    ------
    higher-is-better (1.0 = 每段都在阈值内)。随机方向基线的期望 ≈ thresh_deg/180
    (45 度阈值 -> ≈0.25;30 度阈值 -> ≈0.167),读数时以此为"瞎猜"参考线。

    invalid 光流的处理
    ------------------
    本函数不碰光流。位移过小/无定义的段在上游已是 None,这里 **排除出分母**
    (既不算成功也不算失败),避免把"测不到"污染成"失败"。

    空 / 静止 / 段数不足
    --------------------
    per_segment_errors 为 None / [] / 全 None (静止路径、帧数少于段数) -> 返回 None
    (不是 0.0 —— "没有可评的段" != "全部失败")。
    """
    if per_segment_errors is None:
        return None
    valid = [float(e) for e in per_segment_errors if e is not None]
    if not valid:
        return None
    hits = sum(1 for e in valid if e < float(thresh_deg))
    return float(hits) / float(len(valid))


def turn_detection(
    recovered_path: np.ndarray,
    min_turn_deg: float = 15.0,
    win: int = 3,
) -> List[int]:
    """(v2) 加窗拐点检测 (回答:模型在哪些时刻改变了朝向)。

    v1 只比较 **相邻两个单步** 的朝向,三种真实情形都会漏:段边界的重复帧
    (零步把相邻性打断,v1 直接跳过整个拐角)、光流噪声 (单步朝向抖动被当拐点
    或把真拐点淹没)、跨多帧的缓转 (每步变化 < 阈值)。服务器标定里 gt_oracle
    的 turn_recall 只有 0.25,即为此症。v2 用 ±win 窗口的 **步向量均值** 平滑。

    数学定义
    --------
    逐步向量 s_i = path[i+1] - path[i] (i in [0, T-2]),幅度 m_i = |s_i|。
    对顶点 t in [1, T-2]:
        B_t = mean(s_i, i in [max(0,t-win), t))     (拐点前的平均步向量)
        A_t = mean(s_i, i in [t, min(T-1,t+win)))   (拐点后的平均步向量)
    注意平均的是 **向量** 而非角度 —— 向量均值天然按幅度加权,零步 (重复帧)
    只稀释幅度、不翻转方向,角度均值在 0/360 缠绕处则会算错。
        score_t = 折叠角差( angle(B_t), angle(A_t) ) in [0,180]
    可靠性下限 floor = 0.3 * median({m_i : m_i > 0}):
        候选:  score_t > min_turn_deg 且 |B_t| >= floor 且 |A_t| >= floor
    NMS:按 score 降序贪心保留,每保留一个 t 就抑制 |t' - t| <= win 的其余
    候选 (同一物理拐角在窗口平滑后会在相邻数帧上都给出高分,只留局部最大);
    score 并列取更早帧,保证确定性。

    输入
    ----
    recovered_path: (T,2) content 路径。
    min_turn_deg:   记为拐点的最小朝向变化 (度),默认 15.0。
    win:            平滑/NMS 窗口半径 (帧),默认 3。win=1 退化为 v1 的
                    相邻单步比较 (但仍带 NMS 与向量均值语义)。

    输出 / 单位
    -----------
    List[int],升序的帧索引 (单位 = 帧,frame index),取值 [1, T-2]。

    方向性
    ------
    非达标量 (是一个事件列表,不是分数)。其"好坏"由 turn_recall_and_timing
    对照 GT 拐点后给出。数量本身不是越多越好 —— 多余的拐点会压低 turn_precision。

    invalid 光流的处理
    ------------------
    本函数不碰光流。被上游幅度门控清零的步进入向量均值后只会拉低 |B|/|A|,
    两侧均值幅度不过 floor 的顶点直接不候选,不会把噪声抖动误报成拐点。

    空 / 静止 / 段数不足
    --------------------
    T < 3 -> []。全静止 -> 所有均值幅度 0 < floor(=_EPS 下限) -> [];
    直线 -> score 恒 0 -> []。K 段 GT 应有 K-1 个拐点;本函数不知道 K,
    返回多少是多少,由召回/精确率评判。
    """
    path = np.asarray(recovered_path, dtype=np.float64)
    if len(path) < 3:
        return []
    W = max(1, int(win))
    diffs = np.diff(path, axis=0)                       # (T-1,2) 逐步向量
    mags = np.hypot(diffs[:, 0], diffs[:, 1])           # (T-1,)
    n_steps = len(diffs)
    med = float(np.median(mags[mags > 0])) if np.any(mags > 0) else 0.0
    floor = max(_EPS, 0.3 * med)

    cand: List[tuple] = []  # (vertex_frame, score_deg)
    for t in range(1, n_steps):  # 顶点 t: step t-1 结束 / step t 开始
        B = diffs[max(0, t - W):t].mean(axis=0)
        A = diffs[t:min(n_steps, t + W)].mean(axis=0)
        if math.hypot(B[0], B[1]) < floor or math.hypot(A[0], A[1]) < floor:
            continue
        score = compute_angular_error(
            _vec_angle(float(B[0]), float(B[1])),
            _vec_angle(float(A[0]), float(A[1])),
        )
        if score > float(min_turn_deg):
            cand.append((t, score))
    if not cand:
        return []

    # NMS: score 降序贪心,±win 内抑制;并列取更早帧 (确定性)。
    cand.sort(key=lambda x: (-x[1], x[0]))
    kept: List[int] = []
    for t, _score in cand:
        if all(abs(t - k) > W for k in kept):
            kept.append(t)
    return sorted(kept)


def turn_recall_and_timing(
    recovered_turns: Sequence[int],
    gt_turn_frames: Sequence[int],
    tol: int = 2,
) -> Dict[str, object]:
    """拐点召回率 / 精确率 / 时机误差 (回答:该转的地方转了吗、有没有乱转、转早转晚了吗)。

    数学定义
    --------
    匹配:按 GT 拐点升序做一对一贪心 —— 每个 GT 拐点 g 认领尚未被占用、且
    |r - g| <= tol 的最近恢复拐点 r。matched = 成功认领的 GT 拐点数。
        recall            = matched / |gt_turn_frames|
        precision         = matched / |recovered_turns|
        mean_timing_error = mean_{已匹配} |r - g|
    recall 回答"该转的转了没",precision 回答"转的都该转吗"(惩罚乱转/抖动误报),
    两者必须一起看:狂输出拐点可以刷满 recall,但 precision 会塌。

    退化约定 (显式列出,避免 0/0)
    ---------------------------
      gt 空, rec 空  -> recall=1.0, precision=1.0  (直线 GT,模型也没乱转 = 完全正确)
      gt 空, rec 非空 -> recall=1.0, precision=0.0  (直线 GT,模型全是虚报拐点)
      gt 非空, rec 空 -> recall=0.0, precision=0.0
          注:此时 precision 的分母为 0,严格说 **无定义**。按本仓规范取 0.0 而非
          None/1.0 —— 一个什么拐点都没产出的模型不应在 precision 上得满分,
          且这样 turn_f1 才能保持良定义。recall 独立成列,不受此约定影响。

    输入
    ----
    recovered_turns: turn_detection 输出的帧索引序列。
    gt_turn_frames:  GT 内部结点帧 (K 段 -> K-1 个 = knots[1:-1])。(v2) knots 优先
                     来自真实结点 (knots_from_positions/knots_from_poses),
                     仅在回退时才是均匀 _knot_frames(K+1,T)。
    tol:             匹配容差 (帧),默认 2。

    输出 / 单位
    -----------
    {"recall": float, "precision": float, "f1": float,
     "mean_timing_error": float|None}。
    recall/precision/f1 无量纲 in [0,1];mean_timing_error 单位 = 帧 (frames),
    无任何匹配时 None (不填 0 —— "没转弯"和"转得很准"必须区分开)。

    方向性
    ------
    recall / precision / f1: higher-is-better。
    mean_timing_error: lower-is-better,且只在 recall > 0 时可读
    (它是 **条件** 于"转对了"的时机偏差;不能用它替代 recall)。

    invalid 光流的处理
    ------------------
    本函数纯整数运算,不碰光流。光流不可用时上游不产生 recovered_turns,
    score_trajectory 相关字段保持 None,不会调用本函数。

    空 / 静止 / 段数不足
    --------------------
    见上"退化约定"。静止视频 -> 上游 valid=False,本函数不被调用。
    K=1 (单段直线) -> gt_turn_frames 为空 -> 走 "gt 空" 分支。
    """
    gt = sorted(int(x) for x in gt_turn_frames)
    rec = sorted(int(x) for x in recovered_turns)
    if not gt:
        # GT 是直线:没有该转的地方 -> recall 满分;但凡产出拐点即全为虚报 -> precision=0。
        precision = 1.0 if not rec else 0.0
        return {
            "recall": 1.0,
            "precision": precision,
            "f1": _harmonic_mean(precision, 1.0),
            "mean_timing_error": None,
        }
    used = [False] * len(rec)
    timing_errs: List[int] = []
    matched = 0
    for g in gt:
        best_j, best_d = -1, tol + 1
        for j, r in enumerate(rec):
            if used[j]:
                continue
            d = abs(r - g)
            if d <= tol and d < best_d:
                best_j, best_d = j, d
        if best_j >= 0:
            used[best_j] = True
            matched += 1
            timing_errs.append(best_d)
    recall = matched / len(gt)
    # rec 为空时 precision 分母为 0 -> 按上文退化约定取 0.0 (非 None/1.0)。
    precision = (matched / len(rec)) if rec else 0.0
    mean_timing = float(np.mean(timing_errs)) if timing_errs else None
    return {
        "recall": float(recall),
        "precision": float(precision),
        "f1": _harmonic_mean(float(precision), float(recall)),
        "mean_timing_error": mean_timing,
    }


def waypoint_error(
    recovered_path: np.ndarray,
    gt_path: np.ndarray,
    gt_knot_frames: Sequence[int],
) -> Optional[float]:
    """GT 结点处的 **平均** 归一化位置偏差 (回答:关键途经点整体踩得准不准)。

    数学定义
    --------
    P~ = P / len(P),  Q~ = Q / len(Q)          (各自单位总弧长归一化,去掉尺度差)
    waypoint_error = (1/|F|) * sum_{f in F} |P~[f] - Q~[f]|
    其中 F = { gt_knot_frames 中同时落在两条路径帧范围内的帧 }。
    归一化后总弧长 = 1,故该距离天然是"偏差 / 全程长度"的比值。

    输入
    ----
    recovered_path: (T,2) 恢复的 content 路径 (未归一化亦可,函数内部会归一化)。
    gt_path:        (T,2) GT content 路径 (gt_content_path 输出,或 (v2) 由真实
                    逐帧位置直接构造的 content 路径)。
    gt_knot_frames: GT waypoint 落在的帧索引 (共 K+1 个)。(v2) 优先传真实结点
                    (knots_from_positions/knots_from_poses);均匀 _knot_frames
                    仅作回退 (2 段 18 帧时均匀 [0,8,17] vs 真实 [0,9,17])。
                    (以上是 v2.8 前的 18/27/36 数据;去重后 K=2→17 帧、结点 [0,8,16],均匀恰好对)

    输出 / 单位
    -----------
    float,无量纲 (以总弧长为单位的比值;0.1 = 偏离约全程长度的 10%)。
    无有效结点时 None。

    方向性
    ------
    lower-is-better (0 = 每个途经点都精确踩中)。

    invalid 光流的处理
    ------------------
    本函数不碰光流。光流不可用 -> 上游 recover_content_path 返回 None ->
    score_trajectory 直接把本字段留 None,不会调用本函数。
    恢复路径若因门控退化成全零,总弧长 < _EPS -> _length_normalize 幂等返回零路径,
    此时距离 = GT 结点到原点的距离 (一个大而真实的惩罚,不是 NaN)。

    空 / 静止 / 段数不足
    --------------------
    gt_knot_frames 为空,或所有结点都越界 (帧数少于段数导致的错位) -> None。
    单 waypoint (K=0) -> 只有 1 个结点 (帧 0),两路径起点均为原点 -> 0.0。

    与 waypoint_endpoint_error 的关系
    ---------------------------------
    本函数是 **所有** 结点的均值 (含起点,起点恒为 0);waypoint_endpoint_error
    只看 **终点** 那一个结点,专门度量"最终落点漂没漂"。
    """
    dists = _knot_distances(recovered_path, gt_path, gt_knot_frames)
    if not dists:
        return None
    return float(np.mean([d for _, d in dists]))


def waypoint_endpoint_error(
    recovered_path: np.ndarray,
    gt_path: np.ndarray,
    gt_knot_frames: Sequence[int],
) -> Optional[float]:
    """**终点** 结点处的归一化位置偏差 (回答:最终落点漂了多远)。

    动机:waypoint_error 是全结点均值,起点恒为 0 会稀释它,中途的补偿性误差也会
    互相抵消。终点误差单独拎出来,直接回答"整段走完后停在了该停的地方吗" ——
    这是导航式可控性最关心的一个数,也是与 net_direction_error 最互补的一个数
    (方向对但走过头/没走够时,net_direction_error ≈ 0 而本指标显著 > 0)。

    数学定义
    --------
    P~ = P / len(P),  Q~ = Q / len(Q)          (与 waypoint_error 完全同一套归一化)
    f* = max(F),  F = { gt_knot_frames 中落在两条路径帧范围内的帧 }
    waypoint_endpoint_error = |P~[f*] - Q~[f*]|
    f* 即最后一个 (terminal) waypoint 所在帧,正常流程下 = T-1。

    输入
    ----
    recovered_path: (T,2) 恢复的 content 路径。
    gt_path:        (T,2) GT content 路径。
    gt_knot_frames: GT waypoint 帧索引 (共 K+1 个;(v2) 优先真实结点,均匀仅回退)。

    输出 / 单位
    -----------
    float,无量纲 (以总弧长为单位的比值)。无有效结点时 None。
    量纲上界参考:两条单位长度路径的终点最远相距 2.0。

    方向性
    ------
    lower-is-better (0 = 终点精确落位)。

    invalid 光流的处理
    ------------------
    同 waypoint_error:光流不可用 -> 上游不调用本函数,字段保持 None;
    退化零路径 -> 返回 GT 终点到原点的距离 (真实惩罚,非 NaN)。

    空 / 静止 / 段数不足
    --------------------
    gt_knot_frames 为空或全部越界 -> None。
    单 waypoint (K=0) -> 唯一结点即帧 0 = 起点 -> 0.0 (无终点可言,退化为起点)。
    """
    dists = _knot_distances(recovered_path, gt_path, gt_knot_frames)
    if not dists:
        return None
    return float(dists[-1][1])  # _knot_distances 按帧升序 -> 末项即终点结点


def path_shape_error(recovered_path: np.ndarray, gt_path: np.ndarray) -> Dict[str, float]:
    """离散 Fréchet 距离 + DTW (回答:整条轨迹的形状像不像)。

    这是 Track B 相对 Track A 的核心增量:net_direction_error 只看首尾连线,
    一条"绕大圈但首尾正确"的路径能骗过它;Fréchet/DTW 骗不过。

    数学定义
    --------
    预处理 (两条路径各自独立做,只保留 **形状**):
        P~ = anchor(P) / len(anchor(P)),   anchor(X) = X - X[0]
    即先平移到原点 (平移不变),再除以总弧长 (尺度不变)。弧长平移不变,故两步可交换。
    随后在归一化路径上做 numpy-only 的 O(T^2) DP:
      - 离散 Fréchet (Eiter & Mannila 耦合搜索, max-min 递推):
            ca[i,j] = max( min(ca[i-1,j], ca[i-1,j-1], ca[i,j-1]), d(P~_i, Q~_j) )
            frechet = ca[n-1, m-1]              ("遛狗距离":最坏时刻的偏离)
      - DTW (经典最小累计代价,**未** 做长度归一):
            D[i,j] = d(P~_i, Q~_j) + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
            dtw    = D[n, m]                    (累计偏离,随 T 增长)
    注意保留 **时间顺序**:两者都不允许回溯匹配,故也惩罚"路线对但时序错"。

    输入
    ----
    recovered_path: (T,2) 恢复的 content 路径 (任意尺度/任意全局偏移均可)。
    gt_path:        (T,2) GT content 路径 (gt_content_path 输出)。

    输出 / 单位
    -----------
    {"frechet": float, "dtw": float}。二者均无量纲 (以总弧长为单位)。
      - frechet in [0, 2]:单点最大偏离,可跨样本直接比较。
      - dtw >= 0:沿路累计偏离,量级 ~ O(T) —— **只在同一 T 下横向可比**,
        不同帧数的样本之间不要直接比 DTW。
    路径为空 (n==0 或 m==0) 时两者均为 float("nan")。

    方向性
    ------
    both lower-is-better (0 = 形状完全一致)。

    invalid 光流的处理
    ------------------
    本函数不碰光流。光流不可用 -> 上游 recover_content_path 返回 None ->
    score_trajectory 把 frechet/dtw 留 None,不会调用本函数。
    恢复路径若退化成全零 (静止),弧长 < _EPS -> _length_normalize 幂等返回零路径,
    此时得到的是"零路径 vs 单位 GT 路径"的真实距离 (非 NaN),但正常流程下
    valid=False 已提前拦截,不会走到这里。

    空 / 静止 / 段数不足
    --------------------
    任一路径为空 -> NaN (不是 0 —— 算不出来 != 完美)。
    与段数 K 无关:本函数只比两条折线,不需要知道分了几段。
    两路径长度可以不等 (n != m):Fréchet/DTW 本就支持非等长匹配。
    """
    P = _length_normalize(_anchor_to_origin(np.asarray(recovered_path, dtype=np.float64)))
    Q = _length_normalize(_anchor_to_origin(np.asarray(gt_path, dtype=np.float64)))
    n, m = len(P), len(Q)
    if n == 0 or m == 0:
        return {"frechet": float("nan"), "dtw": float("nan")}
    d = _pairwise_dist(P, Q)  # (n,m)

    # ---- 离散 Fréchet (迭代填表,避免递归深度) ----
    ca = np.empty((n, m), dtype=np.float64)
    ca[0, 0] = d[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d[i, j])
    frechet = float(ca[n - 1, m - 1])

    # ---- DTW ----
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = d[i - 1, j - 1]
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    dtw = float(D[n, m])

    return {"frechet": frechet, "dtw": dtw}


# ============================================================================
# 公开:净方向 / 幅度 / 有效性 (text-only 也成立的部分)
# ============================================================================

def polyline_net_ratio(points) -> float:
    """(v2.1) 折线的净位移比 = |终点-起点| / 总弧长,无量纲 in [0,1]。

    用于 **GT 命令轨迹** 的近闭合判定:waypoints 折线本身就是被命令的路径,
    它的净位移比是"净方向这个问题有没有意义"的唯一裁判 —— 与光流恢复质量无关。
    退化 (点数<2 或弧长≈0) -> 0.0。
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or len(p) < 2:
        return 0.0
    net = p[-1] - p[0]
    arc = _path_length(p)
    return float(math.hypot(net[0], net[1]) / arc) if arc > _EPS else 0.0


def net_direction_from_path(
    recovered_path: np.ndarray,
    gt_content_net_angle: float,
    min_net_ratio: float = 0.2,
    gt_net_ratio: Optional[float] = None,
) -> Dict[str, object]:
    """(v2.1) 端点法净方向误差 + **GT 命令轨迹**净位移门控。

    v1 的 net_direction_error 逐字复用 Track A estimate_motion_direction
    (T//8 步长采样 + 幅度加权全局求和),对多段折线不稳定 —— 服务器标定里
    **真值视频** 都测出 80.67 deg。v2 改为端点法:恢复路径首尾连线的方向。

    [v2.1 修正] v2 曾用 **恢复路径** 的净位移比做门控 —— 用错了对象。
    "净方向有没有意义"由 **GT 命令轨迹** 决定,与光流恢复质量无关:
      * GT 近闭合 (如 gt_ratio=0.054) -> 净方向问题本身无意义 -> 必须门控,
        即使恢复路径碰巧净位移比很大 (服务器样本 00349:恢复 ratio=0.215
        被错误放行,147.36° 垃圾值污染 gt_oracle 均值 -> 42.79° 标定失败)。
      * GT 明确有净方向 (如 gt_ratio=0.216) -> 必须评,即使恢复路径净位移比
        很小 (样本 00331:恢复 ratio=0.074 被错误门控) —— 恢复比小恰恰是
        模型/真值的可测差异,不是"问题无意义"。

    数学定义
    --------
    net       = path[-1] - path[0]                       (净位移向量, x东正 y南正)
    arc       = sum_t |path[t+1] - path[t]|              (总弧长)
    rec_ratio = |net| / arc                              (恢复路径净位移比,诊断量)
    gated     = (gt_net_ratio if 给定 else rec_ratio) < min_net_ratio
    error_deg = gated 时 None;否则折叠角差( atan2(-net.y, net.x), gt ) in [0,180]。
    另:未门控但恢复净位移本身退化 (|net| < _EPS,方向未定义) -> error_deg=None
    + recovered_net_degenerate=True (不是门控,是"恢复不出方向",两者语义不同)。

    输入
    ----
    recovered_path:       (T,2) content 路径 (recover_content_path 输出)。
    gt_content_net_angle: GT **CONTENT** 净方向角 (度) = (camera 净方向 + 180)
                          mod 360,由 _camera_net_angle(waypoints) 提供。
    min_net_ratio:        门控阈值,默认 0.2。
    gt_net_ratio:         **GT 命令轨迹** 的净位移比 (polyline_net_ratio(waypoints));
                          v2.1 起 score_trajectory 一定传它。None 时退回用恢复
                          路径比例门控 (仅兼容旧调用方,新代码不应依赖)。

    输出 / 单位
    -----------
    {"error_deg": float|None, "gt_net_ratio": float|None,
     "recovered_net_ratio": float, "gated": bool,
     "recovered_net_degenerate": bool,
     "recovered_net_angle": float|None}。
    error_deg 单位 = 度 [0,180];两个 ratio 无量纲 in [0,1];
    recovered_net_angle = 恢复净方向原始角 (0-360,未折叠,v2.3 偏置尸检用),
    净位移可定义即报,不受门控约束(error_deg 才受)。

    方向性
    ------
    error_deg: lower-is-better (随机方向期望 90)。recovered_net_ratio: 匹配量,
    应接近 gt_net_ratio (直线 GT ≈1,近闭合 GT 本来就低)。gated: 标志,非分数。

    invalid 光流的处理
    ------------------
    本函数不碰光流;光流不可用时上游没有 recovered_path,不会调用本函数。

    空 / 静止 / 段数不足
    --------------------
    T < 2 -> gated=True + None。与段数 K 无关 (只看首尾 + 弧长)。
    """
    path = np.asarray(recovered_path, dtype=np.float64)
    if len(path) < 2:
        # 短路径 = "恢复不出方向"(degenerate),不是 "问题无意义"(gated)。
        # gated 只反映 GT 命令轨迹:GT 给定且近闭合才 True;GT 给定且明确则
        # False(恢复失败照实标 degenerate);GT 未给定时保守 True(legacy)。
        _g = (float(gt_net_ratio) < float(min_net_ratio)
              if gt_net_ratio is not None else True)
        return {"error_deg": None, "gt_net_ratio": gt_net_ratio,
                "recovered_net_ratio": 0.0, "gated": _g,
                "recovered_net_degenerate": True,
                "recovered_net_angle": None}
    net = path[-1] - path[0]
    net_mag = float(math.hypot(net[0], net[1]))
    arc = _path_length(path)
    rec_ratio = float(net_mag / arc) if arc > _EPS else 0.0
    # (v2.3) 原始恢复净方向角(未折叠,0-360)—— 带符号偏置尸检用:folded
    # error 丢符号,判不出"系统性单向旋偏 vs 对称大误差"。门控/退化时也照报
    # (只要净位移可定义),仅 error_deg 遵守门控语义。
    rec_angle = (_vec_angle(float(net[0]), float(net[1]))
                 if net_mag >= _EPS else None)
    # [v2.1] 门控对象 = GT 命令轨迹的净位移比;恢复比例仅作诊断输出。
    gate_ratio = float(gt_net_ratio) if gt_net_ratio is not None else rec_ratio
    gated = gate_ratio < float(min_net_ratio)
    if gated:
        return {"error_deg": None, "gt_net_ratio": gt_net_ratio,
                "recovered_net_ratio": rec_ratio, "gated": True,
                "recovered_net_degenerate": False,
                "recovered_net_angle": rec_angle}
    if net_mag < _EPS:
        # GT 有净方向但恢复净位移严格退化:方向未定义,如实 None(非门控)。
        return {"error_deg": None, "gt_net_ratio": gt_net_ratio,
                "recovered_net_ratio": rec_ratio, "gated": False,
                "recovered_net_degenerate": True,
                "recovered_net_angle": None}
    err = compute_angular_error(
        _vec_angle(float(net[0]), float(net[1])),
        float(gt_content_net_angle) % 360.0,
    )
    return {"error_deg": err, "gt_net_ratio": gt_net_ratio,
            "recovered_net_ratio": rec_ratio, "gated": False,
            "recovered_net_degenerate": False,
            "recovered_net_angle": rec_angle}


def track_a_net_direction_error(frames: np.ndarray, target_angle: float) -> Optional[float]:
    """净方向误差 —— **旧 Track A 口径** (仅供与 v2 端点法对照,不再被
    score_trajectory 使用;score_trajectory 的 'net_direction_error' 现在来自
    net_direction_from_path 的端点法)。

    数学定义
    --------
    est = metrics.estimate_motion_direction(frames)      # 逐字复用,不重实现
        它以 step = max(1, T//8) 采样帧对做 Farneback,把幅度加权的 (dx,dy) 在
        所有帧对上累加,再折叠成单一角度 est = atan2(-dy_tot, dx_tot) mod 360
        (metrics.py:238-284)。**这是 Track A 的口径,必须逐字对齐,不得替换成
        本模块的逐帧 _compute_flows —— 两者采样步长不同,数值不可互换。**
    net_direction_error = 折叠角差 |est - target_angle| -> [0,180]。
    已知缺陷 (v2 弃用它的原因):T//8 步长的幅度加权全局求和对多段折线不稳定,
    服务器标定里对 **真值视频** 都给出 80.67 deg;且无净位移门控,近闭合路径
    也硬给一个数。别名 net_direction_error_flow 指向同一实现。

    输入
    ----
    frames:       (T,H,W,3) uint8 生成视频。
    target_angle: GT **CONTENT** 方向角 (度) = (camera 方向 + 180) mod 360。
                  传 camera 方向会得到系统性的 180 度偏差 —— 调用方 (score_trajectory)
                  已用 _camera_net_angle(wp) + 180 处理好。

    输出 / 单位
    -----------
    float,单位 = 度 (deg),范围 [0,180]。无法估计时 None。

    方向性
    ------
    lower-is-better (0 = 净方向完全正确)。随机方向的期望值 = 90 度。
    **局限 (Track B 存在的理由)**:它把整段视频折叠成一个角度,只反映首尾净位移。
    绕行/漏拐/走过头都可能得 ≈0 分 —— 必须与 frechet/dtw/segment_* 联读。

    invalid 光流的处理
    ------------------
    estimate_motion_direction 在 cv2 缺失、T < 2、或累计幅度 < 1.0 (近静止,
    方向不可靠) 时返回 None -> 本函数原样返回 None,**不** 用 0/90 度填充。

    空 / 静止 / 段数不足
    --------------------
    T < 2 或静止 -> None。与段数 K 无关 (只看净位移);K>=2 的多段轨迹在这里
    会被压成一个净方向,这正是本指标的已知盲区。
    """
    # [v3.0 核实修正] 显式 step=1。默认值(旧为 len//8)会让 T=17 的样本按
    # 18 px/帧对跑光流,而实测 24 px 处误差已达 128°(近乎反向)。
    est = estimate_motion_direction(frames, step=1)
    if est is None:
        return None
    return compute_angular_error(est, float(target_angle))


# 旧行为的显式别名 (对照实验用):flow 口径 = Track A estimate_motion_direction。
# v2 端点法见 net_direction_from_path;score_trajectory 输出键 'net_direction_error'
# 自 v2 起指端点法,不再是本别名。
net_direction_error_flow = track_a_net_direction_error


def flow_magnitude(
    frames: np.ndarray,
    flows: Optional[List[np.ndarray]] = None,
) -> Optional[float]:
    """逐帧平均光流幅度 —— movement / liveness 检查 (回答:画面动没动、动多快)。

    数学定义
    --------
    mag = (1/(T-1)) * sum_t [ (1/(H*W)) * sum_p |F_t(p)| ],  |F_t(p)| = sqrt(fx^2+fy^2)
    即"先对每步的稠密幅度场取空间均值,再对步取均值"。注意这是 **无加权** 的
    空间均值 (与 _step_vector 的幅度加权平均不同):它度量运动量,不度量方向。

    输入
    ----
    frames: (T,H,W,3) uint8。
    flows:  可选预计算光流 (见 recover_content_path);传入即复用,不重跑 Farneback。

    输出 / 单位
    -----------
    float,单位 = 像素/帧 (px/frame)。无法估计时 None。

    方向性
    ------
    既非 higher- 也非 lower-is-better:这是一个 **匹配量** —— 应接近 GT 幅度
    (生成分辨率 480x832 下 GT 约 34 px/frame;v2 降到原生窗口分辨率 128x222 后
    约 9 px/frame,且后者才在 Farneback 可靠量程内,见模块 docstring v2 节)。
    过低 => 静止/塌缩;过高 => 抖动/漂移。仅作 is_valid 门控与诊断量,不作达标量。

    invalid 光流的处理
    ------------------
    cv2 不可用或 T < 2 -> 返回 None (不返回 0.0,以便与"真的没动"区分开)。

    空 / 静止 / 段数不足
    --------------------
    T < 2 -> None。完全静止 -> 返回 ~0.0 (合法数值)。与段数 K 无关。
    """
    if flows is None:
        flows = _compute_flows(frames)
    if flows is None:
        return None
    return _mean_magnitude_from_flows(flows)


def is_valid(
    frames: np.ndarray,
    min_mag: float = 1.5,
    flows: Optional[List[np.ndarray]] = None,
) -> bool:
    """是否有足够运动可供方向/路径分析 (回答:这条样本值不值得打分)。

    数学定义
    --------
    is_valid = (flow_magnitude(frames) is not None) and (flow_magnitude >= min_mag)

    输入
    ----
    frames:  (T,H,W,3) uint8。
    min_mag: 阈值 (px/frame),默认 1.5 —— **窗口分辨率口径** (v2):Track B 光流
             一律在原生窗口分辨率 (128x222) 上算,GT 在该分辨率下 ~9 px/frame,
             1.5 既滤掉 persistence/近静止,又不会把慢速真实运动误杀。若在
             全分辨率 (480x832, ~34 px/frame) 上调用且未降采样,请自行传旧值 5.0。
    flows:   可选预计算光流;传入即复用。

    输出 / 单位
    -----------
    bool,无量纲。

    方向性
    ------
    higher-is-better (True 好):False 表示该样本的方向/路径指标无定义。
    注意 True 只代表"动了",不代表"动对了"。

    invalid 光流的处理
    ------------------
    cv2 缺失或 T < 2 -> flow_magnitude 返回 None -> 保守返回 False
    (绝不把"算不出来"当成"通过")。

    空 / 静止 / 段数不足
    --------------------
    T < 2 -> False。近静止 (mag < min_mag,如 persistence 基线) -> False。
    与段数 K 无关。
    """
    mag = flow_magnitude(frames, flows=flows)
    if mag is None:
        return False
    return mag >= float(min_mag)


# ============================================================================
# 公开:总调度
# ============================================================================

def score_trajectory(
    gen_frames: np.ndarray,
    waypoints: Sequence,
    segment_angles: Sequence,
    num_frames: Optional[int] = None,
    min_mag: float = 1.5,
    text_only: bool = False,
    window_size: Optional[Sequence[int]] = None,
    frame_positions: Optional[Sequence] = None,
    poses: Optional[Sequence] = None,
    path_estimator: Optional[str] = None,
) -> Dict[str, object]:
    """跑完整 Track B 打分,返回 **扁平** dict (每个 key 一个标量/列表,便于直接聚合)。

    数学定义
    --------
    本函数是编排器,不自带新数学:它按固定顺序调用下列已各自定义好的指标 ——
    (v2) _maybe_downsample -> flow_magnitude -> (liveness 门控)
    -> recover_content_path -> net_direction_from_path (端点法 + 门控)
    -> knots_from_positions/knots_from_poses/_knot_frames (真实结点,含回退)
    -> segment_wise_angular_error -> segment_success_rate -> turn_detection
    -> turn_recall_and_timing -> waypoint_error / waypoint_endpoint_error
    -> path_shape_error,并把结果摊平成单层 dict。各指标的公式见其自身 docstring。

    v2 分辨率约定 (服务器标定,详见模块 docstring v2 节)
    --------------------------------------------------
    window_size=(win_h,win_w) 给定且帧更大时,先把 **所有** 帧一次性降采样到
    原生窗口分辨率再算光流 (480x832 是 128x222 的上采样,~34 px/frame 超出
    Farneback 量程,gt_oracle 都失败;降回窗口分辨率后 ~9 px/frame,GT 逐段
    误差 ~1-10 deg)。waypoints / frame_positions 仍留在 **scene 像素**,不随帧
    降采样:路径 **形状** 比较对两条路径各自做单位弧长归一化 + 起点锚定,
    尺度不变;逐步向量 (恢复路径的步长、turn/segment 的位移) 全部来自同一批
    窗口分辨率光流,**自身一致**,绝不与 scene 像素向量直接相减。

    输入
    ----
    gen_frames: (T,H,W,3) uint8 生成视频。
    waypoints:  [K+1,2] CAMERA 路径像素坐标 (cy,cx),scene 像素。
    segment_angles: [K] CAMERA 段方向角 (度)。
    num_frames: GT 段数插值用的帧数;默认取 len(gen_frames),保证结点对齐。
    window_size: (v2) 可选 (win_h, win_w) —— H5 f['metadata'].attrs
                ['pose_window_size']。给定则降采样后再算光流 (见上节);None
                维持 v1 行为 (全分辨率,已知在 480x832 上不可靠)。
    frame_positions: (v2) 可选 (T,2) **真实逐帧** 窗口中心 (cy,cx),scene 像素
                (v5 源 benchmark 的 'trajectory' 数据集)。给定且与帧数一致时,
                真实结点与 GT 路径都从它取 -> knots_source='positions'。
    poses:      (v2) 可选 (T,4,4) c2w。frame_positions 缺失时从平移列提取
                真实结点 (up-to-scale) -> knots_source='poses';只用于定结点,
                不用于 GT 路径形状 (各向异性归一化会畸变形状,见 knots_from_poses)。
                两者都缺 -> 均匀 linspace 回退 -> knots_source='uniform_fallback'。
    min_mag:    liveness 阈值 (px/frame),低于此视为静止 -> valid=False。
                (v2) 默认 1.5,**窗口分辨率口径** (该分辨率下 GT ~9 px/frame);
                不降采样跑全分辨率时请自行传旧值 5.0。
    text_only:  True 表示条件里只有方向词、没给折线轨迹 -> **只有净方向 +
                段数是被命令的** (语义与 v1 完全一致);分段/拐点/结点/形状指标
                会照常计算,但 `commanded=False` 标注它们并非达标量。

    输出 / 单位 (扁平 dict 的全部 key,附单位与方向性)
    ------------------------------------------------
      valid                    bool  —— 是否动了 (higher better);False 时下列全为 None
      commanded                bool  —— 折线是否真的作为条件给了模型 (非分数)
      num_frames_used          int   —— 实际参与打分的帧数
      num_segments             int   —— K
      flow_magnitude           px/frame —— 匹配量 (窗口分辨率下应接近 GT ~9;
                               全分辨率 ~34)
      net_direction_error      deg [0,180] —— lower better ((v2) 端点法:恢复
                               路径首尾连线 vs GT content 净方向;被门控时 None)
      net_displacement_ratio   [0,1] —— (v2.1) **GT 命令轨迹** 的 |净位移|/总弧长
                               (净方向门控依据;与恢复质量无关)
      recovered_net_displacement_ratio [0,1] —— (v2.1) 恢复路径的比例,仅诊断
      net_direction_gated      bool|None —— (v2.1) True = **GT 命令轨迹** 近闭合
                               (净方向问题无意义;与恢复质量无关)
                               (net_direction_error 为 None 的原因标志)
      knots_source             str|None —— (v2) 'positions'|'poses'|
                               'uniform_fallback';结点来自哪里 (非分数)
      gt_net_content_angle     deg [0,360) —— 参考值,非分数
      segment_angular_errors   [K] deg|None —— lower better (逐段)
      segment_angular_mean     deg [0,180] —— lower better
      segment_success_rate_30  [0,1] —— higher better (逐段 <30 deg 的占比)
      segment_success_rate_45  [0,1] —— higher better (逐段 <45 deg 的占比)
      turn_recall              [0,1] —— higher better
      turn_precision           [0,1] —— higher better (惩罚乱转/虚报)
      turn_f1                  [0,1] —— higher better (recall/precision 调和平均)
      turn_timing_error        frames —— lower better (仅在 recall>0 时可读)
      num_gt_turns             int   —— = max(0, K-1) (K 段路径恰有 K-1 个拐点)
      num_recovered_turns      int   —— 非分数 (配合 precision 读)
      waypoint_error           无量纲 (占总弧长) —— lower better (全结点均值)
      waypoint_endpoint_error  无量纲 (占总弧长) —— lower better (仅终点)
      frechet                  无量纲 [0,2] —— lower better (最坏时刻偏离)
      dtw                      无量纲 —— lower better (累计偏离;仅同 T 可比)

    方向性
    ------
    逐 key 的 higher/lower-is-better 见上表 (valid / *_success_rate / turn_* 为
    higher better;所有 *_error / frechet / dtw 为 lower better;flow_magnitude 为
    匹配量;commanded / gt_net_content_angle / num_* 非分数)。

    invalid 光流的处理
    ------------------
    cv2 不可用 或 T < 2 -> _compute_flows 返回 None -> flow_magnitude=None、
    valid=False,所有方向/路径字段保持 None 并立即返回 (**不静默退化**,
    调用方能从 valid=False + flow_magnitude=None 区分"缺 cv2"与"真静止")。
    若光流可用但某段位移被幅度门控清零,该段的 per-segment 误差为 None 并被
    排除出 mean / success_rate 的分母。任何"算不出来"的量一律 None,绝不填 0 或 90
    (填 0 会让静止基线在 error 类指标上冒充 SOTA)。

    空 / 静止 / 段数不足
    --------------------
    近静止 (flow_magnitude < min_mag,如 persistence 基线) -> valid=False,
    仅 flow_magnitude 有值,其余方向/路径字段全 None。
    waypoints 为 None 或 < 2 个点 -> gt_net_content_angle=None,跳过全部折线类指标。
    K = 0 (segment_angles 为空) -> 跳过分段/成功率 (保持 None),拐点/结点/形状仍计算。
    K = 1 (单段直线) -> num_gt_turns=0,turn_recall 走 "gt 空" 分支 (recall=1.0,
    precision 取决于是否虚报拐点)。
    帧数少于段数 (T-1 < K) -> 塌陷段的 per-segment 为 None,由上述分母规则处理。

    行为
    ----
    若 liveness 不通过 (近静止或 cv2 不可用),把所有方向/路径字段置 None、
    valid=False 并立即返回 (flow_magnitude 本身仍会报告以便诊断);
    否则恢复内容路径并计算全部指标。

    效率
    ----
    (v2) 降采样 (_maybe_downsample) 与稠密光流 (_compute_flows) 各 **只跑一次**,
    喂给 flow_magnitude / recover_content_path / 净方向 (端点法直接读恢复路径)
    共用。v1 里为对齐 Track A 而额外跑的 estimate_motion_direction 已随端点法
    移除 —— 旧口径要对照时显式调用 track_a_net_direction_error /
    net_direction_error_flow。净效果:每样本 1 次 Farneback 全量扫描。
    """
    gen_frames = np.asarray(gen_frames)
    # (v2) 原生窗口分辨率:一次共享降采样,之后所有光流/路径都在这套帧上。
    gen_frames = _maybe_downsample(gen_frames, window_size)
    T = int(num_frames) if num_frames else len(gen_frames)
    T = min(T, len(gen_frames)) if len(gen_frames) else T
    wp = np.asarray(waypoints, dtype=np.float64) if waypoints is not None else None
    seg_ang = np.asarray(segment_angles, dtype=np.float64).ravel() if segment_angles is not None else np.array([])
    K = len(seg_ang)

    # GT content 方向 (net) = camera 净方向 + 180
    gt_net_content = None
    if wp is not None and len(wp) >= 2:
        gt_net_content = (_camera_net_angle(wp) + 180.0) % 360.0

    out: Dict[str, object] = {
        "valid": False,
        # 完整轨迹条件下这些指标"被命令"; text-only 下仅诊断。
        "commanded": (not text_only),
        "num_frames_used": T,
        "num_segments": K,
        # 方向 / 幅度 (text-only 也成立)
        "flow_magnitude": None,
        "net_direction_error": None,
        # (v2.1) net_displacement_ratio = **GT 命令轨迹** 的 |净位移|/总弧长
        # (门控依据;与恢复质量无关)。恢复路径自身的比例另存诊断键。
        "net_displacement_ratio": None,
        "recovered_net_displacement_ratio": None,  # (v2.1) 恢复路径的比例,仅诊断
        "net_direction_gated": None,      # (v2.1) True = GT 命令近闭合,净方向无意义
        "recovered_net_angle": None,      # (v2.3) 恢复净方向原始角(未折叠,偏置尸检)
        "knots_source": None,             # (v2) positions|poses|uniform_fallback
        "gt_net_content_angle": gt_net_content,
        # 分段 / 拐点 / 结点 / 形状 (完整轨迹时被命令)
        "segment_angular_errors": None,
        "segment_angular_mean": None,
        "segment_success_rate_30": None,
        "segment_success_rate_45": None,
        "turn_recall": None,
        "turn_precision": None,
        "turn_f1": None,
        "turn_timing_error": None,
        "num_gt_turns": max(0, K - 1),
        "num_recovered_turns": None,
        "waypoint_error": None,
        "waypoint_endpoint_error": None,
        "frechet": None,
        "dtw": None,
        # [RSRoll-Bench P0#6] 估计器身份与其诊断量。**无论走哪个分支都存在**，
        # 键集随分支变化会让下游聚合器 KeyError 或静默漏掉一整块。
        "path_estimator": None,
        "n_failed_steps": None,
        "failed_indices": None,
        "scale_drift": None,
        "rot_drift_deg": None,
        "degraded": None,
    }

    # [RSRoll-Bench P0#6] 轨迹估计器可插拔。
    #
    # 为什么必须可换：本模块的 12 个指标全部作用在 (T,2) 路径上，与估计器无关；
    # 但路径本身来自 Farnebäck 稠密光流，而稠密光流只在**相邻帧高度重叠**时有效。
    # 本机实测（用本文件同款 Farnebäck 参数，扫 levels∈{3,5,7}×winsize∈{15,31,63}）：
    #   帧间重叠 50%（真实位移 110 px）→ 误差 111–114 px，即**没有恢复出任何位移**；
    #   帧间重叠 10%（真实位移 200 px）→ 误差 ~212 px。配准分别是 0.09 px / 0.00 px。
    # 低重叠是新基准的卖点，估计器先崩则度量的是仪器而不是模型。
    # flow 分支原样保留：两种口径的对照表本身就是关于旧协议的一个发现，且成本为零。
    _est = (path_estimator or os.environ.get("RSWM_PATH_ESTIMATOR", "flow")).lower()
    if _est not in ("flow", "registration"):
        raise ValueError("path_estimator 只支持 flow / registration，收到 %r" % (_est,))
    out["path_estimator"] = _est

    if _est == "registration":
        import geom_registration as _GR
        _rec = _GR.recover_path_by_registration(list(gen_frames))
        out["n_failed_steps"] = _rec.n_failed_steps
        out["failed_indices"] = list(_rec.failed_indices)
        out["scale_drift"] = _rec.scale_drift
        out["rot_drift_deg"] = _rec.rot_drift_deg
        out["degraded"] = bool(_rec.failed_indices)
        # 幅度口径对齐 flow 分支：都用"每帧位移"的中位数，单位 px/frame。
        mag = _rec.median_step_px if _rec.path is not None else None
        out["flow_magnitude"] = mag
        valid = bool(_rec.valid) and (mag is not None) and (mag >= float(min_mag))
        out["valid"] = valid
        if not valid:
            return out
        # 列序：本模块的 recovered_path 口径是 (x,y) **内容**路径，
        # 而 geom_registration 原生给 (cy,cx) **相机**路径。跳过这次转换
        # 会得到一张看着合理、实则错 90°/180° 的表（实测 58° vs 0.06°）。
        rec_path = _GR.camera_path_to_content_xy(_rec.path)
    else:
        # 稠密光流只算一次,供 flow_magnitude + recover_content_path 复用 (见 docstring 效率节)。
        # None => cv2 缺失或帧数 < 2 -> 与旧行为一致:幅度 None、valid=False。
        flows = _compute_flows(gen_frames)

        # 幅度 / liveness (即便无效也报告幅度本身,便于诊断)
        mag = flow_magnitude(gen_frames, flows=flows) if flows is not None else None
        out["flow_magnitude"] = mag
        valid = (mag is not None) and (mag >= float(min_mag))
        out["valid"] = bool(valid)

        if not valid:
            return out  # 近静止 / cv2 缺失 -> 方向与路径无定义,保持 None

        # 恢复内容路径 (复用已算好的 flows,不重跑 Farneback)
        rec_path = recover_content_path(gen_frames, flows=flows)
        if rec_path is None:
            return out
    T = len(rec_path)
    out["num_frames_used"] = T

    # (v2.1) 净方向:端点法 + **GT 命令轨迹** 净位移门控。
    # 门控对象 = waypoints 折线的净位移比 —— "净方向有没有意义"由命令决定,
    # 与恢复质量无关(v2 曾用恢复路径比例门控:GT 近闭合的 00349 被错误放行
    # 污染 gt_oracle 均值,GT 明确的 00331 被错误门控)。
    # 旧 Track A 口径 (estimate_motion_direction) 已弃用,见 track_a_net_direction_error。
    if gt_net_content is not None:
        gt_ratio = polyline_net_ratio(wp)
        nd = net_direction_from_path(rec_path, gt_net_content, gt_net_ratio=gt_ratio)
        out["net_direction_error"] = nd["error_deg"]
        out["net_displacement_ratio"] = gt_ratio                       # GT 比例(门控依据)
        out["recovered_net_displacement_ratio"] = nd["recovered_net_ratio"]  # 恢复比例(诊断)
        out["net_direction_gated"] = nd["gated"]
        out["recovered_net_angle"] = nd["recovered_net_angle"]  # (v2.3) 偏置尸检
        # GT 侧角度已有现成键 gt_net_content_angle(:1540),不再另设重复键。

    if wp is not None and len(wp) >= 2:
        # ---- (v2) 真实结点: 优先 frame_positions, 其次 poses, 否则均匀回退 ----
        # 有效性要求: 逐帧结构与打分帧数一致 (len==T), 且结点数恰为 len(wp)=K+1
        # (相邻段同向会被合并 -> 数目不匹配 -> 无法与 segment_angles 配对 -> 回退)。
        knots = None
        knots_source = "uniform_fallback"
        true_positions = None  # 仅 'positions' 分支可信地重建 GT 路径形状
        if frame_positions is not None:
            fp = np.asarray(frame_positions, dtype=np.float64)
            if fp.ndim == 2 and fp.shape[1] == 2 and len(fp) == T:
                k_pos, _seg_id = knots_from_positions(fp)
                if len(k_pos) == len(wp):
                    knots, knots_source, true_positions = k_pos, "positions", fp
        if knots is None and poses is not None:
            po = np.asarray(poses, dtype=np.float64)
            if po.ndim == 3 and po.shape[0] == T and po.shape[1:] == (4, 4):
                k_po, _seg_id = knots_from_poses(po)
                if len(k_po) == len(wp):
                    # poses 只用于定结点; 形状仍走 waypoints 插值 (见 knots_from_poses)。
                    knots, knots_source = k_po, "poses"
        if knots is None:
            knots = _knot_frames(len(wp), T)
            knots_source = "uniform_fallback"
        out["knots_source"] = knots_source

        # GT content 路径: 有真实逐帧位置时用真实路径 (结点帧恰落在 waypoint 上),
        # 否则回退 waypoints 的 linspace 插值 (v1 行为)。
        if true_positions is not None:
            gt_path = _length_normalize(np.stack(
                [-(true_positions[:, 1] - true_positions[0, 1]),   # content_x = -(cx-cx0)
                 -(true_positions[:, 0] - true_positions[0, 0])],  # content_y = -(cy-cy0)
                axis=-1,
            ))
        else:
            gt_path = gt_content_path(wp, T, normalize=True)

        if K > 0:
            seg = segment_wise_angular_error(rec_path, wp, seg_ang, knot_frames=knots)
            out["segment_angular_errors"] = seg["per_segment"]
            out["segment_angular_mean"] = seg["mean"]
            # 成功率与均值正交: 均值会被单个灾难段拖垮,成功率给出达标段占比。
            out["segment_success_rate_30"] = segment_success_rate(seg["per_segment"], 30.0)
            out["segment_success_rate_45"] = segment_success_rate(seg["per_segment"], 45.0)

        # 拐点: GT 拐点 = 内部结点 knots[1:-1] (K 段 -> K-1 个拐点)。
        # (v2) knots 已是真实结点 (knots_source != 'uniform_fallback' 时),
        # 召回/时机与结点误差全部对齐真实段边界。
        gt_turns = list(int(x) for x in knots[1:-1])
        rec_turns = turn_detection(rec_path)
        out["num_recovered_turns"] = len(rec_turns)
        tr = turn_recall_and_timing(rec_turns, gt_turns)
        out["turn_recall"] = tr["recall"]
        out["turn_precision"] = tr["precision"]
        out["turn_f1"] = tr["f1"]
        out["turn_timing_error"] = tr["mean_timing_error"]

        # 结点误差 (全结点均值 + 仅终点) & 形状误差
        out["waypoint_error"] = waypoint_error(rec_path, gt_path, knots)
        out["waypoint_endpoint_error"] = waypoint_endpoint_error(rec_path, gt_path, knots)
        shape = path_shape_error(rec_path, gt_path)
        out["frechet"] = shape["frechet"]
        out["dtw"] = shape["dtw"]

    return out


# ============================================================================
# __main__ 自测
# ============================================================================

def _synthetic_video(waypoints, T, win=64, shift_gain=1.0, seed=0):
    """由 CAMERA waypoints 造一段合成"平移"视频用于端到端测试 (需要 cv2)。

    相机 = 窗口在大场景上滑动;窗口按 waypoints (cy,cx) 移动,窗口内画面内容
    朝相反方向移动 (content = -camera)。用平滑纹理保证 Farneback 可跟踪。
    返回 (T,win,win,3) uint8。
    """
    import cv2
    rng = np.random.default_rng(seed)
    wp = np.asarray(waypoints, dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(wp))
    tgt = np.linspace(0.0, 1.0, T)
    cy = np.interp(tgt, src, wp[:, 0]) * shift_gain
    cx = np.interp(tgt, src, wp[:, 1]) * shift_gain
    # 平滑低频纹理: 粗随机网格上采样
    span_y = int(np.ceil(cy.max() - cy.min())) + win + 8
    span_x = int(np.ceil(cx.max() - cx.min())) + win + 8
    Hbig, Wbig = span_y + 16, span_x + 16
    coarse = rng.random((max(8, Hbig // 12), max(8, Wbig // 12))).astype(np.float32)
    big = cv2.resize(coarse, (Wbig, Hbig), interpolation=cv2.INTER_CUBIC)
    big = np.clip((big - big.min()) / (np.ptp(big) + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    big3 = np.stack([big, big, big], axis=-1)
    y0 = 8 - int(np.floor(cy.min()))
    x0 = 8 - int(np.floor(cx.min()))
    frames = []
    for t in range(T):
        oy = int(round(y0 + cy[t]))
        ox = int(round(x0 + cx[t]))
        crop = big3[oy:oy + win, ox:ox + win]
        frames.append(crop)
    return np.stack(frames, axis=0)


def _two_leg_path(T, turn_frame, step=4.0, d1=(-1.0, 0.0), d2=(0.0, 1.0)) -> np.ndarray:
    """合成两段折线 content 路径 (纯 numpy,无需 cv2),顶点精确落在 turn_frame。

    帧 [0, turn_frame] 沿 d1 走,帧 [turn_frame, T-1] 沿 d2 走,每步长度 = step。
    构造保证 turn_detection 把拐点报在 path 索引 turn_frame 上:
    diffs[turn_frame-1] 仍是 d1、diffs[turn_frame] 已是 d2,朝向突变的顶点即 path[turn_frame]。
    默认 d1=西 (x 负), d2=南 (y 正) —— 对应 CAMERA "先东后北" 的 content 路径。
    """
    path = np.zeros((T, 2), dtype=np.float64)
    for t in range(1, T):
        d = np.asarray(d1 if t <= turn_frame else d2, dtype=np.float64)
        path[t] = path[t - 1] + d * float(step)
    return path


def _v5_positions(waypoints, steps_per_segment=9) -> np.ndarray:
    """复刻 v5 逐帧位置生成 (纯 numpy):每段独立写 steps_per_segment 帧,含两端点。

    与 build_multi_direction_benchmark_v5.py:562-579 同构:段 k 的末帧与段 k+1 的
    首帧是 **同一位置的重复帧** -> T = K*steps_per_segment,真实结点非均匀
    (2 段 sps=9 -> T=18,真实结点 [0,9,17],均匀假设给 [0,8,17])。
    (以上是 v2.8 前的 18/27/36 数据;去重后 K=2→17 帧、结点 [0,8,16],均匀恰好对)
    返回 (T,2) 的 (cy,cx)。
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    pos = []
    for k in range(len(wp) - 1):
        for step in range(steps_per_segment):
            t = step / (steps_per_segment - 1) if steps_per_segment > 1 else 0.0
            pos.append(wp[k] + t * (wp[k + 1] - wp[k]))
    return np.asarray(pos, dtype=np.float64)


def _detour_path(T, end=(-80.0, 0.0), bulge=60.0) -> np.ndarray:
    """合成"首尾正确但中途绕远"的 content 路径 (纯 numpy,无需 cv2)。

    从 (0,0) 出发,先斜插到 (end_x/2, bulge) 这个南向大鼓包,再折回 end。
    起点与终点都与直线 GT 一致 -> **净方向完全正确**,但形状完全不同。
    这是 Track B 存在的理由的直接反例构造:net_direction_error 会被骗过,
    Fréchet/DTW 不会。
    """
    P0 = np.zeros(2, dtype=np.float64)
    P2 = np.asarray(end, dtype=np.float64)
    mid = np.array([P2[0] / 2.0, float(bulge)], dtype=np.float64)
    h = T // 2
    path = np.zeros((T, 2), dtype=np.float64)
    for t in range(T):
        if t <= h:
            a = t / float(h)
            path[t] = P0 + a * (mid - P0)
        else:
            a = (t - h) / float(T - 1 - h)
            path[t] = mid + a * (P2 - mid)
    return path


def _self_test() -> int:
    failures: List[str] = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    # ---- 1) 纯几何 (numpy-only,不需要 cv2) ----
    print("\n[1/6] S1 正确的分段平移 (纯几何): gt_content_path / 分段角度 / 拐点 / 形状 / 结点误差")
    T = 17
    # 直线东: 相机东移 (cx 增大),content 应西移;segment_angles=[0]
    wp_e = np.array([[100.0, 100.0], [100.0, 180.0]])
    seg_e = [0.0]
    gt_e = gt_content_path(wp_e, T)                 # 单位归一化 content 路径
    # 构造"完美恢复路径" = 未归一化 content 路径 * 任意尺度 (测尺度不变性)
    rec_e = gt_content_path(wp_e, T, normalize=False) * 3.7
    se = segment_wise_angular_error(rec_e, wp_e, seg_e)
    check("直线东: 分段角误差≈0", se["mean"] is not None and se["mean"] < 5.0,
          f"mean={se['mean']}")
    sh_e = path_shape_error(rec_e, gt_e)
    check("直线东: Fréchet≈0", sh_e["frechet"] < 1e-6, f"frechet={sh_e['frechet']:.2e}")
    check("直线东: DTW≈0", sh_e["dtw"] < 1e-6, f"dtw={sh_e['dtw']:.2e}")
    check("直线东: 无拐点", turn_detection(rec_e) == [], f"turns={turn_detection(rec_e)}")

    # L 形: 先东后北 (相机北 = cy 减小);K=2,segment_angles=[0,90]
    wp_l = np.array([[100.0, 100.0], [100.0, 180.0], [20.0, 180.0]])
    seg_l = [0.0, 90.0]
    gt_l = gt_content_path(wp_l, T)
    rec_l = gt_content_path(wp_l, T, normalize=False) * 2.0
    sl = segment_wise_angular_error(rec_l, wp_l, seg_l)
    check("L 形: 两段角误差都≈0",
          sl["mean"] is not None and sl["mean"] < 5.0 and all(
              (e is not None and e < 5.0) for e in sl["per_segment"]),
          f"per_segment={sl['per_segment']}")
    turns_l = turn_detection(rec_l)
    knots_l = _knot_frames(len(wp_l), T)
    gt_turns_l = list(knots_l[1:-1])
    check("L 形: 恰好检出 1 个拐点", len(turns_l) == 1, f"turns={turns_l}")
    tr_l = turn_recall_and_timing(turns_l, gt_turns_l, tol=2)
    check("L 形: 拐点召回=1.0", tr_l["recall"] == 1.0, f"{tr_l}")
    check("L 形: 拐点时机误差<=2",
          tr_l["mean_timing_error"] is not None and tr_l["mean_timing_error"] <= 2,
          f"{tr_l}")
    we_l = waypoint_error(rec_l, gt_l, knots_l)
    check("L 形: 结点误差≈0", we_l is not None and we_l < 1e-6, f"waypoint_error={we_l}")
    sh_l = path_shape_error(rec_l, gt_l)
    check("L 形: Fréchet≈0", sh_l["frechet"] < 1e-6, f"frechet={sh_l['frechet']:.2e}")

    # ---- S2) 反向: 恢复路径方向与 GT 相反 -> 角误差应≈180 ----
    rec_bad = -rec_e
    se_bad = segment_wise_angular_error(rec_bad, wp_e, seg_e)
    check("S2 反向 content -> 角误差≈180",
          se_bad["mean"] is not None and se_bad["mean"] > 170.0, f"mean={se_bad['mean']}")
    check("S2 反向 -> segment_success_rate_45 = 0",
          segment_success_rate(se_bad["per_segment"], 45.0) == 0.0,
          f"ssr45={segment_success_rate(se_bad['per_segment'], 45.0)}")

    # turn_recall_and_timing 边界: GT 无拐点
    check("边界: GT 无拐点 -> recall=1.0",
          turn_recall_and_timing([], [], tol=2)["recall"] == 1.0)

    # ---- 2) 新增指标的纯单元测试 (numpy-only) ----
    print("\n[2/6] 新增指标单测: segment_success_rate / turn_precision+f1 / endpoint")

    # segment_success_rate: 阈值区分 + None 段排除出分母
    check("ssr: [10,40,100] @45 -> 2/3",
          abs(segment_success_rate([10.0, 40.0, 100.0], 45.0) - 2.0 / 3.0) < 1e-12,
          f"{segment_success_rate([10.0, 40.0, 100.0], 45.0)}")
    check("ssr: [10,40,100] @30 -> 1/3 (阈值确实生效)",
          abs(segment_success_rate([10.0, 40.0, 100.0], 30.0) - 1.0 / 3.0) < 1e-12,
          f"{segment_success_rate([10.0, 40.0, 100.0], 30.0)}")
    check("ssr: None 段不进分母 -> [10,None,100]@45 = 1/2 (非 1/3)",
          segment_success_rate([10.0, None, 100.0], 45.0) == 0.5,
          f"{segment_success_rate([10.0, None, 100.0], 45.0)}")
    check("ssr: 全 None -> None (不是 0.0)",
          segment_success_rate([None, None], 45.0) is None)
    check("ssr: 空/None 输入 -> None",
          segment_success_rate([], 45.0) is None and segment_success_rate(None, 45.0) is None)
    check("ssr: 边界严格小于 -> [45.0]@45 = 0.0",
          segment_success_rate([45.0], 45.0) == 0.0)

    # turn precision / f1 的四种退化组合 (见 turn_recall_and_timing docstring)
    t_ee = turn_recall_and_timing([], [], tol=2)
    check("turn: gt空+rec空 -> recall=1, precision=1, f1=1",
          t_ee["recall"] == 1.0 and t_ee["precision"] == 1.0 and t_ee["f1"] == 1.0, f"{t_ee}")
    t_se = turn_recall_and_timing([3], [], tol=2)
    check("turn: gt空+rec非空 -> precision=0 (全虚报), recall 仍=1",
          t_se["recall"] == 1.0 and t_se["precision"] == 0.0 and t_se["f1"] == 0.0, f"{t_se}")
    t_es = turn_recall_and_timing([], [8], tol=2)
    check("turn: gt非空+rec空 -> recall=0, precision=0 (约定), f1=0",
          t_es["recall"] == 0.0 and t_es["precision"] == 0.0 and t_es["f1"] == 0.0, f"{t_es}")
    t_sp = turn_recall_and_timing([8, 2, 14], [8], tol=2)
    check("turn: 1 命中 + 2 虚报 -> recall=1, precision=1/3, f1=0.5",
          t_sp["recall"] == 1.0 and abs(t_sp["precision"] - 1.0 / 3.0) < 1e-12
          and abs(t_sp["f1"] - 0.5) < 1e-12, f"{t_sp}")
    check("turn: 虚报会压低 precision 而 recall 察觉不到 (二者必须联读)",
          t_sp["recall"] > t_sp["precision"], f"{t_sp}")

    # waypoint_endpoint_error vs waypoint_error: 完美 L 形两者都≈0
    check("endpoint: 完美 L 形 -> 终点误差≈0",
          waypoint_endpoint_error(rec_l, gt_l, knots_l) is not None
          and waypoint_endpoint_error(rec_l, gt_l, knots_l) < 1e-6,
          f"{waypoint_endpoint_error(rec_l, gt_l, knots_l)}")

    # ---- 3/4) 拐点转早 / 转晚 ----
    print("\n[3/6] S3/S4 拐点转早 / 转晚: 松容差仍召回, 但 timing 误差 > 0 且随偏移增长")
    gt_turn_frame = int(knots_l[1])          # L 形 GT 拐点 = 中间结点 (T=17 -> 8)
    p_ontime = _two_leg_path(T, gt_turn_frame)
    p_early1 = _two_leg_path(T, gt_turn_frame - 2)   # 早 2 帧
    p_early2 = _two_leg_path(T, gt_turn_frame - 3)   # 早 3 帧
    p_late1 = _two_leg_path(T, gt_turn_frame + 2)    # 晚 2 帧
    p_late2 = _two_leg_path(T, gt_turn_frame + 3)    # 晚 3 帧
    check("S3/S4 合成路径顶点落位正确 (turn_detection 定位到构造帧)",
          turn_detection(p_ontime) == [gt_turn_frame]
          and turn_detection(p_early2) == [gt_turn_frame - 3]
          and turn_detection(p_late2) == [gt_turn_frame + 3],
          f"ontime={turn_detection(p_ontime)}, early2={turn_detection(p_early2)}, "
          f"late2={turn_detection(p_late2)}")

    LOOSE = 4  # 松容差: 足以覆盖 ±3 帧的偏移
    tr_on = turn_recall_and_timing(turn_detection(p_ontime), gt_turns_l, tol=LOOSE)
    tr_e1 = turn_recall_and_timing(turn_detection(p_early1), gt_turns_l, tol=LOOSE)
    tr_e2 = turn_recall_and_timing(turn_detection(p_early2), gt_turns_l, tol=LOOSE)
    tr_l1 = turn_recall_and_timing(turn_detection(p_late1), gt_turns_l, tol=LOOSE)
    tr_l2 = turn_recall_and_timing(turn_detection(p_late2), gt_turns_l, tol=LOOSE)

    check("S3 转早: 松容差下仍被召回 (recall=1, precision=1)",
          tr_e2["recall"] == 1.0 and tr_e2["precision"] == 1.0, f"{tr_e2}")
    check("S3 转早: timing 误差 > 0 (准时基线 = 0)",
          tr_on["mean_timing_error"] == 0.0 and tr_e2["mean_timing_error"] > 0.0,
          f"ontime={tr_on['mean_timing_error']}, early={tr_e2['mean_timing_error']}")
    check("S3 转早: timing 误差随偏移单调增长 (3帧 > 2帧 > 准时)",
          tr_e2["mean_timing_error"] > tr_e1["mean_timing_error"] > tr_on["mean_timing_error"],
          f"e2={tr_e2['mean_timing_error']} > e1={tr_e1['mean_timing_error']} "
          f"> on={tr_on['mean_timing_error']}")
    check("S4 转晚: 松容差下仍被召回 (recall=1, precision=1)",
          tr_l2["recall"] == 1.0 and tr_l2["precision"] == 1.0, f"{tr_l2}")
    check("S4 转晚: timing 误差 > 0 且随偏移单调增长 (对称于 S3)",
          tr_l2["mean_timing_error"] > tr_l1["mean_timing_error"] > tr_on["mean_timing_error"]
          and tr_l1["mean_timing_error"] > 0.0,
          f"l2={tr_l2['mean_timing_error']} > l1={tr_l1['mean_timing_error']} "
          f"> on={tr_on['mean_timing_error']}")
    check("S3/S4 转早与转晚同偏移量 -> timing 误差对称相等",
          tr_e2["mean_timing_error"] == tr_l2["mean_timing_error"],
          f"early={tr_e2['mean_timing_error']}, late={tr_l2['mean_timing_error']}")
    check("S3/S4 严容差 (tol=2) 下 ±3 帧偏移直接漏召回 -> recall=0",
          turn_recall_and_timing(turn_detection(p_early2), gt_turns_l, tol=2)["recall"] == 0.0
          and turn_recall_and_timing(turn_detection(p_late2), gt_turns_l, tol=2)["recall"] == 0.0)
    check("S3/S4 转早/转晚不影响分段净方向 (两段方向仍≈对) -> 说明 timing 是正交信息",
          segment_wise_angular_error(p_ontime, wp_l, seg_l)["mean"] < 1e-6,
          f"mean={segment_wise_angular_error(p_ontime, wp_l, seg_l)['mean']}")

    # ---- 5) 形状正确但整体平移 ----
    print("\n[4/6] S5 形状正确但整体平移: 形状指标必须平移不变 (不得惩罚纯偏移)")
    offset = np.array([37.0, -21.0])
    rec_l_shift = rec_l + offset          # 在 anchor 之前施加全局偏移
    sh_shift = path_shape_error(rec_l_shift, gt_l)
    check("S5 整体平移: Fréchet≈0 (形状未变,平移不该被惩罚)",
          sh_shift["frechet"] < 1e-6, f"frechet={sh_shift['frechet']:.3e}")
    check("S5 整体平移: DTW≈0", sh_shift["dtw"] < 1e-6, f"dtw={sh_shift['dtw']:.3e}")
    check("S5 整体平移: 与未平移结果逐位相同 (严格平移不变)",
          abs(sh_shift["frechet"] - sh_l["frechet"]) < 1e-9
          and abs(sh_shift["dtw"] - sh_l["dtw"]) < 1e-9,
          f"shift={sh_shift}, orig={sh_l}")
    # 平移 + 缩放同时施加 -> 仍只比形状
    sh_shift_scale = path_shape_error(rec_l * 5.0 + offset, gt_l)
    check("S5 平移+缩放: Fréchet/DTW 仍≈0 (尺度 & 平移双不变)",
          sh_shift_scale["frechet"] < 1e-6 and sh_shift_scale["dtw"] < 1e-6,
          f"{sh_shift_scale}")

    # ---- 6) 净方向正确但中途绕远 (Track B 的核心价值) ----
    print("\n[5/6] S6 首尾正确但中途绕远: 净方向被骗过, 形状指标必须抓住")
    knots_e = _knot_frames(len(wp_e), T)      # 单段 -> [0, T-1]
    det = _detour_path(T, end=(-80.0, 0.0), bulge=60.0)
    se_det = segment_wise_angular_error(det, wp_e, seg_e)
    check("S6 绕远: 单段净位移方向≈正确 (net-direction 等价量被骗过)",
          se_det["mean"] is not None and se_det["mean"] < 5.0, f"mean={se_det['mean']}")
    check("S6 绕远: segment_success_rate_30 = 1.0 (分段指标也被骗过)",
          segment_success_rate(se_det["per_segment"], 30.0) == 1.0,
          f"ssr30={segment_success_rate(se_det['per_segment'], 30.0)}")
    sh_det = path_shape_error(det, gt_e)
    check("S6 绕远: Fréchet 显著大于同 GT 的正确直线 (严格不等式)",
          sh_det["frechet"] > sh_e["frechet"] + 0.3,
          f"detour={sh_det['frechet']:.4f} vs straight={sh_e['frechet']:.3e}")
    check("S6 绕远: DTW 显著大于同 GT 的正确直线 (严格不等式)",
          sh_det["dtw"] > sh_e["dtw"] + 1.0,
          f"detour={sh_det['dtw']:.4f} vs straight={sh_e['dtw']:.3e}")
    check("S6 绕远: 关键演示 —— 方向≈0 度误差的同时 Fréchet 很大",
          se_det["mean"] < 5.0 and sh_det["frechet"] > 0.3,
          f"seg_mean={se_det['mean']:.3f} deg, frechet={sh_det['frechet']:.4f}")
    we_det = waypoint_error(det, gt_e, knots_e)
    wee_det = waypoint_endpoint_error(det, gt_e, knots_e)
    check("S6 绕远: 终点误差 > 全结点均值 (起点恒为 0 会稀释均值)",
          wee_det is not None and we_det is not None and wee_det > we_det,
          f"endpoint={wee_det:.4f} > mean={we_det:.4f}")

    # ---- 8) 随机方向 ----
    print("\n[6/6] S8 随机/无关方向: 分段角误差必须显著大 (>45 度)")
    # GT = 相机东 -> content 应向西 (180)。以下恢复路径与之无关。
    rnd_cases = {
        "北 (content 90 vs GT 180 -> 90 度)": np.stack(
            [np.zeros(T), -4.0 * np.arange(T)], axis=-1),
        "东 (content 0 vs GT 180 -> 180 度)": np.stack(
            [4.0 * np.arange(T), np.zeros(T)], axis=-1),
        "南 (content 270 vs GT 180 -> 90 度)": np.stack(
            [np.zeros(T), 4.0 * np.arange(T)], axis=-1),
    }
    for label, p_rnd in rnd_cases.items():
        se_rnd = segment_wise_angular_error(p_rnd, wp_e, seg_e)
        check(f"S8 随机方向 {label}: 分段角误差 > 45",
              se_rnd["mean"] is not None and se_rnd["mean"] > 45.0, f"mean={se_rnd['mean']}")
        check(f"S8 随机方向 {label}: segment_success_rate_45 = 0.0",
              segment_success_rate(se_rnd["per_segment"], 45.0) == 0.0,
              f"ssr45={segment_success_rate(se_rnd['per_segment'], 45.0)}")

    # ========================================================================
    # v2 新增场景 (全部纯 numpy,无需 cv2)
    # ========================================================================

    # ---- v2-1) 真实结点: 非均匀结点必须被精确恢复 ----
    print("\n[v2-1] 真实结点: knots_from_positions / knots_from_poses vs 均匀 linspace")
    # 2 段 v5 路径: E 72px 再 N 72px,sps=9 -> T=18,段边界重复帧 (f8==f9)
    wp_v5 = np.array([[100.0, 100.0], [100.0, 172.0], [28.0, 172.0]])
    pos_v5 = _v5_positions(wp_v5, steps_per_segment=9)
    check("v2-1 v5 路径: T = 2*9 = 18 且段边界重复帧",
          len(pos_v5) == 18 and np.allclose(pos_v5[8], pos_v5[9]),
          f"T={len(pos_v5)}, f8={pos_v5[8]}, f9={pos_v5[9]}")
    k_pos, seg_id = knots_from_positions(pos_v5)
    check("v2-1 knots_from_positions 精确恢复非均匀结点 [0,9,17]",
          list(k_pos) == [0, 9, 17], f"knots={list(k_pos)}")
    k_uni = _knot_frames(3, 18)
    check("v2-1 均匀假设确实是错的 ([0,8,17] != 真实 [0,9,17])",
          list(k_uni) == [0, 8, 17] and list(k_uni) != list(k_pos),
          f"uniform={list(k_uni)}")
    check("v2-1 per_frame_segment_id: 帧 0-8 段 0, 帧 9-17 段 1",
          list(seg_id) == [0] * 9 + [1] * 9, f"seg_id={list(seg_id)}")
    # 4 段 (E,N,W,S 闭合方框): 真实第 3 内部结点 27,均匀给 26 (服务器实测差异)
    wp_v5_4 = np.array([[100.0, 100.0], [100.0, 172.0], [28.0, 172.0],
                        [28.0, 100.0], [100.0, 100.0]])
    pos_v5_4 = _v5_positions(wp_v5_4, steps_per_segment=9)
    k4, _ = knots_from_positions(pos_v5_4)
    k4_uni = _knot_frames(5, len(pos_v5_4))
    check("v2-1 4 段: 真实结点 [0,9,18,27,35],第 3 内部结点 27 vs 均匀 26",
          list(k4) == [0, 9, 18, 27, 35] and int(k4_uni[3]) == 26,
          f"real={list(k4)}, uniform={list(k4_uni)}")
    # poses 平移列 (up-to-scale, 各向异性归一化) -> 结点不变
    poses_v5 = np.tile(np.eye(4, dtype=np.float64), (18, 1, 1))
    poses_v5[:, 0, 3] = (pos_v5[:, 1] - pos_v5[0, 1]) / 222.0   # tx = (cx-cx0)/win_w
    poses_v5[:, 1, 3] = (pos_v5[:, 0] - pos_v5[0, 0]) / 128.0   # ty = (cy-cy0)/win_h
    k_po, _ = knots_from_poses(poses_v5)
    check("v2-1 knots_from_poses (各向异性 up-to-scale) 结点不变 [0,9,17]",
          list(k_po) == [0, 9, 17], f"knots={list(k_po)}")
    # 整数像素舍入抖动 (~9px 步长斜线) 不得产生假结点
    pos_diag = np.array([[int(100 - 4.5 * t), int(100 + 7.794 * t)]
                         for t in range(10)], dtype=np.float64)
    k_diag, seg_diag = knots_from_positions(pos_diag)
    check("v2-1 整数舍入抖动斜线: 无假内部结点 ([0,9])",
          list(k_diag) == [0, 9] and set(seg_diag.tolist()) == {0},
          f"knots={list(k_diag)}")

    # ---- v2-2) 加窗拐点检测: 重复帧 / 缓转 / NMS ----
    print("\n[v2-2] 加窗拐点: 段边界重复帧与跨帧缓转必须被检出, NMS 去重")
    # v5 真实 content 路径 (含零步): 8 步西, 1 零步, 8 步南, 真实结点 9
    cont_v5 = np.stack([-(pos_v5[:, 1] - pos_v5[0, 1]),
                        -(pos_v5[:, 0] - pos_v5[0, 0])], axis=-1)
    turns_dup = turn_detection(cont_v5)
    check("v2-2 重复帧拐角: 恰检出 1 个拐点且落在真实结点 9 的 ±1 帧内"
          " (v1 相邻单步法因零步打断相邻性而漏检)",
          len(turns_dup) == 1 and abs(turns_dup[0] - 9) <= 1, f"turns={turns_dup}")
    tr_dup = turn_recall_and_timing(turns_dup, [9], tol=2)
    check("v2-2 重复帧拐角: 对真实结点 recall=1 且 timing<=1",
          tr_dup["recall"] == 1.0 and tr_dup["mean_timing_error"] is not None
          and tr_dup["mean_timing_error"] <= 1.0, f"{tr_dup}")
    # 跨 8 帧的缓转 (每步仅 10 度 < min_turn_deg=15): v1 逐步比较必然全漏
    slow_heads = [180.0] * 4 + [190.0, 200.0, 210.0, 220.0,
                                230.0, 240.0, 250.0, 260.0] + [270.0] * 4
    slow_steps = np.stack(
        [9.0 * np.cos(np.radians(slow_heads)),
         -9.0 * np.sin(np.radians(slow_heads))], axis=-1)  # dx=cos, dy=-sin (y 下正)
    slow_path = np.zeros((len(slow_steps) + 1, 2))
    slow_path[1:] = np.cumsum(slow_steps, axis=0)
    turns_slow = turn_detection(slow_path)
    check("v2-2 缓转 (逐步 10 度): 加窗后检出 >=1 个拐点且都落在缓转区 [4,12]",
          len(turns_slow) >= 1 and all(4 <= t <= 12 for t in turns_slow),
          f"turns={turns_slow}")
    # NMS: 干净直角在窗口平滑下会产生多帧高分候选,必须只留 1 个
    check("v2-2 NMS: 干净直角只报 1 个拐点 (窗口平滑的多帧候选被去重)",
          turn_detection(_two_leg_path(18, 9)) == [9],
          f"turns={turn_detection(_two_leg_path(18, 9))}")

    # ---- v2-3) 转早/转晚: 时机误差对齐真实结点 (而非均匀假设) ----
    print("\n[v2-3] 转早/转晚 vs 真实结点: 均匀结点会把准时误判成早 1 帧")
    real_turns_v5 = [9]           # 来自 knots_from_positions(pos_v5)[1:-1]
    uni_turns_v5 = [8]            # 均匀假设的内部结点
    p_on18 = _two_leg_path(18, 9)     # 准时: 恰在真实结点 9 处转
    p_e2_18 = _two_leg_path(18, 7)    # 早 2 帧
    p_l2_18 = _two_leg_path(18, 11)   # 晚 2 帧
    tr_on_real = turn_recall_and_timing(turn_detection(p_on18), real_turns_v5, tol=4)
    tr_on_uni = turn_recall_and_timing(turn_detection(p_on18), uni_turns_v5, tol=4)
    tr_e2_real = turn_recall_and_timing(turn_detection(p_e2_18), real_turns_v5, tol=4)
    tr_l2_real = turn_recall_and_timing(turn_detection(p_l2_18), real_turns_v5, tol=4)
    check("v2-3 准时路径 vs 真实结点: timing=0",
          tr_on_real["recall"] == 1.0 and tr_on_real["mean_timing_error"] == 0.0,
          f"{tr_on_real}")
    check("v2-3 同一路径 vs 均匀结点: timing=1 (均匀假设自带 1 帧系统偏差)",
          tr_on_uni["mean_timing_error"] == 1.0, f"{tr_on_uni}")
    check("v2-3 早/晚 2 帧 vs 真实结点: timing 均为 2 且对称",
          tr_e2_real["mean_timing_error"] == 2.0
          and tr_l2_real["mean_timing_error"] == 2.0,
          f"early={tr_e2_real['mean_timing_error']}, late={tr_l2_real['mean_timing_error']}")

    # ---- v2-4) 端点净方向 + [v2.1] GT 命令轨迹门控 ----
    print("\n[v2-4] 端点净方向: v2.1 门控由 GT 命令轨迹决定(与恢复质量无关)")
    T18 = 18
    path_east = np.stack([4.0 * np.arange(T18), np.zeros(T18)], axis=-1)
    # (legacy 兼容分支: 不传 gt_net_ratio 时退回恢复比例门控)
    nd_ok = net_direction_from_path(path_east, 0.0)
    check("v2-4 直线东 vs GT 0 度: error≈0, rec_ratio≈1, 不 gated",
          nd_ok["gated"] is False and nd_ok["error_deg"] is not None
          and nd_ok["error_deg"] < 1e-9 and nd_ok["recovered_net_ratio"] > 0.99,
          f"{nd_ok}")
    nd_rev = net_direction_from_path(path_east, 180.0)
    check("v2-4 直线东 vs GT 180 度: error≈180",
          nd_rev["error_deg"] is not None and nd_rev["error_deg"] > 179.0,
          f"{nd_rev}")
    check("v2-4 退化 (T<2): gated=True + None",
          net_direction_from_path(np.zeros((1, 2)), 0.0)["error_deg"] is None
          and net_direction_from_path(np.zeros((1, 2)), 0.0)["gated"] is True)
    # 近闭合方框恢复路径(净位移≈0),备用于两个 v2.1 案例
    sq_steps = ([( 4.0, 0.0)] * 4 + [(0.0,  4.0)] * 4
                + [(-4.0, 0.0)] * 4 + [(0.0, -4.0)] * 4)
    sq_path = np.zeros((17, 2))
    sq_path[1:] = np.cumsum(np.asarray(sq_steps), axis=0)
    # [v2.1-案例 00349 形态] GT 近闭合 (ratio=0.054) + 恢复路径笔直 (rec≈1):
    # v2 曾错误放行(147.36° 垃圾值污染 gt_oracle 均值);v2.1 必须门控。
    nd_349 = net_direction_from_path(path_east, 0.0, gt_net_ratio=0.054)
    check("v2.1 GT近闭合+恢复笔直(00349形态): 必须 gated=True, error=None",
          nd_349["gated"] is True and nd_349["error_deg"] is None
          and nd_349["recovered_net_ratio"] > 0.99
          and nd_349["gt_net_ratio"] == 0.054, f"{nd_349}")
    # [v2.1-案例 00331 形态] GT 明确 (ratio=0.216) + 恢复近闭合 (rec≈0):
    # v2 曾错误门控;v2.1 不得门控 —— 恢复净位移严格退化时如实 None
    # (recovered_net_degenerate=True,语义≠门控:是"恢复不出",不是"问题无意义")。
    nd_331 = net_direction_from_path(sq_path, 0.0, gt_net_ratio=0.216)
    check("v2.1 GT明确+恢复近闭合(00331形态): gated=False(不因恢复差而门控)",
          nd_331["gated"] is False and nd_331["gt_net_ratio"] == 0.216,
          f"{nd_331}")
    check("v2.1 00331形态: 恢复净位移严格退化 -> error=None + degenerate 标志",
          nd_331["error_deg"] is None
          and nd_331["recovered_net_degenerate"] is True, f"{nd_331}")
    # GT 明确 + 恢复非退化但净位移比小(有噪绕行): 必须给数(即使误差大)
    wobble = np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 6.0], [16.0, 6.0],
                       [16.0, 0.0], [24.0, 0.0]])   # 净=24 弧长=8+6+8+6+8=36 rec≈0.667
    nd_wob = net_direction_from_path(wobble, 0.0, gt_net_ratio=0.9)
    check("v2.1 GT明确+恢复绕行: 不 gated 且给出误差(端点向东 error≈0)",
          nd_wob["gated"] is False and nd_wob["error_deg"] is not None
          and nd_wob["error_deg"] < 1e-9, f"{nd_wob}")
    # polyline_net_ratio 自身: 直线=1, 闭合≈0, L形≈0.707
    check("v2.1 polyline_net_ratio: 直线=1",
          abs(polyline_net_ratio(np.array([[0., 0.], [0., 10.]])) - 1.0) < 1e-12)
    check("v2.1 polyline_net_ratio: 闭合≈0",
          polyline_net_ratio(np.vstack([sq_path, sq_path[:1]])) < 0.05)
    check("v2.1 polyline_net_ratio: L形≈0.707",
          abs(polyline_net_ratio(np.array([[0., 0.], [0., 10.], [10., 10.]]))
              - math.sqrt(0.5)) < 1e-9)
    # L 形 (GT 净位移比 ~0.707) 不应被门控 —— v2.1 用 GT 比例传入
    nd_l = net_direction_from_path(
        rec_l, (_camera_net_angle(wp_l) + 180.0) % 360.0,
        gt_net_ratio=polyline_net_ratio(wp_l))
    check("v2-4 L 形: GT ratio≈0.707 不 gated 且 error≈0",
          nd_l["gated"] is False
          and abs(nd_l["gt_net_ratio"] - math.sqrt(0.5)) < 0.01
          and nd_l["error_deg"] < 1e-6, f"{nd_l}")

    # ---- v2-5) 降采样路径的尺度不变性 (窗口分辨率 vs scene 像素) ----
    print("\n[v2-5] 窗口分辨率恢复路径 vs scene 像素 GT: 形状/角度误差必须≈0")
    # 480x832 -> 128x222 的各向异性缩放系数 (两轴差 0.06%,近各向同性 by design)
    scale = np.array([222.0 / 832.0, 128.0 / 480.0])
    rec_l_win = gt_content_path(wp_l, 17, normalize=False) * scale[None, :]
    sh_win = path_shape_error(rec_l_win, gt_l)
    check("v2-5 窗口分辨率路径: Fréchet < 0.01 (归一化吸收缩放)",
          sh_win["frechet"] < 0.01, f"frechet={sh_win['frechet']:.2e}")
    se_win = segment_wise_angular_error(rec_l_win, wp_l, seg_l)
    check("v2-5 窗口分辨率路径: 分段角误差 < 1 度 (0.06% 各向异性可忽略)",
          se_win["mean"] is not None and se_win["mean"] < 1.0,
          f"mean={se_win['mean']}")
    # 真实结点 + 真实 GT 路径下的完美恢复: 全指标≈0/满分 (v2 主链路自洽)
    rec_v5_perfect = cont_v5 * 0.2667  # 窗口分辨率下的完美恢复路径
    se_v5 = segment_wise_angular_error(rec_v5_perfect, wp_v5, [0.0, 90.0],
                                       knot_frames=k_pos)
    check("v2-5 v5 真实结点 + 完美恢复: 逐段角误差≈0",
          se_v5["mean"] is not None and se_v5["mean"] < 1e-6,
          f"per={se_v5['per_segment']}")
    gt_v5_norm = _length_normalize(cont_v5)
    we_v5 = waypoint_error(rec_v5_perfect, gt_v5_norm, k_pos)
    check("v2-5 v5 真实结点: waypoint_error≈0 (结点帧恰落在真值 waypoint 上)",
          we_v5 is not None and we_v5 < 1e-9, f"we={we_v5}")
    try:
        segment_wise_angular_error(rec_v5_perfect, wp_v5, [0.0, 90.0],
                                   knot_frames=[0, 5, 9, 17])
        _mismatch_raised = False
    except ValueError:
        _mismatch_raised = True
    check("v2-5 knot_frames 数目错配必须 fail-loud (ValueError)", _mismatch_raised)

    # ---- 端到端 (需要 cv2 的 Farneback) ----
    print("\n[E2E] 端到端: 合成平移视频 -> recover_content_path / score_trajectory")
    try:
        import cv2  # noqa: F401
        have_cv2 = True
    except ImportError:
        have_cv2 = False
    if not have_cv2:
        print("  [SKIP] cv2 不可用 (本沙箱无 opencv);端到端光流测试在装有 cv2 的真实栈上运行。")
    else:
        # 直线东: 窗口按 cx 增大移动 (相机东),内容西移
        cam_e = np.array([[128.0, 20.0], [128.0, 20.0 + 8.0 * (T - 1)]])  # ~8px/frame
        vid_e = _synthetic_video(cam_e, T, win=72, seed=1)
        mag_e = flow_magnitude(vid_e)
        check("直线东视频: flow_magnitude>5", (mag_e is not None) and mag_e > 5.0,
              f"mag={mag_e:.2f}" if mag_e else "None")
        res_e = score_trajectory(vid_e, cam_e, seg_e)
        check("直线东视频: valid=True", res_e["valid"] is True)
        check("直线东视频: 净方向误差<30",
              res_e["net_direction_error"] is not None and res_e["net_direction_error"] < 30.0,
              f"net={res_e['net_direction_error']}")
        check("直线东视频: 分段角误差<30",
              res_e["segment_angular_mean"] is not None and res_e["segment_angular_mean"] < 30.0,
              f"seg_mean={res_e['segment_angular_mean']}")

        # L 形视频: 先东 (cx 增) 后北 (cy 减)
        half = 8.0 * (T // 2)
        cam_l = np.array([
            [128.0, 20.0],
            [128.0, 20.0 + half],
            [128.0 - 8.0 * (T - 1 - T // 2), 20.0 + half],
        ])
        vid_l = _synthetic_video(cam_l, T, win=88, seed=2)
        res_l = score_trajectory(vid_l, cam_l, [0.0, 90.0])
        check("L 形视频: valid=True", res_l["valid"] is True)
        check("L 形视频: 分段角误差<35",
              res_l["segment_angular_mean"] is not None and res_l["segment_angular_mean"] < 35.0,
              f"seg_mean={res_l['segment_angular_mean']}, per={res_l['segment_angular_errors']}")
        check("L 形视频: 至少检出 1 个拐点",
              res_l["num_recovered_turns"] is not None and res_l["num_recovered_turns"] >= 1,
              f"n_turns={res_l['num_recovered_turns']}, recall={res_l['turn_recall']}")
        check("L 形视频: num_gt_turns = K-1 = 1 (K 段路径恰有 K-1 个拐点)",
              res_l["num_gt_turns"] == 1, f"{res_l['num_gt_turns']}")
        # 新增指标必须出现在扁平输出里且取值合法
        check("L 形视频: segment_success_rate_45 >= _30 且都在 [0,1]",
              res_l["segment_success_rate_45"] is not None
              and res_l["segment_success_rate_30"] is not None
              and 0.0 <= res_l["segment_success_rate_30"] <= 1.0
              and 0.0 <= res_l["segment_success_rate_45"] <= 1.0
              and res_l["segment_success_rate_45"] >= res_l["segment_success_rate_30"],
              f"30={res_l['segment_success_rate_30']}, 45={res_l['segment_success_rate_45']}")
        check("L 形视频: turn_precision / turn_f1 已输出且在 [0,1]",
              res_l["turn_precision"] is not None and res_l["turn_f1"] is not None
              and 0.0 <= res_l["turn_precision"] <= 1.0 and 0.0 <= res_l["turn_f1"] <= 1.0,
              f"p={res_l['turn_precision']}, f1={res_l['turn_f1']}")
        check("L 形视频: waypoint_endpoint_error 已输出且非负",
              res_l["waypoint_endpoint_error"] is not None
              and res_l["waypoint_endpoint_error"] >= 0.0,
              f"wee={res_l['waypoint_endpoint_error']}")

        # ---- v2 E2E: 窗口分辨率降采样 + 端点净方向 + 结点来源标注 ----
        check("E2E v2: 未给逐帧结构 -> knots_source='uniform_fallback'",
              res_l["knots_source"] == "uniform_fallback",
              f"src={res_l['knots_source']}")
        check("E2E v2: 端点净方向未被门控 (直线视频 ratio 应接近 1)",
              res_e["net_direction_gated"] is False
              and res_e["net_displacement_ratio"] is not None
              and res_e["net_displacement_ratio"] > 0.5,
              f"gated={res_e['net_direction_gated']}, "
              f"ratio={res_e['net_displacement_ratio']}")
        res_e_win = score_trajectory(vid_e, cam_e, seg_e, window_size=(36, 36))
        check("E2E v2 降采样 72->36: valid=True 且幅度低于全分辨率 (位移随分辨率缩)",
              res_e_win["valid"] is True and res_e_win["flow_magnitude"] is not None
              and mag_e is not None and res_e_win["flow_magnitude"] < mag_e,
              f"win_mag={res_e_win['flow_magnitude']}, full_mag={mag_e}")
        check("E2E v2 降采样: 净方向误差仍 <30 (降采样保方向)",
              res_e_win["net_direction_error"] is not None
              and res_e_win["net_direction_error"] < 30.0,
              f"net={res_e_win['net_direction_error']}")
        fp_e = np.stack([np.full(T, 128.0), 20.0 + 8.0 * np.arange(T)], axis=-1)
        res_e_fp = score_trajectory(vid_e, cam_e, seg_e, window_size=(36, 36),
                                    frame_positions=fp_e)
        check("E2E v2 frame_positions: knots_source='positions'",
              res_e_fp["knots_source"] == "positions",
              f"src={res_e_fp['knots_source']}")
        pix_l = np.stack(
            [np.interp(np.linspace(0.0, 1.0, T), np.linspace(0.0, 1.0, len(cam_l)),
                       cam_l[:, 0]),
             np.interp(np.linspace(0.0, 1.0, T), np.linspace(0.0, 1.0, len(cam_l)),
                       cam_l[:, 1])], axis=-1)
        poses_e2e = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
        poses_e2e[:, 0, 3] = (pix_l[:, 1] - pix_l[0, 1]) / 88.0
        poses_e2e[:, 1, 3] = (pix_l[:, 0] - pix_l[0, 0]) / 88.0
        res_l_po = score_trajectory(vid_l, cam_l, [0.0, 90.0], poses=poses_e2e)
        check("E2E v2 poses: knots_source='poses' 且分段角误差仍 <35",
              res_l_po["knots_source"] == "poses"
              and res_l_po["segment_angular_mean"] is not None
              and res_l_po["segment_angular_mean"] < 35.0,
              f"src={res_l_po['knots_source']}, "
              f"seg_mean={res_l_po['segment_angular_mean']}")

    # ---- S7) persistence / 静止 ----
    print("\n[S7] persistence/静止 + text-only 标注: valid=False, commanded=False")
    res_txt = score_trajectory(
        np.zeros((T, 32, 32, 3), dtype=np.uint8), wp_e, seg_e, text_only=True
    )
    check("text-only: commanded=False", res_txt["commanded"] is False)
    check("S7 全零(persistence)视频: valid=False (近静止)", res_txt["valid"] is False)
    # 静止 -> 所有方向/路径字段必须是 None,绝不能填 0 而被误读成"完美"
    _static_none_keys = [
        "net_direction_error", "segment_angular_errors", "segment_angular_mean",
        "segment_success_rate_30", "segment_success_rate_45",
        "turn_recall", "turn_precision", "turn_f1", "turn_timing_error",
        "num_recovered_turns", "waypoint_error", "waypoint_endpoint_error",
        "frechet", "dtw",
        # v2: 静止/无效样本连"净方向是否被门控"都无从谈起 -> 同样必须是 None
        "net_displacement_ratio", "recovered_net_displacement_ratio",
        "net_direction_gated", "knots_source",
    ]
    _leaked = [k for k in _static_none_keys if res_txt[k] is not None]
    check("S7 静止: 全部方向/路径字段为 None (不得填 0 冒充完美)",
          not _leaked, f"非 None 的字段={_leaked}")
    check("S7 静止: 新增指标 key (含 v2) 确实存在于扁平输出中",
          all(k in res_txt for k in ("segment_success_rate_30", "segment_success_rate_45",
                                     "turn_precision", "turn_f1", "waypoint_endpoint_error",
                                     "net_displacement_ratio",
                                     "recovered_net_displacement_ratio",
                                     "net_direction_gated",
                                     "knots_source")))

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s) — {failures}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
