#!/usr/bin/env python3
# =============================================================================
# baseline_outpaint.py — 平凡 outpainting 基线(难度**下界**,role=trivial_baseline)
#
# 为什么必须有这个文件
# --------------------
# RSWM 的样本是"在一张静态卫星图上滑窗"构造出来的。审稿人一定会问:
#   "那这不就是 2D 平移外推吗?一个 crop-and-translate 脚本能不能平凡解决?"
# 这是一个**开放靶子** —— 不能靠一句"我们认为不能"打发,必须自己先跑出来。
# 本文件就是那个脚本:**不含任何神经网络**,只用平移 + 补边,吃与 LoRA 臂
# 完全相同的条件(首帧 + 方向命令 + 帧数),产出同格式产物,进同一套指标。
#
# 它的语义是**难度下界**,不是方法:
#   * 如果它拿到接近 LoRA 的分数 → 这个 benchmark 的该项任务是平凡的,
#     必须在论文里说出来并改任务设计。这件事宁可自己先发现,也不要审稿人发现。
#   * 如果它在某些轴上就是拿不到分 → 那些轴才是 benchmark 的真实难度来源,
#     headline 应该建立在它们上面。
# 读数规则**在跑之前**就钉死(见 PRE_REGISTERED_READING),并随 manifest 落盘,
# 免得看到数字之后再挑一个对自己有利的解释。
#
# 与既有 diagnostic_baselines.py 的分工
# ------------------------------------
#   diagnostic_baselines.deterministic_translation —— 需要**整张场景图**,是
#       "在 GT 场景里查表",回答的是"指标上界在哪"。它看得见模型看不见的东西。
#   本文件                                         —— 只看**首帧**,新暴露的
#       区域必须凭空补出来。这才是审稿人说的"crop-and-translate 脚本",也才是
#       与生成模型可比的对照。
#   diagnostic_baselines.image_outpainting_baseline —— 用 diffusion inpainter 补边,
#       要 GPU + 权重。本文件是它的**零依赖版**:纯 numpy/cv2,纯 CPU,离线可跑。
#
# 两个变体(都不需要网络)
# ----------------------
#   shift_reflect : 平移 + 镜像补边(np.pad reflect)。纹理连续、光流估计干净,
#                   是"外推得尽量像"的那一端。
#   shift_inpaint : 平移 + cv2.inpaint(TELEA)补边。缺 cv2 时**退化为 reflect**
#                   并在 meta 里写 outpaint_fill_degraded=true(绝不静默冒充)。
#
# 为什么是"直接平移"而不是"逐帧递归"
# ----------------------------------
# 递归(每帧从上一帧再平移一点、补一条窄带)会让补边误差逐帧累积。直接从**首帧**
# 平移到第 t 帧的位移,则仍然可见的那部分用的是**原始真实像素**,只有新暴露的
# 区域是编的 —— 在 SSIM/LPIPS/FVD 上**严格不劣于**递归版。既然本基线的用途是
# 求难度下界,就应该给它最有利的算法;递归版只会更差,不可能推翻"下界"的结论。
#
# 位移(速度)从哪来 —— 故意给它开后门
# ----------------------------------
# 默认 --step-px 不传时,每帧位移由**GT 轨迹**反推(net 模式用净位移/(T-1))。
# 这是 construction-aware 的作弊:真实的 crop-and-translate 脚本并不知道相机
# 走多快。之所以这么给,是因为下界要的是"最有利情形":连速度都白送了还赢不了,
# 才叫"任务不平凡"。想跑真·盲测就显式传 --step-px。
#
# 坐标与角度约定(与全仓一致,勿凭记忆改)
# --------------------------------------
#   waypoints [K+1,2] = (cy, cx) = **窗口中心**在 scene 像素里的路径。
#   相机角 = degrees(atan2(-dy, dx)) % 360;east=0/north=90/west=180/south=270。
#   **内容光流 = 相机运动取反**(窗口东移 → 画面内容西移),
#   内容角 = 相机角 + 180。evaluate.py 落进 meta 的 direction_angle 已经是
#   **内容角**(direction_target_from_waypoints:量化到四基向后 +180),
#   所以本文件直接拿它当平移方向,**不要再翻 180**。
#   帧像素 ≠ scene 像素:帧是窗口的上采样(480x832 ← 128x222),位移必须按
#   (H/win_h, W/win_w) 缩放,否则速度差 3.75 倍。
#
# 产物 = 与 evaluate.py 逐字段同构的 eval 目录
# -------------------------------------------
#   <out>/generated/<key>.npy   (T,H,W,3) uint8
#   <out>/gt/<key>.npy          (T,H,W,3) uint8
#   <out>/meta/<key>.json       字段与 evaluate.py:470-507 一一对应 + outpaint_*
#   <out>/run_manifest.json     role=trivial_baseline,headline_eligible 恒 False
# 因此可以直接被:
#   python evaluate.py --metrics_only --output_dir <out> --device cpu
#   python server_run/run_trajectory_eval.py --eval-dirs outpaint=<out> ...
# 吃掉,不需要任何新的评估代码。
#
# 用法
# ----
#   # 真跑(离线、纯 CPU)
#   python baseline_outpaint.py --data_path train.h5 --split_file split.json \
#       --output_dir $WORK/eval_outpaint_reflect --variant shift_reflect
#   # 不需要 H5 的自检 + 打分演示
#   python baseline_outpaint.py --self-test
# =============================================================================
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if (HERE / "wan").is_dir():
    PROJECT_ROOT = HERE
else:
    PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 复用 diagnostic_baselines 的 padding / 插值语义 —— **刻意**不另写一份:
# _crop_window_reflect 的 pad 规则(reflect,pad 过大时退 edge)和
# _interpolate_positions 的 linspace 结点必须与诊断基线、与建库器逐位一致,
# 否则两个基线的"同一条 GT 轨迹"会走出两条不同的路,对照失去意义。
from diagnostic_baselines import (          # noqa: E402
    _crop_window_reflect,
    _interpolate_positions,
    _to_uint8,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger("baseline_outpaint")

METHOD_VERSION = "outpaint-trivial-1"

# role 必须永不进 headline。注意 run_manifest._NEVER_HEADLINE_ROLES 目前**不含**
# 这个名字(它只有 diagnostic_stationary/probe_gate/legacy/variant_attribution),
# 所以 build_manifest() 在 official+drop 环境下会给它算出 eligible=True。
# 本文件因此在写盘前**显式覆盖**为 False(见 _build_baseline_manifest);
# 同时应由主协调者把 "trivial_baseline" 加进那个 frozenset —— 两道防线。
ROLE = "trivial_baseline"

VARIANTS = ("shift_reflect", "shift_inpaint")
TRAJECTORY_SOURCES = ("net_direction", "waypoints")

# 每个 trajectory_source 对应的位姿口径。net_direction 只用了 prompt 里的方向词,
# 位姿通路根本没接 → "drop"/"dropped"(与 pose_mode=drop 的真 text-only 同义);
# waypoints 变体把 GT 折线整条喂给了平移,轨迹是**被命令**的 → "native"。
# 这个值决定 run_trajectory_eval 把逐段/拐点/waypoint 指标标成 commanded 与否,
# 标错等于"拿模型没收到的命令算它的服从度",所以由**基线自己实际用了什么**决定,
# 不继承环境变量。
_TRAJ_TO_POSE = {
    "net_direction": ("drop", "dropped"),
    "waypoints": ("native", "native"),
}

# -----------------------------------------------------------------------------
# 预注册的读数规则 —— 在看到任何数字**之前**写死,随 manifest 落盘。
# 目的:防止事后挑一个对自己有利的解释("你看角误差它也不低嘛")。
# -----------------------------------------------------------------------------
PRE_REGISTERED_READING = {
    "这是什么": (
        "非生成式的平凡基线:首帧 + 方向词 + 帧数 → 平移 + 补边。它是任务难度的"
        "**下界**,不是可部署方法,也不是 SOTA 对照。"),
    "已知的地板(读任何角度数字前必须先扣掉)": {
        "四方向 prompt 的量化地板": "≈22.5°(命令只有四个基向,GT 净方向是连续角)",
        "光流估计器在 GT 上的实测地板": "≈25°(仪器噪声,不是模型的错)",
        "随机方向的期望": "90°",
    },
    "读数规则(跑之前钉死)": [
        "R0 本基线的方向是**构造给的**(平移方向直接取命令角),所以它的 Track A "
        "角误差必然贴着地板。『outpaint 角误差低』本身不构成任何结论,必须与 "
        "R2/R3 联读。",
        "R1 若 outpaint 与 LoRA 臂的 Track A 角误差之差 < 5°(取 5° 是因为它相对"
        "22.5° 量化地板/25° 仪器地板是小量),则判定【净方向合规】这一项被平凡解:"
        "论文不得把它当主要贡献,只能作为 sanity check。",
        "R2 若 outpaint 与 LoRA 臂在 SSIM / LPIPS / FVD 上的差落在各自 bootstrap "
        "CI 的重叠区内,则判定【外观保真】也被平凡解 —— 这是最坏情形,必须在论文"
        "里明说,并重新设计任务(加大位移带、引入窗口外新出现的地物)。",
        "R3 若 outpaint 的 Track B turn_f1 / segment_success_rate_30 明显低于 "
        "LoRA 臂(差值超过 LoRA 臂自身的种子间标准差),则判定【多段轨迹服从】"
        "不是平移可解的 —— 这是 benchmark 的真实难度来源,headline 以 Track B 为主。",
        "R4 shift_reflect 与 shift_inpaint 的差,量的是『补边质量』这一维度本身"
        "有多大空间;两者若几乎无差,说明保真指标对新暴露区域的内容不敏感,"
        "该指标不适合当 headline。",
    ],
    "故意给基线开的后门(使结论偏保守)": [
        "--step-px 不传时,每帧位移由 GT 轨迹反推 —— 真实的 crop-and-translate "
        "脚本不知道相机走多快,这里白送给它。",
        "--trajectory-source waypoints 时,整条 GT 折线(含所有转弯)都白送给它。",
        "直接从首帧平移(而非逐帧递归),仍可见区域用的是原始真实像素,不累积误差。",
    ],
}


# =============================================================================
# 纯图像操作:平移 + 补边
# =============================================================================

def shift_reflect(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """把 frame 的内容平移 (dx 东正, dy 南正) 像素,空出来的边用镜像补。

    等价于"以 (H//2 - dy, W//2 - dx) 为中心、在 reflect-pad 过的 frame 上裁一个
    同尺寸窗口" —— 直接复用 diagnostic_baselines._crop_window_reflect,保证 pad
    语义与诊断基线一字不差(pad 过大时它会退回 edge 模式,不会抛异常)。

    Args:
        frame: (H,W,3) uint8/float。
        dx, dy: **整数**像素位移(调用方先取整;传浮点会因 round(H//2 - dy) 在
                恰好 .5 处走 banker's rounding 而出现 ±1 的不确定性)。
    Returns:
        (H,W,3) uint8。
    """
    f = _to_uint8(np.asarray(frame))
    if f.ndim != 3 or f.shape[2] != 3:
        raise ValueError(f"frame 必须是 (H,W,3);得到 {f.shape}")
    if int(dx) != dx or int(dy) != dy:
        raise ValueError(f"dx/dy 必须是整数像素;得到 ({dx}, {dy})")
    H, W = f.shape[:2]
    return _crop_window_reflect(f, H // 2 - int(dy), W // 2 - int(dx), H, W)


def shift_inpaint(frame: np.ndarray, dx: int, dy: int,
                  radius: int = 3) -> Tuple[np.ndarray, bool]:
    """把内容平移 (dx, dy),新暴露的区域用 cv2.inpaint(TELEA)补。

    Returns:
        (frame_out (H,W,3) uint8, degraded: bool)
        degraded=True 表示这一帧实际走的是 reflect 补边 —— 两种情况:
          (a) 没有 cv2;
          (b) 位移已超过整幅画面,一个原始像素都不剩(mask 全 255),
              cv2.inpaint 对全洞输入没有任何可参考内容,只会吐黑图。
        两种情况都**如实标记**,绝不让 reflect 的结果冒充 inpaint 的结果。
    """
    f = _to_uint8(np.asarray(frame))
    if int(dx) != dx or int(dy) != dy:
        raise ValueError(f"dx/dy 必须是整数像素;得到 ({dx}, {dy})")
    dx, dy = int(dx), int(dy)
    H, W = f.shape[:2]
    if abs(dx) >= W or abs(dy) >= H:
        return shift_reflect(f, dx, dy), True
    try:
        import cv2
    except ImportError:
        return shift_reflect(f, dx, dy), True

    canvas = np.zeros_like(f)
    mask = np.full((H, W), 255, np.uint8)      # 255 = 待补
    x0d, y0d = max(0, dx), max(0, dy)          # 目标位置
    x0s, y0s = max(0, -dx), max(0, -dy)        # 源位置
    w, h = W - abs(dx), H - abs(dy)
    canvas[y0d:y0d + h, x0d:x0d + w] = f[y0s:y0s + h, x0s:x0s + w]
    mask[y0d:y0d + h, x0d:x0d + w] = 0
    out = cv2.inpaint(canvas, mask, float(radius), cv2.INPAINT_TELEA)
    # 已知区域强制保留原像素:inpaint 理论上不动 mask==0 的地方,但显式钉住
    # 更安全 —— 补边基线的卖点就是"可见部分是真的",这一点不能靠实现细节保证。
    keep = mask == 0
    out[keep] = canvas[keep]
    return _to_uint8(out), False


def outpaint_video(first_frame: np.ndarray,
                   offsets: np.ndarray,
                   variant: str = "shift_reflect") -> Tuple[np.ndarray, int]:
    """按逐帧累计位移把首帧外推成一段视频。

    Args:
        first_frame: (H,W,3) 首帧(= 模型拿到的 i2v 条件图)。
        offsets: (T,2) **累计**内容位移 (dx 东正, dy 南正),单位 = 帧像素。
                 offsets[0] 必须是 (0,0)(第 0 帧就是首帧本身)。
        variant: 'shift_reflect' | 'shift_inpaint'。
    Returns:
        (video (T,H,W,3) uint8, n_degraded 有多少帧的 inpaint 退化成了 reflect)
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant 必须是 {VARIANTS};得到 {variant!r}")
    src = _to_uint8(np.asarray(first_frame))
    off = np.asarray(offsets, dtype=np.float64)
    if off.ndim != 2 or off.shape[1] != 2:
        raise ValueError(f"offsets 必须是 (T,2);得到 {off.shape}")
    if len(off) < 1:
        raise ValueError("offsets 至少要有 1 帧")
    if abs(off[0, 0]) > 1e-9 or abs(off[0, 1]) > 1e-9:
        raise ValueError(f"offsets[0] 必须是 (0,0),第 0 帧就是条件首帧;"
                         f"得到 {off[0].tolist()}")

    frames, n_degraded = [], 0
    for dx_f, dy_f in off:
        dx, dy = int(round(float(dx_f))), int(round(float(dy_f)))
        if variant == "shift_reflect":
            frames.append(shift_reflect(src, dx, dy))
        else:
            fr, degraded = shift_inpaint(src, dx, dy)
            frames.append(fr)
            n_degraded += int(degraded)
    return np.stack(frames, axis=0), n_degraded


# =============================================================================
# 条件 → 逐帧位移
# =============================================================================

def content_offsets_from_waypoints(waypoints: Sequence,
                                   T: int,
                                   scale_y: float,
                                   scale_x: float) -> np.ndarray:
    """整条 GT 折线 → 逐帧累计**内容**位移(帧像素)。oracle 变体用。

    content = -camera:窗口中心往东挪 Δcx,画面内容就往西挪 Δcx。
    waypoints 在 scene 像素里,帧是窗口的上采样,所以要乘 (H/win_h, W/win_w)。
    """
    cam = _interpolate_positions(waypoints, int(T))        # (T,2) (cy,cx) scene px
    d = cam - cam[0]
    dx = -d[:, 1] * float(scale_x)
    dy = -d[:, 0] * float(scale_y)
    return np.stack([dx, dy], axis=-1)


def content_offsets_from_direction(content_angle_deg: float,
                                   T: int,
                                   step_px: float) -> np.ndarray:
    """常量朝向 + 常量速度 → 逐帧累计内容位移(帧像素)。

    content_angle_deg 是**内容角**(= 相机角 + 180),即 meta 里的
    direction_angle;这里直接用,不再翻 180。
    """
    r = math.radians(float(content_angle_deg))
    ux, uy = math.cos(r), -math.sin(r)     # 图像 y 向下,故 sin 取负
    t = np.arange(int(T), dtype=np.float64) * float(step_px)
    return np.stack([t * ux, t * uy], axis=-1)


def net_speed_from_waypoints(waypoints: Sequence, T: int,
                             scale_y: float, scale_x: float) -> Optional[float]:
    """GT 的**净位移**速度(帧像素/帧)。net_direction 模式默认速度来源。

    用净位移而不是弧长:net 模式走的是一条直线,拿弧长会系统性冲过头
    (多段折线的弧长恒 >= 净位移),终点误差反而更大 —— 而下界要的是最有利设定。
    """
    if waypoints is None or len(waypoints) < 2 or int(T) < 2:
        return None
    wp = np.asarray(waypoints, dtype=np.float64)
    net_y = (wp[-1, 0] - wp[0, 0]) * float(scale_y)
    net_x = (wp[-1, 1] - wp[0, 1]) * float(scale_x)
    return float(math.hypot(net_x, net_y) / (int(T) - 1))


def build_offsets_for_sample(sample: Dict,
                             T: int,
                             trajectory_source: str,
                             step_px: Optional[float]) -> Tuple[Optional[np.ndarray], float, str]:
    """一个样本 → (offsets (T,2) 或 None, 实际用的 step_px, 说明字符串)。

    返回 None 表示该样本给不出可信的位移(无方向目标 / 无 waypoints 又没给
    --step-px),调用方应**跳过并计数**,绝不用 0 或随便一个方向凑数。
    """
    if trajectory_source not in TRAJECTORY_SOURCES:
        raise ValueError(f"trajectory_source 必须是 {TRAJECTORY_SOURCES}")
    H, W = int(sample["image"].shape[0]), int(sample["image"].shape[1])
    win_h, win_w = (int(v) for v in sample["pose_window_size"])
    if win_h <= 0 or win_w <= 0:
        raise ValueError(f"pose_window_size 非法: {sample['pose_window_size']!r}")
    sy, sx = H / win_h, W / win_w

    wp = sample.get("waypoints")
    if trajectory_source == "waypoints":
        if wp is None or len(wp) < 2:
            return None, 0.0, "无 waypoints,waypoints 变体不可用"
        off = content_offsets_from_waypoints(wp, T, sy, sx)
        eff = (float(np.mean(np.linalg.norm(np.diff(off, axis=0), axis=1)))
               if T > 1 else 0.0)
        return off, eff, "折线逐帧位移(oracle:转弯也白送)"

    ang = sample.get("direction_angle")
    if ang is None:
        return None, 0.0, "该样本没有方向目标(direction_angle=None),不可平移"
    if step_px is not None:
        spd, how = float(step_px), f"固定速度 {float(step_px):.3f} px/帧(盲测)"
    else:
        spd = net_speed_from_waypoints(wp, T, sy, sx)
        if spd is None:
            return None, 0.0, "无 waypoints 反推速度,且未传 --step-px"
        how = "GT 净位移速度(oracle 后门)"
    return content_offsets_from_direction(float(ang), T, spd), spd, how


# =============================================================================
# manifest
# =============================================================================

def _build_baseline_manifest(*, data_h5, split_file, frame_num, gen_seed,
                             pose_mode, prompt_schema, variant,
                             trajectory_source, step_px) -> Dict:
    """建 manifest 并**强制**把 headline 资格钉死为 False。

    build_manifest() 会按 role 查 run_manifest._NEVER_HEADLINE_ROLES —— 那张表里
    没有 'trivial_baseline',于是 official+drop 环境下它会算出 eligible=True。
    平凡基线进 headline 是灾难性的口径污染,所以这里覆盖,并把原因写进产物。
    (真正的修法是把这个 role 加进那张表,由主协调者接线;此处是第二道防线。)
    """
    from run_manifest import build_manifest

    m = build_manifest(
        role=ROLE,
        pose_mode=pose_mode,
        data_h5=data_h5,
        split_file=split_file,
        checkpoint_id=f"none:{METHOD_VERSION}:{variant}",
        prompt_schema=prompt_schema,
        frame_num=frame_num,
        gen_seed=gen_seed,
        extra={
            "outpaint_variant": variant,
            "outpaint_trajectory_source": trajectory_source,
            "outpaint_step_px": (float(step_px) if step_px is not None else None),
            "outpaint_method_version": METHOD_VERSION,
            "pre_registered_reading": PRE_REGISTERED_READING,
        },
    )
    m["headline_eligible"] = False
    m["headline_ineligible_reason"] = (
        f"role={ROLE}:非生成式平凡基线(平移+补边),是任务难度下界,"
        f"按定义永不入 headline")
    return m


# =============================================================================
# 主流程
# =============================================================================

def run_outpaint_baseline(*, data_path: str, split_file: str, output_dir: str,
                          variant: str, trajectory_source: str,
                          step_px: Optional[float], frame_num: int,
                          max_samples: Optional[int], gen_seed: int) -> Dict:
    """跑完整基线,产出 evaluate.py 可直接吃的 eval 目录。返回统计 dict。"""
    from evaluate import load_val_indices, load_val_samples
    from metrics import compute_psnr
    from run_manifest import (read_manifest, write_manifest,
                              assert_compatible, summarize)

    pose_mode, pose_conditioning = _TRAJ_TO_POSE[trajectory_source]
    # 位姿口径由**基线实际用了什么**决定。环境里若已有一个不同的值,说明调用方
    # 以为在跑别的臂 —— 静默覆盖会让 meta 与作业脚本各说各话,直接判死。
    env_pm = os.environ.get("RSWM_POSE_MODE")
    if env_pm is not None and env_pm.strip().lower() != pose_mode:
        raise SystemExit(
            f"[FATAL] RSWM_POSE_MODE={env_pm!r} 与本基线的实际位姿口径 "
            f"{pose_mode!r} 冲突(--trajectory-source={trajectory_source} 时,"
            f"位姿通路{'没接上' if pose_mode == 'drop' else '整条 GT 折线都用了'})。\n"
            f"  平凡基线没有位姿输入这一说,不要给它设 RSWM_POSE_MODE;"
            f"确需覆盖请改 --trajectory-source。")
    os.environ["RSWM_POSE_MODE"] = pose_mode

    out = Path(output_dir)
    gen_dir, gt_dir, meta_dir = out / "generated", out / "gt", out / "meta"
    for d in (gen_dir, gt_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    val_indices = load_val_indices(split_file)
    logger.info(f"验证集 indices: {len(val_indices)} 条")
    samples = load_val_samples(data_path, val_indices, max_samples)
    if not samples:
        raise SystemExit("[FATAL] 没有加载到任何验证样本")

    manifest = _build_baseline_manifest(
        data_h5=data_path, split_file=split_file, frame_num=frame_num,
        gen_seed=gen_seed, pose_mode=pose_mode,
        prompt_schema=samples[0].get("prompt_schema"),
        variant=variant, trajectory_source=trajectory_source, step_px=step_px)
    # 断点续跑的身份门:与 evaluate.py:377-380 同型。变体不同 → checkpoint_id
    # 不同 → 这里就会拒 —— 防止 reflect 与 inpaint 的产物混进同一个目录。
    if any(meta_dir.glob("*.json")):
        assert_compatible(read_manifest(out), manifest, ctx="outpaint 基线断点续跑")
        logger.info(f"[manifest] 复用已有产物,身份一致 — {summarize(manifest)}")
    write_manifest(out, manifest)

    n_done, n_skipped_existing, n_skipped_nodir, n_degraded_total = 0, 0, 0, 0
    # 逐样本的速度 / 剔首帧 PSNR 只写进 meta;末尾的聚合从磁盘上全部 meta 重算
    # (断点续跑时内存里只有本次新做的那部分,见下方注释)。
    for i, s in enumerate(samples):
        key = s["key"]
        if (meta_dir / f"{key}.json").exists():
            n_skipped_existing += 1
            continue
        T = int(min(frame_num, s["num_frames"]))
        offsets, eff_step, how = build_offsets_for_sample(
            s, T, trajectory_source, step_px)
        if offsets is None:
            logger.warning(f"[{i+1}/{len(samples)}] 跳过 {key}:{how}")
            n_skipped_nodir += 1
            continue

        t0 = time.time()
        video, n_deg = outpaint_video(s["image"], offsets, variant)
        elapsed = time.time() - t0
        n_degraded_total += n_deg

        np.save(str(gen_dir / f"{key}.npy"), video)
        gt = s["video_frames"]
        if gt.dtype != np.uint8:
            gt = np.clip(gt, 0, 255).astype(np.uint8)
        np.save(str(gt_dir / f"{key}.npy"), gt)

        # ---- PSNR=inf 是本基线的**结构性**产物,不是 bug,但必须补一个可读数字 ----
        # 建库器里 image = _resize_frame(frames[0]) 且 vf[0] 走同一行
        # (build_dataset_from_benchmark.py:1152-1153)→ H5 的首帧条件图与 GT 首帧
        # **逐字节相同**。本基线第 0 帧位移为 0,于是 gen[0] == gt[0],
        # compute_psnr 该帧 mse=0 → inf,compute_video_psnr 取均值 → 整段 inf,
        # 再取样本均值 → overall PSNR = Infinity。生成模型不会撞上这个
        # (它的首帧是 VAE 重建,与 GT 首帧 MSE ~3e-5),所以这是基线独有的。
        # 不改 metrics.py(不在本次授权内),改为**另算一份剔除首帧的 PSNR**
        # 落进 meta,论文表里该基线的 PSNR 一栏用它、并注明口径。
        _t = min(len(video), len(gt))
        _p = [compute_psnr(video[k], gt[k]) for k in range(1, _t)]
        _p = [v for v in _p if math.isfinite(v)]
        psnr_excl0 = float(np.mean(_p)) if _p else None

        # 字段与 evaluate.py:470-507 一一对应 —— compute_metrics 的口径门
        # (pose_window_size / direction_target_source / pose_mode /
        #  gt_net_displacement_ratio)少一个就整目录判死。
        meta = {
            "key": key,
            "prompt": s["prompt"],
            "scene": s["scene"],
            "direction_angle": s["direction_angle"],
            "num_frames_gt": s["num_frames"],
            "num_frames_gen": int(video.shape[0]),
            "pose_window_size": list(s.get("pose_window_size", (128, 222))),
            "direction_target_source": s.get("direction_target_source"),
            "direction_word": s.get("direction_word"),
            "prompt_schema": s.get("prompt_schema", "net"),
            "rswm_profile": os.environ.get("RSWM_PROFILE", "unset"),
            # 平凡基线永不入 headline —— 与环境无关,写死 False。
            "headline_eligible": False,
            "sem_arm_mode": os.environ.get("SEM_ARM_MODE", "unset"),
            "pose_conditioning": pose_conditioning,
            "pose_mode": pose_mode,
            "gt_net_displacement_ratio": s.get("gt_net_displacement_ratio"),
            "infer_seconds": round(float(elapsed), 3),
            "infer_fps": (round(video.shape[0] / elapsed, 3)
                          if elapsed > 0 else None),
            # 基线自述:哪个变体、位移怎么来的、有几帧的 inpaint 退化了。
            "role": ROLE,
            "outpaint_variant": variant,
            "outpaint_trajectory_source": trajectory_source,
            "outpaint_step_px": float(eff_step),
            "outpaint_step_source": how,
            "outpaint_frames_fill_degraded": int(n_deg),
            "outpaint_fill_degraded": bool(n_deg > 0),
            "outpaint_method_version": METHOD_VERSION,
            "outpaint_psnr_excl_frame0": psnr_excl0,
            "outpaint_psnr_note": (
                "本基线 gen[0] 与 gt[0] 逐字节相同(H5 的 image 就是 vf[0]),"
                "compute_video_psnr 因此恒为 inf;论文里该基线的 PSNR 用 "
                "outpaint_psnr_excl_frame0(剔除首帧)并注明口径。"),
        }
        (meta_dir / f"{key}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False))
        n_done += 1
        if n_done % 20 == 1:
            logger.info(f"[{i+1}/{len(samples)}] {key}: {video.shape} "
                        f"{elapsed:.2f}s step={eff_step:.2f}px/帧 ({how})")

    if n_done == 0 and n_skipped_existing == 0:
        raise SystemExit(
            f"[FATAL] 一个样本都没产出(无方向目标 {n_skipped_nodir} 条)——"
            f" 不产出空目录冒充成功")
    # 聚合值从**磁盘上的全部 meta** 重算,而不是本次循环累积的列表:
    # 断点续跑时本次只新做了一部分样本,用内存列表会把 outpaint_stats.json
    # 覆写成"只统计了新样本"的数字,而文件名看不出这一点。
    all_meta = []
    for mp in sorted(meta_dir.glob("*.json")):
        try:
            all_meta.append(json.loads(mp.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            raise SystemExit(f"[FATAL] 读不了 {mp}: {e} —— 产物不完整,不出统计")
    _psnr = [m["outpaint_psnr_excl_frame0"] for m in all_meta
             if m.get("outpaint_psnr_excl_frame0") is not None]
    _steps = [m["outpaint_step_px"] for m in all_meta
              if m.get("outpaint_step_px") is not None]
    _deg = sum(int(m.get("outpaint_frames_fill_degraded") or 0) for m in all_meta)
    stats = {
        "n_samples_total": len(all_meta),
        "n_generated_this_run": n_done,
        "n_skipped_existing": n_skipped_existing,
        "n_skipped_no_direction": n_skipped_nodir,
        "n_frames_fill_degraded": _deg,
        "n_frames_fill_degraded_this_run": n_degraded_total,
        "mean_step_px": float(np.mean(_steps)) if _steps else None,
        "mean_psnr_excl_frame0": float(np.mean(_psnr)) if _psnr else None,
        "variant": variant,
        "trajectory_source": trajectory_source,
        "pose_mode": pose_mode,
        "output_dir": str(out),
    }
    (out / "outpaint_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False))
    logger.info(f"完成: {json.dumps(stats, ensure_ascii=False)}")
    logger.warning(
        "⚠ 本基线的 overall PSNR 会是 Infinity(gen[0]==gt[0],见 meta 的 "
        "outpaint_psnr_note)。论文表用 mean_psnr_excl_frame0="
        f"{stats['mean_psnr_excl_frame0']} 并注明剔除首帧。")
    if n_degraded_total:
        logger.warning(
            f"{n_degraded_total} 帧的 inpaint 退化成了 reflect(位移超出画面或"
            f"缺 cv2)—— meta 已如实标记 outpaint_fill_degraded=true")
    logger.info(
        "下一步(环境必须与本次一致,否则 manifest 身份门会拒):\n"
        f"  RSWM_PROFILE={os.environ.get('RSWM_PROFILE', 'unset')} "
        f"SEM_ARM_MODE={os.environ.get('SEM_ARM_MODE', 'unset')} "
        f"RSWM_POSE_MODE={pose_mode} \\\n"
        f"    python {HERE / 'evaluate.py'} "
        f"--metrics_only --output_dir {out}")
    return stats


# =============================================================================
# 自检 / 打分演示(不需要 H5、不需要 GPU)
# =============================================================================

def _self_test() -> int:
    """在合成场景上真跑一遍,并把分数打出来。

    合成场景的 GT **就是**建库器的构造方式(在静态图上滑窗),所以这一小节
    回答的正是审稿人的问题:在最理想的合成条件下,平移+补边能拿多少分。
    """
    from diagnostic_baselines import (_fabricate_demo_scene,
                                      deterministic_translation, persistence)
    from metrics import (compute_psnr, compute_video_ssim, compute_video_psnr,
                         estimate_motion_direction, compute_angular_error)

    ok, bad = [], []

    def check(name, cond, detail=""):
        (ok if cond else bad).append(name)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {detail}" if detail and not cond else ""))

    # ---- 1) 纯函数的已知答案 ----
    print("== 1. 平移/补边的已知答案 ==")
    img = _fabricate_demo_scene(40, 60, seed=1)
    check("shift_zero_is_identity",
          np.array_equal(shift_reflect(img, 0, 0), img))
    sh = shift_reflect(img, 5, 0)
    # 内容东移 5px:输出的 x>=5 部分应逐像素等于原图的 x-5 部分
    check("shift_east_moves_content_east",
          np.array_equal(sh[:, 5:], img[:, :-5]),
          "平移方向或符号错了")
    sh2 = shift_reflect(img, 0, -4)   # dy<0 = 内容北移
    check("shift_north_moves_content_north",
          np.array_equal(sh2[:-4, :], img[4:, :]))
    inp, deg = shift_inpaint(img, 6, 0)
    check("inpaint_keeps_visible_pixels",
          (not deg) and np.array_equal(inp[:, 6:], img[:, :-6]),
          f"degraded={deg}")
    check("inpaint_fills_hole_nonblack",
          (not deg) and int(inp[:, :6].max()) > 0,
          "新暴露区域全黑 = inpaint 没生效")
    _, deg2 = shift_inpaint(img, 999, 0)
    check("inpaint_no_overlap_declares_degraded", deg2)

    # ---- 2) 端到端:合成场景 + 真实 GT 构造 ----
    # 合成场景的 GT 就是建库器的构造方式(静态图上滑窗),所以这一节回答的
    # 正是审稿人的问题。跑**两个速度档**,因为它们回答的不是同一件事:
    #   in_range  —— 每帧位移 ~9px,落在 Farneback 的可靠量程内。这一档才配
    #                做 known-answer 断言(读数误差归因于我的管线,不是估计器)。
    #   canonical —— 主库口径:起止窗口 overlap ∈[0,0.10) ⇒ 总位移≈整窗宽,
    #                每帧 ~15px,**超出** Farneback 量程(gt_oracle 自己都读不准
    #                —— 这就是仓库里记的 ≈25° 仪器地板)。只打印,不断言。
    win_h, win_w = 96, 166          # ≈480:832,与 canonical 128x222 同比
    scene = _fabricate_demo_scene(460, 460, seed=7)
    T = 17                          # K=2 → 9 + (2-1)*8

    def _quantize_cam(a):
        return (0.0 if (a < 45 or a >= 315) else
                90.0 if a < 135 else 180.0 if a < 225 else 270.0)

    def _run_scenario(waypoints, label, tight):
        """跑一个速度档,打印表格;tight=True 时下断言。"""
        gt = deterministic_translation(scene, waypoints, (win_h, win_w), T)
        first = gt[0]
        cam_ang = math.degrees(math.atan2(
            -(waypoints[-1][0] - waypoints[0][0]),
            waypoints[-1][1] - waypoints[0][1])) % 360.0
        content_cmd = (_quantize_cam(cam_ang) + 180.0) % 360.0   # meta.direction_angle
        content_true = (cam_ang + 180.0) % 360.0                 # GT 连续内容净角
        sample = {"image": first, "pose_window_size": (win_h, win_w),
                  "waypoints": np.asarray(waypoints, dtype=np.float32),
                  "direction_angle": content_cmd}

        def _psnr_tail(vid):
            """剔除首帧的 PSNR —— 首帧恒等于条件图,compute_video_psnr 会给 inf。"""
            n = min(len(vid), len(gt))
            vals = [compute_psnr(vid[k], gt[k]) for k in range(1, n)]
            vals = [v for v in vals if math.isfinite(v)]
            return float(np.mean(vals)) if vals else float("nan")

        def _row(name, vid, spd, ndeg):
            est = estimate_motion_direction(vid, flow_size=(win_h, win_w), step=1)
            return {"name": name, "ssim": compute_video_ssim(vid, gt),
                    "psnr_all": compute_video_psnr(vid, gt),
                    "psnr_tail": _psnr_tail(vid), "est": est,
                    "e_cmd": (compute_angular_error(est, content_cmd)
                              if est is not None else float("nan")),
                    "e_true": (compute_angular_error(est, content_true)
                               if est is not None else float("nan")),
                    "spd": spd, "ndeg": ndeg}

        rows = []
        for tsrc in TRAJECTORY_SOURCES:
            offs, spd, how = build_offsets_for_sample(sample, T, tsrc, None)
            if tight:
                check(f"[{label}] offsets_built[{tsrc}]", offs is not None, how)
            if offs is None:
                continue
            for var in VARIANTS:
                vid, ndeg = outpaint_video(first, offs, var)
                rows.append(_row(f"{var}/{tsrc}", vid, spd, ndeg))
        rows.append(_row("persistence(静止)", persistence(first, T), 0.0, 0))
        rows.append(_row("gt_oracle(上界)", gt, 0.0, 0))

        print(f"\n  --- {label} ---")
        print(f"  GT 相机净角={cam_ang:.1f}° → 量化 {_quantize_cam(cam_ang):.0f}°;"
              f"命令内容角={content_cmd:.0f}°,GT 真实内容净角={content_true:.1f}°,"
              f"量化残差={compute_angular_error(content_cmd, content_true):.1f}°")
        print(f"  {'方法':26s} {'SSIM':>7s} {'PSNR(t>0)':>10s} {'ε_cmd°':>8s} "
              f"{'ε_true°':>8s} {'step_px':>8s} {'退化帧':>6s}")
        for r in rows:
            print(f"  {r['name']:26s} {r['ssim']:7.4f} {r['psnr_tail']:10.3f} "
                  f"{r['e_cmd']:8.2f} {r['e_true']:8.2f} {r['spd']:8.2f} "
                  f"{r['ndeg']:6d}")
        return {r["name"]: r for r in rows}

    # in_range:东段 72px / 8 帧 = 9px/帧(= 仓库标定的窗口分辨率 GT 速度),
    # 北段 40px / 8 帧 = 5px/帧。净角 atan2(40,72)=29.1° → 量化到 east。
    d = _run_scenario([(300.0, 130.0), (300.0, 202.0), (260.0, 202.0)],
                      "in_range(9px/帧,光流量程内)", tight=True)
    # canonical:东段 ≈ 整窗宽(166px)→ 起止 overlap≈0,与主库 [0.00,0.10) 同量级。
    d_can = _run_scenario([(300.0, 110.0), (300.0, 276.0), (240.0, 276.0)],
                          "canonical(≈15px/帧,overlap≈0,量程外)", tight=False)

    # ---- 已知答案:全部落在 in_range 档,断的是**我的管线**不是估计器 ----
    for k in ("shift_reflect/net_direction", "shift_inpaint/net_direction"):
        # net 变体是按命令角平移出来的 → 光流必须读回命令角。
        # 这条钉整条 offsets 管线的符号与缩放(错一个负号就整整 180°)。
        check(f"{k}:光流读回命令角", d[k]["e_cmd"] < 5.0,
              f"ε_cmd={d[k]['e_cmd']:.2f}°(est={d[k]['est']})")
    # waypoints 变体复刻的是**同一条 GT 相机路径**,所以正确的参照物不是
    # content_true(端点净角),而是 **gt_oracle 自己被读出来的角度**。
    # 两者之差 = 我的管线误差;而 gt_oracle 与 content_true 之差是估计器的
    # 固有偏置(幅度加权把长/快的那一段权重放大,多段折线的"平均方向"本来就
    # 不等于端点净方向 —— trajectory_metrics 里记的 Track A 已知盲区)。
    # 首版把参照物写成 content_true,于是这条断言在量 **估计器**,不是量我。
    _ang_gap = compute_angular_error(d["shift_reflect/waypoints"]["est"],
                                     d["gt_oracle(上界)"]["est"])
    check("shift_reflect/waypoints:与 gt_oracle 读出同一个方向",
          _ang_gap < 8.0,
          f"与 gt_oracle 的读数差 {_ang_gap:.2f}°(gt_oracle 自身相对端点净角"
          f"已偏 {d['gt_oracle(上界)']['e_true']:.2f}°,那是估计器的固有偏置)")
    # inpaint 变体共用同一批 offsets,差别只在补边:TELEA 在大洞上是**涂抹**,
    # 涂抹区域会贡献伪光流,方向读数因此明显变脏。这里只要求它仍显著优于
    # 随机(90°),不要求与 reflect 同精度 —— 这个差正是预注册规则 R4 要量的量。
    _ang_gap_i = compute_angular_error(d["shift_inpaint/waypoints"]["est"],
                                       d["gt_oracle(上界)"]["est"])
    check("shift_inpaint/waypoints:方向仍显著优于随机",
          _ang_gap_i < 45.0,
          f"与 gt_oracle 的读数差 {_ang_gap_i:.2f}°(随机期望 90°)")
    for k in ("shift_reflect/net_direction", "shift_inpaint/net_direction",
              "shift_reflect/waypoints", "shift_inpaint/waypoints"):
        check(f"{k}:不如 GT 自身保真", d[k]["ssim"] < d["gt_oracle(上界)"]["ssim"])
    check("waypoints 变体(转弯白送)SSIM 不低于 net 变体",
          d["shift_reflect/waypoints"]["ssim"]
          >= d["shift_reflect/net_direction"]["ssim"] - 1e-6,
          f"{d['shift_reflect/waypoints']['ssim']:.4f} vs "
          f"{d['shift_reflect/net_direction']['ssim']:.4f}")
    # PSNR 的结构性 inf:钉住它,免得哪天有人看到 Infinity 以为是崩了。
    check("整段 PSNR 恒为 inf(gen[0]==gt[0] 的结构性后果)",
          math.isinf(d["shift_reflect/net_direction"]["psnr_all"]),
          "若这条 FAIL,说明首帧不再等于条件图 —— 反而要查位移取整是不是错了")
    check("剔除首帧后的 PSNR 是有限值",
          math.isfinite(d["shift_reflect/net_direction"]["psnr_tail"]))
    # canonical 档**故意不下断言**:那一档 gt_oracle 自己的角读数已经受量程与
    # 幅度加权双重污染(表里可见它相对端点净角就偏了十几度),任何以它为基准的
    # 不等式都是在赌噪声的符号。只打印,供人读难度,不进 PASS/FAIL。
    print(f"     canonical 档参考:gt_oracle ε_true="
          f"{d_can['gt_oracle(上界)']['e_true']:.2f}°、"
          f"shift_reflect/waypoints ε_true="
          f"{d_can['shift_reflect/waypoints']['e_true']:.2f}°、"
          f"inpaint 退化帧={d_can['shift_inpaint/waypoints']['ndeg']}/{T}"
          f" —— 主库 overlap≈0 意味着末尾几帧的内容**整幅都是编的**。")
    # persistence 的角度读数是**噪声**不是方向:Farneback 在逐字节相同的两帧上
    # 仍吐出 ~1e-3 量级假位移,而 estimate_motion_direction 的幅度门
    # (weight_total < 1.0)是**绝对阈值**,96x166x16 对累计早已 >1 → 门不响,
    # 返回随机角。所以绝不能拿 persistence 的角误差当参照物。
    print(f"     ↑ persistence est={d['persistence(静止)']['est']}"
          f"(静止视频却给出了角度 —— 见 test_metrics_known_answer.py 的 D3)")

    print("\n  读数(预注册规则见 PRE_REGISTERED_READING):ε_cmd 贴地板是"
          "**构造给的**,不构成任何结论;要看的是 SSIM/LPIPS/FVD 与 Track B。")
    print(f"\n汇总: {len(ok)} PASS / {len(bad)} FAIL")
    if bad:
        print("结果: FAIL —— " + ", ".join(bad))
        return 4
    print("结果: 全绿 ✓")
    return 0


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="RSWM 平凡 outpainting 基线(难度下界,role=trivial_baseline)")
    ap.add_argument("--data_path", type=str, default="", help="HDF5 数据集")
    ap.add_argument("--split_file", type=str, default="", help="train_val_split.json")
    ap.add_argument("--output_dir", type=str, default="eval_output_outpaint")
    ap.add_argument("--variant", choices=VARIANTS, default="shift_reflect",
                    help="补边方式;一个目录只放一个变体(manifest 会拒混放)")
    ap.add_argument("--trajectory-source", dest="trajectory_source",
                    choices=TRAJECTORY_SOURCES, default="net_direction",
                    help="net_direction=只用命令方向词(真 text-only 对照);"
                         "waypoints=整条 GT 折线白送(oracle 上位变体)")
    ap.add_argument("--step-px", dest="step_px", type=float, default=None,
                    help="每帧内容位移(帧像素)。不传=从 GT 净位移反推(后门,"
                         "对基线最有利);传了=盲测。仅对 net_direction 有效。")
    ap.add_argument("--frame_num", type=int, default=33)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--gen_seed", type=int, default=42,
                    help="仅为与 LoRA 臂的 manifest 对齐;本基线是确定性的,不用随机数")
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="不读 H5,在合成场景上真跑并打分")
    ap.add_argument("--print-reading", dest="print_reading", action="store_true",
                    help="打印预注册的读数规则后退出")
    args = ap.parse_args()

    if args.print_reading:
        print(json.dumps(PRE_REGISTERED_READING, indent=2, ensure_ascii=False))
        return 0
    if args.self_test:
        return _self_test()

    if not args.data_path or not args.split_file:
        ap.error("--data_path 与 --split_file 必须指定(或用 --self-test)")
    if args.step_px is not None and args.trajectory_source != "net_direction":
        ap.error("--step-px 只对 --trajectory-source net_direction 有意义 —— "
                 "waypoints 变体的速度由折线本身决定")
    if args.step_px is not None and args.step_px <= 0:
        ap.error(f"--step-px 必须为正;得到 {args.step_px}")

    run_outpaint_baseline(
        data_path=args.data_path, split_file=args.split_file,
        output_dir=args.output_dir, variant=args.variant,
        trajectory_source=args.trajectory_source, step_px=args.step_px,
        frame_num=args.frame_num, max_samples=args.max_samples,
        gen_seed=args.gen_seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
