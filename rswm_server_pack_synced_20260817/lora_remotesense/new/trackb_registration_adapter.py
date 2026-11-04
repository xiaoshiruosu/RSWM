#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""trackb_registration_adapter.py — 把配准估计器接进既有 Track B，并加挂 Track C。

**这是 P0 改动 #6 的可运行原型**：既有 `trajectory_metrics.score_trajectory` 内部把
"恢复轨迹"写死成 Farnebäck 光流（`recover_content_path`）。本适配器不改既有文件，
而是复用它的**全部纯几何指标**，只把路径来源换成配准，并额外产出 Track C 的绝对量。
真正落地时按本文件的调用顺序在 `score_trajectory` 里加一个 `--path-estimator` 分支即可。

**为什么不直接改 score_trajectory**：那个函数还负责 liveness 门控、结点来源判定、
text_only 语义标注等一堆与估计器无关的逻辑。先用适配器把"换估计器"这件事单独验证通过，
再做最小侵入的分支，比一次性改一个 1000+ 行的编排器安全。

**列序陷阱**（务必看 `geom_registration` 模块头）：本模块内部已做转换——
`geom_registration` 产出 (cy,cx) **相机**路径，Track B 的 `recovered_path` 参数要的是
(x,y) **内容**路径。适配器负责这次转换，调用方不必再关心。

自检：`python3 code/trackb_registration_adapter.py --self-test`（需要能找到 pack；否则 SKIP）
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_registration as GR   # noqa: E402

# 既有工程的位置（可用环境变量覆盖，便于在服务器上指向真实工程根）
from pack_path import find_pack_root, skip_banner   # noqa: E402

_TM = None


def _load_trajectory_metrics():
    """惰性加载既有 trajectory_metrics（找不到就 fail-loud，绝不静默降级）。"""
    global _TM
    if _TM is not None:
        return _TM
    root = find_pack_root()
    if root is None:
        raise FileNotFoundError(
            "找不到既有工程根（设 RSWM_PACK_ROOT 指向它）；本适配器复用其 Track B 数学，无法降级运行")
    p = os.path.join(root, "lora_remotesense", "new")
    sys.path.insert(0, p)
    import trajectory_metrics as tm   # noqa: E402
    _TM = tm
    return _TM


def score_trajectory_registration(
    gen_frames: np.ndarray,
    waypoints: Sequence,
    segment_angles: Sequence,
    *,
    detector: str = "orb",
    knot_frames: Optional[Sequence[int]] = None,
    gsd_m_per_px: Optional[float] = None,
    text_only: bool = False,
    max_failed_ratio: float = 0.10,
) -> Dict[str, object]:
    """配准版 Track B + Track C 打分，返回**扁平** dict（键与既有 TRACKB_KEYS 对齐）。

    与既有 `score_trajectory` 的口径差异（必须在论文里写明，不能混表）：
      * 路径来源 = 配准（相似变换复合），不是 Farnebäck 积分；
      * 因此 `flow_magnitude` 用逐步位移中位数代替（同量纲，不同估计器）；
      * 新增 `estimator` / `n_failed_steps` / Track C 的全部键。

    退化处理沿用既有规范：不可信一律 None，绝不用 0 冒充完美。
    """
    tm = _load_trajectory_metrics()
    frames = np.asarray(gen_frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("gen_frames 必须是 (T,H,W,3)，收到 %s" % (frames.shape,))

    rec = GR.recover_path_by_registration(list(frames), detector=detector,
                                          max_failed_ratio=max_failed_ratio)
    out: Dict[str, object] = {
        "estimator": "registration:%s" % detector,
        "valid": bool(rec.valid),
        "commanded": (not text_only),
        "n_failed_steps": rec.n_failed_steps,
        "failed_indices": list(rec.failed_indices),
        "scale_drift": rec.scale_drift,
        "rot_drift_deg": rec.rot_drift_deg,
    }
    # 失败时把所有达标量置 None（与既有"静止视频全部字段 None"的规范一致）
    keys_none = ["net_direction_error", "segment_angular_mean", "segment_success_rate_30",
                 "segment_success_rate_45", "turn_recall", "turn_precision", "turn_f1",
                 "turn_timing_error", "waypoint_error", "waypoint_endpoint_error",
                 "frechet", "dtw", "flow_magnitude", "num_gt_turns", "num_recovered_turns",
                 "path_len_rel_err", "step_len_rel_err_mean", "endpoint_abs_err_px",
                 "endpoint_abs_err_m", "drift_rate_px_per_frame", "final_drift_px",
                 "loop_gap_px", "loop_gap_rel", "recovered_net_angle", "gt_net_content_angle",
                 "per_segment_errors"]
    if not rec.valid or rec.path is None:
        # 键集必须与成功路径一致：缺键会让下游聚合器 KeyError 或静默漏掉一整块
        for k in keys_none:
            out[k] = None
        out["degraded"] = True
        out["reason"] = rec.reason
        return out

    out["degraded"] = False
    cam_path = rec.path                                   # (T,2) (cy,cx) 相机
    content_xy = GR.camera_path_to_content_xy(cam_path)   # (T,2) (x,y) 内容 ← Track B 口径
    T = len(content_xy)
    K = len(segment_angles)
    knots = list(knot_frames) if knot_frames is not None else \
        [int(round(v)) for v in np.linspace(0, T - 1, K + 1)]

    gt_content = tm.gt_content_path(waypoints, T=T)

    # ---- Track B（全部复用既有实现，一行数学都没重写）----
    seg = tm.segment_wise_angular_error(content_xy, waypoints, segment_angles, knot_frames=knots)
    out["segment_angular_mean"] = seg["mean"]
    out["per_segment_errors"] = seg["per_segment"]
    out["segment_success_rate_30"] = tm.segment_success_rate(seg["per_segment"], 30.0)
    out["segment_success_rate_45"] = tm.segment_success_rate(seg["per_segment"], 45.0)
    shape = tm.path_shape_error(content_xy, gt_content)
    out["frechet"], out["dtw"] = shape["frechet"], shape["dtw"]
    out["waypoint_error"] = tm.waypoint_error(content_xy, gt_content, knots)
    out["waypoint_endpoint_error"] = tm.waypoint_endpoint_error(content_xy, gt_content, knots)

    rec_ang = GR.bearing_from_path(cam_path)              # 相机净方位角（连续，无分箱）
    gt_cam = GR.bearing_from_path(np.asarray(waypoints, dtype=np.float64))
    out["recovered_net_angle"] = rec_ang
    out["gt_net_content_angle"] = None if gt_cam is None else (gt_cam + 180.0) % 360.0
    out["net_direction_error"] = GR.angular_diff(rec_ang, gt_cam)

    steps = np.hypot(np.diff(cam_path[:, 0]), np.diff(cam_path[:, 1]))
    out["flow_magnitude"] = float(np.median(steps)) if len(steps) else None

    # ---- Track C（新增；absolute_step_error 需要同轴的 GT 相机路径）----
    # 只要有失败步，路径里就有"位置被冻结"的假步 → 路径积分量一律 None（见 geom 模块说明）
    gt_cam_path = _resample_camera_path(waypoints, T, knot_frames=knots)
    fi = rec.failed_indices
    ase = GR.absolute_step_error(cam_path, gt_cam_path, gsd_m_per_px=gsd_m_per_px,
                                 failed_indices=fi)
    out.update({k: ase[k] for k in ("path_len_rel_err", "step_len_rel_err_mean",
                                    "endpoint_abs_err_px", "endpoint_abs_err_m")})
    drift = GR.rollout_drift_curve(cam_path, gt_cam_path, failed_indices=fi)
    out["drift_rate_px_per_frame"] = drift["drift_rate_px_per_frame"]
    out["final_drift_px"] = drift["final_drift_px"]
    out.update(GR.loop_closure_error(cam_path, failed_indices=fi))

    # 逐段指标：跨越失败步的段同样不可信，逐段置 None（分母只数有效段，沿用既有规范）
    if fi:
        out["degraded"] = True
        per = list(out["per_segment_errors"])
        for k in range(len(per)):
            i0, i1 = knots[k], knots[k + 1]
            if any(i0 < f <= i1 for f in fi):
                per[k] = None
        out["per_segment_errors"] = per
        valid_errs = [e for e in per if e is not None]
        out["segment_angular_mean"] = float(np.mean(valid_errs)) if valid_errs else None
        out["segment_success_rate_30"] = tm.segment_success_rate(per, 30.0)
        out["segment_success_rate_45"] = tm.segment_success_rate(per, 45.0)
        # 净方向靠首尾端点，端点在失败步之后就不可信
        if any(f <= T - 1 for f in fi):
            out["net_direction_error"] = None
            out["recovered_net_angle"] = None
        out["reason"] = ("含 %d 个配准失败步 %s：路径积分量与受影响段已置 None，"
                         "不得把仪器故障计入模型误差" % (len(fi), fi))
    return out


def _resample_camera_path(waypoints: Sequence, T: int,
                          knot_frames: Optional[Sequence[int]] = None) -> np.ndarray:
    """把 (K+1,2) 的 (cy,cx) 相机航点重采样成 (T,2)，锚定到原点。

    与 `gt_content_path` 的区别：**不做单位弧长归一化**——Track C 要的就是绝对尺度。

    **必须按结点帧逐段插值，不能按总弧长均匀插值**（2026-07-30 修）：
    builder 是"每段固定 steps_per_segment 帧"，各段长度不同 → 各段的**每帧步长**本就不同。
    若按总弧长均匀重采样，完美视频也会被判出 ~28% 的逐步长误差（实测），
    那是仪器误差被记到模型头上。`knot_frames=None` 时退回均匀结点假设
    （与 `segment_wise_angular_error` 的同名回退一致）。
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2:
        raise ValueError("waypoints 必须是 (K+1,2) 且 K>=1，收到 %s" % (wp.shape,))
    K = len(wp) - 1
    if knot_frames is None:
        knots = [int(round(v)) for v in np.linspace(0, T - 1, K + 1)]
    else:
        knots = [int(v) for v in knot_frames]
        if len(knots) != K + 1:
            raise ValueError("knot_frames 需恰有 K+1=%d 个，收到 %d" % (K + 1, len(knots)))
    out = np.zeros((T, 2), dtype=np.float64)
    for k in range(K):
        i0, i1 = knots[k], knots[k + 1]
        if i1 <= i0:
            continue
        n = i1 - i0
        for j in range(n + 1):
            f = j / n
            out[i0 + j] = wp[k] + f * (wp[k + 1] - wp[k])
    if knots[-1] < T - 1:                      # 尾部若有余帧，保持在终点（不外推）
        out[knots[-1]:] = wp[-1]
    return out - out[0]


# =============================================================================
# 自检
# =============================================================================

def _self_test(allow_skip: bool = False) -> int:
    fails: List[str] = []

    def check(name: str, cond: bool, info: str = "") -> None:
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, (" — " + info) if info else ""))
        if not cond:
            fails.append(name)

    try:
        _load_trajectory_metrics()
    except FileNotFoundError:
        skip_banner("整个适配器自检（它 100% 依赖既有 Track B 数学）")
        return 0 if allow_skip else 2

    WH = (128, 222)
    mos = GR._synth_mosaic(1600, 1600, seed=7)
    gt_cam = GR._polyline((800, 500), [(0.0, 540.0), (58.0, 300.0)], steps_per_seg=4)
    frames = np.stack([GR._crop(mos, p[0], p[1], WH) for p in gt_cam])
    wps = np.array([gt_cam[0], gt_cam[4], gt_cam[8]], dtype=np.float64)
    seg_ang = [0.0, 58.0]

    print("\n[1] 端到端：完美视频应当各项接近满分")
    r = score_trajectory_registration(frames, wps, seg_ang, knot_frames=[0, 4, 8],
                                      gsd_m_per_px=0.3)
    check("valid=True", r["valid"] is True, str(r.get("reason", "")))
    check("估计器已标注", r["estimator"].startswith("registration:"), r["estimator"])
    check("逐段角误差 <2°", (r["segment_angular_mean"] or 99) < 2.0,
          "%.3f" % r["segment_angular_mean"])
    check("分段成功率@30 = 1.0", r["segment_success_rate_30"] == 1.0)
    check("净方向误差 <2°", (r["net_direction_error"] or 99) < 2.0,
          "%.3f" % r["net_direction_error"])
    check("Fréchet <0.05", r["frechet"] < 0.05, "%.5f" % r["frechet"])
    check("waypoint_error <0.05", r["waypoint_error"] < 0.05, "%.6f" % r["waypoint_error"])

    print("\n[2] Track C 的绝对量确实出现（Track B 给不出的部分）")
    for k in ("path_len_rel_err", "step_len_rel_err_mean", "endpoint_abs_err_px",
              "endpoint_abs_err_m", "drift_rate_px_per_frame", "loop_gap_px", "scale_drift"):
        check("键 %s 已产出" % k, r.get(k) is not None, str(r.get(k)))
    check("路径长相对误差 <5%", r["path_len_rel_err"] < 0.05, "%.4f" % r["path_len_rel_err"])
    check("逐步长相对误差 <5%（按结点逐段插值后，仪器不再自带 28% 误差）",
          r["step_len_rel_err_mean"] < 0.05, "%.4f" % r["step_len_rel_err_mean"])
    check("端点误差有米制换算", r["endpoint_abs_err_m"] is not None,
          "%.3f m" % r["endpoint_abs_err_m"])

    print("\n[3] 故障注入：退化视频必须整块 None，不得填 0")
    flat = np.full((9, WH[0], WH[1], 3), 128, np.uint8)
    rf = score_trajectory_registration(flat, wps, seg_ang, knot_frames=[0, 4, 8])
    check("valid=False", rf["valid"] is False)
    nz = [k for k in ("segment_angular_mean", "frechet", "waypoint_error",
                      "net_direction_error", "path_len_rel_err") if rf[k] is not None]
    check("所有达标量为 None（不是 0）", not nz, "非 None 的键=%s" % nz)
    check("给出失败原因", bool(rf.get("reason")), str(rf.get("reason")))

    print("\n[4] 尺度盲修复：走 2 倍远必须被绝对量抓到、且被 Track B 归一化指标漏掉")
    gt2 = _resample_camera_path(wps, 9, knot_frames=[0, 4, 8])
    over = gt2 * 2.0
    ase = GR.absolute_step_error(over, gt2)
    check("绝对量：路径长相对误差 ≈1.0", abs(ase["path_len_rel_err"] - 1.0) < 0.02,
          "%.3f" % ase["path_len_rel_err"])
    tm = _load_trajectory_metrics()
    xy_over = GR.camera_path_to_content_xy(over)
    xy_gt = GR.camera_path_to_content_xy(gt2)
    fr = tm.path_shape_error(xy_over, xy_gt)["frechet"]
    check("Track B 归一化指标对 2 倍远确实无感（≈0）", fr < 1e-6,
          "frechet=%.3e —— 这正是必须加 Track C 的理由" % fr)

    print("\n[6] 仪器故障不得记到模型账上（复现审查发现的场景：损坏 1 帧）")
    broken = frames.copy()
    rng = np.random.default_rng(5)
    broken[4] = rng.integers(0, 255, size=frames[0].shape, dtype=np.uint8)   # 第 4 帧换成噪声
    INTEGRALS = ("path_len_rel_err", "step_len_rel_err_mean", "endpoint_abs_err_px",
                 "drift_rate_px_per_frame", "final_drift_px", "loop_gap_px")

    # regime A：失败比例超阈 → 直接 invalid，全表 None，且键集与成功路径一致
    rb = score_trajectory_registration(broken, wps, seg_ang, knot_frames=[0, 4, 8],
                                       gsd_m_per_px=0.3)
    print("     [A] n_failed=%s valid=%s degraded=%s" % (rb["n_failed_steps"], rb["valid"],
                                                         rb.get("degraded")))
    check("A: 检出失败步", rb["n_failed_steps"] > 0, str(rb["n_failed_steps"]))
    check("A: valid=False 且 degraded=True", rb["valid"] is False and rb["degraded"] is True)
    for k in INTEGRALS:
        check("A: 路径积分量 %s 为 None" % k, rb[k] is None, str(rb[k]))
    check("A: 键集与成功路径一致（含 per_segment_errors）", "per_segment_errors" in rb)
    check("A: 给出原因", bool(rb.get("reason")))

    # regime B：放宽阈值 → valid 但 degraded，路径积分量仍必须 None
    rc = score_trajectory_registration(broken, wps, seg_ang, knot_frames=[0, 4, 8],
                                       gsd_m_per_px=0.3, max_failed_ratio=0.9)
    print("     [B] n_failed=%s valid=%s degraded=%s" % (rc["n_failed_steps"], rc["valid"],
                                                         rc.get("degraded")))
    check("B: valid=True 但 degraded=True（不得因为没超阈就当无事发生）",
          rc["valid"] is True and rc["degraded"] is True)
    for k in INTEGRALS:
        check("B: 路径积分量 %s 仍为 None" % k, rc[k] is None, str(rc[k]))
    check("B: 跨越失败步的段被置 None",
          any(e is None for e in rc["per_segment_errors"]), str(rc["per_segment_errors"]))
    check("B: 净方向也不可信 → None", rc["net_direction_error"] is None)
    check("B: 给出原因", bool(rc.get("reason")))

    # 对照：正常样本不得被误判为 degraded
    check("对照: 完好样本 degraded=False", r.get("degraded") is False)

    print("\n[5] 重采样器的边界行为")
    check("零长度航点 → 全零路径", np.allclose(_resample_camera_path([[0, 0], [0, 0]], 5), 0))
    check("结点数不匹配必须抛错", _raises(lambda: _resample_camera_path(wps, 9, knot_frames=[0, 8])))
    try:
        _resample_camera_path([[0, 0]], 5)
        check("单点航点必须抛错", False)
    except ValueError:
        check("单点航点必须抛错", True)
    rp = _resample_camera_path(wps, 9, knot_frames=[0, 4, 8])
    check("重采样保端点方向", abs((GR.bearing_from_path(rp) or -1) -
                                (GR.bearing_from_path(wps - wps[0]) or -2)) < 0.5)

    print("\n" + "=" * 66)
    if fails:
        print("失败 %d 项：%s" % (len(fails), fails))
        return 1
    print("全部通过。")
    return 0


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="配准版 Track B + Track C 适配器")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--allow-skip", action="store_true",
                    help="没有既有工程时允许跳过并仍返回 0（默认返回 2）")
    a = ap.parse_args()
    if a.self_test:
        return _self_test(allow_skip=a.allow_skip)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
