#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""step_overlap.py — 把"帧间重叠"从隐含副产品变成**可控、可审计、可扫**的轴（P0 改动 #7）。

**为什么存在**：既有 builder 只在**首尾窗口之间**算一次重叠
（`build_multi_direction_benchmark_v5.py:745-747` 调 `_calculate_overlap`），
帧与帧之间的重叠既不控制也不记录。实测（见 `audit_default_config()`）默认配置下
相邻帧重叠约 **95%**——模型消费的一直是稠密高重叠链，
所以"不依赖大 overlap"这句话在既有基准上**只对首尾成立**。

本模块提供三件事：
  1. `window_overlap` —— **逐位复刻** v5 `_calculate_overlap`（含它的裁剪与非对称口径）；
  2. `audit_path` —— 逐帧 step-overlap 审计（min/median/max + 落带检查，fail-loud）；
  3. `steps_for_target_overlap` —— **反解**：要达到目标帧间重叠带，每段该走几帧。
     有了它，step-overlap 才是可扫的自变量，而不是事后测量到的结果。

坐标与既有工程一致：窗口中心 `(cy, cx)`，窗口尺寸 `(win_h, win_w)`。

自检：`python3 code/step_overlap.py --self-test`
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pack_path import resolve_or_exit_code  # noqa: E402


# =============================================================================
# 1. 重叠基元（逐位复刻既有实现）
# =============================================================================

def window_bounds(center: Tuple[float, float], window_size: Tuple[int, int],
                  scene_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int, int, int]:
    """(cy,cx) + (win_h,win_w) → (h_s,h_e,w_s,w_e)。

    **逐位复刻** v5 `_get_bounds`：用整数除 `//2`（因此奇数尺寸下上下不对称），
    并在给了 scene_size 时裁到场景内。裁剪会让边缘窗口面积变小，
    这正是 `_calculate_overlap` 用**非对称** intersection/area1 的原因。
    """
    cy, cx = int(center[0]), int(center[1])
    win_h, win_w = int(window_size[0]), int(window_size[1])
    h_s, h_e = cy - win_h // 2, cy + win_h // 2
    w_s, w_e = cx - win_w // 2, cx + win_w // 2
    if scene_size is not None:
        H, W = int(scene_size[0]), int(scene_size[1])
        h_s, h_e = max(0, h_s), min(H, h_e)
        w_s, w_e = max(0, w_s), min(W, w_e)
    return (h_s, h_e, w_s, w_e)


def window_overlap(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """重叠比 = 交集 / **b1 的面积**（非对称；等尺寸未裁剪时才等于 IoU）。

    非对称这件事必须在论文里写明：边缘被裁的窗口面积更小，
    交换 b1/b2 会得到不同的数。既有实现即如此，这里保持一致以便复现旧数。
    """
    h1s, h1e, w1s, w1e = b1
    h2s, h2e, w2s, w2e = b2
    hs, he = max(h1s, h2s), min(h1e, h2e)
    ws, we = max(w1s, w2s), min(w1e, w2e)
    if hs >= he or ws >= we:
        return 0.0
    inter = (he - hs) * (we - ws)
    area1 = (h1e - h1s) * (w1e - w1s)
    return float(inter / area1) if area1 > 0 else 0.0


def overlap_from_centers(c1: Tuple[float, float], c2: Tuple[float, float],
                         window_size: Tuple[int, int],
                         scene_size: Optional[Tuple[int, int]] = None) -> float:
    """两个窗口中心之间的重叠比（便捷封装）。"""
    return window_overlap(window_bounds(c1, window_size, scene_size),
                          window_bounds(c2, window_size, scene_size))


def analytic_overlap(dy: float, dx: float, window_size: Tuple[int, int]) -> float:
    """未裁剪、等尺寸窗口的解析重叠：(1−|dy|/h)⁺ · (1−|dx|/w)⁺。

    用于反解与自检；与 `overlap_from_centers` 在未触边时应一致（差异仅来自整数取整）。
    """
    h, w = float(window_size[0]), float(window_size[1])
    fy = max(0.0, 1.0 - abs(dy) / h)
    fx = max(0.0, 1.0 - abs(dx) / w)
    return float(fy * fx)


# =============================================================================
# 2. 逐帧审计
# =============================================================================

def step_overlaps(path: Sequence, window_size: Tuple[int, int],
                  scene_size: Optional[Tuple[int, int]] = None) -> List[float]:
    """逐帧（t-1, t）重叠序列，长度 T−1。"""
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        raise ValueError("path 必须是 (T,2) 且 T>=2，收到 %s" % (p.shape,))
    return [overlap_from_centers(p[t - 1], p[t], window_size, scene_size)
            for t in range(1, len(p))]


def audit_path(path: Sequence, window_size: Tuple[int, int],
               *, band: Optional[Tuple[float, float]] = None,
               scene_size: Optional[Tuple[int, int]] = None) -> Dict[str, object]:
    """逐帧 step-overlap + 首尾 endpoint-overlap 审计。

    `band=(lo,hi)` 给定时做落带检查（左闭右开，与既有 overlap 审计同规范）；
    不落带则 `in_band=False` 且给出越界的帧号——调用方应据此**删档重采**，
    与 v5 `_write_overlap_audit` 的"违约即删 h5"同一种硬失败姿态。
    """
    ov = step_overlaps(path, window_size, scene_size)
    p = np.asarray(path, dtype=np.float64)
    ep = overlap_from_centers(p[0], p[-1], window_size, scene_size)
    a = np.asarray(ov, dtype=np.float64)
    out: Dict[str, object] = {
        "n_steps": len(ov),
        "step_overlap_min": float(a.min()),
        "step_overlap_median": float(np.median(a)),
        "step_overlap_max": float(a.max()),
        "step_overlap_mean": float(a.mean()),
        "endpoint_overlap": ep,
        "band": None if band is None else [float(band[0]), float(band[1])],
    }
    if band is None:
        out["in_band"] = None
        out["violating_steps"] = None
    else:
        lo, hi = float(band[0]), float(band[1])
        bad = [i + 1 for i, v in enumerate(ov) if not (lo <= v < hi)]
        out["in_band"] = (len(bad) == 0)
        out["violating_steps"] = bad
    return out


# =============================================================================
# 3. 反解：让 step-overlap 成为自变量
# =============================================================================

def steps_for_target_overlap(segment_length_px: float, target_overlap: float,
                             window_size: Tuple[int, int], bearing_deg: float) -> int:
    """给定段长与目标帧间重叠，反解该段需要的**步数**（≥1）。

    解析式：沿方位角 θ 走一步位移 (dy,dx) = s·(−sinθ, cosθ)，
    未裁剪重叠 = (1−|s·sinθ|/h)(1−|s·cosθ|/w) = target，对 s 求根后 steps = ceil(L/s)。
    退化（θ 轴向）时其中一个因子恒为 1，直接线性求解。
    """
    if not (0.0 <= target_overlap < 1.0):
        raise ValueError("target_overlap 必须在 [0,1)，收到 %r" % target_overlap)
    if segment_length_px <= 0:
        raise ValueError("segment_length_px 必须为正")
    h, w = float(window_size[0]), float(window_size[1])
    th = math.radians(float(bearing_deg))
    ay, ax = abs(math.sin(th)) / h, abs(math.cos(th)) / w
    if ay < 1e-12 and ax < 1e-12:
        raise ValueError("方位角退化")
    if ay < 1e-12:
        s = (1.0 - target_overlap) / ax
    elif ax < 1e-12:
        s = (1.0 - target_overlap) / ay
    else:
        # (1-ay·s)(1-ax·s) = t  →  ay·ax·s² − (ay+ax)·s + (1−t) = 0，取较小正根
        A, B, C = ay * ax, -(ay + ax), 1.0 - target_overlap
        disc = B * B - 4 * A * C
        if disc < 0:
            raise ValueError("目标重叠 %.3f 在该方位角下无解" % target_overlap)
        s = (-B - math.sqrt(disc)) / (2 * A)
    if s <= 0:
        raise ValueError("反解得到非正步长")
    return max(1, int(math.ceil(segment_length_px / s)))


def audit_default_config(scene: int = 512, window_size: Tuple[int, int] = (128, 222),
                         steps_per_segment: int = 9,
                         min_segment_length_frac: float = 0.15,
                         max_segment_length_frac: float = 0.40,
                         n_samples: int = 20000, seed: int = 0) -> Dict[str, float]:
    """复算既有默认配置下的**帧间重叠分布**（用来核实"低重叠"这句话是否成立）。

    **不要只算最短段**（2026-07-30 修）：v5 的段长是
    `distance = np.random.uniform(min_distance, max_distance)`
    （`build_multi_direction_benchmark_v5.py:596`），即 **L ~ U[0.15, 0.40]×512**，
    均值 140.8 px 而非下限 76.8 px。此前本函数只算 `min_segment_length` 那一档，
    得到"每步 9.6 px / 重叠 0.957"，那是**最好情况**不是典型情况，
    据此写进论文会低报模型实际消费的帧间位移近一倍。
    同一个错误常数还被烤进了仪器：`instrument_floor_report.py:100`
    `CANONICAL_GT_PX_PER_PAIR = 9.0`。

    返回分布统计（over 随机段长 × 随机方位角）。`*_best_case` 是下限段长那一档，保留作对照。
    """
    rng = np.random.default_rng(seed)
    L = rng.uniform(min_segment_length_frac, max_segment_length_frac, n_samples) * scene
    bearing = rng.uniform(0.0, 360.0, n_samples)
    step = L / (steps_per_segment - 1)
    th = np.radians(bearing)
    dy, dx = -np.sin(th) * step, np.cos(th) * step
    h, w = float(window_size[0]), float(window_size[1])
    ov = np.maximum(0.0, 1.0 - np.abs(dy) / h) * np.maximum(0.0, 1.0 - np.abs(dx) / w)
    L_min = min_segment_length_frac * scene
    s_min = L_min / (steps_per_segment - 1)
    return {
        "per_step_px_mean": float(step.mean()),
        "per_step_px_p05": float(np.percentile(step, 5)),
        "per_step_px_p95": float(np.percentile(step, 95)),
        "step_overlap_mean": float(ov.mean()),
        "step_overlap_p05": float(np.percentile(ov, 5)),
        "step_overlap_p95": float(np.percentile(ov, 95)),
        "per_step_px_best_case": float(s_min),
        "overlap_east_best_case": analytic_overlap(0.0, s_min, window_size),
    }


# =============================================================================
# 自检
# =============================================================================

def _self_test(allow_skip: bool = False) -> int:  # noqa: C901
    fails: List[str] = []

    def check(name: str, cond: bool, info: str = "") -> None:
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, (" — " + info) if info else ""))
        if not cond:
            fails.append(name)

    WH = (128, 222)

    print("\n[1] 与**真实** v5 `_calculate_overlap` / `_get_bounds` 逐位一致")
    _root, _skip_code = resolve_or_exit_code(
        allow_skip, "与真实 _calculate_overlap 的 400 窗逐位一致性 [1]/[6]")
    _real = None
    if _root is not None:
        import importlib.util
        _p = _root + "/wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py"
        try:
            _spec = importlib.util.spec_from_file_location("_v5", _p)
            _real = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_real)
        except Exception as e:
            print("  [FAIL] 找到工程根但导入 v5 失败：%r" % (e,))
            fails.append("导入 build_multi_direction_benchmark_v5")
    if _real is not None:
        ex = _real.MultiDirectionExtractor.__new__(_real.MultiDirectionExtractor)
        rng = np.random.default_rng(3)
        mism = 0
        for _ in range(400):
            c1 = (int(rng.integers(0, 512)), int(rng.integers(0, 512)))
            c2 = (int(rng.integers(0, 512)), int(rng.integers(0, 512)))
            b1r = ex._get_bounds(c1, WH, (512, 512))
            b2r = ex._get_bounds(c2, WH, (512, 512))
            b1m = window_bounds(c1, WH, (512, 512))
            b2m = window_bounds(c2, WH, (512, 512))
            if b1r != b1m or b2r != b2m:
                mism += 1
                continue
            if abs(ex._calculate_overlap(b1r, b2r) - window_overlap(b1m, b2m)) > 1e-12:
                mism += 1
        check("400 组随机窗口 bounds + overlap 逐位一致", mism == 0, "mismatches=%d" % mism)

    print("\n[2] 解析式与整数实现一致（未触边时）")
    for dx in [0, 10, 50, 111, 200, 221, 222, 300]:
        a = analytic_overlap(0, dx, WH)
        n = overlap_from_centers((800, 800), (800, 800 + dx), WH)
        check("dx=%d 解析 vs 整数（差 <0.01）" % dx, abs(a - n) < 0.01, "%.4f vs %.4f" % (a, n))
    check("完全错开 → 0", overlap_from_centers((0, 0), (0, 400), WH) == 0.0)
    check("同一位置 → 1", abs(overlap_from_centers((300, 300), (300, 300), WH) - 1.0) < 1e-9)

    print("\n[3] 核实既有默认配置的帧间重叠**分布**（论文引用的那个数）")
    d = audit_default_config()
    print("     每步位移 均值 %.2f px（p05 %.2f / p95 %.2f）；帧间重叠 均值 %.3f（p05 %.3f / p95 %.3f）"
          % (d["per_step_px_mean"], d["per_step_px_p05"], d["per_step_px_p95"],
             d["step_overlap_mean"], d["step_overlap_p05"], d["step_overlap_p95"]))
    print("     对照·最短段（此前被误当作典型值）：每步 %.2f px、东向重叠 %.3f"
          % (d["per_step_px_best_case"], d["overlap_east_best_case"]))
    check("典型每步位移显著大于最短段那一档（≥1.5×）",
          d["per_step_px_mean"] > 1.5 * d["per_step_px_best_case"],
          "%.2f vs %.2f px" % (d["per_step_px_mean"], d["per_step_px_best_case"]))
    check("典型帧间重叠仍在高位（>0.80）——即默认配置确实是稠密链",
          d["step_overlap_mean"] > 0.80, "%.3f" % d["step_overlap_mean"])
    check("但已进入既有 Farnebäck 认证区间的边缘（每步 p95 > 16 px）",
          d["per_step_px_p95"] > 16.0, "p95=%.2f px" % d["per_step_px_p95"])

    print("\n[4] 反解：让 step-overlap 成为可扫自变量")
    bad_solve = []
    n_case = 0
    for tgt in [0.9, 0.7, 0.5, 0.3, 0.1]:
        for bearing in [0.0, 90.0, 45.0, 137.0]:
            n_case += 1
            n = steps_for_target_overlap(600.0, tgt, WH, bearing)
            th = math.radians(bearing)
            s = 600.0 / n
            got = analytic_overlap(-math.sin(th) * s, math.cos(th) * s, WH)
            if got < tgt - 1e-6:                        # ceil 只会让重叠更高，不会更低
                bad_solve.append((tgt, bearing, round(got, 4)))
    check("反解在 %d 个 (目标×方位角) 组合上都达标" % n_case, not bad_solve,
          "未达标=%s" % bad_solve[:4])
    # 故障注入：用 floor 代替 ceil，必须出现未达标的组合，证明上条断言有咬合力
    bad_floor = []
    for tgt in [0.9, 0.7, 0.5, 0.3, 0.1]:
        for bearing in [0.0, 90.0, 45.0, 137.0]:
            n = max(1, steps_for_target_overlap(600.0, tgt, WH, bearing) - 1)   # 少走一步
            th = math.radians(bearing)
            s = 600.0 / n
            if analytic_overlap(-math.sin(th) * s, math.cos(th) * s, WH) < tgt - 1e-6:
                bad_floor.append((tgt, bearing))
    check("故障注入：步数减一后确有组合跌破目标（证明上条断言非恒真）",
          len(bad_floor) > 0, "跌破 %d 个组合" % len(bad_floor))
    check("目标越低 → 步数越少", steps_for_target_overlap(600, 0.1, WH, 0)
          < steps_for_target_overlap(600, 0.9, WH, 0))
    check("非法目标必须抛错", _raises(lambda: steps_for_target_overlap(600, 1.0, WH, 0)))
    check("非正段长必须抛错", _raises(lambda: steps_for_target_overlap(0, 0.5, WH, 0)))

    print("\n[5] 逐帧审计与落带硬失败")
    path = [(800, 500 + 20 * t) for t in range(9)]
    a = audit_path(path, WH, band=(0.80, 1.00))
    check("均匀东向路径落带", a["in_band"] is True, str(a["violating_steps"]))
    check("中位重叠 ≈0.91", abs(a["step_overlap_median"] - 0.91) < 0.02,
          "%.4f" % a["step_overlap_median"])
    bad_path = [(800, 500), (800, 520), (800, 900), (800, 920)]   # 中间一步跳出去
    b = audit_path(bad_path, WH, band=(0.80, 1.00))
    check("越界步被点名（不是静默通过）", b["in_band"] is False and b["violating_steps"] == [2],
          str(b["violating_steps"]))
    check("endpoint 与 step 是两个不同的数（这正是旧实现只看前者的问题）",
          abs(b["endpoint_overlap"] - b["step_overlap_median"]) > 0.1,
          "ep=%.3f step_med=%.3f" % (b["endpoint_overlap"], b["step_overlap_median"]))
    check("T<2 必须抛错", _raises(lambda: audit_path([(0, 0)], WH)))

    print("\n[6] 故障注入：把重叠算成对称 IoU，必须与真实实现分歧")
    if _real is not None:
        ex = _real.MultiDirectionExtractor.__new__(_real.MultiDirectionExtractor)
        # 造一个 b1 被场景边界裁掉的情形 → area1 != area2 → 非对称口径才对得上
        c1, c2 = (5, 5), (60, 120)
        b1 = window_bounds(c1, WH, (512, 512))
        b2 = window_bounds(c2, WH, (512, 512))
        real_v = ex._calculate_overlap(b1, b2)
        inter_h = max(0, min(b1[1], b2[1]) - max(b1[0], b2[0]))
        inter_w = max(0, min(b1[3], b2[3]) - max(b1[2], b2[2]))
        inter = inter_h * inter_w
        a1 = (b1[1] - b1[0]) * (b1[3] - b1[2])
        a2 = (b2[1] - b2[0]) * (b2[3] - b2[2])
        iou = inter / (a1 + a2 - inter) if (a1 + a2 - inter) > 0 else 0.0
        check("边界裁剪下 非对称口径 ≠ IoU（故不得随手改成 IoU）",
              abs(real_v - iou) > 1e-6 and abs(real_v - window_overlap(b1, b2)) < 1e-12,
              "real=%.4f iou=%.4f" % (real_v, iou))

    print("\n" + "=" * 66)
    if fails:
        print("失败 %d 项：%s" % (len(fails), fails))
        return 1
    if _skip_code:
        print("本地断言全通过，但跨模块断言被跳过 —— 退出码 %d（跳过不算通过）。" % _skip_code)
        return _skip_code
    print("全部通过。")
    return 0


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="帧间 step-overlap 的审计与反解")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--audit-default", action="store_true", help="打印既有默认配置的帧间重叠")
    ap.add_argument("--allow-skip", action="store_true",
                    help="没有既有工程时允许跳过跨模块断言并仍返回 0（默认返回 2）")
    a = ap.parse_args()
    if a.self_test:
        return _self_test(allow_skip=a.allow_skip)
    if a.audit_default:
        for k, v in audit_default_config().items():
            print("%-22s %.4f" % (k, v))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
