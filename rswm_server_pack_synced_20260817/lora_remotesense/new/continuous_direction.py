#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""continuous_direction.py — 连续方位角指令通道及其打分靶（P0 改动 #4/#5 的实现）。

**为什么存在**：既有工程把连续采样的轨迹角在写进 prompt 前量化成 4 个词
（`build_dataset_from_benchmark.py::_make_prompt`），评测端 `evaluate.py` 用同一套分箱
做打分靶。于是"连续方向 θ∈[0,2π)"这句话**没有可测目标**，且自带 22.5° 量化地板。
本模块提供连续方位角的**生成端**与**评测端**实现，两端必须同时替换、同一个 commit 落地
——只换一端会得到一张看起来正常、实则错 45° 的表。

**约定**（与既有工程逐字一致，勿改）：
  * 相机角 θ_cam = degrees(atan2(-(cy₁-cy₀), cx₁-cx₀)) % 360，east=0/north=90/west=180/south=270；
  * prompt 里写的是**相机**方向；光流/配准估计的是**内容**运动；
    滑窗实现下两者恰好相反，故**打分靶 = 相机角 + 180°**（沿用 evaluate.py 的既有结论）。

**向后兼容**：`schema="net"` 完全复刻旧的 4 词式，逐位可复现旧行为，作为
RSWISE 兼容臂与量化地板对照保留。

自检：`python3 code/continuous_direction.py --self-test`
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from pack_path import resolve_or_exit_code  # noqa: E402

# 与 evaluate.py::_DIR_WORD_TO_CAMERA_ANGLE 一致
_DIR_WORD_TO_CAMERA_ANGLE = {"E": 0.0, "N": 90.0, "W": 180.0, "S": 270.0}
_WORD_OF_ANGLE = ["east", "north", "west", "south"]

SCHEMAS = ("net", "net_continuous", "per_segment_continuous", "stripped")

# "toward a bearing of 137 degrees" / "bearing 41 deg" 都命中；只认整数或一位小数，
# 避免把场景描述里的其它数字误当方位角。
_BEARING_RE = re.compile(r"bearing\s+(?:of\s+)?(\d{1,3}(?:\.\d)?)\s*(?:degrees|degree|deg)\b",
                         re.IGNORECASE)


# =============================================================================
# 角度基元
# =============================================================================

def camera_angle_from_waypoints(waypoints: Sequence, *, legacy_f32: bool = False
                                ) -> Optional[float]:
    """(K+1,2) 的 (cy,cx) 相机航点 → 净**相机**方位角（度）。无位移返回 None。

    `legacy_f32=True` 时用 **float32 + np.arctan2** 逐位复刻既有实现
    （`build_dataset_from_benchmark.py::_make_prompt` 与 `evaluate.py` 都先
    `np.asarray(..., dtype=np.float32)`）。这不是洁癖：在**恰好的分箱边界**上
    （实测 45.000° / 225.000°）float32 与 float64 会落进不同的箱，
    使"旧行为对照臂"与历史数据对不上。连续通路不受影响，仍用 float64。
    """
    if waypoints is None or len(waypoints) < 2:
        return None
    if legacy_f32:
        wp = np.asarray(waypoints, dtype=np.float32)
        dy = wp[-1, 0] - wp[0, 0]
        dx = wp[-1, 1] - wp[0, 1]
        if float(np.hypot(dy, dx)) < 1e-6:
            return None
        return float(np.degrees(np.arctan2(-dy, dx)) % 360)
    wp = np.asarray(waypoints, dtype=np.float64)
    dy = wp[-1, 0] - wp[0, 0]
    dx = wp[-1, 1] - wp[0, 1]
    if math.hypot(dy, dx) < 1e-6:
        return None
    return float(math.degrees(math.atan2(-dy, dx)) % 360.0)


def quantize_to_word(angle_deg: float) -> str:
    """把相机角量化成 4 词（**逐位复刻** build_dataset_from_benchmark.py:806-814 的分箱）。"""
    a = float(angle_deg) % 360.0
    if a < 45 or a >= 315:
        return "east"
    if 45 <= a < 135:
        return "north"
    if 135 <= a < 225:
        return "west"
    return "south"


def content_target_from_camera_angle(cam_deg: Optional[float]) -> Optional[float]:
    """相机角 → 打分靶（内容角）= 相机角 + 180°。None 传播为 None（不填 0）。"""
    if cam_deg is None:
        return None
    return float((float(cam_deg) + 180.0) % 360.0)


# =============================================================================
# 生成端：prompt 构造
# =============================================================================

def make_prompt(
    scene_sentence: str,
    num_segments: int,
    *,
    waypoints: Optional[Sequence] = None,
    segment_angles: Optional[Sequence] = None,
    schema: str = "net_continuous",
) -> str:
    """构造指令 prompt。`scene_sentence` 是已经做好的场景句（如 "A satellite video of ..."）。

    schema:
      net                     4 词式（旧行为，RSWISE 兼容臂 / 量化地板对照）
      net_continuous          连续净方位角                            ← 本文主通路
      per_segment_continuous  逐段连续方位角（最强控制信号）
      stripped                删除一切方向信息（归因对照臂）
    """
    if schema not in SCHEMAS:
        raise ValueError("未知 schema %r，只支持 %s" % (schema, list(SCHEMAS)))
    base = scene_sentence.rstrip()
    if not base.endswith("."):
        base += "."

    if schema == "stripped":
        return f"{base} The camera follows a {num_segments}-segment trajectory."

    if schema == "per_segment_continuous":
        if segment_angles is None or len(segment_angles) == 0:
            raise ValueError("per_segment_continuous 需要 segment_angles")
        legs = ", then ".join(
            "on a bearing of %.1f degrees" % (float(a) % 360.0) for a in segment_angles
        )
        return (f"{base} The camera follows a {num_segments}-segment trajectory, "
                f"first {legs}.")

    # schema="net" 是旧行为对照臂 → 连 dtype 一起复刻，否则分箱边界会与历史数据不一致
    cam = (camera_angle_from_waypoints(waypoints, legacy_f32=(schema == "net"))
           if waypoints is not None else None)
    if cam is None and segment_angles is not None and len(segment_angles):
        cam = float(segment_angles[0]) % 360.0
    if cam is None:
        return base

    if schema == "net":
        return (f"{base} The camera follows a {num_segments}-segment trajectory "
                f"with net displacement toward the {quantize_to_word(cam)}.")
    return (f"{base} The camera follows a {num_segments}-segment trajectory "
            f"with net displacement toward a bearing of {cam:.1f} degrees.")


# =============================================================================
# 评测端：打分靶解析（必须与生成端同一个 commit 落地）
# =============================================================================

def parse_bearings(prompt: str) -> List[float]:
    """按出现顺序解析 prompt 里所有连续方位角（相机角，度）。无则空列表。"""
    out: List[float] = []
    for m in _BEARING_RE.finditer(prompt or ""):
        v = float(m.group(1))
        if 0.0 <= v < 360.0:
            out.append(v)
    return out


def parse_direction_word(prompt: str) -> Optional[str]:
    """旧的 4 词解析（**逐位复刻** evaluate.py::parse_direction_word_from_prompt）。"""
    p = (prompt or "").lower()
    for w, k in (("east", "E"), ("north", "N"), ("west", "W"), ("south", "S")):
        if w in p:
            return k
    return None


def detect_schema(prompt: str) -> str:
    """判定 prompt 属于哪种 schema。**先查连续式**：连续式不含基数词，不会互相误命中。"""
    bearings = parse_bearings(prompt)
    if len(bearings) > 1:
        return "per_segment_continuous"
    if len(bearings) == 1:
        return "net_continuous"
    if parse_direction_word(prompt) is not None:
        return "net"
    return "stripped"


def direction_target_from_prompt(prompt: str) -> Tuple[Optional[float], str]:
    """prompt → (打分靶内容角, 判定出的 schema)。方向被剥离时返回 (None, "stripped")。

    这是替换 `evaluate.py::parse_direction_angle_from_prompt` 的函数：旧实现只认 4 个词，
    连续式 prompt 会**静默**落 None，导致整块 dir_* 统计消失而不报错。
    """
    schema = detect_schema(prompt)
    if schema in ("net_continuous", "per_segment_continuous"):
        b = parse_bearings(prompt)
        return content_target_from_camera_angle(b[0] if schema == "net_continuous" else b[0]), schema
    if schema == "net":
        word = parse_direction_word(prompt)
        return content_target_from_camera_angle(_DIR_WORD_TO_CAMERA_ANGLE[word]), schema
    return None, "stripped"


def per_segment_targets_from_prompt(prompt: str) -> Optional[List[float]]:
    """逐段打分靶（内容角）。仅 per_segment_continuous 有意义，其余返回 None。"""
    if detect_schema(prompt) != "per_segment_continuous":
        return None
    return [content_target_from_camera_angle(b) for b in parse_bearings(prompt)]


def direction_target_from_waypoints(waypoints: Sequence, *, quantize: bool = False
                                    ) -> Optional[float]:
    """从航点直接算打分靶。`quantize=True` 复刻旧的 4 分箱行为（对照臂用，含 float32 口径）。"""
    cam = camera_angle_from_waypoints(waypoints, legacy_f32=quantize)
    if cam is None:
        return None
    if quantize:
        cam = _DIR_WORD_TO_CAMERA_ANGLE[{"east": "E", "north": "N",
                                         "west": "W", "south": "S"}[quantize_to_word(cam)]]
    return content_target_from_camera_angle(cam)


def quantization_floor_deg(n_bins: int = 4) -> float:
    """给定分箱数的期望量化地板（度）= 半个箱宽的均匀期望 = 360/(4·n)。

    n=4 → 22.5°，与既有文档一致；连续式 n→∞ → 0。
    """
    if n_bins <= 0:
        raise ValueError("n_bins 必须为正")
    return 360.0 / (4.0 * n_bins)


# =============================================================================
# 自检
# =============================================================================

def _self_test(allow_skip: bool = False) -> int:  # noqa: C901
    fails: List[str] = []

    def check(name: str, cond: bool, info: str = "") -> None:
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name, (" — " + info) if info else ""))
        if not cond:
            fails.append(name)

    SCENE = "A satellite video of a dense built-up area"

    print("\n[1] 旧 4 词式必须逐位复刻既有实现")
    for ang, word in [(0, "east"), (44.9, "east"), (45, "north"), (134.9, "north"),
                      (135, "west"), (224.9, "west"), (225, "south"), (314.9, "south"),
                      (315, "east"), (359.9, "east")]:
        check("量化 %.1f° → %s" % (ang, word), quantize_to_word(ang) == word,
              quantize_to_word(ang))
    p_old = make_prompt(SCENE, 2, segment_angles=[137.0], schema="net")
    check("旧式 prompt 措辞与既有模板一致",
          "net displacement toward the west" in p_old, p_old)
    tgt, sch = direction_target_from_prompt(p_old)
    check("旧式打分靶 = 量化相机角+180 = 0°", sch == "net" and abs(tgt - 0.0) < 1e-9,
          "schema=%s tgt=%s" % (sch, tgt))

    print("\n[2] 连续式：生成→解析必须无损往返（量化地板 = 0）")
    rng = np.random.default_rng(0)
    worst = 0.0
    for a in list(rng.uniform(0, 360, 200)) + [0.0, 0.1, 89.9, 180.0, 359.9]:
        p = make_prompt(SCENE, 3, segment_angles=[a], schema="net_continuous")
        t, s = direction_target_from_prompt(p)
        if s != "net_continuous" or t is None:
            check("连续式解析 %.2f°" % a, False, "schema=%s" % s)
            break
        rec_cam = (t - 180.0) % 360.0
        err = abs((rec_cam - (a % 360.0) + 180.0) % 360.0 - 180.0)
        worst = max(worst, err)
    check("200+ 个随机角往返误差 ≤0.05°（仅一位小数的舍入）", worst <= 0.05,
          "worst=%.4f°" % worst)

    print("\n[3] schema 判定不得互相误命中")
    cases = [
        (make_prompt(SCENE, 2, segment_angles=[10.0], schema="net"), "net"),
        (make_prompt(SCENE, 2, segment_angles=[10.0], schema="net_continuous"), "net_continuous"),
        (make_prompt(SCENE, 3, segment_angles=[10.0, 200.0], schema="per_segment_continuous"),
         "per_segment_continuous"),
        (make_prompt(SCENE, 2, segment_angles=[10.0], schema="stripped"), "stripped"),
    ]
    for p, want in cases:
        check("判定 %s" % want, detect_schema(p) == want, "got=%s | %s" % (detect_schema(p), p[-60:]))
    check("连续式 prompt 不含任何基数方向词（否则旧解析器会误命中）",
          parse_direction_word(cases[1][0]) is None, cases[1][0])
    check("场景句里的 'south'-like 词不会污染 stripped 判定",
          detect_schema("A satellite video of a rural area. "
                        "The camera follows a 2-segment trajectory.") == "stripped")

    print("\n[4] 逐段连续式")
    p_seg = make_prompt(SCENE, 3, segment_angles=[10.0, 200.0], schema="per_segment_continuous")
    b = parse_bearings(p_seg)
    check("解析出 2 个方位角且保序", b == [10.0, 200.0], str(b))
    tg = per_segment_targets_from_prompt(p_seg)
    check("逐段靶 = 各自 +180", tg == [190.0, 20.0], str(tg))
    check("非逐段式调用返回 None", per_segment_targets_from_prompt(cases[0][0]) is None)

    print("\n[5] 故障注入：旧解析器用在连续式 prompt 上必须失效（而不是给出错误的数）")
    p_cont = make_prompt(SCENE, 2, segment_angles=[137.0], schema="net_continuous")
    check("旧 4 词解析器对连续式返回 None（静默丢失的根因，已由 detect_schema 兜住）",
          parse_direction_word(p_cont) is None)
    old_style_target = None
    w = parse_direction_word(p_cont)
    if w is not None:
        old_style_target = (_DIR_WORD_TO_CAMERA_ANGLE[w] + 180.0) % 360.0
    check("若误用旧解析器，靶会是 None → 该样本整块统计消失（本模块改为显式 schema 判定）",
          old_style_target is None)
    new_t, new_s = direction_target_from_prompt(p_cont)
    check("新解析器给出正确靶 (137+180)%%360 = 317", new_s == "net_continuous"
          and abs(new_t - 317.0) < 0.06, "tgt=%s" % new_t)

    print("\n[6] 量化地板与航点靶")
    check("4 分箱地板 = 22.5°", abs(quantization_floor_deg(4) - 22.5) < 1e-9)
    check("36 分箱地板 = 2.5°", abs(quantization_floor_deg(36) - 2.5) < 1e-9)
    wps = [(800.0, 500.0), (700.0, 600.0)]      # 相机向东北走
    cam = camera_angle_from_waypoints(wps)
    check("航点 → 相机角 45°", abs(cam - 45.0) < 1e-6, "%.4f" % cam)
    check("连续靶 = 225°", abs(direction_target_from_waypoints(wps) - 225.0) < 1e-6)
    check("量化靶（对照臂）= 270°（45° 落 north 箱 → 90+180）",
          abs(direction_target_from_waypoints(wps, quantize=True) - 270.0) < 1e-6)
    check("零位移航点 → None（不填 0）", camera_angle_from_waypoints([(1.0, 1.0), (1.0, 1.0)]) is None)
    check("非法 schema 必须抛错", _raises(lambda: make_prompt(SCENE, 2, schema="nope")))

    print("\n[7] 与**真实** build_dataset_from_benchmark._make_prompt 的逐位一致性")
    _root, _skip_code = resolve_or_exit_code(
        allow_skip, "与真实 _make_prompt 的 310 角逐位一致性 [7]")
    _real = None
    if _root is not None:
        import importlib.util
        _p = _root + "/lora_remotesense/build_dataset_from_benchmark.py"
        try:
            _spec = importlib.util.spec_from_file_location("_bdfb", _p)
            _real = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_real)
        except Exception as e:
            print("  [FAIL] 找到工程根但导入 builder 失败：%r" % (e,))
            fails.append("导入 build_dataset_from_benchmark")
    if _real is not None:
        rng2 = np.random.default_rng(1)
        angles = list(rng2.uniform(0, 360, 300)) + [0.0, 45.0, 44.999, 90.0, 135.0,
                                                    224.999, 225.0, 314.999, 315.0, 359.999]
        mism = []
        for a in angles:
            wps = np.array([[0.0, 0.0],
                            [-np.sin(np.radians(a)) * 100.0, np.cos(np.radians(a)) * 100.0]])
            real_p = _real._make_prompt("urban", 2, [float(a)], waypoints=wps)
            mine_p = make_prompt("A satellite video of x", 2, waypoints=wps, schema="net")
            rw = real_p.split("toward the ")[-1].rstrip(".") if "toward the " in real_p else None
            mw = mine_p.split("toward the ")[-1].rstrip(".") if "toward the " in mine_p else None
            if rw != mw:
                mism.append((round(a, 4), rw, mw))
        check("%d 个角（含全部分箱边界）方向词逐位一致" % len(angles), not mism,
              "mismatches=%s" % mism[:4])
        # 故障注入：把 legacy 臂改成 float64（"顺手统一精度"是最容易犯的改动），
        # 必须重新出现不一致 —— 否则上面那条一致性断言是恒真的、没有咬合力。
        mism64 = []
        for a in angles:
            wps = np.array([[0.0, 0.0],
                            [-np.sin(np.radians(a)) * 100.0, np.cos(np.radians(a)) * 100.0]])
            real_p = _real._make_prompt("urban", 2, [float(a)], waypoints=wps)
            rw = real_p.split("toward the ")[-1].rstrip(".") if "toward the " in real_p else None
            cam64 = camera_angle_from_waypoints(wps)          # 故意用 float64
            if cam64 is not None and quantize_to_word(cam64) != rw:
                mism64.append((round(a, 4), rw, quantize_to_word(cam64)))
        check("故障注入：legacy 臂若用 float64 会与真实 builder 分歧（证明上条断言有咬合力）",
              len(mism64) > 0, "float64 下不一致 %d 处，例：%s" % (len(mism64), mism64[:3]))

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
    ap = argparse.ArgumentParser(description="连续方位角指令通道与打分靶")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--allow-skip", action="store_true",
                    help="没有既有工程时允许跳过跨模块断言并仍返回 0（默认返回 2）")
    a = ap.parse_args()
    if a.self_test:
        return _self_test(allow_skip=a.allow_skip)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
