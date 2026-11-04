#!/usr/bin/env python3
# =============================================================================
# ladder_report.py — 难度阶梯的曲线报告(纯 CPU,登录节点跑)
#
# 读 ladder_spec.json + 各档 eval 产物,输出 ladder_report.md / ladder_report.json。
#
# 这个脚本存在的理由,是审稿人看阶梯表时会问的三个问题:
#
#  1. "这两档的差异是真的吗?" —— 单看均值答不了。每档 n_eval 只有几十条,
#     well-posed 子集更少。所以每档都给 95% CI(优先用 scores.json 的逐样本
#     角误差做 bootstrap,退化时用 Std/√N 的正态近似),两档 CI 不重叠才算可分辨。
#
#  2. "这个数字是模型学到的,还是仪器噪声?" —— 角误差有两个谁也消不掉的地板:
#       (a) 量化地板:评估目标角来自 prompt 的四方位方向词
#           (evaluate.py: parse_direction_angle_from_prompt),而 prompt 的方向词
#           是把净位移角量化到 E/N/W/S 得到的。于是
#             angle_quantize=4 **且 K=1** 的档 → 净位移即该轴 → 量化残差 0°
#             angle_quantize=4 但 K≥2 → 逐段轴向而净位移不轴向 → 残差仍 ≈22.5°
#             angle_quantize=0 的档 → 真值角连续      → 残差 ≈ U[0,45),期望 22.5°
#           ** 地板逐档不同。把 22.5° 画成一条通用横线是错的,那会让"角度轴带来
#              20+° 难度"这种纯仪器假象被当成结论。**
#       (b) 光流地板:Farnebäck 估计器自身的误差。
#     本脚本优先用【该档 GT 上实测】的地板(51 号跑的 direction_validation.json,
#     它天然把 (a)(b) 合在一起);拿不到时退回"从该档 H5 的 waypoints 精确算出的
#     量化残差"+ 一个带出处的光流地板默认值,并在表里显式标注来源,绝不假装实测。
#
#  3. "LoRA 到底改善了什么?" —— Δ 必须对【同一档的 baseline】算。基模的角误差
#     本身就随难度变化(高重叠、四方向对基模也更容易),拿一个全局 baseline 去减,
#     难度曲线会被基模的难度曲线污染。
#
# 用法:
#   python server_run/ladder_report.py --ladder-root $WORK/ladder
#   python server_run/ladder_report.py --ladder-root DIR --spec server_run/ladder_spec.json \
#          --flow-floor 25.0 --bootstrap 2000 --strict
#
# --strict: 有档缺失、或有档的产物自称 headline 合格、或 overlap 带与 spec 不符时
#           以非零码退出(编排/CI 用)。默认只在报告里列出问题,仍然出表。
# =============================================================================
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 角误差折叠到 [0,180];随机方向的期望 = 90°
RANDOM_EXPECTATION_DEG = 90.0

# 光流地板默认值的出处(必须能追溯,不许是拍脑袋的数)：
FLOW_FLOOR_PROVENANCE = (
    "validate_direction_metric.py v2.3 注释记录的 GT 实测:step=1 口径下 GT 地板 "
    "mean 35.93° / median 24.45°(median≈22.5° 的量化地板)。默认值 25.0° 取的是 "
    "median 那一读数。⚠ 该实测是在 canonical(angle_quantize=0)数据上做的,"
    "它其实是「量化残差 + 光流噪声」的合计,不是纯估计器噪声"
    "(纯估计器在合成平移校验上是 0.020°)。所以把 25° 套到 angle_quantize=4 的档"
    "会高估地板 —— 那些档请以 51 号实测的 direction_validation.json 为准。"
)

_ANGLE_KEYS = ("Angular_Error_WellPosed_Mean", "Angular_Error_Mean", "Angular_Error")


# ---------------------------------------------------------------------------
# 读盘工具
# ---------------------------------------------------------------------------
def _load_json(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _pick_angle(overall: dict) -> Tuple[Optional[float], Optional[str]]:
    """→ (角误差, 用了哪个键)。

    显式判 None 而不是 `or` 链:探针若真学到 0.0°,`or` 会把它当假值吞掉并
    静默改用另一个口径(34 号判门踩过同一个坑)。
    """
    for k in _ANGLE_KEYS:
        v = overall.get(k)
        if v is not None:
            return float(v), k
    return None, None


def _per_sample_errors(eval_dir: Path) -> Optional[np.ndarray]:
    """scores.json 里 well-posed 样本的逐样本角误差(bootstrap 用)。

    只取 well_posed is True 的样本 —— 与 eval_results 的 *_WellPosed_* 同口径,
    否则 CI 和点估计算的不是一批样本。
    """
    d = _load_json(eval_dir / "scores.json")
    if not d:
        return None
    rows = d.get("per_sample") or []
    vals = [r.get("Angular_Error") for r in rows
            if r.get("well_posed") is True and r.get("Angular_Error") is not None]
    if not vals:
        # 没有 well-posed 标记(旧目录)时退回全部有角误差的样本,并由调用方标注口径
        vals = [r.get("Angular_Error") for r in rows if r.get("Angular_Error") is not None]
    return np.asarray(vals, dtype=np.float64) if vals else None


def _ci(vals: Optional[np.ndarray], overall: dict, n_boot: int,
        seed: int = 42) -> Tuple[Optional[float], Optional[float], str]:
    """→ (lo, hi, 方法)。优先 bootstrap,退化到正态近似,再退化到无。"""
    if vals is not None and vals.size >= 5:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
        means = vals[idx].mean(axis=1)
        return (float(np.percentile(means, 2.5)),
                float(np.percentile(means, 97.5)),
                f"bootstrap(n={vals.size}, B={n_boot})")
    std = overall.get("Angular_Error_WellPosed_Std")
    n = overall.get("Angular_Error_WellPosed_Count") or overall.get("WellPosed_N")
    mean, _ = _pick_angle(overall)
    if std is not None and n and int(n) >= 2 and mean is not None:
        se = float(std) / math.sqrt(int(n))
        return mean - 1.96 * se, mean + 1.96 * se, f"正态近似(Std/√{int(n)})"
    return None, None, "无(样本不足或缺 Std)"


def _quantization_residual_from_h5(h5_path: Path) -> Optional[Dict[str, float]]:
    """从该档 H5 的 waypoints 精确算出量化残差 = 该档量化地板。

    推导:评估目标 = 四方位方向词 + 180°;完美模型生成的内容运动角 = GT 净位移角
    + 180°。折叠后误差 = |GT净位移角 − 四方位量化角|。所以这不是近似,是该档
    量化地板的精确值 —— 比 0°/22.5° 那两个解析值更可信(它反映该档实际采到的角分布)。
    """
    try:
        import h5py
    except Exception:
        return None
    if not h5_path.is_file():
        return None
    errs: List[float] = []
    try:
        with h5py.File(str(h5_path), "r") as f:
            for k in f["data"]:
                s = f["data"][k]
                if "waypoints" not in s:
                    continue
                wp = np.asarray(s["waypoints"][:], dtype=np.float64)
                if wp.shape[0] < 2:
                    continue
                # 与 build_dataset_from_benchmark._make_prompt / evaluate.py
                # direction_target_from_waypoints 逐位一致的约定
                ang = math.degrees(math.atan2(-(wp[-1, 0] - wp[0, 0]),
                                              wp[-1, 1] - wp[0, 1])) % 360.0
                card = (0.0 if (ang < 45 or ang >= 315) else
                        90.0 if ang < 135 else
                        180.0 if ang < 225 else 270.0)
                d = abs(((ang - card + 180.0) % 360.0) - 180.0)
                errs.append(d)
    except Exception:
        return None
    if not errs:
        return None
    a = np.asarray(errs)
    return {"mean": float(a.mean()), "median": float(np.median(a)), "n": int(a.size)}


def _measured_gt_floor(rdir: Path) -> Optional[Dict[str, Any]]:
    """51 号跑的 validate_direction_metric.py 输出 → 该档 GT 上的实测地板。"""
    d = _load_json(rdir / "direction_validation.json")
    if not d:
        return None
    blk = (d.get("A_gt_floor") or {}).get("vs_content_target_step1")
    if not isinstance(blk, dict):
        blk = (d.get("A_gt_floor") or {}).get("vs_content_target")
        if not isinstance(blk, dict):
            return None
        which = "vs_content_target(默认 step)"
    else:
        which = "vs_content_target_step1"
    out = {"which": which}
    for k in ("mean", "median", "n"):
        if blk.get(k) is not None:
            out[k] = blk[k]
    return out if ("mean" in out or "median" in out) else None


# ---------------------------------------------------------------------------
# 单档收集
# ---------------------------------------------------------------------------
def collect_rung(rdir: Path, rung: dict, args) -> Dict[str, Any]:
    rid = rung["id"]
    aq = int(rung["angle_quantize"])
    rec: Dict[str, Any] = {
        "id": rid,
        "index": int(rung["index"]),
        "label": rung.get("label"),
        "min_segments": int(rung["min_segments"]),
        "max_segments": int(rung["max_segments"]),
        "angle_quantize": aq,
        "overlap_band": [float(rung["min_overlap_ratio"]), float(rung["max_overlap_ratio"])],
        "frames": rung.get("frames"),
        "chains": rung.get("chains") or [],
        "prev_on_chain": rung.get("prev_on_chain"),
        "moved_axis": rung.get("moved_axis"),
        "external_anchor": rung.get("external_anchor"),
        "dir": str(rdir),
        "issues": [],
        "status": "ok",
    }

    lora = _load_json(rdir / "eval_lora" / "eval_results.json")
    base = _load_json(rdir / "eval_base" / "eval_results.json")
    if lora is None:
        rec["status"] = "missing_lora"
        rec["issues"].append("缺 eval_lora/eval_results.json —— 该档没跑或没跑完")
    if base is None:
        rec["status"] = "missing_base" if rec["status"] == "ok" else rec["status"]
        rec["issues"].append("缺 eval_base/eval_results.json —— 该档没有 Δ 的分母")

    # ---- 数据自证:overlap 审计必须与 spec 声明的带一致 ----
    aud = _load_json(rdir / "overlap_audit.json")
    if aud is None:
        rec["issues"].append("缺 overlap_audit.json —— 无法从产物证明该档的 overlap 带")
    else:
        lo_a = aud.get("min_overlap_ratio")
        hi_a = aud.get("max_overlap_ratio")
        rec["overlap_audit"] = {
            "band": [lo_a, hi_a],
            "enforce": aud.get("enforce"),
            "stats": aud.get("overlap_stats"),
            "n_accepted": aud.get("n_accepted"),
        }
        want = rec["overlap_band"]
        if lo_a is None or hi_a is None or abs(float(lo_a) - want[0]) > 1e-9 \
                or abs(float(hi_a) - want[1]) > 1e-9:
            rec["issues"].append(
                f"overlap 审计记的带 [{lo_a},{hi_a}) 与 spec 的 "
                f"[{want[0]},{want[1]}) 不符 —— 这一档的 x 轴不可信")
        if not aud.get("enforce", False):
            rec["issues"].append("overlap enforce=False —— 带没有被强制,只是碰巧落在里面")

    # ---- 身份:阶梯产物绝不能自称 headline ----
    for arm in ("eval_lora", "eval_base"):
        man = _load_json(rdir / arm / "run_manifest.json")
        if man is None:
            rec["issues"].append(f"{arm} 缺 run_manifest.json —— 口径无从核对")
            continue
        if man.get("role") != "ladder":
            rec["issues"].append(f"{arm} 的 role={man.get('role')!r} ≠ ladder")
        if man.get("headline_eligible"):
            rec["issues"].append(
                f"{arm} 自称 headline_eligible=true —— role=ladder 还没被登记为"
                f"永不入 headline,这份产物会污染主表(见 ladder_spec.json 的 wiring_required)")

    # ---- 指标 ----
    def arm_block(res: Optional[dict], d: Path) -> Dict[str, Any]:
        if not res:
            return {}
        ov = res.get("overall", {})
        ang, key = _pick_angle(ov)
        vals = _per_sample_errors(d)
        lo, hi, method = _ci(vals, ov, args.bootstrap)
        n_all = ov.get("num_samples")
        n_wp = ov.get("Angular_Error_WellPosed_Count")
        if n_wp is None:
            n_wp = ov.get("WellPosed_N")
        return {
            "angle_deg": ang,
            "angle_key": key,
            "ci95": [lo, hi],
            "ci_method": method,
            "n_samples": n_all,
            "n_well_posed": n_wp,
            "well_posed_ratio": (float(n_wp) / n_all) if (n_wp is not None and n_all) else None,
            "pose_mode": ov.get("pose_mode"),
            "headline_eligible": ov.get("headline_eligible"),
            "SSIM": ov.get("SSIM"), "PSNR": ov.get("PSNR"),
            "LPIPS": ov.get("LPIPS"), "FVD": ov.get("FVD"),
        }

    rec["lora"] = arm_block(lora, rdir / "eval_lora")
    rec["base"] = arm_block(base, rdir / "eval_base")

    la, ba = rec["lora"].get("angle_deg"), rec["base"].get("angle_deg")
    rec["delta_vs_base_deg"] = (la - ba) if (la is not None and ba is not None) else None

    # ---- 地板 ----
    floor: Dict[str, Any] = {}
    meas = _measured_gt_floor(rdir)
    quant = _quantization_residual_from_h5(rdir / "data" / f"{rid}.h5")
    # [核实修正 2026-07-30] 解析兜底必须区分段数。
    # aq=4 让每【段】轴向,但 prompt 的方向词量化的是【净位移】——
    # 多段路径的净位移不轴向(东→北的净向约 45°),于是残差回到 ≈U[0,45)。
    # 实测(600 次/格,直接驱动 v5 的 RandomPositionPathGenerator):
    #     aq=4 K=1 → 0.00°     aq=4 K=2 → 21.27°     aq=4 K=4 → 19.77°
    #     aq=0 K=1 → 21.48°    aq=0 K=2 → 23.17°     aq=0 K=4 → 23.14°
    # 原写法对 aq=4 一律给 0°,在 K≥2 的档上低估地板约 20° —— 那正是它自己
    # 声称要防的"仪器假象被当成结论"。实测优先,解析只在拿不到 H5 时兜底。
    # 段数从该档的 min/max_segments 取(spec 的字段名)。只有 min==max==1
    # 才是"确定的单段",K 有范围时净位移一般不轴向,必须按 22.5° 算。
    _kmin = rung.get("min_segments")
    _kmax = rung.get("max_segments")
    try:
        _kmin, _kmax = int(_kmin), int(_kmax)
    except (TypeError, ValueError):
        _kmin = _kmax = 0
    _K = _kmin if (_kmin == _kmax and _kmin > 0) else 0
    if aq == 4 and _K == 1:
        floor["analytic_quantization_deg"] = 0.0
        floor["_analytic_note"] = (
            "angle_quantize=4 且 K=1:唯一一段轴向 → 净位移就是该轴 → 量化残差 0°")
    else:
        floor["analytic_quantization_deg"] = 22.5
        floor["_analytic_note"] = (
            f"angle_quantize={aq} K={_K or '?'}:净位移角不落在四方位上"
            f"(aq=4 只保证逐段轴向,多段的净向不轴向)→ |真值−四方位| ≈ U[0,45),"
            f"期望 22.5°。实测 aq=4 K=2 为 21.3°、K=4 为 19.8°,与此一致。")
    if quant:
        floor["waypoint_quantization_deg"] = quant
    if meas:
        floor["measured_gt_deg"] = meas
    floor["flow_floor_default_deg"] = args.flow_floor
    floor["flow_floor_provenance"] = FLOW_FLOOR_PROVENANCE

    # -----------------------------------------------------------------
    # 报告里用哪个数当"这一档的地板"。三档可信度,不许混为一谈:
    #
    #   measured                 —— 该档 GT 上实测,直接可用。
    #   extrapolated_same_regime —— 没实测,但默认光流地板(25°)本来就是在
    #                               angle_quantize=0 的数据上测的,拿来用在
    #                               angle_quantize=0 的档上属同口径外推,可接受。
    #   lower_bound_only         —— angle_quantize=4 的档没实测。**不能**套 25°:
    #                               那个数里含 22.5° 的量化残差,而这一档的量化
    #                               残差是 0°。套上去会把"模型已达仪器极限"这种
    #                               强结论安在一个纯粹高估的地板上。此时只有下界
    #                               (=量化残差)是真的,触底与否无法判定。
    # -----------------------------------------------------------------
    if meas and meas.get("median") is not None:
        floor["effective_deg"] = float(meas["median"])
        floor["confidence"] = "measured"
        floor["effective_source"] = f"该档 GT 实测({meas['which']}, median)"
    elif meas and meas.get("mean") is not None:
        floor["effective_deg"] = float(meas["mean"])
        floor["confidence"] = "measured"
        floor["effective_source"] = f"该档 GT 实测({meas['which']}, mean)"
    elif aq == 0:
        q = float(quant["median"]) if quant else floor["analytic_quantization_deg"]
        qsrc = ("waypoints 量化残差 median" if quant else "解析量化地板")
        floor["effective_deg"] = max(q, float(args.flow_floor))
        floor["confidence"] = "extrapolated_same_regime"
        floor["effective_source"] = (
            f"未实测·同口径外推:max({qsrc} {q:.1f}°, 光流默认 {args.flow_floor:.1f}°)")
    else:
        q = float(quant["median"]) if quant else floor["analytic_quantization_deg"]
        floor["effective_deg"] = q
        floor["confidence"] = "lower_bound_only"
        floor["effective_source"] = (
            f"未实测·仅下界 {q:.1f}°(angle_quantize=4:默认光流地板 "
            f"{args.flow_floor:.1f}° 是在连续角数据上测的,含 22.5° 量化残差,"
            f"套到本档会高估)")
    rec["floor"] = floor

    # ---- 判读 ----
    rec["verdict"] = _verdict(la, rec["lora"].get("ci95"),
                              floor["effective_deg"], floor["confidence"])
    return rec


def _verdict(angle: Optional[float], ci: Optional[List[Optional[float]]],
             floor: float, confidence: str = "measured") -> str:
    if angle is None:
        return "无角误差"
    lo = hi = None
    if ci:
        lo, hi = ci[0], ci[1]
    if lo is None or hi is None:
        if angle >= RANDOM_EXPECTATION_DEG - 5:
            return "≈随机(无 CI,仅点估计)"
        return "无 CI,不判读"
    if lo <= RANDOM_EXPECTATION_DEG <= hi:
        return "与随机(90°)不可分 —— 该档没学到方向"
    if lo > RANDOM_EXPECTATION_DEG:
        return "显著差于随机 —— 系统性反向,查 prompt/位姿约定"
    if lo <= floor:
        if confidence == "lower_bound_only":
            return (f"低于随机,但地板只有下界({floor:.1f}°)—— 是否已触底无法判定,"
                    f"请补跑该档的 validate_direction_metric.py")
        return f"与地板({floor:.1f}°)不可分 —— 该档已达仪器极限"
    if confidence == "lower_bound_only":
        return (f"显著低于随机;地板未实测(仅下界 {floor:.1f}°),"
                f"距真实地板多远待测")
    return f"介于地板({floor:.1f}°)与随机(90°)之间 —— 可分辨"


def _resolvable(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[Optional[bool], str]:
    """两档差异是否可分辨。规则见 ladder_spec.json 的 instrument_floors.resolvable_rule。"""
    ca, cb = a["lora"].get("ci95"), b["lora"].get("ci95")
    if not ca or not cb or None in ca or None in cb:
        return None, "缺 CI"
    overlap = not (ca[1] < cb[0] or cb[1] < ca[0])
    if overlap:
        return False, "两档 95% CI 重叠"
    aa, ab = a["lora"].get("angle_deg"), b["lora"].get("angle_deg")
    fa, fb = a["floor"]["effective_deg"], b["floor"]["effective_deg"]
    above = ((aa is not None and aa > fa) or (ab is not None and ab > fb))
    if not above:
        return False, "两档都趴在各自地板上,差异无意义"
    # "高于地板"这句话的强度取决于地板本身有多可信:仅有下界时,超过下界并不等于
    # 超过真实地板 —— 结论仍可能是"两档都在仪器噪声里"。如实标注,不含糊过去。
    if "lower_bound_only" in (a["floor"].get("confidence"), b["floor"].get("confidence")):
        return True, "CI 不重叠;但地板仅有下界(该档未实测),'高于地板'这一条待补测"
    return True, "CI 不重叠,且至少一档高于自身地板"


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def _f(x, nd=2, suf=""):
    return "—" if x is None else f"{x:.{nd}f}{suf}"


def _ci_str(ci):
    if not ci or ci[0] is None or ci[1] is None:
        return "—"
    return f"[{ci[0]:.1f}, {ci[1]:.1f}]"


def _k_str(r):
    return str(r["min_segments"]) if r["min_segments"] == r["max_segments"] \
        else f"{r['min_segments']}–{r['max_segments']}"


def render_md(spec: dict, recs: List[Dict[str, Any]], args) -> str:
    L: List[str] = []
    A = L.append
    A("# RSWM-Bench 难度阶梯报告")
    A("")
    A(f"- spec: `{spec.get('spec_version')}`  ·  role: `{spec.get('role')}`  "
      f"·  pose_mode: `{spec.get('protocol', {}).get('pose_mode')}`")
    A(f"- 阶梯根目录: `{args.ladder_root}`")
    A(f"- 档数: {len(recs)}  ·  已出结果: {sum(1 for r in recs if r['lora'].get('angle_deg') is not None)}")
    A("")
    A("> **怎么读这张表**  角误差越小越好;`随机期望 = 90°`。每档有自己的**地板**:")
    A("> 完美模型也拿不到比它更好的分数。地板逐档不同 —— `angle_quantize=4` 的档"
      "量化残差是 **0°**,`angle_quantize=0` 的档是 **22.5°**(目标角来自 prompt 的"
      "四方位方向词,而连续角真值与它天然差 U[0,45))。所以**不要**在整张图上画一条"
      "22.5° 的通用横线。")
    A("")

    # ---- 表 1:档位定义 + 产物自证 ----
    A("## 表 1 · 档位定义与产物自证")
    A("")
    A("| # | 档 | K | 角度粒度 | overlap 带(spec) | overlap 实测 min/p50/max | 帧数 | enforce |")
    A("|---|----|---|---------|------------------|--------------------------|------|---------|")
    for r in recs:
        aud = r.get("overlap_audit") or {}
        st = aud.get("stats") or {}
        obs = ("—" if not st else
               f"{st.get('min', float('nan')):.3f}/{st.get('p50', float('nan')):.3f}/"
               f"{st.get('max', float('nan')):.3f}")
        band = r["overlap_band"]
        A(f"| {r['index']} | `{r['id']}` | {_k_str(r)} | "
          f"{'四方向' if r['angle_quantize'] == 4 else '连续'} | "
          f"[{band[0]:.2f}, {band[1]:.2f}) | {obs} | "
          f"{','.join(str(x) for x in (r.get('frames') or []))} | "
          f"{aud.get('enforce')} |")
    A("")

    # ---- 表 2:结果 ----
    A("## 表 2 · 难度曲线(角误差 ↓ 越小越好;随机 = 90°)")
    A("")
    A("| # | 档 | n(well-posed/全部) | LoRA 角误差 | 95% CI | baseline | Δ vs 该档 baseline | 地板 | 地板来源 | 判读 |")
    A("|---|----|--------------------|-------------|--------|----------|--------------------|------|----------|------|")
    for r in recs:
        lo_, ba_ = r["lora"], r["base"]
        n_txt = (f"{lo_.get('n_well_posed')}/{lo_.get('n_samples')}"
                 if lo_.get("n_samples") is not None else "—")
        wr = lo_.get("well_posed_ratio")
        if wr is not None:
            n_txt += f" ({wr:.0%})"
        fl = r["floor"]
        A(f"| {r['index']} | `{r['id']}` | {n_txt} | {_f(lo_.get('angle_deg'), 1, '°')} | "
          f"{_ci_str(lo_.get('ci95'))} | {_f(ba_.get('angle_deg'), 1, '°')} | "
          f"{_f(r.get('delta_vs_base_deg'), 1, '°')} | "
          f"{_f(fl.get('effective_deg'), 1, '°')} | {fl.get('effective_source')} | "
          f"{r['verdict']} |")
    A("")
    A("*Δ 是原始差值(LoRA − 该档 baseline),不是改善量:角误差越小越好,"
      "**负 Δ = LoRA 更好**。每档都用它自己的 baseline —— 基模的角误差本身随难度变化,"
      "用全局 baseline 减会把基模的难度曲线混进来。*")
    A("")

    # ---- 每条阶梯的逐步 Δ ----
    A("## 表 3 · 逐条阶梯:每步只动一个轴")
    A("")
    by_id = {r["id"]: r for r in recs}
    for cname, cinfo in (spec.get("chains") or {}).items():
        if cname.startswith("_") or not isinstance(cinfo, dict):
            continue
        ids = cinfo.get("rungs") or []
        A(f"### 阶梯 `{cname}` — {cinfo.get('name')}")
        A("")
        A("| 步 | 从 → 到 | 动的轴 | 角误差 从 → 到 | Δ(步) | 可分辨? | 依据 |")
        A("|----|---------|--------|----------------|-------|---------|------|")
        for i in range(1, len(ids)):
            a, b = by_id.get(ids[i - 1]), by_id.get(ids[i])
            if a is None or b is None:
                A(f"| {i} | `{ids[i-1]}` → `{ids[i]}` | — | — | — | — | 档不在报告里 |")
                continue
            aa, ab = a["lora"].get("angle_deg"), b["lora"].get("angle_deg")
            d = (ab - aa) if (aa is not None and ab is not None) else None
            res, why = _resolvable(a, b)
            A(f"| {i} | `{ids[i-1]}` → `{ids[i]}` | {b.get('moved_axis') or '—'} | "
              f"{_f(aa,1,'°')} → {_f(ab,1,'°')} | {_f(d,1,'°')} | "
              f"{'是' if res else ('否' if res is False else '—')} | {why} |")
        A("")
        # 跨这一步若同时跨了角度轴,提醒扣地板
        if any((by_id.get(x) or {}).get("moved_axis") == "angle_quantize" for x in ids[1:]):
            A("> ⚠ 这条阶梯跨了**角度粒度轴**:两端的量化地板不同"
              "(四方向 0° vs 连续 22.5°)。这一步的 Δ 里有一部分是纯仪器,"
              "先减掉两端地板之差再谈难度。")
            A("")

    # 交互提示
    ae = (spec.get("chains") or {}).get("A_easy", {}).get("rungs") or []
    ah = (spec.get("chains") or {}).get("A_hard", {}).get("rungs") or []
    if len(ae) == 2 and len(ah) == 2:
        def step(pair):
            """→ (扣掉两端地板差之后的净抬升, 说明)。

            扣地板只有在【两端地板都可信】时才成立:一端只有下界时,fb-fa 本身
            就是个假数,减出来的"净抬升"会比不减还误导(实测中见过 -5° 这种
            物理上说不通的结果)。此时如实返回 None。
            """
            a, b = by_id.get(pair[0]), by_id.get(pair[1])
            if not a or not b:
                return None, "缺档"
            aa, ab = a["lora"].get("angle_deg"), b["lora"].get("angle_deg")
            if aa is None or ab is None:
                return None, "缺角误差"
            ca, cb = a["floor"].get("confidence"), b["floor"].get("confidence")
            if "lower_bound_only" in (ca, cb):
                return None, (f"两端地板可信度不同({ca} / {cb})—— 扣地板无意义,"
                              f"请先补跑该档的 validate_direction_metric.py")
            fa, fb = a["floor"]["effective_deg"], b["floor"]["effective_deg"]
            return (ab - aa) - (fb - fa), ""
        s1, w1 = step(ae)
        s2, w2 = step(ah)
        A("### 角度轴 × overlap 轴 的交互")
        A("")
        A(f"- 容易 overlap 带上跨角度轴(`{ae[0]}`→`{ae[1]}`):"
          + (f"扣地板后净抬升 **{s1:+.1f}°**" if s1 is not None else f"不可算({w1})"))
        A(f"- 困难 overlap 带上跨角度轴(`{ah[0]}`→`{ah[1]}`):"
          + (f"扣地板后净抬升 **{s2:+.1f}°**" if s2 is not None else f"不可算({w2})"))
        if s1 is not None and s2 is not None:
            A(f"- 两者相差 **{abs(s1 - s2):.1f}°**。差得小 → 角度轴的代价与 overlap 水平"
              f"基本无关,三个轴可近似当独立项读;差得大 → 存在交互,"
              f"**不能**把曲线拆成三个可加的贡献。多大算大,拿两条 CI 的宽度作尺子。")
        else:
            A("- 两端不可比,交互暂无结论(别用点估计硬凑)。")
        A("")

    # ---- 参考线 ----
    A("## 参考线与仪器地板")
    A("")
    A(f"- **随机期望**: {RANDOM_EXPECTATION_DEG:.0f}°(方向折叠到 [0,180] 后的均匀随机)")
    A("- **量化地板**(逐档):`angle_quantize=4` → 0° ;`angle_quantize=0` → 22.5°")
    A(f"- **光流地板默认值**: {args.flow_floor:.1f}°")
    A(f"  - 出处:{FLOW_FLOOR_PROVENANCE}")
    A("- **实测优先**:51 号作业会对每档 GT 跑 `validate_direction_metric.py`,"
      "输出落在各档的 `direction_validation.json`。表 2 的「地板来源」列写明每一档"
      "用的是实测值还是回退值 —— 写着「未实测」的行,别把地板当硬结论。")
    A("")
    A("**可分辨判据**:两档差异算可分辨,需要 (a) 两档 95% CI 不重叠,"
      "且 (b) 至少一档明显高于它自己那一档的地板。只满足 (a) 而两档都贴着地板,"
      "差的只是仪器噪声。")
    A("")

    # ---- 外部锚点 ----
    anchors = [r for r in recs if r.get("external_anchor")]
    if anchors:
        A("## 外部锚点(不在同一条曲线上)")
        A("")
        for r in anchors:
            A(f"- `{r['id']}` ↔ {r['external_anchor']}")
        A("")
        A("> 这些是**协议不同**的参照:34b 是 n=20 的训练集过拟合探针,"
          "23 号是 N_MAIN 全量的泛化实验。阶梯把两端在同一协议下重跑了一遍,"
          "就是为了让曲线内部可比;锚点只用来对照,不画进曲线。")
        A("")

    # ---- 问题 ----
    probs = [(r["id"], i) for r in recs for i in r["issues"]]
    A("## 健康检查")
    A("")
    if not probs:
        A("全部档通过:产物齐全、overlap 带与 spec 一致、role=ladder 且未自称 headline。")
    else:
        A(f"发现 {len(probs)} 个问题:")
        A("")
        for rid, msg in probs:
            A(f"- `{rid}`: {msg}")
    A("")
    A("## 砍掉的格子(spec.cut_cells)")
    A("")
    for c in spec.get("cut_cells") or []:
        why = c.get("why")
        why = " ".join(why) if isinstance(why, list) else str(why)
        A(f"- **{c.get('what')}** — {why}")
    A("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="难度阶梯报告:读各档 eval 产物 → markdown + json 难度曲线表")
    ap.add_argument("--ladder-root", required=True,
                    help="50/51 号的 LADDER_ROOT(其下每档一个子目录)")
    ap.add_argument("--spec", default=None,
                    help="ladder_spec.json(默认取 <本脚本目录>/ladder_spec.json)")
    ap.add_argument("-o", "--output-prefix", default=None,
                    help="输出前缀(默认 <ladder-root>/ladder_report)")
    ap.add_argument("--flow-floor", type=float, default=25.0,
                    help="光流估计器地板默认值(度);仅在该档没有实测地板时使用")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap 重采样次数")
    ap.add_argument("--strict", action="store_true",
                    help="有缺档/自称 headline/overlap 带不符时以非零码退出")
    args = ap.parse_args()

    spec_p = Path(args.spec) if args.spec else Path(__file__).resolve().parent / "ladder_spec.json"
    spec = _load_json(spec_p)
    if spec is None:
        print(f"FATAL: 读不了 spec {spec_p}", file=sys.stderr)
        return 1
    root = Path(args.ladder_root)
    if not root.is_dir():
        print(f"FATAL: {root} 不是目录", file=sys.stderr)
        return 1

    recs = [collect_rung(root / r["id"], r, args) for r in spec["rungs"]]

    prefix = Path(args.output_prefix) if args.output_prefix else root / "ladder_report"
    md_p = prefix.with_suffix(".md")
    js_p = prefix.with_suffix(".json")

    payload = {
        "spec_version": spec.get("spec_version"),
        "ladder_root": str(root),
        "role": spec.get("role"),
        "pose_mode": (spec.get("protocol") or {}).get("pose_mode"),
        "random_expectation_deg": RANDOM_EXPECTATION_DEG,
        "flow_floor_default_deg": args.flow_floor,
        "flow_floor_provenance": FLOW_FLOOR_PROVENANCE,
        "rungs": recs,
        "chain_steps": [],
        "issues": [{"rung": r["id"], "issue": i} for r in recs for i in r["issues"]],
    }
    by_id = {r["id"]: r for r in recs}
    for cname, cinfo in (spec.get("chains") or {}).items():
        if cname.startswith("_") or not isinstance(cinfo, dict):
            continue
        ids = cinfo.get("rungs") or []
        for i in range(1, len(ids)):
            a, b = by_id.get(ids[i - 1]), by_id.get(ids[i])
            if a is None or b is None:
                continue
            aa, ab = a["lora"].get("angle_deg"), b["lora"].get("angle_deg")
            res, why = _resolvable(a, b)
            payload["chain_steps"].append({
                "chain": cname, "from": ids[i - 1], "to": ids[i],
                "moved_axis": b.get("moved_axis"),
                "angle_from": aa, "angle_to": ab,
                "delta_deg": (ab - aa) if (aa is not None and ab is not None) else None,
                "floor_from": a["floor"]["effective_deg"],
                "floor_to": b["floor"]["effective_deg"],
                "resolvable": res, "resolvable_reason": why,
            })

    md_p.parent.mkdir(parents=True, exist_ok=True)
    md_p.write_text(render_md(spec, recs, args), encoding="utf-8")
    js_p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    print(f"[ladder] markdown → {md_p}")
    print(f"[ladder] json     → {js_p}")

    n_done = sum(1 for r in recs if r["lora"].get("angle_deg") is not None)
    print(f"[ladder] {n_done}/{len(recs)} 档有结果;问题 {len(payload['issues'])} 条")
    for it in payload["issues"]:
        print(f"   - {it['rung']}: {it['issue']}")
    if args.strict and (payload["issues"] or n_done < len(recs)):
        print("[ladder] --strict:存在问题或有档未完成,非零退出", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
