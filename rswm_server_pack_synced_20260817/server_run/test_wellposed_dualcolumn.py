#!/usr/bin/env python3
# =============================================================================
# test_wellposed_dualcolumn.py — Track A well-posed 双栏的【端到端】冒烟
#
# 为什么必须走 compute_metrics 这一层:双栏的正确性依赖三段接线 ——
#   ① 推理期 meta 落 gt_net_displacement_ratio;
#   ② compute_metrics 读 meta → net_ratios(区分「键缺失」与「值为 null」);
#   ③ _wellposed_block 在 well_posed=True 的子集上复用逐样本角误差。
# 任何一段断了都不会报错、只是双栏字段悄悄消失(或全表口径混算)。只测其中
# 一个函数抓不到接线错 —— 本测试造真 eval 目录(generated/gt/meta)后调
# compute_metrics,断言产出的键与数值。
#
# 覆盖:
#   T1 正常目录:出 Angular_Error_WellPosed_{Mean,Median,Std,Count}/WellPosed_N/
#      WellPosed_Min_Net_Ratio/WellPosed_Ratio_Missing,且数值只统计 ratio>=门限 的样本
#   T2 ratio=null(无 waypoints)的样本:计入 WellPosed_Ratio_Missing,不进 well-posed 均值
#   T3 v2.6 之前的目录(meta 全无该键):不产双栏、其余指标照常、打 WARNING
#   T4 混版本目录(部分 meta 有键、部分没有)→ 硬失败(不静默混算)
#   T5 scores.json 的逐样本行含 gt_net_displacement_ratio 与 well_posed
#
#   python server_run/test_wellposed_dualcolumn.py
# =============================================================================
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lora_remotesense" / "new"))

for _m in ("torch", "cv2", "lpips", "skimage"):
    if _m not in sys.modules:
        try:
            __import__(_m)
        except Exception:
            sys.modules[_m] = types.ModuleType(_m)

import evaluate as EV  # noqa: E402
import metrics as M    # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


# ---- 打桩:重指标全部替换,只保留角误差与接线逻辑 ----
M.compute_video_ssim = lambda a, b: 0.5
M.compute_video_psnr = lambda a, b: 20.0
M.compute_video_lpips = lambda a, b, device="cpu": 0.3
M.compute_fid = lambda *a, **k: 100.0
M.compute_fvd = lambda *a, **k: 50.0
# 角误差 = 视频首像素值(便于构造已知误差),口径参数照收不误
M.estimate_motion_direction = (
    lambda v, flow_size=None, step=None: float(np.asarray(v)[0, 0, 0, 0]))


def make_eval_dir(root, name, rows):
    """rows: [(key, err_deg, ratio_or_None_or_ABSENT)] —— ABSENT 用字符串 'ABSENT'"""
    d = root / name
    for sub in ("generated", "gt", "meta"):
        (d / sub).mkdir(parents=True)
    for key, err, ratio in rows:
        np.save(d / "generated" / f"{key}.npy",
                np.full((2, 1, 1, 1), err, dtype=np.uint8))
        np.save(d / "gt" / f"{key}.npy", np.zeros((2, 1, 1, 1), dtype=np.uint8))
        meta = {
            "key": key, "prompt": "A satellite video of a mixed remote-sensing scene."
                                  " The camera follows a 2-segment trajectory with"
                                  " net displacement toward the east.",
            "scene": "general", "direction_angle": 0.0,
            "num_frames_gt": 2, "num_frames_gen": 2,
            "pose_window_size": [128, 222],
            # v2.5 口径门要求的标记(缺则 compute_metrics 硬失败 —— 那道门
            # 防的是 v2.5 前后 direction_angle 的 180° 约定差异被静默混算)
            "direction_target_source": "prompt",
            "direction_word": "E",
            "pose_conditioning": "identity_neutralized",
        }
        if ratio != "ABSENT":
            meta["gt_net_displacement_ratio"] = ratio
        (d / "meta" / f"{key}.json").write_text(json.dumps(meta))
    return d


def main():
    root = Path(tempfile.mkdtemp(prefix="wp_"))
    thr = EV.WELL_POSED_MIN_NET_RATIO
    print(f"(well-posed 门限 = {thr},应与 Track B 净方向门控同源)")
    try:
        # ---- T1/T2:误差 0/90/10/100,ratio 0.5/0.05/0.9/None ----
        d = make_eval_dir(root, "eval_ok", [
            ("s0", 0, 0.5), ("s1", 90, 0.05), ("s2", 10, 0.9), ("s3", 100, None)])
        res = EV.compute_metrics(str(d), device="cpu")
        ov = res["overall"]
        print("== T1 双栏键与数值 ==")
        check("all_sample_mean", abs(ov["Angular_Error_Mean"] - 50.0) < 1e-6,
              str(ov.get("Angular_Error_Mean")))
        for k in ("Angular_Error_WellPosed_Mean", "Angular_Error_WellPosed_Median",
                  "Angular_Error_WellPosed_Std", "Angular_Error_WellPosed_Count",
                  "WellPosed_N", "WellPosed_Min_Net_Ratio", "WellPosed_Ratio_Missing"):
            check(f"key:{k}", k in ov, f"缺;实有 {sorted(ov)[:10]}")
        # well-posed 子集 = ratio>=thr 的 s0(0°) 与 s2(10°) → 均值 5
        check("wellposed_mean", abs(ov["Angular_Error_WellPosed_Mean"] - 5.0) < 1e-6,
              str(ov.get("Angular_Error_WellPosed_Mean")))
        check("wellposed_n", ov["WellPosed_N"] == 2, str(ov.get("WellPosed_N")))
        print("== T2 ratio=null 单独计数,不进均值 ==")
        check("ratio_missing_counted", ov["WellPosed_Ratio_Missing"] == 1,
              str(ov.get("WellPosed_Ratio_Missing")))
        check("min_ratio_recorded", abs(ov["WellPosed_Min_Net_Ratio"] - thr) < 1e-9)

        print("== T5 scores.json 逐样本行 ==")
        sc = json.loads((d / "scores.json").read_text())
        rows = sc["per_sample"] if isinstance(sc, dict) and "per_sample" in sc else sc
        r0 = rows[0] if isinstance(rows, list) else None
        check("scores_has_ratio", r0 is not None and "gt_net_displacement_ratio" in r0,
              str(sorted(r0)[:8] if r0 else rows))
        check("scores_has_wellposed", r0 is not None and "well_posed" in r0)

        # ---- T3:v2.6 之前的目录(全无该键)----
        print("== T3 旧目录:不产双栏,其余照常 ==")
        d2 = make_eval_dir(root, "eval_legacy", [
            ("s0", 0, "ABSENT"), ("s1", 90, "ABSENT")])
        res2 = EV.compute_metrics(str(d2), device="cpu")
        ov2 = res2["overall"]
        check("legacy_no_dualcolumn",
              not any(k.startswith("Angular_Error_WellPosed") for k in ov2),
              str([k for k in ov2 if "WellPosed" in k]))
        check("legacy_keeps_basic", "Angular_Error_Mean" in ov2 and "SSIM" in ov2)

        # ---- T4:混版本目录 → 硬失败 ----
        print("== T4 混版本目录 → 硬失败 ==")
        d3 = make_eval_dir(root, "eval_mixed", [
            ("s0", 0, 0.5), ("s1", 90, "ABSENT")])
        try:
            EV.compute_metrics(str(d3), device="cpu")
            check("mixed_version_raises", False, "没抛")
        except RuntimeError as e:
            check("mixed_version_raises", "混版本" in str(e) or "缺" in str(e), str(e)[:80])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n汇总: {len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        print("结果: FAIL —— " + ", ".join(FAIL))
        return 4
    print("结果: 全绿 ✓(well-posed 双栏端到端:meta → net_ratios → 双栏块)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
