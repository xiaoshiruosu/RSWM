#!/usr/bin/env python3
# =============================================================================
# autopsy_trackb.py — Track B 标定门失败尸检(纯 CPU,只读)
#
# 前提:用 v2.2 包重跑过 run_trajectory_eval.py --with-baselines,输出 JSON
# 含逐样本 per_sample 行(v2.2 新增;旧 JSON 只有聚合值,本脚本会明确报错)。
#
# 针对全量门 FAIL 的两项(oracle net_direction_error=25.10°、
# knots_source_counts 含 uniform_fallback)按证据逐一尸检:
#   1. oracle 净方向:启用样本按误差降序列表(+GT 净位移比/恢复比/场景),
#      mean/median/10% 截尾;误差 vs GT 净位移比的相关(病态放大假设:
#      误差 ~ 1/ratio);ratio≥0.3 / ≥0.5 的受限均值(若 <15° → 均值被
#      低比值尾部拉爆,门的问题在 min_net_ratio 太宽,不在光流)。
#   2. fallback 样本:找出 knots_source=uniform_fallback 的样本,从 H5 读
#      waypoints 算每个内角的折叠转角 —— 若存在 <20°(angle_tol)的转角,
#      确认"clamp 后近共线段"根因(builder 几何边角料,非指标 bug)。
#   3. 覆盖率:net_direction_gated_pct 与 waypoints 净位移比分布
#      (0.1/0.2/0.3 阈值),核对"18% 是被门控比例而非启用比例"。
#   4. 各配置 mean vs median 并排:均值是否被少数样本拉动。
#
# 用法(登录节点):
#   python server_run/autopsy_trackb.py \
#       --trajectory-eval $WORK/trajectory_eval.json \
#       --gt-h5 $WORK/data/train_geo.h5 \
#       --split $WORK/train_r16_sem_s42/train_val_split.json
# =============================================================================
import argparse
import json
import sys

import numpy as np


def folded_turns(waypoints):
    """内角处的折叠转角(0-180°)列表;<20°(angle_tol)即结点检测危险区。"""
    wp = np.asarray(waypoints, dtype=np.float64)
    segs = np.diff(wp, axis=0)
    angs = np.degrees(np.arctan2(segs[:, 0], segs[:, 1]))  # (dy,dx) 排布无所谓,只看差
    turns = []
    for i in range(1, len(segs)):
        d = abs(angs[i] - angs[i - 1]) % 360
        turns.append(360 - d if d > 180 else d)
    return turns


def polyline_net_ratio(points):
    """|净位移| / 折线弧长 —— 与 trajectory_metrics.polyline_net_ratio 同定义。"""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return 0.0
    arc = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
    if arc <= 0:
        return 0.0
    return float(np.linalg.norm(p[-1] - p[0]) / arc)


def stats(vals):
    v = np.asarray(sorted(vals), dtype=np.float64)
    if len(v) == 0:
        return "n=0"
    k = max(1, int(0.05 * len(v)))
    trim = v[k:-k].mean() if len(v) > 2 * k else v.mean()
    return (f"n={len(v)} mean={v.mean():.2f} median={np.median(v):.2f} "
            f"trim10={trim:.2f} max={v.max():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory-eval", required=True)
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("--split", required=True)
    args = ap.parse_args()

    out = json.load(open(args.trajectory_eval))
    oracle = out.get("baselines", {}).get("gt_oracle")
    if oracle is None:
        sys.exit("[FATAL] JSON 无 baselines.gt_oracle —— 需 --with-baselines 的输出")
    ps = oracle.get("per_sample")
    if not ps:
        sys.exit("[FATAL] JSON 无逐样本行(per_sample)—— 旧版输出;"
                 "请先用 v2.2 包重跑 run_trajectory_eval.py 再尸检")

    # H5:场景 + waypoints(fallback 转角、覆盖率都要)
    import h5py
    scenes, wps = {}, {}
    with h5py.File(args.gt_h5, "r") as f:
        data = f["data"] if "data" in f else f
        val_idx = json.load(open(args.split))["val_indices"]
        keys = sorted(data.keys())
        for i in val_idx:
            if i >= len(keys):
                continue
            k = keys[i]
            g = data[k]
            scenes[k] = g.attrs.get("source_scene_category", "?")
            if "waypoints" in g:
                wps[k] = g["waypoints"][:]

    # ---- 1. oracle 净方向尸检 ----
    rows = []
    for k, r in ps.items():
        if r.get("net_direction_error") is None:
            continue
        rows.append((k, float(r["net_direction_error"]),
                     r.get("net_displacement_ratio"),
                     r.get("recovered_net_displacement_ratio"),
                     scenes.get(k, "?")))
    rows.sort(key=lambda t: -t[1])
    errs = np.array([t[1] for t in rows])
    ratios = np.array([t[2] if t[2] is not None else np.nan for t in rows])

    print("=" * 72)
    print("1) gt_oracle 净方向(启用样本,按误差降序;top-15)")
    print("=" * 72)
    print(f"{'sample':<16}{'net_err':>9}{'gt_ratio':>10}{'rec_ratio':>10}  scene")
    for k, e, gr, rr, sc in rows[:15]:
        print(f"{k:<16}{e:>9.2f}{(gr if gr is not None else float('nan')):>10.3f}"
              f"{(rr if rr is not None else float('nan')):>10.3f}  {sc}")
    print(f"\n全体启用: {stats(errs)}")
    ok = ~np.isnan(ratios)
    if ok.sum() > 3:
        c = float(np.corrcoef(errs[ok], ratios[ok])[0, 1])
        print(f"误差 vs GT净位移比 相关: r={c:.3f}(显著负相关→病态放大假设成立)")
        for th in (0.3, 0.5):
            sel = errs[ok & (ratios >= th)]
            print(f"受限均值 ratio≥{th}: {stats(sel)}"
                  + ("   ← <15 则门败于低比值尾部" if len(sel) else ""))

    # 分场景(哪个场景在拖累均值,通常怀疑 flood/低纹理)
    sc_group = {}
    for k, e, gr, rr, sc in rows:
        sc_group.setdefault(sc, []).append(e)
    print("\n分场景:")
    for sc, v in sorted(sc_group.items()):
        print(f"  {sc:<10} {stats(v)}")

    # [v2.3] 分段长度失真假设:逐段方向好(9.19°)但净向差 → 各段光流幅度
    # 低估程度不同,端点被扭。检验 = 误差 vs 相对比例失配 |rec−gt|/gt 的相关。
    recs = np.array([t[3] if t[3] is not None else np.nan for t in rows])
    ok2 = ok & ~np.isnan(recs) & (ratios > 0)
    if ok2.sum() > 3:
        mis = np.abs(recs[ok2] - ratios[ok2]) / ratios[ok2]
        c2 = float(np.corrcoef(errs[ok2], mis)[0, 1])
        print(f"\n误差 vs 比例失配|rec−gt|/gt 相关: r={c2:.3f}"
              f"(显著正相关→分段长度失真假设成立)")

    # [v2.3] 带符号旋偏检验(需 v2.3 输出的 recovered_net_angle /
    # gt_net_content_angle;老 JSON 没有则跳过):所有启用样本的
    # (恢复角−GT角) 带符号圆均值 —— 合成向量长 R 大且方向一致 → 系统性
    # 单向旋偏(光流偏置);R 小 → 对称大误差(场景性失效/离群)。
    sgn = []
    for k, r in ps.items():
        if r.get("net_direction_error") is None:
            continue
        ra, ga = r.get("recovered_net_angle"), r.get("gt_net_content_angle")
        if ra is None or ga is None:
            continue
        d = (float(ra) - float(ga) + 180.0) % 360.0 - 180.0  # 带符号 [-180,180)
        sgn.append(d)
    if sgn:
        th = np.radians(sgn)
        R = float(np.hypot(np.mean(np.cos(th)), np.mean(np.sin(th))))
        mu = float(np.degrees(np.arctan2(np.mean(np.sin(th)), np.mean(np.cos(th)))))
        print(f"\n带符号旋偏: n={len(sgn)} 圆均值={mu:+.1f}° 合成向量长R={R:.3f}")
        print("  判读: R>0.5 且 |圆均值| 明显非零 → 系统性单向旋偏(查光流偏置);"
              "R<0.3 → 对称失效/离群主导")
    else:
        print("\n带符号旋偏: JSON 无 recovered_net_angle(v2.3 前输出)—— 跳过")

    # ---- 2. fallback 样本转角 ----
    print("\n" + "=" * 72)
    print("2) uniform_fallback 样本(clamp 后近共线假设检验;angle_tol=20°)")
    print("=" * 72)
    n_fb = 0
    for k, r in ps.items():
        if r.get("knots_source") != "uniform_fallback":
            continue
        n_fb += 1
        turns = folded_turns(wps[k]) if k in wps else None
        print(f"{k}: 内角折叠转角 = "
              + (", ".join(f"{t:.1f}°" for t in turns) if turns else "无 waypoints"))
        if turns and min(turns) < 20:
            print(f"  → 最小转角 {min(turns):.1f}° < 20°:确认近共线根因"
                  f"(builder 几何边角料,非指标 bug)")
        elif turns:
            print(f"  → 转角全部 ≥20°,近共线假设不成立,需另查(把本输出发回)")
    if n_fb == 0:
        print("(本次输出无 fallback 样本)")

    # ---- 3. 覆盖率 ----
    print("\n" + "=" * 72)
    print("3) 净方向覆盖率(核对 18% = 被门控比例)")
    print("=" * 72)
    print(f"聚合 net_direction_gated_pct = {oracle.get('net_direction_gated_pct')}")
    if wps:
        rv = np.array([polyline_net_ratio(w) for w in wps.values()])
        for th in (0.1, 0.2, 0.3):
            print(f"waypoints 净位移比 ≥{th}: {100.0 * (rv >= th).mean():.1f}% "
                  f"({int((rv >= th).sum())}/{len(rv)})")
        print(f"比值分布: median={np.median(rv):.3f} "
              f"p10={np.percentile(rv, 10):.3f} p90={np.percentile(rv, 90):.3f}")

    # ---- 4. 各配置 mean vs median ----
    print("\n" + "=" * 72)
    print("4) 各配置 mean vs median(均值是否被少数样本拉动)")
    print("=" * 72)
    keys_show = ["net_direction_error", "segment_angular_mean", "turn_recall",
                 "waypoint_error", "frechet", "dtw"]
    for sect in ("configs", "baselines"):
        for name, blk in out.get(sect, {}).items():
            p = blk.get("per_sample")
            if not p:
                continue
            parts = []
            for mk in keys_show:
                v = [r[mk] for r in p.values() if r.get(mk) is not None]
                parts.append(f"{mk}:{np.mean(v):.2f}/{np.median(v):.2f}"
                             if v else f"{mk}:-")
            print(f"[{sect}/{name}] " + "  ".join(parts))
    print("\n[done] 尸检完毕 —— 把整段输出发回分析。")


if __name__ == "__main__":
    main()
