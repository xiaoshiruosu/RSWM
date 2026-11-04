#!/usr/bin/env python3
# =============================================================================
# metric_sample_size_study.py — 保真指标的样本量敏感性(零 GPU 成本)
#
# 为什么需要它:
#   FVD 的常规做法是 2048+ 样本;FID 在 N=100 上基本是噪声。而本 benchmark 的
#   推理成本(~5 min/条)让"直接堆到 2048"不可行。审稿人一定会问:
#   "你在 N=100/400 上报的 FVD 下降 60%,这个估计稳吗?"
#
#   这个脚本的回答方式是:**不额外推理**,在【已经生成好】的结果上做子采样,
#   给出各指标随 N 的收敛曲线与 bootstrap 置信区间。
#     * 如果曲线在 N* 之后平了、CI 收窄到远小于臂间差异 → 这个 N 够用,有证据;
#     * 如果没平 → 主动承认该指标在本 N 下不可解读,把它降级为参考量。
#   两种结果都比"报一个点估计然后被问"强。
#
# 关键取舍(必须理解,否则会误读输出):
#   1. **FID/FVD 是集合级指标,不是逐样本指标**,不能对样本取平均。
#      所以这里用 **子采样重复**(subsample-with-replacement 的 B 次重复)
#      而不是逐样本 bootstrap:每次抽 n 条重算一次 FID/FVD,得到 n 处的分布。
#   2. 子采样得到的 CI 反映的是**估计量随样本集变化的波动**,
#      它**不能**修正 FID/FVD 在小 N 上的**系统性偏差**(FVD 在小 N 上系统性偏高)。
#      偏差要靠"曲线是否随 N 单调漂移"来看,脚本会显式报告漂移量。
#   3. SSIM/PSNR/LPIPS/角误差是逐样本的,对它们用标准逐样本 bootstrap。
#      两类指标的 CI 含义不同,输出里分开标注。
#
# 用法:
#   python server_run/metric_sample_size_study.py \
#       --eval-dirs headline_drop=$WORK/eval_r16_sem_s42 \
#                   headline_native=$WORK/eval_r16_geo_s42 \
#       --sizes 25,50,100,200,400 --repeats 20 \
#       -o $WORK/sample_size_study.json
# =============================================================================
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 逐样本指标:可以做标准 bootstrap
PER_SAMPLE_METRICS = ("SSIM", "PSNR", "LPIPS", "Angular_Error")
# 集合级指标:只能子采样重算(见 _set_level_study)
SET_LEVEL_METRICS = ("FID", "FVD")


def _fatal(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    raise SystemExit(2)


def load_per_sample(eval_dir: Path):
    """读逐样本分数。优先 scores.json;没有则从 meta/ 拼。

    缺逐样本数据时**不静默降级** —— 没有逐样本分数就做不了 bootstrap,
    这时应该如实说"该目录不支持逐样本 CI",而不是拿聚合值假装。
    """
    sc = eval_dir / "scores.json"
    if sc.is_file():
        try:
            d = json.loads(sc.read_text())
            rows = d.get("per_sample") or d.get("rows") or []
            if rows:
                return rows
        except Exception as e:
            print(f"  [warn] {sc} 解析失败: {e}")
    return []


def bootstrap_ci(vals, *, n_boot=2000, alpha=0.05, rng=None):
    """逐样本指标的 bootstrap CI。返回 (mean, lo, hi, n)。"""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None, None, None, 0
    if v.size == 1:
        return float(v[0]), float(v[0]), float(v[0]), 1
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi), int(v.size)


def subsample_set_metric(compute_fn, keys, sizes, repeats, rng):
    """集合级指标的子采样重复。

    compute_fn(subset_keys) -> float。**每次都真算一遍**(FID/FVD 不能由子集
    平均得到)。这是本脚本唯一昂贵的部分,但仍然是 CPU 上的特征提取,
    没有任何额外的视频生成。
    """
    out = {}
    for n in sizes:
        if n > len(keys):
            out[n] = {"skipped": f"n={n} > 可用样本 {len(keys)}"}
            continue
        vals = []
        for r in range(repeats):
            sub = list(rng.choice(keys, size=n, replace=False))
            try:
                vals.append(float(compute_fn(sub)))
            except Exception as e:
                out[n] = {"error": f"第 {r} 次重复失败: {e}"}
                break
        if vals:
            a = np.asarray(vals, dtype=float)
            out[n] = {
                "n": int(n),
                "mean": float(a.mean()), "std": float(a.std(ddof=1) if a.size > 1 else 0.0),
                "p2.5": float(np.percentile(a, 2.5)),
                "p97.5": float(np.percentile(a, 97.5)),
                "repeats": int(a.size),
            }
    return out


def drift_report(curve):
    """曲线是否还在随 N 漂移 —— 这是判断"N 够不够"的主要依据。

    子采样 CI 只说明估计量的波动,**不能**修正小 N 的系统性偏差;
    偏差表现为均值随 N 单调移动。这里量化最大 N 与次大 N 之间的相对漂移。
    """
    # 按条目里的 n 字段排序 —— JSON 序列化会把 int 键变成字符串,
    # 靠键排序会得到 100 < 200 < 25 < 400 < 50 这种字典序。
    items = sorted((v for v in curve.values()
                    if isinstance(v, dict) and "mean" in v and "n" in v),
                   key=lambda v: v["n"])
    if len(items) < 2:
        return None
    a, b = items[-2]["mean"], items[-1]["mean"]
    ns = [items[-2]["n"], items[-1]["n"]]
    if a == 0:
        return None
    rel = abs(b - a) / abs(a)
    return {
        "from_n": ns[0], "to_n": ns[1], "from": a, "to": b,
        "relative_drift": float(rel),
        "verdict": ("已收敛(最后两档相对漂移 <2%)" if rel < 0.02 else
                    "仍在漂移(>2%)—— 该 N 下的估计不可作为定量结论,"
                    "应降级为参考量或加大 N"),
    }


def _mde_table(vals, sizes, rng, repeats):
    """最小可检测差异(MDE)vs 样本量 —— 逐样本指标该回答的问题。

    做法:SE(n)=s/√n(解析,无 FPC),再按双样本比较取
        MDE = (z_{1-α/2} + z_{1-β}) · s·√(2(1−ρ)) / √n,  α=0.05,功效 0.8
    ρ 是两臂在同一批轨迹上的相关系数(配对能消掉轨迹难度这个共享混杂)。
    含义:两个臂的差异小于 MDE 时,在该 N 下**不可分辨** —— 报出来的差
    有可能只是抽样波动。

    这比"报一个点估计"诚实得多,也是审稿人真正会问的问题。
    """
    import numpy as _np
    out = []
    v = _np.asarray(vals, dtype=float)
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    for n in sizes:
        if n > v.size:
            continue
        # [自查修正] SE 必须用解析式 s/√n,**不能**用不放回子采样的经验 std。
        # 不放回子采样含有限总体校正 √(1−n/N):n=200 抽自 N=400 时,
        # 经验 SE 只有真值的 0.707 倍 —— 实测 2.00 vs 解析 2.35。
        # 而 MDE 问的是"一批【独立】的 n 条样本能分辨多大差异",没有 FPC。
        # 低估 SE 会让差异看起来比实际更可分辨,是最危险的方向。
        se = sd / _np.sqrt(n) if n > 0 else float("nan")
        reps = _np.asarray([rng.choice(v, size=n, replace=False).mean()
                            for _ in range(max(repeats, 30))]) if n < v.size else None
        # [核实修正] 1.96·√2·SE 是**50% 功效的检出阈**(真差恰等于它时只有一半
        # 概率显著),不是 MDE。真正的 MDE 要加 z_{1-β}:
        #     MDE = (z_{1-α/2} + z_{1-β}) · σ_d / √n,  σ_d = s·√(2(1−ρ))
        # instrument_floor_report.py 用的正是这个口径;此前这里把 50% 阈叫成
        # MDE,会系统性**低估**"多大的差才检得出",让不可分辨的差看起来可分辨。
        # ρ=0 是保守假设(两臂无配对相关);同一批 val 轨迹上评测时 ρ>0,
        # 实际 MDE 更小,故这里给出 ρ=0/0.3/0.6 三档。
        _Z_A = 1.959963984540054      # z_{0.975}
        _Z_B = 0.8416212335729143     # z_{0.80}
        _mde = {}
        for rho in (0.0, 0.3, 0.6):
            sd_d = se * _np.sqrt(2.0 * (1.0 - rho))
            _mde[f"rho_{rho:g}"] = float((_Z_A + _Z_B) * sd_d)
        out.append({
            "n": int(n),
            "se": float(se),
            "se_analytic": float(se),
            "se_subsample_fpc": (float(reps.std(ddof=1)) if reps is not None
                                 and reps.size > 1 else None),
            # 保留旧键名以免下游断裂,但语义已更正为"检出阈"
            "detect_threshold_50pct_power": float(_Z_A * _np.sqrt(2.0) * se),
            "mde_alpha05_power80": _mde,
            "mde_alpha05": _mde["rho_0"],      # 默认取保守的 ρ=0
            "note": ("SE=s/√n(无有限总体校正);mde_alpha05 = 80% 功效、ρ=0 的"
                     "保守值;detect_threshold_50pct_power 是旧口径(仅 50% 功效),"
                     "两者相差因子 (z_a+z_b)/z_a ≈ 1.43"),
        })
    return out


def _set_level_study(eval_dir, sizes, repeats, rng, *, device="cpu", strict=True):
    """FID/FVD 的子采样重算。

    集合级指标不能由子集平均得到 —— 每个 n、每次重复都要**真算一遍**。
    这是本脚本唯一昂贵的部分,但仍然只是特征提取,**没有任何额外的视频生成**。

    依赖 torch + 特征提取权重。跑不了就抛(strict),不返回一段"说明"假装做过。
    """
    import numpy as _np
    gen_dir, gt_dir = Path(eval_dir) / "generated", Path(eval_dir) / "gt"
    if not gen_dir.is_dir() or not gt_dir.is_dir():
        msg = f"{eval_dir} 缺 generated/ 或 gt/ —— 无法做集合级子采样"
        if strict:
            raise SystemExit(f"[FATAL] {msg}")
        return {"unavailable": msg}
    keys = sorted(p.stem for p in gen_dir.glob("*.npy")
                  if (gt_dir / p.name).is_file())
    if not keys:
        msg = f"{eval_dir}: generated/ 与 gt/ 没有配对的 .npy"
        if strict:
            raise SystemExit(f"[FATAL] {msg}")
        return {"unavailable": msg}

    _here = Path(__file__).resolve().parent.parent / "lora_remotesense" / "new"
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    try:
        from metrics import compute_fid, compute_fvd
    except Exception as e:
        msg = (f"无法 import metrics.compute_fid/compute_fvd({e})—— "
               f"集合级子采样需要 torch 与特征提取权重,请在 GPU 节点跑。")
        if strict:
            raise SystemExit(f"[FATAL] {msg}")
        return {"unavailable": msg}

    def _load(k):
        return (_np.load(gen_dir / f"{k}.npy"), _np.load(gt_dir / f"{k}.npy"))

    def _fid_on(sub):
        g, t = zip(*(_load(k) for k in sub))
        # FID 吃逐帧图像:把每条视频的帧摊平
        gf = [f for v in g for f in v]
        tf = [f for v in t for f in v]
        return float(compute_fid(gf, tf, device=device))

    def _fvd_on(sub):
        g, t = zip(*(_load(k) for k in sub))
        return float(compute_fvd(list(g), list(t), device=device))

    out = {"available_samples": len(keys)}
    for name, fn in (("FID", _fid_on), ("FVD", _fvd_on)):
        out[name] = subsample_set_metric(fn, keys, sizes, repeats, rng)
        # 集合级指标才需要漂移判定 —— 它们的估计量在小 N 上有系统性偏差,
        # 而逐样本均值没有(子采样对它无偏)。
        out[name]["_drift"] = drift_report(out[name])
    return out


def main():
    ap = argparse.ArgumentParser(
        description="保真指标的样本量敏感性研究(在已生成结果上子采样,零 GPU 成本)")
    ap.add_argument("--eval-dirs", nargs="+", required=True,
                    help="name=path 形式,可多个")
    ap.add_argument("--sizes", default="25,50,100,200,400",
                    help="子采样规模,逗号分隔")
    ap.add_argument("--repeats", type=int, default=20,
                    help="每个规模的子采样重复次数(集合级指标)")
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="逐样本指标的 bootstrap 次数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--set-level", action="store_true",
                    help="同时做 FID/FVD 的子采样重算(需 torch + generated/ + gt/,慢;"
                         "依赖不满足会直接失败,不静默跳过)")
    ap.add_argument("--set-level-soft", action="store_true",
                    help="集合级子采样跑不了时降级为记录原因而不是失败(默认失败)")
    ap.add_argument("--device", default="cpu", help="集合级特征提取的 device")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)
    report = {"sizes": sizes, "repeats": args.repeats, "seed": args.seed, "dirs": {}}

    for spec in args.eval_dirs:
        if "=" not in spec:
            _fatal(f"--eval-dirs 需要 name=path 形式,得到 {spec!r}")
        name, path = spec.split("=", 1)
        d = Path(path)
        if not d.is_dir():
            _fatal(f"{d} 不存在")
        entry = {"path": str(d)}

        # ---- 逐样本指标:标准 bootstrap ----
        rows = load_per_sample(d)
        if not rows:
            entry["per_sample"] = {
                "unavailable": "该目录没有 scores.json 的逐样本分数 —— "
                               "无法给出逐样本 CI。evaluate.py 需要落盘 per-sample 才能算。"}
        else:
            ps = {}
            for m in PER_SAMPLE_METRICS:
                vals = [r.get(m) for r in rows if isinstance(r, dict)]
                mean, lo, hi, n = bootstrap_ci(vals, n_boot=args.n_boot, rng=rng)
                if n:
                    ps[m] = {"mean": mean, "ci95_lo": lo, "ci95_hi": hi, "n": n,
                             "ci_width": (hi - lo) if (hi is not None and lo is not None) else None}
            entry["per_sample"] = ps
            # 逐样本指标也给 N-曲线(直接对子集重算均值,便宜)
            curves = {}
            for m in PER_SAMPLE_METRICS:
                vals = np.asarray([r.get(m) for r in rows
                                   if isinstance(r, dict) and r.get(m) is not None
                                   and np.isfinite(r.get(m))], dtype=float)
                if vals.size == 0:
                    continue
                c = {}
                for n in sizes:
                    if n > vals.size:
                        c[n] = {"skipped": f"n={n} > 可用 {vals.size}"}
                        continue
                    reps = np.asarray([rng.choice(vals, size=n, replace=False).mean()
                                       for _ in range(args.repeats)])
                    c[n] = {"n": int(n),
                            "mean": float(reps.mean()),
                            "std": float(reps.std(ddof=1) if reps.size > 1 else 0.0),
                            "p2.5": float(np.percentile(reps, 2.5)),
                            "p97.5": float(np.percentile(reps, 97.5)),
                            "repeats": int(reps.size),
                            # 不放回抽满 = 每次都是全集,std 恒 0。这**不是**
                            # "收敛得极好",而是该档没有抽样波动可言,不能当证据。
                            "degenerate_full_sample": bool(n == vals.size)}
                # [自查修正] **不对逐样本指标报 drift**。从一个固定样本集里
                # 子采样,均值是无偏的 —— 漂移恒为 0,输出"已收敛"是同义反复,
                # 读起来却像证据。drift 只对 FID/FVD 这类**估计量本身有小 N 偏差**
                # 的集合级指标才有意义。
                # 逐样本指标该回答的是另一个问题:**要多大的 N,CI 才窄到
                # 小于臂间差异**(即最小可检测差异)。
                c["_note"] = ("子采样对逐样本均值无偏,故不做漂移判定;"
                              "该看的是 CI 宽度 vs 臂间差异(见 _mde)。")
                c["_mde"] = _mde_table(vals, sizes, rng, args.repeats)
                curves[m] = c
            entry["per_sample_curves"] = curves

        # ---- 集合级指标:必须子采样重算 ----
        if args.set_level:
            # [自查修正] 这里原本是个**空开关**:传了 --set-level 只会拿到一段
            # "how_to" 说明。而 FID/FVD 恰恰是有 N 问题的那两个指标 ——
            # 开关不做事比没有开关更糟(用户以为跑过了)。现在真做,
            # 依赖不满足就**大声失败**,不静默降级。
            entry["set_level"] = _set_level_study(
                d, sizes, args.repeats, rng, device=args.device,
                strict=not args.set_level_soft)
        report["dirs"][name] = entry

    # ---- markdown ----
    md = ["# 保真指标的样本量敏感性", "",
          "> 在**已生成**的结果上子采样,不额外推理。",
          "> 逐样本指标(SSIM/PSNR/LPIPS/角误差)用标准 bootstrap;",
          "> 集合级指标(FID/FVD)只能子采样重算,且子采样 CI **不修正**小 N 的系统性偏差 ——",
          "> 偏差看的是「均值是否随 N 单调漂移」(下面各条的 drift 判定)。", ""]
    for name, e in report["dirs"].items():
        md.append(f"## {name}\n")
        ps = e.get("per_sample")
        if isinstance(ps, dict) and "unavailable" not in ps:
            md.append("| 指标 | 均值 | 95% CI | CI 宽度 | n |")
            md.append("|---|---|---|---|---|")
            for m, v in ps.items():
                md.append(f"| {m} | {v['mean']:.4g} | "
                          f"[{v['ci95_lo']:.4g}, {v['ci95_hi']:.4g}] | "
                          f"{v['ci_width']:.4g} | {v['n']} |")
            md.append("")
        elif isinstance(ps, dict):
            md.append(f"> ⚠️ {ps['unavailable']}\n")
        for m, c in (e.get("per_sample_curves") or {}).items():
            dr = c.get("_drift")
            _pts = sorted((v for v in c.values()
                           if isinstance(v, dict) and "mean" in v and "n" in v),
                          key=lambda v: v["n"])
            md.append(f"**{m} 随 N**:" + ", ".join(
                f"N={v['n']}: {v['mean']:.4g}±{v['std']:.3g}"
                + ("(全集,无抽样波动)" if v.get("degenerate_full_sample") else "")
                for v in _pts))
            if dr:
                md.append(f"  → {dr['verdict']}(N={dr['from_n']}→{dr['to_n']} "
                          f"相对漂移 {dr['relative_drift']:.1%})")
            _mde = c.get("_mde")
            if _mde:
                md.append(f"  最小可检测差异(α=0.05,两臂):" + ", ".join(
                    f"N={x['n']}: ±{x['mde_alpha05']:.4g}" for x in _mde))
                md.append(f"  → 两个臂的 {m} 差异小于对应 MDE 时,在该 N 下**不可分辨**。")
            md.append("")
        sl = e.get("set_level")
        if isinstance(sl, dict) and "unavailable" not in sl:
            for m in ("FID", "FVD"):
                cur = sl.get(m) or {}
                pts = sorted((v for v in cur.values()
                              if isinstance(v, dict) and "n" in v), key=lambda v: v["n"])
                if not pts:
                    continue
                md.append(f"**{m} 随 N(子采样重算,{cur.get('_drift', {}) and ''})**:"
                          + ", ".join(f"N={v['n']}: {v['mean']:.4g}±{v['std']:.3g}" for v in pts))
                dr = cur.get("_drift")
                if dr:
                    md.append(f"  → {dr['verdict']}(N={dr['from_n']}→{dr['to_n']} "
                              f"相对漂移 {dr['relative_drift']:.1%})")
                md.append("")
        elif isinstance(sl, dict):
            md.append(f"> ⚠️ 集合级未做:{sl['unavailable']}\n")
    out_md = "\n".join(md)

    if args.output:
        p = Path(args.output)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        p.with_suffix(".md").write_text(out_md)
        print(f"[done] {p} + {p.with_suffix('.md')}")
    print(out_md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
