#!/usr/bin/env python
"""collect_results.py — 汇总 server_run 全部实验结果为一张 Markdown 总表。

用法: python server_run/collect_results.py [WORK_DIR]
输出: WORK_DIR/summary.md + summary.json
"""
import json
import sys
from pathlib import Path

METRICS = ["SSIM", "PSNR", "LPIPS", "FID", "FVD", "Angular_Error_Mean"]
# [v2.9.1 拍板] Δ 是【原始差值】,不是"改善量"。方向必须标在表头,否则正 Δ 会被
# 一律读成改善 —— 而 LPIPS/FID/FVD/角误差全部越小越好,正 Δ 恰恰是变差。
METRIC_DIR = {"SSIM": "↑", "PSNR": "↑", "LPIPS": "↓",
              "FID": "↓", "FVD": "↓", "Angular_Error_Mean": "↓"}


def metric_header(prefix=""):
    return " | ".join(f"{prefix}{m} {METRIC_DIR.get(m, '?')}" for m in METRICS)


def improved(metric, delta):
    """原始差值 → "improved"/"worse"/"same"/None(无法判定)。"""
    if not isinstance(delta, (int, float)):
        return None
    if delta == 0:
        return "same"          # 零差值是"没变",不是"变差"
    d = METRIC_DIR.get(metric)
    if d == "↑":
        return "improved" if delta > 0 else "worse"
    if d == "↓":
        return "improved" if delta < 0 else "worse"
    return None


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


# [v2.9.4] 与 run_manifest._NEVER_HEADLINE_ROLES 同一名单(改一处必须改三处:
# 这里 / run_manifest.py / common.sh 的 _RSWM_NEVER_HEADLINE_ROLES;
# test_pose_timeline.py 有用例钉住三者一致)。
DIAGNOSTIC_ROLES = frozenset({
    "diagnostic_stationary", "probe_gate", "variant_attribution",
    # [v3.1] 阶梯进【诊断节】而不是拒收节 —— 它的数字是有意义的(难度曲线),
    # 只是不进主表。放进 rejected 会读成"出了故障",那是错的语义。
    "ladder",
})


def read_run_manifest(eval_dir: Path):
    p = Path(eval_dir) / "run_manifest.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def classify(d, eval_dir=None):
    """→ (tier, reason)。tier ∈ {"headline", "diagnostic", "rejected"}。

    [v2.9.4] 两档不够用:上一版把 36(identity 诊断臂)和 34a/34b(判门)
    与"来路不明的旧产物"一起丢进「已拒收」,读起来像是出了故障 —— 而它们
    恰恰是设计上就该报告、只是不进 headline 的东西。三档才对得上语义。

    同时在这里**读 run_manifest.json**:v2.9.2 建了一整套 manifest 机制,
    汇总层却从不读它,于是 eval_results.json 的 overall 字段是唯一依据 ——
    而那正是最容易被 --metrics_only 之类操作重写的地方。两者不一致即拒收。
    """
    if not d:
        return "rejected", "产物为空或无法解析"
    o = d.get("overall", d)
    if "headline_eligible" not in o:
        return "rejected", ("产物没有 headline_eligible 字段(v2.9 之前的产物,"
                            "或非本管线产出)—— 无法证明其口径,拒收")
    m = read_run_manifest(eval_dir) if eval_dir else None
    if eval_dir is not None:
        if m is None:
            return "rejected", ("目录里没有 run_manifest.json —— eval_results 的"
                                "口径字段无从交叉核对,拒收")
        for k in ("rswm_profile", "pose_mode", "headline_eligible"):
            if k in o and m.get(k) != o.get(k):
                return "rejected", (f"run_manifest 与 eval_results 不一致:"
                                    f"{k} manifest={m.get(k)!r} results={o.get(k)!r}"
                                    f" —— 结果文件被重写过?")
        if m.get("manifest_override"):
            return "rejected", ("产物带 manifest_override 痕迹(用过 "
                                "RSWM_ALLOW_MANIFEST_MISMATCH),不得作为任何结论依据")
    role = (m or {}).get("role") or o.get("role")
    if o.get("headline_eligible") is True:
        return "headline", None
    why = o.get("headline_ineligible_reason") or "未记录理由"
    if role in DIAGNOSTIC_ROLES or o.get("pose_mode") == "identity":
        return "diagnostic", why
    return "rejected", why


def headline_status(d):
    """→ (eligible: bool, reason)。**fail-closed**:缺字段 = 不合格。

    [v2.9.2 P0-3] 旧版对缺 headline_eligible 的产物返回 None,而调用方只拒显式
    False —— 于是 v2.9 之前的、或任何被手工搬进来的产物,都会以"未知"的身份
    进主表。缺失恰恰是旧产物和被篡改产物的共同特征,把它当"未知所以放行"
    就等于没有门。现在:必须【显式为 True】才算合格。
    """
    if not d:
        return False, "产物为空或无法解析"
    o = d.get("overall", d)
    if "headline_eligible" not in o:
        return False, ("产物没有 headline_eligible 字段(v2.9 之前的产物,"
                       "或非本管线产出)—— 无法证明其口径,拒收")
    if o.get("headline_eligible") is not True:
        return False, (o.get("headline_ineligible_reason") or "未记录理由")
    return True, None


def overall_row(d):
    if not d:
        return None
    o = d.get("overall", d)
    return {m: o.get(m) for m in METRICS if o.get(m) is not None}


def fmt(v):
    return f"{v:.4g}" if isinstance(v, (int, float)) else "—"


def first_existing(*paths: Path) -> Path:
    """按优先级选择首个已有产物;都不存在时返回首选路径用于报缺。"""
    return next((path for path in paths if path.exists()), paths[0])


def main(work: Path):
    rows = []  # (组, 名称, 指标dict, 源文件)
    # [v2.7.3 ②/P1 修复] 主表/方向随 D3 拍板指向 r16 主线产物;旧 spec 把
    # eval_r128_sem 当主结果 —— p1→p3 新流程下 summary.md 不会报错,而是
    # 【静默把 r128 消融当主结果】、r16 几何显示缺失。r128 保留为容量消融行。
    spec = [
        # [v2.9 拍板] baseline 按 pose_mode 分三份;禁止用一份去减不同模式的臂。
        ("主表", "Baseline (drop / no-pose)",   work / "eval_base_drop/eval_results.json"),
        ("主表", "Baseline (native Plücker)",   work / "eval_base_native/eval_results.json"),
        ("主表", "Baseline (identity, diag)",   work / "eval_base_identity/eval_results.json"),
        ("主表", "Text/no-pose (drop, headline)", work / "eval_r16_sem_s42/eval_results.json"),
        ("主表", "Native Plücker (headline)",     work / "eval_r16_geo_s42/eval_results.json"),
        # [v2.7.5] corrected 30 号的 r8 是当前 Rank 探针;r32/r64 仅由
        # run_all/legacy 生产(在表头注释中显式标明,保持既有 summary.json 键)。
        ("Rank", "r8 (drop)",                  work / "eval_r8_sem_s42/eval_results.json"),
        ("Rank", "r32 (drop)",                 work / "eval_r32_sem/eval_results.json"),
        ("Rank", "r64 (drop)",                 work / "eval_r64_sem/eval_results.json"),
        # corrected 26 号优先;单机 run_all 的无 seed 后缀目录作回退。
        ("Rank", "r128 (drop)",
         first_existing(work / "eval_r128_sem_s42/eval_results.json",
                        work / "eval_r128_sem/eval_results.json")),
        ("方向", "Text/no-pose (drop, headline)",    work / "eval_r16_sem_s42/eval_results.json"),
        # [v2.7.4] corrected 链(23 号)产出 eval_r16_geo_s42,run_all 链产出
        # eval_r16_geo —— 两个候选按序取第一个存在的
        ("方向", "Native Plücker (headline)",
         first_existing(work / "eval_r16_geo_s42/eval_results.json",
                        work / "eval_r16_geo/eval_results.json")),
        # corrected 27 号优先;单机 run_all 的无 seed 后缀目录作回退。
        ("方向(消融)", "r128 Native Plücker",
         first_existing(work / "eval_r128_geo_s42/eval_results.json",
                        work / "eval_r128_geo/eval_results.json")),
        # [v2.8.1] 34/35/36 之前完全没有行 —— 判门与两个新臂跑完了也不进
        # summary,人要手翻目录才知道结果,等于白跑。
        ("判门", "34a Text-direction learnability (drop)",
         work / "probe34/eval_probe34a_drop/eval_results.json"),
        ("判门", "34b Plücker-direction learnability (native)",
         work / "probe34/eval_probe34b_native/eval_results.json"),
        ("方向", "Rich per-segment text (drop)",
         work / "eval_r16_richtext_s42/eval_results.json"),
        ("诊断", "Stationary-pose diagnostic (identity)",
         work / "eval_r16_nopose_s42/eval_results.json"),
    ]
    for lv in ["trivial", "easy", "medium", "hard"]:
        spec.append(("Overlap-LoRA", lv, work / f"overlap/{lv}/eval_lora/eval_results.json"))
        spec.append(("Overlap-Base", lv, work / f"overlap/{lv}/eval_baseline/eval_results.json"))

    # [v2.9.1 拍板] headline 资格由代码执行,不是打印警告。
    # 任何 headline_eligible=false 的产物一律【不进主表】,改列到"已拒收"节
    # 并写明理由 —— 用 legacy/identity 口径跑出来的数字混进主表,是最容易
    # 变成论文错误的那种污染,而它在表格里长得和合格数字一模一样。
    rejected_rows, diagnostic_rows = [], []
    for group, name, path in spec:
        raw = load(path)
        row = overall_row(raw)
        tier, reason = classify(raw, Path(path).parent if raw else None)
        if row is not None and tier == "rejected":
            rejected_rows.append((group, name, path, reason))
            continue
        if row is not None and tier == "diagnostic":
            diagnostic_rows.append((group, name, row, path, reason))
            continue
        rows.append((group, name, row, path))

    lines = ["# RSWM 从头复跑 — 结果总表\n",
             "> 注:Rank r32/r64 与 Overlap 行仅由 legacy/run_all 生产;"
             "corrected 20–33 链不生产这些目录。\n",
             "> ↑ = 越大越好,↓ = 越小越好;表中为**原始值**。",
             "",
             "| 组 | 配置 | " + metric_header() + " | 状态 |",
             "|---|---|" + "---|" * (len(METRICS) + 1)]
    out = {}
    for group, name, row, path in rows:
        if row:
            cells = " | ".join(fmt(row.get(m)) for m in METRICS)
            lines.append(f"| {group} | {name} | {cells} | ✓ |")
            out[f"{group}/{name}"] = row
        else:
            lines.append(f"| {group} | {name} | " + " | ".join(["—"] * len(METRICS)) + " | 缺 |")

    # ---- [v2.9.4] 诊断臂:设计上就不进 headline,但必须报告 ----
    if diagnostic_rows:
        lines.append("\n## 🔬 诊断臂(**不是** headline,按设计不入主表)\n")
        lines.append("> identity 静止相机臂、34a/34b 判门、shufpose/geostrip 归因臂都在这里。")
        lines.append("> 它们出现在本节是**正常**的,不代表出错。\n")
        lines.append("| 组 | 配置 | " + metric_header() + " | 为什么不入 headline |")
        lines.append("|---|---|" + "---|" * (len(METRICS) + 1))
        for g, n, r, pth, why in diagnostic_rows:
            lines.append(f"| {g} | {n} | " + " | ".join(fmt(r.get(m)) for m in METRICS)
                         + f" | {why} |")
            out[f"诊断/{n}"] = r
        out["_diagnostic_rows"] = [
            {"group": g, "name": n, "reason": w, "path": str(p)}
            for g, n, _, p, w in diagnostic_rows]

    # ---- [v2.9.1] 已拒收(无法证明来路)----
    if rejected_rows:
        lines.append("\n## ⛔ 已拒收:**无法证明来路**,任何结论都不得引用\n")
        lines.append("> 与上面的『诊断臂』不同 —— 这里的产物是口径不明/被重写过/缺 manifest,\n> 属于**故障**,不是设计。\n")
        lines.append("| 组 | 配置 | 拒收理由 | 路径 |")
        lines.append("|---|---|---|---|")
        for g, n, pth, why in rejected_rows:
            lines.append(f"| {g} | {n} | {why} | `{pth}` |")
        out["_rejected_non_headline"] = [
            {"group": g, "name": n, "reason": w, "path": str(pth)}
            for g, n, pth, w in rejected_rows]
        print(f"[collect] ⛔ 拒收 {len(rejected_rows)} 个非 headline 资格产物"
              f"(见 summary.md 的『已拒收』节)")

    # ---- [v2.9 拍板] 按 pose_mode 配对的 Δ ----
    # 每个 LoRA 臂只许减【同模式】的 baseline。共用一份 base 会把"位姿通路开/关"
    # 混进 Δ,而 headline(drop ↔ native)存在的全部意义就是单变量对照。
    ARM_MODE = {
        "eval_r16_sem_s42": "drop",   "eval_r16_sem_s43": "drop",
        "eval_r16_sem_s44": "drop",   "eval_r128_sem_s42": "drop",
        "eval_r8_sem_s42": "drop",    "eval_r16_sem_attncam": "drop",
        "eval_r16_richtext_s42": "drop",
        "eval_r16_geo_s42": "native", "eval_r16_geo_s43": "native",
        "eval_r16_geo_s44": "native", "eval_r128_geo_s42": "native",
        "eval_r16_geo_attncam": "native", "eval_r16_shufpose_s42": "native",
        "eval_r16_geostrip_s42": "native",
        "eval_r16_nopose_s42": "identity",
    }
    # [v2.9.2 P1] baseline 也必须自证身份 —— 只按目录名取用,等于允许把
    # eval_base_drop 里塞一份 native 的结果去减 drop 臂,而 Δ 表全绿看不出来。
    bases, base_bad = {}, {}
    for m in ("drop", "native", "identity"):
        _braw = load(work / f"eval_base_{m}/eval_results.json")
        _bo = ((_braw or {}).get("overall") or {})
        _brec = _bo.get("pose_mode")
        if _braw is None:
            bases[m] = None
            continue
        # [v2.9.4 #7] baseline 也必须过分档门。旧版只查 pose_mode 自报,于是
        # 主表刚拒收的 baseline(缺 manifest / 被重写 / legacy)在 Δ 表里照用,
        # 还给臂打 ✅ —— 而 baseline 是 Δ 的分母,污染的是每一个差值。
        _btier, _bwhy = classify(_braw, work / f"eval_base_{m}")
        if _btier == "rejected":
            base_bad[m] = f"eval_base_{m} 被拒收({_bwhy})—— 不得用作 Δ 的分母"
            bases[m] = None
            print(f"[collect] ⛔ {base_bad[m]}")
            continue
        if _brec is None:
            # [v2.9.4 #32] None 不再放行 —— 删掉 meta/ 就能让这道门闭嘴。
            base_bad[m] = (f"eval_base_{m} 没有自报 pose_mode(旧产物或 meta 被删)"
                           f" —— 无法证明口径,不得用作 Δ 的分母")
            bases[m] = None
            print(f"[collect] ⛔ {base_bad[m]}")
            continue
        if _brec != m:
            base_bad[m] = (f"eval_base_{m} 自报 pose_mode={_brec!r},与目录口径 "
                           f"{m} 不符 —— 该 baseline 不予采用")
            bases[m] = None
            print(f"[collect] ⛔ {base_bad[m]}")
            continue
        bases[m] = overall_row(_braw)
    if base_bad:
        out["_baseline_mode_conflicts"] = base_bad
    lines.append("\n## Δ(LoRA − 同模式 baseline)\n")
    lines.append("> Δ = LoRA − 同模式 baseline,**原始差值**。↑ 指标(SSIM/PSNR)正 Δ 为改善;")
    lines.append("> ↓ 指标(LPIPS/FID/FVD/角误差)**负 Δ 才是改善** —— 不要一律把正 Δ 读成变好。")
    lines.append("> 每格后缀 ✅=改善 / ❌=变差 / ➖=无变化,按各自方向判定。")
    lines.append("")
    lines.append("| 臂 | pose_mode | " + metric_header("Δ") + " | base 源 |")
    lines.append("|---|---|" + "---|" * (len(METRICS) + 1))
    delta_out = {}
    for arm, mode in sorted(ARM_MODE.items()):
        _araw = load(work / arm / "eval_results.json")
        _atier, _arw = classify(_araw, work / arm)
        if _atier == "diagnostic":
            _arow = overall_row(_araw)
            _brow = bases.get(mode)
            if _arow and _brow:
                _dd = {k: ((_arow.get(k) - _brow.get(k))
                           if isinstance(_arow.get(k), (int, float))
                           and isinstance(_brow.get(k), (int, float)) else None)
                       for k in METRICS}
                lines.append(f"| {arm} 🔬 | {mode} | "
                             + " | ".join(fmt(_dd[k]) for k in METRICS)
                             + f" | eval_base_{mode}(诊断臂,非 headline) |")
            continue
        if _atier == "rejected":
            lines.append(f"| {arm} | {mode} | " + " | ".join(["⛔"] * len(METRICS))
                         + f" | 拒收:{_arw} |")
            continue
        # [v2.9.1 C5] 用产物自报的 pose_mode 交叉校验硬编码映射表。
        # 否则 RSWM_PROFILE=legacy(SEM 臂=identity)时这张表会静默拿
        # eval_base_drop 去减 identity 臂,而表格里看不出任何异样。
        _rec = ((_araw or {}).get("overall") or {}).get("pose_mode")
        if _rec is None:
            lines.append(f"| {arm} | {mode} | " + " | ".join(["⚠️"] * len(METRICS))
                         + " | 产物没有自报 pose_mode,无法交叉核对映射表 |")
            out.setdefault("_mode_conflicts", []).append(
                {"arm": arm, "table": mode, "recorded": None})
            continue
        if _rec != mode:
            lines.append(f"| {arm} | {mode} | " + " | ".join(["⚠️"] * len(METRICS))
                         + f" | 口径冲突:产物自报 {_rec},映射表说 {mode} |")
            out.setdefault("_mode_conflicts", []).append(
                {"arm": arm, "table": mode, "recorded": _rec})
            print(f"[collect] ⚠️ {arm}: 产物 pose_mode={_rec} 与映射表 {mode} 冲突,"
                  f"该臂 Δ 不予计算")
            continue
        arow = overall_row(_araw)
        brow = bases.get(mode)
        if not arow or not brow:
            lines.append(f"| {arm} | {mode} | " + " | ".join(["—"] * len(METRICS))
                         + f" | {'缺臂' if not arow else base_bad.get(mode, f'缺 eval_base_{mode}')} |")
            continue
        d = {}
        for k in METRICS:
            a, b = arow.get(k), brow.get(k)
            d[k] = (a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
        def _cell(k):
                v = d[k]
                mark = {"improved": " ✅", "worse": " ❌",
                        "same": " ➖", None: ""}[improved(k, v)]
                return fmt(v) + mark
        lines.append(f"| {arm} | {mode} | " + " | ".join(_cell(k) for k in METRICS)
                     + f" | eval_base_{mode} |")
        delta_out[arm] = {"pose_mode": mode, "delta": d, "base": f"eval_base_{mode}"}
    out["_delta_by_mode"] = delta_out
    # 落盘时也把口径写清楚,防止有人回头拿别的 base 重算
    out["_pose_mode_policy"] = {
        "headline": {"text_only": "20-22 drop + GEO_H5",
                     "native": "23-25 native + GEO_H5"},
        "diagnostic": {"stationary_pose": "36 identity + SEM_H5"},
        "note": "identity ≠ text-only:中和位姿仍走激活的 Plücker 通路,"
                "等价于显式命令相机不动。禁止把 identity 臂称作 text-only。",
    }

    # 对齐对比(headline delta)
    cmp_p = work / "aligned_main/comparison.json"
    cmp = load(cmp_p)
    if cmp:
        lines.append("\n## 对齐对比(aligned_main/comparison.json)\n```json")
        lines.append(json.dumps(cmp, indent=2, ensure_ascii=False)[:3000])
        lines.append("```")

    # re-bin 汇总
    for tag in ["rebin_lora_quartile", "rebin_lora_two_level",
                "rebin_baseline_quartile", "rebin_baseline_two_level"]:
        d = work / "overlap" / tag
        if d.exists():
            lines.append(f"\n- overlap re-bin: `{d}` ✓")

    work.mkdir(parents=True, exist_ok=True)
    md = work / "summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    (work / "summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {md}")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "server_run/work").resolve())
