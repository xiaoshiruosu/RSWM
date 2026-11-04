#!/usr/bin/env python3
# =============================================================================
# generalization_gap_report.py — AID → 第二数据集的泛化间隙(论文表)
#
# 为什么需要它:
#   三个泛化 tier(RESISC45 / PatternNet / WHU-RS19)的建库与评估基础设施早就
#   完整(40_build_resisc45.pbs + qsub_resisc45.sh 都支持 --tier),但**没有任何
#   脚本把"AID 上的分数"与"换数据集之后的分数"放在一起算间隙**。于是即使跑完,
#   也只是散落的几个 eval_results.json,进不了论文。
#
#   审稿人对 benchmark 论文的标准问题是:"你只在一个数据源上做,凭什么叫遥感
#   benchmark?" 泛化间隙表是唯一能正面回答的东西。
#
# 设计要点:
#   1. **test-only**:泛化 tier 上不训练,只拿 AID 训好的 checkpoint 去评。
#      所以间隙 = 同一权重在两个测试集上的差,不掺训练差异。
#   2. **同 pose_mode 配对**:每个臂只与同口径的自己比,并且各自减各自的
#      同模式 baseline —— 否则间隙里混进了"位姿通路开/关"。
#   3. **过 headline 资格门**:与 collect_results 同源。泛化结果本身是正当的
#      论文数字(role=generalization 不在 never-headline 名单里),但来路不明的
#      产物一样不许进表。
#   4. **报可分辨性**:间隙小于该 N 下的最小可检测差异时,显式标注"不可分辨",
#      不让读者把抽样噪声读成泛化间隙。
#
# 用法:
#   python server_run/generalization_gap_report.py \
#       --source-work $WORK \
#       --tier-works resisc45=$W_R45 patternnet=$W_PN whurs19=$W_W19 \
#       --arms r16_sem_s42:drop r16_geo_s42:native \
#       -o $WORK/generalization_gap.json
# =============================================================================
import argparse
import json
import math
import os
import sys
from pathlib import Path

METRICS = ["SSIM", "PSNR", "LPIPS", "FID", "FVD", "Angular_Error_Mean"]
# ↑ 越大越好,↓ 越小越好。间隙的符号解读完全依赖它。
METRIC_DIR = {"SSIM": "↑", "PSNR": "↑", "LPIPS": "↓",
              "FID": "↓", "FVD": "↓", "Angular_Error_Mean": "↓"}


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _manifest_ok(eval_dir: Path):
    """与 collect_results.classify 同源的资格判定(简化版,只判能不能进表)。"""
    mp = eval_dir / "run_manifest.json"
    if not mp.is_file():
        return False, "缺 run_manifest.json —— 无法证明口径"
    m = _load(mp) or {}
    if m.get("manifest_override"):
        return False, "带 manifest_override 痕迹(用过 RSWM_ALLOW_MANIFEST_MISMATCH)"
    if m.get("headline_eligible") is not True:
        return False, (m.get("headline_ineligible_reason") or "不合资格")
    return True, None


def _overall(eval_dir: Path):
    d = _load(eval_dir / "eval_results.json")
    if not d:
        return None
    return d.get("overall", d)


def _mde(o, metric, n_key="num_samples"):
    """该指标在该 N 下的最小可检测差异。

    需要逐样本标准差;eval_results 的 overall 里只有 Angular_Error_Std 这类,
    没有的指标就返回 None —— **不臆造**。有 std 时用 1.96·√2·s/√n
    (与 metric_sample_size_study 同一口径)。
    """
    n = o.get(n_key)
    if not n or n < 2:
        return None
    # [自查修正] 只认真正的 _Std 键。旧写法的 metric.replace("_Mean","_Std")
    # 对不含 "_Mean" 的指标(SSIM/PSNR/LPIPS/FID/FVD)会原样返回指标名,
    # 于是把 **指标值本身** 当成标准差 —— 实测把 SSIM=0.586 当 std,
    # 算出假 MDE 并把真实的 -0.055 间隙误标成"不可分辨"。
    # 拿不到 std 就返回 None(不可判定),绝不臆造。
    cands = [f"{metric}_Std"]
    if metric.endswith("_Mean"):
        cands.append(metric[:-len("_Mean")] + "_Std")
    std = next((o[k] for k in cands if isinstance(o.get(k), (int, float))), None)
    if std is None:
        return None
    try:
        return 1.96 * math.sqrt(2.0) * float(std) / math.sqrt(float(n))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="AID → 第二数据集的泛化间隙报告")
    ap.add_argument("--source-work", required=True,
                    help="AID 主实验的 WORK 目录")
    ap.add_argument("--tier-works", nargs="+", required=True,
                    help="tier=work 形式,如 resisc45=/scratch/.../work_resisc45")
    ap.add_argument("--arms", nargs="+",
                    default=["r16_sem_s42:drop", "r16_geo_s42:native"],
                    help="arm:pose_mode,arm 是 eval_ 后面的名字")
    ap.add_argument("--strict", action="store_true",
                    help="任一产物不合资格即非零退出(用于自动化链)")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    src = Path(args.source_work)
    tiers = {}
    for spec in args.tier_works:
        if "=" not in spec:
            print(f"[FATAL] --tier-works 需要 tier=path 形式,得到 {spec!r}", file=sys.stderr)
            return 2
        k, v = spec.split("=", 1)
        tiers[k] = Path(v)

    report = {"source_work": str(src), "tiers": {k: str(v) for k, v in tiers.items()},
              "arms": {}, "rejected": []}
    lines = ["# 泛化间隙:AID(训练源)→ 第二数据集(test-only)", "",
             "> 同一权重、同一 pose_mode,只换测试集。间隙 = 目标 tier 分数 − AID 分数。",
             "> ↑ 指标掉分为负、↓ 指标掉分为正 —— 表里已按各自方向标注「变差/变好」。",
             "> **间隙小于该 N 的最小可检测差异(MDE)时标 ~,不得解读为真实泛化间隙。**", ""]

    any_bad = False
    for spec in args.arms:
        arm, _, pm = spec.partition(":")
        pm = pm or "native"
        src_dir = src / f"eval_{arm}"
        ok, why = _manifest_ok(src_dir) if src_dir.is_dir() else (False, "目录不存在")
        if not ok:
            report["rejected"].append({"where": str(src_dir), "reason": why})
            lines.append(f"## {arm}({pm})\n\n> ⛔ 源(AID)侧不可用:{why}\n")
            any_bad = True
            continue
        so = _overall(src_dir)
        if not so:
            report["rejected"].append({"where": str(src_dir), "reason": "无 eval_results.json"})
            any_bad = True
            continue
        # 口径必须对得上,否则间隙没有意义
        if so.get("pose_mode") not in (None, pm):
            report["rejected"].append(
                {"where": str(src_dir),
                 "reason": f"自报 pose_mode={so.get('pose_mode')!r} 与请求 {pm!r} 不符"})
            any_bad = True
            continue

        entry = {"pose_mode": pm, "source": {m: so.get(m) for m in METRICS}, "tiers": {}}
        lines.append(f"## {arm}(pose_mode={pm})\n")
        lines.append("| tier | " + " | ".join(f"{m} {METRIC_DIR[m]}" for m in METRICS)
                     + " | N |")
        lines.append("|---|" + "---|" * (len(METRICS) + 1))
        lines.append("| **AID(源)** | "
                     + " | ".join(f"{so.get(m):.4g}" if isinstance(so.get(m), (int, float))
                                  else "—" for m in METRICS)
                     + f" | {so.get('num_samples', '—')} |")

        for tname, twork in tiers.items():
            tdir = twork / f"eval_{arm}"
            tok, twhy = _manifest_ok(tdir) if tdir.is_dir() else (False, "目录不存在")
            if not tok:
                report["rejected"].append({"where": str(tdir), "reason": twhy})
                lines.append(f"| {tname} | " + " | ".join(["⛔"] * len(METRICS))
                             + f" | {twhy} |")
                any_bad = True
                continue
            to = _overall(tdir)
            if not to:
                lines.append(f"| {tname} | " + " | ".join(["—"] * len(METRICS)) + " | 缺结果 |")
                continue
            gaps, cells = {}, []
            for m in METRICS:
                a, b = so.get(m), to.get(m)
                if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
                    gaps[m] = None
                    cells.append("—")
                    continue
                g = b - a
                mde = _mde(to, m)
                if g == 0:
                    worse, marker = None, "持平"   # 零差不是"变好"
                else:
                    worse = (g < 0) if METRIC_DIR[m] == "↑" else (g > 0)
                    marker = "变差" if worse else "变好"
                unres = mde is not None and abs(g) < mde
                gaps[m] = {"gap": g, "target": b, "mde": mde, "worse": worse,
                           # mde 为 None = 该指标没有 _Std,**无法判定**可分辨性,
                           # 不是"可分辨"。表里用 ? 标出来。
                           "resolvable": (not unres) if mde is not None else None}
                _flag = "~" if unres else ("" if mde is not None else "?")
                cells.append(f"{b:.4g} ({g:+.3g} {marker}{_flag})")
            entry["tiers"][tname] = {"overall": {m: to.get(m) for m in METRICS}, "gaps": gaps}
            lines.append(f"| {tname} | " + " | ".join(cells)
                         + f" | {to.get('num_samples', '—')} |")
        lines.append("")
        report["arms"][arm] = entry

    if report["rejected"]:
        lines.append("## ⛔ 被拒收的产物\n")
        lines.append("| 路径 | 理由 |")
        lines.append("|---|---|")
        for r in report["rejected"]:
            lines.append(f"| `{r['where']}` | {r['reason']} |")
        lines.append("")

    lines.append("> `~` = 间隙小于该 N 的最小可检测差异,**不可分辨**;")
    lines.append("> `?` = 该指标没有逐样本标准差(FID/FVD 是集合级指标,本就没有),"
                 "**无法判定**可分辨性 —— 不等于可分辨。")
    md = "\n".join(lines)
    if args.output:
        p = Path(args.output)
        p.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        p.with_suffix(".md").write_text(md)
        print(f"[done] {p} + {p.with_suffix('.md')}")
    print(md)
    if args.strict and any_bad:
        print("[FATAL] --strict:有产物不合资格", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
