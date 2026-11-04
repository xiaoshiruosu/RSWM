#!/usr/bin/env python3
# =============================================================================
# aggregate_seeds.py — 多训练种子统计聚合 (advisor point 7)
#
# 读若干 eval 目录的 eval_results.json,按 config 聚合出 mean/std/median,并对
# --pair A B 做配对分析(逐种子差、bootstrap 95% CI、精确配对置换检验)。
# 输出 4 个聚合文件(inode 紧张,绝不散小文件): .csv / .json / .md / .tex
#
# 用法(与 server_run/pbs/qsub_corrected_rerun.sh:113 的调用完全一致):
#   python server_run/aggregate_seeds.py --seeds 42 43 44 \
#       --eval-dirs r16_sem=$WORK/eval_r16_sem_s{42,43,44} \
#                   r16_geo=$WORK/eval_r16_geo_s{42,43,44} \
#       --pair r16_sem r16_geo --out-prefix $WORK/seed_summary
#
#   注意上面的 s{42,43,44} 是 **shell** 花括号展开,argv 实际收到的是:
#       "r16_sem=<d42>" "<d43>" "<d44>" "r16_geo=<d42>" "<d43>" "<d44>"
#   即「不带 = 的 token 归属于前一个 group」。等价的显式写法(逗号列表)也支持:
#       --eval-dirs r16_sem=<d42>,<d43>,<d44>
#   便捷写法:
#       --eval-root $WORK --pattern 'eval_{config}_s{seed}' --configs r16_sem r16_geo
#
#   DRYRUN=1 python server_run/aggregate_seeds.py ...   # 只打印将要读的文件,exit 0
#
# =============================================================================
# 【两种不确定度,必须都报,永远不能互相替代】
#
#   (1) TRAINING-SEED variance —— 跨 --seeds {42,43,44} 重训 LoRA 的方差。
#       回答:「换个种子重训,这个结论还在吗?」度量:mean ± std(ddof=1, n=种子数)
#       + 对种子重采样的 bootstrap CI。本脚本的主表就是这个。
#
#   (2) TEST-TRAJECTORY bootstrap —— 固定一个已训练好的 checkpoint,对**测试集
#       轨迹**重采样。回答:「换一批测试轨迹,这个数还稳吗?」度量:对 per_sample
#       重采样的 bootstrap CI(见 --per-traj)。
#
#   二者量纲相同、含义完全不同:(2) 窄 **不能** 说明 (1) 窄。一个 checkpoint 的
#   test-trajectory CI 再漂亮,也完全不能证明换种子重训不会翻盘;反之亦然。论文里
#   报 (1) 却把 (2) 的 CI 贴上去(或反过来)是审稿人一眼能抓的错误。本脚本把两者
#   放在**不同的表**里,并各自标注,就是为了防止串台。
# =============================================================================
import argparse
import itertools
import json
import os
import re
import sys
from pathlib import Path

# numpy 只在真正算统计量时才 import:这样 --help 和 DRYRUN=1 在没激活 rswm 环境的
# 登录节点上也能跑(编排器的 DRYRUN 预检就在登录节点执行)。服务器上行为无差异。
_np = None


# =============================================================================
# [v2.9.4] headline 资格门 —— 本脚本产出的是【会进论文】的数字。
# collect_results 有三档分类,但这些独立出口此前完全没有门:一个 legacy/
# identity/来路不明的 eval 目录照样能被聚合进多种子均值或 FVD 重算表。
# 规则与 collect_results.classify 同源:必须有 run_manifest.json、其
# headline_eligible 必须显式为 true、且不得带 manifest_override 痕迹。
# 逃生口 RSWM_SKIP_HEADLINE_GATE=1(产物不得作为 headline 数字)。
# =============================================================================
def _assert_headline_eligible(eval_dir, *, what="该目录"):
    import json as _j, os as _o
    from pathlib import Path as _P
    if _o.environ.get("RSWM_SKIP_HEADLINE_GATE", "0").strip().lower() not in (
            "0", "false", "no", "off", ""):
        print(f"  ⚠️ RSWM_SKIP_HEADLINE_GATE=1,跳过 {what} 的资格校验")
        return
    mp = _P(eval_dir) / "run_manifest.json"
    if not mp.is_file():
        raise SystemExit(f"[headline] {what} {eval_dir} 没有 run_manifest.json —— "
                         f"无法证明口径,拒绝把它的数字聚合进论文表。")
    m = _j.loads(mp.read_text())
    if m.get("manifest_override"):
        raise SystemExit(f"[headline] {what} {eval_dir} 带 manifest_override 痕迹"
                         f"(用过 RSWM_ALLOW_MANIFEST_MISMATCH),不得引用。")
    if m.get("headline_eligible") is not True:
        raise SystemExit(
            f"[headline] {what} {eval_dir} 不合 headline 资格:"
            f"{m.get('headline_ineligible_reason') or '未记录理由'} —— "
            f"诊断臂/判门/legacy 的数字不得进论文主表。")


def _numpy():
    global _np
    if _np is None:
        import numpy as np  # noqa: E402
        _np = np
    return _np


def _fatal(msg: str) -> "None":
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# 指标口径 —— 键名来自 lora_remotesense/new/metrics.py:355-382 (compute_all_metrics)
# 与 evaluate.py:331 (num_samples)。evaluate.py:400 用 json.dump(..., default=str)
# 落盘,所以 numpy 标量可能被写成字符串,取值时要做一次宽松 coerce。
# ---------------------------------------------------------------------------
# [v2.6] Angular_Error_WellPosed_Mean = Track A well-posed 双栏(GT 净位移比
# >= 0.2,与 Track B 门控同口径)的正式角误差读数。v2.6 之前的 eval 目录没有
# 该键 —— 会按既有约定 fail-loud;聚合旧目录请显式 --metrics 去掉它。
DEFAULT_METRICS = ["SSIM", "PSNR", "LPIPS", "FID", "FVD",
                   "Angular_Error_Mean", "Angular_Error_Std",
                   "Angular_Error_WellPosed_Mean"]

# True = 越大越好。用于 md/tex 表头的箭头。
HIGHER_IS_BETTER = {
    "SSIM": True, "PSNR": True,
    "LPIPS": False, "FID": False, "FVD": False,
    "Angular_Error_Mean": False, "Angular_Error_Median": False,
    "Angular_Error_Std": False,
    "Angular_Error_WellPosed_Mean": False,
    "Angular_Error_WellPosed_Median": False,
    "Angular_Error_WellPosed_Std": False,
}


def _arrow(metric: str) -> str:
    hib = HIGHER_IS_BETTER.get(metric)
    return "" if hib is None else (" ↑" if hib else " ↓")


def _tex_arrow(metric: str) -> str:
    hib = HIGHER_IS_BETTER.get(metric)
    return "" if hib is None else (r" $\uparrow$" if hib else r" $\downarrow$")


def _tex_escape(s: str) -> str:
    return str(s).replace("\\", r"\textbackslash{}").replace("_", r"\_") \
                 .replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


# ---------------------------------------------------------------------------
# argv 解析
# ---------------------------------------------------------------------------
def parse_eval_dirs(tokens):
    """把 --eval-dirs 的 argv token 解析成 [(name, [dir, ...]), ...](保序)。

    规则:
      * "name=dir"            -> 开一个新 group(name 里不含 '/' 才算 group 头,
                                 否则可能是路径里恰好有 '=' 的续行 token)
      * "name=d1,d2,d3"       -> 新 group + 逗号列表
      * "dir" / "d1,d2"       -> **续上一个 group**(shell 花括号展开的形态)
    """
    groups = []          # [(name, [dirs])]
    index = {}           # name -> position in groups
    for tok in tokens:
        if not tok:
            continue
        head, sep, tail = tok.partition("=")
        is_group_head = bool(sep) and ("/" not in head) and head.strip() != ""
        if is_group_head:
            name = head.strip()
            dirs = [d for d in tail.split(",") if d]
            if name in index:
                groups[index[name]][1].extend(dirs)
            else:
                index[name] = len(groups)
                groups.append((name, list(dirs)))
        else:
            if not groups:
                _fatal(f"--eval-dirs 的第一个 token 必须是 name=dir 形式,收到: {tok!r}")
            groups[-1][1].extend([d for d in tok.split(",") if d])
    return groups


def build_from_root(root, pattern, configs, seeds):
    out = []
    for c in configs:
        dirs = [str(Path(root) / pattern.format(config=c, seed=s)) for s in seeds]
        out.append((c, dirs))
    return out


_SEED_RE = re.compile(r"_s(\d+)(?:/|$)")


def check_seed_order(name, dirs, seeds):
    """目录名里带 _s<seed> 时,校验它与 --seeds 的顺序一致。

    防的是真实脚手架故障:花括号展开顺序被改过 / 手写 --eval-dirs 顺序写反,
    导致 seed 42 的数被当成 seed 43。全部可解析且不一致 -> fail loud。
    """
    parsed = []
    for d in dirs:
        m = _SEED_RE.search(str(d).rstrip("/") + "/")
        if not m:
            return  # 命名不遵循 _s<seed> 约定:不做检查(不猜)
        parsed.append(int(m.group(1)))
    if parsed != list(seeds):
        _fatal(
            f"config {name!r} 的目录名解析出的种子顺序 {parsed} 与 --seeds {list(seeds)} 不一致。\n"
            f"        目录: {dirs}\n"
            f"        这会把某个种子的数算到另一个种子头上。请修正 --eval-dirs 顺序 "
            f"(或把目录改成不含 _s<seed> 的命名以跳过本检查)。"
        )


# ---------------------------------------------------------------------------
# 读取 eval_results.json
# ---------------------------------------------------------------------------
def _coerce_num(v):
    """evaluate.py 用 default=str 落盘,numpy 标量可能变成字符串。"""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def load_eval_results(eval_dir, scenario, metrics, allow_missing_metric):
    """读一个 eval 目录的 eval_results.json -> {metric: float}。缺文件/缺键 fail loud。"""
    _assert_headline_eligible(eval_dir, what="被聚合的种子目录")
    p = Path(eval_dir) / "eval_results.json"
    if not p.exists():
        _fatal(
            f"缺少 {p} —— 该种子的评估结果不存在。\n"
            f"        绝不在更少的种子上做平均:请先补跑这个种子的 eval,或把它从 --seeds 里去掉"
            f"(并在论文里如实说明 n 变小了)。"
        )
    try:
        doc = json.loads(p.read_text())
    except Exception as e:
        _fatal(f"{p} 读取/解析失败: {type(e).__name__}: {e}")

    if not isinstance(doc, dict) or not doc:
        _fatal(f"{p} 不是非空 JSON object —— evaluate.py:400 应写出 {{'overall': {{...}}, ...}}。")
    if scenario not in doc:
        _fatal(
            f"{p} 里没有场景块 {scenario!r}。可用的块: {sorted(doc.keys())}\n"
            f"        (evaluate.py:326/346/365/385 会写 'overall' + 场景 + dir_<D> + dir_<D>_<场景>)"
        )
    block = doc[scenario]
    if not isinstance(block, dict):
        _fatal(f"{p} 的 {scenario!r} 块不是 object,而是 {type(block).__name__}。")

    row, warnings = {}, []
    for m in metrics:
        if m not in block:
            msg = (f"{p} 的 {scenario!r} 块里没有指标键 {m!r}。该块实际有: {sorted(block.keys())}\n"
                   f"        (metrics.py:357/378 里 LPIPS 与 Angular_* 是**条件**写入的:LPIPS 全部"
                   f"算失败、或没有 target_angles 时不会出现)")
            if allow_missing_metric:
                print(f"[WARN] {msg}\n       --allow-missing-metric 生效:该指标记为缺失。",
                      file=sys.stderr)
                row[m] = None
                warnings.append(f"missing_metric:{m}@{p}")
                continue
            _fatal(msg + f"\n        用 --metrics 缩小指标集,或 --allow-missing-metric 降级为警告。")
        v = _coerce_num(block[m])
        if v is None:
            _fatal(f"{p} 的 {scenario!r}.{m} 不是数值,而是 {block[m]!r}。")
        if v != v or v in (float("inf"), float("-inf")):
            # metrics.py:367/374 在 FID/FVD 算失败时会写 nan。这是「算失败」的真实记录,
            # 不是可以静默吃掉的东西。
            print(f"[WARN] {p} 的 {scenario!r}.{m} = {v} (非有限值)。"
                  f"该指标的聚合结果会是 nan,且不会被静默丢弃。", file=sys.stderr)
            warnings.append(f"nonfinite:{scenario}.{m}={v}@{p}")
        row[m] = v
    n = _coerce_num(block.get("num_samples"))
    return row, (int(n) if n is not None else None), warnings


def load_per_sample(eval_dir):
    """可选的 per-trajectory 数据。

    两个生产者都落 <eval_dir>/scores.json,格式一致(键名大小写不敏感 —— 下面
    按"所有数值键"自动发现,不写死键名):
      - server_run/baselines/run_baseline.py:252  -> {"per_sample":[{"sample_id","ssim",...}]}
      - lora_remotesense/new/evaluate.py(overall 处)-> {"per_sample":[{"sample_id","SSIM",
        "PSNR","LPIPS","Angular_Error","scene","direction"}]}(非数值键会被自动忽略;
        Angular_Error=null 表示该样本无方向词或光流不可靠,自动排除出该指标)
    ⚠️ 在 evaluate.py 加 scores.json 落盘【之前】跑出来的旧 eval 目录没有这个文件 ->
    test-trajectory bootstrap 对该目录不可用,如实报告不可用(不猜、不静默跳过)。
    """
    p = Path(eval_dir) / "scores.json"
    if not p.exists():
        return None, None
    try:
        doc = json.loads(p.read_text())
    except Exception as e:
        _fatal(f"{p} 存在但解析失败: {type(e).__name__}: {e}")
    rows = doc.get("per_sample")
    if not isinstance(rows, list) or not rows:
        return None, str(p)
    return rows, str(p)


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def summarize(values):
    """values: [float|None] -> mean/std(ddof=1)/median/n。含 None/nan 时不静默丢弃。"""
    np = _numpy()
    if any(v is None for v in values):
        return {"mean": None, "std": None, "median": None, "n": 0, "note": "metric missing"}
    a = np.asarray(values, dtype=float)
    n = int(a.size)
    return {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if n >= 2 else None,  # n=1 时样本标准差无定义
        "median": float(np.median(a)),
        "n": n,
    }


def bootstrap_ci_mean(values, n_boot, rng_seed, alpha=0.05):
    """对 values 重采样(有放回)求 mean 的 percentile bootstrap CI。"""
    np = _numpy()
    a = np.asarray([v for v in values], dtype=float)
    if a.size == 0 or not np.all(np.isfinite(a)):
        return {"lo": None, "hi": None, "n_boot": 0,
                "note": "非有限值或空样本,bootstrap 跳过"}
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    means = a[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"lo": float(lo), "hi": float(hi), "n_boot": int(n_boot)}


def exact_paired_permutation(diffs):
    """精确配对置换检验:枚举全部 2^n 个符号翻转,双侧 p。

    n=3 -> 8 种翻转,观测本身与其整体取负必然 >= |观测均值|,故 **p 的下界 = 2/8 = 0.25**。
    也就是说 n=3 时无论效应多大,这个检验永远给不出 p<0.25。它能提供的信息极其有限,
    报出来只是为了诚实,不是为了声称显著性。
    """
    np = _numpy()
    a = np.asarray(diffs, dtype=float)
    n = int(a.size)
    if n == 0 or not np.all(np.isfinite(a)):
        return {"p_two_sided": None, "n_perm": 0, "min_possible_p": None,
                "note": "非有限值或空样本,置换检验跳过"}
    obs = abs(float(np.mean(a)))
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
    perm_means = np.abs((signs * a).mean(axis=1))
    # >= 用了容差:浮点下「与观测等大」的翻转必须被计入,否则 p 会被低估。
    p = float(np.mean(perm_means >= obs - 1e-12))
    return {"p_two_sided": p, "n_perm": int(2 ** n), "min_possible_p": 2.0 / (2 ** n)}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="多训练种子统计聚合(缺种子 fail-loud)。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--seeds", nargs="+", type=int, required=True,
                    help="训练种子,例如 --seeds 42 43 44")
    ap.add_argument("--eval-dirs", nargs="*", default=[],
                    help="name=d1,d2,d3;不带 '=' 的 token 续上一个 name(shell 花括号展开形态)")
    ap.add_argument("--eval-root", default=None, help="便捷写法:配合 --pattern/--configs")
    ap.add_argument("--pattern", default="eval_{config}_s{seed}",
                    help="--eval-root 下的目录名模板,可用 {config} 与 {seed}")
    ap.add_argument("--configs", nargs="*", default=[], help="配合 --eval-root 使用的 config 名")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), default=None,
                    help="配对分析 A-B,可重复")
    ap.add_argument("--out-prefix", required=True, help="输出 <prefix>.csv/.json/.md/.tex")
    ap.add_argument("--scenario", default="overall",
                    help="eval_results.json 里的场景块 (默认 overall)")
    ap.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap 重采样次数")
    ap.add_argument("--rng-seed", type=int, default=0, help="bootstrap 的 RNG 种子(可复现)")
    ap.add_argument("--no-permutation", action="store_true", help="跳过精确配对置换检验")
    ap.add_argument("--per-traj", action="store_true",
                    help="若 eval 目录下有 scores.json(per_sample),额外报 test-trajectory bootstrap")
    ap.add_argument("--allow-missing-metric", action="store_true",
                    help="缺指标键时降级为警告(默认 fail loud)")
    args = ap.parse_args()

    seeds = args.seeds
    if len(set(seeds)) != len(seeds):
        _fatal(f"--seeds 有重复: {seeds}")

    # ---- 组装 config -> dirs ----
    groups = parse_eval_dirs(args.eval_dirs)
    if args.eval_root:
        if not args.configs:
            _fatal("--eval-root 需要配合 --configs 使用。")
        groups += build_from_root(args.eval_root, args.pattern, args.configs, seeds)
    if not groups:
        _fatal("没有任何 eval 目录:请给 --eval-dirs 或 --eval-root/--configs。")

    for name, dirs in groups:
        if len(dirs) != len(seeds):
            _fatal(
                f"config {name!r} 给了 {len(dirs)} 个目录,但 --seeds 有 {len(seeds)} 个"
                f"({seeds})。\n        目录: {dirs}\n"
                f"        绝不在更少的种子上做平均 —— 请补齐目录或改 --seeds。"
            )
        check_seed_order(name, dirs, seeds)

    dryrun = os.environ.get("DRYRUN", "0") == "1"
    print("=" * 72)
    print(f"aggregate_seeds.py | seeds={seeds} | scenario={args.scenario!r}")
    print(f"metrics={args.metrics}")
    print("=" * 72)
    for name, dirs in groups:
        print(f"[config] {name}")
        for s, d in zip(seeds, dirs):
            print(f"    seed {s}: {Path(d) / 'eval_results.json'}")
            if args.per_traj:
                print(f"             (per-traj 可选) {Path(d) / 'scores.json'}")
    pairs = args.pair or []
    for a, b in pairs:
        print(f"[pair] {a} - {b}")
    for ext in ("csv", "json", "md", "tex"):
        print(f"[out] {args.out_prefix}.{ext}")

    if dryrun:
        print("\n[DRYRUN=1] 以上为将要读/写的文件,未做任何 IO。exit 0")
        return 0

    names = [n for n, _ in groups]
    for a, b in pairs:
        for x in (a, b):
            if x not in names:
                _fatal(f"--pair 里的 {x!r} 不在 --eval-dirs 的 config 列表 {names} 中。")

    # ---- 读数(缺文件/缺键在这里 fail loud)----
    per_seed, num_samples, warnings = {}, {}, []
    for name, dirs in groups:
        per_seed[name], num_samples[name] = {}, {}
        for s, d in zip(seeds, dirs):
            row, n, w = load_eval_results(d, args.scenario, args.metrics,
                                          args.allow_missing_metric)
            per_seed[name][s] = row
            num_samples[name][s] = n
            warnings.extend(w)

    # ---- 每 config 的 training-seed 统计 ----
    stats = {}
    for name in names:
        stats[name] = {}
        for m in args.metrics:
            vals = [per_seed[name][s][m] for s in seeds]
            st = summarize(vals)
            st["per_seed"] = {str(s): per_seed[name][s][m] for s in seeds}
            if st["mean"] is not None:
                st["boot_ci95"] = bootstrap_ci_mean(vals, args.bootstrap, args.rng_seed)
            stats[name][m] = st

    # ---- 配对分析 ----
    pair_out = {}
    for a, b in pairs:
        key = f"{a}-{b}"
        pair_out[key] = {"A": a, "B": b, "n_seeds": len(seeds), "metrics": {}}
        for m in args.metrics:
            va = [per_seed[a][s][m] for s in seeds]
            vb = [per_seed[b][s][m] for s in seeds]
            if any(v is None for v in va + vb):
                pair_out[key]["metrics"][m] = {"note": "metric missing in one side"}
                continue
            diffs = [x - y for x, y in zip(va, vb)]
            entry = {
                "per_seed_diff": {str(s): d for s, d in zip(seeds, diffs)},
                "mean_diff": summarize(diffs)["mean"],
                "boot_ci95_mean_diff": bootstrap_ci_mean(diffs, args.bootstrap, args.rng_seed),
            }
            if not args.no_permutation:
                entry["exact_paired_permutation"] = exact_paired_permutation(diffs)
            pair_out[key]["metrics"][m] = entry

    # ---- 可选:test-trajectory bootstrap(与 training-seed 方差是两回事)----
    per_traj = {}
    if args.per_traj:
        for name, dirs in groups:
            per_traj[name] = {}
            for s, d in zip(seeds, dirs):
                rows, path = load_per_sample(d)
                if rows is None:
                    per_traj[name][str(s)] = {
                        "available": False,
                        "reason": (f"{Path(d) / 'scores.json'} 不存在或没有非空 per_sample。"
                                   f"evaluate.py 不产出 per-sample 指标;只有 "
                                   f"server_run/baselines/run_baseline.py:252 会写这个格式。"),
                    }
                    continue
                ent = {"available": True, "source": path, "n_trajectories": len(rows),
                       "metrics": {}}
                lower = {m.lower(): m for m in args.metrics}
                keys = set()
                for r in rows:
                    keys.update(k for k, v in r.items()
                                if k != "sample_id" and isinstance(v, (int, float))
                                and not isinstance(v, bool))
                for k in sorted(keys):
                    vals = [r[k] for r in rows
                            if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)]
                    if len(vals) < 2:
                        continue
                    ci = bootstrap_ci_mean(vals, args.bootstrap, args.rng_seed)
                    ent["metrics"][lower.get(k, k)] = {
                        "mean": summarize(vals)["mean"], "n": len(vals), "boot_ci95": ci,
                    }
                per_traj[name][str(s)] = ent

    # ---- 落盘 4 个文件 ----
    caveats = build_caveats(seeds, pairs, args)
    doc = {
        "scenario": args.scenario,
        "seeds": seeds,
        "metrics": args.metrics,
        "configs": {n: {"dirs": d} for n, d in groups},
        "num_samples": {n: {str(s): num_samples[n][s] for s in seeds} for n in names},
        "training_seed_stats": stats,
        "paired": pair_out,
        "test_trajectory_bootstrap": per_traj if args.per_traj else
            {"note": "未启用 --per-traj。test-trajectory bootstrap 与 training-seed "
                     "方差是两种不同的不确定度,不可互相替代。"},
        "bootstrap": {"n_boot": args.bootstrap, "rng_seed": args.rng_seed,
                      "method": "percentile", "resampled_unit": "training seeds"},
        "warnings": warnings,
        "caveats": caveats,
    }
    # 用 str 拼接而非 with_suffix:--out-prefix 末段若含 '.'(如 seed_summary.v1),
    # with_suffix 会把 '.v1' 替换掉,静默写到别的文件名上。
    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _out = lambda ext: Path(str(prefix) + "." + ext)  # noqa: E731
    _out("json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    _out("csv").write_text(render_csv(names, seeds, args.metrics, stats, pair_out))
    _out("md").write_text(
        render_md(names, seeds, args.metrics, stats, pair_out, per_traj, args, caveats, warnings))
    _out("tex").write_text(render_tex(names, seeds, args.metrics, stats, args))

    print()
    for ext in ("csv", "json", "md", "tex"):
        print(f"[done] -> {_out(ext)}")
    if warnings:
        print(f"[WARN] {len(warnings)} 条告警,详见 JSON 的 warnings 字段。", file=sys.stderr)
    for c in caveats:
        print(f"[caveat] {c}")
    return 0


def build_caveats(seeds, pairs, args):
    n = len(seeds)
    c = [
        f"训练种子数 n={n}。这是一个**极小样本**:mean±std 与 bootstrap CI 都只能当作"
        f"离散度的粗略指示,不是稳健的区间估计。n={n} 的 bootstrap 至多只能重采样出 "
        f"{n}^{n}={n ** n} 种组合,CI 的端点由那几个点直接决定。",
        "TRAINING-SEED variance(本表)与 TEST-TRAJECTORY bootstrap(--per-traj)是两种"
        "不同的不确定度,必须分别报告,不能互相替代。",
    ]
    if pairs and not args.no_permutation:
        c.append(
            f"精确配对置换检验:n={n} -> 只有 2^{n}={2 ** n} 种符号翻转,双侧 p 的**下界**是 "
            f"2/{2 ** n}={2.0 / (2 ** n):.4g}。因此 n={n} 时 p 永远不可能小于 "
            f"{2.0 / (2 ** n):.4g},无论效应多大。不要把它读成「不显著」,它只是**没有分辨率**。"
        )
    return c


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def _f(v, nd=4):
    if v is None:
        return ""
    if v != v:
        return "nan"
    return f"{v:.{nd}f}" if abs(v) < 1000 else f"{v:.1f}"


def render_csv(names, seeds, metrics, stats, pair_out):
    """单个 CSV(inode 紧张:不散文件)。长表:每行 = 一个 config/pair 的一个指标。"""
    hdr = ["kind", "config", "metric"] + [f"seed_{s}" for s in seeds] + \
          ["mean", "std_ddof1", "median", "n_seeds", "boot_ci95_lo", "boot_ci95_hi",
           "perm_p_two_sided"]
    lines = [",".join(hdr)]
    for name in names:
        for m in metrics:
            st = stats[name][m]
            ci = st.get("boot_ci95") or {}
            row = ["config", name, m] + [_f(st["per_seed"][str(s)]) for s in seeds] + [
                _f(st["mean"]), _f(st["std"]), _f(st["median"]), str(st["n"]),
                _f(ci.get("lo")), _f(ci.get("hi")), ""]
            lines.append(",".join(row))
    for key, pd in pair_out.items():
        for m, e in pd["metrics"].items():
            if "mean_diff" not in e:
                continue
            ci = e.get("boot_ci95_mean_diff") or {}
            perm = e.get("exact_paired_permutation") or {}
            row = ["pair_diff", key, m] + \
                  [_f(e["per_seed_diff"][str(s)]) for s in seeds] + [
                      _f(e["mean_diff"]), "", "", str(pd["n_seeds"]),
                      _f(ci.get("lo")), _f(ci.get("hi")), _f(perm.get("p_two_sided"))]
            lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def render_md(names, seeds, metrics, stats, pair_out, per_traj, args, caveats, warnings):
    L = ["# 多种子聚合 (scenario=`%s`)" % args.scenario, "",
         f"训练种子: `{seeds}` (n={len(seeds)}) | bootstrap: {args.bootstrap} 次 "
         f"(percentile, 重采样单位 = **训练种子**, rng_seed={args.rng_seed})", ""]

    L += ["## ⚠️ 读表须知", ""]
    for c in caveats:
        L.append(f"- {c}")
    L.append("")

    L += ["## 1. 每种子原始值", "",
          "| config | metric | " + " | ".join(f"seed {s}" for s in seeds) + " |",
          "|---|---|" + "---|" * len(seeds)]
    for name in names:
        for m in metrics:
            st = stats[name][m]
            L.append(f"| {name} | {m}{_arrow(m)} | " +
                     " | ".join(_f(st["per_seed"][str(s)]) or "—" for s in seeds) + " |")
    L.append("")

    L += ["## 2. 每 config 汇总 (跨训练种子)", "",
          "std = 样本标准差 (ddof=1);CI = 对**训练种子**重采样的 95% percentile bootstrap。", "",
          "| config | metric | mean | std | median | 95% CI (seed-bootstrap) | n |",
          "|---|---|---|---|---|---|---|"]
    for name in names:
        for m in metrics:
            st = stats[name][m]
            ci = st.get("boot_ci95") or {}
            cis = (f"[{_f(ci.get('lo'))}, {_f(ci.get('hi'))}]"
                   if ci.get("lo") is not None else "—")
            L.append(f"| {name} | {m}{_arrow(m)} | {_f(st['mean']) or '—'} | "
                     f"{_f(st['std']) or '—'} | {_f(st['median']) or '—'} | {cis} | {st['n']} |")
    L.append("")

    if pair_out:
        L += ["## 3. 配对分析", ""]
        for key, pd in pair_out.items():
            L += [f"### {key}  (逐种子配对差 A−B, A=`{pd['A']}`, B=`{pd['B']}`)", "",
                  "| metric | " + " | ".join(f"Δ seed {s}" for s in seeds) +
                  " | mean Δ | 95% CI (seed-bootstrap) | exact perm p |",
                  "|---|" + "---|" * (len(seeds) + 3)]
            for m in metrics:
                e = pd["metrics"].get(m, {})
                if "mean_diff" not in e:
                    L.append(f"| {m} | " + " | ".join("—" for _ in seeds) + " | — | — | — |")
                    continue
                ci = e.get("boot_ci95_mean_diff") or {}
                perm = e.get("exact_paired_permutation") or {}
                cis = (f"[{_f(ci.get('lo'))}, {_f(ci.get('hi'))}]"
                       if ci.get("lo") is not None else "—")
                pv = _f(perm.get("p_two_sided")) if perm.get("p_two_sided") is not None else "—"
                L.append(f"| {m}{_arrow(m)} | " +
                         " | ".join(_f(e["per_seed_diff"][str(s)]) for s in seeds) +
                         f" | {_f(e['mean_diff'])} | {cis} | {pv} |")
            L.append("")

    L += ["## 4. Test-trajectory bootstrap", ""]
    if not args.per_traj:
        L += ["未启用 `--per-traj`。", "",
              "**这与上面的 training-seed 方差不是一回事**:上面回答「换种子重训还成立吗」,"
              "这里回答「换一批测试轨迹还稳吗」。两者都要报,谁也替代不了谁。", ""]
    else:
        any_avail = any(e.get("available") for cfg in per_traj.values() for e in cfg.values())
        if not any_avail:
            L += ["请求了 `--per-traj`,但没有任何 eval 目录含非空 `scores.json/per_sample`。", "",
                  "`lora_remotesense/new/evaluate.py` 只落盘聚合后的 `eval_results.json`,"
                  "**不产出** per-sample 指标;当前仓里只有 "
                  "`server_run/baselines/run_baseline.py:252` 会写该格式。"
                  "因此 test-trajectory bootstrap 现在不可用 —— 这是事实陈述,不是失败。", ""]
        else:
            L += ["重采样单位 = **测试轨迹**(不是训练种子)。", "",
                  "| config | seed | metric | mean | n_traj | 95% CI (traj-bootstrap) |",
                  "|---|---|---|---|---|---|"]
            for name in per_traj:
                for s, e in per_traj[name].items():
                    if not e.get("available"):
                        continue
                    for m, v in sorted(e["metrics"].items()):
                        ci = v["boot_ci95"]
                        L.append(f"| {name} | {s} | {m} | {_f(v['mean'])} | {v['n']} | "
                                 f"[{_f(ci.get('lo'))}, {_f(ci.get('hi'))}] |")
            L.append("")

    if warnings:
        L += ["## 告警", ""] + [f"- `{w}`" for w in warnings] + [""]
    return "\n".join(L)


def render_tex(names, seeds, metrics, stats, args):
    """booktabs 表, mean$\\pm$std, 可直接粘进论文。"""
    cols = "l" + "c" * len(metrics)
    L = [f"% 由 server_run/aggregate_seeds.py 生成 | scenario={args.scenario} | seeds={seeds}",
         f"% 单元格 = mean $\\pm$ std(ddof=1),跨 n={len(seeds)} 个**训练种子**。",
         f"% n={len(seeds)} 极小:std 只是离散度的粗略指示,不是稳健区间估计。",
         "% 这是 training-seed 方差,不是 test-trajectory bootstrap CI —— 勿混用。",
         r"\begin{table}[t]", r"  \centering",
         f"  \\caption{{跨 {len(seeds)} 个训练种子(seeds {', '.join(str(s) for s in seeds)})的"
         f"结果,mean $\\pm$ std。std 为训练种子间的样本标准差(ddof=1),"
         f"n={len(seeds)} 极小,仅作离散度指示。}}",
         r"  \label{tab:seed_aggregate}",
         f"  \\begin{{tabular}}{{{cols}}}", r"    \toprule",
         "    Config & " + " & ".join(_tex_escape(m) + _tex_arrow(m) for m in metrics) + r" \\",
         r"    \midrule"]
    for name in names:
        cells = []
        for m in metrics:
            st = stats[name][m]
            if st["mean"] is None:
                cells.append("--")
            elif st["std"] is None:
                cells.append(f"${_f(st['mean'])}$")
            else:
                cells.append(f"${_f(st['mean'])} \\pm {_f(st['std'])}$")
        L.append(f"    {_tex_escape(name)} & " + " & ".join(cells) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
