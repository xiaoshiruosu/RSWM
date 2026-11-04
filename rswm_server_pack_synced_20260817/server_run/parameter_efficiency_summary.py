#!/usr/bin/env python3
"""参数效率汇总 (advisor point 8: "LoRA params may still be too large").

把每个训练目录的 param_audit.json (由 lora_remotesense/new/train.py 写出) + 训练用时 +
对应 eval_results.json 指标聚合成一张表, 输出 .csv/.json/.md/.tex (LaTeX 可直接粘贴)。

用法
----
    python server_run/parameter_efficiency_summary.py \
        --exp-dirs $WORK/train_r16_* $WORK/train_r8_* $WORK/train_r128_* \
        --out-prefix $WORK/param_efficiency

    DRYRUN=1 python server_run/parameter_efficiency_summary.py ...   # 只打印计划, exit 0

调用方: server_run/pbs/qsub_corrected_rerun.sh:118-119 (完全一致的参数形式)。

数据来源 (全部实测字段, 缺失一律 null + 大声告警, 绝不臆造)
------------------------------------------------------------
  trainable/total/ratio/rank/alpha/targets/per-class breakdown
      <exp>/param_audit.json                     (train.py 在两个专家 get_peft_model() 后写)
      fallback: <exp>/best_checkpoint/training_config.json 的 ["param_audit"] 键
  训练 walltime
      <exp>/training_summary.json 的 "total_time_seconds"
      (train.py:~972 确实会写; 但只有训练跑完才有, 中途 kill 的 run 没有 -> null)
  峰值显存 (peak GPU memory)
      <exp>/training_summary.json 的 "peak_gpu_mem_allocated_gb" / "peak_gpu_mem_reserved_gb"。
      train.py 在 train() 开头 reset_peak_memory_stats(), 收尾时写 torch.cuda.max_memory_allocated/
      reserved, 故该值只含【训练本身】、不含模型加载与 VAE/T5 swap 的瞬时峰值。
      ⚠️ 在 train.py 加这个打点【之前】跑出来的旧 run, summary 里没有这两个 key -> 保持 null
      + 逐实验告警, 绝不臆造。要真值必须用新版 train.py 重跑该配置。
      PBS 日志里的 NCI resource usage 是整作业 (含 VAE/T5/数据加载) 的值, 与此不等价。
  SSIM/PSNR/LPIPS/FID/FVD/Angular_Error_Mean
      <eval_dir>/eval_results.json 的 ["overall"] (键名见 lora_remotesense/new/metrics.py:355-382)
      eval_dir 定位: --eval-root <root>/eval_<tag>, 否则同级兄弟目录 <exp>/../eval_<tag>
      (train_<tag> -> eval_<tag>, 约定见 pbs/28_r16_sem_attncam.pbs:17-18)

推荐的最小参数效率矩阵 (回应 advisor point 8)
----------------------------------------------
    rank    ∈ {4, 8, 16, 32}
    targets ∈ {attn_only, attn_cam, all}      -> 4 x 3 = 12 格
理由: rank 与 targets 是两个正交的"参数量旋钮"(rank 缩放每个适配矩阵, targets 决定适配多少
矩阵)。只扫 rank 无法回答"FFN 上的 LoRA 是否白给", 只扫 targets 无法回答"rank 是否过剩"。

**本仓库现有 PBS 只覆盖该矩阵 12 格中的 3 格** (修正重跑作业集 20-31, 由
qsub_corrected_rerun.sh 驱动):
    r16 x all        -> 20-25 (sem/geo x seed 42/43/44)
    r16 x attn_cam   -> 28 (sem), 29 (geo)
    r8  x all        -> 30 (sem)
另有 r128 x all -> 26/27, rank 128 在推荐矩阵之外 (作为"参数量上界"对照仍有用)。
注意: 旧作业 05/06/07 (r16/r32/r64, targets 默认 all) 属于**修正前**的老流水线, 用的是老
builder + 老 WORK, 结果与 20-31 不可比, 不要混进同一张表。

其余 9 格 (r4 全部; r8 x attn_only/attn_cam; r16 x attn_only; r32 全部) **需要新写 PBS 作业**。
本脚本**故意不自动生成全交叉作业** —— 12 格 x (sem/geo) 是一大笔 GPU 预算, 该由人按论文
需要挑子集 (例如先补 r4/r8/r32 x all 把 rank 曲线做完, 再补 r16 x attn_only 把 targets 消融
做完 = 6 格), 而不是脚本闷头铺满。
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# eval_results.json["overall"] 里要抽的指标 (键名与 metrics.py 完全一致)
EVAL_METRICS = ["SSIM", "PSNR", "LPIPS", "FID", "FVD", "Angular_Error_Mean",
                # 推理开销(evaluate.py 从 meta 的 infer_seconds 聚合;旧 eval 目录无 -> null)
                "Infer_Seconds_Mean", "Infer_FPS_Mean"]

_warnings: List[str] = []


def warn(msg: str) -> None:
    """大声告警: 同时进 stderr 和最终报告 (缺字段绝不静默)。"""
    _warnings.append(msg)
    print(f"[WARN] {msg}", file=sys.stderr)


def _clean(x: Any) -> Any:
    """NaN/Inf -> None。metrics.py:367,373 在 FID/FVD 失败时会写 float('nan'),
    直接 json.dump 会产出非法 JSON (NaN 字面量)。"""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _load_json(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        warn(f"{path} 解析失败 ({e!r}) -> 视为缺失")
        return None


def _tag_of(exp_dir: Path) -> str:
    """train_r16_sem_s42 -> r16_sem_s42 (约定见 pbs/28_r16_sem_attncam.pbs:17-18)。"""
    name = exp_dir.name
    return name[len("train_"):] if name.startswith("train_") else name


def _find_eval_dir(exp_dir: Path, tag: str, eval_root: Optional[Path]) -> Optional[Path]:
    cands = [eval_root / f"eval_{tag}"] if eval_root else [exp_dir.parent / f"eval_{tag}"]
    for c in cands:
        if (c / "eval_results.json").is_file():
            return c
    return None


def _get_param_audit(exp_dir: Path, tag: str) -> Optional[dict]:
    """优先 <exp>/param_audit.json; 回退 checkpoint 里折叠的副本。"""
    audit = _load_json(exp_dir / "param_audit.json")
    if audit is not None:
        return audit
    for ckpt in ("best_checkpoint", ):
        cfg = _load_json(exp_dir / ckpt / "training_config.json")
        if cfg and cfg.get("param_audit"):
            warn(f"[{tag}] 没有 param_audit.json, 改用 {ckpt}/training_config.json 里的副本")
            return cfg["param_audit"]
    return None


def collect(exp_dir: Path, eval_root: Optional[Path]) -> Dict[str, Any]:
    tag = _tag_of(exp_dir)
    row: Dict[str, Any] = {
        "tag": tag,
        "exp_dir": str(exp_dir),
        "lora_rank": None, "lora_alpha": None, "lora_targets": None,
        "trainable_params": None, "total_params": None, "trainable_ratio_pct": None,
        "trainable_low": None, "trainable_high": None,
        # peak_gpu_mem_gb: 由 train.py 的 training_summary.json 提供 (见 collect() 下方)
        "peak_gpu_mem_gb": None,
        "peak_gpu_mem_reserved_gb": None,
        "train_walltime_s": None,
        "eval_dir": None,
    }
    for m in EVAL_METRICS:
        row[m] = None
    for c in ("self_attn", "cross_attn", "cam", "ffn", "other"):
        row[f"trainable_{c}"] = None

    if not exp_dir.is_dir():
        warn(f"[{tag}] 实验目录不存在: {exp_dir} -> 整行 null")
        return row

    # --- param_audit.json ---
    audit = _get_param_audit(exp_dir, tag)
    if audit is None:
        warn(f"[{tag}] 缺 param_audit.json (且 checkpoint 里也没有副本) -> "
             f"参数列全部 null。该 run 可能是加审计之前跑的, 或训练没到写审计那步。")
    else:
        row["lora_rank"] = audit.get("lora_rank")
        row["lora_alpha"] = audit.get("lora_alpha")
        row["lora_targets"] = audit.get("lora_targets")
        comb = audit.get("combined") or {}
        row["trainable_params"] = comb.get("trainable_params")
        row["total_params"] = comb.get("total_params")
        row["trainable_ratio_pct"] = _clean(comb.get("trainable_ratio_pct"))
        experts = audit.get("experts") or {}
        for e in ("low", "high"):
            row[f"trainable_{e}"] = (experts.get(e) or {}).get("trainable_params")
        # per-module-class: 用 low 专家 (两专家结构对称, 同 rank/targets)
        low = experts.get("low") or {}
        bc = low.get("trainable_by_module_class") or {}
        if not bc:
            warn(f"[{tag}] param_audit 里没有 trainable_by_module_class -> 分类列 null")
        for c in ("self_attn", "cross_attn", "cam", "ffn", "other"):
            row[f"trainable_{c}"] = (bc.get(c) or {}).get("n_params")

    # --- 训练 walltime ---
    summ = _load_json(exp_dir / "training_summary.json")
    if summ is None:
        warn(f"[{tag}] 缺 training_summary.json -> train_walltime_s=null "
             f"(训练未跑完 / 被中断时不会写这个文件)")
    else:
        row["train_walltime_s"] = _clean(summ.get("total_time_seconds"))

    # --- 峰值显存: train.py 训练收尾时写入 (torch.cuda.max_memory_*),
    #     计数器在 train() 开头 reset,故只含训练本身、不含模型加载瞬时峰值。
    #     老的 run(加打点之前跑的)没有这两个 key -> 保持 null + 告警,绝不臆造。
    if summ is not None:
        row["peak_gpu_mem_gb"] = _clean(summ.get("peak_gpu_mem_allocated_gb"))
        row["peak_gpu_mem_reserved_gb"] = _clean(summ.get("peak_gpu_mem_reserved_gb"))
        if row["peak_gpu_mem_gb"] is None:
            warn(f"[{tag}] peak_gpu_mem_gb=null: training_summary.json 里没有 "
                 f"peak_gpu_mem_allocated_gb —— 该 run 是在 train.py 加显存打点之前跑的, "
                 f"或在 CPU 上跑。重跑该 run 才会有真值;不臆造。")

    # --- eval_results.json ---
    eval_dir = _find_eval_dir(exp_dir, tag, eval_root)
    if eval_dir is None:
        where = f"{eval_root}/eval_{tag}" if eval_root else f"{exp_dir.parent}/eval_{tag}"
        warn(f"[{tag}] 找不到 eval_results.json (期望 {where}) -> 指标列全部 null")
    else:
        row["eval_dir"] = str(eval_dir)
        res = _load_json(eval_dir / "eval_results.json") or {}
        overall = res.get("overall")
        if not overall:
            warn(f"[{tag}] {eval_dir}/eval_results.json 里没有 'overall' 段 -> 指标列 null")
        else:
            for m in EVAL_METRICS:
                if m not in overall:
                    warn(f"[{tag}] eval overall 缺指标 {m} -> null")
                else:
                    row[m] = _clean(overall[m])
    return row


# --------------------------------------------------------------------- emit
def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


_MD_COLS = [
    ("tag", "Config", 0), ("lora_rank", "Rank", 0), ("lora_targets", "Targets", 0),
    ("trainable_params", "Trainable (low+high)", 0),
    ("trainable_ratio_pct", "Trainable %", 4),
    ("peak_gpu_mem_gb", "Peak GPU (GB)", 2),
    ("Infer_Seconds_Mean", "Infer s/sample", 1),
    ("Infer_FPS_Mean", "Infer FPS", 2),
    ("train_walltime_s", "Walltime (s)", 1),
    ("SSIM", "SSIM", 4), ("LPIPS", "LPIPS", 4), ("FID", "FID", 2),
    ("FVD", "FVD", 2), ("Angular_Error_Mean", "AngErr", 2),
]


def write_csv(rows, path: Path) -> None:
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r[k] is None else r[k]) for k in keys})


def write_json(rows, path: Path) -> None:
    payload = {
        "schema_version": 1,
        "generated_by": "server_run/parameter_efficiency_summary.py",
        "note_peak_gpu_mem": "From training_summary.json (torch.cuda.max_memory_*, reset at train() start; training-only, excludes model load). null = run predates the instrumentation or ran on CPU.",
        "warnings": _warnings,
        "rows": rows,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_md(rows, path: Path) -> None:
    L = ["# Parameter efficiency summary (advisor point 8)", ""]
    L.append("Trainable = LoRA params across BOTH experts (low+high), i.e. what ships.")
    L.append("`n/a` = field genuinely missing upstream, never a guessed value.")
    L.append("")
    L.append("| " + " | ".join(c[1] for c in _MD_COLS) + " |")
    L.append("|" + "|".join("---" for _ in _MD_COLS) + "|")
    for r in rows:
        L.append("| " + " | ".join(_fmt(r[k], nd) for k, _, nd in _MD_COLS) + " |")
    L += ["", "## Peak GPU memory", "",
          "All `n/a`: `lora_remotesense/new/train.py` records no CUDA memory statistics",
          "(no `torch.cuda.max_memory_allocated()` call exists). To populate this column,",
          "add a memory probe to train.py first. Values were NOT invented.", ""]
    if _warnings:
        L += ["## Warnings", ""] + [f"- {w}" for w in _warnings] + [""]
    path.write_text("\n".join(L), encoding="utf-8")


def write_tex(rows, path: Path) -> None:
    cols = ["tag", "lora_rank", "lora_targets", "trainable_params",
            "trainable_ratio_pct", "SSIM", "LPIPS", "FID", "FVD", "Angular_Error_Mean"]
    heads = ["Config", "Rank", "Targets", "Trainable", "\\%", "SSIM$\\uparrow$",
             "LPIPS$\\downarrow$", "FID$\\downarrow$", "FVD$\\downarrow$",
             "AngErr$\\downarrow$"]
    nds = [0, 0, 0, 0, 4, 4, 4, 2, 2, 2]

    def esc(s: str) -> str:
        return s.replace("_", "\\_")

    L = ["% Auto-generated by server_run/parameter_efficiency_summary.py -- do not hand-edit.",
         "% Trainable = LoRA params over BOTH experts (low+high). n/a = missing upstream.",
         "\\begin{table}[t]", "\\centering",
         "\\caption{LoRA parameter efficiency. Trainable counts both noise experts.}",
         "\\label{tab:param-efficiency}",
         "\\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", "\\toprule",
         " & ".join(heads) + " \\\\", "\\midrule"]
    for r in rows:
        cells = []
        for k, nd in zip(cols, nds):
            v = _fmt(r[k], nd)
            cells.append(esc(v) if isinstance(r[k], str) or v == "n/a" else v)
        L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="汇总 LoRA 参数效率 (param_audit.json + walltime + eval 指标)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--exp-dirs", nargs="+", required=True,
                    help="训练输出目录 (如 $WORK/train_r16_sem_s42), 支持 shell glob 展开")
    ap.add_argument("--out-prefix", required=True,
                    help="输出前缀 -> PREFIX.{csv,json,md,tex}")
    ap.add_argument("--eval-root", default=None,
                    help="eval 目录根 (找 <root>/eval_<tag>); 默认取 <exp_dir>/../eval_<tag>")
    a = ap.parse_args()

    exp_dirs = [Path(d).resolve() for d in a.exp_dirs]
    eval_root = Path(a.eval_root).resolve() if a.eval_root else None
    prefix = Path(a.out_prefix).resolve()
    # 注意: 用字符串拼接而非 with_suffix —— 前缀里若含 '.' (如 .../param.v2),
    # with_suffix 会把 '.v2' 当后缀替换掉。
    outs = {e: Path(str(prefix) + "." + e) for e in ("csv", "json", "md", "tex")}

    if os.environ.get("DRYRUN") == "1":
        print("=== DRYRUN: 参数效率汇总计划 (不写任何文件) ===")
        print(f"实验目录 ({len(exp_dirs)}):")
        for d in exp_dirs:
            tag = _tag_of(d)
            ed = _find_eval_dir(d, tag, eval_root)
            print(f"  - {d}  [{'存在' if d.is_dir() else '缺失!'}]")
            print(f"      tag={tag}")
            print(f"      param_audit : {d/'param_audit.json'} "
                  f"[{'ok' if (d/'param_audit.json').is_file() else '缺'}]")
            print(f"      walltime    : {d/'training_summary.json'} "
                  f"[{'ok' if (d/'training_summary.json').is_file() else '缺'}]")
            print(f"      eval        : {ed if ed else '(未找到 eval_%s)' % tag}")
        print(f"eval-root: {eval_root or '(默认: 各 exp 的同级 eval_<tag>)'}")
        print("将写出:")
        for e in ("csv", "json", "md", "tex"):
            print(f"  - {outs[e]}")
        print("注: peak_gpu_mem_gb 来自 training_summary.json;旧 run(加打点前跑的)为 null,不臆造。")
        return 0

    rows = [collect(d, eval_root) for d in exp_dirs]
    rows = [r for r in rows if Path(r["exp_dir"]).is_dir()]
    if not rows:
        print("[FATAL] 没有任何有效实验目录, 无法汇总。", file=sys.stderr)
        return 1

    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, outs["csv"])
    write_json(rows, outs["json"])
    write_md(rows, outs["md"])
    write_tex(rows, outs["tex"])

    print(f"[ok] 汇总 {len(rows)} 个实验, {len(_warnings)} 条告警")
    for e in ("csv", "json", "md", "tex"):
        print(f"  -> {outs[e]}")
    if _warnings:
        print(f"[note] {len(_warnings)} 条告警已记入 {outs['json']} / {outs['md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
