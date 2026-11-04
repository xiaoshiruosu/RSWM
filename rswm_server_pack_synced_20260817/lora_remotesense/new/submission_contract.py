#!/usr/bin/env python3
# =============================================================================
# submission_contract.py — 不可作弊的评分契约(唯一权威)
#
# 外部审查(2026-07-30)动态复现了四种作弊,全部通过 **静默截断** 实现:
#
#     完整 17 帧            → PSNR = inf
#     只提交第 1 帧          → PSNR = inf      ← 时间轴 min() 截断
#     只提交左上角 1/4 图像   → PSNR = inf      ← 空间 min() 截断
#
# 根因是 metrics.py 里到处写着 `T = min(len(gen), len(gt))` /
# `h = min(g.shape[1], r.shape[1])` —— 本意是"容错",实际是"少交就少算,
# 分数还更好看"。再叠上 evaluate.py 允许 20% 样本推理失败、以及
# 只评估 generated/ 里**实际存在**的文件(不要求提交完整测试 ID 集),
# 一个什么都不做的提交也能拿到接近满分。
#
# 本模块把"这份提交合不合法"收成单一权威,规则是**精确匹配**:
#   * 样本 ID 集必须与要求的完全一致 —— 不许缺、不许多;
#   * 每个样本的 (T, H, W, C) 必须精确一致 —— 不许裁剪、截断、自动补齐;
#   * dtype 必须一致(uint8 与 float 的值域差会让指标不可比);
#   * 任何样本失败即整份提交失败 —— 不存在"容忍比例";
#   * 条件帧(首帧)是模型的**输入**,必须从保真指标里排除 ——
#     否则每份提交都白拿一帧的满分。
#
# 宽松模式(`RSWM_LENIENT_METRICS=1`)只为**本地排障**保留,并且是**污染性**的:
# 一旦用过,`is_tainted()` 为真,调用方必须把该次结果标成
# headline_eligible=false。宽松而不留痕,就是把作弊做成默认。
# =============================================================================
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

__all__ = [
    "check_same_shape", "check_id_set", "check_same_dtype",
    "check_no_failures", "assert_same_dtype",
    "strict_mode", "is_tainted", "taint_reasons", "reset_taint",
    "assert_same_shape", "assert_id_set", "assert_no_failures",
    "CONDITION_FRAMES", "strip_condition_frames", "MIN_SAMPLES_FOR_DISTRIBUTION",
    "SubmissionError",
]

# 首帧是 i2v 的条件输入,不是模型的预测。把它算进 SSIM/PSNR/LPIPS/FID/FVD
# 等于给每份提交白送一帧的满分,并且系统性抬高所有臂的分数。
CONDITION_FRAMES = 1

# 分布型指标(FID/FVD)在极小分组上无意义。方向×场景交叉块此前只要 n>=2
# 就算 FID/FVD —— 那是纯噪声。这个下限是硬门,不是建议。
MIN_SAMPLES_FOR_DISTRIBUTION = 32

_TAINT: list[str] = []


class SubmissionError(RuntimeError):
    """提交不合契约。**不要**在评分路径上捕获它。"""


def _envflag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def strict_mode() -> bool:
    """默认严格。宽松只能靠显式 RSWM_LENIENT_METRICS 打开。"""
    return not _envflag("RSWM_LENIENT_METRICS", False)


def is_tainted() -> bool:
    return bool(_TAINT)


def taint_reasons() -> list[str]:
    return list(_TAINT)


def reset_taint() -> None:
    _TAINT.clear()


def _taint(msg: str) -> None:
    if msg not in _TAINT:
        _TAINT.append(msg)
        print(f"  ⚠️ [contract] 宽松模式放行:{msg}\n"
              f"     本次结果已被标记为 tainted,不得作为 headline 数字。")


# =============================================================================
# 两种呈现,一个权威。
#   check_*  → 返回违规描述列表(空列表 = 合规)。评分器用它,以便一次报全部违规。
#   assert_* → 第一条违规即抛。指标计算用它,以便绝不带着坏输入往下算。
# 规则本身只写在 check_* 里 —— 上几轮的 P0 全部源自"孪生入口各写一份"。
# =============================================================================

def check_same_shape(gen, gt, *, what: str, key: Optional[str] = None) -> list[str]:
    g = tuple(getattr(gen, "shape", ()))
    t = tuple(getattr(gt, "shape", ()))
    if g == t:
        return []
    where = f"{what}" + (f"(样本 {key})" if key else "")
    return [f"{where}:形状不一致 gen={g} gt={t} —— 评分要求 (T,H,W,C) 精确一致,"
            f"不许裁剪、截断或自动补齐"]


def check_id_set(submitted: Iterable[str], required: Iterable[str],
                 *, what: str = "提交") -> list[str]:
    s, r = set(submitted), set(required)
    missing, extra = sorted(r - s), sorted(s - r)
    out = []
    if missing:
        out.append(f"{what}缺 {len(missing)}/{len(r)} 个样本 —— 必须交齐,"
                   f"不接受部分提交: {missing[:8]}{'…' if len(missing) > 8 else ''}")
    if extra:
        out.append(f"{what}多出 {len(extra)} 个不在要求集里的样本(train/val 的 id?"
                   f"改名?)—— 必须与要求集**恰好**相等: "
                   f"{extra[:8]}{'…' if len(extra) > 8 else ''}")
    return out


def check_same_dtype(gen, gt, *, key: Optional[str] = None) -> list[str]:
    gd, td = getattr(gen, "dtype", None), getattr(gt, "dtype", None)
    if gd == td:
        return []
    return [f"样本 {key}:dtype 不一致 gen={gd} gt={td} —— "
            f"uint8 与 float 的值域差会让指标不可比"]


def check_no_failures(failed: int, attempted: int, *, what: str = "推理") -> list[str]:
    if failed <= 0:
        return []
    return [f"{what}有 {failed}/{attempted} 个样本失败 —— 评分要求零失败;"
            f"容忍失败等于允许提交方放弃最难的样本"]


def _enforce(violations: list[str]) -> None:
    """把 check_* 的结果按严格/宽松处理。"""
    if not violations:
        return
    msg = "\n  ".join(violations)
    if strict_mode():
        raise SubmissionError(msg)
    for v in violations:
        _taint(v)


def assert_same_shape(gen, gt, *, what: str, key: Optional[str] = None) -> None:
    """gen 与 gt 的形状必须精确一致。

    严格模式(默认)下不一致即抛。宽松模式下放行并污染结果 ——
    调用方随后仍然会按 min() 截断,但至少留下了痕迹。
    """
    _enforce(check_same_shape(gen, gt, what=what, key=key))


def assert_same_dtype(gen, gt, *, key: Optional[str] = None) -> None:
    _enforce(check_same_dtype(gen, gt, key=key))


def assert_id_set(submitted: Iterable[str], required: Iterable[str],
                  *, what: str = "提交") -> None:
    """样本 ID 集必须完全一致。

    旧行为是"只评估 generated/ 里实际存在的文件" —— 于是少交样本不会被发现,
    多交样本(比如把训练集的结果混进来)也不会。
    """
    _enforce(check_id_set(submitted, required, what=what))


def assert_no_failures(failed: int, attempted: int, *, what: str = "推理") -> None:
    """任何样本失败即整份失败。

    旧行为容忍 20% 失败 —— 那意味着一份提交可以"选择性放弃"最难的
    五分之一样本,而剩下的分数照进主表。
    """
    _enforce(check_no_failures(failed, attempted, what=what))


def strip_condition_frames(gen, gt, *, n: int = CONDITION_FRAMES):
    """从保真指标里排除条件帧(首帧)。

    首帧是 i2v 的**输入**。把它算进 SSIM/PSNR/LPIPS/FID/FVD 相当于每份提交
    白拿一帧满分,并系统性抬高所有臂的分数 —— 而且抬高的幅度随总帧数变化
    (17 帧时 1/17≈5.9%,33 帧时 1/33≈3.0%),让不同 K 的档不可比。

    返回 (gen_eval, gt_eval)。帧数不足时抛 —— 这种数据本身就不该进评分。
    """
    if n <= 0:
        return gen, gt
    for name, v in (("gen", gen), ("gt", gt)):
        if len(v) <= n:
            raise SubmissionError(
                f"{name} 只有 {len(v)} 帧,排除 {n} 个条件帧后无剩余 —— "
                f"无法评分。canonical 帧数是 9/17/25/33。")
    return gen[n:], gt[n:]


def assert_enough_for_distribution(n: int, *, metric: str, group: str = "") -> bool:
    """分布型指标的样本量下限。返回 True 表示可以算。

    不抛 —— 分组太小时正确的做法是**不算**并如实标注,而不是让整个评估失败。
    但绝不能像旧代码那样在 n>=2 的交叉块上算 FID/FVD 然后当数字用。
    """
    if n >= MIN_SAMPLES_FOR_DISTRIBUTION:
        return True
    print(f"  [contract] 跳过 {metric}{f'({group})' if group else ''}:"
          f"n={n} < {MIN_SAMPLES_FOR_DISTRIBUTION} —— "
          f"分布型指标在这个样本量上是噪声,不予计算(旧代码 n>=2 就算)。")
    return False


def contract_summary() -> dict:
    """落进 eval_results / run_manifest,让下游能看见这次是不是严格评的。"""
    return {
        "strict": strict_mode(),
        "tainted": is_tainted(),
        "taint_reasons": taint_reasons(),
        "condition_frames_excluded": CONDITION_FRAMES,
        "min_samples_for_distribution": MIN_SAMPLES_FOR_DISTRIBUTION,
    }
