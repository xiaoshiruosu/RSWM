#!/usr/bin/env python3
# =============================================================================
# semgate_arbiters.py — 语义门的**多仲裁者**共识与一致性统计(零真人参与)
#
# 用户明确要求语义门审计**不能依赖人工标注**。人工标注里有两样东西要替掉:
#
#   ① 标签  → 用【构造保真值】替(见 semgate_auto_auditset.py):
#             正例 = 类标图的中心窗;硬负例 = 取自其它场景池的窗却声称本池。
#   ② 双人一致性 κ → 用【多个独立仲裁者之间的 κ】替(本文件)。
#
# 为什么 κ 不能省:语义门的判据是 CLIP 零样本,它本身就是一个"标注者"。
# 如果换一组同类仲裁者就给出不同判定,那么拿任何一个去算 precision/recall
# 都没有意义 —— 这与"标注者之间对不齐时下游全不可信"是同一个道理。
# 所以 κ 不达标时本模块**拒绝**输出下游指标(fail-closed),不返回默认值。
#
# 本机现实(2026-07-30 实测):没有 transformers / open_clip / 任何 CLIP 权重,
# 只有 torch 2.13 CPU。所以**真实的 CLIP 仲裁者在本机跑不起来** ——
# 本文件把机制建好并让统计量现在就能闭式验证,真实仲裁需要在集群上跑
# (那里有 offline_prefetch 拉好的权重)。`probe_available_arbiters()` 会
# 如实报告哪些可用、哪些缺什么,**绝不假装某个模型可用**。
#
# 用法:
#   from semgate_arbiters import (probe_available_arbiters, build_arbiters,
#                                 cohens_kappa, fleiss_kappa, wilson_ci,
#                                 assert_kappa_ok)
#   python server_run/semgate_arbiters.py --self-test
# =============================================================================
from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# κ 的常规解读(Landis & Koch 1977):
#   <0.00 差 / 0.00-0.20 轻微 / 0.21-0.40 一般 / 0.41-0.60 中等 /
#   0.61-0.80 显著 / 0.81-1.00 几乎完全
# 本项目取 0.60 作为"能不能拿这批判定去算 precision/recall"的门槛:
# 低于它意味着仲裁者之间连"中等以上"都达不到,那么把任一个当真值都是任意选择。
# 这不是行业标准,是本项目的取舍 —— 写在这里以便被质疑。
KAPPA_GATE_DEFAULT = 0.60

__all__ = [
    "ArbiterSpec", "probe_available_arbiters", "build_arbiters",
    "cohens_kappa", "fleiss_kappa", "wilson_ci", "bootstrap_ci",
    "assert_kappa_ok", "KappaUnavailable", "KAPPA_GATE_DEFAULT",
    "perturbation_flip_rate",
]


class KappaUnavailable(RuntimeError):
    """κ 无法计算或不达标。**不要**在审计路径上捕获它。"""


# =============================================================================
# 统计量:三个都用闭式实现,--self-test 里对教科书例子逐位核对
# =============================================================================

def wilson_ci(x: int, n: int, *, z: float = 1.959963984540054
              ) -> Tuple[float, float]:
    """二项比例的 Wilson 区间。

    为什么不用 Wald(p̂ ± z·√(p̂(1-p̂)/n)):小样本或 p̂ 贴近 0/1 时 Wald 的
    覆盖率崩坏,区间还会越出 [0,1]。本项目每类的审计样本量只有几十个,
    而误拒率恰恰贴近 0 —— 正是 Wald 最不能用的区域。
    """
    if n <= 0:
        raise KappaUnavailable(f"Wilson CI 需要 n>0,得到 n={n}")
    p = x / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def cohens_kappa(a: Sequence[bool], b: Sequence[bool]) -> float:
    """两个仲裁者的 Cohen's κ(二分类)。

    κ = (po - pe) / (1 - pe)。**pe==1 时 κ 未定义** —— 那发生在两个仲裁者
    都把所有项判成同一类的时候(比如都说"全部通过")。此时 po 也是 1,
    0/0。旧式实现常在这里返回 1.0("完全一致"),那是错的:
    一致但零信息量,不能当成"判定可信"的证据。所以抛。
    """
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if a.shape != b.shape:
        raise KappaUnavailable(f"两个仲裁者的判定长度不同 {a.shape} vs {b.shape}")
    n = a.size
    if n == 0:
        raise KappaUnavailable("没有可比的项")
    both_y = int(np.sum(a & b))
    both_n = int(np.sum(~a & ~b))
    po = (both_y + both_n) / n
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if abs(1.0 - pe) < 1e-12:
        raise KappaUnavailable(
            f"pe={pe:.6f} → κ 未定义(两个仲裁者把全部 {n} 项判成同一类)。"
            f"一致但零信息量,不构成『判定可信』的证据。")
    return (po - pe) / (1.0 - pe)


def fleiss_kappa(ratings: np.ndarray) -> float:
    """多仲裁者的 Fleiss' κ。

    ratings: (n_items, n_categories) 的计数矩阵,每行之和 = 仲裁者数 m。
    二分类时 n_categories=2,列 0 = 判"不通过"数,列 1 = 判"通过"数。

    P̄e==1 时同样未定义(所有仲裁者对所有项给同一类)—— 抛,理由同 Cohen。
    """
    r = np.asarray(ratings, dtype=float)
    if r.ndim != 2 or r.shape[0] == 0:
        raise KappaUnavailable(f"ratings 形状非法 {r.shape}")
    row_sums = r.sum(axis=1)
    m = row_sums[0]
    if m < 2:
        raise KappaUnavailable(f"Fleiss' κ 需要 ≥2 个仲裁者,得到 m={m}")
    if not np.allclose(row_sums, m):
        raise KappaUnavailable(
            f"各项的仲裁者数不一致(min={row_sums.min()}, max={row_sums.max()})"
            f" —— Fleiss' κ 要求每项被同样多的仲裁者判过")
    n = r.shape[0]
    p_j = r.sum(axis=0) / (n * m)
    P_i = (np.sum(r * r, axis=1) - m) / (m * (m - 1))
    P_bar = P_i.mean()
    Pe_bar = float(np.sum(p_j * p_j))
    if abs(1.0 - Pe_bar) < 1e-12:
        raise KappaUnavailable(
            f"P̄e={Pe_bar:.6f} → Fleiss' κ 未定义(全部仲裁者对全部项给同一类)")
    return (P_bar - Pe_bar) / (1.0 - Pe_bar)


def bootstrap_ci(stat_fn: Callable[[np.ndarray], float], n_items: int, *,
                 n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> Tuple[Optional[float], Optional[float], int]:
    """对**按项重抽样**的统计量做百分位 bootstrap CI。

    按项(而不是按仲裁者)重抽样:项是独立单位,仲裁者是固定的一组。
    重抽样中若某次 κ 未定义(比如抽到的子集全同类),该次**丢弃**并计数 ——
    不用 0 或 1 顶替。丢弃比例过高时返回 None,由调用方 fail-closed。
    """
    rng = np.random.default_rng(seed)
    vals, dropped = [], 0
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        try:
            vals.append(float(stat_fn(idx)))
        except KappaUnavailable:
            dropped += 1
    if len(vals) < max(50, n_boot // 10):
        return None, None, dropped
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), dropped


def assert_kappa_ok(kappa: Optional[float], *, gate: float = KAPPA_GATE_DEFAULT,
                    ci: Tuple[Optional[float], Optional[float]] = (None, None),
                    what: str = "仲裁者一致性") -> None:
    """κ 不达标即拒绝继续 —— fail-closed。

    判据用 **CI 下界**(有的话)而不是点估计:κ 点估计 0.62 但 CI 下界 0.31
    的时候,说"一致性达标"是过度自信。拿不到 CI 时退回点估计并说明。
    """
    if kappa is None or not np.isfinite(kappa):
        raise KappaUnavailable(
            f"{what}:κ 无法计算 —— 不得据此输出 precision/recall。")
    lo = ci[0]
    basis, val = (("CI 下界", lo) if lo is not None else ("点估计", kappa))
    if val < gate:
        raise KappaUnavailable(
            f"{what}:κ={kappa:.4f}"
            + (f"(95% CI [{ci[0]:.4f}, {ci[1]:.4f}])" if lo is not None else "")
            + f",{basis} {val:.4f} < 门槛 {gate:.2f}。\n"
            f"  仲裁者之间连『中等以上』一致都达不到,把任一个当真值都是任意选择,"
            f"因此**拒绝**输出下游 precision/recall。\n"
            f"  这不是可以放宽的阈值 —— 要么换更一致的仲裁者族,"
            f"要么承认这套判据在本任务上不可靠。")


def perturbation_flip_rate(base: Sequence[bool],
                           perturbed: Sequence[Sequence[bool]]) -> Dict[str, float]:
    """扰动稳定性:**完全不需要标签**的可靠性证据。

    同一个窗在标签保持的扰动(裁剪抖动/翻转/JPEG 重压/亮度)下,判定应当不变。
    翻转率就是该判据的不稳定度。这是最干净的自动证据 ——
    它不问"判得对不对",只问"判得稳不稳",而不稳的判据无论对错都不可用。
    """
    b = np.asarray(base, dtype=bool)
    if not perturbed:
        raise KappaUnavailable("没有扰动样本,无法算稳定性")
    flips, total = 0, 0
    per = []
    for p in perturbed:
        pa = np.asarray(p, dtype=bool)
        if pa.shape != b.shape:
            raise KappaUnavailable(f"扰动判定长度 {pa.shape} != 基线 {b.shape}")
        f = int(np.sum(pa != b))
        flips += f
        total += b.size
        per.append(f / b.size)
    lo, hi = wilson_ci(flips, total)
    return {"flip_rate": flips / total, "ci95_lo": lo, "ci95_hi": hi,
            "n_comparisons": total, "per_perturbation": per}


# =============================================================================
# 仲裁者
# =============================================================================

@dataclass
class ArbiterSpec:
    """一个仲裁者 = 一个与 SemanticGate 的 score_fn 契约兼容的判定器。

    契约(见 lora_remotesense/semantic_gate.py):
        score_fn(frames_uint8_list, category) -> List[bool]
    """
    name: str
    kind: str                      # "clip_model" / "prompt_family" / "aggregation" / "synthetic"
    score_fn: Callable[[List[np.ndarray], str], List[bool]]
    requires: List[str] = field(default_factory=list)
    note: str = ""


def probe_available_arbiters() -> Dict[str, object]:
    """如实报告本机能构造哪些仲裁者。**绝不假装某个模型可用。**"""
    have = {}
    for mod in ("torch", "transformers", "open_clip", "torchvision", "PIL", "cv2"):
        try:
            __import__(mod)
            have[mod] = True
        except ImportError:
            have[mod] = False
    weights = []
    hf_home = os.environ.get("HF_HOME", "")
    for d in (os.environ.get("HF_HUB_CACHE", ""),
              os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
              os.path.join(hf_home, "hub") if hf_home else ""):
        if d and os.path.isdir(d):
            try:
                weights += [x for x in os.listdir(d) if "clip" in x.lower()]
            except OSError:
                pass
    clip_ok = have.get("transformers", False) and bool(weights)
    return {
        "modules": have,
        "clip_weight_dirs": sorted(set(weights)),
        "clip_arbiters_available": clip_ok,
        "verdict": (
            "可构造真实 CLIP 仲裁者族" if clip_ok else
            "**本机无法构造真实 CLIP 仲裁者**(缺 transformers 或 CLIP 权重)。"
            "多模型 κ 必须在集群上跑(offline_prefetch 已拉好权重)。"
            "本机只能用合成仲裁者验证统计量本身。"),
    }


def build_arbiters(*, allow_synthetic_only: bool = False) -> List[ArbiterSpec]:
    """构造可用的仲裁者族。

    真实族(需权重):同一 CLIP 主干 + 三种**决策规则**差异 ——
      A. prompt 措辞族(class_phrase 的模板集换一组)
      B. 聚合口径(类级模板均值 ↔ 池内 max)
      C. 预处理(pad-square ↔ center-crop)
    这三者不是"同一个判定跑三遍",它们的判定会真实分歧 ——
    分歧程度正是我们要量的东西(如果换个措辞就翻案,判据本身就不稳)。

    权重不可用时:**抛**,除非显式 allow_synthetic_only=True。
    悄悄降级成合成仲裁者然后报一个漂亮的 κ,是这个项目历史上最典型的 P0 形态。
    """
    probe = probe_available_arbiters()
    if not probe["clip_arbiters_available"]:
        if not allow_synthetic_only:
            raise KappaUnavailable(
                "无法构造真实仲裁者:" + str(probe["verdict"]) + "\n"
                "  要在本机只验统计量,显式传 allow_synthetic_only=True,"
                "  那样产出的 κ **只证明代码正确,不证明语义门可靠**。")
        return _synthetic_arbiters()
    return _clip_arbiters()


def _clip_arbiters() -> List[ArbiterSpec]:
    """真实 CLIP 仲裁者族。只在有权重的环境构造。

    这里返回的 score_fn 都是薄封装:内部构造 SemanticGate 并改一个决策维度,
    从而与生产判据**同源**(同一模型、同一 taxonomy),只在措辞/聚合/预处理上分歧。
    """
    _here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(_here, "lora_remotesense"))
    import semantic_gate as sg   # noqa: E402

    def _mk(name, kind, note, **kw):
        def fn(frames, category, _kw=kw):
            g = sg.SemanticGate(backend="clip", frames_per_sample=len(frames), **_kw)
            g.ensure_ready()
            v = g.check_frames(frames, category)
            return list(v.per_frame)
        return ArbiterSpec(name=name, kind=kind, score_fn=fn,
                           requires=["transformers", "clip_weights"], note=note)

    return [
        _mk("clip_base", "clip_model",
            "生产判据本身(基线仲裁者)"),
        # 下面两个换 model_name —— 需要对应权重,缺则 ensure_ready 会大声失败,
        # 那是对的:不许静默退回基线然后报"三个仲裁者高度一致"。
        _mk("clip_vit_b16", "clip_model",
            "换主干(ViT-B/16):同 taxonomy 不同视觉表征",
            model_name="openai/clip-vit-base-patch16"),
        _mk("clip_vit_l14", "clip_model",
            "换主干(ViT-L/14):容量差一档",
            model_name="openai/clip-vit-large-patch14"),
    ]


def _synthetic_arbiters() -> List[ArbiterSpec]:
    """合成仲裁者 —— **只用于验证统计量代码本身**,不产生任何关于语义门的结论。

    每个都是确定性的(由 frames 的内容哈希驱动),因此可复现;
    它们之间有可控的一致率,用来核对 κ 的实现。
    """
    def _h(frames, salt):
        acc = 0
        for f in frames:
            a = np.asarray(f)
            acc = (acc * 1000003 + int(a.sum()) + salt) % (2 ** 31 - 1)
        return acc

    def _mk(name, salt, agree_bias):
        def fn(frames, category, _s=salt, _b=agree_bias):
            out = []
            for i, f in enumerate(frames):
                h = _h([f], _s + i)
                out.append(((h % 100) / 100.0) < _b)
            return out
        return ArbiterSpec(name=name, kind="synthetic", score_fn=fn,
                           note="合成仲裁者:只验统计量,不构成语义门结论")

    return [_mk("synth_a", 11, 0.55), _mk("synth_b", 22, 0.55),
            _mk("synth_c", 33, 0.55)]


# =============================================================================
# 自检:统计量必须与闭式/教科书例子逐位吻合
# =============================================================================

def _self_test() -> int:
    P, F = [], []

    def ck(name, cond, detail=""):
        (P if cond else F).append(name)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {detail}" if detail and not cond else ""))

    print("== T1 Cohen's κ 对手算 2×2 例子 ==")
    # a=20 both-yes, b=5 (A yes,B no), c=10 (A no,B yes), d=15 both-no, n=50
    #   po = 35/50 = 0.70
    #   pA1 = 25/50 = 0.5 ; pB1 = 30/50 = 0.6
    #   pe = 0.5*0.6 + 0.5*0.4 = 0.30 + 0.20 = 0.50
    #   κ  = (0.70-0.50)/(1-0.50) = 0.40
    a = [True] * 20 + [True] * 5 + [False] * 10 + [False] * 15
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    k = cohens_kappa(a, b)
    ck("cohen_matches_hand_calc", abs(k - 0.40) < 1e-12, f"得到 {k!r},手算 0.40")

    print("== T2 Cohen's κ 的边界 ==")
    ck("cohen_perfect_is_1",
       abs(cohens_kappa([True, False, True, False],
                        [True, False, True, False]) - 1.0) < 1e-12)
    ck("cohen_total_disagree_is_-1",
       abs(cohens_kappa([True, True, False, False],
                        [False, False, True, True]) - (-1.0)) < 1e-12)
    _raised = False
    try:
        cohens_kappa([True] * 8, [True] * 8)          # pe=1 → 未定义
    except KappaUnavailable:
        _raised = True
    ck("cohen_all_same_raises", _raised,
       "两者都判全 True 时返回了数值 —— 一致但零信息量,不能当可信证据")

    print("== T3 Fleiss' κ ==")
    # 完全一致(3 仲裁者、4 项、二分类,两类都出现过)→ κ=1
    perfect = np.array([[3, 0], [0, 3], [3, 0], [0, 3]], dtype=float)
    ck("fleiss_perfect_is_1", abs(fleiss_kappa(perfect) - 1.0) < 1e-12,
       f"{fleiss_kappa(perfect)!r}")
    # 手算例子:m=3, n=2;item1 = 2 yes/1 no,item2 = 1 yes/2 no
    #   p_j: p_no=(1+2)/6=0.5, p_yes=(2+1)/6=0.5
    #   P_1=(1²+2²-3)/(3*2)=(1+4-3)/6=2/6=1/3 ; P_2 同 = 1/3
    #   P̄=1/3 ; P̄e=0.5²+0.5²=0.5
    #   κ=(1/3-0.5)/(1-0.5) = (-1/6)/0.5 = -1/3
    hand = np.array([[1, 2], [2, 1]], dtype=float)
    kf = fleiss_kappa(hand)
    ck("fleiss_matches_hand_calc", abs(kf - (-1.0 / 3.0)) < 1e-12,
       f"得到 {kf!r},手算 -1/3")
    _raised = False
    try:
        fleiss_kappa(np.array([[0, 3], [0, 3]], dtype=float))   # P̄e=1
    except KappaUnavailable:
        _raised = True
    ck("fleiss_all_same_raises", _raised)
    _raised = False
    try:
        fleiss_kappa(np.array([[1, 2], [1, 1]], dtype=float))   # 行和不等
    except KappaUnavailable:
        _raised = True
    ck("fleiss_ragged_raises", _raised, "各项仲裁者数不一致时应拒")

    print("== T4 Wilson CI ==")
    # 闭式独立复算 n=10, x=3
    z = 1.959963984540054
    p, n = 0.3, 10
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo, hi = wilson_ci(3, 10)
    ck("wilson_matches_formula",
       abs(lo - (c - h)) < 1e-12 and abs(hi - (c + h)) < 1e-12, f"({lo},{hi})")
    lo0, hi0 = wilson_ci(0, 20)
    ck("wilson_x0_in_range", lo0 == 0.0 and 0 < hi0 < 1, f"({lo0},{hi0})")
    lon, hin = wilson_ci(20, 20)
    ck("wilson_xn_in_range", hin == 1.0 and 0 < lon < 1, f"({lon},{hin})")
    # 覆盖率蒙特卡洛:真 p=0.3、n=30,95% 区间的覆盖率应 ≈0.95
    rng = np.random.default_rng(7)
    cov = 0
    TRIALS = 4000
    for _ in range(TRIALS):
        x = int(rng.binomial(30, 0.3))
        l, u = wilson_ci(x, 30)
        cov += (l <= 0.3 <= u)
    rate = cov / TRIALS
    ck("wilson_coverage_near_95", 0.92 <= rate <= 0.985, f"实测 {rate:.3f}")

    print("== T5 κ 门 fail-closed ==")
    _raised = False
    try:
        assert_kappa_ok(0.31, gate=0.60)
    except KappaUnavailable:
        _raised = True
    ck("gate_rejects_low_kappa", _raised)
    _raised = False
    try:
        assert_kappa_ok(0.62, gate=0.60, ci=(0.31, 0.88))
    except KappaUnavailable:
        _raised = True
    ck("gate_uses_ci_lower_bound", _raised,
       "点估计 0.62 过门但 CI 下界 0.31 —— 应按下界拒")
    assert_kappa_ok(0.85, gate=0.60, ci=(0.71, 0.94))
    ck("gate_passes_good_kappa", True)
    _raised = False
    try:
        assert_kappa_ok(None)
    except KappaUnavailable:
        _raised = True
    ck("gate_rejects_none", _raised)

    print("== T6 扰动稳定性 ==")
    base = [True, True, False, False, True]
    ck("flip_rate_stable_is_0",
       perturbation_flip_rate(base, [base, base])["flip_rate"] == 0.0)
    flipped = [not x for x in base]
    ck("flip_rate_all_flip_is_1",
       perturbation_flip_rate(base, [flipped])["flip_rate"] == 1.0)
    half = [True, True, False, True, True]      # 1/5 翻转
    r = perturbation_flip_rate(base, [half])
    ck("flip_rate_partial", abs(r["flip_rate"] - 0.2) < 1e-12, f"{r['flip_rate']}")
    ck("flip_rate_has_ci", 0 <= r["ci95_lo"] <= r["flip_rate"] <= r["ci95_hi"] <= 1)

    print("== T7 bootstrap CI 丢弃未定义重抽样而不是顶替 ==")
    aa = np.asarray([True] * 10 + [False] * 10)
    bb = np.asarray([True] * 8 + [False] * 2 + [True] * 2 + [False] * 8)
    lo, hi, dropped = bootstrap_ci(
        lambda idx: cohens_kappa(aa[idx], bb[idx]), n_items=20, n_boot=500)
    ck("bootstrap_returns_ci", lo is not None and hi is not None and lo <= hi,
       f"({lo},{hi}) dropped={dropped}")
    # 全同类 → 每次重抽样都未定义 → 应返回 None 而不是编一个区间
    cc = np.asarray([True] * 20)
    lo2, hi2, dr2 = bootstrap_ci(
        lambda idx: cohens_kappa(cc[idx], cc[idx]), n_items=20, n_boot=200)
    ck("bootstrap_all_undefined_returns_none",
       lo2 is None and hi2 is None and dr2 == 200, f"({lo2},{hi2}) dropped={dr2}")

    print("== T8 仲裁者构造:权重不可用时必须抛,不许静默降级 ==")
    probe = probe_available_arbiters()
    print(f"  探测结果: {probe['verdict']}")
    if not probe["clip_arbiters_available"]:
        _raised = False
        try:
            build_arbiters()
        except KappaUnavailable:
            _raised = True
        ck("build_raises_without_weights", _raised,
           "无权重却静默返回了合成仲裁者 —— 那会产出一个漂亮但无意义的 κ")
        syn = build_arbiters(allow_synthetic_only=True)
        ck("synthetic_only_is_explicit", len(syn) >= 2
           and all(s.kind == "synthetic" for s in syn))
        # 合成仲裁者跑得通,且 κ 可算(验证端到端接线)
        frames = [np.full((8, 8, 3), i * 7 % 251, dtype=np.uint8) for i in range(9)]
        votes = [s.score_fn(frames, "water") for s in syn]
        ck("synthetic_arbiters_produce_votes",
           all(len(v) == 9 and all(isinstance(x, bool) for x in v) for v in votes))
        mat = np.array([[sum(1 for v in votes if not v[i]),
                         sum(1 for v in votes if v[i])] for i in range(9)], dtype=float)
        try:
            kf2 = fleiss_kappa(mat)
            ck("synthetic_fleiss_computable", np.isfinite(kf2), f"κ={kf2:.4f}")
        except KappaUnavailable as e:
            ck("synthetic_fleiss_computable", False, str(e)[:60])
    else:
        ck("build_returns_clip_arbiters", len(build_arbiters()) >= 2)

    print(f"\n汇总: {len(P)} PASS / {len(F)} FAIL")
    if F:
        print("结果: FAIL —— " + ", ".join(F))
        return 4
    print("结果: 全绿 ✓(κ/Wilson 与闭式吻合;未定义与不达标均 fail-closed)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="语义门的多仲裁者共识与一致性统计(零真人参与)")
    ap.add_argument("--self-test", action="store_true",
                    help="统计量对闭式/教科书例子的核对 + fail-closed 故障注入")
    ap.add_argument("--probe", action="store_true",
                    help="如实报告本机能构造哪些仲裁者")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if args.probe:
        import json
        print(json.dumps(probe_available_arbiters(), indent=2, ensure_ascii=False))
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
