#!/usr/bin/env python3
# =============================================================================
# instrument_floor_report.py — 方向指标的**仪器标定报告**(纯 CPU,可零数据自跑)
#
# 为什么需要这个脚本(这是本 benchmark 最伤的科学问题):
#   方向可控性用 Farnebäck 光流估计运动主方向。这条量尺的两端都不是 0 和 180:
#     * 上端  = 随机猜测的期望 90°(折叠角误差在 [0,180],随机 → E=90°);
#     * 下端  = **地板**,由两部分叠加而成:
#         (a) 四方向 prompt 的**量化地板**:轨迹净方向是连续角,prompt 只说
#             east/north/west/south,量化残差的期望 |offset| = 22.5°(见 §1);
#         (b) **仪器噪声**:光流估计器即使输入完美 GT 视频也测不准,实测 ~25°。
#   于是可用动态范围只有 ~25°–90°。论文里 "77° vs 92°" 这类差落在这个窄带里,
#   审稿人一定会问:**这个差可分辨吗?置信区间多宽?**
#
#   本脚本不回避这个问题,而是把它量化:测地板、给 CI、把仪器噪声与内容噪声
#   分开、并输出"最小可检测差异(MDE)vs 样本量"的表,让读者自己判断哪些差
#   有意义、哪些只是噪声。**结论可能是"我们报的差不可分辨"—— 那也照报。**
#
# 五个部分(与 §编号对应,markdown 可直接贴论文附录):
#   §1 量化地板的解析值 :闭式推导 E|offset| = 22.5° + 蒙特卡洛校核
#                        + (有 H5 时)直接从 waypoints 算**实测**量化残差,
#                          不依赖"净角均匀"这个假设。
#   §2 合成对照         :已知平移量的合成视频(纯位移、无内容变化)→
#                        隔离出**估计器本身**的误差,与 GT 内容导致的误差分开。
#                        扫位移量 × 纹理类型 × 方向角,顺带查系统性偏差。
#   §3 GT 地板          :真实 GT 视频上的角误差分布 + bootstrap CI + 分位数,
#                        并按 "量化 / 仪器 / 内容" 做误差预算分解。
#   §4 分辨力           :MDE vs 样本量表 + (δ, N) 可分辨性矩阵,
#                        分析式与**模拟功效**互相校核(见下面的检验选择说明)。
#   §5 估计器诊断       :已知/新发现的系统性偏差如实列出。
#
# ---------------------------------------------------------------------------
# 检验的选择(§4 用哪个检验,为什么)
# ---------------------------------------------------------------------------
#   * **主检验 = 配对符号翻转置换检验(paired sign-flip permutation)。**
#     - 为什么配对:各臂在**同一批 val 轨迹**上评测(同 GT、同 prompt),
#       逐样本可按 sample_id 对齐。轨迹难度是共享的强混杂,配对能把它消掉;
#       用非配对检验等于白扔功效,且 σ 用错。
#     - 为什么置换而不是 t:折叠角误差有界 [0,180] 且典型是**双峰**的
#       (贴地板的成功样本 + 接近均匀的失败样本),正态性靠不住;
#       置换检验在"同一轨迹上两臂标签可交换"这个尖锐零假设下是**精确**的,
#       不需要任何分布假设。
#     - 检验统计量取差值均值(与论文报的 mean 角误差同口径);
#       p = (1 + #{|mean(d*)| ≥ |mean(d)|}) / (B + 1)(加一修正,不会给出 p=0)。
#   * **配对 t** 作为交叉校核报出(n 较大时两者应当接近;明显分歧 = 分布病态)。
#   * **MDE 的闭式**用正态近似:MDE = (z_{1-α/2} + z_{1-β}) · σ_d / √n,
#     σ_d = 逐轨迹差值的标准差 = σ·√(2(1−ρ))。**必须含 z_{1-β}**:
#     只用 z_{1-α/2}(= 1.96·σ_d/√n)给出的是"恰好显著的边界",在真值等于该
#     边界时功效只有 50% —— 那是"检出阈",不是通常意义的 MDE。本脚本两列都报,
#     以便与 server_run/metric_sample_size_study.py 的 1.96·√2·s/√n 对上:
#     那一列 = **非配对、50% 功效**的检出阈,是本表 ρ=0 且 power=0.5 的特例。
#   * 闭式不被信任 —— §4 用一个物理动机的生成模型(见 _mixture_arms)**模拟**
#     置换检验的实际功效,与闭式对照。两者若分歧,以模拟为准并大声报告。
#
# 用法:
#   # 零数据自检(统计与估计器的正确性验证,CI 覆盖率 + 功效校核)
#   python server_run/instrument_floor_report.py --self-test
#
#   # 纯合成标定(没有 GT 库时;仍产出可用的 §1/§2/§4)
#   python server_run/instrument_floor_report.py -o $WORK/instrument_floor.json
#
#   # 完整标定(有 GT 库;推荐)
#   python server_run/instrument_floor_report.py \
#       --gt-h5 $WORK/data/train_semantic.h5 \
#       --split $WORK/train_main/train_val_split.json \
#       --eval-dirs textonly=$WORK/eval_r16_sem_s42 pluecker=$WORK/eval_r16_geo_s42 \
#       -o $WORK/instrument_floor.json
#
# 口径(与 evaluate.py:646 的 _ANGKW 逐字一致,勿改):
#   flow_size = H5 metadata['pose_window_size'](canonical 128x222)、step = 1。
#   全分辨率(480x832)跑光流是已知的量程事故(GT 地板会读出 ~82°),这里硬拒。
# =============================================================================
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "lora_remotesense" / "new"))

# 逐字复用评估管线的估计器 —— **绝不在这里重实现**。标定的对象必须是论文里
# 真正跑的那份代码,自己抄一份就等于标定了一个不存在的仪器。
from metrics import estimate_motion_direction, compute_angular_error  # noqa: E402
import metrics as _metrics_mod  # [P0#2] 读 MIN_MEAN_FLOW_PX 以判定门的种类

CANONICAL_WINDOW = (128, 222)   # build_dataset_from_benchmark.py canonical 窗口
RANDOM_EXPECT_DEG = 90.0        # 折叠角误差 ~ U(0,180) → 均值 90°
QUANT_BINS = 4                  # prompt 只有 E/N/W/S 四个方向词
# 窗口分辨率下 GT 内容的逐帧位移量级(metrics.py:276-278 的服务器标定值)。
# 没有 GT 库时,"仪器项"必须在这个工作点上取 —— 在 1 px/帧对上标定出来的
# 0.04° 是漂亮但无意义的数字,benchmark 根本不在那个档位运行。
CANONICAL_GT_PX_PER_PAIR = 9.0
N_EVAL_HEADLINE = 400           # common.sh:123(headline 臂评估条数)
N_EVAL_ABLATION = 100           # common.sh:124(消融/诊断臂评估条数)


# ============================================================================
# 基础统计(不依赖 scipy —— 缺 scipy 时静默换算法会让 CI 口径悄悄变)
# ============================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """标准正态分位数(Acklam 有理逼近 + 一步 Halley 修正,|误差| < 1e-12)。

    自带实现而不是 `from scipy.stats import norm`:MDE 的 z 值是本报告所有
    数字的公共因子,它绝不能因为环境里没装 scipy 就换成别的近似 —— 那种
    降级不会报错,只会让表里的数悄悄变。
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"_norm_ppf 需要 0<p<1,得到 {p!r}")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
             ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    else:
        q, r = p - 0.5, (p - 0.5) ** 2
        x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    e = _norm_cdf(x) - p
    u = e * math.sqrt(2 * math.pi) * math.exp(x * x / 2)
    return x - u / (1 + x * u / 2)


def _clean(x) -> np.ndarray:
    a = np.asarray([v for v in np.ravel(np.asarray(x, dtype=object))
                    if v is not None], dtype=float)
    return a[np.isfinite(a)] if a.size else a.astype(float)


def bootstrap_mean_ci(x, *, n_boot: int = 5000, alpha: float = 0.05,
                      seed: int = 0, method: str = "bca") -> Dict:
    """均值的 bootstrap CI。默认 BCa。

    为什么用 BCa 而不是分位数法:角误差有界 [0,180] 且右偏(少数样本贴 180),
    小样本下均值的抽样分布是偏的,分位数法的覆盖率会系统性偏低。BCa 用
    偏差校正 z0 + 刀切加速度 a 修正这两项。样本量足够时两者几乎重合,
    脚本把两种结果都报出来,便于核对。
    """
    v = _clean(x)
    out = {"n": int(v.size), "mean": None, "sd": None,
           "ci_lo": None, "ci_hi": None, "method": method,
           "ci_percentile": [None, None]}
    if v.size == 0:
        return out
    out["mean"] = float(v.mean())
    out["sd"] = float(v.std(ddof=1)) if v.size > 1 else 0.0
    if v.size < 3:
        out["ci_lo"] = out["ci_hi"] = float(v.mean())
        out["method"] = "degenerate(n<3)"
        out["ci_percentile"] = [float(v.mean()), float(v.mean())]
        return out
    rng = np.random.default_rng(seed)
    boots = v[rng.integers(0, v.size, size=(int(n_boot), v.size))].mean(axis=1)
    lo_p, hi_p = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    out["ci_percentile"] = [float(lo_p), float(hi_p)]
    if method != "bca":
        out["ci_lo"], out["ci_hi"] = float(lo_p), float(hi_p)
        return out
    theta = v.mean()
    frac = float((boots < theta).mean())
    frac = min(max(frac, 1.0 / (2 * n_boot)), 1.0 - 1.0 / (2 * n_boot))
    z0 = _norm_ppf(frac)
    s = v.sum()
    jack = (s - v) / (v.size - 1)           # 均值的刀切值,闭式,无需循环
    jbar = jack.mean()
    num = float(((jbar - jack) ** 3).sum())
    den = 6.0 * float(((jbar - jack) ** 2).sum()) ** 1.5
    a_hat = 0.0 if den == 0 else num / den
    zl, zu = _norm_ppf(alpha / 2), _norm_ppf(1 - alpha / 2)
    q = []
    for z in (zl, zu):
        denom = 1 - a_hat * (z0 + z)
        adj = z0 + (z0 + z) / denom if abs(denom) > 1e-12 else z0 + z
        q.append(100.0 * min(max(_norm_cdf(adj), 1e-6), 1 - 1e-6))
    lo, hi = np.percentile(boots, q)
    out["ci_lo"], out["ci_hi"] = float(lo), float(hi)
    out["bca_z0"], out["bca_a"] = float(z0), float(a_hat)
    return out


def quantile_block(x) -> Dict:
    v = _clean(x)
    if v.size == 0:
        return {"n": 0}
    qs = [5, 10, 25, 50, 75, 90, 95]
    return {"n": int(v.size), "mean": float(v.mean()),
            "median": float(np.median(v)),
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            **{f"p{q}": float(np.percentile(v, q)) for q in qs},
            "min": float(v.min()), "max": float(v.max())}


def paired_permutation_test(d, *, n_perm: int = 4999, seed: int = 0) -> Dict:
    """配对符号翻转置换检验(见文件头"检验的选择")。

    零假设:同一轨迹上两臂标签可交换 ⇒ 差值 d_i 关于 0 对称 ⇒ 翻符号同分布。
    统计量 = mean(d),双侧。加一修正保证 p > 0(B 次重抽样给不出 p=0 的证据)。
    """
    v = _clean(d)
    if v.size < 2:
        return {"n": int(v.size), "mean_diff": None, "p_perm": None}
    obs = float(v.mean())
    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(int(n_perm), v.size)) * 2 - 1
    null = (signs * v).mean(axis=1)
    p = (1.0 + float((np.abs(null) >= abs(obs) - 1e-15).sum())) / (n_perm + 1.0)
    return {"n": int(v.size), "mean_diff": obs,
            "sd_diff": float(v.std(ddof=1)),
            "p_perm": float(min(p, 1.0)), "n_perm": int(n_perm)}


def paired_t_test(d) -> Dict:
    """配对 t(交叉校核用)。scipy 缺席时退回自算 t 统计量 + 正态近似 p。"""
    v = _clean(d)
    if v.size < 2:
        return {"t": None, "p": None}
    sd = float(v.std(ddof=1))
    if sd == 0:
        return {"t": None, "p": None, "note": "差值恒等,无方差"}
    t = float(v.mean() / (sd / math.sqrt(v.size)))
    try:
        from scipy import stats as sps
        p = float(sps.t.sf(abs(t), df=v.size - 1) * 2)
        src = "scipy.stats.t"
    except Exception:
        p = float(2 * (1 - _norm_cdf(abs(t))))
        src = "正态近似(无 scipy)"
    return {"t": t, "p": p, "df": int(v.size - 1), "p_source": src}


def mde_analytic(sigma_d: float, n: int, *, alpha: float = 0.05,
                 power: float = 0.8) -> float:
    """配对检验的最小可检测差异(正态近似)。

    MDE = (z_{1-α/2} + z_{1-β}) · σ_d / √n。
    σ_d 是**逐轨迹差值**的标准差;若两臂角误差的相关系数为 ρ、各自标准差 σ,
    则 σ_d = σ·√(2(1−ρ))。ρ>0 时配对严格优于非配对(ρ=0 时退化为 √2·σ)。
    """
    if n <= 0 or not np.isfinite(sigma_d):
        return float("nan")
    return (_norm_ppf(1 - alpha / 2) + _norm_ppf(power)) * sigma_d / math.sqrt(n)


def detection_threshold(sigma_d: float, n: int, *, alpha: float = 0.05) -> float:
    """"恰好显著"的边界 = z_{1-α/2}·σ_d/√n(即 power=50% 的 MDE)。

    单独列出来是因为它常被误当成 MDE:真差刚好等于它时,只有一半的重复实验
    会拿到显著结果。metric_sample_size_study.py 的 1.96·√2·s/√n 就是本量在
    ρ=0(非配对)下的取值 —— 两个脚本的数由此可以对上。
    """
    if n <= 0 or not np.isfinite(sigma_d):
        return float("nan")
    return _norm_ppf(1 - alpha / 2) * sigma_d / math.sqrt(n)


# ============================================================================
# §1 量化地板的解析值
# ============================================================================

def analytic_quantization_floor(*, n_mc: int = 200000, seed: int = 0,
                                empirical_angles: Optional[np.ndarray] = None
                                ) -> Dict:
    """四方向词 → |偏移| 的期望 22.5°:闭式推导 + 蒙特卡洛 + 实测。

    推导:
      建库器 (build_dataset_from_benchmark.py:_make_prompt) 用净位移角
      θ = atan2(-(cy_end-cy_start), cx_end-cx_start) mod 360 分箱:
          θ∈[315,360)∪[0,45) → east(代表角 0°)
          θ∈[45,135)         → north(90°)
          θ∈[135,225)        → west(180°)
          θ∈[225,315)        → south(270°)
      指标的目标角 = 代表角 + 180°(内容流方向,见 evaluate.py:156-170),
      而完美估计器在完美输入上读到的是 θ + 180°。折叠角误差因此是
          e = |fold(θ − D(θ))| = |((θ + 45) mod 90) − 45|
      箱宽 90°,θ 在箱内均匀 ⇒ 偏移 ~ U(−45°, 45°) ⇒ |偏移| ~ U(0°, 45°):
          E[e] = 22.5°   median = 22.5°   sd = 45/√12 = 12.99°   max = 45°
      **这一项与模型无关**:即使生成完全忠实于 GT,方向表也只能报 22.5°。

    重要例外:--angle-quantize 4 建的判门探针库里 θ 本身就只取 {0,90,180,270},
    量化残差恒为 0 —— 判门的地板是**纯仪器噪声**,不含这 22.5°。两库的地板
    不可混用,论文里也不能拿判门的地板去解释主表。

    empirical_angles 给定时(从 H5 waypoints 直接算的净角),额外报**实测**
    量化残差 —— 它不依赖"净角均匀"这个假设(建库器会把越界的 waypoint
    clamp 到场景边界,真实角分布并不严格均匀)。
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 360, size=int(n_mc))
    off = ((theta + 45.0) % 90.0) - 45.0
    mc = np.abs(off)
    out = {
        "derivation": (
            "θ~净位移角;prompt 按 45°/135°/225°/315° 边界量化到 E/N/W/S "
            "(代表角 0/90/180/270);目标角与估计角同时 +180° 抵消 ⇒ "
            "e=|((θ+45) mod 90)−45|;θ 箱内均匀 ⇒ |偏移|~U(0,45°)"),
        "closed_form": {"mean": 22.5, "median": 22.5,
                        "std": float(45.0 / math.sqrt(12.0)),
                        "p95": 42.75, "max": 45.0,
                        "bin_width_deg": 360.0 / QUANT_BINS},
        "monte_carlo_uniform_theta": {
            "n": int(n_mc), "mean": float(mc.mean()),
            "median": float(np.median(mc)), "std": float(mc.std(ddof=1)),
            "p95": float(np.percentile(mc, 95))},
        "probe_gate_note": (
            "--angle-quantize 4 的判门探针库 θ∈{0,90,180,270},量化残差恒 0;"
            "其地板 = 纯仪器噪声,不含 22.5°,不得与主库地板混用"),
    }
    out["closed_form_vs_mc_abs_diff"] = abs(
        out["closed_form"]["mean"] - out["monte_carlo_uniform_theta"]["mean"])
    if empirical_angles is not None and len(empirical_angles):
        ea = np.asarray(empirical_angles, float)
        eoff = np.abs(((ea + 45.0) % 90.0) - 45.0)
        out["empirical_from_waypoints"] = quantile_block(eoff)
        out["empirical_vs_closed_form_delta"] = float(
            eoff.mean() - 22.5)
        out["empirical_note"] = (
            "直接由 H5 waypoints 净角算的量化残差,不假设 θ 均匀;"
            "与 22.5° 的差 = 建库 clamp/拒绝采样带来的角分布非均匀性")
    return out


# ============================================================================
# §2 合成对照:已知平移量的理想视频
# ============================================================================

def _make_plate(h: int, w: int, kind: str, rng) -> np.ndarray:
    """造一块 (h,w,3) uint8 纹理板。三种纹理对应三类失效机理:

      smooth       低通随机场,各向同性 —— 光流的**理想工况**,给出估计器下限;
      stripes      单向正弦光栅 —— **孔径问题**:法向可测、切向不可测,
                   估计方向会被拉向梯度方向。遥感里的农田垄沟/道路/海岸线
                   就是这种结构,所以这不是人造的刁难;
      sparse       大片平坦 + 少量斑点 —— 低纹理(水面/滩涂),
                   光流幅度小。
    """
    if kind == "smooth":
        ch, cw = max(8, h // 12), max(8, w // 12)
        import cv2
        base = cv2.resize(rng.random((ch, cw)).astype(np.float32), (w, h),
                          interpolation=cv2.INTER_CUBIC)
    elif kind == "stripes":
        import cv2
        yy = np.arange(h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
        # 周期取 24 px 而不是 9 px:周期性图案在位移超过半周期后会发生**时间混叠**
        # (相位绕圈),那是另一种失效机理。用长周期把混叠推到 12 px 以上,
        # 让可靠位移带(≤~16 px)内测到的超额误差干净地归因于孔径问题。
        base = 0.5 + 0.5 * np.sin(2 * np.pi * yy / 24.0)    # 水平条纹,梯度沿 y
        ch, cw = max(8, h // 12), max(8, w // 12)
        jitter = cv2.resize(rng.random((ch, cw)).astype(np.float32), (w, h),
                            interpolation=cv2.INTER_CUBIC)
        base = 0.92 * base + 0.08 * jitter                  # 留一点各向同性成分
    elif kind == "sparse":
        import cv2
        base = np.zeros((h, w), np.float32)
        n_blob = max(4, (h * w) // 6000)
        ys = rng.integers(0, h, n_blob)
        xs = rng.integers(0, w, n_blob)
        base[ys, xs] = 1.0
        base = cv2.GaussianBlur(base, (0, 0), 3.0)
    else:
        raise ValueError(f"未知纹理类型 {kind!r}")
    lo, hi = float(base.min()), float(base.max())
    base = (base - lo) / (hi - lo + 1e-8) * 255.0
    return np.repeat(base.astype(np.uint8)[..., None], 3, axis=2)


def synth_translation_video(theta_deg: float, px_per_frame: float, T: int,
                            win: Tuple[int, int], kind: str, rng,
                            noise_sigma: float = 0.0) -> np.ndarray:
    """内容沿 θ(数学系,y 向上)匀速平移的 (T,h,w,3) uint8 视频。

    用 warpAffine 做**亚像素**裁切,而不是整数索引:整数取整会把位移量化到
    1 px,在 1-2 px/pair 的小位移档位上,量化残差本身就有 ~15° 量级的角度
    影响 —— 那会把"估计器在小位移下的表现"测成取整误差。
    """
    import cv2
    h, w = int(win[0]), int(win[1])
    pad = int(math.ceil(abs(px_per_frame) * (T - 1))) + 16
    plate = _make_plate(h + 2 * pad, w + 2 * pad, kind, rng)
    dx = math.cos(math.radians(theta_deg)) * px_per_frame
    dy = -math.sin(math.radians(theta_deg)) * px_per_frame   # 像素 y 向下
    frames = []
    for t in range(T):
        # 裁切窗原点朝 -(dx,dy) 走 ⇒ 窗内内容朝 +(dx,dy) 走 = 沿 θ 运动
        x0 = pad - dx * t
        y0 = pad - dy * t
        M = np.float32([[1, 0, -x0], [0, 1, -y0]])
        f = cv2.warpAffine(plate, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        if noise_sigma > 0:
            f = np.clip(f.astype(np.float32)
                        + rng.normal(0, noise_sigma, f.shape), 0, 255)
        frames.append(f.astype(np.uint8))
    return np.stack(frames)


def _signed_err(est: float, tgt: float) -> float:
    """带符号角误差,折到 (-180, 180]。查系统性偏差必须看符号,不能只看 |e|。"""
    return float(((est - tgt + 180.0) % 360.0) - 180.0)


def synthetic_control(*, win: Tuple[int, int], displacements: Sequence[float],
                      textures: Sequence[str], angles: Sequence[float],
                      T: int, seed: int, noise_sigma: float = 0.0) -> Dict:
    """§2 主体:扫 位移 × 纹理 × 方向角,逐条量估计器的恢复误差。

    这里的输入是**纯位移、无内容变化**的理想视频:窗内每个像素严格按同一
    向量平移。因此测到的任何误差都只能归给**估计器本身**(算法 + 量程 +
    孔径),不含 GT 视频的内容变化、也不含 prompt 量化。这正是把
    "仪器噪声" 与 "GT 内容导致的误差" 分开的手术刀。
    """
    rng_master = np.random.default_rng(seed)
    cells, rows = {}, []
    for kind in textures:
        for px in displacements:
            abs_errs, sgn_errs, n_none = [], [], 0
            per_angle = {}
            for th in angles:
                rng = np.random.default_rng(
                    int(rng_master.integers(0, 2 ** 31 - 1)))
                vid = synth_translation_video(th, px, T, win, kind, rng,
                                              noise_sigma=noise_sigma)
                # 口径与 evaluate.py 一致:step=1;视频已是窗口分辨率,
                # 故不再传 flow_size(传了也不会 resize,只是多一次宽高比校验)。
                est = estimate_motion_direction(vid, step=1)
                if est is None:
                    n_none += 1
                    per_angle[float(th)] = None
                    continue
                ae = compute_angular_error(est, float(th))
                se = _signed_err(est, float(th))
                abs_errs.append(ae)
                sgn_errs.append(se)
                per_angle[float(th)] = {"abs": ae, "signed": se}
            key = f"{kind}@{px:g}px"
            n_try = len(angles)
            cells[key] = {
                "texture": kind, "px_per_pair": float(px), "frames": int(T),
                "n_tried": n_try, "n_valid": n_try - n_none,
                "valid_pct": 100.0 * (n_try - n_none) / max(1, n_try),
                "abs_error": quantile_block(abs_errs),
                "signed_error_mean": (float(np.mean(sgn_errs))
                                      if sgn_errs else None),
                "signed_error_sd": (float(np.std(sgn_errs, ddof=1))
                                    if len(sgn_errs) > 1 else None),
                "per_angle": per_angle,
            }
            rows.append(key)
    return {"_window": [int(win[0]), int(win[1])],
            "_frames": int(T), "_noise_sigma": float(noise_sigma),
            "_angles_deg": [float(a) for a in angles],
            "_caliber": "step=1, 窗口分辨率, 逐字调用 metrics.estimate_motion_direction",
            "cells": cells, "order": rows}


def estimator_diagnostics(control: Dict, *, win: Tuple[int, int],
                          seed: int) -> Dict:
    """§5:把 §2 扫出来的东西读成"估计器有哪些系统性偏差"。

    另加两个**结构性**探针,它们不是靠扫参数能发现的:
      D3 净位移退化:估计器把每对光流按幅度加权求**向量和**,再取该和的
         方向。往返/折返轨迹的向量和接近 0,方向就是噪声 —— 但守卫
         (metrics.py:336 `weight_total < 1.0`)看的是**幅度之和**而不是
         **净向量的模**,所以它不会拒答,而是返回一个几乎任意的角度且
         不给任何"不可靠"信号。这与 Track B 的 well-posed(净位移比)口径
         是同一件事,但 Track A 侧没有对应的门。
      D4 帧对采样恰好整除:默认 step=len//8,循环 range(0,T-step,step)。
         帧数契约 T∈{9,17,25,33} 下 (T-1)/step 恰为 8,不丢尾帧;
         但 T 若偏离该契约(如 T=20 → step=2,尾帧 19 被丢),会静默丢弃
         尾部运动。这是契约在保护指标,不是指标自己稳。
    """
    import cv2  # noqa: F401  (确保诊断段与 §2 同一依赖)
    cells = control["cells"]
    diag: Dict[str, Dict] = {}

    # ---- D1 量程:误差随每对位移的变化 ----
    smooth = {k: v for k, v in cells.items() if v["texture"] == "smooth"}
    reliable_max = None
    if smooth:
        pts = sorted(((v["px_per_pair"], v["abs_error"].get("mean"))
                      for v in smooth.values()
                      if v["abs_error"].get("n")), key=lambda z: z[0])
        # 取**单调**可靠带(从最小位移起连续满足 ≤5°)而不是"所有 ≤5° 档的最大值":
        # 后者会被超量程区偶然回落的读数骗到 —— 24 px 的均值 128° 与 34 px 的 75°
        # 就是这种非单调噪声,不能因为某个大档位碰巧好看就把量程上限推上去。
        for p, e in pts:
            if e is None or e > 5.0:
                break
            reliable_max = p
        diag["D1_dynamic_range"] = {
            "curve_px_to_mean_abs_err": [[float(p), float(e)] for p, e in pts
                                         if e is not None],
            "reliable_px_per_pair_max": reliable_max,
            "criterion": "自最小位移起连续满足 平均|误差| ≤ 5°",
            # 若可靠带上界就是扫描格的最大值,说明**扫描没有覆盖到失效点**,
            # 这个"上限"是被扫描范围截断的,不能当作估计器的真实量程上界。
            "upper_bound_saturated": bool(
                pts and reliable_max is not None
                and abs(reliable_max - pts[-1][0]) < 1e-9),
            "verdict": (
                f"平滑纹理下,每对位移 ≤ {reliable_max:g} px 时平均角误差 ≤5°;"
                + ("而该值恰是扫描格上界 —— 真实量程上限未被本次扫描界定,"
                   "请用更大的 --displacements 重跑"
                   if pts and abs(reliable_max - pts[-1][0]) < 1e-9 else
                   "超过后迅速劣化(Farnebäck winsize=15/levels=3 的量程上限)")
                if reliable_max else "未找到 ≤5° 的位移档 —— 估计器在本配置下全程不可用"),
            "why_it_matters": (
                "480x832 生成分辨率上 GT 内容位移 ~34 px/帧,直接跑光流必然超量程 "
                "(已知事故读数 82°)。所以官方口径强制先降到窗口分辨率 128x222 "
                "且 step=1 —— 这不是调参,是把仪器放回它的量程内。"),
        }

    # ---- D2 孔径:各向异性纹理的方向拉偏 ----
    stripes = {k: v for k, v in cells.items() if v["texture"] == "stripes"}
    if stripes and smooth:
        def _mean_at(d, px):
            for v in d.values():
                if abs(v["px_per_pair"] - px) < 1e-9:
                    return v["abs_error"].get("mean"), v["valid_pct"]
            return None, None
        pxs = sorted({v["px_per_pair"] for v in smooth.values()}
                     & {v["px_per_pair"] for v in stripes.values()})
        rows = []
        for px in pxs:
            ms, _ = _mean_at(smooth, px)
            mt, vt = _mean_at(stripes, px)
            if ms is None or mt is None:
                continue
            rows.append({"px": float(px), "smooth_mean": float(ms),
                         "stripes_mean": float(mt),
                         "excess_deg": float(mt - ms),
                         "in_range": bool(reliable_max is not None
                                          and px <= reliable_max),
                         "stripes_valid_pct": vt})
        # **只在量程内比**:超量程档位上两种纹理都是垃圾读数,它们的差反映的是
        # 两堆噪声之差,不是孔径效应。旧写法取全表最大值会把 @34px 的 62° 当成
        # "孔径效应",那是把量程事故说成纹理问题。
        inr = [r for r in rows if r["in_range"]]
        worst = max(inr, key=lambda r: r["excess_deg"]) if inr else None
        diag["D2_aperture_anisotropy"] = {
            "rows": rows,
            "restricted_to_px_max": reliable_max,
            "worst_excess_deg": (worst["excess_deg"] if worst else None),
            "is_worst_case_bound": True,
            "verdict": (
                f"在量程内(≤{reliable_max:g} px/帧对),单向纹理相对各向同性纹理"
                f"最多多出 {worst['excess_deg']:.1f}° 误差(@{worst['px']:g} px)"
                f"—— 孔径问题真实存在,且与量程无关。这是**最坏情形上界**"
                f"(合成光栅 92% 能量集中在单一朝向),不是典型场景的期望值"
                if worst else "量程内无可比档位"),
            "why_it_matters": (
                "农田垄沟、道路、海岸线都是强单向结构。这类场景的角误差里有一部分"
                "是纹理各向异性带来的,不是模型不听话 —— 分场景报角误差时必须"
                "把它当成场景级的仪器偏差,而不是模型能力差异。量级上它可以逼近"
                "整个可用动态范围(~67°),因此**跨场景比较角误差是不安全的**;"
                "臂间比较必须在同一场景内配对做。"),
        }

    # ---- D3 净位移退化:折返轨迹 ----
    rng = np.random.default_rng(seed + 977)
    h, w = int(win[0]), int(win[1])
    T = 17
    px = 4.0
    pad = int(px * (T + 4)) + 16
    plate = _make_plate(h + 2 * pad, w + 2 * pad, "smooth", rng)
    xs = []
    for t in range(T):
        # 前半段向东、后半段向西:净位移 ≈ 0,总位移 = (T-1)*px
        s = t if t <= (T - 1) // 2 else (T - 1) - t
        xs.append(pad - px * s)
    frames = []
    for x0 in xs:
        M = np.float32([[1, 0, -x0], [0, 1, -float(pad)]])
        frames.append(cv2.warpAffine(plate, M, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT))
    est_deg = estimate_motion_direction(np.stack(frames).astype(np.uint8), step=1)
    diag["D3_net_displacement_degeneracy"] = {
        "setup": f"T={T} 折返轨迹(先东 {px:g}px/帧 后西),净位移≈0,总位移={(T-1)*px:g}px",
        "estimator_returned_none": est_deg is None,
        "estimated_angle_deg": (None if est_deg is None else float(est_deg)),
        "verdict": (
            "估计器**未拒答**,对一个净方向无定义的视频照样报了一个角度 —— "
            "守卫用的是幅度之和而不是净向量的模,退化情形没有任何信号"
            if est_deg is not None else
            "估计器拒答(返回 None)—— 本条不构成问题"),
        "why_it_matters": (
            "K≥2 的折线轨迹净位移可以很小(polyline_net_ratio 低)。这类样本的"
            "角误差本质上是噪声,却会照常进 mean。Track B 已有 well-posed 门"
            "(net_displacement_ratio),Track A 侧目前没有对应的门 —— "
            "报角误差时应同时报 WellPosed_Ratio 的分布,或对低净位移比样本分层。"),
    }

    # ---- D4 帧对采样 ----
    d4 = {}
    for t in (9, 17, 25, 33, 20):
        step = max(1, t // 8)
        n_pairs = len(range(0, t - step, step))
        covered = 0 if n_pairs == 0 else (n_pairs - 1) * step + step
        d4[str(t)] = {"default_step": step, "n_pairs": n_pairs,
                      "last_frame_used": covered,
                      "tail_frames_dropped": int(t - 1 - covered)}
    diag["D4_pair_sampling"] = {
        "table": d4,
        "verdict": ("帧数契约 T≡1 (mod 4) ⇒ T∈{9,17,25,33} 时 (T-1)/step 恰为 8,"
                    "不丢尾帧;T=20 这类非契约帧数会静默丢弃尾帧"),
        "why_it_matters": ("官方口径 step=1 不受影响;但任何拿默认 step 的旧读数"
                           "(含 validate_direction_metric 的对照列)都依赖帧数契约成立。"),
    }

    # ---- D5 近静止拒答门实际上是死的 ----
    # metrics.py:336 的守卫是 `weight_total < 1.0`,而 weight_total 是**所有像素、
    # 所有帧对的光流幅度之和**。128x222 窗口 × 8 对 = 227328 个像素样本,阈值 1.0
    # 相当于平均幅度 4.4e-6 px —— 任何一点像素噪声都能越过它。于是"模型只是把
    # 输入图定住"这种退化生成不会被拒答,而是拿到一个纯噪声方向照常进 mean。
    rng5 = np.random.default_rng(seed + 613)
    src = _make_plate(h, w, "smooth", rng5)
    static = np.repeat(src[None], 9, axis=0)
    est_static = estimate_motion_direction(static, step=1)
    noisy = np.clip(static.astype(np.float32)
                    + rng5.normal(0, 1.0, static.shape), 0, 255).astype(np.uint8)
    est_noisy = estimate_motion_direction(noisy, step=1)
    n_px_samples = h * w * 8
    _mean_guard = getattr(_metrics_mod, "MIN_MEAN_FLOW_PX", None)
    diag["D5_near_static_guard"] = {
        "guard_source": ("metrics.py  `mean_mag < MIN_MEAN_FLOW_PX: return None`"
                         if _mean_guard is not None else
                         "metrics.py:336  `if weight_total < 1.0: return None`"),
        "guard_kind": ("mean_magnitude" if _mean_guard is not None else "sum_magnitude"),
        "mean_magnitude_threshold_px": (_mean_guard if _mean_guard is not None
                                        else 1.0 / n_px_samples),
        "pixel_samples_at_this_window": int(n_px_samples),
        "implied_mean_magnitude_threshold_px": (
            _mean_guard if _mean_guard is not None else 1.0 / n_px_samples),
        "identical_frames_returns_none": est_static is None,
        "noisy_static_sigma1_returns_none": est_noisy is None,
        "noisy_static_estimated_angle_deg": (None if est_noisy is None
                                             else float(est_noisy)),
        "verdict": (
            "逐字相同的帧会被拒答,但只要加 σ=1 灰阶的像素噪声就**不再拒答**,"
            f"估计器给出角度 {est_noisy:.1f}°(纯噪声方向)"
            if est_noisy is not None else
            ("两种静止输入都被拒答 —— 本条已由 P0#2 修复(门改为平均幅度阈值 "
             f"{_mean_guard} px/帧对)" if _mean_guard is not None else
             "两种静止输入都被拒答 —— 本条不构成问题")),
        "why_it_matters": (
            f"守卫阈值是幅度**之和**,它随像素数增长,在 {h}x{w} 窗口下等价于"
            f"平均幅度 {1.0 / n_px_samples:.1e} px —— 这不是一个有意义的运动门。"
            "persistence 类退化生成(diagnostic_baselines.persistence)因此会拿到"
            "一个随机方向而不是 None,把噪声当作 ~90° 的『随机方向』计入均值。"
            "正确做法是把门改成**平均幅度**阈值(如 ≥0.1 px/帧对),"
            "或与 Track B 的 flow_magnitude 一起分层报告。"),
    }
    return diag


# ============================================================================
# §3 GT 地板(真实数据;缺数据时整段标记不可用,绝不拿合成值冒充)
# ============================================================================

def _canonical_bin(theta: float) -> float:
    """净角 → 四方向代表角(与 build_dataset_from_benchmark._make_prompt 同边界)。"""
    a = float(theta) % 360.0
    if a < 45 or a >= 315:
        return 0.0
    if a < 135:
        return 90.0
    if a < 225:
        return 180.0
    return 270.0


def measure_gt_floor(gt_h5: str, split: Optional[str], *, max_samples: int,
                     n_boot: int, alpha: float, seed: int,
                     force_window: Optional[Tuple[int, int]]) -> Dict:
    """在 GT 视频上跑官方口径的方向估计 → "完美输入下的角误差分布"。

    同时算两个目标:
      * vs **量化目标**(prompt 方向词 +180°)= 论文表里那条口径的地板;
      * vs **连续目标**(waypoints 净角 +180°)= 扣掉量化后的地板。
    两者之差就是量化项的实际贡献 —— 这是误差预算分解的核心,
    比"22.5° 与 25° 谁大"的口头比较可靠得多。
    """
    import cv2  # noqa: F401
    import h5py
    out: Dict = {"gt_h5": str(gt_h5), "split": str(split) if split else None}
    with h5py.File(gt_h5, "r") as f:
        grp = f["data"] if "data" in f else (f["samples"] if "samples" in f else f)
        keys = sorted(grp.keys(), key=lambda k: int(k.split("_")[-1]))
        win = force_window
        if win is None and "metadata" in f and "pose_window_size" in f["metadata"].attrs:
            win = tuple(int(v) for v in f["metadata"].attrs["pose_window_size"])
        if win is None:
            # 缺 metadata 就用 canonical 值并大声记录 —— 绝不退回全分辨率,
            # 那是已知会读出 82° 垃圾的量程事故口径。
            win = CANONICAL_WINDOW
            out["_window_source"] = f"metadata 缺失 → canonical 默认 {win}"
        else:
            out["_window_source"] = "H5 metadata.pose_window_size" \
                if force_window is None else "命令行 --window-size 覆盖"
        out["_flow_window"] = [int(win[0]), int(win[1])]
        out["_caliber"] = "flow_size=窗口分辨率, step=1(= evaluate.py 的 _ANGKW)"

        if split and Path(split).is_file():
            idx = json.loads(Path(split).read_text())["val_indices"]
        else:
            idx = list(range(len(keys)))
            out["_split_note"] = "无 split 文件 → 用全部样本"
        if max_samples:
            idx = idx[:max_samples]

        err_quant, err_cont, quant_off = [], [], []
        net_ratio, flow_px, net_theta, n_none = [], [], [], 0
        per_scene: Dict[str, List[float]] = {}
        for i in idx:
            if i >= len(keys):
                continue
            s = grp[keys[i]]
            frames = s["video_frames"][:]
            wp = s["waypoints"][:] if "waypoints" in s else None
            if wp is None or len(wp) < 2:
                continue
            wp = np.asarray(wp, dtype=np.float64)
            theta_cam = float(np.degrees(np.arctan2(
                -(wp[-1, 0] - wp[0, 0]), wp[-1, 1] - wp[0, 1]))) % 360.0
            tgt_cont = (theta_cam + 180.0) % 360.0
            tgt_quant = (_canonical_bin(theta_cam) + 180.0) % 360.0
            net_theta.append(theta_cam)
            quant_off.append(abs(_signed_err(theta_cam,
                                             _canonical_bin(theta_cam))))
            seg = np.sqrt(((wp[1:] - wp[:-1]) ** 2).sum(axis=1)).sum()
            net = float(np.sqrt(((wp[-1] - wp[0]) ** 2).sum()))
            net_ratio.append(net / seg if seg > 0 else 0.0)

            est = estimate_motion_direction(frames, flow_size=win, step=1)
            if est is None:
                n_none += 1
                continue
            eq = compute_angular_error(est, tgt_quant)
            ec = compute_angular_error(est, tgt_cont)
            err_quant.append(eq)
            err_cont.append(ec)
            sc = s.attrs.get("source_scene_category", "")
            sc = sc.decode() if isinstance(sc, bytes) else str(sc or "?")
            per_scene.setdefault(sc or "?", []).append(eq)
            # 观测位移量:用来把 §2 的合成对照对准真实工作点
            fr = frames
            if fr.shape[1] > win[0] and fr.shape[2] > win[1]:
                fr = np.stack([cv2.resize(np.clip(x, 0, 255).astype(np.uint8),
                                          (win[1], win[0]),
                                          interpolation=cv2.INTER_AREA)
                               for x in fr])
            g1 = cv2.cvtColor(fr[0], cv2.COLOR_RGB2GRAY)
            g2 = cv2.cvtColor(fr[1], cv2.COLOR_RGB2GRAY)
            fl = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_px.append(float(np.sqrt(fl[..., 0] ** 2 + fl[..., 1] ** 2).mean()))

    n_tried = len(err_quant) + n_none
    out.update({
        "n_evaluated": len(err_quant), "n_estimator_none": n_none,
        "valid_pct": 100.0 * len(err_quant) / max(1, n_tried),
        "vs_quantized_target": {**quantile_block(err_quant),
                                **bootstrap_mean_ci(err_quant, n_boot=n_boot,
                                                    alpha=alpha, seed=seed)},
        "vs_continuous_target": {**quantile_block(err_cont),
                                 **bootstrap_mean_ci(err_cont, n_boot=n_boot,
                                                     alpha=alpha, seed=seed)},
        "quantization_offset_from_waypoints": quantile_block(quant_off),
        "observed_flow_px_per_pair": quantile_block(flow_px),
        "polyline_net_ratio": quantile_block(net_ratio),
        "per_scene_vs_quantized": {k: quantile_block(v)
                                   for k, v in sorted(per_scene.items())},
        # 逐样本原值必须落盘:§4 的功效模拟要从**真实地板分布**里重抽样,
        # 用 mean/sd 重建一个正态是假的 —— 角误差的双峰形状正是关键。
        "_per_sample_vs_quantized": [float(v) for v in err_quant],
        "_net_theta_deg": [float(t) for t in net_theta],
    })
    return out


def error_budget(gt: Dict, control: Dict, analytic: Dict) -> Dict:
    """误差预算:把 GT 地板拆成 量化 / 仪器 / 内容 三项。

    做法(不是把角度相加 —— 角度不满足可加性):
      * 量化项    = GT 上 vs量化目标 与 vs连续目标 的均值之差,
                    以及直接由 waypoints 算的 |offset| 均值(两者应当接近);
      * 仪器项    = §2 合成对照在**匹配到 GT 实测位移档**上的均值角误差;
      * 内容残差  = vs连续目标 的均值 − 仪器项。
                    它包含:窗口边缘不断进入的新内容、场景纹理各向异性、
                    低纹理区、折线轨迹的净/总位移比偏低等。
    三项都带出处,谁也没被"凑数"。
    """
    if not gt or not gt.get("n_evaluated"):
        return {"unavailable": "无 GT 数据,无法做预算分解"}
    obs_px = gt["observed_flow_px_per_pair"].get("median")
    cells = control["cells"]
    smooth = [v for v in cells.values() if v["texture"] == "smooth"
              and v["abs_error"].get("n")]
    inst = None
    if smooth and obs_px is not None:
        best = min(smooth, key=lambda v: abs(v["px_per_pair"] - obs_px))
        inst = {"px_per_pair_used": best["px_per_pair"],
                "gt_observed_px_per_pair_median": float(obs_px),
                "mean_abs_error": best["abs_error"]["mean"],
                "match_note": ("合成对照按 GT 实测位移中位数就近选档;"
                               "档位与实测位移偏差过大时该项只能当量级参考")}
    q_from_targets = (gt["vs_quantized_target"]["mean"]
                      - gt["vs_continuous_target"]["mean"])
    return {
        "total_floor_vs_quantized_target_mean": gt["vs_quantized_target"]["mean"],
        "term_quantization": {
            "from_target_difference_deg": float(q_from_targets),
            "from_waypoints_mean_deg":
                gt["quantization_offset_from_waypoints"].get("mean"),
            "closed_form_deg": analytic["closed_form"]["mean"],
            "note": ("前两者应当接近;若差很多,说明 vs连续目标 那一列的误差"
                     "已大到与量化项同量级,两项不再近似可分"),
        },
        "term_instrument": inst,
        "term_content_residual_deg": (
            None if inst is None else
            float(gt["vs_continuous_target"]["mean"] - inst["mean_abs_error"])),
        # 残差**可以为负**,那不是 bug:它说明合成对照的工作点比 GT 略苛刻
        # (档位就近取整 + 合成纹理未必比真实场景好跟踪)。负值应读作
        # "内容效应在测量精度内不可见",而不是"内容让误差变小了"。
        "term_content_residual_note": (
            "负值 = 内容效应小于合成对照与 GT 之间的档位失配,应读作『不可见』"
            if inst is not None
            and gt["vs_continuous_target"]["mean"] < inst["mean_abs_error"]
            else "非负,内容效应可分辨"),
        "usable_dynamic_range": {
            "floor_mean_deg": gt["vs_quantized_target"]["mean"],
            "floor_ci95": [gt["vs_quantized_target"]["ci_lo"],
                           gt["vs_quantized_target"]["ci_hi"]],
            "random_expectation_deg": RANDOM_EXPECT_DEG,
            "range_deg": float(RANDOM_EXPECT_DEG
                               - gt["vs_quantized_target"]["mean"]),
            "note": ("可用量程 = [地板, 90°]。落在该区间外的读数(如 >90°)"
                     "统计上与随机不可区分,不应被解读为『比随机更差』"),
        },
    }


# ============================================================================
# §4 分辨力:MDE vs 样本量 + 模拟功效校核
# ============================================================================

def _mixture_arms(n: int, p_fail_a: float, p_fail_b: float,
                  floor_pool: np.ndarray, rng) -> Tuple[np.ndarray, np.ndarray]:
    """两臂角误差的生成模型(用于模拟真实功效)。

    为什么不用 "d ~ N(δ, σ²)":那样模拟出来的功效等于闭式公式自己,
    是同义反复。真实的角误差是**混合分布**:
        以概率 (1−p_fail) 落在地板附近(从实测/合成地板池里抽),
        以概率 p_fail   方向完全失控 → 折叠误差 ~ U(0,180)。
    两臂的差异体现为 p_fail 不同。共享的隐变量 u_i(轨迹难度)让两臂
    在同一条轨迹上同时失败 —— 这自然生成了配对相关性 ρ,而不是硬塞一个 ρ。
    这也正是这个 benchmark 里"臂间差"的物理来源(有的臂更常整体跑飞)。
    """
    u = rng.random(n)                       # 轨迹难度,两臂共享 ⇒ 正相关
    fa, fb = u < p_fail_a, u < p_fail_b
    base_a = floor_pool[rng.integers(0, floor_pool.size, n)]
    base_b = floor_pool[rng.integers(0, floor_pool.size, n)]
    ea = np.where(fa, rng.uniform(0, 180, n), base_a)
    eb = np.where(fb, rng.uniform(0, 180, n), base_b)
    return ea, eb


def _p_fail_for_mean(target_mean: float, floor_mean: float) -> float:
    """混合模型里,给定目标均值反解 p_fail:mean = (1−p)·floor + p·90。"""
    denom = RANDOM_EXPECT_DEG - floor_mean
    if denom <= 0:
        return 0.0
    return float(min(max((target_mean - floor_mean) / denom, 0.0), 1.0))


def simulate_power(floor_pool: np.ndarray, *, base_mean: float, delta: float,
                   n: int, alpha: float, n_sim: int, n_perm: int,
                   seed: int) -> Dict:
    """在混合模型下模拟配对置换检验的实际功效。

    返回 power 及模拟出的 σ_d / ρ —— 后者用来验证闭式 MDE 用的 σ_d 是否合理。
    """
    rng = np.random.default_rng(seed)
    floor_mean = float(floor_pool.mean())
    pa = _p_fail_for_mean(base_mean, floor_mean)
    pb = _p_fail_for_mean(base_mean + delta, floor_mean)
    hits, sds, rhos, realized = 0, [], [], []
    for _ in range(int(n_sim)):
        ea, eb = _mixture_arms(n, pa, pb, floor_pool, rng)
        d = eb - ea
        sds.append(float(np.std(d, ddof=1)))
        if n > 2 and np.std(ea) > 0 and np.std(eb) > 0:
            rhos.append(float(np.corrcoef(ea, eb)[0, 1]))
        realized.append(float(d.mean()))
        obs = float(d.mean())
        signs = rng.integers(0, 2, size=(int(n_perm), n)) * 2 - 1
        null = (signs * d).mean(axis=1)
        p = (1.0 + float((np.abs(null) >= abs(obs) - 1e-15).sum())) / (n_perm + 1.0)
        hits += int(p <= alpha)
    return {"n": int(n), "delta_requested_deg": float(delta),
            "delta_realized_mean_deg": float(np.mean(realized)),
            "p_fail_a": pa, "p_fail_b": pb,
            "power": hits / float(n_sim), "n_sim": int(n_sim),
            "n_perm": int(n_perm),
            "sigma_d_simulated": float(np.mean(sds)),
            "rho_simulated": (float(np.mean(rhos)) if rhos else None)}


def resolution_analysis(*, floor_pool: np.ndarray, base_mean: float,
                        sizes: Sequence[int], deltas: Sequence[float],
                        alpha: float, power: float, rho_scenarios: Sequence[float],
                        real_pairs: Dict, n_sim: int, n_perm: int,
                        seed: int) -> Dict:
    """§4 主体:MDE 表 + 可分辨性矩阵 + 模拟校核。"""
    sigma_arm = float(np.std(floor_pool, ddof=1))
    # 臂的实际 σ 由混合决定,不等于地板池的 σ。用混合模型算解析 σ:
    #   Var = E[Var|z] + Var[E|z];成分:地板池(σ_f, μ_f)与 U(0,180)(σ=51.96, μ=90)
    pf = _p_fail_for_mean(base_mean, float(floor_pool.mean()))
    mu_f, var_f = float(floor_pool.mean()), float(np.var(floor_pool, ddof=1))
    mu_u, var_u = 90.0, 180.0 ** 2 / 12.0
    m = (1 - pf) * mu_f + pf * mu_u
    sigma_arm_mix = math.sqrt((1 - pf) * (var_f + mu_f ** 2)
                              + pf * (var_u + mu_u ** 2) - m ** 2)

    scen = {}
    for rho in rho_scenarios:
        sd = sigma_arm_mix * math.sqrt(max(2.0 * (1.0 - rho), 0.0))
        scen[f"paired_rho={rho:g}"] = {"sigma_arm": sigma_arm_mix,
                                       "rho": float(rho), "sigma_d": sd}
    scen["unpaired"] = {"sigma_arm": sigma_arm_mix, "rho": 0.0,
                        "sigma_d": sigma_arm_mix * math.sqrt(2.0),
                        "note": "非配对两样本比较;等价于 ρ=0 的配对"}
    for name, pr in real_pairs.items():
        if pr.get("sigma_d") is not None:
            scen[f"observed:{name}"] = {
                "sigma_arm": pr.get("sigma_pooled"), "rho": pr.get("rho"),
                "sigma_d": pr["sigma_d"],
                "note": "由真实 eval 目录的逐样本角误差直接算出(最可信)"}

    table = []
    for n in sizes:
        row = {"n": int(n)}
        for name, s in scen.items():
            row[name] = {
                "mde_power%d" % int(round(power * 100)):
                    mde_analytic(s["sigma_d"], n, alpha=alpha, power=power),
                "detection_threshold_power50":
                    detection_threshold(s["sigma_d"], n, alpha=alpha),
            }
        table.append(row)

    # 可分辨性矩阵:用最保守(σ_d 最大)与最乐观(最小)两个场景夹逼
    sds = {k: v["sigma_d"] for k, v in scen.items()
           if v["sigma_d"] and np.isfinite(v["sigma_d"])}
    k_cons = max(sds, key=lambda k: sds[k])
    k_opt = min(sds, key=lambda k: sds[k])
    matrix = []
    for delta in deltas:
        row = {"delta_deg": float(delta)}
        for n in sizes:
            mc = mde_analytic(sds[k_cons], n, alpha=alpha, power=power)
            mo = mde_analytic(sds[k_opt], n, alpha=alpha, power=power)
            row[str(n)] = ("可分辨" if delta >= mc else
                           "不可分辨" if delta < mo else "取决于臂间相关性")
        matrix.append(row)

    sims = []
    # 模拟档位必须覆盖**论文实际用的** N(common.sh 的 400 / 100),
    # 否则校核过的区间与报数字的区间对不上。
    _want = [n for n in (N_EVAL_ABLATION, N_EVAL_HEADLINE) if n in set(sizes)]
    sim_sizes = sorted(set(_want) | set(sorted(set(sizes))[:2]))
    sim_deltas = sorted(set(float(d) for d in deltas))[:3]
    for n in sim_sizes:
        for delta in sim_deltas:
            sims.append(simulate_power(floor_pool, base_mean=base_mean,
                                       delta=delta, n=int(n), alpha=alpha,
                                       n_sim=n_sim, n_perm=n_perm,
                                       seed=seed + 7 * int(n) + int(delta)))
    return {
        "_test": ("主检验 = 配对符号翻转置换检验(mean 差,双侧,加一修正);"
                  "配对 t 作交叉校核;MDE 用正态近似闭式"),
        "_why_paired": ("各臂在同一批 val 轨迹上评测,可按 sample_id 配对;"
                        "轨迹难度是共享混杂,配对消掉它才有功效"),
        "alpha": float(alpha), "power_target": float(power),
        "floor_pool": {"n": int(floor_pool.size),
                       "mean": float(floor_pool.mean()),
                       "sd": sigma_arm},
        "arm_sigma_under_mixture": sigma_arm_mix,
        "arm_base_mean_deg": float(base_mean),
        "implied_p_fail": pf,
        "scenarios": scen,
        "mde_table": table,
        "resolvability_matrix": {
            "conservative_scenario": k_cons, "optimistic_scenario": k_opt,
            "rows": matrix},
        "simulated_power": sims,
        "simulation_vs_analytic": _sim_vs_analytic(sims, alpha),
    }


def _sim_vs_analytic(sims: List[Dict], alpha: float) -> Dict:
    """把模拟功效与闭式对上。

    校核方式:拿**模拟里实际实现的** σ_d 与 δ 代进正态近似的功效公式
    power = Φ(δ/(σ_d/√n) − z_{1-α/2}),再与置换检验的实测功效比。
    两者一致 ⇒ MDE 表(同一套近似的逆解)可信;不一致 ⇒ 正态近似在该区域
    失效,应以模拟为准。**不能**拿"闭式 MDE 处功效=0.8"当验证,那是循环论证。
    """
    rows = []
    for s in sims:
        sd = s["sigma_d_simulated"]
        if not np.isfinite(sd) or sd <= 0:
            continue
        analytic_p = _norm_cdf(
            abs(s["delta_realized_mean_deg"]) / (sd / math.sqrt(s["n"]))
            - _norm_ppf(1 - alpha / 2))
        # 模拟功效本身是 n_sim 次伯努利试验的比例,自带蒙特卡洛误差
        # SE = √(p(1−p)/n_sim)。判"分歧"必须扣掉它,否则 --quick(n_sim=40,
        # SE 高达 0.079)会稳定地把一致判成分歧 —— 那是在报告模拟噪声。
        p = s["power"]
        mc_se = math.sqrt(max(p * (1 - p), 1e-6) / max(s["n_sim"], 1))
        rows.append({"n": s["n"], "delta": s["delta_requested_deg"],
                     "power_simulated": p,
                     "power_analytic": float(analytic_p),
                     "abs_gap": abs(p - analytic_p),
                     "mc_se": float(mc_se),
                     "gap_in_mc_se": float(abs(p - analytic_p) / max(mc_se, 1e-9)),
                     "rho_simulated": s["rho_simulated"]})
    if not rows:
        return {"rows": [], "max_abs_gap": None, "verdict": "无模拟数据"}
    gaps = [r["abs_gap"] for r in rows]
    z = [r["gap_in_mc_se"] for r in rows]
    tol = max(0.05, 3.0 * float(np.mean([r["mc_se"] for r in rows])))
    ok = max(gaps) <= tol
    return {"rows": rows,
            "max_abs_gap": float(max(gaps)),
            "max_gap_in_mc_se": float(max(z)),
            "tolerance": float(tol),
            "tolerance_note": ("容差 = max(0.05, 3×平均蒙特卡洛 SE);"
                               "n_sim 越小容差越宽,这是模拟精度的诚实反映"),
            "verdict": (
                f"闭式与模拟一致(最大功效差 {max(gaps):.3f} ≤ 容差 {tol:.3f}),MDE 表可信"
                if ok else
                f"闭式与模拟分歧(最大功效差 {max(gaps):.3f} > 容差 {tol:.3f})"
                f" —— 以模拟为准,正态近似在该区域不成立")}


# ============================================================================
# 真实 eval 目录:拿逐样本角误差算真实 σ_d / ρ
# ============================================================================

def load_arm_scores(eval_dir: str) -> Dict[str, float]:
    """读 evaluate.py 落的 scores.json 的逐样本 Angular_Error(按 sample_id)。"""
    p = Path(eval_dir) / "scores.json"
    if not p.is_file():
        return {}
    try:
        rows = json.loads(p.read_text()).get("per_sample") or []
    except Exception:
        return {}
    out = {}
    for r in rows:
        sid, e = r.get("sample_id"), r.get("Angular_Error")
        if sid is not None and e is not None and np.isfinite(float(e)):
            out[str(sid)] = float(e)
    return out


def compare_real_arms(eval_dirs: Sequence[str], *, n_boot: int, n_perm: int,
                      alpha: float, power: float, sizes: Sequence[int],
                      seed: int) -> Dict:
    """两两配对比较真实臂:mean 差 + bootstrap CI + 置换 p + 配对 t + 实际 σ_d/ρ。"""
    arms = {}
    for spec in eval_dirs:
        name, d = spec.split("=", 1)
        sc = load_arm_scores(d)
        if sc:
            arms[name] = sc
        else:
            arms.setdefault("_missing", {})[name] = d
    missing = arms.pop("_missing", {})
    out: Dict = {"arms": {k: {"n": len(v), "mean": float(np.mean(list(v.values())))}
                          for k, v in arms.items()},
                 "missing_scores_json": missing, "pairs": {}}
    names = sorted(arms)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            shared = sorted(set(arms[a]) & set(arms[b]))
            if len(shared) < 5:
                out["pairs"][f"{a}_vs_{b}"] = {
                    "skipped": f"共享样本只有 {len(shared)} 条"}
                continue
            ea = np.array([arms[a][k] for k in shared])
            eb = np.array([arms[b][k] for k in shared])
            d = eb - ea
            rho = float(np.corrcoef(ea, eb)[0, 1]) if len(shared) > 2 else None
            sd_d = float(np.std(d, ddof=1))
            sig_pool = float(np.sqrt((np.var(ea, ddof=1) + np.var(eb, ddof=1)) / 2))
            rec = {
                "n_paired": len(shared),
                "mean_a": float(ea.mean()), "mean_b": float(eb.mean()),
                "mean_diff_b_minus_a": float(d.mean()),
                "diff_ci95": bootstrap_mean_ci(d, n_boot=n_boot, alpha=alpha,
                                               seed=seed),
                "permutation": paired_permutation_test(d, n_perm=n_perm,
                                                       seed=seed),
                "paired_t": paired_t_test(d),
                "rho": rho, "sigma_d": sd_d, "sigma_pooled": sig_pool,
                "mde_at_this_n": mde_analytic(sd_d, len(shared), alpha=alpha,
                                              power=power),
                "mde_vs_sizes": {str(n): mde_analytic(sd_d, int(n), alpha=alpha,
                                                      power=power)
                                 for n in sizes},
            }
            rec["resolvable"] = bool(abs(rec["mean_diff_b_minus_a"])
                                     >= rec["mde_at_this_n"])
            rec["verdict"] = (
                f"差 {rec['mean_diff_b_minus_a']:+.2f}°,该 N 下 MDE="
                f"{rec['mde_at_this_n']:.2f}° ⇒ "
                + ("差值达到可检测量级" if rec["resolvable"]
                   else "**差值小于 MDE,不足以支撑臂间结论**"))
            out["pairs"][f"{a}_vs_{b}"] = rec
    return out


# ============================================================================
# 地板池的构造(没有 GT 时的诚实退路)
# ============================================================================

def compose_floor_pool(control: Dict, *, px_target: Optional[float],
                       n_mc: int, seed: int,
                       theta_pool: Optional[np.ndarray] = None) -> Dict:
    """把 §2 的**带符号**仪器误差与量化一起合成出"完美模型的角误差分布"。

    做法(角度不能直接相加,所以走真实的合成路径):
        θ  ~ 真实净角池(有 H5 时)或 U(0,360)
        est = θ + ε,ε 从合成对照的带符号误差池里抽(仪器噪声)
        tgt = D(θ)(四方向量化)
        e   = fold(est − tgt)
    得到的分布就是"量化 + 仪器"两项一起作用下的地板。没有 GT 时用它当
    §4 的 floor_pool;有 GT 时优先用实测地板,这个只作对照。
    """
    smooth = [v for v in control["cells"].values()
              if v["texture"] == "smooth" and v["abs_error"].get("n")]
    if not smooth:
        return {"unavailable": "合成对照无可用档位"}
    # px_target 缺失时**不能**取最小位移档:1 px/帧对上仪器误差是 0.04°,
    # 拿它当仪器项会把地板说得比实际低一个数量级。退回 canonical 工作点。
    px_used = CANONICAL_GT_PX_PER_PAIR if px_target is None else px_target
    cell = min(smooth, key=lambda v: abs(v["px_per_pair"] - px_used))
    eps = np.array([d["signed"] for d in cell["per_angle"].values()
                    if isinstance(d, dict)], dtype=float)
    if eps.size == 0:
        return {"unavailable": "该档位没有有效的带符号误差样本"}
    rng = np.random.default_rng(seed)
    theta = (rng.choice(theta_pool, size=int(n_mc), replace=True)
             if theta_pool is not None and len(theta_pool)
             else rng.uniform(0, 360, size=int(n_mc)))
    est = theta + rng.choice(eps, size=int(n_mc), replace=True)
    tgt = np.array([_canonical_bin(t) for t in theta])
    diff = np.abs((est - tgt + 180.0) % 360.0 - 180.0)
    return {"px_per_pair_used": cell["px_per_pair"],
            "px_target_requested": float(px_used),
            "px_target_source": ("GT 实测位移中位数" if px_target is not None
                                 else f"canonical 工作点 {CANONICAL_GT_PX_PER_PAIR:g} px"
                                      f"(metrics.py:276-278 的服务器标定)"),
            "instrument_signed_error_sd": float(eps.std(ddof=1))
            if eps.size > 1 else 0.0,
            "theta_source": ("H5 waypoints 实测净角" if theta_pool is not None
                             and len(theta_pool) else "U(0,360) 假设"),
            "pool": diff.tolist() if n_mc <= 2000 else None,
            "stats": quantile_block(diff),
            "_pool_array": diff}


# ============================================================================
# 自检:统计过程与估计器的正确性(不需要任何数据)
# ============================================================================

def self_test(*, quick: bool = False) -> int:
    """把本脚本自己的统计过程验一遍。失败 → 退出码 1。

    验的是"我算得对不对",不是"结果好不好看":
      T1 估计器在已知平移上恢复方向(合成视频的真值是构造出来的);
      T2 bootstrap CI 的覆盖率 ≈ 名义 95%;
      T3 配对置换检验在零假设下的 I 类错误率 ≈ α;
      T4 闭式 MDE 处的实际功效 ≈ 目标功效;
      T5 解析量化地板 22.5° 与蒙特卡洛一致。
    """
    ok = True
    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        ok = ok and bool(cond)

    print("== T1 估计器在已知平移上的方向恢复 ==")
    rng = np.random.default_rng(0)
    errs = []
    for th in (0, 45, 90, 135, 180, 225, 270, 315):
        v = synth_translation_video(th, 4.0, 9, CANONICAL_WINDOW, "smooth",
                                    np.random.default_rng(int(th) + 1))
        est = estimate_motion_direction(v, step=1)
        errs.append(None if est is None else compute_angular_error(est, float(th)))
    bad = [e for e in errs if e is None or e > 5.0]
    check("八方向恢复误差 ≤5°", not bad,
          f"errs={[None if e is None else round(e,2) for e in errs]}")

    print("== T2 bootstrap CI 覆盖率 ==")
    pop = np.abs(np.random.default_rng(1).uniform(-45, 45, 200000))  # 真均值 22.5
    true_mean = float(pop.mean())
    n_rep = 200 if quick else 500
    n_boot = 999 if quick else 1999
    cov_b = cov_p = 0
    r2 = np.random.default_rng(2)
    for i in range(n_rep):
        s = r2.choice(pop, size=60, replace=False)
        ci = bootstrap_mean_ci(s, n_boot=n_boot, alpha=0.05, seed=int(i))
        cov_b += int(ci["ci_lo"] <= true_mean <= ci["ci_hi"])
        cov_p += int(ci["ci_percentile"][0] <= true_mean <= ci["ci_percentile"][1])
    cb, cp = cov_b / n_rep, cov_p / n_rep
    check("BCa 覆盖率 ∈ [0.90, 0.99]", 0.90 <= cb <= 0.99, f"实测 {cb:.3f}")
    check("分位数法覆盖率 ∈ [0.88, 0.99]", 0.88 <= cp <= 0.99, f"实测 {cp:.3f}")

    print("== T3 配对置换检验的 I 类错误率 ==")
    r3 = np.random.default_rng(3)
    n_rep3 = 300 if quick else 800
    n_perm3 = 499 if quick else 999
    rej = 0
    for i in range(n_rep3):
        d = r3.normal(0, 10, 40)          # 零假设为真:差值均值 = 0
        rej += int(paired_permutation_test(d, n_perm=n_perm3,
                                           seed=int(i))["p_perm"] <= 0.05)
    a_hat = rej / n_rep3
    check("α=0.05 下 I 类错误率 ∈ [0.02, 0.09]", 0.02 <= a_hat <= 0.09,
          f"实测 {a_hat:.3f}")

    print("== T4 闭式 MDE 处的实际功效 ==")
    sigma_d, n = 30.0, 100
    delta = mde_analytic(sigma_d, n, alpha=0.05, power=0.8)
    r4 = np.random.default_rng(4)
    n_rep4 = 200 if quick else 500
    n_perm4 = 499 if quick else 999
    hit = 0
    for i in range(n_rep4):
        d = r4.normal(delta, sigma_d, n)
        hit += int(paired_permutation_test(d, n_perm=n_perm4,
                                           seed=int(i))["p_perm"] <= 0.05)
    pw = hit / n_rep4
    check("MDE 处功效 ≈ 0.8 (±0.08)", abs(pw - 0.8) <= 0.08,
          f"delta={delta:.3f}° 实测功效 {pw:.3f}")

    print("== T5 解析量化地板 ==")
    a = analytic_quantization_floor(n_mc=200000, seed=5)
    check("闭式 22.5° 与蒙特卡洛一致(<0.1°)",
          a["closed_form_vs_mc_abs_diff"] < 0.1,
          f"MC={a['monte_carlo_uniform_theta']['mean']:.4f}°")
    check("闭式 sd = 45/√12", abs(a["closed_form"]["std"] - 12.9904) < 1e-3)

    print("\n" + ("自检全部通过" if ok else "自检存在失败项"))
    return 0 if ok else 1


# ============================================================================
# markdown 渲染(直接贴论文附录)
# ============================================================================

def _f(x, nd=2, dash="—"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{x:.{nd}f}"


def render_markdown(rep: Dict) -> str:
    L: List[str] = []
    A = L.append
    A("# 附录:方向指标的仪器标定与分辨力")
    A("")
    A("> 方向可控性由 Farnebäck 光流估计运动主方向。这条量尺**两端都不是 0/180**:")
    A("> 上端是随机猜测的期望 90°,下端是由**四方向 prompt 的量化残差**与"
      "**光流估计器自身噪声**共同构成的地板。")
    A("> 本节量化地板、给出置信区间、把仪器噪声与内容噪声分开,"
      "并给出最小可检测差异(MDE)随样本量的变化,")
    A("> 以便判断表中的臂间差是否统计可分辨。所有数字由 "
      "`server_run/instrument_floor_report.py` 生成。")
    A("")

    # ---- §1 ----
    a = rep["S1_analytic_quantization_floor"]
    A("## 1 量化地板的解析值")
    A("")
    A(f"建库器按 45°/135°/225°/315° 把连续净位移角 θ 量化到 E/N/W/S "
      f"(代表角 0/90/180/270);指标目标角与估计角同时 +180° 相消,故折叠角误差")
    A("")
    A("$$e \\;=\\; \\bigl|\\,((\\theta+45)\\bmod 90)-45\\,\\bigr|,\\qquad "
      "\\theta\\ \\text{箱内均匀}\\ \\Rightarrow\\ e\\sim U(0^\\circ,45^\\circ)$$")
    A("")
    A("| 量 | 闭式 | 蒙特卡洛 (N=%s) | 实测(H5 waypoints) |"
      % f"{a['monte_carlo_uniform_theta']['n']:,}")
    A("|---|---|---|---|")
    emp = a.get("empirical_from_waypoints")
    for k, lab in (("mean", "均值"), ("median", "中位数"), ("std", "标准差"),
                   ("p95", "95 分位")):
        A(f"| {lab} | {_f(a['closed_form'].get(k))}° | "
          f"{_f(a['monte_carlo_uniform_theta'].get(k))}° | "
          f"{_f(emp.get(k)) + '°' if emp else '—'} |")
    A("")
    A(f"**这一项与模型无关**:即使生成完全忠实于 GT,方向表也只能报到 22.5°。")
    A(f"> 例外:`--angle-quantize 4` 建的判门探针库 θ∈{{0,90,180,270}},"
      f"量化残差恒为 0,其地板是**纯仪器噪声** —— 两库地板不可混用。")
    A("")

    # ---- §2 ----
    c = rep["S2_synthetic_control"]
    A("## 2 合成对照:隔离估计器自身的误差")
    A("")
    A(f"输入是**纯位移、无内容变化**的合成视频(窗口 "
      f"{c['_window'][0]}×{c['_window'][1]},T={c['_frames']},"
      f"{len(c['_angles_deg'])} 个方向角,亚像素平移)。"
      f"任何误差都只能归给估计器本身,不含内容变化,也不含 prompt 量化。")
    A("")
    A("> 纹理:`smooth`=各向同性低通随机场(理想工况);"
      "`stripes`=单向正弦光栅(周期 24 px,用于暴露**孔径问题**;"
      "长周期是为了把时间混叠推到量程之外,不与孔径效应混淆);"
      "`sparse`=大片平坦 + 少量斑点(低纹理,如水面/滩涂)。")
    A("")
    A("| 纹理 | 位移 (px/帧对) | 有效率 | 平均 \\|误差\\| | 中位 | p95 | 平均带符号误差 |")
    A("|---|---|---|---|---|---|---|")
    for key in c["order"]:
        v = c["cells"][key]
        q = v["abs_error"]
        A(f"| {v['texture']} | {v['px_per_pair']:g} | {_f(v['valid_pct'],0)}% | "
          f"{_f(q.get('mean'))}° | {_f(q.get('median'))}° | {_f(q.get('p95'))}° | "
          f"{_f(v.get('signed_error_mean'))}° |")
    A("")
    d = rep.get("S5_estimator_diagnostics", {})
    if "D1_dynamic_range" in d:
        A(f"- **量程**:{d['D1_dynamic_range']['verdict']}")
        A(f"  {d['D1_dynamic_range']['why_it_matters']}")
    if "D2_aperture_anisotropy" in d:
        A(f"- **孔径/各向异性**:{d['D2_aperture_anisotropy']['verdict']}")
        A(f"  {d['D2_aperture_anisotropy']['why_it_matters']}")
    A("")

    # ---- §3 ----
    g = rep.get("S3_gt_floor") or {}
    A("## 3 GT 视频上的地板(完美输入下的角误差分布)")
    A("")
    if not g.get("n_evaluated"):
        A(f"> ⚠️ **本次运行没有可用的 GT 库**"
          f"({g.get('unavailable', '未提供 --gt-h5')}),"
          f"本节留空。§4 的地板池改用 §1+§2 合成的地板(见下),"
          f"该替代值**不含 GT 内容效应**,是地板的**下界**而非地板本身。")
    else:
        A(f"口径:{g['_caliber']};光流窗口 {g['_flow_window'][0]}×{g['_flow_window'][1]}"
          f"({g.get('_window_source')})。样本 n={g['n_evaluated']}"
          f"(估计器拒答 {g['n_estimator_none']} 条,有效率 {_f(g['valid_pct'],1)}%)。")
        A("")
        A("| 目标口径 | 均值 | 95% CI (BCa) | 中位 | 标准差 | p25 | p75 | p95 |")
        A("|---|---|---|---|---|---|---|---|")
        for key, lab in (("vs_quantized_target", "vs 量化目标(论文口径)"),
                         ("vs_continuous_target", "vs 连续目标(扣掉量化)")):
            b = g[key]
            A(f"| {lab} | **{_f(b.get('mean'))}°** | "
              f"[{_f(b.get('ci_lo'))}, {_f(b.get('ci_hi'))}] | "
              f"{_f(b.get('median'))}° | {_f(b.get('std'))}° | "
              f"{_f(b.get('p25'))}° | {_f(b.get('p75'))}° | {_f(b.get('p95'))}° |")
        A("")
        eb = rep.get("S3b_error_budget") or {}
        if "term_quantization" in eb:
            A("**误差预算分解**")
            A("")
            A("| 分项 | 值 | 来源 |")
            A("|---|---|---|")
            tq = eb["term_quantization"]
            A(f"| 量化项 | {_f(tq.get('from_target_difference_deg'))}° | "
              f"两种目标口径的均值之差 |")
            A(f"| 量化项(核对) | {_f(tq.get('from_waypoints_mean_deg'))}° | "
              f"直接由 waypoints 算 |")
            ti = eb.get("term_instrument")
            A(f"| 仪器项 | {_f(ti.get('mean_abs_error')) if ti else '—'}° | "
              f"合成对照 @{_f(ti.get('px_per_pair_used'),1) if ti else '—'} px/帧对"
              f"(GT 实测位移中位 {_f(ti.get('gt_observed_px_per_pair_median'),1) if ti else '—'} px) |")
            A(f"| 内容残差 | {_f(eb.get('term_content_residual_deg'))}° | "
              f"vs连续目标 − 仪器项({eb.get('term_content_residual_note')}) |")
            A(f"| **合计(实测地板)** | **{_f(eb.get('total_floor_vs_quantized_target_mean'))}°** | "
              f"GT vs 量化目标 |")
            A("")
            u = eb["usable_dynamic_range"]
            A(f"**可用动态范围** = [地板 {_f(u['floor_mean_deg'])}°(95% CI "
              f"[{_f(u['floor_ci95'][0])}, {_f(u['floor_ci95'][1])}]), "
              f"随机 90°],宽 **{_f(u['range_deg'])}°**。{u['note']}")
            A("")
        ps = g.get("per_scene_vs_quantized") or {}
        if len(ps) > 1:
            A("| 场景 | n | 均值 | 中位 | p95 |")
            A("|---|---|---|---|---|")
            for k, v in ps.items():
                A(f"| {k} | {v.get('n')} | {_f(v.get('mean'))}° | "
                  f"{_f(v.get('median'))}° | {_f(v.get('p95'))}° |")
            A("")
    cf = rep.get("S3c_composed_floor") or {}
    if cf.get("stats"):
        A(f"合成地板(量化+仪器,{cf.get('theta_source')},"
          f"仪器档 {_f(cf.get('px_per_pair_used'),1)} px/帧对):"
          f"均值 {_f(cf['stats']['mean'])}°,中位 {_f(cf['stats']['median'])}°,"
          f"标准差 {_f(cf['stats']['std'])}°。")
        A("")

    # ---- §4 ----
    r = rep["S4_resolution"]
    A("## 4 分辨力:两个臂相差多少度才可分辨?")
    A("")
    A(f"**检验**:{r['_test']}。{r['_why_paired']}。")
    A("")
    A("$$\\mathrm{MDE}(n)=\\bigl(z_{1-\\alpha/2}+z_{1-\\beta}\\bigr)"
      "\\frac{\\sigma_d}{\\sqrt n},\\qquad \\sigma_d=\\sigma\\sqrt{2(1-\\rho)}$$")
    A("")
    A(f"其中 σ 是单臂逐样本角误差的标准差(混合模型下 "
      f"σ≈{_f(r['arm_sigma_under_mixture'])}°,以 base mean="
      f"{_f(r['arm_base_mean_deg'])}° 标定),ρ 是两臂在同一轨迹上的相关系数。"
      f"α={r['alpha']},目标功效={r['power_target']}。")
    A("")
    A("> **注意**:只用 $z_{1-\\alpha/2}$ 得到的是「恰好显著」的检出阈"
      "(真差等于它时功效仅 50%),不是 MDE。下表两者都给,"
      "后者与 `metric_sample_size_study.py` 的 $1.96\\sqrt2 s/\\sqrt n$ 对应"
      "(即本表 ρ=0、功效 50% 的特例)。")
    A("")
    names = [k for k in r["scenarios"]]
    A("| N | " + " | ".join(f"MDE ({k})" for k in names)
      + " | 检出阈 (50% 功效, unpaired) |")
    A("|---" * (len(names) + 2) + "|")
    pw_key = "mde_power%d" % int(round(r["power_target"] * 100))
    for row in r["mde_table"]:
        cells = [f"{_f(row[k][pw_key])}°" for k in names]
        A(f"| {row['n']} | " + " | ".join(cells) + " | "
          + f"{_f(row['unpaired']['detection_threshold_power50'])}° |")
    A("")
    rm = r["resolvability_matrix"]
    A(f"**可分辨性矩阵**(保守场景 `{rm['conservative_scenario']}` / "
      f"乐观场景 `{rm['optimistic_scenario']}` 夹逼):")
    A("")
    sizes = [row["n"] for row in r["mde_table"]]
    A("| 臂间差 δ | " + " | ".join(f"N={n}" for n in sizes) + " |")
    A("|---" * (len(sizes) + 1) + "|")
    for row in rm["rows"]:
        A(f"| {row['delta_deg']:g}° | "
          + " | ".join(row[str(n)] for n in sizes) + " |")
    A("")
    sv = r["simulation_vs_analytic"]
    A(f"**闭式与模拟的一致性校核**:在物理动机的混合模型"
      f"(地板池 ↔ 均匀失控,共享轨迹难度产生配对相关)下模拟置换检验的实际功效,"
      f"与闭式对照。{sv['verdict']}({sv.get('tolerance_note')})。")
    A("")
    A("| N | δ | 模拟功效 | 闭式功效 | 模拟 MC SE | 模拟 ρ |")
    A("|---|---|---|---|---|---|")
    for row in sv["rows"]:
        A(f"| {row['n']} | {row['delta']:g}° | {_f(row['power_simulated'],3)} | "
          f"{_f(row['power_analytic'],3)} | ±{_f(row.get('mc_se'),3)} | "
          f"{_f(row.get('rho_simulated'),3)} |")
    A("")

    # ---- 真实臂 ----
    ra = rep.get("S4b_real_arms") or {}
    if ra.get("pairs"):
        A("### 4.1 实际臂间比较(逐样本配对)")
        A("")
        A("| 比较 | n | A 均值 | B 均值 | 差 (B−A) | 95% CI | 置换 p | 配对 t p | ρ | 该 N 的 MDE | 结论 |")
        A("|---|---|---|---|---|---|---|---|---|---|---|")
        for k, v in ra["pairs"].items():
            if v.get("skipped"):
                A(f"| {k} | — | — | — | — | — | — | — | — | — | {v['skipped']} |")
                continue
            ci = v["diff_ci95"]
            A(f"| {k} | {v['n_paired']} | {_f(v['mean_a'])}° | {_f(v['mean_b'])}° | "
              f"{v['mean_diff_b_minus_a']:+.2f}° | "
              f"[{_f(ci['ci_lo'])}, {_f(ci['ci_hi'])}] | "
              f"{_f(v['permutation']['p_perm'],4)} | {_f(v['paired_t']['p'],4)} | "
              f"{_f(v.get('rho'),3)} | {_f(v['mde_at_this_n'])}° | "
              f"{'可分辨' if v['resolvable'] else '**不可分辨**'} |")
        A("")

    # ---- §5 ----
    A("## 5 估计器的已知系统性偏差(如实列出)")
    A("")
    for key in ("D1_dynamic_range", "D2_aperture_anisotropy",
                "D3_net_displacement_degeneracy", "D4_pair_sampling",
                "D5_near_static_guard"):
        v = d.get(key)
        if not v:
            continue
        A(f"**{key}** — {v.get('verdict')}")
        if v.get("why_it_matters"):
            A(f"  {v['why_it_matters']}")
        A("")

    # ---- 结论 ----
    v = rep.get("S6_verdict") or {}
    A("## 6 结论")
    A("")
    for line in v.get("statements", []):
        A(f"- {line}")
    A("")
    A(f"<sub>生成:`{rep['_meta'].get('generated_at')}`;"
      f"code_sha256=`{str(rep['_meta'].get('run_manifest', {}).get('code_sha256'))[:12]}`;"
      f"role=`{rep['_meta'].get('run_manifest', {}).get('role')}`。</sub>")
    return "\n".join(L)


# ============================================================================
# main
# ============================================================================

def _parse_list(s: str, cast=float):
    return [cast(x) for x in str(s).split(",") if str(x).strip() != ""]


def build_meta(args) -> Dict:
    """产物身份。**不写独立的 run_manifest.json**,而是内嵌进报告 ——
    collect_results.py 会把任何含 run_manifest.json 的目录当成 eval 产物扫,
    在报告目录里放一个同名文件等于埋雷。身份信息一样全,只是换个位置。
    另:本报告是**仪器标定**,不是模型臂,headline 资格强制置 False。"""
    meta = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "argv": sys.argv[1:]}
    try:
        from run_manifest import build_manifest
        role = os.environ.get("RSWM_RUN_ROLE", "metric_calibration")
        meta["run_manifest"] = build_manifest(
            role=role, data_h5=args.gt_h5, split_file=args.split,
            extra={
                "headline_eligible": False,
                "headline_ineligible_reason":
                    "instrument_floor_report 是方向指标的仪器标定,不是模型臂,"
                    "其数字不进 headline 表",
                "report_kind": "instrument_floor",
            })
    except Exception as e:                       # 缺依赖也要留痕,不静默
        meta["run_manifest"] = {"unavailable": f"{type(e).__name__}: {e}"}
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="方向指标的仪器地板 + 置信区间 + 分辨力标定报告")
    ap.add_argument("--gt-h5", default=None, help="GT H5(可选;缺则纯合成)")
    ap.add_argument("--split", default=None, help="train_val_split.json(可选)")
    ap.add_argument("--eval-dirs", nargs="*", default=[],
                    help="name=dir,读其 scores.json 的逐样本角误差做真实配对比较")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--window-size", nargs=2, type=int, default=None,
                    metavar=("H", "W"), help="强制光流窗口(默认读 H5 metadata)")
    ap.add_argument("--displacements", default="1,2,4,6,9,12,16,24,34",
                    help="合成对照的每帧对位移(px)")
    ap.add_argument("--textures", default="smooth,stripes,sparse")
    ap.add_argument("--syn-angles", type=int, default=24,
                    help="合成对照扫多少个方向角(均匀铺满 360°)")
    ap.add_argument("--syn-frames", type=int, default=9,
                    help="合成视频帧数(默认 9 = K=1 的契约帧数)")
    ap.add_argument("--noise-sigma", type=float, default=0.0,
                    help="合成帧的加性高斯噪声 σ(灰阶),默认 0")
    ap.add_argument("--sizes", default="25,50,100,200,400,800")
    ap.add_argument("--deltas", default="5,10,15,20,30",
                    help="要判定可分辨性的臂间差(度)")
    ap.add_argument("--rho-scenarios", default="0,0.3,0.6",
                    help="臂间相关系数情景(无真实数据时用)")
    ap.add_argument("--base-mean", type=float, default=None,
                    help="模拟用的基准臂均值角误差(默认取实测/合成地板+55°)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--n-perm", type=int, default=4999)
    ap.add_argument("--n-sim", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true",
                    help="缩小所有重抽样规模(冒烟用,数字不得进论文)")
    ap.add_argument("--self-test", action="store_true",
                    help="只跑统计与估计器自检,不产报告")
    ap.add_argument("-o", "--output", default=None,
                    help="输出 json 路径(同时写同名 .md)")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(quick=args.quick)

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[FATAL] 本报告必须有 OpenCV(标定的对象就是 Farnebäck 光流)",
              file=sys.stderr)
        return 2

    if args.quick:
        args.n_boot = min(args.n_boot, 999)
        args.n_perm = min(args.n_perm, 499)
        args.n_sim = min(args.n_sim, 40)
        args.syn_angles = min(args.syn_angles, 8)

    t0 = time.time()
    rep: Dict = {"_meta": build_meta(args)}
    rep["_meta"]["quick_mode"] = bool(args.quick)
    if args.quick:
        rep["_meta"]["WARNING"] = "--quick:重抽样规模已缩小,数字仅供冒烟,勿进论文"

    win = tuple(args.window_size) if args.window_size else None

    # ---- §3 先跑(它给出真实工作点位移与净角池,后面两节要用)----
    gt: Dict = {}
    if args.gt_h5 and Path(args.gt_h5).is_file():
        print(f"[1/5] GT 地板:{args.gt_h5}")
        gt = measure_gt_floor(args.gt_h5, args.split,
                              max_samples=args.max_samples, n_boot=args.n_boot,
                              alpha=args.alpha, seed=args.seed,
                              force_window=win)
        win = win or tuple(gt["_flow_window"])
    else:
        gt = {"unavailable": ("未提供 --gt-h5 或文件不存在"
                              if not args.gt_h5 else f"{args.gt_h5} 不存在")}
        print(f"[1/5] GT 地板:跳过({gt['unavailable']})")
    win = win or CANONICAL_WINDOW

    # ---- §1 ----
    print("[2/5] 量化地板(解析 + 蒙特卡洛 + 实测)")
    theta_pool = np.asarray(gt.get("_net_theta_deg") or [], dtype=float)
    rep["S1_analytic_quantization_floor"] = analytic_quantization_floor(
        n_mc=20000 if args.quick else 200000, seed=args.seed,
        empirical_angles=theta_pool if theta_pool.size else None)

    # ---- §2 ----
    disp = _parse_list(args.displacements, float)
    obs_px = (gt.get("observed_flow_px_per_pair") or {}).get("median")
    if obs_px is not None and all(abs(obs_px - p) > 0.5 for p in disp):
        # 把 GT 的真实工作点补进扫描格 —— 仪器项必须在真实工作点上取,
        # 否则"仪器噪声"标的是一个benchmark里根本不存在的位移档。
        disp = sorted(disp + [round(float(obs_px), 2)])
    angles = list(np.linspace(0, 360, int(args.syn_angles), endpoint=False))
    print(f"[3/5] 合成对照:{len(disp)} 位移 × {len(args.textures.split(','))} 纹理 "
          f"× {len(angles)} 角度 @ {win}")
    control = synthetic_control(win=win, displacements=disp,
                                textures=args.textures.split(","),
                                angles=angles, T=int(args.syn_frames),
                                seed=args.seed, noise_sigma=args.noise_sigma)
    rep["S2_synthetic_control"] = control
    rep["S5_estimator_diagnostics"] = estimator_diagnostics(
        control, win=win, seed=args.seed)

    rep["S3_gt_floor"] = gt
    rep["S3b_error_budget"] = error_budget(
        gt, control, rep["S1_analytic_quantization_floor"])

    composed = compose_floor_pool(
        control, px_target=obs_px, n_mc=20000 if args.quick else 100000,
        seed=args.seed, theta_pool=theta_pool if theta_pool.size else None)
    pool_arr = composed.pop("_pool_array", None)
    rep["S3c_composed_floor"] = composed

    # ---- §4 ----
    if gt.get("n_evaluated") and gt.get("_per_sample_vs_quantized"):
        # 实测地板池优先:它含 GT 内容效应,是真正的地板
        floor_pool = np.asarray(gt["_per_sample_vs_quantized"], dtype=float)
        pool_src = "GT 实测地板(含内容效应)"
    elif pool_arr is not None:
        floor_pool = np.asarray(pool_arr, dtype=float)
        pool_src = "合成地板(量化+仪器;**不含 GT 内容效应,是地板下界**)"
    else:
        floor_pool = np.abs(np.random.default_rng(args.seed).uniform(
            -45, 45, 20000))
        pool_src = "退化:纯量化地板 U(0,45)(合成对照也不可用)"
    base_mean = (args.base_mean if args.base_mean is not None
                 else min(float(floor_pool.mean()) + 55.0, 88.0))
    print(f"[4/5] 分辨力:地板池={pool_src} (mean={floor_pool.mean():.2f}°), "
          f"base_mean={base_mean:.1f}°")

    real_pairs = {}
    if args.eval_dirs:
        rep["S4b_real_arms"] = compare_real_arms(
            args.eval_dirs, n_boot=args.n_boot, n_perm=args.n_perm,
            alpha=args.alpha, power=args.power,
            sizes=_parse_list(args.sizes, int), seed=args.seed)
        real_pairs = {k: v for k, v in rep["S4b_real_arms"]["pairs"].items()
                      if not v.get("skipped")}

    rep["S4_resolution"] = resolution_analysis(
        floor_pool=floor_pool, base_mean=base_mean,
        sizes=_parse_list(args.sizes, int), deltas=_parse_list(args.deltas, float),
        alpha=args.alpha, power=args.power,
        rho_scenarios=_parse_list(args.rho_scenarios, float),
        real_pairs=real_pairs, n_sim=args.n_sim,
        n_perm=min(args.n_perm, 999), seed=args.seed)
    rep["S4_resolution"]["floor_pool_source"] = pool_src

    # ---- §6 ----
    print("[5/5] 汇总")
    rep["S6_verdict"] = _verdict(rep)
    rep["_meta"]["elapsed_sec"] = round(time.time() - t0, 1)

    md = render_markdown(rep)
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2, ensure_ascii=False,
                                default=str), encoding="utf-8")
        p.with_suffix(".md").write_text(md, encoding="utf-8")
        print(f"[done] {p}\n[done] {p.with_suffix('.md')}")
    else:
        print(md)
    return 0


def _verdict(rep: Dict) -> Dict:
    """把散落的数字读成几句能直接进论文的结论。"""
    st: List[str] = []
    a = rep["S1_analytic_quantization_floor"]
    st.append(f"四方向 prompt 的**量化地板**闭式为 "
              f"E|offset|={a['closed_form']['mean']:.1f}°"
              f"(蒙特卡洛 {a['monte_carlo_uniform_theta']['mean']:.2f}°"
              + (f",H5 实测 {a['empirical_from_waypoints']['mean']:.2f}°"
                 if a.get("empirical_from_waypoints") else "")
              + ");它与模型能力无关。")
    d = rep.get("S5_estimator_diagnostics", {})
    d1 = d.get("D1_dynamic_range", {})
    if d1.get("reliable_px_per_pair_max") is not None:
        st.append(f"估计器在理想合成输入上的**可靠位移带**为 ≤"
                  f"{d1['reliable_px_per_pair_max']:g} px/帧对(平均角误差 ≤5°);"
                  f"超出后读数不可用 —— 这就是官方口径必须在窗口分辨率、"
                  f"step=1 下计算的原因。")
    g = rep.get("S3_gt_floor") or {}
    if g.get("n_evaluated"):
        b = g["vs_quantized_target"]
        st.append(f"**GT 视频上的实测地板** = {b['mean']:.2f}°"
                  f"(95% CI [{b['ci_lo']:.2f}, {b['ci_hi']:.2f}],"
                  f"中位 {b['median']:.2f}°,n={b['n']});"
                  f"可用动态范围仅 [{b['mean']:.1f}°, 90°]。")
        eb = rep.get("S3b_error_budget", {})
        if eb.get("term_instrument"):
            st.append(f"误差预算:量化 "
                      f"{eb['term_quantization']['from_target_difference_deg']:.1f}° / "
                      f"仪器 {eb['term_instrument']['mean_abs_error']:.1f}° / "
                      f"内容残差 {eb['term_content_residual_deg']:.1f}°。")
    else:
        st.append("**未提供 GT 库**,实测地板一栏为空;报告中的地板取自"
                  "「量化+仪器」合成值,它不含 GT 内容效应,是地板的**下界**"
                  "(即真实地板只会更高、可用范围只会更窄)。")
    r = rep["S4_resolution"]
    pw_key = "mde_power%d" % int(round(r["power_target"] * 100))
    rowH = next((x for x in r["mde_table"] if x["n"] == N_EVAL_HEADLINE), None)
    rowA = next((x for x in r["mde_table"] if x["n"] == N_EVAL_ABLATION), None)
    key_cons = r["resolvability_matrix"]["conservative_scenario"]
    if rowH or rowA:
        parts = []
        if rowA:
            parts.append(f"N={N_EVAL_ABLATION}(消融/诊断臂)→ "
                         f"{rowA[key_cons][pw_key]:.1f}°")
        if rowH:
            parts.append(f"N={N_EVAL_HEADLINE}(headline 臂)→ "
                         f"{rowH[key_cons][pw_key]:.1f}°")
        st.append(f"在最保守情景(`{key_cons}`)下的 MDE:" + ";".join(parts)
                  + f"(α={r['alpha']}, 功效={r['power_target']})。"
                    f"**小于该值的臂间差在对应样本量下不可分辨。**")
        if rowH:
            mde_h = rowH[key_cons][pw_key]
            head = (f"**达到**可检测量级(MDE={mde_h:.1f}°),裕度 "
                    f"{15.0 - mde_h:.1f}°,必须同时报 CI"
                    if 15.0 >= mde_h else
                    f"**达不到**可检测量级(MDE={mde_h:.1f}°),"
                    f"不能据此声称两臂有差异")
            tail = ""
            if rowA:
                mde_a = rowA[key_cons][pw_key]
                tail = (f";而在 N={N_EVAL_ABLATION} 的消融/诊断臂上 "
                        f"MDE={mde_a:.1f}°,同样的 15° 差"
                        + ("仍可分辨。" if 15.0 >= mde_a else "**不可分辨**。"))
            st.append(f"直接回答审稿人的问题:形如「77° vs 92°」的 15° 差,"
                      f"在 headline 的 N={N_EVAL_HEADLINE} 下" + head + tail)
    rho_sim = [x.get("rho_simulated") for x in r.get("simulated_power", [])
               if x.get("rho_simulated") is not None]
    if rho_sim:
        st.append(f"配对能省多少:混合模型模拟出的臂间相关 ρ≈"
                  f"{float(np.mean(rho_sim)):.2f} —— 在高误差区两臂的失败样本"
                  f"取值互相独立(只有失败**事件**共享),配对带来的方差削减很小,"
                  f"所以保守情景(ρ=0)才是现实情景,不要指望配对救回功效。")
    sv = r["simulation_vs_analytic"]
    st.append(f"闭式 MDE 已用置换检验的模拟功效校核:{sv['verdict']}。")
    ra = rep.get("S4b_real_arms") or {}
    for k, v in (ra.get("pairs") or {}).items():
        if v.get("skipped"):
            continue
        st.append(f"实际比较 {k}:差 {v['mean_diff_b_minus_a']:+.2f}°,"
                  f"置换 p={v['permutation']['p_perm']:.4f},"
                  f"该 N 的 MDE={v['mde_at_this_n']:.2f}° ⇒ "
                  + ("可分辨。" if v["resolvable"] else "**不可分辨。**"))
    st.append("推论:落在 [地板, 90°] 之外的读数(尤其 >90°)与随机不可区分,"
              "不应解读为『比随机更差』;所有臂间差都应与本节的 MDE 并列报告。")
    return {"statements": st}


if __name__ == "__main__":
    sys.exit(main())
