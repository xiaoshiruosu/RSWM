#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""近重复 / 跨数据集重复 / 拼接接缝 检测器  [v3.1 新增]

═══════════════════════════════════════════════════════════════════════════
为什么需要这个文件
═══════════════════════════════════════════════════════════════════════════

`make_official_splits.py` 保证的是 **source-disjoint**:同一个 source id 不会
同时出现在 train 和 test。这**不等于** content-disjoint。

遥感数据集里两张不同索引的图内容近乎相同是常态(同一地物被重复采样、
数据集自带增强副本、不同版本重新发布)。source-disjoint 拦不住这种泄漏 ——
它只看 id,不看像素。

更要紧的是**跨数据集重复**。本项目四个 tier 全是 Google Earth 派生:

    canonical(train)  blanchon/AID                600²  直裁
    泛化 #1           blanchon/RESISC45           256²  2×2 拼接
    泛化 #2           blanchon/PatternNet         256²  2×2 拼接
    泛化 #3           jonathan-roberts1/WHU-RS19  1001×600²+4 异形图  零拼接

AID 与 WHU-RS19 出自同一课题组，且 WHU 有 1001/1005 为 600² Google Earth
影像。如果两者
在源影像层面有交集,那"泛化 gap"测的就不是泛化 —— 是同分布重测。
这条不核实,`generalization_gap_report.py` 的整张表都没有解释力。

第三件事:**拼接接缝本身是混杂因子**。512 目标尺寸下 256² 源图走
`num_tiles = ceil(512/256) = 2`,所以 resisc45 / patternnet 的 full_scene
在**第 256 行与第 256 列**必然有人工接缝,而 AID / WHU-RS19 直裁没有。
模型没见过接缝 → 在带缝场景上变差,这跟地理/传感器域移完全无关。
也就是说 AID→RESISC45 的 gap 里混着一个纯人造伪影项。

═══════════════════════════════════════════════════════════════════════════
与 check_benchmark_duplicates.py 的分工(**先跑那个**)
═══════════════════════════════════════════════════════════════════════════

仓库里已有 `check_benchmark_duplicates.py`,做的是**逐位精确**重复:
md5(waypoints/video_frames/full_scene) 三类指纹,单库内分组,交叉 train/val。

两者互补,不是重复实现:

    check_benchmark_duplicates.py   md5 精确    单库内   便宜、证据是字节级
    check_near_duplicates.py(本文件) 感知哈希   跨库     抗重编码/缩放/翻转

精确重复 ⊂ 近重复,所以本文件在功能上覆盖它,但**它更该先跑**:
md5 相同是不可争辩的证据,而感知哈希判定永远带 TPR/FPR。
顺序应为:先 `check_benchmark_duplicates.py` 定位字节级重复并修掉,
再用本文件查"改过一遍编码/尺寸就躲过 md5"的那类。

═══════════════════════════════════════════════════════════════════════════
它能做什么、不能做什么(这条必须先说)
═══════════════════════════════════════════════════════════════════════════

**能**检出:同一张源影像经不同管线(重编码 / 缩放 / 轻微裁切 / 翻转旋转)
后出现在两个库里 —— 感知哈希对这类变换稳定。

**不能**检出:同一地理区域被**不同取景**采集(裁框不同、季节不同、
分辨率差一档)。这需要地理坐标,而这四个 HF 数据集都不带 geo-metadata。
所以本工具给出的重复率是**下界**,不是全部。论文里不得把
"本工具未检出重复"写成"两库无交集";只能写成
"未检出可由感知哈希识别的重复(下界)"。

这个区分很重要:遥感数据集之间最常见的交集恰恰是**同区域不同取景**,
而那正是本工具的盲区。

═══════════════════════════════════════════════════════════════════════════
用法
═══════════════════════════════════════════════════════════════════════════

    # 0) 先标定阈值(不给阈值就拒绝判定 —— fail-closed)
    python3 check_near_duplicates.py --calibrate --h5 canonical.h5

    # 1) 划分泄漏:同一库内跨 split 的内容重复
    python3 check_near_duplicates.py --check leakage \\
        --h5 canonical.h5 --splits official_splits.json

    # 2) 跨数据集重复:canonical vs 各泛化 tier
    python3 check_near_duplicates.py --check cross \\
        --h5 canonical.h5 --other resisc45.h5 --other whurs19.h5

    # 3) 拼接接缝:量化伪影混杂
    python3 check_near_duplicates.py --check seam --h5 resisc45.h5

    python3 check_near_duplicates.py --self-test

退出码
    0  通过         3  无法标定出可用阈值 → 拒绝出数(fail-closed)
    4  划分泄漏     5  跨库重复超过预注册上限
    6  输入不可用   7  自检失败
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_NEW = _HERE.parent / "lora_remotesense" / "new"
if str(_NEW) not in sys.path:
    sys.path.insert(0, str(_NEW))


# ══════════════════════════════════════════════════════════════════════
# 预注册读数规则 —— 看到任何数字之前就钉死
# ══════════════════════════════════════════════════════════════════════

PREREG = """
【预注册读数规则】(在看到任何数字之前写定;事后不得修改)

R1 划分泄漏(train↔val↔test 内容重复)
    容忍上限 = **0 对**。这是 benchmark 完整性问题,不是统计问题:
    只要有一对,test 上的数字就不再是"未见过的数据"。
    检出 >0 → 退出码 4,必须重划分后重跑,不接受"比例很小所以忽略"。

R2 跨数据集重复率(canonical ∩ 泛化 tier)
    dup_rate = 被判定与 canonical 中某场景近重复的 tier 场景占比。
      dup_rate = 0                 → 该 tier 可作泛化证据(仅限哈希可见的重复)
      0 < dup_rate ≤ 0.01          → 报告并从该 tier 剔除重复样本后重算 gap
      dup_rate > 0.01              → **该 tier 不得作为泛化证据**(退出码 5)。
                                      不是"扣掉重复部分就行":重复的存在说明
                                      两库共享采集源,未被哈希检出的同区域
                                      不同取景大概率也存在。
    ⚠ dup_rate=0 只能支持"无可由哈希识别的重复",见上文盲区说明。

R3 拼接接缝
    seam_z = 网格边界处相邻行/列差分 相对 非边界位置分布 的 z 分数。
      seam_z < 3    → 接缝不可检出,拼接 tier 与直裁 tier 可直接比较
      seam_z ≥ 3    → 接缝可检出。该 tier 的 gap **混杂人造伪影**,
                       必须在论文中声明,或改用零拼接 tier(whurs19)作主泛化证据
    预期:resisc45 / patternnet 会触发(2×2 拼接),aid / whurs19 不会。
    这条**不是**待验证的假设 —— 它是建库口径的直接推论,检测只是量化幅度。

R4 标定失败即拒绝
    正例(同源不同管线)与负例(不同源)的哈希距离分布若无法分开到
    TPR ≥ 0.90 且 FPR ≤ 0.01,则**不输出任何重复率**(退出码 3)。
    一个分不开正负例的检测器报出的 dup_rate 没有意义,报了比不报更糟 ——
    它会被当成"已验证无重复"。
"""

# R2 的上限,预注册在此
CROSS_DUP_CEILING = 0.01
# R3 的 z 阈,预注册在此
SEAM_Z_GATE = 3.0
# R4 的标定要求
CALIB_MIN_TPR = 0.90
CALIB_MAX_FPR = 0.01


class Unusable(RuntimeError):
    """输入不足以完成判定。fail-closed:抛,不返回默认值。"""


# ══════════════════════════════════════════════════════════════════════
# 图像基础:缩放与灰度
# ══════════════════════════════════════════════════════════════════════

def _require_pil():
    """感知哈希对缩放核敏感 —— 拿不到好的缩放实现就拒跑。

    这里特意**不**退化成 numpy 步进采样(最近邻):最近邻会引入锯齿,
    让同一张图在 600²→32 与 512²→32 两条路上产生不同的哈希,
    于是漏掉真重复。一个会漏报的查重器比没有查重器更危险。
    """
    try:
        from PIL import Image  # noqa: F401
        return Image
    except ImportError as e:
        raise Unusable(
            "本检测器需要 Pillow(感知哈希要抗锯齿缩放;最近邻会让同源图哈希不一致→漏报)。\n"
            "  pip install Pillow"
        ) from e


def to_gray_resized(img: np.ndarray, size: int) -> np.ndarray:
    """(H,W[,C]) uint8/float → (size,size) float64 灰度。LANCZOS 抗锯齿。"""
    Image = _require_pil()
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] >= 3:
        a = a[:, :, :3]
    elif a.ndim == 3 and a.shape[2] == 1:
        a = a[:, :, 0]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8) if a.max() > 1.5 else \
            np.clip(a * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(a)
    if pil.mode != "L":
        pil = pil.convert("L")
    pil = pil.resize((size, size), Image.LANCZOS)
    return np.asarray(pil, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════
# DCT —— 自带实现,不引 scipy
# ══════════════════════════════════════════════════════════════════════

_DCT_CACHE: Dict[int, np.ndarray] = {}


def dct_matrix(n: int) -> np.ndarray:
    """正交 DCT-II 矩阵。M @ x 即 scipy.fft.dct(x, type=2, norm='ortho')。

    自己写是为了不为 32×32 这点计算量引一个依赖 ——
    但 `--self-test` 里会在 scipy 存在时逐位比对,保证不是自造口径。
    """
    if n in _DCT_CACHE:
        return _DCT_CACHE[n]
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    M = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    M *= np.sqrt(2.0 / n)
    M[0, :] *= np.sqrt(0.5)
    _DCT_CACHE[n] = M
    return M


def dct2(a: np.ndarray) -> np.ndarray:
    M = dct_matrix(a.shape[0])
    return M @ a @ M.T


# ══════════════════════════════════════════════════════════════════════
# 三种感知哈希
# ══════════════════════════════════════════════════════════════════════

def phash(img: np.ndarray, *, hash_size: int = 8, factor: int = 4) -> np.ndarray:
    """pHash:32×32 灰度 → DCT → 取左上 8×8 低频(去掉 DC)→ 与中位数比较。

    去掉 DC 项是关键:DC 就是平均亮度,保留它会让"整体调亮"被当成不同图,
    而跨库重编码经常带全局亮度偏移。
    """
    g = to_gray_resized(img, hash_size * factor)
    d = dct2(g)[:hash_size, :hash_size].flatten()[1:]     # 去 DC → 63 位
    return d > np.median(d)


def dhash(img: np.ndarray, *, hash_size: int = 8) -> np.ndarray:
    """dHash:相邻列亮度差的符号。对全局亮度/对比度变化天然免疫。"""
    g = to_gray_resized(img, hash_size + 1)[:hash_size, :]
    return (g[:, 1:] > g[:, :-1]).flatten()


def ahash(img: np.ndarray, *, hash_size: int = 8) -> np.ndarray:
    """aHash:最朴素的一种。单独用容易误报,这里只作交叉印证。"""
    g = to_gray_resized(img, hash_size)
    return (g > g.mean()).flatten()


HASHERS = {"phash": phash, "dhash": dhash, "ahash": ahash}
PRIMARY = "phash"   # 主判据:对缩放/JPEG 最稳


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape:
        raise ValueError(f"哈希长度不一致 {a.shape} vs {b.shape} —— 不能比较")
    return int(np.count_nonzero(a != b))


# ══════════════════════════════════════════════════════════════════════
# 哈希可靠性 —— 低纹理场景的感知哈希是噪声,必须先筛掉
# ══════════════════════════════════════════════════════════════════════

def hash_stability(img: np.ndarray, fn, *, n_probe: int = 3, seed: int = 0) -> int:
    """±1 LSB 抖动下的最大位翻转数。越大 → 该图哈希越不可靠。

    为什么需要这道闸(自检里实测出来的坑):
    pHash 比较的是 63 个低频 DCT 系数与其中位数的大小关系。**近乎无纹理**的图
    (纯渐变、大片同色水面/裸地)把几乎全部能量压在 DC 上,去掉 DC 之后剩下的
    系数只是数值噪声 —— 于是 63 个位本身就是随机的。实测:一张线性渐变图
    经 JPEG-70 重编码后 pHash 距离 22/63,而两张**完全无关**的噪声图距离 22。
    也就是说在这种图上,"同源"与"异源"的距离没有区别。

    此时把它算进判定,后果不是误报而是**漏报** —— 真重复的距离被噪声抬到阈值之上,
    检测器安静地报 "未检出重复"。这正是本项目反复出现的那类 fail-open。

    做法:给图加 ±1 LSB 抖动(肉眼、语义上都是同一张图),看哈希翻多少位。
    翻的位数 ≥ 判定阈值时,这张图**在该阈值下不具备判别力**,如实剔除并计数,
    不当作"已检查过"。
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    h0 = fn(a)
    worst = 0
    for k in range(n_probe):
        d = rng.integers(-1, 2, size=a.shape, dtype=np.int16)
        worst = max(worst, hamming(fn(np.clip(a.astype(np.int16) + d, 0, 255)
                                     .astype(np.uint8)), h0))
    return worst


# 预筛用的宽门:翻位数超过总位数这个比例的图,连标定都不该参与。
# 25% 是宽松的下限过滤(63 位里翻 16 位以上已属随机);
# **真正的判据是"翻位数 ≥ 标定阈值"**,那个在标定完成后才能算。
STABILITY_PREFILTER_FRAC = 0.25


# ══════════════════════════════════════════════════════════════════════
# 二面体群(8 变换)—— 抓翻转/旋转副本
# ══════════════════════════════════════════════════════════════════════

def dihedral_8(img: np.ndarray) -> List[np.ndarray]:
    """恒等 + 3 旋转 + 4 翻转。遥感库里镜像/旋转副本很常见
    (增强后重新发布、或不同库对同一影像取了不同朝向)。"""
    out = []
    for k in range(4):
        r = np.rot90(img, k)
        out.append(r)
        out.append(np.fliplr(r))
    return out


def hash_with_dihedral(img: np.ndarray, fn) -> List[np.ndarray]:
    return [fn(v) for v in dihedral_8(img)]


def min_dist_dihedral(query_hashes: Sequence[np.ndarray], ref: np.ndarray) -> int:
    return min(hamming(q, ref) for q in query_hashes)


# ══════════════════════════════════════════════════════════════════════
# H5 读取
# ══════════════════════════════════════════════════════════════════════

def _h5():
    try:
        import h5py
        return h5py
    except ImportError as e:
        raise Unusable("需要 h5py 才能读数据集。") from e


def _source_key(attrs) -> str:
    """场景身份。用于把"同场景的多个窗口"折叠成一个场景 ——
    同场景窗口之间当然近重复,那是设计,不是泄漏。"""
    for combo in (
        ("source_dataset", "source_class", "source_index"),
        ("source_dataset", "source_index"),
        ("source_scene_category", "source_scene_id"),
    ):
        if all(k in attrs for k in combo):
            return ":".join(str(attrs[k]) for k in combo)
    raise Unusable(
        f"样本缺少可用的源身份字段(需要以下任一组合之一):\n"
        f"  (source_dataset, source_class, source_index)\n"
        f"  (source_dataset, source_index)\n"
        f"  (source_scene_category, source_scene_id)\n"
        f"实际字段: {sorted(attrs.keys())}"
    )


def load_scenes(h5_path: str, *, limit: Optional[int] = None,
                stride: int = 1) -> Tuple[List[str], List[np.ndarray], List[str]]:
    """读每个**唯一场景**的 full_scene。

    返回 (sample_ids, scenes, source_keys) —— 同一 source_key 只取第一次出现。
    """
    h5py = _h5()
    p = Path(h5_path)
    if not p.exists():
        raise Unusable(f"H5 不存在: {h5_path}")
    ids: List[str] = []
    scenes: List[np.ndarray] = []
    keys: List[str] = []
    seen = set()
    n_no_scene = 0
    with h5py.File(str(p), "r") as f:
        # 建库器产出的是 **嵌套** 布局:`data/sample_00000/...` + 顶层 `metadata`。
        # 早先这里按扁平布局读,于是一个样本都读不到 —— 好在 load_scenes 是
        # fail-closed(抛 Unusable),不会静默返回"未检出重复"。
        # 两种布局都支持:旧的/手工构造的 H5 可能是扁平的。
        if "data" in f and isinstance(f["data"], h5py.Group) and \
                any(k.startswith("sample") for k in f["data"].keys()):
            root = f["data"]
        else:
            root = f
        names = sorted(k for k in root.keys() if k != "metadata")
        for i, name in enumerate(names):
            if i % stride:
                continue
            g = root[name]
            k = _source_key(g.attrs)
            if k in seen:
                continue
            if "full_scene" not in g:
                n_no_scene += 1
                continue
            seen.add(k)
            ids.append(name)
            scenes.append(np.asarray(g["full_scene"][:]))
            keys.append(k)
            if limit and len(scenes) >= limit:
                break
    if not scenes:
        raise Unusable(
            f"{h5_path}: 一个 full_scene 都没读到(缺 full_scene 的样本 {n_no_scene} 个)。\n"
            "查重必须看像素 —— 没有 full_scene 就**无法**判定重复,不做无证据的放行。"
        )
    if n_no_scene:
        print(f"  [warn] {h5_path}: {n_no_scene} 个样本无 full_scene,已跳过 "
              f"—— 这些样本**未被查重覆盖**")
    return ids, scenes, keys


# ══════════════════════════════════════════════════════════════════════
# 标定 —— 阈值必须由数据定,不许写魔数
# ══════════════════════════════════════════════════════════════════════

def _simulate_republish(scene: np.ndarray, kind: str) -> np.ndarray:
    """模拟"同一张源影像经另一条发布管线"。

    这些正是跨库重复的现实形态:重新编码、降采样再放大、轻微改裁框。
    """
    Image = _require_pil()
    a = np.asarray(scene)
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    a = a[:, :, :3]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    H, W = a.shape[:2]
    pil = Image.fromarray(a)
    if kind == "jpeg70":
        import io
        b = io.BytesIO()
        pil.save(b, format="JPEG", quality=70)
        b.seek(0)
        return np.asarray(Image.open(b).convert("RGB"))
    if kind == "down256":
        return np.asarray(pil.resize((256, 256), Image.LANCZOS)
                          .resize((W, H), Image.LANCZOS).convert("RGB"))
    if kind == "crop4":
        return np.asarray(pil.crop((4, 4, W - 4, H - 4))
                          .resize((W, H), Image.LANCZOS).convert("RGB"))
    if kind == "bright":
        return np.clip(a.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    if kind == "chain":
        import io
        t = pil.resize((256, 256), Image.LANCZOS).resize((W, H), Image.LANCZOS)
        b = io.BytesIO()
        t.save(b, format="JPEG", quality=75)
        b.seek(0)
        return np.asarray(Image.open(b).convert("RGB"))
    raise ValueError(kind)


REPUBLISH_KINDS = ("jpeg70", "down256", "crop4", "bright", "chain")


def wilson(x: int, n: int) -> Tuple[float, float]:
    """复用 semgate_arbiters 的实现(单一来源);拿不到就本地算同一公式。"""
    try:
        from semgate_arbiters import wilson_ci
        return wilson_ci(x, n)
    except Exception:
        if n == 0:
            return (0.0, 1.0)
        z = 1.959963984540054
        p = x / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, c - h), min(1.0, c + h))


def calibrate(scenes: Sequence[np.ndarray], *, hasher: str = PRIMARY,
              max_scenes: int = 60, seed: int = 42,
              dihedral: bool = True, allow_exclude: bool = False) -> Dict:
    """由数据定阈值。

    正例 = 同一场景 vs 其重发布变体(标签由构造保证为真)
    负例 = 不同场景两两(标签由 source_key 不同保证为真)

    然后扫所有可能阈值,按 R4 要求挑 TPR≥0.90 且 FPR≤0.01 里最宽松的那个。
    挑不出来 → 抛。**不返回"尽力而为"的阈值**:那会让下游把
    一个分不开正负例的检测器的输出当成证据。
    """
    fn = HASHERS[hasher]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(scenes))
    if len(idx) > max_scenes:
        idx = rng.choice(idx, size=max_scenes, replace=False)
    sel_all = [scenes[i] for i in idx]

    # 先按哈希稳定性预筛。低纹理图的哈希是噪声,拿它标定会把阈值抬到
    # "什么都检不出"的高度,或者干脆让正负例无法分开(实测过)。
    nbits_probe = len(fn(sel_all[0]))
    stab = [hash_stability(s, fn, seed=int(seed) + i) for i, s in enumerate(sel_all)]
    keep = [i for i, v in enumerate(stab)
            if v < STABILITY_PREFILTER_FRAC * nbits_probe]
    n_dropped = len(sel_all) - len(keep)
    if len(keep) < 4:
        raise Unusable(
            f"可用于标定的场景不足({len(keep)} 个,已剔除 {n_dropped} 个低纹理/哈希不稳定场景)。\n"
            f"  稳定性(±1LSB 抖动翻位数)min/中位/max = "
            f"{min(stab)}/{float(np.median(stab)):.0f}/{max(stab)},总位数 {nbits_probe}。\n"
            f"  这批场景纹理过弱,感知哈希在其上无判别力 —— **拒绝**给出阈值。")
    sel = [sel_all[i] for i in keep]
    sel_stab = [stab[i] for i in keep]

    base = [fn(s) for s in sel]

    pos: List[int] = []
    for s, h0 in zip(sel, base):
        for kind in REPUBLISH_KINDS:
            v = _simulate_republish(s, kind)
            hv = hash_with_dihedral(v, fn) if dihedral else [fn(v)]
            pos.append(min_dist_dihedral(hv, h0))

    neg: List[int] = []
    neg_pairs: List[Tuple[int, int]] = []          # 与 neg 同序,用于回指是哪两个场景
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            if dihedral:
                neg.append(min_dist_dihedral(hash_with_dihedral(sel[i], fn), base[j]))
            else:
                neg.append(hamming(base[i], base[j]))
            neg_pairs.append((keep[i], keep[j]))   # 回到 scenes 的原索引

    if not pos or not neg:
        raise Unusable(f"标定样本不足(正例 {len(pos)} 负例 {len(neg)})。")

    pos_a, neg_a = np.asarray(pos), np.asarray(neg)
    nbits = len(base[0])

    def _search(pa: np.ndarray, na: np.ndarray):
        b = None
        for t in range(nbits + 1):
            tp = int((pa <= t).sum())
            fp = int((na <= t).sum())
            tpr, fpr = tp / len(pa), fp / len(na)
            if tpr >= CALIB_MIN_TPR and fpr <= CALIB_MAX_FPR:
                b = dict(threshold=t, tpr=tpr, fpr=fpr, tp=tp, fp=fp)   # 取最宽松
        return b

    best = _search(pos_a, neg_a)
    excluded: List[Dict] = []
    if best is None:
        # 标定失败有两种完全不同的原因,必须分开处理 ——
        # 早先这里一律 rc=3,把一个**重要发现**当成了工具故障。
        #
        #   (a) 数据整体不可分(纹理太弱 / 哈希不适用)→ 真的只能拒绝出数
        #   (b) 负例里混着**真重复**:标定假设"source_key 不同 ⇒ 内容不同",
        #       而这假设恰好在有重复时不成立。此时低距离负例尾巴
        #       **本身就是重复的证据**,不是噪声。
        #
        # 判据:把距离 ≤ 正例最大值的负例摘出来。若摘掉它们之后就能标定,
        # 说明是 (b) —— 那些对被**提升为重复发现**(而不是被扫掉),
        # 并允许在 allow_exclude 下继续。
        cut = int(pos_a.max())
        susp = [k for k, d in enumerate(neg_a) if d <= cut]
        if susp and allow_exclude:
            mask = np.ones(len(neg_a), bool)
            for k in susp:
                mask[k] = False
            if mask.sum() >= 4:
                retry = _search(pos_a, neg_a[mask])
                if retry is not None:
                    best = retry
                    excluded = [{"scene_idx_a": neg_pairs[k][0],
                                 "scene_idx_b": neg_pairs[k][1],
                                 "distance": int(neg_a[k])} for k in susp]
        if best is None:
            lines = [
                f"无法标定出满足 R4 的阈值(要求 TPR≥{CALIB_MIN_TPR} 且 FPR≤{CALIB_MAX_FPR})。",
                f"  正例距离 min/中位/max = {pos_a.min()}/{np.median(pos_a):.0f}/{pos_a.max()}",
                f"  负例距离 min/中位/max = {neg_a.min()}/{np.median(neg_a):.0f}/{neg_a.max()}",
            ]
            if susp:
                lines += [
                    f"",
                    f"  ⚠ 有 {len(susp)} 个『异源』对的距离 ≤ 正例最大值 {cut} ——",
                    f"    这**可能不是标定失败,而是这批场景里真有重复**"
                    f"(标定假设 source_key 不同⇒内容不同,该假设此时不成立)。",
                    f"    最近的几对(scenes 原索引): "
                    + ", ".join(f"({neg_pairs[k][0]},{neg_pairs[k][1]})@d={neg_a[k]}"
                                for k in sorted(susp, key=lambda k: neg_a[k])[:8]),
                    f"    → 用 `--calib-h5 <另一个干净库>` 在别处标定,"
                    f"或 `--calib-allow-exclude` 排除这些对后标定"
                    f"(被排除的对会被登记为重复发现,不会被隐藏)。",
                ]
            else:
                lines += [
                    f"  正负例分布重叠,{hasher} 在此数据上分不开同源与异源。",
                    "  按 R4:**拒绝输出任何重复率**。此时报出的 dup_rate 会被误当成"
                    "『已验证无重复』,而它其实什么都没验证。",
                ]
            raise Unusable("\n".join(lines))
    lo_t, hi_t = wilson(best["tp"], len(pos_a))
    lo_f, hi_f = wilson(best["fp"], len(neg_a))
    # 阈值定下来之后才能算真正的判据:抖动翻位数 ≥ 阈值 = 该场景不具判别力。
    n_blind = int(sum(1 for v in sel_stab if v >= best["threshold"]))
    return {
        "hasher": hasher, "n_bits": nbits, "n_scenes_used": len(sel),
        "stability_prefilter": {
            "n_dropped_low_texture": n_dropped,
            "prefilter_frac": STABILITY_PREFILTER_FRAC,
            "kept_stability_max": int(max(sel_stab)),
            "n_blind_at_threshold": n_blind,
            "note": "n_blind_at_threshold>0 表示这些场景的哈希在该阈值下无判别力,"
                    "对它们的『未检出重复』不成立",
        },
        "n_pos": len(pos_a), "n_neg": len(neg_a),
        "threshold": best["threshold"],
        "tpr": best["tpr"], "tpr_ci95": [lo_t, hi_t],
        "fpr": best["fpr"], "fpr_ci95": [lo_f, hi_f],
        "pos_dist": {"min": int(pos_a.min()), "median": float(np.median(pos_a)),
                     "max": int(pos_a.max())},
        "neg_dist": {"min": int(neg_a.min()), "median": float(np.median(neg_a)),
                     "max": int(neg_a.max())},
        "dihedral": bool(dihedral),
        "excluded_suspicious_pairs": excluded,
        "excluded_note": (
            f"为完成标定排除了 {len(excluded)} 个疑似重复的『异源』对 —— "
            f"**这些对即为重复发现**,已登记在此,未被隐藏。" if excluded else ""),
        "republish_kinds": list(REPUBLISH_KINDS),
        "blind_spot": "同区域不同取景不在正例构造里 —— 阈值对那类重复无保证",
    }


# ══════════════════════════════════════════════════════════════════════
# 检查 1:划分泄漏
# ══════════════════════════════════════════════════════════════════════

def check_leakage(h5_path: str, splits_path: str, *, threshold: int,
                  hasher: str = PRIMARY, dihedral: bool = True,
                  limit: Optional[int] = None) -> Dict:
    """train/val/test 之间的**内容**重复。source-disjoint 拦不住这个。"""
    fn = HASHERS[hasher]
    sp = json.loads(Path(splits_path).read_text())
    # 支持三种现实中同时存在的格式：
    # 1) {train:[keys], val:[keys], test:[keys]}（内容级工具原格式）；
    # 2) make_official_splits: splits.<name>.keys；
    # 3) 训练器索引格式。索引要按 H5 sorted(data.keys()) 映射成 key。
    groups = {k: set(v) for k, v in sp.items()
              if isinstance(v, list) and k in ("train", "val", "test")}
    if not groups and isinstance(sp.get("splits"), dict):
        groups = {
            k: set(v.get("keys", []))
            for k, v in sp["splits"].items()
            if k in ("train", "val", "test") and isinstance(v, dict)
        }
    if not groups and "train_indices" in sp and "val_indices" in sp:
        with h5py.File(h5_path, "r") as _sf:
            _keys = sorted((_sf["data"] if "data" in _sf else _sf).keys())
        def _map_idx(values):
            return {_keys[int(i)] for i in values if 0 <= int(i) < len(_keys)}
        groups = {
            "train": _map_idx(sp.get("train_indices", [])),
            "val": _map_idx(sp.get("val_indices", [])),
        }
    groups = {k: v for k, v in groups.items() if v}
    if len(groups) < 2:
        raise Unusable(
            f"{splits_path}: 需要至少两个 split(train/val/test),实得 {sorted(groups)}")

    ids, scenes, keys = load_scenes(h5_path, limit=limit)
    id2split = {}
    for name, members in groups.items():
        for s in members:
            id2split[s] = name

    tagged = [(i, s, k, id2split.get(i)) for i, s, k in zip(ids, scenes, keys)]
    untagged = [t for t in tagged if t[3] is None]
    tagged = [t for t in tagged if t[3] is not None]
    if not tagged:
        raise Unusable(
            f"划分文件里的 id 与 H5 里的样本名没有任何交集 —— 无法判定泄漏。\n"
            f"  H5 样本名示例: {ids[:3]}\n"
            f"  划分 id 示例  : {sorted(id2split)[:3]}")

    hs = {t[0]: (hash_with_dihedral(t[1], fn) if dihedral else [fn(t[1])])
          for t in tagged}
    plain = {t[0]: fn(t[1]) for t in tagged}

    pairs = []
    for a in range(len(tagged)):
        for b in range(a + 1, len(tagged)):
            ia, _, ka, sa = tagged[a]
            ib, _, kb, sb = tagged[b]
            if sa == sb:
                continue                     # 同 split 内重复不是泄漏
            d = min_dist_dihedral(hs[ia], plain[ib])
            if d <= threshold:
                pairs.append({"id_a": ia, "split_a": sa, "src_a": ka,
                              "id_b": ib, "split_b": sb, "src_b": kb,
                              "distance": d,
                              "same_source_key": ka == kb})
    pairs.sort(key=lambda x: x["distance"])
    return {
        "check": "leakage", "h5": h5_path, "splits": splits_path,
        "threshold": threshold, "hasher": hasher, "dihedral": dihedral,
        "n_scenes_checked": len(tagged),
        "n_scenes_not_in_splits": len(untagged),
        "split_sizes": {k: sum(1 for t in tagged if t[3] == k) for k in groups},
        "n_leak_pairs": len(pairs),
        "leak_pairs": pairs[:200],
        "verdict": "PASS" if not pairs else "FAIL",
    }


# ══════════════════════════════════════════════════════════════════════
# 检查 2:跨数据集重复
# ══════════════════════════════════════════════════════════════════════

def check_cross(canonical_h5: str, other_h5s: Sequence[str], *, threshold: int,
                hasher: str = PRIMARY, dihedral: bool = True,
                limit: Optional[int] = None) -> Dict:
    fn = HASHERS[hasher]
    c_ids, c_scenes, c_keys = load_scenes(canonical_h5, limit=limit)
    c_hash = [fn(s) for s in c_scenes]

    per: List[Dict] = []
    for o in other_h5s:
        o_ids, o_scenes, o_keys = load_scenes(o, limit=limit)
        hits = []
        for oi, os_, ok in zip(o_ids, o_scenes, o_keys):
            qh = hash_with_dihedral(os_, fn) if dihedral else [fn(os_)]
            bd, bj = None, None
            for j, ch in enumerate(c_hash):
                d = min_dist_dihedral(qh, ch)
                if bd is None or d < bd:
                    bd, bj = d, j
            if bd is not None and bd <= threshold:
                hits.append({"other_id": oi, "other_src": ok,
                             "canonical_id": c_ids[bj], "canonical_src": c_keys[bj],
                             "distance": bd})
        rate = len(hits) / len(o_ids)
        lo, hi = wilson(len(hits), len(o_ids))
        if rate == 0:
            verdict, why = "PASS", "未检出可由哈希识别的重复(仅下界)"
        elif rate <= CROSS_DUP_CEILING:
            verdict, why = "WARN", f"重复率 {rate:.4f} ≤ {CROSS_DUP_CEILING} → 剔除后重算 gap"
        else:
            verdict, why = "FAIL", (
                f"重复率 {rate:.4f} > {CROSS_DUP_CEILING} → 按 R2 该 tier 不得作泛化证据")
        per.append({
            "other_h5": o, "n_other_scenes": len(o_ids),
            "n_canonical_scenes": len(c_ids),
            "n_dup": len(hits), "dup_rate": rate, "dup_rate_ci95": [lo, hi],
            "verdict": verdict, "reason": why,
            "examples": sorted(hits, key=lambda x: x["distance"])[:20],
        })
    worst = "PASS"
    for r in per:
        if r["verdict"] == "FAIL":
            worst = "FAIL"
        elif r["verdict"] == "WARN" and worst == "PASS":
            worst = "WARN"
    return {"check": "cross", "canonical_h5": canonical_h5,
            "threshold": threshold, "hasher": hasher, "dihedral": dihedral,
            "per_tier": per, "verdict": worst,
            "ceiling": CROSS_DUP_CEILING}


# ══════════════════════════════════════════════════════════════════════
# 检查 3:拼接接缝
# ══════════════════════════════════════════════════════════════════════

def expected_seams(scene_hw: Tuple[int, int], src_hw: Tuple[int, int]) -> Dict:
    """复刻建库器的网格算法(build_dataset_from_benchmark.py:480-483)。

        num_tiles_h = ceil(H / img_h);  tile_h = H // num_tiles_h
    """
    H, W = scene_hw
    ih, iw = src_hw
    nth = int(np.ceil(H / ih))
    ntw = int(np.ceil(W / iw))
    th, tw = H // nth, W // ntw
    return {
        "num_tiles_h": nth, "num_tiles_w": ntw,
        "tile_h": th, "tile_w": tw,
        "row_seams": [th * k for k in range(1, nth)],
        "col_seams": [tw * k for k in range(1, ntw)],
        "stitched": nth * ntw > 1,
    }


def seam_scores(scene: np.ndarray, rows: Sequence[int],
                cols: Sequence[int]) -> Dict:
    """接缝处相邻行/列差分,相对非接缝位置分布的 z 分数。

    用 median/MAD 而不是 mean/std:接缝本身就是离群点,
    用 mean/std 会被它自己抬高基线、把 z 压小(自我掩盖)。
    """
    a = np.asarray(scene, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    H, W = a.shape

    def _z(diff: np.ndarray, at: Sequence[int]) -> Optional[Dict]:
        at = [i for i in at if 1 <= i < len(diff) + 1]
        if not at:
            return None
        prof = diff.copy()
        mask = np.ones(len(prof), bool)
        for i in at:
            for j in range(max(0, i - 2), min(len(prof), i + 1)):
                mask[j] = False           # 接缝及紧邻不进基线
        base = prof[mask]
        med = float(np.median(base))
        mad = float(np.median(np.abs(base - med)))
        sigma = mad * 1.4826 if mad > 0 else float(base.std()) or 1e-9
        vals = {int(i): float(prof[i - 1]) for i in at}
        return {"z": {int(i): (v - med) / sigma for i, v in vals.items()},
                "value": vals, "baseline_median": med, "baseline_sigma": sigma}

    row_diff = np.abs(np.diff(a, axis=0)).mean(axis=1)   # 长 H-1;索引 i-1 = 行 i 与 i-1 之差
    col_diff = np.abs(np.diff(a, axis=1)).mean(axis=0)
    return {"scene_hw": [H, W], "rows": _z(row_diff, rows), "cols": _z(col_diff, cols)}


def check_seam(h5_path: str, *, src_hw: Optional[Tuple[int, int]] = None,
               limit: Optional[int] = 50) -> Dict:
    ids, scenes, _ = load_scenes(h5_path, limit=limit)
    H, W = scenes[0].shape[:2]
    if src_hw is None:
        src_hw = (256, 256)     # 拼接 tier 的源尺寸;直裁 tier 会算出 1×1 网格
    exp = expected_seams((H, W), src_hw)
    if not exp["stitched"]:
        return {"check": "seam", "h5": h5_path, "grid": exp,
                "verdict": "N/A",
                "reason": f"该尺寸下网格为 1×1(源 {src_hw} ≥ 场景 {(H, W)}),"
                          f"无拼接接缝可查 —— 直裁 tier 属于这种"}
    zs_r, zs_c, per = [], [], []
    for i, s in zip(ids, scenes):
        r = seam_scores(s, exp["row_seams"], exp["col_seams"])
        mr = max(r["rows"]["z"].values()) if r["rows"] else None
        mc = max(r["cols"]["z"].values()) if r["cols"] else None
        if mr is not None:
            zs_r.append(mr)
        if mc is not None:
            zs_c.append(mc)
        per.append({"id": i, "max_row_z": mr, "max_col_z": mc})
    med_r = float(np.median(zs_r)) if zs_r else None
    med_c = float(np.median(zs_c)) if zs_c else None
    worst = max([v for v in (med_r, med_c) if v is not None], default=0.0)
    frac = float(np.mean([
        max([v for v in (p["max_row_z"], p["max_col_z"]) if v is not None],
            default=0.0) >= SEAM_Z_GATE for p in per]))
    return {
        "check": "seam", "h5": h5_path, "grid": exp,
        "n_scenes": len(ids),
        "median_max_row_z": med_r, "median_max_col_z": med_c,
        "seam_z": worst, "frac_scenes_detectable": frac,
        "gate": SEAM_Z_GATE,
        "verdict": "SEAM_DETECTABLE" if worst >= SEAM_Z_GATE else "SEAM_NOT_DETECTABLE",
        "implication": (
            f"接缝可检出(z={worst:.1f} ≥ {SEAM_Z_GATE},{frac:.0%} 的场景可检出)"
            "→ 该 tier 的 gap 混杂人造伪影,不是纯域移。按 R3 必须声明,"
            "或改用零拼接 tier 作主泛化证据。"
            if worst >= SEAM_Z_GATE else
            f"接缝不可检出(z={worst:.1f} < {SEAM_Z_GATE})→ 可与直裁 tier 直接比较。"),
        "per_scene": per[:20],
    }


# ══════════════════════════════════════════════════════════════════════
# 自检
# ══════════════════════════════════════════════════════════════════════

def _fractal_rgb(seed: int, n: int = 512, beta: float = 1.8) -> np.ndarray:
    """1/f^beta 谱的合成纹理 —— 自然/航拍影像的频谱近似长这样。

    自检夹具必须在**频谱形状**上像真数据,否则测的是夹具不是工具:
    纯噪声(白谱)与纯渐变(全在 DC)都会给出误导性的哈希行为。
    """
    r = np.random.default_rng(seed)
    fy, fx = np.mgrid[-(n // 2):n - n // 2, -(n // 2):n - n // 2]
    f = np.sqrt(fy.astype(np.float64) ** 2 + fx.astype(np.float64) ** 2)
    f[f == 0] = 1.0
    chans = []
    for c in range(3):
        F = np.fft.fftshift(np.fft.fft2(r.normal(size=(n, n))))
        im = np.real(np.fft.ifft2(np.fft.ifftshift(F / f ** (beta / 2.0))))
        im -= im.min()
        im /= (im.max() or 1.0)
        chans.append((im * 255.0).astype(np.uint8))
    return np.stack(chans, axis=-1)


def _self_test() -> int:
    P = F = 0

    def ck(name, cond, extra=""):
        nonlocal P, F
        if cond:
            P += 1
            print(f"  ✓ {name}")
        else:
            F += 1
            print(f"  ✗ {name}  {extra}")

    print("\n[1] DCT 正确性")
    try:
        from scipy.fft import dct as sdct
        x = np.random.default_rng(0).normal(size=(32, 32))
        mine = dct2(x)
        ref = sdct(sdct(x, type=2, norm="ortho", axis=0),
                   type=2, norm="ortho", axis=1)
        ck("dct2 与 scipy.fft.dct(type=2,norm=ortho) 逐位一致",
           np.allclose(mine, ref, atol=1e-10),
           f"max|Δ|={np.abs(mine - ref).max():.2e}")
    except ImportError:
        M = dct_matrix(8)
        ck("DCT 矩阵正交 (MᵀM=I)", np.allclose(M.T @ M, np.eye(8), atol=1e-12))
        print("      (scipy 缺失 → 退化为正交性检查)")
    M = dct_matrix(16)
    ck("DCT 矩阵正交 (n=16)", np.allclose(M @ M.T, np.eye(16), atol=1e-12))

    print("\n[2] 哈希基本性质")
    rng = np.random.default_rng(7)
    img = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
    try:
        _require_pil()
        have_pil = True
    except Unusable:
        have_pil = False
    if not have_pil:
        print("  [skip] 无 Pillow,跳过全部哈希用例")
        return 0 if F == 0 else 7
    for nm, fn in HASHERS.items():
        h = fn(img)
        ck(f"{nm} 位数 {len(h)}", len(h) in (63, 64))
        ck(f"{nm} 自距离 0", hamming(h, fn(img)) == 0)
    ck("pHash 去掉了 DC(63 位)", len(phash(img)) == 63)

    print("\n[3] 对重发布变换稳定(正例)")
    # 夹具必须像真实影像:自然图像的功率谱近似 1/f^beta。
    # 早先这里用线性渐变,结果正例距离 22-30 —— 因为渐变的能量全在 DC,
    # 去掉 DC 后 63 个位是噪声。那不是哈希实现的 bug,是**夹具不像真数据**,
    # 但它暴露了工具的真缺陷(低纹理图无判别力)→ 已加 hash_stability 闸,
    # 并在 [3b] 单独验证低纹理图会被正确判为不可靠。
    scene = _fractal_rgb(11)
    h0 = phash(scene)
    for kind in REPUBLISH_KINDS:
        d = hamming(phash(_simulate_republish(scene, kind)), h0)
        ck(f"pHash({kind}) 距离 {d} ≤ 12", d <= 12)

    print("\n[3b] 低纹理图被判为哈希不可靠")
    grad = np.zeros((512, 512, 3), np.uint8)
    yy, xx = np.mgrid[0:512, 0:512]
    grad[..., 0] = (yy / 511 * 170).astype(np.uint8)
    grad[..., 1] = (xx / 511 * 170).astype(np.uint8)
    grad[..., 2] = ((xx + yy) / 1022 * 200).astype(np.uint8)
    s_grad = hash_stability(grad, phash, seed=3)
    s_tex = hash_stability(scene, phash, seed=3)
    ck(f"渐变图 ±1LSB 翻 {s_grad} 位 ≥ 25%×63,判为不可靠",
       s_grad >= STABILITY_PREFILTER_FRAC * 63)
    ck(f"纹理图 ±1LSB 只翻 {s_tex} 位,判为可靠",
       s_tex < STABILITY_PREFILTER_FRAC * 63)
    ck("低纹理图上 正例距离 与 异源距离 确实分不开(这才是要筛掉它的理由)",
       hamming(phash(_simulate_republish(grad, "jpeg70")), phash(grad))
       >= 0.5 * hamming(phash(_fractal_rgb(12)), phash(grad)))

    print("\n[4] 对不同图敏感(负例)")
    a = _fractal_rgb(21)
    b = _fractal_rgb(22)
    ck(f"两张独立纹理图 pHash 距离 {hamming(phash(a), phash(b))} ≥ 15",
       hamming(phash(a), phash(b)) >= 15)

    print("\n[5] 二面体变换")
    ck("dihedral_8 出 8 个", len(dihedral_8(scene)) == 8)
    flipped = np.fliplr(scene)
    d_plain = hamming(phash(flipped), h0)
    d_dih = min_dist_dihedral(hash_with_dihedral(flipped, phash), h0)
    ck(f"翻转副本:朴素距离 {d_plain} → 二面体后 {d_dih}", d_dih <= d_plain and d_dih == 0)

    print("\n[6] 接缝网格复刻建库器")
    e = expected_seams((512, 512), (256, 256))
    ck("512/256 → 2×2 网格", e["num_tiles_h"] == 2 and e["num_tiles_w"] == 2)
    ck("接缝在第 256 行/列", e["row_seams"] == [256] and e["col_seams"] == [256])
    e6 = expected_seams((512, 512), (600, 600))
    ck("600² 源 → 1×1 网格、无接缝", not e6["stitched"])

    print("\n[7] 接缝检测(注入真接缝)")
    # 用两块统计不同的纹理拼 2×2 —— 这才是真拼接的样子
    def _tex(seed, n=256):
        r = np.random.default_rng(seed)
        base = r.normal(128, 12, size=(n, n))
        return np.clip(base, 0, 255)
    quad = np.zeros((512, 512), float)
    for k, (r0, c0) in enumerate([(0, 0), (0, 256), (256, 0), (256, 256)]):
        quad[r0:r0 + 256, c0:c0 + 256] = _tex(100 + k) + k * 25
    sc = seam_scores(quad, [256], [256])
    zr, zc = sc["rows"]["z"][256], sc["cols"]["z"][256]
    ck(f"拼接图接缝 z 行={zr:.1f} 列={zc:.1f} ≥ {SEAM_Z_GATE}",
       zr >= SEAM_Z_GATE and zc >= SEAM_Z_GATE)
    single = _tex(999, 512)
    sc2 = seam_scores(single, [256], [256])
    z2 = max(sc2["rows"]["z"][256], sc2["cols"]["z"][256])
    ck(f"单块同质图无接缝 z={z2:.1f} < {SEAM_Z_GATE}", z2 < SEAM_Z_GATE)

    print("\n[8] MAD 基线不自我掩盖")
    # 若用 mean/std,接缝自身会抬高 sigma、把 z 压下去。这里验证 MAD 路线不会。
    a2 = np.asarray(quad, float)
    rd = np.abs(np.diff(a2, axis=0)).mean(axis=1)
    z_mean = (rd[255] - rd.mean()) / (rd.std() or 1e-9)
    ck(f"MAD 口径 z={zr:.1f} 明显大于 mean/std 口径 z={z_mean:.1f}", zr > z_mean)

    print("\n[9] 标定 fail-closed")
    try:
        calibrate([(rng.random((64, 64, 3)) * 255).astype(np.uint8)], max_scenes=1)
        ck("单场景标定应抛", False, "没抛")
    except Unusable:
        ck("单场景标定→Unusable(样本不足不硬撑)", True)
    # 全同图:这**不是**"分不开",而是这批里真有重复 —— 诊断必须说对,
    # 因为两者的处置完全不同(前者=换检测器,后者=去重后重划分)。
    same = [scene.copy() for _ in range(6)]
    try:
        calibrate(same, max_scenes=6, dihedral=False)
        ck("全同图应抛", False, "没抛,给了个假阈值")
    except Unusable as e:
        ck("全同图→Unusable 且诊断为『真有重复』而非『分不开』",
           "真有重复" in str(e) and "@d=" in str(e), str(e)[:100])
        ck("并给出最近的可疑对,可据此定位", "(0,1)@d=0" in str(e), str(e)[:160])
    # allow_exclude 不得把真正不可分的数据救回来 —— 排除只针对"少数真重复",
    # 全同图排除完就没有负例了,必须仍然抛。
    try:
        calibrate(same, max_scenes=6, dihedral=False, allow_exclude=True)
        ck("allow_exclude 不救全同图", False, "把不可分数据救成了阈值")
    except Unusable:
        ck("allow_exclude 不救全同图(排除后负例不足→仍抛)", True)
    # 反过来:少数真重复 + 多数可分 → allow_exclude 应成功,且把被排除的对登记出来
    mixed = [_fractal_rgb(4000 + k) for k in range(10)]
    mixed.append(mixed[0].copy())                      # 注入 1 对真重复
    try:
        cal_x = calibrate(mixed, max_scenes=11, dihedral=False, allow_exclude=True)
        ex = cal_x.get("excluded_suspicious_pairs", [])
        ck(f"少数重复 + allow_exclude → 标定成功(阈值 {cal_x['threshold']})", True)
        ck(f"被排除的 {len(ex)} 对被登记为重复发现(未隐藏)",
           len(ex) >= 1 and any({e2['scene_idx_a'], e2['scene_idx_b']} == {0, 10} for e2 in ex),
           str(ex)[:120])
    except Unusable as e:
        ck("少数重复 + allow_exclude → 标定成功", False, str(e)[:100])

    print("\n[9b] 全低纹理输入 → 拒绝标定")
    try:
        calibrate([grad.copy() for _ in range(8)], max_scenes=8, dihedral=False)
        ck("全低纹理应抛", False, "没抛,会给出一个没有判别力的阈值")
    except Unusable as e:
        ck("全低纹理→Unusable(拒绝给阈值)",
           "无判别力" in str(e) or "不可分" in str(e) or "重叠" in str(e), str(e)[:80])

    print("\n[10] 标定在可分数据上给出阈值")
    # 24 张互不相同的 1/f 纹理 —— 频谱形状与真实航拍影像同量级。
    # (早先用低频正弦组合,f1,f2∈[2,9) 只有 49 种组合、且低纹理,
    #  结果负例最小距离 0 —— 池子里本身就有重复,标定必然失败。)
    pool = [_fractal_rgb(1000 + k) for k in range(24)]
    try:
        cal = calibrate(pool, max_scenes=24, dihedral=False)
        ck(f"标定成功 阈值={cal['threshold']} TPR={cal['tpr']:.2f} FPR={cal['fpr']:.3f}",
           cal["tpr"] >= CALIB_MIN_TPR and cal["fpr"] <= CALIB_MAX_FPR)
        ck("报了 TPR/FPR 的 Wilson CI",
           len(cal["tpr_ci95"]) == 2 and cal["tpr_ci95"][0] <= cal["tpr"])
        ck("如实登记了盲区", "同区域不同取景" in cal["blind_spot"])
        ck("报了低纹理剔除数与阈值下盲场景数",
           "n_dropped_low_texture" in cal["stability_prefilter"]
           and "n_blind_at_threshold" in cal["stability_prefilter"])
    except Unusable as e:
        ck("标定成功", False, str(e)[:120])

    print("\n[11] 输入不可用即拒跑")
    try:
        load_scenes("/nonexistent/xx.h5")
        ck("缺文件应抛", False)
    except Unusable:
        ck("缺文件→Unusable", True)

    print("\n[12] 预注册规则齐备")
    for tag in ("R1", "R2", "R3", "R4"):
        ck(f"PREREG 含 {tag}", tag in PREREG)
    ck("R1 上限是 0 对", "容忍上限 = **0 对**" in PREREG)
    ck("盲区写在文件顶部", "同一地理区域被**不同取景**采集" in __doc__)

    print(f"\n汇总: {P} PASS / {F} FAIL")
    return 0 if F == 0 else 7


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="近重复 / 跨库重复 / 拼接接缝 检测",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", choices=["leakage", "cross", "seam"])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--h5")
    ap.add_argument("--other", action="append", default=[])
    ap.add_argument("--splits")
    ap.add_argument("--calib-h5",
                    help="在**另一个**库上标定(推荐:待查库若真有重复会污染负例)")
    ap.add_argument("--calib-allow-exclude", action="store_true",
                    help="标定时排除疑似重复的异源对(它们会被登记为重复发现)")
    ap.add_argument("--threshold", type=int,
                    help="哈希距离阈值。不给则先自动标定(推荐)")
    ap.add_argument("--hasher", choices=sorted(HASHERS), default=PRIMARY)
    ap.add_argument("--no-dihedral", action="store_true",
                    help="关闭 8 变换匹配(会漏掉翻转/旋转副本)")
    ap.add_argument("--limit", type=int, help="每库最多取多少唯一场景")
    ap.add_argument("--src-hw", type=int, nargs=2, metavar=("H", "W"),
                    help="源图尺寸,用于推接缝网格(默认 256 256)")
    ap.add_argument("--out", help="结果 JSON 落盘路径")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    if a.self_test:
        return _self_test()
    if not (a.check or a.calibrate):
        ap.error("要 --check 或 --calibrate 之一(或 --self-test)")
    if not a.h5:
        ap.error("--h5 必填")

    print(PREREG)

    try:
        dih = not a.no_dihedral
        cal = None
        if a.calibrate or a.threshold is None:
            print("=" * 72)
            print("标定阈值(正例=同源重发布变体 / 负例=异源两两)")
            print("=" * 72)
            calib_src = a.calib_h5 or a.h5
            if a.calib_h5:
                print(f"标定库: {calib_src}(与待查库不同 —— 避免待查库自身的重复污染负例)")
            _, scenes, _ = load_scenes(calib_src, limit=a.limit)
            cal = calibrate(scenes, hasher=a.hasher, dihedral=dih,
                            allow_exclude=a.calib_allow_exclude)
            if cal.get("excluded_suspicious_pairs"):
                print(f"\n  ⚠ {cal['excluded_note']}")
                for e in cal["excluded_suspicious_pairs"][:10]:
                    print(f"      场景 #{e['scene_idx_a']} ↔ #{e['scene_idx_b']}  d={e['distance']}")
            print(json.dumps(cal, ensure_ascii=False, indent=2))
            thr = cal["threshold"]
            print(f"\n→ 采用阈值 {thr}(TPR {cal['tpr']:.2f} / FPR {cal['fpr']:.3f})")
        else:
            thr = a.threshold
            print(f"→ 采用**人工指定**阈值 {thr}。注意:未标定的阈值没有 TPR/FPR 保证,"
                  f"论文里引用前请跑一次 --calibrate。")

        res: Dict = {"prereg": PREREG.strip(), "calibration": cal}

        if a.check == "leakage":
            if not a.splits:
                ap.error("--check leakage 需要 --splits")
            r = check_leakage(a.h5, a.splits, threshold=thr, hasher=a.hasher,
                              dihedral=dih, limit=a.limit)
            res["leakage"] = r
            print("\n" + "=" * 72)
            print(f"划分泄漏: {r['verdict']}   跨 split 内容重复 {r['n_leak_pairs']} 对"
                  f"(检查了 {r['n_scenes_checked']} 个唯一场景)")
            if r["n_scenes_not_in_splits"]:
                print(f"  [warn] {r['n_scenes_not_in_splits']} 个场景不在划分文件里,未参与判定")
            for p in r["leak_pairs"][:10]:
                print(f"    d={p['distance']:2d}  {p['split_a']}/{p['id_a']} ↔ "
                      f"{p['split_b']}/{p['id_b']}"
                      f"{'  [同 source_key!]' if p['same_source_key'] else ''}")
            rc = 0 if r["verdict"] == "PASS" else 4

        elif a.check == "cross":
            if not a.other:
                ap.error("--check cross 需要至少一个 --other")
            r = check_cross(a.h5, a.other, threshold=thr, hasher=a.hasher,
                            dihedral=dih, limit=a.limit)
            res["cross"] = r
            print("\n" + "=" * 72)
            print(f"{'tier':<34} {'场景':>5} {'重复':>5} {'重复率':>9} "
                  f"{'95%CI':>17}  判定")
            for t in r["per_tier"]:
                lo, hi = t["dup_rate_ci95"]
                print(f"{Path(t['other_h5']).name:<34} {t['n_other_scenes']:>5} "
                      f"{t['n_dup']:>5} {t['dup_rate']:>9.4f} "
                      f"[{lo:.4f},{hi:.4f}]  {t['verdict']}")
                print(f"    {t['reason']}")
            rc = {"PASS": 0, "WARN": 0, "FAIL": 5}[r["verdict"]]

        else:  # seam
            r = check_seam(a.h5, src_hw=tuple(a.src_hw) if a.src_hw else None,
                           limit=a.limit or 50)
            res["seam"] = r
            print("\n" + "=" * 72)
            print(f"拼接接缝: {r['verdict']}")
            print(f"  网格 {r['grid']['num_tiles_h']}×{r['grid']['num_tiles_w']}"
                  f"  接缝行 {r['grid']['row_seams']} 列 {r['grid']['col_seams']}")
            if r["verdict"] != "N/A":
                print(f"  seam_z = {r['seam_z']:.2f} (门 {r['gate']})"
                      f"  可检出场景占比 {r['frac_scenes_detectable']:.0%}")
            print(f"  {r.get('implication', r.get('reason', ''))}")
            rc = 0

        if a.out:
            Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                              default=str))
            print(f"\n结果已写入 {a.out}")
        return rc

    except Unusable as e:
        print(f"\n[FAIL-CLOSED] {e}", file=sys.stderr)
        # 标定失败 → 3;其余输入问题 → 6。两者都**不是** 0。
        return 3 if "标定" in str(e) or "R4" in str(e) else 6


if __name__ == "__main__":
    sys.exit(main())
