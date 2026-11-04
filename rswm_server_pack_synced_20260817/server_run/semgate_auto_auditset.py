#!/usr/bin/env python3
# =============================================================================
# semgate_auto_auditset.py — 语义门审计集:**标签由构造保证**,零真人参与
#
# 用户明确要求语义门审计不能依赖人工标注。人工标注问的是"这个窗里还有没有声称
# 的语义",没人就答不了。但有一类问题**标签由构造保证**,不需要人:
#
#   正例(标签必为真):某类标图的【中心窗】。数据集的类标是场景级给的,
#       所以"该窗属于该池"为真。
#   硬负例(标签必为假):取【其它场景池】图片的窗,却声称本池类别。
#       标签为假是构造保证的 —— 不需要任何人看图。
#   分级序列(测决策边界,不给硬标签):同一张图,窗口从中心逐步滑到边角。
#       判定**应当单调劣化**;不单调本身就是门不可靠的证据。
#   扰动组(完全不需要标签):同一窗 + 裁剪抖动/翻转/JPEG/亮度。
#       判定应当不变,翻转率就是不稳定度。
#
# ⚠️ 必须与结论一起写进论文的局限:
#   构造保真值验的是"**能否区分池 A 与池 B**"。这是"裁窗后是否仍保有声称语义"
#   的**必要非充分**条件 —— 硬负例来自不同池,比真实的失败模式
#   (**同池但主体已滑出窗外**,例如 00009 River 窗里只剩草)容易得多。
#   所以这套数字**不是**人工检查的等价替代。分级序列是为了逼近后者,
#   但它只给单调性证据,不给"第几档开始该拒"的绝对答案。
#
# 用法:
#   python server_run/semgate_auto_auditset.py --gt-h5 $WORK/train_geo.h5 \
#       -o $WORK/semgate_auditset.json [--per-pool 40] [--allow-synthetic-arbiters]
#   python server_run/semgate_auto_auditset.py --self-test
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "lora_remotesense"))
sys.path.insert(0, str(_HERE))

import semantic_gate as sg                       # noqa: E402
from semgate_arbiters import (                   # noqa: E402
    KappaUnavailable, assert_kappa_ok, bootstrap_ci, build_arbiters,
    cohens_kappa, fleiss_kappa, perturbation_flip_rate,
    probe_available_arbiters, wilson_ci, KAPPA_GATE_DEFAULT,
)

# 调参集 —— 审计集必须与它无交集,否则测的是"在调参样本上过拟合得多好"
CALIB_LABELS = _HERE / "calibration_labels_smoke_v261.json"


class AuditSetError(RuntimeError):
    """审计集无法构造。fail-closed,不返回半成品。"""


# =============================================================================
# 源身份:决定"与调参集无交集"能判到什么粒度
# =============================================================================

def source_key(attrs) -> Tuple[str, ...]:
    """样本的源身份键。粒度越细越好,但只能用**实际存在**的字段。

    AID 路径有 source_dataset/source_class/source_index/source_path;
    合成源(--no-hf)只有 source_scene_category/source_scene_id。
    缺字段时**降级并如实标注**,不假装判到了更细的粒度 ——
    因为"按 sample_id 无交集"是个假保证:同一源图裁出的不同样本仍然相关。
    """
    def g(k):
        v = attrs.get(k)
        if v is None:
            return None
        return v.decode() if isinstance(v, bytes) else str(v)

    for keys in (("source_dataset", "source_class", "source_index"),
                 ("source_dataset", "source_index"),
                 ("source_scene_category", "source_scene_id")):
        vals = [g(k) for k in keys]
        if all(v is not None for v in vals):
            return tuple([":".join(keys)] + vals)
    raise AuditSetError(
        "样本没有任何可用的源身份字段(期望 source_dataset/source_class/"
        "source_index 或 source_scene_category/source_scene_id)。"
        "无法保证审计集与调参集无交集 —— 拒绝构造。")


def source_key_granularity(k: Tuple[str, ...]) -> str:
    return k[0]


def calib_source_keys(h5_path: Optional[Path]) -> Tuple[set, str]:
    """调参集那 20 个样本的源身份键。

    标注文件只记了 sample_id 与 category,不记源身份。所以要从**它当时用的
    那个 H5** 里反查。拿不到那个 H5 时:如实返回空集 + 说明,并把
    "无法证明无交集"作为一个必须报告的局限,而不是当成"已验证无交集"。
    """
    if not CALIB_LABELS.is_file():
        return set(), f"找不到调参标注 {CALIB_LABELS.name} —— 无法核对交集"
    labels = json.loads(CALIB_LABELS.read_text())
    ids = set(labels.get("samples", {}))
    src_h5 = labels.get("_source_h5")
    if h5_path is None or not Path(h5_path).is_file():
        return set(), (f"调参集 {len(ids)} 个样本的源身份需要它当时的 H5"
                       f"({src_h5!r})才能反查;未提供 → **无法证明无交集**")
    import h5py
    keys = set()
    with h5py.File(str(h5_path), "r") as f:
        for sid in ids:
            if sid in f["data"]:
                keys.add(source_key(dict(f["data"][sid].attrs)))
    if not keys:
        return set(), (f"提供的 H5 里找不到调参集的任何 sample_id "
                       f"(它们来自 {src_h5!r})→ **无法证明无交集**")
    return keys, f"从 {Path(h5_path).name} 反查到 {len(keys)} 个调参源身份键"


# =============================================================================
# 构造保真值的三组样本
# =============================================================================

def _sha256_file(p, chunk: int = 1 << 22) -> str:
    """分块求 SHA256。

    早先写的是 `hashlib.sha256(Path(p).read_bytes())` —— 那会把整个 H5 读进内存。
    canonical 库带 full_scene,12 个样本就 1.3 GB,4000 个样本会是几百 GB:
    实测直接 OOM 被 kill(exit 137)。求哈希从来不需要一次性持有全部字节。
    """
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _center_window(scene: np.ndarray, win_hw: Tuple[int, int]) -> np.ndarray:
    H, W = scene.shape[:2]
    h, w = win_hw
    y0, x0 = (H - h) // 2, (W - w) // 2
    return scene[y0:y0 + h, x0:x0 + w]


def _offset_origin(scene_hw: Tuple[int, int], win_hw: Tuple[int, int],
                   frac: float) -> Tuple[int, int]:
    """偏移后窗口左上角坐标。抽成独立函数是为了可测:

    真正的不变量是**窗口位置随 frac 单调移动**,与场景内容无关。
    早先的自测断言"像素差随 frac 单调增",那只在平滑场景上成立 ——
    合成场景用了 `arange % 251`,模回绕让像素差非单调,断言假红。
    """
    H, W = scene_hw
    h, w = win_hw
    cy0, cx0 = (H - h) // 2, (W - w) // 2
    y0 = int(round(cy0 + frac * (H - h - cy0)))
    x0 = int(round(cx0 + frac * (W - w - cx0)))
    return max(0, min(H - h, y0)), max(0, min(W - w, x0))


def _offset_window(scene: np.ndarray, win_hw: Tuple[int, int],
                   frac: float) -> np.ndarray:
    """从中心按 frac∈[0,1] 向右下滑动。frac=0 是中心,frac=1 贴到边界。"""
    h, w = win_hw
    y0, x0 = _offset_origin(scene.shape[:2], win_hw, frac)
    return scene[y0:y0 + h, x0:x0 + w]


def _tile(win: np.ndarray, n: int) -> List[np.ndarray]:
    """把单个窗复制成 n 帧 —— 语义门是逐帧判定,内容不变时 n 帧同判。

    这不是"伪造视频":审计问的是"这个窗的语义还在不在",与运动无关。
    用 FROZEN_P0_FRAMES 帧是为了让 frac 的分母与生产判据一致
    (阈值 0.2777… = 9 帧里 ≥3 帧同意)。
    """
    return [np.ascontiguousarray(win) for _ in range(n)]


def _perturb(win: np.ndarray) -> Dict[str, np.ndarray]:
    """标签保持的扰动。判定应当不变 —— 变了就是判据不稳。"""
    h, w = win.shape[:2]
    out = {}
    # 裁剪抖动:向内缩 4px 再放回原尺寸(最近邻,不引依赖)
    if h > 12 and w > 12:
        inner = win[4:h - 4, 4:w - 4]
        yi = (np.linspace(0, inner.shape[0] - 1, h)).astype(int)
        xi = (np.linspace(0, inner.shape[1] - 1, w)).astype(int)
        out["jitter4px"] = inner[yi][:, xi]
    out["fliplr"] = win[:, ::-1].copy()
    out["flipud"] = win[::-1, :].copy()
    out["bright+8"] = np.clip(win.astype(np.int16) + 8, 0, 255).astype(np.uint8)
    # JPEG 重压(有 cv2 才做 —— 缺了就少一个扰动,如实记录,不假装做过)
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", win[:, :, ::-1],
                               [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            out["jpeg70"] = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    except Exception:
        pass
    return out


def build_auditset(gt_h5: Path, *, per_pool: int, graded_steps: int,
                   calib_h5: Optional[Path], seed: int) -> Dict:
    """构造审计集。**不落盘图像** —— 只落索引与构造参数,评审时按需重建。

    这样审计集本身是可复现的(同 H5 + 同 seed → 同集合),
    而且不会把派生像素混进仓库(数据许可上也更干净)。
    """
    import h5py
    rng = np.random.default_rng(seed)
    gated = list(sg.GATED_CATEGORIES)
    all_pools = list(sg._TAXONOMY["scene_categories"])
    ungated = [c for c in all_pools if c not in gated]

    calib_keys, calib_note = calib_source_keys(calib_h5)

    with h5py.File(str(gt_h5), "r") as f:
        md = dict(f["metadata"].attrs)
        win_hw = tuple(int(x) for x in md.get("window_size", (128, 222)))
        by_pool: Dict[str, List[str]] = {}
        skeys: Dict[str, Tuple[str, ...]] = {}
        for sid in f["data"]:
            a = dict(f["data"][sid].attrs)
            cat = a.get("source_scene_category")
            cat = cat.decode() if isinstance(cat, bytes) else str(cat)
            k = source_key(a)
            if k in calib_keys:
                continue                     # 与调参集同源 → 排除
            by_pool.setdefault(cat, []).append(sid)
            skeys[sid] = k
        gran = source_key_granularity(next(iter(skeys.values()))) if skeys else "?"

        if not by_pool:
            raise AuditSetError(
                f"{gt_h5} 里没有可用样本(或全部与调参集同源)。")

        # 每池只用**不同源图**的样本,避免同一源图贡献多个"独立"审计项
        for cat in by_pool:
            seen, uniq = set(), []
            for sid in by_pool[cat]:
                if skeys[sid] not in seen:
                    seen.add(skeys[sid]); uniq.append(sid)
            by_pool[cat] = uniq

        items: List[Dict] = []

        # ---- 正例:本池图的中心窗,标签必为真 ----
        for cat in gated + ungated:
            pool = by_pool.get(cat, [])
            take = pool[:per_pool]
            for sid in take:
                items.append({"kind": "positive", "claim_pool": cat,
                              "src_sid": sid, "src_pool": cat,
                              "offset_frac": 0.0, "label": True,
                              "why": "本池类标图的中心窗:场景级该类必然存在"})

        # ---- 硬负例:其它池图的中心窗,却声称本池,标签必为假 ----
        for cat in gated + ungated:
            others = [c for c in all_pools if c != cat and by_pool.get(c)]
            if not others:
                continue
            n_need = min(per_pool, sum(len(by_pool[c]) for c in others))
            picked = 0
            for c in rng.permutation(others):
                for sid in by_pool[str(c)]:
                    if picked >= n_need:
                        break
                    items.append({"kind": "hard_negative", "claim_pool": cat,
                                  "src_sid": sid, "src_pool": str(c),
                                  "offset_frac": 0.0, "label": False,
                                  "why": f"取自 {c} 池的窗却声称 {cat} —— 构造保证为假"})
                    picked += 1
                if picked >= n_need:
                    break

        # ---- 分级序列:同一图窗口从中心滑到边角,不给硬标签 ----
        for cat in gated:
            for sid in by_pool.get(cat, [])[:max(2, per_pool // 4)]:
                for i in range(graded_steps):
                    fr = i / max(1, graded_steps - 1)
                    items.append({"kind": "graded", "claim_pool": cat,
                                  "src_sid": sid, "src_pool": cat,
                                  "offset_frac": float(fr), "label": None,
                                  "graded_group": f"{cat}:{sid}",
                                  "graded_step": i,
                                  "why": "决策边界:判定应随窗口滑离主体单调劣化"})

    n_pos = sum(1 for x in items if x["kind"] == "positive")
    n_neg = sum(1 for x in items if x["kind"] == "hard_negative")
    n_grd = sum(1 for x in items if x["kind"] == "graded")

    # 样本量是否够让每类的 Wilson CI 收窄到可用?反解并报告,不静默
    def _ci_width(n):
        lo, hi = wilson_ci(int(round(0.2 * n)), n)      # 以 p̂≈0.2 估宽度
        return hi - lo
    sizing = {str(n): round(_ci_width(n), 4) for n in (10, 20, 40, 80, 160)}

    return {
        "gt_h5": str(gt_h5),
        "gt_h5_sha256": _sha256_file(gt_h5),
        "window_size": list(win_hw),
        "frames_per_item": sg.FROZEN_P0_FRAMES,
        "seed": seed,
        "gated_categories": gated,
        "ungated_categories": ungated,
        "source_key_granularity": gran,
        "calib_disjoint_note": calib_note,
        "calib_source_keys_found": len(calib_keys),
        "counts": {"positive": n_pos, "hard_negative": n_neg, "graded": n_grd,
                   "total": len(items)},
        "wilson_ci_width_at_phat0.2": sizing,
        "limitations": [
            "构造保真值验的是『能否区分池 A 与池 B』——『裁窗后仍保有声称语义』的"
            "**必要非充分**条件。",
            "硬负例来自不同池,比真实失败模式(同池但主体已滑出窗外)容易得多。",
            "分级序列只给单调性证据,不给『第几档该拒』的绝对答案。",
            "本套数字**不是**人工检查的等价替代;论文里必须分开陈述。",
        ] + ([] if len(calib_keys) else [
            "**未能证明与调参集无交集**:" + calib_note]),
        "items": items,
    }


# =============================================================================
# 跑审计:多仲裁者投票 + κ + 稳定性 + 每类 precision/recall
# =============================================================================

def run_audit(auditset: Dict, *, allow_synthetic: bool,
              kappa_gate: float) -> Dict:
    import h5py
    arbiters = build_arbiters(allow_synthetic_only=allow_synthetic)
    win_hw = tuple(auditset["window_size"])
    nf = int(auditset["frames_per_item"])
    items = auditset["items"]

    votes: Dict[str, List[List[bool]]] = {a.name: [] for a in arbiters}
    stability: Dict[str, List[float]] = {a.name: [] for a in arbiters}

    with h5py.File(auditset["gt_h5"], "r") as f:
        for it in items:
            scene = np.asarray(f["data"][it["src_sid"]]["full_scene"][:])
            if scene.dtype != np.uint8:
                scene = np.clip(scene, 0, 255).astype(np.uint8)
            if scene.shape[2] > 3:
                scene = scene[:, :, :3]
            win = (_center_window(scene, win_hw) if it["offset_frac"] == 0.0
                   else _offset_window(scene, win_hw, it["offset_frac"]))
            frames = _tile(win, nf)
            for a in arbiters:
                per = a.score_fn(frames, it["claim_pool"])
                votes[a.name].append([bool(x) for x in per])
            # 扰动只在正例/硬负例上做(分级序列本身就在动窗口)
            if it["kind"] != "graded":
                pert = _perturb(win)
                for a in arbiters:
                    base = a.score_fn(frames, it["claim_pool"])
                    pv = [a.score_fn(_tile(p, nf), it["claim_pool"])
                          for p in pert.values()]
                    if pv:
                        stability[a.name].append(
                            perturbation_flip_rate(base, pv)["flip_rate"])

    # 逐项的"通过"判定 = frac ≥ 冻结阈值(与生产判据同口径)
    thr = sg.FROZEN_P0_THRESHOLD
    decisions = {name: np.array([np.mean(v) >= thr for v in vlist], dtype=bool)
                 for name, vlist in votes.items()}

    names = [a.name for a in arbiters]
    n_items = len(items)

    # ---- 一致性:Fleiss' κ(全体)+ 两两 Cohen's κ,都带 bootstrap CI ----
    def _mat(idx):
        sub = [decisions[n][idx] for n in names]
        yes = np.sum(np.vstack(sub), axis=0)
        m = len(names)
        return np.stack([m - yes, yes], axis=1).astype(float)

    all_idx = np.arange(n_items)
    try:
        fk = fleiss_kappa(_mat(all_idx))
        fk_lo, fk_hi, fk_drop = bootstrap_ci(
            lambda idx: fleiss_kappa(_mat(idx)), n_items)
    except KappaUnavailable as e:
        return {"kappa_unavailable": str(e), "arbiters": names,
                "n_items": n_items,
                "note": "κ 无法计算 → 按 fail-closed 拒绝输出 precision/recall"}

    pair = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            try:
                k = cohens_kappa(decisions[names[i]], decisions[names[j]])
                lo, hi, _ = bootstrap_ci(
                    lambda idx, a=names[i], b=names[j]:
                        cohens_kappa(decisions[a][idx], decisions[b][idx]),
                    n_items)
                pair[f"{names[i]}|{names[j]}"] = {
                    "kappa": k, "ci95_lo": lo, "ci95_hi": hi}
            except KappaUnavailable as e:
                pair[f"{names[i]}|{names[j]}"] = {"unavailable": str(e)[:120]}

    out = {
        "arbiters": names,
        "arbiter_kinds": {a.name: a.kind for a in arbiters},
        "n_items": n_items,
        "threshold": thr,
        "fleiss_kappa": fk, "fleiss_ci95": [fk_lo, fk_hi],
        "fleiss_bootstrap_dropped": fk_drop,
        "pairwise_cohens_kappa": pair,
        "kappa_gate": kappa_gate,
        "stability": {n: ({"mean_flip_rate": float(np.mean(v)),
                           "max_flip_rate": float(np.max(v)), "n": len(v)}
                          if v else {"unavailable": "无扰动样本"})
                      for n, v in stability.items()},
    }

    # ---- κ 门:不达标即拒绝输出 precision/recall ----
    try:
        assert_kappa_ok(fk, gate=kappa_gate, ci=(fk_lo, fk_hi),
                        what="多仲裁者一致性(Fleiss' κ)")
    except KappaUnavailable as e:
        out["precision_recall"] = None
        out["fail_closed_reason"] = str(e)
        return out

    # ---- 每类 precision / recall / F1 + Wilson CI(用构造保真值)----
    labeled = [(i, it) for i, it in enumerate(items) if it["label"] is not None]
    per_pool = {}
    # 共识判定 = 多数票(仲裁者已通过 κ 门,多数票才有意义)
    maj = np.mean(np.vstack([decisions[n] for n in names]), axis=0) > 0.5
    for cat in auditset["gated_categories"] + auditset["ungated_categories"]:
        idx = [i for i, it in labeled if it["claim_pool"] == cat]
        if not idx:
            continue
        y = np.array([items[i]["label"] for i in idx], dtype=bool)
        p = maj[idx]
        tp = int(np.sum(p & y)); fp = int(np.sum(p & ~y))
        fn = int(np.sum(~p & y)); tn = int(np.sum(~p & ~y))
        prec = tp / (tp + fp) if (tp + fp) else None
        rec = tp / (tp + fn) if (tp + fn) else None
        entry = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(idx),
                 "precision": prec, "recall": rec,
                 "false_reject_rate": (fn / (tp + fn)) if (tp + fn) else None}
        if tp + fp:
            entry["precision_ci95"] = list(wilson_ci(tp, tp + fp))
        if tp + fn:
            entry["recall_ci95"] = list(wilson_ci(tp, tp + fn))
            entry["false_reject_ci95"] = list(wilson_ci(fn, tp + fn))
        if prec is not None and rec is not None and (prec + rec):
            entry["f1"] = 2 * prec * rec / (prec + rec)
        entry["gated"] = cat in auditset["gated_categories"]
        per_pool[cat] = entry
    out["precision_recall"] = per_pool

    # ---- 分级序列的单调性 ----
    groups: Dict[str, List[Tuple[int, float]]] = {}
    for i, it in enumerate(items):
        if it["kind"] == "graded":
            fr = float(np.mean(votes[names[0]][i]))
            groups.setdefault(it["graded_group"], []).append((it["graded_step"], fr))
    mono_ok = mono_bad = 0
    for g, pts in groups.items():
        pts.sort()
        fr = [x[1] for x in pts]
        if all(fr[i] >= fr[i + 1] - 1e-9 for i in range(len(fr) - 1)):
            mono_ok += 1
        else:
            mono_bad += 1
    out["graded_monotonicity"] = {
        "groups": len(groups), "monotone": mono_ok, "non_monotone": mono_bad,
        "non_monotone_frac": (mono_bad / len(groups)) if groups else None,
        "reading": ("判定随窗口滑离主体单调劣化 = 门的决策方向正确;"
                    "非单调比例就是门不可靠的直接证据(它对'主体还在不在'不敏感)"),
    }
    return out


# =============================================================================
def _self_test() -> int:
    """不读真 H5:用合成场景验证构造逻辑与统计接线。"""
    P, F = [], []

    def ck(name, cond, detail=""):
        (P if cond else F).append(name)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f" — {detail}" if detail and not cond else ""))

    print("== T1 源身份键:粒度降级要如实,缺字段要抛 ==")
    k1 = source_key({"source_dataset": "aid", "source_class": "River",
                     "source_index": "7"})
    ck("skey_prefers_finest", source_key_granularity(k1).count("source_class") == 1,
       str(k1))
    k2 = source_key({"source_scene_category": "water", "source_scene_id": "3"})
    ck("skey_degrades", "source_scene_category" in source_key_granularity(k2), str(k2))
    _r = False
    try:
        source_key({"num_frames": 17})
    except AuditSetError:
        _r = True
    ck("skey_missing_raises", _r, "无源身份字段却返回了键 —— 无交集保证会变成假保证")

    print("== T2 窗口构造 ==")
    scene = np.arange(200 * 300 * 3, dtype=np.int64).reshape(200, 300, 3) % 251
    scene = scene.astype(np.uint8)
    c = _center_window(scene, (64, 96))
    ck("center_shape", c.shape == (64, 96, 3), str(c.shape))
    o0 = _offset_window(scene, (64, 96), 0.0)
    ck("offset0_equals_center", np.array_equal(o0, c))
    o1 = _offset_window(scene, (64, 96), 1.0)
    ck("offset1_shape", o1.shape == (64, 96, 3))
    ck("offset1_differs", not np.array_equal(o1, c))
    # 真正的不变量:**窗口位置**随 frac 单调移动(与场景内容无关)。
    # 不要断言"像素差单调" —— 那只在平滑场景上成立。
    origins = [_offset_origin((200, 300), (64, 96), f)
               for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    ck("offset_origin_monotone",
       all(origins[i][0] <= origins[i + 1][0] and origins[i][1] <= origins[i + 1][1]
           for i in range(len(origins) - 1)), str(origins))
    ck("offset_origin_reaches_edge", origins[-1] == (200 - 64, 300 - 96),
       f"frac=1 应贴到 {(200-64, 300-96)},得到 {origins[-1]}")
    ck("offset_origin_starts_center", origins[0] == ((200 - 64) // 2, (300 - 96) // 2))
    # 窗口内容确实随之改变(用平滑场景验,避免模回绕)
    smooth = np.zeros((200, 300, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:200, 0:300]
    smooth[:, :, 0] = (yy * 255 // 199).astype(np.uint8)
    smooth[:, :, 1] = (xx * 255 // 299).astype(np.uint8)
    sc = _center_window(smooth, (64, 96))
    sdiffs = [float(np.mean(np.abs(_offset_window(smooth, (64, 96), f).astype(int)
                                   - sc.astype(int))))
              for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    ck("offset_content_monotone_on_smooth",
       all(sdiffs[i] <= sdiffs[i + 1] + 1e-9 for i in range(len(sdiffs) - 1)),
       str([round(x, 2) for x in sdiffs]))

    print("== T3 扰动组 ==")
    pert = _perturb(c)
    ck("perturb_has_flips", "fliplr" in pert and "flipud" in pert)
    ck("perturb_shapes_match",
       all(v.shape == c.shape for v in pert.values()),
       str({k: v.shape for k, v in pert.items()}))
    ck("perturb_all_uint8", all(v.dtype == np.uint8 for v in pert.values()))
    print(f"      扰动种类: {sorted(pert)}"
          + ("" if "jpeg70" in pert else "  (无 cv2 → 少 jpeg70,已如实记录)"))

    print("== T4 帧复制的分母与冻结阈值一致 ==")
    tiles = _tile(c, sg.FROZEN_P0_FRAMES)
    ck("tile_count_matches_frozen", len(tiles) == 9 == sg.FROZEN_P0_FRAMES)
    ck("frozen_threshold_is_3_of_9",
       abs(sg.FROZEN_P0_THRESHOLD - 2.5 / 9) < 1e-12
       or abs(sg.FROZEN_P0_THRESHOLD - 0.2777777777777778) < 1e-15,
       repr(sg.FROZEN_P0_THRESHOLD))
    ck("threshold_means_ge_3_frames",
       int(np.ceil(sg.FROZEN_P0_THRESHOLD * 9)) == 3,
       f"ceil({sg.FROZEN_P0_THRESHOLD}*9)={int(np.ceil(sg.FROZEN_P0_THRESHOLD*9))}")

    print("== T5 general 不受管辖(与已核实事实一致)==")
    ck("general_not_gated", "general" not in sg.GATED_CATEGORIES,
       str(sg.GATED_CATEGORIES))
    ck("gated_is_three_pools", set(sg.GATED_CATEGORIES) == {"water", "urban", "rural"})

    print("== T6 仲裁者不可用时必须抛 ==")
    probe = probe_available_arbiters()
    print(f"      {probe['verdict'][:88]}")
    if not probe["clip_arbiters_available"]:
        _r = False
        try:
            build_arbiters()
        except KappaUnavailable:
            _r = True
        ck("audit_requires_real_arbiters", _r,
           "无权重却静默用合成仲裁者 → 会产出漂亮但无意义的 κ")

    print(f"\n汇总: {len(P)} PASS / {len(F)} FAIL")
    if F:
        print("结果: FAIL —— " + ", ".join(F))
        return 4
    print("结果: 全绿 ✓(构造逻辑与统计接线正确;真实仲裁需集群权重)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="语义门审计集:标签由构造保证,零真人参与")
    ap.add_argument("--gt-h5", type=Path, help="canonical GEO_H5")
    ap.add_argument("--calib-h5", type=Path, default=None,
                    help="调参集当时用的 H5(用于反查源身份、证明无交集)")
    ap.add_argument("--per-pool", type=int, default=40)
    ap.add_argument("--graded-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kappa-gate", type=float, default=KAPPA_GATE_DEFAULT)
    ap.add_argument("--allow-synthetic-arbiters", action="store_true",
                    help="本机无 CLIP 权重时用合成仲裁者。**产出只证明代码正确,"
                         "不证明语义门可靠** —— 结果里会留痕。")
    ap.add_argument("--run", action="store_true", help="构造后立刻跑审计")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if not args.gt_h5:
        ap.error("--gt-h5 必填(或用 --self-test)")

    aset = build_auditset(args.gt_h5, per_pool=args.per_pool,
                          graded_steps=args.graded_steps,
                          calib_h5=args.calib_h5, seed=args.seed)
    print(f"[auditset] 正例 {aset['counts']['positive']} / "
          f"硬负例 {aset['counts']['hard_negative']} / "
          f"分级 {aset['counts']['graded']}  共 {aset['counts']['total']}")
    print(f"[auditset] 源身份粒度: {aset['source_key_granularity']}")
    print(f"[auditset] 与调参集交集: {aset['calib_disjoint_note']}")
    for lim in aset["limitations"]:
        print(f"[limit] {lim}")

    report = None
    if args.run:
        report = run_audit(aset, allow_synthetic=args.allow_synthetic_arbiters,
                            kappa_gate=args.kappa_gate)
        if args.allow_synthetic_arbiters:
            report["SYNTHETIC_ARBITERS_WARNING"] = (
                "本次用的是合成仲裁者 —— 只证明代码正确,**不构成任何关于"
                "语义门可靠性的结论**。真实仲裁需在集群上跑。")
        print(f"[audit] Fleiss κ = {report.get('fleiss_kappa')}")
        if report.get("precision_recall") is None:
            print(f"[audit] fail-closed:{report.get('fail_closed_reason', report.get('kappa_unavailable'))}")

    if args.output:
        args.output.write_text(json.dumps(
            {"auditset": {k: v for k, v in aset.items() if k != "items"},
             "n_items": len(aset["items"]), "report": report},
            indent=2, ensure_ascii=False))
        (args.output.parent / (args.output.stem + "_items.json")).write_text(
            json.dumps(aset["items"], indent=2, ensure_ascii=False))
        print(f"[done] {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
