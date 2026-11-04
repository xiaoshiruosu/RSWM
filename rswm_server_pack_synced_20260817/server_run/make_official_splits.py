#!/usr/bin/env python3
# =============================================================================
# make_official_splits.py — RSWM-Bench 官方三划分 train / val / test(v1.0)
#
# 为什么要另起一个划分器(而不是扩展 make_precheck_split.py):
#   * make_precheck_split / dataset_loader.split_dataset 是**索引级**随机划分,
#     对"同一张源图裁出的多个样本"一无所知。它们存在的目的是与训练器逐位同源
#     (泄漏门要拿它当对照),因此绝不能改 —— 改了泄漏门自己先失效。
#   * benchmark 发布要的是另一件事:**source-disjoint 的隐藏 test**。同一张
#     AID 源图裁出的两个窗口,视觉内容强相关;它们分居 train/test 时,保真指标
#     (SSIM/PSNR/FID/FVD)测的是记忆而不是泛化。check_benchmark_duplicates.py
#     的 train_val_source_leaks 只能【事后】发现这种泄漏;本脚本从源头杜绝:
#     先按源图分组,再把**整组**投进同一个划分。
#   两者共存:precheck split 继续守训练链的同源性;official splits 是发布物。
#
# 源图身份怎么确定(与 check_benchmark_duplicates.py 的 source_key 口径一致,
# 但更保守 —— 那边只看一个键,这里把所有可用信号做并查集):
#   1) (source_dataset, source_split, source_index)  ← HF 建库的权威键
#   2) source_path                                    ← 本地 AID 目录建库
#   3) (source_scene_category, source_scene_id)       ← 建库登记表键;--no-hf
#      合成库里这是唯一可用的信号
#   4) full_scene 字节 SHA1                           ← 内容级真源:attrs 可以
#      缺、可以被改,像素不会说谎。默认开启(--no-scene-hash 可关)。
#   任一信号相同即判同源 → 并查集合并。样本一个信号都没有 = 无法证明它与谁
#   不同源 → 默认 fail-closed(--allow-unidentified-sources 才放行,并逐样本
#   自成一组,这是**乐观假设**,报告里会点名计数)。
#
# 补充说明(读数时别误解):canonical 主库过 check_benchmark_duplicates
# --expect-zero 时,任何 source_key 复用都是建库失败 → 正常主库里每组恰好
# 1 个样本,分组机制此时是**守卫**而非约束。组大小 >1 只会出现在派生库/
# 旧库/合成库上,报告里会显式给出 max_group_size 供判读。
#
# 分配算法(确定性,固定 seed):
#   * 按 stratify 字段(默认 source_scene_category)把组分桶,桶内用
#     random.Random(f"{seed}:{类别}") 洗牌 —— 每个类别一条独立 RNG 流,
#     增删一个类别不会扰动其它类别的顺序(整体一条流的话会全盘漂移)。
#   * 各类别的组**轮转交错**成一个全局序列,再逐组投给"当前缺口最大"的划分
#     (缺口 = ratio×N − 已分配数;并列时按 train→val→test 固定序)。
#     轮转交错是关键:按类别顺序处理会让前几个类别吃满 train、最后一个类别
#     全进 test(全局比例对、类别分布崩)。交错后全局比例与类别比例同时成立。
#
# 用法:
#   python3 server_run/make_official_splits.py --gt-h5 $WORK/data/train_geo.h5 \
#       -o $WORK/official_splits.json
#   # 发布用:附带产出去标签的公开 test(见 SUBMISSION.md)
#   python3 server_run/make_official_splits.py --gt-h5 ... -o ... \
#       --redact-test --public-dir $WORK/release/test_public
#   # 确定性回归:同 seed 两次必须逐位一致
#   python3 server_run/make_official_splits.py --gt-h5 ... -o new.json \
#       --verify-against old.json
#
# 退出码:0 正常 / 2 用法或数据错误 / 5 自校验 FAIL / 6 --verify-against 不一致
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import h5py

SCHEMA = "rswm_official_splits/v1"
SPLIT_NAMES = ("train", "val", "test")

# 公开 test 里**绝不允许**出现的数据集名。语义分三类:
#   video_frames/full_scene —— 外观真值。泄露 = 提交方直接复制 GT。
#   waypoints/segment_angles/trajectory —— 折线命令真值(scene 像素坐标)。
#     Track A 的任务定义是"只给方向词",连续角一旦公开,22.5° 量化地板就被
#     绕过(见 instrument_floor_report.py §1),Track A 的分数不再可比。
#   poses —— Plücker 位姿。它**就是** Track B 的命令,所以不放主文件、
#     单独出一个 Track B 附件;Track A 参赛者拿不到它。
_FORBIDDEN_PUBLIC_DATASETS = frozenset({
    "video_frames", "full_scene", "waypoints", "segment_angles", "trajectory",
})
# 源图身份 attrs 必须从公开物里抹掉:AID/RESISC45 是公开语料,拿到
# (dataset, split, index) 或 path 就能下载原图,再按公开的窗口尺寸暴力搜索
# 首帧位置 → 直接裁出 GT 视频。这是最省事的作弊路径,必须堵。
_FORBIDDEN_PUBLIC_ATTRS = frozenset({
    "source_dataset", "source_split", "source_index", "source_path",
    "source_class", "source_tile_indices", "source_scene_id",
})
# 公开 test 保留的 attrs:全都是"模型生成时本来就该知道"的条件信息。
# num_segments 与 scene 其实已经写在 prompt 文本里,单独给一份只是省得
# 参赛者去解析英文句子。
_PUBLIC_KEEP_ATTRS = ("prompt", "num_frames", "num_segments",
                      "source_scene_category")


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_obj(obj):
    """对 JSON 可序列化对象取规范化 SHA —— sort_keys + 无空格,跨机器可复算。"""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def h5_str(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


class DSU:
    """并查集。多信号源身份判定:任一信号相同即同源,传递闭包由它负责。"""

    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # 小根优先:组代表恒为组内最小索引 → 分组结果与遍历顺序无关(确定性)
            if ra > rb:
                ra, rb = rb, ra
            self.p[rb] = ra


def chi2_sf(stat, df):
    """卡方上尾概率。scipy 有就用 scipy(权威);没有则退回临界值区间。

    退回时返回 None + 一个区间描述,而不是自己实现不完全伽马函数 ——
    一个只在无 scipy 环境里跑过的数值实现,出错了也没人会发现。
    """
    try:
        from scipy.stats import chi2 as _chi2
        return float(_chi2.sf(stat, df)), "scipy.stats.chi2.sf"
    except Exception:
        return None, "scipy 不可用 → p 值未计算(见 critical_value_005)"


# 卡方 α=0.05 临界值(df 1..12),仅在无 scipy 时用于给出"显著/不显著"判读。
_CHI2_CRIT_005 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070,
                  6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919, 10: 18.307,
                  11: 19.675, 12: 21.026}


# --------------------------------------------------------------------------
# 读 H5:样本身份信号 + 分层/报告用的属性
# --------------------------------------------------------------------------
def read_samples(gt_h5, scene_hash=True, verbose=True):
    rows = []
    with h5py.File(gt_h5, "r") as f:
        if "data" not in f:
            print(f"[FATAL] {gt_h5} 没有 data 组", file=sys.stderr)
            sys.exit(2)
        meta = dict(f["metadata"].attrs) if "metadata" in f else {}
        keys = sorted(f["data"].keys())   # 索引口径 = sorted(keys) 的位置,
        # 与 dataset_loader.load_val_samples / check_benchmark_duplicates /
        # make_precheck_split 完全一致。换成 sample_%05d 数值序会在样本被跳过
        # (builder 的损坏样本 continue)时与它们错位。
        for i, k in enumerate(keys):
            g = f["data"][k]
            a = g.attrs
            sig = []
            if "source_index" in a:
                sig.append(("src", h5_str(a.get("source_dataset", "")),
                            h5_str(a.get("source_split", "")),
                            int(a["source_index"])))
            p = h5_str(a.get("source_path", "") or "")
            if p:
                sig.append(("path", p))
            if "source_scene_id" in a:
                sig.append(("scene", h5_str(a.get("source_scene_category", "")),
                            int(a["source_scene_id"])))
            if scene_hash and "full_scene" in g:
                sig.append(("pix", hashlib.sha1(
                    np.ascontiguousarray(g["full_scene"][:]).tobytes()
                ).hexdigest()))
            rows.append({
                "index": i,
                "key": k,
                "signals": sig,
                "category": h5_str(a.get("source_scene_category", "?")),
                "num_frames": int(a.get("num_frames",
                                        g["video_frames"].shape[0]
                                        if "video_frames" in g else 0)),
                "num_segments": int(a["num_segments"]) if "num_segments" in a else None,
                "overlap_ratio": (float(a["overlap_ratio"])
                                  if "overlap_ratio" in a else None),
            })
            if verbose and scene_hash and (i + 1) % 200 == 0:
                print(f"  … 读源指纹 {i + 1}/{len(keys)}")
    return rows, meta


def build_groups(rows, allow_unidentified=False):
    """按源身份信号做并查集 → (group_of_index, groups: {gid: [idx...]})。"""
    dsu = DSU(len(rows))
    by_sig = defaultdict(list)
    unidentified = []
    for r in rows:
        if not r["signals"]:
            unidentified.append(r["index"])
        for s in r["signals"]:
            by_sig[s].append(r["index"])
    for _s, idxs in by_sig.items():
        for j in idxs[1:]:
            dsu.union(idxs[0], j)
    if unidentified and not allow_unidentified:
        print(f"[FATAL] {len(unidentified)} 个样本没有任何源身份信号"
              f"(如 index {unidentified[:5]})—— 无法证明它们与 test 源图不同源,"
              f"划分出来的 test 不能声称 source-disjoint。\n"
              f"        请用带 source_* 标注的 builder 重建;确知是合成库/无所谓"
              f"时显式加 --allow-unidentified-sources(报告会计数)。",
              file=sys.stderr)
        sys.exit(2)
    groups = defaultdict(list)
    for r in rows:
        groups[dsu.find(r["index"])].append(r["index"])
    return {r["index"]: dsu.find(r["index"]) for r in rows}, dict(groups), unidentified


# --------------------------------------------------------------------------
# 分配
# --------------------------------------------------------------------------
def assign_splits(rows, groups, ratios, seed, stratify=True):
    """→ {split: [index...]}(已排序)。确定性:只依赖 (rows, groups, ratios, seed)。"""
    idx2row = {r["index"]: r for r in rows}
    gids = sorted(groups)
    # 组的分层类别:组内类别的众数(并列取字典序最小)。正常库里组是单样本,
    # 众数就是它自己;跨类别组本身是 check_benchmark_duplicates 的违规项,
    # 这里不判停,只保证分配有定义并在自校验里点名。
    gcat = {}
    for gid in gids:
        c = Counter(idx2row[i]["category"] for i in groups[gid])
        gcat[gid] = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

    buckets = defaultdict(list)
    for gid in gids:
        buckets[gcat[gid] if stratify else "__all__"].append(gid)
    cats = sorted(buckets)
    for c in cats:
        # 每类别一条独立 RNG 流:类别集合变化时不牵连其它类别的顺序。
        # random.Random(str) 在 CPython 里走 sha512(seed)→int,跨平台确定。
        random.Random(f"{seed}:{c}").shuffle(buckets[c])

    # 轮转交错(见文件头):cat0[0], cat1[0], …, cat0[1], cat1[1], …
    order = []
    for j in range(max(len(v) for v in buckets.values()) if buckets else 0):
        for c in cats:
            if j < len(buckets[c]):
                order.append(buckets[c][j])

    n = len(rows)
    target = {s: ratios[s] * n for s in SPLIT_NAMES}
    assigned = {s: 0 for s in SPLIT_NAMES}
    out = {s: [] for s in SPLIT_NAMES}
    for gid in order:
        size = len(groups[gid])
        # 缺口最大者优先;并列按 SPLIT_NAMES 固定序(max 取第一个最大值)。
        # 用绝对缺口而不是"相对负载比"是有意的:后者在小类别上会把整个类别
        # 都塞进 train(比值恒偏向大 ratio 的划分),test 直接空掉。
        best = max(SPLIT_NAMES, key=lambda s: (target[s] - assigned[s]))
        out[best].extend(groups[gid])
        assigned[best] += size
    return {s: sorted(out[s]) for s in SPLIT_NAMES}


# --------------------------------------------------------------------------
# 自校验
# --------------------------------------------------------------------------
def self_check(rows, groups, splits, ratios, chi2_alpha):
    idx2row = {r["index"]: r for r in rows}
    n = len(rows)
    checks, failures = {}, []

    # ① 划分是全集的一个划分(不重不漏)
    sets = {s: set(v) for s, v in splits.items()}
    union = set().union(*sets.values())
    overlaps = {}
    for a in range(len(SPLIT_NAMES)):
        for b in range(a + 1, len(SPLIT_NAMES)):
            sa, sb = SPLIT_NAMES[a], SPLIT_NAMES[b]
            inter = sorted(sets[sa] & sets[sb])
            if inter:
                overlaps[f"{sa}∩{sb}"] = inter[:20]
    part_ok = (not overlaps) and union == set(range(n))
    checks["partition"] = {
        "pass": part_ok, "n_total": n, "n_covered": len(union),
        "missing": sorted(set(range(n)) - union)[:20], "overlaps": overlaps,
    }
    if not part_ok:
        failures.append("三划分不是全集的一个划分(有重叠或有漏样本)")

    # ② source-disjoint:**从写出的划分反查**,而不是信任分配过程。
    #    分配代码将来被改动时,这条能独立发现问题。
    split_of = {i: s for s, v in splits.items() for i in v}
    cross = []
    for gid, members in groups.items():
        ss = sorted({split_of[i] for i in members if i in split_of})
        if len(ss) > 1:
            cross.append({"group": int(gid), "indices": sorted(members),
                          "splits": ss,
                          "keys": [idx2row[i]["key"] for i in sorted(members)]})
    checks["source_disjoint"] = {
        "pass": not cross, "n_groups": len(groups),
        "max_group_size": max((len(v) for v in groups.values()), default=0),
        "n_multi_sample_groups": sum(1 for v in groups.values() if len(v) > 1),
        "cross_split_groups": cross[:20],
        "note": ("组数 == 样本数 → 每张源图恰好一个样本(canonical 主库过 "
                 "check_benchmark_duplicates --expect-zero 后应当如此);"
                 "组数 < 样本数 → 存在源图复用,本划分把整组锁进同一 split"),
    }
    if cross:
        failures.append(f"{len(cross)} 个源图组跨划分 —— source-disjoint 不成立")

    # ③ 类别分布 + 卡方拟合优度(H0: 各划分的类别构成 = 全库构成)
    cats = sorted({r["category"] for r in rows})
    overall = Counter(r["category"] for r in rows)
    cat_block = {"categories": cats,
                 "overall": {c: overall[c] for c in cats}, "per_split": {}}
    for s in SPLIT_NAMES:
        cnt = Counter(idx2row[i]["category"] for i in splits[s])
        ns = len(splits[s])
        exp = {c: ns * overall[c] / n for c in cats} if n else {}
        stat = sum((cnt[c] - exp[c]) ** 2 / exp[c] for c in cats if exp.get(c, 0) > 0)
        df = max(1, sum(1 for c in cats if exp.get(c, 0) > 0) - 1)
        p, psrc = chi2_sf(stat, df)
        crit = _CHI2_CRIT_005.get(df)
        # 期望频数 <5 时卡方近似不成立 —— 明说,不装作 p 值可信。
        small = [c for c in cats if 0 < exp.get(c, 0) < 5]
        ok = True if p is None else (p >= chi2_alpha)
        if p is None and crit is not None:
            ok = stat <= crit
        cat_block["per_split"][s] = {
            "n": ns, "counts": {c: cnt[c] for c in cats},
            "share": {c: (cnt[c] / ns if ns else None) for c in cats},
            "expected": {c: round(exp.get(c, 0.0), 3) for c in cats},
            "chi2": round(float(stat), 4), "df": df, "p_value": p,
            "p_source": psrc, "critical_value_005": crit,
            "alpha": chi2_alpha, "pass": bool(ok),
            "low_expected_cells": small,
        }
        if not ok:
            failures.append(f"{s} 划分的类别分布与全库显著不同"
                            f"(chi2={stat:.3f}, df={df}, p={p})")
    checks["category_balance"] = cat_block

    # ④ 结构分布(帧数 / 段数 / overlap)—— 不判 pass/fail,只让人一眼看出
    #    test 是不是恰好全是 K=4 的长样本这类隐性偏斜。
    struct = {}
    for s in SPLIT_NAMES:
        rs = [idx2row[i] for i in splits[s]]
        ov = [r["overlap_ratio"] for r in rs if r["overlap_ratio"] is not None]
        struct[s] = {
            "n": len(rs),
            "num_frames_hist": dict(sorted(Counter(r["num_frames"] for r in rs).items())),
            "num_segments_hist": dict(sorted(Counter(
                r["num_segments"] for r in rs if r["num_segments"] is not None).items())),
            "overlap_ratio": ({"min": float(np.min(ov)), "max": float(np.max(ov)),
                               "mean": float(np.mean(ov)), "n": len(ov)} if ov else None),
        }
    checks["structure"] = struct

    # ⑤ 帧数契约 T = 9 + (K-1)*8 且 T ≡ 1 (mod 4)
    bad_T = [{"key": r["key"], "T": r["num_frames"], "K": r["num_segments"]}
             for r in rows
             if r["num_segments"] is not None
             and (r["num_frames"] != 9 + (r["num_segments"] - 1) * 8
                  or r["num_frames"] % 4 != 1)]
    checks["frame_contract"] = {
        "pass": not bad_T, "formula": "T = 9 + (K-1)*8, 且 T ≡ 1 (mod 4)",
        "violations": bad_T[:20], "n_violations": len(bad_T)}
    if bad_T:
        failures.append(f"{len(bad_T)} 个样本违反帧数契约 —— 提交协议的帧数校验"
                        f"会拒收整个 test,建库口径先修")

    # ⑥ 实际比例 vs 目标
    checks["ratios"] = {
        "target": ratios,
        "actual": {s: (len(splits[s]) / n if n else 0.0) for s in SPLIT_NAMES},
        "counts": {s: len(splits[s]) for s in SPLIT_NAMES},
    }
    return checks, failures


# --------------------------------------------------------------------------
# 公开 test(去标签)
# --------------------------------------------------------------------------
def write_public_test(gt_h5, test_idx, out_dir, meta, splits_sha, gt_sha):
    """产出公开 test:主文件(Track A/B 共用输入)+ Track B 位姿附件 + manifest。

    为什么位姿单独一个文件而不是两份完整拷贝:首帧 480×832×3 float32 ≈ 4.8 MB/条,
    复制一份 test 就是几个 GB。位姿 (T,4,4) float32 只有几 KB —— 拆文件既省空间,
    又让"Track A 参赛者手里根本没有位姿"成为**物理事实**而不是自觉遵守的约定。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pub = out_dir / "rswm_test_public.h5"
    posef = out_dir / "rswm_test_trackB_poses.h5"
    manifest_path = out_dir / "rswm_test_manifest.json"
    samples = {}

    with h5py.File(gt_h5, "r") as src, \
            h5py.File(pub, "w") as dst, h5py.File(posef, "w") as dpz:
        keys = sorted(src["data"].keys())
        for f_, tag in ((dst, "public_inputs"), (dpz, "trackB_poses")):
            m = f_.create_group("metadata")
            m.attrs["schema"] = f"rswm_public_test/{tag}/v1"
            m.attrs["split"] = "test"
            m.attrs["total_instances"] = len(test_idx)
            m.attrs["source_h5_sha256"] = gt_sha
            m.attrs["official_splits_sha256"] = splits_sha
            for a in ("target_size", "pose_window_size", "prompt_schema",
                      "angle_quantize", "min_overlap_ratio", "max_overlap_ratio",
                      "steps_per_segment", "builder_version", "taxonomy_version",
                      "prompt_template_version"):
                if a in meta:
                    m.attrs[a] = meta[a]
        dpz["metadata"].attrs["pose_injection"] = meta.get(
            "pose_injection", "plucker_from_trajectory")
        gdst = dst.create_group("data")
        gpz = dpz.create_group("data")
        for i in test_idx:
            k = keys[i]
            s = src["data"][k]
            g = gdst.create_group(k)
            g.create_dataset("image", data=s["image"][:], compression="gzip")
            if "intrinsics" in s:
                # dummy 内参(全库同一常量,见 _make_dummy_intrinsics),不含真值信息
                g.create_dataset("intrinsics", data=s["intrinsics"][:])
            for a in _PUBLIC_KEEP_ATTRS:
                if a in s.attrs:
                    g.attrs[a] = s.attrs[a]
            gp = gpz.create_group(k)
            gp.create_dataset("poses", data=s["poses"][:])
            gp.attrs["num_frames"] = int(s.attrs.get("num_frames",
                                                     s["poses"].shape[0]))
            samples[k] = {
                "index": int(i),
                "num_frames": int(s.attrs.get("num_frames", s["poses"].shape[0])),
                "num_segments": (int(s.attrs["num_segments"])
                                 if "num_segments" in s.attrs else None),
                "scene": h5_str(s.attrs.get("source_scene_category", "?")),
                "prompt": h5_str(s.attrs.get("prompt", "")),
            }

    # 去标签自校验:重新打开公开文件,逐组逐 attr 扫一遍禁忌名单。
    # 上面的写入逻辑将来被改(比如有人顺手加一行 copy full_scene),
    # 这里会当场拦下 —— 发布物的机密性不能靠"写的时候记得"。
    leaks = []
    with h5py.File(pub, "r") as f:
        for k in f["data"]:
            for name in f["data"][k]:
                if name in _FORBIDDEN_PUBLIC_DATASETS or name == "poses":
                    leaks.append(f"{k}/{name}")
            for a in f["data"][k].attrs:
                if a in _FORBIDDEN_PUBLIC_ATTRS:
                    leaks.append(f"{k}.attrs[{a}]")
    with h5py.File(posef, "r") as f:
        for k in f["data"]:
            for name in f["data"][k]:
                if name in _FORBIDDEN_PUBLIC_DATASETS:
                    leaks.append(f"poses:{k}/{name}")
            for a in f["data"][k].attrs:
                if a in _FORBIDDEN_PUBLIC_ATTRS:
                    leaks.append(f"poses:{k}.attrs[{a}]")
    if leaks:
        print(f"[FATAL] 公开 test 泄露了真值/源身份字段: {leaks[:10]}", file=sys.stderr)
        sys.exit(5)

    tgt = [int(v) for v in meta.get("target_size", (480, 832))]
    win = [int(v) for v in meta.get("pose_window_size", (128, 222))]
    manifest = {
        "schema": "rswm_public_test_manifest/v1",
        "n_samples": len(samples),
        "submission_contract": {
            "path": "generated/<sample_id>.npy",
            "array_shape": ["num_frames(逐样本,见 samples)", tgt[0], tgt[1], 3],
            "dtype": "uint8",
            "value_range": [0, 255],
            "channel_order": "RGB",
            "frame_formula": "T = 9 + (K-1)*8;T ≡ 1 (mod 4)",
            "note": "每个样本的帧数必须与 samples[<id>].num_frames 完全相等,不得多也不得少",
        },
        "target_size": tgt,
        "pose_window_size": win,
        "tracks": {
            "A": {"inputs": ["image", "prompt"],
                  "file": "rswm_test_public.h5",
                  "note": "text-only(pose_mode=drop):位姿通路不接,只有方向词与段数是被命令的"},
            "B": {"inputs": ["image", "prompt", "poses", "intrinsics"],
                  "file": ["rswm_test_public.h5", "rswm_test_trackB_poses.h5"],
                  "note": "pose_mode=native:Plücker 位姿即完整轨迹命令"},
        },
        "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in (pub, posef)},
        "source_h5_sha256": gt_sha,
        "official_splits_sha256": splits_sha,
        "redacted": {
            "datasets": sorted(_FORBIDDEN_PUBLIC_DATASETS),
            "attrs": sorted(_FORBIDDEN_PUBLIC_ATTRS),
            "why": ("外观真值 + 折线命令真值 + 源图身份三类全部移除;"
                    "poses 移入 Track B 附件,Track A 参赛者物理上拿不到"),
        },
        "samples": samples,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"public_h5": str(pub), "trackB_poses_h5": str(posef),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "files": manifest["files"]}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="RSWM-Bench 官方 train/val/test 三划分(source-disjoint)")
    ap.add_argument("--gt-h5", required=True, help="canonical GEO_H5(建库产物)")
    ap.add_argument("-o", "--output", default="official_splits.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.20)
    ap.add_argument("--no-scene-hash", action="store_true",
                    help="不读 full_scene 做内容指纹(省 I/O)。只有在 attrs 齐全"
                         "且已过 check_benchmark_duplicates 的库上才建议关。")
    ap.add_argument("--allow-unidentified-sources", action="store_true",
                    help="允许样本没有任何源身份信号(每条自成一组的乐观假设)")
    ap.add_argument("--no-stratify", action="store_true",
                    help="不按 source_scene_category 分层(默认分层)")
    ap.add_argument("--chi2-alpha", type=float, default=0.01,
                    help="类别分布卡方检验的显著性水平;p < alpha 判 FAIL")
    ap.add_argument("--min-per-split", type=int, default=1,
                    help="每个划分至少多少样本(ratio>0 的划分);不满足即 FAIL")
    ap.add_argument("--redact-test", action="store_true",
                    help="额外产出去掉 test 标签的公开版本(见 SUBMISSION.md)")
    ap.add_argument("--public-dir", default=None,
                    help="--redact-test 的输出目录(默认 <output 同级>/test_public)")
    ap.add_argument("--emit-trainval", default=None,
                    help="额外写一份 {train_indices, val_indices} 两路文件,"
                         "供只认 train_val_split.json 的旧消费者;test 永不出现在里面")
    ap.add_argument("--verify-against", default=None,
                    help="与既有 official_splits.json 逐位比对(params+splits+groups);"
                         "不一致 exit 6")
    args = ap.parse_args()

    ratios = {"train": args.train_ratio, "val": args.val_ratio,
              "test": args.test_ratio}
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        print(f"[FATAL] 三个 ratio 之和 = {sum(ratios.values())} ≠ 1", file=sys.stderr)
        sys.exit(2)
    if any(v < 0 for v in ratios.values()):
        print("[FATAL] ratio 不能为负", file=sys.stderr)
        sys.exit(2)

    gt_sha = sha256_file(args.gt_h5)
    t0 = time.time()
    rows, meta = read_samples(args.gt_h5, scene_hash=not args.no_scene_hash)
    if not rows:
        print(f"[FATAL] {args.gt_h5} 无样本", file=sys.stderr)
        sys.exit(2)
    _group_of, groups, unidentified = build_groups(
        rows, allow_unidentified=args.allow_unidentified_sources)
    splits = assign_splits(rows, groups, ratios, args.seed,
                           stratify=not args.no_stratify)
    checks, failures = self_check(rows, groups, splits, ratios, args.chi2_alpha)

    for s in SPLIT_NAMES:
        if ratios[s] > 0 and len(splits[s]) < args.min_per_split:
            failures.append(
                f"{s} 只有 {len(splits[s])} 个样本(< --min-per-split="
                f"{args.min_per_split})—— 样本太少或 ratio 太小,该划分不可用")

    idx2key = {r["index"]: r["key"] for r in rows}
    splits_block = {s: {"n": len(splits[s]), "indices": splits[s],
                        "keys": [idx2key[i] for i in splits[s]]}
                    for s in SPLIT_NAMES}
    groups_block = {
        "n_groups": len(groups),
        "signal_coverage": {
            kind: sum(1 for r in rows if any(sg[0] == kind for sg in r["signals"]))
            for kind in ("src", "path", "scene", "pix")},
        "n_unidentified": len(unidentified),
        "unidentified_indices": unidentified[:50],
        # 只落多样本组:单样本组占绝大多数,全量落盘会让文件膨胀到无法阅读。
        "multi_sample_groups": {str(gid): sorted(v)
                                for gid, v in sorted(groups.items())
                                if len(v) > 1},
    }
    params_block = {
        "seed": args.seed, "ratios": ratios,
        "stratify_by": None if args.no_stratify else "source_scene_category",
        "scene_content_hash": not args.no_scene_hash,
        "allow_unidentified_sources": bool(args.allow_unidentified_sources),
        "index_convention": "sorted(f['data'].keys()) 的下标(与 dataset_loader 一致)",
        "algorithm": ("源身份并查集分组 → 类别桶内 Random(f'{seed}:{cat}') 洗牌 → "
                      "类别轮转交错 → 逐组投给缺口最大的划分"),
    }
    # splits_sha256 只覆盖【可复算的划分内容】:params + splits + groups。
    # 不含时间戳/路径 —— 换一台机器、换一个目录重跑,SHA 必须仍然相同,
    # 否则这个 SHA 就不能用来证明"我们说的 test 和你手上的 test 是同一个"。
    splits_sha = sha256_obj({"params": params_block, "splits": splits_block,
                             "groups": groups_block})

    doc = {
        "schema": SCHEMA,
        "provenance": {
            "script": "server_run/make_official_splits.py",
            "script_sha256": sha256_file(__file__),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "argv": sys.argv[1:],
            "elapsed_seconds": round(time.time() - t0, 2),
        },
        "dataset": {
            "path": str(args.gt_h5), "sha256": gt_sha, "n_samples": len(rows),
            "metadata": {k: (v.tolist() if isinstance(v, np.ndarray)
                             else (h5_str(v) if isinstance(v, bytes)
                                   else (v.item() if isinstance(v, np.generic) else v)))
                         for k, v in sorted(meta.items())},
        },
        "params": params_block,
        "splits": splits_block,
        "groups": groups_block,
        "self_check": {"pass": not failures, "failures": failures, "checks": checks},
        "splits_sha256": splits_sha,
    }

    if args.verify_against:
        prev = json.loads(Path(args.verify_against).read_text())
        same = all(prev.get(k) == doc[k] for k in ("params", "splits", "groups"))
        if not same:
            diff = [k for k in ("params", "splits", "groups") if prev.get(k) != doc[k]]
            print(f"[FATAL] 与 {args.verify_against} 不一致(块: {diff};"
                  f"sha {prev.get('splits_sha256')} vs {splits_sha})—— "
                  f"划分不再确定,发布的 test 与已公布的不是同一个,拒绝覆盖。",
                  file=sys.stderr)
            sys.exit(6)
        print(f"[verify] 与 {args.verify_against} 的 params/splits/groups 逐位一致 ✓")

    if args.redact_test:
        if failures:
            print("[FATAL] 自校验未过,拒绝产出公开 test(见下方失败项)", file=sys.stderr)
        else:
            pub_dir = args.public_dir or str(Path(args.output).resolve().parent
                                             / "test_public")
            doc["public_release"] = write_public_test(
                args.gt_h5, splits["test"], pub_dir, meta, splits_sha, gt_sha)
            print(f"[public] 公开 test → {pub_dir}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.emit_trainval:
        # 兼容层:只有 train/val 两路的消费者(dataset_loader 口径)。
        # test 索引绝不出现在这里 —— 训练链看不到 test 是发布物有效性的前提。
        Path(args.emit_trainval).write_text(json.dumps({
            "train_indices": splits["train"], "val_indices": splits["val"],
            "split_seed": args.seed, "train_ratio": None,
            "total": len(rows),
            "source": "make_official_splits.py(source-disjoint 官方划分)",
            "official_splits_sha256": splits_sha,
            "note": ("这是官方三划分的 train/val 两路视图,test 已被排除。"
                     "它与 make_precheck_split.py / dataset_loader.split_dataset "
                     "的索引级划分【不同源】,不要拿去做那条泄漏门的对照。"),
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[trainval] 两路视图 → {args.emit_trainval}")

    # ---- 打印 ----
    print("\n" + "=" * 72)
    print(f"样本 {len(rows)}  源图组 {len(groups)}"
          f"(最大组 {checks['source_disjoint']['max_group_size']},"
          f"多样本组 {checks['source_disjoint']['n_multi_sample_groups']})")
    for s in SPLIT_NAMES:
        b = checks["category_balance"]["per_split"][s]
        print(f"  {s:<5} n={b['n']:<6} 占比={len(splits[s]) / len(rows):.4f}"
              f"(目标 {ratios[s]:.4f})  类别={b['counts']}"
              f"  chi2={b['chi2']} df={b['df']} p={b['p_value']}")
    print(f"source-disjoint: {'PASS' if checks['source_disjoint']['pass'] else 'FAIL'}"
          f"  partition: {'PASS' if checks['partition']['pass'] else 'FAIL'}"
          f"  帧数契约: {'PASS' if checks['frame_contract']['pass'] else 'FAIL'}")
    print(f"splits_sha256 = {splits_sha}")
    print(f"[done] → {args.output}")
    if failures:
        print("\nOFFICIAL SPLIT GATE: FAIL")
        for f_ in failures:
            print(f"  - {f_}", file=sys.stderr)
        sys.exit(5)
    print("OFFICIAL SPLIT GATE: PASS")


if __name__ == "__main__":
    main()
