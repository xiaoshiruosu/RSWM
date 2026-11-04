#!/usr/bin/env python3
# =============================================================================
# check_benchmark_duplicates.py — 跨场景块重复样本判决(纯 CPU,只读)
#
# 触发证据(2026-07-23 Track B 尸检 top-15):不同场景标签的样本出现逐位相同
# 的 (净方向误差, GT ratio, 恢复 ratio) 组——sample_00390/00015/00140 三重、
# 00388/00013、00454/00079、00148/00023 成对;索引差全为 125 的倍数
# (500 = 4 场景 × 125 的块内同偏移)。恢复 ratio 来自对视频帧的光流,不同
# 影像几乎不可能逐位一致 → 两个假设待判决:
#   H-full : 同偏移样本 (源图+轨迹) 全同 —— 视频帧逐字节重复,仅场景标签不同
#   H-traj : 仅轨迹模板重复,源图不同 —— waypoints 同、video_frames 不同
#
# 判决方法:对每个样本取三类指纹
#   traj_fp  = md5(waypoints ∪ segment_angles ∪ trajectory)   (轨迹模板)
#   video_fp = md5(video_frames 首/中/末 3 帧字节 + shape)     (视频内容)
#   scene_fp = md5(full_scene 字节)                            (源场景图)
# 分组报告 >1 的组(成员/场景标签/索引差 mod 125),并交叉 train_val_split:
#   * 同一 video_fp 组横跨 train 与 val → **保真指标泄漏**(最严重)
#   * val 内部 video_fp 重复            → 有效 N < 100
#   * 仅 traj_fp 重复                   → 模板权重问题(Track B oracle 均值被
#                                         同一坏模板的多副本加权)
#
# 用法(登录节点):
#   python server_run/check_benchmark_duplicates.py \
#       --gt-h5 $WORK/data/train_geo.h5 \
#       --split $WORK/train_r16_sem_s42/train_val_split.json \
#       -o $WORK/duplicate_report.json
# =============================================================================
import argparse
import hashlib
import json
import sys
from collections import defaultdict

import numpy as np
import h5py


def load_train_holdout_split(path):
    """兼容训练器两路 split 与 make_official_splits 的三路发布格式。

    重复检查只需要回答“是否跨训练/留出集”。official 格式下 val+test
    都属于 holdout，不能因为键名嵌在 ``splits.*.indices`` 就绕过泄漏门。
    """
    with open(path, encoding="utf-8") as fh:
        split = json.load(fh)
    if "train_indices" in split and "val_indices" in split:
        return set(split["train_indices"]), set(split["val_indices"])
    blocks = split.get("splits")
    if isinstance(blocks, dict):
        def _idx(name):
            b = blocks.get(name, {})
            return list(b.get("indices", [])) if isinstance(b, dict) else []
        train = set(_idx("train"))
        holdout = set(_idx("val")) | set(_idx("test"))
        if train and holdout:
            return train, holdout
    raise ValueError(
        f"{path}: 不支持的 split schema；需要 train_indices/val_indices，"
        f"或 splits.train|val|test.indices")


def md5(*arrays):
    h = hashlib.md5()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("--split", default=None,
                    help="train_val_split.json;省略时跳过 train/val 泄漏交叉"
                         "(建库时 split 尚不存在,用 --expect-zero 模式)")
    ap.add_argument("--expect-zero", action="store_true",
                    help="[新建数据集门] 三类指纹出现【任何】重复组即 exit 5 ——"
                         "v2.3 修复版 builder 的产出必须零重复;供 01_build 作业"
                         "尾部内嵌,失败时 afterok 依赖链自动挂住下游训练。")
    ap.add_argument("--expect-n", type=int, default=0,
                    help="[门模式] 期望样本数;n_samples 不等即 FAIL(v2.4.1 修"
                         "V2-C4:空/截断 H5 曾能空过门)。0=只要求非空。")
    ap.add_argument("--allow-missing-paths", action="store_true",
                    help="[门模式] 允许 label 型数据集(RESISC45 等,HF 内嵌"
                         "bytes 无路径)缺 source_path。默认严格:任何缺路径都"
                         "FAIL;且 AID 样本永不豁免(其路径必然存在,缺=构建"
                         "环节出错)。")
    ap.add_argument("-o", "--output", default="duplicate_report.json")
    args = ap.parse_args()

    if args.split:
        train_set, val_set = load_train_holdout_split(args.split)
    else:
        train_set, val_set = set(), set()
        print("[info] 未给 --split:跳过 train/val 泄漏交叉,只做重复分组"
              + ("(--expect-zero 门模式)" if args.expect_zero else ""))

    traj_g, video_g, scene_g = defaultdict(list), defaultdict(list), defaultdict(list)
    scenes = {}
    src_by_i = {}   # (v2.4 §八) i -> {"key": (ds,split,idx)|None, "path": str}
    with h5py.File(args.gt_h5, "r") as f:
        data = f["data"] if "data" in f else f
        keys = sorted(data.keys())
        n = len(keys)
        print(f"[info] {n} 个样本,逐个取指纹(视频只哈希首/中/末 3 帧)…")
        for i, k in enumerate(keys):
            g = data[k]
            scenes[k] = str(g.attrs.get("source_scene_category", "?"))
            if "source_index" in g.attrs:
                src_by_i[i] = {
                    "key": (str(g.attrs.get("source_dataset", "")),
                            str(g.attrs.get("source_split", "")),
                            int(g.attrs.get("source_index", -1))),
                    "path": str(g.attrs.get("source_path", "") or ""),
                }
            else:
                src_by_i[i] = None
            parts = []
            for name in ("waypoints", "segment_angles", "trajectory"):
                if name in g:
                    parts.append(np.round(np.asarray(g[name][:], dtype=np.float64), 4))
            if parts:
                traj_g[md5(*parts)].append(i)
            vf = g["video_frames"]
            T = vf.shape[0]
            video_g[md5(vf[0], vf[T // 2], vf[T - 1])].append(i)
            if "full_scene" in g:
                scene_g[md5(g["full_scene"][:])].append(i)
            if (i + 1) % 100 == 0:
                print(f"  … {i + 1}/{n}")

    def report(tag, groups):
        dup = {h: idx for h, idx in groups.items() if len(idx) > 1}
        print(f"\n===== {tag}: {len(dup)} 个重复组 "
              f"(涉及 {sum(len(v) for v in dup.values())} 样本) =====")
        rows = []
        for h, idx in sorted(dup.items(), key=lambda kv: -len(kv[1])):
            idx = sorted(idx)
            diffs = sorted({b - a for a, b in zip(idx, idx[1:])})
            in_tr = [i for i in idx if i in train_set]
            in_va = [i for i in idx if i in val_set]
            leak = bool(in_tr and in_va)
            sc = [scenes[keys[i]] for i in idx]
            rows.append({"hash": h, "indices": idx, "scenes": sc,
                         "index_diffs": diffs, "diffs_mod_125_zero":
                         all(d % 125 == 0 for d in diffs),
                         "n_train": len(in_tr), "n_val": len(in_va),
                         "TRAIN_VAL_LEAK": leak})
            flag = "  ⚠️⚠️ TRAIN/VAL 泄漏" if leak else ""
            print(f"  {idx}  场景={sc}  差={diffs}"
                  f"{'(全为125倍数)' if rows[-1]['diffs_mod_125_zero'] else ''}"
                  f"  train={len(in_tr)}/val={len(in_va)}{flag}")
        if not dup:
            print("  (无)")
        return rows

    out = {"n_samples": len(scenes)}
    out["trajectory_template_dups"] = report("轨迹模板 traj_fp", traj_g)
    out["video_content_dups"] = report("视频内容 video_fp", video_g)
    out["scene_image_dups"] = report("源场景图 scene_fp", scene_g)

    # ---- (v2.4 §八) 源图身份检查:source_key 跨场景 / 跨 split / thumbnail ----
    n_missing_attrs = sum(1 for v in src_by_i.values() if v is None)
    present = {i: v for i, v in src_by_i.items() if v is not None}
    src_groups = defaultdict(list)
    for i, v in present.items():
        src_groups[v["key"]].append(i)
    cross_scene, src_leaks, src_dups = [], [], []
    for kk, idx in src_groups.items():
        scs = {scenes[keys[i]] for i in idx}
        if len(idx) > 1:
            # [v2.4.1 修 V2-C2] 任何 source_key 复用(含同场景内)都是违规:
            # 同一源图两个裁剪 → 三类指纹全不同 → 旧门抓不到。
            src_dups.append({"source_key": list(kk), "indices": sorted(idx),
                             "scenes": sorted(scs)})
        if len(scs) > 1:
            cross_scene.append({"source_key": list(kk), "indices": sorted(idx),
                                "scenes": sorted(scs)})
        in_tr = [i for i in idx if i in train_set]
        in_va = [i for i in idx if i in val_set]
        if in_tr and in_va:
            src_leaks.append({"source_key": list(kk), "train": in_tr, "val": in_va})
    thumbs = [i for i, v in present.items()
              if v["path"] and v["path"].lower().endswith("thumbnail.png")]
    nonjpg = [i for i, v in present.items()
              if v["path"] and not v["path"].lower().endswith((".jpg", ".jpeg"))]
    n_no_path = sum(1 for v in present.values() if not v["path"])
    print(f"\n===== 源图身份 (v2.4) =====")
    if not present:
        print("  (样本无 source_* attrs —— v2.4 之前的旧 builder 产物,无法做源身份审计)")
    else:
        print(f"  有 attrs: {len(present)}/{len(src_by_i)};缺 attrs: {n_missing_attrs}")
        print(f"  source_key 复用组(任意场景): {len(src_dups)};"
              f"其中跨场景: {len(cross_scene)}"
              + (f"  {cross_scene[:3]}" if cross_scene else ""))
        print(f"  thumbnail: {len(thumbs)};非 JPG 路径: {len(nonjpg)};无路径: {n_no_path}")
        if args.split:
            print(f"  train/val source 泄漏组: {len(src_leaks)}"
                  + ("  ⚠️⚠️" if src_leaks else ""))
    out["source_identity"] = {
        "n_with_attrs": len(present), "n_missing_attrs": n_missing_attrs,
        "duplicate_source_groups": src_dups,
        "cross_scene_groups": cross_scene, "thumbnail_indices": thumbs,
        "non_jpg_indices": nonjpg, "n_missing_path": n_no_path,
        "train_val_source_leaks": src_leaks,
    }

    # ---- --expect-zero 门(新建数据集):任何重复/源身份违规 = 建库失败 ----
    if args.expect_zero:
        reasons = []
        # [v2.4.1 修 V2-C4] 样本数下限:空/截断 H5 曾能"零重复"空过门
        if len(scenes) == 0:
            reasons.append("H5 无样本")
        elif args.expect_n and len(scenes) != args.expect_n:
            reasons.append(f"样本数 {len(scenes)} != 期望 {args.expect_n}")
        n_bad = (len(out["trajectory_template_dups"])
                 + len(out["video_content_dups"])
                 + len(out["scene_image_dups"]))
        if n_bad:
            reasons.append(f"{n_bad} 个指纹重复组")
        # (v2.4) 源身份条件:新 builder 必须逐样本写 source_* attrs
        if n_missing_attrs:
            reasons.append(f"{n_missing_attrs} 个样本缺 source_* attrs"
                           f"(旧 builder 产物或标注环节被绕过)")
        if src_dups:
            # [v2.4.1 修 V2-C2] 同场景内源复用也拦(不同裁剪的指纹全不同,
            # 只有 source_key 能抓)
            reasons.append(f"{len(src_dups)} 个 source_key 复用组"
                           f"(含同场景;跨场景 {len(cross_scene)})")
        if src_leaks:
            reasons.append(f"{len(src_leaks)} 个 train/val 源泄漏组")
        if thumbs:
            reasons.append(f"{len(thumbs)} 个 thumbnail 源")
        if nonjpg:
            reasons.append(f"{len(nonjpg)} 个非 JPG 源路径")
        # [v2.4.3 修 P1] 缺 source_path 默认严格(旧规则"全缺不判"会放行
        # 整体丢路径的构建);label 型数据集须显式 --allow-missing-paths,
        # 且 AID 永不豁免(其路径必然存在,缺=构建链出错)。
        if n_no_path:
            if not args.allow_missing_paths:
                reasons.append(f"{n_no_path} 个样本缺 source_path"
                               f"(label 型数据集请显式 --allow-missing-paths)")
            else:
                aid_no_path = [i for i, v in present.items()
                               if "aid" in v["key"][0].lower() and not v["path"]]
                if aid_no_path:
                    reasons.append(f"{len(aid_no_path)} 个 AID 样本缺 "
                                   f"source_path(AID 不受豁免)")
        json.dump(out, open(args.output, "w"), indent=2, ensure_ascii=False)
        if reasons:
            print(f"\nDUPLICATE GATE: FAIL —— " + ";".join(reasons)
                  + f"(v2.4 修复版 builder 应全零;建库产物不可用,"
                  f"下游训练不许提交)。报告 → {args.output}")
            sys.exit(5)
        print(f"\nDUPLICATE GATE: PASS —— 指纹零重复、源身份齐全且无跨场景/"
              f"thumbnail 违规。报告 → {args.output}")
        return

    # ---- 判决 ----
    vid_dup = out["video_content_dups"]
    traj_dup = out["trajectory_template_dups"]
    leaks = [r for r in vid_dup if r["TRAIN_VAL_LEAK"]]
    # [v2.4.1 修 V2-C3] full_scene 级泄漏(同一源景、不同轨迹/窗口 → 视频指纹
    # 不同但内容强相关)同样是保真泄漏,必须进 exit 4;旧 H5 无 source attrs
    # 时这是唯一能抓到"源级泄漏"的指纹层。轨迹模板级仅报告不判停。
    scene_leaks = [r for r in out["scene_image_dups"] if r["TRAIN_VAL_LEAK"]]
    val_dup_n = sum(max(0, r["n_val"] - 1) for r in vid_dup)
    print("\n" + "=" * 72)
    if vid_dup:
        verdict = "H-full:存在完整重复样本(视频内容逐位相同,仅标签不同)"
    elif traj_dup:
        verdict = "H-traj:仅轨迹模板重复,视频内容各不相同"
    else:
        verdict = "两个假设都不成立 —— 指纹无重复,top-15 一致性需另查(把本输出发回)"
    print(f"判决: {verdict}")
    if args.split:
        print(f"train/val 泄漏组: 视频级 {len(leaks)} + 场景图级 {len(scene_leaks)}"
              f" + 源身份级 {len(src_leaks)}"
              + ("  ⚠️⚠️ 保真指标(SSIM/FVD…)受污染,须先处理再谈任何论文数字"
                 if (leaks or scene_leaks or src_leaks)
                 else "(无泄漏 → 保真数字不受此问题影响)"))
        print(f"val 内部冗余: {val_dup_n} 个(有效 val N = "
              f"{100 - val_dup_n if len(scenes) >= 100 else '?'} / 100,若为完整重复)")
    else:
        print("(未给 --split,泄漏交叉未做 —— 有 split 后重跑可补)")
    out["verdict"] = verdict
    out["n_leak_groups"] = len(leaks)
    out["n_scene_leak_groups"] = len(scene_leaks)
    out["n_source_leak_groups"] = len(src_leaks)
    out["val_redundant"] = val_dup_n

    json.dump(out, open(args.output, "w"), indent=2, ensure_ascii=False)
    print(f"\n[done] → {args.output}")
    if leaks or scene_leaks or src_leaks:
        sys.exit(4)   # 泄漏(视频/场景图/源身份任一级)= 红色警报,编排判停


if __name__ == "__main__":
    main()
