#!/usr/bin/env python3
"""check_resisc45.py — RESISC45(第二数据集/泛化)CPU 冒烟 + 完整性检查器。

为什么需要它:vendored benchmark 的独立 demo loader 仍允许 HF 加载失败后生成
synthetic scene。official wrapper 现已对 RESISC45/WHU fail-close,但必须同时证明:

  1. 请求的官方数据源确实能完全离线加载;
  2. image/label/ClassLabel 与统一 decoder、正式帧预处理兼容;
  3. 四个 scene pool 非空且互斥,没有退成全量池;
  4. 建成的 H5 在每个样本上记录精确 source_dataset,并有 canonical
     trajectory/camera/frame shape。

因此建库前用 --check-source,建库后用 --check-h5。本脚本任何一项失败都以
非零退出,不允许 PatternNet 或 synthetic silent fallback 混入 official tier。

子命令(可组合,全部走同一份聚合 JSON 报告,不产生零散小文件):
    --check-offline          断言离线环境变量已设(不联网)
    --check-source           离线加载 blanchon/RESISC45,统计类/图数,校验分组过滤
    --check-h5 H5            校验建好的 RESISC45 H5 结构完整性
    --check-split H5 SPLIT   校验 split 文件对该 H5 合法且分层均衡
    --make-split H5 OUT      生成 test-only split(全样本为 val,按场景分层交错)
    --self-test              纯 stdlib 自检(ast 解析 + 分组匹配镜像),无需 numpy

用法:
    # 建库前(登录节点,有 HF 缓存)
    python server_run/check_resisc45.py --check-offline --check-source

    # WHU-RS19:自动查 $HF_HOME，也可显式给 repo cache/snapshot/data/parquet
    python server_run/check_resisc45.py --tier whurs19 --check-source \
        --snapshot-root /nix/rsroll_25828/cache/huggingface/hub/datasets--jonathan-roberts1--WHU-RS19

    # 建库后
    python server_run/check_resisc45.py --check-h5 $WORK/data/train_geo.h5 \
        --make-split $WORK/data/resisc45_val_split.json \
        --report $WORK/resisc45_check.json

    DRYRUN=1 python server_run/check_resisc45.py --check-source   # 只打印计划

退出码: 0=全绿, 1=有 FAIL, 2=用法错误。
"""
import argparse
import ast
from collections import Counter
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "lora_remotesense" / "build_dataset_from_benchmark.py"
sys.path.insert(0, str(ROOT))
from lora_remotesense.hf_image_utils import (  # noqa: E402
    WHURS19_LABEL_NAMES,
    WHURS19_REPO,
    decode_hf_image,
    find_hf_snapshot_parquet,
)

# ---- canonical 期望值(全部有出处,勿凭空改)----
# steps_per_segment=9 (build_dataset_from_benchmark.py:207 DEFAULT_STEPS_PER_SEGMENT)
# num_segments ∈ [min_segments=2, max_segments=4] (build_dataset_from_benchmark.py:672-673)
# → [v2.8 ②] 段界去重后 T = 9 + (K-1)×8 ∈ {17, 25, 33}(4n+1)
EXPECTED_FRAME_COUNTS = (17, 25, 33)
# target 480×832 (build_dataset_from_benchmark.py:850-851)
EXPECTED_H, EXPECTED_W = 480, 832
# 4 个 scene group (build_dataset_from_benchmark.py:667/679)
EXPECTED_SCENARIOS = ("general", "water", "urban", "rural")  # [v2.5] flood 更名 water
# convert_benchmark_to_lingbot 每样本落盘的内容 (build_dataset_from_benchmark.py:791-807)
REQUIRED_DATASETS = ("image", "video_frames", "poses", "intrinsics",
                     "waypoints", "segment_angles", "trajectory")
REQUIRED_ATTRS = ("prompt", "num_frames", "num_segments",
                  "source_scene_category", "source_dataset")
OFFLINE_ENV = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")

# --tier 注册表:(HF repo, builder 场景映射名,统一源图 HW|None)。WHU 是异构尺寸,
# 必须由全量 size distribution 描述,不能再伪装成单一 600×600。
# 默认 resisc45,向后兼容;新 tier 的映射在 builder 中用【精确匹配】。
SCENE_GROUPS_NAME = "RESISC45_SCENE_GROUPS"   # main() 按 --tier 覆盖

TIER_REGISTRY = {
    "resisc45":   ("blanchon/RESISC45",            "RESISC45_SCENE_GROUPS",   (256, 256)),
    "patternnet": ("blanchon/PatternNet",          "PATTERNNET_SCENE_GROUPS", (256, 256)),
    "whurs19":    ("jonathan-roberts1/WHU-RS19",   "WHURS19_SCENE_GROUPS",    None),
}

WHURS19_EXPECTED_SAMPLES = 1005
WHURS19_EXPECTED_LABELS = 19
# Reference fingerprint of the server snapshot validated on 2026-08-14.
# A different distribution is reported as WARN rather than rejected: image
# dimensions are data-version metadata, not a benchmark validity constraint.
WHURS19_REFERENCE_SIZE_DISTRIBUTION = {
    (600, 600): 1001,
    (958, 612): 1,
    (726, 456): 1,
    (600, 500): 1,
    (711, 718): 1,
}

RESISC45_REPO = "blanchon/RESISC45"
RESISC45_SPLIT = "train"
RESISC45_EXPECTED_SAMPLES = 31500
RESISC45_RANDOM_PROBES = 8

# RESISC45 源图尺寸(build_dataset_from_benchmark.py:21 注释 & :310 小图拼接分支的前提)
RESISC45_SRC_HW = (256, 256)

# NWPU-RESISC45 上游 45 类名 —— **仅用作 --self-test 的夹具**。
# 权威类名以运行时 --check-source 从 HF 数据集读到的 features['label'].names 为准;
# 本列表不参与 --check-source 的任何判定。
RESISC45_FIXTURE_CLASSES = [
    "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
    "bridge", "chaparral", "church", "circular_farmland", "cloud",
    "commercial_area", "dense_residential", "desert", "forest", "freeway",
    "golf_course", "ground_track_field", "harbor", "industrial_area",
    "intersection", "island", "lake", "meadow", "medium_residential",
    "mobile_home_park", "mountain", "overpass", "palace", "parking_lot",
    "railway", "railway_station", "rectangular_farmland", "river",
    "roundabout", "runway", "sea_ice", "ship", "snowberg",
    "sparse_residential", "stadium", "storage_tank", "tennis_court",
    "terrace", "thermal_power_station", "wetland",
]


# ============================================================================
# 报告收集
# ============================================================================

class Report:
    """收集 PASS/FAIL/WARN + 结构化数据;最后一次性写一个 JSON(不产生零散文件)。"""

    def __init__(self):
        self.checks = []
        self.data = {}
        self.n_fail = 0

    def ok(self, name, msg=""):
        self.checks.append({"check": name, "status": "PASS", "msg": msg})
        print(f"  [PASS] {name}" + (f" — {msg}" if msg else ""))

    def fail(self, name, msg):
        self.checks.append({"check": name, "status": "FAIL", "msg": msg})
        self.n_fail += 1
        print(f"  [FAIL] {name} — {msg}", file=sys.stderr)

    def warn(self, name, msg):
        self.checks.append({"check": name, "status": "WARN", "msg": msg})
        print(f"  [WARN] {name} — {msg}")

    def expect(self, cond, name, msg_fail, msg_ok=""):
        if cond:
            self.ok(name, msg_ok)
        else:
            self.fail(name, msg_fail)
        return bool(cond)

    def section(self, title):
        print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))

    def finish(self, report_path=None):
        n_pass = sum(1 for c in self.checks if c["status"] == "PASS")
        n_warn = sum(1 for c in self.checks if c["status"] == "WARN")
        print(f"\n{'=' * 66}")
        print(f"汇总: {n_pass} PASS / {n_warn} WARN / {self.n_fail} FAIL")
        if report_path:
            p = Path(report_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {"checks": self.checks, "data": self.data,
                       "n_pass": n_pass, "n_warn": n_warn, "n_fail": self.n_fail}
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            print(f"报告: {p}")
        if self.n_fail:
            print("结果: FAIL —— 见上面 [FAIL] 行,先修再建库/评估。", file=sys.stderr)
            return 1
        print("结果: 全绿 ✓")
        return 0


# ============================================================================
# 从 builder 源码提取 RESISC45_SCENE_GROUPS(ast,不 import → 不需要 numpy/h5py)
# ============================================================================

_TAXONOMY_JSON = Path(__file__).resolve().parent.parent / "lora_remotesense" / "scene_taxonomy.json"
_GROUPS_NAME_TO_DATASET = {
    "AID_SCENE_GROUPS": "blanchon/AID",
    "RESISC45_SCENE_GROUPS": "blanchon/RESISC45",
    "PATTERNNET_SCENE_GROUPS": "blanchon/PatternNet",
    "WHURS19_SCENE_GROUPS": "jonathan-roberts1/WHU-RS19",
}


def extract_scene_groups(builder_path=BUILDER, name="RESISC45_SCENE_GROUPS"):
    """[v2.5] 场景组唯一真源已移到 scene_taxonomy.json —— 直读 JSON
    (stdlib,零重依赖;builder_path 参数保留仅为签名兼容)。旧 AST 提取
    法在组变成 JSON 加载后会取不到字面量,已废。"""
    tax = json.loads(_TAXONOMY_JSON.read_text(encoding="utf-8"))
    ds = _GROUPS_NAME_TO_DATASET.get(name)
    if ds is None or ds not in tax["groups"]:
        raise RuntimeError(f"scene_taxonomy.json 里找不到 {name} → {ds}")
    return tax["groups"][ds]


def mirror_match_classes(label_names, target_classes):
    """Mirror the builder's alphanumeric-normalized *exact* class match."""
    normalize = lambda value: "".join(  # noqa: E731 - compact mirror
        ch.lower() for ch in str(value) if ch.isalnum())
    targets = {normalize(value) for value in target_classes}
    return [name for name in label_names if normalize(name) in targets]


# ============================================================================
# --check-offline
# ============================================================================

def check_offline(rep, dryrun=False):
    rep.section("离线断言(不得触网)")
    if dryrun:
        print("  [DRYRUN] 将断言 " + ", ".join(OFFLINE_ENV) + " 均为 1")
        return
    env = {}
    for var in OFFLINE_ENV:
        val = os.environ.get(var, "")
        env[var] = val
        rep.expect(val == "1", f"offline_env:{var}",
                   f"{var}='{val}' (期望 '1');计算节点断网,联网调用会挂起或报错。"
                   f" 修:export {var}=1(server_run/pbs/_pbs_env.sh:34-36 已默认设好)",
                   f"{var}=1")
    rep.data["offline_env"] = env
    # HF_HOME 必须指向预取好的缓存
    hf_home = os.environ.get("HF_HOME", "")
    rep.data["HF_HOME"] = hf_home
    if not hf_home:
        rep.warn("offline_env:HF_HOME",
                 "HF_HOME 未设；拒绝假设用户主目录缓存。"
                 "A800 应先 source server_run/a800_env.sh；"
                 "Gadi 应指向 ASSET_ROOT/hf_cache(_pbs_env.sh:28)")
    elif not Path(hf_home).is_dir():
        rep.fail("offline_env:HF_HOME", f"HF_HOME={hf_home} 不存在")
    else:
        rep.ok("offline_env:HF_HOME", hf_home)


# ============================================================================
# --check-source
# ============================================================================


def _load_source_dataset(load_dataset, *, snapshot_root=None):
    """Load the requested tier without ever substituting another dataset."""
    if RESISC45_REPO == WHURS19_REPO:
        parquet = find_hf_snapshot_parquet(
            WHURS19_REPO,
            hf_home=os.environ.get("HF_HOME") or None,
            snapshot_root=snapshot_root,
        )
        if snapshot_root and parquet is None:
            raise FileNotFoundError(
                f"--snapshot-root={snapshot_root} 下找不到 data/train-*.parquet")
        if parquet is not None:
            ds = load_dataset(
                "parquet", data_files=str(parquet), split=RESISC45_SPLIT)
            return ds, f"local parquet:{parquet}", parquet
    ds = load_dataset(RESISC45_REPO, split=RESISC45_SPLIT)
    return ds, f"HF repo:{RESISC45_REPO}", None


def _check_whurs19_integrity(rep, ds):
    """Decode every WHU record and return its authoritative label-name map."""
    rep.section("WHU-RS19 真实 Parquet 完整性(全量 CPU 解码)")
    size_counts = Counter()
    mode_counts = Counter()
    label_counts = Counter()
    decode_errors = []

    for i in range(len(ds)):
        try:
            row = ds[i]
            label_counts[int(row["label"])] += 1
        except Exception as exc:
            decode_errors.append({"index": i, "stage": "row/label",
                                  "error": f"{type(exc).__name__}: {exc}"})
            continue
        try:
            image = decode_hf_image(row["image"], context=f"WHU-RS19[{i}].image")
            try:
                size_counts[tuple(int(v) for v in image.size)] += 1  # (W,H)
                mode_counts[str(image.mode)] += 1
            finally:
                image.close()
        except Exception as exc:
            decode_errors.append({"index": i, "stage": "decode",
                                  "error": f"{type(exc).__name__}: {exc}"})

    n_ok = len(ds) - len(decode_errors)
    label_ids = sorted(label_counts)
    rep.data["whurs19_integrity"] = {
        "sample_count": len(ds),
        "decode_success": n_ok,
        "decode_errors": decode_errors,
        "label_ids": label_ids,
        "unique_labels": len(label_ids),
        "image_modes": dict(sorted(mode_counts.items())),
        "image_size_distribution_wh": {
            f"{w}x{h}": count
            for (w, h), count in sorted(size_counts.items())
        },
    }
    rep.expect(len(ds) == WHURS19_EXPECTED_SAMPLES, "whurs19:sample_count",
               f"样本数={len(ds)},当前官方快照期望 {WHURS19_EXPECTED_SAMPLES}",
               f"{len(ds)}")
    rep.expect(not decode_errors, "whurs19:decode_errors",
               f"{len(decode_errors)} 个解码/标签错误: {decode_errors[:5]}", "0")
    rep.expect(n_ok == WHURS19_EXPECTED_SAMPLES, "whurs19:decode_success",
               f"成功 {n_ok},期望 {WHURS19_EXPECTED_SAMPLES}", f"{n_ok}")
    rep.expect(len(label_ids) == WHURS19_EXPECTED_LABELS,
               "whurs19:unique_labels",
               f"唯一标签={len(label_ids)},ids={label_ids};期望 19",
               f"19, ids={label_ids}")
    rep.expect(label_ids == list(range(WHURS19_EXPECTED_LABELS)),
               "whurs19:label_ids", f"ids={label_ids},期望 0..18", "0..18")
    rep.expect(mode_counts == Counter({"RGB": WHURS19_EXPECTED_SAMPLES}),
               "whurs19:rgb_samples",
               f"mode 分布={dict(mode_counts)},期望 RGB:{WHURS19_EXPECTED_SAMPLES}",
               f"RGB:{WHURS19_EXPECTED_SAMPLES}")

    observed_sizes = dict(size_counts)
    size_msg = ", ".join(f"{w}x{h}:{n}" for (w, h), n in sorted(observed_sizes.items()))
    if observed_sizes == WHURS19_REFERENCE_SIZE_DISTRIBUTION:
        rep.ok("whurs19:size_distribution", size_msg)
    else:
        rep.warn(
            "whurs19:size_distribution",
            f"观测 {size_msg};与 2026-08-14 参考快照不同。尺寸分布只作版本指纹,"
            "不会据此删除样本或判定其他合法版本失败。")

    feature_names = []
    try:
        feature_names = list(getattr(ds.features.get("label"), "names", []) or [])
    except Exception:
        feature_names = []
    if feature_names:
        mapping_ok = rep.expect(
            tuple(feature_names) == WHURS19_LABEL_NAMES,
            "whurs19:classlabel_mapping",
            f"ClassLabel.names={feature_names},与官方映射不一致",
            "Parquet ClassLabel.names 与官方 19 类映射一致")
        return feature_names if mapping_ok else []
    if label_ids == list(range(WHURS19_EXPECTED_LABELS)):
        rep.ok("whurs19:classlabel_mapping",
               "Parquet 暴露 plain int64;使用官方 dataset card 的固定 0..18 映射")
        return list(WHURS19_LABEL_NAMES)
    return []


def _check_resisc45_random_records(rep, ds, label_names, builder_module):
    """Decode and preprocess deterministic random real RESISC45 records.

    The full builder smoke remains the end-to-end stitching/trajectory check.
    This cheaper gate uses the same unified decoder and final frame-resize
    helper, catching malformed image structs, missing labels, and bad tensors
    before a build starts.
    """
    import numpy as np

    count = min(RESISC45_RANDOM_PROBES, len(ds))
    indices = sorted(random.Random(42).sample(range(len(ds)), count))

    class DecodeAdapter:
        @staticmethod
        def _to_numpy_image(image):
            return np.asarray(image).astype(np.float32)

    adapter = DecodeAdapter()
    failures = []
    labels_seen = []
    sizes = Counter()
    for index in indices:
        try:
            row = ds[index]
            if "image" not in row or "label" not in row:
                raise KeyError("record must contain both image and label")
            label = int(row["label"])
            if not 0 <= label < len(label_names):
                raise ValueError(
                    f"label={label} outside ClassLabel range 0..{len(label_names) - 1}")
            arr = builder_module._decode_source_array(
                adapter, row["image"], context=f"{RESISC45_REPO}[{index}].image")
            sizes[tuple(int(v) for v in arr.shape[:2])] += 1
            resized = builder_module._resize_frame(
                arr, (EXPECTED_H, EXPECTED_W))
            if tuple(resized.shape) != (EXPECTED_H, EXPECTED_W, 3):
                raise ValueError(f"preprocessed shape={tuple(resized.shape)}")
            if not np.isfinite(resized).all():
                raise ValueError("preprocessed tensor contains non-finite values")
            labels_seen.append(label_names[label])
        except Exception as exc:
            failures.append(f"index={index}: {type(exc).__name__}: {exc}")

    rep.data["resisc45_random_probe"] = {
        "indices": indices,
        "labels": labels_seen,
        "source_sizes": {f"{h}x{w}": n for (h, w), n in sizes.items()},
        "target_shape": [EXPECTED_H, EXPECTED_W, 3],
    }
    if failures:
        rep.fail("resisc45:random_preprocessing", "; ".join(failures))
    else:
        rep.ok(
            "resisc45:random_preprocessing",
            f"{len(indices)}/{len(indices)} 随机样本 image+label 解码并完成正式帧预处理;"
            f"labels={labels_seen}")


def check_source(rep, dryrun=False, snapshot_root=None):
    """离线加载指定 tier,统计源数据并校验 4 个 scene group。"""
    rep.section(f"源数据集离线校验 {RESISC45_REPO}")
    if dryrun:
        whu_plan = (";优先从 --snapshot-root/HF_HOME 的 data/train-*.parquet 读取并"
                    "逐张解码 1005 张" if RESISC45_REPO == WHURS19_REPO else "")
        print(f"  [DRYRUN] 将强制 HF_HUB_OFFLINE=1 并加载 '{RESISC45_REPO}'"
              f" split='{RESISC45_SPLIT}'{whu_plan};统计类/图数 + 分组过滤有效性")
        return

    # 强制离线:本检查的意义就是证明**断网也能加载**
    for var in OFFLINE_ENV:
        os.environ[var] = "1"

    try:
        from datasets import load_dataset
    except Exception as e:
        rep.fail("source:import_datasets", f"import datasets 失败: {e}")
        return

    try:
        ds, load_origin, parquet = _load_source_dataset(
            load_dataset, snapshot_root=snapshot_root)
    except Exception as e:
        rep.fail("source:load_dataset",
                 f"离线加载 {RESISC45_REPO}[{RESISC45_SPLIT}] 失败: "
                 f"{type(e).__name__}: {e}\n"
                 f"         → 检查 HF_HOME/hub/datasets--{RESISC45_REPO.replace('/', '--')}\n"
                 f"         → official tier 禁止改用 RESISC45/PatternNet/合成场景")
        return
    rep.ok("source:load_dataset", f"{load_origin};{len(ds)} 张图")
    rep.ok("source:identity", f"requested source={RESISC45_REPO};未启用替代数据集")
    rep.data["source_repo"] = RESISC45_REPO
    rep.data["source_load_origin"] = load_origin
    rep.data["source_parquet"] = str(parquet) if parquet else None
    rep.data["source_num_images"] = len(ds)

    # ---- 类名 / WHU 全量完整性 ----
    if RESISC45_REPO == WHURS19_REPO:
        label_names = _check_whurs19_integrity(rep, ds)
    else:
        if RESISC45_REPO == "blanchon/RESISC45":
            rep.expect(
                len(ds) == RESISC45_EXPECTED_SAMPLES,
                "resisc45:sample_count",
                f"样本数={len(ds)},期望 {RESISC45_EXPECTED_SAMPLES}",
                f"{RESISC45_EXPECTED_SAMPLES} 张图",
            )
        label_names = None
        try:
            feat = ds.features.get("label")
            label_names = list(getattr(feat, "names", []) or [])
        except Exception as e:
            rep.fail("source:label_feature", f"读取 features['label'] 失败: {e}")
        if not label_names:
            rep.fail("source:label_feature", "数据集没有带 names 的 label 特征")
            return
        rep.expect(len(label_names) == 45, "source:num_classes",
                   f"类数={len(label_names)},期望 45(RESISC45)。类名: {label_names}",
                   "45 类")
        try:
            image = decode_hf_image(ds[0]["image"], context=f"{RESISC45_REPO}[0].image")
            try:
                src_hw = (image.size[1], image.size[0])
            finally:
                image.close()
            rep.data["source_image_hw"] = list(src_hw)
            if src_hw == RESISC45_SRC_HW:
                rep.ok("source:image_size", f"{src_hw[0]}×{src_hw[1]}")
            else:
                rep.warn("source:image_size",
                         f"{src_hw[0]}×{src_hw[1]} != 预期 {RESISC45_SRC_HW}")
        except Exception as e:
            rep.warn("source:image_size", f"探测首图尺寸失败(不阻断): {e}")

    if not label_names:
        rep.fail("source:label_feature", "无法建立标签 id→名称映射,禁止场景过滤")
        return
    rep.data["source_label_names"] = label_names

    # ---- 分组过滤有效性:调用 builder 真身,不用镜像 ----
    groups = extract_scene_groups(name=SCENE_GROUPS_NAME)
    rep.data["scene_groups"] = groups
    sys.path.insert(0, str(ROOT))
    try:
        from lora_remotesense import build_dataset_from_benchmark as builder_module
        _get_scene_filter_indices = builder_module._get_scene_filter_indices
    except Exception as e:
        rep.fail("source:import_builder",
                 f"import build_dataset_from_benchmark 失败: {e}")
        return

    if RESISC45_REPO == "blanchon/RESISC45":
        _check_resisc45_random_records(rep, ds, label_names, builder_module)

    pool = {}
    covered = set()
    for cat in EXPECTED_SCENARIOS:
        matched = mirror_match_classes(label_names, groups.get(cat, []))
        covered.update(matched)
        try:
            idx = _get_scene_filter_indices(ds, cat, RESISC45_REPO)
        except Exception as exc:
            rep.fail(f"source:filter:{cat}",
                     f"{type(exc).__name__}: {exc}")
            pool[cat] = {"num_images": 0, "matched_classes": matched}
            continue
        pool[cat] = {"num_images": len(idx), "matched_classes": matched}
        if len(idx) == 0:
            rep.fail(f"source:filter:{cat}", "过滤出 0 张图")
        elif len(idx) == len(ds):
            # 真身在「匹配不到任何类」时会静默 return list(range(n)) —— 正是这个坑
            rep.fail(f"source:filter:{cat}",
                     f"过滤后 = 全量 {len(ds)} 张 → 过滤**静默失效**"
                     f"(build_dataset_from_benchmark.py:248-249)。"
                     f"RESISC45_SCENE_GROUPS['{cat}']={groups.get(cat)} 与真实类名对不上。"
                     f" 后果:该 group 抽全 45 类,prompt 的场景词(_make_prompt desc_map)"
                     f"与像素不符。")
        else:
            rep.ok(f"source:filter:{cat}",
                   f"{len(idx)} 张 / {len(matched)} 类: {matched}")

    rep.data["scene_pools"] = pool
    uncovered = sorted(set(label_names) - covered)
    rep.data["uncovered_classes"] = uncovered
    print(f"\n  覆盖 {len(covered)}/{len(label_names)} 类;"
          f"未被任何 group 采样的 {len(uncovered)} 类: {uncovered}")
    rep.warn("source:class_coverage",
             f"RESISC45_SCENE_GROUPS 只覆盖 {len(covered)}/{len(label_names)} 类"
             f"(设计如此,非 bug),论文须写明「RESISC45 的 {len(covered)} 类子集」"
             f"而不是「RESISC45 全 45 类」") if uncovered else rep.ok(
        "source:class_coverage", "覆盖全部类")

    # 每场景池子够不够(同 check_aid_pools.py 的地理泄漏逻辑;test-only 不切 train
    # 所以不致命,但池子太小 → scene_id % len(pool) 回绕 → 重复源图)
    for cat, info in pool.items():
        if info["num_images"] < 32:
            rep.warn(f"source:pool_size:{cat}",
                     f"池子仅 {info['num_images']} 张,建库时 scene_id % len(pool) "
                     f"会回绕复用源图")

    if RESISC45_REPO == WHURS19_REPO:
        rep.data["fallback_allowed"] = False
        rep.ok("whurs19:fail_close",
               "official tier=whurs19 只接受 jonathan-roberts1/WHU-RS19;"
               "builder 缺场景或来源名不符会抛 SceneSelectionError")
    elif RESISC45_REPO == "blanchon/RESISC45":
        rep.data["fallback_allowed"] = False
        rep.ok("resisc45:fail_close",
               "official tier=resisc45 只接受 blanchon/RESISC45;"
               "builder 缺场景或来源名不符会抛 SceneSelectionError")


# ============================================================================
# --check-h5
# ============================================================================

def _as_str(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    return str(v)


def check_h5(rep, h5_path, dryrun=False, expected_n=None):
    rep.section(f"H5 完整性 {h5_path}")
    if dryrun:
        print(f"  [DRYRUN] 将校验 {h5_path}: 样本数"
              f"{f'={expected_n}' if expected_n is not None else ''}/必需 dataset/attrs/"
              f"帧数∈{EXPECTED_FRAME_COUNTS}/分辨率 {EXPECTED_H}×{EXPECTED_W}/场景分布")
        return None

    p = Path(h5_path)
    if not p.is_file():
        rep.fail("h5:exists", f"{p} 不存在")
        return None

    import h5py                      # 延迟 import:--check-offline/--self-test 不需要
    import numpy as np

    scen_counts = {}
    frame_counts = {}
    per_sample_cat = []
    source_datasets = Counter()
    identity_poses = 0
    n_seen = 0

    with h5py.File(p, "r") as f:
        if "data" not in f or "metadata" not in f:
            rep.fail("h5:groups", "缺 data/ 或 metadata/ 组")
            return None
        meta = dict(f["metadata"].attrs)
        rep.data["h5_metadata"] = {k: _as_str(v) for k, v in meta.items()}
        keys = sorted(f["data"].keys())
        n = len(keys)
        rep.expect(n > 0, "h5:num_samples", "H5 里 0 个样本", f"{n} 个样本")
        if n == 0:
            return None
        if expected_n is not None:
            rep.expect(
                n == int(expected_n),
                "h5:expected_num_samples",
                f"实际 {n} 个样本 != 调度 smoke 期望 {expected_n}",
                f"实际样本数={n}=期望值",
            )

        declared = int(meta.get("total_instances", -1))
        rep.expect(declared == n, "h5:total_instances",
                   f"metadata.total_instances={declared} != 实际 {n}",
                   f"total_instances={n} 一致")

        tgt = meta.get("target_size", None)
        if tgt is not None:
            rep.expect(tuple(int(x) for x in tgt) == (EXPECTED_H, EXPECTED_W),
                       "h5:target_size",
                       f"target_size={list(tgt)} != canonical ({EXPECTED_H},{EXPECTED_W})",
                       f"{EXPECTED_H}×{EXPECTED_W}")
        rep.expect("pose_window_size" in meta, "h5:pose_window_size",
                   "metadata 缺 pose_window_size(Plücker 平移尺度靠它)",
                   f"pose_window_size={list(meta.get('pose_window_size', []))}")

        for key in keys:
            g = f["data"][key]
            n_seen += 1

            missing_ds = [d for d in REQUIRED_DATASETS if d not in g]
            if missing_ds:
                rep.fail(f"h5:datasets:{key}",
                         f"缺 dataset {missing_ds}"
                         + ("(waypoints/segment_angles 只在有真实路点时写入,"
                            "build_dataset_from_benchmark.py:804-807 → 缺=该样本走了"
                            "无路点回退,Track B 指标无法打分)"
                            if {"waypoints", "segment_angles"} & set(missing_ds) else ""))
                continue
            missing_at = [a for a in REQUIRED_ATTRS if a not in g.attrs]
            if missing_at:
                rep.fail(f"h5:attrs:{key}", f"缺 attrs {missing_at}")
                continue

            T = int(g.attrs["num_frames"])
            K = int(g.attrs["num_segments"])
            cat = _as_str(g.attrs["source_scene_category"])
            source_dataset = _as_str(g.attrs["source_dataset"])
            source_datasets[source_dataset] += 1
            prompt = _as_str(g.attrs["prompt"])
            per_sample_cat.append(cat)
            scen_counts[cat] = scen_counts.get(cat, 0) + 1
            frame_counts[T] = frame_counts.get(T, 0) + 1

            if T not in EXPECTED_FRAME_COUNTS:
                rep.fail(f"h5:num_frames:{key}",
                         f"T={T} 不在 {EXPECTED_FRAME_COUNTS}(=steps 9 × segments 2/3/4)")
            expected_t = 9 + max(0, K - 1) * 8
            if expected_t != T:
                rep.fail(f"h5:frames_vs_segments:{key}",
                         f"num_segments={K},段界去重后期望 T={expected_t},"
                         f"实际 num_frames={T}")

            vf = g["video_frames"]
            if tuple(vf.shape) != (T, EXPECTED_H, EXPECTED_W, 3):
                rep.fail(f"h5:video_frames:{key}",
                         f"shape={tuple(vf.shape)} != ({T},{EXPECTED_H},{EXPECTED_W},3)")
            if tuple(g["image"].shape) != (EXPECTED_H, EXPECTED_W, 3):
                rep.fail(f"h5:image:{key}",
                         f"shape={tuple(g['image'].shape)} != ({EXPECTED_H},{EXPECTED_W},3)")
            if tuple(g["poses"].shape) != (T, 4, 4):
                rep.fail(f"h5:poses:{key}", f"shape={tuple(g['poses'].shape)} != ({T},4,4)")
            if tuple(g["intrinsics"].shape) != (T, 4):
                rep.fail(f"h5:intrinsics:{key}",
                         f"shape={tuple(g['intrinsics'].shape)} != ({T},4)")
            if tuple(g["trajectory"].shape) != (T, 2):
                rep.fail(f"h5:trajectory:{key}",
                         f"shape={tuple(g['trajectory'].shape)} != ({T},2)")
            wp = g["waypoints"]
            if tuple(wp.shape) != (K + 1, 2):
                rep.fail(f"h5:waypoints:{key}",
                         f"shape={tuple(wp.shape)} != (K+1,2)=({K + 1},2)")
            sa = g["segment_angles"]
            if tuple(sa.shape) != (K,):
                rep.fail(f"h5:segment_angles:{key}",
                         f"shape={tuple(sa.shape)} != (K,)=({K},)")
            if cat not in EXPECTED_SCENARIOS:
                rep.fail(f"h5:scene_category:{key}",
                         f"'{cat}' 不在 {EXPECTED_SCENARIOS}")
            if not prompt.strip():
                rep.fail(f"h5:prompt:{key}", "prompt 为空")

            if np.allclose(g["poses"][0], np.eye(4)):
                identity_poses += 1

    # ---- 场景分布 ----
    rep.data["scenario_counts"] = scen_counts
    rep.data["frame_counts"] = {str(k): v for k, v in frame_counts.items()}
    rep.data["source_datasets"] = dict(source_datasets)
    print(f"\n  场景分布: {scen_counts}")
    print(f"  帧数分布: {frame_counts}")
    for cat in EXPECTED_SCENARIOS:
        rep.expect(scen_counts.get(cat, 0) > 0, f"h5:scenario_present:{cat}",
                   f"场景 '{cat}' 0 个样本 → 该 group 建库时全挂",
                   f"{scen_counts.get(cat, 0)} 个")

    rep.expect(
        set(source_datasets) == {RESISC45_REPO}
        and source_datasets.get(RESISC45_REPO, 0) == n_seen,
        "h5:source_dataset",
        f"per-sample source_dataset 分布={dict(source_datasets)},"
        f"期望全部为 {RESISC45_REPO}",
        f"{n_seen}/{n_seen} 样本 source_dataset={RESISC45_REPO}",
    )

    # ---- 通道判定(sem=identity / geo=waypoint Plücker)----
    frac_id = identity_poses / max(1, n_seen)
    rep.data["identity_pose_fraction"] = frac_id
    if frac_id > 0.99:
        rep.ok("h5:pose_channel",
               f"{identity_poses}/{n_seen} 位姿为 identity → 这是**语义通道**(SEM_H5)")
    elif frac_id < 0.01:
        rep.ok("h5:pose_channel",
               f"位姿非 identity → 这是**几何通道**(GEO_H5,waypoint Plücker)")
    else:
        rep.fail("h5:pose_channel",
                 f"位姿混杂({identity_poses}/{n_seen} 为 identity)—— "
                 f"既不是干净的 SEM 也不是干净的 GEO")

    # ---- 文件级来源标记是通用构建器名;真源由逐样本 attrs 硬校验 ----
    src = _as_str(rep.data.get("h5_metadata", {}).get("source", ""))
    rep.ok("h5:source_provenance",
           f"metadata.source='{src}';逐样本 source_dataset 已严格验证为 "
           f"{RESISC45_REPO},可排除 PatternNet/synthetic silent fallback")

    return per_sample_cat


# ============================================================================
# --make-split / --check-split
# ============================================================================

def _read_categories(h5_path):
    """按 sorted(keys) 顺序读每个样本的 source_scene_category。

    顺序很关键:evaluate.py:109-115 用 keys=sorted(f['data'].keys()); key=keys[idx],
    所以 split 里的 index 就是**排序后**的位置。
    """
    import h5py
    cats = []
    with h5py.File(h5_path, "r") as f:
        for key in sorted(f["data"].keys()):
            cats.append(_as_str(f["data"][key].attrs.get("source_scene_category", "?")))
    return cats


def _interleave_by_category(cats):
    """把「按场景成块排列」的样本索引重排成跨场景**交错**的顺序(纯函数,可自检)。

    为什么必须分层:benchmark v5 是一个 scene 一个 scene 顺序写的
    (build_multi_direction_benchmark_v5.py:717-727),所以 sample_00000..N/4
    全是 general。common.sh 的 eval_lora 把 EVAL_N 透传成 evaluate.py 的
    --max_samples,而 evaluate.py:110-115 是按 val_indices **顺序**取前 N 条 →
    不分层的话「评 50 条」= 只评 general 一个场景,泛化结论直接作废。

    返回 [general_0, water_0, urban_0, rural_0, general_1, ...] —— 取前 N 条自动
    跨 4 场景近似均衡。确定性、无随机种子;各场景数量不等时短的桶先耗尽,长的桶
    继续按序补齐(仍是每个索引恰好出现一次)。
    """
    buckets = {}
    for i, c in enumerate(cats):
        buckets.setdefault(c, []).append(i)
    order = sorted(buckets.keys())
    interleaved = []
    pos = 0
    while len(interleaved) < len(cats):
        added = False
        for c in order:
            if pos < len(buckets[c]):
                interleaved.append(buckets[c][pos])
                added = True
        if not added:
            break
        pos += 1
    return interleaved, {c: len(v) for c, v in buckets.items()}


def make_split(rep, h5_path, out_path, dryrun=False):
    """生成 test-only split:全部样本为 val,按场景分层交错(见 _interleave_by_category)。"""
    rep.section(f"生成 test-only split → {out_path}")
    if dryrun:
        print(f"  [DRYRUN] 将从 {h5_path} 读场景标签,写分层交错的全量 val split 到 {out_path}")
        return
    if not Path(h5_path).is_file():
        rep.fail("split:src_h5", f"{h5_path} 不存在")
        return

    cats = _read_categories(h5_path)
    n = len(cats)
    interleaved, scen_counts = _interleave_by_category(cats)

    if len(interleaved) != n or len(set(interleaved)) != n:
        rep.fail("split:interleave",
                 f"交错后 {len(interleaved)} 条 / {len(set(interleaved))} 个唯一索引 "
                 f"!= 样本数 {n}")
        return

    split = {
        # evaluate.py:57-61 只读 val_indices;其余字段是溯源信息
        "val_indices": interleaved,
        "train_indices": [],
        "train_samples": 0,
        "val_samples": n,
        "_note": (
            "TEST-ONLY generalization split:RESISC45 从未参与训练,所以全部样本"
            "都是合法的未见样本,val=全量、train=空。index = sorted(f['data'].keys()) "
            "里的位置(与 evaluate.py:109-115 一致)。顺序按场景分层交错,"
            "取前 N 条即近似 4 场景均衡。"
        ),
        "_source_h5": str(Path(h5_path).resolve()),
        "_scenario_counts": scen_counts,
        "_generated_by": "server_run/check_resisc45.py --make-split",
    }
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, ensure_ascii=False)
    rep.ok("split:written",
           f"{n} 条全部为 val,分层交错;场景 {split['_scenario_counts']} → {outp}")
    rep.data["split_path"] = str(outp)
    rep.data["split_scenario_counts"] = split["_scenario_counts"]


def check_split(rep, h5_path, split_path, dryrun=False):
    rep.section(f"split 合法性 {split_path}")
    if dryrun:
        print(f"  [DRYRUN] 将校验 {split_path} 相对 {h5_path} 的索引合法性与分层均衡")
        return
    sp = Path(split_path)
    if not sp.is_file():
        rep.fail("split:exists", f"{sp} 不存在")
        return
    with open(sp, "r", encoding="utf-8") as f:
        split = json.load(f)
    if "val_indices" not in split:
        rep.fail("split:schema", "缺 val_indices(evaluate.py:57-61 只认这个键)")
        return
    vi = split["val_indices"]
    cats = _read_categories(h5_path)
    n = len(cats)

    rep.expect(len(vi) > 0, "split:non_empty", "val_indices 为空", f"{len(vi)} 条")
    rep.expect(len(set(vi)) == len(vi), "split:unique",
               f"val_indices 有重复({len(vi) - len(set(vi))} 条)", "无重复")
    oob = [i for i in vi if not (0 <= i < n)]
    rep.expect(not oob, "split:in_range",
               f"{len(oob)} 个索引越界(H5 只有 {n} 个样本);"
               f"evaluate.py:113-114 会**静默跳过**越界索引 → 评估样本数悄悄变少。"
               f"最常见原因:错用了 AID 的 train_val_split.json(索引指向另一个 H5)",
               f"全部落在 [0,{n})")

    # 分层均衡:检查前 K 条的场景覆盖(K = 常用 EVAL_N)
    for k in (10, 50):
        if len(vi) >= k:
            prefix = [cats[i] for i in vi[:k] if 0 <= i < n]
            covered = sorted(set(prefix))
            rep.expect(len(covered) == len(EXPECTED_SCENARIOS),
                       f"split:stratified_prefix_{k}",
                       f"前 {k} 条只覆盖场景 {covered} —— 未分层。"
                       f"eval 用 --max_samples {k} 会只评这些场景。"
                       f"修:用 --make-split 重新生成",
                       f"前 {k} 条覆盖全部 4 场景")


# ============================================================================
# --self-test(纯 stdlib,可在无 numpy 的机器上跑)
# ============================================================================

def self_test(rep):
    rep.section("自检(ast 提取 + 分组匹配镜像;夹具类名,非权威)")
    groups = extract_scene_groups(name=SCENE_GROUPS_NAME)
    fixture_classes = (list(WHURS19_LABEL_NAMES)
                       if RESISC45_REPO == WHURS19_REPO
                       else RESISC45_FIXTURE_CLASSES)
    rep.expect(set(groups.keys()) == set(EXPECTED_SCENARIOS), "selftest:groups_keys",
               f"RESISC45_SCENE_GROUPS 的 key={sorted(groups.keys())} "
               f"!= {sorted(EXPECTED_SCENARIOS)}",
               f"4 个 scene group: {sorted(groups.keys())}")
    rep.data["scene_groups"] = groups

    covered = set()
    for cat in EXPECTED_SCENARIOS:
        matched = mirror_match_classes(fixture_classes, groups.get(cat, []))
        covered.update(matched)
        rep.expect(len(matched) > 0, f"selftest:match:{cat}",
                   f"group '{cat}' 在夹具类名上匹配到 0 类 → 真数据上会触发 "
                   f"build_dataset_from_benchmark.py:248-249 的静默全量回退",
                   f"{len(matched)} 类: {matched}")
    unc = sorted(set(fixture_classes) - covered)
    rep.data["selftest_covered"] = sorted(covered)
    rep.data["selftest_uncovered"] = unc
    print(f"\n  夹具覆盖 {len(covered)}/{len(fixture_classes)} 类")
    print(f"  未覆盖 {len(unc)} 类: {unc}")
    print("  (权威结论以 --check-source 在真实 HF 缓存上的输出为准)")

    # ---- 分层交错(--make-split 的核心逻辑)----
    # 夹具 = builder 的真实写出顺序:一个场景一整块
    # (build_multi_direction_benchmark_v5.py:717-727)
    blocked = ([EXPECTED_SCENARIOS[0]] * 50 + [EXPECTED_SCENARIOS[1]] * 50 +
               [EXPECTED_SCENARIOS[2]] * 50 + [EXPECTED_SCENARIOS[3]] * 50)
    idx, counts = _interleave_by_category(blocked)
    rep.expect(len(idx) == 200 and len(set(idx)) == 200, "selftest:interleave_bijective",
               f"交错后 {len(idx)} 条 / {len(set(idx))} 唯一,期望 200/200(必须是双射)",
               "200 条索引,无重复无丢失")
    rep.expect(sorted(idx) == list(range(200)), "selftest:interleave_permutation",
               "交错结果不是 0..199 的排列", "是 0..199 的排列")
    prefix = sorted(set(blocked[i] for i in idx[:8]))
    rep.expect(len(prefix) == 4, "selftest:interleave_prefix",
               f"前 8 条只覆盖 {prefix},未跨 4 场景 → eval --max_samples 会单场景偏置",
               f"前 8 条即覆盖全部 4 场景: {prefix}")
    # 不等长桶:短桶耗尽后仍须保持双射
    ragged = ["general"] * 7 + ["water"] * 2 + ["urban"] * 1
    ridx, _ = _interleave_by_category(ragged)
    rep.expect(sorted(ridx) == list(range(10)), "selftest:interleave_ragged",
               f"不等长桶下交错结果 {sorted(ridx)} != 0..9 的排列(短桶耗尽后丢样本)",
               "不等长桶(7/2/1)下仍是完整排列")


# ============================================================================
# main
# ============================================================================

def main():
    # --tier 需要覆盖这三个模块级常量;global 必须先于任何使用声明。
    global RESISC45_REPO, RESISC45_SRC_HW, SCENE_GROUPS_NAME
    ap = argparse.ArgumentParser(
        description="RESISC45 第二数据集 CPU 冒烟/完整性检查器(任一失败 → 非零退出)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法:")[-1] if "用法:" in __doc__ else None)
    ap.add_argument("--check-offline", action="store_true", help="断言离线环境变量已设")
    ap.add_argument("--check-source", action="store_true",
                    help=f"离线加载 {RESISC45_REPO} 并校验分组过滤")
    ap.add_argument("--check-h5", metavar="H5", default=None, help="校验 RESISC45 H5 完整性")
    ap.add_argument("--expect-n", type=int, default=None,
                    help="配合 --check-h5:要求 H5 恰好包含 N 个样本")
    ap.add_argument("--check-split", nargs=2, metavar=("H5", "SPLIT"), default=None,
                    help="校验 split 对该 H5 合法且分层")
    ap.add_argument("--make-split", metavar="OUT", default=None,
                    help="生成 test-only split(需配 --check-h5 指定源 H5)")
    ap.add_argument("--self-test", action="store_true", help="纯 stdlib 自检")
    ap.add_argument("--report", metavar="JSON", default=None,
                    help="把全部结果写成**一个** JSON(不产生零散文件)")
    ap.add_argument("--tier", choices=sorted(TIER_REGISTRY), default="resisc45",
                    help="泛化数据集(决定 --check-source 的 HF repo 与场景映射;默认 resisc45)")
    ap.add_argument(
        "--snapshot-root", default=None, metavar="PATH",
        help="WHU 可选:HF repo cache、具体 snapshot、data 目录或 train Parquet;"
             "未给时自动查 $HF_HOME/hub")
    ap.add_argument("--dryrun", action="store_true", help="只打印计划(等价 DRYRUN=1)")
    args = ap.parse_args()

    # --tier: 覆盖模块级数据源常量(源检查/AST 提取都据此走对应数据集)。
    RESISC45_REPO, SCENE_GROUPS_NAME, RESISC45_SRC_HW = TIER_REGISTRY[args.tier]

    dryrun = args.dryrun or os.environ.get("DRYRUN", "0") == "1"

    if not any([args.check_offline, args.check_source, args.check_h5,
                args.check_split, args.make_split, args.self_test]):
        ap.print_help()
        print("\nFATAL: 至少指定一个检查项", file=sys.stderr)
        return 2
    if args.make_split and not args.check_h5:
        print("FATAL: --make-split 需要配 --check-h5 H5 指定源 H5", file=sys.stderr)
        return 2
    if args.expect_n is not None and not args.check_h5:
        print("FATAL: --expect-n 需要配 --check-h5 H5", file=sys.stderr)
        return 2
    if args.expect_n is not None and args.expect_n <= 0:
        print("FATAL: --expect-n 必须大于 0", file=sys.stderr)
        return 2

    rep = Report()
    print("=" * 66)
    print(f" check_resisc45.py  ROOT={ROOT}" + ("  [DRYRUN]" if dryrun else ""))
    print("=" * 66)

    if args.self_test:
        self_test(rep)
    if args.check_offline:
        check_offline(rep, dryrun)
    if args.check_source:
        check_source(rep, dryrun, snapshot_root=args.snapshot_root)
    if args.check_h5:
        check_h5(rep, args.check_h5, dryrun, expected_n=args.expect_n)
    if args.make_split:
        make_split(rep, args.check_h5, args.make_split, dryrun)
    if args.check_split:
        check_split(rep, args.check_split[0], args.check_split[1], dryrun)

    return rep.finish(args.report)


if __name__ == "__main__":
    sys.exit(main())
