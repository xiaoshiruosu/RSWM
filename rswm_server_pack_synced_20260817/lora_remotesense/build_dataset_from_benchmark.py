#!/usr/bin/env python3
"""
从 remotesensebenchmark 构建 LingBot LoRA 训练数据集（完整高清版 v3）

修复 + 高清支持:
  [BUG-1] window_size=(480,480) 在 512×512 场景中无法有效滑动
          → 修正为小窗口，保证有足够空间滑动
  [BUG-2] Config 构造兼容性 (v1 output_path vs v5 output_dir)
  [BUG-3] benchmark_h5 文件路径查找不可靠
  [BUG-4] poses 几乎全是常数
  [BUG-5] 正方形窗口 resize 到宽屏导致拉伸变形
          → v3: 使用矩形窗口匹配目标宽高比，resize 时无填充无拉伸
  [BUG-6] prompt 方向用角度平均值不合理
  [HD]    底层 _load_scene_from_hf 强制 2×2 拼接
          → monkey-patch：大图直接裁剪，小图才拼接
  [BUG-7] letterbox 填充导致 GIF 两侧模糊
          → v3: 窗口宽高比 = 目标宽高比，直接 resize，零填充

数据源:
  Tier 1: blanchon/AID                    — 600×600, canonical 主集
  Tier 2: blanchon/RESISC45               — 256×256, 45 类
  Tier 3: blanchon/PatternNet             — 256×256
  Tier 4: jonathan-roberts1/WHU-RS19      — 1005 张,19 类,异构尺寸
  Tier 5: mixed / synthetic               — AID+RESISC45 / 无需下载

用法:
  # 合成数据快速测试
  python lora_remotesense/build_dataset_from_benchmark.py -o demo.h5 -n 20 --no-hf

  # 使用 AID 高清数据（推荐）
  python lora_remotesense/build_dataset_from_benchmark.py -o train.h5 -n 100 --dataset-tier aid

  # 自定义窗口尺寸
  python lora_remotesense/build_dataset_from_benchmark.py -o train.h5 -n 100 \
    --dataset-tier aid --window-height 128 --window-width 222
"""

import sys
import os

# Limit native CPU worker pools before NumPy / torch / CLIP are imported.  A
# large inherited OpenMP value can create hundreds of mostly idle threads and
# make SemanticGate retries appear hung.  Preserve an explicitly smaller
# operator value, but never allow this builder to exceed eight workers.
_NATIVE_THREAD_CAP = 8
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    try:
        _thread_count = int(os.environ.get(_thread_env, str(_NATIVE_THREAD_CAP)))
    except ValueError:
        _thread_count = _NATIVE_THREAD_CAP
    os.environ[_thread_env] = str(max(1, min(_thread_count, _NATIVE_THREAD_CAP)))

import argparse
import hashlib
import json
import time
import zlib
import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import h5py

# Plücker pose helpers (shared with lora_training/build_from_remotesense.py
# so pose and prompt conventions stay byte-consistent with verify_plucker_poses.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lora_training.camera_trajectory import make_poses_from_waypoints
from lora_remotesense.hf_image_utils import (
    WHURS19_LABEL_NAMES,
    WHURS19_REPO,
    decode_hf_image,
    find_hf_snapshot_parquet,
)

# ============================================================================
# remotesensebenchmark 模块路径
# ============================================================================

BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "wan" / "remotesensebenchmark"
BENCHMARK_SCRIPT = BENCHMARK_DIR / "build_multi_direction_benchmark_v5.py"
BENCHMARK_SCRIPT_FALLBACK = BENCHMARK_DIR / "build_multi_direction_benchmark.py"

# ============================================================================
# 高清遥感数据集配置
# ============================================================================

# ============================================================================
# [v2.5 统一分类学 2026-07-26] 场景映射唯一真源 = scene_taxonomy.json。
# flood 更名 water(新样本不再产出 flood;旧数据经 LEGACY_SCENE_ALIASES 归一)。
# 跨数据集统一:beach→water、forest→rural、pond/river/port/harbor→water、
# swimming_pool→urban、shipping_yard→urban、resort→general。
# 改映射 = 改 JSON 并升 taxonomy_version;本文件不再保存字面量组。
# ============================================================================
_TAXONOMY_FILE = Path(__file__).resolve().parent / "scene_taxonomy.json"
TAXONOMY = json.loads(_TAXONOMY_FILE.read_text(encoding="utf-8"))
TAXONOMY_SHA256 = hashlib.sha256(_TAXONOMY_FILE.read_bytes()).hexdigest()
SCENE_CATEGORIES = list(TAXONOMY["scene_categories"])
LEGACY_SCENE_ALIASES = dict(TAXONOMY.get("legacy_aliases", {}))
PROMPT_TEMPLATES = dict(TAXONOMY["prompt_templates"])

AID_SCENE_GROUPS = TAXONOMY["groups"]["blanchon/AID"]
RESISC45_SCENE_GROUPS = TAXONOMY["groups"]["blanchon/RESISC45"]
PATTERNNET_SCENE_GROUPS = TAXONOMY["groups"]["blanchon/PatternNet"]
WHURS19_SCENE_GROUPS = TAXONOMY["groups"]["jonathan-roberts1/WHU-RS19"]


def canonical_scene(name):
    """场景名归一:legacy 别名(flood→water)映射到现行 taxonomy;读旧数据用。"""
    s = str(name)
    return LEGACY_SCENE_ALIASES.get(s, s)


def normalize_class_name(name: str) -> str:
    """类名归一化:仅保留字母数字并小写 —— "BareLand"/"bare_land"/"Bare Land"
    统一为 "bareland"。全部数据集一律归一化后【精确】匹配(双向子串已废除,
    它正是 AID 四池=全集、进而四重完整重复的根因之一)。"""
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _assert_scene_groups_disjoint():
    """[v2.5 §四] 导入期硬校验:①每数据集四场景类别组互斥;②组键 = 现行
    scene_categories 且不含 legacy 名(flood);③新 taxonomy 不允许缺场景。"""
    cats = set(SCENE_CATEGORIES)
    assert "flood" not in cats, "scene_categories 不得含 legacy 名 flood"
    for ds_name, groups in TAXONOMY["groups"].items():
        if set(groups) != cats:
            raise RuntimeError(
                f"{ds_name}: 组键 {sorted(groups)} != scene_categories {sorted(cats)}")
        seen = {}
        for scene, classes in groups.items():
            for c in classes:
                nc = normalize_class_name(c)
                if nc in seen and seen[nc] != scene:
                    raise RuntimeError(
                        f"{ds_name}: 类别 {c!r} 同时归属 {seen[nc]!r} 与 {scene!r} "
                        f"—— 场景池必须互斥,修 scene_taxonomy.json。")
                seen[nc] = scene


_assert_scene_groups_disjoint()

# [v2.5] tier 配置按 SCENE_CATEGORIES 程序化生成 —— 每场景同一数据集;
# 键永远与现行 taxonomy 同步,不再手写(旧字面量曾在 flood→water 更名时漂移)。
def _tier_config(*dataset_entries):
    return {sc: {"datasets": [dict(d) for d in dataset_entries]}
            for sc in SCENE_CATEGORIES}


_AID_DS = {"name": "blanchon/AID", "split": "train", "feature": "image"}
RESISC45_REPO = "blanchon/RESISC45"
_RESISC_DS = {"name": RESISC45_REPO, "split": "train", "feature": "image"}
_PATTERNNET_DS = {"name": "blanchon/PatternNet", "split": "train", "feature": "image"}
_WHURS19_DS = {"name": "jonathan-roberts1/WHU-RS19", "split": "train", "feature": "image"}

HD_DATASET_CONFIG_AID = _tier_config(_AID_DS)            # 600² 直裁,canonical 主集
# [净化] resisc45 tier = 纯 RESISC45(曾无过滤混拌 PatternNet);PatternNet 独立 tier。
HD_DATASET_CONFIG_RESISC45 = _tier_config(_RESISC_DS)    # 256² 同类拼接
HD_DATASET_CONFIG_PATTERNNET = _tier_config(_PATTERNNET_DS)
HD_DATASET_CONFIG_WHURS19 = _tier_config(_WHURS19_DS)    # 异构尺寸,统一大图模式,零拼接
HD_DATASET_CONFIG_MIXED = _tier_config(_AID_DS, _RESISC_DS)

DATASET_TIERS = {
    "aid": HD_DATASET_CONFIG_AID,             # 600×600 直裁,canonical 主数据集
    "resisc45": HD_DATASET_CONFIG_RESISC45,   # 256×256 同类拼接;泛化评估 #1
    "patternnet": HD_DATASET_CONFIG_PATTERNNET,  # 256×256 同类拼接;泛化评估 #2
    "whurs19": HD_DATASET_CONFIG_WHURS19,     # 1005 张异构尺寸;大图归一化后直裁 #3
    "mixed": HD_DATASET_CONFIG_MIXED,
}


# ============================================================================
# [OFFLINE] 本地图像目录数据集适配器（替代 Hugging Face 下载）
# ============================================================================

class _LabelFeature:
    """Mock HF ClassLabel feature."""
    def __init__(self, names):
        self.names = list(names)


class LocalImageDataset:
    """Drop-in replacement for a HF dataset, backed by local image directory.

    Expected layout: root/{class_name}/*.jpg  (subdirectories = class labels)
    Fallback: root/*.jpg  (flat directory, single "unknown" label)
    """

    def __init__(self, root_dir):
        from PIL import Image as _PILImage
        self._PILImage = _PILImage
        self.root = Path(root_dir)
        self.samples = []       # [(path_str, label_idx), ...]
        self.label_names = []

        subdirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if subdirs:
            self.label_names = [d.name for d in subdirs]
            for label_idx, subdir in enumerate(subdirs):
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    for img_path in sorted(subdir.glob(ext)):
                        self.samples.append((str(img_path), label_idx))
        else:
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_path in sorted(self.root.glob(ext)):
                    self.samples.append((str(img_path), 0))
            self.label_names = ["unknown"]

        self.features = {"label": _LabelFeature(self.label_names)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, key):
        if isinstance(key, str):
            if key == "label":
                return [s[1] for s in self.samples]
            raise KeyError(key)
        path, label = self.samples[key]
        return {
            "image": decode_hf_image(
                {"bytes": None, "path": path}, context=f"local image {path}"),
            "label": label,
        }


# ============================================================================
# [BUG-1 + BUG-7 修复] 窗口默认值
# ============================================================================
# 窗口宽高比 = 目标宽高比 (480:832 ≈ 0.577)
# 窗口高 128 → 窗口宽 = round(128 * 832/480) = 222
# 这样 resize 时直接缩放，不需要 letterbox 填充

DEFAULT_SCENE_SIZE = 512
DEFAULT_WINDOW_HEIGHT = 128
DEFAULT_WINDOW_WIDTH = 222   # ← 匹配 480:832 宽高比，消除两侧模糊
DEFAULT_STEPS_PER_SEGMENT = 9


# ============================================================================
# [HD 核心] 按场景类别筛选图像索引
# ============================================================================

class SceneSelectionError(RuntimeError):
    """[v2.4] 场景源图选择失败(池空/映射缺/容量不足/路径解析失败)。
    必须让构建当场崩溃 —— 绝不允许被吞成 None 后静默退回合成场景或
    eurosat(静默回退正是 2026-07-23 四池=全集事故没被发现的原因)。"""


# [v2.4 §六] 构建期源图身份登记表:(scene_category, 块内序号) → source_* 字段。
# _hd_load_scene_from_hf 逐样本登记;build_benchmark 在 builder.build() 后把它
# 回写进 benchmark H5 的 per-sample attrs(转换器再透传到最终 H5)。
_HD_SOURCE_REGISTRY = {}

# [v2.4.1] _raw_image_paths 的按数据集对象缓存:{id(dataset): (len, paths)}。
_RAW_PATHS_CACHE = {}


def _dataset_scene_groups(dataset_name):
    """→ (场景组 dict|None, 是否已知数据集)。已知 = AID/RESISC45/PatternNet/WHU-RS19。"""
    name_l = (dataset_name or "").lower()
    if "aid" in name_l:
        return AID_SCENE_GROUPS, True
    if "resisc" in name_l:
        return RESISC45_SCENE_GROUPS, True
    if "patternnet" in name_l:
        return PATTERNNET_SCENE_GROUPS, True
    if "whu" in name_l or "rs19" in name_l:
        return WHURS19_SCENE_GROUPS, True
    return None, False


def _is_whurs19_name(dataset_name):
    """Exact official-source check used by WHU fail-close paths."""
    return str(dataset_name or "").strip().lower() == WHURS19_REPO.lower()


def _raw_image_paths(dataset, feature_name="image"):
    """全部记录的原始文件路径(不解码图像),长度=len(dataset),缺失为 None。

    blanchon/AID 没有 label 字段、只有 ["image"],真实类别只能从原始路径解析
    (.../data/Airport/airport_1.jpg)。不能用解码后的 PIL .filename(可能为
    None)。LocalImageDataset → samples 直取;HF → 先试 pyarrow 零拷贝取
    struct 的 path 子列(不碰 bytes),失败再退 cast_column(Image(decode=
    False)) 逐条(会读 bytes,慢但稳)。"""
    if hasattr(dataset, "samples"):          # LocalImageDataset
        return [s[0] for s in dataset.samples]
    # [v2.4.1] 记忆化:AID 建池会对同一 dataset 调 5 次(数据集级统计 + 4 场景),
    # 逐次全表扫描既慢又(慢路径下)反复搬 bytes。按对象缓存,长度校验防错配。
    cached = _RAW_PATHS_CACHE.get(id(dataset))
    if cached is not None and cached[0] == len(dataset):
        return cached[1]
    paths = None
    try:
        col = dataset.data.column(feature_name)
        # [v2.4.1] 逐 chunk 取 struct 的 path 子列 —— 绝不 combine_chunks():
        # 合并会把整个 struct 列(含全部图像 bytes,AID ~2.5GB)物化进内存,
        # 登录节点 cgroup 限额下有 OOM 风险;field() 按子列零拷贝抽取。
        chunks = col.chunks if hasattr(col, "chunks") else [col]
        acc = []
        for ch in chunks:
            acc.extend(ch.field("path").to_pylist())
        if len(acc) == len(dataset):
            paths = acc
    except Exception:
        paths = None
    if paths is None:
        from datasets import Image as _HFImage
        rd = dataset.cast_column(feature_name, _HFImage(decode=False))
        paths = []
        for i in range(len(rd)):
            rec = rd[i][feature_name]
            paths.append(rec.get("path") if isinstance(rec, dict)
                         else getattr(rec, "path", None))
    _RAW_PATHS_CACHE[id(dataset)] = (len(dataset), paths)
    return paths


def _class_from_data_path(path):
    """.../data/<CLASS>/<file>.jpg → CLASS;不合规返回 None。
    排除:thumbnail.png 等非 JPG/JPEG、不含 data/ 段、data 后无类目录、缺路径。
    (blanchon/AID 索引 10000 = 仓库 thumbnail.png,由 .png 后缀排除。)"""
    if not path:
        return None
    p = Path(str(path))
    if p.suffix.lower() not in (".jpg", ".jpeg"):
        return None
    parts = p.parts
    if "data" not in parts:
        return None
    dpos = len(parts) - 1 - parts[::-1].index("data")   # 最后一个 data 段
    if dpos + 1 >= len(parts) - 1:                       # data 后须为 <CLASS>/<file>
        return None
    return parts[dpos + 1]


def _scene_pool_with_meta(dataset, scene_category, dataset_name):
    """[v2.4 核心] → (indices, class_by_idx, path_by_idx)。

    已知数据集(AID/RESISC45/PatternNet/WHU-RS19):
      * 一律归一化【精确】匹配(双向子串废除);
      * 路线 A:有 label 字段 → 按标签名匹配;
      * 路线 B:无 label 字段(blanchon/AID)→ 未解码原始路径解析;
      * 两条路线都没匹配到 → SceneSelectionError,【禁止】静默回退全量。
    未知数据集:按"无类别数据集"处理,大声警告后返回全量(唯一允许的全量)。"""
    n = len(dataset)
    groups, known = _dataset_scene_groups(dataset_name)
    if not known:
        print(f"[filter] ⚠ 数据集 {dataset_name!r} 不在已知清单,按'无类别数据集'"
              f"处理:{scene_category!r} 返回全部 {n} 张。若它有类别请补映射。")
        return list(range(n)), {}, {}
    target_classes = groups.get(scene_category, [])
    if not target_classes:
        raise SceneSelectionError(
            f"{dataset_name}: 场景 {scene_category!r} 无类别映射 —— "
            f"已知数据集禁止无过滤回退(v2.4)。")
    norm_targets = {normalize_class_name(t) for t in target_classes}

    label_names = None
    try:
        if hasattr(dataset, "features") and "label" in dataset.features:
            lf = dataset.features["label"]
            if hasattr(lf, "names"):
                label_names = lf.names
    except Exception:
        label_names = None

    # Loading the WHU snapshot directly through the Parquet builder can expose
    # label as plain int64 instead of ClassLabel even though the repository card
    # publishes the stable 0..18 mapping.  Recover only that exact official
    # mapping, and only after proving every observed id is in range.  Other
    # datasets keep the previous behavior.
    if not label_names and _is_whurs19_name(dataset_name):
        try:
            observed_labels = [int(v) for v in dataset["label"]]
        except Exception as exc:
            raise SceneSelectionError(
                f"{WHURS19_REPO}: 无法读取 int64 label 列: {exc}") from exc
        invalid = sorted({v for v in observed_labels
                          if not 0 <= v < len(WHURS19_LABEL_NAMES)})
        if invalid:
            raise SceneSelectionError(
                f"{WHURS19_REPO}: label id 超出官方 0..18 映射: {invalid}")
        label_names = list(WHURS19_LABEL_NAMES)

    indices, class_by_idx, path_by_idx = [], {}, {}
    if label_names is not None:
        target_ids = {i for i, nm in enumerate(label_names)
                      if normalize_class_name(nm) in norm_targets}
        if target_ids:
            labels = dataset["label"]
            for i, lab in enumerate(labels):
                if lab in target_ids:
                    indices.append(i)
                    class_by_idx[i] = label_names[lab]
            if hasattr(dataset, "samples"):      # LocalImageDataset 顺带存路径
                for i in indices:
                    path_by_idx[i] = dataset.samples[i][0]

    if not indices:
        # 路线 B:原始路径解析(AID 无 label 字段;或 label 路线零命中时的复核)
        paths = _raw_image_paths(dataset)
        for i, p in enumerate(paths):
            cls = _class_from_data_path(p)
            if cls is None:
                continue          # thumbnail/非jpg/非 data//缺路径 → 排除
            if normalize_class_name(cls) in norm_targets:
                indices.append(i)
                class_by_idx[i] = cls
                path_by_idx[i] = p

    if not indices:
        raise SceneSelectionError(
            f"No valid indices for dataset={dataset_name}, scene={scene_category} "
            f"—— label 路线与原始路径路线均零命中;检查 *_SCENE_GROUPS 与数据集"
            f"布局。禁止静默回退全量(v2.4)。")
    indices = sorted(set(indices))
    return indices, class_by_idx, path_by_idx


def _get_scene_filter_indices(dataset, scene_category, dataset_name):
    """兼容旧签名的薄封装(外部工具 check_aid_pools 等使用)。"""
    return _scene_pool_with_meta(dataset, scene_category, dataset_name)[0]


# ============================================================================
# [HD 核心] monkey-patch: 替换底层 _load_scene_from_hf
# ============================================================================


def _decode_source_array(self, field, *, context):
    """Decode one source record and convert it to the builder's float array."""
    image = decode_hf_image(field, context=context)
    try:
        arr = self._to_numpy_image(image)
    finally:
        image.close()
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise SceneSelectionError(
            f"{context}: decoded RGB image has invalid shape {arr.shape}")
    return arr[:, :, :3]


def _normalize_whurs19_source(self, image, scene_hw):
    """Deterministically make a WHU source cover the canonical 512 scene.

    WHU-RS19 has two verified records whose height is below 512.  They follow
    the same large-image scene/crop/trajectory pipeline as every other WHU
    record: only an isotropic, aspect-preserving resize-to-cover is inserted
    before the existing random 512 crop.  No padding, deletion, stitching, or
    trajectory change is introduced.
    """
    H, W = (int(scene_hw[0]), int(scene_hw[1]))
    h, w = image.shape[:2]
    meta = {
        "source_original_hw": json.dumps([int(h), int(w)]),
        "source_preprocess": "identity",
        "source_preprocess_scale": 1.0,
    }
    if h >= H and w >= W:
        meta["source_normalized_hw"] = json.dumps([int(h), int(w)])
        return image, meta

    scale = max(H / float(h), W / float(w))
    new_h = max(H, int(np.ceil(h * scale)))
    new_w = max(W, int(np.ceil(w * scale)))
    normalized = self._resize_image(image, (new_h, new_w))
    if normalized.shape[0] < H or normalized.shape[1] < W:
        raise SceneSelectionError(
            f"WHU canonical normalization failed: {h}x{w} -> "
            f"{normalized.shape[:2]}, required >= {H}x{W}")
    meta.update({
        "source_normalized_hw": json.dumps(
            [int(normalized.shape[0]), int(normalized.shape[1])]),
        "source_preprocess": "isotropic_resize_to_cover_scene",
        "source_preprocess_scale": float(scale),
    })
    return normalized, meta


def _hd_load_scene_from_hf(self, scene_category, scene_id, H, W):
    """
    高清版场景加载（替换原始的 2×2 拼接逻辑）

    - 对于 ≥(H,W) 的大图（如 AID 600×600 → 512×512）：直接随机裁剪
    - 对于较小的图（如 RESISC45 256×256）：同场景类别拼接 + 随机增强
    """
    try:
        ds_info = self.hf_datasets[scene_category]
        dataset = ds_info["dataset"]
        feature_name = ds_info["feature"]

        n = len(dataset)
        if n == 0:
            raise SceneSelectionError(f"{scene_category}: 数据集为空")

        # [v2.4.1 修 V1-C1] 数据集名一律以【实际加载】的 info["name"] 为准:
        # mixed tier 下 v5 取"第一个加载成功的"数据集,而 _hd_ds_name_map 恒记
        # datasets[0] —— 若首选加载失败换了备胎,按 name map 过滤/登记会用错
        # 场景组且把 source_dataset 溯源写成没用过的数据集(静默伪造)。
        ds_name = (ds_info.get("name")
                   or getattr(self, '_hd_ds_name_map', {}).get(scene_category, ""))

        # ---- [v2.4] 首次调用即为全部场景急切建池 + 同数据集池间互斥硬检查 ----
        # (惰性建池会让互斥违规拖到第二个场景块才暴露,白烧第一块的构建时间。)
        if not hasattr(self, "_hd_pool_cache"):
            # [v2.4.1 修 V1-N1] 先建临时表、全部成功后一次性提交 —— 中途抛错
            # 不留半初始化缓存(否则重入时 hasattr 判真、_hd_perm_cache KeyError)。
            pool_cache, perm_cache, size_cache, actual_name = {}, {}, {}, {}
            name_map = getattr(self, "_hd_ds_name_map", {})
            for sc in getattr(self.config, "scene_categories", []):
                if sc not in getattr(self, "hf_datasets", {}):
                    continue
                info = self.hf_datasets[sc]
                actual_name[sc] = (info.get("name")
                                   or name_map.get(sc, ""))
                pool_cache[sc] = _scene_pool_with_meta(
                    info["dataset"], sc, actual_name[sc])
                probe_idx = int(pool_cache[sc][0][0])
                probe = _decode_source_array(
                    self, info["dataset"][probe_idx][info["feature"]],
                    context=f"{actual_name[sc]}[{probe_idx}].{info['feature']}")
                size_cache[sc] = probe.shape[:2]
            for a, b in combinations(list(pool_cache), 2):
                if actual_name.get(a, "") != actual_name.get(b, ""):
                    continue    # 不同数据集的场景池天然不相交
                ov = set(pool_cache[a][0]) & set(pool_cache[b][0])
                if ov:
                    raise SceneSelectionError(
                        f"Scene pools overlap: {a}/{b}, count={len(ov)} —— 场景池"
                        f"必须互斥(修 *_SCENE_GROUPS;2026-07-23 四重重复样本"
                        f"事故根因之一)。")
            for sc, (idx, _c, _p) in pool_cache.items():
                # 场景内确定性打乱:池按类别分组排列,顺序取会让前 N 张全来自
                # 同一类;打乱后类别混合,由 (random_seed, 场景名) 决定、可复现,
                # 与逐样本轨迹 RNG(np.random 全局流)相互独立。
                rng = np.random.default_rng(
                    [int(getattr(self.config, "random_seed", 42)),
                     zlib.crc32(sc.encode())])
                perm_cache[sc] = rng.permutation(np.asarray(idx, dtype=np.int64))
            self._hd_pool_cache = pool_cache
            self._hd_perm_cache = perm_cache
            self._hd_size_cache = size_cache
            self._hd_actual_name = actual_name
            for sc in pool_cache:
                print(f"  [pool] {sc}: {len(pool_cache[sc][0])} 张有效源图"
                      f"(数据集={actual_name.get(sc, '?')},互斥校验过,已确定性打乱)")

        if scene_category not in self._hd_pool_cache:
            raise SceneSelectionError(
                f"{scene_category}: 无场景池(hf_datasets 未配置该场景)")
        _pool_idx, class_map, path_map = self._hd_pool_cache[scene_category]
        order = self._hd_perm_cache[scene_category]
        img_h, img_w = self._hd_size_cache[scene_category]

        whurs19_mode = _is_whurs19_name(ds_name)
        if whurs19_mode or (img_h >= H and img_w >= W):
            # ========== 大图模式:直接随机裁剪(每样本恰用 1 张唯一源图) ==========
            # [v2.4 §五] 禁止 % 回绕 —— 回绕会让同一源图进入多个样本/跨 train-val。
            if scene_id >= len(order):
                raise SceneSelectionError(
                    f"Insufficient unique images: scene={scene_category}, "
                    f"requested={scene_id + 1}, available={len(order)} —— "
                    f"减少 instances_per_scene 或扩池,禁止回绕。")
            test_idx = int(order[scene_id])
            test_img = _decode_source_array(
                self, dataset[test_idx][feature_name],
                context=f"{ds_name}[{test_idx}].{feature_name}")
            source_preprocess = {}
            if whurs19_mode:
                test_img, source_preprocess = _normalize_whurs19_source(
                    self, test_img, (H, W))
            elif test_img.shape[0] < H or test_img.shape[1] < W:
                raise SceneSelectionError(
                    f"{ds_name}[{test_idx}] 尺寸 {test_img.shape[:2]} 小于场景 "
                    f"{(H, W)},但数据集首图为 {(img_h, img_w)};"
                    "非 WHU 异构尺寸没有定义 canonical normalization。")
            max_y = test_img.shape[0] - H
            max_x = test_img.shape[1] - W
            y0 = np.random.randint(0, max_y + 1)
            x0 = np.random.randint(0, max_x + 1)
            scene = test_img[y0:y0+H, x0:x0+W, :3].astype(np.float32)
            source_record = {
                "source_dataset": ds_name or ds_info.get("name", ""),
                "source_split": str(ds_info.get("split", "train")),
                "source_index": int(test_idx),
                "source_path": str(path_map.get(test_idx, "") or ""),
                "source_class": str(class_map.get(test_idx, "") or ""),
            }
            source_record.update(source_preprocess)
            _HD_SOURCE_REGISTRY[(scene_category, int(scene_id))] = source_record
        else:
            # ========== 小图模式:智能拼接(每样本用 tiles 张唯一源图) ==========
            num_tiles_h = int(np.ceil(H / img_h))
            num_tiles_w = int(np.ceil(W / img_w))
            tile_h = H // num_tiles_h
            tile_w = W // num_tiles_w
            tiles = num_tiles_h * num_tiles_w
            start = scene_id * tiles
            if start + tiles > len(order):
                raise SceneSelectionError(
                    f"Insufficient unique images (拼接): scene={scene_category}, "
                    f"requested={(scene_id + 1) * tiles}, available={len(order)} "
                    f"—— 减少 instances_per_scene 或扩池,禁止回绕。")
            tile_indices = [int(order[start + t]) for t in range(tiles)]

            scene = np.zeros((H, W, 3), dtype=np.float32)
            for th in range(num_tiles_h):
                for tw in range(num_tiles_w):
                    idx = tile_indices[th * num_tiles_w + tw]
                    example = dataset[int(idx)]
                    img = _decode_source_array(
                        self, example[feature_name],
                        context=f"{ds_name}[{idx}].{feature_name}")

                    if np.random.rand() > 0.5:
                        img = img[:, ::-1, :]
                    if np.random.rand() > 0.5:
                        img = img[::-1, :, :]
                    k = np.random.randint(0, 4)
                    if k > 0:
                        img = np.rot90(img, k=k, axes=(0, 1))

                    img_resized = self._resize_image(img, (tile_h, tile_w))

                    y0 = th * tile_h
                    x0 = tw * tile_w
                    actual_h = min(tile_h, H - y0)
                    actual_w = min(tile_w, W - x0)
                    scene[y0:y0+actual_h, x0:x0+actual_w] = img_resized[:actual_h, :actual_w]

            _HD_SOURCE_REGISTRY[(scene_category, int(scene_id))] = {
                "source_dataset": ds_name or ds_info.get("name", ""),
                "source_split": str(ds_info.get("split", "train")),
                "source_index": int(tile_indices[0]),   # 代表索引 = 首 tile
                "source_path": str(path_map.get(tile_indices[0], "") or ""),
                "source_class": "stitched",
                "source_tile_indices": json.dumps(tile_indices),
            }

        if scene.max() <= 1.0:
            scene = scene * 255.0
        scene = scene.clip(0, 255).astype(np.float32)
        return scene

    except SceneSelectionError:
        raise
    except Exception as e:
        print(f"  FATAL: HD scene load failed for {scene_category}: {e}")
        import traceback
        traceback.print_exc()
        # [v2.4] 不再返回 None —— None 会触发 v5 的合成场景/eurosat 静默回退,
        # 在 canonical 构建里混入假地物。一切失败都必须让构建崩溃。
        raise SceneSelectionError(
            f"HD scene load failed for {scene_category} (scene_id={scene_id}): {e}"
        ) from e


# ============================================================================
# 模块加载与注入
# ============================================================================

def _load_benchmark_module():
    """动态加载 remotesensebenchmark 模块（优先 v5，fallback v1）"""
    script = BENCHMARK_SCRIPT
    if not script.exists():
        # [v2.7.11/A3-N1] v1 构建器零 semantic_gate 引用 —— 注入的门会被整体
        # 无视,产出无 semgate attrs 的 H5(证据链断裂)。门开时禁止静默回退。
        if os.environ.get("RSWM_SEMGATE", "1") != "0":
            raise RuntimeError(
                f"v5 构建器脚本缺失({BENCHMARK_SCRIPT}),而语义门开启 —— "
                f"回退 v1 会静默无视语义门。修复 v5 脚本,或显式 RSWM_SEMGATE=0"
                f"(关门构建不得用于论文)。")
        script = BENCHMARK_SCRIPT_FALLBACK
    if not script.exists():
        raise FileNotFoundError(
            f"未找到 benchmark 脚本:\n"
            f"  尝试 1: {BENCHMARK_SCRIPT}\n"
            f"  尝试 2: {BENCHMARK_SCRIPT_FALLBACK}\n"
            f"请确认 remotesensebenchmark/ 目录位于: {BENCHMARK_DIR}"
        )
    print(f"加载 benchmark 模块: {script.name}")
    spec = importlib.util.spec_from_file_location("rs_benchmark", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _detect_config_version(mod):
    """检测 MultiDirectionConfig 使用 output_dir (v5) 还是 output_path (v1)"""
    import dataclasses
    fields = {f.name for f in dataclasses.fields(mod.MultiDirectionConfig)}
    if "output_dir" in fields:
        return "v5"
    elif "output_path" in fields:
        return "v1"
    else:
        return "unknown"


def _assert_official_whurs19_loaded(scene_generator, expected_scenes):
    """Fail closed unless every requested WHU scene uses the exact WHU repo."""
    _assert_official_tier_loaded(
        scene_generator,
        expected_scenes,
        tier="whurs19",
        expected_repo=WHURS19_REPO,
    )


def _assert_official_tier_loaded(
        scene_generator, expected_scenes, *, tier, expected_repo):
    """Reject missing or substituted sources for an official dataset tier.

    The vendored benchmark loader intentionally catches Hugging Face failures
    so synthetic scenes remain available for its standalone demo.  Official
    WHU-RS19 and RESISC45 builds must never inherit that fallback behavior.
    """
    loaded = getattr(scene_generator, "hf_datasets", {})
    missing = [sc for sc in expected_scenes if sc not in loaded]
    wrong = {
        sc: str(loaded[sc].get("name", ""))
        for sc in expected_scenes
        if (sc in loaded
            and str(loaded[sc].get("name", "")).strip().lower()
            != str(expected_repo).strip().lower())
    }
    if missing or wrong:
        details = []
        if missing:
            details.append(f"missing scenes={missing}")
        if wrong:
            details.append(f"unexpected sources={wrong}")
        fallback_detail = (
            "Refusing synthetic/RESISC45/PatternNet fallback."
            if tier == "whurs19"
            else "Refusing synthetic or alternate-dataset fallback."
        )
        raise SceneSelectionError(
            f"official tier={tier} requires {expected_repo}"
            + " for every scene; " + "; ".join(details)
            + ". " + fallback_detail)


def _assert_official_resisc45_loaded(scene_generator, expected_scenes):
    """Fail closed unless every requested RESISC45 scene uses RESISC45."""
    _assert_official_tier_loaded(
        scene_generator,
        expected_scenes,
        tier="resisc45",
        expected_repo=RESISC45_REPO,
    )


def _install_whurs19_snapshot_loader(mod):
    """Prefer the verified local WHU Parquet when it is present in HF_HOME."""
    if not hasattr(mod, "hf_load_dataset"):
        return None
    snapshot_root = os.environ.get("WHURS19_SNAPSHOT_ROOT") or None
    parquet = find_hf_snapshot_parquet(
        WHURS19_REPO,
        hf_home=os.environ.get("HF_HOME") or None,
        snapshot_root=snapshot_root,
    )
    if snapshot_root and parquet is None:
        raise SceneSelectionError(
            f"WHURS19_SNAPSHOT_ROOT={snapshot_root} 下找不到 data/train-*.parquet")
    if parquet is None:
        return None

    original = mod.hf_load_dataset

    def _load_dataset_strict_whu(path, *args, **kwargs):
        if not _is_whurs19_name(path):
            return original(path, *args, **kwargs)
        if args:
            raise SceneSelectionError(
                f"WHU local Parquet loader received unsupported positional args: {args}")
        call_kwargs = dict(kwargs)
        split = call_kwargs.pop("split", "train")
        call_kwargs.pop("trust_remote_code", None)
        return original(
            "parquet", data_files=str(parquet), split=split, **call_kwargs)

    mod.hf_load_dataset = _load_dataset_strict_whu
    print(f"[WHU] 使用本地 Parquet snapshot: {parquet}")
    return parquet


def _inject_hd_config(mod, dataset_tier="aid", local_aid_dir=None):
    """注入高清数据集配置 + monkey-patch 场景加载方法。"""
    if dataset_tier not in DATASET_TIERS:
        print(f"[HD] 未知 tier: {dataset_tier}，跳过注入")
        return

    hd_config = DATASET_TIERS[dataset_tier]

    if dataset_tier == "whurs19":
        _install_whurs19_snapshot_loader(mod)

    mod.HF_DATASET_CONFIG = hd_config
    print(f"\n[HD] 注入高清数据集配置 (tier={dataset_tier}):")
    for scene, cfg in hd_config.items():
        ds_names = [d["name"] for d in cfg["datasets"]]
        print(f"  {scene}: {', '.join(ds_names)}")

    if hasattr(mod, 'SceneGenerator'):
        _original_load = mod.SceneGenerator._load_scene_from_hf

        def _patched_load(self, scene_category, scene_id, H, W):
            # [v2.4] HD 加载失败一律抛 SceneSelectionError 中止构建 —— 不再回退
            # _original_load(eurosat/合成),canonical 数据里不许混入替代场景。
            return _hd_load_scene_from_hf(self, scene_category, scene_id, H, W)

        mod.SceneGenerator._load_scene_from_hf = _patched_load

        _orig_init = mod.SceneGenerator.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self._hd_ds_name_map = {}
            for sc in self.config.scene_categories:
                if sc in hd_config:
                    self._hd_ds_name_map[sc] = hd_config[sc]["datasets"][0]["name"]
            official_guard = {
                "whurs19": _assert_official_whurs19_loaded,
                "resisc45": _assert_official_resisc45_loaded,
            }.get(dataset_tier)
            if official_guard is not None:
                official_guard(
                    self, [sc for sc in self.config.scene_categories
                           if sc in hd_config])

        mod.SceneGenerator.__init__ = _patched_init

        print("[HD] 已替换场景加载方法（大图裁剪 + 小图智能拼接 + 场景类别筛选）")
    else:
        print("[HD] 警告: 未找到 SceneGenerator 类，无法注入高清加载方法")

    # ------------------------------------------------------------------
    # [OFFLINE] local_aid_dir: 用本地图片目录替代 HF 下载
    # ------------------------------------------------------------------
    if local_aid_dir is not None and hasattr(mod, 'SceneGenerator'):
        _local_ds_cache = {}

        def _patched_load_hf_datasets(self_sg):
            if 'ds' not in _local_ds_cache:
                _local_ds_cache['ds'] = LocalImageDataset(local_aid_dir)
            local_ds = _local_ds_cache['ds']
            print(f"\n[LOCAL] {len(local_ds)} images, "
                  f"{len(local_ds.label_names)} classes from {local_aid_dir}")
            for sc in self_sg.config.scene_categories:
                if sc not in hd_config:
                    continue
                ds_name = hd_config[sc]["datasets"][0]["name"]
                if "AID" not in ds_name.upper():
                    continue
                self_sg.hf_datasets[sc] = {
                    "dataset": local_ds,
                    "feature": hd_config[sc]["datasets"][0]["feature"],
                    "name": ds_name,
                }
                self_sg.hf_indices[sc] = 0
                print(f"  {sc}: local AID ({len(local_ds)} images)")
            print()

        mod.SceneGenerator._load_hf_datasets = _patched_load_hf_datasets

        _prev_init_for_offline = mod.SceneGenerator.__init__

        def _offline_init(self, *args, **kwargs):
            _prev_init_for_offline(self, *args, **kwargs)
            if not self.use_hf:
                self.use_hf = True
                self._load_hf_datasets()

        mod.SceneGenerator.__init__ = _offline_init
        print(f"[LOCAL] Will load AID from: {local_aid_dir}")


# ============================================================================
# [BUG-5 + BUG-7 修复] 智能 resize（自动选择策略）
# ============================================================================

def _resize_frame(frame, target_hw):
    """
    Resize 帧到目标尺寸。

    策略选择:
    1. 如果源和目标宽高比接近（误差<5%）→ 直接 resize（无填充无拉伸）
    2. 如果宽高比差异大 → letterbox（边缘像素填充）

    [BUG-7 修复] 当窗口宽高比匹配目标时（v3 默认），走策略 1，
    两侧不再有模糊填充。
    """
    H_tgt, W_tgt = target_hw
    h_src, w_src = frame.shape[:2]

    src_ratio = w_src / h_src
    tgt_ratio = W_tgt / H_tgt

    try:
        import cv2

        # 宽高比接近：直接 resize，效果最好
        if abs(src_ratio - tgt_ratio) / tgt_ratio < 0.05:
            return cv2.resize(frame, (W_tgt, H_tgt), interpolation=cv2.INTER_LINEAR)

        # 宽高比差异大：letterbox
        scale = H_tgt / h_src
        new_w = int(w_src * scale)

        if new_w >= W_tgt:
            # 源图偏宽，按宽度缩放后裁剪高度
            scale_w = W_tgt / w_src
            new_h = int(h_src * scale_w)
            resized = cv2.resize(frame, (W_tgt, new_h), interpolation=cv2.INTER_LINEAR)
            # 中心裁剪
            if new_h > H_tgt:
                crop_top = (new_h - H_tgt) // 2
                resized = resized[crop_top:crop_top + H_tgt, :]
            elif new_h < H_tgt:
                pad_top = (H_tgt - new_h) // 2
                pad_bot = H_tgt - new_h - pad_top
                resized = cv2.copyMakeBorder(
                    resized, pad_top, pad_bot, 0, 0, cv2.BORDER_REPLICATE)
            return resized
        else:
            # 源图偏高，按高度缩放后填充宽度
            resized = cv2.resize(frame, (new_w, H_tgt), interpolation=cv2.INTER_LINEAR)
            pad_left = (W_tgt - new_w) // 2
            pad_right = W_tgt - new_w - pad_left
            result = cv2.copyMakeBorder(
                resized, 0, 0, pad_left, pad_right, cv2.BORDER_REPLICATE)
            return result
    except ImportError:
        # numpy fallback
        h_idx = np.linspace(0, h_src - 1, H_tgt).astype(int)
        w_idx = np.linspace(0, w_src - 1, W_tgt).astype(int)
        return frame[np.ix_(h_idx, w_idx)]


# ============================================================================
# [BUG-4 修复] 从 waypoints 生成有意义的 poses
# ============================================================================

def _make_poses_from_trajectory(trajectory, window_size=(96, 96)):
    """[v2.8 P0] 直接用【真实逐帧窗口中心】(cy,cx) 生成 c2w 位姿。

    旧路径 _make_poses_from_waypoints 走 interpolate_pixel_positions:把
    K+1 个 waypoint 按弧长【均匀】插值到 T 帧 —— 那是与视频独立的第二条时间轴。
    v5 的真实轨迹每段 9 步且段界含重复帧,两条轴的转弯时刻因此不对齐:pose 说
    "此刻转弯"时视频还没转(或反之)。既然 v5 已把逐帧窗口中心存进
    benchmark H5 的 trajectory,pose 就该由它派生 —— 视频与 pose 严格同源,
    不依赖"段等长"这类恰好成立的巧合。
    """
    from lora_training.camera_trajectory import pixel_to_world_translation
    pix = np.asarray(trajectory, dtype=np.float64)
    trans = pixel_to_world_translation(pix, window_size)
    # 与 make_poses_from_waypoints 同款 c2w:旋转恒为 I(天底视角固定朝向),
    # 只有平移列编码滑窗运动
    poses = np.tile(np.eye(4, dtype=np.float32), (len(trans), 1, 1))
    poses[:, :3, 3] = trans
    return poses.astype(np.float32)


def _make_poses_from_waypoints(num_frames, waypoints, scene_size=(512, 512),
                               window_size=(96, 96)):
    """Plücker-ready c2w 位姿 (T, 4, 4) —— 从 waypoints 生成方向化平移轨迹。

    旧实现用 full-scene 归一化平移 + 每帧 yaw 旋转，yaw 会把 heading 和 in-plane
    图像旋转绞在一起，是 pilot 前角度误差 73-91° 的疑因之一。此处委托到
    lora_training.camera_trajectory.make_poses_from_waypoints：旋转恒为 I、
    平移以单窗口 = 1 个世界单位缩放，与 verify_plucker_poses.py 的约定一致。

    waypoints 缺失或退化时回退到纯恒等位姿。
    """
    if waypoints is None or len(waypoints) < 2:
        return np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    return make_poses_from_waypoints(
        num_frames=num_frames,
        waypoints=waypoints,
        scene_size=scene_size,
        window_size=window_size,
    ).astype(np.float32)


def _make_dummy_intrinsics(num_frames, width, height):
    """生成 dummy 内参 (T, 4): [fx, fy, cx, cy]"""
    intr = np.array([500.0, 500.0, width / 2.0, height / 2.0], dtype=np.float32)
    return np.tile(intr, (num_frames, 1))


# ============================================================================
# [BUG-6 修复] 改进的 prompt 生成
# ============================================================================

# [RSRoll-Bench P0#4] 指令通道 schema。默认保持旧行为 net，
# 换成 net_continuous 才是本文主通路。用环境变量而非新增 CLI 参数：
# 与 pack 里 RSWM_TRAJ_MAX_TRIES / LADDER_POSE_MODE 等既有开关同一种风格，
# 且能被 common.sh 一行透传，不必改动六个调用脚本的参数表。
PROMPT_SCHEMA = os.environ.get("RSWM_PROMPT_SCHEMA", "net")
_VALID_PROMPT_SCHEMAS = ("net", "net_continuous", "per_segment_continuous", "stripped")
if PROMPT_SCHEMA not in _VALID_PROMPT_SCHEMAS:
    raise SystemExit(
        "RSWM_PROMPT_SCHEMA=%r 非法，只支持 %s（fail-loud：拼错了不该静默退回 net，"
        "否则会建出一个自以为是连续式、实则 4 词式的库）"
        % (PROMPT_SCHEMA, list(_VALID_PROMPT_SCHEMAS)))


def _make_prompt(scene_category, num_segments, segment_angles, waypoints=None,
                 include_direction=True, prompt_schema="net"):
    """[v2.5 pt_v3] 场景模板 + 方向独立成句;方向语义绝不混入场景描述。

    场景句用 taxonomy 固定模板(灾害语义已废:旧版把普通水景写成
    "a flooded area",给文本通道灌入错误语义 —— 对专究文本方向信号的研究
    是混淆变量)。方向句保留 "net displacement toward the {DIR}" 措辞,
    与 parse_direction_word_from_prompt 兼容。
    include_direction=False 供 geo_direction_stripped 归因实验派生。

    [RSRoll-Bench P0#4] prompt_schema 决定方向句怎么写：
      net                     4 词式（**旧行为，逐位保持不变**，RSWISE 兼容臂/量化地板对照）
      net_continuous          连续净方位角 "toward a bearing of NNN.N degrees"  ← 主通路
      per_segment_continuous  逐段连续方位角
      stripped                删除一切方向信息（等价于 include_direction=False）
    为什么必须做：轨迹角本来是 uniform(0,360) 连续采样的，但旧实现在写进 prompt 前
    量化成 4 个词，评测端用同样的分箱做靶 —— 于是"连续方向"这句话**没有可测目标**，
    并自带 22.5° 的量化地板。生成端与评测端必须**同一个 commit** 一起换，
    只换一端会得到一张看着正常、实则错 45° 的表。
    """
    scene = canonical_scene(scene_category)
    template = PROMPT_TEMPLATES.get(scene, "a remote-sensing scene")
    base = f"A satellite video of {template}."

    if not include_direction:
        prompt_schema = "stripped"
    if prompt_schema != "net":
        # 非 legacy 分支统一交给单一真源实现（自检里与本函数 legacy 分支逐位比对过）
        _new_dir = str(Path(__file__).resolve().parent / "new")
        if _new_dir not in sys.path:
            sys.path.insert(0, _new_dir)
        from continuous_direction import make_prompt as _cd_make_prompt
        return _cd_make_prompt(f"A satellite video of {template}", num_segments,
                               waypoints=waypoints, segment_angles=segment_angles,
                               schema=prompt_schema)

    DIR = None
    if waypoints is not None and len(waypoints) >= 2:
        wp = np.array(waypoints, dtype=np.float32)
        dy = wp[-1, 0] - wp[0, 0]
        dx = wp[-1, 1] - wp[0, 1]
        angle = np.degrees(np.arctan2(-dy, dx)) % 360
    elif segment_angles is not None and len(segment_angles) > 0:
        angle = float(segment_angles[0]) % 360
    else:
        angle = None
    if angle is not None:
        if angle < 45 or angle >= 315:
            DIR = "east"
        elif 45 <= angle < 135:
            DIR = "north"
        elif 135 <= angle < 225:
            DIR = "west"
        else:
            DIR = "south"

    if include_direction and DIR is not None:
        # 注: K 段折线有 K-1 个转向;{num_segments} 是段数(K)而非转向数。
        return (f"{base} The camera follows a {num_segments}-segment "
                f"trajectory with net displacement toward the {DIR}.")
    if DIR is not None:
        # direction-stripped:保留段数信息,删除一切方向词(归因实验用)。
        return (f"{base} The camera follows a {num_segments}-segment "
                f"trajectory.")
    return base


# ============================================================================
# [BUG-2 & BUG-3 修复] 健壮的 benchmark 构建
# ============================================================================

def build_benchmark(mod, args, use_hf):
    """构建 benchmark，自动适配 v1/v5，支持矩形窗口。"""
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)

    instances_per_scene = max(1, args.instances // 4)
    version = _detect_config_version(mod)
    print(f"Config 版本: {version}")

    win_h = args.window_height
    win_w = args.window_width
    scene_size = args.scene_size

    # 验证窗口不会太大
    max_win_h = int(scene_size * 0.5)
    max_win_w = int(scene_size * 0.8)  # 宽度可以稍大，因为场景是正方形
    if win_h > max_win_h:
        print(f"  ⚠ window_height={win_h} 超过场景的 50%，自动缩小到 {max_win_h}")
        win_h = max_win_h
    if win_w > max_win_w:
        print(f"  ⚠ window_width={win_w} 超过场景的 80%，自动缩小到 {max_win_w}")
        win_w = max_win_w

    margin_h = win_h // 2
    margin_w = win_w // 2
    active_h = scene_size - 2 * margin_h
    active_w = scene_size - 2 * margin_w
    print(f"  场景: {scene_size}×{scene_size}")
    print(f"  窗口: {win_h}×{win_w} (宽高比 {win_w/win_h:.3f})")
    print(f"  目标: {args.target_height}×{args.target_width} (宽高比 {args.target_width/args.target_height:.3f})")
    print(f"  有效活动范围: {active_h}(纵) × {active_w}(横) 像素")
    print(f"  每步约移动: {min(active_h, active_w) / (args.steps * 2):.1f} 像素")

    # 检查宽高比匹配
    win_ratio = win_w / win_h
    tgt_ratio = args.target_width / args.target_height
    ratio_diff = abs(win_ratio - tgt_ratio) / tgt_ratio * 100
    if ratio_diff < 5:
        print(f"  ✓ 宽高比匹配 (差异 {ratio_diff:.1f}%)，resize 无填充无拉伸")
    else:
        print(f"  ⚠ 宽高比不匹配 (差异 {ratio_diff:.1f}%)，resize 时会有 letterbox 填充")

    if active_h < win_h * 0.3 or active_w < win_w * 0.3:
        print(f"  ⚠ 警告: 活动范围过小，帧间差异可能不足！")

    benchmark_h5 = Path(args.temp_dir) / "benchmark.h5"

    # [v2.8.1] 段数范围在分支【之外】解析 —— v5 与 v1 共用。
    # (教训:上一版只改了 v1 分支里的赋值,而实跑一直走 v5 分支的硬编码
    #  min_segments=2/max_segments=4,--min-segments 1 完全无效且不报错,
    #  34 号探针照样产出 K=2..4 的数据。断言"字符串在文件里"证明不了它在活路径上。)
    _ov_lo = getattr(args, "min_overlap_ratio", None)
    _ov_hi = getattr(args, "max_overlap_ratio", None)
    _ov_lo = 0.0 if _ov_lo is None else float(_ov_lo)
    _ov_hi = 0.10 if _ov_hi is None else float(_ov_hi)
    if not (0.0 <= _ov_lo < _ov_hi <= 1.0):
        raise ValueError(f"overlap 区间非法: [{_ov_lo}, {_ov_hi})")
    print(f"  overlap 区间: [{_ov_lo}, {_ov_hi})"
          + ("  ← 主 benchmark 硬约束" if (_ov_lo, _ov_hi) == (0.0, 0.10)
             else "  ← 非默认,产物不得进主表"))
    _aq = int(getattr(args, "angle_quantize", 0) or 0)
    if _aq not in (0, 4):
        raise ValueError(f"--angle-quantize 只支持 0(连续)或 4(四基本方向),得到 {_aq}")
    _min_seg = int(getattr(args, "min_segments", None) or 2)
    _max_seg = int(getattr(args, "max_segments", None) or 4)
    if _min_seg < 1 or _min_seg > _max_seg:
        raise ValueError(f"段数范围非法: --min-segments {_min_seg} / "
                         f"--max-segments {_max_seg}")
    # 4n+1 契约:去重后 T = steps + (K-1)(steps-1) 必须 ≡ 1 (mod 4),
    # 否则训练侧 _collate_fn 会直接拒收(v2.8.1 起非 4n+1 硬错,不再静默截尾)
    _bad = [(_K, args.steps + (_K - 1) * (args.steps - 1))
            for _K in range(_min_seg, _max_seg + 1)
            if (args.steps + (_K - 1) * (args.steps - 1) - 1) % 4 != 0]
    if _bad:
        raise ValueError(
            f"steps={args.steps} 下 K={[k for k, _ in _bad]} 的帧数 "
            f"{[t for _, t in _bad]} 不满足 4n+1,训练侧会拒收整个数据集。"
            f"canonical 用 --steps 9(K=1..4 → T=9/17/25/33)。")
    print(f"  角度量化: {'四基本方向 E/N/W/S' if _aq == 4 else '连续 (canonical)'}")
    print(f"  段数范围: K ∈ [{_min_seg}, {_max_seg}]  steps={args.steps} → "
          f"T ∈ {[args.steps + (k - 1) * (args.steps - 1) for k in range(_min_seg, _max_seg + 1)]}")

    if version == "v5":
        config = mod.MultiDirectionConfig(
            output_dir=str(Path(args.temp_dir)),
            instances_per_scene=instances_per_scene,
            scene_categories=list(SCENE_CATEGORIES),   # [v2.5] taxonomy 单一来源
            full_scene_size=(scene_size, scene_size),
            window_size=(win_h, win_w),
            steps_per_segment=args.steps,
            min_segments=_min_seg,
            max_segments=_max_seg,
            angle_quantize=_aq,
            min_overlap_ratio=_ov_lo,
            max_overlap_ratio=_ov_hi,
            random_seed=42,
        )
    elif version == "v1":
        config = mod.MultiDirectionConfig()
        config.output_path = str(benchmark_h5)
        config.instances_per_scene = instances_per_scene
        config.scene_categories = list(SCENE_CATEGORIES)  # [v2.5]
        config.full_scene_size = (scene_size, scene_size)
        config.window_size = (win_h, win_w)
        config.steps_per_segment = args.steps
        config.min_segments = _min_seg
        config.max_segments = _max_seg
        config.angle_quantize = _aq
        config.min_overlap_ratio = _ov_lo
        config.max_overlap_ratio = _ov_hi
        config.random_seed = 42
    else:
        raise RuntimeError(f"无法识别 Config 版本: {version}")

    _HD_SOURCE_REGISTRY.clear()   # [v2.4] 每次构建从空登记表开始
    builder = mod.MultiDirectionBenchmarkBuilder(config, use_hf=use_hf)
    # [v2.7 语义门] P0 人检失败的修复:裁剪后逐帧场景语义门(拒绝采样)。
    # 只在真实数据(use_hf)时注入;合成分支(--no-hf)按类别程序生成,天然匹配。
    # eager 加载:缺 CLIP 权重要在建库【开工前】炸,不许烧完几小时数据才发现。
    from lora_remotesense import semantic_gate as _sg
    if use_hf and _sg.gate_enabled():
        _gate = _sg.SemanticGate()
        _gate.ensure_ready()
        builder.semantic_gate = _gate
        builder.semantic_gate_max_tries = _sg.max_tries()
        print(f"  [semgate] 语义保持门已启用: {_sg.SEMGATE_VERSION} "
              f"model={_gate.model_name} thresholds={_gate.thresholds} "
              f"max_tries={builder.semantic_gate_max_tries}")
    elif use_hf:
        print("  [semgate] ⚠️⚠️ RSWM_SEMGATE=0 —— 语义保持门被显式关闭。"
              "P0 人检失败(water 样本 10/12 帧纯草地)正是缺这道门;"
              "关门构建的数据集不得用于论文数字。")
    stats = builder.build()

    if not benchmark_h5.exists():
        candidates = sorted(Path(args.temp_dir).glob("*.h5"))
        if candidates:
            benchmark_h5 = candidates[-1]
            print(f"  找到输出文件: {benchmark_h5}")
        else:
            for p in sorted(Path(".").glob("*benchmark*.h5")):
                benchmark_h5 = p
                break

    if not benchmark_h5.exists():
        raise FileNotFoundError(
            f"Benchmark 生成失败，未找到 .h5 文件。\n"
            f"检查目录: {args.temp_dir} 和当前目录"
        )

    print(f"Benchmark 输出: {benchmark_h5}")
    # [v2.7.13 受控 P0 验收埋点] 全部尝试(含被拒)的 scene|sid→class 落盘 ——
    # 逐 source_class 的尝试/拒绝/接受统计由 semgate_build_report.py 交叉最终
    # H5 的 source_scene_id 得出(拒绝尝试的类只存在于 registry)。
    try:
        _att = {f"{sc}|{sid}": {"class": rec.get("source_class"),
                                "index": rec.get("source_index")}
                for (sc, sid), rec in _HD_SOURCE_REGISTRY.items()}
        _att_path = Path(str(benchmark_h5)).parent / "semgate_attempts.json"
        _att_path.write_text(json.dumps(
            {"n_attempts": len(_att), "attempts": _att}, ensure_ascii=False))
        print(f"  [semgate] 尝试登记 {len(_att)} 条 → {_att_path}")
    except Exception as _e:      # 埋点失败不阻塞建库,但要大声
        print(f"  [semgate][WARN] attempts 落盘失败: {_e}")
    _annotate_source_identity(str(benchmark_h5), expect_registry=use_hf,
                              random_seed=int(getattr(config, "random_seed", 42)))
    return str(benchmark_h5), stats


def _annotate_source_identity(benchmark_h5, expect_registry=False, random_seed=42):
    """[v2.4 §六] 把源图身份回写进 benchmark H5。

    per-sample attrs: source_dataset/source_split/source_index/source_path/
    source_class(拼接样本另有 source_tile_indices);metadata attrs:
    builder_version/builder_script_sha256/converter_script_sha256/random_seed。
    登记表键=(scene_category, 块内序号);块内序号按样本在各自场景内的出现序
    重建(块连续或交错都成立)。HF 构建下有登记却有样本对不上 → 硬错误
    (说明有样本绕过了 HD 加载器 —— 合成/回退泄漏)。"""
    sha = {}
    for tag, p in (("builder_script_sha256",
                    BENCHMARK_SCRIPT if BENCHMARK_SCRIPT.exists()
                    else BENCHMARK_SCRIPT_FALLBACK),
                   ("converter_script_sha256", Path(__file__))):
        try:
            sha[tag] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        except Exception:
            sha[tag] = ""
    with h5py.File(benchmark_h5, "a") as f:
        meta = f["metadata"] if "metadata" in f else f.create_group("metadata")
        # [v2.7] 版本串由真源派生,不再手写(曾漂移:串里 taxonomy_v2.5,
        # 实际 taxonomy_version=water_v1_20260726)
        meta.attrs["builder_version"] = (
            "v5_dedup_fix_20260723+bounded_retry_best_candidate_20260821+"
            + str(TAXONOMY["taxonomy_version"])
            + ("+" + __import__("lora_remotesense.semantic_gate",
                                fromlist=["SEMGATE_VERSION"]).SEMGATE_VERSION
               if os.environ.get("RSWM_SEMGATE", "1") != "0" else "+semgate_disabled"))
        # [v2.7.11/A3-C2② 修] 门开关显式落盘 —— 值必须反映【真实注入态】:
        # env 开 且 use_hf(=expect_registry)才算 true。只看 env 会让 --no-hf
        # 合成建库被"开门却无证据"守卫误杀;真实注入态下若 v5 脚本丢失回退 v1
        # (门被整体无视),该 true + 无 semgate_frac 恰好触发 checker exit 8。
        meta.attrs["semgate_enabled"] = bool(
            expect_registry and os.environ.get("RSWM_SEMGATE", "1") != "0")
        meta.attrs["taxonomy_version"] = TAXONOMY["taxonomy_version"]
        meta.attrs["taxonomy_file_sha256"] = TAXONOMY_SHA256
        meta.attrs["prompt_template_version"] = TAXONOMY["prompt_template_version"]
        meta.attrs["builder_script_sha256"] = sha["builder_script_sha256"]
        meta.attrs["converter_script_sha256"] = sha["converter_script_sha256"]
        meta.attrs["random_seed"] = int(random_seed)
        if "data" not in f:
            print("  [source-id] 无 data 组,跳过 per-sample 标注")
            return
        if not _HD_SOURCE_REGISTRY:
            # [v2.4.1 修 V1-C2] use_hf 构建下登记表为空 = 所有样本都绕过了 HD
            # 加载器(HF 全部加载失败 → v5 静默走合成场景;或 LOCAL_AID_DIR 残留
            # 使非 AID tier 一个场景都接不上)。这是"100% 假地物却零报错"的
            # 灾难路径,必须硬错;只有显式 --no-hf(故意合成)才允许空登记表。
            if expect_registry:
                raise SceneSelectionError(
                    "[source-id] use_hf=True 但源图登记表为空 —— 全部样本绕过了 "
                    "HD 加载器(HF 加载全挂→合成回退,或 LOCAL_AID_DIR 与 tier "
                    "不匹配)。构建判废;查 [pool]/加载日志。")
            print("  [source-id] 登记表为空(--no-hf 合成构建),跳过 per-sample 标注")
            return
        per_scene_counter = {}
        n_hit, miss = 0, []
        for key in sorted(f["data"].keys()):
            g = f["data"][key]
            sc = str(g.attrs.get("scene_category", ""))
            i = per_scene_counter.get(sc, 0)
            per_scene_counter[sc] = i + 1
            # [v2.7] 语义门重采后,样本实际消耗的池索引 ≠ 块内序号 —— 优先用
            # 样本自己落盘的 source_scene_id 对账,否则源身份会错配到被拒
            # 尝试的源图(溯源/泄漏检查全跟着错)。旧构建无该 attr 时退回顺序
            # 计数(与 v2.6 行为逐字节一致)。
            if "source_scene_id" in g.attrs:
                i = int(g.attrs["source_scene_id"])
            rec = _HD_SOURCE_REGISTRY.get((sc, i))
            if rec is None:
                miss.append(key)
                continue
            for k, v in rec.items():
                g.attrs[k] = v
            n_hit += 1
        print(f"  [source-id] 源图身份 attrs 写入 {n_hit} 个样本")
        if miss:
            raise SceneSelectionError(
                f"[source-id] {len(miss)} 个样本没有源图登记(如 {miss[:5]})——"
                f"它们绕过了 HD 加载器(合成场景/回退泄漏?),构建判废。")


# ============================================================================
# HDF5 转换
# ============================================================================

def convert_benchmark_to_lingbot(src_h5_path, dst_h5_path,
                                  target_size=(480, 832), max_samples=None):
    """将 remotesensebenchmark HDF5 转换为 LingBot HDF5 格式"""
    _pose_sources_seen = set()   # [v2.8.1] 文件级 pose_injection 的真源
    print(f"\n转换: {src_h5_path} → {dst_h5_path}")
    print(f"目标分辨率: {target_size[0]}×{target_size[1]}")

    H, W = target_size
    t0 = time.time()

    Path(dst_h5_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_h5_path, 'r') as src, h5py.File(dst_h5_path, 'w') as dst:
        total = int(src['metadata'].attrs['total_instances'])
        if max_samples:
            total = min(total, max_samples)

        scene_size = tuple(src['metadata'].attrs.get('full_scene_size', (512, 512)))

        # Benchmark v5 writes window_size to metadata; older versions may not.
        # Plücker translation scale is 1 full-window shift ≈ 1 world unit, so
        # we need the (win_h, win_w) that was actually used at sampling time.
        win_raw = src['metadata'].attrs.get('window_size', (96, 96))
        try:
            win_h, win_w = int(win_raw[0]), int(win_raw[1])
        except (TypeError, IndexError):
            win_h = win_w = int(win_raw)
        window_size = (win_h, win_w)

        meta = dst.create_group('metadata')
        meta.attrs['total_instances'] = total
        meta.attrs['target_size'] = target_size
        meta.attrs['max_frames'] = 0
        meta.attrs['build_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        meta.attrs['source'] = 'remotesensebenchmark_hd_v3_plucker'
        # [v2.8.1] pose_injection 移到样本循环【之后】才写 —— 见文件末尾。
        # (原先在这里恒写 plucker_from_waypoints,而 v2.8 起逐样本实际是
        #  plucker_from_trajectory,文件级与样本级自相矛盾;第一版"修复"把
        #  推导放在这里,但此处 _pose_sources_seen 还是空集,等于没改。)
        meta.attrs['pose_window_size'] = (win_h, win_w)
        # [v2.4 §六] 构建溯源元数据透传(源 benchmark 由 _annotate_source_identity 写)
        for _a in ('builder_version', 'builder_script_sha256',
                   'converter_script_sha256', 'random_seed',
                   'taxonomy_version', 'taxonomy_file_sha256',
                   'prompt_template_version', 'semgate_enabled'):
            if _a in src['metadata'].attrs:
                meta.attrs[_a] = src['metadata'].attrs[_a]

        data = dst.create_group('data')
        max_f = 0
        written = 0

        skipped = []
        missing_trajectory = 0   # 旧版源无 'trajectory' 的样本数(仅计数,不中断)
        missing_full_scene = 0   # 旧版源无 'full_scene' 的样本数(仅计数,不中断)
        for idx in range(total):
            key = f'sample_{idx:05d}'
            if key not in src['data']:
                continue
            try:
                s = src['data'][key]

                frames = s['sliding_frames'][:]  # (T, h, w, 3)
                T = frames.shape[0]
                max_f = max(max_f, T)

                # [BUG-5 + BUG-7] 智能 resize
                image = _resize_frame(frames[0], target_size)
                vf = np.stack([_resize_frame(frames[i], target_size) for i in range(T)])

                scene_cat = canonical_scene(s.attrs['scene_category'])  # [v2.5] flood→water 归一
                n_seg = int(s.attrs['num_segments'])
                seg_angles = s['segment_angles'][:]
                waypoints = s['waypoints'][:] if 'waypoints' in s else None

                # Track B v2 校准修复所需的两个源字段(旧版源可能缺 -> 跳过并计数):
                #   trajectory (T,2)=(cy,cx) 真实逐帧窗口中心;full_scene (512,512,3)。
                trajectory = s['trajectory'][:] if 'trajectory' in s else None
                full_scene = s['full_scene'][:] if 'full_scene' in s else None

                prompt = _make_prompt(scene_cat, n_seg, seg_angles, waypoints,
                                      prompt_schema=PROMPT_SCHEMA)
                intrinsics = _make_dummy_intrinsics(T, W, H)
                # [v2.8 P0] 优先用真实逐帧 trajectory(与视频同一时间轴);
                # 仅旧源缺该字段时回退 waypoints 均匀插值(会有转弯时刻错位,
                # pose_source 如实标注供下游判读)
                if trajectory is not None and len(trajectory) == T:
                    poses = _make_poses_from_trajectory(trajectory, window_size)
                    _pose_from_traj = True
                else:
                    poses = _make_poses_from_waypoints(T, waypoints, scene_size, window_size)
                    _pose_from_traj = False
                pose_source = (
                    ("plucker_from_trajectory" if _pose_from_traj
                     else "plucker_from_waypoints")
                    if (waypoints is not None and len(waypoints) >= 2)
                    else "dummy_identity"
                )
            except (KeyError, OSError) as e:
                print(f"  ⚠ 跳过损坏的 {key}: {e}")
                skipped.append(key)
                continue

            g = data.create_group(key)
            g.create_dataset('image', data=image, compression='gzip')
            g.create_dataset('video_frames', data=vf, compression='gzip')
            g.create_dataset('intrinsics', data=intrinsics)
            g.create_dataset('poses', data=poses)
            g.attrs['prompt'] = prompt
            g.attrs['num_frames'] = T
            # [v2.9.2] 逐样本 overlap 必须随产物走:34 判门要证明"高重叠",
            # 汇总/审稿要能复核 overlap 分布,而 tmp_bench 的审计 JSON 会被清理。
            if 'overlap_ratio' in s.attrs:
                g.attrs['overlap_ratio'] = float(s.attrs['overlap_ratio'])
            g.attrs['pose_source'] = pose_source
            _pose_sources_seen.add(pose_source)   # → 循环后回填文件级 pose_injection
            # 持久化场景类别 —— P5 定性(generate_samples.py 的 --scene 过滤)按它筛样本;
            # 不写的话 pick_keys 拿到空串,--scene urban 匹配 0 个 → 「没有匹配的样本」崩溃。
            g.attrs['source_scene_category'] = scene_cat
            # [v2.4 §六] 源图身份透传 —— 重复/跨场景/跨 split 泄漏审计的依据
            # (check_benchmark_duplicates.py 读这些 attrs 做 source_key 检查)。
            for _a in ('source_dataset', 'source_split', 'source_index',
                       'source_path', 'source_class', 'source_tile_indices',
                       'source_original_hw', 'source_normalized_hw',
                       'source_preprocess', 'source_preprocess_scale',
                       'source_scene_id', 'semgate_frac', 'semgate_threshold',
                       'semgate_n_scored', 'semgate_tries',
                       'semgate_pres_ratio', 'semgate_pres_ratio_raw',
                       'semgate_scene_pool_mass', 'semgate_mean_margin'):
                if _a in s.attrs:
                    g.attrs[_a] = s.attrs[_a]
            # 持久化 GT 轨迹结构 —— 供 Track B 指标离线对生成视频打分。
            # 仅当存在真实 waypoints 时写入; fallback(无路点)样本跳过。
            if waypoints is not None and len(waypoints) >= 2:
                g.create_dataset('waypoints', data=np.asarray(waypoints, dtype=np.float32))
                g.create_dataset('segment_angles', data=np.asarray(seg_angles, dtype=np.float32))
                g.attrs['num_segments'] = int(n_seg)
            # Track B v2 校准修复(服务器 gt_oracle 复盘, net_dir_err=80.67°):
            #   trajectory = 真实逐帧窗口中心 (T,2)=(cy,cx) —— Track B 真实拐点时序的
            #     唯一来源: 真实 knot 帧 ≠ 均匀 linspace —— 该论断针对 v2.8 前的
            #     18/27/36 数据([0,9,17] vs [0,8,17]);v2.8 段界去重后 K=2 是 17 帧、
            #     结点 [0,8,16],均匀恰好正确。逐帧 trajectory 仍是唯一可靠来源。
            #     linspace 只能作为输出中显式标注的 fallback。
            #   full_scene = 完整 512×512 场景 —— deterministic/random 诊断基线必须
            #     从真实全景按原生窗口尺寸裁剪移动窗口;首帧 'image' 只是首帧窗口的
            #     480×832 上采样, 不是全景 —— 服务器上误传 scene_image=g['image']
            #     曾使两基线 valid_pct=0%。
            if trajectory is not None:
                g.create_dataset('trajectory', data=trajectory.astype(np.float32))
            else:
                missing_trajectory += 1
            if full_scene is not None:
                g.create_dataset('full_scene', data=full_scene, compression='gzip')
            else:
                missing_full_scene += 1
            written += 1

            if written % 20 == 0:
                print(f"  进度: {written}/{total}")

        if skipped:
            print(f"  ⚠ 共跳过 {len(skipped)} 个损坏样本: {skipped}")

        if missing_trajectory or missing_full_scene:
            print(f"  ⚠ 旧版源缺字段: trajectory×{missing_trajectory}, "
                  f"full_scene×{missing_full_scene} "
                  f"(这些样本 Track B 只能退回 linspace fallback;"
                  f"deterministic/random 基线诊断不可用)")

        meta.attrs['max_frames'] = max_f
        meta.attrs['total_instances'] = written
        # [v2.8.1] 文件级 pose_injection 由【逐样本实际 pose_source】回填。
        # 必须在样本循环之后写:v2.8 起逐样本是 plucker_from_trajectory(真实
        # 逐帧轨迹),而文件级此前恒写 plucker_from_waypoints —— train.py 的
        # pose_mode 一致性硬门和下游审计都读文件级,读到的是假话。
        _inj = sorted({str(_v) for _v in _pose_sources_seen})
        if len(_inj) > 1:
            raise RuntimeError(
                f"同一 H5 内混了多种 pose_source {_inj} —— 位姿口径不一致,"
                f"commanded 语义无法信任。")
        meta.attrs['pose_injection'] = _inj[0] if _inj else 'plucker_from_waypoints'
        # [RSRoll-Bench P0#4] 写**真实**使用的 schema；写死 'net' 会让
        # 连续式数据集带着 'net' 的指纹进 run_manifest，两批数据静默混用。
        meta.attrs['prompt_schema'] = PROMPT_SCHEMA
        # 生成口径从 benchmark H5 透传(缺则不写,不臆造)
        for _k in ('min_overlap_ratio', 'max_overlap_ratio', 'angle_quantize',
                   'min_segment_length', 'min_segments', 'max_segments',
                   'steps_per_segment'):
            if _k in src['metadata'].attrs:
                meta.attrs[_k] = src['metadata'].attrs[_k]

    elapsed = time.time() - t0
    print(f"转换完成: {written} 样本, 最大帧数 {max_f}, 耗时 {elapsed:.1f}s")
    print(f"输出: {dst_h5_path}")
    return written, max_f


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="从 remotesensebenchmark 构建 LoRA 训练数据集（完整高清版 v3）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
v3 改进:
  窗口宽高比自动匹配目标分辨率，消除 resize 时的 letterbox 模糊填充。
  默认窗口 128×222 对应目标 480×832。

数据集级别:
  aid       — blanchon/AID (600×600, ~2GB) 【推荐】
  resisc45  — blanchon/RESISC45 (256×256) + PatternNet
  mixed     — AID + RESISC45

示例:
  python build_dataset_from_benchmark.py -o demo.h5 -n 20 --no-hf
  python build_dataset_from_benchmark.py -o train.h5 -n 100 --dataset-tier aid
        """
    )
    parser.add_argument("-o", "--output", default="train.h5")
    parser.add_argument("-n", "--instances", type=int, default=100, help="总样本数")
    parser.add_argument("--target-height", type=int, default=480)
    parser.add_argument("--target-width", type=int, default=832)
    parser.add_argument("--no-hf", action="store_true", help="使用合成数据")
    parser.add_argument("--dataset-tier", choices=["aid", "resisc45", "patternnet", "whurs19", "mixed"],
                        default="aid", help="数据集级别 (默认: aid)")
    parser.add_argument("--scene-size", type=int, default=DEFAULT_SCENE_SIZE,
                        help=f"全场景尺寸 (默认 {DEFAULT_SCENE_SIZE})")
    # [v2.8] 段数范围可控 —— 34 号可学习性探针要单段(K=1,无转弯)最简任务;
    # 缺这两个参数时 34 的建库会 argparse unrecognized 当场失败(判门跑不起来)
    parser.add_argument("--min-overlap-ratio", type=float, default=None,
                        help="overlap 区间下界(默认 0.0);34 探针用预注册高重叠带")
    parser.add_argument("--max-overlap-ratio", type=float, default=None,
                        help="overlap 区间上界(默认 0.10 = 主 benchmark 硬约束)")
    parser.add_argument("--angle-quantize", type=int, default=0,
                        help="0=连续角(canonical);4=只走 E/N/W/S(34 判门用)")
    parser.add_argument("--min-segments", type=int, default=None,
                        help="每样本最少段数(默认 v5 config 的 2)")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="每样本最多段数(默认 4);min=max=1 → 单段直线")
    parser.add_argument("--window-height", type=int, default=None,
                        help=f"滑动窗口高度 (默认: 自动匹配目标宽高比)")
    parser.add_argument("--window-width", type=int, default=None,
                        help=f"滑动窗口宽度 (默认: 自动匹配目标宽高比)")
    parser.add_argument("--window-size", type=int, default=None,
                        help="滑动窗口尺寸（正方形，向后兼容）")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS_PER_SEGMENT,
                        help=f"每段步数 (默认 {DEFAULT_STEPS_PER_SEGMENT})")
    parser.add_argument("--temp-dir", default="temp_benchmark")
    parser.add_argument("--local-aid-dir", default=None,
                        help="Local AID image directory (bypasses HF download)")
    args = parser.parse_args()
    print(f"[threads] OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']} "
          f"MKL_NUM_THREADS={os.environ['MKL_NUM_THREADS']}")

    # [v2.4.3 修 P2] -n 硬限 4 的倍数:instances_per_scene = n//4 会静默把
    # -n 21 建成 20、-n 3 建成 4,与 --expect-n 门直接打架;固定四场景均分,
    # 不满足就当场报错而不是悄悄改数。
    if args.instances <= 0:
        parser.error("--instances 必须大于 0")
    if args.instances % 4 != 0:
        parser.error(f"--instances={args.instances} 必须是 4 的倍数"
                     f"(数据固定按 general/water/urban/rural 均分)")

    # ---- 窗口尺寸逻辑 ----
    # 优先级: --window-height/--window-width > --window-size > 自动计算
    if args.window_size is not None:
        # 向后兼容：正方形窗口
        args.window_height = args.window_size
        args.window_width = args.window_size
        print(f"[窗口] 使用正方形窗口: {args.window_size}×{args.window_size}")
        print(f"  ⚠ 提示: 正方形窗口会导致 letterbox 填充。"
              f"建议去掉 --window-size 以自动匹配目标宽高比。")
    else:
        # 自动计算匹配目标宽高比的窗口
        if args.window_height is None:
            args.window_height = DEFAULT_WINDOW_HEIGHT
        if args.window_width is None:
            # 根据 window_height 和目标宽高比计算 window_width
            args.window_width = round(args.window_height * args.target_width / args.target_height)
            # 确保不超过场景尺寸的 80%
            max_w = int(args.scene_size * 0.8)
            if args.window_width > max_w:
                args.window_width = max_w
                print(f"  ⚠ 自动计算的 window_width 超限，已缩小到 {max_w}")

        print(f"[窗口] 矩形窗口: {args.window_height}×{args.window_width}"
              f" (宽高比 {args.window_width/args.window_height:.3f},"
              f" 目标 {args.target_width/args.target_height:.3f})")

    use_hf = not args.no_hf
    local_aid_dir = args.local_aid_dir or os.environ.get("LOCAL_AID_DIR")

    # Step 1: 加载模块
    print(f"\n{'='*60}")
    print("Step 1: 加载 Benchmark 模块")
    print(f"{'='*60}\n")

    mod = _load_benchmark_module()

    if use_hf:
        _inject_hd_config(mod, args.dataset_tier, local_aid_dir=local_aid_dir)

    # Step 2: 构建 benchmark
    print(f"\n{'='*60}")
    print("Step 2: 生成 Benchmark")
    print(f"{'='*60}\n")

    benchmark_h5, stats = build_benchmark(mod, args, use_hf)

    # Step 3: 转换
    print(f"\n{'='*60}")
    print("Step 3: 转换为 LingBot 格式")
    print(f"{'='*60}\n")

    total, max_f = convert_benchmark_to_lingbot(
        benchmark_h5,
        args.output,
        target_size=(args.target_height, args.target_width),
        max_samples=args.instances,
    )

    print(f"\n{'='*60}")
    print("完成!")
    print(f"{'='*60}")
    print(f"输出: {args.output}")
    print(f"样本: {total}, 最大帧数: {max_f}")
    print(f"数据源: {'合成' if args.no_hf else args.dataset_tier.upper()}")
    print(f"\n验证命令:")
    print(f"  python lora_remotesense/check_dataset.py {args.output} --validate")
    print(f"\n帧间差异检查:")
    print(f"  python -c \"")
    print(f"import h5py, numpy as np")
    print(f"f = h5py.File('{args.output}', 'r')")
    print(f"keys = sorted(f['data'].keys())")
    print(f"for k in keys[:5]:")
    print(f"    s = f['data'][k]; vf = s['video_frames'][:]; T = vf.shape[0]")
    print(f"    diffs = [np.abs(vf[i].astype(float)-vf[i-1].astype(float)).mean() for i in range(1,T)]")
    print(f"    print(f'  {{k}}: {{T}}帧 | 帧间差={{np.mean(diffs):.1f}}')")
    print(f"f.close()\"")
    print(f"\n训练命令:")
    print(f"  python lora_remotesense/train.py \\")
    print(f"    --checkpoint_dir <模型目录> \\")
    print(f"    --data_path {args.output} \\")
    print(f"    --output_dir ./lora_output \\")
    print(f"    --num_epochs 10 --batch_size 1 \\")
    print(f"    --max_frames {min(33, max_f)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
