#!/usr/bin/env python3
# =============================================================================
# semantic_gate.py — 裁剪后语义保持门(v2.7,P0 人检失败的修复)
#
# 为什么需要:2026-07-27 服务器 P0 人检坐实 —— 自动四门(池/重复/泄漏/配额)
# 全 PASS,但 sample_00009(water/River)12 帧里 10 帧是纯草地、
# sample_00011(urban/StorageTanks)有 4 帧是 100% 河水。根因:AID 类别标签
# 是【整图】属性,轨迹滑窗裁剪后窗口离开类别主体,prompt 的场景语义
# ("a water-dominated remote-sensing scene")对这些帧就是假监督 —— 而文本
# 条件通道正是本 benchmark 要测的东西。缺的约束 = 裁剪后的多数帧必须仍然
# 符合 scene_category 声明。
#
# 本模块给出逐帧场景语义评分器,两处消费:
#   1. 建库期(v5 _process_scene):拒绝采样 —— 不达标就换源图重采(确定性
#      游标推进);单样本全部候选共享 20 次硬上限,耗尽后仅从已通过
#      overlap/轨迹/source identity 的候选中选最高语义分并大声告警;
#   2. 建后第四道门(server_run/check_semantic_consistency.py):独立复核,
#      与重复/泄漏门同构(sha-bound stamp,FAIL 即不放行)。
#
# 评分口径(gate 只对【声明】负责):
#   * 门只校验 scene_category(prompt 唯一声明的语义);source_class 是源图
#     provenance(对源图为真),不作为逐帧保持目标 —— 没有像素级标注时
#     "轨迹不离开类别主体"只能以场景级逐帧评分为代理。
#   * general 类 prompt 只说 "a mixed remote-sensing scene",无语义声明,不设门。
#   * 判定:CLIP 零样本 —— 帧对正类 prompt 集的最大相似度 > 对负类集的
#     最大相似度 ⇒ 该帧同意声明;样本通过 ⇔ 同意帧占比 ≥ 阈值。
#
# ⚠️ 阈值是【暂定值】,必须先在服务器现有 smoke H5 上跑
#     check_semantic_consistency.py --report-only 校准:人检已定
#     sample_00009/00011 为 FAIL,评分器必须把它们与其余样本显著分开,
#     否则调 prompt 库/阈值,不许带着不可靠的门去建库。
#
# 环境开关:
#   RSWM_SEMGATE=0             显式关门(大声警告;metadata 记录 disabled)
#   RSWM_SEMGATE_MODEL         默认 openai/clip-vit-base-patch32(离线预取)
#   RSWM_SEMGATE_MIN_FRAC      全局阈值(默认 = 冻结值 0.2777…,见 FROZEN_P0_*)
#   RSWM_SEMGATE_MIN_FRAC_WATER / _URBAN / _RURAL   按类覆盖
#   RSWM_SEMGATE_MAX_TRIES     建库期每样本槽源图重采次数(默认 6,硬上限 20)
# =============================================================================
import json
import os

# This module is also used by standalone audit tools.  Apply the same native
# thread ceiling as the canonical dataset entry before NumPy / torch are
# imported, so CLIP CPU fallback cannot fan out to hundreds of workers.
_NATIVE_THREAD_CAP = 8
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    try:
        _thread_count = int(os.environ.get(_thread_env, str(_NATIVE_THREAD_CAP)))
    except ValueError:
        _thread_count = _NATIVE_THREAD_CAP
    os.environ[_thread_env] = str(max(1, min(_thread_count, _NATIVE_THREAD_CAP)))

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from pathlib import Path

import numpy as np

SEMGATE_VERSION = "semgate_v3_1_20260729"  # 评分口径;判定规则见 FROZEN_P0_*
MAX_SEMGATE_TRIES = 20

# =============================================================================
# [v2.7.7 校准失败重设计] prompt 库不再手写,从 scene_taxonomy.json 派生。
#
# 20 样本人检校准坐实 v1 手写库的系统性冲突:干扰负类 "grassland"/"bare land"
# 与 rural 池的合法类 Meadow/BareLand 直接对撞 → 00016 BareLand 得 0.00、
# 00018 Farmland 0.11,比真坏样本(00009=0.22)还低,不存在能分开的全局阈值。
# 根治 = 按构造消灭冲突:
#   正类(cat) = 本池全部类名成句(池覆盖=自动完整);
#   负类(cat) = 其余三池(含 general)类名成句;
#   手写干扰库删除 —— 四池互斥由 taxonomy 导入期断言保证,冲突不可能再出现。
# 库按数据集构建(AID 建的样本用 AID 池;union 仅作无 dataset 信息时的回退,
# 注意 union 会把 RESISC45 的 wetland 引入 water 正类,可能放过 00009 型样本)。
# =============================================================================
_TAXONOMY_FILE = Path(__file__).resolve().parent / "scene_taxonomy.json"
_TAXONOMY = json.loads(_TAXONOMY_FILE.read_text(encoding="utf-8"))
GATED_CATEGORIES = ("water", "urban", "rural")

# [v2.7.11] 互斥轻量断言(本模块被 checker/校准器独立 import,不经过
# build_dataset_from_benchmark 的完整断言 —— taxonomy 若被改坏,评分进程
# 必须自己拦住,不许拿冲突库静默打分)
for _ds, _pools in _TAXONOMY["groups"].items():
    _seen = {}
    for _c, _lst in _pools.items():
        for _x in _lst:
            _n = "".join(ch.lower() for ch in str(_x) if ch.isalnum())
            if _n in _seen and _seen[_n] != _c:
                raise RuntimeError(
                    f"scene_taxonomy.json 池冲突: {_ds} 的 {_x!r} 同时在 "
                    f"{_seen[_n]!r} 与 {_c!r} —— 修 taxonomy 后再评分。")
            _seen[_n] = _c

# 弱分母审查警戒线(单一定义;checker/校准器 import,不各写一份)
WEAK_DENOM_WARN = 0.05
# [v2.7.12/C3] 类质量独立警戒线:单类 softmax 质量天然比 10 类池质量低
# 1-2 个数量级(29 组、TAU=100),复用 0.05 会把绝大多数行标红。暂定值,
# 以真实分布重定。
WEAK_CLS_DENOM_WARN = 0.01

# 类名 → 可读短语(snake_case 拆词;少数类补语境词,只许加不许换义)
_PHRASE_OVERRIDES = {
    # [v2.7.8 R1] 全类覆盖:模板是 "…showing {p}",p 必须是自然名词短语。
    # 只许补语境词不许换义;新类入 taxonomy 时缺表项会被单测抓住。
    # ---- water ----
    "pond": "a pond of water", "port": "a harbor port with water and ships",
    "river": "a river", "beach": "a beach shoreline",
    "island": "an island surrounded by water", "lake": "a lake",
    "sea_ice": "sea ice on water", "wetland": "a wetland with open water",
    "harbor": "a harbor with boats on the water",
    "ferry_terminal": "a ferry terminal with docks on the water",
    # ---- urban ----
    "commercial": "a commercial district with buildings",
    "commercial_area": "a commercial district with buildings",
    "dense_residential": "a dense residential area",
    "medium_residential": "a medium density residential area",
    "industrial": "an industrial area with factories",
    "industrial_area": "an industrial area with factories",
    "parking": "a parking lot with cars", "parking_lot": "a parking lot with cars",
    "parking_space": "parking spaces with cars",
    "railway_station": "a railway station with tracks",
    "railway": "railway tracks", "school": "a school campus",
    "storage_tanks": "industrial storage tanks",
    "storage_tank": "industrial storage tanks",
    "church": "a church building", "overpass": "a highway overpass",
    "mobile_home_park": "a mobile home park",
    "intersection": "a road intersection", "crosswalk": "a road crosswalk",
    "transformer_station": "an electrical transformer station",
    "nursing_home": "a nursing home building",
    "basketball_court": "basketball courts", "tennis_court": "tennis courts",
    "wastewater_treatment_plant": "a wastewater treatment plant",
    "swimming_pool": "a swimming pool", "shipping_yard": "a shipping container yard",
    # ---- rural ----
    "bare_land": "bare land and exposed soil",
    "meadow": "a meadow of grassland",
    "farmland": "farmland with crop fields",
    "circular_farmland": "circular farmland fields",
    "rectangular_farmland": "rectangular farmland fields",
    "sparse_residential": "sparse residential houses among fields",
    "forest": "a forest of trees", "chaparral": "chaparral shrubland",
    "terrace": "terraced fields",
    "christmas_tree_farm": "a christmas tree farm",
    "oil_gas_field": "an oil and gas field", "oil_well": "oil well pads",
    "solar_panel": "solar panel arrays",
    # ---- general ----
    "airport": "an airport with runways", "airplane": "an airplane on the ground",
    "bridge": "a large bridge", "center": "a city center plaza",
    "desert": "a desert of sand", "mountain": "mountains",
    "park": "an urban park with trees and paths",
    "playground": "a playground with sports fields",
    "square": "a public square plaza", "stadium": "a stadium",
    "viaduct": "a viaduct with elevated roads", "resort": "a resort area",
    "palace": "a palace building", "freeway": "a freeway highway",
    "golf_course": "a golf course", "runway": "an airport runway",
    "football_field": "a football field", "baseball_field": "a baseball field",
    "cemetery": "a cemetery", "residential": "a residential area",
}
_CLASS_TEMPLATES = ("an aerial photo showing {p}", "a satellite image showing {p}")
# 每类附加的通用正类(不含任何其他池的类语义,防"开阔水面/纯建筑区"漏接)
_GENERIC_POS = {
    "water": ["an aerial view dominated by open water"],
    "urban": ["an aerial view of a built-up area with buildings and streets"],
    "rural": ["an aerial view of open countryside"],
}


def class_phrase(cls: str) -> str:
    c = str(cls).strip()
    return _PHRASE_OVERRIDES.get(c, c.replace("_", " "))


def _norm_cls(cls: str) -> str:
    return "".join(ch.lower() for ch in str(cls) if ch.isalnum())


def build_prompt_bank(dataset: Optional[str] = None) -> Dict[str, list]:
    """构建 (prompts, cat_of, class_of) 平行表。dataset=None → 全数据集 union。

    条目含四池(general 类只作负类来源,永不作为被门类别)。同一 (cat, 类)
    出两条模板句;通用正类 class_of=None。"""
    groups = _TAXONOMY["groups"]
    if dataset is not None:
        if dataset not in groups:
            raise ValueError(
                f"未知 dataset={dataset!r};taxonomy 里有 {sorted(groups)}")
        pools = {c: {(_norm_cls(x), x) for x in groups[dataset][c]}
                 for c in groups[dataset]}
    else:
        pools = {}
        for ds in groups.values():
            for c, cls_list in ds.items():
                pools.setdefault(c, set()).update((_norm_cls(x), x) for x in cls_list)
    prompts, cat_of, class_of, groups_ = [], [], [], []
    for c in sorted(pools):
        seen = set()
        for nc, raw in sorted(pools[c]):
            if nc in seen:
                continue
            seen.add(nc)
            idxs = []
            for t in _CLASS_TEMPLATES:
                idxs.append(len(prompts))
                prompts.append(t.format(p=class_phrase(raw)))
                cat_of.append(c)
                class_of.append(nc)
            groups_.append({"cat": c, "cls": nc, "idx": idxs})
        for gp in _GENERIC_POS.get(c, []):
            groups_.append({"cat": c, "cls": None, "idx": [len(prompts)]})
            prompts.append(gp)
            cat_of.append(c)
            class_of.append(None)
    # [v2.7.8 R2] groups:类级聚合单元(每类 2 条模板)。评分对组内取均值再比,
    # 单条模板的噪声峰值被抑制 —— 负池条目远多于正池(AID rural 11 vs 50)时,
    # 逐条取 max 的噪声偏置(E[max] 随 n 增长)会把 margin 系统性压负。
    # 池间「类数不对称」保留:那是 taxonomy 的语义覆盖差异,不是噪声。
    return {"prompts": prompts, "cat_of": cat_of, "class_of": class_of,
            "groups": groups_, "dataset": dataset or "union"}


# =============================================================================
# [2026-07-29 拍板冻结 · 受控 P0 规则] 四轮真值校准终局(预注册决策规则执行):
#   规则 = 单特征 frac >= 0.2777777777777778(9 帧;离散帧占比 2/9 与 3/9 的
#   分界中点 —— 必须保存全精度,不许写成 0.28)
#   校准结果 = 0 坏样本放走 / 3 有效样本误拒(00006 Port、00016 BareLand、
#   00019 Farmland;误拒只费池不脏数据,datasheet 记录欠采样偏置)
#   clspres 族含事后 OR 组合均不采纳(复杂 OR + 非完美分离,预注册禁止
#   结果出来后改规则);clspres/raw/scene_cls_mass 仅作诊断证据。
#   决策留档 = 服务器 rule_decision_v2712.txt(只读 + SHA256)。
#   范围 = 受控 P0;生产/全量规则未冻结,第五轮特征迭代禁止。
# =============================================================================
FROZEN_P0_RULE = "frac"
FROZEN_P0_THRESHOLD = 0.2777777777777778
FROZEN_P0_FRAMES = 9


def default_thresholds() -> Dict[str, float]:
    """默认阈值 = 拍板冻结值(env 覆盖仅供校准/诊断,受控 P0 不得覆盖)。"""
    base = float(os.environ.get("RSWM_SEMGATE_MIN_FRAC",
                                repr(FROZEN_P0_THRESHOLD)))
    th = {}
    for c in ("water", "urban", "rural"):
        th[c] = float(os.environ.get(f"RSWM_SEMGATE_MIN_FRAC_{c.upper()}", base))
    return th


@dataclass
class GateVerdict:
    category: str
    passed: bool
    frac: Optional[float]          # 场景同意帧占比;None = 未评(general/关门)
    n_scored: int
    threshold: Optional[float]
    per_frame: List[bool] = field(default_factory=list)
    skipped: bool = False          # general/关门 → True
    # [v2.7.7 双轨+诊断] 校准失败复盘要求:没有逐帧依据就无法调规则
    mean_margin: Optional[float] = None       # mean(pos_max - neg_max)
    retention_frac: Optional[float] = None    # source_class 逐帧 top-1 保持占比
    # [v2.7.9 场景参照保真比] mean_f min(1, 声明池概率质量(帧)/质量(源场景))。
    # 动机(二轮校准实证):零样本文本判据有固有混淆(码头船排≈停车场、
    # BareLand≈Desert),逐帧硬比在 ±0.01 margin 下是掷硬币;而帧与它的
    # full_scene 共享同样的混淆 —— 比值让混淆在分子分母抵消,量的是
    # "源场景里的声明语义有多少比例在裁剪后幸存"。
    pres_ratio: Optional[float] = None        # mean min(1, raw)
    pres_ratio_raw: Optional[float] = None    # mean raw(未截断,饱和分析用)
    scene_pool_mass: Optional[float] = None   # 分母;小值=高保真比不可轻信
    # [v2.7.12 三轮校准后新特征] 具体类别保真比:source_class 单类质量的
    # 帧/场景之比 —— 三轮实证"场景池保真"分不开(Port 帧仍有码头和船,但
    # 池级更像停车场;BareLand 像 Desert 压低 rural 池)。类别保真直接问
    # "River/StorageTanks/Port/BareLand 这个类还在不在",比 retention 的
    # 全库 top-1 宽松,比池质量聚焦。假设待四轮校准检验。
    cls_pres_ratio: Optional[float] = None
    cls_pres_ratio_raw: Optional[float] = None
    scene_cls_mass: Optional[float] = None
    details: List[dict] = field(default_factory=list)   # 逐帧 pos/neg/胜出词
    bank_dataset: Optional[str] = None


class SemanticGate:
    """逐帧场景语义评分器。backend='clip'(生产)或可注入打分函数(测试)。

    score_fn 注入契约:score_fn(frames_uint8_list, category) -> List[bool]
    (逐帧"是否同意该场景声明")。注入时不加载任何模型。
    """

    def __init__(self, backend: str = "clip", thresholds: Optional[Dict[str, float]] = None,
                 frames_per_sample: int = 9, model_name: Optional[str] = None,
                 device: Optional[str] = None, score_fn=None):
        self.backend = backend
        self.thresholds = dict(thresholds) if thresholds else default_thresholds()
        _missing = [c for c in GATED_CATEGORIES if c not in self.thresholds]
        if _missing:
            raise ValueError(f"thresholds 缺类别 {_missing}(白烧一次前向才 KeyError"
                             f"的旧行为已废)")
        self.frames_per_sample = int(frames_per_sample)
        self.model_name = model_name or os.environ.get(
            "RSWM_SEMGATE_MODEL", "openai/clip-vit-base-patch32")
        self._device = device
        self._score_fn = score_fn
        self._model = None
        self._processor = None
        self._banks: Dict[str, dict] = {}
        self._bank_feats: Dict[str, object] = {}

    # ---------- 模型加载(建库开工前 eager 调用,别烧完数据才发现缺权重) ----------
    def ensure_ready(self):
        if self._score_fn is not None:
            return
        if self._model is None:
            try:
                import torch  # noqa: F401
                from transformers import CLIPModel, CLIPProcessor
            except ImportError as e:
                raise RuntimeError(
                    f"语义门需要 torch+transformers(CLIP 零样本):{e}。"
                    f"rswm 环境应已具备;确认后仍失败可 RSWM_SEMGATE=0 显式关门"
                    f"(不推荐 —— P0 人检失败正是缺这道门)。") from e
            try:
                self._model = CLIPModel.from_pretrained(self.model_name).eval()
                self._processor = CLIPProcessor.from_pretrained(self.model_name)
            except Exception as e:
                raise RuntimeError(
                    f"CLIP 权重加载失败({self.model_name}):{e}\n"
                    f"离线节点须先在联网机预取:bash server_run/offline_prefetch.sh"
                    f"(v2.7 起默认包含该权重,~600MB)。") from e
            import torch
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(self._device)

    @staticmethod
    def _coerce_features(out, projection, kind: str):
        """[v2.7.1/v2.7.6 服务器实证修复] 把 get_text_features /
        get_image_features 的返回归一为 2D【投影后】特征张量。Transformers
        的 helper 返回契约经历过两次漂移:
          * 旧版直接返回 Tensor;
          * 新版返回 BaseModelOutputWithPooling,且 pooler_output 可能已经由
            helper 投影到共同空间(v5 实测 image:512),也可能仍是编码器维度
            (兼容分支 image:768)。

        因此绝不能看到 pooler_output 就无条件再过 projection。归一顺序:
          ① 已是张量(有 .norm/.dim)→ 原样;
          ② CLIPOutput 风格:有 {kind}_embeds(已投影)→ 取它;
          ③ BaseModelOutputWithPooling:按 projection 的 in/out_features 判定:
             pooler 尾维=out_features → 已投影,直接用;
             pooler 尾维=in_features → 未投影,只投影一次;
             其他维度 → 硬错。
          ④ 都没有 → 硬错并列出可用字段,不猜。
        当 projection 是方阵(text 512→512)时维度无法区分两种来源,优先遵守
        get_*_features 的 helper 契约,把 pooler_output 视为已投影,避免重复
        投影。末了强制 2D 和共同空间维度校验。
        """
        p_in = getattr(projection, "in_features", None)
        p_out = getattr(projection, "out_features", None)
        weight = getattr(projection, "weight", None)
        weight_shape = getattr(weight, "shape", None)
        if ((p_in is None or p_out is None) and
                weight_shape is not None and len(weight_shape) == 2):
            p_out = weight_shape[0] if p_out is None else p_out
            p_in = weight_shape[1] if p_in is None else p_in

        def _field(name):
            if isinstance(out, Mapping):
                return out.get(name)
            return getattr(out, name, None)

        t = None
        if hasattr(out, "norm") and hasattr(out, "dim"):
            t = out
        elif _field(f"{kind}_embeds") is not None:
            t = _field(f"{kind}_embeds")
        elif _field("pooler_output") is not None:
            pooled = _field("pooler_output")
            if not hasattr(pooled, "dim") or pooled.dim() != 2:
                shape = getattr(pooled, "shape", "?")
                raise RuntimeError(
                    f"{kind} pooler_output 应为 2D (batch, dim),实得 {shape}。")
            if p_in is None or p_out is None:
                raise RuntimeError(
                    f"无法读取 {kind} projection 的 in_features/out_features,"
                    "不能安全判断 pooler_output 是否已经投影。")
            pool_dim = pooled.shape[-1]
            if pool_dim == p_out:
                # Transformers v5 get_*_features:helper 已把 pooler_output 投影。
                t = pooled
            elif pool_dim == p_in:
                # 兼容暴露编码器 pooler 的旧/第三方实现;只在维度吻合时投影。
                t = projection(pooled)
            else:
                raise RuntimeError(
                    f"{kind} pooler_output 尾维 {pool_dim} 既不等于 projection "
                    f"输入维 {p_in},也不等于输出维 {p_out};拒绝猜测投影状态。")
        if t is None:
            available = (list(out.keys())[:12] if isinstance(out, Mapping) else
                         [a for a in dir(out) if not a.startswith("_")][:12])
            raise RuntimeError(
                f"无法从 {type(out).__name__} 提取 {kind} 特征;可用字段: "
                f"{available} —— "
                f"transformers API 又变了,更新 _coerce_features 的归一分支。")
        if t.dim() != 2:
            raise RuntimeError(
                f"{kind} 特征应为 2D (batch, dim),实得 {tuple(t.shape)} —— "
                f"提取到了错误字段(如 last_hidden_state),拒绝继续。")
        if p_out is not None and t.shape[-1] != p_out:
            raise RuntimeError(
                f"{kind} 特征尾维应为共同投影维 {p_out},实得 {t.shape[-1]} —— "
                "返回值既非已投影特征,也未安全通过 projection。")
        return t

    def _bank(self, dataset: Optional[str]):
        key = dataset or "union"
        if key not in self._banks:
            self._banks[key] = build_prompt_bank(dataset)
        return self._banks[key]

    def _bank_text_feats(self, dataset: Optional[str]):
        key = dataset or "union"
        if key not in self._bank_feats:
            import torch
            bank = self._bank(dataset)
            with torch.no_grad():
                tok = self._processor(text=bank["prompts"], return_tensors="pt",
                                      padding=True, truncation=True).to(self._device)
                tf = self._coerce_features(
                    self._model.get_text_features(**tok),
                    self._model.text_projection, "text")
                tf = tf / tf.norm(dim=-1, keepdim=True)
            self._bank_feats[key] = tf
        return self._bank_feats[key]

    @staticmethod
    def _pad_square(img: np.ndarray) -> np.ndarray:
        """[v2.7.11/C2 修复] 边缘复制 pad 成正方形再送 CLIP。

        CLIPProcessor 出厂配置 = 短边 resize 224 + 中心裁剪 224² ——
        矩形帧(宽高比 1.73)会被砍掉 42% 宽度:门只看到窗口中心方块,
        而 512² 的 full_scene 一寸不丢。这既让"窗口离开类别主体"最先发生
        的水平边缘从未被评过分(两轮校准分不开的候选根因),也破坏保真比
        "帧与场景共享同样视野口径"的前提。帧与 full_scene 走同一函数。"""
        h, w = img.shape[:2]
        if h == w:
            return img
        side = max(h, w)
        top = (side - h) // 2
        left = (side - w) // 2
        return np.pad(img, ((top, side - h - top), (left, side - w - left), (0, 0)),
                      mode="edge")

    def _sim_matrix_clip(self, frames: Sequence[np.ndarray],
                         dataset: Optional[str]) -> np.ndarray:
        """(n_frames, n_prompts) 余弦相似度矩阵 —— 一次算全库,场景/保持度共用。
        全部图像先 _pad_square(C2:消除中心裁剪的 42% 视野丢失)。"""
        import torch
        from PIL import Image
        self.ensure_ready()
        tf = self._bank_text_feats(dataset)
        imgs = []
        for f in frames:
            a = np.asarray(f)[..., :3]
            if a.dtype != np.uint8:
                # float32 0-255 是本链常态;值域 ≤1.5 = 归一化图,astype 会截成
                # 全黑喂 CLIP —— 拒绝,不静默产垃圾分
                if a.size and float(np.nanmax(a)) <= 1.5:
                    raise ValueError(
                        f"帧 dtype={a.dtype} 且值域≤1.5(疑似归一化图)—— "
                        f"语义门要求 0-255 量程,拒绝静默截断。")
                a = np.clip(a, 0, 255).astype(np.uint8)
            imgs.append(Image.fromarray(self._pad_square(a)))
        with torch.no_grad():
            px = self._processor(images=imgs, return_tensors="pt").to(self._device)
            feats = self._coerce_features(
                self._model.get_image_features(**px),
                self._model.visual_projection, "image")
            feats = feats / feats.norm(dim=-1, keepdim=True)
            if feats.shape[-1] != tf.shape[-1]:
                raise RuntimeError(
                    f"图文特征维度不一致: image {feats.shape[-1]} vs text "
                    f"{tf.shape[-1]} —— 某一侧走了未投影路径,禁止比相似度。")
            return (feats @ tf.T).cpu().numpy()

    _SOFTMAX_TAU = 100.0    # CLIP 标准 logit scale(≈exp(4.6));比值判据对它不敏感
    _EPS_MASS = 1e-6        # 分母下限。⚠️ 分母(场景池质量)本身小时,高保真比
                            # 可能是"分母太小"而非"真保留" —— raw 值与场景质量
                            # 都落盘,由 checker 警示 + 校准表判读,不静默。

    def _class_mass(self, row: np.ndarray, groups, cls_nc: str) -> float:
        """source_class 单类概率质量(与 _pool_mass 同一 softmax 空间)。"""
        cls_ids = [j for j, g in enumerate(groups) if g["cls"] is not None]
        gsc = np.array([float(np.mean(row[groups[j]["idx"]])) for j in cls_ids])
        z = np.exp(self._SOFTMAX_TAU * (gsc - gsc.max()))
        pz = z / z.sum()
        return float(sum(pz[k] for k, j in enumerate(cls_ids)
                         if groups[j]["cls"] == cls_nc))

    def _pool_mass(self, row: np.ndarray, groups, cat: str) -> float:
        """声明池概率质量:类组(模板均值)分数 softmax 后,本池组的概率和。"""
        cls_ids = [j for j, g in enumerate(groups) if g["cls"] is not None]
        gsc = np.array([float(np.mean(row[groups[j]["idx"]])) for j in cls_ids])
        z = np.exp(self._SOFTMAX_TAU * (gsc - gsc.max()))
        pz = z / z.sum()
        return float(sum(pz[k] for k, j in enumerate(cls_ids)
                         if groups[j]["cat"] == cat))

    def _score_from_sims(self, sims: np.ndarray, cat: str,
                         dataset: Optional[str],
                         source_class: Optional[str],
                         scene_row: Optional[np.ndarray] = None):
        """由相似度矩阵产出 10 元组 (per_frame_agree, details, mean_margin,
        retention_frac, pres_ratio, pres_raw_mean, scene_mass,
        cls_pres, cls_pres_raw, scene_cls)。

        帧同意 ⇔ 本池最大相似度 > 其余三池最大相似度;
        保持度 ⇔ 全库 class 条目 top-1 == source_class(归一化名)。
        """
        bank = self._bank(dataset)
        groups = bank["groups"]
        if not any(g["cat"] == cat and g["cls"] is not None for g in groups):
            # generic 组不算数:类列表为空的池只剩 generic,pool_mass 恒 0,
            # pres 分母退化 —— 守卫按【类组】判
            raise RuntimeError(f"库({bank['dataset']})里没有 {cat} 的正类【类】条目")
        pos_ids = [j for j, g in enumerate(groups) if g["cat"] == cat]
        neg_ids = [j for j, g in enumerate(groups) if g["cat"] != cat]
        cls_ids_all = [j for j, g in enumerate(groups) if g["cls"] is not None]
        src_nc = _norm_cls(source_class) if source_class else None
        # [v2.7.11/C1 修复] 拼接样本 source_class="stitched" 是哨兵不是类名,
        # 旧行为逐帧 top-1 永不等于它 → retention 恒 0.0(假证据,污染待拍板
        # 规则:None 豁免失效、OR 分支恒 False)。哨兵 → None(指标不适用);
        # 其余不在库中的类名 = source_class 与 dataset 错配,fail-loud。
        if src_nc is not None:
            _bank_classes = {g["cls"] for g in groups if g["cls"] is not None}
            if src_nc not in _bank_classes:
                if src_nc == "stitched":
                    src_nc = None
                else:
                    raise ValueError(
                        f"source_class={source_class!r} 不在 {bank['dataset']} 库的"
                        f"类集合中 —— source_class 与 dataset 错配(或类名拼错),"
                        f"拒绝把保持度静默记 0/None。")
        per, details = [], []
        n_ret_agree, n_ret = 0, 0
        for i in range(sims.shape[0]):
            row = sims[i]
            # [R2] 组分 = 组内模板均值(每类 2 条),再在组间比 max ——
            # 抑制单模板噪声峰值造成的大负池 max 偏置
            gs = [float(np.mean(row[g["idx"]])) for g in groups]
            def _best(sub):
                j = max(sub, key=lambda k: gs[k])
                return j, gs[j]
            pj, pv = _best(pos_ids)
            nj, nv = _best(neg_ids)
            agree = bool(pv > nv)
            per.append(agree)
            def _top_prompt(g):
                return bank["prompts"][max(g["idx"], key=lambda k: float(row[k]))]
            d = {"pos_max": pv, "neg_max": nv, "margin": float(pv - nv),
                 "top_pos": _top_prompt(groups[pj]),
                 "top_pos_class": groups[pj]["cls"],
                 "top_neg": _top_prompt(groups[nj]),
                 "top_neg_cat": groups[nj]["cat"],
                 "top_neg_class": groups[nj]["cls"],
                 "agree": agree}
            if src_nc is not None:
                ranked = sorted(cls_ids_all, key=lambda k: gs[k], reverse=True)
                d["top_classes"] = [(groups[j]["cls"], round(gs[j], 4))
                                    for j in ranked[:3]]   # [R4] top-3 供同义类分析
                d["top_class_overall"] = groups[ranked[0]]["cls"]
                d["class_retained"] = bool(groups[ranked[0]]["cls"] == src_nc)
                n_ret += 1
                n_ret_agree += int(d["class_retained"])
            details.append(d)
        mean_margin = float(np.mean([d["margin"] for d in details])) if details else None
        retention = (n_ret_agree / n_ret) if n_ret else None
        pres_ratio, pres_raw_mean, scene_mass = None, None, None
        cls_pres, cls_pres_raw, scene_cls = None, None, None
        if scene_row is not None and sims.shape[0] > 0:   # 0 帧防 NaN 静默落盘
            scene_mass = self._pool_mass(scene_row, groups, cat)
            ratios, raws = [], []
            for i in range(sims.shape[0]):
                fm = self._pool_mass(sims[i], groups, cat)
                raw = fm / max(scene_mass, self._EPS_MASS)
                details[i]["pool_mass"] = round(fm, 4)
                details[i]["pres_raw"] = round(raw, 4)     # 未截断,饱和分析用
                raws.append(raw)
                ratios.append(min(1.0, raw))
            pres_ratio = float(np.mean(ratios))
            pres_raw_mean = float(np.mean(raws))
            # [v2.7.12] 类别保真比(source_class 有效且在库中才算)
            if src_nc is not None:
                scene_cls = self._class_mass(scene_row, groups, src_nc)
                crs, crr = [], []
                for i in range(sims.shape[0]):
                    cm = self._class_mass(sims[i], groups, src_nc)
                    craw = cm / max(scene_cls, self._EPS_MASS)
                    details[i]["cls_mass"] = round(cm, 4)
                    details[i]["cls_pres_raw"] = round(craw, 4)
                    crr.append(craw)
                    crs.append(min(1.0, craw))
                cls_pres = float(np.mean(crs))
                cls_pres_raw = float(np.mean(crr))
        return (per, details, mean_margin, retention,
                pres_ratio, pres_raw_mean, scene_mass,
                cls_pres, cls_pres_raw, scene_cls)

    # ---------- 公共入口 ----------
    @staticmethod
    def sample_frame_indices(T: int, k: int) -> List[int]:
        """确定性抽帧:均匀铺满含首尾;T<=k 全取。"""
        if T <= k:
            return list(range(T))
        return sorted(set(int(round(x)) for x in np.linspace(0, T - 1, k)))

    def check_frames(self, frames, scene_category: str,
                     source_class: Optional[str] = None,
                     dataset: Optional[str] = None,
                     full_scene=None) -> GateVerdict:
        """场景语义门 + [v2.7.7] 可选 source_class 保持度双轨。

        v5 旧调用 check_frames(frames, cat) 完全兼容(保持度轨缺省关闭)。
        判定规则仍是【场景轨】frac ≥ threshold —— 组合规则(margin/保持度
        是否参与判定)待 20 样本真值校准(calibrate_semantic_gate.py)拍板后
        另行启用,不在未校准时擅自改判据。
        """
        cat = str(scene_category)
        # [v2.7.1 F5/P1] 只有 general 有意不设门;未知类别硬错
        if cat == "general":
            return GateVerdict(category=cat, passed=True, frac=None,
                               n_scored=0, threshold=None, skipped=True)
        if cat not in GATED_CATEGORIES:
            raise ValueError(
                f"未知 scene_category={cat!r}(合法: general/water/urban/rural)。"
                f"若 taxonomy 新增了类别,须同步更新 semantic_gate 的 "
                f"GATED_CATEGORIES/_GENERIC_POS 与阈值;拒绝静默跳过。")
        arr = np.asarray(frames)
        idxs = self.sample_frame_indices(len(arr), self.frames_per_sample)
        picked = [arr[i] for i in idxs]
        details, mean_margin, retention = [], None, None
        pres, pres_raw, scene_mass = None, None, None
        cls_pres, cls_pres_raw, scene_cls = None, None, None
        if self._score_fn is not None:
            # 测试桩:旧契约 List[bool];扩展契约 (List[bool], List[dict])
            out = self._score_fn(picked, cat)
            per = list(out[0]) if isinstance(out, tuple) else list(out)
            details = list(out[1]) if isinstance(out, tuple) else []
            if details and len(details) != len(per):
                raise RuntimeError(
                    f"桩评分器 details({len(details)})与判定数({len(per)})不等")
            if details:
                mean_margin = float(np.mean([d.get("margin", 0.0) for d in details]))
                ret = [d["class_retained"] for d in details if "class_retained" in d]
                retention = (sum(ret) / len(ret)) if ret else None
        else:
            # [v2.7.8 R3] 生产路径必须显式指定 dataset —— union 会把 RESISC45
            # 的 wetland 混入 water 正类,00009 型坏样本可能被放过。任何一条
            # PBS 路径漏传都在这里炸,不许猜、不许回退。
            if dataset is None:
                raise RuntimeError(
                    "check_frames(CLIP 路径)未指定 dataset —— 拒绝回退 union "
                    "prompt 库(跨数据集类会污染正类)。调用方必须显式传 "
                    "dataset(如 'blanchon/AID';H5 侧读 source_dataset attr)。")
            batch = picked + ([np.asarray(full_scene)]
                              if full_scene is not None else [])
            sims_all = self._sim_matrix_clip(batch, dataset)
            scene_row = sims_all[-1] if full_scene is not None else None
            sims = sims_all[:len(picked)]
            (per, details, mean_margin, retention,
             pres, pres_raw, scene_mass,
             cls_pres, cls_pres_raw, scene_cls) = \
                self._score_from_sims(sims, cat, dataset, source_class,
                                      scene_row=scene_row)
        if len(per) != len(picked):
            raise RuntimeError(f"评分器返回 {len(per)} 个判定,但送入 {len(picked)} 帧")
        frac = float(sum(bool(x) for x in per)) / max(1, len(per))
        thr = float(self.thresholds[cat])
        return GateVerdict(category=cat, passed=(frac >= thr), frac=frac,
                           n_scored=len(per), threshold=thr,
                           per_frame=[bool(x) for x in per],
                           mean_margin=mean_margin, retention_frac=retention,
                           pres_ratio=pres, pres_ratio_raw=pres_raw,
                           scene_pool_mass=scene_mass,
                           cls_pres_ratio=cls_pres,
                           cls_pres_ratio_raw=cls_pres_raw,
                           scene_cls_mass=scene_cls,
                           details=details,
                           bank_dataset=(dataset or "union"))


def gate_enabled() -> bool:
    return os.environ.get("RSWM_SEMGATE", "1") != "0"


def max_tries() -> int:
    raw = os.environ.get("RSWM_SEMGATE_MAX_TRIES", "6")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"RSWM_SEMGATE_MAX_TRIES 必须是正整数,实得 {raw!r}") from exc
    if value < 1:
        raise ValueError(
            f"RSWM_SEMGATE_MAX_TRIES 必须 >=1,实得 {value}")
    if value > MAX_SEMGATE_TRIES:
        print(f"[semgate][WARN] RSWM_SEMGATE_MAX_TRIES={value} 超过硬上限 "
              f"{MAX_SEMGATE_TRIES};已截断,防止单样本超长重采")
    return min(value, MAX_SEMGATE_TRIES)
