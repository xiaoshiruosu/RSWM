#!/usr/bin/env python3
# =============================================================================
# semgate_frozen_inputs.py — 语义门【判据输入】的全量冻结与核对(外部审查 P0-7)
#
# P0-7 的原话是"CLIP 模型 revision、prompt taxonomy 和权重 SHA 全部冻结"。
# 但只冻这三样是不够的:去读 lora_remotesense/semantic_gate.py 会发现,一帧
# 的 agree/reject 判定还依赖另外几项**同样会静默漂移**的东西 ——
#
#   * 预处理几何:_pad_square(边缘复制 pad 成正方形)。注释里写得很清楚,
#     CLIPProcessor 出厂是"短边 resize + 中心裁剪",矩形帧会被砍掉 42% 宽度。
#     改掉这一项,**每一个分数**都变,而 taxonomy 和权重 SHA 都察觉不到。
#   * 聚合口径:类级(每类 2 条模板)取均值再在组间比 max。改成逐条取 max,
#     大负池的 E[max] 偏置会把 margin 系统性压负(该模块 R2 注释的原话)。
#   * 抽帧:sample_frame_indices(T, 9) 决定「哪 9 帧被评分」。frac 是这 9 帧上
#     的占比,换抽帧规则等于换了统计量的样本。
#   * 环境变量:default_thresholds() 直接读 RSWM_SEMGATE_MIN_FRAC[_WATER…],
#     gate_enabled() 读 RSWM_SEMGATE。也就是说**不改一行代码、不换一个权重**
#     就能改判据。这是最容易被漏掉、也最容易被无心触发的漂移面。
#   * prompt 文本:类名短语表 _PHRASE_OVERRIDES 和模板 _CLASS_TEMPLATES 在
#     .py 里,不在 taxonomy.json 里。只冻 taxonomy 的 SHA 会漏掉它们。
#     本模块因此冻结**最终展开后的 prompt 列表的 SHA**(顺序敏感),一举覆盖
#     taxonomy 类表 + 短语表 + 模板 + 通用正类 + 池排序。
#
# 三态核对(本模块的核心设计,也是它与"打个 SHA 存起来"的区别):
#   match         核对过且一致
#   mismatch      核对过且不一致          → 非零退出
#   unverifiable  无法核对(如权重不在本机) → 同样非零退出
# 第三态必须与第一态严格区分。把"拿不到 → 当作没变"就是 fail-open,而本项目
# 历史上的 P0 几乎全是这么来的。逃生口 --allow-unverifiable 存在,但用了会在
# 输出里留 verification_degraded=true 的痕,且照样打印清单。
#
# 拿不到权重时 clip.weight_files 记 null + 明确 reason,**绝不编一个哈希**。
#
# 用法:
#   # 冻结(默认动作)
#   python server_run/semgate_frozen_inputs.py --dataset blanchon/AID \
#       -o $WORK/semgate_frozen_config.json
#   # 核对(CI / 建库前 / 出数前)
#   python server_run/semgate_frozen_inputs.py --verify $WORK/semgate_frozen_config.json
#   # 自检(不依赖任何外部产物)
#   python server_run/semgate_frozen_inputs.py --self-test
#
# 退出码:0 全部 match(或仅 unverifiable 且显式放行);2 有 mismatch;
#         3 无 mismatch 但有 unverifiable;1 用法/输入错误。
# =============================================================================
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_CONTRACT_DIR = ROOT / "lora_remotesense" / "new"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))

# 评分/校验契约唯一权威。这里只借它的 fail-closed 异常类型与 strict 口径 ——
# 本项目吃过"孪生入口各写一份必然漂移"的亏,新校验不再自己造一套。
try:
    from submission_contract import (SubmissionError, contract_summary,
                                     check_id_set)
except Exception as _e:      # noqa: BLE001
    raise SystemExit(
        f"[FATAL] 无法导入 submission_contract({_e})—— 冻结/核对必须与评分"
        f"共用同一份契约语义,不允许各写一套。") from _e

try:
    from lora_remotesense.semantic_gate import (  # noqa: E402
        FROZEN_P0_FRAMES, FROZEN_P0_RULE, FROZEN_P0_THRESHOLD,
        GATED_CATEGORIES, SEMGATE_VERSION, SemanticGate, _CLASS_TEMPLATES,
        _GENERIC_POS, _TAXONOMY, _TAXONOMY_FILE, build_prompt_bank,
        default_thresholds, gate_enabled)
except Exception as _e:      # noqa: BLE001
    raise SystemExit(
        f"[FATAL] 无法导入 lora_remotesense.semantic_gate({_e})—— "
        f"判据真源缺失时不许冻结一份猜出来的配置。") from _e

# run_manifest 是产物身份的唯一权威(可选:缺了不阻塞冻结,但会记 error)
try:
    from run_manifest import build_manifest as _build_manifest
except Exception:            # noqa: BLE001
    _build_manifest = None

FROZEN_SCHEMA = "rswm.semgate.frozen_inputs"
FROZEN_SCHEMA_VERSION = "1.0"
DEFAULT_OUTPUT_NAME = "semgate_frozen_config.json"
DEFAULT_DATASET = "blanchon/AID"

# 判据相关的环境变量(改了就等于改了判据,一律进 critical)
_VERDICT_ENV = ("RSWM_SEMGATE", "RSWM_SEMGATE_MODEL", "RSWM_SEMGATE_MIN_FRAC",
                "RSWM_SEMGATE_MIN_FRAC_WATER", "RSWM_SEMGATE_MIN_FRAC_URBAN",
                "RSWM_SEMGATE_MIN_FRAC_RURAL")
# 只影响"采样多少次"而非"怎么判"的环境变量 → advisory
_SAMPLING_ENV = ("RSWM_SEMGATE_MAX_TRIES",)

EXIT_OK, EXIT_USAGE, EXIT_MISMATCH, EXIT_UNVERIFIABLE = 0, 1, 2, 3


# =============================================================================
# 哈希工具
# =============================================================================
def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    """文件 SHA256。**不吞异常**:文件不在/读不了必须由调用方决定记
    unverifiable,而不是在这里返回 None 让上层误当成"一致"。"""
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def _sha256_json(obj: Any) -> str:
    """结构化对象的稳定哈希:sort_keys + 紧凑分隔符,避免缩进/键序造成假漂移。"""
    return _sha256_bytes(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8"))


# =============================================================================
# CLIP 快照定位(revision + 权重 SHA 的唯一来源)
# =============================================================================
def _hf_cache_roots() -> List[Path]:
    """HF 缓存候选根。顺序 = transformers 的实际查找优先级。"""
    roots = []
    for env in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(env)
        if v:
            roots.append(Path(v))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    out, seen = [], set()
    for r in roots:
        s = str(r)
        if s not in seen:
            seen.add(s)
            out.append(r)
    return out


def _hash_tree(root: Path) -> List[Dict[str, Any]]:
    """快照目录下**每一个**常规文件的 (相对名, sha256, 字节数)。

    为什么不只挑 *.safetensors:preprocessor_config.json 决定 resize/crop 口径、
    tokenizer* 决定文本切分,它们和权重一样直接改分数。挑着冻 = 留漂移面。
    符号链接(HF 缓存 snapshots/ 全是指向 blobs/ 的链接)按内容读,不记链接目标 ——
    blob 路径含 etag,换机器就变,不是内容身份。
    """
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        if not p.is_file():        # 悬空链接 → 让调用方看见,不静默跳过
            files.append({"name": str(p.relative_to(root)), "sha256": None,
                          "bytes": None, "error": "not a regular file"})
            continue
        files.append({"name": str(p.relative_to(root)),
                      "sha256": _sha256_file(p), "bytes": p.stat().st_size})
    return files


def resolve_clip_snapshot(model_id: str) -> Dict[str, Any]:
    """→ {kind, path, revision, files, snapshot_sha256, reason}

    reason 非 None ⇔ 这一项无法核对(kind=None)。四种情形:
      local_dir  model_id 就是本机目录(有 config.json)→ 无 VCS revision,
                 revision 记 null 但**权重 SHA 可核对**(两者独立,不可混为一谈);
      hf_cache   标准 HF 缓存 → refs/main 给 revision,snapshot 给权重 SHA;
      ambiguous  缓存里多个 snapshot 且没有 refs/main → 无法确定
                 transformers 会加载哪一个,拒绝猜(fail-closed);
      absent     本机没有 → 全部 unverifiable。
    """
    mp = Path(model_id)
    if mp.is_dir() and (mp / "config.json").is_file():
        files = _hash_tree(mp)
        return {"kind": "local_dir", "path": str(mp.resolve()),
                "revision": None, "files": files,
                "snapshot_sha256": _sha256_json(
                    [[f["name"], f["sha256"]] for f in files]),
                "reason": "model_id 是本机目录:没有上游 revision 可记(权重 SHA "
                          "仍已冻结);要 revision 请改用 HF repo id + 缓存"}
    if "/" not in model_id:
        return {"kind": None, "path": None, "revision": None, "files": None,
                "snapshot_sha256": None,
                "reason": f"model_id={model_id!r} 既不是本机目录,也不像 HF "
                          f"repo id(缺 org/name),无法定位权重"}
    org, name = model_id.split("/", 1)
    repo_dir_name = f"models--{org}--{name}".replace("/", "--")
    tried = []
    for root in _hf_cache_roots():
        repo = root / repo_dir_name
        tried.append(str(repo))
        if not repo.is_dir():
            continue
        snaps_dir = repo / "snapshots"
        snaps = sorted([d for d in snaps_dir.iterdir() if d.is_dir()]) \
            if snaps_dir.is_dir() else []
        ref_main = repo / "refs" / "main"
        rev = ref_main.read_text(encoding="utf-8").strip() \
            if ref_main.is_file() else None
        chosen, how = None, None
        if rev:
            cand = snaps_dir / rev
            if cand.is_dir():
                chosen, how = cand, "refs/main"
            else:
                return {"kind": None, "path": str(repo), "revision": rev,
                        "files": None, "snapshot_sha256": None,
                        "reason": f"refs/main={rev} 但 snapshots/{rev} 不存在 —— "
                                  f"缓存不完整,拒绝拿别的 snapshot 顶替"}
        elif len(snaps) == 1:
            chosen, how, rev = snaps[0], "unique snapshot", snaps[0].name
        elif len(snaps) > 1:
            return {"kind": None, "path": str(repo), "revision": None,
                    "files": None, "snapshot_sha256": None,
                    "reason": f"缓存里有 {len(snaps)} 个 snapshot 且无 refs/main —— "
                              f"无法确定 transformers 会加载哪一个,拒绝猜"}
        if chosen is None:
            continue
        files = _hash_tree(chosen)
        return {"kind": "hf_cache", "path": str(chosen), "revision": rev,
                "files": files,
                "snapshot_sha256": _sha256_json(
                    [[f["name"], f["sha256"]] for f in files]),
                "reason": None, "resolved_by": how}
    return {"kind": None, "path": None, "revision": None, "files": None,
            "snapshot_sha256": None,
            "reason": f"本机未找到 {model_id} 的权重(查过 {tried[:4]})—— "
                      f"revision / 权重 SHA 无法核对。联网机预取:"
                      f"bash server_run/offline_prefetch.sh"}


# =============================================================================
# 行为指纹:比字符串标签强得多的口径证据
# =============================================================================
_PAD_PROBE_SHAPE = (3, 5, 3)


def _preprocess_fingerprint() -> Dict[str, Any]:
    """在固定的非方形探针上跑 _pad_square,哈希输出。

    为什么要「跑」而不是记一个 "pad_square" 字符串:标签是人写的,函数体是机器
    执行的。把 mode="edge" 改成 "constant"、或把居中 pad 改成右下 pad,标签
    一个字都不用动,而每张矩形帧的 CLIP 输入都变了。跑一遍是唯一诚实的口径证据。
    """
    a = (np.arange(int(np.prod(_PAD_PROBE_SHAPE)), dtype=np.uint8)
         .reshape(_PAD_PROBE_SHAPE))
    out = SemanticGate._pad_square(a)
    return {"probe_shape": list(_PAD_PROBE_SHAPE),
            "out_shape": list(out.shape),
            "out_sha256": _sha256_bytes(np.ascontiguousarray(out).tobytes())}


def _frame_index_fingerprint() -> Dict[str, Any]:
    """抽帧规则指纹。frac 是"被抽中的 9 帧"上的占比 —— 换抽帧就是换样本。
    覆盖 canonical 帧数 9/17/25/33(见 submission_contract 注释)。"""
    return {str(T): SemanticGate.sample_frame_indices(T, FROZEN_P0_FRAMES)
            for T in (9, 17, 25, 33)}


# =============================================================================
# 采集当前环境的全部判据输入
# =============================================================================
def _field(value, *, severity: str, why: str,
           verifiable: bool = True, reason: Optional[str] = None) -> Dict[str, Any]:
    return {"value": value, "severity": severity, "why": why,
            "verifiable": bool(verifiable), "unverifiable_reason": reason}


def collect_inputs(dataset: str = DEFAULT_DATASET) -> Dict[str, Dict[str, Any]]:
    """采集当前环境下**每一个影响判定结果的输入**。

    severity 只有两档:
      critical  改了就换了判据 → --verify 时 mismatch 即非零退出;
      advisory  会改数值但不是 P0-7 点名的判据输入(如 torch/PIL 小版本)。
                默认不因它退出非零 —— 因为让核对器在每次 pip 升级后都失败,
                真实后果是它被人关掉,那才是彻底的 fail-open。要严格核对
                加 --strict-env。这是明摆着的取舍,不是遗漏。
    """
    f: Dict[str, Dict[str, Any]] = {}

    # ---- 代码本体 ----
    sg_path = ROOT / "lora_remotesense" / "semantic_gate.py"
    f["code.semantic_gate_sha256"] = _field(
        _sha256_file(sg_path), severity="critical",
        why="判据的实现本体;prompt 短语表/模板/聚合/预处理都在这个文件里")
    f["code.semgate_version"] = _field(
        SEMGATE_VERSION, severity="critical",
        why="模块自报的评分口径版本;与 sha 双记,便于人读出'改了哪一代'")

    # ---- 判定规则 ----
    f["decision.rule"] = _field(
        FROZEN_P0_RULE, severity="critical", why="单特征判据名;换特征即换门")
    f["decision.threshold_repr"] = _field(
        repr(FROZEN_P0_THRESHOLD), severity="critical",
        why="全精度阈值(2/9 与 3/9 的分界中点)。写成 0.28 会把 frac=0.2777… "
            "的样本判反,必须保 repr 全精度")
    f["decision.threshold_hex"] = _field(
        float(FROZEN_P0_THRESHOLD).hex(), severity="critical",
        why="阈值的位精确表示;JSON 文本往返若被任何工具重格式化,hex 仍能对上")
    f["decision.frames_per_sample"] = _field(
        int(FROZEN_P0_FRAMES), severity="critical",
        why="frac 的分母;9 帧决定了阈值 3/9 这个离散分界的位置")
    f["decision.thresholds_resolved"] = _field(
        {c: float(default_thresholds()[c]) for c in GATED_CATEGORIES},
        severity="critical",
        why="default_thresholds() 读 RSWM_SEMGATE_MIN_FRAC[_类],env 一设就换判据 —— "
            "冻结的是**解析后**的实际值,不是代码常量")
    f["decision.gate_enabled"] = _field(
        bool(gate_enabled()), severity="critical",
        why="RSWM_SEMGATE=0 会整门关掉;冻结时若为 False,这份配置根本不代表有门")
    f["decision.gated_categories"] = _field(
        list(GATED_CATEGORIES), severity="critical",
        why="general 不设门是刻意的(prompt 无语义声明);这个集合变了 = 管辖范围变了")

    # ---- taxonomy ----
    f["taxonomy.file_sha256"] = _field(
        _sha256_file(_TAXONOMY_FILE), severity="critical",
        why="P0-7 点名项:prompt taxonomy 的内容身份")
    f["taxonomy.taxonomy_version"] = _field(
        _TAXONOMY.get("taxonomy_version"), severity="critical",
        why="taxonomy 自报版本;改类映射必须同步改它(该文件 _notes 的约定)")
    f["taxonomy.prompt_template_version"] = _field(
        _TAXONOMY.get("prompt_template_version"), severity="critical",
        why="场景句模板版本(prompt 里唯一的语义声明来源)")

    # ---- prompt 库(展开后)----
    bank = build_prompt_bank(dataset)
    prompts = list(bank["prompts"])
    f["prompt_bank.dataset"] = _field(
        bank["dataset"], severity="critical",
        why="生产路径强制显式 dataset;union 会把 RESISC45 的 wetland 混进 water "
            "正类,00009 型坏样本可能被放过")
    f["prompt_bank.prompts_sha256"] = _field(
        _sha256_json(prompts), severity="critical",
        why="**最终展开、顺序敏感**的 prompt 列表哈希。这一项同时覆盖 taxonomy "
            "类表、_PHRASE_OVERRIDES 短语表、_CLASS_TEMPLATES、_GENERIC_POS 与"
            "池排序 —— 只冻 taxonomy.json 会漏掉住在 .py 里的那大半")
    f["prompt_bank.n_prompts"] = _field(
        len(prompts), severity="critical", why="条目数;人读的规模校验")
    f["prompt_bank.n_class_groups"] = _field(
        sum(1 for g in bank["groups"] if g["cls"] is not None),
        severity="critical",
        why="类级聚合单元数;组数变了说明池类表变了")
    f["prompt_bank.class_templates"] = _field(
        list(_CLASS_TEMPLATES), severity="critical",
        why="每类展开成句的模板;换措辞即换全部相似度")
    f["prompt_bank.generic_pos"] = _field(
        {k: list(v) for k, v in sorted(_GENERIC_POS.items())},
        severity="critical",
        why="每池附加的通用正类(防'开阔水面/纯建筑区'漏接);增删直接改 pos_max")
    pos_classes = {}
    for c in GATED_CATEGORIES:
        pos_classes[c] = sorted(g["cls"] for g in bank["groups"]
                                if g["cat"] == c and g["cls"] is not None)
    f["prompt_bank.positive_classes"] = _field(
        pos_classes, severity="critical",
        why="各受管辖池的正类归一化名(建库时'合法类'的定义)")

    # ---- 负类池构成 ----
    neg_classes = {}
    for c in GATED_CATEGORIES:
        neg_classes[c] = sorted(g["cls"] for g in bank["groups"]
                                if g["cat"] != c and g["cls"] is not None)
    f["negatives.construction"] = _field(
        "other_pools_incl_general__class_names_as_sentences",
        severity="critical",
        why="负类=其余三池(含 general)全部类名成句。v1 的手写干扰库与 rural 合法类"
            "(Meadow/BareLand)对撞,不存在能分开的全局阈值 —— 构成方式本身是判据")
    f["negatives.classes"] = _field(
        neg_classes, severity="critical",
        why="逐池负类清单;新类进 taxonomy 会改某池的 neg_max,这里能看见")
    f["negatives.n_pos_vs_neg_groups"] = _field(
        {c: [len(pos_classes[c]), len(neg_classes[c])] for c in GATED_CATEGORIES},
        severity="critical",
        why="正负组数不对称(AID rural 5 vs 24)是已知偏置来源:逐条取 max 时 "
            "E[max] 随 n 增长会压负 margin。类级均值聚合就是为压住它,故一并冻")

    # ---- 预处理 / 聚合 ----
    f["preprocess.geometry"] = _field(
        "pad_square_edge_replicate_then_clip_processor", severity="critical",
        why="pad-square(非 center-crop)。CLIPProcessor 出厂的短边 resize+中心裁剪"
            "会砍掉宽高比 1.73 帧的 42% 宽度,窗口水平边缘从未被评分 —— 这一项"
            "改变每一个分数")
    f["preprocess.pad_square_fingerprint"] = _field(
        _preprocess_fingerprint(), severity="critical",
        why="在固定非方形探针上实跑 _pad_square 的输出哈希;标签会骗人,输出不会")
    f["preprocess.frame_indices"] = _field(
        _frame_index_fingerprint(), severity="critical",
        why="抽帧规则:frac 的样本就是这些下标,换规则=换统计量")
    f["aggregation.scheme"] = _field(
        "class_group_template_mean__then_max_over_groups", severity="critical",
        why="组内(每类 2 模板)取均值再组间比 max;改成逐条 max 会引入负池规模偏置")
    f["aggregation.softmax_tau"] = _field(
        float(SemanticGate._SOFTMAX_TAU), severity="critical",
        why="池/类概率质量的温度(诊断特征 pres_ratio 族的量纲)")
    f["aggregation.eps_mass"] = _field(
        float(SemanticGate._EPS_MASS), severity="critical",
        why="保真比分母下限;它决定小分母时比值饱和到多大")

    # ---- CLIP 模型 ----
    model_id = os.environ.get("RSWM_SEMGATE_MODEL",
                              "openai/clip-vit-base-patch32")
    f["clip.model_id"] = _field(
        model_id, severity="critical", why="P0-7 点名项:模型标识")
    snap = resolve_clip_snapshot(model_id)
    unver = snap["reason"] if snap["kind"] is None else None
    f["clip.revision"] = _field(
        snap["revision"], severity="critical",
        why="P0-7 点名项:上游 commit。offline_prefetch.sh 拉的是 main(无 "
            "--revision),所以这一项只能事后从缓存 refs/main 读出来固定",
        verifiable=(snap["revision"] is not None),
        reason=(unver or (snap["reason"] if snap["revision"] is None else None)))
    f["clip.weight_files"] = _field(
        snap["files"], severity="critical",
        why="P0-7 点名项:权重 SHA。含 preprocessor_config.json 与 tokenizer* —— "
            "它们和 safetensors 一样直接改分数",
        verifiable=(snap["files"] is not None), reason=unver)
    f["clip.snapshot_sha256"] = _field(
        snap["snapshot_sha256"], severity="critical",
        why="整快照(名, sha) 列表的聚合哈希;一个数就能对完整个模型目录",
        verifiable=(snap["snapshot_sha256"] is not None), reason=unver)
    f["clip.snapshot_kind"] = _field(
        snap["kind"], severity="advisory",
        why="local_dir / hf_cache:定位方式本身不改分数,但决定 revision 是否存在")

    # ---- 环境变量 ----
    f["env.verdict_overrides"] = _field(
        {k: os.environ[k] for k in _VERDICT_ENV if k in os.environ},
        severity="critical",
        why="不改一行代码、不换一个权重就能改判据的那一组;空字典=没人覆盖")
    f["env.sampling_overrides"] = _field(
        {k: os.environ[k] for k in _SAMPLING_ENV if k in os.environ},
        severity="advisory",
        why="只影响每槽重采次数(池消耗),不改单样本判定")

    # ---- 运行时版本(advisory)----
    versions = {}
    for mod in ("numpy", "torch", "transformers", "PIL"):
        try:
            m = __import__(mod)
            versions[mod] = str(getattr(m, "__version__", "unknown"))
        except Exception:    # noqa: BLE001
            versions[mod] = None
    versions["python"] = sys.version.split()[0]
    f["env.runtime_versions"] = _field(
        versions, severity="advisory",
        why="transformers 的 get_*_features 返回契约变过两次(见 _coerce_features "
            "的双分支),PIL 的 resize 核也可能换 —— 会改数值,但不是 P0-7 点名的"
            "判据输入。默认 advisory,--strict-env 可提为 critical")
    return f


def build_frozen_config(dataset: str = DEFAULT_DATASET) -> Dict[str, Any]:
    fields = collect_inputs(dataset)
    cfg = {
        "schema": FROZEN_SCHEMA,
        "schema_version": FROZEN_SCHEMA_VERSION,
        "frozen_at_utc": _dt.datetime.now(_dt.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": dataset,
        "fields": fields,
        "n_unverifiable_at_freeze": sum(
            1 for v in fields.values() if not v["verifiable"]),
        "contract": contract_summary(),
    }
    if _build_manifest is not None:
        try:
            # role=probe_gate 在 run_manifest 里属 _NEVER_HEADLINE_ROLES ——
            # 判门产物永远不是 headline 数字,这正是它该有的身份。
            cfg["run_manifest"] = _build_manifest(
                role="probe_gate",
                prompt_schema=str(fields["taxonomy.prompt_template_version"]["value"]),
                extra={"artifact": "semgate_frozen_config"})
        except Exception as e:      # noqa: BLE001
            cfg["run_manifest"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        cfg["run_manifest"] = {"error": "run_manifest 不可导入"}
    return cfg


# =============================================================================
# 读取 + 核对
# =============================================================================
def load_frozen_config(path: str | Path) -> Dict[str, Any]:
    """读取并**校验 schema**。下游(如 semgate_auto_report.py)统一走这里,
    不各自 json.load —— 冻结配置的 schema 也只该有一个权威。"""
    p = Path(path)
    if not p.is_file():
        raise SubmissionError(f"冻结配置不存在: {p}")
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:      # noqa: BLE001
        raise SubmissionError(f"冻结配置不是合法 JSON({p}): {e}") from e
    if not isinstance(cfg, dict):
        raise SubmissionError(f"冻结配置顶层应为对象,实得 {type(cfg).__name__}")
    if cfg.get("schema") != FROZEN_SCHEMA:
        raise SubmissionError(
            f"schema 不符:期望 {FROZEN_SCHEMA!r},实得 {cfg.get('schema')!r}")
    if cfg.get("schema_version") != FROZEN_SCHEMA_VERSION:
        raise SubmissionError(
            f"schema_version 不符:期望 {FROZEN_SCHEMA_VERSION!r},实得 "
            f"{cfg.get('schema_version')!r} —— 版本不同的字段语义可能已变,"
            f"拒绝按新代码去核对旧配置")
    fields = cfg.get("fields")
    if not isinstance(fields, dict) or not fields:
        raise SubmissionError("冻结配置缺 fields(或为空)—— 无可核对项")
    for k, v in fields.items():
        missing = [x for x in ("value", "severity", "verifiable")
                   if x not in (v if isinstance(v, dict) else {})]
        if missing:
            raise SubmissionError(f"字段 {k} 缺 {missing} —— 结构不完整,拒绝核对")
    return cfg


def frozen_threshold(cfg: Dict[str, Any]) -> float:
    """从冻结配置取回**位精确**的阈值(优先 hex,回退 repr)。"""
    fl = cfg["fields"]
    hexv = fl.get("decision.threshold_hex", {}).get("value")
    if isinstance(hexv, str) and hexv:
        return float.fromhex(hexv)
    reprv = fl.get("decision.threshold_repr", {}).get("value")
    if isinstance(reprv, str) and reprv:
        return float(reprv)
    raise SubmissionError("冻结配置里没有阈值(threshold_hex/repr 都缺)")


def verify(cfg: Dict[str, Any], *, dataset: Optional[str] = None,
           strict_env: bool = False) -> Dict[str, Any]:
    """逐项三态核对。返回 {status, rows, counts, ...},不在这里决定退出码。"""
    ds = dataset or cfg.get("dataset") or DEFAULT_DATASET
    cur = collect_inputs(ds)
    frozen = cfg["fields"]

    # 字段集合本身也要核对 —— 少一项就是少核对一项,不能当"没变"。
    # 复用 submission_contract 的 ID 集规则(精确相等),不自己写第二套。
    set_violations = check_id_set(cur.keys(), frozen.keys(), what="判据输入字段集")

    rows: List[Dict[str, Any]] = []
    for name in sorted(set(frozen) | set(cur)):
        fz, cu = frozen.get(name), cur.get(name)
        sev = (fz or cu or {}).get("severity", "critical")
        if strict_env and sev == "advisory":
            sev = "critical"
        if fz is None:
            rows.append({"field": name, "severity": sev, "status": "unverifiable",
                         "detail": "当前环境有此项,但冻结配置里没有 —— 无从核对"
                                   "(旧冻结配置?重新冻结)",
                         "frozen": None, "current": cu["value"]})
            continue
        if cu is None:
            rows.append({"field": name, "severity": sev, "status": "unverifiable",
                         "detail": "冻结配置有此项,但当前代码不再产出它 —— "
                                   "判据结构变了,无从核对",
                         "frozen": fz["value"], "current": None})
            continue
        if not fz["verifiable"]:
            # 关键:冻结时就没记上的项,永远不是 match。
            # 即使现在能算出一个值,也**证明不了**当时用的就是它。
            if cu["verifiable"]:
                rows.append({
                    "field": name, "severity": sev, "status": "unverifiable",
                    "detail": f"冻结时未能记录(原因: {fz.get('unverifiable_reason')});"
                              f"当前环境可算出值 —— 但这**证明不了当时用的就是它**,"
                              f"人工确认后请重新冻结",
                    "frozen": None, "current": _short(cu["value"])})
            else:
                rows.append({
                    "field": name, "severity": sev, "status": "unverifiable",
                    # 措辞与上一分支一致:冻结时没记上,就**证明不了当时用的是什么**,
                    # 现在能不能算出来都不改变这一点。两分支同一理由 → 同一说法,
                    # 否则单测只能覆盖恰好走到的那一支(依赖本机有没有权重)。
                    "detail": f"冻结时未能记录(原因: {fz.get('unverifiable_reason')}),"
                              f"当前环境也取不到({cu.get('unverifiable_reason')})"
                              f" —— 无论如何都**证明不了当时用的就是它**",
                    "frozen": None, "current": None})
            continue
        if not cu["verifiable"]:
            rows.append({
                "field": name, "severity": sev, "status": "unverifiable",
                "detail": f"冻结配置里有值,但当前环境取不到:"
                          f"{cu.get('unverifiable_reason')}",
                "frozen": _short(fz["value"]), "current": None})
            continue
        if _sha256_json(fz["value"]) == _sha256_json(cu["value"]):
            rows.append({"field": name, "severity": sev, "status": "match",
                         "detail": None})
        else:
            rows.append({"field": name, "severity": sev, "status": "mismatch",
                         "detail": _diff_detail(name, fz["value"], cu["value"]),
                         "frozen": _short(fz["value"]),
                         "current": _short(cu["value"])})

    counts = {"match": 0, "mismatch": 0, "unverifiable": 0}
    crit = {"mismatch": [], "unverifiable": []}
    adv = {"mismatch": [], "unverifiable": []}
    for r in rows:
        counts[r["status"]] += 1
        if r["status"] in ("mismatch", "unverifiable"):
            (crit if r["severity"] == "critical" else adv)[r["status"]].append(
                r["field"])
    return {"dataset": ds, "rows": rows, "counts": counts,
            "field_set_violations": set_violations,
            "critical": crit, "advisory": adv, "strict_env": bool(strict_env)}


def _short(v: Any, n: int = 220) -> Any:
    """长值(权重清单)在报告里只留摘要 —— 但摘要是内容哈希,不是截断的字符串。"""
    if isinstance(v, (list, dict)) and len(json.dumps(v, ensure_ascii=False)) > n:
        return {"_sha256_of_value": _sha256_json(v),
                "_len": len(v) if isinstance(v, (list, dict)) else None}
    return v


def _diff_detail(name: str, fz: Any, cu: Any) -> str:
    if isinstance(fz, list) and isinstance(cu, list) and fz and isinstance(fz[0], dict) \
            and "name" in (fz[0] or {}):
        fmap = {d.get("name"): d.get("sha256") for d in fz}
        cmap = {d.get("name"): d.get("sha256") for d in cu}
        added = sorted(set(cmap) - set(fmap))
        removed = sorted(set(fmap) - set(cmap))
        changed = sorted(k for k in set(fmap) & set(cmap) if fmap[k] != cmap[k])
        return (f"文件级差异 —— 内容变化 {changed[:6]}{'…' if len(changed) > 6 else ''};"
                f"新增 {added[:4]};缺失 {removed[:4]}")
    if isinstance(fz, dict) and isinstance(cu, dict):
        keys = sorted(set(fz) | set(cu))
        d = [f"{k}: {fz.get(k)!r} → {cu.get(k)!r}" for k in keys
             if fz.get(k) != cu.get(k)]
        return "; ".join(d[:8]) or "结构相同但序列化不同"
    return f"{fz!r} → {cu!r}"


def verdict_exit_code(res: Dict[str, Any], *, allow_unverifiable: bool) -> int:
    if res["field_set_violations"]:
        return EXIT_MISMATCH
    if res["critical"]["mismatch"]:
        return EXIT_MISMATCH
    if res["strict_env"] and res["advisory"]["mismatch"]:
        return EXIT_MISMATCH
    if res["critical"]["unverifiable"] and not allow_unverifiable:
        return EXIT_UNVERIFIABLE
    return EXIT_OK


def print_verify(res: Dict[str, Any], *, allow_unverifiable: bool) -> int:
    code = verdict_exit_code(res, allow_unverifiable=allow_unverifiable)
    print(f"语义门判据输入核对(dataset={res['dataset']},"
          f"strict_env={res['strict_env']})")
    for v in res["field_set_violations"]:
        print(f"  [字段集] {v}")
    order = {"mismatch": 0, "unverifiable": 1, "match": 2}
    for r in sorted(res["rows"], key=lambda x: (order[x["status"]], x["field"])):
        if r["status"] == "match":
            continue
        tag = "变了  " if r["status"] == "mismatch" else "无法核对"
        print(f"  [{tag}][{r['severity']:<8}] {r['field']}: {r['detail']}")
    c = res["counts"]
    print(f"  小计: match={c['match']} mismatch={c['mismatch']} "
          f"unverifiable={c['unverifiable']}")
    if code == EXIT_OK and c["unverifiable"] and allow_unverifiable:
        print("  ⚠️ verification_degraded=true —— 用了 --allow-unverifiable 逃生口,"
              "上面 无法核对 的项**没有被证明一致**;datasheet 必须照写。")
    print({EXIT_OK: "结论: 一致(可放行)",
           EXIT_MISMATCH: "结论: 判据输入已漂移 —— 冻结失效,不得用当前环境的"
                          "语义门数字",
           EXIT_UNVERIFIABLE: "结论: 存在无法核对项 —— 拒绝放行(要放行须显式 "
                              "--allow-unverifiable 并留痕)"}[code])
    return code


# =============================================================================
# 自检(不依赖任何外部产物 / 不依赖另一 agent 的文件)
# =============================================================================
def _fake_model_dir(base: Path, payload: bytes = b"fake-clip-weights") -> Path:
    """造一个「本机目录形态」的假模型 —— 只为让 clip.weight_files 这一条走上
    **可核对**分支,从而真跑出 match / mismatch 两种结局。不加载任何模型。"""
    d = base / "fake_clip"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text('{"model_type": "clip"}', encoding="utf-8")
    (d / "preprocessor_config.json").write_text(
        '{"do_center_crop": true, "size": 224}', encoding="utf-8")
    (d / "model.safetensors").write_bytes(payload)
    return d


def _run_self(argv: List[str], env: Dict[str, str]) -> Tuple[int, str]:
    e = dict(os.environ)
    e.update(env)
    p = subprocess.run([sys.executable, str(Path(__file__).resolve())] + argv,
                       capture_output=True, text=True, env=e)
    return p.returncode, p.stdout + p.stderr


def self_test() -> int:
    print("=" * 72)
    print("semgate_frozen_inputs --self-test")
    print("=" * 72)
    fails: List[str] = []

    def ck(name: str, ok: bool, extra: str = ""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
        if not ok:
            fails.append(name)

    tmp = Path(tempfile.mkdtemp(prefix="semgate_frozen_selftest_"))
    try:
        # ---- 0) 阈值全精度往返 ----
        cfg0 = build_frozen_config(DEFAULT_DATASET)
        p0 = tmp / "cfg_env.json"
        p0.write_text(json.dumps(cfg0, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        back = frozen_threshold(load_frozen_config(p0))
        ck("阈值 JSON 往返位精确", back == FROZEN_P0_THRESHOLD,
           f"{back!r} vs {FROZEN_P0_THRESHOLD!r}")
        ck("阈值未被截断成 0.28", repr(back) == "0.2777777777777778", repr(back))

        # ---- 1) 权重不可得 → 非零(fail-closed),且逃生口能放行 ----
        rc_a, out_a = _run_self(["--verify", str(p0)], {})
        # 本机没有 CLIP 权重时应为 EXIT_UNVERIFIABLE;若某天缓存里有了,则为 0。
        weights_present = (cfg0["fields"]["clip.weight_files"]["verifiable"])
        if weights_present:
            ck("权重可得 → 核对一致 exit 0", rc_a == EXIT_OK, f"rc={rc_a}")
        else:
            ck("权重不可得 → 非零退出(不静默通过)",
               rc_a == EXIT_UNVERIFIABLE, f"rc={rc_a}")
            ck("非零时逐条列出无法核对项",
               "clip.weight_files" in out_a and "无法核对" in out_a)
            rc_b, out_b = _run_self(
                ["--verify", str(p0), "--allow-unverifiable"], {})
            ck("逃生口 → exit 0", rc_b == EXIT_OK, f"rc={rc_b}")
            ck("逃生口留痕 verification_degraded",
               "verification_degraded=true" in out_b)

        # ---- 2) 假本机模型:构造出真正「三态全覆盖」的场景 ----
        mdir = _fake_model_dir(tmp)
        env_m = {"RSWM_SEMGATE_MODEL": str(mdir)}
        p1 = tmp / "cfg_fake.json"
        rc, out = _run_self(["--freeze", "-o", str(p1)], env_m)
        ck("冻结(假模型)成功", rc == EXIT_OK and p1.is_file(), f"rc={rc}")
        cfg1 = load_frozen_config(p1)
        ck("假模型权重 SHA 已冻结(非 null)",
           cfg1["fields"]["clip.weight_files"]["value"] is not None)
        ck("本机目录无 revision → 如实记 null + reason",
           cfg1["fields"]["clip.revision"]["value"] is None
           and cfg1["fields"]["clip.revision"]["unverifiable_reason"] is not None)
        rc, out = _run_self(["--verify", str(p1)], env_m)
        # 本机目录没有 revision 可核对 → 即使权重逐字节一致也**不该** exit 0。
        # 这正是三态设计的意义:一致 ≠ 全部核对过。
        ck("① 权重一致但 revision 不可核对 → 仍非零",
           rc == EXIT_UNVERIFIABLE, f"rc={rc}")
        ck("① 且不误报 mismatch", "变了" not in out)
        rc, out = _run_self(["--verify", str(p1), "--allow-unverifiable"], env_m)
        ck("① 显式放行不可核对项 → exit 0", rc == EXIT_OK, f"rc={rc}")
        ck("① 放行时留痕", "verification_degraded=true" in out)

        # ---- 3) 改权重内容 → mismatch 且指名 ----
        _fake_model_dir(tmp, payload=b"fake-clip-weights-TAMPERED")
        rc, out = _run_self(["--verify", str(p1), "--allow-unverifiable"], env_m)
        ck("② 权重被改 → 非零", rc == EXIT_MISMATCH, f"rc={rc}")
        ck("② 指出是 clip.weight_files/snapshot",
           "clip.weight_files" in out or "clip.snapshot_sha256" in out)
        _fake_model_dir(tmp)      # 复原

        # ---- 4) 改阈值(env)→ mismatch 且指名 thresholds_resolved ----
        env_thr = dict(env_m)
        env_thr["RSWM_SEMGATE_MIN_FRAC_WATER"] = "0.5"
        rc, out = _run_self(["--verify", str(p1), "--allow-unverifiable"], env_thr)
        ck("③ env 改阈值 → 非零", rc == EXIT_MISMATCH, f"rc={rc}")
        ck("③ 指出 decision.thresholds_resolved",
           "decision.thresholds_resolved" in out)
        ck("③ 也抓到 env.verdict_overrides", "env.verdict_overrides" in out)

        # ---- 5) 关门(RSWM_SEMGATE=0)必须被抓 ----
        env_off = dict(env_m)
        env_off["RSWM_SEMGATE"] = "0"
        rc, out = _run_self(["--verify", str(p1), "--allow-unverifiable"], env_off)
        ck("④ RSWM_SEMGATE=0 → 非零", rc == EXIT_MISMATCH, f"rc={rc}")
        ck("④ 指出 decision.gate_enabled", "decision.gate_enabled" in out)

        # ---- 6) 篡改冻结文件里的 taxonomy sha → mismatch ----
        raw = json.loads(p1.read_text(encoding="utf-8"))
        raw["fields"]["taxonomy.file_sha256"]["value"] = "0" * 64
        p2 = tmp / "cfg_tampered.json"
        p2.write_text(json.dumps(raw, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        rc, out = _run_self(["--verify", str(p2), "--allow-unverifiable"], env_m)
        ck("⑤ taxonomy SHA 不符 → 非零 + 指名", rc == EXIT_MISMATCH
           and "taxonomy.file_sha256" in out, f"rc={rc}")

        # ---- 7) 冻结时 null,现在可得 → 仍算 unverifiable(不许洗成 match) ----
        raw2 = json.loads(p1.read_text(encoding="utf-8"))
        raw2["fields"]["clip.weight_files"] = {
            "value": None, "severity": "critical", "verifiable": False,
            "unverifiable_reason": "冻结时权重不在本机", "why": "x"}
        p3 = tmp / "cfg_nullweights.json"
        p3.write_text(json.dumps(raw2, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        res = verify(load_frozen_config(p3), strict_env=False)
        row = next(r for r in res["rows"] if r["field"] == "clip.weight_files")
        ck("⑥ null(冻结) → unverifiable,不是 match",
           row["status"] == "unverifiable", row["status"])
        # 这条**曾经依环境而变**:断言的是"当前可算出值"那一支的措辞,
        # 而本机没有 CLIP 权重时走的是"两边都取不到"那一支 → 假红。
        # 现在两支措辞统一,断言对任何环境都成立。
        ck("⑥ 且理由说明'证明不了当时用的就是它'(两支措辞已统一)",
           "证明不了当时用的就是它" in (row["detail"] or ""), (row["detail"] or "")[:70])
        ck("⑥ 无论当前环境可否取得,结论都不是 match",
           row["status"] != "match")

        # ---- 8) schema 守卫 ----
        p4 = tmp / "cfg_badschema.json"
        p4.write_text('{"schema": "nope", "schema_version": "1.0", "fields": {}}',
                      encoding="utf-8")
        try:
            load_frozen_config(p4)
            ck("⑦ 坏 schema 被拒", False, "未抛异常")
        except SubmissionError as e:
            ck("⑦ 坏 schema 被拒", "schema 不符" in str(e), str(e)[:60])
        p5 = tmp / "cfg_nofields.json"
        p5.write_text(json.dumps({"schema": FROZEN_SCHEMA,
                                  "schema_version": FROZEN_SCHEMA_VERSION,
                                  "fields": {}}), encoding="utf-8")
        try:
            load_frozen_config(p5)
            ck("⑧ 空 fields 被拒", False, "未抛异常")
        except SubmissionError:
            ck("⑧ 空 fields 被拒", True)

        # ---- 9) advisory 默认不致命,--strict-env 提为致命 ----
        raw3 = json.loads(p1.read_text(encoding="utf-8"))
        raw3["fields"]["env.runtime_versions"]["value"]["numpy"] = "0.0.0-fake"
        p6 = tmp / "cfg_advisory.json"
        p6.write_text(json.dumps(raw3, indent=2, ensure_ascii=False),
                      encoding="utf-8")
        rc, out = _run_self(["--verify", str(p6), "--allow-unverifiable"], env_m)
        ck("⑨ advisory 漂移默认不致命", rc == EXIT_OK, f"rc={rc}")
        ck("⑨ 但仍被打印出来", "env.runtime_versions" in out)
        rc, out = _run_self(
            ["--verify", str(p6), "--allow-unverifiable", "--strict-env"], env_m)
        ck("⑩ --strict-env → 非零", rc == EXIT_MISMATCH, f"rc={rc}")

        # ---- 10) prompts_sha256 真的覆盖 .py 里的短语表 ----
        import lora_remotesense.semantic_gate as SG
        h_before = _sha256_json(list(build_prompt_bank(DEFAULT_DATASET)["prompts"]))
        old = SG._PHRASE_OVERRIDES.get("river")
        SG._PHRASE_OVERRIDES["river"] = "a river of lava"
        h_after = _sha256_json(list(build_prompt_bank(DEFAULT_DATASET)["prompts"]))
        if old is None:
            SG._PHRASE_OVERRIDES.pop("river", None)
        else:
            SG._PHRASE_OVERRIDES["river"] = old
        ck("⑪ 改 .py 短语表 → prompts_sha256 变(taxonomy.json 的 SHA 不会变)",
           h_before != h_after)

        # ---- 11) 预处理指纹真的能抓 edge→constant ----
        fp_edge = _preprocess_fingerprint()
        _orig = SemanticGate._pad_square

        def _zero_pad(img):
            h, w = img.shape[:2]
            if h == w:
                return img
            side = max(h, w)
            top, left = (side - h) // 2, (side - w) // 2
            return np.pad(img, ((top, side - h - top), (left, side - w - left),
                                (0, 0)), mode="constant")
        SemanticGate._pad_square = staticmethod(_zero_pad)
        try:
            fp_zero = _preprocess_fingerprint()
        finally:
            SemanticGate._pad_square = staticmethod(_orig)
        ck("⑫ 预处理指纹能区分 edge-pad 与 zero-pad",
           fp_edge["out_sha256"] != fp_zero["out_sha256"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 72)
    if fails:
        print(f"自检失败 {len(fails)} 项: {fails}")
        return 1
    print("自检全部通过")
    return 0


# =============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="语义门判据输入的全量冻结与三态核对(P0-7)")
    ap.add_argument("--freeze", action="store_true",
                    help="冻结当前环境(默认动作)")
    ap.add_argument("--verify", metavar="FROZEN_JSON",
                    help="核对当前环境与该冻结配置;不符非零退出")
    ap.add_argument("-o", "--output", default=None,
                    help=f"冻结输出路径(默认 ./{DEFAULT_OUTPUT_NAME})")
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="prompt 库的数据集(生产路径必须显式;union 会污染正类)")
    ap.add_argument("--allow-unverifiable", action="store_true",
                    help="逃生口:允许「无法核对」项存在而仍 exit 0(输出留痕)")
    ap.add_argument("--strict-env", action="store_true",
                    help="把 advisory(运行时版本等)提为 critical")
    ap.add_argument("--report-json", default=None, help="把核对结果写成 JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="自检:不依赖任何外部产物")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    if args.verify:
        try:
            cfg = load_frozen_config(args.verify)
        except SubmissionError as e:
            print(f"[FATAL] {e}")
            return EXIT_USAGE
        res = verify(cfg, dataset=args.dataset, strict_env=args.strict_env)
        code = print_verify(res, allow_unverifiable=args.allow_unverifiable)
        if args.report_json:
            Path(args.report_json).write_text(json.dumps(
                {"verify": res, "exit_code": code,
                 "allow_unverifiable_used": bool(args.allow_unverifiable),
                 "verification_degraded": bool(
                     args.allow_unverifiable and res["counts"]["unverifiable"])},
                indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"核对结果 → {args.report_json}")
        return code

    # 默认 = 冻结
    out = Path(args.output) if args.output else Path(DEFAULT_OUTPUT_NAME)
    cfg = build_frozen_config(args.dataset)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    n_un = cfg["n_unverifiable_at_freeze"]
    print(f"冻结 {len(cfg['fields'])} 项判据输入 → {out}")
    print(f"  dataset={cfg['dataset']}  阈值={cfg['fields']['decision.threshold_repr']['value']}"
          f"  prompts_sha={cfg['fields']['prompt_bank.prompts_sha256']['value'][:16]}…")
    if n_un:
        print(f"  ⚠️ 有 {n_un} 项冻结时无法取得(如实记 null,**没有编哈希**):")
        for k, v in cfg["fields"].items():
            if not v["verifiable"]:
                print(f"     - {k}: {v['unverifiable_reason']}")
        print("  这些项在 --verify 时永远是「无法核对」(非零退出)—— "
              "拿到权重后请重新冻结。")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
