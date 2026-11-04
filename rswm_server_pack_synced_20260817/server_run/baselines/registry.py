#!/usr/bin/env python3
# =============================================================================
# registry.py — RSWM 统一 baseline 适配器 + 注册表
#
# 目的:让「诊断基线」与「外部生成式基线」吐出 **完全相同** 的记录格式,
# 这样同一套评估脚本(trajectory_metrics.score_trajectory / metrics.py)可以
# 无差别打分,论文表格里每一行都可追溯到同一协议。
#
# 设计约束(务必遵守):
#   1. 本模块 import 时 **不碰 numpy**。numpy/cv2 只在 run()/validate_record()
#      内部惰性 import。这样 `python3 registry.py` 的冒烟自检可以在没有 numpy 的
#      机器上跑通(只测接线,不测数值)。
#   2. 四个诊断基线一律 **包装** lora_remotesense/new/diagnostic_baselines.py 的
#      现成函数,绝不重新实现。特别是 random_direction 的 dir_index/step_px 推导
#      只存在于 run_baselines() 内部(diagnostic_baselines.py:423-436),必须走它。
#   3. 失败必须响亮:缺 key / 形状不对 / dtype 不对 → 抛异常,绝不静默兜底。
#
# 约定备忘(与仓库既有代码逐条对齐,勿凭记忆改):
#   - waypoints [K+1,2] = (cy, cx) 像素坐标 = **相机/窗口中心** 路径
#     (build_multi_direction_benchmark_v5.py:447 `return (cy, cx)`)。
#   - 相机方向角 = degrees(atan2(-dy, dx)) % 360;east=0/north=90/west=180/
#     south=270 (lora_training/camera_trajectory.py:17-21)。
#   - **内容光流 = 相机运动取反**(窗口往东 → 画面内容往西)。
#     content 方向 = camera 方向 + 180。route_metadata.direction_word 记的是
#     **相机** 方向词(= builder prompt 里的词),不是内容流向,别自己再翻 180。
#   - K 段折线有 K-1 个转弯;prompt 里的 "{K}-segment trajectory" 是段数
#     (build_dataset_from_benchmark.py:605-608)。
#   - segment_angles [K] 是 **相机** 段方向角(度)。
# =============================================================================
from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# ---- 仓库根 = server_run/baselines/registry.py 往上两层 ----
ROOT = Path(__file__).resolve().parents[2]
_NEW = ROOT / "lora_remotesense" / "new"

ADAPTER_VERSION = "rswm-baseline-adapter-1"


class SampleUnavailable(RuntimeError):
    """[v2.6] 单样本级失败:该样本此刻拿不出【可信的】生成结果。

    与 NotImplementedError(方法整体没实现)与 KeyError/ValueError(调用方
    传参错误)语义不同:方法可跑,只是这个样本的产物缺失/不可信(如
    MVSplat360 单样本推理失败被黑帧填充)。run_baseline.py --skip-failed 会
    按样本捕获它、显式计数并按 --min-success 门槛判决 —— 绝不静默平均黑帧。
    """

# ---- 契约常量 ----
KINDS = ("diagnostic", "external_generative")
STATUSES = ("implemented", "stub", "unavailable")
CONDITION_TYPES = ("text_only", "text_plucker", "none")
DIRECTION_WORDS = ("east", "north", "west", "south")

RECORD_KEYS = (
    "sample_id",
    "generated_frames",
    "source_frames",
    "target_frames",
    "route_metadata",
    "runtime_metadata",
)
ROUTE_KEYS = (
    "prompt",
    "waypoints",
    "segment_angles",
    "num_segments",
    "direction_word",
    "condition_type",
)
RUNTIME_KEYS = (
    "wall_seconds",
    "gen_seed",
    "resolution",
    "num_frames",
    "device",
    "method_version",
)

# status 语义(README_BASELINES.md 同款定义):
#   implemented  : run() 此时此刻就能产出合规 record。
#   stub         : 适配器在,方法本身还没写(仓库里没有可跑的实现)。
#   unavailable  : 仓库里 **有** 实现代码,但缺依赖(repo/权重/env)跑不了。


# =============================================================================
# 适配器基类
# =============================================================================

class BaselineAdapter:
    """所有 baseline 的统一接口。

    属性:
        name         : 注册名(唯一)。
        kind         : 'diagnostic' | 'external_generative'。
        requires_gpu : 是否需要 GPU。
        status       : 'implemented' | 'stub' | 'unavailable'。
        upper_bound  : True 表示这是 **指标上界/天花板**,不是可部署方法
                       (gt_oracle 专用)。
    run(sample) -> dict,键 **恰好** 为 RECORD_KEYS,见 validate_record。
    """

    name: str = ""
    kind: str = "diagnostic"
    requires_gpu: bool = False
    status: str = "stub"
    upper_bound: bool = False
    description: str = ""

    def run(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.name}.run() 未实现")

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (f"<{type(self).__name__} name={self.name!r} kind={self.kind!r} "
                f"status={self.status!r} gpu={self.requires_gpu}>")


# =============================================================================
# 注册表
# =============================================================================

_REGISTRY: "Dict[str, BaselineAdapter]" = {}


def register(adapter: BaselineAdapter) -> BaselineAdapter:
    """注册一个适配器;重名/字段非法直接抛。"""
    if not isinstance(adapter, BaselineAdapter):
        raise TypeError(f"register: 需要 BaselineAdapter 实例, 得到 {type(adapter)!r}")
    if not adapter.name:
        raise ValueError("register: adapter.name 不能为空")
    if adapter.name in _REGISTRY:
        raise ValueError(f"register: 重复注册 {adapter.name!r}")
    if adapter.kind not in KINDS:
        raise ValueError(f"register: {adapter.name} kind={adapter.kind!r} 不在 {KINDS}")
    if adapter.status not in STATUSES:
        raise ValueError(f"register: {adapter.name} status={adapter.status!r} 不在 {STATUSES}")
    if not isinstance(adapter.requires_gpu, bool):
        raise TypeError(f"register: {adapter.name} requires_gpu 必须是 bool")
    if not callable(getattr(adapter, "run", None)):
        raise TypeError(f"register: {adapter.name} 没有可调用的 run()")
    _REGISTRY[adapter.name] = adapter
    return adapter


def get(name: str) -> BaselineAdapter:
    """按名取适配器;不存在直接抛(附可选列表)。"""
    if name not in _REGISTRY:
        raise KeyError(f"未知 baseline {name!r};可选: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_baselines(kind: Optional[str] = None) -> List[BaselineAdapter]:
    """按注册名排序列出适配器;kind 可选过滤。"""
    if kind is not None and kind not in KINDS:
        raise ValueError(f"list_baselines: kind={kind!r} 不在 {KINDS}")
    out = [a for a in _REGISTRY.values() if kind is None or a.kind == kind]
    return sorted(out, key=lambda a: a.name)


# =============================================================================
# record 校验(失败必抛,绝不静默)
# =============================================================================

def _check_exact_keys(d: Dict[str, Any], want, what: str) -> None:
    if not isinstance(d, dict):
        raise TypeError(f"validate_record: {what} 必须是 dict, 得到 {type(d)!r}")
    have = set(d)
    missing = sorted(set(want) - have)
    extra = sorted(have - set(want))
    if missing:
        raise KeyError(f"validate_record: {what} 缺 key: {missing}")
    if extra:
        raise KeyError(f"validate_record: {what} 多出 key: {extra}(契约要求恰好 {list(want)})")


def _check_frames(np, arr, what: str, name: str):
    """(N,H,W,3) uint8 检查。"""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"validate_record[{name}]: {what} 必须是 np.ndarray, 得到 {type(arr)!r}")
    if arr.ndim != 4 or arr.shape[3] != 3:
        raise ValueError(f"validate_record[{name}]: {what} 形状必须是 (T,H,W,3), 得到 {arr.shape}")
    if arr.shape[0] < 1:
        raise ValueError(f"validate_record[{name}]: {what} 帧数为 0")
    if arr.dtype != np.uint8:
        raise TypeError(f"validate_record[{name}]: {what} dtype 必须是 uint8, 得到 {arr.dtype}")
    return arr


def validate_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """校验一条 baseline 记录;任何缺 key / 形状 / dtype 问题都抛异常。

    校验顺序刻意设计:先做不依赖 numpy 的 key/类型/枚举检查(所以没有 numpy 的
    机器也能测接线),再惰性 import numpy 做形状/dtype 检查。numpy 缺席时 **抛
    RuntimeError**,绝不半途 return(那等于静默放行)。

    通过则原样返回 record。
    """
    # ---- 1. 不依赖 numpy 的部分 ----
    _check_exact_keys(record, RECORD_KEYS, "record")

    sid = record["sample_id"]
    if not isinstance(sid, str) or not sid:
        raise ValueError(f"validate_record: sample_id 必须是非空 str, 得到 {sid!r}")

    route = record["route_metadata"]
    _check_exact_keys(route, ROUTE_KEYS, f"route_metadata[{sid}]")
    if not isinstance(route["prompt"], str) or not route["prompt"]:
        raise ValueError(f"validate_record[{sid}]: route_metadata.prompt 必须是非空 str")
    if route["condition_type"] not in CONDITION_TYPES:
        raise ValueError(f"validate_record[{sid}]: condition_type={route['condition_type']!r} "
                         f"不在 {CONDITION_TYPES}")
    if route["direction_word"] not in DIRECTION_WORDS:
        raise ValueError(f"validate_record[{sid}]: direction_word={route['direction_word']!r} "
                         f"不在 {DIRECTION_WORDS}")
    if not isinstance(route["num_segments"], int) or isinstance(route["num_segments"], bool):
        raise TypeError(f"validate_record[{sid}]: num_segments 必须是 int")
    if route["num_segments"] < 1:
        raise ValueError(f"validate_record[{sid}]: num_segments 必须 >=1, 得到 {route['num_segments']}")

    rt = record["runtime_metadata"]
    _check_exact_keys(rt, RUNTIME_KEYS, f"runtime_metadata[{sid}]")
    if not isinstance(rt["wall_seconds"], float) or rt["wall_seconds"] < 0:
        raise ValueError(f"validate_record[{sid}]: wall_seconds 必须是 >=0 的 float, 得到 {rt['wall_seconds']!r}")
    for k in ("gen_seed", "num_frames"):
        if not isinstance(rt[k], int) or isinstance(rt[k], bool):
            raise TypeError(f"validate_record[{sid}]: runtime_metadata.{k} 必须是 int, 得到 {rt[k]!r}")
    for k in ("device", "method_version"):
        if not isinstance(rt[k], str) or not rt[k]:
            raise ValueError(f"validate_record[{sid}]: runtime_metadata.{k} 必须是非空 str")
    res = rt["resolution"]
    if (not isinstance(res, (list, tuple)) or len(res) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in res)):
        raise ValueError(f"validate_record[{sid}]: resolution 必须是 [H,W] 两个 int, 得到 {res!r}")

    # ---- 2. 依赖 numpy 的形状/dtype 部分 ----
    try:
        import numpy as np
    except ImportError as e:  # 缺 numpy 时必须抛,不能当作「通过」
        raise RuntimeError(
            "validate_record: 形状/dtype 检查需要 numpy,但当前环境没有 numpy。"
            "请在装了 numpy 的环境(rswm conda env)里跑。"
        ) from e

    gen = _check_frames(np, record["generated_frames"], "generated_frames", sid)
    tgt = _check_frames(np, record["target_frames"], "target_frames", sid)
    src = _check_frames(np, record["source_frames"], "source_frames", sid)

    if gen.shape != tgt.shape:
        raise ValueError(f"validate_record[{sid}]: generated_frames {gen.shape} 与 "
                         f"target_frames {tgt.shape} 形状不一致(共享协议要求同分辨率同帧数)")
    if src.shape[1:] != gen.shape[1:]:
        raise ValueError(f"validate_record[{sid}]: source_frames {src.shape} 与 "
                         f"generated_frames {gen.shape} 的 (H,W,3) 不一致")

    wp = np.asarray(route["waypoints"])
    if wp.ndim != 2 or wp.shape[1] != 2:
        raise ValueError(f"validate_record[{sid}]: waypoints 必须是 [K+1,2] 的 (cy,cx), 得到 {wp.shape}")
    if wp.shape[0] < 2:
        raise ValueError(f"validate_record[{sid}]: waypoints 至少 2 个点, 得到 {wp.shape[0]}")
    seg = np.asarray(route["segment_angles"]).ravel()
    K = int(wp.shape[0]) - 1
    if seg.shape[0] != K:
        raise ValueError(f"validate_record[{sid}]: segment_angles 长度 {seg.shape[0]} != K={K} "
                         f"(waypoints [K+1,2] 有 K+1={wp.shape[0]} 个点)")
    if route["num_segments"] != K:
        raise ValueError(f"validate_record[{sid}]: num_segments={route['num_segments']} != K={K}")

    if tuple(res) != (int(gen.shape[1]), int(gen.shape[2])):
        raise ValueError(f"validate_record[{sid}]: runtime_metadata.resolution={tuple(res)} 与 "
                         f"generated_frames 实际 (H,W)={(int(gen.shape[1]), int(gen.shape[2]))} 不符")
    if rt["num_frames"] != int(gen.shape[0]):
        raise ValueError(f"validate_record[{sid}]: runtime_metadata.num_frames={rt['num_frames']} != "
                         f"实际帧数 {int(gen.shape[0])}")
    return record


# =============================================================================
# 工具:惰性 import diagnostic_baselines / 取指纹 / 取方向词
# =============================================================================

def _import_diagnostic_baselines():
    """惰性 import lora_remotesense/new/diagnostic_baselines.py(需要 numpy)。"""
    path = _NEW / "diagnostic_baselines.py"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 {path} —— 诊断基线适配器依赖它(不重复实现)")
    if str(_NEW) not in sys.path:
        sys.path.insert(0, str(_NEW))
    import diagnostic_baselines as DB  # noqa: E402
    return DB


def _source_fingerprint() -> str:
    """diagnostic_baselines.py 的 sha256 前 12 位,写进 method_version 做溯源。

    纯 stdlib(hashlib),没有 numpy 也能算。
    """
    path = _NEW / "diagnostic_baselines.py"
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


_DIR_RE = re.compile(r"toward the (east|north|west|south)\b", re.IGNORECASE)


def direction_word(prompt: str, waypoints) -> str:
    """取 **相机** 净方向词 (east/north/west/south)。

    优先从 prompt 里解析 —— builder 写的就是这个词
    (build_dataset_from_benchmark.py:602-609 "... toward the {DIR}."),
    它才是真正喂给模型的条件,不能自己另算一套。
    prompt 里没有(builder 在缺 waypoints 时会退化成无方向 prompt)才按
    builder:583-590 同款 45 度分箱从 waypoints 现推。

    注意:这是 **相机/窗口** 方向。画面内容流向是它 +180(内容光流 = 相机运动取反)。
    """
    m = _DIR_RE.search(prompt or "")
    if m:
        return m.group(1).lower()
    if waypoints is None or len(waypoints) < 2:
        raise ValueError("direction_word: prompt 里没有方向词, waypoints 也不足 2 点, 无法确定方向")
    import math
    dy = float(waypoints[-1][0]) - float(waypoints[0][0])   # cy
    dx = float(waypoints[-1][1]) - float(waypoints[0][1])   # cx
    angle = math.degrees(math.atan2(-dy, dx)) % 360.0       # camera_trajectory.py:17-21
    if angle < 45 or angle >= 315:
        return "east"
    if 45 <= angle < 135:
        return "north"
    if 135 <= angle < 225:
        return "west"
    return "south"


def _require(sample: Dict[str, Any], key: str, name: str):
    """取必填字段;缺了直接抛(共享协议靠这个兜底,不给默认值)。"""
    if key not in sample or sample[key] is None:
        raise KeyError(f"{name}.run(): sample 缺必填字段 {key!r}。"
                       f"必填: sample_id/prompt/waypoints/segment_angles/window_size/"
                       f"gt_frames/gen_seed")
    return sample[key]


# =============================================================================
# 诊断基线适配器:包装 diagnostic_baselines.run_baselines(不重复实现)
# =============================================================================

class DiagnosticAdapter(BaselineAdapter):
    """把 diagnostic_baselines.run_baselines() 的一路输出包成统一 record。

    为什么统一走 run_baselines() 而不是各自调底层函数:
      - random_direction 的 dir_index(取 GT 象限的对面)和 step_px(匹配 GT 路径
        速度)推导 **只存在于** run_baselines 内部
        (diagnostic_baselines.py:423-436)。自己再推一遍 = 重复实现 + 数字漂移。
      - run_baselines 还负责把四路统一 resize 到同一打分分辨率
        (diagnostic_baselines.py:405-411),保证与 run_trajectory_eval.py 的既有
        数字逐字节可比。
    代价:每次 run() 会把四路都合成一遍(纯 CPU 裁剪,便宜)。PBS 一个作业只跑一个
    baseline,所以实际不重复。

    condition_type 一律 'none':这四个基线 **不读 prompt、不读位姿**,它们是从静态
    场景 + GT 轨迹裁出来的合成视频,不是条件生成。
    """

    kind = "diagnostic"
    requires_gpu = False
    status = "implemented"

    def __init__(self, name: str, output_key: str, description: str,
                 upper_bound: bool = False):
        self.name = name
        # run_baselines() 的返回键;注意 random_direction_translation 的键是
        # 'random_direction'(diagnostic_baselines.py:436),与函数名不同。
        self.output_key = output_key
        self.description = description
        self.upper_bound = upper_bound

    def run(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        DB = _import_diagnostic_baselines()
        import numpy as np

        sid = _require(sample, "sample_id", self.name)
        prompt = _require(sample, "prompt", self.name)
        waypoints = _require(sample, "waypoints", self.name)
        segment_angles = _require(sample, "segment_angles", self.name)
        window_size = _require(sample, "window_size", self.name)
        gt_frames = np.asarray(_require(sample, "gt_frames", self.name))
        gen_seed = _require(sample, "gen_seed", self.name)
        if not isinstance(gen_seed, int) or isinstance(gen_seed, bool):
            raise TypeError(f"{self.name}.run(): gen_seed 必须是 int, 得到 {gen_seed!r}")

        scene = sample.get("scene_image", sample.get("full_scene"))
        T = sample.get("T", sample.get("num_frames")) or len(gt_frames)
        T = int(T)

        t0 = time.perf_counter()
        vids = DB.run_baselines(dict(
            scene_image=scene,
            source_frame=sample.get("source_frame"),
            waypoints=waypoints,
            window_size=window_size,
            T=T,
            gt_frames=gt_frames,
        ))
        wall = float(time.perf_counter() - t0)

        if self.output_key not in vids:
            raise RuntimeError(
                f"{self.name}.run(): run_baselines() 没有返回 {self.output_key!r} "
                f"(实际: {sorted(vids)})。"
                f"{'需要 scene_image/full_scene(整幅静态场景)' if self.output_key in ('deterministic_translation', 'random_direction') else '需要 gt_frames'}"
            )
        gen = vids[self.output_key]

        # 统一到 generated 的 (H,W):run_baselines 已把各路 resize 到 GT 分辨率,
        # 这里对 GT/source 再走一次同一个 resize 助手,尺寸已同则是 no-op。
        hw = (int(gen.shape[1]), int(gen.shape[2]))
        target = DB._resize_video(gt_frames, hw)
        src_frame = sample.get("source_frame")
        src_frame = gt_frames[0] if src_frame is None else np.asarray(src_frame)
        source = DB._resize_video(np.asarray(src_frame)[None, ...], hw)

        wp_list = np.asarray(waypoints, dtype=float).tolist()
        seg_list = np.asarray(segment_angles, dtype=float).ravel().tolist()

        record = {
            "sample_id": str(sid),
            "generated_frames": gen,
            "source_frames": source,
            "target_frames": target,
            "route_metadata": {
                "prompt": str(prompt),
                "waypoints": wp_list,              # [K+1,2] (cy,cx) 相机路径
                "segment_angles": seg_list,        # [K] 相机段角(度)
                "num_segments": len(seg_list),     # K 段 = K-1 个转弯
                "direction_word": direction_word(str(prompt), wp_list),
                "condition_type": "none",          # 诊断基线不吃文本/位姿条件
            },
            "runtime_metadata": {
                "wall_seconds": wall,
                "gen_seed": int(gen_seed),         # 诊断基线是确定性的,记录仅为协议对齐
                "resolution": [hw[0], hw[1]],
                "num_frames": int(gen.shape[0]),
                "device": "cpu",
                "method_version": f"{ADAPTER_VERSION}/diagnostic_baselines.py@{_source_fingerprint()}",
            },
        }
        return validate_record(record)


# ---- 注册四个诊断基线(名字 = diagnostic_baselines.py 的函数名)----

register(DiagnosticAdapter(
    name="persistence",
    output_key="persistence",
    description="重复 source frame T 次:模型「只是把输入图片端住」。预期 ~0 光流、"
                "方向不可估、只有最早几帧 SSIM 高。",
))

register(DiagnosticAdapter(
    name="deterministic_translation",
    output_key="deterministic_translation",
    description="沿 GT waypoints 裁真窗口 = 构造上界(construction-aware upper bound)。"
                "方向/路径近乎完美、SSIM 高,但它是场景查表(需要整幅静态场景),"
                "不是生成模型。",
))

register(DiagnosticAdapter(
    name="random_direction_translation",
    # run_baselines() 里这一路的键是 'random_direction'(diagnostic_baselines.py:436)
    output_key="random_direction",
    description="「画面好但方向错」对照:沿一个确定性选定的错误朝向(GT 象限的对面)"
                "平移窗口。帧很锐、光流幅度对,但角度误差大。",
))

register(DiagnosticAdapter(
    name="gt_oracle",
    output_key="gt_oracle",
    description="GT 帧原样透传 = 指标 **天花板/上界**(SSIM=1, PSNR=inf, 角度误差=量化下限)。"
                "【不是可部署方法】:它直接看了答案。只用来标定指标量程,"
                "论文里必须标注为 upper bound,绝不能当成一个「方法」参与优劣比较。",
    upper_bound=True,
))


# ---- 外部生成式基线:注册在 external_stubs.py(import 时自动 register)----
def _register_external() -> None:
    """import external_stubs 触发外部基线注册。

    单独放一个函数是为了让 import 顺序显式;external_stubs 只依赖 stdlib,
    所以没有 numpy 的机器也能列出它们。
    """
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    # 关键:直接 `python3 registry.py` 时本模块名是 '__main__'。若不先把自己别名成
    # 'registry',external_stubs 里的 `from registry import ...` 会把 registry.py
    # **再执行一遍**,生成第二个模块对象和第二个 _REGISTRY —— 外部基线注册进那一份,
    # 本进程这一份看起来「一个外部基线都没有」。冒烟自检抓到过这个 bug。
    sys.modules.setdefault("registry", sys.modules[__name__])
    try:
        from . import external_stubs  # noqa: F401
    except ImportError:
        # 脚本式运行(非 package 上下文)的正常路径;若 external_stubs 自身有真的
        # ImportError,这一句会照样抛出去,不会被吞掉。
        import external_stubs  # noqa: F401


_register_external()


def _register_result_adapters() -> None:
    """Register bridges for outputs produced by the existing experiment runners.

    These adapters only standardize saved rollouts.  They never construct or alter
    Wan/LingBot, the Video VAE, camera injection, or Plucker rays.
    """
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    sys.modules.setdefault("registry", sys.modules[__name__])
    try:
        from . import result_adapters  # noqa: F401
    except ImportError:
        import result_adapters  # noqa: F401


_register_result_adapters()


# =============================================================================
# 冒烟自检:`python3 registry.py` —— 只测接线,不测数值,无 numpy 也能跑
# =============================================================================

def _smoke() -> int:
    import json
    import os

    # [v2.6] 冒烟测的是【静态接线】,必须在确定性环境里跑:若外面 export 过
    # RSWM_MVSPLAT_EVAL_DIR,mvsplat360 的 run() 会走桥接回收路径而非
    # NotImplementedError,规则 4 会误报 —— 冒烟进程内显式清掉。
    os.environ.pop("RSWM_MVSPLAT_EVAL_DIR", None)

    print(f"[smoke] ROOT={ROOT}")
    print(f"[smoke] diagnostic_baselines.py@{_source_fingerprint()}")

    adapters = list_baselines()
    if not adapters:
        print("[FAIL] 注册表是空的")
        return 1

    print(f"\n{'name':<30} {'kind':<20} {'status':<13} {'gpu':<5} upper_bound")
    print("-" * 84)
    for a in adapters:
        print(f"{a.name:<30} {a.kind:<20} {a.status:<13} "
              f"{str(a.requires_gpu):<5} {a.upper_bound}")

    fails: List[str] = []

    # 1. 契约字段齐全 + 取值合法
    for a in adapters:
        for attr in ("name", "kind", "requires_gpu", "status"):
            if not hasattr(a, attr):
                fails.append(f"{a.name} 缺属性 {attr}")
        if a.kind not in KINDS:
            fails.append(f"{a.name} kind 非法: {a.kind}")
        if a.status not in STATUSES:
            fails.append(f"{a.name} status 非法: {a.status}")
        if not callable(getattr(a, "run", None)):
            fails.append(f"{a.name} run 不可调用")
        if get(a.name) is not a:
            fails.append(f"get({a.name!r}) 没拿回同一个对象")

    # 2. 四个诊断基线都在,且 gt_oracle 标了上界
    diag = {a.name for a in list_baselines(kind="diagnostic")}
    want_diag = {"persistence", "deterministic_translation",
                 "random_direction_translation", "gt_oracle"}
    if diag != want_diag:
        fails.append(f"诊断基线集合不符: 少 {sorted(want_diag - diag)} 多 {sorted(diag - want_diag)}")
    # upper_bound 标记必须 **恰好** 只有 gt_oracle:它是天花板,不是方法。
    # (deterministic_translation 也是「构造上界」,但它至少还合成像素;它的上界
    #  属性写在 description / README / baseline_status.json 的 note 里,不占这个
    #  「直接看了答案」专用的标记。)
    ub = {a.name for a in adapters if a.upper_bound}
    if ub != {"gt_oracle"}:
        fails.append(f"upper_bound 标记应恰好是 ['gt_oracle'],实际 {sorted(ub)}")

    # 3. get() 未知名必须抛
    try:
        get("__nope__")
        fails.append("get() 对未知名没抛")
    except KeyError:
        pass

    # 4. 外部生成式基线:必须 requires_gpu 且 run() 抛 NotImplementedError
    ext = list_baselines(kind="external_generative")
    if not ext:
        fails.append("一个外部生成式基线都没注册(至少 mvsplat360 应在)")
    for a in ext:
        # [v2.5] 规则分两路:未实现的外部基线必须抛 NotImplementedError 并点名
        # 缺什么;【已实现】的(如 image_outpainting)则要求 run() 对残缺 sample
        # 大声拒绝(KeyError/ValueError/TypeError 皆可),而不是静默出数。
        # 旧规则写死"外部基线必然是桩",v2.5 落地 outpaint 后会误报。
        if a.status == "implemented":
            try:
                a.run({"sample_id": "x"})
                fails.append(f"{a.name}.run() 对残缺 sample 没有拒绝(必须硬失败)")
            except NotImplementedError as e:  # noqa: BLE001
                fails.append(f"{a.name} 标 implemented 却抛 NotImplementedError: {e}")
            except (KeyError, ValueError, TypeError):
                pass          # 正确:缺字段被点名拒绝
            except Exception as e:  # noqa: BLE001
                fails.append(f"{a.name}.run() 抛了意外的 {type(e).__name__}: {e}")
            continue
        try:
            a.run({"sample_id": "x"})
            fails.append(f"{a.name}.run() 没抛 NotImplementedError")
        except NotImplementedError as e:
            if not str(e).strip():
                fails.append(f"{a.name}.run() 抛了空消息(必须点名缺什么依赖)")
        except Exception as e:  # noqa: BLE001
            fails.append(f"{a.name}.run() 抛了 {type(e).__name__} 而不是 NotImplementedError: {e}")

    # 5. validate_record 的「不依赖 numpy」分支必须响亮失败
    for bad, why in [
        ({}, "空 dict"),
        ({k: None for k in RECORD_KEYS}, "sample_id=None"),
    ]:
        try:
            validate_record(bad)
            fails.append(f"validate_record 对 {why} 没抛")
        except (KeyError, TypeError, ValueError):
            pass

    rec = {k: None for k in RECORD_KEYS}
    rec["sample_id"] = "s0"
    rec["route_metadata"] = {"prompt": "p", "waypoints": [[0, 0], [0, 1]],
                             "segment_angles": [0.0], "num_segments": 1,
                             "direction_word": "sideways",   # 非法
                             "condition_type": "none"}
    try:
        validate_record(rec)
        fails.append("validate_record 对非法 direction_word 没抛")
    except ValueError:
        pass
    rec["route_metadata"]["direction_word"] = "east"
    rec["route_metadata"]["condition_type"] = "text_and_vibes"   # 非法
    try:
        validate_record(rec)
        fails.append("validate_record 对非法 condition_type 没抛")
    except ValueError:
        pass

    # 6. direction_word 解析(纯 stdlib)
    p = ("A satellite video of an urban area with buildings and roads, following a "
         "3-segment trajectory with net displacement toward the west.")
    if direction_word(p, None) != "west":
        fails.append("direction_word 没能从 builder prompt 里解析出 'west'")
    # prompt 无方向词 → 从 waypoints 现推。窗口中心从 (100,100) 到 (40,100):
    # dy=-60(往上/北), dx=0 → atan2(60,0)=90 → north。
    if direction_word("A satellite video of a rural area.", [(100, 100), (40, 100)]) != "north":
        fails.append("direction_word 的 waypoints 回退分箱不对(期望 north)")

    # 7. baseline_status.json 与注册表对账
    sp = Path(__file__).resolve().parent / "baseline_status.json"
    if not sp.is_file():
        fails.append(f"缺 {sp}")
    else:
        doc = json.loads(sp.read_text())
        entries = {e["name"]: e for e in doc["baselines"]}
        if set(entries) != {a.name for a in adapters}:
            fails.append(f"baseline_status.json 与注册表不符: "
                         f"json 多 {sorted(set(entries) - {a.name for a in adapters})} "
                         f"少 {sorted({a.name for a in adapters} - set(entries))}")
        for a in adapters:
            e = entries.get(a.name)
            if not e:
                continue
            if e["kind"] != a.kind:
                fails.append(f"{a.name}: json kind={e['kind']} != 注册表 {a.kind}")
            if e["implemented"] != (a.status == "implemented"):
                fails.append(f"{a.name}: json implemented={e['implemented']} 与 status={a.status} 矛盾")
            lifecycle = ("configured", "smoke_tested", "gpu_tested", "trained",
                         "evaluated", "result_available")
            missing_lifecycle = [k for k in lifecycle if k not in e]
            if missing_lifecycle:
                fails.append(f"{a.name}: 缺 lifecycle 字段 {missing_lifecycle}")
            # Legacy `tested` is retained for older tooling and now means exactly
            # synthetic/runtime smoke, not an official benchmark result.
            if bool(e.get("tested")) != bool(e.get("smoke_tested")):
                fails.append(f"{a.name}: tested 必须与 smoke_tested 一致")
            if e.get("result_available") and not e.get("evaluated"):
                fails.append(f"{a.name}: result_available=true 但 evaluated=false")
            if not e["implemented"] and not e["blocking_dependency"]:
                fails.append(f"{a.name}: 未实现就必须写明 blocking_dependency")

    # 8. numpy 在场时才做的形状检查(没 numpy 就跳过,不谎报通过)
    try:
        import numpy as np
    except ImportError:
        print("\n[smoke] numpy 不在 → 跳过形状/dtype 检查(未测,不代表通过)")
    else:
        good = {
            "sample_id": "s0",
            "generated_frames": np.zeros((4, 8, 8, 3), np.uint8),
            "source_frames": np.zeros((1, 8, 8, 3), np.uint8),
            "target_frames": np.zeros((4, 8, 8, 3), np.uint8),
            "route_metadata": {"prompt": "p toward the east.", "waypoints": [[0, 0], [0, 5]],
                               "segment_angles": [0.0], "num_segments": 1,
                               "direction_word": "east", "condition_type": "none"},
            "runtime_metadata": {"wall_seconds": 0.1, "gen_seed": 42, "resolution": [8, 8],
                                 "num_frames": 4, "device": "cpu", "method_version": "v"},
        }
        try:
            validate_record(good)
        except Exception as e:  # noqa: BLE001
            fails.append(f"validate_record 拒绝了合法 record: {type(e).__name__}: {e}")
        bad = dict(good, generated_frames=np.zeros((4, 8, 8, 3), np.float32))
        try:
            validate_record(bad)
            fails.append("validate_record 对 float32 帧没抛")
        except TypeError:
            pass
        bad = dict(good, target_frames=np.zeros((3, 8, 8, 3), np.uint8))
        try:
            validate_record(bad)
            fails.append("validate_record 对 generated/target 帧数不符没抛")
        except ValueError:
            pass
        print("\n[smoke] numpy 在场 → 形状/dtype 检查已跑")

    print()
    if fails:
        for f in fails:
            print(f"[FAIL] {f}")
        print(f"\n[smoke] 失败 {len(fails)} 项")
        return 1
    print("[smoke] 接线自检全部通过。")
    smoke_names = [e["name"] for e in json.loads(sp.read_text())["baselines"]
                   if e.get("smoke_tested")] if sp.is_file() else []
    print("[smoke] 注意:注册表自检本身只测接线;单独记录的 synthetic/runtime "
          f"smoke={smoke_names},不等于 GPU_TESTED/EVALUATED/RESULT_AVAILABLE。")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke())
