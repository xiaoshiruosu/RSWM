#!/usr/bin/env python3
"""
RSWM 评估 Pipeline

用法:
  # 快速测试 (5 条)
  python evaluate.py \
    --checkpoint_dir ./lingbot-world-base-cam \
    --lora_path ./outputs/best_checkpoint \
    --data_path ./data/remotesense/train.h5 \
    --split_file ./outputs/train_val_split.json \
    --output_dir eval_output \
    --max_samples 5

  # 全量评估 (100 条验证集)
  python evaluate.py \
    --checkpoint_dir ... --lora_path ... --data_path ... --split_file ... \
    --output_dir eval_output

  # 仅计算指标 (已有推理结果)
  python evaluate.py --metrics_only --output_dir eval_output
"""

import os
import sys
import json
import argparse
import inspect
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import h5py
# [v3.1] torch 惰性导入 —— 同 metrics.py 的理由:
# evaluate.py 只在 run_inference_batch() 里用 torch(清显存),而
# direction_target_from_waypoints / direction_word_from_waypoints 这些
# **方向口径的真源**是纯 numpy。测试与评分器只需要后者,却因为模块级
# import torch 而在无 torch 的 CI 里整个 import 不进来。
def _torch():
    try:
        import torch as _t
        return _t
    except ImportError as e:
        raise RuntimeError(
            "本路径需要 PyTorch(推理与显存管理)。当前环境没有 torch。\n"
            "  纯 CPU 无 torch 环境下可用:方向口径函数、契约校验、--metrics_only "
            "的非深度指标部分。"
        ) from e

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if (HERE / "wan").is_dir():
    PROJECT_ROOT = HERE
else:
    PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# [v2.6] Track A well-posed 双栏:净方向"有没有意义"由 GT 命令轨迹的净位移比
# 决定(近闭合轨迹的净方向问题本身无定义)。比值公式与门限都从
# trajectory_metrics 取 —— 与 Track B 净方向门控【同一真源】,两轨口径永不漂移
# (审稿人小组 2026-07-25:"Track A 需 net-ratio>=0.2 well-posed 双栏,
# 与 Track B 门控口径一致")。
from trajectory_metrics import polyline_net_ratio, net_direction_from_path

WELL_POSED_MIN_NET_RATIO = float(
    inspect.signature(net_direction_from_path).parameters["min_net_ratio"].default)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# 数据加载
# ============================================================================

def load_split_indices(split_file: str, split_name: str = "val") -> List[int]:
    """从训练两路或 official 三路 JSON 加载显式命名的 split。

    ``val`` remains the default so training-time evaluation cannot accidentally
    consume the hidden test.  Formal paper orchestration must opt into ``test``
    explicitly and writes to a separate output directory.
    """
    if split_name not in ("val", "test"):
        raise ValueError(f"unsupported evaluation split: {split_name}")
    with open(split_file, "r", encoding="utf-8") as f:
        split = json.load(f)
    legacy_key = f"{split_name}_indices"
    if legacy_key in split:
        return [int(i) for i in split[legacy_key]]
    try:
        return [int(i) for i in split["splits"][split_name]["indices"]]
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"{split_file}: 不支持的 split schema；缺 {legacy_key} 或 "
            f"splits.{split_name}.indices") from e


def load_val_indices(split_file: str) -> List[int]:
    """Backward-compatible training/validation helper."""
    return load_split_indices(split_file, "val")


# [v2.5 taxonomy] 与 lora_remotesense/scene_taxonomy.json 保持一致(单测校验):
# flood 更名 water;旧 prompt/attrs 里的 flood 经别名归一。此处保留极简副本
# 以免评估核心反向依赖 builder 模块。
SCENE_CATEGORIES = ["general", "water", "urban", "rural"]
_SCENE_ALIASES = {"flood": "water"}


def canonical_scene(name: str) -> str:
    s = str(name)
    return _SCENE_ALIASES.get(s, s)


def direction_word_from_waypoints(waypoints) -> Optional[str]:
    """[v2.5] 从 waypoints 端点净位移推导【相机】方向词 E/N/W/S。
    prompt 无方向词(direction_stripped 归因实验)时,dir_* 分块统计的回退来源
    —— 否则 33 号作业的 eval_results.json 一个 dir_* 块都不会有,而它的整个
    论证(几何通道能否独立携带方向)恰恰需要分方向对比。"""
    if waypoints is None or len(waypoints) < 2:
        return None
    wp = np.asarray(waypoints, dtype=np.float32)
    ang = float(np.degrees(np.arctan2(
        -(wp[-1, 0] - wp[0, 0]), wp[-1, 1] - wp[0, 1]))) % 360
    return ("E" if (ang < 45 or ang >= 315) else
            "N" if ang < 135 else "W" if ang < 225 else "S")


def direction_target_from_waypoints(waypoints) -> Optional[float]:
    """[v2.5] 从 waypoints 端点净位移推导方向目标角(内容角,=量化相机角+180)。
    与建库 _make_prompt 的量化逐位一致 —— direction-stripped 归因实验的
    Track A 目标来源;prompt 有方向词时不走这里。"""
    if waypoints is None or len(waypoints) < 2:
        return None
    wp = np.asarray(waypoints, dtype=np.float32)
    ang = float(np.degrees(np.arctan2(
        -(wp[-1, 0] - wp[0, 0]), wp[-1, 1] - wp[0, 1]))) % 360
    # [RSRoll-Bench P0#5] 连续式下**不得**再量化：22.5° 的地板是这一步引入的。
    # quantize 仅在 legacy 对照臂保留（逐位复刻旧分箱，含 float32 口径）。
    if os.environ.get("RSWM_PROMPT_SCHEMA", "net") != "net":
        return (ang + 180.0) % 360.0
    cam = (0.0 if (ang < 45 or ang >= 315) else
           90.0 if ang < 135 else
           180.0 if ang < 225 else 270.0)
    return (cam + 180.0) % 360.0


def parse_scene_from_prompt(prompt: str) -> str:
    """从 prompt 中解析场景类别(prompt 是回退;attrs 才是首选来源)。"""
    prompt_lower = prompt.lower()
    if "water-dominated" in prompt_lower or "flood" in prompt_lower:
        return "water"
    elif "urban" in prompt_lower or "built-up" in prompt_lower:
        return "urban"
    elif "rural" in prompt_lower or "natural-land" in prompt_lower:
        return "rural"
    return "general"


def parse_direction_word_from_prompt(prompt: str) -> Optional[str]:
    """从 prompt 解析【相机方向词】,返回 'E'/'N'/'W'/'S' 或 None。

    **单一真源**:方向词的匹配规则只在这里实现一次,
    `parse_direction_angle_from_prompt`(合规目标角)与 compute_metrics 的
    分方向统计都调它 —— 避免两处各写一遍、改了 prompt 只改到一处。
    (这正是曾经的 bug:内联版只认旧措辞 "eastward",修正后的
     "toward the east" 全部落 None,dir_* 分方向统计会静默整块丢失。)

    同时兼容旧措辞("moving eastward")与新措辞("toward the east"):
    基数词彼此不互为子串,且 "east" 亦命中 "eastward",一次检查覆盖两种模板。
    场景词(urban/rural/flooded/general)均不含基数方向词,无误命中。
    """
    p = prompt.lower()
    if "east" in p:
        return "E"
    if "north" in p:
        return "N"
    if "west" in p:
        return "W"
    if "south" in p:
        return "S"
    return None


class _BackboneDone(Exception):
    """[P0#8] 内部控制流哨兵：注册表骨干已产出 uint8，跳过 Wan 专用的张量后处理。

    用异常而不是 if/else，是为了让 `finally: shutil.rmtree(action_dir)` 这段
    临时目录清理在两条路径上都成立 —— 复制一份清理代码是此类泄漏的经典来源。
    """


_DIR_WORD_TO_CAMERA_ANGLE = {"E": 0.0, "N": 90.0, "W": 180.0, "S": 270.0}


def parse_direction_angle_from_prompt(prompt: str) -> Optional[float]:
    """从 prompt 解析【合规目标角】(度) = 相机方向 + 180°。

    prompt 里的方向词描述的是【窗口/相机】的移动方向;而 metrics.py 用
    光流估计的是【图像内容】的运动方向。基准把相机平移实现为静态大图上
    的滑窗,窗口向东滑则内容向西流——两者恰好相反。因此合规目标是
    方向词对应的角度旋转 180°:GT 视频在该约定下得分应接近 22.5°
    量化地板;逐字跟随方向词做内容漂移的生成则接近 180°。
    (修正前的实现直接用相机角做目标;由于折叠误差对称,两种约定的
    分数满足 e -> 180 - e 的精确映射。)
    """
    # [RSRoll-Bench P0#5] 先按 schema 判定，再取靶。
    # 旧实现只认 4 个基数词：连续式 prompt（"toward a bearing of 137.0 degrees"）
    # 不含任何基数词 → **静默**落 None → 该样本整块 dir_* 统计消失而不报错。
    # 必须与 build_dataset_from_benchmark 的 prompt_schema 同一个 commit 落地。
    from continuous_direction import direction_target_from_prompt as _cd_target
    tgt, schema = _cd_target(prompt)
    if schema != "net":
        return tgt
    word = parse_direction_word_from_prompt(prompt)
    if word is None:
        return None
    return (_DIR_WORD_TO_CAMERA_ANGLE[word] + 180.0) % 360.0


def load_val_samples(data_path: str, val_indices: List[int], max_samples: Optional[int] = None):
    """加载验证集样本"""
    samples = []
    n_target_from_prompt = n_target_from_wp = n_target_none = 0
    n_pose_identity = n_pose_native = 0
    # [v2.8.1 P0-4] 运行期 pose_mode(native|identity|drop)是位姿口径的真源。
    # H5 里的位姿矩阵只在 pose_mode 确实把位姿送进模型时才说了算。
    _rt_pose_mode = os.environ.get("RSWM_POSE_MODE", "native").strip().lower()
    if _rt_pose_mode not in ("native", "identity", "drop"):
        raise ValueError(f"RSWM_POSE_MODE={_rt_pose_mode!r} 非法(native|identity|drop)")
    with h5py.File(data_path, "r") as f:
        # [v2.5 硬门B] Track A 光流口径 = 原生窗口分辨率;从 metadata 读,缺失
        # 用 canonical 默认 (128,222) 并大声记录(勿静默用生成分辨率)。
        if "metadata" in f and "pose_window_size" in f["metadata"].attrs:
            pose_window_size = tuple(
                int(v) for v in f["metadata"].attrs["pose_window_size"])
        else:
            pose_window_size = (128, 222)
            logger.warning("H5 metadata 缺 pose_window_size,角误差口径回退默认 (128,222)")
        # [v2.6/O11] 位姿口径的文件级标记(neutralize_poses.py 写
        # pose_injection='identity_neutralized';GEO/variant 无此 attr)。
        # 真源是【逐样本 poses 本身】(下方 allclose 判定),attr 只作交叉校验:
        # attr 说被中和、样本却非 identity → 文件被篡改/搞混,硬错。
        file_pose_injection = None
        if "metadata" in f and "pose_injection" in f["metadata"].attrs:
            _pi = f["metadata"].attrs["pose_injection"]
            file_pose_injection = _pi.decode() if isinstance(_pi, bytes) else str(_pi)
        keys = sorted(f["data"].keys())
        for idx in val_indices:
            if max_samples and len(samples) >= max_samples:
                break
            if idx >= len(keys):
                continue
            key = keys[idx]
            s = f["data"][key]
            sample = {
                "key": key,
                "index": idx,
                "image": s["image"][:],              # (H, W, 3)
                "video_frames": s["video_frames"][:], # (T, H, W, 3)
                "poses": s["poses"][:],               # (T, 4, 4)
                "intrinsics": s["intrinsics"][:],     # (T, 4)
                "prompt": s.attrs.get("prompt", "A satellite video of a remote sensing scene."),
                "num_frames": int(s.attrs.get("num_frames", s["video_frames"].shape[0])),
            }
            # 场景:attrs 是首选真源(B1 修复起持久化),prompt 解析只作回退;
            # canonical_scene 把旧数据的 flood 归一为 water。
            sample["scene"] = canonical_scene(
                s.attrs.get("source_scene_category", "")
                or parse_scene_from_prompt(sample["prompt"]))
            sample["waypoints"] = (s["waypoints"][:] if "waypoints" in s else None)
            sample["prompt_schema"] = str(
                s.attrs.get("prompt_schema",
                            f["metadata"].attrs.get("prompt_schema", "net")
                            if "metadata" in f else "net"))
            sample["pose_window_size"] = pose_window_size
            # [v2.6/O11] 逐样本位姿口径判定(commanded 语义的数据级真源,落盘
            # 进 meta 供 run_trajectory_eval 读 —— 取代目录名字符串猜测):
            # 全 identity = 位姿被中和(text-only 路由);否则 native(geo/
            # variant 路由)。neutralize 写的是精确 eye,1e-6 容差只吸浮点往返;
            # native 位姿平移步长 ~1e-2 量级,与 identity 差 >4 个数量级,无误伤。
            _is_identity = bool(np.allclose(
                sample["poses"].astype(np.float64), np.eye(4), atol=1e-6))
            # [v2.8.1 P0-4] pose_mode=drop 时位姿【根本没进模型】(train/infer 都是
            # dit_cond_dict=None),H5 里躺着 native 位姿也不代表模型被命令过。
            # 旧代码只看 H5 矩阵 → 35/36(GEO_H5 + drop)会被写成 native,
            # run_trajectory_eval 据此把逐段/转弯/waypoint 全标 commanded=True,
            # 即"模型没收到的命令却拿来算它的服从度"。
            if _rt_pose_mode == "drop":
                sample["pose_conditioning"] = "dropped"
            else:
                sample["pose_conditioning"] = (
                    "identity_neutralized" if _is_identity else "native")
            if _is_identity:
                n_pose_identity += 1
            else:
                n_pose_native += 1
            if _rt_pose_mode == "native" and _is_identity:
                raise RuntimeError(
                    f"pose_mode=native 但样本 {key} 的位姿是 identity —— 这是"
                    f"把中和过的 SEM_H5 当几何臂在跑,几何通道实际全零,"
                    f"任何 native↔text 对比都无效。请设 POSE_MODE=identity 或换 GEO_H5。")
            if _rt_pose_mode == "identity" and not _is_identity:
                raise RuntimeError(
                    f"pose_mode=identity 但样本 {key} 的位姿非 identity —— "
                    f"静止相机诊断臂却喂了真实轨迹位姿,该臂的语义不成立。")
            if file_pose_injection == "identity_neutralized" and not _is_identity:
                raise RuntimeError(
                    f"{data_path} metadata 声称 pose_injection=identity_neutralized,"
                    f"但样本 {key} 的 poses 不是 identity —— SEM 文件被篡改或"
                    f"传错了数据集,commanded 语义无法信任,拒绝继续。")
            # [v2.6] GT 命令轨迹净位移比(Track A well-posed 双栏依据;公式与
            # Track B 门控同源 polyline_net_ratio)。无 waypoints(旧数据)记
            # None —— 缺数据不臆造,该样本不进 well-posed 子集并被计数。
            sample["gt_net_displacement_ratio"] = (
                float(polyline_net_ratio(sample["waypoints"]))
                if sample["waypoints"] is not None and len(sample["waypoints"]) >= 2
                else None)
            # 方向目标:prompt 优先;prompt 无方向词(direction-stripped 归因
            # 实验)时从 waypoints 推导 —— 复刻建库同款量化(端点净位移→四基
            # 向→内容角=+180),保证与带词 prompt 的目标定义逐位一致。
            # [v2.8.1 P0-3] 真源改为 **waypoints**(数据级),prompt 只在没有
            # waypoints 的旧数据上回退。
            #
            # 原先 prompt 优先。但 parse_direction_*_from_prompt 是按固定序
            # east→north→west→south 取第一个命中的基数词 —— 单方向句 prompt 下
            # 没问题,一旦 prompt 含多个方向词(35 号 rich_persegment 臂:
            # "segment 1 toward the west, then segment 2 toward the north")
            # 就会返回一个既非净方向、也非首段的任意词,Track A 目标随之作废。
            # waypoints 是建库时真正走出来的轨迹,唯一不会被措辞影响。
            _schema = sample.get("prompt_schema", "net")
            _has_wp = (sample["waypoints"] is not None
                       and len(sample["waypoints"]) >= 2)
            _p_ang = parse_direction_angle_from_prompt(sample["prompt"])
            _p_dw = parse_direction_word_from_prompt(sample["prompt"])
            if _has_wp:
                tgt = direction_target_from_waypoints(sample["waypoints"])
                dw = direction_word_from_waypoints(sample["waypoints"])
                sample["direction_target_source"] = "waypoints"
                n_target_from_wp += 1
                # 单方向句 prompt 与 waypoints 必须一致;不一致 = 建库/派生
                # 环节把图像、轨迹、措辞对错位了,任何方向指标都不可信。
                # rich_persegment 含多个方向词,朴素解析器在它上面无意义,不核。
                if (_schema == "net" and _p_ang is not None
                        and abs(((_p_ang - tgt + 180.0) % 360.0) - 180.0) > 1e-6):
                    raise RuntimeError(
                        f"样本 {key}: prompt 净方向({_p_dw},{_p_ang}°)与 waypoints "
                        f"净位移({dw},{tgt}°)矛盾 —— prompt 与轨迹不同源,"
                        f"方向指标全部不可信,拒绝继续。")
            elif _p_ang is not None:
                tgt, dw = _p_ang, _p_dw
                sample["direction_target_source"] = "prompt"
                n_target_from_prompt += 1
                if _schema == "rich_persegment":
                    raise RuntimeError(
                        f"样本 {key}: rich_persegment prompt 含多个方向词,"
                        f"且无 waypoints 可作真源 —— 目标方向无法确定。")
            else:
                tgt, dw = None, None
                sample["direction_target_source"] = None
                n_target_none += 1
            sample["direction_angle"] = tgt
            sample["direction_word"] = dw
            samples.append(sample)
    # [v2.6/O11] 同一 H5 内两种位姿口径混存 = 数据损坏(builder/neutralize 都
    # 不可能产生;variant 派生器也不动语义)。此时任何 commanded 标注都不可信。
    if n_pose_identity > 0 and n_pose_native > 0:
        raise RuntimeError(
            f"{data_path}: 位姿口径混合(identity={n_pose_identity} / "
            f"native={n_pose_native})—— 同一数据集只能有一种口径,拒绝评估。")
    logger.info(f"加载 {len(samples)} 条验证样本;方向目标来源: "
                f"prompt={n_target_from_prompt} waypoints={n_target_from_wp} "
                f"无={n_target_none};位姿口径: "
                f"{'identity_neutralized(text 路由)' if n_pose_identity else 'native(geo/variant 路由)'}")
    return samples


# ============================================================================
# 推理
# ============================================================================

def run_inference_batch(
    samples: List[Dict],
    checkpoint_dir: str,
    lora_path: str,
    output_dir: str,
    device: str = "cuda",
    frame_num: int = 33,
    size: str = "480*832",
    t5_cpu: bool = True,
    framewise: bool = True,
    inactive_model_cpu: bool = True,
    gen_seed: int = 42,
    split_name: str = "val",
):
    """批量推理，保存生成视频和 GT"""
    from inference import load_lora_model, generate_video
    from wan.configs import WAN_CONFIGS
    from PIL import Image
    import shutil
    import tempfile

    gen_dir = Path(output_dir) / "generated"
    gt_dir = Path(output_dir) / "gt"
    meta_dir = Path(output_dir) / "meta"
    gen_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # [v2.9.2 P0-2] 断点续推的身份门 —— 必须在目录创建【之后】、推理循环之前。
    # 覆盖 profile / pose_mode / role / 数据与 split 的 SHA / checkpoint /
    # prompt_schema / frame_num / gen_seed / 代码指纹;缺 manifest 也拒。
    # 旧写法"meta 文件在就跳过"意味着 identity+legacy 的旧结果能在
    # drop+official 环境里被原样复用,再被当前环境写上 official/headline 标记。
    from run_manifest import (build_manifest, read_manifest,
                              write_manifest, assert_compatible, summarize)
    # 评估产物必须能独立回答“这是哪个 rank / target / 训练配方的结果”。
    # 旧版只记录 checkpoint 路径，目录一旦搬家或改名，论文表无法复核。
    _train_manifest = read_manifest(lora_path) or {}
    _cur_manifest = build_manifest(
        role=os.environ.get("RSWM_RUN_ROLE", "eval"),
        data_h5=os.environ.get("RSWM_MANIFEST_DATA_H5") or None,
        split_file=os.environ.get("RSWM_MANIFEST_SPLIT") or None,
        split_name=split_name,
        checkpoint_id=str(lora_path or "base"),
        prompt_schema=(samples[0].get("prompt_schema") if samples else None),
        frame_num=frame_num,
        gen_seed=gen_seed,
        train_seed=_train_manifest.get("train_seed"),
        split_seed=_train_manifest.get("split_seed"),
        dataset_tier=_train_manifest.get("dataset_tier"),
        lora_rank=_train_manifest.get("lora_rank"),
        lora_alpha=_train_manifest.get("lora_alpha"),
        lora_targets=_train_manifest.get("lora_targets"),
        max_train_samples=_train_manifest.get("max_train_samples"),
        num_epochs=_train_manifest.get("num_epochs"),
        batch_size=_train_manifest.get("batch_size"),
        gradient_accumulation_steps=_train_manifest.get(
            "gradient_accumulation_steps"),
        learning_rate=_train_manifest.get("learning_rate"),
        # [P0#8 + P0#1] 骨干与估计器进身份字段：换骨干的产物不可与旧产物互换
        backbone=os.environ.get("RSWM_BACKBONE", "wan-inline"),
        backbone_adaptation=("lora" if lora_path else "base"),
        path_estimator=os.environ.get("RSWM_PATH_ESTIMATOR", "flow"))
    if any(meta_dir.glob("*.json")):
        assert_compatible(read_manifest(output_dir), _cur_manifest,
                          ctx="评测断点续推")
        logger.info(f"[manifest] 复用已有产物,身份一致 — {summarize(_cur_manifest)}")
    write_manifest(output_dir, _cur_manifest)
    meta_dir.mkdir(parents=True, exist_ok=True)

    device_obj = _torch().device(device)
    config = WAN_CONFIGS["i2v-A14B"]

    # [RSRoll-Bench P0#8] 骨干可选。
    #
    # 现状（本改动之前）：本文件把 `WAN_CONFIGS["i2v-A14B"]` 与 `load_lora_model` /
    # `generate_video` 直接写死，而 backbone.py 定义的 VideoBackbone 协议
    # 在 evaluate.py / evaluate_baseline.py / generate_samples.py / common.sh 里
    # **一次都没被引用** —— "统一适配器"因此没有任何可运行路径，
    # 论文里"任何模型都能在同等条件下接入"这句话没有代码支撑。
    #
    # 风险姿态：默认值 "wan-inline" 走**原封不动**的老路径（下面这段代码一字未改），
    # 因此默认数值恒等；只有显式指定别的骨干时才走注册表。
    # 这么做是因为 GPU 生成路径无法在开发机上验证，不能拿默认路径冒险。
    _BACKBONE = os.environ.get("RSWM_BACKBONE", "wan-inline")
    _bb = None
    if _BACKBONE != "wan-inline":
        from backbone import build_backbone, wan_tensor_to_uint8  # noqa: F401
        logger.info("[P0#8] 走 VideoBackbone 注册表：%s", _BACKBONE)
        _bb = build_backbone(_BACKBONE)
        text_encoder = vae = low_model = high_model = None
    else:
        # 加载模型 (只加载一次)
        logger.info("加载模型...")
        text_encoder, vae, low_model, high_model = load_lora_model(
            checkpoint_dir=checkpoint_dir,
            lora_path=lora_path,
            device=device_obj,
            config=config,
            t5_cpu=t5_cpu,
            inactive_model_cpu=inactive_model_cpu,
        )
        logger.info("模型加载完成")

    skipped = 0
    failed = 0          # [v2.7.1 F1/P0] 失败计数 —— 全失败/高失败率必须炸,不许 exit 0
    for i, sample in enumerate(samples):
        # 断点续推: 跳过已完成的样本
        # [v2.7.1 F3/P1 修复] 完成标志改用 meta(写序 gen→gt→meta,它最后落盘):
        # 旧写法查 gen npy,中途崩溃留下的"有 gen 无 GT/meta"半成品会被永久跳过
        # → 指标静默丢样本;半成品 gen/gt 会被下面的 np.save 直接覆盖,无需清理。
        existing_meta = meta_dir / f"{sample['key']}.json"
        if existing_meta.exists():
            # [v2.9.2 P0-2] 复用之前,身份必须已经被 _assert_eval_manifest()
            # 核过(它在进入本循环前跑,缺 manifest 或字段不同即抛)。
            # 旧写法"meta 文件在就跳过"意味着:identity+legacy 的旧结果能在
            # drop+official 环境里被原样复用,而 overall 随后又用当前环境写上
            # official/headline 标记 —— 旧结果被重新认证成合格 headline。
            skipped += 1
            logger.info(f"[{i+1}/{len(samples)}] 跳过 {sample['key']} (已存在,身份已核验)")
            continue

        logger.info(f"[{i+1}/{len(samples)}] 推理 {sample['key']}...")
        t0 = time.time()

        # 保存首帧为临时图片
        img_arr = sample["image"]
        if img_arr.dtype != np.uint8:
            img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_arr)

        # 保存 poses/intrinsics 到临时目录
        action_dir = tempfile.mkdtemp()
        np.save(os.path.join(action_dir, "poses.npy"), sample["poses"])
        np.save(os.path.join(action_dir, "intrinsics.npy"), sample["intrinsics"])

        # 推理
        try:
            if _bb is not None:
                # [P0#8] 注册表路径：协议要求 (T,H,W,3) uint8 直接返回，
                # 因此不走下面 Wan 的 (C,T,H,W) float[-1,1] 转换。
                from backbone import PoseCond as _PoseCond
                # pose_mode 三态与 pose_cond 的对应必须精确，否则
                # backbone.assert_pose_cond_matches_mode 会直接抛：
                #   drop     = 位姿通路整条不接 → pose_cond 必须为 None（真 text-only 臂）
                #   identity = 静止相机诊断臂 → 仍要送位姿，H5 里存的就是单位阵
                #   native   = 真实轨迹
                # 前一版这里无条件传 PoseCond，drop 臂会 100% 抛错 —— 而 drop
                # 正是本文需要的那条真 text-only 臂。
                # 就地读：_rt_pose_mode 是 load_val_samples 的局部变量，
                # 在本函数里引用它是 NameError（只在启用注册表骨干时才炸，
                # 属于最难发现的那类）。语义与 :214 完全一致。
                _pm = os.environ.get("RSWM_POSE_MODE", "native").strip().lower()
                _pc = None
                if _pm != "drop":
                    _pc = _PoseCond(poses=sample["poses"],
                                    intrinsics=sample["intrinsics"])
                gen_video = _bb.generate(
                    first_frame=img_arr,
                    prompt=sample["prompt"],
                    num_frames=min(frame_num, sample["num_frames"]),
                    size=size,
                    pose_cond=_pc,
                    seed=gen_seed,
                )
                raise _BackboneDone
            video_tensor = generate_video(
                text_encoder=text_encoder,
                vae=vae,
                low_noise_model=low_model,
                high_noise_model=high_model,
                config=config,
                prompt=sample["prompt"],
                image=pil_img,
                device=device_obj,
                frame_num=min(frame_num, sample["num_frames"]),
                size=size,
                action_path=action_dir,
                seed=gen_seed,
                t5_cpu=t5_cpu,
                framewise=framewise,
                inactive_model_cpu=inactive_model_cpu,
            )
            # video_tensor: (C, T, H, W) float, [-1, 1]
            # 转为 (T, H, W, C) uint8
            gen_video = video_tensor.clamp(-1, 1).permute(1, 2, 3, 0)
            gen_video = ((gen_video + 1) * 127.5).byte().cpu().numpy()
        except _BackboneDone:
            pass          # [P0#8] 注册表路径已直接产出 uint8，跳过 Wan 的张量转换
        except Exception as e:
            logger.error(f"推理失败 {sample['key']}: {e}")
            failed += 1
            continue
        finally:
            # [v2.7.1 F9/P2] 每样本的 mkdtemp 此前从不清理(成功/失败都泄漏)
            shutil.rmtree(action_dir, ignore_errors=True)

        # 保存
        np.save(str(gen_dir / f"{sample['key']}.npy"), gen_video)

        gt_video = sample["video_frames"]
        if gt_video.dtype != np.uint8:
            gt_video = np.clip(gt_video, 0, 255).astype(np.uint8)
        np.save(str(gt_dir / f"{sample['key']}.npy"), gt_video)

        # 元数据
        meta = {
            "key": sample["key"],
            "prompt": sample["prompt"],
            "scene": sample["scene"],
            "direction_angle": sample["direction_angle"],
            "num_frames_gt": sample["num_frames"],
            "num_frames_gen": gen_video.shape[0],
            "gen_seed": int(gen_seed),
            # [v2.5] 角误差口径与方向目标来源 —— compute_metrics 离线重算要用
            "pose_window_size": list(sample.get("pose_window_size", (128, 222))),
            "direction_target_source": sample.get("direction_target_source"),
            "direction_word": sample.get("direction_word"),
            # [v2.8.1 P0-3/P0-4] prompt 措辞契约:net=只有净方向句;
            # rich_persegment=净方向+逐段方向;stripped=无方向词。
            # 它决定"文本命令了什么",与 pose_conditioning 共同定 commanded 语义。
            "prompt_schema": sample.get("prompt_schema", "net"),
            "rswm_profile": os.environ.get("RSWM_PROFILE", "unset"),
            "headline_eligible": os.environ.get("RSWM_HEADLINE_ELIGIBLE", "0") == "1",
            "sem_arm_mode": os.environ.get("SEM_ARM_MODE", "unset"),
            "direction_target_source": sample.get("direction_target_source"),
            # [v2.6/O11] 位姿口径(commanded 语义真源,来自逐样本 poses 判定)——
            # run_trajectory_eval 据此定 text_only,取代目录名字符串猜测。
            "pose_conditioning": sample.get("pose_conditioning"),
            # [v2.8 ④] 三臂唯一区分点:identity(静止相机诊断臂)与 drop
            # (真 no-pose/text-only)在 H5 层看不出差别,只有它能区分
            "pose_mode": os.environ.get("RSWM_POSE_MODE", "native"),
            # [v2.6] GT 命令轨迹净位移比 —— Track A well-posed 双栏依据
            # (与 Track B polyline_net_ratio 门控同源;无 waypoints 时 null)。
            "gt_net_displacement_ratio": sample.get("gt_net_displacement_ratio"),
        }
        # 推理耗时 / FPS —— 参数效率表要的实测字段。此前只打日志、不落盘,
        # 导致 parameter_efficiency_summary 拿不到 inference walltime。
        # 注意: elapsed 含本样本的 generate_video 全过程(不含模型加载,那是一次性的)。
        elapsed = time.time() - t0
        meta["infer_seconds"] = round(float(elapsed), 3)
        meta["infer_fps"] = (round(gen_video.shape[0] / elapsed, 3)
                             if elapsed > 0 else None)
        with open(str(meta_dir / f"{sample['key']}.json"), "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"  完成, {elapsed:.1f}s ({meta['infer_fps']} fps), gen shape={gen_video.shape}")

        # 清理显存
        _torch().cuda.empty_cache()

    if skipped > 0:
        logger.info(f"跳过 {skipped} 个已完成样本, 新推理 {len(samples) - skipped} 个")
    # [v2.7.1 F1/P0 修复] 旧行为:全部样本推理失败也 exit 0、写空结果、
    # p0_smoke 照样盖"通过"章 —— GPU 一条没成功却永久标记完成。
    attempted = len(samples) - skipped
    if attempted > 0 and failed == attempted:
        raise RuntimeError(f"全部 {attempted} 个新推理样本失败 —— 判死,不产出结果")
    # [v3.0 P0] 零失败容忍。旧行为容忍 20% —— 那等于允许提交方"选择性放弃"
    # 最难的五分之一样本,剩下的分数照进主表。评分口径下任何失败都是整份失败。
    # 排障需要继续时用 RSWM_LENIENT_METRICS=1,该次结果会被标 tainted。
    from submission_contract import assert_no_failures
    assert_no_failures(failed, attempted, what="推理")
    if failed:
        logger.error(f"{failed}/{attempted} 样本推理失败 —— 宽松模式放行,结果已 tainted")
    logger.info("推理全部完成")


# ============================================================================
# 指标计算
# ============================================================================

def compute_metrics(output_dir: str, device: str = "cuda"):
    """从保存的 npy 文件计算所有指标"""
    from metrics import compute_all_metrics

    gen_dir = Path(output_dir) / "generated"
    gt_dir = Path(output_dir) / "gt"
    meta_dir = Path(output_dir) / "meta"

    # 收集所有样本
    gen_files = sorted(gen_dir.glob("*.npy"))
    logger.info(f"找到 {len(gen_files)} 个生成结果")
    # [v3.0 P0] 必须评估**要求的全部** ID,不是"generated/ 里恰好存在的那些"。
    # 旧行为让"少交样本"完全不被发现:漏掉的样本既不报错也不计分,
    # 一份只交了容易样本的提交会拿到干净的高分。
    # required_ids.json 由建库/划分侧落盘;没有它就退回"用 gt/ 作为要求集"
    # —— gt 是真值,提交方无法删改它。
    from submission_contract import assert_id_set
    _req = None
    _rq = Path(output_dir) / "required_ids.json"
    if _rq.is_file():
        try:
            _req = list(json.loads(_rq.read_text())["ids"])
        except Exception as _e:
            logger.warning(f"[contract] {_rq} 解析失败({_e}),退回用 gt/ 作要求集")
    if _req is None:
        _req = sorted(p.stem for p in gt_dir.glob("*.npy"))
    if _req:
        assert_id_set([p.stem for p in gen_files], _req, what="生成结果")
    else:
        logger.warning("[contract] 拿不到要求的 ID 集(gt/ 也是空的)—— "
                       "无法校验完整性,这不该发生在评分路径上")

    gen_videos, gt_videos, target_angles, scenes, directions = [], [], [], [], []
    sample_ids = []   # 与 gen_videos 同序;逐样本指标(scores.json)要用它做 sample_id
    infer_secs = []   # 逐样本推理耗时(来自 meta);聚合进 eval_results.json,供参数效率表用
    net_ratios = []   # [v2.6] GT 净位移比(well-posed 双栏);meta 无该键记 "absent"
    _RATIO_KEY_ABSENT = object()   # 区分「键不存在(v2.6 前目录)」与「值为 null(无 waypoints)」
    pose_metas = []   # [v2.8.2 N7] 逐样本位姿口径,聚合进 overall(复用同一次读盘)

    for gf in gen_files:
        key = gf.stem
        gt_file = gt_dir / f"{key}.npy"
        meta_file = meta_dir / f"{key}.json"

        if not gt_file.exists():
            logger.warning(f"缺少 GT: {key}")
            continue

        gen = np.load(str(gf))
        gt = np.load(str(gt_file))

        if gen.dtype != np.uint8:
            gen = np.clip(gen, 0, 255).astype(np.uint8)
        if gt.dtype != np.uint8:
            gt = np.clip(gt, 0, 255).astype(np.uint8)

        gen_videos.append(gen)
        gt_videos.append(gt)
        sample_ids.append(key)

        # 元数据
        if meta_file.exists():
            with open(str(meta_file)) as f:
                meta = json.load(f)
            # [v2.5 审查修复] 口径版本门:v2.5 之前的 meta 里 direction_angle 是
            # 【相机角】,之后是【内容角】(相机角+180)。断点续推只看 npy 是否
            # 存在、不重写 meta,所以同一目录可能新旧混存 → 一半样本目标角差
            # 180°,Angular_Error_Mean 变成两个模式的静默混合。缺 v2.5 标记字段
            # 即硬失败,不许静默混算。
            if "pose_window_size" not in meta or "direction_target_source" not in meta:
                raise RuntimeError(
                    f"{key}.json 缺 v2.5 口径标记(pose_window_size/"
                    f"direction_target_source)—— 该 eval 目录由 v2.5 之前的评估器"
                    f"产出,direction_angle 的 180° 约定与现版不同,混算会静默出错。"
                    f"请删除 {meta_dir}/ 与 generated/ 重跑,或只对同版本目录重算。")
            # [v2.5] 旧 eval 目录的 meta 里 scene 可能是 flood → 归一为 water,
            # 否则 --metrics_only 重算会劈出 flood/water 两个半桶。
            scenes.append(canonical_scene(meta.get("scene", "general")))
            angle = meta.get("direction_angle")
            target_angles.append(angle)
            # 解析方向 —— 走单一真源(兼容新旧 prompt 措辞)。
            # 曾经这里内联只认 "eastward",修正 prompt("toward the east")后
            # 会全部落 None、dir_* 分方向统计静默消失。
            # [v2.5] 优先用推理期落盘的 direction_word(direction_stripped 数据
            # 的 prompt 里没有方向词,只能靠 waypoints 回退;旧 meta 无此键时
            # 退回 prompt 解析,与 v2.5 之前行为一致)。
            _dw = meta.get("direction_word")
            if _dw is None:
                _dw = parse_direction_word_from_prompt(meta.get("prompt", ""))
            directions.append(_dw)
            # 旧 eval 目录的 meta 没有 infer_seconds(加打点之前跑的)→ 记 None,不臆造
            infer_secs.append(meta.get("infer_seconds"))
            pose_metas.append({k: meta.get(k) for k in
                               ("pose_mode", "pose_conditioning", "prompt_schema")})
            # [v2.6] well-posed 双栏依据;区分「键不存在」与「值 null」两种缺法
            net_ratios.append(meta["gt_net_displacement_ratio"]
                              if "gt_net_displacement_ratio" in meta
                              else _RATIO_KEY_ABSENT)
        else:
            scenes.append("general")
            target_angles.append(None)
            directions.append(None)
            infer_secs.append(None)
            net_ratios.append(_RATIO_KEY_ABSENT)

    if not gen_videos:
        logger.error("没有可评估的样本")
        return {}

    # 全局指标
    logger.info("计算全局指标...")
    # target_angles 与 gen_videos **逐一对齐**(无方向的样本为 None,由
    # compute_batch_angular_error 跳过)。曾经这里传"全量 videos + 过滤后 angles",
    # zip 按短的截断 → 视频与角度错位、角误差全错;现已根治(metrics.py 会硬失败)。
    _has_angle = any(a is not None for a in target_angles)

    # [v2.5 硬门B] Track A 角误差生产口径 = 原生窗口分辨率 + step=1。
    # 旧默认(生成分辨率 + step=len//8)是量程外口径(GT 地板实测 82°),
    # 一律禁用;窗口尺寸取 meta(生成期从 H5 metadata 落盘),缺失回退
    # canonical (128,222) 并大声记录。
    flow_win = None
    for gf in gen_files:
        mf = meta_dir / f"{gf.stem}.json"
        if mf.exists():
            with open(str(mf)) as f:
                _m = json.load(f)
            if _m.get("pose_window_size"):
                flow_win = tuple(int(v) for v in _m["pose_window_size"])
                break
    if flow_win is None:
        flow_win = (128, 222)
        logger.warning("meta 无 pose_window_size(旧 eval 目录?)—— 角误差口径"
                       "回退 canonical (128,222) + step=1")
    logger.info(f"角误差口径: flow_size={flow_win} step=1")
    _ANGKW = dict(flow_size=flow_win, step=1)

    # per_sample_rows: 逐样本指标 → 落单个 scores.json,供 aggregate_seeds.py --per-traj
    # 做【测试轨迹】bootstrap CI。注意这与【训练种子】方差是两种不同的不确定性,
    # 必须都报、不可互相替代。FID/FVD 是分布级指标,定义上无逐样本值,故不在其中。
    per_sample_rows: List[dict] = []
    results = {"overall": compute_all_metrics(
        gen_videos, gt_videos,
        target_angles=target_angles if _has_angle else None,
        device=device,
        per_sample_out=per_sample_rows,
        **_ANGKW,
    )}
    results["overall"]["num_samples"] = len(gen_videos)
    # [v2.8.2 N7] pose 口径必须进【聚合层】。此前只落到逐样本 meta,于是 34 判门
    # 读的 overall、collect_results、aggregate_seeds 在聚合层仍只能靠目录名区分
    # 三臂 —— 正是 v2.6/O11 要根治的形态(目录名字符串猜测)。
    _pms = sorted({m.get("pose_mode") for m in pose_metas if m.get("pose_mode")}) if pose_metas else []
    _pcs = sorted({m.get("pose_conditioning") for m in pose_metas
                   if m.get("pose_conditioning")}) if pose_metas else []
    _pss = sorted({m.get("prompt_schema") for m in pose_metas
                   if m.get("prompt_schema")}) if pose_metas else []
    # [v2.9.1 N6] 缺键的旧 meta 会被 m.get() 过滤掉 —— "旧 meta + 新 drop meta"
    # 混在一个目录时 _pms 长度仍是 1,静默放行。显式统计缺键数并拒收。
    _n_missing = sum(1 for m in pose_metas if not m.get("pose_mode")) if pose_metas else 0
    if pose_metas and 0 < _n_missing < len(pose_metas):
        raise RuntimeError(
            f"eval 目录里 {_n_missing}/{len(pose_metas)} 个 meta 没有 pose_mode"
            f"(v2.8 之前的产物)而其余有 —— 混版本目录,断点续推?"
            f"删除 meta/ 与 generated/ 重跑。")
    if len(_pms) > 1 or len(_pcs) > 1 or len(_pss) > 1:
        raise RuntimeError(
            f"同一 eval 目录混了多种口径(pose_mode={_pms}, "
            f"pose_conditioning={_pcs}, prompt_schema={_pss}) —— 断点续推跨模式?"
            f"任何 commanded 语义都不可信。删除 meta/ 与 generated/ 重跑。")
    results["overall"]["pose_mode"] = _pms[0] if _pms else None
    results["overall"]["rswm_profile"] = os.environ.get("RSWM_PROFILE", "unset")
    results["overall"]["headline_eligible"] = (
        os.environ.get("RSWM_HEADLINE_ELIGIBLE", "0") == "1")
    results["overall"]["headline_ineligible_reason"] = (
        os.environ.get("RSWM_HEADLINE_INELIGIBLE_REASON") or None)
    results["overall"]["sem_arm_mode"] = os.environ.get("SEM_ARM_MODE", "unset")
    results["overall"]["pose_conditioning"] = _pcs[0] if _pcs else None
    results["overall"]["prompt_schema"] = _pss[0] if _pss else None

    # 推理耗时 / FPS 聚合(参数效率表要的实测字段)。
    # 旧 eval 目录 meta 里没有 infer_seconds → 全 None → 这两个键写 None(不臆造)。
    _secs = [s for s in infer_secs if isinstance(s, (int, float))]
    _frames = [len(v) for v, s in zip(gen_videos, infer_secs)
               if isinstance(s, (int, float))]
    if _secs:
        results["overall"]["Infer_Seconds_Mean"] = float(np.mean(_secs))
        results["overall"]["Infer_Seconds_Total"] = float(np.sum(_secs))
        results["overall"]["Infer_FPS_Mean"] = (
            float(np.sum(_frames) / np.sum(_secs)) if np.sum(_secs) > 0 else None)
        results["overall"]["Infer_Timed_Samples"] = len(_secs)
    else:
        results["overall"]["Infer_Seconds_Mean"] = None
        results["overall"]["Infer_FPS_Mean"] = None
        results["overall"]["Infer_Timed_Samples"] = 0

    # ------------------------------------------------------------------
    # [v2.6] Track A well-posed(net-ratio >= 0.2)双栏
    # ------------------------------------------------------------------
    # 动机:GT 命令轨迹近闭合(净位移比 < 门限)时,"净方向"这个问题对该样本
    # 本身无定义 —— 把它们混进角误差均值会同时稀释成功与失败(Track B v2.1
    # 已按同一理由门控)。双栏 = 全样本读数(与历史可比)+ well-posed 子集
    # 读数(与 Track B 门控口径一致的正式读数),两栏并报、各带 n。
    # 版本卫生(镜像 v2.5 口径门):
    #   * meta 全部没有 gt_net_displacement_ratio 键 → v2.6 之前的目录,
    #     大声跳过双栏(不硬错 —— 该目录的其他指标仍有效);
    #   * 部分有部分没有 → v2.6 前后混版本目录(断点续推),硬错。
    _n_ratio_absent = sum(1 for r in net_ratios if r is _RATIO_KEY_ABSENT)
    well_posed_available = (_n_ratio_absent == 0)
    if 0 < _n_ratio_absent < len(net_ratios):
        raise RuntimeError(
            f"{_n_ratio_absent}/{len(net_ratios)} 个样本的 meta 缺 "
            f"gt_net_displacement_ratio 键 —— v2.6 前后混版本 eval 目录"
            f"(断点续推?)。请删除 {meta_dir}/ 与 generated/ 重跑,"
            f"或只对同版本目录重算。")
    if not well_posed_available:
        logger.warning("meta 无 gt_net_displacement_ratio(v2.6 之前的 eval 目录)"
                       "—— 不产 Track A well-posed 双栏,其余指标照常。")
        well_posed = [None] * len(gen_videos)
    else:
        # None(该样本无 waypoints)→ well_posed=None:不进 well-posed 子集,
        # 单独计数 —— 缺数据不臆造,也不静默当 ill-posed。
        well_posed = [
            (None if r is None else bool(float(r) >= WELL_POSED_MIN_NET_RATIO))
            for r in net_ratios
        ]
    _angular_ps = [row.get("Angular_Error") for row in per_sample_rows] \
        if per_sample_rows else [None] * len(gen_videos)

    def _wellposed_block(idx):
        """块 idx 上的 well-posed 角误差统计(复用 overall 的逐样本角误差 ——
        同一估计器/口径,零额外光流)。"""
        sub = [i for i in idx if well_posed[i] is True]
        errs = [_angular_ps[i] for i in sub if _angular_ps[i] is not None]
        return {
            "Angular_Error_WellPosed_Mean": float(np.mean(errs)) if errs else None,
            "Angular_Error_WellPosed_Median": float(np.median(errs)) if errs else None,
            "Angular_Error_WellPosed_Std": float(np.std(errs)) if errs else None,
            "Angular_Error_WellPosed_Count": len(errs),
            "WellPosed_N": len(sub),
        }

    if well_posed_available:
        results["overall"].update(_wellposed_block(range(len(gen_videos))))
        results["overall"]["WellPosed_Min_Net_Ratio"] = WELL_POSED_MIN_NET_RATIO
        results["overall"]["WellPosed_Ratio_Missing"] = sum(
            1 for w in well_posed if w is None)
        logger.info(
            f"well-posed 双栏: 门限 {WELL_POSED_MIN_NET_RATIO}(=Track B 净方向门控),"
            f"well-posed {results['overall']['WellPosed_N']}/{len(gen_videos)},"
            f"ratio 缺失 {results['overall']['WellPosed_Ratio_Missing']}")

    # 落盘:一个聚合文件(不是每样本一个文件 —— rx40 inode 紧张)
    if per_sample_rows:
        scores = {
            "schema_version": 2,
            "note": ("逐样本指标, 与 generated/<sample_id>.npy 一一对应。"
                     "FID/FVD 为分布级指标, 无逐样本值。"
                     "Angular_Error=null 表示该样本无方向词或光流不可靠(未参与角误差统计)。"
                     "well_posed: GT 命令轨迹净位移比 >= 门限(Track B 同口径);"
                     "null = ratio 不可得(旧目录/无 waypoints)。"),
            "per_sample": [
                dict(sample_id=sid, scene=sc, direction=dr,
                     gt_net_displacement_ratio=(
                         None if nr is _RATIO_KEY_ABSENT else nr),
                     well_posed=wp, **row)
                for sid, sc, dr, nr, wp, row in zip(
                    sample_ids, scenes, directions, net_ratios, well_posed,
                    per_sample_rows)
            ],
        }
        scores_path = Path(output_dir) / "scores.json"
        with open(str(scores_path), "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"逐样本指标已保存: {scores_path} ({len(per_sample_rows)} 条)")

    # 按场景分项
    scene_set = sorted(set(scenes))
    for scene in scene_set:
        idx = [i for i, s in enumerate(scenes) if s == scene]
        if len(idx) < 2:
            continue
        s_gen = [gen_videos[i] for i in idx]
        s_gt = [gt_videos[i] for i in idx]
        s_angles = [target_angles[i] for i in idx]
        logger.info(f"计算 {scene} 场景指标 ({len(idx)} 条)...")
        results[scene] = compute_all_metrics(
            s_gen, s_gt,
            target_angles=s_angles if any(a is not None for a in s_angles) else None,
            device=device, **_ANGKW,
        )
        results[scene]["num_samples"] = len(idx)
        if well_posed_available:
            results[scene].update(_wellposed_block(idx))

    # 按方向分项
    dir_set = sorted([d for d in set(directions) if d is not None])
    for d in dir_set:
        idx = [i for i, dd in enumerate(directions) if dd == d]
        if len(idx) < 2:
            continue
        d_gen = [gen_videos[i] for i in idx]
        d_gt = [gt_videos[i] for i in idx]
        d_angles = [target_angles[i] for i in idx]

        logger.info(f"计算方向 {d} 指标 ({len(idx)} 条)...")
        results[f"dir_{d}"] = compute_all_metrics(
            d_gen, d_gt,
            target_angles=d_angles if any(a is not None for a in d_angles) else None,
            device=device, **_ANGKW,
        )
        results[f"dir_{d}"]["num_samples"] = len(idx)
        if well_posed_available:
            results[f"dir_{d}"].update(_wellposed_block(idx))

    # 方向 x 场景交叉分项
    for d in dir_set:
        for scene in scene_set:
            idx = [i for i in range(len(gen_videos))
                   if directions[i] == d and scenes[i] == scene]
            if len(idx) < 2:
                continue
            x_gen = [gen_videos[i] for i in idx]
            x_gt = [gt_videos[i] for i in idx]
            x_angles = [target_angles[i] for i in idx]

            logger.info(f"计算 {d}x{scene} 指标 ({len(idx)} 条)...")
            results[f"dir_{d}_{scene}"] = compute_all_metrics(
                x_gen, x_gt,
                target_angles=x_angles if any(a is not None for a in x_angles) else None,
                device=device, **_ANGKW,
            )
            results[f"dir_{d}_{scene}"]["num_samples"] = len(idx)
            if well_posed_available:
                results[f"dir_{d}_{scene}"].update(_wellposed_block(idx))

    return results


def save_results(results: Dict, output_dir: str):
    """保存结果为 JSON 和 CSV"""
    out = Path(output_dir)

    # JSON
    json_path = out / "eval_results.json"
    with open(str(json_path), "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"结果已保存: {json_path}")

    # CSV (方便粘贴到论文)
    csv_path = out / "eval_results_table.csv"
    with open(str(csv_path), "w") as f:
        # 表头([v2.6] 角误差双栏:全样本 + well-posed 子集,各带 n)
        metrics = ["SSIM", "PSNR", "LPIPS", "FID", "FVD",
                    "Angular_Error_Mean", "Angular_Error_WellPosed_Mean",
                    "Angular_Error_WellPosed_Count", "num_samples"]
        f.write("Scene," + ",".join(metrics) + "\n")
        for scene, vals in results.items():
            row = [scene]
            for m in metrics:
                v = vals.get(m, "")
                if isinstance(v, float):
                    row.append(f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}")
                else:
                    row.append(str(v))
            f.write(",".join(row) + "\n")
    logger.info(f"CSV 已保存: {csv_path}")

    # 终端打印
    print(f"\n{'='*70}")
    print("RSWM 评估结果")
    print(f"{'='*70}")
    for scene, vals in results.items():
        print(f"\n--- {scene} (n={vals.get('num_samples', '?')}) ---")
        for k, v in vals.items():
            if k == "num_samples":
                continue
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.4f}")
            else:
                print(f"  {k:25s}: {v}")
    print(f"{'='*70}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="RSWM 评估 Pipeline")

    # 推理参数
    parser.add_argument("--checkpoint_dir", type=str, help="基础模型目录")
    parser.add_argument("--lora_path", type=str, help="LoRA checkpoint 目录")
    parser.add_argument("--data_path", type=str, help="HDF5 数据集路径")
    parser.add_argument("--split_file", type=str, help="train_val_split.json 路径")
    parser.add_argument("--split_name", choices=("val", "test"), default="val",
                        help="默认 val；正式 paper test 必须显式选择 test")
    parser.add_argument("--output_dir", type=str, default="eval_output", help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None, help="最大评估样本数")
    parser.add_argument("--frame_num", type=int, default=33, help="生成帧数")
    parser.add_argument("--size", type=str, default="480*832", help="视频尺寸")
    parser.add_argument("--t5_cpu", action="store_true", default=True)
    parser.add_argument("--no_t5_cpu", dest="t5_cpu", action="store_false")
    parser.add_argument("--inactive_model_cpu", action="store_true", default=True,
                        help="按 timestep 切换 14B（80GB 默认 ON）")
    parser.add_argument("--no_inactive_model_cpu", dest="inactive_model_cpu",
                        action="store_false")
    parser.add_argument("--device", type=str, default="cuda",
                        help="逻辑设备；物理卡由 CUDA_VISIBLE_DEVICES 选择")
    parser.add_argument("--no_framewise", action="store_true", help="Use cumulative poses instead of framewise for Plücker embeddings")
    parser.add_argument("--gen_seed", type=int, default=42,
                        help="扩散采样种子(多采样种子评估用;训练种子在 train.py --seed)")

    # 仅计算指标
    parser.add_argument("--metrics_only", action="store_true",
                        help="跳过推理，仅从已有结果计算指标")

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.metrics_only:
        # [v2.9.2/v2.9.3] --metrics_only 是身份门的另一扇门:它不走推理,但
        # compute_metrics 仍会用当前环境给 overall 盖 profile/pose_mode/headline。
        from run_manifest import assert_metrics_only_ok, summarize
        _m = assert_metrics_only_ok(args.output_dir, ctx="evaluate --metrics_only")
        logger.info(f"[manifest] --metrics_only 口径一致 — {summarize(_m)}")

    if not args.metrics_only:
        # Step 1: 推理
        assert args.checkpoint_dir, "--checkpoint_dir 必须指定"
        assert args.lora_path, "--lora_path 必须指定"
        assert args.data_path, "--data_path 必须指定"
        assert args.split_file, "--split_file 必须指定"

        val_indices = load_split_indices(args.split_file, args.split_name)
        logger.info(f"{args.split_name} split indices: {len(val_indices)} 条")

        samples = load_val_samples(args.data_path, val_indices, args.max_samples)

        # [v2.9.2] 身份门要的数据/split 路径经环境传入(该函数签名里没有它们,
        # 加形参会牵动所有调用点 —— 这里用最小改动,且两者都在 manifest 里
        # 以 SHA 落盘,路径本身只作说明)。
        os.environ["RSWM_MANIFEST_DATA_H5"] = str(args.data_path)
        os.environ["RSWM_MANIFEST_SPLIT"] = str(args.split_file)
        run_inference_batch(
            samples=samples,
            checkpoint_dir=args.checkpoint_dir,
            lora_path=args.lora_path,
            output_dir=args.output_dir,
            device=args.device,
            frame_num=args.frame_num,
            size=args.size,
            t5_cpu=args.t5_cpu,
            framewise=not args.no_framewise,
            inactive_model_cpu=args.inactive_model_cpu,
            gen_seed=args.gen_seed,
            split_name=args.split_name,
        )

    # Step 2: 计算指标
    logger.info("开始计算指标...")
    results = compute_metrics(args.output_dir, device=args.device)
    # [v2.7.1 F1/P0] 空结果 = 没有任何样本可评 —— 判失败,不写"成功"假象
    if not results:
        logger.error("评估产出为空(compute_metrics 返回空 dict)—— 非零退出")
        sys.exit(1)
    save_results(results, args.output_dir)

    logger.info("评估完成!")


if __name__ == "__main__":
    main()
