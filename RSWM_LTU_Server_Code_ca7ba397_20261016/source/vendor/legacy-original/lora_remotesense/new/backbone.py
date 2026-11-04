#!/usr/bin/env python3
# =============================================================================
# backbone.py — 生成骨干的可插拔协议 + 两个参考实现
#
# 【为什么要有这个文件】
# 到 v2.9.x 为止,整条管线把"生成器"钉死成一个具体实现:
#   train.py:236 / evaluate.py:384 / evaluate_baseline.py:171 /
#   generate_samples.py:183 / inference.py:406 —— 五处各写一遍
#   `WAN_CONFIGS["i2v-A14B"]`,适配方法只有 LoRA 一种。
# 后果有两条,严重程度不同:
#   (1) 科学上:headline 结论"方向控制不迁移"只在【1 个骨干 × 1 种适配】上测过。
#       n=1 的否定结论无法区分"这个能力普遍缺失"与"这个 checkpoint 没学会"。
#   (2) 工程上:别人拿到这个包跑不了自己的模型 —— 而 RSWM-Bench 的主贡献
#       正是"benchmark"。一个只能评一个模型的 benchmark 不是 benchmark。
#
# 本文件只做一件事:把"生成一段视频"这个动作从 Wan 的具体调用里剥出来,
# 定义成一个**最小、可被第三方实现**的协议,并给出两个参考实现证明它可插拔。
#
# 【设计上不显然的几个取舍,以及为什么】
#
# A. 为什么 pose_cond 是显式参数,而不是继续读 RSWM_POSE_MODE 环境变量?
#    现状里"模型有没有真的收到位姿"这件事由两个互不相干的东西共同决定:
#      * 环境变量 RSWM_POSE_MODE(inference.py:246 读它,drop 时把 action_path
#        置 None);
#      * 调用方有没有传 action_path(不传就没有位姿,而且**不报错**)。
#    也就是说 `RSWM_POSE_MODE=native` 但调用方忘了传 action_path 时,跑出来的
#    是 text-only 结果,却会被 evaluate.py:494 写上 pose_mode="native",
#    再被 run_manifest 认证成合格 headline —— 全程静默。
#    协议把它变成一个**必须由调用方在参数里表态**的东西,并在 WanBackbone 里
#    对环境变量做交叉校验(见 assert_pose_cond_matches_mode),两者矛盾即硬失败。
#
# B. 为什么 pose_cond=None 必须是合法输入,而不是"传个单位阵"?
#    三态里 identity 就是"单位阵仍走 Plücker 通路"的静止相机诊断臂,drop 才是
#    真 text-only。如果协议只能靠"传单位阵"表达无位姿,drop 与 identity 在
#    API 层就分不开了 —— 而这两个臂的结论含义完全不同(见 run_manifest.py:121)。
#    所以:pose_cond=None ⟺ drop;pose_cond=单位阵 ⟺ identity。
#
# C. 为什么帧数契约由本文件强制,而不是继续依赖 inference.py 里的隐式钳制?
#    inference.py:188 有一段 `frame_num = min(frame_num, ((len(c2ws)-1)//4)*4+1)`,
#    它**只在 action_path 非 None 时**执行。v2.8.2 N6 的注释明确记着:drop 臂
#    必须先让这段钳制跑完、再把 action_path 置 None,否则 drop 臂的 T 会与
#    native/identity 臂不同,而 FVD/FID 照样出数、只是不可比。
#    这是一个"靠代码行顺序维持的不变量",对第三方骨干完全不可移植。
#    协议改成显式契约:num_frames 必须 ≡1 (mod 4),len(pose_cond) 必须 == num_frames,
#    返回的 T 必须 == num_frames。三条都由 check_output_contract / WanBackbone
#    硬校验 —— 隐式钳制退化成恒等操作,不变量不再依赖行序。
#
# D. 为什么返回 (T,H,W,3) uint8 而不是 torch 张量?
#    下游 metrics.compute_all_metrics 收的就是 (T,H,W,C) uint8 的 numpy list
#    (metrics.py:416),evaluate.py:451-452 也正是把 Wan 的 (C,T,H,W)/[-1,1]
#    转成它。用 numpy 做协议边界,第三方骨干不必依赖 torch,CI 也能跑。
#
# E. 为什么要有能力声明(BackboneCapabilities)?
#    因为"这个骨干不支持位姿条件"不是一个错误,而是一个**结论的适用范围**。
#    不支持位姿的骨干只能跑 drop 臂;拿它的数字去填 native↔drop 的 Δ 表是
#    把"模型没有这个输入"读成"模型不听话"。能力声明让评估侧能自动拒绝
#    不适用的臂,而不是靠人记住。
#
# F. 为什么本模块顶层不 import torch?
#    CI 是纯 CPU、不装 torch 的(见 .github/workflows/ci.yml 第 19-23 行)。
#    TrivialShiftBackbone 必须能在那里真跑,所以 torch / PIL / wan / inference
#    一律延迟到 WanBackbone.load() 内部再 import。
#
# 【本文件不做的事】
#   不改 train.py / evaluate.py / inference.py 的任何一行。把这套协议真正接上
#   评估与训练入口是主协调者的接线工作,清单见 docs/ADDING_A_BACKBONE.md 末节。
#
# 自检(纯 CPU,无 torch 亦可):
#   python lora_remotesense/new/backbone.py --self-test
# =============================================================================
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent

# 三态位姿口径的唯一取值集合。与 inference.py:247 / evaluate.py:181 /
# run_manifest.py 保持同一套字面量(它们各自 raise,这里只做集合定义)。
POSE_MODES: Tuple[str, ...] = ("native", "identity", "drop")

# 某个臂是否要求"位姿张量真的进模型"。
# identity 是 True —— 它是【中和位姿仍走 Plücker 通路】的静止相机诊断臂,
# 位姿照样被编码进模型,只是内容是单位阵。把它当 text-only 是 v2.9 拍板
# 明确否掉的口径(common.sh 顶部 profile 段)。
_ARM_NEEDS_POSE: Dict[str, bool] = {
    "native": True,
    "identity": True,
    "drop": False,
}


# =============================================================================
# 尺寸 / 位姿条件 / 能力声明
# =============================================================================

def parse_size(size: str) -> Tuple[int, int]:
    """`"480*832"` → `(H, W) = (480, 832)`。

    分隔符是 `*` 且【高在前】—— 这不是随手定的,是 wan.configs.SIZE_CONFIGS
    的既有键格式('480*832': (480, 832))。协议沿用同一份字面量,免得第三方
    骨干按 "宽*高" 理解后无声地转置输出(转置后的视频照样能算 FID,只是
    数字全错)。
    """
    if not isinstance(size, str):
        raise TypeError(f"size 必须是形如 '480*832' 的字符串,收到 {type(size).__name__}")
    parts = size.split("*")
    if len(parts) != 2:
        raise ValueError(f"size={size!r} 格式非法,应为 'H*W'(例 '480*832')")
    try:
        h, w = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"size={size!r} 的两段必须是整数") from None
    if h <= 0 or w <= 0:
        raise ValueError(f"size={size!r} 必须为正")
    return h, w


@dataclass(frozen=True)
class PoseCond:
    """一条轨迹的位姿条件。字段与 benchmark H5 逐字节同构。

    c2w:        (T, 4, 4) camera-to-world。建库器的约定(见
                lora_training/camera_trajectory.py:112-116)是:旋转恒为单位阵
                (天底视角固定朝向),平移列 tx=+图像右(东)、ty=+图像下(南)、
                tz=0,单位 = 1 个窗口边长。**滑窗向东移 ⟹ 画面内容向西流**,
                这条反号关系是本 benchmark 全部方向指标的基准约定
                (evaluate.py:156-166 的 +180° 就是它)。
    intrinsics: (T, 4) = [fx, fy, cx, cy],像素口径按 H5 帧尺寸
                (canonical 480x832;inference.py:261-262 把 height_org/width_org
                硬编码成 480/832)。不需要内参的骨干可以直接忽略这一项。

    为什么用 dataclass 而不是裸 dict:两个字段的语义(尤其是 c2w 的轴向与
    反号关系)是最容易被第三方接错的地方,一个有 docstring 的类型比
    `sample["poses"]` 更难接错;同时 __post_init__ 把形状校验做成硬门。
    """

    c2w: np.ndarray
    intrinsics: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        c2w = np.asarray(self.c2w, dtype=np.float64)
        if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
            raise ValueError(f"PoseCond.c2w 形状必须是 (T,4,4),收到 {c2w.shape}")
        if len(c2w) == 0:
            raise ValueError("PoseCond.c2w 为空")
        object.__setattr__(self, "c2w", c2w)
        if self.intrinsics is not None:
            k = np.asarray(self.intrinsics, dtype=np.float64)
            if k.ndim != 2 or k.shape[1] != 4:
                raise ValueError(
                    f"PoseCond.intrinsics 形状必须是 (T,4)=[fx,fy,cx,cy],收到 {k.shape}")
            if len(k) != len(c2w):
                raise ValueError(
                    f"PoseCond 的 c2w({len(c2w)} 帧)与 intrinsics({len(k)} 帧)不等长")
            object.__setattr__(self, "intrinsics", k)

    def __len__(self) -> int:
        return len(self.c2w)

    @property
    def is_identity(self) -> bool:
        """全部位姿都是单位阵 ⟺ 这是 identity(静止相机诊断)臂的位姿。

        容差 1e-6 与 evaluate.py:234 同口径:neutralize_poses.py 写的是精确
        单位阵,1e-6 只吸浮点往返;native 轨迹的平移步长 ~1e-2 量级,差 4 个
        数量级,不会误判。
        """
        return bool(np.allclose(self.c2w, np.eye(4), atol=1e-6))

    @property
    def implied_pose_mode(self) -> str:
        """这份位姿本身对应哪个臂(identity / native)。drop 臂没有 PoseCond。"""
        return "identity" if self.is_identity else "native"

    @classmethod
    def from_sample(cls, sample: Mapping[str, Any]) -> "PoseCond":
        """从 evaluate.load_val_samples 产出的样本 dict 构造。

        键名 poses/intrinsics 与 H5 数据集名一致(evaluate.py:213-214)。
        """
        if "poses" not in sample:
            raise KeyError("样本缺 'poses' —— 无法构造 PoseCond")
        return cls(c2w=sample["poses"], intrinsics=sample.get("intrinsics"))


def coerce_pose_cond(pose_cond: Any) -> Optional[PoseCond]:
    """把调用方可能给的几种形式统一成 PoseCond|None。

    接受:None / PoseCond / {"poses"|"c2w": ..., "intrinsics": ...} /
    裸 (T,4,4) ndarray。**不接受**空 list、全零数组这类"看起来像没有位姿"的
    东西 —— 它们会走进 PoseCond 的形状校验并硬失败,因为"我以为我传了位姿"
    正是本协议要消灭的静默模式。
    """
    if pose_cond is None:
        return None
    if isinstance(pose_cond, PoseCond):
        return pose_cond
    if isinstance(pose_cond, Mapping):
        c2w = pose_cond.get("c2w", pose_cond.get("poses"))
        if c2w is None:
            raise KeyError("pose_cond 映射里既没有 'c2w' 也没有 'poses'")
        return PoseCond(c2w=c2w, intrinsics=pose_cond.get("intrinsics"))
    return PoseCond(c2w=pose_cond)


@dataclass(frozen=True)
class BackboneCapabilities:
    """骨干的能力声明 —— 决定【哪些臂对它有意义】。

    这不是元数据装饰:评估侧据此拒绝不适用的臂。一个不支持位姿条件的骨干
    在 native 臂上的分数不是"它不听话",而是"它根本没有这个输入通道",两者
    在 Δ 表里长得一模一样,只能靠声明区分。
    """

    # 位姿条件是否真的接进模型。False = 该骨干只能跑 drop(text-only)臂。
    supports_pose_cond: bool
    # 位姿的表示形式;不支持时为 None。第三方若用别的表示(如 6-DoF ray map、
    # 相机内外参 JSON),在这里写清楚,读者才知道 PoseCond 是怎么被消费的。
    pose_cond_format: Optional[str] = None
    # 生成帧数的整除约束。"4n+1" 是 Wan 系 VAE 时间下采样的硬约束,也是本
    # benchmark 全部 H5 的产物契约(CI 的 D 步骤断言 T≡1 mod 4)。
    # 骨干若无此约束填 "any" —— 但评估侧仍会要求它按请求的 T 返回。
    frame_count_rule: str = "4n+1"
    # 同一 seed 是否给出同一结果 / 不同 seed 是否真的给出不同结果。
    # False 意味着"多种子聚合"在这个骨干上恒得 0 方差 —— 那不是稳定性证据,
    # aggregate_seeds 不该把它当种子方差报出来。
    stochastic: bool = True
    # 适配方法:none(零样本) / lora / full-ft / prefix / adapter / ...
    # 这是 n=1 的**第二个轴**(第一个是骨干本身),必须能被 manifest 记住。
    adaptation: str = "none"
    trainable_params: Optional[int] = None
    # 这个骨干实际支持的生成尺寸(size 字符串);空 tuple = 任意尺寸。
    supported_sizes: Tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.supports_pose_cond and not self.pose_cond_format:
            raise ValueError(
                "声明 supports_pose_cond=True 就必须写明 pose_cond_format —— "
                "否则第三方无从知道 PoseCond 被怎么消费(是 Plücker?ray map?"
                "还是只用了平移列?),复现无从谈起。")
        if not self.supports_pose_cond and self.pose_cond_format:
            raise ValueError("supports_pose_cond=False 却给了 pose_cond_format,自相矛盾")

    @property
    def applicable_pose_modes(self) -> Tuple[str, ...]:
        """该骨干上有意义的臂。不支持位姿 ⟹ 只有 drop。"""
        if self.supports_pose_cond:
            return POSE_MODES
        return ("drop",)

    def assert_pose_mode_supported(self, pose_mode: str, *, who: str = "backbone") -> None:
        if pose_mode not in POSE_MODES:
            raise ValueError(f"pose_mode={pose_mode!r} 非法(native|identity|drop)")
        if pose_mode not in self.applicable_pose_modes:
            raise ValueError(
                f"{who} 声明 supports_pose_cond=False,只能跑 {self.applicable_pose_modes} 臂;"
                f"当前要求 pose_mode={pose_mode!r}。\n"
                f"  这不是可以忽略的告警:该骨干没有位姿输入通道,它在 native/identity "
                f"臂上的分数无法与有位姿通道的骨干比较,也不能填进 native↔drop 的 Δ 表。\n"
                f"  正确做法:只用它跑 drop 臂,并在结果里按 "
                f"docs/ADDING_A_BACKBONE.md《结果如何声明》一节标注。")

    def as_manifest_fields(self) -> Dict[str, Any]:
        """给 run_manifest.build_manifest(extra=...) 用的扁平字段。"""
        return {
            "backbone_supports_pose_cond": bool(self.supports_pose_cond),
            "backbone_pose_cond_format": self.pose_cond_format,
            "backbone_frame_count_rule": self.frame_count_rule,
            "backbone_stochastic": bool(self.stochastic),
            "backbone_adaptation": self.adaptation,
            "backbone_trainable_params": self.trainable_params,
        }


# =============================================================================
# 协议本体
# =============================================================================

class VideoBackbone(Protocol):
    """一个可插进 RSWM-Bench 的视频生成骨干。

    实现者只需要三样东西:`name`、`capabilities`、`generate`。

    generate 的契约(逐条都被 check_output_contract 硬校验):
      入参
        first_frame : (H0, W0, 3) uint8,RGB,值域 [0,255]。就是 H5 里的
                      `image`(evaluate.py:211)。H0/W0 不保证等于 size。
        prompt      : str。方向词的措辞见 evaluate.parse_direction_word_from_prompt。
        num_frames  : int,**必须** ≡1 (mod 4)(见文件头 C 段)。
        size        : "H*W" 字符串,目标生成分辨率。
        pose_cond   : PoseCond 或 None。**None 是合法输入**,含义是
                      pose_mode=drop(真 text-only,位姿通路整条不接);
                      全单位阵的 PoseCond 含义是 pose_mode=identity
                      (静止相机诊断臂,位姿仍走条件通路)。两者不可互换。
        seed        : int 或 None。None = 用骨干自己的默认种子(不推荐,
                      评估入口应当永远显式传)。
      返回
        (num_frames, H, W, 3) 的 uint8 ndarray,RGB,值域 [0,255],
        其中 (H, W) = parse_size(size)。
        **不允许**返回别的帧数或别的分辨率:metrics.py:103-104 会把 gen 与 gt
        裁到两者的最小形状再算,尺寸不符不会报错,只会静默改变被评的内容。
    """

    name: str
    capabilities: BackboneCapabilities

    def generate(
        self,
        *,
        first_frame: np.ndarray,
        prompt: str,
        num_frames: int,
        size: str,
        pose_cond: Optional[PoseCond] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        ...


def validate_backbone(obj: Any, *, who: str = "backbone") -> None:
    """结构化校验一个对象是否满足 VideoBackbone —— 报可执行的错,而不是 False。

    为什么不直接用 isinstance(obj, VideoBackbone):Protocol 的 runtime 检查
    只看成员在不在,不看签名,而这里最容易接错的恰恰是签名(位置参数 vs
    关键字参数、少了 pose_cond 默认值)。
    """
    missing = [a for a in ("name", "capabilities", "generate") if not hasattr(obj, a)]
    if missing:
        raise TypeError(f"{who} 缺少 VideoBackbone 要求的属性: {missing}")
    if not isinstance(obj.name, str) or not obj.name:
        raise TypeError(f"{who}.name 必须是非空字符串")
    if not isinstance(obj.capabilities, BackboneCapabilities):
        raise TypeError(f"{who}.capabilities 必须是 BackboneCapabilities 实例")
    if not callable(obj.generate):
        raise TypeError(f"{who}.generate 必须可调用")
    import inspect as _inspect

    sig = _inspect.signature(obj.generate)
    required = ("first_frame", "prompt", "num_frames", "size", "pose_cond", "seed")
    params = sig.parameters
    has_var_kw = any(p.kind is _inspect.Parameter.VAR_KEYWORD for p in params.values())
    if not has_var_kw:
        absent = [r for r in required if r not in params]
        if absent:
            raise TypeError(
                f"{who}.generate 缺少关键字参数 {absent};完整签名应为 "
                f"generate(*, first_frame, prompt, num_frames, size, "
                f"pose_cond=None, seed=None)")
        for r in ("pose_cond", "seed"):
            if params[r].default is _inspect.Parameter.empty:
                raise TypeError(
                    f"{who}.generate 的 {r} 必须有默认值 None —— pose_cond=None "
                    f"是 drop 臂的合法输入,不是缺参数")


# =============================================================================
# 契约校验(输入侧 + 输出侧),两个参考实现共用,第三方也应当调
# =============================================================================

def check_frame_count(num_frames: int, rule: str = "4n+1", *, who: str = "backbone") -> int:
    """帧数契约。见文件头 C 段:这是全管线最容易被无声改坏的不变量。"""
    n = int(num_frames)
    if n <= 0:
        raise ValueError(f"{who}: num_frames 必须为正,收到 {num_frames!r}")
    if rule == "4n+1":
        if n % 4 != 1:
            raise ValueError(
                f"{who}: num_frames={n} 不满足 T≡1 (mod 4)。\n"
                f"  benchmark 的帧数契约是 steps=9 时 T = 9+(K-1)*8 "
                f"(K=1..4 → 9/17/25/33),全部 ≡1 (mod 4);Wan 系 VAE 的时间\n"
                f"  下采样也要求这一点。传入非法 T 会被上游静默钳制成另一个 T,"
                f"于是不同臂的 FVD/FID 在不同帧数上算 —— 照样出数,只是不可比。")
    elif rule != "any":
        raise ValueError(f"{who}: 未知的 frame_count_rule={rule!r}(4n+1|any)")
    return n


def check_first_frame(first_frame: Any, *, who: str = "backbone") -> np.ndarray:
    arr = np.asarray(first_frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"{who}: first_frame 形状必须是 (H,W,3),收到 {arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(
            f"{who}: first_frame dtype 必须是 uint8(值域 [0,255]),收到 {arr.dtype}。"
            f"协议边界不做隐式转换 —— float [0,1] 与 float [0,255] 只差一个"
            f"约定,静默 clip 会把两者都变成看似合理的图。")
    return arr


def check_output_contract(video: Any, num_frames: int, size: str, *,
                          who: str = "backbone") -> np.ndarray:
    """返回值契约的**单一实现**,两个参考实现与第三方骨干共用。

    为什么要单独抽出来:metrics.py:103-104 对 gen/gt 取 min 形状再算,
    形状不符不报错。也就是说返回错尺寸/错帧数唯一的表现就是"分数变了",
    没有任何异常。契约必须在协议边界上被主动检查。
    """
    arr = np.asarray(video)
    h, w = parse_size(size)
    if arr.ndim != 4 or arr.shape[3] != 3:
        raise ValueError(f"{who}: 返回值形状必须是 (T,H,W,3),收到 {arr.shape}")
    if arr.dtype != np.uint8:
        raise TypeError(f"{who}: 返回值 dtype 必须是 uint8,收到 {arr.dtype}")
    if arr.shape[0] != int(num_frames):
        raise ValueError(
            f"{who}: 请求 {num_frames} 帧,返回 {arr.shape[0]} 帧。"
            f"帧数必须逐帧对齐,下游按 min 长度裁剪不会报错但会改变被评内容。")
    if arr.shape[1] != h or arr.shape[2] != w:
        raise ValueError(
            f"{who}: 请求 size={size}(H={h},W={w}),返回 "
            f"(H={arr.shape[1]},W={arr.shape[2]})。"
            f"注意 size 是 '高*宽' —— 若你按 '宽*高' 理解,输出会被转置,"
            f"FID/FVD 照算不误。")
    return arr


def assert_pose_cond_matches_mode(pose_cond: Optional[PoseCond], pose_mode: str,
                                  *, who: str = "backbone") -> None:
    """pose_cond 的有无/内容 必须与 pose_mode 三态严格一一对应。

    这是本协议最重要的一道门。旧路径里"模型有没有真的收到位姿"由环境变量和
    调用方是否传 action_path 共同决定,而两者不一致时**没有任何报错**:
      * RSWM_POSE_MODE=native + 调用方没传 action_path → 跑出 text-only 结果,
        却被 evaluate.py:494 写上 pose_mode="native";
      * RSWM_POSE_MODE=drop  + 调用方传了 action_path → inference.py:254 会
        把它悄悄丢掉(这一次是对的,但调用方并不知道自己传了个被忽略的东西)。
    """
    if pose_mode not in POSE_MODES:
        raise ValueError(f"{who}: pose_mode={pose_mode!r} 非法(native|identity|drop)")
    needs_pose = _ARM_NEEDS_POSE[pose_mode]
    if needs_pose and pose_cond is None:
        raise ValueError(
            f"{who}: pose_mode={pose_mode} 要求把位姿送进模型,但 pose_cond=None。\n"
            f"  identity 臂不是 text-only —— 它是【中和位姿仍走 Plücker 通路】的"
            f"静止相机诊断臂,位姿张量照样进模型(全单位阵)。\n"
            f"  真 text-only 请用 pose_mode=drop + pose_cond=None。")
    if not needs_pose and pose_cond is not None:
        raise ValueError(
            f"{who}: pose_mode=drop 意为位姿通路整条不接,但收到了 pose_cond。\n"
            f"  若你其实想跑静止相机诊断臂,请显式用 pose_mode=identity。")
    if pose_cond is not None:
        implied = pose_cond.implied_pose_mode
        if implied != pose_mode:
            raise ValueError(
                f"{who}: pose_mode={pose_mode} 与 pose_cond 的内容不符 —— 传入的位姿"
                f"看起来是 {implied}({'全单位阵' if implied == 'identity' else '非单位阵'})。\n"
                f"  native 臂喂单位阵 = 几何通道实际全零却被当几何臂统计;\n"
                f"  identity 臂喂真实轨迹 = 静止相机诊断臂的语义直接不成立。\n"
                f"  (与 evaluate.py:250-258 的两条硬门同义,在骨干边界上再守一次。)")


def current_pose_mode() -> str:
    """当前环境声明的臂。默认 native 与 inference.py:246 / evaluate.py:180 一致。"""
    pm = os.environ.get("RSWM_POSE_MODE", "native").strip().lower()
    if pm not in POSE_MODES:
        raise ValueError(f"RSWM_POSE_MODE={pm!r} 非法(native|identity|drop)")
    return pm


# =============================================================================
# 参考实现 1:WanBackbone —— 现有 inference.py 的薄封装
# =============================================================================

class WanBackbone:
    """把现有 `inference.generate_video` 包成 VideoBackbone。

    **完全不改 inference.py**:只 import 它的 load_lora_model / generate_video
    (基模型走 evaluate_baseline.load_base_model),把 PoseCond 落成临时目录里的
    poses.npy / intrinsics.npy —— 这正是 evaluate.py:426-428 现在的做法。

    默认超参逐个对齐 evaluate.py 现在的调用,使"换成 WanBackbone"是
    **数值恒等**的改造,而不是顺手改了采样参数:
        sampling_steps=70 / shift=10.0 / guide_scale=(5.0,5.0)
        (evaluate.py:432-448 没传这三个,吃的是 inference.generate_video 的默认值)
        t5_cpu=True / inactive_model_cpu=True / framewise=True

    封装在两处比原调用**更严**,原因见文件头 C 段与 assert_pose_cond_matches_mode:
      1) num_frames 必须 ≡1 (mod 4)。原路径靠 inference.py:188 的隐式钳制,
         而那段只在 action_path 非 None 时执行;显式契约让 drop 臂不再依赖
         "先钳制再置 None" 的行序。
      2) len(pose_cond) 必须 == num_frames。这样上面那段钳制退化成恒等操作,
         三个臂的 T 一定相同。
    这两条都是**收紧**:在既有 canonical 数据(T∈{9,17,25,33},poses 与
    video_frames 等长)上恒成立,不改变任何已有数字。
    """

    name = "wan2.2-i2v-A14B"

    def __init__(
        self,
        checkpoint_dir: str,
        lora_path: Optional[str] = None,
        *,
        device: str = "cuda",
        config_key: str = "i2v-A14B",
        sampling_steps: int = 70,
        shift: float = 10.0,
        guide_scale: float = 5.0,
        t5_cpu: bool = True,
        inactive_model_cpu: bool = True,
        framewise: bool = True,
        default_seed: int = 42,
        name: Optional[str] = None,
    ):
        self.checkpoint_dir = str(checkpoint_dir)
        self.lora_path = str(lora_path) if lora_path else None
        self.device = device
        self.config_key = config_key
        self.sampling_steps = int(sampling_steps)
        self.shift = float(shift)
        self.guide_scale = float(guide_scale)
        self.t5_cpu = bool(t5_cpu)
        self.inactive_model_cpu = bool(inactive_model_cpu)
        self.framewise = bool(framewise)
        self.default_seed = int(default_seed)
        if name:
            self.name = name
        self._loaded = False
        self._parts: Optional[Tuple[Any, Any, Any, Any]] = None
        self._config: Any = None

        self.capabilities = BackboneCapabilities(
            supports_pose_cond=True,
            # 说清楚位姿到底怎么被消费:c2w 先经 interpolate_camera_poses 采样到
            # 潜空间帧数,再 compute_relative_poses(framewise),最后 Plücker 嵌入
            # (inference.py:271-297)。第三方看这一行就知道自己的骨干是不是同一
            # 类条件方式。
            pose_cond_format=(
                "c2w(T,4,4)+intrinsics(T,4)[fx,fy,cx,cy] → interpolate→relative"
                f"({'framewise' if framewise else 'cumulative'})→Plücker embedding"),
            frame_count_rule="4n+1",
            stochastic=True,
            adaptation=("lora" if self.lora_path else "none"),
            supported_sizes=("720*1280", "1280*720", "480*832", "832*480"),
            notes=("intrinsics 按 480x832 像素口径解读 —— inference.py:261-262 把 "
                   "height_org/width_org 硬编码成 480/832,别的原始尺寸会被错误缩放。"),
        )

    # ---------------------------------------------------------------- 身份
    def checkpoint_id(self) -> str:
        """写进 run_manifest 的 checkpoint_id。

        为什么要把骨干名拼进去:run_manifest.IDENTITY_FIELDS 目前没有 "backbone"
        字段,两个不同骨干的产物在同一目录下会被判为"同源"从而被复用。在
        主协调者把 backbone 加进 IDENTITY_FIELDS 之前,checkpoint_id 是唯一
        已经参与身份比对、又能承载骨干信息的字段。
        """
        return f"{self.name}+{self.capabilities.adaptation}:{self.lora_path or 'base'}"

    def manifest_fields(self) -> Dict[str, Any]:
        d = {"backbone": self.name}
        d.update(self.capabilities.as_manifest_fields())
        d.update({
            "backbone_sampling_steps": self.sampling_steps,
            "backbone_shift": self.shift,
            "backbone_guide_scale": self.guide_scale,
            "backbone_framewise": self.framewise,
        })
        return d

    # ---------------------------------------------------------------- 加载
    def load(self) -> "WanBackbone":
        """真正把权重读进显存。延迟到这里(而不是 __init__)有两个原因:
        构造一个 backbone 对象来读它的 capabilities 不该要 80GB 显存;
        以及 torch / wan 的 import 本身在无 GPU 机器上就会失败。
        """
        if self._loaded:
            return self
        # inference.py 顶部自己会 sys.path.insert(HERE),但那要求 import 之前
        # HERE 已在路径上。以包路径(lora_remotesense.new.backbone)导入本模块时
        # 不成立,所以这里补一次。
        for p in (str(HERE), str(HERE.parent.parent)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import torch  # noqa: PLC0415 —— 见文件头 F 段:顶层不能 import
        from wan.configs import WAN_CONFIGS  # noqa: PLC0415

        if self.config_key not in WAN_CONFIGS:
            raise KeyError(
                f"WAN_CONFIGS 没有 {self.config_key!r};可选: {sorted(WAN_CONFIGS)}")
        self._config = WAN_CONFIGS[self.config_key]
        device_obj = torch.device(self.device)
        if self.lora_path:
            from inference import load_lora_model  # noqa: PLC0415
            parts = load_lora_model(
                checkpoint_dir=self.checkpoint_dir,
                lora_path=self.lora_path,
                device=device_obj,
                config=self._config,
                t5_cpu=self.t5_cpu,
                inactive_model_cpu=self.inactive_model_cpu,
            )
        else:
            # 基模型不走 inference.load_lora_model —— 后者在 LoRA 目录缺失时
            # 会硬失败(inference.py:119),那是 v2.7.1 F2 刻意加的"拒绝静默
            # 退化为基模型"。要基模型就得显式走 baseline 的加载函数。
            from evaluate_baseline import load_base_model  # noqa: PLC0415
            parts = load_base_model(
                checkpoint_dir=self.checkpoint_dir,
                device=device_obj,
                config=self._config,
                t5_cpu=self.t5_cpu,
                inactive_model_cpu=self.inactive_model_cpu,
            )
        self._parts = parts
        self._loaded = True
        return self

    # ---------------------------------------------------------------- 生成
    def generate(
        self,
        *,
        first_frame: np.ndarray,
        prompt: str,
        num_frames: int,
        size: str,
        pose_cond: Optional[PoseCond] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        who = f"WanBackbone({self.name})"
        first_frame = check_first_frame(first_frame, who=who)
        num_frames = check_frame_count(
            num_frames, self.capabilities.frame_count_rule, who=who)
        pose_cond = coerce_pose_cond(pose_cond)
        pose_mode = current_pose_mode()
        self.capabilities.assert_pose_mode_supported(pose_mode, who=who)
        assert_pose_cond_matches_mode(pose_cond, pose_mode, who=who)
        if pose_cond is not None and len(pose_cond) != num_frames:
            raise ValueError(
                f"{who}: pose_cond 有 {len(pose_cond)} 帧,请求生成 {num_frames} 帧。\n"
                f"  inference.py:188 会把 num_frames 静默钳到 "
                f"((len(poses)-1)//4)*4+1 —— 于是 drop 臂(不传位姿、不钳制)与 "
                f"native 臂的 T 可能不同,\n"
                f"  FVD/FID 照样出数但不可比(v2.8.2 N6 记录的正是这个故障)。"
                f"这里要求两者显式相等,让那段钳制退化成恒等操作。")
        if (self.capabilities.supported_sizes
                and size not in self.capabilities.supported_sizes):
            raise ValueError(
                f"{who}: size={size!r} 不在支持列表 {self.capabilities.supported_sizes}")

        self.load()
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
        from inference import generate_video  # noqa: PLC0415

        text_encoder, vae, low_model, high_model = self._parts
        pil_img = Image.fromarray(first_frame)
        used_seed = self.default_seed if seed is None else int(seed)

        # 位姿以 poses.npy/intrinsics.npy 落进临时目录 —— 这是 inference 现在
        # 唯一的位姿入口(action_path 目录),薄封装不改它。
        action_dir: Optional[str] = None
        try:
            if pose_cond is not None:
                if pose_cond.intrinsics is None:
                    raise ValueError(
                        f"{who}: Wan 的 Plücker 编码要内参,但 PoseCond.intrinsics=None。"
                        f"canonical H5 里有 intrinsics 数据集,请一起传。")
                action_dir = tempfile.mkdtemp(prefix="rswm_pose_")
                np.save(os.path.join(action_dir, "poses.npy"),
                        pose_cond.c2w.astype(np.float32))
                np.save(os.path.join(action_dir, "intrinsics.npy"),
                        pose_cond.intrinsics.astype(np.float32))
            video_tensor = generate_video(
                text_encoder=text_encoder,
                vae=vae,
                low_noise_model=low_model,
                high_noise_model=high_model,
                config=self._config,
                prompt=prompt,
                image=pil_img,
                device=torch.device(self.device),
                frame_num=num_frames,
                size=size,
                action_path=action_dir,
                seed=used_seed,
                sampling_steps=self.sampling_steps,
                shift=self.shift,
                guide_scale=(self.guide_scale, self.guide_scale),
                t5_cpu=self.t5_cpu,
                framewise=self.framewise,
                inactive_model_cpu=self.inactive_model_cpu,
            )
        finally:
            if action_dir:
                shutil.rmtree(action_dir, ignore_errors=True)

        video = wan_tensor_to_uint8(video_tensor)
        return check_output_contract(video, num_frames, size, who=who)


def wan_tensor_to_uint8(video_tensor: Any) -> np.ndarray:
    """(C,T,H,W) float [-1,1] → (T,H,W,3) uint8。

    与 evaluate.py:451-452 / evaluate_baseline.py:215-216 逐操作一致(clamp →
    permute → (x+1)*127.5 → byte)。抽成函数是因为那两处已经各写了一遍,
    第三处再抄一遍就会漂移;协议边界只认 uint8,转换必须只有一个定义。
    """
    x = video_tensor.clamp(-1, 1).permute(1, 2, 3, 0)
    x = ((x + 1) * 127.5).byte().cpu().numpy()
    return np.asarray(x)


# =============================================================================
# 参考实现 2:TrivialShiftBackbone —— 纯图像平移,无 GPU / 无权重
# =============================================================================

_CAMERA_ANGLE_BY_WORD: Dict[str, float] = {"E": 0.0, "N": 90.0, "W": 180.0, "S": 270.0}

# 画面内容的每帧位移,占帧宽的比例。
# 出处:metrics.py:272-274 记的实测 —— 480x832 生成分辨率下 GT 速度级内容位移
# 约 34 px/帧,34/832 ≈ 0.041。取 0.04 让本基线的运动幅度与 GT 同量级,从而
# 落在 Farnebäck 的可靠量程内(降到窗口分辨率 128x222 后约 8.9 px/帧)。
_DEFAULT_SHIFT_FRACTION = 0.04


def _default_direction_parser(prompt: str) -> Optional[str]:
    """从 prompt 取方向词。**优先复用仓库里的唯一真源**。

    evaluate.parse_direction_word_from_prompt 的 docstring 明确写着"方向词的
    匹配规则只在这里实现一次" —— 曾经因为内联复制一份只认旧措辞 "eastward",
    新措辞 "toward the east" 全部落 None,分方向统计静默整块丢失。
    所以这里不复制规则:能 import 就用它;不能(evaluate.py 顶层 import torch,
    纯 CPU 的 CI 里装不了)就**硬失败**并要求调用方显式注入 parser,
    而不是偷偷用一份可能已经漂移的副本。
    """
    for p in (str(HERE),):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        from evaluate import parse_direction_word_from_prompt  # noqa: PLC0415
    except Exception as e:  # ImportError / 依赖缺失
        raise RuntimeError(
            "无法 import evaluate.parse_direction_word_from_prompt(方向词解析的"
            f"唯一真源): {e!r}\n"
            "  evaluate.py 顶层 import torch/h5py,纯 CPU 环境下装不上。\n"
            "  请显式注入:TrivialShiftBackbone(direction_parser=<callable>),\n"
            "  且该 callable 必须与 evaluate.parse_direction_word_from_prompt 同规则"
            "(测试里用源码切片取真身,见 server_run/test_backbone_protocol.py)。"
        ) from e
    return parse_direction_word_from_prompt(prompt)


class TrivialShiftBackbone:
    """纯图像平移基线:把首帧按命令方向逐帧平移,不生成任何新内容。

    【它是干什么用的】
      1. 证明 VideoBackbone 协议真的可插拔 —— 一个连权重都没有的东西也能
         走完全部评估路径;
      2. 给 CI 一条无 GPU 的端到端通路(.github/workflows/ci.yml 是纯 CPU、
         连 torch 都不装的);
      3. 顺带是一个诚实的参照点:它对方向命令的服从度是"完美"的(平移量直接
         来自命令),但它不生成任何新内容(移出视野的区域只能靠边缘复制填)。
         于是"方向指标很好 + 画质指标很差"这一组合有了明确的基准形状。

    【它不是什么】
      不是论文里的 baseline。它没有生成能力,SSIM/PSNR 只反映"首帧有多像
      后续帧",与 validate_direction_metric.py 的 persistence 基线是近亲。
      任何把它的分数与真实模型并列的表都必须标注这一点。

    【方向约定】(接错就整表反号,所以写清楚)
      benchmark 把相机平移实现为静态大图上的滑窗:窗口向东移 ⟹ 画面内容向西流
      (evaluate.py:156-166)。所以:
        * 有 pose_cond 时,平移量 = -(c2w 平移列的增量) × 帧尺寸。
          c2w 的 tx 以窗口宽为单位(camera_trajectory.py:124),而输出帧就是
          那个窗口重采样的结果,故 1 个世界单位 = W 像素。
        * 无 pose_cond(drop 臂)时,只能从 prompt 的方向词推:
          内容角 = 相机角 + 180°,幅度用 shift_fraction × 帧宽。
        * prompt 里没有方向词、又没有 pose_cond 时,输出**静止视频**
          (全部帧 = 首帧)。这是诚实的"没收到任何命令",不是失败;
          此时 metrics.estimate_motion_direction 会因累积幅度过小返回 None。
    """

    name = "trivial-shift"

    def __init__(
        self,
        *,
        shift_fraction: float = _DEFAULT_SHIFT_FRACTION,
        use_pose_cond: bool = True,
        direction_parser: Optional[Callable[[str], Optional[str]]] = None,
        name: Optional[str] = None,
    ):
        """
        shift_fraction : 无 pose_cond 时,每帧内容位移 / 帧宽。默认 0.04,出处见
                         模块内 _DEFAULT_SHIFT_FRACTION 的注释。
        use_pose_cond  : True = 能力声明为"支持位姿"(平移量读 c2w);
                         False = 声明为**不支持位姿**,只能跑 drop 臂 ——
                         这正是"协议要能表达骨干不支持位姿"的可执行样例,
                         评估侧据此拒绝 native/identity 臂。
        direction_parser: 方向词解析器。默认懒加载 evaluate 里的唯一真源。
        """
        self.shift_fraction = float(shift_fraction)
        self.use_pose_cond = bool(use_pose_cond)
        self._direction_parser = direction_parser
        if name:
            self.name = name
        self.capabilities = BackboneCapabilities(
            supports_pose_cond=self.use_pose_cond,
            pose_cond_format=(
                "c2w(T,4,4) 只用平移列 [tx,ty](单位=窗口边长);内参不使用"
                if self.use_pose_cond else None),
            # 平移基线本身对帧数没有任何约束,但仍声明 4n+1:与 benchmark 数据的
            # 产物契约一致,避免它成为"唯一一个能跑非法 T 的骨干"从而掩盖
            # 上游产生非法 T 的 bug。改成 "any" 也是合法声明。
            frame_count_rule="4n+1",
            stochastic=False,   # 完全确定性 —— 多种子聚合在它上面恒 0 方差
            adaptation="none",
            supported_sizes=(),
            notes=("退化参照,非论文 baseline:不生成任何新内容,移出视野的区域"
                   "用边缘像素复制填充。"),
        )

    # ---------------------------------------------------------------- 身份
    def checkpoint_id(self) -> str:
        return (f"{self.name}:frac={self.shift_fraction:g},"
                f"pose={'on' if self.use_pose_cond else 'off'}")

    def manifest_fields(self) -> Dict[str, Any]:
        d = {"backbone": self.name}
        d.update(self.capabilities.as_manifest_fields())
        d["backbone_shift_fraction"] = self.shift_fraction
        return d

    # ------------------------------------------------------------ 内部工具
    def _parse_direction(self, prompt: str) -> Optional[str]:
        parser = self._direction_parser or _default_direction_parser
        return parser(prompt)

    @staticmethod
    def _resize_nearest(img: np.ndarray, h: int, w: int) -> np.ndarray:
        """最近邻缩放。刻意不用 cv2/PIL:本模块必须在"只有 numpy"的环境里可用,
        而对一个平移基线来说重采样质量不是被研究的变量(后续位移取整像素,
        不引入第二次重采样)。
        """
        if img.shape[0] == h and img.shape[1] == w:
            return img
        yi = (np.arange(h) * (img.shape[0] / h)).astype(np.int64).clip(0, img.shape[0] - 1)
        xi = (np.arange(w) * (img.shape[1] / w)).astype(np.int64).clip(0, img.shape[1] - 1)
        return img[yi][:, xi]

    @staticmethod
    def _shift_edge_clamp(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """把内容平移 (dx 向右为正, dy 向下为正),空出来的区域用边缘像素复制填。

        为什么不用 np.roll:roll 是环绕,接缝处会出现一条与主运动【方向相反】
        的强边缘,光流估计器会被它拉偏 —— 那是实现细节污染指标,不是基线特性。
        边缘复制的空白区几乎无梯度,Farnebäck 在那里给出的流幅度≈0,
        而 estimate_motion_direction 是按幅度加权的(metrics.py:330-333),
        所以空白区对方向读数几乎没有贡献。
        """
        h, w = img.shape[:2]
        if dx == 0 and dy == 0:
            return img.copy()
        # 用夹紧的坐标索引一次成型:src = dst - shift,越界处夹到边缘。
        ys = np.clip(np.arange(h) - dy, 0, h - 1)
        xs = np.clip(np.arange(w) - dx, 0, w - 1)
        return img[ys][:, xs]

    def _offsets_from_pose(self, pose_cond: PoseCond, h: int, w: int) -> np.ndarray:
        """c2w 平移列 → 每帧内容像素位移 (T,2) = (dx, dy)。

        camera_trajectory.pixel_to_world_translation 的定义:
            tx = (cx - cx0) / 窗口宽,  ty = (cy - cy0) / 窗口高
        即 tx 是"窗口向图像右(东)移动了几个窗口宽"。输出帧就是那个窗口
        重采样成 (h,w) 的结果,所以 1 个世界单位 = w(或 h)像素。
        内容与窗口反向,故取负号。
        """
        t = pose_cond.c2w[:, :3, 3]
        tx = t[:, 0] - t[0, 0]
        ty = t[:, 1] - t[0, 1]
        dx = -tx * float(w)
        dy = -ty * float(h)
        return np.stack([dx, dy], axis=-1)

    def _offsets_from_prompt(self, prompt: str, num_frames: int,
                             h: int, w: int) -> np.ndarray:
        word = self._parse_direction(prompt)
        if word is None:
            return np.zeros((num_frames, 2), dtype=np.float64)
        # 内容角 = 相机角 + 180°(滑窗约定);屏幕 y 轴向下,故 dy = -sin。
        theta = math.radians((_CAMERA_ANGLE_BY_WORD[word] + 180.0) % 360.0)
        step = self.shift_fraction * float(w)
        idx = np.arange(num_frames, dtype=np.float64)
        dx = idx * step * math.cos(theta)
        dy = idx * step * (-math.sin(theta))
        return np.stack([dx, dy], axis=-1)

    # ---------------------------------------------------------------- 生成
    def generate(
        self,
        *,
        first_frame: np.ndarray,
        prompt: str,
        num_frames: int,
        size: str,
        pose_cond: Optional[PoseCond] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        who = f"TrivialShiftBackbone({self.name})"
        first_frame = check_first_frame(first_frame, who=who)
        num_frames = check_frame_count(
            num_frames, self.capabilities.frame_count_rule, who=who)
        pose_cond = coerce_pose_cond(pose_cond)
        pose_mode = current_pose_mode()
        self.capabilities.assert_pose_mode_supported(pose_mode, who=who)
        assert_pose_cond_matches_mode(pose_cond, pose_mode, who=who)
        if pose_cond is not None and len(pose_cond) != num_frames:
            raise ValueError(
                f"{who}: pose_cond 有 {len(pose_cond)} 帧,请求 {num_frames} 帧 —— "
                f"必须相等(不做插值:插值会引入第二条与视频不同步的时间轴,"
                f"正是 build_dataset_from_benchmark.py:734-744 记录过的坑)。")
        del seed  # 确定性骨干;capabilities.stochastic=False 已声明这一点

        h, w = parse_size(size)
        base = self._resize_nearest(first_frame, h, w)
        if pose_cond is not None:
            offsets = self._offsets_from_pose(pose_cond, h, w)
        else:
            offsets = self._offsets_from_prompt(prompt, num_frames, h, w)

        frames = np.empty((num_frames, h, w, 3), dtype=np.uint8)
        for i in range(num_frames):
            dx = int(round(float(offsets[i, 0])))
            dy = int(round(float(offsets[i, 1])))
            frames[i] = self._shift_edge_clamp(base, dx, dy)
        return check_output_contract(frames, num_frames, size, who=who)


# =============================================================================
# 注册表 —— 让评估入口能用 `--backbone <name>` 选骨干
# =============================================================================

BACKBONE_BUILDERS: Dict[str, Callable[..., Any]] = {
    "wan": WanBackbone,
    "wan2.2-i2v-A14B": WanBackbone,
    "trivial-shift": TrivialShiftBackbone,
}


def register_backbone(name: str, builder: Callable[..., Any]) -> None:
    """第三方在自己的入口脚本里注册骨干,不必改本文件。

    刻意允许覆盖同名项(便于在测试里替身),但会打印一行 —— 静默覆盖会让
    "我明明选了 X 却跑了 Y" 变成无法追查的事故。
    """
    if name in BACKBONE_BUILDERS:
        print(f"[backbone] 注意:{name!r} 已存在,被新的构造器覆盖")
    BACKBONE_BUILDERS[name] = builder


def build_backbone(name: str, **kwargs: Any) -> Any:
    if name not in BACKBONE_BUILDERS:
        raise KeyError(
            f"未知骨干 {name!r};已注册: {sorted(BACKBONE_BUILDERS)}。"
            f"接入自己的模型见 docs/ADDING_A_BACKBONE.md")
    bb = BACKBONE_BUILDERS[name](**kwargs)
    validate_backbone(bb, who=f"build_backbone({name!r})")
    return bb


# =============================================================================
# 自检
# =============================================================================

def _self_test() -> int:
    ok, bad = [], []

    def check(label: str, cond: bool, detail: str = "") -> None:
        (ok if cond else bad).append(label)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}"
              + (f" — {detail}" if detail and not cond else ""))

    print("== backbone.py 自检 ==")
    check("parse_size", parse_size("480*832") == (480, 832))
    bb = TrivialShiftBackbone(direction_parser=lambda p: "E" if "east" in p else None)
    validate_backbone(bb)
    check("validate_backbone(TrivialShift)", True)
    os.environ["RSWM_POSE_MODE"] = "drop"
    img = (np.arange(480 * 832 * 3, dtype=np.int64) % 251).astype(np.uint8)
    img = img.reshape(480, 832, 3)
    v = bb.generate(first_frame=img, prompt="net displacement toward the east",
                    num_frames=9, size="480*832")
    check("shape/dtype", v.shape == (9, 480, 832, 3) and v.dtype == np.uint8, str(v.shape))
    check("值域", int(v.min()) >= 0 and int(v.max()) <= 255)
    check("首帧不动", bool((v[0] == img).all()))
    check("向东命令 ⟹ 内容左移", bool((v[1][:, :-33] == img[:, 33:]).all()))
    print(f"\n通过 {len(ok)},失败 {len(bad)}")
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RSWM-Bench 骨干协议")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(_self_test())
    print(__doc__ or "见文件头注释;--self-test 跑自检")
