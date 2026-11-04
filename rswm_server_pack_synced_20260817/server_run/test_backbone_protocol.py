#!/usr/bin/env python3
# =============================================================================
# test_backbone_protocol.py — 骨干可插拔协议(lora_remotesense/new/backbone.py)单测
#
# 为什么要有这份测试:backbone.py 存在的理由是"让第三方能把自己的模型插进来"。
# 协议一旦被无声改坏(比如某天有人给 pose_cond 去掉默认值、或把 size 按 宽*高
# 解释),症状不是报错而是"别人接进来跑出一堆看似合理的错数字"。所以协议的
# 每一条契约都必须有一条会红的断言。
#
# 这份测试**不需要 GPU、不需要 torch、不需要任何权重**:
#   * TrivialShiftBackbone 真跑,形状/dtype/值域/帧数/方向逐条验;
#   * WanBackbone 用 mock 验参数转发与 pose_cond=None 的处理 —— 它需要 56GB
#     基模权重,这台机器上没有,**端到端未验证**,这一点在文件末尾也再声明一次。
#
# 覆盖:
#   T1  协议结构校验 validate_backbone:缺属性 / 缺 pose_cond 默认值 / 签名不符
#   T2  能力声明:不支持位姿的骨干只允许 drop 臂,native/identity 硬拒
#   T3  TrivialShift 端到端(drop 臂):形状 (T,H,W,3) / uint8 / [0,255] / T 对齐
#   T4  TrivialShift 方向已知答案:四方位 prompt → 光流读数 = 相机角+180°
#   T5  TrivialShift 位姿驱动(native 臂):位移量 = -Δc2w × 帧尺寸,含尺度已知答案
#   T6  pose_mode 三态 ↔ pose_cond 的一一对应,四种矛盾组合全部硬失败
#   T7  帧数契约 4n+1 与返回值契约(帧数/尺寸/dtype/转置)
#   T8  WanBackbone 契约(mock):参数转发、pose_cond=None→action_path=None、
#       位姿落盘内容、临时目录清理、基模 vs LoRA 走不同加载函数、drop 臂早失败
#   T9  注册表 build_backbone / register_backbone
#
# 用法: python server_run/test_backbone_protocol.py
# =============================================================================
import contextlib
import io
import os
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEW = ROOT / "lora_remotesense" / "new"
sys.path.insert(0, str(NEW))

# 与仓库其它单测同款降级:CI 是纯 CPU、不装 torch 的(.github/workflows/ci.yml
# 第 19-23 行)。metrics.py 顶层 `import torch`,但 T4 只用到里面惰性 import cv2
# 的 estimate_motion_direction,空 ModuleType 足够。
for _m in ("torch", "h5py"):
    if _m not in sys.modules:
        try:
            __import__(_m)
        except Exception:
            sys.modules[_m] = types.ModuleType(_m)

import backbone as B  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not cond else ""))


def raises(fn, *, contains=None):
    """返回 (是否抛异常, 消息)。契约测试只认"抛了" + "消息里有可执行线索"。"""
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if contains and contains not in msg:
            return False, f"抛了但消息里没有 {contains!r}: {msg[:200]}"
        return True, msg
    return False, "没有抛异常"


@contextlib.contextmanager
def pose_mode(mode):
    old = os.environ.get("RSWM_POSE_MODE")
    os.environ["RSWM_POSE_MODE"] = mode
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RSWM_POSE_MODE", None)
        else:
            os.environ["RSWM_POSE_MODE"] = old


def make_texture(h=480, w=832, seed=0):
    """有结构的确定性纹理。8x 放大的粗噪声 —— 特征尺度足够大,Farnebäck
    (winsize=15)在降到 128x222 之后仍能跟住,不会退化成纯噪声匹配。"""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(h // 8 + 2, w // 8 + 2, 3)).astype(np.uint8)
    return np.repeat(np.repeat(coarse, 8, axis=0), 8, axis=1)[:h, :w].copy()


# 方向词解析器:显式注入,不走 backbone 里"懒加载 evaluate"的默认路径 ——
# evaluate.py 顶层 import torch/h5py,纯 CPU 的 CI 里 import 不进来。
# 这里用与 test_route_resolution.py 相同的源码切片手法,从 evaluate.py 里把
# **真身函数**取出来,而不是在测试里复制一份规则(复制就是 v2.x 那次
# "内联版只认 eastward" 事故的复现方式)。
def _real_direction_parser():
    src = (NEW / "evaluate.py").read_text()
    start = src.index("def parse_direction_word_from_prompt")
    end = src.index("_DIR_WORD_TO_CAMERA_ANGLE = {")
    ns = {"Optional": __import__("typing").Optional}
    exec(src[start:end], ns)  # noqa: S102
    return ns["parse_direction_word_from_prompt"]


DIRPARSE = _real_direction_parser()


def trivial(**kw):
    kw.setdefault("direction_parser", DIRPARSE)
    return B.TrivialShiftBackbone(**kw)


# =============================================================================
print("== T1 协议结构校验 validate_backbone ==")

good = trivial()
B.validate_backbone(good)
check("t1_good_passes", True)


class _NoDefault:
    name = "nodefault"
    capabilities = B.BackboneCapabilities(supports_pose_cond=False)

    def generate(self, *, first_frame, prompt, num_frames, size, pose_cond, seed):
        return None


ok, msg = raises(lambda: B.validate_backbone(_NoDefault()), contains="pose_cond")
check("t1_pose_cond_must_default_none", ok, msg)


class _Positional:
    name = "positional"
    capabilities = B.BackboneCapabilities(supports_pose_cond=False)

    def generate(self, first_frame, prompt):
        return None


ok, msg = raises(lambda: B.validate_backbone(_Positional()), contains="缺少关键字参数")
check("t1_missing_kwargs_rejected", ok, msg)

ok, msg = raises(lambda: B.validate_backbone(object()), contains="缺少")
check("t1_missing_attrs_rejected", ok, msg)


class _BadCaps:
    name = "badcaps"
    capabilities = {"supports_pose_cond": False}   # dict 冒充能力声明

    def generate(self, *, first_frame, prompt, num_frames, size,
                 pose_cond=None, seed=None):
        return None


ok, msg = raises(lambda: B.validate_backbone(_BadCaps()), contains="BackboneCapabilities")
check("t1_caps_must_be_dataclass", ok, msg)

# 声明支持位姿却不说清位姿格式 —— 复现无从谈起,构造期就该炸
ok, msg = raises(lambda: B.BackboneCapabilities(supports_pose_cond=True),
                 contains="pose_cond_format")
check("t1_caps_pose_format_required", ok, msg)
ok, msg = raises(
    lambda: B.BackboneCapabilities(supports_pose_cond=False, pose_cond_format="x"),
    contains="自相矛盾")
check("t1_caps_contradiction_rejected", ok, msg)


# =============================================================================
print("\n== T2 能力声明决定哪些臂适用 ==")

nopose = trivial(use_pose_cond=False)
check("t2_nopose_arms", nopose.capabilities.applicable_pose_modes == ("drop",),
      str(nopose.capabilities.applicable_pose_modes))
check("t2_pose_arms",
      trivial().capabilities.applicable_pose_modes == ("native", "identity", "drop"))

img = make_texture()
for pm in ("native", "identity"):
    with pose_mode(pm):
        ok, msg = raises(
            lambda: nopose.generate(first_frame=img, prompt="toward the east",
                                    num_frames=9, size="480*832",
                                    pose_cond=B.PoseCond(
                                        c2w=np.tile(np.eye(4), (9, 1, 1)))),
            contains="supports_pose_cond=False")
    check(f"t2_nopose_rejects_{pm}", ok, msg)

with pose_mode("drop"):
    v = nopose.generate(first_frame=img, prompt="toward the east",
                        num_frames=9, size="480*832")
check("t2_nopose_runs_drop", v.shape == (9, 480, 832, 3))


# =============================================================================
print("\n== T3 TrivialShift 端到端(drop 臂,无 GPU/无权重) ==")

bb = trivial()
with pose_mode("drop"):
    v = bb.generate(first_frame=img, prompt="net displacement toward the east",
                    num_frames=33, size="480*832", seed=7)
check("t3_shape", v.shape == (33, 480, 832, 3), str(v.shape))
check("t3_dtype", v.dtype == np.uint8, str(v.dtype))
check("t3_range", int(v.min()) >= 0 and int(v.max()) <= 255,
      f"[{v.min()},{v.max()}]")
check("t3_frame0_is_input", bool((v[0] == img).all()))
check("t3_frames_actually_move", float(np.abs(
    v[1].astype(np.float64) - v[0].astype(np.float64)).mean()) > 0.0)
# stochastic=False 的声明必须是真的:换 seed 输出必须逐位相同,否则
# aggregate_seeds 会把"其实是噪声"的差异当成种子方差报出来。
with pose_mode("drop"):
    v_other_seed = bb.generate(first_frame=img,
                               prompt="net displacement toward the east",
                               num_frames=33, size="480*832", seed=999)
check("t3_deterministic_matches_declaration",
      bb.capabilities.stochastic is False and bool((v == v_other_seed).all()))

# 非 canonical 尺寸也要严格按 size 返回(而不是按输入图的尺寸)
with pose_mode("drop"):
    v2 = bb.generate(first_frame=img, prompt="toward the north",
                     num_frames=9, size="832*480")
check("t3_size_is_H_times_W", v2.shape == (9, 832, 480, 3), str(v2.shape))

# 没有方向词、也没有位姿 = 没收到任何命令 → 静止视频(诚实,不是失败)
with pose_mode("drop"):
    v3 = bb.generate(first_frame=img, prompt="A satellite video of a rural area.",
                     num_frames=9, size="480*832")
check("t3_no_command_is_static", bool((v3 == v3[0]).all()))


# =============================================================================
print("\n== T4 方向已知答案:四方位 prompt → 光流读数 = 相机角+180° ==")

from metrics import compute_angular_error, estimate_motion_direction  # noqa: E402

# 滑窗约定:窗口向东 ⟹ 内容向西流,故合规目标角 = 相机角 + 180°
# (evaluate.parse_direction_angle_from_prompt 的定义)。
_PROMPTS = {
    "east": 180.0, "north": 270.0, "west": 0.0, "south": 90.0,
}
for word, target in _PROMPTS.items():
    with pose_mode("drop"):
        vv = bb.generate(first_frame=img,
                         prompt=f"A satellite video. Net displacement toward the {word}.",
                         num_frames=9, size="480*832")
    est = estimate_motion_direction(vv, flow_size=(128, 222), step=1)
    err = compute_angular_error(est, target) if est is not None else 999.0
    # 纯全局平移 + 目标本身就是基本方向 ⟹ 误差应≈0(与
    # validate_direction_metric.py 的 A2「人工平移校验预期 ≈0°」同一口径)。
    # 阈值放到 5° 只为吸收 INTER_AREA 降采样与整数取整,仍远小于随机期望 90°
    # 和 22.5° 量化地板 —— 方向若接反(180°)一定会红。
    check(f"t4_{word}", err < 5.0, f"est={est} target={target} err={err:.2f}")


# =============================================================================
print("\n== T5 位姿驱动:位移 = -Δc2w × 帧尺寸(尺度已知答案) ==")

# 造一条"窗口每帧向东移 9 像素"的 c2w(窗口原生分辨率 128x222,与
# canonical H5 的 pose_window_size 一致)。camera_trajectory.pixel_to_world_translation:
#   tx = (cx-cx0)/win_w  ⟹ 每帧 9/222 = 0.04054 个世界单位
# 输出帧就是那个窗口重采样成 832 宽的结果 ⟹ 内容每帧左移 0.04054*832 = 33.7 px。
# 这个 ~34 px/帧 与 metrics.py:272-274 记录的实测 GT 位移量级一致,说明
# 尺度换算没有接错(接错通常差一个 W/win_w = 3.75 倍)。
T = 9
win_w, win_h = 222, 128
c2w = np.tile(np.eye(4), (T, 1, 1)).astype(np.float64)
c2w[:, 0, 3] = np.arange(T) * (9.0 / win_w)      # 向东(图像右)
pc = B.PoseCond(c2w=c2w, intrinsics=np.tile([500., 500., 416., 240.], (T, 1)))
check("t5_posecond_implies_native", pc.implied_pose_mode == "native")
check("t5_identity_detected",
      B.PoseCond(c2w=np.tile(np.eye(4), (T, 1, 1))).implied_pose_mode == "identity")

with pose_mode("native"):
    vp = bb.generate(first_frame=img, prompt="toward the east",
                     num_frames=T, size="480*832", pose_cond=pc)
check("t5_shape", vp.shape == (T, 480, 832, 3), str(vp.shape))
# 第 1 帧应当整体左移 34 px(round(33.73))
shift_px = int(round(9.0 / win_w * 832))
check("t5_scale_is_34px", shift_px == 34, f"算出 {shift_px}")
check("t5_frame1_shifted_left_by_34",
      bool((vp[1][:, :-shift_px] == img[:, shift_px:]).all()))
est = estimate_motion_direction(vp, flow_size=(128, 222), step=1)
check("t5_pose_direction_is_west_content",
      est is not None and compute_angular_error(est, 180.0) < 5.0,
      f"est={est}")

# 位姿帧数与 num_frames 不等 → 硬失败(不插值:插值会引入与视频不同步的第二条时间轴)
with pose_mode("native"):
    ok, msg = raises(lambda: bb.generate(first_frame=img, prompt="east",
                                         num_frames=17, size="480*832",
                                         pose_cond=pc), contains="必须相等")
check("t5_len_mismatch_rejected", ok, msg)


# =============================================================================
print("\n== T6 pose_mode 三态 ↔ pose_cond 的一一对应 ==")

ident = B.PoseCond(c2w=np.tile(np.eye(4), (T, 1, 1)))

with pose_mode("drop"):
    ok, msg = raises(lambda: bb.generate(first_frame=img, prompt="east",
                                         num_frames=T, size="480*832",
                                         pose_cond=pc),
                     contains="位姿通路整条不接")
check("t6_drop_plus_posecond_rejected", ok, msg)

for pm in ("native", "identity"):
    with pose_mode(pm):
        ok, msg = raises(lambda: bb.generate(first_frame=img, prompt="east",
                                             num_frames=T, size="480*832"),
                         contains="pose_cond=None")
    check(f"t6_{pm}_without_posecond_rejected", ok, msg)

with pose_mode("native"):
    ok, msg = raises(lambda: bb.generate(first_frame=img, prompt="east",
                                         num_frames=T, size="480*832",
                                         pose_cond=ident),
                     contains="几何通道实际全零")
check("t6_native_with_identity_poses_rejected", ok, msg)

with pose_mode("identity"):
    ok, msg = raises(lambda: bb.generate(first_frame=img, prompt="east",
                                         num_frames=T, size="480*832",
                                         pose_cond=pc),
                     contains="静止相机诊断臂")
check("t6_identity_with_native_poses_rejected", ok, msg)

with pose_mode("identity"):
    vi = bb.generate(first_frame=img, prompt="toward the east",
                     num_frames=T, size="480*832", pose_cond=ident)
# identity 臂:位姿全是单位阵 ⟹ 平移量恒 0 ⟹ 静止视频。它**不是** text-only:
# 真 text-only(drop)会按 prompt 的方向词动起来,两者输出可区分。
check("t6_identity_is_static", bool((vi == vi[0]).all()))
with pose_mode("drop"):
    vd = bb.generate(first_frame=img, prompt="toward the east",
                     num_frames=T, size="480*832")
check("t6_identity_differs_from_drop", not bool((vi == vd).all()))

with pose_mode("bogus"):
    ok, msg = raises(lambda: bb.generate(first_frame=img, prompt="e", num_frames=T,
                                         size="480*832"), contains="非法")
check("t6_bogus_mode_rejected", ok, msg)


# =============================================================================
print("\n== T7 帧数契约与返回值契约 ==")

with pose_mode("drop"):
    ok, msg = raises(lambda: bb.generate(first_frame=img, prompt="east",
                                         num_frames=32, size="480*832"),
                     contains="T≡1 (mod 4)")
check("t7_frame_count_4n1", ok, msg)
for n in (9, 17, 25, 33):
    check(f"t7_legal_T_{n}", B.check_frame_count(n) == n)

ok, msg = raises(lambda: B.check_output_contract(
    np.zeros((9, 832, 480, 3), np.uint8), 9, "480*832"), contains="'高*宽'")
check("t7_transposed_output_rejected", ok, msg)
ok, msg = raises(lambda: B.check_output_contract(
    np.zeros((8, 480, 832, 3), np.uint8), 9, "480*832"), contains="返回 8 帧")
check("t7_wrong_T_rejected", ok, msg)
ok, msg = raises(lambda: B.check_output_contract(
    np.zeros((9, 480, 832, 3), np.float32), 9, "480*832"), contains="uint8")
check("t7_float_output_rejected", ok, msg)
ok, msg = raises(lambda: bb.generate(first_frame=img.astype(np.float32),
                                     prompt="e", num_frames=9, size="480*832"),
                 contains="uint8")
check("t7_float_input_rejected", ok, msg)
check("t7_parse_size", B.parse_size("480*832") == (480, 832))
ok, msg = raises(lambda: B.parse_size("480x832"), contains="格式非法")
check("t7_bad_size_rejected", ok, msg)


# =============================================================================
print("\n== T8 WanBackbone 契约(mock;真权重未验,见文末声明) ==")


class _FakeTensor:
    """够用地模仿 evaluate.py:451-452 那条转换链:
    clamp(-1,1) → permute(1,2,3,0) → (x+1)*127.5 → byte() → cpu() → numpy()"""

    def __init__(self, a):
        self.a = np.asarray(a)

    def clamp(self, lo, hi):
        return _FakeTensor(np.clip(self.a, lo, hi))

    def permute(self, *ax):
        return _FakeTensor(np.transpose(self.a, ax))

    def __add__(self, o):
        return _FakeTensor(self.a + o)

    def __mul__(self, o):
        return _FakeTensor(self.a * o)

    __rmul__ = __mul__

    def byte(self):
        return _FakeTensor(self.a.astype(np.uint8))

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _WanMocks:
    """把 backbone.WanBackbone 延迟 import 的四个模块整体替身掉。

    为什么用 sys.modules 注入而不是 monkeypatch 属性:WanBackbone 是在函数体
    里 `from inference import generate_video` 的(文件头 F 段:顶层不能 import
    torch)。只有替换模块本身才能拦到它。
    """

    def __init__(self, *, video=None):
        self.calls = {"generate_video": [], "load_lora_model": 0, "load_base_model": 0}
        self.video = video
        self._saved = {}

    def __enter__(self):
        mocks = self

        def _generate_video(**kw):
            # 位姿目录必须在【调用期间】存在且内容正确 —— 调用结束后被删,
            # 所以只能在这里读。
            snap = dict(kw)
            ap = kw.get("action_path")
            if ap is not None:
                snap["_poses"] = np.load(os.path.join(ap, "poses.npy"))
                snap["_intr"] = np.load(os.path.join(ap, "intrinsics.npy"))
                snap["_dir_existed"] = True
            mocks.calls["generate_video"].append(snap)
            f = kw["frame_num"]
            h, w = B.parse_size(kw["size"])
            vid = mocks.video
            if vid is None:
                vid = np.zeros((3, f, h, w), dtype=np.float32)
            return _FakeTensor(vid)

        inference = types.ModuleType("inference")
        inference.generate_video = _generate_video

        def _load_lora_model(**kw):
            mocks.calls["load_lora_model"] += 1
            mocks.calls["lora_kw"] = kw
            return ("T5", "VAE", "LOW", "HIGH")

        inference.load_lora_model = _load_lora_model

        eb = types.ModuleType("evaluate_baseline")

        def _load_base_model(**kw):
            mocks.calls["load_base_model"] += 1
            mocks.calls["base_kw"] = kw
            return ("T5", "VAE", "LOW", "HIGH")

        eb.load_base_model = _load_base_model

        wan = types.ModuleType("wan")
        wan_configs = types.ModuleType("wan.configs")
        cfg = types.SimpleNamespace(sample_fps=16, param_dtype="bf16")
        wan_configs.WAN_CONFIGS = {"i2v-A14B": cfg}
        wan.configs = wan_configs

        torch_mod = types.ModuleType("torch")
        torch_mod.device = lambda s: ("device", s)

        pil = types.ModuleType("PIL")
        pil_image = types.ModuleType("PIL.Image")
        pil_image.fromarray = lambda a: ("PIL", np.asarray(a).shape)
        pil.Image = pil_image

        for k, v in (("inference", inference), ("evaluate_baseline", eb),
                     ("wan", wan), ("wan.configs", wan_configs),
                     ("torch", torch_mod), ("PIL", pil), ("PIL.Image", pil_image)):
            self._saved[k] = sys.modules.get(k, None)
            sys.modules[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        return False


# --- 8a. drop 臂:pose_cond=None 必须原样转成 action_path=None
with _WanMocks() as mk, pose_mode("drop"):
    wb = B.WanBackbone("/nonexistent/ckpt", lora_path="/nonexistent/lora")
    B.validate_backbone(wb)
    out = wb.generate(first_frame=img, prompt="toward the east",
                      num_frames=9, size="480*832", seed=1234)
    call = mk.calls["generate_video"][-1]
check("t8a_output_contract", out.shape == (9, 480, 832, 3) and out.dtype == np.uint8,
      str(out.shape))
check("t8a_action_path_none", call["action_path"] is None, repr(call["action_path"]))
check("t8a_prompt_forwarded", call["prompt"] == "toward the east")
check("t8a_frame_num_forwarded", call["frame_num"] == 9, repr(call["frame_num"]))
check("t8a_size_forwarded", call["size"] == "480*832")
check("t8a_seed_forwarded", call["seed"] == 1234, repr(call["seed"]))
# 默认超参必须与 evaluate.py:432-448 现在实际吃到的 inference 默认值逐个相同,
# 否则"换成 WanBackbone"就悄悄改了采样配置。
check("t8a_sampling_steps_default", call["sampling_steps"] == 70, repr(call["sampling_steps"]))
check("t8a_shift_default", call["shift"] == 10.0, repr(call["shift"]))
check("t8a_guide_scale_default", call["guide_scale"] == (5.0, 5.0), repr(call["guide_scale"]))
check("t8a_t5_cpu_default", call["t5_cpu"] is True)
check("t8a_framewise_default", call["framewise"] is True)
check("t8a_inactive_cpu_default", call["inactive_model_cpu"] is True)
check("t8a_first_frame_as_pil", call["image"] == ("PIL", (480, 832, 3)), repr(call["image"]))
check("t8a_lora_loader_used",
      mk.calls["load_lora_model"] == 1 and mk.calls["load_base_model"] == 0)

# --- 8b. native 臂:位姿写进临时目录,内容 == 输入,调用后目录被清理
with _WanMocks() as mk, pose_mode("native"):
    wb = B.WanBackbone("/nonexistent/ckpt", lora_path="/nonexistent/lora")
    wb.generate(first_frame=img, prompt="east", num_frames=T,
                size="480*832", pose_cond=pc, seed=42)
    call = mk.calls["generate_video"][-1]
check("t8b_action_path_dir", isinstance(call["action_path"], str))
check("t8b_poses_roundtrip", np.allclose(call["_poses"], pc.c2w.astype(np.float32)))
check("t8b_intrinsics_roundtrip",
      np.allclose(call["_intr"], pc.intrinsics.astype(np.float32)))
check("t8b_tempdir_cleaned", not os.path.exists(call["action_path"]),
      call["action_path"])

# --- 8c. 基模(lora_path=None)必须走 load_base_model,而不是 load_lora_model
#     (后者在 LoRA 目录缺失时会硬失败 —— v2.7.1 F2 刻意的"拒绝静默退化为基模型")
with _WanMocks() as mk, pose_mode("drop"):
    wb = B.WanBackbone("/nonexistent/ckpt", lora_path=None)
    wb.generate(first_frame=img, prompt="east", num_frames=9, size="480*832", seed=0)
check("t8c_base_loader_used",
      mk.calls["load_base_model"] == 1 and mk.calls["load_lora_model"] == 0)
check("t8c_adaptation_none", wb.capabilities.adaptation == "none")
check("t8c_checkpoint_id_carries_backbone",
      wb.checkpoint_id() == "wan2.2-i2v-A14B+none:base", wb.checkpoint_id())

# --- 8d. 矛盾组合必须在【加载模型之前】就失败(省掉十几分钟的权重加载)
with _WanMocks() as mk, pose_mode("drop"):
    wb = B.WanBackbone("/nonexistent/ckpt", lora_path="/nonexistent/lora")
    ok, msg = raises(lambda: wb.generate(first_frame=img, prompt="east",
                                         num_frames=T, size="480*832",
                                         pose_cond=pc),
                     contains="位姿通路整条不接")
    check("t8d_drop_plus_pose_rejected", ok, msg)
    check("t8d_failed_before_loading", mk.calls["load_lora_model"] == 0
          and mk.calls["load_base_model"] == 0)
    ok, msg = raises(lambda: wb.generate(first_frame=img, prompt="east",
                                         num_frames=32, size="480*832"),
                     contains="T≡1 (mod 4)")
    check("t8d_illegal_T_rejected_before_loading", ok, msg)

# --- 8e. 位姿帧数 ≠ num_frames 必须硬拒(否则 inference.py:188 的隐式钳制会
#     让 drop 臂与 native 臂的 T 不同 —— v2.8.2 N6 记录的静默不可比故障)
with _WanMocks() as mk, pose_mode("native"):
    wb = B.WanBackbone("/nonexistent/ckpt", lora_path="/nonexistent/lora")
    ok, msg = raises(lambda: wb.generate(first_frame=img, prompt="east",
                                         num_frames=17, size="480*832",
                                         pose_cond=pc), contains="N6")
check("t8e_pose_len_mismatch_rejected", ok, msg)

# --- 8f. 返回值不合契约(帧数被上游钳过)时必须炸,而不是往下游送
with _WanMocks(video=np.zeros((3, 5, 480, 832), np.float32)) as mk, pose_mode("drop"):
    wb = B.WanBackbone("/nonexistent/ckpt", lora_path="/nonexistent/lora")
    ok, msg = raises(lambda: wb.generate(first_frame=img, prompt="east",
                                         num_frames=9, size="480*832"),
                     contains="返回 5 帧")
check("t8f_short_output_rejected", ok, msg)

# --- 8g. [-1,1] float → uint8 的换算与 evaluate.py:451-452 逐值一致
vals = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
tt = _FakeTensor(np.tile(vals.reshape(1, 3, 1, 1), (3, 1, 2, 2)))
conv = B.wan_tensor_to_uint8(tt)
check("t8g_value_mapping", conv.shape == (3, 2, 2, 3)
      and conv[0, 0, 0, 0] == 0 and conv[1, 0, 0, 0] == 127 and conv[2, 0, 0, 0] == 255,
      str(np.unique(conv)))

# --- 8h. 能力声明:Wan 支持位姿,适配方法记 LoRA,manifest 字段可落盘
with _WanMocks():
    wb = B.WanBackbone("/ckpt", lora_path="/lora/r128")
mf = wb.manifest_fields()
check("t8h_caps_supports_pose", wb.capabilities.supports_pose_cond is True)
check("t8h_caps_adaptation_lora", wb.capabilities.adaptation == "lora")
check("t8h_manifest_has_backbone", mf.get("backbone") == "wan2.2-i2v-A14B")
check("t8h_manifest_json_safe",
      __import__("json").loads(__import__("json").dumps(mf)) == mf)
check("t8h_checkpoint_id_discriminates",
      wb.checkpoint_id() != trivial().checkpoint_id())


# =============================================================================
print("\n== T9 注册表 ==")

check("t9_build_trivial",
      B.build_backbone("trivial-shift", direction_parser=DIRPARSE).name == "trivial-shift")
ok, msg = raises(lambda: B.build_backbone("no-such-model"),
                 contains="ADDING_A_BACKBONE.md")
check("t9_unknown_points_to_doc", ok, msg)


class _Custom:
    name = "custom"
    capabilities = B.BackboneCapabilities(supports_pose_cond=False)

    def generate(self, *, first_frame, prompt, num_frames, size,
                 pose_cond=None, seed=None):
        h, w = B.parse_size(size)
        return np.zeros((num_frames, h, w, 3), np.uint8)


buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    B.register_backbone("custom", _Custom)
check("t9_register_and_build", B.build_backbone("custom").name == "custom")
with contextlib.redirect_stdout(buf):
    B.register_backbone("custom", _Custom)
check("t9_overwrite_is_loud", "被新的构造器覆盖" in buf.getvalue())


# =============================================================================
print(f"\n{'=' * 70}")
print(f"通过 {len(PASS)},失败 {len(FAIL)}")
if FAIL:
    print("失败项: " + ", ".join(FAIL))
print("说明:WanBackbone 的真权重端到端推理【未在本机验证】(需 56GB 基模 + GPU),")
print("     本文件只用 mock 固定它的参数转发与三态契约。真跑请见 REPRODUCIBILITY.md。")
sys.exit(1 if FAIL else 0)
