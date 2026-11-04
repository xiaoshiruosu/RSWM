#!/usr/bin/env python3
# =============================================================================
# test_metrics_known_answer.py —— 指标实现的**已知答案**单测
#
# 为什么要单独有这一份
# --------------------
# 仓库里已有的几百条断言,相当一部分是**源码字符串匹配**(检查某段注释/某个
# 常量还在)。那种测试能防"改回去",但**防不住算错**:一个把 y 轴符号搞反的
# 光流方向估计器、一个把角度环绕写成 abs(a-b) 的角误差、一个把 content 角当
# camera 角用的目标定义,源码里那些字符串一个都不会变,而论文里的每一个角度
# 数字都会错 180° 或 358°。
#
# 本文件全部是**真跑**:构造一个数学上答案唯一的输入,调真实实现,核对数值。
# 覆盖(每组末尾都有一次**故障注入**,证明这组断言真的有牙):
#   A  角度环绕      compute_angular_error(359,1)=2 而不是 358
#   B  方向估计      已知平移方向的合成视频 → estimate_motion_direction 已知角
#   C  恒等/已知差    SSIM=1、PSNR=inf、Δ=1 → PSNR=48.1308dB、LPIPS≈0
#   D  净方向轨迹    已知净位移路径 → net_direction_from_path 误差≈0 + 门控语义
#   D3 静止视频      **已知缺陷钉**:静止视频拿到的是噪声角,不是 None
#   E  180° 约定     camera 角 vs content 角的方向性(错一次就整块翻 180°)
#   F  量化地板      四方向命令的 22.5° 平均地板 / 45° 最坏地板(论文引用的数字)
#   G  批量对齐      videos 与 target_angles 必须逐一对齐,长度不等要炸
#
#   python3 server_run/test_metrics_known_answer.py
#   退出码 0=全绿,4=有 FAIL
# =============================================================================
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lora_remotesense" / "new"))
sys.path.insert(0, str(ROOT))

import metrics                                    # noqa: E402
import trajectory_metrics as tm                   # noqa: E402
from evaluate import (direction_target_from_waypoints,      # noqa: E402
                      direction_word_from_waypoints)
from diagnostic_baselines import (_fabricate_demo_scene,    # noqa: E402
                                  deterministic_translation)

PASS, FAIL, SKIP = [], [], []

# 合成平移视频的方向容差。实测(本文件的 B 组、整数像素位移、纹理充分)最差
# 0.5° 以内;取 3° 留出 OpenCV 版本间的余量,但仍远小于 22.5° 量化地板 ——
# 也就是说这个容差足够小,不会把"符号写反 / 少了 y 翻转"这类真错误放过去。
ANG_TOL = 3.0


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not cond else ""))


def note(name, detail=""):
    """既非 PASS 也非 FAIL 的观察值(依赖缺失等),显式列出,绝不静默。"""
    SKIP.append(name)
    print(f"  [SKIP] {name}" + (f" — {detail}" if detail else ""))


def run_cases(cases):
    for n, ok, d in cases:
        check(n, ok, d)


def expect_mutation_caught(label, cases_fn, broken_fn, min_caught):
    """故障注入:把被测函数换成一个**故意写错**的版本,断言原来的用例确实变红。

    这是这份文件唯一的元测试 —— 一组永远绿的断言和没有断言是一回事。
    """
    caught = [n for n, ok, _ in cases_fn(broken_fn) if not ok]
    check(f"[注入] {label}", len(caught) >= min_caught,
          f"注入错误后只有 {len(caught)}/{min_caught} 条用例变红:{caught} "
          f"—— 说明这组断言抓不到该类错误")
    if len(caught) >= min_caught:
        print(f"         注入后变红 {len(caught)} 条(如 {caught[:3]}),断言有效")


# =============================================================================
# 合成素材(整数像素位移 ⇒ 真值角度是**精确**的,不含插值/取整误差)
# =============================================================================

_H, _W, _PAD = 120, 208, 160
_SH, _SW = _H + 2 * _PAD, _W + 2 * _PAD


def _canvas(seed=0):
    """高频纹理 + 噪声:保证 Farneback 每个像素都有可跟踪结构。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:_SH, 0:_SW].astype(np.float32)
    b = 128 + 60 * np.sin(xx / 9.0) * np.cos(yy / 11.0) + 40 * np.sin((xx + yy) / 5.0)
    b = np.stack([b, np.roll(b, 7, axis=1), np.roll(b, 13, axis=0)], -1)
    return np.clip(b + rng.normal(0, 10, b.shape), 0, 255).astype(np.uint8)


_CANVAS = _canvas(0)


def translated_video(dx: int, dy: int, T: int = 17) -> np.ndarray:
    """内容每帧平移 (dx 东正, dy 南正) **整数**像素的视频。

    实现方式是在大画布上反向移动取景框:第 t 帧取 (PAD - dy*t, PAD - dx*t)
    处的 _H×_W 窗口 —— 于是画面内容相对上一帧就是 +(dx, dy)。真值角度
    = degrees(atan2(-dy, dx)) % 360,与 metrics.py:283 同一约定。
    """
    fr = []
    for t in range(T):
        top, left = _PAD - dy * t, _PAD - dx * t
        if not (0 <= top and top + _H <= _SH and 0 <= left and left + _W <= _SW):
            raise ValueError(f"位移 ({dx},{dy})×{T} 帧超出画布,请加大 _PAD")
        fr.append(_CANVAS[top:top + _H, left:left + _W])
    return np.stack(fr)


# 8 个整数位移 → 真值角精确可算。含两个非 45°倍数的角,专门压环绕与象限:
#   (4,1)  → atan2(-1,4)  = -14.04° → 345.96°(必须落在 [0,360) 而不是负数)
#   (4,-1) → atan2(1,4)   =  14.04°
_SHIFTS = [(4, 0), (0, -4), (-4, 0), (0, 4), (4, -4), (-3, 3), (4, 1), (4, -1)]


def _true_angle(dx, dy):
    return math.degrees(math.atan2(-dy, dx)) % 360.0


# =============================================================================
# A. 角度环绕
# =============================================================================

def _wrap_cases(fn):
    """compute_angular_error 的已知答案。fn(a,b) -> [0,180]。"""
    ex = [
        ("wrap_359_vs_1", 359.0, 1.0, 2.0),
        ("wrap_1_vs_359", 1.0, 359.0, 2.0),
        ("wrap_350_vs_10", 350.0, 10.0, 20.0),
        ("wrap_10_vs_350", 10.0, 350.0, 20.0),
        ("same_angle", 137.0, 137.0, 0.0),
        ("opposite_is_180", 0.0, 180.0, 180.0),
        ("just_past_180_folds_back", 0.0, 181.0, 179.0),
        ("quarter_turns", 90.0, 270.0, 180.0),
        ("multiples_of_360_ignored", 365.0, 5.0, 0.0),
        ("negative_input_normalized", -10.0, 350.0, 0.0),
    ]
    out = []
    for name, a, b, want in ex:
        try:
            got = float(fn(a, b))
            ok = abs(got - want) < 1e-9
        except Exception as e:                       # 注入版可能直接抛
            got, ok = repr(e), False
        out.append((name, ok, f"{fn.__name__ if hasattr(fn,'__name__') else fn}"
                              f"({a},{b}) = {got},期望 {want}"))
    return out


def group_A():
    print("== A. 角度环绕(compute_angular_error)==")
    run_cases(_wrap_cases(metrics.compute_angular_error))
    # 注入:最经典的写法错误 —— 忘了 mod 360 与"大于 180 折回"。
    def broken_abs(a, b):
        return abs(float(a) - float(b))
    expect_mutation_caught("环绕算成 abs(a-b)", _wrap_cases, broken_abs, 5)
    # 注入:只做了 mod 360、忘了折回 [0,180]。359 vs 1 会给 358。
    def broken_no_fold(a, b):
        return abs(float(a) - float(b)) % 360.0
    expect_mutation_caught("有 mod 360 但没折回 [0,180]", _wrap_cases,
                           broken_no_fold, 2)


# =============================================================================
# B. 光流方向估计
# =============================================================================

def _direction_cases(estimator):
    """estimator(frames) -> 角度(度)。对每个已知位移核对读数。"""
    out = []
    for dx, dy in _SHIFTS:
        want = _true_angle(dx, dy)
        est = estimator(translated_video(dx, dy))
        if est is None:
            out.append((f"dir_{dx}_{dy}", False, "估计器返回 None"))
            continue
        err = metrics.compute_angular_error(est, want)
        out.append((f"dir_{dx}_{dy}", err < ANG_TOL,
                    f"真值 {want:.2f}°,读数 {est:.2f}°,误差 {err:.2f}° "
                    f"(容差 {ANG_TOL}°)"))
    return out


def group_B():
    print("\n== B. 已知平移方向 → estimate_motion_direction ==")
    # 生产口径:窗口分辨率 + step=1(evaluate.compute_metrics 的 _ANGKW)。
    # 这里帧本身就是窗口尺寸,flow_size 传同值 ⇒ 不缩放,角度不被各向异性扭曲。
    def real(frames):
        return metrics.estimate_motion_direction(frames, flow_size=(_H, _W), step=1)
    run_cases(_direction_cases(real))
    # 注入 1:漏掉图像 y 轴向下的翻转(atan2(dy,dx) 而不是 atan2(-dy,dx))。
    # 这是最容易犯、也最难从源码看出来的错:东/西两个方向**照样对**,
    # 只有南北与斜向会翻 —— 所以只有覆盖了南北与斜向的用例集才抓得到。
    def flipped_y(frames):
        e = real(frames)
        return None if e is None else (-e) % 360.0
    expect_mutation_caught("漏掉 y 轴翻转(atan2 符号)", _direction_cases,
                           flipped_y, 4)
    # 注入 2:把 content 角当 camera 角报(整块翻 180°)。
    def flipped_180(frames):
        e = real(frames)
        return None if e is None else (e + 180.0) % 360.0
    expect_mutation_caught("content/camera 角搞反(整块 180°)",
                           _direction_cases, flipped_180, len(_SHIFTS))


# =============================================================================
# C. 恒等输入与已知像素差
# =============================================================================

def _psnr_cases(fn):
    """fn(img1,img2)->dB。Δ 恒定 ⇒ MSE=Δ²,PSNR=10·log10(255²/Δ²) 精确可算。"""
    base = np.full((32, 48, 3), 100, np.uint8)
    out = []
    for d in (1, 2, 5):
        want = 10.0 * math.log10(255.0 ** 2 / float(d) ** 2)
        got = float(fn(base, base + np.uint8(d)))
        out.append((f"psnr_delta{d}", abs(got - want) < 1e-6,
                    f"读数 {got:.6f} dB,期望 {want:.6f} dB"))
    out.append(("psnr_identical_is_inf", math.isinf(float(fn(base, base.copy()))),
                f"{fn(base, base.copy())}"))
    return out


def group_C():
    print("\n== C. 恒等 / 已知像素差(SSIM·PSNR·LPIPS)==")
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    vid = np.stack([np.roll(img, k, axis=1) for k in range(9)])

    check("ssim_identical_is_1",
          abs(metrics.compute_ssim(img, img.copy()) - 1.0) < 1e-9,
          f"{metrics.compute_ssim(img, img.copy())}")
    check("video_ssim_identical_is_1",
          abs(metrics.compute_video_ssim(vid, vid.copy()) - 1.0) < 1e-9)
    check("ssim_differs_is_below_1",
          metrics.compute_ssim(img, np.roll(img, 3, axis=1)) < 0.99)
    check("psnr_identical_is_inf", math.isinf(metrics.compute_psnr(img, img.copy())))
    check("video_psnr_identical_is_inf",
          math.isinf(metrics.compute_video_psnr(vid, vid.copy())))
    run_cases(_psnr_cases(metrics.compute_psnr))
    # 注入:把 255² 写成 255(峰值信号漏了平方)—— 结果整体差 24.07 dB,
    # 恰好是那种"看起来还是个合理的 dB 数"的错误,只有已知答案能抓。
    def broken_psnr(a, b):
        a = a.astype(np.float64); b = b.astype(np.float64)
        mse = float(np.mean((a - b) ** 2))
        return float("inf") if mse == 0 else 10 * math.log10(255.0 / mse)
    expect_mutation_caught("PSNR 峰值信号漏平方(255 而非 255²)",
                           _psnr_cases, broken_psnr, 3)

    # LPIPS:权重需要下载,离线机上装不了就显式 SKIP,绝不当成通过。
    try:
        v = metrics.compute_video_lpips(vid, vid.copy(), device="cpu")
        check("lpips_identical_is_zero", abs(float(v)) < 1e-4, f"{v}")
    except Exception as e:
        note("lpips_identical_is_zero", f"LPIPS 不可用({type(e).__name__});"
                                        f"有 lpips 包与权重的机器上会真跑")


# =============================================================================
# D. 净位移轨迹 → net_direction_from_path
# =============================================================================

def _straight_path(angle_deg, T=17, step=3.0):
    """朝 content 角 angle_deg 的直线路径,列序 = (x 东正, y 南正)。"""
    r = math.radians(angle_deg)
    t = np.arange(T, dtype=np.float64) * step
    return np.stack([t * math.cos(r), -t * math.sin(r)], axis=-1)


def _netdir_cases(path_adapter):
    """path_adapter(path)->path:注入点在"传给指标的路径"上(列序/符号)。"""
    out = []
    for ang in (0.0, 45.0, 90.0, 180.0, 271.0, 345.0):
        r = tm.net_direction_from_path(path_adapter(_straight_path(ang)),
                                       gt_content_net_angle=ang,
                                       gt_net_ratio=1.0)
        e = r["error_deg"]
        out.append((f"net_dir_{ang:g}", e is not None and e < 1e-6,
                    f"error_deg={e},gated={r['gated']}"))
    return out


def group_D():
    print("\n== D. 已知净位移轨迹 → net_direction_from_path ==")
    run_cases(_netdir_cases(lambda p: p))

    # 目标角错 90° → 误差必须是 90°(不是 0,也不是 270)
    r = tm.net_direction_from_path(_straight_path(30.0),
                                   gt_content_net_angle=120.0, gt_net_ratio=1.0)
    check("net_dir_wrong_target_is_90",
          r["error_deg"] is not None and abs(r["error_deg"] - 90.0) < 1e-6,
          f"{r['error_deg']}")
    # 门控:GT **命令轨迹**近闭合 ⇒ 净方向这个问题无意义 ⇒ None + gated
    r = tm.net_direction_from_path(_straight_path(0.0),
                                   gt_content_net_angle=0.0, gt_net_ratio=0.05)
    check("net_dir_gated_by_gt_ratio",
          r["error_deg"] is None and r["gated"] is True
          and r["recovered_net_degenerate"] is False, f"{r}")
    # 退化:GT 有净方向但恢复路径原地不动 ⇒ None + degenerate(**不是** gated),
    # 两者语义不同,混起来会把"没测出来"读成"这题没意义"。
    r = tm.net_direction_from_path(np.zeros((17, 2)),
                                   gt_content_net_angle=0.0, gt_net_ratio=1.0)
    check("net_dir_degenerate_not_gated",
          r["error_deg"] is None and r["gated"] is False
          and r["recovered_net_degenerate"] is True, f"{r}")

    # polyline_net_ratio 的三个精确值
    check("ratio_straight_is_1",
          abs(tm.polyline_net_ratio([[0, 0], [1, 0], [2, 0], [3, 0]]) - 1.0) < 1e-12)
    check("ratio_closed_is_0",
          abs(tm.polyline_net_ratio([[0, 0], [1, 0], [0.5, 1], [0, 0]])) < 1e-12)
    check("ratio_L_equal_legs_is_1_over_sqrt2",
          abs(tm.polyline_net_ratio([[0, 0], [1, 0], [1, 1]])
              - math.sqrt(2) / 2.0) < 1e-12)

    # 注入:把路径的两列对调(即误把 (cy,cx)/(y,x) 当成 (x,y) 传进来)。
    # 这是跨模块传路径时最常见的错,且**不会**改变路径长度或形状类指标,
    # 只有方向类指标看得见 —— 所以必须由这一组来抓。
    expect_mutation_caught("路径列序 (x,y) 被写成 (y,x)",
                           _netdir_cases, lambda p: p[:, ::-1], 4)


def group_D3():
    print("\n== D3. 静止视频的方向读数(已知缺陷钉,不是契约)==")
    # 契约成立的那一半:完全无纹理 ⇒ 光流恒 0 ⇒ 幅度门生效 ⇒ None。
    flat = np.full((17, _H, _W, 3), 100, np.uint8)
    check("uniform_static_returns_None",
          metrics.estimate_motion_direction(flat, flow_size=(_H, _W), step=1) is None)
    # 契约**不**成立的那一半:有纹理的静止视频(逐字节相同的 T 帧)也会拿到
    # 一个角度。根因:Farneback 在相同两帧上仍返回 ~1e-3 量级的数值残差,而
    # estimate_motion_direction 的可靠性门是 `weight_total < 1.0` —— 一个**绝对**
    # 阈值,与分辨率和帧对数无关地累加,120×208×16 对早就 >> 1,门永远不响。
    # 后果:persistence 类静止基线会拿到一个**随机方向**而不是 None,
    # 谁去比它的角误差谁就被噪声骗(diagnostic_baselines 的 docstring 写的是
    # "unreliable/None",实际拿到的是 unreliable 但**不是** None)。
    ests = []
    for s in (1, 2, 3, 4, 5):
        tex = np.random.default_rng(s).integers(0, 256, (_H, _W, 3), dtype=np.uint8)
        ests.append(metrics.estimate_motion_direction(
            np.repeat(tex[None], 17, 0), flow_size=(_H, _W), step=1))
    got = [e for e in ests if e is not None]
    # [RSRoll-Bench P0#2] 缺陷钉已按本用例自己的指示翻转。
    #
    # 原用例断言"静止纹理仍会拿到角度"(len(got) >= 4),并在失败信息里写明:
    #   "若这条 FAIL 说明幅度门被修好了,请把本用例改成断言 None"。
    # 幅度门已由 P0#2 从**幅度之和** `weight_total < 1.0` 改为**平均幅度**
    # `mean_mag < MIN_MEAN_FLOW_PX (=0.1 px/帧对)`：旧阈值是绝对量，
    # 与分辨率、帧对数无关地累加，120×208×16 对早就 >> 1，门永远不响。
    # 于是这里翻转成正向断言：静止输入**必须**整块 None。
    check("textured_static_returns_none(P0#2 修复后)", len(got) == 0,
          f"{ests} —— 幅度门应拒答静止输入；若这条 FAIL 说明门又退回了绝对阈值")
    # 与之配套：既然全是 None，就不存在"噪声方向"可比，原发散度断言随之失去对象。
    # 保留一条等价的正向陈述，避免把断言删成空白（删断言=悄悄降低覆盖）。
    check("no_noise_direction_to_compare(P0#2 修复后)", len(got) <= 1,
          f"仍有 {len(got)} 段静止视频拿到角度读数，噪声方向会被当成真方向")
    print(f"         静止视频读数 = {[None if e is None else round(e,1) for e in ests]}"
          f" —— 全 None 即为期望：静止基线不再拿到随机方向")


# =============================================================================
# E. 180° 约定(camera 角 vs content 角)
# =============================================================================

def _convention_cases(target_fn):
    """target_fn(waypoints)->目标角。正确实现给 content 角(= camera+180)。"""
    scene = _fabricate_demo_scene(420, 420, seed=11)
    win = (96, 166)
    T = 17
    out = []
    # 四个基本方向的**单段直线**:camera 角精确等于 0/90/180/270。
    for cam, wp in (
        (0.0,   [(230.0, 150.0), (230.0, 222.0)]),      # 窗口向东 → 内容向西
        (90.0,  [(266.0, 200.0), (194.0, 200.0)]),      # 窗口向北 → 内容向南
        (180.0, [(230.0, 222.0), (230.0, 150.0)]),
        (270.0, [(194.0, 200.0), (266.0, 200.0)]),
    ):
        vid = deterministic_translation(scene, wp, win, T)
        est = metrics.estimate_motion_direction(vid, flow_size=win, step=1)
        tgt = target_fn(wp)
        err = (metrics.compute_angular_error(est, tgt)
               if (est is not None and tgt is not None) else float("nan"))
        out.append((f"conv_cam{cam:g}", err < 8.0,
                    f"camera={cam:g}° 目标={tgt} 光流读数={est} 误差={err:.2f}°"))
    return out


def group_E():
    print("\n== E. 180° 约定:content 角 = camera 角 + 180 ==")
    win = (96, 166)
    wp_east = [(230.0, 150.0), (230.0, 222.0)]       # 窗口中心向东
    # 纯几何,不碰光流:三个函数必须在同一约定上
    check("camera_net_angle_east_is_0",
          abs(tm._camera_net_angle(np.asarray(wp_east)) - 0.0) < 1e-9,
          f"{tm._camera_net_angle(np.asarray(wp_east))}")
    check("direction_word_is_camera_word_E",
          direction_word_from_waypoints(wp_east) == "E",
          f"{direction_word_from_waypoints(wp_east)}")
    check("direction_target_is_content_angle_180",
          abs(direction_target_from_waypoints(wp_east) - 180.0) < 1e-9,
          f"{direction_target_from_waypoints(wp_east)} —— 目标角必须是 content 角"
          f"(camera 0° + 180),写成 camera 角会让整块角误差翻 180°")
    gtp = tm.gt_content_path(wp_east, 17, normalize=False)
    check("gt_content_path_goes_west_when_camera_goes_east",
          gtp[-1, 0] < -1e-6 and abs(gtp[-1, 1]) < 1e-6,
          f"终点 (x,y)={gtp[-1].tolist()} —— x 必须为负(内容西移)")

    # 端到端:真的滑一次窗,光流读数必须匹配 content 角而不是 camera 角
    run_cases(_convention_cases(direction_target_from_waypoints))
    scene = _fabricate_demo_scene(420, 420, seed=11)
    vid = deterministic_translation(scene, wp_east, win, 17)
    est = metrics.estimate_motion_direction(vid, flow_size=win, step=1)
    check("flow_vs_camera_angle_is_180_off",
          est is not None
          and metrics.compute_angular_error(est, 0.0) > 172.0,
          f"读数 {est};与 camera 角(0°)之差应≈180°,这正是方向性的证明")

    # score_trajectory 的 GT 目标也必须是 content 角
    st = tm.score_trajectory(vid, np.asarray(wp_east), [0.0],
                             num_frames=17, window_size=win)
    check("score_trajectory_gt_net_content_angle_is_180",
          st.get("gt_net_content_angle") is not None
          and abs(float(st["gt_net_content_angle"]) - 180.0) < 1e-6,
          f"{st.get('gt_net_content_angle')}")
    check("score_trajectory_net_direction_error_small_on_gt",
          st.get("net_direction_error") is not None
          and float(st["net_direction_error"]) < 8.0,
          f"{st.get('net_direction_error')}(GT 自己喂进去应当接近 0)")

    # 注入:目标角写成 **camera** 角(少加那 180°)—— 四条全红。
    def camera_target(wp):
        return tm._camera_net_angle(np.asarray(wp, dtype=np.float64))
    expect_mutation_caught("目标角写成 camera 角(漏了 +180)",
                           _convention_cases, camera_target, 4)


# =============================================================================
# F. 四方向命令的量化地板(论文直接引用的数字)
# =============================================================================

def group_F():
    print("\n== F. 四方向 prompt 的量化地板 ==")
    # direction_target_from_waypoints 把 GT 的**连续**净方向压到四个基向再 +180。
    # 于是即便模型 100% 服从命令,它与 GT 真实内容方向之间也必然差一个量化残差。
    # 在 camera 角均匀分布的假设下:残差 ~ U[0,45),均值 22.5°、最大 45°。
    angs = np.linspace(0.0, 360.0, 36001, endpoint=False)
    R = 100.0
    errs = []
    for a in angs:
        r = math.radians(a)
        wp = [(0.0, 0.0), (-R * math.sin(r), R * math.cos(r))]  # (cy,cx),cam 角 = a
        cmd = direction_target_from_waypoints(wp)               # 量化 content 角
        true_content = (a + 180.0) % 360.0
        errs.append(metrics.compute_angular_error(cmd, true_content))
    errs = np.asarray(errs)
    check("quantization_floor_mean_is_22p5", abs(errs.mean() - 22.5) < 0.05,
          f"均值 {errs.mean():.4f}°(论文引用的 ≈22.5° 地板)")
    check("quantization_floor_max_is_45", abs(errs.max() - 45.0) < 0.02,
          f"最大 {errs.max():.4f}°")
    check("quantization_floor_zero_at_cardinals",
          all(metrics.compute_angular_error(
              direction_target_from_waypoints(
                  [(0.0, 0.0), (-R * math.sin(math.radians(c)),
                                R * math.cos(math.radians(c)))]),
              (c + 180.0) % 360.0) < 1e-6 for c in (0.0, 90.0, 180.0, 270.0)),
          "GT 恰好走基本方向时量化残差应为 0")
    # 随机方向的期望是 90°(角误差折叠到 [0,180] 后的均匀分布均值)——
    # 论文里"随机期望 90°"这句的出处,顺手钉住。
    rng = np.random.default_rng(0)
    rnd = [metrics.compute_angular_error(x, 180.0)
           for x in rng.uniform(0, 360, 200000)]
    check("random_direction_expectation_is_90", abs(np.mean(rnd) - 90.0) < 0.5,
          f"{np.mean(rnd):.3f}°")


# =============================================================================
# G. 批量对齐守卫
# =============================================================================

def group_G():
    print("\n== G. compute_batch_angular_error 的对齐语义 ==")
    vids = [translated_video(4, 0), translated_video(0, -4), translated_video(-4, 0)]
    tgts = [0.0, None, 180.0]
    r = metrics.compute_batch_angular_error(vids, tgts, flow_size=(_H, _W), step=1)
    check("batch_skips_None_target", r["count"] == 2, f"count={r['count']}")
    check("batch_per_sample_aligned_with_videos",
          len(r["per_sample"]) == 3 and r["per_sample"][1] is None,
          f"{r['per_sample']}")
    check("batch_values_match_known_angles",
          r["per_sample"][0] is not None and r["per_sample"][0] < ANG_TOL
          and r["per_sample"][2] is not None and r["per_sample"][2] < ANG_TOL,
          f"{r['per_sample']}")
    # 长度不等必须**炸**:曾经的 bug 是 zip 按短的截断 → 视频与目标角整体错位,
    # 角误差全错却一声不吭。
    try:
        metrics.compute_batch_angular_error(vids, [0.0], flow_size=(_H, _W), step=1)
        check("batch_length_mismatch_raises", False, "长度不等却没有抛异常")
    except ValueError:
        check("batch_length_mismatch_raises", True)


# =============================================================================

def main() -> int:
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[FATAL] 本文件全部是真跑,缺 cv2 就没有可验证的内容 —— 判失败,"
              "不产出'跳过即通过'的假绿。", file=sys.stderr)
        return 3

    group_A()
    group_B()
    group_C()
    group_D()
    group_D3()
    group_E()
    group_F()
    group_G()

    print(f"\n汇总: {len(PASS)} PASS / {len(FAIL)} FAIL / {len(SKIP)} SKIP")
    if SKIP:
        print("SKIP(依赖缺失,不算通过): " + ", ".join(SKIP))
    if FAIL:
        print("结果: FAIL —— " + ", ".join(FAIL))
        return 4
    print("结果: 全绿 ✓(每组都经过故障注入验证:注入错误后确实变红)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
