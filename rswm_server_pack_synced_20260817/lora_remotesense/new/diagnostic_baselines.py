#!/usr/bin/env python3
"""
RSWM diagnostic-baseline video synthesizers.

Purpose
-------
Demonstrate that the RSWM benchmark + metric suite (lora_remotesense/new/metrics.py)
actually *discriminates* failure modes rather than rewarding a single degenerate
strategy. We synthesize four NON-generative baseline videos from a static aerial
scene + its GT sliding-window trajectory, each engineered to trip a different
metric axis:

  1. persistence               -> "model just holds the source image":
                                   ~0 flow magnitude, direction unreliable/random,
                                   high SSIM on the earliest frames only.
  2. deterministic_translation -> construction-aware UPPER BOUND: crop the true
                                   sampling window as it slides along GT waypoints.
                                   Near-perfect direction/path + high SSIM, but it
                                   is a scene lookup, NOT a generative model.
  3. random_direction          -> "good visual, wrong direction" control: slides a
                                   window along a constant, deterministically chosen
                                   direction unrelated to the GT heading. Sharp
                                   frames, correct flow *magnitude*, large angular
                                   error.
  4. gt_oracle                 -> identity passthrough of the GT frames = metric
                                   CEILING (SSIM=1, PSNR=inf, angular error at the
                                   quantization floor).

Convention notes (all cite the shared RSWM code):
  - Poses/waypoints: waypoints are (cy, cx) window-CENTER pixel coords; the window
    is cropped scene[cy-win_h//2 : cy+win_h//2, cx-win_w//2 : cx+win_w//2]
    (wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py:565-576).
  - Per-frame center positions are the waypoints interpolated over T frames with
    uniform normalized knots, matching
    lora_training/camera_trajectory.py:79-94 (interpolate_pixel_positions).
  - Camera-direction angle theta = atan2(-(cy_end-cy_start), cx_end-cx_start) with
    east=0, north=90, west=180, south=270 (lora_training/camera_trajectory.py:59-72,
    build_dataset_from_benchmark.py:580-590).
  - CONTENT optical flow = MINUS camera motion (window slides east -> scene content
    inside the frame flows west). The compliance target for a direction word is the
    camera direction rotated 180 deg, so a faithful generation scores ~22.5 deg and
    a word-following drift ~180 deg (see metrics.py angular convention).
  - Scoring: metrics.compute_ssim/psnr and estimate_motion_direction crop to the
    common min(H,W), so baselines need not match GT resolution byte-for-byte.
    Track B v2 (trajectory_metrics.score_trajectory) computes ALL flow at the
    NATIVE WINDOW resolution (server calibration: 480x832 upscales put content
    displacement ~34 px/frame, beyond Farneback's reliable range; gt_oracle
    scored net_direction_error=80.67 deg until frames were downsampled back to
    the 128x222 window). Therefore run_baselines() by default keeps
    deterministic_translation / random_direction at their native window
    resolution and does NOT upscale them to GT resolution -- upscaling would
    only re-inflate per-frame displacement past the flow range. Callers that
    need a uniform resolution for SSIM/PSNR-type tables can pass
    resize_to_gt=True (or call the _resize_video helper themselves).

GPU / model baselines (NOT implemented here)
--------------------------------------------
Two further baselines require a neural model + GPU and are intentionally left as
NotImplementedError stubs at the bottom of this file:
  - image_outpainting_baseline : per-frame single-image outpainting of the source
    window in the requested direction (e.g. a diffusion outpainter). Tests whether
    a purely spatial generator can fake temporal motion.
  - rswise_recursive_baseline  : autoregressive/recursive world-model rollout
    (RSWISE-style) that feeds each generated frame back as the next conditioning
    frame. Tests error accumulation / consistency over a long horizon.
Add them in a separate GPU-enabled module or wire a model handle into these stubs.

Each public synthesizer returns np.ndarray [T, H, W, 3] uint8 aligned to the GT
frame count. No global RNG is used inside any baseline (reproducibility): the only
np.random calls live in __main__ for fabricating a demo scene.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# small self-contained helpers (no heavy imports: numpy + optional cv2 only)
# ============================================================================

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip to [0,255] and cast to contiguous uint8."""
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))


def _interpolate_positions(
    waypoints: Sequence[Tuple[float, float]],
    num_frames: int,
) -> np.ndarray:
    """Interpolate (cy, cx) waypoints to `num_frames` center positions.

    Mirrors lora_training/camera_trajectory.py:79-94 (uniform normalized knots +
    np.interp) so the synthesized window path is byte-consistent with the pose
    trajectory the dataset builder injects.
    Returns (T, 2) float64 array of (cy, cx).
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.ndim != 2 or wp.shape[1] != 2:
        raise ValueError(f"waypoints must be (N,2) (cy,cx); got shape {wp.shape}")
    if len(wp) == 1:
        return np.repeat(wp, num_frames, axis=0)
    if num_frames == 1:
        return wp[:1].copy()
    src = np.linspace(0.0, 1.0, len(wp))
    tgt = np.linspace(0.0, 1.0, num_frames)
    cy = np.interp(tgt, src, wp[:, 0])
    cx = np.interp(tgt, src, wp[:, 1])
    return np.stack([cy, cx], axis=-1)


def _camera_angle_from_waypoints(waypoints: Sequence[Tuple[float, float]]) -> float:
    """Camera heading theta in degrees [0,360) from start->end displacement.

    theta = atan2(-(cy_end-cy_start), cx_end-cx_start); east=0/north=90/west=180/
    south=270 (lora_training/camera_trajectory.py:59-72,
    build_dataset_from_benchmark.py:580-590).
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if len(wp) < 2:
        return 0.0
    dy = wp[-1, 0] - wp[0, 0]
    dx = wp[-1, 1] - wp[0, 1]
    return float(math.degrees(math.atan2(-dy, dx)) % 360.0)


def _crop_window_reflect(
    scene: np.ndarray,
    cy: float,
    cx: float,
    win_h: int,
    win_w: int,
) -> np.ndarray:
    """Crop a (win_h, win_w) window centered at (cy, cx), reflect-padding at edges.

    Window spans rows [top, top+win_h), cols [left, left+win_w) with
    top = round(cy) - win_h//2, so the returned crop is EXACTLY (win_h, win_w)
    with no resize/stretch (unlike the benchmark extractor which clamps+resizes,
    v5:568-576). Reflect padding keeps texture continuous for optical flow.
    """
    scene = np.asarray(scene)
    H, W = scene.shape[:2]
    top = int(round(cy)) - win_h // 2
    left = int(round(cx)) - win_w // 2

    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, top + win_h - H)
    pad_right = max(0, left + win_w - W)

    if pad_top or pad_bottom or pad_left or pad_right:
        pad_spec = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        # reflect requires pad < dim; fall back to edge for pathologically small scenes
        try:
            scene_p = np.pad(scene, pad_spec, mode="reflect")
        except ValueError:
            scene_p = np.pad(scene, pad_spec, mode="edge")
        y0 = top + pad_top
        x0 = left + pad_left
    else:
        scene_p = scene
        y0 = top
        x0 = left

    return scene_p[y0:y0 + win_h, x0:x0 + win_w]


# Minimum scene-to-window size ratio below which `scene_image` cannot be a real
# full scene (it is then almost certainly a single WINDOW frame). See
# _check_scene_vs_window for the server incident this guards against.
_MIN_SCENE_TO_WINDOW_RATIO = 1.5


def _check_scene_vs_window(scene: np.ndarray, win_h: int, win_w: int) -> None:
    """Fail loudly when the caller passed a window frame as `scene_image`.

    Server incident (Track B calibration): run_trajectory_eval.py passed
    scene_image=g['image'] -- the FIRST FRAME WINDOW (a 480x832 upscale of the
    128x222 window), NOT the full 512x512 scene. Sliding a window through that
    with scene-pixel waypoints produced near-garbage videos and
    deterministic_translation / random_direction scored valid_pct=0%.

    Heuristic: a real scene must be at least ~1.5x the window in BOTH
    dimensions, otherwise a window cannot meaningfully slide inside it.
    Raises ValueError naming the mistake; never degrades silently.
    """
    H, W = int(scene.shape[0]), int(scene.shape[1])
    min_h = int(math.ceil(_MIN_SCENE_TO_WINDOW_RATIO * win_h))
    min_w = int(math.ceil(_MIN_SCENE_TO_WINDOW_RATIO * win_w))
    if H < min_h or W < min_w:
        raise ValueError(
            f"scene_image ({H}x{W}) is smaller than {_MIN_SCENE_TO_WINDOW_RATIO}x "
            f"the window ({win_h}x{win_w}) -- this looks like a WINDOW FRAME, not "
            f"a full scene. The per-sample first-frame 'image' dataset is an "
            f"upscale of the first window crop, NOT the full scene; passing it "
            f"here made deterministic/random baselines score valid_pct=0% on the "
            f"server. Pass the true 'full_scene' dataset (512x512; backfill old "
            f"H5s with server_run/augment_h5_trajectory.py) or omit scene_image "
            f"to skip the scene-dependent baselines."
        )


def _resize_frame(frame: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a single (H,W,3) frame to target (H,W). cv2 if available else numpy NN."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    if frame.shape[0] == th and frame.shape[1] == tw:
        return frame
    try:
        import cv2
        return cv2.resize(_to_uint8(frame), (tw, th), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        h_idx = np.linspace(0, frame.shape[0] - 1, th).astype(int)
        w_idx = np.linspace(0, frame.shape[1] - 1, tw).astype(int)
        return frame[np.ix_(h_idx, w_idx)]


def _resize_video(video: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize every frame of a [T,H,W,3] video to a common (H,W)."""
    th, tw = int(target_hw[0]), int(target_hw[1])
    if video.shape[1] == th and video.shape[2] == tw:
        return _to_uint8(video)
    return _to_uint8(np.stack([_resize_frame(f, (th, tw)) for f in video], axis=0))


def _mean_flow_magnitude(frames: np.ndarray) -> float:
    """Mean per-pixel Farneback flow magnitude across sampled frame pairs.

    Uses the same Farneback parameters as metrics.estimate_motion_direction
    (metrics.py:268-277). Returns 0.0 if cv2 is unavailable or T<2. This is a
    diagnostic-only summary (metrics.py exposes direction, not magnitude).
    """
    try:
        import cv2
    except ImportError:
        return 0.0
    frames = _to_uint8(frames)
    if len(frames) < 2:
        return 0.0
    step = max(1, len(frames) // 8)
    mags = []
    for i in range(0, len(frames) - step, step):
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(frames[i + step], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(mag.mean()))
    return float(np.mean(mags)) if mags else 0.0


# ============================================================================
# public baselines
# ============================================================================

def persistence(source_frame: np.ndarray, T: int) -> np.ndarray:
    """Static baseline: repeat the source frame for every one of T frames.

    Simulates a model that merely holds the input image. Expected metric
    signature: ~0 flow magnitude, unreliable/None or random estimated direction,
    high SSIM only on the earliest frames (before GT has drifted away).

    Args:
        source_frame: (H, W, 3) image (uint8 or float in [0,255]).
        T: number of output frames (GT frame count).
    Returns:
        (T, H, W, 3) uint8 video.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1; got {T}")
    src = _to_uint8(np.asarray(source_frame))
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError(f"source_frame must be (H,W,3); got {src.shape}")
    return np.repeat(src[None, ...], T, axis=0)


def deterministic_translation(
    scene_image: np.ndarray,
    waypoints: Sequence[Tuple[float, float]],
    window_size: Tuple[int, int],
    T: int,
) -> np.ndarray:
    """Construction-aware UPPER BOUND: crop the true window along GT waypoints.

    Slides a (win_h, win_w) window whose center follows the GT waypoints,
    interpolated to T frames (camera_trajectory.py:79-94). Reflect-pads at scene
    edges. By construction the direction/path are near-perfect and SSIM is high,
    but this is a scene lookup rather than a generative model -- it needs the full
    static scene, which a real world model does not see.

    Args:
        scene_image: (H, W, 3) full static aerial scene (uint8 or float).
        waypoints:   sequence of (cy, cx) window-center pixel coords.
        window_size: (win_h, win_w).
        T:           number of output frames (GT frame count).
    Returns:
        (T, win_h, win_w, 3) uint8 video.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1; got {T}")
    scene = np.asarray(scene_image)
    if scene.ndim != 3 or scene.shape[2] != 3:
        raise ValueError(f"scene_image must be (H,W,3); got {scene.shape}")
    win_h, win_w = int(window_size[0]), int(window_size[1])
    centers = _interpolate_positions(waypoints, T)  # (T, 2) (cy, cx)
    frames = np.stack(
        [_crop_window_reflect(scene, cy, cx, win_h, win_w) for (cy, cx) in centers],
        axis=0,
    )
    return _to_uint8(frames)


def random_direction_translation(
    scene_image: np.ndarray,
    window_size: Tuple[int, int],
    T: int,
    dir_index: int,
    step_px: Optional[float] = None,
    start_center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Wrong-direction control: slide the window along a constant fixed direction.

    The direction is derived DETERMINISTICALLY from the integer dir_index (no
    global RNG): theta_cam = (dir_index * 90) deg in the camera-heading frame
    (east=0/north=90/west=180/south=270). The window starts at `start_center`
    (default = scene center) and marches `step_px` per frame. Sharp frames and
    correct flow magnitude, but a heading unrelated to the GT path -> large
    angular error. run_baselines() picks a dir_index guaranteed wrong vs the GT.

    Args:
        scene_image: (H, W, 3) full static aerial scene.
        window_size: (win_h, win_w).
        T:           number of output frames.
        dir_index:   integer selecting the constant camera heading (theta=90*idx).
        step_px:     per-frame center displacement in scene pixels. Default keeps
                     total travel ~0.4*min(H,W) so the window stays mostly on-scene.
        start_center: (cy, cx) start; default = scene center.
    Returns:
        (T, win_h, win_w, 3) uint8 video.
    """
    if T < 1:
        raise ValueError(f"T must be >= 1; got {T}")
    scene = np.asarray(scene_image)
    if scene.ndim != 3 or scene.shape[2] != 3:
        raise ValueError(f"scene_image must be (H,W,3); got {scene.shape}")
    H, W = scene.shape[:2]
    win_h, win_w = int(window_size[0]), int(window_size[1])

    # deterministic camera heading from the integer index
    theta = math.radians((float(dir_index) * 90.0) % 360.0)
    # camera heading -> pixel delta: theta = atan2(-dy, dx) => dx=cos, dy=-sin
    ux = math.cos(theta)
    uy = -math.sin(theta)

    if start_center is None:
        cy0, cx0 = H / 2.0, W / 2.0
    else:
        cy0, cx0 = float(start_center[0]), float(start_center[1])

    if step_px is None:
        total = 0.4 * float(min(H, W))
        step_px = total / max(1, T - 1)

    frames = []
    for t in range(T):
        d = step_px * t
        cy = cy0 + uy * d
        cx = cx0 + ux * d
        frames.append(_crop_window_reflect(scene, cy, cx, win_h, win_w))
    return _to_uint8(np.stack(frames, axis=0))


def gt_oracle(gt_frames: np.ndarray) -> np.ndarray:
    """Identity passthrough of the GT frames = metric CEILING.

    Args:
        gt_frames: (T, H, W, 3) GT video.
    Returns:
        (T, H, W, 3) uint8 copy of gt_frames.
    """
    gt = np.asarray(gt_frames)
    if gt.ndim != 4 or gt.shape[3] != 3:
        raise ValueError(f"gt_frames must be (T,H,W,3); got {gt.shape}")
    return _to_uint8(gt)


# ============================================================================
# batch driver
# ============================================================================

def run_baselines(sample_dict: Dict, resize_to_gt: bool = False) -> Dict[str, np.ndarray]:
    """Synthesize all four non-generative baselines for one benchmark sample.

    Args:
        sample_dict: dict with (keys are looked up flexibly):
            - 'scene_image' or 'full_scene': (H,W,3) full static scene (required
              for deterministic_translation / random_direction). Must be the TRUE
              full scene (e.g. the 512x512 'full_scene' dataset), NOT the
              per-sample first-frame 'image' (that is a window upscale; a
              too-small scene raises ValueError, see _check_scene_vs_window).
            - 'waypoints': sequence of (cy, cx) window-center pixel coords.
            - 'window_size': (win_h, win_w) native window size.
            - 'T' or 'num_frames': GT frame count (falls back to len(gt_frames)).
            - 'gt_frames' (optional): (T,H,W,3) GT video -> enables gt_oracle.
            - 'source_frame' (optional): (H,W,3) start image for persistence;
              defaults to gt_frames[0] else the first GT window crop.
        resize_to_gt: default False. When True, deterministic_translation /
            random_direction are additionally resized to the GT resolution via
            _resize_video -- ONLY for callers that need a uniform resolution for
            SSIM/PSNR-type tables. Track B scoring must use the default
            (native window resolution): its flow runs at window resolution
            anyway, and upscaling would re-inflate per-frame displacement past
            Farneback's reliable range (server calibration incident).

    Returns (per-video output resolution, with the default resize_to_gt=False):
        - 'persistence':               resolution of source_frame (normally the
                                       GT/generation resolution, 480x832) --
                                       unchanged legacy behavior.
        - 'deterministic_translation': NATIVE window resolution (win_h, win_w).
        - 'random_direction':          NATIVE window resolution (win_h, win_w).
        - 'gt_oracle':                 GT resolution (identity passthrough);
                                       omitted (absent key) when no gt_frames.
        deterministic/random are omitted (absent keys) when no scene is given.
        Track B v2 downsamples any larger-than-window video to window resolution
        before flow, so all four are scored on the same footing.
    """
    scene = sample_dict.get("scene_image", sample_dict.get("full_scene"))
    waypoints = sample_dict.get("waypoints")
    window_size = sample_dict.get("window_size")
    gt_frames = sample_dict.get("gt_frames")
    if gt_frames is not None:
        gt_frames = np.asarray(gt_frames)

    T = sample_dict.get("T", sample_dict.get("num_frames"))
    if T is None:
        if gt_frames is not None:
            T = len(gt_frames)
        else:
            raise ValueError("run_baselines: cannot determine T (no T/num_frames/gt_frames)")
    T = int(T)

    if waypoints is None or window_size is None:
        raise ValueError("run_baselines: 'waypoints' and 'window_size' are required")

    win_h, win_w = int(window_size[0]), int(window_size[1])

    # ---- source frame for persistence ----
    source_frame = sample_dict.get("source_frame")
    if source_frame is None:
        if gt_frames is not None and len(gt_frames) > 0:
            source_frame = np.asarray(gt_frames)[0]
        elif scene is not None:
            c0 = _interpolate_positions(waypoints, T)[0]
            source_frame = _crop_window_reflect(scene, c0[0], c0[1], win_h, win_w)
        else:
            raise ValueError("run_baselines: need source_frame, gt_frames, or scene_image")

    # ---- GT resolution (persistence legacy target; det/random only on request) ----
    if gt_frames is not None and len(gt_frames) > 0:
        target_hw = (int(gt_frames.shape[1]), int(gt_frames.shape[2]))
    else:
        sf = np.asarray(source_frame)
        target_hw = (int(sf.shape[0]), int(sf.shape[1]))

    out: Dict[str, np.ndarray] = {}

    # 1. persistence (unchanged): source_frame resolution == target_hw normally.
    out["persistence"] = _resize_video(persistence(source_frame, T), target_hw)

    # scene-dependent baselines
    if scene is not None:
        # Fail loudly if 'scene' is actually a window frame (first-frame 'image'
        # instead of 'full_scene') -- the server calibration incident.
        _check_scene_vs_window(np.asarray(scene), win_h, win_w)

        # 2. deterministic upper bound (true window along GT path).
        # Kept at NATIVE window resolution by default: Track B v2 computes flow
        # at window resolution, and upscaling to GT resolution would re-inflate
        # displacement to ~34 px/frame (beyond Farneback range).
        det = deterministic_translation(scene, waypoints, window_size, T)
        out["deterministic_translation"] = _resize_video(det, target_hw) if resize_to_gt else det

        # 3. wrong-direction control: pick a heading guaranteed off the GT.
        # GT camera quadrant q (0=E,1=N,2=W,3=S); use opposite (q+2) so the
        # content motion is flipped 180 deg from the compliance target.
        # Same native-window-resolution policy as deterministic_translation.
        gt_theta = _camera_angle_from_waypoints(waypoints)
        gt_quad = int(round(gt_theta / 90.0)) % 4
        dir_index = (gt_quad + 2) % 4
        # match deterministic's per-frame speed for a fair (direction-only) contrast
        wp = np.asarray(waypoints, dtype=np.float64)
        path_len = float(np.sum(np.linalg.norm(np.diff(wp, axis=0), axis=1))) if len(wp) > 1 else 0.0
        step_px = path_len / max(1, T - 1) if path_len > 0 else None
        rnd = random_direction_translation(
            scene, window_size, T, dir_index, step_px=step_px
        )
        out["random_direction"] = _resize_video(rnd, target_hw) if resize_to_gt else rnd

    # 4. gt oracle ceiling (only if GT provided)
    if gt_frames is not None and len(gt_frames) > 0:
        out["gt_oracle"] = gt_oracle(gt_frames)

    return out


# ============================================================================
# GPU / model baselines -- intentionally NOT implemented here
# ============================================================================

def image_outpainting_baseline(
    source_frame: np.ndarray,
    direction_angle: float,
    T: int,
    model=None,
    device: str = "cuda",
    inpaint_prompt: str = "",
    generator=None,
    num_inference_steps: int = 25,
) -> np.ndarray:
    """GPU baseline: per-frame single-image outpainting for fake motion.

    Idea: given the source window, repeatedly outpaint new content on the leading
    edge in the CONTENT-motion direction (camera direction rotated 180 deg) and
    crop back to window size, producing a translation-like sequence WITHOUT a
    temporal/world model. Tests whether a purely spatial generator can spoof the
    direction metric while failing consistency/FVD.

    Args:
        source_frame:    (H, W, 3) start window.
        direction_angle: camera-direction angle in degrees (east=0..south=270);
                         internally use the 180-rotated content-motion direction.
        T:               number of frames to synthesize.
        model:           a loaded outpainting/diffusion model handle (required).
        device:          torch device.
    Returns:
        (T, H, W, 3) uint8 video.

    [v2.5 已实现,2026-07-26] 拍板②档 A 的 construction-aware 负对照。
    rollout:每帧把当前窗口沿【内容运动方向】平移 step_px,前沿新暴露的条带
    用 diffusion inpainter 补全(其余像素原样保留)。方向由构造免费拿对,
    内容靠单图外推伪造 —— 用来回答审稿人必问的"为什么不直接平移+outpaint",
    并检验方向指标能否被非世界模型的纯空间生成器钻空子。

    model: diffusers StableDiffusionInpaintPipeline(或兼容对象,须可
           model(prompt=..., image=PIL, mask_image=PIL, ...) 调用并返回
           .images[0])。必填 —— 没有模型时抛错,绝不退化成平移(那是
           deterministic_translation,不同基线不许混淆)。
    generator: torch.Generator;逐帧传给 pipeline 以保证可复现(不传=每帧
           重新随机)。num_inference_steps 默认 25(速度/质量折中)。
    """
    if model is None:
        raise ValueError(
            "image_outpainting_baseline 需要一个已加载的 inpaint 模型 "
            "(diffusers StableDiffusionInpaintPipeline);无模型时【不得】"
            "退化为纯平移 —— 那是 deterministic_translation 基线。")
    from PIL import Image as _PILImage

    src = _to_uint8(np.asarray(source_frame))
    H, W = src.shape[:2]
    T = int(T)
    # 内容运动方向 = 相机方向 + 180(与全仓符号约定一致)
    content_deg = (float(direction_angle) + 180.0) % 360.0
    rad = math.radians(content_deg)
    # 每帧位移:与 GT 速度同量级(窗口短边的 ~7%),与 deterministic_translation
    # 的 step 口径一致 —— 两个基线的"运动量"必须可比,否则对照失去意义。
    step_px = max(1, int(round(min(H, W) * 0.07)))
    dx = math.cos(rad) * step_px
    dy = -math.sin(rad) * step_px          # 图像 y 向下

    prompt = str(inpaint_prompt or "aerial satellite imagery, seamless continuation")
    frames = [src]
    cur = src
    for _ in range(T - 1):
        sx, sy = int(round(dx)), int(round(dy))
        # 内容向 (dx,dy) 移动 = 把当前帧按该位移贴进新画布,空出的是"新暴露"区
        canvas = np.zeros_like(cur)
        mask = np.full((H, W), 255, np.uint8)      # 255=待重绘
        x0d, y0d = max(0, sx), max(0, sy)
        x0s, y0s = max(0, -sx), max(0, -sy)
        w = W - abs(sx)
        h = H - abs(sy)
        if w <= 0 or h <= 0:
            raise ValueError(f"step_px={step_px} 超过窗口尺寸 {H}x{W}")
        canvas[y0d:y0d + h, x0d:x0d + w] = cur[y0s:y0s + h, x0s:x0s + w]
        mask[y0d:y0d + h, x0d:x0d + w] = 0          # 已知区域不重绘
        # generator 必须逐帧显式传入 —— diffusers 不读 pipeline 上的自定义属性,
        # 不传就是"每次调用重新随机",基线不可复现(v2.5 审查抓到的真 bug)。
        kw = dict(prompt=prompt,
                  image=_PILImage.fromarray(canvas),
                  mask_image=_PILImage.fromarray(mask),
                  height=H, width=W,
                  num_inference_steps=int(num_inference_steps))
        if generator is not None:
            kw["generator"] = generator
        out = model(**kw).images[0]
        nxt = np.array(out.convert("RGB"), dtype=np.uint8)   # np.array=可写副本
        if nxt.shape[:2] != (H, W):
            nxt = np.array(out.convert("RGB").resize((W, H)), dtype=np.uint8)
        # 只接受重绘区域的新内容,已知区域强制保留(防 inpainter 漂移整幅)
        keep = mask == 0
        nxt[keep] = canvas[keep]
        frames.append(nxt)
        cur = nxt
    return np.stack(frames[:T])


def rswise_recursive_baseline(
    source_frame: np.ndarray,
    poses: np.ndarray,
    T: int,
    model=None,
    device: str = "cuda",
) -> np.ndarray:
    """GPU baseline (STUB): autoregressive/recursive world-model rollout (RSWISE).

    Idea: seed with the source window, then recursively feed each generated frame
    (plus the next relative pose) back as conditioning to roll out T frames. Tests
    error accumulation / drift and long-horizon consistency (FVD, late-frame SSIM)
    that single-shot baselines cannot expose.

    Args:
        source_frame: (H, W, 3) seed frame.
        poses:        (T, 4, 4) c2w pose trajectory (OpenCV, identity rotation +
                      window-normalized translation; camera_trajectory.py).
        T:            number of frames to synthesize.
        model:        a loaded recursive world model handle (required).
        device:       torch device.
    Returns:
        (T, H, W, 3) uint8 video.

    Not implemented: requires a GPU world model; wire a `model` handle and the
    recursive sampling loop in a GPU-enabled module.
    """
    raise NotImplementedError(
        "rswise_recursive_baseline is a GPU/model baseline; implement the "
        "recursive rollout with a loaded world model in a GPU-enabled module."
    )


# ============================================================================
# demo
# ============================================================================

def _fabricate_demo_scene(H: int = 200, W: int = 200, seed: int = 0) -> np.ndarray:
    """Textured static scene so Farneback flow is estimable (demo only)."""
    rng = np.random.default_rng(seed)  # RNG confined to the demo, not the baselines
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = 128 + 100 * np.sin(xx / 11.0) * np.cos(yy / 13.0)
    g = 128 + 100 * np.sin((xx + yy) / 9.0)
    b = 128 + 100 * np.cos(xx / 7.0) * np.sin(yy / 17.0)
    scene = np.stack([r, g, b], axis=-1)
    scene = scene + rng.normal(0, 8, size=scene.shape)  # break up flat regions
    return _to_uint8(scene)


def _demo() -> int:
    H = W = 200
    win_h, win_w = 64, 111  # ~480:832 aspect (build_dataset DEFAULT 128x222 halved)
    T = 16

    scene = _fabricate_demo_scene(H, W)

    # L-path (2 segments = one heading change): go east, then north.
    # waypoints are (cy, cx) window centers.
    waypoints = [(120, 60), (120, 140), (60, 140)]
    gt_theta = _camera_angle_from_waypoints(waypoints)

    # A stand-in "GT" video: the true deterministic crop (so gt_oracle has input).
    gt_frames = deterministic_translation(scene, waypoints, (win_h, win_w), T)

    sample = {
        "scene_image": scene,
        "waypoints": waypoints,
        "window_size": (win_h, win_w),
        "T": T,
        "gt_frames": gt_frames,
    }

    print(f"demo scene {scene.shape}, window ({win_h},{win_w}), T={T}")
    print(f"GT camera heading theta = {gt_theta:.1f} deg "
          f"(content-motion target ~ {(gt_theta + 180) % 360:.1f} deg)")

    # individual functions
    print("\n-- individual baselines --")
    src = gt_frames[0]
    p = persistence(src, T)
    d = deterministic_translation(scene, waypoints, (win_h, win_w), T)
    r = random_direction_translation(scene, (win_h, win_w), T, dir_index=2)
    o = gt_oracle(gt_frames)
    for name, vid in [("persistence", p), ("deterministic_translation", d),
                      ("random_direction", r), ("gt_oracle", o)]:
        print(f"  {name:26s} shape={tuple(vid.shape)} "
              f"mean_flow_mag={_mean_flow_magnitude(vid):.4f}")

    # batch driver (det/random stay at native window resolution by default;
    # pass resize_to_gt=True only for SSIM-type uniform-resolution tables)
    print("\n-- run_baselines (det/random at native window resolution) --")
    out = run_baselines(sample)
    for name, vid in out.items():
        print(f"  {name:26s} shape={tuple(vid.shape)} "
              f"mean_flow_mag={_mean_flow_magnitude(vid):.4f}")

    # sanity expectations (not asserted; printed for the reader)
    print("\nexpected: persistence flow ~0; deterministic/gt_oracle direction "
          "faithful; random_direction large flow but wrong heading.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_demo())
