#!/usr/bin/env python3
"""
Directional camera trajectory generator for Plücker pose injection.

Replaces the identity-matrix dummy poses used in build_from_remotesense.py
with pose trajectories that actually encode translation direction, so the
video diffusion base model (Wan2.2 i2v-A14B) can learn direction-conditioned
spatial extrapolation via its built-in Plücker embedding pipeline.

Conventions (matching wan/utils/cam_utils.py and benchmark v5):
    - Poses are camera-to-world (c2w), OpenCV coord system (X right, Y down, Z forward)
    - Stored as float32 (T, 4, 4) in HDF5
    - Downstream: compute_relative_poses() subtracts frame-0 and normalizes
      max ||t|| to 1.0, so absolute translation scale is irrelevant — only
      direction and relative step sizes matter
    - Direction angle θ convention (from benchmark v5 line 483):
          θ = arctan2(-(cy_end - cy_start), cx_end - cx_start)
          θ = 0°   → +cx (east / image right)
          θ = 90°  → -cy (north / image up)
          θ = 180° → -cx (west / image left)
          θ = 270° → +cy (south / image down)

Pilot flow (per project memo):
    1. benchmark emits waypoints in pixel coordinates
    2. derive per-frame positions by interpolating waypoints over T frames
    3. convert pixel displacement → world-unit translation via an assumed
       focal length so that a one-window pixel shift ≈ 1.0 world unit
    4. keep rotation = identity (pure translation, camera orientation fixed
       looking straight down at the ground plane)

Usage:
    from camera_trajectory import make_poses_from_waypoints

    poses = make_poses_from_waypoints(
        num_frames=T,
        waypoints=[(cy0, cx0), (cy1, cx1), ...],   # pixel coords
        scene_size=(512, 512),                      # full scene HxW
        window_size=(96, 96),                       # sliding window HxW
    )  # returns (T, 4, 4) float32 c2w matrices

CLI (smoke test):
    python camera_trajectory.py --self-test
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Iterable, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core: direction from waypoints
# ---------------------------------------------------------------------------

def direction_angle_from_waypoints(waypoints: Sequence[Tuple[float, float]]) -> float:
    """Compute dominant motion direction θ (degrees, [0, 360)) from waypoints.

    Uses start→end displacement (not per-segment mean), matching the
    prompt-generation logic intended for build_dataset_from_benchmark.py
    line 498-507. θ convention matches benchmark v5 line 483.
    """
    if len(waypoints) < 2:
        return 0.0
    wp = np.asarray(waypoints, dtype=np.float64)
    dy = wp[-1, 0] - wp[0, 0]  # cy-axis (image y, down-positive)
    dx = wp[-1, 1] - wp[0, 1]  # cx-axis (image x, right-positive)
    theta = math.degrees(math.atan2(-dy, dx)) % 360.0
    return float(theta)


# ---------------------------------------------------------------------------
# Core: pixel waypoints → per-frame world translation
# ---------------------------------------------------------------------------

def interpolate_pixel_positions(
    waypoints: Sequence[Tuple[float, float]],
    num_frames: int,
) -> np.ndarray:
    """Linearly interpolate waypoints to T frame positions in pixel coords.

    Returns (T, 2) array of (cy, cx).
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    if len(wp) == 1:
        return np.repeat(wp, num_frames, axis=0)
    src = np.linspace(0.0, 1.0, len(wp))
    tgt = np.linspace(0.0, 1.0, num_frames)
    cy = np.interp(tgt, src, wp[:, 0])
    cx = np.interp(tgt, src, wp[:, 1])
    return np.stack([cy, cx], axis=-1)


def pixel_to_world_translation(
    pixel_positions: np.ndarray,
    window_size: Tuple[int, int],
) -> np.ndarray:
    """Convert pixel positions (T, 2: cy, cx) to world-unit translations (T, 3).

    Scale rule (overlap-aware, per user spec):
        Use window dimension as the unit length so that a translation equal to
        one window's width/height corresponds to a 1.0 world-unit shift. This
        means "move by one full window" ≈ unit step; 50% overlap between
        consecutive windows ≈ 0.5 world units per pair. Downstream
        compute_relative_poses() re-normalizes the max step to 1.0 anyway,
        so this choice only affects relative step *ratios*, which here are
        faithfully preserved because the mapping is linear.

    OpenCV axis convention:
        X (c2w row 0) ← +cx direction (image right, i.e. east)
        Y (c2w row 1) ← +cy direction (image down, i.e. south)
        Z (c2w row 2) ← forward (into scene); we keep Z = 0 for ground-plane
                        aerial motion (camera translates parallel to the ground)
    """
    win_h, win_w = window_size
    cy = pixel_positions[:, 0]
    cx = pixel_positions[:, 1]
    # Anchor translation so that frame 0 sits at the origin. Downstream
    # compute_relative_poses re-references to frame 0, but anchoring here
    # keeps the raw HDF5 poses interpretable on their own.
    tx = (cx - cx[0]) / float(win_w)
    ty = (cy - cy[0]) / float(win_h)
    tz = np.zeros_like(tx)
    return np.stack([tx, ty, tz], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Core: c2w matrices (identity rotation, translation from above)
# ---------------------------------------------------------------------------

def make_poses_from_waypoints(
    num_frames: int,
    waypoints: Sequence[Tuple[float, float]],
    scene_size: Tuple[int, int] = (512, 512),
    window_size: Tuple[int, int] = (96, 96),
) -> np.ndarray:
    """Generate (T, 4, 4) c2w pose matrices from benchmark waypoints.

    Rotation is kept at identity (camera orientation fixed, nadir-looking).
    Only the translation column encodes the sliding-window motion.

    scene_size is accepted for API symmetry with the existing dummy helper
    but is not needed since window_size determines the normalization scale.
    """
    del scene_size  # unused; kept for signature compatibility

    pix = interpolate_pixel_positions(waypoints, num_frames)
    trans = pixel_to_world_translation(pix, window_size)

    poses = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
    poses[:, :3, 3] = trans
    return poses


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    """Smoke tests covering the four canonical directions + sanity checks."""
    failures: list[str] = []

    def _check(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    # (cy, cx) waypoints. Scene 512, window 96.
    cases = {
        "east (θ=0°)":   ([(256, 128), (256, 384)],   0.0),
        "north (θ=90°)": ([(384, 256), (128, 256)],  90.0),
        "west (θ=180°)": ([(256, 384), (256, 128)], 180.0),
        "south (θ=270°)":([(128, 256), (384, 256)], 270.0),
    }

    print("\n[1/3] direction_angle_from_waypoints")
    for label, (wps, expected) in cases.items():
        theta = direction_angle_from_waypoints(wps)
        _check(label, abs(theta - expected) < 1e-3, f"got {theta:.1f}°")

    print("\n[2/3] make_poses_from_waypoints — structural invariants")
    for label, (wps, expected_theta) in cases.items():
        poses = make_poses_from_waypoints(
            num_frames=9, waypoints=wps, window_size=(96, 96),
        )
        _check(f"{label}: shape", poses.shape == (9, 4, 4))
        _check(f"{label}: dtype", poses.dtype == np.float32)
        # rotation = identity
        rot_ok = np.allclose(poses[:, :3, :3], np.eye(3), atol=1e-6)
        _check(f"{label}: rotation = I", rot_ok)
        # frame 0 at origin
        origin_ok = np.allclose(poses[0, :3, 3], 0.0, atol=1e-6)
        _check(f"{label}: t[0] = 0", origin_ok)
        # last-frame translation direction should match θ in the OpenCV plane
        tx, ty, tz = poses[-1, :3, 3]
        # recover angle (degrees): atan2(-ty, tx) because +ty = south, we flip
        # back to the math convention used by direction_angle_from_waypoints
        recovered = math.degrees(math.atan2(-ty, tx)) % 360.0
        _check(f"{label}: t direction", abs(recovered - expected_theta) < 1e-3,
               f"t=({tx:+.3f}, {ty:+.3f}, {tz:+.3f}) → {recovered:.1f}°")
        _check(f"{label}: tz = 0", abs(tz) < 1e-8)

    print("\n[3/3] scale: one full-window shift = 1.0 world unit")
    # cx shifts by 96 pixels (= win_w) → |tx| should be 1.0
    poses = make_poses_from_waypoints(
        num_frames=2,
        waypoints=[(256, 100), (256, 196)],  # Δcx = 96 = win_w
        window_size=(96, 96),
    )
    tx = poses[-1, 0, 3]
    _check("dcx = win_w -> tx = 1.0", abs(tx - 1.0) < 1e-6, f"got tx={tx:.6f}")

    # monotonic interpolation
    poses = make_poses_from_waypoints(
        num_frames=17,
        waypoints=[(256, 100), (256, 292)],  # 2× window width
        window_size=(96, 96),
    )
    tx_seq = poses[:, 0, 3]
    _check("tx monotonic non-decreasing",
           bool(np.all(np.diff(tx_seq) >= -1e-8)),
           f"first={tx_seq[0]:.3f}, last={tx_seq[-1]:.3f}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s) — {failures}")
        return 1
    print("All checks passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--self-test", action="store_true",
                        help="run built-in smoke tests and exit")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
