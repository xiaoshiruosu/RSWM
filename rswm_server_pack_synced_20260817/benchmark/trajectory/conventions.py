"""Frozen RSWM-Bench V1 direction and overlap conventions."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

from benchmark.core.errors import ContractError
from .types import DirectionTarget

DIRECTION_CONVENTION_VERSION = "rswm.direction/cy-cx-camera-v1"
OVERLAP_DEFINITION_VERSION = "rswm.overlap/condition-window-area-v1"
NET_DIRECTION_MIN_RATIO = 0.2


def normalize_angle(value: float) -> float:
    angle = float(value) % 360.0
    if not math.isfinite(angle):
        raise ContractError("direction angle must be finite")
    return angle


def camera_bearing_deg(delta_cy: float, delta_cx: float) -> float:
    if abs(float(delta_cy)) < 1e-12 and abs(float(delta_cx)) < 1e-12:
        raise ContractError("direction is undefined for zero displacement")
    return normalize_angle(math.degrees(math.atan2(-float(delta_cy), float(delta_cx))))


def content_bearing_deg(camera_bearing: float) -> float:
    return normalize_angle(float(camera_bearing) + 180.0)


def direction_bin(camera_bearing: float) -> str:
    angle = normalize_angle(camera_bearing)
    centers = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
    return centers[int(((angle + 22.5) % 360.0) // 45.0)]


def overlap_level(endpoint_overlap: float) -> Optional[str]:
    value = float(endpoint_overlap)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ContractError("endpoint overlap must be finite and in [0,1]")
    if 0.70 <= value < 0.90:
        return "L1_overlap_80"
    if 0.40 <= value < 0.60:
        return "L2_overlap_50"
    if 0.15 <= value < 0.25:
        return "L3_overlap_20"
    if 0.00 <= value < 0.10:
        return "L4_overlap_lt10"
    return None


def build_direction_target(waypoints: Iterable[Iterable[float]]) -> DirectionTarget:
    points = np.asarray(tuple(tuple(point) for point in waypoints), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ContractError(f"waypoints must have shape (K+1,2), got {points.shape}")
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    if np.any(lengths <= 1e-12):
        raise ContractError("waypoints contain a zero-length segment")
    segment_camera = tuple(camera_bearing_deg(delta[0], delta[1]) for delta in deltas)
    segment_content = tuple(content_bearing_deg(value) for value in segment_camera)
    net_delta = points[-1] - points[0]
    net_length = float(np.linalg.norm(net_delta))
    total_length = float(lengths.sum())
    ratio = net_length / total_length
    if net_length <= 1e-12:
        # A near-closed path has no meaningful net bearing.  A deterministic
        # placeholder is retained only for serialization; well_posed is False.
        net_camera = 0.0
    else:
        net_camera = camera_bearing_deg(net_delta[0], net_delta[1])
    return DirectionTarget(
        camera_bearing_deg=net_camera,
        content_bearing_deg=content_bearing_deg(net_camera),
        segment_camera_bearings_deg=segment_camera,
        segment_content_bearings_deg=segment_content,
        eight_way_bin=direction_bin(net_camera),
        net_displacement_ratio=ratio,
        net_direction_well_posed=(ratio >= NET_DIRECTION_MIN_RATIO),
        convention_version=DIRECTION_CONVENTION_VERSION,
    )
