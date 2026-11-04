"""Immutable trajectory, direction and overlap types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from benchmark.core.errors import ContractError

OVERLAP_LEVELS = frozenset(
    {"L1_overlap_80", "L2_overlap_50", "L3_overlap_20", "L4_overlap_lt10"}
)


def readonly_array(value, *, dtype=None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class DirectionTarget:
    camera_bearing_deg: float
    content_bearing_deg: float
    segment_camera_bearings_deg: Tuple[float, ...]
    segment_content_bearings_deg: Tuple[float, ...]
    eight_way_bin: str
    net_displacement_ratio: float
    net_direction_well_posed: bool
    convention_version: str = "rswm.direction/cy-cx-camera-v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("camera_bearing_deg", self.camera_bearing_deg),
            ("content_bearing_deg", self.content_bearing_deg),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) < 360.0:
                raise ContractError(f"{label} must be finite and in [0,360)")
        if not 0.0 <= float(self.net_displacement_ratio) <= 1.0 + 1e-9:
            raise ContractError("net_displacement_ratio must be in [0,1]")
        object.__setattr__(
            self,
            "segment_camera_bearings_deg",
            tuple(float(value) for value in self.segment_camera_bearings_deg),
        )
        object.__setattr__(
            self,
            "segment_content_bearings_deg",
            tuple(float(value) for value in self.segment_content_bearings_deg),
        )
        if len(self.segment_camera_bearings_deg) != len(
            self.segment_content_bearings_deg
        ):
            raise ContractError("camera/content segment bearing counts differ")


@dataclass(frozen=True)
class OverlapProfile:
    condition_to_frame: Tuple[float, ...]
    adjacent: Tuple[float, ...]
    endpoint: float
    mean_future: float
    minimum_future: float
    maximum_future: float
    level: Optional[str]
    definition_version: str = "rswm.overlap/condition-window-area-v1"

    def __post_init__(self) -> None:
        condition = tuple(float(value) for value in self.condition_to_frame)
        adjacent = tuple(float(value) for value in self.adjacent)
        if len(condition) < 2:
            raise ContractError("overlap profile must contain t=0 and at least one future frame")
        if len(adjacent) != len(condition) - 1:
            raise ContractError("adjacent overlap must contain exactly T-1 values")
        for label, values in (("condition_to_frame", condition), ("adjacent", adjacent)):
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
                raise ContractError(f"{label} overlap values must be finite and in [0,1]")
        future = condition[1:]
        expected = {
            "endpoint": future[-1],
            "mean_future": sum(future) / len(future),
            "minimum_future": min(future),
            "maximum_future": max(future),
        }
        for label, expected_value in expected.items():
            if abs(float(getattr(self, label)) - expected_value) > 1e-8:
                raise ContractError(f"{label} disagrees with condition_to_frame profile")
        if self.level is not None and self.level not in OVERLAP_LEVELS:
            raise ContractError(f"unknown overlap level: {self.level!r}")
        object.__setattr__(self, "condition_to_frame", condition)
        object.__setattr__(self, "adjacent", adjacent)


@dataclass(frozen=True)
class TrajectoryPlan:
    per_frame_centers: np.ndarray
    waypoints: np.ndarray
    poses: Optional[np.ndarray]
    intrinsics: Optional[np.ndarray]
    direction: DirectionTarget
    overlap: OverlapProfile
    source_algorithm: str
    source_version: str

    def __post_init__(self) -> None:
        centers = readonly_array(self.per_frame_centers, dtype=np.float64)
        waypoints = readonly_array(self.waypoints, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) < 2:
            raise ContractError(
                f"per_frame_centers must have shape (T,2), got {centers.shape}"
            )
        if waypoints.ndim != 2 or waypoints.shape[1] != 2 or len(waypoints) < 2:
            raise ContractError(f"waypoints must have shape (K+1,2), got {waypoints.shape}")
        poses = None
        if self.poses is not None:
            poses = readonly_array(self.poses, dtype=np.float64)
            if poses.shape != (len(centers), 4, 4):
                raise ContractError(
                    f"poses must have shape ({len(centers)},4,4), got {poses.shape}"
                )
        intrinsics = None
        if self.intrinsics is not None:
            intrinsics = readonly_array(self.intrinsics, dtype=np.float64)
            if intrinsics.shape != (len(centers), 4):
                raise ContractError(
                    f"intrinsics must have shape ({len(centers)},4), got {intrinsics.shape}"
                )
        if len(self.overlap.condition_to_frame) != len(centers):
            raise ContractError("trajectory and overlap profile frame counts differ")
        object.__setattr__(self, "per_frame_centers", centers)
        object.__setattr__(self, "waypoints", waypoints)
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "intrinsics", intrinsics)

    @property
    def frame_count(self) -> int:
        return int(len(self.per_frame_centers))

    @property
    def segment_count(self) -> int:
        return int(len(self.waypoints) - 1)
