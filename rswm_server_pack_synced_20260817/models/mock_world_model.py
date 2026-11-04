"""Deterministic CPU-only mock world model for contract tests.

This is a test double, not a scientific baseline and never an admitted result.
"""

from __future__ import annotations

import math

import numpy as np

from benchmark.core.contracts import validate_condition_image
from benchmark.core.errors import ContractError
from benchmark.core.types import (
    CapabilityPartition,
    GenerationMetadata,
    PromptSchema,
    Rollout,
    TaskId,
    TrackId,
    TrajectoryCondition,
)
from benchmark.trajectory.conventions import content_bearing_deg
from .world_model import WorldModelCapabilities


def _translate_without_wrap(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    result = np.zeros_like(image)
    height, width = image.shape[:2]
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    if src_y1 > src_y0 and src_x1 > src_x0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return result


class DeterministicMockWorldModel:
    """Shift the condition image according to the commanded net direction."""

    name = "rswm-contract-mock"
    capabilities = WorldModelCapabilities(
        input_type="single_image",
        supported_tasks=frozenset(TaskId),
        supported_tracks=frozenset({TrackId.A_TEXT, TrackId.B_CAMERA}),
        prompt_schemas=frozenset({PromptSchema.NET_CONTINUOUS}),
        trajectory_support="net_direction_and_full_camera_fixture",
        camera_support="poses_intrinsics_fixture",
        supported_horizons=frozenset({17, 25, 33}),
        adaptation_type="none",
        capability_partition=CapabilityPartition.NATIVE,
        condition_view_count=1,
        stochastic=False,
        checkpoint_required=False,
        notes="CPU contract fixture only; not eligible as a benchmark baseline",
    )

    def __init__(self, *, pixels_per_frame: int = 1):
        if int(pixels_per_frame) < 0:
            raise ValueError("pixels_per_frame must be non-negative")
        self.pixels_per_frame = int(pixels_per_frame)

    def generate(
        self,
        condition_image: np.ndarray,
        trajectory_condition: TrajectoryCondition,
        metadata: GenerationMetadata,
    ) -> Rollout:
        image = validate_condition_image(condition_image)
        self.capabilities.assert_eligible(
            task_id=metadata.task_id,
            track_id=metadata.track_id,
            prompt_schema=trajectory_condition.prompt_schema,
            horizon=metadata.horizon,
        )
        if trajectory_condition.track_id != metadata.track_id:
            raise ContractError("mock model received conflicting track identities")
        if tuple(image.shape[:2]) != tuple(metadata.output_size):
            raise ContractError("mock model does not resize condition images")
        content_angle = math.radians(
            content_bearing_deg(trajectory_condition.net_camera_bearing_deg)
        )
        frames = np.empty((metadata.horizon, *metadata.output_size, 3), dtype=np.uint8)
        for index in range(metadata.horizon):
            distance = self.pixels_per_frame * index
            dx = int(round(math.cos(content_angle) * distance))
            dy = int(round(-math.sin(content_angle) * distance))
            frames[index] = _translate_without_wrap(image, dy, dx)
        return Rollout(
            frames=frames,
            sample_id=metadata.sample_id,
            model_id=self.name,
            adapter_version="mock/v1",
            generation_seed=metadata.generation_seed,
            runtime_metadata={
                "mock_only": True,
                "scientific_result_eligible": False,
                "pixels_per_frame": self.pixels_per_frame,
            },
        )
