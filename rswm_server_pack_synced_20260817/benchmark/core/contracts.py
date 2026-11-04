"""Central input/output contract checks for Phase 2 M1-M3."""

from __future__ import annotations

from typing import Any

import numpy as np

from .errors import ContractError
from .types import ModelInput, Rollout, TrackId, ValidationReport

SUPPORTED_HORIZONS = frozenset({17, 25, 33})


def validate_condition_image(image: Any, *, label: str = "condition_image") -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ContractError(f"{label} must have shape (H,W,3), got {array.shape}")
    if array.dtype != np.uint8:
        raise ContractError(f"{label} must have dtype uint8, got {array.dtype}")
    return array


def validate_model_input(value: ModelInput) -> None:
    image = validate_condition_image(value.condition_image)
    condition = value.trajectory_condition
    metadata = value.metadata
    if condition.track_id != metadata.track_id:
        raise ContractError("trajectory condition track differs from generation metadata")
    if condition.frame_count != metadata.horizon:
        raise ContractError("trajectory condition horizon differs from generation metadata")
    if metadata.horizon not in SUPPORTED_HORIZONS:
        raise ContractError(
            f"unsupported V1 horizon {metadata.horizon}; expected {sorted(SUPPORTED_HORIZONS)}"
        )
    if tuple(image.shape[:2]) != tuple(metadata.output_size):
        raise ContractError(
            "M1-M3 adapters do not resize condition images; image size must equal output_size"
        )
    if metadata.track_id == TrackId.A_TEXT:
        if condition.poses is not None or condition.intrinsics is not None:
            raise ContractError("Track A model input may not contain poses or intrinsics")
    elif metadata.track_id == TrackId.B_CAMERA:
        if condition.poses is None or condition.intrinsics is None:
            raise ContractError("Track B model input requires poses and intrinsics")
        if condition.poses.shape != (metadata.horizon, 4, 4):
            raise ContractError("Track B poses must have shape (T,4,4)")
        if condition.intrinsics.shape != (metadata.horizon, 4):
            raise ContractError("Track B intrinsics must have shape (T,4)")
    else:
        raise ContractError("C_multiview input is reserved and not implemented in M1-M3")


def validate_rollout(
    rollout: Rollout,
    model_input: ModelInput,
) -> ValidationReport:
    validate_model_input(model_input)
    frames = np.asarray(rollout.frames)
    metadata = model_input.metadata
    expected = (metadata.horizon, *metadata.output_size, 3)
    if frames.shape != expected:
        raise ContractError(f"rollout shape {frames.shape} != expected {expected}")
    if frames.dtype != np.uint8:
        raise ContractError(f"rollout dtype must be uint8, got {frames.dtype}")
    if rollout.sample_id != metadata.sample_id:
        raise ContractError("rollout sample_id differs from generation request")
    if int(rollout.generation_seed) != int(metadata.generation_seed):
        raise ContractError("rollout generation_seed differs from generation request")
    first = frames[0].astype(np.int16)
    condition = np.asarray(model_input.condition_image).astype(np.int16)
    mae = float(np.abs(first - condition).mean())
    return ValidationReport(
        valid=True,
        sample_id=metadata.sample_id,
        task_id=metadata.task_id,
        track_id=metadata.track_id,
        checks={
            "shape": list(frames.shape),
            "dtype": str(frames.dtype),
            "condition_frame_mae": mae,
            "condition_frame_exact": bool(mae == 0.0),
            "future_only_start_index": 1,
        },
        warnings=(() if mae == 0.0 else ("condition frame is not pixel-exact",)),
    )
