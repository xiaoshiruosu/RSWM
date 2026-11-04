"""Typed benchmark samples, model inputs and rollout outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from benchmark.trajectory.types import DirectionTarget, OverlapProfile, TrajectoryPlan
from .errors import ContractError, IdentityError
from .identity import ExecutionIdentity, ModelIdentity

_SHA256_HEX = frozenset("0123456789abcdef")


def _validate_sha256(value: str, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise ContractError(f"{label} must be a 64-character SHA-256 digest")
    return text


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class TaskId(_StringEnum):
    T1_SPATIAL_ROLLOUT = "T1_spatial_rollout"
    T2_ARBITRARY_DIRECTION = "T2_arbitrary_direction"
    T3_OVERLAP_LADDER = "T3_overlap_ladder"


class TrackId(_StringEnum):
    A_TEXT = "A_text"
    B_CAMERA = "B_camera"
    C_MULTIVIEW = "C_multiview"


class DataDomain(_StringEnum):
    SYNTHETIC_CONTROLLED = "synthetic_controlled"
    GENERALIZATION = "generalization"
    REAL_WORLD = "real_world"


class CapabilityPartition(_StringEnum):
    NATIVE = "native"
    ADAPTED = "adapted"
    MULTIVIEW = "multiview"


class PromptSchema(_StringEnum):
    NET_CONTINUOUS = "net_continuous"
    LEGACY_NET = "net"
    PER_SEGMENT_CONTINUOUS = "per_segment_continuous"
    STRIPPED = "stripped"


def freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    return value


def readonly_array(value: Any) -> np.ndarray:
    array = np.array(value, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class BenchmarkSample:
    id: str
    image: np.ndarray
    trajectory: TrajectoryPlan
    overlap: OverlapProfile
    direction: DirectionTarget
    horizon: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.id):
            raise ContractError("sample.id must be non-empty")
        image = readonly_array(self.image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ContractError(
                f"sample.image must be (H,W,3) uint8, got {image.shape} {image.dtype}"
            )
        if int(self.horizon) != self.trajectory.frame_count:
            raise ContractError("sample horizon differs from trajectory frame count")
        if self.overlap is not self.trajectory.overlap and self.overlap != self.trajectory.overlap:
            raise ContractError("sample overlap differs from trajectory overlap")
        if self.direction is not self.trajectory.direction and self.direction != self.trajectory.direction:
            raise ContractError("sample direction differs from trajectory direction")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


@dataclass(frozen=True)
class GroundTruthBundle:
    rollout: np.ndarray
    rollout_checksum: str
    full_scene: Optional[np.ndarray] = None
    source_reference: Optional[str] = None

    def __post_init__(self) -> None:
        rollout = readonly_array(self.rollout)
        if rollout.ndim != 4 or rollout.shape[-1] != 3 or rollout.dtype != np.uint8:
            raise ContractError(
                f"ground truth rollout must be (T,H,W,3) uint8, got {rollout.shape} {rollout.dtype}"
            )
        full_scene = None if self.full_scene is None else readonly_array(self.full_scene)
        object.__setattr__(self, "rollout", rollout)
        object.__setattr__(self, "full_scene", full_scene)
        object.__setattr__(
            self,
            "rollout_checksum",
            _validate_sha256(self.rollout_checksum, "ground_truth.rollout_checksum"),
        )


@dataclass(frozen=True)
class OrganizerSample:
    sample: BenchmarkSample
    ground_truth: GroundTruthBundle


@dataclass(frozen=True)
class PublicSample:
    id: str
    image: np.ndarray
    horizon: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        image = readonly_array(self.image)
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ContractError("public sample image must be (H,W,3) uint8")
        if int(self.horizon) <= 1:
            raise ContractError("public sample horizon must exceed one")
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))


@dataclass(frozen=True)
class TrajectoryCondition:
    track_id: TrackId
    scene_prompt: str
    prompt_schema: PromptSchema
    segment_count: int
    net_camera_bearing_deg: float
    frame_count: int
    poses: Optional[np.ndarray] = None
    intrinsics: Optional[np.ndarray] = None
    convention_version: str = "rswm.direction/cy-cx-camera-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", TrackId(self.track_id))
        object.__setattr__(self, "prompt_schema", PromptSchema(self.prompt_schema))
        if not self.scene_prompt:
            raise ContractError("trajectory condition requires a non-empty scene prompt")
        if int(self.segment_count) <= 0 or int(self.frame_count) <= 1:
            raise ContractError("segment_count must be positive and frame_count must exceed one")
        bearing = float(self.net_camera_bearing_deg)
        if not np.isfinite(bearing) or not 0.0 <= bearing < 360.0:
            raise ContractError("net_camera_bearing_deg must be finite and in [0,360)")
        object.__setattr__(self, "segment_count", int(self.segment_count))
        object.__setattr__(self, "frame_count", int(self.frame_count))
        object.__setattr__(self, "net_camera_bearing_deg", bearing)
        poses = None if self.poses is None else readonly_array(self.poses)
        intrinsics = None if self.intrinsics is None else readonly_array(self.intrinsics)
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "intrinsics", intrinsics)


@dataclass(frozen=True)
class GenerationMetadata:
    benchmark_version: str
    task_id: TaskId
    track_id: TrackId
    sample_id: str
    horizon: int
    output_size: Tuple[int, int]
    generation_seed: int
    condition_checksum: str
    public: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", TaskId(self.task_id))
        object.__setattr__(self, "track_id", TrackId(self.track_id))
        size = tuple(int(value) for value in self.output_size)
        if len(size) != 2 or any(value <= 0 for value in size):
            raise ContractError("output_size must be (height,width) with positive values")
        object.__setattr__(self, "output_size", size)
        if not str(self.benchmark_version).strip() or not str(self.sample_id).strip():
            raise ContractError("benchmark_version and sample_id must be non-empty")
        if int(self.horizon) <= 1:
            raise ContractError("generation horizon must exceed one")
        object.__setattr__(self, "horizon", int(self.horizon))
        object.__setattr__(self, "generation_seed", int(self.generation_seed))
        object.__setattr__(
            self,
            "condition_checksum",
            _validate_sha256(self.condition_checksum, "generation.condition_checksum"),
        )
        object.__setattr__(self, "public", freeze_json(self.public))


@dataclass(frozen=True)
class ModelInput:
    condition_image: np.ndarray
    trajectory_condition: TrajectoryCondition
    metadata: GenerationMetadata

    def __post_init__(self) -> None:
        object.__setattr__(self, "condition_image", readonly_array(self.condition_image))


@dataclass(frozen=True)
class Rollout:
    frames: np.ndarray
    sample_id: str
    model_id: str
    adapter_version: str
    generation_seed: int
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()
    model_identity: Optional[ModelIdentity] = None
    adapter_identity: Optional[str] = None
    execution_identity: Optional[ExecutionIdentity] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frames", readonly_array(self.frames))
        object.__setattr__(self, "runtime_metadata", freeze_json(self.runtime_metadata))
        object.__setattr__(self, "warnings", tuple(str(value) for value in self.warnings))
        identity_fields = (
            self.model_identity,
            self.adapter_identity,
            self.execution_identity,
        )
        if any(value is not None for value in identity_fields):
            if not all(value is not None for value in identity_fields):
                raise ContractError(
                    "rollout identity must provide model_identity, adapter_identity "
                    "and execution_identity together"
                )
            if not isinstance(self.model_identity, ModelIdentity):
                raise ContractError("rollout.model_identity must be ModelIdentity")
            if not isinstance(self.execution_identity, ExecutionIdentity):
                raise ContractError(
                    "rollout.execution_identity must be ExecutionIdentity"
                )
            adapter_identity = str(self.adapter_identity).strip()
            if not adapter_identity:
                raise ContractError("rollout.adapter_identity must be non-empty")
            object.__setattr__(self, "adapter_identity", adapter_identity)
            try:
                self.model_identity.validate()
            except IdentityError as exc:
                raise ContractError(f"rollout.model_identity is invalid: {exc}") from exc
            if str(self.model_id).strip() != self.model_identity.release_identity:
                raise ContractError("rollout.model_id differs from model_identity release")
            if adapter_identity != self.model_identity.adapter_identity:
                raise ContractError(
                    "rollout.adapter_identity differs from model_identity"
                )
            if (
                self.execution_identity.to_document()
                != self.model_identity.execution_identity.to_document()
            ):
                raise ContractError(
                    "rollout.execution_identity differs from model_identity"
                )


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    sample_id: str
    task_id: TaskId
    track_id: TrackId
    checks: Mapping[str, Any]
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", TaskId(self.task_id))
        object.__setattr__(self, "track_id", TrackId(self.track_id))
        object.__setattr__(self, "checks", freeze_json(self.checks))
        object.__setattr__(self, "warnings", tuple(self.warnings))
