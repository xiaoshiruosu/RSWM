"""Fair-comparison capability contract and partition admission rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Tuple

from benchmark.core.identity import CheckpointIdentity
from benchmark.core.types import CapabilityPartition, TaskId, TrackId

from .errors import BaselineCapabilityError

UNKNOWN = "UNKNOWN"
NOT_APPLICABLE = "NOT_APPLICABLE"


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class InputType(_StringEnum):
    SINGLE_IMAGE = "single_image"
    MULTIVIEW = "multiview"
    VIDEO_PREFIX = "video_prefix"
    UNKNOWN = UNKNOWN


class TrajectorySupport(_StringEnum):
    NONE = "none"
    NET_DIRECTION = "net_direction"
    SEGMENT_DIRECTION = "segment_direction"
    FULL_PATH = "full_path"
    UNKNOWN = UNKNOWN


class CameraSupport(_StringEnum):
    NONE = "none"
    EXTRINSICS = "extrinsics"
    INTRINSICS = "intrinsics"
    PLUCKER = "plucker"
    UNKNOWN = UNKNOWN


class AdaptationType(_StringEnum):
    FROZEN = "frozen"
    PROMPT_BRIDGE = "prompt_bridge"
    RECURSIVE = "recursive"
    LORA = "lora"
    FULL_FINETUNE = "full_finetune"
    UNKNOWN = UNKNOWN


class DisclosureStatus(_StringEnum):
    UNKNOWN = UNKNOWN
    DECLARED = "DECLARED"
    VERIFIED = "VERIFIED"
    NOT_APPLICABLE = NOT_APPLICABLE


@dataclass(frozen=True)
class LicenseDisclosure:
    code: str
    weights: str
    data: str

    def __post_init__(self) -> None:
        for name in ("code", "weights", "data"):
            value = _nonempty(getattr(self, name), f"license.{name}")
            object.__setattr__(self, name, value)

    @property
    def is_complete(self) -> bool:
        return all(value != UNKNOWN for value in (self.code, self.weights, self.data))

    def to_document(self) -> dict[str, str]:
        return {"code": self.code, "weights": self.weights, "data": self.data}

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "LicenseDisclosure":
        return cls(
            code=value.get("code", UNKNOWN),
            weights=value.get("weights", UNKNOWN),
            data=value.get("data", UNKNOWN),
        )


@dataclass(frozen=True)
class ExternalDataDisclosure:
    status: DisclosureStatus
    description: str
    references: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", DisclosureStatus(self.status))
        object.__setattr__(
            self, "description", _nonempty(self.description, "external_data.description")
        )
        references = tuple(
            _nonempty(item, "external_data.references item") for item in self.references
        )
        if len(set(references)) != len(references):
            raise BaselineCapabilityError(
                "external_data.references may not contain duplicates"
            )
        object.__setattr__(self, "references", references)

    @property
    def is_disclosed(self) -> bool:
        return self.status is not DisclosureStatus.UNKNOWN

    def to_document(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "description": self.description,
            "references": list(self.references),
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ExternalDataDisclosure":
        return cls(
            status=value.get("status", UNKNOWN),
            description=value.get("description", UNKNOWN),
            references=tuple(value.get("references") or ()),
        )


@dataclass(frozen=True)
class RuntimeDisclosure:
    framework: str
    framework_version: str
    container_identity: str
    hardware: str
    precision: str

    def __post_init__(self) -> None:
        for name in (
            "framework",
            "framework_version",
            "container_identity",
            "hardware",
            "precision",
        ):
            object.__setattr__(
                self, name, _nonempty(getattr(self, name), f"runtime.{name}")
            )

    @property
    def is_resolved(self) -> bool:
        return all(
            value != UNKNOWN
            for value in (
                self.framework,
                self.framework_version,
                self.container_identity,
                self.hardware,
                self.precision,
            )
        )

    def to_document(self) -> dict[str, str]:
        return {
            "framework": self.framework,
            "framework_version": self.framework_version,
            "container_identity": self.container_identity,
            "hardware": self.hardware,
            "precision": self.precision,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "RuntimeDisclosure":
        return cls(
            framework=value.get("framework", UNKNOWN),
            framework_version=value.get("framework_version", UNKNOWN),
            container_identity=value.get("container_identity", UNKNOWN),
            hardware=value.get("hardware", UNKNOWN),
            precision=value.get("precision", UNKNOWN),
        )


@dataclass(frozen=True)
class BaselineCapabilities:
    input_type: InputType
    trajectory_support: TrajectorySupport
    camera_support: CameraSupport
    adaptation_type: AdaptationType
    checkpoint_identity: CheckpointIdentity
    license: LicenseDisclosure
    tasks: Tuple[TaskId, ...]
    tracks: Tuple[TrackId, ...]
    external_data: ExternalDataDisclosure
    runtime: RuntimeDisclosure
    condition_views: int
    uses_future_gt: bool
    output_format: str

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "input_type", InputType(self.input_type))
            object.__setattr__(
                self, "trajectory_support", TrajectorySupport(self.trajectory_support)
            )
            object.__setattr__(
                self, "camera_support", CameraSupport(self.camera_support)
            )
            object.__setattr__(
                self, "adaptation_type", AdaptationType(self.adaptation_type)
            )
        except ValueError as exc:
            raise BaselineCapabilityError(str(exc)) from exc
        if not isinstance(self.checkpoint_identity, CheckpointIdentity):
            raise BaselineCapabilityError(
                "capabilities.checkpoint_identity must be CheckpointIdentity"
            )
        if not isinstance(self.license, LicenseDisclosure):
            raise BaselineCapabilityError("capabilities.license must be LicenseDisclosure")
        if not isinstance(self.external_data, ExternalDataDisclosure):
            raise BaselineCapabilityError(
                "capabilities.external_data must be ExternalDataDisclosure"
            )
        if not isinstance(self.runtime, RuntimeDisclosure):
            raise BaselineCapabilityError("capabilities.runtime must be RuntimeDisclosure")
        tasks = _enum_tuple(self.tasks, TaskId, "tasks")
        tracks = _enum_tuple(self.tracks, TrackId, "tracks")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "tracks", tracks)
        try:
            views = int(self.condition_views)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BaselineCapabilityError(
                "condition_views must be a positive integer"
            ) from exc
        if isinstance(self.condition_views, bool) or views <= 0:
            raise BaselineCapabilityError("condition_views must be a positive integer")
        object.__setattr__(self, "condition_views", views)
        if not isinstance(self.uses_future_gt, bool):
            raise BaselineCapabilityError("uses_future_gt must be bool")
        object.__setattr__(
            self, "output_format", _nonempty(self.output_format, "output_format")
        )

        if self.input_type is InputType.MULTIVIEW and views <= 1:
            raise BaselineCapabilityError(
                "multiview input requires condition_views > 1"
            )
        if views > 1 and self.input_type not in {
            InputType.MULTIVIEW,
            InputType.UNKNOWN,
        }:
            raise BaselineCapabilityError(
                "multiple condition views require multiview or UNKNOWN input_type"
            )
        if TrackId.C_MULTIVIEW in tracks and self.input_type is not InputType.MULTIVIEW:
            raise BaselineCapabilityError(
                "C_multiview support requires multiview input_type"
            )
        if self.input_type is InputType.MULTIVIEW and any(
            track is not TrackId.C_MULTIVIEW for track in tracks
        ):
            raise BaselineCapabilityError(
                "a multiview entry may only declare C_multiview"
            )
        if TrackId.B_CAMERA in tracks:
            if self.camera_support not in {
                CameraSupport.EXTRINSICS,
                CameraSupport.PLUCKER,
            }:
                raise BaselineCapabilityError(
                    "B_camera support requires extrinsics or Plucker capability"
                )
            if self.trajectory_support is not TrajectorySupport.FULL_PATH:
                raise BaselineCapabilityError(
                    "B_camera support requires full_path trajectory support"
                )

    def assert_partition(self, partition: CapabilityPartition) -> None:
        partition = CapabilityPartition(partition)
        failures = []
        if partition is CapabilityPartition.NATIVE:
            if self.input_type in {InputType.MULTIVIEW, InputType.UNKNOWN}:
                failures.append(f"input_type={self.input_type.value}")
            if self.adaptation_type is not AdaptationType.FROZEN:
                failures.append(f"adaptation_type={self.adaptation_type.value}")
            if TrackId.C_MULTIVIEW in self.tracks:
                failures.append("track=C_multiview")
        elif partition is CapabilityPartition.ADAPTED:
            if self.input_type in {InputType.MULTIVIEW, InputType.UNKNOWN}:
                failures.append(f"input_type={self.input_type.value}")
            if self.adaptation_type in {
                AdaptationType.FROZEN,
                AdaptationType.UNKNOWN,
            }:
                failures.append(f"adaptation_type={self.adaptation_type.value}")
            if TrackId.C_MULTIVIEW in self.tracks:
                failures.append("track=C_multiview")
        else:
            if self.input_type is not InputType.MULTIVIEW:
                failures.append(f"input_type={self.input_type.value}")
            if self.condition_views <= 1:
                failures.append(f"condition_views={self.condition_views}")
            if any(track is not TrackId.C_MULTIVIEW for track in self.tracks):
                failures.append("non-multiview track declared")
        if failures:
            raise BaselineCapabilityError(
                f"capabilities do not satisfy {partition.value} partition: "
                + ", ".join(failures)
            )

    def assert_eligible(
        self,
        *,
        task_id: TaskId,
        track_id: TrackId,
        partition: CapabilityPartition,
    ) -> None:
        """Fail closed; this method never falls back to another partition."""

        task_id = TaskId(task_id)
        track_id = TrackId(track_id)
        partition = CapabilityPartition(partition)
        self.assert_partition(partition)
        failures = []
        if task_id not in self.tasks:
            failures.append(f"task={task_id.value}")
        if track_id not in self.tracks:
            failures.append(f"track={track_id.value}")
        if partition is CapabilityPartition.MULTIVIEW and track_id is not TrackId.C_MULTIVIEW:
            failures.append("multiview partition requires C_multiview")
        if partition is not CapabilityPartition.MULTIVIEW and track_id is TrackId.C_MULTIVIEW:
            failures.append("C_multiview requires multiview partition")
        if track_id is TrackId.B_CAMERA:
            if self.camera_support not in {
                CameraSupport.EXTRINSICS,
                CameraSupport.PLUCKER,
            }:
                failures.append(f"camera_support={self.camera_support.value}")
            if self.trajectory_support is not TrajectorySupport.FULL_PATH:
                failures.append(
                    f"trajectory_support={self.trajectory_support.value}"
                )
        if task_id is TaskId.T2_ARBITRARY_DIRECTION and self.trajectory_support in {
            TrajectorySupport.NONE,
            TrajectorySupport.UNKNOWN,
        }:
            failures.append(
                f"trajectory_support={self.trajectory_support.value}"
            )
        if failures:
            raise BaselineCapabilityError(
                "baseline is not eligible for requested benchmark partition: "
                + ", ".join(failures)
            )

    @property
    def is_admission_disclosed(self) -> bool:
        return (
            self.license.is_complete
            and self.external_data.is_disclosed
            and self.runtime.is_resolved
            and bool(self.tasks)
            and bool(self.tracks)
            and self.input_type is not InputType.UNKNOWN
            and self.trajectory_support is not TrajectorySupport.UNKNOWN
            and self.camera_support is not CameraSupport.UNKNOWN
            and self.adaptation_type is not AdaptationType.UNKNOWN
            and self.output_format != UNKNOWN
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "input_type": self.input_type.value,
            "trajectory_support": self.trajectory_support.value,
            "camera_support": self.camera_support.value,
            "adaptation_type": self.adaptation_type.value,
            "checkpoint_identity": self.checkpoint_identity.to_document(),
            "license": self.license.to_document(),
            "tasks": [value.value for value in self.tasks],
            "tracks": [value.value for value in self.tracks],
            "external_data": self.external_data.to_document(),
            "runtime": self.runtime.to_document(),
            "condition_views": self.condition_views,
            "uses_future_gt": self.uses_future_gt,
            "output_format": self.output_format,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "BaselineCapabilities":
        return cls(
            input_type=value.get("input_type", UNKNOWN),
            trajectory_support=value.get("trajectory_support", UNKNOWN),
            camera_support=value.get("camera_support", UNKNOWN),
            adaptation_type=value.get("adaptation_type", UNKNOWN),
            checkpoint_identity=CheckpointIdentity.from_document(
                _mapping(value.get("checkpoint_identity"), "checkpoint_identity")
            ),
            license=LicenseDisclosure.from_document(
                _mapping(value.get("license"), "license")
            ),
            tasks=tuple(value.get("tasks") or ()),
            tracks=tuple(value.get("tracks") or ()),
            external_data=ExternalDataDisclosure.from_document(
                _mapping(value.get("external_data"), "external_data")
            ),
            runtime=RuntimeDisclosure.from_document(
                _mapping(value.get("runtime"), "runtime")
            ),
            condition_views=value.get("condition_views", 1),
            uses_future_gt=value.get("uses_future_gt", False),
            output_format=value.get("output_format", UNKNOWN),
        )


def _nonempty(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise BaselineCapabilityError(f"{label} must be a non-empty string")
    return text


def _enum_tuple(
    values: Iterable[Any], enum_type: type[Enum], label: str
) -> tuple[Any, ...]:
    try:
        items = tuple(enum_type(value) for value in values)
    except ValueError as exc:
        raise BaselineCapabilityError(f"invalid {label}: {exc}") from exc
    if len(set(items)) != len(items):
        raise BaselineCapabilityError(f"{label} may not contain duplicates")
    return items


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineCapabilityError(f"{label} must be an object")
    return value
