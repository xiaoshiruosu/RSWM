"""General world-model protocol above the existing VideoBackbone interface."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import FrozenSet, Optional, Protocol, Tuple

import numpy as np

from benchmark.core.errors import CapabilityError, ContractError, IdentityError
from benchmark.core.identity import (
    ArchitectureIdentity,
    ExecutionIdentity,
    ModelIdentity,
)
from benchmark.core.types import (
    CapabilityPartition,
    GenerationMetadata,
    PromptSchema,
    Rollout,
    TaskId,
    TrackId,
    TrajectoryCondition,
)


@dataclass(frozen=True)
class WorldModelCapabilities:
    input_type: str
    supported_tasks: FrozenSet[TaskId]
    supported_tracks: FrozenSet[TrackId]
    prompt_schemas: FrozenSet[PromptSchema]
    trajectory_support: str
    camera_support: str
    supported_horizons: FrozenSet[int]
    adaptation_type: str
    capability_partition: CapabilityPartition
    condition_view_count: int = 1
    stochastic: bool = True
    native_output_format: str = "THWC_uint8"
    checkpoint_required: bool = True
    notes: str = ""
    model_identity: Optional[ModelIdentity] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "supported_tasks", frozenset(TaskId(value) for value in self.supported_tasks)
        )
        object.__setattr__(
            self, "supported_tracks", frozenset(TrackId(value) for value in self.supported_tracks)
        )
        object.__setattr__(
            self,
            "prompt_schemas",
            frozenset(PromptSchema(value) for value in self.prompt_schemas),
        )
        object.__setattr__(
            self,
            "capability_partition",
            CapabilityPartition(self.capability_partition),
        )
        horizons = frozenset(int(value) for value in self.supported_horizons)
        if not horizons or any(value <= 1 for value in horizons):
            raise ContractError("supported_horizons must contain positive rollout lengths")
        object.__setattr__(self, "supported_horizons", horizons)
        if int(self.condition_view_count) <= 0:
            raise ContractError("condition_view_count must be positive")
        if TrackId.C_MULTIVIEW in self.supported_tracks:
            if self.capability_partition != CapabilityPartition.MULTIVIEW:
                raise ContractError("C_multiview support requires multiview partition")
        if self.condition_view_count > 1 and self.capability_partition != CapabilityPartition.MULTIVIEW:
            raise ContractError("multiple condition views require multiview partition")
        if self.model_identity is not None and not isinstance(
            self.model_identity, ModelIdentity
        ):
            raise ContractError("model_identity must be ModelIdentity or None")
        if self.model_identity is not None:
            try:
                self.model_identity.validate()
            except IdentityError as exc:
                raise ContractError(f"model_identity is invalid: {exc}") from exc

    @property
    def release_identity(self) -> Optional[str]:
        """Stable release/checkpoint product name; absent for legacy declarations."""

        return (
            self.model_identity.release_identity
            if self.model_identity is not None
            else None
        )

    @property
    def architecture_identity(self) -> Optional[ArchitectureIdentity]:
        """Architecture implementation identity, separate from release branding."""

        return (
            self.model_identity.architecture_identity
            if self.model_identity is not None
            else None
        )

    @property
    def adapter_identity(self) -> Optional[str]:
        """Benchmark adapter identity; not the execution route."""

        return (
            self.model_identity.adapter_identity
            if self.model_identity is not None
            else None
        )

    @property
    def execution_identity(self) -> Optional[ExecutionIdentity]:
        """Runtime route such as wan-inline or a future benchmark adapter path."""

        return (
            self.model_identity.execution_identity
            if self.model_identity is not None
            else None
        )

    def assert_eligible(
        self,
        *,
        task_id: TaskId,
        track_id: TrackId,
        prompt_schema: PromptSchema,
        horizon: int,
    ) -> None:
        failures = []
        if TaskId(task_id) not in self.supported_tasks:
            failures.append(f"task={TaskId(task_id).value}")
        if TrackId(track_id) not in self.supported_tracks:
            failures.append(f"track={TrackId(track_id).value}")
        if PromptSchema(prompt_schema) not in self.prompt_schemas:
            failures.append(f"prompt_schema={PromptSchema(prompt_schema).value}")
        if int(horizon) not in self.supported_horizons:
            failures.append(f"horizon={horizon}")
        if failures:
            raise CapabilityError(
                "world model is not eligible for requested benchmark input: "
                + ", ".join(failures)
            )


class WorldModel(Protocol):
    name: str
    capabilities: WorldModelCapabilities

    def generate(
        self,
        condition_image: np.ndarray,
        trajectory_condition: TrajectoryCondition,
        metadata: GenerationMetadata,
    ) -> Rollout:
        ...


def validate_world_model(value, *, label: str = "world_model") -> None:
    missing = [name for name in ("name", "capabilities", "generate") if not hasattr(value, name)]
    if missing:
        raise TypeError(f"{label} is missing required attributes: {missing}")
    if not isinstance(value.name, str) or not value.name:
        raise TypeError(f"{label}.name must be a non-empty string")
    if not isinstance(value.capabilities, WorldModelCapabilities):
        raise TypeError(f"{label}.capabilities must be WorldModelCapabilities")
    signature = inspect.signature(value.generate)
    required = ("condition_image", "trajectory_condition", "metadata")
    absent = [name for name in required if name not in signature.parameters]
    if absent:
        raise TypeError(f"{label}.generate is missing parameters {absent}")
