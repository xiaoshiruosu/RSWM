"""Explicit full-rollout and future-only views with one slicing owner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

from benchmark.core.types import readonly_array

from .errors import EvaluationError, FutureOnlyOwnershipError
from .identity import FutureOnlyOwner, MetricIdentity


@dataclass(frozen=True)
class FullRolloutView:
    frames: np.ndarray

    def __post_init__(self) -> None:
        frames = readonly_array(self.frames)
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise EvaluationError(
                "FullRolloutView requires (T,H,W,3) uint8 frames"
            )
        if frames.shape[1] <= 0 or frames.shape[2] <= 0:
            raise EvaluationError("rollout height and width must be positive")
        if len(frames) < 2:
            raise EvaluationError("a rollout must contain t=0 and at least one future frame")
        object.__setattr__(self, "frames", frames)

    def future_only(self, owner: FutureOnlyOwner) -> "FutureOnlyView":
        owner = FutureOnlyOwner(owner)
        if owner is not FutureOnlyOwner.EVALUATION_RUNNER:
            raise FutureOnlyOwnershipError(
                "only evaluation_runner may materialize FutureOnlyView"
            )
        return FutureOnlyView(
            frames=self.frames[1:],
            source_frame_count=len(self.frames),
            future_only_owner=owner,
        )


@dataclass(frozen=True)
class FutureOnlyView:
    frames: np.ndarray
    source_frame_count: int
    future_only_owner: FutureOnlyOwner
    source_start_index: int = 1

    def __post_init__(self) -> None:
        frames = readonly_array(self.frames)
        owner = FutureOnlyOwner(self.future_only_owner)
        if owner is not FutureOnlyOwner.EVALUATION_RUNNER:
            raise FutureOnlyOwnershipError(
                "FutureOnlyView must be owned by evaluation_runner"
            )
        if self.source_start_index != 1:
            raise FutureOnlyOwnershipError("future-only view must begin at source t=1")
        if int(self.source_frame_count) != len(frames) + 1:
            raise FutureOnlyOwnershipError(
                "future-only view must remove exactly one condition frame"
            )
        if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
            raise EvaluationError(
                "FutureOnlyView requires (T-1,H,W,3) uint8 frames"
            )
        if frames.shape[1] <= 0 or frames.shape[2] <= 0:
            raise EvaluationError("rollout height and width must be positive")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "source_frame_count", int(self.source_frame_count))
        object.__setattr__(self, "future_only_owner", owner)


RolloutView = Union[FullRolloutView, FutureOnlyView]


@dataclass(frozen=True)
class EvaluationView:
    """Paired rollout views and the sole future-only owner for one evaluation."""

    prediction: FullRolloutView
    ground_truth: FullRolloutView
    future_only_owner: FutureOnlyOwner = FutureOnlyOwner.EVALUATION_RUNNER

    def __post_init__(self) -> None:
        if not isinstance(self.prediction, FullRolloutView) or not isinstance(
            self.ground_truth, FullRolloutView
        ):
            raise EvaluationError("EvaluationView requires paired FullRolloutView values")
        if self.prediction.frames.shape != self.ground_truth.frames.shape:
            raise EvaluationError(
                "prediction and ground truth rollouts must have identical shapes"
            )
        owner = FutureOnlyOwner(self.future_only_owner)
        if owner is FutureOnlyOwner.NOT_APPLICABLE:
            raise FutureOnlyOwnershipError(
                "an evaluation must choose evaluation_runner or legacy_wrapper as its "
                "single future-only owner"
            )
        object.__setattr__(self, "future_only_owner", owner)

    @classmethod
    def create(
        cls,
        prediction: np.ndarray,
        ground_truth: np.ndarray,
        *,
        future_only_owner: FutureOnlyOwner = FutureOnlyOwner.EVALUATION_RUNNER,
    ) -> "EvaluationView":
        return cls(
            FullRolloutView(prediction),
            FullRolloutView(ground_truth),
            future_only_owner,
        )

    def frames_for(self, identity: MetricIdentity) -> Tuple[RolloutView, RolloutView]:
        identity.validate()
        owner = identity.future_only_owner
        if owner is FutureOnlyOwner.NOT_APPLICABLE:
            return self.prediction, self.ground_truth
        if owner is not self.future_only_owner:
            raise FutureOnlyOwnershipError(
                f"evaluation owner={self.future_only_owner.value!r}, but metric "
                f"{identity.metric_id!r} declares owner={owner.value!r}"
            )
        if owner is FutureOnlyOwner.EVALUATION_RUNNER:
            return (
                self.prediction.future_only(owner),
                self.ground_truth.future_only(owner),
            )
        # The legacy wrapper receives full rollouts and is solely responsible for
        # stripping t=0 exactly once. A fine-grained wrapper must never use this path.
        return self.prediction, self.ground_truth
