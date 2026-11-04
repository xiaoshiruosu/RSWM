"""Shared lifecycle for legacy distribution-metric accumulators."""

from __future__ import annotations

from abc import abstractmethod
import math
from typing import Any, Mapping, Optional

from .base import DatasetMetric
from .errors import AccumulatorStateError
from .identity import MetricIdentity
from .result import MetricResult
from .views import EvaluationView, RolloutView


class LegacyDistributionAccumulator(DatasetMetric):
    def __init__(self, identity: MetricIdentity, *, minimum_n: int) -> None:
        if identity.scope.value != "dataset":
            raise ValueError("distribution accumulator identity must be dataset-scoped")
        minimum_n = int(minimum_n)
        if minimum_n < 2:
            raise ValueError("distribution metrics require minimum_n >= 2")
        self.identity = identity
        self.minimum_n = minimum_n
        self._state = "new"
        self._N = 0
        self._predictions: list[Any] = []
        self._ground_truths: list[Any] = []

    @property
    def sample_count(self) -> int:
        return self._N

    def start(self, context: Optional[Mapping[str, Any]] = None) -> None:
        if self._state == "started":
            raise AccumulatorStateError("dataset metric is already started")
        self._state = "started"
        self._N = 0
        self._predictions = []
        self._ground_truths = []

    def update(
        self,
        view: EvaluationView,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self._state != "started":
            raise AccumulatorStateError("call start() before update()")
        prediction, ground_truth = view.frames_for(self.identity)
        self._ingest(prediction, ground_truth)
        self._N += 1

    def finalize(self) -> MetricResult:
        if self._state != "started":
            raise AccumulatorStateError("call start() before finalize()")
        self._state = "finalized"
        if self._N < self.minimum_n:
            return MetricResult.invalid(
                f"insufficient_n:{self._N}<{self.minimum_n}",
                N=self._N,
                metric_identity=self.identity,
            )
        try:
            value = float(self._compute_value())
        except Exception as exc:
            return MetricResult.invalid(
                f"legacy_metric_error:{type(exc).__name__}:{exc}",
                N=self._N,
                metric_identity=self.identity,
            )
        if not math.isfinite(value):
            return MetricResult.invalid(
                "non_finite_metric_value",
                N=self._N,
                metric_identity=self.identity,
            )
        return MetricResult.success(
            value,
            N=self._N,
            metric_identity=self.identity,
        )

    @abstractmethod
    def _ingest(self, prediction: RolloutView, ground_truth: RolloutView) -> None:
        raise NotImplementedError

    @abstractmethod
    def _compute_value(self) -> float:
        raise NotImplementedError
