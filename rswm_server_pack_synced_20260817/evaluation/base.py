"""Unified metric interfaces for sample and dataset scopes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

from .identity import MetricIdentity
from .result import MetricResult
from .views import EvaluationView


class Metric(ABC):
    """Common identity-bearing metric interface."""

    identity: MetricIdentity


class SampleMetric(Metric):
    @abstractmethod
    def compute(
        self,
        view: EvaluationView,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MetricResult:
        raise NotImplementedError


class DatasetMetric(Metric):
    """Dataset accumulator lifecycle; per-sample compute is forbidden."""

    def compute(self, *args: Any, **kwargs: Any) -> MetricResult:
        raise TypeError(
            f"{self.identity.metric_id} is dataset-scoped; use start/update/finalize"
        )

    @abstractmethod
    def start(self, context: Optional[Mapping[str, Any]] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        view: EvaluationView,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> MetricResult:
        raise NotImplementedError
