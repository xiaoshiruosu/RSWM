"""RSWM-Bench M4-M7 evaluation framework (CPU-safe, model-independent)."""

from .base import DatasetMetric, Metric, SampleMetric
from .errors import (
    AccumulatorStateError,
    EvaluationError,
    FutureOnlyOwnershipError,
    InvalidMetricResultError,
    MetricIdentityError,
    MetricRegistryError,
)
from .identity import (
    FutureOnlyOwner,
    MetricCategory,
    MetricDirection,
    MetricIdentity,
    MetricScope,
)
from .registry import MetricRegistry, build_m4_metric_registry
from .bundle import ResultBundle, SampleIdentity
from .result import MetricResult
from .runner import EvaluationRunner
from .views import EvaluationView, FullRolloutView, FutureOnlyView

__all__ = [
    "AccumulatorStateError",
    "DatasetMetric",
    "EvaluationError",
    "EvaluationRunner",
    "EvaluationView",
    "FullRolloutView",
    "FutureOnlyOwner",
    "FutureOnlyOwnershipError",
    "FutureOnlyView",
    "InvalidMetricResultError",
    "Metric",
    "MetricCategory",
    "MetricDirection",
    "MetricIdentity",
    "MetricIdentityError",
    "MetricRegistry",
    "MetricRegistryError",
    "MetricResult",
    "MetricScope",
    "ResultBundle",
    "SampleIdentity",
    "SampleMetric",
    "build_m4_metric_registry",
]
