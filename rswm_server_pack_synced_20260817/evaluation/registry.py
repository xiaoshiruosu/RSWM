"""Fail-closed registry for versioned benchmark metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from .base import DatasetMetric, Metric, SampleMetric
from .errors import MetricRegistryError
from .identity import MetricIdentity, MetricScope


MetricKey = Tuple[str, str]


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: Dict[MetricKey, Metric] = {}

    def register(self, metric: Metric) -> Metric:
        if not isinstance(metric, Metric):
            raise MetricRegistryError("registered object must implement Metric")
        metric.identity.validate()
        scope = metric.identity.scope
        if scope is MetricScope.SAMPLE and not isinstance(metric, SampleMetric):
            raise MetricRegistryError(
                f"sample metric {metric.identity.metric_id!r} lacks compute() contract"
            )
        if scope is MetricScope.DATASET and not isinstance(metric, DatasetMetric):
            raise MetricRegistryError(
                f"dataset metric {metric.identity.metric_id!r} lacks accumulator contract"
            )
        key = (metric.identity.metric_id, metric.identity.version)
        if key in self._metrics:
            raise MetricRegistryError(f"duplicate metric registration: {key!r}")
        self._metrics[key] = metric
        return metric

    def resolve(self, metric_id: str, version: Optional[str] = None) -> Metric:
        if version is not None:
            key = (str(metric_id), str(version))
            try:
                return self._metrics[key]
            except KeyError as exc:
                raise MetricRegistryError(f"unknown metric: {key!r}") from exc
        matches = [
            metric for (registered_id, _), metric in self._metrics.items()
            if registered_id == metric_id
        ]
        if not matches:
            raise MetricRegistryError(f"unknown metric_id: {metric_id!r}")
        if len(matches) != 1:
            versions = sorted(metric.identity.version for metric in matches)
            raise MetricRegistryError(
                f"metric_id {metric_id!r} is ambiguous; specify version from {versions}"
            )
        return matches[0]

    def list(self) -> Tuple[Metric, ...]:
        return tuple(self._metrics[key] for key in sorted(self._metrics))

    def identities(self) -> Tuple[MetricIdentity, ...]:
        return tuple(metric.identity for metric in self.list())

    def __len__(self) -> int:
        return len(self._metrics)

    def __iter__(self) -> Iterable[Metric]:
        return iter(self.list())


def build_m4_metric_registry(
    *, path_estimator: str, device: str = "cpu"
) -> MetricRegistry:
    """Build the M4 panel without importing a model or loading metric weights."""

    from .semantic import UnavailableSemanticMetric
    from .spatial import LegacyTrajectoryMetric
    from .temporal import LegacyR3DFVDAccumulator
    from .visual import (
        LegacyFIDAccumulator,
        legacy_lpips_metric,
        legacy_psnr_metric,
        legacy_ssim_metric,
    )

    registry = MetricRegistry()
    for metric in (
        legacy_ssim_metric(),
        legacy_psnr_metric(),
        legacy_lpips_metric(device=device),
        LegacyFIDAccumulator(device=device),
        LegacyR3DFVDAccumulator(device=device),
    ):
        registry.register(metric)
    for metric_key in (
        "net_direction_error",
        "segment_angular_mean",
        "segment_success_rate_30",
        "segment_success_rate_45",
        "turn_recall",
        "turn_precision",
        "turn_f1",
        "turn_timing_error",
        "waypoint_error",
        "waypoint_endpoint_error",
        "frechet",
        "dtw",
    ):
        registry.register(
            LegacyTrajectoryMetric(metric_key, path_estimator=path_estimator)
        )
    registry.register(UnavailableSemanticMetric())
    return registry
