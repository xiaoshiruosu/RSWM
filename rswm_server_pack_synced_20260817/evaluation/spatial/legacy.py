"""Scalar wrappers around trajectory_metrics.score_trajectory()."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional

from .._legacy import (
    LEGACY_GEOM_REGISTRATION_PATH,
    LEGACY_METRICS_PATH,
    LEGACY_SUBMISSION_CONTRACT_PATH,
    LEGACY_TRAJECTORY_PATH,
    load_legacy_trajectory_metrics,
)
from ..base import SampleMetric
from ..identity import (
    FutureOnlyOwner,
    MetricCategory,
    MetricDirection,
    MetricIdentity,
    MetricScope,
    implementation_hash_for_sources,
)
from ..result import MetricResult
from ..views import EvaluationView

_VERSION = "trajectory-score-v3-wrapper-v1"

_DIRECTIONS = {
    "net_direction_error": MetricDirection.LOWER,
    "segment_angular_mean": MetricDirection.LOWER,
    "segment_success_rate_30": MetricDirection.HIGHER,
    "segment_success_rate_45": MetricDirection.HIGHER,
    "turn_recall": MetricDirection.HIGHER,
    "turn_precision": MetricDirection.HIGHER,
    "turn_f1": MetricDirection.HIGHER,
    "turn_timing_error": MetricDirection.LOWER,
    "waypoint_error": MetricDirection.LOWER,
    "waypoint_endpoint_error": MetricDirection.LOWER,
    "frechet": MetricDirection.LOWER,
    "dtw": MetricDirection.LOWER,
}


def _source_hash(path_estimator: str) -> str:
    sources = {
        "evaluation.spatial.legacy": Path(__file__),
        "lora_remotesense.new.metrics": LEGACY_METRICS_PATH,
        "lora_remotesense.new.submission_contract": LEGACY_SUBMISSION_CONTRACT_PATH,
        "lora_remotesense.new.trajectory_metrics": LEGACY_TRAJECTORY_PATH,
    }
    if path_estimator == "registration":
        sources["lora_remotesense.new.geom_registration"] = (
            LEGACY_GEOM_REGISTRATION_PATH
        )
    return implementation_hash_for_sources(sources)


class LegacyTrajectoryMetric(SampleMetric):
    """Extract one scalar without changing score_trajectory's mathematics."""

    def __init__(self, metric_key: str, *, path_estimator: str) -> None:
        if metric_key not in _DIRECTIONS:
            raise ValueError(
                f"unsupported trajectory scalar {metric_key!r}; "
                f"expected one of {sorted(_DIRECTIONS)}"
            )
        path_estimator = str(path_estimator).lower()
        if path_estimator not in {"flow", "registration"}:
            raise ValueError("path_estimator must be 'flow' or 'registration'")
        self.metric_key = metric_key
        self.path_estimator = path_estimator
        self.identity = MetricIdentity.create(
            metric_id=f"spatial.trajectory.{metric_key}",
            version=_VERSION,
            category=MetricCategory.SPATIAL,
            scope=MetricScope.SAMPLE,
            higher_or_lower_better=_DIRECTIONS[metric_key],
            preprocessing={
                "input_view": "full_rollout",
                "frame_layout": "THWC",
                "path_estimator": path_estimator,
                "legacy_orchestrator": "trajectory_metrics.score_trajectory",
                "scalar_key": metric_key,
            },
            checkpoint_identity=None,
            implementation_hash=_source_hash(path_estimator),
            future_only_owner=FutureOnlyOwner.NOT_APPLICABLE,
        )

    def compute(
        self,
        view: EvaluationView,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MetricResult:
        metadata = metadata or {}
        missing = [name for name in ("waypoints", "segment_angles") if name not in metadata]
        if missing:
            return MetricResult.invalid(
                f"missing_trajectory_metadata:{','.join(missing)}",
                N=1,
                metric_identity=self.identity,
            )
        if "path_estimator" in metadata and str(metadata["path_estimator"]).lower() != self.path_estimator:
            return MetricResult.invalid(
                "path_estimator_identity_mismatch",
                N=1,
                metric_identity=self.identity,
            )
        prediction, _ = view.frames_for(self.identity)
        try:
            num_frames = int(metadata.get("num_frames", len(prediction.frames)))
        except (TypeError, ValueError, OverflowError):
            return MetricResult.invalid(
                "num_frames_is_not_an_integer",
                N=1,
                metric_identity=self.identity,
            )
        if num_frames != len(prediction.frames):
            return MetricResult.invalid(
                f"num_frames_mismatch:{num_frames}!={len(prediction.frames)}",
                N=1,
                metric_identity=self.identity,
            )
        kwargs = {
            "num_frames": num_frames,
            "min_mag": metadata.get("min_mag", 1.5),
            "text_only": metadata.get("text_only", False),
            "window_size": metadata.get("window_size"),
            "frame_positions": metadata.get("frame_positions"),
            "poses": metadata.get("poses"),
            "path_estimator": self.path_estimator,
        }
        try:
            score = load_legacy_trajectory_metrics().score_trajectory(
                prediction.frames,
                metadata["waypoints"],
                metadata["segment_angles"],
                **kwargs,
            )
        except Exception as exc:
            return MetricResult.invalid(
                f"legacy_metric_error:{type(exc).__name__}:{exc}",
                N=1,
                metric_identity=self.identity,
            )
        if not bool(score.get("valid")):
            return MetricResult.invalid(
                "trajectory_estimator_invalid",
                N=1,
                metric_identity=self.identity,
            )
        value = score.get(self.metric_key)
        if value is None:
            reason = "metric_not_defined_for_sample"
            if score.get("net_direction_gated") and self.metric_key == "net_direction_error":
                reason = "net_direction_gated_by_gt_path"
            return MetricResult.invalid(
                reason,
                N=1,
                metric_identity=self.identity,
            )
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            return MetricResult.invalid(
                "legacy_metric_value_is_not_scalar",
                N=1,
                metric_identity=self.identity,
            )
        if not math.isfinite(value):
            return MetricResult.invalid(
                "non_finite_metric_value",
                N=1,
                metric_identity=self.identity,
            )
        return MetricResult.success(value, N=1, metric_identity=self.identity)
