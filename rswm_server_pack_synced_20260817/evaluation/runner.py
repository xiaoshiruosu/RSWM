"""CPU-safe sample evaluation runner for the M7 contract dry run.

The runner owns construction of the paired evaluation view.  Metric wrappers
remain the only owners of their mathematical definitions; this module only
enforces identity, visibility and future-only ownership contracts.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from benchmark.core.types import (
    BenchmarkSample,
    GroundTruthBundle,
    Rollout,
    TaskId,
    TrackId,
)

from .base import SampleMetric
from .errors import EvaluationError, InvalidMetricResultError
from .identity import FutureOnlyOwner
from .result import MetricResult
from .views import EvaluationView


class EvaluationRunner:
    """Evaluate one canonical rollout with identity-bearing sample metrics.

    Dataset accumulators retain the M4 ``start/update/finalize`` interface and
    are intentionally not collapsed into this per-sample dry-run path.
    """

    def __init__(
        self,
        metrics: Iterable[SampleMetric],
        *,
        fail_closed: bool = True,
    ) -> None:
        resolved = tuple(metrics)
        if not resolved:
            raise EvaluationError("EvaluationRunner requires at least one metric")
        if any(not isinstance(metric, SampleMetric) for metric in resolved):
            raise EvaluationError(
                "M7 sample EvaluationRunner accepts SampleMetric instances only; "
                "dataset metrics use start/update/finalize"
            )
        keys = [
            (metric.identity.metric_id, metric.identity.version)
            for metric in resolved
        ]
        if len(set(keys)) != len(keys):
            raise EvaluationError("EvaluationRunner metric identities must be unique")
        for metric in resolved:
            metric.identity.validate()

        slicing_owners = {
            metric.identity.future_only_owner
            for metric in resolved
            if metric.identity.future_only_owner is not FutureOnlyOwner.NOT_APPLICABLE
        }
        if len(slicing_owners) > 1:
            raise EvaluationError(
                "one evaluation may have only one future-only owner"
            )
        self.metrics: Tuple[SampleMetric, ...] = resolved
        self.fail_closed = bool(fail_closed)
        self.future_only_owner = (
            next(iter(slicing_owners))
            if slicing_owners
            else FutureOnlyOwner.EVALUATION_RUNNER
        )

    def evaluate(
        self,
        *,
        task_id: TaskId,
        track_id: TrackId,
        rollout: Rollout,
        ground_truth: GroundTruthBundle,
        sample: BenchmarkSample,
    ) -> Tuple[MetricResult, ...]:
        task_id = TaskId(task_id)
        track_id = TrackId(track_id)
        if rollout.sample_id != sample.id:
            raise EvaluationError("rollout sample_id differs from benchmark sample")
        if len(rollout.frames) != sample.horizon:
            raise EvaluationError("rollout horizon differs from benchmark sample")
        if len(ground_truth.rollout) != sample.horizon:
            raise EvaluationError("ground-truth horizon differs from benchmark sample")
        if rollout.model_identity is None:
            raise EvaluationError("M7 rollout is missing structured model identity")

        view = EvaluationView.create(
            np.asarray(rollout.frames),
            np.asarray(ground_truth.rollout),
            future_only_owner=self.future_only_owner,
        )
        metadata = {
            "task_id": task_id.value,
            "track_id": track_id.value,
            "sample_id": sample.id,
            "horizon": sample.horizon,
        }
        results = tuple(metric.compute(view, metadata) for metric in self.metrics)
        if any(not isinstance(result, MetricResult) for result in results):
            raise EvaluationError("metric compute() must return MetricResult")
        if any(not result.valid for result in results) and self.fail_closed:
            raise InvalidMetricResultError(results)
        return results
