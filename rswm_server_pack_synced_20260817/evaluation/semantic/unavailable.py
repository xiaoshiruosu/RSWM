"""Explicit placeholder for the not-yet-frozen output semantic metric."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

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


class UnavailableSemanticMetric(SampleMetric):
    """Return NOT_AVAILABLE; SemanticGate is not an output metric."""

    def __init__(self) -> None:
        self.identity = MetricIdentity.create(
            metric_id="semantic.output_preservation_not_available",
            version="placeholder-v1",
            category=MetricCategory.SEMANTIC,
            scope=MetricScope.SAMPLE,
            higher_or_lower_better=MetricDirection.HIGHER,
            preprocessing={
                "input_view": "future_only",
                "status": "NOT_AVAILABLE",
                "semantic_gate_reused": False,
                "headline_eligible": False,
            },
            checkpoint_identity=None,
            implementation_hash=implementation_hash_for_sources(
                {"evaluation.semantic.unavailable": Path(__file__)}
            ),
            future_only_owner=FutureOnlyOwner.EVALUATION_RUNNER,
        )

    def compute(
        self,
        view: EvaluationView,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MetricResult:
        view.frames_for(self.identity)
        return MetricResult.invalid(
            "NOT_AVAILABLE:no_frozen_output_level_semantic_metric",
            N=1,
            metric_identity=self.identity,
        )
