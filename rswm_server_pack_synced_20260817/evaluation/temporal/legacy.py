"""Dataset wrapper for the existing R3D-18 FVD proxy implementation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark.core.identity import ComponentIdentity

from .._distribution import LegacyDistributionAccumulator
from .._legacy import (
    LEGACY_METRICS_PATH,
    LEGACY_SUBMISSION_CONTRACT_PATH,
    load_legacy_metrics,
)
from ..identity import (
    FutureOnlyOwner,
    MetricCategory,
    MetricDirection,
    MetricIdentity,
    MetricScope,
    implementation_hash_for_sources,
)
from ..views import RolloutView

_VERSION = "legacy-r3d18-proxy-wrapper-v1"


def _source_hash() -> str:
    return implementation_hash_for_sources(
        {
            "evaluation.temporal.legacy": Path(__file__),
            "lora_remotesense.new.metrics": LEGACY_METRICS_PATH,
            "lora_remotesense.new.submission_contract": LEGACY_SUBMISSION_CONTRACT_PATH,
        }
    )


class LegacyR3DFVDAccumulator(LegacyDistributionAccumulator):
    """R3D-18 proxy FVD; deliberately not named or admitted as I3D-FVD."""

    def __init__(self, *, device: str = "cpu", minimum_n: int = 32) -> None:
        if int(minimum_n) < 32:
            raise ValueError("FVD requires at least 32 full rollout samples")
        self.device = str(device)
        checkpoint = ComponentIdentity(
            kind="metric_checkpoint",
            artifact_id="torchvision-r3d18-kinetics400-proxy",
            schema="rswm.metric_checkpoint/v1",
            version="runtime-resolved",
            sha256=None,
            status="declared",
        )
        identity = MetricIdentity.create(
            metric_id="temporal.fvd_r3d18_proxy",
            version=_VERSION,
            category=MetricCategory.TEMPORAL,
            scope=MetricScope.DATASET,
            higher_or_lower_better=MetricDirection.LOWER,
            preprocessing={
                "input_view": "future_only",
                "clip_layout": "THWC",
                "temporal_sampling": "uniform_16_frames_with_last_frame_padding",
                "resize": [16, 112, 112],
                "feature_extractor": "torchvision_r3d_18",
                "standard_i3d_fvd": False,
                "minimum_sample_count": int(minimum_n),
                "device": self.device,
            },
            checkpoint_identity=checkpoint,
            implementation_hash=_source_hash(),
            future_only_owner=FutureOnlyOwner.EVALUATION_RUNNER,
        )
        super().__init__(identity, minimum_n=minimum_n)

    def _ingest(self, prediction: RolloutView, ground_truth: RolloutView) -> None:
        self._predictions.append(np.array(prediction.frames, copy=True))
        self._ground_truths.append(np.array(ground_truth.frames, copy=True))

    def _compute_value(self) -> float:
        return float(
            load_legacy_metrics().compute_fvd(
                self._predictions,
                self._ground_truths,
                device=self.device,
            )
        )
