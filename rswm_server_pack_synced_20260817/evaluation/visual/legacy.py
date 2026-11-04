"""Thin wrappers around the frozen visual metrics in new/metrics.py."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from benchmark.core.identity import ComponentIdentity

from .._distribution import LegacyDistributionAccumulator
from .._legacy import (
    LEGACY_METRICS_PATH,
    LEGACY_SUBMISSION_CONTRACT_PATH,
    load_legacy_metrics,
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
from ..views import EvaluationView, RolloutView

_VERSION = "legacy-v3.0-wrapper-v1"


def _source_hash() -> str:
    return implementation_hash_for_sources(
        {
            "evaluation.visual.legacy": Path(__file__),
            "lora_remotesense.new.metrics": LEGACY_METRICS_PATH,
            "lora_remotesense.new.submission_contract": LEGACY_SUBMISSION_CONTRACT_PATH,
        }
    )


def _declared_checkpoint(artifact_id: str, version: str) -> ComponentIdentity:
    return ComponentIdentity(
        kind="metric_checkpoint",
        artifact_id=artifact_id,
        schema="rswm.metric_checkpoint/v1",
        version=version,
        sha256=None,
        status="declared",
    )


_SAMPLE_DEFINITIONS = {
    "ssim": {
        "metric_id": "visual.ssim",
        "function": "compute_video_ssim",
        "direction": MetricDirection.HIGHER,
        "preprocessing": {
            "input_view": "future_only",
            "frame_layout": "THWC",
            "dtype": "uint8",
            "channel_axis": 2,
            "data_range": 255,
            "aggregation": "arithmetic_mean_over_future_frames",
        },
        "checkpoint": None,
    },
    "psnr": {
        "metric_id": "visual.psnr",
        "function": "compute_video_psnr",
        "direction": MetricDirection.HIGHER,
        "preprocessing": {
            "input_view": "future_only",
            "frame_layout": "THWC",
            "dtype": "uint8",
            "data_range": 255,
            "aggregation": "arithmetic_mean_over_future_frames",
        },
        "checkpoint": None,
    },
    "lpips": {
        "metric_id": "visual.lpips_alex",
        "function": "compute_video_lpips",
        "direction": MetricDirection.LOWER,
        "preprocessing": {
            "input_view": "future_only",
            "frame_layout": "THWC",
            "dtype": "uint8",
            "range_conversion": "uint8_[0,255]_to_float_[-1,1]",
            "aggregation": "arithmetic_mean_over_future_frames",
        },
        "checkpoint": _declared_checkpoint("lpips-alex", "runtime-resolved"),
    },
}


class LegacyVisualSampleMetric(SampleMetric):
    """Call one existing per-video function after runner-owned t=0 removal."""

    def __init__(self, metric: str, *, device: str = "cpu") -> None:
        try:
            definition = _SAMPLE_DEFINITIONS[metric]
        except KeyError as exc:
            raise ValueError(f"unsupported legacy visual metric: {metric!r}") from exc
        self.metric = metric
        self.device = str(device)
        preprocessing = dict(definition["preprocessing"])
        if metric == "lpips":
            preprocessing["device"] = self.device
        self.identity = MetricIdentity.create(
            metric_id=definition["metric_id"],
            version=_VERSION,
            category=MetricCategory.VISUAL,
            scope=MetricScope.SAMPLE,
            higher_or_lower_better=definition["direction"],
            preprocessing=preprocessing,
            checkpoint_identity=definition["checkpoint"],
            implementation_hash=_source_hash(),
            future_only_owner=FutureOnlyOwner.EVALUATION_RUNNER,
        )
        self._function_name = definition["function"]

    def compute(
        self,
        view: EvaluationView,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> MetricResult:
        prediction, ground_truth = view.frames_for(self.identity)
        function = getattr(load_legacy_metrics(), self._function_name)
        try:
            if self.metric == "lpips":
                value = function(prediction.frames, ground_truth.frames, self.device)
            else:
                value = function(prediction.frames, ground_truth.frames)
            value = float(value)
        except Exception as exc:
            return MetricResult.invalid(
                f"legacy_metric_error:{type(exc).__name__}:{exc}",
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


class LegacyFIDAccumulator(LegacyDistributionAccumulator):
    """Dataset-level FID over every predicted frame, never per-sample FID."""

    def __init__(
        self,
        *,
        device: str = "cpu",
        batch_size: int = 64,
        minimum_n: int = 32,
    ) -> None:
        if int(minimum_n) < 32:
            raise ValueError("FID requires at least 32 full rollout samples")
        self.device = str(device)
        self.batch_size = int(batch_size)
        identity = MetricIdentity.create(
            metric_id="visual.fid_inception_v3",
            version=_VERSION,
            category=MetricCategory.VISUAL,
            scope=MetricScope.DATASET,
            higher_or_lower_better=MetricDirection.LOWER,
            preprocessing={
                "input_view": "future_only",
                "frame_selection": "all_predicted_frames",
                "frame_layout": "HWC",
                "feature_extractor": "torchvision_inception_v3",
                "minimum_sample_count": int(minimum_n),
                "device": self.device,
                "batch_size": self.batch_size,
            },
            checkpoint_identity=_declared_checkpoint(
                "torchvision-inception-v3", "runtime-resolved"
            ),
            implementation_hash=_source_hash(),
            future_only_owner=FutureOnlyOwner.EVALUATION_RUNNER,
        )
        super().__init__(identity, minimum_n=minimum_n)

    def _ingest(self, prediction: RolloutView, ground_truth: RolloutView) -> None:
        self._predictions.extend(np.array(frame, copy=True) for frame in prediction.frames)
        self._ground_truths.extend(
            np.array(frame, copy=True) for frame in ground_truth.frames
        )

    def _compute_value(self) -> float:
        return float(
            load_legacy_metrics().compute_fid(
                self._predictions,
                self._ground_truths,
                device=self.device,
                batch_size=self.batch_size,
            )
        )


def legacy_ssim_metric() -> LegacyVisualSampleMetric:
    return LegacyVisualSampleMetric("ssim")


def legacy_psnr_metric() -> LegacyVisualSampleMetric:
    return LegacyVisualSampleMetric("psnr")


def legacy_lpips_metric(*, device: str = "cpu") -> LegacyVisualSampleMetric:
    return LegacyVisualSampleMetric("lpips", device=device)
