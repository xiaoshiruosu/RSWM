from __future__ import annotations

import unittest

import numpy as np

from benchmark.core.identity import ComponentIdentity, canonical_sha256
from evaluation import (
    DatasetMetric,
    EvaluationError,
    EvaluationView,
    FutureOnlyOwner,
    MetricCategory,
    MetricDirection,
    MetricIdentity,
    MetricRegistry,
    MetricRegistryError,
    MetricResult,
    MetricScope,
    SampleMetric,
    build_m4_metric_registry,
)


def _digest(label: str) -> str:
    return canonical_sha256({"test": label})


def _identity(*, metric_id: str, scope: MetricScope) -> MetricIdentity:
    return MetricIdentity.create(
        metric_id=metric_id,
        version="fixture-v1",
        category=MetricCategory.VISUAL,
        scope=scope,
        higher_or_lower_better=MetricDirection.HIGHER,
        preprocessing={"input_view": "future_only", "resize": [8, 12]},
        checkpoint_identity=None,
        implementation_hash=_digest(metric_id),
        future_only_owner=FutureOnlyOwner.EVALUATION_RUNNER,
    )


class _SampleMetric(SampleMetric):
    def __init__(self, metric_id: str = "fixture.sample") -> None:
        self.identity = _identity(metric_id=metric_id, scope=MetricScope.SAMPLE)

    def compute(self, view, metadata=None):
        view.frames_for(self.identity)
        return MetricResult.success(1.0, N=1, metric_identity=self.identity)


class _DatasetMetric(DatasetMetric):
    def __init__(self) -> None:
        self.identity = _identity(
            metric_id="fixture.dataset", scope=MetricScope.DATASET
        )

    def start(self, context=None):
        return None

    def update(self, view, metadata=None):
        return None

    def finalize(self):
        return MetricResult.invalid(
            "fixture", N=0, metric_identity=self.identity
        )


class MetricIdentityRegistryTests(unittest.TestCase):
    def test_metric_identity_roundtrip_and_lineage_binding(self):
        checkpoint = ComponentIdentity(
            kind="metric_checkpoint",
            artifact_id="fixture-feature-extractor",
            schema="rswm.metric_checkpoint/v1",
            version="fixture-v1",
            sha256=_digest("checkpoint"),
            status="verified",
        )
        identity = MetricIdentity.create(
            metric_id="visual.fixture",
            version="v1",
            category=MetricCategory.VISUAL,
            scope=MetricScope.SAMPLE,
            higher_or_lower_better=MetricDirection.LOWER,
            preprocessing={"input_view": "future_only", "shape": [8, 12]},
            checkpoint_identity=checkpoint,
            implementation_hash=_digest("implementation"),
            future_only_owner=FutureOnlyOwner.EVALUATION_RUNNER,
        )
        restored = MetricIdentity.from_document(identity.to_document())
        self.assertEqual(identity, restored)
        artifact = identity.artifact_ref()
        self.assertEqual(artifact.kind, "metric")
        self.assertEqual(artifact.sha256, identity.identity_checksum)
        self.assertEqual(artifact.artifact_id, "visual.fixture@v1")

    def test_result_binds_full_metric_identity_and_null_invalid_policy(self):
        identity = _identity(metric_id="fixture.bound", scope=MetricScope.SAMPLE)
        result = MetricResult.success(0.5, N=1, metric_identity=identity)
        document = result.to_document()
        self.assertEqual(
            document["metric_identity"]["identity_checksum"],
            identity.identity_checksum,
        )
        invalid = MetricResult.invalid(
            "NOT_AVAILABLE", N=3, metric_identity=identity
        )
        self.assertIsNone(invalid.value)
        with self.assertRaises(EvaluationError):
            MetricResult(0.0, False, "missing", 1, identity)

    def test_registry_enforces_scope_and_duplicate_identity(self):
        registry = MetricRegistry()
        sample = registry.register(_SampleMetric())
        dataset = registry.register(_DatasetMetric())
        self.assertIs(registry.resolve("fixture.sample"), sample)
        self.assertIs(
            registry.resolve("fixture.dataset", "fixture-v1"), dataset
        )
        self.assertEqual(len(registry), 2)
        with self.assertRaises(MetricRegistryError):
            registry.register(_SampleMetric())

    def test_dataset_metric_refuses_per_sample_compute(self):
        metric = _DatasetMetric()
        frames = np.zeros((3, 8, 12, 3), dtype=np.uint8)
        view = EvaluationView.create(frames, frames)
        with self.assertRaises(TypeError):
            metric.compute(view)

    def test_m4_registry_has_all_categories_without_loading_models(self):
        registry = build_m4_metric_registry(
            path_estimator="registration", device="cpu"
        )
        categories = {identity.category for identity in registry.identities()}
        self.assertEqual(categories, set(MetricCategory))
        self.assertEqual(len(registry), 18)
        self.assertEqual(
            registry.resolve("visual.fid_inception_v3").identity.scope,
            MetricScope.DATASET,
        )
        proxy_id = registry.resolve("temporal.fvd_r3d18_proxy").identity.metric_id
        self.assertNotIn("i3d", proxy_id)


if __name__ == "__main__":
    unittest.main()
