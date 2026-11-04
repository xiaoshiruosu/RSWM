from __future__ import annotations

import builtins
import hashlib
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmark.core.identity import (
    ArtifactRef,
    CheckpointIdentity,
    canonical_sha256,
    sha256_file,
)
from benchmark.core.errors import ArtifactLineageError
from benchmark.core.types import TrackId
from benchmark.datasets import LegacyH5Adapter
from benchmark.runtime import CpuEndToEndDryRun
from benchmark.spec import load_benchmark_spec
from benchmark.tasks import build_task_from_spec
from evaluation import (
    EvaluationError,
    EvaluationRunner,
    FutureOnlyOwner,
    InvalidMetricResultError,
    MetricCategory,
    MetricDirection,
    MetricIdentity,
    MetricResult,
    MetricScope,
    ResultBundle,
    SampleMetric,
)
from evaluation.visual import legacy_psnr_metric
from lora_remotesense.new import submission_contract as legacy_submission_contract
from models import LingBotRuntimeConfig, LingBotWorldModelAdapter
from tests.benchmark.fixture_factory import FixtureTempDirectory, create_fixture


class _VisibilityWanMock:
    name = "m7-wan-cpu-mock"
    capabilities = object()

    def __init__(self) -> None:
        self.calls = []

    def generate(
        self,
        *,
        first_frame,
        prompt,
        num_frames,
        size,
        pose_cond=None,
        seed=None,
    ):
        self.calls.append(
            {
                "first_frame": first_frame,
                "prompt": prompt,
                "num_frames": num_frames,
                "size": size,
                "pose_cond": pose_cond,
                "seed": seed,
                "pose_mode": os.environ.get("RSWM_POSE_MODE"),
            }
        )
        image = np.asarray(first_frame)
        frames = np.empty((num_frames, *image.shape), dtype=np.uint8)
        frames[0] = image
        # Keep t=0 exact and every future frame finite under legacy PSNR.
        frames[1:] = 255 - image
        return frames


class _BackendFactory:
    def __init__(self) -> None:
        self.kwargs = []
        self.backends = []

    def __call__(self, **kwargs):
        self.kwargs.append(kwargs)
        backend = _VisibilityWanMock()
        self.backends.append(backend)
        return backend


class _RecordingFutureMetric(SampleMetric):
    def __init__(self) -> None:
        self.identity = MetricIdentity.create(
            metric_id="visual.m7_future_owner_probe",
            version="fixture-v1",
            category=MetricCategory.VISUAL,
            scope=MetricScope.SAMPLE,
            higher_or_lower_better=MetricDirection.HIGHER,
            preprocessing={"input_view": "future_only"},
            implementation_hash=canonical_sha256({"fixture": "future-owner"}),
            future_only_owner=FutureOnlyOwner.EVALUATION_RUNNER,
        )
        self.calls = []

    def compute(self, view, metadata=None):
        prediction, ground_truth = view.frames_for(self.identity)
        self.calls.append(
            {
                "prediction_count": len(prediction.frames),
                "ground_truth_count": len(ground_truth.frames),
                "source_frame_count": prediction.source_frame_count,
                "source_start_index": prediction.source_start_index,
                "first_future": np.array(prediction.frames[0], copy=True),
            }
        )
        return MetricResult.success(
            float(np.mean(prediction.frames[0])),
            N=1,
            metric_identity=self.identity,
        )


class _InvalidMetric(SampleMetric):
    def __init__(self) -> None:
        self.identity = MetricIdentity.create(
            metric_id="visual.m7_invalid_probe",
            version="fixture-v1",
            category=MetricCategory.VISUAL,
            scope=MetricScope.SAMPLE,
            higher_or_lower_better=MetricDirection.HIGHER,
            preprocessing={"input_view": "future_only"},
            implementation_hash=canonical_sha256({"fixture": "invalid"}),
            future_only_owner=FutureOnlyOwner.EVALUATION_RUNNER,
        )

    def compute(self, view, metadata=None):
        view.frames_for(self.identity)
        return MetricResult.invalid(
            "fixture_invalid_status",
            N=1,
            metric_identity=self.identity,
        )


class M7EndToEndRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = FixtureTempDirectory()
        self.paths = create_fixture(Path(self.temp.name), sample_count=2)
        self.datasets = []

    def tearDown(self):
        for dataset in self.datasets:
            dataset.close()
        self.temp.cleanup()

    def _pipeline(self, track_id: TrackId, metric: SampleMetric):
        frozen = load_benchmark_spec(self.paths.spec)
        spec = replace(
            frozen,
            conditioning_track=TrackId(track_id),
            spec_checksum=None,
        ).with_checksum()
        dataset = LegacyH5Adapter(
            self.paths.h5,
            self.paths.manifest,
            benchmark_spec=spec,
        )
        self.datasets.append(dataset)
        factory = _BackendFactory()
        checkpoint = CheckpointIdentity(
            repository="robbyant/lingbot-world-base-cam",
            revision="m7-cpu-mock-no-weights",
            status="declared",
        )
        adapter = LingBotWorldModelAdapter(
            checkpoint_identity=checkpoint,
            checkpoint_path="/fixture/not-read/lingbot-world-base-cam",
            runtime_config=LingBotRuntimeConfig(device="cpu"),
            backbone_factory=factory,
        )
        checkpoint_artifact = ArtifactRef(
            kind="checkpoint",
            artifact_id=checkpoint.repository,
            sha256=canonical_sha256(
                {"fixture": "m7-mock-checkpoint", "weights_loaded": False}
            ),
            schema="rswm.mock_checkpoint/v1",
            version="m7-fixture",
            metadata={"mock": True, "weights_loaded": False},
        )
        adapter_path = Path(__file__).resolve().parents[2] / "models" / "adapters" / "lingbot_adapter.py"
        adapter_artifact = ArtifactRef(
            kind="adapter",
            artifact_id=adapter.adapter_identity,
            sha256=sha256_file(adapter_path),
            schema="rswm.world_model_adapter/v1",
            version="rswm.lingbot_world_adapter/v1",
        )
        pipeline = CpuEndToEndDryRun(
            dataset=dataset,
            task=build_task_from_spec(spec),
            world_model=adapter,
            evaluator=EvaluationRunner((metric,)),
            checkpoint_artifact=checkpoint_artifact,
            adapter_artifact=adapter_artifact,
        )
        return pipeline, factory, dataset

    def test_track_a_hides_pose_and_gt_and_binds_complete_identity(self):
        pipeline, factory, _dataset = self._pipeline(
            TrackId.A_TEXT, legacy_psnr_metric()
        )
        outcome = pipeline.run_sample("sample_00000", generation_seed=101)
        call = factory.backends[0].calls[0]

        self.assertIsNone(call["pose_cond"])
        self.assertEqual(call["pose_mode"], "drop")
        self.assertIsNone(outcome.model_input.trajectory_condition.poses)
        self.assertIsNone(outcome.model_input.trajectory_condition.intrinsics)
        self.assertFalse(hasattr(outcome.model_input, "ground_truth"))
        self.assertTrue(outcome.validation_report.valid)

        bundle = outcome.result_bundle
        self.assertTrue(bundle.valid)
        self.assertFalse(bundle.admission_eligible)
        self.assertEqual(bundle.model_identity.release_identity, "LingBot-World-base-cam")
        self.assertEqual(bundle.adapter_identity, "LingBotWorldAdapter")
        self.assertEqual(bundle.execution_identity.route, "benchmark-adapter")
        self.assertEqual(bundle.sample_identity.track_id, TrackId.A_TEXT)
        self.assertEqual(
            tuple(item.kind for item in bundle.artifact_lineage.artifacts),
            (
                "checkpoint",
                "adapter",
                "benchmark_spec",
                "manifest",
                "metric",
                "result",
            ),
        )
        self.assertEqual(
            bundle.artifact_lineage.artifacts[-1].sha256,
            bundle.result_payload_checksum,
        )
        self.assertEqual(
            bundle.metric_results[0].metric_identity.identity_checksum,
            bundle.artifact_lineage.artifacts[-2].sha256,
        )
        document = bundle.to_document()
        self.assertNotIn("ground_truth", json.dumps(document, sort_keys=True))
        restored = ResultBundle.from_document(document)
        self.assertEqual(restored.bundle_checksum, bundle.bundle_checksum)
        with self.assertRaises(EvaluationError):
            restored.assert_admission_eligible()

    def test_track_b_passes_camera_condition(self):
        pipeline, factory, _dataset = self._pipeline(
            TrackId.B_CAMERA, legacy_psnr_metric()
        )
        outcome = pipeline.run_sample("sample_00000", generation_seed=102)
        call = factory.backends[0].calls[0]
        condition = outcome.model_input.trajectory_condition

        self.assertEqual(call["pose_mode"], "native")
        self.assertEqual(condition.poses.shape, (17, 4, 4))
        self.assertEqual(condition.intrinsics.shape, (17, 4))
        np.testing.assert_array_equal(call["pose_cond"]["c2w"], condition.poses)
        np.testing.assert_array_equal(
            call["pose_cond"]["intrinsics"], condition.intrinsics
        )
        self.assertEqual(outcome.result_bundle.sample_identity.track_id, TrackId.B_CAMERA)
        self.assertEqual(outcome.rollout.model_identity, outcome.result_bundle.model_identity)

    def test_result_or_lineage_tampering_fails_closed(self):
        pipeline, _factory, _dataset = self._pipeline(
            TrackId.B_CAMERA, legacy_psnr_metric()
        )
        document = pipeline.run_sample("sample_00000").result_bundle.to_document()

        changed_value = json.loads(json.dumps(document))
        changed_value["metric_results"][0]["value"] += 0.5
        with self.assertRaises(EvaluationError):
            ResultBundle.from_document(changed_value)

        changed_lineage = json.loads(json.dumps(document))
        changed_lineage["artifact_lineage"]["artifacts"][4]["sha256"] = "0" * 64
        with self.assertRaises(ArtifactLineageError):
            ResultBundle.from_document(changed_lineage)

    def test_future_only_is_materialized_exactly_once(self):
        metric = _RecordingFutureMetric()
        pipeline, _factory, _dataset = self._pipeline(TrackId.B_CAMERA, metric)
        outcome = pipeline.run_sample("sample_00000")

        self.assertEqual(len(metric.calls), 1)
        observed = metric.calls[0]
        self.assertEqual(observed["prediction_count"], 16)
        self.assertEqual(observed["ground_truth_count"], 16)
        self.assertEqual(observed["source_frame_count"], 17)
        self.assertEqual(observed["source_start_index"], 1)
        np.testing.assert_array_equal(
            observed["first_future"], outcome.rollout.frames[1]
        )

    def test_invalid_metric_status_fails_closed_before_bundle(self):
        pipeline, _factory, _dataset = self._pipeline(
            TrackId.B_CAMERA, _InvalidMetric()
        )
        with self.assertRaises(InvalidMetricResultError) as caught:
            pipeline.run_sample("sample_00000")
        self.assertEqual(caught.exception.results[0].reason, "fixture_invalid_status")

    def test_legacy_metric_submission_and_new_bundle_parity_without_torch(self):
        metric = legacy_psnr_metric()
        pipeline, _factory, dataset = self._pipeline(TrackId.B_CAMERA, metric)
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in {"torch", "transformers"}:
                raise AssertionError(f"forbidden M7 dependency import: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            outcome = pipeline.run_sample("sample_00001", generation_seed=303)

        generated = np.asarray(outcome.rollout.frames)
        ground_truth = np.asarray(dataset.ground_truth("sample_00001").rollout)
        before = hashlib.sha256(generated.tobytes()).hexdigest()
        self.assertEqual(
            legacy_submission_contract.check_same_shape(
                generated, ground_truth, what="video", key="sample_00001"
            ),
            [],
        )
        self.assertEqual(
            legacy_submission_contract.check_same_dtype(
                generated, ground_truth, key="sample_00001"
            ),
            [],
        )
        legacy_generated, legacy_gt = legacy_submission_contract.strip_condition_frames(
            generated, ground_truth
        )
        self.assertEqual(len(legacy_generated), 16)
        self.assertEqual(len(legacy_gt), 16)

        from evaluation._legacy import load_legacy_metrics

        direct = float(
            load_legacy_metrics().compute_video_psnr(legacy_generated, legacy_gt)
        )
        self.assertAlmostEqual(outcome.metric_results[0].value, direct, places=12)
        self.assertAlmostEqual(
            outcome.result_bundle.metric_results[0].value, direct, places=12
        )
        restored = ResultBundle.from_document(outcome.result_bundle.to_document())
        self.assertAlmostEqual(restored.metric_results[0].value, direct, places=12)
        after = hashlib.sha256(generated.tobytes()).hexdigest()
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
