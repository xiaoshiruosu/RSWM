from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path

from benchmark.core.errors import ContractError
from benchmark.core.types import PromptSchema, TrackId
from benchmark.datasets import LegacyH5Adapter
from benchmark.tasks import (
    T1SpatialRolloutTask,
    T2ArbitraryDirectionTask,
    T3OverlapLadderTask,
    build_task_from_spec,
)
from models import DeterministicMockWorldModel
from tests.benchmark.fixture_factory import FixtureTempDirectory, create_fixture


class _SpyEvaluator:
    def __init__(self):
        self.call = None

    def evaluate(self, **kwargs):
        self.call = kwargs
        return {"delegated": True, "task_id": kwargs["task_id"].value}


class TaskContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = FixtureTempDirectory()
        self.paths = create_fixture(Path(self.temp.name))
        self.dataset = LegacyH5Adapter(
            self.paths.h5,
            self.paths.manifest,
            benchmark_spec=self.paths.spec,
        )
        self.organizer = self.dataset.organizer_sample("sample_00000")
        self.sample = self.organizer.sample
        self.model = DeterministicMockWorldModel()

    def tearDown(self):
        self.dataset.close()
        self.temp.cleanup()

    def test_track_a_hides_camera_and_ground_truth(self):
        task = T1SpatialRolloutTask()
        model_input = task.prepare_input(
            self.sample,
            TrackId.A_TEXT,
            self.model.capabilities,
        )
        condition = model_input.trajectory_condition
        self.assertIsNone(condition.poses)
        self.assertIsNone(condition.intrinsics)
        self.assertFalse(hasattr(model_input, "ground_truth"))
        forbidden = {"trajectory", "poses", "intrinsics", "full_scene", "gt_rollout"}
        self.assertTrue(forbidden.isdisjoint(model_input.metadata.public))

    def test_track_b_exposes_only_camera_attachment(self):
        task = T1SpatialRolloutTask()
        model_input = task.prepare_input(
            self.sample,
            TrackId.B_CAMERA,
            self.model.capabilities,
        )
        condition = model_input.trajectory_condition
        self.assertEqual(condition.poses.shape, (17, 4, 4))
        self.assertEqual(condition.intrinsics.shape, (17, 4))
        self.assertFalse(hasattr(condition, "waypoints"))
        self.assertFalse(hasattr(condition, "per_frame_centers"))

    def test_mock_generation_and_output_contract(self):
        task = T2ArbitraryDirectionTask()
        model_input = task.prepare_input(
            self.sample,
            TrackId.A_TEXT,
            self.model.capabilities,
            generation_seed=123,
        )
        rollout = self.model.generate(
            model_input.condition_image,
            model_input.trajectory_condition,
            model_input.metadata,
        )
        report = task.validate_output(rollout, model_input)
        self.assertTrue(report.valid)
        self.assertEqual(rollout.frames.shape, (17, 8, 12, 3))
        self.assertTrue(report.checks["condition_frame_exact"])
        self.assertEqual(report.checks["future_only_start_index"], 1)

    def test_t3_level_is_enforced(self):
        good = T3OverlapLadderTask(required_level="L4_overlap_lt10")
        good.prepare_input(self.sample, TrackId.B_CAMERA, self.model.capabilities)
        bad = T3OverlapLadderTask(required_level="L1_overlap_80")
        with self.assertRaises(ContractError):
            bad.prepare_input(self.sample, TrackId.B_CAMERA, self.model.capabilities)

    def test_task_registry_consumes_frozen_spec(self):
        task = build_task_from_spec(self.dataset.spec)
        self.assertIsInstance(task, T3OverlapLadderTask)
        self.assertEqual(task.required_level, "L4_overlap_lt10")
        task.prepare_input(self.sample, TrackId.B_CAMERA, self.model.capabilities)

    def test_legacy_prompt_is_not_relabelled_official(self):
        metadata = dict(self.sample.metadata)
        metadata["prompt_schema"] = PromptSchema.LEGACY_NET.value
        legacy = replace(self.sample, metadata=metadata)
        with self.assertRaises(ContractError):
            T1SpatialRolloutTask().prepare_input(
                legacy,
                TrackId.A_TEXT,
                self.model.capabilities,
            )

    def test_evaluate_is_only_a_delegate_contract(self):
        task = T1SpatialRolloutTask()
        model_input = task.prepare_input(
            self.sample,
            TrackId.A_TEXT,
            self.model.capabilities,
        )
        rollout = self.model.generate(
            model_input.condition_image,
            model_input.trajectory_condition,
            model_input.metadata,
        )
        spy = _SpyEvaluator()
        result = task.evaluate(
            rollout,
            self.organizer.ground_truth,
            self.sample,
            TrackId.A_TEXT,
            spy,
            model_input=model_input,
        )
        self.assertTrue(result["delegated"])
        self.assertEqual(spy.call["track_id"], TrackId.A_TEXT)


if __name__ == "__main__":
    unittest.main()
