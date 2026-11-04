from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from evaluation import EvaluationView
from evaluation._legacy import (
    load_legacy_metrics,
    load_legacy_trajectory_metrics,
)
from evaluation.semantic import UnavailableSemanticMetric
from evaluation.spatial import LegacyTrajectoryMetric
from evaluation.visual import legacy_psnr_metric


class LegacyMetricParityTests(unittest.TestCase):
    def test_visual_wrapper_matches_existing_future_frame_math(self):
        ground_truth = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        prediction = ground_truth.copy()
        prediction[1] = 70
        prediction[2] = 30
        prediction[3] = 10
        legacy = load_legacy_metrics().compute_video_psnr(
            prediction[1:], ground_truth[1:]
        )
        wrapped = legacy_psnr_metric().compute(
            EvaluationView.create(prediction, ground_truth)
        )
        self.assertTrue(wrapped.valid)
        self.assertEqual(wrapped.value, legacy)

    def test_trajectory_wrapper_matches_score_trajectory_scalar(self):
        trajectory = load_legacy_trajectory_metrics()
        prediction = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        ground_truth = prediction.copy()
        flows = []
        for _ in range(3):
            flow = np.zeros((4, 4, 2), dtype=np.float32)
            flow[..., 0] = -2.0
            flows.append(flow)
        metadata = {
            "waypoints": np.asarray([[0.0, 0.0], [0.0, 6.0]]),
            "segment_angles": np.asarray([0.0]),
            "min_mag": 0.1,
        }
        with patch.object(trajectory, "_compute_flows", return_value=flows):
            direct = trajectory.score_trajectory(
                prediction,
                metadata["waypoints"],
                metadata["segment_angles"],
                num_frames=4,
                min_mag=0.1,
                path_estimator="flow",
            )
            wrapped = LegacyTrajectoryMetric(
                "net_direction_error", path_estimator="flow"
            ).compute(
                EvaluationView.create(prediction, ground_truth),
                metadata,
            )
        self.assertTrue(direct["valid"])
        self.assertTrue(wrapped.valid)
        self.assertEqual(wrapped.value, direct["net_direction_error"])

    def test_semantic_placeholder_is_explicitly_unavailable(self):
        frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
        result = UnavailableSemanticMetric().compute(
            EvaluationView.create(frames, frames)
        )
        self.assertFalse(result.valid)
        self.assertIsNone(result.value)
        self.assertIn("NOT_AVAILABLE", result.reason)
        self.assertFalse(
            result.metric_identity.preprocessing["semantic_gate_reused"]
        )

    def test_trajectory_wrapper_rejects_frame_count_truncation(self):
        frames = np.zeros((4, 4, 4, 3), dtype=np.uint8)
        result = LegacyTrajectoryMetric(
            "net_direction_error", path_estimator="flow"
        ).compute(
            EvaluationView.create(frames, frames),
            {
                "waypoints": np.asarray([[0.0, 0.0], [0.0, 6.0]]),
                "segment_angles": np.asarray([0.0]),
                "num_frames": 3,
            },
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "num_frames_mismatch:3!=4")


if __name__ == "__main__":
    unittest.main()
