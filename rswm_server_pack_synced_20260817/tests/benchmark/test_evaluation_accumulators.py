from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from evaluation import AccumulatorStateError, EvaluationView, MetricScope
from evaluation.temporal import LegacyR3DFVDAccumulator
from evaluation.visual import LegacyFIDAccumulator


def _view(offset: int) -> EvaluationView:
    ground_truth = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    prediction = ground_truth.copy()
    prediction[1:] = offset
    return EvaluationView.create(prediction, ground_truth)


class DatasetAccumulatorTests(unittest.TestCase):
    def test_fid_accumulates_all_future_frames_then_finalizes_once(self):
        metric = LegacyFIDAccumulator()
        self.assertEqual(metric.identity.scope, MetricScope.DATASET)
        with self.assertRaises(TypeError):
            metric.compute(_view(1))
        metric.start()
        for index in range(32):
            metric.update(_view(index + 1))
        self.assertEqual(metric.sample_count, 32)
        self.assertEqual(len(metric._predictions), 96)
        with patch.object(metric, "_compute_value", return_value=7.25) as compute:
            result = metric.finalize()
        compute.assert_called_once_with()
        self.assertTrue(result.valid)
        self.assertEqual(result.N, 32)
        self.assertEqual(result.value, 7.25)
        with self.assertRaises(AccumulatorStateError):
            metric.finalize()

    def test_fvd_accumulates_whole_future_clips_not_sample_scores(self):
        metric = LegacyR3DFVDAccumulator()
        metric.start()
        for index in range(32):
            metric.update(_view(index + 1))
        self.assertEqual(len(metric._predictions), 32)
        self.assertEqual(metric._predictions[0].shape, (3, 8, 8, 3))
        with patch.object(metric, "_compute_value", return_value=11.5):
            result = metric.finalize()
        self.assertTrue(result.valid)
        self.assertEqual(result.N, 32)
        self.assertIn("r3d18_proxy", result.metric_identity.metric_id)
        self.assertNotIn("i3d", result.metric_identity.metric_id)

    def test_distribution_minimum_n_returns_null_without_loading_features(self):
        metric = LegacyFIDAccumulator()
        metric.start()
        metric.update(_view(10))
        with patch.object(metric, "_compute_value") as compute:
            result = metric.finalize()
        compute.assert_not_called()
        self.assertFalse(result.valid)
        self.assertIsNone(result.value)
        self.assertEqual(result.reason, "insufficient_n:1<32")
        self.assertEqual(result.N, 1)

    def test_update_before_start_is_rejected(self):
        with self.assertRaises(AccumulatorStateError):
            LegacyR3DFVDAccumulator().update(_view(1))

    def test_distribution_threshold_cannot_drop_below_protocol_minimum(self):
        with self.assertRaises(ValueError):
            LegacyFIDAccumulator(minimum_n=31)
        with self.assertRaises(ValueError):
            LegacyR3DFVDAccumulator(minimum_n=31)


if __name__ == "__main__":
    unittest.main()
