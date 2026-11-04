from __future__ import annotations

import unittest

import numpy as np

from evaluation import EvaluationView, FutureOnlyOwner, FutureOnlyOwnershipError
from evaluation.visual import legacy_psnr_metric


def _rollouts():
    ground_truth = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    prediction = ground_truth.copy()
    prediction[1] = 80
    prediction[2] = 20
    prediction[3] = 30
    return prediction, ground_truth


class EvaluationViewTests(unittest.TestCase):
    def test_t0_pollution_does_not_change_future_metric(self):
        prediction, ground_truth = _rollouts()
        metric = legacy_psnr_metric()
        clean = metric.compute(EvaluationView.create(prediction, ground_truth))
        polluted = prediction.copy()
        polluted[0] = 255
        changed_t0 = metric.compute(EvaluationView.create(polluted, ground_truth))
        self.assertTrue(clean.valid)
        self.assertTrue(changed_t0.valid)
        self.assertEqual(clean.value, changed_t0.value)

    def test_deleting_t1_changes_metric_and_proves_no_double_crop(self):
        prediction, ground_truth = _rollouts()
        metric = legacy_psnr_metric()
        complete = metric.compute(EvaluationView.create(prediction, ground_truth))
        without_t1 = metric.compute(
            EvaluationView.create(
                np.delete(prediction, 1, axis=0),
                np.delete(ground_truth, 1, axis=0),
            )
        )
        self.assertTrue(complete.valid)
        self.assertTrue(without_t1.valid)
        self.assertNotEqual(complete.value, without_t1.value)

    def test_future_view_removes_exactly_one_condition_frame(self):
        prediction, ground_truth = _rollouts()
        evaluation = EvaluationView.create(prediction, ground_truth)
        pred_view, gt_view = evaluation.frames_for(legacy_psnr_metric().identity)
        self.assertEqual(len(pred_view.frames), len(prediction) - 1)
        self.assertEqual(pred_view.source_start_index, 1)
        self.assertEqual(gt_view.source_start_index, 1)
        self.assertFalse(hasattr(pred_view, "future_only"))

    def test_owner_mismatch_fails_closed(self):
        prediction, ground_truth = _rollouts()
        evaluation = EvaluationView.create(
            prediction,
            ground_truth,
            future_only_owner=FutureOnlyOwner.LEGACY_WRAPPER,
        )
        with self.assertRaises(FutureOnlyOwnershipError):
            legacy_psnr_metric().compute(evaluation)


if __name__ == "__main__":
    unittest.main()
