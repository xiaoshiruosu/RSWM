from __future__ import annotations

from dataclasses import replace
import unittest

from benchmark.core.types import CapabilityPartition, TaskId, TrackId
from baselines import (
    AdmissionState,
    AdmissionStateError,
    BaselineCardV2,
)
from tests.benchmark.baseline_fixture_factory import (
    advance_to_admitted,
    declared_card,
    evidence,
)


class BaselineCardV2Tests(unittest.TestCase):
    def test_state_machine_rejects_skipped_stage(self):
        card = declared_card()
        with self.assertRaises(AdmissionStateError):
            card.advance(evidence(AdmissionState.CHECKPOINT_VERIFIED))

    def test_sequential_progression_reaches_admitted(self):
        card = advance_to_admitted(declared_card())
        self.assertEqual(card.state, AdmissionState.ADMITTED)
        self.assertEqual(len(card.state_history), 5)
        card.assert_admitted_for(
            task_id=TaskId.T1_SPATIAL_ROLLOUT,
            track_id=TrackId.B_CAMERA,
            partition=CapabilityPartition.NATIVE,
        )
        self.assertEqual(card, BaselineCardV2.from_document(card.to_document()))
        with self.assertRaises(AdmissionStateError):
            card.advance(evidence(AdmissionState.ADMITTED))

    def test_history_tampering_is_rejected(self):
        card = advance_to_admitted(declared_card())
        document = card.to_document()
        document["state_history"].pop(2)
        document["overlay_checksum"] = None
        with self.assertRaises(AdmissionStateError):
            BaselineCardV2.from_document(document)

    def test_future_gt_can_be_evaluated_but_not_admitted(self):
        card = declared_card()
        card = replace(
            card,
            capabilities=replace(card.capabilities, uses_future_gt=True),
            overlay_checksum=None,
        ).with_checksum()
        for state in (
            AdmissionState.ADAPTER_AVAILABLE,
            AdmissionState.CHECKPOINT_VERIFIED,
            AdmissionState.RUNNABLE,
            AdmissionState.EVALUATED,
        ):
            card = card.advance(evidence(state))
        self.assertEqual(card.state, AdmissionState.EVALUATED)
        with self.assertRaises(AdmissionStateError):
            card.advance(evidence(AdmissionState.ADMITTED))

    def test_declared_card_is_not_admitted(self):
        card = declared_card()
        with self.assertRaises(AdmissionStateError):
            card.assert_admitted_for(
                task_id=TaskId.T1_SPATIAL_ROLLOUT,
                track_id=TrackId.B_CAMERA,
                partition=CapabilityPartition.NATIVE,
            )


if __name__ == "__main__":
    unittest.main()
