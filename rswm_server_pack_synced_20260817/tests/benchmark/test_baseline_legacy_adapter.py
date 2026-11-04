from __future__ import annotations

import unittest

from benchmark.core.identity import sha256_file
from benchmark.core.types import CapabilityPartition
from baselines import (
    AdmissionState,
    CameraSupport,
    InputType,
    LegacyBaselineAdapter,
    TrajectorySupport,
    UNKNOWN,
)
from baselines.legacy import load_legacy_cards_module


class LegacyBaselineAdapterTests(unittest.TestCase):
    def test_all_v1_cards_remain_loadable_and_declared(self):
        names = LegacyBaselineAdapter.list_names()
        self.assertEqual(names, tuple(load_legacy_cards_module().list_cards()))
        self.assertGreater(len(names), 0)
        for name in names:
            with self.subTest(name=name):
                original = LegacyBaselineAdapter(name).load_v1()
                overlay = LegacyBaselineAdapter(name).overlay()
                self.assertEqual(original["name"], name)
                self.assertEqual(overlay.state, AdmissionState.DECLARED)
                self.assertEqual(overlay.baseline_identity.baseline_id, name)
                self.assertEqual(overlay.legacy_card.name, name)

    def test_v1_card_hash_is_referenced_without_rewrite(self):
        adapter = LegacyBaselineAdapter("persistence")
        path = (
            __import__("pathlib").Path(__file__).resolve().parents[2]
            / "server_run"
            / "baselines"
            / "cards"
            / "persistence.json"
        )
        before = sha256_file(path)
        overlay = adapter.overlay()
        after = sha256_file(path)
        self.assertEqual(before, after)
        self.assertEqual(overlay.legacy_card.sha256, before)

    def test_unknown_fields_are_not_guessed_for_single_view_card(self):
        overlay = LegacyBaselineAdapter("frozen_i2v").overlay()
        caps = overlay.capabilities
        self.assertEqual(caps.input_type, InputType.UNKNOWN)
        self.assertEqual(caps.camera_support, CameraSupport.UNKNOWN)
        self.assertEqual(caps.trajectory_support, TrajectorySupport.UNKNOWN)
        self.assertEqual(caps.checkpoint_identity.repository, UNKNOWN)
        self.assertEqual(caps.license.code, UNKNOWN)
        self.assertEqual(caps.tasks, ())
        self.assertEqual(caps.tracks, ())
        self.assertIsNone(overlay.capability_partition)

    def test_only_structured_multiview_fact_is_mapped(self):
        overlay = LegacyBaselineAdapter("mvsplat360").overlay()
        self.assertEqual(overlay.capabilities.input_type, InputType.MULTIVIEW)
        self.assertGreater(overlay.capabilities.condition_views, 1)
        self.assertEqual(
            overlay.capability_partition, CapabilityPartition.MULTIVIEW
        )
        self.assertEqual(overlay.capabilities.tasks, ())
        self.assertEqual(overlay.capabilities.tracks, ())
        self.assertEqual(overlay.state, AdmissionState.DECLARED)

    def test_explicit_ignored_trajectory_maps_to_none_only(self):
        overlay = LegacyBaselineAdapter("persistence").overlay()
        self.assertEqual(overlay.capabilities.camera_support, CameraSupport.NONE)
        self.assertEqual(
            overlay.capabilities.trajectory_support, TrajectorySupport.NONE
        )


if __name__ == "__main__":
    unittest.main()
