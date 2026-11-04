from __future__ import annotations

from dataclasses import replace
import unittest

from benchmark.core.types import CapabilityPartition, TaskId, TrackId
from baselines import (
    AdaptationType,
    BaselineCapabilityError,
    CameraSupport,
    InputType,
    TrajectorySupport,
)
from tests.benchmark.baseline_fixture_factory import capabilities


class BaselineCapabilityTests(unittest.TestCase):
    def test_native_camera_entry_is_eligible_only_in_native_partition(self):
        caps = capabilities(CapabilityPartition.NATIVE)
        caps.assert_eligible(
            task_id=TaskId.T1_SPATIAL_ROLLOUT,
            track_id=TrackId.B_CAMERA,
            partition=CapabilityPartition.NATIVE,
        )
        with self.assertRaises(BaselineCapabilityError):
            caps.assert_eligible(
                task_id=TaskId.T1_SPATIAL_ROLLOUT,
                track_id=TrackId.B_CAMERA,
                partition=CapabilityPartition.ADAPTED,
            )

    def test_no_camera_or_intrinsics_only_cannot_declare_b_camera(self):
        caps = capabilities(CapabilityPartition.NATIVE)
        for camera in (CameraSupport.NONE, CameraSupport.INTRINSICS):
            with self.subTest(camera=camera):
                with self.assertRaises(BaselineCapabilityError):
                    replace(caps, camera_support=camera)

    def test_b_camera_requires_full_path(self):
        caps = capabilities(CapabilityPartition.NATIVE)
        with self.assertRaises(BaselineCapabilityError):
            replace(caps, trajectory_support=TrajectorySupport.NET_DIRECTION)

    def test_adapted_entry_does_not_fall_back_to_native(self):
        caps = capabilities(CapabilityPartition.ADAPTED)
        caps.assert_eligible(
            task_id=TaskId.T2_ARBITRARY_DIRECTION,
            track_id=TrackId.B_CAMERA,
            partition=CapabilityPartition.ADAPTED,
        )
        with self.assertRaises(BaselineCapabilityError):
            caps.assert_eligible(
                task_id=TaskId.T2_ARBITRARY_DIRECTION,
                track_id=TrackId.B_CAMERA,
                partition=CapabilityPartition.NATIVE,
            )

    def test_multiview_is_separate_from_single_view_partitions(self):
        caps = capabilities(CapabilityPartition.MULTIVIEW)
        caps.assert_eligible(
            task_id=TaskId.T1_SPATIAL_ROLLOUT,
            track_id=TrackId.C_MULTIVIEW,
            partition=CapabilityPartition.MULTIVIEW,
        )
        with self.assertRaises(BaselineCapabilityError):
            caps.assert_eligible(
                task_id=TaskId.T1_SPATIAL_ROLLOUT,
                track_id=TrackId.C_MULTIVIEW,
                partition=CapabilityPartition.NATIVE,
            )
        with self.assertRaises(BaselineCapabilityError):
            replace(caps, input_type=InputType.SINGLE_IMAGE)

    def test_unknown_capability_is_not_promoted(self):
        caps = capabilities(CapabilityPartition.NATIVE)
        unresolved = replace(
            caps,
            tracks=(),
            input_type=InputType.UNKNOWN,
            trajectory_support=TrajectorySupport.UNKNOWN,
            camera_support=CameraSupport.UNKNOWN,
            adaptation_type=AdaptationType.UNKNOWN,
        )
        with self.assertRaises(BaselineCapabilityError):
            unresolved.assert_partition(CapabilityPartition.NATIVE)


if __name__ == "__main__":
    unittest.main()
