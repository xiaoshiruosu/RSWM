from __future__ import annotations

import unittest

from benchmark.trajectory.conventions import (
    build_direction_target,
    camera_bearing_deg,
    content_bearing_deg,
    direction_bin,
    overlap_level,
)


class TrajectoryConventionTests(unittest.TestCase):
    def test_cardinal_and_content_directions(self):
        cases = {
            "E": ((0, 1), 0.0),
            "N": ((-1, 0), 90.0),
            "W": ((0, -1), 180.0),
            "S": ((1, 0), 270.0),
        }
        for label, (delta, expected) in cases.items():
            with self.subTest(label=label):
                camera = camera_bearing_deg(*delta)
                self.assertAlmostEqual(camera, expected)
                self.assertAlmostEqual(content_bearing_deg(camera), (expected + 180) % 360)
                self.assertEqual(direction_bin(camera), label)

    def test_eight_way_half_open_boundaries(self):
        self.assertEqual(direction_bin(337.5), "E")
        self.assertEqual(direction_bin(22.499999), "E")
        self.assertEqual(direction_bin(22.5), "NE")
        self.assertEqual(direction_bin(67.5), "N")
        self.assertEqual(direction_bin(292.5), "SE")

    def test_overlap_ladder_boundaries_and_gaps(self):
        self.assertEqual(overlap_level(0.00), "L4_overlap_lt10")
        self.assertIsNone(overlap_level(0.10))
        self.assertEqual(overlap_level(0.15), "L3_overlap_20")
        self.assertIsNone(overlap_level(0.25))
        self.assertEqual(overlap_level(0.40), "L2_overlap_50")
        self.assertIsNone(overlap_level(0.60))
        self.assertEqual(overlap_level(0.70), "L1_overlap_80")
        self.assertIsNone(overlap_level(0.90))

    def test_net_ratio_threshold_is_inclusive(self):
        exactly = build_direction_target(((0, 0), (0, 3), (0, 1)))
        below = build_direction_target(((0, 0), (0, 3), (0, 0.99)))
        self.assertAlmostEqual(exactly.net_displacement_ratio, 0.2)
        self.assertTrue(exactly.net_direction_well_posed)
        self.assertLess(below.net_displacement_ratio, 0.2)
        self.assertFalse(below.net_direction_well_posed)


if __name__ == "__main__":
    unittest.main()
