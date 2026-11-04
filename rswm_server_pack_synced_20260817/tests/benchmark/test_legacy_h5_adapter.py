from __future__ import annotations

import json
import unittest
from pathlib import Path

from benchmark.core.errors import DatasetContractError
from benchmark.core.identity import sha256_file
from benchmark.datasets import LegacyH5Adapter
from tests.benchmark.fixture_factory import FixtureTempDirectory, create_fixture


class LegacyH5AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = FixtureTempDirectory()
        self.paths = create_fixture(Path(self.temp.name))

    def tearDown(self):
        self.temp.cleanup()

    def test_read_only_mapping_and_hash_stability(self):
        before = sha256_file(self.paths.h5)
        with LegacyH5Adapter(
            self.paths.h5,
            self.paths.manifest,
            benchmark_spec=self.paths.spec,
        ) as dataset:
            self.assertEqual(dataset.sample_ids, ("sample_00000", "sample_00001"))
            sample = dataset.benchmark_sample("sample_00000")
            self.assertEqual(sample.horizon, 17)
            self.assertEqual(sample.overlap.level, "L4_overlap_lt10")
            self.assertAlmostEqual(sample.direction.camera_bearing_deg, 0.0)
            self.assertFalse(sample.image.flags.writeable)
            self.assertFalse(sample.trajectory.per_frame_centers.flags.writeable)
            with self.assertRaises(ValueError):
                sample.image[0, 0, 0] = 0
            self.assertEqual(dataset._file.mode, "r")
        self.assertEqual(before, sha256_file(self.paths.h5))

    def test_public_sample_excludes_hidden_artifacts(self):
        with LegacyH5Adapter(
            self.paths.h5,
            self.paths.manifest,
            benchmark_spec=self.paths.spec,
        ) as dataset:
            public = dataset.public_sample("sample_00000")
            self.assertFalse(hasattr(public, "trajectory"))
            self.assertFalse(hasattr(public, "ground_truth"))
            forbidden = {
                "poses",
                "intrinsics",
                "trajectory",
                "full_scene",
                "gt_rollout_checksum",
                "source_reference",
                "source_image",
            }
            self.assertTrue(forbidden.isdisjoint(public.metadata))

    def test_ground_truth_is_organizer_only(self):
        with LegacyH5Adapter(self.paths.h5, self.paths.manifest) as dataset:
            organizer = dataset.organizer_sample("sample_00000")
            self.assertEqual(organizer.ground_truth.rollout.shape, (17, 8, 12, 3))
            self.assertIsNotNone(organizer.ground_truth.full_scene)
            self.assertNotIn("gt_rollout_checksum", organizer.sample.metadata)

    def test_manifest_tampering_is_rejected(self):
        document = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
        document["samples"][0]["endpoint_overlap"] = 0.5
        tampered = Path(self.temp.name) / "tampered_manifest.json"
        tampered.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(DatasetContractError):
            LegacyH5Adapter(self.paths.h5, tampered)

    def test_closed_adapter_rejects_reads(self):
        dataset = LegacyH5Adapter(self.paths.h5, self.paths.manifest)
        dataset.close()
        with self.assertRaises(RuntimeError):
            dataset.benchmark_sample("sample_00000")


if __name__ == "__main__":
    unittest.main()
