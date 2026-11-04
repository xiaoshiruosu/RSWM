#!/usr/bin/env python3
"""CPU-only WHU-RS19 loader and canonical preprocessing regression tests."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_remotesense import build_dataset_from_benchmark as builder  # noqa: E402
from lora_remotesense.hf_image_utils import (  # noqa: E402
    WHURS19_LABEL_NAMES,
    WHURS19_REPO,
    ImageFieldDecodeError,
    decode_hf_image,
    find_hf_snapshot_parquet,
)
from server_run import check_resisc45 as checker  # noqa: E402


def _jpeg_bytes(size, color=(32, 96, 160)):
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format="JPEG", quality=90)
    return stream.getvalue()


class _LabelFeature:
    names = list(WHURS19_LABEL_NAMES)


class _SingleImageDataset:
    def __init__(self, field, *, label_names=None, label=0):
        self.field = field
        self.label = int(label)
        feature = _LabelFeature()
        if label_names is not None:
            feature.names = list(label_names)
        self.features = {"label": feature}

    def __len__(self):
        return 1

    def __getitem__(self, key):
        if key == "label":
            return [self.label]
        if key != 0:
            raise IndexError(key)
        return {"image": self.field, "label": self.label}


class WHURS19DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = builder._load_benchmark_module()

    def test_decode_pil_rgb_detached(self):
        source = Image.new("L", (13, 7), 127)
        decoded = decode_hf_image(source, context="pil-fixture")
        source.close()
        self.assertIsInstance(decoded, Image.Image)
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (13, 7))
        self.assertEqual(decoded.getpixel((0, 0)), (127, 127, 127))
        decoded.close()

    def test_decode_struct_bytes(self):
        decoded = decode_hf_image(
            {"bytes": _jpeg_bytes((17, 11)), "path": None},
            context="bytes-fixture")
        self.assertIsInstance(decoded, Image.Image)
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (17, 11))
        decoded.close()

    def test_decode_struct_path(self):
        path = ROOT / "tests" / "fixtures" / "whurs19_path_fixture.ppm"
        decoded = decode_hf_image(
            {"bytes": None, "path": path}, context="path-fixture")
        self.assertEqual(decoded.mode, "RGB")
        self.assertEqual(decoded.size, (2, 1))
        decoded.close()

    def test_decode_rejects_illegal_fields(self):
        for value in ({}, None, 123):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)) as caught:
                    decode_hf_image(value, context="bad-fixture")
                self.assertTrue(
                    "Unsupported image field type" in str(caught.exception)
                    or "neither non-null" in str(caught.exception))
        with self.assertRaises(ImageFieldDecodeError):
            decode_hf_image({"bytes": b"not-an-image", "path": None})

    def _make_generator(self, field, *, repo=WHURS19_REPO,
                        label_names=WHURS19_LABEL_NAMES):
        dataset = _SingleImageDataset(field, label_names=label_names)
        scene_generator = object.__new__(self.benchmark.SceneGenerator)
        scene_generator.config = SimpleNamespace(
            scene_categories=["general"], random_seed=42)
        scene_generator.hf_datasets = {
            "general": {
                "dataset": dataset,
                "feature": "image",
                "name": repo,
                "split": "train",
            }
        }
        scene_generator._hd_ds_name_map = {"general": repo}
        return scene_generator

    def _run_canonical_pipeline(self, size):
        generator = self._make_generator({
            "bytes": _jpeg_bytes(size), "path": None})
        builder._HD_SOURCE_REGISTRY.clear()
        np.random.seed(20260814)
        scene = builder._hd_load_scene_from_hf(
            generator, "general", scene_id=0, H=512, W=512)
        self.assertEqual(scene.shape, (512, 512, 3))

        cfg = self.benchmark.MultiDirectionConfig()
        cfg.full_scene_size = (512, 512)
        cfg.window_size = (128, 222)
        cfg.steps_per_segment = 9
        cfg.min_segments = 2
        cfg.max_segments = 4
        extractor = self.benchmark.MultiDirectionExtractor(cfg)
        path_info = {
            "waypoints": [(160, 160), (160, 300), (320, 300)],
            "segment_angles": [0.0, 270.0],
            "num_segments": 2,
            "steps_per_segment": 9,
            "start_position": (160, 160),
            "end_position": (320, 300),
        }
        result = extractor.extract_multi_direction_sequence(scene, path_info)
        self.assertEqual(result["sliding_frames"].shape, (17, 128, 222, 3))
        self.assertTrue(np.all(result["trajectory"][:, 0] >= 0))
        self.assertTrue(np.all(result["trajectory"][:, 0] < 512))
        self.assertTrue(np.all(result["trajectory"][:, 1] >= 0))
        self.assertTrue(np.all(result["trajectory"][:, 1] < 512))
        return builder._HD_SOURCE_REGISTRY[("general", 0)]

    def test_600_and_all_verified_non600_sizes_use_same_pipeline(self):
        sizes = ((600, 600), (958, 612), (726, 456),
                 (600, 500), (711, 718))
        for width, height in sizes:
            with self.subTest(size=(width, height)):
                record = self._run_canonical_pipeline((width, height))
                self.assertEqual(json.loads(record["source_original_hw"]),
                                 [height, width])
                normalized_h, normalized_w = json.loads(
                    record["source_normalized_hw"])
                self.assertGreaterEqual(normalized_h, 512)
                self.assertGreaterEqual(normalized_w, 512)
                expected = ("isotropic_resize_to_cover_scene"
                            if height < 512 or width < 512 else "identity")
                self.assertEqual(record["source_preprocess"], expected)

    def test_aid_600_pil_large_crop_regression(self):
        yy, xx = np.indices((600, 600))
        pixels = np.stack((yy % 256, xx % 256, (yy + xx) % 256), axis=-1).astype(np.uint8)
        source = Image.fromarray(pixels, mode="RGB")
        generator = self._make_generator(
            source, repo="blanchon/AID", label_names=WHURS19_LABEL_NAMES)

        np.random.seed(314159)
        y0 = int(np.random.randint(0, 600 - 512 + 1))
        x0 = int(np.random.randint(0, 600 - 512 + 1))
        expected = pixels[y0:y0 + 512, x0:x0 + 512].astype(np.float32)

        builder._HD_SOURCE_REGISTRY.clear()
        np.random.seed(314159)
        actual = builder._hd_load_scene_from_hf(
            generator, "general", scene_id=0, H=512, W=512)
        np.testing.assert_array_equal(actual, expected)
        self.assertNotIn("source_preprocess",
                         builder._HD_SOURCE_REGISTRY[("general", 0)])
        source.close()

    def test_official_whu_tier_fails_closed(self):
        class DummySceneGenerator:
            _load_scene_from_hf = lambda *args, **kwargs: None

            def __init__(self, source=None):
                self.config = SimpleNamespace(scene_categories=["general"])
                self.hf_datasets = {}
                if source is not None:
                    self.hf_datasets["general"] = {
                        "dataset": object(), "feature": "image", "name": source}

        module = SimpleNamespace(SceneGenerator=DummySceneGenerator)
        builder._inject_hd_config(module, dataset_tier="whurs19")
        for source in (None, "blanchon/RESISC45", "blanchon/PatternNet"):
            with self.subTest(source=source):
                with self.assertRaises(builder.SceneSelectionError) as caught:
                    module.SceneGenerator(source)
                self.assertIn("Refusing synthetic/RESISC45/PatternNet fallback",
                              str(caught.exception))

    def test_plain_int64_labels_use_official_whu_mapping(self):
        dataset = _SingleImageDataset(
            {"bytes": _jpeg_bytes((600, 600)), "path": None}, label=6)
        dataset.features = {"label": object()}  # Parquet Value(int64), no names
        indices, classes, _paths = builder._scene_pool_with_meta(
            dataset, "general", WHURS19_REPO)
        self.assertEqual(indices, [0])
        self.assertEqual(classes[0], "football field")

    def test_whu_integrity_checker_exact_1005_fingerprint(self):
        sizes = ([(600, 600)] * 1001
                 + [(958, 612), (726, 456), (600, 500), (711, 718)])

        class FakeDataset:
            features = {"label": object()}

            def __len__(self):
                return 1005

            def __getitem__(self, index):
                return {"image": index, "label": index % 19}

        class FakeImage:
            mode = "RGB"

            def __init__(self, size):
                self.size = size

            def close(self):
                pass

        report = checker.Report()
        with patch.object(
                checker, "decode_hf_image",
                side_effect=lambda field, context="": FakeImage(sizes[field])):
            names = checker._check_whurs19_integrity(report, FakeDataset())
        self.assertEqual(report.n_fail, 0)
        self.assertEqual(tuple(names), WHURS19_LABEL_NAMES)
        integrity = report.data["whurs19_integrity"]
        self.assertEqual(integrity["decode_success"], 1005)
        self.assertEqual(integrity["unique_labels"], 19)
        self.assertEqual(integrity["image_modes"], {"RGB": 1005})

    def test_snapshot_parquet_resolver(self):
        expected = Path(__file__).resolve()
        self.assertEqual(
            find_hf_snapshot_parquet(
                WHURS19_REPO,
                snapshot_root=ROOT / "tests",
                pattern="test_whurs19_data.py"),
            expected)

    def test_builder_prefers_visible_whu_snapshot_parquet(self):
        calls = []

        def fake_load_dataset(path, *args, **kwargs):
            calls.append((path, args, kwargs))
            return "dataset"

        module = SimpleNamespace(hf_load_dataset=fake_load_dataset)
        parquet = Path("/cache/WHU-RS19/data/train-fixture.parquet")
        with patch.object(
                builder, "find_hf_snapshot_parquet", return_value=parquet):
            self.assertEqual(builder._install_whurs19_snapshot_loader(module), parquet)
        self.assertEqual(
            module.hf_load_dataset(
                WHURS19_REPO, split="train", trust_remote_code=True),
            "dataset")
        self.assertEqual(calls[-1][0], "parquet")
        self.assertEqual(calls[-1][2]["data_files"], str(parquet))
        self.assertNotIn("trust_remote_code", calls[-1][2])

        module.hf_load_dataset("blanchon/AID", split="train")
        self.assertEqual(calls[-1][0], "blanchon/AID")


if __name__ == "__main__":
    unittest.main(verbosity=2)
