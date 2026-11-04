#!/usr/bin/env python3
"""CPU-only RESISC45 decode, preprocessing, taxonomy, and fail-close tests."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lora_remotesense import build_dataset_from_benchmark as builder  # noqa: E402
from lora_remotesense.hf_image_utils import decode_hf_image  # noqa: E402
from server_run import check_resisc45 as checker  # noqa: E402


def _jpeg_bytes(size=(256, 256), color=(42, 96, 150)):
    stream = BytesIO()
    Image.new("RGB", size, color).save(stream, format="JPEG", quality=90)
    return stream.getvalue()


class _LabelFeature:
    def __init__(self, names):
        self.names = list(names)


class _RESISCFixtureDataset:
    """One Hugging Face-style ``{bytes,path}`` record per official class."""

    def __init__(self):
        self.label_names = list(checker.RESISC45_FIXTURE_CLASSES)
        self.features = {"label": _LabelFeature(self.label_names)}
        payload = _jpeg_bytes()
        self.rows = [
            {"image": {"bytes": payload, "path": None}, "label": label}
            for label in range(len(self.label_names))
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if key == "label":
            return [row["label"] for row in self.rows]
        return self.rows[key]


class RESISC45DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = builder._load_benchmark_module()

    def test_unified_decoder_handles_resisc_struct(self):
        decoded = decode_hf_image(
            {"bytes": _jpeg_bytes((256, 256)), "path": None},
            context="RESISC45 fixture",
        )
        try:
            self.assertEqual(decoded.mode, "RGB")
            self.assertEqual(decoded.size, (256, 256))
        finally:
            decoded.close()

    def test_fixture_represents_45_labeled_classes(self):
        dataset = _RESISCFixtureDataset()
        self.assertEqual(len(dataset.features["label"].names), 45)
        self.assertEqual(len(set(dataset["label"])), 45)
        self.assertEqual(dataset[0]["label"], 0)
        self.assertIn("image", dataset[0])

    def _make_generator(self):
        dataset = _RESISCFixtureDataset()
        generator = object.__new__(self.benchmark.SceneGenerator)
        generator.config = SimpleNamespace(
            scene_categories=list(builder.SCENE_CATEGORIES), random_seed=42)
        generator.hf_datasets = {
            scene: {
                "dataset": dataset,
                "feature": "image",
                "name": builder.RESISC45_REPO,
                "split": "train",
            }
            for scene in builder.SCENE_CATEGORIES
        }
        generator._hd_ds_name_map = {
            scene: builder.RESISC45_REPO for scene in builder.SCENE_CATEGORIES
        }
        return generator

    def test_randomized_official_preprocessing_accepts_struct_records(self):
        generator = self._make_generator()
        builder._HD_SOURCE_REGISTRY.clear()
        np.random.seed(20260814)
        for scene in builder.SCENE_CATEGORIES:
            with self.subTest(scene=scene):
                result = builder._hd_load_scene_from_hf(
                    generator, scene, scene_id=0, H=512, W=512)
                self.assertEqual(result.shape, (512, 512, 3))
                self.assertTrue(np.isfinite(result).all())
                self.assertGreaterEqual(float(result.min()), 0.0)
                self.assertLessEqual(float(result.max()), 255.0)
                provenance = builder._HD_SOURCE_REGISTRY[(scene, 0)]
                self.assertEqual(
                    provenance["source_dataset"], builder.RESISC45_REPO)
                self.assertEqual(
                    len(json.loads(provenance["source_tile_indices"])), 4)

    def test_resisc_scene_pools_are_nonempty_and_disjoint(self):
        dataset = _RESISCFixtureDataset()
        pools = {
            scene: set(builder._get_scene_filter_indices(
                dataset, scene, builder.RESISC45_REPO))
            for scene in builder.SCENE_CATEGORIES
        }
        for scene, indices in pools.items():
            self.assertTrue(indices, scene)
        scenes = list(pools)
        for i, first in enumerate(scenes):
            for second in scenes[i + 1:]:
                self.assertFalse(pools[first] & pools[second])

    def test_official_resisc_tier_fails_closed(self):
        class DummySceneGenerator:
            _load_scene_from_hf = lambda *args, **kwargs: None

            def __init__(self, source=None):
                self.config = SimpleNamespace(scene_categories=["general"])
                self.hf_datasets = {}
                if source is not None:
                    self.hf_datasets["general"] = {
                        "dataset": object(), "feature": "image", "name": source}

        module = SimpleNamespace(SceneGenerator=DummySceneGenerator)
        builder._inject_hd_config(module, dataset_tier="resisc45")

        accepted = module.SceneGenerator(builder.RESISC45_REPO)
        self.assertEqual(
            accepted.hf_datasets["general"]["name"], builder.RESISC45_REPO)
        for source in (None, "blanchon/PatternNet", "synthetic"):
            with self.subTest(source=source):
                with self.assertRaises(builder.SceneSelectionError) as caught:
                    module.SceneGenerator(source)
                self.assertIn("official tier=resisc45", str(caught.exception))
                self.assertIn("Refusing", str(caught.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
