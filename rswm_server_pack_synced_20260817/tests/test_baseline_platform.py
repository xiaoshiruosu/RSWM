#!/usr/bin/env python3
"""CPU-only end-to-end tests for the RS-Roll baseline platform."""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import unittest
import uuid
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
BASELINES = ROOT / "server_run" / "baselines"
sys.path.insert(0, str(BASELINES))

from cards import list_cards, load_card
from check_dataset_leakage import audit as leakage_audit
from protocol import (MANIFEST_SCHEMA, ProtocolError, normalize_video,
                      validate_manifest_document)


class BaselinePlatformTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Avoid tempfile's Windows chmod/ACL normalization: native HDF5 can be denied
        # access to those directories even when Python itself created them.
        cls.work = ROOT / "tests" / f"_baseline_platform_{uuid.uuid4().hex}"
        cls.work.mkdir()
        cls.h5 = cls.work / "synthetic.h5"
        cls.split = cls.work / "split.json"
        cls.manifest = cls.work / "manifest.json"
        cls._write_fixture()
        run = cls._run([
            sys.executable, str(BASELINES / "build_benchmark_manifest.py"),
            "--h5", str(cls.h5), "--split", str(cls.split),
            "--split-name", "test", "--source-dataset", "AID",
            "--output", str(cls.manifest),
        ])
        if run.returncode != 0:
            raise RuntimeError(run.stdout + run.stderr)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    @classmethod
    def _run(cls, command, *, env=None):
        merged = dict(os.environ)
        merged["CUDA_VISIBLE_DEVICES"] = ""
        if env:
            merged.update(env)
        return subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                              env=merged)

    @classmethod
    def _write_fixture(cls):
        rng = np.random.default_rng(7)
        frame_counts = (17, 25, 33, 17)
        condition_sizes = ((8, 12), (10, 14), (12, 16), (9, 13))
        endpoint_dx = (4.0, 9.0, 15.0, 22.0)
        with h5py.File(cls.h5, "w") as file:
            meta = file.create_group("metadata")
            meta.attrs["pose_window_size"] = [8, 12]
            meta.attrs["full_scene_size"] = [64, 64]
            data = file.create_group("data")
            for index, (frames, image_hw, dx) in enumerate(
                    zip(frame_counts, condition_sizes, endpoint_dx)):
                key = f"sample_{index:05d}"
                group = data.create_group(key)
                base = rng.integers(0, 180, size=(8, 12, 3), dtype=np.uint8)
                condition = cv2.resize(base, (image_hw[1], image_hw[0]),
                                       interpolation=cv2.INTER_LINEAR)
                video = np.stack([
                    np.clip(base.astype(np.int16) + t * 2, 0, 255).astype(np.uint8)
                    for t in range(frames)
                ])
                full_scene = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
                trajectory = np.stack([
                    np.full(frames, 32.0),
                    np.linspace(24.0, 24.0 + dx, frames),
                ], axis=1)
                group.create_dataset("image", data=condition)
                group.create_dataset("video_frames", data=video)
                group.create_dataset("full_scene", data=full_scene)
                group.create_dataset("trajectory", data=trajectory)
                group.create_dataset("waypoints", data=trajectory[[0, -1]])
                group.create_dataset("segment_angles", data=np.array([0.0]))
                group.attrs["prompt"] = (
                    "A satellite video following a 1-segment trajectory with net "
                    "displacement toward the east.")
                group.attrs["num_frames"] = frames
                group.attrs["trajectory_type"] = "straight"
                group.attrs["source_image"] = f"synthetic/AID/{key}.png"
                group.attrs["category"] = "synthetic"
        cls.split.write_text(json.dumps({"test_indices": [0, 1, 2, 3]}),
                             encoding="utf-8")

    def test_manifest_records_required_protocol(self):
        doc = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], MANIFEST_SCHEMA)
        self.assertEqual(doc["frame_counts"], [17, 25, 33])
        self.assertEqual(doc["resolutions"], [[8, 12]])
        self.assertEqual(len(doc["condition_resolutions"]), 4)
        self.assertEqual(len(doc["samples"]), 4)
        self.assertEqual(doc["difficulty_bins"]["status"],
                         "FROZEN_FROM_MANIFEST_DISTRIBUTION")
        for sample in doc["samples"]:
            self.assertEqual(len(sample["overlap_profile"]), sample["frame_count"])
            self.assertEqual(sample["overlap_profile"]["frame_0"], 1.0)
            self.assertIn(sample["difficulty_bin"],
                          ("Dense", "Medium", "Sparse", "Extreme"))
            self.assertEqual(len(sample["checksum"]), 64)
            self.assertEqual(sample["rollout_size"], [8, 12])
        validate_manifest_document(doc)

    def test_manifest_malformed_and_empty_fail(self):
        doc = json.loads(self.manifest.read_text(encoding="utf-8"))
        broken = dict(doc)
        broken["samples"] = []
        with self.assertRaises(ProtocolError):
            validate_manifest_document(broken, require_files=False)
        broken = dict(doc)
        broken.pop("split_checksum")
        with self.assertRaises(ProtocolError):
            validate_manifest_document(broken, require_files=False)
        tampered = json.loads(json.dumps(doc))
        tampered["samples"][0]["source_image"] = "silently/changed.png"
        with self.assertRaisesRegex(ProtocolError, "manifest_checksum"):
            validate_manifest_document(tampered, require_files=False)

    def test_diagnostic_manifest_flags_hash_skip_and_subset(self):
        output = self.work / "diagnostic_manifest.json"
        run = self._run([
            sys.executable, str(BASELINES / "build_benchmark_manifest.py"),
            "--h5", str(self.h5), "--split", str(self.split),
            "--split-name", "test", "--source-dataset", "AID",
            "--skip-h5-hash", "--max-samples", "2", "--output", str(output),
        ])
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        doc = json.loads(output.read_text(encoding="utf-8"))
        self.assertFalse(doc["headline_eligible"])
        self.assertIsNone(doc["dataset_checksum"])
        self.assertEqual(doc["sample_selection"], "diagnostic_prefix_subset")
        validate_manifest_document(doc, require_files=True)

    def test_normalize_rejects_single_image_and_empty_rollout(self):
        with self.assertRaises(ProtocolError):
            normalize_video(np.zeros((8, 12, 3), np.uint8))
        with self.assertRaises(ProtocolError):
            normalize_video(np.zeros((0, 8, 12, 3), np.uint8))
        value = normalize_video(np.zeros((17, 3, 8, 12), np.uint8), expected_t=17)
        self.assertEqual(value.shape, (17, 8, 12, 3))
        unit = np.zeros((2, 2, 2, 3), np.float32)
        unit[1] = 1.0
        mapped_unit = normalize_video(unit)
        self.assertEqual(int(mapped_unit[0, 0, 0, 0]), 0)
        self.assertEqual(int(mapped_unit[1, 0, 0, 0]), 255)
        signed = np.full((2, 2, 2, 3), -1.0, np.float32)
        signed[1] = 1.0
        mapped_signed = normalize_video(signed)
        self.assertEqual(int(mapped_signed[0, 0, 0, 0]), 0)
        self.assertEqual(int(mapped_signed[1, 0, 0, 0]), 255)

    def test_cards_cover_registry(self):
        run = self._run([sys.executable, str(BASELINES / "run_baseline.py"), "--list"])
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        methods = {line.split()[0] for line in run.stdout.splitlines() if line.strip()}
        self.assertEqual(methods, set(list_cards()))
        for method in methods:
            card = load_card(method)
            self.assertIn("native_output", card)
            self.assertIsInstance(card["uses_future_gt"], bool)

    def test_persistence_end_to_end_future_only(self):
        output = self.work / "persistence_result"
        run = self._run([
            sys.executable, str(ROOT / "run_baseline.py"),
            "--method", "persistence", "--manifest", str(self.manifest),
            "--output", str(output),
        ])
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        required = (
            "manifest.json", "config.json", "baseline_card.json",
            "sample_metrics.jsonl", "aggregate_metrics.json", "scores.json",
            "resolved_manifest.json", "frames.npz", "videos/index.json",
            "logs/run_status.json",
        )
        for relative in required:
            self.assertTrue((output / relative).is_file(), relative)
        aggregate = json.loads((output / "aggregate_metrics.json").read_text(
            encoding="utf-8"))
        self.assertEqual(aggregate["scoring_scope"], "future_frames_only")
        self.assertEqual(aggregate["condition_frames_excluded"], 1)
        self.assertLess(aggregate["SSIM"], 1.0)
        self.assertIsNone(aggregate["FID"])
        self.assertIsNone(aggregate["LPIPS"])
        self.assertIsNone(aggregate["FVD_R3D_PROXY"])
        self.assertIsNone(aggregate["FVD_I3D"])
        identity = aggregate["experiment_identity"]
        required_identity = {
            "git_commit", "code_version", "dataset_checksum", "split_checksum",
            "checkpoint", "lora_target", "lora_rank", "pose_route", "frames",
            "training_samples", "test_samples", "seed", "learning_rate",
            "training_steps", "batch_size", "precision", "GPU",
        }
        self.assertTrue(required_identity.issubset(identity))
        frozen_manifest = json.loads(self.manifest.read_text(encoding="utf-8"))
        self.assertEqual(identity["dataset_checksum"],
                         frozen_manifest["dataset_checksum"])
        self.assertEqual(identity["split_checksum"],
                         frozen_manifest["split_checksum"])
        self.assertEqual(identity["test_samples"], 4)
        rows = [json.loads(line) for line in
                (output / "sample_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual({row["frame_count"] for row in rows}, {17, 25, 33})
        self.assertEqual({row["condition_frames_excluded"] for row in rows}, {1})
        self.assertEqual({row["scored_future_frames"] for row in rows}, {16, 24, 32})

        eval_run = self._run([
            sys.executable, str(ROOT / "server_run" / "evaluate_baseline_results.py"),
            "--result-dir", str(output), "--device", "cpu", "--update",
        ])
        self.assertEqual(eval_run.returncode, 0, eval_run.stdout + eval_run.stderr)
        refreshed = json.loads((output / "aggregate_metrics.json").read_text(
            encoding="utf-8"))
        self.assertEqual(refreshed["evaluator_provenance"]["n_samples"], 4)
        self.assertEqual(refreshed["evaluator_provenance"]["condition_frames_excluded"], 1)
        # Distribution metrics must fail before any feature model can load/download.
        too_small = self._run([
            sys.executable, str(ROOT / "server_run" / "evaluate_baseline_results.py"),
            "--result-dir", str(output), "--device", "cpu", "--fid",
        ])
        self.assertNotEqual(too_small.returncode, 0)
        self.assertIn("at least 32 complete clips", too_small.stdout + too_small.stderr)

    def test_dry_run_never_starts_training(self):
        output = self.work / "dryrun_no_write"
        run = self._run([
            sys.executable, str(ROOT / "run_baseline.py"),
            "--method", "persistence", "--manifest", str(self.manifest),
            "--output", str(output), "--dry-run",
        ])
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("TRAINING_STARTED=0", run.stdout)
        self.assertFalse(output.exists())

    def test_unavailable_external_baseline_fails_loudly(self):
        output = self.work / "rswise_stub"
        run = self._run([
            sys.executable, str(ROOT / "run_baseline.py"),
            "--method", "rswise_recursive", "--manifest", str(self.manifest),
            "--output", str(output),
        ])
        self.assertNotEqual(run.returncode, 0)
        status = json.loads((output / "logs" / "run_status.json").read_text(
            encoding="utf-8"))
        self.assertEqual(status["status"], "FAILED")

    def test_empty_bridge_output_fails_loudly(self):
        producer = self.work / "empty_producer"
        (producer / "generated").mkdir(parents=True)
        output = self.work / "empty_bridge"
        run = self._run([
            sys.executable, str(ROOT / "run_baseline.py"),
            "--method", "rswise_recursive_bridge",
            "--manifest", str(self.manifest), "--output", str(output),
        ], env={"RSWM_RSWISE_EVAL_DIR": str(producer)})
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("missing", run.stdout + run.stderr)

    def test_bridge_rejects_mismatched_producer_identity(self):
        producer = self.work / "wrong_identity_producer"
        generated = producer / "generated"
        meta = producer / "meta"
        generated.mkdir(parents=True)
        meta.mkdir()
        with h5py.File(self.h5, "r") as file:
            for key in json.loads(self.manifest.read_text(
                    encoding="utf-8"))["sample_keys"]:
                np.save(generated / f"{key}.npy", file["data"][key]["video_frames"][:])
        first = json.loads(self.manifest.read_text(encoding="utf-8"))["sample_keys"][0]
        (meta / f"{first}.json").write_text(json.dumps({
            "key": first, "gen_seed": 42,
            "condition_type": "text_plucker",  # no-camera requires text_only
        }), encoding="utf-8")
        output = self.work / "wrong_identity_result"
        run = self._run([
            sys.executable, str(ROOT / "run_baseline.py"),
            "--method", "video_no_camera", "--manifest", str(self.manifest),
            "--output", str(output),
        ], env={"RSWM_NO_CAMERA_EVAL_DIR": str(producer)})
        self.assertNotEqual(run.returncode, 0)
        status = json.loads((output / "logs" / "run_status.json").read_text(
            encoding="utf-8"))
        self.assertEqual(status["status"], "FAILED")
        self.assertIn("condition_type", run.stdout + run.stderr)

    def test_leakage_audit_detects_cross_dataset_copy(self):
        doc = json.loads(self.manifest.read_text(encoding="utf-8"))
        second = json.loads(json.dumps(doc))
        second["source_dataset"] = "WHU-RS19"
        for item in second["samples"]:
            item["source_dataset"] = "WHU-RS19"
            item["source_split"] = "test"
        second_path = self.work / "whu_duplicate_manifest.json"
        second_path.write_text(json.dumps(second), encoding="utf-8")
        result = leakage_audit([self.manifest, second_path], near_threshold=0)
        self.assertFalse(result["pass"])
        self.assertGreater(len(result["exact_cross_split_duplicates"]), 0)

    def test_matched_configs_and_rank_sweep_semantics(self):
        expected = {
            ("native", "attn_only"), ("native", "attn_cam"),
            ("drop", "attn_only"), ("drop", "attn_cam"),
        }
        seen = set()
        recipes = []
        for path in sorted((ROOT / "configs" / "benchmark_ablation").glob("*.yaml")):
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            seen.add((doc["dataset"]["pose_mode"], doc["lora"]["target_modules"]))
            recipes.append((doc["training"], doc["evaluation"]))
            self.assertEqual(doc["lora"]["rank"], 16)
            self.assertEqual(doc["evaluation"]["max_samples"], 400)
        self.assertEqual(seen, expected)
        first_training, first_eval = recipes[0]
        for training, evaluation in recipes[1:]:
            self.assertEqual(training, first_training)
            self.assertEqual(evaluation, first_eval)
        ranks = []
        for path in sorted((ROOT / "configs" / "rank_sweep").glob("*.yaml")):
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            ranks.append(doc["lora"]["rank"])
            self.assertEqual(doc["dataset"]["pose_mode"], "native")
            self.assertEqual(doc["lora"]["target_modules"], "attn_cam")
            self.assertEqual(doc["lora"]["alpha"], doc["lora"]["rank"])
        self.assertEqual(ranks, [16, 32, 4, 64, 8])  # lexical file order
        self.assertEqual(sorted(ranks), [4, 8, 16, 32, 64])

    def test_table_export_is_automatic_and_preserves_null(self):
        result = self.work / "persistence_result"
        if not result.is_dir():
            self.test_persistence_end_to_end_future_only()
        paper = self.work / "paper_results"
        run = self._run([
            sys.executable, str(ROOT / "tools" / "export_paper_tables.py"),
            "--results", str(result), "--output", str(paper),
        ])
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        expected = (
            "main_results.csv", "overlap_results.csv", "trajectory_results.csv",
            "rank_ablation.csv", "target_ablation.csv",
            "cross_dataset_results.csv", "efficiency_results.csv",
        )
        for name in expected:
            self.assertTrue((paper / name).is_file(), name)
        with (paper / "main_results.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["FID"], "")
        self.assertEqual(rows[0]["FVD_I3D"], "")

        tampered = self.work / "tampered_result"
        shutil.copytree(result, tampered)
        aggregate_path = tampered / "aggregate_metrics.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        aggregate["N"] = 999
        aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")
        rejected = self._run([
            sys.executable, str(ROOT / "tools" / "export_paper_tables.py"),
            "--results", str(tampered), "--output", str(self.work / "rejected_table"),
        ])
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("row count", rejected.stdout + rejected.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
