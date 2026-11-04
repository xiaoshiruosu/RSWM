from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECK_DATASET = ROOT / "server_run" / "check_dataset.py"


@contextmanager
def temporary_path(suffix: str):
    stream = tempfile.NamedTemporaryFile(
        dir=ROOT / "tests", suffix=suffix, delete=False)
    path = Path(stream.name)
    stream.close()
    path.unlink()
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def make_canonical_fixture(path: Path) -> None:
    height, width, frames = 480, 832, 17
    yy = np.arange(height, dtype=np.uint16)[:, None]
    xx = np.arange(width, dtype=np.uint16)[None, :]
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = (xx % 256).astype(np.uint8)
    image[..., 1] = (yy % 256).astype(np.uint8)
    image[..., 2] = ((xx + yy) % 256).astype(np.uint8)

    with h5py.File(path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["total_instances"] = 4
        metadata.attrs["target_size"] = (480, 832)
        metadata.attrs["pose_window_size"] = (128, 222)
        metadata.attrs["steps_per_segment"] = 9
        metadata.attrs["angle_quantize"] = 0
        metadata.attrs["min_overlap_ratio"] = 0.0
        metadata.attrs["max_overlap_ratio"] = 0.1
        metadata.attrs["semgate_enabled"] = True
        metadata.attrs["builder_version"] = "v5+semgate_v3_1_20260729"
        metadata.attrs["pose_injection"] = "plucker_from_trajectory"
        data = handle.create_group("data")
        for index, scene in enumerate(("general", "water", "urban", "rural")):
            group = data.create_group(f"sample_{index:05d}")
            group.create_dataset("image", data=image, compression="gzip")
            video = group.create_dataset(
                "video_frames", shape=(frames, height, width, 3), dtype="uint8",
                chunks=(1, 64, 64, 3), fillvalue=127,
            )
            video[0] = image
            video[-1] = np.flip(image, axis=1)
            group.create_dataset("intrinsics", shape=(frames, 4), dtype="float32")
            group.create_dataset("poses", shape=(frames, 4, 4), dtype="float32")
            group.create_dataset("trajectory", shape=(frames, 2), dtype="float32")
            group.create_dataset("waypoints", shape=(3, 2), dtype="float32")
            group.create_dataset("segment_angles", shape=(2,), dtype="float32")
            group.create_dataset("full_scene", shape=(512, 512, 3), dtype="uint8")
            group.attrs["prompt"] = f"canonical {scene} trajectory"
            group.attrs["num_frames"] = frames
            group.attrs["num_segments"] = 2
            group.attrs["overlap_ratio"] = 0.05
            group.attrs["pose_source"] = "plucker_from_trajectory"
            group.attrs["source_scene_category"] = scene
            group.attrs["source_dataset"] = "blanchon/AID"
            group.attrs["source_split"] = "train"
            group.attrs["source_index"] = index
            group.attrs["source_path"] = f"data/{scene}/sample_{index}.jpg"
            if scene != "general":
                group.attrs["semgate_frac"] = 0.9
                group.attrs["semgate_threshold"] = 0.5
                group.attrs["semgate_n_scored"] = 9
                group.attrs["semgate_tries"] = 1


class CanonicalDatasetValidatorTests(unittest.TestCase):
    def test_valid_fixture_passes_and_writes_a_bound_report(self) -> None:
        with temporary_path(".h5") as h5_path, temporary_path(".json") as report_path:
            make_canonical_fixture(h5_path)
            result = subprocess.run(
                [sys.executable, str(CHECK_DATASET), str(h5_path), "--validate",
                 "--expect-n", "4", "--report", str(report_path)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["summary"]["n_samples"], 4)
            self.assertEqual(report["summary"]["semantic_evidence_samples"], 3)
            self.assertGreater(report["summary"]["pixel_probe_std"], 10.0)

    def test_disabled_semantic_gate_is_a_hard_failure(self) -> None:
        with temporary_path(".h5") as h5_path:
            make_canonical_fixture(h5_path)
            with h5py.File(h5_path, "a") as handle:
                handle["metadata"].attrs["semgate_enabled"] = False
            result = subprocess.run(
                [sys.executable, str(CHECK_DATASET), str(h5_path), "--validate",
                 "--expect-n", "4"], text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 8)
            self.assertIn("semgate_enabled", result.stderr)


class FullPipelineWiringTests(unittest.TestCase):
    def test_build_entry_is_frozen_and_offline(self) -> None:
        text = (ROOT / "server_run" / "build_correct.sh").read_text(encoding="utf-8")
        self.assertIn("lora_remotesense/build_dataset_from_benchmark.py", text)
        self.assertNotIn("build_from_remotesense.py", text)
        for token in ("--window-height 128", "--window-width 222", "--steps 9",
                      "--target-height 480", "--target-width 832",
                      "--min-overlap-ratio 0.0", "--max-overlap-ratio 0.1"):
            self.assertIn(token, text)
        self.assertIn('RSWM_SEMGATE_MODEL', text)
        self.assertIn('export HF_HUB_OFFLINE=1', text)
        self.assertIn('PARTIAL="$OUT.partial"', text)
        self.assertIn('quarantine_invalid_h5', text)
        self.assertIn('final dataset failed canonical validation; rebuilding', text)
        self.assertIn('partial dataset failed canonical validation; rebuilding', text)
        self.assertIn('mv -- "$PARTIAL" "$OUT"', text)

    def test_semantic_gate_uses_clip_helpers_without_return_dict(self) -> None:
        text = (ROOT / "lora_remotesense" / "semantic_gate.py").read_text(
            encoding="utf-8")
        self.assertNotIn("return_dict=True", text)
        self.assertIn("get_text_features(**tok)", text)
        self.assertIn("get_image_features(**px)", text)
        self.assertIn("self._coerce_features(", text)
        from lora_remotesense.semantic_gate import default_thresholds
        overrides = {
            "RSWM_SEMGATE_MIN_FRAC": "0.2",
            "RSWM_SEMGATE_MIN_FRAC_WATER": "0.1",
            "RSWM_SEMGATE_MIN_FRAC_URBAN": "0.3",
            "RSWM_SEMGATE_MIN_FRAC_RURAL": "0.4",
        }
        with patch.dict(os.environ, overrides):
            self.assertEqual(default_thresholds(), {
                "water": 0.1, "urban": 0.3, "rural": 0.4,
            })

    def test_full_auto_launcher_waits_for_a_numeric_gpu_and_logs(self) -> None:
        path = ROOT / "server_run" / "start_full_auto_pipeline_v20260817.sh"
        text = path.read_text(encoding="utf-8")
        self.assertIn('RSWM_GPU_POLL_SECONDS:-30', text)
        self.assertIn('rswm_gpu_selector.py', text)
        self.assertIn('--poll-seconds "$GPU_POLL_SECONDS"', text)
        self.assertIn('[[ "$GPU" =~ ^[0-9]+$ ]]', text)
        self.assertIn('echo "GPU=$GPU"', text)
        self.assertIn('export CUDA_VISIBLE_DEVICES="$GPU"', text)
        self.assertIn('run_iclr_full_pipeline.sh', text)
        self.assertIn('exec > >(tee -a "$LOG_FILE") 2>&1', text)
        self.assertNotIn('GPU="$GPU,"', text)

    def test_full_pipeline_uses_only_formal_entries_and_has_all_checkpoints(self) -> None:
        text = (ROOT / "server_run" / "run_iclr_full_pipeline.sh").read_text(
            encoding="utf-8")
        self.assertIn('MODE="plan"', text)
        self.assertIn('readonly DATASET_N=2000', text)
        self.assertIn('bash "$SCRIPT_DIR/build_correct.sh"', text)
        self.assertNotIn("build_from_remotesense.py", text)
        self.assertIn("run_iclr_main_a800.sh", text)
        self.assertIn("run_iclr_evaluation_a800.sh", text)
        self.assertIn("RSWM_CONFIRM_FULL_PIPELINE", text)
        for step in ("01_environment", "02_aid_snapshot", "03_clip_snapshot",
                     "04_build_dataset", "05a_dataset_structure",
                     "05b_semantic_recheck", "05c_duplicate_gate",
                     "06_official_splits", "06b_aid_manifest", "07_train_s42",
                     "07_train_s43", "07_train_s44", "08a_rollout_",
                     "08b_standardize_", "08c_evaluate_", "09_tables_"):
            self.assertIn(step, text)
        self.assertIn("--wait-for-data", text)
        self.assertIn("write_checkpoint", text)

    def test_evaluation_wrapper_can_checkpoint_evaluation_and_export_separately(self) -> None:
        text = (ROOT / "server_run" / "iclr_a800" /
                "run_iclr_evaluation_a800.sh").read_text(encoding="utf-8")
        self.assertIn("--evaluation-only", text)
        self.assertIn("--export-only", text)
        self.assertIn('ACTION="evaluation"', text)
        self.assertIn('ACTION="export"', text)

    def test_paper_test_rollout_is_explicit_and_separate_from_validation(self) -> None:
        evaluator = (ROOT / "lora_remotesense" / "new" / "evaluate.py").read_text(
            encoding="utf-8")
        wrapper = (ROOT / "server_run" / "run_evaluation.sh").read_text(
            encoding="utf-8")
        self.assertIn('choices=("val", "test")', evaluator)
        self.assertIn('default="val"', evaluator)
        self.assertIn('--split-name test requires a separate --output-dir', wrapper)
        self.assertIn('EVALUATION_LOG="$EXPERIMENT_DIR/logs/evaluation.log"', wrapper)

    def test_primary_result_group_contains_exactly_three_seed_aliases(self) -> None:
        plan = json.loads((ROOT / "configs" / "iclr_main" /
                           "experiment_plan.json").read_text(encoding="utf-8"))
        self.assertEqual(plan["baseline_groups"]["primary"], [
            "rsroll_primary_s42", "rsroll_primary_s43", "rsroll_primary_s44",
        ])


class ServerHotfixRegressionTests(unittest.TestCase):
    @staticmethod
    def _load_v5_module():
        path = (ROOT / "wan" / "remotesensebenchmark" /
                "build_multi_direction_benchmark_v5.py")
        spec = importlib.util.spec_from_file_location("rswm_v5_hotfix_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    def test_official_hf_failure_never_falls_back_to_synthetic(self) -> None:
        module = self._load_v5_module()
        config = module.MultiDirectionConfig(scene_categories=["general"])
        with patch.object(
                module, "hf_load_dataset", side_effect=RuntimeError("429 rate limit")):
            with self.assertRaisesRegex(RuntimeError, "synthetic fallback disabled"):
                module.SceneGenerator(config, use_hf=True)

    def test_bounded_semgate_retry_selects_best_candidate(self) -> None:
        module = self._load_v5_module()
        from lora_remotesense.semantic_gate import GateVerdict, max_tries

        class NoopVideoWriter:
            def __init__(self, *args, **kwargs):
                pass

            def write_frames(self, frames):
                return True

        class RejectSequence:
            def __init__(self):
                self.calls = 0

            def check_frames(self, frames, category):
                score = (0.1, 0.4, 0.2)[self.calls]
                self.calls += 1
                return GateVerdict(
                    category=category, passed=False, frac=score,
                    n_scored=9, threshold=0.6)

        config = module.MultiDirectionConfig(
            output_dir=".", instances_per_scene=1,
            scene_categories=["rural"], full_scene_size=(256, 256),
            window_size=(64, 64), random_seed=42)
        builder = module.MultiDirectionBenchmarkBuilder(config, use_hf=False)
        gate = RejectSequence()
        builder.semantic_gate = gate
        builder.semantic_gate_max_tries = 3
        builder.max_retry_per_sample = 3
        builder.enforce_overlap = False
        with patch.object(module, "VideoWriter", NoopVideoWriter):
            samples, end_index = builder._process_scene("rural", Path("."), 0)

        self.assertEqual(gate.calls, 3)
        self.assertEqual(end_index, 1)
        self.assertEqual(samples[0]["source_scene_id"], 1)
        self.assertEqual(samples[0]["semgate"]["frac"], 0.4)
        self.assertEqual(samples[0]["semgate"]["threshold"], 0.4)
        self.assertEqual(builder._overlap_audit["n_semantic_fallback"], 1)

        with patch.dict(os.environ, {"RSWM_SEMGATE_MAX_TRIES": "100"}):
            self.assertEqual(max_tries(), 20)

        for relative in (
                "lora_remotesense/build_dataset_from_benchmark.py",
                "lora_remotesense/semantic_gate.py",
                "wan/remotesensebenchmark/build_multi_direction_benchmark_v5.py"):
            text = (ROOT / relative).read_text(encoding="utf-8")
            self.assertLess(text.index('os.environ[_thread_env]'),
                            text.index("import numpy"), relative)

    def test_dataset_source_is_real_aid_and_demo_remains_explicit(self) -> None:
        module = self._load_v5_module()
        config = module.MultiDirectionConfig(scene_categories=["general"])
        aid_config = {
            "general": {"datasets": [{
                "name": "blanchon/AID", "split": "train", "feature": "image",
            }]},
        }
        with patch.object(module, "HF_DATASET_CONFIG", aid_config), \
                patch.object(module, "HF_DATASETS_CACHE", {}), \
                patch.object(module, "hf_load_dataset", return_value=[{"image": None}]):
            source = module.SceneGenerator(config, use_hf=True)
        self.assertEqual(source.get_dataset_source("general"), "blanchon/AID")
        source._load_scene_from_hf = lambda *args, **kwargs: None
        with self.assertRaisesRegex(RuntimeError, "synthetic fallback disabled"):
            source.generate_scene("general", 0)

        demo = module.SceneGenerator(config, use_hf=False)
        self.assertEqual(demo.get_dataset_source("general"), "standalone_demo")
        self.assertNotEqual(demo.get_dataset_source("general"), "synthetic")

        mirror = (ROOT / "lora_remotesense" / "new" / "wan" /
                  "remotesensebenchmark" / "build_multi_direction_benchmark_v5.py")
        primary = (ROOT / "wan" / "remotesensebenchmark" /
                   "build_multi_direction_benchmark_v5.py")
        self.assertEqual(primary.read_bytes(), mirror.read_bytes())

    def test_vae_resize_preserves_cthw_layout_and_dtype(self) -> None:
        import torch

        path = ROOT / "lora_remotesense" / "new" / "inference.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_resize_decoded_video"
        )
        isolated = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(isolated)
        namespace = {"torch": torch}
        exec(compile(isolated, str(path), "exec"), namespace)
        resize = namespace["_resize_decoded_video"]

        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                video = torch.randn(3, 17, 464, 832, dtype=dtype)
                resized = resize(video)
                self.assertEqual(tuple(resized.shape), (3, 17, 480, 832))
                self.assertEqual(resized.dtype, video.dtype)
                self.assertIs(resize(resized), resized)

    def test_semgate_report_handles_missing_overlap_audit(self) -> None:
        script = ROOT / "server_run" / "semgate_build_report.py"
        with temporary_path(".h5") as h5_path, \
                temporary_path(".attempts.json") as attempts_path, \
                temporary_path(".report.json") as report_path:
            self.assertFalse((attempts_path.parent / "overlap_audit.json").exists())
            with h5py.File(h5_path, "w") as handle:
                metadata = handle.create_group("metadata")
                metadata.attrs["semgate_enabled"] = True
                data = handle.create_group("data")
                sample = data.create_group("sample_00000")
                sample.attrs["source_scene_category"] = "water"
                sample.attrs["source_class"] = "River"
                sample.attrs["source_scene_id"] = 1
                sample.attrs["semgate_frac"] = 0.8
                sample.attrs["semgate_tries"] = 1
            attempts_path.write_text(json.dumps({
                "attempts": {
                    "water|1": {"class": "River", "index": 1},
                    "water|2": {"class": "River", "index": 2},
                },
            }), encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(script), "--gt-h5", str(h5_path),
                 "--attempts", str(attempts_path), "-o", str(report_path)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["attempts_available"])
            river = next(row for row in report["per_class"]
                         if row["scene"] == "water" and row["class"] == "river")
            self.assertEqual(river["accepted"], 1)
            self.assertEqual(river["rejected"], 1)

    def test_evaluation_dependencies_and_optional_lpips_contract(self) -> None:
        required = ("imageio", "imageio-ffmpeg", "scikit-image", "lpips")
        for relative in ("requirements-lock.txt",
                         "lora_remotesense/new/requirements.txt"):
            text = (ROOT / relative).read_text(encoding="utf-8").lower()
            for package in required:
                self.assertRegex(text, rf"(?m)^{package}(?:[<>=]|$)")
        metrics = (ROOT / "lora_remotesense" / "new" / "metrics.py").read_text(
            encoding="utf-8")
        self.assertIn("LPIPS 初始化失败，跳过后续 LPIPS 计算", metrics)
        self.assertIn("except Exception as e:\n            print(f\"Warning: LPIPS 计算失败", metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
