#!/usr/bin/env python3
"""CPU-only tests for the formal experiment orchestration layer."""

from __future__ import annotations

import copy
import contextlib
import importlib.util
import json
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = PROJECT_ROOT / "server_run" / "generate_manifest.py"
SPEC = importlib.util.spec_from_file_location("generate_manifest", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
manifest_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest_tool)

PREFLIGHT_PATH = PROJECT_ROOT / "server_run" / "preflight_a800.py"
PREFLIGHT_SPEC = importlib.util.spec_from_file_location("preflight_a800", PREFLIGHT_PATH)
assert PREFLIGHT_SPEC is not None and PREFLIGHT_SPEC.loader is not None
preflight_a800 = importlib.util.module_from_spec(PREFLIGHT_SPEC)
PREFLIGHT_SPEC.loader.exec_module(preflight_a800)

FORMAL_RSWM_BASE = "/nix/rsroll_25828"


def load_formal_config(name: str):
    with mock.patch.dict(os.environ, {"RSWM_BASE": FORMAL_RSWM_BASE}):
        return manifest_tool.load_config(
            PROJECT_ROOT / "configs" / name,
            project_root=PROJECT_ROOT,
        )


@contextlib.contextmanager
def temporary_test_directory():
    """Create a portable test directory without tempfile's Windows ACL mode."""
    path = PROJECT_ROOT / "tests" / f".runner_test_{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


class ExperimentConfigTests(unittest.TestCase):
    def test_all_formal_configs_parse_with_expected_arms(self) -> None:
        expected = {
            "native_attn_cam_r16.yaml": ("native", "native", "attn_cam", 16),
            "drop_attn_only_r16.yaml": ("drop", "drop", "attn_only", 16),
            "all_r16_ablation.yaml": ("drop", "drop", "all", 16),
        }
        for name, values in expected.items():
            with self.subTest(config=name):
                config = load_formal_config(name)
                self.assertEqual(
                    (config["experiment"]["role"],
                     config["dataset"]["pose_mode"],
                     config["lora"]["target_modules"],
                     config["lora"]["rank"]),
                    values,
                )
                plan = manifest_tool.plan_values(
                    config,
                    config_path=PROJECT_ROOT / "configs" / name,
                    experiment_id=f"20990101_{config['experiment']['name']}",
                    experiments_root=PROJECT_ROOT / "experiments",
                )
                self.assertEqual(plan["PLAN_SEM_ARM_MODE"], values[1])
                self.assertEqual(config["hardware"]["profile"], "a800_80g")
                self.assertEqual(config["training"]["batch_size"], 1)

    def test_dry_run_report_uses_config_semantics(self) -> None:
        runner = (PROJECT_ROOT / "server_run" / "rswm_train_runner.sh").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("export SEM_ARM_MODE=drop", runner)
        self.assertIn('export SEM_ARM_MODE="$PLAN_SEM_ARM_MODE"', runner)
        self.assertIn('echo "SEM_ARM_MODE: $PLAN_SEM_ARM_MODE"', runner)
        for required in (
            "EXPERIMENT DRY RUN", "experiment_id:", "config:", "pose_mode:",
            "LoRA rank:", "target_modules:", "model_path:", "dataset_path:",
            "GPU mode:", "checkpoint path:", "TRAINING_STARTED=0",
        ):
            self.assertIn(required, runner)

    def test_rswm_base_paths_expand_and_absolute_paths_remain_compatible(self) -> None:
        config_path = PROJECT_ROOT / "configs" / "native_attn_cam_r16.yaml"
        self.assertIn("${RSWM_BASE}", config_path.read_text(encoding="utf-8"))
        config = load_formal_config("native_attn_cam_r16.yaml")
        self.assertEqual(
            config["model"]["checkpoint"],
            "/nix/rsroll_25828/models/lingbot-world-base-cam",
        )
        self.assertEqual(
            config["dataset"]["train"],
            "/nix/rsroll_25828/work/rsroll_official/data/train_geo.h5",
        )

        raw = manifest_tool.load_yaml(config_path)
        raw["model"]["checkpoint"] = "/nix/absolute/model"
        raw["dataset"]["train"] = "/nix/absolute/train.h5"
        raw["dataset"]["validation"] = "/nix/absolute/split.json"
        with mock.patch.dict(os.environ):
            os.environ.pop("RSWM_BASE", None)
            absolute = manifest_tool.validate_config(raw, project_root=PROJECT_ROOT)
        self.assertEqual(absolute["model"]["checkpoint"], "/nix/absolute/model")

    def test_unresolved_rswm_base_fails_before_training(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("RSWM_BASE", None)
            with self.assertRaises(manifest_tool.ConfigError):
                manifest_tool.load_config(
                    PROJECT_ROOT / "configs" / "native_attn_cam_r16.yaml",
                    project_root=PROJECT_ROOT,
                )

    def test_unsupported_step_and_save_semantics_fail_closed(self) -> None:
        source = manifest_tool.load_yaml(
            PROJECT_ROOT / "configs" / "native_attn_cam_r16.yaml"
        )
        for key, value in (("max_steps", 100), ("save_interval", 2)):
            invalid = copy.deepcopy(source)
            invalid["training"][key] = value
            with self.subTest(key=key), mock.patch.dict(
                os.environ, {"RSWM_BASE": FORMAL_RSWM_BASE}
            ), self.assertRaises(manifest_tool.ConfigError):
                manifest_tool.validate_config(invalid, project_root=PROJECT_ROOT)

    def test_commands_use_the_existing_entries_and_native_flags(self) -> None:
        config = load_formal_config("native_attn_cam_r16.yaml")
        experiment_dir = PROJECT_ROOT / "experiments" / "20990101_native_attn_cam_r16"
        command = manifest_tool.build_train_command(
            config,
            python="/usr/bin/python3",
            project_root=PROJECT_ROOT,
            experiment_dir=experiment_dir,
        )
        self.assertEqual(Path(command[1]).name, "train.py")
        self.assertIn("--lora_targets", command)
        self.assertEqual(command[command.index("--lora_targets") + 1], "attn_cam")
        self.assertEqual(command[command.index("--pose_mode") + 1], "native")
        self.assertIn("--gradient_checkpointing", command)
        self.assertIn("--no_8bit_adam", command)
        self.assertNotIn("--max_steps", command)

        evaluation = manifest_tool.build_evaluation_command(
            config,
            python="/usr/bin/python3",
            project_root=PROJECT_ROOT,
            experiment_dir=experiment_dir,
        )
        self.assertEqual(Path(evaluation[1]).name, "evaluate.py")
        self.assertIn("--lora_path", evaluation)
        self.assertIn("--output_dir", evaluation)


class ManifestAndResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = PROJECT_ROOT / "configs" / "native_attn_cam_r16.yaml"
        self.config = load_formal_config("native_attn_cam_r16.yaml")

    @staticmethod
    def _complete_checkpoint(path: Path) -> None:
        path.mkdir(parents=True)
        for name in ("training_state.pt", "training_config.json", "run_manifest.json"):
            (path / name).write_text("{}", encoding="utf-8")
        for expert in ("low_noise", "high_noise"):
            adapter = path / expert
            adapter.mkdir()
            (adapter / "adapter_model.safetensors").write_bytes(b"test")

    def test_manifest_creation_roundtrip_and_layout(self) -> None:
        with temporary_test_directory() as root:
            with mock.patch.object(
                manifest_tool,
                "_runtime_versions",
                return_value={
                    "python_path": "/usr/bin/python3",
                    "python_version": "3.test",
                    "torch_version": "test",
                    "cuda_version": "test",
                },
            ), mock.patch.object(manifest_tool, "_git_commit", return_value="abc123"):
                path = manifest_tool.create_or_resume_manifest(
                    self.config,
                    config_path=self.config_path,
                    experiment_id="20990101_native_attn_cam_r16",
                    experiments_root=root,
                    project_root=PROJECT_ROOT,
                    python_path="/usr/bin/python3",
                    requested_gpu="auto",
                    physical_gpu=None,
                    cuda_visible_devices=None,
                    status="WAITING_GPU",
                    dry_run=False,
                )
            experiment_dir = path.parent
            for child in ("logs", "checkpoints", "results"):
                self.assertTrue((experiment_dir / child).is_dir())
            self.assertTrue((experiment_dir / "config.yaml").is_file())

            manifest = manifest_tool.read_manifest(experiment_dir)
            self.assertEqual(manifest["experiment_id"], "20990101_native_attn_cam_r16")
            self.assertEqual(manifest["status"], "WAITING_GPU")
            self.assertEqual(manifest["git_commit"], "abc123")
            self.assertEqual(manifest["lora"]["rank"], 16)
            self.assertEqual(manifest["training"]["batch_size"], 1)
            self.assertIsInstance(manifest["training"]["command"], list)

    def test_latest_complete_checkpoint_ignores_partial_directory(self) -> None:
        with temporary_test_directory() as experiment_dir:
            checkpoints = experiment_dir / "checkpoints"
            first = checkpoints / "checkpoint-epoch-1"
            second = checkpoints / "checkpoint-epoch-10_20990101_010101"
            partial = checkpoints / "checkpoint-epoch-3"
            self._complete_checkpoint(first)
            self._complete_checkpoint(second)
            partial.mkdir(parents=True)
            (partial / "training_state.pt").write_text("partial", encoding="utf-8")
            os.utime(first, (100, 100))
            os.utime(second, (2, 2))
            os.utime(partial, (3, 3))
            self.assertEqual(
                manifest_tool.latest_complete_checkpoint(experiment_dir), second
            )

    def test_resume_checkpoint_cannot_escape_experiment(self) -> None:
        with temporary_test_directory() as root:
            experiment_dir = root / "experiment"
            (experiment_dir / "checkpoints").mkdir(parents=True)
            outside = root / "checkpoint-epoch-9"
            self._complete_checkpoint(outside)
            with self.assertRaises(manifest_tool.ConfigError):
                manifest_tool.validate_resume_checkpoint(experiment_dir, outside)

    def test_resume_records_attempt_and_native_resume_argument(self) -> None:
        with temporary_test_directory() as root:
            experiment_id = "20990101_native_attn_cam_r16"
            with mock.patch.object(
                manifest_tool,
                "_runtime_versions",
                return_value={"python_path": "/usr/bin/python3"},
            ), mock.patch.object(manifest_tool, "_git_commit", return_value="abc123"):
                manifest_tool.create_or_resume_manifest(
                    self.config,
                    config_path=self.config_path,
                    experiment_id=experiment_id,
                    experiments_root=root,
                    project_root=PROJECT_ROOT,
                    python_path="/usr/bin/python3",
                    requested_gpu="auto",
                    physical_gpu=None,
                    cuda_visible_devices=None,
                    status="WAITING_GPU",
                    dry_run=False,
                )
            checkpoint = root / experiment_id / "checkpoints" / "checkpoint-epoch-2"
            self._complete_checkpoint(checkpoint)
            manifest_tool.create_or_resume_manifest(
                self.config,
                config_path=root / experiment_id / "config.yaml",
                experiment_id=experiment_id,
                experiments_root=root,
                project_root=PROJECT_ROOT,
                python_path="/usr/bin/python3",
                requested_gpu="2",
                physical_gpu=None,
                cuda_visible_devices=None,
                status="WAITING_GPU",
                dry_run=False,
                resume_from=str(checkpoint),
                resume=True,
            )
            manifest = manifest_tool.read_manifest(root / experiment_id)
            self.assertEqual(len(manifest["attempts"]), 2)
            command = manifest["training"]["command"]
            self.assertIn("--resume_from", command)
            self.assertEqual(command[command.index("--resume_from") + 1], str(checkpoint))


class SafetyTests(unittest.TestCase):
    def test_a800_free_vram_threshold_mock(self) -> None:
        threshold = load_formal_config("native_attn_cam_r16.yaml")["hardware"][
            "min_free_gib"
        ]
        self.assertEqual(threshold, 70.0)
        expected = {75: True, 60: False, 50: False}
        for free_gib, passes in expected.items():
            with self.subTest(free_gib=free_gib):
                self.assertEqual(
                    preflight_a800.gpu_free_memory_passes(free_gib * 1024, threshold),
                    passes,
                )

    def test_gpu_request_parsing(self) -> None:
        self.assertEqual(manifest_tool.resolve_gpu_request(None, "auto"), "auto")
        self.assertEqual(manifest_tool.resolve_gpu_request("02", "auto"), "2")
        with self.assertRaises(manifest_tool.ConfigError):
            manifest_tool.resolve_gpu_request("-1", "auto")
        with self.assertRaises(manifest_tool.ConfigError):
            manifest_tool.resolve_gpu_request("GPU-uuid", "auto")

    def test_gpu_request_respects_inherited_visibility(self) -> None:
        manifest_tool.assert_gpu_allowed("2", "1,2,3")
        with self.assertRaises(manifest_tool.ConfigError):
            manifest_tool.assert_gpu_allowed("4", "1,2,3")
        with self.assertRaises(manifest_tool.ConfigError):
            manifest_tool.assert_gpu_allowed("2", "GPU-uuid")

    def test_runtime_path_and_disk_checks(self) -> None:
        raw = manifest_tool.load_yaml(
            PROJECT_ROOT / "configs" / "native_attn_cam_r16.yaml"
        )
        with temporary_test_directory() as root:
            data = root / "train_geo.h5"
            split = root / "official_splits.json"
            model = root / "model"
            data.write_bytes(b"test")
            split.write_text(
                json.dumps({"train_indices": [0], "val_indices": [1]}),
                encoding="utf-8",
            )
            model.mkdir()
            for name in (
                "configuration.json",
                "Wan2.1_VAE.pth",
                "models_t5_umt5-xxl-enc-bf16.pth",
            ):
                (model / name).write_text("test", encoding="utf-8")
            (model / "low_noise_model").mkdir()
            (model / "high_noise_model").mkdir()
            raw["dataset"]["train"] = str(data)
            raw["dataset"]["validation"] = str(split)
            raw["model"]["checkpoint"] = str(model)
            raw["hardware"]["min_disk_free_gib"] = 0.000001
            config = manifest_tool.validate_config(raw, project_root=PROJECT_ROOT)
            checks = manifest_tool.check_runtime_paths(
                config,
                experiments_root=root / "experiments",
                project_root=PROJECT_ROOT,
            )
            self.assertTrue(any(value.startswith("dataset=") for value in checks))
            self.assertTrue(any(value.startswith("disk_free_gib=") for value in checks))


if __name__ == "__main__":
    unittest.main(verbosity=2)
