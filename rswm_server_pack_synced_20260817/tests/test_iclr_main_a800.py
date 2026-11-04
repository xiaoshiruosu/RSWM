from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_TOOL = ROOT / "server_run" / "iclr_a800" / "iclr_main_plan.py"
SPEC = importlib.util.spec_from_file_location("iclr_main_plan", PLAN_TOOL)
assert SPEC and SPEC.loader
planner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = planner
SPEC.loader.exec_module(planner)

sys.path.insert(0, str(ROOT / "server_run"))
import generate_manifest  # noqa: E402


RSWM_BASE = "/nix/rsroll_25828"


class ICLRMainPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan, cls.configs = planner.load_and_validate(rswm_base=RSWM_BASE)

    def test_primary_multiseed_semantics(self) -> None:
        ids = planner.group_run_ids(self.plan, "primary")
        self.assertEqual(len(ids), 3)
        self.assertEqual([self.configs[item]["training"]["seed"] for item in ids],
                         [42, 43, 44])
        for run_id in ids:
            config = self.configs[run_id]
            self.assertEqual(config["experiment"]["role"], "native")
            self.assertEqual(config["dataset"]["pose_mode"], "native")
            self.assertEqual(config["lora"], {
                "rank": 16, "alpha": 16, "target_modules": "attn_cam"})
            self.assertEqual(config["training"]["split_seed"], 42)
            self.assertEqual(config["evaluation"]["gen_seed"], 42)
            self.assertEqual(config["evaluation"]["max_samples"], 400)

    def test_matched_2x2_and_rank_sweep_reuse_canonical_r16(self) -> None:
        matched = planner.group_run_ids(self.plan, "matched")
        cells = {(self.configs[item]["dataset"]["pose_mode"],
                  self.configs[item]["lora"]["target_modules"]) for item in matched}
        self.assertEqual(cells, {
            ("native", "attn_only"), ("native", "attn_cam"),
            ("drop", "attn_only"), ("drop", "attn_cam")})
        rank_ids = planner.group_run_ids(self.plan, "rank")
        self.assertEqual([self.configs[item]["lora"]["rank"] for item in rank_ids],
                         [4, 8, 16, 32, 64])
        self.assertIn("iclr_primary_native_attn_cam_r16_s42", rank_ids)
        self.assertEqual(len(planner.group_run_ids(self.plan, "all")), 12)

    def test_every_config_is_accepted_by_formal_runner_and_a800_safe(self) -> None:
        for run_id, config in self.configs.items():
            with self.subTest(run_id=run_id):
                self.assertEqual(config["hardware"]["profile"], "a800_80g")
                self.assertEqual(config["hardware"]["offload"], "1")
                self.assertEqual(config["hardware"]["min_free_gib"], 70.0)
                self.assertEqual(config["training"]["batch_size"], 1)
                self.assertEqual(config["training"]["gradient_accumulation_steps"], 4)
                self.assertEqual(config["training"]["max_frames"], 33)
                values = generate_manifest.plan_values(
                    config,
                    config_path=ROOT / self.plan["runs"][run_id]["config"],
                    experiment_id=run_id,
                    experiments_root=ROOT / "experiments",
                )
                self.assertEqual(values["PLAN_SEM_ARM_MODE"],
                                 config["dataset"]["pose_mode"])
                command = generate_manifest.build_train_command(
                    config, python="/usr/bin/python3", project_root=ROOT,
                    experiment_dir=ROOT / "experiments" / run_id)
                self.assertEqual(command[command.index("--device_id") + 1], "0")
                self.assertEqual(command[command.index("--mixed_precision") + 1], "bf16")
                self.assertEqual(command[command.index("--pose_mode") + 1],
                                 config["dataset"]["pose_mode"])
                self.assertEqual(command[command.index("--lora_targets") + 1],
                                 config["lora"]["target_modules"])

    def test_cross_dataset_manifests_are_test_only(self) -> None:
        manifests = self.plan["benchmark_manifests"]
        self.assertEqual(set(manifests), {"aid", "whurs19", "resisc45"})
        for name in ("whurs19", "resisc45"):
            self.assertEqual(manifests[name]["role"], "cross_dataset_test_only")
            self.assertIs(manifests[name]["train_allowed"], False)
        self.assertEqual(self.plan["protocol"]["score_frames"], "1...T-1")

    def test_no_camera_baseline_holds_attn_cam_target_fixed(self) -> None:
        entry = self.plan["baselines"]["video_no_camera"]
        config = self.configs[entry["config_run"]]
        self.assertEqual(config["dataset"]["pose_mode"], "drop")
        self.assertEqual(config["lora"]["target_modules"], "attn_cam")
        self.assertEqual(config["lora"]["rank"], 16)

    def test_primary_baseline_ingest_tracks_all_training_seeds(self) -> None:
        entries = [self.plan["baselines"][f"rsroll_primary_s{seed}"]
                   for seed in (42, 43, 44)]
        self.assertEqual([self.configs[item["config_run"]]["training"]["seed"]
                          for item in entries], [42, 43, 44])
        self.assertTrue(all(item["method"] == "rsroll_native_attn_cam_r16"
                            for item in entries))
        self.assertEqual(len({item["required_env"] for item in entries}), 3)
        self.assertEqual(len({item["adapter_env"] for item in entries}), 1)

    def test_plan_mutation_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.plan)
        mutated["runs"]["iclr_primary_native_attn_cam_r16_s43"]["train_seed"] = 99
        with mock.patch.object(planner, "_load_json", return_value=mutated), \
                self.assertRaises(planner.PlanError):
            planner.load_and_validate(rswm_base=RSWM_BASE)

    def test_cli_validation_is_cpu_plan_only(self) -> None:
        run = subprocess.run(
            [sys.executable, str(PLAN_TOOL), "--rswm-base", RSWM_BASE, "validate"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("PLAN_VALID", run.stdout)
        self.assertIn("GPU_ACCESSED=0", run.stdout)
        self.assertIn("TRAINING_STARTED=0", run.stdout)


class ICLRA800ShellSafetyTests(unittest.TestCase):
    def test_wrappers_default_to_plan_and_require_explicit_confirmation(self) -> None:
        directory = ROOT / "server_run" / "iclr_a800"
        expected = {
            "run_iclr_main_a800.sh": "RSWM_CONFIRM_FORMAL_RUN",
            "run_iclr_baselines_a800.sh": "RSWM_CONFIRM_BASELINE_RUN",
            "run_iclr_evaluation_a800.sh": "RSWM_CONFIRM_EVALUATION_RUN",
        }
        for name, confirmation in expected.items():
            text = (directory / name).read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertIn('MODE="plan"', text)
                self.assertIn(confirmation, text)
                self.assertIn("TRAINING_STARTED=0", text)

    def test_training_wrapper_delegates_to_existing_formal_runner(self) -> None:
        text = (ROOT / "server_run" / "iclr_a800" /
                "run_iclr_main_a800.sh").read_text(encoding="utf-8")
        self.assertIn("server_run/rswm_train_runner.sh", text)
        self.assertNotIn("lora_remotesense/new/train.py", text)
        self.assertIn("--dry-run", text)
        self.assertIn("READY_FOR_TRAINING", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
