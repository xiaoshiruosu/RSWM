#!/usr/bin/env python3
"""A800 80GB 配置的 CPU-only 回归测试。"""

from __future__ import annotations

import ast
import json
import logging
import os
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "lora_remotesense" / "new" / "train.py"
COMMON = ROOT / "server_run" / "common.sh"
PBS_ENV = ROOT / "server_run" / "pbs" / "_pbs_env.sh"
A800_ENV = ROOT / "server_run" / "a800_env.sh"
RUN_TRAIN = ROOT / "lora_remotesense" / "new" / "run_train.sh"
ATTENTION = ROOT / "wan" / "modules" / "attention.py"
PREFLIGHT = ROOT / "server_run" / "preflight_a800.py"
SMOKE = ROOT / "server_run" / "run_a800_smoke.sh"
LOCK = ROOT / "requirements-a800-verified.txt"
SCRIPT_ENTRY = ROOT / "scripts" / "preflight_a800.py"
TEST_ENTRY = ROOT / "tests" / "test_a800_smoke.py"


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"PASS {name}")


def load_function(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {name}
    if name == "_apply_hardware_profile":
        wanted.update({"_load_hardware_profile_spec", "_resolve_offload_mode"})
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in wanted]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {
        "HERE": path.parent,
        "json": json,
        "os": os,
        "torch": __import__("torch"),
        "logger": logging.getLogger("a800-profile-test"),
    }
    exec(compile(module, str(path), "exec"), ns)
    return ns[name]


def args(**overrides):
    values = dict(
        hardware_profile="a800_80g",
        low_mem=False,
        t5_cpu=False,
        vae_cpu=False,
        gradient_checkpointing=False,
        mixed_precision="no",
        offload="auto",
        inactive_model_cpu=None,
        use_8bit_adam=True,
        gradient_accumulation_steps=4,
        batch_size=1,
        num_workers=None,
        prefetch_factor=None,
        persistent_workers=False,
        attention_backend=None,
        device_id=None,
    )
    values.update(overrides)
    return Namespace(**values)


def main() -> None:
    apply_profile = load_function(TRAIN, "_apply_hardware_profile")
    a = apply_profile(args())
    check("profile_forces_bf16", a.mixed_precision == "bf16")
    check("profile_forces_cpu_conditioners", a.t5_cpu and a.vae_cpu)
    check("profile_forces_checkpointing", a.gradient_checkpointing)
    check("profile_auto_uses_safe_cpu_default",
          a.offload_requested == "auto"
          and a.offload_resolved == "1"
          and a.inactive_model_cpu)
    check("profile_disables_bnb", not a.use_8bit_adam)
    check("profile_preserves_accum", a.gradient_accumulation_steps == 4)

    resident = apply_profile(args(offload="0"))
    check("profile_allows_explicit_resident_test",
          resident.offload_requested == "0"
          and resident.offload_resolved == "0"
          and not resident.inactive_model_cpu
          and not resident.use_8bit_adam)

    class _Props:
        def __init__(self, total_mib):
            self.total_memory = total_mib * 1024 ** 2

    class _FakeCuda:
        def __init__(self, total_mib, free_mib):
            self.total_mib = total_mib
            self.free_mib = free_mib

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        def get_device_properties(self, _device):
            return _Props(self.total_mib)

        def mem_get_info(self, _device):
            return self.free_mib * 1024 ** 2, self.total_mib * 1024 ** 2

    class _FakeTorch:
        def __init__(self, total_mib, free_mib):
            self.cuda = _FakeCuda(total_mib, free_mib)

    real_torch = apply_profile.__globals__["torch"]
    try:
        apply_profile.__globals__["torch"] = _FakeTorch(81920, 81000)
        auto_a800 = apply_profile(args())
        check("auto_a800_resolves_offload",
              auto_a800.offload_resolved == "1"
              and auto_a800.inactive_model_cpu)

        apply_profile.__globals__["torch"] = _FakeTorch(143000, 140000)
        auto_h200 = apply_profile(args(hardware_profile="h200_141g"))
        check("auto_h200_resolves_resident",
              auto_h200.offload_resolved == "0"
              and not auto_h200.inactive_model_cpu
              and auto_h200.use_8bit_adam)

        apply_profile.__globals__["torch"] = _FakeTorch(143000, 50000)
        busy_h200 = apply_profile(args(hardware_profile="h200_141g"))
        check("auto_busy_h200_resolves_offload",
              busy_h200.offload_resolved == "1"
              and busy_h200.inactive_model_cpu
              and not busy_h200.use_8bit_adam)
    finally:
        apply_profile.__globals__["torch"] = real_torch

    try:
        apply_profile(args(batch_size=2))
    except ValueError as exc:
        check("batch_gt1_rejected", "batch_size 1" in str(exc))
    else:
        raise AssertionError("FAIL batch_gt1_rejected")

    try:
        apply_profile(args(hardware_profile="auto", inactive_model_cpu=True,
                           use_8bit_adam=True))
    except ValueError as exc:
        check("offload_bnb_rejected", "AdamW8bit" in str(exc))
    else:
        raise AssertionError("FAIL offload_bnb_rejected")

    train = TRAIN.read_text(encoding="utf-8")
    common = COMMON.read_text(encoding="utf-8")
    pbs = PBS_ENV.read_text(encoding="utf-8")
    a800_env = A800_ENV.read_text(encoding="utf-8")
    launcher = RUN_TRAIN.read_text(encoding="utf-8")
    attention = ATTENTION.read_text(encoding="utf-8")
    preflight = PREFLIGHT.read_text(encoding="utf-8")
    smoke = SMOKE.read_text(encoding="utf-8")
    lock = LOCK.read_text(encoding="utf-8")

    check("optimizer_state_migrates",
          "def _move_optimizer_state" in train
          and "self._move_optimizer_state(self.low_noise_model" in train
          and "self._move_optimizer_state(self.high_noise_model" in train)
    check("experts_offloaded_before_conditioners",
          "if self.args.inactive_model_cpu:\n            self._offload_all_experts()" in train)
    check("common_passes_profile",
          'TRAIN_HARDWARE_FLAG=(--hardware_profile "$HARDWARE_PROFILE")' in common)
    check("common_supports_offload_auto",
          'OFFLOAD_REQUESTED="${OFFLOAD:-auto}"' in common
          and 'OFFLOAD_RESOLVED=1' in common
          and 'TRAIN_OFFLOAD_FLAG=(--offload "$OFFLOAD_RESOLVED")' in common)
    check("common_disables_bnb_for_offload",
          "TRAIN_OPTIMIZER_FLAG=(--no_8bit_adam)" in common)
    check("pbs_preserves_h200_auto_path",
          'HARDWARE_PROFILE="${HARDWARE_PROFILE:-auto}"' in pbs
          and 'OFFLOAD="${OFFLOAD:-}"' in pbs
          and 'ACCUM="${ACCUM:-4}"' in pbs)
    check("shared_a800_env_is_explicit",
          'HARDWARE_PROFILE="${HARDWARE_PROFILE:-a800_80g}"' in a800_env
          and 'OFFLOAD="${OFFLOAD:-auto}"' in a800_env
          and 'auto|0|1' in a800_env
          and 'NUM_WORKERS="${NUM_WORKERS:-0}"' in a800_env
          and 'RSWM_BASE:-/nix/rsroll_25828' in a800_env
          and "export CUDA_VISIBLE_DEVICES" not in a800_env)
    check("launcher_defaults_33_frames",
          'MAX_FRAMES="${MAX_FRAMES:-33}"' in launcher)
    check("launcher_passes_a800_profile",
          '--hardware_profile "$HARDWARE_PROFILE"' in launcher
          and "--no_8bit_adam" in launcher)
    check("real_attention_entry_dispatches",
          "def _flash_attention_impl(" in attention
          and "def flash_attention(" in attention
          and "backend, selected_fa_version = _select_backend(q, version)" in attention
          and "falling back to PyTorch SDPA" in attention)
    check("preflight_checks_exact_cuda_stack",
          'torch.__version__ != "2.4.0+cu121"' in preflight
          and 'torch.version.cuda != "12.1"' in preflight
          and 'platform.python_version() != "3.10.20"' in preflight)
    check("smoke_caps_train_and_validation",
          "--max_train_samples 1 --max_val_samples 1" in smoke
          and "--max_frames 33" in smoke
          and "export OFFLOAD=1" in smoke)
    check("standard_deployment_entrypoints",
          SCRIPT_ENTRY.is_file() and TEST_ENTRY.is_file()
          and "server_run" in SCRIPT_ENTRY.read_text(encoding="utf-8")
          and "server_run" in TEST_ENTRY.read_text(encoding="utf-8"))
    for requirement in (
            "torch==2.4.0", "torchvision==0.19.0", "flash-attn==2.7.3",
            "transformers==4.51.3", "tokenizers==0.21.4",
            "diffusers==0.39.0", "accelerate==1.14.0", "peft==0.19.1",
            "numpy==1.26.4", "huggingface-hub==0.36.2"):
        check(f"lock_{requirement.split('==')[0]}", requirement in lock)

    print("A800 profile regression tests: all passed")


if __name__ == "__main__":
    main()
