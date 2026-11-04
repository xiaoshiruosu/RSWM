#!/usr/bin/env python3
"""Strict experiment config, manifest, command, and resume utilities.

This module intentionally uses only the Python standard library.  The verified
A800 environment is not modified and does not need an additional YAML package.
The accepted YAML subset is deliberately small: nested mappings and scalar
values.  That is sufficient for the checked-in experiment configs and makes
unknown/ambiguous syntax fail closed instead of being guessed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.yaml"
CONFIG_SNAPSHOT_NAME = "config.yaml"
EXPERIMENT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,79}$")
CHECKPOINT_NAME_RE = re.compile(r"^checkpoint-epoch-([0-9]+)(?:_[0-9]{8}_[0-9]{6})?$")
ALLOWED_LORA_TARGETS = {"all", "attn_cam", "attn_only"}
ALLOWED_POSE_MODES = {"native", "identity", "drop"}
ALLOWED_DATASET_TIERS = {"aid", "resisc45", "whurs19", "mixed"}
ALLOWED_HARDWARE_PROFILES = {"auto", "a800_80g", "h200_141g"}
ALLOWED_ATTENTION_BACKENDS = {"auto", "flash_attn", "sdpa"}


class ConfigError(ValueError):
    """Experiment configuration is invalid or ambiguous."""


def _strip_comment(line: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(line):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None:
            return line[:index]
    return line


def _parse_scalar(value: str, *, line_number: int) -> Any:
    value = value.strip()
    if not value:
        raise ConfigError(f"line {line_number}: missing scalar value")
    lowered = value.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.startswith(('"', "'")):
        if len(value) < 2 or value[-1] != value[0]:
            raise ConfigError(f"line {line_number}: unterminated quoted scalar")
        if value[0] == '"':
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise ConfigError(
                    f"line {line_number}: invalid quoted scalar: {exc}"
                ) from exc
        return value[1:-1].replace("''", "'")
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"line {line_number}: inline collections must use JSON syntax"
            ) from exc
    if re.fullmatch(r"[-+]?(0|[1-9][0-9]*)", value):
        return int(value)
    if re.fullmatch(
        r"[-+]?(?:[0-9]+\.[0-9]*|\.[0-9]+|[0-9]+(?:[eE][-+]?[0-9]+)|"
        r"[0-9]+\.[0-9]*(?:[eE][-+]?[0-9]+))",
        value,
    ):
        return float(value)
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load the strict mapping-only YAML subset used by experiment configs."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read config {source}: {exc}") from exc
    if text.lstrip().startswith("{"):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid JSON-compatible YAML: {exc}") from exc
        if not isinstance(value, dict):
            raise ConfigError("config root must be a mapping")
        return value

    records: list[tuple[int, str, str, int]] = []
    for number, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw:
            raise ConfigError(f"line {number}: tabs are forbidden in YAML indentation")
        clean = _strip_comment(raw).rstrip()
        if not clean.strip() or clean.lstrip().startswith("---"):
            continue
        indent = len(clean) - len(clean.lstrip(" "))
        if indent % 2:
            raise ConfigError(f"line {number}: indentation must use multiples of two")
        body = clean[indent:]
        if body.startswith("-"):
            raise ConfigError(
                f"line {number}: block lists are unsupported; use an inline JSON list"
            )
        if ":" not in body:
            raise ConfigError(f"line {number}: expected 'key: value'")
        key, value = body.split(":", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
            raise ConfigError(f"line {number}: invalid mapping key {key!r}")
        records.append((indent, key, value.strip(), number))

    if not records:
        raise ConfigError(f"config {source} is empty")

    def parse_block(index: int, expected_indent: int) -> tuple[dict[str, Any], int]:
        result: dict[str, Any] = {}
        while index < len(records):
            indent, key, value, number = records[index]
            if indent < expected_indent:
                break
            if indent > expected_indent:
                raise ConfigError(f"line {number}: unexpected indentation")
            if key in result:
                raise ConfigError(f"line {number}: duplicate key {key!r}")
            if value:
                result[key] = _parse_scalar(value, line_number=number)
                index += 1
                continue
            if index + 1 >= len(records) or records[index + 1][0] <= indent:
                raise ConfigError(f"line {number}: mapping {key!r} has no child values")
            child_indent = records[index + 1][0]
            if child_indent != indent + 2:
                raise ConfigError(
                    f"line {records[index + 1][3]}: child indentation must increase by two"
                )
            child, index = parse_block(index + 1, child_indent)
            result[key] = child
        return result, index

    root, consumed = parse_block(0, records[0][0])
    if records[0][0] != 0:
        raise ConfigError("top-level YAML keys must not be indented")
    if consumed != len(records):
        raise ConfigError(f"line {records[consumed][3]}: could not parse YAML")
    return root


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value), ensure_ascii=False)


def dump_yaml(value: Any, *, indent: int = 0) -> str:
    """Serialize mappings/lists as conservative, human-readable YAML."""
    prefix = " " * indent
    lines: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, Mapping):
                lines.append(f"{prefix}{key}:")
                lines.append(dump_yaml(item, indent=indent + 2).rstrip("\n"))
            elif isinstance(item, list):
                # Inline JSON lists are valid YAML and keep the strict parser
                # dependency-free.  Commands/attempt history remain exact.
                lines.append(
                    f"{prefix}{key}: {json.dumps(item, ensure_ascii=False)}"
                )
            else:
                lines.append(f"{prefix}{key}: {_yaml_scalar(item)}")
    elif isinstance(value, list):
        lines.append(f"{prefix}{json.dumps(value, ensure_ascii=False)}")
    else:
        lines.append(f"{prefix}{_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def atomic_write(path: str | Path, text: str, *, mode: int = 0o600) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.chmod(temporary, mode)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def mkdir_private(path: str | Path, *, parents: bool = False) -> Path:
    """Create a private directory without Windows' mode-specific ACL behavior."""
    target = Path(path)
    target.mkdir(parents=parents)
    if os.name != "nt":
        os.chmod(target, 0o700)
    return target


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return dict(value)


def _require_keys(section: str, values: Mapping[str, Any], keys: Iterable[str]) -> None:
    missing = [key for key in keys if key not in values]
    if missing:
        raise ConfigError(f"{section} is missing required keys: {', '.join(missing)}")


def _check_unknown(section: str, values: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigError(f"{section} contains unknown keys: {', '.join(unknown)}")


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"{name} must be a positive number")
    return float(value)


def _path_value(value: Any, name: str, project_root: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty path string")
    raw = os.path.expandvars(os.path.expanduser(value.strip()))
    unresolved = re.search(r"\$(?:\{[A-Za-z_][A-Za-z0-9_]*\}|[A-Za-z_][A-Za-z0-9_]*)", raw)
    if unresolved:
        raise ConfigError(
            f"{name} contains unresolved environment variable {unresolved.group(0)}"
        )
    # Formal configs target the Linux A800 server. Preserve POSIX absolute
    # paths when validating or dry-running the portable bundle on Windows.
    if raw.startswith("/"):
        return raw
    expanded = Path(raw)
    if not expanded.is_absolute():
        expanded = project_root / expanded
    return str(expanded.resolve(strict=False))


def validate_config(raw: Mapping[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    """Validate and normalize the formal train/evaluation config."""
    root = Path(project_root).resolve()
    allowed_top = {"experiment", "dataset", "model", "lora", "training", "evaluation", "hardware"}
    _check_unknown("config", raw, allowed_top)
    if set(raw) != allowed_top:
        missing = sorted(allowed_top - set(raw))
        raise ConfigError(f"config is missing sections: {', '.join(missing)}")

    experiment = _mapping(raw, "experiment")
    dataset = _mapping(raw, "dataset")
    model = _mapping(raw, "model")
    lora = _mapping(raw, "lora")
    training = _mapping(raw, "training")
    evaluation = _mapping(raw, "evaluation")
    hardware = _mapping(raw, "hardware")

    _check_unknown("experiment", experiment, {"name", "description", "role"})
    _check_unknown("dataset", dataset, {"train", "validation", "tier", "pose_mode"})
    _check_unknown("model", model, {"checkpoint"})
    _check_unknown("lora", lora, {"rank", "alpha", "target_modules"})
    _check_unknown(
        "training", training,
        {"batch_size", "learning_rate", "num_epochs", "max_steps", "save_interval",
         "max_frames", "gradient_accumulation_steps", "seed", "split_seed",
         "max_train_samples", "max_val_samples"},
    )
    _check_unknown("evaluation", evaluation, {"enabled", "max_samples", "gen_seed"})
    _check_unknown(
        "hardware", hardware,
        {"gpu", "profile", "offload", "attention_backend", "min_free_gib", "min_disk_free_gib"},
    )
    _require_keys("experiment", experiment, {"name"})
    _require_keys("dataset", dataset, {"train", "validation", "tier", "pose_mode"})
    _require_keys("model", model, {"checkpoint"})
    _require_keys("lora", lora, {"rank", "target_modules"})
    _require_keys(
        "training", training,
        {"batch_size", "learning_rate", "num_epochs", "max_steps", "save_interval"},
    )
    _require_keys("evaluation", evaluation, {"enabled"})
    _require_keys("hardware", hardware, {"gpu"})

    name = experiment["name"]
    if not isinstance(name, str) or not EXPERIMENT_NAME_RE.fullmatch(name):
        raise ConfigError(
            "experiment.name must match [a-z0-9][a-z0-9_-]{1,79}"
        )
    role = experiment.get("role", "experiment")
    if not isinstance(role, str) or not role.strip():
        raise ConfigError("experiment.role must be a non-empty string")

    tier = dataset["tier"]
    if tier not in ALLOWED_DATASET_TIERS:
        raise ConfigError(f"dataset.tier must be one of {sorted(ALLOWED_DATASET_TIERS)}")
    pose_mode = dataset["pose_mode"]
    if pose_mode not in ALLOWED_POSE_MODES:
        raise ConfigError(f"dataset.pose_mode must be one of {sorted(ALLOWED_POSE_MODES)}")

    rank = _positive_int(lora["rank"], "lora.rank")
    alpha = _positive_int(lora.get("alpha", rank), "lora.alpha")
    targets = lora["target_modules"]
    if targets not in ALLOWED_LORA_TARGETS:
        raise ConfigError(
            f"lora.target_modules must be one of {sorted(ALLOWED_LORA_TARGETS)}"
        )

    batch_size = _positive_int(training["batch_size"], "training.batch_size")
    if batch_size != 1:
        raise ConfigError("the verified A800 profile requires training.batch_size: 1")
    max_steps = training["max_steps"]
    if max_steps is not None:
        raise ConfigError(
            "training.max_steps must be null: the real train.py has no --max_steps; "
            "use num_epochs without changing the training algorithm"
        )
    save_interval = training["save_interval"]
    if save_interval != 1:
        raise ConfigError(
            "training.save_interval must be 1: the real train.py saves every epoch"
        )

    eval_enabled = evaluation["enabled"]
    if not isinstance(eval_enabled, bool):
        raise ConfigError("evaluation.enabled must be true or false")

    gpu = hardware["gpu"]
    if isinstance(gpu, int) and not isinstance(gpu, bool):
        if gpu < 0:
            raise ConfigError("hardware.gpu must be auto or a non-negative physical GPU id")
        normalized_gpu: str | int = gpu
    elif gpu == "auto":
        normalized_gpu = "auto"
    else:
        raise ConfigError("hardware.gpu must be auto or a non-negative physical GPU id")
    profile = hardware.get("profile", "a800_80g")
    if profile not in ALLOWED_HARDWARE_PROFILES:
        raise ConfigError(f"hardware.profile must be one of {sorted(ALLOWED_HARDWARE_PROFILES)}")
    backend = hardware.get("attention_backend", "auto")
    if backend not in ALLOWED_ATTENTION_BACKENDS:
        raise ConfigError(
            f"hardware.attention_backend must be one of {sorted(ALLOWED_ATTENTION_BACKENDS)}"
        )
    offload = str(hardware.get("offload", "1"))
    if offload not in {"auto", "0", "1"}:
        raise ConfigError("hardware.offload must be auto, 0, or 1")

    normalized = {
        "experiment": {
            "name": name,
            "description": str(experiment.get("description", "")),
            "role": role,
        },
        "dataset": {
            "train": _path_value(dataset["train"], "dataset.train", root),
            "validation": _path_value(dataset["validation"], "dataset.validation", root),
            "tier": tier,
            "pose_mode": pose_mode,
        },
        "model": {
            "checkpoint": _path_value(model["checkpoint"], "model.checkpoint", root),
        },
        "lora": {
            "rank": rank,
            "alpha": alpha,
            "target_modules": targets,
        },
        "training": {
            "batch_size": batch_size,
            "learning_rate": _positive_float(training["learning_rate"], "training.learning_rate"),
            "num_epochs": _positive_int(training["num_epochs"], "training.num_epochs"),
            "max_steps": None,
            "save_interval": 1,
            "max_frames": _positive_int(training.get("max_frames", 33), "training.max_frames"),
            "gradient_accumulation_steps": _positive_int(
                training.get("gradient_accumulation_steps", 4),
                "training.gradient_accumulation_steps",
            ),
            "seed": _positive_int(training.get("seed", 42), "training.seed"),
            "split_seed": _positive_int(training.get("split_seed", 42), "training.split_seed"),
            "max_train_samples": _optional_positive_int(
                training.get("max_train_samples"), "training.max_train_samples"
            ),
            "max_val_samples": _optional_positive_int(
                training.get("max_val_samples"), "training.max_val_samples"
            ),
        },
        "evaluation": {
            "enabled": eval_enabled,
            "max_samples": _optional_positive_int(
                evaluation.get("max_samples"), "evaluation.max_samples"
            ),
            "gen_seed": _positive_int(evaluation.get("gen_seed", 42), "evaluation.gen_seed"),
        },
        "hardware": {
            "gpu": normalized_gpu,
            "profile": profile,
            "offload": offload,
            "attention_backend": backend,
            "min_free_gib": _positive_float(
                hardware.get("min_free_gib", 70), "hardware.min_free_gib"
            ),
            "min_disk_free_gib": _positive_float(
                hardware.get("min_disk_free_gib", 50), "hardware.min_disk_free_gib"
            ),
        },
    }
    if normalized["training"]["max_frames"] % 4 != 1:
        raise ConfigError("training.max_frames must satisfy 4n+1 for the formal video protocol")
    return normalized


def load_config(path: str | Path, *, project_root: str | Path) -> dict[str, Any]:
    return validate_config(load_yaml(path), project_root=project_root)


def experiment_id_for(config: Mapping[str, Any], requested: str | None = None) -> str:
    value = requested or f"{dt.datetime.now().strftime('%Y%m%d')}_{config['experiment']['name']}"
    if not EXPERIMENT_NAME_RE.fullmatch(value):
        raise ConfigError(f"invalid experiment id {value!r}")
    return value


def resolve_gpu_request(cli_gpu: str | None, config_gpu: str | int) -> str:
    requested = cli_gpu if cli_gpu is not None else str(config_gpu)
    if requested == "auto":
        return requested
    if not re.fullmatch(r"[0-9]+", requested):
        raise ConfigError("--gpu must be a non-negative physical GPU id")
    return str(int(requested))


def assert_gpu_allowed(physical_gpu: str, original_visibility: str | None) -> None:
    if not original_visibility:
        return
    values = [value.strip() for value in original_visibility.split(",") if value.strip()]
    if not values or any(not re.fullmatch(r"[0-9]+", value) for value in values):
        raise ConfigError(
            "pre-existing CUDA_VISIBLE_DEVICES must contain numeric physical ids"
        )
    if physical_gpu not in values:
        raise ConfigError(
            f"physical GPU {physical_gpu} is outside pre-existing CUDA_VISIBLE_DEVICES="
            f"{original_visibility}"
        )


def build_train_command(
    config: Mapping[str, Any], *, python: str, project_root: str | Path,
    experiment_dir: str | Path, resume_from: str | None = None,
) -> list[str]:
    root = Path(project_root).resolve()
    exp = Path(experiment_dir).resolve()
    train = config["training"]
    command = [
        python,
        str(root / "lora_remotesense" / "new" / "train.py"),
        "--checkpoint_dir", config["model"]["checkpoint"],
        "--data_path", config["dataset"]["train"],
        "--split_file", config["dataset"]["validation"],
        "--output_dir", str(exp / "checkpoints"),
        "--num_epochs", str(train["num_epochs"]),
        "--batch_size", str(train["batch_size"]),
        "--learning_rate", str(train["learning_rate"]),
        "--lora_rank", str(config["lora"]["rank"]),
        "--lora_alpha", str(config["lora"]["alpha"]),
        "--lora_targets", config["lora"]["target_modules"],
        "--pose_mode", config["dataset"]["pose_mode"],
        "--max_frames", str(train["max_frames"]),
        "--gradient_accumulation_steps", str(train["gradient_accumulation_steps"]),
        "--seed", str(train["seed"]),
        "--split_seed", str(train["split_seed"]),
        "--hardware_profile", config["hardware"]["profile"],
        "--offload", config["hardware"]["offload"],
        "--attention_backend", config["hardware"]["attention_backend"],
        "--mixed_precision", "bf16",
        "--gradient_checkpointing",
        "--no_8bit_adam",
        "--device_id", "0",
    ]
    if train["max_train_samples"] is not None:
        command.extend(["--max_train_samples", str(train["max_train_samples"])])
    if train["max_val_samples"] is not None:
        command.extend(["--max_val_samples", str(train["max_val_samples"])])
    if resume_from:
        command.extend(["--resume_from", str(Path(resume_from).resolve())])
    return command


def build_evaluation_command(
    config: Mapping[str, Any], *, python: str, project_root: str | Path,
    experiment_dir: str | Path,
) -> list[str]:
    root = Path(project_root).resolve()
    exp = Path(experiment_dir).resolve()
    command = [
        python,
        str(root / "lora_remotesense" / "new" / "evaluate.py"),
        "--checkpoint_dir", config["model"]["checkpoint"],
        "--lora_path", str(exp / "checkpoints" / "best_checkpoint"),
        "--data_path", config["dataset"]["train"],
        "--split_file", config["dataset"]["validation"],
        "--output_dir", str(exp / "results"),
        "--frame_num", str(config["training"]["max_frames"]),
        "--device", "cuda",
        "--gen_seed", str(config["evaluation"]["gen_seed"]),
        "--inactive_model_cpu",
    ]
    if config["evaluation"]["max_samples"] is not None:
        command.extend(["--max_samples", str(config["evaluation"]["max_samples"])])
    return command


def _git_commit(project_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "--show-toplevel", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "NOT_AVAILABLE"
    values = completed.stdout.splitlines()
    if completed.returncode != 0 or len(values) != 2:
        return "NOT_AVAILABLE"
    try:
        repository_root = Path(values[0]).resolve()
    except OSError:
        return "NOT_AVAILABLE"
    # Do not accidentally record a commit from an unrelated parent checkout
    # when this portable server bundle itself has no Git metadata.
    if repository_root != project_root.resolve():
        return "NOT_AVAILABLE"
    return values[1].strip() or "NOT_AVAILABLE"


def _runtime_versions(python_path: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_path": python_path,
        "python_version": platform.python_version(),
        "torch_version": "NOT_AVAILABLE",
        "cuda_version": "NOT_AVAILABLE",
    }
    try:
        import torch
        result["torch_version"] = str(torch.__version__)
        result["cuda_version"] = str(torch.version.cuda)
    except Exception as exc:  # Import diagnostics belong in the manifest.
        result["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def build_manifest(
    config: Mapping[str, Any], *, config_path: str | Path,
    experiment_id: str, experiment_dir: str | Path, project_root: str | Path,
    python_path: str, requested_gpu: str, physical_gpu: str | None,
    cuda_visible_devices: str | None, status: str, dry_run: bool,
    resume_from: str | None = None,
) -> dict[str, Any]:
    config_source = Path(config_path).resolve()
    exp = Path(experiment_dir).resolve()
    command = build_train_command(
        config, python=python_path, project_root=project_root,
        experiment_dir=exp, resume_from=resume_from,
    )
    eval_command = build_evaluation_command(
        config, python=python_path, project_root=project_root, experiment_dir=exp,
    )
    created = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "experiment_id": experiment_id,
        "timestamp": created,
        "status": status,
        "dry_run": bool(dry_run),
        "hostname": socket.gethostname(),
        "git_commit": _git_commit(Path(project_root).resolve()),
        "runtime": _runtime_versions(python_path),
        "gpu": {
            "requested": requested_gpu,
            "physical_gpu_id": physical_gpu,
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "process_local_device": ("cuda:" + "0") if physical_gpu is not None else None,
        },
        "model": {"checkpoint": config["model"]["checkpoint"]},
        "dataset": {
            "train": config["dataset"]["train"],
            "validation": config["dataset"]["validation"],
            "tier": config["dataset"]["tier"],
            "pose_mode": config["dataset"]["pose_mode"],
        },
        "lora": dict(config["lora"]),
        "training": {
            "batch_size": config["training"]["batch_size"],
            "lr": config["training"]["learning_rate"],
            "steps": config["training"]["max_steps"],
            "num_epochs": config["training"]["num_epochs"],
            "save_interval_epochs": config["training"]["save_interval"],
            "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
            "seed": config["training"]["seed"],
            "split_seed": config["training"]["split_seed"],
            "command": command,
        },
        "evaluation": {
            **config["evaluation"],
            "command": eval_command,
        },
        "hardware": dict(config["hardware"]),
        "experiment": dict(config["experiment"]),
        "paths": {
            "experiment_dir": str(exp),
            "logs": str(exp / "logs"),
            "checkpoints": str(exp / "checkpoints"),
            "results": str(exp / "results"),
            "config_snapshot": str(exp / CONFIG_SNAPSHOT_NAME),
        },
        "config": {
            "source": str(config_source),
            "sha256": sha256_file(config_source),
        },
        "resume": {
            "requested": resume_from is not None,
            "checkpoint": str(Path(resume_from).resolve()) if resume_from else None,
        },
        "attempts": [{
            "timestamp": created,
            "kind": "resume" if resume_from else "new",
            "status": status,
            "checkpoint": str(Path(resume_from).resolve()) if resume_from else None,
        }],
    }
    return manifest


def read_manifest(experiment_dir: str | Path) -> dict[str, Any]:
    path = Path(experiment_dir) / MANIFEST_NAME
    value = load_yaml(path)
    if not isinstance(value, dict):
        raise ConfigError(f"manifest root is invalid: {path}")
    return value


def create_or_resume_manifest(
    config: Mapping[str, Any], *, config_path: str | Path,
    experiment_id: str, experiments_root: str | Path, project_root: str | Path,
    python_path: str, requested_gpu: str, physical_gpu: str | None,
    cuda_visible_devices: str | None, status: str, dry_run: bool,
    resume_from: str | None = None, resume: bool = False,
) -> Path:
    root = Path(experiments_root).resolve()
    if not EXPERIMENT_NAME_RE.fullmatch(experiment_id):
        raise ConfigError(f"invalid experiment id {experiment_id!r}")
    requested_gpu = resolve_gpu_request(requested_gpu, "auto")
    if physical_gpu is not None:
        if not re.fullmatch(r"[0-9]+", str(physical_gpu)):
            raise ConfigError("physical GPU must be a non-negative integer")
        physical_gpu = str(int(str(physical_gpu)))
    exp = (root / experiment_id).resolve()
    if exp.parent != root:
        raise ConfigError("experiment id escapes the experiments root")
    source_config = Path(config_path).resolve()
    if resume:
        if not resume_from:
            raise ConfigError("resume requires a complete --resume-from checkpoint")
        resume_from = str(validate_resume_checkpoint(exp, resume_from))
        if not exp.is_dir():
            raise ConfigError(f"cannot resume missing experiment directory: {exp}")
        snapshot = exp / CONFIG_SNAPSHOT_NAME
        if not snapshot.is_file():
            raise ConfigError(f"cannot resume without config snapshot: {snapshot}")
        if sha256_file(snapshot) != sha256_file(source_config):
            raise ConfigError("resume config differs from the experiment config snapshot")
        manifest = read_manifest(exp)
        previous_id = manifest.get("experiment_id")
        if previous_id != experiment_id:
            raise ConfigError(
                f"manifest experiment_id={previous_id!r} does not match {experiment_id!r}"
            )
        manifest["status"] = status
        manifest["dry_run"] = bool(dry_run)
        manifest["gpu"] = {
            "requested": requested_gpu,
            "physical_gpu_id": physical_gpu,
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "process_local_device": ("cuda:" + "0") if physical_gpu is not None else None,
        }
        manifest["resume"] = {
            "requested": True,
            "checkpoint": str(Path(resume_from).resolve()) if resume_from else None,
        }
        now = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
        manifest.setdefault("attempts", []).append({
            "timestamp": now,
            "kind": "resume",
            "status": status,
            "checkpoint": str(Path(resume_from).resolve()) if resume_from else None,
        })
        manifest["training"]["command"] = build_train_command(
            config, python=python_path, project_root=project_root,
            experiment_dir=exp, resume_from=resume_from,
        )
    else:
        if resume_from is not None:
            raise ConfigError("--resume-from requires --resume")
        if exp.exists():
            raise ConfigError(
                f"experiment already exists: {exp}; use resume_training.sh instead"
            )
        mkdir_private(exp, parents=True)
        for child in ("logs", "checkpoints", "results"):
            mkdir_private(exp / child)
        shutil.copy2(source_config, exp / CONFIG_SNAPSHOT_NAME)
        os.chmod(exp / CONFIG_SNAPSHOT_NAME, 0o600)
        manifest = build_manifest(
            config,
            config_path=source_config,
            experiment_id=experiment_id,
            experiment_dir=exp,
            project_root=project_root,
            python_path=python_path,
            requested_gpu=requested_gpu,
            physical_gpu=physical_gpu,
            cuda_visible_devices=cuda_visible_devices,
            status=status,
            dry_run=dry_run,
            resume_from=resume_from,
        )
        # The copied snapshot is the durable resume source.
        manifest["config"]["snapshot_sha256"] = sha256_file(exp / CONFIG_SNAPSHOT_NAME)
    return atomic_write(exp / MANIFEST_NAME, dump_yaml(manifest))


def update_manifest_status(
    experiment_dir: str | Path, *, status: str, message: str | None = None,
    physical_gpu: str | None = None, cuda_visible_devices: str | None = None,
    exit_code: int | None = None,
) -> Path:
    exp = Path(experiment_dir).resolve()
    manifest = read_manifest(exp)
    now = dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")
    manifest["status"] = status
    manifest["last_update"] = now
    if message is not None:
        manifest["status_message"] = message
    if exit_code is not None:
        manifest["exit_code"] = int(exit_code)
    if physical_gpu is not None:
        manifest.setdefault("gpu", {})["physical_gpu_id"] = physical_gpu
        manifest["gpu"]["process_local_device"] = "cuda:" + "0"
    if cuda_visible_devices is not None:
        manifest.setdefault("gpu", {})["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return atomic_write(exp / MANIFEST_NAME, dump_yaml(manifest))


def checkpoint_complete(path: str | Path) -> tuple[bool, list[str]]:
    root = Path(path)
    missing: list[str] = []
    for required in ("training_state.pt", "training_config.json", "run_manifest.json"):
        if not (root / required).is_file():
            missing.append(required)
    for expert in ("low_noise", "high_noise"):
        adapter = root / expert / "adapter_model.safetensors"
        fallback = root / expert / "adapter_model.bin"
        if not adapter.is_file() and not fallback.is_file():
            missing.append(f"{expert}/adapter_model.(safetensors|bin)")
    return not missing, missing


def checkpoint_epoch(path: str | Path) -> int:
    match = CHECKPOINT_NAME_RE.fullmatch(Path(path).name)
    if not match:
        raise ConfigError(
            f"checkpoint directory must match {CHECKPOINT_NAME_RE.pattern}: {path}"
        )
    value = int(match.group(1))
    if value < 1:
        raise ConfigError(f"checkpoint epoch must be positive: {path}")
    return value


def validate_resume_checkpoint(
    experiment_dir: str | Path, checkpoint: str | Path,
) -> Path:
    exp = Path(experiment_dir).resolve()
    checkpoint_root = (exp / "checkpoints").resolve()
    candidate = Path(checkpoint).resolve()
    try:
        candidate.relative_to(checkpoint_root)
    except ValueError as exc:
        raise ConfigError(
            f"resume checkpoint must stay under {checkpoint_root}: {candidate}"
        ) from exc
    checkpoint_epoch(candidate)
    if not candidate.is_dir():
        raise ConfigError(f"resume checkpoint directory does not exist: {candidate}")
    complete, missing = checkpoint_complete(candidate)
    if not complete:
        raise ConfigError(
            f"resume checkpoint is incomplete: {candidate}; missing={', '.join(missing)}"
        )
    return candidate


def latest_complete_checkpoint(experiment_dir: str | Path) -> Path:
    root = Path(experiment_dir).resolve() / "checkpoints"
    candidates: list[tuple[int, int, Path]] = []
    if root.is_dir():
        for path in root.glob("checkpoint-epoch-*"):
            match = CHECKPOINT_NAME_RE.fullmatch(path.name)
            if path.is_dir() and match and checkpoint_complete(path)[0]:
                candidates.append((checkpoint_epoch(path), path.stat().st_mtime_ns, path))
    if not candidates:
        raise ConfigError(f"no complete checkpoint-epoch-* found under {root}")
    return max(candidates, key=lambda value: (value[0], value[1], value[2].name))[2]


def check_runtime_paths(
    config: Mapping[str, Any], *, experiments_root: str | Path,
    project_root: str | Path,
) -> list[str]:
    checks: list[str] = []
    data = Path(config["dataset"]["train"])
    split = Path(config["dataset"]["validation"])
    checkpoint = Path(config["model"]["checkpoint"])
    if not data.is_file():
        raise ConfigError(f"training dataset does not exist: {data}")
    checks.append(f"dataset={data}")
    if not split.is_file():
        raise ConfigError(f"validation split does not exist: {split}")
    try:
        split_doc = json.loads(split.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"validation split is not valid JSON: {split}: {exc}") from exc
    if isinstance(split_doc, dict) and all(
        isinstance(split_doc.get(key), list)
        for key in ("train_indices", "val_indices")
    ):
        train_values = split_doc["train_indices"]
        val_values = split_doc["val_indices"]
    elif isinstance(split_doc, dict) and isinstance(split_doc.get("splits"), dict):
        try:
            train_values = split_doc["splits"]["train"]["indices"]
            val_values = split_doc["splits"]["val"]["indices"]
        except (KeyError, TypeError) as exc:
            raise ConfigError(
                "official validation split must contain splits.train/val.indices"
            ) from exc
    else:
        raise ConfigError(
            "validation split must use train_indices/val_indices or "
            "splits.train/val.indices"
        )
    if not isinstance(train_values, list) or not isinstance(val_values, list):
        raise ConfigError("validation split train/val indices must be arrays")
    if not train_values or not val_values:
        raise ConfigError("validation split train and val indices must be non-empty")
    checks.append(f"validation_split={split}")
    if not checkpoint.is_dir():
        raise ConfigError(f"model checkpoint directory does not exist: {checkpoint}")
    expected = [
        checkpoint / "configuration.json",
        checkpoint / "Wan2.1_VAE.pth",
        checkpoint / "models_t5_umt5-xxl-enc-bf16.pth",
        checkpoint / "low_noise_model",
        checkpoint / "high_noise_model",
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise ConfigError("model checkpoint is incomplete: " + ", ".join(missing))
    checks.append(f"model_checkpoint={checkpoint}")
    training_entry = Path(project_root) / "lora_remotesense" / "new" / "train.py"
    evaluation_entry = Path(project_root) / "lora_remotesense" / "new" / "evaluate.py"
    for entry in (training_entry, evaluation_entry):
        if not entry.is_file():
            raise ConfigError(f"required project entry is missing: {entry}")
    experiment_root = Path(experiments_root).resolve()
    experiment_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(experiment_root)
    free_gib = usage.free / 1024 ** 3
    minimum = config["hardware"]["min_disk_free_gib"]
    if free_gib < minimum:
        raise ConfigError(
            f"insufficient disk space under {experiment_root}: "
            f"free={free_gib:.2f}GiB required={minimum:.2f}GiB"
        )
    checks.append(f"disk_free_gib={free_gib:.2f}")
    return checks


def plan_values(
    config: Mapping[str, Any], *, config_path: str | Path,
    experiment_id: str, experiments_root: str | Path,
) -> dict[str, str]:
    exp = Path(experiments_root).resolve() / experiment_id
    return {
        "PLAN_EXPERIMENT_ID": experiment_id,
        "PLAN_EXPERIMENT_DIR": str(exp),
        "PLAN_CONFIG_PATH": str(Path(config_path).resolve()),
        "PLAN_EXPERIMENT_NAME": config["experiment"]["name"],
        "PLAN_ROLE": config["experiment"]["role"],
        "PLAN_DATA_TRAIN": config["dataset"]["train"],
        "PLAN_DATA_VALIDATION": config["dataset"]["validation"],
        "PLAN_DATASET_TIER": config["dataset"]["tier"],
        "PLAN_POSE_MODE": config["dataset"]["pose_mode"],
        # SEM arm is a scientific identity field. It must come from the parsed
        # config, never from an experiment-name heuristic or a shell default.
        "PLAN_SEM_ARM_MODE": config["dataset"]["pose_mode"],
        "PLAN_MODEL_CHECKPOINT": config["model"]["checkpoint"],
        "PLAN_LORA_RANK": str(config["lora"]["rank"]),
        "PLAN_LORA_ALPHA": str(config["lora"]["alpha"]),
        "PLAN_LORA_TARGETS": config["lora"]["target_modules"],
        "PLAN_EVAL_ENABLED": "1" if config["evaluation"]["enabled"] else "0",
        "PLAN_GPU_REQUEST": str(config["hardware"]["gpu"]),
        "PLAN_MIN_FREE_GIB": str(config["hardware"]["min_free_gib"]),
        "PLAN_HARDWARE_PROFILE": config["hardware"]["profile"],
        "PLAN_OFFLOAD": config["hardware"]["offload"],
        "PLAN_ATTENTION_BACKEND": config["hardware"]["attention_backend"],
        "PLAN_PROMPT_SCHEMA": "net_continuous",
        "PLAN_PATH_ESTIMATOR": "registration",
        "PLAN_BACKBONE": "wan-inline",
    }


def write_shell_plan(path: str | Path, values: Mapping[str, str]) -> Path:
    text = "".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items())
    return atomic_write(path, text)


def _command_output(command: list[str], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(command, ensure_ascii=False))
        return
    if output_format == "shell":
        print(" ".join(shlex.quote(value) for value in command))
        return
    if output_format == "nul":
        sys.stdout.buffer.write(b"".join(value.encode("utf-8") + b"\0" for value in command))
        return
    raise ConfigError(f"unsupported command output format: {output_format}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def config_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--config", required=True)
        target.add_argument("--project-root", required=True)

    validate = sub.add_parser("validate", help="parse and validate one experiment config")
    config_args(validate)
    validate.add_argument("--check-paths", action="store_true")
    validate.add_argument("--experiments-root")

    plan = sub.add_parser("plan", help="resolve config and emit a shell plan")
    config_args(plan)
    plan.add_argument("--experiment-id")
    plan.add_argument("--experiments-root", required=True)
    plan.add_argument("--gpu")
    plan.add_argument("--output-shell", required=True)

    manifest = sub.add_parser("manifest", help="create or update manifest.yaml")
    config_args(manifest)
    manifest.add_argument("--experiment-id", required=True)
    manifest.add_argument("--experiments-root", required=True)
    manifest.add_argument("--python", required=True)
    manifest.add_argument("--requested-gpu", required=True)
    manifest.add_argument("--physical-gpu")
    manifest.add_argument("--cuda-visible-devices")
    manifest.add_argument("--status", default="PREPARED")
    manifest.add_argument("--dry-run", action="store_true")
    manifest.add_argument("--resume", action="store_true")
    manifest.add_argument("--resume-from")

    status = sub.add_parser("status", help="atomically update an existing manifest")
    status.add_argument("--experiment-dir", required=True)
    status.add_argument("--status", required=True)
    status.add_argument("--message")
    status.add_argument("--physical-gpu")
    status.add_argument("--cuda-visible-devices")
    status.add_argument("--exit-code", type=int)

    latest = sub.add_parser("latest-checkpoint", help="print newest complete epoch checkpoint")
    latest.add_argument("--experiment-dir", required=True)

    checkpoint = sub.add_parser("check-checkpoint", help="validate one native resume checkpoint")
    checkpoint.add_argument("--experiment-dir", required=True)
    checkpoint.add_argument("--checkpoint", required=True)

    field = sub.add_parser("field", help="print one dotted manifest field")
    field.add_argument("--experiment-dir", required=True)
    field.add_argument("--name", required=True)

    command = sub.add_parser("command", help="emit the exact train or evaluation argv")
    config_args(command)
    command.add_argument("--kind", choices=("train", "evaluation"), required=True)
    command.add_argument("--experiment-dir", required=True)
    command.add_argument("--python", required=True)
    command.add_argument("--resume-from")
    command.add_argument("--format", choices=("json", "shell", "nul"), default="json")

    gpu = sub.add_parser("check-gpu-request", help="validate GPU id against inherited visibility")
    gpu.add_argument("--gpu", required=True)
    gpu.add_argument("--visible-devices")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "status":
            path = update_manifest_status(
                args.experiment_dir, status=args.status, message=args.message,
                physical_gpu=args.physical_gpu,
                cuda_visible_devices=args.cuda_visible_devices,
                exit_code=args.exit_code,
            )
            print(path)
            return 0
        if args.command == "latest-checkpoint":
            print(latest_complete_checkpoint(args.experiment_dir))
            return 0
        if args.command == "check-checkpoint":
            print(validate_resume_checkpoint(args.experiment_dir, args.checkpoint))
            return 0
        if args.command == "field":
            value: Any = read_manifest(args.experiment_dir)
            for part in args.name.split("."):
                if not isinstance(value, dict) or part not in value:
                    raise ConfigError(f"manifest field is missing: {args.name}")
                value = value[part]
            if value is None or isinstance(value, (dict, list)):
                raise ConfigError(f"manifest field is not a scalar: {args.name}")
            if isinstance(value, bool):
                print("true" if value else "false")
            else:
                print(value)
            return 0
        if args.command == "check-gpu-request":
            if not re.fullmatch(r"[0-9]+", args.gpu):
                raise ConfigError("physical GPU must be a non-negative integer")
            assert_gpu_allowed(str(int(args.gpu)), args.visible_devices)
            print(str(int(args.gpu)))
            return 0

        config = load_config(args.config, project_root=args.project_root)
        if args.command == "validate":
            print(f"CONFIG_VALID name={config['experiment']['name']}")
            if args.check_paths:
                if not args.experiments_root:
                    raise ConfigError("--check-paths requires --experiments-root")
                for check in check_runtime_paths(
                    config, experiments_root=args.experiments_root,
                    project_root=args.project_root,
                ):
                    print(f"PASS {check}")
            return 0
        if args.command == "plan":
            experiment_id = experiment_id_for(config, args.experiment_id)
            requested_gpu = resolve_gpu_request(args.gpu, config["hardware"]["gpu"])
            values = plan_values(
                config, config_path=args.config, experiment_id=experiment_id,
                experiments_root=args.experiments_root,
            )
            values["PLAN_GPU_REQUEST"] = requested_gpu
            write_shell_plan(args.output_shell, values)
            print(json.dumps(values, ensure_ascii=False))
            return 0
        if args.command == "manifest":
            path = create_or_resume_manifest(
                config,
                config_path=args.config,
                experiment_id=args.experiment_id,
                experiments_root=args.experiments_root,
                project_root=args.project_root,
                python_path=args.python,
                requested_gpu=args.requested_gpu,
                physical_gpu=args.physical_gpu,
                cuda_visible_devices=args.cuda_visible_devices,
                status=args.status,
                dry_run=args.dry_run,
                resume_from=args.resume_from,
                resume=args.resume,
            )
            print(path)
            return 0
        if args.command == "command":
            if args.kind == "train":
                result = build_train_command(
                    config, python=args.python, project_root=args.project_root,
                    experiment_dir=args.experiment_dir,
                    resume_from=args.resume_from,
                )
            else:
                result = build_evaluation_command(
                    config, python=args.python, project_root=args.project_root,
                    experiment_dir=args.experiment_dir,
                )
            _command_output(result, args.format)
            return 0
        raise ConfigError(f"unhandled command {args.command}")
    except ConfigError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
