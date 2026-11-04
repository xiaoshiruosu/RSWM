#!/usr/bin/env python3
"""Validate and materialize the ICLR A800 experiment plan without launching work.

This module is deliberately CPU-only and standard-library-only.  It delegates every
training-config decision to the same strict parser used by ``rswm_train_runner.sh``.
It never imports torch, probes CUDA, creates a checkpoint, or starts a subprocess.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVER_RUN = PROJECT_ROOT / "server_run"
BASELINES = SERVER_RUN / "baselines"
DEFAULT_PLAN = PROJECT_ROOT / "configs" / "iclr_main" / "experiment_plan.json"
SCHEMA = "rsroll.iclr_main_a800_plan/v1"
RSWM_BASE_TOKEN = re.compile(r"\$\{RSWM_BASE\}")

sys.path.insert(0, str(SERVER_RUN))
import generate_manifest as formal_config  # noqa: E402


class PlanError(ValueError):
    """The preregistered run plan is incomplete or internally inconsistent."""


@contextlib.contextmanager
def _temporary_rswm_base(value: str) -> Iterator[None]:
    before = os.environ.get("RSWM_BASE")
    os.environ["RSWM_BASE"] = value
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("RSWM_BASE", None)
        else:
            os.environ["RSWM_BASE"] = before


def _project_path(value: str, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{field} must be a non-empty project-relative path")
    candidate = (PROJECT_ROOT / value).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise PlanError(f"{field} escapes the project root: {value}") from exc
    return candidate


def _expand_base(value: str, rswm_base: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlanError("runtime path must be a non-empty string")
    expanded = RSWM_BASE_TOKEN.sub(rswm_base, value)
    if re.search(r"\$\{[^}]+\}", expanded):
        raise PlanError(f"unsupported unresolved variable in runtime path: {value}")
    return expanded


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read plan {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanError("plan root must be an object")
    return value


def _field(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise PlanError(f"validated config is missing {dotted}")
        value = value[key]
    return value


FIXED_RECIPE_FIELDS = (
    "dataset.train", "dataset.validation", "dataset.tier",
    "model.checkpoint",
    "training.batch_size", "training.learning_rate", "training.num_epochs",
    "training.max_steps", "training.save_interval", "training.max_frames",
    "training.gradient_accumulation_steps", "training.split_seed",
    "training.max_train_samples", "training.max_val_samples",
    "evaluation.enabled", "evaluation.max_samples", "evaluation.gen_seed",
    "hardware.gpu", "hardware.profile", "hardware.offload",
    "hardware.attention_backend", "hardware.min_free_gib",
    "hardware.min_disk_free_gib",
)


def _run_config(plan: dict[str, Any], run_id: str, rswm_base: str) -> tuple[Path, dict[str, Any]]:
    runs = plan.get("runs")
    if not isinstance(runs, dict) or run_id not in runs:
        raise PlanError(f"unknown run id: {run_id}")
    run = runs[run_id]
    if not isinstance(run, dict):
        raise PlanError(f"runs.{run_id} must be an object")
    path = _project_path(run.get("config"), field=f"runs.{run_id}.config")
    if not path.is_file():
        raise PlanError(f"run config is missing: {path}")
    with _temporary_rswm_base(rswm_base):
        try:
            config = formal_config.load_config(path, project_root=PROJECT_ROOT)
        except formal_config.ConfigError as exc:
            raise PlanError(f"{run_id}: formal runner rejects {path}: {exc}") from exc
    return path, config


def group_run_ids(plan: dict[str, Any], group: str) -> list[str]:
    groups = plan.get("groups")
    if not isinstance(groups, dict):
        raise PlanError("groups must be an object")
    if group == "all":
        order = ("primary", "matched", "target_all", "rank")
        values: list[str] = []
        for name in order:
            for run_id in groups.get(name, []):
                if run_id not in values:
                    values.append(run_id)
        return values
    values = groups.get(group)
    if not isinstance(values, list) or not values:
        raise PlanError(f"unknown or empty run group: {group}")
    if not all(isinstance(value, str) for value in values):
        raise PlanError(f"groups.{group} must contain run-id strings")
    return list(values)


def baseline_ids(plan: dict[str, Any], group: str) -> list[str]:
    groups = plan.get("baseline_groups")
    if not isinstance(groups, dict):
        raise PlanError("baseline_groups must be an object")
    values = groups.get(group)
    if not isinstance(values, list) or not values:
        raise PlanError(f"unknown or empty baseline group: {group}")
    return list(values)


def load_and_validate(
    path: str | Path = DEFAULT_PLAN,
    *,
    rswm_base: str = "/nix/rsroll_25828",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Return a validated plan and its normalized formal-runner configs."""
    source = Path(path).resolve()
    plan = _load_json(source)
    errors: list[str] = []
    if plan.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if plan.get("status") != "CONFIGURED_NOT_RUN":
        errors.append("status must remain CONFIGURED_NOT_RUN before formal execution")
    if not rswm_base.startswith("/"):
        errors.append("RSWM_BASE must be an absolute POSIX path for the A800 server")

    runs = plan.get("runs")
    groups = plan.get("groups")
    if not isinstance(runs, dict) or not runs:
        errors.append("runs must be a non-empty object")
        runs = {}
    if not isinstance(groups, dict):
        errors.append("groups must be an object")
        groups = {}

    expected_groups = {"primary", "matched", "target_all", "rank"}
    if set(groups) != expected_groups:
        errors.append(f"groups must be exactly {sorted(expected_groups)}")
    referenced = []
    for name, ids in groups.items():
        if not isinstance(ids, list) or not ids:
            errors.append(f"groups.{name} must be a non-empty list")
            continue
        referenced.extend(ids)
        missing = [run_id for run_id in ids if run_id not in runs]
        if missing:
            errors.append(f"groups.{name} references unknown runs: {missing}")
    unreferenced = sorted(set(runs) - set(referenced))
    if unreferenced:
        errors.append(f"runs are not assigned to any group: {unreferenced}")

    configs: dict[str, dict[str, Any]] = {}
    paths: set[Path] = set()
    for run_id, run in runs.items():
        if not isinstance(run, dict):
            errors.append(f"runs.{run_id} must be an object")
            continue
        try:
            path_obj, config = _run_config(plan, run_id, rswm_base)
        except PlanError as exc:
            errors.append(str(exc))
            continue
        if path_obj in paths:
            errors.append(f"multiple canonical runs reuse config path {path_obj}")
        paths.add(path_obj)
        configs[run_id] = config
        expected = {
            "experiment.name": run_id,
            "experiment.role": run.get("role"),
            "dataset.pose_mode": run.get("pose_mode"),
            "lora.target_modules": run.get("target_modules"),
            "lora.rank": run.get("rank"),
            "training.seed": run.get("train_seed"),
        }
        for field, value in expected.items():
            if _field(config, field) != value:
                errors.append(
                    f"{run_id}: {field}={_field(config, field)!r}, expected {value!r}")
        if config["lora"]["alpha"] != config["lora"]["rank"]:
            errors.append(f"{run_id}: LoRA alpha must equal rank")
        a800_expected = {
            "hardware.profile": "a800_80g",
            "hardware.offload": "1",
            "hardware.min_free_gib": 70.0,
            "training.batch_size": 1,
            "training.gradient_accumulation_steps": 4,
            "training.max_frames": 33,
            "training.split_seed": 42,
            "evaluation.gen_seed": 42,
            "evaluation.max_samples": 400,
        }
        for field, value in a800_expected.items():
            if _field(config, field) != value:
                errors.append(f"{run_id}: {field} must be {value!r}")

    if configs:
        reference = configs.get("iclr_primary_native_attn_cam_r16_s42")
        if reference is None:
            errors.append("canonical seed-42 primary config is missing")
        else:
            for run_id, config in configs.items():
                for field in FIXED_RECIPE_FIELDS:
                    if _field(config, field) != _field(reference, field):
                        errors.append(
                            f"{run_id}: fixed recipe field {field} differs from primary s42")

    try:
        primary = group_run_ids(plan, "primary")
        primary_seeds = [configs[run_id]["training"]["seed"] for run_id in primary]
        if primary_seeds != [42, 43, 44]:
            errors.append(f"primary training seeds must be [42, 43, 44], got {primary_seeds}")
        for run_id in primary:
            cfg = configs[run_id]
            if (cfg["dataset"]["pose_mode"], cfg["lora"]["target_modules"],
                    cfg["lora"]["rank"]) != ("native", "attn_cam", 16):
                errors.append(f"{run_id}: invalid primary semantics")

        matched = group_run_ids(plan, "matched")
        cells = {(configs[item]["dataset"]["pose_mode"],
                  configs[item]["lora"]["target_modules"]) for item in matched}
        expected_cells = {
            ("native", "attn_only"), ("native", "attn_cam"),
            ("drop", "attn_only"), ("drop", "attn_cam"),
        }
        if cells != expected_cells:
            errors.append(f"matched group must be the full 2x2 design, got {sorted(cells)}")

        ranks = [configs[item]["lora"]["rank"] for item in group_run_ids(plan, "rank")]
        if ranks != [4, 8, 16, 32, 64]:
            errors.append(f"rank group must be [4, 8, 16, 32, 64], got {ranks}")
    except (PlanError, KeyError) as exc:
        errors.append(str(exc))

    protocol = plan.get("protocol", {})
    required_protocol = {
        "training_seeds": [42, 43, 44],
        "split_seed": 42,
        "generation_seed": 42,
        "frame_counts": [17, 25, 33],
        "condition_frames": 1,
        "headline_eval_n": 400,
        "gpu_profile": "a800_80g",
    }
    for key, value in required_protocol.items():
        if protocol.get(key) != value:
            errors.append(f"protocol.{key} must be {value!r}")

    manifests = plan.get("benchmark_manifests")
    if not isinstance(manifests, dict) or set(manifests) != {"aid", "whurs19", "resisc45"}:
        errors.append("benchmark_manifests must contain aid, whurs19, and resisc45")
    else:
        for dataset in ("whurs19", "resisc45"):
            entry = manifests[dataset]
            if entry.get("role") != "cross_dataset_test_only" or entry.get("train_allowed") is not False:
                errors.append(f"{dataset} must remain cross_dataset_test_only")

    baselines = plan.get("baselines")
    if not isinstance(baselines, dict):
        errors.append("baselines must be an object")
        baselines = {}
    cards = BASELINES / "cards"
    for alias, entry in baselines.items():
        method = entry.get("method") if isinstance(entry, dict) else None
        card_path = cards / f"{method}.json"
        if not card_path.is_file():
            errors.append(f"baseline {alias}: card is missing for method={method!r}")
        if not isinstance(entry, dict):
            errors.append(f"baseline {alias}: plan entry must be an object")
            continue
        if not isinstance(method, str) or not method:
            errors.append(f"baseline {alias}: method must be a non-empty string")
        config_run = entry.get("config_run")
        if config_run is not None and config_run not in runs:
            errors.append(f"baseline {alias}: unknown config_run={config_run}")
        required_env = entry.get("required_env")
        adapter_env = entry.get("adapter_env")
        if (required_env is None) != (adapter_env is None):
            errors.append(
                f"baseline {alias}: required_env and adapter_env must both be null or strings")
    try:
        for name in plan.get("baseline_groups", {}):
            missing = [method for method in baseline_ids(plan, name) if method not in baselines]
            if missing:
                errors.append(f"baseline_groups.{name} has unknown methods: {missing}")
    except PlanError as exc:
        errors.append(str(exc))

    if errors:
        raise PlanError("invalid ICLR A800 plan:\n  - " + "\n  - ".join(errors))
    return plan, configs


def check_runtime(
    plan: dict[str, Any],
    configs: dict[str, dict[str, Any]],
    *,
    rswm_base: str,
    experiments_root: Path,
    datasets: list[str],
) -> list[str]:
    """Validate staged server paths and frozen manifests; still never accesses CUDA."""
    checks: list[str] = []
    canonical = configs["iclr_primary_native_attn_cam_r16_s42"]
    checks.extend(formal_config.check_runtime_paths(
        canonical,
        experiments_root=experiments_root,
        project_root=PROJECT_ROOT,
    ))
    sys.path.insert(0, str(BASELINES))
    from protocol import validate_manifest_document  # noqa: E402

    for dataset in datasets:
        entry = plan["benchmark_manifests"].get(dataset)
        if entry is None:
            raise PlanError(f"unknown benchmark dataset: {dataset}")
        path = Path(_expand_base(entry["path"], rswm_base))
        if not path.is_file():
            raise PlanError(f"{dataset}: frozen benchmark manifest is missing: {path}")
        document = _load_json(path)
        try:
            validate_manifest_document(document, require_files=True)
        except Exception as exc:
            raise PlanError(f"{dataset}: invalid frozen benchmark manifest: {exc}") from exc
        if not document.get("headline_eligible", False):
            raise PlanError(f"{dataset}: manifest is not headline_eligible")
        checks.append(f"manifest[{dataset}]={path}")
    return checks


def _plan_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _list_runs(plan: dict[str, Any], configs: dict[str, dict[str, Any]], args: argparse.Namespace) -> int:
    ids = group_run_ids(plan, args.group)
    if args.run:
        if args.run not in ids:
            raise PlanError(f"--run {args.run} is not in group {args.group}")
        ids = [args.run]
    rows = []
    for run_id in ids:
        path = _project_path(plan["runs"][run_id]["config"], field=f"runs.{run_id}.config")
        rows.append((run_id, str(path), configs[run_id]))
    if args.format == "nul":
        for run_id, path, _config in rows:
            sys.stdout.buffer.write(run_id.encode() + b"\0" + path.encode() + b"\0")
        return 0
    if args.format == "json":
        print(json.dumps([
            {"experiment_id": run_id, "config": path,
             "pose_mode": cfg["dataset"]["pose_mode"],
             "target_modules": cfg["lora"]["target_modules"],
             "rank": cfg["lora"]["rank"], "seed": cfg["training"]["seed"]}
            for run_id, path, cfg in rows
        ], indent=2))
        return 0
    if args.format == "tsv":
        for run_id, path, _config in rows:
            print(f"{run_id}\t{path}")
        return 0
    print("EXPERIMENT_ID\tPOSE\tTARGETS\tRANK\tSEED\tCONFIG")
    for run_id, path, cfg in rows:
        print(f"{run_id}\t{cfg['dataset']['pose_mode']}\t{cfg['lora']['target_modules']}\t"
              f"{cfg['lora']['rank']}\t{cfg['training']['seed']}\t{path}")
    return 0


def _list_baselines(plan: dict[str, Any], args: argparse.Namespace) -> int:
    aliases = baseline_ids(plan, args.group)
    if args.method:
        if args.method not in aliases:
            raise PlanError(f"--method {args.method} is not in baseline group {args.group}")
        aliases = [args.method]
    rows = []
    for alias in aliases:
        entry = plan["baselines"][alias]
        method = entry["method"]
        config_run = entry.get("config_run")
        config = "-" if config_run is None else str(_project_path(
            plan["runs"][config_run]["config"], field=f"runs.{config_run}.config"))
        rows.append((alias, method, config, entry.get("required_env") or "-",
                     entry.get("adapter_env") or "-",
                     "1" if entry.get("requires_generation_gpu") else "0"))
    if args.format == "json":
        print(json.dumps([
            {"alias": row[0], "method": row[1],
             "config": None if row[2] == "-" else row[2],
             "required_env": None if row[3] == "-" else row[3],
             "adapter_env": None if row[4] == "-" else row[4],
             "requires_generation_gpu": row[5] == "1"}
            for row in rows
        ], indent=2))
    else:
        for row in rows:
            print("\t".join(row))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--rswm-base", default=os.environ.get("RSWM_BASE", "/nix/rsroll_25828"))
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate plan and formal runner configs")
    validate.add_argument("--check-runtime", action="store_true")
    validate.add_argument("--experiments-root", default="experiments")
    validate.add_argument("--datasets", nargs="+", default=["aid"])

    runs = sub.add_parser("list-runs", help="materialize one training group")
    runs.add_argument("--group", choices=("primary", "matched", "target_all", "rank", "all"), default="primary")
    runs.add_argument("--run")
    runs.add_argument("--format", choices=("table", "tsv", "nul", "json"), default="table")

    baselines = sub.add_parser("list-baselines", help="materialize one baseline group")
    baselines.add_argument("--group", choices=("lower_bound", "primary", "main"),
                           default="lower_bound")
    baselines.add_argument("--method")
    baselines.add_argument("--format", choices=("tsv", "json"), default="tsv")

    manifest = sub.add_parser("manifest-path", help="print one expanded frozen manifest path")
    manifest.add_argument("--dataset", choices=("aid", "whurs19", "resisc45"), required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        plan_path = Path(args.plan).resolve()
        plan, configs = load_and_validate(plan_path, rswm_base=args.rswm_base)
        if args.command == "validate":
            print(f"PLAN_VALID schema={SCHEMA} runs={len(configs)} sha256={_plan_hash(plan_path)}")
            print("GPU_ACCESSED=0 TRAINING_STARTED=0")
            if args.check_runtime:
                checks = check_runtime(
                    plan, configs, rswm_base=args.rswm_base,
                    experiments_root=Path(args.experiments_root), datasets=args.datasets,
                )
                for check in checks:
                    print(f"RUNTIME_PASS {check}")
            return 0
        if args.command == "list-runs":
            return _list_runs(plan, configs, args)
        if args.command == "list-baselines":
            return _list_baselines(plan, args)
        if args.command == "manifest-path":
            print(_expand_base(plan["benchmark_manifests"][args.dataset]["path"], args.rswm_base))
            return 0
        raise AssertionError(args.command)
    except (PlanError, formal_config.ConfigError) as exc:
        print(f"PLAN_INVALID: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
