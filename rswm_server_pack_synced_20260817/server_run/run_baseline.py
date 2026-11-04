#!/usr/bin/env python3
"""Paper-facing unified baseline runner for RS-Roll.

It extends the existing ``server_run/baselines/run_baseline.py`` implementation and
keeps its validated adapters/trajectory evaluator.  On success it materializes the
paper-facing result contract without starting any training job.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
BASELINES = ROOT / "server_run" / "baselines"
LEGACY_RUNNER = BASELINES / "run_baseline.py"
sys.path.insert(0, str(BASELINES))

from cards import load_card  # noqa: E402
from protocol import (MANIFEST_SCHEMA, RESULT_SCHEMA, null_metric_block,
                      stable_json_sha256, validate_manifest_document)  # noqa: E402

CODE_VERSION = "rsroll-baseline-extension-20260816"
ENV_TOKEN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if resolved is None:
            raise ValueError(
                f"unresolved config variable ${{{name}}}; set {name} before the run")
        return resolved

    return ENV_TOKEN.sub(replace, value)


def _load_structured(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    value = (json.loads(text) if path.suffix.lower() == ".json"
             else yaml.safe_load(text))
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected an object/mapping")
    return _expand_env(value)


def _nested(doc: dict[str, Any], *keys: str) -> Any:
    value: Any = doc
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _git_commit() -> str | None:
    try:
        run = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            capture_output=True, check=True,
        )
        value = run.stdout.strip()
        return value or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _finite_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return sum(values) / len(values) if values else None


def _identity(config: dict[str, Any], manifest: dict[str, Any], card: dict[str, Any],
              method: str, n_samples: int) -> dict[str, Any]:
    training = config.get("training") if isinstance(config.get("training"), dict) else {}
    lora = config.get("lora") if isinstance(config.get("lora"), dict) else {}
    dataset = config.get("dataset") if isinstance(config.get("dataset"), dict) else {}
    hardware = config.get("hardware") if isinstance(config.get("hardware"), dict) else {}
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    evaluation = (config.get("evaluation")
                  if isinstance(config.get("evaluation"), dict) else {})
    reported_identity = (config.get("experiment_identity")
                         if isinstance(config.get("experiment_identity"), dict) else {})
    generation_seed = evaluation.get("gen_seed", manifest.get("gen_seed"))
    return {
        "git_commit": _git_commit(),
        "code_version": CODE_VERSION,
        "dataset_checksum": manifest.get("dataset_checksum"),
        "split_checksum": manifest.get("split_checksum"),
        "manifest_checksum": manifest.get("manifest_checksum") or stable_json_sha256(manifest),
        "checkpoint": model.get("checkpoint") or card.get("checkpoint"),
        "checkpoint_sha256": reported_identity.get("checkpoint_sha256"),
        "lora_target": lora.get("target_modules"),
        "lora_rank": lora.get("rank"),
        "trainable_parameters": reported_identity.get("trainable_parameters"),
        "pose_route": dataset.get("pose_mode"),
        "frames": manifest.get("frame_counts") or manifest.get("num_frames"),
        "training_samples": training.get("max_train_samples"),
        "test_samples": n_samples,
        "seed": generation_seed,
        "generation_seed": generation_seed,
        "train_seed": training.get("seed"),
        "split_seed": training.get("split_seed"),
        "learning_rate": training.get("learning_rate"),
        "training_steps": training.get("max_steps"),
        "num_epochs": training.get("num_epochs"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "batch_size": training.get("batch_size"),
        "precision": hardware.get("precision"),
        "GPU": hardware.get("profile"),
        "headline_eligible_manifest": bool(manifest.get("headline_eligible", False)),
        "method": method,
    }


def _augment_rows(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    index = {str(s["sample_id"]): s for s in manifest.get("samples", [])}
    out = []
    for row in rows:
        item = dict(row)
        sample = index.get(str(row.get("sample_id")), {})
        item.update({
            "source_dataset": sample.get("source_dataset", manifest.get("source_dataset")),
            "source_split": sample.get("source_split", manifest.get("source_split")),
            "trajectory_id": sample.get("trajectory_id"),
            "trajectory_type": sample.get("trajectory_type"),
            "frame_count": sample.get("frame_count"),
            "endpoint_overlap": sample.get("endpoint_overlap"),
            "mean_future_overlap": sample.get("mean_future_overlap"),
            "minimum_future_overlap": sample.get("minimum_future_overlap"),
            "difficulty_bin": sample.get("difficulty_bin"),
            "sample_checksum": sample.get("checksum"),
        })
        out.append(item)
    return out


def _by_difficulty(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    labels = sorted({str(r["difficulty_bin"]) for r in rows
                     if r.get("difficulty_bin") is not None})
    for label in labels:
        subset = [row for row in rows if str(row.get("difficulty_bin")) == label]
        result[label] = {
            "N": len(subset),
            "SSIM": _finite_mean(subset, "ssim"),
            "PSNR": _finite_mean(subset, "psnr"),
            "direction_error": _finite_mean(subset, "net_direction_error"),
            "endpoint_error": _finite_mean(subset, "waypoint_endpoint_error"),
            "mean_future_overlap": _finite_mean(subset, "mean_future_overlap"),
        }
    return result


def _materialize(args, manifest: dict[str, Any], config: dict[str, Any],
                 card: dict[str, Any]) -> None:
    output = Path(args.output)
    scores_path = output / "scores.json"
    resolved_path = output / "resolved_manifest.json"
    if not scores_path.is_file() or not resolved_path.is_file():
        raise FileNotFoundError("legacy runner did not produce scores/resolved_manifest")
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    rows = _augment_rows(scores.get("per_sample", []), manifest)
    if not rows:
        raise RuntimeError("empty sample metrics; refusing to publish a zero-row result")
    (output / "videos").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    archive = output / "frames.npz"
    video_index = {
        "storage": "../frames.npz" if archive.is_file() else None,
        "format": "npz mapping sample_id -> [T,H,W,C] uint8",
        "sample_ids": [row["sample_id"] for row in rows],
        "saved": archive.is_file(),
    }
    (output / "videos" / "index.json").write_text(
        json.dumps(video_index, indent=2, ensure_ascii=False), encoding="utf-8")

    resolved_keys = [str(value) for value in (resolved.get("sample_keys") or [])]
    frozen_keys = [str(value) for value in manifest.get("sample_keys", [])]
    if resolved_keys != frozen_keys:
        raise RuntimeError(
            "legacy runner resolved a different sample order/set than the frozen manifest")
    # Copy the frozen manifest byte-for-meaning without adding runtime fields.  Adding
    # `resolved_sample_keys` here used to invalidate its manifest_checksum.
    manifest_out = dict(manifest)
    (output / "manifest.json").write_text(
        json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "baseline_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
    identity = _identity(config, manifest, card, args.method, len(rows))
    config_out = {
        "schema": "rsroll.experiment_config_snapshot/v1",
        "experiment_id": args.experiment_id or output.name,
        "method": args.method,
        "source_config": str(Path(args.config).resolve()) if args.config else None,
        "config": config,
        "config_checksum": stable_json_sha256(config),
        "resolved_sample_keys_checksum": stable_json_sha256(resolved_keys),
        "experiment_identity": identity,
    }
    (output / "config.json").write_text(
        json.dumps(config_out, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "sample_metrics.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    raw_aggregate = scores.get("aggregate", {})
    metrics = null_metric_block()
    metrics.update({
        "SSIM": raw_aggregate.get("ssim"),
        "PSNR": raw_aggregate.get("psnr"),
        "direction_error": raw_aggregate.get("net_direction_error"),
        "endpoint_error": raw_aggregate.get("waypoint_endpoint_error"),
        "registration_success_rate": (
            float(raw_aggregate["valid_pct"]) / 100.0
            if raw_aggregate.get("valid_pct") is not None else None),
    })
    aggregate = {
        "schema": RESULT_SCHEMA,
        "experiment_id": config_out["experiment_id"],
        "method": args.method,
        "dataset": manifest.get("source_dataset", "unknown"),
        "split": manifest.get("source_split", "unknown"),
        "N": len(rows),
        "T": manifest.get("frame_counts") or manifest.get("num_frames"),
        "overlap_bin": "all",
        "scoring_scope": "future_frames_only",
        "condition_frames_excluded": 1,
        **metrics,
        "trajectory_metrics": raw_aggregate,
        "overlap_results": _by_difficulty(rows),
        "success": {
            "n_requested": scores.get("protocol", {}).get("n_samples"),
            "n_success": scores.get("n_success"),
            "n_failed": scores.get("n_failed"),
            "failed_keys": scores.get("failed_keys", []),
            "success_rate": scores.get("success_rate"),
        },
        "experiment_identity": identity,
        "headline_eligible": bool(manifest.get("headline_eligible", False)),
        "missing_metrics_are_null_not_zero": True,
        "R3D_metric_name_guard": "FVD_R3D_PROXY is not standard I3D-FVD",
        "geometry_claim_guard": (
            "trajectory/registration metrics measure viewpoint-conditioned 2D "
            "consistency; they do not prove explicit 3D geometry understanding"
        ),
        "result_available": False,
        "result_status": "LOCAL_EVALUATION_COMPLETE_PAPER_RELEASE_PENDING",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    (output / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", "--baseline", dest="method")
    parser.add_argument("--manifest")
    parser.add_argument("--output", "--out", dest="output")
    parser.add_argument("--config")
    parser.add_argument("--experiment-id")
    parser.add_argument("--skip-failed", action="store_true")
    parser.add_argument("--min-success", type=float, default=1.0)
    parser.add_argument("--dry-run", "--dryrun", dest="dry_run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--no-save-videos", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        return subprocess.run([sys.executable, str(LEGACY_RUNNER), "--list"]).returncode
    for name in ("method", "manifest", "output"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required unless --list is used")
    manifest_path = Path(args.manifest)
    manifest = _load_structured(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"paper-facing runner requires {MANIFEST_SCHEMA}; use the legacy "
            "server_run/baselines/run_baseline.py only for non-paper diagnostics")
    validate_manifest_document(manifest, require_files=True)
    card = load_card(args.method)
    config = _load_structured(Path(args.config) if args.config else None)
    config_manifest = _nested(config, "evaluation", "manifest")
    if config_manifest is not None and \
            Path(str(config_manifest)).resolve() != manifest_path.resolve():
        raise ValueError(
            f"config evaluation.manifest={config_manifest} differs from --manifest="
            f"{manifest_path}; refusing a mismatched experiment identity")
    output = Path(args.output)
    if not args.dry_run and output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(
            f"{output} is non-empty; use a new experiment_id/output or explicit --force")
    command = [
        sys.executable, str(LEGACY_RUNNER),
        "--baseline", args.method,
        "--manifest", str(manifest_path),
        "--out", str(output),
        "--min-success", str(args.min_success),
    ]
    if args.dry_run:
        command.append("--dryrun")
    elif not args.no_save_videos:
        command.append("--save-frames")
    if args.skip_failed:
        command.append("--skip-failed")
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    sys.stdout.write(run.stdout)
    sys.stderr.write(run.stderr)
    if args.dry_run:
        print(f"[CARD] native_output={card['native_output']}")
        print(f"[CARD] trajectory_support={card.get('trajectory_support')}")
        print("[DRY-RUN] TRAINING_STARTED=0 GPU_TRAINING_STARTED=0")
        return run.returncode
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "logs" / "runner.log").write_text(
        run.stdout + ("\n[stderr]\n" + run.stderr if run.stderr else ""),
        encoding="utf-8",
    )
    if run.returncode != 0:
        (output / "logs" / "run_status.json").write_text(json.dumps({
            "status": "FAILED", "returncode": run.returncode,
            "method": args.method, "training_started": False,
        }, indent=2), encoding="utf-8")
        return run.returncode
    try:
        _materialize(args, manifest, config, card)
    except Exception as exc:  # fail closed after generation/standardization errors
        with (output / "logs" / "runner.log").open("a", encoding="utf-8") as stream:
            stream.write(
                f"\n[standardization-error]\n{type(exc).__name__}: {exc}\n")
        (output / "logs" / "run_status.json").write_text(json.dumps({
            "status": "FAILED", "returncode": 4, "method": args.method,
            "stage": "STANDARDIZATION", "error_type": type(exc).__name__,
            "error": str(exc), "training_started": False,
        }, indent=2), encoding="utf-8")
        print(f"[FAIL] standardization: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 4
    (output / "logs" / "run_status.json").write_text(json.dumps({
        "status": "PASS", "returncode": 0, "method": args.method,
        "training_started": False,
    }, indent=2), encoding="utf-8")
    print(f"[PASS] standard result contract -> {output}")
    print("[INFO] TRAINING_STARTED=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
