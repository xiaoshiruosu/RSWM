#!/usr/bin/env python3
"""Export paper tables from standardized RS-Roll result directories.

Missing metrics are written as empty CSV cells.  The exporter never substitutes zero
and never derives a synthetic composite score.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
BASELINES = ROOT / "server_run" / "baselines"
sys.path.insert(0, str(BASELINES))

from protocol import (CARD_SCHEMA, RESULT_SCHEMA, sha256_file,
                      validate_manifest_document)

TABLES = (
    "main_results.csv",
    "overlap_results.csv",
    "trajectory_results.csv",
    "rank_ablation.csv",
    "target_ablation.csv",
    "cross_dataset_results.csv",
    "efficiency_results.csv",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _write(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key)
                             for key in fields})


def _context(metric_path: Path) -> dict[str, Any]:
    root = metric_path.parent
    required = ("baseline_card.json", "config.json", "manifest.json",
                "sample_metrics.jsonl")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{root}: incomplete standardized result, missing {missing}")
    metric = _load(metric_path)
    card = _load(root / "baseline_card.json")
    config = _load(root / "config.json")
    manifest = _load(root / "manifest.json")
    if metric.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"{metric_path}: schema must be {RESULT_SCHEMA}")
    if card.get("schema") != CARD_SCHEMA:
        raise ValueError(f"{root / 'baseline_card.json'}: invalid card schema")
    validate_manifest_document(manifest, require_files=False)
    if metric.get("method") != card.get("name") or \
            config.get("method") != card.get("name"):
        raise ValueError(f"{root}: method identity differs across result/card/config")
    identity = metric.get("experiment_identity") or config.get("experiment_identity") or {}
    for key in ("dataset_checksum", "split_checksum", "manifest_checksum"):
        expected = manifest.get(key)
        if identity.get(key) != expected:
            raise ValueError(f"{root}: experiment_identity.{key} != frozen manifest")
    rows = []
    for line_number, line in enumerate(
            (root / "sample_metrics.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not row.get("sample_id"):
            raise ValueError(f"{root}: invalid sample_metrics line {line_number}")
        rows.append(row)
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{root}: duplicate sample_id in sample_metrics.jsonl")
    if sample_ids != [str(value) for value in manifest.get("sample_keys", [])]:
        raise ValueError(f"{root}: sample metrics do not match frozen manifest order/set")
    if int(metric.get("N", -1)) != len(rows):
        raise ValueError(f"{root}: aggregate N differs from sample_metrics row count")
    if not isinstance(metric.get("result_available"), bool):
        raise ValueError(f"{root}: result_available must be an explicit boolean")
    return {"root": root, "metric": metric, "card": card,
            "config": config, "manifest": manifest, "identity": identity,
            "metric_sha256": sha256_file(metric_path)}


def _base(ctx: dict[str, Any]) -> dict[str, Any]:
    metric, card, identity = ctx["metric"], ctx["card"], ctx["identity"]
    return {
        "experiment_id": metric.get("experiment_id", ctx["root"].name),
        "Method": metric.get("method") or card.get("name"),
        "Dataset": metric.get("dataset"),
        "Split": metric.get("split"),
        "N": metric.get("N"),
        "T": json.dumps(metric.get("T"), ensure_ascii=False)
             if isinstance(metric.get("T"), (list, dict)) else metric.get("T"),
        "Native Output Type": card.get("native_output"),
        "Trajectory Conditioning": card.get("trajectory_support"),
        "Adaptation": card.get("adaptation"),
        "Trainable Params": identity.get("trainable_parameters"),
        "LoRA Rank": identity.get("lora_rank"),
        "LoRA Target": identity.get("lora_target"),
        "Pose Route": identity.get("pose_route"),
        "FID": metric.get("FID"),
        "LPIPS": metric.get("LPIPS"),
        "FVD_I3D": metric.get("FVD_I3D"),
        "FVD_R3D_PROXY": metric.get("FVD_R3D_PROXY"),
        "direction_error": metric.get("direction_error"),
        "endpoint_error": metric.get("endpoint_error"),
        "trajectory_alignment": metric.get("trajectory_alignment"),
        "registration_success_rate": metric.get("registration_success_rate"),
        "Result Status": metric.get("result_status"),
        "Result Available": metric.get("result_available"),
    }


def export(contexts: list[dict[str, Any]], output: Path) -> dict[str, int]:
    experiment_ids = [str(ctx["metric"].get("experiment_id")) for ctx in contexts]
    duplicates = sorted({value for value in experiment_ids
                         if experiment_ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate experiment_id values: {duplicates}")
    base_rows = [_base(ctx) for ctx in contexts]
    main_fields = [
        "experiment_id", "Method", "Native Output Type", "Trajectory Conditioning",
        "Adaptation", "Trainable Params", "Dataset", "Split", "N", "T",
        "FID", "LPIPS", "FVD_I3D", "FVD_R3D_PROXY", "direction_error",
        "endpoint_error", "trajectory_alignment", "registration_success_rate",
        "Result Status", "Result Available",
    ]
    _write(output / "main_results.csv", main_fields, base_rows)

    overlap_rows = []
    trajectory_rows = []
    for ctx, base in zip(contexts, base_rows):
        metric = ctx["metric"]
        for label, values in (metric.get("overlap_results") or {}).items():
            overlap_rows.append({
                "experiment_id": base["experiment_id"], "Method": base["Method"],
                "Dataset": base["Dataset"], "overlap_bin": label, **values,
            })
        trajectory_rows.append({
            "experiment_id": base["experiment_id"], "Method": base["Method"],
            "Dataset": base["Dataset"], "Split": base["Split"], "N": base["N"],
            **(metric.get("trajectory_metrics") or {}),
        })
    overlap_fields = [
        "experiment_id", "Method", "Dataset", "overlap_bin", "N", "SSIM", "PSNR",
        "direction_error", "endpoint_error", "mean_future_overlap",
    ]
    _write(output / "overlap_results.csv", overlap_fields, overlap_rows)
    trajectory_keys = sorted({key for row in trajectory_rows for key in row}
                             - {"experiment_id", "Method", "Dataset", "Split", "N"})
    trajectory_fields = ["experiment_id", "Method", "Dataset", "Split", "N"] + trajectory_keys
    _write(output / "trajectory_results.csv", trajectory_fields, trajectory_rows)

    rank_rows = [row for row in base_rows if row.get("LoRA Rank") is not None]
    rank_fields = [
        "experiment_id", "Method", "Dataset", "Pose Route", "LoRA Target", "LoRA Rank",
        "Trainable Params", "FID", "LPIPS", "FVD_I3D", "FVD_R3D_PROXY",
        "direction_error", "endpoint_error", "trajectory_alignment", "Result Status",
    ]
    _write(output / "rank_ablation.csv", rank_fields, rank_rows)

    target_rows = [row for row in base_rows if row.get("LoRA Target") is not None]
    target_fields = [
        "experiment_id", "Method", "Dataset", "Pose Route", "LoRA Target", "LoRA Rank",
        "Trainable Params", "FID", "LPIPS", "FVD_I3D", "FVD_R3D_PROXY",
        "direction_error", "endpoint_error", "trajectory_alignment", "Result Status",
    ]
    _write(output / "target_ablation.csv", target_fields, target_rows)

    cross_fields = [
        "experiment_id", "Method", "Dataset", "Split", "N", "FID", "LPIPS",
        "FVD_I3D", "FVD_R3D_PROXY", "direction_error", "endpoint_error",
        "trajectory_alignment", "registration_success_rate", "Result Status",
    ]
    _write(output / "cross_dataset_results.csv", cross_fields, base_rows)

    efficiency_rows = []
    for ctx, base in zip(contexts, base_rows):
        metric, identity = ctx["metric"], ctx["identity"]
        efficiency = metric.get("efficiency") or {}
        efficiency_rows.append({
            "experiment_id": base["experiment_id"], "Method": base["Method"],
            "Dataset": base["Dataset"], "LoRA Rank": base["LoRA Rank"],
            "LoRA Target": base["LoRA Target"],
            "Trainable Params": base["Trainable Params"],
            "training_memory_gib": efficiency.get("training_memory_gib"),
            "training_time_seconds": efficiency.get("training_time_seconds"),
            "inference_time_seconds": efficiency.get("inference_time_seconds"),
            "GPU": identity.get("GPU"),
            "precision": identity.get("precision"),
            "Result Status": base["Result Status"],
        })
    efficiency_fields = [
        "experiment_id", "Method", "Dataset", "LoRA Rank", "LoRA Target",
        "Trainable Params", "training_memory_gib", "training_time_seconds",
        "inference_time_seconds", "GPU", "precision", "Result Status",
    ]
    _write(output / "efficiency_results.csv", efficiency_fields, efficiency_rows)
    return {
        "main_results.csv": len(base_rows),
        "overlap_results.csv": len(overlap_rows),
        "trajectory_results.csv": len(trajectory_rows),
        "rank_ablation.csv": len(rank_rows),
        "target_ablation.csv": len(target_rows),
        "cross_dataset_results.csv": len(base_rows),
        "efficiency_results.csv": len(efficiency_rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True,
                        help="result directories or roots searched recursively")
    parser.add_argument("--output", default="paper_results")
    args = parser.parse_args(argv)
    files = []
    for raw in args.results:
        path = Path(raw)
        if path.is_file() and path.name == "aggregate_metrics.json":
            files.append(path)
        elif path.is_dir() and (path / "aggregate_metrics.json").is_file():
            files.append(path / "aggregate_metrics.json")
        elif path.is_dir():
            files.extend(path.rglob("aggregate_metrics.json"))
    files = sorted(set(path.resolve() for path in files))
    contexts = [_context(path) for path in files]
    output = Path(args.output)
    counts = export(contexts, output)
    manifest = {
        "schema": "rsroll.paper_table_export/v1",
        "input_results": [str(path) for path in files],
        "input_aggregate_sha256": {
            str(path): ctx["metric_sha256"] for path, ctx in zip(files, contexts)
        },
        "row_counts": counts,
        "missing_values": "empty CSV cells; never zero-filled",
        "claim_guard": "tables contain recorded results only; no conclusion is inferred",
    }
    (output / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    state = "PASS" if contexts else "NOT RUN"
    print(f"[{state}] {len(contexts)} standardized results -> {output}")
    for name in TABLES:
        print(f"  {name}: {counts[name]} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
