"""Execute the frozen three-diagnostic matrix through the actual suite pipeline."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path
from .geometry import canonical, digest, file_hash
from .materialized import audit_materialized, read_json
from .data import write_json
from .suite import build_plan, save_plan, execute_plan, replay_suite, _load_task, code_fingerprint
from .aggregate import aggregate_suite
from .export_tables import export_tables

DIAGNOSTIC_MODELS = [{"model_id": "oracle", "adapter": "oracle"},
                     {"model_id": "copy", "adapter": "copy"},
                     {"model_id": "static-noise-map", "adapter": "noise-map"}]
OUTPUTS = {"F2": 32, "F3": 64, "F5": 96, "F7": 160}


def accept_smoke12(build, manifest, inventory, source_dir, out, *, code_commit, max_output_gb=16., task_limit=None, execute=True):
    """No asset generation/download or alternate benchmark runner is used here."""
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Acceptance output must be empty.")
    if task_limit is not None and (type(task_limit) is not int or task_limit < 1):
        raise ValueError("task_limit must be a positive integer.")
    if max_output_gb <= 0 or max_output_gb != max_output_gb or max_output_gb == float("inf"):
        raise ValueError("Output budget must be positive and finite.")
    audit = audit_materialized(build, manifest, inventory, source_dir)
    if audit["acceptance_status"] != "ACCEPTED_IN_STATED_SCOPE":
        raise ValueError("Duplicate pixels need review before acceptance.")
    fixture = audit["input_kind"] == "synthetic_fixture"
    if audit["profile"] != "smoke12" and not (fixture and task_limit is not None):
        raise ValueError("Full acceptance requires exactly the frozen smoke12 profile.")
    data = read_json(Path(build)/"private/index.json")
    rows = data["tasks"][:task_limit] if task_limit is not None else data["tasks"]
    selected_worlds = sorted({r["world_id"] for r in rows})
    full_matrix = task_limit is None and not fixture
    if full_matrix and (len(rows) != 192 or len(selected_worlds) != 12 or sum(OUTPUTS[r["family"]] for r in rows) != 16896):
        raise ValueError("Frozen smoke12 expected task/output matrix differs.")
    planned_per_method = sum(OUTPUTS[r["family"]] for r in rows)
    output_budget = planned_per_method*3*240*416*3
    if output_budget > max_output_gb*1_000_000_000:
        raise ValueError("Uncompressed diagnostic RGB exceeds the explicit output budget.")
    root.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(root.parent).free
    # A conservative bound includes uncompressed outputs plus private snapshots.
    required_bytes = int(output_budget*1.25)
    if free_bytes < required_bytes:
        raise OSError("Insufficient free disk for RGB outputs and snapshot overhead.")
    root.mkdir(exist_ok=True)
    write_json(root/"input-audit.json", audit)
    write_json(root/"models.json", DIAGNOSTIC_MODELS)
    worlds = {w["world_id"]: w for w in data["worlds"]}
    task_rows = []
    for row in rows:
        tp = Path(build)/"private"/row["task_path"]
        task, _ = _load_task(tp)
        world = worlds[task["world_id"]]
        task_rows.append({**row, "task_path": str(tp.resolve()),
                          "world_path": str((Path(build)/"private/worlds"/world["path"]).resolve()),
                          "world_file_sha256": world["world_file_sha256"], "world_pixel_sha256": world["world_pixel_sha256"]})
    plan = build_plan(task_rows, DIAGNOSTIC_MODELS, [0], data["source_manifest_sha256"], code_commit,
                      study=data.get("study", "core"))
    plan.pop("plan_sha256")
    plan["diagnostic_acceptance_profile"] = {"profile": "smoke12" if full_matrix else "engineering_subset",
                                           "input_kind": audit["input_kind"], "formal_test": False}
    plan["plan_sha256"] = digest(plan)
    save_plan(plan, root/"plan.json")
    report = {"schema": "rswm-smoke12-acceptance-v1", "status": "PLANNED_NOT_RUN", "input_kind": audit["input_kind"],
              "scope": "smoke12 + three diagnostic models + Phase A pipeline" if full_matrix else "explicit engineering subset",
              "full_smoke12_matrix": full_matrix, "selected_worlds": selected_worlds,
              "planned_runs_per_model": len(rows), "planned_runs_total": len(plan["expected"]),
              "planned_unique_outputs_per_model": planned_per_method, "planned_unique_outputs_total": planned_per_method*3,
              "uncompressed_rgb_bytes": output_budget, "disk_free_bytes_at_start": free_bytes,
              "output_budget_gb": max_output_gb, "plan_sha256": plan["plan_sha256"], "models": DIAGNOSTIC_MODELS,
              "source_code_sha256": plan["source_code_sha256"], "code_commit": code_commit,
              "real_generation_model": "NOT_RUN", "real_gpu": "NOT_RUN", "fault_controls": "SEPARATE_G1_EVIDENCE",
              "real_data_validated": False}
    if execute:
        started = time.monotonic()
        execute_plan(plan, root/"suite")
        replay = replay_suite(plan, root/"suite")
        analysis = aggregate_suite(plan, root/"suite", bootstrap=20, seed=0)
        export_tables(analysis, root/"analysis")
        if code_fingerprint() != plan["source_code_sha256"]:
            raise ValueError("Source changed during diagnostic acceptance; retain evidence and rerun after freeze.")
        by_model = {}
        for model in DIAGNOSTIC_MODELS:
            mid = model["model_id"]
            complete = [r for r in analysis["run_completeness"] if r["model_id"] == mid]
            by_model[mid] = {"runs": len({r["run_id"] for r in complete}), "planned_outputs": sum(r["planned_outputs"] for r in complete),
                             "valid_outputs": sum(r["valid_outputs"] for r in complete),
                             "resource_complete_runs": sum(r["model_id"] == mid and r["measurement_complete"] for r in analysis["resources"])}
            if by_model[mid] != {"runs": len(rows), "planned_outputs": planned_per_method,
                                 "valid_outputs": planned_per_method, "resource_complete_runs": len(rows)}:
                raise ValueError("Per-model unique-output/resource acceptance failed.")
        for cell in analysis["cells"]:
            if cell.get("applicability") == "not_applicable":
                continue
            if cell["model_id"] == "oracle" and (cell["mae"] != 0 or cell["coverage"] != 1):
                raise ValueError("Oracle RGB/coverage acceptance failed.")
            if cell["model_id"] == "static-noise-map" and cell["metric"] != "groundtruth" and cell["mae"] != 0:
                raise ValueError("StaticNoiseMap consistency boundary acceptance failed.")
        if analysis["acceptance_status"] != "ACCEPTED" or any(r["status"] != "COMPLETED" for r in replay):
            raise ValueError("Suite integrity/replay acceptance failed.")
        report.update(status="REAL_DATA_VALIDATED" if full_matrix else "ENGINEERING_SUBSET_VALIDATED",
                      real_data_validated=full_matrix, per_model=by_model, analysis_id=analysis["analysis_id"],
                      analysis_status=analysis["acceptance_status"], elapsed_seconds=time.monotonic()-started,
                      failed_runs=analysis["failure_summary"]["failed_or_unverifiable_runs"])
    report["artifact_sha256"] = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}
    write_json(root/"acceptance.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("build", "manifest", "inventory", "source-dir", "out", "code-commit"):
        p.add_argument("--"+key, required=True)
    p.add_argument("--max-output-gb", type=float, default=16.)
    p.add_argument("--task-limit", type=int, help="Explicit engineering subset; never full smoke12 validation")
    p.add_argument("--execute", action="store_true", help="Generate and replay the diagnostic matrix after all preflight checks")
    args = p.parse_args(argv)
    try:
        result = accept_smoke12(args.build, args.manifest, args.inventory, args.source_dir, args.out,
                               code_commit=args.code_commit, max_output_gb=args.max_output_gb, task_limit=args.task_limit, execute=args.execute)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result = {"schema": "rswm-smoke12-acceptance-v1", "status": "BLOCKED" if isinstance(exc, FileNotFoundError) or str(exc).startswith("Missing artifact:") else "FAILED",
                  "real_data_validated": False, "error_type": type(exc).__name__, "reason": str(exc)}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        write_json(Path(args.out)/"acceptance-failure.json", result)
        print(json.dumps(result)); return 2
    print(json.dumps({k: result[k] for k in ("status", "planned_runs_total", "planned_unique_outputs_total", "real_data_validated")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
