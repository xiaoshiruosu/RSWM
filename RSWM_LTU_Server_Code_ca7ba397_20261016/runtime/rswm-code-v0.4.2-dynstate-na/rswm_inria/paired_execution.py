"""Independent same-start suite, mechanism execution and whole-world pairing."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import numpy as np

from .geometry import canonical, digest, file_hash
from .paired import STUDY
from .tasks import validate_task
from .suite import (build_plan, validate_plan, execute_plan, replay_suite, save_plan,
                    plan_from_indexes, model_factory, safe_resource_attempts,
                    DEFAULT_RETRY_POLICY, _save, suite_read_lock)
from .suite import integrity_summary
from .replay import load_json, load_rgb_png
from .aggregate import resource_totals
from .history_mechanisms import validate_policy, HistoryCompleterAdapter


PAIR_SCHEMA = "rswm-same-start-execution-v1"
AUXILIARY_METRIC = "first-reference-final-target-v1"
ESTIMAND = "equal applicable pair/origin/seed cells within world; whole-world bootstrap"


def pair_manifest(tasks):
    pairs = {}
    for task in tasks.values():
        validate_task(task)
        if task.get("study") != STUDY or task["family"] not in {"F5", "F7"}:
            raise ValueError("Same-start tasks require their independent study")
        pid = task.get("pair_id")
        variant = task["metadata"].get("variant")
        if not isinstance(pid, str) or not pid or variant not in {"exact", "offset"}:
            raise ValueError("Pair identity/variant missing")
        if variant in pairs.setdefault(pid, {}):
            raise ValueError("Duplicate pair variant")
        pairs[pid][variant] = task
    result = {}
    for pid, variants in pairs.items():
        if set(variants) != {"exact", "offset"}:
            raise ValueError("Each expected pair must retain both variants")
        a, b = variants["exact"], variants["offset"]
        for key in ("world_id", "role", "origin_index", "origin", "family", "protocol"):
            if a[key] != b[key]:
                raise ValueError("Paired world/origin/protocol differs")
        if a["metadata"]["target_center"] != b["metadata"]["target_center"]:
            raise ValueError("Paired target differs")
        for stream in a["programs"]:
            if a["programs"][stream][:81] != b["programs"][stream][:81]:
                raise ValueError("Exact and offset must share their first 80 queries")
            if a["programs"][stream][-1] == b["programs"][stream][-1]:
                raise ValueError("Offset endpoint must differ")
        result[pid] = {"world_id": a["world_id"], "family": a["family"], "role": a["role"],
                       "origin_index": a["origin_index"],
                       "exact_task_id": a["task_id"], "offset_task_id": b["task_id"],
                       "shared_prefix_sha256": digest({k: q[:81] for k, q in a["programs"].items()})}
    return result


def attach_pairs(plan, mechanisms=None):
    """Freeze extension identities while retaining the core v3 attempt machinery."""
    if plan.get("study") != STUDY:
        raise ValueError("Core and paired studies require separate plans")
    plan = copy.deepcopy(plan)
    manifest = pair_manifest(plan["tasks"])
    mechanisms = mechanisms or {}
    for mid, policy in mechanisms.items():
        validate_policy(policy)
        if mid not in plan["models"] or plan["models"][mid]["config"]["adapter"] != "lazy-map":
            raise ValueError("Mechanism requires a frozen completer configuration")
        if not mid.endswith("-" + digest(policy)[:16]):
            raise ValueError("Mechanism model_id must bind its policy hash")
    for row in plan["expected"]:
        row["pair_id"] = plan["tasks"][row["task_id"]]["pair_id"]
    for pid in manifest:
        by_variant = {v: [r for r in plan["expected"] if r["task_id"] == manifest[pid][v + "_task_id"]]
                      for v in ("exact", "offset")}
        for key in ("world_file_sha256", "world_pixel_sha256", "initial_sha256"):
            if {r[key] for rows in by_variant.values() for r in rows}.__len__() != 1:
                raise ValueError("Paired I0/world bytes differ")
    plan["paired_execution"] = {"schema": PAIR_SCHEMA, "pairs": manifest,
                                 "mechanisms": copy.deepcopy(mechanisms),
                                 "auxiliary_metric": AUXILIARY_METRIC, "estimand": ESTIMAND}
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = digest(plan)
    validate_paired_plan(plan)
    return plan


def build_paired_plan(tasks, models, seeds, source_manifest_sha256, code_commit, retry_policy=None, mechanisms=None):
    return attach_pairs(build_plan(tasks, models, seeds, source_manifest_sha256, code_commit,
                                   study=STUDY, retry_policy=retry_policy), mechanisms)


def validate_paired_plan(plan):
    validate_plan(plan)
    ext = plan.get("paired_execution", {})
    if ext.get("schema") != PAIR_SCHEMA or ext.get("pairs") != pair_manifest(plan["tasks"]):
        raise ValueError("Paired manifest mismatch")
    if ext.get("auxiliary_metric") != AUXILIARY_METRIC or ext.get("estimand") != ESTIMAND:
        raise ValueError("Paired frozen analysis contract differs")
    for row in plan["expected"]:
        if row.get("pair_id") != plan["tasks"][row["task_id"]]["pair_id"]:
            raise ValueError("Expected pair identity mismatch")
    for mid, policy in ext.get("mechanisms", {}).items():
        validate_policy(policy)
        if mid not in plan["models"] or plan["models"][mid]["config"]["adapter"] != "lazy-map" or not mid.endswith("-" + digest(policy)[:16]):
            raise ValueError("Mechanism identity mismatch")


def execute_paired_plan(plan, out, resume=False, retry_failed=False, factory=None):
    validate_paired_plan(plan)
    policies = plan["paired_execution"]["mechanisms"]
    def make(config, world, task):
        if factory is not None:
            return factory(config, world, task)
        if config["model_id"] not in policies:
            return model_factory(config, world, task)
        from .baselines import NoiseCompleter, DiffusersCompleter
        completer = NoiseCompleter() if config.get("diagnostic_noise") else DiffusersCompleter(
            config["model_dir"], config.get("device", "cuda"), config.get("steps", 30),
            config.get("prompt", "nadir orthorectified RGB aerial photograph"))
        return HistoryCompleterAdapter(completer, policies[config["model_id"]])
    return execute_plan(plan, out, resume=resume, retry_failed=retry_failed, factory=make)


def _target_arrays(task, record, suite_root):
    """Read a verified first reference and both auxiliary evaluations once."""
    labels = validate_task(task)
    result = {}
    if not record.get("replay_verified"):
        return {name: {"valid": np.zeros((128, 128), bool), "error": np.zeros((128, 128), np.int64),
                       "final_valid": np.zeros((128, 128), bool), "final_error": np.zeros((128, 128), np.int64)} for name in labels}
    root = Path(suite_root)/"runs"/record["run_id"]/f"attempt-{record['attempt']:03d}"/"submission"
    statuses = {(r["stream"], r["event"]): r["status"] for r in
                (load_json(line) for line in (root/"events.jsonl").read_text().splitlines()) if r["kind"] == "output"}
    images = {}
    def image(stream, event):
        key = (stream, int(event))
        if statuses.get(key) != "valid":
            return None
        if key not in images:
            images[key] = load_rgb_png(root/stream/f"{int(event):04d}.png")
        return images[key]
    for name, label in labels.items():
        queries = task["programs"][name]
        first, returns = label["first_event"], label["reentry_event"]
        reference = np.zeros((128, 128, 3), np.uint8)
        rv = np.zeros((128, 128), bool)
        predicted = np.zeros_like(reference); pv = np.zeros_like(rv)
        for events, dest, valid, prefix in ((first, reference, rv, True), (returns, predicted, pv, False)):
            for event in np.unique(events):
                stream = "prefix" if task["family"] == "F7" and event <= 32 else name
                frame = image(stream, event)
                if frame is None:
                    continue
                mask = events == event
                q = queries[int(event)]
                ix, iy = label["x"][mask] - q[0] + 208, label["y"][mask] - q[1] + 120
                dest[mask] = frame[iy, ix]; valid[mask] = True
        last = image(name, 96)
        final = np.zeros_like(reference); fv = np.zeros_like(rv)
        if last is not None:
            q = queries[-1]
            ix, iy = label["x"] - q[0] + 208, label["y"] - q[1] + 120
            if np.any(ix < 0) or np.any(ix >= 416) or np.any(iy < 0) or np.any(iy >= 240):
                raise ValueError("Final target is not fully visible")
            final = last[iy, ix].copy(); fv[:] = True
        result[name] = {"valid": rv & pv, "error": np.abs(predicted.astype(np.int16) - reference.astype(np.int16)).sum(2, dtype=np.int64),
                        "final_valid": rv & fv, "final_error": np.abs(final.astype(np.int16) - reference.astype(np.int16)).sum(2, dtype=np.int64)}
    return result


def _world_summaries(cells, bootstrap, seed):
    """A world is complete only when every planned pair/origin/seed cell is."""
    summaries = []
    keys = sorted({(c["model_id"], c["role"], c["family"], c["metric"]) for c in cells})
    for mid, role, family, metric in keys:
        selected = [c for c in cells if (c["model_id"], c["role"], c["family"], c["metric"]) == (mid, role, family, metric)]
        worlds = sorted({c["world_id"] for c in selected})
        values, world_rows = [], []
        for wid in worlds:
            group = [c for c in selected if c["world_id"] == wid]
            observed = [c["exact_mae_on_common"] - c["offset_mae_on_common"] for c in group if c["complete_pair"]]
            complete = len(observed) == len(group)
            value = float(np.mean(observed)) if complete else None
            values.append(value)
            world_rows.append({"world_id": wid, "planned_cells": len(group), "complete_cells": len(observed),
                               "cell_coverage": len(observed)/len(group), "complete": complete,
                               "mean_exact_minus_offset": value,
                               "observed_complete_cell_mean": float(np.mean(observed)) if observed else None})
        rng = np.random.default_rng(seed)
        replicates = []
        all_complete = all(v is not None for v in values)
        if all_complete:
            for _ in range(bootstrap):
                replicates.append(float(np.mean([values[i] for i in rng.integers(len(values), size=len(values))])))
        summaries.append({"model_id": mid, "role": role, "family": family, "metric": metric,
                          "planned_worlds": len(worlds), "complete_paired_worlds": sum(v is not None for v in values),
                          "mean_exact_minus_offset": float(np.mean(values)) if all_complete else None,
                          "whole_world_bootstrap_95_interval": np.quantile(replicates, [.025, .975]).tolist() if replicates else None,
                          "valid_bootstrap_replicates": len(replicates), "bootstrap_replicates": bootstrap,
                          "worlds": world_rows})
    return summaries


def aggregate_paired(plan, root, bootstrap=2000, seed=0):
    with suite_read_lock(root):
        return _aggregate_paired(plan, root, bootstrap, seed)


def _aggregate_paired(plan, root, bootstrap, seed):
    validate_paired_plan(plan)
    if type(bootstrap) is not int or bootstrap < 1:
        raise ValueError("Positive bootstrap required")
    analyses = {}
    for selection in ("first_attempt_only", "latest_attempt_within_budget"):
        records = {r["run_id"]: r for r in replay_suite(plan, root, selection)}
        cells = []
        for pid, pair in plan["paired_execution"]["pairs"].items():
            for mid in plan["models"]:
                for inference_seed in sorted({r["seed"] for r in plan["expected"]}):
                    rows = {variant: next(r for r in plan["expected"] if r["task_id"] == pair[variant + "_task_id"] and r["model_id"] == mid and r["seed"] == inference_seed)
                            for variant in ("exact", "offset")}
                    arrays = {v: _target_arrays(plan["tasks"][r["task_id"]], records[r["run_id"]], root) for v, r in rows.items()}
                    for metric, mask_key, error_key in (("reentry", "valid", "error"), ("final_target", "final_valid", "final_error")):
                        common, expected, errors_a, errors_b = 0, 0, 0, 0
                        for stream in arrays["exact"]:
                            a, b = arrays["exact"][stream], arrays["offset"][stream]
                            valid = a[mask_key] & b[mask_key]
                            expected += valid.size; common += int(valid.sum())
                            errors_a += int(a[error_key][valid].sum()); errors_b += int(b[error_key][valid].sum())
                        cells.append({"pair_id": pid, "world_id": pair["world_id"], "model_id": mid, "seed": inference_seed,
                                      "role": pair["role"], "family": pair["family"], "origin_index": pair["origin_index"],
                                      "metric": metric, "expected_pixels": expected, "common_valid_pixels": common,
                                      "common_pixel_coverage": common/expected,
                                      "exact_mae_on_common": errors_a/(255*3*common) if common else None,
                                      "offset_mae_on_common": errors_b/(255*3*common) if common else None,
                                      "complete_pair": common == expected,
                                      "resource_statuses": {v: records[r["run_id"]]["resource_status"] for v, r in rows.items()},
                                      "statuses": {v: records[r["run_id"]]["status"] for v, r in rows.items()}})
        summaries = _world_summaries(cells, bootstrap, seed)
        analyses[selection] = {"cells": cells, "world_summaries": summaries,
                               "records": list(records.values()),
                               "expected_runs": len(plan["expected"]), "executed_runs": sum(r["attempt"] is not None for r in records.values()),
                               "completed_runs": sum(r["status"] == "COMPLETED" for r in records.values()),
                               "failed_or_missing_runs": sum(r["status"] != "COMPLETED" for r in records.values()),
                               "replay_verified_runs": sum(r["replay_verified"] for r in records.values()),
                               "complete_resource_runs": sum(r["resource_status"] == "COMPLETE" for r in records.values()),
                               "integrity_error_runs": sum(bool(r.get("integrity_errors")) for r in records.values()),
                               "planned_outputs": sum(160 if r["family"] == "F7" else 96 for r in plan["expected"]),
                               "valid_outputs": sum((r["reports"]["valid_outputs"] if "valid_outputs" in r["reports"] else r["reports"]["main"]["valid_outputs"]) for r in records.values())}
    attempts = safe_resource_attempts(plan, root)
    first_records = analyses["first_attempt_only"]["records"]
    return {"schema": PAIR_SCHEMA, "study": STUDY, "plan_sha256": plan["plan_sha256"],
            "analyses": analyses, "all_attempt_resources": attempts, "resource_totals": resource_totals(attempts, plan),
            "acceptance": integrity_summary(plan, root, first_records),
            "real_model_validation": "NOT_RUN", "auxiliary_metric": AUXILIARY_METRIC,
            "limitations": ["offset changes geometry; not a pure memory causal intervention", "world resampling does not establish geographic independence", "real_model_validation is not inferred from diagnostic runs"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--task-index", type=Path, required=True); p.add_argument("--world-index", type=Path, required=True)
    p.add_argument("--world-dir", type=Path); p.add_argument("--models", type=Path, required=True)
    p.add_argument("--mechanisms", type=Path); p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--code-commit", required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-attempts-per-run", type=int, default=1)
    for command in ("run", "verify", "aggregate"):
        p = sub.add_parser(command); p.add_argument("--plan", type=Path, required=True); p.add_argument("--root", type=Path, required=True)
        if command == "run":
            p.add_argument("--resume", action="store_true"); p.add_argument("--retry-failed", action="store_true")
        elif command == "aggregate":
            p.add_argument("--out", type=Path, required=True); p.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            policy = {**DEFAULT_RETRY_POLICY, "max_attempts_per_run": args.max_attempts_per_run}
            plan = plan_from_indexes(args.task_index, args.models, args.seeds, args.code_commit, args.world_index, args.world_dir, retry_policy=policy)
            save_plan(attach_pairs(plan, load_json(args.mechanisms.read_text()) if args.mechanisms else None), args.out)
        elif args.command in {"run", "verify"}:
            plan = load_json(args.plan.read_text())
            validate_paired_plan(plan)
            if args.command == "run":
                execute_paired_plan(plan, args.root, args.resume, args.retry_failed)
            rows = replay_suite(plan, args.root)
            return 0 if integrity_summary(plan, args.root, rows)["accepted"] else 1
        else:
            result = aggregate_paired(load_json(args.plan.read_text()), args.root, args.bootstrap)
            _save(args.out, result, exclusive=True)
            return 0 if result["acceptance"]["accepted"] else 1
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
