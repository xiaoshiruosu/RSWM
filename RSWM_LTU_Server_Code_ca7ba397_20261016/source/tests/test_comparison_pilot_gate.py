"""Qualification controls; synthetic CPU evidence never proves a real pilot."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from rswm_inria import experiments as e
from rswm_inria.geometry import digest, file_hash


ROLES = {"reference": "reference", "comparison": "comparison"}


def control_population():
    """Pure contract fixture, with no actual model, images or replay claims."""
    rows, records = [], []
    for model_id in ROLES:
        for world in ("w0", "w1"):
            for family, planned in (("F2", 32), ("F3", 64), ("F5", 96), ("F7", 160)):
                run_id = f"{model_id}-{world}-{family}"
                valid = planned if model_id == "reference" else planned // 2
                rows.append({"run_id": run_id, "model_id": model_id, "world_id": world,
                    "origin_index": 0, "family": family, "seed": 0, "role": "dev", "city": "kitsap",
                    "source_manifest_sha256": "a" * 64,
                    "model_manifest_sha256": digest([model_id, "synthetic schema control"] )})
                records.append({"run_id": run_id, "attempt": 0, "status": "COMPLETED" if valid == planned else "FAILED_GENERATE",
                    "replay_verified": True, "submission_status": "VERIFIED", "resource_status": "COMPLETE",
                    "reports": {"metrics": {}, "valid_outputs": valid, "planned_outputs": planned},
                    "output_slots": [{"valid": n < valid} for n in range(planned)],
                    "resources": {"schema": "rswm-resource-v3", "measurement_complete": True,
                        "accounting_calls": {"valid_outputs": valid, "planned_output_slots": planned}}})
    plan = {"plan_sha256": "b" * 64, "expected": rows,
            "models": {model_id: {"config": {"model_id": model_id, "adapter": "copy"}} for model_id in ROLES}}
    integrity = {"accepted": False, "verified_submissions": 16, "errors": [], "unexpected_runs": [],
                 "all_attempt_errors": [], "resource_incomplete_runs": 0, "resource_incomplete_attempts": 0}
    return plan, records, integrity


def seal(value, key):
    value = {k: v for k, v in value.items() if k != key}
    return {**value, key: digest(value)}


def publish(root, receipt):
    receipt = seal(receipt, "receipt_sha256")
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return receipt


def retained_control(tmp_path, monkeypatch):
    """Physical analysis files + explicit replay mock; not real-pilot evidence."""
    plan, records, integrity = control_population()
    (tmp_path / "suite").mkdir()
    (tmp_path / "analysis").mkdir()
    code = e.code_fingerprint()
    matrix = e._matrix(plan, "pilot")
    experiment = {"experiment_id": "c" * 64, "kind": "pilot", "purpose": "real",
                  "pilot_qualification_roles": ROLES,
                  "budget": {"max_wall_seconds": 100., "max_output_bytes": 1_000_000}}
    configuration = {"bootstrap": 5, "bootstrap_seed": 0, "selection": "first_attempt_only"}
    report = {"analysis_configuration": configuration, "input_results_sha256": digest(records),
              "acceptance_status": "INCOMPLETE", "fixture": "synthetic contract control"}
    aggregation = tmp_path / "analysis/aggregation.json"
    aggregation.write_text(json.dumps(report), encoding="utf-8")
    table = tmp_path / "analysis/results.csv"
    table.write_text("synthetic schema control; no real inference\n", encoding="utf-8")
    analysis = {"plan_sha256": plan["plan_sha256"], "source_code_sha256": code,
        "analysis_status": "ANALYSIS_WRITTEN", "acceptance_status": "INCOMPLETE",
        "analysis_configuration": configuration, "input_results_sha256": report["input_results_sha256"],
        "files": {"aggregation.json": file_hash(aggregation), "results.csv": file_hash(table)}}
    provenance = tmp_path / "analysis/provenance.json"
    provenance.write_text(json.dumps(analysis), encoding="utf-8")
    (tmp_path / "experiment.json").write_text(json.dumps(experiment), encoding="utf-8")
    (tmp_path / "suite/plan.json").write_text(json.dumps(plan), encoding="utf-8")
    receipt = {"schema": "rswm-experiment-receipt-v1", "kind": "pilot", "purpose": "real",
        "acceptance_status": e.COMPARISON_PILOT, "pilot_qualification_roles": ROLES,
        "source_manifest_sha256": "a" * 64, "source_code_sha256": code,
        "experiment_id": experiment["experiment_id"], "plan_sha256": plan["plan_sha256"],
        "model_manifest_sha256": sorted({r["model_manifest_sha256"] for r in plan["expected"]}),
        "matrix": matrix, "expected_runs": 16, "verified_outputs": 1056, "integrity": integrity,
        "evidence_root": str(tmp_path), "receipt_path": "receipt.json",
        "analysis_provenance_path": "analysis/provenance.json", "analysis_provenance_sha256": file_hash(provenance),
        "budget_usage": {"exhausted": False, "elapsed_total_seconds": 1., "output_bytes": 1000,
                         "reserved_publication_bytes": 4096}}
    monkeypatch.setattr(e, "validate_experiment", lambda *args: None)
    monkeypatch.setattr(e, "replay_suite", lambda *args: records)
    monkeypatch.setattr(e, "integrity_summary", lambda *args: integrity)
    monkeypatch.setattr(e, "aggregate_suite", lambda *args, **kwargs: report)
    return publish(tmp_path, receipt), plan, records, integrity


def test_reference_complete_comparison_failure_is_qualification_not_acceptance():
    plan, records, integrity = control_population()
    assert e._pilot_execution_status(plan, records, integrity, ROLES) == e.COMPARISON_PILOT
    assert integrity["accepted"] is False
    assert sum(r["reports"]["valid_outputs"] for r in records) == 1056
    assert e._matrix(plan, "pilot")["planned_unique_outputs"] == 1408
    assert sum(r["reports"]["planned_outputs"] for r in records if r["run_id"].endswith("F7")) == 640
    broken_reference = copy.deepcopy(records)
    broken_reference[0]["status"] = "FAILED_GENERATE"
    broken_reference[0]["reports"]["valid_outputs"] -= 1
    broken_reference[0]["resources"]["accounting_calls"]["valid_outputs"] -= 1
    broken_reference[0]["output_slots"][-1]["valid"] = False
    with pytest.raises(ValueError, match="Reference pilot"):
        e._pilot_execution_status(plan, broken_reference, integrity, ROLES)


def test_comparison_zero_valid_outputs_still_retains_all_executed_slots():
    plan, records, integrity = control_population()
    for record in records[8:]:
        record["reports"]["valid_outputs"] = 0
        record["resources"]["accounting_calls"]["valid_outputs"] = 0
        record["output_slots"] = [{"valid": False} for _ in record["output_slots"]]
    assert e._pilot_execution_status(plan, records, integrity, ROLES) == e.COMPARISON_PILOT
    assert sum(r["reports"]["valid_outputs"] for r in records) == 704
    assert e._matrix(plan, "pilot")["planned_unique_outputs"] == 1408


def test_formal_terminal_trace_allows_reference_failure_after_strict_pilot():
    """Pure retained-trace contract, not an executed 24-world formal matrix."""
    plan, records, integrity = control_population()
    reference = records[0]
    reference["status"] = "FAILED_GENERATE"
    reference["reports"]["valid_outputs"] -= 1
    reference["resources"]["accounting_calls"]["valid_outputs"] -= 1
    reference["output_slots"][-1]["valid"] = False
    assert e._pilot_execution_status(plan, records, integrity, ROLES, formal=True) == e.FORMAL_RETAINED_FAILURES
    assert integrity["accepted"] is False
    with pytest.raises(ValueError, match="Reference pilot"):
        e._pilot_execution_status(plan, records, integrity, ROLES)


def test_recomputed_analysis_only_ignores_metric_table_row_order():
    rows = [{"metric": "overlap", "expected_pixels": 10, "valid_pixels": 5},
            {"metric": "history", "expected_pixels": 8, "valid_pixels": 3}]
    saved = {field: copy.deepcopy(rows) for field in ("cells", "first_attempt", "recovery_attempt")}
    saved["replay_status"] = [{"event": 1}, {"event": 2}]
    fresh = copy.deepcopy(saved)
    for field in ("cells", "first_attempt", "recovery_attempt"):
        fresh[field].reverse()
    assert e._same_aggregation(saved, fresh)
    changed = copy.deepcopy(fresh)
    changed["cells"][0]["valid_pixels"] += 1
    assert not e._same_aggregation(saved, changed)
    missing = copy.deepcopy(fresh)
    missing["cells"].pop()
    assert not e._same_aggregation(saved, missing)
    reordered_trace = copy.deepcopy(fresh)
    reordered_trace["replay_status"].reverse()
    assert not e._same_aggregation(saved, reordered_trace)


@pytest.mark.parametrize("fault", ["unattempted", "load", "replay", "resource", "identity"])
def test_formal_retained_failure_status_requires_full_trustworthy_execution(fault):
    """Pure contract negative; no claim of real formal execution."""
    plan, records, integrity = control_population()
    if fault == "unattempted": records[8]["attempt"] = None
    elif fault == "load": records[8]["status"] = "FAILED_LOAD"
    elif fault == "replay": records[8]["replay_verified"] = False
    elif fault == "resource": integrity["resource_incomplete_attempts"] = 1
    else: integrity["all_attempt_errors"] = [{"code": "MODEL_IDENTITY_MISMATCH"}]
    with pytest.raises(ValueError, match="verification|complete attempted trace"):
        e._pilot_execution_status(plan, records, integrity, ROLES, formal=True)


@pytest.mark.parametrize("fault", ["not-run", "load", "crash", "replay", "attempt", "missing-slots",
                                  "resources", "accounting", "metadata", "old-resources", "extra-run"])
def test_comparison_never_qualifies_unattempted_or_corrupt_evidence(fault):
    plan, records, integrity = control_population()
    row = records[8]
    if fault in {"not-run", "load", "crash"}:
        row["status"] = {"not-run": "NOT_RUN", "load": "FAILED_LOAD", "crash": "CRASHED"}[fault]
    elif fault == "replay": row["replay_verified"] = False
    elif fault == "attempt": row["attempt"] = 1
    elif fault == "missing-slots": row["output_slots"].pop()
    elif fault == "resources": row["resources"]["measurement_complete"] = False
    elif fault == "accounting": row["resources"]["accounting_calls"]["valid_outputs"] -= 1
    elif fault == "metadata": integrity["errors"] = [{"code": "IDENTITY_MISMATCH"}]
    elif fault == "old-resources": integrity["resource_incomplete_attempts"] = 1
    else: integrity["unexpected_runs"] = ["unplanned"]
    with pytest.raises(ValueError, match="verification|complete attempted trace"):
        e._pilot_execution_status(plan, records, integrity, ROLES)


@pytest.mark.parametrize("roles", [{}, {"reference": "reference"}, {"reference": "reference", "comparison": "optional"},
                                  {"reference": "reference", "comparison": []}])
def test_qualification_roles_bind_the_full_frozen_population(roles):
    plan, _, _ = control_population()
    with pytest.raises(ValueError, match="qualification roles"):
        e._pilot_roles(plan, roles)


def test_formal_requires_a_reference_and_default_remains_strict():
    plan, _, _ = control_population()
    assert e._pilot_roles(plan, None, formal=True) is None
    with pytest.raises(ValueError, match="at least one reference"):
        e._pilot_roles(plan, {model_id: "comparison" for model_id in ROLES}, formal=True)


def test_default_freeze_serialization_does_not_add_a_role_field(monkeypatch):
    plan, _, _ = control_population()
    plan["retry_policy"] = {"primary_selection": "first_attempt_only"}
    monkeypatch.setattr(e, "validate_plan", lambda *args: None)
    options = {"kind": "pilot", "purpose": "diagnostic", "budget": {"max_wall_seconds": 100., "max_output_bytes": 1_000_000}}
    before = e.freeze_experiment(plan, **options)
    explicit_default = e.freeze_experiment(plan, **options, pilot_qualification_roles=None)
    assert before == explicit_default and "pilot_qualification_roles" not in before
    with pytest.raises(ValueError, match="actual real-generator"):
        e.freeze_experiment(plan, **options, pilot_qualification_roles=ROLES)


def test_retained_comparison_requires_frozen_roles_and_independent_analysis(tmp_path, monkeypatch):
    receipt, plan, _, _ = retained_control(tmp_path, monkeypatch)
    expected = {row["model_manifest_sha256"] for row in plan["expected"]}
    assert e.validate_pilot_receipt(receipt, plan, pilot_qualification_roles=ROLES) == expected
    with pytest.raises(ValueError, match="actual real acceptance"):
        e.validate_pilot_receipt(receipt, plan)
    with pytest.raises(ValueError, match="preregistered formal roles"):
        e.validate_pilot_receipt(receipt, plan, pilot_qualification_roles={model_id: "reference" for model_id in ROLES})
    monkeypatch.setattr(e, "aggregate_suite", lambda *args, **kwargs: {"changed": "independent replay result"})
    with pytest.raises(ValueError, match="independently recomputed failures"):
        e.validate_pilot_receipt(receipt, plan, pilot_qualification_roles=ROLES)


@pytest.mark.parametrize("fault", ["budget", "fake-complete", "role-change", "output-count", "analysis-tamper"])
def test_comparison_receipt_never_upgrades_missing_or_changed_evidence(tmp_path, monkeypatch, fault):
    receipt, plan, _, _ = retained_control(tmp_path, monkeypatch)
    if fault == "budget": receipt["budget_usage"]["exhausted"] = True
    elif fault == "fake-complete": receipt["acceptance_status"] = "ACCEPTED_REAL_PILOT"
    elif fault == "role-change": receipt["pilot_qualification_roles"] = {model_id: "reference" for model_id in ROLES}
    elif fault == "output-count": receipt["verified_outputs"] = 1408
    else: (tmp_path / "analysis/results.csv").write_text("tampered\n", encoding="utf-8")
    receipt = publish(tmp_path, receipt)
    with pytest.raises(ValueError, match="budget|qualification|denominator|analysis artifact"):
        e.validate_pilot_receipt(receipt, plan, pilot_qualification_roles=ROLES)


@pytest.mark.parametrize("status", [e.COMPARISON_PILOT, e.FORMAL_RETAINED_FAILURES])
def test_comparison_cli_exit_does_not_claim_all_success(monkeypatch, status):
    monkeypatch.setattr(e, "_read", lambda path: {})
    monkeypatch.setattr(e, "execute_experiment", lambda *args, **kwargs: {"acceptance_status": status})
    assert e.main(["run", "--plan", "unused", "--experiment", "unused", "--out", "unused"]) == 1


@pytest.mark.skipif(importlib.util.find_spec("fcntl") is None, reason="Actual suite needs Unix fcntl; never mock its lock")
def test_actual_synthetic_suite_preserves_comparison_failures_and_denominator(tmp_path, monkeypatch):
    """Actual PNG/journal/resources/replay/aggregate; synthetic generator only.

    Mock ONLY real-asset eligibility, because this fixture cannot be a real
    source/weight pilot. The exercised downstream evidence is genuinely produced.
    """
    from rswm_inria import registry, materialized, suite
    from rswm_inria.baselines import CopyModel
    from rswm_inria.tasks import packages

    class MissingHalf(CopyModel):
        def generate(self, queries):
            images = super().generate(queries)
            return [image if index % 2 == 0 else None for index, image in enumerate(images)]

    inputs = []
    for index in range(2):
        world_path = tmp_path / f"synthetic-world-{index}.png"
        Image.fromarray(np.zeros((2048, 2048, 3), dtype=np.uint8)).save(world_path)
        for task, _ in packages(f"synthetic-pilot-{index}", "dev", 0):
            task_path = tmp_path / (task["task_id"] + ".json")
            task_path.write_text(json.dumps(task), encoding="utf-8")
            inputs.append({"task_path": str(task_path), "world_path": str(world_path), "city": "kitsap"})
    plan = suite.build_plan(inputs, [{"model_id": model_id, "adapter": "copy", "device": "cpu"} for model_id in ROLES],
                            [0], "a" * 64, "b" * 40)
    monkeypatch.setattr(e, "capabilities", lambda config: {"purpose": "real_generator"})
    monkeypatch.setattr(materialized, "validate_audit_receipt", lambda *args, **kwargs: {"synthetic_control_only": True})
    monkeypatch.setattr(registry, "create_adapter", lambda config: MissingHalf() if config["model_id"] == "comparison" else CopyModel())
    experiment = e.freeze_experiment(plan, kind="pilot", purpose="real",
        budget={"max_wall_seconds": 600., "max_output_bytes": 100_000_000},
        source_audit={"synthetic_control_only": True},
        rights={"data_use_approved": True, "weight_use_approved": True, "scope": "synthetic software control only"},
        pilot_qualification_roles=ROLES)
    root = tmp_path / "synthetic-experiment"
    receipt = e.execute_experiment(experiment, plan, root)
    assert receipt["acceptance_status"] == e.COMPARISON_PILOT
    assert receipt["matrix"]["expected_runs"] == 16 and receipt["matrix"]["planned_unique_outputs"] == 1408
    assert receipt["verified_outputs"] == 1056
    assert receipt["integrity"]["accepted"] is False and receipt["integrity"]["verified_submissions"] == 16
    aggregation = json.loads((root / "analysis/aggregation.json").read_text())
    assert aggregation["acceptance_status"] == "INCOMPLETE"
    assert aggregation["failure_summary"]["expected_runs"] == 16
    assert len(aggregation["failure_summary"]["failed_or_unverifiable_runs"]) == 8
    assert aggregation["primary_selection"] == "first_attempt_only"
    assert e.validate_pilot_receipt(receipt, plan, pilot_qualification_roles=ROLES) == {r["model_manifest_sha256"] for r in plan["expected"]}
    with pytest.raises(ValueError, match="actual real acceptance"):
        e.validate_pilot_receipt(receipt, plan)
