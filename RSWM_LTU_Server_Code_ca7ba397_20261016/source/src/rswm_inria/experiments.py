"""Executable pilot/formal gates over the immutable suite, without downloads.

Receipts are reproducibility records, not cryptographic attestations. A diagnostic
execution can exercise the software but cannot authorize a formal model study.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import time
from .geometry import canonical, digest, file_hash
from .suite import validate_plan, execute_plan, replay_suite, integrity_summary, code_fingerprint, _json, _save
from .registry import capabilities
from .export_tables import export_tables
from .aggregate import aggregate_suite

SCHEMA = "rswm-experiment-v1"
COMPARISON_PILOT = "EXECUTED_REAL_COMPARISON_PILOT_WITH_FAILURES"
FORMAL_RETAINED_FAILURES = "EXECUTED_FORMAL_WITH_RETAINED_FAILURES_PENDING_INSTRUMENT_ANALYSIS"
MODEL_FAILURES = {"INCOMPLETE", "FAILED_INITIALIZATION", "FAILED_GENERATE",
                  "FAILED_SNAPSHOT", "FAILED_RESTORE"}


def _read(path):
    return _json(path)


def _write(path, value):
    p = Path(path)
    if p.is_symlink() or any(x.is_symlink() for x in p.parents):
        raise ValueError("Experiment path symlink")
    if p.exists():
        if _read(p) != value: raise FileExistsError(p)
        return
    _save(p, value, exclusive=True)


def _sealed(value, field):
    if not isinstance(value, dict) or value.get(field) != digest({k: v for k, v in value.items() if k != field}):
        raise ValueError("Receipt checksum mismatch")


def _source_id(plan):
    ids={r["source_manifest_sha256"] for r in plan["expected"]}
    if len(ids)!=1:raise ValueError("Experiment source identity mismatch")
    return next(iter(ids))


def _verified_outputs(record):
    if not record.get("replay_verified"):return 0
    reports=record["reports"]
    streams=[reports] if "metrics" in reports else reports.values()
    return sum(s["valid_outputs"] for s in streams)


def _receipt_evidence_path(root,relative):
    if not isinstance(relative,str):raise ValueError("Pilot evidence path missing")
    p=Path(relative)
    if p.is_absolute() or ".." in p.parts:raise ValueError("Pilot evidence path escape")
    target=root/p
    if target.is_symlink() or any(a.is_symlink() for a in target.parents):
        raise ValueError("Pilot evidence path symlink")
    return target


def _pilot_roles(plan, roles, *, formal=False):
    """An absent mapping preserves the original all-models-strict contract."""
    if roles is None:
        return None
    if (not isinstance(roles, dict) or set(roles) != set(plan["models"]) or
        any(not isinstance(role, str) or role not in {"reference", "comparison"} for role in roles.values())):
        raise ValueError("Pilot qualification roles must cover every frozen model with reference/comparison")
    if formal and "reference" not in roles.values():
        raise ValueError("Formal experiment requires at least one reference qualification")
    return dict(sorted(roles.items()))


def _pilot_execution_status(plan, records, integrity, roles, *, formal=False):
    """Qualify evidenced model failures without accepting missing execution.

    Suite and analysis acceptance remain strict. Only the experiment prerequisite
    distinguishes an intact model failure from infrastructure corruption. The
    pilot reference must be complete; formal reference failures are outcomes.
    """
    expected = {row["run_id"]: row for row in plan["expected"]}
    if (len(records) != len(expected) or {r.get("run_id") for r in records} != set(expected) or
        integrity.get("verified_submissions") != len(expected) or
        integrity.get("unexpected_runs") or integrity.get("errors") or integrity.get("all_attempt_errors") or
        integrity.get("resource_incomplete_runs") != 0 or integrity.get("resource_incomplete_attempts") != 0):
        raise ValueError("Pilot retained submissions/resources failed independent verification")
    model_failed = False
    for record in records:
        row = expected[record["run_id"]]
        resource = record.get("resources") or {}
        calls = resource.get("accounting_calls") or {}
        planned = {"F2": 32, "F3": 64, "F5": 96, "F7": 160}[row["family"]]
        slots = record.get("output_slots")
        outputs = _verified_outputs(record)
        if (record.get("replay_verified") is not True or record.get("submission_status") != "VERIFIED" or
            type(record.get("attempt")) is not int or record["attempt"] != 0 or
            record.get("resource_status") != "COMPLETE" or resource.get("schema") != "rswm-resource-v3" or
            resource.get("measurement_complete") is not True or calls.get("planned_output_slots") != planned or
            type(calls.get("valid_outputs")) is not int or calls["valid_outputs"] != outputs or
            not isinstance(slots, list) or len(slots) != planned or
            any(type(slot.get("valid")) is not bool for slot in slots) or
            sum(slot["valid"] for slot in slots) != outputs or not 0 <= outputs <= planned or
            record.get("status") not in MODEL_FAILURES | {"COMPLETED"} or
            (record.get("status") == "COMPLETED") != (outputs == planned)):
            raise ValueError("Pilot comparison requires complete attempted trace, replay and resources")
        if roles[row["model_id"]] == "reference" and not formal:
            if record["status"] != "COMPLETED" or outputs != planned:
                raise ValueError("Reference pilot requires all eight packages and 704 complete outputs")
        elif record["status"] != "COMPLETED" or outputs != planned:
            model_failed = True
    if formal:
        return FORMAL_RETAINED_FAILURES if model_failed else "EXECUTED_FORMAL_PENDING_INSTRUMENT_ANALYSIS"
    return COMPARISON_PILOT if model_failed else "ACCEPTED_REAL_PILOT"


def _same_aggregation(saved, recomputed):
    """Metric cell tables have no row-order semantics; traces still do.

    Canonical plan JSON sorts planned-report dictionaries, while a freshly built
    plan can enumerate them in construction order. Preserve every row and value.
    """
    def ordered(value):
        if not isinstance(value, dict):
            return value
        result = dict(value)
        for field in ("cells", "first_attempt", "recovery_attempt"):
            if isinstance(result.get(field), list):
                result[field] = sorted(result[field], key=canonical)
        return result
    return digest(ordered(saved)) == digest(ordered(recomputed))


def validate_pilot_receipt(receipt, plan, *, pilot_qualification_roles=None):
    """Recompute retained pilot submissions; a thin self-signed PASS is invalid.

    This is internal evidence consistency, not external timestamp authentication.
    Original backend execution is never rerun while checking its retained PNGs.
    """
    _sealed(receipt,"receipt_sha256")
    requested_roles = _pilot_roles(plan, pilot_qualification_roles)
    permitted_statuses = {"ACCEPTED_REAL_PILOT"} | ({COMPARISON_PILOT} if requested_roles is not None else set())
    if (receipt.get("schema")!="rswm-experiment-receipt-v1" or receipt.get("kind")!="pilot" or
        receipt.get("purpose")!="real" or receipt.get("acceptance_status") not in permitted_statuses):
        raise ValueError("Pilot has no actual real acceptance")
    if receipt.get("source_manifest_sha256")!=_source_id(plan) or receipt.get("source_code_sha256")!=code_fingerprint():
        raise ValueError("Pilot source/code identity mismatch")
    value=receipt.get("evidence_root")
    if not isinstance(value,str) or not value:raise ValueError("Pilot retained evidence root required")
    root=Path(value)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):raise ValueError("Pilot evidence root symlink")
    if _read(_receipt_evidence_path(root,receipt.get("receipt_path")))!=receipt:
        raise ValueError("Pilot published receipt identity mismatch")
    actual_plan=_read(root/"suite/plan.json")
    actual_experiment=_read(root/"experiment.json")
    validate_experiment(actual_experiment,actual_plan)
    if actual_experiment["kind"]!="pilot" or actual_experiment["purpose"]!="real" or actual_experiment["experiment_id"]!=receipt.get("experiment_id"):
        raise ValueError("Pilot frozen experiment mismatch")
    actual_roles = _pilot_roles(actual_plan, actual_experiment.get("pilot_qualification_roles"))
    if receipt.get("pilot_qualification_roles") != actual_roles:
        raise ValueError("Pilot frozen qualification roles mismatch")
    if actual_roles is not None:
        desired = ({model_id: "reference" for model_id in actual_roles} if requested_roles is None else
                   {model_id: requested_roles.get(model_id) for model_id in actual_roles})
        if actual_roles != desired:
            raise ValueError("Pilot qualification roles differ from preregistered formal roles")
    elif requested_roles is not None and any(requested_roles.get(model_id) != "reference" for model_id in actual_plan["models"]):
        raise ValueError("Pilot qualification roles must be frozen before comparison execution")
    matrix=_matrix(actual_plan,"pilot")
    if actual_plan["plan_sha256"]!=receipt.get("plan_sha256") or receipt.get("matrix")!=matrix:
        raise ValueError("Pilot task matrix identity mismatch")
    if _source_id(actual_plan)!=_source_id(plan) or receipt.get("expected_runs")!=matrix["expected_runs"]:
        raise ValueError("Pilot source/population mismatch")
    expected_models=sorted({r["model_manifest_sha256"] for r in actual_plan["expected"]})
    if receipt.get("model_manifest_sha256")!=expected_models:
        raise ValueError("Pilot frozen model identity mismatch")
    records=replay_suite(actual_plan,root/"suite")
    integrity=integrity_summary(actual_plan,root/"suite",records)
    if integrity!=receipt.get("integrity") or integrity["verified_submissions"]!=matrix["expected_runs"]:
        raise ValueError("Pilot retained submissions/resources failed independent verification")
    if actual_roles is None:
        if not integrity["accepted"]:
            raise ValueError("Pilot retained submissions/resources failed independent verification")
    elif _pilot_execution_status(actual_plan, records, integrity, actual_roles) != receipt["acceptance_status"]:
        raise ValueError("Pilot qualification status differs from independently verified execution")
    outputs=sum(r["resources"]["accounting_calls"]["valid_outputs"] for r in records)
    if (actual_roles is None and outputs!=matrix["planned_unique_outputs"]) or outputs!=receipt.get("verified_outputs"):
        raise ValueError("Pilot output denominator mismatch")
    usage=receipt.get("budget_usage",{})
    limits=actual_experiment["budget"]
    if (usage.get("exhausted") is not False or type(usage.get("elapsed_total_seconds")) not in (int,float) or
        not math.isfinite(usage["elapsed_total_seconds"]) or not 0<=usage["elapsed_total_seconds"]<limits["max_wall_seconds"] or
        type(usage.get("output_bytes")) is not int or usage["output_bytes"]<0 or
        type(usage.get("reserved_publication_bytes")) is not int or usage["reserved_publication_bytes"]<0 or
        usage["output_bytes"]+usage["reserved_publication_bytes"]>=limits["max_output_bytes"]):
        raise ValueError("Pilot execution budget was not accepted")
    provenance=_receipt_evidence_path(root,receipt.get("analysis_provenance_path"))
    if file_hash(provenance)!=receipt.get("analysis_provenance_sha256"):
        raise ValueError("Pilot analysis provenance changed")
    analysis=_read(provenance)
    expected_analysis = "ACCEPTED" if integrity["accepted"] else "INCOMPLETE"
    if analysis.get("plan_sha256")!=actual_plan["plan_sha256"] or analysis.get("source_code_sha256")!=code_fingerprint() or analysis.get("acceptance_status")!=expected_analysis:
        raise ValueError("Pilot analysis identity/acceptance mismatch")
    if not isinstance(analysis.get("files"),dict) or not analysis["files"]:
        raise ValueError("Pilot analysis artifacts missing")
    for relative,sha in analysis["files"].items():
        artifact=_receipt_evidence_path(provenance.parent,relative)
        if file_hash(artifact)!=sha:raise ValueError("Pilot retained analysis artifact changed")
    if actual_roles is not None:
        if analysis.get("analysis_status") != "ANALYSIS_WRITTEN" or "aggregation.json" not in analysis["files"]:
            raise ValueError("Pilot complete retained analysis required")
        configuration = analysis.get("analysis_configuration") or {}
        bootstrap, seed = configuration.get("bootstrap"), configuration.get("bootstrap_seed")
        if type(bootstrap) is not int or bootstrap < 1 or type(seed) is not int or seed < 0:
            raise ValueError("Pilot retained analysis configuration invalid")
        recomputed = aggregate_suite(actual_plan, root/"suite", bootstrap=bootstrap, seed=seed)
        if (not _same_aggregation(_read(provenance.parent/"aggregation.json"), recomputed) or
            analysis.get("input_results_sha256") != recomputed["input_results_sha256"] or
            configuration != recomputed["analysis_configuration"]):
            raise ValueError("Pilot retained analysis differs from independently recomputed failures")
    return set(expected_models)


def _instrument_receipts(receipts,source_audit,source_id,expected_configs):
    """Delegate release decisions to each instrument's actual strict validator."""
    identities={}
    import re
    if not isinstance(expected_configs,dict) or set(expected_configs)!={"motion","quality"} or any(not isinstance(v,str) or not re.fullmatch(r"[0-9a-f]{64}",v) for v in expected_configs.values()):
        raise ValueError("BLOCKED: preregistered motion/quality config SHA256 required")
    for receipt in receipts:
        if not isinstance(receipt,dict):raise ValueError("Instrument release receipt required")
        schema=receipt.get("schema")
        if schema=="rswm-motion-validation-v1":
            from .motion_protocol import validate_release
            validate_release(receipt,receipt["config"])
            name="motion";config_id=receipt["config"]["config_sha256"]
            if config_id!=expected_configs[name]:raise ValueError("Preregistered motion config mismatch")
        elif schema=="rswm-feature-release-validation-v1":
            from .features import validate_feature_release
            config_id=expected_configs["quality"]
            validate_feature_release(receipt,config_id,source_id)
            name="quality"
        else:
            raise ValueError("BLOCKED: actual motion/quality release receipt required; thin PASS or candidate controls are insufficient")
        if name in identities:raise ValueError("Duplicate instrument release receipt")
        if receipt.get("source_manifest_sha256")!=source_id:raise ValueError("Instrument source manifest mismatch")
        for world in receipt.get("source_worlds",[]):
            audited=source_audit["worlds"].get(world["world_id"])
            if not audited or any(world.get(k)!=audited.get(k) for k in ("city","role","world_pixel_sha256","world_file_sha256","source_sha256")):
                raise ValueError("Instrument audited source/role identity mismatch")
        identities[name]={"config_sha256":config_id,"validation_sha256":receipt["validation_sha256"],
                          "source_manifest_sha256":source_id}
    if set(identities)!={"motion","quality"}:raise ValueError("BLOCKED: both frozen instruments required")
    return identities


def validate_smoke12_evidence(evidence_root,source_id):
    """Full real-source diagnostic acceptance is distinct from core120 pixels."""
    if not isinstance(evidence_root,str) or not evidence_root:
        raise ValueError("BLOCKED: retained real smoke12 diagnostic evidence required")
    root=Path(evidence_root)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):raise ValueError("Smoke12 evidence root symlink")
    report=_read(root/"acceptance.json");plan=_read(root/"plan.json");audit=_read(root/"input-audit.json")
    from .materialized import validate_audit_receipt
    from .smoke12 import DIAGNOSTIC_MODELS
    validate_plan(plan);validate_audit_receipt(audit,plan,required_profile="smoke12")
    failed=report.get("failed_runs")
    # The canonical producer retains failure rows; older receipts used a count.
    no_failures=(type(failed) is list and not failed) or (type(failed) is int and failed==0)
    if (report.get("schema")!="rswm-smoke12-acceptance-v1" or report.get("status")!="REAL_DATA_VALIDATED" or
        report.get("full_smoke12_matrix") is not True or report.get("real_data_validated") is not True or
        report.get("input_kind")!="source_identity_checked_geography_unknown" or report.get("models")!=DIAGNOSTIC_MODELS or
        report.get("planned_runs_total")!=576 or report.get("planned_unique_outputs_total")!=50_688 or
        not no_failures or report.get("analysis_status")!="ACCEPTED"):
        raise ValueError("BLOCKED: full real smoke12 diagnostic matrix was not accepted")
    if report.get("plan_sha256")!=plan["plan_sha256"] or report.get("source_code_sha256")!=code_fingerprint() or _source_id(plan)!=source_id:
        raise ValueError("Smoke12 source/code/plan identity mismatch")
    if (len(plan["expected"])!=576 or len({r["task_id"] for r in plan["expected"]})!=192 or
        len({r["world_id"] for r in plan["expected"]})!=12 or {r["seed"] for r in plan["expected"]}!={0} or
        set(plan["models"])!={m["model_id"] for m in DIAGNOSTIC_MODELS}):
        raise ValueError("Smoke12 expected matrix differs")
    for model in DIAGNOSTIC_MODELS:
        frozen=plan["models"][model["model_id"]]["config"]
        if frozen!={**model,"device":"cpu"}:raise ValueError("Smoke12 diagnostic adapter configuration changed")
    records=replay_suite(plan,root/"suite");integrity=integrity_summary(plan,root/"suite",records)
    if not integrity["accepted"] or integrity["verified_submissions"]!=576 or sum(_verified_outputs(r) for r in records)!=50_688:
        raise ValueError("Smoke12 retained submissions/resources failed independent verification")
    artifacts=report.get("artifact_sha256")
    if not isinstance(artifacts,dict) or not artifacts:raise ValueError("Smoke12 retained artifact manifest missing")
    for relative,sha in artifacts.items():
        if file_hash(_receipt_evidence_path(root,relative))!=sha:raise ValueError("Smoke12 retained artifact changed")
    return {"evidence_root":str(root.resolve()),"acceptance_file_sha256":file_hash(root/"acceptance.json"),
            "plan_sha256":plan["plan_sha256"],"source_manifest_sha256":source_id,
            "planned_runs":576,"verified_outputs":50_688}


def _matrix(plan, kind):
    """Every model/seed has the same predeclared cartesian task matrix."""
    rows = plan["expected"]
    worlds = {r["world_id"] for r in rows}
    role = "dev" if kind == "pilot" else "test_unseen_city"
    origins = {r["origin_index"] for r in rows}
    if any(r["role"] != role for r in rows): raise ValueError("Experiment role violation")
    if len(worlds) != (2 if kind == "pilot" else 24): raise ValueError("World quota violation")
    if len(origins) != (1 if kind == "pilot" else 4): raise ValueError("Origin quota violation")
    if kind == "formal" and ({r["city"] for r in rows} != {"sfo", "innsbruck"} or
                             any(len({r["world_id"] for r in rows if r["city"] == c}) != 12 for c in ("sfo", "innsbruck"))):
        raise ValueError("Formal city quota violation")
    groups = {}
    for row in rows:
        groups.setdefault((row["model_id"], row["seed"]), []).append(row)
    expected = {(w, o, f) for w in worlds for o in origins for f in ("F2", "F3", "F5", "F7")}
    for group in groups.values():
        keys = [(r["world_id"], r["origin_index"], r["family"]) for r in group]
        if len(keys) != len(expected) or set(keys) != expected: raise ValueError("Incomplete experiment matrix")
    if kind == "pilot" and len({r["seed"] for r in rows}) != 1: raise ValueError("Pilot requires one inference seed")
    return {"expected_runs": len(rows), "worlds": len(worlds), "origins": len(origins),
            "planned_unique_outputs": sum({"F2": 32, "F3": 64, "F5": 96, "F7": 160}[r["family"]] for r in rows),
            "model_seed_groups": len(groups)}


def freeze_experiment(plan, *, kind, purpose, budget, source_audit=None,
                      pilot_receipts=(), metric_receipts=(), rights=None,
                      instrument_config_sha256=None,smoke12_evidence_root=None,
                      pilot_qualification_roles=None):
    validate_plan(plan)
    if kind not in {"pilot", "formal"} or purpose not in {"real", "diagnostic"}: raise ValueError("Unknown experiment type")
    if not isinstance(budget, dict) or set(budget) != {"max_wall_seconds", "max_output_bytes"}:
        raise ValueError("Explicit execution budget required")
    if any(type(v) not in {int, float} or not math.isfinite(v) or v <= 0 for v in budget.values()):
        raise ValueError("Positive finite budget required")
    if type(budget["max_output_bytes"]) is not int: raise ValueError("Byte budget must be integer")
    matrix = _matrix(plan, kind)
    qualification_roles = _pilot_roles(plan, pilot_qualification_roles, formal=kind=="formal")
    if qualification_roles is not None and purpose != "real":
        raise ValueError("Pilot qualification roles require an actual real-generator experiment")
    real_models = all(capabilities(m["config"])["purpose"] == "real_generator" for m in plan["models"].values())
    if purpose == "real" and not real_models: raise ValueError("Diagnostic adapter cannot stand in for a real generator")
    if kind == "formal" and purpose != "real": raise ValueError("Diagnostic execution cannot authorize formal test")
    dependencies = {}
    if purpose == "real":
        from .materialized import validate_audit_receipt
        validate_audit_receipt(source_audit,plan,required_profile="core120" if kind=="formal" else None)
        if not rights or rights.get("data_use_approved") is not True or rights.get("weight_use_approved") is not True:
            raise ValueError("BLOCKED: data and weight use decisions absent")
        dependencies["source_audit"] = source_audit
    if kind == "formal":
        dependencies["smoke12"]=validate_smoke12_evidence(smoke12_evidence_root,_source_id(plan))
        locked_models = {r["model_manifest_sha256"] for r in plan["expected"]}
        accepted_models = set()
        for receipt in pilot_receipts:
            accepted_models.update(validate_pilot_receipt(receipt,plan,pilot_qualification_roles=qualification_roles))
        if not locked_models.issubset(accepted_models): raise ValueError("BLOCKED: each current model needs its own real pilot")
        instruments=_instrument_receipts(metric_receipts,source_audit,_source_id(plan),instrument_config_sha256)
        dependencies.update(pilot_receipts=list(pilot_receipts),metric_receipts=list(metric_receipts),instruments=instruments)
    value = {"schema": SCHEMA, "kind": kind, "purpose": purpose, "plan_sha256": plan["plan_sha256"],
             "source_code_sha256": code_fingerprint(), "retry_policy": plan["retry_policy"],
             "budget": budget, "matrix": matrix, "dependencies": dependencies, "rights": rights,
             "instrument_config_sha256":instrument_config_sha256 if kind=="formal" else None}
    if qualification_roles is not None:
        value["pilot_qualification_roles"] = qualification_roles
    value["experiment_id"] = digest(value)
    return value


def validate_experiment(experiment, plan):
    validate_plan(plan); _sealed(experiment, "experiment_id")
    if experiment.get("schema") != SCHEMA or experiment["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("Experiment/plan mismatch")
    if experiment["source_code_sha256"] != code_fingerprint() or experiment["retry_policy"] != plan["retry_policy"]:
        raise ValueError("Experiment environment or retry policy changed")
    rebuilt = freeze_experiment(plan, kind=experiment["kind"], purpose=experiment["purpose"],
                               budget=experiment["budget"], rights=experiment["rights"],
                               source_audit=experiment["dependencies"].get("source_audit"),
                               pilot_receipts=experiment["dependencies"].get("pilot_receipts", ()),
                               metric_receipts=experiment["dependencies"].get("metric_receipts", ()),
                               instrument_config_sha256=experiment.get("instrument_config_sha256"),
                               smoke12_evidence_root=experiment["dependencies"].get("smoke12",{}).get("evidence_root"),
                               pilot_qualification_roles=experiment.get("pilot_qualification_roles"))
    if rebuilt != experiment: raise ValueError("Frozen experiment contract changed")


@contextmanager
def _execution_lock(root):
    import fcntl
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):raise ValueError("Invalid experiment root path")
    root.mkdir(parents=True,exist_ok=True)
    path=root/"experiment.lock"
    if path.is_symlink():raise ValueError("Experiment lock symlink")
    with path.open("a+") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError("Another experiment coordinator is active") from exc
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


class _ExecutionBudget:
    """Persistent cooperative budget; interrupted active sessions fail closed.

    An active session left by hard exit is charged through resume wall time,
    including downtime. This conservative scope prevents a restart resetting
    the budget; it does not pretend to preempt an in-process model call.
    """
    def __init__(self,root,experiment,*,resume):
        self.root=root;self.path=root/"budget.json";self.limits=experiment["budget"]
        if self.path.exists():
            if not resume:raise FileExistsError("Existing execution budget requires explicit resume")
            state=_read(self.path)
            if state.get("schema")!="rswm-experiment-budget-v1" or state.get("experiment_id")!=experiment["experiment_id"]:
                raise ValueError("Execution budget identity mismatch")
            for key in ("elapsed_total_seconds","observed_session_seconds"):
                v=state.get(key)
                if type(v) not in (int,float) or not math.isfinite(v) or v<0:raise ValueError("Execution budget elapsed value invalid")
            if type(state.get("next_invocation")) is not int or state["next_invocation"]<1:raise ValueError("Execution budget invocation invalid")
            if type(state.get("exhausted")) is not bool or state.get("limits")!=self.limits:
                raise ValueError("Execution budget limits/status invalid")
            base=state["elapsed_total_seconds"]
            active=state.get("active_started_at_epoch")
            if active is not None:
                if type(active) not in (int,float) or not math.isfinite(active) or active<0:raise ValueError("Execution budget active session invalid")
                base+=max(state["observed_session_seconds"],max(0.,time.time()-active))
            self.invocation=state["next_invocation"]
            self.exhausted=state.get("exhausted") is True
        else:
            if resume:raise ValueError("Cannot resume an experiment without its persistent budget ledger")
            base=0.;self.invocation=0;self.exhausted=False
        self.base=base;self.started=time.monotonic()
        self.state={"schema":"rswm-experiment-budget-v1","experiment_id":experiment["experiment_id"],
                    "elapsed_total_seconds":base,"observed_session_seconds":0.,
                    "active_started_at_epoch":time.time(),"next_invocation":self.invocation+1,
                    "exhausted":self.exhausted,"limits":self.limits,
                    "scope":"cooperative all execution/replay/analysis; hard-exit session includes downtime until resume"}
        _save(self.path,self.state)

    def observe(self,*,reserve_publication_bytes=0):
        session=max(0.,time.monotonic()-self.started)
        elapsed=self.base+session
        size=0
        for p in self.root.rglob("*"):
            if p.is_symlink():raise ValueError("Experiment artifact symlink")
            if p.is_file():size+=p.stat().st_size
        self.exhausted=self.exhausted or elapsed>=self.limits["max_wall_seconds"] or size+reserve_publication_bytes>=self.limits["max_output_bytes"]
        self.state.update(observed_session_seconds=session,exhausted=self.exhausted)
        _save(self.path,self.state)
        return {"elapsed_total_seconds":elapsed,"output_bytes":size,"reserved_publication_bytes":reserve_publication_bytes,
                "exhausted":self.exhausted,"scope":self.state["scope"]}

    def finish(self,*,reserve_publication_bytes=0):
        usage=self.observe(reserve_publication_bytes=reserve_publication_bytes)
        self.state.update(elapsed_total_seconds=usage["elapsed_total_seconds"],active_started_at_epoch=None,
                          observed_session_seconds=0.,last_usage=usage)
        _save(self.path,self.state)
        return usage


def execute_experiment(experiment, plan, out, *, resume=False, retry_failed=False):
    """Actually run and replay the suite. Budget checks are coordinated per run.

    A running uncooperative generator needs D11's worker deadline; cooperative
    in-process budgets are checked before each next run, not falsely preemptive.
    """
    validate_experiment(experiment, plan)
    if retry_failed and not resume:raise ValueError("Explicit experiment retry requires resume")
    root = Path(out)
    from .registry import create_adapter
    from .baselines import OracleMap
    with _execution_lock(root):
        existing=[p for p in root.iterdir() if p.name!="experiment.lock"]
        if existing and not resume:raise FileExistsError("Use explicit resume; prior experiment evidence is immutable")
        if resume and (not (root/"experiment.json").exists() or not (root/"budget.json").exists()):
            raise ValueError("Cannot resume an uninitialized experiment")
        _write(root/"experiment.json",experiment)
        budget=_ExecutionBudget(root,experiment,resume=resume)
        number=budget.invocation
        analysis_path=root/("analysis" if number==0 else f"analyses/invocation-{number:06d}")
        receipt_path=root/("receipt.json" if number==0 else f"receipts/invocation-{number:06d}.json")
        try:
            def factory(config,world,task):
                if budget.observe()["exhausted"]:raise RuntimeError("EXPERIMENT_BUDGET_EXHAUSTED")
                return OracleMap(world,task["origin"]) if config["adapter"]=="oracle" else create_adapter(config)
            execute_plan(plan,root/"suite",resume=resume,retry_failed=retry_failed,factory=factory)
            records=replay_suite(plan,root/"suite")
            integrity=integrity_summary(plan,root/"suite",records)
            export_tables(aggregate_suite(plan,root/"suite"),analysis_path)
            status=(("ACCEPTED_REAL_PILOT" if experiment["kind"]=="pilot" else "EXECUTED_FORMAL_PENDING_INSTRUMENT_ANALYSIS")
                    if integrity["accepted"] and experiment["purpose"]=="real" else "DIAGNOSTIC_ONLY" if integrity["accepted"] else "INCOMPLETE")
            qualification_error = None
            qualification_roles = experiment.get("pilot_qualification_roles")
            if qualification_roles is not None:
                try:
                    status = _pilot_execution_status(plan, records, integrity, qualification_roles,
                                                    formal=experiment["kind"]=="formal")
                except ValueError as exc:
                    status, qualification_error = "INCOMPLETE", str(exc)
            receipt={"schema":"rswm-experiment-receipt-v1","experiment_id":experiment["experiment_id"],
                     "purpose":experiment["purpose"],"kind":experiment["kind"],"acceptance_status":status,
                     "plan_sha256":plan["plan_sha256"],"source_manifest_sha256":_source_id(plan),
                     "source_code_sha256":code_fingerprint(),"matrix":experiment["matrix"],
                     "model_manifest_sha256":sorted({r["model_manifest_sha256"] for r in plan["expected"]}),
                     "integrity":integrity,"expected_runs":len(plan["expected"]),
                     "verified_outputs":sum(_verified_outputs(r) for r in records),
                     "invocation":number,"evidence_root":str(root.resolve()),
                     "receipt_path":receipt_path.relative_to(root).as_posix(),
                     "deadline_mode":"cooperative_before_next_run_and_terminal_check; persistent across resume",
                     "analysis_provenance_path":(analysis_path/"provenance.json").relative_to(root).as_posix(),
                     "analysis_provenance_sha256":file_hash(analysis_path/"provenance.json"),
                     "publication_acceptance":"NOT_READY"}
            if qualification_roles is not None:
                receipt["pilot_qualification_roles"] = qualification_roles
            if qualification_error is not None:
                receipt["qualification_error"] = qualification_error
            # Reserve the exact current receipt estimate plus bounded metadata
            # headroom; a terminal overshoot never earns accepted status.
            from .geometry import canonical
            usage=budget.finish(reserve_publication_bytes=len(canonical(receipt))+4096)
            receipt.update(budget_usage=usage,elapsed_wall_seconds=usage["elapsed_total_seconds"])
            if usage["exhausted"]:receipt["acceptance_status"]="BUDGET_EXHAUSTED"
            receipt["receipt_sha256"]=digest(receipt)
            _write(receipt_path,receipt)
            _save(root/"receipt-head.json",{"receipt_path":receipt["receipt_path"],"receipt_sha256":receipt["receipt_sha256"],"invocation":number})
            return receipt
        except BaseException:
            budget.finish()
            raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--plan", required=True); f.add_argument("--kind", choices=("pilot", "formal"), required=True)
    f.add_argument("--purpose", choices=("real", "diagnostic"), required=True); f.add_argument("--inputs", required=True); f.add_argument("--out", required=True)
    r = sub.add_parser("run"); r.add_argument("--plan", required=True); r.add_argument("--experiment", required=True); r.add_argument("--out", required=True)
    r.add_argument("--resume", action="store_true"); r.add_argument("--retry-failed", action="store_true")
    a = p.parse_args(argv)
    try:
        plan = _read(a.plan)
        if a.command == "freeze":
            value = freeze_experiment(plan, kind=a.kind, purpose=a.purpose, **_read(a.inputs)); _write(a.out, value)
        else: value = execute_experiment(_read(a.experiment), plan, a.out, resume=a.resume, retry_failed=a.retry_failed)
        print(json.dumps(value, sort_keys=True, allow_nan=False))
        return 1 if value.get("acceptance_status") in {"INCOMPLETE","BUDGET_EXHAUSTED","EXECUTED_FORMAL_PENDING_INSTRUMENT_ANALYSIS",COMPARISON_PILOT,FORMAL_RETAINED_FAILURES} else 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_INVALID", "error": str(exc)})); return 2


if __name__ == "__main__": raise SystemExit(main())
