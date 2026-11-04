"""Retry policy and metadata integrity regression tests on the current tree."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from rswm_inria import suite
from rswm_inria.baselines import CopyModel
from rswm_inria.geometry import digest
from rswm_inria.tasks import packages


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "中文 inputs"
    root.mkdir()
    world = np.zeros((2048, 2048, 3), np.uint8)
    Image.fromarray(world).save(root / "world.png")
    task = next(t for t, _ in packages("suite-v3", "diagnostic_only", 0) if t["family"] == "F2")
    path = root / "task.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    return [{"task_path": str(path), "world_path": str(root / "world.png"), "city": "sfo"}]


def make_plan(inputs, **kwargs):
    return suite.build_plan(inputs, [{"model_id": "copy", "adapter": "copy"}], [0], "a" * 64, "b" * 40, **kwargs)


def test_v3_default_is_first_attempt_and_frozen_single_attempt(inputs):
    plan = make_plan(inputs)
    assert plan["schema"] == "rswm-suite-v3"
    assert plan["retry_policy"]["primary_selection"] == "first_attempt_only"
    assert plan["retry_policy"]["max_attempts_per_run"] == 1
    assert plan["retry_policy"]["automatic_retry"] is False


def test_corrupt_resources_do_not_remove_authenticated_rgb(inputs, tmp_path):
    plan = make_plan(inputs)
    root = tmp_path / "suite"
    outcome = suite.execute_plan(plan, root)[0]
    Path(outcome["resource_path"]).write_text('{"cut":', encoding="utf-8")
    record = suite.replay_suite(plan, root)[0]
    assert record["replay_verified"]
    assert record["submission_status"] == "VERIFIED"
    assert record["resource_status"] == "METADATA_CORRUPT"
    assert record["resources"] is None


def test_recursive_source_fingerprint_changes_for_submodule(monkeypatch, tmp_path):
    root = tmp_path / "pkg"
    (root / "adapters").mkdir(parents=True)
    (root / "suite.py").write_text("pass\n")
    sub = root / "adapters" / "backend.py"
    sub.write_text("value = 1\n")
    monkeypatch.setattr(suite, "__file__", str(root / "suite.py"))
    before = suite.code_fingerprint()
    sub.write_text("value = 2\n")
    assert suite.code_fingerprint() != before


def retry_plan(inputs, budget=2):
    policy=dict(suite.DEFAULT_RETRY_POLICY,max_attempts_per_run=budget)
    return make_plan(inputs,retry_policy=policy)


def test_first_failure_recovery_success_separate_and_costs_retained(inputs,tmp_path):
    plan=retry_plan(inputs)
    root=tmp_path/"suite"
    def fail(*args):
        raise RuntimeError("load failed")
    first=suite.execute_plan(plan,root,factory=fail)[0]
    assert first["status"]=="FAILED_LOAD"
    recovered=suite.execute_plan(plan,root,resume=True,retry_failed=True)[0]
    assert recovered["status"]=="COMPLETED" and recovered["attempt"]==1
    primary=suite.replay_suite(plan,root)[0]
    secondary=suite.replay_suite(plan,root,"latest_attempt_within_budget")[0]
    assert primary["attempt"]==0 and not primary["replay_verified"]
    assert secondary["attempt"]==1 and secondary["replay_verified"]
    attempts=suite.safe_resource_attempts(plan,root)
    assert len(attempts)==2 and all(r["resources"] is not None for r in attempts)
    assert suite.execute_plan(plan,root,resume=True,retry_failed=True)[0]["attempt"]==1


def test_default_budget_refuses_retry_with_event_and_no_partial_attempt(inputs,tmp_path):
    plan=make_plan(inputs);root=tmp_path/"suite"
    suite.execute_plan(plan,root,factory=lambda *args:(_ for _ in ()).throw(RuntimeError("load")))
    with pytest.raises(ValueError,match="RETRY_BUDGET_EXHAUSTED"):
        suite.execute_plan(plan,root,resume=True,retry_failed=True)
    assert not (root/"runs"/plan["expected"][0]["run_id"]/"attempt-001").exists()
    event=json.loads((root/"retry-events.jsonl").read_text().splitlines()[-1])
    assert event["code"]=="RETRY_BUDGET_EXHAUSTED"


def test_failed_status_retry_requires_frozen_allowed_status(inputs,tmp_path):
    plan=retry_plan(inputs)
    class Broken(CopyModel):
        def reset(self,*a):
            raise RuntimeError("init")
    root=tmp_path/"suite"
    suite.execute_plan(plan,root,factory=lambda *a:Broken())
    with pytest.raises(ValueError,match="RETRY_STATUS_NOT_ALLOWED"):
        suite.execute_plan(plan,root,resume=True,retry_failed=True)


def test_missing_attempt_zero_never_falls_forward(inputs,tmp_path):
    plan=retry_plan(inputs)
    root=tmp_path/"suite";root.mkdir()
    suite._save(root/"plan.json",plan)
    suite.write_expected(plan["expected"],root/"expected.jsonl")
    (root/"runs"/plan["expected"][0]["run_id"]/"attempt-001").mkdir(parents=True)
    result=suite.replay_suite(plan,root)[0]
    assert result["status"]=="NOT_RUN" and result["attempt"] is None


def test_later_failed_attempt_does_not_replace_primary(inputs,tmp_path):
    plan=retry_plan(inputs);root=tmp_path/"suite"
    suite.execute_plan(plan,root)
    (root/"runs"/plan["expected"][0]["run_id"]/"attempt-001").mkdir()
    assert suite.replay_suite(plan,root)[0]["replay_verified"]
    assert suite.replay_suite(plan,root,"latest_attempt_within_budget")[0]["status"]=="CRASHED"


def test_numeric_attempt_order(tmp_path):
    for i in (1000,999,9,10):
        (tmp_path/f"attempt-{i:03d}").mkdir()
    assert [int(p.name.split("-")[-1]) for p in suite._attempt_folders(tmp_path)]==[9,10,999,1000]


@pytest.mark.parametrize("name",["outcome","started","resource","receipt","head"])
@pytest.mark.parametrize("bad",["truncated","not_object","identity","attempt","plan","missing","duplicate","nan","external_path"])
def test_metadata_fault_is_isolated_and_resource_fault_keeps_rgb(inputs,tmp_path,name,bad):
    plan=make_plan(inputs);root=tmp_path/"suite"
    outcome=suite.execute_plan(plan,root)[0]
    folder=Path(outcome["run_dir"]).parent
    path=(folder/("resources.json" if name=="resource" else name+".json"))
    if name in {"receipt","head"}:
        path=root/("receipts" if name=="receipt" else "heads")/(outcome["run_id"]+"-000.json")
    value=json.loads(path.read_text())
    if bad=="truncated":
        content='{"cut":'
    elif bad=="not_object":
        content='[]'
    elif bad=="duplicate":
        content='{"run_id":"a","run_id":"b"}'
    elif bad=="nan":
        content='{"value":NaN}'
    elif bad=="external_path":
        # A symlink is rejected before parsing, including structurally correct JSON.
        external=tmp_path/"external.json";external.write_text(json.dumps(value))
        path.unlink();path.symlink_to(external)
        content=None
    elif bad=="missing":
        value.pop("run_id");content=json.dumps(value)
    elif bad=="attempt":
        value["attempt"]=42;content=json.dumps(value)
    elif bad=="plan":
        value["plan_sha256"]="0"*64;content=json.dumps(value)
    else:
        value["run_id"]="wrong";content=json.dumps(value)
    if content is not None:
        path.write_text(content)
    result=suite.replay_suite(plan,root)[0]
    assert result["integrity_errors"]
    assert sum(s["planned_outputs"] for s in (result["reports"].values() if "metrics" not in result["reports"] else [result["reports"]]))==32
    if name=="resource":
        assert result["replay_verified"] and result["resource_status"]=="METADATA_CORRUPT"
    else:
        assert not result["replay_verified"]


def test_corrupt_old_resource_does_not_break_recovery_rgb(inputs,tmp_path):
    plan=retry_plan(inputs);root=tmp_path/"suite"
    first=suite.execute_plan(plan,root,factory=lambda *a:(_ for _ in ()).throw(RuntimeError("load")))[0]
    suite.execute_plan(plan,root,resume=True,retry_failed=True)
    Path(first["resource_path"]).write_text("[]")
    attempts=suite.safe_resource_attempts(plan,root)
    assert len(attempts)==2 and attempts[0]["resource_status"]=="METADATA_CORRUPT"
    assert suite.replay_suite(plan,root,"latest_attempt_within_budget")[0]["replay_verified"]


def test_root_expected_corruption_stops_instead_of_fabricating_denominator(inputs,tmp_path):
    plan=make_plan(inputs);root=tmp_path/"suite"
    suite.execute_plan(plan,root)
    (root/"expected.jsonl").write_text("[]\n")
    with pytest.raises((ValueError,TypeError)):
        suite.replay_suite(plan,root)


def test_active_lock_blocks_export_and_initialization(inputs,tmp_path):
    import fcntl
    plan=make_plan(inputs);root=tmp_path/"suite";root.mkdir()
    with (root/"coordinator.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match="active"):
            suite.execute_plan(plan,root)
        assert not (root/"plan.json").exists()
        with pytest.raises(RuntimeError,match="active"):
            with suite.suite_read_lock(root):
                pass


def test_cli_verify_fault_exit_and_root_error(inputs,tmp_path):
    plan=make_plan(inputs);root=tmp_path/"suite";path=tmp_path/"plan.json"
    suite.save_plan(plan,path)
    assert suite.main(["run","--plan",str(path),"--out",str(root)])==0
    assert suite.main(["verify","--plan",str(path),"--suite",str(root),"--out",str(tmp_path/"verify.json")])==0
    resource=next((root/"runs").glob("*/attempt-000/resources.json"));resource.write_text("[]")
    assert suite.main(["verify","--plan",str(path),"--suite",str(root)])==1
    (root/"expected.jsonl").write_text('{}\n')
    assert suite.main(["inspect","--plan",str(path),"--suite",str(root)])==2


def test_current_source_fingerprint_mismatch_is_not_bypassed(inputs,monkeypatch):
    plan=make_plan(inputs)
    monkeypatch.setattr(suite,"code_fingerprint",lambda:"0"*64)
    with pytest.raises(ValueError,match="Executing code"):
        suite.validate_plan(plan)


def test_policy_change_cannot_resume_old_plan(inputs,tmp_path):
    plan=make_plan(inputs);root=tmp_path/"suite"
    suite.execute_plan(plan,root)
    changed=copy.deepcopy(plan);changed["retry_policy"]["max_attempts_per_run"]=2
    changed.pop("plan_sha256");changed["plan_sha256"]=digest(changed)
    with pytest.raises(ValueError,match="mismatch"):
        suite.execute_plan(changed,root,resume=True)


def test_two_run_fault_does_not_abort_other_run(inputs,tmp_path):
    plan=suite.build_plan(inputs,[{"model_id":"copy","adapter":"copy"},{"model_id":"random","adapter":"random"}],[0],"a"*64,"b"*40)
    root=tmp_path/"suite";outcomes=suite.execute_plan(plan,root)
    bad=Path(outcomes[0]["run_dir"]).parent/"outcome.json";bad.write_text('{"cut":')
    records=suite.replay_suite(plan,root)
    assert len(records)==2 and not records[0]["replay_verified"] and records[1]["replay_verified"]


def test_finish_failure_keeps_original_model_failure(inputs,tmp_path,monkeypatch):
    plan=make_plan(inputs)
    def fail_finish(*a):
        raise OSError("collector output")
    monkeypatch.setattr(suite.ResourceCollector,"finish",fail_finish)
    def fail_model(*a):
        raise RuntimeError("original model failure")
    outcome=suite.execute_plan(plan,tmp_path/"suite",factory=fail_model)[0]
    assert outcome["status"]=="FAILED_LOAD" and outcome["error_type"]=="RuntimeError"
    assert outcome["resource_error_type"]=="OSError"


def test_cli_resource_failure_is_not_accepted(inputs,tmp_path,monkeypatch):
    plan=make_plan(inputs);path=tmp_path/"plan.json";root=tmp_path/"suite"
    suite.save_plan(plan,path)
    def fail_finish(*a):
        raise OSError("collector output")
    monkeypatch.setattr(suite.ResourceCollector,"finish",fail_finish)
    assert suite.main(["run","--plan",str(path),"--out",str(root)])==1
    summary=json.loads((root/"integrity-summary.json").read_text())
    assert summary["accepted"] is False and summary["verified_submissions"]==1


def test_actual_cli_subprocess_run_verify_corruption(inputs,tmp_path):
    import subprocess
    import sys
    plan=make_plan(inputs);path=tmp_path/"计划 空格.json";root=tmp_path/"suite 中文"
    suite.save_plan(plan,path)
    command=[sys.executable,"-m","rswm_inria.suite"]
    run=subprocess.run(command+["run","--plan",str(path),"--out",str(root)],capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stderr
    verify=subprocess.run(command+["verify","--plan",str(path),"--suite",str(root)],capture_output=True,text=True,timeout=60)
    assert verify.returncode==0,verify.stderr
    next((root/"runs").glob("*/attempt-000/outcome.json")).write_text("[]")
    verify=subprocess.run(command+["verify","--plan",str(path),"--suite",str(root)],capture_output=True,text=True,timeout=60)
    assert verify.returncode==1
    assert json.loads(verify.stdout)["accepted"] is False


def test_sigterm_is_interrupted_and_recovery_is_fresh_attempt(inputs,tmp_path):
    import subprocess
    import sys
    plan=retry_plan(inputs);path=tmp_path/"plan.json";root=tmp_path/"suite"
    suite.save_plan(plan,path)
    code='''import os,signal,sys
from rswm_inria import suite
from rswm_inria.baselines import CopyModel
class Terminated(CopyModel):
    def generate(self,q):
        os.kill(os.getpid(),signal.SIGTERM)
suite.model_factory=lambda *a:Terminated()
raise SystemExit(suite.main(["run","--plan",sys.argv[1],"--out",sys.argv[2]]))
'''
    child=subprocess.run([sys.executable,"-c",code,str(path),str(root)],capture_output=True,text=True,timeout=60)
    assert child.returncode==1
    first=json.loads(next((root/"runs").glob("*/attempt-000/outcome.json")).read_text())
    assert first["status"]=="INTERRUPTED"
    result=suite.execute_plan(plan,root,resume=True,retry_failed=True)[0]
    assert result["status"]=="COMPLETED" and result["attempt"]==1
    assert not suite.replay_suite(plan,root)[0]["replay_verified"]


def rehash(plan):
    plan.pop("plan_sha256",None)
    plan["plan_sha256"]=digest(plan)
    return plan


def test_validate_reuses_geometry_once_per_task_and_only_inside_one_call(inputs,monkeypatch):
    plan=suite.build_plan(inputs,[{"model_id":"a","adapter":"copy"},{"model_id":"b","adapter":"random"}],[0,1,2],"a"*64,"b"*40)
    original_validate=suite.validate_task;original_reports=suite.planned_reports
    calls={"validate":0,"reports":0}
    def validate(task):
        calls["validate"]+=1
        return original_validate(task)
    def reports(task):
        calls["reports"]+=1
        return original_reports(task)
    monkeypatch.setattr(suite,"validate_task",validate)
    monkeypatch.setattr(suite,"planned_reports",reports)
    suite.validate_plan(plan)
    assert calls=={"validate":1,"reports":1}
    suite.validate_plan(plan)
    assert calls=={"validate":2,"reports":2}


@pytest.mark.parametrize("kind",["mixed_world","mixed_source","mixed_commit","task_study","unknown_config"])
def test_rehash_does_not_bypass_plan_semantic_binding(inputs,kind):
    plan=suite.build_plan(inputs,[{"model_id":"copy","adapter":"copy"}],[0,1],"a"*64,"b"*40)
    if kind=="mixed_world":
        plan["expected"][1]["city"]="innsbruck"
    elif kind=="mixed_source":
        plan["expected"][1]["source_manifest_sha256"]="c"*64
    elif kind=="mixed_commit":
        plan["expected"][1]["code_commit"]="c"*40
    elif kind=="task_study":
        next(iter(plan["tasks"].values()))["study"]="paired"
    else:
        plan["models"]["copy"]["config"]["ignored_field"]="not consumed"
    rehash(plan)
    with pytest.raises(ValueError):
        suite.validate_plan(plan)


@pytest.mark.parametrize("fault",["schema","missing_measurement","missing_scalar","scalar_mismatch","partial_mismatch","incomplete_total","na_nonnull","counter_mismatch"])
def test_v3_resource_contract_fault_verify_and_aggregate_agree(inputs,tmp_path,fault):
    from rswm_inria.aggregate import aggregate_suite
    plan=make_plan(inputs);root=tmp_path/"suite";outcome=suite.execute_plan(plan,root)[0]
    folder=Path(outcome["run_dir"]).parent;path=folder/"resources.json"
    resource=json.loads(path.read_text())
    if fault=="schema":
        resource["schema"]="rswm-resource-v2"
    elif fault=="missing_measurement":
        resource["measurements"].pop("generate_seconds")
    elif fault=="missing_scalar":
        resource.pop("generate_seconds")
    elif fault=="scalar_mismatch":
        resource["generate_seconds"]+=1
    elif fault=="partial_mismatch":
        resource["measurements"]["generate_seconds"]["measured_sum"]+=1
    elif fault=="incomplete_total":
        resource["measurements"]["generate_seconds"]["complete"]=False
        resource["measurement_complete"]=False
    elif fault=="counter_mismatch":
        resource["adapter_counters"]["requested_outputs"]+=1
    else:
        field="gpu_synchronized_model_wall_seconds"
        resource[field]=1
        resource["measurements"][field]["value"]=1
        resource["measurements"][field]["measured_sum"]=1
    path.write_text(json.dumps(resource))
    manifest=json.loads((folder/"artifact-manifest.json").read_text())
    manifest["files"]["resources.json"]=suite.file_hash(path)
    (folder/"artifact-manifest.json").write_text(json.dumps(manifest))
    record=suite.replay_suite(plan,root)[0]
    assert record["replay_verified"] and record["resource_status"]=="METADATA_CORRUPT"
    assert record["resources"] is None
    report=aggregate_suite(plan,root,bootstrap=2)
    assert report["replay_status"][0]["resource_status"]==record["resource_status"]
    assert report["resource_totals"][0]["metrics"]["generate_seconds"]["complete"] is False
    assert suite.main(["verify","--plan",str(root/"plan.json"),"--suite",str(root)])==1


def test_evaluator_output_accounting_uses_frozen_slots_and_observed_journal(inputs,tmp_path):
    plan=make_plan(inputs);root=tmp_path/"suite";outcome=suite.execute_plan(plan,root)[0]
    resource=json.loads(Path(outcome["resource_path"]).read_text())
    assert resource["accounting_calls"]["planned_output_slots"]==32
    assert resource["accounting_calls"]["valid_outputs"]==32
    def fail_model(*args):
        raise RuntimeError("loading failed")
    failed=suite.execute_plan(plan,tmp_path/"failed",factory=fail_model)[0]
    resource=json.loads(Path(failed["resource_path"]).read_text())
    assert resource["accounting_calls"]["planned_output_slots"]==32
    assert resource["accounting_calls"]["valid_outputs"]==0
    assert resource["accounting_calls"]["requests_issued"]==0


def test_interruption_counts_already_submitted_slots_from_existing_journal(inputs,tmp_path):
    # F3 has two generate calls: the first 32 outputs survive interruption of the second.
    source=Path(inputs[0]["task_path"])
    task=next(t for t,_ in packages("suite-v3","diagnostic_only",0) if t["family"]=="F3")
    source.write_text(json.dumps(task))
    plan=retry_plan(inputs);root=tmp_path/"suite"
    class Interrupted(CopyModel):
        def __init__(self):
            self.entered=0
        def generate(self,q):
            self.entered+=1
            if self.entered==2:
                raise KeyboardInterrupt()
            return super().generate(q)
    with pytest.raises(KeyboardInterrupt):
        suite.execute_plan(plan,root,factory=lambda *a:Interrupted())
    resource=json.loads(next((root/"runs").glob("*/attempt-000/resources.json")).read_text())
    assert resource["accounting_calls"]["planned_output_slots"]==64
    assert resource["accounting_calls"]["valid_outputs"]==32


@pytest.mark.parametrize("name",["runs","heads","receipts"])
def test_symlink_ancestor_is_rejected_without_dropping_plan(inputs,tmp_path,name):
    plan=make_plan(inputs);root=tmp_path/"suite";suite.execute_plan(plan,root)
    external=tmp_path/("external-"+name)
    (root/name).rename(external);(root/name).symlink_to(external,target_is_directory=True)
    record=suite.replay_suite(plan,root)[0]
    assert not record["replay_verified"] and record["integrity_errors"]
    assert record["reports"]==plan["expected"][0]["planned_reports"]


def test_symlink_run_ancestor_blocks_new_attempt_writes(inputs,tmp_path):
    plan=make_plan(inputs);root=tmp_path/"suite";root.mkdir()
    external=tmp_path/"external";external.mkdir()
    (root/"runs").symlink_to(external,target_is_directory=True)
    suite._save(root/"plan.json",plan);suite.write_expected(plan["expected"],root/"expected.jsonl")
    result=suite.execute_plan(plan,root,resume=True)[0]
    assert result["status"]=="METADATA_CORRUPT"
    assert list(external.iterdir())==[]


@pytest.mark.parametrize("fault",["v2_claims_complete","v3_missing_measurements","v3_wrong_identity","v2_wrong_optional_identity"])
def test_legacy_resource_cannot_claim_complete_without_current_contract(inputs,tmp_path,fault):
    plan=make_plan(inputs,schema=suite.LEGACY_SCHEMA);root=tmp_path/"suite"
    outcome=suite.execute_plan(plan,root)[0]
    path=Path(outcome["resource_path"]);resource=json.loads(path.read_text())
    if fault=="v2_claims_complete":
        resource["schema"]="rswm-resource-v2"
    elif fault=="v3_missing_measurements":
        resource.pop("measurements")
    elif fault=="v3_wrong_identity":
        resource["run_id"]="wrong"
    else:
        resource["schema"]="rswm-resource-v2"
        resource["attempt"]=100
    resource["measurement_complete"]=True
    path.write_text(json.dumps(resource))
    manifest_path=path.parent/"artifact-manifest.json"
    manifest=json.loads(manifest_path.read_text())
    manifest["files"]["resources.json"]=suite.file_hash(path)
    manifest_path.write_text(json.dumps(manifest))
    record=suite.replay_suite(plan,root)[0]
    assert record["replay_verified"]
    assert record["resource_status"] in {"INCOMPLETE","METADATA_CORRUPT"}
    assert suite.main(["verify","--plan",str(root/"plan.json"),"--suite",str(root)])==1


@pytest.mark.parametrize("kind",["head","receipt","manifest"])
def test_legacy_optional_new_metadata_is_validated_when_present(inputs,tmp_path,kind):
    plan=make_plan(inputs,schema=suite.LEGACY_SCHEMA);root=tmp_path/"suite"
    outcome=suite.execute_plan(plan,root)[0]
    folder=Path(outcome["run_dir"]).parent
    if kind=="manifest":
        path=folder/"artifact-manifest.json"
    else:
        path=root/("heads" if kind=="head" else "receipts")/(outcome["run_id"]+"-000.json")
    value=json.loads(path.read_text());value["plan_sha256"]="0"*64
    path.write_text(json.dumps(value))
    result=suite.replay_suite(plan,root)[0]
    assert not result["replay_verified"] and result["integrity_errors"]


def test_legacy_original_absent_receipt_manifest_and_head_plan_field_still_replays(inputs,tmp_path):
    plan=make_plan(inputs,schema=suite.LEGACY_SCHEMA);root=tmp_path/"suite"
    outcome=suite.execute_plan(plan,root)[0]
    folder=Path(outcome["run_dir"]).parent
    (folder/"artifact-manifest.json").unlink()
    (root/"receipts"/(outcome["run_id"]+"-000.json")).unlink()
    head_path=root/"heads"/(outcome["run_id"]+"-000.json")
    head=json.loads(head_path.read_text());head.pop("plan_sha256")
    head_path.write_text(json.dumps(head))
    result=suite.replay_suite(plan,root)[0]
    assert result["replay_verified"] and result["legacy_latest_attempt_policy"]


@pytest.mark.parametrize("legacy",[False,True])
def test_healthy_recovery_cannot_certify_unknown_older_attempt_cost(inputs,tmp_path,legacy):
    plan=(make_plan(inputs,schema=suite.LEGACY_SCHEMA) if legacy else retry_plan(inputs))
    root=tmp_path/"suite"
    def fail(*args):
        raise RuntimeError("load failed")
    first=suite.execute_plan(plan,root,factory=fail)[0]
    suite.execute_plan(plan,root,resume=True,retry_failed=True)
    path=Path(first["resource_path"]);resource=json.loads(path.read_text())
    resource["schema"]="rswm-resource-v2";resource["measurement_complete"]=True
    path.write_text(json.dumps(resource))
    manifest_path=path.parent/"artifact-manifest.json"
    manifest=json.loads(manifest_path.read_text());manifest["files"]["resources.json"]=suite.file_hash(path)
    manifest_path.write_text(json.dumps(manifest))
    records=suite.replay_suite(plan,root,"latest_attempt_within_budget")
    assert records[0]["replay_verified"] and records[0]["resource_status"]=="COMPLETE"
    summary=suite.integrity_summary(plan,root,records)
    assert summary["accepted"] is False and summary["resource_incomplete_attempts"]==1
    assert suite.main(["verify","--plan",str(root/"plan.json"),"--suite",str(root),"--selection","latest_attempt_within_budget"])==1


SCOPE_FAULTS=[("reason_codes",[{}]),("reason_codes",[1]),("reason_codes",["KNOWN",1]),
              ("reason_codes",{"bad":"container"}),("unit",{}),("scope",[])]


@pytest.mark.parametrize("key,bad",SCOPE_FAULTS,ids=["dict_code","int_code","mixed_codes","codes_container","unit_container","scope_container"])
@pytest.mark.parametrize("counter",[False,True])
def test_malformed_measurement_scope_is_isolated_in_export(inputs,tmp_path,key,bad,counter):
    from rswm_inria.aggregate import aggregate_suite
    from rswm_inria.export_tables import export_tables
    plan=make_plan(inputs);root=tmp_path/"suite";outcome=suite.execute_plan(plan,root)[0]
    path=Path(outcome["resource_path"]);resource=json.loads(path.read_text())
    table="counter_measurements" if counter else "measurements"
    field="requested_outputs" if counter else "generate_seconds"
    resource[table][field][key]=bad
    path.write_text(json.dumps(resource))
    manifest_path=path.parent/"artifact-manifest.json"
    manifest=json.loads(manifest_path.read_text());manifest["files"]["resources.json"]=suite.file_hash(path)
    manifest_path.write_text(json.dumps(manifest))
    record=suite.replay_suite(plan,root)[0]
    assert record["replay_verified"] and record["resource_status"]=="METADATA_CORRUPT"
    assert record["resources"] is None
    report=aggregate_suite(plan,root,bootstrap=2)
    export_tables(report,tmp_path/"tables")
    assert report["acceptance_status"]=="INCOMPLETE" and report["replay_status"][0]["replay_verified"]


@pytest.mark.parametrize("key,bad",SCOPE_FAULTS,ids=["dict_code","int_code","mixed_codes","codes_container","unit_container","scope_container"])
def test_direct_resource_aggregation_rejects_invalid_scope_container(key,bad):
    from rswm_inria.aggregate import resource_totals
    measurement={"value":1,"measured_sum":1,"applicability":"applicable","eligible_units":1,
                 "measured_units":1,"complete":True,"aggregation":"sum","unit":"seconds",
                 "scope":"direct observer","reason_codes":[]}
    measurement[key]=bad
    result=resource_totals([{"run_id":"r","study":"core","model_id":"m","schema":"rswm-resource-v3",
                             "attempt":0,"generate_seconds":1,"measurements":{"generate_seconds":measurement}}])
    metric=result[0]["metrics"]["generate_seconds"]
    assert metric["complete"] is False and metric["value"] is None
    assert "INVALID_MEASUREMENT_SCOPE" in metric["reason_codes"]
