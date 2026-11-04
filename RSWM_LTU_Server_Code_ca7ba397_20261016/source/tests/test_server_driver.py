"""Actual CPU process/retention controls; no real generator or data PASS."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from rswm_inria import server
from rswm_inria.geometry import file_hash


SOURCE = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def project(tmp_path):
    location = server.init_project(tmp_path / "diagnostic-project", SOURCE)
    return location, json.loads(location.read_text())


def funded(location, p, stage="pilot-data", *, wall=30, size=2_000_000, attempts=2):
    p["code_commit"] = "d" * 40  # Diagnostic ref, never a source-authenticity claim.
    p["rights"] = {"data_use_approved": True, "weight_use_approved": True,
                   "decision_reference": "SOFTWARE_CONTROL_ONLY_NOT_OWNER_APPROVAL"}
    p["budgets"][stage] = {"max_wall_seconds": wall, "max_output_bytes": size, "max_attempts": attempts}
    write(location, p)
    return server._project(location)[1]


def diagnostic_steps(monkeypatch, *, body=None):
    """Mock the real producer with a distinctly named CPU process control."""
    body = body or "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('diagnostic subprocess, not real data')"
    def compile_steps(location, p, stage, products, *, resume):
        return [server._step("DIAGNOSTIC_CPU_PROCESS_ONLY", [sys.executable, "-c", body, str(products / "proof.txt")], outputs=[products / "proof.txt"])]
    monkeypatch.setattr(server, "_build_steps", compile_steps)
    monkeypatch.setattr(server, "_semantic", lambda *a: "SOFTWARE_CONTROL_ONLY_NOT_REAL_ACCEPTANCE")


def test_init_creates_unapproved_editable_templates_without_model(project):
    location, p = project
    assert p["rights"]["data_use_approved"] is False and p["rights"]["weight_use_approved"] is False
    assert all(value is None for value in p["budgets"].values())
    assert json.loads((location.parent / "owner/wan-use-decision.json").read_text())["approved"] is False
    spec = json.loads((location.parent / "owner/training-spec.json").read_text())
    assert spec["updates"] is None and spec["trainable_policy"]["mode"] is None
    assert not Path(p["run_root"]).exists()


def test_init_never_overwrites_operator_project(project):
    location, p = project; before = location.read_bytes()
    with pytest.raises(FileExistsError): server.init_project(location.parent, SOURCE)
    assert location.read_bytes() == before


def test_check_plan_and_rejected_run_launch_no_child_and_do_not_infer_approval(project, monkeypatch):
    location, p = project
    monkeypatch.setattr(server.subprocess, "Popen", lambda *a, **k: pytest.fail("An unapproved project launched a child"))
    plan = server.plan_project(location, "pilot")
    assert all(row["status"] == "INPUT_OR_DEPENDENCY_BLOCKED" for row in plan["stages"])
    assert plan["goal_completed"] is False
    run = server.run_project(location, "pilot")
    assert run["status"] == "HANDOFF_REQUIRED" and run["goal_completed"] is False
    assert json.loads(location.read_text())["rights"] == p["rights"]
    assert json.loads((Path(p["run_root"]) / "cursor.json").read_text())["next_command"][-1] == "--resume"
    assert not (Path(p["run_root"]) / "stages/pilot-data/attempts").exists()


def test_arbitrary_shell_job_fields_rejected_before_launch(project):
    location, p = project; p["jobs"] = [{"shell": "echo arbitrary"}]; write(location, p)
    with pytest.raises(ValueError, match="arbitrary"): server.plan_project(location, "pilot-data")


@pytest.mark.skipif(os.name != "posix", reason="Production driver requires POSIX flock/process groups")
def test_real_cpu_child_completes_and_hash_checked_resume_skips(project, monkeypatch):
    location, p = project; p = funded(location, p); diagnostic_steps(monkeypatch)
    first = server.run_project(location, "pilot-data")
    assert first["status"] == "REQUESTED_CODE_STAGES_EXECUTED" and first["goal_completed"] is False
    stage = Path(p["run_root"]) / "stages/pilot-data"
    start = json.loads((stage / "attempts/attempt-000/commands/command-000/start.json").read_text())
    exit_record = json.loads((stage / "attempts/attempt-000/commands/command-000/exit.json").read_text())
    assert start["pid"] > 0 and start["shell"] is False and exit_record["exit_code"] == 0 and exit_record["live"] is False
    original = (stage / "completed.json").read_bytes()
    second = server.run_project(location, "pilot-data", resume=True)
    assert second["stages"][0]["status"] == "SKIPPED_VERIFIED_COMPLETED"
    assert not (stage / "attempts/attempt-001").exists() and (stage / "completed.json").read_bytes() == original


@pytest.mark.skipif(os.name != "posix", reason="Production driver requires POSIX")
def test_completed_artifact_tamper_and_budget_change_rejected(project, monkeypatch):
    location, p = project; p = funded(location, p); diagnostic_steps(monkeypatch)
    assert server.run_project(location, "pilot-data")["status"] == "REQUESTED_CODE_STAGES_EXECUTED"
    stage = Path(p["run_root"]) / "stages/pilot-data"
    marker = json.loads((stage / "completed.json").read_text()); proof = Path(marker["products"]) / "proof.txt"
    original = proof.read_bytes(); proof.write_bytes(b"tampered retained output")
    report = server.run_project(location, "pilot-data", resume=True)
    assert report["status"] == "HANDOFF_REQUIRED" and "artifacts changed" in report["reason"]
    proof.write_bytes(original)  # Restore only this explicitly diagnostic fixture.
    p["budgets"]["pilot-data"]["max_wall_seconds"] += 1; write(location, p)
    report = server.run_project(location, "pilot-data", resume=True)
    assert report["status"] == "HANDOFF_REQUIRED" and "configuration changed" in report["reason"]
    assert not (stage / "attempts/attempt-001").exists()


@pytest.mark.skipif(os.name != "posix", reason="Production driver requires POSIX")
def test_child_failure_retained_and_resume_uses_new_attempt_with_all_costs(project, monkeypatch, tmp_path):
    location, p = project; p = funded(location, p)
    trigger = tmp_path / "diagnostic-first-attempt-seen"
    body = "from pathlib import Path; import sys; p=Path(" + repr(str(trigger)) + "); fail=not p.exists(); p.write_text('retained'); Path(sys.argv[1]).write_text('failed partial' if fail else 'success control'); raise SystemExit(7 if fail else 0)"
    diagnostic_steps(monkeypatch, body=body)
    first = server.run_project(location, "pilot-data")
    assert first["status"] == "HANDOFF_REQUIRED"
    stage = Path(p["run_root"]) / "stages/pilot-data"
    failed = stage / "attempts/attempt-000/products/proof.txt"
    failure_bytes = failed.read_bytes()
    assert json.loads((stage / "attempts/attempt-000/commands/command-000/exit.json").read_text())["exit_code"] == 7
    assert server.run_project(location, "pilot-data")["status"] == "HANDOFF_REQUIRED"
    second = server.run_project(location, "pilot-data", resume=True)
    assert second["status"] == "REQUESTED_CODE_STAGES_EXECUTED" and failed.read_bytes() == failure_bytes
    marker = json.loads((stage / "completed.json").read_text())
    assert marker["attempt"] == "attempt-001"
    rows, elapsed = server._prior(stage)
    assert len(rows) == 2 and rows[0]["status"] == "FAILED" and rows[1]["status"] == "EXECUTED" and elapsed > 0


@pytest.mark.skipif(os.name != "posix", reason="Production driver requires POSIX")
def test_wall_budget_kills_own_process_group_and_preserves_real_pid_exit(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    body = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)"
    folder = tmp_path / "command"; budget = {"max_wall_seconds": .6, "max_output_bytes": 1_000_000, "max_attempts": 1}
    with pytest.raises(server.StageFailure, match="WALL_BUDGET"):
        server._run_process([sys.executable, "-c", body, str(pid_file)], folder, cwd=tmp_path, budget=budget,
                            roots=[tmp_path], prior_seconds=0, attempt_started=time.monotonic())
    start = json.loads((folder / "start.json").read_text()); outcome = json.loads((folder / "exit.json").read_text())
    assert start["pid"] > 0 and outcome["reason"] == "WALL_BUDGET_EXHAUSTED" and outcome["exit_code"] != 0 and outcome["live"] is False
    assert server._pid_token(start["pid"]) is None
    assert pid_file.exists() and server._pid_token(int(pid_file.read_text())) is None


@pytest.mark.skipif(os.name != "posix", reason="Production driver requires POSIX")
def test_output_budget_retains_overshoot_and_no_false_completion(tmp_path):
    folder = tmp_path / "command"; payload = tmp_path / "retained-large.bin"
    body = "from pathlib import Path; import sys,time; Path(sys.argv[1]).write_bytes(b'x'*200000); time.sleep(30)"
    budget = {"max_wall_seconds": 10, "max_output_bytes": 50_000, "max_attempts": 1}
    with pytest.raises(server.StageFailure, match="OUTPUT_BUDGET"):
        server._run_process([sys.executable, "-c", body, str(payload)], folder, cwd=tmp_path, budget=budget,
                            roots=[tmp_path], prior_seconds=0, attempt_started=time.monotonic())
    result = json.loads((folder / "exit.json").read_text())
    assert payload.stat().st_size == 200000 and result["observed_output_bytes"] >= 200000
    assert result["reason"] == "OUTPUT_BUDGET_EXHAUSTED" and result["live"] is False


@pytest.mark.skipif(os.name != "posix", reason="Production driver requires POSIX")
def test_exited_leader_does_not_leave_own_descendant_running(tmp_path):
    pid_file = tmp_path / "child.pid"
    body = "import subprocess,sys; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); Path(sys.argv[1]).write_text(str(p.pid))"
    folder = tmp_path / "command"
    code = server._run_process([sys.executable, "-c", body, str(pid_file)], folder, cwd=tmp_path,
        budget={"max_wall_seconds": 10, "max_output_bytes": 1_000_000, "max_attempts": 1}, roots=[tmp_path],
        prior_seconds=0, attempt_started=time.monotonic())
    result = json.loads((folder / "exit.json").read_text()); child = int(pid_file.read_text())
    assert code == 0 and result["live"] is False and not result["remaining_owned_members"]
    assert any(row["pid"] == child for row in result["owned_members_observed"])
    assert server._pid_token(child) is None


@pytest.mark.skipif(os.name != "posix", reason="Production driver requires POSIX")
def test_recycled_pid_birth_token_never_signals_unrelated_group(monkeypatch):
    from types import SimpleNamespace
    proxy = SimpleNamespace(pid=os.getpid(), _rswm_birth_token="0", returncode=0, poll=lambda: 0)
    monkeypatch.setattr(server.os, "killpg", lambda *a: pytest.fail("A recycled PID group was signalled"))
    assert server._stop_group(proxy) == 0


@pytest.mark.skipif(os.name != "posix", reason="Production driver requires POSIX")
def test_flock_rejects_second_coordinator_without_deleting_lock(tmp_path):
    root = tmp_path / "run"
    with server._coordinator(root):
        with pytest.raises(ValueError, match="coordinator"): 
            with server._coordinator(root): pass
    assert (root / "server-coordinator.lock").exists()


def test_current_comparison_pilot_receipt_exit_one_is_qualified_but_incomplete_not_accepted(tmp_path):
    root = tmp_path / "products/experiment"; root.mkdir(parents=True)
    write(root / "receipt-head.json", {"receipt_path": "receipt.json"})
    receipt = {"acceptance_status": "EXECUTED_REAL_COMPARISON_PILOT_WITH_FAILURES", "budget_usage": {"exhausted": False}}
    write(root / "receipt.json", receipt)
    # Qualification enum logic is tested separately from source/plan/replay
    # binding. This thin fixture must never pass the real stage verifier.
    assert server._qualified_status("pilot", receipt) == "EXECUTED_REAL_COMPARISON_PILOT_WITH_FAILURES"
    with pytest.raises((server.StageFailure, FileNotFoundError)):
        server._semantic("pilot", root.parent, {})
    for status in ("INCOMPLETE", "BUDGET_EXHAUSTED"):
        receipt["acceptance_status"] = status; write(root / "receipt.json", receipt)
        with pytest.raises(server.StageFailure): server._qualified_status("pilot", receipt)


def test_data_identity_does_not_change_when_later_weight_approval_is_filled(project):
    location, p = project; p = funded(location, p)
    p["rights"]["weight_use_approved"] = False
    before = server._stage_identity(location, p, "pilot-data")
    p["rights"]["weight_use_approved"] = True
    assert server._stage_identity(location, p, "pilot-data") == before


def test_later_core_recovery_does_not_invalidate_completed_pilot_data_identity(project):
    location, p = project; p = funded(location, p)
    before = server._stage_identity(location, p, "pilot-data")
    p["data"].update(core_source_dir="/diagnostic/new/core/source", core_download=True,
                     core_max_source_bytes=9_000_000_000, existing_core_build="/diagnostic/retained/core-build")
    assert server._stage_identity(location, p, "pilot-data") == before


def test_first_use_instrument_ledgers_are_left_for_original_producer_to_create(project, monkeypatch, tmp_path):
    location, p = project
    build = tmp_path / "core-path-control"; (build / "private/worlds").mkdir(parents=True)
    write(build / "private/index.json", {"scope": "PATH_CONTROL_ONLY_NOT_DATA_AUDIT"})
    monkeypatch.setattr(server, "_data_paths", lambda *a: (build, tmp_path / "audit-not-used"))
    m = p["measurement"]; calibration = tmp_path / "calibration.json"; write(calibration, {"scope": "path control, not a release"})
    ledger = tmp_path / "persistent-ledgers/new-motion.json"
    m.update(motion_calibration=str(calibration), motion_exposure_ledger=str(ledger), pairs_per_world=1, pair_seed=0)
    steps = server._build_steps(location, p, "motion-validation", tmp_path / "motion-products", resume=False)
    assert str(ledger) in steps[-1]["argv"] and not ledger.exists()
    feature = {key: str(calibration) for key in ("config", "acceptance_config", "reference_mosaics", "patch_plan", "suite_plan", "source_audit", "calibration_receipt")}
    feature.update(cache=str(tmp_path / "cache"), exposure_ledger=str(tmp_path / "persistent-ledgers/new-feature.json"))
    feature_file = tmp_path / "feature-inputs.json"; write(feature_file, feature); m["feature_inputs"] = str(feature_file)
    step = server._build_steps(location, p, "feature-validation", tmp_path / "feature-products", resume=False)[0]
    assert feature["exposure_ledger"] in step["argv"] and not Path(feature["exposure_ledger"]).exists()


def training_source_inputs(project, monkeypatch, tmp_path, block_cpu_offload):
    location, p = project; p = funded(location, p, "training-source")
    runtime = {"schema": "rswm-legacy-wan-backend-v1", "mode": "base", "config_key": "i2v-A14B", "lora_subdir": None, "training": None,
        "sampling": {"prompt": "SOFTWARE_CONFIGURATION_CONTROL_ONLY", "negative_prompt": "", "steps": 1, "shift": 1.,
                     "guide_scale": [1., 1.], "boundary": .9, "param_dtype": "bf16", "t5_dtype": "bf16", "t5_cpu": False, "inactive_model_cpu": False}}
    if block_cpu_offload is not None:
        runtime["sampling"]["block_cpu_offload"] = block_cpu_offload
    if block_cpu_offload:
        runtime["sampling"].update(t5_cpu=True, inactive_model_cpu=True)
    training_runtime = location.parent / p["training"]["runtime"]; write(training_runtime, runtime)
    inference = copy.deepcopy(runtime); inference["sampling"]["inactive_model_cpu"] = True
    inference_path = location.parent / p["models"]["runtime"]; write(inference_path, inference)
    spec_path = location.parent / p["training"]["spec"]; spec = json.loads(spec_path.read_text())
    spec.update(updates=3, seed=0, trainable_policy={"mode": "new_lora", "rank": 4, "alpha": 4, "targets": "attn_cam", "gradient_checkpointing": True})
    write(spec_path, spec)
    manifest = server._json(server._frozen()[0]); dev = [r["world_id"] for r in manifest["worlds"] if r["role"] == "dev"]
    p["training"]["validation_world_ids"] = [dev[2]]
    weights = tmp_path / "metadata-only-not-real-weights"; weights.mkdir(); (weights / "placeholder").write_text("software metadata control")
    p["models"]["numeric_weights_dir"] = str(weights)
    monkeypatch.setattr(server, "_completed", lambda *a: {"marker_sha256": "a" * 64})
    return location, p, runtime, spec, dev, training_runtime, inference, inference_path, spec_path


@pytest.mark.parametrize("block_cpu_offload", [None, False, True])
def test_training_source_accepts_explicit_off_pilot_dev_and_independent_runtime(project, monkeypatch, tmp_path, block_cpu_offload):
    location, p, runtime, spec, dev, training_runtime, inference, inference_path, spec_path = training_source_inputs(
        project, monkeypatch, tmp_path, block_cpu_offload)
    first_identity = server._stage_identity(location, p, "training-source")
    products = tmp_path / "derived-training-inputs"
    steps = server._build_steps(location, p, "training-source", products, resume=False)
    before_inference = inference_path.read_bytes(); steps[0]["action"]()
    derived = json.loads((products / "training-runtime.json").read_text())
    assert derived["training"]["validation"]["world_ids"] == [dev[2]] and derived["training"]["validation"]["max_total_forwards"] == 3
    assert derived["training"]["expert_residency"] == ("block_cpu" if block_cpu_offload else "resident")
    assert derived["sampling"] == runtime["sampling"] and inference_path.read_bytes() == before_inference
    assert json.loads(training_runtime.read_text()) == runtime and json.loads(spec_path.read_text()) == spec
    inference["sampling"]["prompt"] = "changed inference control, independent training remains frozen"; write(inference_path, inference)
    assert server._stage_identity(location, p, "training-source") == first_identity
    manifest = server._json(server._frozen()[0])
    p["training"]["validation_world_ids"] = [next(r["world_id"] for r in manifest["worlds"] if r["role"] == "train")]
    with pytest.raises(ValueError, match="twelve-world"):
        server._build_steps(location, p, "training-source", tmp_path / "rejected", resume=False)


@pytest.mark.parametrize("changed_field", ["t5_cpu", "inactive_model_cpu", "block_cpu_offload", "gradient_checkpointing"])
def test_training_source_rejects_incomplete_block_cpu_policy_without_rewriting_inputs(project, monkeypatch, tmp_path, changed_field):
    location, p, runtime, spec, _dev, training_runtime, _inference, _inference_path, spec_path = training_source_inputs(
        project, monkeypatch, tmp_path, True)
    if changed_field == "gradient_checkpointing":
        spec["trainable_policy"][changed_field] = False
        write(spec_path, spec)
    else:
        runtime["sampling"][changed_field] = False
        write(training_runtime, runtime)
    before_runtime, before_spec = training_runtime.read_bytes(), spec_path.read_bytes()
    products = tmp_path / "rejected-policy"
    with pytest.raises(ValueError):
        server._build_steps(location, p, "training-source", products, resume=False)
    assert training_runtime.read_bytes() == before_runtime and spec_path.read_bytes() == before_spec
    assert not products.exists()


@pytest.mark.parametrize("block_cpu_offload", [False, True])
def test_training_source_preserves_matching_locked_policy_and_rejects_policy_drift(project, monkeypatch, tmp_path, block_cpu_offload):
    location, p, runtime, spec, dev, training_runtime, _inference, _inference_path, _spec_path = training_source_inputs(
        project, monkeypatch, tmp_path, block_cpu_offload)
    policy = {"trainable_policy": spec["trainable_policy"], "timestep_sampling": "uniform_integer",
        "expert_residency": "block_cpu" if block_cpu_offload else "resident", "precision": spec["precision"],
        "validation": {"world_ids": [dev[2]], "family": "F2", "origin_index": 0, "seed": 0, "max_total_forwards": 3}}
    runtime["training"] = policy
    write(training_runtime, runtime)
    before_runtime = training_runtime.read_bytes()
    products = tmp_path / "matching-locked-policy"
    steps = server._build_steps(location, p, "training-source", products, resume=False)
    steps[0]["action"]()
    assert json.loads((products / "training-runtime.json").read_text())["training"] == policy
    assert training_runtime.read_bytes() == before_runtime
    p["training"]["validation_world_ids"] = [dev[3]]
    with pytest.raises(ValueError, match="Locked Wan training policy differs"):
        server._build_steps(location, p, "training-source", tmp_path / "rejected-locked-policy", resume=False)
    assert training_runtime.read_bytes() == before_runtime


def test_training_marker_allows_later_attempt_additions_but_preserves_old_costs(project, tmp_path):
    _location, p = project
    training = server._train_root(p); write(training / "training-contract.json", {"scope": "DIAGNOSTIC_SNAPSHOT_CONTROL_ONLY"})
    (training / "update-00000001.pt").write_bytes(b"NOT_A_REAL_CHECKPOINT")
    write(training / "update-00000001.json", {"scope": "diagnostic, not a strong receipt"})
    (training / "attempts/attempt-000").mkdir(parents=True)
    (training / "attempts/attempt-000/events.jsonl").write_text("diagnostic old costs retained\n")
    products = tmp_path / "diagnostic-train-products"; products.mkdir()
    marker = {"stage": "train", "products": str(products), "files": {}, "external_artifacts": {str(training): server._inventory(training)},
              "append_only_training_roots": [str(training)], "research_acceptance": "NOT_EVALUATED"}
    marker["marker_sha256"] = server.digest(marker)
    write(Path(p["run_root"]) / "stages/train/completed.json", marker)
    (training / "attempts/attempt-001").mkdir(); (training / "attempts/attempt-001/events.jsonl").write_text("new diagnostic resume cost\n")
    assert server._completed(p, "train")["marker_sha256"] == marker["marker_sha256"]
    (training / "attempts/attempt-000/events.jsonl").write_text("old costs deleted")
    with pytest.raises(ValueError, match="cost evidence changed"): server._completed(p, "train")


@pytest.mark.skipif(os.name != "posix", reason="Production driver requires POSIX")
def test_partial_feature_bundle_exit_one_continues_but_missing_bundle_is_failure(project, monkeypatch, tmp_path):
    location, p = project; p = funded(location, p)
    # Actual existing bundle producer writes a single MISSING_PATCH row. It is
    # explicitly diagnostic and cannot become an instrument release.
    body = """from rswm_inria.geometry import digest
from rswm_inria.features import save_feature_bundle
import sys
backend={'diagnostic_features':True,'release_validated':False}
backend['feature_identity_sha256']=digest(backend)
row={'status':'MISSING_PATCH','complete':False,'planned_pixels':16384,'valid_pixels':0,'feature':None,'diagnostic_features':True,'release_validated':False,'feature_dimension':4,'feature_identity_sha256':backend['feature_identity_sha256']}
save_feature_bundle([row],backend,sys.argv[1],'d'*64)
raise SystemExit(1)
"""
    def compile_steps(location, p, stage, products, *, resume):
        bundle_root = products / "generated-features"
        def validate():
            from rswm_inria.features import load_feature_bundle
            bundle, _rows = load_feature_bundle(bundle_root / "features.json")
            assert bundle["diagnostic_features"] is True
        return [server._step("diagnostic-partial-feature-producer", [sys.executable, "-c", body, str(bundle_root)],
                            outputs=[bundle_root, bundle_root / "features.json"], exits=(0, 1), validate=validate)]
    monkeypatch.setattr(server, "_build_steps", compile_steps); monkeypatch.setattr(server, "_semantic", lambda *a: "DIAGNOSTIC_PARTIAL_FEATURE_CONTROL")
    result = server.run_project(location, "pilot-data")
    assert result["status"] == "REQUESTED_CODE_STAGES_EXECUTED" and result["goal_completed"] is False
    # A separate project with the same rc=1 but no output must remain failed.
    second = server.init_project(tmp_path / "missing-bundle-project", SOURCE)
    q = funded(second, json.loads(second.read_text()))
    def absent(location, p, stage, products, *, resume):
        return [server._step("blocked-feature-no-weights-control", [sys.executable, "-c", "raise SystemExit(1)"],
                            outputs=[products / "features.json"], exits=(0, 1))]
    monkeypatch.setattr(server, "_build_steps", absent)
    assert server.run_project(second, "pilot-data")["status"] == "HANDOFF_REQUIRED"
    assert not (Path(q["run_root"]) / "stages/pilot-data/completed.json").exists()


@pytest.mark.skipif(os.name != "posix", reason="Production driver requires POSIX")
def test_actual_retained_safe_vendor_prepares_source_without_any_weights_or_model(tmp_path):
    vendor = SOURCE / "vendor/legacy-original"; inventory = SOURCE / "vendor/legacy-source-inventory.json"
    if not vendor.exists(): pytest.skip("Complete source bundle vendor is unavailable in this installation")
    runtime = {"schema": "rswm-legacy-wan-backend-v1", "mode": "base", "config_key": "i2v-A14B", "lora_subdir": None, "training": None,
        "sampling": {"prompt": "SOFTWARE_SOURCE_PREPARATION_CONTROL_ONLY", "negative_prompt": "", "steps": 1, "shift": 1.,
                     "guide_scale": [1., 1.], "boundary": .9, "param_dtype": "bf16", "t5_dtype": "bf16", "t5_cpu": False, "inactive_model_cpu": False}}
    runtime_path = tmp_path / "runtime.json"; write(runtime_path, runtime)
    out = tmp_path / "prepared-source"
    argv = [sys.executable, str(SOURCE / "scripts/prepare_legacy_wan.py"), "--source-dir", str(vendor), "--source-inventory", str(inventory),
            "--backend-config", str(runtime_path), "--out", str(out)]
    exit_code = server._run_process(argv, tmp_path / "command", cwd=SOURCE,
        budget={"max_wall_seconds": 30, "max_output_bytes": 4_000_000, "max_attempts": 1}, roots=[tmp_path],
        prior_seconds=0, attempt_started=time.monotonic())
    assert exit_code == 0 and (out / "backend.py").exists() and (out / "backend_config.json").exists()
    assert not (out / "wan/remotesensebenchmark").exists()
    result = json.loads((tmp_path / "command/stdout.log").read_text())
    assert result["status"] == "PREPARED_SOURCE_ONLY" and result["real_model_validation"] == "NOT_RUN"
