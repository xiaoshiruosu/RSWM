"""Retained software telemetry, never a real generator or layer acceptance."""
import copy
import json

import numpy as np
from PIL import Image
import pytest

from rswm_inria import suite
from rswm_inria.baselines import CopyModel
from rswm_inria.geometry import file_hash
from rswm_inria.tasks import packages


class ObservedCopy(CopyModel):
    def __init__(self):
        self.calls = []

    def generate(self, queries):
        self.calls.append({"call_id": len(self.calls), "native_frames": len(queries)+1,
                           "scope": "software_fixture", "real_backend_layer_consumption": "NOT_RUN"})
        return super().generate(queries)

    def snapshot(self):
        return {"initial": self.initial.copy()}

    def restore(self, state):
        self.initial = state["initial"].copy()

    def generation_evidence(self):
        return {"scope": "software_fixture", "calls": copy.deepcopy(self.calls)}


def plan_for(tmp_path, family="F7"):
    Image.fromarray(np.zeros((2048, 2048, 3), np.uint8)).save(tmp_path/"world.png")
    task = next(t for t, _ in packages("retained-observation-control", "diagnostic_only", 0) if t["family"] == family)
    (tmp_path/"task.json").write_text(json.dumps(task))
    return suite.build_plan([{"task_path":str(tmp_path/"task.json"),"world_path":str(tmp_path/"world.png"),"city":"kitsap"}],
                            [{"model_id":"observed-copy","adapter":"copy"}], [0], "a"*64, "b"*40)


@pytest.fixture(scope="module")
def retained(tmp_path_factory):
    root = tmp_path_factory.mktemp("software_backend_evidence")
    plan = plan_for(root)
    out = root/"suite"
    suite.execute_plan(plan, out, factory=lambda *args: ObservedCopy())
    folder = out/"runs"/plan["expected"][0]["run_id"]/"attempt-000"
    return plan, out, folder


def test_f7_observations_survive_restore_and_prefix_is_generated_once(retained):
    plan, out, folder = retained
    evidence = json.loads((folder/"backend-evidence.json").read_text())
    calls = evidence["observations"]["calls"]
    assert [c["call_id"] for c in calls] == list(range(10))
    assert all(c["native_frames"] == 17 and c["scope"] == "software_fixture" for c in calls)
    assert evidence["research_acceptance"] == "NOT_EVALUATED"
    assert evidence["model_manifest_sha256"] == plan["expected"][0]["model_manifest_sha256"]
    manifest = json.loads((folder/"artifact-manifest.json").read_text())
    assert manifest["files"]["backend-evidence.json"] == file_hash(folder/"backend-evidence.json")
    record = suite.replay_suite(plan, out)[0]
    assert record["replay_verified"] and not record["integrity_errors"]
    rows = [json.loads(r) for r in (folder/"submission/events.jsonl").read_text().splitlines()]
    assert sum(r["kind"] == "output" for r in rows) == 160
    assert sum(r["kind"] == "output" and r["stream"] == "prefix" for r in rows) == 32


@pytest.mark.parametrize("fault", ["bytes", "missing", "wrong_seed", "wrong_model", "wrong_task", "wrong_attempt", "wrong_source", "remove_manifest_entry"])
def test_bad_retained_observations_fail_integrity_without_deleting_rgb(retained, fault):
    plan, out, folder = retained
    path = folder/"backend-evidence.json"
    original, manifest_path = path.read_bytes(), folder/"artifact-manifest.json"
    original_manifest = manifest_path.read_bytes()
    try:
        evidence = json.loads(original)
        if fault in {"missing", "remove_manifest_entry"}:
            path.unlink()
            if fault == "remove_manifest_entry":
                manifest = json.loads(original_manifest); manifest["files"].pop("backend-evidence.json")
                manifest_path.write_text(json.dumps(manifest))
        elif fault == "bytes":
            path.write_bytes(b'{"truncated":')
        else:
            field = {"wrong_seed":"inference_seed", "wrong_model":"model_manifest_sha256", "wrong_task":"task_sha256",
                     "wrong_attempt":"attempt", "wrong_source":"source_code_sha256"}[fault]
            evidence[field] = 1 if field in {"inference_seed", "attempt"} else "c"*64
            path.write_text(json.dumps(evidence))
            # Re-signing the local artifact list cannot repair a wrong identity.
            manifest = json.loads(original_manifest); manifest["files"][path.name] = file_hash(path)
            manifest_path.write_text(json.dumps(manifest))
        record = suite.replay_suite(plan, out)[0]
        assert record["replay_verified"] and record["submission_status"] == "VERIFIED"
        assert any(e["object"] == "backend_evidence" for e in record["integrity_errors"])
        assert suite.integrity_summary(plan, out, [record])["accepted"] is False
    finally:
        path.write_bytes(original); manifest_path.write_bytes(original_manifest)


@pytest.mark.parametrize("bad", [float("nan"), ["invalid object"]])
def test_bad_live_observer_is_retained_as_failure_and_never_fabricates_measurement(tmp_path, bad):
    class BadObserver(ObservedCopy):
        def generation_evidence(self):
            return {"value":bad} if isinstance(bad, float) else bad
    plan = plan_for(tmp_path, "F2"); out = tmp_path/"suite"
    result = suite.execute_plan(plan, out, factory=lambda *args: BadObserver())[0]
    assert result["status"] == "COMPLETED" and result["backend_evidence_status"] == "OBSERVATION_FAILED"
    record = suite.replay_suite(plan, out)[0]
    assert record["replay_verified"] and any(e["object"] == "backend_evidence" for e in record["integrity_errors"])


def test_models_without_observer_keep_existing_rgb_export_path(tmp_path):
    plan = plan_for(tmp_path, "F2"); out = tmp_path/"suite"
    result = suite.execute_plan(plan, out)[0]
    folder = out/"runs"/plan["expected"][0]["run_id"]/"attempt-000"
    assert "backend_evidence_status" not in result and not (folder/"backend-evidence.json").exists()
    assert suite.replay_suite(plan, out)[0]["replay_verified"]
