import copy
import hashlib
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest

from rswm_inria import lr64
from rswm_inria.geometry import digest
from rswm_inria.suite import DEFAULT_RETRY_POLICY


def world():
    yy, xx = np.indices((1024, 1024))
    return np.stack([xx % 251, yy % 253, (xx + 3*yy) % 255], axis=2).astype(np.uint8)


def inputs(tmp_path):
    pixels = world(); wp = tmp_path/"world.png"; Image.fromarray(pixels).save(wp)
    tasks = []
    for variant in ("exact", "offset"):
        task = lr64.task_for("synthetic-lr64", variant)
        path = tmp_path/(variant + ".json"); lr64._save(path, task, exclusive=True)
        tasks.append({"task_path": path, "world_path": wp})
    return tasks, pixels


def plan_for(tmp_path, adapter="oracle", retry=None):
    entries, pixels = inputs(tmp_path)
    return lr64.build_plan(entries, [{"model_id": adapter, "adapter": adapter, "device": "cpu"}], [0], "a"*64, "b"*40, retry), pixels


@pytest.mark.parametrize("q", [(288, 288), (-288, -288), (-288, 288), (288, -288)])
def test_native_bounds_pixels_without_resize(q):
    pixels = world(); output = lr64.crop(pixels, q)
    x, y = 512 + q[0], 512 + q[1]
    assert np.array_equal(output, pixels[y-120:y+120, x-208:x+208])
    assert output.shape == (240, 416, 3)


@pytest.mark.parametrize("q", [(289, 0), (-289, 0), (0, 289), (0, -289), (2**64-1, 0)])
def test_native_bounds_reject(q):
    with pytest.raises(ValueError):
        lr64.crop(world(), q)


@pytest.mark.parametrize("variant,return_range,gap", [("exact", [59, 63], [34, 42]), ("offset", [58, 62], [33, 41])])
def test_all_target_labels_recomputed_with_scalar_oracle(variant, return_range, gap):
    task = lr64.task_for("w", variant); labels = lr64.validate_task(task)
    q = np.array(task["q"])
    assert [int(labels["reentry_event"].min()), int(labels["reentry_event"].max())] == return_range
    assert [int(labels["invisible_query_gap"].min()), int(labels["invisible_query_gap"].max())] == gap
    for iy, ix in ((0, 0), (3, 91), (64, 64), (127, 127)):
        x, y = int(labels["x"][iy, ix]), int(labels["y"][iy, ix])
        visible = [event for event, (qx, qy) in enumerate(q) if qx-208 <= x < qx+208 and qy-120 <= y < qy+120]
        assert labels["first_event"][iy, ix] == min(e for e in visible if e <= 32)
        assert labels["reentry_event"][iy, ix] == min(e for e in visible if e > 32)
    bad = {k: v.copy() for k, v in labels.items()}; bad["reentry_event"][0, 0] += 1
    with pytest.raises(ValueError):
        lr64.validate_task(task, bad)


def test_first_missing_cannot_repair_native_target():
    task = lr64.task_for("w", "exact"); labels = lr64.validate_task(task)
    pixels = world(); ledger = lr64.Ledger(lr64.crop(pixels, (0, 0)), labels)
    missing = set(int(v) for v in np.unique(labels["first_event"]))
    for event, q in enumerate(task["q"][1:], 1):
        ledger.submit(event, q, None if event in missing else lr64.crop(pixels, q))
    report = ledger.report()
    assert report["metrics"]["reentry"]["expected_pixels"] == 16384
    assert report["metrics"]["reentry"]["valid_pixels"] == 0
    assert report["metrics"]["final_target"]["valid_pixels"] == 0
    assert ledger.reference.shape == (1024, 1024, 3)


@pytest.mark.parametrize("variant", ["exact", "offset"])
def test_oracle_native_runner_and_independent_replay(tmp_path, variant):
    task = lr64.task_for("w", variant); pixels = world()
    result, journal = lr64.run_task(task, pixels, lr64._NativeOracle(pixels), out_dir=tmp_path/"run")
    replayed = lr64.replay_run(task, lr64.crop(pixels, (0, 0)), tmp_path/"run", journal.head)
    assert result["reports"] == replayed["reports"]
    assert replayed["verified_png_outputs"] == 64
    assert len([r for r in journal.rows if r["kind"] == "request"]) == 4
    assert all(m["mae"] == 0 and m["coverage"] == 1 for m in replayed["reports"]["metrics"].values())
    assert journal.rows[0]["world_spec"]["world_hw"] == [1024, 1024]


def test_2048_adapter_must_explicitly_reject_lr64(tmp_path):
    from rswm_inria.baselines import CopyModel
    with pytest.raises(ValueError, match="explicitly support"):
        lr64.run_task(lr64.task_for("w", "exact"), world(), CopyModel())


def test_invalid_batch_and_reset_failure_keep64_slots(tmp_path):
    class Broken(lr64._NativeCopy):
        def reset(self, *args):
            raise RuntimeError("reset failed")
    task = lr64.task_for("w", "exact")
    _, journal = lr64.run_task(task, world(), Broken(), out_dir=tmp_path/"run")
    replayed = lr64.replay_run(task, lr64.crop(world(), (0, 0)), tmp_path/"run", journal.head)
    assert replayed["planned_output_events"] == 64 and replayed["verified_png_outputs"] == 0


def test_plan_native_pair_execution_and_world_aggregation(tmp_path):
    plan, _ = plan_for(tmp_path)
    outcomes = lr64.execute_plan(plan, tmp_path/"suite")
    assert all(r["status"] == "COMPLETED" for r in outcomes)
    result = lr64.aggregate(plan, tmp_path/"suite", bootstrap=20)
    primary = result["analyses"]["first_attempt_only"]
    assert primary["expected_runs"] == primary["completed_runs"] == 2
    assert primary["planned_outputs"] == primary["valid_outputs"] == 128
    assert all(s["mean"] == 0 and s["complete_worlds"] == 1 for s in primary["summaries"])
    assert len(result["all_attempt_resources"]) == 2
    assert all(a["resources"]["accounting_calls"]["planned_output_slots"] == 64 for a in result["all_attempt_resources"])
    assert result["real_model_validation"] == "NOT_RUN"


def test_resource_corruption_retains_verified_native_rgb(tmp_path):
    plan, _ = plan_for(tmp_path)
    lr64.execute_plan(plan, tmp_path/"suite")
    resource = tmp_path/"suite"/"runs"/plan["expected"][0]["run_id"]/"attempt-000"/"resources.json"
    resource.write_text("{")
    records, _ = lr64.replay_suite(plan, tmp_path/"suite")
    assert records[0]["resource_status"] == "METADATA_CORRUPT"
    assert records[0]["replay_verified"] and records[0]["reports"]["valid_outputs"] == 64


def test_outcome_corruption_retains_full_denominator(tmp_path):
    plan, _ = plan_for(tmp_path)
    lr64.execute_plan(plan, tmp_path/"suite")
    outcome = tmp_path/"suite"/"runs"/plan["expected"][0]["run_id"]/"attempt-000"/"outcome.json"
    outcome.write_text("{")
    records, _ = lr64.replay_suite(plan, tmp_path/"suite")
    assert records[0]["status"] == "METADATA_CORRUPT"
    assert records[0]["reports"]["planned_outputs"] == 64
    assert records[0]["reports"]["valid_outputs"] == 0


def test_native_retry_first_attempt_remains_primary(tmp_path):
    policy = {**DEFAULT_RETRY_POLICY, "max_attempts_per_run": 2}
    plan, _ = plan_for(tmp_path, retry=policy)
    def fail(config, world, task):
        raise RuntimeError("load failed")
    first = lr64.execute_plan(plan, tmp_path/"suite", factory=fail)
    assert all(r["status"] == "FAILED_LOAD" for r in first)
    lr64.execute_plan(plan, tmp_path/"suite", resume=True, retry_failed=True)
    primary, _ = lr64.replay_suite(plan, tmp_path/"suite")
    recovery, attempts = lr64.replay_suite(plan, tmp_path/"suite", "latest_attempt_within_budget")
    assert all(r["status"] == "FAILED_LOAD" and r["reports"]["valid_outputs"] == 0 for r in primary)
    assert all(r["status"] == "COMPLETED" for r in recovery)
    assert len(attempts) == 4


def test_native_plan_rejects_partial_pair_or_identity_drift(tmp_path):
    plan, _ = plan_for(tmp_path)
    bad = copy.deepcopy(plan); bad["expected"][0]["variant"] = "offset"
    bad["plan_sha256"] = digest({k:v for k,v in bad.items() if k != "plan_sha256"})
    with pytest.raises(ValueError):
        lr64.validate_plan(bad)
    bad = copy.deepcopy(plan); bad["tasks"].pop(next(iter(bad["tasks"])))
    bad["expected"] = bad["expected"][1:]
    bad["plan_sha256"] = digest({k:v for k,v in bad.items() if k != "plan_sha256"})
    with pytest.raises(ValueError):
        lr64.validate_plan(bad)


def test_replay_rejects_unlogged_repairs_and_tampered_pixels(tmp_path):
    pixels = world(); task = lr64.task_for("w", "exact")
    _, journal = lr64.run_task(task, pixels, lr64._NativeOracle(pixels), out_dir=tmp_path/"run")
    Image.fromarray(np.zeros((240, 416, 3), np.uint8)).save(tmp_path/"run"/"main"/"0001.png")
    with pytest.raises(ValueError, match="pixel hash"):
        lr64.replay_run(task, lr64.crop(pixels, (0, 0)), tmp_path/"run", journal.head)


def test_diagnostic_index_plan_cli_is_executable_and_preserves_independent_scope(tmp_path):
    entries, _ = inputs(tmp_path)
    index = {"schema": "rswm-lr64-diagnostic-index-v1", "study": lr64.STUDY,
             "diagnostic_only": True, "source_manifest_sha256": "a"*64,
             "tasks": [{"task_path": e["task_path"].name, "world_path": e["world_path"].name} for e in entries]}
    lr64._save(tmp_path/"index.json", index, exclusive=True)
    lr64._save(tmp_path/"models.json", [{"model_id": "oracle", "adapter": "oracle", "device": "cpu"}], exclusive=True)
    args = ["plan", "--task-index", str(tmp_path/"index.json"), "--models", str(tmp_path/"models.json"),
            "--code-commit", "b"*40, "--out", str(tmp_path/"plan.json"), "--max-attempts-per-run", "2"]
    assert lr64.main(args) == 0
    plan = lr64._json(tmp_path/"plan.json")
    assert len(plan["expected"]) == 2 and plan["retry_policy"]["max_attempts_per_run"] == 2
    assert all(row["role"] == "diagnostic_only" for row in plan["expected"])
    assert lr64.main(["run", "--plan", str(tmp_path/"plan.json"), "--root", str(tmp_path/"suite")]) == 0
    assert lr64.main(["verify", "--plan", str(tmp_path/"plan.json"), "--root", str(tmp_path/"suite")]) == 0


def test_native_recovery_after_real_process_exit_preserves_unknown_old_cost(tmp_path):
    plan, _ = plan_for(tmp_path, retry={**DEFAULT_RETRY_POLICY, "max_attempts_per_run": 2})
    pp = tmp_path/"plan.json"; lr64._save(pp, plan, exclusive=True)
    root = tmp_path/"suite"
    code = "import os,sys; from rswm_inria import lr64; lr64.execute_plan(lr64._json(sys.argv[1]),sys.argv[2],factory=lambda *args: os._exit(7))"
    child = subprocess.run([sys.executable, "-c", code, str(pp), str(root)], capture_output=True, text=True)
    assert child.returncode == 7, child.stderr
    recovered = lr64.execute_plan(plan, root, resume=True, retry_failed=True)
    assert all(r["status"] == "COMPLETED" for r in recovered)
    result = lr64.aggregate(plan, root, bootstrap=10)
    first = result["analyses"]["first_attempt_only"]
    recovery = result["analyses"]["latest_attempt_within_budget"]
    assert first["valid_outputs"] == 64 and recovery["valid_outputs"] == 128
    assert len(result["all_attempt_resources"]) == 3
    crashed = next(a for a in result["all_attempt_resources"] if a["status"] == "CRASHED")
    assert crashed["resources"] is None and crashed["resource_status"] == "METADATA_CORRUPT"
    assert result["acceptance"]["accepted"] is False
    assert lr64.main(["verify", "--plan", str(pp), "--root", str(root)]) == 1


def test_native_retained_initial_tampering_rejected_by_standalone_replay(tmp_path):
    pixels = world(); task = lr64.task_for("w", "exact")
    _, journal = lr64.run_task(task, pixels, lr64._NativeOracle(pixels), out_dir=tmp_path/"run")
    Image.fromarray(np.zeros((240, 416, 3), np.uint8)).save(tmp_path/"run"/"initial.png")
    with pytest.raises(ValueError, match="initial pixels"):
        lr64.replay_run(task, lr64.crop(pixels, (0, 0)), tmp_path/"run", journal.head)


def test_native_config_rejects_recorded_but_unexecuted_cuda():
    with pytest.raises(ValueError, match="cannot claim CUDA"):
        lr64._config({"model_id": "copy", "adapter": "copy", "device": "cuda"})


def test_materialized_source_lock_rejected_before_any_fake_worlds_are_read(tmp_path):
    selection = {"study": lr64.STUDY, "repo_id": "blanchon/LEVIR_CDPlus",
                 "revision": "8062826f12e4f3cd5690b84278ad67c1f42acc46",
                 "shards": [], "row_counts": {}, "selected_rows": []}
    lr64._save(tmp_path/"selection.json", selection, exclusive=True)
    with pytest.raises(ValueError, match="shard/count lock"):
        lr64.plan_from_materialized(tmp_path, [{"model_id": "oracle", "adapter": "oracle"}], [0], "b"*40)
