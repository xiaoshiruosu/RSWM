import copy
import numpy as np
from PIL import Image
import pytest

from rswm_inria.geometry import digest
from rswm_inria.paired import paired_packages
from rswm_inria import paired_execution as paired
from rswm_inria.suite import _save


def fixtures(tmp_path, family="F5"):
    yy, xx = np.indices((2048, 2048))
    world = np.stack((xx % 251, yy % 253, (xx+yy) % 255), axis=2).astype(np.uint8)
    wp = tmp_path/"world.png"; Image.fromarray(world).save(wp)
    entries = []
    for task, labels in paired_packages("synthetic-pair", "diagnostic_only", 0):
        if task["family"] != family:
            continue
        path = tmp_path/(task["task_id"] + ".json"); _save(path, task, exclusive=True)
        entries.append({"task_path": path, "world_path": wp, "city": "austin"})
    return entries, world


def make_plan(tmp_path, family="F5", models=None, mechanisms=None):
    entries, _ = fixtures(tmp_path, family)
    return paired.build_paired_plan(entries, models or [{"model_id": "oracle", "adapter": "oracle", "device": "cpu"}],
                                    [0], "a"*64, "b"*40, mechanisms=mechanisms)


def test_pairs_verify_geometry_prefix_i0_and_independent_identity(tmp_path):
    plan = make_plan(tmp_path)
    assert plan["study"] == "same-start-return-pair-v0.1"
    assert len(plan["expected"]) == 2
    assert len({r["pair_id"] for r in plan["expected"]}) == 1
    assert len({r["run_id"] for r in plan["expected"]}) == 2
    assert len({r["initial_sha256"] for r in plan["expected"]}) == 1
    bad = copy.deepcopy(plan)
    bad["expected"][0]["pair_id"] = "forged"
    bad["plan_sha256"] = digest({k:v for k,v in bad.items() if k != "plan_sha256"})
    with pytest.raises(ValueError, match="pair identity"):
        paired.validate_paired_plan(bad)
    bad_tasks = copy.deepcopy(plan["tasks"])
    bad_tasks.pop(next(iter(bad_tasks)))
    with pytest.raises(ValueError, match="both variants"):
        paired.pair_manifest(bad_tasks)


@pytest.mark.parametrize("family,count", [("F5", 192), ("F7", 320)])
def test_actual_paired_execution_replay_cost_and_common_pixels(tmp_path, family, count):
    plan = make_plan(tmp_path, family)
    outcomes = paired.execute_paired_plan(plan, tmp_path/"suite")
    assert all(r["status"] == "COMPLETED" for r in outcomes)
    analysis = paired.aggregate_paired(plan, tmp_path/"suite", bootstrap=20)
    primary = analysis["analyses"]["first_attempt_only"]
    assert primary["expected_runs"] == primary["completed_runs"] == 2
    assert primary["planned_outputs"] == primary["valid_outputs"] == count
    assert all(c["complete_pair"] and c["exact_mae_on_common"] == c["offset_mae_on_common"] == 0 for c in primary["cells"])
    assert all(s["planned_worlds"] == 1 and s["mean_exact_minus_offset"] == 0 for s in primary["world_summaries"])
    assert len(analysis["all_attempt_resources"]) == 2


def test_pair_failure_keeps_missing_side_and_expected_denominator(tmp_path):
    from rswm_inria.baselines import OracleMap
    from rswm_inria.suite import _Missing
    plan = make_plan(tmp_path)
    def factory(config, world, task):
        return _Missing() if task["metadata"]["variant"] == "offset" else OracleMap(world, task["origin"])
    paired.execute_paired_plan(plan, tmp_path/"suite", factory=factory)
    primary = paired.aggregate_paired(plan, tmp_path/"suite", bootstrap=20)["analyses"]["first_attempt_only"]
    assert primary["expected_runs"] == 2 and primary["completed_runs"] == 1
    assert primary["planned_outputs"] == 192 and primary["valid_outputs"] == 96
    assert all(c["expected_pixels"] == 16384 and c["common_valid_pixels"] == 0 for c in primary["cells"])
    assert all(s["mean_exact_minus_offset"] is None for s in primary["world_summaries"])


def test_mechanism_policy_bound_to_real_execution_identity(tmp_path):
    from rswm_inria.history_mechanisms import POLICY_SCHEMA
    policy = {"schema": POLICY_SCHEMA, "mechanism": "frame_cache", "capacity": 3,
              "protect_initial": True, "protect_history": True}
    mid = "frame-cache-" + digest(policy)[:16]
    plan = make_plan(tmp_path, models=[{"model_id": mid, "adapter": "lazy-map", "device": "cpu", "diagnostic_noise": True}], mechanisms={mid: policy})
    outcomes = paired.execute_paired_plan(plan, tmp_path/"suite")
    assert all(r["status"] == "COMPLETED" for r in outcomes)
    resources = paired.safe_resource_attempts(plan, tmp_path/"suite")
    assert all(r["adapter_counters"]["completion_calls"] == 96 for r in resources)
    assert all(r["adapter_counters"]["retrievals"] == 96 for r in resources)
    assert all(r["adapter_counters"]["native_frames"] == 96 for r in resources)
    bad = copy.deepcopy(plan)
    bad["paired_execution"]["mechanisms"][mid]["capacity"] = 4
    bad["plan_sha256"] = digest({k:v for k,v in bad.items() if k != "plan_sha256"})
    with pytest.raises(ValueError, match="Mechanism identity"):
        paired.validate_paired_plan(bad)


def test_partial_pair_origin_seed_world_is_not_a_complete_world():
    base = {"model_id": "m", "role": "diagnostic_only", "family": "F5", "metric": "reentry",
            "world_id": "w", "exact_mae_on_common": .2, "offset_mae_on_common": .1}
    rows = [{**base, "complete_pair": True}, {**base, "complete_pair": False}]
    summary = paired._world_summaries(rows, 40, 0)[0]
    assert summary["planned_worlds"] == 1 and summary["complete_paired_worlds"] == 0
    assert summary["mean_exact_minus_offset"] is None
    assert summary["whole_world_bootstrap_95_interval"] is None
    assert summary["valid_bootstrap_replicates"] == 0
    assert summary["worlds"][0]["cell_coverage"] == .5
    assert summary["worlds"][0]["observed_complete_cell_mean"] == pytest.approx(.1)


def test_world_bootstrap_keeps_all_pair_seed_cells_and_family_roles_separate():
    base = {"model_id": "m", "role": "diagnostic_only", "family": "F5", "metric": "reentry", "complete_pair": True,
            "offset_mae_on_common": 0}
    rows = [{**base, "world_id": "a", "exact_mae_on_common": .1},
            {**base, "world_id": "a", "exact_mae_on_common": .3},
            {**base, "world_id": "b", "exact_mae_on_common": .4},
            {**base, "world_id": "a", "family": "F7", "exact_mae_on_common": .8}]
    summaries = paired._world_summaries(rows, 50, 0)
    f5 = next(s for s in summaries if s["family"] == "F5")
    assert f5["mean_exact_minus_offset"] == pytest.approx(.3)
    assert f5["valid_bootstrap_replicates"] == 50
    assert f5["worlds"][0]["planned_cells"] == 2
    assert next(s for s in summaries if s["family"] == "F7")["mean_exact_minus_offset"] == .8


def test_pair_resource_fault_keeps_verified_rgb_and_cli_returns_failure(tmp_path):
    plan = make_plan(tmp_path)
    plan_path = tmp_path/"plan.json"; _save(plan_path, plan, exclusive=True)
    root = tmp_path/"suite"; paired.execute_paired_plan(plan, root)
    resource = root/"runs"/plan["expected"][0]["run_id"]/"attempt-000"/"resources.json"
    resource.write_text("{")
    analysis = paired.aggregate_paired(plan, root, 10)
    primary = analysis["analyses"]["first_attempt_only"]
    assert primary["valid_outputs"] == 192 and primary["replay_verified_runs"] == 2
    assert primary["complete_resource_runs"] == 1
    assert analysis["acceptance"]["accepted"] is False
    assert paired.main(["verify", "--plan", str(plan_path), "--root", str(root)]) == 1
    assert paired.main(["aggregate", "--plan", str(plan_path), "--root", str(root), "--out", str(tmp_path/"analysis.json"), "--bootstrap", "10"]) == 1


def test_pair_retry_keeps_primary_failure_and_every_attempt_cost(tmp_path):
    from rswm_inria.suite import DEFAULT_RETRY_POLICY
    entries, _ = fixtures(tmp_path)
    plan = paired.build_paired_plan(entries, [{"model_id": "oracle", "adapter": "oracle", "device": "cpu"}],
                                    [0], "a"*64, "b"*40,
                                    retry_policy={**DEFAULT_RETRY_POLICY, "max_attempts_per_run": 2})
    def fail(*args):
        raise RuntimeError("model load failed")
    root = tmp_path/"suite"
    paired.execute_paired_plan(plan, root, factory=fail)
    paired.execute_paired_plan(plan, root, resume=True, retry_failed=True)
    analysis = paired.aggregate_paired(plan, root, 10)
    first = analysis["analyses"]["first_attempt_only"]
    recovery = analysis["analyses"]["latest_attempt_within_budget"]
    assert first["expected_runs"] == 2 and first["valid_outputs"] == 0
    assert recovery["valid_outputs"] == 192 and recovery["completed_runs"] == 2
    assert len(analysis["all_attempt_resources"]) == 4
    assert analysis["acceptance"]["accepted"] is False


def test_pair_auxiliary_metric_version_cannot_be_resealed_to_change_semantics(tmp_path):
    plan = make_plan(tmp_path)
    plan["paired_execution"]["auxiliary_metric"] = "pick-best-final-sample"
    plan["plan_sha256"] = digest({k:v for k,v in plan.items() if k != "plan_sha256"})
    with pytest.raises(ValueError, match="analysis contract"):
        paired.validate_paired_plan(plan)
