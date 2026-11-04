"""G1 resource/analysis integrity regressions importing the current working tree."""
import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from rswm_inria import suite
from rswm_inria.aggregate import (aggregate_suite, hierarchical, metric_cells, missing_bound,
                                  resource_totals, run_completeness, seed_tables)
from rswm_inria.export_tables import export_tables, latex_escape, main
from rswm_inria.geometry import digest, file_hash
from rswm_inria.tasks import packages


def measurement(value, eligible, measured, partial, complete):
    return {"value":value,"measured_sum":partial,"eligible_units":eligible,"measured_units":measured,
            "complete":complete,"aggregation":"sum","applicability":"applicable","unit":"seconds",
            "scope":"declared synthetic observer intervals",
            "reason_codes":[] if complete else ["GPU_POST_SYNC_FAILED"]}


def attempt(run, n, m):
    return {"run_id":run,"study":"core","model_id":"m","attempt":n,"resources":{
            "schema":"rswm-resource-v3","measurements":{"gpu_time_seconds":m}}}


def test_partial_resource_total_does_not_upgrade_to_complete():
    rows=[attempt("r",0,measurement(.5,1,1,.5,True)),attempt("r",1,measurement(None,2,1,.25,False))]
    total=resource_totals(rows)[0]["metrics"]["gpu_time_seconds"]
    assert total["value"] is None and total["full_total"] is None and not total["complete"]
    assert total["measured_partial_sum"]==.75
    assert total["eligible_measurement_units"]==3 and total["measured_units"]==2
    assert total["complete_attempts"]==1


def test_legacy_scalar_and_missing_attempt_remain_unknown():
    rows=[{"run_id":"r","study":"core","model_id":"m","attempt":0,"resources":{
        "schema":"rswm-resource-v2","generate_seconds":2.}},
        {"run_id":"r","study":"core","model_id":"m","attempt":1,"resources":None}]
    total=resource_totals(rows)[0]
    m=total["metrics"]["generate_seconds"]
    assert total["started_attempts"]==2 and total["resource_readable_attempts"]==1
    assert m["value"] is None and m["measured_partial_sum"]==2.
    assert m["unknown_attempts"]==2 and not m["eligible_units_complete"]
    assert "legacy_completeness_unknown" in m["reason_codes"]


def test_peak_has_separate_observed_value():
    m={"value":None,"observed_peak":123,"eligible_units":2,"measured_units":1,"complete":False,
       "aggregation":"max","applicability":"applicable","unit":"bytes","scope":"declared synthetic state samples","reason_codes":["STATE_READ_FAILED"]}
    rows=[{"run_id":"r","study":"core","model_id":"m","resources":{
        "schema":"rswm-resource-v3","measurements":{"dynamic_state_bytes":m}}}]
    total=resource_totals(rows)[0]["metrics"]["dynamic_state_bytes"]
    assert total["full_peak"] is None and total["observed_peak"]==123


def cell(world,city,seed,error=None,role="diagnostic_only"):
    valid=0 if error is None else 100
    return {"study":"core","model_id":"m","family":"F2","protocol":"offline","variant":"default",
            "role":role,"stream":"main","metric":"known","world_id":world,"city":city,"seed":seed,
            "run_id":f"{world}-{seed}","task_id":"t","origin_index":0,"mae":error,"coverage":valid/100,
            "missing_bound":missing_bound(error,valid/100),"expected_pixels":100,"valid_pixels":valid,
            "absolute_error_rgb":int((error or 0)*255*3*valid),"planned_outputs":32,"valid_outputs":32 if valid else 0}


def test_seed_tables_do_not_replace_frozen_estimand():
    rows=[cell("w","sfo",0,0),cell("w","sfo",0,1),cell("w","sfo",1,1),cell("w","sfo",1)]
    for row in rows:
        # Standalone helper cells are legacy identities; actual suite rows have locks.
        row.update(benchmark_config_sha256="b",model_manifest_sha256="m",metric_config_sha256="p")
    primary=hierarchical(rows)["worlds"][0]["mae"]
    seed_worlds,coverage=seed_tables(rows)
    assert primary==pytest.approx(2/3)
    assert sum(r["mae"] for r in seed_worlds)/2==.75
    assert coverage[0]["seed_coverage"]==1 and coverage[0]["measured_cells"]==3


def test_incomplete_test_city_set_is_not_main_leaderboard():
    result=hierarchical([cell("w","sfo",0,.2,role="test_unseen_city")])["equal_city"][0]
    assert result["report_scope"]=="subset" and result["required_cities"]==["innsbruck","sfo"]
    assert result["mae"] is None


@pytest.fixture
def inputs(tmp_path):
    world=np.zeros((2048,2048,3),np.uint8)
    Image.fromarray(world).save(tmp_path/"world.png")
    items=[]
    for task,_ in packages("export-fixture","diagnostic_only",0):
        path=tmp_path/(task["family"]+".json")
        path.write_text(json.dumps(task))
        items.append({"task_path":str(path),"world_path":str(tmp_path/"world.png"),"city":"sfo"})
    return items


def make_plan(items,families=("F2",),model_id="copy",**kwargs):
    return suite.build_plan([i for i in items if Path(i["task_path"]).stem in families],
                            [{"model_id":model_id,"adapter":"copy"}],[0],"a"*64,"b"*40,**kwargs)


def test_generation_output_slots_not_repeated_across_metrics_or_f7_views(inputs):
    from rswm_inria.replay import schedule
    plan=make_plan(inputs,("F2","F3","F5","F7"))
    results=[]
    for row in plan["expected"]:
        slots=[{"stream":s["stream"],"event":s["event"],"valid":True}
               for s in schedule(plan["tasks"][row["task_id"]]) if s["kind"]=="output"]
        results.append({"run_id":row["run_id"],"status":"COMPLETED","replay_verified":True,"output_slots":slots})
    rows=run_completeness(plan,results)
    assert sum(r["planned_outputs"] for r in rows)==352
    assert sum(r["valid_outputs"] for r in rows)==352
    f7=[r for r in rows if r["family"]=="F7"]
    assert sum(r["planned_outputs"] for r in f7)==160 and {r["stream"] for r in f7}=={"prefix","a","b"}


def test_unstarted_run_coverage_is_preserved(inputs):
    plan=make_plan(inputs)
    rows=resource_totals([],plan)
    assert len(rows)==1 and rows[0]["expected_runs"]==1 and rows[0]["observed_runs"]==0
    assert rows[0]["started_attempts"]==0 and rows[0]["run_coverage"]==0
    assert rows[0]["metrics"]["attempt_wall_seconds"]["value"] is None


def test_identity_hash_mismatch_rejected(inputs):
    plan=make_plan(inputs)
    bad=copy.deepcopy(plan)
    bad["expected"][0]["metric_config_sha256"]="f"*64
    bad["plan_sha256"]=digest({k:v for k,v in bad.items() if k!="plan_sha256"})
    with pytest.raises(ValueError,match="[Mm]etric"):
        metric_cells(bad,[])


def test_resource_half_measurement_stays_partial_in_latex(tmp_path):
    total=resource_totals([attempt("r",0,measurement(None,2,1,.25,False))])
    report={"plan_sha256":"a"*64,"source_code_sha256":"b"*64,"input_results_sha256":"c"*64,
            "weighting":"equal cities","resource_totals":total,"primary_selection":"first_attempt_only",
            "estimand_id":"test","acceptance_status":"INCOMPLETE"}
    export_tables(report,tmp_path/"tables")
    text=(tmp_path/"tables"/"tables"/"resources.tex").read_text()
    assert "PARTIAL" in text and "0.250000" in text and "--" in text
    assert "GPU\\_POST\\_SYNC\\_FAILED" in text


def test_latex_escapes_identifiers_and_provenance_is_deterministic(inputs,tmp_path):
    special=r"m_%&#${}~^\injected"
    plan=make_plan(inputs,model_id=special)
    out=tmp_path/"suite"
    suite.execute_plan(plan,out)
    report=aggregate_suite(plan,out,bootstrap=5,seed=123)
    export_tables(report,tmp_path/"t1");export_tables(report,tmp_path/"t2")
    assert latex_escape(special) in (tmp_path/"t1"/"tables"/"main_control.tex").read_text()
    files1={p.relative_to(tmp_path/"t1").as_posix():file_hash(p) for p in (tmp_path/"t1").rglob("*") if p.is_file()}
    files2={p.relative_to(tmp_path/"t2").as_posix():file_hash(p) for p in (tmp_path/"t2").rglob("*") if p.is_file()}
    assert files1==files2
    provenance=json.loads((tmp_path/"t1"/"provenance.json").read_text())
    assert all("tables/"+name in provenance["files"] for name in ("main_control.tex","main_persistence.tex","coverage.tex","resources.tex"))
    assert provenance["input_artifact_hashes"]


@pytest.mark.parametrize("bad_file",["outcome.json","resources.json"])
def test_cli_bad_run_metadata_writes_full_denominator(inputs,tmp_path,bad_file):
    plan=make_plan(inputs)
    out=tmp_path/"suite"
    outcome=suite.execute_plan(plan,out)[0]
    suite.save_plan(plan,tmp_path/"plan.json")
    (Path(outcome["run_dir"]).parent/bad_file).write_text('{"truncated":')
    rc=main(["--plan",str(tmp_path/"plan.json"),"--suite",str(out),"--out",str(tmp_path/"analysis"),"--bootstrap","5"])
    assert rc==1
    report=json.loads((tmp_path/"analysis"/"aggregation.json").read_text())
    assert len(report["run_completeness"])==1 and report["run_completeness"][0]["planned_outputs"]==32
    assert sum(r["expected_pixels"] for r in report["cells"])>0
    if bad_file=="resources.json":
        assert any(r["valid_pixels"]>0 for r in report["cells"])
    else:
        assert all(r["valid_pixels"]==0 for r in report["cells"])
    assert (tmp_path/"analysis"/"integrity-summary.json").exists()


def test_cli_root_corruption_has_no_fabricated_denominator(tmp_path):
    (tmp_path/"bad-plan.json").write_text('{"broken":')
    rc=main(["--plan",str(tmp_path/"bad-plan.json"),"--suite",str(tmp_path/"missing-suite"),
             "--out",str(tmp_path/"analysis"),"--bootstrap","5"])
    assert rc==2 and not (tmp_path/"analysis"/"aggregation.json").exists()


@pytest.mark.parametrize("legacy",[False,True])
def test_retry_primary_and_recovery_analysis_are_versioned(inputs,tmp_path,legacy):
    kwargs={"schema":suite.LEGACY_SCHEMA} if legacy else {"retry_policy":dict(suite.DEFAULT_RETRY_POLICY,max_attempts_per_run=2)}
    plan=make_plan(inputs,**kwargs)
    out=tmp_path/"suite"
    def fail(*args):
        raise RuntimeError("load failure")
    suite.execute_plan(plan,out,factory=fail)
    suite.execute_plan(plan,out,resume=True,retry_failed=True)
    report=aggregate_suite(plan,out,bootstrap=5)
    assert all(r["valid_pixels"]==0 for r in report["first_attempt"])
    assert any(r["valid_pixels"]>0 for r in report["recovery_attempt"])
    if legacy:
        assert report["legacy_latest_attempt_policy"] and report["primary_selection"]=="legacy_latest_attempt_policy"
        assert any(r["valid_pixels"]>0 for r in report["cells"])
    else:
        assert not report["legacy_latest_attempt_policy"] and report["primary_selection"]=="first_attempt_only"
        assert all(r["valid_pixels"]==0 for r in report["cells"])
        assert all(r["analysis_selection"]=="first_attempt_only" for r in report["cells"])
    assert len(report["resource_attempts"])==2 and report["resource_totals"][0]["started_attempts"]==2
    assert len(report["input_artifact_hashes"])==2


def test_f7_failed_second_branch_is_not_an_extra_output_view(inputs,tmp_path):
    from rswm_inria.baselines import CopyModel
    class BrokenRestore(CopyModel):
        restores=0
        def restore(self,state):
            type(self).restores+=1
            if type(self).restores==2:
                raise RuntimeError("branch b failure")
            return super().restore(state)
    plan=make_plan(inputs,("F7",))
    out=tmp_path/"suite"
    suite.execute_plan(plan,out,factory=lambda *args:BrokenRestore())
    report=aggregate_suite(plan,out,bootstrap=5)
    counts={r["stream"]:r for r in report["run_completeness"]}
    assert sum(r["planned_outputs"] for r in counts.values())==160
    assert counts["prefix"]["valid_outputs"]==32 and counts["a"]["valid_outputs"]==64
    assert counts["b"]["valid_outputs"]==0
    paired=[r for r in report["cells"] if r["stream"]=="paired"]
    assert paired and paired[0]["valid_pixels"]==0


def test_invalid_not_applicable_measurement_is_not_complete():
    m={"value":0,"measured_sum":0,"eligible_units":0,"measured_units":0,"complete":True,
       "aggregation":"sum","applicability":"not_applicable","unit":"seconds","scope":"CPU synthetic fixture","reason_codes":[]}
    result=resource_totals([attempt("r",0,m)])[0]["metrics"]["gpu_time_seconds"]
    assert result["value"] is None and not result["complete"]
    assert "INVALID_NOT_APPLICABLE_MEASUREMENT" in result["reason_codes"]


def test_cpu_gpu_peak_keeps_peak_contract_and_null():
    m={"value":None,"observed_peak":None,"eligible_units":0,"measured_units":0,"complete":True,
       "aggregation":"max","applicability":"not_applicable","unit":"bytes","scope":"CPU synthetic fixture","reason_codes":["NOT_APPLICABLE"]}
    rows=[{"run_id":"r","study":"core","model_id":"m","attempt":0,"resources":{
        "schema":"rswm-resource-v3","measurements":{"gpu_peak_allocated_bytes":m}}}]
    result=resource_totals(rows)[0]["metrics"]["gpu_peak_allocated_bytes"]
    assert result["operation"]=="max" and result["full_peak"] is None and result["observed_peak"] is None
    assert result["applicability"]=="not_applicable"


def test_partial_counter_survives_resource_latex_export(tmp_path):
    m={"value":None,"measured_sum":10,"eligible_units":3,"measured_units":1,"complete":False,
       "aggregation":"sum","applicability":"applicable","unit":"count","scope":"declared synthetic cumulative observations","reason_codes":["COUNTER_READ_FAILED"]}
    rows=[{"run_id":"r","study":"core","model_id":"m","attempt":0,"resources":{
        "schema":"rswm-resource-v3","measurements":{},"counter_measurements":{"native_frames":m}}}]
    totals=resource_totals(rows)
    assert totals[0]["adapter_counters"]["native_frames"]["unit"]=="count"
    report={"plan_sha256":"a"*64,"source_code_sha256":"b"*64,"input_results_sha256":"c"*64,
            "weighting":"equal cities","resource_totals":totals,"primary_selection":"first_attempt_only",
            "estimand_id":"test","acceptance_status":"INCOMPLETE"}
    export_tables(report,tmp_path/"tables")
    text=(tmp_path/"tables"/"tables"/"resources.tex").read_text()
    assert r"counter/native\_frames" in text
    assert r"COUNTER\_READ\_FAILED" in text
    assert "count" in text and "PARTIAL" in text and "Overall analysis acceptance: INCOMPLETE" in text
    assert any(r"\multicolumn" in line and "Overall analysis acceptance: INCOMPLETE" in line for line in text.splitlines())
    row=next(line for line in text.splitlines() if r"counter/native\_frames" in line)
    assert " & -- & 10 & PARTIAL & " in row
