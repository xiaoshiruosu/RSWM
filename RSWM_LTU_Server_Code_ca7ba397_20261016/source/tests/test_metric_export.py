"""Export compatibility, absence semantics and statistical-unit references."""
import json
import pytest
from rswm_inria.export_tables import export_tables, _metric_pointer


def empty_rgb_report():
    return {"plan_sha256":"a"*64,"source_code_sha256":"b"*64,"input_results_sha256":"c"*64,
            "weighting":"equal task/origin/seed within world then equal cities",
            "primary_selection":"first_attempt_only","acceptance_status":"ACCEPTED"}


def test_existing_rgb_api_exports_absence_and_separate_artifacts(tmp_path):
    out=export_tables(empty_rgb_report(),tmp_path/"out")
    result=json.loads((out/"metric-results.json").read_text())
    assert result["motion_rows"]==result["quality_rows"]==[]
    for kind in ("motion","quality"):
        assert result["availability"][kind]["status"]=="NOT_AVAILABLE"
        assert result["availability"][kind]["reason_codes"]==["NO_RETAINED_MODEL_EVALUATION_PROVIDED"]
        assert (out/(kind+".csv")).exists()
        assert (out/"tables"/(kind+".tex")).exists()
        assert json.loads((out/"figure-data"/(kind+".json")).read_text())["rows"]==[]
    provenance=json.loads((out/"provenance.json").read_text())
    assert "metric-results.json" in provenance["files"]
    assert "figure-data/quality.json" in provenance["files"]


def test_result_paths_require_plan_and_retained_suite_before_publication(tmp_path):
    with pytest.raises(ValueError,match="frozen suite plan"):
        export_tables(empty_rgb_report(),tmp_path/"out",motion_results=[1.23])
    assert not (tmp_path/"out").exists()


def test_rgb_metric_code_identity_cannot_differ(tmp_path):
    with pytest.raises(ValueError,match="identity mismatch"):
        export_tables(empty_rgb_report(),tmp_path/"out",plan={"plan_sha256":"d"*64,"source_code_sha256":"b"*64})
    assert not (tmp_path/"out").exists()


def test_main_table_never_broadcasts_distribution_score_or_recovery():
    rgb={"model_id":"m","study":"core","role":"test_unseen_city","family":"F2","protocol":"offline","variant":"default"}
    row={**rgb,"release_validated":True,"analysis_selection":"latest_attempt_within_budget",
         "polynomial_mmd_u":-.12}
    metrics={"quality_rows":[row]}
    assert _metric_pointer(metrics,"quality",rgb,"first_attempt_only")=="NOT_AVAILABLE"
    row["analysis_selection"]="first_attempt_only"
    assert _metric_pointer(metrics,"quality",rgb,"first_attempt_only")=="SEPARATE_QUALITY_TABLE"
    row["diagnostic_only"]=True
    assert _metric_pointer(metrics,"quality",rgb,"first_attempt_only")=="NOT_AVAILABLE"
    row["diagnostic_only"]=False; row["model_id"]="another"
    assert _metric_pointer(metrics,"quality",rgb,"first_attempt_only")=="NOT_AVAILABLE"


@pytest.mark.parametrize("feature_unavailable",[True,False])
def test_cli_unavailable_feature_runtime_or_coordinator_has_structured_failure(tmp_path,monkeypatch,capsys,feature_unavailable):
    from contextlib import nullcontext
    from rswm_inria import export_tables as exporter,suite
    from rswm_inria.features import FeatureUnavailable
    error=FeatureUnavailable("Approved feature runtime unavailable") if feature_unavailable else RuntimeError("Another suite coordinator is active")
    monkeypatch.setattr(suite,"suite_read_lock",lambda _:nullcontext())
    monkeypatch.setattr(exporter,"_json",lambda _: {})
    monkeypatch.setattr(exporter,"aggregate_suite",lambda *args:empty_rgb_report())
    def unavailable(*args,**kwargs):raise error
    monkeypatch.setattr(exporter,"export_tables",unavailable)
    out=tmp_path/"out"
    assert exporter.main(["--plan","plan.json","--suite","suite","--out",str(out)])==2
    result=json.loads(capsys.readouterr().out)
    assert result["analysis_status"]=="ROOT_ERROR" and result["error_type"]==type(error).__name__
    assert result["error"]==str(error) and not out.exists()
