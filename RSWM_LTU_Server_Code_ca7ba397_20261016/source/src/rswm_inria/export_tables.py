"""Recompute a suite and export exact CSV/JSON/LaTeX tables with provenance."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .aggregate import aggregate_suite
from .geometry import canonical, file_hash
from .suite import _json, _save


def latex_escape(value):
    """Treat identifiers as text, including backslashes and every TeX special."""
    escapes={"\\":r"\textbackslash{}","&":r"\&","%":r"\%","$":r"\$","#":r"\#",
             "_":r"\_","{":r"\{","}":r"\}","~":r"\textasciitilde{}","^":r"\textasciicircum{}"}
    return "".join(escapes.get(c,c) for c in str(value))


def _display(value):
    if value is None:
        return "--"
    if isinstance(value,float):
        return format(value,".6f")
    if isinstance(value,(dict,list)):
        return latex_escape(canonical(value).decode())
    return latex_escape(value)


def _ordered(rows):
    return sorted(rows,key=lambda row:canonical(row))


def _latex(path, headers, rows, notes):
    with path.open("x",encoding="utf-8") as f:
        f.write("% Generated from exact JSON sufficient statistics.\n")
        for note in notes:
            f.write("% "+latex_escape(note).replace("\n"," ")+"\n")
        f.write(r"\begin{tabular}{"+"l"*len(headers)+"}\n"+r"\hline"+"\n")
        for note in notes:
            if note.startswith(("Overall analysis acceptance:","Primary attempt selection:")):
                f.write(r"\multicolumn{"+str(len(headers))+r"}{l}{"+latex_escape(note)+r"} \\"+"\n")
        f.write(" & ".join(latex_escape(h) for h in headers)+r" \\"+"\n"+r"\hline"+"\n")
        for row in rows:
            f.write(" & ".join(_display(v) for v in row)+r" \\"+"\n")
        f.write(r"\hline"+"\n"+r"\end{tabular}"+"\n")


def _metric_pointer(metrics, kind, rgb_row, selection):
    """Refer to the appropriate statistical unit; never broadcast its score."""
    keys=("model_id","study","role","family","protocol","variant")
    rows=[r for r in metrics.get(kind+"_rows",[]) if r.get("release_validated") is True
          and r.get("diagnostic_only") is not True and r.get("analysis_selection")==selection and all(
        r.get(k)==rgb_row.get(k) for k in keys if k in r)]
    if not rows:
        return "NOT_AVAILABLE"
    return "SEPARATE_"+kind.upper()+"_TABLE"


def _paper_tables(report, folder, metrics):
    folder.mkdir()
    notes=["Null is --: unmeasured, missing, or not applicable; never zero.",
           "Primary attempt selection: "+str(report.get("primary_selection","unknown")),
           "Overall analysis acceptance: "+str(report.get("acceptance_status","INCOMPLETE")),
           "Estimand: "+str(report.get("estimand_id","unknown")),
           "Missing bounds are descriptive RGB sensitivity ranges, not confidence intervals.",
           "Motion and quality use separate statistical units and tables; no combined score.",
           "NOT_AVAILABLE means no retained, verified model evaluation for this identity."]
    headers=["Model","Role","Family","Protocol","Variant","Stream","Metric","MAE","Coverage","Missing bound","Scope","Motion","Quality"]
    for filename,families in (("main_control.tex",{"F2","F3"}),("main_persistence.tex",{"F5","F7"})):
        rows=[[r.get(k) for k in ("model_id","role","family","protocol","variant","stream","metric","mae","coverage","missing_bound","report_scope")]
              +[_metric_pointer(metrics,"motion",r,report.get("primary_selection")),
                _metric_pointer(metrics,"quality",r,report.get("primary_selection"))]
              for r in _ordered(report.get("equal_city",[])) if r.get("family") in families]
        _latex(folder/filename,headers,rows,notes)
    rows=[[r.get(k) for k in ("model_id","role","world_id","family","stream","seed","planned_outputs","valid_outputs","output_completeness","status")]
          for r in _ordered(report.get("run_completeness",[]))]
    _latex(folder/"coverage.tex",["Model","Role","World","Family","Stream","Seed","Planned outputs","Valid outputs","Coverage","Status"],rows,
           notes+["Unique run/stream/event generation slots; F7 branch and paired evaluation views add no outputs."])
    rows=[]
    for total in _ordered(report.get("resource_totals",[])):
        for table in ("metrics","adapter_counters"):
            for field,m in sorted(total.get(table,{}).items()):
                identifier="counter/"+field if table=="adapter_counters" else field
                unit=m.get("unit") or ("count" if table=="adapter_counters" else
                                        "seconds" if field.endswith("_seconds") else "bytes")
                rows.append([total.get("model_id"),total.get("study"),total.get("role"),total.get("protocol"),identifier,unit,
                             m.get("value"),m.get("measured_partial_sum",m.get("observed_peak")),
                             "complete" if m.get("complete") else "PARTIAL",m.get("eligible_measurement_units"),
                             m.get("measured_units"),m.get("unknown_attempts"),total.get("run_coverage"),m.get("reason_codes")])
    _latex(folder/"resources.tex",["Model","Study","Role","Protocol","Resource","Unit","Full total/peak","Measured partial/observed peak",
                                  "Measurement","Known eligible units","Measured units","Unknown attempts","Run coverage","Reasons"],rows,
           notes+["Partial sums and observed peaks are not complete resource totals; legacy v2 completeness is unknown.",
                  "Unknown attempts mean the full eligible-unit denominator cannot be recovered.",
                  "gpu_time_seconds is a compatibility alias for gpu_synchronized_model_wall_seconds, not an additional cost.",
                  "RSS is process lifetime high water, including evaluator/dependencies/shared weights."])
    for name,columns in (
        ("motion",("model_id","study","role","family","protocol","variant","training_seed","seed",
                   "analysis_selection","planned_edges","measured_edges","trusted_coverage",
                   "ambiguous_edges","insufficient_evidence_edges","generation_failure_edges",
                   "mean_normalized_step_error","conditional_pooled_edge_error","missing_worlds","scope")),
        ("quality",("model_id","study","role","family","protocol","variant","training_seed","seed",
                    "analysis_selection","planned_tiles","complete_tiles","missing_pixels","tile_coverage","pixel_coverage",
                    "polynomial_mmd_u","conditional_complete_sample_mmd_u","missing_source_worlds",
                    "real_reference","real_real_controls","feature_identity_sha256","scope"))):
        _latex(folder/(name+".tex"),list(columns),
               [[r.get(k) for k in columns] for r in _ordered(metrics.get(name+"_rows",[]))],
               notes+["Availability: "+canonical(metrics.get("availability",{}).get(name,{})).decode(),
                      "Diagnostic/synthetic scopes do not approve formal research gates."])


def export_tables(report, out, *, plan=None, suite_root=None, motion_results=(), quality_results=()):
    """Export RGB and independently verified retained instrument evaluations.

    Optional inputs are producer artifact paths, never caller-supplied scores.
    Validate them before publishing any output so a bad binding fails closed.
    """
    if motion_results or quality_results:
        if plan is None or suite_root is None:
            raise ValueError("Metric result consumption requires the frozen suite plan and retained suite")
    if plan is not None:
        if report.get("plan_sha256")!=plan.get("plan_sha256") or report.get("source_code_sha256")!=plan.get("source_code_sha256"):
            raise ValueError("RGB and metric evaluation plan/code identity mismatch")
        from .metric_results import consume_metric_results
        metrics=consume_metric_results(plan,suite_root,motion_results=motion_results,quality_results=quality_results)
    else:
        from .metric_results import SCHEMA
        metrics={"schema":SCHEMA,"motion_rows":[],"quality_rows":[],"provenance":[],
                 "availability":{name:{"status":"NOT_AVAILABLE","formal_status":"NOT_AVAILABLE",
                                       "diagnostic_available":False,"formal_metric_eligible":False,
                                       "reason_codes":["NO_RETAINED_MODEL_EVALUATION_PROVIDED"]}
                                 for name in ("motion","quality")}}
    root=Path(out)
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Table output directory must be empty")
    _save(root/"aggregation.json",report,exclusive=True)
    tables=("cells","worlds","cities","equal_city","resources","resource_attempts","resource_totals","paired_comparisons",
            "first_attempt","recovery_attempt","recovery_equal_city","seed_worlds","seed_coverage","run_completeness")
    table_rows={name:report.get(name,[]) for name in tables}
    table_rows.update({name:metrics.get(name+"_rows",[]) for name in ("motion","quality")})
    for name,source_rows in table_rows.items():
        rows=_ordered(source_rows)
        fields=sorted({k for row in rows for k in row})
        filename="run-completeness" if name=="run_completeness" else name
        with (root/(filename+".csv")).open("x",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k:canonical(v).decode() if isinstance(v,(dict,list)) else v for k,v in row.items()})
    _save(root/"failure-summary.json",report.get("failure_summary",{}),exclusive=True)
    _save(root/"integrity-summary.json",report.get("integrity_summary",{}),exclusive=True)
    _save(root/"metric-results.json",metrics,exclusive=True)
    _paper_tables(report,root/"tables",metrics)
    (root/"figure-data").mkdir()
    _save(root/"figure-data"/"rgb-coverage.json",{"estimand_id":report.get("estimand_id"),
          "equal_city":report.get("equal_city",[]),"seed_coverage":report.get("seed_coverage",[])},exclusive=True)
    for name in ("motion","quality"):
        _save(root/"figure-data"/(name+".json"),{
            "rows":metrics.get(name+"_rows",[]),"availability":metrics.get("availability",{}).get(name),
            "scope":"Verified retained evaluations; preserve each row's statistical unit and selection."},exclusive=True)
    _save(root/"provenance.json",{
        "analysis_id":report.get("analysis_id"),"analysis_configuration":report.get("analysis_configuration"),
        "plan_sha256":report["plan_sha256"],"source_code_sha256":report["source_code_sha256"],
        "input_results_sha256":report["input_results_sha256"],"input_artifact_hashes":report.get("input_artifact_hashes",{}),
        "weighting":report["weighting"],"estimand_id":report.get("estimand_id"),
        "metric_results_provenance":metrics.get("provenance",{}),"metric_availability":metrics.get("availability",{}),
        "files":{p.relative_to(root).as_posix():file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()},
        "null_in_csv":"empty cell; never interpret as zero","null_in_latex":"--; unmeasured/missing/not applicable",
        "latex_float_format":"fixed six decimal places; exact values remain in JSON",
        "analysis_status":"ANALYSIS_WRITTEN","acceptance_status":report.get("acceptance_status","INCOMPLETE"),
        "real_model_validation":False,"protocol_security_verified":False},exclusive=True)
    return root


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",required=True);parser.add_argument("--suite",required=True)
    parser.add_argument("--out",required=True);parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--motion-result",action="append",default=[],help="Retained motion model evaluation JSON; repeat for independent selections")
    parser.add_argument("--quality-result",action="append",default=[],help="Retained quality model evaluation JSON; repeat for actual frozen populations")
    args=parser.parse_args(argv)
    if args.bootstrap<1:
        parser.error("--bootstrap must be positive")
    try:
        from .suite import suite_read_lock
        # Keep the shared lock through hashing and publication of the analysis.
        with suite_read_lock(args.suite):
            plan=_json(args.plan)
            report=aggregate_suite(plan,args.suite,args.bootstrap,args.seed)
            export_tables(report,args.out,plan=plan,suite_root=args.suite,
                          motion_results=args.motion_result,quality_results=args.quality_result)
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(json.dumps({"analysis_status":"ROOT_ERROR","error_type":type(exc).__name__,"error":str(exc)}))
        return 2
    print(json.dumps({"world_rows":len(report["worlds"]),"city_rows":len(report["cities"]),"out":args.out,
                      "analysis_status":"ANALYSIS_WRITTEN","acceptance_status":report["acceptance_status"]}))
    return 0 if report["acceptance_status"]=="ACCEPTED" else 1


if __name__=="__main__":
    raise SystemExit(main())
