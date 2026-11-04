"""Expected-plan RGB aggregation: task/seed within world, then city, then cities.

Conditional errors average measurable task cells equally. Coverage and bounded
missing sensitivity retain every applicable planned cell, including total failure.
Pixel-pooled errors are supplemental and explicitly named, not the main estimand.
"""
from __future__ import annotations
from collections import defaultdict
import math
from .geometry import digest
from .stats import paired_city_summary
from .resources import ResourceRecord
from .suite import replay_suite, validate_plan

GROUP = ("benchmark_config_sha256","study","role","model_id","model_manifest_sha256",
         "family","protocol","variant","stream","metric","metric_config_sha256")
RESOURCE_GROUP = tuple(k for k in GROUP if k not in {"family","stream","metric"})
ESTIMAND_ID = "rswm.rgb.equal-task-origin-seed.world-city.v1"


def _identity(row):
    """Explicit compatibility identities for standalone, pre-suite helpers."""
    return {k:row.get(k, "legacy-unspecified" if k.endswith("sha256") else
                     "unspecified" if k=="role" else "core") for k in GROUP}


def _plan_identity(plan, row):
    benchmark = plan.get("benchmark_config_sha256") or digest({
        "suite_schema":plan["schema"], "study":plan["study"],
        "source_manifest_sha256":row.get("source_manifest_sha256"),
        "metric_config":plan["metric_config"]})
    return {**_identity(row), "benchmark_config_sha256":benchmark}



def missing_bound(error, coverage):
    if coverage is None:
        return None
    if not math.isfinite(coverage) or not 0<=coverage<=1:
        raise ValueError("Invalid coverage")
    if coverage==0:
        return [0.,1.]
    if error is None or not math.isfinite(error) or not 0<=error<=1:
        raise ValueError("RGB error must be finite in [0,1]")
    return [coverage*error, coverage*error+1-coverage]


def _mean(values):
    present = [v for v in values if v is not None]
    return sum(present)/len(present) if present else None


def aggregate_world(records):
    """Compatibility helper; supplied records must include all planned slots."""
    grouped = defaultdict(list)
    seen = set()
    for r in records:
        value = r.get("value")
        if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value)):
            raise ValueError("Values must be finite or None")
        if "run_id" in r:
            key = (r["world_id"],r["run_id"])
            if key in seen:
                raise ValueError("Duplicate run")
            seen.add(key)
        grouped[r["world_id"]].append(r)
    return {world:{"planned":len(rows),"valid":sum(r.get("value") is not None for r in rows),
                   "mean":_mean([r.get("value") for r in rows])} for world,rows in grouped.items()}


def aggregate_city(world_records):
    cities = defaultdict(list)
    for row in world_records:
        if row["mean"] is not None and not math.isfinite(row["mean"]):
            raise ValueError("Nonfinite world mean")
        cities[row["city"]].append(row)
    return {city:{"worlds":len(rows),"valid_worlds":sum(r["mean"] is not None for r in rows),
                  "mean":_mean([r["mean"] for r in rows])} for city,rows in cities.items()}


def _streams(reports):
    if "metrics" in reports:
        yield "main",reports
        if "branches" in reports:
            branches = list(reports["branches"].values())
            e = sum(r["expected_pixels"] for r in branches)
            both = all(r["expected_pixels"]>0 and r["valid_pixels"]==r["expected_pixels"] for r in branches)
            v = e if both else 0
            a = sum(r["absolute_error_rgb"] for r in branches) if both else 0
            metric = {"expected_pixels":e,"valid_pixels":v,"absolute_error_rgb":a,
                      "mae":a/(255*3*v) if v else None,"coverage":v/e if e else None}
            yield "paired",{"metrics":{"paired_reentry":metric},"planned_outputs":None,"valid_outputs":None}
        for branch,metric in reports.get("branches",{}).items():
            yield "branch-"+branch,{"metrics":{"reentry":metric},"planned_outputs":None,"valid_outputs":None}
    else:
        yield from reports.items()


def metric_cells(plan, results):
    """Left join frozen expected; only independently replayed scores are accepted."""
    validate_plan(plan)
    expected = {r["run_id"]:r for r in plan["expected"]}
    indexed = {}
    for r in results:
        if r["run_id"] not in expected or r["run_id"] in indexed:
            raise ValueError("Unknown or duplicate run result")
        indexed[r["run_id"]] = r
    cells = []
    bindings = {}
    for row in plan["expected"]:
        binding = (row["city"], row.get("role"), row.get("world_file_sha256"), row.get("world_pixel_sha256"))
        if row["world_id"] in bindings and bindings[row["world_id"]]!=binding:
            raise ValueError("World assigned to inconsistent city/role/pixels")
        bindings[row["world_id"]] = binding
        if row.get("metric_config_sha256")!=digest(plan["metric_config"]):
            raise ValueError("Metric config identity mismatch")
        if row.get("model_manifest_sha256")!=digest(plan["models"][row["model_id"]]):
            raise ValueError("Model manifest identity mismatch")
        result = indexed.get(row["run_id"],{})
        wanted = dict(_streams(row["planned_reports"]))
        actual = dict(_streams(result["reports"])) if result.get("replay_verified") else wanted
        if set(actual)!=set(wanted):
            raise ValueError("Unexpected report streams")
        for stream,baseline in wanted.items():
            observed = actual[stream]
            if set(observed["metrics"])!=set(baseline["metrics"]):
                raise ValueError("Unexpected metrics")
            for metric,planned in baseline["metrics"].items():
                score = observed["metrics"][metric]
                e,v,a = (score[k] for k in ("expected_pixels","valid_pixels","absolute_error_rgb"))
                if any(type(x) is not int for x in (e,v,a)) or e!=planned["expected_pixels"] or not 0<=v<=e or not 0<=a<=255*3*v:
                    raise ValueError("Invalid metric denominator/error")
                error = a/(255*3*v) if v else None
                coverage = v/e if e else None
                if score.get("mae")!=error or score.get("coverage")!=coverage:
                    raise ValueError("Metric report disagrees with sufficient statistics")
                po,vo = observed["planned_outputs"],observed["valid_outputs"]
                if po!=baseline["planned_outputs"] or (po is not None and (type(vo) is not int or not 0<=vo<=po)):
                    raise ValueError("Invalid output counts")
                cells.append({**{k:v for k,v in _plan_identity(plan,row).items() if k not in {"stream","metric"}},
                              "stream":stream,"metric":metric,"run_id":row["run_id"],"world_id":row["world_id"],
                              "city":row["city"],"task_id":row["task_id"],"seed":row["seed"],"origin_index":row["origin_index"],
                              "attempt":result.get("attempt"),"status":result.get("status","NOT_RUN"),
                              "execution_status":result.get("execution_status",result.get("status","NOT_RUN")),
                              "submission_status":result.get("submission_status","NOT_VERIFIED"),
                              "resource_status":result.get("resource_status","UNKNOWN"),
                              "analysis_selection":result.get("selection","legacy_latest_attempt_policy" if isinstance(plan.get("retry_policy"),str) else plan.get("retry_policy",{}).get("primary_selection")),
                              "applicability":"applicable" if e else "not_applicable",
                              "expected_pixels":e,"valid_pixels":v,"absolute_error_rgb":a,
                              "mae":error,"coverage":coverage,"missing_bound":missing_bound(error,coverage),
                              "planned_outputs":po,"valid_outputs":vo})
    return cells


def _summarize(rows, level):
    # Includes all planned applicable units; conditional means do not impute error.
    errors = [r["mae"] for r in rows]
    bounds = [r["missing_bound"] for r in rows if r["missing_bound"] is not None]
    e = sum(r["expected_pixels"] for r in rows)
    v = sum(r["valid_pixels"] for r in rows)
    a = sum(r["absolute_error_rgb"] for r in rows)
    po = [r["planned_outputs"] for r in rows if r["planned_outputs"] is not None]
    vo = [r["valid_outputs"] for r in rows if r["valid_outputs"] is not None]
    return {"level":level,"planned_units":len(rows),"measured_units":sum(v is not None for v in errors),
            "mae":_mean(errors),"coverage":_mean([r["coverage"] for r in rows]),
            "missing_bound":[_mean([b[0] for b in bounds]),_mean([b[1] for b in bounds])] if bounds else None,
            "expected_pixels":e,"valid_pixels":v,"absolute_error_rgb":a,
            "pixel_pooled_mae":a/(255*3*v) if v else None,
            "pixel_pooled_coverage":v/e if e else None,
            "planned_outputs":sum(po) if po else None,"valid_outputs":sum(vo) if po else None,
            "output_completeness":_mean([r["output_completeness"] if "output_completeness" in r else
                                         (r["valid_outputs"]/r["planned_outputs"] if r["planned_outputs"] else None) for r in rows])}


def hierarchical(cells):
    cells = [{**_identity(row),**row} for row in cells]
    grouped = defaultdict(list)
    world_cities = {}
    for cell in cells:
        if cell["world_id"] in world_cities and world_cities[cell["world_id"]]!=(cell["city"],cell["role"]):
            raise ValueError("World assigned to multiple cities")
        world_cities[cell["world_id"]]=(cell["city"],cell["role"])
        grouped[tuple(cell[k] for k in GROUP)+(cell["world_id"],cell["city"])].append(cell)
    worlds = [{**dict(zip(GROUP,key[:len(GROUP)])),"world_id":key[-2],"city":key[-1],
               **_summarize(rows,"world")} for key,rows in sorted(grouped.items())]
    grouped = defaultdict(list)
    for row in worlds:
        grouped[tuple(row[k] for k in GROUP)+(row["city"],)].append(row)
    cities = [{**dict(zip(GROUP,key[:-1])),"city":key[-1],**_summarize(rows,"city")}
              for key,rows in sorted(grouped.items())]
    grouped = defaultdict(list)
    for row in cities:
        grouped[tuple(row[k] for k in GROUP)].append(row)
    totals = []
    for key,rows in sorted(grouped.items()):
        total = {**dict(zip(GROUP,key)),**_summarize(rows,"equal_city"),"cities":[r["city"] for r in rows]}
        # Do not silently drop a completely unmeasurable city from the main error.
        required = {"sfo","innsbruck"} if total["role"]=="test_unseen_city" else set()
        total["required_cities"] = sorted(required)
        total["report_scope"] = "subset" if required-set(total["cities"]) else "complete_planned_cities"
        if any(r["mae"] is None for r in rows) or total["report_scope"]=="subset":
            total["mae"]=None
        totals.append(total)
    return {"worlds":worlds,"cities":cities,"equal_city":totals}


def comparisons(cells, bootstrap=2000, seed=0):
    """Match the same task/origin/seed cells before world-level paired statistics."""
    cells = [{**_identity(row),**row} for row in cells]
    groups = defaultdict(list)
    # Model weight hashes identify each side, not the shared comparison stratum.
    fields = tuple(k for k in GROUP if k not in {"model_id","model_manifest_sha256"})
    for row in cells:
        groups[tuple(row[k] for k in fields)].append(row)
    output = []
    for key,rows in sorted(groups.items()):
        model_ids = sorted({r["model_id"] for r in rows})
        expected = [{"world_id":w,"city":c} for w,c in sorted({(r["world_id"],r["city"]) for r in rows})]
        by_model = defaultdict(dict)
        for row in rows:
            slot = (row["world_id"],row["task_id"],row["origin_index"],row["seed"])
            if slot in by_model[row["model_id"]]:
                raise ValueError("Duplicate comparison cell")
            by_model[row["model_id"]][slot] = row
        for i,a in enumerate(model_ids):
            for b in model_ids[i+1:]:
                ra,rb,world_pairs = [],[],[]
                for world in expected:
                    slots = {s for s in set(by_model[a])|set(by_model[b]) if s[0]==world["world_id"]}
                    applicable = [s for s in slots if any(by_model[m].get(s,{}).get("expected_pixels",0)>0 for m in (a,b))]
                    matched = [s for s in applicable if all(by_model[m].get(s,{}).get("mae") is not None for m in (a,b))]
                    va = _mean([by_model[a][s]["mae"] for s in matched])
                    vb = _mean([by_model[b][s]["mae"] for s in matched])
                    ra.append({"world_id":world["world_id"],"value":va})
                    rb.append({"world_id":world["world_id"],"value":vb})
                    world_pairs.append({**world,"planned_applicable_cells":len(applicable),"common_measured_cells":len(matched),
                                        "common_cell_coverage":len(matched)/len(applicable) if applicable else None,
                                        "conditional_mean_a":va,"conditional_mean_b":vb})
                output.append({**dict(zip(fields,key)),"model_a":a,"model_b":b,"common_world_cells":world_pairs,
                               "model_manifest_sha256_a":next(r["model_manifest_sha256"] for r in rows if r["model_id"]==a),
                               "model_manifest_sha256_b":next(r["model_manifest_sha256"] for r in rows if r["model_id"]==b),
                               "difference_direction":"A-B; lower RGB error is better",
                               "matching":"same task_id/origin_index/inference seed within world",
                               **paired_city_summary(expected,ra,rb,bootstrap,seed)})
    return output


def _unknown_measurement(field, reason):
    peak = field in {"gpu_peak_allocated_bytes","gpu_peak_reserved_bytes","process_peak_rss_bytes","dynamic_state_bytes"}
    return {"value":None,"applicability":"unknown","eligible_units":None,"measured_units":0,
            "complete":False,"aggregation":"max" if peak else "sum",
            "unit":"seconds" if field.endswith("_seconds") else "bytes" if field.endswith("_bytes") else "count",
            "reason_codes":[reason],"observed_peak" if peak else "measured_sum":None}


def _measurement(resource, field, counter=False):
    """Never infer v2 completeness or upgrade an incomplete scalar to a total."""
    resource = resource or {}
    table = resource.get("counter_measurements" if counter else "measurements", {})
    if resource.get("schema")=="rswm-resource-v3" and isinstance(table,dict) and field in table:
        m = table[field]
        if not isinstance(m,dict):
            return _unknown_measurement(field,"INVALID_MEASUREMENT_OBJECT")
        m = dict(m)
        reasons=m.get("reason_codes")
        if not isinstance(reasons,list) or not all(isinstance(reason,str) for reason in reasons) or not isinstance(m.get("unit"),str) or not isinstance(m.get("scope"),str):
            return _unknown_measurement(field,"INVALID_MEASUREMENT_SCOPE")
        if m.get("applicability")=="not_applicable":
            if (m.get("eligible_units")!=0 or m.get("measured_units")!=0 or
                any(m.get(k) is not None for k in ("value","measured_sum","observed_peak"))):
                return _unknown_measurement(field,"INVALID_NOT_APPLICABLE_MEASUREMENT")
            return {**m,"value":None,"measured_sum":None,"observed_peak":None,
                    "eligible_units":0,"measured_units":0,"complete":True}
        if m.get("applicability")!="applicable":
            return _unknown_measurement(field,"INVALID_MEASUREMENT_APPLICABILITY")
        eligible,measured = m.get("eligible_units"),m.get("measured_units")
        if (type(eligible) is not int or type(measured) is not int or not 0<=measured<=eligible or
            m.get("aggregation") not in {"sum","max"}):
            return _unknown_measurement(field,"INVALID_MEASUREMENT_COUNTS")
        for key in ("value","measured_sum","observed_peak"):
            v=m.get(key)
            if v is not None and (type(v) not in {int,float} or not math.isfinite(v) or v<0):
                return _unknown_measurement(field,"INVALID_MEASUREMENT_VALUE")
        complete = m.get("complete") is True and measured==eligible and (m.get("value") is not None or eligible==0)
        m["complete"] = complete
        if not complete:
            m["value"] = None
        partial="observed_peak" if m["aggregation"]=="max" else "measured_sum"
        if complete and m.get(partial) is None and m.get("value") is not None:
            m[partial]=m["value"]
        return m
    reason = ("V3_MEASUREMENT_MISSING" if resource.get("schema")=="rswm-resource-v3" else
              "legacy_completeness_unknown" if resource else "RESOURCE_NOT_READABLE")
    result = _unknown_measurement(field,reason)
    value = resource.get("adapter_counters",{}).get(field) if counter else resource.get(field)
    if type(value) in {int,float} and math.isfinite(value) and value>=0:
        result["observed_peak" if result["aggregation"]=="max" else "measured_sum"] = value
        result["measured_units"] = 1
    return result


def _resource_fields(resource):
    measurements=resource.get("measurements",{})
    measurements=measurements if isinstance(measurements,dict) else {}
    return sorted({k for k in [*ResourceRecord().to_dict(),*resource,*measurements]
                   if isinstance(k,str) and k.endswith(("_seconds","_bytes"))})


def resource_rows(plan, results):
    indexed = {r["run_id"]:r for r in results}
    rows=[]
    for row in plan["expected"]:
        result=indexed.get(row["run_id"],{})
        resource=result.get("resources") or {}
        fields=_resource_fields(resource)
        measurements={field:_measurement(resource,field) for field in fields}
        rows.append({**{k:v for k,v in _plan_identity(plan,row).items() if k in RESOURCE_GROUP},
                     "run_id":row["run_id"],"world_id":row["world_id"],"city":row["city"],
                     "status":result.get("status","NOT_RUN"),"attempt":result.get("attempt"),
                     "execution_status":result.get("execution_status",result.get("status","NOT_RUN")),
                     "submission_status":result.get("submission_status","NOT_VERIFIED"),
                     "resource_status":result.get("resource_status","UNKNOWN"),
                     **{field:m["value"] for field,m in measurements.items()},"measurements":measurements,
                     "execution_complete":resource.get("execution_complete"),
                     "measurement_complete":resource.get("measurement_complete") is True and all(m["complete"] for m in measurements.values()),
                     "native_extra_frames":resource.get("native_extra_frames"),
                     "measurement_scope":resource.get("measurement_scope"),"unavailable":resource.get("unavailable")})
    return rows


def resource_attempts(plan, root):
    """All started attempts are retained, even missing or corrupt resource JSON."""
    from .suite import safe_resource_attempts
    return safe_resource_attempts(plan,root)


def _total_measurements(rows, field, expected_runs, observed_runs, counter=False):
    ms=[_measurement(row.get("resources",row),field,counter) for row in rows]
    applicable=[m for m in ms if m.get("applicability")!="not_applicable"]
    all_na=bool(ms) and not applicable
    operation=next((m["aggregation"] for m in ms),
                   "max" if field in {"gpu_peak_allocated_bytes","gpu_peak_reserved_bytes","process_peak_rss_bytes","dynamic_state_bytes"} else "sum")
    partial_key="observed_peak" if operation=="max" else "measured_sum"
    # Full values may be retained as a measured component by a compliant v3 writer.
    values=[m.get(partial_key,m.get("value")) for m in applicable
            if m.get(partial_key,m.get("value")) is not None]
    observed=(max(values) if operation=="max" else sum(values)) if values else None
    complete=bool(rows) and observed_runs==expected_runs and all(m.get("complete") for m in ms)
    full=(max([m["value"] for m in applicable if m.get("value") is not None]) if operation=="max" else
          sum(m["value"] for m in applicable if m.get("value") is not None)) if complete and applicable and any(m.get("value") is not None for m in applicable) else None
    unknown=sum(m.get("eligible_units") is None for m in applicable)
    reasons=sorted({reason for m in ms for reason in m.get("reason_codes",[])})
    if observed_runs<expected_runs:
        reasons.append("EXPECTED_RUN_NOT_STARTED")
    return {"value":None if all_na else full,"full_peak" if operation=="max" else "full_total":None if all_na else full,
            "observed_peak" if operation=="max" else "measured_partial_sum":None if all_na else observed,
            "applicability":"not_applicable" if all_na else "applicable" if not unknown else "unknown",
            "measured_attempts":sum(m.get("measured_units",0)>0 for m in applicable),
            "planned_attempts":len(rows),"complete_attempts":sum(bool(m.get("complete")) for m in ms),
            "eligible_measurement_units":sum(m["eligible_units"] for m in applicable if m.get("eligible_units") is not None),
            "measured_units":sum(m.get("measured_units",0) for m in applicable),
            "unknown_attempts":unknown,"eligible_units_complete":unknown==0,
            "operation":operation,"complete":complete,"reason_codes":reasons,
            "unit":next((m.get("unit") for m in ms if m.get("unit")),
                        "count" if counter else "seconds" if field.endswith("_seconds") else "bytes"),
            "compatibility_alias_for":next((m.get("compatibility_alias_for") for m in ms if m.get("compatibility_alias_for")),None)}


def resource_totals(attempts, plan=None):
    groups=defaultdict(list)
    planned=defaultdict(set)
    identities={}
    if plan is not None:
        for row in plan["expected"]:
            identity={k:v for k,v in _plan_identity(plan,row).items() if k in RESOURCE_GROUP}
            identities[row["run_id"]]=identity
            key=tuple(identity[k] for k in RESOURCE_GROUP)
            planned[key].add(row["run_id"])
            groups[key]
    for row in attempts:
        identity=identities.get(row["run_id"],{k:v for k,v in _identity(row).items() if k in RESOURCE_GROUP})
        key=tuple(identity[k] for k in RESOURCE_GROUP)
        groups[key].append(row)
        if plan is None:
            planned[key].add(row["run_id"])
    output=[]
    for key,rows in sorted(groups.items()):
        known_attempts=[r for r in rows if r.get("attempt",0) is not None]
        observed_runs=len({r["run_id"] for r in known_attempts})
        expected_runs=len(planned[key])
        records=[r.get("resources",r) or {} for r in rows]
        fields=sorted({k for r in [ResourceRecord().to_dict(),*records] for k in _resource_fields(r)})
        counters=sorted({k for r in records for table in ("adapter_counters","counter_measurements")
                         for k in (r.get(table,{}) if isinstance(r.get(table,{}),dict) else {}) if isinstance(k,str)})
        output.append({**dict(zip(RESOURCE_GROUP,key)),"attempts":len(rows),"expected_runs":expected_runs,
                       "observed_runs":observed_runs,"run_coverage":observed_runs/expected_runs if expected_runs else None,
                       "started_attempts":len(known_attempts),"unreadable_run_directories":len(rows)-len(known_attempts),
                       "resource_readable_attempts":sum(bool(r) for r in records),
                       "metrics":{field:_total_measurements(rows,field,expected_runs,observed_runs) for field in fields},
                       "adapter_counters":{field:_total_measurements(rows,field,expected_runs,observed_runs,True) for field in counters},
                       "scope":"all started attempts; planned run coverage retained; RSS is process lifetime high water"})
    return output


def run_completeness(plan, results):
    """Count generation slots once; F7 branch/paired metric views add no outputs."""
    from .replay import schedule
    indexed={r["run_id"]:r for r in results}
    output=[]
    for row in plan["expected"]:
        result=indexed.get(row["run_id"],{})
        slots={(s["stream"],s["event"]) for s in schedule(plan["tasks"][row["task_id"]]) if s["kind"]=="output"}
        valid={(s["stream"],s["event"]) for s in result.get("output_slots",[]) if s.get("valid")}
        if not result.get("replay_verified"):
            valid=set()
        if not valid<=slots:
            raise ValueError("Unknown output slot")
        for stream in sorted({s[0] for s in slots}):
            planned_stream={s for s in slots if s[0]==stream}
            measured=valid&planned_stream
            output.append({**_plan_identity(plan,row),"run_id":row["run_id"],"task_id":row["task_id"],
                           "world_id":row["world_id"],"city":row["city"],"seed":row["seed"],
                           "stream":stream,"metric":"unique_output_slots","attempt":result.get("attempt"),
                           "status":result.get("status","NOT_RUN"),"planned_outputs":len(planned_stream),
                           "valid_outputs":len(measured),"output_completeness":len(measured)/len(planned_stream)})
    return output


def seed_tables(cells):
    seeds=sorted({r["seed"] for r in cells})
    per_seed=[]
    for seed in seeds:
        per_seed.extend([{**r,"seed":seed} for r in hierarchical([r for r in cells if r["seed"]==seed])["worlds"]])
    groups=defaultdict(list)
    for cell in cells:
        groups[tuple(cell[k] for k in GROUP)+(cell["world_id"],cell["city"])].append(cell)
    coverage=[]
    for key,rows in sorted(groups.items()):
        applicable=[r for r in rows if r["expected_pixels"]>0]
        planned={r["seed"] for r in applicable}
        measured={r["seed"] for r in applicable if r["mae"] is not None}
        coverage.append({**dict(zip(GROUP,key[:len(GROUP)])),"world_id":key[-2],"city":key[-1],
                         "planned_seeds":sorted(planned),"measured_seeds":sorted(measured),
                         "seed_coverage":len(measured)/len(planned) if planned else None,
                         "planned_applicable_cells":len(applicable),"measured_cells":sum(r["mae"] is not None for r in applicable),
                         "scope":"supplemental seed coverage; main estimand retains equal task/origin/seed cells"})
    return per_seed,coverage


def aggregate_suite(plan, root, bootstrap=2000, seed=0):
    from .suite import suite_read_lock, integrity_summary
    validate_plan(plan)
    with suite_read_lock(root):
        results=replay_suite(plan,root)
        first_results=replay_suite(plan,root,selection="first_attempt_only")
        recovery_results=replay_suite(plan,root,selection="latest_attempt_within_budget")
        attempts=resource_attempts(plan,root)
        integrity=integrity_summary(plan,root,results)
    cells=metric_cells(plan,results)
    tables=hierarchical(cells)
    seed_worlds,seed_coverage=seed_tables(cells)
    resources=resource_rows(plan,results)
    policy=plan.get("retry_policy")
    selection=policy.get("primary_selection") if isinstance(policy,dict) else "legacy_latest_attempt_policy"
    identities={r["run_id"]+"-"+str(r.get("attempt")):r.get("provenance",{}) for r in attempts}
    all_errors = [{"run_id":r["run_id"],"scope":"all_attempt_metadata",**e}
                  for r in attempts for e in r.get("integrity_errors",[])]
    integrity["all_attempt_errors"] = all_errors
    integrity["accepted"] = integrity.get("accepted",False) and not all_errors
    input_hash=digest({"results":results,"attempts":attempts,"integrity":integrity})
    failure_rows=[{"run_id":r["run_id"],"attempt":r.get("attempt"),"execution_status":r.get("execution_status",r["status"]),
                   "submission_status":r.get("submission_status"),"resource_status":r.get("resource_status"),
                   "integrity_errors":r.get("integrity_errors",[])} for r in results
                  if r["status"]!="COMPLETED" or r.get("integrity_errors") or r.get("resource_status")!="COMPLETE"]
    accepted=not failure_rows and integrity.get("accepted",False) and all(r["measurement_complete"] for r in resources)
    identity={"plan_sha256":plan["plan_sha256"],"input_results_sha256":input_hash,"retry_policy":policy,
              "estimand_id":ESTIMAND_ID,"bootstrap":bootstrap,"bootstrap_seed":seed,"selection":selection}
    return {"schema":"rswm-aggregation-v2","analysis_id":digest(identity),"analysis_configuration":identity,
            "plan_sha256":plan["plan_sha256"],"source_code_sha256":plan["source_code_sha256"],
            "input_results_sha256":input_hash,"input_artifact_hashes":identities,
            "estimand_id":ESTIMAND_ID,"primary_selection":selection,
            "legacy_latest_attempt_policy":selection=="legacy_latest_attempt_policy",
            "weighting":"equal applicable task/origin/seed cells within world; equal worlds within city; equal cities",
            "missing_bound_scope":"bounded RGB missing sensitivity, not confidence intervals; not applicable to MMD",
            "cells":cells,**tables,"seed_worlds":seed_worlds,"seed_coverage":seed_coverage,
            "run_completeness":run_completeness(plan,results),
            "first_attempt":metric_cells(plan,first_results),"recovery_attempt":metric_cells(plan,recovery_results),
            "recovery_equal_city":hierarchical(metric_cells(plan,recovery_results))["equal_city"],
            "paired_comparisons":comparisons(cells,bootstrap,seed),"resources":resources,
            "resource_attempts":attempts,"resource_totals":resource_totals(attempts,plan),
            "resource_attempt_scope":"primary selection in main; all attempts in resource totals; recovery supplemental",
            "replay_status":results,"failure_summary":{"failed_or_unverifiable_runs":failure_rows,"expected_runs":len(plan["expected"])},
            "integrity_summary":integrity,"analysis_status":"ANALYSIS_WRITTEN","acceptance_status":"ACCEPTED" if accepted else "INCOMPLETE",
            "real_model_validation":False,"protocol_security_verified":False}
