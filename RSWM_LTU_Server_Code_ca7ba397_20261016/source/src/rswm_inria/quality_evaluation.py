"""Frozen-feature distribution diagnostics with world-level sampling.

Generic polynomial MMD-U is not Inception KID or geographic/semantic accuracy.
Conditional complete-sample estimates always retain full planned coverage beside
scores. World bootstrap intervals describe these fixed source-world populations.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
from pathlib import Path
import numpy as np
from .geometry import canonical,digest,rng_for
from .quality import polynomial_mmd_unbiased
from .features import load_feature_bundle

GROUP=("study","role","family","protocol","variant","lineage")


def _source(row):
    return row.get("source_identity",row["world_id"])


def _valid(row):
    feature=row.get("feature")
    if feature is None:return False
    a=np.asarray(feature)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():raise ValueError("QUALITY_FEATURE_SHAPE_OR_FINITE")
    if row.get("status")!="ENCODED" or not row.get("complete"):raise ValueError("QUALITY_INCOMPLETE_PATCH_HAS_FEATURE")
    return True


def coverage(records):
    seen=set();worlds={};sources={};planned=valid=0;complete=0
    for row in records:
        key=row.get("sample_id",row["comparison_id"])
        if key in seen:raise ValueError("QUALITY_DUPLICATE_SAMPLE")
        seen.add(key);planned+=row["planned_pixels"];valid+=row["valid_pixels"]
        if not 0<=row["valid_pixels"]<=row["planned_pixels"]:raise ValueError("QUALITY_PIXEL_DENOMINATOR_INVALID")
        is_valid=_valid(row);complete+=is_valid
        info=worlds.setdefault(row["world_id"],{"planned_tiles":0,"complete_tiles":0})
        info["planned_tiles"]+=1;info["complete_tiles"]+=is_valid
        source=sources.setdefault(_source(row),{"planned_tiles":0,"complete_tiles":0})
        source["planned_tiles"]+=1;source["complete_tiles"]+=is_valid
    return {"planned_tiles":len(records),"complete_tiles":complete,"planned_pixels":planned,"valid_pixels":valid,
            "missing_pixels":planned-valid,"pixel_coverage":valid/planned if planned else None,
            "tile_coverage":complete/len(records) if records else None,"worlds":dict(sorted(worlds.items())),
            "planned_worlds":len(worlds),"complete_sample_worlds":sum(v["complete_tiles"]>0 for v in worlds.values()),
            "source_worlds":dict(sorted(sources.items())),"planned_source_worlds":len(sources),
            "complete_sample_source_worlds":sum(v["complete_tiles"]>0 for v in sources.values())}


def _balanced(records,cap,namespace):
    groups=defaultdict(list)
    for row in records:
        if _valid(row):groups[_source(row)].append(row)
    if not groups:return np.empty((0,0)),[],0
    per_world=min(cap,min(len(rows) for rows in groups.values()))
    chosen=[]
    for world,rows in sorted(groups.items()):
        ordered=sorted(rows,key=lambda r:digest([namespace,world,r["comparison_id"],r.get("sample_id")]))
        chosen.extend(ordered[:per_world])
    dimensions={len(r["feature"]) for r in chosen}
    if len(dimensions)!=1:raise ValueError("QUALITY_FEATURE_DIMENSIONS_DIFFER")
    return np.stack([r["feature"] for r in chosen]),chosen,per_world


def _statistic(xrows,yrows,cap,namespace):
    x,chosen_x,kx=_balanced(xrows,cap,namespace+"/generated")
    y,chosen_y,ky=_balanced(yrows,cap,namespace+"/reference")
    details={"generated_n":len(x),"reference_n":len(y),"generated_worlds":len({_source(r) for r in chosen_x}),
             "reference_worlds":len({_source(r) for r in chosen_y}),"generated_per_world":kx,"reference_per_world":ky,
             "generated_sample_ids":[r.get("sample_id",r["comparison_id"]) for r in chosen_x],
             "reference_sample_ids":[r.get("sample_id",r["comparison_id"]) for r in chosen_y]}
    if len(x)<2 or len(y)<2:return {**details,"status":"INSUFFICIENT_SAMPLES","polynomial_mmd_u":None}
    report=polynomial_mmd_unbiased(x,y)
    if not np.isfinite(report["polynomial_mmd_u"]):raise ValueError("QUALITY_KERNEL_NONFINITE")
    return {**report,**details,"status":"MEASURED_CONDITIONAL_COMPLETE_SAMPLES"}


def evaluate_distribution(generated,reference,*,kind="generated-real",cap=16,bootstrap=200,seed=0):
    if kind not in {"generated-real","real-real"}:raise ValueError("QUALITY_COMPARISON_KIND_INVALID")
    if type(cap) is not int or cap<1 or type(bootstrap) is not int or bootstrap<1 or type(seed) is not int or seed<0:
        raise ValueError("QUALITY_SAMPLING_CONFIGURATION_INVALID")
    ids={r["feature_identity_sha256"] for r in [*generated,*reference]}
    if len(ids)>1:raise ValueError("QUALITY_FEATURE_PROTOCOLS_DIFFER")
    if kind=="real-real":
        xids={_source(r) for r in generated}
        yids={_source(r) for r in reference}
        if xids&yids:raise ValueError("REAL_REAL_SOURCE_WORLDS_MUST_BE_DISJOINT")
    xc,yc=coverage(generated),coverage(reference)
    scope="fixed planned source worlds; complete-sample conditional distribution diagnostic"
    value=_statistic(generated,reference,cap,"rswm-quality-world-balanced-v1")
    bindings={}
    for row in [*generated,*reference]:
        stratum=(row.get("role"),row.get("city"))
        source=_source(row)
        if source in bindings and bindings[source]!=stratum:raise ValueError("QUALITY_SOURCE_ROLE_CITY_CONFLICT")
        bindings[source]=stratum
    rng=rng_for("quality-world-bootstrap-v1",seed);replicates=[]
    for _ in range(bootstrap):
        xb,yb=[],[]
        if kind=="real-real":
            # Independent reference populations retain their own stratum sizes.
            populations=[("x",generated,xb),("y",reference,yb)]
        else:
            # Generated and corresponding real worlds travel together.
            populations=[("paired",[*generated,*reference],None)]
        for population,rows,target in populations:
            population_clusters=defaultdict(set)
            for row in rows:population_clusters[(row.get("role"),row.get("city"))].add(_source(row))
            for stratum,worlds in sorted(population_clusters.items(),key=lambda p:str(p[0])):
                worlds=sorted(worlds)
                selected=[worlds[i] for i in rng.integers(len(worlds),size=len(worlds))]
                for index,world in enumerate(selected):
                    # Every origin/task/seed from a source world travels together.
                    sampled_id=digest([population,stratum,index,world])
                    if target is not None:
                        target.extend({**r,"world_id":sampled_id,"source_identity":sampled_id} for r in rows if _source(r)==world)
                    else:
                        xb.extend({**r,"world_id":sampled_id,"source_identity":sampled_id} for r in generated if _source(r)==world)
                        yb.extend({**r,"world_id":sampled_id,"source_identity":sampled_id} for r in reference if _source(r)==world)
        measured=_statistic(xb,yb,cap,"rswm-quality-world-balanced-v1")
        if measured["polynomial_mmd_u"] is not None:replicates.append(measured["polynomial_mmd_u"])
    has_worlds=value["generated_worlds"]>=2 and value["reference_worlds"]>=2
    interval=np.quantile(replicates,[.025,.975]).tolist() if replicates and has_worlds else None
    return {"schema":"rswm-quality-distribution-v1","metric_name":"polynomial_mmd_u","comparison_kind":kind,
            "sampling_protocol":"same deterministic number of complete planned samples per available world; no replacement tiles",
            "sampling_cap":cap,"sampling_seed":seed,**value,"generated_coverage":xc,"reference_coverage":yc,
            "world_bootstrap_95_sensitivity_interval":interval,"planned_bootstrap_replicates":bootstrap,
            "valid_bootstrap_replicates":len(replicates),"bootstrap_unit":"entire source world, stratified by fixed role/city",
            "bootstrap_pairing":"independent populations" if kind=="real-real" else "paired generated/reference source worlds",
            "bootstrap_status":"MEASURED" if interval is not None else "INSUFFICIENT_COMPLETE_SOURCE_WORLDS_OR_REPLICATES",
            "scope":scope,"diagnostic_features":any(r.get("diagnostic_features",False) for r in [*generated,*reference]),
            "release_validated":False,"semantic_accuracy_claim":False,"is_inception_kid":False}


def evaluate_methods(generated,reference,*,split_seeds=False,**config):
    population=("training_seed","seed","analysis_selection") if split_seeds else ()
    groups=defaultdict(list)
    for row in generated:groups[tuple(row.get(k,"unspecified") for k in GROUP+population)].append(row)
    output=[]
    for key,rows in sorted(groups.items()):
        by_model=defaultdict(list)
        for row in rows:by_model[row.get("model_id","unspecified")].append(row)
        refs=[r for r in reference if tuple(r.get(k,"unspecified") for k in GROUP)==key[:len(GROUP)]]
        if not refs:raise ValueError("QUALITY_REFERENCE_GROUP_MISSING")
        complete_sets=[{r["comparison_id"] for r in method if _valid(r)} for method in by_model.values()]
        common=set.intersection(*complete_sets) if complete_sets else set()
        common_patch_ids={r["patch_id"] for r in rows if r["comparison_id"] in common}
        common_refs=[r for r in refs if r["patch_id"] in common_patch_ids]
        for model,method in sorted(by_model.items()):
            own=evaluate_distribution(method,refs,**config)
            matched=[r for r in method if r["comparison_id"] in common]
            sensitivity=evaluate_distribution(matched,common_refs,**config)
            output.append({**dict(zip(GROUP+population,key)),"model_id":model,"own_complete_sample_diagnostic":own,
                           "common_planned_subset_sensitivity":sensitivity,"common_comparison_units":len(common),
                           "planned_comparison_units":len({r["comparison_id"] for r in method}),
                           "common_subset_is_supplemental":True})
    return output


def evaluate_suite(plan,suite_root,generated_features,reference_features,generated_mosaics,reference_mosaics,
                   patch_plan,out,*,selection=None,cap=16,bootstrap=200,seed=0,validation_receipt=None,
                   feature_config_sha256=None,source_manifest_sha256=None):
    """Publish distribution results bound to retained model PNGs and fixed tiles.

    Old unbound diagnostic reports remain readable as history. Table consumption
    requires this retained suite binding; instrument controls alone are insufficient.
    """
    from .metric_results import evaluation_binding,quality_suite_binding,_feature_mosaic_equal
    from .features import validate_feature_release
    from .suite import suite_read_lock
    with suite_read_lock(suite_root):
        binding=evaluation_binding(plan,suite_root,selection)
        xm,xp,ym,yp,pp=quality_suite_binding(plan,suite_root,generated_mosaics,reference_mosaics,patch_plan,binding["analysis_selection"])
        xb,x=load_feature_bundle(generated_features);yb,y=load_feature_bundle(reference_features)
        _feature_mosaic_equal(x,xp);_feature_mosaic_equal(y,yp)
        if (xb["mosaic_bundle_sha256"]!=xm["bundle_sha256"] or yb["mosaic_bundle_sha256"]!=ym["bundle_sha256"] or
            xb["backend_manifest"]!=yb["backend_manifest"]):
            raise ValueError("QUALITY_FEATURE_MOSAIC_CONFIGURATION_CHANGED")
        release=None
        if validation_receipt is not None:
            release=validate_feature_release(validation_receipt,feature_config_sha256,source_manifest_sha256)
            frozen=release["acceptance_config"]
            if cap!=frozen["primary_sample_cap"] or bootstrap!=frozen["statistics"]["bootstrap"] or seed!=frozen["statistics"]["seed"]:
                raise ValueError("QUALITY_FORMAL_STATISTICAL_PROTOCOL_CHANGED")
            if (source_manifest_sha256 not in binding["source_manifest_sha256"] or xb["diagnostic_features"] or yb["diagnostic_features"] or
                release["report"]["feature_backend_manifest"]!=xb["backend_manifest"]):
                raise ValueError("QUALITY_FORMAL_SOURCE_OR_FEATURE_BACKEND_CHANGED")
        elif feature_config_sha256 or source_manifest_sha256:
            raise ValueError("BLOCKED_QUALITY_REAL_RELEASE_RECEIPT_REQUIRED")
        reports=evaluate_methods(x,y,split_seeds=True,kind="generated-real",cap=cap,bootstrap=bootstrap,seed=seed)
        report={"schema":"rswm-quality-evaluation-v1","generated_bundle_sha256":xb["feature_bundle_sha256"],
                "reference_bundle_sha256":yb["feature_bundle_sha256"],"feature_identity_sha256":xb["backend_manifest"]["feature_identity_sha256"],
                "reports":reports,"diagnostic_features":xb["diagnostic_features"] or yb["diagnostic_features"],
                "release_validated":release is not None,"real_weight_validation":"NOT_RUN" if release is None else "HELDOUT_VALIDATED_IN_STATED_SCOPE",
                "evaluation_binding":binding,"patch_plan_sha256":pp["patch_plan_sha256"],
                "retained_inputs":{"generated_features":str(Path(generated_features).resolve()),
                                   "reference_features":str(Path(reference_features).resolve()),
                                   "generated_mosaics":str(Path(generated_mosaics).resolve()),
                                   "reference_mosaics":str(Path(reference_mosaics).resolve()),
                                   "patch_plan":pp,"validation_receipt":validation_receipt}}
        if release is not None:
            report.update(instrument_release_validation_sha256=release["validation_sha256"],
                          preregistered_feature_config_sha256=feature_config_sha256,source_manifest_sha256=source_manifest_sha256)
        report["statistical_config"]={"schema":"rswm-quality-statistical-protocol-v1","kind":"generated-real", "cap":cap,
                                      "bootstrap":bootstrap,"seed":seed,"kernel":"(x.y/d+1)^3 unbiased U",
                                      "population_grouping":"training_seed,inference_seed,attempt_selection"}
        report["statistical_config_sha256"]=digest(report["statistical_config"])
        report["evaluation_sha256"]=digest(report)
        target=Path(out);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as f:f.write(canonical(report)+b"\n")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated",required=True);parser.add_argument("--reference",required=True);parser.add_argument("--out",required=True)
    parser.add_argument("--kind",choices=["generated-real","real-real"],default="generated-real")
    parser.add_argument("--cap",type=int,default=16);parser.add_argument("--bootstrap",type=int,default=200);parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--validation-receipt");parser.add_argument("--feature-config-sha256");parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--suite-plan");parser.add_argument("--suite");parser.add_argument("--generated-mosaics")
    parser.add_argument("--reference-mosaics");parser.add_argument("--patch-plan")
    parser.add_argument("--selection",choices=["first_attempt_only","latest_attempt_within_budget"])
    args=parser.parse_args(argv)
    try:
        bindings=(args.suite_plan,args.suite,args.generated_mosaics,args.reference_mosaics,args.patch_plan)
        if any(bindings):
            if not all(bindings) or args.kind!="generated-real":raise ValueError("QUALITY_MODEL_SUITE_BINDING_ARGUMENTS_REQUIRED")
            from .replay import load_json
            report=evaluate_suite(load_json(Path(args.suite_plan).read_text(encoding="utf-8")),args.suite,args.generated,args.reference,
                                  args.generated_mosaics,args.reference_mosaics,args.patch_plan,args.out,selection=args.selection,
                                  cap=args.cap,bootstrap=args.bootstrap,seed=args.seed,
                                  validation_receipt=load_json(Path(args.validation_receipt).read_text(encoding="utf-8")) if args.validation_receipt else None,
                                  feature_config_sha256=args.feature_config_sha256,source_manifest_sha256=args.source_manifest_sha256)
            print(canonical({"evaluation_sha256":report["evaluation_sha256"],"groups":len(report["reports"]),
                             "diagnostic_features":report["diagnostic_features"],"release_validated":report["release_validated"],
                             "suite_bound":True}).decode())
            return 0 if report["reports"] and all(r["own_complete_sample_diagnostic"]["polynomial_mmd_u"] is not None for r in report["reports"]) else 1
        xb,x=load_feature_bundle(args.generated);yb,y=load_feature_bundle(args.reference)
        release=None
        if args.validation_receipt:
            from .features import validate_feature_release
            from .replay import load_json
            release=validate_feature_release(load_json(Path(args.validation_receipt).read_text(encoding="utf-8")),
                                             args.feature_config_sha256,args.source_manifest_sha256)
            frozen=release["acceptance_config"]
            if args.cap!=frozen["primary_sample_cap"] or args.bootstrap!=frozen["statistics"]["bootstrap"] or args.seed!=frozen["statistics"]["seed"]:
                raise ValueError("QUALITY_FORMAL_STATISTICAL_PROTOCOL_CHANGED")
            identity=release["report"]["feature_backend_manifest"]["feature_identity_sha256"]
            if (xb["diagnostic_features"] or yb["diagnostic_features"] or
                any(b["backend_manifest"].get("feature_identity_sha256")!=identity for b in (xb,yb))):
                raise ValueError("QUALITY_FEATURES_DIFFER_FROM_VALIDATED_FROZEN_BACKEND")
            worlds=release["source_audit"]["worlds"]
            for row in [*x,*y]:
                world=worlds.get(row["world_id"],{})
                if (row.get("role")!=world.get("role") or row.get("city")!=world.get("city") or
                    row.get("source_identity")!="world-pixels-sha256:"+world.get("world_pixel_sha256","")):
                    raise ValueError("QUALITY_FORMAL_FEATURE_SOURCE_MANIFEST_BINDING")
        elif args.feature_config_sha256 or args.source_manifest_sha256:
            raise ValueError("BLOCKED_QUALITY_REAL_RELEASE_RECEIPT_REQUIRED")
        reports=evaluate_methods(x,y,kind=args.kind,cap=args.cap,bootstrap=args.bootstrap,seed=args.seed)
        report={"schema":"rswm-quality-evaluation-v1","generated_bundle_sha256":xb["feature_bundle_sha256"],
                "reference_bundle_sha256":yb["feature_bundle_sha256"],"feature_identity_sha256":xb["backend_manifest"]["feature_identity_sha256"],
                "reports":reports,"diagnostic_features":xb["diagnostic_features"] or yb["diagnostic_features"],
                "release_validated":release is not None,"real_weight_validation":"NOT_RUN" if release is None else "HELDOUT_VALIDATED_IN_STATED_SCOPE"}
        if release is not None:
            report["instrument_release_validation_sha256"]=release["validation_sha256"]
            report["preregistered_feature_config_sha256"]=args.feature_config_sha256
            report["source_manifest_sha256"]=args.source_manifest_sha256
        report["statistical_config"]={"schema":"rswm-quality-statistical-protocol-v1","kind":args.kind,
                                     "cap":args.cap,"bootstrap":args.bootstrap,"seed":args.seed,"kernel":"(x.y/d+1)^3 unbiased U"}
        report["statistical_config_sha256"]=digest(report["statistical_config"])
        report["evaluation_sha256"]=digest(report)
        out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
        with out.open("xb") as f:f.write(canonical(report)+b"\n")
        print(canonical({"evaluation_sha256":report["evaluation_sha256"],"groups":len(reports),
                         "diagnostic_features":report["diagnostic_features"],"release_validated":report["release_validated"]}).decode())
        return 0 if reports and all(r["own_complete_sample_diagnostic"]["polynomial_mmd_u"] is not None for r in reports) else 1
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print(canonical({"status":"FAILED","reason":str(exc),"error_type":type(exc).__name__}).decode());return 2


if __name__=="__main__":raise SystemExit(main())
