"""Preregistered feature acceptance and independent held-out release gates.

Acceptance bounds are supplied by a reviewer after calibration, never invented
by the runner or selected using held-out/test results. Receipts and the exposure
ledger check retained provenance; they are not authenticated global histories.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import stat
import uuid
from .geometry import canonical,digest,file_hash
from .replay import load_json

ACCEPTANCE_SCHEMA="rswm-feature-acceptance-config-v1"
CALIBRATION_SCHEMA="rswm-feature-calibration-v1"
RELEASE_SCHEMA="rswm-feature-release-validation-v1"
CONTROLS=("oracle_reference","fixed_noise_map","tiled","blur","color_perturbation","artificial_seam","independent_real_real")


def _sha(value):return isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)
def _read(path):return load_json(Path(path).read_text(encoding="utf-8"))
def _sealed(value,key):return {**value,key:digest(value)}
def _check(value,key,schema):
    if not isinstance(value,dict) or value.get("schema")!=schema or value.get(key)!=digest({k:v for k,v in value.items() if k!=key}):
        raise ValueError("FEATURE_RELEASE_ARTIFACT_SCHEMA_OR_HASH")
def _write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise ValueError("FEATURE_RELEASE_SYMLINK")
    with path.open("xb") as f:f.write(canonical(value)+b"\n")


def validate_acceptance_config(config):
    _check(config,"acceptance_sha256",ACCEPTANCE_SCHEMA)
    keys={"schema","feature_config_sha256","source_manifest_sha256","calibration_reference_bundle_sha256",
          "calibration_patch_plan_sha256","validation_patch_plan_sha256",
          "statistics","primary_sample_cap","min_complete_tile_coverage","min_valid_pixel_coverage","min_real_real_source_worlds_per_group",
          "control_ranges","interpretation","review_decision","acceptance_sha256"}
    if set(config)!=keys or any(not _sha(config.get(k)) for k in ("feature_config_sha256","source_manifest_sha256","calibration_reference_bundle_sha256","calibration_patch_plan_sha256","validation_patch_plan_sha256")):
        raise ValueError("FEATURE_ACCEPTANCE_IDENTITY_OR_FIELDS")
    stats=config["statistics"]
    if (not isinstance(stats,dict) or set(stats)!={"sample_caps","bootstrap","seed"} or not stats["sample_caps"] or
        len(stats["sample_caps"])!=len(set(stats["sample_caps"])) or any(type(n) is not int or n<1 for n in stats["sample_caps"]) or
        type(stats["bootstrap"]) is not int or stats["bootstrap"]<1 or type(stats["seed"]) is not int or stats["seed"]<0):
        raise ValueError("FEATURE_ACCEPTANCE_STATISTICS_INVALID")
    if type(config["primary_sample_cap"]) is not int or config["primary_sample_cap"] not in stats["sample_caps"]:
        raise ValueError("FEATURE_ACCEPTANCE_PRIMARY_SAMPLE_CAP_NOT_FROZEN")
    for k in ("min_complete_tile_coverage","min_valid_pixel_coverage"):
        if type(config[k]) not in (int,float) or not math.isfinite(config[k]) or not 0<=config[k]<=1:
            raise ValueError("FEATURE_ACCEPTANCE_COVERAGE_INVALID")
    if type(config["min_real_real_source_worlds_per_group"]) is not int or not 2<=config["min_real_real_source_worlds_per_group"]<=6:
        raise ValueError("FEATURE_ACCEPTANCE_INDEPENDENT_WORLD_COUNT")
    ranges=config["control_ranges"]
    if not isinstance(ranges,list):raise ValueError("FEATURE_ACCEPTANCE_CONTROL_RANGES_REQUIRED")
    seen=set()
    for row in ranges:
        if not isinstance(row,dict) or set(row)!={"control_name","sample_cap","minimum","maximum"}:raise ValueError("FEATURE_ACCEPTANCE_CONTROL_RANGE_FIELDS")
        pair=(row["control_name"],row["sample_cap"])
        if pair in seen or pair not in {(name,cap) for name in CONTROLS for cap in stats["sample_caps"]}:raise ValueError("FEATURE_ACCEPTANCE_CONTROL_RANGE_IDENTITY")
        seen.add(pair)
        if any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) for k in ("minimum","maximum")) or row["minimum"]>row["maximum"]:
            raise ValueError("FEATURE_ACCEPTANCE_CONTROL_RANGE_FINITE")
    if seen!={(name,cap) for name in CONTROLS for cap in stats["sample_caps"]}:raise ValueError("FEATURE_ACCEPTANCE_ALL_CONTROL_RANGES_REQUIRED")
    decision=config["review_decision"]
    if (config["interpretation"]!="frozen-feature-distribution-diagnostic-only" or not isinstance(decision,dict) or
        set(decision)!={"reviewer","decision_reference","decision_sha256"} or
        any(not isinstance(decision[k],str) or not decision[k] for k in ("reviewer","decision_reference")) or not _sha(decision["decision_sha256"])):
        raise ValueError("BLOCKED_FEATURE_ACCEPTANCE_REVIEW_DECISION_NOT_PROVIDED")
    return config


def _population(worlds,role):
    city={"metric_calibration":"bellingham","metric_validation":"bloomington"}[role]
    if not isinstance(worlds,list) or len(worlds)!=12 or len({w.get("world_id") for w in worlds})!=12:
        raise ValueError("FEATURE_RELEASE_COMPLETE_ROLE_POPULATION_REQUIRED")
    for w in worlds:
        if (w.get("role")!=role or w.get("city")!=city or not w.get("source_path") or
            any(not _sha(w.get(k)) for k in ("world_file_sha256","world_pixel_sha256","source_sha256"))):
            raise ValueError("FEATURE_RELEASE_SOURCE_ROLE_CITY_HASH")
    if any(len({w[k] for w in worlds})!=12 for k in ("world_pixel_sha256","source_sha256")):
        raise ValueError("FEATURE_RELEASE_DUPLICATE_SOURCE_WORLDS")
    return worlds


def _source_bindings(plan,audit,bundle,role,patch_plan):
    from .suite import validate_plan
    from .materialized import validate_audit_receipt
    from .quality_protocol import validate_patch_plan,_validate_suite_binding
    validate_plan(plan);validate_audit_receipt(audit,plan,required_profile="core120")
    validate_patch_plan(patch_plan);_validate_suite_binding(plan,patch_plan)
    if bundle.get("suite_plan_sha256")!=plan["plan_sha256"]:raise ValueError("FEATURE_RELEASE_REFERENCE_SUITE_IDENTITY")
    if bundle.get("patch_plan_sha256")!=patch_plan["patch_plan_sha256"]:raise ValueError("FEATURE_RELEASE_PATCH_PLAN_IDENTITY")
    patches={p["patch_id"]:p for p in patch_plan["patches"]}
    if len(bundle["patches"])!=len(patches) or {p["patch_id"] for p in bundle["patches"]}!=set(patches):
        raise ValueError("FEATURE_RELEASE_REFERENCE_PATCH_MATRIX")
    by_id=audit["worlds"];world_ids={p["world_id"] for p in bundle["patches"]}
    worlds=[by_id[w] for w in sorted(world_ids)];_population(worlds,role)
    for p in bundle["patches"]:
        if any(p.get(k)!=v for k,v in patches[p["patch_id"]].items()):raise ValueError("FEATURE_RELEASE_REFERENCE_GEOMETRY_CHANGED")
        world=by_id[p["world_id"]]
        if (p.get("role")!=role or p.get("city")!=world["city"] or p.get("reference_kind")!="frozen-real-world" or
            p.get("source_world_file_sha256")!=world["world_file_sha256"] or
            p.get("source_identity")!="world-pixels-sha256:"+world["world_pixel_sha256"]):
            raise ValueError("FEATURE_RELEASE_REFERENCE_SOURCE_BINDING")
    return worlds


def _verify_reference_pixels(patches,plan):
    import numpy as np
    from .suite import _world
    worlds={};bindings={r["world_id"]:r for r in plan["expected"]}
    for p in patches:
        row=bindings[p["world_id"]]
        if p["world_id"] not in worlds:
            image=_world(row["world_path"])
            import hashlib
            if file_hash(row["world_path"])!=row["world_file_sha256"] or hashlib.sha256(image.tobytes()).hexdigest()!=row["world_pixel_sha256"]:
                raise ValueError("FEATURE_RELEASE_WORLD_PIXELS_CHANGED")
            worlds[p["world_id"]]=image
        x,y=p["xy"]
        if not p["complete"] or not np.array_equal(p["image"],worlds[p["world_id"]][y:y+128,x:x+128]):
            raise ValueError("FEATURE_RELEASE_REFERENCE_PIXELS_NOT_REAL_WORLD_CROP")


def _acceptance(report,config,role):
    _check(report,"validation_sha256","rswm-feature-control-validation-v1");validate_acceptance_config(config)
    from .features import validate_backend_manifest
    backend=validate_backend_manifest(report["feature_backend_manifest"])
    if (backend.get("config_sha256")!=config["feature_config_sha256"] or report.get("sample_caps")!=config["statistics"]["sample_caps"] or
        any(report.get(k)!=config["statistics"][k] for k in ("bootstrap","seed"))):raise ValueError("FEATURE_RELEASE_CONTROL_CONFIG_IDENTITY")
    rows=report["reports"];expected={(n,c) for n in CONTROLS for c in config["statistics"]["sample_caps"]}
    if (len(rows)!=len(expected) or {(r.get("control_name"),r.get("sample_cap")) for r in rows}!=expected or
        any(r.get("role")!=role for r in rows)):
        raise ValueError("FEATURE_RELEASE_CONTROL_MATRIX_OR_ROLE")
    limits={(r["control_name"],r["sample_cap"]):r for r in config["control_ranges"]};reasons=[]
    for row in rows:
        stat=row["distribution"];limit=limits[(row["control_name"],row["sample_cap"])];value=stat.get("polynomial_mmd_u")
        if (stat.get("schema")!="rswm-quality-distribution-v1" or stat.get("metric_name")!="polynomial_mmd_u" or
            stat.get("sampling_cap")!=row["sample_cap"] or stat.get("sampling_seed")!=report["seed"] or
            stat.get("planned_bootstrap_replicates")!=report["bootstrap"] or stat.get("is_inception_kid") is not False or
            stat.get("diagnostic_features") is not backend["diagnostic_features"]):
            raise ValueError("FEATURE_RELEASE_CONTROL_STATISTICAL_IDENTITY")
        if type(value) not in (int,float) or not math.isfinite(value) or not limit["minimum"]<=value<=limit["maximum"]:
            reasons.append("CONTROL_RANGE_FAILED_"+row["control_name"])
        for key in ("generated_coverage","reference_coverage"):
            cov=stat[key]
            if (cov.get("tile_coverage") is None or cov["tile_coverage"]<config["min_complete_tile_coverage"] or
                cov.get("pixel_coverage") is None or cov["pixel_coverage"]<config["min_valid_pixel_coverage"]):
                reasons.append("CONTROL_COVERAGE_FAILED_"+row["control_name"])
        if row["control_name"]=="independent_real_real":
            if (stat.get("comparison_kind")!="real-real" or stat.get("bootstrap_pairing")!="independent populations" or
                min(stat.get("generated_worlds",0),stat.get("reference_worlds",0))<config["min_real_real_source_worlds_per_group"]):
                reasons.append("INDEPENDENT_REAL_REAL_POPULATION_INSUFFICIENT")
    return {"acceptance_passed":not reasons,"reason_codes":sorted(set(reasons))}


def _validate_calibration(receipt,config):
    _check(receipt,"calibration_sha256",CALIBRATION_SCHEMA);validate_acceptance_config(config)
    if (receipt.get("acceptance_config")!=config or receipt.get("feature_config_sha256")!=config["feature_config_sha256"] or
        receipt.get("source_manifest_sha256")!=config["source_manifest_sha256"] or
        receipt.get("reference_bundle_sha256")!=config["calibration_reference_bundle_sha256"]):
        raise ValueError("FEATURE_CALIBRATION_FROZEN_IDENTITY")
    _population(receipt["source_worlds"],"metric_calibration")
    _verify_retained_inputs(receipt,"metric_calibration",config)
    result=_acceptance(receipt["report"],config,"metric_calibration")
    if (receipt.get("status")!="CONFIG_FROZEN" or receipt.get("real_calibration_executed") is not True or
        receipt["report"].get("diagnostic_features") is not False or receipt["report"].get("real_data") is not True or
        result["acceptance_passed"] is not True or receipt.get("acceptance_result")!=result):
        raise ValueError("BLOCKED_FEATURE_REAL_CALIBRATION_NOT_ACCEPTED")
    return receipt


def _verify_retained_inputs(receipt,role,config):
    from .features import validate_feature_config
    from .quality_protocol import load_mosaics
    feature_config=_read(receipt["feature_config_path"])
    if digest(feature_config)!=config["feature_config_sha256"]:raise ValueError("FEATURE_RELEASE_RETAINED_CONFIG_CHANGED")
    validate_feature_config(feature_config,Path(receipt["feature_config_path"]).parent)
    bundle,patches=load_mosaics(receipt["reference_mosaics_path"])
    if bundle["bundle_sha256"]!=receipt["reference_bundle_sha256"]:raise ValueError("FEATURE_RELEASE_RETAINED_MOSAICS_CHANGED")
    worlds=_source_bindings(receipt["source_plan"],receipt["source_audit"],bundle,role,receipt["patch_plan"])
    if worlds!=receipt["source_worlds"]:raise ValueError("FEATURE_RELEASE_RETAINED_SOURCE_BINDING_CHANGED")
    required=config["calibration_patch_plan_sha256" if role=="metric_calibration" else "validation_patch_plan_sha256"]
    if receipt["patch_plan"]["patch_plan_sha256"]!=required:raise ValueError("FEATURE_RELEASE_PREREGISTERED_PATCH_PLAN_CHANGED")
    _verify_reference_pixels(patches,receipt["source_plan"])


def _register_exposure(worlds,config,calibration,bundle,ledger_path):
    import fcntl
    path=Path(ledger_path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):raise ValueError("FEATURE_EXPOSURE_SYMLINK")
    fd=os.open(path.with_name(path.name+".lock"),os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise ValueError("FEATURE_EXPOSURE_LOCK_NOT_REGULAR")
        fcntl.flock(fd,fcntl.LOCK_EX)
        ledger=_read(path) if path.exists() else {"schema":"rswm-feature-validation-exposure-v1","exposures":[]}
        if ledger.get("schema")!="rswm-feature-validation-exposure-v1" or not isinstance(ledger.get("exposures"),list):raise ValueError("FEATURE_EXPOSURE_SCHEMA")
        keys=sorted({digest([kind,w[field]]) for w in worlds for kind,field in (("original_source","source_sha256"),("world_pixels","world_pixel_sha256"))})
        for entry in ledger["exposures"]:
            if (not isinstance(entry,dict) or any(not _sha(entry.get(k)) for k in ("reference_bundle_sha256","feature_config_sha256","acceptance_sha256","calibration_sha256")) or
                not isinstance(entry.get("source_population_keys"),list) or not entry["source_population_keys"] or
                len(entry["source_population_keys"])!=len(set(entry["source_population_keys"])) or any(not _sha(k) for k in entry["source_population_keys"])):
                raise ValueError("FEATURE_EXPOSURE_ENTRY_IDENTITY")
        previous=[e for e in ledger["exposures"] if e["reference_bundle_sha256"]==bundle["bundle_sha256"] or set(e["source_population_keys"])&set(keys)]
        changed=any(e["feature_config_sha256"]!=config["feature_config_sha256"] or e["acceptance_sha256"]!=config["acceptance_sha256"] or
                    e["calibration_sha256"]!=calibration["calibration_sha256"] for e in previous)
        entry={"reference_bundle_sha256":bundle["bundle_sha256"],"feature_config_sha256":config["feature_config_sha256"],
               "acceptance_sha256":config["acceptance_sha256"],"calibration_sha256":calibration["calibration_sha256"],"source_population_keys":keys}
        if entry not in ledger["exposures"]:
            ledger["exposures"].append(entry)
            if path.exists():
                temp=path.with_name(path.name+".tmp-"+uuid.uuid4().hex);_write(temp,ledger);os.replace(temp,path)
            else:_write(path,ledger)
        return {"previously_viewed":bool(previous),"used_for_new_config":changed,"source_population_keys":keys,
                "ledger_sha256_at_execution":file_hash(path),"ledger_path":str(path.resolve()),"ledger_snapshot":ledger,
                "history_scope":"caller-retained ledger; not authenticated global non-exposure proof"}
    finally:os.close(fd)


def validate_feature_release(receipt,feature_config_sha256,source_manifest_sha256):
    _check(receipt,"validation_sha256",RELEASE_SCHEMA)
    if not _sha(feature_config_sha256) or not _sha(source_manifest_sha256):raise ValueError("FEATURE_FORMAL_PREREGISTERED_IDENTITIES_REQUIRED")
    if (receipt.get("feature_config_sha256")!=feature_config_sha256 or receipt.get("source_manifest_sha256")!=source_manifest_sha256 or
        receipt.get("status")!="VALIDATED" or receipt.get("release_validated") is not True or receipt.get("real_validation_executed") is not True or
        receipt.get("source_kind")!="canonical_inria" or receipt.get("instrument")!="quality" or receipt.get("role")!="metric_validation"):
        raise ValueError("BLOCKED_FEATURE_REAL_HELDOUT_RELEASE_NOT_VALIDATED")
    config=validate_acceptance_config(receipt["acceptance_config"]);calibration=_validate_calibration(receipt["calibration_receipt"],config)
    if config["feature_config_sha256"]!=feature_config_sha256 or config["source_manifest_sha256"]!=source_manifest_sha256:
        raise ValueError("FEATURE_RELEASE_ACCEPTANCE_PREREGISTRATION_MISMATCH")
    worlds=_population(receipt["source_worlds"],"metric_validation")
    _verify_retained_inputs(receipt,"metric_validation",config)
    for key in ("world_id","world_pixel_sha256","source_sha256"):
        if {w[key] for w in worlds}&{w[key] for w in calibration["source_worlds"]}:raise ValueError("FEATURE_CALIBRATION_VALIDATION_SOURCE_OVERLAP")
    report=receipt["report"]
    if report.get("diagnostic_features") is not False or report.get("real_data") is not True or report.get("real_validation")!="EXECUTED_CANDIDATE_CONTROLS":
        raise ValueError("BLOCKED_FEATURE_DIAGNOSTIC_CANNOT_AUTHORIZE_RELEASE")
    _check(report,"validation_sha256","rswm-feature-control-validation-v1")
    if report.get("reference_bundle_sha256")!=receipt.get("reference_bundle_sha256"):raise ValueError("FEATURE_RELEASE_REFERENCE_HASH_MISMATCH")
    backend=report["feature_backend_manifest"]
    if backend.get("config_sha256")!=feature_config_sha256 or backend.get("weights_sha256")!=receipt.get("weights_sha256"):
        raise ValueError("FEATURE_RELEASE_FROZEN_WEIGHTS_MISMATCH")
    if backend.get("feature_identity_sha256")!=calibration["report"]["feature_backend_manifest"].get("feature_identity_sha256"):
        raise ValueError("FEATURE_RELEASE_CALIBRATED_BACKEND_RUNTIME_CHANGED")
    result=_acceptance(report,config,"metric_validation")
    if not result["acceptance_passed"] or receipt.get("acceptance_result")!=result:raise ValueError("FEATURE_RELEASE_ACCEPTANCE_FAILED")
    exposure=receipt["exposure"]
    expected=sorted({digest([kind,w[field]]) for w in worlds for kind,field in (("original_source","source_sha256"),("world_pixels","world_pixel_sha256"))})
    if (exposure.get("used_for_new_config") is not False or exposure.get("source_population_keys")!=expected or
        not _sha(exposure.get("ledger_sha256_at_execution"))):raise ValueError("FEATURE_RELEASE_HELDOUT_EXPOSURE_CHANGED_CONFIG")
    audit=receipt["source_audit"]
    if (audit.get("schema")!="rswm-materialized-audit-v1" or audit.get("input_kind")!="source_identity_checked_geography_unknown" or
        audit.get("status")!="MATERIALIZED_PIXELS_VALIDATED" or audit.get("acceptance_status")!="ACCEPTED_IN_STATED_SCOPE" or
        audit.get("manifest_sha256")!=source_manifest_sha256 or audit.get("profile")!="core120" or
        any(audit.get("worlds",{}).get(w["world_id"])!=w for w in [*worlds,*calibration["source_worlds"]])):
        raise ValueError("FEATURE_RELEASE_SOURCE_AUDIT_BINDING")
    return receipt


def main(argv=None):
    """Run calibration or held-out validation after explicit acceptance freeze."""
    from .features import load_backend,FeatureUnavailable,validate_feature_controls
    from .quality_protocol import load_mosaics,MOSAIC_SCHEMA
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("mode",choices=["calibrate","validate"])
    for name in ("config","acceptance-config","reference-mosaics","patch-plan","suite-plan","source-audit","cache","out"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--calibration-receipt");parser.add_argument("--exposure-ledger");args=parser.parse_args(argv)
    try:
        config=validate_acceptance_config(_read(args.acceptance_config));feature_config=_read(args.config)
        if digest(feature_config)!=config["feature_config_sha256"]:raise ValueError("FEATURE_RELEASE_CONFIGURATION_CHANGED")
        backend=load_backend(feature_config,Path(args.config).parent)
        if backend.manifest["diagnostic_features"]:raise ValueError("BLOCKED_FEATURE_DIAGNOSTIC_BACKEND")
        # Inspect identity metadata before recording exposure and reading arrays.
        bundle=_read(args.reference_mosaics);_check(bundle,"bundle_sha256",MOSAIC_SCHEMA)
        plan=_read(args.suite_plan);audit=_read(args.source_audit);patch_plan=_read(args.patch_plan);role="metric_calibration" if args.mode=="calibrate" else "metric_validation"
        worlds=_source_bindings(plan,audit,bundle,role,patch_plan)
        if patch_plan["patch_plan_sha256"]!=config["calibration_patch_plan_sha256" if args.mode=="calibrate" else "validation_patch_plan_sha256"]:
            raise ValueError("FEATURE_RELEASE_PREREGISTERED_PATCH_PLAN_CHANGED")
        if audit["manifest_sha256"]!=config["source_manifest_sha256"]:raise ValueError("FEATURE_RELEASE_SOURCE_MANIFEST_CHANGED")
        calibration=None;exposure=None
        if args.mode=="calibrate":
            if bundle["bundle_sha256"]!=config["calibration_reference_bundle_sha256"]:raise ValueError("FEATURE_CALIBRATION_REFERENCE_CHANGED")
        else:
            if not args.calibration_receipt or not args.exposure_ledger:raise ValueError("BLOCKED_FEATURE_CALIBRATION_AND_EXPOSURE_REQUIRED")
            calibration=_validate_calibration(_read(args.calibration_receipt),config)
            if backend.manifest["feature_identity_sha256"]!=calibration["report"]["feature_backend_manifest"]["feature_identity_sha256"]:
                raise ValueError("FEATURE_RELEASE_CALIBRATED_BACKEND_RUNTIME_CHANGED")
            for key in ("world_id","world_pixel_sha256","source_sha256"):
                if {w[key] for w in worlds}&{w[key] for w in calibration["source_worlds"]}:raise ValueError("FEATURE_CALIBRATION_VALIDATION_SOURCE_OVERLAP")
            # Failed inference still leaves this held-out exposure in the ledger.
            exposure=_register_exposure(worlds,config,calibration,bundle,args.exposure_ledger)
        _,patches=load_mosaics(args.reference_mosaics)
        _verify_reference_pixels(patches,plan)
        report=validate_feature_controls(patches,backend,args.cache,**config["statistics"])
        report.pop("validation_sha256");report["reference_bundle_sha256"]=bundle["bundle_sha256"];report=_sealed(report,"validation_sha256")
        result=_acceptance(report,config,role);common={"feature_config_sha256":config["feature_config_sha256"],
            "source_manifest_sha256":config["source_manifest_sha256"],"source_worlds":worlds,"source_audit":audit,
            "source_plan":plan,"patch_plan":patch_plan,"feature_config_path":str(Path(args.config).resolve()),
            "reference_mosaics_path":str(Path(args.reference_mosaics).resolve()),
            "reference_bundle_sha256":bundle["bundle_sha256"],"acceptance_config":config,"acceptance_result":result,"report":report}
        if args.mode=="calibrate":
            receipt=_sealed({**common,"schema":CALIBRATION_SCHEMA,"status":"CONFIG_FROZEN" if result["acceptance_passed"] else "ACCEPTANCE_FAILED",
                             "real_calibration_executed":True,"release_validated":False},"calibration_sha256")
        else:
            passed=result["acceptance_passed"] and not exposure["used_for_new_config"]
            receipt=_sealed({**common,"schema":RELEASE_SCHEMA,"instrument":"quality","role":"metric_validation","source_kind":"canonical_inria",
                             "weights_sha256":backend.manifest["weights_sha256"],"calibration_receipt":calibration,"exposure":exposure,
                             "status":"VALIDATED" if passed else "ACCEPTANCE_FAILED","real_validation_executed":True,"release_validated":bool(passed)},"validation_sha256")
            if passed:validate_feature_release(receipt,config["feature_config_sha256"],config["source_manifest_sha256"])
        _write(args.out,receipt);print(canonical({"status":receipt["status"],"release_validated":receipt["release_validated"],"acceptance_passed":result["acceptance_passed"]}).decode())
        return 0 if result["acceptance_passed"] and (args.mode=="calibrate" or receipt["release_validated"]) else 1
    except FeatureUnavailable as exc:
        print(canonical({"status":"BLOCKED","reason_code":exc.reason_code,"real_validation":"NOT_RUN","release_validated":False}).decode());return 1
    except (ValueError,KeyError,TypeError,OSError,RuntimeError) as exc:
        print(canonical({"status":"BLOCKED" if "BLOCKED" in str(exc) else "FAILED","reason":str(exc),"error_type":type(exc).__name__,"release_validated":False}).decode());return 2


if __name__=="__main__":raise SystemExit(main())
