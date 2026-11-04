"""Consume retained model measurements, never instrument PASS flags as scores.

The suite is independently replayed again. Motion is recalculated from submitted
PNGs; quality mosaics are reconstructed from their frozen first source events and
distribution statistics are recalculated from retained feature arrays. Formal
quality additionally repeats frozen feature inference, including its weights and
preprocessing. Distributions stay at their actual set level, separate from RGB.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import tempfile

import numpy as np

from .geometry import canonical,digest, file_hash
from .replay import load_json


SCHEMA = "rswm-model-metric-consumption-v1"
GROUP = ("model_id", "study", "role", "family", "protocol", "variant", "training_seed", "seed", "analysis_selection")


def _read(path):
    path = _regular(path)
    return load_json(path.read_text(encoding="utf-8"))


def _regular(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("METRIC_RETAINED_PATH_SYMLINK")
    if not path.is_file():
        raise ValueError("METRIC_RETAINED_FILE_MISSING_" + str(path))
    return path


def _finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)
    elif isinstance(value, (float, np.floating)) and not math.isfinite(value):
        raise ValueError("METRIC_NONFINITE_VALUE")


def _sealed(report, key, schema):
    _finite(report)
    if not isinstance(report,dict) or report.get("schema") != schema:
        raise ValueError("MODEL_EVALUATION_REQUIRED_INSTRUMENT_RECEIPT_IS_NOT_MODEL_RESULT")
    if report.get(key) != digest({k: v for k, v in report.items() if k != key}):
        raise ValueError("METRIC_RESULT_INTEGRITY")


def evaluation_binding(plan, suite_root, selection=None, *, results=None):
    """Bind model weights, every frozen run, and the actual selected attempt."""
    from .suite import validate_plan, replay_suite, model_manifest,_world
    validate_plan(plan)
    # A retained plan checksum alone cannot verify weights still exist unchanged.
    for model_id, locked in plan["models"].items():
        if model_manifest(locked["config"]) != locked:
            raise ValueError("METRIC_MODEL_WEIGHTS_CHANGED_" + model_id)
    results = replay_suite(plan, suite_root, selection) if results is None else results
    by_run = {r["run_id"]: r for r in results}
    expected = {r["run_id"]: r for r in plan["expected"]}
    if len(by_run) != len(results) or set(by_run) != set(expected):
        raise ValueError("METRIC_REPLAY_RUN_MATRIX_CHANGED")
    selections = {r["selection"] for r in results}
    if len(selections) != 1:
        raise ValueError("METRIC_MIXED_ATTEMPT_SELECTION")
    worlds={}
    for row in expected.values():
        if row["world_id"] in worlds:continue
        pixels=_world(_regular(row["world_path"]))
        world_file_sha256=file_hash(row["world_path"])
        world_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()
        if world_file_sha256!=row["world_file_sha256"] or world_pixel_sha256!=row["world_pixel_sha256"]:
            raise ValueError("METRIC_RETAINED_SOURCE_WORLD_CHANGED")
        worlds[row["world_id"]]={"path":row["world_path"],"world_file_sha256":world_file_sha256,
                                "world_pixel_sha256":world_pixel_sha256,"city":row["city"],"role":row["role"]}
    runs = []
    for run_id, row in sorted(expected.items()):
        actual = by_run[run_id]
        files={}
        if actual.get("attempt") is not None:
            folder=Path(suite_root)/"runs"/run_id/f"attempt-{actual['attempt']:03d}"/"submission"
            if folder.is_dir():
                for path in sorted(folder.rglob("*")):
                    if path.is_symlink():raise ValueError("METRIC_RETAINED_SUBMISSION_SYMLINK")
                    if path.is_file():files[path.relative_to(folder).as_posix()]=file_hash(_regular(path))
        runs.append({"run_id": run_id, "expected_sha256": digest(row),
                     "attempt": actual.get("attempt"), "replay_head": actual.get("replay_head"),
                     "replay_verified": actual["replay_verified"],
                     "submission_status": actual["submission_status"],
                     "output_slots_sha256": digest(actual.get("output_slots", [])),
                     "retained_submission_files":files,
                     "integrity_errors_sha256": digest(actual.get("integrity_errors", []))})
    body = {"schema": "rswm-model-evaluation-binding-v1", "plan_sha256": plan["plan_sha256"],
            "study": plan["study"], "source_code_sha256": plan["source_code_sha256"],
            "source_manifest_sha256": sorted({r["source_manifest_sha256"] for r in expected.values()}),
            "task_set_sha256": digest({k: digest(v) for k, v in sorted(plan["tasks"].items())}),
            "models_sha256": digest(plan["models"]), "metric_config_sha256": digest(plan["metric_config"]),
            "analysis_selection": selections.pop(), "runs": runs,"retained_worlds":worlds}
    return {**body, "binding_sha256": digest(body)}


def _binding(plan, suite_root, report):
    binding = report.get("evaluation_binding")
    if not isinstance(binding, dict):
        raise ValueError("METRIC_MODEL_RESULT_HAS_NO_SUITE_BINDING")
    actual = evaluation_binding(plan, suite_root, binding.get("analysis_selection"))
    if binding != actual:
        raise ValueError("METRIC_SUITE_MODEL_SEED_ATTEMPT_SOURCE_STUDY_CONFIG_BINDING_CHANGED")
    return actual


def _exports(report, result_path):
    hashes = report.get("export_hashes")
    inventories={"rswm-motion-suite-v1":{"motion_edges.csv","motion_run_coverage.csv"},
                 "rswm-motion-validation-v1":{"pairs.csv","worlds.csv","risk_coverage.csv"},
                 "rswm-motion-calibration-v1":{"candidate-results.json","candidate_table.csv"}}
    if not isinstance(hashes,dict) or set(hashes)!=inventories.get(report.get("schema")):
        raise ValueError("METRIC_RETAINED_EXPORTS_REQUIRED")
    root = Path(result_path).parent.resolve()
    for name, sha in hashes.items():
        target = root / name
        if Path(name).is_absolute() or target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError("METRIC_RETAINED_EXPORT_PATH")
        if file_hash(_regular(target)) != sha:
            raise ValueError("METRIC_RETAINED_EXPORT_CHANGED")


def _motion_sources(manifest,pair_root):
    from .motion_protocol import _source_worlds,_load_world,_pair_images,controlled_pair_images,validate_pair_manifest
    validate_pair_manifest(manifest)
    retained=manifest.get("retained_world_inputs",{})
    if not isinstance(retained,dict) or not retained.get("world_index") or not retained.get("world_root"):
        raise ValueError("MOTION_RELEASE_RETAINED_SOURCE_WORLDS_REQUIRED")
    index=retained["world_index"]
    if not isinstance(index,dict) or index.get("source_manifest_sha256")!=manifest["source_manifest_sha256"]:
        raise ValueError("MOTION_RELEASE_RETAINED_SOURCE_MANIFEST_CHANGED")
    diagnostic=manifest["source_kind"]=="synthetic_fixture"
    sources=_source_worlds(index,manifest["role"],diagnostic)
    reduced=[{k:s.get(k) for k in ("world_id","role","city","world_file_sha256","world_pixel_sha256","source_sha256","source_path")} for s in sources]
    if reduced!=manifest["source_worlds"]:raise ValueError("MOTION_RELEASE_RETAINED_SOURCE_WORLD_BINDING_CHANGED")
    for source in sources:
        world=_load_world(source,retained["world_root"])
        for pair in manifest["pairs"]:
            if pair["source_world_id"]!=source["world_id"]:continue
            expected=controlled_pair_images(world,pair)
            actual=_pair_images(pair,pair_root)
            if any((a is None)!=(e is None) or (a is not None and not np.array_equal(a,e)) for a,e in zip(actual,expected)):
                raise ValueError("MOTION_RELEASE_CONTROLLED_PIXELS_NOT_RETAINED_REAL_SOURCE_CROP")


def _motion_release(receipt, config, source_ids):
    """Re-run retained controlled pairs and calibration, keeping old receipts intact."""
    from .motion_protocol import validate_release, validate_calibration, analyze_pairs
    if not isinstance(receipt,dict):raise ValueError("MOTION_RETAINED_RELEASE_RECEIPT_REQUIRED")
    validate_release(receipt, config)
    if receipt["source_manifest_sha256"] not in source_ids:
        raise ValueError("MOTION_RELEASE_SOURCE_MANIFEST_CHANGED")
    retained = receipt.get("retained_inputs", {})
    if not isinstance(retained,dict):raise ValueError("MOTION_RELEASE_RETAINED_INPUTS_REQUIRED")
    if not retained.get("pair_manifest") or not retained.get("pair_root") or not retained.get("calibration_receipt"):
        raise ValueError("MOTION_RELEASE_RETAINED_PAIRS_AND_CALIBRATION_REQUIRED")
    manifest = retained["pair_manifest"]
    ledger=_read(retained.get("exposure_ledger_path",""))
    snapshot=retained.get("exposure_ledger_snapshot",{})
    if (not isinstance(ledger,dict) or not isinstance(snapshot,dict) or
            not isinstance(ledger.get("exposures"),list) or not isinstance(snapshot.get("exposures"),list)):
        raise ValueError("MOTION_RELEASE_RETAINED_EXPOSURE_LEDGER_CHANGED")
    if (ledger.get("schema")!="rswm-motion-validation-exposure-v1" or
            snapshot.get("schema")!=ledger["schema"] or
            hashlib.sha256(canonical(snapshot)+b"\n").hexdigest()!=receipt["exposure_ledger_sha256_at_execution"] or
            ledger.get("exposures",[])[:len(snapshot.get("exposures",[]))]!=snapshot.get("exposures")):
        raise ValueError("MOTION_RELEASE_RETAINED_EXPOSURE_LEDGER_CHANGED")
    keys=sorted({digest([kind,world[field]]) for world in manifest["source_worlds"]
                 for kind,field in (("world_pixels","world_pixel_sha256"),("original_source","source_sha256"))})
    entry={"pair_manifest_sha256":receipt["pair_manifest_sha256"],"config_sha256":config["config_sha256"],
           "calibration_sha256":receipt["calibration_sha256"],"source_population_keys":keys}
    if entry not in snapshot.get("exposures",[]):raise ValueError("MOTION_RELEASE_RETAINED_EXPOSURE_ENTRY_MISSING")
    if manifest["pair_manifest_sha256"] != receipt["pair_manifest_sha256"]:
        raise ValueError("MOTION_RELEASE_RETAINED_PAIR_IDENTITY")
    _motion_sources(manifest,retained["pair_root"])
    report = analyze_pairs(manifest, retained["pair_root"], config)
    if report != receipt["report"]:
        raise ValueError("MOTION_RELEASE_RETAINED_MEASUREMENTS_CHANGED")
    _exports(receipt, Path(retained["output_root"]) / "validation.json")
    calibration = validate_calibration(retained["calibration_receipt"])
    if calibration["calibration_sha256"] != receipt["calibration_sha256"] or calibration["source_manifest_sha256"] not in source_ids:
        raise ValueError("MOTION_RELEASE_CALIBRATION_IDENTITY_CHANGED")
    cinputs = calibration.get("retained_inputs", {})
    if not isinstance(cinputs,dict):raise ValueError("MOTION_CALIBRATION_RETAINED_INPUTS_REQUIRED")
    if not cinputs.get("pair_manifest") or not cinputs.get("pair_root") or not cinputs.get("candidate_configs"):
        raise ValueError("MOTION_CALIBRATION_RETAINED_INPUTS_REQUIRED")
    _motion_sources(cinputs["pair_manifest"],cinputs["pair_root"])
    reports = [analyze_pairs(cinputs["pair_manifest"], cinputs["pair_root"], c) for c in cinputs["candidate_configs"]]
    retained_reports = _read(Path(cinputs["output_root"]) / "candidate-results.json")
    if retained_reports != {"schema": "rswm-motion-candidate-results-v1", "reports": reports}:
        raise ValueError("MOTION_CALIBRATION_RETAINED_MEASUREMENTS_CHANGED")
    _exports(calibration, Path(cinputs["output_root"]) / "calibration.json")
    eligible = [r for r in reports if r["acceptance_passed"]]
    if not eligible:
        raise ValueError("MOTION_CALIBRATION_REPLAY_HAS_NO_ELIGIBLE_CONFIG")
    chosen = min(eligible, key=lambda r: (-r["equal_world_coverage"], r["equal_world_false_confident_rate"], r["config_sha256"]))
    if chosen["config_sha256"] != config["config_sha256"]:
        raise ValueError("MOTION_CALIBRATION_REPLAY_SELECTED_CONFIG_CHANGED")


def _diagnostic(plan,model_id,role):
    from .registry import capabilities
    return role=="diagnostic_only" or capabilities(plan["models"][model_id]["config"])["purpose"]!="real_generator"


def _motion_rows(report,plan):
    groups = defaultdict(list)
    for edge in report["edges"]:
        groups[tuple(edge.get(k) for k in GROUP)].append(edge)
    rows = []
    for key, edges in sorted(groups.items(), key=lambda x: str(x[0])):
        diagnostic=not report["release_validated"] or _diagnostic(plan,edges[0]["model_id"],edges[0]["role"])
        statuses = Counter(e["status"] for e in edges)
        measured = [e for e in edges if e["status"] == "measured"]
        worlds = sorted({e["world_id"] for e in edges})
        world_rows = []
        for world in worlds:
            subset = [e for e in edges if e["world_id"] == world]
            values = [e["normalized_command_error"] for e in subset if e["status"] == "measured"]
            world_rows.append({"world_id": world, "city": subset[0]["city"], "planned_edges": len(subset),
                               "measured_edges": len(values), "coverage": len(values) / len(subset),
                               "mean_normalized_step_error": float(np.mean(values)) if values else None})
        # Fixed equal-city/equal-world weights do not silently move away from a
        # wholly unmeasured world. Conditional pooled edge error is kept separate.
        cities = defaultdict(list)
        for w in world_rows:
            cities[w["city"]].append(w)
        value = (float(np.mean([np.mean([w["mean_normalized_step_error"] for w in ws]) for ws in cities.values()]))
                 if all(w["mean_normalized_step_error"] is not None for w in world_rows) else None)
        rows.append({**dict(zip(GROUP, key)), "statistical_unit": "unique generated edge; fixed equal city/world conditional error",
                     "scope": "DIAGNOSTIC_ONLY" if diagnostic else "VALIDATED_INSTRUMENT_MODEL_EVALUATION",
                     "release_validated":report["release_validated"],"formal_metric_eligible":not diagnostic,
                     "diagnostic_only":diagnostic,"planned_edges":len(edges),
                     "measured_edges": len(measured), "trusted_coverage": len(measured) / len(edges) if edges else None,
                     "coverage":len(measured)/len(edges) if edges else None,
                     "ambiguous_edges": statuses["ambiguous"], "insufficient_evidence_edges": statuses["insufficient_evidence"],
                     "generation_failure_edges": statuses["submission_failure"], "status_counts": dict(statuses),
                     "mean_normalized_step_error": value,
                     "conditional_pooled_edge_error": float(np.mean([e["normalized_command_error"] for e in measured])) if measured else None,
                     "worlds": world_rows, "missing_worlds": [w["world_id"] for w in world_rows if not w["measured_edges"]],
                     "task_ids": sorted({e["task_id"] for e in edges}), "task_set_sha256": digest(sorted({e["task_id"] for e in edges})),
                     "metric_config_sha256": report["config"]["config_sha256"], "model_manifest_sha256": edges[0]["model_manifest_sha256"],
                     "source_manifest_sha256":edges[0]["source_manifest_sha256"],
                     "source_code_sha256":report["evaluation_binding"]["source_code_sha256"]})
    return rows


def consume_motion(plan, suite_root, path):
    from .motion_protocol import measure_suite
    report = _read(path)
    _sealed(report, "motion_suite_sha256", "rswm-motion-suite-v1")
    binding = _binding(plan, suite_root, report)
    _exports(report, path)
    diagnostic = report.get("release_validated") is not True
    if report.get("status") != ("DIAGNOSTIC_ONLY" if diagnostic else "VALIDATED_INSTRUMENT"):
        raise ValueError("MOTION_MODEL_RESULT_RELEASE_STATUS")
    retained=report.get("retained_inputs",{})
    if not isinstance(retained,dict):raise ValueError("MOTION_MODEL_RESULT_RETAINED_INPUTS_REQUIRED")
    release=retained.get("validation_receipt")
    if not diagnostic:
        _motion_release(release, report["config"], binding["source_manifest_sha256"])
    rebuilt = measure_suite(plan, suite_root, report["config"], release, diagnostic, binding["analysis_selection"])
    if rebuilt != {k: v for k, v in report.items() if k not in {"motion_suite_sha256", "retained_inputs", "export_hashes"}}:
        raise ValueError("MOTION_MODEL_RESULT_REPLAY_MISMATCH")
    return _motion_rows(report,plan), {"kind": "motion", "path": str(Path(path).resolve()), "sha256": file_hash(path),
                                  "evaluation_sha256": report["motion_suite_sha256"], "binding": binding,
                                  "retained_png_measurement_replayed": True, "instrument_release_validated": not diagnostic,
                                  "metric_configuration":report["config"],"instrument_release_receipt":release}


def _mosaic_equal(actual, reconstructed):
    if len(actual) != len(reconstructed):
        raise ValueError("QUALITY_PLANNED_PATCH_MATRIX_CHANGED")
    for a, b in zip(actual, reconstructed):
        for key in set(a) | set(b):
            if key in {"data_file", "data_file_sha256"}:
                continue
            av, bv = a.get(key), b.get(key)
            if isinstance(av, np.ndarray) or isinstance(bv, np.ndarray):
                if not isinstance(av, np.ndarray) or not isinstance(bv, np.ndarray) or av.dtype != bv.dtype or not np.array_equal(av, bv):
                    raise ValueError("QUALITY_RETAINED_MOSAIC_PNG_OR_REFERENCE_MISMATCH")
            elif av != bv:
                raise ValueError("QUALITY_RETAINED_MOSAIC_BINDING_CHANGED_" + key)


def _feature_mosaic_equal(features, patches):
    if len(features) != len(patches):
        raise ValueError("QUALITY_FEATURE_PATCH_MATRIX_CHANGED")
    for feature, patch in zip(features, patches):
        for k, value in patch.items():
            if isinstance(value, np.ndarray) or k=="diagnostic_features":
                continue
            if feature.get(k) != value:
                raise ValueError("QUALITY_FEATURE_MOSAIC_IDENTITY_CHANGED_" + k)


def _feature_caches(bundle_path,bundle,features,patches):
    from types import SimpleNamespace
    from .features import feature_cache_key
    manifest=bundle["backend_manifest"]
    backend=SimpleNamespace(manifest=manifest)
    artifacts={}
    for row,patch in zip(features,patches):
        key=feature_cache_key(patch,backend)
        if row.get("feature_cache_key")!=key:raise ValueError("QUALITY_RETAINED_CACHE_SOURCE_CONFIG_KEY_CHANGED")
        if row["feature"] is None:continue
        if not row.get("feature_cache_file") or not row.get("feature_cache_metadata_file"):
            raise ValueError("QUALITY_ORIGINAL_FEATURE_CACHE_RETENTION_REQUIRED")
        data=_regular(Path(bundle_path).parent/row["feature_cache_file"])
        meta_path=_regular(Path(bundle_path).parent/row["feature_cache_metadata_file"])
        info=_read(meta_path);sha=file_hash(data)
        expected={"schema":"rswm-feature-cache-entry-v1","cache_key":key,"feature_sha256":sha,
                  "feature_identity_sha256":manifest["feature_identity_sha256"],"feature_dimension":row["feature_dimension"],
                  "diagnostic_features":manifest["diagnostic_features"]}
        cached=np.load(data,allow_pickle=False)
        if info!=expected or cached.dtype!=np.float32 or not np.array_equal(cached,row["feature"]):
            raise ValueError("QUALITY_ORIGINAL_FEATURE_CACHE_INTEGRITY")
        artifacts[str(data.resolve())]=sha;artifacts[str(meta_path.resolve())]=file_hash(meta_path)
    return artifacts


def quality_suite_binding(plan, suite_root, generated_mosaics, reference_mosaics, patch_plan, selection=None):
    """Independently authenticate frozen geometry, PNGs, and real reference crops."""
    from .quality_protocol import (load_mosaics, validate_patch_plan, _validate_suite_binding,
                                   assemble_suite_mosaics, assemble_reference_mosaics)
    patch_plan = _read(patch_plan) if not isinstance(patch_plan, dict) else patch_plan
    validate_patch_plan(patch_plan)
    _validate_suite_binding(plan, patch_plan)
    generated_bundle, generated = load_mosaics(_regular(generated_mosaics))
    reference_bundle, reference = load_mosaics(_regular(reference_mosaics))
    for bundle in (generated_bundle, reference_bundle):
        if bundle["suite_plan_sha256"] != plan["plan_sha256"] or bundle["patch_plan_sha256"] != patch_plan["patch_plan_sha256"]:
            raise ValueError("QUALITY_MOSAIC_SUITE_PATCH_BINDING_CHANGED")
    with tempfile.TemporaryDirectory(prefix="rswm-metric-replay-") as temp:
        assemble_suite_mosaics(plan, suite_root, patch_plan, Path(temp) / "generated", selection)
        assemble_reference_mosaics(plan, patch_plan, Path(temp) / "reference")
        _, expected_generated = load_mosaics(Path(temp) / "generated" / "mosaics.json")
        _, expected_reference = load_mosaics(Path(temp) / "reference" / "mosaics.json")
        _mosaic_equal(generated, expected_generated)
        _mosaic_equal(reference, expected_reference)
    return generated_bundle, generated, reference_bundle, reference, patch_plan


def _quality_rows(report, generated, plan, release):
    expected = {r["run_id"]: r for r in plan["expected"]}
    groups = defaultdict(list)
    group_fields=("study","role","family","protocol","variant","lineage","model_id","training_seed","seed","analysis_selection")
    for record in generated:
        row = expected[record["run_id"]]
        key = tuple(record.get(k, "unspecified") for k in group_fields)
        groups[key].append((record, row))
    rows = []
    for entry in report["reports"]:
        key = tuple(entry.get(k, "unspecified") for k in group_fields)
        selected = groups.pop(key, None)
        if selected is None:
            raise ValueError("QUALITY_UNEXPECTED_OR_DUPLICATE_DISTRIBUTION_GROUP")
        stat = entry["own_complete_sample_diagnostic"]
        cov = stat["generated_coverage"]
        missing = [k for k, v in cov["source_worlds"].items() if v["complete_tiles"] == 0]
        model_manifests = sorted({r["model_manifest_sha256"] for _, r in selected})
        if len(model_manifests) != 1:
            raise ValueError("QUALITY_DISTRIBUTION_MIXED_MODEL_WEIGHTS")
        task_ids = sorted({p["task_id"] for p, _ in selected})
        diagnostic=not release or _diagnostic(plan,entry["model_id"],entry["role"])
        rows.append({**{k: entry[k] for k in ("study", "role", "family", "protocol", "variant", "lineage", "model_id")},
                     "statistical_unit":"frozen patch population; world balanced distribution per training/inference seed",
                     "training_seed":entry["training_seed"],"seed":entry["seed"],
                     "analysis_selection": report["evaluation_binding"]["analysis_selection"],
                     "training_seeds": sorted({r["training_seed"] for _, r in selected}, key=str),
                     "inference_seeds": sorted({r["seed"] for _, r in selected}),
                     "task_ids": task_ids, "task_set_sha256": digest(task_ids),
                     "sample_set_sha256": digest(sorted(p["sample_id"] for p, _ in selected)),
                     "model_manifest_sha256": model_manifests[0],
                     "source_manifest_sha256":selected[0][1]["source_manifest_sha256"],
                     "source_code_sha256":report["evaluation_binding"]["source_code_sha256"],
                     "scope":"DIAGNOSTIC_ONLY" if diagnostic else "VALIDATED_INSTRUMENT_MODEL_EVALUATION",
                     "release_validated":bool(release),"formal_metric_eligible":not diagnostic,"diagnostic_only":bool(diagnostic),
                     "polynomial_mmd_u": stat["polynomial_mmd_u"] if not missing else None,
                     "conditional_complete_sample_mmd_u": stat["polynomial_mmd_u"],
                     "missing_source_worlds": missing, "planned_tiles": cov["planned_tiles"], "complete_tiles": cov["complete_tiles"],
                     "planned_pixels": cov["planned_pixels"], "missing_pixels": cov["missing_pixels"],
                     "tile_coverage": cov["tile_coverage"], "pixel_coverage": cov["pixel_coverage"],
                     "generated_coverage": cov, "real_reference": stat["reference_coverage"],
                     "real_real_controls": [r for r in release["report"]["reports"] if r["control_name"] == "independent_real_real"] if release else [],
                     "feature_identity_sha256": report["feature_identity_sha256"],
                     "statistical_config": report["statistical_config"], "statistical_config_sha256": report["statistical_config_sha256"],
                     "common_planned_subset_sensitivity": entry["common_planned_subset_sensitivity"],
                     "reason_codes": ["MISSING_SOURCE_WORLD_NO_WEIGHT_REALLOCATION"] if missing else []})
    if groups:
        raise ValueError("QUALITY_DISTRIBUTION_GROUP_MISSING")
    return rows


def _recompute_feature_release(release,backend):
    """Check measured control values against retained pixels, without exposure writes."""
    from .quality_protocol import load_mosaics
    from .features import validate_feature_controls
    statistics=release["acceptance_config"]["statistics"]
    with tempfile.TemporaryDirectory(prefix="rswm-feature-release-replay-") as temp:
        for role,receipt in (("calibration",release["calibration_receipt"]),("validation",release)):
            bundle,patches=load_mosaics(_regular(receipt["reference_mosaics_path"]))
            rebuilt=validate_feature_controls(patches,backend,Path(temp)/role,**statistics)
            rebuilt.pop("validation_sha256")
            rebuilt["reference_bundle_sha256"]=bundle["bundle_sha256"]
            rebuilt["validation_sha256"]=digest(rebuilt)
            if rebuilt!=receipt["report"]:
                raise ValueError("QUALITY_RELEASE_RETAINED_CONTROL_INFERENCE_MISMATCH_"+role)


def _feature_exposure(release):
    exposure=release["exposure"]
    if not isinstance(exposure,dict) or not exposure.get("ledger_path") or not isinstance(exposure.get("ledger_snapshot"),dict):
        raise ValueError("QUALITY_RELEASE_RETAINED_EXPOSURE_LEDGER_REQUIRED")
    ledger=_read(exposure["ledger_path"]);snapshot=exposure["ledger_snapshot"]
    if (not isinstance(ledger,dict) or not isinstance(ledger.get("exposures"),list) or not isinstance(snapshot.get("exposures"),list) or
        ledger.get("schema")!="rswm-feature-validation-exposure-v1" or
        snapshot.get("schema")!=ledger["schema"] or
        hashlib.sha256(canonical(snapshot)+b"\n").hexdigest()!=exposure["ledger_sha256_at_execution"] or
        ledger.get("exposures",[])[:len(snapshot.get("exposures",[]))]!=snapshot.get("exposures")):
        raise ValueError("QUALITY_RELEASE_RETAINED_EXPOSURE_LEDGER_CHANGED")
    config=release["acceptance_config"]
    expected={"reference_bundle_sha256":release["reference_bundle_sha256"],"feature_config_sha256":config["feature_config_sha256"],
              "acceptance_sha256":config["acceptance_sha256"],"calibration_sha256":release["calibration_receipt"]["calibration_sha256"],
              "source_population_keys":exposure["source_population_keys"]}
    if expected not in snapshot.get("exposures",[]):raise ValueError("QUALITY_RELEASE_RETAINED_EXPOSURE_ENTRY_MISSING")
    return {"path":exposure["ledger_path"],"sha256":file_hash(exposure["ledger_path"]),
            "sha256_at_execution":exposure["ledger_sha256_at_execution"]}


def _bundle_provenance(path,bundle,kind):
    root=Path(path).parent
    files={str(Path(path).resolve()):file_hash(path)}
    for row in bundle["features" if kind=="feature" else "patches"]:
        name=row.get("feature_file" if kind=="feature" else "data_file")
        if name:files[str((root/name).resolve())]=file_hash(_regular(root/name))
    return {"path":str(Path(path).resolve()),"identity_sha256":bundle["feature_bundle_sha256" if kind=="feature" else "bundle_sha256"],
            "planned_tiles":bundle["planned_tiles"],"complete_tiles":bundle["complete_tiles"],"retained_files":files}


def consume_quality(plan, suite_root, path):
    from .features import load_feature_bundle, load_backend, validate_feature_release
    from .quality_evaluation import evaluate_methods
    report = _read(path)
    _sealed(report, "evaluation_sha256", "rswm-quality-evaluation-v1")
    binding = _binding(plan, suite_root, report)
    retained = report.get("retained_inputs", {})
    if not isinstance(retained,dict):raise ValueError("QUALITY_MODEL_RESULT_RETAINED_INPUTS_REQUIRED")
    required = ("generated_features", "reference_features", "generated_mosaics", "reference_mosaics", "patch_plan")
    if any(not retained.get(k) for k in required):
        raise ValueError("QUALITY_MODEL_RESULT_RETAINED_INPUTS_REQUIRED")
    xb, x = load_feature_bundle(_regular(retained["generated_features"]))
    yb, y = load_feature_bundle(_regular(retained["reference_features"]))
    xm, xp, ym, yp, pp = quality_suite_binding(plan, suite_root, retained["generated_mosaics"], retained["reference_mosaics"],
                                              retained["patch_plan"], binding["analysis_selection"])
    _feature_mosaic_equal(x, xp)
    _feature_mosaic_equal(y, yp)
    caches={**_feature_caches(retained["generated_features"],xb,x,xp),**_feature_caches(retained["reference_features"],yb,y,yp)}
    if (xb["mosaic_bundle_sha256"] != xm["bundle_sha256"] or yb["mosaic_bundle_sha256"] != ym["bundle_sha256"] or
            report["generated_bundle_sha256"] != xb["feature_bundle_sha256"] or report["reference_bundle_sha256"] != yb["feature_bundle_sha256"] or
            xb["backend_manifest"] != yb["backend_manifest"] or report["feature_identity_sha256"] != xb["backend_manifest"]["feature_identity_sha256"] or
            report.get("patch_plan_sha256") != pp["patch_plan_sha256"]):
        raise ValueError("QUALITY_FEATURE_PATCH_CONFIGURATION_BINDING_CHANGED")
    config = report["statistical_config"]
    if (not isinstance(config,dict) or set(config)!={"schema","kind","cap","bootstrap","seed","kernel","population_grouping"} or
            config.get("schema")!="rswm-quality-statistical-protocol-v1" or config.get("kernel")!="(x.y/d+1)^3 unbiased U" or
            config.get("population_grouping")!="training_seed,inference_seed,attempt_selection" or
            digest(config)!=report["statistical_config_sha256"] or config.get("kind")!="generated-real"):
        raise ValueError("QUALITY_MODEL_STATISTICAL_CONFIG_CHANGED")
    rebuilt = evaluate_methods(x,y,split_seeds=True,kind=config["kind"],cap=config["cap"],bootstrap=config["bootstrap"],seed=config["seed"])
    if rebuilt != report["reports"]:
        raise ValueError("QUALITY_MODEL_RESULT_RECOMPUTATION_MISMATCH")
    release = None
    if report.get("release_validated") is True:
        if xb["diagnostic_features"] or yb["diagnostic_features"]:
            raise ValueError("QUALITY_DIAGNOSTIC_FEATURES_CANNOT_AUTHORIZE_FORMAL_METRICS")
        receipt = retained.get("validation_receipt")
        if receipt is None:
            raise ValueError("QUALITY_RETAINED_RELEASE_RECEIPT_REQUIRED")
        release = validate_feature_release(receipt, report["preregistered_feature_config_sha256"], report["source_manifest_sha256"])
        _feature_exposure(release)
        if (release["source_manifest_sha256"] not in binding["source_manifest_sha256"] or
                release["validation_sha256"] != report["instrument_release_validation_sha256"] or
                release["report"]["feature_backend_manifest"] != xb["backend_manifest"]):
            raise ValueError("QUALITY_RELEASE_SOURCE_OR_FEATURE_BACKEND_CHANGED")
        for row in [*x,*y]:
            world=release["source_audit"]["worlds"].get(row["world_id"],{})
            if (row.get("role")!=world.get("role") or row.get("city")!=world.get("city") or
                    row.get("source_identity")!="world-pixels-sha256:"+world.get("world_pixel_sha256","")):
                raise ValueError("QUALITY_FORMAL_FEATURE_SOURCE_MANIFEST_BINDING")
        frozen = release["acceptance_config"]
        if (config["cap"] != frozen["primary_sample_cap"] or config["bootstrap"] != frozen["statistics"]["bootstrap"] or config["seed"] != frozen["statistics"]["seed"]):
            raise ValueError("QUALITY_FORMAL_STATISTICAL_PROTOCOL_CHANGED")
        backend = load_backend(xb["backend_manifest"]["config"])
        if backend.manifest != xb["backend_manifest"]:
            raise ValueError("QUALITY_RETAINED_FEATURE_RUNTIME_CHANGED")
        _recompute_feature_release(release,backend)
        for features, patches in ((x, xp), (y, yp)):
            valid = [(f, p) for f, p in zip(features, patches) if f["feature"] is not None]
            values = backend.encode([p["image"] for _, p in valid])
            if any(not np.array_equal(value, f["feature"]) for value, (f, _) in zip(values, valid)):
                raise ValueError("QUALITY_RETAINED_FEATURE_INFERENCE_MISMATCH")
    elif report.get("diagnostic_features") is not (xb["diagnostic_features"] or yb["diagnostic_features"]):
        raise ValueError("QUALITY_DIAGNOSTIC_CLASSIFICATION_CHANGED")
    return _quality_rows(report, x, plan, release), {"kind": "quality", "path": str(Path(path).resolve()), "sha256": file_hash(path),
        "evaluation_sha256": report["evaluation_sha256"], "binding": binding, "patch_plan_sha256": pp["patch_plan_sha256"],
        "retained_mosaics_replayed": True, "distribution_recomputed": True, "feature_inference_verified": bool(release),
        "instrument_release_validated": bool(release),"feature_backend_manifest":xb["backend_manifest"],
        "generated_feature_bundle":_bundle_provenance(retained["generated_features"],xb,"feature"),
        "reference_feature_bundle":_bundle_provenance(retained["reference_features"],yb,"feature"),
        "generated_mosaic_bundle":_bundle_provenance(retained["generated_mosaics"],xm,"mosaic"),
        "reference_mosaic_bundle":_bundle_provenance(retained["reference_mosaics"],ym,"mosaic"),
        "retained_feature_caches":caches,"statistical_config":config,
        "instrument_release_receipt":None if release is None else {
            "validation_sha256":release["validation_sha256"],"calibration_sha256":release["calibration_receipt"]["calibration_sha256"],
            "feature_config_path":release["feature_config_path"],"feature_config_file_sha256":file_hash(release["feature_config_path"]),
            "source_manifest_sha256":release["source_manifest_sha256"],"reference_mosaics_path":release["reference_mosaics_path"],
            "exposure_ledger":_feature_exposure(release)}}


def consume_metric_results(plan, suite_root, *, motion_results=(), quality_results=()):
    """Reject invalid supplied results; return explicit absence for missing inputs."""
    from .suite import suite_read_lock, validate_plan
    validate_plan(plan)
    output = {"schema": SCHEMA, "motion_rows": [], "quality_rows": [], "provenance": [], "availability": {}}
    paths_seen = set()
    rows_seen = set()
    configurations={}
    with suite_read_lock(suite_root):
        for kind, paths, consume in (("motion", motion_results, consume_motion), ("quality", quality_results, consume_quality)):
            if isinstance(paths, (str, Path)):
                paths = [paths]
            for path in paths:
                resolved = str(Path(path).resolve())
                if resolved in paths_seen:
                    raise ValueError("METRIC_DUPLICATE_RESULT_INPUT")
                paths_seen.add(resolved)
                rows, provenance = consume(plan, suite_root, path)
                for row in rows:
                    configuration=(row.get("metric_config_sha256"),row.get("feature_identity_sha256"),row.get("statistical_config_sha256"))
                    if kind in configurations and configurations[kind]!=configuration:
                        raise ValueError("METRIC_RESULT_CONFIGURATIONS_DIFFER")
                    configurations[kind]=configuration
                    identity = (kind, tuple(row.get(k) for k in GROUP), row.get("lineage"),
                                tuple(row.get("inference_seeds", [])), row["task_set_sha256"])
                    if identity in rows_seen:
                        raise ValueError("METRIC_DUPLICATE_RESULT_GROUP")
                    rows_seen.add(identity)
                output[kind + "_rows"].extend(rows)
                output["provenance"].append(provenance)
            rows = output[kind + "_rows"]
            formal=any(r["formal_metric_eligible"] for r in rows)
            diagnostic=any(r["diagnostic_only"] for r in rows)
            output["availability"][kind]={"status":"AVAILABLE" if formal else "AVAILABLE_DIAGNOSTIC_ONLY" if rows else "NOT_AVAILABLE",
                "formal_status":"AVAILABLE" if formal else "NOT_AVAILABLE","diagnostic_available":diagnostic,
                "reason_codes":[] if formal else ["DIAGNOSTIC_MODEL_EVALUATION_ONLY"] if rows else ["NO_RETAINED_MODEL_EVALUATION_PROVIDED"],
                "formal_metric_eligible":formal}
    _finite(output)
    output["consumption_sha256"] = digest(output)
    return output
