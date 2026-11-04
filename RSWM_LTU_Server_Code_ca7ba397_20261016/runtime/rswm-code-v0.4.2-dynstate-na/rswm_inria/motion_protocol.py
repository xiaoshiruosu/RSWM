"""Versioned motion calibration and validation; image-only instrument inputs.

Synthetic fixtures can exercise this workflow but never issue a release-valid
calibration or validation receipt. Commands and reference displacements are used
by the outer evaluator only after the image-only estimate has returned.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import os
import stat
import uuid
import numpy as np
from PIL import Image
from .geometry import canonical, crop, digest, file_hash, WIDTH, HEIGHT, WORLD, DIAGONAL
from .motion import MotionConfig, estimate_translation
from .replay import load_json, load_rgb_png

PROTOCOL_SCHEMA = "rswm-motion-protocol-v1"
PAIR_SCHEMA = "rswm-motion-pairs-v1"
CALIBRATION_SCHEMA = "rswm-motion-calibration-v1"
VALIDATION_SCHEMA = "rswm-motion-validation-v1"
CONFIG_SCHEMA = "rswm-motion-config-v1"
ROLES = {"metric_calibration": "bellingham", "metric_validation": "bloomington"}
CONTROLS = ("ordinary", "reverse", "half", "double", "boundary_inside", "boundary",
            "outside_domain", "stationary", "periodic", "low_contrast", "missing_current",
            "deformation", "color_shift")
PARAMETERS = set(asdict(MotionConfig()))
DEFAULT_ACCEPTANCE = {"pixel_tolerance": 1.0, "max_false_confident_rate": .01,
                      "min_measurable_coverage": .5}


def _finite(value):
    if isinstance(value, dict):
        for item in value.values(): _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: _finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite protocol number")


def _read(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Protocol symlinks are not accepted")
    value = load_json(path.read_text(encoding="utf-8"))
    _finite(value)
    return value


def _write(path, value):
    """Immutable complete JSON; different existing evidence is never overwritten."""
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Protocol symlinks are not accepted")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical(value) + b"\n"
    if path.exists():
        if path.read_bytes() != text: raise FileExistsError(path)
        return
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("xb") as output:
        output.write(text); output.flush(); os.fsync(output.fileno())
    # Hard-link creation protects the immutable destination from racing writers.
    try: os.link(temp, path)
    finally: temp.unlink(missing_ok=True)


def _sealed(body, key):
    if key in body: raise ValueError("Hash key is reserved")
    return {**body, key: digest(body)}


def _csv(out, name, rows):
    """Nullable scientific values stay empty; nested diagnostics retain JSON."""
    import io
    path = Path(out)/name
    keys = sorted({key for row in rows for key in row})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output,fieldnames=keys,lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key:json.dumps(value,sort_keys=True,allow_nan=False) if isinstance(value,(dict,list)) else value for key,value in row.items()})
    data = output.getvalue().encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents): raise ValueError("CSV symlink")
    if path.exists() and path.read_bytes() != data: raise FileExistsError(path)
    if not path.exists():
        temp=path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
        with temp.open("xb") as handle:handle.write(data);handle.flush();os.fsync(handle.fileno())
        try:os.link(temp,path)
        finally:temp.unlink(missing_ok=True)
    return file_hash(path)


def _verify_hash(value, key, schema):
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError("Motion schema mismatch")
    if value.get(key) != digest({k: v for k, v in value.items() if k != key}):
        raise ValueError("Motion artifact hash mismatch")


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _contained(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Motion image paths must be relative")
    candidate = root / relative
    if candidate.is_symlink() or any(p.is_symlink() for p in candidate.parents):
        raise ValueError("Motion image symlink")
    path = candidate.resolve()
    if root not in path.parents: raise ValueError("Motion image path escapes manifest directory")
    return path


def motion_config(parameters=None, acceptance=None, boundary_abstain=True):
    parameters = dict(asdict(MotionConfig()) if parameters is None else parameters)
    if set(parameters) != PARAMETERS: raise ValueError("Motion estimator parameter keys mismatch")
    for key, value in parameters.items():
        if key == "peak_exclusion_radius":
            if type(value) is not int: raise ValueError("Peak exclusion radius must be an integer")
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Motion thresholds must be finite numbers")
    MotionConfig(**parameters)
    acceptance = dict(DEFAULT_ACCEPTANCE if acceptance is None else acceptance)
    if set(acceptance) != set(DEFAULT_ACCEPTANCE): raise ValueError("Acceptance keys mismatch")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in acceptance.values()):
        raise ValueError("Acceptance thresholds must be finite")
    if acceptance["pixel_tolerance"] < 0 or not 0 <= acceptance["max_false_confident_rate"] <= 1 or not 0 <= acceptance["min_measurable_coverage"] <= 1:
        raise ValueError("Invalid preregistered acceptance thresholds")
    if type(boundary_abstain) is not bool: raise ValueError("boundary_abstain must be Boolean")
    return _sealed({"schema": CONFIG_SCHEMA, "protocol": PROTOCOL_SCHEMA,
                    "instrument": "linear_zncc_v1", "grid_hw": [HEIGHT, WIDTH],
                    "normalization": "submitted canonical RGB PNG; no per-model resize",
                    "parameters": parameters, "boundary_abstain": boundary_abstain,
                    "acceptance": acceptance,
                    "sign_convention": "viewport translation is negative image-content translation"}, "config_sha256")


def validate_config(config):
    _verify_hash(config, "config_sha256", CONFIG_SCHEMA)
    rebuilt = motion_config(config.get("parameters"), config.get("acceptance"), config.get("boundary_abstain"))
    if rebuilt != config: raise ValueError("Unknown or altered motion configuration fields")
    return config


def search_domain(config):
    validate_config(config)
    overlap = config["parameters"]["min_overlap"]
    # Match the estimator's finite integer predicate, including float arithmetic.
    dx_bound = max(dx for dx in range(WIDTH) if _in_domain((dx, 0), config))
    dy_bound = max(dy for dy in range(HEIGHT) if _in_domain((0, dy), config))
    return {"grid_hw": [HEIGHT, WIDTH], "minimum_overlap_fraction": overlap,
            "dx_bounds": [-dx_bound, dx_bound],
            "dy_bounds": [-dy_bound, dy_bound],
            "predicate": "(width-abs(dx))*(height-abs(dy)) >= minimum_overlap_fraction*width*height"}


def _in_domain(delta, config):
    dx, dy = delta
    return (WIDTH-abs(dx))*(HEIGHT-abs(dy)) >= config["parameters"]["min_overlap"]*WIDTH*HEIGHT and abs(dx) < WIDTH and abs(dy) < HEIGHT


def estimate_pair(previous, current, config):
    """Only two images and a fixed config enter the estimator, never commands."""
    validate_config(config)
    base = {"status": "submission_failure", "content_translation_xy": None,
            "viewport_translation_xy": None, "confidence": None,
            "correlation": None, "second_peak": None, "peak_gap": None,
            "configured_search_domain": search_domain(config), "boundary_hit": False,
            "config_sha256": config["config_sha256"], "instrument": config["instrument"]}
    if previous is None or current is None: return base
    for image in (previous, current):
        if not isinstance(image, np.ndarray) or image.shape != (HEIGHT, WIDTH, 3) or image.dtype != np.uint8:
            return {**base, "status": "out_of_domain", "input_error": "NONCANONICAL_GRID_OR_RGB"}
    result = estimate_translation(previous, current, MotionConfig(**config["parameters"]))
    evidence = {**base, "status": result["status"], **{key: result.get(key) for key in ("correlation", "second_peak", "peak_gap")}}
    peak = result.get("correlation")
    evidence["confidence"] = peak if peak is not None else None
    if result["status"] != "measured": return evidence
    dx, dy = result["delta"]
    boundary = any(not _in_domain([dx+sx, dy+sy], config) for sx, sy in
                   ((1 if dx >= 0 else -1, 0), (0, 1 if dy >= 0 else -1)))
    evidence["boundary_hit"] = boundary
    if boundary and config["boundary_abstain"]:
        return {**evidence, "status": "out_of_domain", "candidate_viewport_translation_xy": [dx, dy]}
    return {**evidence, "content_translation_xy": [-dx, -dy], "viewport_translation_xy": [dx, dy]}


def _source_worlds(index, role, diagnostic_fixture):
    if role not in ROLES: raise ValueError("Motion role must be calibration or validation")
    if not isinstance(index, dict) or not _sha(index.get("source_manifest_sha256")):
        raise ValueError("Frozen source manifest identity is required")
    worlds = [dict(w) for w in index.get("worlds", []) if w.get("role") == role]
    if not worlds or (not diagnostic_fixture and len(worlds) != 12):
        raise ValueError("Real motion profile requires all twelve role worlds")
    if len({w.get("world_id") for w in worlds}) != len(worlds): raise ValueError("Duplicate motion world")
    for world in worlds:
        if not world.get("world_id") or world.get("city") != ROLES[role]:
            raise ValueError("Frozen motion city/role mismatch")
        if not _sha(world.get("world_file_sha256")) or not _sha(world.get("world_pixel_sha256")):
            raise ValueError("World file and pixel identity are required")
        if not diagnostic_fixture and (not _sha(world.get("source_sha256")) or
            not isinstance(world.get("source_path"), str) or world.get("source_crop_xyxy") != [452,452,4548,4548] or
            world.get("resampling") != "uint16 sum 2x2; (sum+2)//4" or world.get("shape") != [WORLD,WORLD,3]):
            raise ValueError("Real canonical Inria source conversion evidence is required")
    return sorted(worlds, key=lambda w: w["world_id"])


def _load_world(row, root):
    path = _contained(root, row.get("path"))
    if file_hash(path) != row["world_file_sha256"]: raise ValueError("Motion world file hash mismatch")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB": raise ValueError("Canonical world requires RGB PNG")
        pixels = np.asarray(image).copy()
    if pixels.shape != (WORLD,WORLD,3) or pixels.dtype != np.uint8: raise ValueError("Canonical world shape/dtype mismatch")
    if hashlib.sha256(pixels.tobytes()).hexdigest() != row["world_pixel_sha256"]:
        raise ValueError("Motion world pixel hash mismatch")
    return pixels


def controlled_pair_images(world,planned):
    """The frozen controlled operations, reused when retained pixels are read back."""
    before=crop(world,[0,0],planned["center_xy"]).copy()
    after=crop(world,planned["reference_displacement"],planned["center_xy"]).copy()
    control=planned["control"]
    if control=="periodic":
        before=np.tile(before[:8,:8],(HEIGHT//8,WIDTH//8,1))
        dx,dy=planned["reference_displacement"];after=np.roll(before,(-dy,-dx),axis=(0,1))
    elif control=="low_contrast":
        before=np.full_like(before,int(round(float(before.mean()))))
        after=np.full_like(after,int(round(float(after.mean()))))
    elif control=="deformation":
        for y in range(HEIGHT):after[y]=np.roll(after[y],int(round(8*math.sin(y/16))),axis=0)
    elif control=="color_shift":
        after=np.clip(after.astype(np.int16)+np.array([16,-12,8]),0,255).astype(np.uint8)
    return before,None if control=="missing_current" else after


def build_pairs(index, world_root, out, role, pairs_per_world=256, seed=0, diagnostic_fixture=False):
    """Freeze legal centers/displacements before reading or perturbing image pairs."""
    if type(pairs_per_world) is not int or not 1 <= pairs_per_world <= 4096 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid motion pair budget/seed")
    if type(diagnostic_fixture) is not bool: raise ValueError("diagnostic_fixture must be explicit Boolean")
    worlds = _source_worlds(index, role, diagnostic_fixture)
    root = Path(out)
    if root.exists() and any(root.iterdir()): raise FileExistsError("Pair destination must be empty")
    root.mkdir(parents=True, exist_ok=True)
    sampling = {"schema": "rswm-motion-controlled-operations-v1", "pairs_per_world": pairs_per_world,
                "seed": seed, "directions": [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]],
                "magnitudes": [8,16,24,32,48,64], "controls": list(CONTROLS),
                "canonical_crop_hw": [HEIGHT,WIDTH], "real_source": not diagnostic_fixture,
                "nonrigid_interpretation": "errors relative to pre-perturbation controlled viewport command; not a physical scene-motion claim"}
    frozen_plan = []
    for row in worlds:
        rng = np.random.default_rng(int(digest([row["world_id"], seed, sampling])[:16], 16))
        for i in range(pairs_per_world):
            control = CONTROLS[i % len(CONTROLS)]
            direction = sampling["directions"][(i // len(CONTROLS)) % 8]
            magnitude = sampling["magnitudes"][(i // len(CONTROLS)) % len(sampling["magnitudes"])]
            if control == "reverse": direction = [-x for x in direction]
            if control == "half": magnitude //= 2
            if control == "double": magnitude *= 2
            if control in {"boundary_inside", "boundary", "outside_domain"}:
                # Perturbations probe the preregistered default search domain.
                direction = [1,0]
                magnitude = math.floor(WIDTH*(1-MotionConfig().min_overlap)) + {"boundary_inside":-1,"boundary":0,"outside_domain":1}[control]
            delta = [d*magnitude for d in direction] if control != "stationary" else [0,0]
            dx, dy = delta
            xmin, xmax = WIDTH//2 + max(0,-dx), WORLD-WIDTH//2 - max(0,dx)
            ymin, ymax = HEIGHT//2 + max(0,-dy), WORLD-HEIGHT//2 - max(0,dy)
            center = [int(rng.integers(xmin, xmax+1)), int(rng.integers(ymin, ymax+1))]
            pair_id = digest([row["world_id"], sampling, i, center, delta])[:32]
            frozen_plan.append({"pair_id": pair_id, "source_world_id": row["world_id"],
                                "case_index": i, "control": control, "center_xy": center,
                                "reference_displacement": delta, "event_pair": [0,1]})
    _write(root/"pair-plan.json", _sealed({"schema": "rswm-motion-pair-plan-v1", "role": role,
                                          "source_manifest_sha256": index["source_manifest_sha256"],
                                          "sampling": sampling, "pairs": frozen_plan}, "pair_plan_sha256"))
    rows = []
    lookup = {w["world_id"]: w for w in worlds}
    for world_row in worlds:
        world = _load_world(world_row, world_root)
        for planned in (p for p in frozen_plan if p["source_world_id"] == world_row["world_id"]):
            before,after=controlled_pair_images(world,planned)
            control = planned["control"]
            paths, hashes = {}, {}
            for side, image in (("previous", before), ("current", after)):
                if control == "missing_current" and side == "current": paths[side] = hashes[side] = None; continue
                name = f"images/{planned['pair_id']}-{side}.png"
                target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(image).save(target)
                paths[side], hashes[side] = name, file_hash(target)
            rows.append({**planned, "images": paths, "image_sha256": hashes,
                         "source_world_pixel_sha256": lookup[planned["source_world_id"]]["world_pixel_sha256"]})
    manifest = _sealed({"schema": PAIR_SCHEMA, "protocol": PROTOCOL_SCHEMA, "role": role,
                        "city": ROLES[role], "source_manifest_sha256": index["source_manifest_sha256"],
                        "source_kind": "synthetic_fixture" if diagnostic_fixture else "canonical_inria",
                        "retained_world_inputs":{"world_index":index,"world_root":str(Path(world_root).resolve())},
                        "sampling": sampling, "source_worlds": [{k: row.get(k) for k in
                            ("world_id","role","city","world_file_sha256","world_pixel_sha256","source_sha256","source_path")} for row in worlds],
                        "pair_plan_sha256": _read(root/"pair-plan.json")["pair_plan_sha256"], "pairs": rows}, "pair_manifest_sha256")
    _write(root/"pairs.json", manifest)
    return manifest


def validate_pair_manifest(manifest, role=None):
    _verify_hash(manifest, "pair_manifest_sha256", PAIR_SCHEMA)
    if manifest.get("protocol") != PROTOCOL_SCHEMA or manifest.get("role") not in ROLES or manifest.get("city") != ROLES[manifest["role"]]:
        raise ValueError("Motion pair role/city mismatch")
    if role is not None and manifest["role"] != role: raise ValueError("Motion role is frozen; validation cannot calibrate")
    if manifest.get("source_kind") not in {"canonical_inria", "synthetic_fixture"} or not _sha(manifest.get("source_manifest_sha256")):
        raise ValueError("Unknown pair source kind/identity")
    worlds = manifest.get("source_worlds", [])
    by_world = {w["world_id"]:w for w in worlds}
    if len(worlds) != len(by_world) or not worlds or (manifest["source_kind"] == "canonical_inria" and len(worlds) != 12):
        raise ValueError("Invalid pair world population")
    for world in worlds:
        if world.get("role") != manifest["role"] or world.get("city") != manifest["city"] or not _sha(world.get("world_pixel_sha256")) or not _sha(world.get("world_file_sha256")):
            raise ValueError("Pair source world role/city/hash mismatch")
        if manifest["source_kind"] == "canonical_inria" and (not _sha(world.get("source_sha256")) or not world.get("source_path")):
            raise ValueError("Real pair source identity is absent")
    sampling = manifest.get("sampling", {})
    if sampling.get("controls") != list(CONTROLS) or sampling.get("canonical_crop_hw") != [HEIGHT,WIDTH] or sampling.get("real_source") is not (manifest["source_kind"] == "canonical_inria"):
        raise ValueError("Pair sampling operation profile mismatch")
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs or len({p.get("pair_id") for p in pairs}) != len(pairs):
        raise ValueError("Invalid or duplicate motion pair IDs")
    planned = Counter()
    for pair in pairs:
        if pair.get("source_world_id") not in by_world or pair.get("control") not in CONTROLS:
            raise ValueError("Unknown motion world/control")
        if pair.get("source_world_pixel_sha256") != by_world[pair["source_world_id"]]["world_pixel_sha256"]:
            raise ValueError("Pair world identity mismatch")
        if not isinstance(pair.get("reference_displacement"), list) or len(pair["reference_displacement"]) != 2 or any(type(x) is not int for x in pair["reference_displacement"]):
            raise ValueError("Invalid controlled displacement")
        for side in ("previous", "current"):
            name, checksum = pair.get("images", {}).get(side), pair.get("image_sha256", {}).get(side)
            if name is None:
                if not (pair["control"] == "missing_current" and side == "current" and checksum is None):
                    raise ValueError("Unplanned missing controlled image")
            elif not isinstance(name, str) or not _sha(checksum): raise ValueError("Invalid pair image identity")
        planned[pair["source_world_id"]] += 1
    expected = manifest.get("sampling", {}).get("pairs_per_world")
    if type(expected) is not int or any(planned[w] != expected for w in by_world):
        raise ValueError("Frozen per-world pair count changed")
    return manifest


def _pair_images(pair, root):
    images = []
    for side in ("previous", "current"):
        name = pair["images"][side]
        if name is None: images.append(None); continue
        path = _contained(root, name)
        if file_hash(path) != pair["image_sha256"][side]: raise ValueError("Controlled PNG hash mismatch")
        images.append(load_rgb_png(path))
    return images


def analyze_pairs(manifest, root, config):
    validate_pair_manifest(manifest); validate_config(config)
    pair_plan = _read(Path(root)/"pair-plan.json")
    _verify_hash(pair_plan,"pair_plan_sha256","rswm-motion-pair-plan-v1")
    fields = ("pair_id","source_world_id","case_index","control","center_xy","reference_displacement","event_pair")
    if pair_plan["pair_plan_sha256"] != manifest["pair_plan_sha256"] or pair_plan["role"] != manifest["role"] or pair_plan["source_manifest_sha256"] != manifest["source_manifest_sha256"] or pair_plan["sampling"] != manifest["sampling"] or pair_plan["pairs"] != [{key:pair[key] for key in fields} for pair in manifest["pairs"]]:
        raise ValueError("Pair manifest differs from preregistered crop/operation plan")
    rows = []
    for pair in manifest["pairs"]:
        previous, current = _pair_images(pair, root)
        estimate = estimate_pair(previous, current, config)
        # Ground truth is attached only after estimation has finished.
        error = None
        if estimate["status"] == "measured":
            error = float(np.linalg.norm(np.asarray(estimate["viewport_translation_xy"])-pair["reference_displacement"]))
        rows.append({"pair_id":pair["pair_id"], "source_world_id":pair["source_world_id"],
                     "event_pair":pair["event_pair"], "control":pair["control"], **estimate,
                     "reference_displacement":pair["reference_displacement"], "command_error":error,
                     "normalized_command_error":error/DIAGONAL if error is not None else None})
    return _summarize(rows, manifest, config)


def _summarize(rows, manifest, config):
    tolerance = config["acceptance"]["pixel_tolerance"]
    worlds = []
    for source in manifest["source_worlds"]:
        subset = [row for row in rows if row["source_world_id"] == source["world_id"]]
        measured = [row for row in subset if row["status"] == "measured"]
        wrong = sum(row["command_error"] > tolerance for row in measured)
        worlds.append({"source_world_id":source["world_id"], "planned_pairs":len(subset),
                       "measured_pairs":len(measured), "false_confident_pairs":wrong,
                       "coverage":len(measured)/len(subset) if subset else None,
                       "false_confident_rate":wrong/len(measured) if measured else None,
                       "mean_error_pixels":float(np.mean([r["command_error"] for r in measured])) if measured else None,
                       "status_counts":dict(Counter(row["status"] for row in subset))})
    measured = sorted((row for row in rows if row["status"] == "measured"),
                      key=lambda row: (-(row["confidence"] or 0), row["pair_id"]))
    risk = []
    for k in range(1, len(measured)+1):
        chosen = measured[:k]
        risk.append({"retained_pairs":k, "coverage":k/len(rows), "confidence_threshold":chosen[-1]["confidence"],
                     "risk":sum(row["command_error"] > tolerance for row in chosen)/k})
    average_coverage = float(np.mean([world["coverage"] for world in worlds]))
    error_rates = [world["false_confident_rate"] for world in worlds if world["false_confident_rate"] is not None]
    world_risk = float(np.mean(error_rates)) if error_rates else None
    acceptance = (all(world["measured_pairs"] > 0 for world in worlds) and
                  average_coverage >= config["acceptance"]["min_measurable_coverage"] and
                  world_risk is not None and world_risk <= config["acceptance"]["max_false_confident_rate"])
    return {"config_sha256":config["config_sha256"], "pair_manifest_sha256":manifest["pair_manifest_sha256"],
            "worlds":worlds, "pairs":rows, "planned_pairs":len(rows), "measured_pairs":len(measured),
            "equal_world_coverage":average_coverage, "equal_world_false_confident_rate":world_risk,
            "status_counts":dict(Counter(row["status"] for row in rows)), "risk_coverage":risk,
            "preregistered_acceptance":config["acceptance"], "acceptance_passed":bool(acceptance)}


def calibrate(manifest, root, candidates, out):
    validate_pair_manifest(manifest, "metric_calibration")
    if not isinstance(candidates, list) or not candidates: raise ValueError("Calibration requires frozen candidates")
    candidates = [validate_config(config) for config in candidates]
    if len({config["config_sha256"] for config in candidates}) != len(candidates): raise ValueError("Duplicate candidate configuration")
    if len({digest(config["acceptance"]) for config in candidates}) != 1:
        raise ValueError("Candidate acceptance rules must be fixed before threshold selection")
    selection = {"schema":"rswm-motion-candidate-selection-v1", "rule":"pass preregistered risk/coverage, maximize equal-world coverage, then minimize equal-world false-confident rate, then config hash",
                 "candidate_config_sha256":[config["config_sha256"] for config in candidates]}
    out = Path(out); _write(out/"candidate-plan.json", _sealed(selection, "selection_sha256"))
    reports = [analyze_pairs(manifest, root, config) for config in candidates]
    table = [{k: report[k] for k in ("config_sha256","planned_pairs","measured_pairs","equal_world_coverage","equal_world_false_confident_rate","acceptance_passed")} for report in reports]
    eligible = [report for report in reports if report["acceptance_passed"]]
    selected = min(eligible, key=lambda report:(-report["equal_world_coverage"], report["equal_world_false_confident_rate"], report["config_sha256"])) if eligible else None
    selected_config = next((config for config in candidates if selected and config["config_sha256"] == selected["config_sha256"]), None)
    real = manifest["source_kind"] == "canonical_inria"
    _write(out/"candidate-results.json", {"schema":"rswm-motion-candidate-results-v1", "reports":reports})
    exported = {"candidate-results.json":file_hash(out/"candidate-results.json"),
                "candidate_table.csv":_csv(out,"candidate_table.csv",table)}
    receipt = _sealed({"schema":CALIBRATION_SCHEMA, "protocol":PROTOCOL_SCHEMA,
                       "status":"CONFIG_FROZEN" if selected and real else "DIAGNOSTIC_ONLY",
                       "source_kind":manifest["source_kind"], "pair_manifest_sha256":manifest["pair_manifest_sha256"],
                       "source_manifest_sha256":manifest["source_manifest_sha256"], "source_worlds":manifest["source_worlds"],
                       "selection_plan":_read(out/"candidate-plan.json"), "candidate_table":table,
                       "retained_inputs":{"pair_manifest":manifest,"pair_root":str(Path(root).resolve()),
                                          "candidate_configs":candidates,"output_root":str(out.resolve())},
                       "export_hashes":exported,
                       "selected_config":selected_config, "calibration_role":"metric_calibration",
                       "release_validated":False, "real_calibration_executed":real,
                       "reason_codes":[] if selected and real else ["SYNTHETIC_FIXTURE" if not real else "NO_CANDIDATE_PASSED"]}, "calibration_sha256")
    _write(out/"calibration.json", receipt)
    return receipt


def validate_calibration(receipt):
    _verify_hash(receipt, "calibration_sha256", CALIBRATION_SCHEMA)
    if receipt.get("protocol") != PROTOCOL_SCHEMA or receipt.get("calibration_role") != "metric_calibration":
        raise ValueError("Wrong calibration role/protocol")
    if receipt.get("status") not in {"CONFIG_FROZEN","DIAGNOSTIC_ONLY"} or receipt.get("source_kind") not in {"canonical_inria","synthetic_fixture"}:
        raise ValueError("Invalid calibration status/source")
    if receipt["status"] == "CONFIG_FROZEN" and (receipt["source_kind"] != "canonical_inria" or len(receipt["source_worlds"]) != 12 or receipt.get("real_calibration_executed") is not True):
        raise ValueError("Synthetic or incomplete calibration cannot freeze a real release")
    if any(world.get("role") != "metric_calibration" or world.get("city") != "bellingham" for world in receipt["source_worlds"]):
        raise ValueError("Wrong calibration source population")
    if receipt.get("selected_config") is None: raise ValueError("No calibrated configuration was selected")
    validate_config(receipt["selected_config"])
    _verify_hash(receipt["selection_plan"], "selection_sha256", "rswm-motion-candidate-selection-v1")
    hashes = receipt["selection_plan"]["candidate_config_sha256"]
    if receipt["selected_config"]["config_sha256"] not in hashes or {row["config_sha256"] for row in receipt["candidate_table"]} != set(hashes):
        raise ValueError("Calibration candidate identity mismatch")
    if not any(row["config_sha256"] == receipt["selected_config"]["config_sha256"] and row["acceptance_passed"] is True for row in receipt["candidate_table"]):
        raise ValueError("Selected calibration candidate did not pass frozen acceptance")
    return receipt


def _register_validation_exposure(manifest, calibration, exposure_ledger):
    """Record source exposure before estimation, serialized across evaluators.

    Regenerating held-out pairs with another seed does not erase the fact that
    the same source population was viewed while using a different configuration.
    A caller-supplied ledger is a reproducibility record, not an authenticated
    global history; deleting or substituting it cannot prove independence.
    """
    import fcntl
    exposure_ledger = Path(exposure_ledger)
    if exposure_ledger.is_symlink() or any(p.is_symlink() for p in exposure_ledger.parents):
        raise ValueError("Exposure ledger symlinks are not accepted")
    exposure_ledger.parent.mkdir(parents=True, exist_ok=True)
    lock = exposure_ledger.with_name(exposure_ledger.name + ".lock")
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("Exposure ledger lock must be a regular file")
        fcntl.flock(fd, fcntl.LOCK_EX)
        if exposure_ledger.exists():
            ledger = _read(exposure_ledger)
            if ledger.get("schema") != "rswm-motion-validation-exposure-v1" or not isinstance(ledger.get("exposures"), list):
                raise ValueError("Exposure ledger schema mismatch")
        else:
            ledger = {"schema":"rswm-motion-validation-exposure-v1", "exposures":[]}
        for entry in ledger["exposures"]:
            if not isinstance(entry, dict) or any(not _sha(entry.get(key)) for key in
                ("pair_manifest_sha256", "config_sha256", "calibration_sha256")):
                raise ValueError("Exposure ledger entry identity mismatch")
            keys = entry.get("source_population_keys")
            if keys is not None and (not isinstance(keys, list) or not keys or
                                     len(keys) != len(set(keys)) or any(not _sha(key) for key in keys)):
                raise ValueError("Exposure ledger source population identity mismatch")
        source_keys = sorted({digest([kind, world[key]]) for world in manifest["source_worlds"]
                              for kind, key in (("world_pixels", "world_pixel_sha256"),
                                                ("original_source", "source_sha256")) if world.get(key)})
        previous_pairs = [entry for entry in ledger["exposures"]
                          if entry["pair_manifest_sha256"] == manifest["pair_manifest_sha256"]]
        previous_sources = [entry for entry in ledger["exposures"]
                            if set(entry.get("source_population_keys", [])) & set(source_keys)]
        previous = [entry for entry in ledger["exposures"]
                    if entry in previous_pairs or entry in previous_sources]
        config_sha = calibration["selected_config"]["config_sha256"]
        changed = any(entry["config_sha256"] != config_sha for entry in previous)
        entry = {"pair_manifest_sha256":manifest["pair_manifest_sha256"],
                 "config_sha256":config_sha, "calibration_sha256":calibration["calibration_sha256"],
                 "source_population_keys":source_keys}
        if entry not in ledger["exposures"]:
            ledger["exposures"].append(entry)
            if exposure_ledger.exists():
                temp = exposure_ledger.with_name(exposure_ledger.name+".tmp-"+uuid.uuid4().hex)
                _write(temp, ledger)
                os.replace(temp, exposure_ledger)
            else:
                _write(exposure_ledger, ledger)
        return {"previously_viewed":bool(previous), "pair_set_previously_viewed":bool(previous_pairs),
                "source_population_previously_viewed":bool(previous_sources),
                "used_for_new_config":changed, "ledger_sha256":file_hash(exposure_ledger),
                "ledger_path":str(exposure_ledger.resolve()),"ledger_snapshot":ledger}
    finally:
        os.close(fd)


def validate_motion(manifest, root, calibration, out, exposure_ledger, diagnostic_fixture=False):
    validate_pair_manifest(manifest, "metric_validation"); validate_calibration(calibration)
    if not diagnostic_fixture and (manifest["source_kind"] != "canonical_inria" or calibration["status"] != "CONFIG_FROZEN"):
        raise ValueError("Real validation requires real frozen calibration and independent real pair manifest")
    if manifest["pair_manifest_sha256"] == calibration["pair_manifest_sha256"]:
        raise ValueError("Calibration and validation pair sets cannot coincide")
    for key in ("world_id", "world_pixel_sha256", "source_sha256"):
        calibration_ids = {w.get(key) for w in calibration["source_worlds"] if w.get(key) is not None}
        validation_ids = {w.get(key) for w in manifest["source_worlds"] if w.get(key) is not None}
        if calibration_ids & validation_ids: raise ValueError("Calibration/validation source overlap")
    config = calibration["selected_config"]
    # Register exposure BEFORE image evaluation, including runs that later fail.
    exposure = _register_validation_exposure(manifest, calibration, exposure_ledger)
    report = analyze_pairs(manifest, root, config)
    real = manifest["source_kind"] == calibration["source_kind"] == "canonical_inria" and not diagnostic_fixture
    passed = real and report["acceptance_passed"] and not exposure["used_for_new_config"]
    exported = {name:_csv(out,name,report[key]) for name,key in
                (("pairs.csv","pairs"),("worlds.csv","worlds"),("risk_coverage.csv","risk_coverage"))}
    receipt = _sealed({"schema":VALIDATION_SCHEMA, "protocol":PROTOCOL_SCHEMA,
                       "status":"VALIDATED" if passed else "DIAGNOSTIC_ONLY",
                       "pair_manifest_sha256":manifest["pair_manifest_sha256"],
                       "source_manifest_sha256":manifest["source_manifest_sha256"],
                       "calibration_sha256":calibration["calibration_sha256"], "config":config,
                       "retained_inputs":{"pair_manifest":manifest,"pair_root":str(Path(root).resolve()),
                                          "calibration_receipt":calibration,"output_root":str(Path(out).resolve()),
                                          "exposure_ledger_path":exposure["ledger_path"],
                                          "exposure_ledger_snapshot":exposure["ledger_snapshot"]},
                       "source_kind":manifest["source_kind"], "source_worlds":manifest["source_worlds"],
                       "report":report, "release_validated":bool(passed), "real_validation_executed":real,
                       "export_hashes":exported,
                       "validation_previously_viewed":exposure["previously_viewed"],
                       "validation_pair_set_previously_viewed":exposure["pair_set_previously_viewed"],
                       "validation_source_population_previously_viewed":exposure["source_population_previously_viewed"],
                       "validation_used_for_new_config":exposure["used_for_new_config"],
                       "exposure_ledger_sha256_at_execution":exposure["ledger_sha256"],
                       "reason_codes": (["VALIDATION_ALREADY_VIEWED_WITH_DIFFERENT_CONFIG"] if exposure["used_for_new_config"] else []) +
                          ([] if real else ["SYNTHETIC_FIXTURE"]) + ([] if report["acceptance_passed"] else ["ACCEPTANCE_FAILED"])}, "validation_sha256")
    _write(Path(out)/"validation.json", receipt)
    return receipt


def diagnose_images(images, queries, config, event_start=0):
    """Outer evaluator compares commands only after image-only edge estimation."""
    validate_config(config)
    if not images or len(images) != len(queries): raise ValueError("One image slot per query is required")
    if any(not isinstance(q, (list,tuple,np.ndarray)) or len(q) != 2 or
           any(isinstance(x, (bool,np.bool_)) or not isinstance(x, (int,np.integer)) for x in q) for q in queries):
        raise ValueError("Integer xy command pairs are required")
    edges, position, anchored, anchored_edges, path_length = [], np.zeros(2), True, 0, 0.
    for i in range(1,len(images)):
        estimate = estimate_pair(images[i-1], images[i], config)
        command = np.asarray(queries[i],dtype=float)-np.asarray(queries[i-1],dtype=float)
        error = None
        if estimate["status"] == "measured":
            delta = np.asarray(estimate["viewport_translation_xy"],dtype=float)
            error = float(np.linalg.norm(delta-command))
            path_length += float(np.linalg.norm(delta))
            if anchored: position += delta; anchored_edges += 1
        else: anchored = False
        edges.append({**estimate, "event_pair":[event_start+i-1,event_start+i],
                      "command_displacement_xy":command.astype(int).tolist(),
                      "command_error":error, "normalized_command_error":error/DIAGONAL if error is not None else None})
    complete = bool(edges) and all(edge["status"] == "measured" for edge in edges)
    errors = [edge["normalized_command_error"] for edge in edges if edge["normalized_command_error"] is not None]
    return {"schema":"rswm-motion-sequence-v1", "config_sha256":config["config_sha256"], "edges":edges,
            "planned_edges":len(edges), "measured_edges":len(errors), "coverage":len(errors)/len(edges) if edges else None,
            "mean_normalized_step_error":float(np.mean(errors)) if errors else None,
            "status_counts":dict(Counter(edge["status"] for edge in edges)), "anchored_prefix_edges":anchored_edges,
            "full_path_measurable":complete, "estimated_full_path_length":path_length if complete else None,
            "estimated_endpoint_translation_xy":position.tolist() if complete else None,
            "normalized_endpoint_error":float(np.linalg.norm(position-(np.asarray(queries[-1])-np.asarray(queries[0]))))/DIAGONAL if complete else None}


def paired_motion_ratio(small, large):
    if small["config_sha256"] != large["config_sha256"]: raise ValueError("F3 paired motion configurations differ")
    eligible = bool(small["full_path_measurable"] and large["full_path_measurable"] and small["estimated_full_path_length"] > 0)
    return {"planned_pairs":1, "measurable_pairs":int(eligible), "paired_coverage":float(eligible),
            "large_over_small":large["estimated_full_path_length"]/small["estimated_full_path_length"] if eligible else None}


def _branch_path(prefix, suffix, queries):
    complete = prefix["full_path_measurable"] and suffix["full_path_measurable"]
    endpoint = np.asarray(prefix["estimated_endpoint_translation_xy"])+np.asarray(suffix["estimated_endpoint_translation_xy"]) if complete else None
    return {"full_path_measurable":complete,
            "planned_edges":prefix["planned_edges"]+suffix["planned_edges"],
            "measured_edges":prefix["measured_edges"]+suffix["measured_edges"],
            "anchored_prefix_edges":prefix["anchored_prefix_edges"]+(suffix["anchored_prefix_edges"] if prefix["full_path_measurable"] else 0),
            "estimated_full_path_length":prefix["estimated_full_path_length"]+suffix["estimated_full_path_length"] if complete else None,
            "estimated_endpoint_translation_xy":endpoint.tolist() if complete else None,
            "normalized_endpoint_error":float(np.linalg.norm(endpoint-(np.asarray(queries[-1])-np.asarray(queries[0]))))/DIAGONAL if complete else None}


def validate_release(receipt, config):
    _verify_hash(receipt, "validation_sha256", VALIDATION_SCHEMA)
    if receipt.get("status") != "VALIDATED" or receipt.get("release_validated") is not True or receipt.get("real_validation_executed") is not True or receipt.get("source_kind") != "canonical_inria":
        raise ValueError("Motion instrument is DIAGNOSTIC_ONLY; real validation has not passed")
    if receipt.get("validation_used_for_new_config") or receipt.get("config") != config:
        raise ValueError("Release config or independent-validation identity mismatch")
    if not _sha(receipt.get("source_manifest_sha256")) or not _sha(receipt.get("pair_manifest_sha256")):
        raise ValueError("Release source manifest and pair identity are required")
    worlds=receipt.get("source_worlds",[])
    if len(worlds)!=12 or len({world.get("world_id") for world in worlds})!=12 or any(
        world.get("role")!="metric_validation" or world.get("city")!="bloomington" or any(not _sha(world.get(key))
        for key in ("world_pixel_sha256","world_file_sha256","source_sha256")) for world in worlds):
        raise ValueError("Release requires the complete independent Bloomington source population")
    report=receipt.get("report",{})
    if report.get("acceptance_passed") is not True or report.get("config_sha256")!=config["config_sha256"]:
        raise ValueError("Release acceptance was not passed")
    if type(report.get("equal_world_coverage")) not in (int,float) or type(report.get("equal_world_false_confident_rate")) not in (int,float) or report["equal_world_coverage"]<config["acceptance"]["min_measurable_coverage"] or report["equal_world_false_confident_rate"]>config["acceptance"]["max_false_confident_rate"]:
        raise ValueError("Release contradicts preregistered acceptance thresholds")
    return receipt


def measure_suite(plan, suite_root, config, validation_receipt=None, diagnostic=False, selection=None):
    """Measure replay-verified submission PNGs and retain every expected slot."""
    validate_config(config)
    if not diagnostic:
        if validation_receipt is None: raise ValueError("A real validated motion release is required")
        validate_release(validation_receipt,config)
    from .suite import replay_suite, suite_read_lock
    rows, all_edges = [], []
    with suite_read_lock(suite_root):
        results = {row["run_id"]:row for row in replay_suite(plan,suite_root,selection)}
        for expected in plan["expected"]:
            task = plan["tasks"][expected["task_id"]]
            result = results.get(expected["run_id"], {})
            verified = result.get("submission_status") == "VERIFIED" and result.get("replay_verified") is True
            # Paths derive from the authenticated suite identity, never mutable
            # outcome run_dir metadata. Independent replay returns slot validity.
            root = Path(suite_root)/"runs"/expected["run_id"]/f"attempt-{result['attempt']:03d}"/"submission" if verified else None
            initial = load_rgb_png(root/"initial.png") if verified else None
            slots = {(row["stream"],row["event"]):row.get("valid") is True for row in result.get("output_slots",[])}
            def frame(stream,event):
                return load_rgb_png(root/stream/f"{event:04d}.png") if verified and slots.get((stream,event)) else None
            streams, branch_paths = {}, {}
            if task["family"] == "F7":
                q = task["programs"]["a"]
                prefix_images = [initial]+[frame("prefix",event) for event in range(1,33)]
                streams["prefix"] = diagnose_images(prefix_images,q[:33],config)
                for branch in ("a","b"):
                    queries = task["programs"][branch]
                    suffix = [prefix_images[-1]]+[frame(branch,event) for event in range(33,97)]
                    streams[branch] = diagnose_images(suffix,queries[32:],config,event_start=32)
                    branch_paths[branch] = _branch_path(streams["prefix"],streams[branch],queries)
            else:
                for stream,queries in task["programs"].items():
                    images = [initial]+[frame(stream,event) for event in range(1,len(queries))]
                    streams[stream] = diagnose_images(images,queries,config)
            identity = {key:expected[key] for key in ("run_id","task_id","model_id","world_id","city","family","protocol","seed",
                                                    "study","role","variant","training_seed","model_manifest_sha256",
                                                    "source_manifest_sha256","metric_config_sha256","task_sha256")}
            identity.update(attempt=result.get("attempt"),analysis_selection=result.get("selection"))
            unique = []
            for stream,report in streams.items():
                for edge in report["edges"]:
                    row = {**identity,"stream":stream,"pair_id":digest([expected["run_id"],stream,edge["event_pair"]]),
                           "source_world_id":expected["world_id"],**edge}
                    unique.append(row); all_edges.append(row)
            planned_outputs = sum(len(queries)-1 for queries in task["programs"].values()) if task["family"] != "F7" else 160
            if len(unique) != planned_outputs: raise ValueError("Motion planned slot count mismatch")
            rows.append({**identity,"submission_status":result.get("submission_status","NOT_VERIFIED"),
                         "streams":streams,"branch_paths":branch_paths,"unique_planned_edges":len(unique),
                         "unique_measured_edges":sum(row["status"] == "measured" for row in unique),
                         "f3_ratio":paired_motion_ratio(streams["small"],streams["large"]) if task["family"] == "F3" else None})
    from .metric_results import evaluation_binding
    return {"schema":"rswm-motion-suite-v1", "plan_sha256":plan["plan_sha256"],"config":config,
            "validation_sha256":validation_receipt.get("validation_sha256") if validation_receipt else None,
            "release_validated":not diagnostic,"status":"DIAGNOSTIC_ONLY" if diagnostic else "VALIDATED_INSTRUMENT",
            "evaluation_binding":evaluation_binding(plan,suite_root,selection,results=list(results.values())),
            "runs":rows,"edges":all_edges,"expected_runs":len(plan["expected"]),
            "planned_edges":len(all_edges),"measured_edges":sum(row["status"] == "measured" for row in all_edges)}


def evaluate_suite(plan, suite_root, config, out, validation_receipt=None, diagnostic=False, selection=None):
    """Publish actual model measurements, separately from instrument validation."""
    from .suite import suite_read_lock
    with suite_read_lock(suite_root):
        report=measure_suite(plan,suite_root,config,validation_receipt,diagnostic,selection)
        report["retained_inputs"]={"suite_root":str(Path(suite_root).resolve()),"validation_receipt":validation_receipt}
        report["export_hashes"]={"motion_edges.csv":_csv(out,"motion_edges.csv",report["edges"]),
                                "motion_run_coverage.csv":_csv(out,"motion_run_coverage.csv",[
                                    {key:row[key] for key in ("run_id","task_id","model_id","world_id","city","family","protocol","seed",
                                                           "submission_status","unique_planned_edges","unique_measured_edges")}
                                    for row in report["runs"]])}
        receipt=_sealed(report,"motion_suite_sha256")
        _write(Path(out)/"motion.json",receipt)
    return receipt


def _cli_failure(out, command, exc):
    missing = isinstance(exc, FileNotFoundError)
    record = {"schema":"rswm-motion-execution-status-v1","command":command,
              "status":"BLOCKED" if missing else "FAILED", "real_calibration":"NOT_RUN",
              "real_validation":"NOT_RUN", "error_code":type(exc).__name__, "detail":str(exc)}
    _write(Path(out)/"execution-status.json",record)
    print(json.dumps(record,sort_keys=True))
    return 2 if missing else 1


def calibration_main(argv=None):
    parser = argparse.ArgumentParser(description="Build fixed image pairs or calibrate motion on Bellingham only")
    commands = parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("build-pairs");p.add_argument("--world-index",required=True);p.add_argument("--world-root",required=True)
    p.add_argument("--role",choices=sorted(ROLES),required=True);p.add_argument("--pairs-per-world",type=int,default=256)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--diagnostic-fixture",action="store_true");p.add_argument("--out",required=True)
    p=commands.add_parser("calibrate");p.add_argument("--pair-manifest",required=True);p.add_argument("--candidates",required=True);p.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    try:
        if args.command == "build-pairs":
            result=build_pairs(_read(args.world_index),args.world_root,args.out,args.role,args.pairs_per_world,args.seed,args.diagnostic_fixture)
        else:
            result=calibrate(_read(args.pair_manifest),Path(args.pair_manifest).parent,_read(args.candidates),args.out)
    except (OSError,ValueError,KeyError,TypeError) as exc: return _cli_failure(args.out,args.command,exc)
    print(json.dumps({"command":args.command,"status":result.get("status","PAIRS_FROZEN"),"out":args.out},sort_keys=True))
    return 0 if args.command == "build-pairs" or result["status"] == "CONFIG_FROZEN" else 1


def validation_main(argv=None):
    parser=argparse.ArgumentParser(description="Validate a frozen motion config on held-out Bloomington or evaluate verified PNGs")
    commands=parser.add_subparsers(dest="command",required=True)
    p=commands.add_parser("validate");p.add_argument("--pair-manifest",required=True);p.add_argument("--calibration",required=True)
    p.add_argument("--exposure-ledger",required=True);p.add_argument("--diagnostic-fixture",action="store_true");p.add_argument("--out",required=True)
    p=commands.add_parser("evaluate-suite");p.add_argument("--plan",required=True);p.add_argument("--suite",required=True)
    p.add_argument("--config",required=True);p.add_argument("--validation");p.add_argument("--diagnostic",action="store_true");p.add_argument("--out",required=True)
    p.add_argument("--selection",choices=["first_attempt_only","latest_attempt_within_budget"])
    args=parser.parse_args(argv)
    try:
        if args.command == "validate":
            result=validate_motion(_read(args.pair_manifest),Path(args.pair_manifest).parent,_read(args.calibration),args.out,args.exposure_ledger,args.diagnostic_fixture)
        else:
            result=evaluate_suite(_read(args.plan),args.suite,_read(args.config),args.out,_read(args.validation) if args.validation else None,args.diagnostic,args.selection)
    except (OSError,ValueError,KeyError,TypeError) as exc: return _cli_failure(args.out,args.command,exc)
    print(json.dumps({"command":args.command,"status":result["status"],"out":args.out},sort_keys=True))
    return 0 if result["status"] in {"VALIDATED","VALIDATED_INSTRUMENT"} else 1
