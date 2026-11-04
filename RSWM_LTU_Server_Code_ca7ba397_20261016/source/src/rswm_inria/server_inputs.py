"""Prepare real local server inputs without model loading or inferred approval.

These commands derive identities and existing CLI inputs, not research results.
They never download, import a vendor entrypoint, instantiate Torch, or rewrite
an audited build. Output files are published exclusively outside input trees.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from .adapters.camera_conditioning import CameraConfig
from .data import INRIA_REVISION, REPO, select_manifest
from .geometry import VERSION, digest, file_hash
from .hf_materialize import PILOT_DEV2_SOURCES
from .legacy_wan import inspect_backend
from .registry import numeric_contract, validate_config
from .suite import _json, _load_task, _save, _world, code_fingerprint
from .tasks import packages
from .training import ClipDataset, locked_tree, validate_dev_index


CAMERA_FIELDS = {"translation_scaling", "framewise", "camera_channel", "temporal_stride"}
TRAINING_SPEC_FIELDS = {"updates", "batch_size", "accumulation", "lr", "seed", "precision",
                        "device", "camera", "trainable_policy", "checkpoint_every", "backend_manifest"}


def _safe_path(value, *, base=None, file=False, directory=False):
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("A nonempty local path is required")
    path = Path(value)
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError("Server input symlinks are not allowed")
    path = path.resolve()
    if file and not path.is_file():
        raise FileNotFoundError(path)
    if directory and not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _publish(path, value, *, protected=()):
    output = _safe_path(path)
    for root in protected:
        root = _safe_path(root)
        if output == root or (root.is_dir() and output.is_relative_to(root)):
            raise ValueError("Derived inputs must be outside the original input/build trees")
    _save(output, value, exclusive=True)
    return output


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _camera(path, *, base=None):
    location = _safe_path(path, base=base, file=True)
    value = _json(location)
    if not isinstance(value, dict) or set(value) != CAMERA_FIELDS:
        raise ValueError("Camera input must contain exactly the existing four camera fields")
    camera = CameraConfig(**value)
    if camera.temporal_stride != 4:
        raise ValueError("The selected original Wan VAE requires temporal_stride=4")
    return value, digest(camera.manifest())


def _inspect_complete(source_dir, weights_dir):
    source = _safe_path(source_dir, directory=True)
    weights = _safe_path(weights_dir, directory=True)
    inspected = inspect_backend(source_dir=source, weights_dir=weights)
    components = inspected.get("components")
    if not isinstance(components, dict) or components.get("complete") is not True:
        raise ValueError("BLOCKED: actual local Wan components are missing or empty")
    # Recheck actual bytes; values from an old inspection are never copied in.
    source_hash, weights_hash = digest(locked_tree(source)), digest(locked_tree(weights))
    if inspected["source_tree_sha256"] != source_hash or components["weights_tree_sha256"] != weights_hash:
        raise ValueError("Source/weights changed during server input preparation")
    return source, weights, inspected, source_hash, weights_hash


def prepare_numeric_model(*, source_dir, weights_dir, camera, decisions, model_id, device, out):
    """Create the existing numeric-camera registry config from observed files."""
    decision_path = _safe_path(decisions, file=True)
    decision = _json(decision_path)
    if (not isinstance(decision, dict) or set(decision) != {"approved", "license_record"}
            or decision["approved"] is not True or not isinstance(decision["license_record"], str)
            or not decision["license_record"].strip()):
        raise ValueError("BLOCKED: operator approval and actual license/use decision are required")
    camera_value, camera_hash = _camera(camera)
    source, weights, _inspection, source_hash, weights_hash = _inspect_complete(source_dir, weights_dir)
    manifest = {"source_dir": str(source), "source_tree_sha256": source_hash, "entrypoint": "backend.py",
                "weights_dir": str(weights), "weights_tree_sha256": weights_hash, "objective": "flow_matching",
                "approved": decision["approved"], "license_record": decision["license_record"]}
    value = {"model_id": model_id, "adapter": "numeric-camera", "device": device,
             "backend_manifest": manifest, "camera": camera_value, "training_camera_sha256": camera_hash}
    # A trained release needs its original binding and training seed. Do not
    # infer those from filenames or silently fall back to a base manifest.
    release = _inspection["config"].get("adapter_release")
    if release is not None:
        from .legacy_export import validate_adapter_release
        retained = validate_adapter_release(release, expected_weights_tree_sha256=weights_hash,
                                            expected_camera_sha256=camera_hash)
        value["adapter_release"] = copy.deepcopy(release)
        value["training_seed"] = retained["training"]["training_seed"]
    validate_config(value)
    numeric_contract(value)
    _publish(out, value, protected=(source, weights, decision_path, _safe_path(camera, file=True)))
    return value


def _selection(build, profile):
    root = _safe_path(build, directory=True)
    selection = _json(_safe_path(root / "build-selection.json", file=True))
    fields = {"core_manifest_sha256", "repo_id", "revision", "profile", "full_core", "source_bytes", "worlds", "input_kind"}
    if (not isinstance(selection, dict) or set(selection) != fields or selection["profile"] != profile
            or selection["repo_id"] != REPO or selection["revision"] != INRIA_REVISION
            or selection["input_kind"] != "source_identity_checked_geography_unknown"
            or type(selection["full_core"]) is not bool or selection["full_core"] != (profile == "core120")
            or not _sha(selection["core_manifest_sha256"])):
        raise ValueError("Build must retain the exact real frozen source/profile selection")
    rows = selection["worlds"]
    if profile == "pilot-dev2":
        expected = [{**fixed, "selection_hash": hashlib.sha256((VERSION + "/" + fixed["path"]).encode()).hexdigest()}
                    for fixed in PILOT_DEV2_SOURCES]
        # JSON booleans must not masquerade as selection ranks or byte sizes.
        if rows != expected or any(type(row.get(key)) is not type(fixed[key])
                                   for row, fixed in zip(rows, expected) for key in fixed):
            raise ValueError("Pilot requires the fixed Kitsap dev2 source identities and order")
    else:
        if not isinstance(rows, list) or len(rows) != 120:
            raise ValueError("Training requires the complete core120 build")
        manifest = select_manifest({"repo_id": REPO, "revision": INRIA_REVISION, "files": rows})
        if manifest["worlds"] != rows or manifest["manifest_sha256"] != selection["core_manifest_sha256"]:
            raise ValueError("Core120 selected paths/roles/quotas/hash do not match the frozen source contract")
    if type(selection["source_bytes"]) is not int or selection["source_bytes"] != sum(row["size_bytes"] for row in rows):
        raise ValueError("Build source byte count differs from its selection")
    index = _json(_safe_path(root / "private/index.json", file=True))
    if (not isinstance(index, dict) or set(index) != {"tasks", "worlds", "source_manifest_sha256"}
            or index["source_manifest_sha256"] != selection["core_manifest_sha256"]):
        raise ValueError("Private index must retain its complete parent manifest identity")
    worlds = index["worlds"]
    if not isinstance(worlds, list) or len(worlds) != len(rows) or [w["world_id"] for w in worlds] != [w["world_id"] for w in rows]:
        raise ValueError("Private world index is incomplete, reordered or repeated")
    for source, world in zip(rows, worlds):
        if (any(world.get(key) != value or type(world.get(key)) is not type(value) for key, value in source.items() if key != "path")
                or world.get("source_path") != source["path"] or world.get("source_sha256") != source["lfs_sha256"]
                or world.get("path") != world["world_id"] + ".png"
                or not _sha(world.get("world_file_sha256")) or not _sha(world.get("world_pixel_sha256"))):
            raise ValueError("Private world index source/role/hash identity differs")
    completion = _json(_safe_path(root / "BUILD_COMPLETE.json", file=True))
    expected_completion = {"profile": profile, "source_bytes": selection["source_bytes"], "worlds": len(rows),
                           "initial_images": 4 * len(rows), "task_packages": 16 * len(rows),
                           "query_outputs_all_roles": len(rows) * 4 * 352,
                           "real_model_inference": False, "public_export_is_security_sandbox": False}
    if completion != expected_completion or any(type(completion[k]) is not type(v) for k, v in expected_completion.items()):
        raise ValueError("Build completion scope/counts are inconsistent")
    return root, selection, index


def _indexed_tasks(root, index, *, frozen=False):
    """Bind complete parent index entries to actual task bytes and label hashes."""
    worlds = {row["world_id"]: row for row in index["worlds"]}
    expected_rows, bound = [], []
    if not isinstance(index["tasks"], list) or len(index["tasks"]) != 16 * len(worlds):
        raise ValueError("Complete parent task inventory is required")
    for world in index["worlds"]:
        wid, role = world["world_id"], world["role"]
        for origin in range(4):
            if frozen:
                generated = [json.loads(json.dumps(task)) for task, _labels in packages(wid, role, origin)]
            else:
                generated = [{"task_id": f"{wid}-o{origin}-{family}", "family": family,
                              "protocol": "online16" if family in {"F5", "F7"} else "offline",
                              "origin_index": origin, "world_id": wid, "role": role}
                             for family in ("F2", "F3", "F5", "F7")]
            for expected in generated:
                tid = expected["task_id"]
                expected_row = {"task_id": tid, "session_id": digest([tid, "public-session"])[:24],
                                "family": expected["family"], "world_id": wid, "role": role, "city": world["city"],
                                "task_path": "tasks/" + tid + ".json", "world_path": "worlds/" + world["path"]}
                expected_rows.append(expected_row)
                path = _safe_path(root / "private" / expected_row["task_path"], file=True)
                task, _labels = _load_task(path)
                label_names = {"main"} if task["family"] == "F5" else {"a", "b"} if task["family"] == "F7" else set()
                if set(task.get("private_label_files", {})) != label_names:
                    raise ValueError("Complete parent task private-label bindings are required")
                bare = {k: v for k, v in task.items() if k != "private_label_files"}
                if (frozen and bare != expected) or any(task.get(k) != v or type(task.get(k)) is not type(v)
                                                     for k, v in expected.items() if k in {"task_id", "world_id", "role", "origin_index", "family", "protocol"}):
                    raise ValueError("Actual task bytes differ from the fixed parent task identity")
                bound.append((expected_row, task))
    if index["tasks"] != expected_rows:
        raise ValueError("Parent task index has omitted, reordered or substituted entries")
    return bound


def prepare_pilot_index(*, build, out):
    """Select origin0's eight runs while leaving the 32-package build intact."""
    root, selection, index = _selection(build, "pilot-dev2")
    for world in index["worlds"]:
        path = _safe_path(root / "private/worlds" / world["path"], file=True)
        if (file_hash(path) != world["world_file_sha256"]
                or hashlib.sha256(_world(path).tobytes()).hexdigest() != world["world_pixel_sha256"]):
            raise ValueError("Pilot world file/pixel checksum differs from its parent index")
    bound = _indexed_tasks(root, index, frozen=True)
    clips = _json(_safe_path(root / "private/training-clips.json", file=True))
    if clips != {"clips": [], "clip_count": 0, "view_events": 0}:
        raise ValueError("Pilot-dev2 cannot contain training clips")
    selected = [{**row, "task_path": str(root / "private" / row["task_path"])}
                for row, task in bound if task["origin_index"] == 0]
    if len(selected) != 8:
        raise ValueError("Fixed pilot must retain exactly eight task packages")
    value = {"tasks": selected, "worlds": copy.deepcopy(index["worlds"]),
             "source_manifest_sha256": index["source_manifest_sha256"],
             "world_root": str(root / "private/worlds"), "parent_build": str(root),
             "parent_index_sha256": file_hash(root / "private/index.json"),
             "parent_selection_sha256": digest(selection),
             "selection_scope": "fixed-dev2-origin0-eight-packages-not-a-data-audit"}
    _publish(out, value, protected=(root,))
    return value


def _manifest(path, *, base):
    location = _safe_path(path, base=base, file=True)
    value = _json(location)
    if isinstance(value, dict) and "backend_manifest" in value:
        value = value["backend_manifest"]
    required = {"source_dir", "source_tree_sha256", "entrypoint", "weights_dir", "weights_tree_sha256", "objective", "approved", "license_record"}
    if (not isinstance(value, dict) or set(value) != required or value["approved"] is not True
            or not isinstance(value["license_record"], str) or not value["license_record"].strip()
            or value["objective"] != "flow_matching" or value["entrypoint"] != "backend.py"):
        raise ValueError("BLOCKED: original Wan requires an actual approved backend manifest")
    value = copy.deepcopy(value)
    source, weights, inspected, source_hash, weights_hash = _inspect_complete(
        _safe_path(value["source_dir"], base=location.parent),
        _safe_path(value["weights_dir"], base=location.parent))
    if value["source_tree_sha256"] != source_hash or value["weights_tree_sha256"] != weights_hash:
        raise ValueError("Approved backend source/weight checksum differs from actual files")
    value.update(source_dir=str(source), weights_dir=str(weights))
    return value, inspected, location


def prepare_training_config(*, spec, build, source_audit, out):
    """Derive driver dev records from approved Wan policy and current core120."""
    specification = _safe_path(spec, file=True)
    value = _json(specification)
    if not isinstance(value, dict) or set(value) != TRAINING_SPEC_FIELDS:
        raise ValueError("Exactly eleven operator training spec fields are required; validation_index is derived")
    for key in ("updates", "batch_size", "accumulation", "checkpoint_every"):
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError("Positive explicit integer training budget is required")
    if (type(value["seed"]) is not int or value["seed"] < 0 or type(value["lr"]) not in (int, float)
            or not math.isfinite(value["lr"]) or value["lr"] <= 0
            or not isinstance(value["device"], str) or re.fullmatch(r"cuda(?::[0-9]+)?", value["device"]) is None
            or value["precision"] != "bf16"):
        raise ValueError("Explicit original-Wan CUDA, bf16, seed and positive finite learning rate are required")
    root, selection, index = _selection(build, "core120")
    audit_path = _safe_path(source_audit, file=True)
    audit = _json(audit_path)
    if (not isinstance(audit, dict) or audit.get("profile") != "core120"
            or audit.get("source_code_sha256") != code_fingerprint()
            or audit.get("manifest_sha256") != selection["core_manifest_sha256"]):
        raise ValueError("Training requires a current matching full core120 source audit")
    # Validate the complete role matrix and physical artifact bytes before
    # deriving references. No field of an old audit is relabelled here.
    bound = _indexed_tasks(root, index)
    worlds = {row["world_id"]: row for row in index["worlds"]}
    expected = [{**{k: task[k] for k in ("task_id", "world_id", "role", "origin_index", "family", "protocol")},
                 **{k: worlds[task["world_id"]][k] for k in ("city", "world_file_sha256", "world_pixel_sha256")},
                 "task_sha256": digest(task), "source_manifest_sha256": index["source_manifest_sha256"]}
                for _row, task in bound]
    from .materialized import validate_audit_receipt
    validate_audit_receipt(audit, {"expected": expected, "source_code_sha256": code_fingerprint()}, required_profile="core120")
    artifacts = audit["artifact_sha256"]
    for name in ("build-selection.json", "BUILD_COMPLETE.json", "private/index.json", "private/training-clips.json"):
        if artifacts.get(name) != file_hash(_safe_path(root / name, file=True)):
            raise ValueError("Current core120 input files differ from their retained source audit")
    for name, expected_hash in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source audit artifact path escapes the build")
        if file_hash(_safe_path(root / relative, file=True)) != expected_hash:
            raise ValueError("Retained core120 artifact checksum differs: " + name)
    clips = _json(_safe_path(root / "private/training-clips.json", file=True))
    dataset = ClipDataset(index, clips, root / "private/worlds", source_audit=audit)
    camera_value, camera_hash = _camera(value["camera"], base=specification.parent)
    manifest, inspection, manifest_path = _manifest(value["backend_manifest"], base=specification.parent)
    policy = inspection["config"].get("training")
    if (not isinstance(policy, dict) or value["trainable_policy"] != policy["trainable_policy"]
            or value["precision"] != policy["precision"]):
        raise ValueError("Operator training policy must match the actual locked original-Wan source configuration")
    mode = value["trainable_policy"]["mode"]
    if (mode == "new_lora" and inspection["config"]["mode"] != "base") or (mode == "restored_lora" and inspection["config"]["mode"] != "restored_lora"):
        raise ValueError("Training initialization does not match the approved source mode")
    validation = policy["validation"]
    if validation["max_total_forwards"] != value["updates"] * len(validation["world_ids"]):
        raise ValueError("Locked validation forward budget must equal approved updates times dev-world count")
    rows = []
    for wid in validation["world_ids"]:
        if wid not in worlds:
            raise ValueError("Locked validation world is absent from current core120")
        world = worlds[wid]
        rows.append({k: world[k] for k in ("world_id", "city", "role", "source_sha256", "world_file_sha256", "world_pixel_sha256", "path")})
    validate_dev_index(rows, dataset)
    numeric_contract({"model_id": "training-input-preflight", "adapter": "numeric-camera", "device": value["device"],
                      "backend_manifest": manifest, "camera": camera_value, "training_camera_sha256": camera_hash})
    result = {**copy.deepcopy(value), "camera": camera_value, "backend_manifest": manifest, "validation_index": rows}
    _publish(out, result, protected=(root, specification, audit_path, manifest_path,
                                    _safe_path(manifest["source_dir"]), _safe_path(manifest["weights_dir"])))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    numeric = commands.add_parser("numeric-model")
    for name in ("source-dir", "weights-dir", "camera", "decisions", "model-id", "device", "out"):
        numeric.add_argument("--" + name, required=True)
    pilot = commands.add_parser("pilot-index")
    pilot.add_argument("--build", required=True); pilot.add_argument("--out", required=True)
    training = commands.add_parser("training-config")
    for name in ("spec", "build", "source-audit", "out"):
        training.add_argument("--" + name, required=True)
    args = vars(parser.parse_args(argv)); command = args.pop("command")
    try:
        value = {"numeric-model": prepare_numeric_model, "pilot-index": prepare_pilot_index,
                 "training-config": prepare_training_config}[command](**args)
        print(json.dumps({"status": "INPUT_PREPARED_NOT_MODEL_VALIDATED", "command": command,
                          "out": str(Path(args["out"]).resolve()), "input_sha256": digest(value),
                          "real_model_validation": "NOT_RUN", "research_acceptance": "NOT_EVALUATED"}, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_INVALID", "error_type": type(exc).__name__, "reason": str(exc)}, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
