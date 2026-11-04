"""Independent checks of Inria build products and bounded diagnostic acceptance.

The auditor reads every declared product. It recomputes source pixels with
strided slices rather than the materializer's reshape/average implementation,
and audits private labels using independent per-pixel visibility histories.
It never downloads inputs and never labels a synthetic fixture as real data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
from .data import verify_manifest, source_pixels, build_world, write_json
from .geometry import ORIGINS, file_hash, digest
from .tasks import packages, training_clips


class AuditError(ValueError):
    pass


def read_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise AuditError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise AuditError("Non-finite JSON number: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid)


def _path(root, relative):
    root = Path(root).absolute()
    p = root / relative
    try:
        p.absolute().relative_to(root)
        p.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise AuditError("Artifact path leaves its declared root.") from exc
    for ancestor in (p, *p.parents):
        if ancestor.is_symlink():
            raise AuditError("Symlinks are not accepted in source/build inventories.")
    if not p.is_file():
        raise AuditError("Missing artifact: " + str(p))
    return p


def independent_world(source):
    a = source[452:4548, 452:4548]
    # Four disjoint strided samples; no call to downsample2/reshape reduction.
    sums = a[0::2, 0::2].astype(np.uint16)
    sums += a[0::2, 1::2]
    sums += a[1::2, 0::2]
    sums += a[1::2, 1::2]
    return ((sums + 2) // 4).astype(np.uint8)


def _png(path, shape):
    try:
        with Image.open(path) as im:
            if im.format != "PNG" or im.mode != "RGB" or getattr(im, "n_frames", 1) != 1:
                raise AuditError("Product must be a single RGB PNG.")
            pixels = np.asarray(im).copy()
    except (OSError, EOFError, SyntaxError) as exc:
        raise AuditError("Invalid product image.") from exc
    if pixels.shape != shape or pixels.dtype != np.uint8:
        raise AuditError("Product pixel shape/dtype mismatch.")
    return pixels


def first_source(manifest_path, inventory_path, source_dir, out):
    """Actually build and independently compare the fixed first selected source."""
    manifest, inventory = read_json(manifest_path), read_json(inventory_path)
    verify_manifest(manifest, inventory)
    row = manifest["worlds"][0]
    path = _path(source_dir, row["path"])
    root = Path(out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("First-source output must be empty.")
    root.mkdir(parents=True, exist_ok=True)
    source, checksum, _ = source_pixels(path, row)
    record = build_world(path, row, root / "worlds")
    pixels = _png(root / "worlds" / record["path"], (2048, 2048, 3))
    if not np.array_equal(pixels, independent_world(source)):
        raise AuditError("Independent first-source world pixels differ.")
    initial_hashes = {}
    for oi, (x, y) in enumerate(ORIGINS):
        initial = pixels[y-120:y+120, x-208:x+208]
        target = root / f"I0-o{oi}.png"
        Image.fromarray(initial).save(target)
        if not np.array_equal(_png(target, (240, 416, 3)), initial):
            raise AuditError("First-source I0 differs.")
        initial_hashes[target.name] = file_hash(target)
    report = {"schema": "rswm-first-source-audit-v1", "status": "FIRST_SOURCE_PIXELS_VALIDATED",
              "scope": "one fixed selected source; not smoke12 or geographic validation",
              "manifest_sha256": manifest["manifest_sha256"], "inventory_sha256": file_hash(inventory_path),
              "repo_id": manifest["repo_id"], "revision": manifest["revision"], "world_id": row["world_id"],
              "source_path": row["path"], "source_sha256": checksum, "source_format": "TIFF",
              "source_shape": [5000, 5000, 3], "source_dtype": "uint8", "single_page": True,
              "world_pixel_sha256": record["world_pixel_sha256"], "initial_file_sha256": initial_hashes,
              "real_model_inference": False, "geographic_relationship": "unknown"}
    write_json(root / "first-source-audit.json", report)
    return report


def audit_materialized(build_root, manifest_path, inventory_path, source_dir):
    from .hf_materialize import select_profile
    from .materialized_geometry import audit_task_geometry
    from .suite import code_fingerprint
    started = time.monotonic()
    root = Path(build_root)
    if root.is_symlink():
        raise AuditError("Build root is a symlink.")
    manifest, inventory = read_json(manifest_path), read_json(inventory_path)
    verify_manifest(manifest, inventory)
    expected_files = {"build-selection.json", "BUILD_COMPLETE.json", "private/index.json", "private/training-clips.json"}
    selection = read_json(_path(root, "build-selection.json"))
    profile = selection.get("profile")
    if profile in ("smoke12", "core120", "pilot-dev2"):
        selected = select_profile(manifest, profile)
        if len(selected) != {"smoke12": 12, "core120": 120, "pilot-dev2": 2}[profile]:
            raise AuditError("Wrong frozen profile quotas.")
        input_kind = selection.get("input_kind", "source_identity_checked_geography_unknown")
        if input_kind == "synthetic_fixture":
            raise AuditError("Synthetic fixtures cannot claim a real profile.")
    elif profile == "synthetic_fixture" and selection.get("input_kind") == "synthetic_fixture":
        selected = selection.get("worlds", [])
        if not selected or any(row not in manifest["worlds"] for row in selected):
            raise AuditError("Synthetic fixture must explicitly identify frozen manifest rows.")
        input_kind = "synthetic_fixture"
    else:
        raise AuditError("Unknown materialization profile.")
    if selection.get("worlds") != selected or len({r["world_id"] for r in selected}) != len(selected):
        raise AuditError("Selection differs from frozen profile or repeats a world.")
    expected_selection = {"core_manifest_sha256": manifest["manifest_sha256"], "repo_id": manifest["repo_id"],
                          "revision": manifest["revision"], "profile": profile,
                          "full_core": profile == "core120", "source_bytes": sum(r["size_bytes"] for r in selected),
                          "worlds": selected}
    if {k: v for k, v in selection.items() if k != "input_kind"} != expected_selection:
        raise AuditError("Build selection metadata mismatch.")
    index = read_json(_path(root, "private/index.json"))
    if set(index) != {"tasks", "worlds", "source_manifest_sha256"} or index["source_manifest_sha256"] != manifest["manifest_sha256"]:
        raise AuditError("Private index schema/manifest mismatch.")
    if len(index["worlds"]) != len(selected) or len({r["world_id"] for r in index["worlds"]}) != len(selected):
        raise AuditError("World index omission/duplicate.")
    worlds = {r["world_id"]: r for r in index["worlds"]}
    expected_index, expected_clips, source_hashes, pixel_hashes, task_bindings = [], [], {}, {}, {}
    counts = {"sources": 0, "worlds": 0, "initial_images": 0, "task_packages": 0,
              "private_label_arrays": 0, "independent_target_pixels": 0, "public_sessions": 0}
    for row in selected:
        wid = row["world_id"]
        source_path = _path(source_dir, row["path"])
        source, source_hash, geotags = source_pixels(source_path, row)
        source_hashes[row["path"]] = source_hash
        pixels = independent_world(source)
        del source
        png_name, side_name = f"private/worlds/{wid}.png", f"private/worlds/{wid}.json"
        expected_files.update((png_name, side_name))
        world_path = _path(root, png_name)
        actual = _png(world_path, (2048, 2048, 3))
        if not np.array_equal(actual, pixels):
            raise AuditError("Source-to-world pixel mismatch: " + wid)
        record = read_json(_path(root, side_name))
        expected_record = {**row, "source_path": row["path"], "source_sha256": source_hash,
                           "world_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                           "world_file_sha256": file_hash(world_path), "shape": [2048, 2048, 3],
                           "source_crop_xyxy": [452, 452, 4548, 4548], "resampling": "uint16 sum 2x2; (sum+2)//4",
                           "nominal_gsd_m": .6, "nominal_extent_m": [1228.8, 1228.8],
                           "source_geotiff_tags": geotags,
                           "georeference_status": "raw tags retained; not geospatially validated" if geotags else "not available",
                           "path": wid + ".png"}
        if record != expected_record or worlds.get(wid) != expected_record:
            raise AuditError("World sidecar/index/source binding mismatch.")
        pixel_hashes[wid] = record["world_pixel_sha256"]
        counts["sources"] += 1; counts["worlds"] += 1
        for oi, (x, y) in enumerate(ORIGINS):
            iid = digest([wid, oi, "initial"])[:24]
            initial_name = "public/initial/" + iid + ".png"
            expected_files.add(initial_name)
            initial = _png(_path(root, initial_name), (240, 416, 3))
            if not np.array_equal(initial, pixels[y-120:y+120, x-208:x+208]):
                raise AuditError("World-to-I0 pixel mismatch.")
            counts["initial_images"] += 1
            # packages binds the exact frozen task identity/queries; independent
            # label validation below does not reuse packages' computed labels.
            for expected_task, _ in packages(wid, row["role"], oi):
                expected_task = json.loads(json.dumps(expected_task))
                tid = expected_task["task_id"]
                task_name = "private/tasks/" + tid + ".json"
                expected_files.add(task_name)
                task = read_json(_path(root, task_name))
                private = task.get("private_label_files", {})
                if {k: v for k, v in task.items() if k != "private_label_files"} != expected_task:
                    raise AuditError("Task identity/queries/metadata mismatch.")
                labels = {}
                expected_streams = set(task["programs"]) if task["family"] in ("F5", "F7") else set()
                if set(private) != expected_streams:
                    raise AuditError("Private label stream inventory mismatch.")
                for name, info in private.items():
                    label_name = "private/tasks/labels/" + tid + "-" + name + ".npz"
                    expected_files.add(label_name)
                    if set(info) != {"path", "sha256"} or info["path"] != "labels/" + tid + "-" + name + ".npz":
                        raise AuditError("Private label path schema mismatch.")
                    path = _path(root, label_name)
                    if file_hash(path) != info["sha256"]:
                        raise AuditError("Private label file hash mismatch.")
                    try:
                        with np.load(path, allow_pickle=False) as z:
                            if len(z.files) != len(set(z.files)):
                                raise AuditError("Duplicate private NPZ keys.")
                            labels[name] = {key: z[key] for key in z.files}
                    except (OSError, ValueError, EOFError) as exc:
                        raise AuditError("Invalid private NPZ.") from exc
                    if any(a.shape != (128, 128) or a.dtype.kind not in "iu" for a in labels[name].values()):
                        raise AuditError("Private NPZ array shape/dtype mismatch.")
                    counts["private_label_arrays"] += len(labels[name])
                    counts["independent_target_pixels"] += 128 * 128
                audit_task_geometry(task, labels)
                task_bindings[tid] = {"task_sha256": digest(task), "world_id": wid, "role": row["role"],
                                      "origin_index": oi, "family": task["family"], "protocol": task["protocol"]}
                sid = digest([tid, "public-session"])[:24]
                session_name = "public/sessions/" + sid + ".json"
                expected_files.add(session_name)
                public = read_json(_path(root, session_name))
                expected_public = {"session_id": sid, "initial_image": "../initial/" + iid + ".png",
                                   "viewport_hw": [240, 416], "bounds_xy": [[-704, 704], [-704, 704]],
                                   "coordinate_normalizer": float(np.hypot(416, 240)), "q0": [0, 0],
                                   "protocol": task["protocol"], "reset_between_streams": task["family"] == "F3"}
                if task["protocol"] == "offline":
                    expected_public["programs"] = task["programs"]
                else:
                    expected_public["query_delivery"] = "evaluator sends current 16-query call only"
                if public != expected_public:
                    raise AuditError("Public session whitelist/queries/I0 binding mismatch.")
                expected_index.append({"task_id": tid, "session_id": sid, "family": task["family"],
                                       "world_id": wid, "role": row["role"], "city": row["city"],
                                       "task_path": "tasks/" + tid + ".json", "world_path": "worlds/" + wid + ".png"})
                counts["task_packages"] += 1; counts["public_sessions"] += 1
        if row["role"] == "train":
            expected_clips.extend(training_clips(wid, row["role"]))
    if index["tasks"] != expected_index:
        raise AuditError("Task index has omitted, extra, repeated or reordered entries.")
    clips = read_json(_path(root, "private/training-clips.json"))
    # Convert tuples to their JSON list representation, without altering generation.
    normalized_clips = json.loads(json.dumps(expected_clips))
    if clips != {"clips": normalized_clips, "clip_count": len(expected_clips), "view_events": 33 * len(expected_clips)}:
        raise AuditError("Training clip inventory/roles/coordinates mismatch.")
    completion = read_json(_path(root, "BUILD_COMPLETE.json"))
    expected_completion = {"profile": profile, "source_bytes": selection["source_bytes"], "worlds": len(selected),
                           "initial_images": 4 * len(selected), "task_packages": 16 * len(selected),
                           "query_outputs_all_roles": len(selected) * 4 * 352,
                           "real_model_inference": False, "public_export_is_security_sandbox": False}
    if completion != expected_completion:
        raise AuditError("BUILD_COMPLETE counts/status mismatch.")
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AuditError("Build inventory contains a symlink.")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise AuditError("Complete artifact inventory mismatch: " + json.dumps({"missing": sorted(expected_files-actual_files), "extra": sorted(actual_files-expected_files)}))
    duplicates = {}
    for wid, h in pixel_hashes.items():
        duplicates.setdefault(h, []).append(wid)
    duplicate_groups = [ids for ids in duplicates.values() if len(ids) > 1]
    report = {"schema": "rswm-materialized-audit-v1", "status": "SYNTHETIC_FIXTURE_VALIDATED" if input_kind == "synthetic_fixture" else "MATERIALIZED_PIXELS_VALIDATED",
              "profile": profile, "input_kind": input_kind, "manifest_sha256": manifest["manifest_sha256"],
              "inventory_sha256": file_hash(inventory_path), "repo_id": manifest["repo_id"], "revision": manifest["revision"],
              "source_code_sha256": code_fingerprint(),
              "counts": {**counts, "training_clips": len(expected_clips), "training_view_events": 33*len(expected_clips),
                         "planned_unique_outputs_per_model_seed": len(selected)*4*352},
              "source_sha256": source_hashes, "world_pixel_sha256": pixel_hashes,
              "worlds": {wid: {k: row[k] for k in ("world_id", "city", "role", "source_path", "source_sha256",
                                                   "world_file_sha256", "world_pixel_sha256")}
                         for wid, row in worlds.items()}, "tasks": task_bindings,
              "artifact_sha256": {p: file_hash(root/p) for p in sorted(expected_files)},
              "semantic_sha256": digest({"world_pixels": pixel_hashes, "tasks": expected_index, "clips": normalized_clips}),
              "exact_duplicate_world_pixel_groups": duplicate_groups, "geographic_relationship": "unknown",
              "exact_deduplication_proves_geographic_independence": False,
              "real_model_inference": False, "real_gpu_validation": "NOT_RUN", "full_smoke12_diagnostic_matrix": "NOT_RUN",
              "environment": {"python": sys.version, "numpy": np.__version__, "pillow": Image.__version__, "platform": platform.platform()},
              "elapsed_seconds": time.monotonic()-started}
    # Report duplicates rather than silently changing the preselected sample.
    report["acceptance_status"] = "DUPLICATE_PIXELS_REVIEW_REQUIRED" if duplicate_groups else "ACCEPTED_IN_STATED_SCOPE"
    if profile == "pilot-dev2":
        report["scope"] = "fixed Kitsap dev two-world data audit; not smoke12, core120, training or a real generation pilot"
    return report


def validate_audit_receipt(report, plan, *, required_profile=None):
    """Check a real-data audit's exact bindings before pilot/formal execution.

    This is provenance validation, not signature authentication. The producer
    must obtain the receipt from audit_materialized; callers still validate the
    suite's immutable physical input hashes with validate_plan before execution.
    """
    if not isinstance(report, dict) or report.get("schema") != "rswm-materialized-audit-v1":
        raise AuditError("A current materialized audit receipt is required.")
    if (report.get("status") != "MATERIALIZED_PIXELS_VALIDATED" or
        report.get("input_kind") != "source_identity_checked_geography_unknown" or
        report.get("acceptance_status") != "ACCEPTED_IN_STATED_SCOPE"):
        raise AuditError("A synthetic, failed or thin source receipt cannot authorize real data.")
    profile = report.get("profile")
    if profile not in ("smoke12", "core120", "pilot-dev2") or (required_profile is not None and profile != required_profile):
        raise AuditError("Required materialized data profile differs.")
    if {row.get("source_manifest_sha256") for row in plan.get("expected", [])} != {report.get("manifest_sha256")}:
        raise AuditError("Materialized receipt and suite source manifest differ.")
    if report.get("source_code_sha256") != plan.get("source_code_sha256"):
        raise AuditError("Materialized receipt and suite source code differ.")
    n = {"smoke12": 12, "core120": 120, "pilot-dev2": 2}[profile]
    train_worlds = {"smoke12": 9, "core120": 60, "pilot-dev2": 0}[profile]
    counts = report.get("counts", {})
    for key, value in {"sources": n, "worlds": n, "initial_images": 4*n, "task_packages": 16*n,
                       "public_sessions": 16*n, "training_clips": 64*train_worlds,
                       "training_view_events": 2112*train_worlds,
                       "planned_unique_outputs_per_model_seed": 1408*n}.items():
        if type(counts.get(key)) is not int or counts[key] != value:
            raise AuditError("Materialized receipt profile counts disagree.")
    worlds, tasks, sources = report.get("worlds"), report.get("tasks"), report.get("source_sha256")
    if not isinstance(worlds, dict) or not isinstance(tasks, dict) or not isinstance(sources, dict) or len(worlds) != n or len(tasks) != 16*n or len(sources) != n:
        raise AuditError("Materialized receipt does not enumerate all source/world/task bindings.")
    if profile == "pilot-dev2":
        from .data import REPO, INRIA_REVISION
        from .hf_materialize import PILOT_DEV2_SOURCES
        if report.get("repo_id") != REPO or report.get("revision") != INRIA_REVISION:
            raise AuditError("pilot-dev2 receipt differs from the frozen source revision.")
        fixed_worlds = {row["world_id"]: row for row in PILOT_DEV2_SOURCES}
        if set(worlds) != set(fixed_worlds) or sources != {r["path"]: r["lfs_sha256"] for r in PILOT_DEV2_SOURCES}:
            raise AuditError("pilot-dev2 receipt differs from the fixed dev sources.")
        expected_tasks = {f"{wid}-o{oi}-{family}": (wid, oi, family)
                          for wid in fixed_worlds for oi in range(4) for family in ("F2", "F3", "F5", "F7")}
        if set(tasks) != set(expected_tasks):
            raise AuditError("pilot-dev2 receipt differs from the complete dev task set.")
        for wid, fixed in fixed_worlds.items():
            world = worlds[wid]
            if any(world.get(key) != value for key, value in
                   {"source_path": fixed["path"], "source_sha256": fixed["lfs_sha256"],
                    "role": "dev", "city": "kitsap"}.items()):
                raise AuditError("pilot-dev2 world/source/dev role differs.")
        for tid, (wid, oi, family) in expected_tasks.items():
            task = tasks[tid]
            if any(task.get(key) != value or type(task.get(key)) is not type(value) for key, value in
                   {"world_id": wid, "origin_index": oi, "family": family, "role": "dev",
                    "protocol": "online16" if family in ("F5", "F7") else "offline"}.items()):
                raise AuditError("pilot-dev2 task identity differs.")
    def sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    if not sha(report.get("inventory_sha256")) or any(not sha(v) for v in sources.values()):
        raise AuditError("Complete frozen inventory/source hashes are required.")
    artifacts = report.get("artifact_sha256", {})
    if not isinstance(artifacts, dict) or any(not sha(v) for v in artifacts.values()):
        raise AuditError("Materialized artifact hashes are invalid.")
    for wid, world in worlds.items():
        if world.get("world_id") != wid or sources.get(world.get("source_path")) != world.get("source_sha256"):
            raise AuditError("World/source binding is inconsistent.")
        if artifacts.get("private/worlds/"+wid+".png") != world.get("world_file_sha256") or not sha(world.get("world_pixel_sha256")):
            raise AuditError("World artifact/pixel hash binding is incomplete.")
    for row in plan.get("expected", []):
        world, task = worlds.get(row.get("world_id")), tasks.get(row.get("task_id"))
        if not isinstance(world, dict) or not isinstance(task, dict):
            raise AuditError("A planned world/task is absent from materialized audit.")
        for key in ("city", "role", "world_file_sha256", "world_pixel_sha256"):
            if world.get(key) != row.get(key):
                raise AuditError("Planned world/city/role/pixels differ from materialized audit.")
        for key in ("task_sha256", "world_id", "role", "origin_index", "family", "protocol"):
            if task.get(key) != row.get(key):
                raise AuditError("Planned task identity differs from materialized audit.")
    if not plan.get("expected"):
        raise AuditError("Empty plans cannot consume a materialized receipt.")
    return {"status": "MATERIALIZED_RECEIPT_BOUND", "profile": profile,
            "manifest_sha256": report["manifest_sha256"], "source_code_sha256": report["source_code_sha256"],
            "matched_planned_tasks": len({r["task_id"] for r in plan["expected"]}),
            "real_model_validation": False, "geographic_relationship": "unknown"}


def compare_materialized_semantics(first, second):
    """Compare declared semantic fingerprints, independent of NPZ timestamps."""
    return {"equal": first["semantic_sha256"] == second["semantic_sha256"],
            "first_semantic_sha256": first["semantic_sha256"], "second_semantic_sha256": second["semantic_sha256"],
            "container_hashes_may_differ": True}


def audit_main(argv=None):
    p = argparse.ArgumentParser(description="Independently audit materialized pixels, task labels and full inventories.")
    p.add_argument("--build", required=True); p.add_argument("--manifest", required=True); p.add_argument("--inventory", required=True)
    p.add_argument("--source-dir", required=True); p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        report = audit_materialized(a.build, a.manifest, a.inventory, a.source_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report = {"schema": "rswm-materialized-audit-v1", "status": "BLOCKED" if isinstance(exc, FileNotFoundError) or str(exc).startswith("Missing artifact:") else "FAILED",
                  "error_type": type(exc).__name__, "reason": str(exc), "real_data_validated": False}
        write_json(a.out, report); print(json.dumps(report)); return 2
    write_json(a.out, report); print(json.dumps({"status": report["status"], "counts": report["counts"]}))
    return 0 if report["acceptance_status"] == "ACCEPTED_IN_STATED_SCOPE" else 1


if __name__ == "__main__":
    raise SystemExit(audit_main())
