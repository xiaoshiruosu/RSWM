"""Fixed dev data-profile software checks; synthetic pixels are never real data."""
import copy
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from rswm_inria import data
from rswm_inria.geometry import digest, file_hash
from rswm_inria.hf_materialize import PILOT_DEV2_SOURCES, materialize_inria, select_profile
from rswm_inria.materialized import AuditError, audit_materialized, validate_audit_receipt


def parent_inventory():
    """Complete metadata fixture, with the actual two fixed dev identities."""
    fixed = {row["path"]: row for row in PILOT_DEV2_SOURCES}
    files = []
    for city in data.ROLES:
        split = "train" if city in ("austin", "chicago", "vienna", "kitsap") else "test"
        for j in range(1, 37):
            path = f"data/{split}/images/{city}{j}.tif"
            binding = fixed.get(path, {})
            files.append({"path": path, "size_bytes": binding.get("size_bytes", 75000000),
                          "lfs_sha256": binding.get("lfs_sha256", hashlib.sha256(path.encode()).hexdigest())})
    return {"repo_id": data.REPO, "revision": data.INRIA_REVISION, "files": files}


def test_fixed_selection_uses_full_parent_and_preserves_source_split_dev_role():
    manifest = data.select_manifest(parent_inventory())
    assert len(manifest["worlds"]) == 120
    rows = select_profile(manifest, "pilot-dev2")
    assert [{key: row[key] for key in fixed} for row, fixed in zip(rows, PILOT_DEV2_SOURCES)] == list(PILOT_DEV2_SOURCES)
    assert sum(row["size_bytes"] for row in rows) == 150080706
    assert all(row["source_split"] == "train" and row["role"] == "dev" for row in rows)
    assert len(select_profile(manifest, "smoke12")) == 12
    assert len(select_profile(manifest, "core120")) == 120


@pytest.mark.parametrize("field,value", [("path", "data/train/images/kitsap14.tif"),
    ("world_id", "another-world"), ("lfs_sha256", "a" * 64), ("size_bytes", 1),
    ("role", "train"), ("selection_rank", 3), ("selection_rank", True), ("source_split", "test")])
def test_fixed_selection_does_not_substitute_or_relabel(field, value):
    manifest = data.select_manifest(parent_inventory())
    row = next(row for row in manifest["worlds"] if row["path"] == PILOT_DEV2_SOURCES[0]["path"])
    row[field] = value
    with pytest.raises(ValueError, match="fixed Kitsap"):
        select_profile(manifest, "pilot-dev2")


def test_mutable_or_different_revision_and_reversed_dev_rows_rejected():
    manifest = data.select_manifest(parent_inventory())
    changed = copy.deepcopy(manifest); changed["revision"] = "main"
    with pytest.raises(ValueError, match="revision"):
        select_profile(changed, "pilot-dev2")
    changed = copy.deepcopy(manifest); changed["worlds"].reverse()
    with pytest.raises(ValueError, match="fixed Kitsap"):
        select_profile(changed, "pilot-dev2")


def test_budget_and_incomplete_parent_rejected_before_download_or_build(tmp_path):
    inventory = parent_inventory(); manifest = data.select_manifest(inventory)
    rows = select_profile(manifest, "pilot-dev2")
    with pytest.raises(ValueError, match="budget"):
        materialize_inria(manifest, inventory, rows, tmp_path / "sources", tmp_path / "build",
                          "pilot-dev2", .150080705, download=True)
    assert not (tmp_path / "build").exists()
    partial = copy.deepcopy(manifest); partial["worlds"] = rows
    partial.pop("manifest_sha256"); partial["manifest_sha256"] = digest(partial)
    with pytest.raises(ValueError, match="120"):
        materialize_inria(partial, inventory, rows, tmp_path / "sources", tmp_path / "partial",
                          "pilot-dev2", 1.)
    assert not (tmp_path / "partial").exists()
    for profile in ("smoke12", "core120"):
        with pytest.raises(ValueError, match="quotas"):
            materialize_inria(manifest, inventory, rows, tmp_path / "sources", tmp_path / profile,
                              profile, 10.)
        assert not (tmp_path / profile).exists()


@pytest.fixture(scope="module")
def synthetic_dev2(tmp_path_factory):
    root = tmp_path_factory.mktemp("explicit_synthetic_dev2")
    inventory = parent_inventory()
    for i, fixed in enumerate(PILOT_DEV2_SOURCES):
        target = root / "sources" / fixed["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.zeros((5000, 5000, 3), np.uint8)
        pixels[:, :, 0] = np.arange(5000, dtype=np.uint16)[None, :] % 256
        pixels[:, :, 1] = np.arange(5000, dtype=np.uint16)[:, None] % 256
        pixels[:, :, 2] = 173 + i
        Image.fromarray(pixels).save(target, compression="tiff_deflate")
        for row in inventory["files"]:
            if row["path"] == fixed["path"]:
                row.update(size_bytes=target.stat().st_size, lfs_sha256=file_hash(target))
    manifest = data.select_manifest(inventory)
    rows = [row for row in manifest["worlds"] if row["role"] == "dev" and row["selection_rank"] <= 2]
    data.write_json(root / "manifest.json", manifest); data.write_json(root / "inventory.json", inventory)
    materialize_inria(manifest, inventory, rows, root / "sources", root / "build", "synthetic_fixture", 1.)
    return root


def test_two_dev_pixel_i0_task_label_audit_remains_synthetic_and_cannot_be_relabelled(synthetic_dev2):
    root = synthetic_dev2
    report = audit_materialized(root / "build", root / "manifest.json", root / "inventory.json", root / "sources")
    assert report["status"] == "SYNTHETIC_FIXTURE_VALIDATED"
    assert report["counts"]["worlds"] == 2 and report["counts"]["initial_images"] == 8
    assert report["counts"]["task_packages"] == 32
    assert report["counts"]["training_clips"] == report["counts"]["training_view_events"] == 0
    assert report["counts"]["planned_unique_outputs_per_model_seed"] == 2816
    with pytest.raises(AuditError, match="synthetic"):
        validate_audit_receipt(report, {"expected": []})
    selection = root / "build" / "build-selection.json"
    original = selection.read_bytes()
    value = json.loads(original); value.update(profile="pilot-dev2", input_kind="source_identity_checked_geography_unknown")
    selection.write_text(json.dumps(value))
    try:
        with pytest.raises(ValueError, match="fixed Kitsap"):
            audit_materialized(root / "build", root / "manifest.json", root / "inventory.json", root / "sources")
    finally:
        selection.write_bytes(original)


def schema_control():
    """Binding control only; no TIFF/PNG or model is attested by this fixture."""
    worlds = {}; sources = {}; tasks = {}; artifacts = {}; planned = []
    for fixed in PILOT_DEV2_SOURCES:
        wid = fixed["world_id"]
        world = {"world_id": wid, "source_path": fixed["path"], "source_sha256": fixed["lfs_sha256"],
                 "city": "kitsap", "role": "dev", "world_file_sha256": digest([wid, "schema PNG"]),
                 "world_pixel_sha256": digest([wid, "schema pixels"])}
        worlds[wid] = world; sources[fixed["path"]] = fixed["lfs_sha256"]
        artifacts["private/worlds/" + wid + ".png"] = world["world_file_sha256"]
        for oi in range(4):
            for family in ("F2", "F3", "F5", "F7"):
                tid = f"{wid}-o{oi}-{family}"
                task = {"world_id": wid, "origin_index": oi, "family": family, "role": "dev",
                        "protocol": "online16" if family in ("F5", "F7") else "offline",
                        "task_sha256": digest([tid, "schema task"])}
                tasks[tid] = task
                if oi == 0:
                    planned.append({**task, "task_id": tid, "city": "kitsap", "source_manifest_sha256": "a" * 64,
                                    "world_file_sha256": world["world_file_sha256"], "world_pixel_sha256": world["world_pixel_sha256"]})
    report = {"schema": "rswm-materialized-audit-v1", "status": "MATERIALIZED_PIXELS_VALIDATED",
              "input_kind": "source_identity_checked_geography_unknown", "acceptance_status": "ACCEPTED_IN_STATED_SCOPE",
              "profile": "pilot-dev2", "repo_id": data.REPO, "revision": data.INRIA_REVISION,
              "manifest_sha256": "a" * 64, "inventory_sha256": "b" * 64, "source_code_sha256": "c" * 64,
              "counts": {"sources": 2, "worlds": 2, "initial_images": 8, "task_packages": 32,
                         "public_sessions": 32, "training_clips": 0, "training_view_events": 0,
                         "planned_unique_outputs_per_model_seed": 2816},
              "worlds": worlds, "source_sha256": sources, "tasks": tasks, "artifact_sha256": artifacts}
    return report, {"expected": planned, "source_code_sha256": "c" * 64}


def test_subset_receipt_binds_pilot_tasks_but_never_replaces_full_profiles():
    report, plan = schema_control()
    result = validate_audit_receipt(report, plan)
    assert result["profile"] == "pilot-dev2" and result["matched_planned_tasks"] == 8
    assert result["real_model_validation"] is False
    for required in ("core120", "smoke12"):
        with pytest.raises(AuditError, match="profile"):
            validate_audit_receipt(report, plan, required_profile=required)
        relabelled = copy.deepcopy(report); relabelled["profile"] = required
        with pytest.raises(AuditError, match="counts"):
            validate_audit_receipt(relabelled, plan, required_profile=required)


@pytest.mark.parametrize("fault", ["source", "world", "dev_role", "missing_task", "unplanned_task", "boolean_origin", "training", "revision"])
def test_subset_receipt_rejects_wrong_source_role_tasks_training_and_revision(fault):
    report, plan = schema_control()
    first = PILOT_DEV2_SOURCES[0]; wid = first["world_id"]
    if fault == "source": report["source_sha256"][first["path"]] = "d" * 64
    elif fault == "world": report["worlds"]["another-world"] = report["worlds"].pop(wid)
    elif fault == "dev_role": report["worlds"][wid]["role"] = "train"
    elif fault == "missing_task": report["tasks"].pop(next(iter(report["tasks"])))
    elif fault == "unplanned_task":
        tid = f"{wid}-o3-F7"; report["tasks"]["other-task"] = report["tasks"].pop(tid)
    elif fault == "boolean_origin": report["tasks"][f"{wid}-o1-F7"]["origin_index"] = True
    elif fault == "training": report["counts"]["training_clips"] = 1
    else: report["revision"] = "d" * 40
    with pytest.raises(AuditError): validate_audit_receipt(report, plan)
