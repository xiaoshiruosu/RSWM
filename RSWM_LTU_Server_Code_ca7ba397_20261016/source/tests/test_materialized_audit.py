"""Synthetic files exercise the full G2 boundary; they are not real Inria."""
import copy
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from rswm_inria import data
from rswm_inria.data import ROLES, REPO, build_world, freeze_metadata, source_pixels, write_json
from rswm_inria.geometry import file_hash
from rswm_inria.hf_materialize import materialize_inria
from rswm_inria.materialized import AuditError, audit_materialized, compare_materialized_semantics, first_source, audit_main
from rswm_inria.smoke12 import accept_smoke12, DIAGNOSTIC_MODELS


def synthetic_inventory():
    rows = []
    for city in ROLES:
        split = "train" if city in ("austin", "chicago", "vienna", "kitsap") else "test"
        for j in range(1, 37):
            path = f"data/{split}/images/{city}{j}.tif"
            rows.append({"path": path, "size_bytes": 75000000, "lfs_sha256": hashlib.sha256(path.encode()).hexdigest()})
    return {"repo_id": REPO, "revision": data.INRIA_REVISION, "files": rows}


@pytest.fixture(scope="module")
def materialized_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("g2_explicit_synthetic_fixture")
    inventory = synthetic_inventory()
    first = data.select_manifest(inventory)["worlds"][0]
    path = root/"sources"/first["path"]
    path.parent.mkdir(parents=True)
    pixels = np.zeros((5000, 5000, 3), np.uint8)
    pixels[:, :, 0] = np.arange(5000, dtype=np.uint16)[None, :] % 256
    pixels[:, :, 1] = np.arange(5000, dtype=np.uint16)[:, None] % 256
    pixels[:, :, 2] = 173
    Image.fromarray(pixels).save(path, compression="tiff_deflate")
    for row in inventory["files"]:
        if row["path"] == first["path"]:
            row.update(size_bytes=path.stat().st_size, lfs_sha256=file_hash(path))
    manifest = data.select_manifest(inventory)
    first = manifest["worlds"][0]
    write_json(root/"manifest.json", manifest); write_json(root/"inventory.json", inventory)
    materialize_inria(manifest, inventory, [first], root/"sources", root/"build", "synthetic_fixture", 1.)
    return root


def run_audit(root, build=None):
    return audit_materialized(build or root/"build", root/"manifest.json", root/"inventory.json", root/"sources")


def test_actual_subset_success_receipt_retains_canonical_empty_failure_list(materialized_fixture,tmp_path):
    """Run the real producer on synthetic files; this is never real smoke12."""
    root=materialized_fixture
    report=accept_smoke12(root/"build",root/"manifest.json",root/"inventory.json",root/"sources",tmp_path/"subset",
                          code_commit="b"*40,task_limit=1,max_output_gb=.1,execute=True)
    assert report["status"]=="ENGINEERING_SUBSET_VALIDATED" and report["real_data_validated"] is False
    assert type(report["failed_runs"]) is list and report["failed_runs"]==[]


def test_all_products_independent_pixels_geometry_inventory_and_train(materialized_fixture):
    report = run_audit(materialized_fixture)
    assert report["status"] == "SYNTHETIC_FIXTURE_VALIDATED"
    assert report["counts"] == {"sources": 1, "worlds": 1, "initial_images": 4, "task_packages": 16,
        "private_label_arrays": 84, "independent_target_pixels": 196608, "public_sessions": 16,
        "training_clips": 64, "training_view_events": 2112, "planned_unique_outputs_per_model_seed": 1408}
    assert report["geographic_relationship"] == "unknown"
    assert report["full_smoke12_diagnostic_matrix"] == "NOT_RUN"


def test_fixed_first_source_actually_builds_and_checks_four_i0(materialized_fixture):
    root = materialized_fixture
    result = first_source(root/"manifest.json", root/"inventory.json", root/"sources", root/"first-source")
    assert result["source_format"] == "TIFF" and len(result["initial_file_sha256"]) == 4
    assert result["world_id"] == data.select_manifest(synthetic_inventory())["worlds"][0]["world_id"]
    assert result["status"] == "FIRST_SOURCE_PIXELS_VALIDATED"


@pytest.mark.parametrize("kind", ["world_pixel", "initial_pixel", "label_pixel_rehashed", "label_extra_key_rehashed",
    "label_missing_key_rehashed", "online_future_leak", "index_duplicate", "world_duplicate", "clip_duplicate",
    "dev_training", "missing_file", "extra_file", "completion_count", "label_path_escape", "symlink"])
def test_tampered_products_fail_even_with_updated_container_hash(materialized_fixture, tmp_path, kind):
    source = materialized_fixture/"build"; build = tmp_path/"build"; shutil.copytree(source, build)
    def edit(path, mutate):
        value = json.loads(path.read_text()); mutate(value); path.write_text(json.dumps(value))
    if kind in ("world_pixel", "initial_pixel"):
        p = next((build/("private/worlds" if kind == "world_pixel" else "public/initial")).glob("*.png"))
        with Image.open(p) as im: pixels = np.asarray(im).copy()
        pixels[0,0,0] ^= 1; Image.fromarray(pixels).save(p)
    elif kind.startswith("label_") and kind != "label_path_escape":
        tp = next(p for p in (build/"private/tasks").glob("*F5.json"))
        task = json.loads(tp.read_text()); info = task["private_label_files"]["main"]
        p = tp.parent/info["path"]
        with np.load(p, allow_pickle=False) as z: arrays = {k:z[k] for k in z.files}
        if kind == "label_pixel_rehashed": arrays["first_event"][0,0] += 1
        elif kind == "label_extra_key_rehashed": arrays["future_secret"] = np.zeros((128,128), np.int16)
        else: arrays.pop("reentry_event")
        np.savez_compressed(p, **arrays); info["sha256"] = file_hash(p); tp.write_text(json.dumps(task))
    elif kind == "online_future_leak":
        p = next(p for p in (build/"public/sessions").glob("*.json") if json.loads(p.read_text())["protocol"] == "online16")
        edit(p, lambda v:v.update(programs={"future":[[0,0]]}))
    elif kind == "index_duplicate": edit(build/"private/index.json", lambda v:v["tasks"].append(v["tasks"][0]))
    elif kind == "world_duplicate": edit(build/"private/index.json", lambda v:v["worlds"].append(v["worlds"][0]))
    elif kind == "clip_duplicate": edit(build/"private/training-clips.json", lambda v:v["clips"].append(v["clips"][0]))
    elif kind == "dev_training": edit(build/"private/training-clips.json", lambda v:v["clips"][0].update(world_id="dev-world"))
    elif kind == "missing_file": next((build/"public/initial").glob("*.png")).unlink()
    elif kind == "extra_file": (build/"private/hidden.bin").write_bytes(b"undeclared")
    elif kind == "completion_count": edit(build/"BUILD_COMPLETE.json", lambda v:v.update(task_packages=17))
    elif kind == "label_path_escape":
        p = next((build/"private/tasks").glob("*F5.json"))
        edit(p, lambda v:v["private_label_files"]["main"].update(path="../../../../secret.npz"))
    else:
        p = next((build/"public/initial").glob("*.png")); external = tmp_path/"external.png"; p.rename(external); p.symlink_to(external)
    with pytest.raises((AuditError, ValueError)): run_audit(materialized_fixture, build)


def test_npz_container_reencoding_preserves_semantic_identity(materialized_fixture, tmp_path):
    root = materialized_fixture; build = tmp_path/"build"; shutil.copytree(root/"build", build)
    tp = next((build/"private/tasks").glob("*F5.json")); task = json.loads(tp.read_text()); info = task["private_label_files"]["main"]
    p = tp.parent/info["path"]
    with np.load(p, allow_pickle=False) as z: arrays = {k:z[k] for k in z.files}
    old = file_hash(p); np.savez(p, **arrays); assert file_hash(p) != old
    info["sha256"] = file_hash(p); tp.write_text(json.dumps(task))
    assert compare_materialized_semantics(run_audit(root), run_audit(root, build))["equal"]


@pytest.mark.parametrize("kind", ["disguised_png", "multipage", "rgba", "grayscale", "uint16_rgb", "empty", "truncated", "wrong_size", "wrong_sha"])
def test_original_tiff_container_fail_closed(tmp_path, kind):
    p = tmp_path/"source.tif"
    if kind == "disguised_png": Image.new("RGB", (8,8)).save(p, format="PNG")
    elif kind == "multipage": Image.new("RGB", (8,8)).save(p, format="TIFF", save_all=True, append_images=[Image.new("RGB", (8,8))])
    elif kind == "rgba": Image.new("RGBA", (8,8)).save(p)
    elif kind == "grayscale": Image.new("L", (8,8)).save(p)
    elif kind == "uint16_rgb":
        import tifffile
        tifffile.imwrite(p, np.zeros((8,8,3), np.uint16), photometric="rgb")
    elif kind == "empty": p.write_bytes(b"")
    else:
        Image.new("RGB", (5000,5000), (31,73,127)).save(p, compression="tiff_deflate")
        if kind == "truncated": p.write_bytes(p.read_bytes()[:1024])
    row = {"size_bytes":p.stat().st_size,"lfs_sha256":file_hash(p),"world_id":"synthetic-invalid"}
    if kind == "wrong_size": row["size_bytes"] += 1
    if kind == "wrong_sha": row["lfs_sha256"] = "a"*64
    with pytest.raises((ValueError, OSError)): source_pixels(p, row)


def test_freeze_rejects_mutable_revision_before_network():
    with pytest.raises(ValueError, match="immutable"): freeze_metadata("main")


def test_missing_assets_produce_real_blocked_record(tmp_path):
    out = tmp_path/"audit.json"
    assert audit_main(["--build",str(tmp_path/"missing"),"--manifest",str(tmp_path/"missing-manifest.json"),
        "--inventory",str(tmp_path/"missing-inventory.json"),"--source-dir",str(tmp_path/"missing-sources"),"--out",str(out)]) == 2
    assert json.loads(out.read_text())["status"] == "BLOCKED"


def test_static_noise_map_is_distinct_and_subset_plan_counts_actual_matrix(materialized_fixture):
    assert DIAGNOSTIC_MODELS[-1]["adapter"] == "noise-map"
    root = materialized_fixture
    report = accept_smoke12(root/"build",root/"manifest.json",root/"inventory.json",root/"sources",root/"planned-subset",
        code_commit="c"*40,task_limit=4,max_output_gb=1.,execute=False)
    assert report["planned_runs_total"] == 12 and report["planned_unique_outputs_total"] == 1056
    assert report["real_data_validated"] is False and report["full_smoke12_matrix"] is False
    assert report["status"] == "PLANNED_NOT_RUN"


def test_output_budget_stops_before_execution(materialized_fixture):
    root = materialized_fixture
    with pytest.raises(ValueError, match="budget"):
        accept_smoke12(root/"build",root/"manifest.json",root/"inventory.json",root/"sources",root/"no-budget",
            code_commit="c"*40,task_limit=4,max_output_gb=.001,execute=True)
    assert not (root/"no-budget").exists()


def test_audit_receipt_rejects_synthetic_and_thin_pass_receipts(materialized_fixture):
    from rswm_inria.materialized import validate_audit_receipt
    report = run_audit(materialized_fixture)
    with pytest.raises(AuditError, match="synthetic"):
        validate_audit_receipt(report, {"expected": []})
    with pytest.raises(AuditError, match="current"):
        validate_audit_receipt({"status":"PASS","source_manifest_sha256":"a"*64}, {"expected": []})
