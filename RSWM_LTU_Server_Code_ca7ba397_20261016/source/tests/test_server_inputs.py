"""Server preparation controls; these fixtures certify no real weights/data."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from rswm_inria import server_inputs as inputs
from rswm_inria.data import INRIA_REVISION, REPO
from rswm_inria.geometry import VERSION, digest, file_hash
from rswm_inria.hf_materialize import PILOT_DEV2_SOURCES
from rswm_inria.tasks import packages
from rswm_inria.training import locked_tree


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def local_backend(tmp_path):
    """Actual tiny bytes/configs; no tensor or vendor entrypoint is loaded."""
    source, weights = tmp_path / "prepared", tmp_path / "weights"
    source.mkdir(); weights.mkdir()
    names = ("inference.py", "base_loader.py", "wan/__init__.py", "wan/modules/model.py",
             "wan/modules/vae2_1.py", "wan/modules/t5.py", "wan/utils/cam_utils.py", "wan/configs/wan_i2v_A14B.py")
    for name in names:
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise AssertionError('Never import software-control source')\n", encoding="utf-8")
    save(source / "source_inventory.json", {name: file_hash(source / name) for name in names})
    (source / "backend.py").write_text("raise AssertionError('Never load a model during preparation')\n", encoding="utf-8")
    sampling = {"prompt": "software control", "negative_prompt": "", "steps": 1, "shift": 1.0,
                "guide_scale": [1., 1.], "boundary": .9, "param_dtype": "bf16", "t5_dtype": "bf16",
                "t5_cpu": False, "inactive_model_cpu": False}
    config = {"schema": "rswm-legacy-wan-backend-v1", "mode": "base", "config_key": "i2v-A14B",
              "sampling": sampling, "lora_subdir": None, "training": None}
    save(source / "backend_config.json", config)
    for name in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
        (weights / name).write_bytes(b"NOT_A_REAL_MODEL_TENSOR")
    save(weights / "google/umt5-xxl/tokenizer_config.json", {})
    save(weights / "google/umt5-xxl/tokenizer.json", {})
    for tag in ("low", "high"):
        save(weights / (tag + "_noise_model/config.json"), {"model_type": "i2v", "in_dim": 36})
        (weights / (tag + "_noise_model/diffusion_pytorch_model.bin")).write_bytes(b"NOT_A_REAL_EXPERT")
    camera = save(tmp_path / "camera.json", {"translation_scaling": "fixed_units", "framewise": False,
                                            "camera_channel": "native", "temporal_stride": 4})
    decisions = save(tmp_path / "decisions.json", {"approved": True, "license_record": "SOFTWARE_CONTROL_ONLY_NOT_REAL_WEIGHT_APPROVAL"})
    return SimpleNamespace(source=source, weights=weights, config=config, camera=camera,
                           decisions=decisions, root=tmp_path)


def numeric_args(backend, out):
    return dict(source_dir=backend.source, weights_dir=backend.weights, camera=backend.camera,
                decisions=backend.decisions, model_id="software-control-model", device="cuda:0", out=out)


def test_numeric_preparation_hashes_actual_files_without_loading_model(local_backend, monkeypatch):
    import rswm_inria.training as training
    monkeypatch.setattr(training, "load_approved_backend", lambda *a, **k: pytest.fail("Preparation loaded a model"))
    before_source = locked_tree(local_backend.source); before_weights = locked_tree(local_backend.weights)
    out = local_backend.root / "numeric-model.json"
    result = inputs.prepare_numeric_model(**numeric_args(local_backend, out))
    assert json.loads(out.read_text()) == result
    assert result["backend_manifest"]["source_tree_sha256"] == digest(before_source)
    assert result["backend_manifest"]["weights_tree_sha256"] == digest(before_weights)
    assert result["training_camera_sha256"] == digest(inputs.CameraConfig().manifest())
    assert locked_tree(local_backend.source) == before_source and locked_tree(local_backend.weights) == before_weights
    assert "real_model_validation" not in result and "research_acceptance" not in result


@pytest.mark.parametrize("decision", [{}, {"approved": False, "license_record": "control"},
    {"approved": True, "license_record": ""}, {"approved": 1, "license_record": "control"}])
def test_numeric_rejects_absent_operator_approval_before_publishing(local_backend, decision):
    save(local_backend.decisions, decision)
    out = local_backend.root / "rejected.json"
    with pytest.raises(ValueError, match="operator approval"):
        inputs.prepare_numeric_model(**numeric_args(local_backend, out))
    assert not out.exists()


@pytest.mark.parametrize("fault", ["source", "missing_expert", "missing_tokenizer", "stride"])
def test_numeric_rejects_wrong_source_or_incomplete_components(local_backend, fault):
    if fault == "source":
        (local_backend.source / "wan/modules/model.py").write_text("changed control source")
    elif fault == "missing_expert":
        (local_backend.weights / "high_noise_model/diffusion_pytorch_model.bin").unlink()
    elif fault == "missing_tokenizer":
        (local_backend.weights / "google/umt5-xxl/tokenizer.json").unlink()
    else:
        value = json.loads(local_backend.camera.read_text()); value["temporal_stride"] = 1
        save(local_backend.camera, value)
    out = local_backend.root / "rejected.json"
    with pytest.raises(ValueError):
        inputs.prepare_numeric_model(**numeric_args(local_backend, out))
    assert not out.exists()


def test_numeric_output_exclusive_and_outside_source_tree(local_backend):
    existing = local_backend.root / "existing.json"; existing.write_bytes(b"retain operator work")
    with pytest.raises(FileExistsError):
        inputs.prepare_numeric_model(**numeric_args(local_backend, existing))
    assert existing.read_bytes() == b"retain operator work"
    with pytest.raises(ValueError, match="outside"):
        inputs.prepare_numeric_model(**numeric_args(local_backend, local_backend.source / "generated.json"))
    assert not (local_backend.source / "generated.json").exists()


@pytest.fixture(scope="module")
def pilot_structure(tmp_path_factory):
    """Fixed metadata and synthetic PNGs; source TIFFs/audit are not supplied."""
    root = tmp_path_factory.mktemp("pilot_index_structure_only") / "build"
    rows = [{**fixed, "selection_hash": hashlib.sha256((VERSION + "/" + fixed["path"]).encode()).hexdigest()}
            for fixed in PILOT_DEV2_SOURCES]
    selection = {"core_manifest_sha256": "a" * 64, "repo_id": REPO, "revision": INRIA_REVISION,
                 "profile": "pilot-dev2", "full_core": False, "source_bytes": sum(r["size_bytes"] for r in rows),
                 "worlds": rows, "input_kind": "source_identity_checked_geography_unknown"}
    save(root / "build-selection.json", selection)
    worlds, task_index = [], []
    for i, row in enumerate(rows):
        wid = row["world_id"]
        world_path = root / "private/worlds" / (wid + ".png"); world_path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.zeros((2048, 2048, 3), dtype=np.uint8); pixels[:, :, i] = 89
        Image.fromarray(pixels).save(world_path)
        worlds.append({**row, "path": world_path.name, "source_path": row["path"], "source_sha256": row["lfs_sha256"],
                       "world_file_sha256": file_hash(world_path), "world_pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest()})
        for oi in range(4):
            for task, label_sets in packages(wid, "dev", oi):
                private = {}
                for name, arrays in label_sets.items():
                    label = root / "private/tasks/labels" / (task["task_id"] + "-" + name + ".npz")
                    label.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(label, **arrays)
                    private[name] = {"path": "labels/" + label.name, "sha256": file_hash(label)}
                task["private_label_files"] = private
                path = root / "private/tasks" / (task["task_id"] + ".json"); save(path, task)
                task_index.append({"task_id": task["task_id"], "session_id": digest([task["task_id"], "public-session"])[:24],
                                   "family": task["family"], "world_id": wid, "role": "dev", "city": "kitsap",
                                   "task_path": "tasks/" + path.name, "world_path": "worlds/" + world_path.name})
    save(root / "private/index.json", {"tasks": task_index, "worlds": worlds, "source_manifest_sha256": "a" * 64})
    save(root / "private/training-clips.json", {"clips": [], "clip_count": 0, "view_events": 0})
    save(root / "BUILD_COMPLETE.json", {"profile": "pilot-dev2", "source_bytes": selection["source_bytes"],
                                      "worlds": 2, "initial_images": 8, "task_packages": 32, "query_outputs_all_roles": 2816,
                                      "real_model_inference": False, "public_export_is_security_sandbox": False})
    return root


def tree_files(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in root.rglob("*") if p.is_file()}


def test_pilot_selector_keeps_complete_parent_and_fixed_origin0(pilot_structure, tmp_path):
    before = tree_files(pilot_structure)
    out = tmp_path / "run-inputs/pilot-index.json"
    result = inputs.prepare_pilot_index(build=pilot_structure, out=out)
    assert len(result["tasks"]) == 8 and len(result["worlds"]) == 2
    assert result["worlds"] == json.loads((pilot_structure / "private/index.json").read_text())["worlds"]
    assert {Path(r["task_path"]).name for r in result["tasks"]} == {
        fixed["world_id"] + "-o0-" + family + ".json" for fixed in PILOT_DEV2_SOURCES for family in ("F2", "F3", "F5", "F7")}
    assert all(Path(r["task_path"]).is_absolute() for r in result["tasks"])
    assert Path(result["world_root"]) == pilot_structure / "private/worlds"
    assert tree_files(pilot_structure) == before
    assert "not-a-data-audit" in result["selection_scope"]


@pytest.mark.parametrize("fault", ["source", "role", "profile", "tasks", "pixels", "task_program"])
def test_pilot_rejects_source_role_checksum_or_task_substitution(pilot_structure, tmp_path, fault):
    import shutil
    root = tmp_path / "bad-build"; shutil.copytree(pilot_structure, root)
    if fault in {"source", "role", "profile"}:
        path = root / "build-selection.json"; selection = json.loads(path.read_text())
        if fault == "source": selection["worlds"][0]["lfs_sha256"] = "d" * 64
        elif fault == "role": selection["worlds"][0]["role"] = "train"
        else: selection["profile"] = "smoke12"
        save(path, selection)
    elif fault == "tasks":
        path = root / "private/index.json"; index = json.loads(path.read_text()); index["tasks"].pop(); save(path, index)
    elif fault == "pixels":
        first = root / "private/worlds" / (PILOT_DEV2_SOURCES[0]["world_id"] + ".png")
        Image.fromarray(np.zeros((2048, 2048, 3), np.uint8)).save(first)
    else:
        path = root / "private/tasks" / (PILOT_DEV2_SOURCES[0]["world_id"] + "-o0-F2.json")
        task = json.loads(path.read_text()); task["programs"]["main"][1][0] += 1; save(path, task)
    out = tmp_path / "rejected.json"
    with pytest.raises(ValueError): inputs.prepare_pilot_index(build=root, out=out)
    assert not out.exists()


def test_pilot_selector_does_not_add_file_inside_audited_build(pilot_structure):
    before = tree_files(pilot_structure)
    with pytest.raises(ValueError, match="outside"):
        inputs.prepare_pilot_index(build=pilot_structure, out=pilot_structure / "private/pilot-index.json")
    assert tree_files(pilot_structure) == before


def training_spec(backend):
    numeric = inputs.prepare_numeric_model(**numeric_args(backend, backend.root / "numeric.json"))
    save(backend.root / "backend-manifest.json", numeric["backend_manifest"])
    policy = {"mode": "new_lora", "rank": 4, "alpha": 4, "targets": "attn_cam", "gradient_checkpointing": True}
    backend.config["training"] = {"trainable_policy": policy, "timestep_sampling": "uniform_integer",
                                  "expert_residency": "resident", "precision": "bf16",
                                  "validation": {"world_ids": ["control-dev-0"], "family": "F2", "origin_index": 0,
                                                 "seed": 0, "max_total_forwards": 2}}
    save(backend.source / "backend_config.json", backend.config)
    manifest = numeric["backend_manifest"]; manifest["source_tree_sha256"] = digest(locked_tree(backend.source))
    save(backend.root / "backend-manifest.json", manifest)
    spec = {"updates": 2, "batch_size": 1, "accumulation": 1, "lr": .0001, "seed": 17, "precision": "bf16",
            "device": "cuda:0", "camera": "camera.json", "trainable_policy": policy, "checkpoint_every": 1,
            "backend_manifest": "backend-manifest.json"}
    return save(backend.root / "training-spec.json", spec), manifest


def test_training_config_cannot_use_dev2_instead_of_core120(local_backend, pilot_structure):
    spec, _manifest = training_spec(local_backend)
    audit = save(local_backend.root / "audit.json", {"profile": "pilot-dev2"})
    out = local_backend.root / "rejected.json"
    with pytest.raises(ValueError, match="profile"):
        inputs.prepare_training_config(spec=spec, build=pilot_structure, source_audit=audit, out=out)
    assert not out.exists()


def test_training_config_rejects_implicit_or_bad_hyperparameters(local_backend, pilot_structure):
    spec, _manifest = training_spec(local_backend)
    value = json.loads(spec.read_text()); value["updates"] = True; save(spec, value)
    out = local_backend.root / "rejected.json"
    with pytest.raises(ValueError, match="integer"):
        inputs.prepare_training_config(spec=spec, build=pilot_structure, source_audit="absent-audit", out=out)
    assert not out.exists()


def mocked_core_structure(tmp_path, monkeypatch):
    """Isolate config derivation; real core audit/ClipDataset are explicitly mocked."""
    root = tmp_path / "core-structure"; (root / "private/worlds").mkdir(parents=True)
    path = root / "private/worlds/control-dev-0.png"
    Image.fromarray(np.zeros((2048, 2048, 3), np.uint8)).save(path)
    world = {"world_id": "control-dev-0", "city": "kitsap", "role": "dev", "path": path.name,
             "source_sha256": "e" * 64, "world_file_sha256": file_hash(path), "world_pixel_sha256": "f" * 64}
    index = {"worlds": [world], "tasks": [], "source_manifest_sha256": "a" * 64}
    selection = {"core_manifest_sha256": "a" * 64}
    for name, value in (("build-selection.json", selection), ("BUILD_COMPLETE.json", {}),
                        ("private/index.json", index), ("private/training-clips.json", {"clips": []})):
        save(root / name, value)
    audit = {"profile": "core120", "manifest_sha256": "a" * 64, "source_code_sha256": inputs.code_fingerprint(),
             "worlds": {world["world_id"]: {k: world[k] for k in ("world_id", "city", "role", "source_sha256", "world_file_sha256", "world_pixel_sha256")}},
             "artifact_sha256": tree_files(root)}
    audit_path = save(tmp_path / "core-audit-control.json", audit)
    monkeypatch.setattr(inputs, "_selection", lambda build, profile: (root, selection, index))
    monkeypatch.setattr(inputs, "_indexed_tasks", lambda *a, **k: [])
    import rswm_inria.materialized as materialized
    monkeypatch.setattr(materialized, "validate_audit_receipt", lambda *a, **k: {"status": "MOCKED_ONLY_NOT_REAL_ACCEPTANCE"})
    monkeypatch.setattr(inputs, "ClipDataset", lambda *a, **k: SimpleNamespace(root=root / "private/worlds", worlds={world["world_id"]: world}, source_audit=audit))
    return root, audit_path


def test_training_derives_actual_dev_records_and_preserves_operator_values(local_backend, tmp_path, monkeypatch):
    spec, manifest = training_spec(local_backend)
    root, audit = mocked_core_structure(tmp_path, monkeypatch)
    out = tmp_path / "training.json"
    result = inputs.prepare_training_config(spec=spec, build=root, source_audit=audit, out=out)
    assert set(result) == inputs.TRAINING_SPEC_FIELDS | {"validation_index"}
    assert result["updates"] == 2 and result["seed"] == 17 and result["lr"] == .0001
    assert result["backend_manifest"] == manifest and result["camera"] == json.loads(local_backend.camera.read_text())
    assert result["validation_index"][0]["world_id"] == "control-dev-0"
    assert result["validation_index"][0]["role"] == "dev"
    assert result["validation_index"][0]["world_file_sha256"] == file_hash(root / "private/worlds/control-dev-0.png")
    assert json.loads(out.read_text()) == result


@pytest.mark.parametrize("fault", ["stale_audit", "source_checksum", "weight_checksum", "validation_role", "validation_budget", "approval"])
def test_training_rejects_changed_source_audit_policy_or_approval(local_backend, tmp_path, monkeypatch, fault):
    spec, manifest = training_spec(local_backend)
    root, audit_path = mocked_core_structure(tmp_path, monkeypatch)
    if fault == "stale_audit":
        audit = json.loads(audit_path.read_text()); audit["source_code_sha256"] = "0" * 64; save(audit_path, audit)
    elif fault in {"source_checksum", "weight_checksum", "approval"}:
        key = {"source_checksum": "source_tree_sha256", "weight_checksum": "weights_tree_sha256", "approval": "approved"}[fault]
        manifest[key] = False if fault == "approval" else "0" * 64
        save(local_backend.root / "backend-manifest.json", manifest)
    elif fault == "validation_budget":
        local_backend.config["training"]["validation"]["max_total_forwards"] = 3
        save(local_backend.source / "backend_config.json", local_backend.config)
        manifest["source_tree_sha256"] = digest(locked_tree(local_backend.source)); save(local_backend.root / "backend-manifest.json", manifest)
    else:
        original = inputs.validate_dev_index
        def changed_role(rows, dataset):
            rows[0]["role"] = "train"; return original(rows, dataset)
        monkeypatch.setattr(inputs, "validate_dev_index", changed_role)
    out = tmp_path / "rejected.json"
    with pytest.raises(ValueError):
        inputs.prepare_training_config(spec=spec, build=root, source_audit=audit_path, out=out)
    assert not out.exists()


def test_cli_failure_returns_two_and_does_not_publish(local_backend, capsys):
    save(local_backend.decisions, {"approved": False, "license_record": "software-control"})
    out = local_backend.root / "rejected.json"
    args = ["numeric-model", "--source-dir", str(local_backend.source), "--weights-dir", str(local_backend.weights),
            "--camera", str(local_backend.camera), "--decisions", str(local_backend.decisions),
            "--model-id", "control", "--device", "cuda", "--out", str(out)]
    assert inputs.main(args) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "BLOCKED_OR_INVALID" and not out.exists()
