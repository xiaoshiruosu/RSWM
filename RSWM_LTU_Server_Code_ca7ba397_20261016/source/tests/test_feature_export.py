"""Actual small Torch serialization controls, never a formal DINOv2 instrument."""
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rswm_inria import feature_export as export
from rswm_inria.features import FrozenTorchScriptFeatures
from rswm_inria.geometry import file_hash


class DiagnosticTokenBackbone(torch.nn.Module):
    def __init__(self, dimension=4):
        super().__init__(); self.projection = torch.nn.Linear(3, dimension)
    def forward_features(self, x):
        return {"x_norm_clstoken": self.projection(x.mean(dim=(2, 3)))}


class NonfiniteTokenBackbone(DiagnosticTokenBackbone):
    def forward_features(self, x):
        return {"x_norm_clstoken": super().forward_features(x)["x_norm_clstoken"] * float("nan")}


def preprocessing():
    return {"resize": [28, 28], "center_crop": None, "resample": "bicubic", "scale": "uint8/255",
            "mean": [.485, .456, .406], "std": [.229, .224, .225]}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def diagnostic_checkpoint(tmp_path):
    torch.manual_seed(17)
    original = DiagnosticTokenBackbone().eval()
    path = tmp_path / "diagnostic-weights.pt"; torch.save(original.state_dict(), path)
    return original, path


def test_strict_load_replaces_every_initialized_parameter(diagnostic_checkpoint):
    original, path = diagnostic_checkpoint
    torch.manual_seed(91); loaded = DiagnosticTokenBackbone().eval()
    assert not torch.equal(loaded.projection.weight, original.projection.weight)
    report = export.strict_load_state_dict(loaded, path)
    assert report["strict_loaded"] and report["complete_keys"] == 2
    assert all(torch.equal(value, original.state_dict()[name]) for name, value in loaded.state_dict().items())
    assert all(parameter.requires_grad is False for parameter in loaded.parameters())


@pytest.mark.parametrize("fault", ["missing", "unexpected", "shape", "nested", "nonfinite", "nontensor"])
def test_strict_load_rejects_incomplete_or_bad_weights(diagnostic_checkpoint, tmp_path, fault):
    original, _path = diagnostic_checkpoint
    state = copy.deepcopy(original.state_dict())
    if fault == "missing": state.pop("projection.bias")
    elif fault == "unexpected": state["classifier.random"] = torch.zeros(1)
    elif fault == "shape": state["projection.weight"] = torch.zeros(2, 2)
    elif fault == "nested": state = {"state_dict": state}
    elif fault == "nonfinite": state["projection.bias"][0] = float("nan")
    else: state["projection.bias"] = "not a tensor"
    path = tmp_path / "bad-weights.pt"; torch.save(state, path)
    with pytest.raises(ValueError): export.strict_load_state_dict(DiagnosticTokenBackbone(), path)


def test_diagnostic_module_jit_readback_fresh_process_and_existing_feature_backend(diagnostic_checkpoint, tmp_path):
    original, weights = diagnostic_checkpoint
    model = DiagnosticTokenBackbone().eval(); export.strict_load_state_dict(model, weights)
    root = tmp_path / "diagnostic-export"; root.mkdir()
    config, report = export._export_loaded_model(model, root=root, dimension=4, preprocessing=preprocessing(),
        model_repo="DIAGNOSTIC_SOFTWARE_CONTROL_NOT_DINOV2", model_revision="diagnostic-fixture-v1", device="cpu",
        batch_size=2, diagnostic_features=True, timeout_seconds=30)
    assert config["diagnostic_features"] is True and config["feature_dimension"] == 4
    assert config["weights"]["sha256"] == file_hash(root / "encoder.torchscript.pt")
    assert json.loads((root / "feature-config.json").read_text()) == config
    assert [row["batch"] for row in report["comparisons"]] == [1, 2]
    assert all(row["fresh_process_max_abs_error"] <= 1e-5 for row in report["comparisons"])
    process = json.loads((root / "roundtrip/process-exit.json").read_text())
    assert process["exit_code"] == 0 and process["live"] is False and process["pid"] > 0
    backend = FrozenTorchScriptFeatures(config, root)
    values = backend.encode([np.zeros((128, 128, 3), np.uint8), np.full((128, 128, 3), 133, np.uint8)])
    assert values.shape == (2, 4) and values.dtype == np.float32 and np.isfinite(values).all()
    assert backend.manifest["diagnostic_features"] is True and backend.manifest["release_validated"] is False


def test_nonfinite_outputs_rejected_before_candidate_config(tmp_path):
    root = tmp_path / "nonfinite-export"; root.mkdir()
    with pytest.raises(ValueError, match="finite"):
        export._export_loaded_model(NonfiniteTokenBackbone().eval(), root=root, dimension=4,
            preprocessing=preprocessing(), model_repo="DIAGNOSTIC", model_revision="fixture", device="cpu",
            batch_size=2, diagnostic_features=True, timeout_seconds=30)
    assert not (root / "feature-config.json").exists()


@pytest.mark.parametrize("fault", ["patch_multiple", "std", "fields", "oversize_crop"])
def test_preprocessing_is_complete_and_shape_frozen(fault):
    value = preprocessing()
    if fault == "patch_multiple": value["resize"] = [27, 28]
    elif fault == "std": value["std"][0] = 0
    elif fault == "fields": value.pop("resample")
    else: value["center_crop"] = [42, 42]
    with pytest.raises(ValueError): export.validate_preprocessing(value)


@pytest.fixture
def local_git_control(tmp_path):
    """Only local revision-lock mechanics; this is not actual official source."""
    root = tmp_path / "source-control"; root.mkdir()
    for name in ("hubconf.py", "LICENSE", "dinov2/hub/backbones.py", "dinov2/models/vision_transformer.py"):
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("SOFTWARE_CONTROL_ONLY_NOT_OFFICIAL_SOURCE\n")
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout.decode().strip()
    git("init", "-q"); git("add", ".")
    git("-c", "user.name=Software Control", "-c", "user.email=software-control@example.invalid", "commit", "-qm", "Diagnostic source lock fixture")
    git("remote", "add", "origin", "https://github.com/facebookresearch/dinov2.git")
    return root, git("rev-parse", "HEAD"), git


def test_source_lock_verifies_actual_revision_and_preserves_changes(local_git_control):
    root, revision, git = local_git_control
    result = export.verify_local_source(root, revision)
    assert result["revision"] == revision and result["files"]["hubconf.py"] == file_hash(root / "hubconf.py")
    with pytest.raises(ValueError, match="revision"):
        export.verify_local_source(root, "0" * 40)
    before = (root / "hubconf.py").read_bytes(); (root / "hubconf.py").write_bytes(before + b"operator change")
    with pytest.raises(ValueError, match="operator changes"):
        export.verify_local_source(root, revision)
    assert (root / "hubconf.py").read_bytes() == before + b"operator change"


def test_unknown_architecture_and_absent_decision_do_not_initialize_or_download(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "_load_original_model", lambda *a, **k: pytest.fail("An unsupported export initialized a model"))
    args = dict(source_dir="absent", weights="absent", architecture="some-other-hf-model", revision="0" * 40,
                decisions="absent", preprocessing="absent", device="cpu", batch_size=2, out=tmp_path / "never")
    with pytest.raises(ValueError, match="DINOv2"): export.export_dinov2(**args)
    assert not (tmp_path / "never").exists()
    args["architecture"] = "dinov2_vits14"
    args["decisions"] = save(tmp_path / "decision.json", {"approved": False, "license_record": "diagnostic", "decision_reference": "control"})
    with pytest.raises(ValueError, match="owner-approved"): export.export_dinov2(**args)
    assert not (tmp_path / "never").exists()


def test_public_export_failure_preserves_contract_and_actual_error_without_config(local_git_control, tmp_path, monkeypatch):
    root, revision, _git = local_git_control
    decision = save(tmp_path / "inputs/decision.json", {"approved": True, "license_record": "SOFTWARE_CONTROL_ONLY",
                                                       "decision_reference": "diagnostic fixture, never formal approval"})
    prep = save(tmp_path / "inputs/prep.json", preprocessing())
    weights = tmp_path / "weights/empty-state-dict.pt"; weights.parent.mkdir(); torch.save({}, weights)
    # This patched model tests only failure retention after contract publication;
    # its strict load fails and no non-diagnostic candidate is ever published.
    monkeypatch.setattr(export, "_load_original_model", lambda *a: DiagnosticTokenBackbone())
    out = tmp_path / "failed-candidate"
    with pytest.raises(ValueError, match="state_dict"):
        export.export_dinov2(source_dir=root, weights=weights, architecture="dinov2_vits14", revision=revision,
            decisions=decision, preprocessing=prep, device="cpu", batch_size=2, out=out, timeout_seconds=30)
    assert (out / "export-contract.json").exists() and (out / "export-failure.json").exists()
    report = json.loads((out / "export-failure.json").read_text())
    assert report["status"] == "FAILED" and report["release_validated"] is False
    assert not (out / "feature-config.json").exists() and not (out / "export-report.json").exists()
    with pytest.raises(FileExistsError):
        export.export_dinov2(source_dir=root, weights=weights, architecture="dinov2_vits14", revision=revision,
            decisions=decision, preprocessing=prep, device="cpu", batch_size=2, out=out, timeout_seconds=30)


def test_cli_unknown_family_returns_two_without_fallback(tmp_path, capsys):
    args = ["dinov2", "--source-dir", "absent", "--weights", "absent", "--architecture", "dinov3_vits16",
            "--revision", "0" * 40, "--decisions", "absent", "--preprocessing", "absent", "--device", "cpu",
            "--batch-size", "2", "--out", str(tmp_path / "never")]
    assert export.main(args) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["release_validated"] is False and report["research_acceptance"] == "NOT_EVALUATED"


def test_cli_corrupt_checkpoint_records_failure_and_returns_two(local_git_control, tmp_path, monkeypatch, capsys):
    root, revision, _git = local_git_control
    decision = save(tmp_path / "inputs/decision.json", {"approved": True, "license_record": "SOFTWARE_CONTROL_ONLY",
                                                       "decision_reference": "diagnostic corrupt-checkpoint rejection"})
    prep = save(tmp_path / "inputs/prep.json", preprocessing())
    weights = tmp_path / "weights/corrupt.pt"; weights.parent.mkdir(); weights.write_bytes(b"not a torch checkpoint")
    monkeypatch.setattr(export, "_load_original_model", lambda *a: DiagnosticTokenBackbone())
    out = tmp_path / "corrupt-attempt"
    args = ["dinov2", "--source-dir", str(root), "--weights", str(weights), "--architecture", "dinov2_vits14",
            "--revision", revision, "--decisions", str(decision), "--preprocessing", str(prep), "--device", "cpu",
            "--batch-size", "2", "--out", str(out)]
    assert export.main(args) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "BLOCKED_OR_FAILED"
    assert (out / "export-failure.json").exists() and not (out / "feature-config.json").exists()
