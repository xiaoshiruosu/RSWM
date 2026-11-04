"""Actual PEFT CPU software controls; never pretrained Wan/train60 acceptance.

The tolerances below are fixed before executing any round-trip. The tiny
backbone is a software input, while every adapter operation uses real PEFT.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
from safetensors.torch import load_file, save_file

from rswm_inria.geometry import digest, file_hash
from rswm_inria.legacy_wan import strict_lora_from_pretrained

import importlib.util
_helper = Path(__file__).resolve().parents[1]/"scripts"/"reproduce_trained_adapter.py"
_spec = importlib.util.spec_from_file_location("tiny_peft_control", _helper)
_control = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_control)
TinyBackbone = _control.TinyBackbone
TinyBackend = _control.TinyBackend
POLICY = _control.POLICY
UNMERGED_TOLERANCE = _control.UNMERGED_TOLERANCE
MERGED_TOLERANCE = _control.MERGED_TOLERANCE
fixed_input = _control.fixed_input
trained_tiny_backend = _control.trained_tiny_backend
create_driver_fixture = _control.create_driver_fixture


def test_actual_peft_updates_both_experts_and_frozen_parameters():
    backend, _, audit = trained_tiny_backend()
    assert all(isinstance(m, peft.PeftModel) for m in backend.model.values())
    assert all("lora_" in n for n, p in backend.model.named_parameters() if p.requires_grad)
    assert len(audit) == 2
    # B starts at zero: A first obtains a nonzero gradient on the second step.
    assert all(any(n.startswith(tag+".") and "lora_A" in n for n in audit[1]["nonzero_gradients"])
               for tag in ("low", "high"))


@pytest.fixture(scope="module")
def driver_fixture(tmp_path_factory):
    return create_driver_fixture(tmp_path_factory.mktemp("actual-peft-driver")/"work")


def test_actual_driver_checkpoint_preserves_real_peft_state_and_receipt(driver_fixture):
    from rswm_inria import training
    from rswm_inria.adapters.camera_conditioning import CameraConfig
    from rswm_inria.training import tensor_state_sha256
    f = driver_fixture
    contract = digest({"config": f["config"], "dataset": digest({"index": f["context"]["index"], "clips": f["context"]["clips"]}),
                       "source_audit": digest(f["context"]["source_audit"]), "camera": CameraConfig(**f["config"]["camera"]).manifest()})
    receipt = training.validate_checkpoint_receipt(f["checkpoint"], contract)
    state = torch.load(f["checkpoint"], map_location="cpu", weights_only=True)
    assert receipt["schema"] == "rswm-training-checkpoint-v2"
    assert receipt["training"]["step"] == state["step"] == 2
    assert receipt["training"]["clip_cursor"] == state["position"] == 2
    assert receipt["training"]["training_seed"] == 83
    for tag, model in f["trained"].model.items():
        retained = {n.removeprefix(tag+"."): v for n, v in state["model"].items() if n.startswith(tag+".")}
        assert set(retained) == set(model.state_dict())
        assert receipt["expert_state_sha256"][tag] == tensor_state_sha256(retained)
        assert all(torch.equal(value, model.state_dict()[name]) for name, value in retained.items())
        assert any(n.startswith(tag+".") and "lora_A" in n for n in receipt["audit"][-1]["nonzero_gradients"])
        base = TinyBackbone(11 if tag == "low" else 29)
        for name, parameter in model.get_base_model().named_parameters():
            if "lora_" not in name:
                original_name = name.replace(".base_layer", "")
                assert torch.equal(parameter, dict(base.named_parameters())[original_name])
    assert training.validate_dev_index is not _control.software_dev_checker
    assert f["context"]["scope"] == "software_control"
    assert f["context"]["source_audit"]["synthetic"] is True
    assert all(row["role"] == "diagnostic_only" for row in f["config"]["validation_index"])


def export_fixture(f, tmp_path, *, backend=None, checkpoint=None, config=None, context=None):
    from rswm_inria.legacy_export import _export_bound_backend
    backend = TinyBackend(f["weights"], f["source"], f["config"]["trainable_policy"]) if backend is None else backend
    backend.source_dir = f["source"]
    backend.weights_dir = f["weights"]
    return _export_bound_backend(backend, checkpoint=f["checkpoint"] if checkpoint is None else checkpoint,
                                 config=f["config"] if config is None else config,
                                 context=f["context"] if context is None else context,
                                 inference_config=f["inference_config"], out=tmp_path/"release",
                                 deployment_out=tmp_path/"deployment")


def resealed_checkpoint(f, tmp_path, mutation):
    from rswm_inria.training import tensor_state_sha256
    driver = tmp_path/"driver"
    shutil.copytree(f["checkpoint"].parent, driver)
    checkpoint = driver/f["checkpoint"].name
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mutation(state)
    torch.save(state, checkpoint)
    receipt_path = checkpoint.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    receipt["checkpoint_sha256"] = file_hash(checkpoint)
    receipt["expert_state_sha256"] = {tag: tensor_state_sha256({n.removeprefix(tag+"."): v
                       for n, v in state["model"].items() if n.startswith(tag+".")}) for tag in ("low", "high")}
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False)+"\n")
    return checkpoint


@pytest.mark.parametrize("section", ["source", "base", "camera", "policy", "seed", "dataset", "audit", "code"])
def test_export_rejects_wrong_bound_identity(driver_fixture, tmp_path, section):
    f = driver_fixture
    config, context = copy.deepcopy(f["config"]), copy.deepcopy(f["context"])
    if section == "source": config["backend_manifest"]["source_tree_sha256"] = "0"*64
    elif section == "base": config["backend_manifest"]["weights_tree_sha256"] = "0"*64
    elif section == "camera": config["camera"]["translation_scaling"] = "legacy_maxnorm"
    elif section == "policy": config["trainable_policy"]["rank"] = 3
    elif section == "seed": config["seed"] += 1
    elif section == "dataset": context["index"]["clips"] = 3
    elif section == "audit": context["source_audit"]["synthetic"] = False
    else: context["source_code_sha256"] = "0"*64
    with pytest.raises((ValueError, RuntimeError)):
        export_fixture(f, tmp_path, config=config, context=context)
    assert not (tmp_path/"release/release.json").exists()


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "nan", "inf", "swapped", "frozen", "cursor"])
def test_export_rejects_semantically_bad_state_even_resealed(driver_fixture, tmp_path, corruption):
    def mutation(state):
        values = state["model"]
        key = next(n for n in sorted(values) if "lora_A" in n and n.startswith("low."))
        if corruption == "missing": values.pop(key)
        elif corruption == "unexpected": values["low.unexpected.lora_A.weight"] = values[key].clone()
        elif corruption == "shape": values[key] = values[key][:-1].contiguous()
        elif corruption in {"nan", "inf"}: values[key].view(-1)[0] = float(corruption)
        elif corruption == "swapped":
            old = {n: v.clone() for n, v in values.items()}
            for name in values:
                tag, tail = name.split(".", 1)
                values[name] = old[("high" if tag == "low" else "low")+"."+tail]
        elif corruption == "frozen":
            frozen = next(n for n in sorted(values) if "lora_" not in n)
            values[frozen].view(-1)[0] += .1
        else: state["position"] += 1
    # Nonfinite tensors are rejected by the receipt digest helper itself. For
    # all other cases both byte and expert digests are correctly resealed.
    with pytest.raises((ValueError, RuntimeError)):
        checkpoint = resealed_checkpoint(driver_fixture, tmp_path, mutation)
        export_fixture(driver_fixture, tmp_path, checkpoint=checkpoint)
    assert not (tmp_path/"release/release.json").exists()


@pytest.mark.parametrize("corruption", ["missing", "hash", "malformed", "attempt_missing", "attempt_tamper"])
def test_export_requires_actual_retained_checkpoint_receipt_and_attempt(driver_fixture, tmp_path, corruption):
    driver = tmp_path/"driver"
    shutil.copytree(driver_fixture["checkpoint"].parent, driver)
    checkpoint = driver/driver_fixture["checkpoint"].name
    receipt_path = checkpoint.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    if corruption == "missing": receipt_path.unlink()
    elif corruption == "hash": receipt["checkpoint_sha256"] = "0"*64; receipt_path.write_text(json.dumps(receipt))
    elif corruption == "malformed": receipt_path.write_text("[]")
    else:
        attempt = driver/receipt["attempt"]["journal_path"]
        if corruption == "attempt_missing": attempt.unlink()
        else: attempt.write_bytes(attempt.read_bytes()+b"tamper\n")
    with pytest.raises((ValueError, RuntimeError, OSError)):
        export_fixture(driver_fixture, tmp_path, checkpoint=checkpoint)
    assert not (tmp_path/"release/release.json").exists()


def test_export_exclusivity_and_partial_publication_fail_closed(driver_fixture, tmp_path, monkeypatch):
    existing = tmp_path/"release"; existing.mkdir(); (existing/"owner.txt").write_text("keep")
    with pytest.raises((FileExistsError, ValueError)):
        export_fixture(driver_fixture, tmp_path)
    assert (existing/"owner.txt").read_text() == "keep"
    second = tmp_path/"second"; second.mkdir()
    original = peft.PeftModel.save_pretrained
    calls = []
    def fail_high(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2: raise OSError("software injected second-expert publication failure")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(peft.PeftModel, "save_pretrained", fail_high)
    with pytest.raises(OSError, match="publication failure"):
        export_fixture(driver_fixture, second)
    assert not (second/"release/release.json").exists()


def test_mid_publication_failure_leaves_no_completion_marker_and_cannot_resume(driver_fixture, tmp_path, monkeypatch):
    from rswm_inria import legacy_export
    from rswm_inria.training import locked_tree
    original = legacy_export.save_bytes_exclusive
    def fail_after_retained_low(path, content):
        if Path(path).parent.name == "high_noise":
            raise OSError("software injected mid-publication failure")
        return original(path, content)
    monkeypatch.setattr(legacy_export, "save_bytes_exclusive", fail_after_retained_low)
    with pytest.raises(OSError, match="mid-publication"):
        export_fixture(driver_fixture, tmp_path)
    root = tmp_path/"release"
    assert (root/"low_noise/adapter_model.safetensors").is_file()
    assert not (root/"release.json").exists()
    retained = {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    binding = {"schema": "rswm-trained-peft-release-ref-v1", "root": str(root), "tree_sha256": digest(locked_tree(root)),
               "manifest_sha256": "0"*64, "scope": "software_control"}
    with pytest.raises(ValueError, match="incomplete"):
        legacy_export.validate_adapter_release(binding, allow_software=True)
    monkeypatch.setattr(legacy_export, "save_bytes_exclusive", original)
    with pytest.raises(FileExistsError):
        export_fixture(driver_fixture, tmp_path)
    assert retained == {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_publication_directory_race_never_overwrites_other_writer(driver_fixture, tmp_path, monkeypatch):
    output = tmp_path/"release"
    original = Path.mkdir
    raced = []
    def create_other_writer(path, *args, **kwargs):
        if path == output and not raced:
            raced.append(True)
            original(path, parents=True)
            (path/"other-writer.txt").write_bytes(b"retain other writer")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "mkdir", create_other_writer)
    with pytest.raises(FileExistsError):
        export_fixture(driver_fixture, tmp_path)
    assert (output/"other-writer.txt").read_bytes() == b"retain other writer"
    assert sorted(p.name for p in output.iterdir()) == ["other-writer.txt"]


@pytest.mark.parametrize("kind", ["non_lora_trainable", "rslora", "unexpected_scaling"])
def test_export_rejects_trainables_not_exactly_represented_by_pure_lora(driver_fixture, tmp_path, kind):
    class InvalidPolicyBackend(TinyBackend):
        def make_lora_expert(self, *, model, rank, alpha, targets):
            if kind == "rslora":
                from rswm_inria.legacy_training import _expected_lora_targets
                return peft.get_peft_model(model, peft.LoraConfig(r=rank, lora_alpha=alpha, bias="none",
                          use_rslora=True, target_modules=sorted(_expected_lora_targets(model, targets))))
            return super().make_lora_expert(model=model, rank=rank, alpha=alpha, targets=targets)
        def configure_trainable(self, policy):
            super().configure_trainable(policy)
            if kind == "non_lora_trainable":
                self.model["low"].get_base_model().blocks[0].ffn.weight.requires_grad_(True)
            elif kind == "unexpected_scaling":
                self.model["low"].get_base_model().blocks[0].self_attn.q.scaling["default"] *= 2
    backend = InvalidPolicyBackend(driver_fixture["weights"], driver_fixture["source"])
    with pytest.raises(ValueError, match="LoRA|scaling"):
        export_fixture(driver_fixture, tmp_path, backend=backend)
    assert not (tmp_path/"release/release.json").exists()


def test_release_symlink_and_existing_deployment_are_never_followed_or_overwritten(driver_fixture, tmp_path):
    actual = tmp_path/"owner-directory"; actual.mkdir(); (actual/"keep").write_bytes(b"original")
    (tmp_path/"release").symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        export_fixture(driver_fixture, tmp_path)
    assert (actual/"keep").read_bytes() == b"original"
    second = tmp_path/"second"; second.mkdir(); (second/"deployment").mkdir()
    (second/"deployment/keep").write_bytes(b"original source")
    with pytest.raises(FileExistsError):
        export_fixture(driver_fixture, second)
    assert (second/"deployment/keep").read_bytes() == b"original source"
    assert not (second/"release").exists()


def test_bad_receipt_fails_before_tensor_deserialization_or_model_configuration(driver_fixture, tmp_path, monkeypatch):
    driver = tmp_path/"driver"; shutil.copytree(driver_fixture["checkpoint"].parent, driver)
    checkpoint = driver/driver_fixture["checkpoint"].name
    receipt = json.loads(checkpoint.with_suffix(".json").read_text())
    receipt["checkpoint_sha256"] = "0"*64
    checkpoint.with_suffix(".json").write_text(json.dumps(receipt))
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Bad receipt must fail before loading tensors or configuring experts")
    backend = TinyBackend(driver_fixture["weights"], driver_fixture["source"])
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(backend, "configure_trainable", forbidden)
    with pytest.raises(ValueError, match="receipt/hash"):
        export_fixture(driver_fixture, tmp_path, backend=backend, checkpoint=checkpoint)


def test_provider_subtree_dummy_is_never_parsed_or_redistributed(driver_fixture, tmp_path, monkeypatch):
    from rswm_inria import legacy_export
    from rswm_inria.training import locked_tree
    source = tmp_path/"unsafe-software-source"; shutil.copytree(driver_fixture["source"], source)
    dummy = source/"wan/remotesensebenchmark/config.json"; dummy.parent.mkdir(parents=True)
    dummy.write_bytes(b'{"provider":{"api_key":"SYNTHETIC-DUMMY-NOT-A-CREDENTIAL"}}\n')
    original = dummy.read_bytes()
    config = copy.deepcopy(driver_fixture["config"])
    config["backend_manifest"]["source_dir"] = str(source)
    config["backend_manifest"]["source_tree_sha256"] = digest(locked_tree(source))
    read = legacy_export._json
    def reject_provider_parse(path):
        assert "remotesensebenchmark" not in Path(path).parts
        return read(path)
    monkeypatch.setattr(legacy_export, "_json", reject_provider_parse)
    with pytest.raises(ValueError, match="Provider subtree"):
        export_fixture(driver_fixture, tmp_path, config=config)
    assert dummy.read_bytes() == original
    assert not (tmp_path/"release").exists() and not (tmp_path/"deployment").exists()


def test_synthetic_marker_relabel_cannot_authorize_retained_formal_data(driver_fixture):
    from rswm_inria.legacy_export import _formal_retained_data
    context = copy.deepcopy(driver_fixture["context"])
    context["scope"] = "trained_model_candidate"
    with pytest.raises(ValueError, match="Synthetic/software"):
        _formal_retained_data(context)
    for name in ("index", "clips", "source_audit"):
        context[name].pop("synthetic", None); context[name].pop("scope", None)
    with pytest.raises(ValueError, match="Complete retained"):
        _formal_retained_data(context)


def test_actual_wan_component_path_rejects_software_release(driver_fixture, tmp_path):
    from rswm_inria.legacy_wan import component_inventory
    exported = export_fixture(driver_fixture, tmp_path)
    config = json.loads((tmp_path/"deployment/backend_config.json").read_text())
    with pytest.raises(ValueError, match="Software-control"):
        component_inventory(driver_fixture["weights"], config)


def test_resealed_serialized_adapter_digest_lie_and_missing_inventory_are_rejected(driver_fixture, tmp_path):
    from rswm_inria.legacy_export import validate_adapter_release, validate_adapter_tensor_content
    from rswm_inria.geometry import canonical
    from rswm_inria.training import locked_tree
    exported = export_fixture(driver_fixture, tmp_path)
    root = tmp_path/"release"
    release = json.loads((root/"release.json").read_text())
    tensors = load_file(root/"low_noise/adapter_model.safetensors")
    tensors[sorted(tensors)[0]].view(-1)[0] += .25
    save_file(tensors, root/"low_noise/adapter_model.safetensors")
    release["files"]["low_noise/adapter_model.safetensors"] = file_hash(root/"low_noise/adapter_model.safetensors")
    (root/"release.json").write_bytes(canonical(release)+b"\n")
    binding = dict(exported["adapter_release"], tree_sha256=digest(locked_tree(root)), manifest_sha256=file_hash(root/"release.json"))
    assert validate_adapter_release(binding, allow_software=True)["scope"] == "software_control"
    with pytest.raises(ValueError, match="tensor content"):
        validate_adapter_tensor_content(binding, allow_software=True)
    (root/"high_noise/adapter_model.safetensors").unlink()
    release["files"].pop("high_noise/adapter_model.safetensors")
    (root/"release.json").write_bytes(canonical(release)+b"\n")
    binding.update(tree_sha256=digest(locked_tree(root)), manifest_sha256=file_hash(root/"release.json"))
    with pytest.raises(ValueError, match="Incomplete"):
        validate_adapter_release(binding, allow_software=True)


@pytest.mark.parametrize("mutation", ["swapped", "config"])
def test_loaded_same_shape_expert_swap_and_complete_config_mismatch_rejected(driver_fixture, tmp_path, mutation):
    from rswm_inria.legacy_export import validate_loaded_expert
    exported = export_fixture(driver_fixture, tmp_path)
    tag = "high" if mutation == "swapped" else "low"
    model = strict_lora_from_pretrained(peft.PeftModel, TinyBackbone(29 if tag == "high" else 11),
                                       tmp_path/"release"/(tag+"_noise"), is_trainable=False)
    if mutation == "config": model.peft_config["default"].lora_alpha += 1
    with pytest.raises(ValueError, match="complete trained checkpoint|configuration"):
        validate_loaded_expert(model, "low", exported["adapter_release"], allow_software=True)


@pytest.mark.parametrize("mode", ["new_lora", "restored_lora"])
def test_actual_driver_export_fresh_registry_suite_two_initialization_modes(tmp_path, mode):
    result = _control.reproduce_one(tmp_path/"process-cwd", tmp_path/"control", mode)
    assert result["initialization_mode"] == mode
    assert result["scope"] == "software_control" and result["synthetic"] is True
    assert result["suite"]["runs"] == 1 and result["suite"]["completed_outputs"] == 32
    assert result["suite"]["replay_verified"] is True and result["suite"]["real_pilot_outputs"] == 0
    assert result["fresh_process"]["pid"] != os.getpid()
    assert result["capabilities"]["purpose"] == "diagnostic"
    assert result["real_model_validation"] == result["real_gpu_validation"] == result["train60_acceptance"] == "NOT_RUN"
    assert result["export"]["owner_approval"] == "REQUIRED_NOT_CREATED"
    assert result["real_experiment_gate"]["status"] == "REJECTED_SOFTWARE_FIXTURE"
    from rswm_inria.legacy_export import validate_adapter_release
    with pytest.raises(ValueError, match="Software-control"):
        validate_adapter_release(result["export"]["adapter_release"])


def test_public_exporter_still_requires_real_train60(driver_fixture, tmp_path):
    from rswm_inria.legacy_export import export_checkpoint
    f = driver_fixture
    index = {"worlds": [{"world_id": "synthetic-only", "role": "diagnostic_only"}]}
    clips = {"clips": [{"world_id": "synthetic-only"}], "clip_count": 1, "view_events": 33}
    paths = {}
    for name, value in (("config", f["config"]), ("index", index), ("clips", clips),
                        ("source_audit", f["context"]["source_audit"]), ("inference_config", f["inference_config"])):
        paths[name] = tmp_path/(name+".json"); paths[name].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="complete train60/3840clips"):
        export_checkpoint(checkpoint=f["checkpoint"], world_root=tmp_path,
                          out=tmp_path/"release", deployment_out=tmp_path/"deployment", **paths)
    assert not (tmp_path/"release").exists()


@pytest.mark.parametrize("field", ["training_seed", "camera", "release_camera"])
def test_registry_rejects_release_seed_camera_mismatch(driver_fixture, tmp_path, field):
    from rswm_inria.registry import validate_config
    exported = export_fixture(driver_fixture, tmp_path)
    config = _control.registry_config(exported)
    if field == "training_seed": config[field] += 1
    elif field == "camera": config["camera"]["camera_channel"] = "identity"
    else:
        camera = _control.CameraConfig(camera_channel="identity")
        config["camera"]["camera_channel"] = "identity"
        config["training_camera_sha256"] = digest(camera.manifest())
    with pytest.raises(ValueError, match="seed|camera|mismatch"):
        validate_config(config)


def test_actual_peft_save_strict_restore_merge_fresh_process(tmp_path):
    backend, _, _ = trained_tiny_backend()
    expected = {tag: model(fixed_input()).detach() for tag, model in backend.model.items()}
    for tag, seed in (("low", 11), ("high", 29)):
        path = tmp_path/tag
        backend.model[tag].save_pretrained(path, safe_serialization=True)
        official = peft.get_peft_model_state_dict(backend.model[tag], adapter_name="default", save_embedding_layers=False)
        retained = load_file(path/"adapter_model.safetensors")
        assert set(retained) == set(official)
        assert all(torch.equal(retained[name], value.cpu()) for name, value in official.items())
        restored = strict_lora_from_pretrained(peft.PeftModel, TinyBackbone(seed), path, is_trainable=False)
        torch.testing.assert_close(restored(fixed_input()), expected[tag], **UNMERGED_TOLERANCE)
        merged = restored.merge_and_unload(safe_merge=True)
        torch.testing.assert_close(merged(fixed_input()), expected[tag], **MERGED_TOLERANCE)
    torch.save(expected, tmp_path/"expected.pt")
    script = """
import runpy,sys,torch,peft
from pathlib import Path
scope=runpy.run_path(sys.argv[1]); root=Path(sys.argv[2])
expected=torch.load(root/'expected.pt',map_location='cpu',weights_only=True)
for tag,seed in [('low',11),('high',29)]:
    restored=scope['strict_lora_from_pretrained'](peft.PeftModel,scope['TinyBackbone'](seed),root/tag,is_trainable=False)
    torch.testing.assert_close(restored(scope['fixed_input']()),expected[tag],**scope['UNMERGED_TOLERANCE'])
    torch.testing.assert_close(restored.merge_and_unload(safe_merge=True)(scope['fixed_input']()),expected[tag],**scope['MERGED_TOLERANCE'])
print('PASS_ACTUAL_PEFT_CPU_SOFTWARE_ROUNDTRIP_ONLY')
"""
    completed = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), str(tmp_path)],
                               text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "PASS_ACTUAL_PEFT_CPU_SOFTWARE_ROUNDTRIP_ONLY" in completed.stdout


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "nan", "inf"])
def test_actual_peft_strict_restore_rejects_corrupt_adapter(tmp_path, corruption):
    backend, _, _ = trained_tiny_backend()
    backend.model["low"].save_pretrained(tmp_path, safe_serialization=True)
    path = tmp_path/"adapter_model.safetensors"
    tensors = load_file(path)
    key = sorted(tensors)[0]
    if corruption == "missing": tensors.pop(key)
    elif corruption == "unexpected": tensors["base_model.model.unexpected.lora_A.weight"] = tensors[key].clone()
    elif corruption == "shape": tensors[key] = tensors[key][:-1].contiguous()
    else: tensors[key].view(-1)[0] = float(corruption)
    save_file(tensors, path)
    with pytest.raises((ValueError, RuntimeError)):
        strict_lora_from_pretrained(peft.PeftModel, TinyBackbone(11), tmp_path, is_trainable=False)
