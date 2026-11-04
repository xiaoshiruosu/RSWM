"""Real CPU autograd controls for original-Wan offload integration.

The API fixture has tiny frozen Linears, not Wan weights. Native clip geometry,
flow objective, expert boundary and checkpoint machinery are exercised, but
these tests cannot establish A100 memory fit, speed or research acceptance.
"""
import json
from types import MethodType

import numpy as np
import pytest

from test_legacy_training import make_backend, training_spec
from test_legacy_wan import config as backend_config


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def offload_backend(torch, tmp_path, monkeypatch, *, configure=True):
    """Keep the existing fixture loss while using actual checkpointed Linear."""
    backend, config, camera = make_backend(torch, tmp_path, configure=False)
    spec = backend.runtime_config["training"]
    spec["expert_residency"] = "block_cpu"
    spec["trainable_policy"]["gradient_checkpointing"] = True
    backend.runtime_config["sampling"].update(
        block_cpu_offload=True, inactive_model_cpu=True, t5_cpu=True)
    for expert in backend.model.values():
        linear = expert.blocks[0].self_attn.q.base_layer
        with torch.no_grad():
            linear.weight.fill_(1)
            linear.bias.zero_()
        expert.checkpoint_enables = 0
        expert.checkpoint_disables = 0

        def enable(self):
            self.checkpoint_enables += 1
            self.gradient_checkpointing = True

        def disable(self):
            self.checkpoint_disables += 1
            self.gradient_checkpointing = False

        original_forward = expert.forward

        def forward(self, _original=original_forward, **kwargs):
            # Retain the old fixture's native I2V mask, prompt/camera and shape
            # assertions, then route LoRA through a genuinely frozen Linear.
            baseline = _original(**kwargs)[0]
            from torch.utils.checkpoint import checkpoint
            layer = self.blocks[0].self_attn.q
            projected = checkpoint(layer.base_layer, layer.lora_A.reshape(1, 1),
                                   use_reentrant=False).reshape(())
            return [baseline - layer.lora_A + projected]

        monkeypatch.setattr(expert, "gradient_checkpointing_enable", MethodType(enable, expert), raising=False)
        monkeypatch.setattr(expert, "gradient_checkpointing_disable", MethodType(disable, expert), raising=False)
        monkeypatch.setattr(expert, "forward", MethodType(forward, expert))
    if configure:
        backend.configure_trainable(config["trainable_policy"])
    return backend, config, camera


def fixed_noise(torch, monkeypatch):
    monkeypatch.setattr(torch, "randint", lambda _low, _high, shape, **_kwargs:
                        torch.tensor([0, 947][:shape[0]]))
    monkeypatch.setattr(torch, "randn_like", lambda x, **kwargs:
                        torch.ones_like(x, dtype=kwargs["dtype"]))


def native_batch(torch, config, camera, batch=2):
    from rswm_inria.adapters.camera_conditioning import backend_payload, build_camera_batch
    images = torch.zeros(1, 1, 3, 1, 1).expand(batch, 33, 3, 480, 832)
    payload = backend_payload(build_camera_batch(np.zeros((32, 2), np.int64), config=camera))
    return images, [payload] * batch


def test_offload_checkpointed_native_loss_gradients_and_optimizer_identity(torch, tmp_path, monkeypatch):
    from rswm_inria.training import (audit_trainable_gradients, parameter_changes,
                                    parameter_snapshot, verify_parameter_binding)
    backend, config, camera = offload_backend(torch, tmp_path, monkeypatch)
    fixed_noise(torch, monkeypatch)
    params = dict(backend.model.named_parameters())
    trainable = {name: p for name, p in params.items() if p.requires_grad}
    identities = {name: id(p) for name, p in params.items()}
    before = parameter_snapshot(params, trainable)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=.01)
    optimizer_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(trainable) == 4 and all("lora_" in name for name in trainable)
    assert len(backend._offload_controllers) == 2
    assert all(expert.checkpoint_enables == 1 and not expert.checkpoint_disables
               for expert in backend.model.values())

    images, cameras = native_batch(torch, config, camera)
    loss = backend.loss(images=images, cameras=cameras, update=0)
    assert loss.item() == pytest.approx(.36, abs=1e-6)
    loss.backward()
    active = {name: set(group) for name, group in backend.active_trainable_groups().items()}
    gradients = audit_trainable_gradients(trainable, active)
    assert set(active) == {"low", "high"} and gradients == set(trainable)
    for parameter in trainable.values():
        assert parameter.grad.item() == pytest.approx(-.6, abs=1e-6)
    assert backend.model["low"].calls[0]["t"] == [0]
    assert backend.model["high"].calls[0]["t"] == [947]
    assert backend.text_encoder.prompts == [["fixed aerial scene"] * 2]

    for controller in backend._offload_controllers.values():
        report = controller.report()
        assert report["cpu_simulation"] is True and report["real_40gb_model_validation"] is False
        assert report["weight_onload_bytes"] > 0
        assert report["module_onloads"] >= 2, "Checkpoint backward must recompute the frozen Linear"
        assert report["saved_tensor_restore_bytes"] > 0
        assert report["staged_frozen_bytes"] == 0
    optimizer.step()
    verify_parameter_binding(backend.model, params, trainable)
    assert identities == {name: id(p) for name, p in backend.model.named_parameters()}
    assert optimizer_ids == [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert set(parameter_changes(params, before)) == set(trainable)
    assert all(p.device.type == "cpu" for p in params.values())
    backend.quiesce_training()


def test_offload_expert_hash_quiesces_real_controllers_and_retains_validation_state(torch, tmp_path, monkeypatch):
    from rswm_inria.training import tensor_state_sha256
    backend, _config, _camera = offload_backend(torch, tmp_path, monkeypatch)
    calls = []
    for tag, controller in backend._offload_controllers.items():
        original = controller.quiesce

        def observed(_original=original, _tag=tag):
            calls.append(_tag)
            return _original()

        monkeypatch.setattr(controller, "quiesce", observed)
    expected = {tag: tensor_state_sha256(expert.state_dict()) for tag, expert in backend.model.items()}
    assert backend.checkpoint_expert_state_sha256() == expected
    assert calls == ["low", "high"]
    backend._validation_forwards = 1
    checkpoint = backend.checkpoint_training_state()
    assert checkpoint == {"validation_forwards": 1}
    backend._validation_forwards = 2
    backend.load_checkpoint_training_state(checkpoint, step=1)
    assert backend._validation_forwards == 2, "An older checkpoint cannot delete consumed forwards"
    with pytest.raises(ValueError, match="retained validation"):
        backend.load_checkpoint_training_state({"validation_forwards": 0}, step=1)
    with torch.no_grad():
        backend.model["low"].blocks[0].self_attn.q.base_layer.weight.add_(1)
    changed = backend.checkpoint_expert_state_sha256()
    assert changed["low"] != expected["low"] and changed["high"] == expected["high"]
    assert calls == ["low", "high", "low", "high"]


@pytest.mark.parametrize("field", ["block_cpu_offload", "inactive_model_cpu", "t5_cpu"])
def test_training_offload_requires_every_explicit_sampler_setting(torch, tmp_path, monkeypatch, field):
    backend, config, _camera = offload_backend(torch, tmp_path, monkeypatch, configure=False)
    backend.runtime_config["sampling"][field] = False
    with pytest.raises(ValueError, match="requires locked"):
        backend.configure_trainable(config["trainable_policy"])
    assert not hasattr(backend, "_lora_training_configured")


@pytest.mark.parametrize("shape", [(1, 17, 3, 480, 832), (1, 33, 3, 240, 416), (1, 33, 1, 480, 832)])
def test_offload_does_not_relax_frozen_native_frames(torch, tmp_path, monkeypatch, shape):
    backend, _config, _camera = offload_backend(torch, tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="33-frame native 480x832"):
        backend.loss(images=torch.zeros(1).expand(shape), cameras=[{}], update=0)
    assert all(not expert.calls for expert in backend.model.values())


@pytest.mark.parametrize("field,value", [("num_train_timesteps", 999), ("boundary", .9),
                                        ("vae_stride", (2, 8, 8)), ("patch_size", (1, 1, 1))])
def test_offload_keeps_original_expert_sampler_mapping(torch, tmp_path, monkeypatch, field, value):
    backend, config, camera = offload_backend(torch, tmp_path, monkeypatch)
    setattr(backend.wan_config, field, value)
    images, cameras = native_batch(torch, config, camera, batch=1)
    with pytest.raises(ValueError, match="expert mapping changed"):
        backend.loss(images=images, cameras=cameras, update=0)
    assert not backend.vae.encodes and all(not expert.calls for expert in backend.model.values())


def locked_offload_config():
    value = backend_config()
    value["sampling"]["block_cpu_offload"] = True
    value["training"] = training_spec()
    value["training"]["expert_residency"] = "block_cpu"
    value["training"]["trainable_policy"]["gradient_checkpointing"] = True
    return value


def test_backend_config_accepts_only_explicit_matching_offload_profile():
    from rswm_inria.legacy_wan import validate_backend_config
    value = locked_offload_config()
    returned = validate_backend_config(value)
    assert returned == value
    returned["sampling"]["block_cpu_offload"] = False
    assert value["sampling"]["block_cpu_offload"] is True


@pytest.mark.parametrize("change", ["resident_with_offload", "block_without_flag", "block_flag_false",
                                     "no_checkpoint", "t5_gpu", "inactive_gpu", "unknown_flag", "flag_integer"])
def test_backend_config_rejects_mismatched_or_unknown_offload_profiles(change):
    from rswm_inria.legacy_wan import validate_backend_config
    value = locked_offload_config()
    if change == "resident_with_offload":
        value["training"]["expert_residency"] = "resident"
    elif change == "block_without_flag":
        del value["sampling"]["block_cpu_offload"]
    elif change == "block_flag_false":
        value["sampling"]["block_cpu_offload"] = False
    elif change == "no_checkpoint":
        value["training"]["trainable_policy"]["gradient_checkpointing"] = False
    elif change == "t5_gpu":
        value["sampling"]["t5_cpu"] = False
    elif change == "inactive_gpu":
        value["sampling"]["inactive_model_cpu"] = False
    elif change == "unknown_flag":
        value["sampling"]["automatically_approved_40gb"] = True
    else:
        value["sampling"]["block_cpu_offload"] = 1
    with pytest.raises(ValueError):
        validate_backend_config(value)


def test_driver_actual_offload_checkpoint_and_same_directory_resume(torch, tmp_path, monkeypatch):
    """Audited-source preflight is mocked, never a formal training receipt."""
    import rswm_inria.training as training
    import rswm_inria.adapters.camera_conditioning as conditioning
    from rswm_inria.geometry import digest
    monkeypatch.setattr(training, "validate_dev_index", lambda *_args: None)
    monkeypatch.setattr(conditioning, "to_native_rgb", lambda _frame: np.zeros((480, 832, 3), np.uint8))
    fixed_noise(torch, monkeypatch)

    class Dataset:
        diagnostic = False  # Explicit preflight mock; not an audited dataset.
        index_sha256 = "a" * 64
        source_audit_sha256 = "b" * 64
        root = tmp_path

        def __len__(self):
            return 1

        def __getitem__(self, _index):
            return {"images": np.zeros((33, 1, 1, 3), np.uint8), "q": np.zeros((33, 2), np.int64)}

    first, partial, _camera = offload_backend(torch, tmp_path, monkeypatch, configure=False)
    first.runtime_config["training"]["validation"]["max_total_forwards"] = 1
    config = {**partial, "updates": 1, "batch_size": 2, "accumulation": 1,
              "lr": .01, "seed": 23, "device": "cpu", "checkpoint_every": 1,
              "backend_manifest": {"purpose": "CPU_SOFTWARE_SIMULATION_NO_A100_OR_MODEL_ACCEPTANCE"}}
    events = []
    quiesce = first.quiesce_training
    checkpoint_state = first.checkpoint_training_state

    def observe_quiesce():
        events.append("quiesce")
        return quiesce()

    def observe_state():
        assert events[-1] == "quiesce", "Driver must quiesce before reading checkpoint state"
        events.append("checkpoint_state")
        return checkpoint_state()

    monkeypatch.setattr(first, "quiesce_training", observe_quiesce)
    monkeypatch.setattr(first, "checkpoint_training_state", observe_state)
    out = tmp_path / "driver"
    result = training.train(Dataset(), first, config, out)
    assert result["updates"] == 1 and result["clip_cursor"] == 2
    checkpoint = out / "update-00000001.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    receipt = json.loads(checkpoint.with_suffix(".json").read_text())
    assert state["backend_training"] == {"validation_forwards": 1}
    assert receipt["expert_state_sha256"] == first.checkpoint_expert_state_sha256()
    assert set(receipt["audit"][0]["active_trainable_groups"]) == {"low", "high"}
    assert "checkpoint_state" in events

    restored, _partial, _camera = offload_backend(torch, tmp_path, monkeypatch, configure=False)
    restored.runtime_config["training"]["validation"]["max_total_forwards"] = 1
    resumed = training.train(Dataset(), restored, config, out, resume=checkpoint)
    assert resumed["clip_cursor"] == 2 and resumed["updates"] == 1
    assert restored.checkpoint_training_state() == {"validation_forwards": 1}
    assert restored.checkpoint_expert_state_sha256() == receipt["expert_state_sha256"]
    for name, tensor in first.model.state_dict().items():
        assert torch.equal(tensor, restored.model.state_dict()[name])
    assert all(controller.report()["staged_frozen_bytes"] == 0
               for controller in restored._offload_controllers.values())
    assert receipt["training"]["backend_manifest_sha256"] == digest(config["backend_manifest"])
