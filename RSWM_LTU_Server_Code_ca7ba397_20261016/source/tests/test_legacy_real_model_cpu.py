"""CPU numerical checks of the retained WanModel, not pretrained/A100 evidence.

Only the vendor package initializer is bypassed: it imports a T5 constructor
whose default argument requests a CUDA device. The actual model.py and
attention.py execute unchanged under an isolated package name. Models and
inputs are deliberately tiny synthetic software controls.
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import types

import pytest


@pytest.fixture(scope="module")
def real_wan_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("einops")
    root = Path(__file__).resolve().parents[1]
    modules = root / "vendor/legacy-original/lora_remotesense/new/wan/modules"
    package_name = "_rswm_retained_wan_cpu_numeric"
    package = types.ModuleType(package_name)
    package.__path__ = [str(modules)]
    before = set(sys.modules)
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(package_name + ".model", modules / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        spec.loader.exec_module(module)
        assert not any(name == "wan" or name.startswith("wan.") for name in set(sys.modules) - before)
        yield torch, module.WanModel
    finally:
        torch.set_num_threads(previous_threads)
        for name in list(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)


def _model(torch, model_class, *, dtype=None, text_dim=24):
    torch.manual_seed(17)
    model = model_class(model_type="i2v", in_dim=36, out_dim=16, dim=32,
                        ffn_dim=64, freq_dim=8, text_dim=text_dim, text_len=6,
                        num_heads=2, num_layers=2)
    # Vendor initializes the output projection to zero. Give this synthetic
    # control a nonzero output projection so upstream gradients are observable.
    with torch.no_grad():
        model.head.head.weight.normal_(std=0.07)
    return model.to(device="cpu", dtype=dtype or torch.float32).train()


def _inputs(torch, dtype, *, text_dim=24):
    generator = torch.Generator(device="cpu").manual_seed(29)
    def sample(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype)
    return {"x": [sample((16, 2, 4, 4)).requires_grad_()],
            "y": [sample((20, 2, 4, 4)).requires_grad_()],
            "t": torch.tensor([932.0]), "context": [sample((3, text_dim))],
            "seq_len": 8,
            "dit_cond_dict": {"c2ws_plucker_emb": [sample((1, 384, 2, 4, 4)).requires_grad_()]}}


def _forward_backward(torch, model, inputs):
    output = model(**inputs)[0]
    assert output.shape == (16, 2, 4, 4) and output.dtype == torch.float32
    assert torch.isfinite(output).all()
    loss = output.square().mean() + output.mean() * 0.01
    loss.backward()
    assert torch.isfinite(loss)
    gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                 if parameter.grad is not None}
    assert gradients and all(torch.isfinite(value).all() for value in gradients.values())
    return output.detach(), loss.detach(), gradients


def _cpu_emulation_of_cuda_autocast(monkeypatch, torch):
    """Model CUDA runtime contexts on CPU; this does not exercise CUDA kernels.

    The vendor's CUDA float32 autocast context disables CUDA autocast because
    CUDA autocast supports fp16/bf16. An outer CPU bf16 context otherwise has a
    different scope, so map only these runtime contexts to CPU-disabled ones.
    """
    original = torch.amp.autocast
    def controlled(device_type, *args, **kwargs):
        if device_type == "cuda" and kwargs.get("dtype") == torch.float32:
            return original("cpu", enabled=False)
        return original(device_type, *args, **kwargs)
    monkeypatch.setattr(torch.amp, "autocast", controlled)


def test_retained_fp32_camera_root_and_head_backward(real_wan_cpu, monkeypatch):
    torch, model_class = real_wan_cpu
    monkeypatch.setenv("RSWM_ATTENTION_BACKEND", "sdpa")
    model = _model(torch, model_class)
    inputs = _inputs(torch, torch.float32)
    _, _, gradients = _forward_backward(torch, model, inputs)
    for name in ("patch_embedding.weight", "time_embedding.0.weight", "time_projection.1.weight",
                 "text_embedding.0.weight", "patch_embedding_wancamctrl.weight",
                 "c2ws_hidden_states_layer1.weight", "blocks.0.cam_scale_layer.weight",
                 "blocks.1.self_attn.q.weight", "head.head.weight"):
        assert name in gradients and torch.count_nonzero(gradients[name])
    for value in (inputs["x"][0], inputs["y"][0], inputs["dit_cond_dict"]["c2ws_plucker_emb"][0]):
        assert value.grad is not None and torch.isfinite(value.grad).all() and torch.count_nonzero(value.grad)


def test_retained_full_bf16_fails_at_fp32_time_input(real_wan_cpu, monkeypatch):
    torch, model_class = real_wan_cpu
    monkeypatch.setenv("RSWM_ATTENTION_BACKEND", "sdpa")
    model = _model(torch, model_class, dtype=torch.bfloat16)
    observed = []
    handle = model.time_embedding[0].register_forward_pre_hook(
        lambda module, args: observed.append((args[0].dtype, module.weight.dtype)))
    try:
        with pytest.raises(RuntimeError, match="same dtype|scalar type"):
            model(**_inputs(torch, torch.bfloat16))
    finally:
        handle.remove()
    assert observed == [(torch.float32, torch.bfloat16)]


def test_retained_bf16_autocast_requires_fp32_time_norm_and_head(real_wan_cpu, monkeypatch):
    torch, model_class = real_wan_cpu
    monkeypatch.setenv("RSWM_ATTENTION_BACKEND", "sdpa")
    _cpu_emulation_of_cuda_autocast(monkeypatch, torch)
    model = _model(torch, model_class, dtype=torch.bfloat16)
    inputs = _inputs(torch, torch.bfloat16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="same dtype|scalar type"):
            model(**inputs)
    # Test-only proposed precision layout; production code is not changed.
    model.time_embedding.float()
    model.time_projection.float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="mixed dtype.*Float"):
            model(**inputs)
    # WanLayerNorm explicitly feeds float32 to its affine normalization.
    for block in model.blocks:
        block.norm3.float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="same dtype|scalar type"):
            model(**inputs)
    model.head.head.float()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _forward_backward(torch, model, _inputs(torch, torch.bfloat16))


def test_retained_checkpoint_lora_gradients_match_without_checkpoint(real_wan_cpu, monkeypatch):
    torch, model_class = real_wan_cpu
    peft = pytest.importorskip("peft")
    monkeypatch.setenv("RSWM_ATTENTION_BACKEND", "sdpa")
    original = _model(torch, model_class)
    for parameter in original.parameters():
        parameter.requires_grad_(False)
    policy = peft.LoraConfig(r=2, lora_alpha=2, target_modules=[
        "blocks.0.self_attn.q", "blocks.1.self_attn.q",
        "blocks.0.cam_scale_layer", "blocks.1.cam_scale_layer"], init_lora_weights=True)
    resident = peft.get_peft_model(original, policy)
    checkpointed = copy.deepcopy(resident)
    checkpointed.get_base_model().gradient_checkpointing_enable()
    before = {name: parameter.detach().clone() for name, parameter in checkpointed.named_parameters()
              if not parameter.requires_grad}
    bindings = {name: id(parameter) for name, parameter in checkpointed.named_parameters()}
    output_a, loss_a, grads_a = _forward_backward(torch, resident, _inputs(torch, torch.float32))
    output_b, loss_b, grads_b = _forward_backward(torch, checkpointed, _inputs(torch, torch.float32))
    torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    assert set(grads_a) == set(grads_b) and all("lora_" in name for name in grads_b)
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], rtol=0, atol=0)
    for name, parameter in checkpointed.named_parameters():
        assert id(parameter) == bindings[name]
        if name in before:
            assert parameter.grad is None and torch.equal(parameter.detach(), before[name])
    assert any("cam_scale_layer" in name and torch.count_nonzero(value) for name, value in grads_b.items())
    assert any("self_attn.q" in name and torch.count_nonzero(value) for name, value in grads_b.items())


def test_retained_cpu_checkpoint_resume_preserves_lora_identity(real_wan_cpu, monkeypatch, tmp_path):
    torch, model_class = real_wan_cpu
    peft = pytest.importorskip("peft")
    monkeypatch.setenv("RSWM_ATTENTION_BACKEND", "sdpa")
    original = _model(torch, model_class)
    original.requires_grad_(False)
    config = peft.LoraConfig(r=2, lora_alpha=2,
                             target_modules=["blocks.0.self_attn.q", "blocks.1.cam_scale_layer"])
    continuous = peft.get_peft_model(original, config)
    continuous.get_base_model().gradient_checkpointing_enable()
    resumed = copy.deepcopy(continuous)
    bindings = {name: id(parameter) for name, parameter in resumed.named_parameters()}
    frozen = {name: parameter.detach().clone() for name, parameter in continuous.named_parameters()
              if not parameter.requires_grad}
    def optimizer_for(model):
        return torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        result = _forward_backward(torch, model, _inputs(torch, torch.float32))
        optimizer.step()
        return result
    optimizer = optimizer_for(continuous)
    update(continuous, optimizer)
    checkpoint = tmp_path / "tiny-retained-wan-cpu.pt"
    torch.save({"model": continuous.state_dict(), "optimizer": optimizer.state_dict(),
                "torch_rng": torch.get_rng_state()}, checkpoint)
    output_a, loss_a, grads_a = update(continuous, optimizer)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    resumed.load_state_dict(state["model"], strict=True)
    resumed_optimizer = optimizer_for(resumed)
    resumed_optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    output_b, loss_b, grads_b = update(resumed, resumed_optimizer)
    torch.testing.assert_close(output_a, output_b, rtol=0, atol=0)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    assert set(grads_a) == set(grads_b)
    for name in grads_a:
        torch.testing.assert_close(grads_a[name], grads_b[name], rtol=0, atol=0)
    expected = dict(continuous.named_parameters())
    for name, parameter in resumed.named_parameters():
        assert id(parameter) == bindings[name]
        assert torch.equal(parameter, expected[name])
        if name in frozen:
            assert parameter.grad is None and torch.equal(parameter, frozen[name])


def test_retained_diffusers_keep_fp32_exact_token_loading(real_wan_cpu, monkeypatch, tmp_path):
    """Check real local Diffusers loading, including FP32 source-bit retention."""
    torch, model_class = real_wan_cpu
    monkeypatch.setenv("RSWM_ATTENTION_BACKEND", "sdpa")
    # text_dim is intentionally omitted from the original saved config. Use
    # its true constructor default in this local loader round-trip control.
    source = _model(torch, model_class, text_dim=4096)
    source.save_pretrained(tmp_path, safe_serialization=True)
    keep = ["time_embedding", "time_projection", "head", "norm3"]
    monkeypatch.setattr(model_class, "_keep_in_fp32_modules", keep, raising=False)
    loaded, info = model_class.from_pretrained(tmp_path, torch_dtype=torch.bfloat16,
                                               local_files_only=True, output_loading_info=True)
    assert all(not info[key] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"))
    reference = source.state_dict()
    assert set(reference) == set(loaded.state_dict())
    for name, tensor in loaded.state_dict().items():
        fp32 = any(token in name.split(".") for token in keep)
        assert tensor.dtype == (torch.float32 if fp32 else torch.bfloat16), name
        assert torch.equal(tensor, reference[name] if fp32 else reference[name].to(torch.bfloat16)), name
    assert loaded.head.modulation.dtype == torch.float32
    assert loaded.blocks[0].norm3.weight.dtype == torch.float32
    assert loaded.blocks[0].self_attn.q.weight.dtype == torch.bfloat16
    _cpu_emulation_of_cuda_autocast(monkeypatch, torch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _forward_backward(torch, loaded, _inputs(torch, torch.bfloat16, text_dim=4096))


@pytest.mark.parametrize("checkpointing", [False, True])
def test_retained_real_wan_cpu_offload_matches_lora_gradients(real_wan_cpu, monkeypatch, checkpointing):
    """Exercise production staging on real Wan; CPU simulation has no GPU claim."""
    torch, model_class = real_wan_cpu
    peft = pytest.importorskip("peft")
    from rswm_inria.legacy_offload import FrozenCPUOffload
    monkeypatch.setenv("RSWM_ATTENTION_BACKEND", "sdpa")
    original = _model(torch, model_class)
    original.requires_grad_(False)
    policy = peft.LoraConfig(r=2, lora_alpha=2, target_modules=[
        "blocks.0.self_attn.q", "blocks.1.self_attn.q",
        "blocks.0.cam_scale_layer", "blocks.1.cam_scale_layer"])
    resident = peft.get_peft_model(original, policy)
    if checkpointing:
        resident.get_base_model().gradient_checkpointing_enable()
    offloaded = copy.deepcopy(resident)
    bindings = {name: id(parameter) for name, parameter in offloaded.named_parameters()}
    frozen = {name: parameter.detach().clone() for name, parameter in offloaded.named_parameters()
              if not parameter.requires_grad}
    output_a, loss_a, gradients_a = _forward_backward(torch, resident, _inputs(torch, torch.float32))
    controller = FrozenCPUOffload(offloaded, device="cpu", dtype=None)
    try:
        inputs = _inputs(torch, torch.float32)
        with controller.forward_context():
            output_b = offloaded(**inputs)[0]
            loss_b = output_b.square().mean() + output_b.mean() * 0.01
        # The real driver returns from backend.loss before calling backward.
        # Non-reentrant checkpoint recomputation must still stage the weights.
        loss_b.backward()
        controller.quiesce()
        gradients_b = {name: parameter.grad.detach().clone() for name, parameter in offloaded.named_parameters()
                       if parameter.grad is not None}
        torch.testing.assert_close(output_a, output_b.detach(), rtol=0, atol=0)
        torch.testing.assert_close(loss_a, loss_b.detach(), rtol=0, atol=0)
        assert set(gradients_a) == set(gradients_b)
        for name in gradients_a:
            torch.testing.assert_close(gradients_a[name], gradients_b[name], rtol=0, atol=0)
        for name, parameter in offloaded.named_parameters():
            assert id(parameter) == bindings[name]
            if name in frozen:
                assert parameter.grad is None and torch.equal(parameter, frozen[name])
        report = controller.report()
        assert report["cpu_simulation"] is True and report["real_40gb_model_validation"] is False
        assert report["staged_frozen_bytes"] == 0 and report["frozen_saved_view_bytes"] > 0
    finally:
        controller.close()
