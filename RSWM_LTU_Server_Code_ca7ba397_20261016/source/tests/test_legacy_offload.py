"""Autograd equivalence, storage integrity and explicit CUDA-only coverage."""
import copy
import gc
import io

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.checkpoint import checkpoint

from rswm_inria.legacy_offload import FrozenCPUOffload, attach_frozen_cpu_offload, rswm_expert_forward


class AdapterLinear(nn.Module):
    """A composite owner with frozen base layer and live trainable adapters."""
    def __init__(self, width=6):
        super().__init__()
        self.base_layer = nn.Linear(width, width)
        self.lora_A = nn.Linear(width, 2, bias=False)
        self.lora_B = nn.Linear(2, width, bias=False)
        self.base_layer.requires_grad_(False)

    def forward(self, x):
        return self.base_layer(x) + self.lora_B(self.lora_A(x)) * .25


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.modulation = nn.Parameter(torch.randn(6), requires_grad=False)
        self.norm = nn.LayerNorm(6).requires_grad_(False)
        self.projection = AdapterLinear()
        self.register_buffer("calls", torch.tensor(0, dtype=torch.int64))

    def forward(self, x):
        self.calls.add_(1)
        return x + torch.tanh(self.projection(self.norm(x)) + self.modulation)


class TinyWan(nn.Module):
    def __init__(self, checkpointing=False):
        super().__init__()
        self.patch_embedding = nn.Linear(6, 6).requires_grad_(False)
        self.blocks = nn.ModuleList([Block() for _ in range(3)])
        self.head = nn.Linear(6, 4).requires_grad_(False)
        self.checkpointing = checkpointing

    def forward(self, x):
        assert self.patch_embedding.weight.device == x.device
        x = self.patch_embedding(x)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if self.checkpointing else block(x)
        return self.head(x)


def _trainable(model):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _frozen(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if not p.requires_grad}


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_real_backward_matches_resident_and_keeps_optimizer_identity(checkpointing, dtype):
    torch.manual_seed(731)
    resident = TinyWan(checkpointing).to(dtype=dtype)
    offloaded = copy.deepcopy(resident)
    optimizer = torch.optim.AdamW(_trainable(offloaded), lr=.003)
    reference_optimizer = torch.optim.AdamW(_trainable(resident), lr=.003)
    identities = {name: id(p) for name, p in offloaded.named_parameters()}
    frozen_before = _frozen(offloaded)
    manager = FrozenCPUOffload(offloaded, device="cpu", dtype=dtype)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        inputs = torch.randn(2, 7, 6, dtype=dtype)
        reference_input = inputs.detach().clone().requires_grad_(True)
        offloaded_input = inputs.detach().clone().requires_grad_(True)
        expected = resident(reference_input)
        with manager.forward_context():
            actual = offloaded(offloaded_input)
        # The offload context has ended before autograd, including recompute.
        expected.float().square().mean().backward()
        actual.float().square().mean().backward()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(offloaded_input.grad, reference_input.grad, rtol=0, atol=0)
        for actual_parameter, expected_parameter in zip(_trainable(offloaded), _trainable(resident)):
            assert actual_parameter.grad is not None
            torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
        manager.quiesce()
        assert manager.report()["staged_frozen_bytes"] == 0
        assert identities == {name: id(p) for name, p in offloaded.named_parameters()}
        for name, value in frozen_before.items():
            torch.testing.assert_close(dict(offloaded.named_parameters())[name], value, rtol=0, atol=0)
        for left, right in zip(_trainable(offloaded), _trainable(resident)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert manager.report()["cpu_simulation"] is True
    assert manager.report()["real_40gb_model_validation"] is False
    assert manager.report()["frozen_saved_view_bytes"] > 0
    assert manager.report()["maximum_staged_frozen_bytes"] < manager.report()["frozen_parameter_bytes"]
    manager.close()


def test_cross_expert_accumulation_checkpoint_state_and_rebind():
    torch.manual_seed(928)
    experts = nn.ModuleDict({tag: TinyWan(True) for tag in ("low", "high")})
    managers = {tag: FrozenCPUOffload(expert, device="cpu") for tag, expert in experts.items()}
    optimizer = torch.optim.AdamW(_trainable(experts), lr=.001)
    identities = [id(p) for p in optimizer.param_groups[0]["params"]]
    for tag in ("low", "high", "low"):
        with managers[tag].forward_context():
            loss = experts[tag](torch.randn(1, 9, 6)).square().mean()
        loss.backward()
    optimizer.step()
    for manager in managers.values():
        manager.quiesce()
    buffer = io.BytesIO()
    torch.save({"experts": experts.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=True)
    for manager in managers.values():
        manager.close()
    experts.load_state_dict(restored["experts"])
    optimizer.load_state_dict(restored["optimizer"])
    managers = {tag: attach_frozen_cpu_offload(expert, device="cpu") for tag, expert in experts.items()}
    assert identities == [id(p) for p in optimizer.param_groups[0]["params"]]
    optimizer.zero_grad(set_to_none=True)
    with managers["high"].forward_context():
        experts["high"](torch.randn(1, 9, 6)).square().mean().backward()
    assert all(p.grad is None for p in experts["low"].parameters() if p.requires_grad)
    assert all(p.grad is not None for p in experts["high"].parameters() if p.requires_grad)
    for manager in managers.values():
        manager.close()


def test_checkpoint_early_stop_releases_nonleaf_frozen_modulation_and_buffers():
    model = TinyWan(True)
    manager = FrozenCPUOffload(model, device="cpu")
    with manager.forward_context():
        loss = model(torch.randn(1, 5, 6)).sum()
    loss.backward()
    assert all(entry.count == 0 and entry.staged is None for entry in manager._entries.values())
    assert all(not stack for stack in manager._leases.values())
    assert not manager._prime_leases
    assert all(block.calls.item() == 2 for block in model.blocks)
    manager.close()


def test_frozen_data_mutation_is_detected_before_master_reset():
    class Mutating(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4), requires_grad=False)

        def forward(self, x):
            self.weight.data.add_(1)
            return x * self.weight

    model = Mutating()
    manager = FrozenCPUOffload(model, device="cpu")
    with pytest.raises(RuntimeError, match="Frozen parameter changed"):
        with manager.forward_context():
            model(torch.ones(4, requires_grad=True))
    assert manager.report()["failure"]
    assert manager.report()["staged_frozen_bytes"] == 0
    with pytest.raises(RuntimeError, match="offload failed"):
        with manager.forward_context():
            pass
    manager.close()


def test_scalar_frozen_signed_zero_mutation_is_not_hidden_by_value_equality():
    class Scalar(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.), requires_grad=False)

        def forward(self, x):
            self.weight.data.neg_()
            return x + self.weight

    model = Scalar()
    manager = FrozenCPUOffload(model, device="cpu")
    with pytest.raises(RuntimeError, match="Frozen parameter changed"):
        rswm_expert_forward(model, torch.ones(1, requires_grad=True))
    assert manager.report()["failure"]
    assert manager.report()["staged_frozen_bytes"] == 0
    manager.close()


def test_same_value_staging_storage_replacement_is_not_hidden_by_byte_equality():
    class StorageRewrite(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4), requires_grad=False)

        def forward(self, x):
            output = x * self.weight
            # Same values do not make this safe: checkpoint may already hold
            # this Tensor object, and its storage/device must stay immutable.
            self.weight.data = self.weight.detach().clone()
            return output

    model = StorageRewrite()
    original = model.weight
    manager = FrozenCPUOffload(model, device="cpu")
    with pytest.raises(RuntimeError, match="Frozen parameter changed"):
        rswm_expert_forward(model, torch.ones(4, requires_grad=True))
    assert model.weight is original
    assert manager.report()["failure"]
    assert manager.report()["staged_frozen_bytes"] == 0
    assert not manager._staged_storages
    manager.close()


def test_forward_exception_and_duplicate_controller_cleanup():
    model = TinyWan(True)
    manager = FrozenCPUOffload(model, device="cpu")
    with pytest.raises(ValueError, match="already has"):
        FrozenCPUOffload(model, device="cpu")

    def fail(module, args):
        raise RuntimeError("retained original failure")

    handle = model.blocks[1].projection.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="retained original failure"):
        with manager.forward_context():
            model(torch.randn(1, 5, 6))
    handle.remove()
    assert manager.report()["staged_frozen_bytes"] == 0
    assert all(not stack for stack in manager._leases.values())
    assert not manager._prime_leases
    manager.close()
    assert not hasattr(model, "_rswm_frozen_cpu_offload")
    assert all(not module._forward_pre_hooks and not module._forward_hooks for module in model.modules())


def test_trainable_backbone_rejected_and_saved_gpu_copies_are_not_cached():
    with pytest.raises(ValueError, match="LoRA parameters only"):
        FrozenCPUOffload(nn.Linear(3, 3), device="cpu")
    model = TinyWan()
    manager = FrozenCPUOffload(model, device="cpu")
    tensor = torch.randn(6)
    saved = manager._pack(tensor)
    first = manager._unpack(saved)
    second = manager._unpack(saved)
    assert first.data_ptr() != second.data_ptr()
    assert not hasattr(saved, "device_cache")
    manager.close()


def test_explicit_context_required_and_mixed_original_dtypes_preserved():
    model = TinyWan().to(dtype=torch.bfloat16)
    model.head.to(dtype=torch.float32)
    before = {name: parameter.dtype for name, parameter in model.named_parameters()}
    manager = FrozenCPUOffload(model, device="cpu")
    assert before == {name: parameter.dtype for name, parameter in model.named_parameters()}
    with pytest.raises(RuntimeError, match="forward_context"):
        model(torch.randn(1, 5, 6, dtype=torch.bfloat16))
    assert manager.report()["staged_frozen_bytes"] == 0
    manager.close()
    with pytest.raises(RuntimeError, match="controller missing"):
        rswm_expert_forward(model, torch.randn(1, 5, 6))


def test_shared_checkpoint_inputs_reuse_cpu_snapshots_without_gpu_or_pointer_cache():
    model = TinyWan()
    manager = FrozenCPUOffload(model, device="cpu")
    tensor = torch.randn(128, 6)
    first = manager._pack(tensor)
    second = manager._pack(tensor)
    assert first.cpu is second.cpu
    assert first.cpu.data_ptr() != tensor.data_ptr()
    assert manager.report()["saved_tensor_reused_bytes"] == tensor.numel() * tensor.element_size()
    tensor.add_(1)
    third = manager._pack(tensor)
    assert third.cpu is not first.cpu
    torch.testing.assert_close(third.cpu, first.cpu + 1, rtol=0, atol=0)
    del first, second, third
    gc.collect()
    assert manager.report()["saved_cpu_snapshot_cache_entries"] == 0
    saved = manager._pack(tensor)
    del tensor
    gc.collect()
    # Source lifetime cannot be extended by the cache; pointer reuse can never
    # retrieve this old snapshot for an unrelated subsequent CUDA allocation.
    assert manager.report()["saved_cpu_snapshot_cache_entries"] == 0
    assert saved.cpu.shape == (128, 6)
    manager.close()


def test_checkpoint_recomputation_frozen_saved_objects_keep_their_storage(monkeypatch):
    """Record actual Torch checkpoint saves; CPU detects .data alias changes.

    CUDA job 423050 saved frozen LayerNorm Parameters directly, then their
    .data was replaced with CPU storage. CPU cannot reproduce the device jump,
    but the same PyTorch 2.10 hook and the same mutable object alias are visible
    through a changed storage pointer. Do not disable checkpoint determinism.
    """
    import torch.utils.checkpoint as checkpoint_module

    original_hook = checkpoint_module._recomputation_hook
    saved = []

    class RecordingRecomputation(original_hook):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            original_pack = self.pack_hook

            def recording_pack(tensor):
                if not tensor.requires_grad:
                    saved.append((tensor, tensor.data_ptr(), tensor.device,
                                  tensor.dtype, tuple(tensor.shape)))
                return original_pack(tensor)

            self.pack_hook = recording_pack

    monkeypatch.setattr(checkpoint_module, "_recomputation_hook", RecordingRecomputation)
    model = TinyWan(True).to(dtype=torch.bfloat16)
    manager = FrozenCPUOffload(model, device="cpu", dtype=torch.bfloat16)
    canonical = {name: parameter for name, parameter in model.named_parameters()}
    with manager.forward_context():
        loss = model(torch.randn(1, 7, 6, dtype=torch.bfloat16)).float().square().mean()
    loss.backward()
    frozen_layernorm_saves = [item for item in saved if item[4] == (6,) and item[3] == torch.bfloat16]
    assert frozen_layernorm_saves, "The actual checkpoint must save frozen LayerNorm operands"
    for tensor, pointer, device, dtype, shape in saved:
        assert not isinstance(tensor, nn.Parameter), "Checkpoint must not cache a mutable canonical frozen Parameter"
        assert (tensor.data_ptr(), tensor.device, tensor.dtype, tuple(tensor.shape)) == (pointer, device, dtype, shape)
    assert all(dict(model.named_parameters())[name] is parameter for name, parameter in canonical.items())
    assert all(isinstance(parameter, nn.Parameter) for parameter in model.parameters())
    assert manager.report()["staged_frozen_bytes"] == 0
    manager.close()


def test_shared_frozen_parameter_slots_restore_one_canonical_parameter():
    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(4, 4, bias=False).requires_grad_(False)
            self.second = nn.Linear(4, 4, bias=False).requires_grad_(False)
            self.second.weight = self.first.weight
            self.lora_offset = nn.Parameter(torch.ones(4))

        def forward(self, x):
            return self.second(self.first(x) + self.lora_offset)

    torch.manual_seed(157)
    model = Shared()
    expected_model = copy.deepcopy(model)
    weight = model.first.weight
    manager = FrozenCPUOffload(model, device="cpu")
    inputs = torch.randn(2, 4)
    expected = expected_model(inputs)
    actual = rswm_expert_forward(model, inputs)
    expected.sum().backward()
    actual.sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.lora_offset.grad, expected_model.lora_offset.grad, rtol=0, atol=0)
    assert model.first.weight is model.second.weight is weight
    assert manager.report()["staged_frozen_bytes"] == 0
    manager.close()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_actual_peft_composite_wrapper_gradient_identity_and_restore(checkpointing):
    peft = pytest.importorskip("peft")
    torch.manual_seed(547)
    reference = peft.get_peft_model(TinyWan(checkpointing), peft.LoraConfig(
        r=2, lora_alpha=2, target_modules=["base_layer"], init_lora_weights=False))
    candidate = copy.deepcopy(reference)
    identities = {name: id(parameter) for name, parameter in candidate.named_parameters()}
    optimizer = torch.optim.AdamW(_trainable(candidate), lr=.001)
    reference_optimizer = torch.optim.AdamW(_trainable(reference), lr=.001)
    manager = FrozenCPUOffload(candidate, device="cpu")
    x = torch.randn(1, 7, 6)
    expected = reference(x)
    actual = rswm_expert_forward(candidate, x)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for left, right in zip(_trainable(candidate), _trainable(reference)):
        assert left.grad is not None
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    optimizer.step()
    reference_optimizer.step()
    assert identities == {name: id(parameter) for name, parameter in candidate.named_parameters()}
    for left, right in zip(_trainable(candidate), _trainable(reference)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    manager.quiesce()
    checkpoint_state = copy.deepcopy(candidate.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    manager.close()
    candidate.load_state_dict(checkpoint_state)
    optimizer.load_state_dict(optimizer_state)
    manager = attach_frozen_cpu_offload(candidate, device="cpu")
    assert identities == {name: id(parameter) for name, parameter in candidate.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    rswm_expert_forward(candidate, x).square().mean().backward()
    assert all(parameter.grad is not None for parameter in _trainable(candidate))
    manager.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required; CPU regressions do not validate GPU memory")
@pytest.mark.parametrize("checkpointing", [False, True])
def test_cuda_actual_device_transfer_gradient_and_residency(checkpointing):
    device = torch.device("cuda")
    torch.manual_seed(943)
    resident = TinyWan(checkpointing).to(device=device, dtype=torch.bfloat16)
    model = copy.deepcopy(resident).cpu()
    manager = FrozenCPUOffload(model, device=device, dtype=torch.bfloat16)
    identities = {name: id(p) for name, p in model.named_parameters()}
    x = torch.randn(2, 7, 6, device=device, dtype=torch.bfloat16)
    reference_x = x.clone().requires_grad_(True)
    candidate_x = x.clone().requires_grad_(True)
    expected = resident(reference_x)
    with manager.forward_context():
        actual = model(candidate_x)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate_x.grad, reference_x.grad, rtol=0, atol=0)
    for candidate, reference in zip(_trainable(model), _trainable(resident)):
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=0, atol=0)
    assert identities == {name: id(p) for name, p in model.named_parameters()}
    assert all(p.device.type == ("cuda" if p.requires_grad else "cpu") for p in model.parameters())
    report = manager.report()
    assert report["maximum_staged_frozen_bytes"] < report["frozen_parameter_bytes"]
    assert report["staged_frozen_bytes"] == 0
    assert report["real_40gb_model_validation"] is False
    manager.close()
