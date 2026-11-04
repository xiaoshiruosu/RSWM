"""Real autograd alias regression for non-reentrant checkpoint metadata.

CPU tests use an exact-valued FP32/FP64 storage swap to expose the same mutable
Parameter-reference problem as the observed CUDA/CPU swap. Neither disables
determinism checks. CUDA coverage is optional and is not certified by CPU runs.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.checkpoint import CheckpointError, checkpoint


def test_parameter_data_swap_mutates_parameter_metadata_but_not_detached_view():
    parameter = nn.Parameter(torch.ones(6, dtype=torch.float32), requires_grad=False)
    direct = parameter
    detached = parameter.detach()
    identity = id(parameter)
    parameter.data = parameter.data.to(torch.float64)
    assert direct is parameter and id(parameter) == identity
    assert direct.dtype == torch.float64
    assert detached.dtype == torch.float32
    assert torch.equal(detached, torch.ones(6))


def test_layernorm_saves_direct_frozen_parameter_objects_at_observed_positions():
    norm = nn.LayerNorm(6).requires_grad_(False)
    saved = []
    def pack(value):
        saved.append((tuple(value.shape), value is norm.bias, value is norm.weight,
                      isinstance(value, nn.Parameter), value.requires_grad))
        return value.detach()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        result = norm(torch.randn(2, 7, 6, requires_grad=True))
    result.square().mean().backward()
    # These are exactly the two shape[6] positions reported by CUDA job 423050.
    assert saved[0] == ((6,), True, False, True, False)
    assert saved[2] == ((6,), False, True, True, False)


class _StorageSwapNorm(nn.Module):
    """A deterministic small reproduction, independent of production engine."""
    def __init__(self, *, detached_compute, execution_device="cpu"):
        super().__init__()
        self.norm = nn.LayerNorm(6).requires_grad_(False)
        self.detached_compute = detached_compute
        self.execution_device = torch.device(execution_device)
        self.execution_dtype = torch.bfloat16 if self.execution_device.type == "cuda" else torch.float32
        self.storage_dtype = self.execution_dtype if self.execution_device.type == "cuda" else torch.float64
        self.norm.to(dtype=self.storage_dtype)
        self.masters = {name: parameter.detach().clone() for name, parameter in self.norm.named_parameters()}
        self.identities = {name: id(parameter) for name, parameter in self.norm.named_parameters()}
        self.saw_compute_dtypes = []

    def forward(self, value):
        for name, parameter in self.norm.named_parameters():
            parameter.data = self.masters[name].to(self.execution_device, dtype=self.execution_dtype, copy=True)
        self.saw_compute_dtypes.append((self.norm.weight.dtype, self.norm.weight.device.type))
        try:
            if self.detached_compute:
                # TensorImpl metadata remains stable when the original frozen
                # Parameter's .data is restored after this operator returns.
                return torch.nn.functional.layer_norm(value, (6,), self.norm.weight.detach(),
                                                       self.norm.bias.detach(), self.norm.eps)
            return self.norm(value)
        finally:
            for name, parameter in self.norm.named_parameters():
                parameter.data = self.masters[name]

    def assert_restored(self):
        assert self.identities == {name: id(parameter) for name, parameter in self.norm.named_parameters()}
        for name, parameter in self.norm.named_parameters():
            assert parameter.device.type == "cpu" and parameter.grad is None
            assert parameter.requires_grad is False and torch.equal(parameter, self.masters[name])


def _input(device="cpu"):
    generator = torch.Generator(device="cpu").manual_seed(451)
    value = torch.randn((2, 3, 6), generator=generator)
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    return value.to(device=device, dtype=dtype).requires_grad_()


def _loss(value):
    factors = torch.arange(1, 7, device=value.device, dtype=torch.float32)
    return (value.float() * factors).mean() + value.float().square().mean() * 0.1


def test_checkpoint_mutable_frozen_parameter_metadata_is_not_stable():
    model = _StorageSwapNorm(detached_compute=False)
    output = checkpoint(model, _input(), use_reentrant=False)
    with pytest.raises(CheckpointError, match="different metadata") as failure:
        _loss(output).backward()
    message = str(failure.value)
    assert "torch.Size([6])" in message and "torch.float32" in message and "torch.float64" in message
    model.assert_restored()


def test_checkpoint_detached_frozen_compute_views_preserve_exact_math_and_identity():
    model = _StorageSwapNorm(detached_compute=True)
    reference = nn.LayerNorm(6).requires_grad_(False)
    candidate_input, reference_input = _input(), _input()
    expected = reference(reference_input)
    actual = checkpoint(model, candidate_input, use_reentrant=False)
    _loss(expected).backward()
    _loss(actual).backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate_input.grad, reference_input.grad, rtol=0, atol=0)
    assert model.saw_compute_dtypes == [(torch.float32, "cpu"), (torch.float32, "cpu")]
    model.assert_restored()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Allocated CUDA required; CPU is not GPU evidence")
def test_cuda_checkpoint_metadata_reproducer_and_detached_view_solution():
    device = torch.device("cuda")
    broken = _StorageSwapNorm(detached_compute=False, execution_device=device)
    with pytest.raises(CheckpointError, match="different metadata") as failure:
        _loss(checkpoint(broken, _input(device), use_reentrant=False)).backward()
    message = str(failure.value)
    assert "device(type='cuda'" in message and "device(type='cpu')" in message
    broken.assert_restored()
    fixed = _StorageSwapNorm(detached_compute=True, execution_device=device)
    reference = nn.LayerNorm(6).requires_grad_(False).to(device=device, dtype=torch.bfloat16)
    candidate_input, reference_input = _input(device), _input(device)
    actual = checkpoint(fixed, candidate_input, use_reentrant=False)
    expected = reference(reference_input)
    _loss(actual).backward()
    _loss(expected).backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(candidate_input.grad, reference_input.grad, rtol=0, atol=0)
    fixed.assert_restored()
