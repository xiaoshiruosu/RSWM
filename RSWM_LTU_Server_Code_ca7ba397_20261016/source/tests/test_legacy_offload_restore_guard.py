"""F7 restore controls on the existing CPU software backend, never real GPUs."""
import numpy as np
import pytest

from test_legacy_wan import cpu_backend

from rswm_inria.legacy_offload import rswm_expert_forward
from rswm_inria.legacy_wan import LegacyWanBackend


def _offloaded(cpu_backend):
    original, _, _ = cpu_backend
    config = original._clone(original.runtime_config)
    config["sampling"]["block_cpu_offload"] = True
    return LegacyWanBackend(
        parts=(original.text_encoder, original.vae,
               original.model["low"], original.model["high"]),
        inference=original.inference, wan_config=original.wan_config,
        runtime_config=config, source_dir=original.source_dir,
        source_tree=original.source_tree, components=original.components,
        torch=original.torch, device=original.device,
    )


def _equal(torch, left, right):
    if torch.is_tensor(left):
        assert left.dtype == right.dtype and left.device == right.device
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _equal(torch, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _equal(torch, a, b)
    else:
        assert left == right


@pytest.mark.parametrize("tag", ["low", "high"])
def test_f7_offload_cuda_placement_rejected_before_transfer_or_mutation(cpu_backend, monkeypatch, tag):
    backend = _offloaded(cpu_backend)
    torch = backend.torch
    backend.reset(seed=734)
    state = backend.snapshot()
    state["mutable"]["placements"][tag] = "cuda:0"
    # This valid earlier buffer/mode would expose any partial restore.
    state["mutable"]["buffers"]["low"]["entered"].fill_(777)
    state["mutable"]["training_modes"]["low"][""] = True
    before = backend._clone(backend.resource_state())
    before_rng = backend._rng()
    identities = {name: (id(p), p.data_ptr()) for name, p in backend.model.named_parameters()}
    controllers = dict(backend._offload_controllers)
    counters = dict(backend.counters)

    def forbidden_transfer(*args, **kwargs):
        pytest.fail("Rejected F7 snapshot must never enter whole-module .to()")

    monkeypatch.setattr(torch.nn.Module, "to", forbidden_transfer)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda *a, **k: pytest.fail(
        "Placement rejection must precede CUDA RNG handling"))
    # Make the tampered target equal the selected CUDA address. This is a CPU
    # control, with no allocation: the original generic device scope check
    # would accept cuda:0 here, so it specifically exercises the new guard.
    backend.device = torch.device("cuda:0")
    try:
        with pytest.raises(ValueError, match="CPU expert placement"):
            backend.restore(state)
    finally:
        backend.device = torch.device("cpu")

    _equal(torch, before, backend.resource_state())
    _equal(torch, before_rng, backend._rng())
    assert backend.counters == counters
    assert controllers == backend._offload_controllers
    assert identities == {name: (id(p), p.data_ptr()) for name, p in backend.model.named_parameters()}
    assert all(manager.report()["staged_frozen_bytes"] == 0 for manager in controllers.values())
    for manager in controllers.values():
        manager.close()


def test_f7_cpu_restore_preserves_live_controller_parameter_identity_and_buffer_master(cpu_backend):
    backend = _offloaded(cpu_backend)
    torch = backend.torch
    backend.reset(seed=953)
    state = backend.snapshot()
    controllers = dict(backend._offload_controllers)
    identities = {name: (id(p), p.data_ptr()) for name, p in backend.model.named_parameters()}
    hooks = {id(module): (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
             for module in backend.model.modules()}
    assert state["mutable"]["placements"]["low"] == "cpu"
    assert state["mutable"]["placements"]["high"] == "cpu"
    latent = torch.zeros(16, 5, 1, 1)
    camera = torch.ones(2, 1)
    for tag in ("low", "high"):
        rswm_expert_forward(backend.model[tag], [latent], camera=camera)
        assert backend.model[tag].entered.item() == 1
    backend.vae.model._feat_map[0] = torch.tensor([735])
    backend.restore(state)
    assert backend.vae.model._feat_map[0] is None
    assert controllers == backend._offload_controllers
    assert identities == {name: (id(p), p.data_ptr()) for name, p in backend.model.named_parameters()}
    assert hooks == {id(module): (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
                     for module in backend.model.modules()}
    for tag in ("low", "high"):
        assert backend.model[tag].entered.item() == 0
        rswm_expert_forward(backend.model[tag], [latent], camera=camera)
        # A stale controller buffer master would start again at 1 and yield 2.
        assert backend.model[tag].entered.item() == 1
        report = controllers[tag].report()
        assert report["weight_onload_bytes"] > 0
        assert report["staged_frozen_bytes"] == 0
        assert report["cpu_simulation"] is True
        assert report["real_40gb_model_validation"] is False
        controllers[tag].close()
