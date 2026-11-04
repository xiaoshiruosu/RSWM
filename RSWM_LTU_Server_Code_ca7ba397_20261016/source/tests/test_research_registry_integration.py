"""Research adapters use current factories and lifetime resource contracts."""
import numpy as np
import pytest
from rswm_inria.registry import create_adapter, validate_config, capabilities, ADAPTERS
from rswm_inria import suite
from rswm_inria.frontier import AuditedWindowMap
from rswm_inria.baselines import WindowMap
from rswm_inria.resources import ResourceCollector, state_storage
from rswm_inria.missing_bounds import rgb_missing_bounds
from rswm_inria.aggregate import missing_bound

SPEC = {"viewport_hw": [240, 416], "query_bounds_xy": [[-704, 704], [-704, 704]]}


@pytest.mark.parametrize("kind", ["raster", "frontier", "lazy"])
def test_audited_factory_is_actual_and_strictly_explicit(kind):
    config = {"model_id": kind, "adapter": kind, "diagnostic_noise": True}
    model = create_adapter(config)
    assert isinstance(model, AuditedWindowMap) and model.order == kind
    assert suite.ADAPTERS is ADAPTERS
    assert capabilities(config)["purpose"] == "diagnostic"
    assert capabilities(config)["real_model_validation"] == "NOT_RUN"
    with pytest.raises(ValueError, match="Local weights"):
        validate_config({"model_id": kind, "adapter": kind})
    with pytest.raises(ValueError, match="consume"):
        validate_config(dict(config, prompt="unused"))
    with pytest.raises(ValueError, match="Unsupported"):
        validate_config(dict(config, private_origin=[1024, 1024]))


@pytest.mark.parametrize("kind", ["raster", "frontier", "lazy"])
def test_audited_counter_epoch_storage_and_collector_no_double_count(kind):
    m = create_adapter({"model_id": kind, "adapter": kind, "diagnostic_noise": True})
    collector = ResourceCollector("cpu")
    epoch = m.resource_counter_epoch
    with collector.phase("initialize", m):
        m.reset(np.zeros((240, 416, 3), np.uint8), SPEC, 0)
    with collector.phase("generate", m):
        m.generate([[480, 0]])
    parent = m.snapshot()
    with collector.phase("generate", m):
        m.generate([[0, 480]])
    before = m.resource_calls
    with collector.phase("restore", m):
        m.restore(parent)
    assert m.resource_calls == before and m.resource_counter_epoch == epoch
    with collector.phase("generate", m):
        m.generate([[0, -480]])
    measured = collector.finish()
    assert measured["adapter_counters"]["completion_calls"] == m.resource_calls
    assert m.resource_calls == len(m.operation_log)
    assert m.resource_generated_pixels == sum(r["committed_core_pixels"] for r in m.operation_log)
    assert set(m.resource_state()) == {"map", "known", "initial_mask", "operation_log"}
    assert state_storage(m.resource_state())["unique_numpy_buffer_bytes"] == 2048 * 2048 * 5
    assert "completer" not in parent["state"]


def test_real_factory_consumes_prompt_and_steps_with_mocked_constructor(monkeypatch, tmp_path):
    """Factory contract only; mock does not establish checkpoint/backend validity."""
    from rswm_inria import baselines
    (tmp_path / "config.json").write_text("{}")
    calls = []
    class ConstructorSpy:
        def __init__(self, *args):
            calls.append(args)
    monkeypatch.setattr(baselines, "DiffusersCompleter", ConstructorSpy)
    config = {"model_id": "real", "adapter": "frontier", "model_dir": str(tmp_path),
              "device": "cpu", "steps": 7, "prompt": "explicit aerial scene"}
    model = create_adapter(config)
    assert isinstance(model, AuditedWindowMap)
    assert calls == [(str(tmp_path), "cpu", 7, "explicit aerial scene")]


def test_missing_bounds_delegates_to_authoritative_formula(monkeypatch):
    from rswm_inria import aggregate
    calls = []
    real = aggregate.missing_bound
    def spy(error, coverage):
        calls.append((error, coverage))
        return real(error, coverage)
    monkeypatch.setattr(aggregate, "missing_bound", spy)
    result = rgb_missing_bounds(100, 95, 765)
    assert calls == [(765 / (765 * 95), .95)]
    assert [result["lower"], result["upper"]] == missing_bound(*calls[0])
