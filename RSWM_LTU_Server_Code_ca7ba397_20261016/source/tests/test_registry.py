import numpy as np
import pytest
import sys
from types import SimpleNamespace
from PIL import Image
from rswm_inria.registry import validate_config, create_adapter, capabilities
from rswm_inria.baselines import WindowMap


def test_unknown_configuration_is_rejected_instead_of_ignored():
    with pytest.raises(ValueError, match="Unsupported"):
        validate_config({"model_id": "c", "adapter": "copy", "prompt": "ignored"})
    with pytest.raises(ValueError, match="consume"):
        validate_config({"model_id": "n", "adapter": "lazy-map", "diagnostic_noise": True, "steps": 3})


def test_real_generator_requires_nonempty_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="Nonempty"):
        validate_config({"model_id": "w", "adapter": "wan-i2v", "model_dir": str(tmp_path)})


def test_ordinary_factory_receives_no_private_task_or_world():
    m = create_adapter({"model_id": "c", "adapter": "copy"})
    m.reset(np.zeros((240, 416, 3), np.uint8), {}, 0)
    assert m.generate([[0, 0]])[0].shape == (240, 416, 3)
    assert capabilities({"model_id": "c", "adapter": "copy"})["purpose"] == "diagnostic"
    with pytest.raises(ValueError, match="evaluator-only"):
        create_adapter({"model_id": "o", "adapter": "oracle"})


def test_counter_epoch_is_not_rolled_back_by_reset_or_restore():
    m = WindowMap()
    epoch = m.resource_counter_epoch
    initial = np.zeros((240, 416, 3), np.uint8)
    m.reset(initial, {}, 0)
    state = m.snapshot()
    m.generate([[300, 300]])
    calls = m.resource_calls
    m.restore(state)
    assert m.resource_counter_epoch == epoch and m.resource_calls == calls
    m.reset(initial, {}, 0)
    assert m.resource_counter_epoch == epoch and m.resource_calls == calls


def test_mock_native_frame_conversion_failure_counts_actual_work(monkeypatch):
    """Control-flow mock only; neither torch nor real weights are loaded."""
    from rswm_inria.video import WanI2VAdapter
    class Generator:
        def __init__(self, **kwargs): pass
        def manual_seed(self, seed): return self
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(Generator=Generator))
    output = [Image.new("RGB", (16, 16)) for _ in range(17)]
    output[8] = Image.new("L", (16, 16))
    m = WanI2VAdapter(device="cpu", pipe=lambda **kwargs: SimpleNamespace(frames=[output]))
    m.reset(np.zeros((240, 416, 3), np.uint8), {}, 0)
    with pytest.raises(ValueError, match="RGB"):
        m.generate([[0, 0]] * 16)
    assert m.resource_counters() == {"native_frames": 17, "context_frames": 1,
                                     "discarded_frames": 16, "submitted_native_frames": 0}
