import numpy as np
import pytest
from rswm_inria.history_mechanisms import HistoryCompleterAdapter, POLICY_SCHEMA
from rswm_inria.resources import ResourceCollector


def policy(mechanism="last_frame", capacity=1, initial=True, history=True):
    return {"schema": POLICY_SCHEMA, "mechanism": mechanism, "capacity": capacity,
            "protect_initial": initial, "protect_history": history}


def spec():
    return {"world_hw": [1024, 1024], "viewport_hw": [240, 416], "query_bounds_xy": [[-288, 288], [-288, 288]]}


class Fill:
    def __init__(self):
        self.inputs = []
    def __call__(self, image, known, seed):
        self.inputs.append((image.copy(), known.copy(), seed))
        return np.full(image.shape, len(self.inputs)*17 % 256, np.uint8)


def test_initial_and_history_protection_independent():
    initial = np.full((240, 416, 3), 9, np.uint8)
    for protect_i0, expected in ((True, 9), (False, 17)):
        completer = Fill(); adapter = HistoryCompleterAdapter(completer, policy(initial=protect_i0, history=True))
        adapter.reset(initial, spec(), 0)
        output = adapter.generate(np.array([[0, 0]]))[0]
        assert np.all(output == expected)
    completer = Fill(); adapter = HistoryCompleterAdapter(completer, policy(initial=False, history=True))
    adapter.reset(initial, spec(), 0)
    first, second = adapter.generate(np.array([[256, 256], [256, 256]]))
    assert np.array_equal(first, second)
    completer = Fill(); adapter = HistoryCompleterAdapter(completer, policy(initial=False, history=False))
    adapter.reset(initial, spec(), 0)
    first, second = adapter.generate(np.array([[256, 256], [256, 256]]))
    assert not np.array_equal(first, second)


def test_history_mechanisms_retrieve_different_committed_state():
    initial = np.zeros((240, 416, 3), np.uint8)
    traces = {}
    outputs = {}
    for name, cap in (("last_frame", 1), ("frame_cache", 3), ("map_assisted", 1)):
        adapter = HistoryCompleterAdapter(Fill(), policy(name, cap, initial=False))
        adapter.reset(initial, spec(), 0)
        frames = adapter.generate(np.array([[256, 256], [-256, -256], [256, 256]]))
        traces[name] = adapter.retrieval_trace[-1]; outputs[name] = frames[-1]
        assert len(adapter.cache) <= cap
    assert traces["last_frame"]["selected_frame_q"] == [-256, -256]
    assert traces["frame_cache"]["selected_frame_q"] == [256, 256]
    assert traces["map_assisted"]["selected_frame_q"] is None
    assert np.all(outputs["last_frame"] == 51)
    assert np.all(outputs["frame_cache"] == 17)
    assert np.all(outputs["map_assisted"] == 17)


def test_branch_restore_reproduces_outputs_without_rolling_back_cost():
    from rswm_inria.baselines import NoiseCompleter
    adapter = HistoryCompleterAdapter(NoiseCompleter(), policy("frame_cache", 3))
    collector = ResourceCollector("cpu")
    with collector.phase("initialize", adapter):
        adapter.reset(np.zeros((240, 416, 3), np.uint8), spec(), 42)
    with collector.phase("generate", adapter, count=1):
        adapter.generate(np.array([[256, 256]]))
    snapshot = adapter.snapshot()
    with collector.phase("generate", adapter, count=1):
        a = adapter.generate(np.array([[-256, -256]]))[0]
    with collector.phase("restore", adapter):
        adapter.restore(snapshot)
    with collector.phase("generate", adapter, count=1):
        b = adapter.generate(np.array([[-256, -256]]))[0]
    assert np.array_equal(a, b)
    resource = collector.finish()
    assert resource["adapter_counters"]["completion_calls"] == 3
    assert resource["adapter_counters"]["native_frames"] == 3
    assert resource["measurements"]["dynamic_state_bytes"]["value"] > 3*1024*1024


def test_no_future_trajectory_accepted_and_bounds_enforced():
    adapter = HistoryCompleterAdapter(Fill(), policy())
    adapter.reset(np.zeros((240, 416, 3), np.uint8), spec(), 0)
    with pytest.raises(ValueError):
        adapter.generate(np.array([[289, 0]]))
    with pytest.raises(ValueError):
        adapter.generate(np.zeros((65, 2), int))
    assert "world" not in adapter.__dict__
    with pytest.raises(ValueError):
        adapter.generate(np.zeros((0, 2), int))


def test_snapshot_cannot_overwrite_cumulative_counter_lifetime():
    adapter = HistoryCompleterAdapter(Fill(), policy())
    adapter.reset(np.zeros((240, 416, 3), np.uint8), spec(), 0)
    state = adapter.snapshot(); state["counters"] = {"native_frames": 0}
    with pytest.raises(ValueError, match="logical state"):
        adapter.restore(state)


@pytest.mark.parametrize("bad", [policy("last_frame", 2), policy("invalid"), {"schema": POLICY_SCHEMA}])
def test_invalid_policy_rejected(bad):
    with pytest.raises(ValueError):
        HistoryCompleterAdapter(Fill(), bad)
