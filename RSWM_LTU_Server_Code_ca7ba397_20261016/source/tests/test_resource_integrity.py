"""Resource contract regressions import the active package. All GPU APIs are synthetic."""
from types import SimpleNamespace

import numpy as np
import pytest

from rswm_inria.resources import ResourceCollector, state_storage


class SyntheticCUDA:
    """Deterministic observer failures; this is never evidence of real GPU use."""
    def __init__(self):
        self.fail = False
        self.sync_calls = 0

    def is_available(self):
        return True

    def synchronize(self, device):
        self.sync_calls += 1
        if self.fail:
            raise RuntimeError("synthetic synchronization failure")

    def reset_peak_memory_stats(self, device):
        pass

    def max_memory_allocated(self, device):
        return 100

    def max_memory_reserved(self, device):
        return 200


def test_r01_partial_gpu_time_has_no_full_scalar():
    cuda = SyntheticCUDA()
    collector = ResourceCollector("cuda:0", SimpleNamespace(cuda=cuda))
    with collector.phase("generate", count=16):
        pass
    cuda.fail = True
    with collector.phase("generate", count=16):
        pass
    result = collector.finish()
    assert result["phases"][1]["gpu_synchronized_wall_seconds"] is None
    assert result["gpu_time_seconds"] is None


def test_r02_external_owner_is_counted_once():
    shared = bytearray(16)
    first = np.frombuffer(shared, dtype=np.uint8)
    second = np.frombuffer(shared, dtype=np.uint8)
    assert np.shares_memory(first, second)
    measured = state_storage({"first": first, "second": second})
    assert measured["unique_numpy_buffer_bytes"] == 16


def test_r03_unknown_pre_counter_does_not_recount_history():
    class Counter:
        def __init__(self):
            self.value = 0
            self.fail_once = False

        def resource_counters(self):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("synthetic pre-read failure")
            return {"completion_calls": self.value}

    model = Counter()
    collector = ResourceCollector("cpu")
    with collector.phase("generate", model, count=16):
        model.value = 10
    model.fail_once = True
    with collector.phase("generate", model, count=16):
        model.value = 12
    result = collector.finish()
    assert result["adapter_counters"].get("completion_calls") in (None, 10, 12)
    assert result["adapter_counters"]["completion_calls"] is None
    assert result["counter_measurements"]["completion_calls"]["measured_sum"] == 10


@pytest.mark.parametrize("failure_at,reason", [(2, "GPU_PRE_SYNC_FAILED"), (3, "GPU_POST_SYNC_FAILED")])
def test_gpu_pre_and_post_sync_failures_keep_normal_wall(failure_at, reason):
    class FailingCUDA(SyntheticCUDA):
        def synchronize(self, device):
            self.sync_calls += 1
            if self.sync_calls == failure_at:
                raise RuntimeError("synthetic failure")
    cuda = FailingCUDA()
    collector = ResourceCollector("cuda:0", SimpleNamespace(cuda=cuda))
    with collector.phase("generate", count=1):
        pass
    result = collector.finish()
    measurement = result["measurements"]["gpu_synchronized_model_wall_seconds"]
    assert measurement["eligible_units"] == 1
    assert measurement["measured_units"] == 0
    assert measurement["value"] is None and measurement["complete"] is False
    assert reason in measurement["reason_codes"]
    assert result["generate_seconds"] >= 0
    assert result["phases"][0]["gpu_error_code"] == reason


@pytest.mark.parametrize("fail_first", [False, True])
def test_gpu_partial_measurements_cannot_be_overwritten_by_success(fail_first):
    cuda = SyntheticCUDA()
    collector = ResourceCollector("cuda:0", SimpleNamespace(cuda=cuda))
    for fail in (fail_first, not fail_first):
        cuda.fail = fail
        with collector.phase("generate"):
            pass
    result = collector.finish()
    measurement = result["measurements"]["gpu_synchronized_model_wall_seconds"]
    assert measurement["eligible_units"] == 2
    assert measurement["measured_units"] == 1
    assert measurement["measured_sum"] >= 0
    assert measurement["value"] is None and measurement["complete"] is False
    assert result["gpu_peak_allocated_bytes"] is None
    assert result["measurements"]["gpu_peak_allocated_bytes"]["observed_peak"] == 100


def test_complete_gpu_scope_excludes_evaluator_and_replay():
    cuda = SyntheticCUDA()
    collector = ResourceCollector("cuda:0", SimpleNamespace(cuda=cuda))
    for name in ("model_load", "initialize", "generate", "snapshot", "restore",
                 "input_validation", "output_conversion", "evaluator_io", "replay"):
        with collector.phase(name):
            pass
    result = collector.finish()
    measurement = result["measurements"]["gpu_synchronized_model_wall_seconds"]
    assert measurement["eligible_units"] == measurement["measured_units"] == 5
    assert measurement["complete"] is True
    assert measurement["value"] == sum(row["gpu_synchronized_wall_seconds"] for row in result["phases"][:5])
    assert all(row["gpu_synchronized_wall_seconds"] is None for row in result["phases"][5:])
    assert cuda.sync_calls == 11


@pytest.mark.parametrize("field", ["gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"])
def test_peak_reader_failure_is_per_field(field):
    cuda = SyntheticCUDA()
    def broken(device):
        raise RuntimeError("synthetic peak failure")
    setattr(cuda, "max_memory_allocated" if "allocated" in field else "max_memory_reserved", broken)
    collector = ResourceCollector("cuda:0", SimpleNamespace(cuda=cuda))
    with collector.phase("generate"):
        pass
    result = collector.finish()
    assert result[field] is None
    assert result["measurements"][field]["complete"] is False
    assert result["measurements"]["gpu_synchronized_model_wall_seconds"]["complete"] is True


def test_cpu_gpu_not_applicable_and_unentered_stage_is_separate():
    result = ResourceCollector("cpu").finish()
    for field in ("gpu_synchronized_model_wall_seconds", "gpu_time_seconds", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
        measurement = result["measurements"][field]
        assert measurement["applicability"] == "not_applicable"
        assert measurement["value"] is None
        assert measurement.get("observed_peak", measurement.get("measured_sum")) is None
        assert measurement["eligible_units"] == 0
    assert result["measurements"]["generate_seconds"]["eligible_units"] == 0
    assert result["generate_seconds"] is None
    assert result["execution_complete"] is True


def test_nested_gpu_phases_and_counter_intervals_are_not_double_added():
    model = EpochCounter()
    cuda = SyntheticCUDA()
    collector = ResourceCollector("cuda:0", SimpleNamespace(cuda=cuda))
    with collector.phase("generate", model):
        with collector.phase("generate", model):
            model.value = 10
    result = collector.finish()
    assert len({row["phase_id"] for row in result["phases"]}) == 2
    assert result["phases"][1]["parent_phase_id"] == result["phases"][0]["phase_id"]
    assert result["measurements"]["gpu_time_seconds"]["eligible_units"] == 1
    assert result["gpu_time_seconds"] == result["phases"][0]["gpu_synchronized_wall_seconds"]
    assert result["generate_seconds"] == result["phases"][0]["wall_seconds"]
    assert result["adapter_counters"]["completion_calls"] == 10
    assert cuda.sync_calls == 3


def test_observer_failure_does_not_replace_model_error():
    class BrokenState:
        def resource_state(self):
            raise RuntimeError("synthetic observer failure")
    collector = ResourceCollector()
    with pytest.raises(ValueError, match="model error"):
        with collector.phase("generate", BrokenState()):
            raise ValueError("model error")
    result = collector.finish()
    assert result["phases"][0]["error"] == "ValueError"
    assert result["execution_complete"] is False
    assert result["dynamic_state_bytes"] is None
    assert "STATE_SAMPLE_FAILED" in result["measurements"]["dynamic_state_bytes"]["reason_codes"]


def test_failure_execution_can_have_complete_timing_scope():
    collector = ResourceCollector()
    with pytest.raises(ValueError):
        with collector.phase("generate"):
            raise ValueError("synthetic model failure")
    result = collector.finish()
    assert result["measurements"]["generate_seconds"]["complete"] is True
    assert result["execution_complete"] is False


@pytest.mark.parametrize("state_fn,unique,logical", [
    (lambda a: {"owner": a, "left": a[:10], "right": a[10:]}, 200, 400),
    (lambda a: {"owner": a, "negative": a[::-1], "noncontiguous": a[:, ::2]}, 200, 500),
    (lambda a: {"first": a[:, ::2], "second": a[:, 1::2]}, 200, 200),
    (lambda a: {"deep": a.view().view()[::2].view()}, 200, 100),
    (lambda a: {"empty": a[:0]}, 200, 0),
])
def test_numpy_owner_variants(state_fn, unique, logical):
    result = state_storage(state_fn(np.zeros((20, 10), np.uint8)))
    assert result["unique_supported_storage_bytes"] == unique
    assert result["logical_array_bytes"] == logical
    assert result["complete"] is True


@pytest.mark.parametrize("make_owner", [bytearray, bytes])
def test_external_readonly_and_multilayer_memoryview_owner(make_owner):
    owner = make_owner(16)
    view = memoryview(memoryview(owner)[2:14])
    result = state_storage([np.frombuffer(owner, np.uint8), np.frombuffer(view, np.uint8), np.frombuffer(owner, np.uint8)[::-1]])
    assert result["unique_supported_storage_bytes"] == 16
    assert result["logical_array_bytes"] == 44
    assert result["complete"] is True


def test_distinct_equal_buffers_are_not_deduplicated():
    first, second = bytearray(16), bytearray(16)
    result = state_storage([np.frombuffer(first, np.uint8), np.frombuffer(second, np.uint8)])
    assert result["unique_supported_storage_bytes"] == 32
    assert result["supported_owners"] == 2


def test_nested_cyclic_containers_and_zero_owner():
    owner = np.zeros(0, np.uint8)
    state = {"list": [owner, owner]}
    state["loop"] = state
    result = state_storage(state)
    assert result["complete"] is True
    assert result["unique_supported_storage_bytes"] == result["logical_array_bytes"] == 0
    assert result["supported_owners"] == 1


@pytest.mark.parametrize("unsupported", [object(), np.zeros(3, object)])
def test_mixed_supported_unknown_state_never_has_full_scalar(unsupported):
    result = state_storage({"known": np.zeros(7, np.uint8), "unknown": unsupported})
    assert result["complete"] is False
    assert result["unique_numpy_buffer_bytes"] is None
    assert result["unique_supported_storage_bytes"] == 7
    assert result["reason_codes"]


def test_memmap_and_opaque_buffer_are_conservative(tmp_path):
    import ctypes
    mapped = np.memmap(tmp_path / "map.bin", dtype=np.uint8, mode="w+", shape=(16,))
    opaque = np.ctypeslib.as_array((ctypes.c_uint8 * 16)())
    for value in (mapped, opaque):
        result = state_storage(value)
        assert result["complete"] is False
        assert result["unique_numpy_buffer_bytes"] is None


def test_state_gap_remains_unknown_after_later_success():
    class State:
        def __init__(self):
            self.fail = False
        def resource_state(self):
            if self.fail:
                raise RuntimeError("synthetic state gap")
            return np.zeros(16, np.uint8)
    model = State()
    collector = ResourceCollector()
    collector.sample_state(model, "first")
    model.fail = True
    collector.sample_state(model, "missing")
    model.fail = False
    collector.sample_state(model, "last")
    result = collector.finish()
    measured = result["measurements"]["dynamic_state_bytes"]
    assert result["dynamic_state_bytes"] is None
    assert measured["observed_peak"] == 16
    assert (measured["eligible_units"], measured["measured_units"]) == (3, 2)
    assert measured["complete"] is False


class EpochCounter:
    resource_counter_epoch = "synthetic-lifetime-1"
    def __init__(self):
        self.value = 0
        self.fail_reads = set()
        self.reads = 0
        self.missing_reads = set()
    def resource_counters(self):
        self.reads += 1
        if self.reads in self.fail_reads:
            raise RuntimeError("synthetic counter read failure")
        if self.reads in self.missing_reads:
            return {}
        return {"completion_calls": self.value}


@pytest.mark.parametrize("fail_reads", [{3}, {2}, {2, 3}, {2, 3, 4}])
def test_declared_epoch_counter_gaps_and_attribution(fail_reads):
    model = EpochCounter()
    model.fail_reads = fail_reads
    collector = ResourceCollector()
    with collector.phase("generate", model):
        model.value = 10
    with collector.phase("generate", model):
        model.value = 12
    result = collector.finish()
    measured = result["counter_measurements"]["completion_calls"]
    assert result["adapter_counters"]["completion_calls"] != 22
    if 4 in fail_reads:
        assert measured["complete"] is False and measured["value"] is None
    else:
        assert measured["complete"] is True and measured["value"] == 12
        assert measured["phase_attribution_complete"] is False


def test_declared_epoch_missing_key_can_reconcile_lifetime_without_phase_delta():
    model = EpochCounter()
    model.missing_reads = {3}
    collector = ResourceCollector()
    with collector.phase("generate", model):
        model.value = 10
    with collector.phase("generate", model):
        model.value = 12
    result = collector.finish()
    assert result["adapter_counters"]["completion_calls"] == 12
    assert "COUNTER_KEY_MISSING" in result["counter_measurements"]["completion_calls"]["reason_codes"]
    assert result["counter_intervals"][-1]["phase_delta"] is None


def test_new_counter_key_without_initial_value_is_unknown():
    class NewKey:
        def __init__(self):
            self.added = False
        def resource_counters(self):
            return {"new_key": 9} if self.added else {}
    model = NewKey()
    collector = ResourceCollector()
    with collector.phase("generate", model):
        model.added = True
    result = collector.finish()
    assert result["adapter_counters"]["new_key"] is None
    assert "COUNTER_INITIAL_VALUE_UNKNOWN" in result["counter_measurements"]["new_key"]["reason_codes"]


def test_new_counter_key_with_explicit_initial_value():
    class NewKey:
        def __init__(self):
            self.added = False
        def resource_counters(self):
            return {"values": {"new_key": 9} if self.added else {}, "counter_epoch": "new", "initial_values": {"new_key": 0}}
    model = NewKey()
    collector = ResourceCollector()
    with collector.phase("generate", model):
        model.added = True
    assert collector.finish()["adapter_counters"]["new_key"] == 9


@pytest.mark.parametrize("epoch_change", [False, True])
def test_counter_rollback_and_epoch_transition_are_explicit(epoch_change):
    model = EpochCounter()
    collector = ResourceCollector()
    with collector.phase("generate", model):
        model.value = 10
    with collector.phase("initialize", model):
        if epoch_change:
            model.resource_counter_epoch = "synthetic-lifetime-2"
        model.value = 0
    result = collector.finish()
    measurement = result["counter_measurements"]["completion_calls"]
    assert measurement["value"] is None
    assert ("COUNTER_EPOCH_CHANGED" if epoch_change else "COUNTER_ROLLBACK") in measurement["reason_codes"]


def test_replaced_models_keep_separate_counter_baselines():
    collector = ResourceCollector()
    first, second = EpochCounter(), EpochCounter()
    second.value = 100
    with collector.phase("generate", first):
        first.value = 10
    with collector.phase("generate", second):
        second.value = 102
    result = collector.finish()
    assert result["adapter_counters"]["completion_calls"] == 12
    assert result["counter_measurements"]["completion_calls"]["eligible_units"] == 2


@pytest.mark.parametrize("bad_value", [-1, 1.0, True, "1"])
def test_counter_type_error_is_unknown_not_zero(bad_value):
    model = EpochCounter()
    collector = ResourceCollector()
    with collector.phase("generate", model):
        model.value = bad_value
    result = collector.finish()
    assert result["adapter_counters"]["completion_calls"] is None
    assert result["counter_measurements"]["completion_calls"]["complete"] is False


def test_unknown_first_counter_read_does_not_count_preexisting_work():
    model = EpochCounter()
    model.value = 100
    model.fail_reads = {1}
    collector = ResourceCollector()
    with collector.phase("generate", model):
        model.value = 102
    result = collector.finish()
    assert result["adapter_counters"]["completion_calls"] is None
    assert result["counter_measurements"]["completion_calls"]["measured_sum"] == 0


def test_reset_restore_and_failed_eager_work_keep_lifetime_cost():
    model = EpochCounter()
    collector = ResourceCollector()
    for phase in ("initialize", "generate", "initialize", "snapshot", "restore", "generate", "restore", "generate"):
        with collector.phase(phase, model):
            if phase in {"initialize", "generate"}:
                model.value += 1
    with pytest.raises(RuntimeError):
        with collector.phase("initialize", model):
            model.value += 1
            raise RuntimeError("synthetic eager fill failure")
    assert collector.finish()["adapter_counters"]["completion_calls"] == 6


def test_finish_is_idempotent_and_caller_cannot_mutate_cached_receipt(tmp_path):
    model = EpochCounter()
    collector = ResourceCollector()
    with collector.phase("generate", model):
        model.value = 10
    (tmp_path / "payload.bin").write_bytes(b"1234")
    first = collector.finish(tmp_path)
    (tmp_path / "resources.json").write_bytes(b"new receipt")
    second = collector.finish(tmp_path)
    assert second == first
    second["adapter_counters"]["completion_calls"] = 22
    assert collector.finish()["adapter_counters"]["completion_calls"] == 10
    with pytest.raises(RuntimeError, match="finished"):
        with collector.phase("generate", model):
            pass


def test_guarded_missing_response_is_request_not_model_call():
    collector = ResourceCollector()
    with collector.phase("generate", count=16, model_call_attempted=False):
        pass
    result = collector.finish()
    assert result["accounting_calls"]["requests_issued"] == 16
    assert result["accounting_calls"]["model_calls_attempted"] == 0
    assert result["generate_seconds"] is None


def test_native_extra_frames_require_explicit_same_scope_counter():
    class Native:
        def __init__(self):
            self.frames = 0
        def resource_counters(self):
            return {"native_frames": self.frames}
    model = Native()
    collector = ResourceCollector()
    with collector.phase("generate", model, count=16):
        model.frames = 10
    result = collector.finish()
    assert result["native_frames"] == 10
    assert result["native_extra_frames"] is None


def test_counter_read_statuses_and_attribute_observer_failure_are_typed():
    class NotInitialized:
        def resource_counters(self):
            return {"status": "not_initialized", "values": {}}
    class AttributeFailure:
        @property
        def resource_counters(self):
            raise RuntimeError("synthetic observer property failure")
        @property
        def resource_state(self):
            raise RuntimeError("synthetic observer property failure")
    collector = ResourceCollector()
    assert collector._counters(None).status == "not_implemented"
    assert collector._counters(NotInitialized()).status == "not_initialized"
    assert collector._counters(AttributeFailure()).status == "read_failed"
    with pytest.raises(ValueError, match="original model"):
        with collector.phase("generate", AttributeFailure()):
            raise ValueError("original model")
    assert collector.finish()["phases"][0]["error"] == "ValueError"


def test_disappeared_counter_contract_does_not_retain_complete_old_value():
    class Contract(EpochCounter):
        def __init__(self):
            super().__init__()
            self.enabled = True
        def resource_counters(self):
            return super().resource_counters() if self.enabled else {"status": "not_implemented", "values": {}}
    model = Contract()
    collector = ResourceCollector()
    with collector.phase("generate", model):
        model.value = 10
    model.enabled = False
    with collector.phase("generate", model):
        model.value = 12
    result = collector.finish()
    assert result["adapter_counters"]["completion_calls"] is None
    assert result["counter_measurements"]["completion_calls"]["measured_sum"] == 10


def test_nonexistent_disk_root_is_unknown_and_symlinks_excluded(tmp_path):
    unknown = ResourceCollector().finish(tmp_path / "missing")
    assert unknown["disk_bytes"] is None
    assert "DISK_INVENTORY_FAILED" in unknown["measurements"]["disk_bytes"]["reason_codes"]
    (tmp_path / "payload").write_bytes(b"abcd")
    (tmp_path / "alias").symlink_to(tmp_path / "payload")
    measured = ResourceCollector().finish(tmp_path)
    assert measured["disk_bytes"] == 4


@pytest.mark.parametrize("family", ["F2", "F3", "F7"])
def test_collector_does_not_change_pixels_or_rgb_sufficient_statistics(family):
    from rswm_inria.baselines import WindowMap
    from rswm_inria.runner import run_task
    from rswm_inria.tasks import packages
    task = next(task for task, _ in packages("resource-observer-synthetic", "diagnostic_only", 0) if task["family"] == family)
    world = np.zeros((2048, 2048, 3), np.uint8)
    collector = ResourceCollector()
    observed, observed_journal = run_task(task, world, WindowMap(), seed=19, resources=collector)
    plain, plain_journal = run_task(task, world, WindowMap(), seed=19)
    assert observed["reports"] == plain["reports"]
    outputs = lambda journal: [(row["stream"], row["event"], row["status"], row["pixel_sha256"])
                               for row in journal.rows if row["kind"] == "output"]
    assert outputs(observed_journal) == outputs(plain_journal)
    assert collector.finish()["adapter_counters"]["completion_calls"] > 0


def test_counter_all_reads_failed_preserves_unknown_contract_coverage():
    class FailedCounter:
        def resource_counters(self):
            raise RuntimeError("synthetic always-failed read")
        def resource_state(self):
            return {}
    collector = ResourceCollector()
    with collector.phase("generate", FailedCounter(), count=32):
        pass
    result = collector.finish()
    assert result["measurement_complete"] is False
    coverage = result["counter_read_coverage"]
    assert coverage["applicability"] == "applicable"
    assert (coverage["eligible_units"], coverage["measured_units"]) == (2, 0)
    assert coverage["value"] is None and coverage["complete"] is False
    assert len(result["counter_reads"]) == 2
    assert result["counter_measurements"]["counter_contract_read_observations"] == coverage
    assert result["adapter_counters"].get("native_frames") is None


@pytest.mark.parametrize("status", ["not_initialized", "read_failed"])
def test_counter_unknown_status_keeps_each_read_in_failed_denominator(status):
    class Counter:
        def resource_counters(self):
            return {"status": status, "values": {}}
        def resource_state(self):
            return {}
    model = Counter()
    collector = ResourceCollector()
    for _ in range(3):
        with collector.phase("generate", model, count=16):
            pass
    result = collector.finish()
    coverage = result["counter_read_coverage"]
    assert coverage["eligible_units"] == 6
    assert coverage["measured_units"] == 0
    assert coverage["status_counts"] == {status: 6}
    assert coverage["complete"] is False and coverage["value"] is None
    assert result["measurement_complete"] is False


@pytest.mark.parametrize("structured", [False, True])
def test_unimplemented_counter_contract_is_not_applicable_and_not_a_failure(structured):
    from rswm_inria.baselines import CopyModel
    class NoCounters(CopyModel):
        def resource_counters(self):
            return {"status": "not_implemented", "values": {}}
    model = NoCounters() if structured else CopyModel()
    collector = ResourceCollector()
    with collector.phase("generate", model, count=32):
        pass
    result = collector.finish()
    assert result["measurement_complete"] is True
    coverage = result["counter_read_coverage"]
    assert coverage["applicability"] == "not_applicable"
    assert coverage["eligible_units"] == coverage["measured_units"] == 0
    assert coverage["complete"] is True
    assert coverage["value"] is coverage["measured_sum"] is None


def test_recovered_epoch_total_and_counter_read_coverage_have_distinct_completeness():
    model = EpochCounter()
    model.fail_reads = {3}
    collector = ResourceCollector()
    with collector.phase("generate", model):
        model.value = 10
    with collector.phase("generate", model):
        model.value = 12
    result = collector.finish()
    assert result["counter_measurements"]["completion_calls"]["value"] == 12
    assert result["counter_measurements"]["completion_calls"]["complete"] is True
    assert result["counter_measurements"]["completion_calls"]["phase_attribution_complete"] is False
    coverage = result["counter_read_coverage"]
    assert (coverage["eligible_units"], coverage["measured_units"]) == (4, 3)
    assert coverage["complete"] is False
    assert result["measurement_complete"] is False


def test_unentered_stages_are_inapplicable_with_explicit_stage_reason():
    result = ResourceCollector().finish()
    for field in ("initialize_seconds", "generate_seconds", "snapshot_seconds", "restore_seconds"):
        measurement = result["measurements"][field]
        assert measurement["applicability"] == "not_applicable"
        assert measurement["value"] is measurement["measured_sum"] is None
        assert measurement["eligible_units"] == measurement["measured_units"] == 0
        assert "STAGE_NOT_ENTERED" in measurement["reason_codes"]
    for field, measurement in result["counter_measurements"].items():
        assert field in result["adapter_counters"]
        assert result["adapter_counters"][field] == measurement["value"]


def test_guard_requests_do_not_create_counter_read_failures():
    model = EpochCounter()
    collector = ResourceCollector()
    with collector.phase("initialize", model):
        model.value = 1
    with collector.phase("generate", model, count=32, model_call_attempted=False):
        pass
    result = collector.finish()
    assert result["adapter_counters"]["completion_calls"] == 1
    coverage = result["counter_read_coverage"]
    assert coverage["eligible_units"] == coverage["measured_units"] == 2
    assert coverage["complete"] is True


@pytest.mark.parametrize("name", ["requested_outputs", "counter_contract_read_observations"])
def test_adapter_cannot_replace_evaluator_owned_counter_scope(name):
    class Collision:
        def resource_counters(self):
            return {name: 1}
        def resource_state(self):
            return {}
    collector = ResourceCollector()
    with collector.phase("generate", Collision(), count=32):
        pass
    result = collector.finish()
    assert result["measurement_complete"] is False
    assert result["adapter_counters"]["requested_outputs"] == 32
    assert result["counter_read_coverage"]["complete"] is False
