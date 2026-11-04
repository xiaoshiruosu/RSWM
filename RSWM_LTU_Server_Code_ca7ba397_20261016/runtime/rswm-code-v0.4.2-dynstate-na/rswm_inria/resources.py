"""Cooperative resource observations with explicit completeness and storage scope.

Observation failures never stand in for zero work and never change model exceptions.
GPU durations are synchronized host wall time, not CUDA kernel durations.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from pathlib import Path
import copy
import math
import sys
import time
import numpy as np


@dataclass
class ResourceRecord:
    schema: str = "rswm-resource-v3"
    attempt_wall_seconds: float | None = None
    model_load_seconds: float | None = None
    initialize_seconds: float | None = None
    generate_seconds: float | None = None
    output_conversion_seconds: float | None = None
    output_validation_seconds: float | None = None
    evaluator_io_seconds: float | None = None
    input_validation_seconds: float | None = None
    snapshot_seconds: float | None = None
    restore_seconds: float | None = None
    replay_seconds: float | None = None
    gpu_time_seconds: float | None = None
    gpu_synchronized_model_wall_seconds: float | None = None
    gpu_peak_allocated_bytes: int | None = None
    gpu_peak_reserved_bytes: int | None = None
    process_peak_rss_bytes: int | None = None
    dynamic_state_bytes: int | None = None
    disk_bytes: int | None = None

    def to_dict(self):
        return asdict(self)


class ResourceTimer:
    def __enter__(self):
        self.start = time.perf_counter()
        self.elapsed = None
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.start


def process_rss_bytes():
    """Process-lifetime high-water RSS. Native Windows is not measured here."""
    try:
        if sys.platform == "win32":
            return None
        import resource
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, AttributeError, OSError):
        return None


def state_storage(state):
    """Deduplicate supported root owners, never infer physical memory from slices.

    Logical bytes count each explicit ndarray reference. Root references are retained
    throughout this sample, so identities cannot be reused. Memmaps, object dtype
    contents, Tensor storage and opaque buffer owners are deliberately unknown.
    """
    buffers, visited = {}, set()
    logical = 0
    unsupported, reasons = set(), set()

    def unsupported_owner(value, code):
        unsupported.add(type(value).__name__)
        reasons.add(code)

    def owner(value):
        seen = set()
        while True:
            if id(value) in seen:
                unsupported_owner(value, "STORAGE_OWNER_CYCLE")
                return
            seen.add(id(value))
            if isinstance(value, np.memmap):
                unsupported_owner(value, "MEMMAP_STORAGE_UNSUPPORTED")
                return
            if isinstance(value, np.ndarray):
                if value.dtype.hasobject:
                    unsupported_owner(value, "OBJECT_DTYPE_STORAGE_UNSUPPORTED")
                    return
                if value.base is not None:
                    value = value.base
                    continue
                if not value.flags.owndata:
                    unsupported_owner(value, "STORAGE_OWNER_UNCONFIRMED")
                    return
                buffers[id(value)] = (value, int(value.nbytes))
                return
            if isinstance(value, memoryview):
                value = value.obj
                continue
            if isinstance(value, (bytes, bytearray)):
                buffers[id(value)] = (value, len(value))
                return
            unsupported_owner(value, "STORAGE_OWNER_UNSUPPORTED")
            return

    def walk(value):
        nonlocal logical
        if isinstance(value, np.ndarray):
            logical += int(value.nbytes)
            owner(value)
        elif isinstance(value, (bytes, bytearray, memoryview)):
            owner(value)
        elif isinstance(value, (dict, list, tuple)):
            if id(value) in visited:
                return
            visited.add(id(value))
            for item in (value.values() if isinstance(value, dict) else value):
                walk(item)
        elif value is not None and not isinstance(value, (str, int, float, bool, np.generic)):
            unsupported_owner(value, "EXPLICIT_STATE_TYPE_UNSUPPORTED")

    walk(state)
    partial = sum(size for _, size in buffers.values())
    complete = not reasons
    return {"unique_numpy_buffer_bytes": partial if complete else None,
            "logical_numpy_bytes": logical,
            "logical_array_bytes": logical,
            "unique_supported_storage_bytes": partial,
            "complete": complete, "supported_owners": len(buffers),
            "unsupported_types": sorted(unsupported), "reason_codes": sorted(reasons),
            "state_scope": "adapter-declared ndarray/buffer state; excludes weights, Python overhead and invisible backend caches",
            "scope": "supported owning NumPy/bytes/bytearray buffers; physical allocator memory is not measured"}


@dataclass(frozen=True)
class CounterRead:
    """An absent or failed cumulative read is not the integer zero."""
    status: str
    values: dict = field(default_factory=dict)
    counter_epoch: str | int | None = None
    initial_values: dict = field(default_factory=dict)
    error_code: str | None = None


MODEL_PHASES = frozenset({"model_load", "initialize", "generate", "snapshot", "restore"})
PEAK_FIELDS = frozenset({"gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes",
                         "process_peak_rss_bytes", "dynamic_state_bytes"})
GPU_FIELDS = frozenset({"gpu_time_seconds", "gpu_synchronized_model_wall_seconds",
                       "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"})


def _measurement(name, applicable=True, scope=None):
    peak = name in PEAK_FIELDS
    return {"value": None, "observed_peak" if peak else "measured_sum": None,
            "applicability": "applicable" if applicable else "not_applicable",
            "eligible_units": 0, "measured_units": 0, "complete": True,
            "aggregation": "max" if peak else "sum",
            "unit": "seconds" if name.endswith("_seconds") else "bytes",
            "reason_codes": [], "scope": scope or name}


def _reason(measurement, code):
    if code not in measurement["reason_codes"]:
        measurement["reason_codes"].append(code)


def _union_seconds(intervals):
    end, total = None, 0.
    for start, stop in sorted(intervals):
        total += max(0., stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


class ResourceCollector:
    """Outside adapter snapshots; actual work cannot roll back with F7 state."""
    def __init__(self, device="cpu", torch_module=None):
        self.start_wall = time.perf_counter()
        self.device = device
        self.record = ResourceRecord()
        self.execution_complete = True
        self.phases, self.state_samples, self.counter_intervals = [], [], []
        self.counter_reads, self._counter_contract_seen = [], set()
        self.counters, self.unavailable = {}, {}
        self._stack, self._wall_intervals = [], {}
        self._counter_states, self._model_refs = {}, {}
        self._finished = None
        self.cuda = None
        self.measurements = {name: _measurement(name, not (name in GPU_FIELDS and not str(device).startswith("cuda")))
                             for name in self.record.to_dict() if name != "schema"}
        self.calls = {"scheduled_calls": 0, "entered_calls": 0, "model_calls": 0,
                      "measurement_eligible_calls": 0, "planned_output_slots": None,
                      "requests_issued": 0, "model_calls_attempted": 0,
                      "valid_outputs": None}
        self._gpu_initialized = False
        if str(device).startswith("cuda"):
            try:
                if torch_module is None:
                    import torch as torch_module
                self.cuda = torch_module.cuda
                if not self.cuda.is_available():
                    raise RuntimeError("CUDA unavailable")
                self.cuda.synchronize(device)
                self.cuda.reset_peak_memory_stats(device)
                self._gpu_initialized = True
            except Exception as exc:
                self.unavailable["gpu"] = type(exc).__name__
                self.cuda = None
                for name in GPU_FIELDS:
                    _reason(self.measurements[name], "GPU_INITIALIZATION_FAILED")
        else:
            self.unavailable["gpu"] = "CPU adapter; GPU metrics not applicable"
        for name in GPU_FIELDS:
            self.measurements[name]["scope"] = (
                "mutually nonoverlapping entered model stages; synchronized host wall, not kernel time"
                if name.endswith("seconds") else
                "device allocator high water from collector initialization; includes shared weights")
            self.measurements[name]["device"] = str(device)
        self.measurements["process_peak_rss_bytes"]["scope"] = "process lifetime high water, including dependencies and previous attempts"
        self.measurements["dynamic_state_bytes"]["scope"] = "observed peak of adapter-declared supported state owners; excludes weights, Python overhead and invisible backend caches"
        self.measurements["disk_bytes"]["scope"] = "logical file lengths within attempt directory at finish; excludes later resource/receipt writes"

    def _counters(self, model):
        if model is None:
            return CounterRead("not_implemented", error_code="COUNTERS_NOT_IMPLEMENTED")
        try:
            if not hasattr(model, "resource_counters"):
                return CounterRead("not_implemented", error_code="COUNTERS_NOT_IMPLEMENTED")
            raw = model.resource_counters()
            if not isinstance(raw, dict):
                raise TypeError("Counter mapping required")
            if "values" in raw:
                status = raw.get("status", "ok")
                if status in {"not_initialized", "not_implemented", "read_failed"}:
                    return CounterRead(status, error_code="COUNTERS_" + status.upper())
                if status != "ok":
                    raise ValueError("Unknown counter read status")
                values = raw["values"]
                epoch = raw.get("counter_epoch")
                initial = raw.get("initial_values", {})
            else:
                values, initial = raw, {}
                epoch = getattr(model, "resource_counter_epoch", None)
                if callable(epoch):
                    epoch = epoch()
            if epoch is not None and (type(epoch) not in (str, int) or epoch == ""):
                raise ValueError("Stable counter epoch must be a string or integer")
            for mapping in (values, initial):
                if not isinstance(mapping, dict) or any(not isinstance(k, str) or not k or type(v) is not int or v < 0 for k, v in mapping.items()):
                    raise ValueError("Nonnegative integer counters required")
                if set(mapping) & {"requested_outputs", "counter_contract_read_observations"}:
                    raise ValueError("Evaluator observer counter names are reserved")
            return CounterRead("ok", dict(values), epoch, dict(initial))
        except Exception as exc:
            self.unavailable["adapter_counters"] = type(exc).__name__
            return CounterRead("read_failed", error_code="COUNTER_READ_" + type(exc).__name__.upper())

    def _observe_counters(self, model, read, boundary, phase_id):
        if model is None:
            return
        identity = id(model)
        self._model_refs[identity] = model
        if read.status != "not_implemented":
            self._counter_contract_seen.add(identity)
        # Coverage belongs to the declared counter contract even when every read
        # fails and no counter key has ever become known. A missing cooperative
        # contract, however, is not an applicable measurement for Copy/Oracle.
        eligible = identity in self._counter_contract_seen
        self.counter_reads.append({"phase_id": phase_id, "boundary": boundary,
            "model_identity": str(identity), "status": read.status,
            "counter_epoch": read.counter_epoch, "error_code": read.error_code,
            "measurement_eligible": eligible, "known_keys": sorted(read.values)})
        previous = {key for owner, key in self._counter_states if owner == identity}
        missing = previous - set(read.values) if read.status == "ok" else previous
        for key in missing:
            state = self._counter_states[(identity, key)]
            state["gap"] = True
            state["reasons"].add("COUNTER_KEY_MISSING" if read.status == "ok" else read.error_code)
        for key, value in read.values.items():
            token = (identity, key)
            if token not in self._counter_states:
                baseline = value if boundary == "before" else read.initial_values.get(key)
                self._counter_states[token] = {"last": value, "epoch": read.counter_epoch,
                    "sum": value - baseline if baseline is not None and value >= baseline else 0,
                    "invalid": baseline is None or (baseline is not None and value < baseline),
                    "gap": False, "reasons": set() if baseline is not None else {"COUNTER_INITIAL_VALUE_UNKNOWN"},
                    "first_phase_id": phase_id, "last_phase_id": phase_id}
                continue
            state = self._counter_states[token]
            state["last_phase_id"] = phase_id
            if read.counter_epoch != state["epoch"]:
                # New lifecycle must be declared. Its first read is a baseline;
                # no unknown old terminal value or new absolute count is invented.
                state["invalid"] = True
                state["reasons"].add("COUNTER_EPOCH_CHANGED")
                state["last"], state["epoch"], state["gap"] = value, read.counter_epoch, False
            elif value < state["last"]:
                state["invalid"] = True
                state["reasons"].add("COUNTER_ROLLBACK")
                state["last"], state["gap"] = value, False
            elif state["gap"] and read.counter_epoch is None:
                state["invalid"] = True
                state["reasons"].add("COUNTER_GAP_WITHOUT_STABLE_EPOCH")
                state["last"], state["gap"] = value, False
            else:
                state["sum"] += value - state["last"]
                state["last"], state["gap"] = value, False

    def _register(self, name):
        measurement = self.measurements[name]
        measurement["eligible_units"] += 1
        return measurement

    def _sample(self, name, value=None, reason=None):
        measurement = self.measurements[name]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
            value, reason = None, "MEASUREMENT_VALUE_INVALID"
        if value is None:
            measurement["complete"] = False
            _reason(measurement, reason or "MEASUREMENT_UNAVAILABLE")
            return
        measurement["measured_units"] += 1
        partial = "observed_peak" if measurement["aggregation"] == "max" else "measured_sum"
        previous = measurement[partial]
        measurement[partial] = (max(previous, value) if previous is not None else value) if partial == "observed_peak" else (previous or 0) + value

    @contextmanager
    def phase(self, name, model=None, **metadata):
        if self._finished is not None:
            raise RuntimeError("ResourceCollector is finished")
        phase_id = f"phase-{len(self.phases):06d}"
        # Reserve on entry, so nested phases also get a unique stable identity.
        row = {**metadata, "phase": name, "phase_id": phase_id,
               "parent_phase_id": self._stack[-1]["phase_id"] if self._stack else None,
               "stream": metadata.get("stream"), "call": metadata.get("call"),
               "event_range": metadata.get("event_range")}
        self.phases.append(row)
        self.calls["scheduled_calls"] += 1
        self.calls["entered_calls"] += 1
        attempted = metadata.get("model_call_attempted", True) is not False
        model_stage = name in MODEL_PHASES and attempted
        if model_stage:
            self.calls["model_calls"] += 1
            self.calls["model_calls_attempted"] += 1
        if name == "generate":
            self.calls["requests_issued"] += metadata.get("count", 0)
        field_name = name + "_seconds" if (attempted or name not in MODEL_PHASES) else None
        if field_name in self.measurements:
            self._register(field_name)
        gpu_eligible = (model_stage and str(self.device).startswith("cuda") and
                        not any(parent.get("gpu_atomic") for parent in self._stack))
        row["gpu_atomic"] = gpu_eligible
        row["gpu_measurement_role"] = "atomic" if gpu_eligible else "covered_by_parent" if model_stage and self._stack else "not_eligible"
        if gpu_eligible:
            self.calls["measurement_eligible_calls"] += 1
            self._register("gpu_synchronized_model_wall_seconds")
        before = self._counters(model) if attempted else CounterRead("not_implemented")
        if attempted:
            self._observe_counters(model, before, "before", phase_id)
        gpu_ok, gpu_error = self.cuda is not None, None
        if gpu_eligible:
            if self.cuda is None:
                gpu_ok, gpu_error = False, "GPU_INITIALIZATION_FAILED"
            else:
                try:
                    self.cuda.synchronize(self.device)
                except Exception as exc:
                    gpu_ok, gpu_error = False, "GPU_PRE_SYNC_FAILED"
                    self.unavailable["gpu"] = type(exc).__name__
        start = time.perf_counter()
        self._stack.append(row)
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            self.execution_complete = False
            raise
        finally:
            # Observer errors are recorded; they cannot replace the original error.
            try:
                if gpu_eligible and gpu_ok:
                    try:
                        self.cuda.synchronize(self.device)
                    except Exception as exc:
                        gpu_ok, gpu_error = False, "GPU_POST_SYNC_FAILED"
                        self.unavailable["gpu"] = type(exc).__name__
                stop = time.perf_counter()
                elapsed = stop - start
                row.update(wall_seconds=elapsed, wall_start=start-self.start_wall,
                           wall_end=stop-self.start_wall, error=error,
                           gpu_synchronized_wall_seconds=elapsed if gpu_eligible and gpu_ok else None,
                           gpu_error_code=gpu_error)
                if field_name in self.measurements:
                    self._sample(field_name, elapsed)
                    self._wall_intervals.setdefault(field_name, []).append((start, stop))
                    self.measurements[field_name]["measured_sum"] = _union_seconds(self._wall_intervals[field_name])
                if gpu_eligible:
                    self._sample("gpu_synchronized_model_wall_seconds", elapsed if gpu_ok else None, gpu_error)
                after = self._counters(model) if attempted else CounterRead("not_implemented")
                if attempted:
                    self._observe_counters(model, after, "after", phase_id)
                keys = set(before.values) | set(after.values)
                for key in sorted(keys):
                    delta = None
                    if before.status == after.status == "ok" and key in before.values and key in after.values and before.counter_epoch == after.counter_epoch and after.values[key] >= before.values[key]:
                        delta = after.values[key] - before.values[key]
                    self.counter_intervals.append({"phase_id": phase_id, "counter": key,
                        "model_identity": str(id(model)), "counter_epoch": after.counter_epoch,
                        "phase_delta": delta, "phase_attribution_complete": delta is not None,
                        "before_status": before.status, "after_status": after.status})
                if attempted:
                    self.sample_state(model, name)
            except Exception as exc:
                self.unavailable["observer"] = type(exc).__name__
                if field_name in self.measurements:
                    self._sample(field_name, reason="OBSERVER_FAILED")
                if gpu_eligible:
                    self._sample("gpu_synchronized_model_wall_seconds", reason="OBSERVER_FAILED")
            finally:
                self._stack.pop()

    def sample_state(self, model, phase):
        if model is None:
            return
        self._register("dynamic_state_bytes")
        try:
            if not hasattr(model, "resource_state"):
                self._sample("dynamic_state_bytes", reason="STATE_CONTRACT_NOT_IMPLEMENTED")
                self.unavailable["dynamic_state_bytes"] = "Adapter has no explicit resource_state contract"
                return
            measured = state_storage(model.resource_state())
            self.state_samples.append({"phase": phase, **measured})
            if measured["complete"]:
                self._sample("dynamic_state_bytes", measured["unique_supported_storage_bytes"])
            else:
                unsupported = set(
                    measured.get("unsupported_types", [])
                )

                torch_internal_state = (
                    bool(unsupported)
                    and unsupported <= {
                        "Tensor",
                        "device",
                        "dtype",
                    }
                )

                if torch_internal_state:
                    measurement = self.measurements[
                        "dynamic_state_bytes"
                    ]
                    measurement["applicability"] = (
                        "not_applicable"
                    )
                    measurement["complete"] = True
                    measurement["eligible_units"] = 0
                    measurement["measured_units"] = 0
                    measurement["observed_peak"] = None
                    _reason(
                        measurement,
                        "STATE_CONTRACT_NOT_BYTE_ACCOUNTABLE",
                    )
                    self.unavailable[
                        "dynamic_state_bytes"
                    ] = (
                        "Backend state has no supported "
                        "byte-level accounting contract"
                    )
                    return

                self._sample(
                    "dynamic_state_bytes",
                    reason="STATE_STORAGE_INCOMPLETE",
                )
                # A supported partial peak is evidence, never a full-state peak.
                current = self.measurements[
                    "dynamic_state_bytes"
                ]["observed_peak"]
                self.measurements[
                    "dynamic_state_bytes"
                ]["observed_peak"] = max(
                    current or 0,
                    measured[
                        "unique_supported_storage_bytes"
                    ],
                )
                for reason in measured["reason_codes"]:
                    _reason(
                        self.measurements[
                            "dynamic_state_bytes"
                        ],
                        reason,
                    )
                self.unavailable[
                    "dynamic_state_bytes"
                ] = "Unsupported explicit state storage"
        except Exception as exc:
            self._sample("dynamic_state_bytes", reason="STATE_SAMPLE_FAILED")
            self.unavailable["dynamic_state_bytes"] = type(exc).__name__

    def finish(self, root=None):
        if self._finished is not None:
            return copy.deepcopy(self._finished)
        if self._stack:
            raise RuntimeError("Cannot finish within an entered phase")
        self._register("attempt_wall_seconds")
        self._register("process_peak_rss_bytes")
        try:
            self._sample("process_peak_rss_bytes", process_rss_bytes(), "RSS_UNAVAILABLE")
        except Exception as exc:
            self._sample("process_peak_rss_bytes", reason="RSS_SAMPLE_FAILED")
            self.unavailable["process_peak_rss_bytes"] = type(exc).__name__
        for name, getter in (("gpu_peak_allocated_bytes", "max_memory_allocated"),
                             ("gpu_peak_reserved_bytes", "max_memory_reserved")):
            if str(self.device).startswith("cuda"):
                self._register(name)
                gpu_sync_complete = self.measurements["gpu_synchronized_model_wall_seconds"]["complete"]
                try:
                    value = int(getattr(self.cuda, getter)(self.device)) if self.cuda is not None else None
                    if value is not None and value < 0:
                        raise ValueError("Negative GPU allocator peak")
                    self._sample(name, value, "GPU_PEAK_UNAVAILABLE")
                    if not gpu_sync_complete or not self._gpu_initialized:
                        self.measurements[name]["complete"] = False
                        _reason(self.measurements[name], "GPU_RANGE_SYNCHRONIZATION_INCOMPLETE")
                except Exception as exc:
                    self._sample(name, reason="GPU_PEAK_READ_FAILED")
                    self.unavailable[name] = type(exc).__name__
        if root is not None:
            self._register("disk_bytes")
            try:
                size = None
                last_error = None

                for retry_index in range(3):
                    try:
                        if not Path(root).is_dir():
                            raise FileNotFoundError(
                                "Attempt resource directory is absent"
                            )

                        size = sum(
                            p.stat().st_size
                            for p in Path(root).rglob("*")
                            if p.is_file() and not p.is_symlink()
                        )
                        break

                    except FileNotFoundError as exc:
                        last_error = exc
                        if retry_index < 2:
                            time.sleep(0.2 * (retry_index + 1))

                if size is None:
                    raise last_error

                self._sample("disk_bytes", size)

            except Exception as exc:
                self._sample(
                    "disk_bytes",
                    reason="DISK_INVENTORY_FAILED",
                )
                self.unavailable["disk_bytes"] = type(exc).__name__
        counter_measurements = {}
        for key in sorted({key for _, key in self._counter_states}):
            states = [state for (_, field), state in self._counter_states.items() if field == key]
            good = sum(not state["invalid"] and not state["gap"] for state in states)
            complete = good == len(states)
            partial = sum(state["sum"] for state in states)
            counter_measurements[key] = {"value": partial if complete else None,
                "measured_sum": partial, "applicability": "applicable",
                "eligible_units": len(states), "measured_units": good, "complete": complete,
                "aggregation": "sum", "unit": "count",
                "reason_codes": sorted({reason for state in states for reason in state["reasons"] if reason}),
                "scope": "nonoverlapping cumulative observations from first trusted pre-stage baseline; inaccessible earlier load work is excluded; unknown boundaries are not zero",
                "phase_attribution_complete": all(interval["phase_attribution_complete"] for interval in self.counter_intervals if interval["counter"] == key)}
        self.counters = {key: measurement["value"] for key, measurement in counter_measurements.items()}
        self.counters["requested_outputs"] = self.calls["requests_issued"]
        counter_measurements["requested_outputs"] = {"value": self.calls["requests_issued"],
            "measured_sum": self.calls["requests_issued"], "applicability": "applicable",
            "eligible_units": sum(row["phase"] == "generate" for row in self.phases),
            "measured_units": sum(row["phase"] == "generate" for row in self.phases), "complete": True,
            "aggregation": "sum", "unit": "count", "reason_codes": [],
            "scope": "evaluator request slots, including guarded missing responses; not native model work"}
        reads = [read for read in self.counter_reads if read["measurement_eligible"]]
        measured_reads = sum(read["status"] == "ok" for read in reads)
        counter_read_coverage = {"value": measured_reads if reads and measured_reads == len(reads) else None,
            "measured_sum": measured_reads if reads else None,
            "applicability": "applicable" if reads else "not_applicable",
            "eligible_units": len(reads), "measured_units": measured_reads,
            "complete": measured_reads == len(reads), "aggregation": "sum", "unit": "count",
            "reason_codes": sorted({read["error_code"] or "COUNTER_READ_UNKNOWN" for read in reads if read["status"] != "ok"}) if reads else ["COUNTER_CONTRACT_NOT_APPLICABLE"],
            "scope": "coverage of applicable pre/post counter-contract reads, including unknown counter names; observation count is not native model work",
            "status_counts": {status: sum(read["status"] == status for read in self.counter_reads)
                              for status in sorted({read["status"] for read in self.counter_reads})}}
        # A distinct observer field carries failed-read denominators through the
        # existing per-field resource aggregation, without inventing native keys.
        counter_measurements["counter_contract_read_observations"] = counter_read_coverage
        if not counter_measurements["requested_outputs"]["eligible_units"]:
            counter_measurements["requested_outputs"].update(applicability="not_applicable", value=None, measured_sum=None,
                                                               reason_codes=["STAGE_NOT_ENTERED"])
        self.counters.update({key: measurement["value"] for key, measurement in counter_measurements.items()})
        self.measurements["gpu_time_seconds"] = copy.deepcopy(self.measurements["gpu_synchronized_model_wall_seconds"])
        self.measurements["gpu_time_seconds"]["compatibility_alias_for"] = "gpu_synchronized_model_wall_seconds"
        self._sample("attempt_wall_seconds", time.perf_counter() - self.start_wall)
        for name in GPU_FIELDS:
            self.measurements[name]["measurement_start_wall_seconds"] = 0.
            self.measurements[name]["measurement_end_wall_seconds"] = time.perf_counter() - self.start_wall
            self.measurements[name]["initial_synchronization_complete"] = self._gpu_initialized
            self.measurements[name]["allocator_definition"] = "torch.cuda device caching allocator" if "peak" in name else None
        for key, measurement in self.measurements.items():
            partial = "observed_peak" if measurement["aggregation"] == "max" else "measured_sum"
            if measurement["applicability"] == "not_applicable":
                measurement["value"] = measurement[partial] = None
                _reason(measurement, "NOT_APPLICABLE")
            elif not measurement["eligible_units"]:
                _reason(measurement, "STAGE_NOT_ENTERED")
                measurement["applicability"] = "not_applicable"
                measurement["value"] = measurement[partial] = None
            elif measurement["complete"] and measurement["measured_units"] == measurement["eligible_units"]:
                measurement["value"] = measurement[partial]
            else:
                measurement["complete"] = False
            setattr(self.record, key, measurement["value"])
            if measurement["value"] is None:
                self.unavailable.setdefault(key, ", ".join(measurement["reason_codes"]))
        native = self.counters.get("native_frames")
        # Requests and native frames can count different things. Only an adapter's
        # explicit same-scope discarded frame count is defensible as an extra.
        extras = self.counters.get("discarded_frames")
        if extras is None:
            self.unavailable["native_extra_frames"] = "No explicit same-scope discarded-frame counter; native-requested is not inferred"
        measurement_complete = all(value["complete"] for value in [*self.measurements.values(), *counter_measurements.values()] if value["applicability"] == "applicable")
        self._finished = {**self.record.to_dict(), "measurements": self.measurements,
            "counter_measurements": counter_measurements,
            "counter_read_coverage": counter_read_coverage,
            "counter_reads": self.counter_reads,
            "measurement_complete": measurement_complete,
            "execution_complete": self.execution_complete, "accounting_calls": self.calls,
            "phases": self.phases, "state_samples": self.state_samples,
            "counter_intervals": self.counter_intervals, "adapter_counters": self.counters,
            "native_extra_frames": extras, "native_frames": native,
            "context_frames": self.counters.get("context_frames"),
            "discarded_frames": self.counters.get("discarded_frames"),
            "unavailable": self.unavailable,
            "measurement_scope": {
                "attempt_wall": "collector creation through final sampling; includes input checks, scoring/IO and replay; excludes final resource/receipt writes",
                "rss": "process lifetime high water, not adapter-exclusive or per-attempt allocation",
                "gpu_time": "nonoverlapping model phase synchronized host wall; not CUDA kernel time",
                "gpu_peak": "device allocator high water since collector initialization; includes shared weights",
                "dynamic_state": "peak supported adapter-declared state storage; excludes weights/Python/invisible caches",
                "disk": "logical file lengths at sampling; excludes later resource/receipt writes",
                "counter_source": "adapter cumulative contract only; operation logs are consistency evidence, not added again"}}
        return copy.deepcopy(self._finished)
