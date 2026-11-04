"""Public-input worker. Only I0/current commands cross this process boundary."""
from __future__ import annotations
import copy
import io
import json
import sys
import uuid
import numpy as np
from .worker_ipc import VERSION, WorkerProtocolError, read_frame, write_frame, public_error_code
from .resources import ResourceCollector, state_storage
from .registry import create_adapter, validate_config

SHAPE = (240, 416, 3)
FRAME_BYTES = int(np.prod(SHAPE))
PUBLIC_SPEC_FIELDS = {"viewport_hw", "query_bounds_xy", "coordinate_unit", "normalization", "x_direction", "y_direction", "world_hw"}


class _DiscardLog:
    def write(self, value):
        return len(value)
    def flush(self):
        pass


def validate_public_spec(spec):
    if not isinstance(spec, dict) or set(spec) - PUBLIC_SPEC_FIELDS or spec.get("viewport_hw") != [240, 416]:
        raise WorkerProtocolError("PUBLIC_SPEC_INVALID")
    bounds = spec.get("query_bounds_xy")
    if (not isinstance(bounds, list) or len(bounds) != 2 or
        any(not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair)
            or not -704 <= pair[0] < 0 < pair[1] <= 704 for pair in bounds)):
        raise WorkerProtocolError("PUBLIC_QUERY_BOUNDS_INVALID")
    if spec.get("x_direction") != "right" or spec.get("y_direction") != "down":
        raise WorkerProtocolError("PUBLIC_AXIS_INVALID")
    if "world_hw" in spec and spec["world_hw"] not in ([2048, 2048], [1024, 1024]):
        raise WorkerProtocolError("PUBLIC_WORLD_SHAPE_INVALID")
    return spec


class PublicSession:
    def __init__(self):
        self.next_request = 1
        self.model = self.collector = None
        self.ready, self.finished = False, False
        self.snapshots, self.snapshot_samples = {}, []
        self.max_query_batch = 32
        self.spec = None

    def handle(self, message, payload):
        if self.finished or message.get("version") != VERSION or type(message.get("request_id")) is not int or message["request_id"] != self.next_request:
            raise WorkerProtocolError("PUBLIC_REQUEST_IDENTITY_INVALID")
        self.next_request += 1
        method = message.get("method")
        fields = {"hello": {"config", "max_query_batch"}, "reset": {"spec", "seed"}, "generate": {"queries"},
                  "snapshot": set(), "restore": {"snapshot_id"}, "counters": set(), "close": set()}
        if method not in fields or set(message) != {"version", "request_id", "method"} | fields[method]:
            raise WorkerProtocolError("PUBLIC_REQUEST_FIELDS_INVALID")
        if method != "reset" and payload:
            raise WorkerProtocolError("PUBLIC_UNEXPECTED_PAYLOAD")
        result, outgoing = {}, b""
        if method == "hello":
            if self.model is not None or message["max_query_batch"] not in (16, 32) or type(message["max_query_batch"]) is not int:
                raise WorkerProtocolError("PUBLIC_HELLO_INVALID")
            config = validate_config(message["config"])
            if config["adapter"] == "oracle":
                raise WorkerProtocolError("ORACLE_NOT_PUBLIC")
            self.max_query_batch = message["max_query_batch"]
            self.collector = ResourceCollector(config.get("device", "cpu"))
            with self.collector.phase("model_load"):
                self.model = create_adapter(config)
            result = {"loaded": True, "resource_owner": "worker_process"}
        elif self.model is None:
            raise WorkerProtocolError("PUBLIC_HELLO_REQUIRED")
        elif method == "reset":
            self.ready = False
            spec = validate_public_spec(message["spec"])
            if type(message["seed"]) is not int or not 0 <= message["seed"] < 2 ** 64 or len(payload) != FRAME_BYTES:
                raise WorkerProtocolError("PUBLIC_INITIAL_IMAGE_OR_SEED_INVALID")
            initial = np.frombuffer(payload, dtype=np.uint8).reshape(SHAPE).copy()
            with self.collector.phase("initialize", self.model):
                self.model.reset(initial, copy.deepcopy(spec), message["seed"])
            self.spec, self.ready = copy.deepcopy(spec), True
            # Tokens identify this initialization, never a previous task/reset.
            self.snapshots.clear()
            result = {"initialized": True}
        elif method == "generate":
            queries = np.asarray(message["queries"])
            if not self.ready or queries.ndim != 2 or queries.shape[1] != 2 or not 0 < len(queries) <= self.max_query_batch or queries.dtype.kind not in "iu":
                raise WorkerProtocolError("PUBLIC_QUERY_SHAPE_INVALID")
            bounds = self.spec["query_bounds_xy"]
            if any(np.any(queries[:, axis] < pair[0]) or np.any(queries[:, axis] > pair[1]) for axis, pair in enumerate(bounds)):
                raise WorkerProtocolError("PUBLIC_QUERY_RANGE_INVALID")
            with self.collector.phase("generate", self.model, count=len(queries)):
                response = self.model.generate(queries.astype(np.int64).copy())
            # Output conversion is evaluator-side here and never called model work.
            with self.collector.phase("output_validation"):
                images, positions = [], []
                if isinstance(response, (list, tuple)) and len(response) == len(queries):
                    for index, image in enumerate(response):
                        if isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.shape == SHAPE:
                            images.append(np.ascontiguousarray(image).copy().tobytes()); positions.append(index)
                outgoing = b"".join(images)
                self.collector.calls["valid_outputs"] = (self.collector.calls["valid_outputs"] or 0) + len(positions)
                result = {"count": len(queries), "valid_indices": positions, "shape": list(SHAPE), "dtype": "uint8"}
        elif method == "snapshot":
            if not self.ready or len(self.snapshots) >= 8:
                raise WorkerProtocolError("PUBLIC_SNAPSHOT_LIMIT_OR_STATE_INVALID")
            with self.collector.phase("snapshot", self.model):
                state = copy.deepcopy(self.model.snapshot())
                token = uuid.uuid4().hex
                self.snapshots[token] = state
            self.snapshot_samples.append(state_storage(self.snapshots))
            result = {"snapshot_id": token}
        elif method == "restore":
            self.ready = False
            token = message["snapshot_id"]
            if not isinstance(token, str) or token not in self.snapshots:
                raise WorkerProtocolError("PUBLIC_SNAPSHOT_UNKNOWN")
            with self.collector.phase("restore", self.model):
                self.model.restore(copy.deepcopy(self.snapshots[token]))
            self.ready = True
            result = {"restored": True}
        elif method == "counters":
            read = self.collector._counters(self.model)
            result = {"status": read.status, "values": read.values, "counter_epoch": read.counter_epoch,
                      "initial_values": read.initial_values, "error_code": read.error_code}
        elif method == "close":
            resource = self.collector.finish()
            complete = all(sample["complete"] for sample in self.snapshot_samples)
            partial = max((sample["unique_supported_storage_bytes"] for sample in self.snapshot_samples), default=None)
            measurement = {"value": partial if complete else None, "observed_peak": partial,
                           "eligible_units": len(self.snapshot_samples), "measured_units": sum(sample["complete"] for sample in self.snapshot_samples),
                           "complete": complete, "aggregation": "max", "unit": "bytes",
                           "applicability": "applicable" if self.snapshot_samples else "not_applicable",
                           "reason_codes": sorted({code for sample in self.snapshot_samples for code in sample["reason_codes"]}),
                           "scope": "worker-managed immutable snapshot stores, observed at snapshot creation; unsupported owners remain unknown"}
            resource["worker_snapshot_storage_bytes"] = measurement["value"]
            resource["measurements"]["worker_snapshot_storage_bytes"] = measurement
            resource["measurement_complete"] = resource["measurement_complete"] and complete
            resource["resource_owner"] = "worker_process"
            resource["measurement_scope"]["boundary"] = "worker construction through close; evaluator ledger/PNG/replay and client IPC waits excluded"
            result = {"resources": resource}
            self.finished = True
        return {"version": VERSION, "request_id": message["request_id"], "status": "ok", "result": result}, outgoing


def main():
    # Models may log arbitrary text: it must not corrupt IPC or disclose exception text.
    input_stream, output_stream = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = _DiscardLog()
    sys.stderr = _DiscardLog()
    session = PublicSession()
    while not session.finished:
        message = {}
        try:
            message, payload = read_frame(input_stream)
            response, outgoing = session.handle(message, payload)
        except WorkerProtocolError as exc:
            response = {"version": VERSION, "request_id": message.get("request_id", 0), "status": "error", "error_code": public_error_code(exc)}
            outgoing = b""
        except Exception:
            response = {"version": VERSION, "request_id": message.get("request_id", 0), "status": "error",
                        "error_code": "WORKER_EXECUTION_ERROR"}
            outgoing = b""
        try:
            write_frame(output_stream, response, outgoing)
        except (BrokenPipeError, WorkerProtocolError):
            return 1
        if response["status"] == "error":
            # A malformed or failed operation cannot leave an ambiguous live state.
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
