"""Public adapter over bounded IPC, with explicit cooperative/restricted profiles."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
from .isolation_backend import IsolationBlocked, WorkerLimits, launch_description, permission_probe
from .worker_ipc import VERSION, PipeTransport, WorkerProtocolError, ERROR_CODES
from .worker_public import FRAME_BYTES, SHAPE, validate_public_spec


class WorkerClient:
    def __init__(self, config, profile="isolated_worker", max_query_batch=32, limits=None):
        if type(max_query_batch) is not int or max_query_batch not in (16, 32):
            raise ValueError("Public call limit must be 16 or 32")
        self.limits = limits or WorkerLimits()
        self.description = launch_description(config, profile, self.limits)
        self.profile, self.max_query_batch = profile, max_query_batch
        self.protocol_security_verified = False
        self.worker_resources = None
        self._next_request, self._closed = 1, False
        self._temporary = tempfile.TemporaryDirectory(prefix="rswm-public-client-")
        self.transport = PipeTransport(self.description["command"], self.limits.timeout_seconds,
                                       self.description["environment"], self._temporary.name,
                                       lifetime_timeout=self.limits.lifetime_seconds)
        try:
            self._rpc("hello", config=self.description["config"], max_query_batch=max_query_batch)
        except Exception:
            self.close(finalize=False)
            raise

    def _rpc(self, method, payload=b"", **fields):
        if self._closed:
            raise WorkerProtocolError("WORKER_CLIENT_CLOSED")
        request = {"version": VERSION, "request_id": self._next_request, "method": method, **fields}
        self._next_request += 1
        try:
            response, outgoing = self.transport.exchange(request, payload)
        except BaseException:
            self.close(finalize=False)
            raise
        if response.get("status") == "error":
            code = response.get("error_code")
            if set(response) != {"version", "request_id", "status", "error_code"} or outgoing or not isinstance(code, str) or code not in ERROR_CODES:
                code = "IPC_RESPONSE_FIELDS_INVALID"
            self._invalid(code)
        if set(response) != {"version", "request_id", "status", "result"} or response.get("status") != "ok":
            self._invalid("IPC_RESPONSE_FIELDS_INVALID")
        if not isinstance(response.get("result"), dict):
            self._invalid("IPC_RESULT_NOT_OBJECT")
        return response["result"], outgoing

    def _invalid(self, code):
        self.close(finalize=False)
        raise WorkerProtocolError(code)

    def reset(self, initial, spec, seed):
        validate_public_spec(spec)
        if not isinstance(initial, np.ndarray) or initial.dtype != np.uint8 or initial.shape != SHAPE:
            raise WorkerProtocolError("PUBLIC_INITIAL_IMAGE_INVALID")
        if type(seed) is not int or not 0 <= seed < 2 ** 64:
            raise WorkerProtocolError("PUBLIC_SEED_INVALID")
        self._rpc("reset", np.ascontiguousarray(initial).copy().tobytes(), spec=copy.deepcopy(spec), seed=seed)

    def generate(self, queries):
        q = np.asarray(queries)
        if q.ndim != 2 or q.shape[1] != 2 or q.dtype.kind not in "iu" or not 0 < len(q) <= self.max_query_batch:
            raise WorkerProtocolError("PUBLIC_QUERY_SHAPE_INVALID")
        # Bounds are checked again inside the worker against the public reset spec.
        if np.any(q > 704) or (q.dtype.kind == "i" and np.any(q < -704)):
            raise WorkerProtocolError("PUBLIC_QUERY_RANGE_INVALID")
        result, outgoing = self._rpc("generate", queries=q.astype(np.int64).tolist())
        positions = result.get("valid_indices")
        if (type(result.get("count")) is not int or result["count"] != len(q) or result.get("shape") != list(SHAPE) or result.get("dtype") != "uint8"
            or not isinstance(positions, list) or any(type(p) is not int or not 0 <= p < len(q) for p in positions)
            or positions != sorted(set(positions)) or len(outgoing) != len(positions) * FRAME_BYTES):
            self._invalid("PUBLIC_RESPONSE_IMAGES_INVALID")
        images = [None] * len(q)
        for offset, index in enumerate(positions):
            images[index] = np.frombuffer(outgoing[offset * FRAME_BYTES:(offset + 1) * FRAME_BYTES], dtype=np.uint8).reshape(SHAPE).copy()
        return images

    def snapshot(self):
        result, outgoing = self._rpc("snapshot")
        token = result.get("snapshot_id")
        if outgoing or not isinstance(token, str) or len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
            self._invalid("PUBLIC_SNAPSHOT_TOKEN_INVALID")
        return {"schema": "rswm-worker-snapshot-v1", "snapshot_id": token}

    def restore(self, snapshot):
        if not isinstance(snapshot, dict) or set(snapshot) != {"schema", "snapshot_id"} or snapshot["schema"] != "rswm-worker-snapshot-v1":
            raise WorkerProtocolError("PUBLIC_SNAPSHOT_TOKEN_INVALID")
        self._rpc("restore", snapshot_id=snapshot["snapshot_id"])

    def resource_counters(self):
        result, outgoing = self._rpc("counters")
        if outgoing:
            self._invalid("PUBLIC_COUNTER_PAYLOAD_INVALID")
        return result

    def resource_state(self):
        # The client's own buffers are not native worker model state. The actual
        # worker collector samples that state and emits its receipt at close.
        raise WorkerProtocolError("REMOTE_STATE_MEASURED_ONLY_BY_WORKER")

    def close(self, finalize=True):
        if self._closed:
            return copy.deepcopy(self.worker_resources)
        try:
            if finalize:
                result, outgoing = self._rpc("close")
                if outgoing or not isinstance(result.get("resources"), dict):
                    raise WorkerProtocolError("PUBLIC_RESOURCE_RECEIPT_INVALID")
                self.worker_resources = copy.deepcopy(result["resources"])
                self.worker_resources["execution_mode"] = self.profile
                self.worker_resources["protocol_security_verified"] = False
                self.worker_resources["worker_collector_trust"] = "in-worker observations; not hostile-code attestation"
        finally:
            self._closed = True
            self.transport.close()
            if self.worker_resources is not None:
                self.worker_resources["external_process_observation"] = copy.deepcopy(self.transport.external_process_observation)
            self._temporary.cleanup()
        return copy.deepcopy(self.worker_resources)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        # Preserve an original evaluator/model failure if cleanup also fails.
        try:
            self.close(finalize=kind is None)
        except Exception:
            if kind is None:
                raise


def snapshot_branch_probe(client, initial, spec, seed=0):
    """A diagnostic gate: repeated identical branch from one snapshot must agree.

    This deliberately uses a fresh public session and no private task/schedule.
    It tests a restoration property, without treating subprocess transport as
    an isolation proof or missing outputs as successful matching states.
    """
    queries = np.zeros((16, 2), np.int64)
    client.reset(initial, spec, seed)
    client.generate(queries)
    token = client.snapshot()
    first = client.generate(queries)
    client.restore(token)
    second = client.generate(queries)
    valid = sum(a is not None and b is not None for a, b in zip(first, second))
    matching = sum(a is not None and b is not None and np.array_equal(a, b) for a, b in zip(first, second))
    return {"schema": "rswm-worker-snapshot-branch-probe-v1", "status": "PASS" if matching == 16 else "FAIL",
            "expected_branch_outputs": 16, "both_branches_valid": valid, "matching_outputs": matching,
            "protocol_security_verified": False, "scope": "identical public query branch restored from one opaque snapshot"}


def run_public_worker(task, world, config, out, profile="isolated_worker", limits=None, seed=0):
    """Evaluator-side entry: private arrays stay here, only runner API enters IPC."""
    from .runner import run_task
    from .replay import replay_run
    from .geometry import crop, canonical
    target = Path(out)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError("Worker evaluation output must be empty")
    client = WorkerClient(config, profile, 16 if task["protocol"] == "online16" else 32, limits)
    resource_error = None
    try:
        result, journal = run_task(task, world, client, seed=seed, out_dir=target / "submission")
    except BaseException:
        try:
            client.close(finalize=False)
        except Exception:
            pass
        raise
    try:
        client.close()
    except WorkerProtocolError as exc:
        resource_error = str(exc)
    replayed = replay_run(task, crop(world, (0, 0), task["origin"]), target / "submission", expected_head=journal.head)
    resources = client.worker_resources or {"schema": "rswm-worker-resource-unavailable-v1", "measurement_complete": False,
                                           "resource_status": "NOT_READABLE", "error_code": resource_error,
                                           "external_process_observation": client.transport.external_process_observation}
    # This evaluator-owned denominator never crosses IPC to the model. The
    # actual model phases and counters in the receipt remain worker observations.
    resources.setdefault("evaluator_accounting", {}).update(
        planned_output_slots=replayed["planned_output_events"],
        replay_verified_outputs=replayed["verified_png_outputs"],
        scope="private evaluator schedule and immutable submission replay; not sent to worker")
    (target / "worker-resources.json").write_bytes(canonical(resources) + b"\n")
    report = {"schema": "rswm-public-worker-run-v1", "execution_mode": profile, "reports": replayed["reports"],
              "journal_head": journal.head, "verified_png_outputs": replayed["verified_png_outputs"],
              "planned_output_events": replayed["planned_output_events"], "worker_resources_path": "worker-resources.json",
              "protocol_security_verified": False, "real_model_validation": "NOT_RUN", "inference_seed": seed,
              "acceptance_status": "ACCEPTED" if resources.get("measurement_complete") and replayed["verified_png_outputs"] == replayed["planned_output_events"] else "INCOMPLETE"}
    (target / "worker-report.json").write_bytes(canonical(report) + b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-isolation", action="store_true")
    parser.add_argument("--profile", choices=["isolated_worker", "cooperative_subprocess"], default="isolated_worker")
    parser.add_argument("--task"); parser.add_argument("--world"); parser.add_argument("--config"); parser.add_argument("--out")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.probe_isolation:
        report = permission_probe(); print(json.dumps(report))
        return 0 if report["status"] == "PASS" else 3
    if not all((args.task, args.world, args.config, args.out)):
        parser.error("--task --world --config --out are required")
    try:
        from .suite import _json, _world, _load_task
        task, _ = _load_task(args.task)
        report = run_public_worker(task, _world(args.world), _json(args.config), args.out, args.profile, seed=args.seed)
    except IsolationBlocked as exc:
        print(json.dumps(exc.report)); return 3
    except (ValueError, OSError, WorkerProtocolError) as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__, "error_code": str(exc) if isinstance(exc, WorkerProtocolError) else "WORKER_INPUT_OR_EXECUTION_ERROR"})); return 1
    print(json.dumps({"status": "COMPLETED", "execution_mode": args.profile, "protocol_security_verified": False,
                      "planned_outputs": report["planned_output_events"], "verified_outputs": report["verified_png_outputs"]}))
    return 0 if report["acceptance_status"] == "ACCEPTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
