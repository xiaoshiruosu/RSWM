"""Actual IPC/lifecycle tests are cooperative; OS permission acceptance is separate."""
import copy
import io
import json
import os
from pathlib import Path
import select
import signal
import struct
import subprocess
import sys
import time
import numpy as np
import pytest
from rswm_inria.worker_ipc import VERSION, HEADER, MAGIC, MAX_METADATA, MAX_PAYLOAD, PipeTransport, WorkerProtocolError, encode_frame, decode_frame, read_frame
from rswm_inria.worker_public import PublicSession, FRAME_BYTES
from rswm_inria.worker_client import WorkerClient, run_public_worker, snapshot_branch_probe
from rswm_inria.isolation_backend import IsolationBlocked, WorkerLimits, public_environment, launch_description, probe_isolation, permission_probe
from rswm_inria.runner import PUBLIC_SPEC
from rswm_inria.tasks import packages


def request(identifier, method, **fields):
    return {"version": VERSION, "request_id": identifier, "method": method, **fields}


def process_has_exited(pid, timeout):
    """Observe the kernel process identity in our PID namespace.

    A host-mounted /proc can resolve the same numeric PID to an unrelated host
    process. pidfds also become readable for exited, not-yet-reaped descendants.
    """
    if not hasattr(os, "pidfd_open"):
        pytest.skip("Actual namespace-aware lifecycle check requires Linux pidfds")
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return True
    try:
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        return bool(poller.poll(round(timeout * 1000)))
    finally:
        os.close(descriptor)


def test_actual_lifecycle_observer_rejects_a_live_process():
    process = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"])
    try:
        assert not process_has_exited(process.pid, .01)
    finally:
        process.kill()
        process.wait(timeout=2.)
    assert process_has_exited(process.pid, .01)


def test_binary_round_trip_and_bounded_lengths():
    message = request(1, "reset", spec=PUBLIC_SPEC, seed=0)
    payload = bytes(FRAME_BYTES)
    encoded = encode_frame(message, payload)
    assert decode_frame(encoded) == (message, payload)
    assert read_frame(io.BytesIO(encoded)) == (message, payload)
    for header in (HEADER.pack(b"BAD!", 1, 0), HEADER.pack(MAGIC, MAX_METADATA + 1, 0), HEADER.pack(MAGIC, 2, MAX_PAYLOAD + 1)):
        with pytest.raises(WorkerProtocolError):
            read_frame(io.BytesIO(header))


@pytest.mark.parametrize("metadata", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}', b'[]', b'{"x":'])
def test_invalid_json_rejected(metadata):
    with pytest.raises(WorkerProtocolError):
        decode_frame(HEADER.pack(MAGIC, len(metadata), 0) + metadata)


def test_truncation_and_unexpected_tail_rejected():
    valid = encode_frame(request(1, "close"))
    for bad in (valid[:5], valid[:-1], valid + b"secret"):
        with pytest.raises(WorkerProtocolError):
            decode_frame(bad)


def test_session_only_public_fields_current_batch_and_ids():
    session = PublicSession()
    session.handle(request(1, "hello", config={"model_id": "copy", "adapter": "copy"}, max_query_batch=16), b"")
    with pytest.raises(WorkerProtocolError, match="PUBLIC_SPEC_INVALID"):
        session.handle(request(2, "reset", spec={**PUBLIC_SPEC, "future_queries": [[1, 2]]}, seed=0), bytes(FRAME_BYTES))
    with pytest.raises(WorkerProtocolError, match="IDENTITY"):
        session.handle(request(2, "close"), b"")
    session.handle(request(3, "reset", spec=PUBLIC_SPEC, seed=0), bytes(FRAME_BYTES))
    with pytest.raises(WorkerProtocolError, match="QUERY_SHAPE"):
        session.handle(request(4, "generate", queries=[[0, 0]] * 17), b"")


@pytest.mark.parametrize("q", [[[1.5, 0]], [[705, 0]], [[0, -705]], [[True, False]], ["future"], []])
def test_worker_query_attack_rejected(q):
    session = PublicSession()
    session.handle(request(1, "hello", config={"model_id": "copy", "adapter": "copy"}, max_query_batch=32), b"")
    session.handle(request(2, "reset", spec=PUBLIC_SPEC, seed=0), bytes(FRAME_BYTES))
    with pytest.raises(WorkerProtocolError):
        session.handle(request(3, "generate", queries=q), b"")


def test_actual_cooperative_worker_outputs_owned_copy_and_native_resources():
    initial = np.zeros((240, 416, 3), np.uint8)
    with WorkerClient({"model_id": "copy", "adapter": "copy"}, "cooperative_subprocess") as client:
        client.reset(initial, PUBLIC_SPEC, 3)
        output = client.generate(np.array([[16, 0], [32, 0]], dtype=np.int64))
        output[0][:] = 99
        later = client.generate(np.array([[48, 0]], dtype=np.int64))
        assert later[0].max() == 0 and output[1].max() == 0
        assert client.protocol_security_verified is False
        pid = client.transport.process.pid
    assert client.transport.process.returncode is not None
    assert client.worker_resources["resource_owner"] == "worker_process"
    assert client.worker_resources["execution_mode"] == "cooperative_subprocess"
    assert client.worker_resources["process_peak_rss_bytes"] > 0
    assert client.worker_resources["generate_seconds"] > 0
    assert client.worker_resources["adapter_counters"]["requested_outputs"] == 3
    assert client.worker_resources["external_process_observation"]["source"] == "evaluator os.wait4"
    assert client.worker_resources["external_process_observation"]["process_peak_rss_bytes"] > 0
    assert client.close() == client.worker_resources


def test_actual_random_worker_snapshot_restores_rng_branch_state():
    with WorkerClient({"model_id": "random", "adapter": "random"}, "cooperative_subprocess") as client:
        client.reset(np.zeros((240, 416, 3), np.uint8), PUBLIC_SPEC, 11)
        snapshot = client.snapshot()
        branch_a = client.generate(np.array([[16, 0]], np.int64))[0]
        client.generate(np.array([[32, 0]], np.int64))
        client.restore(snapshot)
        branch_b = client.generate(np.array([[16, 0]], np.int64))[0]
        assert np.array_equal(branch_a, branch_b)
        assert set(snapshot) == {"schema", "snapshot_id"}  # no model state/path returns to evaluator
        with pytest.raises(WorkerProtocolError):
            client.restore({**snapshot, "state_path": "/private/snapshot"})
    # Explicit unsupported RNG object storage cannot become a complete memory claim.
    assert not client.worker_resources["measurements"]["worker_snapshot_storage_bytes"]["complete"]


def test_window_worker_f7_restore_does_not_rollback_native_cost():
    config = {"model_id": "lazy", "adapter": "lazy-map", "diagnostic_noise": True}
    with WorkerClient(config, "cooperative_subprocess", max_query_batch=16) as client:
        client.reset(np.zeros((240, 416, 3), np.uint8), PUBLIC_SPEC, 1)
        snapshot = client.snapshot()
        client.generate(np.array([[448, 0]], np.int64))
        first = client.resource_counters()["values"]["completion_calls"]
        client.restore(snapshot)
        client.generate(np.array([[448, 0]], np.int64))
        second = client.resource_counters()["values"]["completion_calls"]
        assert second == first * 2 and first > 0


def test_actual_transport_timeout_kills_worker(tmp_path):
    command = [sys.executable, "-c", "import time;time.sleep(30)"]
    transport = PipeTransport(command, timeout=.1, env=public_environment())
    with pytest.raises(WorkerProtocolError, match="WORKER_TIMEOUT"):
        transport.exchange(request(1, "hello"))
    assert transport.process.returncode is not None


def test_actual_transport_lifetime_kills_idle_worker_and_is_idempotent():
    transport = PipeTransport([sys.executable, "-c", "import time;time.sleep(30)"],
                              timeout=5., lifetime_timeout=.15, env=public_environment())
    deadline = time.monotonic() + 2.
    while transport.process.returncode is None and time.monotonic() < deadline:
        time.sleep(.01)
    assert transport.process.returncode is not None
    assert transport.external_process_observation["cpu_user_seconds"] >= 0
    with pytest.raises(WorkerProtocolError, match="WORKER_LIFETIME_EXCEEDED"):
        transport.exchange(request(1, "hello"))
    transport.close()


def test_actual_repeated_successful_calls_cannot_reset_worker_lifetime():
    script = """import sys,time
from rswm_inria.worker_ipc import *
while True:
 m,p=read_frame(sys.stdin.buffer)
 time.sleep(.05)
 write_frame(sys.stdout.buffer,{'version':VERSION,'request_id':m['request_id'],'status':'ok','result':{}})
"""
    transport = PipeTransport([sys.executable, "-c", script], timeout=2., lifetime_timeout=.5,
                              env=public_environment(Path(__file__).resolve().parents[1] / "src"))
    completed = 0
    with pytest.raises(WorkerProtocolError, match="WORKER_LIFETIME_EXCEEDED"):
        for identifier in range(1, 21):
            transport.exchange(request(identifier, "hello")); completed += 1
    assert 0 < completed < 20
    assert transport.process.returncode is not None


@pytest.mark.parametrize("body,expected", [
    (b"truncated", "WORKER_EOF"),
    (HEADER.pack(MAGIC, MAX_METADATA + 1, 0), "IPC_LENGTH_EXCEEDED"),
    (HEADER.pack(MAGIC, 2, 0) + b"[]", "IPC_METADATA_NOT_OBJECT"),
    (HEADER.pack(MAGIC, 1, 0) + b"{", "IPC_JSON_INVALID"),
])
def test_actual_malformed_peer_is_rejected_and_reaped(body, expected):
    script = "import os,sys;os.read(0,4096);os.write(1," + repr(body) + ")"
    transport = PipeTransport([sys.executable, "-c", script], timeout=1., env=public_environment())
    with pytest.raises(WorkerProtocolError, match=expected):
        transport.exchange(request(1, "hello"))
    assert transport.process.returncode is not None
    assert transport.external_process_observation is not None


def test_actual_client_rejects_unknown_error_without_echo_and_cleans_temp(monkeypatch):
    import rswm_inria.worker_client as module
    script = "from rswm_inria.worker_ipc import *;import sys;m,p=read_frame(sys.stdin.buffer);write_frame(sys.stdout.buffer,{'version':VERSION,'request_id':m['request_id'],'status':'error','error_code':'PRIVATE_FILE_CONTENT_SENTINEL'})"
    environment = public_environment(Path(__file__).resolve().parents[1] / "src")
    monkeypatch.setattr(module, "launch_description", lambda *a: {
        "command": [sys.executable, "-c", script], "environment": environment,
        "config": {"model_id": "copy", "adapter": "copy"}})
    clients = []
    transport_type = module.PipeTransport
    def capture(*args, **kwargs):
        transport = transport_type(*args, **kwargs); clients.append(transport); return transport
    monkeypatch.setattr(module, "PipeTransport", capture)
    with pytest.raises(WorkerProtocolError, match="^IPC_RESPONSE_FIELDS_INVALID$") as error:
        module.WorkerClient({"model_id": "copy", "adapter": "copy"}, "cooperative_subprocess")
    assert "SENTINEL" not in str(error.value)
    assert clients[0].process.returncode is not None


def test_actual_worker_model_protocol_exception_cannot_disclose_text():
    script = """import rswm_inria.worker_public as worker
from rswm_inria.worker_ipc import WorkerProtocolError
class MaliciousModel:
 def reset(self,*args): raise WorkerProtocolError('PRIVATE_FILE_CONTENT_SENTINEL')
worker.create_adapter=lambda config:MaliciousModel()
raise SystemExit(worker.main())
"""
    transport = PipeTransport([sys.executable, "-c", script], timeout=3.,
                              env=public_environment(Path(__file__).resolve().parents[1] / "src"))
    try:
        transport.exchange(request(1, "hello", config={"model_id": "copy", "adapter": "copy"}, max_query_batch=16))
        response, body = transport.exchange(request(2, "reset", spec=PUBLIC_SPEC, seed=0), bytes(FRAME_BYTES))
        assert response["error_code"] == "WORKER_EXECUTION_ERROR" and body == b""
        assert "SENTINEL" not in json.dumps(response)
    finally:
        transport.close()


def test_snapshot_token_cannot_restore_prior_initialization():
    session = PublicSession()
    session.handle(request(1, "hello", config={"model_id": "copy", "adapter": "copy"}, max_query_batch=16), b"")
    session.handle(request(2, "reset", spec=PUBLIC_SPEC, seed=1), bytes(FRAME_BYTES))
    response, _ = session.handle(request(3, "snapshot"), b"")
    token = response["result"]["snapshot_id"]
    session.handle(request(4, "reset", spec=PUBLIC_SPEC, seed=2), bytes(FRAME_BYTES))
    with pytest.raises(WorkerProtocolError, match="PUBLIC_SNAPSHOT_UNKNOWN"):
        session.handle(request(5, "restore", snapshot_id=token), b"")


def test_actual_snapshot_branch_probe_passes_random_state_restore():
    with WorkerClient({"model_id": "random", "adapter": "random"}, "cooperative_subprocess", max_query_batch=16) as client:
        result = snapshot_branch_probe(client, np.zeros((240, 416, 3), np.uint8), PUBLIC_SPEC, 7)
        assert result["status"] == "PASS" and result["matching_outputs"] == 16
        assert result["protocol_security_verified"] is False


def test_actual_snapshot_branch_pollution_is_a_failed_gate(monkeypatch):
    import rswm_inria.worker_client as module
    script = """import numpy as np
import rswm_inria.worker_public as worker
class ContaminatedModel:
 def reset(self,*args): self.cursor=0
 def generate(self,q):
  result=[np.full((240,416,3),self.cursor+i,np.uint8) for i in range(len(q))]
  self.cursor+=len(q);return result
 def snapshot(self): return {'cursor':self.cursor}
 def restore(self,state): pass
worker.create_adapter=lambda config:ContaminatedModel()
raise SystemExit(worker.main())
"""
    description = launch_description({"model_id": "copy", "adapter": "copy"}, "cooperative_subprocess")
    prefix = description["command"][:description["command"].index("--") + 1]
    description["command"] = prefix + [sys.executable, "-c", script]
    monkeypatch.setattr(module, "launch_description", lambda *args: description)
    with WorkerClient({"model_id": "copy", "adapter": "copy"}, "cooperative_subprocess", max_query_batch=16) as client:
        result = snapshot_branch_probe(client, np.zeros((240, 416, 3), np.uint8), PUBLIC_SPEC, 7)
        assert result["both_branches_valid"] == 16
        assert result["matching_outputs"] == 0 and result["status"] == "FAIL"
        assert result["protocol_security_verified"] is False


def test_actual_prlimit_values_observed_in_child_process():
    description = launch_description({"model_id": "copy", "adapter": "copy"}, "cooperative_subprocess")
    prefix = description["command"][:description["command"].index("--") + 1]
    script = "import json,resource;print(json.dumps({key:list(resource.getrlimit(getattr(resource,key))) for key in ('RLIMIT_AS','RLIMIT_CPU','RLIMIT_FSIZE','RLIMIT_NOFILE','RLIMIT_NPROC','RLIMIT_CORE')}))"
    completed = subprocess.run(prefix + [sys.executable, "-c", script], env=description["environment"], capture_output=True, text=True, timeout=3.)
    assert completed.returncode == 0
    values = json.loads(completed.stdout)
    limits = WorkerLimits()
    assert values == {"RLIMIT_AS": [limits.address_space_bytes] * 2, "RLIMIT_CPU": [limits.cpu_seconds] * 2,
                      "RLIMIT_FSIZE": [limits.file_bytes] * 2, "RLIMIT_NOFILE": [limits.open_files] * 2,
                      "RLIMIT_NPROC": [limits.processes] * 2, "RLIMIT_CORE": [0, 0]}


def test_actual_transport_cleans_inherited_group_child(tmp_path):
    script = """import os,sys,time
from rswm_inria.worker_ipc import *
message,body=read_frame(sys.stdin.buffer)
child=os.fork()
if child==0:
 time.sleep(30);os._exit(0)
write_frame(sys.stdout.buffer,{'version':VERSION,'request_id':message['request_id'],'status':'ok','child_pid':child})
time.sleep(30)
"""
    transport = PipeTransport([sys.executable, "-c", script], env=public_environment(Path(__file__).resolve().parents[1] / "src"))
    response, _ = transport.exchange(request(1, "hello"))
    child = response["child_pid"]
    transport.close()
    assert process_has_exited(child, 2.), "Inherited-group child continued running after transport close"
    assert transport.process.returncode is not None


def test_actual_permission_probe_timeout_cleans_its_child_group(tmp_path):
    from rswm_inria.isolation_backend import _run_probe
    child_record = tmp_path / "probe-child-pid"
    script = """import os,pathlib,sys,time
child=os.fork()
if child==0:
 time.sleep(30);os._exit(0)
pathlib.Path(sys.argv[1]).write_text(str(child))
time.sleep(30)
"""
    with pytest.raises(subprocess.TimeoutExpired):
        _run_probe([sys.executable, "-c", script, str(child_record)], public_environment(), .2)
    child = int(child_record.read_text())
    assert process_has_exited(child, 2.), "Denied probe left an inherited-group child running"


def test_actual_public_wire_has_no_private_or_future_fields():
    with WorkerClient({"model_id": "copy", "adapter": "copy"}, "cooperative_subprocess", max_query_batch=16) as client:
        captured = []
        exchange = client.transport.exchange
        def spy(message, payload=b""):
            captured.append(copy.deepcopy(message))
            return exchange(message, payload)
        client.transport.exchange = spy
        client.reset(np.zeros((240, 416, 3), np.uint8), PUBLIC_SPEC, 7)
        client.generate(np.array([[16, 0], [32, 0]], np.int64))
        encoded = json.dumps(captured)
        assert not any(key in encoded for key in ("world_id", "origin", "city", "task_id", "labels", "future"))
        assert captured[1]["queries"] == [[16, 0], [32, 0]]


def test_actual_wrong_response_id_is_rejected_and_process_cleaned():
    script = "from rswm_inria.worker_ipc import *;import sys;message,body=read_frame(sys.stdin.buffer);write_frame(sys.stdout.buffer,{'version':VERSION,'request_id':999,'status':'ok'},b'')"
    environment = public_environment(Path(__file__).resolve().parents[1] / "src")
    transport = PipeTransport([sys.executable, "-c", script], env=environment)
    with pytest.raises(WorkerProtocolError, match="IDENTITY_MISMATCH"):
        transport.exchange(request(1, "hello"))
    assert transport.process.returncode is not None


def test_environment_allowlist_no_private_sentinel(monkeypatch):
    monkeypatch.setenv("PRIVATE_ENV_SENTINEL", "synthetic-secret")
    monkeypatch.setenv("HTTP_PROXY", "private-proxy")
    monkeypatch.setenv("PRIVATE_FUTURE_FILE", "/private/future")
    environment = public_environment("/public/code")
    assert not {"PRIVATE_ENV_SENTINEL", "HTTP_PROXY", "PRIVATE_FUTURE_FILE"} & set(environment)
    assert environment["HF_HUB_OFFLINE"] == "1"


def test_restricted_profile_never_silently_falls_back(monkeypatch):
    import rswm_inria.isolation_backend as backend
    blocked = {"status": "BLOCKED", "reason_code": "SYNTHETIC_HOST_CAPABILITY_DENIED", "protocol_security_verified": False}
    monkeypatch.setattr(backend, "probe_isolation", lambda: blocked)
    with pytest.raises(IsolationBlocked) as error:
        launch_description({"model_id": "copy", "adapter": "copy"}, "isolated_worker")
    assert error.value.report == blocked


@pytest.mark.parametrize("config,code", [
    ({"model_id": "numeric", "adapter": "numeric-camera"}, "NUMERIC_BACKEND_ISOLATION_NOT_VALIDATED"),
    ({"model_id": "gpu", "adapter": "copy", "device": "cuda:0"}, "GPU_ISOLATION_PROFILE_NOT_VALIDATED"),
    ({"model_id": "wan", "adapter": "wan-i2v"}, "REAL_MODEL_ISOLATION_PROFILE_NOT_VALIDATED"),
    ({"model_id": "map", "adapter": "frontier", "model_dir": "/not/mounted/weights"}, "REAL_MODEL_ISOLATION_PROFILE_NOT_VALIDATED"),
])
def test_unvalidated_real_and_numeric_profiles_fail_before_namespace_or_asset_loading(monkeypatch, config, code):
    import rswm_inria.isolation_backend as backend
    monkeypatch.setattr(backend, "probe_isolation", lambda: pytest.fail("Unsupported profile must fail before OS probe"))
    with pytest.raises(IsolationBlocked) as error:
        launch_description(config, "isolated_worker")
    assert error.value.report["reason_code"] == code
    assert error.value.report["protocol_security_verified"] is False


@pytest.mark.parametrize("field,value", [("timeout_seconds", float("nan")), ("lifetime_seconds", float("inf")), ("lifetime_seconds", True), ("lifetime_seconds", 0)])
def test_nonfinite_or_invalid_limits_cannot_launch(field, value):
    with pytest.raises(ValueError):
        WorkerLimits(**{field: value}).validate()


def test_cpu_cooperative_complete_f2_evaluator_chain(tmp_path):
    task = next(t for t, _ in packages("worker-end-to-end", "diagnostic_only", 0) if t["family"] == "F2")
    report = run_public_worker(task, np.zeros((2048, 2048, 3), np.uint8),
                               {"model_id": "copy", "adapter": "copy"}, tmp_path / "out", "cooperative_subprocess")
    assert report["planned_output_events"] == report["verified_png_outputs"] == 32
    assert report["protocol_security_verified"] is False
    assert json.loads((tmp_path / "out/worker-resources.json").read_text())["resource_owner"] == "worker_process"


def test_cpu_cooperative_f7_single_prefix_two_immutable_branches_worker_cost(tmp_path):
    task = next(t for t, _ in packages("worker-f7-end-to-end", "diagnostic_only", 0) if t["family"] == "F7")
    report = run_public_worker(task, np.zeros((2048, 2048, 3), np.uint8),
                               {"model_id": "copy", "adapter": "copy"}, tmp_path / "f7", "cooperative_subprocess")
    assert report["planned_output_events"] == report["verified_png_outputs"] == 160
    resources = json.loads((tmp_path / "f7/worker-resources.json").read_text())
    assert resources["accounting_calls"]["model_calls"] == 15  # load/reset/10 generate/snapshot/2 restore
    assert resources["measurements"]["generate_seconds"]["eligible_units"] == 10
    assert resources["accounting_calls"]["requests_issued"] == 160
    assert resources["adapter_counters"]["requested_outputs"] == 160
    assert resources["accounting_calls"]["valid_outputs"] == 160
    assert resources["measurements"]["snapshot_seconds"]["eligible_units"] == 1
    assert resources["measurements"]["restore_seconds"]["eligible_units"] == 2
    assert resources["worker_snapshot_storage_bytes"] == FRAME_BYTES
    assert resources["evaluator_accounting"]["planned_output_slots"] == 160
    assert resources["resource_owner"] == "worker_process"
    assert resources["measurement_complete"] is True
    assert report["protocol_security_verified"] is False


def test_actual_os_permission_gate_or_explicit_blocked():
    result = permission_probe()
    if result["status"] == "BLOCKED":
        assert result["protocol_security_verified"] is False
        assert all(value == "NOT_RUN" for value in result["checks"].values())
        pytest.skip("Real OS permission gate BLOCKED: " + result.get("reason_code", "unknown"))
    assert result["status"] == "PASS" and all(result["checks"].values())
