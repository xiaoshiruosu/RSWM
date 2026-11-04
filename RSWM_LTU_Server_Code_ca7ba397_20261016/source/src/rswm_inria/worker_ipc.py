"""Bounded binary public-worker protocol; transport is not an OS sandbox."""
from __future__ import annotations
import json
import math
import os
import select
import signal
import struct
import subprocess
import sys
import threading
import time

VERSION = "rswm-public-worker-v1"
MAGIC = b"RWI1"
HEADER = struct.Struct(">4sII")
MAX_METADATA = 65536
MAX_PAYLOAD = 32 * 240 * 416 * 3
# Only these literal errors may cross the public boundary. A model can raise
# WorkerProtocolError too; its arbitrary exception text is never a public code.
ERROR_CODES = frozenset({
    "IPC_DUPLICATE_KEY", "IPC_DEPTH_EXCEEDED", "IPC_NONFINITE", "IPC_BAD_MAGIC",
    "IPC_LENGTH_EXCEEDED", "IPC_FRAME_TYPE", "IPC_JSON_INVALID", "IPC_TRUNCATED_HEADER",
    "IPC_LENGTH_MISMATCH", "IPC_METADATA_NOT_OBJECT", "IPC_EOF", "IPC_REQUEST_ID_INVALID",
    "IPC_RESPONSE_IDENTITY_MISMATCH", "IPC_RESPONSE_FIELDS_INVALID", "IPC_RESULT_NOT_OBJECT",
    "WORKER_TIMEOUT", "WORKER_LIFETIME_EXCEEDED", "WORKER_TRANSPORT_CLOSED", "WORKER_EOF",
    "WORKER_TRANSPORT_FAILED", "WORKER_CLIENT_CLOSED", "WORKER_EXECUTION_ERROR",
    "PUBLIC_REQUEST_IDENTITY_INVALID", "PUBLIC_REQUEST_FIELDS_INVALID", "PUBLIC_UNEXPECTED_PAYLOAD",
    "PUBLIC_HELLO_INVALID", "PUBLIC_HELLO_REQUIRED", "ORACLE_NOT_PUBLIC", "PUBLIC_SPEC_INVALID",
    "PUBLIC_QUERY_BOUNDS_INVALID", "PUBLIC_AXIS_INVALID", "PUBLIC_WORLD_SHAPE_INVALID",
    "PUBLIC_INITIAL_IMAGE_OR_SEED_INVALID", "PUBLIC_QUERY_SHAPE_INVALID", "PUBLIC_QUERY_RANGE_INVALID",
    "PUBLIC_SNAPSHOT_LIMIT_OR_STATE_INVALID", "PUBLIC_SNAPSHOT_UNKNOWN",
    "PUBLIC_INITIAL_IMAGE_INVALID", "PUBLIC_SEED_INVALID", "PUBLIC_RESPONSE_IMAGES_INVALID",
    "PUBLIC_SNAPSHOT_TOKEN_INVALID", "PUBLIC_COUNTER_PAYLOAD_INVALID", "PUBLIC_RESOURCE_RECEIPT_INVALID",
    "REMOTE_STATE_MEASURED_ONLY_BY_WORKER",
})


class WorkerProtocolError(RuntimeError):
    """Only stable codes are exposed; never child exception text."""


def public_error_code(error):
    code = str(error)
    return code if code in ERROR_CODES else "WORKER_EXECUTION_ERROR"


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkerProtocolError("IPC_DUPLICATE_KEY")
        result[key] = value
    return result


def _finite(value, depth=0):
    if depth > 32:
        raise WorkerProtocolError("IPC_DEPTH_EXCEEDED")
    if isinstance(value, float) and not math.isfinite(value):
        raise WorkerProtocolError("IPC_NONFINITE")
    if isinstance(value, dict):
        for item in value.values():
            _finite(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _finite(item, depth + 1)


def _limits(header):
    magic, metadata, payload = HEADER.unpack(header)
    if magic != MAGIC:
        raise WorkerProtocolError("IPC_BAD_MAGIC")
    if not 0 < metadata <= MAX_METADATA or payload > MAX_PAYLOAD:
        raise WorkerProtocolError("IPC_LENGTH_EXCEEDED")
    return metadata, payload


def encode_frame(message, payload=b""):
    if not isinstance(message, dict) or not isinstance(payload, bytes):
        raise WorkerProtocolError("IPC_FRAME_TYPE")
    _finite(message)
    try:
        metadata = json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise WorkerProtocolError("IPC_JSON_INVALID") from exc
    header = HEADER.pack(MAGIC, len(metadata), len(payload))
    _limits(header)
    return header + metadata + payload


def decode_frame(blob):
    if len(blob) < HEADER.size:
        raise WorkerProtocolError("IPC_TRUNCATED_HEADER")
    metadata, payload = _limits(blob[:HEADER.size])
    if len(blob) != HEADER.size + metadata + payload:
        raise WorkerProtocolError("IPC_LENGTH_MISMATCH")
    try:
        message = json.loads(blob[HEADER.size:HEADER.size + metadata], object_pairs_hook=_object,
                             parse_constant=lambda _: (_ for _ in ()).throw(WorkerProtocolError("IPC_NONFINITE")))
    except WorkerProtocolError:
        raise
    except (ValueError, TypeError, RecursionError, UnicodeDecodeError) as exc:
        raise WorkerProtocolError("IPC_JSON_INVALID") from exc
    if not isinstance(message, dict):
        raise WorkerProtocolError("IPC_METADATA_NOT_OBJECT")
    _finite(message)
    return message, blob[HEADER.size + metadata:]


def _read_exact(stream, count):
    result = bytearray()
    while len(result) < count:
        block = stream.read(count - len(result))
        if not block:
            raise WorkerProtocolError("IPC_EOF")
        result.extend(block)
    return bytes(result)


def read_frame(stream):
    header = _read_exact(stream, HEADER.size)
    metadata, payload = _limits(header)
    return decode_frame(header + _read_exact(stream, metadata + payload))


def write_frame(stream, message, payload=b""):
    stream.write(encode_frame(message, payload))
    stream.flush()


class PipeTransport:
    """Deadline-bounded pipes and process group cleanup; no isolation claim."""
    def __init__(self, command, timeout=10., env=None, cwd=None, max_stderr=4096, lifetime_timeout=None):
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Positive finite worker timeout required")
        self.timeout, self.max_stderr = float(timeout), int(max_stderr)
        if lifetime_timeout is not None and (type(lifetime_timeout) not in (int, float) or
                not math.isfinite(lifetime_timeout) or lifetime_timeout <= 0):
            raise ValueError("Positive finite worker lifetime required")
        self._lifetime_deadline = None if lifetime_timeout is None else time.monotonic() + lifetime_timeout
        self._lifetime_expired = False
        self._close_lock = threading.Lock()
        self._lifetime_timer = None
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, env=env, cwd=cwd,
                                        close_fds=True, start_new_session=True, bufsize=0)
        self.stderr = bytearray()
        self._closed = False
        self.external_process_observation = None
        self._lock = threading.Lock()
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self._drainer = threading.Thread(target=self._drain_stderr, daemon=True)
        self._drainer.start()
        if lifetime_timeout is not None:
            self._lifetime_timer = threading.Timer(lifetime_timeout, self._expire_lifetime)
            self._lifetime_timer.daemon = True
            self._lifetime_timer.start()

    def _expire_lifetime(self):
        self._lifetime_expired = True
        self.close()

    def _drain_stderr(self):
        try:
            while True:
                block = self.process.stderr.read(4096)
                if not block:
                    return
                remaining = self.max_stderr - len(self.stderr)
                if remaining > 0:
                    self.stderr.extend(block[:remaining])
        except (OSError, ValueError):
            pass

    def _ready(self, fd, writing, deadline):
        if self._lifetime_deadline is not None:
            deadline = min(deadline, self._lifetime_deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WorkerProtocolError("WORKER_LIFETIME_EXCEEDED" if self._lifetime_deadline is not None and
                                      time.monotonic() >= self._lifetime_deadline else "WORKER_TIMEOUT")
        readable, writable, _ = select.select([] if writing else [fd], [fd] if writing else [], [], remaining)
        if not (writable if writing else readable):
            raise WorkerProtocolError("WORKER_LIFETIME_EXCEEDED" if self._lifetime_deadline is not None and
                                      time.monotonic() >= self._lifetime_deadline else "WORKER_TIMEOUT")

    def _receive(self, count, deadline):
        result = bytearray()
        fd = self.process.stdout.fileno()
        while len(result) < count:
            self._ready(fd, False, deadline)
            try:
                block = os.read(fd, count - len(result))
            except BlockingIOError:
                continue
            if not block:
                raise WorkerProtocolError("WORKER_EOF")
            result.extend(block)
        return bytes(result)

    def exchange(self, message, payload=b""):
        with self._lock:
            try:
                if self._closed:
                    raise WorkerProtocolError("WORKER_LIFETIME_EXCEEDED" if self._lifetime_expired else "WORKER_TRANSPORT_CLOSED")
                request_id = message.get("request_id")
                if type(request_id) is not int or not 0 < request_id < 2 ** 31:
                    raise WorkerProtocolError("IPC_REQUEST_ID_INVALID")
                deadline = time.monotonic() + self.timeout
                encoded = encode_frame(message, payload)
                offset, fd = 0, self.process.stdin.fileno()
                while offset < len(encoded):
                    self._ready(fd, True, deadline)
                    try:
                        offset += os.write(fd, encoded[offset:offset + 65536])
                    except BlockingIOError:
                        continue
                header = self._receive(HEADER.size, deadline)
                metadata, length = _limits(header)
                response, body = decode_frame(header + self._receive(metadata + length, deadline))
                if response.get("version") != VERSION or type(response.get("request_id")) is not int or response["request_id"] != request_id:
                    raise WorkerProtocolError("IPC_RESPONSE_IDENTITY_MISMATCH")
                return response, body
            except (OSError, ValueError, WorkerProtocolError) as exc:
                self.close()
                if self._lifetime_expired:
                    raise WorkerProtocolError("WORKER_LIFETIME_EXCEEDED") from exc
                if isinstance(exc, WorkerProtocolError):
                    raise
                raise WorkerProtocolError("WORKER_TRANSPORT_FAILED") from exc
            except BaseException:
                self.close()
                raise

    def close(self):
        with self._close_lock:
            self._close_locked()

    def _close_locked(self):
        if self._closed:
            return
        self._closed = True
        if self._lifetime_timer is not None:
            self._lifetime_timer.cancel()
        if self.process.returncode is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                deadline = time.monotonic() + .25
                while True:
                    pid, status, usage = os.wait4(self.process.pid, os.WNOHANG)
                    if pid:
                        break
                    if time.monotonic() >= deadline:
                        os.killpg(self.process.pid, signal.SIGKILL)
                        pid, status, usage = os.wait4(self.process.pid, 0)
                        break
                    time.sleep(.005)
                self.process.returncode = os.waitstatus_to_exitcode(status)
                self.external_process_observation = {
                    "source": "evaluator os.wait4", "cpu_user_seconds": usage.ru_utime,
                    "cpu_system_seconds": usage.ru_stime,
                    "process_peak_rss_bytes": int(usage.ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
                    "scope": "reaped direct worker/launcher process; CPU may include descendants it reaped, not unobserved descendant work; setup and IPC included; Linux RSS high water is not summed allocation"}
            except (OSError, ChildProcessError):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if self.process.returncode is None:
                    self.process.wait(timeout=2.)
        # Kill inherited-group descendants even if the immediate worker exited.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        self._drainer.join(timeout=.25)
