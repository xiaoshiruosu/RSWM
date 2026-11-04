"""Explicit Linux bubblewrap profile. An unavailable boundary never falls back."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile


class IsolationBlocked(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(report.get("reason_code", "ISOLATION_BLOCKED"))


@dataclass(frozen=True)
class WorkerLimits:
    address_space_bytes: int = 4 * 1024 ** 3
    cpu_seconds: int = 60
    file_bytes: int = 64 * 1024 ** 2
    open_files: int = 128
    processes: int = 128
    timeout_seconds: float = 30.
    lifetime_seconds: float = 300.
    def validate(self):
        for name in ("address_space_bytes", "cpu_seconds", "file_bytes", "open_files", "processes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError("Positive worker resource limits required")
        for value in (self.timeout_seconds, self.lifetime_seconds):
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 86400:
                raise ValueError("Finite worker wall timeout and lifetime required")
        return self


def public_environment(package_root=None):
    """Literal allowlist; inherited credentials, proxies and future paths disappear."""
    result = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "TMPDIR": "/tmp",
              "LANG": "C.UTF-8", "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1",
              "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
              "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    if package_root is not None:
        result["PYTHONPATH"] = str(package_root)
    return result


def _bwrap_base():
    executable = shutil.which("bwrap")
    if not executable:
        raise IsolationBlocked({"status": "BLOCKED", "reason_code": "BWRAP_NOT_INSTALLED",
                                "protocol_security_verified": False})
    command = [executable, "--unshare-all", "--uid", "65534", "--gid", "65534", "--cap-drop", "ALL",
               "--clearenv", "--new-session", "--die-with-parent", "--hostname", "rswm-worker"]
    mounted = set()
    for path in ("/usr", "/lib", "/lib64", sys.base_prefix, sys.prefix):
        source = Path(path)
        if source.exists() and str(source) not in mounted:
            command.extend(["--ro-bind", str(source), str(source)])
            mounted.add(str(source))
    if Path("/etc/ld.so.cache").is_file():
        command.extend(["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"])
    command.extend(["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/work", "--chdir", "/work"])
    return command


def _run_probe(command, environment, timeout):
    """Own a separate process group so denied/hanging probes leave no child."""
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, close_fds=True, start_new_session=True)
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate(timeout=2.)
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from exc
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.returncode is None:
            process.wait(timeout=2.)


def probe_isolation(timeout=5.):
    """Actually execute the intended namespaces, UID and mounts before using them."""
    command = None
    try:
        command = _bwrap_base() + ["--", "/usr/bin/true"]
        result = _run_probe(command, public_environment(), timeout)
    except IsolationBlocked as exc:
        return exc.report
    except (OSError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", None) or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        return {"schema": "rswm-isolation-capability-v1", "status": "BLOCKED", "reason_code": "ISOLATION_PREFLIGHT_FAILED",
                "error_type": type(exc).__name__, "command": command, "stderr": stderr[:2048],
                "timeout_seconds": timeout, "profile": "linux-bwrap-cpu-v1", "protocol_security_verified": False}
    return {"schema": "rswm-isolation-capability-v1", "status": "AVAILABLE" if result.returncode == 0 else "BLOCKED",
            "reason_code": None if result.returncode == 0 else "ISOLATION_NAMESPACE_OR_MOUNT_DENIED",
            "returncode": result.returncode, "stderr": result.stderr[:2048],
            "command": command, "profile": "linux-bwrap-cpu-v1", "protocol_security_verified": False,
            "scope": "namespace creation preflight only; permission/IPC/lifecycle gates remain separate"}


def launch_description(config, profile="isolated_worker", limits=None):
    from .registry import validate_config
    if not isinstance(config, dict):
        raise ValueError("Public model configuration must be an object")
    # A valid cooperative configuration is not a validated sandbox profile.
    # In particular numeric-camera embeds original source/weight directories;
    # the present CPU mount contract has never audited or mapped those paths.
    if profile == "isolated_worker":
        kind = config.get("adapter") if isinstance(config, dict) else None
        if kind == "numeric-camera":
            raise IsolationBlocked({"status": "BLOCKED", "reason_code": "NUMERIC_BACKEND_ISOLATION_NOT_VALIDATED",
                                    "protocol_security_verified": False, "profile": "linux-bwrap-cpu-v1"})
        if config.get("device", "cpu") != "cpu":
            raise IsolationBlocked({"status": "BLOCKED", "reason_code": "GPU_ISOLATION_PROFILE_NOT_VALIDATED",
                                    "protocol_security_verified": False, "profile": "linux-bwrap-cpu-v1"})
        diagnostic = kind in {"copy", "random", "noise-map"} or (
            kind in {"lazy-map", "eager-map", "raster", "frontier", "lazy"} and config.get("diagnostic_noise") is True)
        if kind != "oracle" and not diagnostic:
            raise IsolationBlocked({"status": "BLOCKED", "reason_code": "REAL_MODEL_ISOLATION_PROFILE_NOT_VALIDATED",
                                    "protocol_security_verified": False, "profile": "linux-bwrap-cpu-v1"})
    validate_config(config)
    if config["adapter"] == "oracle":
        raise ValueError("Oracle is evaluator-only and cannot enter a public worker")
    if profile not in {"isolated_worker", "cooperative_subprocess"}:
        raise ValueError("Unknown worker execution profile")
    limits = (limits or WorkerLimits()).validate()
    limiter = shutil.which("prlimit")
    if not limiter:
        raise IsolationBlocked({"status": "BLOCKED", "reason_code": "PRLIMIT_NOT_INSTALLED",
                                "protocol_security_verified": False})
    prefix = [limiter, f"--as={limits.address_space_bytes}", f"--cpu={limits.cpu_seconds}",
              f"--fsize={limits.file_bytes}", f"--nofile={limits.open_files}", f"--nproc={limits.processes}",
              "--core=0", "--"]
    package_root = Path(__file__).resolve().parent.parent
    public = dict(config)
    if profile == "cooperative_subprocess":
        return {"command": prefix + [sys.executable, "-m", "rswm_inria.worker_public"],
                "environment": public_environment(package_root), "config": public, "profile": profile,
                "protocol_security_verified": False, "resource_scope": "actual worker process, not IPC client"}
    capability = probe_isolation()
    if capability["status"] != "AVAILABLE":
        raise IsolationBlocked(capability)
    command = _bwrap_base() + ["--ro-bind", str(package_root), "/opt/rswm-public"]
    if config.get("model_dir"):
        weights = Path(config["model_dir"])
        if weights.is_symlink() or any(p.is_symlink() for p in weights.parents) or any(p.is_symlink() for p in weights.rglob("*")):
            raise ValueError("Worker weights must not contain or follow symlinks")
        command.extend(["--ro-bind", str(weights.resolve()), "/weights"])
        public["model_dir"] = "/weights"
    environment = public_environment("/opt/rswm-public")
    for name, value in environment.items():
        command.extend(["--setenv", name, value])
    command.extend(["--", sys.executable, "-m", "rswm_inria.worker_public"])
    return {"command": prefix + command, "environment": public_environment(), "config": public,
            "profile": profile, "capability": capability, "protocol_security_verified": False,
            "resource_scope": "actual restricted worker process; CPU-only profile; requires deployment permission gate"}


def permission_probe():
    """Dedicated harmless sentinels: actual OS boundary or explicit BLOCKED."""
    capability = probe_isolation()
    names = ("nonprivileged_uid", "hidden_file", "future_file", "submitted_png", "private_environment", "symlink_escape", "network", "readonly_weights", "weights_readable", "submission_write_denied", "host_proc_escape", "resource_limits")
    if capability["status"] != "AVAILABLE":
        return {**capability, "checks": {name: "NOT_RUN" for name in names}, "acceptance_status": "BLOCKED"}
    with tempfile.TemporaryDirectory(prefix="rswm-permission-probe-") as directory:
        root = Path(directory)
        for name in ("hidden", "future", "submitted.png"):
            (root / name).write_text("SYNTHETIC_PERMISSION_SENTINEL")
        (root / "weights").mkdir()
        (root / "weights/weight.bin").write_text("SYNTHETIC_WEIGHT")
        server = socket.socket()
        try:
            server.bind(("127.0.0.1", 0)); server.listen(1)
            port = server.getsockname()[1]
            # Establish that the same endpoint is reachable from the host first.
            with socket.create_connection(("127.0.0.1", port), timeout=1.):
                pass
            connection, _ = server.accept(); connection.close()
            script = """import json,os,pathlib,resource,socket,sys
paths=json.loads(sys.argv[1]);result={'nonprivileged_uid':os.getuid()!=0,'private_environment':'PRIVATE_ENV_SENTINEL' not in os.environ}
for name,path in paths.items():
 try:pathlib.Path(path).read_bytes();result[name]=False
 except (OSError,ValueError):result[name]=True
try:pathlib.Path(paths['submitted_png']).write_bytes(b'attack');result['submission_write_denied']=False
except OSError:result['submission_write_denied']=True
try:pathlib.Path('/proc/1/root'+paths['hidden_file']).read_bytes();result['host_proc_escape']=False
except OSError:result['host_proc_escape']=True
try:pathlib.Path('/weights/weight.bin').write_bytes(b'attack');result['readonly_weights']=False
except OSError:result['readonly_weights']=True
try:result['weights_readable']=pathlib.Path('/weights/weight.bin').read_bytes()==b'SYNTHETIC_WEIGHT'
except OSError:result['weights_readable']=False
limits=json.loads(sys.argv[3]);result['resource_limits']=all(list(resource.getrlimit(getattr(resource,key)))==value for key,value in limits.items())
pathlib.Path('/work/escape').symlink_to(paths['hidden_file'])
try:pathlib.Path('/work/escape').read_bytes();result['symlink_escape']=False
except OSError:result['symlink_escape']=True
try:socket.create_connection(('127.0.0.1',int(sys.argv[2])),timeout=.5);result['network']=False
except OSError:result['network']=True
print(json.dumps(result))
"""
            paths = {"hidden_file": str(root / "hidden"), "future_file": str(root / "future"), "submitted_png": str(root / "submitted.png")}
            limiter = shutil.which("prlimit")
            if not limiter:
                return {"status": "BLOCKED", "reason_code": "PRLIMIT_NOT_INSTALLED", "checks": {name: "NOT_RUN" for name in names}, "protocol_security_verified": False}
            limits = WorkerLimits().validate()
            expected = {"RLIMIT_AS": [limits.address_space_bytes] * 2,
                        "RLIMIT_CPU": [limits.cpu_seconds] * 2,
                        "RLIMIT_FSIZE": [limits.file_bytes] * 2,
                        "RLIMIT_NOFILE": [limits.open_files] * 2,
                        "RLIMIT_NPROC": [limits.processes] * 2, "RLIMIT_CORE": [0, 0]}
            prefix = [limiter, f"--as={limits.address_space_bytes}", f"--cpu={limits.cpu_seconds}",
                      f"--fsize={limits.file_bytes}", f"--nofile={limits.open_files}", f"--nproc={limits.processes}", "--core=0", "--"]
            command = prefix + _bwrap_base() + ["--ro-bind", str(root / "weights"), "/weights", "--", sys.executable, "-c", script, json.dumps(paths), str(port), json.dumps(expected)]
            environment = public_environment(); environment["PRIVATE_ENV_SENTINEL"] = "SYNTHETIC_ONLY"
            try:
                completed = _run_probe(command, environment, 5.)
            except (OSError, subprocess.TimeoutExpired) as exc:
                return {"status": "BLOCKED", "reason_code": "PERMISSION_PROBE_EXECUTION_FAILED", "error_type": type(exc).__name__,
                        "checks": {name: "NOT_RUN" for name in names}, "protocol_security_verified": False}
            if completed.returncode != 0:
                return {"status": "BLOCKED", "reason_code": "PERMISSION_PROBE_EXECUTION_FAILED", "returncode": completed.returncode,
                        "stderr": completed.stderr[:2048], "checks": {name: "NOT_RUN" for name in names}, "protocol_security_verified": False}
            try:
                checks = json.loads(completed.stdout)
            except (ValueError, UnicodeDecodeError):
                return {"status": "FAIL", "reason_code": "PERMISSION_PROBE_RESPONSE_INVALID", "checks": {name: False for name in names}, "protocol_security_verified": False}
            if not isinstance(checks, dict) or set(checks) != set(names):
                return {"status": "FAIL", "reason_code": "PERMISSION_PROBE_RESPONSE_INVALID", "checks": {name: False for name in names}, "protocol_security_verified": False}
            return {"status": "PASS" if all(checks.get(name) is True for name in names) else "FAIL", "checks": checks,
                    "profile": "linux-bwrap-cpu-v1", "protocol_security_verified": False, "acceptance_status": "OS_PERMISSION_GATE_ONLY",
                    "command": command, "environment_keys": sorted(public_environment()), "returncode": completed.returncode,
                    "scope": "OS permission sentinels only; combined IPC/lifecycle/branch deployment validation required"}
        finally:
            server.close()
