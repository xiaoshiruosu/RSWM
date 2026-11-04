"""Closed-stage Linux server driver for retained RSWM research execution.

Project JSON describes RSWM inputs, never arbitrary commands. Existing producers
remain responsible for data, model, experiment and metric contracts. This driver
owns only process groups, finite cumulative stage budgets and resumable evidence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import importlib.resources
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from .data import verify_manifest
from .geometry import canonical, digest, file_hash
from .hf_materialize import PILOT_DEV2_SOURCES
from .registry import validate_config
from .server_inputs import _safe_path
from .suite import _json, _save, code_fingerprint


SCHEMA = "rswm-server-project-v1"
STAGES = ("pilot-data", "prepare-models", "pilot", "core-data", "training-source", "training-inputs",
          "train", "resume", "export", "trained-pilot", "formal", "metrics", "motion-calibration",
          "motion-validation", "feature-calibration", "feature-validation", "paired", "lr64")
WORKFLOW = ["pilot-data", "prepare-models", "pilot", "core-data", "training-source", "training-inputs",
            "train", "export", "trained-pilot", "formal", "metrics"]
GPU_STAGES = {"pilot", "train", "resume", "export", "trained-pilot", "formal", "paired", "lr64"}
REUSABLE = {"pilot", "formal", "paired", "lr64"}
GROUPS = {"pilot-data": ["data"], "core-data": ["data"], "prepare-models": ["models"],
          "pilot": ["formal"], "training-source": ["models", "training"], "training-inputs": ["training"],
          "train": [], "resume": ["training"], "export": ["training"], "trained-pilot": ["models"],
          "formal": ["formal"], "metrics": ["measurement"], "motion-calibration": ["measurement"],
          "motion-validation": ["measurement"], "feature-calibration": ["measurement"],
          "feature-validation": ["measurement"], "paired": ["independent"], "lr64": ["independent"]}


class StageFailure(RuntimeError):
    pass


def _default_source_commit(source_root):
    source = Path(source_root)
    try:
        top = subprocess.run(["git", "-C", str(source), "rev-parse", "--show-toplevel"], capture_output=True, check=True, text=True).stdout.strip()
        if Path(top).resolve() == source.resolve():
            head = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, check=True, text=True).stdout.strip()
            if re.fullmatch(r"[0-9a-f]{40}", head):
                dirty = bool(subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--", "src", "README.md", "docs", "scripts", "pyproject.toml", "setup.py", "MANIFEST.in"],
                                           capture_output=True, check=True, text=True).stdout.strip())
                return head, "git_head_plus_dirty_snapshot" if dirty else "git_head_clean", dirty
    except (OSError, subprocess.SubprocessError): pass
    for info_path in (source / "src/rswm_inria/_build_info.json", Path(__file__).parent / "_build_info.json"):
        if info_path.is_file():
            info = _json(info_path); commit = info.get("source_commit")
            if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit):
                return commit, "base_commit_plus_dirty_snapshot", True
    return None, "unknown_source_commit", None


def project_template(source_root, run_root, *, derive_source=True):
    commit, mode, dirty = _default_source_commit(source_root) if derive_source else (None, "unknown_source_commit", None)
    return {"schema": SCHEMA, "source_root": str(Path(source_root).resolve()), "run_root": str(Path(run_root).resolve()),
            "code_commit": commit, "source_identity_mode": mode, "source_dirty": dirty, "device": "cuda:0", "workflow": WORKFLOW,
            "data": {"pilot_source_dir": None, "core_source_dir": None, "existing_pilot_build": None, "existing_core_build": None,
                     "pilot_download": False, "core_download": False, "pilot_max_source_bytes": None, "core_max_source_bytes": None},
            "models": {"numeric_weights_dir": None, "video_weights_dir": None, "completer_weights_dir": None,
                       "runtime": "owner/wan-runtime.json", "camera": "owner/camera.json",
                       "decisions": "owner/wan-use-decision.json", "trained_decisions": "owner/trained-use-decision.json",
                       "video_steps": None, "completer_steps": None, "video_prompt": None, "completer_prompt": None,
                       "model_ids": {"numeric": "wan-numeric-reference", "video": "wan-without-numeric",
                                     "frontier": "completion-frontier", "lazy": "completion-lazy", "trained": "wan-trained-reference"}},
            "rights": {"data_use_approved": False, "weight_use_approved": False, "decision_reference": None},
            "training": {"spec": "owner/training-spec.json", "checkpoint": None, "runtime": "owner/wan-training-runtime.json",
                         "inference_runtime": "owner/wan-runtime.json", "validation_world_ids": None},
            "formal": {"include_trained": True, "seeds": None, "max_attempts_per_run": 1,
                       "motion_validation": None, "quality_validation": None, "instrument_config_sha256": {"motion": None, "quality": None},
                       "smoke12_evidence_root": None},
            "measurement": {"motion_config": None, "motion_validation": None, "feature_config": None, "feature_validation": None,
                            "patch_count": None, "cap": None, "bootstrap": None, "seed": None,
                            "motion_candidates": None, "motion_calibration": None, "motion_exposure_ledger": None,
                            "pairs_per_world": None, "pair_seed": None, "feature_inputs": None},
            "independent": {"models": None, "paired_task_index": None, "paired_world_index": None, "paired_world_dir": None,
                            "mechanisms": None, "lr64_materialized": None, "seeds": None, "bootstrap": None},
            "budgets": {stage: None for stage in STAGES}}


def init_project(out, source_root):
    root = _safe_path(out); source = _safe_path(source_root, directory=True)
    root.mkdir(parents=True, exist_ok=False)
    templates = importlib.resources.files("rswm_inria").joinpath("configs/examples")
    for name in ("wan-runtime.json", "wan-training-runtime.json", "camera.json", "wan-use-decision.json", "trained-use-decision.json", "training-spec.json"):
        _save(root / "owner" / name, json.loads(templates.joinpath(name).read_text(encoding="utf-8")), exclusive=True)
    _save(root / "project.json", project_template(source, root / "runs"), exclusive=True)
    return root / "project.json"


def _project(path):
    location = _safe_path(path, file=True); value = _json(location)
    expected = project_template(location.parent, location.parent / "runs", derive_source=False)
    if not isinstance(value, dict) or set(value) != set(expected) or value.get("schema") != SCHEMA:
        raise ValueError("Exact rswm-server-project-v1 fields are required; arbitrary jobs are unsupported")
    for key in ("data", "models", "rights", "training", "formal", "measurement", "independent", "budgets"):
        if not isinstance(value[key], dict) or set(value[key]) != set(expected[key]):
            raise ValueError("Project fields differ in " + key)
    if not isinstance(value["workflow"], list) or not value["workflow"] or len(set(value["workflow"])) != len(value["workflow"]) or any(s not in STAGES for s in value["workflow"]):
        raise ValueError("Workflow must contain unique known RSWM stages")
    for key in ("source_root", "run_root"):
        value[key] = str(_safe_path(value[key], base=location.parent))
    source, run = Path(value["source_root"]), Path(value["run_root"])
    if source == run or source.is_relative_to(run) or any(run.is_relative_to(source / name) for name in ("src", "vendor", "scripts", "docs", "tests")):
        raise ValueError("run_root must not mutate or contain the retained source/implementation trees")
    if not isinstance(value["device"], str) or re.fullmatch(r"cuda(?::[0-9]+)?", value["device"]) is None:
        raise ValueError("Server generation/training device must be NVIDIA CUDA")
    if any(type(value["data"][key]) is not bool for key in ("pilot_download", "core_download")) or type(value["formal"]["include_trained"]) is not bool:
        raise ValueError("Download/include_trained must be explicit booleans")
    return location, value


def _path(project_path, value, *, file=False, directory=False):
    return _safe_path(value, base=project_path.parent, file=file, directory=directory)


def _frozen():
    root = Path(__file__).parent / "configs/frozen"
    manifest, inventory = root / "manifest.json", root / "source-inventory.json"
    verify_manifest(_json(manifest), _json(inventory))
    return manifest, inventory


def _dependencies(stage, p):
    base = {"pilot-data": [], "prepare-models": [], "pilot": ["pilot-data", "prepare-models"], "core-data": [],
            "training-source": ["core-data"], "training-inputs": ["core-data", "training-source"],
            "train": ["training-inputs"], "resume": ["training-inputs"], "export": ["training-inputs"],
            "trained-pilot": ["pilot-data", "export"], "formal": ["core-data", "pilot"], "metrics": ["formal"],
            "motion-calibration": ["core-data"], "motion-validation": ["core-data"],
            "feature-calibration": ["core-data"], "feature-validation": ["core-data"], "paired": [], "lr64": []}
    if stage == "formal" and p["formal"]["include_trained"]:
        base[stage].append("trained-pilot")
    return base[stage]


def _order(stage, p):
    selected = p["workflow"] if stage == "all" else [stage]
    out = []
    def visit(name):
        for parent in _dependencies(name, p): visit(parent)
        if name not in out: out.append(name)
    for name in selected:
        if name not in STAGES: raise ValueError("Unknown RSWM server stage")
        visit(name)
    return out


def _budget(p, stage):
    value = p["budgets"][stage]
    if not isinstance(value, dict) or set(value) != {"max_wall_seconds", "max_output_bytes", "max_attempts"}:
        raise ValueError("Fill this stage's finite wall/output/attempt budget before run")
    if (type(value["max_wall_seconds"]) not in (int, float) or not math.isfinite(value["max_wall_seconds"]) or value["max_wall_seconds"] <= 0
            or type(value["max_output_bytes"]) is not int or value["max_output_bytes"] <= 0
            or type(value["max_attempts"]) is not int or value["max_attempts"] <= 0):
        raise ValueError("Stage budgets must be positive finite explicit values")
    return value


def _rights(p, stage):
    rights = p["rights"]
    if rights["data_use_approved"] is not True or not isinstance(rights["decision_reference"], str) or not rights["decision_reference"].strip():
        raise ValueError("Actual data-use decision and reference are required")
    if stage not in {"pilot-data", "core-data", "motion-calibration", "motion-validation"} and rights["weight_use_approved"] is not True:
        raise ValueError("Actual weight-use decision is required")


def _source_identity(p):
    root = _safe_path(p["source_root"], directory=True)
    if not (root / "scripts/prepare_legacy_wan.py").is_file():
        raise ValueError("source_root must be the complete source bundle; a wheel alone lacks deployment scripts")
    package = root / "src/rswm_inria"
    identity = digest({path.relative_to(package).as_posix(): file_hash(path) for path in sorted(package.rglob("*.py"))})
    if identity != code_fingerprint():
        raise ValueError("Installed driver/source_root Python fingerprint mismatch")
    if not isinstance(p["code_commit"], str) or re.fullmatch(r"[0-9a-f]{40}", p["code_commit"]) is None:
        raise ValueError("Record the verified full source commit in project.json")
    return identity


def _input_hashes(location, value, *, prefix=""):
    """Freeze referenced JSON files; mutable exposure ledgers retain their paths."""
    hashes = {}
    if isinstance(value, dict):
        for key, member in value.items(): hashes.update(_input_hashes(location, member, prefix=prefix + key + "."))
    elif isinstance(value, list):
        for i, member in enumerate(value): hashes.update(_input_hashes(location, member, prefix=prefix + str(i) + "."))
    elif isinstance(value, str) and value.endswith(".json") and "exposure_ledger" not in prefix:
        candidate = _path(location, value)
        if candidate.is_file(): hashes[prefix.rstrip(".")] = {"path": str(candidate), "sha256": file_hash(candidate)}
    return hashes


def _stage_identity(location, p, stage):
    chosen = {name: copy.deepcopy(p[name]) for name in GROUPS[stage]}
    if stage in {"pilot-data", "core-data"}:
        profile = "pilot" if stage == "pilot-data" else "core"
        chosen["data"] = {key: p["data"][key] for key in (profile + "_source_dir", "existing_" + profile + "_build",
                                                             profile + "_download", profile + "_max_source_bytes")}
    # A later explicit checkpoint/use decision must not invalidate completed
    # data preparation or original pilot stages.
    if stage in {"training-source", "training-inputs"}:
        chosen["training"].pop("checkpoint", None); chosen["training"].pop("inference_runtime", None)
    if stage == "prepare-models":
        chosen["models"].pop("trained_decisions", None)
        chosen["models"]["model_ids"].pop("trained", None)
    if stage == "training-source":
        chosen["models"] = {key: p["models"][key] for key in ("numeric_weights_dir", "decisions")}
    if stage == "trained-pilot":
        chosen["models"] = {key: p["models"][key] for key in ("numeric_weights_dir", "trained_decisions")}
        chosen["models"]["trained_model_id"] = p["models"]["model_ids"]["trained"]
    if stage == "pilot":
        chosen = {"max_attempts_per_run": p["formal"]["max_attempts_per_run"]}
    parents = {}
    for parent in _dependencies(stage, p):
        marker = _completed(p, parent)
        if marker is None: raise ValueError("Required stage is not completed: " + parent)
        parents[parent] = marker["marker_sha256"]
    manifest, inventory = _frozen()
    value = {"stage": stage, "source_root": p["source_root"], "run_root": p["run_root"],
             "source_code_sha256": _source_identity(p), "code_commit": p["code_commit"], "device": p["device"],
             "source_identity_mode": p["source_identity_mode"], "source_dirty": p["source_dirty"],
             "rights": ({key: p["rights"][key] for key in ("data_use_approved", "decision_reference")}
                        if stage in {"pilot-data", "core-data", "motion-calibration", "motion-validation"} else p["rights"]),
             "budget": _budget(p, stage), "inputs": chosen,
             "input_files": _input_hashes(location, chosen), "parents": parents,
             "manifest_sha256": file_hash(manifest), "inventory_sha256": file_hash(inventory)}
    value["identity_sha256"] = digest(value)
    return value


def _inventory(root):
    root = _safe_path(root, directory=True); files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink(): raise ValueError("Stage artifact symlink")
        if path.is_file(): files[path.relative_to(root).as_posix()] = file_hash(path)
    return files


def _tree_bytes(roots):
    seen, total = set(), 0
    for root in roots:
        if not root.exists(): continue
        for path in root.rglob("*"):
            if path.is_symlink(): raise ValueError("Stage output symlink")
            if path.is_file() and str(path) not in seen:
                try: size = path.stat().st_size
                except FileNotFoundError: continue  # Atomic publication may remove a temporary between observations.
                seen.add(str(path)); total += size
    return total


def _completed(p, stage):
    head = Path(p["run_root"]) / "stages" / stage / "completed.json"
    if not head.exists(): return None
    marker = _json(head)
    if marker.get("marker_sha256") != digest({k: v for k, v in marker.items() if k != "marker_sha256"}):
        raise ValueError("Completed stage marker hash mismatch")
    root = _safe_path(marker["products"], directory=True)
    if _inventory(root) != marker["files"]:
        raise ValueError("Completed stage retained artifacts changed; preserve them and investigate")
    for path, expected in marker.get("external_artifacts", {}).items():
        actual = _inventory(Path(path)) if Path(path).is_dir() else file_hash(path)
        if path in marker.get("append_only_training_roots", []):
            if any(actual.get(name) != sha for name, sha in expected.items()):
                raise ValueError("Completed training contract/checkpoint/snapshot/cost evidence changed")
            extra = set(actual) - set(expected)
            if any(not (name.startswith("attempts/attempt-") or re.fullmatch(r"update-[0-9]{8}\.(?:pt|json)", name)) for name in extra):
                raise ValueError("Unexpected appended files in retained training output")
        elif actual != expected: raise ValueError("Completed stage external retained artifacts changed")
    return marker


def _pid_token(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw.rsplit(")", 1)[1].split()
        return fields[19] if fields[0] not in {"Z", "X"} else None
    except (FileNotFoundError, OSError, IndexError): return None


def _live(record):
    pid, token = record.get("pid"), record.get("pid_start_token")
    return type(pid) is int and token is not None and _pid_token(pid) == token


@contextmanager
def _coordinator(root):
    if os.name != "posix": raise ValueError("rswm-server run requires Linux/POSIX process groups and flock")
    import fcntl
    root = _safe_path(root); root.mkdir(parents=True, exist_ok=True)
    with (root / "server-coordinator.lock").open("a+") as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise ValueError("Another server coordinator is active") from exc
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)


def _prior(stage_root):
    rows, spent = [], 0.
    for folder in sorted((stage_root / "attempts").glob("attempt-*")):
        start = _json(folder / "start.json")
        for path in folder.glob("commands/*/start.json"):
            record = _json(path)
            exit_path = path.parent / "exit.json"
            if record.get("process_group") is not None and (not exit_path.exists() or _json(exit_path).get("live") is True):
                proxy = SimpleNamespace(pid=record["pid"], _rswm_birth_token=record.get("pid_start_token"))
                if _owned_group(proxy):
                    raise ValueError("An earlier owned process group remains live; observe those PIDs instead of restarting")
        if (folder / "finish.json").exists():
            finish = _json(folder / "finish.json"); elapsed = finish["elapsed_seconds"]
        else:
            if _live(start): raise ValueError("An earlier coordinator remains live")
            elapsed = max(0., time.time() - start["started_epoch"])
            finish = {"status": "UNFINISHED_NO_LIVE_HANDLE", "elapsed_seconds": elapsed, "downtime_charged": True}
        spent += elapsed; rows.append({"attempt": folder.name, **finish})
    return rows, spent


def _gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError: return False


def _stop_group(process):
    members = _owned_group(process)
    observed = getattr(process, "_rswm_observed_members", [])
    known = {(row["pid"], row["pid_start_token"]) for row in observed}
    process._rswm_observed_members = observed + [row for row in members if (row["pid"], row["pid_start_token"]) not in known]
    if not members: return process.poll()
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    deadline = time.monotonic() + 5.
    while _owned_group(process) and time.monotonic() < deadline:
        process.poll(); time.sleep(.05)
    if _owned_group(process):
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    if process.poll() is None: process.wait(timeout=5)
    if _owned_group(process): raise StageFailure("Owned process-group members remain after bounded cleanup")
    return process.returncode


def _owned_group(process):
    """Use captured leader birth token plus SID/PGID; never a recycled PID."""
    original = getattr(process, "_rswm_birth_token", None)
    current = _pid_token(process.pid)
    if current is not None and current != original:
        # A different birth token belongs to a new PID lifecycle. Never signal
        # its group; Linux retains the old PGID while old members remain.
        return []
    rows = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal(): continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] in {"Z", "X"} or int(fields[2]) != process.pid or int(fields[3]) != process.pid: continue
            if original is None or int(fields[19]) < int(original):
                raise StageFailure("Owned process-group birth identity could not be confirmed")
            rows.append({"pid": int(path.name), "pid_start_token": fields[19], "sid": int(fields[3]), "pgid": int(fields[2])})
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError): continue
    return rows


def _run_process(argv, folder, *, cwd, budget, roots, prior_seconds, attempt_started):
    """Real argv/PID/exit and bounded own-group termination; no shell."""
    folder.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    started = time.monotonic(); process = None; reason = None
    with (folder / "stdout.log").open("xb") as stdout, (folder / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen([str(x) for x in argv], cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        process._rswm_birth_token = _pid_token(process.pid)
        _save(folder / "start.json", {"argv": [str(x) for x in argv], "pid": process.pid, "pid_start_token": process._rswm_birth_token,
              "started_epoch": time.time(), "process_group": process.pid, "shell": False}, exclusive=True)
        next_size_observation = 0.
        try:
            while process.poll() is None:
                if prior_seconds + time.monotonic() - attempt_started >= budget["max_wall_seconds"]:
                    reason = "WALL_BUDGET_EXHAUSTED"; _stop_group(process); break
                if time.monotonic() >= next_size_observation:
                    next_size_observation = time.monotonic() + 1.
                    if _tree_bytes(roots) >= budget["max_output_bytes"]:
                        reason = "OUTPUT_BUDGET_EXHAUSTED"; _stop_group(process); break
                time.sleep(.1)
            exit_code = process.wait()
        except BaseException:
            reason = "INTERRUPTED"; _stop_group(process)
            raise
        finally:
            cleanup_error, remaining = None, []
            try:
                _stop_group(process); remaining = _owned_group(process)
            except (StageFailure, OSError, subprocess.SubprocessError) as exc:
                cleanup_error = str(exc); reason = "OWNED_GROUP_CLEANUP_FAILED"
            _save(folder / "exit.json", {"pid": process.pid, "exit_code": process.returncode,
                  "live": process.poll() is None or bool(remaining) or cleanup_error is not None,
                  "elapsed_seconds": time.monotonic() - started, "reason": reason,
                  "owned_group_cleanup_error": cleanup_error, "remaining_owned_members": remaining,
                  "owned_members_observed": getattr(process, "_rswm_observed_members", []),
                  "observed_output_bytes": _tree_bytes(roots)}, exclusive=True)
    if reason: raise StageFailure(reason)
    return exit_code


def _module(module, *args): return [sys.executable, "-m", "rswm_inria." + module, *map(str, args)]


def _step(label, argv=None, action=None, outputs=(), exits=(0,), validate=None):
    return {"label": label, "argv": argv, "action": action, "outputs": list(outputs), "exits": list(exits), "validate": validate}


def _write(path, value): _save(path, value, exclusive=True)


def _required_integer(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1): raise ValueError("Explicit integer required: " + name)
    return value


def _seeds(value):
    if not isinstance(value, list) or not value or len(set(value)) != len(value) or any(type(x) is not int or x < 0 for x in value):
        raise ValueError("Explicit distinct nonnegative inference seeds are required")
    return list(map(str, value))


def _parent_products(p, name):
    marker = _completed(p, name)
    if marker is None: raise ValueError("Missing completed dependency: " + name)
    return Path(marker["products"])


def _data_paths(p, name):
    products = _parent_products(p, name)
    build = Path(_json(products / "data-location.json")["build"])
    return build, products / "source-audit.json"


def _train_root(p): return Path(p["run_root"]) / "training/wan"


def _checkpoint(location, p):
    path = _path(location, p["training"]["checkpoint"], file=True)
    if path.parent != _train_root(p): raise ValueError("Explicit checkpoint must retain the original training output directory")
    receipt = _json(path.with_suffix(".json"))
    if receipt.get("schema") != "rswm-training-checkpoint-v2" or receipt.get("training", {}).get("source_code_sha256") != code_fingerprint():
        raise ValueError("Explicit checkpoint must carry the current strong driver receipt")
    from .training import validate_checkpoint_receipt
    validate_checkpoint_receipt(path, receipt["contract_sha256"])
    return path


def _experiment_inputs(p, stage, audit, roles, *, pilots=(), metrics=(), smoke12_evidence_root=None):
    limits = _budget(p, stage)
    value = {"budget": {k: limits[k] for k in ("max_wall_seconds", "max_output_bytes")}, "source_audit": _json(audit),
             "rights": p["rights"], "pilot_qualification_roles": roles}
    if stage == "formal":
        value.update(pilot_receipts=[_json(path) for path in pilots], metric_receipts=[_json(path) for path in metrics],
                     instrument_config_sha256=p["formal"]["instrument_config_sha256"],
                     smoke12_evidence_root=smoke12_evidence_root)
    return value


def _build_steps(location, p, stage, products, *, resume):
    """Compile the fixed RSWM operations; no argv is read from project JSON."""
    source = Path(p["source_root"]); manifest, inventory = _frozen(); models = p["models"]
    steps = []; add = steps.append
    def path(value, **kw): return _path(location, value, **kw)
    def write(label, target, value): add(_step(label, action=lambda: _write(target, value() if callable(value) else value), outputs=[target]))
    if stage in {"pilot-data", "core-data"}:
        profile = "pilot-dev2" if stage == "pilot-data" else "core120"
        key = "pilot" if stage == "pilot-data" else "core"
        download = p["data"][key + "_download"]
        source_dir = path(p["data"][key + "_source_dir"], directory=not download)
        limit = _required_integer(p["data"][key + "_max_source_bytes"], "data." + key + "_max_source_bytes")
        existing = p["data"]["existing_pilot_build" if stage == "pilot-data" else "existing_core_build"]
        build = path(existing, directory=True) if existing else products / "build"
        if existing is None:
            argv = _module("hf_materialize", "inria", "--manifest", manifest, "--inventory", inventory, "--profile", profile,
                           "--source-dir", source_dir, "--out", build, "--max-source-gb", limit / 1_000_000_000)
            if download: argv.append("--download")
            add(_step("materialize", argv, outputs=[build]))
        write("bind-data-location", products / "data-location.json", {"build": str(build), "profile": profile, "reused_existing_build": existing is not None})
        if existing is not None: steps[-1]["outputs"].append(build)
        add(_step("independent-current-audit", _module("materialized", "--build", build, "--manifest", manifest, "--inventory", inventory,
                 "--source-dir", source_dir, "--out", products / "source-audit.json"), outputs=[products / "source-audit.json"]))
        if stage == "pilot-data":
            add(_step("select-fixed-origin0", _module("server_inputs", "pilot-index", "--build", build, "--out", products / "pilot-index.json"), outputs=[products / "pilot-index.json"]))
    elif stage in {"prepare-models", "training-source"}:
        runtime = path(p["training"]["runtime"] if stage == "training-source" else models["runtime"], file=True)
        camera = path(models["camera"], file=True)
        if stage == "training-source":
            spec_path = path(p["training"]["spec"], file=True); spec = _json(spec_path)
            if spec.get("device") != p["device"]: raise ValueError("Training spec and project CUDA device must explicitly match")
            from .legacy_wan import validate_backend_config
            runtime_value = validate_backend_config(_json(runtime))
            updates = _required_integer(spec["updates"], "training updates")
            worlds = p["training"]["validation_world_ids"]
            fixed = {r["world_id"] for r in _json(manifest)["worlds"] if r["role"] == "dev"}
            if not isinstance(worlds, list) or not worlds or len(set(worlds)) != len(worlds) or not set(worlds).issubset(fixed):
                raise ValueError("Training validation must explicitly select from the frozen twelve-world dev population")
            block_cpu = runtime_value["sampling"].get("block_cpu_offload", False)
            training_policy = {"trainable_policy": spec["trainable_policy"], "timestep_sampling": "uniform_integer",
                "expert_residency": "block_cpu" if block_cpu else "resident", "precision": spec["precision"],
                "validation": {"world_ids": worlds, "family": "F2", "origin_index": 0, "seed": spec["seed"], "max_total_forwards": updates * len(worlds)}}
            if runtime_value["training"] is not None and runtime_value["training"] != training_policy:
                raise ValueError("Locked Wan training policy differs from the explicit project/spec selection")
            runtime_value["training"] = training_policy
            validate_backend_config(runtime_value)
            runtime = products / "training-runtime.json"; write("derive-locked-training-runtime", runtime, runtime_value)
            camera = _safe_path(spec["camera"], base=spec_path.parent, file=True)
        vendor = source / "vendor/legacy-original"; retained = source / "vendor/legacy-source-inventory.json"
        _safe_path(vendor, directory=True); _safe_path(retained, file=True)
        argv = [sys.executable, str(source / "scripts/prepare_legacy_wan.py"), "--source-dir", str(vendor), "--source-inventory", str(retained),
                "--backend-config", str(runtime), "--out", str(products / "wan-source")]
        add(_step("prepare-original-safe-source", argv, outputs=[products / "wan-source"]))
        add(_step("derive-numeric-identity", _module("server_inputs", "numeric-model", "--source-dir", products / "wan-source",
            "--weights-dir", path(models["numeric_weights_dir"], directory=True), "--camera", camera,
            "--decisions", path(models["decisions"], file=True), "--model-id", models["model_ids"]["numeric"],
            "--device", p["device"], "--out", products / "numeric-model.json"), outputs=[products / "numeric-model.json"]))
        if stage == "prepare-models":
            video_dir = path(models["video_weights_dir"], directory=True); completer_dir = path(models["completer_weights_dir"], directory=True)
            video_steps = _required_integer(models["video_steps"], "video steps"); completion_steps = _required_integer(models["completer_steps"], "completer steps")
            if any(not isinstance(models[k], str) or not models[k].strip() for k in ("video_prompt", "completer_prompt")):
                raise ValueError("Explicit fixed video and completion prompts are required")
            def compose():
                rows = [_json(products / "numeric-model.json"), {"model_id": models["model_ids"]["video"], "adapter": "wan-i2v", "device": p["device"],
                    "model_dir": str(video_dir), "steps": video_steps, "prompt": models["video_prompt"]}]
                rows.extend({"model_id": models["model_ids"][kind], "adapter": kind, "device": p["device"], "model_dir": str(completer_dir),
                             "steps": completion_steps, "prompt": models["completer_prompt"]} for kind in ("frontier", "lazy"))
                if len({row["model_id"] for row in rows}) != 4: raise ValueError("Four model IDs must be distinct")
                for row in rows: validate_config(row, check_local_assets=False)
                return rows
            write("compose-four-real-systems", products / "models.json", compose)
    elif stage == "training-inputs":
        build, audit = _data_paths(p, "core-data"); src = _parent_products(p, "training-source")
        spec_path = path(p["training"]["spec"], file=True); spec = _json(spec_path)
        spec["camera"] = str(_safe_path(spec["camera"], base=spec_path.parent, file=True))
        spec["backend_manifest"] = str(src / "numeric-model.json")
        write("bind-current-training-source", products / "training-spec.json", spec)
        add(_step("derive-train60-config", _module("server_inputs", "training-config", "--spec", products / "training-spec.json",
                  "--build", build, "--source-audit", audit, "--out", products / "training-config.json"), outputs=[products / "training-config.json"]))
    elif stage in {"train", "resume", "export"}:
        build, audit = _data_paths(p, "core-data"); cfg = _parent_products(p, "training-inputs") / "training-config.json"
        common = ["--config", cfg, "--index", build / "private/index.json", "--clips", build / "private/training-clips.json", "--source-audit", audit]
        if stage != "export":
            argv = _module("training", *common, "--root", build / "private/worlds", "--out", _train_root(p))
            if stage == "resume" or resume:
                argv.extend(["--resume", str(_checkpoint(location, p))])
            add(_step("actual-wan-training" if stage == "train" else "actual-wan-resume", argv, outputs=[_train_root(p)]))
        else:
            add(_step("export-retained-driver-checkpoint", _module("legacy_export", *common, "--checkpoint", _checkpoint(location, p),
                "--world-root", build / "private/worlds", "--inference-config", path(p["training"]["inference_runtime"], file=True),
                "--out", products / "adapter-release", "--deployment-out", products / "inference-source"), outputs=[products / "adapter-release", products / "inference-source"]))
    elif stage in {"pilot", "trained-pilot", "formal"}:
        data_name = "core-data" if stage == "formal" else "pilot-data"; build, audit = _data_paths(p, data_name)
        if stage == "trained-pilot":
            exported = _parent_products(p, "export"); trained_cfg = _parent_products(p, "training-inputs") / "training-config.json"
            camera_file = products / "training-camera.json"; write("retain-training-camera", camera_file, lambda: _json(trained_cfg)["camera"])
            add(_step("fresh-trained-registry-config", _module("server_inputs", "numeric-model", "--source-dir", exported / "inference-source",
                "--weights-dir", path(models["numeric_weights_dir"], directory=True), "--camera", camera_file,
                "--decisions", path(models["trained_decisions"], file=True), "--model-id", models["model_ids"]["trained"],
                "--device", p["device"], "--out", products / "numeric-model.json"), outputs=[products / "numeric-model.json"]))
            write("retain-trained-model-list", products / "models.json", lambda: [_json(products / "numeric-model.json")])
            model_file = products / "models.json"; roles = {models["model_ids"]["trained"]: "reference"}
        else:
            base_models = _parent_products(p, "prepare-models") / "models.json"
            model_file = products / "models.json"
            def model_list():
                value = _json(base_models)
                if stage == "formal" and p["formal"]["include_trained"]:
                    value.extend(_json(_parent_products(p, "trained-pilot") / "models.json"))
                return value
            write("retain-exact-model-matrix", model_file, model_list)
            roles = {mid: "reference" if key in {"numeric", "trained"} else "comparison" for key, mid in models["model_ids"].items()
                     if key != "trained" or (stage == "formal" and p["formal"]["include_trained"])}
        plan = products / "suite-plan.json"; input_file = products / "experiment-inputs.json"; frozen = products / "experiment.json"
        task_index = build / "private/index.json" if stage == "formal" else _parent_products(p, "pilot-data") / "pilot-index.json"
        seeds = _seeds(p["formal"]["seeds"]) if stage == "formal" else ["0"]
        argv = _module("suite", "plan", "--task-index", task_index, "--world-dir", build / "private/worlds", "--models", model_file,
            "--seeds", *seeds, "--code-commit", p["code_commit"], "--max-attempts-per-run", p["formal"]["max_attempts_per_run"], "--out", plan)
        if stage == "formal": argv.extend(["--role", "test_unseen_city"])
        add(_step("freeze-suite", argv, outputs=[plan]))
        pilots, metrics, smoke = [], [], None
        if stage == "formal":
            pilots = [_receipt_path(_parent_products(p, "pilot") / "experiment")]
            if p["formal"]["include_trained"]: pilots.append(_receipt_path(_parent_products(p, "trained-pilot") / "experiment"))
            metrics = [path(p["formal"][name], file=True) for name in ("motion_validation", "quality_validation")]
            smoke = str(path(p["formal"]["smoke12_evidence_root"], directory=True))
        write("inline-actual-experiment-evidence", input_file, lambda: _experiment_inputs(p, stage, audit, roles, pilots=pilots, metrics=metrics, smoke12_evidence_root=smoke))
        add(_step("freeze-real-experiment", _module("experiments", "freeze", "--plan", plan, "--kind", "formal" if stage == "formal" else "pilot",
                  "--purpose", "real", "--inputs", input_file, "--out", frozen), outputs=[frozen]))
        argv = _module("experiments", "run", "--plan", plan, "--experiment", frozen, "--out", products / "experiment")
        if resume and (products / "experiment/budget.json").exists(): argv.append("--resume")
        add(_step("execute-frozen-real-experiment", argv, outputs=[products / "experiment"], exits=(0, 1)))
    elif stage == "metrics":
        formal = _parent_products(p, "formal"); plan = formal / "suite-plan.json"; suite = formal / "experiment/suite"; m = p["measurement"]
        motion = path(m["motion_config"], file=True); motion_validation = path(m["motion_validation"], file=True)
        features = path(m["feature_config"], file=True); quality_validation = path(m["feature_validation"], file=True)
        count, cap, bootstrap = [_required_integer(m[k], k) for k in ("patch_count", "cap", "bootstrap")]; seed = _required_integer(m["seed"], "seed", zero=True)
        add(_step("model-motion-evaluation", [sys.executable, "-c", "from rswm_inria.motion_protocol import validation_main; raise SystemExit(validation_main())",
            "evaluate-suite", "--plan", str(plan), "--suite", str(suite), "--config", str(motion), "--validation", str(motion_validation), "--out", str(products / "motion")], outputs=[products / "motion"], exits=(0, 1)))
        patch = products / "patch-plan.json"
        add(_step("freeze-first-submission-quality-tiles", _module("quality_protocol", "plan", "--suite-plan", plan, "--count", count, "--out", patch), outputs=[patch]))
        add(_step("assemble-reference", _module("quality_protocol", "reference", "--suite-plan", plan, "--patch-plan", patch, "--out", products / "reference"), outputs=[products / "reference"]))
        add(_step("assemble-submitted-first-reference", _module("quality_protocol", "mosaics", "--suite-plan", plan, "--suite", suite,
            "--patch-plan", patch, "--out", products / "generated"), outputs=[products / "generated"], exits=(0, 1)))
        for kind in ("generated", "reference"):
            feature_root = products / (kind + "-features")
            def validate_bundle(target=feature_root / "features.json"):
                from .features import load_feature_bundle
                load_feature_bundle(target)
            add(_step("extract-" + kind + "-features", _module("features", "--config", features, "--mosaics", products / kind / "mosaics.json",
                "--cache", products / "feature-cache", "--out", feature_root), outputs=[feature_root, feature_root / "features.json"],
                exits=(0, 1), validate=validate_bundle))
        source_id = _json(_data_paths(p, "core-data")[1])["manifest_sha256"]
        add(_step("formal-quality-statistics", _module("quality_evaluation", "--generated", products / "generated-features/features.json", "--reference", products / "reference-features/features.json",
            "--out", products / "quality.json", "--cap", cap, "--bootstrap", bootstrap, "--seed", seed,
            "--validation-receipt", quality_validation, "--feature-config-sha256", digest(_json(features)), "--source-manifest-sha256", source_id,
            "--suite-plan", plan, "--suite", suite, "--generated-mosaics", products / "generated/mosaics.json",
            "--reference-mosaics", products / "reference/mosaics.json", "--patch-plan", patch), outputs=[products / "quality.json"], exits=(0, 1)))
        add(_step("recompute-all-metric-tables", _module("export_tables", "--plan", plan, "--suite", suite, "--bootstrap", bootstrap, "--seed", seed,
            "--motion-result", products / "motion/motion.json", "--quality-result", products / "quality.json", "--out", products / "tables"), outputs=[products / "tables"], exits=(0, 1)))
    elif stage in {"motion-calibration", "motion-validation"}:
        build, _audit = _data_paths(p, "core-data"); m = p["measurement"]
        role = "metric_calibration" if stage == "motion-calibration" else "metric_validation"
        argv = [sys.executable, "-c", "from rswm_inria.motion_protocol import calibration_main; raise SystemExit(calibration_main())",
            "build-pairs", "--world-index", str(build / "private/index.json"), "--world-root", str(build / "private/worlds"), "--role", role,
            "--pairs-per-world", str(_required_integer(m["pairs_per_world"], "pairs_per_world")), "--seed", str(_required_integer(m["pair_seed"], "pair_seed", zero=True)), "--out", str(products / "pairs")]
        add(_step("freeze-controlled-" + role + "-pairs", argv, outputs=[products / "pairs"]))
        if stage == "motion-calibration":
            argv = [sys.executable, "-c", "from rswm_inria.motion_protocol import calibration_main; raise SystemExit(calibration_main())",
                "calibrate", "--pair-manifest", str(products / "pairs/pairs.json"), "--candidates", str(path(m["motion_candidates"], file=True)), "--out", str(products / "calibration")]
        else:
            argv = [sys.executable, "-c", "from rswm_inria.motion_protocol import validation_main; raise SystemExit(validation_main())",
                "validate", "--pair-manifest", str(products / "pairs/pairs.json"), "--calibration", str(path(m["motion_calibration"], file=True)),
                "--exposure-ledger", str(path(m["motion_exposure_ledger"])), "--out", str(products / "validation")]
        add(_step("run-real-instrument-producer", argv, outputs=[products / ("calibration" if stage == "motion-calibration" else "validation")]))
    elif stage in {"feature-calibration", "feature-validation"}:
        inputs = _json(path(p["measurement"]["feature_inputs"], file=True))
        names = ("config", "acceptance_config", "reference_mosaics", "patch_plan", "suite_plan", "source_audit", "cache")
        required = set(names) | ({"calibration_receipt", "exposure_ledger"} if stage == "feature-validation" else set())
        if set(inputs) != required: raise ValueError("Feature inputs must exactly match the existing release producer inputs")
        mode = "validate" if stage == "feature-validation" else "calibrate"
        argv = _module("feature_release", mode)
        base = path(p["measurement"]["feature_inputs"], file=True).parent
        for key in sorted(required):
            resolved = _safe_path(inputs[key], base=base, file=key not in {"cache", "exposure_ledger"})
            argv.extend(["--" + key.replace("_", "-"), str(resolved)])
        argv.extend(["--out", str(products / "feature-receipt.json")])
        add(_step("real-feature-" + mode, argv, outputs=[products / "feature-receipt.json"]))
    elif stage in {"paired", "lr64"}:
        study = p["independent"]; model_file = path(study["models"], file=True); seeds = _seeds(study["seeds"])
        plan = products / "plan.json"; root = products / "suite"
        if stage == "paired":
            argv = _module("paired_execution", "plan", "--task-index", path(study["paired_task_index"], file=True),
                "--world-index", path(study["paired_world_index"], file=True), "--world-dir", path(study["paired_world_dir"], directory=True))
            if study["mechanisms"] is not None: argv.extend(["--mechanisms", str(path(study["mechanisms"], file=True))])
            module = "paired_execution"
        else:
            argv = _module("lr64", "plan", "--materialized", path(study["lr64_materialized"], directory=True)); module = "lr64"
        argv.extend(["--models", str(model_file), "--seeds", *seeds, "--code-commit", p["code_commit"], "--out", str(plan)])
        add(_step("freeze-independent-namespace", argv, outputs=[plan]))
        argv = _module(module, "run", "--plan", plan, "--root", root)
        if resume and root.exists(): argv.append("--resume")
        add(_step("execute-independent-study", argv, outputs=[root], exits=(0, 1)))
        add(_step("independent-replay", _module(module, "verify", "--plan", plan, "--root", root), exits=(0, 1)))
        add(_step("independent-world-aggregation", _module(module, "aggregate", "--plan", plan, "--root", root,
            "--bootstrap", _required_integer(study["bootstrap"], "independent.bootstrap"), "--out", products / "analysis.json"), outputs=[products / "analysis.json"], exits=(0, 1)))
    return steps


def _receipt_path(root):
    head = _json(root / "receipt-head.json")
    relative = Path(head["receipt_path"])
    if relative.is_absolute() or ".." in relative.parts: raise ValueError("Experiment receipt head path escape")
    return root / relative


def _qualified_status(stage, receipt):
    allowed = {"ACCEPTED_REAL_PILOT", "EXECUTED_REAL_COMPARISON_PILOT_WITH_FAILURES"} if stage != "formal" else {
        "EXECUTED_FORMAL_PENDING_INSTRUMENT_ANALYSIS", "EXECUTED_FORMAL_WITH_RETAINED_FAILURES_PENDING_INSTRUMENT_ANALYSIS"}
    if receipt.get("acceptance_status") not in allowed or receipt.get("budget_usage", {}).get("exhausted") is not False:
        raise StageFailure("Actual experiment did not reach its contract-qualified terminal status")
    return receipt["acceptance_status"]


def _semantic(stage, products, p):
    if stage in {"pilot-data", "core-data"}:
        report = _json(products / "source-audit.json")
        if report.get("source_code_sha256") != code_fingerprint() or report.get("acceptance_status") != "ACCEPTED_IN_STATED_SCOPE":
            raise StageFailure("Current independent data audit did not accept this scope")
        from .materialized import validate_audit_receipt
        expected = [{"task_id": tid, **task, "city": report["worlds"][task["world_id"]]["city"],
                     "world_file_sha256": report["worlds"][task["world_id"]]["world_file_sha256"],
                     "world_pixel_sha256": report["worlds"][task["world_id"]]["world_pixel_sha256"],
                     "source_manifest_sha256": report["manifest_sha256"]} for tid, task in report["tasks"].items()]
        validate_audit_receipt(report, {"expected": expected, "source_code_sha256": code_fingerprint()},
                               required_profile="pilot-dev2" if stage == "pilot-data" else "core120")
    elif stage in {"pilot", "trained-pilot", "formal"}:
        evidence_root = products / "experiment"; receipt_path = _receipt_path(evidence_root)
        receipt = _json(receipt_path); plan = _json(products / "suite-plan.json"); experiment = _json(products / "experiment.json")
        if (receipt.get("schema") != "rswm-experiment-receipt-v1" or receipt.get("receipt_sha256") != digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
                or experiment.get("experiment_id") != digest({k: v for k, v in experiment.items() if k != "experiment_id"})
                or receipt.get("source_code_sha256") != code_fingerprint() or plan.get("source_code_sha256") != code_fingerprint()
                or receipt.get("plan_sha256") != plan.get("plan_sha256") or receipt.get("experiment_id") != experiment.get("experiment_id")
                or receipt.get("purpose") != "real" or receipt.get("kind") != ("formal" if stage == "formal" else "pilot")
                or receipt.get("evidence_root") != str(evidence_root.resolve())
                or receipt.get("receipt_path") != receipt_path.relative_to(evidence_root).as_posix()
                or receipt.get("expected_runs") != len(plan["expected"])
                or receipt.get("model_manifest_sha256") != sorted({row["model_manifest_sha256"] for row in plan["expected"]})):
            raise StageFailure("Retained real experiment receipt/plan/source binding is missing or inconsistent")
        integrity = receipt.get("integrity", {})
        if (integrity.get("verified_submissions") != len(plan["expected"]) or any(integrity.get(key) for key in ("errors", "unexpected_runs", "all_attempt_errors"))
                or integrity.get("resource_incomplete_runs") != 0 or integrity.get("resource_incomplete_attempts") != 0):
            raise StageFailure("Retained experiment submissions/resources are incomplete")
        return _qualified_status(stage, receipt)
    elif stage in {"train", "resume"}:
        config = _json(_parent_products(p, "training-inputs") / "training-config.json")
        checkpoint = _train_root(p) / f"update-{config['updates']:08d}.pt"
        receipt = _json(checkpoint.with_suffix(".json")); binding = _json(_train_root(p) / "training-contract.json")
        if (receipt.get("schema") != "rswm-training-checkpoint-v2" or receipt.get("contract_sha256") != binding.get("contract_sha256")
                or receipt.get("training", {}).get("config_sha256") != digest(config)
                or receipt.get("training", {}).get("source_code_sha256") != code_fingerprint()
                or receipt.get("training", {}).get("step") != config["updates"]
                or set(receipt.get("expert_state_sha256", {})) != {"low", "high"}):
            raise StageFailure("Actual final Wan checkpoint lacks current complete strong training evidence")
        from .training import validate_checkpoint_receipt
        validate_checkpoint_receipt(checkpoint, binding["contract_sha256"])
    return "STAGE_EXECUTED_RESEARCH_ACCEPTANCE_NOT_EVALUATED"


def run_stage(location, p, stage, *, resume=False):
    _rights(p, stage); identity = _stage_identity(location, p, stage)
    done = _completed(p, stage)
    if done is not None:
        if done["identity_sha256"] != identity["identity_sha256"]: raise ValueError("Completed stage configuration changed")
        return {"stage": stage, "status": "SKIPPED_VERIFIED_COMPLETED", "products": done["products"]}
    root = Path(p["run_root"]) / "stages" / stage
    contract = root / "stage-contract.json"
    if contract.exists() and _json(contract) != identity: raise ValueError("Started stage configuration/budget/source changed; do not reset its costs")
    prior, spent = _prior(root)
    if prior and not resume: raise ValueError("Existing failed/interrupted stage requires explicit --resume")
    budget = _budget(p, stage)
    if len(prior) >= budget["max_attempts"] or spent >= budget["max_wall_seconds"]: raise StageFailure("Cumulative stage attempt/wall budget is exhausted")
    if stage in GPU_STAGES and not _gpu_available(): raise ValueError("Actual CUDA Torch/NVIDIA target must be available before model execution")
    attempt = root / "attempts" / f"attempt-{len(prior):03d}"
    products = root / "products" if stage in REUSABLE else attempt / "products"
    steps = _build_steps(location, p, stage, products, resume=bool(prior) or stage == "resume")
    if not steps: raise ValueError("Stage has no implemented RSWM operations")
    products.mkdir(parents=True, exist_ok=True)
    if not contract.exists(): _save(contract, identity, exclusive=True)
    start = time.monotonic()
    _save(attempt / "start.json", {"pid": os.getpid(), "pid_start_token": _pid_token(os.getpid()), "stage": stage,
          "started_epoch": time.time(), "identity_sha256": identity["identity_sha256"], "prior_attempts": prior}, exclusive=True)
    roots = [root] + ([_train_root(p)] if stage in {"train", "resume"} else [])
    old_term = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame): raise KeyboardInterrupt("SIGTERM")
    signal.signal(signal.SIGTERM, interrupted)
    status = "FAILED"; error = None
    try:
        for number, step in enumerate(steps):
            marker_path = root / "steps" / f"step-{number:03d}.json" if stage in REUSABLE else attempt / "steps" / f"step-{number:03d}.json"
            if marker_path.exists():
                old = _json(marker_path)
                if old["label"] != step["label"] or any((_inventory(Path(path)) if Path(path).is_dir() else file_hash(path)) != sha for path, sha in old["outputs"].items()):
                    raise ValueError("Completed command artifacts changed")
                # A retained successful command is evidence, including a
                # terminal child whose parent crashed before stage publication.
                # Preserve its original first attempt; recheck stage semantics.
                continue
            if spent + time.monotonic() - start >= budget["max_wall_seconds"] or _tree_bytes(roots) >= budget["max_output_bytes"]:
                raise StageFailure("Cumulative stage budget is exhausted before next command")
            folder = attempt / "commands" / f"command-{number:03d}"
            if step["argv"] is not None:
                exit_code = _run_process(step["argv"], folder, cwd=Path(p["source_root"]), budget=budget, roots=roots,
                                         prior_seconds=spent, attempt_started=start)
            else:
                folder.mkdir(parents=True, exist_ok=False); before = time.monotonic()
                _save(folder / "start.json", {"kind": "fixed_internal_derivation", "label": step["label"], "pid": os.getpid(), "pid_start_token": _pid_token(os.getpid())}, exclusive=True)
                internal_exit, internal_error = 2, None
                try:
                    step["action"](); internal_exit = 0
                except BaseException as exc:
                    internal_error = str(exc); raise
                finally:
                    _save(folder / "exit.json", {"exit_code": internal_exit, "error": internal_error,
                          "elapsed_seconds": time.monotonic() - before, "live": False}, exclusive=True)
                exit_code = internal_exit
            if exit_code not in step["exits"]: raise StageFailure("Child command failed with actual exit " + str(exit_code) + ": " + step["label"])
            if step.get("validate") is not None: step["validate"]()
            outputs = {str(path): _inventory(Path(path)) if Path(path).is_dir() else file_hash(path) for path in step["outputs"]}
            # Contract-qualified experiment receipt decides terminal success;
            # rc=1 alone must not turn INCOMPLETE into completion.
            if step["label"] == "execute-frozen-real-experiment": _semantic(stage, products, p)
            _save(marker_path, {"label": step["label"], "exit_code": exit_code, "outputs": outputs}, exclusive=True)
        semantic = _semantic(stage, products, p)
        if spent + time.monotonic() - start >= budget["max_wall_seconds"] or _tree_bytes(roots) + 65536 >= budget["max_output_bytes"]:
            raise StageFailure("Stage exceeded budget including reserved publication metadata")
        external = {}
        for step in steps:
            for path in step["outputs"]:
                path = Path(path)
                if not path.is_relative_to(products): external[str(path)] = _inventory(path) if path.is_dir() else file_hash(path)
        marker = {"stage": stage, "identity_sha256": identity["identity_sha256"], "products": str(products), "files": _inventory(products),
                  "external_artifacts": external,
                  "append_only_training_roots": [str(_train_root(p))] if stage in {"train", "resume"} else [],
                  "status": semantic, "research_acceptance": "NOT_EVALUATED", "attempt": attempt.name}
        marker["marker_sha256"] = digest(marker)
        if spent + time.monotonic() - start >= budget["max_wall_seconds"] or _tree_bytes(roots) + len(canonical(marker)) + 8192 >= budget["max_output_bytes"]:
            raise StageFailure("Stage exhausted budget during retained-artifact verification/publication reserve")
        _save(root / "completed.json", marker, exclusive=True)
        status = "EXECUTED"; return {"stage": stage, "status": semantic, "products": str(products)}
    except BaseException as exc:
        status = "INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "FAILED"; error = str(exc); raise
    finally:
        signal.signal(signal.SIGTERM, old_term)
        _save(attempt / "finish.json", {"status": status, "error": error, "elapsed_seconds": time.monotonic() - start,
              "prior_elapsed_seconds": spent, "output_bytes": _tree_bytes(roots), "live": False,
              "next_command": ["rswm-server", "run", "--project", str(location), "--stage", stage, "--resume"],
              "research_acceptance": "NOT_EVALUATED"}, exclusive=True)


def run_project(project, stage, *, resume=False):
    location, p = _project(project); results = []
    with _coordinator(Path(p["run_root"])):
        for current in _order(stage, p):
            try: results.append(run_stage(location, p, current, resume=resume))
            except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError, KeyboardInterrupt) as exc:
                value = {"status": "HANDOFF_REQUIRED", "stage": current, "reason": str(exc), "completed_this_invocation": results,
                         "goal": "RSWM_V0_1_RESEARCH_PACKAGE_ACCEPTED", "goal_completed": False,
                         "next_command": ["rswm-server", "run", "--project", str(location), "--stage", current, "--resume"]}
                _save(Path(p["run_root"]) / "cursor.json", value)
                return value
        value = {"status": "REQUESTED_CODE_STAGES_EXECUTED", "stages": results, "goal_completed": False,
                 "research_acceptance": "NOT_EVALUATED", "next_command": None}
        _save(Path(p["run_root"]) / "cursor.json", value); return value


def status_project(project):
    location, p = _project(project); stages = []
    for stage in STAGES:
        root = Path(p["run_root"]) / "stages" / stage
        try:
            completed = _completed(p, stage); rows, spent = _prior(root)
            stages.append({"stage": stage, "status": completed["status"] if completed else "NOT_STARTED" if not rows else rows[-1]["status"],
                           "attempts": rows, "elapsed_charged_seconds": spent, "products": completed["products"] if completed else None})
        except (ValueError, OSError, KeyError) as exc: stages.append({"stage": stage, "status": "ATTENTION_REQUIRED", "reason": str(exc)})
    cursor = Path(p["run_root"]) / "cursor.json"
    return {"stages": stages, "cursor": _json(cursor) if cursor.exists() else None, "goal_completed": False}


def plan_project(project, stage):
    location, p = _project(project); rows = []
    for current in _order(stage, p):
        try:
            _rights(p, current); identity = _stage_identity(location, p, current)
            done = _completed(p, current)
            if done is not None and done["identity_sha256"] != identity["identity_sha256"]: raise ValueError("Completed stage configuration changed")
            products = Path(p["run_root"]) / "stages" / current / "products"
            steps = _build_steps(location, p, current, products, resume=False) if done is None else []
            rows.append({"stage": current, "status": "VERIFIED_COMPLETED" if done else "READY_TO_EXECUTE", "identity_sha256": identity["identity_sha256"],
                         "operations": [{"label": s["label"], "argv": s["argv"], "kind": "child_process" if s["argv"] else "fixed_input_derivation"} for s in steps]})
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            rows.append({"stage": current, "status": "INPUT_OR_DEPENDENCY_BLOCKED", "reason": str(exc)})
    return {"scope": "plan only; no children launched", "stages": rows, "goal_completed": False}


def check_project(project, stage):
    result = plan_project(project, stage)
    _location, p = _project(project)
    root = Path(p["run_root"])
    while not root.exists(): root = root.parent
    result["environment"] = {"python": sys.version, "platform": sys.platform, "posix_run_supported": os.name == "posix",
                             "disk_free_bytes": shutil.disk_usage(root).free, "cuda_torch_available": _gpu_available(),
                             "source_code_sha256": code_fingerprint(), "scope": "read-only environment/config check; no model loaded"}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init"); init.add_argument("--source-root", required=True); init.add_argument("--out", required=True)
    for name in ("check", "plan", "run", "status"):
        command = commands.add_parser(name); command.add_argument("--project", required=True)
        if name in {"check", "plan", "run"}: command.add_argument("--stage", choices=(*STAGES, "all"), default="pilot")
        if name == "run": command.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "init": result = {"project": str(init_project(args.out, args.source_root)), "status": "TEMPLATE_CREATED_UNAPPROVED"}
        elif args.command == "status": result = status_project(args.project)
        elif args.command == "run": result = run_project(args.project, args.stage, resume=args.resume)
        elif args.command == "check": result = check_project(args.project, args.stage)
        else: result = plan_project(args.project, args.stage)
        print(json.dumps(result, allow_nan=False))
        return 1 if result.get("status") == "HANDOFF_REQUIRED" or any(r.get("status") == "INPUT_OR_DEPENDENCY_BLOCKED" for r in result.get("stages", [])) else 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_INVALID", "reason": str(exc), "goal_completed": False})); return 2


if __name__ == "__main__": raise SystemExit(main())
