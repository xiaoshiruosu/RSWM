"""Estimate, or explicitly perform, storage deduplication after full smoke success.

Only SHA-identical journal-authorized PNGs are linked. Original path names and
all file bytes are retained. This is a post-run storage operation: resource
receipts remain the original logical-byte measurements, and no original model
runtime or storage saving is claimed. Hardlinked names share the canonical
inode's mode and mtime; these metadata changes are recorded separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time

from rswm_inria.geometry import canonical, digest
from rswm_inria.replay import load_json, schedule
from rswm_inria.runner import verify_journal
from rswm_inria.suite import _metadata, _read_attempt, _root_identity, validate_plan

FULL_RUNS = 576
FULL_OUTPUTS = 50688
SHA = re.compile(r"[0-9a-f]{64}\Z")
STREAMS = {"main", "small", "large", "prefix", "a", "b"}


def safe_path(value):
    if ".." in Path(value).parts:
        raise ValueError("Parent traversal components are forbidden")
    path = Path(value).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink paths or ancestors are forbidden")
    return path


def stat_state(value):
    return {"dev": value.st_dev, "inode": value.st_ino, "size": value.st_size,
            "mtime_ns": value.st_mtime_ns, "mode": stat.S_IMODE(value.st_mode),
            "nlink": value.st_nlink, "st_blocks": value.st_blocks, "allocated_bytes": value.st_blocks * 512}


def file_state(path):
    path = safe_path(path)
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Only ordinary files are accepted: " + str(path))
    return stat_state(value)


class FileStateChanged(ValueError):
    """Retain the observed mismatch, without attributing its cause or writer."""

    def __init__(self, path, phase, before, after):
        changed = {field: {"before": before[field], "after": after[field]}
                   for field in before if before[field] != after[field]}
        self.diagnostic = {"path": str(path), "phase": phase,
                           "changed_fields": changed, "before": before, "after": after,
                           "cause": "UNKNOWN"}
        super().__init__("File changed while " + phase + ": " + str(path) +
                         "; fields=" + json.dumps(changed, sort_keys=True))


def file_hash(path):
    before = file_state(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        opened = stat_state(os.fstat(fd))
        if opened != before:
            raise FileStateChanged(path, "opening", before, opened)
        result = hashlib.sha256()
        while block := os.read(fd, 1024 * 1024):
            result.update(block)
        descriptor_after = stat_state(os.fstat(fd))
        if descriptor_after != before:
            raise FileStateChanged(path, "hashing open descriptor", before, descriptor_after)
    finally:
        os.close(fd)
    after = file_state(path)
    if after != before:
        raise FileStateChanged(path, "hashing pathname", before, after)
    return result.hexdigest(), before


def inventory(root):
    root = safe_path(root)
    if not root.is_dir() or any("forensic" in p.lower() for p in root.parts):
        raise ValueError("A completed raw directory is required; forensics are forbidden")
    files = set()
    for folder, directories, names in os.walk(root, followlinks=False):
        for name in directories:
            path = safe_path(Path(folder) / name)
            if not path.is_dir() or "forensic" in name.lower():
                raise ValueError("Unexpected or forensic directory")
        for name in names:
            path = safe_path(Path(folder) / name)
            file_state(path)
            files.add(path.relative_to(root).as_posix())
    return files


def read_json(path):
    return load_json(Path(path).read_text(encoding="utf-8"))


def load_hash_map(path, root):
    value = read_json(path)
    if isinstance(value, dict) and value.get("schema") == "rswm-completed-smoke-sha256-map-v1":
        if value.get("raw_root") != str(root):
            raise ValueError("SHA map names a different raw root")
        value = value.get("files")
    if not isinstance(value, dict) or not value:
        raise ValueError("A complete original relative-path SHA map is required")
    for name, checksum in value.items():
        path = PurePosixPath(name) if isinstance(name, str) else None
        if (path is None or name != path.as_posix() or path.is_absolute() or
                "\\" in name or any(p in {".", "..", ""} for p in name.split("/")) or
                not isinstance(checksum, str) or not SHA.fullmatch(checksum)):
            raise ValueError("Unsafe path or checksum in original SHA map")
    return value


def verify_map(root, original):
    if inventory(root) != set(original):
        raise ValueError("Raw file path inventory differs from the complete original SHA map")
    states = {}
    for name, expected in sorted(original.items()):
        actual, state = file_hash(root / name)
        if actual != expected:
            raise ValueError("Original file bytes changed: " + name)
        states[name] = {**state, "sha256": actual}
    return states


def guard(root, name, expected, *, nlink=None, content=True):
    actual = file_state(root / name)
    for field in ("dev", "inode", "size", "mtime_ns", "mode"):
        if actual[field] != expected[field]:
            raise ValueError("Stable file guard changed: " + name + "; field=" + field +
                             "; expected=" + str(expected[field]) + "; actual=" + str(actual[field]))
    if actual["nlink"] != (expected["nlink"] if nlink is None else nlink):
        raise ValueError("Hardlink count changed: " + name + "; expected=" +
                         str(expected["nlink"] if nlink is None else nlink) + "; actual=" + str(actual["nlink"]))
    if content and file_hash(root / name)[0] != expected["sha256"]:
        raise ValueError("Guarded file SHA changed: " + name)
    return actual


def accepted_command(commands_path, raw, acceptance):
    value = read_json(commands_path)
    rows = value.get("commands") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("Actual command records must be a list")
    for row in rows:
        argv = row.get("argv") if isinstance(row, dict) else None
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            continue
        entry = any(a == "rswm_inria.smoke12" or Path(a).name in {"rswm-smoke12", "accept_smoke12.py"} for a in argv)
        try:
            output = safe_path(argv[argv.index("--out") + 1])
            commit = argv[argv.index("--code-commit") + 1]
        except (ValueError, IndexError):
            continue
        code = row.get("exit_code", row.get("returncode"))
        if (entry and output == raw and "--execute" in argv and "--task-limit" not in argv and
                commit == acceptance["code_commit"] and type(code) is int and code == 0 and not row.get("timeout", False)):
            return row
    raise ValueError("No successful full-smoke --execute command binds this raw root")


def full_gate(raw, original, states, commands_path):
    acceptance = read_json(raw / "acceptance.json")
    if not isinstance(acceptance, dict):
        raise ValueError("Final acceptance must be a JSON object")
    failed = acceptance.get("failed_runs")
    no_failures = (type(failed) is int and failed == 0) or (type(failed) is list and not failed)
    if (acceptance.get("schema") != "rswm-smoke12-acceptance-v1" or
            acceptance.get("status") != "REAL_DATA_VALIDATED" or acceptance.get("real_data_validated") is not True or
            acceptance.get("full_smoke12_matrix") is not True or acceptance.get("planned_runs_total") != FULL_RUNS or
            acceptance.get("planned_unique_outputs_total") != FULL_OUTPUTS or
            acceptance.get("analysis_status") != "ACCEPTED" or not no_failures or
            acceptance.get("input_kind") != "source_identity_checked_geography_unknown"):
        raise ValueError("Only successful complete real full smoke12 is eligible")
    if acceptance.get("artifact_sha256") != {p: h for p, h in original.items() if p != "acceptance.json"}:
        raise ValueError("Acceptance artifact map does not bind the complete original raw inventory")
    accepted_command(commands_path, raw, acceptance)
    plan = read_json(raw / "plan.json")
    validate_plan(plan)
    if (len(plan["expected"]) != FULL_RUNS or plan.get("retry_policy", {}).get("max_attempts_per_run") != 1 or
            plan.get("plan_sha256") != acceptance["plan_sha256"] or
            {r["code_commit"] for r in plan["expected"]} != {acceptance["code_commit"]} or
            plan.get("source_code_sha256") != acceptance["source_code_sha256"] or
            plan.get("diagnostic_acceptance_profile") != {"profile": "smoke12", "input_kind": acceptance["input_kind"], "formal_test": False}):
        raise ValueError("Final acceptance does not match the frozen full first-attempt plan")
    suite = raw / "suite"
    _root_identity(plan, suite)
    expected_ids = {r["run_id"] for r in plan["expected"]}
    if {p.name for p in (suite / "runs").iterdir()} != expected_ids:
        raise ValueError("Unexpected or missing runs")
    candidates, witnesses, total = {}, {}, 0
    by_model = defaultdict(lambda: {"runs": 0, "planned_outputs": 0, "valid_outputs": 0, "resource_complete_runs": 0})
    for row in plan["expected"]:
        run_id = row["run_id"]
        run_parent = suite / "runs" / run_id
        if {p.name for p in run_parent.iterdir()} != {"attempt-000"}:
            raise ValueError("Only the completed first attempt is eligible")
        folder = run_parent / "attempt-000"
        outcome = _metadata(folder / "outcome.json", row, plan, 0, "outcome", folder)
        if (outcome.get("status") != "COMPLETED" or outcome.get("execution_status") != "COMPLETED" or
                outcome.get("submission_status") != "VERIFIED" or outcome.get("resource_status") != "COMPLETE" or
                outcome.get("replay_verified") is not True):
            raise ValueError("Incomplete or unverified run")
        metadata = _read_attempt(plan, suite, row, folder, verify=False)
        if metadata["integrity_errors"] or metadata["resource_status"] != "COMPLETE":
            raise ValueError("Final metadata/resource authentication failed")
        head_path, receipt_path = suite / "heads" / (run_id + "-000.json"), suite / "receipts" / (run_id + "-000.json")
        head = _metadata(head_path, row, plan, 0, "head")
        receipt = _metadata(receipt_path, row, plan, 0, "receipt")
        if (receipt.get("status") != "COMPLETED" or receipt.get("replay_head") != head.get("journal_head") or
                receipt.get("outcome_sha256") != states[(folder / "outcome.json").relative_to(raw).as_posix()]["sha256"] or
                outcome.get("replay_head") != head.get("journal_head")):
            raise ValueError("Final receipt/outcome/head mismatch")
        submission = folder / "submission"
        task = plan["tasks"][row["task_id"]]
        if digest(read_json(submission / "task.json")) != row["task_sha256"]:
            raise ValueError("Submission task differs from its frozen plan")
        replayed = read_json(folder / "replayed.json")
        checks = replayed.get("artifact_checks", {})
        if (replayed.get("journal_head") != head["journal_head"] or replayed.get("seed") != row["seed"] or
                canonical(replayed.get("reports")) != canonical(outcome["reports"]) or
                any(checks.get(k) is not True for k in ("trace_matches_plan", "pixel_hashes_verified", "summary_matches", "externally_anchored_head"))):
            raise ValueError("Final independently anchored replay is incomplete")
        journal = [load_json(line) for line in (submission / "events.jsonl").read_text().splitlines()]
        if verify_journal(journal) != head["journal_head"]:
            raise ValueError("Journal does not authenticate against the retained head")
        header = journal[0]
        if any(header.get(k) != v for k, v in {"kind": "session", "journal_schema": "rswm-journal-v1", "task_id": row["task_id"],
                "task_sha256": row["task_sha256"], "seed": row["seed"], "initial_sha256": row["initial_sha256"]}.items()):
            raise ValueError("Journal session differs from its frozen identity")
        trace = [j for j in journal if j.get("kind") in {"request", "response", "output"}]
        expected = list(schedule(task))
        if len(trace) != len(expected) or any(any(canonical(j.get(k)) != canonical(v) for k, v in want.items()) for j, want in zip(trace, expected)):
            raise ValueError("Journal trace differs from the planned schedule")
        outputs = [j for j in trace if j["kind"] == "output"]
        if any(j.get("status") != "valid" or not SHA.fullmatch(j.get("pixel_sha256", "")) for j in outputs):
            raise ValueError("A planned output is invalid")
        if replayed.get("verified_png_outputs") != len(outputs) or replayed.get("planned_output_events") != len(outputs):
            raise ValueError("Final replay output denominator differs")
        files = {submission / "initial.png": "initial"}
        for j in outputs:
            if j["stream"] not in STREAMS or type(j["event"]) is not int:
                raise ValueError("Unsafe output pathname")
            files[submission / j["stream"] / f"{j['event']:04d}.png"] = "output"
        if set(submission.rglob("*.png")) != set(files):
            raise ValueError("Unplanned or missing submission PNG")
        witness = [p for p in folder.rglob("*") if p.is_file() and p.suffix != ".png"] + [head_path, receipt_path]
        witnesses[run_id] = {p.relative_to(raw).as_posix() for p in witness}
        for path, kind in files.items():
            name = path.relative_to(raw).as_posix()
            if name not in states:
                raise ValueError("Candidate is absent from complete original SHA map")
            candidates[name] = {"scope": (row["model_id"], row["world_id"], tuple(task["origin"]), kind), "run_id": run_id}
        total += len(outputs)
        model = by_model[row["model_id"]]
        for field, amount in (("runs", 1), ("planned_outputs", len(outputs)), ("valid_outputs", len(outputs)), ("resource_complete_runs", 1)):
            model[field] += amount
    if total != FULL_OUTPUTS or dict(by_model) != acceptance.get("per_model"):
        raise ValueError("Full completed 576/50688 matrix differs from acceptance")
    return candidates, witnesses


def duplicate_groups(candidates, states):
    groups, excluded = defaultdict(list), []
    for name, item in sorted(candidates.items()):
        value = states[name]
        if value["nlink"] != 1:
            excluded.append({"path": name, "reason": "INITIAL_NLINK_NOT_ONE", "nlink": value["nlink"]})
            continue
        groups[(item["scope"], value["sha256"], value["dev"], value["mode"])].append(name)
    return [names for names in groups.values() if len(names) > 1], excluded


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def log_row(handle, value):
    handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def replace_group(raw, names, states, witness_names, stage, ledger, group_number, global_guard=None, scope=None):
    """Replace only after all guards; failures retain stage and append-only evidence."""
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("A distinct duplicate group is required")
    identity = {(states[n]["sha256"], states[n]["dev"], states[n]["mode"]) for n in names}
    if len(identity) != 1 or any(states[n]["nlink"] != 1 for n in names):
        raise ValueError("Only initially unlinked identical regular files may share an inode")
    if safe_path(stage) == safe_path(raw) or safe_path(stage).is_relative_to(safe_path(raw)):
        raise ValueError("Stage must remain outside the original raw tree")
    for name in names:
        if Path(name).suffix != ".png":
            raise ValueError("Only authorized PNG paths may be replaced")
        with safe_path(raw / name).open("rb") as stream:
            if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError("A candidate is not a PNG container")
    if safe_path(stage).stat().st_dev != states[names[0]]["dev"]:
        raise ValueError("Stage and PNGs must use the same filesystem")
    if global_guard:
        global_guard()
    for name in witness_names:
        guard(raw, name, states[name])
    for name in names:
        guard(raw, name, states[name])
    canonical_name = names[0]
    before_free = shutil.disk_usage(stage).free
    log_row(ledger, {"kind": "GROUP_PREPARED", "group": group_number, "scope": scope, "canonical": canonical_name,
                     "original": {name: states[name] for name in names}, "filesystem_free_bytes": before_free})
    for number, target_name in enumerate(names[1:], 1):
        if global_guard:
            global_guard()
        for name in witness_names:
            guard(raw, name, states[name])
        guard(raw, canonical_name, states[canonical_name], nlink=number)
        guard(raw, target_name, states[target_name])
        temporary = stage / f"group-{group_number:06d}-target-{number:06d}.png"
        if temporary.exists() or temporary.is_symlink():
            raise FileExistsError("Staging evidence must never be overwritten")
        os.link(raw / canonical_name, temporary, follow_symlinks=False)
        fd = os.open(temporary, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        fsync_directory(stage)
        guard(raw, canonical_name, states[canonical_name], nlink=number + 1)
        guard(raw, target_name, states[target_name])
        for name in witness_names:
            guard(raw, name, states[name])
        if global_guard:
            global_guard()
        os.replace(temporary, raw / target_name)
        fsync_directory((raw / target_name).parent)
        fsync_directory(stage)
        result = {**file_state(raw / target_name), "sha256": file_hash(raw / target_name)[0]}
        if result["sha256"] != states[target_name]["sha256"] or result["inode"] != states[canonical_name]["inode"]:
            raise ValueError("Post-replacement inode or byte identity mismatch")
        log_row(ledger, {"kind": "PATH_REPLACED", "group": group_number, "path": target_name,
                         "canonical": canonical_name, "original": states[target_name], "result": result,
                         "mtime_policy": "Target now shares canonical mtime; canonical mtime is unchanged."})
    result = {name: {**file_state(raw / name), "sha256": file_hash(raw / name)[0]} for name in names}
    log_row(ledger, {"kind": "GROUP_COMPLETE", "group": group_number, "canonical": canonical_name,
                     "result": result, "filesystem_free_bytes": shutil.disk_usage(stage).free})


def totals(states):
    unique = {(s["dev"], s["inode"]): s["allocated_bytes"] for s in states.values()}
    return {"files": len(states), "logical_bytes_per_path": sum(s["size"] for s in states.values()),
            "unique_inodes": len(unique), "unique_inode_allocated_bytes": sum(unique.values())}


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(Path(path).parent)


def gzip_copy(source, target):
    """Retain the exact independently supplied map bytes without a large text copy."""
    source = safe_path(source)
    with source.open("rb") as original, Path(target).open("xb") as encoded:
        with gzip.GzipFile(filename="", mode="wb", fileobj=encoded, mtime=0) as compressed:
            shutil.copyfileobj(original, compressed)
        encoded.flush()
        os.fsync(encoded.fileno())
    checksum, state = file_hash(target)
    return {"path": Path(target).name, "gzip_sha256": checksum, "gzip_bytes": state["size"],
            "original_sha256": file_hash(source)[0], "original_bytes": file_state(source)["size"], "gzip_mtime": 0}


def compressed_receipt(path):
    checksum, state = file_hash(path)
    result = {"gzip_sha256": checksum, "gzip_bytes": state["size"], "gzip_mtime": 0}
    try:
        raw_hash, raw_bytes = hashlib.sha256(), 0
        with gzip.open(path, "rb") as stream:
            while block := stream.read(1024 * 1024):
                raw_hash.update(block)
                raw_bytes += len(block)
        result.update(raw_sha256=raw_hash.hexdigest(), raw_bytes=raw_bytes, decompression_complete=True)
    except (OSError, EOFError) as error:
        result.update(decompression_complete=False, error_type=type(error).__name__)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--sha-map", required=True, type=Path, help="Independently retained complete original relative-path SHA256 map")
    parser.add_argument("--commands", required=True, type=Path, help="Original successful full-smoke command record")
    parser.add_argument("--out", "--evidence", dest="out", required=True, type=Path, help="New compact evidence directory outside raw")
    parser.add_argument("--work", required=True, type=Path, help="New staging/reserve directory outside raw and evidence, on the same filesystem")
    parser.add_argument("--reserve-bytes", type=int, default=32 * 1024 ** 2)
    parser.add_argument("--execute", action="store_true", help="Explicit post-run atomic hardlink replacement; default only estimates")
    args = parser.parse_args(argv)
    out = None
    lock = None
    report = {"schema": "rswm-completed-smoke-storage-v1", "status": "FAILED", "execute": args.execute,
              "scope": "post-run storage only; original timing/resource/acceptance files remain byte-identical",
              "original_run_storage_savings": "NOT_CLAIMED", "trusted_source_authenticity": False,
              "mtime_policy": "Hardlinked targets share canonical mode/mtime; canonical bytes and mtime remain unchanged."}
    try:
        raw, requested_out, work = map(safe_path, (args.raw_root, args.out, args.work))
        external = [safe_path(args.sha_map), safe_path(args.commands)]
        if (not raw.is_dir() or any(p == raw or p.is_relative_to(raw) for p in [requested_out, work, *external]) or
                raw.is_relative_to(requested_out) or raw.is_relative_to(work) or requested_out == work or
                requested_out.is_relative_to(work) or work.is_relative_to(requested_out)):
            raise ValueError("SHA map, commands, evidence, reserve and stage must be outside raw; work and evidence must be separate")
        for path in (requested_out, work):
            if path.exists() and (not path.is_dir() or any(path.iterdir())):
                raise FileExistsError("Evidence and work directories must be new or empty")
            path.mkdir(parents=True, exist_ok=True)
            if path.stat().st_dev != raw.stat().st_dev:
                raise ValueError("Evidence, reserve, stage and raw must use the same filesystem")
        out = requested_out
        if not 1024 ** 2 <= args.reserve_bytes <= 256 * 1024 ** 2:
            raise ValueError("Reserve must be between 1 and 256 MiB")
        file_state(raw / "suite/coordinator.lock")
        lock = (raw / "suite/coordinator.lock").open("rb")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = load_hash_map(external[0], raw)
        states = verify_map(raw, original)
        control = {}
        for path in external:
            checksum, state = file_hash(path)
            control[str(path)] = {**state, "sha256": checksum}
        candidates, witnesses = full_gate(raw, original, states, external[1])
        if any(states[name]["dev"] != raw.stat().st_dev for name in candidates):
            raise ValueError("All candidate PNGs must use the preflighted raw filesystem")
        groups, excluded = duplicate_groups(candidates, states)
        estimate = sum(states[name]["allocated_bytes"] for names in groups for name in names[1:])
        report.update(raw_root=str(raw), sha_map=str(external[0]), commands=str(external[1]), work=str(work),
                      original_sha_map_sha256=control[str(external[0])]["sha256"],
                      commands_sha256=control[str(external[1])]["sha256"], before=totals(states),
                      candidate_files=len(candidates), duplicate_groups=len(groups),
                      candidate_scope="same model_id/world_id/origin; initial and output PNGs grouped separately",
                      accepted_runs=FULL_RUNS, verified_output_pngs=FULL_OUTPUTS,
                      acceptance_sha256=original["acceptance.json"],
                      original_path_inventory_sha256=digest(sorted(original)),
                      original_sha_map_content_digest=digest(original),
                      replaceable_paths=sum(len(g) - 1 for g in groups), excluded=excluded,
                      estimated_reclaimable_inode_allocated_bytes=estimate,
                      estimate_scope="Sum of initially nlink=1 target st_blocks*512; no original runtime saving claim.",
                      filesystem_free_bytes_before=shutil.disk_usage(raw).free)
        report["original_sha_map_copy"] = gzip_copy(external[0], out / "original-sha-map.json.gz")
        save_json(out / "preflight.json", report)
        global_names = ["acceptance.json", "plan.json", "suite/plan.json", "suite/expected.jsonl", "suite/run-index.jsonl", "suite/replay-status.json", "suite/replay-status.jsonl", "suite/receipts.jsonl"]
        def global_guard():
            for name in global_names:
                guard(raw, name, states[name], content=False)
            for path in external:
                guard(path.parent, path.name, control[str(path)], content=False)
        global_guard()
        stage = work / "stage"
        if args.execute:
            # Free this reservation only on success. Failures retain all work.
            reserve = work / "reserve.bin"
            with reserve.open("xb") as handle:
                os.posix_fallocate(handle.fileno(), 0, args.reserve_bytes)
                os.fsync(handle.fileno())
            stage.mkdir()
            fsync_directory(work)
            with (out / "inode-ledger.jsonl.gz").open("xb") as encoded:
                with gzip.GzipFile(filename="", mode="wb", fileobj=encoded, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8") as ledger:
                        for number, names in enumerate(groups):
                            relevant = set().union(*(witnesses[candidates[name]["run_id"]] for name in names))
                            replace_group(raw, names, states, relevant, stage, ledger, number, global_guard, candidates[names[0]]["scope"])
            report["inode_ledger_gzip_sha256"] = file_hash(out / "inode-ledger.jsonl.gz")[0]
            report["inode_ledger_gzip_bytes"] = file_state(out / "inode-ledger.jsonl.gz")["size"]
            final = verify_map(raw, original)
            if totals(final)["logical_bytes_per_path"] != totals(states)["logical_bytes_per_path"]:
                raise ValueError("Original logical bytes per path changed")
            changed_group_names = {name for group in groups for name in group}
            for name, state in states.items():
                if name not in changed_group_names and final[name] != state:
                    raise ValueError("Protected untouched file metadata changed: " + name)
            for group in groups:
                canonical_state = states[group[0]]
                for name in group:
                    result = final[name]
                    if any(result[k] != canonical_state[k] for k in ("dev", "inode", "size", "mode", "mtime_ns", "sha256")) or result["nlink"] != len(group):
                        raise ValueError("Final hardlink group metadata differs from its canonical inode")
            for path in external:
                guard(path.parent, path.name, control[str(path)])
            reserve.unlink()
            fsync_directory(work)
            report.update(status="POST_RUN_DEDUPLICATED_BYTES_PRESERVED", after=totals(final),
                          scoped_inode_allocated_bytes_reduction=totals(states)["unique_inode_allocated_bytes"] - totals(final)["unique_inode_allocated_bytes"],
                          original_path_inventory_and_all_file_sha256_preserved=True,
                          final_path_inventory_sha256=digest(sorted(final)),
                          final_verified_sha_map_content_digest=digest({p: s["sha256"] for p, s in final.items()}),
                          original_file_count_verified_after=len(final))
        else:
            if verify_map(raw, original) != states:
                raise ValueError("Estimate mode observed a concurrent metadata change")
            report.update(status="ESTIMATE_ONLY_NO_RAW_MUTATION", original_path_inventory_and_all_file_sha256_preserved=True,
                          final_path_inventory_sha256=digest(sorted(states)),
                          final_verified_sha_map_content_digest=digest({p: s["sha256"] for p, s in states.items()}),
                          original_file_count_verified_after=len(states))
        report["filesystem_free_bytes_after"] = shutil.disk_usage(raw).free
        report["filesystem_free_delta_attribution"] = "Observed during this storage operation; evidence/reserve writes and concurrent filesystem users may affect free bytes."
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="FAILED", error_type=type(error).__name__, reason=str(error),
                      failure_policy="Retain append-only ledger, reserve and any staged hardlinks; do not clear failure evidence or resume automatically.")
        if isinstance(error, FileStateChanged):
            report["file_state_mismatch"] = error.diagnostic
    finally:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    if out is not None and out.is_dir() and not (out / "summary.json").exists():
        for name in ("original-sha-map.json.gz", "inode-ledger.jsonl.gz"):
            if (out / name).exists():
                report.setdefault("retained_compressed_evidence", {})[name] = compressed_receipt(out / name)
        report["elapsed_finished_at_unix"] = time.time()
        save_json(out / "summary.json", report)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0 if report["status"] != "FAILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
