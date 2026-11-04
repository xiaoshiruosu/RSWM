#!/usr/bin/env python3
"""Verify supplied final archives in a new offline environment outside checkout.

This script does not build archives or substitute an earlier wheel. Raw CPU
diagnostic assets live only under --work; --out contains command logs, hashes,
selected text tables and a small acceptance summary.
"""
from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import time
import zipfile


VERSION = "0.3.0"
PROJECT = "rswm-inria-120"
MAX_FILE_BYTES = 32 * 1024 ** 2
MAX_ARCHIVE_BYTES = 128 * 1024 ** 2
FORBIDDEN_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npz", ".npy",
                      ".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".hdf5",
                      ".bin", ".onnx", ".gguf", ".pkl", ".pickle", ".parquet", ".arrow", ".h5ad",
                      ".key", ".pem", ".p12", ".pfx", ".pyc"}
USER_PATHS = re.compile(rb"/(?:Users|home)/[^/\s\"']+/|/workspace/scratch/[A-Za-z0-9_-]+/")


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def succeeded(command):
    return command["returncode"] == 0 and not command["timeout"]


def fresh_directory(path, label):
    target = Path(path).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError(label + " must be a new or empty directory")
    target.mkdir(parents=True, exist_ok=True)
    return target


def archive_entries(path):
    """Read only bounded ordinary members; do not extract a caller's archive."""
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                mode = (member.external_attr >> 16) & 0xffff
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise ValueError("Unsupported wheel member type: " + member.filename)
                if member.file_size > MAX_FILE_BYTES:
                    raise ValueError("Oversized archive member: " + member.filename)
                yield member.filename, None if member.is_dir() else archive.read(member), member.file_size
    else:
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                if not (member.isdir() or member.isfile()):
                    raise ValueError("Unsupported sdist member type: " + member.name)
                if member.size > MAX_FILE_BYTES:
                    raise ValueError("Oversized archive member: " + member.name)
                if member.isdir():
                    yield member.name, None, 0
                else:
                    with archive.extractfile(member) as stream:
                        yield member.name, stream.read(MAX_FILE_BYTES + 1), member.size


def scan_archive(path, expected_commit, expected_source, *, allow_dirty_candidate=False):
    files, python_sources, provenance, metadata = [], {}, [], []
    seen, total = set(), 0
    for name, content, size in archive_entries(path):
        clean = PurePosixPath(name)
        if (name.startswith("/") or "\\" in name or "\0" in name or
                any(part in {".", ".."} for part in name.rstrip("/").split("/"))):
            raise ValueError("Unsafe archive path: " + name)
        normalized = clean.as_posix().rstrip("/")
        if normalized in seen:
            raise ValueError("Duplicate archive member: " + normalized)
        seen.add(normalized)
        if content is None:
            continue
        total += size
        if total > MAX_ARCHIVE_BYTES or len(content) != size:
            raise ValueError("Archive size or member length invalid")
        parts, leaf = clean.parts, clean.name.lower()
        if (clean.suffix.lower() in FORBIDDEN_SUFFIXES or leaf == ".env" or leaf.startswith(".env.") or
                leaf in {"credentials", "id_rsa", "id_ed25519", "token", "token.txt"} or
                any(part.lower() in {".git", ".ssh", ".aws", ".cache", "__pycache__",
                                     "weights", "runs", "evidence", "data"} for part in parts)):
            raise ValueError("Forbidden packaged file: " + normalized)
        if re.search(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\bsk-[A-Za-z0-9_-]{32,}\b", content):
            raise ValueError("Credential material in packaged file: " + normalized)
        if clean.suffix == ".py" and USER_PATHS.search(content):
            raise ValueError("Absolute user path in packaged source: " + normalized)
        files.append({"path": normalized, "bytes": size, "sha256": digest_bytes(content)})
        package_start = None
        if parts and parts[0] == "rswm_inria":
            package_start = 1
        elif len(parts) > 2 and parts[1:3] == ("src", "rswm_inria"):
            package_start = 3
        if package_start is not None:
            relative = PurePosixPath(*parts[package_start:]).as_posix()
            if clean.suffix == ".py":
                python_sources[relative] = digest_bytes(content)
            if relative == "_build_info.json":
                provenance.append(json.loads(content))
        if clean.name in {"METADATA", "PKG-INFO"}:
            info = Parser().parsestr(content.decode("utf-8"))
            metadata.append({"path": normalized, "name": info.get("Name"), "version": info.get("Version")})
    calculated = digest_bytes(json.dumps(python_sources, ensure_ascii=False, sort_keys=True,
                                         separators=(",", ":")).encode("utf-8"))
    if not python_sources or calculated != expected_source:
        raise ValueError("Packaged Python source fingerprint mismatch")
    if len(provenance) != 1:
        raise ValueError("Exactly one package build provenance required")
    identity = provenance[0]
    dirty = identity.get("source_dirty")
    permitted = dirty is False or (allow_dirty_candidate and dirty is True)
    if (identity.get("schema") != "rswm-build-info-v1" or identity.get("source_commit") != expected_commit or
            identity.get("source_code_sha256") != expected_source or not permitted):
        raise ValueError("Final archive must name the supplied clean commit and source fingerprint")
    if not metadata or any(item["name"] != PROJECT or item["version"] != VERSION for item in metadata):
        raise ValueError("Package project/version metadata mismatch")
    return {"path": str(path), "sha256": file_hash(path), "bytes": path.stat().st_size,
            "status": "VERIFIED_DIRTY_ENGINEERING_CANDIDATE" if dirty else "PASS", "source_code_sha256": calculated, "build_info": identity,
            "project_metadata": metadata, "member_count": len(files), "members": sorted(files, key=lambda row: row["path"])}


class Commands:
    def __init__(self, out, cwd, environment):
        self.out, self.cwd, self.environment, self.rows = out, cwd, environment, []

    def run(self, label, command, timeout=600.):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise ValueError("Unsafe command log label")
        started = time.monotonic()
        process = None
        try:
            process = subprocess.Popen(command, cwd=self.cwd, env=self.environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True, close_fds=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                code, timed_out = process.returncode, False
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=5.)
                code, timed_out = process.returncode, True
        except OSError as error:
            stdout, stderr, code, timed_out = "", str(error), None, False
        finally:
            if process is not None and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5.)
        stdout_path = self.out / (label + ".stdout.log")
        stderr_path = self.out / (label + ".stderr.log")
        stdout_path.write_text(stdout, encoding="utf-8"); stderr_path.write_text(stderr, encoding="utf-8")
        row = {"label": label, "argv": [str(value) for value in command], "cwd": str(self.cwd),
               "returncode": code, "timeout": timed_out, "elapsed_seconds": time.monotonic() - started,
               "stdout": stdout_path.name, "stdout_sha256": file_hash(stdout_path),
               "stderr": stderr_path.name, "stderr_sha256": file_hash(stderr_path)}
        self.rows.append(row)
        save_json(self.out / "commands.json", self.rows)
        return row, stdout


PROVENANCE_SCRIPT = """import importlib.metadata,json,pathlib,sys
import rswm_inria
from rswm_inria.acceptance import build_identity
dist=importlib.metadata.distribution('rswm-inria-120')
identity=build_identity()
print(json.dumps({'distribution_version':dist.version,'package_version':rswm_inria.__version__,
 'package_file':rswm_inria.__file__,'prefix':sys.prefix,'executable':sys.executable,'identity':identity,
 'console_scripts':sorted(ep.name for ep in dist.entry_points if ep.group=='console_scripts'),
 'imported_module_paths':{name:value.__file__ for name,value in sorted(sys.modules.items()) if name.startswith('rswm_inria') and getattr(value,'__file__',None)}}))
"""


def run_acceptance(args, out, work):
    summaries, checks = {}, []
    for label, value in (("wheel", args.wheel), ("sdist", args.sdist)):
        summary = scan_archive(value, args.commit, args.source_fingerprint,
                               allow_dirty_candidate=getattr(args, "allow_dirty_candidate", False))
        save_json(out / (label + "-inventory.json"), summary)
        summaries[label] = {key: summary[key] for key in ("path", "sha256", "bytes", "status", "source_code_sha256", "build_info", "member_count")}
    venv, cwd = work / "venv", work / "outside-checkout"
    cwd.mkdir(); (work / "home").mkdir(); (work / "tmp").mkdir()
    environment = {"PATH": os.defpath, "HOME": str(work / "home"), "TMPDIR": str(work / "tmp"),
                   "LANG": "C.UTF-8", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "PIP_NO_INDEX": "1",
                   "PIP_NO_INPUT": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "HF_HUB_OFFLINE": "1",
                   "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""}
    commands = Commands(out, cwd, environment)
    command, _ = commands.run("new-venv", [sys.executable, "-I", "-m", "venv", str(venv)])
    if not succeeded(command):
        raise ValueError("New venv creation failed")
    bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
    python = bin_dir / ("python.exe" if os.name == "nt" else "python")
    environment["PATH"] = str(bin_dir) + os.pathsep + os.defpath
    install_from = getattr(args, "install_from", "wheel")
    if install_from == "sdist":
        command, _ = commands.run("offline-sdist-build-dependencies", [str(python), "-I", "-m", "pip", "install",
            "--no-index", "--find-links", str(args.dependencies), "--only-binary=:all:", "--no-compile",
            "setuptools>=68", "wheel"])
        if not succeeded(command):
            raise ValueError("Offline sdist build dependencies unavailable")
    install = [str(python), "-I", "-m", "pip", "install", "--no-index", "--find-links", str(args.dependencies),
               "--no-compile"]
    if install_from == "wheel":
        install.extend(["--only-binary=:all:", str(args.wheel)])
    else:
        install.extend(["--no-build-isolation", "--no-binary=" + PROJECT, str(args.sdist)])
    command, _ = commands.run("offline-" + install_from + "-install", install)
    if not succeeded(command):
        raise ValueError("Offline " + install_from + "/core dependency installation failed")
    command, stdout = commands.run("installed-provenance", [str(python), "-I", "-c", PROVENANCE_SCRIPT])
    if not succeeded(command):
        raise ValueError("Installed import/build provenance check failed")
    installed = json.loads(stdout); identity = installed["identity"]
    if (installed["distribution_version"] != VERSION or installed["package_version"] != VERSION or
            Path(installed["prefix"]).resolve() != venv.resolve() or
            identity["code_commit"] != args.commit or identity["source_code_sha256"] != args.source_fingerprint or
            not (identity["build_info"].get("source_dirty") is False or
                 (getattr(args, "allow_dirty_candidate", False) and identity["build_info"].get("source_dirty") is True))):
        raise ValueError("Installed package identity differs from supplied final clean checkpoint")
    if not installed["imported_module_paths"] or any(not Path(path).resolve().is_relative_to(venv.resolve())
            for path in installed["imported_module_paths"].values()):
        raise ValueError("An installed module was imported from outside the new venv")
    save_json(out / "installed-provenance.json", installed)
    checks.append({"name": "installed-provenance", "status": "PASS"})
    console_names = installed["console_scripts"]
    if not console_names or not {"rswm-inria", "rswm-suite", "rswm-export", "rswm-worker", "rswm-acceptance"}.issubset(console_names):
        raise ValueError("Required installed console scripts missing")
    for name in console_names:
        if not re.fullmatch(r"rswm-[A-Za-z0-9_.-]+", name):
            raise ValueError("Unexpected console name in project metadata")
        command, _ = commands.run(name + "-help", [str(bin_dir / name), "--help"], timeout=30.)
        checks.append({"name": name + "-help", "status": "PASS" if succeeded(command) else "FAIL"})
    cpu = work / "cpu-synthetic"
    command, _ = commands.run("default-cpu-acceptance", [str(bin_dir / "rswm-acceptance"), "--out", str(cpu)])
    if not succeeded(command):
        raise ValueError("Default installed CPU acceptance failed")
    receipt = read_json(cpu / "acceptance.json")
    if (receipt["status"] != "PASS_SOFTWARE_DIAGNOSTIC" or receipt["expected_runs"] != 2 or
            receipt["completed_runs"] != 2 or receipt["planned_outputs"] != 192 or receipt["verified_outputs"] != 192 or
            receipt["identity"]["code_commit"] != args.commit or receipt["identity"]["source_code_sha256"] != args.source_fingerprint or
            any(receipt.get(field) != "NOT_RUN" for field in ("real_data_validation", "real_model_validation", "real_gpu_validation"))):
        raise ValueError("Default CPU receipt is not the expected F2/F7 192-output diagnostic")
    plan = read_json(cpu / "plan.json")
    if sorted(row["family"] for row in plan["expected"]) != ["F2", "F7"]:
        raise ValueError("Default CPU suite families differ from F2/F7")
    resources = [read_json(path) for path in sorted((cpu / "suite").rglob("resources.json"))]
    if (len(resources) != 2 or any(item.get("measurement_complete") is not True for item in resources) or
            sum(item["accounting_calls"]["valid_outputs"] for item in resources) != 192 or
            sum(item["accounting_calls"]["planned_output_slots"] for item in resources) != 192):
        raise ValueError("Actual native resource receipts or output accounting incomplete")
    save_json(out / "cpu-acceptance.json", receipt)
    save_json(out / "resource-summary.json", [{"schema": item["schema"], "measurement_complete": item["measurement_complete"],
        "accounting_calls": item["accounting_calls"], "measurements": item["measurements"],
        "adapter_counters": item["adapter_counters"]} for item in resources])
    command, _ = commands.run("installed-suite-replay", [str(bin_dir / "rswm-suite"), "verify", "--plan", str(cpu / "plan.json"),
        "--suite", str(cpu / "suite"), "--out", str(work / "verified-records.json")])
    if not succeeded(command) or read_json(work / "verified-records.json").get("accepted") is not True:
        raise ValueError("Actual installed CLI replay failed")
    exported = work / "cli-export"
    command, _ = commands.run("installed-table-export", [str(bin_dir / "rswm-export"), "--plan", str(cpu / "plan.json"),
        "--suite", str(cpu / "suite"), "--out", str(exported), "--bootstrap", "5", "--seed", "7"])
    required = ("tables/main_control.tex", "tables/main_persistence.tex", "tables/coverage.tex", "tables/resources.tex",
                "run-completeness.csv", "cells.csv", "provenance.json", "integrity-summary.json")
    if not succeeded(command) or any(not (exported / name).is_file() for name in required):
        raise ValueError("Installed export did not produce required real tables")
    (out / "tables").mkdir()
    for name in required:
        source = exported / name
        target = out / (name if name.startswith("tables/") else "tables/" + name)
        shutil.copyfile(source, target)
    checks.append({"name": "default-generation-replay-resources-export", "status": "PASS", "planned_outputs": 192, "verified_outputs": 192})
    gates = {}
    for profile in ("real-model-pilot", "isolated-release"):
        destination = work / profile
        command, stdout = commands.run(profile, [str(bin_dir / "rswm-acceptance"), "--profile", profile, "--out", str(destination)])
        gate = read_json(destination / "acceptance.json") if (destination / "acceptance.json").is_file() else json.loads(stdout)
        save_json(out / (profile + ".json"), gate); gates[profile] = gate
        expected = command["returncode"] == 2 and not command["timeout"] and gate.get("status") == "BLOCKED"
        if profile == "isolated-release":
            expected = expected and gate.get("gate", {}).get("protocol_security_verified") is False
        checks.append({"name": profile + "-honest-blocked", "status": "PASS" if expected else "FAIL", "returncode": command["returncode"], "gate_status": gate.get("status")})
    artifacts = {str(path.relative_to(cpu)): file_hash(path) for path in sorted(cpu.rglob("*")) if path.is_file()}
    save_json(out / "cpu-artifact-hashes.json", artifacts)
    return {"archives": summaries, "installed": installed, "checks": checks, "console_script_count": len(console_names),
            "default_cpu": {"expected_runs": 2, "completed_runs": 2, "planned_outputs": 192, "verified_outputs": 192},
            "gates": {name: {"status": value.get("status"), "publication_acceptance": value.get("publication_acceptance")} for name, value in gates.items()},
            "os_security_acceptance": gates["isolated-release"].get("status"),
            "commands": commands.rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True); parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--install-from", choices=("wheel", "sdist"), default="wheel",
                        help="Install retained wheel or actually rebuild/install retained sdist offline")
    parser.add_argument("--out", required=True); parser.add_argument("--work", required=True)
    parser.add_argument("--commit", required=True); parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--allow-dirty-candidate", action="store_true",
                        help="Verify a modified engineering snapshot against its base commit and actual fingerprint; never a clean final release")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.commit) or not re.fullmatch(r"[0-9a-f]{64}", args.source_fingerprint):
        parser.error("--commit and --source-fingerprint require full lowercase SHA identities")
    args.wheel, args.sdist = args.wheel.resolve(), args.sdist.resolve()
    args.dependencies = (args.dependencies or args.wheel.parent).resolve()
    if not args.wheel.is_file() or not args.sdist.is_file() or not args.dependencies.is_dir():
        parser.error("Supplied final wheel, sdist and offline dependency directory must exist")
    checkout = Path(__file__).resolve().parents[1]
    work_path, out_path = Path(args.work).resolve(), Path(args.out).resolve()
    if work_path.is_relative_to(checkout) or out_path.is_relative_to(work_path) or work_path.is_relative_to(out_path):
        parser.error("--work must be outside checkout and separate from compact --out")
    out = fresh_directory(out_path, "--out"); work = fresh_directory(work_path, "--work")
    summary = {"schema": "rswm-installed-release-acceptance-v1", "expected_version": VERSION,
               "expected_source_commit": args.commit, "expected_source_code_sha256": args.source_fingerprint,
               "work_directory": str(work), "raw_assets_location": "work only; not copied into compact evidence",
               "real_data_validation": "NOT_RUN", "real_model_validation": "NOT_RUN", "real_gpu_validation": "NOT_RUN",
               "os_security_acceptance": "NOT_RUN", "publication_acceptance": "NOT_READY",
               "installed_from": args.install_from, "sdist_offline_rebuild": "NOT_RUN",
               "engineering_candidate_mode": args.allow_dirty_candidate,
               "clean_final_release_asserted": not args.allow_dirty_candidate}
    try:
        summary.update(run_acceptance(args, out, work))
        if args.install_from == "sdist":
            summary["sdist_offline_rebuild"] = "PASS"
        success = all(check["status"] == "PASS" for check in summary["checks"])
        summary["status"] = (("PASS_DIRTY_ENGINEERING_CANDIDATE_SOFTWARE_ONLY" if args.allow_dirty_candidate else
                              "PASS_INSTALLED_SOFTWARE_WITH_REAL_GATES_BLOCKED") if success else "FAIL")
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        summary.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    summary["evidence_sha256"] = {str(path.relative_to(out)): file_hash(path) for path in sorted(out.rglob("*")) if path.is_file()}
    save_json(out / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "summary": str(out / "summary.json"), "summary_sha256": file_hash(out / "summary.json")}, sort_keys=True))
    return 0 if summary["status"] in {"PASS_INSTALLED_SOFTWARE_WITH_REAL_GATES_BLOCKED", "PASS_DIRTY_ENGINEERING_CANDIDATE_SOFTWARE_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
