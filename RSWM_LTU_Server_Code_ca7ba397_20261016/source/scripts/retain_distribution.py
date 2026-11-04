#!/usr/bin/env python3
"""Build, retain and independently read back the actual engineering distributions.

Checksums attest retained bytes, not research results or permission to use data.
The externally supplied commit and optional source fingerprint bind read-back to
the intended source revision. This script never publishes a release or tag.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import shutil
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

# This is the existing installed acceptance implementation, not another runner.
from reproduce_installed_release import (file_hash, fresh_directory,
                                        read_json, save_json, scan_archive)

SCHEMA = "rswm-retained-distribution-v1"
MAX_BUNDLE_BYTES = 512 * 1024 ** 2
GENERATED_WHEEL_METADATA = {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt"}
GENERATED_SDIST_METADATA = {"PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt",
                            "requires.txt", "top_level.txt"}


def source_identity(root):
    import setuptools
    original = setuptools.setup
    try:
        setuptools.setup = lambda **kwargs: None
        return runpy.run_path(str(root / "setup.py"))["provenance"](root)
    finally:
        setuptools.setup = original


def worktree_status(root):
    return subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines()


def dependency_environment():
    return {"python": sys.version, "executable": sys.executable,
            "platform": sys.platform,
            "packages": sorted(({"name": row.metadata["Name"], "version": row.version}
                                for row in importlib.metadata.distributions()), key=lambda row: (row["name"].lower(), row["version"]))}


def source_input_mapping(summary, kind):
    """Resolve the provenance's inputs to bytes actually present in each archive."""
    inputs = summary["build_info"]["source_archive_inputs"]
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Missing authored source archive inputs")
    mapping, observed, roots = [], {}, set()
    for member in summary["members"]:
        path = PurePosixPath(member["path"])
        roots.add(path.parts[0])
        source = None
        generated = False
        if kind == "sdist" and len(path.parts) > 1:
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative in inputs:
                source = relative
            generated = (relative in {"PKG-INFO", "setup.cfg", "src/rswm_inria/_build_info.json"} or
                         (len(path.parts) == 4 and path.parts[1] == "src" and path.parts[2].endswith(".egg-info")
                          and path.name in GENERATED_SDIST_METADATA))
        elif kind == "wheel" and path.parts[0] == "rswm_inria" and path.suffix == ".py":
            source = "src/" + path.as_posix()
        elif (kind == "wheel" and len(path.parts) > 2 and path.parts[:2] == ("rswm_inria", "configs")
              and path.suffix == ".json"):
            source = "src/" + path.as_posix()
        elif (kind == "wheel" and len(path.parts) == 3 and path.parts[0].endswith(".dist-info")
              and path.parts[1] == "licenses" and path.name == "NOTICE.md"):
            source = path.name
        if kind == "wheel":
            generated = (path.as_posix() == "rswm_inria/_build_info.json" or
                         (len(path.parts) == 2 and path.parts[0].endswith(".dist-info")
                          and path.name in GENERATED_WHEEL_METADATA))
        if source is not None:
            if source in observed or source not in inputs or member["sha256"] != inputs[source]:
                raise ValueError("Actual packaged source input differs from build provenance: " + member["path"])
            observed[source] = member["sha256"]
        elif not generated:
            raise ValueError("Unexpected unrecorded packaged input: " + member["path"])
        mapping.append({**member, "source_input": source,
                        "role": "authored-input" if source is not None else "generated-package-metadata"})
    required = inputs if kind == "sdist" else {name: sha for name, sha in inputs.items()
                                                if (name.startswith("src/rswm_inria/") and name.endswith(".py"))
                                                or (name.startswith("src/rswm_inria/configs/") and name.endswith(".json"))
                                                or name == "NOTICE.md"}
    if observed != required:
        raise ValueError("Required authored inputs missing from actual " + kind)
    if kind == "sdist" and len(roots) != 1:
        raise ValueError("Source archive must have one package root")
    return mapping


def inspect_distributions(dist, commit, source):
    files = sorted(path for path in dist.iterdir() if path.is_file())
    wheels = [path for path in files if path.name.endswith(".whl")]
    sdists = [path for path in files if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1 or len(files) != 2:
        raise ValueError("Exactly one wheel and one sdist, and no unexpected dist files, required")
    rows = []
    for kind, path in (("wheel", wheels[0]), ("sdist", sdists[0])):
        summary = scan_archive(path, commit, source)
        rows.append({"kind": kind, "path": "dist/" + path.name,
                     "sha256": summary["sha256"], "bytes": summary["bytes"],
                     "build_info": summary["build_info"],
                     "packaged_inputs": source_input_mapping(summary, kind)})
    if rows[0]["build_info"] != rows[1]["build_info"]:
        raise ValueError("Wheel and sdist do not share the same build identity")
    return rows


def retained_files(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Retained artifacts must not contain symbolic links")
        relative = path.relative_to(root).as_posix()
        if path.is_file() and relative not in {"build-manifest.json", "SHA256SUMS"}:
            result[relative] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    return result


def build_retained(args):
    root = args.root.resolve()
    if Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root, text=True).strip()).resolve() != root:
        raise ValueError("Build root must be the intended Git checkout")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != args.commit:
        raise ValueError("Build HEAD differs from the explicitly supplied commit")
    before, status = source_identity(root), worktree_status(root)
    if before["source_dirty"] is not False or status:
        raise ValueError("Final retained build requires a clean committed worktree")
    out = Path(args.out).resolve()
    if out.is_relative_to(root):
        raise ValueError("Retained build output must be outside source checkout")
    fresh_directory(out, "--out")
    (out / "dist").mkdir()
    command = [sys.executable, "-m", "build", "--no-isolation", "--outdir", str(out / "dist"), str(root)]
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True)
    (out / "build.stdout.log").write_text(result.stdout)
    (out / "build.stderr.log").write_text(result.stderr)
    if result.returncode:
        raise ValueError("Distribution build failed; logs retained under --out")
    after = source_identity(root)
    if after != before or worktree_status(root):
        raise ValueError("Source/worktree changed during distribution build")
    archives = inspect_distributions(out / "dist", head, before["source_code_sha256"])
    (out / "dependencies").mkdir()
    if args.dependencies:
        dependencies = args.dependencies.resolve()
        if not dependencies.is_dir():
            raise ValueError("--dependencies must be an existing directory of offline wheels")
        for path in sorted(dependencies.glob("*.whl")):
            if path.is_symlink():
                raise ValueError("Dependency wheels must be retained ordinary files")
            shutil.copyfile(path, out / "dependencies" / path.name)
    manifest = {"schema": SCHEMA, "source_commit": head, "source_dirty": False,
                "worktree_dirty": False, "source_code_sha256": before["source_code_sha256"],
                "source_archive_inputs": before["source_archive_inputs"],
                "archives": archives, "environment": dependency_environment(),
                "build": {"argv": command, "returncode": result.returncode,
                          "elapsed_seconds": time.monotonic() - started},
                "scope": "Internal engineering distribution; research/model/GPU/publication acceptance is not asserted.",
                "retained_files": retained_files(out)}
    save_json(out / "build-manifest.json", manifest)
    hashes = {**manifest["retained_files"], "build-manifest.json": {"sha256": file_hash(out / "build-manifest.json")}}
    (out / "SHA256SUMS").write_text("".join(row["sha256"] + "  " + name + "\n" for name, row in sorted(hashes.items())))
    verify_retained(out, args.commit, before["source_code_sha256"])
    return manifest


def safe_relative(name):
    return (bool(name) and not name.startswith("/") and "\\" not in name and "\0" not in name
            and all(part not in {"", ".", ".."} for part in name.split("/")))


def verify_retained(root, commit, source=None):
    root = Path(root)
    manifest = read_json(root / "build-manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("source_commit") != commit or
            manifest.get("source_dirty") is not False or manifest.get("worktree_dirty") is not False):
        raise ValueError("Retained build identity must match the supplied clean commit")
    fingerprint = manifest["source_code_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint) or (source and source != fingerprint):
        raise ValueError("Retained package source fingerprint mismatch")
    expected = manifest["retained_files"]
    if not isinstance(expected, dict) or any(not safe_relative(name) for name in expected):
        raise ValueError("Unsafe retained file inventory")
    observed = retained_files(root)
    if observed != expected:
        raise ValueError("Retained bytes missing, changed, duplicated or unexpected")
    required = {"build.stdout.log", "build.stderr.log"}
    if not required.issubset(observed):
        raise ValueError("Required build command evidence missing")
    sums = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        if not re.fullmatch(r"[0-9a-f]{64}  .+", line):
            raise ValueError("Malformed SHA256SUMS")
        sha, name = line.split("  ", 1)
        if not safe_relative(name) or name in sums:
            raise ValueError("Unsafe or duplicate SHA256SUMS entry")
        sums[name] = sha
    wanted = {name: row["sha256"] for name, row in observed.items()}
    wanted["build-manifest.json"] = file_hash(root / "build-manifest.json")
    if sums != wanted:
        raise ValueError("SHA256SUMS must identify all retained bytes and the build manifest")
    archives = inspect_distributions(root / "dist", commit, fingerprint)
    if archives != manifest["archives"]:
        raise ValueError("Actual archive byte/input mapping differs from retained build manifest")
    if archives[0]["build_info"]["source_archive_inputs"] != manifest["source_archive_inputs"]:
        raise ValueError("Retained authored source identity mismatch")
    return manifest


def pack_retained(args):
    verify_retained(args.retained, args.commit, args.source_fingerprint)
    bundle = args.bundle.resolve()
    if bundle.exists() or bundle.is_relative_to(args.retained.resolve()):
        raise ValueError("--bundle must be a new file outside retained directory")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.retained.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(args.retained).as_posix())
    return {"artifact_zip": str(bundle), "artifact_zip_sha256": file_hash(bundle), "bytes": bundle.stat().st_size}


def readback_bundle(bundle, expected_zip_sha, destination):
    if file_hash(bundle) != expected_zip_sha:
        raise ValueError("Downloaded artifact ZIP SHA256 differs from retained/upload digest")
    destination = fresh_directory(destination, "--readback")
    total, seen = 0, set()
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            name = member.filename
            mode = (member.external_attr >> 16) & 0xffff
            if (not safe_relative(name) or name in seen or stat.S_ISLNK(mode) or
                    stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError("Unsafe or duplicate retained ZIP member")
            seen.add(name)
            total += member.file_size
            if total > MAX_BUNDLE_BYTES or member.file_size > MAX_BUNDLE_BYTES:
                raise ValueError("Retained bundle exceeds read-back byte budget")
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    return destination


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def download_artifact(args):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository) or not re.fullmatch(r"[0-9]+", args.artifact_id):
        raise ValueError("Invalid GitHub repository or artifact id")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("GH_TOKEN is required to download the requested CI artifact")
    target = args.bundle.resolve()
    if target.exists():
        raise ValueError("Download destination must be a new file")
    target.parent.mkdir(parents=True, exist_ok=True)
    api = "https://api.github.com/repos/" + args.repository + "/actions/artifacts/" + args.artifact_id + "/zip"
    request = urllib.request.Request(api, headers={"Authorization": "Bearer " + token,
              "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
    # Never forward the API bearer token to the artifact storage redirect.
    try:
        response = urllib.request.build_opener(NoRedirect).open(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code not in {301, 302, 303, 307, 308}:
            raise ValueError("GitHub artifact download failed with HTTP " + str(error.code)) from None
        location = error.headers.get("Location", "")
        if not location.startswith("https://"):
            raise ValueError("Artifact redirect must use HTTPS")
        response = urllib.request.urlopen(location, timeout=60)
    with response, target.open("xb") as output:
        count = 0
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            count += len(block)
            if count > MAX_BUNDLE_BYTES:
                raise ValueError("Artifact download exceeds byte budget")
            output.write(block)
    actual = file_hash(target)
    if actual != args.bundle_sha256:
        raise ValueError("Downloaded artifact ZIP SHA256 does not match upload-artifact digest")
    return {"repository": args.repository, "artifact_id": args.artifact_id,
            "artifact_zip_sha256": actual, "bytes": target.stat().st_size}


def accept_readback(args):
    readback = readback_bundle(args.bundle, args.bundle_sha256, args.readback)
    manifest = verify_retained(readback, args.commit, args.source_fingerprint)
    out = fresh_directory(args.out, "--out")
    work = Path(args.work).resolve()
    checkout = Path(__file__).resolve().parents[1]
    if work.is_relative_to(checkout) or work.is_relative_to(out) or out.is_relative_to(work):
        raise ValueError("Acceptance work must be outside checkout and separate from retained evidence")
    fresh_directory(work, "--work")
    archives = {row["kind"]: readback / row["path"] for row in manifest["archives"]}
    checks = []
    for kind in ("wheel", "sdist"):
        command = [sys.executable, str(Path(__file__).with_name("reproduce_installed_release.py")),
                   "--wheel", str(archives["wheel"]), "--sdist", str(archives["sdist"]),
                   "--dependencies", str(readback / "dependencies"), "--install-from", kind,
                   "--out", str(out / kind), "--work", str(work / kind), "--commit", args.commit,
                   "--source-fingerprint", manifest["source_code_sha256"]]
        result = subprocess.run(command, capture_output=True, text=True)
        (out / (kind + ".stdout.log")).write_text(result.stdout)
        (out / (kind + ".stderr.log")).write_text(result.stderr)
        summary = read_json(out / kind / "summary.json") if (out / kind / "summary.json").is_file() else {}
        checks.append({"kind": kind, "argv": command, "returncode": result.returncode,
                       "status": summary.get("status", "FAIL")})
    receipt = {"schema": "rswm-retained-readback-acceptance-v1", "source_commit": args.commit,
               "source_code_sha256": manifest["source_code_sha256"],
               "artifact_zip_sha256": file_hash(args.bundle), "artifact_zip_bytes": args.bundle.stat().st_size,
               "distribution_hashes": {row["kind"]: row["sha256"] for row in manifest["archives"]},
               "status": "PASS_RETAINED_INSTALLED_SOFTWARE" if all(row["returncode"] == 0 for row in checks) else "FAIL",
               "checks": checks, "real_model_validation": "NOT_RUN", "real_gpu_validation": "NOT_RUN",
               "publication_acceptance": "NOT_READY"}
    save_json(out / "readback-summary.json", receipt)
    if receipt["status"] == "FAIL":
        raise ValueError("Downloaded wheel/sdist installed acceptance failed; command evidence retained")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--dependencies", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("--retained", type=Path, required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--retained", type=Path, required=True)
    pack.add_argument("--bundle", type=Path, required=True)
    download = commands.add_parser("download-artifact")
    download.add_argument("--repository", required=True)
    download.add_argument("--artifact-id", required=True)
    download.add_argument("--bundle", type=Path, required=True)
    download.add_argument("--bundle-sha256", required=True)
    accept = commands.add_parser("accept")
    accept.add_argument("--bundle", type=Path, required=True)
    accept.add_argument("--bundle-sha256", required=True)
    accept.add_argument("--readback", type=Path, required=True)
    accept.add_argument("--out", type=Path, required=True)
    accept.add_argument("--work", type=Path, required=True)
    for command in (build, verify, pack, accept):
        command.add_argument("--commit", required=True)
    for command in (verify, pack, accept):
        command.add_argument("--source-fingerprint")
    args = parser.parse_args(argv)
    if hasattr(args, "commit") and not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit requires the full expected lowercase commit SHA")
    if hasattr(args, "source_fingerprint") and args.source_fingerprint and not re.fullmatch(r"[0-9a-f]{64}", args.source_fingerprint):
        parser.error("--source-fingerprint requires a lowercase SHA256")
    if hasattr(args, "bundle_sha256") and not re.fullmatch(r"[0-9a-f]{64}", args.bundle_sha256):
        parser.error("--bundle-sha256 requires the independently retained lowercase ZIP SHA256")
    try:
        if args.command == "build":
            result = build_retained(args)
        elif args.command == "verify":
            result = verify_retained(args.retained, args.commit, args.source_fingerprint)
        elif args.command == "pack":
            result = pack_retained(args)
        elif args.command == "download-artifact":
            result = download_artifact(args)
        else:
            result = accept_readback(args)
        print(json.dumps({"status": result.get("status", "PASS"), "source_commit": result.get("source_commit"),
                          "source_code_sha256": result.get("source_code_sha256"),
                          "artifact_zip_sha256": result.get("artifact_zip_sha256")}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "FAIL", "error_type": type(error).__name__, "error": str(error)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
