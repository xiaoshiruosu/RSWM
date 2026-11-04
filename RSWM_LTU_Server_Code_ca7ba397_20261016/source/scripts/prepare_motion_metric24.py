"""Bounded real Inria preparation and one preregistered motion validation attempt.

This does not create a smoke12/core120 materialization or authorize a formal
model experiment. Existing provisional instrument defaults are explicitly
registered as this attempt's candidate, not as project-owner approved values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from rswm_inria.data import (INRIA_REVISION, REPO, build_world, source_pixels,
                             validate_source_file, verify_manifest, write_json)
from rswm_inria.geometry import digest, file_hash
from rswm_inria.materialized import independent_world, read_json
from rswm_inria.motion_protocol import motion_config, validate_release
from rswm_inria.suite import code_fingerprint, validate_plan

MANIFEST_SHA = "4ef2dffd76c962ec68ac1b0797b039363ef163f23053552cc149fb239d4ec51d"
INVENTORY_FILE_SHA = "3ec82f07e387eda4fc8af253aff40f7889100c8f06fe68d29f99773b692eacea"
ROLE_CITIES = {"metric_calibration": "bellingham", "metric_validation": "bloomington"}
OUTPUTS = {"F2": 32, "F3": 64, "F5": 96, "F7": 160}
RGB_BYTES = 240 * 416 * 3
SOURCE_BYTES = 1_800_968_676
WORLD_RGB_BYTES = 24 * 2048 * 2048 * 3
PAIR_RGB_BYTES = 24 * (2 * 256 - 19) * RGB_BYTES
SAFETY_BYTES = 1_000_000_000


def no_symlinks(path):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("Parent traversal paths are not accepted")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink inputs and output ancestors are not accepted")
    return path


def write(path, value):
    path = no_symlinks(path)
    write_json(path, value)


def new_invocation(evidence):
    folder = no_symlinks(evidence / "invocations")
    folder.mkdir(parents=True, exist_ok=True)
    for number in range(10000):
        target = folder / f"invocation-{number:04d}"
        try:
            target.mkdir()
        except FileExistsError:
            continue
        return target
    raise FileExistsError("Too many preserved invocations")


def copy_exclusive(source, target):
    source, target = no_symlinks(source), no_symlinks(target)
    data = source.read_bytes()
    if target.exists():
        if target.read_bytes() != data:
            raise FileExistsError("Refusing to replace differing copied evidence")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(data)


def frozen_selection(manifest_path, inventory_path, max_source_gb):
    manifest_path, inventory_path = map(no_symlinks, (manifest_path, inventory_path))
    if not math.isfinite(max_source_gb) or max_source_gb <= 0:
        raise ValueError("Source budget must be positive and finite")
    if file_hash(inventory_path) != INVENTORY_FILE_SHA:
        raise ValueError("Inventory differs from the fixed complete 542-file snapshot")
    manifest, inventory = read_json(manifest_path), read_json(inventory_path)
    verify_manifest(manifest, inventory)
    if (manifest["manifest_sha256"] != MANIFEST_SHA or
            manifest["revision"] != INRIA_REVISION or manifest["repo_id"] != REPO or
            len(inventory["files"]) != 542):
        raise ValueError("Fixed complete Inria source identity differs")
    rows = [row for row in manifest["worlds"] if row["role"] in ROLE_CITIES]
    for role, city in ROLE_CITIES.items():
        subset = [row for row in rows if row["role"] == role]
        if len(subset) != 12 or {row["city"] for row in subset} != {city}:
            raise ValueError("Frozen twelve-world motion role quota differs")
    if len(rows) != 24 or len({row["world_id"] for row in rows}) != 24:
        raise ValueError("Repeated or missing metric world")
    if sum(row["size_bytes"] for row in rows) != SOURCE_BYTES:
        raise ValueError("Frozen source payload differs")
    if SOURCE_BYTES > max_source_gb * 1_000_000_000:
        raise ValueError("Metric24 source payload exceeds the explicit source budget")
    return manifest, rows


def remaining_smoke_slots(plan, suite):
    """Release space only for authenticated, completed first attempts."""
    remaining = 0
    completed = 0
    for row in plan["expected"]:
        slots = OUTPUTS[row["family"]]
        path = no_symlinks(Path(suite) / "runs" / row["run_id"] / "attempt-000" / "outcome.json")
        outcome = None
        if path.is_file():
            try:
                outcome = read_json(path)
            except (OSError, ValueError):
                pass  # A partial/invalid outcome never releases reserved space.
        done = False
        if isinstance(outcome, dict):
            try:
                receipt = read_json(no_symlinks(Path(suite) / "receipts" / (row["run_id"] + "-000.json")))
                head = read_json(no_symlinks(Path(suite) / "heads" / (row["run_id"] + "-000.json")))
                manifest = read_json(no_symlinks(path.parent / "artifact-manifest.json"))
                resource_path = no_symlinks(path.parent / "resources.json")
                resource = read_json(resource_path)
                identity = {"plan_sha256": plan["plan_sha256"], "run_id": row["run_id"], "attempt": 0}
                outcome_sha = file_hash(path)
                replay_head = head.get("journal_head")
                done = all(obj.get(key) == value for obj in (outcome, receipt, head, manifest, resource)
                           for key, value in identity.items()) and all((
                    outcome.get("status") == "COMPLETED",
                    outcome.get("execution_status") == "COMPLETED",
                    outcome.get("submission_status") == "VERIFIED",
                    outcome.get("resource_status") == "COMPLETE",
                    outcome.get("replay_verified") is True,
                    resource.get("measurement_complete") is True,
                    receipt.get("status") == "COMPLETED",
                    receipt.get("outcome_sha256") == outcome_sha,
                    isinstance(replay_head, str) and len(replay_head) == 64 and
                        all(char in "0123456789abcdef" for char in replay_head),
                    receipt.get("replay_head") == outcome.get("replay_head") == replay_head,
                    manifest.get("files", {}).get("outcome.json") == outcome_sha,
                    manifest.get("files", {}).get("resources.json") == file_hash(resource_path),
                ))
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass  # Missing, replaced or inconsistent receipts keep the reservation.
        if done:
            completed += 1
        else:
            remaining += slots
    return remaining, completed


def disk_preflight(smoke_root, target, additional_bytes):
    smoke_root = no_symlinks(smoke_root)
    plan_path = smoke_root / "suite" / "plan.json"
    plan = read_json(no_symlinks(plan_path))
    validate_plan(plan)
    if (len(plan["expected"]) != 576 or sum(OUTPUTS[r["family"]] for r in plan["expected"]) != 50688 or
            plan.get("retry_policy", {}).get("max_attempts_per_run") != 1 or
            {r["source_manifest_sha256"] for r in plan["expected"]} != {MANIFEST_SHA}):
        raise ValueError("Reservation requires the actual frozen full smoke12 first-attempt matrix")
    target = no_symlinks(target)
    existing = target
    while not existing.exists():
        existing = existing.parent
    if existing.stat().st_dev != smoke_root.stat().st_dev:
        raise ValueError("This reservation calculation requires the same filesystem")
    slots, completed = remaining_smoke_slots(plan, smoke_root / "suite")
    reserve = math.ceil(slots * RGB_BYTES * 1.25)
    free = shutil.disk_usage(existing).free
    body = {"schema": "rswm-motion-metric24-disk-preflight-v1",
            "smoke_root": str(smoke_root), "smoke_plan_sha256": plan["plan_sha256"],
            "completed_smoke_runs": completed, "remaining_smoke_slots": slots,
            "remaining_smoke_reserve_bytes": reserve, "additional_bytes": additional_bytes,
            "safety_bytes": SAFETY_BYTES, "disk_free_bytes": free,
            "allowed": free >= reserve + additional_bytes + SAFETY_BYTES}
    return {**body, "preflight_sha256": digest(body)}


def preregistration(args, manifest, rows, out, evidence):
    config = motion_config()
    write(evidence / "candidates.json", [config])
    commands = []
    for role, city in ROLE_CITIES.items():
        commands.append([sys.executable, "scripts/calibrate_motion.py", "build-pairs",
                         "--world-index", str(out / "world-index.json"),
                         "--world-root", str(out / "worlds"), "--role", role,
                         "--pairs-per-world", "256", "--seed", "0", "--out", str(out / (city + "-pairs"))])
    commands.extend([
        [sys.executable, "scripts/calibrate_motion.py", "calibrate", "--pair-manifest",
         str(out / "bellingham-pairs/pairs.json"), "--candidates", str(evidence / "candidates.json"),
         "--out", str(out / "calibration")],
        [sys.executable, "scripts/validate_motion.py", "validate", "--pair-manifest",
         str(out / "bloomington-pairs/pairs.json"), "--calibration", str(out / "calibration/calibration.json"),
         "--exposure-ledger", str(no_symlinks(args.exposure_ledger)), "--out", str(out / "validation")],
    ])
    body = {"schema": "rswm-motion-real-attempt-preregistration-v1", "study": "metric24-controlled-inria-v1",
            "scope": "one real validation attempt; not a formal model experiment",
            "candidate_basis": "existing provisional linear_zncc_v1 defaults; not owner-approved final parameters",
            "source_manifest_sha256": manifest["manifest_sha256"], "inventory_file_sha256": INVENTORY_FILE_SHA,
            "repo_id": REPO, "revision": INRIA_REVISION, "source_rows": rows,
            "config": config, "candidate_file_sha256": file_hash(evidence / "candidates.json"),
            "pairs_per_world": 256, "pair_seed": 0, "planned_pairs_per_role": 3072,
            "planned_pairs_total": 6144, "stored_pair_pngs": 11832,
            "source_payload_bytes": SOURCE_BYTES, "max_source_gb": args.max_source_gb,
            "world_uncompressed_rgb_bytes": WORLD_RGB_BYTES, "pair_uncompressed_rgb_bytes": PAIR_RGB_BYTES,
            "disk_reservation": "remaining full-smoke slots * canonical RGB bytes * 1.25, plus 1GB safety",
            "source_code_sha256": code_fingerprint(), "preparation_script_sha256": file_hash(__file__),
            "raw_root": str(out), "source_dir": str(no_symlinks(args.source_dir or out / "source")),
            "persistent_exposure_ledger": str(no_symlinks(args.exposure_ledger)),
            "smoke_root": str(Path(args.smoke_root).absolute()), "commands": commands,
            "validation_policy": "only after CONFIG_FROZEN; once; no retuning using Bloomington or formal test",
            "license_status": manifest["license_status"], "geographic_independence": "not established",
            "real_generation_model": "NOT_RUN", "real_gpu": "NOT_RUN"}
    registration = {**body, "preregistration_sha256": digest(body)}
    write(evidence / "preregistration.json", registration)
    return registration


def prepare(args, manifest, rows, out, evidence, invocation):
    source_dir, world_dir = no_symlinks(args.source_dir or out / "source"), out / "worlds"
    source_existing = source_dir
    while not source_existing.exists():
        source_existing = source_existing.parent
    if source_existing.stat().st_dev != out.stat().st_dev:
        raise ValueError("Shared source cache must use the preflighted filesystem")
    missing = sum(row["size_bytes"] for row in rows if not (source_dir / row["path"]).exists())
    missing_worlds = sum(2048 * 2048 * 3 for row in rows if not (world_dir / (row["world_id"] + ".png")).exists())
    # Additional space includes full uncompressed world data and a small container reserve.
    check = disk_preflight(args.smoke_root, out, missing + missing_worlds + 20_000_000)
    write(invocation / "preparation-disk-preflight.json", check)
    if not check["allowed"]:
        raise OSError("Disk reservation would intrude on the running full smoke12 budget")
    records, checks = [], []
    for i, row in enumerate(rows):
        source = no_symlinks(source_dir / row["path"])
        if not source.exists():
            if not args.download:
                raise FileNotFoundError(source)
            from huggingface_hub import hf_hub_download
            cache_name = source_dir / ".cache" / "huggingface" / "download" / row["path"]
            for cache_path in (cache_name, Path(str(cache_name) + ".metadata"), Path(str(cache_name) + ".lock")):
                no_symlinks(cache_path)
            remote = hf_hub_download(REPO, filename=row["path"], repo_type="dataset",
                                    revision=INRIA_REVISION, local_dir=source_dir, dry_run=True)
            if remote.file_size != row["size_bytes"] or remote.commit_hash != INRIA_REVISION:
                raise ValueError("Remote immutable source size/revision differs before transfer")
            downloaded = no_symlinks(hf_hub_download(REPO, filename=row["path"], repo_type="dataset",
                                      revision=INRIA_REVISION, local_dir=source_dir))
            if downloaded != source:
                raise ValueError("Download path differs from bounded frozen source path")
        validate_source_file(source, row)
        source_array, source_sha, _ = source_pixels(source, row)
        no_symlinks(world_dir / (row["world_id"] + ".png"))
        no_symlinks(world_dir / (row["world_id"] + ".json"))
        record = build_world(source, row, world_dir)
        png = no_symlinks(world_dir / record["path"])
        with Image.open(png) as image:
            if image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("World must be a single RGB PNG")
            actual = np.asarray(image).copy()
        independent = independent_world(source_array)
        if (actual.shape != (2048, 2048, 3) or actual.dtype != np.uint8 or
                not np.array_equal(actual, independent) or file_hash(png) != record["world_file_sha256"] or
                hashlib.sha256(actual.tobytes()).hexdigest() != record["world_pixel_sha256"] or
                source_sha != row["lfs_sha256"] or record["source_sha256"] != source_sha):
            raise ValueError("Independent full-pixel source/world identity mismatch")
        records.append(record)
        checks.append({"world_id": row["world_id"], "city": row["city"], "role": row["role"],
                       "source_path": row["path"], "source_bytes": source.stat().st_size,
                       "source_sha256": source_sha, "world_file_sha256": record["world_file_sha256"],
                       "world_pixel_sha256": record["world_pixel_sha256"], "independent_pixel_equal": True})
        write(evidence / "world-checks" / (row["world_id"] + ".json"), checks[-1])
        print(json.dumps({"prepared": i + 1, "total": 24, "world_id": row["world_id"]}), flush=True)
        del source_array, actual, independent
    index = {"source_manifest_sha256": MANIFEST_SHA, "worlds": records}
    write(out / "world-index.json", index)
    body = {"schema": "rswm-motion-metric24-world-audit-v1", "status": "METRIC24_SOURCE_WORLD_PIXELS_VALIDATED",
            "scope": "metric_calibration12 and metric_validation12; not smoke12/core120/source rights validation",
            "source_manifest_sha256": MANIFEST_SHA, "inventory_file_sha256": INVENTORY_FILE_SHA,
            "source_bytes": SOURCE_BYTES, "source_count": 24, "world_count": 24,
            "world_index_sha256": file_hash(out / "world-index.json"), "world_checks": checks,
            "source_code_sha256": code_fingerprint(), "license_status": manifest["license_status"]}
    write(evidence / "source-world-audit.json", {**body, "audit_sha256": digest(body)})


def verify_prepared_worlds(args, out, evidence, rows):
    audit = read_json(no_symlinks(evidence / "source-world-audit.json"))
    index_path = no_symlinks(out / "world-index.json")
    if (audit.get("schema") != "rswm-motion-metric24-world-audit-v1" or
            audit.get("audit_sha256") != digest({k: v for k, v in audit.items() if k != "audit_sha256"}) or
            audit.get("source_manifest_sha256") != MANIFEST_SHA or
            audit.get("inventory_file_sha256") != INVENTORY_FILE_SHA or
            audit.get("world_index_sha256") != file_hash(index_path)):
        raise ValueError("Prepared source/world audit identity differs")
    index = read_json(index_path)
    if index.get("source_manifest_sha256") != MANIFEST_SHA or len(index.get("worlds", [])) != 24:
        raise ValueError("Prepared world index population differs")
    records = {row["world_id"]: row for row in index["worlds"]}
    checks = {row["world_id"]: row for row in audit.get("world_checks", [])}
    if len(records) != 24 or set(records) != {row["world_id"] for row in rows} or set(checks) != set(records):
        raise ValueError("Prepared world index contains substituted/duplicate worlds")
    source_dir = no_symlinks(args.source_dir or out / "source")
    for row in rows:
        record, check = records[row["world_id"]], checks[row["world_id"]]
        expected = {**row, "source_path": row["path"], "source_sha256": row["lfs_sha256"],
                    "path": row["world_id"] + ".png", "shape": [2048, 2048, 3],
                    "source_crop_xyxy": [452, 452, 4548, 4548], "resampling": "uint16 sum 2x2; (sum+2)//4"}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Prepared source metadata differs from its frozen selected row")
        source, source_sha, _ = source_pixels(no_symlinks(source_dir / row["path"]), row)
        png_path = no_symlinks(out / "worlds" / record["path"])
        with Image.open(png_path) as image:
            if image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Prepared world is no longer canonical PNG")
            actual = np.asarray(image).copy()
        if (not np.array_equal(actual, independent_world(source)) or
                file_hash(png_path) != record["world_file_sha256"] or
                hashlib.sha256(actual.tobytes()).hexdigest() != record["world_pixel_sha256"] or
                any(record[key] != check[key] for key in ("source_sha256", "world_file_sha256", "world_pixel_sha256")) or
                source_sha != row["lfs_sha256"]):
            raise ValueError("Previously audited source/world pixels or hashes changed")


def run_motion(args, out, evidence, registration, invocation):
    check = disk_preflight(args.smoke_root, out, math.ceil(PAIR_RGB_BYTES * 1.01) + 150_000_000)
    write(invocation / "pair-disk-preflight.json", check)
    if not check["allowed"]:
        raise OSError("Pair generation would intrude on the full smoke12 reserve and 1GB safety")
    repository = Path(__file__).absolute().parents[1]
    commands = registration["commands"]

    def command(label, argv):
        if code_fingerprint() != registration["source_code_sha256"]:
            raise ValueError("Package source changed after preregistration")
        started = time.monotonic()
        stdout_path, stderr_path = invocation / (label + ".stdout.log"), invocation / (label + ".stderr.log")
        with stdout_path.open("x") as stdout, stderr_path.open("x") as stderr:
            result = subprocess.run(argv, cwd=repository, stdout=stdout, stderr=stderr)
        write(invocation / (label + "-command.json"), {"argv": argv, "cwd": str(repository),
              "elapsed_seconds": time.monotonic() - started, "exit_code": result.returncode,
              "stdout_sha256": file_hash(stdout_path), "stderr_sha256": file_hash(stderr_path)})
        return result.returncode

    if command("build-bellingham-pairs", commands[0]) != 0:
        raise RuntimeError("Bellingham pair construction failed")
    calibration_rc = command("calibrate", commands[2])
    calibration = read_json(out / "calibration/calibration.json")
    write(evidence / "calibration.json", calibration)
    copy_exclusive(out / "calibration/candidate_table.csv", evidence / "candidate_table.csv")
    if calibration_rc != 0 or calibration["status"] != "CONFIG_FROZEN":
        return {"status": "NOT_RELEASED", "real_calibration": "EXECUTED_DIAGNOSTIC_ONLY",
                "real_validation": "NOT_RUN", "reason": "NO_CANDIDATE_PASSED"}
    if command("build-bloomington-pairs", commands[1]) != 0:
        raise RuntimeError("Bloomington pair construction failed")
    validation_rc = command("validate", commands[3])
    receipt = read_json(out / "validation/validation.json")
    write(evidence / "validation.json", receipt)
    for name in ("worlds.csv", "risk_coverage.csv", "pairs.csv"):
        copy_exclusive(out / "validation" / name, evidence / name)
    copy_exclusive(no_symlinks(args.exposure_ledger), evidence / "validation-exposure-ledger.json")
    if validation_rc == 0 and receipt["status"] == "VALIDATED":
        validate_release(receipt, registration["config"])
        return {"status": "VALIDATED_IN_STATED_SCOPE", "real_calibration": "CONFIG_FROZEN",
                "real_validation": "VALIDATED", "validation_sha256": receipt["validation_sha256"]}
    return {"status": "NOT_RELEASED", "real_calibration": "CONFIG_FROZEN",
            "real_validation": "EXECUTED_DIAGNOSTIC_ONLY", "reason_codes": receipt.get("reason_codes")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--source-dir", help="Shared source cache; only the fixed metric24 rows are added")
    parser.add_argument("--exposure-ledger", required=True, help="Persistent ledger shared across candidate versions and raw roots")
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--max-source-gb", type=float, default=1.81)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--prepare-worlds", action="store_true", help="Explicit offline source/world preparation")
    parser.add_argument("--execute-motion", action="store_true")
    args = parser.parse_args(argv)
    out, evidence = no_symlinks(args.out), no_symlinks(args.evidence)
    phase = "motion" if args.execute_motion else ("preparation" if args.download or args.prepare_worlds else "plan")
    invocation = new_invocation(evidence)
    try:
        manifest, rows = frozen_selection(args.manifest, args.inventory, args.max_source_gb)
        out.mkdir(parents=True, exist_ok=True)
        evidence.mkdir(parents=True, exist_ok=True)
        registration = preregistration(args, manifest, rows, out, evidence)
        if phase == "plan":
            check = disk_preflight(args.smoke_root, out, SOURCE_BYTES + WORLD_RGB_BYTES + 20_000_000)
            write(invocation / "plan-disk-preflight.json", check)
            result = {"status": "PLANNED_NOT_RUN", "real_calibration": "NOT_RUN", "real_validation": "NOT_RUN",
                      "source_budget_verified": True, "disk_preflight_allowed": check["allowed"]}
        elif not args.execute_motion:
            prepare(args, manifest, rows, out, evidence, invocation)
            result = {"status": "METRIC24_SOURCE_WORLD_PIXELS_VALIDATED", "real_calibration": "NOT_RUN", "real_validation": "NOT_RUN"}
        else:
            verify_prepared_worlds(args, out, evidence, rows)
            result = run_motion(args, out, evidence, registration, invocation)
        if code_fingerprint() != registration["source_code_sha256"]:
            raise ValueError("Package source changed during the real attempt")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        evidence.mkdir(parents=True, exist_ok=True)
        result = {"status": "BLOCKED", "error_type": type(exc).__name__, "reason": str(exc),
                  "real_calibration": "CHECK_PRESERVED_COMMAND_RECEIPTS", "real_validation": "CHECK_PRESERVED_COMMAND_RECEIPTS"}
    ledger = no_symlinks(args.exposure_ledger)
    if args.execute_motion and ledger.is_file():
        snapshot = invocation / "validation-exposure-ledger.original.json"
        copy_exclusive(ledger, snapshot)
        write(invocation / "exposure-ledger-recovery.json", {
            "schema": "rswm-motion-exposure-ledger-recovery-v1", "original_path": str(ledger),
            "snapshot_path": str(snapshot), "original_byte_sha256": file_hash(ledger),
            "snapshot_byte_sha256": file_hash(snapshot),
            "policy": "Restore and reuse this historical ledger if the original is lost; never replace with an empty ledger to clear held-out exposure",
            "authenticated_global_history": False})
    result.update(schema="rswm-motion-real-attempt-status-v1", phase=phase, invocation=str(invocation),
                  real_generation_model="NOT_RUN", real_gpu="NOT_RUN")
    write(invocation / (phase + "-status.json"), result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["status"] in {"PLANNED_NOT_RUN", "METRIC24_SOURCE_WORLD_PIXELS_VALIDATED", "VALIDATED_IN_STATED_SCOPE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
