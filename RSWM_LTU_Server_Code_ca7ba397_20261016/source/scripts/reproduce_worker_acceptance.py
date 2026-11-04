#!/usr/bin/env python3
"""Record actual OS permission and public-worker F2/F7 CLI receipts.

All generated data are explicitly synthetic diagnostics. A denied namespace
never becomes an isolation pass through the cooperative diagnostic cases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image
from rswm_inria.geometry import canonical, digest
from rswm_inria.tasks import packages


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_command(command, out, label):
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=360., close_fds=True)
    stdout, stderr = out / (label + ".stdout.log"), out / (label + ".stderr.log")
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    try:
        response = json.loads(completed.stdout)
    except ValueError:
        response = {"status": "NOT_READABLE"}
    return {"command": command, "returncode": completed.returncode,
            "elapsed_seconds": time.monotonic() - started, "response": response,
            "stdout": stdout.name, "stderr": stderr.name,
            "stdout_sha256": sha256(stdout), "stderr_sha256": sha256(stderr)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Receipt destination must be empty; preserve previous evidence")
    out.mkdir(parents=True, exist_ok=True)
    permission = record_command([sys.executable, "-m", "rswm_inria.worker_client", "--probe-isolation"], out, "permission-probe")
    permission_pass = permission["returncode"] == 0 and permission["response"].get("status") == "PASS"
    profile = "isolated_worker" if permission_pass else "cooperative_subprocess"
    inputs = out / "synthetic-inputs"; inputs.mkdir()
    world = inputs / "world.png"
    Image.fromarray(np.zeros((2048, 2048, 3), np.uint8)).save(world)
    config = inputs / "copy.json"
    config.write_bytes(canonical({"model_id": "copy", "adapter": "copy"}) + b"\n")
    cases, selected = [], {}
    for task, _ in packages("worker-cli-synthetic", "diagnostic_only", 0):
        if task["family"] in {"F2", "F7"} and task["family"] not in selected:
            selected[task["family"]] = task
    for family in ("F2", "F7"):
        task = selected[family]
        task_path = inputs / (family + "-task.json")
        task_path.write_bytes(canonical(task) + b"\n")
        destination = out / (family + "-worker-output")
        command = [sys.executable, "-m", "rswm_inria.worker_client", "--profile", profile,
                   "--task", str(task_path), "--world", str(world), "--config", str(config),
                   "--out", str(destination), "--seed", "0"]
        receipt = record_command(command, out, family)
        receipt.update(family=family, task_sha256=digest(task), synthetic_data=True,
                       model="CopyModel", weight_sha256=None, real_model_validation="NOT_RUN",
                       real_gpu_validation="NOT_RUN", execution_profile=profile)
        if (destination / "worker-report.json").is_file():
            receipt["run_report"] = json.loads((destination / "worker-report.json").read_text())
            receipt["resource_receipt"] = json.loads((destination / "worker-resources.json").read_text())
            receipt["artifact_hashes"] = {str(p.relative_to(destination)): sha256(p)
                for p in sorted(destination.rglob("*")) if p.is_file()}
        cases.append(receipt)
    branch_script = """import json,sys,numpy as np
from rswm_inria.worker_client import WorkerClient,snapshot_branch_probe
from rswm_inria.runner import PUBLIC_SPEC
with WorkerClient({'model_id':'random','adapter':'random'},sys.argv[1],max_query_batch=16) as client:
 report=snapshot_branch_probe(client,np.zeros((240,416,3),np.uint8),PUBLIC_SPEC,7)
report['resources']=client.worker_resources
print(json.dumps(report))
raise SystemExit(0 if report['status']=='PASS' else 1)
"""
    branch = record_command([sys.executable, "-c", branch_script, profile], out, "snapshot-branch")
    branch.update(synthetic_data=True, model="RandomModel", execution_profile=profile,
                  real_model_validation="NOT_RUN", scope="separate diagnostic restore gate; not part of F2/F7 expected-run totals")
    import rswm_inria.worker_client as client
    source_root = Path(client.__file__).resolve().parent
    source = {name: sha256(source_root / name) for name in
              ("worker_ipc.py", "worker_public.py", "worker_client.py", "isolation_backend.py")}
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    planned = sum(case.get("run_report", {}).get("planned_output_events", 0) for case in cases)
    verified = sum(case.get("run_report", {}).get("verified_png_outputs", 0) for case in cases)
    functional = (all(case["returncode"] == 0 and case.get("run_report", {}).get("acceptance_status") == "ACCEPTED" for case in cases)
                  and branch["returncode"] == 0 and branch["response"].get("status") == "PASS")
    summary = {"schema": "rswm-worker-acceptance-receipt-v1", "source_commit": git.stdout.strip() if git.returncode == 0 else None,
               "source_file_sha256": source, "permission_probe": permission, "cases": cases, "snapshot_branch_probe": branch,
               "expected_runs": 2, "executed_runs": len(cases), "completed_runs": sum(case["returncode"] == 0 for case in cases),
               "failed_runs": sum(case["returncode"] != 0 for case in cases),
               "planned_outputs": planned, "valid_replay_verified_outputs": verified,
               "diagnostic_execution_status": "PASS" if functional else "FAIL",
               "os_permission_status": "PASS" if permission_pass else "BLOCKED",
               "security_acceptance_status": "CPU_DIAGNOSTIC_DEPLOYMENT_CHECKS_ONLY" if permission_pass and functional else "BLOCKED",
               "protocol_security_verified": False,
               "limits": "CPU diagnostic profile only; real/numeric/GPU adapters are BLOCKED pending deployment audit",
               "reference_policy": "existing frozen RGB/first-planned-exposure semantics; no reference repair",
               "real_data_validation": "NOT_RUN", "real_model_validation": "NOT_RUN", "real_gpu_validation": "NOT_RUN"}
    (out / "receipt.json").write_bytes(canonical(summary) + b"\n")
    print(json.dumps({"receipt": str(out / "receipt.json"), "receipt_sha256": sha256(out / "receipt.json"),
                      "os_permission_status": summary["os_permission_status"], "diagnostic_execution_status": summary["diagnostic_execution_status"],
                      "planned_outputs": planned, "verified_outputs": verified}))
    return 1 if not functional else 0 if permission_pass else 3


if __name__ == "__main__":
    raise SystemExit(main())
