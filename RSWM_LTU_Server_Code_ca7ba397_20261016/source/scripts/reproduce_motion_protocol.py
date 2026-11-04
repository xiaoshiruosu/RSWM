"""Run diagnostic motion CLI/replay integration and preserve real evidence.

All pixels are seeded synthetic fixtures. A successful engineering run remains
DIAGNOSTIC_ONLY and does not certify calibration on Bellingham/Bloomington.
Raw worlds and submission PNGs go to --work, outside the small evidence folder.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from rswm_inria.geometry import canonical, digest, file_hash, rng_for
from rswm_inria.motion_protocol import ROLES, motion_config
from rswm_inria.suite import build_plan, code_fingerprint, save_plan
from rswm_inria.tasks import packages


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work", required=True)
    args = parser.parse_args(argv)
    out, work = Path(args.out).resolve(), Path(args.work).resolve()
    if out.exists() or work.exists():
        parser.error("Evidence and fixture destinations must be new")
    out.mkdir(parents=True); work.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    start = time.time()
    commands = []
    evidence = {"schema":"rswm-motion-cli-evidence-v1", "source_kind":"synthetic_fixture",
                "diagnostic_only":True, "real_calibration":"NOT_RUN", "real_validation":"NOT_RUN",
                "gpu":"NOT_RUN", "model":"oracle evaluator-only", "weight_hashes":[],
                "head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=repository,text=True).strip(),
                "code_sha256_before":code_fingerprint(), "commands":commands}
    package_root = Path(sys.modules["rswm_inria.suite"].__file__).parent
    evidence["executed_source_files"] = {path.relative_to(package_root).as_posix():file_hash(path)
                                        for path in sorted(package_root.rglob("*.py"))}
    evidence["executed_source_is_checkout"] = package_root.resolve()==(repository/"src/rswm_inria").resolve()

    def write(path, value):
        path.write_bytes(canonical(value)+b"\n")

    def command(label, arguments, expected):
        invocation = [sys.executable, *arguments]
        result = subprocess.run(invocation,cwd=repository,capture_output=True,text=True)
        (out/(label+".stdout.log")).write_text(result.stdout)
        (out/(label+".stderr.log")).write_text(result.stderr)
        commands.append({"label":label,"argv":invocation,"exit_code":result.returncode,
                         "expected_exit_code":expected})
        write(out/"commands.json",commands)
        if result.returncode != expected:
            raise RuntimeError(f"{label} returned {result.returncode}, expected {expected}")

    try:
        indexes = {}
        world_arrays = {}
        for role, city in ROLES.items():
            root = work/city; root.mkdir()
            pixels = rng_for("motion-cli-"+city).integers(0,256,(2048,2048,3),dtype=np.uint8)
            image_path = root/"world.png"; Image.fromarray(pixels).save(image_path)
            row = {"world_id":"synthetic-motion-"+city,"role":role,"city":city,"path":"world.png",
                   "world_file_sha256":file_hash(image_path),
                   "world_pixel_sha256":hashlib.sha256(pixels.tobytes()).hexdigest()}
            index = {"source_manifest_sha256":digest(["synthetic_fixture",city]),"worlds":[row]}
            path = root/"index.json"; write(path,index)
            indexes[role] = path; world_arrays[role] = image_path
            command("pairs-"+city,[str(repository/"scripts/calibrate_motion.py"),"build-pairs",
                    "--world-index",str(path),"--world-root",str(root),"--role",role,
                    "--pairs-per-world","13","--diagnostic-fixture","--out",str(work/(city+"-pairs"))],0)
        config = motion_config(); config_path = work/"config.json"; write(config_path,config)
        candidates = work/"candidates.json"; write(candidates,[config])
        command("calibrate",[str(repository/"scripts/calibrate_motion.py"),"calibrate",
                "--pair-manifest",str(work/"bellingham-pairs/pairs.json"),"--candidates",str(candidates),
                "--out",str(work/"calibration")],1)
        command("validate",[str(repository/"scripts/validate_motion.py"),"validate",
                "--pair-manifest",str(work/"bloomington-pairs/pairs.json"),
                "--calibration",str(work/"calibration/calibration.json"),
                "--exposure-ledger",str(work/"exposures.json"),"--diagnostic-fixture",
                "--out",str(work/"validation")],1)
        tasks = []
        for task,_ in packages("synthetic-motion-bellingham","diagnostic_only",0):
            path = work/(task["family"]+".json"); write(path,task)
            tasks.append({"task_path":str(path),"world_path":str(world_arrays["metric_calibration"]),"city":"bellingham"})
        plan = build_plan(tasks,[{"model_id":"oracle","adapter":"oracle"}],[0],
                          digest("synthetic-motion-cli-source"),evidence["head"])
        plan_path = work/"plan.json"; save_plan(plan,plan_path)
        command("suite-run",["-m","rswm_inria.suite","run","--plan",str(plan_path),"--out",str(work/"suite")],0)
        command("suite-verify",["-m","rswm_inria.suite","verify","--plan",str(plan_path),"--suite",str(work/"suite"),
                "--out",str(work/"verify.json")],0)
        command("suite-motion",[str(repository/"scripts/validate_motion.py"),"evaluate-suite","--plan",str(plan_path),
                "--suite",str(work/"suite"),"--config",str(config_path),"--diagnostic","--out",str(work/"motion")],1)
        motion = json.loads((work/"motion/motion.json").read_text())
        assert motion["expected_runs"]==4 and motion["planned_edges"]==352
        assert motion["status"]=="DIAGNOSTIC_ONLY" and motion["release_validated"] is False
        assert all(run["submission_status"]=="VERIFIED" for run in motion["runs"])
        f7 = next(run for run in motion["runs"] if run["family"]=="F7")
        assert f7["unique_planned_edges"]==160 and f7["streams"]["prefix"]["planned_edges"]==32
        assert len({edge["pair_id"] for edge in motion["edges"]})==352
        evidence["suite"] = {"expected":4,"executed":4,"completed":4,"failed":0,
                             "planned_outputs":352,"valid_outputs":352,"replay_verified":4,
                             "first_attempt_only":True,"recovery_attempts":0,
                             "planned_motion_edges":352,"measured_motion_edges":motion["measured_edges"],
                             "f3_ratio":next(run["f3_ratio"] for run in motion["runs"] if run["family"]=="F3"),
                             "f7_unique_edges":160,"f7_prefix_edges":32}
        calibration = json.loads((work/"calibration/calibration.json").read_text())
        validation = json.loads((work/"validation/validation.json").read_text())
        assert calibration["status"]==validation["status"]=="DIAGNOSTIC_ONLY"
        assert calibration["real_calibration_executed"] is validation["real_validation_executed"] is False
        evidence["calibration"] = {key:calibration[key] for key in ("status","calibration_sha256","reason_codes")}
        evidence["validation"] = {key:validation[key] for key in
            ("status","validation_sha256","reason_codes","validation_previously_viewed","validation_used_for_new_config")}
        for name in ("calibration/candidate_table.csv","calibration/calibration.json",
                     "validation/pairs.csv","validation/worlds.csv","validation/risk_coverage.csv","validation/validation.json",
                     "motion/motion_edges.csv","motion/motion_run_coverage.csv"):
            target = out/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(work/name,target)
        evidence["status"] = "ENGINEERING_PASS_DIAGNOSTIC_ONLY"
    except Exception as exc:
        evidence["status"] = "FAILED"
        evidence["error_type"] = type(exc).__name__; evidence["error"] = str(exc)
    evidence["code_sha256_after"] = code_fingerprint()
    evidence["source_changed_during_execution"] = evidence["code_sha256_before"]!=evidence["code_sha256_after"]
    if evidence["source_changed_during_execution"]:
        evidence["status"] = "FAILED_SOURCE_CHANGED"
    evidence["elapsed_seconds"] = time.time()-start
    evidence["artifact_hashes"] = {str(path.relative_to(out)):file_hash(path)
                                  for path in sorted(out.rglob("*")) if path.is_file()}
    write(out/"summary.json",evidence)
    print(json.dumps({"status":evidence["status"],"out":str(out),"suite":evidence.get("suite")},sort_keys=True))
    return 0 if evidence["status"]=="ENGINEERING_PASS_DIAGNOSTIC_ONLY" else 1


if __name__=="__main__":
    raise SystemExit(main())
