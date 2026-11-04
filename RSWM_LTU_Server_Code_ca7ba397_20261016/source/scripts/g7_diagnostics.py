"""Run real paired/LR64 CLIs on small synthetic worlds and retain fault evidence.

No source download, pretrained model, GPU, or real-data acceptance is implied.
The output directory must not exist; previous evidence is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from rswm_inria import lr64
from rswm_inria.geometry import digest, file_hash
from rswm_inria.paired import STUDY as PAIR_STUDY, paired_packages
from rswm_inria.suite import _save, _json, code_fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out.exists():
        raise ValueError("G7 evidence directory must be new")
    out.mkdir(parents=True)
    source_before = code_fingerprint()
    repo = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    commands, cases = [], []

    def command(module, parts, expected):
        cmd = [sys.executable, "-m", module, *map(str, parts)]
        run = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        row = {"command": cmd, "exit_code": run.returncode, "expected_exit_code": expected,
               "stdout": run.stdout, "stderr": run.stderr}
        commands.append(row)
        _save(out/"commands.json", commands)
        if run.returncode != expected:
            raise RuntimeError(f"Unexpected CLI result: {run.returncode}, expected {expected}; {run.stderr}")

    for name, size, module in (("paired", 2048, "rswm_inria.paired_execution"),
                               ("lr64", 1024, "rswm_inria.lr64")):
        work = out/name; inputs = work/"inputs"; inputs.mkdir(parents=True)
        yy, xx = np.indices((size, size))
        pixels = np.stack([xx % 251, yy % 253, (xx + 3*yy) % 255], axis=2).astype(np.uint8)
        world = inputs/"world.png"; Image.fromarray(pixels).save(world)
        ph = hashlib.sha256(pixels.tobytes()).hexdigest()
        _save(inputs/"models.json", [{"model_id": "oracle", "adapter": "oracle", "device": "cpu"}], exclusive=True)
        manifest_hash = digest({"synthetic_only": True, "world_pixel_sha256": ph})
        if name == "paired":
            task_rows = []
            for task, _ in paired_packages("synthetic-pair", "diagnostic_only", 0):
                if task["family"] != "F5":
                    continue
                path = inputs/(task["task_id"] + ".json"); _save(path, task, exclusive=True)
                task_rows.append({"task_id": task["task_id"], "path": path.name})
            _save(inputs/"index.json", {"study": PAIR_STUDY, "source_manifest_sha256": manifest_hash, "tasks": task_rows}, exclusive=True)
            _save(inputs/"world-index.json", {"worlds": [{"world_id": "synthetic-pair", "path": "world.png",
                  "city": "austin", "role": "diagnostic_only", "world_file_sha256": file_hash(world), "world_pixel_sha256": ph}]}, exclusive=True)
            plan_args = ["--task-index", inputs/"index.json", "--world-index", inputs/"world-index.json"]
            expected_outputs = 192
        else:
            task_rows = []
            for variant in ("exact", "offset"):
                task = lr64.task_for("synthetic-lr64", variant)
                path = inputs/(variant + ".json"); _save(path, task, exclusive=True)
                task_rows.append({"task_path": path.name, "world_path": "world.png"})
            _save(inputs/"index.json", {"schema": "rswm-lr64-diagnostic-index-v1", "study": lr64.STUDY,
                  "diagnostic_only": True, "source_manifest_sha256": manifest_hash, "tasks": task_rows}, exclusive=True)
            plan_args = ["--task-index", inputs/"index.json"]
            expected_outputs = 128
        pp, root = work/"plan.json", work/"suite"
        command(module, ["plan", *plan_args, "--models", inputs/"models.json", "--code-commit", head,
                         "--out", pp, "--max-attempts-per-run", "2"], 0)
        command(module, ["run", "--plan", pp, "--root", root], 0)
        plan = _json(pp)
        first = root/"runs"/plan["expected"][0]["run_id"]/"attempt-000"
        for fault, expected_rc, valid in (("healthy", 0, expected_outputs),
                                          ("resources", 1, expected_outputs),
                                          ("outcome", 1, expected_outputs//2),
                                          ("root-plan", 2, None)):
            victim = {"resources": first/"resources.json", "outcome": first/"outcome.json", "root-plan": root/"plan.json"}.get(fault)
            backup = victim.read_bytes() if victim else None
            analysis = work/(fault + "-analysis.json")
            if victim:
                victim.write_text("{", encoding="utf-8")
                (work/"faults").mkdir(exist_ok=True)
                (work/"faults"/(fault + ".json")).write_bytes(victim.read_bytes())
            try:
                command(module, ["verify", "--plan", pp, "--root", root], expected_rc)
                command(module, ["aggregate", "--plan", pp, "--root", root, "--out", analysis, "--bootstrap", "20"], expected_rc)
                if fault == "root-plan":
                    if analysis.exists():
                        raise RuntimeError("Corrupt root must not produce an analysis")
                    cases.append({"study": name, "fault": fault, "exit_code": expected_rc, "analysis_written": False})
                    continue
                report = _json(analysis)
                primary = report["analyses"]["first_attempt_only"]
                if primary["expected_runs"] != 2 or primary["planned_outputs"] != expected_outputs or primary["valid_outputs"] != valid:
                    raise RuntimeError("Fault handling changed frozen expected denominators")
                cases.append({"study": name, "fault": fault, "exit_code": expected_rc,
                              "expected_runs": primary["expected_runs"], "executed_runs": primary["executed_runs"],
                              "completed_runs": primary["completed_runs"], "planned_outputs": primary["planned_outputs"],
                              "valid_outputs": primary["valid_outputs"], "replay_verified_runs": primary["replay_verified_runs"],
                              "acceptance": report["acceptance"], "analysis_sha256": file_hash(analysis)})
            finally:
                if victim:
                    victim.write_bytes(backup)
    source_after = code_fingerprint()
    _save(out/"summary.json", {"schema": "rswm-g7-cli-evidence-v1", "head": head,
          "source_code_sha256_before": source_before, "source_code_sha256_after": source_after,
          "source_stable": source_before == source_after, "cases": cases,
          "data": "synthetic generated RGB; zero source images downloaded", "model": "evaluator-only Oracle",
          "weights": None, "gpu": "NOT_RUN", "real_data_validation": "NOT_RUN", "real_model_validation": "NOT_RUN"}, exclusive=True)
    if source_before != source_after:
        raise RuntimeError("Source changed during evidence run; repeat after source freeze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
