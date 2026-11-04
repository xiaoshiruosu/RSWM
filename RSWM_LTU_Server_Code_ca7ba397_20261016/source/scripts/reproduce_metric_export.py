"""Exercise retained motion/quality producers and export CLI on synthetic PNGs.

This is a small CPU diagnostic reproduction, not a real-model pilot or an
instrument release. Worlds, Oracle submissions and feature arrays come from the
current-worktree regression fixture; every producer/export command is retained.
Use new --work and --out directories after the package source is frozen.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time

from rswm_inria.geometry import canonical, digest, file_hash
from rswm_inria.motion_protocol import motion_config
from rswm_inria.suite import code_fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Small command logs and evidence")
    parser.add_argument("--work", required=True, help="Retained synthetic PNGs, bundles and exports")
    args = parser.parse_args(argv)
    out, work = Path(args.out).resolve(), Path(args.work).resolve()
    if out == work or out in work.parents or work in out.parents:
        parser.error("Evidence and fixture destinations must be separate")
    if out.exists() or work.exists():
        parser.error("Evidence and fixture destinations must be new")
    out.mkdir(parents=True)
    work.mkdir(parents=True)
    repository = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    commands = []
    checks = {}
    evidence = {
        "schema": "rswm-retained-metric-cli-evidence-v1",
        "source_kind": "synthetic_fixture",
        "diagnostic_only": True,
        "input_scope": "Two constant synthetic worlds; two F2 Oracle runs; diagnostic mean features",
        "real_model_validation": "NOT_RUN",
        "real_data_validation": "NOT_RUN",
        "real_instrument_validation": "NOT_RUN",
        "real_gpu_validation": "NOT_RUN",
        "publication_acceptance": "NOT_READY",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        "dirty_before": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True)),
        "reproducer_source_sha256_before": file_hash(Path(__file__)),
        "code_sha256_before": code_fingerprint(),
        "commands": commands,
        "checks": checks,
    }

    def write(path, value):
        Path(path).write_bytes(canonical(value) + b"\n")

    def command(label, arguments, expected, error_contains=None):
        invocation = [sys.executable, *map(str, arguments)]
        command_started = time.monotonic()
        completed = subprocess.run(invocation, cwd=repository, capture_output=True, text=True, timeout=180)
        stdout, stderr = out / (label + ".stdout.log"), out / (label + ".stderr.log")
        stdout.write_text(completed.stdout, encoding="utf-8")
        stderr.write_text(completed.stderr, encoding="utf-8")
        entry = {
            "label": label,
            "argv": invocation,
            "exit_code": completed.returncode,
            "expected_exit_code": expected,
            "elapsed_seconds": time.monotonic() - command_started,
            "stdout_sha256": file_hash(stdout),
            "stderr_sha256": file_hash(stderr),
        }
        if error_contains:
            entry["required_error_text"] = error_contains
            entry["error_text_present"] = error_contains in completed.stdout + completed.stderr
        commands.append(entry)
        write(out / "commands.json", commands)
        if completed.returncode != expected:
            raise RuntimeError(f"{label}: exit {completed.returncode}, expected {expected}")
        if error_contains and not entry["error_text_present"]:
            raise RuntimeError(f"{label}: expected error reason {error_contains!r}")
        return completed

    def export_args(fixture_root, target, *metric_arguments):
        return ["-m", "rswm_inria.export_tables", "--plan", fixture_root / "plan.json",
                "--suite", fixture_root / "suite", "--out", target,
                "--bootstrap", "2", "--seed", "0", *metric_arguments]

    def sealed(report, key):
        body = copy.deepcopy(report)
        body.pop(key, None)
        body[key] = digest(body)
        return body

    try:
        package_root = Path(sys.modules["rswm_inria.suite"].__file__).resolve().parent
        if package_root != (repository / "src/rswm_inria").resolve():
            raise ValueError("Reproduction must import the current checkout")
        evidence["executed_source_is_checkout"] = True
        evidence["executed_source_files"] = {
            p.relative_to(package_root).as_posix(): file_hash(p) for p in sorted(package_root.rglob("*.py"))
        }
        fixture_module = repository / "tests/test_metric_results.py"
        evidence["fixture_source_sha256"] = file_hash(fixture_module)
        fixture = runpy.run_path(str(fixture_module))["make_fixture"](work / "fixture")
        root = fixture["root"]
        evidence["planned_runs"] = len(fixture["plan"]["expected"])
        evidence["planned_new_outputs"] = 64
        evidence["fixture_scope"] = "Synthetic preparation uses current regression helper; CLI producer results below are independently generated"

        command("suite-verify", ["-m", "rswm_inria.suite", "verify", "--plan", root / "plan.json",
                                 "--suite", fixture["suite"]], 0)
        evidence["verified_new_submission_pngs"] = len([
            p for p in (fixture["suite"] / "runs").rglob("*.png")
            if "submission" in p.parts and p.name != "initial.png"
        ])
        assert evidence["verified_new_submission_pngs"] == evidence["planned_new_outputs"]
        write(root / "motion-config.json", motion_config())
        command("motion-producer", [repository / "scripts/validate_motion.py", "evaluate-suite",
                "--plan", root / "plan.json", "--suite", fixture["suite"], "--config", root / "motion-config.json",
                "--diagnostic", "--out", root / "motion-cli"], 1)
        command("quality-producer", [repository / "scripts/evaluate_quality.py",
                "--generated", root / "generated-features/features.json", "--reference", root / "reference-features/features.json",
                "--suite-plan", root / "plan.json", "--suite", fixture["suite"],
                "--generated-mosaics", root / "generated-mosaics/mosaics.json",
                "--reference-mosaics", root / "reference-mosaics/mosaics.json", "--patch-plan", root / "patch-plan.json",
                "--cap", "2", "--bootstrap", "2", "--seed", "0", "--out", root / "quality-cli.json"], 0)
        motion_path, quality_path = root / "motion-cli/motion.json", root / "quality-cli.json"
        motion, quality = json.loads(motion_path.read_text()), json.loads(quality_path.read_text())
        command("export-rgb-only", export_args(root, work / "rgb-only"), 0)
        no_metrics = json.loads((work / "rgb-only/metric-results.json").read_text())
        assert not no_metrics["motion_rows"] and not no_metrics["quality_rows"]
        assert all(no_metrics["availability"][k]["status"] == "NOT_AVAILABLE" for k in ("motion", "quality"))
        checks["no_metric_inputs_rgb_export"] = True

        command("export-retained-metrics", export_args(root, work / "analysis",
                "--motion-result", motion_path, "--quality-result", quality_path), 0)
        analysis = work / "analysis"
        metrics = json.loads((analysis / "metric-results.json").read_text())
        assert metrics["motion_rows"] and metrics["quality_rows"]
        assert all(r["coverage"] == 0 and r["mean_normalized_step_error"] is None for r in metrics["motion_rows"])
        assert all(r["polynomial_mmd_u"] < 0 for r in metrics["quality_rows"])
        assert all(r["diagnostic_only"] and not r["formal_metric_eligible"] for name in ("motion", "quality") for r in metrics[name + "_rows"])
        assert all("task_id" not in r for r in metrics["quality_rows"])
        checks.update(null_motion_error_at_zero_coverage=True, legitimate_negative_mmd_u_preserved=True,
                      frozen_population_not_broadcast_to_tasks=True, fixture_not_formal_metric_eligible=True)
        required = ["aggregation.json", "metric-results.json", "motion.csv", "quality.csv", "tables/motion.tex",
                    "tables/quality.tex", "figure-data/motion.json", "figure-data/quality.json", "provenance.json"]
        assert all((analysis / name).is_file() for name in required)
        provenance = json.loads((analysis / "provenance.json").read_text())
        assert provenance["metric_results_provenance"]
        for name, sha in provenance["files"].items():
            assert file_hash(analysis / name) == sha
        with (analysis / "motion.csv").open(newline="") as f:
            assert all(r["mean_normalized_step_error"] == "" for r in csv.DictReader(f))
        with (analysis / "quality.csv").open(newline="") as f:
            assert all(float(r["polynomial_mmd_u"]) < 0 for r in csv.DictReader(f))
        assert r"DIAGNOSTIC\_ONLY" in (analysis / "tables/motion.tex").read_text()
        assert r"DIAGNOSTIC\_ONLY" in (analysis / "tables/quality.tex").read_text()
        checks["json_csv_latex_figure_data_and_provenance_verified"] = True
        evidence["metric_rows"] = {name: len(metrics[name + "_rows"]) for name in ("motion", "quality")}
        # CI retains --out, while reproducible fixture pixels live under --work.
        # Preserve the actual small table/figure/provenance bytes in that artifact.
        shutil.copytree(analysis, out / "analysis")
        shutil.copytree(work / "rgb-only", out / "rgb-only-analysis")
        for name, sha in provenance["files"].items():
            assert file_hash(out / "analysis" / name) == sha
        checks["actual_export_bytes_retained_in_ci_evidence"] = True

        receipt = work / "instrument-only.json"
        write(receipt, {"schema": "rswm-motion-validation-v1", "release_validated": True})
        command("reject-instrument-only", export_args(root, work / "rejected-instrument",
                "--motion-result", receipt), 2, "MODEL_EVALUATION_REQUIRED")
        command("reject-missing-result", export_args(root, work / "rejected-missing",
                "--motion-result", work / "absent-motion.json"), 2, "METRIC_RETAINED_FILE_MISSING")
        command("reject-duplicate-result", export_args(root, work / "rejected-duplicate",
                "--quality-result", quality_path, "--quality-result", quality_path), 2, "DUPLICATE_RESULT_INPUT")

        # Resealing a result must not grant authority to changed run identities.
        for field in ("model_id", "model_manifest_sha256", "seed", "training_seed", "attempt", "study",
                      "source_manifest_sha256", "metric_config_sha256"):
            mutated = copy.deepcopy(motion)
            mutated["edges"][0][field] = "wrong"
            # Preserve the producer's relative CSV directory so the negative
            # reaches the altered identity, rather than a missing-export error.
            case = motion_path.parent / ("wrong-" + field + ".json")
            write(case, sealed(mutated, "motion_suite_sha256"))
            command("reject-wrong-" + field, export_args(root, work / ("rejected-wrong-" + field),
                    "--motion-result", case), 2, "REPLAY_MISMATCH")

        tampered = copy.deepcopy(quality)
        tampered["reports"][0]["own_complete_sample_diagnostic"]["polynomial_mmd_u"] += 1
        tampered_path = work / "tampered-quality.json"
        write(tampered_path, sealed(tampered, "evaluation_sha256"))
        command("reject-resealed-quality-value", export_args(root, work / "rejected-tamper",
                "--quality-result", tampered_path), 2, "RECOMPUTATION_MISMATCH")

        def copied_fixture(label):
            destination = work / label
            shutil.copytree(root, destination)
            report = copy.deepcopy(quality)
            for key, value in report["retained_inputs"].items():
                if isinstance(value, str):
                    report["retained_inputs"][key] = value.replace(str(root), str(destination))
            write(destination / "quality-cli.json", sealed(report, "evaluation_sha256"))
            return destination

        broken = copied_fixture("broken-cache")
        broken_bundle = json.loads((broken / "generated-features/features.json").read_text())
        cache_row = next(r for r in broken_bundle["features"] if r["status"] == "ENCODED")
        cache_metadata = broken / "generated-features" / cache_row["feature_cache_metadata_file"]
        cache_info = json.loads(cache_metadata.read_text())
        cache_info["feature_sha256"] = "0" * 64
        write(cache_metadata, cache_info)
        # Keep the suite's original directory: its resource receipts and binding
        # authenticate that location, while metric artifacts are copied safely.
        command("reject-broken-feature-cache", export_args(root, work / "rejected-cache",
                "--quality-result", broken / "quality-cli.json"), 2, "QUALITY_ORIGINAL_FEATURE_CACHE_INTEGRITY")
        missing = copied_fixture("missing-mosaic")
        missing_mosaic = next((missing / "generated-mosaics").glob("*.npz"))
        missing_mosaic.unlink()
        command("reject-missing-retained-mosaic", export_args(root, work / "rejected-mosaic",
                "--quality-result", missing / "quality-cli.json"), 2, missing_mosaic.name)
        unexpected = copy.deepcopy(quality)
        unexpected["reports"].append(copy.deepcopy(unexpected["reports"][0]))
        unexpected_path = work / "unexpected-quality.json"
        write(unexpected_path, sealed(unexpected, "evaluation_sha256"))
        command("reject-unexpected-result-group", export_args(root, work / "rejected-group",
                "--quality-result", unexpected_path), 2, "RECOMPUTATION_MISMATCH")
        nonfinite_path = work / "nonfinite-quality.json"
        nonfinite_path.write_text('{"illegal_number":NaN,' + quality_path.read_text()[1:], encoding="utf-8")
        command("reject-nonfinite-result", export_args(root, work / "rejected-nonfinite",
                "--quality-result", nonfinite_path), 2, "Non-finite JSON number")
        checks["negative_cases_fail_closed"] = True
        evidence["status"] = "PASS_SOFTWARE_DIAGNOSTIC"
    except Exception as exc:
        evidence.update(status="FAILED_SOFTWARE_DIAGNOSTIC", error_type=type(exc).__name__, error=str(exc))
    evidence["code_sha256_after"] = code_fingerprint()
    evidence["head_after"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    evidence["dirty_after"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True))
    evidence["reproducer_source_sha256_after"] = file_hash(Path(__file__))
    if evidence["code_sha256_after"] != evidence["code_sha256_before"]:
        evidence.update(status="FAILED_SOFTWARE_DIAGNOSTIC", error="Package source changed during CLI reproduction")
    if evidence["reproducer_source_sha256_after"] != evidence["reproducer_source_sha256_before"]:
        evidence.update(status="FAILED_SOFTWARE_DIAGNOSTIC", error="Reproducer source changed during execution")
    evidence["elapsed_seconds"] = time.monotonic() - started
    evidence["evidence_files"] = {p.relative_to(out).as_posix(): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}
    evidence["summary_sha256"] = digest(evidence)
    write(out / "summary.json", evidence)
    print(canonical({"status": evidence["status"], "commands": len(commands), "out": str(out),
                     "diagnostic_only": True, "publication_acceptance": "NOT_READY"}).decode())
    return 0 if evidence["status"] == "PASS_SOFTWARE_DIAGNOSTIC" else 1


if __name__ == "__main__":
    raise SystemExit(main())
