#!/usr/bin/env python3
"""Actual CPU TorchScript CLI controls; never real feature or model acceptance.

Inputs are retained synthetic first-commit mosaics. The deliberately trivial
local mean module is explicitly diagnostic, not a replacement feature backbone.
Weights, caches and arrays stay in the separate work directory; compact command
receipts and raw stdout/stderr are saved in the evidence directory.
"""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from rswm_inria.features import FEATURE_SCHEMA
from rswm_inria.geometry import file_hash
from rswm_inria.suite import code_fingerprint


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosaics", required=True)
    parser.add_argument("--reference-mosaics", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--evidence-dir", required=True)
    args = parser.parse_args()
    for source in (args.mosaics, args.reference_mosaics):
        metadata = json.loads(Path(source).read_text())
        if not metadata.get("patches") or any(row.get("role") != "diagnostic_only" for row in metadata["patches"]):
            raise ValueError("This reproduction accepts synthetic diagnostic-only mosaics")
    work, evidence = Path(args.work_dir).absolute(), Path(args.evidence_dir).absolute()
    for directory in (work, evidence):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Use fresh work and evidence directories")
        directory.mkdir(parents=True, exist_ok=True)
    fingerprint = code_fingerprint()

    class DiagnosticMean(torch.nn.Module):
        def forward(self, images):
            return images.mean(dim=(2, 3))

    weights = work / "diagnostic-mean-not-real-weights.pt"
    torch.jit.trace(DiagnosticMean().eval(), torch.zeros(1, 3, 128, 128)).save(str(weights))
    config = {
        "schema": FEATURE_SCHEMA, "backend": "torchscript-local-v1",
        "model_repo": "test-only/local-channel-mean", "model_revision": "diagnostic-v1",
        "output_layer": "serialized_module_output", "feature_dimension": 3,
        "weights": {"path": str(weights), "sha256": file_hash(weights)},
        "dtype": "float32", "color_input": "RGB", "input_shape": [128, 128, 3],
        "device": "cpu", "batch_size": 2, "diagnostic_features": True,
        "runtime_versions": {"torch": torch.__version__, "numpy": np.__version__,
                             "Pillow": importlib.metadata.version("Pillow")},
        "preprocessing": {"resize": [128, 128], "center_crop": None, "resample": "bilinear",
                          "scale": "uint8/255", "mean": [0, 0, 0], "std": [1, 1, 1]},
    }
    config_path = work / "diagnostic-config.json"
    write(config_path, config)
    commands = []

    def run(label, console, argv, expected=0):
        command = [str(Path(sys.executable).parent / console), *map(str, argv)]
        start = time.monotonic()
        with (evidence / (label + ".stdout.log")).open("x") as stdout, \
                (evidence / (label + ".stderr.log")).open("x") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        commands.append({"label": label, "argv": command, "cwd": str(Path.cwd()),
                         "expected_returncode": expected, "returncode": result.returncode,
                         "elapsed_seconds": time.monotonic() - start})
        # Preserve the actual failure before rejecting an unexpected result.
        (evidence / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode != expected:
            raise RuntimeError(label + ": unexpected CLI exit code")

    def extract(label, mosaics, output, selected_config=config_path, expected=0):
        run(label, "rswm-features", ["--config", selected_config, "--mosaics", mosaics,
            "--cache", work / "cache", "--out", work / output], expected)

    extract("generated", args.mosaics, "generated")
    extract("reference", args.reference_mosaics, "reference")
    cache_before = {p.name: file_hash(p) for p in (work / "cache").iterdir()}
    extract("generated-cache-reuse", args.mosaics, "generated-reuse")
    cache_after = {p.name: file_hash(p) for p in (work / "cache").iterdir()}
    if cache_before != cache_after:
        raise ValueError("Cache reuse mutated retained entries")
    generated = json.loads((work / "generated/features.json").read_text())
    reference = json.loads((work / "reference/features.json").read_text())
    reused = json.loads((work / "generated-reuse/features.json").read_text())
    if generated != reused:
        raise ValueError("Cache reuse changed feature bundle")
    for bundle in (generated, reference):
        if bundle["diagnostic_features"] is not True or bundle["release_validated"] is not False:
            raise ValueError("Diagnostic controls incorrectly authorized release")
        if bundle["complete_tiles"] != bundle["planned_tiles"] or not bundle["complete_tiles"]:
            raise ValueError("Feature output incomplete")
    run("controls", "rswm-validate-features", ["--config", config_path,
        "--reference-mosaics", args.reference_mosaics, "--cache", work / "control-cache",
        "--out", evidence / "control-validation.json", "--sample-caps", 1, 2,
        "--bootstrap", 3, "--seed", 7])
    run("quality", "rswm-quality", ["--generated", work / "generated/features.json",
        "--reference", work / "reference/features.json", "--out", evidence / "quality.json",
        "--cap", 2, "--bootstrap", 3, "--seed", 7])
    missing = copy.deepcopy(config)
    missing["weights"]["path"] = str(work / "missing-control-weights.pt")
    write(work / "missing-config.json", missing)
    extract("missing-weights", args.mosaics, "missing-output", work / "missing-config.json", 1)
    corrupted = copy.deepcopy(config)
    bad_weights = work / "corrupt-diagnostic-weights.pt"
    bad_weights.write_bytes(b"corrupt control archive")
    corrupted["weights"]["path"] = str(bad_weights)
    write(work / "corrupt-config.json", corrupted)
    extract("corrupt-weights", args.mosaics, "corrupt-output", work / "corrupt-config.json", 2)
    metadata = work / "cache" / (generated["features"][0]["feature_cache_key"] + ".json")
    before = file_hash(metadata)
    value = json.loads(metadata.read_text())
    value["feature_sha256"] = "0" * 64
    metadata.write_text(json.dumps(value) + "\n")
    extract("corrupt-cache", args.mosaics, "corrupt-cache-output", expected=2)
    if code_fingerprint() != fingerprint:
        raise ValueError("Source changed during actual CLI verification")
    summary = {
        "status": "PASS_SOFTWARE_DIAGNOSTIC", "source_code_sha256": fingerprint,
        "reproducer_sha256": file_hash(__file__),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "diagnostic_features": True, "real_model_validation": "NOT_RUN",
        "real_data_validation": "NOT_RUN", "real_gpu_validation": "NOT_RUN",
        "publication_acceptance": "NOT_READY", "release_validated": False,
        "generated_tiles": generated["complete_tiles"], "reference_tiles": reference["complete_tiles"],
        "command_count": len(commands), "cache_reuse_identical": True,
        "weights_sha256": config["weights"]["sha256"], "weight_scope": "locally generated diagnostic module only",
        "corrupt_cache_metadata": {"path": str(metadata), "before_sha256": before, "after_sha256": file_hash(metadata)},
        "input_mosaic_sha256": {"generated": file_hash(args.mosaics), "reference": file_hash(args.reference_mosaics)},
        "raw_assets_location": str(work),
        "evidence_sha256": {p.name: file_hash(p) for p in sorted(evidence.iterdir()) if p.is_file()},
    }
    write(evidence / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
