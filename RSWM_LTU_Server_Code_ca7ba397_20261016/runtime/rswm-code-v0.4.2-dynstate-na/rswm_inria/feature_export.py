"""Export an explicitly chosen local DINOv2 backbone as a feature candidate.

No weights/source are downloaded. A strict full state_dict replaces every
initialized tensor before forward. Float32 CPU round trips establish software
serialization compatibility, never reviewed instrument acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time
import warnings

import numpy as np

from .features import FEATURE_SCHEMA, validate_feature_config
from .geometry import digest, file_hash
from .server_inputs import _safe_path
from .suite import _json, _save


ARCHITECTURES = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024, "dinov2_vitg14": 1536,
                 "dinov2_vits14_reg": 384, "dinov2_vitb14_reg": 768, "dinov2_vitl14_reg": 1024, "dinov2_vitg14_reg": 1536}
REPO = "facebookresearch/dinov2"
ROUNDTRIP_POLICY = {"dtype": "float32", "device": "cpu", "batches": [1, 2], "atol": 1e-5, "rtol": 1e-5,
                    "input": "deterministic linspace[-3,3] normalized-image-tensor probes at one frozen spatial size",
                    "dimension_rule": "exact (batch, approved architecture embedding dimension)",
                    "finite_required": True}
CHILD = """import json, sys, numpy as np, torch
model = torch.jit.load(sys.argv[1], map_location='cpu').eval()
values = {}
for batch in (1, 2):
    x = np.load(sys.argv[2] + '/input-b' + str(batch) + '.npy', allow_pickle=False)
    with torch.inference_mode(): y = model(torch.from_numpy(x))
    if not isinstance(y, torch.Tensor) or y.dtype != torch.float32 or y.shape != (batch, int(sys.argv[4])) or not torch.isfinite(y).all():
        raise ValueError('FEATURE_EXPORT_CHILD_SHAPE_DTYPE_OR_FINITE')
    values['batch' + str(batch)] = y.cpu().numpy()
np.savez(sys.argv[3], **values)
print(json.dumps({'status':'JIT_FORWARD_EXECUTED', 'batches':[1,2], 'dimension':int(sys.argv[4])}))
"""


def _git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode != 0:
        raise ValueError("A readable, revision-locked DINOv2 Git checkout is required")
    return result.stdout.decode("utf-8").strip()


def verify_local_source(source_dir, revision):
    """Lock actual clean local files; this is no external source attestation."""
    root = _safe_path(source_dir, directory=True)
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("A full immutable DINOv2 commit SHA is required")
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root or _git(root, "rev-parse", "HEAD") != revision:
        raise ValueError("Local DINOv2 checkout differs from the explicitly requested revision")
    origin = _git(root, "remote", "get-url", "origin")
    if origin not in {"https://github.com/facebookresearch/dinov2", "https://github.com/facebookresearch/dinov2.git",
                      "git@github.com:facebookresearch/dinov2.git", "ssh://git@github.com/facebookresearch/dinov2.git"}:
        raise ValueError("Local checkout must identify the official facebookresearch/dinov2 origin")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Local DINOv2 checkout has operator changes; keep them and use a separate clean source checkout")
    for path in root.rglob("*"):
        if ".git" not in path.relative_to(root).parts and (path.is_symlink() or path.suffix in {".pyc", ".pyo"}):
            raise ValueError("Local source contains symlinks or bytecode caches; supply an independently prepared clean tree")
    raw = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True).stdout
    files = {}
    for name in raw.decode("utf-8").split("\0"):
        if not name:
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Local source inventory path escape")
        files[name] = file_hash(_safe_path(root / relative, file=True))
    for required in ("hubconf.py", "LICENSE", "dinov2/hub/backbones.py", "dinov2/models/vision_transformer.py"):
        if required not in files:
            raise ValueError("Selected source lacks the official DINOv2 hub/backbone interface")
    return {"root": str(root), "repo": REPO, "revision": revision, "files": files,
            "source_tree_sha256": digest(files), "source_authentication": "local Git/files consistency, not external attestation"}


def validate_preprocessing(value):
    fields = {"resize", "center_crop", "resample", "scale", "mean", "std"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Complete explicit feature preprocessing is required")
    resize, crop = value["resize"], value["center_crop"]
    if not isinstance(resize, list) or len(resize) != 2 or any(type(x) is not int or x <= 0 for x in resize):
        raise ValueError("Positive integer [height,width] resize is required")
    if crop is not None and (not isinstance(crop, list) or len(crop) != 2
                             or any(type(x) is not int or x <= 0 for x in crop) or any(x > y for x, y in zip(crop, resize))):
        raise ValueError("Center crop must fit inside the explicit resize")
    shape = crop or resize
    if any(x % 14 for x in shape):
        raise ValueError("DINOv2 serialized input height/width must be multiples of patch size 14")
    if value["resample"] not in {"bicubic", "bilinear"} or value["scale"] != "uint8/255":
        raise ValueError("Explicit supported RGB resampling and uint8/255 conversion are required")
    for name in ("mean", "std"):
        values = value[name]
        if (not isinstance(values, list) or len(values) != 3
                or any(type(x) not in (float, int) or not math.isfinite(x) or (name == "std" and x <= 0) for x in values)):
            raise ValueError("Finite RGB mean and positive finite std are required")
    return list(shape)


def _load_original_model(source, architecture):
    import torch
    if any(name == "dinov2" or name.startswith("dinov2.") for name in sys.modules):
        raise ValueError("Start export in a fresh process without another imported DINOv2 source")
    previous_bytecode = sys.dont_write_bytecode
    previous_xformers = os.environ.get("XFORMERS_DISABLED")
    sys.dont_write_bytecode = True
    os.environ["XFORMERS_DISABLED"] = "1"
    try:
        model = torch.hub.load(source["root"], architecture, source="local", pretrained=False)
    finally:
        sys.dont_write_bytecode = previous_bytecode
        if previous_xformers is None:
            os.environ.pop("XFORMERS_DISABLED", None)
        else:
            os.environ["XFORMERS_DISABLED"] = previous_xformers
    root = Path(source["root"])
    for name, module in sys.modules.copy().items():
        if name == "dinov2" or name.startswith("dinov2."):
            path = getattr(module, "__file__", None)
            if path is None or not Path(path).resolve().is_relative_to(root):
                raise ValueError("Imported DINOv2 module is outside the locked local checkout")
    if not isinstance(model, torch.nn.Module) or not callable(getattr(model, "forward_features", None)):
        raise ValueError("Official local hub did not return the required DINOv2 forward_features model")
    return model.cpu().float().eval()


def strict_load_state_dict(model, weights):
    """Load the official raw state dict; no stripping prefixes or nested fallback."""
    import torch
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected) or not state:
        raise ValueError("Complete raw pretrained state_dict keys must match exactly")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[name].shape or value.is_complex():
            raise ValueError("Pretrained state_dict tensor/shape/type mismatch: " + name)
        if not torch.isfinite(value).all():
            raise ValueError("Pretrained state_dict contains nonfinite tensors")
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise ValueError("Strict pretrained load reported missing or unexpected keys")
    actual = model.state_dict()
    if any(not torch.equal(actual[name].cpu(), tensor.to(dtype=actual[name].dtype).cpu()) for name, tensor in state.items()):
        raise ValueError("Actual loaded tensor differs from the complete retained checkpoint")
    dtypes = dict(Counter(str(tensor.dtype) for tensor in state.values()))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {"complete_keys": len(state), "checkpoint_dtypes": dtypes, "export_dtype": "float32", "strict_loaded": True}


def _wrapper(model):
    import torch
    class ClassToken(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__(); self.backbone = backbone
        def forward(self, x):
            return self.backbone.forward_features(x)["x_norm_clstoken"].float()
    return ClassToken(model).eval()


def _check_output(value, batch, dimension):
    import torch
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or tuple(value.shape) != (batch, dimension) or not torch.isfinite(value).all()):
        raise ValueError("Feature output must be finite float32 with exact (batch,dimension) shape")


def _array_save(path, value):
    with Path(path).open("xb") as stream:
        np.save(stream, value, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())


def _roundtrip(model, root, shape, dimension, *, timeout_seconds):
    import torch
    wrap = _wrapper(model)
    probes, expected = {}, {}
    for batch in ROUNDTRIP_POLICY["batches"]:
        value = torch.linspace(-3., 3., batch * 3 * shape[0] * shape[1], dtype=torch.float32).reshape(batch, 3, *shape)
        with torch.inference_mode(): result = wrap(value)
        _check_output(result, batch, dimension)
        probes[batch], expected[batch] = value, result.detach().cpu().numpy().copy()
    with warnings.catch_warnings(record=True) as observed, torch.inference_mode():
        traced = torch.jit.trace(wrap, probes[1], strict=True, check_trace=False)
    validation = root / "roundtrip"; validation.mkdir()
    target = root / "encoder.torchscript.pt"
    with target.open("xb") as stream:
        torch.jit.save(traced, stream); stream.flush(); os.fsync(stream.fileno())
    reloaded = torch.jit.load(str(target), map_location="cpu").eval()
    comparisons = []
    for batch, value in probes.items():
        with torch.inference_mode(): result = reloaded(value)
        _check_output(result, batch, dimension)
        torch.testing.assert_close(result, torch.from_numpy(expected[batch]),
                                   atol=ROUNDTRIP_POLICY["atol"], rtol=ROUNDTRIP_POLICY["rtol"])
        comparisons.append({"batch": batch, "same_process_max_abs_error": float(np.max(np.abs(result.numpy() - expected[batch])))})
        _array_save(validation / f"input-b{batch}.npy", value.numpy())
        _array_save(validation / f"eager-b{batch}.npy", expected[batch])
    del reloaded
    command = [sys.executable, "-c", CHILD, str(target), str(validation), str(validation / "fresh-process.npz"), str(dimension)]
    started = time.monotonic()
    with (validation / "stdout.log").open("xb") as stdout, (validation / "stderr.log").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=root, stdout=stdout, stderr=stderr)
        _save(validation / "process-start.json", {"pid": process.pid, "command": command, "timeout_seconds": timeout_seconds}, exclusive=True)
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True; process.kill(); exit_code = process.wait()
        _save(validation / "process-exit.json", {"pid": process.pid, "exit_code": exit_code, "timed_out": timed_out,
              "elapsed_seconds": time.monotonic() - started, "live": False}, exclusive=True)
    if timed_out or exit_code != 0:
        raise ValueError("Fresh-process JIT validation failed; retain its process/log records")
    with np.load(validation / "fresh-process.npz", allow_pickle=False) as fresh:
        if set(fresh.files) != {"batch1", "batch2"}:
            raise ValueError("Fresh-process result has missing or extra batches")
        for row in comparisons:
            batch = row["batch"]; actual = fresh[f"batch{batch}"]
            if actual.dtype != np.float32 or actual.shape != (batch, dimension) or not np.isfinite(actual).all():
                raise ValueError("Fresh-process feature shape/dtype/finite check failed")
            np.testing.assert_allclose(actual, expected[batch], atol=ROUNDTRIP_POLICY["atol"], rtol=ROUNDTRIP_POLICY["rtol"])
            row["fresh_process_max_abs_error"] = float(np.max(np.abs(actual - expected[batch])))
    return target, {"policy": ROUNDTRIP_POLICY, "comparisons": comparisons,
                    "tracer_warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in observed]}


def _export_loaded_model(model, *, root, dimension, preprocessing, model_repo, model_revision,
                         device, batch_size, diagnostic_features, timeout_seconds):
    """Shared serialization primitive; software fixtures must remain diagnostic."""
    import torch
    shape = validate_preprocessing(preprocessing)
    target, roundtrip = _roundtrip(model, root, shape, dimension, timeout_seconds=timeout_seconds)
    config = {"schema": FEATURE_SCHEMA, "backend": "torchscript-local-v1", "model_repo": model_repo,
              "model_revision": model_revision, "output_layer": "serialized_module_output", "feature_dimension": dimension,
              "weights": {"path": target.name, "sha256": file_hash(target)}, "dtype": "float32", "color_input": "RGB",
              "input_shape": [128, 128, 3], "output_reduction": "none", "output_selector": None,
              "device": device, "batch_size": batch_size, "preprocessing": preprocessing,
              "runtime_versions": {"torch": torch.__version__, "numpy": np.__version__, "Pillow": importlib.metadata.version("Pillow")},
              "diagnostic_features": diagnostic_features}
    validate_feature_config(config, root)
    _save(root / "feature-config.json", config, exclusive=True)
    return config, roundtrip


def export_dinov2(*, source_dir, weights, architecture, revision, decisions, preprocessing,
                   device, batch_size, out, timeout_seconds=300):
    if architecture not in ARCHITECTURES:
        raise ValueError("Only explicitly named supported DINOv2 backbones are accepted")
    if (not isinstance(device, str) or re.fullmatch(r"cpu|cuda(?::[0-9]+)?", device) is None
            or type(batch_size) is not int or batch_size <= 0 or type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds) or timeout_seconds <= 0):
        raise ValueError("Explicit feature device, positive batch size and finite child timeout are required")
    decision_path = _safe_path(decisions, file=True); decision = _json(decision_path)
    if (not isinstance(decision, dict) or set(decision) != {"approved", "license_record", "decision_reference"}
            or decision["approved"] is not True or any(not isinstance(decision[k], str) or not decision[k].strip()
                                                   for k in ("license_record", "decision_reference"))):
        raise ValueError("Actual owner-approved source/weight use decision and reference are required")
    preparation_path = _safe_path(preprocessing, file=True); preparation = _json(preparation_path)
    validate_preprocessing(preparation)
    checkpoint = _safe_path(weights, file=True)
    if checkpoint.stat().st_size <= 0:
        raise ValueError("Actual nonempty local pretrained weights are required")
    source = verify_local_source(source_dir, revision)
    root = _safe_path(out)
    for protected in (Path(source["root"]), checkpoint.parent, decision_path, preparation_path):
        if root == protected or (protected.is_dir() and root.is_relative_to(protected)):
            raise ValueError("Feature export output must be independent of retained source/input trees")
    checkpoint_hash = file_hash(checkpoint)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    contract = {"scope": "local-pretrained-feature-candidate-not-instrument-release", "architecture": architecture,
                "source": source, "checkpoint_sha256": checkpoint_hash, "checkpoint_bytes": checkpoint.stat().st_size,
                "owner_decision": decision, "preprocessing": preparation, "roundtrip_policy": ROUNDTRIP_POLICY,
                "export_device": "cpu", "export_dtype": "float32", "xformers": "disabled before original source import",
                "requested_feature_device": device, "batch_size": batch_size, "release_validated": False}
    _save(root / "export-contract.json", contract, exclusive=True)
    try:
        model = _load_original_model(source, architecture)
        strict = strict_load_state_dict(model, checkpoint)
        config, roundtrip = _export_loaded_model(model, root=root, dimension=ARCHITECTURES[architecture],
            preprocessing=preparation, model_repo=REPO, model_revision=revision, device=device, batch_size=batch_size,
            diagnostic_features=False, timeout_seconds=timeout_seconds)
        if file_hash(checkpoint) != checkpoint_hash or verify_local_source(source_dir, revision) != source:
            raise ValueError("Retained source/weights changed during export")
        report = {"status": "EXPORTED_FEATURE_CANDIDATE_NOT_INSTRUMENT_VALIDATED", "contract_sha256": digest(contract),
                  "feature_config_sha256": digest(config), "weights_sha256": config["weights"]["sha256"],
                  "strict_load": strict, "roundtrip": roundtrip, "elapsed_seconds": time.monotonic() - started,
                  "source_architecture_forward": "forward_features[x_norm_clstoken]", "release_validated": False,
                  "heldout_validation": "NOT_RUN", "research_acceptance": "NOT_EVALUATED",
                  "gpu_validation": "NOT_RUN", "config_path": str(root / "feature-config.json")}
        _save(root / "export-report.json", report, exclusive=True)
        return report
    except Exception as exc:
        _save(root / "export-failure.json", {"status": "FAILED", "error_type": type(exc).__name__, "reason": str(exc),
               "elapsed_seconds": time.monotonic() - started, "partial_files_retained": True,
               "release_validated": False, "research_acceptance": "NOT_EVALUATED"}, exclusive=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("dinov2")
    for name in ("source-dir", "weights", "architecture", "revision", "decisions", "preprocessing", "device", "out"):
        command.add_argument("--" + name, required=True)
    command.add_argument("--batch-size", type=int, required=True)
    command.add_argument("--timeout-seconds", type=float, default=300)
    args = vars(parser.parse_args(argv)); args.pop("command")
    try:
        report = export_dinov2(**args)
        print(json.dumps({k: report[k] for k in ("status", "config_path", "release_validated", "heldout_validation", "research_acceptance")}))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError, AssertionError,
            pickle.UnpicklingError, EOFError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__, "reason": str(exc),
                          "release_validated": False, "research_acceptance": "NOT_EVALUATED"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
