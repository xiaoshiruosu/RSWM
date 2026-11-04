"""Explicit cooperative adapter factories; public configuration only.

Oracle construction remains evaluator-only in suite.py. Local checkpoints are
never downloaded here. Backend availability and actual restoration are separate
from validating a frozen configuration.
"""
import copy
import re
from pathlib import Path

ADAPTERS = {"copy", "random", "noise-map", "oracle", "lazy-map", "eager-map", "wan-i2v",
            "raster", "frontier", "lazy", "numeric-camera"}
_MAPS = {"lazy-map", "eager-map", "raster", "frontier", "lazy"}
_COMMON = {"model_id", "adapter", "device", "training_seed"}
_GENERATIVE = {"model_dir", "steps", "prompt", "diagnostic_noise"}
_NUMERIC = {"backend_manifest", "camera", "training_camera_sha256", "adapter_release"}


def numeric_contract(config, *, check_local_assets=True):
    """Validate and hash the approved local backend without importing its code.

    Approval and a hash are source identities, not real-model validation. The
    inference configuration must match the declared training camera contract.
    """
    from .adapters.camera_conditioning import CameraConfig
    from .geometry import digest
    from .training import locked_tree
    if "device" not in config:
        raise ValueError("Explicit backend device required")
    manifest = config.get("backend_manifest")
    fields = {"source_dir", "source_tree_sha256", "entrypoint", "weights_dir",
              "weights_tree_sha256", "objective", "approved", "license_record"}
    if not isinstance(manifest, dict) or set(manifest) != fields or manifest["approved"] is not True:
        raise ValueError("BLOCKED: approved original backend manifest required")
    if manifest["objective"] not in {"flow_matching", "diffusion"} or not isinstance(manifest["license_record"], str) or not manifest["license_record"].strip():
        raise ValueError("BLOCKED: original backend objective and rights record required")
    for key in ("source_dir", "weights_dir", "entrypoint"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ValueError("Backend paths must be nonempty text")
    for key in ("source_tree_sha256", "weights_tree_sha256"):
        if not isinstance(manifest[key], str) or not re.fullmatch(r"[0-9a-f]{64}", manifest[key]):
            raise ValueError("Full backend tree SHA256 required")
    relative = Path(manifest["entrypoint"])
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
        raise ValueError("Invalid backend entrypoint")
    camera = config.get("camera")
    if not isinstance(camera, dict) or set(camera) != {"translation_scaling", "framewise", "camera_channel", "temporal_stride"}:
        raise ValueError("Explicit complete camera configuration required")
    camera_manifest = CameraConfig(**camera).manifest()
    camera_sha256 = digest(camera_manifest)
    if config.get("training_camera_sha256") != camera_sha256:
        raise ValueError("Training/inference camera contract mismatch")
    result = {"backend_manifest": copy.deepcopy(manifest),
            "camera_manifest": camera_manifest, "camera_manifest_sha256": camera_sha256,
            "training_camera_sha256": config["training_camera_sha256"]}
    binding = config.get("adapter_release")
    if "adapter_release" in config:
        from .legacy_export import validate_release_binding
        binding = validate_release_binding(binding)
        if type(config.get("training_seed")) is not int or config["training_seed"] < 0:
            raise ValueError("Trained adapter requires explicit training_seed")
        result["adapter_release"] = binding
    if check_local_assets:
        source = locked_tree(manifest["source_dir"])
        weights = locked_tree(manifest["weights_dir"])
        if digest(source) != manifest["source_tree_sha256"]:
            raise ValueError("Backend source changed")
        if digest(weights) != manifest["weights_tree_sha256"]:
            raise ValueError("Backend base weights changed")
        if relative.as_posix() not in source:
            raise ValueError("BLOCKED: original backend entrypoint absent")
        if "backend_config.json" in source:
            from .suite import _json
            source_config = _json(Path(manifest["source_dir"]) / "backend_config.json")
            actual = source_config.get("adapter_release")
            if source_config.get("schema") == "rswm-legacy-wan-backend-v2" or actual is not None or binding is not None:
                from .legacy_export import validate_adapter_release
                if source_config.get("schema") != "rswm-legacy-wan-backend-v2" or actual != binding:
                    raise ValueError("Model and fresh source trained-adapter reference mismatch")
                validate_adapter_release(binding, expected_weights_tree_sha256=manifest["weights_tree_sha256"],
                    expected_camera_sha256=camera_sha256, expected_training_seed=config["training_seed"], allow_software=True)
        elif binding is not None:
            raise ValueError("Trained-adapter source configuration absent")
        result.update(source=source, weights=weights)
    return result


def validate_config(config, *, check_local_assets=True):
    if not isinstance(config, dict) or config.get("adapter") not in ADAPTERS:
        raise ValueError("Unknown adapter configuration")
    if not isinstance(config.get("model_id"), str) or not config["model_id"]:
        raise ValueError("Nonempty model_id required")
    kind = config["adapter"]
    allowed = _COMMON | (_GENERATIVE if kind in _MAPS | {"wan-i2v"} else _NUMERIC if kind == "numeric-camera" else set())
    unknown = set(config) - allowed
    if unknown:
        raise ValueError("Unsupported adapter fields: " + ", ".join(sorted(unknown)))
    device = config.get("device", "cpu")
    if not isinstance(device, str) or not (device == "cpu" or device == "cuda" or
                                             (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("Supported device is cpu or cuda[:index]")
    if config.get("training_seed") is not None and (type(config["training_seed"]) is not int or config["training_seed"] < 0):
        raise ValueError("Invalid training seed")
    if "steps" in config and (type(config["steps"]) is not int or config["steps"] <= 0):
        raise ValueError("steps must be positive")
    if "prompt" in config and not isinstance(config["prompt"], str):
        raise ValueError("prompt must be text")
    if "diagnostic_noise" in config and type(config["diagnostic_noise"]) is not bool:
        raise ValueError("diagnostic_noise must be boolean")
    if config.get("diagnostic_noise") and config.get("model_dir"):
        raise ValueError("diagnostic_noise and real model_dir are mutually exclusive")
    if kind in _MAPS | {"wan-i2v"}:
        if kind == "wan-i2v" and config.get("diagnostic_noise"):
            raise ValueError("Wan has no diagnostic noise fallback")
        if not config.get("model_dir") and not config.get("diagnostic_noise"):
            raise ValueError("Local weights or explicit diagnostic_noise required")
        if config.get("model_dir"):
            root = Path(config["model_dir"])
            if not root.is_dir() or not any(p.is_file() for p in root.rglob("*")):
                raise ValueError("Nonempty local checkpoint directory required")
            if root.is_symlink() or any(p.is_symlink() for p in root.parents) or any(p.is_symlink() for p in root.rglob("*")):
                raise ValueError("Checkpoint symlinks are unsupported")
        elif "steps" in config or "prompt" in config:
            raise ValueError("Noise completer does not consume prompt or steps")
    if kind == "numeric-camera":
        numeric_contract(config, check_local_assets=check_local_assets)
    return copy.deepcopy(config)


def capabilities(config):
    validate_config(config)
    kind = config["adapter"]
    result = {"execution_mode": "cooperative_inprocess", "offline32": True,
            "online16": True, "snapshot": True, "world_hw": [2048, 2048],
            "numerical_input": "ignored" if kind in {"wan-i2v", "copy", "random"} else "spatial_query",
            "purpose": "evaluator_diagnostic" if kind == "oracle" else
                       "diagnostic" if kind in {"copy", "random", "noise-map"} or config.get("diagnostic_noise") else "real_generator",
            "real_model_validation": "NOT_RUN", "real_gpu_validation": "NOT_RUN"}
    if kind == "numeric-camera":
        if config.get("adapter_release", {}).get("scope") == "software_control":
            result["purpose"] = "diagnostic"
        result.update(numerical_input="camera_" + config["camera"]["camera_channel"],
                      camera_manifest_sha256=config["training_camera_sha256"],
                      real_backend_layer_consumption="NOT_RUN",
                      canonical_hw=[240, 416], native_hw=[480, 832],
                      native_context_slots=1,
                      missing_endpoint_policy="terminate_stream_without_reference_repair")
    return result


def create_adapter(config):
    validate_config(config)
    from .baselines import CopyModel, RandomModel, StaticNoiseMap, WindowMap, NoiseCompleter, DiffusersCompleter
    kind = config["adapter"]
    if kind == "oracle":
        raise ValueError("Oracle requires evaluator-only construction")
    if kind in {"copy", "random", "noise-map"}:
        return {"copy": CopyModel, "random": RandomModel, "noise-map": StaticNoiseMap}[kind]()
    if kind == "numeric-camera":
        from .training import load_approved_backend
        from .numeric import NumericLastFrameAdapter
        from .adapters.camera_conditioning import CameraConfig
        backend = load_approved_backend(copy.deepcopy(config["backend_manifest"]),
                                        device=config.get("device", "cuda"))
        return NumericLastFrameAdapter(backend, CameraConfig(**config["camera"]),
                                       expected_camera_sha256=config["training_camera_sha256"])
    if kind == "wan-i2v":
        from .video import WanI2VAdapter
        return WanI2VAdapter(config["model_dir"], config.get("device", "cuda"),
                             config.get("steps", 30), prompt=config.get("prompt"))
    completer = NoiseCompleter() if config.get("diagnostic_noise") else DiffusersCompleter(
        config["model_dir"], config.get("device", "cuda"), config.get("steps", 30),
        config.get("prompt", "nadir orthorectified RGB aerial photograph"))
    if kind in {"raster", "frontier", "lazy"}:
        from .frontier import AuditedWindowMap
        return AuditedWindowMap(completer, order=kind)
    return WindowMap(completer, eager=kind == "eager-map")
