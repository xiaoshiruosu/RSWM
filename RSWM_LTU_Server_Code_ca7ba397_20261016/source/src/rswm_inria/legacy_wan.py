"""Bridge to the supplied, hash-locked Wan camera sampler, without a fallback.

The source deployment is made by ``scripts/prepare_legacy_wan.py``.  Importing
this module needs no Torch; loading the actual original models requires their
complete local components and CUDA.  Software doubles never certify weights,
GPU operation or actual pretrained camera-layer consumption.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import random
import re
import sys
import threading
import uuid

import numpy as np
from PIL import Image

from .geometry import digest, file_hash
from .legacy_training import LegacyTrainingMixin
from .numeric import payload_hash
from .training import locked_tree

SCHEMA = "rswm-legacy-wan-backend-v1"
SCHEMA_V2 = "rswm-legacy-wan-backend-v2"
CONFIG_FIELDS = {"schema", "mode", "config_key", "sampling", "lora_subdir", "training"}
SAMPLING_FIELDS = {"prompt", "negative_prompt", "steps", "shift", "guide_scale", "boundary",
                   "param_dtype", "t5_dtype", "t5_cpu", "inactive_model_cpu"}
FP32_COMPONENTS = ["time_embedding", "time_projection", "head", "norm3"]
_SOURCE_LOCK = None
_LOAD_LOCK = threading.RLock()
_CACHE_FIELDS = ("_conv_num", "_conv_idx", "_feat_map", "_enc_conv_num", "_enc_conv_idx", "_enc_feat_map")


def _json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def _relative(value):
    p = Path(value) if isinstance(value, str) else None
    if p is None or not value or not p.parts or "\\" in value or p.is_absolute() or any(part in {".", ".."} for part in p.parts):
        raise ValueError("Safe nonempty relative component path required")
    return p


def validate_backend_config(value):
    """Validate explicit settings, never manufacture an approved configuration."""
    if not isinstance(value, dict) or value.get("schema") not in {SCHEMA, SCHEMA_V2}:
        raise ValueError("Complete locked legacy backend configuration required")
    v2 = value["schema"] == SCHEMA_V2
    if set(value) != CONFIG_FIELDS | ({"adapter_release"} if v2 else set()):
        raise ValueError("Complete locked legacy backend configuration required")
    if v2:
        from .legacy_export import validate_release_binding
        validate_release_binding(value["adapter_release"])
        if value["mode"] != "restored_lora" or value["lora_subdir"] is not None or value["training"] is not None:
            raise ValueError("Trained release requires fresh restored inference with training=null")
    if value["mode"] not in {"base", "restored_lora"} or value["config_key"] != "i2v-A14B":
        raise ValueError("Explicit base/restored_lora i2v-A14B mode required")
    sampling = value["sampling"]
    if not isinstance(sampling, dict) or set(sampling) not in (SAMPLING_FIELDS, SAMPLING_FIELDS | {"block_cpu_offload"}):
        raise ValueError("Complete locked sampling settings required")
    if not isinstance(sampling["prompt"], str) or not sampling["prompt"].strip() or not isinstance(sampling["negative_prompt"], str):
        raise ValueError("Explicit fixed positive/negative text required")
    if type(sampling["steps"]) is not int or sampling["steps"] <= 0:
        raise ValueError("Positive integer sampling steps required")
    for name in ("shift", "boundary"):
        x = sampling[name]
        if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
            raise ValueError("Finite positive sampling " + name + " required")
    if sampling["boundary"] >= 1:
        raise ValueError("Expert boundary must lie strictly between zero and one")
    guide = sampling["guide_scale"]
    if not isinstance(guide, list) or len(guide) != 2 or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in guide):
        raise ValueError("Finite nonnegative low/high guidance pair required")
    if sampling["param_dtype"] != "bf16" or sampling["t5_dtype"] != "bf16":
        raise ValueError("Original selected loaders require explicit bf16 DiT/T5")
    if any(type(sampling[k]) is not bool for k in ("t5_cpu", "inactive_model_cpu")):
        raise ValueError("Explicit boolean offload settings required")
    if "block_cpu_offload" in sampling and type(sampling["block_cpu_offload"]) is not bool:
        raise ValueError("Explicit boolean block CPU offload setting required")
    if sampling.get("block_cpu_offload", False) and not (sampling["t5_cpu"] and sampling["inactive_model_cpu"]):
        raise ValueError("Block CPU offload requires T5 and inactive experts loaded on CPU")
    if value["mode"] == "base":
        if value["lora_subdir"] is not None:
            raise ValueError("Base mode cannot load a LoRA directory")
    elif not v2:
        _relative(value["lora_subdir"])
    if value["training"] is not None and not isinstance(value["training"], dict):
        raise ValueError("Training settings must be explicit object or null")
    if value["training"] is not None:
        from .legacy_training import validate_training_spec
        spec = validate_training_spec(value["training"])
        block_cpu = spec["expert_residency"] == "block_cpu"
        if block_cpu != sampling.get("block_cpu_offload", False):
            raise ValueError("Training expert residency and sampler block offload disagree")
        if sampling["inactive_model_cpu"] and not block_cpu:
            raise ValueError("Training requires explicit resident experts without optimizer offload")
    return copy.deepcopy(value)


def _source(source_dir):
    root = Path(source_dir)
    tree = locked_tree(root)
    root = root.resolve()
    inventory = _json(root / "source_inventory.json")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("Prepared source inventory required")
    for name, expected in inventory.items():
        _relative(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected) or tree.get(name) != expected:
            raise ValueError("Prepared legacy source changed: " + name)
    allowed = {"source_inventory.json", "backend_config.json", "backend.py"}
    if set(tree) - set(inventory) - allowed:
        raise ValueError("Unexpected unlocked prepared source files")
    for required in ("inference.py", "base_loader.py", "wan/__init__.py", "wan/modules/model.py",
                     "wan/modules/vae2_1.py", "wan/modules/t5.py", "wan/utils/cam_utils.py", "wan/configs/wan_i2v_A14B.py"):
        if required not in inventory:
            raise ValueError("Prepared executable source absent: " + required)
    config = validate_backend_config(_json(root / "backend_config.json"))
    return root, tree, config


def component_inventory(weights_dir, config, *, require_complete=False):
    """Read actual local configs, index shards and component bytes without Torch."""
    config = validate_backend_config(config)
    root = Path(weights_dir)
    if root.exists():
        tree = locked_tree(root)
    else:
        tree = {}
    rows = []

    def add(label, name, *, directory=False):
        relative = _relative(name).as_posix()
        selected = {k: v for k, v in tree.items() if k.startswith(relative + "/")} if directory else ({relative: tree[relative]} if relative in tree else {})
        files = [{"path": k, "sha256": v, "bytes": (root / k).stat().st_size} for k, v in sorted(selected.items())]
        status = "PRESENT" if files and all(f["bytes"] > 0 for f in files) else "MISSING_OR_EMPTY"
        row = {"component": label, "path": relative, "status": status, "files": files}
        rows.append(row)
        return row

    add("t5", "models_t5_umt5-xxl-enc-bf16.pth")
    add("vae", "Wan2.1_VAE.pth")
    tokenizer = add("tokenizer", "google/umt5-xxl", directory=True)
    token_names = {f["path"] for f in tokenizer["files"]}
    if "google/umt5-xxl/tokenizer_config.json" not in token_names or not ({"google/umt5-xxl/tokenizer.json", "google/umt5-xxl/spiece.model"} & token_names):
        tokenizer["status"] = "MISSING_OR_EMPTY"
        tokenizer["reason"] = "Actual local tokenizer config and tokenizer vocabulary required"
    for tag in ("low", "high"):
        name = tag + "_noise_model"
        row = add(tag + "_expert", name, directory=True)
        if name + "/config.json" not in tree:
            row["status"] = "MISSING_OR_EMPTY"
            continue
        model_config = _json(root / name / "config.json")
        row["checkpoint_config"] = model_config
        if not isinstance(model_config, dict) or model_config.get("model_type") != "i2v" or model_config.get("in_dim") != 36:
            raise ValueError("Actual expert checkpoint must declare i2v in_dim=36")
        candidates = {name + "/diffusion_pytorch_model.safetensors", name + "/diffusion_pytorch_model.bin"} & set(tree)
        indices = sorted({name + "/diffusion_pytorch_model.safetensors.index.json", name + "/diffusion_pytorch_model.bin.index.json"} & set(tree))
        if not candidates and not indices:
            row["status"] = "MISSING_OR_EMPTY"
        for index in indices:
            value = _json(root / index)
            mapping = value.get("weight_map") if isinstance(value, dict) else None
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Actual expert shard index has no weight_map")
            shards = sorted({_relative(v).as_posix() for v in mapping.values()})
            row.setdefault("index_shards", {})[index] = shards
            if any(name + "/" + shard not in tree or (root / name / shard).stat().st_size <= 0 for shard in shards):
                row["status"] = "MISSING_OR_EMPTY"
                row["reason"] = "Checkpoint index refers to a missing/empty shard"
    if config["mode"] == "restored_lora":
        if config["schema"] == SCHEMA_V2:
            from .legacy_export import validate_adapter_release
            release = validate_adapter_release(config["adapter_release"], expected_weights_tree_sha256=digest(tree))
            inference_config = _json(Path(config["adapter_release"]["root"]) / "inference-config.json")
            if config["sampling"] != inference_config["sampling"]:
                raise ValueError("Trained release inference sampling settings changed")
            release_root = Path(config["adapter_release"]["root"])
            for tag in ("low", "high"):
                name = release["experts"][tag]["path"]
                files = [{"path": k, "sha256": v, "bytes": (release_root / k).stat().st_size}
                         for k, v in sorted(release["files"].items()) if k.startswith(name + "/")]
                rows.append({"component": name + "_lora", "path": str(release_root / name), "status": "PRESENT", "files": files})
        else:
            for tag in ("low_noise", "high_noise"):
                name = (_relative(config["lora_subdir"]) / tag).as_posix()
                row = add(tag + "_lora", name, directory=True)
                names = {f["path"] for f in row["files"]}
                if name + "/adapter_config.json" not in names or not ({name + "/adapter_model.safetensors", name + "/adapter_model.bin"} & names):
                    row["status"] = "MISSING_OR_EMPTY"
                    row["reason"] = "Each restored expert needs actual adapter config and weights"
    complete = all(row["status"] == "PRESENT" for row in rows)
    result = {"schema": "rswm-legacy-wan-components-v1", "weights_dir": str(root.resolve()),
              "mode": config["mode"], "components": rows, "complete": complete,
              "weights_tree_sha256": digest(tree) if tree else None,
              "scope": "Actual component bytes/config inventory; not pretrained forward validation"}
    if require_complete and not complete:
        raise ValueError("BLOCKED: actual local Wan components missing: " + ", ".join(r["component"] for r in rows if r["status"] != "PRESENT"))
    return result


def inspect_backend(*, source_dir, weights_dir=None):
    root, tree, config = _source(source_dir)
    return {"source_dir": str(root), "source_tree_sha256": digest(tree), "config": config,
            "config_sha256": digest(config), "components": component_inventory(weights_dir, config) if weights_dir is not None else None,
            "real_model_validation": "NOT_RUN", "real_gpu_validation": "NOT_RUN"}


def _confirm_loaded_inputs(root, tree, config, weights_dir, components):
    """Check retained bytes again after the original loading transaction."""
    current_root, current_tree, current_config = _source(root)
    if current_root != root or current_tree != tree or current_config != config:
        raise ValueError("Prepared source/config changed during actual model loading")
    if digest(locked_tree(weights_dir)) != components["weights_tree_sha256"]:
        raise ValueError("Actual weight components changed during model loading")
    if config["schema"] == SCHEMA_V2:
        from .legacy_export import validate_adapter_release
        validate_adapter_release(config["adapter_release"], expected_weights_tree_sha256=components["weights_tree_sha256"])


def strict_from_pretrained(model_class, *args, **kwargs):
    """Use the original loader, rejecting its otherwise silent random weights."""
    # Original Wan explicitly computes time embeddings/output projection and
    # affine LayerNorm in FP32. Preserve checkpoint values at load time,
    # before a BF16 conversion can discard precision. DiT blocks stay BF16.
    model_class._keep_in_fp32_modules = FP32_COMPONENTS.copy()
    kwargs.update(local_files_only=True, output_loading_info=True)
    loaded = model_class.from_pretrained(*args, **kwargs)
    if not isinstance(loaded, tuple) or len(loaded) != 2 or not isinstance(loaded[1], dict):
        raise ValueError("Actual original loader must return weight loading information")
    model, info = loaded
    keys = {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
    if not keys.issubset(info) or any(info[key] for key in keys):
        raise ValueError("Original checkpoint loading incomplete: " + json.dumps(info, default=str))
    if getattr(model, "model_type", None) != "i2v" or tuple(getattr(model, "patch_size", ())) != (1, 2, 2) or getattr(model, "in_dim", None) != 36:
        raise ValueError("Loaded original expert architecture does not match i2v native mapping")
    expected = {"dim": 5120, "ffn_dim": 13824, "freq_dim": 256, "num_heads": 40,
                "num_layers": 40, "text_len": 512, "out_dim": 16}
    if any(getattr(model, key, None) != value for key, value in expected.items()) or len(getattr(model, "blocks", ())) != 40:
        raise ValueError("Loaded expert differs from the selected original i2v-A14B architecture")
    projector = getattr(model, "patch_embedding_wancamctrl", None)
    if getattr(projector, "in_features", None) != 1536 or getattr(projector, "out_features", None) != getattr(model, "dim", None):
        raise ValueError("Loaded expert lacks the actual Wan camera projector")
    for block in getattr(model, "blocks", ()):
        if any(not callable(getattr(block, name, None)) for name in
               ("cam_injector_layer1", "cam_injector_layer2", "cam_scale_layer", "cam_shift_layer")):
            raise ValueError("Loaded expert lacks actual camera injection layers")
    if not len(getattr(model, "blocks", ())):
        raise ValueError("Loaded original expert has no transformer blocks")
    return model


def strict_lora_from_pretrained(model_class, model, path, **kwargs):
    """Compare retained LoRA keys and values before the original merge step."""
    root = Path(path)
    if (not root.is_dir() or not (root / "adapter_config.json").is_file()
            or not any((root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))):
        raise ValueError("Actual local LoRA config and weights required")
    if kwargs.get("adapter_name", "default") != "default" or kwargs.get("key_mapping") is not None:
        raise ValueError("Original default LoRA identity cannot be remapped")
    torch = importlib.import_module("torch")
    helper = importlib.import_module("peft.utils.save_and_load")
    retained = helper.load_peft_weights(str(root.resolve()), device="cpu", local_files_only=True)
    kwargs["local_files_only"] = True
    loaded = model_class.from_pretrained(model, str(root.resolve()), **kwargs)
    actual = helper.get_peft_model_state_dict(loaded, adapter_name="default", save_embedding_layers=False)
    if not isinstance(retained, dict) or not retained or not isinstance(actual, dict) or set(retained) != set(actual):
        raise ValueError("Actual LoRA checkpoint has missing/unexpected adapter keys")
    for key, expected in retained.items():
        value = actual[key]
        if (not torch.is_tensor(expected) or not torch.is_tensor(value) or expected.shape != value.shape
                or not torch.isfinite(expected).all().item() or not torch.isfinite(value).all().item()):
            raise ValueError("Actual LoRA checkpoint shape/nonfinite tensor mismatch: " + key)
        converted = expected.detach().to(device="cpu", dtype=value.dtype)
        if not torch.isfinite(converted).all().item() or not torch.equal(value.detach().cpu(), converted):
            raise ValueError("Actual LoRA loaded tensor differs from retained checkpoint: " + key)
    return loaded


def _module_origin(module, root, tree):
    path = getattr(module, "__file__", None)
    if not path:
        raise ValueError("Locked original module has no source file")
    p = Path(path)
    if p.suffix != ".py" or p.is_symlink():
        raise ValueError("Locked original module must import source bytes")
    try:
        relative = p.resolve().relative_to(root).as_posix()
    except ValueError:
        raise ValueError("Imported original module escaped selected prepared source: " + str(p)) from None
    if tree.get(relative) != file_hash(p):
        raise ValueError("Imported original source differs from locked bytes")
    return {"path": str(p.resolve()), "sha256": tree[relative]}


def _origins(root, tree, modules=()):
    result = {}
    for name, module in list(sys.modules.items()):
        if name == "wan" or name.startswith("wan.") or name in {"inference", "evaluate_baseline", "base_loader"}:
            result[name] = _module_origin(module, root, tree)
    for module in modules:
        result[module.__name__] = _module_origin(module, root, tree)
    return result


def _import_file(root, tree, filename):
    path = root / filename
    name = "rswm_locked_legacy_" + digest(tree)[:16] + "_" + path.stem
    if name in sys.modules:
        module = sys.modules[name]
        _module_origin(module, root, tree)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_backend(*, source_dir, weights_dir, device):
    """Registry entrypoint for actual original base or restored-LoRA components."""
    global _SOURCE_LOCK
    with _LOAD_LOCK:
        root, tree, config = _source(source_dir)
        components = component_inventory(weights_dir, config, require_complete=True)
        _origins(root, tree)
        unowned = [name for name in sys.modules if name == "wan" or name.startswith("wan.") or name.startswith("rswm_locked_legacy_") or name in {"inference", "evaluate_baseline", "base_loader"}]
        if _SOURCE_LOCK is None and unowned:
            raise ValueError("Pre-existing unowned original module cache cannot certify current locked bytes")
        identity = (str(root), digest(tree))
        if _SOURCE_LOCK is not None and _SOURCE_LOCK != identity:
            raise ValueError("One process cannot mix different locked Wan source trees")
        if not isinstance(device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", device):
            raise ValueError("BLOCKED: original actual Wan sampler requires CUDA; CPU doubles are software tests only")
        try:
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ValueError("BLOCKED: actual Wan Torch dependency unavailable") from exc
        if not torch.cuda.is_available():
            raise ValueError("BLOCKED: actual CUDA device unavailable")
        device_obj = torch.device(device)
        if device_obj.index is None:
            device_obj = torch.device("cuda", torch.cuda.current_device())
        if device_obj.index is not None and device_obj.index >= torch.cuda.device_count():
            raise ValueError("BLOCKED: requested actual CUDA device unavailable")
        old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
        before_modules = set(sys.modules)
        try:
            sys.path.insert(0, str(root)); sys.dont_write_bytecode = True
            inference = _import_file(root, tree, "inference.py")
            baseline = _import_file(root, tree, "base_loader.py")
            configs = importlib.import_module("wan.configs")
            wan_config = copy.deepcopy(configs.WAN_CONFIGS[config["config_key"]])
            if tuple(wan_config.vae_stride) != (4, 8, 8) or tuple(wan_config.patch_size) != (1, 2, 2):
                raise ValueError("Actual VAE/DiT mapping incompatible with 17/33-frame contract")
            sampling = config["sampling"]
            wan_config.boundary = sampling["boundary"]
            wan_config.param_dtype = torch.bfloat16; wan_config.t5_dtype = torch.bfloat16
            wan_config.sample_neg_prompt = sampling["negative_prompt"]
            arguments = {"checkpoint_dir": str(Path(weights_dir).resolve()), "device": device_obj,
                         "config": wan_config, "t5_cpu": sampling["t5_cpu"],
                         "inactive_model_cpu": sampling["inactive_model_cpu"]}
            if config["mode"] == "base":
                parts = baseline.load_base_model(**arguments)
            else:
                if config["schema"] == SCHEMA_V2:
                    from .legacy_export import validate_adapter_tensor_content
                    validate_adapter_tensor_content(config["adapter_release"])
                    arguments["lora_path"] = config["adapter_release"]["root"]
                else:
                    arguments["lora_path"] = str(Path(weights_dir).resolve() / config["lora_subdir"])
                arguments["merge_lora"] = config["training"] is None and config["schema"] != SCHEMA_V2
                parts = inference.load_lora_model(**arguments)
                if config["schema"] == SCHEMA_V2:
                    from .legacy_export import validate_loaded_expert
                    text_encoder, vae, low, high = parts
                    merged = []
                    for tag, expert in (("low", low), ("high", high)):
                        # PEFT may upcast adapter tensors on load. Restore the
                        # driver's explicit LoRA BF16 without rounding the
                        # original FP32 time/head/norm backbone components.
                        for name, parameter in expert.named_parameters():
                            if "lora_" in name.lower():
                                parameter.data = parameter.data.to(dtype=torch.bfloat16)
                        validate_loaded_expert(expert, tag, config["adapter_release"])
                        model = expert.merge_and_unload(safe_merge=True)
                        model.eval().requires_grad_(False)
                        merged.append(model)
                    parts = (text_encoder, vae, *merged)
            _confirm_loaded_inputs(root, tree, config, weights_dir, components)
            origins = _origins(root, tree, (inference, baseline))
            backend = LegacyWanBackend(parts=parts, inference=inference, wan_config=wan_config,
                                       runtime_config=config, source_dir=root, source_tree=tree,
                                       components=components, torch=torch, device=device_obj,
                                       actual_weights_loaded=True)
            _SOURCE_LOCK = identity
            backend.imported_source = origins
            return backend
        except (ImportError, ModuleNotFoundError) as exc:
            raise ValueError("BLOCKED: original Wan runtime dependency unavailable: " + str(exc)) from exc
        finally:
            sys.path[:] = old_path; sys.dont_write_bytecode = old_bytecode
            # A failed import must not reserve a partly loaded different source.
            if _SOURCE_LOCK != identity:
                for name in set(sys.modules) - before_modules:
                    if name == "wan" or name.startswith("wan.") or name.startswith("rswm_locked_legacy_"):
                        sys.modules.pop(name, None)


class LegacyWanBackend(LegacyTrainingMixin):
    objective = "flow_matching"

    def __init__(self, *, parts, inference, wan_config, runtime_config, source_dir, source_tree,
                 components, torch, device, actual_weights_loaded=False):
        self.torch, self.device, self.dtype = torch, device, torch.bfloat16
        self.text_encoder, self.vae, low, high = parts
        self.model = torch.nn.ModuleDict({"low": low, "high": high})
        self.inference, self.wan_config = inference, wan_config
        self.runtime_config = validate_backend_config(runtime_config)
        self._runtime_config_sha256 = digest(self.runtime_config)
        self._wan_config_sha256 = digest(self._wan_settings())
        self.source_dir, self.source_tree = Path(source_dir), copy.deepcopy(source_tree)
        self.weights_dir = Path(components["weights_dir"])
        self.components = copy.deepcopy(components)
        self.identity_sha256 = digest({"source": source_tree, "config": runtime_config,
                                      "weights": components["weights_tree_sha256"]})
        self.actual_weights_loaded = actual_weights_loaded
        self.resource_counter_epoch = uuid.uuid4().hex
        self.counters = {"native_frames": 0, "dit_forward_calls": 0, "dit_forward_frame_spans": 0, "decoded_frames": 0,
                         "generate_calls": 0, "failed_generate_calls": 0, "snapshot_calls": 0, "restore_calls": 0}
        self.call_index = 0; self.last_layer_observation = None; self._busy = False
        self.generation_observations = []
        self.imported_source = {}
        if runtime_config["training"] is None:
            self.model.eval().requires_grad_(False)
        self._offload_controllers = {}
        if self.runtime_config["sampling"].get("block_cpu_offload", False):
            from .legacy_offload import attach_frozen_cpu_offload
            # Loaders must leave original experts on CPU; never transiently
            # materialize both camera backbones on the selected GPU.
            for tag in ("low", "high"):
                self.model[tag].requires_grad_(False)
                self._offload_controllers[tag] = attach_frozen_cpu_offload(self.model[tag], device=self.device)

    def _check_source(self):
        if digest(self.runtime_config) != self._runtime_config_sha256 or digest(self._wan_settings()) != self._wan_config_sha256:
            raise ValueError("Loaded in-memory sampling/training configuration changed")
        if locked_tree(self.source_dir) != self.source_tree:
            raise ValueError("Loaded legacy source/config changed")
        if self.actual_weights_loaded:
            self.imported_source = _origins(self.source_dir, self.source_tree, (self.inference,))

    def _wan_settings(self):
        keys = ("boundary", "num_train_timesteps", "vae_stride", "patch_size", "sample_neg_prompt",
                "dim", "ffn_dim", "freq_dim", "num_heads", "num_layers", "text_len", "t5_checkpoint",
                "t5_tokenizer", "vae_checkpoint", "low_noise_checkpoint", "high_noise_checkpoint")
        return {**{k: getattr(self.wan_config, k, None) for k in keys},
                **{k: str(getattr(self.wan_config, k, None)) for k in ("param_dtype", "t5_dtype")}}

    def training_components(self):
        return {"vae": self.vae, "text_encoder": self.text_encoder,
                "experts": {key: self.model[key] for key in ("low", "high")},
                "wan_config": self.wan_config, "device": self.device, "dtype": self.dtype}

    def replace_training_experts(self, experts):
        if not isinstance(experts, dict) or set(experts) != {"low", "high"}:
            raise ValueError("Both actual low/high experts required")
        for key in ("low", "high"):
            self.model[key] = experts[key]

    def make_lora_expert(self, *, model, rank, alpha, targets):
        self._check_source()
        helper = importlib.import_module("wan.modules.animate.animate_utils")
        _module_origin(helper, self.source_dir, self.source_tree)
        peft = importlib.import_module("peft")
        config = helper.get_loraconfig(model, rank=rank, alpha=alpha, targets=targets, init_lora_weights="gaussian")
        return peft.get_peft_model(model, config)

    def _tensor_summary(self, tensor):
        torch = self.torch
        value = tensor.detach()
        if not torch.isfinite(value).all().item():
            raise ValueError("Nonfinite actual camera layer input")
        flat = value.reshape(-1)
        # Float32 linspace can round numel-1 up to numel for large camera
        # tensors. Keep the selected positions integral before indexing.
        sample_count = min(128, flat.numel())
        ids = torch.tensor([i*(flat.numel()-1)//max(1, sample_count-1) for i in range(sample_count)],
                           device=flat.device, dtype=torch.int64)
        samples = flat[ids].float().cpu().numpy()
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
                "finite": True, "l2": float(torch.linalg.vector_norm(value, dtype=torch.float64).item()),
                "sample_indices": ids.cpu().tolist(), "sample_values": samples.tolist(),
                "sample_sha256": hashlib.sha256(samples.tobytes()).hexdigest(),
                "scope": "Actual hooked tensor with fixed selected-value summary, not a full tensor hash"}

    def _hooks(self, native_frames):
        records, counts, forwards, handles = {}, {}, {}, []
        layers = {}
        for tag in ("low", "high"):
            names = dict(self.model[tag].named_modules())
            for name in ("patch_embedding_wancamctrl", "blocks.0.cam_injector_layer1",
                         "blocks.0.cam_scale_layer", "blocks.0.cam_shift_layer"):
                if name not in names:
                    raise ValueError("Actual camera hook layer absent: " + tag + "." + name)
                layers[tag, name] = names[name]
        counted_native_span = []
        for tag in ("low", "high"):
            expert = self.model[tag]
            def forward_hook(module, args, kwargs, tag=tag):
                x = args[0] if args else kwargs.get("x")
                if not isinstance(x, (tuple, list)) or len(x) != 1 or not self.torch.is_tensor(x[0]):
                    raise ValueError("Actual entered DiT latent input unavailable")
                span = (x[0].shape[1] - 1) * 4 + 1
                if span != native_frames:
                    raise ValueError("Actual entered DiT temporal mapping changed")
                self.counters["dit_forward_calls"] += 1
                self.counters["dit_forward_frame_spans"] += span
                forwards[tag] = forwards.get(tag, 0) + 1
                # Observe native work when a real DiT call is entered, not by
                # predicting valid PNGs. CFG/sampling repetitions are separate.
                if not counted_native_span:
                    self.counters["native_frames"] += span
                    counted_native_span.append(span)
            handles.append(expert.register_forward_pre_hook(forward_hook, with_kwargs=True))
            for name in ("patch_embedding_wancamctrl", "blocks.0.cam_injector_layer1",
                         "blocks.0.cam_scale_layer", "blocks.0.cam_shift_layer"):
                layer = layers[tag, name]
                key = tag + "." + name
                def camera_hook(module, args, key=key):
                    counts[key] = counts.get(key, 0) + 1
                    if key not in records:
                        if len(args) != 1 or not self.torch.is_tensor(args[0]):
                            raise ValueError("Actual camera layer tensor input absent")
                        records[key] = self._tensor_summary(args[0])
                handles.append(layer.register_forward_pre_hook(camera_hook))
        return handles, records, counts, forwards

    def generate(self, *, image, camera, native_frames, seed):
        try:
            camera_sha = payload_hash(camera)
        except (TypeError, ValueError, AttributeError):
            camera_sha = None
        record = {"call_id": len(self.generation_observations) + 1, "seed": seed if type(seed) is int else None,
                  "native_frames": native_frames if type(native_frames) is int else None, "camera_payload_sha256": camera_sha,
                  "camera_manifest_sha256": camera.get("camera_manifest_sha256") if isinstance(camera, dict) and isinstance(camera.get("camera_manifest_sha256"), str) else None,
                  "backend_config_sha256": digest(self.runtime_config), "backend_identity_sha256": self.identity_sha256,
                  "scope": "actual_model_layer_observation" if self.actual_weights_loaded else "software_test_double",
                  "real_backend_layer_consumption": "NOT_RUN",
                  "counters_before": dict(self.counters), "records": {}, "layer_call_counts": {}, "forward_calls_by_expert": {}}
        self.generation_observations.append(record)
        try:
            result = self._generate(image=image, camera=camera, native_frames=native_frames, seed=seed, observation=record)
            record.update(status="SUCCEEDED", returned_native_frames=len(result["frames"]), error_type=None)
            return result
        except BaseException as exc:
            record.update(status="FAILED", returned_native_frames=None, error_type=type(exc).__name__)
            raise
        finally:
            record["counters_after"] = dict(self.counters)
            if self.actual_weights_loaded and record["records"]:
                record["real_backend_layer_consumption"] = "OBSERVED_NOT_RELEASED" if record["status"] == "SUCCEEDED" else "PARTIAL_OBSERVATION_NOT_RELEASED"

    def _generate(self, *, image, camera, native_frames, seed, observation):
        self._check_source()
        if self.runtime_config["training"] is not None or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("Generation requires a separate explicit frozen inference configuration/checkpoint")
        if self._busy or type(native_frames) is not int or native_frames not in (17, 33) or type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Complete explicit 17/33 native frame call/seed required")
        if not isinstance(image, np.ndarray) or image.shape != (480, 832, 3) or image.dtype != np.uint8:
            raise ValueError("Native RGB480x832 uint8 initial image required")
        # Validate the shared public camera before any expensive real forward.
        from .legacy_camera import validate_prepared_camera
        validate_prepared_camera(camera, native_frames, self.wan_config, 480, 832, 60, 104)
        self._busy = True; self.counters["generate_calls"] += 1
        handles = []
        try:
            handles, records, counts, forwards = self._hooks(native_frames)
            observation.update(records=records, layer_call_counts=counts, forward_calls_by_expert=forwards,
                               camera_channel=camera["camera_manifest"]["camera_channel"])
            sampling = self.runtime_config["sampling"]
            offload_options = {"block_cpu_offload": True} if sampling.get("block_cpu_offload", False) else {}
            video = self.inference.generate_video(
                text_encoder=self.text_encoder, vae=self.vae, low_noise_model=self.model["low"],
                high_noise_model=self.model["high"], config=self.wan_config,
                prompt=sampling["prompt"], image=Image.fromarray(image), device=self.device,
                frame_num=native_frames, size="480*832", action_path=None, seed=seed,
                sampling_steps=sampling["steps"], shift=sampling["shift"],
                guide_scale=tuple(sampling["guide_scale"]), t5_cpu=sampling["t5_cpu"],
                framewise=camera["camera_manifest"]["framewise"],
                inactive_model_cpu=sampling["inactive_model_cpu"], prepared_camera=camera, **offload_options)
            if not self.torch.is_tensor(video) or tuple(video.shape) != (3, native_frames, 480, 832) or not self.torch.isfinite(video).all().item():
                raise ValueError("Actual original sampler returned invalid native RGB tensor")
            self.counters["decoded_frames"] += native_frames
            channel = camera["camera_manifest"]["camera_channel"]
            if channel == "drop" and counts:
                raise ValueError("Drop condition entered an actual camera layer")
            expected_layers = {tag + "." + name for tag in forwards for name in
                               ("patch_embedding_wancamctrl", "blocks.0.cam_injector_layer1",
                                "blocks.0.cam_scale_layer", "blocks.0.cam_shift_layer")}
            if not forwards or channel != "drop" and set(records) != expected_layers:
                raise ValueError("Camera condition did not enter any actual model camera layer")
            self.last_layer_observation = {"scope": "actual_model_layer_observation" if self.actual_weights_loaded else "software_test_double",
                "camera_payload_sha256": payload_hash(camera), "camera_channel": channel,
                "records": records, "layer_call_counts": counts, "forward_calls_by_expert": forwards,
                "real_backend_layer_consumption": ("OBSERVED_NOT_RELEASED" if records else "NO_CAMERA_PATH_NOT_RELEASED") if self.actual_weights_loaded else "NOT_RUN"}
            # Match the supplied backbone conversion, without video compression.
            array = ((video.clamp(-1, 1).permute(1, 2, 3, 0) + 1) * 127.5).byte().cpu().numpy()
            self.call_index += 1
            return {"frames": [Image.fromarray(frame.copy(), "RGB") for frame in array],
                    "camera_payload_sha256": payload_hash(camera),
                    "conditioning_observation": copy.deepcopy(self.last_layer_observation)}
        except BaseException:
            self.counters["failed_generate_calls"] += 1
            raise
        finally:
            for handle in handles:
                handle.remove()
            self.quiesce_training()
            self._busy = False

    def _clone(self, value):
        if self.torch.is_tensor(value):
            return value.detach().clone()
        if isinstance(value, dict):
            return {k: self._clone(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._clone(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self._clone(v) for v in value)
        return copy.deepcopy(value)

    def _mutable(self, *, clone=True):
        modules = {"low": self.model["low"], "high": self.model["high"], "vae": self.vae.model, "text": self.text_encoder.model}
        retain = self._clone if clone else lambda value: value
        buffers = {key: {name: retain(value) for name, value in module.named_buffers()} for key, module in modules.items()}
        return {"buffers": buffers, "freqs": {tag: retain(getattr(self.model[tag], "freqs", None)) for tag in ("low", "high")},
                "vae_cache": {k: retain(getattr(self.vae.model, k)) for k in _CACHE_FIELDS if hasattr(self.vae.model, k)},
                "vae_attributes": {k: retain(getattr(self.vae, k)) for k in ("mean", "std", "scale", "device", "dtype") if hasattr(self.vae, k)},
                "placements": {key: str(next(module.parameters()).device) for key, module in modules.items()},
                "training_modes": {key: {name: m.training for name, m in module.named_modules()} for key, module in modules.items()}}

    def _rng(self):
        torch = self.torch
        return {"python": random.getstate(), "numpy": self._clone(np.random.get_state()), "torch": torch.get_rng_state().clone(),
                "cuda": torch.cuda.get_rng_state(self.device).clone() if self.device.type == "cuda" else None}

    def reset(self, *, seed):
        if type(seed) is not int or not 0 <= seed < 2**63 or self._busy:
            raise ValueError("Explicit reset seed at call boundary required")
        self._check_source(); random.seed(seed); np.random.seed(seed % 2**32); self.torch.random.default_generator.manual_seed(seed)
        if self.device.type == "cuda":
            with self.torch.cuda.device(self.device): self.torch.cuda.manual_seed(seed)
        self.vae.model.clear_cache()
        self.call_index = 0; self.last_layer_observation = None

    def snapshot(self):
        if self._busy or self.runtime_config["training"] is not None or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("F7 snapshot requires frozen actual experts at call boundary")
        self.quiesce_training()
        self._check_source(); self.counters["snapshot_calls"] += 1
        return {"schema": "rswm-legacy-wan-state-v1", "owner": self.resource_counter_epoch,
                "identity_sha256": self.identity_sha256, "call_index": self.call_index,
                "mutable": self._mutable(), "rng": self._rng()}

    def _validate_restore(self, mutable, rng, modules):
        """Reject malformed snapshots before changing modules or global RNG."""
        torch = self.torch

        def exact(value, keys):
            if not isinstance(value, dict) or set(value) != set(keys):
                raise ValueError("Mutable actual state keys mismatch")

        def device(value):
            try:
                parsed = torch.device(value)
            except (TypeError, RuntimeError, ValueError):
                raise ValueError("Invalid actual state device") from None
            if parsed != torch.device("cpu") and parsed != self.device:
                raise ValueError("Actual state device outside selected CPU/CUDA scope")
            return parsed

        def tensor(value, current=None):
            if not torch.is_tensor(value) or value.layout != torch.strided or value.requires_grad:
                raise ValueError("Actual state tensor required")
            device(value.device)
            if current is not None and (value.shape != current.shape or value.dtype != current.dtype):
                raise ValueError("Mutable actual tensor shape/dtype mismatch")
            if not torch.isfinite(value).all().item():
                raise ValueError("Nonfinite actual state tensor")

        def attribute(value, current):
            if torch.is_tensor(current):
                tensor(value, current)
            elif isinstance(current, (list, tuple)):
                if type(value) is not type(current) or len(value) != len(current):
                    raise ValueError("Actual VAE attribute structure mismatch")
                for item, existing in zip(value, current):
                    attribute(item, existing)
            elif isinstance(current, torch.device):
                if not isinstance(value, torch.device):
                    raise ValueError("Actual VAE device type mismatch")
                device(value)
            elif type(value) is not type(current) or value != current:
                raise ValueError("Actual VAE attribute mismatch")

        exact(mutable, {"buffers", "freqs", "vae_cache", "vae_attributes", "placements", "training_modes"})
        for key in ("buffers", "placements", "training_modes"):
            exact(mutable[key], modules)
        for key, module in modules.items():
            current = dict(module.named_buffers())
            exact(mutable["buffers"][key], current)
            for name, value in current.items():
                tensor(mutable["buffers"][key][name], value)
            placement = mutable["placements"][key]
            if self.runtime_config["sampling"].get("block_cpu_offload", False) and key in {"low", "high"} and placement != "cpu":
                raise ValueError("Block CPU offload snapshot requires CPU expert placement")
            if not isinstance(placement, str) or str(device(placement)) != placement:
                raise ValueError("Canonical actual module placement required")
            exact(mutable["training_modes"][key], dict(module.named_modules()))
            if any(type(value) is not bool for value in mutable["training_modes"][key].values()):
                raise ValueError("Actual module training modes must be booleans")
        exact(mutable["freqs"], {"low", "high"})
        for key in ("low", "high"):
            current = getattr(self.model[key], "freqs", None)
            if current is None:
                if mutable["freqs"][key] is not None:
                    raise ValueError("Unexpected actual expert frequency state")
            else:
                tensor(mutable["freqs"][key], current)
        cache_keys = {key for key in _CACHE_FIELDS if hasattr(self.vae.model, key)}
        exact(mutable["vae_cache"], cache_keys)
        for key in cache_keys:
            value, current = mutable["vae_cache"][key], getattr(self.vae.model, key)
            if key.endswith("_num"):
                if type(value) is not int or value != current:
                    raise ValueError("Actual VAE cache size mismatch")
            elif key.endswith("_idx"):
                limit = getattr(self.vae.model, key.replace("_idx", "_num"), None)
                if (not isinstance(value, list) or len(value) != len(current)
                        or any(type(x) is not int or x < 0 or (limit is not None and x > limit) for x in value)):
                    raise ValueError("Actual VAE cache index mismatch")
            else:
                if not isinstance(value, list) or len(value) != len(current):
                    raise ValueError("Actual VAE feature cache structure mismatch")
                for item in value:
                    if item is None or (isinstance(item, str) and item == "Rep"):
                        continue
                    tensor(item)
                    if item.numel() == 0 or (self.actual_weights_loaded and item.ndim != 5):
                        raise ValueError("Actual VAE feature cache tensor shape mismatch")
        attribute_keys = {key for key in ("mean", "std", "scale", "device", "dtype") if hasattr(self.vae, key)}
        exact(mutable["vae_attributes"], attribute_keys)
        for key in attribute_keys:
            attribute(mutable["vae_attributes"][key], getattr(self.vae, key))
        exact(rng, {"python", "numpy", "torch", "cuda"})
        cpu_rng = rng["torch"]
        if (not torch.is_tensor(cpu_rng) or cpu_rng.dtype != torch.uint8 or cpu_rng.device.type != "cpu"
                or cpu_rng.shape != torch.get_rng_state().shape):
            raise ValueError("Actual CPU RNG state mismatch")
        numpy_rng = rng["numpy"]
        if (not isinstance(numpy_rng, tuple) or len(numpy_rng) != 5 or numpy_rng[0] != "MT19937"
                or not isinstance(numpy_rng[1], np.ndarray) or numpy_rng[1].dtype != np.uint32
                or numpy_rng[1].shape != (624,) or type(numpy_rng[2]) is not int or not 0 <= numpy_rng[2] <= 624
                or type(numpy_rng[3]) is not int or numpy_rng[3] not in (0, 1)
                or type(numpy_rng[4]) not in (int, float) or not math.isfinite(numpy_rng[4])):
            raise ValueError("Actual NumPy RNG state mismatch")
        try:
            random.Random().setstate(rng["python"])
            np.random.RandomState().set_state(numpy_rng)
            torch.Generator(device="cpu").set_state(cpu_rng)
            if self.device.type == "cuda":
                cuda_rng = rng["cuda"]
                if (not torch.is_tensor(cuda_rng) or cuda_rng.dtype != torch.uint8 or cuda_rng.device.type != "cpu"
                        or cuda_rng.shape != torch.cuda.get_rng_state(self.device).shape):
                    raise ValueError("Actual selected CUDA RNG state mismatch")
                torch.Generator(device=self.device).set_state(cuda_rng)
            elif rng["cuda"] is not None:
                raise ValueError("Unexpected CUDA RNG state on CPU")
        except (TypeError, RuntimeError, ValueError) as exc:
            raise ValueError("Invalid actual RNG state") from exc

    def restore(self, state):
        fields = {"schema", "owner", "identity_sha256", "call_index", "mutable", "rng"}
        if self._busy or not isinstance(state, dict) or set(state) != fields or state["schema"] != "rswm-legacy-wan-state-v1" or state["owner"] != self.resource_counter_epoch or state["identity_sha256"] != self.identity_sha256:
            raise ValueError("Foreign/mismatched actual backend state")
        if self.runtime_config["training"] is not None or any(p.requires_grad for p in self.model.parameters()) or type(state["call_index"]) is not int or state["call_index"] < 0:
            raise ValueError("F7 restore requires frozen experts and valid call index")
        self.quiesce_training()
        self._check_source()
        modules = {"low": self.model["low"], "high": self.model["high"], "vae": self.vae.model, "text": self.text_encoder.model}
        self._validate_restore(state["mutable"], state["rng"], modules)
        mutable, rng = self._clone(state["mutable"]), self._clone(state["rng"])
        for key, module in modules.items():
            module.to(mutable["placements"][key])
            for name, value in module.named_buffers():
                value.copy_(mutable["buffers"][key][name].to(value.device))
            for name, submodule in module.named_modules():
                submodule.training = mutable["training_modes"][key][name]
        for tag in ("low", "high"):
            if mutable["freqs"][tag] is not None:
                self.model[tag].freqs = self._clone(mutable["freqs"][tag])
        for key, value in mutable["vae_cache"].items():
            setattr(self.vae.model, key, self._clone(value))
        for key, value in mutable["vae_attributes"].items():
            setattr(self.vae, key, self._clone(value))
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
        self.torch.set_rng_state(rng["torch"].clone().cpu())
        if self.device.type == "cuda": self.torch.cuda.set_rng_state(rng["cuda"].clone().cpu(), self.device)
        self.call_index = state["call_index"]
        self.counters["restore_calls"] += 1

    def resource_counters(self):
        return {"status": "ok", "counter_epoch": self.resource_counter_epoch, "values": dict(self.counters),
                "native_frames_semantics": "Native spans of calls actually entering DiT, once per call including later failures; repeated CFG/sampling DiT calls and decoded frames are separate"}

    def resource_state(self):
        return {"call_index": self.call_index, "mutable": self._mutable(clone=False), "rng": self._rng(),
                "last_layer_observation": copy.deepcopy(self.last_layer_observation),
                "generation_observations": copy.deepcopy(self.generation_observations)}

    def generation_evidence(self):
        """Lifetime telemetry survives F7 restore; it is never model state."""
        return {"schema": "rswm-legacy-wan-generation-evidence-v1", "backend_identity_sha256": self.identity_sha256,
                "source_tree_sha256": digest(self.source_tree), "backend_config_sha256": digest(self.runtime_config),
                "weights_tree_sha256": self.components["weights_tree_sha256"],
                "imported_source": copy.deepcopy(self.imported_source),
                "actual_weights_loaded": self.actual_weights_loaded,
                "real_model_validation": "NOT_RUN", "publication_acceptance": "NOT_READY",
                "observations": copy.deepcopy(self.generation_observations), "counters": dict(self.counters)}
