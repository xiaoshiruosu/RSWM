"""Strict driver-checkpoint to original low/high PEFT inference delivery.

The public exporter requires the existing complete train60 data contract. Tiny
PEFT controls exercise the private software path without approving real Wan.
No base checkpoint is copied, and no source/weight usage approval is created.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile

from .artifact_io import save_bytes_exclusive
from .geometry import canonical, digest, file_hash
from .training import (ClipDataset, locked_tree, load_approved_backend,
                       parameter_content_hash, tensor_state_sha256,
                       validate_checkpoint_receipt, validate_dev_index,
                       validate_resume_cursor)

RELEASE_SCHEMA = "rswm-driver-peft-release-v1"
BINDING_SCHEMA = "rswm-trained-peft-release-ref-v1"
SCOPES = {"trained_model_candidate", "software_control"}


def _json(path):
    from .suite import _json as read
    return read(path)


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(x in "0123456789abcdef" for x in value)


def _path(path):
    value = Path(path)
    if value.is_symlink() or any(parent.is_symlink() for parent in value.parents):
        raise ValueError("Trained-adapter path symlink")
    return value.resolve()


def validate_release_binding(binding):
    fields = {"schema", "root", "tree_sha256", "manifest_sha256", "scope"}
    if (not isinstance(binding, dict) or set(binding) != fields or binding["schema"] != BINDING_SCHEMA
            or binding["scope"] not in SCOPES or not isinstance(binding["root"], str)
            or not binding["root"] or not Path(binding["root"]).is_absolute()
            or not _hash(binding["tree_sha256"]) or not _hash(binding["manifest_sha256"])):
        raise ValueError("Complete versioned trained-adapter reference required")
    return copy.deepcopy(binding)


def _identity(config, context, step, position):
    from .adapters.camera_conditioning import CameraConfig
    return {"config_sha256": digest(config), "dataset_sha256": digest({"index": context["index"], "clips": context["clips"]}),
            "source_audit_sha256": digest(context["source_audit"]),
            "camera_manifest_sha256": digest(CameraConfig(**config["camera"]).manifest()),
            "backend_manifest_sha256": digest(config["backend_manifest"]),
            "source_code_sha256": context["source_code_sha256"], "training_seed": config["seed"],
            "step": step, "clip_cursor": position}


def _contract(config, context):
    from .adapters.camera_conditioning import CameraConfig
    return digest({"config": config, "dataset": digest({"index": context["index"], "clips": context["clips"]}),
                   "source_audit": digest(context["source_audit"]), "camera": CameraConfig(**config["camera"]).manifest()})


def _validate_boundary(receipt, step, position, rows):
    audit = receipt["audit"]
    if (not isinstance(audit, list) or not audit or not isinstance(audit[-1], dict)
            or type(audit[-1].get("update")) is not int or audit[-1]["update"] != step
            or type(audit[-1].get("clip_cursor")) is not int or audit[-1]["clip_cursor"] != position):
        raise ValueError("Trained checkpoint requires a retained successful optimizer-boundary audit")
    for key in ("changed_parameters", "nonzero_gradients"):
        names = audit[-1].get(key)
        if not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
            raise ValueError("Trained checkpoint lacks finite-gradient/update audit identities")
    ordered = []
    for kind in ("optimizer_step_started", "optimizer_step_finished", "validation_call_started", "validation_call_finished"):
        selected = [i for i, row in enumerate(rows) if row.get("kind") == kind and row.get("update") == step]
        if len(selected) != 1 or (kind.endswith("finished") and rows[selected[0]].get("status") != "SUCCEEDED"):
            raise ValueError("Trained checkpoint lacks its completed optimizer/validation attempt")
        ordered.append(selected[0])
    if ordered != sorted(ordered):
        raise ValueError("Trained checkpoint optimizer/validation order differs")


def _checkpoint_preflight(checkpoint, config, context):
    contract = _contract(config, context)
    receipt = validate_checkpoint_receipt(checkpoint, contract)
    if receipt.get("schema") != "rswm-training-checkpoint-v2" or set(receipt.get("expert_state_sha256", {})) != {"low", "high"}:
        raise ValueError("Strong checkpoint source/seed/expert receipt required before model loading")
    row = receipt["training"]
    step, position = validate_resume_cursor({"step": row["step"], "position": row["clip_cursor"]}, config)
    if step <= 0 or row != _identity(config, context, step, position):
        raise ValueError("Updated checkpoint identity/config/camera/source/seed mismatch")
    from .replay import load_json
    rows = [load_json(line) for line in (Path(checkpoint).parent / receipt["attempt"]["journal_path"]).read_text().splitlines()]
    _validate_boundary(receipt, step, position, rows)
    return receipt, contract, row


def _formal_retained_data(context):
    """Validate retained fixed inventories; pixels remain the producer's gate."""
    from collections import Counter
    from .data import ROLES
    from .tasks import training_clips
    from .materialized import validate_audit_receipt
    index, clips, audit = (context[k] for k in ("index", "clips", "source_audit"))
    for value in (index, clips, audit):
        if not isinstance(value, dict) or value.get("synthetic") or value.get("software_control") or value.get("scope") == "software_control":
            raise ValueError("Synthetic/software provenance cannot authorize trained model candidates")
    world_rows = index.get("worlds")
    task_rows = index.get("tasks")
    if not isinstance(world_rows, list) or not isinstance(task_rows, list) or not isinstance(audit.get("worlds"), dict) or not isinstance(audit.get("tasks"), dict):
        raise ValueError("Complete retained world/task inventories required")
    worlds = {row["world_id"]: row for row in world_rows}
    tasks = {row["task_id"]: row for row in task_rows}
    if (len(worlds) != len(world_rows) or len(tasks) != len(task_rows) or set(worlds) != set(audit["worlds"])
            or set(tasks) != set(audit["tasks"]) or len(worlds) != 120):
        raise ValueError("Retained index/audit worlds or tasks omitted/extra/duplicated")
    if Counter((row["city"], row["role"]) for row in worlds.values()) != Counter({(city, role): count for city, (role, count) in ROLES.items()}):
        raise ValueError("Retained core120 city/role quotas differ")
    expected, expected_clips = [], {}
    for wid, world in worlds.items():
        audited = audit["worlds"][wid]
        for key in ("source_path", "source_sha256", "city", "role", "world_file_sha256", "world_pixel_sha256"):
            if world.get(key) != audited.get(key):
                raise ValueError("Retained world/source identity mismatch")
        for origin in range(4):
            for family in ("F2", "F3", "F5", "F7"):
                tid = f"{wid}-o{origin}-{family}"
                task = {"task_id": tid, "world_id": wid, "role": world["role"], "origin_index": origin,
                        "family": family, "protocol": "offline" if family in {"F2", "F3"} else "online16"}
                row = tasks.get(tid)
                frozen = {k: task[k] for k in ("task_id", "world_id", "role", "origin_index", "family", "protocol")}
                if row is None or any(row.get(k) != task[k] for k in ("task_id", "world_id", "role", "family")) or row.get("city") != world["city"] or row.get("task_path") != "tasks/" + tid + ".json":
                    raise ValueError("Retained index differs from frozen task inventory")
                expected.append({**frozen, **{k: world[k] for k in ("city", "world_file_sha256", "world_pixel_sha256")},
                                 "task_sha256": audit["tasks"][tid]["task_sha256"], "source_manifest_sha256": index.get("source_manifest_sha256")})
        if world["role"] == "train":
            for clip in training_clips(wid, "train"):
                expected_clips[(wid, clip["clip_id"])] = digest(clip)
    retained = clips.get("clips")
    if not isinstance(retained, list) or len(retained) != 3840 or clips.get("clip_count") != 3840 or clips.get("view_events") != 126720:
        raise ValueError("Retained complete train60/3840/126720 clips required")
    actual_clips = {(row["world_id"], row["clip_id"]): digest(row) for row in retained}
    if len(actual_clips) != len(retained) or actual_clips != expected_clips:
        raise ValueError("Retained training clip geometry/roles/unique inventory differs")
    validate_audit_receipt(audit, {"expected": expected, "source_code_sha256": context["source_code_sha256"]}, required_profile="core120")


def validate_adapter_release(binding, *, expected_weights_tree_sha256=None,
                             expected_camera_sha256=None, expected_training_seed=None,
                             allow_software=False):
    """Read actual retained files and provenance; this grants no model PASS."""
    binding = validate_release_binding(binding)
    if binding["scope"] == "software_control" and not allow_software:
        raise ValueError("Software-control adapters cannot load as the actual Wan model")
    root = _path(binding["root"])
    tree = locked_tree(root)
    if digest(tree) != binding["tree_sha256"] or tree.get("release.json") != binding["manifest_sha256"]:
        raise ValueError("Trained-adapter retained tree/manifest changed or incomplete")
    release = _json(root / "release.json")
    fields = {"schema", "scope", "files", "checkpoint", "training", "experts", "inference_config_sha256"}
    if not isinstance(release, dict) or set(release) != fields or release["schema"] != RELEASE_SCHEMA or release["scope"] != binding["scope"]:
        raise ValueError("Trained-adapter release format/scope mismatch")
    files = release["files"]
    if (not isinstance(files, dict) or not files or set(tree) != set(files) | {"release.json"}
            or any(not _hash(sha) or tree.get(name) != sha for name, sha in files.items())):
        raise ValueError("Missing/extra/changed trained-adapter release files")
    required = {"training/config.json", "training/receipt.json", "training/index.json", "training/clips.json",
                "training/source-audit.json", "training/source-tree.json", "training/weights-tree.json",
                "training/attempt.events.jsonl", "inference-config.json"}
    for tag in ("low", "high"):
        required |= {tag + "_noise/adapter_config.json", tag + "_noise/adapter_model.safetensors"}
    if set(files) != required:
        raise ValueError("Incomplete trained-adapter retained artifacts")
    config, receipt = _json(root / "training/config.json"), _json(root / "training/receipt.json")
    training_fields = {"config_sha256", "dataset_sha256", "source_audit_sha256", "camera_manifest_sha256", "backend_manifest_sha256", "source_code_sha256", "training_seed", "step", "clip_cursor"}
    receipt_fields = {"schema", "contract_sha256", "checkpoint_sha256", "audit", "training", "attempt", "expert_state_sha256"}
    if (not isinstance(release["training"], dict) or set(release["training"]) != training_fields
            or any(not _hash(release["training"][k]) for k in training_fields if k.endswith("sha256"))
            or any(type(release["training"][k]) is not int or release["training"][k] < 0 for k in ("training_seed", "step", "clip_cursor"))
            or not isinstance(receipt, dict) or set(receipt) != receipt_fields):
        raise ValueError("Invalid retained trained-release training/receipt fields")
    context = {"index": _json(root / "training/index.json"), "clips": _json(root / "training/clips.json"),
               "source_audit": _json(root / "training/source-audit.json"),
               "source_code_sha256": release["training"].get("source_code_sha256")}
    checkpoint = release["checkpoint"]
    if (not isinstance(checkpoint, dict) or set(checkpoint) != {"sha256", "receipt_sha256", "contract_sha256", "step", "position"}
            or type(checkpoint["step"]) is not int or checkpoint["step"] <= 0
            or not _hash(checkpoint["sha256"]) or not _hash(checkpoint["receipt_sha256"])):
        raise ValueError("Invalid trained checkpoint identity")
    step, position = validate_resume_cursor({"step": checkpoint["step"], "position": checkpoint["position"]}, config)
    identity = _identity(config, context, step, position)
    if (receipt.get("schema") != "rswm-training-checkpoint-v2" or receipt.get("training") != identity
            or release["training"] != identity or digest(receipt) != checkpoint["receipt_sha256"]
            or receipt.get("checkpoint_sha256") != checkpoint["sha256"]):
        raise ValueError("Trained release checkpoint/config/source/seed identity mismatch")
    contract = _contract(config, context)
    if receipt.get("contract_sha256") != contract or checkpoint["contract_sha256"] != contract:
        raise ValueError("Trained release contract mismatch")
    source_tree, weights_tree = _json(root / "training/source-tree.json"), _json(root / "training/weights-tree.json")
    parent = config["backend_manifest"]
    if digest(source_tree) != parent["source_tree_sha256"] or digest(weights_tree) != parent["weights_tree_sha256"]:
        raise ValueError("Trained release parent source/base identity mismatch")
    if expected_weights_tree_sha256 is not None and expected_weights_tree_sha256 != parent["weights_tree_sha256"]:
        raise ValueError("Trained adapter belongs to different base weights")
    if expected_camera_sha256 is not None and expected_camera_sha256 != identity["camera_manifest_sha256"]:
        raise ValueError("Trained adapter camera identity mismatch")
    if expected_training_seed is not None and (type(expected_training_seed) is not int or expected_training_seed != identity["training_seed"]):
        raise ValueError("Trained adapter training seed mismatch")
    from .replay import load_json
    from .runner import verify_journal
    raw_log = (root / "training/attempt.events.jsonl").read_bytes()
    rows = [load_json(line) for line in raw_log.decode("utf-8").splitlines()]
    attempt = receipt.get("attempt")
    if (not raw_log.endswith(b"\n") or not rows or not isinstance(attempt, dict)
            or set(attempt) != {"journal_path", "journal_sha256", "journal_head_sha256"}
            or hashlib.sha256(raw_log).hexdigest() != attempt.get("journal_sha256")
            or verify_journal(rows) != attempt.get("journal_head_sha256")
            or rows[0].get("contract_sha256") != contract
            or rows[0].get("training") != {k: v for k, v in identity.items() if k not in {"step", "clip_cursor"}}):
        raise ValueError("Trained release retained attempt identity mismatch")
    _validate_boundary(receipt, step, position, rows)
    if (not isinstance(release["experts"], dict) or set(release["experts"]) != {"low", "high"}
            or not isinstance(receipt.get("expert_state_sha256"), dict) or set(receipt["expert_state_sha256"]) != {"low", "high"}):
        raise ValueError("Both trained expert identities required")
    for tag, row in release["experts"].items():
        if (not isinstance(row, dict) or set(row) != {"path", "checkpoint_state_sha256", "frozen_state_sha256", "adapter_config_sha256", "adapter_state_sha256"}
                or row["path"] != tag + "_noise" or row["checkpoint_state_sha256"] != receipt["expert_state_sha256"][tag]
                or any(not _hash(row[k]) for k in row if k != "path")
                or row["adapter_config_sha256"] != files[tag + "_noise/adapter_config.json"]):
            raise ValueError("Trained expert mapping/content identity mismatch")
    if digest(_json(root / "inference-config.json")) != release["inference_config_sha256"]:
        raise ValueError("Trained release inference configuration changed")
    if binding["scope"] == "trained_model_candidate":
        _formal_retained_data(context)
    return release


def validate_adapter_tensor_content(binding, *, allow_software=False):
    """Check official PEFT retained tensors, after byte-level release validation."""
    release = validate_adapter_release(binding, allow_software=allow_software)
    from peft.utils.save_and_load import load_peft_weights
    for row in release["experts"].values():
        state = load_peft_weights(str(Path(binding["root"]) / row["path"]), device="cpu", local_files_only=True)
        if tensor_state_sha256(state) != row["adapter_state_sha256"]:
            raise ValueError("Trained adapter tensor content/expert mapping mismatch")
    return release


def validate_loaded_expert(expert, tag, binding, *, allow_software=False):
    """Verify the full unmerged low/high state before any inference merge."""
    release = validate_adapter_tensor_content(binding, allow_software=allow_software)
    if tag not in {"low", "high"}:
        raise ValueError("Explicit low/high loaded expert required")
    row = release["experts"][tag]
    if tensor_state_sha256(expert.state_dict()) != row["checkpoint_state_sha256"]:
        raise ValueError("Loaded expert differs from complete trained checkpoint: " + tag)
    configs = getattr(expert, "peft_config", None)
    if not isinstance(configs, dict) or set(configs) != {"default"}:
        raise ValueError("Loaded expert must preserve one unmerged default PEFT configuration")
    # Serving freezes the same adapter: inference_mode is the sole intentional
    # configuration difference from the unmerged training loader.
    actual = configs["default"].to_dict()
    actual["inference_mode"] = True
    expected = _json(Path(binding["root"]) / row["path"] / "adapter_config.json")
    normalized = json.loads(json.dumps(actual, default=lambda x: sorted(x) if isinstance(x, set) else x))
    if isinstance(expected.get("target_modules"), list):
        targets = expected["target_modules"]
        if any(not isinstance(name, str) for name in targets) or len(set(targets)) != len(targets):
            raise ValueError("Retained PEFT target_modules must be an exact unique set")
        expected["target_modules"] = sorted(targets)
    if normalized != expected:
        raise ValueError("Loaded actual PEFT configuration differs from retained training adapter")
    return release


def _disjoint(output, inputs):
    for value in inputs:
        if output == value or output.is_relative_to(value) or value.is_relative_to(output):
            raise ValueError("New deployment/release must be disjoint from immutable inputs")


def _pure_lora(expert, policy):
    from .legacy_training import _expected_lora_targets
    configs = getattr(expert, "peft_config", None)
    if not isinstance(configs, dict) or set(configs) != {"default"}:
        raise ValueError("Exactly one unmerged actual PEFT adapter required")
    config = configs["default"]
    if (getattr(config, "bias", "none") != "none" or getattr(config, "modules_to_save", None)
            or getattr(config, "use_dora", False) or getattr(config, "use_rslora", False)
            or getattr(config, "rank_pattern", None) or getattr(config, "alpha_pattern", None)
            or config.r != policy["rank"] or config.lora_alpha != policy["alpha"]
            or set(config.target_modules) != _expected_lora_targets(expert, policy["targets"])):
        raise ValueError("Only exact pure LoRA rank/alpha/targets policy can be exported")
    trainable = {name for name, value in expert.named_parameters() if value.requires_grad}
    if not trainable or any("lora_" not in name.lower() for name in trainable):
        raise ValueError("Non-LoRA trained parameters cannot be discarded by adapter export")
    # Official PEFT removes only its registered adapter-name component.
    # Check the actual export set using parameter identities, not blind key rewrites.
    from peft.utils.save_and_load import get_peft_model_state_dict
    actual = get_peft_model_state_dict(expert, adapter_name="default", save_embedding_layers=False)
    from peft.tuners.lora.layer import LoraLayer
    layers = [layer for layer in expert.modules() if isinstance(layer, LoraLayer)]
    if not layers or any(layer.r != {"default": policy["rank"]}
                         or layer.lora_alpha != {"default": policy["alpha"]}
                         or layer.scaling != {"default": policy["alpha"] / policy["rank"]} for layer in layers):
        raise ValueError("Actual LoRA layer scaling differs from the retained pure LoRA policy")
    parameter_pointers = {(p.untyped_storage().data_ptr(), p.storage_offset(), tuple(p.shape)) for name, p in expert.named_parameters() if name in trainable}
    exported_pointers = {(p.untyped_storage().data_ptr(), p.storage_offset(), tuple(p.shape)) for p in actual.values()}
    if parameter_pointers != exported_pointers or len(actual) != len(trainable):
        raise ValueError("PEFT export does not preserve every and only trained parameter")
    return actual, trainable


def _export_bound_backend(backend, *, checkpoint, config, context, inference_config, out, deployment_out):
    """Private real-PEFT software control; public CLI builds formal context."""
    import torch
    if (not isinstance(context, dict) or set(context) != {"scope", "index", "clips", "source_audit", "source_code_sha256"}
            or context["scope"] not in SCOPES or not _hash(context["source_code_sha256"])):
        raise ValueError("Exact trained export context required")
    from .legacy_wan import validate_backend_config, SCHEMA_V2
    inference_config = validate_backend_config(inference_config)
    if inference_config["schema"] != "rswm-legacy-wan-backend-v1" or inference_config["mode"] != "base" or inference_config["training"] is not None:
        raise ValueError("Explicit fresh base-shaped inference settings with training=null required")
    parent = config["backend_manifest"]
    source, weights, checkpoint = _path(parent["source_dir"]), _path(parent["weights_dir"]), _path(checkpoint)
    output, deployment = _path(out), _path(deployment_out)
    immutable = [source, weights, checkpoint.parent]
    _disjoint(output, immutable + [deployment]); _disjoint(deployment, immutable)
    if output.exists() or deployment.exists():
        raise FileExistsError("New exclusive trained release and deployment directories required")
    source_tree, weights_tree = locked_tree(source), locked_tree(weights)
    if any(name == "wan/remotesensebenchmark" or name.startswith("wan/remotesensebenchmark/") for name in source_tree):
        raise ValueError("Provider subtree cannot be redistributed in fresh inference source")
    if digest(source_tree) != parent["source_tree_sha256"] or digest(weights_tree) != parent["weights_tree_sha256"]:
        raise ValueError("Original training source/base weights changed")
    if getattr(backend, "objective", None) != "flow_matching":
        raise ValueError("Original flow-matching training backend required")
    receipt, contract, identity = _checkpoint_preflight(checkpoint, config, context)
    if context["scope"] == "trained_model_candidate":
        _formal_retained_data(context)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    step, position = validate_resume_cursor(state, config)
    if step <= 0:
        raise ValueError("Updated training checkpoint required")
    if identity != _identity(config, context, step, position):
        raise ValueError("Checkpoint tensor payload step/cursor differs from retained receipt")
    if (receipt.get("schema") != "rswm-training-checkpoint-v2" or receipt.get("training") != identity
            or set(receipt.get("expert_state_sha256", {})) != {"low", "high"} or state.get("contract_sha256") != contract):
        raise ValueError("Strong checkpoint receipt/config/camera/source/seed/expert identity required")
    backend.configure_trainable(config["trainable_policy"])
    if callable(getattr(backend, "quiesce_training", None)):
        backend.quiesce_training()
    if not isinstance(backend.model, torch.nn.ModuleDict) or set(backend.model) != {"low", "high"}:
        raise ValueError("Actual low/high ModuleDict checkpoint required")
    expected, saved = backend.model.state_dict(), state.get("model")
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise ValueError("Checkpoint model missing/extra keys")
    trainable = {name for name, value in backend.model.named_parameters() if value.requires_grad}
    frozen = {}
    for name, current in expected.items():
        value = saved[name]
        if not torch.is_tensor(value) or value.shape != current.shape or value.dtype != current.dtype or not torch.isfinite(value).all().item():
            raise ValueError("Checkpoint model shape/dtype/nonfinite tensor: " + name)
        if name not in trainable:
            actual_hash = parameter_content_hash(current)
            if parameter_content_hash(value) != actual_hash:
                raise ValueError("Checkpoint frozen base/buffer changed or experts swapped: " + name)
            frozen[name] = actual_hash
    expert_hashes = {tag: tensor_state_sha256({name: saved[tag + "." + name] for name in backend.model[tag].state_dict()}) for tag in ("low", "high")}
    if expert_hashes != receipt["expert_state_sha256"]:
        raise ValueError("Checkpoint low/high trained expert identities swapped/changed")
    for expert in backend.model.values():
        _pure_lora(expert, config["trainable_policy"])
    backend.model.load_state_dict(saved, strict=True)
    if callable(getattr(backend, "quiesce_training", None)):
        backend.quiesce_training()
    artifacts = {"training/config.json": canonical(config) + b"\n", "training/receipt.json": canonical(receipt) + b"\n",
                 "training/index.json": canonical(context["index"]) + b"\n", "training/clips.json": canonical(context["clips"]) + b"\n",
                 "training/source-audit.json": canonical(context["source_audit"]) + b"\n", "training/source-tree.json": canonical(source_tree) + b"\n",
                 "training/weights-tree.json": canonical(weights_tree) + b"\n", "inference-config.json": canonical(inference_config) + b"\n",
                 "training/attempt.events.jsonl": (checkpoint.parent / receipt["attempt"]["journal_path"]).read_bytes()}
    experts = {}
    from peft.utils.save_and_load import load_peft_weights
    with tempfile.TemporaryDirectory(prefix="rswm-peft-export-") as temporary:
        for tag, expert in backend.model.items():
            actual, _ = _pure_lora(expert, config["trainable_policy"])
            destination = Path(temporary) / tag
            expert.save_pretrained(destination, safe_serialization=True, save_embedding_layers=False)
            retained = load_peft_weights(str(destination), device="cpu", local_files_only=True)
            if set(retained) != set(actual) or any(not torch.equal(retained[name], value.detach().cpu()) for name, value in actual.items()):
                raise ValueError("Actual PEFT serialization did not preserve trained tensors")
            for name in ("adapter_config.json", "adapter_model.safetensors"):
                artifacts[tag + "_noise/" + name] = (destination / name).read_bytes()
            experts[tag] = {"path": tag + "_noise", "checkpoint_state_sha256": expert_hashes[tag],
                            "frozen_state_sha256": digest({name: sha for name, sha in frozen.items() if name.startswith(tag + ".")}),
                            "adapter_config_sha256": hashlib.sha256(artifacts[tag + "_noise/adapter_config.json"]).hexdigest(),
                            "adapter_state_sha256": tensor_state_sha256(retained)}
    if locked_tree(source) != source_tree or locked_tree(weights) != weights_tree or file_hash(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("Training source/base/checkpoint changed during export")
    release = {"schema": RELEASE_SCHEMA, "scope": context["scope"], "files": {name: hashlib.sha256(value).hexdigest() for name, value in artifacts.items()},
               "checkpoint": {"sha256": receipt["checkpoint_sha256"], "receipt_sha256": digest(receipt), "contract_sha256": contract, "step": step, "position": position},
               "training": identity, "experts": experts, "inference_config_sha256": digest(inference_config)}
    released_bytes = canonical(release) + b"\n"
    released_sha = hashlib.sha256(released_bytes).hexdigest()
    binding = {"schema": BINDING_SCHEMA, "root": str(output), "tree_sha256": digest({**release["files"], "release.json": released_sha}),
               "manifest_sha256": released_sha, "scope": context["scope"]}
    target_config = {**copy.deepcopy(inference_config), "schema": SCHEMA_V2, "mode": "restored_lora", "lora_subdir": None, "adapter_release": binding}
    validate_backend_config(target_config)
    output.mkdir(parents=True); deployment.mkdir(parents=True)
    for name, value in artifacts.items():
        path = output / name; path.parent.mkdir(parents=True, exist_ok=True); save_bytes_exclusive(path, value)
    for name in source_tree:
        if name == "backend_config.json":
            continue
        path = deployment / name; path.parent.mkdir(parents=True, exist_ok=True); save_bytes_exclusive(path, (source / name).read_bytes())
    save_bytes_exclusive(deployment / "backend_config.json", canonical(target_config) + b"\n")
    if any(file_hash(output / name) != sha for name, sha in release["files"].items()):
        raise ValueError("Trained-adapter publication readback mismatch")
    if any(file_hash(deployment / name) != sha for name, sha in source_tree.items() if name != "backend_config.json"):
        raise ValueError("Fresh inference source readback mismatch")
    # This last publication is the only completion marker. Partial directories
    # remain evidence of failure and are rejected by every v2 consumer.
    save_bytes_exclusive(output / "release.json", released_bytes)
    validate_adapter_release(binding, expected_weights_tree_sha256=parent["weights_tree_sha256"],
                             expected_camera_sha256=identity["camera_manifest_sha256"], expected_training_seed=config["seed"], allow_software=context["scope"] == "software_control")
    return {"status": "EXPORTED_TRAINED_PEFT_SOURCE_LINKED_NOT_MODEL_VALIDATED", "scope": context["scope"], "adapter_release": binding,
            "source_dir": str(deployment), "source_tree_sha256": digest(locked_tree(deployment)), "weights_dir": str(weights),
            "weights_tree_sha256": parent["weights_tree_sha256"], "entrypoint": parent["entrypoint"], "objective": "flow_matching",
            "training_seed": config["seed"], "camera": copy.deepcopy(config["camera"]), "training_camera_sha256": identity["camera_manifest_sha256"],
            "owner_approval": "REQUIRED_NOT_CREATED", "real_model_validation": "NOT_RUN", "publication_acceptance": "NOT_READY"}


def export_checkpoint(*, checkpoint, config, index, clips, world_root, source_audit, inference_config, out, deployment_out):
    """Formal producer: all data/model gates remain the existing ones."""
    from .suite import code_fingerprint
    config, index, clips, audit = _json(config), _json(index), _json(clips), _json(source_audit)
    dataset = ClipDataset(index, clips, world_root, source_audit=audit)
    validate_dev_index(config["validation_index"], dataset)
    from .legacy_wan import _source
    _, tree, runtime = _source(config["backend_manifest"]["source_dir"])
    if runtime["training"] is None or runtime["training"]["trainable_policy"] != config["trainable_policy"]:
        raise ValueError("Original training source must retain the exact nonnull trainable policy")
    if any(name.startswith("wan/remotesensebenchmark/") for name in tree):
        raise ValueError("Provider subtree cannot be redistributed in fresh inference source")
    context = {"scope": "trained_model_candidate", "index": index, "clips": clips, "source_audit": audit, "source_code_sha256": code_fingerprint()}
    _checkpoint_preflight(_path(checkpoint), config, context)
    backend = load_approved_backend(config["backend_manifest"], device=config["device"])
    backend.bind_training_context(dataset, config)
    return _export_bound_backend(backend, checkpoint=checkpoint, config=config,
        context=context,
        inference_config=_json(inference_config), out=out, deployment_out=deployment_out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "index", "clips", "world-root", "source-audit", "inference-config", "out", "deployment-out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(export_checkpoint(**vars(args)), sort_keys=True, allow_nan=False)); return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "BLOCKED_OR_INVALID", "error": str(exc)}, allow_nan=False)); return 2


if __name__ == "__main__":
    raise SystemExit(main())
