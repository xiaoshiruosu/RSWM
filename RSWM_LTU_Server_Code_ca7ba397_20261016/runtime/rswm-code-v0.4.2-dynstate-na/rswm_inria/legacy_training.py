"""Original Wan flow-matching loss, driven by the existing RSWM optimizer.

The objective/mask/expert boundary follow the retained original
``lora_remotesense/new/train.py``. Dataset selection, optimizer steps, cursor,
checkpoint and RNG restoration remain owned by ``training.train``. This module
does not load HDF5, construct an optimizer, or certify real weights/GPU results.
"""
from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import crop, digest, file_hash, rgb


def validate_training_spec(spec):
    """Require an explicit source-tree-locked policy; no legacy epoch defaults."""
    required = {"trainable_policy", "timestep_sampling", "expert_residency", "precision", "validation"}
    if not isinstance(spec, dict) or set(spec) != required:
        raise ValueError("Explicit locked Wan training/validation policy required")
    if spec["timestep_sampling"] != "uniform_integer" or spec["expert_residency"] not in {"resident", "block_cpu"}:
        raise ValueError("Wan training requires original uniform timesteps and explicit resident/block_cpu experts")
    if spec["precision"] != "bf16":
        raise ValueError("Actual Wan bridge training requires explicitly locked bf16 precision")
    policy = spec["trainable_policy"]
    if not isinstance(policy, dict) or set(policy) != {"mode", "rank", "alpha", "targets", "gradient_checkpointing"}:
        raise ValueError("Exact locked LoRA training policy required")
    if policy["mode"] not in {"new_lora", "restored_lora"}:
        raise ValueError("Training must distinguish new LoRA from restored LoRA")
    if type(policy["rank"]) is not int or policy["rank"] <= 0:
        raise ValueError("Positive LoRA rank required")
    if type(policy["alpha"]) not in {int, float} or not math.isfinite(policy["alpha"]) or policy["alpha"] <= 0:
        raise ValueError("Finite positive LoRA alpha required")
    if policy["targets"] not in {"all", "attn_cam", "attn_only"} or type(policy["gradient_checkpointing"]) is not bool:
        raise ValueError("Invalid original LoRA targets/checkpointing policy")
    if spec["expert_residency"] == "block_cpu" and not policy["gradient_checkpointing"]:
        raise ValueError("block_cpu training requires explicit gradient checkpointing")
    validation = spec["validation"]
    if not isinstance(validation, dict) or set(validation) != {"world_ids", "family", "origin_index", "seed", "max_total_forwards"}:
        raise ValueError("Explicit dev validation forward budget required")
    worlds = validation["world_ids"]
    if not isinstance(worlds, list) or not worlds or any(not isinstance(w, str) or not w for w in worlds) or len(set(worlds)) != len(worlds):
        raise ValueError("Distinct frozen dev validation worlds required")
    if validation["family"] != "F2" or type(validation["origin_index"]) is not int or not 0 <= validation["origin_index"] < 4:
        raise ValueError("Dev loss validation requires a frozen F2 origin")
    if type(validation["seed"]) is not int or validation["seed"] < 0 or type(validation["max_total_forwards"]) is not int or validation["max_total_forwards"] <= 0:
        raise ValueError("Finite explicit validation seed/forward budget required")
    return copy.deepcopy(spec)


def flow_matching_inputs(latents, num_train_timesteps, *, noise_dtype=None):
    """Original z_t=(1-t/N)z+(t/N)noise and velocity target noise-z."""
    import torch
    if type(num_train_timesteps) is not int or num_train_timesteps <= 0:
        raise ValueError("Invalid original timestep count")
    timesteps = torch.randint(0, num_train_timesteps, (len(latents),), device=latents.device).long()
    # Original VAE encode returns float32 latents but the trainer samples bf16
    # noise and forwards bf16 noisy latents. Preserve that actual objective.
    noise_dtype = torch.bfloat16 if noise_dtype is None else noise_dtype
    noise = torch.randn_like(latents, dtype=noise_dtype)
    fraction = timesteps.reshape(-1, 1, 1, 1, 1).float() / num_train_timesteps
    noisy = (1 - fraction.to(latents)) * latents + fraction.to(latents) * noise
    return noisy.to(noise_dtype), (noise - latents).float(), timesteps


def _expected_lora_targets(expert, targets):
    import torch
    base = expert.get_base_model() if hasattr(expert, "get_base_model") else expert
    result = set()
    for name, module in base.named_modules():
        # PEFT wraps the original Linear in a module with base_layer. Do not
        # mistake its child adapter Linears for original backbone targets.
        if ".base_layer" in name or ".lora_" in name:
            continue
        linear = getattr(module, "base_layer", module)
        if "blocks" not in name or "face" in name or "modulation" in name or not isinstance(linear, torch.nn.Linear):
            continue
        attn = "self_attn" in name or "cross_attn" in name
        cam = any(x in name for x in ("cam_injector", "cam_scale", "cam_shift"))
        if targets == "all" or attn or targets == "attn_cam" and cam:
            result.add(name)
    if not result:
        raise ValueError("Original LoRA target filter matched no backbone modules")
    return result


class LegacyTrainingMixin:
    """Training half of the same loaded legacy backend and shared camera path."""

    def _training_spec(self):
        spec = validate_training_spec(self.runtime_config.get("training"))
        offload = spec["expert_residency"] == "block_cpu"
        sampling = self.runtime_config["sampling"]
        if offload and not (sampling.get("block_cpu_offload", False) and sampling["inactive_model_cpu"] and sampling["t5_cpu"]):
            raise ValueError("block_cpu training requires locked block, inactive-expert and T5 CPU settings")
        if not offload and (sampling["inactive_model_cpu"] or sampling.get("block_cpu_offload", False)):
            raise ValueError("Training expert offload is unsupported by the single driver optimizer")
        return spec

    def quiesce_training(self):
        for controller in getattr(self, "_offload_controllers", {}).values():
            controller.quiesce()

    def training_memory_observation(self):
        """Actual process peaks, including allocations preceding cleanup."""
        import sys
        import torch
        result = {"scope": "process lifetime peak; not a per-update reset or formal score",
                  "offload": {tag: controller.report() for tag, controller in
                              getattr(self, "_offload_controllers", {}).items()}}
        try:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["host_peak_rss_bytes"] = int(rss if sys.platform == "darwin" else rss * 1024)
        except ImportError:
            result["host_peak_rss_bytes"] = None
        device = self.training_components()["device"]
        if device.type == "cuda":
            result.update(gpu_current_allocated_bytes=torch.cuda.memory_allocated(device),
                          gpu_current_reserved_bytes=torch.cuda.memory_reserved(device),
                          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        else:
            result.update(gpu_current_allocated_bytes=None, gpu_current_reserved_bytes=None,
                          gpu_peak_allocated_bytes=None, gpu_peak_reserved_bytes=None)
        return result

    def bind_training_context(self, dataset, config):
        if callable(getattr(self, "_check_source", None)):
            self._check_source()
        spec = self._training_spec()
        validation = spec["validation"]
        rows = config["validation_index"]
        if [r["world_id"] for r in rows] != validation["world_ids"]:
            raise ValueError("Driver dev worlds differ from locked Wan validation population")
        if validation["max_total_forwards"] != config["updates"] * len(rows):
            raise ValueError("Locked dev forward budget does not match approved optimizer updates")
        if config["trainable_policy"] != spec["trainable_policy"]:
            raise ValueError("Driver trainable policy differs from locked Wan policy")
        if config["precision"] != spec["precision"]:
            raise ValueError("Driver precision differs from supported locked Wan precision")
        from .adapters.camera_conditioning import CameraConfig
        self._training_camera_sha256 = digest(CameraConfig(**config["camera"]).manifest())
        self._training_world_root = Path(dataset.root)
        self._training_dev_rows = copy.deepcopy(rows)
        self._validation_forwards = getattr(self, "_validation_forwards", 0)

    def checkpoint_training_state(self):
        return {"validation_forwards": self._validation_forwards}

    def checkpoint_expert_state_sha256(self):
        from .training import tensor_state_sha256
        self.quiesce_training()
        return {tag:tensor_state_sha256(expert.state_dict()) for tag,expert in self.training_components()["experts"].items()}

    def bind_validation_journal(self, journal, *, consumed_forwards):
        if type(consumed_forwards) is not int or consumed_forwards < self._validation_forwards:
            raise ValueError("Resume validation budget lacks retained training attempts")
        if consumed_forwards > self._training_spec()["validation"]["max_total_forwards"]:
            raise ValueError("Retained validation attempts exceed the approved budget")
        self._validation_forwards = consumed_forwards
        self._validation_journal = journal

    def check_remaining_validation_budget(self, *, updates_remaining):
        policy = self._training_spec()["validation"]
        if self._validation_forwards + updates_remaining*len(policy["world_ids"]) > policy["max_total_forwards"]:
            raise ValueError("Approved validation budget cannot cover remaining updates and retained failures")

    def load_checkpoint_training_state(self, state, *, step):
        policy = self._training_spec()["validation"]
        if type(step) is not int or step < 0:
            raise ValueError("Wan resume optimizer boundary invalid")
        minimum = step * len(policy["world_ids"])
        if not isinstance(state, dict) or set(state) != {"validation_forwards"} or type(state["validation_forwards"]) is not int or not minimum <= state["validation_forwards"] <= policy["max_total_forwards"]:
            raise ValueError("Wan resume lacks exact retained validation budget state")
        # Same-lifecycle failed forwards never disappear on restoring an older
        # checkpoint. A new process requires the actual retained checkpoint.
        self._validation_forwards = max(self._validation_forwards, state["validation_forwards"])

    def configure_trainable(self, policy):
        import torch
        if callable(getattr(self, "_check_source", None)):
            self._check_source()
        spec = self._training_spec()
        if policy != spec["trainable_policy"]:
            raise ValueError("Trainable policy differs from locked Wan configuration")
        if getattr(self, "_lora_training_attempted", False):
            raise ValueError("LoRA training was already attempted; reload after failure rather than stack adapters")
        self._lora_training_attempted = True
        parts = self.training_components()
        experts = parts["experts"]
        for controller in getattr(self, "_offload_controllers", {}).values():
            controller.close()
        self._offload_controllers = {}
        mode = self.runtime_config["mode"]
        if policy["mode"] == "new_lora" and mode != "base":
            raise ValueError("New LoRA requires explicit base initialization")
        if policy["mode"] == "restored_lora" and mode != "restored_lora":
            raise ValueError("Restored LoRA requires actual retained adapter initialization")
        configured = {}
        for tag, expert in experts.items():
            expert.requires_grad_(False)
            if policy["mode"] == "new_lora":
                if getattr(expert, "peft_config", None):
                    raise ValueError("New LoRA cannot overwrite/stack a retained adapter")
                expert = self.make_lora_expert(model=expert, rank=policy["rank"], alpha=policy["alpha"], targets=policy["targets"])
            configs = getattr(expert, "peft_config", None)
            if not isinstance(configs, dict) or len(configs) != 1:
                raise ValueError("Actual unmerged single PEFT adapter required for training")
            adapter_config = next(iter(configs.values()))
            if adapter_config.r != policy["rank"] or adapter_config.lora_alpha != policy["alpha"] or set(adapter_config.target_modules) != _expected_lora_targets(expert, policy["targets"]):
                raise ValueError("Actual LoRA rank/alpha/targets differ from locked policy")
            count = 0
            for name, parameter in expert.named_parameters():
                parameter.requires_grad_("lora_" in name.lower())
                count += parameter.numel() if parameter.requires_grad else 0
            if not count:
                raise ValueError("Restored adapter was merged or has no trainable LoRA parameters")
            if policy["gradient_checkpointing"]:
                expert.gradient_checkpointing_enable()
            elif hasattr(expert, "gradient_checkpointing_disable"):
                expert.gradient_checkpointing_disable()
            if spec["expert_residency"] == "block_cpu":
                from .legacy_offload import attach_frozen_cpu_offload
                for parameter in expert.parameters():
                    if parameter.requires_grad:
                        parameter.data = parameter.data.to(device=parts["device"], dtype=parts["dtype"])
                self._offload_controllers[tag] = attach_frozen_cpu_offload(expert, device=parts["device"])
            else:
                if getattr(self, "actual_weights_loaded", False):
                    # Preserve loader-selected FP32 time/head/norm modules.
                    expert.to(device=parts["device"])
                    for parameter in expert.parameters():
                        if parameter.requires_grad:
                            parameter.data = parameter.data.to(dtype=parts["dtype"])
                else:
                    expert.to(device=parts["device"], dtype=parts["dtype"])
            configured[tag] = expert
        self.replace_training_experts(configured)
        # VAE/T5 are frozen and excluded from the optimizer holder. Their
        # requires_grad contract is checked on every actual encode as well.
        for component in (parts["vae"].model, parts["text_encoder"].model):
            component.eval().requires_grad_(False)
        if any(p.requires_grad and "lora_" not in n.lower() for n, p in self.model.named_parameters()):
            raise ValueError("Non-LoRA backbone parameter unexpectedly trainable")
        self._lora_training_configured = True
        self._loss_active_groups = {}

    def active_trainable_groups(self):
        return copy.deepcopy(self._loss_active_groups)

    def _encode_video(self, videos, parts):
        import torch
        vae = parts["vae"]
        if any(p.requires_grad for p in vae.model.parameters()):
            raise ValueError("Frozen VAE parameters became trainable")
        original_device = next(vae.model.parameters()).device
        # Explicit component placement; expert/optimizer parameters never swap.
        vae.model.to(parts["device"])
        for name in ("mean", "std"):
            if hasattr(vae, name):
                setattr(vae, name, getattr(vae, name).to(parts["device"]))
        if hasattr(vae, "mean") and hasattr(vae, "std"):
            vae.scale = [vae.mean, 1.0 / vae.std]
        try:
            with torch.no_grad():
                return vae.encode(videos)
        finally:
            vae.model.to(original_device)
            for name in ("mean", "std"):
                if hasattr(vae, name):
                    setattr(vae, name, getattr(vae, name).to(original_device))
            if hasattr(vae, "mean") and hasattr(vae, "std"):
                vae.scale = [vae.mean, 1.0 / vae.std]

    def loss(self, *, images, cameras, update):
        import torch
        import torch.nn.functional as functional
        from .legacy_camera import camera_condition
        if callable(getattr(self, "_check_source", None)):
            self._check_source()
        if not getattr(self, "_lora_training_configured", False) or not hasattr(self, "_training_camera_sha256"):
            raise ValueError("Wan loss requires bound audited training and configured LoRA")
        if images.ndim != 5 or tuple(images.shape[1:]) != (33, 3, 480, 832) or not len(images):
            raise ValueError("Actual Wan training requires complete 33-frame native 480x832 RGB clips")
        if not torch.isfinite(images).all().item() or images.min().item() < -1 or images.max().item() > 1:
            raise ValueError("Training RGB must be finite in [-1,1]")
        if type(update) is not int or update < 0 or len(cameras) != len(images):
            raise ValueError("Training batch/update mismatch")
        parts = self.training_components(); cfg = parts["wan_config"]
        if tuple(cfg.vae_stride) != (4, 8, 8) or tuple(cfg.patch_size) != (1, 2, 2) or cfg.num_train_timesteps != 1000 or cfg.boundary != .947:
            raise ValueError("Original Wan VAE/patch/timestep expert mapping changed")
        if images.device != parts["device"]:
            raise ValueError("Training tensors and loaded Wan device differ")
        conditions = []
        for camera in cameras:
            if camera.get("camera_manifest_sha256") != self._training_camera_sha256:
                raise ValueError("Training and inference camera identity differ")
            conditions.append(camera_condition(camera, 33, cfg, 480, 832, 60, 104, parts["device"]))
        latents = torch.stack(self._encode_video([x.permute(1, 0, 2, 3).to(parts["dtype"]) for x in images], parts))
        if tuple(latents.shape[1:]) != (16, 9, 60, 104) or not torch.isfinite(latents).all().item():
            raise ValueError("Actual VAE latent temporal/spatial mapping differs")
        # Original 4-channel I2V mask: repeat the first RGB-frame mask four
        # times, then group time by four. No full-video target leaks into y.
        mask = torch.zeros(1, 36, 60, 104, device=parts["device"], dtype=latents.dtype)
        mask[:, :4] = 1
        mask = mask.view(1, 9, 4, 60, 104).transpose(1, 2)[0]
        y_list = []
        for image in images[:, 0]:
            initial_video = torch.cat([image.to(parts["dtype"]).unsqueeze(1), torch.zeros(3, 32, 480, 832, device=parts["device"], dtype=parts["dtype"])], dim=1)
            encoded = self._encode_video([initial_video], parts)[0]
            if tuple(encoded.shape) != (16, 9, 60, 104):
                raise ValueError("I2V first-frame VAE condition mapping differs")
            y_list.append(torch.cat([mask.to(encoded), encoded], dim=0))
        text_encoder = parts["text_encoder"]
        if any(p.requires_grad for p in text_encoder.model.parameters()):
            raise ValueError("Frozen text encoder parameters became trainable")
        original_device = next(text_encoder.model.parameters()).device
        text_device = torch.device("cpu") if self._training_spec()["expert_residency"] == "block_cpu" else parts["device"]
        text_encoder.model.to(text_device)
        try:
            with torch.no_grad():
                context = text_encoder([self.runtime_config["sampling"]["prompt"]] * len(images), text_device)
                context = [value.to(parts["device"]) for value in context]
        finally:
            text_encoder.model.to(original_device)
        noisy, target, timesteps = flow_matching_inputs(latents, cfg.num_train_timesteps, noise_dtype=parts["dtype"])
        predictions = []; active = set()
        for index in range(len(images)):
            tag = "low" if timesteps[index].item() < int(cfg.boundary * cfg.num_train_timesteps) else "high"
            active.add(tag)
            # Original Wan camera preparation concatenates sample sequences
            # along dim1. Forward each sample separately to prevent B>1 camera
            # conditions from crossing sample boundaries.
            from contextlib import nullcontext
            controller = getattr(self, "_offload_controllers", {}).get(tag)
            with controller.forward_context() if controller is not None else nullcontext():
                prediction = parts["experts"][tag](x=[noisy[index]], t=timesteps[index:index+1], context=[context[index]], seq_len=9*60*104//4, y=[y_list[index]], dit_cond_dict=conditions[index])
            if not isinstance(prediction, (list, tuple)) or len(prediction) != 1 or prediction[0].shape != target[index].shape:
                raise ValueError("Actual Wan expert prediction shape mismatch")
            predictions.append(prediction[0])
        names = dict(self.model.named_parameters())
        groups = {}
        for tag in active:
            ids = {id(p) for p in parts["experts"][tag].parameters() if p.requires_grad}
            group = [name for name, p in names.items() if id(p) in ids]
            if not group:
                raise ValueError("Active Wan expert has no audited trainable parameters")
            groups[tag] = group
        self._loss_active_groups = groups
        return functional.mse_loss(torch.stack(predictions).float(), target, reduction="mean")

    def validate(self, validation_index, *, camera_manifest):
        import torch
        from .adapters.camera_conditioning import CameraConfig, backend_payload, build_camera_batch, to_native_rgb
        from .tasks import packages
        spec = self._training_spec(); policy = spec["validation"]
        if not hasattr(self, "_training_world_root") or validation_index != self._training_dev_rows:
            raise ValueError("Actual dev validation context not bound")
        if digest(camera_manifest) != self._training_camera_sha256:
            raise ValueError("Dev/training camera identity mismatch")
        if self._validation_forwards + len(validation_index) > policy["max_total_forwards"]:
            raise ValueError("Approved dev validation forward budget exhausted")
        if getattr(self, "actual_weights_loaded", False) and not hasattr(self, "_validation_journal"):
            raise ValueError("Actual dev validation requires retained driver attempt costs")
        cam = CameraConfig(**{k: camera_manifest[k] for k in ("translation_scaling", "framewise", "camera_channel", "temporal_stride")})
        parts = self.training_components(); device = parts["device"]
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        groups_before = copy.deepcopy(self._loss_active_groups); records = []
        try:
            # Loss validation uses a fixed seed and restores the training RNG.
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                torch.random.default_generator.manual_seed(policy["seed"])
                if device.type == "cuda":
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(policy["seed"])
                for row in validation_index:
                    relative = Path(row["path"]); path = self._training_world_root / relative
                    if row.get("role") != "dev" or row.get("city") != "kitsap" or relative.is_absolute() or ".." in relative.parts or path.is_symlink() or any(p.is_symlink() for p in path.parents):
                        raise ValueError("Dev validation path/role violation")
                    if file_hash(path) != row["world_file_sha256"]:
                        raise ValueError("Dev validation retained PNG hash mismatch")
                    with Image.open(path) as image:
                        if image.format != "PNG" or image.mode != "RGB":
                            raise ValueError("Dev validation requires lossless RGB PNG")
                        world = rgb(np.asarray(image), (2048, 2048, 3))
                    if hashlib.sha256(world.tobytes()).hexdigest() != row["world_pixel_sha256"]:
                        raise ValueError("Dev validation pixel identity mismatch")
                    task = next(task for task, _ in packages(row["world_id"], "dev", policy["origin_index"]) if task["family"] == "F2")
                    q = np.asarray(task["programs"]["main"], dtype=np.int64)
                    frames = np.stack([to_native_rgb(crop(world, point, task["origin"])) for point in q])
                    images = torch.from_numpy(frames).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2).unsqueeze(0) / 127.5 - 1
                    cameras = [backend_payload(build_camera_batch(q[1:], q[0], cam), expected_config_sha256=self._training_camera_sha256)]
                    # An attempted forward consumes its frozen budget even if
                    # VAE/model execution later fails; no retry counter rollback.
                    forward = self._validation_forwards+1
                    journal = getattr(self, "_validation_journal", None)
                    if journal is not None:
                        journal.append("validation_forward_started", forward=forward, world_id=row["world_id"],
                                       task_sha256=digest(task), world_file_sha256=row["world_file_sha256"],
                                       world_pixel_sha256=row["world_pixel_sha256"], camera_manifest_sha256=self._training_camera_sha256,
                                       validation_seed=policy["seed"])
                    self._validation_forwards = forward
                    # The driver's validation call is outside its training
                    # autocast context. Actual bf16 experts need the same
                    # source-locked precision here, including time embedding.
                    started = time.perf_counter(); error = None
                    try:
                        with torch.autocast(device_type=device.type, dtype=parts["dtype"], enabled=parts["dtype"] != torch.float32):
                            loss = self.loss(images=images, cameras=cameras, update=0)
                        if loss.ndim != 0 or not torch.isfinite(loss).item():
                            raise ValueError("Actual dev forward loss is not finite scalar")
                    except BaseException as exc:
                        error = type(exc).__name__
                        raise
                    finally:
                        if journal is not None:
                            journal.append("validation_forward_finished", forward=forward,
                                           status="FAILED" if error else "SUCCEEDED", error_type=error,
                                           wall_seconds=time.perf_counter()-started)
                    records.append({"world_id": row["world_id"], "task_id": task["task_id"], "task_sha256": digest(task), "world_file_sha256": row["world_file_sha256"], "world_pixel_sha256": row["world_pixel_sha256"], "frames": 33, "loss": loss.item()})
        finally:
            self._loss_active_groups = groups_before
        return {"status": "MEASURED_DEV_LOSS_NO_RELEASE_ACCEPTANCE", "objective": "flow_matching_velocity_MSE", "camera_manifest_sha256": self._training_camera_sha256, "validation_seed": policy["seed"], "forward_count": len(records), "loss": sum(r["loss"] for r in records)/len(records), "worlds": records, "publication_acceptance": "NOT_READY"}
