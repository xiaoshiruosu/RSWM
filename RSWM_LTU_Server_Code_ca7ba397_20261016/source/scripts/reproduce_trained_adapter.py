"""Reproducible actual tiny PEFT CPU controls, never real Wan validation.

This script uses real PEFT objects, adapter bytes and fresh processes. Its
synthetic input cannot pass the production train60 gate or certify a model.
"""
from __future__ import annotations
import copy
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
import peft
from PIL import Image
from rswm_inria.geometry import digest, file_hash
from rswm_inria.legacy_training import LegacyTrainingMixin, _expected_lora_targets
from rswm_inria.legacy_wan import strict_lora_from_pretrained
from rswm_inria.adapters.camera_conditioning import CameraConfig

UNMERGED_TOLERANCE = {"atol": 1e-6, "rtol": 1e-6}
MERGED_TOLERANCE = {"atol": 1e-5, "rtol": 1e-5}
POLICY = {"mode": "new_lora", "rank": 2, "alpha": 2,
          "targets": "attn_cam", "gradient_checkpointing": False}

class TinyAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(8, 8)
        self.v = torch.nn.Linear(8, 8)

    def forward(self, x):
        return torch.tanh(self.q(x)) + self.v(x)


class TinyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.cam_injector_layer1 = torch.nn.Linear(8, 8)
        self.ffn = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.ffn(torch.tanh(self.self_attn(x) + self.cam_injector_layer1(x)))


class TinyBackbone(torch.nn.Module):
    """Small deterministic input; its dimensions do not represent A14B."""
    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.blocks = torch.nn.ModuleList([TinyBlock()])
        self.config = {"software_control": True, "width": 8}

    def forward(self, x):
        return self.blocks[0](x)


def runtime_config(policy=None):
    policy = copy.deepcopy(POLICY if policy is None else policy)
    restored = policy["mode"] == "restored_lora"
    return {"schema": "rswm-legacy-wan-backend-v1", "mode": "restored_lora" if restored else "base", "config_key": "i2v-A14B",
            "sampling": {"prompt": "CPU software control only", "negative_prompt": "",
                         "steps": 2, "shift": 5, "guide_scale": [3, 3], "boundary": .875,
                         "param_dtype": "bf16", "t5_dtype": "bf16", "t5_cpu": True,
                         "inactive_model_cpu": False}, "lora_subdir": "initial-adapters" if restored else None,
            "training": {"trainable_policy": policy, "timestep_sampling": "uniform_integer",
                         "expert_residency": "resident", "precision": "bf16",
                         "validation": {"world_ids": ["software-dev"], "family": "F2",
                                        "origin_index": 0, "seed": 9, "max_total_forwards": 2}}}


class TinyBackend(LegacyTrainingMixin):
    """Shared real PEFT policy/configuration path, explicitly software-only."""
    objective = "flow_matching"
    def __init__(self, weights_dir=None, source_dir=None, policy=None):
        self.runtime_config = runtime_config(policy)
        self.model = torch.nn.ModuleDict({"low": TinyBackbone(11), "high": TinyBackbone(29)})
        self.weights_dir = Path(weights_dir).resolve() if weights_dir is not None else None
        self.source_dir = Path(source_dir).resolve() if source_dir is not None else None
        if self.weights_dir is not None:
            from safetensors.torch import load_file
            for tag, model in self.model.items():
                model.load_state_dict(load_file(self.weights_dir/tag/"model.safetensors"), strict=True)
            if self.runtime_config["mode"] == "restored_lora":
                self.model = torch.nn.ModuleDict({tag: strict_lora_from_pretrained(peft.PeftModel, model,
                           self.weights_dir/"initial-adapters"/(tag+"_noise"), is_trainable=True)
                           for tag, model in self.model.items()})
        self.vae = SimpleNamespace(model=torch.nn.Linear(1, 1))
        self.text_encoder = SimpleNamespace(model=torch.nn.Linear(1, 1))
        self._validation_forwards = 0

    def training_components(self):
        return {"experts": dict(self.model.items()), "device": torch.device("cpu"),
                "dtype": torch.float32, "vae": self.vae, "text_encoder": self.text_encoder}

    def make_lora_expert(self, *, model, rank, alpha, targets):
        return peft.get_peft_model(model, peft.LoraConfig(r=rank, lora_alpha=alpha,
                                    target_modules=sorted(_expected_lora_targets(model, targets)),
                                    lora_dropout=0, bias="none"))

    def replace_training_experts(self, configured):
        self.model = torch.nn.ModuleDict(configured)

    def bind_training_context(self, dataset, config):
        if not isinstance(dataset, TinyDataset) or config["precision"] != "fp32":
            raise ValueError("This CPU engine fixture accepts only its explicit software dataset")
        self._training_camera_sha256 = digest(CameraConfig(**config["camera"]).manifest())

    def loss(self, *, images, cameras, update):
        if tuple(images.shape[1:]) != (33, 3, 480, 832) or len(cameras) != len(images):
            raise ValueError("Software driver fixture still requires the native frame/camera contract")
        x = fixed_input() + images[:, 0].mean().float() * .05
        self._loss_active_groups = {tag: [n for n, p in self.model.named_parameters()
                                                if n.startswith(tag+".") and p.requires_grad]
                                    for tag in ("low", "high")}
        return sum((self.model[tag](x) - (x*.3 + offset)).square().mean()
                   for tag, offset in (("low", .2), ("high", -.4)))

    def validate(self, rows, *, camera_manifest):
        if any(row.get("role") != "diagnostic_only" for row in rows):
            raise ValueError("The software fixture cannot evaluate real dev worlds")
        self._validation_forwards += 1
        started = time.perf_counter()
        self._validation_journal.append("validation_forward_started", forward=self._validation_forwards,
                                        world_id="software-dev", scope="software_control")
        outputs = {tag: model(fixed_input()).detach().tolist() for tag, model in self.model.items()}
        self._validation_journal.append("validation_forward_finished", forward=self._validation_forwards,
                                       status="SUCCEEDED", wall_seconds=time.perf_counter()-started,
                                       scope="software_control")
        return {"status": "SOFTWARE_CONTROL_ONLY", "outputs": outputs,
                "camera_manifest_sha256": digest(camera_manifest)}


def fixed_input():
    return torch.linspace(-.7, .9, 32, dtype=torch.float32).reshape(4, 8)


def trained_tiny_backend():
    torch.manual_seed(83)
    backend = TinyBackend()
    backend.configure_trainable(POLICY)
    frozen = {n: p.detach().clone() for n, p in backend.model.named_parameters() if not p.requires_grad}
    initial = {n: p.detach().clone() for n, p in backend.model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW((p for p in backend.model.parameters() if p.requires_grad), lr=.03)
    audit = []
    x = fixed_input()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = sum((backend.model[tag](x) - (x*.3 + offset)).square().mean()
                   for tag, offset in (("low", .2), ("high", -.4)))
        assert torch.isfinite(loss).item()
        loss.backward()
        nonzero = []
        for name, parameter in backend.model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
                if parameter.grad.count_nonzero().item():
                    nonzero.append(name)
        assert all(any(n.startswith(tag+".") for n in nonzero) for tag in ("low", "high"))
        optimizer.step()
        audit.append({"update": step+1, "nonzero_gradients": sorted(nonzero), "loss": loss.item()})
    for name, parameter in backend.model.named_parameters():
        if parameter.requires_grad:
            assert torch.isfinite(parameter).all().item() and not torch.equal(parameter, initial[name])
        else:
            assert torch.equal(parameter, frozen[name])
    return backend, optimizer, audit


class TinyDataset:
    """Engine input only, deliberately never constructed through ClipDataset.

    ``diagnostic`` is the engine guard field, not an audit classification. All
    retained identities explicitly say synthetic/diagnostic_only. Production
    ClipDataset and the public training CLI remain unchanged and reject these
    inputs. The dedicated checker below runs only inside a try/finally scope.
    """
    diagnostic = False

    def __init__(self):
        self.index = {"schema": "tiny-peft-software-index-v1", "synthetic": True,
                      "scope": "software_control", "role": "diagnostic_only", "clips": 2}
        self.clips = {"schema": "tiny-peft-software-clips-v1", "synthetic": True,
                      "scope": "software_control", "clips": [{"id": 0}, {"id": 1}]}
        self.source_audit = {"schema": "tiny-peft-software-audit-v1", "synthetic": True,
                             "scope": "software_control", "worlds": {"software-dev": {
                                 "role": "diagnostic_only", "city": "synthetic", "synthetic": True}}}
        self.index_sha256 = digest({"index": self.index, "clips": self.clips})
        self.source_audit_sha256 = digest(self.source_audit)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        frame = np.empty((240, 416, 3), dtype=np.uint8)
        frame[:] = [64 + index*16, 128, 192 - index*16]
        q = np.column_stack((np.arange(33, dtype=np.int64), np.zeros(33, dtype=np.int64)))
        return {"images": np.repeat(frame[None], 33, axis=0), "q": q}


def software_dev_checker(rows, dataset):
    if not isinstance(dataset, TinyDataset) or rows != [{"world_id": "software-dev", "role": "diagnostic_only",
                                                        "city": "synthetic", "synthetic": True}]:
        raise ValueError("Software driver boundary requires exact synthetic diagnostic identity")


def train_with_existing_driver(config, out):
    """Execute the current driver body with actual PEFT, then restore its gate."""
    from rswm_inria import training
    dataset = TinyDataset()
    backend = TinyBackend(config["backend_manifest"]["weights_dir"], config["backend_manifest"]["source_dir"], config["trainable_policy"])
    original = training.validate_dev_index
    training.validate_dev_index = software_dev_checker
    try:
        result = training.train(dataset, backend, config, out)
    finally:
        training.validate_dev_index = original
    if training.validate_dev_index is not original:
        raise RuntimeError("Production validation gate was not restored")
    return dataset, backend, result


def create_driver_fixture(work, mode="new_lora"):
    """Retain actual tiny base bytes, source and current-driver checkpoint."""
    from safetensors.torch import save_file
    from rswm_inria.training import locked_tree
    from rswm_inria.suite import code_fingerprint
    work = Path(work)
    work.mkdir()
    source, weights = work/"software-source", work/"tiny-base-weights"
    source.mkdir(); weights.mkdir()
    # Self-contained source deployment; no tests, old source ZIP or real weights.
    (source/"software_model.py").write_bytes(Path(__file__).read_bytes())
    (source/"backend.py").write_text("from pathlib import Path\nfrom software_model import load_software_backend\ndef load_backend(*, weights_dir, device):\n    return load_software_backend(Path(__file__).resolve().parent, weights_dir, device)\n")
    policy = dict(POLICY, mode=mode)
    (source/"backend_config.json").write_text(json.dumps(runtime_config(policy), sort_keys=True)+"\n")
    (source/"LICENSE.txt").write_text("Original RSWM tiny PEFT software control, no pretrained weights; internal software execution permitted.\n")
    base = TinyBackend()
    for tag, model in base.model.items():
        (weights/tag).mkdir()
        save_file({n: value.detach().contiguous() for n, value in model.state_dict().items()}, weights/tag/"model.safetensors")
        (weights/tag/"config.json").write_text(json.dumps({"software_control": True, "width": 8, "seed": 11 if tag == "low" else 29})+"\n")
    if mode == "restored_lora":
        initial, _, _ = trained_tiny_backend()
        for tag, model in initial.model.items():
            model.save_pretrained(weights/"initial-adapters"/(tag+"_noise"), safe_serialization=True, save_embedding_layers=False)
    elif mode != "new_lora":
        raise ValueError("Explicit new_lora/restored_lora software mode required")
    manifest = {"source_dir": str(source.resolve()), "source_tree_sha256": digest(locked_tree(source)),
                "entrypoint": "backend.py", "weights_dir": str(weights.resolve()),
                "weights_tree_sha256": digest(locked_tree(weights)), "objective": "flow_matching", "approved": True,
                "license_record": "Permission applies only to this original tiny synthetic CPU software fixture; no approval of real Wan/model weights."}
    config = {"updates": 2, "batch_size": 1, "accumulation": 1, "lr": .03, "seed": 83,
              "precision": "fp32", "device": "cpu", "checkpoint_every": 1,
              "trainable_policy": policy, "backend_manifest": manifest,
              "validation_index": [{"world_id": "software-dev", "role": "diagnostic_only", "city": "synthetic", "synthetic": True}],
              "camera": {"translation_scaling": "fixed_units", "framewise": False, "camera_channel": "native", "temporal_stride": 4}}
    dataset, trained, result = train_with_existing_driver(config, work/"driver")
    context = {"scope": "software_control", "index": dataset.index, "clips": dataset.clips,
               "source_audit": dataset.source_audit, "source_code_sha256": code_fingerprint()}
    inference = runtime_config(); inference["training"] = None
    for name, value in (("driver-config", config), ("software-context", context), ("inference-config", inference), ("driver-result", result)):
        (work/(name+".json")).write_text(json.dumps(value, sort_keys=True, allow_nan=False)+"\n")
    return {"work": work, "source": source, "weights": weights, "config": config, "context": context,
            "inference_config": inference, "checkpoint": work/"driver/update-00000002.pt", "trained": trained}


class TinyInferenceBackend:
    """Actual trained tiny PEFT inference, explicitly not an original Wan model."""
    objective = "flow_matching"

    def __init__(self, models):
        self.models = models
        self.call_index = 0
        self.native_frames = 0

    def reset(self, *, seed):
        self.call_index = 0

    def generate(self, *, image, camera, native_frames, seed):
        from rswm_inria.numeric import payload_hash
        x = fixed_input()
        if camera["camera_poses"] is not None:
            translation = np.asarray(camera["camera_poses"])[:, :3, 3]
            x = x + float(translation.mean()) * .01
        with torch.no_grad():
            signal = sum(model(x).mean().item() for model in self.models.values())
        delta = int(round(signal*20))
        pixels = np.asarray(image, dtype=np.int16)
        output = Image.fromarray(np.clip(pixels + delta, 0, 255).astype(np.uint8))
        frames = [image.copy()] + [output.copy() for _ in range(native_frames-1)]
        self.call_index += 1
        self.native_frames += native_frames
        return {"frames": frames, "camera_payload_sha256": payload_hash(camera)}

    def snapshot(self):
        return {"call_index": self.call_index}

    def restore(self, state):
        self.call_index = state["call_index"]

    def resource_counters(self):
        return {"status": "ok", "counter_epoch": "actual-tiny-peft-software-lifecycle", "values": {"native_frames": self.native_frames}}

    def resource_state(self):
        return {"call_index": self.call_index}


def load_software_backend(source_dir, weights_dir, device):
    from safetensors.torch import load_file
    from rswm_inria.training import locked_tree
    from rswm_inria.legacy_export import validate_adapter_release, validate_loaded_expert
    if device != "cpu":
        raise ValueError("This explicit tiny software deployment only supports CPU")
    config = json.loads((Path(source_dir)/"backend_config.json").read_text())
    if config["training"] is not None:
        raise ValueError("Fresh inference deployment must have training=null")
    reference = config["adapter_release"]
    release = validate_adapter_release(reference, expected_weights_tree_sha256=digest(locked_tree(weights_dir)), allow_software=True)
    if release["scope"] != "software_control":
        raise ValueError("Tiny loader cannot represent a real trained Wan candidate")
    original_inference = json.loads((Path(reference["root"])/"inference-config.json").read_text())
    if config["sampling"] != original_inference["sampling"] or config["config_key"] != original_inference["config_key"]:
        raise ValueError("Tiny deployment changed retained inference configuration")
    models = {}
    for tag, seed in (("low", 11), ("high", 29)):
        base = TinyBackbone(seed)
        base.load_state_dict(load_file(Path(weights_dir)/tag/"model.safetensors"), strict=True)
        models[tag] = strict_lora_from_pretrained(peft.PeftModel, base, Path(reference["root"])/(tag+"_noise"), is_trainable=False)
        validate_loaded_expert(models[tag], tag, reference, allow_software=True)
    return TinyInferenceBackend(models)


def registry_config(exported):
    manifest = {name: exported[name] for name in ("source_dir", "source_tree_sha256", "weights_dir", "weights_tree_sha256", "entrypoint", "objective")}
    manifest.update(approved=True, license_record="Only this original tiny CPU software fixture is approved for software execution; no real Wan/model approval.")
    return {"model_id": "actual-tiny-peft-software-control", "adapter": "numeric-camera", "device": "cpu",
            "training_seed": exported["training_seed"], "backend_manifest": manifest,
            "camera": exported["camera"], "training_camera_sha256": exported["training_camera_sha256"],
            "adapter_release": exported["adapter_release"]}


def fresh_process_check(config_path, expected_path, result_path):
    from safetensors.torch import load_file, save_file
    from rswm_inria.registry import create_adapter, capabilities
    config = json.loads(Path(config_path).read_text())
    adapter = create_adapter(config)
    expected = load_file(expected_path)
    x = load_file(Path(expected_path).with_name("fixed-input.safetensors"))["input"]
    errors = {}
    unmerged_outputs, merged_outputs = {}, {}
    for tag, model in adapter.backend.models.items():
        with torch.no_grad():
            unmerged = model(x)
            torch.testing.assert_close(unmerged, expected[tag], **UNMERGED_TOLERANCE)
            merged = model.merge_and_unload(safe_merge=True)(x)
            torch.testing.assert_close(merged, expected[tag], **MERGED_TOLERANCE)
        errors[tag] = {"unmerged_max_abs": (unmerged-expected[tag]).abs().max().item(),
                       "merged_max_abs": (merged-expected[tag]).abs().max().item()}
        unmerged_outputs[tag], merged_outputs[tag] = unmerged.contiguous(), merged.contiguous()
    root = Path(result_path).parent
    save_file(unmerged_outputs, root/"fresh-unmerged-forward.safetensors")
    save_file(merged_outputs, root/"fresh-merged-forward.safetensors")
    result = {"status": "PASS_ACTUAL_PEFT_FRESH_PROCESS_SOFTWARE_ONLY", "scope": "software_control",
              "pid": os.getpid(), "errors": errors, "capabilities": capabilities(config),
              "tolerances": {"unmerged": UNMERGED_TOLERANCE, "merged": MERGED_TOLERANCE},
              "real_model_validation": "NOT_RUN", "real_gpu_validation": "NOT_RUN"}
    result["retained_forward"] = {name: file_hash(root/name) for name in
                                  ("fixed-input.safetensors", "expected-forward.safetensors",
                                   "fresh-unmerged-forward.safetensors", "fresh-merged-forward.safetensors")}
    Path(result_path).write_text(json.dumps(result, sort_keys=True, allow_nan=False)+"\n")
    return result


def reproduce_one(work, out, mode="new_lora"):
    from safetensors.torch import save_file
    from rswm_inria.legacy_export import _export_bound_backend, validate_adapter_release
    from rswm_inria.registry import capabilities
    from rswm_inria import suite
    from rswm_inria.tasks import packages
    work, out = Path(work).resolve(), Path(out).resolve()
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True)
    # This is retained before training/export/any numeric comparison runs.
    tolerances = {"scope": "software_control", "chosen_before_roundtrip": True,
                  "unmerged": UNMERGED_TOLERANCE, "merged": MERGED_TOLERANCE}
    (out/"frozen-tolerances.json").write_text(json.dumps(tolerances, sort_keys=True)+"\n")
    fixture = create_driver_fixture(out/"fixture", mode)
    save_file({"input": fixed_input()}, out/"fixed-input.safetensors")
    expected = {tag: model(fixed_input()).detach().contiguous() for tag, model in fixture["trained"].model.items()}
    save_file(expected, out/"expected-forward.safetensors")
    exported = _export_bound_backend(TinyBackend(fixture["weights"], fixture["source"], fixture["config"]["trainable_policy"]), checkpoint=fixture["checkpoint"],
                                    config=fixture["config"], context=fixture["context"], inference_config=fixture["inference_config"],
                                    out=out/"release", deployment_out=out/"deployment")
    config = registry_config(exported)
    (out/"numeric-config.json").write_text(json.dumps(config, sort_keys=True)+"\n")
    validate_adapter_release(exported["adapter_release"], expected_weights_tree_sha256=exported["weights_tree_sha256"],
                             expected_camera_sha256=exported["training_camera_sha256"], expected_training_seed=83, allow_software=True)
    child_environment = os.environ.copy()
    if "PYTHONPATH" in child_environment:
        child_environment["PYTHONPATH"] = os.pathsep.join(str(Path(part).resolve()) for part in child_environment["PYTHONPATH"].split(os.pathsep) if part)
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--fresh-check", str(out/"numeric-config.json"),
                                "--expected", str(out/"expected-forward.safetensors"), "--fresh-result", str(out/"fresh-process.json")],
                               cwd=work, env=child_environment, text=True, capture_output=True, check=False)
    (out/"fresh-process.stdout.log").write_text(completed.stdout)
    (out/"fresh-process.stderr.log").write_text(completed.stderr)
    if completed.returncode:
        raise RuntimeError("Actual fresh-process PEFT control failed; retained stderr available")
    inputs = out/"suite-inputs"; inputs.mkdir()
    world = np.zeros((2048, 2048, 3), np.uint8); world[:] = [64, 128, 192]
    Image.fromarray(world).save(inputs/"world.png")
    task = packages("actual-tiny-peft-software-world", "diagnostic_only", 0)[0][0]
    (inputs/"task.json").write_text(json.dumps(task)+"\n")
    # Existing suite city vocabulary is fixed. "sfo" is only the stratum key
    # for this diagnostic fixture; these pixels have no geographic provenance.
    plan = suite.build_plan([{"task_path": str(inputs/"task.json"), "world_path": str(inputs/"world.png"), "city": "sfo"}],
                            [config], [0], digest({"synthetic": True, "scope": "software_control"}), "0"*40)
    outcomes = suite.execute_plan(plan, out/"suite")
    replay = suite.replay_suite(plan, out/"suite")
    if len(outcomes) != 1 or outcomes[0]["status"] != "COMPLETED" or not replay[0]["replay_verified"]:
        raise RuntimeError("Actual tiny PEFT synthetic suite/replay failed")
    from rswm_inria.experiments import freeze_experiment
    try:
        freeze_experiment(plan, kind="pilot", purpose="real", budget={"max_wall_seconds": 1, "max_output_bytes": 1_000_000})
    except ValueError as error:
        real_gate = {"status": "REJECTED_SOFTWARE_FIXTURE", "reason": str(error)}
    else:
        raise RuntimeError("Synthetic PEFT fixture unexpectedly passed a real research gate")
    result = {"status": "PASS_ACTUAL_PEFT_DRIVER_EXPORT_REGISTRY_SUITE_SOFTWARE_ONLY", "scope": "software_control",
              "synthetic": True, "source_code_sha256": suite.code_fingerprint(), "driver_checkpoint_sha256": file_hash(fixture["checkpoint"]),
              "driver_receipt_sha256": file_hash(fixture["checkpoint"].with_suffix(".json")), "export": exported,
              "fresh_process": json.loads((out/"fresh-process.json").read_text()), "capabilities": capabilities(config),
              "real_experiment_gate": real_gate,
              "suite": {"runs": len(outcomes), "planned_outputs": 32, "completed_outputs": 32, "replay_verified": True,
                        "primary_selection": "first_attempt_only", "purpose": "diagnostic", "real_pilot_outputs": 0},
              "real_model_validation": "NOT_RUN", "real_gpu_validation": "NOT_RUN", "train60_acceptance": "NOT_RUN",
              "publication_acceptance": "NOT_READY"}
    result["initialization_mode"] = mode
    result["suite"]["world_scope"] = "Original synthetic pixels; sfo is a fixture stratum key, not a geolocation claim"
    (out/"receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return result


def reproduce(work, out):
    out = Path(out).resolve(); out.mkdir(parents=True)
    modes = {mode: reproduce_one(work, out/mode, mode) for mode in ("new_lora", "restored_lora")}
    result = {"status": "PASS_ACTUAL_PEFT_DRIVER_EXPORT_REGISTRY_SUITE_SOFTWARE_ONLY", "scope": "software_control",
              "synthetic": True, "modes": modes, "real_model_validation": "NOT_RUN", "real_gpu_validation": "NOT_RUN",
              "train60_acceptance": "NOT_RUN", "publication_acceptance": "NOT_READY"}
    (out/"receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work"); parser.add_argument("--out")
    parser.add_argument("--fresh-check"); parser.add_argument("--expected"); parser.add_argument("--fresh-result")
    args = parser.parse_args(argv)
    if args.fresh_check:
        fresh_process_check(args.fresh_check, args.expected, args.fresh_result)
    else:
        if not args.work or not args.out: parser.error("--work and --out are required")
        result = reproduce(args.work, args.out)
        print(json.dumps({"status": result["status"], "scope": result["scope"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
