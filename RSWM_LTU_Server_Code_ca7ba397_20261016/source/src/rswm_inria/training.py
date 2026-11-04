"""Train-only indexed RGB clips and checkpointed actual-backend training.

No surrogate loss or pretrained backend is supplied here. The approved backend
must implement its own flow-matching/diffusion objective and weight restoration.
Absent original source/weights is a hard preflight failure, never a toy fallback.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import digest, file_hash, crop, rgb
from .tasks import training_clips


def locked_tree(root):
    p = Path(root)
    if not p.is_dir() or p.is_symlink() or any(x.is_symlink() for x in p.parents):
        raise ValueError("BLOCKED: approved local source/weight directory required")
    files = sorted(p.rglob("*"))
    if any(x.is_symlink() for x in files): raise ValueError("Backend tree symlink")
    if any(x.is_file() and x.suffix in {".pyc", ".pyo"} for x in files):
        raise ValueError("Backend bytecode cache is not approved source; remove cached bytecode before locking")
    result = {x.relative_to(p).as_posix(): file_hash(x) for x in files if x.is_file() and "__pycache__" not in x.parts}
    if not result: raise ValueError("BLOCKED: empty backend tree")
    return result


def load_approved_backend(manifest, *, device):
    """Import precisely the user-approved source file, no installed-name guessing."""
    required = {"source_dir", "source_tree_sha256", "entrypoint", "weights_dir", "weights_tree_sha256", "objective", "approved", "license_record"}
    if not isinstance(manifest, dict) or set(manifest) != required or manifest["approved"] is not True:
        raise ValueError("BLOCKED: approved original backend manifest required")
    if manifest["objective"] not in {"flow_matching", "diffusion"} or not manifest["license_record"]:
        raise ValueError("BLOCKED: actual backend objective and rights record required")
    if digest(locked_tree(manifest["source_dir"])) != manifest["source_tree_sha256"]:
        raise ValueError("Backend source changed")
    if digest(locked_tree(manifest["weights_dir"])) != manifest["weights_tree_sha256"]:
        raise ValueError("Backend base weights changed")
    relative = Path(manifest["entrypoint"])
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py": raise ValueError("Invalid backend entrypoint")
    entry = Path(manifest["source_dir"]) / relative
    if not entry.is_file(): raise ValueError("BLOCKED: original backend entrypoint absent")
    spec = importlib.util.spec_from_file_location("rswm_approved_backend_" + manifest["source_tree_sha256"][:16], entry)
    module = importlib.util.module_from_spec(spec)
    previous_path=list(sys.path);previous_bytecode=sys.dont_write_bytecode
    sys.modules[spec.name]=module
    try:
        sys.path.insert(0,str(Path(manifest['source_dir']).resolve()))
        sys.dont_write_bytecode=True
        # Compile the locked bytes, rather than consulting a timestamp .pyc.
        exec(compile(entry.read_bytes(), str(entry), "exec"), module.__dict__)
        backend = module.load_backend(weights_dir=manifest["weights_dir"], device=device)
    finally:
        sys.path[:]=previous_path;sys.dont_write_bytecode=previous_bytecode
    if getattr(backend, "objective", None) != manifest["objective"]:
        raise ValueError("Loaded backend objective mismatch")
    return backend


class ClipDataset:
    """Deterministic 33-view train60 index; no dev/test/calibration image access."""
    def __init__(self, index, clips, root, *, diagnostic=False, source_audit=None):
        self.root = Path(root).resolve(); self.diagnostic = diagnostic
        self.worlds = {row["world_id"]: row for row in index["worlds"]}
        if len(self.worlds) != len(index["worlds"]): raise ValueError("Duplicate world")
        self.clips = clips["clips"]
        if clips.get("clip_count") != len(self.clips): raise ValueError("Clip count mismatch")
        if clips.get("view_events") != 33 * len(self.clips): raise ValueError("Clip exposure count mismatch")
        selected = {c["world_id"] for c in self.clips}
        if not diagnostic and (len(selected) != 60 or len(self.clips) != 3840):
            raise ValueError("Formal training requires complete train60/3840clips")
        if not diagnostic:
            from collections import Counter
            from .data import ROLES
            quotas = {city: count for city, (role, count) in ROLES.items() if role == "train"}
            if Counter(self.worlds[w].get("city") for w in selected if w in self.worlds) != quotas:
                raise ValueError("Formal train60 city/role quotas differ from the frozen source selection")
            self._bind_source_audit(index, selected, source_audit)
        self._verified = set()
        expected = {w: {c["clip_id"]: c for c in training_clips(w, "train")} for w in selected}
        seen = set()
        for c in self.clips:
            w = self.worlds.get(c["world_id"])
            if not w or w.get("role") != "train": raise ValueError("Training role violation")
            key = (c["world_id"], c["clip_id"])
            if key in seen or digest(expected[key[0]].get(key[1])) != digest(c): raise ValueError("Clip geometry/index mismatch")
            seen.add(key)
        if not diagnostic and any({cid for wid,cid in seen if wid==w} != set(expected[w]) for w in selected):
            raise ValueError("Incomplete world clip inventory")
        self.index_sha256 = digest({"index": index, "clips": clips})

    def _bind_source_audit(self, index, selected, report):
        from .materialized import validate_audit_receipt
        from .suite import code_fingerprint, _load_task
        if not isinstance(report, dict):
            raise ValueError("Formal training requires a canonical core120 source audit")
        expected = []
        task_ids = set()
        for row in index.get("tasks", []):
            if row.get("world_id") not in selected:
                continue
            relative = Path(row.get("task_path", ""))
            if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("tasks",):
                raise ValueError("Training task path escape")
            path = self.root.parent / relative
            if path.is_symlink() or any(p.is_symlink() for p in path.parents):
                raise ValueError("Training task symlink")
            task, _labels = _load_task(path)
            world = self.worlds[row["world_id"]]
            if any(task.get(k) != row.get(k) for k in ("task_id", "world_id", "role", "family")):
                raise ValueError("Training task/index identity mismatch")
            if task["task_id"] in task_ids:
                raise ValueError("Duplicate training task")
            task_ids.add(task["task_id"])
            expected.append({**{k: task[k] for k in ("task_id", "world_id", "role", "origin_index", "family", "protocol")},
                             **{k: world[k] for k in ("city", "world_file_sha256", "world_pixel_sha256")},
                             "task_sha256": digest(task), "source_manifest_sha256": index.get("source_manifest_sha256")})
        if len(expected) != 16 * len(selected) or any(sum(r["world_id"] == w for r in expected) != 16 for w in selected):
            raise ValueError("Formal training requires each frozen world's complete task inventory")
        validate_audit_receipt(report, {"expected": expected, "source_code_sha256": code_fingerprint()}, required_profile="core120")
        for w in selected:
            row, audited = self.worlds[w], report["worlds"][w]
            if any(row.get(k) != audited.get(k) for k in ("source_path", "source_sha256", "city", "role", "world_file_sha256", "world_pixel_sha256")):
                raise ValueError("Training world/source audit mismatch")
            if row.get("path") != w + ".png":
                raise ValueError("Training world filename differs from audited artifact")
        self.source_audit_sha256 = digest(report)
        self.source_audit = report

    def __len__(self): return len(self.clips)

    def __getitem__(self, i):
        c = self.clips[i]; row = self.worlds[c["world_id"]]
        relative = Path(row["path"])
        p = self.root / relative
        if relative.is_absolute() or ".." in relative.parts or p.is_symlink() or any(a.is_symlink() for a in p.parents):
            raise ValueError("Training world path escape")
        if file_hash(p) != row.get("world_file_sha256", row.get("sha256")):
            raise ValueError("Training world hash mismatch")
        with Image.open(p) as im:
            if im.format != "PNG" or im.mode != "RGB": raise ValueError("Lossless RGB world required")
            world = rgb(np.asarray(im), (2048, 2048, 3))
        if row.get("world_pixel_sha256"):
            import hashlib
            if hashlib.sha256(world.tobytes()).hexdigest() != row["world_pixel_sha256"]:
                raise ValueError("Training world pixel hash mismatch")
        frames = np.stack([crop(world, q, c["origin"]) for q in c["q"]])
        return {"clip_id": c["clip_id"], "world_id": c["world_id"], "images": frames,
                "q": np.asarray(c["q"], dtype=np.int64), "role": "train"}


def cursor_sample(dataset, seed, position):
    if type(seed) is not int or seed < 0 or type(position) is not int or position < 0 or not len(dataset):
        raise ValueError("Invalid data cursor")
    epoch, offset = divmod(position, len(dataset))
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(len(dataset))
    return dataset[int(order[offset])]


def configure_parameters(backend, policy):
    # LoRA/adapters may be inserted by configure_trainable; enumerate afterward.
    backend.configure_trainable(policy)
    params=dict(backend.model.named_parameters())
    trainable={name:parameter for name,parameter in params.items() if parameter.requires_grad}
    if not trainable:raise ValueError('No trainable parameters')
    return params,trainable


def _parameter_byte_chunks(parameter, max_bytes):
    """Copy bounded CPU chunks, including noncontiguous parameter views."""
    import torch
    if type(max_bytes) is not int or max_bytes < parameter.element_size():
        raise ValueError("Parameter audit chunk must hold at least one element")
    value = parameter.detach()
    count = max_bytes // value.element_size()
    def cpu_bytes(piece):
        # Even a one-element strided slice may report is_contiguous() while
        # retaining a non-unit stride, which uint8 reinterpretation rejects.
        # Allocate only this bounded chunk in canonical CPU storage.
        retained = torch.empty(piece.shape, dtype=piece.dtype, device="cpu")
        retained.copy_(piece)
        return retained.reshape(-1).view(torch.uint8).numpy()
    if value.is_contiguous():
        flat = value.view(-1)
        for start in range(0, flat.numel(), count):
            yield cpu_bytes(flat[start:start+count])
    elif value.numel() <= count:
        yield cpu_bytes(value)
    elif value.ndim == 1:
        for start in range(0, value.numel(), count):
            yield cpu_bytes(value[start:start+count])
    else:
        for row in value:
            yield from _parameter_byte_chunks(row, max_bytes)


def parameter_content_hash(parameter, *, max_bytes=4*1024*1024):
    """Frozen content identity; tensor version counters cannot detect .data edits."""
    from .geometry import canonical
    checksum = hashlib.sha256(canonical({"dtype": str(parameter.dtype), "shape": list(parameter.shape)}))
    for chunk in _parameter_byte_chunks(parameter, max_bytes):
        checksum.update(memoryview(chunk))
    return checksum.hexdigest()


def tensor_state_sha256(state):
    """Bind exact named tensor content without cloning a frozen backbone."""
    import torch
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or not torch.is_tensor(v) for k,v in state.items()):
        raise ValueError("Exact named tensor state required")
    if any(not torch.isfinite(v).all().item() for v in state.values()):
        raise ValueError("Nonfinite checkpoint tensor state")
    return digest({name: parameter_content_hash(value) for name,value in state.items()})


def parameter_snapshot(params, trainable, *, max_bytes=4*1024*1024):
    # Only adapters are cloned. A 28B frozen backbone is hashed in bounded CPU
    # chunks rather than doubling resident GPU memory on every update.
    return {"trainable": {name: p.detach().clone() for name, p in params.items() if name in trainable},
            "frozen": {name: parameter_content_hash(p, max_bytes=max_bytes) for name, p in params.items() if name not in trainable}}


def parameter_changes(params, snapshot, *, max_bytes=4*1024*1024):
    import torch
    changed = [name for name, before in snapshot["trainable"].items() if not torch.equal(before, params[name].detach())]
    changed.extend(name for name, before in snapshot["frozen"].items() if parameter_content_hash(params[name], max_bytes=max_bytes) != before)
    return changed


def verify_parameter_binding(model, params, trainable):
    """Do not let a backend replace or unfreeze parameters after enumeration."""
    current = dict(model.named_parameters())
    if set(current) != set(params) or any(current[n] is not p for n, p in params.items()):
        raise ValueError("Backend model parameter binding changed")
    if any(p.requires_grad != (name in trainable) for name, p in params.items()):
        raise ValueError("Backend trainable/frozen parameter selection changed")


def merge_active_groups(active, reported, trainable):
    """Optional real backend activity adds checks; it never weakens global gates."""
    if not isinstance(reported, dict) or not reported:
        raise ValueError("Actual backend must identify each active trainable expert")
    for group, names in reported.items():
        if not isinstance(group, str) or not group or not isinstance(names, list) or not names or any(not isinstance(n, str) or n not in trainable for n in names) or len(set(names)) != len(names):
            raise ValueError("Invalid active trainable parameter group")
        selected = set(names)
        if group in active and active[group] != selected:
            raise ValueError("Active trainable group identity changed")
        active[group] = selected


def audit_trainable_gradients(trainable, active_groups):
    import torch
    if any(p.grad is not None and not torch.isfinite(p.grad).all().item() for p in trainable.values()):
        raise ValueError("Nonfinite gradient")
    gradients = {name for name, p in trainable.items() if p.grad is not None and torch.count_nonzero(p.grad).item()}
    if not gradients:
        raise ValueError("No finite nonzero trainable gradients")
    for group, names in active_groups.items():
        if not gradients.intersection(names):
            raise ValueError("No finite nonzero gradients for active expert: " + group)
    return gradients


def validate_resume_cursor(state, config):
    step, position = state.get("step"), state.get("position")
    if type(step) is not int or not 0 <= step <= config["updates"]:
        raise ValueError("Resume optimizer step is outside the frozen training budget")
    if type(position) is not int or position != step * config["batch_size"] * config["accumulation"]:
        raise ValueError("Resume data cursor does not match its optimizer boundary")
    return step, position


def validate_checkpoint_receipt(checkpoint, contract):
    from .suite import _json
    path = Path(checkpoint)
    if path.is_symlink() or any(a.is_symlink() for a in path.parents):
        raise ValueError("Checkpoint path symlink")
    receipt_path = path.with_suffix(".json")
    if receipt_path.is_symlink():
        raise ValueError("Checkpoint receipt symlink")
    receipt = _json(receipt_path)
    if not isinstance(receipt, dict):
        raise ValueError("Checkpoint receipt must be an object")
    if receipt.get("contract_sha256") != contract or receipt.get("checkpoint_sha256") != file_hash(path):
        raise ValueError("Checkpoint receipt/hash mismatch")
    if "schema" in receipt:
        required = {"schema", "contract_sha256", "checkpoint_sha256", "audit", "training", "attempt"}
        if receipt["schema"] != "rswm-training-checkpoint-v2" or set(receipt) not in (required, required|{"expert_state_sha256"}):
            raise ValueError("Unknown checkpoint receipt format")
        fields = {"config_sha256", "dataset_sha256", "source_audit_sha256", "camera_manifest_sha256",
                  "backend_manifest_sha256", "source_code_sha256", "training_seed", "step", "clip_cursor"}
        binding = receipt["training"]
        if not isinstance(binding, dict) or set(binding) != fields or any(type(binding[k]) is not int or binding[k] < 0 for k in ("training_seed","step","clip_cursor")):
            raise ValueError("Checkpoint training identity fields invalid")
        if any(not isinstance(binding[k],str) or len(binding[k])!=64 or any(c not in "0123456789abcdef" for c in binding[k]) for k in fields if k.endswith("sha256")):
            raise ValueError("Checkpoint training identity hash invalid")
        experts = receipt.get("expert_state_sha256")
        if "expert_state_sha256" in receipt and (not isinstance(experts,dict) or set(experts)!={"low","high"} or any(not isinstance(v,str) or len(v)!=64 or any(c not in "0123456789abcdef" for c in v) for v in experts.values())):
            raise ValueError("Checkpoint expert state identity invalid")
        attempt = receipt["attempt"]
        if not isinstance(attempt, dict) or set(attempt) != {"journal_path", "journal_sha256", "journal_head_sha256"}:
            raise ValueError("Checkpoint attempt identity invalid")
        relative = Path(attempt["journal_path"])
        log = path.parent/relative
        if relative.is_absolute() or ".." in relative.parts or log.is_symlink() or any(p.is_symlink() for p in log.parents) or file_hash(log) != attempt["journal_sha256"]:
            raise ValueError("Checkpoint retained attempt journal mismatch")
        from .replay import load_json
        from .runner import verify_journal
        rows = [load_json(line) for line in log.read_text().splitlines()]
        if not rows or verify_journal(rows) != attempt["journal_head_sha256"] or rows[0].get("contract_sha256") != contract or rows[0].get("training") != {k:v for k,v in binding.items() if k not in {"step","clip_cursor"}}:
            raise ValueError("Checkpoint journal source/config binding mismatch")
    elif any(k not in {"contract_sha256", "checkpoint_sha256", "audit"} for k in receipt):
        raise ValueError("Unversioned checkpoint identity fields")
    return receipt


def retained_training_attempts(out, contract, training_identity=None):
    """Read existing training Journal chains; a started forward remains charged."""
    from .replay import load_json
    from .runner import verify_journal
    root = Path(out)/"attempts"
    if not root.exists():
        if (Path(out)/"training-contract.json").exists():
            raise ValueError("Missing retained training attempt costs")
        return [], 0
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("Training attempt path symlink")
    retained, forwards = [], 0
    for number, path in enumerate(sorted(root.iterdir())):
        if path.name != f"attempt-{number:03d}" or not path.is_dir() or path.is_symlink():
            raise ValueError("Unexpected training attempt inventory")
        log = path/"events.jsonl"
        if log.is_symlink() or not log.is_file():
            raise ValueError("Missing retained training attempt journal")
        content = log.read_bytes()
        if not content.endswith(b"\n"):
            raise ValueError("Incomplete retained training attempt journal")
        rows = [load_json(line) for line in content.decode("utf-8").splitlines()]
        if not rows or rows[0].get("kind") != "training_session" or rows[0].get("contract_sha256") != contract:
            raise ValueError("Training attempt contract mismatch")
        if training_identity is not None and rows[0].get("training") != training_identity:
            raise ValueError("Training attempt source/config identity mismatch")
        head = verify_journal(rows)
        pending = set(); calls = {}
        phases = {"training_update", "training_loss", "optimizer_step", "validation_call"}
        for row in rows[1:]:
            if row.get("kind") == "validation_forward_started":
                if type(row.get("forward")) is not int or row["forward"] != forwards+1:
                    raise ValueError("Training validation cost sequence mismatch")
                forwards += 1; pending.add(row["forward"])
            elif row.get("kind") == "validation_forward_finished":
                if row.get("forward") not in pending or row.get("status") not in {"SUCCEEDED", "FAILED"} or type(row.get("wall_seconds")) not in {int,float} or not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] < 0:
                    raise ValueError("Training validation cost outcome mismatch")
                pending.remove(row["forward"])
            elif row.get("kind", "").removesuffix("_started") in phases:
                key = (row["kind"].removesuffix("_started"), row.get("update"), row.get("microbatch"))
                if key in calls:
                    raise ValueError("Duplicate training cost call")
                calls[key] = row
            elif row.get("kind", "").removesuffix("_finished") in phases:
                key = (row["kind"].removesuffix("_finished"), row.get("update"), row.get("microbatch"))
                if key not in calls or type(row.get("wall_seconds")) not in {int,float} or not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] < 0:
                    raise ValueError("Training cost call outcome mismatch")
                del calls[key]
            elif row.get("kind") != "training_completed":
                raise ValueError("Unexpected training cost journal event")
        retained.append({"journal_path": log.relative_to(Path(out)).as_posix(),
                         "journal_sha256": hashlib.sha256(content).hexdigest(), "journal_head_sha256":head,
                         "unfinished_validation_forwards": sorted(pending),
                         "unfinished_calls": [{"phase":k[0],"update":k[1],"microbatch":k[2],
                                               "status":"UNKNOWN_NO_COMPLETION"} for k in calls]})
    return retained, forwards


def start_training_attempt(out, contract, training_identity):
    from .runner import Journal
    from .artifact_io import save_bytes_exclusive
    from .geometry import canonical
    from .suite import _json
    root = Path(out)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("Training output path symlink")
    root.mkdir(parents=True, exist_ok=True)
    binding = root/"training-contract.json"
    expected = {"contract_sha256":contract, "training":training_identity}
    new_binding = not binding.exists() and not binding.is_symlink()
    if not new_binding:
        if binding.is_symlink() or _json(binding) != expected:
            raise ValueError("Retained training source/config contract mismatch")
    else:
        if any(p.name != "training-coordinator.lock" for p in root.iterdir()):
            raise FileExistsError("Training output lacks its retained attempt contract")
        save_bytes_exclusive(binding, canonical(expected)+b"\n")
        # The first attempt is created only after the context bytes are durable.
        (root/"attempts").mkdir()
    retained, forwards = retained_training_attempts(root, contract, training_identity)
    if not new_binding and not retained:
        raise ValueError("Missing retained training attempt costs")
    attempts = root/"attempts"; attempts.mkdir(exist_ok=True)
    # mkdir is the cooperative exclusive claim; never resume a failed writer.
    path = attempts/f"attempt-{len(retained):03d}"; path.mkdir()
    journal = Journal(path)
    journal.append("training_session", contract_sha256=contract, training=training_identity,
                   prior_attempts=retained, retained_validation_forwards=forwards)
    return journal, forwards


def validate_dev_index(validation, dataset):
    if not isinstance(validation, list) or not validation:
        raise ValueError("Training validation must bind actual dev worlds")
    seen = set()
    for row in validation:
        wid = row.get("world_id")
        audited = dataset.source_audit["worlds"].get(wid)
        indexed = dataset.worlds.get(wid)
        if wid in seen or not audited or not indexed or audited.get("city") != "kitsap" or audited.get("role") != "dev":
            raise ValueError("Training validation must use distinct audited Kitsap dev worlds")
        seen.add(wid)
        for key in ("city", "role", "source_sha256", "world_file_sha256", "world_pixel_sha256"):
            if row.get(key) != audited.get(key) or indexed.get(key) != audited.get(key):
                raise ValueError("Dev validation source identity mismatch")
        if row.get("path") != indexed.get("path") or indexed.get("path") != wid + ".png":
            raise ValueError("Dev validation world path mismatch")
        path = dataset.root / row["path"]
        if path.is_symlink() or file_hash(path) != audited["world_file_sha256"]:
            raise ValueError("Dev validation world bytes differ from audited source")


@contextmanager
def _training_execution_lock(out):
    """One cooperative driver owns cost inspection and training for this run."""
    import fcntl
    root = Path(out)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("Training output path symlink")
    root.mkdir(parents=True, exist_ok=True)
    path = root/"training-coordinator.lock"
    if path.is_symlink():
        raise ValueError("Training coordinator lock symlink")
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another training coordinator is active") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def train(dataset, backend, config, out, *, resume=None):
    """Run checkpointed backend training under exclusive run ownership."""
    # Hold ownership before reading prior attempts, including all failure exits.
    with _training_execution_lock(out):
        return _train(dataset, backend, config, out, resume=resume)


def _train(dataset, backend, config, out, *, resume=None):
    """AMP/accumulation driver. Backend owns the actual backbone's loss schedule.

    Checkpoints are taken at optimizer boundaries and restore model, optimizer,
    scheduler, scaler, Python/NumPy/Torch/CUDA RNG and exact clip cursor. No
    validation world enters gradients; backend validation receives only dev.
    """
    import torch
    from .adapters.camera_conditioning import CameraConfig, build_camera_batch, backend_payload, to_native_rgb
    required = {"updates", "batch_size", "accumulation", "lr", "seed", "precision", "device", "camera", "trainable_policy", "checkpoint_every", "backend_manifest", "validation_index"}
    if set(config) != required: raise ValueError("Frozen training configuration fields required")
    for k in ("updates", "batch_size", "accumulation", "checkpoint_every"):
        if type(config[k]) is not int or config[k] <= 0: raise ValueError("Positive integer training budget required")
    if type(config["seed"]) is not int or config["seed"] < 0 or not math.isfinite(config["lr"]) or config["lr"] <= 0:
        raise ValueError("Training seed/lr invalid")
    if config["precision"] not in {"fp32", "fp16", "bf16"}: raise ValueError("Unknown precision")
    if dataset.diagnostic: raise ValueError("Diagnostic fixture cannot authorize actual training")
    validate_dev_index(config["validation_index"], dataset)
    cam = CameraConfig(**config["camera"])
    contract = digest({"config": config, "dataset": dataset.index_sha256,
                       "source_audit": dataset.source_audit_sha256, "camera": cam.manifest()})
    from .suite import code_fingerprint
    identity = {"config_sha256": digest(config), "dataset_sha256": dataset.index_sha256,
                "source_audit_sha256":dataset.source_audit_sha256, "camera_manifest_sha256":digest(cam.manifest()),
                "backend_manifest_sha256":digest(config["backend_manifest"]), "source_code_sha256":code_fingerprint(),
                "training_seed":config["seed"]}
    if resume and callable(getattr(backend, "bind_validation_journal", None)) and Path(resume).parent.resolve() != Path(out).resolve():
        raise ValueError("Wan resume must retain its training output directory and attempt costs")
    journal, retained_forwards = start_training_attempt(out, contract, identity)
    if callable(getattr(backend, "bind_training_context", None)):
        backend.bind_training_context(dataset, config)
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config["seed"])
    params,trainable=configure_parameters(backend,config["trainable_policy"])
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1.-step/config["updates"]))
    use_amp = config["precision"] != "fp32"
    dtype = torch.float16 if config["precision"] == "fp16" else torch.bfloat16
    use_scaler=config["precision"] == "fp16" and config["device"].startswith("cuda")
    scaler=(torch.amp.GradScaler("cuda",enabled=use_scaler) if hasattr(torch.amp,'GradScaler')
            else torch.cuda.amp.GradScaler(enabled=use_scaler))
    step = position = 0
    if resume:
        # Only this driver's own checkpoint is loaded; never arbitrary weights.
        resume_receipt = validate_checkpoint_receipt(resume, contract)
        state = torch.load(resume, map_location='cpu', weights_only=True)
        if state["contract_sha256"] != contract: raise ValueError("Resume contract mismatch")
        step, position = validate_resume_cursor(state, config)
        if resume_receipt.get("schema") == "rswm-training-checkpoint-v2" and resume_receipt["training"] != {**identity,"step":step,"clip_cursor":position}:
            raise ValueError("Resume receipt differs from actual training identity/step/cursor")
        backend.model.load_state_dict(state["model"]); scheduler.load_state_dict(state["scheduler"])
        if callable(getattr(backend, "quiesce_training", None)):
            backend.quiesce_training()
        if "expert_state_sha256" in resume_receipt:
            if not callable(getattr(backend,"checkpoint_expert_state_sha256",None)) or backend.checkpoint_expert_state_sha256() != resume_receipt["expert_state_sha256"]:
                raise ValueError("Resume actual expert state differs from retained receipt")
        optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        if callable(getattr(backend, "load_checkpoint_training_state", None)):
            backend.load_checkpoint_training_state(state.get("backend_training"), step=step)
        random.setstate(state["python_rng"])
        ns = state["numpy_rng"]; np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"]: torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng"]])
    if callable(getattr(backend, "bind_validation_journal", None)):
        backend.bind_validation_journal(journal, consumed_forwards=retained_forwards)
    if callable(getattr(backend, "check_remaining_validation_budget", None)):
        backend.check_remaining_validation_budget(updates_remaining=config["updates"]-step)
    out = Path(out)
    backend.model.train(); audit = []
    for update in range(step, config["updates"]):
        if callable(getattr(backend, "quiesce_training", None)):
            backend.quiesce_training()
        update_started = time.perf_counter()
        journal.append("training_update_started", update=update+1, clip_cursor=position)
        before = parameter_snapshot(params, trainable)
        optimizer.zero_grad(set_to_none=True); active_groups = {}
        for micro in range(config["accumulation"]):
            samples = [cursor_sample(dataset, config["seed"], position+j) for j in range(config["batch_size"])]
            position += config["batch_size"]
            native = np.stack([np.stack([to_native_rgb(f) for f in s["images"]]) for s in samples])
            cameras = [backend_payload(build_camera_batch(s["q"][1:], s["q"][0], cam), expected_config_sha256=digest(cam.manifest())) for s in samples]
            journal.append("training_loss_started", update=update+1,microbatch=micro,clip_cursor=position)
            loss_started=time.perf_counter(); loss_error=None
            try:
                images = torch.from_numpy(native.copy()).to(device=config["device"], dtype=torch.float32).permute(0,1,4,2,3) / 127.5 - 1
                with torch.autocast(device_type=config["device"].split(':')[0], dtype=dtype, enabled=use_amp):
                    loss = backend.loss(images=images, cameras=cameras, update=update)
                    if loss.ndim != 0 or not torch.isfinite(loss).item(): raise ValueError("Actual backend loss must be finite scalar")
                    loss = loss / config["accumulation"]
                if callable(getattr(backend, "active_trainable_groups", None)):
                    merge_active_groups(active_groups, backend.active_trainable_groups(), trainable)
                verify_parameter_binding(backend.model, params, trainable)
                scaler.scale(loss).backward()
            except BaseException as exc:
                loss_error=type(exc).__name__
                raise
            finally:
                try:
                    if callable(getattr(backend, "quiesce_training", None)):
                        backend.quiesce_training()
                except BaseException as exc:
                    loss_error = loss_error or type(exc).__name__
                    raise
                finally:
                    memory = (backend.training_memory_observation() if callable(
                        getattr(backend, "training_memory_observation", None)) else None)
                    journal.append("training_loss_finished",update=update+1,microbatch=micro,
                                   status="FAILED" if loss_error else "SUCCEEDED",error_type=loss_error,
                                   wall_seconds=time.perf_counter()-loss_started, memory=memory)
        scaler.unscale_(optimizer)
        gradients = audit_trainable_gradients(trainable, active_groups)
        journal.append("optimizer_step_started",update=update+1)
        optimizer_started=time.perf_counter(); optimizer_error=None
        try:
            scaler.step(optimizer); scaler.update(); scheduler.step()
        except BaseException as exc:
            optimizer_error=type(exc).__name__
            raise
        finally:
            journal.append("optimizer_step_finished",update=update+1,
                           status="FAILED" if optimizer_error else "SUCCEEDED",error_type=optimizer_error,
                           wall_seconds=time.perf_counter()-optimizer_started)
        verify_parameter_binding(backend.model, params, trainable)
        changed = parameter_changes(params, before)
        if not changed or any(name not in trainable for name in changed): raise ValueError("Parameter update/freeze contract violated")
        if any(not torch.isfinite(p).all().item() for p in trainable.values()):
            raise ValueError("Nonfinite updated trainable parameter")
        if any(not set(changed).intersection(names) for names in active_groups.values()):
            raise ValueError("Active expert did not update")
        if callable(getattr(backend, "quiesce_training", None)):
            backend.quiesce_training()
        validation_before = parameter_snapshot(params, trainable)
        mode=backend.model.training
        journal.append("validation_call_started", update=update+1, planned_worlds=len(config["validation_index"]))
        validation_started = time.perf_counter(); validation_error = None
        try:
            backend.model.eval()
            with torch.no_grad():
                validation_result=backend.validate(config["validation_index"],camera_manifest=cam.manifest())
        except BaseException as exc:
            validation_error = type(exc).__name__
            raise
        finally:
            backend.model.train(mode)
            memory = (backend.training_memory_observation() if callable(
                getattr(backend, "training_memory_observation", None)) else None)
            journal.append("validation_call_finished", update=update+1,
                           status="FAILED" if validation_error else "SUCCEEDED", error_type=validation_error,
                           wall_seconds=time.perf_counter()-validation_started, memory=memory)
        verify_parameter_binding(backend.model, params, trainable)
        if parameter_changes(params, validation_before):
            raise ValueError('Dev validation mutated model parameters')
        audit.append({"update": update+1, "clip_cursor": position, "view_exposure": position*33,
                      "nonzero_gradients": sorted(gradients), "changed_parameters": changed, "dev_validation": validation_result,
                      "active_trainable_groups": {k: sorted(v) for k,v in active_groups.items()}})
        if (update+1) % config["checkpoint_every"] == 0 or update+1 == config["updates"]:
            if callable(getattr(backend, "quiesce_training", None)):
                backend.quiesce_training()
            ns=np.random.get_state()
            state={"contract_sha256":contract,"model":backend.model.state_dict(),"optimizer":optimizer.state_dict(),
                   "scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"step":update+1,"position":position,
                   "python_rng":random.getstate(),"numpy_rng":[ns[0],ns[1].tolist(),ns[2],ns[3],ns[4]],
                   "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
            if callable(getattr(backend, "checkpoint_training_state", None)):
                state["backend_training"] = backend.checkpoint_training_state()
            from .artifact_io import save_stream_exclusive, save_bytes_exclusive
            from .geometry import canonical
            path=out/f"update-{update+1:08d}.pt"; receipt_path=path.with_suffix('.json')
            if path.exists() or receipt_path.exists(): raise FileExistsError(path)
            journal.verify_persisted()
            save_stream_exclusive(path, lambda stream: torch.save(state, stream))
            snapshot = journal.root/f"checkpoint-{update+1:08d}.events.jsonl"
            save_bytes_exclusive(snapshot, (journal.root/"events.jsonl").read_bytes())
            snapshot_head = journal.verify_persisted()
            receipt={"schema":"rswm-training-checkpoint-v2", "contract_sha256":contract,"checkpoint_sha256":file_hash(path),"audit":audit,
                     "training":{**identity,"step":update+1,"clip_cursor":position},
                     "attempt":{"journal_path":snapshot.relative_to(out).as_posix(),
                                "journal_sha256":file_hash(snapshot), "journal_head_sha256":snapshot_head}}
            if callable(getattr(backend, "checkpoint_expert_state_sha256", None)):
                receipt["expert_state_sha256"] = backend.checkpoint_expert_state_sha256()
            save_bytes_exclusive(receipt_path, canonical(receipt)+b"\n")
            validate_checkpoint_receipt(path, contract)
        journal.append("training_update_finished", update=update+1, clip_cursor=position,
                       wall_seconds=time.perf_counter()-update_started)
    journal.append("training_completed", updates=config["updates"],clip_cursor=position)
    journal.verify_persisted()
    return {"updates":config["updates"],"clip_cursor":position,"contract_sha256":contract,"audit":audit}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--index',required=True);p.add_argument('--clips',required=True)
    p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--resume')
    p.add_argument('--source-audit', required=True)
    a=p.parse_args(argv)
    try:
        from .suite import _json
        config=_json(a.config)
        dataset=ClipDataset(_json(a.index),_json(a.clips),a.root,source_audit=_json(a.source_audit))
        # The CLI claims the run before loading the expensive approved model.
        with _training_execution_lock(a.out):
            backend=load_approved_backend(config['backend_manifest'],device=config['device'])
            result=_train(dataset,backend,config,a.out,resume=a.resume)
            print(json.dumps(result,allow_nan=False));return 0
    except (ValueError,OSError,KeyError,TypeError,ImportError) as exc:
        print(json.dumps({'status':'BLOCKED_OR_INVALID','error':str(exc)}));return 2


if __name__=='__main__':raise SystemExit(main())
