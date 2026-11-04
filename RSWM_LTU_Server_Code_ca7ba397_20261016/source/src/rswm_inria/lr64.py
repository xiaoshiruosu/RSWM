"""Native 1024-square LEVIR-LR64 execution, immutable scoring and replay.

This independent study does not enter the Inria core leaderboard. Its runner
uses the original 240x416 viewport and four online calls of sixteen queries.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import numpy as np
from PIL import Image

from .geometry import box, canonical, digest, file_hash, intersection, rgb
from .score_rgb import Accumulator
from .runner import Journal, GuardedAdapter, validate_response, verify_journal
from .artifact_io import save_bytes_exclusive, save_png_exclusive
from .replay import load_json, load_rgb_png
from .resources import ResourceCollector
from .suite import (DEFAULT_RETRY_POLICY, _policy, _save, _json, _metadata,
                    _attempt_folders, code_fingerprint, model_manifest,
                    suite_read_lock, SCHEMA as CORE_SUITE_SCHEMA)


STUDY = "rswm-levir100-lr64-v0.1"
TASK_SCHEMA = "rswm-lr64-task-v1"
PLAN_SCHEMA = "rswm-lr64-suite-v1"
METRIC_CONFIG = {"schema": "rswm-lr64-rgb-ledger-v1", "first_reference": "first_planned_exposure",
                 "missing": "never_repair", "auxiliary_final_target": "lr64-first-reference-final-target-v1"}


@dataclass(frozen=True)
class WorldSpec:
    world_hw: tuple = (1024, 1024)
    viewport_hw: tuple = (240, 416)
    origin: tuple = (512, 512)
    query_bounds_xy: tuple = ((-288, 288), (-288, 288))
    schema: str = "rswm-lr64-world-v1"
    study: str = STUDY

    def public(self):
        return {"world_hw": list(self.world_hw), "viewport_hw": list(self.viewport_hw),
                "query_bounds_xy": [list(pair) for pair in self.query_bounds_xy],
                "schema": self.schema, "study": self.study,
                "coordinate_unit": "native source pixel", "normalization": float(np.hypot(416, 240)),
                "x_direction": "right", "y_direction": "down"}


SPEC = WorldSpec()


def coordinates(q, *, trajectory=False):
    a = np.asarray(q)
    if a.ndim != 2 or a.shape[1] != 2 or a.dtype.kind not in "iu" or not len(a):
        raise ValueError("Integer [T,2] coordinates required")
    if np.any(a > 288) or (a.dtype.kind == "i" and np.any(a < -288)):
        raise ValueError("LR64 coordinates outside [-288,288]")
    if trajectory and (len(a) != 65 or np.any(a[0] != 0)):
        raise ValueError("LR64 requires q0 and 64 new queries")
    return a.astype(np.int64)


def crop(world, q):
    rgb(world, (1024, 1024, 3))
    p = coordinates(np.asarray(q)[None])[0]
    x0, y0, x1, y1 = box(p, SPEC.origin)
    if min(x0, y0) < 0 or max(x1, y1) > 1024:
        raise ValueError("LR64 viewport outside native world")
    return world[y0:y1, x0:x1].copy()


def paths():
    result = {}
    for variant, end in (("exact", (256, 256)), ("offset", (288, 280))):
        points = [(0, 0), (256, 256), (-256, 256), (-256, -256), end]
        q = [np.array(points[0], np.int64)]
        for a, b in zip(points[:-1], points[1:]):
            q.extend(np.rint(np.array(a) + (np.array(b) - a) * t / 16).astype(np.int64) for t in range(1, 17))
        result[variant] = np.array(q, np.int64)
    return result


def geometry_labels(q):
    q = coordinates(q, trajectory=True)
    y, x = np.mgrid[192:320, 192:320]
    visible = np.stack([(x >= cx - 208) & (x < cx + 208) & (y >= cy - 120) & (y < cy + 120) for cx, cy in q])
    pre, post = visible[:33], visible[33:]
    if visible[0].any() or visible[32].any() or not pre.any(0).all() or not post.any(0).all():
        raise ValueError("LR64 target must be new, fully seen, absent and reentered")
    first = pre.argmax(0).astype(np.int16)
    last = (32 - pre[::-1].argmax(0)).astype(np.int16)
    reentry = (33 + post.argmax(0)).astype(np.int16)
    gap = reentry - last - 1
    delta = np.abs(np.diff(q, axis=0))
    overlap = (1 - delta[:, 0]/416) * (1 - delta[:, 1]/240)
    if gap.min() < 32 or overlap.min() < .65 - 1e-12:
        raise ValueError("LR64 target gap/adjacent overlap invalid")
    labels = {"x": x, "y": y, "first_event": first, "last_prefix_visible_event": last,
              "reentry_event": reentry, "invisible_query_gap": gap,
              "fully_absent_calls": ((reentry - 1)//16 - (last - 1)//16 - 1).astype(np.int16)}
    return labels


def task_for(world_id, variant, role="diagnostic_only"):
    if not isinstance(world_id, str) or not world_id or variant not in paths() or not isinstance(role, str) or not role:
        raise ValueError("Invalid LR64 task identity")
    return {"schema": TASK_SCHEMA, "study": STUDY, "task_id": world_id + "-LR64-" + variant,
            "world_id": world_id, "pair_id": world_id + "-LR64", "role": role,
            "family": "LR64", "variant": variant, "protocol": "online16",
            "world_hw": [1024, 1024], "viewport_hw": [240, 416], "origin": [512, 512],
            "query_bounds_xy": [[-288, 288], [-288, 288]], "q": paths()[variant].tolist(),
            "target_center": [256, 256], "target_size": 128}


def validate_task(task, labels=None):
    if not isinstance(task, dict) or task.get("study") != STUDY or task.get("family") != "LR64":
        raise ValueError("Independent LR64 task required")
    expected = task_for(task.get("world_id"), task.get("variant"), task.get("role", "supplemental_unvalidated"))
    for key, value in expected.items():
        if key in {"schema", "role"} and key not in task:
            continue  # Original pinned materializer predates the execution schema.
        if task.get(key) != value:
            raise ValueError("LR64 frozen task specification mismatch: " + key)
    recomputed = geometry_labels(task["q"])
    if labels is not None and (set(labels) != set(recomputed) or any(not np.array_equal(labels[k], v) for k, v in recomputed.items())):
        raise ValueError("LR64 private labels disagree with explicit geometry")
    return recomputed


class Ledger:
    """Native 1024 allocation; failed first exposures remain permanently invalid."""
    def __init__(self, initial, labels):
        self.labels = labels
        self.known = np.zeros((1024, 1024), bool)
        self.seen = np.zeros_like(self.known); self.valid = np.zeros_like(self.known)
        self.reference = np.zeros((1024, 1024, 3), np.uint8)
        x0, y0, x1, y1 = box((0, 0), SPEC.origin)
        self.known[y0:y1, x0:x1] = self.seen[y0:y1, x0:x1] = self.valid[y0:y1, x0:x1] = True
        self.reference[y0:y1, x0:x1] = rgb(initial).copy()
        self.previous = rgb(initial).copy(); self.previous_q = (0, 0)
        self.event = self.submitted = self.new_planned_pixels = 0
        self.counts = {k: Accumulator() for k in ("overlap", "known", "history", "reentry", "final_target")}

    def submit(self, event, q, image):
        if type(event) is not int or event != self.event + 1 or event > 64:
            raise ValueError("Immutable LR64 planned event order required")
        p = coordinates(np.asarray(q)[None])[0]
        a = None if image is None else rgb(image).copy()
        x0, y0, x1, y1 = box(p, SPEC.origin)
        sl = np.s_[y0:y1, x0:x1]
        known, historic, new = self.known[sl], self.seen[sl] & ~self.known[sl], ~self.seen[sl]
        ref, valid = self.reference[sl], self.valid[sl]
        for key, mask in (("known", known), ("history", historic)):
            self.counts[key].add(int(mask.sum()), a[mask] if a is not None else None, ref[mask], valid[mask])
        pair = intersection(box(self.previous_q), box(p))
        if pair is not None:
            old, current = pair
            pixels = (current[0].stop-current[0].start)*(current[1].stop-current[1].start)
            self.counts["overlap"].add(pixels, a[current] if a is not None else None,
                                       self.previous[old] if self.previous is not None else None)
        for key, mask in (("reentry", self.labels["reentry_event"] == event),
                          ("final_target", np.ones((128, 128), bool) if event == 64 else np.zeros((128, 128), bool))):
            if mask.any():
                x, y = self.labels["x"][mask], self.labels["y"][mask]
                ix, iy = x - p[0] + 208, y - p[1] + 120
                wx, wy = x + 512, y + 512
                self.counts[key].add(int(mask.sum()), a[iy, ix] if a is not None else None,
                                     self.reference[wy, wx], self.valid[wy, wx])
        if a is not None:
            ref[new] = a[new]; valid[new] = True; self.submitted += 1
        self.seen[sl] = True; self.new_planned_pixels += int(new.sum())
        self.previous, self.previous_q, self.event = a, tuple(p), event

    def report(self):
        return {"planned_outputs": self.event, "valid_outputs": self.submitted,
                "output_completeness": self.submitted/self.event if self.event else None,
                "new_planned_pixels": self.new_planned_pixels,
                "metrics": {k: v.report() for k, v in self.counts.items()}}


def planned_report(task):
    ledger = Ledger(np.zeros((240, 416, 3), np.uint8), validate_task(task))
    for event, q in enumerate(task["q"][1:], 1):
        ledger.submit(event, q, None)
    return ledger.report()


def run_task(task, world, model, seed=0, out_dir=None, resources=None):
    labels = validate_task(task)
    rgb(world, (1024, 1024, 3))
    if (1024, 1024) not in getattr(model, "supported_world_hw", set()):
        raise ValueError("Adapter does not explicitly support native 1024 worlds")
    initial = crop(world, (0, 0))
    ledger = Ledger(initial, labels); journal = Journal(out_dir)
    journal.append("session", journal_schema="rswm-lr64-journal-v1", study=STUDY,
                   task_id=task["task_id"], task_sha256=digest(task), seed=int(seed),
                   initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                   world_spec=SPEC.public(), threat_model="cooperative_inprocess; not a security boundary")
    if journal.root:
        save_png_exclusive(journal.root/"initial.png",initial)
        save_bytes_exclusive(journal.root/"task.json",canonical(task)+b"\n")
    adapter = GuardedAdapter(model, journal, resources)
    adapter.reset(initial.copy(), SPEC.public(), seed)
    journal.append("initialize", stream="main")
    for lo in range(1, 65, 16):
        hi = lo + 16; q = np.array(task["q"][lo:hi], np.int64)
        journal.append("request", stream="main", start=lo, stop=hi, queries=q.tolist())
        error = None
        try:
            if resources:
                with resources.phase("generate", model, stream="main", count=16, model_call_attempted=adapter.ready):
                    output = adapter.generate(q.copy())
                with resources.phase("output_conversion"):
                    images, statuses = validate_response(output, 16)
            else:
                images, statuses = validate_response(adapter.generate(q.copy()), 16)
        except Exception as exc:
            images, statuses = [None]*16, ["model_exception"]*16; error = type(exc).__name__
        journal.append("response", stream="main", start=lo, stop=hi, error=error)
        for event, point, image, status in zip(range(lo, hi), q, images, statuses):
            ledger.submit(event, point, image); journal.image("main", event, image)
            journal.append("output", stream="main", event=event, q=point.tolist(), status=status,
                           pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest() if image is not None else None)
    result = {"schema": "rswm-lr64-run-v1", "study": STUDY, "task_id": task["task_id"],
              "seed": int(seed), "reports": ledger.report(), "journal_head": journal.head,
              "real_model_validation": False, "protocol_security_verified": False}
    journal.verify_persisted()
    if journal.root:
        save_bytes_exclusive(journal.root/"summary.json",canonical(result)+b"\n")
    return result, journal


def replay_run(task, initial, run_dir, expected_head=None):
    labels = validate_task(task); initial = rgb(initial).copy()
    root = Path(run_dir)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("Symlinked LR64 run path")
    rows = [load_json(line) for line in (root/"events.jsonl").read_text().splitlines()]
    if len(rows) > 200 or not all(isinstance(r, dict) for r in rows):
        raise ValueError("LR64 journal event count/type mismatch")
    head = verify_journal(rows)
    if expected_head is not None and head != expected_head:
        raise ValueError("LR64 journal head mismatch")
    if any(r.get("kind") not in {"session", "initialize", "adapter_failure", "request", "response", "output"} for r in rows):
        raise ValueError("Unknown LR64 journal event")
    rows = [r for r in rows if r.get("kind") != "adapter_failure"]
    if len(rows) != 74:
        raise ValueError("LR64 journal planned event count mismatch")
    header = rows[0]
    required = {"kind": "session", "journal_schema": "rswm-lr64-journal-v1", "study": STUDY,
                "task_id": task["task_id"], "task_sha256": digest(task), "world_spec": SPEC.public(),
                "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest()}
    if any(header.get(k) != v for k, v in required.items()) or type(header.get("seed")) is not int:
        raise ValueError("LR64 journal identity differs")
    if not np.array_equal(load_rgb_png(root/"initial.png"), initial):
        raise ValueError("LR64 retained initial pixels differ")
    if rows[1].get("kind") != "initialize" or rows[1].get("stream") != "main":
        raise ValueError("LR64 initialization missing")
    ledger = Ledger(initial, labels); offset = 2; logged = {"initial.png"}
    for lo in range(1, 65, 16):
        hi = lo + 16
        request, response = rows[offset:offset+2]; offset += 2
        want = {"kind": "request", "stream": "main", "start": lo, "stop": hi, "queries": task["q"][lo:hi]}
        if any(request.get(k) != v for k, v in want.items()) or any(response.get(k) != v for k, v in {"kind": "response", "stream": "main", "start": lo, "stop": hi}.items()):
            raise ValueError("LR64 request order differs")
        for event in range(lo, hi):
            row = rows[offset]; offset += 1
            if any(row.get(k) != v for k, v in {"kind": "output", "stream": "main", "event": event, "q": task["q"][event]}.items()):
                raise ValueError("LR64 output order differs")
            path = root/"main"/f"{event:04d}.png"
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("LR64 PNG symlink")
            if row.get("status") == "valid":
                image = load_rgb_png(path)
                if hashlib.sha256(image.tobytes()).hexdigest() != row.get("pixel_sha256"):
                    raise ValueError("LR64 pixel hash differs")
                logged.add(path.relative_to(root).as_posix())
            elif row.get("status") in {"missing", "invalid_image", "invalid_batch", "model_exception"}:
                image = None
                if path.exists() or row.get("pixel_sha256") is not None:
                    raise ValueError("Missing LR64 output was repaired")
            else:
                raise ValueError("Invalid LR64 output status")
            ledger.submit(event, row["q"], image)
    if {p.relative_to(root).as_posix() for p in root.rglob("*.png")} - logged:
        raise ValueError("Unlogged LR64 PNG")
    report = ledger.report()
    if (root/"summary.json").exists() and _json(root/"summary.json").get("reports") != report:
        raise ValueError("LR64 summary disagrees with replay")
    return {"task_id": task["task_id"], "study": STUDY, "seed": header["seed"], "reports": report,
            "journal_head": head, "planned_output_events": 64, "verified_png_outputs": ledger.submitted,
            "real_model_validation": False, "protocol_security_verified": False}


def _config(config):
    allowed = {"model_id", "adapter", "device", "model_dir", "steps", "prompt", "diagnostic_noise", "policy"}
    if not isinstance(config, dict) or set(config) - allowed or not isinstance(config.get("model_id"), str) or not config["model_id"]:
        raise ValueError("Invalid LR64 model configuration")
    if config.get("adapter") not in {"copy", "random", "native-noise-map", "oracle", "history-completer"}:
        raise ValueError("Adapter has no native LR64 capability")
    if config.get("device", "cpu") != "cpu" and not re.fullmatch(r"cuda(?::[0-9]+)?", str(config["device"])):
        raise ValueError("Invalid LR64 device")
    if (config["adapter"] != "history-completer" or config.get("diagnostic_noise") is True) and config.get("device", "cpu") != "cpu":
        raise ValueError("CPU native diagnostic adapters cannot claim CUDA execution")
    if config["adapter"] != "history-completer" and set(config) - {"model_id", "adapter", "device"}:
        raise ValueError("Unused LR64 adapter configuration")
    if config["adapter"] == "history-completer":
        from .history_mechanisms import validate_policy
        validate_policy(config.get("policy"))
        if config.get("diagnostic_noise") is not True and not config.get("model_dir"):
            raise ValueError("Frozen local completer weights required")
        if config.get("diagnostic_noise") and config.get("model_dir"):
            raise ValueError("Noise and local weights are mutually exclusive")
        if config.get("diagnostic_noise") and ("steps" in config or "prompt" in config):
            raise ValueError("Diagnostic noise ignores steps/prompt")
        if "diagnostic_noise" in config and type(config["diagnostic_noise"]) is not bool:
            raise ValueError("diagnostic_noise must be boolean")
        if "steps" in config and (type(config["steps"]) is not int or config["steps"] < 1):
            raise ValueError("steps must be a positive integer")
        if "prompt" in config and not isinstance(config["prompt"], str):
            raise ValueError("prompt must be text")
        if config.get("model_dir"):
            root = Path(config["model_dir"])
            if not root.is_dir() or root.is_symlink() or not any(p.is_file() for p in root.rglob("*")) or any(p.is_symlink() for p in root.rglob("*")):
                raise ValueError("Nonempty local weights without symlinks required")
    return config


class _NativeCopy:
    supported_world_hw = {(1024, 1024)}
    def reset(self, initial, spec, seed):
        self.initial = rgb(initial).copy(); self.rng = np.random.default_rng(seed)
    def generate(self, queries):
        return [self.initial.copy() for _ in coordinates(queries)]
    def resource_state(self):
        return {"initial": self.initial}


class _NativeRandom(_NativeCopy):
    def generate(self, queries):
        return [self.rng.integers(0, 256, (240, 416, 3), dtype=np.uint8) for _ in coordinates(queries)]


class _NativeNoise(_NativeCopy):
    def reset(self, initial, spec, seed):
        super().reset(initial, spec, seed)
        self.map = self.rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
        self.map[392:632, 304:720] = self.initial
    def generate(self, queries):
        return [crop(self.map, q) for q in coordinates(queries)]
    def resource_state(self):
        return {"map": self.map, "initial": self.initial}


class _NativeOracle(_NativeCopy):
    def __init__(self, world):
        self.world = rgb(world, (1024, 1024, 3))
    def generate(self, queries):
        return [crop(self.world, q) for q in coordinates(queries)]


def model_factory(config, world, task):
    _config(config)
    kind = config["adapter"]
    if kind == "oracle":
        return _NativeOracle(world)
    if kind != "history-completer":
        return {"copy": _NativeCopy, "random": _NativeRandom, "native-noise-map": _NativeNoise}[kind]()
    from .history_mechanisms import HistoryCompleterAdapter
    from .baselines import NoiseCompleter, DiffusersCompleter
    completer = NoiseCompleter() if config.get("diagnostic_noise") else DiffusersCompleter(
        config["model_dir"], config.get("device", "cuda"), config.get("steps", 30),
        config.get("prompt", "nadir orthorectified RGB aerial photograph"))
    return HistoryCompleterAdapter(completer, config["policy"])


def _world(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("World symlink forbidden")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Lossless native RGB PNG required")
        return rgb(np.asarray(image).copy(), (1024, 1024, 3))


def build_plan(tasks, models, seeds, source_manifest_sha256, code_commit, retry_policy=None):
    if not tasks or not models or not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Complete nonempty LR64 product required")
    if not re.fullmatch(r"[a-f0-9]{64}", source_manifest_sha256) or not re.fullmatch(r"[a-f0-9]{40}", code_commit):
        raise ValueError("Frozen source/code identities required")
    frozen, locks, rows = {}, {}, []
    for config in models:
        _config(config)
        if config["model_id"] in locks:
            raise ValueError("Duplicate LR64 model identity")
        locks[config["model_id"]] = model_manifest(config)
    for entry in tasks:
        if any(Path(entry[k]).is_symlink() or any(p.is_symlink() for p in Path(entry[k]).parents) for k in ("task_path", "world_path")):
            raise ValueError("LR64 input symlink")
        path = Path(entry["task_path"]).resolve()
        task = _json(path); validate_task(task)
        if task["task_id"] in frozen:
            raise ValueError("Duplicate LR64 task")
        frozen[task["task_id"]] = task
        world_path = Path(entry["world_path"]).resolve(); world = _world(world_path)
        for mid, model in sorted(locks.items()):
            for seed in sorted(seeds):
                row = {"study": STUDY, "task_id": task["task_id"], "world_id": task["world_id"], "pair_id": task["pair_id"],
                       "role": task.get("role", "supplemental_unvalidated"),
                       "variant": task["variant"], "model_id": mid, "seed": seed, "protocol": "online16", "family": "LR64",
                       "task_path": str(path), "world_path": str(world_path), "task_sha256": digest(task),
                       "world_file_sha256": file_hash(world_path), "world_pixel_sha256": hashlib.sha256(world.tobytes()).hexdigest(),
                       "initial_sha256": hashlib.sha256(crop(world, (0, 0)).tobytes()).hexdigest(),
                       "source_manifest_sha256": source_manifest_sha256, "model_manifest_sha256": digest(model),
                       "metric_config_sha256": digest(METRIC_CONFIG), "code_commit": code_commit, "planned_reports": planned_report(task)}
                row["run_id"] = digest({k: row[k] for k in ("study", "task_id", "model_manifest_sha256", "seed", "metric_config_sha256")})
                rows.append(row)
    plan = {"schema": PLAN_SCHEMA, "study": STUDY, "world_spec": SPEC.public(), "tasks": frozen, "models": locks,
            "expected": rows, "source_code_sha256": code_fingerprint(), "metric_config": METRIC_CONFIG,
            "retry_policy": _policy(retry_policy or DEFAULT_RETRY_POLICY)}
    plan["plan_sha256"] = digest(plan); validate_plan(plan)
    return plan


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA or plan.get("study") != STUDY:
        raise ValueError("Independent LR64 plan required")
    if digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan.get("plan_sha256") or plan.get("source_code_sha256") != code_fingerprint():
        raise ValueError("LR64 frozen plan/code hash mismatch")
    if plan.get("world_spec") != SPEC.public() or plan.get("metric_config") != METRIC_CONFIG:
        raise ValueError("LR64 world/metric protocol mismatch")
    _policy(plan.get("retry_policy"))
    pairs = {}; tasks = plan["tasks"]; models = plan["models"]
    for tid, task in tasks.items():
        validate_task(task)
        if task["task_id"] != tid or task["variant"] in pairs.setdefault(task["pair_id"], {}):
            raise ValueError("LR64 duplicate task/pair variant")
        pairs[task["pair_id"]][task["variant"]] = task
    for variants in pairs.values():
        if set(variants) != {"exact", "offset"} or variants["exact"]["world_id"] != variants["offset"]["world_id"] or variants["exact"].get("role") != variants["offset"].get("role") or variants["exact"]["q"][:49] != variants["offset"]["q"][:49]:
            raise ValueError("LR64 same-world first48 paired population differs")
    for mid, lock in models.items():
        _config(lock["config"])
        if lock["config"]["model_id"] != mid:
            raise ValueError("LR64 model identity mismatch")
    seen = set(); world_bindings = {}; identities = set(); reports = {tid: planned_report(t) for tid, t in tasks.items()}
    for row in plan["expected"]:
        task = tasks[row["task_id"]]; model = models[row["model_id"]]
        if any(row.get(k) != task[k] for k in ("task_id", "world_id", "pair_id", "variant", "family", "protocol", "study")) or row["task_sha256"] != digest(task) or row["model_manifest_sha256"] != digest(model) or row["metric_config_sha256"] != digest(METRIC_CONFIG):
            raise ValueError("LR64 expected identity mismatch")
        if row.get("role") != task.get("role", "supplemental_unvalidated"):
            raise ValueError("LR64 expected role differs")
        for key in ("source_manifest_sha256", "world_file_sha256", "world_pixel_sha256", "initial_sha256"):
            if not isinstance(row.get(key), str) or not re.fullmatch(r"[a-f0-9]{64}", row[key]):
                raise ValueError("LR64 expected hash malformed")
        if not isinstance(row.get("code_commit"), str) or not re.fullmatch(r"[a-f0-9]{40}", row["code_commit"]):
            raise ValueError("LR64 code identity malformed")
        if type(row["seed"]) is not int or row["seed"] < 0 or row["planned_reports"] != reports[row["task_id"]]:
            raise ValueError("LR64 denominator/seed mismatch")
        if row["run_id"] != digest({k: row[k] for k in ("study", "task_id", "model_manifest_sha256", "seed", "metric_config_sha256")}):
            raise ValueError("LR64 run identity mismatch")
        key = (row["task_id"], row["model_id"], row["seed"])
        if key in seen:
            raise ValueError("LR64 duplicate expected sample")
        seen.add(key)
        binding = tuple(row[k] for k in ("world_file_sha256", "world_pixel_sha256", "initial_sha256"))
        if row["world_id"] in world_bindings and world_bindings[row["world_id"]] != binding:
            raise ValueError("LR64 paired world/I0 bytes differ")
        world_bindings[row["world_id"]] = binding
        identities.add((row["source_manifest_sha256"], row["code_commit"]))
    seeds = {r["seed"] for r in plan["expected"]}
    if len(identities) != 1 or not seen or seen != {(tid, mid, s) for tid in tasks for mid in models for s in seeds}:
        raise ValueError("Incomplete LR64 expected product")


def _read_attempt(plan, root, row, folder):
    attempt = int(folder.name.split("-")[-1])
    result = {"run_id": row["run_id"], "attempt": attempt, "status": "METADATA_CORRUPT", "replay_verified": False,
              "reports": row["planned_reports"], "resources": None, "resource_status": "NOT_MEASURED", "integrity_errors": []}
    try:
        outcome = _metadata(folder/"outcome.json", row, plan, attempt, "outcome", folder)
        manifest = _metadata(folder/"artifact-manifest.json", row, plan, attempt, "manifest")
        if manifest.get("schema") != "rswm-attempt-artifacts-v1":
            raise ValueError("LR64 artifact manifest schema mismatch")
        if any(name not in {"started.json", "outcome.json", "resources.json", "replayed.json"} or (name != "resources.json" and file_hash(folder/name) != checksum) for name, checksum in manifest["files"].items()) or not {"started.json", "outcome.json"}.issubset(manifest["files"]):
            raise ValueError("LR64 attempt artifacts changed")
        _metadata(folder/"started.json", row, plan, attempt, "started")
        receipt = _metadata(root/"receipts"/f"{row['run_id']}-{attempt:03d}.json", row, plan, attempt, "receipt")
        if receipt.get("outcome_sha256") != file_hash(folder/"outcome.json"):
            raise ValueError("LR64 outcome receipt mismatch")
        result.update(outcome)
        if outcome["status"] == "COMPLETED" and (not outcome["replay_verified"] or outcome["reports"].get("valid_outputs") != 64):
            raise ValueError("LR64 completed outcome lacks complete replay")
        if not outcome["replay_verified"] and outcome["reports"] != row["planned_reports"]:
            raise ValueError("LR64 unverified outcome changed frozen denominators")
        if outcome["replay_verified"]:
            head = _metadata(root/"heads"/f"{row['run_id']}-{attempt:03d}.json", row, plan, attempt, "head")
            if head.get("journal_head") != outcome.get("replay_head"):
                raise ValueError("LR64 replay head identity mismatch")
            task = plan["tasks"][row["task_id"]]
            initial = load_rgb_png(folder/"submission"/"initial.png")
            if hashlib.sha256(initial.tobytes()).hexdigest() != row["initial_sha256"]:
                raise ValueError("LR64 initial pixels differ")
            replayed = replay_run(task, initial, folder/"submission", head["journal_head"])
            if replayed["seed"] != row["seed"]:
                raise ValueError("LR64 replay seed differs from expected sample")
            result["reports"] = replayed["reports"]
            if result["reports"] != outcome["reports"]:
                raise ValueError("LR64 outcome disagrees with independent replay")
        try:
            resource = _metadata(folder/"resources.json", row, {**plan, "schema": CORE_SUITE_SCHEMA}, attempt, "resource")
            if manifest["files"].get("resources.json") != file_hash(folder/"resources.json"):
                raise ValueError("LR64 resource manifest mismatch")
            result["resources"] = resource
            result["resource_status"] = "COMPLETE" if resource["measurement_complete"] else "INCOMPLETE"
        except (ValueError, OSError, KeyError, TypeError) as exc:
            result["resource_status"] = "METADATA_CORRUPT"
            result["integrity_errors"].append({"object": "resources", "error_type": type(exc).__name__})
    except (ValueError, OSError, KeyError, TypeError) as exc:
        result.update(status="METADATA_CORRUPT", replay_verified=False, reports=row["planned_reports"], error_type=type(exc).__name__)
        result["integrity_errors"].append({"object": "attempt", "error_type": type(exc).__name__})
    return result


def _attempt(plan, root, row, attempt, factory):
    folder = root/"runs"/row["run_id"]/f"attempt-{attempt:03d}"
    if folder.is_symlink() or any(p.is_symlink() for p in folder.parents):
        raise ValueError("LR64 attempt path escape")
    folder.mkdir(parents=True, exist_ok=False)
    identity = {"run_id": row["run_id"], "attempt": attempt, "plan_sha256": plan["plan_sha256"]}
    _save(folder/"started.json", {**identity, "status": "RUNNING"}, exclusive=True)
    outcome = {**identity, "status": "FAILED_INPUT", "run_dir": str(folder/"submission"), "resource_path": str(folder/"resources.json"),
               "reports": row["planned_reports"], "replay_verified": False, "replay_head": None, "error_type": None}
    config = plan["models"][row["model_id"]]["config"]
    collector = ResourceCollector(config.get("device", "cpu")); collector.calls["planned_output_slots"] = 64
    journal = None
    try:
        with collector.phase("input_validation"):
            task = _json(row["task_path"]); validate_task(task); world = _world(row["world_path"])
            if digest(task) != row["task_sha256"] or file_hash(row["world_path"]) != row["world_file_sha256"] or hashlib.sha256(world.tobytes()).hexdigest() != row["world_pixel_sha256"] or digest(model_manifest(config)) != row["model_manifest_sha256"]:
                raise ValueError("Frozen LR64 source/model changed")
        outcome["status"] = "FAILED_LOAD"
        with collector.phase("model_load"):
            model = factory(config, world, task)
        outcome["status"] = "FAILED_GENERATE"
        result, journal = run_task(task, world, model, row["seed"], folder/"submission", collector)
        _save(root/"heads"/f"{row['run_id']}-{attempt:03d}.json", {**identity, "journal_head": journal.head}, exclusive=True)
        with collector.phase("replay"):
            replayed = replay_run(task, crop(world, (0, 0)), folder/"submission", journal.head)
        _save(folder/"replayed.json", replayed, exclusive=True)
        outcome.update(status="COMPLETED" if replayed["reports"]["valid_outputs"] == 64 else "INCOMPLETE",
                       reports=replayed["reports"], replay_verified=True, replay_head=journal.head)
    except Exception as exc:
        outcome["error_type"] = type(exc).__name__
    except BaseException as exc:
        outcome.update(status="INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "CRASHED", error_type=type(exc).__name__)
        raise
    finally:
        observed = journal.rows if journal else []
        if journal is None and (folder/"submission"/"events.jsonl").exists():
            try:
                observed = [load_json(line) for line in (folder/"submission"/"events.jsonl").read_text().splitlines()]
            except (ValueError, OSError):
                observed = None
        collector.calls["valid_outputs"] = sum(r.get("kind") == "output" and r.get("status") == "valid" for r in observed) if observed is not None else None
        collector.execution_complete = outcome["status"] == "COMPLETED"
        resource = collector.finish(folder); resource.update(identity)
        _save(folder/"resources.json", resource, exclusive=True)
        outcome["resource_status"] = "COMPLETE" if resource["measurement_complete"] else "INCOMPLETE"
        _save(folder/"outcome.json", outcome, exclusive=True)
        _save(root/"receipts"/f"{row['run_id']}-{attempt:03d}.json", {**identity, "status": outcome["status"], "outcome_sha256": file_hash(folder/"outcome.json")}, exclusive=True)
        _save(folder/"artifact-manifest.json", {**identity, "schema": "rswm-attempt-artifacts-v1", "files": {
            name: file_hash(folder/name) for name in ("started.json", "outcome.json", "resources.json", "replayed.json") if (folder/name).is_file()}}, exclusive=True)
    return outcome


def _recover_crash(plan, root, row, folder):
    """A new attempt may recover a process crash; its old cost stays unknown."""
    if (folder/"outcome.json").exists():
        return
    attempt = int(folder.name.split("-")[-1])
    _metadata(folder/"started.json", row, plan, attempt, "started")
    identity = {"run_id": row["run_id"], "attempt": attempt, "plan_sha256": plan["plan_sha256"]}
    outcome = {**identity, "status": "CRASHED", "run_dir": str(folder/"submission"),
               "resource_path": str(folder/"resources.json"), "reports": row["planned_reports"],
               "replay_verified": False, "replay_head": None,
               "error_type": "ProcessTerminatedWithoutOutcome", "resource_status": "NOT_MEASURED"}
    _save(folder/"outcome.json", outcome, exclusive=True)
    _save(root/"receipts"/f"{row['run_id']}-{attempt:03d}.json", {**identity, "status": "CRASHED",
          "outcome_sha256": file_hash(folder/"outcome.json")}, exclusive=True)
    _save(folder/"artifact-manifest.json", {**identity, "schema": "rswm-attempt-artifacts-v1", "files": {
        name: file_hash(folder/name) for name in ("started.json", "outcome.json", "resources.json") if (folder/name).is_file()}}, exclusive=True)


def execute_plan(plan, out, resume=False, retry_failed=False, factory=None):
    validate_plan(plan)
    if retry_failed and not resume:
        raise ValueError("LR64 retries require explicit resume")
    root = Path(out)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("LR64 root path escape")
    root = root.resolve(); root.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = root/"coordinator.lock"
    if lock.is_symlink():
        raise ValueError("LR64 lock symlink")
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        existing = [p for p in root.iterdir() if p.name != "coordinator.lock"]
        if existing:
            if not resume or _json(root/"plan.json") != plan:
                raise ValueError("Explicit same-plan resume required")
        else:
            if resume:
                raise ValueError("Cannot resume an uninitialized LR64 suite")
            _save(root/"plan.json", plan, exclusive=True)
        outcomes = []
        for row in plan["expected"]:
            folders = _attempt_folders(root/"runs"/row["run_id"])
            attempt = 0
            if folders:
                _recover_crash(plan, root, row, folders[-1])
                latest = _read_attempt(plan, root, row, folders[-1]); attempt = latest["attempt"] + 1
                if not retry_failed or latest["status"] == "COMPLETED":
                    outcomes.append(latest); continue
                if latest["status"] not in plan["retry_policy"]["allowed_retry_statuses"] or attempt >= plan["retry_policy"]["max_attempts_per_run"]:
                    raise ValueError("LR64 retry status/budget disallows attempt")
            outcomes.append(_attempt(plan, root, row, attempt, factory or model_factory))
        return outcomes


def replay_suite(plan, root, selection="first_attempt_only"):
    validate_plan(plan)
    if selection not in {"first_attempt_only", "latest_attempt_within_budget"}:
        raise ValueError("Unsupported LR64 attempt selection")
    root = Path(root)
    records, all_attempts = [], []
    with suite_read_lock(root):
        if _json(root/"plan.json") != plan:
            raise ValueError("LR64 root plan identity differs")
        for row in plan["expected"]:
            missing = {"run_id": row["run_id"], "attempt": None, "status": "NOT_RUN", "reports": row["planned_reports"],
                       "replay_verified": False, "resources": None, "resource_status": "NOT_MEASURED", "integrity_errors": []}
            try:
                folders = _attempt_folders(root/"runs"/row["run_id"])
                attempts = [_read_attempt(plan, root, row, f) for f in folders]
                all_attempts.extend(attempts)
                if any(a["attempt"] >= plan["retry_policy"]["max_attempts_per_run"] for a in attempts):
                    missing.update(status="METADATA_CORRUPT", integrity_errors=[{"object": "attempt", "code": "RETRY_BUDGET_EXCEEDED"}])
                else:
                    selected = [a for a in attempts if a["attempt"] == 0] if selection == "first_attempt_only" else attempts
                    if selected:
                        missing = selected[-1]
                    elif attempts:
                        missing["integrity_errors"] = [{"object": "attempt", "code": "FIRST_ATTEMPT_MISSING"}]
            except (ValueError, OSError, KeyError, TypeError) as exc:
                missing.update(status="METADATA_CORRUPT", error_type=type(exc).__name__)
            records.append(missing)
    return records, all_attempts


def aggregate(plan, root, bootstrap=2000, seed=0):
    with suite_read_lock(root):
        return _aggregate(plan, root, bootstrap, seed)


def _aggregate(plan, root, bootstrap, seed):
    if type(bootstrap) is not int or bootstrap < 1:
        raise ValueError("Positive whole-world bootstrap required")
    analyses = {}; attempts = []
    for selection in ("first_attempt_only", "latest_attempt_within_budget"):
        records, attempts = replay_suite(plan, root, selection)
        by_id = {r["run_id"]: r for r in records}; summaries = []
        for mid, role in sorted({(r["model_id"], r["role"]) for r in plan["expected"]}):
            selected = [r for r in plan["expected"] if r["model_id"] == mid and r["role"] == role]
            for metric in ("overlap", "known", "history", "reentry", "final_target"):
                world_values = []
                for wid in sorted({r["world_id"] for r in selected}):
                    rows = [r for r in selected if r["world_id"] == wid]
                    values = [by_id[r["run_id"]]["reports"]["metrics"][metric] for r in rows]
                    complete = [v["mae"] for v in values if v["coverage"] == 1.]
                    world_values.append(float(np.mean(complete)) if len(complete) == len(values) else None)
                rng = np.random.default_rng(seed); replicas = []
                if all(v is not None for v in world_values):
                    for _ in range(bootstrap):
                        picked = [world_values[i] for i in rng.integers(len(world_values), size=len(world_values))]
                        replicas.append(float(np.mean(picked)))
                summaries.append({"model_id": mid, "role": role, "metric": metric, "planned_worlds": len(world_values),
                                  "complete_worlds": sum(v is not None for v in world_values),
                                  "mean": float(np.mean(world_values)) if all(v is not None for v in world_values) else None,
                                  "whole_world_bootstrap_95_interval": np.quantile(replicas, [.025, .975]).tolist() if replicas else None,
                                  "valid_bootstrap_replicates": len(replicas), "bootstrap_replicates": bootstrap})
        analyses[selection] = {"records": records, "summaries": summaries, "expected_runs": len(records),
                               "executed_runs": sum(r["attempt"] is not None for r in records),
                               "completed_runs": sum(r["status"] == "COMPLETED" for r in records),
                               "failed_or_missing_runs": sum(r["status"] != "COMPLETED" for r in records),
                               "complete_resource_runs": sum(r["resource_status"] == "COMPLETE" for r in records),
                               "integrity_error_runs": sum(bool(r.get("integrity_errors")) for r in records),
                               "planned_outputs": len(records)*64, "valid_outputs": sum(r["reports"]["valid_outputs"] for r in records),
                               "replay_verified_runs": sum(r["replay_verified"] for r in records)}
    from .aggregate import resource_totals
    resources = [{**{k: row[k] for k in ("model_id", "world_id", "pair_id", "variant", "seed")}, **a}
                 for row in plan["expected"] for a in attempts if a["run_id"] == row["run_id"]]
    return {"schema": "rswm-lr64-analysis-v1", "study": STUDY, "plan_sha256": plan["plan_sha256"],
            "analyses": analyses, "all_attempt_resources": [{"run_id": a["run_id"], "attempt": a["attempt"], "status": a["status"],
                                                               "resource_status": a["resource_status"], "resources": a["resources"],
                                                               **{k: a[k] for k in ("model_id", "world_id", "pair_id", "variant", "seed")}} for a in resources],
            "resource_totals": resource_totals(resources, plan),
            "acceptance": _acceptance(plan, root, analyses["first_attempt_only"]["records"], attempts),
            "statistical_scope": "whole-world resampling retains exact/offset and seed dependence; geography unknown; separate from Inria",
            "real_data_validation": "NOT_RUN", "real_model_validation": "NOT_RUN"}


def _acceptance(plan, root, records, attempts):
    expected = {r["run_id"] for r in plan["expected"]}
    runs = Path(root)/"runs"
    unexpected = sorted(p.name for p in runs.iterdir() if p.name not in expected or p.is_symlink() or not p.is_dir()) if runs.exists() else []
    good = all(r["status"] == "COMPLETED" and r["replay_verified"] and r["resource_status"] == "COMPLETE" and not r.get("integrity_errors") for r in records)
    costs = all(a["resource_status"] == "COMPLETE" and not a.get("integrity_errors") for a in attempts)
    return {"accepted": good and costs and not unexpected,
            "scope": "first attempt RGB and metadata; all started attempt resource cost",
            "expected_runs": len(expected), "first_attempt_verified_runs": sum(r["replay_verified"] for r in records),
            "started_attempts": len(attempts), "complete_resource_attempts": sum(a["resource_status"] == "COMPLETE" for a in attempts),
            "unexpected_runs": unexpected}


def plan_from_materialized(root, models, seeds, code_commit, retry_policy=None):
    """Consume the pinned materializer tree without decoding/resizing image2."""
    root = Path(root); selection = _json(root/"selection.json")
    if selection.get("study") != STUDY or selection.get("repo_id") != "blanchon/LEVIR_CDPlus" or selection.get("revision") != "8062826f12e4f3cd5690b84278ad67c1f42acc46":
        raise ValueError("Pinned LEVIR selection identity mismatch")
    from .hf_materialize import SHARDS, choose_levir_rows
    counts = selection.get("row_counts", {})
    expected_shards = [{"path": p, "size_bytes": b, "sha256": h} for p, b, h in SHARDS]
    if selection.get("shards") != expected_shards or set(counts) != {s[0] for s in SHARDS} or any(type(c) is not int or c < 0 for c in counts.values()) or sum(counts.values()) != 348:
        raise ValueError("Pinned LEVIR source shard/count lock mismatch")
    selected = [{"rank": k, "hash": h, "shard": p, "row_index": i}
                for k, (h, p, i) in enumerate(choose_levir_rows(counts), 1)]
    if selection.get("selected_rows") != selected:
        raise ValueError("LEVIR rows must be selected by the frozen pre-pixel hash")
    worlds = _json(root/"private"/"world-index.json")
    if len(worlds) != 100 or len({w["world_id"] for w in worlds}) != 100:
        raise ValueError("LEVIR100 requires all 100 frozen worlds")
    rows = {(r["shard"], r["row_index"]): r for r in selected}
    if {(w["shard"], w["row_index"]) for w in worlds} != set(rows) or len({w["world_pixel_sha256"] for w in worlds}) != 100:
        raise ValueError("LEVIR worlds do not match all 100 unique selected source rows")
    tasks = []
    for world in worlds:
        wid = "levir-" + digest([selection["repo_id"], selection["revision"], world["shard"], world["row_index"]])[:16]
        if world["world_id"] != wid or world["selection_rank"] != rows[(world["shard"], world["row_index"])]["rank"] or _json(root/"private"/"worlds"/(wid + ".json")) != world:
            raise ValueError("LEVIR world/source provenance mismatch")
        world_path = root/"private"/"worlds"/(wid + ".png")
        pixels = _world(world_path)
        if hashlib.sha256(pixels.tobytes()).hexdigest() != world["world_pixel_sha256"] or world.get("shape") != [1024, 1024, 3] or world.get("resampling") != "none":
            raise ValueError("Native LEVIR world identity mismatch")
        iid = digest([wid, "initial"])[:24]
        if not np.array_equal(load_rgb_png(root/"public"/"initial"/(iid + ".png")), crop(pixels, (0, 0))):
            raise ValueError("Native LEVIR public I0 differs")
        for variant in ("exact", "offset"):
            task_path = root/"private"/"tasks"/(wid + "-LR64-" + variant + ".json")
            task = _json(task_path); labels_path = root/"private"/"labels"/(task["task_id"] + ".npz")
            if task.get("label_file") != "../labels/" + labels_path.name or task.get("label_sha256") != file_hash(labels_path):
                raise ValueError("LR64 label path/hash mismatch")
            with np.load(labels_path, allow_pickle=False) as labels:
                validate_task(task, {k: labels[k] for k in labels.files})
            sid = digest([task["task_id"], "public"])[:24]
            public = _json(root/"public"/"sessions"/(sid + ".json"))
            expected_public = {"session_id": sid, "initial_image": "../initial/" + iid + ".png",
                               "protocol": "online16", "viewport_hw": [240, 416], "bounds_xy": [[-288, 288]]*2,
                               "coordinate_normalizer": float(np.hypot(416, 240)), "q0": [0, 0],
                               "query_delivery": "current 16 queries only; reset between exact/offset"}
            if public != expected_public:
                raise ValueError("LEVIR public session contract differs")
            tasks.append({"task_path": task_path, "world_path": world_path})
    return build_plan(tasks, models, seeds, file_hash(root/"selection.json"), code_commit, retry_policy)


def plan_from_index(path, models, seeds, code_commit, retry_policy=None):
    """Small diagnostic pilots use an explicit index; they never claim LEVIR100."""
    path = Path(path); index = _json(path)
    if index.get("schema") != "rswm-lr64-diagnostic-index-v1" or index.get("study") != STUDY or index.get("diagnostic_only") is not True:
        raise ValueError("Explicit independent LR64 diagnostic index required")
    tasks = []
    for row in index["tasks"]:
        entry = {}
        for key in ("task_path", "world_path"):
            relative = Path(row[key])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("LR64 diagnostic index paths must stay inside its directory")
            entry[key] = path.parent/relative
        if _json(entry["task_path"]).get("role") != "diagnostic_only":
            raise ValueError("Diagnostic pilot cannot claim a real-data role")
        tasks.append(entry)
    return build_plan(tasks, models, seeds, index["source_manifest_sha256"], code_commit, retry_policy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--materialized", type=Path); inputs.add_argument("--task-index", type=Path)
    p.add_argument("--models", type=Path, required=True); p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--code-commit", required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-attempts-per-run", type=int, default=1)
    for name in ("run", "verify", "aggregate"):
        p = sub.add_parser(name); p.add_argument("--plan", type=Path, required=True); p.add_argument("--root", type=Path, required=True)
        if name == "run":
            p.add_argument("--resume", action="store_true"); p.add_argument("--retry-failed", action="store_true")
        if name == "aggregate":
            p.add_argument("--out", type=Path, required=True); p.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            policy = {**DEFAULT_RETRY_POLICY, "max_attempts_per_run": args.max_attempts_per_run}
            builder, source = (plan_from_materialized, args.materialized) if args.materialized else (plan_from_index, args.task_index)
            _save(args.out, builder(source, _json(args.models), args.seeds, args.code_commit, policy), exclusive=True)
            return 0
        plan = _json(args.plan)
        if args.command == "run":
            execute_plan(plan, args.root, args.resume, args.retry_failed)
            rows, attempts = replay_suite(plan, args.root)
        elif args.command == "verify":
            rows, attempts = replay_suite(plan, args.root)
        else:
            result = aggregate(plan, args.root, args.bootstrap); _save(args.out, result, exclusive=True)
            return 0 if result["acceptance"]["accepted"] else 1
        return 0 if _acceptance(plan, args.root, rows, attempts)["accepted"] else 1
    except (ValueError, OSError, KeyError, TypeError, RuntimeError):
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
