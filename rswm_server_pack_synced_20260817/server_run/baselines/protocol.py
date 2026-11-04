#!/usr/bin/env python3
"""Shared protocol helpers for the RS-Roll baseline platform.

This module is intentionally model-free.  It defines only auditable file/tensor
contracts, checksums, overlap summaries, and empirically derived difficulty bins.
The Wan/LingBot backbone and camera-conditioning implementation are not imported.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MANIFEST_SCHEMA = "rsroll.benchmark_manifest/v2"
RESULT_SCHEMA = "rsroll.baseline_result/v2"
CARD_SCHEMA = "rsroll.baseline_card/v1"
CONDITION_FRAMES = 1
SUPPORTED_FRAME_COUNTS = (17, 25, 33)


class ProtocolError(ValueError):
    """Raised when an artifact would make methods incomparable."""


def sha256_file(path: str | Path, chunk_bytes: int = 4 << 20) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(parts: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.hexdigest()


def sha256_array(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return sha256_bytes((str(arr.dtype).encode(), str(arr.shape).encode(), arr.tobytes()))


def stable_json_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalize_video(value: Any, *, name: str = "video", expected_t: int | None = None,
                    allow_float: bool = True) -> np.ndarray:
    """Normalize a rollout to canonical ``(T,H,W,C) uint8``.

    Both ``[T,C,H,W]`` and ``[T,H,W,C]`` are accepted at the adapter boundary.
    A single image is deliberately rejected: image-output methods must make their
    recursive/per-waypoint adaptation explicit instead of being presented as native
    video generators.
    """
    arr = np.asarray(value)
    if arr.ndim != 4:
        raise ProtocolError(f"{name}: expected a 4-D rollout, got shape={arr.shape}")
    if arr.shape[-1] == 3:
        out = arr
    elif arr.shape[1] == 3:
        out = np.transpose(arr, (0, 2, 3, 1))
    else:
        raise ProtocolError(
            f"{name}: expected [T,H,W,3] or [T,3,H,W], got shape={arr.shape}")
    if len(out) < 2:
        raise ProtocolError(f"{name}: rollout must include condition + future frames")
    if expected_t is not None and len(out) != int(expected_t):
        raise ProtocolError(f"{name}: T={len(out)} != expected T={expected_t}")
    if not np.all(np.isfinite(out)):
        raise ProtocolError(f"{name}: contains NaN/Inf")
    if out.dtype != np.uint8:
        if not allow_float:
            raise ProtocolError(f"{name}: expected uint8, got {out.dtype}")
        if np.issubdtype(out.dtype, np.floating) and out.size:
            lo, hi = float(out.min()), float(out.max())
            # Test [0,1] first: it is a strict subset of [-1,1].  Reversing these
            # branches maps black (0.0) to mid-gray (127.5), silently corrupting
            # every float image produced in the common [0,1] convention.
            if 0.0 <= lo and hi <= 1.01:
                out = np.clip(out, 0.0, 1.0) * 255.0
            elif -1.01 <= lo and hi <= 1.01:
                out = (np.clip(out, -1.0, 1.0) + 1.0) * 127.5
        out = np.clip(out, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def validate_overlap_profile(profile: Mapping[str, Any], frame_count: int) -> list[float]:
    expected = [f"frame_{i}" for i in range(frame_count)]
    missing = [k for k in expected if k not in profile]
    extra = sorted(set(profile) - set(expected))
    if missing or extra:
        raise ProtocolError(
            f"overlap_profile keys mismatch: missing={missing[:3]} extra={extra[:3]}")
    values = [float(profile[k]) for k in expected]
    if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in values):
        raise ProtocolError("overlap_profile values must be finite and within [0,1]")
    if abs(values[0] - 1.0) > 1e-8:
        raise ProtocolError(f"overlap_profile.frame_0 must be 1.0, got {values[0]}")
    return values


def summarize_overlap(profile: Mapping[str, Any], frame_count: int) -> dict[str, Any]:
    values = validate_overlap_profile(profile, frame_count)
    future = values[CONDITION_FRAMES:]
    if not future:
        raise ProtocolError("a benchmark sample needs at least one future frame")
    return {
        "endpoint_overlap": float(values[-1]),
        "mean_future_overlap": float(np.mean(future)),
        "minimum_future_overlap": float(np.min(future)),
        "maximum_future_overlap": float(np.max(future)),
        "condition_frames_excluded": CONDITION_FRAMES,
    }


def empirical_difficulty_bins(values: Sequence[float]) -> dict[str, Any]:
    """Derive named bins from the observed mean-future-overlap distribution.

    No fixed thresholds are guessed.  Quartiles are emitted only when the observed
    values have enough spread to create non-empty ordered intervals.  Otherwise the
    manifest records the bins as unresolved and callers must not invent assignments.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or len(arr) < 4 or not np.isfinite(arr).all():
        return {
            "status": "UNRESOLVED",
            "reason": "need at least four finite samples",
            "basis": "mean_future_overlap",
            "thresholds": None,
        }
    q25, q50, q75 = [float(x) for x in np.quantile(arr, [0.25, 0.50, 0.75])]
    if not (q25 < q50 < q75):
        return {
            "status": "UNRESOLVED",
            "reason": "empirical quartiles are not strictly ordered",
            "basis": "mean_future_overlap",
            "quantiles": {"q25": q25, "q50": q50, "q75": q75},
            "thresholds": None,
        }
    return {
        "status": "FROZEN_FROM_MANIFEST_DISTRIBUTION",
        "basis": "mean_future_overlap",
        "method": "empirical_quartiles",
        "higher_overlap_is_easier": True,
        "quantiles": {"q25": q25, "q50": q50, "q75": q75},
        "thresholds": {
            "Extreme": [0.0, q25],
            "Sparse": [q25, q50],
            "Medium": [q50, q75],
            "Dense": [q75, 1.0],
        },
        "interval_semantics": "[lo,hi), except Dense includes 1.0",
        "endpoint_lt_0_1_is_reported_separately": True,
    }


def assign_difficulty(value: float, spec: Mapping[str, Any]) -> str | None:
    if spec.get("status") != "FROZEN_FROM_MANIFEST_DISTRIBUTION":
        return None
    v = float(value)
    for label in ("Extreme", "Sparse", "Medium", "Dense"):
        lo, hi = [float(x) for x in spec["thresholds"][label]]
        if lo <= v < hi or (label == "Dense" and v == 1.0):
            return label
    raise ProtocolError(f"difficulty value {v} does not fit frozen intervals")


SAMPLE_REQUIRED = (
    "sample_id", "source_dataset", "source_image", "source_split",
    "trajectory_id", "trajectory_type", "frame_count", "condition_frame",
    "gt_rollout", "source_image_size", "rollout_size",
    "condition_checksum", "gt_rollout_checksum", "trajectory_checksum",
    "overlap_profile", "adjacent_overlap_profile", "endpoint_overlap",
    "mean_future_overlap", "minimum_future_overlap", "maximum_future_overlap",
    "checksum",
)


def _require_sha256(value: Any, label: str, *, nullable: bool = False) -> None:
    if value is None and nullable:
        return
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in text):
        raise ProtocolError(f"{label} must be a 64-character SHA-256 hex digest")


def validate_manifest_document(doc: Mapping[str, Any], *, require_files: bool = True) -> dict:
    if doc.get("schema") != MANIFEST_SCHEMA:
        raise ProtocolError(f"manifest.schema must be {MANIFEST_SCHEMA!r}")
    for key in ("h5", "split", "gen_seed", "samples", "dataset_checksum",
                "split_checksum", "difficulty_bins"):
        if key not in doc:
            raise ProtocolError(f"manifest missing {key!r}")
    if require_files:
        for key in ("h5", "split"):
            if not Path(str(doc[key])).is_file():
                raise FileNotFoundError(f"manifest.{key}: {doc[key]}")
    _require_sha256(doc["dataset_checksum"], "manifest.dataset_checksum", nullable=True)
    _require_sha256(doc["split_checksum"], "manifest.split_checksum")
    _require_sha256(doc.get("manifest_checksum"), "manifest.manifest_checksum")
    expected_manifest_checksum = stable_json_sha256(
        {key: value for key, value in doc.items() if key != "manifest_checksum"})
    if str(doc["manifest_checksum"]).lower() != expected_manifest_checksum.lower():
        raise ProtocolError(
            "manifest_checksum disagrees with the manifest content; the frozen "
            "manifest was modified without being rebuilt")
    if doc["dataset_checksum"] is None and bool(doc.get("headline_eligible", True)):
        raise ProtocolError(
            "a manifest without dataset_checksum cannot be headline_eligible")
    samples = doc["samples"]
    if not isinstance(samples, list) or not samples:
        raise ProtocolError("manifest.samples must be a non-empty list")
    seen: set[str] = set()
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise ProtocolError(f"samples[{index}] is not an object")
        missing = [key for key in SAMPLE_REQUIRED if key not in sample]
        if missing:
            raise ProtocolError(f"samples[{index}] missing {missing}")
        sid = str(sample["sample_id"])
        if not sid or sid in seen:
            raise ProtocolError(f"duplicate/empty sample_id: {sid!r}")
        seen.add(sid)
        t = int(sample["frame_count"])
        if t < 2:
            raise ProtocolError(f"{sid}: frame_count must be >=2")
        if int(sample["condition_frame"]) != 0:
            raise ProtocolError(f"{sid}: condition_frame must be 0")
        summary = summarize_overlap(sample["overlap_profile"], t)
        for key in ("endpoint_overlap", "mean_future_overlap",
                    "minimum_future_overlap", "maximum_future_overlap"):
            if abs(summary[key] - float(sample[key])) > 1e-8:
                raise ProtocolError(f"{sid}: {key} disagrees with overlap_profile")
        adjacent = sample["adjacent_overlap_profile"]
        expected_steps = [f"step_{index}" for index in range(1, t)]
        if not isinstance(adjacent, Mapping) or set(adjacent) != set(expected_steps):
            raise ProtocolError(
                f"{sid}: adjacent_overlap_profile must contain exactly "
                f"step_1..step_{t - 1}")
        if any(not math.isfinite(float(adjacent[key])) or
               not 0.0 <= float(adjacent[key]) <= 1.0 for key in expected_steps):
            raise ProtocolError(f"{sid}: adjacent overlap values must be within [0,1]")
        for key in ("source_image_size", "rollout_size"):
            size = sample[key]
            if (not isinstance(size, Sequence) or isinstance(size, (str, bytes)) or
                    len(size) != 2 or any(int(value) <= 0 for value in size)):
                raise ProtocolError(f"{sid}: {key} must be [height,width] with positive values")
        for key in ("condition_checksum", "gt_rollout_checksum",
                    "trajectory_checksum", "checksum"):
            _require_sha256(sample[key], f"{sid}.{key}")
        if doc.get("source_dataset") is not None and \
                sample["source_dataset"] != doc["source_dataset"]:
            raise ProtocolError(f"{sid}: source_dataset disagrees with manifest")
        if doc.get("source_split") is not None and sample["source_split"] != doc["source_split"]:
            raise ProtocolError(f"{sid}: source_split disagrees with manifest")
        expected_bin = assign_difficulty(summary["mean_future_overlap"],
                                         doc["difficulty_bins"])
        if sample.get("difficulty_bin") != expected_bin:
            raise ProtocolError(f"{sid}: difficulty_bin disagrees with frozen bins")
    if doc.get("sample_keys") is not None:
        if list(doc["sample_keys"]) != [str(s["sample_id"]) for s in samples]:
            raise ProtocolError("manifest.sample_keys must match samples order exactly")
    frame_counts = sorted({int(sample["frame_count"]) for sample in samples})
    if list(doc.get("frame_counts") or []) != frame_counts:
        raise ProtocolError("manifest.frame_counts disagrees with sample frame counts")
    rollout_resolutions = sorted({tuple(int(v) for v in sample["rollout_size"])
                                  for sample in samples})
    condition_resolutions = sorted({tuple(int(v) for v in sample["source_image_size"])
                                    for sample in samples})
    if doc.get("resolutions") != [list(value) for value in rollout_resolutions]:
        raise ProtocolError("manifest.resolutions must describe rollout resolutions")
    if doc.get("condition_resolutions") != [list(value) for value in condition_resolutions]:
        raise ProtocolError(
            "manifest.condition_resolutions disagrees with condition image sizes")
    return dict(doc)


def sample_index(doc: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    samples = doc.get("samples") or []
    return {str(item["sample_id"]): dict(item) for item in samples}


def null_metric_block() -> dict[str, Any]:
    """Canonical nullable metric fields; missing measurements are never zero-filled."""
    return {
        "FID": None,
        "LPIPS": None,
        "FVD_R3D_PROXY": None,
        "FVD_I3D": None,
        "direction_error": None,
        "endpoint_error": None,
        "trajectory_alignment": None,
        "registration_success_rate": None,
    }
