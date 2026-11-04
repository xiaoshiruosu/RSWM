#!/usr/bin/env python3
"""Build the frozen RS-Roll benchmark manifest from an H5 and split file.

The manifest records every formal sample, condition-to-future overlap curve,
content/provenance checksums, and empirically derived difficulty bins.  It reads data
only and never downloads, trains, or accesses a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NEW = ROOT / "lora_remotesense" / "new"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(NEW))

from protocol import (MANIFEST_SCHEMA, assign_difficulty, empirical_difficulty_bins,
                      sha256_array, sha256_bytes, sha256_file,
                      stable_json_sha256, summarize_overlap,
                      validate_manifest_document)
from step_overlap import overlap_from_centers, step_overlaps


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _split_indices(path: Path, split_name: str) -> list[int]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    old = f"{split_name}_indices"
    if old in doc:
        values = doc[old]
    else:
        try:
            values = doc["splits"][split_name]["indices"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"{path}: expected {old} or splits.{split_name}.indices") from exc
    if not isinstance(values, list):
        raise TypeError(f"{path}: split indices must be a list")
    out = [int(i) for i in values]
    if len(out) != len(set(out)):
        raise ValueError(f"{path}: duplicate indices in {split_name} split")
    return out


def _resample_waypoints(waypoints: np.ndarray, frame_count: int) -> np.ndarray:
    """Arc-length interpolation used only with explicit opt-in."""
    wp = np.asarray(waypoints, dtype=np.float64)
    if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 2:
        raise ValueError(f"waypoints must be [K+1,2], got {wp.shape}")
    lengths = np.linalg.norm(np.diff(wp, axis=0), axis=1)
    total = float(lengths.sum())
    if total <= 0:
        raise ValueError("waypoint path has zero length")
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    requested = np.linspace(0.0, total, frame_count)
    out = np.empty((frame_count, 2), dtype=np.float64)
    for axis in (0, 1):
        out[:, axis] = np.interp(requested, cumulative, wp[:, axis])
    return out


def _trajectory(group, frame_count: int, allow_interpolation: bool) -> tuple[np.ndarray, str]:
    for key in ("trajectory", "frame_centers", "camera_centers"):
        if key in group:
            value = np.asarray(group[key][:], dtype=np.float64)
            if value.shape != (frame_count, 2):
                raise ValueError(f"{group.name}/{key}: {value.shape} != {(frame_count, 2)}")
            return value, f"h5:{key}"
    if allow_interpolation and "waypoints" in group:
        return _resample_waypoints(group["waypoints"][:], frame_count), \
            "derived:arc_length_interpolation_of_waypoints"
    raise ValueError(
        f"{group.name}: no per-frame trajectory. Rebuild official H5 with trajectory, "
        "or use --allow-waypoint-interpolation for non-headline diagnostics.")


def _scene_size(group, metadata: dict[str, Any]) -> tuple[int, int]:
    if "full_scene" in group:
        return tuple(int(v) for v in group["full_scene"].shape[:2])
    if "full_scene_size" in metadata:
        return tuple(int(v) for v in metadata["full_scene_size"])
    raise ValueError(f"{group.name}: missing full_scene/full_scene_size")


def _image_dhash(image: np.ndarray) -> str:
    import cv2
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"condition image must be HWC RGB, got {arr.shape}")
    gray = cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).reshape(-1)
    value = sum(int(bit) << index for index, bit in enumerate(bits))
    return f"{value:016x}"


def _trajectory_type(group, waypoints: np.ndarray, frame_count: int) -> tuple[str, list[str]]:
    explicit = group.attrs.get("trajectory_type")
    n_segments = max(1, len(waypoints) - 1)
    if explicit is not None:
        base = str(_jsonable(explicit))
    elif n_segments == 1:
        base = "straight"
    elif n_segments == 2:
        base = "turning"
    else:
        base = "multi-segment"
    tags = [base]
    if n_segments > 1 and "turning" not in tags:
        tags.append("turning")
    if frame_count >= 33:
        tags.append("long-horizon")
    if group.attrs.get("continuous_angle", False):
        tags.append("continuous-angle")
    return base, sorted(set(tags))


def _sample_checksum(condition: np.ndarray, video: np.ndarray, trajectory: np.ndarray,
                     identity: dict[str, Any]) -> str:
    return sha256_bytes((
        sha256_array(condition).encode(),
        sha256_array(video).encode(),
        sha256_array(trajectory).encode(),
        stable_json_sha256(identity).encode(),
    ))


def build(args: argparse.Namespace) -> dict[str, Any]:
    h5_path = Path(args.h5).resolve()
    split_path = Path(args.split).resolve()
    if not h5_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"h5={h5_path} split={split_path}")
    indices = _split_indices(split_path, args.split_name)
    samples: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as file:
        data = file["data"] if "data" in file else file
        all_keys = sorted(data.keys())
        bad = [i for i in indices if i < 0 or i >= len(all_keys)]
        if bad:
            raise IndexError(f"split indices out of range: {bad[:5]}")
        keys = [all_keys[i] for i in indices]
        if args.max_samples:
            keys = keys[:args.max_samples]
        if not keys:
            raise ValueError("selected sample set is empty")
        metadata = (_jsonable(dict(file["metadata"].attrs))
                    if "metadata" in file else {})
        if "pose_window_size" not in metadata:
            raise ValueError("H5 metadata.pose_window_size is required")
        window = tuple(int(v) for v in metadata["pose_window_size"])
        for key in keys:
            group = data[key]
            if "image" not in group or "video_frames" not in group:
                raise ValueError(f"{group.name}: missing image/video_frames")
            image = np.asarray(group["image"][:])
            video = np.asarray(group["video_frames"][:])
            if video.ndim != 4 or video.shape[-1] != 3 or len(video) < 2:
                raise ValueError(f"{group.name}/video_frames invalid: {video.shape}")
            frame_count = int(len(video))
            path, path_source = _trajectory(
                group, frame_count, args.allow_waypoint_interpolation)
            scene_size = _scene_size(group, metadata)
            overlaps = [overlap_from_centers(path[0], path[t], window, scene_size)
                        for t in range(frame_count)]
            profile = {f"frame_{i}": float(v) for i, v in enumerate(overlaps)}
            summary = summarize_overlap(profile, frame_count)
            adjacent = step_overlaps(path, window, scene_size)
            waypoints = (np.asarray(group["waypoints"][:], dtype=np.float64)
                         if "waypoints" in group else path[[0, -1]])
            trajectory_type, tags = _trajectory_type(group, waypoints, frame_count)
            source_image = str(_jsonable(
                group.attrs.get("source_image", group.attrs.get(
                    "source_path", f"{h5_path}::{group.name}/image"))))
            category = _jsonable(group.attrs.get(
                "category", group.attrs.get("scene", group.attrs.get("class", None))))
            trajectory_id = str(group.attrs.get(
                "trajectory_id", sha256_array(path)[:16]))
            identity = {
                "sample_id": str(key),
                "source_dataset": args.source_dataset,
                "source_image": source_image,
                "source_split": args.split_name,
                "trajectory_id": trajectory_id,
                "frame_count": frame_count,
            }
            item = {
                **identity,
                "source_category": category,
                "source_image_size": [int(image.shape[0]), int(image.shape[1])],
                "rollout_size": [int(video.shape[1]), int(video.shape[2])],
                "condition_checksum": sha256_array(image),
                "condition_dhash": _image_dhash(image),
                "trajectory_type": trajectory_type,
                "trajectory_tags": tags,
                "trajectory_source": path_source,
                "trajectory_checksum": sha256_array(path),
                "frame_count": frame_count,
                "condition_frame": 0,
                "gt_rollout": f"{h5_path}::{group.name}/video_frames",
                "gt_rollout_checksum": sha256_array(video),
                "overlap_profile": profile,
                **summary,
                "adjacent_overlap_profile": {
                    f"step_{i + 1}": float(value)
                    for i, value in enumerate(adjacent)
                },
                "endpoint_lt_0_1": bool(summary["endpoint_overlap"] < 0.1),
                "checksum": _sample_checksum(image, video, path, identity),
                "difficulty_bin": None,
            }
            samples.append(item)
    bins = empirical_difficulty_bins(
        [float(item["mean_future_overlap"]) for item in samples])
    for item in samples:
        item["difficulty_bin"] = assign_difficulty(item["mean_future_overlap"], bins)
    frame_counts = sorted({int(item["frame_count"]) for item in samples})
    condition_resolutions = sorted(
        {tuple(item["source_image_size"]) for item in samples})
    rollout_resolutions = sorted({tuple(item["rollout_size"]) for item in samples})
    headline_eligible = not any((
        args.allow_waypoint_interpolation,
        args.skip_h5_hash,
        args.max_samples is not None,
    ))
    doc = {
        "schema": MANIFEST_SCHEMA,
        "task": "single remote-sensing image + continuous trajectory -> video rollout",
        "source_dataset": args.source_dataset,
        "source_split": args.split_name,
        "h5": str(h5_path),
        "split": str(split_path),
        "dataset_checksum": (None if args.skip_h5_hash else sha256_file(h5_path)),
        "split_checksum": sha256_file(split_path),
        "gen_seed": int(args.gen_seed),
        "num_frames": frame_counts[0] if len(frame_counts) == 1 else None,
        "frame_counts": frame_counts,
        # `resolution(s)` always describes the required generated/GT rollout.
        # Condition images can be heterogeneous and are recorded separately.
        "resolution": (list(rollout_resolutions[0])
                       if len(rollout_resolutions) == 1 else None),
        "resolutions": [list(v) for v in rollout_resolutions],
        "condition_resolution": (list(condition_resolutions[0])
                                  if len(condition_resolutions) == 1 else None),
        "condition_resolutions": [list(v) for v in condition_resolutions],
        "sample_keys": [item["sample_id"] for item in samples],
        "samples": samples,
        "overlap_definition": "O_t = overlap(condition view t=0, frame t)",
        "adjacent_overlap_definition": "overlap(frame t-1, frame t); continuity audit only",
        "difficulty_bins": bins,
        "resize_crop_protocol": args.resize_crop_protocol,
        "allow_waypoint_interpolation": bool(args.allow_waypoint_interpolation),
        "max_samples": args.max_samples,
        "sample_selection": ("full_frozen_split" if args.max_samples is None
                             else "diagnostic_prefix_subset"),
        "headline_eligible": headline_eligible,
        "manifest_checksum": None,
    }
    doc["manifest_checksum"] = stable_json_sha256(
        {k: v for k, v in doc.items() if k != "manifest_checksum"})
    validate_manifest_document(doc, require_files=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--split-name", choices=("train", "val", "test"), default="test")
    parser.add_argument("--source-dataset", required=True,
                        choices=("AID", "WHU-RS19", "RESISC45"))
    parser.add_argument("--gen-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--allow-waypoint-interpolation", action="store_true",
                        help="diagnostic only; marks manifest non-headline")
    parser.add_argument("--skip-h5-hash", action="store_true",
                        help="faster diagnostic build; marks manifest non-headline")
    parser.add_argument(
        "--resize-crop-protocol",
        default="Use H5 condition/GT pixels as built; evaluator performs no hidden crop or resize",
    )
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    doc = build(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] {len(doc['samples'])} samples -> {output}")
    print(f"[INFO] frames={doc['frame_counts']} rollout_resolutions={doc['resolutions']} "
          f"condition_resolutions={doc['condition_resolutions']}")
    print(f"[INFO] difficulty_bins={doc['difficulty_bins']['status']}")
    print(f"[INFO] headline_eligible={doc['headline_eligible']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
