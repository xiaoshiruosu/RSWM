#!/usr/bin/env python3
"""Fail-closed validation for the canonical RSRoll AID training HDF5.

This is the formal ``server_run`` entry used by the ICLR pipeline.  It checks
the stored construction protocol and SemanticGate evidence without loading an
entire (hundreds-of-GB) dataset into memory.  The independent CLIP re-check is
kept in ``check_semantic_consistency.py`` and is run by the full pipeline as a
separate, checkpointed gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA = "rswm_dataset_validation/v1"
FRAME_COUNTS = {17, 25, 33}
TARGET_HW = (480, 832)
WINDOW_HW = (128, 222)
STEPS_PER_SEGMENT = 9
OVERLAP_MIN = 0.0
OVERLAP_MAX = 0.10  # canonical builder enforces [0.0, 0.10)
SEMANTIC_SCENES = {"water", "urban", "rural"}
KNOWN_SCENES = SEMANTIC_SCENES | {"general"}
REQUIRED_DATASETS = ("image", "video_frames", "intrinsics", "poses")
REQUIRED_ATTRS = ("prompt", "num_frames", "source_scene_category")


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_pair(value: Any) -> tuple[int, int] | None:
    value = _plain(value)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def validate(path: Path, *, expect_n: int | None, pixel_stride: int) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    frame_counts: Counter[int] = Counter()
    scene_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    overlaps: list[float] = []
    semantic_evidence = 0
    probe_sum = 0.0
    probe_sumsq = 0.0
    probe_n = 0
    low_std_samples: list[str] = []
    metadata: dict[str, Any] = {}

    def fail(message: str) -> None:
        # Keep the report useful without allowing a corrupt file to exhaust RAM.
        if len(errors) < 500:
            errors.append(message)

    with h5py.File(path, "r") as handle:
        if "metadata" not in handle:
            fail("missing /metadata group")
            meta_attrs: dict[str, Any] = {}
        else:
            meta_attrs = {key: _plain(value)
                          for key, value in handle["metadata"].attrs.items()}
            metadata = dict(sorted(meta_attrs.items()))
        if "data" not in handle:
            fail("missing /data group")
            keys: list[str] = []
        else:
            keys = sorted(handle["data"].keys())

        if not keys:
            fail("dataset contains zero samples")
        if expect_n is not None and len(keys) != expect_n:
            fail(f"sample count {len(keys)} != expected {expect_n}")

        total_instances = meta_attrs.get("total_instances")
        try:
            if int(total_instances) != len(keys):
                fail(f"metadata.total_instances={total_instances!r} != {len(keys)}")
        except (TypeError, ValueError):
            fail("metadata.total_instances is missing or not an integer")

        if _as_pair(meta_attrs.get("target_size")) != TARGET_HW:
            fail(f"metadata.target_size must be {TARGET_HW}")
        if _as_pair(meta_attrs.get("pose_window_size")) != WINDOW_HW:
            fail(f"metadata.pose_window_size must be {WINDOW_HW}")
        try:
            if int(meta_attrs.get("steps_per_segment")) != STEPS_PER_SEGMENT:
                fail(f"metadata.steps_per_segment must be {STEPS_PER_SEGMENT}")
        except (TypeError, ValueError):
            fail("metadata.steps_per_segment is missing or invalid")
        try:
            if int(meta_attrs.get("angle_quantize")) != 0:
                fail("metadata.angle_quantize must be 0 (continuous canonical trajectory)")
        except (TypeError, ValueError):
            fail("metadata.angle_quantize is missing or invalid")

        meta_lo = _finite_float(meta_attrs.get("min_overlap_ratio"))
        meta_hi = _finite_float(meta_attrs.get("max_overlap_ratio"))
        if meta_lo is None or not math.isclose(meta_lo, OVERLAP_MIN, abs_tol=1e-12):
            fail(f"metadata.min_overlap_ratio must be {OVERLAP_MIN}")
        if meta_hi is None or not math.isclose(meta_hi, OVERLAP_MAX, abs_tol=1e-12):
            fail(f"metadata.max_overlap_ratio must be {OVERLAP_MAX}")

        if meta_attrs.get("semgate_enabled") is not True:
            fail("metadata.semgate_enabled must be true")
        builder_version = str(meta_attrs.get("builder_version", ""))
        if "semgate" not in builder_version.lower() or "disabled" in builder_version.lower():
            fail("metadata.builder_version does not prove an enabled SemanticGate build")
        if str(meta_attrs.get("pose_injection", "")) != "plucker_from_trajectory":
            fail("metadata.pose_injection must be plucker_from_trajectory")

        for key in keys:
            group = handle["data"][key]
            for name in REQUIRED_DATASETS:
                if name not in group:
                    fail(f"{key}: missing dataset {name}")
            for name in REQUIRED_ATTRS:
                if name not in group.attrs:
                    fail(f"{key}: missing attribute {name}")
            if "video_frames" not in group:
                continue

            video = group["video_frames"]
            shape = tuple(int(value) for value in video.shape)
            if len(shape) != 4:
                fail(f"{key}: video_frames rank is {len(shape)}, expected 4")
                continue
            frames, height, width, channels = shape
            frame_counts[frames] += 1
            if frames not in FRAME_COUNTS:
                fail(f"{key}: frame count {frames} not in {sorted(FRAME_COUNTS)}")
            if (height, width) != TARGET_HW or channels != 3:
                fail(f"{key}: video shape {shape} is not (T,480,832,3)")
            if video.dtype != np.dtype("uint8"):
                fail(f"{key}: video_frames dtype {video.dtype} is not uint8")

            try:
                stored_frames = int(group.attrs.get("num_frames"))
            except (TypeError, ValueError):
                stored_frames = -1
            if stored_frames != frames:
                fail(f"{key}: num_frames={stored_frames} != video length {frames}")
            for name in ("intrinsics", "poses"):
                if name in group and (not group[name].shape or group[name].shape[0] != frames):
                    fail(f"{key}: {name} first dimension does not match {frames}")
            if "image" in group:
                image = group["image"]
                if tuple(image.shape) != (*TARGET_HW, 3):
                    fail(f"{key}: image shape {tuple(image.shape)} != (480,832,3)")
                if image.dtype != np.dtype("uint8"):
                    fail(f"{key}: image dtype {image.dtype} is not uint8")

            try:
                segments = int(group.attrs.get("num_segments"))
            except (TypeError, ValueError):
                segments = -1
            expected_frames = STEPS_PER_SEGMENT + (segments - 1) * (STEPS_PER_SEGMENT - 1)
            if segments not in (2, 3, 4) or frames != expected_frames:
                fail(f"{key}: num_segments={segments} is inconsistent with T={frames}")
            if str(group.attrs.get("pose_source", "")) != "plucker_from_trajectory":
                fail(f"{key}: pose_source is not plucker_from_trajectory")
            for name in ("trajectory", "waypoints", "segment_angles", "full_scene"):
                if name not in group:
                    fail(f"{key}: missing canonical geometry field {name}")

            overlap = _finite_float(group.attrs.get("overlap_ratio"))
            if overlap is None:
                fail(f"{key}: overlap_ratio is missing or non-finite")
            else:
                overlaps.append(overlap)
                if overlap < OVERLAP_MIN - 1e-12 or overlap >= OVERLAP_MAX + 1e-12:
                    fail(f"{key}: overlap_ratio={overlap:.8f} outside [0.0,0.1)")

            scene = str(_plain(group.attrs.get("source_scene_category", "")))
            scene_counts[scene] += 1
            if scene not in KNOWN_SCENES:
                fail(f"{key}: invalid source_scene_category={scene!r}")
            source = str(_plain(group.attrs.get("source_dataset", "")))
            source_counts[source] += 1
            if "aid" not in source.lower():
                fail(f"{key}: source_dataset={source!r} is not AID")

            if scene in SEMANTIC_SCENES:
                evidence_names = ("semgate_frac", "semgate_threshold",
                                  "semgate_n_scored", "semgate_tries")
                missing = [name for name in evidence_names if name not in group.attrs]
                if missing:
                    fail(f"{key}: missing SemanticGate evidence {missing}")
                else:
                    fraction = _finite_float(group.attrs["semgate_frac"])
                    threshold = _finite_float(group.attrs["semgate_threshold"])
                    try:
                        n_scored = int(group.attrs["semgate_n_scored"])
                        tries = int(group.attrs["semgate_tries"])
                    except (TypeError, ValueError):
                        n_scored, tries = 0, 0
                    if fraction is None or threshold is None or fraction < threshold - 1e-9:
                        fail(f"{key}: stored SemanticGate decision did not pass")
                    elif n_scored <= 0 or tries <= 0:
                        fail(f"{key}: invalid SemanticGate n_scored/tries")
                    else:
                        semantic_evidence += 1

            # Two spatially-strided frames per sample provide a bounded quality
            # probe (~5 MB total for N=2000) instead of reading ~400 GB.
            frame_ids = [0] if frames == 1 else [0, frames - 1]
            probe = np.asarray(video[frame_ids, ::pixel_stride, ::pixel_stride, :],
                               dtype=np.float64)
            if probe.size:
                sample_std = float(probe.std())
                if not math.isfinite(sample_std):
                    fail(f"{key}: non-finite pixel std")
                elif sample_std < 1.0:
                    low_std_samples.append(key)
                probe_sum += float(probe.sum())
                probe_sumsq += float(np.square(probe).sum())
                probe_n += int(probe.size)

    if semantic_evidence == 0:
        errors.append("zero semantic samples carry passing SemanticGate evidence")
    if set(scene_counts) != KNOWN_SCENES:
        errors.append(f"scene categories {sorted(scene_counts)} != {sorted(KNOWN_SCENES)}")
    if probe_n:
        mean = probe_sum / probe_n
        global_std = math.sqrt(max(0.0, probe_sumsq / probe_n - mean * mean))
    else:
        global_std = 0.0
    if global_std < 10.0:
        errors.append(f"dataset pixel probe std {global_std:.3f} < 10.0")
    if low_std_samples:
        warnings.append(
            f"{len(low_std_samples)} samples have probe std < 1.0; first={low_std_samples[:10]}")

    stat = path.stat()
    report = {
        "schema": SCHEMA,
        "status": "PASS" if not errors else "FAIL",
        "input": {
            "path": str(path.resolve()),
            "bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        },
        "protocol": {
            "dataset": "AID",
            "window_hw": list(WINDOW_HW),
            "target_hw": list(TARGET_HW),
            "steps_per_segment": STEPS_PER_SEGMENT,
            "frame_counts": sorted(FRAME_COUNTS),
            "overlap_interval": [OVERLAP_MIN, OVERLAP_MAX],
            "semgate_required": True,
        },
        "summary": {
            "n_samples": int(sum(frame_counts.values())),
            "frame_counts": {str(key): value for key, value in sorted(frame_counts.items())},
            "scene_counts": dict(sorted(scene_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "semantic_evidence_samples": semantic_evidence,
            "overlap_min": min(overlaps) if overlaps else None,
            "overlap_max": max(overlaps) if overlaps else None,
            "pixel_probe_std": global_std,
        },
        "metadata": metadata,
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="canonical train_geo.h5")
    parser.add_argument("--validate", action="store_true",
                        help="enforce the full canonical paper protocol")
    parser.add_argument("--expect-n", type=int, default=None,
                        help="require an exact sample count (full pipeline uses 2000)")
    parser.add_argument("--pixel-stride", type=int, default=32,
                        help="spatial stride for the bounded pixel-std probe")
    parser.add_argument("--report", default=None, help="optional JSON report path")
    args = parser.parse_args(argv)
    path = Path(args.input)
    if not path.is_file():
        print(f"[FAIL] dataset does not exist: {path}", file=sys.stderr)
        return 2
    if args.expect_n is not None and args.expect_n <= 0:
        parser.error("--expect-n must be positive")
    if args.pixel_stride <= 0:
        parser.error("--pixel-stride must be positive")

    try:
        report = validate(path, expect_n=args.expect_n,
                          pixel_stride=args.pixel_stride)
    except (OSError, KeyError, ValueError) as exc:
        print(f"[FAIL] cannot validate {path}: {exc}", file=sys.stderr)
        return 8 if args.validate else 2

    summary = report["summary"]
    print(f"dataset={path}")
    print(f"samples={summary['n_samples']} frames={summary['frame_counts']}")
    print(f"scenes={summary['scene_counts']}")
    print(f"resolution={TARGET_HW[0]}x{TARGET_HW[1]} window={WINDOW_HW[0]}x{WINDOW_HW[1]}")
    print(f"overlap=[{summary['overlap_min']},{summary['overlap_max']}] ")
    print(f"pixel_probe_std={summary['pixel_probe_std']:.3f}")
    print(f"semgate_evidence_samples={summary['semantic_evidence_samples']}")
    for warning in report["warnings"]:
        print(f"[WARN] {warning}", file=sys.stderr)
    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        os.replace(temporary, output)
        print(f"report={output}")

    if report["errors"]:
        label = "FAIL" if args.validate else "SUMMARY"
        print(f"[{label}] canonical checks found {len(report['errors'])} error(s):",
              file=sys.stderr)
        for message in report["errors"][:50]:
            print(f"  - {message}", file=sys.stderr)
        if len(report["errors"]) > 50:
            print(f"  ... {len(report['errors']) - 50} additional errors in report",
                  file=sys.stderr)
        return 8 if args.validate else 0
    print("[PASS] canonical AID dataset validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
