"""Create tiny, workspace-local legacy HDF5/manifest/spec fixtures.

Each test receives a unique disposable directory under ``temp_benchmark``;
no formal HDF5, checkpoint, result or historical workspace artifact is touched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from benchmark.core.identity import ComponentIdentity, sha256_file
from benchmark.core.types import CapabilityPartition, DataDomain, TaskId, TrackId
from benchmark.spec.schema import BenchmarkSpec, schema_bundle_identity
from benchmark.trajectory.conventions import (
    DIRECTION_CONVENTION_VERSION,
    OVERLAP_DEFINITION_VERSION,
)
from server_run.baselines.build_benchmark_manifest import build as build_manifest


@dataclass(frozen=True)
class FixturePaths:
    root: Path
    h5: Path
    split: Path
    manifest: Path
    spec: Path


class FixtureTempDirectory:
    """Portable workspace-local temp directory.

    Some managed Windows sandboxes create ``tempfile.TemporaryDirectory`` with
    ACLs that native HDF5 cannot open.  Explicit workspace directories avoid
    that platform artifact while CI continues to use the same CPU-only files.
    """

    def __init__(self):
        package_root = Path(__file__).resolve().parents[2]
        temp_root = package_root / "temp_benchmark"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.path = temp_root / f"phase2_fixture_{uuid.uuid4().hex}"
        self.path.mkdir()
        self.name = str(self.path)

    def cleanup(self) -> None:
        if self.path.exists():
            shutil.rmtree(self.path)


def _translated_video(image: np.ndarray, frame_count: int, axis: int) -> np.ndarray:
    frames = np.empty((frame_count, *image.shape), dtype=np.uint8)
    for index in range(frame_count):
        frames[index] = np.roll(image, shift=-(index // 4), axis=axis)
    frames[0] = image
    return frames


def create_fixture(root: Path, *, sample_count: int = 2) -> FixturePaths:
    root.mkdir(parents=True, exist_ok=True)
    h5_path = root / "tiny_legacy.h5"
    split_path = root / "official_splits.json"
    manifest_path = root / "manifest_v2.json"
    spec_path = root / "benchmark_spec_v1.json"
    frame_count = 17
    window = (4, 6)

    with h5py.File(h5_path, "w") as handle:
        metadata = handle.create_group("metadata")
        metadata.attrs["pose_window_size"] = window
        metadata.attrs["full_scene_size"] = (32, 32)
        metadata.attrs["prompt_schema"] = "net_continuous"
        metadata.attrs["builder_version"] = "cpu-fixture/v1"
        metadata.attrs["total_instances"] = sample_count
        data = handle.create_group("data")
        for index in range(sample_count):
            key = f"sample_{index:05d}"
            group = data.create_group(key)
            rng = np.random.default_rng(100 + index)
            image = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
            if index % 2 == 0:
                start = np.array([10.0, 10.0])
                end = np.array([10.0, 16.0])  # east, endpoint overlap 0
                axis = 1
                bearing = 0.0
            else:
                start = np.array([12.0, 12.0])
                end = np.array([8.0, 12.0])  # north, endpoint overlap 0
                axis = 0
                bearing = 90.0
            trajectory = np.linspace(start, end, frame_count, dtype=np.float32)
            waypoints = np.stack([start, end]).astype(np.float32)
            poses = np.tile(np.eye(4, dtype=np.float32), (frame_count, 1, 1))
            poses[:, 0, 3] = (trajectory[:, 1] - start[1]) / window[1]
            poses[:, 1, 3] = (trajectory[:, 0] - start[0]) / window[0]
            intrinsics = np.tile(
                np.array([10.0, 10.0, 6.0, 4.0], dtype=np.float32),
                (frame_count, 1),
            )
            video = _translated_video(image, frame_count, axis)
            full_scene = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            group.create_dataset("image", data=image)
            group.create_dataset("video_frames", data=video)
            group.create_dataset("trajectory", data=trajectory)
            group.create_dataset("waypoints", data=waypoints)
            group.create_dataset("segment_angles", data=np.array([bearing], dtype=np.float32))
            group.create_dataset("poses", data=poses)
            group.create_dataset("intrinsics", data=intrinsics)
            group.create_dataset("full_scene", data=full_scene)
            group.attrs["prompt"] = (
                "A satellite video over a rural area following a 1-segment "
                f"trajectory with continuous net camera bearing {bearing:.1f} degrees."
            )
            group.attrs["prompt_schema"] = "net_continuous"
            group.attrs["num_frames"] = frame_count
            group.attrs["num_segments"] = 1
            group.attrs["trajectory_type"] = "straight"
            group.attrs["source_scene_category"] = "rural"
            group.attrs["category"] = "rural"
            group.attrs["source_image"] = f"fixture-source-{index}"

    split_path.write_text(
        json.dumps({"test_indices": list(range(sample_count))}, indent=2),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        h5=str(h5_path),
        split=str(split_path),
        split_name="test",
        source_dataset="AID",
        gen_seed=42,
        max_samples=None,
        allow_waypoint_interpolation=False,
        skip_h5_hash=False,
        resize_crop_protocol="fixture pixels unchanged; no hidden crop or resize",
    )
    manifest = build_manifest(args)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    package_root = Path(__file__).resolve().parents[2]
    protocol_path = package_root / "RSWM_BENCH_PROTOCOL_V1.md"
    spec = BenchmarkSpec.create(
        benchmark_version="RSWM-Bench-v1",
        task_id=TaskId.T3_OVERLAP_LADDER,
        conditioning_track=TrackId.B_CAMERA,
        data_domain=DataDomain.SYNTHETIC_CONTROLLED,
        capability_partition=CapabilityPartition.NATIVE,
        manifest_path=str(manifest_path.resolve()),
        manifest=ComponentIdentity(
            kind="manifest",
            artifact_id="cpu-fixture-manifest-v2",
            schema=manifest["schema"],
            version="v2",
            sha256=manifest["manifest_checksum"],
            status="verified",
        ),
        protocol=ComponentIdentity(
            kind="protocol",
            artifact_id="RSWM-Bench-Protocol-V1",
            schema="rswm.protocol/v1",
            version="v1",
            sha256=sha256_file(protocol_path),
            status="verified",
        ),
        schema_bundle=schema_bundle_identity(),
        evaluator=ComponentIdentity(
            kind="evaluator",
            artifact_id="rswm-evaluator-m4-not-implemented",
            schema="rswm.evaluator/v1",
            version="planned-m4",
            sha256=None,
            status="declared",
        ),
        metric_panel_version="rswm.metric-panel/v1-declared",
        direction_convention_version=DIRECTION_CONVENTION_VERSION,
        overlap_definition_version=OVERLAP_DEFINITION_VERSION,
        task_parameters={"overlap_level": "L4_overlap_lt10"},
        headline_eligible=True,
        legacy_compatible=False,
    )
    spec_path.write_text(
        json.dumps(spec.to_document(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return FixturePaths(root, h5_path, split_path, manifest_path, spec_path)
