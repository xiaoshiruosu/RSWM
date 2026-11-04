"""Read-only adapter from canonical legacy HDF5 + manifest v2 to typed samples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np

from benchmark.core.errors import DatasetContractError
from benchmark.core.types import (
    BenchmarkSample,
    GroundTruthBundle,
    OrganizerSample,
    PublicSample,
)
from benchmark.spec import BenchmarkSpec, load_benchmark_spec
from benchmark.trajectory.conventions import (
    OVERLAP_DEFINITION_VERSION,
    build_direction_target,
    overlap_level,
)
from benchmark.trajectory.types import OverlapProfile, TrajectoryPlan
from .base import DatasetIdentity
from .manifest_v2 import LegacyManifestV2, legacy_protocol, load_manifest_v2


def _plain(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


class LegacyH5Adapter:
    """Map existing artifacts without modifying or rebuilding them.

    The object keeps one HDF5 handle open in mode ``r``.  ``benchmark_sample``
    is an organizer-side view containing full trajectory data.  ``public_sample``
    deliberately excludes poses, trajectory, GT and source locators.  A task is
    the only component allowed to project organizer data into a track-specific
    ``ModelInput``.
    """

    def __init__(
        self,
        h5_path: str | Path,
        manifest_path: str | Path,
        *,
        benchmark_spec: Optional[BenchmarkSpec | str | Path] = None,
        official: bool = True,
        verify_file_hashes: bool = True,
    ):
        self._closed = False
        self.manifest: LegacyManifestV2 = load_manifest_v2(
            manifest_path,
            require_files=True,
            verify_file_hashes=verify_file_hashes,
            require_headline_eligible=official,
        )
        requested_h5 = Path(h5_path).resolve()
        if requested_h5 != self.manifest.h5_path:
            raise DatasetContractError(
                f"adapter HDF5 {requested_h5} != manifest HDF5 {self.manifest.h5_path}"
            )
        self.spec: Optional[BenchmarkSpec] = None
        self._spec_source: Optional[Path] = None
        if benchmark_spec is not None:
            if isinstance(benchmark_spec, BenchmarkSpec):
                self.spec = benchmark_spec
            else:
                self._spec_source = Path(benchmark_spec).resolve()
                self.spec = load_benchmark_spec(self._spec_source)
            self._validate_spec_reference()
        self._file = h5py.File(str(requested_h5), "r")
        if self._file.mode != "r":  # pragma: no cover - h5py invariant
            self._file.close()
            raise DatasetContractError("LegacyH5Adapter must open HDF5 read-only")
        if "data" not in self._file or not isinstance(self._file["data"], h5py.Group):
            self._file.close()
            raise DatasetContractError("HDF5 is missing /data group")
        self._data = self._file["data"]
        missing = [sample_id for sample_id in self.manifest.sample_ids if sample_id not in self._data]
        if missing:
            self._file.close()
            raise DatasetContractError(f"manifest sample IDs missing from HDF5: {missing[:5]}")
        metadata = self._file.get("metadata")
        self._root_metadata = (
            {str(key): _plain(value) for key, value in metadata.attrs.items()}
            if isinstance(metadata, h5py.Group)
            else {}
        )
        if official and "pose_window_size" not in self._root_metadata:
            self._file.close()
            raise DatasetContractError("official HDF5 requires metadata.pose_window_size")
        release_identity = self.spec.release_identity() if self.spec is not None else None
        self._identity = DatasetIdentity(
            manifest_schema=self.manifest.schema,
            manifest_checksum=self.manifest.checksum,
            dataset_checksum=str(self.manifest.document["dataset_checksum"]),
            split_checksum=str(self.manifest.document["split_checksum"]),
            benchmark_spec_checksum=(self.spec.spec_checksum if self.spec else None),
            release_identity=release_identity,
        )
        self.official = bool(official)

    def _validate_spec_reference(self) -> None:
        assert self.spec is not None
        self.spec.validate()
        if self.spec.manifest.sha256 != self.manifest.checksum:
            raise DatasetContractError(
                "benchmark spec manifest identity differs from manifest v2 checksum"
            )
        if self.spec.manifest.schema != self.manifest.schema:
            raise DatasetContractError("benchmark spec manifest schema differs from manifest v2")
        raw = Path(self.spec.manifest_path)
        if raw.is_absolute():
            expected = raw.resolve()
        elif self._spec_source is not None:
            expected = (self._spec_source.parent / raw).resolve()
        else:
            expected = raw.resolve()
        if expected != self.manifest.path:
            raise DatasetContractError(
                f"benchmark spec manifest_path {expected} != loaded {self.manifest.path}"
            )

    def _ensure_open(self) -> None:
        if self._closed or not self._file.id.valid:
            raise RuntimeError("LegacyH5Adapter is closed")

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return self.manifest.sample_ids

    @property
    def identity(self) -> DatasetIdentity:
        return self._identity

    def _item_and_group(self, sample_id: str):
        self._ensure_open()
        sample_id = str(sample_id)
        try:
            item = self.manifest.sample_index[sample_id]
        except KeyError as exc:
            raise KeyError(f"sample_id not in frozen manifest: {sample_id!r}") from exc
        return item, self._data[sample_id]

    @staticmethod
    def _require_dataset(group: h5py.Group, name: str) -> np.ndarray:
        if name not in group:
            raise DatasetContractError(f"{group.name} is missing required dataset {name!r}")
        return np.asarray(group[name][:])

    def _read(self, sample_id: str, *, include_ground_truth: bool):
        item, group = self._item_and_group(sample_id)
        image = self._require_dataset(group, "image")
        if "video_frames" not in group:
            raise DatasetContractError(f"{group.name} is missing required dataset 'video_frames'")
        video_dataset = group["video_frames"]
        # build_benchmark_manifest._trajectory() freezes the path as float64
        # before computing trajectory_checksum.  Match that existing manifest
        # contract exactly; hashing the raw HDF5 float32 bytes is a different
        # identity and would reject every valid canonical manifest.
        trajectory = np.asarray(
            self._require_dataset(group, "trajectory"),
            dtype=np.float64,
        )
        poses = self._require_dataset(group, "poses")
        intrinsics = self._require_dataset(group, "intrinsics")
        waypoints = self._require_dataset(group, "waypoints")
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise DatasetContractError(f"{group.name}/image is not HWC uint8: {image.shape} {image.dtype}")
        if video_dataset.dtype != np.uint8 or video_dataset.ndim != 4 or video_dataset.shape[-1] != 3:
            raise DatasetContractError(
                f"{group.name}/video_frames is not THWC uint8: "
                f"{video_dataset.shape} {video_dataset.dtype}"
            )
        frame_count = int(item["frame_count"])
        if len(video_dataset) != frame_count or len(trajectory) != frame_count:
            raise DatasetContractError(f"{group.name}: HDF5 frame counts differ from manifest")
        if list(image.shape[:2]) != [int(value) for value in item["source_image_size"]]:
            raise DatasetContractError(f"{group.name}: condition image size differs from manifest")
        if list(video_dataset.shape[1:3]) != [int(value) for value in item["rollout_size"]]:
            raise DatasetContractError(f"{group.name}: rollout size differs from manifest")
        checks = {
            "condition": (image, item["condition_checksum"]),
            "trajectory": (trajectory, item["trajectory_checksum"]),
        }
        for label, (array, expected) in checks.items():
            actual = legacy_protocol.sha256_array(array)
            if actual.lower() != str(expected).lower():
                raise DatasetContractError(
                    f"{group.name}: {label} checksum {actual} != manifest {expected}"
                )
        overlap_values = legacy_protocol.validate_overlap_profile(
            item["overlap_profile"], frame_count
        )
        adjacent = tuple(
            float(item["adjacent_overlap_profile"][f"step_{index}"])
            for index in range(1, frame_count)
        )
        profile = OverlapProfile(
            condition_to_frame=tuple(overlap_values),
            adjacent=adjacent,
            endpoint=float(item["endpoint_overlap"]),
            mean_future=float(item["mean_future_overlap"]),
            minimum_future=float(item["minimum_future_overlap"]),
            maximum_future=float(item["maximum_future_overlap"]),
            level=overlap_level(float(item["endpoint_overlap"])),
            definition_version=OVERLAP_DEFINITION_VERSION,
        )
        direction = build_direction_target(waypoints)
        plan = TrajectoryPlan(
            per_frame_centers=trajectory,
            waypoints=waypoints,
            poses=poses,
            intrinsics=intrinsics,
            direction=direction,
            overlap=profile,
            source_algorithm=str(item.get("trajectory_source", "legacy_h5")),
            source_version=str(self._root_metadata.get("builder_version", "legacy")),
        )
        prompt = _plain(group.attrs.get("prompt", ""))
        prompt_schema = _plain(
            group.attrs.get(
                "prompt_schema",
                self._root_metadata.get("prompt_schema", "net_continuous"),
            )
        )
        metadata = {
            "prompt": prompt,
            "prompt_schema": prompt_schema,
            "condition_checksum": str(item["condition_checksum"]).lower(),
            "trajectory_id": str(item["trajectory_id"]),
            "manifest_checksum": self.manifest.checksum,
            "source_category": _plain(item.get("source_category")),
            "scene": _plain(group.attrs.get("source_scene_category", item.get("source_category"))),
            "data_domain": (
                self.spec.data_domain.value if self.spec is not None else "synthetic_controlled"
            ),
        }
        sample = BenchmarkSample(
            id=str(sample_id),
            image=image,
            trajectory=plan,
            overlap=profile,
            direction=direction,
            horizon=frame_count,
            metadata=metadata,
        )
        if not include_ground_truth:
            return sample
        video = np.asarray(video_dataset[:])
        actual_gt = legacy_protocol.sha256_array(video)
        expected_gt = str(item["gt_rollout_checksum"])
        if actual_gt.lower() != expected_gt.lower():
            raise DatasetContractError(
                f"{group.name}: ground_truth checksum {actual_gt} != manifest {expected_gt}"
            )
        full_scene = np.asarray(group["full_scene"][:]) if "full_scene" in group else None
        gt = GroundTruthBundle(
            rollout=video,
            rollout_checksum=str(item["gt_rollout_checksum"]).lower(),
            full_scene=full_scene,
            source_reference=f"{self.manifest.h5_path}::{group.name}/video_frames",
        )
        return sample, gt

    def benchmark_sample(self, sample_id: str) -> BenchmarkSample:
        return self._read(sample_id, include_ground_truth=False)

    def public_sample(self, sample_id: str) -> PublicSample:
        sample = self.benchmark_sample(sample_id)
        public_metadata = {
            key: sample.metadata[key]
            for key in ("prompt", "prompt_schema", "source_category", "scene", "data_domain")
            if key in sample.metadata
        }
        return PublicSample(
            id=sample.id,
            image=sample.image,
            horizon=sample.horizon,
            metadata=public_metadata,
        )

    def ground_truth(self, sample_id: str) -> GroundTruthBundle:
        _sample, truth = self._read(sample_id, include_ground_truth=True)
        return truth

    def organizer_sample(self, sample_id: str) -> OrganizerSample:
        sample, truth = self._read(sample_id, include_ground_truth=True)
        return OrganizerSample(sample=sample, ground_truth=truth)

    def close(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True

    def __enter__(self) -> "LegacyH5Adapter":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
