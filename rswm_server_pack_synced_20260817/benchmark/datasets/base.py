"""Dataset protocol independent of HDF5 and training loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

from benchmark.core.identity import BenchmarkReleaseIdentity
from benchmark.core.types import BenchmarkSample, GroundTruthBundle, OrganizerSample, PublicSample


@dataclass(frozen=True)
class DatasetIdentity:
    manifest_schema: str
    manifest_checksum: str
    dataset_checksum: str
    split_checksum: str
    benchmark_spec_checksum: Optional[str] = None
    release_identity: Optional[BenchmarkReleaseIdentity] = None


class BenchmarkDataset(Protocol):
    @property
    def sample_ids(self) -> Tuple[str, ...]:
        ...

    @property
    def identity(self) -> DatasetIdentity:
        ...

    def benchmark_sample(self, sample_id: str) -> BenchmarkSample:
        ...

    def public_sample(self, sample_id: str) -> PublicSample:
        ...

    def ground_truth(self, sample_id: str) -> GroundTruthBundle:
        ...

    def organizer_sample(self, sample_id: str) -> OrganizerSample:
        ...

    def close(self) -> None:
        ...
