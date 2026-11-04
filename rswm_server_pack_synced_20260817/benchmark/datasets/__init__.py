"""Read-only benchmark dataset adapters."""

from .base import BenchmarkDataset, DatasetIdentity
from .legacy_h5 import LegacyH5Adapter
from .manifest_v2 import LegacyManifestV2, load_manifest_v2

__all__ = [
    "BenchmarkDataset",
    "DatasetIdentity",
    "LegacyH5Adapter",
    "LegacyManifestV2",
    "load_manifest_v2",
]
