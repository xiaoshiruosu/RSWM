"""RSWM-Bench public abstraction layer.

Phase 2 M1-M3 intentionally contains only CPU-safe contracts, schemas, tasks and
read-only legacy dataset adapters.  It does not import or launch Wan, LoRA, GPU
evaluation, or external baselines.
"""

from .core.types import (
    BenchmarkSample,
    CapabilityPartition,
    DataDomain,
    GroundTruthBundle,
    ModelInput,
    OrganizerSample,
    PublicSample,
    Rollout,
    TaskId,
    TrackId,
)

__all__ = [
    "BenchmarkSample",
    "CapabilityPartition",
    "DataDomain",
    "GroundTruthBundle",
    "ModelInput",
    "OrganizerSample",
    "PublicSample",
    "Rollout",
    "TaskId",
    "TrackId",
]
