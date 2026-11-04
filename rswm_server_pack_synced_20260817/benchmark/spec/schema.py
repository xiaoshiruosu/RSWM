"""Checksum-protected RSWM-Bench specification sidecar.

The sidecar references, but never rewrites, a frozen manifest v2.  Its own hash
binds task, track, domain, capability partition, protocol, schema bundle,
evaluator declaration and manifest identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from benchmark.core.errors import IdentityError, SchemaError
from benchmark.core.identity import (
    BenchmarkReleaseIdentity,
    ComponentIdentity,
    MODEL_IDENTITY_SCHEMA,
    canonical_sha256,
)
from benchmark.core.types import (
    CapabilityPartition,
    DataDomain,
    TaskId,
    TrackId,
    freeze_json,
)

BENCHMARK_SPEC_SCHEMA = "rswm.benchmark_spec/v1"
BENCHMARK_SPEC_VERSION = "v1"
SCHEMA_BUNDLE_SCHEMA = "rswm.schema_bundle/v1"
SAMPLE_VISIBILITY_POLICY = "rswm.sample_visibility/track-whitelist-v1"

_SCHEMA_DIRECTORY = Path(__file__).resolve().parent / "schemas"
_SCHEMA_FILES = (
    "benchmark_spec_v1.schema.json",
    "model_identity_v1.schema.json",
    "release_identity_v1.schema.json",
    "artifact_lineage_v1.schema.json",
)


def schema_bundle_descriptor() -> dict[str, Any]:
    files = {}
    for name in _SCHEMA_FILES:
        path = _SCHEMA_DIRECTORY / name
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SchemaError(f"cannot load schema bundle file {path}: {exc}") from exc
        files[name] = {
            "$id": document.get("$id"),
            "sha256": canonical_sha256(document),
        }
    return {
        "schema": SCHEMA_BUNDLE_SCHEMA,
        "version": "v1",
        "files": files,
        "external_contracts": {
            "legacy_manifest": "rsroll.benchmark_manifest/v2",
            "sample_visibility": SAMPLE_VISIBILITY_POLICY,
            "world_model_output": "rswm.rollout/thwc-uint8-v1",
            "model_identity": MODEL_IDENTITY_SCHEMA,
        },
    }


def schema_bundle_identity() -> ComponentIdentity:
    descriptor = schema_bundle_descriptor()
    return ComponentIdentity(
        kind="schema_bundle",
        artifact_id="rswm-bench-schema-bundle-v1",
        schema=SCHEMA_BUNDLE_SCHEMA,
        version="v1",
        sha256=canonical_sha256(descriptor),
        status="verified",
    )


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_version: str
    task_id: TaskId
    conditioning_track: TrackId
    data_domain: DataDomain
    capability_partition: CapabilityPartition
    manifest_path: str
    manifest: ComponentIdentity
    protocol: ComponentIdentity
    schema_bundle: ComponentIdentity
    evaluator: ComponentIdentity
    metric_panel_version: str
    direction_convention_version: str
    overlap_definition_version: str
    sample_visibility_policy: str = SAMPLE_VISIBILITY_POLICY
    task_parameters: Mapping[str, Any] = field(default_factory=dict)
    headline_eligible: bool = False
    legacy_compatible: bool = False
    spec_checksum: Optional[str] = None
    schema: str = BENCHMARK_SPEC_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", TaskId(self.task_id))
        object.__setattr__(self, "conditioning_track", TrackId(self.conditioning_track))
        object.__setattr__(self, "data_domain", DataDomain(self.data_domain))
        object.__setattr__(
            self,
            "capability_partition",
            CapabilityPartition(self.capability_partition),
        )
        if self.schema != BENCHMARK_SPEC_SCHEMA:
            raise SchemaError(
                f"spec.schema must be {BENCHMARK_SPEC_SCHEMA!r}, got {self.schema!r}"
            )
        for label in (
            "benchmark_version",
            "manifest_path",
            "metric_panel_version",
            "direction_convention_version",
            "overlap_definition_version",
            "sample_visibility_policy",
        ):
            if not str(getattr(self, label)).strip():
                raise SchemaError(f"spec.{label} must be non-empty")
        expected_kinds = {
            "manifest": "manifest",
            "protocol": "protocol",
            "schema_bundle": "schema_bundle",
            "evaluator": "evaluator",
        }
        for label, expected in expected_kinds.items():
            component = getattr(self, label)
            if not isinstance(component, ComponentIdentity) or component.kind != expected:
                raise SchemaError(f"spec.{label} must be a {expected!r} ComponentIdentity")
        if self.manifest.status != "verified":
            raise SchemaError("benchmark spec requires a verified manifest identity")
        if self.protocol.status != "verified" or self.schema_bundle.status != "verified":
            raise SchemaError("protocol and schema bundle must be verified")
        if self.sample_visibility_policy != SAMPLE_VISIBILITY_POLICY:
            raise SchemaError(
                f"unknown sample visibility policy: {self.sample_visibility_policy!r}"
            )
        if self.conditioning_track == TrackId.C_MULTIVIEW:
            if self.capability_partition != CapabilityPartition.MULTIVIEW:
                raise SchemaError("C_multiview must use the multiview capability partition")
        elif self.capability_partition == CapabilityPartition.MULTIVIEW:
            raise SchemaError("single-view tracks may not use the multiview partition")
        parameters = freeze_json(self.task_parameters)
        object.__setattr__(self, "task_parameters", parameters)
        checksum = self.spec_checksum
        if checksum is not None:
            checksum = str(checksum).lower()
            if len(checksum) != 64 or any(ch not in "0123456789abcdef" for ch in checksum):
                raise SchemaError("spec_checksum must be a 64-character SHA-256 digest")
            if checksum != self.compute_checksum():
                raise SchemaError("spec_checksum disagrees with benchmark spec content")
            object.__setattr__(self, "spec_checksum", checksum)

    def unsigned_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "benchmark_version": self.benchmark_version,
            "task_id": self.task_id.value,
            "conditioning_track": self.conditioning_track.value,
            "data_domain": self.data_domain.value,
            "capability_partition": self.capability_partition.value,
            "manifest_path": self.manifest_path,
            "manifest": self.manifest.to_document(),
            "protocol": self.protocol.to_document(),
            "schema_bundle": self.schema_bundle.to_document(),
            "evaluator": self.evaluator.to_document(),
            "metric_panel_version": self.metric_panel_version,
            "direction_convention_version": self.direction_convention_version,
            "overlap_definition_version": self.overlap_definition_version,
            "sample_visibility_policy": self.sample_visibility_policy,
            "task_parameters": _plain(self.task_parameters),
            "headline_eligible": bool(self.headline_eligible),
            "legacy_compatible": bool(self.legacy_compatible),
        }

    def compute_checksum(self) -> str:
        return canonical_sha256(self.unsigned_document())

    def with_checksum(self) -> "BenchmarkSpec":
        return replace(self, spec_checksum=self.compute_checksum())

    def validate(self) -> None:
        if self.spec_checksum is None:
            raise SchemaError("benchmark spec is missing spec_checksum")

    def to_document(self) -> dict[str, Any]:
        value = self.unsigned_document()
        value["spec_checksum"] = self.spec_checksum or self.compute_checksum()
        return value

    def release_identity(self) -> BenchmarkReleaseIdentity:
        self.validate()
        spec_identity = ComponentIdentity(
            kind="benchmark_spec",
            artifact_id=(
                f"{self.benchmark_version}:{self.task_id.value}:"
                f"{self.conditioning_track.value}:{self.spec_checksum[:12]}"
            ),
            schema=self.schema,
            version=BENCHMARK_SPEC_VERSION,
            sha256=self.spec_checksum,
            status="verified",
        )
        return BenchmarkReleaseIdentity.create(
            benchmark_version=self.benchmark_version,
            protocol=self.protocol,
            benchmark_spec=spec_identity,
            schema_bundle=self.schema_bundle,
            evaluator=self.evaluator,
            manifest=self.manifest,
        )

    @classmethod
    def create(cls, **kwargs: Any) -> "BenchmarkSpec":
        return cls(**kwargs).with_checksum()

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "BenchmarkSpec":
        try:
            result = cls(
                schema=value.get("schema", ""),
                benchmark_version=value.get("benchmark_version", ""),
                task_id=value.get("task_id", ""),
                conditioning_track=value.get("conditioning_track", ""),
                data_domain=value.get("data_domain", ""),
                capability_partition=value.get("capability_partition", ""),
                manifest_path=value.get("manifest_path", ""),
                manifest=ComponentIdentity.from_document(value.get("manifest") or {}),
                protocol=ComponentIdentity.from_document(value.get("protocol") or {}),
                schema_bundle=ComponentIdentity.from_document(
                    value.get("schema_bundle") or {}
                ),
                evaluator=ComponentIdentity.from_document(value.get("evaluator") or {}),
                metric_panel_version=value.get("metric_panel_version", ""),
                direction_convention_version=value.get(
                    "direction_convention_version", ""
                ),
                overlap_definition_version=value.get("overlap_definition_version", ""),
                sample_visibility_policy=value.get(
                    "sample_visibility_policy", SAMPLE_VISIBILITY_POLICY
                ),
                task_parameters=value.get("task_parameters") or {},
                headline_eligible=bool(value.get("headline_eligible", False)),
                legacy_compatible=bool(value.get("legacy_compatible", False)),
                spec_checksum=value.get("spec_checksum"),
            )
        except (ValueError, TypeError, IdentityError) as exc:
            if isinstance(exc, SchemaError):
                raise
            raise SchemaError(f"invalid benchmark spec: {exc}") from exc
        result.validate()
        return result


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value
