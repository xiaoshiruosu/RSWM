"""Versioned, content-bound identities for benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from benchmark.core.identity import (
    ArtifactRef,
    ComponentIdentity,
    canonical_sha256,
    sha256_file,
)
from benchmark.core.types import freeze_json

from .errors import MetricIdentityError

METRIC_IDENTITY_SCHEMA = "rswm.metric_identity/v1"
_SHA256_HEX = frozenset("0123456789abcdef")


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class MetricCategory(_StringEnum):
    VISUAL = "visual"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SEMANTIC = "semantic"


class MetricScope(_StringEnum):
    SAMPLE = "sample"
    DATASET = "dataset"


class MetricDirection(_StringEnum):
    HIGHER = "higher"
    LOWER = "lower"


class FutureOnlyOwner(_StringEnum):
    EVALUATION_RUNNER = "evaluation_runner"
    LEGACY_WRAPPER = "legacy_wrapper"
    NOT_APPLICABLE = "not_applicable"


def _nonempty(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise MetricIdentityError(f"{label} must be a non-empty string")
    return text


def _sha256(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise MetricIdentityError(f"{label} must be a 64-character SHA-256 digest")
    return text


def implementation_hash_for_sources(sources: Mapping[str, str | Path]) -> str:
    """Hash source contents under stable logical names, never absolute paths."""

    if not sources:
        raise MetricIdentityError("at least one implementation source is required")
    document = []
    for logical_name, source_path in sorted(sources.items()):
        document.append(
            {
                "logical_name": _nonempty(logical_name, "source logical name"),
                "sha256": sha256_file(Path(source_path)),
            }
        )
    return canonical_sha256(document)


@dataclass(frozen=True)
class MetricIdentity:
    """Complete identity of one metric definition and its evaluation ownership."""

    metric_id: str
    version: str
    category: MetricCategory
    scope: MetricScope
    higher_or_lower_better: MetricDirection
    preprocessing: Mapping[str, Any]
    implementation_hash: str
    future_only_owner: FutureOnlyOwner
    checkpoint_identity: Optional[ComponentIdentity] = None
    identity_checksum: Optional[str] = None
    schema: str = field(default=METRIC_IDENTITY_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_id", _nonempty(self.metric_id, "metric_id"))
        object.__setattr__(self, "version", _nonempty(self.version, "version"))
        try:
            object.__setattr__(self, "category", MetricCategory(self.category))
            object.__setattr__(self, "scope", MetricScope(self.scope))
            object.__setattr__(
                self,
                "higher_or_lower_better",
                MetricDirection(self.higher_or_lower_better),
            )
            object.__setattr__(
                self, "future_only_owner", FutureOnlyOwner(self.future_only_owner)
            )
        except ValueError as exc:
            raise MetricIdentityError(str(exc)) from exc
        if not isinstance(self.preprocessing, Mapping):
            raise MetricIdentityError("preprocessing must be a mapping")
        object.__setattr__(self, "preprocessing", freeze_json(self.preprocessing))
        object.__setattr__(
            self,
            "implementation_hash",
            _sha256(self.implementation_hash, "implementation_hash"),
        )
        if self.checkpoint_identity is not None and not isinstance(
            self.checkpoint_identity, ComponentIdentity
        ):
            raise MetricIdentityError(
                "checkpoint_identity must be ComponentIdentity or None"
            )
        checksum = self.identity_checksum
        if checksum is not None:
            checksum = _sha256(checksum, "identity_checksum")
            if checksum != self.compute_checksum():
                raise MetricIdentityError(
                    "identity_checksum disagrees with metric identity content"
                )
        object.__setattr__(self, "identity_checksum", checksum)

    def unsigned_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "metric_id": self.metric_id,
            "version": self.version,
            "category": self.category.value,
            "scope": self.scope.value,
            "higher_or_lower_better": self.higher_or_lower_better.value,
            "preprocessing": self.preprocessing,
            "checkpoint_identity": (
                self.checkpoint_identity.to_document()
                if self.checkpoint_identity is not None
                else None
            ),
            "implementation_hash": self.implementation_hash,
            "future_only_owner": self.future_only_owner.value,
        }

    def compute_checksum(self) -> str:
        return canonical_sha256(self.unsigned_document())

    def with_checksum(self) -> "MetricIdentity":
        return replace(self, identity_checksum=self.compute_checksum())

    def validate(self) -> None:
        if self.identity_checksum is None:
            raise MetricIdentityError("metric identity is missing identity_checksum")
        if self.identity_checksum != self.compute_checksum():
            raise MetricIdentityError("metric identity checksum mismatch")

    @property
    def is_verified(self) -> bool:
        return (
            self.checkpoint_identity is None
            or self.checkpoint_identity.status == "verified"
        )

    def to_document(self) -> dict[str, Any]:
        value = dict(self.unsigned_document())
        value["preprocessing"] = _thaw(self.preprocessing)
        value["identity_checksum"] = self.identity_checksum or self.compute_checksum()
        return value

    def artifact_ref(self) -> ArtifactRef:
        """Bind this metric directly into the P1-17 artifact lineage."""

        self.validate()
        return ArtifactRef(
            kind="metric",
            artifact_id=f"{self.metric_id}@{self.version}",
            sha256=self.identity_checksum,
            schema=self.schema,
            version=self.version,
            metadata={
                "category": self.category.value,
                "scope": self.scope.value,
                "future_only_owner": self.future_only_owner.value,
            },
        )

    @classmethod
    def create(cls, **kwargs: Any) -> "MetricIdentity":
        return cls(**kwargs).with_checksum()

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "MetricIdentity":
        if value.get("schema") != METRIC_IDENTITY_SCHEMA:
            raise MetricIdentityError(
                f"metric identity schema must be {METRIC_IDENTITY_SCHEMA!r}"
            )
        checkpoint = value.get("checkpoint_identity")
        result = cls(
            metric_id=value.get("metric_id", ""),
            version=value.get("version", ""),
            category=value.get("category", ""),
            scope=value.get("scope", ""),
            higher_or_lower_better=value.get("higher_or_lower_better", ""),
            preprocessing=value.get("preprocessing") or {},
            checkpoint_identity=(
                ComponentIdentity.from_document(checkpoint)
                if checkpoint is not None
                else None
            ),
            implementation_hash=value.get("implementation_hash", ""),
            future_only_owner=value.get("future_only_owner", ""),
            identity_checksum=value.get("identity_checksum"),
        )
        result.validate()
        return result


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
