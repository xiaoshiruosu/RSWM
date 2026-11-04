"""Release identity and artifact-lineage primitives.

P1-16 freezes one identity record spanning protocol, benchmark spec, schema
bundle, evaluator and manifest.  P1-17 records the auditable chain from a model
checkpoint through an adapter and benchmark inputs to metrics and results.

The module is deliberately CPU-only and performs no filesystem mutation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

from .errors import ArtifactLineageError, IdentityError

RELEASE_IDENTITY_SCHEMA = "rswm.benchmark_release_identity/v1"
ARTIFACT_LINEAGE_SCHEMA = "rswm.artifact_lineage/v1"
MODEL_IDENTITY_SCHEMA = "rswm.model_identity/v1"
IDENTITY_STATUSES = frozenset({"declared", "verified", "not_available"})
RELEASE_COMPONENT_KINDS = (
    "protocol",
    "benchmark_spec",
    "schema_bundle",
    "evaluator",
    "manifest",
)
LINEAGE_KIND_ORDER = {
    "checkpoint": 0,
    "adapter": 1,
    "benchmark_spec": 2,
    "manifest": 3,
    "metric": 4,
    "result": 5,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise IdentityError("identity documents may not contain NaN or infinity")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes for identity hashing."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _require_nonempty(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise IdentityError(f"{label} must be a non-empty string")
    return text


def _normalize_sha256(value: Optional[str], label: str, *, required: bool) -> Optional[str]:
    if value is None:
        if required:
            raise IdentityError(f"{label} is required")
        return None
    text = str(value).lower()
    if not _SHA256_RE.fullmatch(text):
        raise IdentityError(f"{label} must be a 64-character SHA-256 digest")
    return text


def _normalize_status(value: Any, label: str) -> str:
    status = str(value).strip().lower()
    if status not in IDENTITY_STATUSES:
        raise IdentityError(
            f"{label}={status!r}; expected one of {sorted(IDENTITY_STATUSES)}"
        )
    return status


def _optional_nonempty(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    return _require_nonempty(value, label)


def _string_tuple(value: Iterable[Any], label: str) -> Tuple[str, ...]:
    items = tuple(_require_nonempty(item, label) for item in value)
    if len(set(items)) != len(items):
        raise IdentityError(f"{label} may not contain duplicate values")
    return items


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IdentityError(f"{label} must be an object")
    return value


@dataclass(frozen=True)
class CheckpointIdentity:
    """Repository and content anchors for one model checkpoint release."""

    repository: str
    revision: Optional[str] = None
    sha256_manifest: Optional[str] = None
    config_hash: Optional[str] = None
    status: str = "declared"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "repository",
            _require_nonempty(self.repository, "checkpoint_identity.repository"),
        )
        object.__setattr__(
            self,
            "revision",
            _optional_nonempty(self.revision, "checkpoint_identity.revision"),
        )
        object.__setattr__(
            self,
            "sha256_manifest",
            _normalize_sha256(
                self.sha256_manifest,
                "checkpoint_identity.sha256_manifest",
                required=False,
            ),
        )
        object.__setattr__(
            self,
            "config_hash",
            _normalize_sha256(
                self.config_hash,
                "checkpoint_identity.config_hash",
                required=False,
            ),
        )
        status = _normalize_status(self.status, "checkpoint_identity.status")
        object.__setattr__(self, "status", status)
        if status == "verified":
            missing = [
                name
                for name in ("revision", "sha256_manifest", "config_hash")
                if getattr(self, name) is None
            ]
            if missing:
                raise IdentityError(
                    "verified checkpoint identity requires revision, sha256_manifest "
                    f"and config_hash; missing={missing}"
                )

    def to_document(self) -> dict[str, Any]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "sha256_manifest": self.sha256_manifest,
            "config_hash": self.config_hash,
            "status": self.status,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "CheckpointIdentity":
        return cls(
            repository=value.get("repository", ""),
            revision=value.get("revision"),
            sha256_manifest=value.get("sha256_manifest"),
            config_hash=value.get("config_hash"),
            status=value.get("status", "declared"),
        )


@dataclass(frozen=True)
class ArchitectureIdentity:
    """Architecture implementation identity, independent of checkpoint branding."""

    transformer_type: str
    vae: str
    text_encoder: str
    camera_conditioning_type: str
    implementation_aliases: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "transformer_type",
            "vae",
            "text_encoder",
            "camera_conditioning_type",
        ):
            object.__setattr__(
                self,
                name,
                _require_nonempty(getattr(self, name), f"architecture_identity.{name}"),
            )
        object.__setattr__(
            self,
            "implementation_aliases",
            _string_tuple(
                self.implementation_aliases,
                "architecture_identity.implementation_aliases",
            ),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "transformer_type": self.transformer_type,
            "vae": self.vae,
            "text_encoder": self.text_encoder,
            "camera_conditioning_type": self.camera_conditioning_type,
            "implementation_aliases": list(self.implementation_aliases),
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ArchitectureIdentity":
        return cls(
            transformer_type=value.get("transformer_type", ""),
            vae=value.get("vae", ""),
            text_encoder=value.get("text_encoder", ""),
            camera_conditioning_type=value.get("camera_conditioning_type", ""),
            implementation_aliases=tuple(value.get("implementation_aliases") or ()),
        )


@dataclass(frozen=True)
class CameraConditionIdentity:
    """Versioned representation and injection contract for camera conditioning."""

    encoding: str
    pose_injection: str
    injector_type: str
    version: str = "v1"

    def __post_init__(self) -> None:
        for name in ("encoding", "pose_injection", "injector_type", "version"):
            object.__setattr__(
                self,
                name,
                _require_nonempty(getattr(self, name), f"camera_condition_identity.{name}"),
            )

    def to_document(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "pose_injection": self.pose_injection,
            "injector_type": self.injector_type,
            "version": self.version,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "CameraConditionIdentity":
        return cls(
            encoding=value.get("encoding", ""),
            pose_injection=value.get("pose_injection", ""),
            injector_type=value.get("injector_type", ""),
            version=value.get("version", ""),
        )


@dataclass(frozen=True)
class LoraIdentity:
    """LoRA adaptation identity without embedding or loading any adapter weights."""

    target_modules: Tuple[str, ...]
    rank: int
    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "target_modules",
            _string_tuple(self.target_modules, "lora_identity.target_modules"),
        )
        if not self.target_modules:
            raise IdentityError("lora_identity.target_modules may not be empty")
        try:
            rank = int(self.rank)
            integral = float(self.rank) == rank
        except (TypeError, ValueError, OverflowError):
            integral = False
            rank = 0
        if isinstance(self.rank, bool) or not integral or rank <= 0:
            raise IdentityError("lora_identity.rank must be a positive integer")
        object.__setattr__(self, "rank", rank)
        try:
            alpha = float(self.alpha)
        except (TypeError, ValueError, OverflowError) as exc:
            raise IdentityError(
                "lora_identity.alpha must be finite and positive"
            ) from exc
        if isinstance(self.alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
            raise IdentityError("lora_identity.alpha must be finite and positive")
        object.__setattr__(self, "alpha", alpha)

    def to_document(self) -> dict[str, Any]:
        return {
            "target_modules": list(self.target_modules),
            "rank": self.rank,
            "alpha": self.alpha,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "LoraIdentity":
        return cls(
            target_modules=tuple(value.get("target_modules") or ()),
            rank=value.get("rank", 0),
            alpha=value.get("alpha", 0),
        )


@dataclass(frozen=True)
class ExecutionIdentity:
    """Execution route identity; legacy names such as wan/wan-inline stay valid."""

    route: str
    entrypoint: Optional[str] = None
    legacy_aliases: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "route", _require_nonempty(self.route, "execution_identity.route")
        )
        object.__setattr__(
            self,
            "entrypoint",
            _optional_nonempty(self.entrypoint, "execution_identity.entrypoint"),
        )
        object.__setattr__(
            self,
            "legacy_aliases",
            _string_tuple(self.legacy_aliases, "execution_identity.legacy_aliases"),
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "entrypoint": self.entrypoint,
            "legacy_aliases": list(self.legacy_aliases),
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ExecutionIdentity":
        return cls(
            route=value.get("route", ""),
            entrypoint=value.get("entrypoint"),
            legacy_aliases=tuple(value.get("legacy_aliases") or ()),
        )


@dataclass(frozen=True)
class ModelIdentity:
    """Layered identity for a release, architecture, adapter and execution route."""

    model_family: str
    release_identity: str
    adapter_identity: str
    checkpoint_identity: CheckpointIdentity
    architecture_identity: ArchitectureIdentity
    camera_condition_identity: CameraConditionIdentity
    execution_identity: ExecutionIdentity
    lora_identity: Optional[LoraIdentity] = None
    status: str = "declared"
    identity_checksum: Optional[str] = None
    schema: str = MODEL_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != MODEL_IDENTITY_SCHEMA:
            raise IdentityError(
                f"model_identity.schema must be {MODEL_IDENTITY_SCHEMA!r}, "
                f"got {self.schema!r}"
            )
        for name in ("model_family", "release_identity", "adapter_identity"):
            object.__setattr__(
                self,
                name,
                _require_nonempty(getattr(self, name), f"model_identity.{name}"),
            )
        expected_types = {
            "checkpoint_identity": CheckpointIdentity,
            "architecture_identity": ArchitectureIdentity,
            "camera_condition_identity": CameraConditionIdentity,
            "execution_identity": ExecutionIdentity,
        }
        for name, expected in expected_types.items():
            if not isinstance(getattr(self, name), expected):
                raise IdentityError(
                    f"model_identity.{name} must be {expected.__name__}"
                )
        if self.lora_identity is not None and not isinstance(
            self.lora_identity, LoraIdentity
        ):
            raise IdentityError("model_identity.lora_identity must be LoraIdentity or None")
        status = _normalize_status(self.status, "model_identity.status")
        object.__setattr__(self, "status", status)
        if status == "verified" and self.checkpoint_identity.status != "verified":
            raise IdentityError(
                "verified model identity requires a verified checkpoint identity"
            )
        checksum = _normalize_sha256(
            self.identity_checksum,
            "model_identity.identity_checksum",
            required=False,
        )
        object.__setattr__(self, "identity_checksum", checksum)
        if checksum is not None and checksum != self.compute_checksum():
            raise IdentityError(
                "identity_checksum disagrees with layered model identity content"
            )

    @property
    def is_verified(self) -> bool:
        return self.status == "verified" and self.checkpoint_identity.status == "verified"

    def unsigned_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "model_family": self.model_family,
            "release_identity": self.release_identity,
            "adapter_identity": self.adapter_identity,
            "checkpoint_identity": self.checkpoint_identity.to_document(),
            "architecture_identity": self.architecture_identity.to_document(),
            "camera_condition_identity": self.camera_condition_identity.to_document(),
            "lora_identity": (
                self.lora_identity.to_document() if self.lora_identity is not None else None
            ),
            "execution_identity": self.execution_identity.to_document(),
            "status": self.status,
        }

    def compute_checksum(self) -> str:
        return canonical_sha256(self.unsigned_document())

    def with_checksum(self) -> "ModelIdentity":
        return replace(self, identity_checksum=self.compute_checksum())

    def validate(self, *, require_verified: bool = False) -> None:
        if self.identity_checksum is None:
            raise IdentityError("model identity is missing identity_checksum")
        if require_verified and not self.is_verified:
            raise IdentityError(
                "model identity is not verified; both model and checkpoint status "
                "must be verified"
            )

    def to_document(self) -> dict[str, Any]:
        value = self.unsigned_document()
        value["identity_checksum"] = self.identity_checksum or self.compute_checksum()
        return value

    @classmethod
    def create(cls, **kwargs: Any) -> "ModelIdentity":
        return cls(**kwargs).with_checksum()

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ModelIdentity":
        value = _require_mapping(value, "model_identity")
        lora_value = value.get("lora_identity")
        if lora_value is not None:
            lora_value = _require_mapping(lora_value, "model_identity.lora_identity")
        result = cls(
            schema=value.get("schema", ""),
            model_family=value.get("model_family", ""),
            release_identity=value.get("release_identity", ""),
            adapter_identity=value.get("adapter_identity", ""),
            checkpoint_identity=CheckpointIdentity.from_document(
                _require_mapping(
                    value.get("checkpoint_identity"),
                    "model_identity.checkpoint_identity",
                )
            ),
            architecture_identity=ArchitectureIdentity.from_document(
                _require_mapping(
                    value.get("architecture_identity"),
                    "model_identity.architecture_identity",
                )
            ),
            camera_condition_identity=CameraConditionIdentity.from_document(
                _require_mapping(
                    value.get("camera_condition_identity"),
                    "model_identity.camera_condition_identity",
                )
            ),
            lora_identity=(
                LoraIdentity.from_document(lora_value) if lora_value is not None else None
            ),
            execution_identity=ExecutionIdentity.from_document(
                _require_mapping(
                    value.get("execution_identity"),
                    "model_identity.execution_identity",
                )
            ),
            status=value.get("status", "declared"),
            identity_checksum=value.get("identity_checksum"),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ComponentIdentity:
    """Identity of one release component.

    ``declared`` supports M1-M3 placeholders such as the not-yet-implemented
    evaluator.  A release-ready identity requires every component to be
    ``verified`` and to carry a SHA-256 digest.
    """

    kind: str
    artifact_id: str
    schema: str
    version: str
    sha256: Optional[str]
    status: str = "verified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _require_nonempty(self.kind, "component.kind"))
        object.__setattr__(
            self, "artifact_id", _require_nonempty(self.artifact_id, "component.artifact_id")
        )
        object.__setattr__(self, "schema", _require_nonempty(self.schema, "component.schema"))
        object.__setattr__(self, "version", _require_nonempty(self.version, "component.version"))
        status = str(self.status).strip().lower()
        if status not in IDENTITY_STATUSES:
            raise IdentityError(
                f"component.status={status!r}; expected one of {sorted(IDENTITY_STATUSES)}"
            )
        object.__setattr__(self, "status", status)
        digest = _normalize_sha256(
            self.sha256,
            "component.sha256",
            required=(status == "verified"),
        )
        object.__setattr__(self, "sha256", digest)

    def to_document(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "artifact_id": self.artifact_id,
            "schema": self.schema,
            "version": self.version,
            "sha256": self.sha256,
            "status": self.status,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ComponentIdentity":
        return cls(
            kind=value.get("kind", ""),
            artifact_id=value.get("artifact_id", ""),
            schema=value.get("schema", ""),
            version=value.get("version", ""),
            sha256=value.get("sha256"),
            status=value.get("status", "verified"),
        )


@dataclass(frozen=True)
class BenchmarkReleaseIdentity:
    """Unified P1-16 benchmark release identity."""

    benchmark_version: str
    protocol: ComponentIdentity
    benchmark_spec: ComponentIdentity
    schema_bundle: ComponentIdentity
    evaluator: ComponentIdentity
    manifest: ComponentIdentity
    release_checksum: Optional[str] = None
    schema: str = RELEASE_IDENTITY_SCHEMA
    model_identity: Optional[ModelIdentity] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "benchmark_version",
            _require_nonempty(self.benchmark_version, "release.benchmark_version"),
        )
        if self.schema != RELEASE_IDENTITY_SCHEMA:
            raise IdentityError(
                f"release.schema must be {RELEASE_IDENTITY_SCHEMA!r}, got {self.schema!r}"
            )
        for field_name, expected_kind in zip(
            RELEASE_COMPONENT_KINDS,
            RELEASE_COMPONENT_KINDS,
        ):
            component = getattr(self, field_name)
            if not isinstance(component, ComponentIdentity):
                raise IdentityError(f"release.{field_name} must be ComponentIdentity")
            if component.kind != expected_kind:
                raise IdentityError(
                    f"release.{field_name}.kind={component.kind!r}; expected {expected_kind!r}"
                )
        if self.model_identity is not None and not isinstance(
            self.model_identity, ModelIdentity
        ):
            raise IdentityError("release.model_identity must be ModelIdentity or None")
        if self.model_identity is not None:
            self.model_identity.validate()
        checksum = _normalize_sha256(
            self.release_checksum,
            "release.release_checksum",
            required=False,
        )
        object.__setattr__(self, "release_checksum", checksum)
        if checksum is not None and checksum != self.compute_checksum():
            raise IdentityError("release_checksum disagrees with release identity content")

    def unsigned_document(self) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "benchmark_version": self.benchmark_version,
            "protocol": self.protocol.to_document(),
            "benchmark_spec": self.benchmark_spec.to_document(),
            "schema_bundle": self.schema_bundle.to_document(),
            "evaluator": self.evaluator.to_document(),
            "manifest": self.manifest.to_document(),
        }
        # Omitting the optional extension preserves legacy v1 documents and their
        # checksums byte-for-byte.  New result-bound releases can opt in explicitly.
        if self.model_identity is not None:
            value["model_identity"] = self.model_identity.to_document()
        return value

    def compute_checksum(self) -> str:
        return canonical_sha256(self.unsigned_document())

    def with_checksum(self) -> "BenchmarkReleaseIdentity":
        return replace(self, release_checksum=self.compute_checksum())

    @property
    def is_release_ready(self) -> bool:
        components_ready = all(
            getattr(self, name).status == "verified"
            for name in RELEASE_COMPONENT_KINDS
        )
        model_ready = self.model_identity is None or self.model_identity.is_verified
        return components_ready and model_ready

    def validate(
        self,
        *,
        require_verified: bool = False,
        require_model_identity: bool = False,
    ) -> None:
        if self.release_checksum is None:
            raise IdentityError("release identity is missing release_checksum")
        if require_model_identity and self.model_identity is None:
            raise IdentityError("release identity is missing model_identity")
        if require_verified and not self.is_release_ready:
            pending = [
                name
                for name in RELEASE_COMPONENT_KINDS
                if getattr(self, name).status != "verified"
            ]
            if self.model_identity is not None and not self.model_identity.is_verified:
                pending.append("model_identity")
            raise IdentityError(f"release identity is not release-ready; pending={pending}")

    def to_document(self) -> dict[str, Any]:
        value = self.unsigned_document()
        value["release_checksum"] = self.release_checksum or self.compute_checksum()
        return value

    @classmethod
    def create(cls, **kwargs: Any) -> "BenchmarkReleaseIdentity":
        return cls(**kwargs).with_checksum()

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "BenchmarkReleaseIdentity":
        result = cls(
            schema=value.get("schema", ""),
            benchmark_version=value.get("benchmark_version", ""),
            protocol=ComponentIdentity.from_document(value.get("protocol") or {}),
            benchmark_spec=ComponentIdentity.from_document(
                value.get("benchmark_spec") or {}
            ),
            schema_bundle=ComponentIdentity.from_document(value.get("schema_bundle") or {}),
            evaluator=ComponentIdentity.from_document(value.get("evaluator") or {}),
            manifest=ComponentIdentity.from_document(value.get("manifest") or {}),
            model_identity=(
                ModelIdentity.from_document(value["model_identity"])
                if value.get("model_identity") is not None
                else None
            ),
            release_checksum=value.get("release_checksum"),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ArtifactRef:
    """Immutable reference to one artifact in a P1-17 lineage."""

    kind: str
    artifact_id: str
    sha256: str
    schema: Optional[str] = None
    version: Optional[str] = None
    uri: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = _require_nonempty(self.kind, "artifact.kind")
        if kind not in LINEAGE_KIND_ORDER:
            raise ArtifactLineageError(
                f"artifact.kind={kind!r}; expected one of {sorted(LINEAGE_KIND_ORDER)}"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "artifact_id", _require_nonempty(self.artifact_id, "artifact.artifact_id")
        )
        digest = _normalize_sha256(self.sha256, "artifact.sha256", required=True)
        object.__setattr__(self, "sha256", digest)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_document(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "schema": self.schema,
            "version": self.version,
            "uri": self.uri,
            "metadata": _jsonable(self.metadata),
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        return cls(
            kind=value.get("kind", ""),
            artifact_id=value.get("artifact_id", ""),
            sha256=value.get("sha256", ""),
            schema=value.get("schema"),
            version=value.get("version"),
            uri=value.get("uri"),
            metadata=value.get("metadata") or {},
        )


@dataclass(frozen=True)
class ArtifactLineage:
    """Ordered checkpoint -> adapter -> spec -> manifest -> metric -> result chain."""

    artifacts: Tuple[ArtifactRef, ...]
    lineage_checksum: Optional[str] = None
    schema: str = ARTIFACT_LINEAGE_SCHEMA
    model_identity: Optional[ModelIdentity] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if self.schema != ARTIFACT_LINEAGE_SCHEMA:
            raise ArtifactLineageError(
                f"lineage.schema must be {ARTIFACT_LINEAGE_SCHEMA!r}"
            )
        if self.model_identity is not None and not isinstance(
            self.model_identity, ModelIdentity
        ):
            raise ArtifactLineageError(
                "lineage.model_identity must be ModelIdentity or None"
            )
        if self.model_identity is not None:
            try:
                self.model_identity.validate()
            except IdentityError as exc:
                raise ArtifactLineageError(
                    f"lineage.model_identity is invalid: {exc}"
                ) from exc
        self._validate_order()
        checksum = _normalize_sha256(
            self.lineage_checksum,
            "lineage.lineage_checksum",
            required=False,
        )
        object.__setattr__(self, "lineage_checksum", checksum)
        if checksum is not None and checksum != self.compute_checksum():
            raise ArtifactLineageError(
                "lineage_checksum disagrees with artifact lineage content"
            )

    def _validate_order(self) -> None:
        if not self.artifacts:
            raise ArtifactLineageError("artifact lineage may not be empty")
        kinds = [item.kind for item in self.artifacts]
        for singleton in ("checkpoint", "adapter", "benchmark_spec", "manifest", "result"):
            if kinds.count(singleton) > 1:
                raise ArtifactLineageError(f"lineage contains duplicate {singleton!r}")
        if kinds[0] != "checkpoint":
            raise ArtifactLineageError("artifact lineage must start with checkpoint")
        transitions = {
            "checkpoint": {"adapter"},
            "adapter": {"benchmark_spec"},
            "benchmark_spec": {"manifest"},
            "manifest": {"metric"},
            "metric": {"metric", "result"},
            "result": set(),
        }
        for left, right in zip(kinds, kinds[1:]):
            if right not in transitions[left]:
                raise ArtifactLineageError(
                    "artifact lineage must follow checkpoint -> adapter -> "
                    "benchmark_spec -> manifest -> metric(s) -> result; "
                    f"observed invalid transition {left!r} -> {right!r}"
                )
        if "result" in kinds:
            required = {"checkpoint", "adapter", "benchmark_spec", "manifest", "metric"}
            missing = sorted(required - set(kinds))
            if missing:
                raise ArtifactLineageError(
                    f"a result lineage is incomplete; missing {missing}"
                )
            if kinds[-1] != "result":
                raise ArtifactLineageError("result must be the final lineage artifact")

    def unsigned_document(self) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "artifacts": [item.to_document() for item in self.artifacts],
            "edges": [
                {
                    "from_sha256": left.sha256,
                    "to_sha256": right.sha256,
                    "relation": f"{left.kind}_to_{right.kind}",
                }
                for left, right in zip(self.artifacts, self.artifacts[1:])
            ],
        }
        # As with BenchmarkReleaseIdentity, absence preserves legacy checksums.
        if self.model_identity is not None:
            value["model_identity"] = self.model_identity.to_document()
        return value

    def compute_checksum(self) -> str:
        return canonical_sha256(self.unsigned_document())

    def with_checksum(self) -> "ArtifactLineage":
        return replace(self, lineage_checksum=self.compute_checksum())

    def validate(
        self,
        *,
        require_complete: bool = False,
        require_model_identity: bool = False,
    ) -> None:
        if self.lineage_checksum is None:
            raise ArtifactLineageError("artifact lineage is missing lineage_checksum")
        if require_model_identity and self.model_identity is None:
            raise ArtifactLineageError("artifact lineage is missing model_identity")
        if require_complete:
            kinds = [item.kind for item in self.artifacts]
            required = {"checkpoint", "adapter", "benchmark_spec", "manifest", "metric", "result"}
            missing = sorted(required - set(kinds))
            if missing:
                raise ArtifactLineageError(f"complete lineage is missing {missing}")

    def assert_matches_release(self, release: BenchmarkReleaseIdentity) -> None:
        by_kind = {item.kind: item for item in self.artifacts if item.kind != "metric"}
        spec = by_kind.get("benchmark_spec")
        manifest = by_kind.get("manifest")
        if spec is not None and spec.sha256 != release.benchmark_spec.sha256:
            raise ArtifactLineageError(
                "lineage benchmark_spec digest differs from release identity"
            )
        if manifest is not None and manifest.sha256 != release.manifest.sha256:
            raise ArtifactLineageError(
                "lineage manifest digest differs from release identity"
            )
        if release.model_identity is not None:
            if self.model_identity is None:
                raise ArtifactLineageError(
                    "release identity includes model_identity but lineage does not"
                )
            if (
                self.model_identity.compute_checksum()
                != release.model_identity.compute_checksum()
            ):
                raise ArtifactLineageError(
                    "lineage model identity differs from release identity"
                )

    def to_document(self) -> dict[str, Any]:
        value = self.unsigned_document()
        value["lineage_checksum"] = self.lineage_checksum or self.compute_checksum()
        return value

    @classmethod
    def create(
        cls,
        artifacts: Iterable[ArtifactRef],
        *,
        model_identity: Optional[ModelIdentity] = None,
    ) -> "ArtifactLineage":
        return cls(tuple(artifacts), model_identity=model_identity).with_checksum()

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ArtifactLineage":
        artifacts = tuple(
            ArtifactRef.from_document(item) for item in value.get("artifacts") or ()
        )
        result = cls(
            artifacts=artifacts,
            model_identity=(
                ModelIdentity.from_document(value["model_identity"])
                if value.get("model_identity") is not None
                else None
            ),
            schema=value.get("schema", ""),
            lineage_checksum=value.get("lineage_checksum"),
        )
        result.validate()
        expected_edges = result.unsigned_document()["edges"]
        if value.get("edges") != expected_edges:
            raise ArtifactLineageError("lineage edges disagree with ordered artifacts")
        return result
