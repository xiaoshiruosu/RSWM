"""Structured identity for one benchmark baseline entry."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional

from benchmark.core.identity import (
    CheckpointIdentity,
    ComponentIdentity,
    ExecutionIdentity,
    ModelIdentity,
    canonical_sha256,
)

from .errors import BaselineIdentityError

BASELINE_IDENTITY_SCHEMA = "rswm.baseline_identity/v1"
_SHA256_HEX = frozenset("0123456789abcdef")


def _nonempty(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise BaselineIdentityError(f"{label} must be a non-empty string")
    return text


def _sha256(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise BaselineIdentityError(f"{label} must be a 64-character SHA-256 digest")
    return text


@dataclass(frozen=True)
class BaselineIdentity:
    """Bind model, adapter, checkpoint and execution identities without aliases."""

    baseline_id: str
    model_identity: ModelIdentity
    adapter_identity: ComponentIdentity
    checkpoint_identity: CheckpointIdentity
    execution_identity: ExecutionIdentity
    identity_checksum: Optional[str] = None
    schema: str = field(default=BASELINE_IDENTITY_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "baseline_id", _nonempty(self.baseline_id, "baseline_id")
        )
        if not isinstance(self.model_identity, ModelIdentity):
            raise BaselineIdentityError("model_identity must be ModelIdentity")
        if not isinstance(self.adapter_identity, ComponentIdentity):
            raise BaselineIdentityError("adapter_identity must be ComponentIdentity")
        if not isinstance(self.checkpoint_identity, CheckpointIdentity):
            raise BaselineIdentityError(
                "checkpoint_identity must be CheckpointIdentity"
            )
        if not isinstance(self.execution_identity, ExecutionIdentity):
            raise BaselineIdentityError("execution_identity must be ExecutionIdentity")
        self.model_identity.validate()
        if self.adapter_identity.kind != "baseline_adapter":
            raise BaselineIdentityError(
                "adapter_identity.kind must be 'baseline_adapter'"
            )
        if self.adapter_identity.artifact_id != self.model_identity.adapter_identity:
            raise BaselineIdentityError(
                "adapter identity differs from model_identity.adapter_identity"
            )
        if (
            self.checkpoint_identity.to_document()
            != self.model_identity.checkpoint_identity.to_document()
        ):
            raise BaselineIdentityError(
                "checkpoint identity differs from model_identity.checkpoint_identity"
            )
        if (
            self.execution_identity.to_document()
            != self.model_identity.execution_identity.to_document()
        ):
            raise BaselineIdentityError(
                "execution identity differs from model_identity.execution_identity"
            )
        checksum = self.identity_checksum
        if checksum is not None:
            checksum = _sha256(checksum, "identity_checksum")
            if checksum != self.compute_checksum():
                raise BaselineIdentityError(
                    "identity_checksum disagrees with baseline identity content"
                )
        object.__setattr__(self, "identity_checksum", checksum)

    def unsigned_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "baseline_id": self.baseline_id,
            "model_identity": self.model_identity.to_document(),
            "adapter_identity": self.adapter_identity.to_document(),
            "checkpoint_identity": self.checkpoint_identity.to_document(),
            "execution_identity": self.execution_identity.to_document(),
        }

    def compute_checksum(self) -> str:
        return canonical_sha256(self.unsigned_document())

    def with_checksum(self) -> "BaselineIdentity":
        return replace(self, identity_checksum=self.compute_checksum())

    def validate(self, *, require_verified: bool = False) -> None:
        if self.identity_checksum is None:
            raise BaselineIdentityError("baseline identity is missing identity_checksum")
        if self.identity_checksum != self.compute_checksum():
            raise BaselineIdentityError("baseline identity checksum mismatch")
        if require_verified:
            self.model_identity.validate(require_verified=True)
            if self.adapter_identity.status != "verified":
                raise BaselineIdentityError(
                    "verified baseline identity requires a verified adapter identity"
                )
            if self.checkpoint_identity.status != "verified":
                raise BaselineIdentityError(
                    "verified baseline identity requires a verified checkpoint identity"
                )

    def to_document(self) -> dict[str, Any]:
        value = self.unsigned_document()
        value["identity_checksum"] = self.identity_checksum or self.compute_checksum()
        return value

    @classmethod
    def create(cls, **kwargs: Any) -> "BaselineIdentity":
        return cls(**kwargs).with_checksum()

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "BaselineIdentity":
        if value.get("schema") != BASELINE_IDENTITY_SCHEMA:
            raise BaselineIdentityError(
                f"baseline identity schema must be {BASELINE_IDENTITY_SCHEMA!r}"
            )
        result = cls(
            baseline_id=value.get("baseline_id", ""),
            model_identity=ModelIdentity.from_document(
                _mapping(value.get("model_identity"), "model_identity")
            ),
            adapter_identity=ComponentIdentity.from_document(
                _mapping(value.get("adapter_identity"), "adapter_identity")
            ),
            checkpoint_identity=CheckpointIdentity.from_document(
                _mapping(value.get("checkpoint_identity"), "checkpoint_identity")
            ),
            execution_identity=ExecutionIdentity.from_document(
                _mapping(value.get("execution_identity"), "execution_identity")
            ),
            identity_checksum=value.get("identity_checksum"),
        )
        result.validate()
        return result


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineIdentityError(f"{label} must be an object")
    return value
