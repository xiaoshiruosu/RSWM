"""Immutable Baseline Card v2 overlay and sequential admission state machine."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from benchmark.core.identity import canonical_sha256
from benchmark.core.types import CapabilityPartition

from .capability import BaselineCapabilities
from .errors import AdmissionStateError, BaselineError
from .identity import BaselineIdentity

BASELINE_CARD_OVERLAY_SCHEMA = "rswm.baseline_card_overlay/v2"
LEGACY_CARD_SCHEMA = "rsroll.baseline_card/v1"
_SHA256_HEX = frozenset("0123456789abcdef")


class AdmissionState(str, Enum):
    DECLARED = "DECLARED"
    ADAPTER_AVAILABLE = "ADAPTER_AVAILABLE"
    CHECKPOINT_VERIFIED = "CHECKPOINT_VERIFIED"
    RUNNABLE = "RUNNABLE"
    EVALUATED = "EVALUATED"
    ADMITTED = "ADMITTED"

    def __str__(self) -> str:
        return self.value


ADMISSION_SEQUENCE = tuple(AdmissionState)
_EVIDENCE_KIND = {
    AdmissionState.ADAPTER_AVAILABLE: "adapter",
    AdmissionState.CHECKPOINT_VERIFIED: "checkpoint",
    AdmissionState.RUNNABLE: "runtime_probe",
    AdmissionState.EVALUATED: "result",
    AdmissionState.ADMITTED: "admission_decision",
}


@dataclass(frozen=True)
class LegacyCardReference:
    name: str
    sha256: str
    relative_path: str
    schema: str = LEGACY_CARD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "legacy_card.name"))
        object.__setattr__(
            self, "sha256", _sha256(self.sha256, "legacy_card.sha256")
        )
        object.__setattr__(
            self,
            "relative_path",
            _nonempty(self.relative_path, "legacy_card.relative_path"),
        )
        if self.schema != LEGACY_CARD_SCHEMA:
            raise BaselineError(
                f"legacy card schema must remain {LEGACY_CARD_SCHEMA!r}"
            )

    def to_document(self) -> dict[str, str]:
        return {
            "schema": self.schema,
            "name": self.name,
            "sha256": self.sha256,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "LegacyCardReference":
        return cls(
            schema=value.get("schema", ""),
            name=value.get("name", ""),
            sha256=value.get("sha256", ""),
            relative_path=value.get("relative_path", ""),
        )


@dataclass(frozen=True)
class StateEvidence:
    target_state: AdmissionState
    kind: str
    artifact_id: str
    sha256: str
    note: Optional[str] = None

    def __post_init__(self) -> None:
        state = AdmissionState(self.target_state)
        if state is AdmissionState.DECLARED:
            raise AdmissionStateError("DECLARED is the initial state, not a transition")
        object.__setattr__(self, "target_state", state)
        kind = _nonempty(self.kind, "evidence.kind")
        expected = _EVIDENCE_KIND[state]
        if kind != expected:
            raise AdmissionStateError(
                f"{state.value} requires evidence kind {expected!r}, got {kind!r}"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self, "artifact_id", _nonempty(self.artifact_id, "evidence.artifact_id")
        )
        object.__setattr__(self, "sha256", _sha256(self.sha256, "evidence.sha256"))
        if self.note is not None:
            object.__setattr__(self, "note", _nonempty(self.note, "evidence.note"))

    def to_document(self) -> dict[str, Any]:
        return {
            "target_state": self.target_state.value,
            "kind": self.kind,
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "note": self.note,
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "StateEvidence":
        return cls(
            target_state=value.get("target_state", ""),
            kind=value.get("kind", ""),
            artifact_id=value.get("artifact_id", ""),
            sha256=value.get("sha256", ""),
            note=value.get("note"),
        )


@dataclass(frozen=True)
class BaselineCardV2:
    """Additive overlay; the referenced v1 card remains immutable and valid."""

    baseline_identity: BaselineIdentity
    capabilities: BaselineCapabilities
    legacy_card: LegacyCardReference
    capability_partition: Optional[CapabilityPartition] = None
    state: AdmissionState = AdmissionState.DECLARED
    state_history: Tuple[StateEvidence, ...] = ()
    overlay_checksum: Optional[str] = None
    schema: str = field(default=BASELINE_CARD_OVERLAY_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_identity, BaselineIdentity):
            raise BaselineError("baseline_identity must be BaselineIdentity")
        self.baseline_identity.validate()
        if not isinstance(self.capabilities, BaselineCapabilities):
            raise BaselineError("capabilities must be BaselineCapabilities")
        if not isinstance(self.legacy_card, LegacyCardReference):
            raise BaselineError("legacy_card must be LegacyCardReference")
        if self.legacy_card.name != self.baseline_identity.baseline_id:
            raise BaselineError(
                "legacy card name differs from baseline_identity.baseline_id"
            )
        if (
            self.capabilities.checkpoint_identity.to_document()
            != self.baseline_identity.checkpoint_identity.to_document()
        ):
            raise BaselineError(
                "capability checkpoint differs from baseline identity checkpoint"
            )
        partition = self.capability_partition
        if partition is not None:
            try:
                partition = CapabilityPartition(partition)
            except ValueError as exc:
                raise BaselineError(str(exc)) from exc
        object.__setattr__(self, "capability_partition", partition)
        state = AdmissionState(self.state)
        history = tuple(self.state_history)
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "state_history", history)
        self._validate_history()
        self._validate_current_state()
        checksum = self.overlay_checksum
        if checksum is not None:
            checksum = _sha256(checksum, "overlay_checksum")
            if checksum != self.compute_checksum():
                raise BaselineError(
                    "overlay_checksum disagrees with baseline card content"
                )
        object.__setattr__(self, "overlay_checksum", checksum)

    def _validate_history(self) -> None:
        state_index = ADMISSION_SEQUENCE.index(self.state)
        if len(self.state_history) != state_index:
            raise AdmissionStateError(
                f"state={self.state.value} requires {state_index} sequential evidence "
                f"records, got {len(self.state_history)}"
            )
        expected_states = ADMISSION_SEQUENCE[1 : state_index + 1]
        observed_states = tuple(item.target_state for item in self.state_history)
        if observed_states != expected_states:
            raise AdmissionStateError(
                "admission history must follow DECLARED -> ADAPTER_AVAILABLE -> "
                "CHECKPOINT_VERIFIED -> RUNNABLE -> EVALUATED -> ADMITTED"
            )

    def _validate_current_state(self) -> None:
        index = ADMISSION_SEQUENCE.index(self.state)
        if index >= ADMISSION_SEQUENCE.index(AdmissionState.ADAPTER_AVAILABLE):
            if self.baseline_identity.adapter_identity.status != "verified":
                raise AdmissionStateError(
                    "ADAPTER_AVAILABLE requires verified adapter identity"
                )
        if index >= ADMISSION_SEQUENCE.index(AdmissionState.CHECKPOINT_VERIFIED):
            if self.baseline_identity.checkpoint_identity.status != "verified":
                raise AdmissionStateError(
                    "CHECKPOINT_VERIFIED requires verified checkpoint identity"
                )
        if index >= ADMISSION_SEQUENCE.index(AdmissionState.RUNNABLE):
            if self.capability_partition is None:
                raise AdmissionStateError("RUNNABLE requires a resolved partition")
            if not self.capabilities.tasks or not self.capabilities.tracks:
                raise AdmissionStateError("RUNNABLE requires declared tasks and tracks")
            if not self.capabilities.runtime.is_resolved:
                raise AdmissionStateError("RUNNABLE requires resolved runtime identity")
            self.capabilities.assert_partition(self.capability_partition)
        if self.state is AdmissionState.ADMITTED:
            self._assert_admission_ready()

    def _assert_admission_ready(self) -> None:
        self.baseline_identity.validate(require_verified=True)
        if self.capability_partition is None:
            raise AdmissionStateError("ADMITTED requires a resolved partition")
        if self.capabilities.uses_future_gt:
            raise AdmissionStateError("future-GT methods cannot be ADMITTED")
        if not self.capabilities.is_admission_disclosed:
            raise AdmissionStateError(
                "ADMITTED requires complete license, external-data, runtime and "
                "capability disclosures"
            )
        self.capabilities.assert_partition(self.capability_partition)

    def next_state(self) -> Optional[AdmissionState]:
        index = ADMISSION_SEQUENCE.index(self.state)
        if index + 1 == len(ADMISSION_SEQUENCE):
            return None
        return ADMISSION_SEQUENCE[index + 1]

    def advance(self, evidence: StateEvidence) -> "BaselineCardV2":
        expected = self.next_state()
        if expected is None:
            raise AdmissionStateError("ADMITTED is terminal")
        if evidence.target_state is not expected:
            raise AdmissionStateError(
                f"cannot transition {self.state.value} -> {evidence.target_state.value}; "
                f"next required state is {expected.value}"
            )
        return replace(
            self,
            state=expected,
            state_history=self.state_history + (evidence,),
            overlay_checksum=None,
        ).with_checksum()

    def assert_admitted_for(
        self,
        *,
        task_id,
        track_id,
        partition: CapabilityPartition,
    ) -> None:
        if self.state is not AdmissionState.ADMITTED:
            raise AdmissionStateError(
                f"baseline state={self.state.value}; ADMITTED is required"
            )
        if self.capability_partition is not CapabilityPartition(partition):
            raise AdmissionStateError(
                "requested partition differs from admitted card partition"
            )
        self.capabilities.assert_eligible(
            task_id=task_id,
            track_id=track_id,
            partition=partition,
        )

    def unsigned_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "baseline_identity": self.baseline_identity.to_document(),
            "capabilities": self.capabilities.to_document(),
            "legacy_card": self.legacy_card.to_document(),
            "capability_partition": (
                self.capability_partition.value
                if self.capability_partition is not None
                else "UNKNOWN"
            ),
            "state": self.state.value,
            "state_history": [item.to_document() for item in self.state_history],
        }

    def compute_checksum(self) -> str:
        return canonical_sha256(self.unsigned_document())

    def with_checksum(self) -> "BaselineCardV2":
        return replace(self, overlay_checksum=self.compute_checksum())

    def validate(self) -> None:
        if self.overlay_checksum is None:
            raise BaselineError("baseline card overlay is missing overlay_checksum")
        if self.overlay_checksum != self.compute_checksum():
            raise BaselineError("baseline card overlay checksum mismatch")

    def to_document(self) -> dict[str, Any]:
        value = self.unsigned_document()
        value["overlay_checksum"] = self.overlay_checksum or self.compute_checksum()
        return value

    @classmethod
    def create_declared(
        cls,
        *,
        baseline_identity: BaselineIdentity,
        capabilities: BaselineCapabilities,
        legacy_card: LegacyCardReference,
        capability_partition: Optional[CapabilityPartition],
    ) -> "BaselineCardV2":
        return cls(
            baseline_identity=baseline_identity,
            capabilities=capabilities,
            legacy_card=legacy_card,
            capability_partition=capability_partition,
        ).with_checksum()

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "BaselineCardV2":
        if value.get("schema") != BASELINE_CARD_OVERLAY_SCHEMA:
            raise BaselineError(
                f"baseline card schema must be {BASELINE_CARD_OVERLAY_SCHEMA!r}"
            )
        partition_value = value.get("capability_partition", "UNKNOWN")
        partition = (
            None
            if partition_value == "UNKNOWN"
            else CapabilityPartition(partition_value)
        )
        result = cls(
            baseline_identity=BaselineIdentity.from_document(
                _mapping(value.get("baseline_identity"), "baseline_identity")
            ),
            capabilities=BaselineCapabilities.from_document(
                _mapping(value.get("capabilities"), "capabilities")
            ),
            legacy_card=LegacyCardReference.from_document(
                _mapping(value.get("legacy_card"), "legacy_card")
            ),
            capability_partition=partition,
            state=value.get("state", ""),
            state_history=tuple(
                StateEvidence.from_document(_mapping(item, "state_history item"))
                for item in value.get("state_history") or ()
            ),
            overlay_checksum=value.get("overlay_checksum"),
        )
        result.validate()
        return result


def _nonempty(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise BaselineError(f"{label} must be a non-empty string")
    return text


def _sha256(value: Any, label: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(character not in _SHA256_HEX for character in text):
        raise BaselineError(f"{label} must be a 64-character SHA-256 digest")
    return text


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BaselineError(f"{label} must be an object")
    return value
