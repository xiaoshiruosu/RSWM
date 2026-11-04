"""Content-bound M7 result bundle and per-sample identity.

The result payload is hashed before the final result artifact is appended to
the P1-17 lineage.  This avoids a circular hash while still binding the full
``checkpoint -> adapter -> benchmark spec -> manifest -> metric -> result``
chain into a second bundle checksum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional, Tuple

from benchmark.core.identity import (
    ArtifactLineage,
    ArtifactRef,
    BenchmarkReleaseIdentity,
    ExecutionIdentity,
    ModelIdentity,
    canonical_sha256,
)
from benchmark.core.types import ModelInput, TaskId, TrackId

from .errors import EvaluationError, InvalidMetricResultError
from .result import MetricResult

SAMPLE_RESULT_IDENTITY_SCHEMA = "rswm.sample_result_identity/v1"
RESULT_BUNDLE_SCHEMA = "rswm.result_bundle/v1"
M7_RESULT_PURPOSE = "cpu_end_to_end_dry_run"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _digest(value: Any, label: str) -> str:
    text = str(value).lower()
    if not _SHA256_RE.fullmatch(text):
        raise EvaluationError(f"{label} must be a 64-character SHA-256 digest")
    return text


def _nonempty(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise EvaluationError(f"{label} must be non-empty")
    return text


@dataclass(frozen=True)
class SampleIdentity:
    """Public, manifest-bound identity of one evaluated generation request."""

    benchmark_version: str
    task_id: TaskId
    track_id: TrackId
    sample_id: str
    manifest_checksum: str
    condition_checksum: str
    generation_seed: int
    horizon: int
    identity_checksum: Optional[str] = None
    schema: str = field(default=SAMPLE_RESULT_IDENTITY_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "benchmark_version", _nonempty(self.benchmark_version, "benchmark_version")
        )
        object.__setattr__(self, "task_id", TaskId(self.task_id))
        object.__setattr__(self, "track_id", TrackId(self.track_id))
        object.__setattr__(self, "sample_id", _nonempty(self.sample_id, "sample_id"))
        object.__setattr__(
            self,
            "manifest_checksum",
            _digest(self.manifest_checksum, "manifest_checksum"),
        )
        object.__setattr__(
            self,
            "condition_checksum",
            _digest(self.condition_checksum, "condition_checksum"),
        )
        object.__setattr__(self, "generation_seed", int(self.generation_seed))
        horizon = int(self.horizon)
        if horizon not in (17, 25, 33):
            raise EvaluationError("sample identity horizon must be 17, 25 or 33")
        object.__setattr__(self, "horizon", horizon)
        checksum = self.identity_checksum
        if checksum is not None:
            checksum = _digest(checksum, "sample identity checksum")
            if checksum != self.compute_checksum():
                raise EvaluationError(
                    "sample identity checksum disagrees with identity content"
                )
        object.__setattr__(self, "identity_checksum", checksum)

    def unsigned_document(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "benchmark_version": self.benchmark_version,
            "task_id": self.task_id.value,
            "track_id": self.track_id.value,
            "sample_id": self.sample_id,
            "manifest_checksum": self.manifest_checksum,
            "condition_checksum": self.condition_checksum,
            "generation_seed": self.generation_seed,
            "horizon": self.horizon,
        }

    def compute_checksum(self) -> str:
        return canonical_sha256(self.unsigned_document())

    def with_checksum(self) -> "SampleIdentity":
        return replace(self, identity_checksum=self.compute_checksum())

    def validate(self) -> None:
        if self.identity_checksum is None:
            raise EvaluationError("sample identity is missing identity_checksum")
        if self.identity_checksum != self.compute_checksum():
            raise EvaluationError("sample identity checksum mismatch")

    def to_document(self) -> dict[str, Any]:
        value = self.unsigned_document()
        value["identity_checksum"] = self.identity_checksum or self.compute_checksum()
        return value

    @classmethod
    def from_model_input(
        cls,
        model_input: ModelInput,
        *,
        manifest_checksum: str,
    ) -> "SampleIdentity":
        metadata = model_input.metadata
        return cls(
            benchmark_version=metadata.benchmark_version,
            task_id=metadata.task_id,
            track_id=metadata.track_id,
            sample_id=metadata.sample_id,
            manifest_checksum=manifest_checksum,
            condition_checksum=metadata.condition_checksum,
            generation_seed=metadata.generation_seed,
            horizon=metadata.horizon,
        ).with_checksum()

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "SampleIdentity":
        if value.get("schema") != SAMPLE_RESULT_IDENTITY_SCHEMA:
            raise EvaluationError(
                f"sample identity schema must be {SAMPLE_RESULT_IDENTITY_SCHEMA!r}"
            )
        result = cls(
            benchmark_version=value.get("benchmark_version", ""),
            task_id=value.get("task_id", ""),
            track_id=value.get("track_id", ""),
            sample_id=value.get("sample_id", ""),
            manifest_checksum=value.get("manifest_checksum", ""),
            condition_checksum=value.get("condition_checksum", ""),
            generation_seed=value.get("generation_seed", 0),
            horizon=value.get("horizon", 0),
            identity_checksum=value.get("identity_checksum"),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ResultBundle:
    """Standard, serializable M7 result object.

    ``valid`` means all metric contracts succeeded.  M7 bundles remain
    explicitly ineligible for leaderboard or headline admission because they
    are produced by a mock CPU dry run.
    """

    benchmark_identity: BenchmarkReleaseIdentity
    model_identity: ModelIdentity
    adapter_identity: str
    execution_identity: ExecutionIdentity
    sample_identity: SampleIdentity
    metric_results: Tuple[MetricResult, ...]
    artifact_lineage: ArtifactLineage
    result_payload_checksum: str
    bundle_checksum: Optional[str] = None
    purpose: str = M7_RESULT_PURPOSE
    schema: str = field(default=RESULT_BUNDLE_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.benchmark_identity, BenchmarkReleaseIdentity):
            raise EvaluationError(
                "benchmark_identity must be BenchmarkReleaseIdentity"
            )
        self.benchmark_identity.validate()
        if not isinstance(self.model_identity, ModelIdentity):
            raise EvaluationError("model_identity must be ModelIdentity")
        self.model_identity.validate()
        adapter_identity = _nonempty(self.adapter_identity, "adapter_identity")
        object.__setattr__(self, "adapter_identity", adapter_identity)
        if adapter_identity != self.model_identity.adapter_identity:
            raise EvaluationError("adapter identity differs from model identity")
        if not isinstance(self.execution_identity, ExecutionIdentity):
            raise EvaluationError("execution_identity must be ExecutionIdentity")
        if (
            self.execution_identity.to_document()
            != self.model_identity.execution_identity.to_document()
        ):
            raise EvaluationError("execution identity differs from model identity")
        if not isinstance(self.sample_identity, SampleIdentity):
            raise EvaluationError("sample_identity must be SampleIdentity")
        self.sample_identity.validate()
        if self.sample_identity.benchmark_version != self.benchmark_identity.benchmark_version:
            raise EvaluationError("sample benchmark version differs from release identity")
        if (
            self.sample_identity.manifest_checksum
            != self.benchmark_identity.manifest.sha256
        ):
            raise EvaluationError("sample manifest checksum differs from release identity")
        if self.purpose != M7_RESULT_PURPOSE:
            raise EvaluationError(f"M7 result purpose must be {M7_RESULT_PURPOSE!r}")

        results = tuple(self.metric_results)
        if not results or any(not isinstance(item, MetricResult) for item in results):
            raise EvaluationError("metric_results must contain MetricResult values")
        keys = [
            (item.metric_identity.metric_id, item.metric_identity.version)
            for item in results
        ]
        if len(set(keys)) != len(keys):
            raise EvaluationError("result bundle metric identities must be unique")
        object.__setattr__(self, "metric_results", results)

        if not isinstance(self.artifact_lineage, ArtifactLineage):
            raise EvaluationError("artifact_lineage must be ArtifactLineage")
        self.artifact_lineage.validate(
            require_complete=True,
            require_model_identity=True,
        )
        self.artifact_lineage.assert_matches_release(self.benchmark_identity)
        if (
            self.artifact_lineage.model_identity.compute_checksum()
            != self.model_identity.compute_checksum()
        ):
            raise EvaluationError("lineage model identity differs from result bundle")
        self._validate_lineage_bindings()

        payload_checksum = _digest(
            self.result_payload_checksum, "result_payload_checksum"
        )
        object.__setattr__(self, "result_payload_checksum", payload_checksum)
        if payload_checksum != self.compute_result_payload_checksum():
            raise EvaluationError("result payload checksum mismatch")
        result_artifact = self.artifact_lineage.artifacts[-1]
        if result_artifact.sha256 != payload_checksum:
            raise EvaluationError("lineage result digest differs from result payload")
        if result_artifact.artifact_id != self.bundle_id:
            raise EvaluationError("lineage result artifact_id differs from bundle_id")

        checksum = self.bundle_checksum
        if checksum is not None:
            checksum = _digest(checksum, "bundle_checksum")
            if checksum != self.compute_bundle_checksum():
                raise EvaluationError("bundle checksum disagrees with result content")
        object.__setattr__(self, "bundle_checksum", checksum)

    @property
    def valid(self) -> bool:
        return all(item.valid for item in self.metric_results)

    @property
    def invalid_reasons(self) -> Tuple[str, ...]:
        return tuple(
            f"{item.metric_identity.metric_id}:{item.reason}"
            for item in self.metric_results
            if not item.valid
        )

    @property
    def admission_eligible(self) -> bool:
        return False

    @property
    def bundle_id(self) -> str:
        return (
            f"{self.sample_identity.sample_id}:"
            f"{self.result_payload_checksum[:16]}"
        )

    @staticmethod
    def _payload_document(
        *,
        benchmark_identity: BenchmarkReleaseIdentity,
        model_identity: ModelIdentity,
        adapter_identity: str,
        execution_identity: ExecutionIdentity,
        sample_identity: SampleIdentity,
        metric_results: Tuple[MetricResult, ...],
    ) -> dict[str, Any]:
        valid = all(item.valid for item in metric_results)
        invalid_reasons = [
            f"{item.metric_identity.metric_id}:{item.reason}"
            for item in metric_results
            if not item.valid
        ]
        return {
            "schema": RESULT_BUNDLE_SCHEMA,
            "purpose": M7_RESULT_PURPOSE,
            "admission_eligible": False,
            "benchmark_identity": benchmark_identity.to_document(),
            "model_identity": model_identity.to_document(),
            "adapter_identity": adapter_identity,
            "execution_identity": execution_identity.to_document(),
            "sample_identity": sample_identity.to_document(),
            "metric_results": [item.to_document() for item in metric_results],
            "valid": valid,
            "invalid_reasons": invalid_reasons,
        }

    def result_payload_document(self) -> dict[str, Any]:
        return self._payload_document(
            benchmark_identity=self.benchmark_identity,
            model_identity=self.model_identity,
            adapter_identity=self.adapter_identity,
            execution_identity=self.execution_identity,
            sample_identity=self.sample_identity,
            metric_results=self.metric_results,
        )

    def compute_result_payload_checksum(self) -> str:
        return canonical_sha256(self.result_payload_document())

    def unsigned_document(self) -> dict[str, Any]:
        value = self.result_payload_document()
        value.update(
            {
                "bundle_id": self.bundle_id,
                "result_payload_checksum": self.result_payload_checksum,
                "artifact_lineage": self.artifact_lineage.to_document(),
            }
        )
        return value

    def compute_bundle_checksum(self) -> str:
        return canonical_sha256(self.unsigned_document())

    def with_checksum(self) -> "ResultBundle":
        return replace(self, bundle_checksum=self.compute_bundle_checksum())

    def validate(self, *, require_valid: bool = False) -> None:
        if self.bundle_checksum is None:
            raise EvaluationError("result bundle is missing bundle_checksum")
        if self.bundle_checksum != self.compute_bundle_checksum():
            raise EvaluationError("result bundle checksum mismatch")
        if require_valid and not self.valid:
            raise InvalidMetricResultError(self.metric_results)

    def assert_admission_eligible(self) -> None:
        self.validate(require_valid=True)
        raise EvaluationError(
            "CPU end-to-end dry-run bundles are not leaderboard/headline eligible"
        )

    def _validate_lineage_bindings(self) -> None:
        artifacts = self.artifact_lineage.artifacts
        expected_kinds = (
            "checkpoint",
            "adapter",
            "benchmark_spec",
            "manifest",
            *("metric" for _ in self.metric_results),
            "result",
        )
        if tuple(item.kind for item in artifacts) != expected_kinds:
            raise EvaluationError("artifact lineage kinds do not match result bundle")
        checkpoint_ref, adapter_ref = artifacts[:2]
        checkpoint = self.model_identity.checkpoint_identity
        if (
            checkpoint_ref.artifact_id != checkpoint.repository
            and checkpoint_ref.metadata.get("repository") != checkpoint.repository
        ):
            raise EvaluationError("lineage checkpoint does not bind model repository")
        if (
            checkpoint.sha256_manifest is not None
            and checkpoint_ref.sha256 != checkpoint.sha256_manifest
        ):
            raise EvaluationError("lineage checkpoint digest differs from model identity")
        if adapter_ref.artifact_id != self.adapter_identity:
            raise EvaluationError("lineage adapter artifact differs from adapter identity")
        expected_metrics = tuple(
            result.metric_identity.artifact_ref().to_document()
            for result in self.metric_results
        )
        observed_metrics = tuple(
            item.to_document() for item in artifacts[4:-1]
        )
        if observed_metrics != expected_metrics:
            raise EvaluationError("lineage metric artifacts differ from metric results")

    def to_document(self) -> dict[str, Any]:
        value = self.unsigned_document()
        value["bundle_checksum"] = self.bundle_checksum or self.compute_bundle_checksum()
        return value

    @classmethod
    def create(
        cls,
        *,
        benchmark_identity: BenchmarkReleaseIdentity,
        model_identity: ModelIdentity,
        adapter_identity: str,
        execution_identity: ExecutionIdentity,
        sample_identity: SampleIdentity,
        metric_results: Iterable[MetricResult],
        checkpoint_artifact: ArtifactRef,
        adapter_artifact: ArtifactRef,
        fail_closed: bool = True,
    ) -> "ResultBundle":
        results = tuple(metric_results)
        if not results:
            raise EvaluationError("cannot create a result bundle without metrics")
        if fail_closed and any(not item.valid for item in results):
            raise InvalidMetricResultError(results)
        if checkpoint_artifact.kind != "checkpoint":
            raise EvaluationError("checkpoint_artifact must have kind='checkpoint'")
        if adapter_artifact.kind != "adapter":
            raise EvaluationError("adapter_artifact must have kind='adapter'")
        benchmark_identity.validate()
        sample_identity.validate()
        payload = cls._payload_document(
            benchmark_identity=benchmark_identity,
            model_identity=model_identity,
            adapter_identity=adapter_identity,
            execution_identity=execution_identity,
            sample_identity=sample_identity,
            metric_results=results,
        )
        payload_checksum = canonical_sha256(payload)
        result_id = f"{sample_identity.sample_id}:{payload_checksum[:16]}"
        spec = benchmark_identity.benchmark_spec
        manifest = benchmark_identity.manifest
        if spec.sha256 is None or manifest.sha256 is None:
            raise EvaluationError(
                "benchmark spec and manifest require verified digests for lineage"
            )
        metric_artifacts = tuple(
            result.metric_identity.artifact_ref() for result in results
        )
        lineage = ArtifactLineage.create(
            (
                checkpoint_artifact,
                adapter_artifact,
                ArtifactRef(
                    kind="benchmark_spec",
                    artifact_id=spec.artifact_id,
                    sha256=spec.sha256,
                    schema=spec.schema,
                    version=spec.version,
                ),
                ArtifactRef(
                    kind="manifest",
                    artifact_id=manifest.artifact_id,
                    sha256=manifest.sha256,
                    schema=manifest.schema,
                    version=manifest.version,
                ),
                *metric_artifacts,
                ArtifactRef(
                    kind="result",
                    artifact_id=result_id,
                    sha256=payload_checksum,
                    schema=RESULT_BUNDLE_SCHEMA,
                    version="v1",
                    metadata={
                        "purpose": M7_RESULT_PURPOSE,
                        "admission_eligible": False,
                    },
                ),
            ),
            model_identity=model_identity,
        )
        return cls(
            benchmark_identity=benchmark_identity,
            model_identity=model_identity,
            adapter_identity=adapter_identity,
            execution_identity=execution_identity,
            sample_identity=sample_identity,
            metric_results=results,
            artifact_lineage=lineage,
            result_payload_checksum=payload_checksum,
        ).with_checksum()

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ResultBundle":
        if value.get("schema") != RESULT_BUNDLE_SCHEMA:
            raise EvaluationError(
                f"result bundle schema must be {RESULT_BUNDLE_SCHEMA!r}"
            )
        result = cls(
            benchmark_identity=BenchmarkReleaseIdentity.from_document(
                value.get("benchmark_identity") or {}
            ),
            model_identity=ModelIdentity.from_document(
                value.get("model_identity") or {}
            ),
            adapter_identity=value.get("adapter_identity", ""),
            execution_identity=ExecutionIdentity.from_document(
                value.get("execution_identity") or {}
            ),
            sample_identity=SampleIdentity.from_document(
                value.get("sample_identity") or {}
            ),
            metric_results=tuple(
                MetricResult.from_document(item)
                for item in value.get("metric_results") or ()
            ),
            artifact_lineage=ArtifactLineage.from_document(
                value.get("artifact_lineage") or {}
            ),
            result_payload_checksum=value.get("result_payload_checksum", ""),
            bundle_checksum=value.get("bundle_checksum"),
            purpose=value.get("purpose", ""),
        )
        if value.get("bundle_id") != result.bundle_id:
            raise EvaluationError("serialized bundle_id is inconsistent")
        if bool(value.get("valid", False)) != result.valid:
            raise EvaluationError("serialized result validity is inconsistent")
        if list(value.get("invalid_reasons") or ()) != list(result.invalid_reasons):
            raise EvaluationError("serialized invalid reasons are inconsistent")
        if bool(value.get("admission_eligible", True)) is not False:
            raise EvaluationError("M7 result must remain admission_eligible=false")
        result.validate()
        return result
