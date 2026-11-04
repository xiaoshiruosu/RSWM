"""Model-independent benchmark task contract."""

from __future__ import annotations

import re
from abc import ABC
from typing import Any, Protocol

from benchmark.core.contracts import validate_model_input, validate_rollout
from benchmark.core.errors import ContractError
from benchmark.core.types import (
    BenchmarkSample,
    GenerationMetadata,
    GroundTruthBundle,
    ModelInput,
    PromptSchema,
    Rollout,
    TaskId,
    TrackId,
    TrajectoryCondition,
    ValidationReport,
)
from models.world_model import WorldModelCapabilities

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class EvaluationDelegate(Protocol):
    """Narrow seam for the future M4 EvaluationRunner."""

    def evaluate(
        self,
        *,
        task_id: TaskId,
        track_id: TrackId,
        rollout: Rollout,
        ground_truth: GroundTruthBundle,
        sample: BenchmarkSample,
    ) -> Any:
        ...


class BaseTask(ABC):
    task_id: TaskId
    protocol_version = "RSWM-Bench-Protocol-V1"

    def __init__(self, *, benchmark_version: str = "RSWM-Bench-v1"):
        if not benchmark_version:
            raise ValueError("benchmark_version must be non-empty")
        self.benchmark_version = str(benchmark_version)

    def validate_sample(self, sample: BenchmarkSample) -> None:
        if sample.horizon not in (17, 25, 33):
            raise ContractError(
                f"{self.task_id.value}: sample horizon must be 17/25/33, got {sample.horizon}"
            )
        if sample.trajectory.frame_count != sample.horizon:
            raise ContractError("sample trajectory and horizon differ")

    def prepare_input(
        self,
        sample: BenchmarkSample,
        track: TrackId,
        model_capabilities: WorldModelCapabilities,
        *,
        generation_seed: int = 42,
    ) -> ModelInput:
        self.validate_sample(sample)
        track = TrackId(track)
        if track == TrackId.C_MULTIVIEW:
            raise ContractError("C_multiview is reserved and not implemented in M1-M3")
        prompt_raw = sample.metadata.get("prompt")
        schema_raw = sample.metadata.get("prompt_schema", PromptSchema.NET_CONTINUOUS.value)
        if not isinstance(prompt_raw, str) or not prompt_raw.strip():
            raise ContractError(f"{sample.id}: missing public scene prompt")
        try:
            prompt_schema = PromptSchema(str(schema_raw))
        except ValueError as exc:
            raise ContractError(f"{sample.id}: unknown prompt_schema={schema_raw!r}") from exc
        if prompt_schema != PromptSchema.NET_CONTINUOUS:
            raise ContractError(
                f"{sample.id}: V1 official tasks require prompt_schema=net_continuous; "
                f"legacy {prompt_schema.value!r} may not be relabeled"
            )
        model_capabilities.assert_eligible(
            task_id=self.task_id,
            track_id=track,
            prompt_schema=prompt_schema,
            horizon=sample.horizon,
        )
        poses = intrinsics = None
        if track == TrackId.B_CAMERA:
            poses = sample.trajectory.poses
            intrinsics = sample.trajectory.intrinsics
            if poses is None or intrinsics is None:
                raise ContractError(f"{sample.id}: Track B requires poses and intrinsics")
        condition = TrajectoryCondition(
            track_id=track,
            scene_prompt=prompt_raw,
            prompt_schema=prompt_schema,
            segment_count=sample.trajectory.segment_count,
            net_camera_bearing_deg=sample.direction.camera_bearing_deg,
            frame_count=sample.horizon,
            poses=poses,
            intrinsics=intrinsics,
            convention_version=sample.direction.convention_version,
        )
        checksum = str(sample.metadata.get("condition_checksum", ""))
        if not _SHA256_RE.fullmatch(checksum):
            raise ContractError(f"{sample.id}: condition_checksum is missing or invalid")
        public_metadata = {
            key: sample.metadata[key]
            for key in ("source_category", "scene", "data_domain")
            if key in sample.metadata
        }
        metadata = GenerationMetadata(
            benchmark_version=self.benchmark_version,
            task_id=self.task_id,
            track_id=track,
            sample_id=sample.id,
            horizon=sample.horizon,
            output_size=tuple(int(value) for value in sample.image.shape[:2]),
            generation_seed=int(generation_seed),
            condition_checksum=checksum.lower(),
            public=public_metadata,
        )
        result = ModelInput(
            condition_image=sample.image,
            trajectory_condition=condition,
            metadata=metadata,
        )
        validate_model_input(result)
        return result

    def validate_output(
        self,
        rollout: Rollout,
        model_input: ModelInput,
    ) -> ValidationReport:
        if model_input.metadata.task_id != self.task_id:
            raise ContractError("model input belongs to a different task")
        return validate_rollout(rollout, model_input)

    def evaluate(
        self,
        rollout: Rollout,
        ground_truth: GroundTruthBundle,
        sample: BenchmarkSample,
        track: TrackId,
        evaluator: EvaluationDelegate,
        *,
        model_input: ModelInput,
    ) -> Any:
        self.validate_output(rollout, model_input)
        if len(ground_truth.rollout) != sample.horizon:
            raise ContractError("ground-truth horizon differs from sample")
        return evaluator.evaluate(
            task_id=self.task_id,
            track_id=TrackId(track),
            rollout=rollout,
            ground_truth=ground_truth,
            sample=sample,
        )
