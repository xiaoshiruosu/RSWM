"""CPU-only M7 benchmark runtime orchestration.

Ground truth is intentionally loaded only after model generation.  This keeps
the model-facing path limited to ``BenchmarkSample -> ModelInput`` and makes
the Track A/Track B visibility boundary observable in fixture tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

from benchmark.core.errors import ContractError
from benchmark.core.identity import ArtifactRef
from benchmark.core.types import ModelInput, Rollout, ValidationReport
from benchmark.spec.schema import BenchmarkSpec
from benchmark.tasks.base import BaseTask
from evaluation.bundle import ResultBundle, SampleIdentity
from evaluation.result import MetricResult
from evaluation.runner import EvaluationRunner
from models.world_model import validate_world_model


@dataclass(frozen=True)
class DryRunOutcome:
    """In-memory evidence from one CPU contract run; never a formal result."""

    model_input: ModelInput
    rollout: Rollout
    validation_report: ValidationReport
    metric_results: Tuple[MetricResult, ...]
    result_bundle: ResultBundle


class CpuEndToEndDryRun:
    """Join the frozen dataset, task, model and evaluation abstractions."""

    def __init__(
        self,
        *,
        dataset: Any,
        task: BaseTask,
        world_model: Any,
        evaluator: EvaluationRunner,
        checkpoint_artifact: ArtifactRef,
        adapter_artifact: ArtifactRef,
    ) -> None:
        validate_world_model(world_model)
        if not isinstance(task, BaseTask):
            raise TypeError("task must be a BaseTask")
        if not isinstance(evaluator, EvaluationRunner):
            raise TypeError("evaluator must be EvaluationRunner")
        if checkpoint_artifact.kind != "checkpoint":
            raise ContractError("checkpoint_artifact must have kind='checkpoint'")
        if adapter_artifact.kind != "adapter":
            raise ContractError("adapter_artifact must have kind='adapter'")
        spec = getattr(dataset, "spec", None)
        if not isinstance(spec, BenchmarkSpec):
            raise ContractError("M7 dataset must be bound to a BenchmarkSpec")
        spec.validate()
        if task.task_id != spec.task_id:
            raise ContractError("task identity differs from benchmark spec")
        if task.benchmark_version != spec.benchmark_version:
            raise ContractError("task benchmark version differs from benchmark spec")
        release = getattr(getattr(dataset, "identity", None), "release_identity", None)
        if release is None:
            raise ContractError("M7 dataset identity is missing benchmark release identity")
        release.validate()
        if release.release_checksum != spec.release_identity().release_checksum:
            raise ContractError("dataset release identity differs from benchmark spec")

        self.dataset = dataset
        self.spec = spec
        self.task = task
        self.world_model = world_model
        self.evaluator = evaluator
        self.checkpoint_artifact = checkpoint_artifact
        self.adapter_artifact = adapter_artifact

    def run_sample(
        self,
        sample_id: str,
        *,
        generation_seed: int = 42,
    ) -> DryRunOutcome:
        sample_id = str(sample_id)
        if sample_id not in self.dataset.sample_ids:
            raise ContractError(f"sample_id is not in frozen dataset: {sample_id!r}")

        # Load only the model-visible sample before generation.  Ground truth is
        # deliberately absent from every object passed to WorldModel.generate().
        sample = self.dataset.benchmark_sample(sample_id)
        model_input = self.task.prepare_input(
            sample,
            self.spec.conditioning_track,
            self.world_model.capabilities,
            generation_seed=generation_seed,
        )
        rollout = self.world_model.generate(
            condition_image=model_input.condition_image,
            trajectory_condition=model_input.trajectory_condition,
            metadata=model_input.metadata,
        )
        validation = self.task.validate_output(rollout, model_input)
        if not validation.valid:
            raise ContractError("rollout validation returned invalid status")

        # Organizer-only data enters only after the model output is frozen.
        ground_truth = self.dataset.ground_truth(sample_id)
        metric_results = self.task.evaluate(
            rollout,
            ground_truth,
            sample,
            self.spec.conditioning_track,
            self.evaluator,
            model_input=model_input,
        )
        sample_identity = SampleIdentity.from_model_input(
            model_input,
            manifest_checksum=self.dataset.identity.manifest_checksum,
        )
        if rollout.model_identity is None:
            raise ContractError("rollout is missing model identity")
        result_bundle = ResultBundle.create(
            benchmark_identity=self.dataset.identity.release_identity,
            model_identity=rollout.model_identity,
            adapter_identity=rollout.adapter_identity,
            execution_identity=rollout.execution_identity,
            sample_identity=sample_identity,
            metric_results=metric_results,
            checkpoint_artifact=self.checkpoint_artifact,
            adapter_artifact=self.adapter_artifact,
            fail_closed=True,
        )
        result_bundle.validate(require_valid=True)
        return DryRunOutcome(
            model_input=model_input,
            rollout=rollout,
            validation_report=validation,
            metric_results=tuple(metric_results),
            result_bundle=result_bundle,
        )

    def run_all(self, *, generation_seed: int = 42) -> Tuple[DryRunOutcome, ...]:
        return tuple(
            self.run_sample(sample_id, generation_seed=generation_seed)
            for sample_id in self.dataset.sample_ids
        )
