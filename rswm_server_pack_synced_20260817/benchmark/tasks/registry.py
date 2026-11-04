"""Explicit task registry keyed by the frozen task IDs."""

from __future__ import annotations

from typing import Callable, Dict

from benchmark.core.types import TaskId
from benchmark.spec.schema import BenchmarkSpec
from .arbitrary_direction import T2ArbitraryDirectionTask
from .base import BaseTask
from .overlap_ladder import T3OverlapLadderTask
from .spatial_rollout import T1SpatialRolloutTask

_TASK_BUILDERS: Dict[TaskId, Callable[..., BaseTask]] = {
    TaskId.T1_SPATIAL_ROLLOUT: T1SpatialRolloutTask,
    TaskId.T2_ARBITRARY_DIRECTION: T2ArbitraryDirectionTask,
    TaskId.T3_OVERLAP_LADDER: T3OverlapLadderTask,
}


def list_tasks() -> tuple[str, ...]:
    return tuple(task.value for task in _TASK_BUILDERS)


def build_task(task_id: TaskId, **kwargs) -> BaseTask:
    task_id = TaskId(task_id)
    return _TASK_BUILDERS[task_id](**kwargs)


def build_task_from_spec(spec: BenchmarkSpec) -> BaseTask:
    spec.validate()
    kwargs = {"benchmark_version": spec.benchmark_version}
    if spec.task_id == TaskId.T3_OVERLAP_LADDER:
        kwargs["required_level"] = spec.task_parameters.get("overlap_level")
    return build_task(spec.task_id, **kwargs)
