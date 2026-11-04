"""Frozen RSWM-Bench V1 task contracts."""

from .arbitrary_direction import T2ArbitraryDirectionTask
from .base import BaseTask, EvaluationDelegate
from .overlap_ladder import T3OverlapLadderTask
from .registry import build_task, build_task_from_spec, list_tasks
from .spatial_rollout import T1SpatialRolloutTask

__all__ = [
    "BaseTask",
    "EvaluationDelegate",
    "T1SpatialRolloutTask",
    "T2ArbitraryDirectionTask",
    "T3OverlapLadderTask",
    "build_task",
    "build_task_from_spec",
    "list_tasks",
]
