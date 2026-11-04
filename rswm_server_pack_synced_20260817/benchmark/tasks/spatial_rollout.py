"""T1 trajectory-conditioned single-view spatial rollout contract."""

from benchmark.core.types import TaskId
from .base import BaseTask


class T1SpatialRolloutTask(BaseTask):
    task_id = TaskId.T1_SPATIAL_ROLLOUT
