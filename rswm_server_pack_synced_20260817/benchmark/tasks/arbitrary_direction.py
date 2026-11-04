"""T2 arbitrary-direction spatial extrapolation contract."""

import math

from benchmark.core.errors import ContractError
from benchmark.core.types import BenchmarkSample, TaskId
from .base import BaseTask


class T2ArbitraryDirectionTask(BaseTask):
    task_id = TaskId.T2_ARBITRARY_DIRECTION

    def validate_sample(self, sample: BenchmarkSample) -> None:
        super().validate_sample(sample)
        angle = float(sample.direction.camera_bearing_deg)
        if not math.isfinite(angle) or not 0.0 <= angle < 360.0:
            raise ContractError(f"{sample.id}: T2 requires a continuous camera bearing")
        if not sample.direction.segment_camera_bearings_deg:
            raise ContractError(f"{sample.id}: T2 requires segment direction metadata")
