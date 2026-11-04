"""T3 endpoint-overlap ladder contract."""

from __future__ import annotations

from typing import Optional

from benchmark.core.errors import ContractError
from benchmark.core.types import BenchmarkSample, TaskId
from benchmark.trajectory.types import OVERLAP_LEVELS
from .base import BaseTask


class T3OverlapLadderTask(BaseTask):
    task_id = TaskId.T3_OVERLAP_LADDER

    def __init__(
        self,
        *,
        required_level: Optional[str] = None,
        benchmark_version: str = "RSWM-Bench-v1",
    ):
        super().__init__(benchmark_version=benchmark_version)
        if required_level is not None and required_level not in OVERLAP_LEVELS:
            raise ValueError(f"unknown T3 overlap level: {required_level!r}")
        self.required_level = required_level

    def validate_sample(self, sample: BenchmarkSample) -> None:
        super().validate_sample(sample)
        level = sample.overlap.level
        if level is None:
            raise ContractError(
                f"{sample.id}: endpoint overlap does not belong to a frozen T3 level"
            )
        if self.required_level is not None and level != self.required_level:
            raise ContractError(
                f"{sample.id}: overlap level {level!r} != required {self.required_level!r}"
            )
