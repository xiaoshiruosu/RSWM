"""Metric result records with explicit invalid/null semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Tuple

from .errors import EvaluationError
from .identity import MetricIdentity


@dataclass(frozen=True)
class MetricResult:
    value: Optional[float]
    valid: bool
    reason: Optional[str]
    N: int
    metric_identity: MetricIdentity
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.metric_identity, MetricIdentity):
            raise EvaluationError("metric_identity must be MetricIdentity")
        self.metric_identity.validate()
        try:
            count = int(self.N)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EvaluationError("MetricResult.N must be a non-negative integer") from exc
        if isinstance(self.N, bool) or count < 0 or float(self.N) != count:
            raise EvaluationError("MetricResult.N must be a non-negative integer")
        object.__setattr__(self, "N", count)
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))

        if self.valid:
            if self.value is None or count <= 0:
                raise EvaluationError("valid MetricResult requires value and N > 0")
            numeric = float(self.value)
            if not math.isfinite(numeric):
                raise EvaluationError("valid MetricResult value must be finite")
            if self.reason is not None:
                raise EvaluationError("valid MetricResult may not carry an invalid reason")
            object.__setattr__(self, "value", numeric)
        else:
            if self.value is not None:
                raise EvaluationError("invalid MetricResult must use value=None")
            if not str(self.reason or "").strip():
                raise EvaluationError("invalid MetricResult requires a reason")
            object.__setattr__(self, "reason", str(self.reason).strip())

    @classmethod
    def success(
        cls,
        value: float,
        *,
        N: int,
        metric_identity: MetricIdentity,
        warnings: Tuple[str, ...] = (),
    ) -> "MetricResult":
        return cls(value, True, None, N, metric_identity, warnings)

    @classmethod
    def invalid(
        cls,
        reason: str,
        *,
        N: int,
        metric_identity: MetricIdentity,
        warnings: Tuple[str, ...] = (),
    ) -> "MetricResult":
        return cls(None, False, reason, N, metric_identity, warnings)

    def to_document(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "valid": self.valid,
            "reason": self.reason,
            "N": self.N,
            "scope": self.metric_identity.scope.value,
            "warnings": list(self.warnings),
            "metric_identity": self.metric_identity.to_document(),
        }

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "MetricResult":
        identity = MetricIdentity.from_document(value.get("metric_identity") or {})
        declared_scope = value.get("scope")
        if declared_scope is not None and str(declared_scope) != identity.scope.value:
            raise EvaluationError(
                "MetricResult scope differs from embedded metric identity"
            )
        return cls(
            value=value.get("value"),
            valid=bool(value.get("valid", False)),
            reason=value.get("reason"),
            N=value.get("N", -1),
            metric_identity=identity,
            warnings=tuple(value.get("warnings") or ()),
        )
