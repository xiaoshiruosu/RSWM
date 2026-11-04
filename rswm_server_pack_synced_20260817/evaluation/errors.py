"""Fail-loud errors for the RSWM-Bench evaluation abstraction."""

from benchmark.core.errors import ContractError, IdentityError


class EvaluationError(ContractError):
    """Base class for evaluator contract failures."""


class MetricIdentityError(IdentityError, EvaluationError):
    """A metric identity is malformed or not content-bound."""


class MetricRegistryError(EvaluationError):
    """A metric cannot be registered or resolved unambiguously."""


class FutureOnlyOwnershipError(EvaluationError):
    """More than one component attempted to own future-only slicing."""


class AccumulatorStateError(EvaluationError):
    """A dataset metric lifecycle method was called out of order."""


class InvalidMetricResultError(EvaluationError):
    """A metric returned an explicit invalid result in fail-closed mode."""

    def __init__(self, results) -> None:
        self.results = tuple(results)
        details = ", ".join(
            f"{item.metric_identity.metric_id}:{item.reason}"
            for item in self.results
            if not item.valid
        )
        super().__init__(f"evaluation produced invalid metric result(s): {details}")
