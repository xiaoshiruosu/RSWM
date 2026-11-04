"""Fail-closed errors for the RSWM-Bench baseline capability layer."""

from benchmark.core.errors import CapabilityError, ContractError, IdentityError


class BaselineError(ContractError):
    """Base class for baseline abstraction failures."""


class BaselineIdentityError(IdentityError, BaselineError):
    """A baseline identity mixes or mismatches identity layers."""


class BaselineCapabilityError(CapabilityError, BaselineError):
    """A baseline is not eligible for the requested capability partition."""


class AdmissionStateError(BaselineError):
    """A card attempted an invalid or unsupported admission transition."""


class LegacyBaselineError(BaselineError):
    """A legacy baseline card or registry entry cannot be wrapped safely."""
