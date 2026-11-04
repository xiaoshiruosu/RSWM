"""RSWM-Bench M5 baseline identity, capability and admission framework."""

from .capability import (
    AdaptationType,
    BaselineCapabilities,
    CameraSupport,
    DisclosureStatus,
    ExternalDataDisclosure,
    InputType,
    LicenseDisclosure,
    RuntimeDisclosure,
    TrajectorySupport,
    UNKNOWN,
)
from .card_v2 import (
    AdmissionState,
    BaselineCardV2,
    LegacyCardReference,
    StateEvidence,
)
from .errors import (
    AdmissionStateError,
    BaselineCapabilityError,
    BaselineError,
    BaselineIdentityError,
    LegacyBaselineError,
)
from .identity import BaselineIdentity
from .legacy import LegacyBaselineAdapter

__all__ = [
    "AdaptationType",
    "AdmissionState",
    "AdmissionStateError",
    "BaselineCapabilities",
    "BaselineCapabilityError",
    "BaselineCardV2",
    "BaselineError",
    "BaselineIdentity",
    "BaselineIdentityError",
    "CameraSupport",
    "DisclosureStatus",
    "ExternalDataDisclosure",
    "InputType",
    "LegacyBaselineAdapter",
    "LegacyBaselineError",
    "LegacyCardReference",
    "LicenseDisclosure",
    "RuntimeDisclosure",
    "StateEvidence",
    "TrajectorySupport",
    "UNKNOWN",
]
