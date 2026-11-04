"""Fail-loud exception hierarchy for the benchmark abstraction layer."""


class BenchmarkError(RuntimeError):
    """Base class for benchmark-layer failures."""


class ContractError(BenchmarkError):
    """An input or output violates a frozen benchmark contract."""


class CapabilityError(ContractError):
    """A method is not eligible for the requested task/track partition."""


class DatasetContractError(ContractError):
    """HDF5, manifest, split or sample content is inconsistent."""


class SchemaError(ContractError):
    """A versioned schema document is malformed or has been tampered with."""


class IdentityError(ContractError):
    """A release component identity is incomplete or inconsistent."""


class ArtifactLineageError(IdentityError):
    """Checkpoint-to-result lineage is missing, reordered or inconsistent."""
