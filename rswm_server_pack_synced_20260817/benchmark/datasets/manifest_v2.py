"""Compatibility bridge to the existing frozen manifest v2 validator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

from benchmark.core.errors import DatasetContractError

try:
    from server_run.baselines import protocol as legacy_protocol
except ImportError as exc:  # pragma: no cover - installation/layout error
    raise ImportError(
        "RSWM package root must be on sys.path so server_run.baselines.protocol is available"
    ) from exc


@dataclass(frozen=True)
class LegacyManifestV2:
    path: Path
    document: Mapping[str, Any]
    h5_path: Path
    split_path: Path
    sample_ids: Tuple[str, ...]
    sample_index: Mapping[str, Mapping[str, Any]]

    @property
    def checksum(self) -> str:
        return str(self.document["manifest_checksum"]).lower()

    @property
    def schema(self) -> str:
        return str(self.document["schema"])


def _resolve(reference: Any, *, parent: Path) -> Path:
    path = Path(str(reference))
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def load_manifest_v2(
    path: str | Path,
    *,
    require_files: bool = True,
    verify_file_hashes: bool = True,
    require_headline_eligible: bool = False,
) -> LegacyManifestV2:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"manifest does not exist: {source}")
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetContractError(f"invalid manifest JSON {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise DatasetContractError(f"manifest {source} must contain one JSON object")
    try:
        validated = legacy_protocol.validate_manifest_document(
            document,
            require_files=False,
        )
    except Exception as exc:
        raise DatasetContractError(f"manifest v2 validation failed: {exc}") from exc
    h5_path = _resolve(validated["h5"], parent=source.parent)
    split_path = _resolve(validated["split"], parent=source.parent)
    if require_files:
        missing = [str(item) for item in (h5_path, split_path) if not item.is_file()]
        if missing:
            raise DatasetContractError(f"manifest references missing files: {missing}")
    if require_headline_eligible:
        if not bool(validated.get("headline_eligible", False)):
            raise DatasetContractError("official adapter requires headline_eligible manifest")
        if validated.get("allow_waypoint_interpolation"):
            raise DatasetContractError("official adapter forbids waypoint interpolation")
        if validated.get("max_samples") is not None:
            raise DatasetContractError("official adapter forbids diagnostic prefix subsets")
        if validated.get("dataset_checksum") is None:
            raise DatasetContractError("official adapter requires dataset checksum")
    if require_files and verify_file_hashes:
        expected_h5 = validated.get("dataset_checksum")
        if expected_h5 is not None:
            actual_h5 = legacy_protocol.sha256_file(h5_path)
            if actual_h5.lower() != str(expected_h5).lower():
                raise DatasetContractError(
                    f"HDF5 checksum mismatch: {actual_h5} != {expected_h5}"
                )
        actual_split = legacy_protocol.sha256_file(split_path)
        if actual_split.lower() != str(validated["split_checksum"]).lower():
            raise DatasetContractError(
                f"split checksum mismatch: {actual_split} != {validated['split_checksum']}"
            )
    index = legacy_protocol.sample_index(validated)
    ids = tuple(str(value) for value in validated.get("sample_keys") or index)
    if tuple(index) != ids:
        # sample_index preserves document order.  The explicit check prevents a
        # future refactor from silently changing split/sample ordering.
        raise DatasetContractError("manifest sample index order differs from sample_keys")
    return LegacyManifestV2(
        path=source,
        document=validated,
        h5_path=h5_path,
        split_path=split_path,
        sample_ids=ids,
        sample_index=index,
    )
