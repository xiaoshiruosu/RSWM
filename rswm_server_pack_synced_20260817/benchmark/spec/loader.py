"""Read-only benchmark-spec loader."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.core.errors import SchemaError
from .schema import BenchmarkSpec


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"benchmark spec does not exist: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON in benchmark spec {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaError(f"benchmark spec {source} must contain one JSON object")
    return BenchmarkSpec.from_document(value)
