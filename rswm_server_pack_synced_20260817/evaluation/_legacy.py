"""Lazy CPU-safe access to the frozen legacy metric truth sources."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType

from .errors import EvaluationError

_REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_METRICS_PATH = _REPO_ROOT / "lora_remotesense" / "new" / "metrics.py"
LEGACY_SUBMISSION_CONTRACT_PATH = (
    _REPO_ROOT / "lora_remotesense" / "new" / "submission_contract.py"
)
LEGACY_TRAJECTORY_PATH = (
    _REPO_ROOT / "lora_remotesense" / "new" / "trajectory_metrics.py"
)
LEGACY_GEOM_REGISTRATION_PATH = (
    _REPO_ROOT / "lora_remotesense" / "new" / "geom_registration.py"
)
_LEGACY_DIR = LEGACY_METRICS_PATH.parent
_LOCK = RLock()


def _load_exact(module_name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if existing_path is None or Path(existing_path).resolve() != path.resolve():
            raise EvaluationError(
                f"top-level module name collision for {module_name!r}: {existing_path!r}"
            )
        return existing
    if not path.is_file():
        raise EvaluationError(f"legacy metric source is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise EvaluationError(f"cannot load legacy metric source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_legacy_metrics() -> ModuleType:
    """Load metrics.py without importing torch or changing its numerical code."""

    with _LOCK:
        _load_exact("submission_contract", LEGACY_SUBMISSION_CONTRACT_PATH)
        return _load_exact("metrics", LEGACY_METRICS_PATH)


def load_legacy_trajectory_metrics() -> ModuleType:
    """Load trajectory_metrics.py after binding its historical absolute import."""

    with _LOCK:
        load_legacy_metrics()
        legacy_dir = str(_LEGACY_DIR)
        if legacy_dir not in sys.path:
            # Required only for trajectory_metrics' optional registration backend.
            sys.path.insert(0, legacy_dir)
        return _load_exact("trajectory_metrics", LEGACY_TRAJECTORY_PATH)
