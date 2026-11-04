"""Conservative read-only bridge for server_run/baselines card v1."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from threading import RLock
from types import ModuleType
from typing import Any, Optional

from benchmark.core.identity import (
    ArchitectureIdentity,
    CameraConditionIdentity,
    CheckpointIdentity,
    ComponentIdentity,
    ExecutionIdentity,
    ModelIdentity,
    sha256_file,
)
from benchmark.core.types import CapabilityPartition

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
from .card_v2 import BaselineCardV2, LegacyCardReference
from .errors import LegacyBaselineError
from .identity import BaselineIdentity

_ROOT = Path(__file__).resolve().parents[1]
_LEGACY_DIR = _ROOT / "server_run" / "baselines"
_PROTOCOL_PATH = _LEGACY_DIR / "protocol.py"
_CARDS_PATH = _LEGACY_DIR / "cards.py"
_REGISTRY_PATH = _LEGACY_DIR / "registry.py"
_CARD_DIR = _LEGACY_DIR / "cards"
_LOCK = RLock()


def _load_exact(module_name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(module_name)
    if existing is not None:
        existing_path = getattr(existing, "__file__", None)
        if existing_path is None or Path(existing_path).resolve() != path.resolve():
            raise LegacyBaselineError(
                f"top-level module collision for {module_name!r}: {existing_path!r}"
            )
        return existing
    if not path.is_file():
        raise LegacyBaselineError(f"legacy baseline source missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise LegacyBaselineError(f"cannot load legacy baseline source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_legacy_cards_module() -> ModuleType:
    with _LOCK:
        _load_exact("protocol", _PROTOCOL_PATH)
        return _load_exact("_rswm_legacy_baseline_cards", _CARDS_PATH)


def load_legacy_registry_module() -> ModuleType:
    with _LOCK:
        _load_exact("protocol", _PROTOCOL_PATH)
        legacy_dir = str(_LEGACY_DIR)
        if legacy_dir not in sys.path:
            sys.path.insert(0, legacy_dir)
        return _load_exact("registry", _REGISTRY_PATH)


class LegacyBaselineAdapter:
    """Create a DECLARED v2 overlay without changing or upgrading card v1."""

    def __init__(self, name: str) -> None:
        self.name = str(name).strip()
        if not self.name:
            raise LegacyBaselineError("legacy baseline name must be non-empty")

    @classmethod
    def list_names(cls) -> tuple[str, ...]:
        return tuple(load_legacy_cards_module().list_cards())

    def load_v1(self) -> dict[str, Any]:
        return load_legacy_cards_module().load_card(self.name)

    def overlay(self) -> BaselineCardV2:
        card = self.load_v1()
        card_path = _CARD_DIR / f"{self.name}.json"
        try:
            registry_adapter = load_legacy_registry_module().get(self.name)
        except KeyError:
            registry_adapter = None

        adapter_name = f"LegacyBaselineAdapter:{self.name}"
        adapter_implemented = (
            registry_adapter is not None
            and getattr(registry_adapter, "status", None) == "implemented"
        )
        adapter_identity = ComponentIdentity(
            kind="baseline_adapter",
            artifact_id=adapter_name,
            schema="rswm.baseline_adapter/v1",
            version=(
                str(load_legacy_registry_module().ADAPTER_VERSION)
                if registry_adapter is not None
                else UNKNOWN
            ),
            sha256=sha256_file(_REGISTRY_PATH) if registry_adapter is not None else None,
            status="verified" if adapter_implemented else "declared",
        )

        checkpoint = CheckpointIdentity(
            repository=(
                "NOT_APPLICABLE" if card.get("checkpoint") is None else UNKNOWN
            ),
            status="declared",
        )
        execution = ExecutionIdentity(
            route=(
                "legacy-baseline-registry"
                if registry_adapter is not None
                else "legacy-card-only"
            ),
            entrypoint=(
                "server_run/baselines/registry.py"
                if registry_adapter is not None
                else "server_run/baselines/cards.py"
            ),
            legacy_aliases=(type(registry_adapter).__name__,)
            if registry_adapter is not None
            else (),
        )
        uses_camera = bool(card["uses_camera_pose"])
        model_identity = ModelIdentity.create(
            model_family=UNKNOWN,
            release_identity=UNKNOWN,
            adapter_identity=adapter_name,
            checkpoint_identity=checkpoint,
            architecture_identity=ArchitectureIdentity(
                transformer_type=UNKNOWN,
                vae=UNKNOWN,
                text_encoder=UNKNOWN,
                camera_conditioning_type=UNKNOWN if uses_camera else "none",
            ),
            camera_condition_identity=CameraConditionIdentity(
                encoding=UNKNOWN if uses_camera else "none",
                pose_injection=UNKNOWN if uses_camera else "none",
                injector_type=UNKNOWN if uses_camera else "none",
            ),
            lora_identity=None,
            execution_identity=execution,
            status="declared",
        )
        identity = BaselineIdentity.create(
            baseline_id=self.name,
            model_identity=model_identity,
            adapter_identity=adapter_identity,
            checkpoint_identity=checkpoint,
            execution_identity=execution,
        )

        views = int(card["number_of_condition_views"])
        input_type = InputType.MULTIVIEW if views > 1 else InputType.UNKNOWN
        trajectory_support = (
            TrajectorySupport.NONE
            if str(card.get("trajectory_support", "")).strip().lower()
            in {"none", "ignored"}
            else TrajectorySupport.UNKNOWN
        )
        camera_support = (
            CameraSupport.UNKNOWN if uses_camera else CameraSupport.NONE
        )
        runtime = RuntimeDisclosure(
            framework=(
                "server_run.baselines.registry"
                if registry_adapter is not None
                else UNKNOWN
            ),
            framework_version=(
                str(load_legacy_registry_module().ADAPTER_VERSION)
                if registry_adapter is not None
                else UNKNOWN
            ),
            container_identity=UNKNOWN,
            hardware=UNKNOWN,
            precision=UNKNOWN,
        )
        capabilities = BaselineCapabilities(
            input_type=input_type,
            trajectory_support=trajectory_support,
            camera_support=camera_support,
            adaptation_type=AdaptationType.UNKNOWN,
            checkpoint_identity=checkpoint,
            license=LicenseDisclosure(UNKNOWN, UNKNOWN, UNKNOWN),
            tasks=(),
            tracks=(),
            external_data=ExternalDataDisclosure(
                status=DisclosureStatus.UNKNOWN,
                description=UNKNOWN,
                references=(),
            ),
            runtime=runtime,
            condition_views=views,
            uses_future_gt=bool(card["uses_future_gt"]),
            output_format=UNKNOWN,
        )
        partition: Optional[CapabilityPartition] = (
            CapabilityPartition.MULTIVIEW if views > 1 else None
        )
        legacy_ref = LegacyCardReference(
            name=self.name,
            sha256=sha256_file(card_path),
            relative_path=f"server_run/baselines/cards/{self.name}.json",
        )
        return BaselineCardV2.create_declared(
            baseline_identity=identity,
            capabilities=capabilities,
            legacy_card=legacy_ref,
            capability_partition=partition,
        )
