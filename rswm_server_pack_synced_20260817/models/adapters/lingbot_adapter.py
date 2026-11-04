"""WorldModel facade for LingBot-World base-cam via the existing Wan backbone.

This module deliberately contains no checkpoint loading, LoRA merge, sampling,
VAE decode or video canonicalization implementation.  Those responsibilities
remain in ``lora_remotesense.new.backbone.WanBackbone`` and its existing
inference boundary.  The adapter only validates benchmark inputs, translates
them to ``VideoBackbone.generate`` arguments, and binds structured identity to
the canonical rollout.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import math
import os
from threading import RLock
from typing import Any, Callable, Iterator, Optional

import numpy as np

from benchmark.core.contracts import validate_model_input, validate_rollout
from benchmark.core.errors import ContractError, IdentityError
from benchmark.core.identity import (
    ArchitectureIdentity,
    CameraConditionIdentity,
    CheckpointIdentity,
    ExecutionIdentity,
    LoraIdentity,
    ModelIdentity,
)
from benchmark.core.types import (
    CapabilityPartition,
    GenerationMetadata,
    ModelInput,
    PromptSchema,
    Rollout,
    TaskId,
    TrackId,
    TrajectoryCondition,
)
from models.world_model import WorldModelCapabilities

LINGBOT_RELEASE_IDENTITY = "LingBot-World-base-cam"
LINGBOT_ARCHITECTURE_IDENTITY = "Wan-I2V-A14B"
LINGBOT_ADAPTER_IDENTITY = "LingBotWorldAdapter"
LINGBOT_EXECUTION_IDENTITY = "benchmark-adapter"
LINGBOT_ADAPTER_VERSION = "rswm.lingbot_world_adapter/v1"
LINGBOT_RUNTIME_SCHEMA = "rswm.lingbot_runtime/v1"

_POSE_MODE_LOCK = RLock()


def _nonempty(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ContractError(f"{label} must be a non-empty string")
    return text


@dataclass(frozen=True)
class LingBotRuntimeConfig:
    """Runtime-only WanBackbone constructor parameters.

    Checkpoint and LoRA paths are intentionally not part of this object: they
    are separate constructor arguments so a registry cannot silently drop or
    replace their identities.
    """

    device: str = "cuda"
    config_key: str = "i2v-A14B"
    sampling_steps: int = 70
    shift: float = 10.0
    guide_scale: float = 5.0
    t5_cpu: bool = True
    inactive_model_cpu: bool = True
    framewise: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _nonempty(self.device, "runtime.device"))
        object.__setattr__(
            self, "config_key", _nonempty(self.config_key, "runtime.config_key")
        )
        if self.config_key != "i2v-A14B":
            raise ContractError(
                "LingBot-World-base-cam identity requires config_key='i2v-A14B'"
            )
        try:
            steps = int(self.sampling_steps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ContractError("runtime.sampling_steps must be positive") from exc
        if isinstance(self.sampling_steps, bool) or steps <= 0:
            raise ContractError("runtime.sampling_steps must be positive")
        object.__setattr__(self, "sampling_steps", steps)
        for name in ("shift", "guide_scale"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ContractError(f"runtime.{name} must be finite and positive") from exc
            if not math.isfinite(value) or value <= 0:
                raise ContractError(f"runtime.{name} must be finite and positive")
            object.__setattr__(self, name, value)
        for name in ("t5_cpu", "inactive_model_cpu", "framewise"):
            if not isinstance(getattr(self, name), bool):
                raise ContractError(f"runtime.{name} must be bool")

    def backbone_kwargs(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "config_key": self.config_key,
            "sampling_steps": self.sampling_steps,
            "shift": self.shift,
            "guide_scale": self.guide_scale,
            "t5_cpu": self.t5_cpu,
            "inactive_model_cpu": self.inactive_model_cpu,
            "framewise": self.framewise,
        }

    def to_document(self) -> dict[str, Any]:
        return {"schema": LINGBOT_RUNTIME_SCHEMA, **self.backbone_kwargs()}


def _default_wan_backbone_factory(**kwargs: Any) -> Any:
    """Build the existing registered WanBackbone with explicit path kwargs."""

    # Lazy import keeps adapter construction and identity inspection CPU-only.
    from lora_remotesense.new.backbone import build_backbone

    return build_backbone("wan", **kwargs)


def _validate_video_backbone(backbone: Any) -> None:
    missing = [
        name
        for name in ("name", "capabilities", "generate")
        if not hasattr(backbone, name)
    ]
    if missing:
        raise TypeError(f"LingBot backend is missing VideoBackbone fields: {missing}")
    if not isinstance(backbone.name, str) or not backbone.name:
        raise TypeError("LingBot backend name must be a non-empty string")
    if not callable(backbone.generate):
        raise TypeError("LingBot backend generate must be callable")
    signature = inspect.signature(backbone.generate)
    required = ("first_frame", "prompt", "num_frames", "size", "pose_cond", "seed")
    if not any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        missing_parameters = [
            name for name in required if name not in signature.parameters
        ]
        if missing_parameters:
            raise TypeError(
                "LingBot backend does not implement VideoBackbone.generate; "
                f"missing={missing_parameters}"
            )


@contextmanager
def _wan_pose_mode(track_id: TrackId) -> Iterator[str]:
    """Align WanBackbone's legacy environment guard with the explicit track.

    ``VideoBackbone.generate`` receives pose data explicitly, but the existing
    Wan implementation additionally cross-checks ``RSWM_POSE_MODE``.  The lock
    prevents concurrent calls from observing another request's temporary mode.
    The previous process value is restored after generation.
    """

    mode = "drop" if TrackId(track_id) is TrackId.A_TEXT else "native"
    with _POSE_MODE_LOCK:
        previous = os.environ.get("RSWM_POSE_MODE")
        os.environ["RSWM_POSE_MODE"] = mode
        try:
            yield mode
        finally:
            if previous is None:
                os.environ.pop("RSWM_POSE_MODE", None)
            else:
                os.environ["RSWM_POSE_MODE"] = previous


class LingBotWorldModelAdapter:
    """Thin WorldModel adapter for one LingBot release and Wan architecture."""

    name = LINGBOT_RELEASE_IDENTITY

    def __init__(
        self,
        *,
        checkpoint_identity: CheckpointIdentity,
        checkpoint_path: os.PathLike[str] | str,
        lora_identity: Optional[LoraIdentity] = None,
        lora_path: Optional[os.PathLike[str] | str] = None,
        runtime_config: Optional[LingBotRuntimeConfig] = None,
        backbone_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if not isinstance(checkpoint_identity, CheckpointIdentity):
            raise IdentityError("checkpoint_identity must be CheckpointIdentity")
        checkpoint_path_text = _nonempty(
            os.fspath(checkpoint_path), "checkpoint_path"
        )
        if lora_identity is not None and not isinstance(lora_identity, LoraIdentity):
            raise IdentityError("lora_identity must be LoraIdentity or None")
        lora_path_text = None
        if lora_path is not None:
            lora_path_text = _nonempty(os.fspath(lora_path), "lora_path")
        if (lora_identity is None) != (lora_path_text is None):
            raise ContractError(
                "lora_identity and lora_path must either both be provided or both be absent"
            )
        if runtime_config is None:
            runtime_config = LingBotRuntimeConfig()
        if not isinstance(runtime_config, LingBotRuntimeConfig):
            raise ContractError("runtime_config must be LingBotRuntimeConfig")
        if backbone_factory is not None and not callable(backbone_factory):
            raise TypeError("backbone_factory must be callable")

        self.checkpoint_identity = checkpoint_identity
        self.checkpoint_path = checkpoint_path_text
        self.lora_identity = lora_identity
        self.lora_path = lora_path_text
        self.runtime_config = runtime_config
        self._backbone_factory_override = backbone_factory is not None
        self._backbone_factory = backbone_factory or _default_wan_backbone_factory
        self._backbone: Optional[Any] = None
        self._backbone_lock = RLock()

        execution_identity = ExecutionIdentity(
            route=LINGBOT_EXECUTION_IDENTITY,
            entrypoint="models/adapters/lingbot_adapter.py",
            legacy_aliases=("wan", "wan-inline"),
        )
        identity_status = (
            "verified"
            if (
                checkpoint_identity.status == "verified"
                and lora_identity is None
                and not self._backbone_factory_override
            )
            else "declared"
        )
        self.model_identity = ModelIdentity.create(
            model_family=LINGBOT_ARCHITECTURE_IDENTITY,
            release_identity=LINGBOT_RELEASE_IDENTITY,
            adapter_identity=LINGBOT_ADAPTER_IDENTITY,
            checkpoint_identity=checkpoint_identity,
            architecture_identity=ArchitectureIdentity(
                transformer_type=LINGBOT_ARCHITECTURE_IDENTITY,
                vae="Wan2.1_VAE",
                text_encoder="umT5-XXL",
                camera_conditioning_type="Plucker-ray camera conditioning",
                implementation_aliases=("wan", "i2v-A14B"),
            ),
            camera_condition_identity=CameraConditionIdentity(
                encoding="plucker_ray_v1",
                pose_injection=(
                    "global projection plus per-transformer-block modulation"
                ),
                injector_type="camera injector with scale and shift",
            ),
            lora_identity=lora_identity,
            execution_identity=execution_identity,
            status=identity_status,
        )
        partition = (
            CapabilityPartition.ADAPTED
            if lora_identity is not None
            else CapabilityPartition.NATIVE
        )
        self.capabilities = WorldModelCapabilities(
            input_type="single_image",
            supported_tasks=frozenset(TaskId),
            supported_tracks=frozenset({TrackId.A_TEXT, TrackId.B_CAMERA}),
            prompt_schemas=frozenset({PromptSchema.NET_CONTINUOUS}),
            trajectory_support="net_direction+full_path",
            camera_support="plucker_ray_v1",
            supported_horizons=frozenset({17, 25, 33}),
            adaptation_type="lora" if lora_identity is not None else "frozen",
            capability_partition=partition,
            condition_view_count=1,
            stochastic=True,
            native_output_format="THWC_uint8",
            checkpoint_required=True,
            notes=(
                "LingBot release identity over the existing Wan I2V A14B "
                "VideoBackbone interface"
            ),
            model_identity=self.model_identity,
        )

    @property
    def execution_identity(self) -> ExecutionIdentity:
        return self.model_identity.execution_identity

    @property
    def adapter_identity(self) -> str:
        return self.model_identity.adapter_identity

    def _get_backbone(self) -> Any:
        with self._backbone_lock:
            if self._backbone is None:
                kwargs = {
                    "checkpoint_dir": self.checkpoint_path,
                    "lora_path": self.lora_path,
                    **self.runtime_config.backbone_kwargs(),
                }
                backbone = self._backbone_factory(**kwargs)
                _validate_video_backbone(backbone)
                self._backbone = backbone
            return self._backbone

    def generate(
        self,
        condition_image: np.ndarray,
        trajectory_condition: TrajectoryCondition,
        metadata: GenerationMetadata,
    ) -> Rollout:
        model_input = ModelInput(
            condition_image=condition_image,
            trajectory_condition=trajectory_condition,
            metadata=metadata,
        )
        validate_model_input(model_input)
        self.capabilities.assert_eligible(
            task_id=metadata.task_id,
            track_id=metadata.track_id,
            prompt_schema=trajectory_condition.prompt_schema,
            horizon=metadata.horizon,
        )

        pose_cond: Optional[dict[str, np.ndarray]] = None
        if metadata.track_id is TrackId.B_CAMERA:
            # Mapping form is consumed by the existing WanBackbone.coerce_pose_cond.
            pose_cond = {
                "c2w": trajectory_condition.poses,
                "intrinsics": trajectory_condition.intrinsics,
            }
        height, width = metadata.output_size
        backbone = self._get_backbone()
        with _wan_pose_mode(metadata.track_id) as pose_mode:
            frames = backbone.generate(
                first_frame=model_input.condition_image,
                prompt=trajectory_condition.scene_prompt,
                num_frames=metadata.horizon,
                size=f"{height}*{width}",
                pose_cond=pose_cond,
                seed=metadata.generation_seed,
            )

        rollout = Rollout(
            frames=np.asarray(frames),
            sample_id=metadata.sample_id,
            model_id=self.model_identity.release_identity,
            adapter_version=LINGBOT_ADAPTER_VERSION,
            generation_seed=metadata.generation_seed,
            runtime_metadata={
                "backbone": backbone.name,
                "backbone_contract": "VideoBackbone.generate/v1",
                "pose_mode": pose_mode,
                "checkpoint_repository": self.checkpoint_identity.repository,
                "lora_enabled": self.lora_identity is not None,
                "backend_factory_override": self._backbone_factory_override,
                "runtime_config": self.runtime_config.to_document(),
                "model_identity_checksum": self.model_identity.identity_checksum,
            },
            model_identity=self.model_identity,
            adapter_identity=self.model_identity.adapter_identity,
            execution_identity=self.model_identity.execution_identity,
        )
        validate_rollout(rollout, model_input)
        return rollout
