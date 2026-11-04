from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

from benchmark.core.errors import ContractError
from benchmark.core.identity import (
    CheckpointIdentity,
    LoraIdentity,
    canonical_sha256,
)
from benchmark.core.types import (
    CapabilityPartition,
    GenerationMetadata,
    PromptSchema,
    Rollout,
    TaskId,
    TrackId,
    TrajectoryCondition,
)
from models import (
    LingBotRuntimeConfig,
    LingBotWorldModelAdapter,
    validate_world_model,
)


def _digest(label: str) -> str:
    return canonical_sha256({"fixture": label})


def _checkpoint_identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        repository="robbyant/lingbot-world-base-cam",
        revision="fixture-revision",
        status="declared",
    )


def _verified_checkpoint_identity() -> CheckpointIdentity:
    return CheckpointIdentity(
        repository="robbyant/lingbot-world-base-cam",
        revision="fixture-revision",
        sha256_manifest=_digest("checkpoint-manifest"),
        config_hash=_digest("checkpoint-config"),
        status="verified",
    )


def _model_input(track_id: TrackId, *, seed: int = 123):
    horizon = 17
    image = (np.arange(8 * 12 * 3, dtype=np.uint16) % 251).astype(np.uint8)
    image = image.reshape(8, 12, 3)
    poses = intrinsics = None
    if track_id is TrackId.B_CAMERA:
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], horizon, axis=0)
        poses[:, 0, 3] = np.linspace(0.0, 0.2, horizon)
        intrinsics = np.repeat(
            np.array([[8.0, 8.0, 6.0, 4.0]], dtype=np.float32),
            horizon,
            axis=0,
        )
    prompt = "A satellite video following a 9-segment trajectory toward east."
    condition = TrajectoryCondition(
        track_id=track_id,
        scene_prompt=prompt,
        prompt_schema=PromptSchema.NET_CONTINUOUS,
        segment_count=9,
        net_camera_bearing_deg=90.0,
        frame_count=horizon,
        poses=poses,
        intrinsics=intrinsics,
    )
    metadata = GenerationMetadata(
        benchmark_version="RSWM-Bench-v1",
        task_id=TaskId.T1_SPATIAL_ROLLOUT,
        track_id=track_id,
        sample_id=f"fixture-{track_id.value}",
        horizon=horizon,
        output_size=(8, 12),
        generation_seed=seed,
        condition_checksum=_digest(f"condition-{track_id.value}"),
    )
    return image, condition, metadata


class _SpyWanBackbone:
    name = "wan-backbone-mock"
    capabilities = object()

    def __init__(self, *, invalid_output: bool = False):
        self.invalid_output = invalid_output
        self.calls = []

    def generate(
        self,
        *,
        first_frame,
        prompt,
        num_frames,
        size,
        pose_cond=None,
        seed=None,
    ):
        self.calls.append(
            {
                "first_frame": first_frame,
                "prompt": prompt,
                "num_frames": num_frames,
                "size": size,
                "pose_cond": pose_cond,
                "seed": seed,
                "pose_mode": os.environ.get("RSWM_POSE_MODE"),
            }
        )
        frames = np.repeat(np.asarray(first_frame)[None], num_frames, axis=0)
        if self.invalid_output:
            frames = frames[:, :-1]
        return frames


class _SpyBackboneFactory:
    def __init__(self, *, invalid_output: bool = False):
        self.invalid_output = invalid_output
        self.kwargs = []
        self.backends = []

    def __call__(self, **kwargs):
        self.kwargs.append(kwargs)
        backend = _SpyWanBackbone(invalid_output=self.invalid_output)
        self.backends.append(backend)
        return backend


class LingBotWorldModelAdapterTests(unittest.TestCase):
    def _adapter(self, factory, *, with_lora: bool = False):
        kwargs = {}
        if with_lora:
            kwargs.update(
                lora_identity=LoraIdentity(
                    target_modules=("attn_cam",), rank=16, alpha=16
                ),
                lora_path="/fixture/lora",
            )
        return LingBotWorldModelAdapter(
            checkpoint_identity=_checkpoint_identity(),
            checkpoint_path="/fixture/lingbot-world-base-cam",
            runtime_config=LingBotRuntimeConfig(
                device="cpu",
                sampling_steps=11,
                shift=8.0,
                guide_scale=4.0,
                t5_cpu=True,
                inactive_model_cpu=False,
                framewise=True,
            ),
            backbone_factory=factory,
            **kwargs,
        )

    def test_world_model_to_wan_mock_parameter_parity_and_lineage(self):
        factory = _SpyBackboneFactory()
        adapter = self._adapter(factory, with_lora=True)
        validate_world_model(adapter)
        self.assertEqual(factory.kwargs, [])

        image, condition, metadata = _model_input(TrackId.B_CAMERA)
        with patch.dict(os.environ, {"RSWM_POSE_MODE": "identity"}, clear=False):
            rollout = adapter.generate(image, condition, metadata)
            self.assertEqual(os.environ["RSWM_POSE_MODE"], "identity")

        self.assertEqual(len(factory.kwargs), 1)
        self.assertEqual(factory.kwargs[0]["checkpoint_dir"], "/fixture/lingbot-world-base-cam")
        self.assertEqual(factory.kwargs[0]["lora_path"], "/fixture/lora")
        self.assertEqual(factory.kwargs[0]["device"], "cpu")
        self.assertEqual(factory.kwargs[0]["config_key"], "i2v-A14B")
        self.assertEqual(factory.kwargs[0]["sampling_steps"], 11)

        call = factory.backends[0].calls[0]
        self.assertEqual(call["prompt"], condition.scene_prompt)
        self.assertEqual(call["num_frames"], 17)
        self.assertEqual(call["size"], "8*12")
        self.assertEqual(call["seed"], 123)
        self.assertEqual(call["pose_mode"], "native")
        np.testing.assert_array_equal(call["first_frame"], image)
        np.testing.assert_array_equal(call["pose_cond"]["c2w"], condition.poses)
        np.testing.assert_array_equal(
            call["pose_cond"]["intrinsics"], condition.intrinsics
        )

        self.assertEqual(rollout.frames.shape, (17, 8, 12, 3))
        self.assertEqual(rollout.frames.dtype, np.uint8)
        self.assertFalse(rollout.frames.flags.writeable)
        self.assertEqual(rollout.generation_seed, 123)
        self.assertEqual(rollout.model_id, "LingBot-World-base-cam")
        self.assertEqual(rollout.adapter_identity, "LingBotWorldAdapter")
        self.assertEqual(rollout.execution_identity.route, "benchmark-adapter")
        self.assertEqual(rollout.model_identity, adapter.model_identity)
        self.assertEqual(
            rollout.runtime_metadata["model_identity_checksum"],
            adapter.model_identity.identity_checksum,
        )
        self.assertEqual(
            adapter.capabilities.capability_partition,
            CapabilityPartition.ADAPTED,
        )

    def test_track_a_passes_no_pose_and_uses_drop_mode(self):
        factory = _SpyBackboneFactory()
        adapter = self._adapter(factory)
        image, condition, metadata = _model_input(TrackId.A_TEXT, seed=44)
        with patch.dict(os.environ, {"RSWM_POSE_MODE": "native"}, clear=False):
            adapter.generate(image, condition, metadata)
            self.assertEqual(os.environ["RSWM_POSE_MODE"], "native")
        call = factory.backends[0].calls[0]
        self.assertIsNone(call["pose_cond"])
        self.assertEqual(call["pose_mode"], "drop")
        self.assertEqual(call["seed"], 44)
        self.assertEqual(
            adapter.capabilities.capability_partition,
            CapabilityPartition.NATIVE,
        )

    def test_backend_is_built_once_and_checkpoint_chain_is_not_registry_global(self):
        factory = _SpyBackboneFactory()
        adapter = self._adapter(factory)
        image, condition, metadata = _model_input(TrackId.B_CAMERA)
        adapter.generate(image, condition, metadata)
        adapter.generate(image, condition, metadata)
        self.assertEqual(len(factory.kwargs), 1)
        self.assertEqual(len(factory.backends[0].calls), 2)

    def test_identity_namespaces_are_exact(self):
        adapter = self._adapter(_SpyBackboneFactory())
        identity = adapter.model_identity
        self.assertEqual(identity.release_identity, "LingBot-World-base-cam")
        self.assertEqual(identity.model_family, "Wan-I2V-A14B")
        self.assertEqual(
            identity.architecture_identity.transformer_type, "Wan-I2V-A14B"
        )
        self.assertEqual(identity.adapter_identity, "LingBotWorldAdapter")
        self.assertEqual(identity.execution_identity.route, "benchmark-adapter")
        self.assertIn("wan-inline", identity.execution_identity.legacy_aliases)
        self.assertEqual(adapter.capabilities.model_identity, identity)

    def test_mock_factory_cannot_promote_model_identity_to_verified(self):
        adapter = LingBotWorldModelAdapter(
            checkpoint_identity=_verified_checkpoint_identity(),
            checkpoint_path="/fixture/checkpoint",
            runtime_config=LingBotRuntimeConfig(device="cpu"),
            backbone_factory=_SpyBackboneFactory(),
        )
        self.assertEqual(adapter.checkpoint_identity.status, "verified")
        self.assertEqual(adapter.model_identity.status, "declared")

    def test_lora_identity_and_path_must_be_paired(self):
        lora = LoraIdentity(target_modules=("attn_cam",), rank=16, alpha=16)
        with self.assertRaises(ContractError):
            LingBotWorldModelAdapter(
                checkpoint_identity=_checkpoint_identity(),
                checkpoint_path="/fixture/checkpoint",
                lora_identity=lora,
                backbone_factory=_SpyBackboneFactory(),
            )
        with self.assertRaises(ContractError):
            LingBotWorldModelAdapter(
                checkpoint_identity=_checkpoint_identity(),
                checkpoint_path="/fixture/checkpoint",
                lora_path="/fixture/lora",
                backbone_factory=_SpyBackboneFactory(),
            )

    def test_runtime_identity_rejects_non_a14b_config(self):
        with self.assertRaises(ContractError):
            LingBotRuntimeConfig(config_key="ti2v-5B")

    def test_noncanonical_backend_output_is_rejected(self):
        adapter = self._adapter(_SpyBackboneFactory(invalid_output=True))
        image, condition, metadata = _model_input(TrackId.B_CAMERA)
        with self.assertRaises(ContractError):
            adapter.generate(image, condition, metadata)

    def test_rollout_rejects_partial_or_mismatched_identity(self):
        adapter = self._adapter(_SpyBackboneFactory())
        frames = np.zeros((17, 8, 12, 3), dtype=np.uint8)
        common = dict(
            frames=frames,
            sample_id="fixture",
            model_id="LingBot-World-base-cam",
            adapter_version="fixture",
            generation_seed=42,
        )
        with self.assertRaises(ContractError):
            Rollout(**common, model_identity=adapter.model_identity)
        with self.assertRaises(ContractError):
            Rollout(
                **common,
                model_identity=adapter.model_identity,
                adapter_identity="WrongAdapter",
                execution_identity=adapter.execution_identity,
            )
        wrong_model = dict(common)
        wrong_model["model_id"] = "WrongRelease"
        with self.assertRaises(ContractError):
            Rollout(
                **wrong_model,
                model_identity=adapter.model_identity,
                adapter_identity=adapter.adapter_identity,
                execution_identity=adapter.execution_identity,
            )


if __name__ == "__main__":
    unittest.main()
