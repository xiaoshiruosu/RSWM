from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from benchmark.core.errors import (
    ArtifactLineageError,
    ContractError,
    IdentityError,
    SchemaError,
)
from benchmark.core.identity import (
    ArchitectureIdentity,
    ArtifactLineage,
    ArtifactRef,
    BenchmarkReleaseIdentity,
    CameraConditionIdentity,
    CheckpointIdentity,
    ComponentIdentity,
    ExecutionIdentity,
    LoraIdentity,
    ModelIdentity,
    canonical_sha256,
)
from benchmark.spec import load_benchmark_spec, schema_bundle_descriptor
from models.mock_world_model import DeterministicMockWorldModel
from tests.benchmark.fixture_factory import FixtureTempDirectory, create_fixture


def digest(label: str) -> str:
    return canonical_sha256({"fixture": label})


def lingbot_model_identity(
    *,
    status: str = "declared",
    checkpoint_status: str = "declared",
    route: str = "wan-inline",
) -> ModelIdentity:
    verified = checkpoint_status == "verified"
    return ModelIdentity.create(
        model_family="Wan-I2V-A14B",
        release_identity="LingBot-World-base-cam",
        adapter_identity="LingBotWorldAdapter",
        checkpoint_identity=CheckpointIdentity(
            repository="robbyant/lingbot-world-base-cam",
            revision="fixture-revision" if verified else None,
            sha256_manifest=digest("checkpoint-manifest") if verified else None,
            config_hash=digest("checkpoint-config") if verified else None,
            status=checkpoint_status,
        ),
        architecture_identity=ArchitectureIdentity(
            transformer_type="WanModel/I2V-A14B",
            vae="Wan2.1_VAE",
            text_encoder="umT5-XXL",
            camera_conditioning_type="Plucker-ray camera conditioning",
            implementation_aliases=("wan",),
        ),
        camera_condition_identity=CameraConditionIdentity(
            encoding="plucker_ray_v1",
            pose_injection="global projection plus per-transformer-block modulation",
            injector_type="camera injector with scale and shift",
        ),
        lora_identity=LoraIdentity(
            target_modules=("attn_cam",),
            rank=16,
            alpha=16,
        ),
        execution_identity=ExecutionIdentity(
            route=route,
            entrypoint="lora_remotesense/new/evaluate.py",
            legacy_aliases=("wan", "wan-inline"),
        ),
        status=status,
    )


class ReleaseIdentityAndSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temp = FixtureTempDirectory()
        self.paths = create_fixture(Path(self.temp.name))
        self.spec = load_benchmark_spec(self.paths.spec)

    def tearDown(self):
        self.temp.cleanup()

    def test_declared_evaluator_is_not_release_ready(self):
        release = self.spec.release_identity()
        self.assertFalse(release.is_release_ready)
        release.validate(require_verified=False)
        with self.assertRaises(IdentityError):
            release.validate(require_verified=True)

    def test_verified_evaluator_completes_release_identity(self):
        evaluator = ComponentIdentity(
            kind="evaluator",
            artifact_id="fixture-evaluator",
            schema="rswm.evaluator/v1",
            version="v1",
            sha256=digest("evaluator"),
            status="verified",
        )
        verified_spec = replace(
            self.spec,
            evaluator=evaluator,
            spec_checksum=None,
        ).with_checksum()
        release = verified_spec.release_identity()
        self.assertTrue(release.is_release_ready)
        release.validate(require_verified=True)
        self.assertEqual(release, type(release).from_document(release.to_document()))

    def test_spec_checksum_detects_tampering(self):
        document = json.loads(self.paths.spec.read_text(encoding="utf-8"))
        document["task_parameters"]["overlap_level"] = "L1_overlap_80"
        tampered = Path(self.temp.name) / "tampered_spec.json"
        tampered.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(SchemaError):
            load_benchmark_spec(tampered)

    def test_machine_readable_schema_bundle_is_bound(self):
        descriptor = schema_bundle_descriptor()
        self.assertEqual(len(descriptor["files"]), 4)
        self.assertEqual(
            descriptor["files"]["benchmark_spec_v1.schema.json"]["$id"],
            "rswm.benchmark_spec/v1",
        )
        self.assertEqual(
            descriptor["files"]["model_identity_v1.schema.json"]["$id"],
            "urn:rswm:model-identity:v1",
        )
        self.assertEqual(
            descriptor["external_contracts"]["model_identity"],
            "rswm.model_identity/v1",
        )
        self.assertEqual(
            self.spec.schema_bundle.sha256,
            self.spec.release_identity().schema_bundle.sha256,
        )

    def test_model_identity_schema_is_optional_extension_of_release_and_lineage(self):
        schema_dir = Path(__file__).resolve().parents[2] / "benchmark" / "spec" / "schemas"
        model_schema = json.loads(
            (schema_dir / "model_identity_v1.schema.json").read_text(encoding="utf-8")
        )
        required = set(model_schema["required"])
        self.assertTrue(
            {
                "model_family",
                "release_identity",
                "adapter_identity",
                "checkpoint_identity",
                "architecture_identity",
                "camera_condition_identity",
                "lora_identity",
                "execution_identity",
            }.issubset(required)
        )
        for filename in (
            "release_identity_v1.schema.json",
            "artifact_lineage_v1.schema.json",
        ):
            document = json.loads((schema_dir / filename).read_text(encoding="utf-8"))
            self.assertIn("model_identity", document["properties"])
            self.assertNotIn("model_identity", document["required"])

    def test_layered_model_identity_roundtrip_preserves_namespaces(self):
        identity = lingbot_model_identity()
        document = identity.to_document()
        self.assertEqual(document["release_identity"], "LingBot-World-base-cam")
        self.assertEqual(document["adapter_identity"], "LingBotWorldAdapter")
        self.assertEqual(
            document["architecture_identity"]["implementation_aliases"], ["wan"]
        )
        self.assertEqual(document["execution_identity"]["route"], "wan-inline")
        self.assertIn("wan-inline", document["execution_identity"]["legacy_aliases"])
        self.assertEqual(identity, ModelIdentity.from_document(document))

    def test_verified_checkpoint_identity_requires_content_anchors(self):
        with self.assertRaises(IdentityError):
            CheckpointIdentity(
                repository="robbyant/lingbot-world-base-cam",
                status="verified",
            )
        identity = lingbot_model_identity(
            status="verified", checkpoint_status="verified"
        )
        identity.validate(require_verified=True)

    def test_legacy_release_document_and_checksum_omit_optional_model_identity(self):
        release = self.spec.release_identity()
        document = release.to_document()
        self.assertNotIn("model_identity", document)
        self.assertEqual(release, BenchmarkReleaseIdentity.from_document(document))
        positional = BenchmarkReleaseIdentity(
            release.benchmark_version,
            release.protocol,
            release.benchmark_spec,
            release.schema_bundle,
            release.evaluator,
            release.manifest,
            release.release_checksum,
            release.schema,
        )
        self.assertEqual(release, positional)
        with self.assertRaises(IdentityError):
            release.validate(require_model_identity=True)

    def test_release_and_lineage_bind_layered_model_identity(self):
        identity = lingbot_model_identity()
        legacy_release = self.spec.release_identity()
        release = BenchmarkReleaseIdentity.create(
            benchmark_version=legacy_release.benchmark_version,
            protocol=legacy_release.protocol,
            benchmark_spec=legacy_release.benchmark_spec,
            schema_bundle=legacy_release.schema_bundle,
            evaluator=legacy_release.evaluator,
            manifest=legacy_release.manifest,
            model_identity=identity,
        )
        artifacts = (
            ArtifactRef("checkpoint", "lingbot-world-base-cam", digest("checkpoint")),
            ArtifactRef("adapter", "LingBotWorldAdapter", digest("adapter")),
            ArtifactRef(
                "benchmark_spec",
                "fixture-spec",
                release.benchmark_spec.sha256,
            ),
            ArtifactRef("manifest", "fixture-manifest", release.manifest.sha256),
            ArtifactRef("metric", "fixture-metric", digest("metric")),
            ArtifactRef("result", "fixture-result", digest("result")),
        )
        lineage = ArtifactLineage.create(artifacts, model_identity=identity)
        lineage.validate(require_complete=True, require_model_identity=True)
        lineage.assert_matches_release(release)
        self.assertEqual(lineage, ArtifactLineage.from_document(lineage.to_document()))

        different_identity = lingbot_model_identity(route="benchmark-adapter")
        mismatched = ArtifactLineage.create(
            artifacts, model_identity=different_identity
        )
        with self.assertRaises(ArtifactLineageError):
            mismatched.assert_matches_release(release)

    def test_world_model_capabilities_expose_optional_layered_identity(self):
        legacy = DeterministicMockWorldModel.capabilities
        self.assertIsNone(legacy.model_identity)
        self.assertIsNone(legacy.release_identity)
        identity = lingbot_model_identity()
        aligned = replace(legacy, model_identity=identity)
        self.assertEqual(aligned.release_identity, "LingBot-World-base-cam")
        self.assertEqual(aligned.adapter_identity, "LingBotWorldAdapter")
        self.assertEqual(aligned.execution_identity.route, "wan-inline")

        unchecked = replace(identity, identity_checksum=None)
        with self.assertRaises(ContractError):
            replace(legacy, model_identity=unchecked)

    def test_complete_artifact_lineage_matches_release(self):
        release = self.spec.release_identity()
        artifacts = (
            ArtifactRef("checkpoint", "mock-no-weights", digest("checkpoint")),
            ArtifactRef("adapter", "mock-adapter-v1", digest("adapter")),
            ArtifactRef(
                "benchmark_spec",
                "fixture-spec",
                release.benchmark_spec.sha256,
                schema=release.benchmark_spec.schema,
            ),
            ArtifactRef(
                "manifest",
                "fixture-manifest",
                release.manifest.sha256,
                schema=release.manifest.schema,
            ),
            ArtifactRef("metric", "fixture-metric", digest("metric")),
            ArtifactRef("result", "fixture-result", digest("result")),
        )
        lineage = ArtifactLineage.create(artifacts)
        lineage.validate(require_complete=True)
        lineage.assert_matches_release(release)
        self.assertEqual(lineage, ArtifactLineage.from_document(lineage.to_document()))
        self.assertEqual(
            lineage,
            ArtifactLineage(
                lineage.artifacts,
                lineage.lineage_checksum,
                lineage.schema,
            ),
        )

    def test_lineage_order_is_fail_closed(self):
        with self.assertRaises(ArtifactLineageError):
            ArtifactLineage.create(
                (
                    ArtifactRef("checkpoint", "c", digest("c")),
                    ArtifactRef("manifest", "m", digest("m")),
                    ArtifactRef("adapter", "a", digest("a")),
                )
            )

    def test_lineage_may_not_skip_adapter(self):
        with self.assertRaises(ArtifactLineageError):
            ArtifactLineage.create(
                (
                    ArtifactRef("checkpoint", "c", digest("c-skip")),
                    ArtifactRef("benchmark_spec", "s", digest("s-skip")),
                )
            )

    def test_verified_component_requires_digest(self):
        with self.assertRaises(IdentityError):
            ComponentIdentity(
                kind="evaluator",
                artifact_id="bad",
                schema="x/v1",
                version="v1",
                sha256=None,
                status="verified",
            )


if __name__ == "__main__":
    unittest.main()
