from __future__ import annotations

from dataclasses import replace
import unittest

from benchmark.core.identity import CheckpointIdentity, ComponentIdentity
from baselines import BaselineIdentity, BaselineIdentityError
from tests.benchmark.baseline_fixture_factory import baseline_identity, digest


class BaselineIdentityTests(unittest.TestCase):
    def test_roundtrip_keeps_model_adapter_checkpoint_execution_separate(self):
        identity = baseline_identity()
        document = identity.to_document()
        self.assertEqual(
            document["adapter_identity"]["artifact_id"],
            document["model_identity"]["adapter_identity"],
        )
        self.assertEqual(
            document["checkpoint_identity"],
            document["model_identity"]["checkpoint_identity"],
        )
        self.assertEqual(
            document["execution_identity"],
            document["model_identity"]["execution_identity"],
        )
        self.assertEqual(identity, BaselineIdentity.from_document(document))

    def test_mismatched_checkpoint_is_rejected(self):
        identity = baseline_identity()
        other = CheckpointIdentity(repository="fixture/other", status="declared")
        with self.assertRaises(BaselineIdentityError):
            replace(identity, checkpoint_identity=other, identity_checksum=None)

    def test_free_string_cannot_replace_structured_adapter_identity(self):
        identity = baseline_identity()
        with self.assertRaises(BaselineIdentityError):
            replace(identity, adapter_identity="some model name", identity_checksum=None)

    def test_adapter_artifact_must_match_model_identity(self):
        identity = baseline_identity()
        mismatched = ComponentIdentity(
            kind="baseline_adapter",
            artifact_id="DifferentAdapter",
            schema="rswm.baseline_adapter/v1",
            version="v1",
            sha256=digest("different-adapter"),
            status="verified",
        )
        with self.assertRaises(BaselineIdentityError):
            replace(
                identity,
                adapter_identity=mismatched,
                identity_checksum=None,
            )


if __name__ == "__main__":
    unittest.main()
