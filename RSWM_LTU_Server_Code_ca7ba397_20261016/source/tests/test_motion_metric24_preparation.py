"""The bounded data helper must reject unsafe inputs before any network access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest
from rswm_inria import data
from rswm_inria.geometry import file_hash


REPOSITORY = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("metric24_preparation", REPOSITORY / "scripts/prepare_motion_metric24.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.fixture
def frozen_source_fixture(tmp_path, monkeypatch):
    """Synthetic complete metadata exercises the real selector, without real evidence."""
    files = []
    for city in data.ROLES:
        split = "train" if city in ("austin", "chicago", "vienna", "kitsap") else "test"
        for number in range(1, 68):
            path = f"data/{split}/images/{city}{number}.tif"
            files.append({"path": path, "size_bytes": 75_040_362,
                          "lfs_sha256": hashlib.sha256(path.encode()).hexdigest()})
    for number in range(6):
        path = f"metadata/synthetic-{number}.json"
        files.append({"path": path, "size_bytes": 75_040_362,
                      "lfs_sha256": hashlib.sha256(path.encode()).hexdigest()})
    inventory = {"repo_id": data.REPO, "revision": data.INRIA_REVISION, "files": files}
    manifest = data.select_manifest(inventory)
    manifest_path, inventory_path = tmp_path / "manifest.json", tmp_path / "inventory.json"
    data.write_json(manifest_path, manifest)
    data.write_json(inventory_path, inventory)
    metric_rows = [row for row in manifest["worlds"] if row["role"] in helper.ROLE_CITIES]
    source_bytes = sum(row["size_bytes"] for row in metric_rows)
    assert len(files) == 542 and len(manifest["worlds"]) == 120 and len(metric_rows) == 24
    assert source_bytes == 1_800_968_688
    monkeypatch.setattr(helper, "MANIFEST_SHA", manifest["manifest_sha256"])
    monkeypatch.setattr(helper, "INVENTORY_FILE_SHA", file_hash(inventory_path))
    monkeypatch.setattr(helper, "SOURCE_BYTES", source_bytes)
    # Establish that this exact fixture passes all selection checks at a sufficient budget.
    assert helper.frozen_selection(manifest_path, inventory_path, 1.81) == (manifest, metric_rows)
    return manifest_path, inventory_path


def test_source_budget_rejects_before_network_or_raw_artifacts(tmp_path, monkeypatch, frozen_source_fixture):
    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *a, **kw: pytest.fail("download before budget rejection")
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    manifest_path, inventory_path = frozen_source_fixture
    raw, evidence = tmp_path / "raw", tmp_path / "evidence"
    rc = helper.main(["--manifest", str(manifest_path), "--inventory", str(inventory_path),
                      "--out", str(raw), "--evidence", str(evidence), "--smoke-root", str(tmp_path / "not-created"),
                      "--exposure-ledger", str(tmp_path / "persistent-ledger.json"),
                      "--max-source-gb", "1.8", "--download"])
    assert rc == 1 and not raw.exists()
    status_path = next((evidence / "invocations").glob("*/preparation-status.json"))
    status = json.loads(status_path.read_text())
    assert status["status"] == "BLOCKED" and "source budget" in status["reason"]


def test_truncated_inventory_cannot_certify_same_selected_manifest(tmp_path, frozen_source_fixture):
    manifest_path, inventory_path = frozen_source_fixture
    inventory = json.loads(inventory_path.read_text())
    selected = {row["path"] for row in json.loads(manifest_path.read_text())["worlds"]}
    inventory["files"] = [row for row in inventory["files"] if row["path"] in selected]
    path = tmp_path / "incomplete-inventory.json"
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="fixed complete"):
        helper.frozen_selection(manifest_path, path, 1.81)


def test_symlink_source_ancestor_is_rejected_before_read(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        helper.no_symlinks(alias / "source.tif")


def test_parent_traversal_is_rejected_before_read(tmp_path):
    ordinary = tmp_path / "ordinary"
    ordinary.mkdir()
    source = tmp_path / "source.tif"
    source.write_bytes(b"preserve-source")
    with pytest.raises(ValueError, match="Parent traversal"):
        helper.no_symlinks(ordinary / ".." / source.name)
    assert source.read_bytes() == b"preserve-source"


@pytest.mark.parametrize("changed", ["plan_sha256", "run_id", "submission_status", "replay_verified"])
def test_untrusted_completed_outcome_does_not_release_smoke_reservation(tmp_path, changed):
    plan = {"plan_sha256": "current", "expected": [{"run_id": "run", "family": "F7"}]}
    outcome = {"plan_sha256": "current", "run_id": "run", "attempt": 0, "status": "COMPLETED",
               "execution_status": "COMPLETED", "submission_status": "VERIFIED", "replay_verified": True}
    outcome[changed] = False if changed == "replay_verified" else "wrong"
    path = tmp_path / "runs/run/attempt-000/outcome.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(outcome))
    assert helper.remaining_smoke_slots(plan, tmp_path) == (160, 0)


@pytest.mark.parametrize("tamper", ["outcome", "receipt", "resource", "head"])
def test_changed_completed_artifact_keeps_full_reservation(tmp_path, tamper):
    """A completed marker cannot release space after its receipt chain changes."""
    plan = {"plan_sha256": "current", "expected": [{"run_id": "run", "family": "F7"}]}
    identity = {"plan_sha256": "current", "run_id": "run", "attempt": 0}
    folder = tmp_path / "runs/run/attempt-000"
    outcome_path, resource_path = folder / "outcome.json", folder / "resources.json"
    receipt_path, head_path = tmp_path / "receipts/run-000.json", tmp_path / "heads/run-000.json"
    head = "a" * 64
    outcome = {**identity, "status": "COMPLETED", "execution_status": "COMPLETED",
               "submission_status": "VERIFIED", "resource_status": "COMPLETE", "replay_verified": True,
               "replay_head": head}
    resource = {**identity, "measurement_complete": True}

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    save(outcome_path, outcome)
    save(resource_path, resource)
    save(receipt_path, {**identity, "status": "COMPLETED", "replay_head": head,
                        "outcome_sha256": helper.file_hash(outcome_path)})
    save(head_path, {**identity, "journal_head": head})
    save(folder / "artifact-manifest.json", {**identity, "files": {
        "outcome.json": helper.file_hash(outcome_path), "resources.json": helper.file_hash(resource_path)}})
    assert helper.remaining_smoke_slots(plan, tmp_path) == (0, 1)
    if tamper == "outcome":
        save(outcome_path, {**outcome, "unbound_change": "after-completion"})
    elif tamper == "receipt":
        receipt_path.unlink()
    elif tamper == "resource":
        save(resource_path, {**resource, "unbound_change": "after-completion"})
    else:
        save(head_path, {**identity, "journal_head": "b" * 64})
    assert helper.remaining_smoke_slots(plan, tmp_path) == (160, 0)
