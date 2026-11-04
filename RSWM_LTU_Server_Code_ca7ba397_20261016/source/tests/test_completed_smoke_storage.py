"""Temporary storage controls; no real smoke data or production raw is modified."""
import gzip
import importlib.util
import json
import os
from pathlib import Path

from PIL import Image
import pytest


spec = importlib.util.spec_from_file_location("completed_smoke_storage", Path(__file__).parents[1] / "scripts/deduplicate_completed_smoke.py")
storage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(storage)


@pytest.fixture
def sample(tmp_path):
    raw = tmp_path / "raw"
    submission = raw / "suite/runs/run/attempt-000/submission/main"
    submission.mkdir(parents=True)
    names = [f"suite/runs/run/attempt-000/submission/main/{i:04d}.png" for i in (1, 2)]
    for i, name in enumerate(names, 1):
        Image.new("RGB", (8, 8), (31, 73, 127)).save(raw / name)
        os.utime(raw / name, ns=(i * 10 ** 9, i * 10 ** 9))
    witness = "suite/receipts/run-000.json"
    (raw / witness).parent.mkdir()
    (raw / witness).write_text('{"retained":"original receipt"}\n')
    (raw / "suite/coordinator.lock").write_bytes(b"")
    original = {name: storage.file_hash(raw / name)[0] for name in storage.inventory(raw)}
    states = storage.verify_map(raw, original)
    stage = tmp_path / "stage"
    stage.mkdir()
    candidates = {name: {"scope": ("copy", "world", (960, 960), "output"), "run_id": "run"} for name in names}
    return raw, names, witness, original, states, stage, candidates


def replace(sample, tmp_path):
    raw, names, witness, _, states, stage, _ = sample
    with (tmp_path / "ledger.jsonl").open("x") as ledger:
        storage.replace_group(raw, names, states, {witness}, stage, ledger, 0)


def test_atomic_identical_group_retains_bytes_paths_and_original_resource_metadata(sample, tmp_path):
    raw, names, witness, original, before, stage, _ = sample
    replace(sample, tmp_path)
    after = storage.verify_map(raw, original)
    assert after[names[0]]["inode"] == after[names[1]]["inode"] == before[names[0]]["inode"]
    assert after[names[0]]["nlink"] == after[names[1]]["nlink"] == 2
    assert after[names[0]]["mtime_ns"] == before[names[0]]["mtime_ns"]
    assert after[names[1]]["mtime_ns"] != before[names[1]]["mtime_ns"]
    assert after[witness] == before[witness]
    assert storage.totals(after)["logical_bytes_per_path"] == storage.totals(before)["logical_bytes_per_path"]
    assert storage.totals(after)["unique_inode_allocated_bytes"] < storage.totals(before)["unique_inode_allocated_bytes"]
    assert not list(stage.iterdir())
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    replaced = next(r for r in rows if r["kind"] == "PATH_REPLACED")
    assert replaced["original"]["inode"] == before[names[1]]["inode"]
    assert replaced["result"]["sha256"] == original[names[1]]


def test_nonidentical_pngs_cannot_be_linked_even_if_caller_forces_a_group(sample, tmp_path, monkeypatch):
    raw, names, witness, _, _, stage, candidates = sample
    Image.new("RGB", (8, 8), (2, 4, 8)).save(raw / names[1])
    original = {name: storage.file_hash(raw / name)[0] for name in storage.inventory(raw)}
    states = storage.verify_map(raw, original)
    assert storage.duplicate_groups(candidates, states)[0] == []
    monkeypatch.setattr(storage.os, "link", lambda *a, **kw: pytest.fail("nonidentical mutation"))
    with (tmp_path / "ledger.jsonl").open("x") as ledger, pytest.raises(ValueError, match="identical"):
        storage.replace_group(raw, names, states, {witness}, stage, ledger, 0)


def test_symlink_substitution_fails_complete_map_preflight(sample, tmp_path):
    raw, names, _, original, before, _, _ = sample
    outside = tmp_path / "outside.png"
    outside.write_bytes((raw / names[1]).read_bytes())
    (raw / names[1]).unlink()
    (raw / names[1]).symlink_to(outside)
    with pytest.raises(ValueError, match="Symlink"):
        storage.verify_map(raw, original)
    assert storage.file_state(raw / names[0])["nlink"] == before[names[0]]["nlink"] == 1


def test_changed_receipt_between_preflight_and_group_prevents_any_link(sample, tmp_path, monkeypatch):
    raw, _, witness, _, _, stage, _ = sample
    (raw / witness).write_text('{"retained":"changed receipt after preflight"}\n')
    monkeypatch.setattr(storage.os, "link", lambda *a, **kw: pytest.fail("changed receipt mutation"))
    with pytest.raises(ValueError, match="guard changed"):
        replace(sample, tmp_path)
    assert not list(stage.iterdir())


def test_target_race_after_staging_retains_stage_and_never_replaces_target(sample, tmp_path, monkeypatch):
    raw, names, _, _, states, stage, _ = sample
    original_link = storage.os.link
    def race(source, destination, **kwargs):
        original_link(source, destination, **kwargs)
        (raw / names[1]).write_bytes(b"concurrent external writer")
    monkeypatch.setattr(storage.os, "link", race)
    monkeypatch.setattr(storage.os, "replace", lambda *a, **kw: pytest.fail("raced target replacement"))
    with pytest.raises(ValueError, match="guard changed"):
        replace(sample, tmp_path)
    staged = list(stage.iterdir())
    assert len(staged) == 1 and storage.file_hash(staged[0])[0] == states[names[0]]["sha256"]
    assert (raw / names[1]).read_bytes() == b"concurrent external writer"
    assert storage.file_state(raw / names[1])["inode"] == states[names[1]]["inode"]


def test_link_count_change_during_hash_is_rejected_with_exact_observed_fields(sample, monkeypatch):
    raw, names, _, original, before, stage, _ = sample
    original_read = storage.os.read
    changed = False
    def concurrent_link(fd, count):
        nonlocal changed
        block = original_read(fd, count)
        if block and not changed:
            changed = True
            os.link(raw / names[0], stage / "unexpected-link.png")
        return block
    monkeypatch.setattr(storage.os, "read", concurrent_link)
    with pytest.raises(storage.FileStateChanged) as error:
        storage.file_hash(raw / names[0])
    diagnostic = error.value.diagnostic
    assert diagnostic["changed_fields"] == {"nlink": {"before": 1, "after": 2}}
    assert diagnostic["phase"] == "hashing open descriptor" and diagnostic["cause"] == "UNKNOWN"
    assert storage.inventory(raw) == set(original)
    assert storage.verify_map(raw, original)[names[1]] == before[names[1]]


def test_same_bytes_pathname_substitution_during_hash_is_still_rejected(sample, monkeypatch):
    raw, names, _, original, before, stage, _ = sample
    replacement = stage / "replacement.png"
    replacement.write_bytes((raw / names[0]).read_bytes())
    os.utime(replacement, ns=(before[names[0]]["mtime_ns"], before[names[0]]["mtime_ns"]))
    original_read = storage.os.read
    changed = False
    def concurrent_substitution(fd, count):
        nonlocal changed
        block = original_read(fd, count)
        if block and not changed:
            changed = True
            os.rename(raw / names[0], stage / "retained-original.png")
            os.rename(replacement, raw / names[0])
        return block
    monkeypatch.setattr(storage.os, "read", concurrent_substitution)
    with pytest.raises(storage.FileStateChanged) as error:
        storage.file_hash(raw / names[0])
    diagnostic = error.value.diagnostic
    assert diagnostic["phase"] == "hashing pathname"
    assert diagnostic["changed_fields"] == {"inode": {"before": before[names[0]]["inode"],
                                                       "after": storage.file_state(raw / names[0])["inode"]}}
    assert storage.verify_map(raw, original)[names[0]]["sha256"] == original[names[0]]


def test_352_alias_group_verifies_every_retained_path_after_atomic_replacements(tmp_path):
    raw, stage = tmp_path / "raw", tmp_path / "stage"
    raw.mkdir()
    stage.mkdir()
    names = [f"{n:04d}.png" for n in range(352)]
    for n, name in enumerate(names, 1):
        Image.new("RGB", (8, 8), (31, 73, 127)).save(raw / name)
        os.utime(raw / name, ns=(n * 10 ** 9, n * 10 ** 9))
    original = {name: storage.file_hash(raw / name)[0] for name in names}
    before = storage.verify_map(raw, original)
    with (tmp_path / "ledger.jsonl").open("x") as ledger:
        storage.replace_group(raw, names, before, set(), stage, ledger, 0)
    after = storage.verify_map(raw, original)
    assert all(state["inode"] == before[names[0]]["inode"] and state["nlink"] == 352
               for state in after.values())
    assert all(state["mtime_ns"] == before[names[0]]["mtime_ns"] for state in after.values())
    assert not list(stage.iterdir())


@pytest.mark.parametrize("change", ["origin", "kind", "mode", "prelinked"])
def test_scope_mode_and_initial_link_count_limit_candidates(sample, tmp_path, change):
    raw, names, _, _, states, _, candidates = sample
    if change == "origin":
        candidates[names[1]]["scope"] = ("copy", "world", (1088, 960), "output")
    elif change == "kind":
        candidates[names[1]]["scope"] = ("copy", "world", (960, 960), "initial")
    elif change == "mode":
        os.chmod(raw / names[1], 0o600)
        states[names[1]] = {**storage.file_state(raw / names[1]), "sha256": storage.file_hash(raw / names[1])[0]}
    else:
        os.link(raw / names[0], tmp_path / "existing-external-link.png")
        states[names[0]] = {**storage.file_state(raw / names[0]), "sha256": storage.file_hash(raw / names[0])[0]}
    groups, excluded = storage.duplicate_groups(candidates, states)
    assert groups == []
    if change == "prelinked":
        assert excluded == [{"path": names[0], "reason": "INITIAL_NLINK_NOT_ONE", "nlink": 2}]


def command_args(raw, tmp_path, acceptance):
    (raw / "acceptance.json").write_text(json.dumps(acceptance))
    original = {name: storage.file_hash(raw / name)[0] for name in storage.inventory(raw)}
    mapping = tmp_path / "original-map.json"
    mapping.write_text(json.dumps(original))
    commands = tmp_path / "commands.json"
    commands.write_text(json.dumps([{"argv": ["python", "-m", "rswm_inria.smoke12", "--out", str(raw),
        "--code-commit", "a" * 40, "--execute"], "exit_code": 0}]))
    return ["--raw-root", str(raw), "--sha-map", str(mapping), "--commands", str(commands),
            "--out", str(tmp_path / "evidence"), "--work", str(tmp_path / "work"), "--reserve-bytes", str(1024 ** 2), "--execute"], original


@pytest.mark.parametrize("bad", [{"status": "PLANNED_NOT_RUN"}, {"full_smoke12_matrix": False},
    {"real_data_validated": False}, {"planned_runs_total": 575}, {"planned_unique_outputs_total": 50687},
    {"failed_runs": 1}, {"failed_runs": False}, {"failed_runs": ["incomplete-run"]}])
def test_partial_or_failed_smoke_never_mutates_raw(sample, tmp_path, monkeypatch, bad):
    raw = sample[0]
    acceptance = {"schema": "rswm-smoke12-acceptance-v1", "status": "REAL_DATA_VALIDATED", "real_data_validated": True,
        "full_smoke12_matrix": True, "planned_runs_total": 576, "planned_unique_outputs_total": 50688,
        "analysis_status": "ACCEPTED", "failed_runs": 0, "input_kind": "source_identity_checked_geography_unknown", **bad}
    args, original = command_args(raw, tmp_path, acceptance)
    before = storage.verify_map(raw, original)
    monkeypatch.setattr(storage.os, "link", lambda *a, **kw: pytest.fail("partial mutation"))
    assert storage.main(args) == 1
    assert storage.verify_map(raw, original) == before
    assert json.loads((tmp_path / "evidence/summary.json").read_text())["status"] == "FAILED"
    assert not (tmp_path / "work/stage").exists()


def test_evidence_inside_raw_is_rejected_without_writing_a_failure_file(sample, tmp_path):
    raw = sample[0]
    args, original = command_args(raw, tmp_path, {})
    args[args.index("--out") + 1] = str(raw / "unsafe-evidence")
    assert storage.main(args) == 1
    assert storage.inventory(raw) == set(original)
    assert not (raw / "unsafe-evidence").exists()


def test_malformed_acceptance_records_failed_without_any_raw_mutation(sample, tmp_path):
    raw = sample[0]
    args, original = command_args(raw, tmp_path, [])
    before = storage.verify_map(raw, original)
    assert storage.main(args) == 1
    assert storage.verify_map(raw, original) == before
    assert json.loads((tmp_path / "evidence/summary.json").read_text())["status"] == "FAILED"


@pytest.mark.parametrize("option", ["--out", "--work"])
def test_parent_traversal_cannot_place_evidence_or_stage_inside_raw(sample, tmp_path, option):
    raw = sample[0]
    args, original = command_args(raw, tmp_path, {})
    alias_parent = tmp_path / "alias-parent"
    alias_parent.mkdir()
    args[args.index(option) + 1] = str(alias_parent / ".." / "raw" / "unsafe-output")
    files_before = storage.verify_map(raw, original)
    tree_before = sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*"))
    assert storage.main(args) == 1
    assert sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*")) == tree_before
    assert storage.verify_map(raw, original) == files_before
    assert not (raw / "unsafe-output").exists()


def test_complete_sha_map_rejects_added_files_and_changed_bytes(sample):
    raw, names, _, original, _, _, _ = sample
    (raw / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory differs"):
        storage.verify_map(raw, original)
    (raw / "unexpected.json").unlink()
    (raw / names[1]).write_bytes(b"changed original bytes")
    with pytest.raises(ValueError, match="bytes changed"):
        storage.verify_map(raw, original)


@pytest.mark.parametrize("bad", [{"exit_code": 2}, {"exit_code": False}, {"timeout": True}])
def test_command_record_must_show_actual_completed_zero_exit(tmp_path, bad):
    raw = tmp_path / "raw"
    raw.mkdir()
    command = {"argv": ["python", "-m", "rswm_inria.smoke12", "--out", str(raw), "--code-commit", "a" * 40, "--execute"], "exit_code": 0, **bad}
    path = tmp_path / "commands.json"
    path.write_text(json.dumps([command]))
    with pytest.raises(ValueError, match="successful full-smoke"):
        storage.accepted_command(path, raw, {"code_commit": "a" * 40})


def test_forensics_directories_are_never_eligible(tmp_path):
    raw = tmp_path / "forensics-failed-smoke/raw"
    raw.mkdir(parents=True)
    with pytest.raises(ValueError, match="forensics"):
        storage.inventory(raw)


def test_sha_map_gzip_retains_exact_original_bytes_with_deterministic_header(tmp_path):
    original = tmp_path / "map.json"
    original.write_bytes(b'{"one.png":"retained original formatting"}\n')
    record = storage.gzip_copy(original, tmp_path / "map.json.gz")
    assert gzip.decompress((tmp_path / "map.json.gz").read_bytes()) == original.read_bytes()
    assert record["original_sha256"] == storage.file_hash(original)[0]
    assert (tmp_path / "map.json.gz").read_bytes()[4:8] == b"\0\0\0\0"


def test_actual_temporary_suite_journal_and_receipts_authorize_only_planned_png_paths(tmp_path, monkeypatch):
    """Actual tiny CPU pipeline, with gate quotas injected only for this fixture.

    These synthetic marker/command records test storage contract wiring; they
    are never retained as real-data experiment evidence or passed to a user.
    """
    import numpy as np
    from rswm_inria import suite
    from rswm_inria.aggregate import aggregate_suite
    from rswm_inria.tasks import packages
    task, _ = next(p for p in packages("storage-tiny-cpu-fixture", "dev", 0) if p[0]["family"] == "F2")
    task_path, world_path = tmp_path / "task.json", tmp_path / "world.png"
    task_path.write_text(json.dumps(task))
    Image.fromarray(np.zeros((2048, 2048, 3), np.uint8)).save(world_path)
    plan = suite.build_plan([{"task_path": str(task_path), "world_path": str(world_path), "city": "kitsap"}],
                            [{"model_id": "copy", "adapter": "copy"}], [0], "a" * 64, "b" * 40)
    kind = "source_identity_checked_geography_unknown"
    plan["diagnostic_acceptance_profile"] = {"profile": "smoke12", "input_kind": kind, "formal_test": False}
    plan["plan_sha256"] = storage.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    raw = tmp_path / "tiny-raw"
    raw.mkdir()
    (raw / "plan.json").write_text(json.dumps(plan))
    assert suite.execute_plan(plan, raw / "suite")[0]["status"] == "COMPLETED"
    analysis = aggregate_suite(plan, raw / "suite", bootstrap=1, seed=0)
    assert analysis["failure_summary"]["failed_or_unverifiable_runs"] == []
    acceptance = {"schema": "rswm-smoke12-acceptance-v1", "status": "REAL_DATA_VALIDATED", "real_data_validated": True,
        "full_smoke12_matrix": True, "planned_runs_total": 1, "planned_unique_outputs_total": 32,
        "analysis_status": analysis["acceptance_status"],
        "failed_runs": analysis["failure_summary"]["failed_or_unverifiable_runs"],
        "input_kind": kind, "plan_sha256": plan["plan_sha256"],
        "code_commit": "b" * 40, "source_code_sha256": plan["source_code_sha256"],
        "per_model": {"copy": {"runs": 1, "planned_outputs": 32, "valid_outputs": 32, "resource_complete_runs": 1}},
        "artifact_sha256": {p: storage.file_hash(raw / p)[0] for p in storage.inventory(raw)}}
    (raw / "acceptance.json").write_text(json.dumps(acceptance))
    original = {p: storage.file_hash(raw / p)[0] for p in storage.inventory(raw)}
    states = storage.verify_map(raw, original)
    commands = tmp_path / "tiny-commands.json"
    commands.write_text(json.dumps([{"argv": ["python", "-m", "rswm_inria.smoke12", "--execute", "--out", str(raw),
                                             "--code-commit", "b" * 40], "exit_code": 0}]))
    monkeypatch.setattr(storage, "FULL_RUNS", 1)
    monkeypatch.setattr(storage, "FULL_OUTPUTS", 32)
    candidates, witnesses = storage.full_gate(raw, original, states, commands)
    run = plan["expected"][0]["run_id"]
    prefix = f"suite/runs/{run}/attempt-000/submission/"
    assert set(candidates) == {prefix + "initial.png", *(prefix + f"main/{n:04d}.png" for n in range(1, 33))}
    assert f"suite/receipts/{run}-000.json" in witnesses[run]
    groups, _ = storage.duplicate_groups(candidates, states)
    assert len(groups) == 1 and len(groups[0]) == 32
    assert storage.verify_map(raw, original) == states  # This contract check performs no replacement.
    mapping = tmp_path / "tiny-original-map.json"
    mapping.write_text(json.dumps(original))
    assert storage.main(["--raw-root", str(raw), "--sha-map", str(mapping), "--commands", str(commands),
                         "--evidence", str(tmp_path / "estimate-evidence"), "--work", str(tmp_path / "estimate-work")]) == 0
    report = json.loads((tmp_path / "estimate-evidence/summary.json").read_text())
    assert report["status"] == "ESTIMATE_ONLY_NO_RAW_MUTATION" and report["replaceable_paths"] == 31
    assert not (tmp_path / "estimate-work/reserve.bin").exists()
    assert storage.verify_map(raw, original) == states


def test_failed_atomic_replacement_preserves_compressed_ledger_stage_and_reserve(sample, tmp_path, monkeypatch):
    raw, names, witness, _, before, _, candidates = sample
    for name in ("plan.json", "suite/plan.json", "suite/expected.jsonl", "suite/run-index.jsonl",
                 "suite/replay-status.json", "suite/replay-status.jsonl", "suite/receipts.jsonl"):
        (raw / name).write_text("{}\n")
    args, original = command_args(raw, tmp_path, {})
    # Inject only the small pure eligibility result, after exercising real map
    # preflight. Production CLI keeps its hard-coded full gate unchanged.
    monkeypatch.setattr(storage, "full_gate", lambda *a: (candidates, {"run": {witness}}))
    def denied(*args):
        raise OSError("fixture atomic replacement failure")
    monkeypatch.setattr(storage.os, "replace", denied)
    assert storage.main(args) == 1
    summary = json.loads((tmp_path / "evidence/summary.json").read_text())
    assert summary["status"] == "FAILED"
    ledger = tmp_path / "evidence/inode-ledger.jsonl.gz"
    records = [json.loads(row) for row in gzip.decompress(ledger.read_bytes()).splitlines()]
    assert [r["kind"] for r in records] == ["GROUP_PREPARED"]
    assert (tmp_path / "work/reserve.bin").is_file()
    staged = list((tmp_path / "work/stage").iterdir())
    assert len(staged) == 1 and storage.file_hash(staged[0])[0] == original[names[0]]
    assert storage.file_state(raw / names[1])["inode"] == before[names[1]]["inode"]
    assert storage.verify_map(raw, original)[witness]["sha256"] == original[witness]


def test_cli_failed_hash_retains_concrete_link_race_diagnostic(sample, tmp_path, monkeypatch):
    raw, names, witness, _, _, stage, candidates = sample
    for name in ("plan.json", "suite/plan.json", "suite/expected.jsonl", "suite/run-index.jsonl",
                 "suite/replay-status.json", "suite/replay-status.jsonl", "suite/receipts.jsonl"):
        (raw / name).write_text("{}\n")
    args, original = command_args(raw, tmp_path, {})
    armed, changed = False, False
    canonical_inode = storage.file_state(raw / names[0])["inode"]
    def eligible(*args):
        nonlocal armed
        armed = True
        return candidates, {"run": {witness}}
    original_read = storage.os.read
    def concurrent_link(fd, count):
        nonlocal changed
        block = original_read(fd, count)
        if armed and block and not changed and os.fstat(fd).st_ino == canonical_inode:
            changed = True
            os.link(raw / names[0], stage / "unexpected-link.png")
        return block
    monkeypatch.setattr(storage, "full_gate", eligible)
    monkeypatch.setattr(storage.os, "read", concurrent_link)
    monkeypatch.setattr(storage.os, "replace", lambda *a, **kw: pytest.fail("raced canonical replacement"))
    assert storage.main(args) == 1
    summary = json.loads((tmp_path / "evidence/summary.json").read_text())
    assert summary["status"] == "FAILED" and summary["error_type"] == "FileStateChanged"
    assert summary["file_state_mismatch"]["changed_fields"] == {"nlink": {"before": 1, "after": 2}}
    assert summary["file_state_mismatch"]["cause"] == "UNKNOWN"
    assert (tmp_path / "work/reserve.bin").is_file()
    assert storage.verify_map(raw, original)[witness]["sha256"] == original[witness]
