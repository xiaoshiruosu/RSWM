"""I/O faults cannot advance a journal or publish an accepted summary."""
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rswm_inria import runner
from rswm_inria.baselines import OracleMap
from rswm_inria.geometry import crop
from rswm_inria.replay import load_rgb_png, replay_run, schedule
from rswm_inria.tasks import packages


class FaultStream:
    def __init__(self, stream, fault):
        self.stream, self.fault = stream, fault
    def __enter__(self):
        self.stream.__enter__()
        return self
    def __exit__(self, *args):
        result = self.stream.__exit__(*args)
        if self.fault == 'close':
            raise OSError('injected close failure')
        return result
    def __getattr__(self, name):
        return getattr(self.stream, name)
    def write(self, content):
        if self.fault == 'write':
            raise OSError('injected write failure')
        if self.fault == 'short':
            return self.stream.write(content[:-1])
        return self.stream.write(content)
    def flush(self):
        if self.fault == 'flush':
            raise OSError('injected flush failure')
        return self.stream.flush()


@pytest.mark.parametrize('fault', ['open', 'write', 'short', 'flush', 'fsync', 'close'])
def test_failed_append_never_commits_memory_or_continues(tmp_path, monkeypatch, fault):
    journal = runner.Journal(tmp_path / 'run')
    original_open = builtins.open
    def failing_open(path, *args, **kwargs):
        if Path(path).name == 'events.jsonl':
            if fault == 'open':
                raise OSError('injected open failure')
            return FaultStream(original_open(path, *args, **kwargs), fault)
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(runner, 'open', failing_open, raising=False)
        if fault == 'fsync':
            patch.setattr(runner.os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError('injected sync failure')))
        with pytest.raises(OSError):
            journal.append('request', stream='main', start=1, stop=2, queries=[[0, 0]])
    assert journal.rows == [] and journal.head == '0' * 64
    # The possibly written prefix remains as evidence; a failed writer is not resumed.
    with pytest.raises(OSError):
        journal.append('response', stream='main', start=1, stop=2)


@pytest.fixture
def f2():
    task, labels = packages('journal-io-control', 'diagnostic_only', 0)[0]
    world = np.full((2048, 2048, 3), 73, np.uint8)
    return task, labels, world


def test_tail_loss_before_summary_is_rejected(tmp_path, monkeypatch, f2):
    task, labels, world = f2
    original_append = runner.Journal.append
    root = tmp_path / 'run'
    def truncate_after_last_output(journal, kind, **payload):
        result = original_append(journal, kind, **payload)
        if kind == 'output' and payload['event'] == len(task['programs']['main']) - 1:
            path = root / 'events.jsonl'
            lines = path.read_bytes().splitlines(keepends=True)
            path.write_bytes(b''.join(lines[:-1]))
        return result
    monkeypatch.setattr(runner.Journal, 'append', truncate_after_last_output)
    with pytest.raises((OSError, ValueError)):
        runner.run_task(task, world, OracleMap(world, task['origin']), labels=labels, out_dir=root)
    assert not (root / 'summary.json').exists()
    assert (root / 'main' / '0032.png').exists()


def test_replacement_before_summary_is_rejected(tmp_path, monkeypatch, f2):
    task, labels, world = f2
    original_append = runner.Journal.append
    root = tmp_path / 'run'
    def replace_after_last_output(journal, kind, **payload):
        result = original_append(journal, kind, **payload)
        if kind == 'output' and payload['event'] == len(task['programs']['main']) - 1:
            path = root / 'events.jsonl'
            replacement = root / 'foreign.jsonl'
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        return result
    monkeypatch.setattr(runner.Journal, 'append', replace_after_last_output)
    with pytest.raises((OSError, ValueError)):
        runner.run_task(task, world, OracleMap(world, task['origin']), labels=labels, out_dir=root)
    assert not (root / 'summary.json').exists()


def test_same_inode_same_length_edit_before_summary_is_rejected(tmp_path, monkeypatch, f2):
    task, labels, world = f2
    original_append = runner.Journal.append
    root = tmp_path / 'run'
    def edit_after_last_output(journal, kind, **payload):
        result = original_append(journal, kind, **payload)
        if kind == 'output' and payload['event'] == len(task['programs']['main']) - 1:
            path = root / 'events.jsonl'
            original = path.read_bytes()
            edited = original.replace(b'"seed": 0', b'"seed": 1', 1)
            assert edited != original and len(edited) == len(original)
            inode = path.stat().st_ino
            path.write_bytes(edited)
            assert path.stat().st_ino == inode
        return result
    monkeypatch.setattr(runner.Journal, 'append', edit_after_last_output)
    with pytest.raises(OSError, match='Retained journal bytes'):
        runner.run_task(task, world, OracleMap(world, task['origin']), labels=labels, out_dir=root)
    assert not (root / 'summary.json').exists()


def test_lr64_tail_loss_cannot_publish_summary(tmp_path, monkeypatch):
    from rswm_inria import lr64
    task = lr64.task_for('journal-io-lr64', 'exact')
    world = np.full((1024, 1024, 3), 73, np.uint8)
    root = tmp_path / 'run'
    original_append = runner.Journal.append
    def truncate_after_last_output(journal, kind, **payload):
        result = original_append(journal, kind, **payload)
        if kind == 'output' and payload['event'] == len(task['q']) - 1:
            path = root / 'events.jsonl'
            path.write_bytes(b''.join(path.read_bytes().splitlines(keepends=True)[:-1]))
        return result
    monkeypatch.setattr(runner.Journal, 'append', truncate_after_last_output)
    with pytest.raises(OSError):
        lr64.run_task(task, world, lr64._NativeOracle(world), out_dir=root)
    assert not (root / 'summary.json').exists()


def test_suite_evaluator_io_failure_keeps_denominator_and_attempt_cost(tmp_path, monkeypatch, f2):
    from PIL import Image
    from rswm_inria import suite
    task, _, world = f2
    tp, wp = tmp_path / 'task.json', tmp_path / 'world.png'
    tp.write_text(json.dumps(task))
    Image.fromarray(world).save(wp)
    plan = suite.build_plan([{'task_path':str(tp), 'world_path':str(wp), 'city':'kitsap'}],
                            [{'model_id':'oracle', 'adapter':'oracle', 'device':'cpu'}], [0], 'a'*64, 'b'*40)
    original_append = runner.Journal.append
    def fail_output(journal, kind, **payload):
        if kind == 'output':
            if journal.root is not None:
                raise OSError('injected evaluator journal failure')
        return original_append(journal, kind, **payload)
    monkeypatch.setattr(runner.Journal, 'append', fail_output)
    root = tmp_path / 'suite'
    outcome = suite.execute_plan(plan, root)[0]
    assert outcome['status'] == 'CRASHED' and outcome['error_type'] == 'OSError'
    assert not outcome['replay_verified'] and outcome['submission_status'] == 'NOT_VERIFIED'
    resource = suite.safe_resource_attempts(plan, root)[0]['resources']
    assert resource['accounting_calls']['planned_output_slots'] == 32
    # Offline F2 enters one generate phase for all 32 planned output slots.
    assert resource['accounting_calls']['requests_issued'] == 32
    generate = [phase for phase in resource['phases'] if phase['phase'] == 'generate']
    assert len(generate) == 1 and generate[0]['count'] == 32
    assert resource['accounting_calls']['model_calls'] == 3  # load, initialize, generate
    assert resource['execution_complete'] is False
    assert generate[0]['wall_seconds'] > 0  # observed cost survives evaluator failure
    from rswm_inria.aggregate import aggregate_suite
    report = aggregate_suite(plan, root, bootstrap=2, seed=0)
    assert report['acceptance_status'] != 'ACCEPTED' and report['failure_summary']['expected_runs'] == 1


def test_failed_png_directory_prevents_writer_continuation(tmp_path, monkeypatch):
    journal = runner.Journal(tmp_path / 'run')
    journal.append('session', diagnostic=True)
    original_mkdir = Path.mkdir
    def fail_directory(path, *args, **kwargs):
        if path.name == 'main':
            raise OSError('injected PNG directory failure')
        return original_mkdir(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'mkdir', fail_directory)
        with pytest.raises(OSError):
            journal.image('main', 1, np.zeros((240,416,3), np.uint8))
    with pytest.raises(OSError):
        journal.append('output', diagnostic=True)
    assert len(journal.rows) == 1


@pytest.mark.parametrize('fault', ['short', 'close'])
def test_incomplete_artifact_write_cannot_publish_final_name(tmp_path, monkeypatch, fault):
    from rswm_inria.artifact_io import save_png_exclusive
    original_open = Path.open
    def failing_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return FaultStream(stream, fault) if path.suffix == '.tmp' else stream
    monkeypatch.setattr(Path, 'open', failing_open)
    with pytest.raises(OSError):
        save_png_exclusive(tmp_path / 'output.png', np.zeros((4,4,3),np.uint8))
    assert list(tmp_path.iterdir()) == []


def test_directory_sync_failure_retains_complete_published_artifact(tmp_path, monkeypatch):
    from rswm_inria import artifact_io
    from PIL import Image
    original_fsync = artifact_io.os.fsync
    calls = []
    def fail_directory(fd):
        import stat
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError('injected directory synchronization failure')
        calls.append(fd)
        original_fsync(fd)
    monkeypatch.setattr(artifact_io.os, 'fsync', fail_directory)
    pixels = np.full((4,4,3),73,np.uint8)
    path = tmp_path / 'output.png'
    with pytest.raises(OSError):
        artifact_io.save_png_exclusive(path,pixels)
    with Image.open(path) as image:
        assert np.array_equal(np.asarray(image), pixels)
    assert calls and list(tmp_path.iterdir()) == [path]


def test_process_exit_before_summary_remains_crashed_first_attempt(tmp_path, f2):
    from PIL import Image
    from rswm_inria import suite
    task, _, world = f2
    tp, wp = tmp_path / 'task.json', tmp_path / 'world.png'
    tp.write_text(json.dumps(task))
    Image.fromarray(world).save(wp)
    plan = suite.build_plan([{'task_path':str(tp), 'world_path':str(wp), 'city':'kitsap'}],
                            [{'model_id':'oracle', 'adapter':'oracle', 'device':'cpu'}], [0], 'a'*64, 'b'*40)
    pp = tmp_path / 'plan.json'
    pp.write_text(json.dumps(plan))
    root = tmp_path / 'suite'
    driver = "import json,os,sys;from rswm_inria import runner,suite;original=runner.Journal.verify_persisted;runner.Journal.verify_persisted=lambda self:os._exit(77) if self.root else original(self);suite.execute_plan(json.load(open(sys.argv[1])),sys.argv[2])"
    env = {**os.environ, 'PYTHONPATH':str(Path(runner.__file__).resolve().parents[1])}
    completed = subprocess.run([sys.executable, '-c', driver, str(pp), str(root)],
                               cwd=tmp_path, env=env, capture_output=True, timeout=30)
    assert completed.returncode == 77, completed.stderr.decode()
    attempt = root / 'runs' / plan['expected'][0]['run_id'] / 'attempt-000'
    assert len(list((attempt / 'submission' / 'main').glob('*.png'))) == 32
    assert not (attempt / 'submission' / 'summary.json').exists()
    assert not (attempt / 'outcome.json').exists()
    outcome = suite.execute_plan(plan, root, resume=True)[0]
    assert outcome['status'] == 'CRASHED' and outcome['attempt'] == 0
    assert not outcome['replay_verified']
    assert not (attempt.parent / 'attempt-001').exists()


def test_same_empty_attempt_has_exactly_one_writer(tmp_path):
    # Empty pre-created worker directories stay supported, but ownership must be exclusive.
    root = tmp_path / 'run'
    root.mkdir()
    first = runner.Journal(root)
    with pytest.raises(FileExistsError):
        runner.Journal(root)
    first.append('session', diagnostic=True)
    assert len(first.rows) == 1


def test_png_preceding_failed_journal_keeps_failure_evidence(tmp_path, monkeypatch, f2):
    task, labels, world = f2
    original_append = runner.Journal.append
    root = tmp_path / 'run'
    def fail_output(journal, kind, **payload):
        if kind == 'output':
            raise OSError('injected journal failure after PNG')
        return original_append(journal, kind, **payload)
    monkeypatch.setattr(runner.Journal, 'append', fail_output)
    with pytest.raises(OSError):
        runner.run_task(task, world, OracleMap(world, task['origin']), labels=labels, out_dir=root)
    assert (root / 'main' / '0001.png').is_file()
    assert not (root / 'summary.json').exists()
    with pytest.raises(ValueError):
        replay_run(task, crop(world, (0, 0), task['origin']), root)


def test_append_detects_external_truncation_and_cannot_repair(tmp_path):
    journal = runner.Journal(tmp_path / 'run')
    journal.append('session', diagnostic=True)
    path = journal.root / 'events.jsonl'
    path.write_bytes(b'')
    with pytest.raises(OSError):
        journal.append('response', diagnostic=True)
    assert len(journal.rows) == 1 and path.read_bytes() == b''


def test_submission_schedule_and_png_identity_stay_unchanged(tmp_path, f2):
    task, labels, world = f2
    result, journal = runner.run_task(task, world, OracleMap(world, task['origin']), labels=labels, out_dir=tmp_path / 'run')
    rows = [json.loads(line) for line in (journal.root / 'events.jsonl').read_text().splitlines()]
    expected = list(schedule(task))
    actual = [r for r in rows if r['kind'] in {'request', 'response', 'output'}]
    assert len(actual) == len(expected)
    assert len(rows) == 1 + len(task['programs']) + len(expected)
    replayed = replay_run(task, load_rgb_png(journal.root / 'initial.png'), journal.root, result['journal_head'])
    assert replayed['verified_png_outputs'] == 32 and replayed['reports'] == result['reports']
