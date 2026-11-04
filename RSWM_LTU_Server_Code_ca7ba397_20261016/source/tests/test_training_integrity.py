"""Backend/checkpoint provenance controls; no toy objective or real model claim."""
import json
import py_compile
from types import SimpleNamespace
import pytest
from rswm_inria.geometry import digest, file_hash
from rswm_inria.training import (locked_tree, load_approved_backend,
    validate_resume_cursor, validate_checkpoint_receipt, validate_dev_index)


def test_stale_entry_and_sibling_bytecode_cannot_bypass_locked_source(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    entry = source / 'backend.py'
    entry.write_text('def load_backend(**kwargs):\n return None\n')
    weights = tmp_path / 'weights'; weights.mkdir()
    (weights / 'fixture.bin').write_bytes(b'fixture only')
    manifest = {'source_dir': str(source), 'source_tree_sha256': digest(locked_tree(source)),
                'entrypoint': 'backend.py', 'weights_dir': str(weights),
                'weights_tree_sha256': digest(locked_tree(weights)), 'objective': 'diffusion',
                'approved': True, 'license_record': 'test-owned fixture only'}
    py_compile.compile(str(entry), doraise=True)
    with pytest.raises(ValueError, match='bytecode'):
        load_approved_backend(manifest, device='cpu')


@pytest.mark.parametrize('step,position', [(True, 4), (-1, -4), (11, 44), (2., 8), (2, True), (2, 7), (2, 9)])
def test_resume_rejects_invalid_boundary(step, position):
    with pytest.raises(ValueError, match='Resume'):
        validate_resume_cursor({'step': step, 'position': position}, {'updates': 10, 'batch_size': 2, 'accumulation': 2})


def test_resume_accepts_exact_optimizer_boundary():
    assert validate_resume_cursor({'step': 7, 'position': 28}, {'updates': 10, 'batch_size': 2, 'accumulation': 2}) == (7, 28)


def test_checkpoint_bytes_must_match_independent_sidecar(tmp_path):
    checkpoint = tmp_path / 'update-00000001.pt'; checkpoint.write_bytes(b'test checkpoint bytes')
    checkpoint.with_suffix('.json').write_text(json.dumps({'contract_sha256': 'a'*64, 'checkpoint_sha256': file_hash(checkpoint)}))
    validate_checkpoint_receipt(checkpoint, 'a'*64)
    checkpoint.write_bytes(b'different retained checkpoint')
    with pytest.raises(ValueError, match='hash'):
        validate_checkpoint_receipt(checkpoint, 'a'*64)


def test_dev_validation_requires_audited_kitsap_bytes(tmp_path):
    path = tmp_path / 'dev-fixture.png'; path.write_bytes(b'identity-control only')
    row = {'world_id': 'dev-fixture', 'city': 'kitsap', 'role': 'dev', 'path': path.name,
           'source_sha256': 'a'*64, 'world_file_sha256': file_hash(path), 'world_pixel_sha256': 'b'*64}
    dataset = SimpleNamespace(root=tmp_path, worlds={'dev-fixture': row}, source_audit={'worlds': {'dev-fixture': row}})
    validate_dev_index([row], dataset)
    with pytest.raises(ValueError, match='identity'):
        validate_dev_index([{**row, 'source_sha256': 'c'*64}], dataset)
    dataset.source_audit['worlds']['dev-fixture'] = {**row, 'city': 'bloomington', 'role': 'metric_validation'}
    with pytest.raises(ValueError, match='Kitsap'):
        validate_dev_index([row], dataset)
