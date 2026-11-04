import json
from pathlib import Path
import pytest
from rswm_inria import acceptance


def test_missing_real_inputs_do_not_become_skipped_pass():
    for profile in ('smoke12-diagnostic','real-model-pilot','metric-validation','core-formal'):
        r=acceptance.actual_profile(profile,None,'unused',{'code_commit':'a'*40})
        assert r['status']=='BLOCKED' and r['publication_acceptance']=='NOT_READY'


def test_full_build_cli_preserves_blocked_gates(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(acceptance,'build_identity',lambda c:{'code_commit':'a'*40})
    monkeypatch.setattr(acceptance,'cpu_synthetic',lambda *a,**kw:{'status':'PASS_SOFTWARE_DIAGNOSTIC'})
    from rswm_inria import isolation_backend
    monkeypatch.setattr(isolation_backend,'permission_probe',lambda:{'status':'BLOCKED','protocol_security_verified':False})
    assert acceptance.main(['--profile','full-build','--out',str(tmp_path/'out')])==2
    report=json.loads(capsys.readouterr().out)
    assert report['status']=='INCOMPLETE' and len(report['gates'])==5
    assert all(g['status']=='BLOCKED' for g in report['gates'])
    assert report['software']['status']=='PASS_SOFTWARE_DIAGNOSTIC'


def test_ambiguous_manifest_json_cannot_be_used_for_acceptance(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(acceptance,'build_identity',lambda c:{'code_commit':'a'*40})
    p=tmp_path/'inputs.json';p.write_text('{"plan":"first","plan":"second"}')
    assert acceptance.main(['--profile','real-model-pilot','--inputs',str(p),'--out',str(tmp_path/'out')])==2
    assert json.loads(capsys.readouterr().out)['status']=='BLOCKED_OR_INVALID'


def test_build_provenance_rejects_changed_installed_source(monkeypatch,tmp_path):
    monkeypatch.setattr(acceptance,'__file__',str(tmp_path/'acceptance.py'))
    (tmp_path/'_build_info.json').write_text(json.dumps({'source_commit':'a'*40,'source_code_sha256':'wrong'}))
    with pytest.raises(ValueError,match='source differs'):acceptance.build_identity()
