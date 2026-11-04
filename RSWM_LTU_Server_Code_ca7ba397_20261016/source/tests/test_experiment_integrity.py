"""Gate and persistent-budget controls; no real pilot or instrument claims."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from rswm_inria import experiments as e
from rswm_inria.geometry import digest


def sealed(value,key):return {**value,key:digest(value)}


def test_thin_pilot_and_metrics_never_become_real_evidence():
    thin=sealed({'purpose':'real','acceptance_status':'ACCEPTED_REAL_PILOT',
                 'model_manifest_sha256':['a'*64]},'receipt_sha256')
    with pytest.raises(ValueError,match='actual real'):
        e.validate_pilot_receipt(thin,{'expected':[{'source_manifest_sha256':'b'*64}]})
    metrics=[sealed({'real_data_validation':'PASS','role':'metric_validation','instrument':name},'validation_sha256')
             for name in ('motion','quality')]
    with pytest.raises(ValueError,match='actual motion/quality'):
        e._instrument_receipts(metrics,{},'b'*64,{'motion':'c'*64,'quality':'d'*64})


def test_instrument_configuration_is_preregistered_before_receipt_acceptance():
    with pytest.raises(ValueError,match='preregistered'):
        e._instrument_receipts([],{},'b'*64,None)
    candidate=sealed({'schema':'rswm-feature-control-validation-v1','release_validated':False},'validation_sha256')
    with pytest.raises(ValueError,match='candidate controls'):
        e._instrument_receipts([candidate],{},'b'*64,{'motion':'c'*64,'quality':'d'*64})


def test_materialized_pixels_alone_cannot_replace_real_smoke12_diagnostics():
    with pytest.raises(ValueError,match='real smoke12 diagnostic evidence'):
        e.validate_smoke12_evidence(None,'a'*64)


def test_actual_motion_release_validator_rejects_diagnostic_receipt():
    from rswm_inria.motion_protocol import motion_config, VALIDATION_SCHEMA
    receipt=sealed({'schema':VALIDATION_SCHEMA,'status':'DIAGNOSTIC_ONLY','config':motion_config(),
                    'source_manifest_sha256':'a'*64},'validation_sha256')
    with pytest.raises(ValueError,match='DIAGNOSTIC_ONLY'):
        e._instrument_receipts([receipt],{},'a'*64,{'motion':receipt['config']['config_sha256'],'quality':'b'*64})


def current_materialized_schema_control():
    """Full schema binding fixture, not an actual materialization/real receipt."""
    worlds={};tasks={};sources={};artifacts={};rows=[]
    for n in range(12):
        wid=f'schema-control-{n}'
        role='train' if n<9 else 'dev'
        city=('austin','chicago','vienna')[n//3] if n<9 else 'kitsap'
        source=f'data/train/images/{city}{n+1}.tif'
        world={'world_id':wid,'source_path':source,'source_sha256':digest([source,'synthetic hash']),
               'world_file_sha256':digest([wid,'synthetic png hash']),
               'world_pixel_sha256':digest([wid,'synthetic pixel hash']),'role':role,'city':city}
        worlds[wid]=world;sources[source]=world['source_sha256']
        artifacts['private/worlds/'+wid+'.png']=world['world_file_sha256']
        for origin in range(4):
            for family in ('F2','F3','F5','F7'):
                tid=f'{wid}-{origin}-{family}'
                task={'task_sha256':digest([tid,'synthetic task hash']),'world_id':wid,'role':role,
                      'origin_index':origin,'family':family,'protocol':'online16' if family in ('F5','F7') else 'offline'}
                tasks[tid]=task
                if n in (9,10) and origin==0:
                    rows.append({**task,'task_id':tid,'city':city,'source_manifest_sha256':'a'*64,
                                 'world_file_sha256':world['world_file_sha256'],'world_pixel_sha256':world['world_pixel_sha256']})
    report={'schema':'rswm-materialized-audit-v1','status':'MATERIALIZED_PIXELS_VALIDATED',
            'input_kind':'source_identity_checked_geography_unknown','acceptance_status':'ACCEPTED_IN_STATED_SCOPE',
            'profile':'smoke12','manifest_sha256':'a'*64,'inventory_sha256':'b'*64,'source_code_sha256':'c'*64,
            'counts':{'sources':12,'worlds':12,'initial_images':48,'task_packages':192,'public_sessions':192,
                      'training_clips':576,'training_view_events':19008,'planned_unique_outputs_per_model_seed':16896},
            'worlds':worlds,'tasks':tasks,'source_sha256':sources,'artifact_sha256':artifacts}
    return report,{'expected':rows,'source_code_sha256':'c'*64}


def test_actual_materialized_receipt_schema_binds_worlds_tasks_and_roles():
    from rswm_inria.materialized import validate_audit_receipt
    report,plan=current_materialized_schema_control()
    bound=validate_audit_receipt(report,plan)
    assert bound['status']=='MATERIALIZED_RECEIPT_BOUND' and bound['matched_planned_tasks']==8
    bad=copy.deepcopy(report);bad['worlds'][plan['expected'][0]['world_id']]['role']='test_unseen_city'
    with pytest.raises(ValueError,match='world/city/role'):
        validate_audit_receipt(bad,plan)
    with pytest.raises(ValueError,match='profile'):
        validate_audit_receipt(report,plan,required_profile='core120')
    # No original TIFF/PNG was read: this proves schema identity, not real data.


def retained_smoke12_schema_control(tmp_path,monkeypatch):
    """Full schema fixture with replay/plan-validation mocks, never real evidence."""
    from rswm_inria.smoke12 import DIAGNOSTIC_MODELS
    audit,_=current_materialized_schema_control()
    code=e.code_fingerprint();audit['source_code_sha256']=code
    rows=[]
    for tid,task in audit['tasks'].items():
        world=audit['worlds'][task['world_id']]
        for model in DIAGNOSTIC_MODELS:
            rows.append({**task,'task_id':tid,'model_id':model['model_id'],'seed':0,
                         'city':world['city'],'source_manifest_sha256':'a'*64,
                         'world_file_sha256':world['world_file_sha256'],
                         'world_pixel_sha256':world['world_pixel_sha256']})
    plan={'expected':rows,'source_code_sha256':code,
          'models':{m['model_id']:{'config':{**m,'device':'cpu'}} for m in DIAGNOSTIC_MODELS}}
    plan=sealed(plan,'plan_sha256')
    (tmp_path/'suite').mkdir()
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    (tmp_path/'input-audit.json').write_text(json.dumps(audit))
    artifact=tmp_path/'retained-fixture.txt'
    artifact.write_text('Synthetic schema control; no actual smoke12 outputs.\n')
    report={'schema':'rswm-smoke12-acceptance-v1','status':'REAL_DATA_VALIDATED',
            'full_smoke12_matrix':True,'real_data_validated':True,
            'input_kind':'source_identity_checked_geography_unknown','models':DIAGNOSTIC_MODELS,
            'planned_runs_total':576,'planned_unique_outputs_total':50688,
            'failed_runs':[],'analysis_status':'ACCEPTED','plan_sha256':plan['plan_sha256'],
            'source_code_sha256':code,'artifact_sha256':{'retained-fixture.txt':e.file_hash(artifact)}}
    (tmp_path/'acceptance.json').write_text(json.dumps(report))
    records=[{'replay_verified':True,'reports':{'metrics':{},'valid_outputs':50688}}]
    monkeypatch.setattr(e,'validate_plan',lambda *args:None)
    monkeypatch.setattr(e,'replay_suite',lambda *args:records)
    monkeypatch.setattr(e,'integrity_summary',lambda *args:{'accepted':True,'verified_submissions':576})
    return report,records


@pytest.mark.parametrize('failed_runs',[[],0],ids=['canonical-empty-list','legacy-integer-zero'])
def test_smoke12_gate_accepts_success_failure_representation(tmp_path,monkeypatch,failed_runs):
    report,_=retained_smoke12_schema_control(tmp_path,monkeypatch)
    report['failed_runs']=failed_runs
    (tmp_path/'acceptance.json').write_text(json.dumps(report))
    bound=e.validate_smoke12_evidence(str(tmp_path),'a'*64)
    assert bound['planned_runs']==576 and bound['verified_outputs']==50688


@pytest.mark.parametrize('failed_runs',[False,True,0.0,-1,1,['failed-run'],{},'',None])
def test_smoke12_gate_rejects_failures_and_wrong_failure_types(tmp_path,monkeypatch,failed_runs):
    report,_=retained_smoke12_schema_control(tmp_path,monkeypatch)
    report['failed_runs']=failed_runs
    (tmp_path/'acceptance.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='matrix was not accepted'):
        e.validate_smoke12_evidence(str(tmp_path),'a'*64)


def test_smoke12_gate_requires_failure_field(tmp_path,monkeypatch):
    report,_=retained_smoke12_schema_control(tmp_path,monkeypatch)
    del report['failed_runs']
    (tmp_path/'acceptance.json').write_text(json.dumps(report))
    with pytest.raises(ValueError,match='matrix was not accepted'):
        e.validate_smoke12_evidence(str(tmp_path),'a'*64)


@pytest.mark.parametrize('fault',['integrity','submission-count','output-count','unverified','artifact'])
def test_smoke12_success_list_preserves_independent_verification_vetoes(tmp_path,monkeypatch,fault):
    _,records=retained_smoke12_schema_control(tmp_path,monkeypatch)
    if fault=='integrity':
        monkeypatch.setattr(e,'integrity_summary',lambda *args:{'accepted':False,'verified_submissions':576})
    elif fault=='submission-count':
        monkeypatch.setattr(e,'integrity_summary',lambda *args:{'accepted':True,'verified_submissions':575})
    elif fault=='output-count':records[0]['reports']['valid_outputs']-=1
    elif fault=='unverified':records[0]['replay_verified']=False
    else:(tmp_path/'retained-fixture.txt').write_text('tampered\n')
    with pytest.raises(ValueError,match='artifact changed' if fault=='artifact' else 'independent verification'):
        e.validate_smoke12_evidence(str(tmp_path),'a'*64)


def retained_pilot_schema_control(tmp_path,monkeypatch):
    """Physical manifest fixture + explicit replay mock; never a real pilot."""
    (tmp_path/'suite').mkdir();(tmp_path/'analysis').mkdir()
    model='d'*64;source='a'*64;code=e.code_fingerprint()
    plan={'plan_sha256':'b'*64,'expected':[{'source_manifest_sha256':source,'model_manifest_sha256':model}]}
    experiment={'experiment_id':'c'*64,'kind':'pilot','purpose':'real',
                'budget':{'max_wall_seconds':100.,'max_output_bytes':1_000_000}}
    matrix={'expected_runs':8,'planned_unique_outputs':704,'worlds':2,'origins':1,'model_seed_groups':1}
    integrity={'accepted':True,'verified_submissions':8,'diagnostic_mock_only':True}
    (tmp_path/'suite/plan.json').write_text(json.dumps(plan))
    (tmp_path/'experiment.json').write_text(json.dumps(experiment))
    (tmp_path/'analysis/results.csv').write_text('diagnostic schema fixture; no actual pilot\n')
    from rswm_inria.geometry import file_hash
    analysis={'plan_sha256':plan['plan_sha256'],'source_code_sha256':code,'acceptance_status':'ACCEPTED',
              'files':{'results.csv':file_hash(tmp_path/'analysis/results.csv')}}
    (tmp_path/'analysis/provenance.json').write_text(json.dumps(analysis))
    receipt=sealed({'schema':'rswm-experiment-receipt-v1','kind':'pilot','purpose':'real',
        'acceptance_status':'ACCEPTED_REAL_PILOT','source_manifest_sha256':source,'source_code_sha256':code,
        'experiment_id':experiment['experiment_id'],'plan_sha256':plan['plan_sha256'],'model_manifest_sha256':[model],
        'matrix':matrix,'expected_runs':8,'verified_outputs':704,'integrity':integrity,'evidence_root':str(tmp_path),
        'receipt_path':'receipt.json','analysis_provenance_path':'analysis/provenance.json',
        'analysis_provenance_sha256':file_hash(tmp_path/'analysis/provenance.json'),
        'budget_usage':{'exhausted':False,'elapsed_total_seconds':1.,'output_bytes':1000,'reserved_publication_bytes':4096}},'receipt_sha256')
    (tmp_path/'receipt.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(e,'validate_experiment',lambda *args:None)
    monkeypatch.setattr(e,'_matrix',lambda *args:matrix)
    monkeypatch.setattr(e,'replay_suite',lambda *args:[{'resources':{'accounting_calls':{'valid_outputs':704}}}])
    monkeypatch.setattr(e,'integrity_summary',lambda *args:integrity)
    return receipt,plan


def test_complete_pilot_schema_requires_published_receipt_and_intact_analysis(tmp_path,monkeypatch):
    receipt,plan=retained_pilot_schema_control(tmp_path,monkeypatch)
    assert e.validate_pilot_receipt(receipt,plan)=={'d'*64}
    unpublished=copy.deepcopy(receipt);unpublished['expected_runs']=1
    unpublished=sealed({k:v for k,v in unpublished.items() if k!='receipt_sha256'},'receipt_sha256')
    with pytest.raises(ValueError,match='published receipt'):
        e.validate_pilot_receipt(unpublished,plan)
    (tmp_path/'analysis/results.csv').write_text('tampered\n')
    with pytest.raises(ValueError,match='analysis artifact changed'):
        e.validate_pilot_receipt(receipt,plan)


def test_complete_pilot_schema_never_trusts_claim_when_replay_or_resources_fail(tmp_path,monkeypatch):
    receipt,plan=retained_pilot_schema_control(tmp_path,monkeypatch)
    monkeypatch.setattr(e,'integrity_summary',lambda *args:{'accepted':False,'verified_submissions':0})
    with pytest.raises(ValueError,match='independent verification'):
        e.validate_pilot_receipt(receipt,plan)


class Clock:
    def __init__(self):self.now=0.
    def monotonic(self):return self.now
    def time(self):return 1_000.+self.now


def budget_contract(max_seconds=100.,max_bytes=10_000_000):
    return {'experiment_id':'c'*64,'budget':{'max_wall_seconds':max_seconds,'max_output_bytes':max_bytes}}


def test_total_elapsed_budget_survives_resume_and_terminal_completion(tmp_path,monkeypatch):
    clock=Clock();monkeypatch.setattr(e,'time',clock)
    experiment=budget_contract()
    first=e._ExecutionBudget(tmp_path,experiment,resume=False)
    clock.now=4.;usage=first.finish()
    assert usage['elapsed_total_seconds']==4.
    clock.now=20.;second=e._ExecutionBudget(tmp_path,experiment,resume=True)
    assert second.invocation==1
    clock.now=23.;assert second.finish()['elapsed_total_seconds']==7.
    with pytest.raises(FileExistsError):e._ExecutionBudget(tmp_path,experiment,resume=False)


def test_hard_exit_activity_charged_conservatively_on_resume(tmp_path,monkeypatch):
    clock=Clock();monkeypatch.setattr(e,'time',clock)
    first=e._ExecutionBudget(tmp_path,budget_contract(max_seconds=10.),resume=False)
    clock.now=3.;first.observe()  # persisted observation, no clean finish
    clock.now=12.;second=e._ExecutionBudget(tmp_path,budget_contract(max_seconds=10.),resume=True)
    usage=second.observe()
    assert usage['elapsed_total_seconds']==12. and usage['exhausted']


def test_terminal_byte_overshoot_and_bad_budget_identity_fail_closed(tmp_path,monkeypatch):
    clock=Clock();monkeypatch.setattr(e,'time',clock)
    budget=e._ExecutionBudget(tmp_path,budget_contract(max_bytes=2_000),resume=False)
    (tmp_path/'model-output.bin').write_bytes(b'x'*3_000)
    assert budget.finish()['exhausted']
    changed=budget_contract();changed['experiment_id']='d'*64
    with pytest.raises(ValueError,match='identity'):e._ExecutionBudget(tmp_path,changed,resume=True)


def execution_control(monkeypatch,clock,*,last_run_elapsed=2.):
    """Synthetic coordinator stub; genuine suite/replay are covered elsewhere."""
    monkeypatch.setattr(e,'time',clock)
    monkeypatch.setattr(e,'validate_experiment',lambda experiment,plan:None)
    reports={'main':{'valid_outputs':32,'metrics':{}}}
    records=[{'replay_verified':True,'reports':reports,
              'resources':{'accounting_calls':{'valid_outputs':32}},'resource_status':'COMPLETE','status':'COMPLETED'}]
    integrity={'accepted':True,'verified_submissions':1,'schema':'DIAGNOSTIC_COORDINATOR_STUB'}
    def execute(plan,out,*,factory,**kwargs):
        Path(out).mkdir(exist_ok=True)
        factory({'model_id':'copy','adapter':'copy'},None,None)
        clock.now+=last_run_elapsed
    def export(report,out):
        Path(out).mkdir(parents=True)
        (Path(out)/'provenance.json').write_text(json.dumps({'diagnostic_coordinator_stub':True}))
    monkeypatch.setattr(e,'execute_plan',execute)
    monkeypatch.setattr(e,'replay_suite',lambda *args:records)
    monkeypatch.setattr(e,'integrity_summary',lambda *args:integrity)
    monkeypatch.setattr(e,'aggregate_suite',lambda *args:{})
    monkeypatch.setattr(e,'export_tables',export)
    return {'schema':'DIAGNOSTIC_CONTROL','purpose':'diagnostic','kind':'pilot','experiment_id':'e'*64,
            'matrix':{'expected_runs':1,'planned_unique_outputs':32},
            'budget':{'max_wall_seconds':100.,'max_output_bytes':1_000_000}},


def control_plan():
    return {'plan_sha256':'f'*64,'expected':[{'source_manifest_sha256':'a'*64,'model_manifest_sha256':'b'*64}]}


def test_last_successful_run_cannot_upgrade_overspent_budget_to_accepted(tmp_path,monkeypatch):
    clock=Clock();experiment=execution_control(monkeypatch,clock,last_run_elapsed=20.)[0]
    experiment['budget']['max_wall_seconds']=10.
    receipt=e.execute_experiment(experiment,control_plan(),tmp_path)
    assert receipt['verified_outputs']==32 and receipt['integrity']['accepted']
    assert receipt['acceptance_status']=='BUDGET_EXHAUSTED' and receipt['budget_usage']['exhausted']


def test_resume_publishes_new_analysis_and_receipt_preserving_original_evidence(tmp_path,monkeypatch):
    clock=Clock();experiment=execution_control(monkeypatch,clock)[0]
    first=e.execute_experiment(experiment,control_plan(),tmp_path)
    original=(tmp_path/'receipt.json').read_bytes()
    second=e.execute_experiment(experiment,control_plan(),tmp_path,resume=True)
    assert (tmp_path/'receipt.json').read_bytes()==original
    assert first['acceptance_status']==second['acceptance_status']=='DIAGNOSTIC_ONLY'
    assert second['budget_usage']['elapsed_total_seconds']==4.
    assert second['receipt_path']=='receipts/invocation-000001.json'
    assert second['analysis_provenance_path']=='analyses/invocation-000001/provenance.json'
    assert json.loads((tmp_path/'receipt-head.json').read_text())['receipt_sha256']==second['receipt_sha256']


@pytest.mark.parametrize('status,code',[('INCOMPLETE',1),('BUDGET_EXHAUSTED',1),
    ('EXECUTED_FORMAL_PENDING_INSTRUMENT_ANALYSIS',1),('DIAGNOSTIC_ONLY',0)])
def test_cli_status_exit_codes_are_not_silently_successful(monkeypatch,status,code):
    monkeypatch.setattr(e,'_read',lambda path:{})
    monkeypatch.setattr(e,'execute_experiment',lambda *args,**kwargs:{'acceptance_status':status})
    assert e.main(['run','--plan','unused','--experiment','unused','--out','unused'])==code
