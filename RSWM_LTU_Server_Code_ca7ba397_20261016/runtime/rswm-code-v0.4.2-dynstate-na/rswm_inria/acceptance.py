"""Small installed software acceptance and explicit, non-skipping real gates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from PIL import Image
from .geometry import digest,file_hash
from .tasks import packages
from .suite import build_plan,execute_plan,integrity_summary,replay_suite,code_fingerprint
from .aggregate import aggregate_suite
from .export_tables import export_tables
from .suite import _json

PROFILES=('cpu-synthetic','smoke12-diagnostic','real-model-pilot','metric-validation','core-formal','isolated-release','full-build')


def _save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if p.exists() and p.read_text()!=text:raise FileExistsError(p)
    p.write_text(text)


def build_identity(commit=None):
    info=Path(__file__).with_name('_build_info.json')
    saved=json.loads(info.read_text()) if info.exists() else {}
    if saved and saved.get('source_code_sha256')!=code_fingerprint():raise ValueError('Installed package source differs from build provenance')
    if commit is None:commit=saved.get('source_commit')
    if commit is None:
        try:commit=subprocess.check_output(['git','rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.CalledProcessError):raise ValueError('Source commit unavailable: pass --code-commit or install a provenance-bearing wheel')
    if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('Full code commit required')
    return {'code_commit':commit,'source_code_sha256':code_fingerprint(),'build_info':saved}


def cpu_synthetic(out,identity,*,families=('F2','F7')):
    root=Path(out)
    if root.exists() and any(root.iterdir()):raise FileExistsError('Acceptance output must be empty')
    inputs=root/'inputs';inputs.mkdir(parents=True)
    world=np.full((2048,2048,3),73,np.uint8);Image.fromarray(world).save(inputs/'world.png')
    source={'schema':'rswm-cpu-acceptance-source-v1','input_type':'synthetic','world_pixel_sha256':hashlib.sha256(world.tobytes()).hexdigest()}
    rows=[]
    for task,labels in packages('installed-cpu-control','diagnostic_only',0):
        if task['family'] not in families:continue
        task['private_label_files']={}
        for stream,arrays in labels.items():
            path=inputs/(task['task_id']+'-'+stream+'.npz');np.savez_compressed(path,**arrays)
            task['private_label_files'][stream]={'path':path.name,'sha256':file_hash(path)}
        p=inputs/(task['task_id']+'.json');_save(p,task)
        rows.append({'task_path':str(p),'world_path':str(inputs/'world.png'),'city':'kitsap'})
    if not rows:raise ValueError('No acceptance tasks')
    plan=build_plan(rows,[{'model_id':'oracle-control','adapter':'oracle','device':'cpu'}],[0],digest(source),identity['code_commit'])
    _save(root/'plan.json',plan);execute_plan(plan,root/'suite')
    records=replay_suite(plan,root/'suite');integrity=integrity_summary(plan,root/'suite',records)
    report=aggregate_suite(plan,root/'suite',bootstrap=5,seed=7);export_tables(report,root/'analysis')
    outputs=sum(r['resources']['accounting_calls']['valid_outputs'] for r in records)
    assert integrity['accepted'] and report['acceptance_status']=='ACCEPTED'
    assert all(c['mae'] in (0,None) for c in report['cells'])
    expected=sum({'F2':32,'F3':64,'F5':96,'F7':160}[r['family']] for r in plan['expected'])
    assert outputs==expected
    receipt={'schema':'rswm-build-acceptance-v1','profile':'cpu-synthetic','status':'PASS_SOFTWARE_DIAGNOSTIC',
             'identity':identity,'input_type':'synthetic','models':['oracle-control'],
             'expected_runs':len(rows),'completed_runs':len(records),'planned_outputs':expected,'verified_outputs':outputs,
             'integrity':integrity,'real_data_validation':'NOT_RUN','real_model_validation':'NOT_RUN','real_gpu_validation':'NOT_RUN',
             'publication_acceptance':'NOT_READY','analysis_provenance_sha256':file_hash(root/'analysis/provenance.json')}
    receipt['receipt_sha256']=digest(receipt);_save(root/'acceptance.json',receipt)
    return receipt


def actual_profile(profile,inputs,out,identity):
    """Execute available gates; absent inputs never become a skipped total PASS."""
    if profile=='isolated-release':
        from .isolation_backend import permission_probe
        gate=permission_probe()
        return {'profile':profile,'status':'BLOCKED' if gate['status']!='PASS' else 'OS_PERMISSION_GATE_ONLY',
                'publication_acceptance':'NOT_READY','gate':gate,'identity':identity}
    if not inputs:
        return {'profile':profile,'status':'BLOCKED','reason':'Required real input/weight/frozen protocol manifest not supplied',
                'real_data_validation':'NOT_RUN','real_model_validation':'NOT_RUN','real_gpu_validation':'NOT_RUN','publication_acceptance':'NOT_READY','identity':identity}
    if profile in {'real-model-pilot','core-formal'}:
        from .experiments import execute_experiment
        plan=_json(inputs['plan']);experiment=_json(inputs['experiment'])
        if experiment['purpose']!='real' or experiment['kind']!=('pilot' if profile=='real-model-pilot' else 'formal'):
            raise ValueError('Actual real profile/experiment identity mismatch')
        receipt=execute_experiment(experiment,plan,out)
        return {'profile':profile,'status':receipt['acceptance_status'],'receipt':receipt,'identity':identity,'publication_acceptance':'NOT_READY'}
    if profile=='smoke12-diagnostic':
        from .smoke12 import accept_smoke12
        receipt=accept_smoke12(**inputs,out=out,code_commit=identity['code_commit'])
        return {'profile':profile,'status':receipt.get('acceptance_status',receipt.get('status','INCOMPLETE')),
                'receipt':receipt,'identity':identity,'publication_acceptance':'NOT_READY'}
    if profile=='metric-validation':
        from .motion_protocol import validate_motion, validate_release
        motion=inputs.get('motion')
        if not motion or not inputs.get('quality'):
            raise ValueError('Both frozen instrument inputs required')
        manifest=_json(motion['manifest']);calibration=_json(motion['calibration'])
        motion_receipt=validate_motion(manifest,motion['root'],calibration,Path(out)/'motion',motion['exposure_ledger'])
        validate_release(motion_receipt,motion_receipt['config'])
        # Run the actual held-out feature release, including its retained
        # calibration, reviewed criteria, source audit and exposure ledger.
        q=inputs['quality'];target=Path(out)/'quality-validation.json'
        args=['validate']
        for field in ('config','acceptance_config','reference_mosaics','patch_plan','suite_plan',
                      'source_audit','cache','calibration_receipt','exposure_ledger'):
            args.extend(['--'+field.replace('_','-'),q[field]])
        args.extend(['--out',str(target)])
        completed=subprocess.run([sys.executable,'-c',
            'from rswm_inria.features import release_main; raise SystemExit(release_main())',*args],capture_output=True,text=True)
        command={'name':'quality-heldout-release','argv':args,'returncode':completed.returncode,
                 'stdout':completed.stdout,'stderr':completed.stderr}
        if completed.returncode:
            return {'profile':profile,'status':'INCOMPLETE','motion':motion_receipt,'commands':[command],'publication_acceptance':'NOT_READY'}
        from .features import validate_feature_release
        quality_receipt=_json(target)
        feature_id=digest(_json(q['config']));source_id=motion_receipt['source_manifest_sha256']
        validate_feature_release(quality_receipt,feature_id,source_id)
        return {'profile':profile,'status':'VALIDATED_REAL_INSTRUMENTS','motion':motion_receipt,
                'quality':quality_receipt,'commands':[command],'identity':identity,'publication_acceptance':'NOT_READY'}
    raise ValueError('Unsupported acceptance profile')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=PROFILES,default='cpu-synthetic');p.add_argument('--out',required=True)
    p.add_argument('--inputs');p.add_argument('--code-commit');p.add_argument('--families',nargs='+',choices=('F2','F3','F5','F7'),default=['F2','F7'])
    a=p.parse_args(argv)
    try:
        identity=build_identity(a.code_commit);inputs=_json(a.inputs) if a.inputs else None
        if a.profile=='cpu-synthetic':report=cpu_synthetic(a.out,identity,families=tuple(a.families))
        elif a.profile=='full-build':
            software=cpu_synthetic(Path(a.out)/'cpu-synthetic',identity,families=tuple(a.families))
            gates=[]
            for profile in PROFILES[1:-1]:
                try:gate=actual_profile(profile,(inputs or {}).get(profile),Path(a.out)/profile,identity)
                except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
                    gate={'profile':profile,'status':'BLOCKED_OR_INVALID','error':str(exc),'publication_acceptance':'NOT_READY'}
                gates.append(gate)
            report={'schema':'rswm-full-build-acceptance-v1','profile':a.profile,'status':'INCOMPLETE','software':software,'gates':gates,'publication_acceptance':'NOT_READY'}
            _save(Path(a.out)/'acceptance.json',report)
        else:
            report=actual_profile(a.profile,inputs,a.out,identity);_save(Path(a.out)/'acceptance.json',report)
        print(json.dumps(report,sort_keys=True,allow_nan=False))
        if report['status'] in {'PASS_SOFTWARE_DIAGNOSTIC','ACCEPTED_REAL_PILOT','REAL_DATA_VALIDATED','VALIDATED_REAL_INSTRUMENTS'}:return 0
        if a.profile=='full-build' and any(g['status'].startswith('BLOCKED') for g in report['gates']):return 2
        return 1 if report['status'] in {'BUDGET_EXHAUSTED','INCOMPLETE','EXECUTED_FORMAL_PENDING_INSTRUMENT_ANALYSIS'} else 2
    except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        report={'profile':a.profile,'status':'BLOCKED_OR_INVALID','error':str(exc),'publication_acceptance':'NOT_READY'}
        print(json.dumps(report));return 2


if __name__=='__main__':raise SystemExit(main())
