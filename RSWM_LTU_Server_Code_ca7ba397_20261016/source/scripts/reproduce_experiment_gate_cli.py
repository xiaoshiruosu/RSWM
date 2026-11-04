"""Actual diagnostic experiment CLI, immutable resume and budget failure.

Synthetic images use the known dev city key as a fixture placeholder only. No
real source, trained weights, instrument release or formal test is claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from rswm_inria.geometry import digest,file_hash
from rswm_inria.tasks import packages
from rswm_inria.suite import code_fingerprint


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--evidence-dir',type=Path,required=True)
    a=p.parse_args();a.work_dir.mkdir(parents=True,exist_ok=False);a.evidence_dir.mkdir(parents=True,exist_ok=False)
    root=a.work_dir.resolve();evidence=a.evidence_dir.resolve();inputs=root/'inputs';inputs.mkdir()
    source={'schema':'rswm-experiment-cli-synthetic-source-v1','real_inria':False,'world_values':[71,89]}
    tasks=[];worlds=[]
    for i,value in enumerate(source['world_values']):
        wid=f'diagnostic-pilot-control-{i}';pixels=np.full((2048,2048,3),value,np.uint8)
        worldpath=inputs/(wid+'.png');Image.fromarray(pixels).save(worldpath)
        worlds.append({'world_id':wid,'city':'kitsap','role':'dev','path':worldpath.name,
                       'world_file_sha256':file_hash(worldpath),'world_pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()})
        for task,labels in packages(wid,'dev',0):
            task['private_label_files']={}
            for name,arrays in labels.items():
                path=inputs/(task['task_id']+'-'+name+'.npz');np.savez_compressed(path,**arrays)
                task['private_label_files'][name]={'path':path.name,'sha256':file_hash(path)}
            path=inputs/(task['task_id']+'.json');save(path,task)
            tasks.append({'path':path.name,'city':'kitsap','role':'dev'})
    save(inputs/'index.json',{'tasks':tasks,'worlds':worlds,'source_manifest_sha256':digest(source)})
    save(inputs/'models.json',[{'model_id':'copy-control','adapter':'copy','device':'cpu'}])
    save(inputs/'normal-gates.json',{'budget':{'max_wall_seconds':1000.,'max_output_bytes':1_000_000_000}})
    save(inputs/'budget-gates.json',{'budget':{'max_wall_seconds':1000.,'max_output_bytes':1}})
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();fingerprint=code_fingerprint();commands=[]
    def command(name,argv,expected=0):
        cmd=[sys.executable,*argv];result=subprocess.run(cmd,capture_output=True,text=True)
        (evidence/(name+'.stdout.log')).write_text(result.stdout)
        (evidence/(name+'.stderr.log')).write_text(result.stderr)
        commands.append({'name':name,'command':cmd,'exit_code':result.returncode,'expected_exit_code':expected})
        (evidence/'commands.json').write_text(json.dumps(commands,indent=2,sort_keys=True)+'\n')
        if result.returncode!=expected:raise RuntimeError(name+' actual CLI return code mismatch')
    command('plan',['-m','rswm_inria.suite','plan','--task-index',str(inputs/'index.json'),
        '--world-dir',str(inputs),'--models',str(inputs/'models.json'),'--seeds','0','--code-commit',commit,'--out',str(root/'plan.json')])
    for name in ('normal','budget'):
        command('freeze-'+name,['-m','rswm_inria.experiments','freeze','--plan',str(root/'plan.json'),
            '--kind','pilot','--purpose','diagnostic','--inputs',str(inputs/(name+'-gates.json')),'--out',str(root/(name+'-experiment.json'))])
    command('run',['-m','rswm_inria.experiments','run','--plan',str(root/'plan.json'),
        '--experiment',str(root/'normal-experiment.json'),'--out',str(root/'normal')])
    first=(root/'normal/receipt.json').read_bytes();first_sha=hashlib.sha256(first).hexdigest()
    command('resume',['-m','rswm_inria.experiments','run','--plan',str(root/'plan.json'),
        '--experiment',str(root/'normal-experiment.json'),'--out',str(root/'normal'),'--resume'])
    assert (root/'normal/receipt.json').read_bytes()==first
    command('no-implicit-resume',['-m','rswm_inria.experiments','run','--plan',str(root/'plan.json'),
        '--experiment',str(root/'normal-experiment.json'),'--out',str(root/'normal')],2)
    command('budget-stop',['-m','rswm_inria.experiments','run','--plan',str(root/'plan.json'),
        '--experiment',str(root/'budget-experiment.json'),'--out',str(root/'budget')],1)
    normal=json.loads(first);resume=json.loads((root/'normal/receipts/invocation-000001.json').read_text())
    budget=json.loads((root/'budget/receipt.json').read_text())
    assert normal['acceptance_status']==resume['acceptance_status']=='DIAGNOSTIC_ONLY'
    assert normal['expected_runs']==8 and normal['verified_outputs']==704
    assert resume['budget_usage']['elapsed_total_seconds']>=normal['budget_usage']['elapsed_total_seconds']
    assert budget['acceptance_status']=='BUDGET_EXHAUSTED' and budget['verified_outputs']==0 and budget['expected_runs']==8
    if code_fingerprint()!=fingerprint:raise RuntimeError('Source changed; retain intermediate evidence and rerun after freeze')
    summary={'schema':'rswm-experiment-cli-engineering-evidence-v1','source_code_sha256':fingerprint,'base_commit':commit,
        'data_kind':'synthetic constant RGB; kitsap administrative fixture only','real_data_validation':'NOT_RUN',
        'real_model_validation':'NOT_RUN','real_gpu_validation':'NOT_RUN','publication_acceptance':'NOT_READY',
        'expected_runs':8,'planned_outputs':704,'normal_verified_outputs':normal['verified_outputs'],
        'normal_status':normal['acceptance_status'],'resume_status':resume['acceptance_status'],
        'budget_status':budget['acceptance_status'],'budget_verified_outputs':budget['verified_outputs'],
        'normal_budget_usage':normal['budget_usage'],'resume_budget_usage':resume['budget_usage'],
        'original_receipt_sha256':first_sha,'original_receipt_preserved':True,'raw_artifacts':str(root),
        'commands':commands,'receipts':{name:file_hash(root/path) for name,path in {
            'first':'normal/receipt.json','resume':'normal/receipts/invocation-000001.json','budget':'budget/receipt.json'}.items()}}
    save(evidence/'summary.json',summary)
    print(json.dumps(summary,sort_keys=True))


if __name__=='__main__':main()
