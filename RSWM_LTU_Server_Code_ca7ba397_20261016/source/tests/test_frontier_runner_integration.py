"""Full-core integration; executed by repository CI in addition to targeted tests."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from rswm_inria.tasks import packages
from rswm_inria.runner import run_task
from rswm_inria.replay import replay_run
from rswm_inria.frontier import AuditedWindowMap
from rswm_inria.baselines import OracleMap
from rswm_inria.geometry import crop


@pytest.mark.parametrize('family,expected',[('F2',32),('F3',64),('F5',96),('F7',160)])
def test_audited_map_runs_core_and_replays(tmp_path,family,expected):
    task,labels=next(x for x in packages('frontier-runner-review','dev',0) if x[0]['family']==family)
    world=np.full((2048,2048,3),91,np.uint8)
    model=AuditedWindowMap(order='frontier')
    report,journal=run_task(task,world,model,labels=labels,out_dir=tmp_path/'run')
    replay=replay_run(task,crop(world,(0,0),task['origin']),tmp_path/'run',journal.head)
    assert replay['reports']==report['reports']
    assert replay['planned_output_events']==expected
    assert replay['verified_png_outputs']==expected
    rows=[report['reports']] if family=='F7' else report['reports'].values()
    for row in rows:
        assert row['output_completeness']==1
        assert all(s['mae']==0 for s in row['metrics'].values() if s['expected_pixels'])
    assert model.activity_report()['successful_fill_calls']==(128 if family=='F3' else 64)
    assert model.operation_log[0]['real_known_pixels']>0


@pytest.mark.parametrize('family,expected',[('F2',32),('F3',64),('F5',96),('F7',160)])
def test_four_family_evaluator_oracle_remains_exact_after_research_integration(tmp_path,family,expected):
    task,labels=next(x for x in packages('oracle-research-composition','dev',0) if x[0]['family']==family)
    # Actual synthetic pixels, not real Inria imagery or trained quality evidence.
    y,x=np.indices((2048,2048))
    world=np.stack([x%256,y%256,(x+y)%256],axis=-1).astype(np.uint8)
    report,journal=run_task(task,world,OracleMap(world,task['origin']),labels=labels,out_dir=tmp_path/'oracle')
    replay=replay_run(task,crop(world,(0,0),task['origin']),tmp_path/'oracle',journal.head)
    assert replay['reports']==report['reports']
    assert replay['verified_png_outputs']==replay['planned_output_events']==expected
    rows=[report['reports']] if family=='F7' else report['reports'].values()
    for row in rows:
        assert row['output_completeness']==1
        assert all(s['mae']==0 for s in row['metrics'].values() if s['expected_pixels'])


def test_audited_map_cli_requires_explicit_completer_and_replays(tmp_path):
    task,_=packages('frontier-cli-review','dev',0)[0]
    taskpath=tmp_path/'task.json';taskpath.write_text(json.dumps(task),encoding='utf-8')
    worldpath=tmp_path/'world.png';Image.fromarray(np.zeros((2048,2048,3),np.uint8)).save(worldpath)
    script=Path(__file__).resolve().parents[1]/'scripts'/'run_audited_map.py'
    command=[sys.executable,str(script),'--task',str(taskpath),'--world',str(worldpath),'--out',str(tmp_path/'out')]
    assert subprocess.run(command,capture_output=True).returncode!=0
    subprocess.run(command+['--diagnostic-noise','--order','lazy'],check=True,capture_output=True)
    result=json.loads((tmp_path/'out'/'recomputed.json').read_text())
    assert result['verified_png_outputs']==32
    info=json.loads((tmp_path/'out'/'map-run.json').read_text())
    assert info['completer']=='diagnostic_noise' and not info['real_model_validated']
    resources=json.loads((tmp_path/'out'/'resources.json').read_text())
    assert resources['schema']=='rswm-resource-v3' and resources['measurement_complete']
    assert resources['accounting_calls']['valid_outputs']==resources['accounting_calls']['planned_output_slots']==32
    assert resources['adapter_counters']['completion_calls']>0
    assert subprocess.run(command+['--diagnostic-noise'],capture_output=True).returncode!=0
