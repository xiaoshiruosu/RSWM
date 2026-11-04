"""Training publication faults; synthetic engine controls, no real training PASS."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def engine_control(monkeypatch):
    torch = pytest.importorskip('torch')
    from rswm_inria import training
    from rswm_inria.adapters import camera_conditioning
    # Isolate driver mechanics. This does not construct or weaken ClipDataset.
    monkeypatch.setattr(training, 'validate_dev_index', lambda *_args: None)
    monkeypatch.setattr(camera_conditioning, 'to_native_rgb', lambda _frame: np.zeros((2,2,3), np.uint8))
    class Dataset:
        diagnostic = False
        index_sha256 = 'a'*64; source_audit_sha256 = 'b'*64
        def __len__(self): return 1
        def __getitem__(self, _index):
            return {'images': np.zeros((33,1,1,3),np.uint8), 'q': np.zeros((33,2),np.int64)}
    class Backend:
        def __init__(self):
            self.model = torch.nn.Linear(1,1); self.count = 0
        def configure_trainable(self, _policy): pass
        def loss(self, **_kwargs): return (self.model.weight-.7).square().mean()
        def validate(self, *_args, **_kwargs):
            self.count += 1
            return {'status': 'SOFTWARE_MECHANICS_ONLY', 'forwards': self.count}
    config = {'updates':1,'batch_size':1,'accumulation':1,'lr':.01,'seed':7,'precision':'fp32',
              'device':'cpu','checkpoint_every':1,'trainable_policy':{'software_only':True},
              'backend_manifest':{'scope':'synthetic_engine_control'},'validation_index':[{'world_id':'software-only'}],
              'camera':{'translation_scaling':'fixed_units','framewise':False,'camera_channel':'drop','temporal_stride':4}}
    return SimpleNamespace(torch=torch, training=training, dataset=Dataset(), backend=Backend(), config=config)


def test_existing_receipt_is_never_overwritten(engine_control, tmp_path):
    original = b'previous retained receipt\n'; receipt = tmp_path/'update-00000001.json'; receipt.write_bytes(original)
    with pytest.raises(FileExistsError):
        engine_control.training.train(engine_control.dataset,engine_control.backend,engine_control.config,tmp_path)
    assert receipt.read_bytes() == original


def test_checkpoint_publish_race_is_exclusive(engine_control, tmp_path, monkeypatch):
    path = tmp_path/'update-00000001.pt'; original_save = engine_control.torch.save
    def racing_save(*args, **kwargs):
        result = original_save(*args, **kwargs)
        path.write_bytes(b'other cooperative writer checkpoint')
        return result
    monkeypatch.setattr(engine_control.torch,'save',racing_save)
    with pytest.raises(FileExistsError):
        engine_control.training.train(engine_control.dataset,engine_control.backend,engine_control.config,tmp_path)
    assert path.read_bytes() == b'other cooperative writer checkpoint'
    assert not path.with_suffix('.json').exists()


class ShortStream:
    def __init__(self, stream): self.stream=stream
    def __enter__(self): self.stream.__enter__(); return self
    def __exit__(self,*args): return self.stream.__exit__(*args)
    def __getattr__(self,name): return getattr(self.stream,name)
    def write(self,data): return self.stream.write(data[:-5])


def test_short_receipt_write_cannot_report_completion(engine_control, tmp_path, monkeypatch):
    original_open = Path.open
    def short_open(path,*args,**kwargs):
        stream = original_open(path,*args,**kwargs)
        return ShortStream(stream) if path.name.startswith('update-') and '.json' in path.name or path.name.startswith('.update-') and '.json' in path.name else stream
    monkeypatch.setattr(Path,'open',short_open)
    with pytest.raises(OSError):
        engine_control.training.train(engine_control.dataset,engine_control.backend,engine_control.config,tmp_path)
    if (tmp_path/'update-00000001.json').exists():
        with pytest.raises((ValueError,OSError)):
            engine_control.training.validate_checkpoint_receipt(tmp_path/'update-00000001.pt', 'invalid')


def test_checkpoint_sync_failure_cannot_report_completion(engine_control,tmp_path,monkeypatch):
    original_fsync=engine_control.training.os.fsync
    def fail_checkpoint(fd):
        target=os.readlink(f'/proc/self/fd/{fd}')
        if '.pt.' in target and target.endswith('.tmp'):
            raise OSError('injected checkpoint sync')
        return original_fsync(fd)
    monkeypatch.setattr(engine_control.training.os,'fsync',fail_checkpoint)
    with pytest.raises(OSError):
        engine_control.training.train(engine_control.dataset,engine_control.backend,engine_control.config,tmp_path)
    assert not (tmp_path/'update-00000001.json').exists()


def test_receipt_publish_race_retains_other_writer(engine_control,tmp_path,monkeypatch):
    original_save=engine_control.torch.save
    path=tmp_path/'update-00000001.json'
    def receipt_race(*args,**kwargs):
        result=original_save(*args,**kwargs)
        path.write_bytes(b'other writer receipt')
        return result
    monkeypatch.setattr(engine_control.torch,'save',receipt_race)
    with pytest.raises(FileExistsError):
        engine_control.training.train(engine_control.dataset,engine_control.backend,engine_control.config,tmp_path)
    assert path.read_bytes()==b'other writer receipt'
    assert (tmp_path/'update-00000001.pt').exists()  # Complete orphan, never an endorsed checkpoint.


def test_uncheckpointed_failed_validation_keeps_attempt_cost(engine_control,tmp_path,monkeypatch):
    def failure(*_args, **_kwargs):
        engine_control.backend.count += 1
        raise OSError('injected real validation call failure')
    monkeypatch.setattr(engine_control.backend,'validate',failure)
    with pytest.raises(OSError):
        engine_control.training.train(engine_control.dataset,engine_control.backend,engine_control.config,tmp_path)
    assert not (tmp_path/'update-00000001.pt').exists()
    path = tmp_path/'attempts/attempt-000/events.jsonl'
    assert path.is_file(), 'Attempt costs must survive before any checkpoint exists'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert any(row['kind']=='validation_call_started' for row in rows)
    finish = next(row for row in rows if row['kind']=='validation_call_finished')
    assert finish['status']=='FAILED' and finish['wall_seconds']>=0


def test_receipt_step_and_cursor_must_match_actual_checkpoint(engine_control,tmp_path):
    root=tmp_path/'run'
    engine_control.training.train(engine_control.dataset,engine_control.backend,engine_control.config,root)
    path=root/'update-00000001.json'; receipt=json.loads(path.read_text())
    receipt['training']['step']=2; path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='identity/step/cursor'):
        engine_control.training.train(engine_control.dataset,engine_control.backend,engine_control.config,root,
                                      resume=root/'update-00000001.pt')


def test_missing_or_corrupt_attempts_cannot_reset_costs(tmp_path):
    from rswm_inria.training import start_training_attempt, retained_training_attempts
    identity={'software_scope':'not_real_training'}
    journal,_=start_training_attempt(tmp_path/'run','a'*64,identity)
    journal.append('validation_forward_started',forward=1)
    logs,count=retained_training_attempts(tmp_path/'run','a'*64,identity)
    assert count==1 and logs[0]['unfinished_validation_forwards']==[1]
    # Hard crash retains unknown duration rather than claiming a zero cost.
    assert 'wall_seconds' not in logs[0]
    log=journal.root/'events.jsonl'; log.write_bytes(log.read_bytes()[:-1])
    with pytest.raises(ValueError,match='Incomplete'):
        retained_training_attempts(tmp_path/'run','a'*64,identity)
    log.unlink()
    with pytest.raises(ValueError,match='Missing'):
        start_training_attempt(tmp_path/'run','a'*64,identity)


def test_validation_cost_survives_actual_new_process(tmp_path):
    pytest.importorskip('torch')
    script=tmp_path/'controlled_validation.py'
    script.write_text('''import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from rswm_inria.geometry import digest,file_hash
from rswm_inria.adapters.camera_conditioning import CameraConfig
from rswm_inria.legacy_training import LegacyTrainingMixin
from rswm_inria.training import start_training_attempt
root=Path(sys.argv[1]); phase=sys.argv[2]; root.mkdir(exist_ok=True)
world=root/'world.png'
if not world.exists(): Image.fromarray(np.zeros((2048,2048,3),np.uint8)).save(world)
row={'world_id':'software-dev','city':'kitsap','role':'dev','path':'world.png','world_file_sha256':file_hash(world),'world_pixel_sha256':hashlib.sha256(np.zeros((2048,2048,3),np.uint8).tobytes()).hexdigest()}
cam=CameraConfig(translation_scaling='fixed_units',framewise=False,camera_channel='drop',temporal_stride=4)
class Backend(LegacyTrainingMixin):
 def __init__(self):
  self.runtime_config={'sampling':{'inactive_model_cpu':False},'training':{'trainable_policy':{'mode':'new_lora','rank':2,'alpha':2,'targets':'attn_only','gradient_checkpointing':False},'timestep_sampling':'uniform_integer','expert_residency':'resident','precision':'bf16','validation':{'world_ids':['software-dev'],'family':'F2','origin_index':0,'seed':9,'max_total_forwards':1}}}
  self._training_world_root=root;self._training_dev_rows=[row];self._training_camera_sha256=digest(cam.manifest());self._validation_forwards=0;self._loss_active_groups={}
 def training_components(self): return {'device':torch.device('cpu'),'dtype':torch.float32}
 def loss(self,**kwargs): raise OSError('controlled model forward failure')
backend=Backend();journal,count=start_training_attempt(root/'run','a'*64,{'scope':'software-only'})
backend.bind_validation_journal(journal,consumed_forwards=count)
try: backend.validate([row],camera_manifest=cam.manifest())
except (OSError,ValueError) as exc:
 print(json.dumps({'phase':phase,'counter':backend._validation_forwards,'error_type':type(exc).__name__,'budget_exhausted':'budget exhausted' in str(exc)}))
else: raise AssertionError('must fail closed')
''')
    env=dict(os.environ); env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
    first=subprocess.run([sys.executable,str(script),str(tmp_path/'assets'),'first'],env=env,text=True,capture_output=True,check=True)
    second=subprocess.run([sys.executable,str(script),str(tmp_path/'assets'),'second'],env=env,text=True,capture_output=True,check=True)
    a=json.loads(first.stdout);b=json.loads(second.stdout)
    assert a=={'phase':'first','counter':1,'error_type':'OSError','budget_exhausted':False}
    assert b=={'phase':'second','counter':1,'error_type':'ValueError','budget_exhausted':True}
    from rswm_inria.training import retained_training_attempts
    retained,count=retained_training_attempts(tmp_path/'assets/run','a'*64,{'scope':'software-only'})
    assert count==1 and len(retained)==2 and retained[0]['unfinished_validation_forwards']==[]


@pytest.mark.parametrize('fault',['short','close'])
def test_stream_checkpoint_fault_has_no_final_name(tmp_path,monkeypatch,fault):
    from rswm_inria.artifact_io import save_stream_exclusive
    original_open=Path.open
    class Fault(ShortStream):
        def write(self,data): return super().write(data) if fault=='short' else self.stream.write(data)
        def __exit__(self,*args):
            result=self.stream.__exit__(*args)
            if fault=='close': raise OSError('controlled close fault')
            return result
    monkeypatch.setattr(Path,'open',lambda path,*args,**kwargs: Fault(original_open(path,*args,**kwargs)))
    with pytest.raises(OSError): save_stream_exclusive(tmp_path/'model.pt',lambda stream:stream.write(b'complete software control'))
    assert list(tmp_path.iterdir())==[]


def test_concurrent_driver_rejected_and_process_death_preserves_costs(engine_control,tmp_path):
    """Actual separate writers; the tiny model proves software mechanics only."""
    import select
    root=tmp_path/'run'
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(engine_control.config))
    script=tmp_path/'active_driver.py'
    script.write_text('''import json,sys
from pathlib import Path
import numpy as np
import torch
from rswm_inria import training
from rswm_inria.adapters import camera_conditioning
training.validate_dev_index=lambda *_args:None
camera_conditioning.to_native_rgb=lambda _frame:np.zeros((2,2,3),np.uint8)
class Dataset:
 diagnostic=False;index_sha256='a'*64;source_audit_sha256='b'*64
 def __len__(self):return 1
 def __getitem__(self,_index):return {'images':np.zeros((33,1,1,3),np.uint8),'q':np.zeros((33,2),np.int64)}
class Backend:
 def __init__(self):self.model=torch.nn.Linear(1,1)
 def configure_trainable(self,_policy):pass
 def loss(self,**_kwargs):return (self.model.weight-.7).square().mean()
 def bind_validation_journal(self,journal,*,consumed_forwards):self.journal=journal;self.forwards=consumed_forwards
 def validate(self,*_args,**_kwargs):
  self.journal.append('validation_forward_started',forward=self.forwards+1)
  self.model(torch.ones(1,1))
  print('VALIDATION_ACTIVE',flush=True)
  sys.stdin.read()
  raise AssertionError('test process must be killed during this charged validation')
training.train(Dataset(),Backend(),json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
''')
    env=dict(os.environ);env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
    process=subprocess.Popen([sys.executable,str(script),str(config_path),str(root)],
                             env=env,text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        ready,_,_=select.select([process.stdout],[],[],30)
        assert ready, 'Controlled child did not reach its validation forward'
        assert process.stdout.readline().strip()=='VALIDATION_ACTIVE'
        with pytest.raises(ValueError,match='Another training coordinator is active'):
            engine_control.training.train(engine_control.dataset,engine_control.backend,engine_control.config,root)
        assert engine_control.backend.count==0
        assert sorted(p.name for p in (root/'attempts').iterdir())==['attempt-000']
        # SIGKILL releases the kernel lock, while all durable starts stay charged.
        process.kill();process.wait(timeout=10)
        session=json.loads((root/'attempts/attempt-000/events.jsonl').read_text().splitlines()[0])
        retained,count=engine_control.training.retained_training_attempts(root,session['contract_sha256'],session['training'])
        assert count==1 and retained[0]['unfinished_validation_forwards']==[1]
        assert any(call['phase']=='validation_call' and call['status']=='UNKNOWN_NO_COMPLETION'
                   for call in retained[0]['unfinished_calls'])
        with engine_control.training._training_execution_lock(root):
            journal,forwards=engine_control.training.start_training_attempt(root,session['contract_sha256'],session['training'])
        assert forwards==1 and journal.root.name=='attempt-001'
    finally:
        if process.poll() is None:process.kill()
        process.communicate(timeout=10)


def test_cli_rejects_active_run_before_approved_model_load(engine_control,tmp_path,monkeypatch):
    root=tmp_path/'run';config=tmp_path/'config.json';config.write_text(json.dumps(engine_control.config))
    monkeypatch.setattr(engine_control.training,'ClipDataset',lambda *_args,**_kwargs:engine_control.dataset)
    paths={}
    for name in ('index','clips','audit'):
        path=tmp_path/(name+'.json');path.write_text('{}');paths[name]=str(path)
    def unexpected_load(*_args,**_kwargs):raise AssertionError('Concurrent CLI must reject before model load')
    monkeypatch.setattr(engine_control.training,'load_approved_backend',unexpected_load)
    with engine_control.training._training_execution_lock(root):
        assert engine_control.training.main(['--config',str(config),'--index',paths['index'],
            '--clips',paths['clips'],'--source-audit',paths['audit'],'--root',str(tmp_path),
            '--out',str(root)])==2
    assert not (root/'training-contract.json').exists()
