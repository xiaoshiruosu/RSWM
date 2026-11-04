"""CPU engineering controls; no pretrained model or real scene validation."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from rswm_inria.geometry import digest, file_hash, crop
from rswm_inria.tasks import training_clips
from rswm_inria.training import ClipDataset, cursor_sample, locked_tree, load_approved_backend, configure_parameters
from rswm_inria import experiments


def dataset_fixture(tmp_path):
    world=np.zeros((2048,2048,3),np.uint8)
    world[:,:,0]=np.arange(2048,dtype=np.uint16)[None,:]%256
    world[:,:,1]=np.arange(2048,dtype=np.uint16)[:,None]%256
    Image.fromarray(world).save(tmp_path/'world.png')
    row={'world_id':'train-control','role':'train','path':'world.png','world_file_sha256':file_hash(tmp_path/'world.png'),'world_pixel_sha256':hashlib.sha256(world.tobytes()).hexdigest()}
    clips=training_clips('train-control','train')[:2]
    index={'worlds':[row]};data={'clips':clips,'clip_count':2,'view_events':66}
    return index,data,world


def test_train_loader_exact_crops_and_resume_cursor(tmp_path):
    index,data,world=dataset_fixture(tmp_path)
    ds=ClipDataset(index,data,tmp_path,diagnostic=True)
    sample=ds[0]
    assert sample['images'].shape==(33,240,416,3)
    for frame,q in zip(sample['images'],sample['q']):
        assert np.array_equal(frame,crop(world,q,data['clips'][0]['origin']))
    uninterrupted=[cursor_sample(ds,19,i)['clip_id'] for i in range(5)]
    resumed=[cursor_sample(ds,19,i)['clip_id'] for i in range(2,5)]
    assert resumed==uninterrupted[2:]


@pytest.mark.parametrize('role',['dev','metric_calibration','metric_validation','test_unseen_city'])
def test_training_rejects_every_nontrain_role(tmp_path,role):
    index,data,_=dataset_fixture(tmp_path);index['worlds'][0]['role']=role
    with pytest.raises(ValueError,match='role'):
        ClipDataset(index,data,tmp_path,diagnostic=True)


def test_training_rejects_geometry_hash_and_incomplete_formal(tmp_path):
    index,data,_=dataset_fixture(tmp_path)
    with pytest.raises(ValueError,match='train60'):ClipDataset(index,data,tmp_path)
    altered=copy.deepcopy(data);altered['clips'][0]['q'][1][0]+=1
    with pytest.raises(ValueError,match='geometry'):ClipDataset(index,altered,tmp_path,diagnostic=True)
    ds=ClipDataset(index,data,tmp_path,diagnostic=True)
    (tmp_path/'world.png').write_bytes(b'not image')
    with pytest.raises(ValueError,match='hash'):ds[0]


def test_original_backend_required_and_tree_is_locked(tmp_path):
    with pytest.raises(ValueError,match='approved'):load_approved_backend({},device='cpu')
    source=tmp_path/'source';source.mkdir();(source/'backend.py').write_text('def load_backend(**kwargs):\n    return type("Backend",(),{"objective":"diffusion"})()\n')
    weights=tmp_path/'weights';weights.mkdir();(weights/'model.bin').write_bytes(b'control fixture not real weights')
    manifest={'source_dir':str(source),'source_tree_sha256':digest(locked_tree(source)),
              'weights_dir':str(weights),'weights_tree_sha256':digest(locked_tree(weights)),
              'entrypoint':'backend.py','objective':'diffusion','approved':True,'license_record':'synthetic control only'}
    assert load_approved_backend(manifest,device='cpu').objective=='diffusion'
    (source/'backend.py').write_text('# changed source')
    with pytest.raises(ValueError,match='source changed'):load_approved_backend(manifest,device='cpu')


def test_trainable_parameters_inserted_by_lora_configuration_are_included():
    from types import SimpleNamespace
    class Backend:
        def __init__(self):
            self.parameters={'frozen':SimpleNamespace(requires_grad=False)}
            self.model=SimpleNamespace(named_parameters=lambda:list(self.parameters.items()))
        def configure_trainable(self,policy):
            self.parameters['lora']=SimpleNamespace(requires_grad=True)
    params,trainable=configure_parameters(Backend(),{'kind':'fixture'})
    assert set(params)=={'frozen','lora'} and set(trainable)=={'lora'}


def matrix(kind='pilot'):
    worlds=['w0','w1'] if kind=='pilot' else [f'{c}-{n}' for c in ('sfo','innsbruck') for n in range(12)]
    origins=[0] if kind=='pilot' else range(4)
    rows=[{'world_id':w,'origin_index':o,'family':f,'model_id':'copy','seed':0,'role':'dev' if kind=='pilot' else 'test_unseen_city','city':'kitsap' if kind=='pilot' else w.split('-')[0],'model_manifest_sha256':'a'*64,'source_manifest_sha256':'b'*64}
          for w in worlds for o in origins for f in ('F2','F3','F5','F7')]
    return {'expected':rows,'models':{'copy':{'config':{'model_id':'copy','adapter':'copy','device':'cpu'}}},'plan_sha256':'c'*64,'retry_policy':{'primary_selection':'first_attempt_only'}}


def test_pilot_and_formal_matrix_real_counts():
    assert experiments._matrix(matrix(),'pilot')['planned_unique_outputs']==704
    formal=experiments._matrix(matrix('formal'),'formal')
    assert formal['expected_runs']==384 and formal['planned_unique_outputs']==33792
    bad=matrix();bad['expected'].pop()
    with pytest.raises(ValueError,match='matrix'):experiments._matrix(bad,'pilot')


def test_formal_cannot_be_authorized_by_diagnostic_or_missing_pilot(monkeypatch):
    monkeypatch.setattr(experiments,'validate_plan',lambda p:None)
    with pytest.raises(ValueError,match='Diagnostic execution'):
        experiments.freeze_experiment(matrix('formal'),kind='formal',purpose='diagnostic',budget={'max_wall_seconds':10,'max_output_bytes':100})
    with pytest.raises(ValueError,match='Diagnostic adapter'):
        experiments.freeze_experiment(matrix(),kind='pilot',purpose='real',budget={'max_wall_seconds':10,'max_output_bytes':100})


def test_thin_self_signed_pass_source_cannot_authorize_formal(monkeypatch):
    monkeypatch.setattr(experiments,'validate_plan',lambda p:None)
    monkeypatch.setattr(experiments,'capabilities',lambda c:{'purpose':'real_generator'})
    audit={'real_data_validation':'PASS','source_manifest_sha256':'b'*64};audit['audit_sha256']=digest(audit)
    options={'kind':'formal','purpose':'real','budget':{'max_wall_seconds':10,'max_output_bytes':100},'source_audit':audit,'rights':{'data_use_approved':True,'weight_use_approved':True}}
    with pytest.raises(ValueError,match='current materialized audit'):
        experiments.freeze_experiment(matrix('formal'),**options)
    # A prior test accepted this thin mock. It now serves only as a negative
    # fixture; actual source, pilot and instrument producers must be used.
