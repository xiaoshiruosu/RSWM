import copy
import hashlib
import json
import shutil
import numpy as np
import pytest
from PIL import Image
from rswm_inria.data import REPO, ROLES, select_manifest, verify_manifest, download_sources, build_world
from rswm_inria.geometry import VERSION, ORIGINS, coordinates, box, crop, digest, canonical, file_hash
from rswm_inria.tasks import packages, validate_task, transform
from rswm_inria.paired import paired_packages, STUDY
from rswm_inria.runner import run_task
from rswm_inria.replay import replay_run, load_rgb_png, load_json


def inventory():
    rows = []
    for city in ROLES:
        split = 'train' if city in ('austin','chicago','vienna','kitsap') else 'test'
        for tile in range(1,37):
            path = f'data/{split}/images/{city}{tile}.tif'
            rows.append({'path':path,'size_bytes':75000000,'lfs_sha256':hashlib.sha256(path.encode()).hexdigest()})
    return {'repo_id':REPO,'revision':'a'*40,'files':rows}


def rehash_manifest(m):
    m.pop('manifest_sha256', None)
    m['manifest_sha256'] = digest(m)
    return m


def test_manifest_verification_scopes():
    inv = inventory(); manifest = select_manifest(inv)
    assert verify_manifest(manifest)['top_k_against_inventory_verified'] is False
    assert verify_manifest(manifest,inv)['top_k_against_inventory_verified'] is True
    assert verify_manifest(manifest,inv)['authenticity_verified'] is False


@pytest.mark.parametrize('field,value',[
    ('repo_id','another/repository'),('revision','main'),('revision','b'*40),
    ('version','another-version'),('total_source_bytes',1),('aliases',{}),('selection_namespace','wrong/'),
])
def test_rehashed_top_level_tampering(field,value):
    manifest = select_manifest(inventory()); manifest[field] = value
    with pytest.raises(ValueError):verify_manifest(rehash_manifest(manifest))


@pytest.mark.parametrize('field,value',[
    ('role','test_unseen_city'),('city','sfo'),('source_split','test'),('selection_rank',2),
    ('world_id','inria-'+'0'*16),('selection_hash','0'*64),('size_bytes',True),
    ('size_bytes',-1),('lfs_sha256','invalid'),('path','../../escape.tif'),
    ('path','data/train/gt/austin1.tif'),
])
def test_rehashed_row_tampering(field,value):
    manifest = select_manifest(inventory()); manifest['worlds'][0][field] = value
    with pytest.raises(ValueError):verify_manifest(rehash_manifest(manifest))


def test_valid_subset_cannot_certify_full_inventory_topk():
    inv = inventory(); original = select_manifest(inv)
    bad_inv = copy.deepcopy(inv)
    removed = original['worlds'][0]['path']
    bad_inv['files'] = [r for r in bad_inv['files'] if r['path'] != removed]
    changed = select_manifest(bad_inv)
    assert verify_manifest(changed)['structure_verified'] is True
    with pytest.raises(ValueError):verify_manifest(changed,inv)


def test_manifest_reordering_detected():
    m = select_manifest(inventory());m['worlds'].reverse()
    with pytest.raises(ValueError):verify_manifest(rehash_manifest(m))


@pytest.mark.parametrize('limit',[-1,121,1.5,True])
def test_download_limit_rejected_before_network(limit):
    with pytest.raises(ValueError):download_sources(select_manifest(inventory()),'unused',limit)


def test_zero_download_limit_without_network():
    assert download_sources(select_manifest(inventory()),'unused',0) == []


@pytest.mark.parametrize('value,dtype',[(2**64-1,np.uint64),(2**63,np.uint64),(-2**63,np.int64)])
def test_coordinate_overflow_does_not_alias_into_valid_range(value,dtype):
    q = np.array([[0,0],[value,0]],dtype=dtype)
    with pytest.raises(ValueError):coordinates(q)


def test_box_addition_uses_unbounded_python_integers():
    assert box(np.array([30000,30000],np.int16),np.array([30000,30000],np.int16)) == (59792,59880,60208,60120)


@pytest.mark.parametrize('oi',range(4))
def test_same_start_pair_keeps_prefix_and_labels(oi):
    pairs = paired_packages('paired-fixture','dev',oi)
    assert len(pairs)==4
    again = paired_packages('paired-fixture','dev',oi)
    for (task,labels),(other,_) in zip(pairs,again):
        assert canonical(task)==canonical(other)
        assert task['study']==STUDY and task['version']==VERSION
        validate_task(task,labels)
    for family in ('F5','F7'):
        (e,el),(o,ol) = [x for x in pairs if x[0]['family']==family]
        assert e['pair_id']==o['pair_id'] and e['origin']==o['origin']==list(ORIGINS[oi])
        displacement=transform(np.array([64,48]),e['metadata']['orientation'])
        for name in e['programs']:
            a,b=np.asarray(e['programs'][name]),np.asarray(o['programs'][name])
            assert np.array_equal(a[:81],b[:81])
            assert np.array_equal(b[-1]-a[-1],displacement)
            assert np.array_equal(el[name]['first_event'],ol[name]['first_event'])
            assert np.array_equal(el[name]['last_prefix_visible_event'],ol[name]['last_prefix_visible_event'])
            assert ol[name]['invisible_query_gap'].min()>=32


def test_core_definition_not_silently_replaced():
    core = packages('core-fixture','dev',0)
    paired_packages('core-fixture','dev',0)
    assert [canonical(t) for t,l in core]==[canonical(t) for t,l in packages('core-fixture','dev',0)]
    assert len(core)==4 and all('study' not in t for t,l in core)


def test_pair_no_fallback():
    with pytest.raises(ValueError):paired_packages('a','dev',0,max_attempts=0)


class OracleFixture:
    """Evaluator-only positive control; never a permitted ordinary model."""
    def __init__(self,world,origin,fail_prefix=False):
        self.world,self.origin,self.fail_prefix=world,origin,fail_prefix
    def reset(self,*args):self.calls=0
    def generate(self,queries):
        self.calls+=1
        return [None if self.fail_prefix and self.calls==1 else crop(self.world,q,self.origin) for q in queries]
    def snapshot(self):return self.calls
    def restore(self,state):self.calls=state


@pytest.fixture(scope='module')
def saved_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp('replay-fixtures')
    y,x = np.mgrid[:2048,:2048]
    world = np.stack([x%251,y%253,(x//8+y//8)%247],axis=-1).astype(np.uint8)
    result={}
    for task,labels in packages('replay-fixture','metric_validation',0):
        folder=root/task['family']
        original,journal=run_task(task,world,OracleFixture(world,task['origin']),labels=labels,out_dir=folder)
        result[task['family']] = (task,crop(world,(0,0),task['origin']),folder,original)
    task,labels=packages('replay-fail','metric_validation',0)[3]
    folder=root/'failed-first';original,_=run_task(task,world,OracleFixture(world,task['origin'],True),labels=labels,out_dir=folder)
    result['failed-first']=(task,crop(world,(0,0),task['origin']),folder,original)
    return result


@pytest.mark.parametrize('family',['F2','F3','F5','F7'])
def test_file_replay_equals_online_scores(saved_runs,family):
    task,initial,folder,original=saved_runs[family]
    # JSON serialization changes key ordering: F3 stream order must still be stable.
    stored=load_json((folder/'task.json').read_text())
    report=replay_run(stored,load_rgb_png(folder/'initial.png'),folder,original['journal_head'])
    assert report['reports']==original['reports']
    assert report['artifact_checks']['externally_anchored_head']
    assert report['protocol_security_verified'] is False
    assert report['planned_output_events']=={'F2':32,'F3':64,'F5':96,'F7':160}[family]


def test_missing_first_exposure_is_not_repaired_on_replay(saved_runs):
    task,initial,folder,original=saved_runs['failed-first']
    r=replay_run(task,initial,folder)
    assert r['reports']==original['reports']
    assert r['reports']['planned_outputs']==160
    assert r['reports']['valid_outputs']==144
    assert r['reports']['paired_reentry_mae'] is None
    assert r['reports']['metrics']['reentry']['coverage']==0


def copied_run(saved_runs,tmp_path):
    task,initial,source,original=saved_runs['F2']
    folder=tmp_path/'copy';shutil.copytree(source,folder)
    return copy.deepcopy(task),initial,folder,original


def rechain(folder, mutate):
    path=folder/'events.jsonl';rows=[load_json(line) for line in path.read_text().splitlines()]
    mutate(rows);head='0'*64
    for index,row in enumerate(rows):
        row.pop('hash',None);row['index']=index;row['previous_hash']=head
        row['hash']=hashlib.sha256(canonical(row)).hexdigest();head=row['hash']
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))


@pytest.mark.parametrize('attack',['pixel','delete','unlogged','failed-replacement','hash','coordinate','drop-output','drop-response','task','head','summary','duplicate-output'])
def test_replay_rejects_artifact_attacks(saved_runs,tmp_path,attack):
    task,initial,folder,original=copied_run(saved_runs,tmp_path)
    path=folder/'main'/'0001.png'
    expected_head=None
    if attack=='pixel':
        a=load_rgb_png(path);a[0,0,0]^=1;Image.fromarray(a).save(path)
    elif attack=='delete':path.unlink()
    elif attack=='unlogged':shutil.copy(path,folder/'main'/'9999.png')
    elif attack=='failed-replacement':
        def change(rows):
            r=next(r for r in rows if r['kind']=='output');r['status']='missing';r['pixel_sha256']=None
        rechain(folder,change)
    elif attack=='hash':
        p=folder/'events.jsonl';p.write_text(p.read_text().replace('"seed": 0','"seed": 1',1))
    elif attack=='coordinate':
        def change(rows):next(r for r in rows if r['kind']=='output')['q'][0]+=1
        rechain(folder,change)
    elif attack in ('drop-output','drop-response'):
        kind=attack.split('-')[1]
        def change(rows):rows.remove(next(r for r in rows if r['kind']==kind))
        rechain(folder,change)
    elif attack=='duplicate-output':
        def change(rows):rows.append(copy.deepcopy(next(r for r in rows if r['kind']=='output')))
        rechain(folder,change)
    elif attack=='task':task['world_id']='different-world'
    elif attack=='head':expected_head='f'*64
    elif attack=='summary':
        p=folder/'summary.json';r=load_json(p.read_text());r['reports']={};p.write_text(json.dumps(r))
    with pytest.raises(ValueError):replay_run(task,initial,folder,expected_head)


@pytest.mark.parametrize('text',['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'])
def test_json_ambiguities_rejected(text):
    with pytest.raises(ValueError):load_json(text)


def test_world_sidecar_preserves_source_path(tmp_path):
    a=np.zeros((5000,5000,3),np.uint8);a[:,:,1]=37
    source=tmp_path/'source.tif';Image.fromarray(a).save(source,compression='tiff_deflate')
    row={'world_id':'synthetic-path-check','path':'data/train/images/austin1.tif',
         'size_bytes':source.stat().st_size,'lfs_sha256':file_hash(source)}
    record=build_world(source,row,tmp_path/'worlds')
    assert record['source_path']==row['path']
    assert record['path']=='synthetic-path-check.png'
