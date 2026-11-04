import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image
from rswm_inria.adapters.camera_conditioning import (
    CameraConfig, build_camera_batch, frame_indices, plucker_at_pixels,
    iter_native_plucker, to_native_rgb, from_native_rgb, scale_audit,
)
from rswm_inria.frontier import AuditedWindowMap
from rswm_inria.baselines import WindowMap
from rswm_inria.geometry import DIAGONAL, file_hash
from rswm_inria.missing_bounds import rgb_missing_bounds, weighted_bounds
from rswm_inria.source_probe import PROFILES, select_probe, verify_probe, inspect_image, inspect_local, save_new_json

SPEC = {'viewport_hw': [240,416], 'query_bounds_xy':[[-704,704],[-704,704]]}


def query32():
    return np.vstack([np.arange(1,17)[:,None]*[12,0], [192,0]+np.arange(1,17)[:,None]*[0,12]])


@pytest.mark.parametrize('framewise',[False,True])
@pytest.mark.parametrize('stride',[1,2,4,8,16])
def test_fixed_retains_scale_and_legacy_can_alias(framewise,stride):
    for mode,ratio in [('fixed_units',2),('legacy_maxnorm',1)]:
        cfg=CameraConfig(mode,framewise,temporal_stride=stride)
        a=build_camera_batch(query32(),config=cfg);b=build_camera_batch(2*query32(),config=cfg)
        np.testing.assert_allclose(np.linalg.norm(b.camera_poses[:,:3,3])/np.linalg.norm(a.camera_poses[:,:3,3]),ratio)
        ra=plucker_at_pixels(a.camera_poses,a.intrinsics_fx_fy_cx_cy,[[.5,.5],[416.5,240.5]])
        rb=plucker_at_pixels(b.camera_poses,b.intrinsics_fx_fy_cx_cy,[[.5,.5],[416.5,240.5]])
        np.testing.assert_allclose(ra[:,:,:3],rb[:,:,:3])
        np.testing.assert_allclose(rb[:,:,3:],ratio*ra[:,:,3:])
        if mode=='legacy_maxnorm':assert np.array_equal(ra,rb)


@pytest.mark.parametrize('n,expected',[(17,[0,4,8,12,16]),(33,[0,4,8,12,16,20,24,28,32])])
def test_explicit_temporal_indices(n,expected):
    assert frame_indices(n).tolist()==expected


@pytest.mark.parametrize('n',[0,1,16,18,32,81,17.,True])
def test_native_frame_counts_rejected(n):
    with pytest.raises(ValueError):frame_indices(n)


@pytest.mark.parametrize('kwargs',[{'translation_scaling':'per_path'}, {'framewise':1}, {'camera_channel':'foo'},
                                    {'temporal_stride':0},{'temporal_stride':3},{'temporal_stride':True}])
def test_bad_camera_config(kwargs):
    with pytest.raises(ValueError):CameraConfig(**kwargs)


@pytest.mark.parametrize('value',[
    np.zeros((16,2),dtype=float),np.zeros((16,2),dtype=bool),np.zeros((15,2),dtype=int),
    np.zeros((16,3),dtype=int),np.full((16,2),705,dtype=int),
    np.full((16,2),2**64-1,dtype=np.uint64),np.full((16,2),-2**63,dtype=np.int64),
])
def test_bad_queries(value):
    with pytest.raises(ValueError):build_camera_batch(value)


def test_bad_anchor_and_large_jump():
    with pytest.raises(ValueError):build_camera_batch(np.zeros((16,2),int),(0.,0.))
    with pytest.raises(ValueError):build_camera_batch(np.zeros((16,2),int),(705,0))
    with pytest.raises(ValueError):build_camera_batch(np.tile([400,0],(16,1)))


def test_current_call_is_anchor_relative_not_task_rezeroing():
    q=query32()[:16];a=build_camera_batch(q)
    b=build_camera_batch(q+[96,64],anchor_global_xy=(96,64))
    np.testing.assert_allclose(a.camera_poses,b.camera_poses)
    assert b.native_queries_xy[0].tolist()==[96,64]
    assert b.native_queries_xy[-1].tolist()==[288,64]


def test_no_future_dependence_under_fixed_units():
    q=query32();r=q.copy();r[16:]=[192,0]+np.arange(1,17)[:,None]*[0,24]
    for framewise in (False,True):
        cfg=CameraConfig(framewise=framewise)
        a=build_camera_batch(q,config=cfg);b=build_camera_batch(r,config=cfg)
        np.testing.assert_array_equal(a.camera_poses[:5],b.camera_poses[:5])
    a=build_camera_batch(q,config=CameraConfig('legacy_maxnorm'))
    b=build_camera_batch(r,config=CameraConfig('legacy_maxnorm'))
    assert not np.array_equal(a.camera_poses[:5],b.camera_poses[:5])


def test_irregular_segments_not_retimed():
    q=np.vstack([np.arange(1,7)[:,None]*[24,0],[144,0]+np.arange(1,27)[:,None]*[0,16]])
    batch=build_camera_batch(q,config=CameraConfig(temporal_stride=1))
    assert batch.native_queries_xy[6].tolist()==[144,0]
    assert batch.native_queries_xy[7].tolist()==[144,16]
    np.testing.assert_allclose(batch.camera_poses[6,:2,3],[144/DIAGONAL,0])


@pytest.mark.parametrize('framewise',[False,True])
def test_fixed_units_does_not_claim_to_solve_temporal_aliasing(framewise):
    q=np.tile([[16,0],[16,16],[0,16],[0,0]],(4,1))
    a=build_camera_batch(q,config=CameraConfig(framewise=framewise))
    b=build_camera_batch(np.zeros((16,2),int),config=CameraConfig(framewise=framewise))
    assert not np.array_equal(a.native_queries_xy,b.native_queries_xy)
    assert np.array_equal(a.camera_poses,b.camera_poses)


def test_drop_is_not_identity_and_no_extra_numeric_payload():
    dropped=build_camera_batch(query32(),config=CameraConfig(camera_channel='drop'))
    identity=build_camera_batch(query32(),config=CameraConfig(camera_channel='identity'))
    assert dropped.camera_poses is None and dropped.intrinsics_fx_fy_cx_cy is None
    assert dropped.native_queries_xy is None and identity.native_queries_xy is None
    assert list(iter_native_plucker(dropped))==[]
    assert np.array_equal(identity.camera_poses,np.tile(np.eye(4),(9,1,1)))
    assert len({dropped.config_sha256,identity.config_sha256,build_camera_batch(query32()).config_sha256})==3


def test_virtual_plane_sign_and_units():
    q=np.tile([24,16],(16,1));b=build_camera_batch(q)
    t=b.camera_poses[-1,:3,3];fx,fy,cx,cy=b.intrinsics_fx_fy_cx_cy[-1]
    native_shift=-t[:2]*[fx,fy]
    np.testing.assert_allclose(native_shift,[-48,-32])


def test_plucker_order_and_orthogonality():
    b=build_camera_batch(query32())
    rays=plucker_at_pixels(b.camera_poses,b.intrinsics_fx_fy_cx_cy,[[416,240],[.5,.5]])
    np.testing.assert_allclose(np.linalg.norm(rays[:,:,:3],axis=-1),1,atol=1e-6)
    np.testing.assert_allclose((rays[:,:,:3]*rays[:,:,3:]).sum(-1),0,atol=1e-6)
    assert rays.dtype==np.float32
    assert next(iter_native_plucker(b)).shape==(480,832,6)


@pytest.mark.parametrize('bad',['rotation','bottom','focal','nan','shape'])
def test_ray_validation(bad):
    b=build_camera_batch(query32());p=b.camera_poses.copy();k=b.intrinsics_fx_fy_cx_cy.copy();xy=np.array([[1.,1.]])
    if bad=='rotation':p[0,0,0]=2
    if bad=='bottom':p[0,3,0]=1
    if bad=='focal':k[0,0]=0
    if bad=='nan':xy[0,0]=np.nan
    if bad=='shape':k=k[:1]
    with pytest.raises(ValueError):plucker_at_pixels(p,k,xy)


def test_image_conversion_contract():
    a=np.full((240,416,3),127,np.uint8);n=to_native_rgb(a)
    assert n.shape==(480,832,3) and np.array_equal(from_native_rgb(n),a)
    n[:]=0;n[0,0]=1;n[0,1]=1
    assert np.all(from_native_rgb(n)[0,0]==1)
    with pytest.raises(ValueError):from_native_rgb(n.astype(float))


class ConstantCompleter:
    def __call__(self,image,known,seed):return np.full(image.shape,123,np.uint8)


@pytest.mark.parametrize('order',['raster','frontier','lazy'])
def test_map_preserves_initial_and_traces_shape(order):
    initial=np.full((240,416,3),17,np.uint8)
    model=AuditedWindowMap(ConstantCompleter(),order);model.reset(initial,SPEC,0)
    assert np.array_equal(model.generate([[0,0]])[0],initial)
    model.generate([[480,0]])
    for r in model.operation_log:
        assert sum(r[k] for k in ('real_known_pixels','generated_history_pixels','unknown_in_world_pixels','padding_pixels'))==512**2
        assert r['cpu_wall_seconds']>=0 and r['gpu_synchronized_seconds'] is None
    report=model.activity_report();assert report['resource_accounting_complete'] is False
    if order!='lazy':assert model.known.all() and model.generated_pixels==2048**2-240*416


def test_frontier_never_starts_without_known_support():
    a=np.zeros((240,416,3),np.uint8)
    front=AuditedWindowMap(ConstantCompleter(),'frontier');front.reset(a,SPEC,0)
    raster=AuditedWindowMap(ConstantCompleter(),'raster');raster.reset(a,SPEC,0)
    assert front.operation_log[0]['real_known_pixels']>0
    assert raster.operation_log[0]['real_known_pixels']==0
    assert all(r['real_known_pixels']+r['generated_history_pixels']>0 for r in front.operation_log)
    assert len(front.operation_log)==64 and len({tuple(r['window_xy']) for r in front.operation_log})==64


@pytest.mark.parametrize('order',['raster','lazy'])
def test_wrapper_keeps_original_map_pixels(order):
    a=np.full((240,416,3),17,np.uint8)
    new=AuditedWindowMap(order=order);old=WindowMap(eager=order=='raster')
    new.reset(a,SPEC,42);old.reset(a,SPEC,42)
    qs=[[0,0],[480,0],[0,480],[480,0]]
    for x,y in zip(new.generate(qs),old.generate(qs)):assert np.array_equal(x,y)


def test_frontier_deterministic():
    a=np.zeros((240,416,3),np.uint8);x=AuditedWindowMap(ConstantCompleter());y=AuditedWindowMap(ConstantCompleter())
    x.reset(a,SPEC,2);y.reset(a,SPEC,2)
    assert np.array_equal(x.map,y.map)
    assert [r['window_xy'] for r in x.operation_log]==[r['window_xy'] for r in y.operation_log]


def test_branch_state_isolated_actual_cost_never_rolled_back():
    a=np.zeros((240,416,3),np.uint8);m=AuditedWindowMap(ConstantCompleter(),'lazy');m.reset(a,SPEC,0)
    m.generate([[480,0]]);parent=m.snapshot();parent_pixels=parent['state']['map'].copy()
    m.generate([[0,480]]);n=len(m.operation_log)
    m.restore(parent);assert len(m.operation_log)==n
    m.generate([[0,-480]]);assert len(m.operation_log)>n
    assert np.array_equal(parent['state']['map'],parent_pixels)
    m.map[:]=0;assert np.array_equal(parent['state']['map'],parent_pixels)
    report=m.activity_report();assert report['attempted_fill_calls']>m.calls
    report['operation_log'].clear();assert m.operation_log


def test_fill_failure_recorded_without_committing():
    class Broken:
        def __call__(self,*args):raise RuntimeError('fake')
    m=AuditedWindowMap(Broken(),'frontier')
    with pytest.raises(RuntimeError):m.reset(np.zeros((240,416,3),np.uint8),SPEC,0)
    assert m.operation_log[0]['status']=='failed' and m.generated_pixels==0
    assert m.operation_log[0]['committed_core_pixels']==0
    with pytest.raises(RuntimeError):m.generate([[0,0]])


@pytest.mark.parametrize('queries',[[],[[0.,0.]],[[705,0]],np.full((1,2),2**64-1,dtype=np.uint64)])
def test_audited_map_rejects_invalid_queries(queries):
    m=AuditedWindowMap(order='lazy');m.reset(np.zeros((240,416,3),np.uint8),SPEC,0)
    with pytest.raises(ValueError):m.generate(queries)


def test_map_snapshot_contract():
    m=AuditedWindowMap(order='lazy');m.reset(np.zeros((240,416,3),np.uint8),SPEC,0);s=m.snapshot()
    s['order']='raster'
    with pytest.raises(ValueError):m.restore(s)
    with pytest.raises(RuntimeError):m.generate([[0,0]])


@pytest.mark.parametrize('triple',[(0,0,0),(100,0,0),(100,100,0),(100,100,76500),(100,95,765)])
def test_missing_bounds_formula(triple):
    n,v,e=triple;r=rgb_missing_bounds(n,v,e)
    if not n:assert r['lower'] is None and r['state']=='not_applicable'
    else:
        assert r['lower']==pytest.approx(e/(765*n),abs=1e-15)
        assert r['upper']==pytest.approx(r['lower']+1-v/n)
        assert r['conditional_mae'] is None if v==0 else r['conditional_mae']==e/(765*v)


@pytest.mark.parametrize('triple',[(-1,0,0),(1,2,0),(1,0,1),(1,1,766),(True,0,0),(1.,0,0)])
def test_bounds_reject_invalid_counts(triple):
    with pytest.raises(ValueError):rgb_missing_bounds(*triple)


def test_bounds_plan_weights_not_valid_weighted():
    a=rgb_missing_bounds(100,100,0);b=rgb_missing_bounds(100,0,0)
    r=weighted_bounds([a,b],[.5,.5]);assert r['lower']==0 and r['upper']==.5
    with pytest.raises(ValueError):weighted_bounds([a,b],[.9,.9])
    with pytest.raises(ValueError):weighted_bounds([a,b],[1.,0.])
    assert weighted_bounds([a,rgb_missing_bounds(0,0,0)],[.5,.5])['lower'] is None


def fair_inventory():
    cfg=PROFILES['fair1m32']
    return {'repo_id':cfg['repo_id'],'revision':cfg['revision'],
            'files':[{'path':f'data/images/{i}.tif','size_bytes':4_000_000,
                      'lfs_sha256':hashlib.sha256(str(i).encode()).hexdigest()} for i in range(40)]}


def test_probe_is_bounded_hash_selected_and_order_invariant():
    inv=fair_inventory();a=select_probe(inv,'fair1m32');inv['files'].reverse();b=select_probe(inv,'fair1m32')
    assert a==b and a['total_source_bytes']==128_000_000 and len(a['files'])==32
    verify_probe(a,inv)
    assert a['core_leaderboard'] is False and a['source_pixels_validated'] is False


@pytest.mark.parametrize('attack',['revision','repo','duplicate','missing','no-sha','negative-size','budget','plan'])
def test_candidate_probe_rejects_bad_metadata(attack):
    inv=fair_inventory();plan=select_probe(inv,'fair1m32')
    if attack=='revision':inv['revision']='main'
    if attack=='repo':inv['repo_id']='other/repo'
    if attack=='duplicate':inv['files'].append(inv['files'][0])
    if attack=='missing':inv['files']=inv['files'][:10]
    if attack=='no-sha':inv['files'][0]['lfs_sha256']=None
    if attack=='negative-size':inv['files'][0]['size_bytes']=-1
    if attack=='budget':
        for row in inv['files']:row['size_bytes']=20_000_000
    if attack=='plan':plan['files'][0]['path']='../../outside'
    with pytest.raises(ValueError):verify_probe(plan,inv)


def test_oscd_fixed_pointer_and_no_training_shard():
    cfg=PROFILES['oscd-test'];inv={'repo_id':cfg['repo_id'],'revision':cfg['revision'],'files':[
        {'path':'data/test-00000-of-00001.parquet','size_bytes':12121086,
         'lfs_sha256':'22844e3529f150724007c959788ff2f1dc99e97c8622ba374aecfafd9c9404ff'},
        {'path':'data/train-00000-of-00001.parquet','size_bytes':25899561,'lfs_sha256':None}]}
    p=select_probe(inv,'oscd-test');assert p['total_source_bytes']==12121086
    inv['files'][0]['size_bytes']+=1
    with pytest.raises(ValueError):select_probe(inv,'oscd-test')


def test_inspect_real_header_not_filename_or_dataset_card(tmp_path):
    p=tmp_path/'not-tiff.tif';Image.new('RGB',(1024,1024)).save(p,format='PNG')
    with Image.open(p) as im:r=inspect_image(im,'TIFF')
    assert r['format']=='PNG' and not r['strict_rgb']
    p2=tmp_path/'small.tif';Image.new('RGB',(600,600)).save(p2,format='TIFF')
    with Image.open(p2) as im:r=inspect_image(im,'TIFF')
    assert r['fits_viewport'] and not r['fits_1024_world'] and r['width']==600


def test_inspection_local_file_manifest_full_roundtrip(tmp_path):
    inv=fair_inventory()
    for row in inv['files']:
        p=tmp_path/row['path'];p.parent.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(8,8)).save(p,format='TIFF')
        row['size_bytes']=p.stat().st_size;row['lfs_sha256']=file_hash(p)
    plan=select_probe(inv,'fair1m32');r=inspect_local(plan,inv,tmp_path)
    assert len(r['rows'])==32 and all(not x['fits_1024_world'] for x in r['rows'])
    assert r['benchmark_materialized'] is False


def test_json_exclusive_and_no_nan_file(tmp_path):
    p=tmp_path/'a.json';save_new_json(p,{'a':1})
    with pytest.raises(FileExistsError):save_new_json(p,{'a':2})
    q=tmp_path/'b.json'
    with pytest.raises(ValueError):save_new_json(q,{'bad':float('nan')})
    assert not q.exists()


def test_camera_audit_cli(tmp_path):
    p=tmp_path/'audit.json'
    subprocess.run([sys.executable,'scripts/audit_camera_contract.py','--out',str(p)],check=True)
    r=json.loads(p.read_text());assert len(r['records'])==4 and r['real_weights_used'] is False


def test_source_probe_cli_plan_does_not_download(tmp_path):
    inv=tmp_path/'source.json';inv.write_text(json.dumps(fair_inventory()))
    out=tmp_path/'plan.json'
    subprocess.run([sys.executable,'-m','rswm_inria.source_probe','plan','--profile','fair1m32','--inventory',str(inv),'--out',str(out)],check=True)
    assert len(json.loads(out.read_text())['files'])==32
    assert {p.name for p in tmp_path.iterdir()}=={'source.json','plan.json','plan-inventory.json'}


@pytest.mark.parametrize('bad',['{"files":[],"files":[]}', '{"unknown":NaN}'])
def test_source_probe_cli_rejects_ambiguous_json_before_publishing(tmp_path,bad):
    inv=tmp_path/'bad.json';inv.write_text(bad)
    out=tmp_path/'plan.json'
    result=subprocess.run([sys.executable,'-m','rswm_inria.source_probe','plan','--profile','fair1m32',
                           '--inventory',str(inv),'--out',str(out)],capture_output=True)
    assert result.returncode!=0 and not out.exists()
