import copy
import hashlib
import numpy as np
import pytest
from PIL import Image
from rswm_inria.data import REPO,ROLES,select_manifest,verify_manifest,downsample2,build_world,validate_source_file,write_json
from rswm_inria.geometry import file_hash,rng_for,crop,ORIGINS
from rswm_inria.score_rgb import Ledger
from rswm_inria.quality import plan_tiles,assemble_tiles,polynomial_mmd_unbiased
from rswm_inria.stats import paired_city_summary


def synthetic_inventory():
    rows=[]
    for city,(role,_) in ROLES.items():
        split="train" if city in ("austin","chicago","vienna","kitsap") else "test"
        for tile in range(1,37):
            path=f"data/{split}/images/{city}{tile}.tif"
            rows.append({"path":path,"size_bytes":75000000,"lfs_sha256":hashlib.sha256(path.encode()).hexdigest()})
    rows.append({"path":"data/train/gt/austin1.tif","size_bytes":25,"lfs_sha256":None})
    return {"repo_id":REPO,"revision":"a"*40,"files":rows}


def test_manifest_deterministic_city_counts():
    inventory=synthetic_inventory();a=select_manifest(inventory)
    inventory["files"].reverse();b=select_manifest(inventory)
    assert a==b and len(a["worlds"])==120
    assert sum(r["role"]=="train" for r in a["worlds"])==60
    assert sum(r["role"]=="test_unseen_city" for r in a["worlds"])==24
    assert all("/images/" in r["path"] for r in a["worlds"])
    verify_manifest(a)


@pytest.mark.parametrize("revision",["main","abc","A"*40,"a"*39])
def test_unpinned_metadata_rejected(revision):
    inventory=synthetic_inventory();inventory["revision"]=revision
    with pytest.raises(ValueError):select_manifest(inventory)


def test_missing_city_does_not_fallback():
    inventory=synthetic_inventory();inventory["files"]=[r for r in inventory["files"] if "sfo" not in r["path"]]
    with pytest.raises(ValueError):select_manifest(inventory)


def test_explicit_city_alias():
    inventory=synthetic_inventory()
    for row in inventory["files"]:row["path"]=row["path"].replace("sfo","sanfrancisco")
    assert sum(r["city"]=="sfo" for r in select_manifest(inventory)["worlds"])==12


def test_manifest_tampering():
    m=select_manifest(synthetic_inventory());m["worlds"][0]["role"]="test_unseen_city"
    with pytest.raises(ValueError):verify_manifest(m)


@pytest.mark.parametrize("values,expected",[([0,0,0,0],0),([0,0,1,1],1),([0,0,0,1],0),([255]*4,255)])
def test_exact_half_up_without_overflow(values,expected):
    a=np.repeat(np.array(values,np.uint8).reshape(2,2,1),3,axis=2)
    assert np.all(downsample2(a)==expected)


def test_source_hash_and_atomic_metadata(tmp_path):
    p=tmp_path/"file.bin";p.write_bytes(b"abc")
    row={"size_bytes":3,"lfs_sha256":file_hash(p)};validate_source_file(p,row)
    row["size_bytes"]=4
    with pytest.raises(ValueError):validate_source_file(p,row)
    j=tmp_path/"record.json";write_json(j,{"a":1});write_json(j,{"a":1})
    with pytest.raises(FileExistsError):write_json(j,{"a":2})


def test_synthetic_fullsize_tiff_roundtrip(tmp_path):
    a=np.zeros((5000,5000,3),np.uint8);a[:,:,0]=73;a[:,:,1]=151
    p=tmp_path/"synthetic.tif";Image.fromarray(a).save(p,compression="tiff_deflate")
    row={"world_id":"synthetic-validation-not-inria","size_bytes":p.stat().st_size,"lfs_sha256":file_hash(p)}
    result=build_world(p,row,tmp_path/"worlds")
    assert result["nominal_gsd_m"]==.6 and result["source_crop_xyxy"]==[452,452,4548,4548]
    with Image.open(tmp_path/"worlds"/result["path"]) as im:
        world=np.asarray(im);assert world.shape==(2048,2048,3) and (world[:,:,0]==73).all()
    assert result["georeference_status"]=="not available"


def test_masks_never_implicitly_converted(tmp_path):
    p=tmp_path/"mask.tif";Image.fromarray(np.zeros((8,8),np.uint8)).save(p)
    with pytest.raises(ValueError):build_world(p,{"size_bytes":p.stat().st_size,"lfs_sha256":None,"world_id":"mask"},tmp_path/"out")


def test_geometry_tile_plan_not_replaced_on_failure():
    q=[[0,0],[128,0],[256,0],[384,0],[512,0]];origin=(960,960)
    plan=plan_tiles(q,origin,"tiles",count=8)
    world=np.full((2048,2048,3),127,np.uint8);ledger=Ledger(crop(world,(0,0),origin),origin)
    for t,query in enumerate(q[1:],1):ledger.submit(t,query,None)
    outputs,coverage=assemble_tiles(ledger,plan)
    assert plan["eligible_tiles"]>0 and not outputs and coverage["coverage"]==0


def test_feature_mmd_not_clipped_and_validation():
    x=np.array([[0.,0.],[1.,1.],[2.,2.]])
    r=polynomial_mmd_unbiased(x,x)
    assert r["polynomial_mmd_u"]<0 and r["release_validated"] is False
    with pytest.raises(ValueError):polynomial_mmd_unbiased(x[:1],x)


def test_stats_preserve_failed_and_absent_worlds():
    expected=[{"world_id":"a","city":"sfo"},{"world_id":"b","city":"sfo"},{"world_id":"c","city":"innsbruck"}]
    a=[{"world_id":"a","value":2.},{"world_id":"c","value":5.}]
    b=[{"world_id":"a","value":1.},{"world_id":"b","value":None},{"world_id":"c","value":3.}]
    result=paired_city_summary(expected,a,b,bootstrap=100)
    assert result["equal_city_mean_a_minus_b"]==1.5
    assert result["cities"]["sfo"]["planned_worlds"]==2 and result["cities"]["sfo"]["paired_coverage"]==.5
