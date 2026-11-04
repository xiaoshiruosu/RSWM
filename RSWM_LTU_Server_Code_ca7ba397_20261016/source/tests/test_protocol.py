import copy
import numpy as np
import pytest
from rswm_inria.geometry import ORIGINS,box,crop,coordinates,rng_for,validate_path,DIAGONAL
from rswm_inria.tasks import packages,polyline,interpolate,target_labels,transform,training_clips
from rswm_inria.score_rgb import Ledger,Accumulator,branch_report
from rswm_inria.runner import run_task,verify_journal,validate_response,PUBLIC_SPEC
from rswm_inria.baselines import OracleMap,CopyModel,StaticNoiseMap,WindowMap,RandomModel


@pytest.fixture(scope="module")
def world():
    return rng_for("tests-world").integers(0,256,(2048,2048,3),dtype=np.uint8)


@pytest.mark.parametrize("origin",ORIGINS)
@pytest.mark.parametrize("q",[(-704,-704),(-704,704),(704,-704),(704,704)])
def test_all_legal_extremes(origin,q,world):
    assert crop(world,q,origin).shape==(240,416,3)


@pytest.mark.parametrize("q",[[[0.,0.],[1.,1.]],[[0,0],[705,0]],[[1,0],[2,0]],[[True,False],[False,True]]])
def test_reject_invalid_coordinates(q):
    with pytest.raises(ValueError):coordinates(q)


@pytest.mark.parametrize("seed",range(12))
def test_polyline_pair_and_determinism(seed):
    for paired in (False,True):
        q,meta=polyline(rng_for("poly-test",seed),paired)
        q2,meta2=polyline(rng_for("poly-test",seed),paired)
        assert np.array_equal(q,q2) and meta==meta2 and len(q)==33
        assert sum(meta["steps_per_segment"])==32 and min(meta["steps_per_segment"])>=6
        validate_path(q,(.6,1.25) if paired else (1,2.5))
        if paired:validate_path(q*2,(1.2,2.5))


def test_no_fallback():
    with pytest.raises(RuntimeError):polyline(rng_for(0),max_attempts=0)


def test_reference_reentry():
    q=interpolate([(0,0),(480,0),(480,480),(-480,480),(-480,-480),(480,0)],[16,16,32,16,16])
    labels=target_labels(q,(480,0))
    assert [labels["reentry_event"].min(),labels["reentry_event"].max()]==[92,95]
    assert [labels["invisible_query_gap"].min(),labels["invisible_query_gap"].max()]==[72,76]
    assert np.all(labels["fully_absent_calls"]==3)


@pytest.mark.parametrize("origin",range(4))
def test_package_budget(origin):
    bundle=packages("test-bundle","dev",origin)
    assert [t["family"] for t,l in bundle]==["F2","F3","F5","F7"]
    assert bundle[2][0]["metadata"]["variant"]==("offset" if origin in (1,3) else "exact")
    total=0
    for task,labels in bundle:
        total+=sum(len(p)-1 for p in task["programs"].values())-(32 if task["family"]=="F7" else 0)
        for q in task["programs"].values(): validate_path(q)
        for arrays in labels.values(): assert arrays["invisible_query_gap"].min()>=32
    assert total==352
    a,b=bundle[3][0]["programs"].values(); assert a[:33]==b[:33]


def test_rectangle_orientation_not_invariant():
    q=np.array([[0,0],[140,0]])
    validate_path(q)
    with pytest.raises(ValueError):validate_path(transform(q,1))


def test_training_scope_count():
    clips=training_clips("test-train","train")
    assert len(clips)==64 and sum(len(c["q"]) for c in clips)==2112
    with pytest.raises(ValueError):training_clips("test","dev")


def test_failed_first_exposure_never_repaired(world):
    origin=ORIGINS[0];initial=crop(world,(0,0),origin);ledger=Ledger(initial,origin)
    ledger.submit(1,(24,0),None)
    ledger.submit(2,(48,0),crop(world,(48,0),origin))
    history=ledger.counts["history"].report()
    assert history["expected_pixels"]==24*240 and history["valid_pixels"]==0 and history["mae"] is None
    wx,wy=origin[0]+220,origin[1]
    assert ledger.first_event[wy,wx]==1 and not ledger.valid[wy,wx]
    ledger.submit(3,(24,0),crop(world,(24,0),origin))
    assert not ledger.valid[wy,wx]


def test_output_is_copied_and_event_order_enforced(world):
    origin=ORIGINS[0];ledger=Ledger(crop(world,(0,0),origin),origin)
    frame=crop(world,(24,0),origin);before=frame.copy();ledger.submit(1,(24,0),frame);frame[:]=0
    assert np.array_equal(ledger.previous,before)
    with pytest.raises(ValueError):ledger.submit(1,(24,0),frame)


def test_accumulator_nulls_and_max_error():
    a=Accumulator();assert a.report()["state"]=="not_applicable"
    a.add(2);assert a.report()["mae"] is None and a.report()["coverage"]==0
    b=Accumulator();b.add(1,np.array([[0,0,0]],np.uint8),np.array([[255,255,255]],np.uint8))
    assert b.report()["mae"]==1


@pytest.mark.parametrize("family",["F2","F3","F5","F7"])
def test_oracle_zero_full_coverage(family,world):
    task,labels=next(x for x in packages("oracle-test","metric_validation",0) if x[0]["family"]==family)
    result,journal=run_task(task,world,OracleMap(world,task["origin"]),labels=labels)
    reports=[result["reports"]] if family=="F7" else result["reports"].values()
    for report in reports:
        assert report["output_completeness"]==1
        for score in report["metrics"].values():
            if score["expected_pixels"]:assert score["mae"]==0 and score["coverage"]==1
    assert verify_journal(journal.rows)==result["journal_head"]
    if family=="F7":assert result["reports"]["planned_outputs"]==160 and result["reports"]["paired_reentry_mae"]==0


def test_noise_map_can_have_perfect_consistency(world):
    task,labels=packages("noise-test","dev",0)[2]
    result,_=run_task(task,world,StaticNoiseMap(),labels=labels)
    scores=result["reports"]["main"]["metrics"]
    assert all(v["mae"]==0 for v in scores.values())


def test_copy_fails_spatial_control(world):
    task,labels=packages("copy-test","dev",0)[0]
    result,_=run_task(task,world,CopyModel(),labels=labels)
    assert result["reports"]["main"]["metrics"]["overlap"]["mae"]>.2


def test_fork_isolates_mutable_state(world):
    initial=crop(world,(0,0),ORIGINS[0]);model=WindowMap();model.reset(initial,PUBLIC_SPEC,2)
    state=model.snapshot();before=state["map"].copy();model.generate([[480,0]])
    assert np.array_equal(state["map"],before)
    model.restore(state);model.map[:]=0
    assert np.array_equal(state["map"],before)


def test_random_replay(world):
    model=RandomModel();model.reset(crop(world,(0,0),ORIGINS[0]),PUBLIC_SPEC,7);state=model.snapshot()
    a=model.generate([[0,0]])[0];model.restore(state);b=model.generate([[0,0]])[0]
    assert np.array_equal(a,b)


def test_invalid_batch_keeps_slots():
    a,state=validate_response([],16);assert len(a)==16 and all(x is None for x in a)
    a,state=validate_response([np.zeros((240,416,3),np.float32)],1);assert a==[None]


def test_hash_chain_detects_edits(world):
    task,labels=packages("hash-test","dev",0)[0]
    _,journal=run_task(task,world,CopyModel(),labels=labels)
    rows=copy.deepcopy(journal.rows);rows[1]["changed"]=True
    with pytest.raises(ValueError):verify_journal(rows)


def test_failed_initialization_keeps_full_denominator(world):
    class Broken(CopyModel):
        def reset(self,*args):raise RuntimeError("test failure")
    task,labels=packages("broken-test","dev",0)[3]
    result,_=run_task(task,world,Broken(),labels=labels)
    assert result["reports"]["planned_outputs"]==160 and result["reports"]["valid_outputs"]==0
    assert result["reports"]["paired_reentry_mae"] is None
    assert result["reports"]["metrics"]["reentry"]["expected_pixels"]==2*128*128


def test_public_spec_excludes_private_geometry():
    assert not {"world_id","city","origin","target","labels","revision"}.intersection(PUBLIC_SPEC)


def test_private_labels_cannot_be_moved_to_later_reentry():
    from rswm_inria.tasks import validate_task
    task,labels=packages("tamper-geometry","dev",0)[2]
    bad=copy.deepcopy(labels);bad["main"]["reentry_event"][:]=96
    with pytest.raises(ValueError):validate_task(task,bad)


def test_public_protocol_and_pair_validation():
    from rswm_inria.tasks import validate_task
    task,_=packages("tamper-protocol","dev",0)[1]
    task["programs"]["large"][5][0]+=1
    with pytest.raises(ValueError):validate_task(task)
