import numpy as np
import pytest
pytest.importorskip("scipy")
from rswm_inria.motion import estimate_translation,diagnose_sequence,paired_path_ratio
from rswm_inria.geometry import rng_for,crop,ORIGINS


def test_motion_sign_and_magnitude():
    world=rng_for("motion-tests").integers(0,256,(2048,2048,3),dtype=np.uint8)
    a=crop(world,(0,0),ORIGINS[0]);b=crop(world,(24,-16),ORIGINS[0])
    estimate=estimate_translation(a,b)
    assert estimate["status"]=="measured" and estimate["delta"]==[24,-16]


def test_flat_and_missing_abstain():
    a=np.full((80,120,3),128,np.uint8)
    assert estimate_translation(a,a)["status"]=="insufficient_evidence"
    assert estimate_translation(a,None)["status"]=="submission_failure"


def test_periodic_texture_ambiguous():
    yy,xx=np.mgrid[:80,:120];a=np.repeat((((xx//4+yy//4)%2)*255).astype(np.uint8)[:,:,None],3,axis=2)
    assert estimate_translation(a,a)["status"]=="ambiguous"


def test_chain_does_not_reanchor():
    world=rng_for("motion-chain").integers(0,256,(2048,2048,3),dtype=np.uint8)
    q=[[0,0],[16,0],[32,0],[48,0]]
    images=[crop(world,p,ORIGINS[0]) for p in q];images[1]=None
    result=diagnose_sequence(images,q)
    assert result["anchored_prefix_edges"]==0 and result["normalized_endpoint_error"] is None
    assert result["measured_edges"]==1 and not result["full_path_measurable"]
    assert paired_path_ratio(result,result)["large_over_small"] is None
