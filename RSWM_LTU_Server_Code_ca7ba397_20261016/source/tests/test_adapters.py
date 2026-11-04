from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from rswm_inria.baselines import WindowMap,DiffusersCompleter
from rswm_inria.runner import PUBLIC_SPEC,run_task
from rswm_inria.geometry import rng_for,crop,ORIGINS
from rswm_inria.tasks import packages


def test_same_window_eager_lazy_hard_preservation():
    class Spy:
        def __init__(self):self.shapes=[]
        def __call__(self,image,known,seed):
            self.shapes.append(image.shape);return np.full(image.shape,200,np.uint8)
    initial=np.full((240,416,3),13,np.uint8)
    for eager in (False,True):
        spy=Spy();model=WindowMap(spy,eager);model.reset(initial,PUBLIC_SPEC,0)
        assert np.array_equal(model.generate([[0,0]])[0],initial)
        a=model.generate([[480,0]])[0];b=model.generate([[480,0]])[0]
        assert np.array_equal(a,b) and set(spy.shapes)=={(512,512,3)}


def test_both_branches_same_wrong_content_is_not_success():
    class Repaint:
        def reset(self,*args):self.calls=0
        def generate(self,queries):
            self.calls+=1
            return [np.full((240,416,3),0 if self.calls<=2 else 255,np.uint8) for _ in queries]
        def snapshot(self):return self.calls
        def restore(self,state):self.calls=state
    task,labels=packages("repaint-test","dev",0)[3]
    result,_=run_task(task,np.zeros((2048,2048,3),np.uint8),Repaint(),labels=labels)
    assert result["reports"]["paired_reentry_mae"]==1.


def test_wan_mock_context_count_and_query_independence():
    pytest.importorskip("torch")
    from rswm_inria.video import WanI2VAdapter
    class FakePipe:
        def __init__(self):self.args=[]
        def __call__(self,**kwargs):
            self.args.append(kwargs)
            return SimpleNamespace(frames=[[Image.new("RGB",(832,480),(17,23,31)) for _ in range(kwargs["num_frames"])]])
    pipe=FakePipe();model=WanI2VAdapter(device="cpu",pipe=pipe)
    model.reset(np.zeros((240,416,3),np.uint8),PUBLIC_SPEC,0);state=model.snapshot()
    a=model.generate([[0,0]]*16);model.restore(state);b=model.generate([[99,55]]*16)
    assert len(a)==16 and all(x.shape==(240,416,3) for x in a)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    assert pipe.args[0]["num_frames"]==17 and "queries" not in pipe.args[0]
    assert pipe.args[0]["generator"].initial_seed()==pipe.args[1]["generator"].initial_seed()


def test_diffusers_mock_mask_semantics_and_known_protection():
    pytest.importorskip("torch")
    class FakePipe:
        def __call__(self,**kwargs):
            assert np.asarray(kwargs["mask_image"])[0,0]==0
            assert np.asarray(kwargs["mask_image"])[1,1]==255
            return SimpleNamespace(images=[Image.new("RGB",(512,512),(255,255,255))])
    model=DiffusersCompleter.__new__(DiffusersCompleter)
    model.pipe=FakePipe();model.device="cpu";model.steps=1;model.prompt="test"
    image=np.zeros((512,512,3),np.uint8);known=np.zeros((512,512),bool);known[0,0]=True
    out=model(image,known,0)
    assert (out[0,0]==0).all() and (out[1,1]==255).all()
