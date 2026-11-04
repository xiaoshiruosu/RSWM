"""Current prepared conditions and the actual supplied sampler, software scope.

The original function body is executed with explicit expensive-component doubles.
None of these tests validates original weights, neural camera layer consumption,
GPU resources, or the real-model experiment gate.
"""
import ast
import copy
import hashlib
import importlib.util
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rswm_inria.adapters.camera_conditioning import (
    CameraConfig, backend_payload, build_camera_batch, plucker_at_pixels,
)
from rswm_inria.geometry import digest
from rswm_inria.legacy_camera import camera_condition, validate_prepared_camera

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_sampler.py"
SCRIPT = Path(__file__).parents[1] / "scripts" / "prepare_legacy_wan.py"
spec = importlib.util.spec_from_file_location("prepare_legacy_wan_test", SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def payload(frames=17, channel="native", scaling="fixed_units", framewise=False, scale=1):
    # A turn occurs before an endpoint; the selected actual events remain frozen.
    count = frames - 1
    q = np.array([[min(i, count//2)*3, max(0, i-count//2)*2] for i in range(1, frames)], dtype=np.int64)*scale
    return backend_payload(build_camera_batch(q, config=CameraConfig(scaling, framewise, channel, 4)))


def cfg():
    return SimpleNamespace(vae_stride=(4,8,8), patch_size=(1,2,2))


@pytest.mark.parametrize("frames", [17,33])
@pytest.mark.parametrize("channel", ["native", "identity", "drop"])
def test_prepared_exact_events_and_drop(frames, channel):
    p = payload(frames, channel)
    assert validate_prepared_camera(p, frames, cfg(), 480,832,60,104) == channel
    assert list(p["selected_event_indices"]) == list(range(0, frames,4))
    if channel == "drop":
        assert camera_condition(p,frames,cfg(),480,832,60,104,"cpu") is None
        assert p["camera_poses"] is p["intrinsics_fx_fy_cx_cy"] is None


@pytest.mark.parametrize("mutation", ["extra", "indices", "float_indices", "hash", "manifest", "float32", "nan", "anchor", "rotation", "intrinsics", "huge"])
def test_prepared_corruption_is_rejected(mutation):
    p = copy.deepcopy(payload())
    # Existing payload arrays are immutable; corrupt independent copies.
    p["camera_poses"] = np.array(p["camera_poses"])
    p["intrinsics_fx_fy_cx_cy"] = np.array(p["intrinsics_fx_fy_cx_cy"])
    if mutation == "extra": p["q"] = np.zeros((16,2))
    elif mutation == "indices": p["selected_event_indices"] = np.array([0,4,8,13,16])
    elif mutation == "float_indices": p["selected_event_indices"] = np.array([0,4,8,12,16], dtype=float)
    elif mutation == "hash": p["camera_manifest_sha256"] = "0"*64
    elif mutation == "manifest":
        p["camera_manifest"]["backend_validated"] = True
        p["camera_manifest_sha256"] = digest(p["camera_manifest"])
    elif mutation == "float32": p["camera_poses"] = p["camera_poses"].astype(np.float32)
    elif mutation == "nan": p["camera_poses"][1,0,3] = np.nan
    elif mutation == "anchor": p["camera_poses"][0,0,3] = .1
    elif mutation == "rotation": p["camera_poses"][1,0,0] = 0
    elif mutation == "intrinsics": p["intrinsics_fx_fy_cx_cy"][1,0] *= 2
    elif mutation == "huge": p["camera_poses"][1,0,3] = 1e300
    with pytest.raises(ValueError): validate_prepared_camera(p,17,cfg(),480,832,60,104)


@pytest.mark.parametrize("frames,settings", [(16,cfg()), (True,cfg()), (17,SimpleNamespace(vae_stride=(8,8,8),patch_size=(1,2,2))), (17,SimpleNamespace(vae_stride=(4,8,8),patch_size=(2,2,2))), (17,SimpleNamespace(vae_stride=None,patch_size=(1,2,2))), (17,SimpleNamespace(vae_stride=(4,8,8),patch_size=(True,2,2)))])
def test_native_contract_does_not_clamp_or_guess(frames, settings):
    with pytest.raises(ValueError): validate_prepared_camera(payload(),frames,settings,480,832,60,104)


def test_drop_and_identity_cannot_carry_second_channel():
    drop = payload(channel="drop")
    drop["camera_poses"] = payload()["camera_poses"]
    with pytest.raises(ValueError): validate_prepared_camera(drop,17,cfg(),480,832,60,104)
    identity = payload(channel="identity")
    identity["camera_poses"] = payload()["camera_poses"]
    with pytest.raises(ValueError): validate_prepared_camera(identity,17,cfg(),480,832,60,104)


@pytest.fixture
def torch_cpu():
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield torch
    torch.set_num_threads(previous)


def unpack_sparse(tensor, pixels):
    return np.array([[[tensor[0,c*64+(y%8)*8+x%8,f,y//8,x//8].item() for c in range(6)]
                      for x,y in pixels] for f in range(tensor.shape[2])], dtype=np.float32)


@pytest.mark.parametrize("framewise", [False,True])
@pytest.mark.parametrize("scaling", ["fixed_units", "legacy_maxnorm"])
def test_actual_torch_rays_match_original_convention_without_second_normalization(torch_cpu, framewise, scaling):
    c = cfg(); c.param_dtype = torch_cpu.float32
    pixels = [(0,0), (415,239), (416,240), (831,479)]
    p, twice = payload(17,scaling=scaling,framewise=framewise), payload(17,scaling=scaling,framewise=framewise,scale=2)
    a = camera_condition(p,17,c,480,832,60,104,"cpu")["c2ws_plucker_emb"][0]
    b = camera_condition(twice,17,c,480,832,60,104,"cpu")["c2ws_plucker_emb"][0]
    assert tuple(a.shape) == (1,384,5,60,104)
    expected = plucker_at_pixels(p["camera_poses"], p["intrinsics_fx_fy_cx_cy"], np.array(pixels)+.5)
    actual = unpack_sparse(a,pixels)
    np.testing.assert_allclose(actual, expected, rtol=3e-6,atol=1e-7)
    np.testing.assert_allclose(unpack_sparse(b,pixels)[:,:,3:], actual[:,:,3:]*(2 if scaling=="fixed_units" else 1), rtol=3e-6,atol=1e-7)
    assert not p["camera_manifest"]["backend_validated"]


def original_function(name):
    source = FIXTURE.read_text()
    return prepare._function(source,name)[1]


def test_actual_function_fixture_identity_and_patches():
    assert hashlib.sha256(original_function("generate_video").encode()).hexdigest() == "74c905c7ec6ddc76aceb6386d4beb79d47d5fe6189d119799d4aa00d1a553453"
    assert "compute_relative_poses" in original_function("generate_video")
    assert "prepared_camera" not in original_function("generate_video")
    assert "prepared_camera" in prepare.patch_sampler(FIXTURE.read_text())


def test_actual_original_grid_expression_exposes_floating_floor_bug():
    # This is why the prepared-only fixed native grid is necessary. The old
    # action/environment path retains the actual supplied expression.
    aspect=480/832
    assert np.sqrt((480*832)*aspect)//8//2*2 == 58
    assert "np.sqrt(max_area * aspect_ratio) // vae_stride[1]" in prepare.patch_sampler(FIXTURE.read_text())


def sampler_doubles(torch, frames):
    calls = {"texts":[], "encode":[], "decode":[], "steps":[], "experts":[], "moves":[]}
    class Text:
        def __call__(self, prompts, device):
            calls["texts"].append((prompts,str(device)))
            return [torch.zeros(2,4,device=device)]
    class VAE:
        def encode(self, videos):
            calls["encode"].append(tuple(videos[0].shape))
            return [torch.zeros(16,(frames-1)//4+1,60,104)]
        def decode(self, latents):
            calls["decode"].append(tuple(latents[0].shape))
            return [torch.zeros(1).expand(3,frames,480,832)]
    class DiT:
        def __init__(self,tag): self.tag=tag
        def cpu(self): calls["moves"].append((self.tag,"cpu")); return self
        def to(self,device): calls["moves"].append((self.tag,str(device))); return self
        def __call__(self, x, **kwargs):
            calls["experts"].append((self.tag, kwargs))
            return [torch.zeros_like(x[0])]
    class Scheduler:
        def __init__(self,**kwargs): calls["scheduler"] = kwargs
        def set_timesteps(self,steps,device,shift):
            calls["schedule"]=(steps,str(device),shift)
            self.timesteps=torch.tensor([999.,0.],device=device)
        def step(self,pred,t,latent,**kwargs):
            calls["steps"].append((t.item(),tuple(pred.shape),kwargs["return_dict"]))
            return (latent,)
    def forbidden(*args,**kwargs): raise AssertionError("Prepared camera entered legacy preprocessing or pose files")
    globals_ = {"torch":torch,"np":np,"os":os,"logger":logging.getLogger(__name__), "Image":Image,
                "MAX_AREA_CONFIGS":{"480*832":480*832}, "TF":SimpleNamespace(to_tensor=lambda image:torch.zeros(3,image.height,image.width)),
                "tqdm":lambda items,**kwargs:items,"FlowUniPCMultistepScheduler":Scheduler,
                "interpolate_camera_poses":forbidden,"compute_relative_poses":forbidden,
                "get_Ks_transformed":forbidden,"get_plucker_embeddings":forbidden}
    patched = prepare.patch_sampler(FIXTURE.read_text())
    fn = prepare._function(patched,"generate_video")[0]
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(FIXTURE),"exec"),globals_)
    c=cfg(); c.param_dtype=torch.float32; c.boundary=.947; c.num_train_timesteps=1000; c.sample_neg_prompt="negative"
    return globals_["generate_video"], calls, (Text(),VAE(),DiT("low"),DiT("high"),c), forbidden


@pytest.mark.filterwarnings("ignore:User provided device_type of 'cuda':UserWarning")
@pytest.mark.parametrize("frames", [17,33])
@pytest.mark.parametrize("channel", ["native", "identity", "drop"])
def test_patched_original_sampler_complete_loop_uses_only_prepared_conditions(torch_cpu,monkeypatch,frames,channel):
    generate,calls,models,forbidden = sampler_doubles(torch_cpu,frames)
    monkeypatch.setenv("RSWM_POSE_MODE", "drop" if channel=="native" else "invalid-do-not-read")
    monkeypatch.setattr(np,"load",forbidden)
    result=generate(*models,prompt="selected prompt",image=Image.new("RGB",(832,480)),device=torch_cpu.device("cpu"),
                    frame_num=frames,sampling_steps=2,prepared_camera=payload(frames,channel))
    assert tuple(result.shape)==(3,frames,480,832)
    assert calls["encode"]==[(3,frames,480,832)]
    assert calls["decode"]==[(16,(frames-1)//4+1,60,104)]
    assert [x[0] for x in calls["experts"]]==["high","high","low","low"]
    assert [x[0] for x in calls["steps"]]==[999.,0.]
    assert len(calls["texts"])==2
    conditions=[x[1]["dit_cond_dict"] for x in calls["experts"]]
    assert all(value is conditions[0] for value in conditions)
    assert all(set(x[1])=={"t","context","seq_len","y","dit_cond_dict"} for x in calls["experts"])
    if channel=="drop": assert conditions==[None]*4
    else:
        tensor=conditions[0]["c2ws_plucker_emb"][0]
        assert tuple(tensor.shape)==(1,384,(frames-1)//4+1,60,104)
        assert bool(torch_cpu.count_nonzero(tensor[:,192:])) is (channel=="native")


def test_prepared_and_action_path_fail_before_any_sampling(torch_cpu):
    generate,calls,models,_ = sampler_doubles(torch_cpu,17)
    with pytest.raises(ValueError,match="mutually exclusive"):
        generate(*models,prompt="p",image=Image.new("RGB",(832,480)),device=torch_cpu.device("cpu"),
                 action_path="must-not-read",prepared_camera=payload())
    assert not calls["texts"] and not calls["experts"]


@pytest.mark.parametrize("size,hw", [("832*480",(480,832)),("480*832",(464,832))])
def test_prepared_sampler_rejects_wrong_native_input_and_size_before_encoding(torch_cpu,size,hw):
    generate,calls,models,_=sampler_doubles(torch_cpu,17)
    with pytest.raises(ValueError,match="exact native480x832"):
        generate(*models,prompt="p",image=Image.new("RGB",(hw[1],hw[0])),device=torch_cpu.device("cpu"),
                 frame_num=17,size=size,prepared_camera=payload())
    assert not calls["texts"] and not calls["encode"] and not calls["experts"]
