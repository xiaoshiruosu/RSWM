"""Software boundary tests only; no original pretrained weights or GPU used."""
import copy
import json
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from rswm_inria.adapters.camera_conditioning import CameraConfig, backend_payload, build_camera_batch
from rswm_inria.geometry import file_hash
from rswm_inria.legacy_wan import (LegacyWanBackend, component_inventory, inspect_backend,
                                  load_backend, strict_from_pretrained, strict_lora_from_pretrained, validate_backend_config)
from rswm_inria.numeric import payload_hash
from rswm_inria.training import locked_tree


def config():
    return {"schema": "rswm-legacy-wan-backend-v1", "mode": "base", "config_key": "i2v-A14B",
            "sampling": {"prompt": "fixed software control text", "negative_prompt": "fixed negative text",
                         "steps": 2, "shift": 10., "guide_scale": [5., 5.], "boundary": .947,
                         "param_dtype": "bf16", "t5_dtype": "bf16", "t5_cpu": True,
                         "inactive_model_cpu": True}, "lora_subdir": None, "training": None}


def source(tmp_path):
    root = tmp_path / "source"; root.mkdir()
    names = ["inference.py", "base_loader.py", "wan/__init__.py", "wan/modules/model.py",
             "wan/modules/vae2_1.py", "wan/modules/t5.py", "wan/utils/cam_utils.py", "wan/configs/wan_i2v_A14B.py"]
    for name in names:
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("# SOFTWARE TEST DOUBLE ONLY\n")
    root.joinpath("source_inventory.json").write_text(json.dumps({n: file_hash(root / n) for n in names}))
    root.joinpath("backend_config.json").write_text(json.dumps(config()))
    root.joinpath("backend.py").write_text("# Test-owned entrypoint, not real vendor loading\n")
    return root


def weights(tmp_path):
    root = tmp_path / "weights"; root.mkdir()
    for name in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "google/umt5-xxl/tokenizer_config.json",
                 "google/umt5-xxl/tokenizer.json"):
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"synthetic component inventory only")
    for tag in ("low", "high"):
        d = root / (tag + "_noise_model"); d.mkdir()
        d.joinpath("config.json").write_text(json.dumps({"model_type": "i2v", "in_dim": 36}))
        d.joinpath("diffusion_pytorch_model.safetensors").write_bytes(b"synthetic, not actual weights")
    return root


def test_locked_configuration_has_no_implicit_sampling_or_lora_fallback():
    value = config(); returned = validate_backend_config(value)
    returned["sampling"]["prompt"] = "changed"
    assert value["sampling"]["prompt"] == "fixed software control text"
    for key in ("prompt", "steps", "inactive_model_cpu", "boundary"):
        broken = config(); del broken["sampling"][key]
        with pytest.raises(ValueError, match="Complete locked sampling"):
            validate_backend_config(broken)
    value["lora_subdir"] = "adapter"
    with pytest.raises(ValueError, match="Base mode"):
        validate_backend_config(value)


@pytest.mark.parametrize("name,value", [("steps", True), ("shift", float("nan")), ("boundary", 1.),
                                       ("guide_scale", [float("inf"), 5.]), ("t5_cpu", 1), ("param_dtype", "fp32")])
def test_nonfinite_or_unrecorded_original_settings_rejected(name, value):
    c = config(); c["sampling"][name] = value
    with pytest.raises(ValueError): validate_backend_config(c)


def test_actual_component_inventory_is_not_a_weight_or_forward_acceptance(tmp_path):
    root = weights(tmp_path); receipt = component_inventory(root, config(), require_complete=True)
    assert receipt["complete"] and len(receipt["components"]) == 5
    assert all(f["sha256"] == file_hash(root/f["path"]) for row in receipt["components"] for f in row["files"])
    assert "not pretrained forward" in receipt["scope"]
    (root/"high_noise_model"/"diffusion_pytorch_model.safetensors").unlink()
    with pytest.raises(ValueError, match="high_expert"):
        component_inventory(root, config(), require_complete=True)


def test_actual_index_shards_and_two_lora_components_are_required(tmp_path):
    root = weights(tmp_path)
    index = root/"low_noise_model"/"diffusion_pytorch_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"camera.weight": "missing-shard.safetensors"}}))
    assert not component_inventory(root, config())["complete"]
    index.write_text(json.dumps({"weight_map": {"camera.weight": "../escape.safetensors"}}))
    with pytest.raises(ValueError, match="relative"):
        component_inventory(root, config())
    index.unlink()
    c = config(); c.update(mode="restored_lora", lora_subdir="old_adapter")
    with pytest.raises(ValueError, match="low_noise_lora, high_noise_lora"):
        component_inventory(root, c, require_complete=True)


def test_missing_weights_fail_before_torch_or_any_generation(tmp_path, monkeypatch):
    root = source(tmp_path)
    import rswm_inria.legacy_wan as bridge
    monkeypatch.setattr(bridge.importlib, "import_module", lambda *a, **k: pytest.fail("No expensive imports for missing components"))
    with pytest.raises(ValueError, match="actual local Wan components missing"):
        load_backend(source_dir=root, weights_dir=tmp_path/"absent", device="cuda")
    result = inspect_backend(source_dir=root)
    assert result["real_model_validation"] == "NOT_RUN" and result["components"] is None


def test_changed_source_and_foreign_cached_vendor_cannot_be_loaded(tmp_path, monkeypatch):
    root = source(tmp_path); w = weights(tmp_path)
    other = tmp_path/"foreign.py"; other.write_text("# external installed vendor\n")
    monkeypatch.setitem(sys.modules, "wan", SimpleNamespace(__file__=str(other)))
    with pytest.raises(ValueError, match="escaped"):
        load_backend(source_dir=root, weights_dir=w, device="cuda")
    monkeypatch.delitem(sys.modules, "wan")
    (root/"inference.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source changed"):
        inspect_backend(source_dir=root)


def test_same_path_unowned_module_cache_is_not_locked_current_source(tmp_path, monkeypatch):
    import rswm_inria.legacy_wan as bridge
    root = source(tmp_path); w = weights(tmp_path)
    monkeypatch.setattr(bridge, "_SOURCE_LOCK", None)
    monkeypatch.setitem(sys.modules, "wan", SimpleNamespace(__file__=str(root/"wan"/"__init__.py")))
    with pytest.raises(ValueError, match="unowned original module cache"):
        load_backend(source_dir=root, weights_dir=w, device="cuda")


def test_nested_component_names_cannot_replace_actual_loader_paths(tmp_path):
    root = weights(tmp_path)
    p=root/"google"/"umt5-xxl"/"tokenizer_config.json";nested=p.parent/"nested";nested.mkdir();p.rename(nested/p.name)
    with pytest.raises(ValueError, match="tokenizer"):
        component_inventory(root, config(), require_complete=True)


def test_actual_sampler_cuda_contract_does_not_turn_cpu_into_noise(tmp_path):
    root = source(tmp_path); w = weights(tmp_path)
    with pytest.raises(ValueError, match="sampler requires CUDA"):
        load_backend(source_dir=root, weights_dir=w, device="cpu")


def test_loading_transaction_rechecks_actual_retained_weight_bytes(tmp_path):
    import rswm_inria.legacy_wan as bridge
    root = source(tmp_path); w = weights(tmp_path)
    root, tree, value = bridge._source(root)
    components = component_inventory(w, value, require_complete=True)
    bridge._confirm_loaded_inputs(root, tree, value, w, components)
    def original_loader_double():
        original = (w/"Wan2.1_VAE.pth").read_bytes()
        (w/"Wan2.1_VAE.pth").write_bytes(original + b" replaced during loading")
    original_loader_double()
    with pytest.raises(ValueError, match="weight components changed during"):
        bridge._confirm_loaded_inputs(root, tree, value, w, components)


def architecture():
    return SimpleNamespace(model_type="i2v", in_dim=36, patch_size=(1,2,2), dim=5120,
        ffn_dim=13824, freq_dim=256, num_heads=40, num_layers=40, text_len=512, out_dim=16,
        patch_embedding_wancamctrl=SimpleNamespace(in_features=1536, out_features=5120),
        blocks=[SimpleNamespace(**{k: lambda x: x for k in ("cam_injector_layer1", "cam_injector_layer2", "cam_scale_layer", "cam_shift_layer")}) for _ in range(40)])


@pytest.mark.parametrize("failure", ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"])
def test_original_loader_warnings_cannot_random_initialize_missing_camera_weights(failure):
    info = {k: [] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}; info[failure] = ["camera.weight"]
    class OriginalLoader:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs["local_files_only"] is True and kwargs["output_loading_info"] is True
            return architecture(), info
    with pytest.raises(ValueError, match="loading incomplete"):
        strict_from_pretrained(OriginalLoader, "explicit local test directory")
    info[failure] = []
    assert strict_from_pretrained(OriginalLoader, "explicit local test directory").model_type == "i2v"


def test_loaded_camera_architecture_and_loading_information_are_mandatory():
    class Loader:
        @classmethod
        def from_pretrained(cls, *a, **k): return architecture()
    with pytest.raises(ValueError, match="loading information"): strict_from_pretrained(Loader, "local")
    model = architecture(); model.patch_embedding_wancamctrl.in_features = 6
    Loader.from_pretrained = lambda *a, **k: (model, {x: [] for x in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")})
    with pytest.raises(ValueError, match="camera projector"): strict_from_pretrained(Loader, "local")


@pytest.mark.parametrize("failure", [None, "missing", "unexpected", "shape", "nonfinite", "random_value"])
def test_restored_lora_loader_cannot_silently_keep_random_or_extra_adapter_tensors(tmp_path, monkeypatch, failure):
    torch = pytest.importorskip("torch")
    import rswm_inria.legacy_wan as bridge
    root = tmp_path/"lora"; root.mkdir()
    root.joinpath("adapter_config.json").write_text("{}")
    root.joinpath("adapter_model.safetensors").write_bytes(b"software loader fixture only")
    retained = {"original.camera.lora_A.weight": torch.ones(2, 2)}
    actual = {key: value.clone() for key, value in retained.items()}
    if failure == "missing": actual.clear()
    elif failure == "unexpected": actual["unexpected.weight"] = torch.ones(1)
    elif failure == "shape": actual[next(iter(actual))] = torch.ones(3, 2)
    elif failure == "nonfinite": actual[next(iter(actual))][0, 0] = float("nan")
    elif failure == "random_value": actual[next(iter(actual))].zero_()
    loaded = object()
    class OriginalPeftLoader:
        @classmethod
        def from_pretrained(cls, model, path, **kwargs):
            assert path == str(root.resolve()) and kwargs["local_files_only"] is True
            return loaded
    def retained_weights(path, **kwargs):
        assert kwargs == {"device": "cpu", "local_files_only": True}
        return retained
    def actual_weights(model, **kwargs):
        assert model is loaded and kwargs == {"adapter_name": "default", "save_embedding_layers": False}
        return actual
    helper = SimpleNamespace(load_peft_weights=retained_weights, get_peft_model_state_dict=actual_weights)
    real_import = bridge.importlib.import_module
    monkeypatch.setattr(bridge.importlib, "import_module", lambda name: helper if name == "peft.utils.save_and_load" else real_import(name))
    if failure is None:
        assert strict_lora_from_pretrained(OriginalPeftLoader, object(), root, is_trainable=False) is loaded
    else:
        with pytest.raises(ValueError, match="LoRA"):
            strict_lora_from_pretrained(OriginalPeftLoader, object(), root, is_trainable=False)


@pytest.fixture
def cpu_backend(tmp_path):
    torch = pytest.importorskip("torch")
    old_threads = torch.get_num_threads(); torch.set_num_threads(1)
    class CacheModel(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(()), requires_grad=False); self.clear_cache()
        def clear_cache(self): self._conv_idx = [0]; self._feat_map = [None]; self._enc_conv_idx = [0]; self._enc_feat_map = [None]
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in ("cam_injector_layer1", "cam_injector_layer2", "cam_scale_layer", "cam_shift_layer"):
                setattr(self, name, torch.nn.Linear(1, 1))
    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.patch_embedding_wancamctrl = torch.nn.Linear(1, 1); self.blocks = torch.nn.ModuleList([Block()]); self.freqs = torch.ones(2); self.register_buffer("entered", torch.zeros((),dtype=torch.int64))
        def forward(self, x, *, camera):
            self.entered += 1
            if camera is not None:
                value = self.patch_embedding_wancamctrl(camera)
                block = self.blocks[0]; value = block.cam_injector_layer1(value); block.cam_scale_layer(value); block.cam_shift_layer(value)
            return [x[0]]
    vae = SimpleNamespace(model=CacheModel(), mean=torch.ones(1), std=torch.ones(1), scale=[torch.ones(1),torch.ones(1)], device=torch.device("cpu"), dtype=torch.float32)
    text = SimpleNamespace(model=CacheModel()); requests = []; failure = [False]
    def original_sampler_double(**request):
        requests.append(request)
        c = request["prepared_camera"]; channel = c["camera_manifest"]["camera_channel"]
        value = None if channel == "drop" else torch.tensor(np.array(c["camera_poses"][:,:3,3].reshape(-1,1),copy=True),dtype=torch.float32)
        latent = torch.zeros(16,(request["frame_num"]-1)//4+1,1,1)
        request["low_noise_model"]([latent],camera=value)
        if failure[0]: raise RuntimeError("software failure after actual entered forward")
        pixel = (random.randrange(16)+np.random.randint(0,16)+int(torch.randint(0,16,()).item())+int(request["low_noise_model"].entered))%32
        vae.model._feat_map[0] = torch.tensor([pixel])
        return torch.full((3,1,1,1),pixel/32).expand(3,request["frame_num"],480,832)
    root=tmp_path/"loaded-source"; root.mkdir(); root.joinpath("source.txt").write_text("software-only source state\n")
    backend=LegacyWanBackend(parts=(text,vae,Expert(),Expert()),inference=SimpleNamespace(generate_video=original_sampler_double),
        wan_config=SimpleNamespace(vae_stride=(4,8,8),patch_size=(1,2,2),param_dtype=torch.bfloat16), runtime_config=config(),
        source_dir=root,source_tree=locked_tree(root),components={"weights_dir":str(tmp_path),"weights_tree_sha256":"0"*64},torch=torch,device=torch.device("cpu"))
    yield backend, requests, failure
    torch.set_num_threads(old_threads)


def camera(channel="native", n=17):
    queries=np.column_stack([np.arange(1,n)*4,np.zeros(n-1,dtype=np.int64)])
    return backend_payload(build_camera_batch(queries,config=CameraConfig(camera_channel=channel)))


@pytest.mark.parametrize("size", [0, 1, 2, 127, 128, 129, 16777220, 21565440])
def test_camera_summary_samples_large_tensor_without_float32_endpoint_rounding(size):
    torch = pytest.importorskip("torch")
    backend = SimpleNamespace(torch=torch)
    value = torch.arange(size, dtype=torch.float32)
    count = min(128, size)
    expected = [i*(size-1)//(count-1) for i in range(count)] if count > 1 else list(range(count))
    if size > 2**24:
        old = torch.linspace(0, size-1, count).long()
        assert old[-1].item() == size
        with pytest.raises(IndexError):
            value[old]
    result = LegacyWanBackend._tensor_summary(backend, value)
    assert result["sample_indices"] == expected
    assert all(0 <= i < size for i in result["sample_indices"])
    assert result["sample_values"] == value[expected].tolist()
    assert result["shape"] == [size] and result["finite"] is True


def test_camera_summary_keeps_noncontiguous_values_and_rejects_nonfinite_input():
    torch = pytest.importorskip("torch")
    backend = SimpleNamespace(torch=torch)
    value = torch.arange(15, dtype=torch.bfloat16).reshape(3, 5).transpose(0, 1)
    result = LegacyWanBackend._tensor_summary(backend, value)
    assert result["shape"] == [5, 3] and result["sample_indices"] == list(range(15))
    assert result["sample_values"] == value.reshape(-1).float().tolist()
    value[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite actual camera layer input"):
        LegacyWanBackend._tensor_summary(backend, value)


@pytest.mark.parametrize("channel", ["native", "identity", "drop"])
@pytest.mark.parametrize("native_frames", [17,33])
def test_real_function_boundary_native_frames_settings_hooks_and_ablations_are_explicit(cpu_backend, channel, native_frames, monkeypatch):
    backend,requests,_=cpu_backend; backend.reset(seed=7)
    monkeypatch.setenv("RSWM_POSE_MODE","an environment value cannot choose this branch")
    c=camera(channel,native_frames);result=backend.generate(image=np.zeros((480,832,3),np.uint8),camera=c,native_frames=native_frames,seed=9)
    assert len(result["frames"])==native_frames and all(frame.mode=="RGB" and frame.size==(832,480) for frame in result["frames"])
    assert result["camera_payload_sha256"]==payload_hash(c)
    assert set(requests[0])=={"text_encoder","vae","low_noise_model","high_noise_model","config","prompt","image","device","frame_num","size","action_path","seed","sampling_steps","shift","guide_scale","t5_cpu","framewise","inactive_model_cpu","prepared_camera"}
    assert requests[0]["prepared_camera"] is c and requests[0]["action_path"] is None
    observation=result["conditioning_observation"]
    assert observation["scope"]=="software_test_double" and observation["real_backend_layer_consumption"]=="NOT_RUN"
    assert bool(observation["records"])==(channel!="drop")
    assert backend.resource_counters()["values"]["native_frames"]==native_frames


def test_f7_clones_mutable_state_rng_and_never_rolls_back_cost(cpu_backend):
    backend,_,_=cpu_backend; backend.reset(seed=17); saved=backend.snapshot()
    pointers={n:p.data_ptr() for n,p in backend.model.named_parameters()}; c=camera();image=np.zeros((480,832,3),np.uint8)
    a=backend.generate(image=image,camera=c,native_frames=17,seed=5)
    backend.restore(saved)
    assert backend.vae.model._feat_map[0] is None
    b=backend.generate(image=image,camera=c,native_frames=17,seed=5)
    assert np.array_equal(np.asarray(a["frames"][-1]),np.asarray(b["frames"][-1]))
    assert pointers=={n:p.data_ptr() for n,p in backend.model.named_parameters()}
    assert backend.resource_counters()["values"]["native_frames"]==34
    evidence=backend.generation_evidence()
    assert [r["call_id"] for r in evidence["observations"]]==[1,2]
    assert "observations" not in saved and all(r["status"]=="SUCCEEDED" for r in evidence["observations"])
    assert set(saved["mutable"]["placements"])=={"low","high","vae","text"}
    assert saved["mutable"]["buffers"]["low"]["entered"].data_ptr()!=backend.model["low"].entered.data_ptr()
    backend.vae.model._feat_map[0][0]=999
    assert saved["mutable"]["vae_cache"]["_feat_map"][0] is None


def test_work_before_sampler_failure_is_retained_after_restore(cpu_backend):
    backend,_,failure=cpu_backend;backend.reset(seed=0);state=backend.snapshot();failure[0]=True
    with pytest.raises(RuntimeError,match="after actual entered"):
        backend.generate(image=np.zeros((480,832,3),np.uint8),camera=camera(),native_frames=17,seed=0)
    assert backend.counters["native_frames"]==17 and backend.counters["decoded_frames"]==0
    backend.restore(state)
    assert backend.counters["native_frames"]==17 and backend.counters["failed_generate_calls"]==1
    assert not backend._busy and not backend.model["low"]._forward_pre_hooks
    record=backend.generation_evidence()["observations"][0]
    assert record["status"]=="FAILED" and record["error_type"]=="RuntimeError"
    assert record["records"] and record["counters_after"]["native_frames"]==17


def test_changed_loaded_source_extra_camera_channel_and_trainable_snapshot_rejected(cpu_backend):
    backend,requests,_=cpu_backend;c=camera();c["private_world"]=np.zeros((4,4))
    with pytest.raises(ValueError,match="public payload"):
        backend.generate(image=np.zeros((480,832,3),np.uint8),camera=c,native_frames=17,seed=0)
    assert not requests
    next(backend.model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError,match="frozen"):backend.snapshot()
    backend.model.requires_grad_(False)
    backend.source_dir.joinpath("changed.txt").write_text("unexpected source bytes")
    with pytest.raises(ValueError,match="source/config changed"):backend.reset(seed=0)


def test_missing_hook_layer_is_rejected_before_any_hooks_are_registered(cpu_backend):
    backend,_,_=cpu_backend
    del backend.model["high"].blocks[0].cam_shift_layer
    with pytest.raises(ValueError,match="hook layer absent"):
        backend.generate(image=np.zeros((480,832,3),np.uint8),camera=camera(),native_frames=17,seed=0)
    assert all(not module._forward_pre_hooks for module in backend.model.modules())
    assert backend.counters["native_frames"]==0 and not backend._busy


def test_runtime_config_cannot_silently_change_locked_prompt_or_vae_mapping(cpu_backend):
    backend,_,_=cpu_backend
    backend.runtime_config["sampling"]["prompt"]="an unrecorded replacement"
    with pytest.raises(ValueError,match="in-memory"):
        backend.generate(image=np.zeros((480,832,3),np.uint8),camera=camera(),native_frames=17,seed=0)
    backend.runtime_config["sampling"]["prompt"]=config()["sampling"]["prompt"]
    backend.wan_config.vae_stride=(8,8,8)
    with pytest.raises(ValueError,match="in-memory"):backend.reset(seed=0)


def test_rejected_nonfinite_seed_keeps_serializable_failure_telemetry(cpu_backend):
    backend,_,_=cpu_backend
    with pytest.raises(ValueError,match="seed required"):
        backend.generate(image=np.zeros((480,832,3),np.uint8),camera=camera(),native_frames=17,seed=float("nan"))
    receipt=backend.generation_evidence()
    assert receipt["observations"][0]["seed"] is None
    assert receipt["observations"][0]["status"]=="FAILED"
    json.dumps(receipt,allow_nan=False)


@pytest.mark.parametrize("corruption", ["vae_attribute", "missing_mode", "invalid_rng", "freq_shape", "foreign_device", "cache_key"])
def test_malformed_restore_rejected_before_any_actual_state_mutation(cpu_backend, corruption):
    backend, _, _ = cpu_backend; backend.reset(seed=27); state = backend.snapshot()
    if corruption == "vae_attribute": state["mutable"]["vae_attributes"]["model"] = "unexpected object"
    elif corruption == "missing_mode": del state["mutable"]["training_modes"]["high"][""]
    elif corruption == "invalid_rng": state["rng"]["python"] = (3, (0,), None)
    elif corruption == "freq_shape": state["mutable"]["freqs"]["low"] = backend.torch.ones(3)
    elif corruption == "foreign_device": state["mutable"]["placements"]["high"] = "cuda:7"
    elif corruption == "cache_key": state["mutable"]["vae_cache"]["model"] = "unexpected object"
    # If checks happened late, this valid first buffer would have been written.
    state["mutable"]["buffers"]["low"]["entered"].fill_(777)
    before = backend._clone(backend.resource_state()); counters = dict(backend.counters)
    with pytest.raises(ValueError): backend.restore(state)
    after = backend.resource_state()
    def equal(left, right):
        if backend.torch.is_tensor(left):
            assert left.dtype == right.dtype and left.device == right.device and backend.torch.equal(left, right)
        elif isinstance(left, np.ndarray): assert np.array_equal(left, right)
        elif isinstance(left, dict):
            assert set(left) == set(right)
            for key in left: equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert type(left) is type(right) and len(left) == len(right)
            for a, b in zip(left, right): equal(a, b)
        else: assert left == right
    equal(before, after)
    assert backend.counters == counters and backend.vae.model is not None


def test_resource_observer_keeps_actual_aliases_snapshot_owns_independent_tensors(cpu_backend):
    backend, _, _ = cpu_backend
    backend.vae.model._feat_map[0] = backend.model["low"].entered
    observed = backend.resource_state()["mutable"]; saved = backend.snapshot()["mutable"]
    owner = backend.model["low"].entered.data_ptr()
    assert observed["buffers"]["low"]["entered"].data_ptr() == owner
    assert observed["vae_cache"]["_feat_map"][0].data_ptr() == owner
    assert saved["buffers"]["low"]["entered"].data_ptr() != owner
    assert saved["vae_cache"]["_feat_map"][0].data_ptr() != owner
