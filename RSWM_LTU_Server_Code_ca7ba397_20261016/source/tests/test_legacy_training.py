"""CPU software wiring controls, never real Wan weights or training acceptance."""
import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from rswm_inria.geometry import canonical, digest, file_hash
from rswm_inria.legacy_training import validate_training_spec


def training_spec():
    return {"trainable_policy": {"mode": "new_lora", "rank": 2, "alpha": 2,
                                 "targets": "attn_only", "gradient_checkpointing": False},
            "timestep_sampling": "uniform_integer", "expert_residency": "resident", "precision": "bf16",
            "validation": {"world_ids": ["dev-software-only"], "family": "F2", "origin_index": 0,
                           "seed": 9, "max_total_forwards": 2}}


def test_no_legacy_epoch_or_unbudgeted_training_defaults():
    spec = training_spec()
    assert validate_training_spec(spec) == spec
    with pytest.raises(ValueError, match="locked"):
        validate_training_spec(None)
    for key in spec:
        altered = copy.deepcopy(spec); altered.pop(key)
        with pytest.raises(ValueError): validate_training_spec(altered)


@pytest.mark.parametrize("section,key,value", [
    (None, "timestep_sampling", "alternating_experts"), (None, "expert_residency", "cpu_swap"),
    (None, "precision", "fp32"), ("trainable_policy", "mode", "restore_or_create"),
    ("trainable_policy", "rank", True), ("trainable_policy", "alpha", float("nan")),
    ("trainable_policy", "targets", "ffn_only"), ("trainable_policy", "gradient_checkpointing", 1),
    ("validation", "world_ids", ["same", "same"]), ("validation", "family", "F7"),
    ("validation", "origin_index", 4), ("validation", "seed", True),
    ("validation", "max_total_forwards", 0),
])
def test_locked_policy_and_validation_negatives(section, key, value):
    spec = training_spec()
    (spec if section is None else spec[section])[key] = value
    with pytest.raises(ValueError): validate_training_spec(spec)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def test_original_flow_velocity_noise_and_expert_boundary(torch, monkeypatch):
    from rswm_inria.legacy_training import flow_matching_inputs
    z = torch.ones(2, 16, 1, 1, 1, dtype=torch.float32) * .25
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([0, 947]))
    monkeypatch.setattr(torch, "randn_like", lambda x, **kwargs: torch.ones_like(x, dtype=kwargs["dtype"]))
    noisy, target, steps = flow_matching_inputs(z, 1000)
    assert noisy.dtype == torch.bfloat16 and target.dtype == torch.float32
    assert steps.tolist() == [0, 947]
    assert torch.equal(noisy[0].float(), z[0])
    expected = ((1-.947)*.25+.947)
    assert noisy[1, 0, 0, 0, 0].item() == torch.tensor(expected, dtype=torch.bfloat16).item()
    assert torch.equal(target, torch.full_like(target, .75))


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_frozen_content_hash_is_bounded_and_catches_data_edits(torch, monkeypatch, dtype):
    from rswm_inria.training import _parameter_byte_chunks, parameter_snapshot, parameter_changes, parameter_content_hash
    frozen = torch.nn.Parameter(torch.arange(35, dtype=getattr(torch, dtype)).reshape(5, 7).t(), requires_grad=False)
    trainable = torch.nn.Parameter(torch.tensor([.2]))
    original_clone = torch.Tensor.clone
    def reject_frozen_clone(tensor, *args, **kwargs):
        assert tensor.data_ptr() != frozen.data_ptr(), "Frozen backbone must not be cloned"
        return original_clone(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "clone", reject_frozen_clone)
    original_empty = torch.empty
    def bounded_empty(shape, *args, **kwargs):
        assert np.prod(shape) * torch.tensor([], dtype=kwargs["dtype"]).element_size() <= 16
        return original_empty(shape, *args, **kwargs)
    monkeypatch.setattr(torch, "empty", bounded_empty)
    chunks = list(_parameter_byte_chunks(frozen, 16))
    assert chunks and all(c.nbytes <= 16 for c in chunks)
    expected = hashlib.sha256(canonical({"dtype": str(frozen.dtype), "shape": list(frozen.shape)}))
    expected.update(frozen.detach().contiguous().view(torch.uint8).numpy().tobytes())
    assert parameter_content_hash(frozen, max_bytes=16) == expected.hexdigest()
    params = {"backbone": frozen, "lora": trainable}
    snapshot = parameter_snapshot(params, {"lora": trainable}, max_bytes=16)
    assert parameter_changes(params, snapshot, max_bytes=16) == []
    version = frozen._version
    frozen.data[0, 0] += 1
    assert frozen._version == version
    assert parameter_changes(params, snapshot, max_bytes=16) == ["backbone"]


def test_active_experts_do_not_relax_global_gradient_gate(torch):
    from rswm_inria.training import audit_trainable_gradients, merge_active_groups
    low = torch.nn.Parameter(torch.tensor(1.)); high = torch.nn.Parameter(torch.tensor(2.))
    low.grad = torch.tensor(1.)
    trainable = {"low.lora": low, "high.lora": high}; groups = {}
    merge_active_groups(groups, {"low": ["low.lora"]}, trainable)
    assert audit_trainable_gradients(trainable, groups) == {"low.lora"}
    merge_active_groups(groups, {"high": ["high.lora"]}, trainable)
    with pytest.raises(ValueError, match="active expert: high"):
        audit_trainable_gradients(trainable, groups)
    high.grad = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="Nonfinite"):
        audit_trainable_gradients(trainable, {"low": {"low.lora"}})
    with pytest.raises(ValueError, match="Invalid"):
        merge_active_groups({}, {"low": ["not_a_trainable"]}, trainable)


def make_backend(torch, tmp_path, *, configure=True):
    """Tiny API fixtures for the real loss wiring; not a generation backend."""
    from rswm_inria.legacy_training import LegacyTrainingMixin
    from rswm_inria.adapters.camera_conditioning import CameraConfig
    class AdapterLinear(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.base_layer = torch.nn.Linear(1, 1)
            self.lora_A = torch.nn.Parameter(torch.tensor(.15)); self.lora_B = torch.nn.Parameter(torch.tensor(.25))
    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__(); block = torch.nn.Module(); block.self_attn = torch.nn.Module()
            block.self_attn.q = AdapterLinear(); self.blocks = torch.nn.ModuleList([block]); self.calls = []
            self.peft_config = {"default": SimpleNamespace(r=2, lora_alpha=2, target_modules={"blocks.0.self_attn.q"})}
        def get_base_model(self): return self
        def forward(self, *, x, t, context, seq_len, y, dit_cond_dict):
            assert len(x) == len(context) == len(y) == 1
            assert y[0].shape == (20, 9, 60, 104) and seq_len == 14040
            assert torch.equal(y[0][:4, :, 0, 0], torch.tensor([[1]+[0]*8]*4, dtype=y[0].dtype))
            self.calls.append({"t": t.tolist(), "camera": dit_cond_dict, "y_first": y[0][4:, 0, 0, 0].detach().clone()})
            layer = self.blocks[0].self_attn.q
            return [torch.zeros_like(x[0]) + layer.lora_A + layer.lora_B]
    class VAE:
        def __init__(self): self.model = torch.nn.Linear(1, 1).requires_grad_(False); self.encodes = []
        def encode(self, videos):
            result = []
            for video in videos:
                assert video.shape == (3, 33, 480, 832)
                self.encodes.append((video.shape, video[:, -1, 0, 0].clone()))
                result.append(video[:1, ::4, ::8, ::8].float().repeat(16, 1, 1, 1))
            return result
    class Text:
        def __init__(self): self.model = torch.nn.Linear(1, 1).requires_grad_(False); self.prompts = []
        def __call__(self, prompts, device):
            self.prompts.append(prompts); return [torch.zeros(1, 1, device=device) for _ in prompts]
    class Backend(LegacyTrainingMixin):
        def __init__(self):
            self.runtime_config = {"mode": "restored_lora", "training": training_spec(), "sampling": {"inactive_model_cpu": False, "prompt": "fixed aerial scene"}}
            self.runtime_config["training"]["trainable_policy"]["mode"] = "restored_lora"
            self.model = torch.nn.ModuleDict({"low": Expert(), "high": Expert()}); self.vae = VAE(); self.text_encoder = Text()
            self.device = torch.device("cpu"); self.dtype = torch.float32
            self.wan_config = SimpleNamespace(vae_stride=(4,8,8), patch_size=(1,2,2), num_train_timesteps=1000, boundary=.947, param_dtype=torch.float32)
        def training_components(self):
            return {"vae": self.vae, "text_encoder": self.text_encoder, "experts": dict(self.model.items()), "wan_config": self.wan_config, "device": self.device, "dtype": self.dtype}
        def replace_training_experts(self, experts): self.model.update(experts)
    backend = Backend()
    world = np.zeros((2048, 2048, 3), np.uint8); path = tmp_path / 'dev-software-only.png'; Image.fromarray(world).save(path)
    row = {"world_id": "dev-software-only", "path": path.name, "city": "kitsap", "role": "dev", "world_file_sha256": file_hash(path), "world_pixel_sha256": hashlib.sha256(world.tobytes()).hexdigest()}
    config = {"validation_index": [row], "updates": 2, "trainable_policy": backend.runtime_config["training"]["trainable_policy"], "precision": "bf16", "camera": {"translation_scaling": "fixed_units", "framewise": False, "camera_channel": "drop", "temporal_stride": 4}}
    backend.bind_training_context(SimpleNamespace(root=tmp_path), config)
    if configure:
        backend.configure_trainable(config["trainable_policy"])
    return backend, config, CameraConfig(**config['camera'])


def test_actual_loss_wiring_separates_experts_samples_mask_and_prompt(torch, tmp_path, monkeypatch):
    from rswm_inria.adapters.camera_conditioning import backend_payload, build_camera_batch
    from rswm_inria.training import audit_trainable_gradients
    backend, config, cam = make_backend(torch, tmp_path)
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([0, 947]))
    monkeypatch.setattr(torch, "randn_like", lambda x, **kwargs: torch.ones_like(x, dtype=kwargs["dtype"]))
    images = torch.zeros(1, 1, 3, 1, 1).expand(2, 33, 3, 480, 832)
    camera = backend_payload(build_camera_batch(np.zeros((32,2), dtype=np.int64), config=cam))
    loss = backend.loss(images=images, cameras=[camera, camera], update=0)
    assert loss.item() == pytest.approx(.36, abs=1e-6)
    loss.backward()
    params = {name:p for name,p in backend.model.named_parameters() if p.requires_grad}
    gradients = audit_trainable_gradients(params, {k:set(v) for k,v in backend.active_trainable_groups().items()})
    assert any(name.startswith('low.') for name in gradients) and any(name.startswith('high.') for name in gradients)
    assert backend.model['low'].calls[0]['t'] == [0] and backend.model['high'].calls[0]['t'] == [947]
    assert all(call['camera'] is None for expert in backend.model.values() for call in expert.calls)
    assert backend.text_encoder.prompts == [['fixed aerial scene', 'fixed aerial scene']]
    assert all(not p.requires_grad for p in backend.vae.model.parameters())
    assert all(not p.requires_grad for p in backend.text_encoder.model.parameters())


def test_training_rejects_missing_budget_offload_and_frames(torch, tmp_path):
    backend, config, _ = make_backend(torch, tmp_path)
    with pytest.raises(ValueError, match='33-frame'):
        backend.loss(images=torch.zeros(1,17,3,2,2), cameras=[{}], update=0)
    backend.runtime_config['training']['validation']['max_total_forwards'] = 3
    with pytest.raises(ValueError, match='forward budget'):
        backend.bind_training_context(SimpleNamespace(root=tmp_path), config)
    backend.runtime_config['training']['validation']['max_total_forwards'] = 2
    backend.runtime_config['sampling']['inactive_model_cpu'] = True
    with pytest.raises(ValueError, match='offload'):
        backend.bind_training_context(SimpleNamespace(root=tmp_path), config)


@pytest.mark.parametrize("mismatch", ["merged", "targets", "rank", "mode"])
def test_restored_training_requires_actual_unmerged_matching_lora(torch, tmp_path, mismatch):
    backend, config, _ = make_backend(torch, tmp_path, configure=False)
    if mismatch == "merged": backend.model['low'].peft_config = {}
    elif mismatch == "targets": backend.model['low'].peft_config['default'].target_modules = {'wrong'}
    elif mismatch == "rank": backend.model['low'].peft_config['default'].r = 4
    else: backend.runtime_config['mode'] = 'base'
    with pytest.raises(ValueError, match='adapter|rank/alpha/targets|Restored'):
        backend.configure_trainable(config['trainable_policy'])
    with pytest.raises(ValueError, match='already attempted'):
        backend.configure_trainable(config['trainable_policy'])


def test_parameter_binding_rejects_unfreezing_and_replacement(torch):
    from rswm_inria.training import verify_parameter_binding
    model = torch.nn.Linear(1, 1); model.bias.requires_grad_(False)
    params = dict(model.named_parameters()); trainable = {'weight': model.weight}
    verify_parameter_binding(model, params, trainable)
    model.bias.requires_grad_(True)
    with pytest.raises(ValueError, match='selection changed'):
        verify_parameter_binding(model, params, trainable)
    model.bias.requires_grad_(False); model.weight = torch.nn.Parameter(model.weight.detach().clone())
    with pytest.raises(ValueError, match='binding changed'):
        verify_parameter_binding(model, params, trainable)


def test_validation_reads_dev_pixels_preserves_rng_and_restores_progress(torch, tmp_path, monkeypatch):
    backend, config, cam = make_backend(torch, tmp_path)
    seen = []
    def observed_loss(*, images, cameras, update):
        assert images.shape == (1,33,3,480,832) and torch.equal(images, torch.full_like(images, -1))
        assert cameras[0]['camera_manifest_sha256'] == digest(cam.manifest())
        seen.append(images.shape); torch.rand(3)
        return torch.tensor(.125)
    monkeypatch.setattr(backend, 'loss', observed_loss)
    before = torch.get_rng_state().clone()
    result = backend.validate(config['validation_index'], camera_manifest=cam.manifest())
    assert result['status'] == 'MEASURED_DEV_LOSS_NO_RELEASE_ACCEPTANCE' and result['loss'] == .125
    assert result['forward_count'] == 1 and seen and torch.equal(torch.get_rng_state(), before)
    assert backend.checkpoint_training_state() == {'validation_forwards': 1}
    backend.load_checkpoint_training_state({'validation_forwards': 1}, step=1)
    backend._validation_forwards = 2
    backend.load_checkpoint_training_state({'validation_forwards': 1}, step=1)
    assert backend._validation_forwards == 2
    with pytest.raises(ValueError, match='budget exhausted'):
        backend.validate(config['validation_index'], camera_manifest=cam.manifest())
    with pytest.raises(ValueError, match='retained validation'):
        backend.load_checkpoint_training_state(None, step=1)
    assert result['publication_acceptance'] == 'NOT_READY'


def test_dev_missing_or_tampered_pixels_cannot_be_a_validation_pass(torch, tmp_path, monkeypatch):
    backend, config, cam = make_backend(torch, tmp_path)
    monkeypatch.setattr(backend, 'loss', lambda **kwargs: (_ for _ in ()).throw(AssertionError('must not forward bad source')))
    (tmp_path/config['validation_index'][0]['path']).write_bytes(b'not retained pixels')
    with pytest.raises(ValueError, match='PNG hash'):
        backend.validate(config['validation_index'], camera_manifest=cam.manifest())
    assert backend._validation_forwards == 0


def test_driver_checkpoint_resume_active_experts_and_frozen_parameters(torch, tmp_path, monkeypatch):
    """Exercise driver mechanics with explicit contract/preprocess mocks.

    No ClipDataset, audited source receipt, real model or formal training gate is
    represented by this isolated unit control. It must not yield a PASS receipt.
    """
    import rswm_inria.training as training
    import rswm_inria.adapters.camera_conditioning as conditioning
    monkeypatch.setattr(training, 'validate_dev_index', lambda *_args: None)
    monkeypatch.setattr(conditioning, 'to_native_rgb', lambda _frame: np.zeros((2, 2, 3), np.uint8))
    class Dataset:
        diagnostic = False  # Contract validation is mocked only in this test.
        index_sha256 = 'a'*64; source_audit_sha256 = 'b'*64  # Mocked digests, never an audit receipt.
        def __len__(self): return 1
        def __getitem__(self, index):
            return {'images': np.zeros((33, 1, 1, 3), np.uint8), 'q': np.zeros((33, 2), np.int64)}
    class Expert(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.lora = torch.nn.Parameter(torch.tensor(.2))
            self.frozen = torch.nn.Parameter(torch.arange(7, dtype=torch.float32), requires_grad=False)
    class Backend:
        def __init__(self):
            self.model = torch.nn.ModuleDict({'low': Expert(), 'high': Expert()})
            self.forwards = 0; self.active = {}; self.bound = False
        def configure_trainable(self, policy): assert policy == {'fixture': 'software-only'}
        def bind_training_context(self, dataset, config): self.bound = True
        def loss(self, *, images, cameras, update):
            assert self.bound and images.shape == (1, 33, 3, 2, 2)
            tag = 'low' if update == 0 else 'high'; self.active = {tag: [tag+'.lora']}
            return (self.model[tag].lora - (torch.rand(())+.5)).square()
        def active_trainable_groups(self): return self.active
        def validate(self, rows, *, camera_manifest):
            self.forwards += 1
            return {'status': 'SOFTWARE_MECHANICS_ONLY', 'forwards': self.forwards}
        def checkpoint_training_state(self): return {'forwards': self.forwards}
        def load_checkpoint_training_state(self, state, *, step):
            assert state['forwards'] == step; self.forwards = state['forwards']
    config = {'updates': 2, 'batch_size': 1, 'accumulation': 1, 'lr': .01, 'seed': 23,
              'precision': 'fp32', 'device': 'cpu', 'checkpoint_every': 1,
              'camera': {'translation_scaling': 'fixed_units', 'framewise': False,
                         'camera_channel': 'drop', 'temporal_stride': 4},
              'trainable_policy': {'fixture': 'software-only'},
              'backend_manifest': {'purpose': 'software-only'}, 'validation_index': [{'fixture': 'dev'}]}
    first = Backend(); full = training.train(Dataset(), first, config, tmp_path/'full')
    second = Backend(); resumed = training.train(Dataset(), second, config, tmp_path/'resumed', resume=tmp_path/'full/update-00000001.pt')
    assert full['clip_cursor'] == resumed['clip_cursor'] == 2 and first.forwards == second.forwards == 2
    assert full['audit'][0]['active_trainable_groups'] == {'low': ['low.lora']}
    assert full['audit'][1]['active_trainable_groups'] == {'high': ['high.lora']}
    for name, value in first.model.state_dict().items():
        assert torch.equal(value, second.model.state_dict()[name])
    assert torch.equal(first.model['low'].frozen, torch.arange(7, dtype=torch.float32))
    state = torch.load(tmp_path/'resumed/update-00000002.pt', weights_only=True)
    assert state['backend_training'] == {'forwards': 2}
    assert state['position'] == 2 and state['step'] == 2
