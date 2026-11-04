"""Actual local import/factory control; no trained weights or backend validation."""
import copy
import numpy as np
import pytest
from rswm_inria.geometry import digest
from rswm_inria.adapters.camera_conditioning import CameraConfig
from rswm_inria.numeric import NumericLastFrameAdapter
from rswm_inria.registry import validate_config, create_adapter, capabilities
from rswm_inria.suite import model_manifest, model_factory
from rswm_inria.training import locked_tree


def configuration(tmp_path):
    source = tmp_path / "original-source"; source.mkdir()
    source.joinpath("backend.py").write_text('''
from PIL import Image
from rswm_inria.numeric import payload_hash
class SyntheticControlBackend:
    objective = "diffusion"
    def __init__(self, weights_dir, device):
        self.weights_dir = weights_dir
        self.device = device
        self.calls = []
    def generate(self, **request):
        self.calls.append(request)
        return {"frames": [Image.new("RGB", (832,480), (10,20,30)) for _ in range(request["native_frames"])],
                "camera_payload_sha256": payload_hash(request["camera"])}
    def snapshot(self): return {"calls": len(self.calls)}
    def restore(self, state): pass
    def resource_state(self): return {}
def load_backend(*, weights_dir, device):
    return SyntheticControlBackend(weights_dir, device)
''', encoding="utf-8")
    weights = tmp_path / "control-weights"; weights.mkdir()
    weights.joinpath("fixture.bin").write_bytes(b"SYNTHETIC CONTROL ONLY: NOT TRAINED WEIGHTS")
    camera = CameraConfig()
    manifest = {"source_dir": str(source), "source_tree_sha256": digest(locked_tree(source)),
                "weights_dir": str(weights), "weights_tree_sha256": digest(locked_tree(weights)),
                "entrypoint": "backend.py", "objective": "diffusion", "approved": True,
                "license_record": "Test-owned synthetic control source and bytes only"}
    return {"model_id": "numeric-control", "adapter": "numeric-camera", "device": "cpu",
            "backend_manifest": manifest, "camera": {k:getattr(camera,k) for k in
                ("translation_scaling", "framewise", "camera_channel", "temporal_stride")},
            "training_camera_sha256": digest(camera.manifest())}


def test_actual_factory_loads_explicit_original_source_and_consumes_current_public_camera(tmp_path):
    config = configuration(tmp_path)
    model = create_adapter(config)
    assert isinstance(model, NumericLastFrameAdapter)
    assert type(model.backend).__name__ == "SyntheticControlBackend"
    assert model.backend.device == "cpu"
    model.reset(np.zeros((240,416,3),np.uint8), {}, 2)
    q = np.column_stack([np.arange(1,17)*4, np.zeros(16,dtype=np.int64)])
    images = model.generate(q)
    assert len(images)==16 and images[-1].shape==(240,416,3)
    request = model.backend.calls[0]
    assert set(request)=={"image","camera","native_frames","seed"}
    assert request["camera"]["camera_manifest_sha256"]==config["training_camera_sha256"]
    assert "native_queries_xy" not in request["camera"]
    assert capabilities(config)["real_backend_layer_consumption"]=="NOT_RUN"
    # Standard suite factory still supplies only config to this ordinary model.
    assert isinstance(model_factory(config, object(), {"secret": "unavailable"}), NumericLastFrameAdapter)


def test_model_manifest_locks_actual_recursive_source_weights_and_camera(tmp_path):
    config = configuration(tmp_path)
    frozen = model_manifest(config)
    assert digest(frozen["source"])==config["backend_manifest"]["source_tree_sha256"]
    assert digest(frozen["weights"])==config["backend_manifest"]["weights_tree_sha256"]
    assert frozen["camera_manifest_sha256"]==frozen["training_camera_sha256"]
    assert frozen["camera_manifest"]["canonical_to_native"]=="Pillow_BILINEAR_2x"
    assert frozen["camera_manifest"]["native_to_canonical"]=="exact_2x2_uint16_sum_plus2_integer_div4"
    config["backend_manifest"]["approved"]=False
    assert frozen["backend_manifest"]["approved"] is True


@pytest.mark.parametrize("field", ["source_dir", "weights_dir"])
def test_changed_original_source_or_weights_rejected_before_execution(tmp_path, field):
    from pathlib import Path
    config = configuration(tmp_path)
    Path(config["backend_manifest"][field], "new-unlocked.bin").write_bytes(b"changed")
    for operation in (validate_config, create_adapter, model_manifest):
        with pytest.raises(ValueError, match="changed"):
            operation(config)


@pytest.mark.parametrize("extra", ["prompt", "diagnostic_noise", "model_dir", "private_world", "future_queries"])
def test_numeric_public_configuration_cannot_hide_an_unused_or_private_channel(tmp_path, extra):
    config=configuration(tmp_path);config[extra]="ignored"
    with pytest.raises(ValueError, match="Unsupported"):
        validate_config(config)


def test_camera_mismatch_and_absent_original_backend_have_no_fallback(tmp_path):
    config=configuration(tmp_path)
    wrong=copy.deepcopy(config);wrong["camera"]["camera_channel"]="drop"
    with pytest.raises(ValueError,match="mismatch"):create_adapter(wrong)
    absent=copy.deepcopy(config);absent["backend_manifest"]={}
    with pytest.raises(ValueError,match="approved original"):create_adapter(absent)
    incomplete=copy.deepcopy(config);del incomplete["camera"]["framewise"]
    with pytest.raises(ValueError,match="complete camera"):create_adapter(incomplete)
    no_device=copy.deepcopy(config);del no_device["device"]
    with pytest.raises(ValueError,match="Explicit backend device"):create_adapter(no_device)


@pytest.mark.parametrize("channel", ["identity", "drop"])
def test_ablated_camera_factory_never_sends_a_second_command_trace(tmp_path, channel):
    config=configuration(tmp_path);camera=CameraConfig(camera_channel=channel)
    config["camera"]["camera_channel"]=channel
    config["training_camera_sha256"]=digest(camera.manifest())
    model=create_adapter(config);model.reset(np.zeros((240,416,3),np.uint8),{},0)
    model.generate(np.column_stack([np.arange(1,17)*4,np.zeros(16,dtype=np.int64)]))
    payload=model.backend.calls[0]["camera"]
    assert "native_queries_xy" not in payload
    assert payload["camera_manifest"]["camera_channel"]==channel
    if channel=="drop":assert payload["camera_poses"] is None
    else:assert not np.any(payload["camera_poses"][:,:3,3])


def frozen_plan(tmp_path):
    import json
    from PIL import Image
    from rswm_inria.tasks import packages
    from rswm_inria.suite import build_plan
    config=configuration(tmp_path)
    task,_=packages("numeric-registry-control","diagnostic_only",0)[0]
    taskpath=tmp_path/"task.json";taskpath.write_text(json.dumps(task))
    worldpath=tmp_path/"world.png"
    Image.fromarray(np.zeros((2048,2048,3),np.uint8)).save(worldpath)
    plan=build_plan([{"task_path":str(taskpath),"world_path":str(worldpath),"city":"kitsap"}],
                    [config],[0],"a"*64,"b"*40)
    return plan,config


def test_replay_plan_validates_retained_backend_identity_without_reloading_missing_weights(tmp_path):
    from pathlib import Path
    from rswm_inria.suite import validate_plan, execute_plan
    plan,config=frozen_plan(tmp_path)
    for field in ("source_dir","weights_dir"):
        p=Path(config["backend_manifest"][field]);p.rename(p.with_name(p.name+"-unavailable"))
    validate_plan(plan)
    outcome=execute_plan(plan,tmp_path/"suite")[0]
    assert outcome["status"]=="FAILED_INPUT"
    assert outcome["error_type"]=="ValueError"
    assert sum(s["planned_outputs"] for s in plan["expected"][0]["planned_reports"].values())==32


@pytest.mark.parametrize("field", ["source", "weights", "camera_manifest"])
def test_frozen_model_cannot_replace_backend_or_camera_provenance(tmp_path,field):
    from rswm_inria.suite import validate_plan
    plan,_=frozen_plan(tmp_path)
    model=plan["models"]["numeric-control"]
    if field=="camera_manifest":model[field]["camera_channel"]="drop"
    else:model[field][next(iter(model[field]))]="0"*64
    plan["plan_sha256"]=digest({k:v for k,v in plan.items() if k!="plan_sha256"})
    with pytest.raises(ValueError,match="Frozen numeric"):
        validate_plan(plan)


def test_synthetic_local_factory_valid_png_replay_does_not_certify_unknown_backend_cost(tmp_path):
    from rswm_inria.suite import execute_plan, replay_suite
    plan,_=frozen_plan(tmp_path)
    outcome=execute_plan(plan,tmp_path/"suite")[0]
    assert outcome["status"]=="COMPLETED" and outcome["replay_verified"]
    assert outcome["resource_status"]=="INCOMPLETE"
    record=replay_suite(plan,tmp_path/"suite")[0]
    assert record["replay_verified"] and record["resource_status"]=="INCOMPLETE"
    assert record["resources"]["accounting_calls"]["valid_outputs"]==32
