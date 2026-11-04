"""Dependency-free rejection controls, without model/data acceptance."""
import copy
import subprocess
import sys

import pytest

from rswm_inria.legacy_export import (BINDING_SCHEMA, _formal_retained_data,
                                      validate_release_binding)
from rswm_inria.legacy_wan import (SCHEMA_V2, component_inventory,
                                   validate_backend_config)
from rswm_inria.registry import validate_config
from rswm_inria.adapters.camera_conditioning import CameraConfig
from rswm_inria.geometry import digest


def reference(tmp_path, scope="software_control"):
    return {"schema": BINDING_SCHEMA, "root": str(tmp_path / "retained"),
            "tree_sha256": "1" * 64, "manifest_sha256": "2" * 64, "scope": scope}


def backend_config(tmp_path):
    return {"schema": SCHEMA_V2, "mode": "restored_lora", "config_key": "i2v-A14B",
            "sampling": {"prompt": "Locked explicit setting", "negative_prompt": "",
                         "steps": 2, "shift": 5, "guide_scale": [3, 3], "boundary": .875,
                         "param_dtype": "bf16", "t5_dtype": "bf16", "t5_cpu": True,
                         "inactive_model_cpu": False}, "lora_subdir": None, "training": None,
            "adapter_release": reference(tmp_path)}


@pytest.mark.parametrize("corruption", ["relative", "extra", "hash", "scope", "version"])
def test_release_reference_requires_complete_explicit_identity(tmp_path, corruption):
    value = reference(tmp_path)
    if corruption == "relative": value["root"] = "relative/release"
    elif corruption == "extra": value["approved"] = True
    elif corruption == "hash": value["tree_sha256"] = "1"
    elif corruption == "scope": value["scope"] = "REAL_MODEL_VALIDATED"
    else: value["schema"] = "unknown"
    with pytest.raises(ValueError):
        validate_release_binding(value)


@pytest.mark.parametrize("field,value", [("training", {}), ("lora_subdir", "adapters"), ("mode", "base")])
def test_v2_is_a_single_fresh_inference_path(tmp_path, field, value):
    config = backend_config(tmp_path)
    config[field] = value
    with pytest.raises(ValueError):
        validate_backend_config(config)


def test_actual_wan_rejects_software_release_before_torch_or_weights(tmp_path):
    with pytest.raises(ValueError, match="Software-control"):
        component_inventory(tmp_path / "absent-base", backend_config(tmp_path))


@pytest.mark.parametrize("location", ["index", "clips", "source_audit"])
def test_formal_candidate_cannot_relabel_explicit_software_inputs(location):
    context = {"index": {}, "clips": {}, "source_audit": {}, "source_code_sha256": "1" * 64}
    context[location]["synthetic"] = True
    with pytest.raises(ValueError, match="Synthetic/software"):
        _formal_retained_data(context)


def test_offline_numeric_replay_keeps_trained_pointer_without_original_assets(tmp_path):
    camera = {"translation_scaling": "fixed_units", "framewise": True,
              "camera_channel": "native", "temporal_stride": 4}
    config = {"model_id": "software-peft", "adapter": "numeric-camera", "device": "cpu",
              "training_seed": 83, "camera": camera, "training_camera_sha256": digest(CameraConfig(**camera).manifest()),
              "adapter_release": reference(tmp_path),
              "backend_manifest": {"source_dir": str(tmp_path / "absent-source"), "source_tree_sha256": "3" * 64,
                  "entrypoint": "backend.py", "weights_dir": str(tmp_path / "absent-base"), "weights_tree_sha256": "4" * 64,
                  "objective": "flow_matching", "approved": True, "license_record": "Explicit software-only control"}}
    assert validate_config(config, check_local_assets=False) == config
    missing = copy.deepcopy(config); missing.pop("training_seed")
    with pytest.raises(ValueError, match="explicit training_seed"):
        validate_config(missing, check_local_assets=False)


def test_exporter_help_does_not_import_optional_runtime_dependencies():
    code = """import sys
from rswm_inria.legacy_export import main
assert 'torch' not in sys.modules and 'peft' not in sys.modules
try:
    main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
assert 'torch' not in sys.modules and 'peft' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
