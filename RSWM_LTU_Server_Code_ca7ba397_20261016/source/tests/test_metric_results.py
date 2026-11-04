"""Retained PNG/mosaic/feature consumers, with explicit synthetic scope."""
import copy
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from rswm_inria import suite, motion_protocol, quality_evaluation
from rswm_inria.geometry import canonical,digest
from rswm_inria.metric_results import consume_metric_results,_motion_rows
from rswm_inria.quality_protocol import build_patch_plan,assemble_suite_mosaics,assemble_reference_mosaics,load_mosaics
from rswm_inria.features import encode_patches,save_feature_bundle


class DiagnosticMeanFeatures:
    feature_dimension=3
    def __init__(self):
        self.manifest={"schema":"rswm-diagnostic-feature-backend-v1","provider":"synthetic-mean-v1",
                       "feature_dimension":3,"diagnostic_features":True,"release_validated":False}
        self.manifest["feature_identity_sha256"]=digest(self.manifest)
    def encode(self,images):
        return np.stack([image.mean(axis=(0,1))/255 for image in images]).astype(np.float32)


def make_fixture(root,families=("F2",),seeds=(0,),execute=True):
    """Root's CLI reproduction may call this; all pixels/models/features synthetic."""
    from rswm_inria.tasks import packages
    root=Path(root);root.mkdir(parents=True,exist_ok=True);items=[]
    for index,value in enumerate((0,255)):
        world_id=f"synthetic-metric-{index}";world=root/f"world-{index}.png"
        Image.fromarray(np.full((2048,2048,3),value,np.uint8)).save(world)
        for task,_ in packages(world_id,"diagnostic_only",0):
            if task["family"] not in families or task["origin_index"]!=0:continue
            path=root/f"{task['task_id']}.json";path.write_bytes(canonical(task)+b"\n")
            items.append({"task_path":str(path),"world_path":str(world),"city":("sfo","innsbruck")[index]})
    plan=suite.build_plan(items,[{"model_id":"oracle","adapter":"oracle","training_seed":7}],list(seeds),"a"*64,"b"*40)
    suite_root=root/"suite"
    (root/"plan.json").write_bytes(canonical(plan)+b"\n")
    metadata={r["world_id"]:{"city":r["city"],"role":r["role"],"world_file_sha256":r["world_file_sha256"],
                            "source_identity":"world-pixels-sha256:"+r["world_pixel_sha256"]} for r in plan["expected"]}
    patch_plan=build_patch_plan(list(plan["tasks"].values()),count=2,world_metadata=metadata)
    (root/"patch-plan.json").write_bytes(canonical(patch_plan)+b"\n")
    if execute:suite.execute_plan(plan,suite_root)
    else:
        suite_root.mkdir();(suite_root/"plan.json").write_bytes(canonical(plan)+b"\n")
        (suite_root/"expected.jsonl").write_bytes(b"".join(canonical(r)+b"\n" for r in plan["expected"]))
    assemble_suite_mosaics(plan,suite_root,patch_plan,root/"generated-mosaics")
    assemble_reference_mosaics(plan,patch_plan,root/"reference-mosaics")
    backend=DiagnosticMeanFeatures()
    for kind in ("generated","reference"):
        bundle,patches=load_mosaics(root/(kind+"-mosaics")/"mosaics.json")
        encoded=encode_patches(patches,backend,root/(kind+"-cache"))
        save_feature_bundle(encoded,backend.manifest,root/(kind+"-features"),bundle["bundle_sha256"])
    quality=quality_evaluation.evaluate_suite(plan,suite_root,root/"generated-features/features.json",root/"reference-features/features.json",
        root/"generated-mosaics/mosaics.json",root/"reference-mosaics/mosaics.json",root/"patch-plan.json",root/"quality.json",cap=2,bootstrap=2,seed=0)
    motion=motion_protocol.evaluate_suite(plan,suite_root,motion_protocol.motion_config(),root/"motion",diagnostic=True)
    return {"root":root,"plan":plan,"suite":suite_root,"quality":quality,"motion":motion,
            "quality_path":root/"quality.json","motion_path":root/"motion/motion.json"}


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):return make_fixture(tmp_path_factory.mktemp("retained-metric-baseline"))


@pytest.fixture
def fx(tmp_path,baseline):
    # Mutate independent artifact copies; generate the expensive PNG fixture once.
    shutil.copytree(baseline["root"],tmp_path,dirs_exist_ok=True)
    report=copy.deepcopy(baseline["quality"])
    for key,value in report["retained_inputs"].items():
        if isinstance(value,str):report["retained_inputs"][key]=value.replace(str(baseline["root"]),str(tmp_path))
    report=reseal(report,"evaluation_sha256");write(tmp_path/"quality.json",report)
    return {"root":tmp_path,"plan":baseline["plan"],"suite":baseline["suite"],"quality":report,
            "motion":baseline["motion"],"quality_path":tmp_path/"quality.json","motion_path":tmp_path/"motion/motion.json"}


def reseal(report,key):
    report=copy.deepcopy(report);report.pop(key,None);report[key]=digest(report);return report


def write(path,report):Path(path).write_bytes(canonical(report)+b"\n")


def test_no_metrics_does_not_infer_scores(fx):
    output=consume_metric_results(fx["plan"],fx["suite"])
    assert not output["motion_rows"] and not output["quality_rows"]
    assert output["availability"]["motion"]["status"]=="NOT_AVAILABLE"


def test_retained_pipeline_negative_mmd_null_motion_and_explicit_fixture_scope(fx):
    output=consume_metric_results(fx["plan"],fx["suite"],motion_results=[fx["motion_path"]],quality_results=[fx["quality_path"]])
    motion=output["motion_rows"][0];quality=output["quality_rows"][0]
    assert motion["coverage"]==0 and motion["mean_normalized_step_error"] is None
    assert quality["polynomial_mmd_u"]<0 and quality["planned_tiles"]==quality["complete_tiles"]==4
    assert quality["inference_seeds"]==[0] and quality["training_seed"]==7
    assert quality["statistical_unit"].startswith("frozen patch population") and "task_id" not in quality
    for row in (*output["motion_rows"],*output["quality_rows"]):
        assert row["diagnostic_only"] and row["scope"]=="DIAGNOSTIC_ONLY" and not row["formal_metric_eligible"]
    assert output["provenance"][0]["binding"]["runs"][0]["retained_submission_files"]


@pytest.mark.parametrize("artifact",[{"schema":"rswm-motion-validation-v1","release_validated":True},[],3])
def test_instrument_or_nonobject_cannot_produce_model_scores(fx,artifact):
    path=fx["root"]/"instrument-only.json";write(path,artifact)
    with pytest.raises(ValueError,match="MODEL_EVALUATION_REQUIRED"):
        consume_metric_results(fx["plan"],fx["suite"],motion_results=[path])


@pytest.mark.parametrize("field",["model_id","model_manifest_sha256","seed","training_seed","attempt","study","source_manifest_sha256","metric_config_sha256"])
def test_resealed_wrong_run_identity_cannot_enter_table(fx,field):
    report=copy.deepcopy(fx["motion"]);report["edges"][0][field]="wrong"
    write(fx["motion_path"],reseal(report,"motion_suite_sha256"))
    with pytest.raises(ValueError,match="REPLAY_MISMATCH"):
        consume_metric_results(fx["plan"],fx["suite"],motion_results=[fx["motion_path"]])


@pytest.mark.parametrize("field",["study","models_sha256","source_manifest_sha256","metric_config_sha256","source_code_sha256","analysis_selection"])
def test_resealed_wrong_evaluation_binding_cannot_enter_table(fx,field):
    report=copy.deepcopy(fx["motion"]);report["evaluation_binding"][field]="wrong"
    write(fx["motion_path"],reseal(report,"motion_suite_sha256"))
    with pytest.raises(ValueError,match="BINDING_CHANGED|Unknown analysis"):
        consume_metric_results(fx["plan"],fx["suite"],motion_results=[fx["motion_path"]])


def test_resealed_tampered_quality_value_is_recomputed(fx):
    report=copy.deepcopy(fx["quality"]);report["reports"][0]["own_complete_sample_diagnostic"]["polynomial_mmd_u"]+=1
    write(fx["quality_path"],reseal(report,"evaluation_sha256"))
    with pytest.raises(ValueError,match="RECOMPUTATION_MISMATCH"):
        consume_metric_results(fx["plan"],fx["suite"],quality_results=[fx["quality_path"]])


@pytest.mark.parametrize("field",["schema","kernel","population_grouping"])
def test_resealed_wrong_statistical_configuration_is_rejected(fx,field):
    report=copy.deepcopy(fx["quality"]);report["statistical_config"][field]="wrong"
    report["statistical_config_sha256"]=digest(report["statistical_config"])
    write(fx["quality_path"],reseal(report,"evaluation_sha256"))
    with pytest.raises(ValueError,match="STATISTICAL_CONFIG_CHANGED"):
        consume_metric_results(fx["plan"],fx["suite"],quality_results=[fx["quality_path"]])


@pytest.mark.parametrize("kind",["missing_feature","corrupt_feature","missing_mosaic","changed_export"])
def test_retained_files_required_not_only_resealed_pass(fx,kind):
    if kind in {"missing_feature","corrupt_feature"}:
        path=next((fx["root"]/"generated-features").glob("*.npy"))
        if kind=="missing_feature":path.unlink()
        else:path.write_bytes(b"broken-cache")
    elif kind=="missing_mosaic":next((fx["root"]/"generated-mosaics").glob("*.npz")).unlink()
    else:(fx["root"]/"motion/motion_edges.csv").write_bytes(b"changed")
    with pytest.raises((ValueError,OSError)):
        consume_metric_results(fx["plan"],fx["suite"],motion_results=[fx["motion_path"]],quality_results=[fx["quality_path"]])


def test_duplicate_and_unexpected_result_groups_rejected(fx):
    with pytest.raises(ValueError,match="DUPLICATE_RESULT_INPUT"):
        consume_metric_results(fx["plan"],fx["suite"],quality_results=[fx["quality_path"],fx["quality_path"]])
    report=copy.deepcopy(fx["quality"]);report["reports"].append(report["reports"][0])
    write(fx["quality_path"],reseal(report,"evaluation_sha256"))
    with pytest.raises(ValueError,match="RECOMPUTATION_MISMATCH"):
        consume_metric_results(fx["plan"],fx["suite"],quality_results=[fx["quality_path"]])


def test_nonfinite_numbers_rejected_even_with_result_hash(fx):
    raw=fx["quality_path"].read_text().replace('"release_validated":false','"illegal_number":NaN,"release_validated":false')
    fx["quality_path"].write_text(raw)
    with pytest.raises(ValueError):consume_metric_results(fx["plan"],fx["suite"],quality_results=[fx["quality_path"]])


def test_absent_planned_world_keeps_null_primary_quality_and_full_denominator(tmp_path):
    fx=make_fixture(tmp_path,execute=False)
    output=consume_metric_results(fx["plan"],fx["suite"],quality_results=[fx["quality_path"]])
    row=output["quality_rows"][0]
    assert row["planned_tiles"]==4 and row["complete_tiles"]==0 and row["missing_pixels"]==4*128**2
    assert row["polynomial_mmd_u"] is None and row["conditional_complete_sample_mmd_u"] is None
    assert len(row["missing_source_worlds"])==2 and row["real_reference"]["complete_tiles"]==4


def test_seed_populations_and_first_recovery_are_separate(tmp_path):
    fx=make_fixture(tmp_path,seeds=(0,1))
    output=consume_metric_results(fx["plan"],fx["suite"],quality_results=[fx["quality_path"]])
    assert {r["seed"] for r in output["quality_rows"]}=={0,1}
    assert all(r["inference_seeds"]==[r["seed"]] and r["planned_tiles"]==4 for r in output["quality_rows"])
    secondary=motion_protocol.evaluate_suite(fx["plan"],fx["suite"],motion_protocol.motion_config(),tmp_path/"secondary-motion",diagnostic=True,selection="latest_attempt_within_budget")
    output=consume_metric_results(fx["plan"],fx["suite"],motion_results=[fx["motion_path"],tmp_path/"secondary-motion/motion.json"])
    assert {r["analysis_selection"] for r in output["motion_rows"]}=={"first_attempt_only","latest_attempt_within_budget"}


def test_f7_shared_prefix_edges_are_not_repeated(tmp_path):
    fx=make_fixture(tmp_path,families=("F7",))
    output=consume_metric_results(fx["plan"],fx["suite"],motion_results=[fx["motion_path"]])
    assert sum(row["planned_edges"] for row in output["motion_rows"])==2*160
    assert len({e["pair_id"] for e in fx["motion"]["edges"]})==2*160


def test_instrument_release_never_promotes_diagnostic_model_or_pixels(fx):
    report=copy.deepcopy(fx["motion"]);report["release_validated"]=True
    rows=_motion_rows(report,fx["plan"])
    assert all(r["release_validated"] and r["diagnostic_only"] and not r["formal_metric_eligible"] for r in rows)


@pytest.mark.parametrize("kind",["missing_cache_metadata","corrupt_cache_vector"])
def test_original_cache_retained_files_are_verified(fx,kind):
    if kind=="missing_cache_metadata":next((fx["root"]/"generated-cache").glob("*.json")).unlink()
    else:next((fx["root"]/"generated-cache").glob("*.npy")).write_bytes(b"broken-cache")
    with pytest.raises(ValueError,match="RETAINED_FILE_MISSING|ORIGINAL_FEATURE_CACHE|allow_pickle"):
        consume_metric_results(fx["plan"],fx["suite"],quality_results=[fx["quality_path"]])


def test_resealed_missing_required_motion_export_inventory_rejected(fx):
    report=copy.deepcopy(fx["motion"]);report["export_hashes"].pop("motion_edges.csv")
    (fx["root"]/"motion/motion_edges.csv").unlink()
    write(fx["motion_path"],reseal(report,"motion_suite_sha256"))
    with pytest.raises(ValueError,match="RETAINED_EXPORTS_REQUIRED"):
        consume_metric_results(fx["plan"],fx["suite"],motion_results=[fx["motion_path"]])


@pytest.mark.parametrize("payload",[[],3])
def test_quality_release_nonobject_exposure_ledger_fails_closed(tmp_path,payload):
    from rswm_inria.metric_results import _feature_exposure
    path=tmp_path/"ledger.json";write(path,payload)
    exposure={"ledger_path":str(path),"ledger_snapshot":{"schema":"rswm-feature-validation-exposure-v1","exposures":[]},
              "ledger_sha256_at_execution":"a"*64}
    with pytest.raises(ValueError,match="EXPOSURE_LEDGER_CHANGED"):_feature_exposure({"exposure":exposure})


def test_quality_exposure_append_preserves_history_and_deletion_fails(tmp_path):
    import hashlib
    from rswm_inria.metric_results import _feature_exposure
    entry={"reference_bundle_sha256":"a"*64,"feature_config_sha256":"b"*64,"acceptance_sha256":"c"*64,
           "calibration_sha256":"d"*64,"source_population_keys":["e"*64]}
    snapshot={"schema":"rswm-feature-validation-exposure-v1","exposures":[entry]}
    path=tmp_path/"ledger.json";write(path,snapshot)
    receipt={"exposure":{"ledger_path":str(path),"ledger_snapshot":snapshot,
                        "ledger_sha256_at_execution":hashlib.sha256(canonical(snapshot)+b"\n").hexdigest(),
                        "source_population_keys":entry["source_population_keys"]},
             "reference_bundle_sha256":"a"*64,"acceptance_config":{"feature_config_sha256":"b"*64,"acceptance_sha256":"c"*64},
             "calibration_receipt":{"calibration_sha256":"d"*64}}
    assert _feature_exposure(receipt)["sha256_at_execution"]==receipt["exposure"]["ledger_sha256_at_execution"]
    write(path,{"schema":snapshot["schema"],"exposures":[entry,{**entry,"reference_bundle_sha256":"f"*64}]})
    assert _feature_exposure(receipt)["sha256"]!=receipt["exposure"]["ledger_sha256_at_execution"]
    write(path,{"schema":snapshot["schema"],"exposures":[]})
    with pytest.raises(ValueError,match="EXPOSURE_LEDGER_CHANGED"):_feature_exposure(receipt)


def test_motion_retained_controls_must_match_actual_source_pixels(tmp_path):
    import hashlib
    from rswm_inria.geometry import file_hash
    from rswm_inria.metric_results import _motion_sources
    world=np.zeros((2048,2048,3),np.uint8);path=tmp_path/"world.png";Image.fromarray(world).save(path)
    source={"world_id":"synthetic-control","role":"metric_calibration","city":"bellingham","path":"world.png",
            "world_file_sha256":file_hash(path),"world_pixel_sha256":hashlib.sha256(world.tobytes()).hexdigest()}
    manifest=motion_protocol.build_pairs({"source_manifest_sha256":"a"*64,"worlds":[source]},tmp_path,tmp_path/"pairs",
                                       "metric_calibration",pairs_per_world=2,diagnostic_fixture=True)
    _motion_sources(manifest,tmp_path/"pairs")
    changed=copy.deepcopy(manifest);pair=changed["pairs"][0]
    image=tmp_path/"pairs"/pair["images"]["current"];Image.fromarray(np.ones((240,416,3),np.uint8)).save(image)
    pair["image_sha256"]["current"]=file_hash(image);changed=reseal(changed,"pair_manifest_sha256")
    with pytest.raises(ValueError,match="CONTROLLED_PIXELS_NOT_RETAINED_REAL_SOURCE_CROP"):_motion_sources(changed,tmp_path/"pairs")
