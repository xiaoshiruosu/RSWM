"""D06 geometry/feature/cache/statistics tests; diagnostic providers are explicit."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from rswm_inria.geometry import digest,crop
from rswm_inria.tasks import packages
from rswm_inria.quality_protocol import (build_patch_plan,validate_patch_plan,assemble_patch,save_mosaics,load_mosaics,
                                         _source_indices,_lineages)
from rswm_inria.features import (FeatureUnavailable,validate_feature_config,load_backend,feature_cache_key,encode_patches,
                                 save_feature_bundle,load_feature_bundle,FEATURE_SCHEMA)
from rswm_inria.quality_evaluation import evaluate_distribution,evaluate_methods
from rswm_inria import feature_release


@pytest.fixture(scope="module")
def tasks():return [t for t,_ in packages("quality-test","diagnostic_only",0)]


def images_for(task,color=123):
    return {(stream,event):np.full((240,416,3),color,np.uint8)
            for stream,queries in _lineages(task).items() for event in range(1,len(queries))}


def test_geometry_plan_frozen_before_outputs(tasks):
    plan=build_patch_plan(tasks,count=16)
    validate_patch_plan(plan)
    assert plan["selection_depends_on_images"] is False and plan["patches"]
    assert plan==build_patch_plan(tasks,count=16)
    changed=copy.deepcopy(plan);changed["patches"][0]["xy"][0]+=1
    with pytest.raises(ValueError,match="hash"):validate_patch_plan(changed)


def test_first_planned_failure_cannot_be_repaired(tasks):
    task=next(t for t in tasks if t["family"]=="F5")
    plan=build_patch_plan([task],count=32);patch=plan["patches"][0]
    events,streams,names=_source_indices(task,patch["lineage"],patch["xy"])
    images=images_for(task);first=int(events.min());images[(names[int(streams[events==first][0])],first)]=None
    mosaic=assemble_patch(patch,task,images,diagnostic=True)
    assert not mosaic["complete"] and (mosaic["valid_mask"][events==first]==False).all()
    assert (mosaic["image"][events==first]==0).all()
    assert mosaic["valid_pixels"]<mosaic["planned_pixels"]
    assert all(image is not None for key,image in images.items() if key[1]>first)


def test_f7_prefix_sample_once_and_branch_sources_independent(tasks):
    task=next(t for t in tasks if t["family"]=="F7")
    plan=build_patch_plan([task],count=256)
    prefix={tuple(p["xy"]) for p in plan["patches"] if p["lineage"]=="prefix"}
    for branch in ("a","b"):
        for patch in (p for p in plan["patches"] if p["lineage"]==branch):
            events,_,_=_source_indices(task,branch,patch["xy"])
            assert (events>32).any() and tuple(patch["xy"]) not in prefix
    images=images_for(task)
    for key in images:
        if key[0]=="a":images[key].fill(11)
        if key[0]=="b":images[key].fill(222)
    a=next(p for p in plan["patches"] if p["lineage"]=="a")
    m=assemble_patch(a,task,images,diagnostic=True)
    assert (m["image"][m["first_source_stream"]==1]==11).all()
    assert len({p["patch_id"] for p in plan["patches"]})==len(plan["patches"])


def test_oracle_and_real_reference_identical_first_pipeline(tasks):
    task=next(t for t in tasks if t["family"]=="F7")
    world=np.zeros((2048,2048,3),np.uint8);world[:,:,0]=np.arange(2048)[None,:]%256
    plan=build_patch_plan([task],count=2)
    images={(stream,event):crop(world,q,task["origin"]) for stream,qs in _lineages(task).items() for event,q in enumerate(qs) if event}
    for patch in plan["patches"]:
        mosaic=assemble_patch(patch,task,images,diagnostic=True)
        x,y=patch["xy"]
        assert mosaic["complete"] and np.array_equal(mosaic["image"],world[y:y+128,x:x+128])


def test_unverified_submission_not_allowed(tasks):
    task=tasks[2];patch=build_patch_plan([task])["patches"][0]
    with pytest.raises(ValueError,match="verified"):assemble_patch(patch,task,images_for(task))


class DiagnosticMeanFeatures:
    feature_dimension=3
    def __init__(self,identity="test-mean-v1"):
        self.calls=0;self.manifest={"schema":"rswm-diagnostic-feature-backend-v1","provider":identity,
                                  "feature_dimension":3,"diagnostic_features":True,"release_validated":False}
        self.manifest["feature_identity_sha256"]=digest(self.manifest)
    def encode(self,images):
        self.calls+=1
        return np.stack([image.mean(axis=(0,1))/255 for image in images]).astype(np.float32)


def patch_fixture(value=100,complete=True):
    image=np.full((128,128,3),value,np.uint8);mask=np.ones((128,128),bool)
    if not complete:mask[0,0]=False
    return {"patch_id":"p","comparison_id":"p-s0","sample_id":"m-p-s0","world_id":"w","role":"diagnostic_only",
            "lineage":"main","model_id":"m","complete":complete,"planned_pixels":128**2,"valid_pixels":int(mask.sum()),
            "first_source_sha256":"a"*64,"image":image,"valid_mask":mask,
            "pixel_sha256":hashlib.sha256(image.tobytes()).hexdigest(),"mask_sha256":hashlib.sha256(mask.tobytes()).hexdigest()}


def test_missing_patch_never_encoded_as_black(tmp_path):
    backend=DiagnosticMeanFeatures()
    row=encode_patches([patch_fixture(0,False)],backend,tmp_path)[0]
    assert backend.calls==0 and row["feature"] is None and row["status"]=="MISSING_PATCH"
    assert row["diagnostic_features"] is True


def test_cache_changes_on_pixels_mask_sources_or_feature_weights(tmp_path):
    a=patch_fixture();backend=DiagnosticMeanFeatures();key=feature_cache_key(a,backend)
    assert feature_cache_key(patch_fixture(101),backend)!=key
    assert feature_cache_key(patch_fixture(100,False),backend)!=key
    b=copy.deepcopy(a);b["first_source_sha256"]="b"*64
    assert feature_cache_key(b,backend)!=key
    assert feature_cache_key(a,DiagnosticMeanFeatures("different-frozen-weight-sha"))!=key
    first=encode_patches([a],backend,tmp_path)[0];second=encode_patches([a],backend,tmp_path)[0]
    assert backend.calls==1 and np.array_equal(first["feature"],second["feature"])
    data=Path(first["feature_file"]);data.write_bytes(b"corrupt")
    with pytest.raises(ValueError,match="CACHE_INTEGRITY"):encode_patches([a],backend,tmp_path)


@pytest.mark.parametrize("kind",["shape","dtype","nan"])
def test_feature_dimension_dtype_and_nan_rejected(tmp_path,kind):
    class Bad(DiagnosticMeanFeatures):
        def encode(self,images):
            if kind=="shape":return np.zeros((1,4),np.float32)
            if kind=="dtype":return np.zeros((1,3),np.float64)
            return np.full((1,3),np.nan,np.float32)
    with pytest.raises(ValueError,match="SHAPE_OR_FINITE"):encode_patches([patch_fixture()],Bad(),tmp_path)


def test_multiple_uncached_patches_reach_provider_together_and_identical_cache_keys_deduplicate(tmp_path):
    class Batch(DiagnosticMeanFeatures):
        def encode(self,images):
            self.batch_length=len(images)
            return super().encode(images)
    backend=Batch();patches=[patch_fixture(1),patch_fixture(2),patch_fixture(1)]
    rows=encode_patches(patches,backend,tmp_path)
    assert backend.calls==1 and backend.batch_length==2
    assert len(rows)==3 and np.array_equal(rows[0]["feature"],rows[2]["feature"])


def feature_config(path):
    return {"schema":FEATURE_SCHEMA,"backend":"torchscript-local-v1","model_repo":"local/frozen-candidate",
            "model_revision":"sha-locked-v1","output_layer":"serialized_module_output","feature_dimension":3,
            "weights":{"path":str(path),"sha256":"a"*64},"dtype":"float32","color_input":"RGB","input_shape":[128,128,3],
            "device":"cpu","batch_size":8,"output_reduction":"none",
            "preprocessing":{"resize":[128,128],"center_crop":None,"resample":"bilinear","scale":"uint8/255","mean":[0,0,0],"std":[1,1,1]}}


def test_missing_weights_fail_preflight_without_random_fallback(tmp_path):
    with pytest.raises(FeatureUnavailable,match="WEIGHTS_NOT_AVAILABLE"):load_backend(feature_config(tmp_path/"missing.pt"))
    path=tmp_path/"weights.pt";path.write_bytes(b"not a model")
    with pytest.raises(ValueError,match="HASH_MISMATCH"):validate_feature_config(feature_config(path))


def test_actual_feature_loader_with_local_diagnostic_torchscript(tmp_path):
    torch=pytest.importorskip("torch")
    class Tiny(torch.nn.Module):
        def forward(self,x):return x.mean(dim=(2,3))
    path=tmp_path/"diagnostic.pt";torch.jit.trace(Tiny().eval(),torch.zeros(1,3,128,128)).save(str(path))
    config=feature_config(path);config["weights"]["sha256"]=hashlib.sha256(path.read_bytes()).hexdigest();config["diagnostic_features"]=True
    backend=load_backend(config)
    features=backend.encode([patch_fixture(255)["image"]])
    assert features.shape==(1,3) and np.allclose(features,1)
    assert backend.manifest["diagnostic_features"] is True and backend.manifest["release_validated"] is False


def feature_row(world,value,model="m",complete=True):
    return {"world_id":world,"source_identity":world,"city":"sfo","role":"diagnostic_only","study":"core","family":"F5",
            "protocol":"offline","variant":"exact","lineage":"main","model_id":model,"patch_id":world,
            "comparison_id":world+"-s0","sample_id":model+"-"+world,"planned_pixels":128**2,"valid_pixels":128**2 if complete else 0,
            "complete":complete,"status":"ENCODED" if complete else "MISSING_PATCH","feature_identity_sha256":"a"*64,
            "feature":np.array([value],np.float32) if complete else None,"diagnostic_features":True}


def test_negative_u_is_preserved_and_world_bootstrap_reproducible():
    x=[feature_row("w0",0),feature_row("w1",1)]
    y=[feature_row("w0",0,"reference"),feature_row("w1",1,"reference")]
    report=evaluate_distribution(x,y,bootstrap=30,seed=7)
    assert report["polynomial_mmd_u"]==-3.5
    assert report==evaluate_distribution(x,y,bootstrap=30,seed=7)
    assert report["is_inception_kid"] is False and report["semantic_accuracy_claim"] is False
    assert report["world_bootstrap_95_sensitivity_interval"] is not None


def test_insufficient_samples_not_zero_and_world_missing_denominator():
    x=[feature_row("w0",0),feature_row("w1",1,complete=False)]
    y=[feature_row("w0",0,"ref"),feature_row("w1",1,"ref")]
    report=evaluate_distribution(x,y,bootstrap=5)
    assert report["polynomial_mmd_u"] is None and report["status"]=="INSUFFICIENT_SAMPLES"
    assert report["generated_coverage"]["planned_worlds"]==2 and report["generated_coverage"]["missing_pixels"]==128**2
    assert report["world_bootstrap_95_sensitivity_interval"] is None


def test_real_real_sources_must_be_disjoint():
    with pytest.raises(ValueError,match="DISJOINT"):
        evaluate_distribution([feature_row("w0",0)],[feature_row("w0",0,"ref")],kind="real-real",bootstrap=5)


def test_common_subset_does_not_hide_full_own_coverage():
    a=[feature_row("w0",0,"a"),feature_row("w1",1,"a",False)]
    b=[feature_row("w0",0,"b"),feature_row("w1",1,"b")]
    refs=[feature_row("w0",0,"ref"),feature_row("w1",1,"ref")]
    result=evaluate_methods(a+b,refs,bootstrap=5)
    assert all(r["common_comparison_units"]==1 for r in result)
    assert result[0]["own_complete_sample_diagnostic"]["generated_coverage"]["planned_tiles"]==2
    assert result[0]["own_complete_sample_diagnostic"]["generated_coverage"]["complete_tiles"]==1


def test_feature_bundle_preserves_diagnostic_flag_and_integrity(tmp_path):
    backend=DiagnosticMeanFeatures();rows=encode_patches([patch_fixture()],backend,tmp_path/"cache")
    saved=save_feature_bundle(rows,backend.manifest,tmp_path/"features","b"*64)
    bundle,loaded=load_feature_bundle(tmp_path/"features"/"features.json")
    assert bundle==saved and bundle["diagnostic_features"] is True and loaded[0]["feature"].shape==(3,)
    data=tmp_path/"features"/bundle["features"][0]["feature_file"];data.write_bytes(b"bad")
    with pytest.raises(ValueError,match="FILE_INTEGRITY"):load_feature_bundle(tmp_path/"features"/"features.json")


def test_extraction_cli_blocks_absent_weights_before_mosaic_read(tmp_path):
    config=tmp_path/"config.json";config.write_text(json.dumps(feature_config(tmp_path/"missing.pt")))
    repo=Path(__file__).resolve().parents[1]
    proc=subprocess.run([sys.executable,str(repo/"scripts"/"extract_features.py"),"--config",str(config),"--mosaics",str(tmp_path/"nonexistent.json"),
                         "--cache",str(tmp_path/"cache"),"--out",str(tmp_path/"output")],capture_output=True,text=True,timeout=30)
    assert proc.returncode==1 and '"status":"BLOCKED"' in proc.stdout and "WEIGHTS_NOT_AVAILABLE" in proc.stdout
    assert not (tmp_path/"output"/"features.json").exists()


def test_cache_metadata_cannot_change_feature_identity_or_diagnostic_flag(tmp_path):
    backend=DiagnosticMeanFeatures();row=encode_patches([patch_fixture()],backend,tmp_path)[0]
    meta=Path(row["feature_file"]).with_suffix(".json");saved=json.loads(meta.read_text())
    for field,value in [("diagnostic_features",False),("feature_dimension",999),("feature_identity_sha256","0"*64)]:
        bad={**saved,field:value};meta.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match="CACHE_INTEGRITY"):encode_patches([patch_fixture()],backend,tmp_path)


@pytest.mark.parametrize("place",["bundle","row","backend"])
def test_diagnostic_feature_bundle_cannot_be_promoted_by_rehashing(tmp_path,place):
    backend=DiagnosticMeanFeatures();rows=encode_patches([patch_fixture()],backend,tmp_path/"cache")
    saved=save_feature_bundle(rows,backend.manifest,tmp_path/"features","b"*64)
    bad=copy.deepcopy(saved)
    if place=="bundle":bad["diagnostic_features"]=False
    if place=="row":bad["features"][0]["diagnostic_features"]=False
    if place=="backend":
        bad["backend_manifest"]["diagnostic_features"]=False
        bad["backend_manifest"]["feature_identity_sha256"]=digest({k:v for k,v in bad["backend_manifest"].items() if k!="feature_identity_sha256"})
    bad["feature_bundle_sha256"]=digest({k:v for k,v in bad.items() if k!="feature_bundle_sha256"})
    path=tmp_path/"features"/"features.json";path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match="DIAGNOSTIC|REAL_BACKEND"):load_feature_bundle(path)


def test_source_world_aliases_are_one_bootstrap_unit_and_cannot_be_real_real():
    x=[feature_row("alias0",0),feature_row("alias1",1)]
    for row in x:row["source_identity"]="same-actual-world"
    refs=[{**row,"model_id":"ref","sample_id":"ref-"+row["sample_id"]} for row in x]
    report=evaluate_distribution(x,refs,bootstrap=5,cap=1)
    assert report["generated_worlds"]==1 and report["generated_n"]==1
    assert report["world_bootstrap_95_sensitivity_interval"] is None
    with pytest.raises(ValueError,match="DISJOINT"):evaluate_distribution(x,[{**refs[0],"world_id":"other-alias"}],kind="real-real",bootstrap=5)


def test_real_real_bootstrap_resamples_each_independent_population():
    x=[feature_row("x0",0),feature_row("x1",1)]
    y=[feature_row("y0",2,"ref"),feature_row("y1",3,"ref")]
    report=evaluate_distribution(x,y,kind="real-real",bootstrap=20)
    assert report["bootstrap_pairing"]=="independent populations"
    assert report["valid_bootstrap_replicates"]==20 and report["reference_worlds"]==2


def test_feature_configuration_rejects_ignored_preprocessing_fields(tmp_path):
    config=feature_config(tmp_path/"weights.pt");config["preprocessing"]["resize_after_crop"]=True
    with pytest.raises(ValueError,match="PREPROCESSING_FIELDS"):validate_feature_config(config)


def test_verified_suite_png_to_frozen_mosaics_features_and_cli_statistics(tmp_path,monkeypatch):
    from PIL import Image
    from rswm_inria import suite,features,quality_protocol,quality_evaluation
    items=[]
    for index,value in enumerate((20,200)):
        world_id=f"quality-world-{index}";world=np.full((2048,2048,3),value,np.uint8)
        world[:,:,0]=(np.arange(2048,dtype=np.uint16)[None,:]+value)%256
        wp=tmp_path/(world_id+".png");Image.fromarray(world).save(wp)
        task=next(t for t,_ in packages(world_id,"diagnostic_only",0) if t["family"]=="F2")
        tp=tmp_path/(world_id+".json");tp.write_text(json.dumps(task))
        items.append({"task_path":str(tp),"world_path":str(wp),"city":"sfo"})
    plan=suite.build_plan(items,[{"model_id":"oracle","adapter":"oracle"},{"model_id":"copy","adapter":"copy"}],
                          [0],"a"*64,"b"*40)
    outcomes=suite.execute_plan(plan,tmp_path/"suite")
    assert len(outcomes)==4 and all(o["status"]=="COMPLETED" for o in outcomes)
    pp=tmp_path/"plan.json";pp.write_text(json.dumps(plan))
    patch_plan=tmp_path/"patch-plan.json"
    assert quality_protocol.main(["plan","--suite-plan",str(pp),"--count","2","--out",str(patch_plan)])==0
    frozen=json.loads(patch_plan.read_text());assert frozen["selection_depends_on_images"] is False
    assert all(p["source_identity"].startswith("world-pixels-sha256:") for p in frozen["patches"])
    for command,name in [("mosaics","generated"),("reference","reference")]:
        args=[command,"--suite-plan",str(pp),"--patch-plan",str(patch_plan),"--out",str(tmp_path/name)]
        if command=="mosaics":args.extend(["--suite",str(tmp_path/"suite")])
        assert quality_protocol.main(args)==0
    reference_bundle,refs=load_mosaics(tmp_path/"reference"/"mosaics.json")
    generated_bundle,generated=load_mosaics(tmp_path/"generated"/"mosaics.json")
    assert generated_bundle["planned_tiles"]==8 and reference_bundle["planned_tiles"]==4
    assert all(r["reference_kind"]=="diagnostic-world" for r in refs)
    reference_by_patch={r["patch_id"]:r for r in refs}
    assert all(np.array_equal(r["image"],reference_by_patch[r["patch_id"]]["image"]) for r in generated if r["model_id"]=="oracle")
    config=tmp_path/"test-config.json";config.write_text(json.dumps({"test_only_diagnostic_provider":True}))
    # This mock injection exists only inside a test; the installed extraction
    # command has no diagnostic fallback or arbitrary provider import option.
    monkeypatch.setattr(features,"load_backend",lambda *_:DiagnosticMeanFeatures())
    for name in ("generated","reference"):
        assert features.main(["--config",str(config),"--mosaics",str(tmp_path/name/"mosaics.json"),"--cache",str(tmp_path/"cache"),
                              "--out",str(tmp_path/(name+"-features"))])==0
    out=tmp_path/"evaluation.json"
    assert quality_evaluation.main(["--generated",str(tmp_path/"generated-features"/"features.json"),
                                   "--reference",str(tmp_path/"reference-features"/"features.json"),"--out",str(out),
                                   "--cap","1","--bootstrap","7","--seed","11"])==0
    report=json.loads(out.read_text())
    assert report["diagnostic_features"] is True and report["release_validated"] is False
    assert report["statistical_config"]=={"schema":"rswm-quality-statistical-protocol-v1","kind":"generated-real",
                                         "cap":1,"bootstrap":7,"seed":11,"kernel":"(x.y/d+1)^3 unbiased U"}
    assert all(r["own_complete_sample_diagnostic"]["planned_bootstrap_replicates"]==7 for r in report["reports"])
    validation=tmp_path/"control-validation.json"
    assert features.validation_main(["--config",str(config),"--reference-mosaics",str(tmp_path/"reference"/"mosaics.json"),
                                     "--cache",str(tmp_path/"control-cache"),"--out",str(validation),"--sample-caps","1","2",
                                     "--bootstrap","3","--seed","7"])==0
    controls=json.loads(validation.read_text())
    assert len(controls["reports"])==14 and controls["real_validation"]=="NOT_RUN" and controls["diagnostic_features"] is True
    assert {r["control_name"] for r in controls["reports"]}=={"oracle_reference","fixed_noise_map","tiled","blur",
                                                             "color_perturbation","artificial_seam","independent_real_real"}
    assert controls["sample_caps"]==[1,2] and controls["planned_reference_tiles"]==4
    assert all(r["distribution"]["is_inception_kid"] is False for r in controls["reports"])
    with pytest.raises(ValueError,match="CALIBRATION_OR_VALIDATION_ROLE"):
        features.validate_feature_controls([{**refs[0],"role":"test_unseen_city"}],DiagnosticMeanFeatures(),tmp_path/"rejected")
    # The source wrapper delegates to the same callable; its concrete parameters
    # are exercised by this child process, without a mocked statistics pipeline.
    repo=Path(__file__).resolve().parents[1]
    child_out=tmp_path/"evaluation-script.json"
    child=subprocess.run([sys.executable,str(repo/"scripts"/"evaluate_quality.py"),"--generated",str(tmp_path/"generated-features"/"features.json"),
                          "--reference",str(tmp_path/"reference-features"/"features.json"),"--out",str(child_out),
                          "--cap","1","--bootstrap","7","--seed","11"],capture_output=True,text=True,timeout=30)
    assert child.returncode==0,child.stdout+child.stderr
    assert json.loads(child_out.read_text())==report
    denied=tmp_path/"unvalidated-formal-evaluation.json"
    assert quality_evaluation.main(["--generated",str(tmp_path/"generated-features"/"features.json"),
                                    "--reference",str(tmp_path/"reference-features"/"features.json"),"--out",str(denied),
                                    "--feature-config-sha256","a"*64])==2
    assert not denied.exists()


def test_validation_cli_blocks_absent_weights_without_feature_fallback(tmp_path):
    config=tmp_path/"config.json";config.write_text(json.dumps(feature_config(tmp_path/"missing.pt")))
    repo=Path(__file__).resolve().parents[1]
    proc=subprocess.run([sys.executable,str(repo/"scripts"/"validate_features.py"),"--config",str(config),
                         "--reference-mosaics",str(tmp_path/"nonexistent.json"),"--cache",str(tmp_path/"cache"),
                         "--out",str(tmp_path/"validation.json"),"--sample-caps","1","2","--bootstrap","3","--seed","7"],
                        capture_output=True,text=True,timeout=30)
    assert proc.returncode==1 and '"status":"BLOCKED"' in proc.stdout and "WEIGHTS_NOT_AVAILABLE" in proc.stdout
    assert not (tmp_path/"validation.json").exists()


def acceptance_fixture(feature_config_sha256="a"*64,caps=(1,2)):
    """Unit-test contract bounds, never a real acceptance decision."""
    config={"schema":feature_release.ACCEPTANCE_SCHEMA,"feature_config_sha256":feature_config_sha256,
            "source_manifest_sha256":"b"*64,"calibration_reference_bundle_sha256":"c"*64,
            "calibration_patch_plan_sha256":"d"*64,"validation_patch_plan_sha256":"e"*64,
            "statistics":{"sample_caps":list(caps),"bootstrap":3,"seed":7},
            "primary_sample_cap":caps[-1],
            "min_complete_tile_coverage":1,"min_valid_pixel_coverage":1,"min_real_real_source_worlds_per_group":2,
            "control_ranges":[{"control_name":name,"sample_cap":cap,"minimum":-1000000,"maximum":1000000}
                              for name in feature_release.CONTROLS for cap in caps],
            "interpretation":"frozen-feature-distribution-diagnostic-only",
            "review_decision":{"reviewer":"unit-test-fixture","decision_reference":"test-only-not-approved-for-release","decision_sha256":"f"*64}}
    return feature_release._sealed(config,"acceptance_sha256")


def population_fixture(role="metric_validation",namespace="unit-test-synthetic"):
    city="bloomington" if role=="metric_validation" else "bellingham"
    return [{"world_id":namespace+str(i),"role":role,"city":city,"source_path":namespace+str(i)+".tif",
             **{key:digest([namespace,i,key]) for key in ("world_file_sha256","world_pixel_sha256","source_sha256")}}
            for i in range(12)]


def test_acceptance_contract_requires_reviewed_complete_bounds_and_freezes_statistical_identity():
    config=acceptance_fixture();assert feature_release.validate_acceptance_config(config)==config
    for change in ("missing_range","missing_review","nan","missing_validation_geometry"):
        bad=copy.deepcopy(config);bad.pop("acceptance_sha256")
        if change=="missing_range":bad["control_ranges"].pop()
        if change=="missing_review":bad["review_decision"]["decision_reference"]=""
        if change=="nan":bad["control_ranges"][0]["maximum"]=float("nan")
        if change=="missing_validation_geometry":bad.pop("validation_patch_plan_sha256")
        if change=="nan":
            # Canonical serialization itself forbids nonfinite preregistration.
            with pytest.raises(ValueError):feature_release._sealed(bad,"acceptance_sha256")
        else:
            bad=feature_release._sealed(bad,"acceptance_sha256")
            with pytest.raises(ValueError):feature_release.validate_acceptance_config(bad)


def test_reviewed_ranges_execute_against_actual_diagnostic_controls_without_promoting_them(tmp_path):
    from rswm_inria.features import validate_feature_controls
    backend=DiagnosticMeanFeatures();backend.manifest["config_sha256"]="a"*64
    backend.manifest["feature_identity_sha256"]=digest({k:v for k,v in backend.manifest.items() if k!="feature_identity_sha256"})
    patches=[]
    for i,value in enumerate((10,70,140,230)):
        row=patch_fixture(value)
        row.update(world_id=f"fixture-world-{i}",source_identity=f"fixture-source-{i}",role="metric_calibration",city="bellingham",
                   xy=[128*i,0],sample_id=f"fixture-sample-{i}",comparison_id=f"fixture-comparison-{i}",reference_kind="diagnostic-world")
        patches.append(row)
    report=validate_feature_controls(patches,backend,tmp_path/"cache",sample_caps=(1,2),bootstrap=3,seed=7)
    config=acceptance_fixture();result=feature_release._acceptance(report,config,"metric_calibration")
    assert result["acceptance_passed"] and report["diagnostic_features"] is True and report["real_validation"]=="NOT_RUN"
    assert report["release_validated"] is False
    bad=copy.deepcopy(config);bad.pop("acceptance_sha256")
    observed=report["reports"][0];limit=next(r for r in bad["control_ranges"] if r["control_name"]==observed["control_name"] and r["sample_cap"]==observed["sample_cap"])
    limit.update(minimum=observed["distribution"]["polynomial_mmd_u"]+1,maximum=observed["distribution"]["polynomial_mmd_u"]+2)
    denied=feature_release._acceptance(report,feature_release._sealed(bad,"acceptance_sha256"),"metric_calibration")
    assert not denied["acceptance_passed"] and denied["reason_codes"]


def test_release_population_rejects_role_city_or_original_source_duplicates():
    worlds=population_fixture();assert feature_release._population(worlds,"metric_validation")==worlds
    for key,value in [("role","metric_calibration"),("city","bellingham"),("source_sha256",worlds[1]["source_sha256"])]:
        bad=copy.deepcopy(worlds);bad[0][key]=value
        with pytest.raises(ValueError):feature_release._population(bad,"metric_validation")


def test_heldout_exposure_tracks_original_source_aliases_configuration_acceptance_and_calibration(tmp_path):
    config=acceptance_fixture();worlds=population_fixture();cal={"calibration_sha256":"0"*64};bundle={"bundle_sha256":"1"*64}
    path=tmp_path/"exposure.json"
    first=feature_release._register_exposure(worlds,config,cal,bundle,path)
    assert not first["previously_viewed"] and not first["used_for_new_config"] and path.exists()
    repeated=feature_release._register_exposure(worlds,config,cal,bundle,path)
    assert repeated["previously_viewed"] and not repeated["used_for_new_config"]
    aliases=[{**w,"world_id":"alias-"+w["world_id"]} for w in worlds]
    changed={**config,"acceptance_sha256":"2"*64}
    again=feature_release._register_exposure(aliases,changed,cal,{"bundle_sha256":"3"*64},path)
    assert again["previously_viewed"] and again["used_for_new_config"]
    changed_cal=feature_release._register_exposure(worlds,config,{"calibration_sha256":"4"*64},bundle,path)
    assert changed_cal["used_for_new_config"]
    ledger=json.loads(path.read_text());ledger["exposures"][0]["source_population_keys"]=[];path.write_text(json.dumps(ledger))
    with pytest.raises(ValueError,match="EXPOSURE_ENTRY"):feature_release._register_exposure(worlds,config,cal,bundle,path)


@pytest.mark.parametrize("field,value",[("release_validated",False),("real_validation_executed",False),("source_kind","synthetic_fixture"),
                                         ("role","metric_calibration"),("feature_config_sha256","0"*64)])
def test_thin_or_diagnostic_quality_receipts_cannot_authorize_formal_release(field,value):
    receipt={"schema":feature_release.RELEASE_SCHEMA,"instrument":"quality","role":"metric_validation","status":"VALIDATED",
             "release_validated":True,"real_validation_executed":True,"source_kind":"canonical_inria",
             "feature_config_sha256":"a"*64,"source_manifest_sha256":"b"*64,field:value}
    receipt=feature_release._sealed(receipt,"validation_sha256")
    from rswm_inria.features import validate_feature_release
    with pytest.raises(ValueError,match="BLOCKED_FEATURE"):validate_feature_release(receipt,"a"*64,"b"*64)


def test_even_all_green_thin_receipt_is_rejected_by_public_release_api():
    from rswm_inria.features import validate_feature_release
    thin={"schema":feature_release.RELEASE_SCHEMA,"instrument":"quality","role":"metric_validation","status":"VALIDATED",
          "release_validated":True,"real_validation_executed":True,"source_kind":"canonical_inria",
          "feature_config_sha256":"a"*64,"source_manifest_sha256":"b"*64}
    with pytest.raises(ValueError,match="INCOMPLETE"):
        validate_feature_release(feature_release._sealed(thin,"validation_sha256"),"a"*64,"b"*64)


def test_release_cli_real_weight_preflight_is_blocked_before_source_or_image_read(tmp_path):
    config=feature_config(tmp_path/"missing.pt");cp=tmp_path/"config.json";cp.write_text(json.dumps(config))
    ap=tmp_path/"acceptance.json";ap.write_text(json.dumps(acceptance_fixture(digest(config))))
    repo=Path(__file__).resolve().parents[1];out=tmp_path/"release.json"
    proc=subprocess.run([sys.executable,str(repo/"scripts"/"release_features.py"),"validate","--config",str(cp),"--acceptance-config",str(ap),
                         "--reference-mosaics",str(tmp_path/"missing-mosaic.json"),"--patch-plan",str(tmp_path/"missing-patches.json"),
                         "--suite-plan",str(tmp_path/"missing-plan.json"),"--source-audit",str(tmp_path/"missing-audit.json"),
                         "--cache",str(tmp_path/"cache"),"--out",str(out)],capture_output=True,text=True,timeout=30)
    assert proc.returncode==1 and "FEATURE_WEIGHTS_NOT_AVAILABLE" in proc.stdout and '"status":"BLOCKED"' in proc.stdout
    assert not out.exists()
