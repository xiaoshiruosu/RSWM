"""Synthetic engineering controls; these never certify real-scene calibration."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
import pytest
pytest.importorskip("scipy")
from rswm_inria.geometry import crop, digest, file_hash, rng_for
from rswm_inria import motion_protocol as m


@pytest.fixture(scope="module")
def textured_world():
    return rng_for("motion-protocol-synthetic").integers(0,256,(2048,2048,3),dtype=np.uint8)


@pytest.fixture(scope="module")
def materialized_pairs(tmp_path_factory):
    root=tmp_path_factory.mktemp("motion-pair-fixtures")
    manifests={}
    for role,city in m.ROLES.items():
        world_root=root/city;world_root.mkdir()
        world=rng_for("motion-"+city).integers(0,256,(2048,2048,3),dtype=np.uint8)
        path=world_root/"world.png";Image.fromarray(world).save(path)
        row={"world_id":city+"-synthetic", "role":role,"city":city,"path":"world.png",
             "world_file_sha256":file_hash(path),"world_pixel_sha256":hashlib.sha256(world.tobytes()).hexdigest()}
        index={"source_manifest_sha256":"a"*64,"worlds":[row]}
        pairs_root=root/(city+"-pairs")
        manifest=m.build_pairs(index,world_root,pairs_root,role,pairs_per_world=8,diagnostic_fixture=True)
        manifests[role]=(manifest,pairs_root,index,world_root)
    return root,manifests


@pytest.mark.parametrize("delta", [[24,-16],[-32,16],[0,0],[48,0],[0,-32]])
def test_image_only_sign_and_magnitude(textured_world,delta):
    a=crop(textured_world,[0,0],[1024,1024]);b=crop(textured_world,delta,[1024,1024])
    result=m.estimate_pair(a,b,m.motion_config())
    assert result["status"]=="measured"
    assert result["viewport_translation_xy"]==delta
    assert result["content_translation_xy"]==[-x for x in delta]
    assert list(inspect.signature(m.estimate_pair).parameters)==["previous","current","config"]


@pytest.mark.parametrize("case", ["missing","low_texture","periodic","wrong_grid"])
def test_abstention_and_noncanonical_grid(case,textured_world):
    image=crop(textured_world,[0,0],[1024,1024]);other=image
    if case=="missing":other=None
    if case=="low_texture":image=other=np.full((240,416,3),128,np.uint8)
    if case=="periodic":
        yy,xx=np.mgrid[:240,:416];image=other=np.repeat((((xx//4+yy//4)%2)*255).astype(np.uint8)[:,:,None],3,axis=2)
    if case=="wrong_grid":other=image[:64,:64]
    result=m.estimate_pair(image,other,m.motion_config())
    assert result["status"]=={"missing":"submission_failure","low_texture":"insufficient_evidence","periodic":"ambiguous","wrong_grid":"out_of_domain"}[case]
    assert result["viewport_translation_xy"] is None
    assert result["content_translation_xy"] is None


@pytest.mark.parametrize("min_overlap,axis,bound", [
    (.65, 0, 145), (.65, 1, 83), (.55, 0, 187), (.55, 1, 108),
])
@pytest.mark.parametrize("sign", [-1, 1])
def test_fixed_domain_boundary_abstains(textured_world,min_overlap,axis,bound,sign):
    parameters=m.motion_config()["parameters"].copy();parameters["min_overlap"]=min_overlap
    config=m.motion_config(parameters);domain=m.search_domain(config);delta=[0,0]
    delta[axis]=sign*bound
    assert domain[("dx_bounds", "dy_bounds")[axis]]==[-bound,bound]
    result=m.estimate_pair(crop(textured_world,[0,0],[1024,1024]),crop(textured_world,delta,[1024,1024]),config)
    assert result["status"]=="out_of_domain" and result["boundary_hit"] is True
    assert result["candidate_viewport_translation_xy"]==delta
    assert result["viewport_translation_xy"] is None


@pytest.mark.parametrize("min_overlap,bounds", [
    (.55, (187, 108)), (.60, (166, 96)), (.65, (145, 83)),
    (1e-300, (415, 239)), (.9999999999999999, (0, 0)), (1., (0, 0)),
])
def test_search_domain_bounds_match_integer_overlap_mask(min_overlap,bounds):
    parameters=m.motion_config()["parameters"].copy();parameters["min_overlap"]=min_overlap
    config=m.motion_config(parameters);domain=m.search_domain(config)
    assert domain["dx_bounds"]==[-bounds[0],bounds[0]]
    assert domain["dy_bounds"]==[-bounds[1],bounds[1]]
    # The solver searches this finite linear grid and uses this multiplication order.
    yy,xx=np.meshgrid(np.arange(1-m.HEIGHT,m.HEIGHT),np.arange(1-m.WIDTH,m.WIDTH),indexing="ij")
    overlap_pixels=(m.WIDTH-np.abs(xx))*(m.HEIGHT-np.abs(yy))
    allowed=overlap_pixels>=min_overlap*m.WIDTH*m.HEIGHT
    assert [int(xx[allowed].min()),int(xx[allowed].max())]==domain["dx_bounds"]
    assert [int(yy[allowed].min()),int(yy[allowed].max())]==domain["dy_bounds"]
    for axis,bound in enumerate(bounds):
        for sign in (-1,1):
            edge=[0,0];edge[axis]=sign*bound
            beyond=[0,0];beyond[axis]=sign*(bound+1)
            assert m._in_domain(edge,config)
            assert not m._in_domain(beyond,config)


@pytest.mark.parametrize("min_overlap,inside,outside", [
    (.55, [(187,0),(0,108),(104,64)], [(188,0),(0,109),(105,64),(187,108)]),
    (.65, [(145,0),(0,83),(91,40)], [(146,0),(0,84),(92,40),(145,83)]),
])
def test_search_domain_requires_overlap_predicate_at_corners(min_overlap,inside,outside):
    parameters=m.motion_config()["parameters"].copy();parameters["min_overlap"]=min_overlap
    config=m.motion_config(parameters)
    # Axis bounds describe a bounding box, not a rectangular search domain.
    for sx in (-1,1):
        for sy in (-1,1):
            for dx,dy in inside:
                assert m._in_domain([sx*dx,sy*dy],config)
            for dx,dy in outside:
                assert not m._in_domain([sx*dx,sy*dy],config)


@pytest.mark.parametrize("field,value", [("min_overlap",float("nan")),("min_std",True),("min_peak_gap",-1),("peak_exclusion_radius",1.5)])
def test_strict_config(field,value):
    parameters=m.motion_config()["parameters"].copy();parameters[field]=value
    with pytest.raises(ValueError):m.motion_config(parameters)


def test_hash_and_acceptance_tampering_rejected():
    config=m.motion_config();config["parameters"]["min_correlation"]+=.01
    with pytest.raises(ValueError,match="hash"):m.validate_config(config)
    with pytest.raises(ValueError):m.motion_config(acceptance={"pixel_tolerance":1})


def test_pair_builder_fixed_sampling_roles_and_no_source_mutation(materialized_pairs):
    _,bundles=materialized_pairs
    manifest,root,index,world_root=bundles["metric_calibration"]
    assert manifest["source_kind"]=="synthetic_fixture"
    assert len(manifest["pairs"])==8
    assert manifest["sampling"]["controls"]==list(m.CONTROLS)
    assert all((root/row["images"]["previous"]).is_file() for row in manifest["pairs"])
    assert file_hash(world_root/"world.png")==index["worlds"][0]["world_file_sha256"]
    with pytest.raises(FileExistsError):m.build_pairs(index,world_root,root,"metric_calibration",8,diagnostic_fixture=True)
    with pytest.raises(ValueError,match="twelve"):m.build_pairs(index,world_root,root.parent/"real","metric_calibration",8)


def test_all_perturbations_are_materialized_and_missing_slot_is_explicit(tmp_path,materialized_pairs):
    _,bundles=materialized_pairs
    _,_,index,world_root=bundles["metric_calibration"]
    manifest=m.build_pairs(index,world_root,tmp_path/"all-controls","metric_calibration",len(m.CONTROLS),diagnostic_fixture=True)
    assert {row["control"] for row in manifest["pairs"]}==set(m.CONTROLS)
    missing=next(row for row in manifest["pairs"] if row["control"]=="missing_current")
    assert missing["images"]["current"] is None and missing["image_sha256"]["current"] is None
    analysis=m.analyze_pairs(manifest,tmp_path/"all-controls",m.motion_config())
    assert analysis["planned_pairs"]==len(m.CONTROLS)
    assert analysis["status_counts"]["submission_failure"]==1
    assert all(row["command_error"] is None for row in analysis["pairs"] if row["status"]!="measured")


def test_role_and_pair_plan_cannot_be_rehashed_to_different_meaning(materialized_pairs):
    _,bundles=materialized_pairs
    manifest,root,_,_=bundles["metric_calibration"]
    changed=copy.deepcopy(manifest);changed["source_worlds"][0]["role"]="metric_validation"
    changed["pair_manifest_sha256"]=digest({k:v for k,v in changed.items() if k!="pair_manifest_sha256"})
    with pytest.raises(ValueError,match="role/city"):m.validate_pair_manifest(changed)
    changed=copy.deepcopy(manifest);changed["pairs"][0]["reference_displacement"][0]+=1
    changed["pair_manifest_sha256"]=digest({k:v for k,v in changed.items() if k!="pair_manifest_sha256"})
    with pytest.raises(ValueError,match="preregistered"):m.analyze_pairs(changed,root,m.motion_config())
    with pytest.raises(ValueError,match="role"):m.calibrate(bundles["metric_validation"][0],bundles["metric_validation"][1],[m.motion_config()],root/"bad")


def test_pair_hash_and_symlink_damage_rejected(tmp_path,materialized_pairs):
    _,bundles=materialized_pairs
    manifest,_,index,world_root=bundles["metric_calibration"]
    local=m.build_pairs(index,world_root,tmp_path/"pairs","metric_calibration",1,diagnostic_fixture=True)
    image=tmp_path/"pairs"/local["pairs"][0]["images"]["current"]
    image.write_bytes(b"invalid")
    with pytest.raises(ValueError,match="hash"):m.analyze_pairs(local,tmp_path/"pairs",m.motion_config())
    changed=copy.deepcopy(manifest);changed["pairs"][0]["images"]["current"]="../escape.png"
    changed["pair_manifest_sha256"]=digest({k:v for k,v in changed.items() if k!="pair_manifest_sha256"})
    with pytest.raises(ValueError,match="escapes"):m.analyze_pairs(changed,bundles["metric_calibration"][1],m.motion_config())


def test_calibration_freezes_candidates_and_synthetic_never_claims_real(tmp_path,materialized_pairs):
    _,bundles=materialized_pairs
    manifest,root,_,_=bundles["metric_calibration"]
    config=m.motion_config()
    receipt=m.calibrate(manifest,root,[config],tmp_path/"calibration")
    assert receipt["status"]=="DIAGNOSTIC_ONLY" and receipt["release_validated"] is False
    assert receipt["selected_config"]==config and receipt["real_calibration_executed"] is False
    assert receipt["candidate_table"][0]["acceptance_passed"] is True
    assert (tmp_path/"calibration"/"candidate-results.json").is_file()
    with pytest.raises(ValueError,match="Duplicate"):m.calibrate(manifest,root,[config,config],tmp_path/"duplicate")
    other=m.motion_config(acceptance={**m.DEFAULT_ACCEPTANCE,"pixel_tolerance":2})
    with pytest.raises(ValueError,match="fixed"):m.calibrate(manifest,root,[config,other],tmp_path/"changed-rule")


def test_heldout_validation_and_exposure_prevent_false_independence(tmp_path,materialized_pairs):
    _,bundles=materialized_pairs
    cal,calroot,_,_=bundles["metric_calibration"];val,valroot,_,_=bundles["metric_validation"]
    config=m.motion_config();receipt=m.calibrate(cal,calroot,[config],tmp_path/"cal")
    ledger=tmp_path/"exposures.json"
    result=m.validate_motion(val,valroot,receipt,tmp_path/"v1",ledger,diagnostic_fixture=True)
    assert result["status"]=="DIAGNOSTIC_ONLY" and result["real_validation_executed"] is False
    assert result["source_manifest_sha256"]==val["source_manifest_sha256"]
    assert result["validation_previously_viewed"] is False
    assert result["report"]["risk_coverage"]
    assert all(row["planned_pairs"]==8 for row in result["report"]["worlds"])
    repeat=m.validate_motion(val,valroot,receipt,tmp_path/"repeat",ledger,diagnostic_fixture=True)
    assert repeat["validation_previously_viewed"] is True and repeat["validation_used_for_new_config"] is False
    changed=m.motion_config({**config["parameters"],"min_correlation":.8})
    newcal=m.calibrate(cal,calroot,[changed],tmp_path/"cal2")
    revisited=m.validate_motion(val,valroot,newcal,tmp_path/"v2",ledger,diagnostic_fixture=True)
    assert revisited["validation_used_for_new_config"] is True
    assert "VALIDATION_ALREADY_VIEWED_WITH_DIFFERENT_CONFIG" in revisited["reason_codes"]
    with pytest.raises(ValueError,match="Real validation"):m.validate_motion(val,valroot,receipt,tmp_path/"illegal",ledger)
    with pytest.raises(ValueError):m.validate_release(result,config)


def test_new_pair_seed_does_not_erase_viewed_validation_sources(tmp_path,materialized_pairs):
    _,bundles=materialized_pairs
    cal,calroot,_,_=bundles["metric_calibration"]
    val,valroot,index,world_root=bundles["metric_validation"]
    first=m.calibrate(cal,calroot,[m.motion_config()],tmp_path/"cal")
    ledger=tmp_path/"exposures.json"
    m.validate_motion(val,valroot,first,tmp_path/"first",ledger,diagnostic_fixture=True)
    newroot=tmp_path/"new-pairs"
    newpairs=m.build_pairs(index,world_root,newroot,"metric_validation",8,seed=37,diagnostic_fixture=True)
    config=m.motion_config({**m.motion_config()["parameters"],"min_correlation":.8})
    second=m.calibrate(cal,calroot,[config],tmp_path/"cal2")
    result=m.validate_motion(newpairs,newroot,second,tmp_path/"second",ledger,diagnostic_fixture=True)
    assert newpairs["pair_manifest_sha256"]!=val["pair_manifest_sha256"]
    assert result["validation_pair_set_previously_viewed"] is False
    assert result["validation_source_population_previously_viewed"] is True
    assert result["validation_previously_viewed"] is True and result["validation_used_for_new_config"] is True
    assert result["release_validated"] is False


def test_exposure_is_registered_before_damaged_image_evaluation(tmp_path,materialized_pairs):
    _,bundles=materialized_pairs
    cal,calroot,_,_=bundles["metric_calibration"]
    _,_,index,world_root=bundles["metric_validation"]
    root=tmp_path/"val-pairs"
    val=m.build_pairs(index,world_root,root,"metric_validation",1,diagnostic_fixture=True)
    receipt=m.calibrate(cal,calroot,[m.motion_config()],tmp_path/"cal")
    (root/val["pairs"][0]["images"]["current"]).write_bytes(b"damaged")
    ledger=tmp_path/"exposures.json"
    with pytest.raises(ValueError,match="hash"):
        m.validate_motion(val,root,receipt,tmp_path/"failed",ledger,diagnostic_fixture=True)
    exposure=json.loads(ledger.read_text())["exposures"]
    assert len(exposure)==1 and exposure[0]["pair_manifest_sha256"]==val["pair_manifest_sha256"]
    assert not (tmp_path/"failed"/"validation.json").exists()


def test_concurrent_validation_registration_preserves_all_entries(tmp_path,materialized_pairs):
    from concurrent.futures import ThreadPoolExecutor
    _,bundles=materialized_pairs
    val=bundles["metric_validation"][0]
    ledger=tmp_path/"ledger.json"
    def register(index):
        config=m.motion_config({**m.motion_config()["parameters"],"min_correlation":index/10})
        return m._register_validation_exposure(val,{"selected_config":config,"calibration_sha256":digest(index)},ledger)
    with ThreadPoolExecutor(max_workers=8) as workers:
        results=list(workers.map(register,range(8)))
    entries=json.loads(ledger.read_text())["exposures"]
    assert len(entries)==8 and len({entry["config_sha256"] for entry in entries})==8
    assert sum(result["previously_viewed"] is False for result in results)==1
    assert sum(result["used_for_new_config"] is True for result in results)==7


def test_release_rejects_missing_source_identity_even_when_flags_claim_success():
    """Negative structural fixture, never an actual real-validation receipt."""
    config=m.motion_config()
    worlds=[{"world_id":f"negative-control-{i}","role":"metric_validation","city":"bloomington",
             "world_pixel_sha256":digest(["pixels",i]),"world_file_sha256":digest(["file",i]),
             "source_sha256":digest(["source",i])} for i in range(12)]
    body={"schema":m.VALIDATION_SCHEMA,"status":"VALIDATED","release_validated":True,
          "real_validation_executed":True,"source_kind":"canonical_inria","config":config,
          "source_worlds":worlds,"pair_manifest_sha256":digest("negative-pairs"),
          "report":{"acceptance_passed":True,"config_sha256":config["config_sha256"],
                    "equal_world_coverage":1.,"equal_world_false_confident_rate":0.}}
    with pytest.raises(ValueError,match="source manifest"):
        m.validate_release(m._sealed(body,"validation_sha256"),config)
    body["source_manifest_sha256"]=digest("negative-source")
    body["source_worlds"][0].pop("source_sha256")
    with pytest.raises(ValueError,match="source population"):
        m.validate_release(m._sealed(body,"validation_sha256"),config)


def test_paths_do_not_reanchor_and_f3_zero_denominator(textured_world):
    config=m.motion_config();queries=[[0,0],[16,0],[32,0],[48,0]]
    images=[crop(textured_world,q,[1024,1024]) for q in queries];images[1]=None
    sequence=m.diagnose_images(images,queries,config)
    assert sequence["anchored_prefix_edges"]==0
    assert sequence["measured_edges"]==1 and sequence["normalized_endpoint_error"] is None
    assert sequence["estimated_endpoint_translation_xy"] is None
    assert m.paired_motion_ratio(sequence,sequence)["large_over_small"] is None
    same=[images[0],images[0]]
    zero=m.diagnose_images(same,[[0,0],[0,0]],config)
    assert zero["full_path_measurable"] is True and zero["estimated_full_path_length"]==0
    assert m.paired_motion_ratio(zero,zero)["large_over_small"] is None


def test_f3_ratio_on_two_measurable_scales(textured_world):
    config=m.motion_config();small=[[0,0],[16,0],[32,16]];large=[[0,0],[32,0],[64,32]]
    result=[]
    for queries in (small,large):
        result.append(m.diagnose_images([crop(textured_world,q,[1024,1024]) for q in queries],queries,config))
    ratio=m.paired_motion_ratio(*result)
    assert ratio["measurable_pairs"]==1 and ratio["large_over_small"]==2


def test_real_missing_assets_cli_writes_blocked_not_synthetic(tmp_path):
    output=tmp_path/"blocked"
    code=m.calibration_main(["build-pairs","--world-index",str(tmp_path/"missing.json"),"--world-root",str(tmp_path),"--role","metric_calibration","--out",str(output)])
    assert code==2
    record=json.loads((output/"execution-status.json").read_text())
    assert record["status"]=="BLOCKED" and record["real_calibration"]=="NOT_RUN"
    assert not (output/"pairs.json").exists()


def test_cli_scripts_are_actual_entrypoints_with_failure_receipts(tmp_path):
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,str(root/"scripts"/"validate_motion.py"),"validate","--pair-manifest",str(tmp_path/"missing.json"),"--calibration",str(tmp_path/"missing-cal.json"),"--exposure-ledger",str(tmp_path/"exposed.json"),"--out",str(tmp_path/"blocked")]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==2
    assert json.loads((tmp_path/"blocked"/"execution-status.json").read_text())["real_validation"]=="NOT_RUN"


def test_actual_suite_verified_png_motion_and_failure_slots(tmp_path,textured_world):
    """Real software execution on synthetic pixels; no real data/model claim."""
    from rswm_inria import suite
    from rswm_inria.tasks import packages
    task=next(task for task,_ in packages("motion-suite-synthetic","diagnostic_only",0) if task["family"]=="F2")
    path=tmp_path/"task.json";path.write_text(json.dumps(task))
    world_path=tmp_path/"world.png";Image.fromarray(textured_world).save(world_path)
    plan=suite.build_plan([{"task_path":str(path),"world_path":str(world_path),"city":"sfo"}],
                          [{"model_id":"oracle","adapter":"oracle"}],[0],"a"*64,"b"*40)
    suite_root=tmp_path/"suite";outcome=suite.execute_plan(plan,suite_root)[0]
    report=m.evaluate_suite(plan,suite_root,m.motion_config(),tmp_path/"motion",diagnostic=True)
    assert report["status"]=="DIAGNOSTIC_ONLY" and report["release_validated"] is False
    assert report["planned_edges"]==report["measured_edges"]==32
    assert all(row["command_error"]==0 for row in report["edges"])
    assert report["runs"][0]["submission_status"]=="VERIFIED"
    assert (tmp_path/"motion"/"motion_edges.csv").is_file()
    (Path(outcome["run_dir"])/"main"/"0001.png").write_bytes(b"tampered")
    damaged=m.evaluate_suite(plan,suite_root,m.motion_config(),tmp_path/"motion-damaged",diagnostic=True)
    assert damaged["planned_edges"]==32 and damaged["measured_edges"]==0
    assert all(row["status"]=="submission_failure" and row["command_error"] is None for row in damaged["edges"])


def test_f7_prefix_estimated_once_and_suffixes_share_continuity(tmp_path,monkeypatch):
    """Synthetic estimator stub verifies branch routing, not numeric accuracy."""
    from rswm_inria import suite
    from rswm_inria.tasks import packages
    task=next(task for task,_ in packages("motion-f7-synthetic","diagnostic_only",0) if task["family"]=="F7")
    path=tmp_path/"task.json";path.write_text(json.dumps(task))
    world=tmp_path/"world.png";Image.fromarray(np.zeros((2048,2048,3),np.uint8)).save(world)
    plan=suite.build_plan([{"task_path":str(path),"world_path":str(world),"city":"sfo"}],
                          [{"model_id":"copy","adapter":"copy"}],[0],"a"*64,"b"*40)
    suite_root=tmp_path/"suite";suite.execute_plan(plan,suite_root)
    calls=[]
    def synthetic_estimator(previous,current,config):
        calls.append((previous is not None,current is not None))
        return {"status":"measured","viewport_translation_xy":[0,0],"content_translation_xy":[0,0],"confidence":1.0,
                "configured_search_domain":m.search_domain(config),"boundary_hit":False,"config_sha256":config["config_sha256"]}
    monkeypatch.setattr(m,"estimate_pair",synthetic_estimator)
    report=m.evaluate_suite(plan,suite_root,m.motion_config(),tmp_path/"motion",diagnostic=True)
    assert len(calls)==160 and report["planned_edges"]==160
    assert len({row["pair_id"] for row in report["edges"]})==160
    assert report["runs"][0]["streams"]["prefix"]["planned_edges"]==32
    for branch in ("a","b"):
        assert report["runs"][0]["streams"][branch]["planned_edges"]==64
        assert report["runs"][0]["branch_paths"][branch]["planned_edges"]==96
    assert report["release_validated"] is False
