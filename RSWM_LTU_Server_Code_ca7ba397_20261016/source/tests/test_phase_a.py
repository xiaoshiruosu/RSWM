"""Phase A integration and adversarial accounting cases; synthetic data only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from rswm_inria import suite
from rswm_inria.aggregate import aggregate_suite, metric_cells, hierarchical, missing_bound
from rswm_inria.baselines import CopyModel, WindowMap
from rswm_inria.export_tables import export_tables
from rswm_inria.geometry import digest, file_hash
from rswm_inria.resources import ResourceCollector, process_rss_bytes, state_storage
from rswm_inria.runner import run_task
from rswm_inria.tasks import packages


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("phasea-input")
    world = np.zeros((2048,2048,3),np.uint8)
    world[:,:,0] = np.arange(2048,dtype=np.uint16)[None,:]%256
    Image.fromarray(world).save(root/"world.png")
    items = []
    for task,_ in packages("synthetic-phasea","diagnostic_only",0):
        path = root/(task["task_id"]+".json")
        path.write_text(json.dumps(task))
        items.append({"task_path":str(path),"world_path":str(root/"world.png"),"city":"sfo"})
    return root,items,world


def plan(dataset, families=None, models=None, schema=suite.SCHEMA):
    items = dataset[1]
    if families:
        items = [i for i in items if json.loads(Path(i["task_path"]).read_text())["family"] in families]
    return suite.build_plan(items,models or [{"model_id":"copy","adapter":"copy"}],
                            [0],"a"*64,"b"*40,schema=schema)


def test_expected_rejects_duplicate_and_bad_seed(tmp_path):
    r = suite.RunSpec("t","m",0)
    with pytest.raises(ValueError,match="duplicate"):
        suite.write_expected([r,r],tmp_path/"expected.jsonl")
    assert not (tmp_path/"expected.jsonl").exists()
    with pytest.raises(ValueError):
        suite.RunSpec("t","m",True)
    suite.write_expected([r],tmp_path/"expected.jsonl")
    assert suite.load_expected(tmp_path/"expected.jsonl")[0]["seed"]==0
    with pytest.raises(FileExistsError):
        suite.write_expected([r],tmp_path/"expected.jsonl")


def test_plan_identity_and_role(dataset,tmp_path):
    p = plan(dataset)
    assert len(p["expected"])==4
    assert sum(r["planned_reports"].get("planned_outputs",sum(s["planned_outputs"] for s in r["planned_reports"].values() if isinstance(s,dict) and "planned_outputs" in s)) for r in p["expected"])==352
    suite.save_plan(p,tmp_path/"plan.json")
    changed = copy.deepcopy(p);changed["expected"][0]["seed"]=12
    with pytest.raises(ValueError,match="hash"):
        suite.validate_plan(changed)
    with pytest.raises(ValueError,match="separate"):
        suite.build_plan(dataset[1],[{"model_id":"c","adapter":"copy"}],[0],"a"*64,"b"*40,study="extension")
    wrong = dict(dataset[1][0],world_pixel_sha256="0"*64)
    with pytest.raises(ValueError,match="hash"):
        suite.build_plan([wrong],[{"model_id":"c","adapter":"copy"}],[0],"a"*64,"b"*40)


def test_four_family_execution_replay_export(dataset,tmp_path):
    p = plan(dataset,models=[{"model_id":"oracle","adapter":"oracle"},{"model_id":"copy","adapter":"copy"}])
    out = tmp_path/"suite"
    results = suite.execute_plan(p,out)
    assert len(results)==8 and all(r["status"]=="COMPLETED" and r["replay_verified"] for r in results)
    report = aggregate_suite(p,out,bootstrap=20)
    oracle = [r for r in report["worlds"] if r["model_id"]=="oracle" and r["expected_pixels"]>0]
    assert oracle and all(r["mae"]==0 and r["coverage"]==1 for r in oracle)
    assert any(r["stream"]=="paired" for r in report["worlds"])
    assert {r["stream"] for r in report["worlds"] if r["family"]=="F3"}=={"small","large"}
    assert report["paired_comparisons"]
    resources = json.loads(Path(results[0]["resource_path"]).read_text())
    assert resources["model_load_seconds"]>=0 and resources["replay_seconds"]>=0
    assert resources["attempt_wall_seconds"]>=sum(resources[k] or 0 for k in ("model_load_seconds","initialize_seconds","generate_seconds","replay_seconds"))
    assert resources["gpu_time_seconds"] is None
    assert resources["process_peak_rss_bytes"]>1024
    export_tables(report,tmp_path/"tables")
    provenance = json.loads((tmp_path/"tables"/"provenance.json").read_text())
    assert provenance["files"]["worlds.csv"]==file_hash(tmp_path/"tables"/"worlds.csv")
    assert suite.execute_plan(p,out,resume=True)==results
    with pytest.raises(FileExistsError):
        suite.execute_plan(p,out)


@pytest.mark.parametrize("phase,family,status",[("reset","F2","FAILED_INITIALIZATION"),
    ("generate","F2","FAILED_GENERATE"),("snapshot","F7","FAILED_SNAPSHOT"),
    ("restore","F7","FAILED_RESTORE"),("load","F2","FAILED_LOAD")])
def test_failures_keep_all_slots(dataset,tmp_path,phase,family,status):
    p = plan(dataset,[family])
    class Broken(CopyModel):
        def reset(self,*a):
            if phase=="reset":raise RuntimeError("test")
            return super().reset(*a)
        def generate(self,q):
            if phase=="generate":raise RuntimeError("test")
            return super().generate(q)
        def snapshot(self):
            if phase=="snapshot":raise RuntimeError("test")
            return super().snapshot()
        def restore(self,s):
            if phase=="restore":raise RuntimeError("test")
            return super().restore(s)
    def factory(*args):
        if phase=="load":raise RuntimeError("test")
        return Broken()
    result = suite.execute_plan(p,tmp_path/"suite",factory=factory)[0]
    assert result["status"]==status
    folder = Path(result["run_dir"]).parent
    slots = json.loads((folder/"output-slots.json").read_text())
    assert len(slots)==(160 if family=="F7" else 32)
    report = aggregate_suite(p,tmp_path/"suite",bootstrap=5)
    assert report["cells"] and sum(r["expected_pixels"] for r in report["cells"])>0
    assert any(r["coverage"]==0 and r["missing_bound"]==[0.,1.] for r in report["cells"])


def test_crash_is_new_attempt_never_png_resume(dataset,tmp_path):
    p = plan(dataset,["F2"],schema=suite.LEGACY_SCHEMA)
    class Crash(CopyModel):
        def generate(self,q):
            raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        suite.execute_plan(p,tmp_path/"suite",factory=lambda *a:Crash())
    result = suite.execute_plan(p,tmp_path/"suite",resume=True)[0]
    assert result["status"]=="CRASHED" and result["attempt"]==0
    result = suite.execute_plan(p,tmp_path/"suite",resume=True,retry_failed=True)[0]
    assert result["status"]=="COMPLETED" and result["attempt"]==1
    old = Path(result["run_dir"]).parents[1]/"attempt-000"/"outcome.json"
    assert json.loads(old.read_text())["status"]=="CRASHED"
    assert suite.execute_plan(p,tmp_path/"suite",resume=True,retry_failed=True)[0]["attempt"]==1
    report=aggregate_suite(p,tmp_path/"suite",bootstrap=5)
    assert len(report["resource_attempts"])==2
    assert report["resource_totals"][0]["attempts"]==2
    assert report["resource_totals"][0]["metrics"]["model_load_seconds"]["measured_attempts"]==2


def test_latest_failed_attempt_not_best_score(dataset,tmp_path):
    p = plan(dataset,["F2"],schema=suite.LEGACY_SCHEMA)
    out = tmp_path/"suite"
    suite.execute_plan(p,out)
    row = p["expected"][0]
    last = out/"runs"/row["run_id"]/"attempt-001"
    last.mkdir()
    # An interrupted later attempt is retained even when an older one succeeded.
    report = aggregate_suite(p,out,bootstrap=5)
    assert all(r["valid_pixels"]==0 for r in report["cells"])
    assert report["replay_status"][0]["status"]=="CRASHED"


def test_replay_tamper_fails_closed_without_losing_plan(dataset,tmp_path):
    p = plan(dataset,["F2"])
    result = suite.execute_plan(p,tmp_path/"suite")[0]
    image = Path(result["run_dir"])/"main"/"0001.png"
    image.write_bytes(b"bad")
    report = aggregate_suite(p,tmp_path/"suite",bootstrap=5)
    assert report["replay_status"][0]["status"]=="FAILED_REPLAY"
    assert all(r["valid_pixels"]==0 for r in report["cells"])
    assert sum(r["expected_pixels"] for r in report["cells"])>0


def test_input_hash_failure_keeps_plan(dataset,tmp_path):
    root,items,world = dataset
    local = tmp_path/"task.json"
    local.write_text(Path(items[0]["task_path"]).read_text())
    p = suite.build_plan([dict(items[0],task_path=str(local))],[{"model_id":"c","adapter":"copy"}],[0],"a"*64,"b"*40)
    t = json.loads(local.read_text());t["role"]="changed";local.write_text(json.dumps(t))
    result = suite.execute_plan(p,tmp_path/"suite")[0]
    assert result["status"]=="FAILED_INPUT"
    assert len(json.loads((Path(result["run_dir"]).parent/"output-slots.json").read_text()))==32


def test_rss_units_and_alias_storage(monkeypatch):
    import resource
    monkeypatch.setattr(resource,"getrusage",lambda *_:SimpleNamespace(ru_maxrss=42))
    monkeypatch.setattr("rswm_inria.resources.sys.platform","linux")
    assert process_rss_bytes()==42*1024
    monkeypatch.setattr("rswm_inria.resources.sys.platform","darwin")
    assert process_rss_bytes()==42
    array = np.zeros((20,10),np.uint8)
    state = state_storage({"own":array,"alias":array[:10]})
    assert state["unique_numpy_buffer_bytes"]==200 and state["logical_numpy_bytes"]==300


def test_gpu_sync_boundaries_mock_only():
    class CUDA:
        def __init__(self):self.sync=0
        def is_available(self):return True
        def synchronize(self,d):self.sync+=1
        def reset_peak_memory_stats(self,d):pass
        def max_memory_allocated(self,d):return 123
        def max_memory_reserved(self,d):return 456
    cuda = CUDA();collector = ResourceCollector("cuda:0",SimpleNamespace(cuda=cuda))
    with collector.phase("generate",count=16):pass
    result = collector.finish()
    assert cuda.sync==3 and result["gpu_time_seconds"]>=0
    assert result["gpu_peak_allocated_bytes"]==123


def test_f7_cumulative_map_costs_survive_restore(dataset):
    task = [t for t,_ in packages("cost-test","diagnostic_only",0) if t["family"]=="F7"][0]
    model = WindowMap();resources = ResourceCollector()
    run_task(task,dataset[2],model,resources=resources)
    data = resources.finish()
    assert data["adapter_counters"]["completion_calls"]==model.resource_calls
    assert model.resource_calls>model.calls
    assert data["dynamic_state_bytes"]>=2048*2048*4


def _cell(world,city,error,coverage,model="a"):
    expected=100;valid=int(expected*coverage);absolute=int(error*255*3*valid)
    return {"study":"core","model_id":model,"family":"F2","protocol":"offline","variant":"default",
            "stream":"main","metric":"known","world_id":world,"city":city,"mae":error if valid else None,
            "coverage":coverage,"missing_bound":missing_bound(error if valid else None,coverage),
            "expected_pixels":expected,"valid_pixels":valid,"absolute_error_rgb":absolute,
            "planned_outputs":32,"valid_outputs":int(32*coverage)}


def test_hierarchy_equal_city_and_all_missing():
    rows = [_cell("w1","sfo",0,1),_cell("w2","sfo",0,1),_cell("w3","innsbruck",1,1)]
    result = hierarchical(rows)
    assert result["equal_city"][0]["mae"]==.5
    rows[2]=_cell("w3","innsbruck",0,0)
    result=hierarchical(rows)
    assert len(result["worlds"])==3 and result["equal_city"][0]["mae"] is None
    assert result["equal_city"][0]["missing_bound"]==[0.,.5]


def test_aggregate_rejects_unknown_duplicate_nan(dataset):
    from rswm_inria.aggregate import aggregate_world
    p=plan(dataset,["F2"])
    with pytest.raises(ValueError):metric_cells(p,[{"run_id":"unknown"}])
    r={"run_id":p["expected"][0]["run_id"]}
    with pytest.raises(ValueError):metric_cells(p,[r,r])
    with pytest.raises(ValueError):aggregate_world([{"world_id":"w","value":float("nan")}])
    assert missing_bound(None,None) is None
    assert missing_bound(None,0)==[0.,1.]
    with pytest.raises(ValueError):missing_bound(float("nan"),.5)


def test_hard_process_exit_releases_lock_and_marks_crash(dataset,tmp_path):
    import os
    import subprocess
    import sys
    p=plan(dataset,["F2"],schema=suite.LEGACY_SCHEMA)
    suite.save_plan(p,tmp_path/"plan.json")
    code = '''import os,sys
from rswm_inria.suite import execute_plan,_json
from rswm_inria.baselines import CopyModel
class HardExit(CopyModel):
    def generate(self,q):os._exit(9)
execute_plan(_json(sys.argv[1]),sys.argv[2],factory=lambda *a:HardExit())
'''
    child = subprocess.run([sys.executable,"-c",code,str(tmp_path/"plan.json"),str(tmp_path/"suite")],
                           env=os.environ.copy(),timeout=60)
    assert child.returncode==9
    result = suite.execute_plan(p,tmp_path/"suite",resume=True)[0]
    assert result["status"]=="CRASHED" and result["attempt"]==0
    assert suite.execute_plan(p,tmp_path/"suite",resume=True,retry_failed=True)[0]["attempt"]==1


def test_index_cli_contract(dataset,tmp_path):
    root,items,_=dataset
    tasks=[]
    for item in items[:1]:
        task=json.loads(Path(item["task_path"]).read_text())
        tasks.append({**{k:task[k] for k in ("task_id","world_id","family","role")},
                      "city":"sfo","task_path":item["task_path"]})
    world=suite._world(root/"world.png")
    import hashlib
    index={"source_manifest_sha256":"a"*64,"tasks":tasks,"worlds":[{
        "world_id":"synthetic-phasea","city":"sfo","path":"world.png",
        "world_file_sha256":file_hash(root/"world.png"),
        "world_pixel_sha256":hashlib.sha256(world.tobytes()).hexdigest()}]}
    (tmp_path/"index.json").write_text(json.dumps(index))
    (tmp_path/"models.json").write_text(json.dumps([{"model_id":"copy","adapter":"copy"}]))
    assert suite.main(["plan","--task-index",str(tmp_path/"index.json"),"--world-dir",str(root),
                       "--models",str(tmp_path/"models.json"),"--seeds","0","--code-commit","b"*40,
                       "--out",str(tmp_path/"plan.json")])==0
    assert suite.main(["run","--plan",str(tmp_path/"plan.json"),"--out",str(tmp_path/"suite")])==0
    from rswm_inria.export_tables import main
    assert main(["--plan",str(tmp_path/"plan.json"),"--suite",str(tmp_path/"suite"),
                 "--out",str(tmp_path/"tables"),"--bootstrap","5"])==0


def test_blanket_missing_world_retained_in_plan(dataset,tmp_path):
    root,items,_=dataset
    task=json.loads(Path(items[0]["task_path"]).read_text())
    task["world_id"]="missing-world";task["task_id"]="missing-world-o0-F2"
    tp=tmp_path/"missing.json";tp.write_text(json.dumps(task))
    source=items[:1]+[{"task_path":str(tp),"world_path":str(root/"world.png"),"city":"innsbruck"}]
    p=suite.build_plan(source,[{"model_id":"c","adapter":"copy"}],[0],"a"*64,"b"*40)
    rows=metric_cells(p,[])
    report=hierarchical(rows)
    assert len(report["worlds"])==8
    assert {r["city"] for r in report["cities"]}=={"sfo","innsbruck"}
    assert all(r["mae"] is None and r["coverage"]==0 for r in report["worlds"] if r["expected_pixels"]>0)


def test_resource_failed_completer_call_counted():
    def fail(*args):raise RuntimeError("completer failure")
    model=WindowMap(completer=fail,eager=True)
    collector=ResourceCollector()
    with pytest.raises(RuntimeError):
        with collector.phase("initialize",model):
            model.reset(np.zeros((240,416,3),np.uint8),{},0)
    assert collector.finish()["adapter_counters"]["completion_calls"]==1


def test_pairing_uses_same_cells_not_different_measurable_subsets():
    from rswm_inria.aggregate import comparisons
    rows=[]
    for model,task,error,coverage in [("a","t1",0,1),("a","t2",0,0),("b","t1",0,0),("b","t2",1,1)]:
        row=_cell("w","sfo",error,coverage,model)
        row.update(task_id=task,origin_index=0,seed=0)
        rows.append(row)
    result=comparisons(rows,bootstrap=5)[0]
    assert result["cities"]["sfo"]["paired_worlds"]==0
    assert result["common_world_cells"][0]["planned_applicable_cells"]==2
    assert result["common_world_cells"][0]["common_measured_cells"]==0
    assert result["equal_city_mean_a_minus_b"] is None
