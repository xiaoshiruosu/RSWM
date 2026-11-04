"""Frozen expected plans, complete attempts and independent batch replay.

Cooperative in-process execution only. Suite receipts are outside each run
folder, but must be copied to independent storage for a trusted external anchor.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import uuid
import math
import signal
import threading
import numpy as np
from PIL import Image
from .geometry import canonical, digest, file_hash, rgb
from .tasks import validate_task
from .runner import run_task
from .replay import replay_run, schedule, load_json
from .resources import ResourceCollector, ResourceRecord

SCHEMA = "rswm-suite-v3"
LEGACY_SCHEMA = "rswm-suite-v2"
DEFAULT_RETRY_POLICY = {"primary_selection":"first_attempt_only", "max_attempts_per_run":1,
                        "automatic_retry":False,"allowed_retry_statuses":["CRASHED","INTERRUPTED","FAILED_LOAD"],
                        "secondary_selection":"latest_attempt_within_budget","cost_scope":"all_started_attempts"}
METRIC_CONFIG = {"schema":"rswm-rgb-ledger-v1","range":[0,1],
                 "first_reference":"first_planned_exposure","missing":"never_repair"}
STATES = {"PLANNED","RUNNING","COMPLETED","INCOMPLETE","FAILED_LOAD",
          "FAILED_INITIALIZATION","FAILED_GENERATE","FAILED_SNAPSHOT",
          "FAILED_RESTORE","FAILED_REPLAY","FAILED_INPUT","CRASHED","INTERRUPTED","NOT_RUN","METADATA_CORRUPT"}
from .registry import ADAPTERS


@dataclass(frozen=True)
class RunSpec:
    task_id: str
    model_id: str
    seed: int
    attempt: int = 0
    protocol: str = "offline"
    family: str = "unknown"
    status: str = "PLANNED"

    def __post_init__(self):
        if self.status not in STATES or self.protocol not in {"offline","online16"}:
            raise ValueError("Invalid suite status/protocol")
        if any(type(x) is not int or x < 0 for x in (self.seed,self.attempt)):
            raise ValueError("seed/attempt must be nonnegative integers")
        if not self.task_id or not self.model_id:
            raise ValueError("Empty run identity")


def _json(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("SYMLINK_NOT_ALLOWED")
    value=load_json(path.read_text(encoding="utf-8"))
    def finite(item):
        if isinstance(item,float) and not math.isfinite(item):
            raise ValueError("JSON_NUMBER_NONFINITE")
        if isinstance(item,dict):
            for member in item.values():finite(member)
        elif isinstance(item,list):
            for member in item:finite(member)
    finite(value)
    return value


def _save(path, data, exclusive=False):
    """Atomic complete JSON; never expose a partially written outcome."""
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("SYMLINK_NOT_ALLOWED")
    path.parent.mkdir(parents=True,exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    temp = path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    with temp.open("x",encoding="utf-8") as f:
        f.write(canonical(data).decode()+"\n")
        f.flush();os.fsync(f.fileno())
    if exclusive and path.exists():
        temp.unlink();raise FileExistsError(path)
    if exclusive:
        try:
            os.link(temp,path)
        finally:
            temp.unlink()
    else:
        temp.replace(path)
    fd = os.open(path.parent,os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_expected(records, path):
    records = [asdict(r) if isinstance(r,RunSpec) else r for r in records]
    keys = [(r.get("study","core"),r["task_id"],r["model_id"],r["seed"],r["protocol"],r.get("attempt",0)) for r in records]
    if not records or len(keys)!=len(set(keys)):
        raise ValueError("Empty or duplicate expected plan")
    for r in records:
        RunSpec(**{k:r[k] for k in RunSpec.__dataclass_fields__ if k in r})
    text = "".join(canonical(r).decode()+"\n" for r in records)
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:
        f.write(text);f.flush();os.fsync(f.fileno())


def load_expected(path):
    path=Path(path)
    if path.is_symlink():
        raise ValueError("EXPECTED_PATH_ESCAPE")
    lines=path.read_text(encoding="utf-8").splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("Blank expected JSONL row")
    rows = [load_json(line) for line in lines]
    if not all(isinstance(row,dict) and all(k in row for k in ("task_id","model_id","seed","protocol")) for row in rows):
        raise ValueError("Expected rows must be identity objects")
    keys = [(r.get("study","core"),r["task_id"],r["model_id"],r["seed"],r["protocol"],r.get("attempt",0)) for r in rows]
    if not rows or len(keys)!=len(set(keys)):
        raise ValueError("Empty or duplicate expected plan")
    for r in rows:
        RunSpec(**{k:r[k] for k in RunSpec.__dataclass_fields__ if k in r})
    return rows


def code_fingerprint():
    root = Path(__file__).parent
    return digest({p.relative_to(root).as_posix():file_hash(p) for p in sorted(root.rglob("*.py"))})


def _policy(value):
    if not isinstance(value,dict) or set(value)!=set(DEFAULT_RETRY_POLICY):
        raise ValueError("Invalid retry policy schema")
    if value["primary_selection"]!="first_attempt_only" or value["secondary_selection"]!="latest_attempt_within_budget":
        raise ValueError("Unsupported primary/secondary retry selection")
    if type(value["max_attempts_per_run"]) is not int or not 1<=value["max_attempts_per_run"]<=10000:
        raise ValueError("Finite positive retry budget required")
    if value["automatic_retry"] is not False or value["cost_scope"]!="all_started_attempts":
        raise ValueError("Only explicit retries with all-attempt cost accounting are supported")
    allowed=value["allowed_retry_statuses"]
    if not isinstance(allowed,list) or len(allowed)!=len(set(allowed)) or any(s not in STATES-{"COMPLETED","PLANNED","RUNNING","NOT_RUN"} for s in allowed):
        raise ValueError("Invalid allowed retry statuses")
    return dict(value,allowed_retry_statuses=list(allowed))


def _sample_id(row):
    return digest({k:row[k] for k in ("study","task_id","model_manifest_sha256","seed","protocol","metric_config_sha256")})


def model_manifest(model):
    result = {"config":model,"weights":{}}
    if model.get("adapter") == "numeric-camera":
        from .registry import numeric_contract
        result.update(numeric_contract(model))
    if model.get("model_dir"):
        root = Path(model["model_dir"])
        if not root.is_dir():
            raise ValueError("Local model directory required")
        files = sorted(root.rglob("*"))
        if any(p.is_symlink() for p in files):
            raise ValueError("Model symlinks are not supported in frozen manifests")
        result["weights"] = {str(p.relative_to(root)):file_hash(p) for p in files if p.is_file()}
    return result


def _load_task(path):
    task = _json(path);labels = {}
    for name, info in task.get("private_label_files",{}).items():
        relative=Path(info["path"])
        root=Path(path).parent.resolve()
        p=root/relative
        private_root = root.parent if root.name == "tasks" else root
        if relative.is_absolute() or p.is_symlink() or any(parent.is_symlink() for parent in p.parents) or not p.resolve().is_relative_to(private_root):
            raise ValueError("PRIVATE_LABEL_PATH_ESCAPE")
        if file_hash(p)!=info["sha256"]:
            raise ValueError("Private label hash mismatch")
        with np.load(p,allow_pickle=False) as z:
            labels[name] = {k:z[k] for k in z.files}
    validate_task(task,labels)
    return task, labels


def _world(path):
    with Image.open(path) as im:
        if im.format!="PNG" or im.mode!="RGB":
            raise ValueError("Lossless RGB world PNG required")
        return rgb(np.asarray(im).copy(),(2048,2048,3))


class _Missing:
    def reset(self,*args):
        pass
    def generate(self,q):
        return [None]*len(q)
    def snapshot(self):
        return {}
    def restore(self,state):
        pass


def planned_reports(task):
    """Use the unchanged ledger on geometry only; no model/real pixels consulted."""
    result,_ = run_task(task,np.zeros((2048,2048,3),np.uint8),_Missing())
    return result["reports"]


def build_plan(tasks, models, seeds, source_manifest_sha256, code_commit, study="core", retry_policy=None, schema=SCHEMA):
    """tasks contain task_path, world_path and city, with optional locked hashes."""
    if not re.fullmatch(r"[0-9a-f]{64}",source_manifest_sha256):
        raise ValueError("Source manifest SHA256 required")
    if not re.fullmatch(r"[0-9a-f]{40}",code_commit):
        raise ValueError("Full source commit SHA required; working source hash is also recorded")
    if not tasks or not models or not seeds or len(seeds)!=len(set(seeds)):
        raise ValueError("Empty plan or duplicate seeds")
    if any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Inference seeds must be nonnegative integers")
    if schema not in {SCHEMA,LEGACY_SCHEMA}:
        raise ValueError("Unknown suite schema")
    if schema==LEGACY_SCHEMA and retry_policy is not None:
        raise ValueError("v2 legacy policy is immutable; create a separate v3 plan")
    policy = _policy(retry_policy or DEFAULT_RETRY_POLICY) if schema==SCHEMA else "latest attempted result; retry only non-COMPLETED runs; never select by score"
    models = [dict(m,device=m.get("device","cuda" if m.get("model_dir") or m.get("adapter")=="numeric-camera" else "cpu")) for m in models]
    ids = [m["model_id"] for m in models]
    if len(ids)!=len(set(ids)) or any(not isinstance(i,str) or not i for i in ids):
        raise ValueError("Duplicate or empty model IDs")
    model_locks = {}
    for model in models:
        from .registry import validate_config
        validate_config(model)
        if model.get("adapter") not in ADAPTERS:
            raise ValueError("Unknown adapter")
        if model["adapter"] in {"wan-i2v","eager-map","lazy-map"} and not model.get("model_dir") and not model.get("diagnostic_noise"):
            raise ValueError("Generative adapters need local weights or explicit diagnostic_noise")
        if model["adapter"]=="wan-i2v" and not model.get("model_dir"):
            raise ValueError("Wan requires local weights")
        if "training_seed" in model and model["training_seed"] is not None and (type(model["training_seed"]) is not int or model["training_seed"]<0):
            raise ValueError("Invalid training seed")
        model_locks[model["model_id"]] = model_manifest(model)
    rows, seen, worlds, frozen_tasks, world_bindings = [], set(), {}, {}, {}
    for item in sorted(tasks,key=lambda r:str(r["task_path"])):
        for raw in (item["task_path"],item["world_path"]):
            path=Path(raw)
            if path.is_symlink() or any(p.is_symlink() for p in path.parents):
                raise ValueError("Input symlinks are not supported")
        task_path,world_path = Path(item["task_path"]).resolve(),Path(item["world_path"]).resolve()
        task,_ = _load_task(task_path)
        task_study = task.get("study","core")
        if task_study!=study:
            raise ValueError("Core and extensions require separate plans")
        if task["task_id"] in seen or not item.get("city"):
            raise ValueError("Duplicate task or missing city")
        seen.add(task["task_id"])
        for key in ("task_id","world_id","family","role"):
            if key in item and item[key]!=task[key]:
                raise ValueError("Task index metadata mismatch")
        if world_path not in worlds:
            world = _world(world_path)
            worlds[world_path] = (file_hash(world_path),hashlib.sha256(world.tobytes()).hexdigest())
        wf,wp = worlds[world_path]
        binding = (wf,wp,item["city"],task["role"])
        if task["world_id"] in world_bindings and world_bindings[task["world_id"]]!=binding:
            raise ValueError("Inconsistent world identity/city/role")
        world_bindings[task["world_id"]] = binding
        for key,value in (("world_file_sha256",wf),("world_pixel_sha256",wp)):
            if key in item and item[key]!=value:
                raise ValueError("World index hash mismatch")
        frozen_tasks[task["task_id"]] = task
        world = _world(world_path)
        missing = planned_reports(task)
        for model in sorted(models,key=lambda m:m["model_id"]):
            for seed in sorted(seeds):
                row = {**asdict(RunSpec(task["task_id"],model["model_id"],seed,protocol=task["protocol"],family=task["family"])),
                       "study":study,"world_id":task["world_id"],"city":item["city"],"role":task["role"],
                       "origin_index":task["origin_index"],"variant":task["metadata"].get("variant","default"),
                       "training_seed":model.get("training_seed"),"task_path":str(task_path),"world_path":str(world_path),
                       "task_sha256":digest(task),"world_file_sha256":wf,"world_pixel_sha256":wp,
                       "initial_sha256":hashlib.sha256(world[task["origin"][1]-120:task["origin"][1]+120,task["origin"][0]-208:task["origin"][0]+208].tobytes()).hexdigest(),
                       "source_manifest_sha256":source_manifest_sha256,
                       "model_manifest_sha256":digest(model_locks[model["model_id"]]),
                       "metric_config_sha256":digest(METRIC_CONFIG),"code_commit":code_commit,
                       "planned_reports":missing}
                row["run_id"] = (_sample_id(row) if schema==SCHEMA else digest({k:row[k] for k in ("study","task_id","model_id","seed","protocol")}))
                if schema==SCHEMA:
                    row["sample_id"] = row["run_id"]
                rows.append(row)
    plan = {"schema":schema,"study":study,"expected":rows,"models":model_locks,"tasks":frozen_tasks,
            "source_code_sha256":code_fingerprint(),"metric_config":METRIC_CONFIG,
            "retry_policy":policy,
            "environment":{"python":sys.version,"numpy":np.__version__,"Pillow":importlib.metadata.version("Pillow")}}
    plan["plan_sha256"] = digest(plan)
    return plan


def validate_plan(plan):
    if not isinstance(plan,dict):
        raise ValueError("Plan must be an object")
    body = {k:v for k,v in plan.items() if k!="plan_sha256"}
    if plan.get("schema") not in {SCHEMA,LEGACY_SCHEMA} or digest(body)!=plan.get("plan_sha256"):
        raise ValueError("Plan hash/schema mismatch")
    if plan["source_code_sha256"]!=code_fingerprint() or plan["metric_config"]!=METRIC_CONFIG:
        raise ValueError("Executing code/metric config differs from the frozen plan")
    if plan["schema"]==SCHEMA:
        _policy(plan.get("retry_policy"))
    elif plan.get("retry_policy")!="latest attempted result; retry only non-COMPLETED runs; never select by score":
        raise ValueError("Legacy retry policy differs")
    expected=plan.get("expected")
    if not isinstance(expected,list) or not expected:
        raise ValueError("Expected plan is missing")
    keys = [r["run_id"] for r in expected]
    if not keys or len(keys)!=len(set(keys)):
        raise ValueError("Empty or duplicate run IDs")
    from .data import ROLES
    from .registry import validate_config
    tasks=plan["tasks"]
    models=plan["models"]
    task_identity={}
    for task_id,task in tasks.items():
        validate_task(task)
        if task.get("task_id")!=task_id or task.get("study","core")!=plan["study"]:
            raise ValueError("Frozen task identity/study mismatch")
        task_identity[task_id]=(digest(task),digest(planned_reports(task)))
    for model_id,model in models.items():
        validate_config(model["config"], check_local_assets=False)
        if model["config"]["model_id"]!=model_id:
            raise ValueError("Frozen model identity mismatch")
        if model["config"]["adapter"]=="numeric-camera":
            from .registry import numeric_contract
            contract=numeric_contract(model["config"],check_local_assets=False)
            if set(model)!={"config","source","weights",*contract} or any(model[k]!=v for k,v in contract.items()):
                raise ValueError("Frozen numeric backend/camera identity mismatch")
            for field,sha_field in (("source","source_tree_sha256"),("weights","weights_tree_sha256")):
                table=model[field]
                if not isinstance(table,dict) or not table or any(not isinstance(k,str) or not isinstance(v,str) or not re.fullmatch(r"[0-9a-f]{64}",v) for k,v in table.items()) or digest(table)!=contract["backend_manifest"][sha_field]:
                    raise ValueError("Frozen numeric backend tree mismatch")
    model_identity={model_id:digest(model) for model_id,model in models.items()}
    metric_identity=digest(plan["metric_config"])
    seen=set();world_bindings={};source_bindings=set()
    for row in expected:
        RunSpec(**{k:row[k] for k in RunSpec.__dataclass_fields__})
        task=tasks[row["task_id"]]
        model=models[row["model_id"]]
        if row["attempt"]!=0 or row["status"]!="PLANNED":
            raise ValueError("Expected rows must contain attempt-zero planned samples")
        for name in ("world_id","family","protocol","role","origin_index"):
            if row[name]!=task[name]:
                raise ValueError("Expected/task metadata mismatch")
        if row["study"]!=plan["study"] or row["city"] not in ROLES:
            raise ValueError("Unknown city or study")
        if row["role"]!="diagnostic_only" and row["role"]!=ROLES[row["city"]][0]:
            raise ValueError("Frozen city/role mismatch")
        if model["config"]["model_id"]!=row["model_id"] or model_identity[row["model_id"]]!=row["model_manifest_sha256"] or task_identity[row["task_id"]][0]!=row["task_sha256"] or metric_identity!=row["metric_config_sha256"]:
            raise ValueError("Expected/model/task/metric identity mismatch")
        if row["variant"]!=task["metadata"].get("variant","default") or row["training_seed"]!=model["config"].get("training_seed"):
            raise ValueError("Expected variant/training seed mismatch")
        for name in ("source_manifest_sha256","world_file_sha256","world_pixel_sha256","initial_sha256"):
            if not isinstance(row[name],str) or not re.fullmatch(r"[0-9a-f]{64}",row[name]):
                raise ValueError("Invalid frozen input hash")
        if not isinstance(row["code_commit"],str) or not re.fullmatch(r"[0-9a-f]{40}",row["code_commit"]):
            raise ValueError("Invalid frozen source commit")
        source_bindings.add((row["source_manifest_sha256"],row["code_commit"]))
        binding=(row["world_file_sha256"],row["world_pixel_sha256"],row["city"],row["role"])
        if row["world_id"] in world_bindings and world_bindings[row["world_id"]]!=binding:
            raise ValueError("Inconsistent frozen world identity/city/role")
        world_bindings[row["world_id"]]=binding
        run_id=(_sample_id(row) if plan["schema"]==SCHEMA else digest({k:row[k] for k in ("study","task_id","model_id","seed","protocol")}))
        if row["run_id"]!=run_id or (plan["schema"]==SCHEMA and row.get("sample_id")!=run_id):
            raise ValueError("Sample identity mismatch")
        reports=row["planned_reports"]
        streams=[reports] if "metrics" in reports else list(reports.values())
        if sum(s["planned_outputs"] for s in streams)!={"F2":32,"F3":64,"F5":96,"F7":160}[task["family"]] or any(s["valid_outputs"]!=0 for s in streams):
            raise ValueError("Frozen planned output denominator mismatch")
        if digest(reports)!=task_identity[row["task_id"]][1]:
            raise ValueError("Frozen geometric score denominator mismatch")
        key=(row["task_id"],row["model_id"],row["seed"])
        if key in seen:
            raise ValueError("Duplicate expected sample")
        seen.add(key)
    if {r["task_id"] for r in expected}!=set(plan["tasks"]) or {r["model_id"] for r in expected}!=set(plan["models"]):
        raise ValueError("Expected task/model population differs")
    if len(source_bindings)!=1:
        raise ValueError("Inconsistent frozen source manifest/commit")
    seeds={r["seed"] for r in expected}
    if seen!={(t,m,s) for t in plan["tasks"] for m in plan["models"] for s in seeds}:
        raise ValueError("Incomplete expected task/model/seed product")


def save_plan(plan,path):
    validate_plan(plan);_save(path,plan,exclusive=True)


def model_factory(config,world,task):
    from .baselines import OracleMap
    from .registry import create_adapter
    if config["adapter"]=="oracle":
        return OracleMap(world,task["origin"])
    return create_adapter(config)


def _append(path,row):
    path=Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("SYMLINK_NOT_ALLOWED")
    with Path(path).open("a",encoding="utf-8") as f:
        f.write(canonical(row).decode()+"\n");f.flush();os.fsync(f.fileno())


def _slots(task, root, status, journal=None):
    if journal is not None:
        rows = [{k:r[k] for k in ("stream","event","q","status")} for r in journal.rows if r["kind"]=="output"]
    else:
        rows = [{"stream":r["stream"],"event":r["event"],"q":r["q"],"status":"missing","failure":status}
                for r in schedule(task) if r["kind"]=="output"]
    _save(root/"output-slots.json",rows)


def _failure_from_journal(journal, reports):
    failures = [r.get("phase") for r in journal.rows if r["kind"]=="adapter_failure"]
    for phase,status in (("reset","FAILED_INITIALIZATION"),("snapshot","FAILED_SNAPSHOT"),("restore","FAILED_RESTORE")):
        if phase in failures:
            return status
    if any(r["kind"]=="output" and r["status"] in {"model_exception","invalid_batch","invalid_image"} for r in journal.rows):
        return "FAILED_GENERATE"
    streams = [reports] if "metrics" in reports else list(reports.values())
    return "COMPLETED" if all(r["output_completeness"]==1 for r in streams) else "INCOMPLETE"


@contextmanager
def suite_read_lock(root):
    """Shared nonblocking lock: an active coordinator cannot be read as final."""
    import fcntl
    root=Path(root)
    if not root.is_dir() or root.is_symlink() or any(p.is_symlink() for p in root.parents) or (root/"coordinator.lock").is_symlink():
        raise ValueError("Invalid suite root/lock path")
    lock=(root/"coordinator.lock").open("a+")
    try:
        try:
            fcntl.flock(lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another suite coordinator is active") from exc
        yield
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


def _root_identity(plan,root):
    saved=_json(root/"plan.json")
    validate_plan(saved)
    if saved["plan_sha256"]!=plan["plan_sha256"]:
        raise ValueError("Suite plan mismatch")
    if canonical(load_expected(root/"expected.jsonl"))!=canonical(plan["expected"]):
        raise ValueError("Expected index differs from frozen plan")


def _attempt_folders(parent):
    if parent.is_symlink() or any(p.is_symlink() for p in parent.parents):
        raise ValueError("RUN_PATH_ESCAPE")
    if not parent.exists():
        return []
    folders=[]
    for folder in parent.iterdir():
        match=re.fullmatch(r"attempt-(\d{3,})",folder.name)
        if not match or not folder.is_dir() or folder.is_symlink():
            raise ValueError("UNEXPECTED_ATTEMPT_PATH")
        number=int(match[1])
        if folder.name!=f"attempt-{number:03d}":
            raise ValueError("NONCANONICAL_ATTEMPT_PATH")
        folders.append((number,folder))
    return [p for _,p in sorted(folders)]


def _metadata(path,row,plan,attempt,kind,folder=None):
    value=_json(path)
    if not isinstance(value,dict):
        raise ValueError("METADATA_NOT_OBJECT")
    required={"run_id":row["run_id"],"attempt":attempt,"plan_sha256":plan["plan_sha256"]}
    if kind=="head" and plan["schema"]==LEGACY_SCHEMA and "plan_sha256" not in value:
        required.pop("plan_sha256")
    if kind=="resource" and plan["schema"]==LEGACY_SCHEMA and value.get("schema")!="rswm-resource-v3":
        # Old resource records have no identity contract. Any identities they
        # explicitly carry must still agree; their measurement scope is unknown.
        required={k:v for k,v in required.items() if k in value}
    if any(value.get(k)!=v or (k=="attempt" and type(value.get(k)) is not int) for k,v in required.items()):
        raise ValueError("METADATA_IDENTITY_MISMATCH")
    if kind in {"started","outcome","receipt"}:
        if value.get("status") not in STATES:
            raise ValueError("METADATA_STATUS_INVALID")
    if kind=="started" and value["status"]!="RUNNING":
        raise ValueError("STARTED_STATUS_INVALID")
    if kind=="outcome":
        if not isinstance(value.get("replay_verified"),bool) or not isinstance(value.get("reports"),dict):
            raise ValueError("OUTCOME_FIELDS_MISSING")
        if value.get("run_dir")!=str(folder/"submission"):
            raise ValueError("OUTCOME_PATH_ESCAPE")
        if value.get("resource_path") not in {None,str(folder/"resources.json")}:
            raise ValueError("RESOURCE_PATH_ESCAPE")
    if kind=="resource":
        for name,item in value.items():
            if name.endswith(("_seconds","_bytes")) and item is not None and (type(item) not in {int,float} or not math.isfinite(item) or item<0):
                raise ValueError("RESOURCE_NUMBER_INVALID")
        if not isinstance(value.get("measurement_scope"),dict):
            raise ValueError("RESOURCE_FIELDS_MISSING")
        if not isinstance(value.get("adapter_counters"),dict):
            raise ValueError("RESOURCE_COUNTER_SCHEMA_INVALID")
        for name,item in value.get("adapter_counters",{}).items():
            if item is not None and (type(item) not in {int,float} or not math.isfinite(item) or item<0):
                raise ValueError("RESOURCE_COUNTER_INVALID")
        if plan["schema"]==SCHEMA or value.get("schema")=="rswm-resource-v3":
            expected_fields=set(ResourceRecord().to_dict())-{"schema"}
            if value.get("schema")!="rswm-resource-v3":
                raise ValueError("RESOURCE_SCHEMA_DOWNGRADED")
            if not expected_fields.issubset(value):
                raise ValueError("RESOURCE_REQUIRED_FIELDS_MISSING")
            if type(value.get("measurement_complete")) is not bool:
                raise ValueError("RESOURCE_COMPLETENESS_MISSING")
            for collection in ("measurements","counter_measurements"):
                fields=value.get(collection)
                if not isinstance(fields,dict) or not fields:
                    raise ValueError("RESOURCE_MEASUREMENT_SCHEMA_INVALID")
                if collection=="measurements" and not expected_fields.issubset(fields):
                    raise ValueError("RESOURCE_REQUIRED_MEASUREMENTS_MISSING")
                for name,measurement in fields.items():
                    if not isinstance(measurement,dict):
                        raise ValueError("RESOURCE_MEASUREMENT_NOT_OBJECT")
                    if measurement.get("applicability") not in {"applicable","not_applicable","unknown"} or measurement.get("aggregation") not in {"sum","max"} or type(measurement.get("complete")) is not bool:
                        raise ValueError("RESOURCE_MEASUREMENT_SCHEMA_INVALID")
                    eligible,measured=measurement.get("eligible_units"),measurement.get("measured_units")
                    if type(eligible) is not int or type(measured) is not int or not 0<=measured<=eligible:
                        raise ValueError("RESOURCE_MEASUREMENT_RANGE_INVALID")
                    if "value" not in measurement or ("observed_peak" if measurement["aggregation"]=="max" else "measured_sum") not in measurement:
                        raise ValueError("RESOURCE_MEASUREMENT_VALUES_MISSING")
                    for field in ("value","measured_sum","observed_peak"):
                        number=measurement.get(field)
                        if number is not None and (type(number) not in {int,float} or not math.isfinite(number) or number<0):
                            raise ValueError("RESOURCE_MEASUREMENT_NUMBER_INVALID")
                    partial=measurement["observed_peak" if measurement["aggregation"]=="max" else "measured_sum"]
                    number=measurement["value"]
                    if measurement["applicability"]=="not_applicable":
                        if eligible!=0 or measured!=0 or any(measurement.get(field) is not None for field in ("value","measured_sum","observed_peak")):
                            raise ValueError("RESOURCE_NOT_APPLICABLE_VALUES_INVALID")
                    elif measurement["complete"]:
                        if measurement["applicability"]!="applicable" or eligible==0 or measured!=eligible or number is None or number!=partial:
                            raise ValueError("RESOURCE_MEASUREMENT_COMPLETENESS_INVALID")
                    elif number is not None:
                        raise ValueError("RESOURCE_PARTIAL_TOTAL_NOT_NULL")
                    target=value.get("adapter_counters",{}) if collection=="counter_measurements" else value
                    if name not in target or target[name]!=number:
                        raise ValueError("RESOURCE_SCALAR_MEASUREMENT_MISMATCH")
                    reasons=measurement.get("reason_codes")
                    if not isinstance(reasons,list) or not all(isinstance(reason,str) for reason in reasons) or not isinstance(measurement.get("scope"),str) or not isinstance(measurement.get("unit"),str):
                        raise ValueError("RESOURCE_MEASUREMENT_SCOPE_INVALID")
            complete=all(m["complete"] and m["applicability"]!="unknown" for collection in ("measurements","counter_measurements") for m in value[collection].values() if m["applicability"]!="not_applicable")
            if value["measurement_complete"]!=complete:
                raise ValueError("RESOURCE_OVERALL_COMPLETENESS_INVALID")
    return value


def _error(kind,exc,attempt=None):
    return {"object":kind,"attempt":attempt,"code":str(exc) if isinstance(exc,ValueError) else type(exc).__name__,"error_type":type(exc).__name__}


def _artifact_hashes(folder,root,row,attempt):
    paths={name:folder/name for name in ("started.json","outcome.json","resources.json","replayed.json","backend-evidence.json","artifact-manifest.json")}
    paths["head.json"]=root/"heads"/(row["run_id"]+f"-{attempt:03d}.json")
    paths["receipt.json"]=root/"receipts"/(row["run_id"]+f"-{attempt:03d}.json")
    return {name:{"path":str(p.relative_to(root)),"sha256":file_hash(p)} for name,p in paths.items()
            if p.is_file() and not p.is_symlink() and not any(parent.is_symlink() for parent in p.parents)}


def _read_attempt(plan,root,row,folder,verify=True):
    attempt=int(folder.name.split("-")[-1])
    result={"run_id":row["run_id"],"attempt":attempt,"status":"CRASHED","execution_status":"CRASHED",
            "submission_status":"NOT_VERIFIED","resource_status":"NOT_MEASURED","replay_verified":False,
            "reports":row["planned_reports"],"resources":None,"error_type":None,"integrity_errors":[],
            "legacy_latest_attempt_policy":plan["schema"]==LEGACY_SCHEMA}
    outcome=None
    for kind,path in (("started",folder/"started.json"),("outcome",folder/"outcome.json")):
        try:
            value=_metadata(path,row,plan,attempt,kind,folder)
            if kind=="outcome":
                outcome=value
                result.update(status=value["status"],execution_status=value["status"],error_type=value.get("error_type"))
        except Exception as exc:
            result["integrity_errors"].append(_error(kind,exc,attempt))
            if path.exists() or path.is_symlink():
                result.update(status="METADATA_CORRUPT",execution_status="METADATA_CORRUPT",submission_status="METADATA_CORRUPT",error_type=type(exc).__name__)
    metadata_ok=not result["integrity_errors"]
    if outcome is not None and not metadata_ok:
        result.update(status="METADATA_CORRUPT",submission_status="METADATA_CORRUPT")
    manifest=None
    manifest_path=folder/"artifact-manifest.json"
    if plan["schema"]==SCHEMA or manifest_path.exists() or manifest_path.is_symlink():
        try:
            manifest=_metadata(folder/"artifact-manifest.json",row,plan,attempt,"manifest",folder)
            if manifest.get("schema")!="rswm-attempt-artifacts-v1" or not isinstance(manifest.get("files"),dict):
                raise ValueError("ARTIFACT_MANIFEST_INVALID")
            for name in ("started.json","outcome.json"):
                if manifest["files"].get(name)!=file_hash(folder/name):
                    raise ValueError("ARTIFACT_IDENTITY_MISMATCH")
        except Exception as exc:
            # An interrupted execution may not have written final metadata.
            result["integrity_errors"].append(_error("manifest",exc,attempt))
            metadata_ok=False
            if outcome is not None:
                result.update(status="METADATA_CORRUPT",submission_status="METADATA_CORRUPT")
    resource_path=folder/"resources.json"
    if resource_path.exists() or resource_path.is_symlink():
        try:
            resource=_metadata(resource_path,row,plan,attempt,"resource",folder)
            if manifest is not None and manifest["files"].get("resources.json")!=file_hash(resource_path):
                raise ValueError("RESOURCE_ARTIFACT_MISMATCH")
            result["resources"]=resource
            result["resource_status"]="COMPLETE" if resource.get("schema")=="rswm-resource-v3" and resource.get("measurement_complete",False) else "INCOMPLETE"
        except Exception as exc:
            result["resource_status"]="METADATA_CORRUPT"
            result["integrity_errors"].append(_error("resource",exc,attempt))
    evidence_path=folder/"backend-evidence.json"
    manifest_files=manifest.get("files",{}) if isinstance(manifest,dict) else {}
    if not isinstance(manifest_files,dict):manifest_files={}
    if (evidence_path.exists() or evidence_path.is_symlink() or
        "backend-evidence.json" in manifest_files or
        outcome is not None and "backend_evidence_status" in outcome):
        try:
            evidence=_metadata(evidence_path,row,plan,attempt,"backend_evidence",folder)
            if (manifest_files.get("backend-evidence.json")!=file_hash(evidence_path) or
                evidence.get("schema")!="rswm-retained-backend-observations-v1" or
                evidence.get("model_manifest_sha256")!=row["model_manifest_sha256"] or
                evidence.get("task_id")!=row["task_id"] or evidence.get("model_id")!=row["model_id"] or
                evidence.get("task_sha256")!=row["task_sha256"] or
                evidence.get("source_code_sha256")!=plan["source_code_sha256"] or
                evidence.get("inference_seed")!=row["seed"] or type(evidence.get("inference_seed")) is not int or
                evidence.get("research_acceptance")!="NOT_EVALUATED" or
                evidence.get("status") not in {"RETAINED_OBSERVATIONS","OBSERVATION_FAILED"}):
                raise ValueError("BACKEND_EVIDENCE_IDENTITY_MISMATCH")
            if evidence["status"]!="RETAINED_OBSERVATIONS" or not isinstance(evidence.get("observations"),dict):
                raise ValueError("BACKEND_EVIDENCE_UNAVAILABLE")
            result["backend_evidence"]={"path":str(evidence_path),"sha256":file_hash(evidence_path),
                                        "research_acceptance":"NOT_EVALUATED"}
        except Exception as exc:
            result["integrity_errors"].append(_error("backend_evidence",exc,attempt))
    if verify and metadata_ok and outcome is not None and outcome["replay_verified"]:
        try:
            from .replay import load_rgb_png
            head=_metadata(root/"heads"/(row["run_id"]+f"-{attempt:03d}.json"),row,plan,attempt,"head")
            if head.get("journal_head")!=outcome.get("replay_head"):
                raise ValueError("HEAD_IDENTITY_MISMATCH")
            receipt_path=root/"receipts"/(row["run_id"]+f"-{attempt:03d}.json")
            if plan["schema"]==SCHEMA or receipt_path.exists() or receipt_path.is_symlink():
                receipt=_metadata(receipt_path,row,plan,attempt,"receipt")
                if receipt.get("replay_head")!=head["journal_head"] or receipt.get("outcome_sha256")!=file_hash(folder/"outcome.json"):
                    raise ValueError("RECEIPT_IDENTITY_MISMATCH")
            submission=folder/"submission"
            if submission.is_symlink() or any(p.is_symlink() for p in submission.rglob("*")):
                raise ValueError("SUBMISSION_PATH_ESCAPE")
            if digest(_json(submission/"task.json"))!=row["task_sha256"]:
                raise ValueError("SUBMITTED_TASK_MISMATCH")
            initial=load_rgb_png(submission/"initial.png")
            if hashlib.sha256(initial.tobytes()).hexdigest()!=row["initial_sha256"]:
                raise ValueError("INITIAL_IDENTITY_MISMATCH")
            replayed=replay_run(plan["tasks"][row["task_id"]],initial,submission,expected_head=head["journal_head"])
            if replayed["seed"]!=row["seed"]:
                raise ValueError("INFERENCE_SEED_MISMATCH")
            slots=[{"stream":r["stream"],"event":r["event"],"valid":r["status"]=="valid"} for r in
                   (load_json(line) for line in (submission/"events.jsonl").read_text(encoding="utf-8").splitlines()) if r.get("kind")=="output"]
            result.update(replay_verified=True,submission_status="VERIFIED",reports=replayed["reports"],replay_head=head["journal_head"],output_slots=slots)
        except Exception as exc:
            result.update(status="FAILED_REPLAY",submission_status="FAILED_REPLAY",error_type=type(exc).__name__)
            result["integrity_errors"].append(_error("replay",exc,attempt))
    result["provenance"]=_artifact_hashes(folder,root,row,attempt)
    return result


def safe_resource_attempts(plan,root):
    """Every existing attempt remains in costs even when metadata is unreadable."""
    validate_plan(plan);root=Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("SUITE_ROOT_PATH_ESCAPE")
    root=root.resolve();_root_identity(plan,root)
    output=[]
    for row in plan["expected"]:
        try:
            folders=_attempt_folders(root/"runs"/row["run_id"])
        except Exception as exc:
            output.append({**{k:row[k] for k in ("run_id","study","model_id","world_id","city")},"attempt":None,
                           "status":"METADATA_CORRUPT","resources":None,"resource_status":"METADATA_CORRUPT","integrity_errors":[_error("attempt_path",exc)]})
            continue
        for folder in folders:
            record=_read_attempt(plan,root,row,folder,verify=False)
            output.append({**{k:row[k] for k in ("run_id","study","model_id","world_id","city")},
                           **(record.get("resources") or {}),**{k:record[k] for k in ("attempt","status","resources","resource_status","integrity_errors","provenance")}})
    return output


def _attempt(row, plan, root, attempt, factory):
    folder = root/"runs"/row["run_id"]/f"attempt-{attempt:03d}"
    if folder.is_symlink() or any(p.is_symlink() for p in folder.parents):
        raise ValueError("ATTEMPT_PATH_ESCAPE")
    folder.mkdir(parents=True,exist_ok=False)
    started = {"run_id":row["run_id"],"attempt":attempt,"status":"RUNNING","plan_sha256":plan["plan_sha256"]}
    _save(folder/"started.json",started,exclusive=True)
    _append(root/"receipts.jsonl",started)
    outcome = {**started,"run_dir":str(folder/"submission"),"reports":row["planned_reports"],
               "replay_verified":False,"replay_head":None,"error_type":None}
    config = plan["models"][row["model_id"]]["config"]
    collector = ResourceCollector(config.get("device","cpu"))
    collector.calls["planned_output_slots"]=sum(r["kind"]=="output" for r in schedule(plan["tasks"][row["task_id"]]))
    task = None
    journal = None
    model = None
    try:
        with collector.phase("input_validation"):
            task,labels = _load_task(row["task_path"])
            world = _world(row["world_path"])
            if digest(task)!=row["task_sha256"] or file_hash(row["world_path"])!=row["world_file_sha256"] or hashlib.sha256(world.tobytes()).hexdigest()!=row["world_pixel_sha256"]:
                raise ValueError("Frozen task/world identity mismatch")
            if digest(model_manifest(config))!=row["model_manifest_sha256"]:
                raise ValueError("Frozen model identity mismatch")
        outcome["status"] = "FAILED_LOAD"
        with collector.phase("model_load"):
            model = factory(config,world,task)
        outcome["status"] = "CRASHED"
        result,journal = run_task(task,world,model,row["seed"],labels,folder/"submission",resources=collector)
        outcome["status"] = "FAILED_REPLAY"
        head = {"run_id":row["run_id"],"attempt":attempt,"journal_head":journal.head,"plan_sha256":plan["plan_sha256"]}
        _save(root/"heads"/(row["run_id"]+f"-{attempt:03d}.json"),head,exclusive=True)
        with collector.phase("replay"):
            replayed = replay_run(task,world[task["origin"][1]-120:task["origin"][1]+120,
                                          task["origin"][0]-208:task["origin"][0]+208].copy(),
                                  folder/"submission",expected_head=head["journal_head"])
        _save(folder/"replayed.json",replayed,exclusive=True)
        failure_types = [r["error"] for r in journal.rows if r.get("error")]
        outcome.update(status=_failure_from_journal(journal,replayed["reports"]),reports=replayed["reports"],error_type=failure_types[0] if failure_types else None,
                       replay_verified=True,replay_head=head["journal_head"])
        _slots(task,folder,outcome["status"],journal)
    except Exception as exc:
        if outcome["status"]=="RUNNING":
            outcome["status"]="FAILED_INPUT"
        outcome["error_type"]=type(exc).__name__
        if task is None:
            # The frozen geometry is separately retained in plan.json by execution.
            task = plan["tasks"][row["task_id"]]
        _slots(task,folder,outcome["status"])
    except BaseException as exc:
        status="INTERRUPTED" if plan["schema"]==SCHEMA and isinstance(exc,KeyboardInterrupt) else "CRASHED"
        outcome.update(status=status,error_type=type(exc).__name__,replay_verified=False,reports=row["planned_reports"])
        _slots(plan["tasks"][row["task_id"]],folder,status)
        raise
    finally:
        outcome["execution_status"]=outcome["status"]
        outcome["submission_status"]="VERIFIED" if outcome["replay_verified"] else "NOT_VERIFIED"
        observer=getattr(model,"generation_evidence",None)
        if callable(observer):
            evidence={"schema":"rswm-retained-backend-observations-v1",
                      "run_id":row["run_id"],"attempt":attempt,"plan_sha256":plan["plan_sha256"],
                      "task_id":row["task_id"],"model_id":row["model_id"],
                      "model_manifest_sha256":row["model_manifest_sha256"],"task_sha256":row["task_sha256"],
                      "source_code_sha256":plan["source_code_sha256"],"inference_seed":row["seed"],
                      "research_acceptance":"NOT_EVALUATED"}
            try:
                observations=observer()
                if observations is not None:
                    if not isinstance(observations,dict):raise ValueError("Backend observations must be an object")
                    evidence.update(status="RETAINED_OBSERVATIONS",observations=observations)
                    # Validate finite JSON before opening the retained artifact.
                    canonical(evidence)
            except Exception as exc:
                evidence.update(status="OBSERVATION_FAILED",observations=None,error_type=type(exc).__name__)
            if "status" in evidence:
                _save(folder/"backend-evidence.json",evidence,exclusive=True)
                outcome["backend_evidence_status"]=evidence["status"]
        try:
            observed=journal.rows if journal is not None else []
            journal_path=folder/"submission"/"events.jsonl"
            if journal is None and journal_path.exists():
                observed=[load_json(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
            collector.calls["valid_outputs"]=sum(r.get("kind")=="output" and r.get("status")=="valid" for r in observed)
        except Exception as exc:
            collector.calls["valid_outputs"]=None
            collector.unavailable["valid_outputs"]="JOURNAL_OBSERVATION_FAILED: "+type(exc).__name__
        try:
            collector.execution_complete=outcome["status"]=="COMPLETED"
            resource=collector.finish(folder)
            resource.update(run_id=row["run_id"],attempt=attempt,plan_sha256=plan["plan_sha256"])
            _save(folder/"resources.json",resource)
            outcome["resource_path"]=str(folder/"resources.json")
            outcome["resource_status"]="COMPLETE" if resource.get("measurement_complete",False) else "INCOMPLETE"
        except Exception as exc:
            outcome.update(resource_path=None,resource_status="METADATA_CORRUPT",resource_error_type=type(exc).__name__)
        _save(folder/"outcome.json",outcome,exclusive=True)
        receipt={k:outcome[k] for k in ("run_id","attempt","status","plan_sha256","replay_head","error_type")}
        receipt["outcome_sha256"]=file_hash(folder/"outcome.json")
        _save(root/"receipts"/(row["run_id"]+f"-{attempt:03d}.json"),receipt,exclusive=True)
        _append(root/"receipts.jsonl",receipt)
        manifest={"schema":"rswm-attempt-artifacts-v1","run_id":row["run_id"],"attempt":attempt,"plan_sha256":plan["plan_sha256"],
                  "files":{name:file_hash(folder/name) for name in ("started.json","outcome.json","resources.json","replayed.json","backend-evidence.json") if (folder/name).exists()}}
        _save(folder/"artifact-manifest.json",manifest,exclusive=True)
    return outcome


def execute_plan(plan, out, resume=False, retry_failed=False, factory=None):
    validate_plan(plan)
    if retry_failed and not resume:
        raise ValueError("Explicit retry requires resume")
    original=Path(out)
    if original.is_symlink() or any(p.is_symlink() for p in original.parents):
        raise ValueError("Invalid suite root path")
    root=original.resolve();root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock_path=root/"coordinator.lock"
    if lock_path.is_symlink():
        raise ValueError("Invalid coordinator lock")
    lock=lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another suite coordinator is active") from exc
        # The lock is held before inspecting or initializing root identity.
        existing=[p for p in root.iterdir() if p.name!="coordinator.lock"]
        if existing:
            if not resume:
                raise FileExistsError("Use explicit resume; existing runs are never overwritten")
            _root_identity(plan,root)
        else:
            if resume:
                raise ValueError("Cannot resume an uninitialized suite")
            _save(root/"plan.json",plan,exclusive=True)
            write_expected(plan["expected"],root/"expected.jsonl")
        factory=factory or model_factory
        outcomes=[]
        for row in plan["expected"]:
            try:
                old=_attempt_folders(root/"runs"/row["run_id"])
            except Exception as exc:
                outcomes.append({"run_id":row["run_id"],"attempt":None,"status":"METADATA_CORRUPT",
                                 "plan_sha256":plan["plan_sha256"],"reports":row["planned_reports"],
                                 "replay_verified":False,"error_type":type(exc).__name__})
                continue
            if old:
                last=old[-1];attempt=int(last.name.split("-")[-1])
                if (last/"outcome.json").exists():
                    try:
                        outcome=_metadata(last/"outcome.json",row,plan,attempt,"outcome",last)
                    except Exception as exc:
                        outcomes.append({"run_id":row["run_id"],"attempt":attempt,"status":"METADATA_CORRUPT",
                                         "reports":row["planned_reports"],"replay_verified":False,"error_type":type(exc).__name__})
                        continue
                else:
                    # SIGKILL and hard exit leave a started attempt without final outcome.
                    outcome={"run_id":row["run_id"],"attempt":attempt,"status":"CRASHED",
                             "execution_status":"CRASHED","submission_status":"NOT_VERIFIED","resource_status":"NOT_MEASURED",
                             "plan_sha256":plan["plan_sha256"],"reports":row["planned_reports"],
                             "replay_verified":False,"replay_head":None,"error_type":"InterruptedAttempt",
                             "run_dir":str(last/"submission"),"resource_path":None}
                    _slots(plan["tasks"][row["task_id"]],last,"CRASHED")
                    _save(last/"outcome.json",outcome,exclusive=True)
                    receipt={k:outcome[k] for k in ("run_id","attempt","status","plan_sha256","replay_head","error_type")}
                    receipt["outcome_sha256"]=file_hash(last/"outcome.json")
                    _save(root/"receipts"/(row["run_id"]+f"-{attempt:03d}.json"),receipt,exclusive=True)
                    _append(root/"receipts.jsonl",receipt)
                    _save(last/"artifact-manifest.json",{"schema":"rswm-attempt-artifacts-v1", "run_id":row["run_id"],"attempt":attempt,
                          "plan_sha256":plan["plan_sha256"],"files":{n:file_hash(last/n) for n in ("started.json","outcome.json") if (last/n).exists()}},exclusive=True)
                if not retry_failed or outcome["status"]=="COMPLETED":
                    outcomes.append(outcome);continue
                if plan["schema"]==SCHEMA:
                    policy=plan["retry_policy"]
                    code=None
                    if attempt+1>=policy["max_attempts_per_run"]:
                        code="RETRY_BUDGET_EXHAUSTED"
                    elif outcome["status"] not in policy["allowed_retry_statuses"]:
                        code="RETRY_STATUS_NOT_ALLOWED"
                    if code:
                        _append(root/"retry-events.jsonl",{"run_id":row["run_id"],"attempt":attempt,"plan_sha256":plan["plan_sha256"],"code":code,"status":outcome["status"]})
                        raise ValueError(code)
                attempt+=1
            else:
                attempt=0
            outcomes.append(_attempt(row,plan,root,attempt,factory))
        _save(root/"replay-status.json",outcomes)
        _save_jsonl(root/"run-index.jsonl",outcomes)
        _save_jsonl(root/"replay-status.jsonl",outcomes)
        return outcomes
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


def _save_jsonl(path,rows):
    path=Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("JSONL_PATH_ESCAPE")
    temp=path.with_name(path.name+".tmp-"+uuid.uuid4().hex)
    with temp.open("x",encoding="utf-8") as stream:
        stream.write("".join(canonical(r).decode()+"\n" for r in rows))
        stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def replay_suite(plan, root, selection=None):
    """Authenticate each selected attempt, retaining every frozen run denominator."""
    validate_plan(plan);root=Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise ValueError("SUITE_ROOT_PATH_ESCAPE")
    root=root.resolve()
    if selection is None:
        selection="legacy_latest_attempt_policy" if plan["schema"]==LEGACY_SCHEMA else "first_attempt_only"
    if selection not in {"first_attempt_only","latest_attempt_within_budget","legacy_latest_attempt_policy"}:
        raise ValueError("Unknown analysis attempt selection")
    if selection=="legacy_latest_attempt_policy" and plan["schema"]!=LEGACY_SCHEMA:
        raise ValueError("Legacy latest policy is only supported for v2")
    records=[]
    with suite_read_lock(root):
        _root_identity(plan,root)
        for row in plan["expected"]:
            result={"run_id":row["run_id"],"attempt":None,"status":"NOT_RUN","execution_status":"NOT_RUN",
                    "submission_status":"NOT_VERIFIED","resource_status":"NOT_MEASURED","replay_verified":False,
                    "reports":row["planned_reports"],"resources":None,"error_type":None,"integrity_errors":[],
                    "legacy_latest_attempt_policy":plan["schema"]==LEGACY_SCHEMA,"selection":selection}
            try:
                folders=_attempt_folders(root/"runs"/row["run_id"])
                if plan["schema"]==SCHEMA and any(int(p.name.split("-")[-1])>=plan["retry_policy"]["max_attempts_per_run"] for p in folders):
                    result["integrity_errors"].append({"object":"attempt_path","attempt":None,"code":"UNEXPECTED_ATTEMPT_BUDGET","error_type":"ValueError"})
                if selection=="first_attempt_only":
                    if folders and not any(int(p.name.split("-")[-1])==0 for p in folders):
                        result["integrity_errors"].append({"object":"attempt_path","attempt":0,"code":"FIRST_ATTEMPT_MISSING","error_type":"ValueError"})
                    folders=[p for p in folders if int(p.name.split("-")[-1])==0]
                elif plan["schema"]==SCHEMA:
                    folders=[p for p in folders if int(p.name.split("-")[-1])<plan["retry_policy"]["max_attempts_per_run"]]
                if folders:
                    prior_errors=result["integrity_errors"]
                    result.update(_read_attempt(plan,root,row,folders[-1]))
                    result["integrity_errors"]=prior_errors+result["integrity_errors"]
            except Exception as exc:
                result.update(status="METADATA_CORRUPT",execution_status="METADATA_CORRUPT",submission_status="METADATA_CORRUPT",error_type=type(exc).__name__)
                result["integrity_errors"].append(_error("attempt_path",exc))
            records.append(result)
    return records


def integrity_summary(plan,root,records):
    root=Path(root)
    expected={r["run_id"] for r in plan["expected"]}
    unexpected=[]
    runs=root/"runs"
    if runs.exists():
        unexpected=[p.name for p in runs.iterdir() if p.name not in expected or p.is_symlink() or not p.is_dir()]
    errors=[dict(run_id=r["run_id"],**e) for r in records for e in r.get("integrity_errors",[])]
    # Acceptance covers the frozen all-started-attempt cost scope. A healthy
    # latest result cannot certify an older attempt's missing/corrupt resources.
    with suite_read_lock(root):
        attempts=safe_resource_attempts(plan,root)
    historical_errors=[dict(run_id=r["run_id"],scope="all_attempt_metadata",**e)
                       for r in attempts for e in r.get("integrity_errors",[])]
    incomplete_attempts=sum(r.get("resource_status")!="COMPLETE" for r in attempts)
    return {"schema":"rswm-integrity-summary-v1","plan_sha256":plan["plan_sha256"],"planned_runs":len(expected),
            "verified_submissions":sum(r["replay_verified"] for r in records),"errors":errors,"unexpected_runs":sorted(unexpected),
            "resource_incomplete_runs":sum(r.get("resource_status")!="COMPLETE" for r in records),
            "resource_incomplete_attempts":incomplete_attempts,"all_attempt_errors":historical_errors,
            "analysis_written":True,"accepted":not unexpected and not errors and not historical_errors and not incomplete_attempts and all(r["status"]=="COMPLETED" and r.get("resource_status")=="COMPLETE" for r in records),
            "trust_scope":"retained hashes check internal identity; same-disk receipts are not trusted timestamps or source authenticity"}


def plan_from_indexes(task_index, models, seeds, code_commit, world_index=None, world_dir=None, role=None, retry_policy=None):
    path = Path(task_index).resolve();data = _json(path)
    wd = _json(world_index) if world_index else data
    world_rows = wd["worlds"] if isinstance(wd,dict) else wd
    worlds = {r["world_id"]:r for r in world_rows}
    if len(worlds)!=len(world_rows):
        raise ValueError("Duplicate world index entries")
    root = Path(world_dir).resolve() if world_dir else (Path(world_index).resolve().parent if world_index else path.parent/"worlds")
    tasks = []
    for row in data["tasks"]:
        tp = path.parent/row.get("task_path",row.get("path",""))
        task,_ = _load_task(tp)
        if role and task["role"]!=role:
            continue
        wr = worlds[task["world_id"]]
        if row.get("city",wr["city"])!=wr["city"] or ("role" in wr and wr["role"]!=task["role"]):
            raise ValueError("World/task index city or role mismatch")
        tasks.append({**row,"task_path":str(tp),"world_path":str(root/wr["path"]),
                      "city":row.get("city",wr["city"]),
                      "world_file_sha256":wr["world_file_sha256"],"world_pixel_sha256":wr["world_pixel_sha256"]})
    plan = build_plan(tasks,_json(models),seeds,data.get("source_manifest_sha256",data.get("manifest_sha256")),
                      code_commit,study=data.get("study","core"),retry_policy=retry_policy)
    plan.pop("plan_sha256")
    plan["materialized_indexes"]={"task_index_sha256":file_hash(task_index),"model_config_sha256":file_hash(models)}
    if world_index:
        plan["materialized_indexes"]["world_index_sha256"]=file_hash(world_index)
    plan["plan_sha256"]=digest(plan)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("plan")
    p.add_argument("--task-index",required=True);p.add_argument("--world-index");p.add_argument("--world-dir")
    p.add_argument("--models",required=True);p.add_argument("--seeds",type=int,nargs="+",required=True)
    p.add_argument("--code-commit",required=True);p.add_argument("--role");p.add_argument("--out",required=True)
    p.add_argument("--max-attempts-per-run",type=int,default=1)
    p.add_argument("--allowed-retry-statuses",nargs="+",default=DEFAULT_RETRY_POLICY["allowed_retry_statuses"])
    p = sub.add_parser("run");p.add_argument("--plan",required=True);p.add_argument("--out",required=True)
    p.add_argument("--resume",action="store_true");p.add_argument("--retry-failed",action="store_true")
    for name in ("verify","inspect"):
        p=sub.add_parser(name);p.add_argument("--plan",required=True);p.add_argument("--suite",required=True)
        p.add_argument("--selection",choices=["first_attempt_only","latest_attempt_within_budget"])
        p.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        if args.command=="plan":
            policy=dict(DEFAULT_RETRY_POLICY,max_attempts_per_run=args.max_attempts_per_run,allowed_retry_statuses=args.allowed_retry_statuses)
            plan = plan_from_indexes(args.task_index,args.models,args.seeds,args.code_commit,args.world_index,args.world_dir,args.role,policy)
            save_plan(plan,args.out)
            print(json.dumps({"expected_runs":len(plan["expected"]),"plan_sha256":plan["plan_sha256"]}))
            return 0
        plan=_json(args.plan)
        if args.command in {"verify","inspect"}:
            records=replay_suite(plan,args.suite,args.selection)
            summary=integrity_summary(plan,args.suite,records)
            summary["records"]=records
            if args.out:
                _save(args.out,summary,exclusive=True)
            print(json.dumps({k:v for k,v in summary.items() if k!="records"}))
            return 0 if summary["accepted"] else 1
        if args.retry_failed and not args.resume:
            parser.error("--retry-failed requires --resume")
        old_handler=None
        if threading.current_thread() is threading.main_thread():
            old_handler=signal.getsignal(signal.SIGTERM)
            def interrupted(signum,frame):
                raise KeyboardInterrupt("SIGTERM")
            signal.signal(signal.SIGTERM,interrupted)
        try:
            outcomes=execute_plan(plan,args.out,args.resume,args.retry_failed)
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGTERM,old_handler)
        complete=sum(r["status"]=="COMPLETED" for r in outcomes)
        inspected=replay_suite(plan,args.out,"latest_attempt_within_budget" if args.retry_failed else None)
        summary=integrity_summary(plan,args.out,inspected)
        _save(Path(args.out)/"integrity-summary.json",summary)
        print(json.dumps({"planned":len(outcomes),"completed":complete,"accepted":summary["accepted"],"legacy_latest_attempt_policy":plan["schema"]==LEGACY_SCHEMA}))
        return 0 if summary["accepted"] else 1
    except KeyboardInterrupt:
        print(json.dumps({"error_code":"INTERRUPTED","status":"INTERRUPTED"}),file=sys.stderr)
        return 1
    except (ValueError,KeyError,TypeError,FileNotFoundError,FileExistsError,RuntimeError) as exc:
        print(json.dumps({"error_code":"CONFIG_OR_ROOT_IDENTITY_ERROR","error_type":type(exc).__name__,"message":str(exc)}),file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
