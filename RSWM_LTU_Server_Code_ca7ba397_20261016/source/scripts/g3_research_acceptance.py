"""Execute synthetic G3 plan/run/verify/export through the actual public CLIs.

Raw submissions go to --work-dir; compact provenance/logs go to --evidence-dir.
This is engineering evidence, never a real Inria or model-quality acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from rswm_inria.geometry import digest, file_hash
from rswm_inria.tasks import packages
from rswm_inria.suite import code_fingerprint


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir",type=Path,required=True)
    parser.add_argument("--evidence-dir",type=Path,required=True)
    parser.add_argument("--adapters",choices=["oracle","raster","frontier","lazy"],nargs="+",
                        default=["oracle","raster","frontier","lazy"])
    args=parser.parse_args()
    args.work_dir.mkdir(parents=True,exist_ok=False)
    args.evidence_dir.mkdir(parents=True,exist_ok=False)
    root=args.work_dir.resolve(); evidence=args.evidence_dir.resolve()
    inputs=root/"inputs";inputs.mkdir()
    world=np.full((2048,2048,3),91,np.uint8)
    Image.fromarray(world).save(inputs/"world.png")
    taskrows=[]
    for family in ("F2","F3","F5","F7"):
        task,labels=next((t,l) for t,l in packages("synthetic-g3-cli","diagnostic_only",0) if t["family"]==family)
        private={}
        for name, arrays in labels.items():
            path=inputs/(task["task_id"]+"-"+name+".npz")
            np.savez_compressed(path,**arrays)
            private[name]={"path":path.name,"sha256":file_hash(path)}
        task["private_label_files"]=private
        taskpath=inputs/(task["task_id"]+".json");save(taskpath,task)
        # The suite requires a known city key even for diagnostic fixtures.
        # This is an administrative placeholder, not a Kitsap source image.
        taskrows.append({"path":taskpath.name,"city":"kitsap","role":"diagnostic_only"})
    source={"schema":"rswm-synthetic-g3-source-v1","real_data":False,"value":91,"shape":[2048,2048,3]}
    save(inputs/"index.json",{"tasks":taskrows,"source_manifest_sha256":digest(source),"worlds":[{
        "world_id":"synthetic-g3-cli","city":"kitsap","role":"diagnostic_only","path":"world.png",
        "world_file_sha256":file_hash(inputs/"world.png"),"world_pixel_sha256":hashlib.sha256(world.tobytes()).hexdigest()}]})
    if len(args.adapters)!=len(set(args.adapters)):
        parser.error("duplicate adapters are unsupported")
    models=[{"model_id":name,"adapter":name,**({} if name=="oracle" else
        {"diagnostic_noise":True,"device":"cpu"})} for name in args.adapters]
    save(inputs/"models.json",models)
    sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    fingerprint=code_fingerprint(); commands=[]
    def command(name,parts):
        cmd=[sys.executable]+parts
        result=subprocess.run(cmd,capture_output=True,text=True)
        (evidence/(name+".stdout.log")).write_text(result.stdout)
        (evidence/(name+".stderr.log")).write_text(result.stderr)
        commands.append({"name":name,"command":cmd,"exit_code":result.returncode})
        save(evidence/"commands.json",commands)
        if result.returncode:
            raise RuntimeError(name+" CLI failed; inspect real logs")
    command("plan",["-m","rswm_inria.suite","plan","--task-index",str(inputs/"index.json"),
        "--world-dir",str(inputs),"--models",str(inputs/"models.json"),"--seeds","0",
        "--code-commit",sha,"--out",str(root/"plan.json")])
    command("run",["-m","rswm_inria.suite","run","--plan",str(root/"plan.json"),"--out",str(root/"suite")])
    command("verify",["-m","rswm_inria.suite","verify","--plan",str(root/"plan.json"),"--suite",str(root/"suite"),
        "--out",str(root/"verified.json")])
    command("export",["-m","rswm_inria.export_tables","--plan",str(root/"plan.json"),"--suite",str(root/"suite"),
        "--out",str(root/"analysis"),"--bootstrap","10"])
    end_fingerprint=code_fingerprint()
    if fingerprint!=end_fingerprint:
        raise RuntimeError("Source changed during acceptance; evidence is intermediate and must be rerun")
    plan=json.loads((root/"plan.json").read_text()); verified=json.loads((root/"verified.json").read_text())
    analysis=json.loads((root/"analysis"/"aggregation.json").read_text())
    expected_runs=4*len(models); expected_outputs=352*len(models)
    assert len(plan["expected"])==expected_runs and verified["accepted"]
    assert analysis["acceptance_status"]=="ACCEPTED"
    assert all(r["replay_verified"] for r in verified["records"])
    assert sum(r["resources"]["accounting_calls"]["valid_outputs"] for r in verified["records"])==expected_outputs
    for cell in analysis["cells"]:
        if cell["model_id"]=="oracle" and cell["coverage"]:
            assert cell["mae"]==0
    hashes={str(p.relative_to(root)):file_hash(p) for p in sorted(root.rglob("*")) if p.is_file()}
    save(evidence/"artifact-hashes.json",hashes)
    summary={"source_code_sha256":fingerprint,"base_commit":sha,"research_source_commit":"656ed3f",
        "adapters":args.adapters,"planned_runs":expected_runs,"completed_runs":expected_runs,
        "planned_outputs":expected_outputs,"valid_outputs":expected_outputs,
        "first_attempt_selection":analysis["primary_selection"],"accepted":verified["accepted"],
        "measurement_complete_runs":sum(r["resource_status"]=="COMPLETE" for r in verified["records"]),
        "data_kind":"synthetic constant RGB","city_key_scope":"kitsap administrative fixture placeholder; no Inria pixels",
        "oracle_scope":"evaluator-only",
        "frontier_lazy_completer":"diagnostic noise","real_data_validation":"NOT_RUN",
        "real_weights_validation":"BLOCKED","gpu_validation":"NOT_RUN",
        "raw_artifacts":str(root),"source_authenticity":"retained internal hashes; not external timestamp authentication"}
    save(evidence/"summary.json",summary)
    print(json.dumps(summary,sort_keys=True))


if __name__=="__main__":
    main()
