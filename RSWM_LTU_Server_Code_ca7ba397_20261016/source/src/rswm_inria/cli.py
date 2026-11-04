"""Command line entry points; no automatic full-dataset or model download."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
from .data import freeze_metadata,select_manifest,download_sources,build_world,write_json,verify_manifest,INRIA_REVISION
from .geometry import VERSION,canonical,file_hash,ORIGINS,rng_for
from .tasks import packages,training_clips
from .runner import run_task
from .baselines import CopyModel,RandomModel,StaticNoiseMap,OracleMap,WindowMap,DiffusersCompleter


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_packages(manifest,out):
    verify_manifest(manifest);root=Path(out);root.mkdir(parents=True,exist_ok=True);index=[]
    for row in manifest["worlds"]:
        for origin in range(4):
            for task,labels in packages(row["world_id"],row["role"],origin):
                label_paths={}
                for stream,arrays in labels.items():
                    path=root/f"{task['task_id']}-{stream}.npz"
                    if path.exists():
                        with np.load(path,allow_pickle=False) as stored:
                            if set(stored.files)!=set(arrays) or any(not np.array_equal(stored[k],v) for k,v in arrays.items()):
                                raise FileExistsError(f"Different existing labels: {path}")
                    else: np.savez_compressed(path,**arrays)
                    label_paths[stream]={"path":path.name,"sha256":file_hash(path)}
                task["private_label_files"]=label_paths;path=root/(task["task_id"]+".json")
                write_json(path,task);index.append({"task_id":task["task_id"],"path":path.name,"family":task["family"],
                                                   "world_id":row["world_id"],"role":row["role"],"city":row["city"]})
    result={"manifest_sha256":manifest["manifest_sha256"],"task_packages":len(index),"tasks":index}
    write_json(root/"index.json",result);return result


def audit(worlds=120):
    proposals=Counter();orientations=Counter();gap_min=10000;gap_max=0;re_min=10000;re_max=0;count=0
    for i in range(worlds):
        for origin in range(4):
            bundle=packages(f"geometry-fixture-{i:03d}","audit_only",origin)
            for task,labels in bundle:
                count+=1;proposals[str(task["metadata"]["proposal_count"])]+=1
                if task["family"]=="F5": orientations[str(task["metadata"]["orientation"])]+=1
                for arrays in labels.values():
                    gap_min=min(gap_min,int(arrays["invisible_query_gap"].min()));gap_max=max(gap_max,int(arrays["invisible_query_gap"].max()))
                    re_min=min(re_min,int(arrays["reentry_event"].min()));re_max=max(re_max,int(arrays["reentry_event"].max()))
    return {"version":VERSION,"kind":"geometry_only_synthetic_world_identifiers","world_identifiers":worlds,
            "task_packages":count,"unique_new_output_events":worlds*4*352,
            "proposal_histogram":dict(proposals),"f5_orientation_histogram":dict(orientations),
            "invisible_query_gap_range":[gap_min,gap_max],"reentry_event_range":[re_min,re_max],
            "real_images_downloaded":0,"hf_source_manifest_frozen":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("freeze");p.add_argument("--revision",default=INRIA_REVISION);p.add_argument("--inventory");p.add_argument("--out",required=True)
    p=sub.add_parser("first-source");p.add_argument("--manifest",required=True);p.add_argument("--inventory",required=True)
    p.add_argument("--source-dir",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("download");p.add_argument("--manifest",required=True);p.add_argument("--out",required=True);p.add_argument("--limit",type=int)
    p=sub.add_parser("worlds");p.add_argument("--manifest",required=True);p.add_argument("--source-dir",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("tasks");p.add_argument("--manifest",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("training-clips");p.add_argument("--manifest",required=True);p.add_argument("--out",required=True)
    p=sub.add_parser("audit");p.add_argument("--worlds",type=int,default=120);p.add_argument("--out",required=True)
    p=sub.add_parser("demo");p.add_argument("--out",required=True);p.add_argument("--seed",type=int,default=0)
    p.add_argument("--model",choices=["copy","random","noise-map","oracle","lazy-map","eager-map"],default="oracle")
    p.add_argument("--origin",type=int,choices=range(4),default=0);p.add_argument("--save-images",action="store_true")
    p=sub.add_parser("run");p.add_argument("--task",required=True);p.add_argument("--world",required=True);p.add_argument("--out",required=True)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--model",choices=["copy","random","noise-map","oracle","lazy-map","eager-map","wan-i2v"],required=True)
    p.add_argument("--model-dir");p.add_argument("--device",default="cuda");p.add_argument("--steps",type=int,default=30)
    args=parser.parse_args(argv)
    if args.command=="freeze":
        inventory=read_json(args.inventory) if args.inventory else freeze_metadata(args.revision)
        if inventory.get("revision")!=args.revision:
            raise ValueError("Frozen inventory revision differs from --revision.")
        result=select_manifest(inventory);write_json(Path(args.out).with_name("source-inventory.json"),inventory);write_json(args.out,result)
    elif args.command=="first-source":
        from .materialized import first_source
        result=first_source(args.manifest,args.inventory,args.source_dir,args.out)
    elif args.command=="download":
        if args.limit is not None and args.limit<=0: parser.error("--limit must be positive")
        result={"downloaded":download_sources(read_json(args.manifest),args.out,args.limit)}
    elif args.command=="worlds":
        m=read_json(args.manifest);verify_manifest(m)
        result={"worlds":[build_world(Path(args.source_dir)/r["path"],r,args.out) for r in m["worlds"]]}
        write_json(Path(args.out)/"index.json",result)
    elif args.command=="tasks": result=save_packages(read_json(args.manifest),args.out)
    elif args.command=="training-clips":
        m=read_json(args.manifest);verify_manifest(m)
        clips=[clip for row in m["worlds"] if row["role"]=="train" for clip in training_clips(row["world_id"],row["role"])]
        result={"manifest_sha256":m["manifest_sha256"],"clips":clips,"clip_count":len(clips),"view_events":len(clips)*33};write_json(args.out,result)
    elif args.command=="audit":
        if args.worlds<=0:parser.error("--worlds must be positive")
        result=audit(args.worlds);write_json(args.out,result)
    else:
        if args.command=="demo":
            world=rng_for("synthetic-world",0).integers(0,256,(2048,2048,3),dtype=np.uint8)
            bundle=packages("synthetic-demo","diagnostic_only",args.origin)
        else:
            task=read_json(args.task);root=Path(args.task).parent;labels={}
            for name,info in task.get("private_label_files",{}).items():
                path=root/info["path"]
                if file_hash(path)!=info["sha256"]:raise ValueError("Private label checksum mismatch.")
                with np.load(path,allow_pickle=False) as z:labels[name]={key:z[key] for key in z.files}
            with Image.open(args.world) as im:
                if im.mode!="RGB":raise ValueError("World must be RGB.")
                world=np.asarray(im).copy()
            bundle=[(task,labels)]
        results=[];root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
        for task,labels in bundle:
            model={"copy":CopyModel,"random":RandomModel,"noise-map":StaticNoiseMap}.get(args.model,lambda:None)()
            if args.model=="oracle":model=OracleMap(world,task["origin"])
            if args.model=="wan-i2v":
                from .video import WanI2VAdapter
                model=WanI2VAdapter(args.model_dir,args.device,args.steps)
            if args.model in ("lazy-map","eager-map"):
                completer=DiffusersCompleter(args.model_dir,args.device,args.steps) if getattr(args,"model_dir",None) else None
                model=WindowMap(completer,eager=args.model=="eager-map")
            save=args.command=="run" or args.save_images
            report,journal=run_task(task,world,model,args.seed,labels,root/task["task_id"] if save else None)
            report["model"]=args.model;report["input_kind"]="synthetic" if args.command=="demo" else "user_provided_world"
            results.append(report)
        result={"runs":results};write_json(root/"results.json",result)
    print(json.dumps({"command":args.command,"status":"completed","output":args.out},ensure_ascii=False))
    return 0


if __name__=="__main__":
    main()
