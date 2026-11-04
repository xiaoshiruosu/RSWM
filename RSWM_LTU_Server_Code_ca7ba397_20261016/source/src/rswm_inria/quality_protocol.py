"""Geometry-frozen unknown-area tiles and immutable first-commit mosaics.

Production assembly accepts independently replayed submissions only. F7 shared
prefix tiles are sampled once; mixed branch tiles retain branch-specific lineage.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import uuid
import numpy as np
from .geometry import WORLD, box, canonical, digest, file_hash, rgb
from .quality import plan_tiles
from .tasks import validate_task
from .replay import load_json, load_rgb_png

PATCH_SCHEMA="rswm-quality-patch-plan-v1"
MOSAIC_SCHEMA="rswm-quality-mosaics-v1"
SELECTION_NAMESPACE="rswm-quality-grid128-first-planned-v1"


def _first_map(queries,origin):
    first=np.full((WORLD,WORLD),-1,np.int16)
    x0,y0,x1,y1=box((0,0),origin);first[y0:y1,x0:x1]=0
    for event,q in enumerate(queries[1:],1):
        x0,y0,x1,y1=box(q,origin)
        part=first[y0:y1,x0:x1];part[part<0]=event
    return first


def _lineages(task):
    if task["family"]=="F7":
        return {"prefix":task["programs"]["a"][:33],"a":task["programs"]["a"],"b":task["programs"]["b"]}
    return {name:task["programs"][name] for name in (("small","large") if task["family"]=="F3" else ("main",))}


def _source_indices(task,lineage,xy,size=128,first_map=None):
    first=_first_map(_lineages(task)[lineage],task["origin"]) if first_map is None else first_map
    x,y=xy;events=np.ascontiguousarray(first[y:y+size,x:x+size],dtype="<i2")
    if events.shape!=(size,size) or (events<=0).any():
        raise ValueError("Tile includes I0 or pixels outside the planned unknown union")
    streams=np.zeros((size,size),np.uint8)
    if task["family"]=="F7":
        names={0:"prefix",1:"a",2:"b"}
        if lineage!="prefix":streams[events>32]=1 if lineage=="a" else 2
    else:names={0:lineage}
    return events,streams,names


def _source_hash(events,streams):
    return hashlib.sha256(events.astype("<i2",copy=False).tobytes()+streams.astype(np.uint8,copy=False).tobytes()).hexdigest()


def build_patch_plan(tasks,count=16,namespace=SELECTION_NAMESPACE,world_metadata=None):
    if type(count) is not int or count<1 or not namespace:
        raise ValueError("Positive fixed per-lineage count and namespace required")
    metadata=world_metadata or {};stored={};patches=[];lineage_rows=[];world_roles={}
    for task in sorted(tasks,key=lambda t:t["task_id"]):
        validate_task(task)
        if task["task_id"] in stored:raise ValueError("Duplicate task")
        if task["world_id"] in world_roles and world_roles[task["world_id"]]!=task["role"]:
            raise ValueError("Same source world assigned multiple roles")
        world_roles[task["world_id"]]=task["role"];stored[task["task_id"]]=task
        for lineage,queries in _lineages(task).items():
            ns=namespace+"/"+task["task_id"]+"/"+lineage
            # Reuse the established grid/hash rule, considering every candidate
            # before removing shared-prefix duplicates and applying the fixed cap.
            candidates=plan_tiles(queries,task["origin"],ns,count=(WORLD//128)**2,size=128)["tiles"]
            first=_first_map(queries,task["origin"])
            if task["family"]=="F7" and lineage!="prefix":
                candidates=[xy for xy in candidates if (first[xy[1]:xy[1]+128,xy[0]:xy[0]+128]>32).any()]
            lineage_rows.append({"task_id":task["task_id"],"lineage":lineage,"eligible_tiles":len(candidates),
                                 "planned_tiles":min(count,len(candidates)),"requested_tiles":count,"namespace":ns})
            for xy in candidates[:count]:
                events,streams,_=_source_indices(task,lineage,xy,first_map=first)
                identity={"world_id":task["world_id"],"task_id":task["task_id"],"origin_index":task["origin_index"],
                          "role":task["role"],"study":task.get("study","core"),"family":task["family"],
                          "variant":task.get("metadata",{}).get("variant","default"),"protocol":task["protocol"],"lineage":lineage,
                          "origin":task["origin"],"xy":xy,"size":128,"task_sha256":digest(task),
                          "first_source_sha256":_source_hash(events,streams)}
                patches.append({**identity,"patch_id":digest(identity),"planned_pixels":128**2,
                                "source_identity":metadata.get(task["world_id"],{}).get("source_identity",task["world_id"]),
                                "city":metadata.get(task["world_id"],{}).get("city")})
    plan={"schema":PATCH_SCHEMA,"algorithm":"grid128-before-output-first-planned-v1","namespace":namespace,
          "requested_tiles_per_lineage":count,"tile_size":128,"tasks":stored,"world_metadata":metadata,
          "patches":patches,"lineages":lineage_rows,"selection_depends_on_images":False,
          "f7_rule":"fully shared-prefix tiles sampled only as prefix; mixed tiles preserve independent branch lineage"}
    plan["patch_plan_sha256"]=digest(plan)
    return plan


def validate_patch_plan(plan):
    if not isinstance(plan,dict) or plan.get("schema")!=PATCH_SCHEMA or plan.get("tile_size")!=128:
        raise ValueError("Invalid patch plan schema")
    if digest({k:v for k,v in plan.items() if k!="patch_plan_sha256"})!=plan.get("patch_plan_sha256"):
        raise ValueError("Patch plan hash mismatch")
    rebuilt=build_patch_plan(list(plan["tasks"].values()),plan["requested_tiles_per_lineage"],plan["namespace"],plan["world_metadata"])
    if rebuilt["patch_plan_sha256"]!=plan["patch_plan_sha256"]:
        raise ValueError("Patch plan differs from deterministic frozen geometry")


def assemble_patch(patch,task,images,*,verified=False,diagnostic=False,seed=0,model_id=None,run_id=None):
    if not verified and not diagnostic:raise ValueError("Independently verified submission required")
    if digest(task)!=patch["task_sha256"]:raise ValueError("Frozen task mismatch")
    events,streams,names=_source_indices(task,patch["lineage"],patch["xy"])
    if _source_hash(events,streams)!=patch["first_source_sha256"]:raise ValueError("First-source identity mismatch")
    result=np.zeros((128,128,3),np.uint8);valid=np.zeros((128,128),bool)
    yy,xx=np.mgrid[:128,:128];wx=xx+patch["xy"][0];wy=yy+patch["xy"][1]
    programs=_lineages(task)
    for code in np.unique(streams):
        stream=names[int(code)]
        for event in np.unique(events[streams==code]):
            image=images.get((stream,int(event)))
            if image is None:continue  # The first failed event remains missing forever.
            image=rgb(image)
            q=programs[stream][int(event)]
            x0,y0,_,_=box(q,task["origin"])
            keep=(streams==code)&(events==event)
            ix,iy=wx[keep]-x0,wy[keep]-y0
            if (ix<0).any() or (ix>=416).any() or (iy<0).any() or (iy>=240).any():
                raise ValueError("First-source viewport mapping invalid")
            result[keep]=image[iy,ix];valid[keep]=True
    pixel_hash=hashlib.sha256(result.tobytes()).hexdigest();mask_hash=hashlib.sha256(valid.tobytes()).hexdigest()
    comparison=digest({"patch_id":patch["patch_id"],"inference_seed":seed})
    return {**patch,"seed":seed,"model_id":model_id,"run_id":run_id,"comparison_id":comparison,
            "sample_id":digest({"comparison_id":comparison,"run_id":run_id,"model_id":model_id}),
            "image":result,"valid_mask":valid,"first_event":events,"first_source_stream":streams,
            "complete":bool(valid.all()),"valid_pixels":int(valid.sum()),"pixel_sha256":pixel_hash,"mask_sha256":mask_hash,
            "patch_sha256":digest({"patch_id":patch["patch_id"],"pixel_sha256":pixel_hash,"mask_sha256":mask_hash,
                                   "first_source_sha256":patch["first_source_sha256"]}),
            "independently_verified":bool(verified),"diagnostic_features":bool(diagnostic)}


def _save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:f.write(canonical(value)+b"\n")


def save_mosaics(entries,out,patch_plan_sha256,suite_plan_sha256=None):
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise FileExistsError("Mosaic output must be empty")
    records=[]
    for index,entry in enumerate(entries):
        name=f"patch-{index:06d}.npz";path=root/name
        with path.open("xb") as f:np.savez_compressed(f,image=entry["image"],valid_mask=entry["valid_mask"],
                                                   first_event=entry["first_event"],first_source_stream=entry["first_source_stream"])
        records.append({**{k:v for k,v in entry.items() if not isinstance(v,np.ndarray)},
                        "data_file":name,"data_file_sha256":file_hash(path)})
    bundle={"schema":MOSAIC_SCHEMA,"patch_plan_sha256":patch_plan_sha256,"suite_plan_sha256":suite_plan_sha256,
            "patches":records,"planned_pixels":sum(r["planned_pixels"] for r in records),
            "valid_pixels":sum(r["valid_pixels"] for r in records),"planned_tiles":len(records),
            "complete_tiles":sum(r["complete"] for r in records),"release_validated":False}
    bundle["bundle_sha256"]=digest(bundle);_save_json(root/"mosaics.json",bundle)
    return bundle


def load_mosaics(path):
    path=Path(path);bundle=load_json(path.read_text(encoding="utf-8"))
    if bundle.get("schema")!=MOSAIC_SCHEMA or digest({k:v for k,v in bundle.items() if k!="bundle_sha256"})!=bundle.get("bundle_sha256"):
        raise ValueError("Mosaic bundle identity mismatch")
    rows=[];root=path.parent.resolve()
    for row in bundle["patches"]:
        p=root/row["data_file"]
        if p.is_symlink() or not p.resolve().is_relative_to(root) or file_hash(p)!=row["data_file_sha256"]:
            raise ValueError("Mosaic file identity/path mismatch")
        with np.load(p,allow_pickle=False) as z:
            image=z["image"].copy();mask=z["valid_mask"].copy();events=z["first_event"].copy();streams=z["first_source_stream"].copy()
        if image.dtype!=np.uint8 or image.shape!=(128,128,3) or mask.dtype!=bool or mask.shape!=(128,128):
            raise ValueError("Mosaic shape/dtype invalid")
        if events.shape!=(128,128) or events.dtype.kind!="i" or streams.shape!=(128,128) or streams.dtype!=np.uint8:
            raise ValueError("First-source shape/dtype invalid")
        if (hashlib.sha256(image.tobytes()).hexdigest()!=row["pixel_sha256"] or
            hashlib.sha256(mask.tobytes()).hexdigest()!=row["mask_sha256"] or
            _source_hash(events,streams)!=row["first_source_sha256"] or
            int(mask.sum())!=row["valid_pixels"] or bool(mask.all())!=row["complete"]):
            raise ValueError("Mosaic payload disagrees with sufficient integrity metadata")
        if row.get("planned_pixels")!=128**2 or (events<=0).any():
            raise ValueError("Mosaic planned denominator/source events invalid")
        rows.append({**row,"image":image,"valid_mask":mask,"first_event":events,"first_source_stream":streams})
    if (bundle.get("planned_tiles")!=len(rows) or bundle.get("complete_tiles")!=sum(r["complete"] for r in rows) or
        bundle.get("planned_pixels")!=sum(r["planned_pixels"] for r in rows) or
        bundle.get("valid_pixels")!=sum(r["valid_pixels"] for r in rows)):
        raise ValueError("Mosaic bundle denominator mismatch")
    return bundle,rows


def assemble_suite_mosaics(plan,suite_root,patch_plan,out,selection=None):
    from .suite import replay_suite,suite_read_lock,validate_plan
    validate_plan(plan);validate_patch_plan(patch_plan)
    _validate_suite_binding(plan,patch_plan)
    indexed={}
    for patch in patch_plan["patches"]:indexed.setdefault(patch["task_id"],[]).append(patch)
    entries=[]
    with suite_read_lock(suite_root):
        results={r["run_id"]:r for r in replay_suite(plan,suite_root,selection)}
        for row in plan["expected"]:
            result=results[row["run_id"]];images={}
            if result["replay_verified"]:
                submission=Path(suite_root)/"runs"/row["run_id"]/f'attempt-{result["attempt"]:03d}'/"submission"
                from .runner import verify_journal
                journal=[load_json(line) for line in (submission/"events.jsonl").read_text().splitlines()]
                if verify_journal(journal)!=result["replay_head"]:raise ValueError("Submission changed after independent replay")
                outputs={(s["stream"],s["event"]):s for s in journal if s["kind"]=="output"}
                needed={(s["stream"],s["event"]) for s in result.get("output_slots",[]) if s["valid"]}
                for stream,event in needed:
                    image=load_rgb_png(submission/stream/f"{event:04d}.png")
                    if hashlib.sha256(image.tobytes()).hexdigest()!=outputs[(stream,event)]["pixel_sha256"]:
                        raise ValueError("First-source PNG changed after independent replay")
                    images[(stream,event)]=image
            for patch in indexed.get(row["task_id"],[]):
                # Unverifiable submissions contribute entirely missing planned tiles.
                entry=assemble_patch(patch,plan["tasks"][row["task_id"]],images,verified=True,
                                     seed=row["seed"],model_id=row["model_id"],run_id=row["run_id"])
                entry.update(model_manifest_sha256=row["model_manifest_sha256"],metric_config_sha256=row["metric_config_sha256"],
                             study=row["study"],training_seed=row["training_seed"],source_manifest_sha256=row["source_manifest_sha256"],
                             independently_verified=result["replay_verified"],submission_status=result["submission_status"],
                             analysis_selection=result.get("selection"),attempt=result["attempt"])
                entries.append(entry)
        return save_mosaics(entries,out,patch_plan["patch_plan_sha256"],plan["plan_sha256"])



def assemble_reference_mosaics(plan,patch_plan,out):
    """Ground-truth references use exactly the same frozen first-source mapping."""
    from .suite import validate_plan,_world
    from .geometry import crop
    validate_plan(plan);validate_patch_plan(patch_plan);_validate_suite_binding(plan,patch_plan)
    bindings={};entries=[]
    for row in plan["expected"]:
        if row["world_id"] in bindings and bindings[row["world_id"]]["world_file_sha256"]!=row["world_file_sha256"]:
            raise ValueError("Reference source identity conflict")
        bindings[row["world_id"]]=row
    worlds={};image_cache={}
    for patch in patch_plan["patches"]:
        row=bindings[patch["world_id"]]
        if patch["task_sha256"]!=digest(plan["tasks"][patch["task_id"]]):raise ValueError("Reference task identity mismatch")
        if patch["world_id"] not in worlds:
            world=_world(row["world_path"])
            if file_hash(row["world_path"])!=row["world_file_sha256"] or hashlib.sha256(world.tobytes()).hexdigest()!=row["world_pixel_sha256"]:
                raise ValueError("Reference source pixels differ from frozen world")
            worlds[patch["world_id"]]=world
        task=plan["tasks"][patch["task_id"]];world=worlds[patch["world_id"]]
        if task["task_id"] not in image_cache:
            image_cache[task["task_id"]]={(stream,event):crop(world,q,task["origin"]) for stream,queries in _lineages(task).items() for event,q in enumerate(queries) if event}
        images=image_cache[task["task_id"]]
        entry=assemble_patch(patch,task,images,verified=True,seed=0,model_id="real-reference",run_id=None)
        entry.update(reference_kind="diagnostic-world" if task["role"]=="diagnostic_only" else "frozen-real-world",
                     source_world_file_sha256=row["world_file_sha256"],study=row["study"],source_manifest_sha256=row["source_manifest_sha256"])
        entries.append(entry)
    return save_mosaics(entries,out,patch_plan["patch_plan_sha256"],plan["plan_sha256"])


def _validate_suite_binding(plan,patch_plan):
    for task_id,task in patch_plan["tasks"].items():
        if digest(task)!=digest(plan["tasks"].get(task_id)):raise ValueError("Patch/suite task lineage mismatch")
    bindings={r["world_id"]:r for r in plan["expected"]}
    for world in {t["world_id"] for t in patch_plan["tasks"].values()}:
        row=bindings[world]
        expected={"city":row["city"],"role":row["role"],"world_file_sha256":row["world_file_sha256"],
                  "source_identity":"world-pixels-sha256:"+row["world_pixel_sha256"]}
        if patch_plan["world_metadata"].get(world)!=expected:
            raise ValueError("Patch/suite source-world identity mismatch")

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("plan");p.add_argument("--suite-plan",required=True);p.add_argument("--out",required=True);p.add_argument("--count",type=int,default=16)
    p=sub.add_parser("mosaics");p.add_argument("--suite-plan",required=True);p.add_argument("--suite",required=True)
    p.add_argument("--patch-plan",required=True);p.add_argument("--out",required=True);p.add_argument("--selection",choices=["first_attempt_only","latest_attempt_within_budget"])
    p=sub.add_parser("reference");p.add_argument("--suite-plan",required=True);p.add_argument("--patch-plan",required=True);p.add_argument("--out",required=True)
    args=parser.parse_args(argv)
    try:
        from .suite import _json,validate_plan
        plan=_json(args.suite_plan);validate_plan(plan)
        if args.command=="plan":
            worlds={r["world_id"]:{"city":r["city"],"role":r["role"],"world_file_sha256":r["world_file_sha256"],
                                  "source_identity":"world-pixels-sha256:"+r["world_pixel_sha256"]} for r in plan["expected"]}
            patch_plan=build_patch_plan(list(plan["tasks"].values()),args.count,world_metadata=worlds)
            _save_json(args.out,patch_plan);print(canonical({"patch_plan_sha256":patch_plan["patch_plan_sha256"],"planned_tiles":len(patch_plan["patches"])}).decode());return 0
        if args.command=="reference":
            bundle=assemble_reference_mosaics(plan,_json(args.patch_plan),args.out)
        else:
            bundle=assemble_suite_mosaics(plan,args.suite,_json(args.patch_plan),args.out,args.selection)
        print(canonical({"bundle_sha256":bundle["bundle_sha256"],"planned_tiles":bundle["planned_tiles"],"complete_tiles":bundle["complete_tiles"]}).decode())
        return 0 if bundle["complete_tiles"]==bundle["planned_tiles"] else 1
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print(canonical({"status":"FAILED","reason":str(exc),"error_type":type(exc).__name__}).decode());return 2


if __name__=="__main__":raise SystemExit(main())
