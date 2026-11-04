"""Reference online harness. API discipline and hash chains are NOT a sandbox."""
from __future__ import annotations
import hashlib
import json
import time
import copy
import os
import stat
from contextlib import nullcontext
from pathlib import Path
import numpy as np
from .geometry import WIDTH,HEIGHT,LIMIT,DIAGONAL,canonical,crop,rgb
from .score_rgb import Ledger,branch_report
from .tasks import validate_task
from .artifact_io import save_bytes_exclusive, save_png_exclusive

PUBLIC_SPEC = {"viewport_hw":[HEIGHT,WIDTH],"query_bounds_xy":[[-LIMIT,LIMIT],[-LIMIT,LIMIT]],
               "coordinate_unit":"nominal 0.6m canonical pixel","normalization":DIAGONAL,
               "x_direction":"right","y_direction":"down"}


class Journal:
    def __init__(self,root=None):
        self.root=Path(root) if root else None
        self._failed=False
        self._committed_bytes=0
        self._encoded_rows=[]
        if self.root:
            if self.root.is_symlink() or any(p.is_symlink() for p in self.root.parents):
                raise ValueError("Symlinked journal directory")
            if self.root.exists() and any(self.root.iterdir()):
                raise FileExistsError("Run directory must be empty; never overwrite old submissions.")
            self.root.mkdir(parents=True,exist_ok=True)
            # Claim even an empty pre-created worker directory before the first event.
            # A second cooperative writer cannot pass the same-directory race.
            with (self.root/"events.jsonl").open("xb") as stream:
                info=os.fstat(stream.fileno())
                self._file_identity=(info.st_dev,info.st_ino)
        self.rows=[];self.head="0"*64

    def _check_file(self,stream):
        info=os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or
                (info.st_dev,info.st_ino)!=self._file_identity or
                info.st_size!=self._committed_bytes or (self.root/"events.jsonl").is_symlink()):
            raise OSError("Journal file identity or committed length changed")

    def append(self,kind,**payload):
        if self._failed:
            raise OSError("Failed journal writer cannot continue")
        entry={"index":len(self.rows),"previous_hash":self.head,"kind":kind,**payload}
        entry["hash"]=hashlib.sha256(canonical(entry)).hexdigest()
        encoded=(json.dumps(entry,ensure_ascii=False,sort_keys=True,allow_nan=False)+"\n").encode("utf-8")
        if self.root:
            try:
                # r+b cannot recreate a removed log. Length/inode checks reject
                # truncation/replacement; uncertain writes are never repaired.
                with open(self.root/"events.jsonl","r+b") as stream:
                    self._check_file(stream)
                    stream.seek(0,os.SEEK_END)
                    if stream.write(encoded)!=len(encoded):
                        raise OSError("Incomplete journal write")
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                self._failed=True
                raise
        # Close, flush and synchronization must all succeed before committing memory.
        self.head=entry["hash"];self.rows.append(entry)
        self._committed_bytes+=len(encoded)
        self._encoded_rows.append(encoded)
        return entry

    def verify_persisted(self):
        """Check actual retained bytes before summary/head publication.

        This is an I/O boundary, not protection against a later external actor
        or proof of a storage system's power-loss guarantees.
        """
        if self._failed:
            raise OSError("Failed journal writer cannot publish")
        if verify_journal(self.rows)!=self.head:
            raise ValueError("Journal memory head mismatch")
        if self.root:
            try:
                with open(self.root/"events.jsonl","rb") as stream:
                    self._check_file(stream)
                    if stream.read()!=b"".join(self._encoded_rows):
                        raise OSError("Retained journal bytes differ from committed events")
            except BaseException:
                self._failed=True
                raise
        return self.head

    def image(self,stream,event,image):
        if self._failed:
            raise OSError("Failed journal writer cannot publish")
        if self.root and image is not None:
            try:
                folder=self.root/stream;folder.mkdir(exist_ok=True)
                path=folder/f"{event:04d}.png"
                save_png_exclusive(path,image)
            except BaseException:
                self._failed=True
                raise


def verify_journal(rows):
    head="0"*64
    for index,item in enumerate(rows):
        row=dict(item);checksum=row.pop("hash")
        if row["index"] != index or row["previous_hash"] != head or hashlib.sha256(canonical(row)).hexdigest() != checksum:
            raise ValueError("Event chain mismatch.")
        head=checksum
    return head


def validate_response(response,count):
    if not isinstance(response,(list,tuple)) or len(response) != count:
        return [None]*count,["invalid_batch"]*count
    images=[];status=[]
    for frame in response:
        try:
            images.append(rgb(frame).copy());status.append("valid")
        except (ValueError,TypeError):
            images.append(None);status.append("missing" if frame is None else "invalid_image")
    return images,status


def _execute(model,ledger,queries,start,stop,block_size,stream,journal,resources=None):
    for lo in range(start,stop,block_size):
        hi=min(stop,lo+block_size);batch=np.array(queries[lo:hi],dtype=np.int64)
        journal.append("request",stream=stream,start=lo,stop=hi,queries=batch.tolist())
        before=time.perf_counter();error=None
        try:
            with resources.phase("generate", model.model, stream=stream, count=len(batch),
                                 model_call_attempted=model.ready) if resources else nullcontext():
                output=model.generate(batch.copy())
            with resources.phase("output_conversion", stream=stream) if resources else nullcontext():
                images,status=validate_response(output,len(batch))
        except Exception as exc:
            images,status=[None]*len(batch),["model_exception"]*len(batch)
            error=type(exc).__name__  # Do not persist possibly sensitive exception messages.
        elapsed=time.perf_counter()-before
        journal.append("response",stream=stream,start=lo,stop=hi,wall_seconds=elapsed,error=error)
        # Copy all returned arrays before accepting the batch; later model writes cannot alter history.
        for event,q,image,state in zip(range(lo,hi),batch,images,status):
            ledger.submit(event,q,image);journal.image(stream,event,image)
            journal.append("output",stream=stream,event=event,q=q.tolist(),status=state,
                           pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest() if image is not None else None)



class GuardedAdapter:
    """Infrastructure failures keep the complete planned denominator."""
    def __init__(self,model,journal,resources=None):
        self.model,self.journal,self.ready=model,journal,False
        self.resources=resources
    def reset(self,*args):
        try:
            with self.resources.phase("initialize", self.model) if self.resources else nullcontext():
                self.model.reset(*args)
            self.ready=True
        except Exception as exc:
            self.ready=False;self.journal.append("adapter_failure",phase="reset",error=type(exc).__name__)
    def generate(self,queries):
        if not self.ready: return [None]*len(queries)
        return self.model.generate(queries)
    def snapshot(self):
        if not self.ready: return {"ok":False}
        try:
            with self.resources.phase("snapshot", self.model) if self.resources else nullcontext():
                return {"ok":True,"state":copy.deepcopy(self.model.snapshot())}
        except Exception as exc:
            self.journal.append("adapter_failure",phase="snapshot",error=type(exc).__name__)
            return {"ok":False}
    def restore(self,snapshot):
        self.ready=False
        if not snapshot["ok"]: return
        try:
            with self.resources.phase("restore", self.model) if self.resources else nullcontext():
                self.model.restore(copy.deepcopy(snapshot["state"]))
            self.ready=True
        except Exception as exc:
            self.journal.append("adapter_failure",phase="restore",error=type(exc).__name__)


def run_task(task,world,model,seed=0,labels=None,out_dir=None,resources=None):
    rgb(world,(2048,2048,3));labels=validate_task(task,labels);journal=Journal(out_dir)
    model=GuardedAdapter(model,journal,resources)
    initial=crop(world,(0,0),task["origin"])
    journal.append("session",task_id=task["task_id"],seed=int(seed),protocol=task["protocol"],
                   initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                   task_sha256=hashlib.sha256(canonical(task)).hexdigest(),
                   journal_schema="rswm-journal-v1",
                   threat_model="cooperative in-process adapter; not a security boundary")
    if journal.root:
        # Evaluator-owned replay inputs. They are never passed to the model adapter.
        save_png_exclusive(journal.root/"initial.png",initial)
        save_bytes_exclusive(journal.root/"task.json",canonical(task)+b"\n")
    block_size=16 if task["protocol"] == "online16" else 32
    reports={};before=time.perf_counter()
    if task["family"] == "F7":
        qa,qb=task["programs"]["a"],task["programs"]["b"]
        if qa[:33] != qb[:33] or len(qa) != 97 or len(qb) != 97:
            raise ValueError("F7 requires one identical 32-event prefix and two 64-event suffixes.")
        t=time.perf_counter();model.reset(initial.copy(),copy.deepcopy(PUBLIC_SPEC),seed)
        journal.append("initialize",wall_seconds=time.perf_counter()-t)
        prefix=Ledger(initial,task["origin"])
        _execute(model,prefix,qa,1,33,16,"prefix",journal,resources)
        t=time.perf_counter();state=model.snapshot()
        journal.append("snapshot",wall_seconds=time.perf_counter()-t)
        branches={}
        for name,q in (("a",qa),("b",qb)):
            t=time.perf_counter();model.restore(state)
            journal.append("restore",stream=name,wall_seconds=time.perf_counter()-t)
            ledger=prefix.fork(labels.get(name))
            _execute(model,ledger,q,33,97,16,name,journal,resources);branches[name]=ledger
        reports=branch_report(prefix,branches["a"],branches["b"])
    else:
        stream_order = ("small", "large") if task["family"] == "F3" else ("main",)
        for name in stream_order:
            q = task["programs"][name]
            t=time.perf_counter();model.reset(initial.copy(),copy.deepcopy(PUBLIC_SPEC),seed)
            journal.append("initialize",stream=name,wall_seconds=time.perf_counter()-t)
            ledger=Ledger(initial,task["origin"],labels.get(name))
            _execute(model,ledger,q,1,len(q),block_size,name,journal,resources)
            reports[name]=ledger.report()
    result={"task_id":task["task_id"],"family":task["family"],"seed":int(seed),"reports":reports,
            "wall_seconds_including_evaluator_io":time.perf_counter()-before,
            "journal_head":journal.head,"journal_events":len(journal.rows),
            "resource_limitations":{"gpu_synchronized_time":None,"gpu_peak_bytes":None,"process_peak_rss_bytes":None,
                                     "note":"CPU wall time is not a GPU benchmark; no absent value is filled with zero."},
            "real_model_validation":False,"protocol_security_verified":False}
    journal.verify_persisted()
    if journal.root:
        save_bytes_exclusive(journal.root/"summary.json",(json.dumps(result,indent=2,allow_nan=False)+"\n").encode("utf-8"))
    return result,journal
