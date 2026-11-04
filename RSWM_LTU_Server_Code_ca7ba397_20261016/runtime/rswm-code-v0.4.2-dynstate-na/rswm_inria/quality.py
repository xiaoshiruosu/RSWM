"""Geometry-only tile selection and a generic polynomial MMD-U diagnostic.

No semantic accuracy or standard Inception-KID claim is made for arbitrary features.
"""
import hashlib
import numpy as np
from .geometry import WORLD,box,coordinates


def plan_tiles(queries,origin,namespace,count=16,size=128):
    if type(count) is not int or count <= 0 or type(size) is not int or size <= 0 or WORLD%size:
        raise ValueError("Positive tile count and a tile size dividing 2048 are required.")
    q=coordinates(queries);planned=np.zeros((WORLD,WORLD),bool)
    for position in q:
        x0,y0,x1,y1=box(position,origin);planned[y0:y1,x0:x1]=True
    x0,y0,x1,y1=box((0,0),origin);planned[y0:y1,x0:x1]=False
    candidates=[]
    for y in range(0,WORLD,size):
        for x in range(0,WORLD,size):
            if planned[y:y+size,x:x+size].all():
                key=hashlib.sha256(f"{namespace}/{x}/{y}/{size}".encode()).hexdigest()
                candidates.append((key,x,y))
    chosen=sorted(candidates)[:count]
    return {"requested_tiles":count,"eligible_tiles":len(candidates),"tile_size":size,
            "tiles":[[x,y] for _,x,y in chosen],"selection":"geometry only, before inspecting outputs"}


def assemble_tiles(ledger,plan):
    size=plan["tile_size"];output=[];missing=[]
    for x,y in plan["tiles"]:
        if not ledger.valid[y:y+size,x:x+size].all():
            missing.append([x,y]);continue
        output.append(ledger.reference[y:y+size,x:x+size].copy())
    return output,{"planned_tiles":len(plan["tiles"]),"valid_tiles":len(output),"missing_tiles":missing,
                   "coverage":len(output)/len(plan["tiles"]) if plan["tiles"] else None}


def polynomial_mmd_unbiased(x,y):
    """Algebraic unbiased MMD estimator for independent features; may be negative.

    Correlated tiles require geographic/scene-level resampling, not an iid CI.
    The caller must freeze feature weights, preprocessing, counts, and provenance.
    """
    x,y=np.asarray(x,dtype=np.float64),np.asarray(y,dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or not x.shape[1]:
        raise ValueError("Expected feature matrices with equal nonzero dimension.")
    if len(x)<2 or len(y)<2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Need at least two finite feature vectors in each set.")
    d=x.shape[1];kxx=(x@x.T/d+1)**3;kyy=(y@y.T/d+1)**3;kxy=(x@y.T/d+1)**3
    value=(kxx.sum()-np.trace(kxx))/(len(x)*(len(x)-1))+(kyy.sum()-np.trace(kyy))/(len(y)*(len(y)-1))-2*kxy.mean()
    return {"polynomial_mmd_u":float(value),"generated_n":len(x),"reference_n":len(y),
            "feature_dimension":d,"release_validated":False}
