"""RGB scoring with immutable FIRST-PLANNED exposure and explicit coverage."""
from __future__ import annotations
import copy
from dataclasses import dataclass, asdict
import numpy as np
from .geometry import WORLD, WIDTH, HEIGHT, box, intersection, rgb


@dataclass
class Accumulator:
    expected_pixels: int = 0
    valid_pixels: int = 0
    absolute_error_rgb: int = 0

    def add(self, expected, predicted=None, reference=None, valid=None):
        self.expected_pixels += int(expected)
        if predicted is None or reference is None:
            return
        a, b = np.asarray(predicted).reshape(-1,3), np.asarray(reference).reshape(-1,3)
        if a.shape != b.shape:
            raise ValueError("Mismatched comparison shapes.")
        keep = np.ones(len(a),dtype=bool) if valid is None else np.asarray(valid,dtype=bool).reshape(-1)
        if keep.size != len(a) or int(keep.sum()) > expected:
            raise ValueError("Invalid comparison mask.")
        self.valid_pixels += int(keep.sum())
        self.absolute_error_rgb += int(np.abs(a[keep].astype(np.int16)-b[keep].astype(np.int16)).sum(dtype=np.int64))

    def report(self):
        return {**asdict(self),"mae": self.absolute_error_rgb/(255*3*self.valid_pixels) if self.valid_pixels else None,
                "coverage":self.valid_pixels/self.expected_pixels if self.expected_pixels else None,
                "state":"not_applicable" if not self.expected_pixels else ("measured" if self.valid_pixels else "missing")}


class Ledger:
    def __init__(self, initial, origin, labels=None):
        self.origin = tuple(origin); self.labels = labels; self.event = 0
        self.known = np.zeros((WORLD,WORLD),bool)
        self.seen = np.zeros_like(self.known); self.valid = np.zeros_like(self.known)
        self.reference = np.zeros((WORLD,WORLD,3),np.uint8)
        self.first_event = np.full((WORLD,WORLD),-1,np.int16)
        x0,y0,x1,y1 = box((0,0),origin); sl = np.s_[y0:y1,x0:x1]
        self.known[sl]=True; self.seen[sl]=True; self.valid[sl]=True
        self.reference[sl]=rgb(initial).copy(); self.first_event[sl]=0
        self.previous = rgb(initial).copy(); self.previous_q = (0,0)
        self.counts = {key:Accumulator() for key in ("overlap","known","history","reentry")}
        self.submitted = 0; self.new_planned_pixels = 0

    def fork(self, labels=None):
        other = copy.deepcopy(self)
        if labels is not None:
            other.labels = labels
        return other

    def submit(self, event, q, image):
        if event != self.event+1:
            raise ValueError("Submissions must follow the immutable planned event order.")
        a = None if image is None else rgb(image).copy()
        x0,y0,x1,y1 = box(q,self.origin)
        if min(x0,y0) < 0 or max(x1,y1) > WORLD:
            raise ValueError("Out-of-bounds submission.")
        sl = np.s_[y0:y1,x0:x1]
        k = self.known[sl]; p = self.seen[sl] & ~k; n = ~self.seen[sl]
        ref, valid = self.reference[sl], self.valid[sl]
        for key,mask in (("known",k),("history",p)):
            if a is None:
                self.counts[key].add(mask.sum())
            else:
                self.counts[key].add(mask.sum(),a[mask],ref[mask],valid[mask])
        pair = intersection(box(self.previous_q),box(q))
        if pair is not None:
            old,new = pair
            pixels = (new[0].stop-new[0].start)*(new[1].stop-new[1].start)
            self.counts["overlap"].add(pixels, a[new] if a is not None else None,
                                       self.previous[old] if self.previous is not None else None)
        if self.labels is not None:
            mask = self.labels["reentry_event"] == event
            ux,uy = self.labels["x"][mask],self.labels["y"][mask]
            ix,iy = ux-int(q[0])+WIDTH//2,uy-int(q[1])+HEIGHT//2
            wx,wy = ux+self.origin[0],uy+self.origin[1]
            self.counts["reentry"].add(mask.sum(), a[iy,ix] if a is not None else None,
                                      self.reference[wy,wx],self.valid[wy,wx])
        # Record first planned appearance even on failure. Later output may not repair it.
        if a is not None:
            ref[n] = a[n]; valid[n] = True; self.submitted += 1
        self.first_event[sl][n] = event
        self.new_planned_pixels += int(n.sum()); self.seen[sl] = True
        self.previous,self.previous_q,self.event = a,tuple(q),event

    def report(self):
        return {"planned_outputs":self.event,"valid_outputs":self.submitted,
                "output_completeness":self.submitted/self.event if self.event else None,
                "new_planned_pixels":self.new_planned_pixels,
                "metrics":{key:value.report() for key,value in self.counts.items()}}


def branch_report(prefix, a, b):
    totals = {}
    for key in a.counts:
        values = {field:getattr(a.counts[key],field)+getattr(b.counts[key],field)-getattr(prefix.counts[key],field)
                  for field in ("expected_pixels","valid_pixels","absolute_error_rgb")}
        totals[key] = Accumulator(**values).report()
    ar,br = a.counts["reentry"].report(),b.counts["reentry"].report()
    both = (ar["expected_pixels"] > 0 and br["expected_pixels"] > 0 and ar["coverage"] == br["coverage"] == 1.)
    planned = a.event+b.event-prefix.event
    valid = a.submitted+b.submitted-prefix.submitted
    return {"planned_outputs":planned,"valid_outputs":valid,"output_completeness":valid/planned,
            "metrics":totals,"branches":{"a":ar,"b":br},"both_branches_complete":both,
            "paired_reentry_mae":(ar["mae"]+br["mae"])/2 if both else None}
