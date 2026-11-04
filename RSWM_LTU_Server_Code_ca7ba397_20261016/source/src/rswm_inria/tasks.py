"""Deterministic F2/F3 and jointly accepted F5/F7 query packages."""
from __future__ import annotations
import numpy as np
from .geometry import VERSION, ORIGINS, rng_for, validate_path, coordinates, WIDTH, HEIGHT

DIRECTIONS = np.array([(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)])


def polyline(rng, paired=False, max_attempts=10000):
    stepsizes = (8,12,16) if paired else (16,24,32)
    limits = (.60,1.25) if paired else (1.,2.5)
    for attempt in range(1, max_attempts+1):
        k = int(rng.choice([2,3,4]))
        steps = np.full(k, 6) + rng.multinomial(32-6*k, np.full(k, 1/k))
        direction = int(rng.integers(8)); increments = []; turns = []; dirs = []
        for segment, n in enumerate(steps):
            if segment:
                turn = int(rng.choice([-3,-2,-1,1,2,3])); direction = (direction+turn) % 8
                turns.append({"event": int(steps[:segment].sum()), "angle_deg": 45*turn})
            delta = DIRECTIONS[direction] * int(rng.choice(stepsizes))
            dirs.append(delta.tolist()); increments.extend([delta] * int(n))
        q = np.vstack([np.zeros((1,2), dtype=int), np.cumsum(increments, axis=0)])
        try:
            stats = validate_path(q, limits)
            if paired:
                validate_path(2*q, (1.2,2.5))
        except ValueError:
            continue
        return q, {"proposal_count": attempt, "steps_per_segment": steps.tolist(),
                   "segment_increments": dirs, "turns": turns, "stats": stats}
    raise RuntimeError("Polyline proposals exhausted; no fallback is permitted.")


def interpolate(points, durations):
    if len(points) != len(durations)+1 or any(n <= 0 for n in durations):
        raise ValueError("Invalid interpolation layout.")
    result = [np.asarray(points[0], dtype=int)]
    for a, b, n in zip(points[:-1], points[1:], durations):
        a, b = np.asarray(a), np.asarray(b)
        result.extend(np.rint(a+(b-a)*t/n).astype(np.int64) for t in range(1,n+1))
    return np.array(result, dtype=np.int64)


def transform(q, orientation):
    # D4 acts on commands, NOT the rectangular viewport or the source imagery.
    if orientation not in range(8):
        raise ValueError("orientation must be an integer in [0,7]")
    a = np.asarray(q).copy()
    if orientation >= 4:
        a[...,0] *= -1
    for _ in range(orientation % 4):
        a = np.stack([-a[...,1], a[...,0]], axis=-1)
    return a


def target_labels(q, center, prefix_end=32, target_size=128, min_gap=32):
    a = coordinates(q)
    yy, xx = np.mgrid[:target_size, :target_size]
    xx = xx + int(center[0])-target_size//2
    yy = yy + int(center[1])-target_size//2
    visible = ((xx[None] >= a[:,0,None,None]-WIDTH//2) &
               (xx[None] < a[:,0,None,None]+WIDTH//2) &
               (yy[None] >= a[:,1,None,None]-HEIGHT//2) &
               (yy[None] < a[:,1,None,None]+HEIGHT//2))
    before, after = visible[:prefix_end+1], visible[prefix_end+1:]
    if visible[0].any() or visible[prefix_end].any() or not before.any(0).all() or not after.any(0).all():
        raise ValueError("Target must be new, seen in prefix, absent at C, and re-enter completely.")
    first = before.argmax(0).astype(np.int16)
    last = (prefix_end-before[::-1].argmax(0)).astype(np.int16)
    reentry = (prefix_end+1+after.argmax(0)).astype(np.int16)
    gap = reentry-last-1
    first_block, return_block = (first-1)//16, (reentry-1)//16
    if gap.min() < min_gap or np.any(first_block == return_block):
        raise ValueError("Target invisibility gap or cross-call constraint violated.")
    # Fully absent complete calls between last prefix visibility and first reentry.
    absent_calls = (reentry-1)//16 - (last-1)//16 - 1
    return {"first_event": first, "last_prefix_visible_event": last,
            "reentry_event": reentry, "invisible_query_gap": gap,
            "fully_absent_calls": absent_calls.astype(np.int16), "x": xx, "y": yy}


def long_paths(rng, offset=False, max_attempts=1000):
    choose = lambda values: int(rng.choice(values))
    for attempt in range(1,max_attempts+1):
        b = (choose([448,480,512,544]), 0)
        c = (b[0]+choose([-64,-32,0,32,64]), choose([416,448,480,512,544]))
        d = (choose([-544,-512,-480,-448]), choose([416,448,480,512,544]))
        e = (choose([-544,-512,-480,-448]), choose([-544,-512,-480,-448]))
        end = (b[0]+64,48) if offset else b
        f = (b[0]+choose([-32,0,32]),e[1]+choose([-32,0,32]))
        p = (choose([-64,0,64]), choose([576,608,640,672]))
        z = (choose([-672,-640,-608,-576]),choose([-64,0,64]))
        r = (choose([-64,0,64]),choose([-672,-640,-608,-576]))
        orientation = int(rng.integers(8))
        paths = {"f5": interpolate([(0,0),b,c,d,e,end],[16,16,32,16,16]),
                 "a": interpolate([(0,0),b,c,d,e,f,end],[16]*6),
                 "b": interpolate([(0,0),b,c,p,z,r,end],[16]*6)}
        paths = {name: transform(q,orientation) for name,q in paths.items()}
        target = transform(np.array(b),orientation).tolist()
        try:
            stats = {name: validate_path(q) for name,q in paths.items()}
            labels = {name: target_labels(q,target) for name,q in paths.items()}
        except ValueError:
            continue
        assert np.array_equal(paths["a"][:33],paths["b"][:33])
        return paths, labels, {"proposal_count": attempt, "orientation": orientation,
                 "variant": "offset" if offset else "exact", "target_center": target,
                 "untransformed_waypoints": {"B": b,"C":c,"D":d,"E":e,"F":f,"P":p,"Q":z,"R":r,"end":end},
                 "stats": stats}
    raise RuntimeError("Long-path proposals exhausted; no fallback is permitted.")


def packages(world_id: str, role: str, origin_index: int):
    if origin_index not in range(4):
        raise ValueError("origin_index must be 0..3")
    base = {"version": VERSION, "world_id": world_id, "role": role,
            "origin_index": origin_index, "origin": list(ORIGINS[origin_index])}
    f2, m2 = polyline(rng_for(world_id,role,origin_index,"F2"))
    f3, m3 = polyline(rng_for(world_id,role,origin_index,"F3"),paired=True)
    paths, labels, ml = long_paths(rng_for(world_id,role,origin_index,"F5-F7"),offset=origin_index in (1,3))
    definitions = [("F2","offline",{"main":f2},m2,{}),
                   ("F3","offline",{"small":f3,"large":2*f3},m3,{}),
                   ("F5","online16",{"main":paths["f5"]},ml,{"main":labels["f5"]}),
                   ("F7","online16",{"a":paths["a"],"b":paths["b"]},ml,{"a":labels["a"],"b":labels["b"]})]
    out = []
    for family, protocol, programs, meta, private in definitions:
        task = dict(base, task_id=f"{world_id}-o{origin_index}-{family}", family=family,
                    protocol=protocol, programs={n:q.tolist() for n,q in programs.items()}, metadata=meta)
        out.append((task,private))
    return out


def training_clips(world_id: str, role: str):
    if role != "train":
        raise ValueError("Training clip export is restricted to train worlds.")
    out = []
    for j in range(32):
        q, meta = polyline(rng_for(world_id,"train-clips","F2",j))
        out.append({"world_id":world_id,"clip_id":f"F2-{j}","origin":ORIGINS[j%4],"q":q.tolist(),"metadata":meta})
    for j in range(16):
        q, meta = polyline(rng_for(world_id,"train-clips","F3",j),paired=True)
        for name,multiplier in (("small",1),("large",2)):
            out.append({"world_id":world_id,"clip_id":f"F3-{j}-{name}","pair_id":f"F3-{j}",
                        "origin":ORIGINS[j%4],"q":(multiplier*q).tolist(),"metadata":meta})
    return out


def validate_task(task, labels=None):
    """Recompute private geometry before scoring; a checksum alone is not validation."""
    family=task["family"]
    expected={"F2":({"main"},33,"offline"),"F3":({"small","large"},33,"offline"),
              "F5":({"main"},97,"online16"),"F7":({"a","b"},97,"online16")}
    if task.get("version")!=VERSION or family not in expected:
        raise ValueError("Unknown task family or protocol version.")
    names,length,protocol=expected[family]
    if set(task["programs"])!=names or task["protocol"]!=protocol:
        raise ValueError("Task program names or execution protocol mismatch.")
    oi=task["origin_index"]
    if type(oi) is not int or oi not in range(4) or tuple(task["origin"])!=ORIGINS[oi]:
        raise ValueError("Invalid frozen origin specification.")
    for q in task["programs"].values():
        if len(q)!=length:raise ValueError("Wrong number of planned queries.")
        validate_path(q)
    if family=="F3" and not np.array_equal(np.array(task["programs"]["small"])*2,task["programs"]["large"]):
        raise ValueError("F3 must be an exact integer 2x pair.")
    if family=="F7" and task["programs"]["a"][:33]!=task["programs"]["b"][:33]:
        raise ValueError("F7 prefixes must match.")
    if family not in ("F5","F7"):return {}
    target=task["metadata"]["target_center"];computed={}
    for name,q in task["programs"].items():
        if q[16]!=target:raise ValueError("Target must be centered at planned B.")
        computed[name]=target_labels(q,target)
        if labels and name in labels:
            if set(labels[name])!=set(computed[name]) or any(not np.array_equal(labels[name][k],v) for k,v in computed[name].items()):
                raise ValueError("Stored private labels disagree with planned geometry.")
    return computed
