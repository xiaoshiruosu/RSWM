"""Optional SAME-START exact/offset diagnostic, separate from the frozen core.

Jointly accept six paths from the same waypoints. Do not append these packages
silently to the 1920-package core leaderboard. No RGB is consulted in selection.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from .geometry import VERSION, ORIGINS, rng_for, validate_path, digest
from .tasks import long_paths, interpolate, transform, target_labels, validate_task
from .data import verify_manifest, write_json

STUDY = "same-start-return-pair-v0.1"


def paired_packages(world_id: str, role: str, origin_index: int, max_attempts: int = 1000):
    if type(origin_index) is not int or origin_index not in range(4):
        raise ValueError("origin_index must be 0..3.")
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer.")
    rng = rng_for(STUDY, world_id, role, origin_index)
    proposals = 0
    while proposals < max_attempts:
        exact, le, meta = long_paths(rng, offset=False, max_attempts=max_attempts-proposals)
        proposals += meta["proposal_count"]
        orientation, target = meta["orientation"], meta["target_center"]
        wp = meta["untransformed_waypoints"]
        endpoint = transform(np.array([wp["B"][0]+64, 48]), orientation)
        offset = {}
        for name, q in exact.items():
            # Only the final 16-event segment changes; all prior commands match.
            tail = interpolate([q[80], endpoint], [16])
            offset[name] = np.vstack([q[:81], tail[1:]])
        try:
            stats = {name: validate_path(q) for name, q in offset.items()}
            lo = {name: target_labels(q, target) for name, q in offset.items()}
        except ValueError:
            continue
        mo = copy.deepcopy(meta)
        mo.update(variant="offset", stats=stats)
        mo["untransformed_waypoints"]["end"] = [wp["B"][0]+64, 48]
        pair_id = digest([STUDY, world_id, role, origin_index])[:24]
        result = []
        for variant, paths, labels, m in (("exact",exact,le,meta),("offset",offset,lo,mo)):
            m = copy.deepcopy(m); m["joint_proposal_count"] = proposals
            for family, programs, private in (
                ("F5", {"main": paths["f5"]}, {"main": labels["f5"]}),
                ("F7", {"a": paths["a"], "b": paths["b"]}, {"a": labels["a"], "b": labels["b"]}),
            ):
                task = {"version": VERSION, "study": STUDY, "pair_id": pair_id+"-"+family,
                        "world_id": world_id, "role": role, "origin_index": origin_index,
                        "origin": list(ORIGINS[origin_index]), "family": family, "protocol": "online16",
                        "task_id": f"{world_id}-o{origin_index}-{family}-{STUDY}-{variant}",
                        "programs": {name:q.tolist() for name,q in programs.items()}, "metadata": m}
                validate_task(task, private)
                result.append((task, private))
        return result
    raise RuntimeError("Joint exact/offset proposals exhausted; no fallback is permitted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", default="test_unseen_city")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_manifest(manifest)
    worlds = [r for r in manifest["worlds"] if r["role"] == args.role]
    if not worlds:
        raise ValueError("No worlds for the requested role; check spelling.")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Paired diagnostic output directory must be empty.")
    args.out.mkdir(parents=True, exist_ok=True)
    index = []
    for row in worlds:
        for oi in range(4):
            for task, labels in paired_packages(row["world_id"], row["role"], oi):
                filename = task["task_id"]+".json"
                write_json(args.out/filename, task)
                for stream, arrays in labels.items():
                    np.savez_compressed(args.out/(task["task_id"]+"-"+stream+".npz"), **arrays)
                index.append({"task_id":task["task_id"], "pair_id":task["pair_id"], "path":filename})
    write_json(args.out/"index.json", {"study":STUDY, "core_leaderboard":False,
                                     "source_manifest_sha256":manifest["manifest_sha256"], "tasks":index})


if __name__ == "__main__":
    main()
