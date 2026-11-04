#!/usr/bin/env python3
"""Validate and materialize the preregistered ICLR LoRA experiment matrix.

This planner is CPU-only and never launches training.  It emits one JSON plan whose
entries map directly to ``common.sh::train_lora``.  Expensive execution remains an
explicit, separate cluster action.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

SCHEMA = "rswm.iclr_experiment_matrix/v1"
RANKS = [4, 8, 16, 32, 64]
TARGETS = ["attn_only", "attn_cam", "all"]


def load_and_validate(path: str | Path) -> dict:
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    errors = []
    if doc.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA!r}")
    protocol = doc.get("protocol", {})
    if protocol.get("profile") != "official":
        errors.append("protocol.profile must be official")
    if protocol.get("prompt_schema") != "net_continuous":
        errors.append("official prompt_schema must be net_continuous")
    if protocol.get("path_estimator") != "registration":
        errors.append("official path_estimator must be registration")
    task = doc.get("task", {})
    if task.get("condition_frames") != 1:
        errors.append("condition_frames must be exactly 1")
    if task.get("fidelity_scope") != "predicted_frames_only":
        errors.append("fidelity_scope must be predicted_frames_only")
    ad = doc.get("adaptation", {})
    if ad.get("rank_sweep", {}).get("ranks") != RANKS:
        errors.append(f"rank_sweep.ranks must be {RANKS}")
    if ad.get("rank_sweep", {}).get("targets") != "attn_cam":
        errors.append("rank_sweep must fix targets=attn_cam")
    if ad.get("rank_sweep", {}).get("pose_mode") != "native":
        errors.append("rank_sweep must fix pose_mode=native")
    if ad.get("target_ablation", {}).get("targets") != TARGETS:
        errors.append(f"target_ablation.targets must be {TARGETS}")
    matched = ad.get("matched_pose_target_ablation", {})
    if matched.get("pose_modes") != ["native", "drop"]:
        errors.append("matched pose modes must be [native,drop]")
    if matched.get("targets") != ["attn_only", "attn_cam"]:
        errors.append("matched targets must be [attn_only,attn_cam]")
    ov = doc.get("benchmark_tracks", {}).get("overlap_curve", {})
    if ov.get("anchors_pct") != [80, 50, 30, 10, 5]:
        errors.append("overlap anchors must be [80,50,30,10,5]")
    if "condition frame" not in ov.get("definition", ""):
        errors.append("overlap must be defined against the condition frame")
    if doc.get("benchmark_tracks", {}).get("long_horizon", {}).get(
            "frame_counts") != [17, 25, 33]:
        errors.append("long-horizon frame_counts must be [17,25,33]")
    if errors:
        raise ValueError("invalid ICLR matrix:\n  - " + "\n  - ".join(errors))
    return doc


def materialize(doc: dict) -> list[dict]:
    p = doc["protocol"]
    common = {
        "dataset_tier": p["dataset_tier"],
        "profile": p["profile"],
        "prompt_schema": p["prompt_schema"],
        "path_estimator": p["path_estimator"],
        "pose_mode": p["pose_mode"],
        "split_seed": 42,
        "eval_n": p["headline_eval_n"],
    }
    runs = []
    rs = doc["adaptation"]["rank_sweep"]
    for rank in rs["ranks"]:
        runs.append({
            **common,
            "pose_mode": rs["pose_mode"],
            "id": f"rank_r{rank}_{rs['targets']}_s{rs['seed']}",
            "family": "rank_sweep",
            "adaptation": "lora",
            "rank": rank,
            "alpha": rank,
            "targets": rs["targets"],
            "train_seed": rs["seed"],
        })
    ta = doc["adaptation"]["target_ablation"]
    for target in ta["targets"]:
        # r16/attn_cam is already in the rank sweep; keep one canonical run ID.
        if target == rs["targets"] and ta["rank"] in rs["ranks"]:
            continue
        runs.append({
            **common,
            "pose_mode": ta["pose_mode"],
            "id": f"target_r{ta['rank']}_{target}_s{ta['seed']}",
            "family": "target_ablation",
            "adaptation": "lora",
            "rank": ta["rank"],
            "alpha": ta["alpha"],
            "targets": target,
            "train_seed": ta["seed"],
        })
    # Matched 2x2 design.  Native/attn_cam is already the rank-sweep r16 point and
    # native/attn_only is already the target ablation; materialize only missing cells.
    matched = doc["adaptation"]["matched_pose_target_ablation"]
    existing = {(r.get("pose_mode"), r.get("targets"), r.get("rank")) for r in runs}
    for pose_mode in matched["pose_modes"]:
        for target in matched["targets"]:
            key = (pose_mode, target, matched["rank"])
            if key in existing:
                continue
            runs.append({
                **common,
                "pose_mode": pose_mode,
                "id": f"matched_{pose_mode}_r{matched['rank']}_{target}_s{matched['seed']}",
                "family": "pose_target_matched",
                "adaptation": "lora",
                "rank": matched["rank"],
                "alpha": matched["alpha"],
                "targets": target,
                "train_seed": matched["seed"],
                "fixed_fields": matched["fixed_fields"],
            })
            existing.add(key)
    # Primary configuration gets two extra seeds for mean±std.
    primary = doc["adaptation"]["primary"]
    for seed in p["seeds"]:
        if seed == rs["seed"]:
            continue
        runs.append({
            **common,
            "pose_mode": primary.get("pose_mode", common["pose_mode"]),
            "id": f"primary_r{primary['rank']}_{primary['targets']}_s{seed}",
            "family": "primary_multiseed",
            "adaptation": "lora",
            "rank": primary["rank"],
            "alpha": primary["alpha"],
            "targets": primary["targets"],
            "train_seed": seed,
        })
    runs.append({
        **common,
        "id": "frozen_base_s42",
        "family": "baseline",
        "adaptation": "frozen",
        "train_seed": None,
        "rank": None,
        "alpha": None,
        "targets": None,
    })
    full = doc["adaptation"]["full_finetuning"]
    runs.append({
        **common,
        "id": "full_ft_unavailable",
        "family": "baseline",
        "adaptation": "full",
        "runnable": False,
        "status": full["status"],
        "reason": full["reason"],
        "train_seed": 42,
        "rank": None,
        "alpha": None,
        "targets": None,
    })
    for run in runs:
        run.setdefault("runnable", True)
    ids = [r["id"] for r in runs]
    if len(ids) != len(set(ids)):
        raise AssertionError("materialized run IDs are not unique")
    return runs


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", default=str(here / "iclr_experiment_matrix.json"))
    ap.add_argument("-o", "--output")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)
    doc = load_and_validate(args.matrix)
    runs = materialize(doc)
    source = Path(args.matrix).resolve()
    out = {
        "schema": "rswm.iclr_execution_plan/v1",
        "source_matrix": str(source),
        "source_matrix_sha256": sha256_file(source),
        "n_runs": len(runs),
        "n_runnable": sum(bool(r["runnable"]) for r in runs),
        "runs": runs,
    }
    print(f"[PASS] matrix valid; {out['n_runnable']}/{out['n_runs']} runs runnable")
    for r in runs:
        state = "RUN" if r["runnable"] else "BLOCKED"
        print(f"  [{state:7}] {r['id']}")
    if args.output and not args.check_only:
        op = Path(args.output)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[write] {op}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
