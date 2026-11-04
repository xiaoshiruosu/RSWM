#!/usr/bin/env python3
"""Paper-protocol evaluator for overlap robustness and rollout horizon.

The overlap axis is deliberately ``condition t=0 -> future t``.  Adjacent-frame
overlap is reported separately as a continuity audit; asking every adjacent pair
to have 5% overlap would turn a video rollout into disconnected image jumps.

This tool only computes paired/framewise metrics.  FID and FVD remain sample-level
distribution metrics in the official scorer and ``recompute_fvd_i3d.py``; treating
correlated frames inside a tiny overlap bin as independent samples would inflate N.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEW = ROOT / "lora_remotesense" / "new"
sys.path.insert(0, str(NEW))

from metrics import compute_ssim, compute_psnr  # noqa: E402
from submission_contract import (  # noqa: E402
    CONDITION_FRAMES, assert_id_set, assert_same_shape,
)
from step_overlap import overlap_from_centers, step_overlaps  # noqa: E402

BIN_SPECS = (
    ("dense_80", 0.65, 1.0, 80),
    ("sparse_50", 0.40, 0.65, 50),
    ("sparse_30", 0.20, 0.40, 30),
    ("sparse_10", 0.075, 0.20, 10),
    ("sparse_05", 0.0, 0.075, 5),
)
HORIZONS = (17, 25, 33)


def assign_overlap_bin(value: float) -> str:
    v = float(value)
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"overlap out of range: {v}")
    for name, lo, hi, _anchor in BIN_SPECS:
        if lo <= v < hi or (hi == 1.0 and v == 1.0):
            return name
    raise AssertionError(f"no overlap bin for {v}")


def _mean(values):
    return float(np.mean(values)) if values else None


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "n_frames": len(rows),
        "n_samples": len({r["sample_id"] for r in rows}),
        "SSIM": _mean([r["SSIM"] for r in rows]),
        "PSNR": _mean([r["PSNR"] for r in rows]),
        "condition_overlap_mean": _mean([r["condition_overlap"] for r in rows]),
        "condition_overlap_min": (min(r["condition_overlap"] for r in rows)
                                  if rows else None),
        "condition_overlap_max": (max(r["condition_overlap"] for r in rows)
                                  if rows else None),
    }


def flow_vector_consistency(gen: np.ndarray, gt: np.ndarray,
                            window_size: tuple[int, int]) -> dict:
    """Farneback global-motion agreement, a 2D geometry/control *proxy*.

    This is not depth, pose recovery, or proof of explicit 3D awareness.  It asks
    only whether generated inter-frame content motion agrees with GT motion.
    """
    from trajectory_metrics import _maybe_downsample, _compute_flows, _step_vector
    assert_same_shape(gen, gt, what="flow consistency video")
    g = _maybe_downsample(gen, window_size)
    r = _maybe_downsample(gt, window_size)
    gf, rf = _compute_flows(g), _compute_flows(r)
    if gf is None or rf is None or len(gf) != len(rf):
        return {"available": False, "n_steps": 0}
    angles, rel_mag, rel_epe = [], [], []
    live_gen = live_gt = 0
    eps = 1e-6
    for a, b in zip(gf, rf):
        gv = np.asarray(_step_vector(a)[:2], dtype=np.float64)
        rv = np.asarray(_step_vector(b)[:2], dtype=np.float64)
        gm, rm = float(np.linalg.norm(gv)), float(np.linalg.norm(rv))
        live_gen += int(gm > eps)
        live_gt += int(rm > eps)
        rel_mag.append(abs(gm - rm) / max(rm, eps))
        rel_epe.append(float(np.linalg.norm(gv - rv)) / max(rm, eps))
        if gm > eps and rm > eps:
            cos = float(np.clip(np.dot(gv, rv) / (gm * rm), -1.0, 1.0))
            angles.append(math.degrees(math.acos(cos)))
    n = len(gf)
    return {
        "available": True,
        "n_steps": n,
        "angle_error_deg_mean": _mean(angles),
        "angle_valid_steps": len(angles),
        "relative_magnitude_error_mean": _mean(rel_mag),
        "relative_vector_epe_mean": _mean(rel_epe),
        "generated_liveness_rate": live_gen / n if n else None,
        "gt_liveness_rate": live_gt / n if n else None,
        "claim_guard": "2D optical-flow consistency proxy; not explicit 3D geometry",
    }


def load_split_indices(path: str | None, name: str) -> list[int] | None:
    if not path:
        return None
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    old = f"{name}_indices"
    if old in doc:
        return [int(i) for i in doc[old]]
    try:
        return [int(i) for i in doc["splits"][name]["indices"]]
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path}: missing {old} or splits.{name}.indices") from e


def _scene_size(group, meta) -> tuple[int, int]:
    if "full_scene" in group:
        return tuple(int(v) for v in group["full_scene"].shape[:2])
    if "full_scene_size" in meta:
        return tuple(int(v) for v in meta["full_scene_size"])
    raise ValueError("H5 has neither per-sample full_scene nor metadata.full_scene_size")


def evaluate(args) -> dict:
    eval_dir = Path(args.eval_dir)
    gen_dir = eval_dir / "generated"
    if not gen_dir.is_dir():
        raise FileNotFoundError(f"missing {gen_dir}")
    manifest_path = eval_dir / "run_manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else None)
    if not args.allow_nonheadline:
        if not manifest or manifest.get("headline_eligible") is not True \
                or manifest.get("manifest_override"):
            raise RuntimeError(
                f"{eval_dir}: missing/ineligible/overridden run_manifest; "
                f"use --allow-nonheadline only for diagnostics")

    with h5py.File(args.gt_h5, "r") as f:
        data = f["data"] if "data" in f else f
        keys = sorted(data.keys())
        indices = load_split_indices(args.split_file, args.split_name)
        if indices is None:
            wanted = sorted(p.stem for p in gen_dir.glob("*.npy"))
            unknown = sorted(set(wanted) - set(keys))
            if unknown:
                raise KeyError(f"generated IDs absent from H5: {unknown[:5]}")
        else:
            bad = [i for i in indices if i < 0 or i >= len(keys)]
            if bad:
                raise IndexError(f"split indices out of range: {bad[:5]}")
            wanted = [keys[i] for i in indices]
            if args.max_samples:
                wanted = wanted[:args.max_samples]
        submitted = sorted(p.stem for p in gen_dir.glob("*.npy"))
        assert_id_set(submitted, wanted, what="rollout protocol evaluation")

        meta = dict(f["metadata"].attrs) if "metadata" in f else {}
        if "pose_window_size" not in meta:
            raise ValueError("H5 metadata.pose_window_size is required")
        win = tuple(int(v) for v in meta["pose_window_size"])
        rows: list[dict] = []
        sample_flow: dict[str, dict] = {}
        adjacent_all = []
        for n, key in enumerate(wanted, 1):
            group = data[key]
            if "trajectory" not in group:
                raise ValueError(f"{key}: missing trajectory")
            gt = np.asarray(group["video_frames"][:])
            gen = np.load(gen_dir / f"{key}.npy")
            if gen.dtype != np.uint8:
                if not np.all(np.isfinite(gen)):
                    raise ValueError(f"{key}: generated video contains NaN/Inf")
                gen = np.clip(gen, 0, 255).astype(np.uint8)
            gt = np.clip(gt, 0, 255).astype(np.uint8)
            assert_same_shape(gen, gt, what="rollout protocol video", key=key)
            traj = np.asarray(group["trajectory"][:], dtype=np.float64)
            if len(traj) != len(gt):
                raise ValueError(f"{key}: trajectory T={len(traj)} != video T={len(gt)}")
            scene_hw = _scene_size(group, meta)
            adj = step_overlaps(traj, win, scene_hw)
            adjacent_all.extend(adj)
            for t in range(CONDITION_FRAMES, len(gt)):
                ov = overlap_from_centers(traj[0], traj[t], win, scene_hw)
                rows.append({
                    "sample_id": key,
                    "t": t,
                    "clip_frames": len(gt),
                    "condition_overlap": float(ov),
                    "overlap_bin": assign_overlap_bin(ov),
                    "SSIM": float(compute_ssim(gen[t], gt[t])),
                    "PSNR": float(compute_psnr(gen[t], gt[t])),
                })
            sample_flow[key] = flow_vector_consistency(gen, gt, win)
            if args.verbose and n % 25 == 0:
                print(f"  ... {n}/{len(wanted)}")

    by_overlap = {}
    for name, lo, hi, anchor in BIN_SPECS:
        rr = [r for r in rows if r["overlap_bin"] == name]
        by_overlap[name] = {
            "anchor_pct": anchor,
            "interval": [lo, hi],
            "interval_semantics": "[lo,hi), except 1.0 included in dense_80",
            **summarize_rows(rr),
        }
    per_frame = {}
    for t in sorted({r["t"] for r in rows}):
        per_frame[str(t)] = summarize_rows([r for r in rows if r["t"] == t])
    prefix = {}
    exact = {}
    for h in HORIZONS:
        eligible = {r["sample_id"] for r in rows if r["clip_frames"] >= h}
        prefix[str(h)] = summarize_rows([
            r for r in rows if r["sample_id"] in eligible and r["t"] < h])
        exact[str(h)] = summarize_rows([
            r for r in rows if r["clip_frames"] == h])
    flow_available = [v for v in sample_flow.values() if v.get("available")]
    flow_agg = {
        "n_samples": len(sample_flow),
        "n_available": len(flow_available),
        "angle_error_deg_mean": _mean([
            v["angle_error_deg_mean"] for v in flow_available
            if v.get("angle_error_deg_mean") is not None]),
        "relative_magnitude_error_mean": _mean([
            v["relative_magnitude_error_mean"] for v in flow_available]),
        "relative_vector_epe_mean": _mean([
            v["relative_vector_epe_mean"] for v in flow_available]),
        "generated_liveness_rate": _mean([
            v["generated_liveness_rate"] for v in flow_available]),
        "gt_liveness_rate": _mean([
            v["gt_liveness_rate"] for v in flow_available]),
        "claim_guard": "2D optical-flow consistency proxy; not explicit 3D geometry",
    }
    return {
        "schema": "rswm.rollout_protocol_results/v1",
        "task": "continuous trajectory-conditioned spatial video rollout",
        "not_temporal_forecasting": True,
        "gt_h5": str(Path(args.gt_h5).resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "split_file": str(Path(args.split_file).resolve()) if args.split_file else None,
        "split_name": args.split_name,
        "run_manifest": manifest,
        "headline_eligible": bool(manifest and manifest.get("headline_eligible") is True
                                  and not manifest.get("manifest_override")),
        "condition_frames_excluded": CONDITION_FRAMES,
        "n_samples": len(wanted),
        "n_predicted_frames": len(rows),
        "overlap_definition": "area overlap(condition frame t=0, predicted frame t)",
        "overlap_bins": by_overlap,
        "adjacent_step_overlap_audit": {
            "definition": "area overlap(frame t-1, frame t)",
            "n_steps": len(adjacent_all),
            "min": min(adjacent_all) if adjacent_all else None,
            "median": float(np.median(adjacent_all)) if adjacent_all else None,
            "mean": _mean(adjacent_all),
            "max": max(adjacent_all) if adjacent_all else None,
            "note": "continuity audit only; never substituted for condition overlap bins",
        },
        "horizon": {
            "total_frames_include_one_condition_frame": True,
            "prefix_aggregate": prefix,
            "exact_length_cohorts": exact,
            "per_frame_curve": per_frame,
        },
        "flow_vector_consistency": flow_agg,
        "per_sample_flow": sample_flow,
        "distribution_metrics_note": (
            "FID/FVD are intentionally not recomputed per overlap frame-bin; use "
            "the official scorer and recompute_fvd_i3d.py at sample level."),
    }


def self_test() -> int:
    checks = []
    def ck(name, cond):
        checks.append(bool(cond)); print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    ck("bin_100", assign_overlap_bin(1.0) == "dense_80")
    ck("bin_80", assign_overlap_bin(.80) == "dense_80")
    ck("bin_50", assign_overlap_bin(.50) == "sparse_50")
    ck("bin_30", assign_overlap_bin(.30) == "sparse_30")
    ck("bin_10", assign_overlap_bin(.10) == "sparse_10")
    ck("bin_05", assign_overlap_bin(.05) == "sparse_05")
    ck("bin_0", assign_overlap_bin(0.0) == "sparse_05")
    rows = [
        {"sample_id": "a", "SSIM": .5, "PSNR": 20., "condition_overlap": .8},
        {"sample_id": "a", "SSIM": .7, "PSNR": 22., "condition_overlap": .5},
        {"sample_id": "b", "SSIM": .6, "PSNR": 21., "condition_overlap": .3},
    ]
    s = summarize_rows(rows)
    ck("aggregate_sample_unit", s["n_frames"] == 3 and s["n_samples"] == 2)
    ck("aggregate_known_answer", abs(s["SSIM"] - .6) < 1e-12)
    # Geometry primitive known answer: horizontal half-window shift -> 50% overlap.
    ck("overlap_known_answer",
       abs(overlap_from_centers((100, 100), (100, 150), (100, 100)) - .5) < 1e-12)
    return 0 if all(checks) else 7


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-h5")
    ap.add_argument("--eval-dir")
    ap.add_argument("--split-file")
    ap.add_argument("--split-name", choices=("train", "val", "test"), default="val")
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--allow-nonheadline", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("-o", "--output", default="rollout_protocol_results.json")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.gt_h5 or not args.eval_dir:
        ap.error("--gt-h5 and --eval-dir are required")
    result = evaluate(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[PASS] {result['n_samples']} samples / "
          f"{result['n_predicted_frames']} predicted frames -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
