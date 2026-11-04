#!/usr/bin/env python3
"""Evaluate a standardized RS-Roll result directory with future-only scoring.

Paired SSIM/PSNR always run on CPU.  LPIPS, FID, the R3D proxy and standard I3D-FVD
are opt-in so an offline CPU audit never downloads weights or touches a GPU.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEW = ROOT / "lora_remotesense" / "new"
BASELINES = ROOT / "server_run" / "baselines"
sys.path.insert(0, str(NEW))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BASELINES))

from metrics import (compute_fid, compute_fvd, compute_psnr, compute_ssim,
                     compute_video_lpips)
from protocol import RESULT_SCHEMA, validate_manifest_document
from submission_contract import MIN_SAMPLES_FOR_DISTRIBUTION

CONDITION_FRAMES = 1


def _pairs(result_dir: Path):
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_manifest_document(manifest, require_files=True)
    archive = result_dir / "frames.npz"
    if not archive.is_file():
        raise FileNotFoundError(f"{archive}: rerun without --no-save-videos")
    keys = manifest.get("resolved_sample_keys") or manifest.get("sample_keys")
    if not keys:
        raise ValueError("manifest has no resolved sample keys")
    out = []
    with np.load(archive, allow_pickle=False) as generated, \
            h5py.File(manifest["h5"], "r") as file:
        data = file["data"] if "data" in file else file
        if set(generated.files) != set(keys):
            raise ValueError(
                f"frames.npz IDs differ from manifest: generated={len(generated.files)} "
                f"manifest={len(keys)}")
        for key in keys:
            gen = np.asarray(generated[key])
            gt = np.asarray(data[key]["video_frames"][:])
            if gen.shape != gt.shape:
                raise ValueError(f"{key}: generated {gen.shape} != GT {gt.shape}")
            if len(gen) <= CONDITION_FRAMES:
                raise ValueError(f"{key}: no future frames")
            if gen.dtype != np.uint8:
                raise ValueError(
                    f"{key}: standardized frames.npz must be uint8, got {gen.dtype}")
            if gt.ndim != 4 or gt.shape[-1] != 3 or not np.isfinite(gt).all():
                raise ValueError(f"{key}: invalid GT rollout shape/dtype")
            gt = np.clip(gt, 0, 255).astype(np.uint8)
            out.append((key, gen, gt))
    return manifest, out


def _mean(values):
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else None


def _i3d(pairs, checkpoint: str, device: str, frames: int) -> tuple[float, dict[str, Any]]:
    from recompute_fvd_i3d import (_load_extractor, frechet_distance,
                                   gaussian_stats)
    extractor, checksum = _load_extractor(checkpoint, device)
    gen_features, gt_features = [], []
    for key, gen, gt in pairs:
        if len(gen) < CONDITION_FRAMES + frames:
            raise ValueError(
                f"{key}: T={len(gen)} < condition {CONDITION_FRAMES} + I3D frames {frames}")
        # One clip per batch supports heterogeneous spatial sizes while retaining the
        # exact StyleGAN-V I3D preprocessing contract inside the extractor.
        gen_features.append(extractor(gen[None, CONDITION_FRAMES:CONDITION_FRAMES + frames]))
        gt_features.append(extractor(gt[None, CONDITION_FRAMES:CONDITION_FRAMES + frames]))
    gen_features = np.concatenate(gen_features, axis=0)
    gt_features = np.concatenate(gt_features, axis=0)
    result = frechet_distance(*gaussian_stats(gen_features),
                              *gaussian_stats(gt_features))
    return result["fvd"], {
        "implementation": "StyleGAN-V i3d_torchscript.pt protocol",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": checksum,
        "feature_dim": int(gen_features.shape[1]),
        "frame_selection": "first predicted frames after condition",
        "num_frames": frames,
        "n_samples": len(pairs),
        "sqrtm_regularized": result["regularized"],
    }


def evaluate(args) -> dict[str, Any]:
    result_dir = Path(args.result_dir)
    manifest, pairs = _pairs(result_dir)
    aggregate_path = result_dir / "aggregate_metrics.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"aggregate_metrics.json schema must be {RESULT_SCHEMA}")
    if int(aggregate.get("N", -1)) != len(pairs):
        raise ValueError(
            f"aggregate N={aggregate.get('N')} differs from {len(pairs)} saved rollouts")
    if args.i3d_frames <= 0:
        raise ValueError("--i3d-frames must be positive")
    requested_distribution = bool(args.fid or args.r3d_proxy or args.i3d_checkpoint)
    if requested_distribution and len(pairs) < MIN_SAMPLES_FOR_DISTRIBUTION:
        raise ValueError(
            f"distribution metrics require at least {MIN_SAMPLES_FOR_DISTRIBUTION} "
            f"complete clips, got {len(pairs)}")
    ssim, psnr = [], []
    for _key, gen, gt in pairs:
        for t in range(CONDITION_FRAMES, len(gen)):
            ssim.append(compute_ssim(gen[t], gt[t]))
            psnr.append(compute_psnr(gen[t], gt[t]))
    metrics: dict[str, Any] = {
        "SSIM": _mean(ssim),
        "PSNR": _mean(psnr),
        "LPIPS": None,
        "FID": None,
        "FVD_R3D_PROXY": None,
        "FVD_I3D": None,
    }
    provenance: dict[str, Any] = {
        "scoring_scope": "future_frames_only",
        "condition_frames_excluded": CONDITION_FRAMES,
        "n_samples": len(pairs),
        "n_future_frames": sum(len(g) - CONDITION_FRAMES for _, g, _ in pairs),
        "requested_optional_metrics": [],
        "minimum_samples_for_distribution_metrics": MIN_SAMPLES_FOR_DISTRIBUTION,
    }
    future_gen = [gen[CONDITION_FRAMES:] for _, gen, _ in pairs]
    future_gt = [gt[CONDITION_FRAMES:] for _, _, gt in pairs]
    if args.lpips:
        provenance["requested_optional_metrics"].append("LPIPS")
        values = [compute_video_lpips(g, r, args.device)
                  for g, r in zip(future_gen, future_gt)]
        metrics["LPIPS"] = _mean(values)
    if args.fid:
        provenance["requested_optional_metrics"].append("FID")
        metrics["FID"] = float(compute_fid(
            [frame for video in future_gen for frame in video],
            [frame for video in future_gt for frame in video], args.device))
        provenance["FID"] = "distribution over future frames only"
    if args.r3d_proxy:
        provenance["requested_optional_metrics"].append("FVD_R3D_PROXY")
        metrics["FVD_R3D_PROXY"] = float(compute_fvd(future_gen, future_gt, args.device))
        provenance["FVD_R3D_PROXY"] = (
            "torchvision R3D-18 proxy; not standard FVD or I3D-FVD")
    if args.i3d_checkpoint:
        provenance["requested_optional_metrics"].append("FVD_I3D")
        value, meta = _i3d(pairs, args.i3d_checkpoint, args.device, args.i3d_frames)
        metrics["FVD_I3D"] = float(value)
        provenance["FVD_I3D"] = meta
    aggregate.update(metrics)
    aggregate["evaluator_provenance"] = provenance
    aggregate["missing_metrics_are_null_not_zero"] = True
    if args.update:
        aggregate_path.write_text(
            json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"metrics": metrics, "provenance": provenance,
            "manifest_checksum": manifest.get("manifest_checksum")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--r3d-proxy", action="store_true")
    parser.add_argument("--i3d-checkpoint")
    parser.add_argument("--i3d-frames", type=int, default=16)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("-o", "--output")
    args = parser.parse_args(argv)
    result = evaluate(args)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print("[PASS] condition frame excluded; optional unrequested metrics remain null")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
