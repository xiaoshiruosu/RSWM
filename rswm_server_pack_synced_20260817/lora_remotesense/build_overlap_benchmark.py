#!/usr/bin/env python3
"""
Build overlap-controlled benchmark via rejection sampling.

Generates 4 benchmark.h5 files with well-separated overlap distributions:
  trivial (50-80%), easy (25-50%), medium (10-25%), hard (<10%)

Trajectory generation is pure math (microseconds each). Only accepted
trajectories trigger the expensive scene loading + frame extraction.

Usage:
  # Synthetic dry-run
  python lora_remotesense/build_overlap_benchmark.py --output-base "$TMPDIR/overlap_test" --no-hf -n 5

  # Full run with AID
  python lora_remotesense/build_overlap_benchmark.py --output-base /scratch/.../overlap_v2 -n 25
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Import v5 builder classes
# ---------------------------------------------------------------------------

V5_SCRIPT = Path(__file__).resolve().parent.parent / "wan" / "remotesensebenchmark" / "build_multi_direction_benchmark_v5.py"


def _load_v5_module():
    if not V5_SCRIPT.exists():
        raise FileNotFoundError(f"Benchmark builder not found: {V5_SCRIPT}")
    spec = importlib.util.spec_from_file_location("rs_benchmark_v5", V5_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# HD dataset injection (reuse from build_dataset_from_benchmark.py)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset_from_benchmark import _inject_hd_config

# ---------------------------------------------------------------------------
# Overlap bins & per-bin generation hints
# ---------------------------------------------------------------------------

OVERLAP_BINS = {
    "trivial": (0.50, 0.80),
    "easy":    (0.25, 0.50),
    "medium":  (0.10, 0.25),
    "hard":    (0.00, 0.10),
}

BIN_HINTS = {
    "trivial": {"min_segment_length": 0.01, "min_segments": 2, "max_segments": 2, "max_distance_frac": 0.10},
    "easy":    {"min_segment_length": 0.04, "min_segments": 2, "max_segments": 3, "max_distance_frac": 0.25},
    "medium":  {"min_segment_length": 0.08, "min_segments": 2, "max_segments": 4, "max_distance_frac": 0.40},
    "hard":    {"min_segment_length": 0.15, "min_segments": 2, "max_segments": 4, "max_distance_frac": 0.40},
}

LEVEL_ORDER = ["trivial", "easy", "medium", "hard"]
# [v2.5 审查修复] 旧代码硬编码含 flood,而 builder 注入的 HD_DATASET_CONFIG
# 键已是 water → v5 的 "if scene_category not in HF_DATASET_CONFIG: continue"
# 会【静默跳过】该场景 → 落到程序合成噪声分支,产出 25% 假地物且零报错
# (v2.4 的禁静默回退守卫只挂在 build_dataset_from_benchmark 上,不覆盖这里)。
# 现从 taxonomy 单一真源读。
# ⚠️ 另注(2026-07-26 用户拍板③):overlap 分层实验【整节删除、不重跑】,
# 本文件仅为历史保留;真要跑必须先补 v2.4 级别的池/重复/泄漏门。
from lora_remotesense.build_dataset_from_benchmark import SCENE_CATEGORIES  # noqa: E402

# ---------------------------------------------------------------------------
# Cheap overlap computation (no image data needed)
# ---------------------------------------------------------------------------


def compute_overlap_cheap(
    start_wp: Tuple[int, int],
    end_wp: Tuple[int, int],
    window_size: Tuple[int, int],
    scene_size: Tuple[int, int],
) -> float:
    """Overlap ratio between start and end windows. Mirrors v5.py:622-642."""
    win_h, win_w = window_size
    H, W = scene_size

    def _bounds(cy, cx):
        return (
            max(0, cy - win_h // 2), min(H, cy + win_h // 2),
            max(0, cx - win_w // 2), min(W, cx + win_w // 2),
        )

    h1s, h1e, w1s, w1e = _bounds(*start_wp)
    h2s, h2e, w2s, w2e = _bounds(*end_wp)

    ih = min(h1e, h2e) - max(h1s, h2s)
    iw = min(w1e, w2e) - max(w1s, w2s)
    if ih <= 0 or iw <= 0:
        return 0.0
    area1 = (h1e - h1s) * (w1e - w1s)
    return (ih * iw) / area1 if area1 > 0 else 0.0


def classify_direction(waypoints) -> str:
    """Cardinal direction from start-to-end displacement."""
    wp = np.asarray(waypoints, dtype=float)
    dy = wp[-1, 0] - wp[0, 0]
    dx = wp[-1, 1] - wp[0, 1]
    theta = np.degrees(np.arctan2(-dy, dx)) % 360
    if theta < 45 or theta >= 315:
        return "E"
    if theta < 135:
        return "N"
    if theta < 225:
        return "W"
    return "S"


# ---------------------------------------------------------------------------
# Tuned path generator (subclass with configurable max_distance)
# ---------------------------------------------------------------------------


def _make_tuned_generator(mod, config, max_distance_frac: float):
    """Create a path generator with capped max waypoint distance."""

    class TunedPathGenerator(mod.RandomPositionPathGenerator):
        def __init__(self, cfg, max_dist_frac):
            super().__init__(cfg)
            self._max_dist_frac = max_dist_frac

        def _generate_next_waypoint(self, current_pos, prev_angle=None):
            H, W = self.config.full_scene_size
            min_cy, max_cy, min_cx, max_cx = self._get_valid_bounds()
            min_distance = max(H, W) * self.config.min_segment_length
            max_distance = max(H, W) * self._max_dist_frac

            if max_distance < min_distance:
                max_distance = min_distance * 1.5

            for _ in range(50):
                if prev_angle is not None:
                    angle_offset = np.random.uniform(30, 330)
                    angle = (prev_angle + angle_offset) % 360
                else:
                    angle = np.random.uniform(0, 360)

                distance = np.random.uniform(min_distance, max_distance)
                angle_rad = np.deg2rad(angle)
                dx = np.cos(angle_rad)
                dy = -np.sin(angle_rad)

                new_cx = int(current_pos[1] + dx * distance)
                new_cy = int(current_pos[0] + dy * distance)

                new_cy = np.clip(new_cy, min_cy, max_cy)
                new_cx = np.clip(new_cx, min_cx, max_cx)

                actual_distance = np.sqrt(
                    (new_cy - current_pos[0]) ** 2 + (new_cx - current_pos[1]) ** 2
                )
                if actual_distance >= min_distance * 0.8:
                    actual_angle = (
                        np.degrees(
                            np.arctan2(
                                -(new_cy - current_pos[0]),
                                new_cx - current_pos[1],
                            )
                        )
                        % 360
                    )
                    return (int(new_cy), int(new_cx)), actual_angle

            new_pos = self._random_position()
            actual_angle = (
                np.degrees(
                    np.arctan2(
                        -(new_pos[0] - current_pos[0]),
                        new_pos[1] - current_pos[1],
                    )
                )
                % 360
            )
            return new_pos, actual_angle

    return TunedPathGenerator(config, max_distance_frac)


# ---------------------------------------------------------------------------
# Single-level builder with rejection sampling
# ---------------------------------------------------------------------------


def build_single_level(
    mod,
    level: str,
    overlap_lo: float,
    overlap_hi: float,
    output_dir: Path,
    config,
    use_hf: bool,
    n_per_scene: int = 25,
    max_attempts: int = 5000,
    seed: int = 42,
):
    hints = BIN_HINTS[level]
    level_config = replace(
        config,
        min_segment_length=hints["min_segment_length"],
        min_segments=hints["min_segments"],
        max_segments=hints["max_segments"],
        output_dir=str(output_dir),
    )

    scene_gen = mod.SceneGenerator(level_config, use_hf=use_hf)
    path_gen = _make_tuned_generator(mod, level_config, hints["max_distance_frac"])
    extractor = mod.MultiDirectionExtractor(level_config)

    window_size = level_config.window_size
    scene_size = level_config.full_scene_size

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(exist_ok=True)

    all_samples = []
    global_idx = 0

    print(f"\n{'='*60}")
    print(f"Level: {level}  overlap [{overlap_lo:.0%}, {overlap_hi:.0%})")
    print(f"  min_seg_len={hints['min_segment_length']}, "
          f"segments={hints['min_segments']}-{hints['max_segments']}, "
          f"max_dist={hints['max_distance_frac']}")
    print(f"{'='*60}")

    total_attempts = 0
    total_accepted = 0

    for scene_cat in SCENE_CATEGORIES:
        np.random.seed(seed + hash(scene_cat) % 10000 + hash(level) % 10000)

        DIRECTIONS = ["E", "N", "W", "S"]
        dir_min = n_per_scene // 4
        dir_max = n_per_scene - dir_min * 3

        accepted_paths = []
        dir_buckets: Dict[str, list] = {d: [] for d in DIRECTIONS}
        scene_id_counter = 0
        attempts = 0

        # Phase 1: reject-sample with round-robin direction balance
        while len(accepted_paths) < n_per_scene and attempts < max_attempts:
            num_segments = np.random.randint(
                hints["min_segments"], hints["max_segments"] + 1
            )
            path_info = path_gen.generate_path(num_segments)
            wp = path_info["waypoints"]
            overlap = compute_overlap_cheap(wp[0], wp[-1], window_size, scene_size)

            if overlap_lo <= overlap < overlap_hi:
                direction = classify_direction(wp)
                unfilled = [d for d in DIRECTIONS if len(dir_buckets[d]) < dir_min]
                if unfilled:
                    if direction in unfilled:
                        dir_buckets[direction].append((path_info, overlap, scene_id_counter))
                        accepted_paths.append((path_info, overlap, scene_id_counter))
                elif len(dir_buckets[direction]) < dir_max:
                    dir_buckets[direction].append((path_info, overlap, scene_id_counter))
                    accepted_paths.append((path_info, overlap, scene_id_counter))

            scene_id_counter += 1
            attempts += 1

        if len(accepted_paths) < n_per_scene:
            print(f"  WARNING: {scene_cat} only got {len(accepted_paths)}/{n_per_scene} "
                  f"after {attempts} attempts")

        total_attempts += attempts
        total_accepted += len(accepted_paths)

        # Phase 2: extract frames for accepted trajectories (expensive)
        dataset_source = scene_gen.get_dataset_source(scene_cat)

        for i, (path_info, overlap, sid) in enumerate(accepted_paths):
            scene_img = scene_gen.generate_scene(scene_cat, sid)
            extraction = extractor.extract_multi_direction_sequence(scene_img, path_info)

            video_path = output_dir / "videos" / f"sample_{global_idx:05d}.mp4"

            sample = {
                "sample_id": global_idx,
                "scene_category": scene_cat,
                "dataset_source": dataset_source,
                "video_path": str(video_path.relative_to(output_dir)),
                "sliding_frames": extraction["sliding_frames"],
                "center_frame": extraction["center_frame"],
                "target_frame": extraction["target_frame"],
                "full_scene": extraction["full_scene"],
                "direction_embedding": extraction["direction_embedding"],
                "trajectory": extraction["trajectory"],
                "segment_embeddings": extraction["segment_embeddings"],
                "segment_angles": extraction["segment_angles"],
                "waypoints": np.array(extraction["waypoints"]),
                "num_segments": extraction["num_segments"],
                "total_frames": extraction["total_frames"],
                "overlap_ratio": extraction["overlap_ratio"],
                "start_bounds": extraction["start_bounds"],
                "end_bounds": extraction["end_bounds"],
                "start_position": extraction["start_position"],
                "end_position": extraction["end_position"],
                "steps_per_segment": level_config.steps_per_segment,
            }
            all_samples.append(sample)
            global_idx += 1

        overlaps = [o for _, o, _ in accepted_paths]
        dir_counts = {d: len(v) for d, v in dir_buckets.items()}
        print(f"  {scene_cat}: {len(accepted_paths)}/{n_per_scene} accepted "
              f"({attempts} attempts, rate={len(accepted_paths)/max(attempts,1):.1%})")
        if overlaps:
            print(f"    overlap: mean={np.mean(overlaps):.3f} "
                  f"median={np.median(overlaps):.3f} "
                  f"range=[{min(overlaps):.3f}, {max(overlaps):.3f}]")
            print(f"    dirs: {dir_counts}")

    # Write HDF5
    h5_path = output_dir / "benchmark.h5"
    _write_h5(h5_path, all_samples, level_config, level, overlap_lo, overlap_hi)

    accept_rate = total_accepted / max(total_attempts, 1)
    print(f"\n  {level}: {len(all_samples)} samples written to {h5_path}")
    print(f"  Overall acceptance rate: {accept_rate:.1%} "
          f"({total_accepted}/{total_attempts})")

    return all_samples


def _write_h5(
    h5_path: Path,
    samples: List[Dict],
    config,
    level: str,
    overlap_lo: float,
    overlap_hi: float,
):
    """Write benchmark.h5 matching v5 schema exactly."""
    with h5py.File(h5_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["version"] = "5.0.0"
        meta.attrs["benchmark_type"] = "multi_direction_v5_overlap_controlled"
        meta.attrs["total_instances"] = len(samples)
        meta.attrs["scene_categories"] = config.scene_categories
        meta.attrs["full_scene_size"] = config.full_scene_size
        meta.attrs["window_size"] = config.window_size
        meta.attrs["embed_dim"] = config.embed_dim
        meta.attrs["min_segments"] = config.min_segments
        meta.attrs["max_segments"] = config.max_segments
        meta.attrs["steps_per_segment"] = config.steps_per_segment
        meta.attrs["position_margin"] = config.position_margin
        meta.attrs["min_segment_length"] = config.min_segment_length
        meta.attrs["video_fps"] = config.video_fps
        meta.attrs["creation_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        meta.attrs["overlap_bin"] = level
        meta.attrs["overlap_range_lo"] = overlap_lo
        meta.attrs["overlap_range_hi"] = overlap_hi

        data = f.create_group("data")
        for idx, sample in enumerate(samples):
            grp = data.create_group(f"sample_{idx:05d}")
            grp.create_dataset("sliding_frames", data=sample["sliding_frames"], compression="gzip")
            grp.create_dataset("center_frame", data=sample["center_frame"], compression="gzip")
            grp.create_dataset("target_frame", data=sample["target_frame"], compression="gzip")
            grp.create_dataset("full_scene", data=sample["full_scene"], compression="gzip")
            grp.create_dataset("direction_embedding", data=sample["direction_embedding"])
            grp.create_dataset("trajectory", data=sample["trajectory"])
            grp.create_dataset("segment_embeddings", data=sample["segment_embeddings"])
            grp.create_dataset("segment_angles", data=sample["segment_angles"])
            grp.create_dataset("waypoints", data=sample["waypoints"])
            grp.attrs["sample_id"] = sample["sample_id"]
            grp.attrs["scene_category"] = sample["scene_category"]
            grp.attrs["dataset_source"] = sample.get("dataset_source", "unknown")
            grp.attrs["video_path"] = sample["video_path"]
            grp.attrs["num_segments"] = sample["num_segments"]
            grp.attrs["total_frames"] = sample["total_frames"]
            grp.attrs["overlap_ratio"] = sample["overlap_ratio"]
            grp.attrs["start_bounds"] = sample["start_bounds"]
            grp.attrs["end_bounds"] = sample["end_bounds"]
            grp.attrs["start_position"] = sample["start_position"]
            grp.attrs["end_position"] = sample["end_position"]
            grp.attrs["steps_per_segment"] = sample["steps_per_segment"]

            if idx % 20 == 0 and idx > 0:
                f.flush()

        index = f.create_group("index")
        for sc in config.scene_categories:
            indices = [i for i, s in enumerate(samples) if s["scene_category"] == sc]
            index.create_dataset(sc, data=np.array(indices))

        seg_index = f.create_group("index_by_segments")
        for n_seg in range(config.min_segments, config.max_segments + 1):
            indices = [i for i, s in enumerate(samples) if s["num_segments"] == n_seg]
            if indices:
                seg_index.create_dataset(f"segments_{n_seg}", data=np.array(indices))


# ---------------------------------------------------------------------------
# Build all levels
# ---------------------------------------------------------------------------


def build_all_levels(
    output_base: str,
    use_hf: bool = True,
    dataset_tier: str = "aid",
    n_per_scene: int = 25,
    seed: int = 42,
    levels: Optional[List[str]] = None,
    steps_per_segment: int = 9,
    window_height: int = 128,
    window_width: int = 222,
    local_aid_dir: str = None,
):
    mod = _load_v5_module()

    if use_hf:
        _inject_hd_config(mod, dataset_tier, local_aid_dir=local_aid_dir)

    base_config = mod.MultiDirectionConfig(
        instances_per_scene=n_per_scene,
        scene_categories=list(SCENE_CATEGORIES),
        steps_per_segment=steps_per_segment,
        window_size=(window_height, window_width),
        random_seed=seed,
    )

    if levels is None:
        levels = LEVEL_ORDER

    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    summary = {}
    for level in levels:
        if level not in OVERLAP_BINS:
            print(f"Unknown level: {level}, skipping")
            continue
        lo, hi = OVERLAP_BINS[level]
        level_dir = output_base / level / "benchmark"
        samples = build_single_level(
            mod, level, lo, hi, level_dir,
            base_config, use_hf, n_per_scene,
            seed=seed,
        )
        overlaps = [s["overlap_ratio"] for s in samples]
        dirs = Counter(classify_direction(s["waypoints"]) for s in samples)
        summary[level] = {
            "n": len(samples),
            "overlap_mean": float(np.mean(overlaps)) if overlaps else 0,
            "overlap_median": float(np.median(overlaps)) if overlaps else 0,
            "overlap_min": float(min(overlaps)) if overlaps else 0,
            "overlap_max": float(max(overlaps)) if overlaps else 0,
            "directions": dict(dirs),
        }

    # Print summary
    print(f"\n{'='*60}")
    print("OVERLAP BENCHMARK SUMMARY")
    print(f"{'='*60}")
    for level in levels:
        if level not in summary:
            continue
        s = summary[level]
        lo, hi = OVERLAP_BINS[level]
        print(f"\n  {level} (target {lo:.0%}-{hi:.0%}): N={s['n']}")
        print(f"    overlap: mean={s['overlap_mean']:.3f} "
              f"median={s['overlap_median']:.3f} "
              f"range=[{s['overlap_min']:.3f}, {s['overlap_max']:.3f}]")
        print(f"    dirs: {s['directions']}")

    summary_path = output_base / "overlap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build overlap-controlled benchmark via rejection sampling",
    )
    parser.add_argument("--output-base", required=True,
                        help="Base output directory (creates trivial/easy/medium/hard subdirs)")
    parser.add_argument("-n", "--instances-per-scene", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-tier", choices=["aid", "resisc45", "mixed"], default="aid")
    parser.add_argument("--no-hf", action="store_true", help="Use synthetic data")
    parser.add_argument("--levels", nargs="+", choices=LEVEL_ORDER, default=None,
                        help="Build only specific levels (default: all)")
    parser.add_argument("--steps", type=int, default=9, help="Steps per segment")
    parser.add_argument("--window-height", type=int, default=128)
    parser.add_argument("--window-width", type=int, default=222)
    parser.add_argument("--local-aid-dir", default=None,
                        help="Local AID image directory (bypasses HF download)")
    args = parser.parse_args()

    local_aid_dir = args.local_aid_dir or os.environ.get("LOCAL_AID_DIR")
    build_all_levels(
        output_base=args.output_base,
        use_hf=not args.no_hf,
        dataset_tier=args.dataset_tier,
        n_per_scene=args.instances_per_scene,
        seed=args.seed,
        levels=args.levels,
        steps_per_segment=args.steps,
        window_height=args.window_height,
        window_width=args.window_width,
        local_aid_dir=local_aid_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
