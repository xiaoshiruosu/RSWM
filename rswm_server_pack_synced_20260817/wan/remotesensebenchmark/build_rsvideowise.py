import os
import math
import random
from typing import Dict, Tuple, Optional

import numpy as np

# Optional imports for real remote sensing data handling
try:  # GDAL
    from osgeo import gdal  # type: ignore
except Exception:  # pragma: no cover
    gdal = None

try:  # rasterio
    import rasterio  # type: ignore
except Exception:  # pragma: no cover
    rasterio = None

try:  # torch for direction embeddings
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None

try:  # OpenCV for basic frame handling if available
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:  # HuggingFace datasets
    from datasets import load_dataset as hf_load_dataset  # type: ignore
except Exception:  # pragma: no cover
    hf_load_dataset = None

try:
    from PIL import Image as PILImage  # type: ignore
except Exception:  # pragma: no cover
    PILImage = None

import h5py


# ----------------------
# Global configuration
# ----------------------

IMG_HEIGHT = 96
IMG_WIDTH = 96
NUM_CHANNELS = 3
TOTAL_FRAMES = 16  # full temporal depth loaded per scene
SEQ_LEN = 8        # frames per V_c and V_t (8-16 is allowed; we choose 8)

GRID_ROWS = 3
GRID_COLS = 3

SCENES = ["general", "flood", "urban", "rural"]
INSTANCES_PER_SCENE = 500

MAX_OVERLAP = 0.10  # <10% overlap constraint
MAX_DIR_ERROR_DEG = 15.0

OUTPUT_PATH = "rsvideowise_benchmark.h5"
RANDOM_SEED = 42

# Optional local GeoTIFF root (set to a string path or leave as None)
DATA_ROOT: Optional[str] = None
DATA_SCENE_SUBDIRS = {
    "general": "general",
    "flood": "flood",
    "urban": "urban",
    "rural": "rural",
}

# HuggingFace dataset configuration per scene. Fill in dataset ids and feature
# names as needed. If "dataset" is None or empty, that scene will not use HF.
# HF_SCENE_CONFIG: Dict[str, Dict[str, Optional[str]]] = {
#     "general": {"dataset": None, "split": "train", "feature": None},
#     "flood": {"dataset": None, "split": "train", "feature": None},
#     "urban": {"dataset": None, "split": "train", "feature": None},
#     "rural": {"dataset": None, "split": "train", "feature": None},
# }
HF_SCENE_CONFIG: Dict[str, Dict[str, Optional[str]]] = {
    "general": {
        "dataset": "google/dynamicearth",
        "split": "train",
        "feature": "image"  # RGB images from Sentinel-2
    },
    "flood": {
        "dataset": "tum-vision/fmow_rgb",
        "split": "train",
        "feature": "image"  # RGB images from WorldView-3
    },
    "urban": {
        "dataset": "tum-vision/fmow_rgb",
        "split": "train",
        "feature": "image"  # RGB images from WorldView-3
    },
    "rural": {
        "dataset": "tum-vision/fmow_rgb",
        "split": "train",
        "feature": "image"  # RGB images from WorldView-3
    }
}
HF_DATASETS: Dict[str, object] = {}
HF_FEATURES: Dict[str, str] = {}


# ----------------------
# Utility functions
# ----------------------

BBox = Tuple[int, int, int, int]  # (y0, y1, x0, x1)


def _to_numpy_image(x) -> np.ndarray:
    if PILImage is not None and isinstance(x, PILImage.Image):
        arr = np.asarray(x)
    else:
        arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim not in (3, 4):
        raise RuntimeError("HF feature must be a 2D, 3D or 4D image-like array.")
    return arr


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return abs(diff)


def compute_overlap_fraction(b1: BBox, b2: BBox) -> float:
    y0_1, y1_1, x0_1, x1_1 = b1
    y0_2, y1_2, x0_2, x1_2 = b2

    inter_y0 = max(y0_1, y0_2)
    inter_y1 = min(y1_1, y1_2)
    inter_x0 = max(x0_1, x0_2)
    inter_x1 = min(x1_1, x1_2)

    inter_h = max(0, inter_y1 - inter_y0)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_area = inter_h * inter_w

    patch_area = (y1_1 - y0_1) * (x1_1 - x0_1)
    if patch_area <= 0:
        return 0.0
    return float(inter_area) / float(patch_area)


def compute_box_center(b: BBox) -> Tuple[float, float]:
    y0, y1, x0, x1 = b
    cy = 0.5 * (y0 + y1)
    cx = 0.5 * (x0 + x1)
    return cy, cx


def compute_direction_between_boxes(b_center: BBox, b_target: BBox) -> float:
    cy_c, cx_c = compute_box_center(b_center)
    cy_t, cx_t = compute_box_center(b_target)
    dy = cy_t - cy_c
    dx = cx_t - cx_c
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad) % 360.0
    return angle_deg


# ----------------------
# Data loading / synthesis
# ----------------------


def discover_geotiff_paths(root_dir: str) -> list:
    paths = []
    if not root_dir or not os.path.isdir(root_dir):
        return paths
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in (".tif", ".tiff"):
                paths.append(os.path.join(dirpath, name))
    paths.sort()
    return paths


def _load_sequence_gdal(paths, total_frames: int) -> Optional[np.ndarray]:
    if gdal is None or not paths:
        return None
    frames = []
    for p in paths[:total_frames]:
        ds = gdal.Open(p)
        if ds is None:
            continue
        arr = ds.ReadAsArray()  # (C, H, W) or (H, W)
        if arr.ndim == 2:
            arr = arr[None, ...]
        arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)
        frames.append(arr.astype(np.float32))
    if not frames:
        return None
    frames = np.stack(frames, axis=0)  # (T, H, W, C)
    return frames


def _load_sequence_rasterio(paths, total_frames: int) -> Optional[np.ndarray]:
    if rasterio is None or not paths:
        return None
    frames = []
    for p in paths[:total_frames]:
        with rasterio.open(p) as src:
            arr = src.read()  # (C, H, W)
            if arr.ndim == 2:
                arr = arr[None, ...]
            arr = np.transpose(arr, (1, 2, 0))  # (H, W, C)
            frames.append(arr.astype(np.float32))
    if not frames:
        return None
    frames = np.stack(frames, axis=0)
    return frames


def _generate_synthetic_sequence(scene: str,
                                 total_frames: int,
                                 height: int,
                                 width: int,
                                 channels: int) -> np.ndarray:
    """Generate a synthetic remote-sensing-like sequence for testing."""
    seq = np.zeros((total_frames, height, width, channels), dtype=np.float32)

    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]

    for t in range(total_frames):
        base = np.zeros((height, width, channels), dtype=np.float32)

        if scene == "flood":
            # Rising water-like horizontal gradient with time-varying level
            water_level = (t / max(1, total_frames - 1))
            mask = (y < water_level).astype(np.float32)
            base[..., 1] = mask  # greenish
        elif scene == "urban":
            # Checkerboard / grid pattern
            grid_y = ((np.floor(y * 10) % 2) * 0.7).astype(np.float32)
            grid_x = ((np.floor(x * 10) % 2) * 0.7).astype(np.float32)
            pattern = (grid_y + grid_x) / 2.0
            for c in range(channels):
                base[..., c] = pattern
        elif scene == "rural":
            # Smooth sinusoidal fields
            pattern = 0.5 * (np.sin(2 * math.pi * x + 0.3 * t) +
                             np.cos(2 * math.pi * y + 0.1 * t))
            pattern = (pattern - pattern.min()) / max(1e-6, (pattern.max() - pattern.min()))
            for c in range(channels):
                base[..., c] = pattern
        else:  # general
            # Combination of gradient and low-frequency noise
            gradient = (x + y) / 2.0
            noise = np.random.RandomState(t + 123).randn(height, width, channels).astype(np.float32) * 0.05
            base = np.repeat(gradient[..., None], channels, axis=-1) + noise

        # Simple temporal brightness modulation
        base *= (0.8 + 0.4 * (t / max(1, total_frames - 1)))
        seq[t] = base

    # Normalize to [0,1]
    seq_min = seq.min()
    seq_max = seq.max()
    if seq_max > seq_min:
        seq = (seq - seq_min) / (seq_max - seq_min)
    return seq.astype(np.float32)


def _load_sequence_hf(scene: str, total_frames: int) -> Optional[np.ndarray]:
    if hf_load_dataset is None:
        return None
    ds = HF_DATASETS.get(scene)
    feature_name = HF_FEATURES.get(scene)
    if ds is None or not feature_name:
        return None
    n = len(ds)
    if n <= 0:
        return None
    idx0 = random.randint(0, n - 1)
    example = ds[int(idx0)]
    value = example[feature_name]
    arr0 = _to_numpy_image(value)
    if arr0.ndim == 4:
        t0 = arr0.shape[0]
        if t0 >= total_frames:
            start_t = random.randint(0, t0 - total_frames)
            seq = arr0[start_t:start_t + total_frames]
        else:
            reps = int(math.ceil(float(total_frames) / float(t0)))
            seq = np.tile(arr0, (reps, 1, 1, 1))[:total_frames]
        return seq.astype(np.float32)
    if arr0.ndim == 3:
        frames = [arr0.astype(np.float32)]
        while len(frames) < total_frames:
            idx = random.randint(0, n - 1)
            ex = ds[int(idx)]
            val = ex[feature_name]
            arr = _to_numpy_image(val)
            if arr.ndim != 3:
                continue
            frames.append(arr.astype(np.float32))
        return np.stack(frames, axis=0)
    return None


def load_sequence(geotiff_paths: Optional[list], scene: str) -> np.ndarray:
    """Load temporal sequence from GeoTIFF paths or synthesize data.

    Returns an array of shape (T, H, W, C) with float32 values in [0,1].
    """
    # Try HF datasets first, then GeoTIFF paths
    arr: Optional[np.ndarray] = None
    hf_arr = _load_sequence_hf(scene, TOTAL_FRAMES)
    if hf_arr is not None:
        arr = hf_arr
    elif geotiff_paths:
        if len(geotiff_paths) >= TOTAL_FRAMES:
            start = random.randint(0, len(geotiff_paths) - TOTAL_FRAMES)
            paths_slice = geotiff_paths[start:start + TOTAL_FRAMES]
        else:
            indices = np.random.choice(len(geotiff_paths), size=TOTAL_FRAMES, replace=True)
            paths_slice = [geotiff_paths[i] for i in indices]

        arr = _load_sequence_rasterio(paths_slice, TOTAL_FRAMES)
        if arr is None:
            arr = _load_sequence_gdal(paths_slice, TOTAL_FRAMES)

    if arr is None:
        raise RuntimeError(
            f"No real data available for scene '{scene}'. "
            f"Configure HuggingFace datasets or GeoTIFF paths before building the benchmark."
        )

    if arr is not None:
        # Resize / crop or pad to desired spatial size if needed
        t, h, w, c = arr.shape
        if c > NUM_CHANNELS:
            arr = arr[..., :NUM_CHANNELS]
        elif c < NUM_CHANNELS:
            pad_c = NUM_CHANNELS - c
            arr = np.pad(arr, ((0, 0), (0, 0), (0, 0), (0, pad_c)), mode="constant")

        if h != IMG_HEIGHT or w != IMG_WIDTH:
            # Use OpenCV if available for resizing, otherwise simple central crop/pad
            if cv2 is not None:
                resized = []
                for i in range(t):
                    frame = arr[i]
                    frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    resized.append(frame_resized)
                arr = np.stack(resized, axis=0)
            else:
                # central crop/pad
                arr_resized = np.zeros((t, IMG_HEIGHT, IMG_WIDTH, NUM_CHANNELS), dtype=np.float32)
                y0 = max(0, (h - IMG_HEIGHT) // 2)
                x0 = max(0, (w - IMG_WIDTH) // 2)
                y1 = min(h, y0 + IMG_HEIGHT)
                x1 = min(w, x0 + IMG_WIDTH)
                yy0 = max(0, (IMG_HEIGHT - (y1 - y0)) // 2)
                xx0 = max(0, (IMG_WIDTH - (x1 - x0)) // 2)
                arr_resized[:, yy0:yy0 + (y1 - y0), xx0:xx0 + (x1 - x0), :] = arr[:, y0:y1, x0:x1, :]
                arr = arr_resized

        if t < TOTAL_FRAMES:
            # Tile temporally if not enough frames
            reps = int(math.ceil(TOTAL_FRAMES / t))
            arr = np.tile(arr, (reps, 1, 1, 1))[:TOTAL_FRAMES]
        elif t > TOTAL_FRAMES:
            arr = arr[:TOTAL_FRAMES]

        # Normalize to [0,1]
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        arr = arr.astype(np.float32)

    return arr


# ----------------------
# Grid partitioning
# ----------------------


def partition_grid(video: np.ndarray,
                   num_rows: int = GRID_ROWS,
                   num_cols: int = GRID_COLS) -> Tuple[Dict[Tuple[int, int], np.ndarray],
                                                       Dict[Tuple[int, int], BBox]]:
    """Partition (T, H, W, C) video into a num_rows x num_cols spatial grid.

    Returns:
      patches: dict[(r,c)] -> view onto video[:, y0:y1, x0:x1, :]
      bboxes: dict[(r,c)] -> (y0, y1, x0, x1)
    """
    assert video.ndim == 4, "video must be (T, H, W, C)"
    t, h, w, c = video.shape
    patch_h = h // num_rows
    patch_w = w // num_cols

    patches: Dict[Tuple[int, int], np.ndarray] = {}
    bboxes: Dict[Tuple[int, int], BBox] = {}

    for r in range(num_rows):
        for col in range(num_cols):
            y0 = r * patch_h
            y1 = (r + 1) * patch_h if r < num_rows - 1 else h
            x0 = col * patch_w
            x1 = (col + 1) * patch_w if col < num_cols - 1 else w
            patches[(r, col)] = video[:, y0:y1, x0:x1, :]
            bboxes[(r, col)] = (y0, y1, x0, x1)

    return patches, bboxes


# ----------------------
# Direction embedding
# ----------------------


def generate_direction_embed(angle_deg: float) -> np.ndarray:
    """Generate continuous sin/cos embedding for given angle in degrees.

    Returns a 2D embedding [sin(theta), cos(theta)] as numpy array, using torch
    tensors internally if available.
    """
    angle_rad = math.radians(angle_deg)
    sin_val = math.sin(angle_rad)
    cos_val = math.cos(angle_rad)

    if torch is not None:
        tensor = torch.tensor([sin_val, cos_val], dtype=torch.float32)
        return tensor.cpu().numpy().astype(np.float32)
    else:
        return np.asarray([sin_val, cos_val], dtype=np.float32)


# ----------------------
# Target extraction and validation
# ----------------------


def _sample_target_bbox(center_bbox: BBox,
                         angle_deg: float,
                         img_height: int,
                         img_width: int,
                         max_overlap: float,
                         max_dir_error_deg: float,
                         max_trials: int = 64) -> Tuple[BBox, float, float]:
    """Sample a target bounding box along direction with overlap and direction checks."""
    cy_c, cx_c = compute_box_center(center_bbox)
    y0_c, y1_c, x0_c, x1_c = center_bbox
    patch_h = y1_c - y0_c
    patch_w = x1_c - x0_c

    angle_rad = math.radians(angle_deg)

    # Distance range: at least roughly one patch away, at most near image border
    min_radius = 0.8 * max(patch_h, patch_w)
    max_radius = 0.5 * min(img_height, img_width)
    if max_radius <= min_radius:
        max_radius = min_radius + 1.0

    best_bbox = center_bbox
    best_overlap = 1.0
    best_dir_err = 180.0

    for _ in range(max_trials):
        r = random.uniform(min_radius, max_radius)
        cx_new = cx_c + r * math.cos(angle_rad)
        cy_new = cy_c + r * math.sin(angle_rad)

        x0 = int(round(cx_new - patch_w / 2.0))
        x1 = x0 + patch_w
        y0 = int(round(cy_new - patch_h / 2.0))
        y1 = y0 + patch_h

        if x0 < 0 or y0 < 0 or x1 > img_width or y1 > img_height:
            continue

        candidate = (y0, y1, x0, x1)
        overlap = compute_overlap_fraction(center_bbox, candidate)
        est_angle = compute_direction_between_boxes(center_bbox, candidate)
        dir_err = angular_difference_deg(angle_deg, est_angle)

        if overlap <= best_overlap:
            best_overlap = overlap
            best_bbox = candidate
            best_dir_err = dir_err

        if overlap <= max_overlap and dir_err <= max_dir_error_deg:
            return candidate, overlap, dir_err

    # Fallback: return best candidate even if constraints are slightly violated
    return best_bbox, best_overlap, best_dir_err


def extract_target(video: np.ndarray,
                   center_bbox: BBox,
                   angle_deg: float,
                   t_start: int,
                   seq_len: int,
                   max_overlap: float = MAX_OVERLAP,
                   max_dir_error_deg: float = MAX_DIR_ERROR_DEG) -> Tuple[np.ndarray, BBox, float, float]:
    """Extract target sequence V_t along direction with minimal overlap.

    Returns (V_t, target_bbox, overlap_fraction, direction_error_deg).
    """
    t_total, img_h, img_w, _ = video.shape
    assert 0 <= t_start < t_total and t_start + seq_len <= t_total

    target_bbox, overlap_frac, dir_err = _sample_target_bbox(
        center_bbox=center_bbox,
        angle_deg=angle_deg,
        img_height=img_h,
        img_width=img_w,
        max_overlap=max_overlap,
        max_dir_error_deg=max_dir_error_deg,
    )

    y0, y1, x0, x1 = target_bbox
    V_t = video[t_start:t_start + seq_len, y0:y1, x0:x1, :].astype(np.float32)
    return V_t, target_bbox, overlap_frac, dir_err


def validate_overlap(center_bbox: BBox,
                     target_bbox: BBox,
                     angle_deg: float,
                     max_overlap: float = MAX_OVERLAP,
                     max_dir_error_deg: float = MAX_DIR_ERROR_DEG) -> Tuple[bool, float, float]:
    """Validate that overlap and direction are within constraints."""
    overlap_frac = compute_overlap_fraction(center_bbox, target_bbox)
    est_angle = compute_direction_between_boxes(center_bbox, target_bbox)
    dir_err = angular_difference_deg(angle_deg, est_angle)

    ok = overlap_frac <= max_overlap and dir_err <= max_dir_error_deg
    return ok, overlap_frac, dir_err


# ----------------------
# Main benchmark builder
# ----------------------


def build_benchmark(geotiff_paths_per_scene: Optional[Dict[str, list]] = None,
                    output_path: str = OUTPUT_PATH) -> Dict[str, float]:
    if geotiff_paths_per_scene is None:
        geotiff_paths_per_scene = {scene: [] for scene in SCENES}

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if os.path.exists(output_path):
        os.remove(output_path)

    overlap_values = []
    dir_error_values = []

    with h5py.File(output_path, "w") as f:
        f.attrs["description"] = "RSVideoWise benchmark for remote sensing video extrapolation"
        f.attrs["img_height"] = IMG_HEIGHT
        f.attrs["img_width"] = IMG_WIDTH
        f.attrs["num_channels"] = NUM_CHANNELS
        f.attrs["total_frames"] = TOTAL_FRAMES
        f.attrs["seq_len"] = SEQ_LEN
        f.attrs["instances_per_scene"] = INSTANCES_PER_SCENE
        f.attrs["scenes"] = np.string_(";".join(SCENES))

        instance_counter = 0

        for scene in SCENES:
            scene_group = f.create_group(scene)
            paths_for_scene = geotiff_paths_per_scene.get(scene, [])

            for idx in range(INSTANCES_PER_SCENE):
                video = load_sequence(paths_for_scene, scene)
                patches, bboxes = partition_grid(video, GRID_ROWS, GRID_COLS)

                center_patch = patches[(1, 1)]
                center_bbox = bboxes[(1, 1)]

                # Temporal indices for V_c and V_t
                t_start_c = 0
                t_start_t = SEQ_LEN
                assert TOTAL_FRAMES >= 2 * SEQ_LEN, "TOTAL_FRAMES must be at least 2*SEQ_LEN"

                angle_deg = random.uniform(0.0, 360.0)
                direction_embed = generate_direction_embed(angle_deg)

                y0_c, y1_c, x0_c, x1_c = center_bbox
                V_c = video[t_start_c:t_start_c + SEQ_LEN, y0_c:y1_c, x0_c:x1_c, :].astype(np.float32)

                V_t, target_bbox, overlap_frac, dir_err = extract_target(
                    video=video,
                    center_bbox=center_bbox,
                    angle_deg=angle_deg,
                    t_start=t_start_t,
                    seq_len=SEQ_LEN,
                    max_overlap=MAX_OVERLAP,
                    max_dir_error_deg=MAX_DIR_ERROR_DEG,
                )

                ok, overlap_check, dir_err_check = validate_overlap(
                    center_bbox=center_bbox,
                    target_bbox=target_bbox,
                    angle_deg=angle_deg,
                    max_overlap=MAX_OVERLAP,
                    max_dir_error_deg=MAX_DIR_ERROR_DEG,
                )

                overlap_values.append(overlap_check)
                dir_error_values.append(dir_err_check)

                inst_group = scene_group.create_group(f"instance_{idx:05d}")
                inst_group.create_dataset("V_c", data=V_c, compression="gzip")
                inst_group.create_dataset("V_t", data=V_t, compression="gzip")
                inst_group.create_dataset("direction_angle_deg", data=np.array(angle_deg, dtype=np.float32))
                inst_group.create_dataset("direction_embed", data=direction_embed.astype(np.float32))
                inst_group.create_dataset("center_bbox", data=np.array(center_bbox, dtype=np.int32))
                inst_group.create_dataset("target_bbox", data=np.array(target_bbox, dtype=np.int32))
                inst_group.create_dataset("overlap_fraction", data=np.array(overlap_frac, dtype=np.float32))
                inst_group.create_dataset("direction_error_deg", data=np.array(dir_err, dtype=np.float32))
                inst_group.attrs["scene_type"] = scene
                inst_group.attrs["valid"] = bool(ok)

                instance_counter += 1

    overlap_values_np = np.asarray(overlap_values, dtype=np.float32)
    dir_error_values_np = np.asarray(dir_error_values, dtype=np.float32)

    total_instances = instance_counter
    stats = {
        "total_instances": float(total_instances),
        "avg_overlap": float(overlap_values_np.mean()) if overlap_values_np.size > 0 else float("nan"),
        "max_overlap": float(overlap_values_np.max()) if overlap_values_np.size > 0 else float("nan"),
        "avg_direction_error_deg": float(dir_error_values_np.mean()) if dir_error_values_np.size > 0 else float("nan"),
        "max_direction_error_deg": float(dir_error_values_np.max()) if dir_error_values_np.size > 0 else float("nan"),
    }

    return stats


def main() -> None:
    global HF_DATASETS, HF_FEATURES
    geotiff_paths_per_scene: Dict[str, list] = {}

    for scene in SCENES:
        subdir_name = DATA_SCENE_SUBDIRS.get(scene, scene)
        scene_root = os.path.join(DATA_ROOT, subdir_name) if DATA_ROOT else ""
        paths = discover_geotiff_paths(scene_root) if scene_root else []
        geotiff_paths_per_scene[scene] = paths

        cfg = HF_SCENE_CONFIG.get(scene, {})
        dataset_id = (cfg.get("dataset") or "").strip() if isinstance(cfg.get("dataset"), str) else None
        split = (cfg.get("split") or "train").strip() if isinstance(cfg.get("split"), str) else "train"
        feature = (cfg.get("feature") or "").strip() if isinstance(cfg.get("feature"), str) else None

        if dataset_id:
            if hf_load_dataset is None:
                raise RuntimeError(
                    "HuggingFace 'datasets' library is not available. Install 'datasets' to use HF sources."
                )
            if not feature:
                raise RuntimeError(
                    f"HF_SCENE_CONFIG for scene '{scene}' must define a non-empty 'feature' name "
                    f"when using a HuggingFace dataset."
                )
            ds = hf_load_dataset(dataset_id, split=split)
            HF_DATASETS[scene] = ds
            HF_FEATURES[scene] = feature

    stats = build_benchmark(geotiff_paths_per_scene=geotiff_paths_per_scene)
    print("Benchmark built successfully")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Total instances: {int(stats['total_instances'])}")
    print(
        "Average overlap fraction: "
        f"{stats['avg_overlap']:.4f} (max: {stats['max_overlap']:.4f})"
    )
    print(
        "Average direction error (deg): "
        f"{stats['avg_direction_error_deg']:.2f} (max: {stats['max_direction_error_deg']:.2f})"
    )


if __name__ == "__main__":
    main()
