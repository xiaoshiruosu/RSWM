"""Pure NumPy camera contract for the static 2-D RSWM core.

This is a virtual camera, not recovered Inria extrinsics and not a validated
LingBot backend. Temporal subsampling is explicit and can still lose within-cell
motion. Directions precede moments (o cross d), matching the supplied legacy
wan/utils/cam_utils.py convention; no vendor package is imported or vendored.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator
import numpy as np
from PIL import Image
from ..geometry import DIAGONAL, HEIGHT, WIDTH, LIMIT, digest, rgb

SCHEMA = "rswm-camera-contract-v1"


@dataclass(frozen=True)
class CameraConfig:
    translation_scaling: str = "fixed_units"
    framewise: bool = False
    camera_channel: str = "native"
    temporal_stride: int = 4

    def __post_init__(self):
        if self.translation_scaling not in ("fixed_units", "legacy_maxnorm"):
            raise ValueError("Unknown translation scaling.")
        if type(self.framewise) is not bool:
            raise ValueError("framewise must be a bool.")
        if self.camera_channel not in ("native", "identity", "drop"):
            raise ValueError("camera_channel must be native, identity or drop.")
        if type(self.temporal_stride) is not int or self.temporal_stride not in (1, 2, 4, 8, 16):
            raise ValueError("Only explicit strides dividing 16 are supported.")

    def manifest(self) -> dict:
        return {"schema": SCHEMA, **asdict(self), "coordinate_order": "xy",
                "normalization_scale": DIAGONAL, "query_limit": LIMIT,
                "native_hw": [2 * HEIGHT, 2 * WIDTH], "pose_convention": "c2w",
                "intrinsics_policy": "virtual_plane_z1_v1",
                "ray_channels": ["dx", "dy", "dz", "mx", "my", "mz"],
                "moment": "origin_cross_direction", "pixel_centers": "index+0.5",
                "canonical_rgb": {"hw": [HEIGHT, WIDTH], "dtype": "uint8", "channels": "RGB"},
                "native_rgb": {"hw": [2 * HEIGHT, 2 * WIDTH], "dtype": "uint8", "channels": "RGB"},
                "canonical_to_native": "Pillow_BILINEAR_2x",
                "native_to_canonical": "exact_2x2_uint16_sum_plus2_integer_div4",
                "pose_dtype": "float64", "ray_dtype": "float32",
                "temporal_alignment": "anchor_then_current_call_queries_sample_before_delta",
                "framewise_first_pose": "zero_translation",
                "backend_validation": "NOT_RUN",
                "backend_validated": False}


def _bounded_integer_xy(value, *, rows: bool) -> np.ndarray:
    a = np.asarray(value)
    shape_ok = (a.ndim == 2 and a.shape[1] == 2) if rows else a.shape == (2,)
    if not shape_ok or a.dtype.kind not in "iu":
        raise ValueError("Expected integer xy coordinates; floats/bools are not converted.")
    # Compare BEFORE int64 conversion: uint64/int64 extremes must not wrap.
    if np.any(a > LIMIT) or (a.dtype.kind == "i" and np.any(a < -LIMIT)):
        raise ValueError("Coordinates outside the frozen core bounds.")
    return a.astype(np.int64, copy=True)


def frame_indices(native_frames: int, temporal_stride: int = 4) -> np.ndarray:
    if type(native_frames) is not int or native_frames not in (17, 33):
        raise ValueError("Expected anchor plus 16 or 32 new views.")
    if type(temporal_stride) is not int or temporal_stride not in (1, 2, 4, 8, 16):
        raise ValueError("Unsupported temporal stride.")
    return np.arange(0, native_frames, temporal_stride, dtype=np.int64)


def _immutable_array(value):
    if value is None:
        return None
    a = np.ascontiguousarray(value)
    # Immutable bytes backing also prevents callers from re-enabling writes.
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class CameraBatch:
    # Native conditions may expose a mathematical trace for audits. A drop or
    # identity condition contains no trace; neither sends one to a backend.
    native_queries_xy: np.ndarray | None
    selected_event_indices: np.ndarray
    camera_poses: np.ndarray | None
    intrinsics_fx_fy_cx_cy: np.ndarray | None
    config: CameraConfig

    def __post_init__(self):
        if not isinstance(self.config, CameraConfig):
            raise TypeError("config must be CameraConfig.")
        if self.config.camera_channel != "native" and self.native_queries_xy is not None:
            raise ValueError("Ablated cameras must not contain command trace data.")
        for name in ("native_queries_xy", "selected_event_indices", "camera_poses", "intrinsics_fx_fy_cx_cy"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))

    @property
    def config_sha256(self) -> str:
        return digest(self.config.manifest())


def backend_payload(batch: CameraBatch, *, expected_config_sha256=None) -> dict:
    """Public conditioning whitelist shared by inference and training callers.

    Command traces never enter the payload, including native conditions. The
    expected hash binds a caller's frozen training/inference configuration.
    Backend restoration and VAE stride compatibility need separate real tests.
    """
    if not isinstance(batch, CameraBatch):
        raise TypeError("batch must be CameraBatch.")
    if expected_config_sha256 is not None and expected_config_sha256 != batch.config_sha256:
        raise ValueError("Camera configuration hash mismatch.")
    return {"camera_poses": batch.camera_poses,
            "intrinsics_fx_fy_cx_cy": batch.intrinsics_fx_fy_cx_cy,
            "selected_event_indices": batch.selected_event_indices,
            "camera_manifest": batch.config.manifest(),
            "camera_manifest_sha256": batch.config_sha256}


def build_camera_batch(current_queries_xy, anchor_global_xy=(0, 0),
                       config: CameraConfig | None = None) -> CameraBatch:
    """Build conditions from THIS call only; no private task/world/future argument.

    Each native pose corresponds to an actual query, not uniformly interpolated
    turn keypoints. Sampling occurs before framewise deltas / legacy scaling.
    The stride must be checked against the chosen VAE before real inference.
    """
    cfg = CameraConfig() if config is None else config
    if not isinstance(cfg, CameraConfig):
        raise TypeError("config must be CameraConfig.")
    q = _bounded_integer_xy(current_queries_xy, rows=True)
    anchor = _bounded_integer_xy(anchor_global_xy, rows=False)
    if len(q) not in (16, 32):
        raise ValueError("Only complete online16 or offline32 calls are supported.")
    all_q = np.vstack([anchor, q])
    steps = np.diff(all_q, axis=0)
    ov = np.maximum(0., 1 - np.abs(steps[:, 0]) / WIDTH) * np.maximum(0., 1 - np.abs(steps[:, 1]) / HEIGHT)
    if np.min(ov) < .65 - 1e-12:
        raise ValueError("Adjacent command overlap below 0.65.")
    ids = frame_indices(len(all_q), cfg.temporal_stride)
    if cfg.camera_channel == "drop":
        return CameraBatch(None, ids, None, None, cfg)
    translations = np.zeros((len(ids), 3), dtype=np.float64)
    translations[:, :2] = (all_q[ids] - anchor) / DIAGONAL
    if cfg.framewise:
        translations[1:] = np.diff(translations, axis=0)
    if cfg.translation_scaling == "legacy_maxnorm":
        scale = float(np.linalg.norm(translations, axis=1).max())
        if scale > 0:
            translations /= scale
    if cfg.camera_channel == "identity":
        translations[:] = 0
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(ids), axis=0)
    poses[:, :3, 3] = translations
    # z=1 plane: a canonical dx causes native image content displacement -2*dx.
    k = np.tile([2 * DIAGONAL, 2 * DIAGONAL, float(WIDTH), float(HEIGHT)], (len(ids), 1))
    return CameraBatch(all_q if cfg.camera_channel == "native" else None, ids, poses, k, cfg)


def plucker_at_pixels(poses, intrinsics, pixels_xy) -> np.ndarray:
    """Return [T,P,6] float32 at supplied pixel-centre coordinates.

    Callers can sample sparse rays for tests or stream full grids. This routine
    deliberately accepts no command coordinates and does not normalize moments.
    """
    p = np.asarray(poses, dtype=np.float64)
    k = np.asarray(intrinsics, dtype=np.float64)
    xy = np.asarray(pixels_xy, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (4, 4) or len(p) == 0:
        raise ValueError("Expected nonempty [T,4,4] poses.")
    if k.shape != (len(p), 4) or xy.ndim != 2 or xy.shape[1] != 2 or not len(xy):
        raise ValueError("Invalid intrinsics or pixel grid shape.")
    if not all(np.isfinite(a).all() for a in (p, k, xy)) or np.any(k[:, :2] <= 0):
        raise ValueError("Nonfinite data or nonpositive focal length.")
    if not np.allclose(p[:, 3], [0., 0., 0., 1.], atol=1e-9, rtol=0):
        raise ValueError("Invalid homogeneous row.")
    r = p[:, :3, :3]
    if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-7, rtol=0) or not np.allclose(np.linalg.det(r), 1., atol=1e-7, rtol=0):
        raise ValueError("Expected proper rotations.")
    directions = np.ones((len(p), len(xy), 3), dtype=np.float64)
    directions[:, :, :2] = (xy[None] - k[:, None, 2:]) / k[:, None, :2]
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = directions @ r.transpose(0, 2, 1)
    moments = np.cross(p[:, None, :3, 3], directions)
    return np.concatenate([directions, moments], axis=-1).astype(np.float32)


def iter_native_plucker(batch: CameraBatch) -> Iterator[np.ndarray]:
    """Yield one [480,832,6] frame at a time (not the whole dense video)."""
    if batch.camera_poses is None:
        return
    yy, xx = np.mgrid[:2 * HEIGHT, :2 * WIDTH]
    xy = np.stack([xx.ravel() + .5, yy.ravel() + .5], axis=1)
    for pose, k in zip(batch.camera_poses, batch.intrinsics_fx_fy_cx_cy):
        yield plucker_at_pixels(pose[None], k[None], xy)[0].reshape(2 * HEIGHT, 2 * WIDTH, 6)


def to_native_rgb(initial: np.ndarray) -> np.ndarray:
    a = rgb(initial)
    return np.asarray(Image.fromarray(a).resize((2 * WIDTH, 2 * HEIGHT), Image.Resampling.BILINEAR)).copy()


def from_native_rgb(frame: np.ndarray) -> np.ndarray:
    a = rgb(frame, (2 * HEIGHT, 2 * WIDTH, 3))
    sums = a.reshape(HEIGHT, 2, WIDTH, 2, 3).sum(axis=(1, 3), dtype=np.uint16)
    return ((sums + 2) // 4).astype(np.uint8)


def scale_audit() -> dict:
    """A deterministic condition test, NOT a generated-image experiment."""
    small = np.vstack([np.arange(17)[:, None] * [12, 0],
                       np.array([192, 0]) + np.arange(1, 17)[:, None] * [0, 12]])
    records = []
    for framewise in (False, True):
        for mode in ("fixed_units", "legacy_maxnorm"):
            cfg = CameraConfig(mode, framewise)
            a = build_camera_batch(small[1:], config=cfg)
            b = build_camera_batch(2 * small[1:], config=cfg)
            xy = np.array([[.5, .5], [WIDTH + .5, HEIGHT + .5]])
            ra = plucker_at_pixels(a.camera_poses, a.intrinsics_fx_fy_cx_cy, xy)
            rb = plucker_at_pixels(b.camera_poses, b.intrinsics_fx_fy_cx_cy, xy)
            ta, tb = a.camera_poses[:, :3, 3], b.camera_poses[:, :3, 3]
            records.append({"translation_scaling": mode, "framewise": framewise,
                            "camera_manifest": cfg.manifest(),
                            "camera_manifest_sha256": a.config_sha256,
                            "translation_l2_ratio": float(np.linalg.norm(tb) / np.linalg.norm(ta)),
                            "pose_max_abs_difference": float(np.max(np.abs(a.camera_poses - b.camera_poses))),
                            "ray_max_abs_difference": float(np.max(np.abs(ra - rb))),
                            "selected_event_indices": a.selected_event_indices.tolist()})
    return {"schema": "rswm-condition-audit-v1", "records": records,
            "scope": "virtual pure-translation conditions only",
            "real_images_used": False, "real_weights_used": False,
            "real_backend_validation": "NOT_RUN", "real_vae_stride_validation": "NOT_RUN",
            "temporal_subsampling_may_alias": True, "backend_validated": False}
