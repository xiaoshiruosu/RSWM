"""Consume the existing prepared camera payload once for the selected Wan sampler.

The public boundary contains no command trace. This module does not interpolate,
difference, or normalize poses. Tensor construction follows the actual selected
legacy cam_utils.py ray convention and inference.py spatial channel packing;
it is not evidence that a pretrained model has consumed those tensors.
"""
from __future__ import annotations
import numpy as np
from .adapters.camera_conditioning import CameraConfig, frame_indices
from .geometry import DIAGONAL, HEIGHT, WIDTH, LIMIT, digest

PAYLOAD_KEYS = {"camera_poses", "intrinsics_fx_fy_cx_cy", "selected_event_indices",
                "camera_manifest", "camera_manifest_sha256"}


def _setting(config, key):
    try:
        return config[key] if isinstance(config, dict) else getattr(config, key)
    except (KeyError, AttributeError) as exc:
        raise ValueError("Selected Wan configuration is missing " + key) from exc


def validate_prepared_camera(payload, native_frames, config, h, w, lat_h, lat_w):
    """Validate fixed-grid prepared conditions without importing Torch or a vendor.

    Returns the camera channel. Drop still validates the frame/event/configuration
    contract and must contain no poses or intrinsics.
    """
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise ValueError("Prepared camera must contain exactly the public payload keys")
    if type(native_frames) is not int or native_frames not in (17, 33):
        raise ValueError("Selected Wan requires exactly 17 or 33 native frames")
    if any(type(x) is not int for x in (h, w, lat_h, lat_w)) or (h, w, lat_h, lat_w) != (480, 832, 60, 104):
        raise ValueError("Prepared camera requires native480x832 and latent60x104")
    vae_stride, patch_size = _setting(config, "vae_stride"), _setting(config, "patch_size")
    if not isinstance(vae_stride, (tuple, list)) or not isinstance(patch_size, (tuple, list)) or any(type(x) is not int for x in (*vae_stride,*patch_size)) or tuple(vae_stride) != (4, 8, 8) or tuple(patch_size) != (1, 2, 2):
        raise ValueError("Selected Wan VAE/patch temporal and spatial strides are incompatible")
    manifest = payload["camera_manifest"]
    if not isinstance(manifest, dict):
        raise ValueError("Prepared camera manifest must be an object")
    try:
        camera = CameraConfig(**{key: manifest[key] for key in CameraConfig.__dataclass_fields__})
    except (KeyError, TypeError) as exc:
        raise ValueError("Prepared camera configuration fields are invalid") from exc
    if camera.temporal_stride != 4:
        raise ValueError("Selected Wan supports prepared event stride4 only")
    if manifest != camera.manifest() or payload["camera_manifest_sha256"] != digest(manifest):
        raise ValueError("Prepared camera manifest identity mismatch")
    indices = np.asarray(payload["selected_event_indices"])
    expected = frame_indices(native_frames, 4)
    if indices.dtype.kind not in "iu" or indices.shape != expected.shape or not np.array_equal(indices, expected):
        raise ValueError("Prepared camera events must be exact0,4,...native endpoint indices")
    poses, intrinsics = payload["camera_poses"], payload["intrinsics_fx_fy_cx_cy"]
    if camera.camera_channel == "drop":
        if poses is not None or intrinsics is not None:
            raise ValueError("Drop must not carry a second numerical camera channel")
        return camera.camera_channel
    p, k = np.asarray(poses), np.asarray(intrinsics)
    if p.dtype != np.float64 or k.dtype != np.float64 or p.shape != (len(expected), 4, 4) or k.shape != (len(expected), 4):
        raise ValueError("Prepared pose/intrinsics shape or declared float64 dtype mismatch")
    if not np.isfinite(p).all() or not np.isfinite(k).all():
        raise ValueError("Prepared camera values must be finite")
    eye = np.eye(4, dtype=np.float64)
    if not np.array_equal(p[0], eye) or not np.array_equal(p[:, :3, :3], np.repeat(eye[None, :3, :3], len(p), axis=0)):
        raise ValueError("Prepared static nadir camera must retain identity rotation and anchor")
    if not np.array_equal(p[:, 3], np.tile([0., 0., 0., 1.], (len(p), 1))) or np.any(p[:, 2, 3] != 0):
        raise ValueError("Prepared camera homogeneous or planar convention mismatch")
    if camera.camera_channel == "identity" and not np.array_equal(p, np.repeat(eye[None], len(p), axis=0)):
        raise ValueError("Identity camera must contain only identity poses")
    if camera.translation_scaling == "fixed_units" and np.any(np.abs(p[:, :2, 3]) > 2 * LIMIT / DIAGONAL + 1e-12):
        raise ValueError("Prepared camera translation exceeds the frozen coordinate bounds")
    if camera.translation_scaling == "legacy_maxnorm" and np.any(np.linalg.norm(p[:, :3, 3], axis=1) > 1 + 1e-12):
        raise ValueError("Prepared legacy-scaled translation exceeds its declared unit bound")
    expected_k = np.tile([2 * DIAGONAL, 2 * DIAGONAL, float(WIDTH), float(HEIGHT)], (len(p), 1))
    if not np.array_equal(k, expected_k):
        raise ValueError("Prepared virtual-plane intrinsics mismatch; no second resize is allowed")
    return camera.camera_channel


def camera_condition(payload, native_frames, config, h, w, lat_h, lat_w, device):
    """Return the original sampler's dit_cond_dict, without legacy preprocessing.

    Each prepared pose is used at its actual selected event. Rays are ordered
    [direction, origin cross direction] on index+0.5 pixel centres. The original
    8x8 spatial packing yields [1,384,latent_frames,60,104] and one tuple chunk.
    Drop returns None and does not import Torch.
    """
    channel = validate_prepared_camera(payload, native_frames, config, h, w, lat_h, lat_w)
    if channel == "drop":
        return None
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Actual Torch is required for selected Wan camera tensors") from exc
    dtype = _setting(config, "param_dtype")
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Selected Wan camera tensor dtype must be float32 or bfloat16")
    poses = torch.as_tensor(np.array(payload["camera_poses"], copy=True), dtype=torch.float32, device=device)
    intrinsics = torch.as_tensor(np.array(payload["intrinsics_fx_fy_cx_cy"], copy=True), dtype=torch.float32, device=device)
    with torch.no_grad():
        yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device),
                                torch.arange(w, dtype=torch.float32, device=device), indexing="ij")
        output = torch.empty((1, 384, len(poses), lat_h, lat_w), dtype=dtype, device=device)
        for index, (pose, k) in enumerate(zip(poses, intrinsics)):
            direction = torch.stack(((xx + .5 - k[2]) / k[0],
                                     (yy + .5 - k[3]) / k[1], torch.ones_like(xx)), dim=-1)
            direction = direction / direction.norm(dim=-1, keepdim=True)
            direction = direction @ pose[:3, :3].transpose(-1, -2)
            origin = pose[:3, 3].expand_as(direction)
            moment = torch.cross(origin, direction, dim=-1)
            rays = torch.cat((direction, moment), dim=-1)
            packed = rays.reshape(lat_h, 8, lat_w, 8, 6).permute(4, 1, 3, 0, 2).reshape(384, lat_h, lat_w)
            output[0, :, index] = packed.to(dtype)
    return {"c2ws_plucker_emb": output.chunk(1, dim=0)}
