"""Half-open, integer-coordinate geometry. No image-dependent alignment."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = "rswm-inria-120-v0.1"
HEIGHT, WIDTH, WORLD, LIMIT = 240, 416, 2048, 704
DIAGONAL = float(np.hypot(WIDTH, HEIGHT))
ORIGINS = ((960, 960), (1088, 960), (960, 1088), (1088, 1088))


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rng_for(*parts) -> np.random.Generator:
    # Explicit algorithm and namespace; never Python's process-randomized hash().
    seed = int.from_bytes(hashlib.sha256(canonical([VERSION, *parts])).digest()[:16], "big")
    return np.random.Generator(np.random.PCG64(seed))


def coordinates(q) -> np.ndarray:
    a = np.asarray(q)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or a.dtype.kind not in "iu":
        raise ValueError("Queries must be an integer [T,2] array, including q0.")
    if not np.array_equal(a[0], [0, 0]) or (np.any(a > LIMIT) or (a.dtype.kind == "i" and np.any(a < -LIMIT))):
        raise ValueError("q0 must be zero and every coordinate must be in [-704,704].")
    return a.astype(np.int64)


def box(q, origin=(0, 0)) -> tuple[int, int, int, int]:
    a = np.asarray(q)
    o = np.asarray(origin)
    if a.shape != (2,) or o.shape != (2,) or a.dtype.kind not in "iu" or o.dtype.kind not in "iu":
        raise ValueError("Centers and origins must be integer pairs.")
    x, y = int(a[0]) + int(o[0]), int(a[1]) + int(o[1])
    return x - WIDTH // 2, y - HEIGHT // 2, x + WIDTH // 2, y + HEIGHT // 2


def crop(world: np.ndarray, q, origin) -> np.ndarray:
    x0, y0, x1, y1 = box(q, origin)
    if min(x0, y0) < 0 or x1 > world.shape[1] or y1 > world.shape[0]:
        raise ValueError("Viewport outside world.")
    return world[y0:y1, x0:x1].copy()


def intersection(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x0 >= x1 or y0 >= y1:
        return None
    return ((slice(y0-a[1], y1-a[1]), slice(x0-a[0], x1-a[0])),
            (slice(y0-b[1], y1-b[1]), slice(x0-b[0], x1-b[0])))


def path_stats(q) -> dict:
    a = coordinates(q)
    delta = np.diff(a, axis=0)
    overlap = np.maximum(0., 1-np.abs(delta[:, 0])/WIDTH) * np.maximum(0., 1-np.abs(delta[:, 1])/HEIGHT)
    return {"new_queries": len(a)-1, "length_px": float(np.linalg.norm(delta, axis=1).sum()),
            "max_radius_px": float(np.linalg.norm(a, axis=1).max()),
            "max_step_px": float(np.linalg.norm(delta, axis=1).max()),
            "min_overlap": float(overlap.min()), "delta_q": delta.tolist(),
            "adjacent_overlap": overlap.tolist()}


def validate_path(q, length_range=None) -> dict:
    s = path_stats(q)
    if s["min_overlap"] < .65 - 1e-12:
        raise ValueError("Adjacent overlap below 0.65 after integer discretization.")
    if length_range and not length_range[0]*DIAGONAL <= s["length_px"] <= length_range[1]*DIAGONAL:
        raise ValueError("Path length outside configured range.")
    return s


def rgb(image, shape=(HEIGHT, WIDTH, 3)) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.shape != shape:
        raise ValueError(f"Expected uint8 RGB with shape {shape}; implicit conversions are forbidden.")
    return np.ascontiguousarray(image)
