"""Image-only integer translation diagnostics with explicit abstention.

The estimator never receives query coordinates. It searches linear (not circular)
normalized cross-correlation over a fixed overlap domain. Thresholds are provisional
until calibrated and independently validated; scores do not certify scene realism.
"""
from dataclasses import dataclass, asdict
from collections import Counter
import numpy as np
from .geometry import rgb, DIAGONAL

def validate_rgb(a):
    return rgb(a, a.shape)


@dataclass(frozen=True)
class MotionConfig:
    min_overlap: float = 0.65
    min_std: float = 2 / 255
    min_correlation: float = 0.65
    min_peak_gap: float = 0.04
    peak_exclusion_radius: int = 2

    def __post_init__(self):
        if not (0 < self.min_overlap <= 1 and 0 < self.min_std < 1):
            raise ValueError("Invalid overlap or texture threshold")
        if not (-1 <= self.min_correlation <= 1 and 0 <= self.min_peak_gap <= 2):
            raise ValueError("Invalid correlation thresholds")
        if type(self.peak_exclusion_radius) is not int or self.peak_exclusion_radius < 0:
            raise ValueError("Invalid peak exclusion radius")


def _integral(a):
    return np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


def _rect(s, x0, y0, x1, y1):
    return s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]


def estimate_translation(previous, current, config=None):
    """Return viewport dx,dy: rightward view motion is leftward content motion."""
    config = config or MotionConfig()
    base = {"delta": None, "correlation": None, "second_peak": None,
            "peak_gap": None, "instrument": "linear_zncc_v1",
            "calibration": "provisional_not_independently_validated"}
    if previous is None or current is None:
        return {**base, "status": "submission_failure"}
    validate_rgb(previous)
    validate_rgb(current)
    if previous.shape != current.shape:
        raise ValueError("Motion inputs must have the same scoring grid")
    from scipy.signal import fftconvolve
    # A fixed RGB-to-luminance projection; no per-pair tuning using the command.
    a = previous.astype(np.float64).mean(2) / 255
    b = current.astype(np.float64).mean(2) / 255
    h, w = a.shape
    if min(h, w) < 4:
        return {**base, "status": "insufficient_evidence"}
    yy, xx = np.meshgrid(np.arange(1-h, h), np.arange(1-w, w), indexing="ij")
    ax0, ax1 = np.maximum(0, xx), np.minimum(w, w + xx)
    ay0, ay1 = np.maximum(0, yy), np.minimum(h, h + yy)
    bx0, bx1 = np.maximum(0, -xx), np.minimum(w, w - xx)
    by0, by1 = np.maximum(0, -yy), np.minimum(h, h - yy)
    n = (ax1 - ax0) * (ay1 - ay0)
    sa = _rect(_integral(a), ax0, ay0, ax1, ay1)
    sb = _rect(_integral(b), bx0, by0, bx1, by1)
    va = np.maximum(0, _rect(_integral(a*a), ax0, ay0, ax1, ay1) - sa*sa/n)
    vb = np.maximum(0, _rect(_integral(b*b), bx0, by0, bx1, by1) - sb*sb/n)
    covariance = fftconvolve(a, b[::-1, ::-1], mode="full") - sa*sb/n
    valid = (n >= config.min_overlap*w*h) & (va/n >= config.min_std**2) & (vb/n >= config.min_std**2)
    corr = np.full(n.shape, -np.inf)
    corr[valid] = np.clip(covariance[valid] / np.sqrt(va[valid]*vb[valid]), -1, 1)
    if not valid.any():
        return {**base, "status": "insufficient_evidence"}
    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    peak = float(corr[iy, ix])
    dx, dy = int(xx[iy, ix]), int(yy[iy, ix])
    competitors = valid & ((np.abs(xx-dx) > config.peak_exclusion_radius) |
                           (np.abs(yy-dy) > config.peak_exclusion_radius))
    second = float(corr[competitors].max()) if competitors.any() else None
    gap = peak-second if second is not None else None
    evidence = {**base, "correlation": peak, "second_peak": second, "peak_gap": gap}
    if peak < config.min_correlation:
        return {**evidence, "status": "insufficient_evidence"}
    if second is not None and gap < config.min_peak_gap:
        return {**evidence, "status": "ambiguous"}
    return {**evidence, "status": "measured", "delta": [dx, dy]}


def diagnose_sequence(images, queries, config=None):
    """Commands are used only AFTER estimation to compute diagnostic errors."""
    config = config or MotionConfig()
    if len(images) != len(queries) or not images:
        raise ValueError("One image slot per query, including the initial view, is required")
    edges, errors = [], []
    anchored = True
    predicted = np.zeros(2, dtype=float)
    anchored_edges = 0
    length = 0.0
    for i in range(1, len(images)):
        result = estimate_translation(images[i-1], images[i], config)
        command = np.asarray(queries[i], dtype=float)-np.asarray(queries[i-1], dtype=float)
        result["edge"] = i
        result["normalized_step_error"] = None
        if result["status"] == "measured":
            delta = np.asarray(result["delta"], dtype=float)
            result["normalized_step_error"] = float(np.linalg.norm(delta-command)/DIAGONAL)
            errors.append(result["normalized_step_error"])
            length += float(np.linalg.norm(delta))
            if anchored:
                predicted += delta
                anchored_edges += 1
        else:
            anchored = False  # Never re-anchor with ground truth after a broken edge.
        edges.append(result)
    complete = bool(edges) and len(errors) == len(edges)
    return {"status": "diagnostic_only", "config": asdict(config), "edges": edges,
            "planned_edges": len(edges), "measured_edges": len(errors),
            "coverage": len(errors)/len(edges) if edges else None,
            "mean_normalized_step_error": float(np.mean(errors)) if errors else None,
            "state_counts": dict(Counter(e["status"] for e in edges)),
            "anchored_prefix_edges": anchored_edges,
            "full_path_measurable": complete,
            "estimated_full_path_length": length if complete else None,
            "normalized_endpoint_error": float(np.linalg.norm(predicted -
                (np.asarray(queries[-1])-np.asarray(queries[0])))/DIAGONAL) if complete else None,
            "release_validated": False}


def paired_path_ratio(small, large):
    eligible = (small["full_path_measurable"] and large["full_path_measurable"] and
                small["estimated_full_path_length"] > 0)
    return {"planned_pairs": 1, "measurable_pairs": int(eligible),
            "large_over_small": large["estimated_full_path_length"]/small["estimated_full_path_length"] if eligible else None}
