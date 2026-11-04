"""Descriptive completion bounds for bounded RGB error; NOT confidence intervals."""
from __future__ import annotations
import math


def rgb_missing_bounds(expected_pixels: int, valid_pixels: int,
                       absolute_error_rgb: int) -> dict:
    for value in (expected_pixels, valid_pixels, absolute_error_rgb):
        if type(value) is not int or value < 0:
            raise ValueError("Counts and uint8 absolute error sums must be nonnegative integers.")
    if valid_pixels > expected_pixels or absolute_error_rgb > 3 * 255 * valid_pixels:
        raise ValueError("Counts or error sum outside normalized RGB bounds.")
    if expected_pixels == 0:
        return {"state": "not_applicable", "coverage": None, "conditional_mae": None,
                "lower": None, "upper": None, "interpretation": "not_a_confidence_interval"}
    from .aggregate import missing_bound
    c = valid_pixels / expected_pixels
    conditional = absolute_error_rgb / (3 * 255 * valid_pixels) if valid_pixels else None
    lower, upper = missing_bound(conditional, c)
    return {"state": "measured" if valid_pixels else "missing", "coverage": c,
            "conditional_mae": conditional,
            "lower": lower, "upper": upper,
            "interpretation": "all missing per-pixel normalized RGB errors range over [0,1]"}


def weighted_bounds(rows, weights):
    """Apply explicitly supplied planned weights; never drop missing rows.

    Structural not-applicability is returned as null, requiring the caller to
    define its denominator before invoking this function. Weights must sum to 1.
    """
    if not rows or len(rows) != len(weights):
        raise ValueError("Require equally sized nonempty rows and weights.")
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Planned weights must be finite and positive.")
    if not math.isclose(sum(weights), 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError("Planned weights must sum to one; no implicit reweighting.")
    for r in rows:
        if not isinstance(r, dict) or not {"lower", "upper"} <= r.keys():
            raise ValueError("Malformed bound row.")
        lo, hi = r["lower"], r["upper"]
        if lo is None or hi is None:
            if lo is not None or hi is not None:
                raise ValueError("Both endpoints must be null together.")
            continue
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (lo, hi)) or not 0 <= lo <= hi <= 1 + 1e-12:
            raise ValueError("Invalid RGB completion bounds.")
    if any(r["lower"] is None for r in rows):
        return {"lower": None, "upper": None, "state": "contains_not_applicable",
                "note": "Explicitly define structural applicability before aggregation."}
    return {"lower": sum(w * r["lower"] for w, r in zip(weights, rows)),
            "upper": sum(w * r["upper"] for w, r in zip(weights, rows)),
            "state": "descriptive_completion_bounds", "is_confidence_interval": False}
