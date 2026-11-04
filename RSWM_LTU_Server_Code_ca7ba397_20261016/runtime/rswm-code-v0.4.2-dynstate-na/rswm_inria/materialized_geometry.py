"""Independent, deliberately slow checks of materialized task geometry.

Only the frozen protocol constants are shared with the generator.  Visibility
is evaluated separately for every event and every target row; no task label or
path-validation implementation is used here.  This provides a second algorithm
for auditing persisted private labels rather than repeating their checksum.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .geometry import HEIGHT, LIMIT, ORIGINS, VERSION, WIDTH


LABEL_FIELDS = frozenset({
    "first_event", "last_prefix_visible_event", "reentry_event",
    "invisible_query_gap", "fully_absent_calls", "x", "y",
})


def _integer_path(q, *, length=None):
    path = np.asarray(q)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or path.dtype.kind not in "iu":
        raise ValueError("Audit queries must be an integer [T,2] array.")
    if length is not None and len(path) != length:
        raise ValueError("Audit query count disagrees with the frozen family.")
    # Check before converting unsigned values to signed int64.
    if np.any(path > LIMIT) or (path.dtype.kind == "i" and np.any(path < -LIMIT)):
        raise ValueError("Audit query coordinates exceed the frozen limits.")
    path = path.astype(np.int64)
    if np.any(path[0] != 0):
        raise ValueError("Audit q0 must be the origin.")
    for previous, current in zip(path[:-1], path[1:]):
        dx, dy = abs(int(current[0]) - int(previous[0])), abs(int(current[1]) - int(previous[1]))
        overlap_area = max(0, WIDTH - dx) * max(0, HEIGHT - dy)
        if 100 * overlap_area < 65 * WIDTH * HEIGHT:
            raise ValueError("Audit adjacent viewport overlap is below 0.65.")
    return path


def independent_target_labels(q, center, *, prefix_end=32, target_size=128, min_gap=32):
    """Derive every target-pixel label using event-by-event half-open boxes.

    Event 0 is I0.  Calls contain events 1..16, 17..32, and so on.  F7 streams
    must each include their own shared prefix when passed to this function.
    Returned coordinates are relative to the initial viewport center, matching
    the materialized label archive contract.
    """
    path = _integer_path(q)
    if len(path) - 1 > np.iinfo(np.int16).max:
        raise ValueError("Audit event indices exceed the frozen label integer range.")
    target = np.asarray(center)
    if target.shape != (2,) or target.dtype.kind not in "iu":
        raise ValueError("Audit target center must be an integer pair.")
    if np.any(target > LIMIT) or (target.dtype.kind == "i" and np.any(target < -LIMIT)):
        raise ValueError("Audit target center exceeds the frozen limits.")
    if type(prefix_end) is not int or not 0 < prefix_end < len(path) - 1:
        raise ValueError("Audit prefix must leave a nonempty suffix.")
    if type(target_size) is not int or target_size <= 0 or target_size % 2:
        raise ValueError("Audit target size must be a positive even integer.")
    if type(min_gap) is not int or min_gap < 32:
        raise ValueError("Audit minimum gap cannot weaken the frozen 32-query contract.")

    arrays = {name: np.empty((target_size, target_size), dtype=np.int16)
              for name in LABEL_FIELDS - {"x", "y"}}
    arrays["x"] = np.empty((target_size, target_size), dtype=np.int64)
    arrays["y"] = np.empty((target_size, target_size), dtype=np.int64)
    x_values = np.arange(int(target[0]) - target_size // 2,
                         int(target[0]) + target_size // 2, dtype=np.int64)
    y_start = int(target[1]) - target_size // 2
    for row in range(target_size):
        y = y_start + row
        first = np.full(target_size, -1, dtype=np.int64)
        last = np.full(target_size, -1, dtype=np.int64)
        reentry = np.full(target_size, -1, dtype=np.int64)
        for event, (qx, qy) in enumerate(path):
            # Deliberately reconstruct this event's viewport, independently of
            # previous visibility and without a [T,H,W] generator mask.
            x0, x1 = int(qx) - WIDTH // 2, int(qx) + WIDTH // 2
            y0, y1 = int(qy) - HEIGHT // 2, int(qy) + HEIGHT // 2
            visible = ((x_values >= x0) & (x_values < x1)) if y0 <= y < y1 else np.zeros(target_size, bool)
            if event in (0, prefix_end) and visible.any():
                raise ValueError("Audit target is visible in I0 or at prefix end.")
            if event <= prefix_end:
                first[(first < 0) & visible] = event
                last[visible] = event
            else:
                reentry[(reentry < 0) & visible] = event
        if np.any(first < 1) or np.any(last < 1) or np.any(reentry < 0):
            raise ValueError("Audit target lacks complete prefix exposure or suffix reentry.")
        gap = reentry - last - 1
        if np.any(gap < min_gap):
            raise ValueError("Audit target invisibility gap is too short.")
        if np.any((first - 1) // 16 == (reentry - 1) // 16):
            raise ValueError("Audit first appearance and reentry share a call.")
        # Count complete calls explicitly, without the generator's difference
        # of block-index formula.  A call is wholly inside the invisible gap.
        absent = np.zeros(target_size, dtype=np.int16)
        for call_start in range(1, len(path), 16):
            call_end = call_start + 15
            if call_end < len(path):
                absent += ((call_start > last) & (call_end < reentry)).astype(np.int16)
        arrays["first_event"][row] = first
        arrays["last_prefix_visible_event"][row] = last
        arrays["reentry_event"][row] = reentry
        arrays["invisible_query_gap"][row] = gap
        arrays["fully_absent_calls"][row] = absent
        arrays["x"][row] = x_values
        arrays["y"][row] = y
    return arrays


def audit_task_geometry(task, stored_labels=None):
    """Check an entire frozen task and compare every stored label element.

    Returns stream-to-label dictionaries for F5/F7 and an empty dictionary for
    F2/F3.  When labels are supplied, exact stream names, field names, shapes,
    integer dtypes, and values are required, including all 16,384 target pixels.
    """
    families = {"F2": ({"main"}, 33, "offline"),
                "F3": ({"small", "large"}, 33, "offline"),
                "F5": ({"main"}, 97, "online16"),
                "F7": ({"a", "b"}, 97, "online16")}
    if not isinstance(task, Mapping) or task.get("version") != VERSION or task.get("family") not in families:
        raise ValueError("Audit task family or version is invalid.")
    names, length, protocol = families[task["family"]]
    programs = task.get("programs")
    if not isinstance(programs, Mapping) or set(programs) != names or task.get("protocol") != protocol:
        raise ValueError("Audit program names or protocol disagree with the family.")
    origin_index = task.get("origin_index")
    if type(origin_index) is not int or origin_index not in range(4):
        raise ValueError("Audit origin index is invalid.")
    origin = np.asarray(task.get("origin"))
    if origin.shape != (2,) or origin.dtype.kind not in "iu" or tuple(origin) != ORIGINS[origin_index]:
        raise ValueError("Audit origin disagrees with the frozen specification.")
    paths = {name: _integer_path(q, length=length) for name, q in programs.items()}
    family = task["family"]
    if family == "F3" and not np.array_equal(paths["small"] * 2, paths["large"]):
        raise ValueError("Audit F3 programs are not an exact integer 2x pair.")
    if family == "F7" and not np.array_equal(paths["a"][:33], paths["b"][:33]):
        raise ValueError("Audit F7 streams do not share all 33 prefix events.")
    computed = {}
    if family in ("F5", "F7"):
        metadata = task.get("metadata")
        if not isinstance(metadata, Mapping) or "target_center" not in metadata:
            raise ValueError("Audit persistence task has no target center.")
        target = metadata["target_center"]
        for name, path in paths.items():
            if not np.array_equal(path[16], target):
                raise ValueError("Audit target is not centered at planned B.")
            computed[name] = independent_target_labels(path, target)
    if stored_labels is not None:
        if not isinstance(stored_labels, Mapping) or set(stored_labels) != set(computed):
            raise ValueError("Audit stored private label stream names disagree.")
        for name, expected in computed.items():
            actual = stored_labels[name]
            if not isinstance(actual, Mapping) or set(actual) != LABEL_FIELDS:
                raise ValueError("Audit stored private label fields disagree.")
            for field, expected_values in expected.items():
                actual_values = np.asarray(actual[field])
                if (actual_values.dtype.kind not in "iu" or actual_values.shape != expected_values.shape
                        or not np.array_equal(actual_values, expected_values)):
                    raise ValueError(f"Audit stored labels disagree: {name}.{field}.")
    return computed
