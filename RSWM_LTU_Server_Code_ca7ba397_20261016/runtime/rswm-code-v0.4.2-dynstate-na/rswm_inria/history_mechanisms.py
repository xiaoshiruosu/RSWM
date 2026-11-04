"""Explicit history retrieval mechanisms using only committed public frames.

The completer receives a viewport and protection mask. These wrappers are
mechanism experiments, not reproductions of a particular trained backbone.
"""
from __future__ import annotations

import copy
import hashlib
import uuid
import numpy as np

from .geometry import box, canonical, intersection, rgb


POLICY_SCHEMA = "rswm-history-policy-v1"


def validate_policy(policy):
    required = {"schema", "mechanism", "capacity", "protect_initial", "protect_history"}
    if not isinstance(policy, dict) or set(policy) != required or policy["schema"] != POLICY_SCHEMA:
        raise ValueError("Invalid history policy schema")
    if policy["mechanism"] not in {"last_frame", "frame_cache", "map_assisted"}:
        raise ValueError("Unknown history mechanism")
    if type(policy["capacity"]) is not int or not 1 <= policy["capacity"] <= 256:
        raise ValueError("Bounded positive frame capacity required")
    if policy["mechanism"] == "last_frame" and policy["capacity"] != 1:
        raise ValueError("LastFrame retains exactly one committed frame")
    if any(type(policy[key]) is not bool for key in ("protect_initial", "protect_history")):
        raise ValueError("Protection flags must be boolean")
    return copy.deepcopy(policy)


class HistoryCompleterAdapter:
    """LastFrame, bounded FrameCache or generated MapAssisted context.

    No private world is accepted. Failed frames do not become cache entries.
    Counter lifetime is outside logical snapshots and never rolls back.
    """
    supported_world_hw = {(1024, 1024), (2048, 2048)}

    def __init__(self, completer, policy):
        self.completer = completer
        self.policy = validate_policy(policy)
        self.resource_counter_epoch = uuid.uuid4().hex
        self.counters = {key: 0 for key in ("completion_calls", "native_frames", "retrievals", "retrieval_hits", "retrieved_pixels", "committed_frames", "cache_evictions")}

    def reset(self, initial, spec, seed):
        shape = tuple(spec.get("world_hw", (2048, 2048)))
        if shape not in self.supported_world_hw or spec.get("viewport_hw") != [240, 416]:
            raise ValueError("Unsupported world/viewport specification")
        self.shape, self.origin = shape, (shape[1] // 2, shape[0] // 2)
        self.bounds = copy.deepcopy(spec["query_bounds_xy"])
        self.initial = rgb(initial).copy()
        self.seed, self.call_index = int(seed), 0
        self.cache = [(np.array([0, 0], np.int64), self.initial.copy(), True)]
        self.map = np.zeros((*shape, 3), np.uint8)
        self.seen = np.zeros(shape, bool)
        x0, y0, x1, y1 = box((0, 0), self.origin)
        self.map[y0:y1, x0:x1] = self.initial
        self.retrieval_trace = []

    def _validate_queries(self, queries):
        q = np.asarray(queries)
        if q.ndim != 2 or q.shape[1] != 2 or q.dtype.kind not in "iu" or not 1 <= len(q) <= 32:
            raise ValueError("Bounded integer query batch required")
        for axis in range(2):
            lo, hi = self.bounds[axis]
            if np.any(q[:, axis] > hi) or (q.dtype.kind == "i" and np.any(q[:, axis] < lo)):
                raise ValueError("Query outside declared bounds")
        return q.astype(np.int64)

    def _context(self, q):
        image = np.zeros((240, 416, 3), np.uint8)
        available = np.zeros((240, 416), bool)
        mechanism = self.policy["mechanism"]
        selected = None
        selected_is_initial = False
        if mechanism == "map_assisted":
            x0, y0, x1, y1 = box(q, self.origin)
            image[:] = self.map[y0:y1, x0:x1]
            available[:] = self.seen[y0:y1, x0:x1]
        else:
            candidates = []
            for i, (old_q, frame, is_initial) in enumerate(self.cache):
                pair = intersection(box(old_q), box(q))
                area = 0 if pair is None else (pair[1][0].stop - pair[1][0].start) * (pair[1][1].stop - pair[1][1].start)
                candidates.append((area, i, pair, old_q, frame, is_initial))
            if candidates:
                area, i, pair, old_q, frame, selected_is_initial = max(candidates, key=lambda item: (item[0], item[1]))
                selected = old_q.tolist()
                if pair is not None:
                    image[pair[1]] = frame[pair[0]]
                    available[pair[1]] = True
        protected = available.copy() if self.policy["protect_history"] and not selected_is_initial else np.zeros_like(available)
        initial_pair = intersection(box((0, 0)), box(q))
        if initial_pair is not None:
            # The only real observation is I0. Never consult the private world.
            if self.policy["protect_initial"]:
                image[initial_pair[1]] = self.initial[initial_pair[0]]
                protected[initial_pair[1]] = True
            elif not self.policy["protect_history"]:
                image[initial_pair[1]] = self.initial[initial_pair[0]]
        self.counters["retrievals"] += 1
        self.counters["retrieval_hits"] += int(available.any())
        self.counters["retrieved_pixels"] += int(available.sum())
        self.retrieval_trace.append({"q": q.tolist(), "selected_frame_q": selected,
                                     "retrieved_pixels": int(available.sum()), "protected_pixels": int(protected.sum())})
        return image, protected

    def generate(self, queries):
        outputs = []
        for q in self._validate_queries(queries):
            context, protected = self._context(q)
            seed = int.from_bytes(hashlib.sha256(canonical([self.seed, self.call_index, q.tolist()])).digest()[:8], "big") % (2**63 - 1)
            self.call_index += 1
            self.counters["completion_calls"] += 1
            result = self.completer(context.copy(), protected.copy(), seed)
            # A returned malformed frame still represents attempted native work.
            self.counters["native_frames"] += 1
            frame = rgb(result).copy()
            frame[protected] = context[protected]
            x0, y0, x1, y1 = box(q, self.origin)
            self.map[y0:y1, x0:x1] = frame
            self.seen[y0:y1, x0:x1] = True
            self.cache.append((q.copy(), frame.copy(), False))
            while len(self.cache) > self.policy["capacity"]:
                self.cache.pop(0)
                self.counters["cache_evictions"] += 1
            self.counters["committed_frames"] += 1
            outputs.append(frame)
        return outputs

    def resource_state(self):
        return {"initial": self.initial, "map": self.map, "seen": self.seen,
                "cache": [frame for _, frame, _ in self.cache]}

    def resource_counters(self):
        return dict(self.counters)

    def snapshot(self):
        keys = ("shape", "origin", "bounds", "initial", "seed", "call_index", "cache", "map", "seen", "retrieval_trace")
        return {key: copy.deepcopy(getattr(self, key)) for key in keys}

    def restore(self, state):
        required = {"shape", "origin", "bounds", "initial", "seed", "call_index", "cache", "map", "seen", "retrieval_trace"}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("History snapshot must contain only logical state")
        for key, value in state.items():
            setattr(self, key, copy.deepcopy(value))
