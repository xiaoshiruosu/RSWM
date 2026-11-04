"""Audited raster/frontier/lazy maps sharing the unmodified WindowMap completer.

This module guarantees bookkeeping and hard pixel preservation, not generated
remote-sensing quality. Completers must be stateless given (image, known, seed).
The operation log measures actual calls and is NOT rolled back with F7 state.
"""
from __future__ import annotations
import copy
import time
import numpy as np
from .baselines import WindowMap
from .geometry import WORLD, WIDTH, HEIGHT, LIMIT, box, rgb


class AuditedWindowMap(WindowMap):
    def __init__(self, completer=None, order="frontier"):
        if order not in ("raster", "frontier", "lazy"):
            raise ValueError("order must be raster, frontier or lazy.")
        super().__init__(completer, eager=False)
        self.order = order
        self.operation_log = []
        self.session_index = -1
        self._phase = "uninitialized"
        self._ready = False

    @staticmethod
    def _window(ix, iy):
        x, y = ix * 256, iy * 256
        return max(0, x - 128), max(0, y - 128), min(WORLD, x + 384), min(WORLD, y + 384)

    def _context_counts(self, ix, iy):
        xa, ya, xb, yb = self._window(ix, iy)
        area = (xb - xa) * (yb - ya)
        real = int(self.initial_mask[ya:yb, xa:xb].sum())
        known = int(self.known[ya:yb, xa:xb].sum())
        return {"real_known_pixels": real, "generated_history_pixels": known - real,
                "unknown_in_world_pixels": area - known, "padding_pixels": 512**2 - area,
                "new_core_pixels": int((~self.known[iy*256:(iy+1)*256, ix*256:(ix+1)*256]).sum())}

    def reset(self, initial, spec, seed):
        self._ready = False
        rgb(initial)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if not isinstance(spec, dict) or spec.get("viewport_hw") != [HEIGHT, WIDTH]:
            raise ValueError("This map supports only the frozen Inria viewport.")
        if spec.get("query_bounds_xy") != [[-LIMIT, LIMIT], [-LIMIT, LIMIT]]:
            raise ValueError("This map supports only the frozen Inria query bounds.")
        self.session_index += 1
        self._phase = "initialize"
        self.initial_mask = np.zeros((WORLD, WORLD), dtype=bool)
        xa, ya, xb, yb = box((0, 0), (1024, 1024))
        self.initial_mask[ya:yb, xa:xb] = True
        super().reset(initial, spec, seed)
        if self.order == "raster":
            for iy in range(8):
                for ix in range(8):
                    self._fill(ix, iy)
        elif self.order == "frontier":
            pending = {(ix, iy) for iy in range(8) for ix in range(8)}
            while pending:
                candidates = []
                completed = []
                for ix, iy in sorted(pending, key=lambda p: (p[1], p[0])):
                    c = self._context_counts(ix, iy)
                    if c["new_core_pixels"] == 0:
                        completed.append((ix, iy))
                    elif c["real_known_pixels"] + c["generated_history_pixels"] > 0:
                        distance = (ix * 256 + 128 - 1024)**2 + (iy * 256 + 128 - 1024)**2
                        candidates.append((-c["real_known_pixels"], -c["generated_history_pixels"],
                                           distance, iy, ix))
                pending.difference_update(completed)
                if not pending:
                    break
                if not candidates:
                    raise RuntimeError("No known-support frontier; no unconditional fallback.")
                _, _, _, iy, ix = min(candidates)
                self._fill(ix, iy)
                pending.remove((ix, iy))
        self._phase = "generate"
        self._ready = True

    def _fill(self, ix, iy):
        if type(ix) is not int or type(iy) is not int or not (0 <= ix < 8 and 0 <= iy < 8):
            raise ValueError("Window index outside map.")
        c = self._context_counts(ix, iy)
        if c["new_core_pixels"] == 0:
            return
        row = {"operation_index": len(self.operation_log), "session_index": self.session_index,
               "order": self.order, "phase": self._phase, "window_xy": [ix, iy],
               **c, "status": "failed", "committed_core_pixels": 0, "error_type": None}
        before = time.perf_counter()
        try:
            # Reuse original cropping, padding, per-window seed and hard protection.
            super()._fill(ix, iy)
            row["status"] = "completed"
            row["committed_core_pixels"] = c["new_core_pixels"]
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            raise
        finally:
            row["cpu_wall_seconds"] = time.perf_counter() - before
            row["gpu_synchronized_seconds"] = None
            self.operation_log.append(row)

    def generate(self, queries):
        if not self._ready:
            raise RuntimeError("A successful reset/restore is required.")
        a = np.asarray(queries)
        if a.ndim != 2 or a.shape[1] != 2 or a.dtype.kind not in "iu" or not len(a):
            raise ValueError("Expected nonempty integer query batch.")
        if np.any(a > LIMIT) or (a.dtype.kind == "i" and np.any(a < -LIMIT)):
            raise ValueError("Query outside the frozen core bounds.")
        return super().generate(a.astype(np.int64))

    def snapshot(self):
        if not self._ready:
            raise RuntimeError("Cannot snapshot an uninitialized or failed map.")
        return {"schema": "rswm-audited-map-state-v1", "order": self.order,
                "state": super().snapshot()}

    def restore(self, state):
        self._ready = False
        if not isinstance(state, dict) or state.get("schema") != "rswm-audited-map-state-v1" or state.get("order") != self.order:
            raise ValueError("Snapshot schema or map order mismatch.")
        s = state.get("state")
        if not isinstance(s, dict) or set(s) != {"map", "known", "seed", "calls", "generated_pixels"}:
            raise ValueError("Incomplete map state.")
        rgb(s["map"], (WORLD, WORLD, 3))
        if not isinstance(s["known"], np.ndarray) or s["known"].dtype != bool or s["known"].shape != (WORLD, WORLD):
            raise ValueError("Invalid known map.")
        for field in ("seed", "calls", "generated_pixels"):
            if type(s[field]) is not int or s[field] < 0:
                raise ValueError("Invalid state counter.")
        if s["generated_pixels"] != int(s["known"].sum()) - HEIGHT * WIDTH:
            raise ValueError("Known pixels and state counter disagree.")
        self.initial_mask = np.zeros((WORLD, WORLD), dtype=bool)
        x0, y0, x1, y1 = box((0, 0), (1024, 1024))
        self.initial_mask[y0:y1, x0:x1] = True
        if not s["known"][self.initial_mask].all():
            raise ValueError("Snapshot lost the real initial region.")
        super().restore(s)
        self._phase = "generate"
        self._ready = True
        # Deliberately retain all actually executed operation_log entries.

    def activity_report(self):
        rows = copy.deepcopy(self.operation_log)
        return {"schema": "rswm-map-activity-v1", "order": self.order,
                "attempted_fill_calls": len(rows),
                "successful_fill_calls": sum(r["status"] == "completed" for r in rows),
                "actually_committed_core_pixels": sum(r["committed_core_pixels"] for r in rows),
                "cpu_wall_seconds": sum(r["cpu_wall_seconds"] for r in rows),
                "gpu_synchronized_seconds": None, "operation_log": rows,
                "scope": "object lifetime, including work later rolled back by restore",
                "completer_contract": "stateless(image,known,seed); no backend cache certification",
                "resource_accounting_complete": False, "real_model_validated": False}

    def resource_state(self):
        # These arrays include both the real and generated regions. The frozen
        # completer is shared, never copied into dynamic state or snapshots.
        # Log containers are declared too; state_storage explicitly excludes
        # Python object overhead rather than pretending to measure their RAM.
        return {name: getattr(self, name) for name in ("map", "known", "initial_mask", "operation_log")
                if hasattr(self, name)}
