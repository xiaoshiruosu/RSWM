#!/usr/bin/env python3
"""Fail-closed, testable GPU-idle confirmation for the RSWM scheduler.

This helper only reads ``nvidia-smi``.  It never allocates CUDA memory, changes
GPU state, or signals another process.  The shell scheduler owns the overall
workflow; keeping the polling algorithm here makes its race-sensitive behavior
unit-testable without a GPU.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence


GPU_QUERY = (
    "index,name,memory.total,memory.used,memory.free,utilization.gpu"
)


class GPUSelectionError(RuntimeError):
    """GPU state could not be read or was unsafe to interpret."""


class NoGPUAvailable(GPUSelectionError):
    """A bounded test scan ended without a confirmed GPU."""


@dataclass(frozen=True)
class GPUStat:
    index: str
    name: str
    total_mb: float
    used_mb: float
    free_mb: float
    utilization: float

    def is_idle(self, *, max_util: float, max_used_mb: float) -> bool:
        return self.utilization < max_util and self.used_mb < max_used_mb


@dataclass(frozen=True)
class GPUSelection:
    stat: GPUStat
    baseline_pids: tuple[int, ...]
    observations: int


def parse_gpu_rows(text: str) -> list[GPUStat]:
    rows: list[GPUStat] = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) != 6:
            raise GPUSelectionError(
                f"nvidia-smi row {line_number} has {len(parts)} fields, expected 6: {raw!r}"
            )
        try:
            rows.append(GPUStat(
                index=parts[0],
                name=parts[1],
                total_mb=float(parts[2]),
                used_mb=float(parts[3]),
                free_mb=float(parts[4]),
                utilization=float(parts[5]),
            ))
        except ValueError as exc:
            raise GPUSelectionError(
                f"nvidia-smi row {line_number} contains a non-numeric metric: {raw!r}"
            ) from exc
    if not rows:
        raise GPUSelectionError("nvidia-smi returned no GPU rows")
    if len({row.index for row in rows}) != len(rows):
        raise GPUSelectionError("nvidia-smi returned duplicate physical GPU indices")
    return rows


class NvidiaSMIProbe:
    """Read physical GPU metrics and compute-process PIDs via nvidia-smi."""

    def __init__(self, executable: str = "nvidia-smi", timeout: float = 15.0):
        self.executable = executable
        self.timeout = timeout

    def __call__(self) -> list[GPUStat]:
        cmd = [
            self.executable,
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ]
        try:
            cp = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GPUSelectionError(f"cannot execute nvidia-smi: {exc}") from exc
        if cp.returncode != 0:
            detail = (cp.stderr or cp.stdout).strip()
            raise GPUSelectionError(
                f"nvidia-smi GPU query failed with exit {cp.returncode}: {detail}"
            )
        return parse_gpu_rows(cp.stdout)

    def compute_pids(self, physical_gpu: str) -> set[int]:
        cmd = [
            self.executable,
            "-i", str(physical_gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
        try:
            cp = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise GPUSelectionError(
                f"cannot query compute processes for GPU {physical_gpu}: {exc}"
            ) from exc
        if cp.returncode != 0:
            detail = (cp.stderr or cp.stdout).strip()
            raise GPUSelectionError(
                f"compute-process query for GPU {physical_gpu} failed with "
                f"exit {cp.returncode}: {detail}"
            )
        pids: set[int] = set()
        for raw in cp.stdout.splitlines():
            value = raw.strip()
            if not value or value.lower().startswith("no running"):
                continue
            try:
                pids.add(int(value.split(",", 1)[0].strip()))
            except ValueError as exc:
                raise GPUSelectionError(
                    f"unrecognized compute-process row for GPU {physical_gpu}: {raw!r}"
                ) from exc
        return pids


def resolve_allowed_indices(
        cuda_visible_devices: str | None,
        rows: Sequence[GPUStat]) -> list[str]:
    """Map a numeric CUDA_VISIBLE_DEVICES restriction to physical GPU IDs.

    UUID/MIG selectors are rejected rather than guessed.  That is conservative
    on the current homogeneous A800 node and guarantees the scheduler never
    widens a user-provided visibility restriction.
    """

    available = {row.index for row in rows}
    if cuda_visible_devices is None:
        return sorted(available, key=int)
    value = cuda_visible_devices.strip()
    if not value or value == "-1":
        raise GPUSelectionError(
            "CUDA_VISIBLE_DEVICES permits no GPU; unset it or provide numeric physical IDs"
        )
    tokens = [token.strip() for token in value.split(",")]
    if any(not token.isdigit() for token in tokens):
        raise GPUSelectionError(
            "CUDA_VISIBLE_DEVICES must contain numeric physical GPU IDs for the "
            f"shared-node scheduler; got {cuda_visible_devices!r}"
        )
    ordered: list[str] = []
    for token in tokens:
        canonical = str(int(token))
        if canonical not in available:
            raise GPUSelectionError(
                f"CUDA_VISIBLE_DEVICES includes GPU {canonical}, absent from nvidia-smi"
            )
        if canonical not in ordered:
            ordered.append(canonical)
    return ordered


def select_idle_gpu(
        probe: Callable[[], Sequence[GPUStat]],
        process_probe: Callable[[str], Iterable[int]],
        *,
        cuda_visible_devices: str | None,
        confirm_seconds: float,
        poll_seconds: float,
        max_util: float,
        max_used_mb: float,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        on_event: Callable[[str, Mapping[str, object]], None] | None = None,
        max_scans: int | None = None,
        ) -> GPUSelection:
    """Wait for one candidate to remain idle for the full confirmation window.

    A candidate is checked on every poll.  Utilization/memory threshold failure,
    disappearance, or any compute PID not present at candidacy resets the whole
    scan.  Even a zero-second test window requires two metric observations.
    """

    if confirm_seconds < 0:
        raise ValueError("confirm_seconds must be non-negative")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    if not 0 <= max_util <= 100:
        raise ValueError("max_util must be between 0 and 100")
    if max_used_mb <= 0:
        raise ValueError("max_used_mb must be positive")

    emit = on_event or (lambda _event, _payload: None)
    scans = 0

    def query() -> tuple[list[GPUStat], list[str]]:
        nonlocal scans
        if max_scans is not None and scans >= max_scans:
            raise NoGPUAvailable(
                f"no GPU confirmed within the bounded {max_scans}-scan test"
            )
        rows = list(probe())
        scans += 1
        allowed = resolve_allowed_indices(cuda_visible_devices, rows)
        emit("scan", {"rows": rows, "allowed": allowed, "scan": scans})
        return rows, allowed

    while True:
        rows, allowed = query()
        rank = {gpu: pos for pos, gpu in enumerate(allowed)}
        candidates = [
            row for row in rows
            if row.index in rank
            and row.is_idle(max_util=max_util, max_used_mb=max_used_mb)
        ]
        if not candidates:
            emit("waiting", {"reason": "all allowed GPUs are busy"})
            sleep(poll_seconds)
            continue

        candidate = min(
            candidates,
            key=lambda row: (row.used_mb, row.utilization, rank[row.index]),
        )
        baseline_pids = set(process_probe(candidate.index))
        started = monotonic()
        observations = 1
        emit("candidate", {
            "gpu": candidate.index,
            "stat": candidate,
            "baseline_pids": sorted(baseline_pids),
        })

        while True:
            elapsed = max(0.0, monotonic() - started)
            remaining = max(0.0, confirm_seconds - elapsed)
            sleep(min(poll_seconds, remaining) if remaining > 0 else poll_seconds)
            rows, allowed = query()
            observations += 1
            current = next(
                (row for row in rows
                 if row.index == candidate.index and row.index in allowed),
                None,
            )
            current_pids = set(process_probe(candidate.index))
            new_pids = current_pids - baseline_pids
            elapsed = max(0.0, monotonic() - started)

            if current is None:
                emit("reset", {
                    "gpu": candidate.index,
                    "reason": "candidate disappeared or visibility changed",
                })
                break
            if not current.is_idle(max_util=max_util, max_used_mb=max_used_mb):
                emit("reset", {
                    "gpu": candidate.index,
                    "reason": (
                        f"busy: util={current.utilization:g}% "
                        f"used={current.used_mb:g}MiB"
                    ),
                })
                break
            if new_pids:
                emit("reset", {
                    "gpu": candidate.index,
                    "reason": f"new compute process(es): {sorted(new_pids)}",
                })
                break

            emit("progress", {
                "gpu": candidate.index,
                "elapsed": min(elapsed, confirm_seconds),
                "required": confirm_seconds,
                "stat": current,
            })
            if elapsed >= confirm_seconds:
                emit("confirmed", {
                    "gpu": candidate.index,
                    "stat": current,
                    "baseline_pids": sorted(current_pids),
                })
                return GPUSelection(
                    stat=current,
                    baseline_pids=tuple(sorted(current_pids)),
                    observations=observations,
                )


def check_gpu_now(
        probe: Callable[[], Sequence[GPUStat]],
        process_probe: Callable[[str], Iterable[int]],
        physical_gpu: str,
        *,
        baseline_pids: Iterable[int],
        max_util: float,
        max_used_mb: float,
        ) -> tuple[bool, str, GPUStat | None, tuple[int, ...]]:
    """Perform the immediate pre-launch metric and new-process check."""

    rows = list(probe())
    current = next((row for row in rows if row.index == str(physical_gpu)), None)
    if current is None:
        return False, "selected GPU disappeared from nvidia-smi", None, ()
    pids = tuple(sorted(set(process_probe(str(physical_gpu)))))
    if not current.is_idle(max_util=max_util, max_used_mb=max_used_mb):
        return (
            False,
            f"GPU is busy: util={current.utilization:g}% used={current.used_mb:g}MiB",
            current,
            pids,
        )
    new_pids = set(pids) - set(baseline_pids)
    if new_pids:
        return False, f"new compute process(es): {sorted(new_pids)}", current, pids
    return True, "idle thresholds and compute-process baseline still hold", current, pids


def _atomic_env(path: str | os.PathLike[str], values: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    lines = [f"{key}={shlex.quote(str(value))}\n" for key, value in values.items()]
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S%z")


def _format_stat(stat: GPUStat) -> str:
    return (
        f"GPU {stat.index} {stat.name} util={stat.utilization:g}% "
        f"used={stat.used_mb:g}MiB free={stat.free_mb:g}MiB/{stat.total_mb:g}MiB"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Confirm one shared-server GPU is stably idle using nvidia-smi"
    )
    parser.add_argument("--confirm-seconds", type=float, default=600.0)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--max-util", type=float, default=10.0)
    parser.add_argument("--max-used-mb", type=float, default=5000.0)
    parser.add_argument("--visible-devices", default=None)
    parser.add_argument("--result-file")
    parser.add_argument("--state-file")
    parser.add_argument("--check-now", metavar="PHYSICAL_GPU")
    parser.add_argument("--baseline-pids", default="")
    parser.add_argument(
        "--nvidia-smi", default=os.environ.get("RSWM_NVIDIA_SMI", "nvidia-smi")
    )
    args = parser.parse_args(argv)

    probe = NvidiaSMIProbe(args.nvidia_smi)
    baseline = {
        int(value) for value in args.baseline_pids.split(",") if value.strip()
    }
    if args.check_now is not None:
        ok, reason, stat, pids = check_gpu_now(
            probe, probe.compute_pids, args.check_now,
            baseline_pids=baseline,
            max_util=args.max_util,
            max_used_mb=args.max_used_mb,
        )
        if stat is not None:
            print(f"[GPU {_timestamp()}] immediate check: {_format_stat(stat)}", flush=True)
        print(f"[GPU {_timestamp()}] immediate check: {'PASS' if ok else 'RETRY'}: {reason}", flush=True)
        return 0 if ok else 3

    if not args.result_file:
        parser.error("--result-file is required unless --check-now is used")

    def on_event(event: str, payload: Mapping[str, object]) -> None:
        candidate = payload.get("gpu", "")
        elapsed = payload.get("elapsed", "")
        if event == "scan":
            allowed = set(payload["allowed"])
            for stat in payload["rows"]:
                marker = "allowed" if stat.index in allowed else "restricted"
                print(f"[GPU {_timestamp()}] {_format_stat(stat)} [{marker}]", flush=True)
        elif event == "candidate":
            print(
                f"[GPU {_timestamp()}] GPU {candidate} candidate; "
                f"baseline_compute_pids={payload['baseline_pids']}", flush=True,
            )
        elif event == "progress":
            print(
                f"[GPU {_timestamp()}] GPU {candidate} idle "
                f"{float(elapsed):.0f}/{float(payload['required']):.0f} sec",
                flush=True,
            )
        elif event == "reset":
            print(
                f"[GPU {_timestamp()}] GPU {candidate} candidate cancelled: "
                f"{payload['reason']}; rescanning", flush=True,
            )
        elif event == "waiting":
            print(f"[GPU {_timestamp()}] WAITING: {payload['reason']}", flush=True)
        elif event == "confirmed":
            print(f"[GPU {_timestamp()}] GPU {candidate} confirmed idle", flush=True)

        if args.state_file:
            _atomic_env(args.state_file, {
                "CANDIDATE_GPU": candidate,
                "IDLE_ELAPSED_SECONDS": elapsed,
                "SELECTOR_EVENT": event,
                "SELECTOR_LAST_UPDATE": _timestamp(),
            })

    try:
        selected = select_idle_gpu(
            probe,
            probe.compute_pids,
            cuda_visible_devices=args.visible_devices,
            confirm_seconds=args.confirm_seconds,
            poll_seconds=args.poll_seconds,
            max_util=args.max_util,
            max_used_mb=args.max_used_mb,
            on_event=on_event,
        )
    except (GPUSelectionError, ValueError) as exc:
        print(f"[GPU {_timestamp()}] FATAL: {exc}", file=sys.stderr, flush=True)
        return 2

    _atomic_env(args.result_file, {
        "PHYSICAL_GPU_ID": selected.stat.index,
        "GPU_NAME": selected.stat.name,
        "GPU_USED_MB": f"{selected.stat.used_mb:g}",
        "GPU_UTIL": f"{selected.stat.utilization:g}",
        "BASELINE_COMPUTE_PIDS": ",".join(map(str, selected.baseline_pids)),
        "GPU_CONFIRM_OBSERVATIONS": selected.observations,
        "GPU_CONFIRMED_AT": _timestamp(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
