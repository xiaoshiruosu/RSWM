#!/usr/bin/env python3
"""Read-only preflight for the shared 4×A800 server.

It reports risks and versions. It never selects a GPU, kills a process, clears
shared memory, downloads a model, or writes a cache.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


EXPECTED = {
    "torch": "2.4.0",
    "torchvision": "0.19.0",
    "flash_attn": "2.7.3",
    "transformers": "4.51.3",
    "tokenizers": "0.21.4",
    "diffusers": "0.39.0",
    "accelerate": "1.14.0",
    "peft": "0.19.1",
    "numpy": "1.26.4",
    "huggingface_hub": "0.36.2",
}


def source_safety_scan(root: Path):
    """Find exact device/all-GPU literals; never modifies the source tree."""
    forbidden = (
        "cuda:" + "0",
        "CUDA_VISIBLE_DEVICES=" + "0,1,2,3",
        "--nproc_per_node=" + "4",
        "--nproc-per-node=" + "4",
        "torch.cuda.device_count() == " + "4",
    )
    matches = []
    extensions = {".py", ".sh", ".yaml", ".yml", ".json"}
    this_file = Path(__file__).resolve()
    for path in root.rglob("*"):
        if (not path.is_file() or path.suffix.lower() not in extensions
                or path.resolve() == this_file or "patches" in path.parts
                or ".git" in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        for number, line in enumerate(lines, 1):
            for literal in forbidden:
                if literal in line:
                    matches.append({
                        "file": str(path.relative_to(root)),
                        "line": number, "literal": literal,
                    })
    return matches


def dist_version(import_name: str) -> str | None:
    dist_name = "flash-attn" if import_name == "flash_attn" else import_name
    try:
        return md.version(dist_name)
    except md.PackageNotFoundError:
        return None


def base_version(value: str | None) -> str | None:
    if value is None:
        return None
    return value.split("+")[0]


def nvidia_smi_rows():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], str(exc)
    if cp.returncode != 0:
        return [], (cp.stderr or cp.stdout).strip()
    rows = []
    for line in cp.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 6:
            rows.append({
                "physical_index": parts[0], "name": parts[1],
                "total_mib": parts[2], "free_mib": parts[3],
                "compute_capability": parts[4],
                "driver_version": parts[5],
            })
    return rows, None


def gpu_free_memory_passes(free_mib, min_free_gib):
    """Pure comparison used by the runner preflight and CPU-only mocks."""
    return float(free_mib) / 1024 >= float(min_free_gib)


def disk_report(path: str):
    p = Path(path)
    if not p.exists():
        return {"path": path, "exists": False}
    try:
        usage = shutil.disk_usage(p)
        return {
            "path": path, "exists": True,
            "total_gib": round(usage.total / 1024 ** 3, 2),
            "free_gib": round(usage.free / 1024 ** 3, 2),
            "used_pct": round(100 * usage.used / usage.total, 2),
        }
    except OSError as exc:
        return {"path": path, "exists": True, "error": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="version/path/GPU mismatches make the command fail")
    ap.add_argument("--require-single-visible", action="store_true")
    ap.add_argument("--min-free-gib", type=float, default=0.0,
                    help="checked only when exactly one numeric GPU is visible")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    warnings, failures = [], []
    if platform.python_version() != "3.10.20":
        failures.append(
            f"python: expected 3.10.20, got {platform.python_version()}")
    versions = {name: dist_version(name) for name in EXPECTED}
    for name, expected in EXPECTED.items():
        actual = versions[name]
        if base_version(actual) != expected:
            failures.append(f"{name}: expected {expected}, got {actual}")

    torch_info = {"import_ok": False}
    visible = None
    try:
        import torch
        torch_info.update({
            "import_ok": True,
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "visible_gpu_count": torch.cuda.device_count(),
        })
        visible = torch.cuda.device_count()
        torch_info["visible_gpus"] = []
        for i in range(visible):
            props = torch.cuda.get_device_properties(i)
            free_bytes, _ = torch.cuda.mem_get_info(i)
            torch_info["visible_gpus"].append({
                "logical_index": i,
                "name": props.name,
                "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
                "total_vram_gib": round(props.total_memory / 1024 ** 3, 2),
                "free_vram_gib": round(free_bytes / 1024 ** 3, 2),
            })
        if not torch.cuda.is_available():
            failures.append("torch.cuda.is_available() is False")
        if torch.__version__ != "2.4.0+cu121":
            failures.append(
                f"torch build: expected 2.4.0+cu121, got {torch.__version__}")
        if torch.version.cuda != "12.1":
            failures.append(
                f"torch CUDA runtime: expected 12.1, got {torch.version.cuda}")
    except Exception as exc:
        torch_info["error"] = f"{type(exc).__name__}: {exc}"
        failures.append(f"torch/CUDA import failed: {exc}")

    try:
        import torchvision
        torch_info["torchvision_version"] = torchvision.__version__
        if torchvision.__version__ != "0.19.0+cu121":
            failures.append(
                "torchvision build: expected 0.19.0+cu121, got "
                f"{torchvision.__version__}")
    except Exception as exc:
        torch_info["torchvision_error"] = f"{type(exc).__name__}: {exc}"
        failures.append(f"torchvision import failed: {exc}")

    if args.require_single_visible and visible != 1:
        failures.append(f"single-GPU smoke requires exactly 1 visible GPU; got {visible}")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        warnings.append("CUDA_VISIBLE_DEVICES is unset on a shared server")

    smi_rows, smi_error = nvidia_smi_rows()
    if smi_error:
        warnings.append(f"nvidia-smi unavailable: {smi_error}")
    else:
        drivers = sorted({row["driver_version"] for row in smi_rows})
        if drivers != ["535.261.03"]:
            warnings.append(
                "node driver differs from verified 535.261.03: "
                f"{drivers}; re-run all smoke tests before experiments")
        if len(smi_rows) != 4 or any(
                "A800" not in row["name"] or row["total_mib"] not in
                ("81920", "81919") for row in smi_rows):
            warnings.append(
                "physical node layout differs from verified 4×A800-SXM4-80GB")
    if args.min_free_gib and cvd and cvd.strip().isdigit() and "," not in cvd:
        row = next((r for r in smi_rows if r["physical_index"] == cvd.strip()), None)
        if row is None:
            failures.append("cannot map CUDA_VISIBLE_DEVICES to nvidia-smi for free-VRAM check")
        elif not gpu_free_memory_passes(row["free_mib"], args.min_free_gib):
            failures.append(
                f"physical GPU {cvd} has {float(row['free_mib']) / 1024:.1f}GiB free; "
                f"requires {args.min_free_gib:.1f}GiB")

    env_keys = [
        "RSWM_BASE", "TMPDIR", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME",
        "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
        "PIP_CACHE_DIR", "CONDA_PKGS_DIRS", "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK", "RANK", "WORLD_SIZE", "RSWM_HARDWARE_PROFILE",
        "RSWM_ATTENTION_BACKEND", "PYTORCH_CUDA_ALLOC_CONF", "OFFLOAD",
        "RSWM_OFFLOAD_RESOLVED", "TORCH_CUDA_ARCH_LIST",
    ]
    env = {k: os.environ.get(k) for k in env_keys}
    safe_root = Path(os.environ.get("RSWM_BASE", "/nix/rsroll_25828"))
    for key in (
            "TMPDIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME",
            "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "PIP_CACHE_DIR",
            "CONDA_PKGS_DIRS"):
        value = env.get(key)
        if not value:
            failures.append(f"{key} is unset")
            continue
        try:
            Path(value).resolve().relative_to(safe_root.resolve())
        except (OSError, ValueError):
            failures.append(f"{key}={value} is outside {safe_root}")

    tmp_value = env.get("TMPDIR") or ""
    if tmp_value in ("/tmp", "/dev/shm") or tmp_value.startswith("/root"):
        failures.append(f"unsafe TMPDIR={tmp_value}")

    fp8_env = {k: v for k, v in os.environ.items()
               if ("FP8" in k.upper() or "FLOAT8" in k.upper())
               and v not in ("", "0", "false", "False")}
    if fp8_env:
        failures.append(f"FP8-related environment is enabled: {fp8_env}")

    hopper_env = {k: v for k, v in os.environ.items()
                  if ("SM90" in k.upper() or "COMPUTE_90" in str(v).upper()
                      or (k == "TORCH_CUDA_ARCH_LIST"
                          and any(x in str(v).lower() for x in ("9.0", "sm90"))))
                  and v not in ("", "0", "false", "False")}

    backend = os.environ.get("RSWM_ATTENTION_BACKEND", "auto")
    if backend not in ("auto", "flash_attn", "sdpa"):
        failures.append(f"invalid RSWM_ATTENTION_BACKEND={backend}")

    profile = os.environ.get("RSWM_HARDWARE_PROFILE", "a800_80g")
    if profile == "a800_80g":
        for gpu in torch_info.get("visible_gpus", []):
            if ("A800" not in gpu["name"]
                    or gpu["compute_capability"] != "8.0"
                    or not 75 <= gpu["total_vram_gib"] < 90):
                failures.append(
                    "a800_80g mismatch: "
                    f"{gpu['name']} sm={gpu['compute_capability']} "
                    f"VRAM={gpu['total_vram_gib']}GiB")
        if hopper_env:
            failures.append(
                f"A800 profile has sm90/Hopper-only environment enabled: {hopper_env}")

    offload_requested = os.environ.get("OFFLOAD", "auto").strip().lower()
    if offload_requested not in ("auto", "0", "1"):
        failures.append(
            f"invalid OFFLOAD={offload_requested!r}; expected auto, 0, or 1")
        offload_resolved = None
    elif offload_requested == "auto":
        gpu0 = (torch_info.get("visible_gpus") or [None])[0]
        offload_resolved = "0" if (
            gpu0 is not None
            and gpu0["total_vram_gib"] * 1024 >= 97000
            and gpu0["free_vram_gib"] * 1024 >= 90000) else "1"
    else:
        offload_resolved = offload_requested
    if profile == "a800_80g" and offload_resolved == "0":
        warnings.append(
            "A800 OFFLOAD=0 is an explicit throughput/OOM-boundary test; "
            "do not use it for the formal matrix before real 33-frame smoke passes")

    visible_caps = [g["compute_capability"]
                    for g in torch_info.get("visible_gpus", [])]
    if any(cap.startswith("8.") for cap in visible_caps):
        attention_policy = "flash_attn_2 -> sdpa"
    elif any(cap.startswith("9.") for cap in visible_caps):
        attention_policy = "flash_attn_3 -> flash_attn_2 -> sdpa"
    else:
        attention_policy = "sdpa (no supported visible CUDA capability detected)"

    import_checks = {}
    for name in ("flash_attn", "transformers", "diffusers", "accelerate", "peft"):
        try:
            importlib.import_module(name)
            import_checks[name] = "OK"
        except Exception as exc:
            import_checks[name] = f"FAIL: {type(exc).__name__}: {exc}"
            if name == "flash_attn" and backend in ("auto", "sdpa"):
                warnings.append(f"flash_attn import failed; SDPA fallback remains available: {exc}")
                if args.strict:
                    failures.append(
                        "verified formal stack requires flash_attn 2.7.3 to import; "
                        f"got: {exc}")
            else:
                failures.append(f"{name} import failed: {exc}")

    source_scan = source_safety_scan(Path(__file__).resolve().parents[1])
    if source_scan:
        failures.append(
            f"source safety scan found hard-coded device/all-GPU literals: {source_scan}")

    shm = disk_report("/dev/shm")
    if shm.get("exists") and shm.get("used_pct", 0) >= 95:
        warnings.append(f"/dev/shm is {shm['used_pct']}% used; keep num_workers=0")

    report = {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "expected_versions": EXPECTED,
        "torch": torch_info,
        "nvidia_smi": smi_rows,
        "environment": env,
        "attention_backend": backend,
        "attention_policy": attention_policy,
        "hardware_profile": profile,
        "precision": "bf16",
        "offload": {
            "requested": offload_requested,
            "resolved": offload_resolved,
            "auto_resident_thresholds_mib": {
                "total": 97000, "free": 90000,
            },
        },
        "import_checks": import_checks,
        "storage": {
            "root": disk_report("/"), "tmp": disk_report("/tmp"),
            "dev_shm": shm, "nix": disk_report("/nix"),
            "workspace": disk_report("/workspace"),
        },
        "fp8_environment": fp8_env,
        "hopper_only_environment": hopper_env,
        "source_safety_scan": source_scan,
        "warnings": warnings,
        "failures": failures,
        "result": "FAIL" if failures else "PASS",
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"A800 preflight: {report['result']}")
        print(f"hostname={report['hostname']} python={report['python']}")
        print(f"torch={versions['torch']} cuda_build={torch_info.get('cuda_build')} "
              f"cuda_available={torch_info.get('cuda_available')} visible={visible}")
        print("stack=" + " ".join(
            f"{name}={versions[name]}" for name in (
                "torchvision", "flash_attn", "transformers", "tokenizers",
                "diffusers", "accelerate", "peft", "numpy",
                "huggingface_hub")))
        print(f"CUDA_VISIBLE_DEVICES={cvd!r} attention={backend} "
              f"policy={attention_policy} precision=bf16 "
              f"offload={offload_requested}->{offload_resolved}")
        for gpu in torch_info.get("visible_gpus", []):
            print("visible GPU {logical_index}: {name}, sm={compute_capability}, "
                  "VRAM={free_vram_gib}/{total_vram_gib}GiB free/total".format(**gpu))
        for row in smi_rows:
            print(f"physical GPU {row['physical_index']}: {row['name']}, "
                  f"free={row['free_mib']}MiB/{row['total_mib']}MiB, sm={row['compute_capability']}")
        for key in (
                "TMPDIR", "HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME",
                "TRITON_CACHE_DIR", "PIP_CACHE_DIR"):
            print(f"{key}={env.get(key)!r}")
        print(f"storage /tmp={report['storage']['tmp']} /dev/shm={shm} "
              f"/nix={report['storage']['nix']}")
        print(f"source safety scan: {len(source_scan)} forbidden literal(s)")
        for msg in warnings:
            print(f"WARN: {msg}")
        for msg in failures:
            print(f"FAIL: {msg}")

    return 2 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
