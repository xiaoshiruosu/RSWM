"""Bounded HF candidate inspection, separate from the frozen Inria benchmark.

Plan, download and inspect are separate operations. A metadata plan is NOT a
validated benchmark, license grant, geography split or image-quality selection.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import numpy as np
from PIL import Image
from .geometry import digest, file_hash
from .replay import load_json

PROFILES = {
    "fair1m32": {"repo_id": "blanchon/FAIR1M",
                 "revision": "a10d702ce4832c5ccf5b25896462f8b2c6b4a270",
                 "pattern": r"data/images/[0-9]+\.tif", "count": 32, "budget_bytes": 512_000_000},
    "oscd-test": {"repo_id": "blanchon/OSCD_RGB",
                  "revision": "ac2502d7cc5ca016bb1a30a6a6c2ff6d628e7c58",
                  "pattern": r"data/test-00000-of-00001\.parquet", "count": 1, "budget_bytes": 20_000_000},
}


def save_new_json(path, value):
    """Publish a new file; never silently overwrite. Interrupted writes need review."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as f:
        f.write(text)


def select_probe(inventory, profile):
    if profile not in PROFILES or not isinstance(inventory, dict):
        raise ValueError("Unknown profile or malformed inventory.")
    cfg = PROFILES[profile]
    if inventory.get("repo_id") != cfg["repo_id"] or inventory.get("revision") != cfg["revision"]:
        raise ValueError("Candidate source does not match the pinned profile.")
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ValueError("Inventory must include all source files.")
    seen, candidates = set(), []
    for item in files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Malformed source row.")
        path = item["path"]
        if path in seen:
            raise ValueError("Duplicate inventory path.")
        seen.add(path)
        if not re.fullmatch(cfg["pattern"], path):
            continue
        size, sha = item.get("size_bytes"), item.get("lfs_sha256")
        if type(size) is not int or size <= 0 or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("Exact bytes and LFS SHA256 are required for selected file types.")
        candidates.append({"path": path, "size_bytes": size, "lfs_sha256": sha,
                           "selection_hash": hashlib.sha256(("rswm-source-probe-v1/" + profile + "/" + path).encode()).hexdigest()})
    selected = sorted(candidates, key=lambda r: (r["selection_hash"], r["path"]))[:cfg["count"]]
    if len(selected) != cfg["count"]:
        raise ValueError("Insufficient candidates; no replacement source is allowed.")
    total = sum(r["size_bytes"] for r in selected)
    if total > cfg["budget_bytes"]:
        raise ValueError("Hash-selected files exceed budget; do not substitute cheaper files.")
    if profile == "oscd-test" and (total != 12121086 or selected[0]["lfs_sha256"] != "22844e3529f150724007c959788ff2f1dc99e97c8622ba374aecfafd9c9404ff"):
        raise ValueError("OSCD file disagrees with the independently read LFS pointer.")
    result = {"schema": "rswm-source-probe-v1", "profile": profile,
              "repo_id": cfg["repo_id"], "revision": cfg["revision"],
              "files": selected, "total_source_bytes": total,
              "budget_bytes": cfg["budget_bytes"], "inventory_sha256": digest({**inventory, "files": sorted(files, key=lambda r: r["path"])}),
              "core_leaderboard": False, "source_pixels_validated": False,
              "rights_status": "mirror license unknown; original rights must be reviewed"}
    result["plan_sha256"] = digest(result)
    return result


def verify_probe(plan, inventory):
    if not isinstance(plan, dict) or plan != select_probe(inventory, plan.get("profile")):
        raise ValueError("Probe plan differs from the complete pinned inventory selection.")


def inspect_image(image, source_format=None):
    """Report actual headers without coercing channels or resampling pixels."""
    w, h = image.size
    valid = image.mode == "RGB" and (source_format is None or image.format == source_format)
    return {"width": w, "height": h, "mode": image.mode, "format": image.format,
            "strict_rgb": valid, "fits_1024_world": valid and min(w, h) >= 1024,
            "fits_viewport": valid and w >= 416 and h >= 240,
            "core_world_validated": False,
            "note": "Dimensions alone do not establish a CRS, GSD, orthorectification or geographic independence."}


def inspect_local(plan, inventory, source_dir):
    verify_probe(plan, inventory)
    source_dir = Path(source_dir)
    if source_dir.is_symlink() or any(p.is_symlink() for p in source_dir.parents):
        raise ValueError("Source directory symlinks are unsupported.")
    root = source_dir.resolve()
    result = []
    for row in plan["files"]:
        path = root.joinpath(*PurePosixPath(row["path"]).parts)
        if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.resolve().is_relative_to(root) or not path.is_file():
            raise ValueError("Missing file, symlink or path escape.")
        if path.stat().st_size != row["size_bytes"] or file_hash(path) != row["lfs_sha256"]:
            raise ValueError("File bytes disagree with pinned metadata.")
        if path.suffix == ".tif":
            with Image.open(path) as image:
                result.append({"path": row["path"], **inspect_image(image, "TIFF")})
        else:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(path)
            if pf.metadata.num_rows != 10 or "image1" not in pf.schema_arrow.names:
                raise ValueError("Expected ten OSCD test rows with image1.")
            offset = 0
            for batch in pf.iter_batches(batch_size=2, columns=["image1"]):
                for item in batch.to_pylist():
                    encoded = item["image1"]
                    if not isinstance(encoded, dict) or not isinstance(encoded.get("bytes"), bytes):
                        raise ValueError("Expected embedded image1 bytes; do not follow external image paths.")
                    with Image.open(io.BytesIO(encoded["bytes"])) as image:
                        result.append({"path": row["path"], "row_index": offset, **inspect_image(image)})
                    offset += 1
    return {"schema": "rswm-source-inspection-v1", "profile": plan["profile"],
            "plan_sha256": plan["plan_sha256"], "rows": result,
            "benchmark_materialized": False, "real_pixels_decoded": False,
            "inspection_scope": "file hash and image headers; not full pixel or geography validation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--profile", choices=sorted(PROFILES), required=True)
    p.add_argument("--inventory", type=Path)
    p.add_argument("--out", type=Path, required=True)
    for name in ("download", "inspect"):
        p = sub.add_parser(name)
        p.add_argument("--plan", type=Path, required=True)
        p.add_argument("--inventory", type=Path, required=True)
        p.add_argument("--source-dir", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Output already exists.")
    if args.command == "plan":
        if args.inventory:
            inventory = load_json(args.inventory.read_text(encoding="utf-8"))
        else:
            from huggingface_hub import HfApi
            cfg = PROFILES[args.profile]
            info = HfApi().dataset_info(cfg["repo_id"], revision=cfg["revision"], files_metadata=True)
            files = []
            for item in info.siblings:
                lfs = item.lfs
                sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
                files.append({"path": item.rfilename, "size_bytes": item.size, "lfs_sha256": sha})
            inventory = {"repo_id": cfg["repo_id"], "revision": info.sha, "files": files}
        plan = select_probe(inventory, args.profile)
        save_new_json(args.out.with_name(args.out.stem + "-inventory.json"), inventory)
        save_new_json(args.out, plan)
    else:
        plan = load_json(args.plan.read_text(encoding="utf-8"))
        inventory = load_json(args.inventory.read_text(encoding="utf-8"))
        verify_probe(plan, inventory)
        if args.command == "download":
            from huggingface_hub import hf_hub_download
            for row in plan["files"]:
                local = Path(hf_hub_download(repo_id=plan["repo_id"], filename=row["path"],
                                            revision=plan["revision"], repo_type="dataset", local_dir=args.source_dir))
                if local.stat().st_size != row["size_bytes"] or file_hash(local) != row["lfs_sha256"]:
                    raise ValueError("Downloaded file failed byte/hash validation.")
            result = {"schema": "rswm-probe-download-v1", "plan_sha256": plan["plan_sha256"],
                      "downloaded_files": len(plan["files"]), "source_bytes": plan["total_source_bytes"],
                      "benchmark_materialized": False}
        else:
            result = inspect_local(plan, inventory, args.source_dir)
        save_new_json(args.out, result)


if __name__ == "__main__":
    main()
