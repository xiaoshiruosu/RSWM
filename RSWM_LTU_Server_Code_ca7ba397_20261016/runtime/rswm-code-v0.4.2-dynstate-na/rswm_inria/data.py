"""Pinned metadata selection and exact 5000 -> 4096 -> 2048 world conversion."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import VERSION, WORLD, file_hash, digest, rgb

REPO = "blanchon/INRIA-Aerial-Image-Labeling"
INRIA_REVISION = "ad8315c82c8a3021b93a7d52922d19ea76201f60"
PATTERN = re.compile(r"data/(train|test)/images/([A-Za-z-]+)(\d+)\.tif\Z")
ROLES = {"austin":("train",20),"chicago":("train",20),"vienna":("train",20),
         "kitsap":("dev",12),"bellingham":("metric_calibration",12),
         "bloomington":("metric_validation",12),"sfo":("test_unseen_city",12),
         "innsbruck":("test_unseen_city",12)}
ALIASES = {"sanfrancisco":"sfo"}


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n"
    # Build products are immutable: identical reruns are safe, changed content must use a new location.
    if p.exists() and p.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"Refusing to overwrite different artifact: {p}")
    p.write_text(text,encoding="utf-8")


def freeze_metadata(revision=INRIA_REVISION):
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An explicit full immutable commit SHA is required.")
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(REPO, revision=revision, files_metadata=True)
    if not re.fullmatch(r"[0-9a-f]{40}", info.sha):
        raise ValueError("Hub did not provide a full commit SHA.")
    if info.sha != revision:
        raise ValueError("Hub source identity differs from the requested immutable revision.")
    rows = []
    for item in info.siblings:
        lfs = item.lfs
        checksum = lfs.get("sha256") if isinstance(lfs,dict) else getattr(lfs,"sha256",None)
        rows.append({"path":item.rfilename,"size_bytes":item.size,"lfs_sha256":checksum})
    return {"repo_id":REPO,"revision":info.sha,"files":rows}


def select_manifest(inventory):
    revision = inventory["revision"]
    if inventory.get("repo_id") != REPO or not re.fullmatch(r"[0-9a-f]{40}",revision):
        raise ValueError("Expected the Inria mirror and a full immutable commit SHA.")
    groups = {city:[] for city in ROLES}; seen = set()
    for item in inventory["files"]:
        path = item["path"]
        if path in seen:
            raise ValueError("Duplicate source path in inventory.")
        seen.add(path)
        match = PATTERN.fullmatch(path)
        if not match:
            continue
        source_split, raw_city, tile = match.groups()
        city = ALIASES.get(raw_city,raw_city)
        if city not in groups:
            continue
        expected = "train" if city in ("austin","chicago","vienna","kitsap") else "test"
        if source_split != expected:
            raise ValueError(f"Unexpected source split for {city}; review and version the schema.")
        size = item.get("size_bytes"); checksum = item.get("lfs_sha256")
        if type(size) is not int or size <= 0:
            raise ValueError("Exact positive byte sizes are required before freezing.")
        if checksum is not None and not re.fullmatch(r"[0-9a-f]{64}",checksum):
            raise ValueError("Invalid LFS SHA256.")
        key = hashlib.sha256((VERSION+"/"+path).encode()).hexdigest()
        groups[city].append({"path":path,"source_split":source_split,"city":city,
                             "size_bytes":size,"lfs_sha256":checksum,"selection_hash":key,
                             "world_id":"inria-"+digest([REPO,revision,path])[:16]})
    selected = []
    for city,(role,k) in ROLES.items():
        candidates = sorted(groups[city],key=lambda r:(r["selection_hash"],r["path"]))
        if len(candidates) < k:
            raise ValueError(f"{city}: need {k} RGB paths, found {len(candidates)}.")
        for rank,item in enumerate(candidates[:k],1):
            selected.append(dict(item,role=role,selection_rank=rank))
    hashes = [r["lfs_sha256"] for r in selected if r["lfs_sha256"]]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Selected source files contain identical LFS content; audit rather than replace silently.")
    result = {"version":VERSION,"repo_id":REPO,"revision":revision,"worlds":selected,
              "total_source_bytes":sum(r["size_bytes"] for r in selected),
              "license_status":"unresolved: mirror declares unknown; verify original terms",
              "aliases":ALIASES,"selection_namespace":VERSION+"/"}
    result["manifest_sha256"] = digest(result)
    return result


def verify_manifest(manifest, inventory=None):
    """Validate schema and selection; an unkeyed digest is NOT authentication.

    With a complete frozen inventory, also verify the actual city top-k selection.
    Without it, only structure and ordering within the selected set are verifiable.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a mapping.")
    value = dict(manifest); expected = value.pop("manifest_sha256", None)
    if not isinstance(expected, str) or digest(value) != expected:
        raise ValueError("Manifest checksum mismatch.")
    rows = manifest.get("worlds")
    if not isinstance(rows, list) or len(rows) != 120:
        raise ValueError("Expected 120 selected worlds.")
    try:
        selected_inventory = {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
                              "files": [{key: row[key] for key in ("path", "size_bytes", "lfs_sha256")}
                                        for row in rows]}
        rebuilt = select_manifest(selected_inventory)
        if rebuilt != manifest:
            raise ValueError("Manifest violates frozen paths, roles, quotas, ranks, IDs or totals.")
        if inventory is not None and select_manifest(inventory) != manifest:
            raise ValueError("Manifest is not the top-k selection of the supplied complete inventory.")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed manifest schema.") from exc
    return {"structure_verified": True,
            "top_k_against_inventory_verified": inventory is not None,
            "source_bytes_verified": False,
            "authenticity_verified": False}


def download_sources(manifest, directory, limit=None):
    verify_manifest(manifest)
    if limit is not None and (type(limit) is not int or not 0 <= limit <= len(manifest["worlds"])):
        raise ValueError("limit must be an integer from 0 to the manifest length, or None.")
    if limit == 0:
        return []
    from huggingface_hub import hf_hub_download
    output = []
    for row in manifest["worlds"][:limit]:
        path = hf_hub_download(manifest["repo_id"],row["path"],repo_type="dataset",
                               revision=manifest["revision"],local_dir=directory)
        validate_source_file(path,row); output.append(str(path))
    return output


def validate_source_file(path,row):
    if Path(path).stat().st_size != row["size_bytes"]:
        raise ValueError("Source byte-size mismatch.")
    checksum = file_hash(path)
    if row.get("lfs_sha256") and checksum != row["lfs_sha256"]:
        raise ValueError("Source SHA256 mismatch.")
    return checksum


def source_pixels(path, row):
    """Validate the original TIFF container before allowing Pillow decoding.

    TIFF tag values are inspected before decoded uint8 pixels: a decoder's RGB
    conversion must not hide extra samples, signed/float data or higher bit depth.
    This does not alter the frozen crop or rounding rule.
    """
    if not isinstance(row.get("lfs_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["lfs_sha256"]):
        raise ValueError("Source pixels require a previously frozen full SHA256.")
    checksum = validate_source_file(path, row)
    try:
        with Image.open(path) as im:
            if im.format != "TIFF" or getattr(im, "n_frames", 1) != 1:
                raise ValueError("Expected a real single-page TIFF source.")
            tags = im.tag_v2
            bits = tags.get(258)
            bits = tuple(bits) if isinstance(bits, (tuple, list)) else (bits,)
            samples = tags.get(277)
            sample_format = tags.get(339, (1, 1, 1))
            sample_format = (tuple(sample_format) if isinstance(sample_format, (tuple, list)) else (sample_format,))
            if bits != (8, 8, 8) or samples != 3 or any(x != 1 for x in sample_format):
                raise ValueError("Expected three unsigned 8-bit TIFF RGB samples.")
            if tags.get(262) != 2 or tags.get(338) is not None:
                raise ValueError("Expected RGB photometric TIFF without extra samples.")
            if im.mode != "RGB" or im.size != (5000, 5000):
                raise ValueError("Expected original 5000x5000 RGB; no conversion is allowed.")
            geotags = {str(k): repr(v) for k, v in tags.items()
                       if k in (33550, 33922, 34264, 34735, 34736, 34737)}
            a = np.asarray(im).copy()
            rgb(a, (5000, 5000, 3))
    except (OSError, EOFError, SyntaxError) as exc:
        raise ValueError("Source TIFF could not be completely decoded.") from exc
    return a, checksum, geotags


def downsample2(image):
    a = rgb(image, image.shape)
    h,w,c = a.shape
    if c != 3 or h%2 or w%2:
        raise ValueError("2x2 average requires even RGB dimensions.")
    sums = a.reshape(h//2,2,w//2,2,3).sum(axis=(1,3),dtype=np.uint16)
    return ((sums+2)//4).astype(np.uint8)


def build_world(path, row, out_dir):
    a, checksum, tags = source_pixels(path, row)
    world = downsample2(a[452:4548,452:4548])
    root = Path(out_dir); root.mkdir(parents=True,exist_ok=True)
    target = root/(row["world_id"]+".png")
    if target.exists():
        with Image.open(target) as old:
            if old.mode != "RGB" or not np.array_equal(np.asarray(old),world):
                raise FileExistsError("Refusing to replace a different world.")
    else:
        from .artifact_io import save_png_exclusive
        save_png_exclusive(target,world,compress_level=6)
    record = {**row,"source_path":row.get("path"),"source_sha256":checksum,"world_pixel_sha256":hashlib.sha256(world.tobytes()).hexdigest(),
              "world_file_sha256":file_hash(target),"shape":[WORLD,WORLD,3],
              "source_crop_xyxy":[452,452,4548,4548],"resampling":"uint16 sum 2x2; (sum+2)//4",
              "nominal_gsd_m":.6,"nominal_extent_m":[1228.8,1228.8],
              "source_geotiff_tags":tags,"georeference_status":"raw tags retained; not geospatially validated" if tags else "not available",
              "path":target.name}
    write_json(target.with_suffix(".json"),record)
    return record
