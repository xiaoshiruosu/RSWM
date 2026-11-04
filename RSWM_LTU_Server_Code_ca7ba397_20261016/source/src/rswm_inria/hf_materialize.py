"""Materialize two bounded HF sources; never download a whole Hub repository.

Inria preserves the frozen pixel and task algorithms with strict source checks. LEVIR-LR64 is
an explicitly separate construction-only extension, NOT accepted by the core
runner or included in the Inria leaderboard. No pretrained inference here.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image

INRIA_REVISION = "ad8315c82c8a3021b93a7d52922d19ea76201f60"
# Fixed dev input identity; these are source bindings, not use approvals.
PILOT_DEV2_SOURCES = (
    {"path": "data/train/images/kitsap31.tif", "world_id": "inria-c5685116911b80af",
     "size_bytes": 75040353, "lfs_sha256": "458a62ab4e8f2f3ff248c3284c7e126a8c477257b60fd46a5995fb1eaf75ee0a",
     "source_split": "train", "city": "kitsap", "role": "dev", "selection_rank": 1},
    {"path": "data/train/images/kitsap19.tif", "world_id": "inria-46fecc9b5e0ee303",
     "size_bytes": 75040353, "lfs_sha256": "1581dc1994714c851c1ceb00494d5f5043638649dacf3135d53f46bfdafda981",
     "source_split": "train", "city": "kitsap", "role": "dev", "selection_rank": 2},
)
LEVIR_REPO = "blanchon/LEVIR_CDPlus"
LEVIR_REVISION = "8062826f12e4f3cd5690b84278ad67c1f42acc46"
LEVIR_STUDY = "rswm-levir100-lr64-v0.1"
SHARDS = [
    ("data/test-00000-of-00003.parquet", 439171835, "75f64aef4b9969de998d3a3454802bf21da01cd1fbe3980a191d15181f3f1353"),
    ("data/test-00001-of-00003.parquet", 439361709, "6e34db27af2b3b27825134ceba85afd504745359e70780d00ec4fd7949eab6f6"),
    ("data/test-00002-of-00003.parquet", 437594257, "763eb1135eed51b24d50f24a05c67bb11619a08a8b87c5404227548c99af90fb"),
]
W, H = 416, 240


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(canonical(value) + b"\n")


def fresh_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Output must be empty; retain failed builds and use a new directory.")
    path.mkdir(parents=True, exist_ok=True)


def save_png(path: Path, pixels: np.ndarray) -> None:
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("Expected uint8 RGB.")
    path.parent.mkdir(parents=True, exist_ok=True)
    from .artifact_io import save_png_exclusive
    save_png_exclusive(path,pixels,compress_level=6)


def interpolate(points, steps) -> np.ndarray:
    if len(points) != len(steps) + 1 or any(type(n) is not int or n <= 0 for n in steps):
        raise ValueError("Invalid segment durations.")
    out = [np.asarray(points[0], dtype=np.int64)]
    for p, q, n in zip(points[:-1], points[1:], steps):
        p, q = np.asarray(p), np.asarray(q)
        out.extend(np.rint(p+(q-p)*t/n).astype(np.int64) for t in range(1, n+1))
    return np.asarray(out, dtype=np.int64)


def geometry_labels(q, center, prefix_end=32, limit=704):
    """Independent explicit visibility computation, coordinates relative to I0."""
    a = np.asarray(q)
    if a.ndim != 2 or a.shape[1] != 2 or a.dtype.kind not in "iu" or len(a) <= prefix_end + 1:
        raise ValueError("Expected a complete integer trajectory.")
    if np.any(a > limit) or (a.dtype.kind == "i" and np.any(a < -limit)) or np.any(a[0] != 0):
        raise ValueError("Illegal relative coordinates.")
    a = a.astype(np.int64)
    dy, dx = np.mgrid[-64:64, -64:64]
    xx, yy = dx+int(center[0]), dy+int(center[1])
    visible = np.stack([(xx >= x-W//2) & (xx < x+W//2) &
                        (yy >= y-H//2) & (yy < y+H//2) for x, y in a])
    pre, post = visible[:prefix_end+1], visible[prefix_end+1:]
    if visible[0].any() or visible[prefix_end].any() or not pre.any(0).all() or not post.any(0).all():
        raise ValueError("Target is not completely new/absent/reentered.")
    first = pre.argmax(0).astype(np.int16)
    last = (prefix_end-pre[::-1].argmax(0)).astype(np.int16)
    reentry = (prefix_end+1+post.argmax(0)).astype(np.int16)
    gap = reentry-last-1
    delta = np.abs(np.diff(a, axis=0))
    overlap = np.maximum(0, 1-delta[:, 0]/W)*np.maximum(0, 1-delta[:, 1]/H)
    if gap.min() < 32 or overlap.min() < .65-1e-12:
        raise ValueError("Insufficient target gap or adjacent overlap.")
    labels = {"x": xx, "y": yy, "first_event": first,
              "last_prefix_visible_event": last, "reentry_event": reentry,
              "invisible_query_gap": gap,
              "fully_absent_calls": ((reentry-1)//16-(last-1)//16-1).astype(np.int16)}
    stats = {"planned_outputs": len(a)-1, "min_adjacent_overlap": float(overlap.min()),
             "path_length_px": float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum()),
             "max_radius_px": float(np.linalg.norm(a, axis=1).max()),
             "max_step_px": float(np.linalg.norm(np.diff(a, axis=0), axis=1).max())}
    for key in ("first_event", "last_prefix_visible_event", "reentry_event", "invisible_query_gap", "fully_absent_calls"):
        stats[key+"_range"] = [int(labels[key].min()), int(labels[key].max())]
    return labels, stats


def levir_paths():
    out = {}
    for name, end in (("exact", (256, 256)), ("offset", (288, 280))):
        q = interpolate([(0, 0), (256, 256), (-256, 256), (-256, -256), end], [16]*4)
        labels, stats = geometry_labels(q, (256, 256), limit=288)
        out[name] = (q, labels, stats)
    return out


def export_examples(root: Path) -> None:
    fresh_root(root)
    paths = {
        "F5-exact": interpolate([(0,0),(480,0),(480,480),(-480,480),(-480,-480),(480,0)], [16,16,32,16,16]),
        "F5-offset": interpolate([(0,0),(480,0),(480,480),(-480,480),(-480,-480),(544,48)], [16,16,32,16,16]),
        "F7-a": interpolate([(0,0),(480,0),(480,480),(-480,480),(-480,-480),(480,-480),(480,0)], [16]*6),
        "F7-b": interpolate([(0,0),(480,0),(480,480),(0,640),(-640,0),(0,-640),(480,0)], [16]*6),
    }
    results = {}
    for name, q in paths.items():
        labels, stats = geometry_labels(q, (480,0))
        np.savez_compressed(root/(name+".npz"), **labels)
        save_json(root/(name+".json"), {"kind":"reference_geometry_not_random_core_task", "queries":q.tolist(), "stats":stats})
        results[name] = stats
    for name, (q, labels, stats) in levir_paths().items():
        np.savez_compressed(root/("LR64-"+name+".npz"), **labels)
        save_json(root/("LR64-"+name+".json"), {"study":LEVIR_STUDY, "queries":q.tolist(), "stats":stats})
        results["LR64-"+name] = stats
    save_json(root/"geometry-summary.json", {"real_images_read":0, "model_outputs_generated":0, "results":results})


def select_profile(manifest, profile):
    if profile == "core120":
        return manifest["worlds"]
    if profile == "smoke12":
        return [r for r in manifest["worlds"] if r["city"] in ("austin","chicago","vienna","kitsap")
                and r["selection_rank"] <= 3]
    if profile == "pilot-dev2":
        from .data import REPO
        if manifest.get("repo_id") != REPO or manifest.get("revision") != INRIA_REVISION:
            raise ValueError("pilot-dev2 requires the frozen parent source revision.")
        rows = [r for r in manifest["worlds"] if r.get("role") == "dev" and r.get("selection_rank", 0) in (1, 2)]
        if len(rows) != 2 or any(any(row.get(key) != value or type(row.get(key)) is not type(value)
                                     for key, value in fixed.items())
                                 for row, fixed in zip(rows, PILOT_DEV2_SOURCES)):
            raise ValueError("pilot-dev2 requires the fixed Kitsap dev source identities and order.")
        return rows
    raise ValueError("Unknown profile.")


def build_inria(args):
    # Reuse the actual repository task and conversion code, NOT the example templates.
    from rswm_inria.data import verify_manifest, build_world, validate_source_file
    from rswm_inria.tasks import packages, validate_task, training_clips
    from rswm_inria.geometry import crop
    m = json.loads(args.manifest.read_text(encoding="utf-8"))
    inv = json.loads(args.inventory.read_text(encoding="utf-8"))
    verify_manifest(m, inv)
    return materialize_inria(m, inv, select_profile(m, args.profile), args.source_dir,
                             args.out, args.profile, args.max_source_gb, args.download)


def materialize_inria(m, inv, selected, source_dir, out, profile, max_source_gb, download=False):
    """Bounded builder; the explicitly named fixture profile is never real data."""
    from types import SimpleNamespace
    from rswm_inria.data import verify_manifest, build_world, validate_source_file
    from rswm_inria.tasks import packages, validate_task, training_clips
    from rswm_inria.geometry import crop
    verify_manifest(m, inv)
    if profile in ("smoke12", "core120", "pilot-dev2"):
        expected_count = {"core120": 120, "smoke12": 12, "pilot-dev2": 2}[profile]
        if selected != select_profile(m, profile) or len(selected) != expected_count:
            raise ValueError("Profile quotas do not match the frozen core manifest.")
    elif profile != "synthetic_fixture" or not selected or any(r not in m["worlds"] for r in selected):
        raise ValueError("Unknown profile or unbound synthetic fixture selection.")
    if len({r["world_id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate selected world.")
    if not np.isfinite(max_source_gb) or max_source_gb <= 0:
        raise ValueError("Source budget must be positive and finite.")
    args = SimpleNamespace(source_dir=Path(source_dir), out=Path(out), profile=profile,
                           max_source_gb=max_source_gb, download=download)
    total = sum(r["size_bytes"] for r in selected)
    if total > args.max_source_gb * 1_000_000_000:
        raise ValueError(f"Source payload {total} bytes exceeds explicit download budget.")
    # Offline missing/hash errors stop before creating a misleading build root.
    if not download:
        for row in selected:
            source = args.source_dir / row["path"]
            if source.is_symlink():
                raise ValueError("Source symlink is not an immutable source file.")
            validate_source_file(source, row)
    fresh_root(args.out)
    save_json(args.out/"build-selection.json", {"core_manifest_sha256":m["manifest_sha256"],
              "repo_id":m["repo_id"], "revision":m["revision"], "profile":args.profile,
              "full_core":args.profile=="core120", "source_bytes":total, "worlds":selected,
              "input_kind":"synthetic_fixture" if profile=="synthetic_fixture" else "source_identity_checked_geography_unknown"})
    index, worlds, clips = [], [], []
    for row in selected:
        source = args.source_dir / row["path"]
        if not source.exists() and args.download:
            from huggingface_hub import hf_hub_download
            source = Path(hf_hub_download(m["repo_id"], filename=row["path"], repo_type="dataset",
                                         revision=m["revision"], local_dir=args.source_dir))
        validate_source_file(source, row)
        record = build_world(source, row, args.out/"private"/"worlds")
        worlds.append(record)
        with Image.open(args.out/"private"/"worlds"/record["path"]) as im:
            world = np.asarray(im).copy()
        for oi in range(4):
            bundle = packages(row["world_id"], row["role"], oi)
            initial_id = digest([row["world_id"], oi, "initial"])[:24]
            save_png(args.out/"public"/"initial"/(initial_id+".png"), crop(world, (0,0), bundle[0][0]["origin"]))
            for task, labels in bundle:
                validate_task(task, labels)
                label_files = {}
                for name, arrays in labels.items():
                    p = args.out/"private"/"tasks"/"labels"/(task["task_id"]+"-"+name+".npz")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(p, **arrays)
                    label_files[name] = {"path":"labels/"+p.name, "sha256":file_hash(p)}
                task["private_label_files"] = label_files
                p = args.out/"private"/"tasks"/(task["task_id"]+".json")
                save_json(p, task)
                sid = digest([task["task_id"], "public-session"])[:24]
                public = {"session_id":sid, "initial_image":"../initial/"+initial_id+".png",
                          "viewport_hw":[H,W], "bounds_xy":[[-704,704],[-704,704]],
                          "coordinate_normalizer":float(np.hypot(W,H)), "q0":[0,0],
                          "protocol":task["protocol"], "reset_between_streams":task["family"]=="F3"}
                if task["protocol"] == "offline":
                    public["programs"] = task["programs"]
                else:
                    public["query_delivery"] = "evaluator sends current 16-query call only"
                save_json(args.out/"public"/"sessions"/(sid+".json"), public)
                index.append({"task_id":task["task_id"], "session_id":sid, "family":task["family"],
                              "world_id":row["world_id"], "role":row["role"], "city":row["city"],
                              "task_path":"tasks/"+p.name, "world_path":"worlds/"+record["path"]})
        if row["role"] == "train":
            clips.extend(training_clips(row["world_id"], row["role"]))
    save_json(args.out/"private"/"index.json", {"tasks":index,"worlds":worlds,"source_manifest_sha256":m["manifest_sha256"]})
    save_json(args.out/"private"/"training-clips.json", {"clips":clips,"clip_count":len(clips),"view_events":33*len(clips)})
    save_json(args.out/"BUILD_COMPLETE.json", {"profile":args.profile, "source_bytes":total, "worlds":len(worlds),
              "initial_images":4*len(worlds),"task_packages":len(index), "query_outputs_all_roles":len(worlds)*4*352,
              "real_model_inference":False,"public_export_is_security_sandbox":False})


def choose_levir_rows(counts, n=100):
    if not counts or any(type(v) is not int or v < 0 for v in counts.values()) or type(n) is not int or n < 1:
        raise ValueError("Invalid row counts.")
    candidates = [(hashlib.sha256(f'{LEVIR_STUDY}/{path}/{i}'.encode()).hexdigest(), path, i)
                  for path, count in counts.items() for i in range(count)]
    if len(candidates) < n:
        raise ValueError("Not enough rows; no replacement sampling.")
    return sorted(candidates)[:n]


def decode_image1(cell):
    if not isinstance(cell, dict) or not isinstance(cell.get("bytes"), bytes):
        raise ValueError("Expected embedded image1 bytes; refuse external paths or guessed field names.")
    raw = cell["bytes"]
    with Image.open(io.BytesIO(raw)) as im:
        if im.size != (1024,1024) or im.mode != "RGB":
            raise ValueError("Expected unresized 1024-square RGB image1.")
        pixels = np.asarray(im).copy()
    if pixels.dtype != np.uint8:
        raise ValueError("Expected uint8 source pixels.")
    return pixels, hashlib.sha256(raw).hexdigest(), cell.get("path")


def build_levir(args):
    import pyarrow.parquet as pq
    if sum(s[1] for s in SHARDS) > args.max_source_gb * 1_000_000_000:
        raise ValueError("Explicit source budget too small.")
    files, counts = {}, {}
    for name, size, checksum in SHARDS:
        path = args.source_dir/name
        if not path.exists() and args.download:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(LEVIR_REPO, filename=name, repo_type="dataset",
                                       revision=LEVIR_REVISION, local_dir=args.source_dir))
        if path.stat().st_size != size or file_hash(path) != checksum:
            raise ValueError("Pinned LEVIR shard size/hash mismatch.")
        pf = pq.ParquetFile(path)
        if set(pf.schema_arrow.names) != {"image1","image2","mask"}:
            raise ValueError("Unexpected source schema; review before changing the build version.")
        files[name], counts[name] = pf, pf.metadata.num_rows
    if sum(counts.values()) != 348:
        raise ValueError("Pinned test split must contain 348 source pairs.")
    selected = choose_levir_rows(counts)
    rank = {(p,i):(k,h) for k,(h,p,i) in enumerate(selected,1)}
    fresh_root(args.out)
    save_json(args.out/"selection.json", {"study":LEVIR_STUDY,"repo_id":LEVIR_REPO,"revision":LEVIR_REVISION,
              "shards":[{"path":p,"size_bytes":b,"sha256":h} for p,b,h in SHARDS],
              "row_counts":counts,"selected_rows":[{"rank":k,"hash":h,"shard":p,"row_index":i}
                                                     for k,(h,p,i) in enumerate(selected,1)]})
    index, pixel_hashes = [], set()
    paths = levir_paths()
    for path in sorted(files):
        offset = 0
        for batch in files[path].iter_batches(batch_size=8, columns=["image1"]):
            for j, row in enumerate(batch.to_pylist()):
                key = (path, offset+j)
                if key not in rank:
                    continue
                pixels, encoded_hash, original_name = decode_image1(row["image1"])
                ph = hashlib.sha256(pixels.tobytes()).hexdigest()
                if ph in pixel_hashes:
                    raise ValueError("Duplicate selected image1 pixels; audit, do not silently replace.")
                pixel_hashes.add(ph)
                wid = "levir-"+digest([LEVIR_REPO,LEVIR_REVISION,path,offset+j])[:16]
                save_png(args.out/"private"/"worlds"/(wid+".png"), pixels)
                iid = digest([wid,"initial"])[:24]
                save_png(args.out/"public"/"initial"/(iid+".png"),pixels[392:632,304:720])
                record = {"world_id":wid,"shard":path,"row_index":offset+j,"selection_rank":rank[key][0],
                          "source_image_path":original_name,"image1_encoded_sha256":encoded_hash,
                          "world_pixel_sha256":ph,"nominal_gsd_m":.5,"shape":[1024,1024,3],
                          "resampling":"none","geographic_group":"unknown","source_pair_key":path+":"+str(offset+j)}
                save_json(args.out/"private"/"worlds"/(wid+".json"),record)
                for name,(q,labels,stats) in paths.items():
                    tid = wid+"-LR64-"+name
                    npz = args.out/"private"/"labels"/(tid+".npz")
                    npz.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(npz,**labels)
                    task = {"study":LEVIR_STUDY,"task_id":tid,"world_id":wid,"pair_id":wid+"-LR64",
                            "family":"LR64","variant":name,"protocol":"online16","origin":[512,512],
                            "world_hw":[1024,1024],"viewport_hw":[H,W],"query_bounds_xy":[[-288,288]]*2,
                            "q":q.tolist(),"target_center":[256,256],"target_size":128,"stats":stats,
                            "label_file":"../labels/"+npz.name,"label_sha256":file_hash(npz)}
                    save_json(args.out/"private"/"tasks"/(tid+".json"),task)
                    sid = digest([tid,"public"])[:24]
                    save_json(args.out/"public"/"sessions"/(sid+".json"),
                              {"session_id":sid,"initial_image":"../initial/"+iid+".png",
                               "protocol":"online16","viewport_hw":[H,W],"bounds_xy":[[-288,288]]*2,
                               "coordinate_normalizer":float(np.hypot(W,H)),"q0":[0,0],
                               "query_delivery":"current 16 queries only; reset between exact/offset"})
                index.append(record)
            offset += batch.num_rows
    if len(index) != 100:
        raise ValueError("Not all preselected source rows were materialized.")
    save_json(args.out/"private"/"world-index.json", sorted(index,key=lambda x:x["selection_rank"]))
    save_json(args.out/"BUILD_COMPLETE.json", {"study":LEVIR_STUDY,"worlds":100,"initial_images":100,
              "task_packages":200,"planned_outputs_per_model_seed":12800,"source_bytes":sum(s[1] for s in SHARDS),
              "same_size_as_inria":False,"core_runner_compatible":False,"model_evaluation_implemented_here":False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("examples"); p.add_argument("--out",type=Path,required=True)
    p = sub.add_parser("inria")
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--profile",choices=["smoke12","core120","pilot-dev2"],default="smoke12")
    p.add_argument("--source-dir",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--download",action="store_true");p.add_argument("--max-source-gb",type=float,default=1.0)
    p = sub.add_parser("levir")
    p.add_argument("--source-dir",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--download",action="store_true");p.add_argument("--max-source-gb",type=float,default=1.4)
    args = parser.parse_args(argv)
    if hasattr(args,"max_source_gb") and (not np.isfinite(args.max_source_gb) or args.max_source_gb <= 0):
        parser.error("--max-source-gb must be positive and finite")
    if args.command == "examples": export_examples(args.out)
    elif args.command == "inria": build_inria(args)
    else: build_levir(args)
    print(json.dumps({"command":args.command,"output":str(args.out),"status":"complete"}))


if __name__ == "__main__":
    main()
