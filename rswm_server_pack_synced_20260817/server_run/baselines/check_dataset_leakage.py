#!/usr/bin/env python3
"""Audit exact and near-duplicate leakage across frozen dataset manifests."""
from __future__ import annotations

import argparse
import json
import itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def audit(paths: list[Path], near_threshold: int) -> dict[str, Any]:
    records = []
    manifests = []
    for path in paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        manifests.append({
            "path": str(path.resolve()),
            "source_dataset": doc.get("source_dataset"),
            "source_split": doc.get("source_split"),
            "manifest_checksum": doc.get("manifest_checksum"),
            "dataset_checksum": doc.get("dataset_checksum"),
            "split_checksum": doc.get("split_checksum"),
            "resize_crop_protocol": doc.get("resize_crop_protocol"),
        })
        for sample in doc.get("samples", []):
            records.append({
                "manifest": str(path.resolve()),
                "dataset": sample.get("source_dataset", doc.get("source_dataset")),
                "split": sample.get("source_split", doc.get("source_split")),
                "sample_id": sample.get("sample_id"),
                "source_image": sample.get("source_image"),
                "category": sample.get("source_category"),
                "size": sample.get("source_image_size"),
                "sha256": sample.get("condition_checksum"),
                "dhash": sample.get("condition_dhash"),
            })
    if not records:
        raise ValueError("no manifest samples")
    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["sha256"]:
            exact_groups[str(record["sha256"])].append(record)
        if record["source_image"]:
            source_groups[str(record["source_image"])].append(record)

    def cross_group(values):
        return [group for group in values if len(group) > 1 and
                len({(r["dataset"], r["split"]) for r in group}) > 1]

    exact = cross_group(exact_groups.values())
    source = cross_group(source_groups.values())
    near = []
    # Compare only across dataset/split boundaries and omit exact duplicates already
    # reported.  This is O(N^2), intended for benchmark-scale manifests; callers can
    # partition very large training manifests before running.
    for left, right in itertools.combinations(records, 2):
        if (left["dataset"], left["split"]) == (right["dataset"], right["split"]):
            continue
        if left["sha256"] and left["sha256"] == right["sha256"]:
            continue
        if not left["dhash"] or not right["dhash"]:
            continue
        distance = _hamming_hex(str(left["dhash"]), str(right["dhash"]))
        if distance <= near_threshold:
            near.append({"left": left, "right": right, "dhash_distance": distance})
    category_sets = defaultdict(set)
    size_counts = defaultdict(Counter)
    for record in records:
        key = f"{record['dataset']}:{record['split']}"
        if record["category"] is not None:
            category_sets[key].add(str(record["category"]))
        if record["size"] is not None:
            size_counts[key][str(record["size"])] += 1
    return {
        "schema": "rsroll.dataset_leakage_audit/v1",
        "manifests": manifests,
        "n_samples": len(records),
        "near_duplicate_dhash_threshold": near_threshold,
        "exact_cross_split_duplicates": exact,
        "near_cross_split_duplicates": near,
        "source_path_cross_split_collisions": source,
        "category_sets": {k: sorted(v) for k, v in category_sets.items()},
        "image_size_counts": {k: dict(v) for k, v in size_counts.items()},
        "pass": not exact and not near and not source,
        "claim_guard": (
            "dHash is a screening proxy, not proof of semantic identity; every flagged "
            "pair requires visual/manual review before exclusion"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--near-threshold", type=int, default=5)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.near_threshold <= 64:
        parser.error("--near-threshold must be in [0,64]")
    result = audit([Path(p) for p in args.manifests], args.near_threshold)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    state = "PASS" if result["pass"] else "FAIL"
    print(f"[{state}] exact={len(result['exact_cross_split_duplicates'])} "
          f"near={len(result['near_cross_split_duplicates'])} "
          f"source={len(result['source_path_cross_split_collisions'])} -> {output}")
    return 0 if result["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

