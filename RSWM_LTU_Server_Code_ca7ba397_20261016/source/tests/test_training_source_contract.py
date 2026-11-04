"""CPU source-contract controls; no real data audit or training is claimed.

The positive bridge test deliberately mocks the real audit validator. Its disk
files are synthetic and contain duplicate zero pixels; they could never pass a
real materialized audit. These fixtures produce no real PASS receipt artifact.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
from collections import Counter
import numpy as np
import pytest
from PIL import Image
from rswm_inria.data import ROLES
from rswm_inria.geometry import digest, file_hash
from rswm_inria.tasks import packages, training_clips
from rswm_inria.training import ClipDataset
from rswm_inria import materialized


def train_index(*, cities=None):
    cities = cities or [city for city in ("austin", "chicago", "vienna") for _ in range(20)]
    worlds = [{"world_id": f"cpu-training-source-control-{i}", "city": city, "role": "train",
               "path": f"cpu-training-source-control-{i}.png", "world_file_sha256": "a"*64,
               "world_pixel_sha256": "b"*64} for i, city in enumerate(cities)]
    clips = [clip for row in worlds for clip in training_clips(row["world_id"], "train")]
    # Canonical serialized clips turn tuple origins into JSON arrays.
    clips = json.loads(json.dumps(clips))
    return {"worlds": worlds, "tasks": [], "source_manifest_sha256": "c"*64}, {
        "clips": clips, "clip_count": len(clips), "view_events": 33*len(clips)}


@pytest.fixture(scope="module")
def balanced_train_index():
    return train_index()


@pytest.mark.parametrize("cities", [
    ["austin"]*60,
    ["austin"]*20+["chicago"]*19+["vienna"]*21,
    ["austin"]*20+["chicago"]*20+["kitsap"]*20,
    ["austin"]*20+["chicago"]*20+["bellingham"]*20,
    ["austin"]*20+["chicago"]*20+["sfo"]*20,
])
def test_formal_role_train_does_not_override_fixed_city_quota(tmp_path, balanced_train_index, cities):
    index, clips = copy.deepcopy(balanced_train_index)
    for row, city in zip(index["worlds"], cities):
        row["city"] = city
    # All 60 worlds and 3840 clips still advertise role=train.
    assert all(row["role"] == "train" for row in index["worlds"])
    assert clips["clip_count"] == 3840
    with pytest.raises(ValueError, match="[Cc]ity|[Cc]ities|quota|Austin|train.*source"):
        ClipDataset(index, clips, tmp_path/"private/worlds", source_audit=None)


def test_formal_canonical_counts_still_require_a_source_audit(tmp_path, balanced_train_index):
    index, clips = copy.deepcopy(balanced_train_index)
    with pytest.raises(ValueError, match="audit|receipt|source"):
        ClipDataset(index, clips, tmp_path/"private/worlds", source_audit=None)


def test_formal_thin_pass_receipt_cannot_authorize_source(complete_disk_structure):
    root, index, clips, _hashes, _rows = complete_disk_structure
    with pytest.raises(ValueError, match="audit|receipt|source"):
        ClipDataset(index, clips, root, source_audit={
            "status": "PASS", "real_data_validation": "PASS", "source_manifest_sha256": "c"*64})


@pytest.fixture(scope="module")
def complete_disk_structure(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("training_source_cpu_structure_only")
    root = tmp_path/"private/worlds"; root.mkdir(parents=True)
    task_root = root.parent/"tasks"; task_root.mkdir()
    base_png = tmp_path/"synthetic-zero-world.png"
    world = np.zeros((2048,2048,3), np.uint8); Image.fromarray(world).save(base_png)
    pixel_hash = hashlib.sha256(world.tobytes()).hexdigest()
    rows, tasks, clips, task_hashes = [], [], [], {}
    for city, (role, n) in ROLES.items():
        for i in range(n):
            wid = f"cpu-structure-only-{city}-{i:02d}"
            target = root/(wid+".png"); shutil.copyfile(base_png, target)
            rows.append({"world_id":wid,"city":city,"role":role,"path":target.name,
                         "world_file_sha256":file_hash(target),"world_pixel_sha256":pixel_hash})
            for oi in range(4):
                for task, _ in packages(wid, role, oi):
                    # No private labels are claimed; the mock only checks source
                    # bridge plumbing. A real audit rejects this fixture.
                    task["private_label_files"] = {}
                    path = task_root/(task["task_id"]+".json")
                    path.write_text(json.dumps(task,sort_keys=True))
                    task = json.loads(path.read_text())
                    task_hashes[task["task_id"]] = digest(task)
                    tasks.append({"task_id":task["task_id"],"world_id":wid,"city":city,"role":role,
                                  "family":task["family"],"task_path":"tasks/"+path.name,
                                  "world_path":"worlds/"+target.name})
            if role == "train": clips.extend(training_clips(wid, role))
    index = {"worlds":rows,"tasks":tasks,"source_manifest_sha256":"c"*64}
    data = json.loads(json.dumps({"clips":clips,"clip_count":len(clips),"view_events":33*len(clips)}))
    (root.parent/"index.json").write_text(json.dumps(index,sort_keys=True))
    (root.parent/"training-clips.json").write_text(json.dumps(data,sort_keys=True))
    return root, index, data, task_hashes, rows


def test_formal_source_bridge_reads_actual_task_hashes_and_core120_profile(complete_disk_structure, monkeypatch):
    root, index, data, task_hashes, rows = complete_disk_structure
    calls = []
    marker = {"purpose":"MOCKED_CPU_BRIDGE_ONLY_NOT_REAL_VALIDATION", "worlds":{r["world_id"]:r for r in rows}}
    def mocked_receipt(report, plan, *, required_profile=None):
        assert report is marker
        assert required_profile == "core120"
        assert plan["source_code_sha256"]
        selected = {r["world_id"] for r in plan["expected"]}
        assert len(selected) == 60
        assert Counter({row["world_id"]:row["city"] for row in rows if row["world_id"] in selected}.values()) == {
            "austin":20,"chicago":20,"vienna":20}
        assert len(plan["expected"]) == 960
        world_map = {r["world_id"]:r for r in rows}
        for row in plan["expected"]:
            assert row["role"] == "train"
            assert row["task_sha256"] == task_hashes[row["task_id"]]
            assert row["source_manifest_sha256"] == index["source_manifest_sha256"]
            assert row["world_file_sha256"] == world_map[row["world_id"]]["world_file_sha256"]
            assert row["world_pixel_sha256"] == world_map[row["world_id"]]["world_pixel_sha256"]
        calls.append(plan)
        return {"status":"MOCKED_BRIDGE_BOUND_NOT_REAL_ACCEPTANCE"}
    monkeypatch.setattr(materialized,"validate_audit_receipt",mocked_receipt)
    import rswm_inria.training as training
    # Covers either a module-level import or a local import in the constructor.
    if hasattr(training,"validate_audit_receipt"):
        monkeypatch.setattr(training,"validate_audit_receipt",mocked_receipt)
    dataset = ClipDataset(index,data,root,source_audit=marker)
    assert len(dataset) == 3840 and len(calls) == 1
    assert dataset.diagnostic is False
