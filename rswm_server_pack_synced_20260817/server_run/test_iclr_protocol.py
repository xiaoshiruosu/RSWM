#!/usr/bin/env python3
"""CPU-only behavioral regression tests for the ICLR protocol additions."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NEW = ROOT / "lora_remotesense" / "new"
SERVER = ROOT / "server_run"
sys.path.insert(0, str(NEW))
sys.path.insert(0, str(SERVER))

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name}: {detail}")


def expect_raises(name, exc, fn):
    try:
        fn()
    except exc:
        check(name, True)
    except Exception as e:
        check(name, False, f"wrong exception {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


def test_matrix():
    import iclr_matrix
    p = SERVER / "iclr_experiment_matrix.json"
    doc = iclr_matrix.load_and_validate(p)
    runs = iclr_matrix.materialize(doc)
    runnable = [r for r in runs if r["runnable"]]
    rank = [r for r in runnable if r["family"] == "rank_sweep"]
    target = {r["targets"] for r in runnable
              if r["family"] in ("rank_sweep", "target_ablation")
              and r.get("rank") == 16}
    check("matrix_rank_sweep", [r["rank"] for r in rank] == [4, 8, 16, 32, 64])
    check("matrix_target_ablation", target == {"attn_only", "attn_cam", "all"})
    check("matrix_frozen", any(r["adaptation"] == "frozen" for r in runnable))
    full = [r for r in runs if r["adaptation"] == "full"]
    check("matrix_full_ft_not_faked",
          len(full) == 1 and full[0]["runnable"] is False
          and full[0]["status"] == "not_implemented")
    check("matrix_official_continuous_registration",
          all(r["prompt_schema"] == "net_continuous"
              and r["path_estimator"] == "registration" for r in runs))


def test_manifest():
    import run_manifest as rm
    old = {k: os.environ.get(k) for k in (
        "RSWM_PROFILE", "SEM_ARM_MODE", "RSWM_PROMPT_SCHEMA",
        "RSWM_PATH_ESTIMATOR", "RSWM_POSE_MODE")}
    try:
        os.environ.update({
            "RSWM_PROFILE": "official", "SEM_ARM_MODE": "drop",
            "RSWM_PROMPT_SCHEMA": "net_continuous",
            "RSWM_PATH_ESTIMATOR": "registration", "RSWM_POSE_MODE": "native",
        })
        base = rm.build_manifest(
            role="headline_native", lora_rank=16, lora_alpha=16,
            lora_targets="attn_cam", max_train_samples=128,
            train_seed=42, split_seed=42, num_epochs=5,
            batch_size=1, gradient_accumulation_steps=4, learning_rate=1e-4)
        check("manifest_headline_ok", base["headline_eligible"] is True)
        required = ("lora_targets", "max_train_samples", "train_seed", "split_seed",
                    "num_epochs", "batch_size", "gradient_accumulation_steps",
                    "learning_rate", "prompt_schema", "path_estimator")
        check("manifest_records_recipe", all(k in base for k in required))
        changed = dict(base); changed["lora_targets"] = "all"
        expect_raises("manifest_target_mismatch_blocks", RuntimeError,
                      lambda: rm.assert_compatible(base, changed, ctx="test"))
        changed = dict(base); changed["max_train_samples"] = 64
        expect_raises("manifest_cap_mismatch_blocks", RuntimeError,
                      lambda: rm.assert_compatible(base, changed, ctx="test"))
        missing_prompt = rm.headline_eligible(
            "official", "drop", "native", "headline_native",
            prompt_schema=None, path_estimator="registration")
        check("manifest_missing_prompt_fail_closed", missing_prompt[0] is False)
        flow = rm.headline_eligible(
            "official", "drop", "native", "headline_native",
            prompt_schema="net_continuous", path_estimator="flow")
        check("manifest_flow_not_headline", flow[0] is False)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_split_compatibility(tmp: Path):
    from dataset_loader import LingBotDataset, split_dataset_from_file
    import evaluate
    import check_benchmark_duplicates as cbd

    h5 = tmp / "tiny.h5"
    with h5py.File(h5, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["total_instances"] = 6
        data = f.create_group("data")
        for i in range(6):
            g = data.create_group(f"sample_{i:05d}")
            g.create_dataset("image", data=np.zeros((4, 4, 3), np.uint8))
            g.create_dataset("video_frames", data=np.zeros((2, 4, 4, 3), np.uint8))
            g.create_dataset("poses", data=np.repeat(np.eye(4)[None], 2, axis=0))
            g.create_dataset("intrinsics", data=np.zeros((2, 4), np.float32))
            g.attrs["prompt"] = "probe"
            g.attrs["num_frames"] = 2
    official = tmp / "official.json"
    official.write_text(json.dumps({"splits": {
        "train": {"indices": [0, 1, 2], "keys": [f"sample_{i:05d}" for i in (0, 1, 2)]},
        "val": {"indices": [3, 4], "keys": [f"sample_{i:05d}" for i in (3, 4)]},
        "test": {"indices": [5], "keys": ["sample_00005"]},
    }}), encoding="utf-8")
    ds = LingBotDataset(str(h5), normalize=False)
    tr, va, tri, vai = split_dataset_from_file(ds, str(official))
    check("official_split_train_val", len(tr) == 3 and len(va) == 2
          and tri == [0, 1, 2] and vai == [3, 4])
    check("official_test_excluded_from_trainer", 5 not in tri and 5 not in vai)
    check("evaluate_reads_official_val", evaluate.load_val_indices(str(official)) == [3, 4])
    check("paper_evaluate_explicit_test",
          evaluate.load_split_indices(str(official), "test") == [5])
    train, holdout = cbd.load_train_holdout_split(str(official))
    check("duplicate_gate_official_holdout", train == {0, 1, 2}
          and holdout == {3, 4, 5})
    bad = tmp / "bad.json"
    bad.write_text(json.dumps({"train_indices": [0, 1], "val_indices": [1, 2]}),
                   encoding="utf-8")
    expect_raises("split_overlap_blocks", ValueError,
                  lambda: split_dataset_from_file(ds, str(bad)))


def test_rollout_end_to_end(tmp: Path):
    import evaluate_rollout_protocol as rp
    h5 = tmp / "rollout.h5"
    rng = np.random.default_rng(7)
    gt = rng.integers(1, 250, size=(17, 16, 16, 3), dtype=np.uint8)
    gen = gt.copy()
    gen[1:] = np.clip(gen[1:].astype(np.int16) + 1, 0, 255).astype(np.uint8)
    traj = np.stack([np.full(17, 150.0), 150.0 + 6.0 * np.arange(17)], axis=1)
    with h5py.File(h5, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["pose_window_size"] = (100, 100)
        data = f.create_group("data")
        g = data.create_group("sample_00000")
        g.create_dataset("video_frames", data=gt)
        g.create_dataset("trajectory", data=traj)
        g.create_dataset("full_scene", data=np.zeros((300, 300, 3), np.uint8))
    ev = tmp / "eval"
    (ev / "generated").mkdir(parents=True)
    np.save(ev / "generated" / "sample_00000.npy", gen)
    (ev / "run_manifest.json").write_text(json.dumps({
        "headline_eligible": True, "rswm_profile": "official",
        "prompt_schema": "net_continuous", "path_estimator": "registration",
    }), encoding="utf-8")
    result = rp.evaluate(SimpleNamespace(
        gt_h5=str(h5), eval_dir=str(ev), split_file=None, split_name="val",
        max_samples=None, allow_nonheadline=False, verbose=False))
    check("rollout_excludes_condition_frame",
          result["n_samples"] == 1 and result["n_predicted_frames"] == 16)
    check("rollout_condition_overlap_bins",
          all(result["overlap_bins"][k]["n_frames"] > 0 for k in
              ("dense_80", "sparse_50", "sparse_30", "sparse_10", "sparse_05")))
    check("rollout_horizon_exact17",
          result["horizon"]["exact_length_cohorts"]["17"]["n_frames"] == 16)
    check("rollout_adjacent_separate",
          result["adjacent_step_overlap_audit"]["definition"] ==
          "area overlap(frame t-1, frame t)")
    check("rollout_flow_claim_guard",
          "not explicit 3D geometry" in
          result["flow_vector_consistency"]["claim_guard"])


def test_common_static():
    src = (SERVER / "common.sh").read_text(encoding="utf-8")
    check("common_forwards_max_train_samples",
          '--max_train_samples "$MAX_TRAIN_SAMPLES"' in src)
    check("common_forwards_split_file", '--split_file "$TRAIN_SPLIT_FILE"' in src)
    check("common_official_defaults_continuous",
          "_RSWM_DEFAULT_PROMPT_SCHEMA=net_continuous" in src)
    check("common_official_defaults_registration",
          "_RSWM_DEFAULT_PATH_ESTIMATOR=registration" in src)


def main():
    test_matrix()
    test_manifest()
    test_common_static()
    with tempfile.TemporaryDirectory(dir=os.environ.get("RSWM_TEST_TMP") or ".") as td:
        tmp = Path(td)
        test_split_compatibility(tmp)
        test_rollout_end_to_end(tmp)
    print(f"\nsummary: {PASS} PASS / {FAIL} FAIL")
    return 0 if FAIL == 0 else 7


if __name__ == "__main__":
    raise SystemExit(main())
