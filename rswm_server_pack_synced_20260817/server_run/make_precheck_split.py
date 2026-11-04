#!/usr/bin/env python3
# =============================================================================
# make_precheck_split.py — 建库后立即预生成确定性 train/val split(v2.4.3)
#
# 动机(P1 修复):train/val 泄漏门(check_benchmark_duplicates --split)此前
# 只能等 job20 训练时产出 split 才能跑 —— "三道绿灯"的第三道不在自动编排里,
# 且等到 job20 时 21-30 已在烧 GPU。split 对 (total, split_seed) 完全确定,
# 建库后就能预生成、当场过泄漏门,门 PASS 才放行 20-31。
#
# ⚠️ 逐行同源约定:本脚本必须与 lora_remotesense/new/dataset_loader.py 的
# split_dataset 产生【逐位相同】的划分 —— 那边是
#     random.seed(seed); np.random.seed(seed);
#     indices = list(range(total)); random.shuffle(indices);
#     train = indices[:int(total*train_ratio)]; val = indices[train_n:]
# 改那边必须同步改这里(两处 RNG 语义一致性是泄漏门有效性的前提)。
# job20 产出真 split 后,可用 --verify-against 与预生成版逐位比对(防漂移)。
#
# 用法:
#   python server_run/make_precheck_split.py --gt-h5 $WORK/data/train_geo.h5 \
#       -o $WORK/train_val_split_precheck.json
#   python server_run/make_precheck_split.py --gt-h5 ... \
#       --verify-against $WORK/train_r16_sem_s42/train_val_split.json
# =============================================================================
import argparse
import json
import random
import sys

import numpy as np
import h5py


def make_split(total, split_seed=42, train_ratio=0.8):
    """与 dataset_loader.split_dataset 逐行同源(见文件头约定)。"""
    random.seed(split_seed)
    np.random.seed(split_seed)
    indices = list(range(total))
    random.shuffle(indices)
    train_n = int(total * train_ratio)
    return indices[:train_n], indices[train_n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("-o", "--output", default="train_val_split_precheck.json")
    ap.add_argument("--verify-against", default=None,
                    help="与训练作业产出的真 split 逐位比对(防两处 RNG 漂移);"
                         "不一致 exit 6")
    args = ap.parse_args()

    with h5py.File(args.gt_h5, "r") as f:
        if "data" in f:
            total = len(f["data"].keys())
        else:
            total = int(f["metadata"].attrs.get("total_instances", 0))
    if total <= 0:
        sys.exit(f"[FATAL] {args.gt_h5} 无样本")

    train_idx, val_idx = make_split(total, args.split_seed, args.train_ratio)
    out = {"train_indices": train_idx, "val_indices": val_idx,
           "split_seed": args.split_seed, "train_ratio": args.train_ratio,
           "total": total, "note": "precheck split — 与 dataset_loader 同源生成"}

    if args.verify_against:
        real = json.load(open(args.verify_against))
        same = (list(real.get("train_indices", [])) == train_idx
                and list(real.get("val_indices", [])) == val_idx)
        if not same:
            print(f"[FATAL] 预生成 split 与 {args.verify_against} 不一致 —— "
                  f"dataset_loader 的 RNG 语义漂移了,泄漏门判决作废,先修同源性!")
            sys.exit(6)
        print(f"[verify] 与 {args.verify_against} 逐位一致 ✓")

    json.dump(out, open(args.output, "w"))
    print(f"[done] total={total} train={len(train_idx)} val={len(val_idx)} "
          f"seed={args.split_seed} → {args.output}")


if __name__ == "__main__":
    main()
