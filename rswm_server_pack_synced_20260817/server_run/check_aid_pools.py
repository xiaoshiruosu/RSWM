#!/usr/bin/env python
"""check_aid_pools.py — AID 四场景源图池严格验证(v2.4 重写,2026-07-23)。

旧版只查 `len(pool) >= threshold`,四个完全相同的 10001 全集池(含非法
thumbnail.png 索引 10000)也判 PASS —— 正是四重完整重复样本事故没被拦住的
原因。本版按 §七 规格同时验证:

  单池条件:非空 / >= threshold / 池内无重复 / 每个索引都是合法训练图
            (data/<CLASS>/*.jpg,排除 thumbnail.png、非 data/ 路径、非 JPG)
  池间条件:两两交集 = 0 / 四池不全同 / 并集大小 = 各池之和
  容量条件:len(pool) >= threshold,不允许 % 回绕补足(builder v2.4 已硬禁)

退出码:PASS -> 0,FAIL -> 4。FAIL 时不输出任何"论文 claim 成立"文本。

用法(工程根下、主环境、可访问 HF 缓存):
    python server_run/check_aid_pools.py [--threshold 125]
    python server_run/check_aid_pools.py --local-aid-dir /path/to/AID   # 本地目录版
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lora_remotesense.build_dataset_from_benchmark import (  # noqa: E402
    SCENE_CATEGORIES,
    LocalImageDataset,
    SceneSelectionError,
    _class_from_data_path,
    _raw_image_paths,
    _scene_pool_with_meta,
)

SCENES = list(SCENE_CATEGORIES)   # [v2.5] taxonomy 单一来源(flood 已更名 water)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=int, default=125,
                    help="每场景所需最小源图数 = N_MAIN/4")
    ap.add_argument("--local-aid-dir", default=None,
                    help="本地 AID 目录(root/<CLASS>/*.jpg);不给则走 HF 缓存")
    args = ap.parse_args()

    if args.local_aid_dir:
        print(f"[load] 本地 AID: {args.local_aid_dir}")
        ds = LocalImageDataset(args.local_aid_dir)
        ds_name = "blanchon/AID"
    else:
        from datasets import load_dataset
        print("[load] blanchon/AID (走 HF 缓存)...")
        ds = load_dataset("blanchon/AID", split="train")
        ds_name = "blanchon/AID"

    n = len(ds)
    failures = []
    is_local = hasattr(ds, "samples")

    # [v2.4.1] 合法性判据按数据集类型分路:HF blanchon/AID 的缓存路径有
    # .../data/<CLASS>/<f>.jpg 结构 → 用完整解析;本地目录是 <root>/<CLASS>/
    # <f>.jpg,不含字面 "data" 段 → 只查后缀 + thumbnail(类别由目录名/label
    # 路线保证)。旧版对两种模式套同一 data/ 判据:本地模式必然全场景误判
    # FAIL,HF label 路线则真空通过 —— 都错。
    def img_ok(p):
        if not p:
            return False
        pp = Path(str(p))
        if pp.suffix.lower() not in (".jpg", ".jpeg"):
            return False
        if pp.name.lower() == "thumbnail.png":
            return False
        return True if is_local else _class_from_data_path(p) is not None

    # ---- 数据集级:真实图片 / thumbnail 计数 ----
    paths = _raw_image_paths(ds)
    n_valid_img = sum(1 for p in paths if img_ok(p))
    n_thumb = sum(1 for p in paths
                  if p and Path(str(p)).name.lower() == "thumbnail.png")
    n_nonjpg = sum(1 for p in paths
                   if p and Path(str(p)).suffix.lower() not in (".jpg", ".jpeg"))
    print(f"[dataset] 总记录 {n}({'本地目录' if is_local else 'HF'});"
          f"合法训练图 {n_valid_img};thumbnail {n_thumb};非 JPG {n_nonjpg}")

    # ---- 建池(builder 同源逻辑;失败=硬错误而非静默全量) ----
    pools = {}
    for sc in SCENES:
        try:
            idx, cls_map, path_map = _scene_pool_with_meta(ds, sc, ds_name)
        except SceneSelectionError as e:
            failures.append(f"{sc}: 建池失败 —— {e}")
            continue
        pools[sc] = idx

        # 单池条件
        if not idx:
            failures.append(f"{sc}: 池为空")
            continue
        if len(idx) != len(set(idx)):
            failures.append(f"{sc}: 池内索引重复 {len(idx) - len(set(idx))} 个")
        if len(idx) < args.threshold:
            failures.append(f"{sc}: 池 {len(idx)} < 阈值 {args.threshold}"
                            f"(回绕已被 builder 硬禁,容量必须达标)")
        bad = [i for i in idx if not img_ok(paths[i])]
        if bad:
            failures.append(f"{sc}: {len(bad)} 个索引不是合法训练图"
                            f"(如 idx={bad[:3]};thumbnail/非 data//非 JPG)")
        print(f"  {sc:8s} pool={len(idx):5d}  类别数={len(set(cls_map.values()))}")

    # ---- 池间条件 ----
    have = [sc for sc in SCENES if sc in pools and pools[sc]]
    for a, b in combinations(have, 2):
        ov = set(pools[a]) & set(pools[b])
        if ov:
            failures.append(f"Scene pools overlap: {a}/{b}, count={len(ov)}")
    if len(have) == len(SCENES) and len({tuple(pools[sc]) for sc in have}) == 1:
        failures.append("四个池完全相同(过滤根本没生效)")
    if have:
        union = set().union(*(set(pools[sc]) for sc in have))
        total = sum(len(pools[sc]) for sc in have)
        if len(union) != total:
            failures.append(f"并集 {len(union)} != 各池之和 {total}(存在交叠)")

    # ---- 判决 ----
    if failures:
        print("\n[FAIL] 共 {} 项不满足:".format(len(failures)))
        for f_ in failures:
            print(f"  ✗ {f_}")
        return 4
    print("\n[PASS] AID 池预检通过:四池互斥、全部为合法训练图、容量达标(无回绕)。"
          "\n注意:这只是三道门的第一道。benchmark 重复门(--expect-zero)与"
          "\nsplit 泄漏门(--split)全部通过前,不得启用论文 disjoint-source "
          "claim、不得删论文 TODO。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
