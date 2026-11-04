#!/usr/bin/env python3
# =============================================================================
# make_variant_h5.py — 归因实验派生数据集(v2.5,拍板④)
#
# 两种 variant(都从修复版 GEO_H5 派生,图像/轨迹 GT 不动):
#   shuffled_pose      : poses 在样本间确定性打乱(错位配对)。图像、prompt、
#                        目标序列不变 —— 区分"编码真实轨迹的信息代价"与
#                        "任意 OOD pose 对预训练 pose 通路的通用扰动"。
#                        若它同样费 ~8 点 FVD,"几何费保真"实为通路脆弱。
#   direction_stripped : 保留几何轨迹(poses 原样),prompt 删除一切方向词
#                        (pt_v3 include_direction=False 模板重生成)——
#                        检验几何通道能否【独立】携带方向。只有此实验给出
#                        正证据,论文才允许写 either-route 措辞(拍板④)。
#
# 安全:dst 必须不存在(不覆盖);src 只读;metadata 写 variant 溯源
# (variant/variant_of_sha256/pose_permutation_seed);DRYRUN=1 只打印。
#
# 用法(登录节点或建库作业内):
#   python server_run/make_variant_h5.py --mode shuffled_pose \
#       --src $WORK/data/train_geo.h5 --dst $WORK/data/train_shufpose.h5
#   python server_run/make_variant_h5.py --mode direction_stripped \
#       --src $WORK/data/train_geo.h5 --dst $WORK/data/train_geostrip.h5
# =============================================================================
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import h5py



def _sha256_file_chunked(p, chunk: int = 1 << 22) -> str:
    """分块 SHA256。优先复用 run_manifest.sha256_file(单一来源),
    拿不到就本地实现 —— 两者算法相同(标准 SHA256,分块不改结果)。"""
    try:
        _new = Path(__file__).resolve().parent.parent / "lora_remotesense" / "new"
        if str(_new) not in sys.path:
            sys.path.insert(0, str(_new))
        from run_manifest import sha256_file          # noqa: E402
        return sha256_file(p)
    except Exception:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lora_remotesense.build_dataset_from_benchmark import (  # noqa: E402
    TAXONOMY,
    TAXONOMY_SHA256,
    _make_prompt,
    canonical_scene,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["shuffled_pose", "direction_stripped", "rich_text"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffled_pose 的置换种子(确定性)")
    args = ap.parse_args()
    dryrun = os.environ.get("DRYRUN", "0") == "1"

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_file():
        sys.exit(f"[FATAL] src 不存在: {src}")
    if dst.exists():
        sys.exit(f"[FATAL] dst 已存在,拒绝覆盖: {dst}(删除或换名后重跑)")
    if src.resolve() == dst.resolve():
        sys.exit("[FATAL] src == dst")

    # [v3.1 P0] 分块求哈希。原写法 `hashlib.sha256(src.read_bytes())` 会把
    # **整个源 H5** 读进内存 —— canonical 库带 full_scene,12 个样本就 1.3 GB,
    # N_MAIN=4000 时是几百 GB,正式跑 32/33/35 变体臂时必然 OOM 被 kill(exit 137)。
    # 求哈希从来不需要一次性持有全部字节。
    src_sha = _sha256_file_chunked(src)
    print(f"[variant] mode={args.mode}  src={src}(sha {src_sha[:12]}…)\n"
          f"          dst={dst}  seed={args.seed}  DRYRUN={int(dryrun)}")
    if dryrun:
        print("[DRYRUN] 不写任何文件")
        return

    shutil.copyfile(src, dst)
    n_mod = 0
    with h5py.File(dst, "a") as f:
        data = f["data"]
        keys = sorted(data.keys())
        n = len(keys)

        if args.mode == "shuffled_pose":
            # 确定性无不动点置换(derangement),**在同帧数分组内**进行。
            # [v2.5 审查修复;v2.8 起去重为 17/25/33] 样本帧数不定长;
            # 跨组换 pose 会让 poses 长度 ≠ 该样本帧数 —— inference.py 用
            # len(c2ws) 反向钳制生成长度,约 1/3 样本直接 RuntimeError(被
            # evaluate 的 except 吞掉 → 静默丢样本),其余一半生成帧数被供体
            # 截短 → SSIM/FID/FVD 与 23 号不可比,而"只有 pose 内容变了"正是
            # 本实验唯一要保证的前提。分组置换后帧数严格不变。
            poses_all = [data[k]["poses"][:] for k in keys]
            lens = [len(p_) for p_ in poses_all]
            groups = {}
            for i, L in enumerate(lens):
                groups.setdefault(int(L), []).append(i)
            perm = np.arange(n)
            rng = np.random.default_rng(args.seed)
            singleton = []
            for L, idxs in sorted(groups.items()):
                m = len(idxs)
                if m < 2:
                    singleton.extend(idxs)     # 组内只有 1 个 → 无法错位
                    continue
                # 组内确定性 derangement:随机置换后把不动点与下一位对换
                sub = rng.permutation(m)
                for _ in range(1000):
                    fixed = np.where(sub == np.arange(m))[0]
                    if len(fixed) == 0:
                        break
                    for a in fixed:
                        b = (a + 1) % m
                        sub[a], sub[b] = sub[b], sub[a]
                if np.any(sub == np.arange(m)):
                    raise RuntimeError(f"帧数组 T={L}(n={m})无法构造 derangement")
                for a, b in enumerate(sub):
                    perm[idxs[a]] = idxs[int(b)]
            if singleton:
                raise RuntimeError(
                    f"以下样本的帧数在数据集中唯一,无法在同帧数组内错位换 pose: "
                    f"{[keys[i] for i in singleton][:5]}(共 {len(singleton)});"
                    f"这类样本若保留原 pose 就不是扰动对照 —— 请扩大样本数或"
                    f"改用 --allow-singleton(尚未实现)显式声明处理方式。")
            assert not np.any(perm == np.arange(n)), "置换仍含不动点"
            assert sorted(perm.tolist()) == list(range(n)), "置换非双射"
            assert all(lens[int(perm[i])] == lens[i] for i in range(n)), \
                "存在跨帧数组的换 pose(帧数必须严格不变)"
            for i, k in enumerate(keys):
                del data[k]["poses"]
                data[k].create_dataset("poses", data=poses_all[int(perm[i])])
                n_mod += 1
            f["metadata"].attrs["pose_injection"] = "shuffled_across_samples_same_T"
            f["metadata"].attrs["pose_permutation_seed"] = int(args.seed)
            print(f"[variant] 同帧数分组置换: " +
                  ", ".join(f"T={L}:{len(v)}样本" for L, v in sorted(groups.items())))

        elif args.mode == "rich_text":
            # [v2.8 · 35 号臂] 逐段方向全部语言化 —— 与 direction_stripped 相反:
            # 那个删方向词测"几何能否独立带方向",这个把【每一段】的方向都写进
            # prompt,测"文本通道的失败是词表太穷还是通道本身带不动方向"。
            # 图像/轨迹/位姿一律不动,只改 prompt(原 prompt 留档)。
            src_tax = f["metadata"].attrs.get("taxonomy_file_sha256")
            if src_tax is None or str(src_tax) != TAXONOMY_SHA256:
                raise RuntimeError(
                    f"taxonomy 指纹不一致/缺失(src={src_tax})—— rich_text 会用当前"
                    f"模板重写 prompt,与原 prompt 不同源。请用同版 builder 重建。")
            _DIRW = {0: "east", 90: "north", 180: "west", 270: "south"}
            for k in keys:
                g = data[k]
                sc = canonical_scene(g.attrs.get("source_scene_category", "general"))
                sa = g["segment_angles"][:] if "segment_angles" in g else []
                # [v2.8.1 P0-3] 净方向句必须保留。此前 include_direction=False 把
                # 净方向句删掉、只留逐段词,而评估侧 parse_direction_*_from_prompt
                # 按固定序 east→north→west→south 取【第一个命中词】——
                # "segment 1 toward the west, then segment 2 toward the north"
                # 会解析成 N,既不是净方向也不是第一段。35 号臂的 Track A 目标
                # 因此基本是随机的(实测 3/4 样本与 waypoints 净方向差 90°)。
                # 现在:净方向句 + 逐段描述,prompt 自洽;评估侧另行改成以
                # waypoints 为真源(见 evaluate.py 的 direction_target_source)。
                base = _make_prompt(sc, int(g.attrs.get("num_segments", 2)), sa,
                                    waypoints=(g["waypoints"][:] if "waypoints" in g else None),
                                    include_direction=True)
                # 逐段:相机角量化到四方位(与 _make_prompt 同一量化口径)
                segs = []
                for j, a in enumerate(np.asarray(sa, dtype=float).ravel()):
                    ang = float(a) % 360
                    q = (0 if (ang < 45 or ang >= 315) else
                         90 if ang < 135 else 180 if ang < 225 else 270)
                    segs.append(f"segment {j + 1} toward the {_DIRW[q]}")
                rich = base.rstrip()
                if segs:
                    rich += " The camera moves " + ", then ".join(segs) + "."
                g.attrs["prompt_original"] = g.attrs.get("prompt", "")
                g.attrs["prompt"] = rich
                # 文本命令了什么 —— Track B 的 commanded 语义要用
                g.attrs["prompt_schema"] = "rich_persegment"
                n_mod += 1
            f["metadata"].attrs["prompt_schema"] = "rich_persegment"
            f["metadata"].attrs["prompt_template_version"] = (
                str(f["metadata"].attrs.get("prompt_template_version",
                                            TAXONOMY["prompt_template_version"]))
                + "+rich_text_persegment")

        else:  # direction_stripped
            # [v2.5 审查修复] prompt 整句用【当前】JSON 重生成 —— 若 src 建库时
            # 的 taxonomy 与现在不同,geostrip 与 23 号的差异就从"有无方向词"
            # 变成"方向词+场景措辞"两处(拍板④要的 either-route 正证据被污染),
            # 且 metadata 版本号会主动撒谎。指纹不一致即拒跑,逼人显式重建。
            src_tax = f["metadata"].attrs.get("taxonomy_file_sha256")
            if src_tax is None:
                raise RuntimeError(
                    "src H5 无 taxonomy_file_sha256(v2.5 之前的构建)—— "
                    "direction_stripped 会用当前模板重写 prompt,与原 prompt "
                    "不同源。请用 v2.5 builder 重建后再派生。")
            if str(src_tax) != TAXONOMY_SHA256:
                raise RuntimeError(
                    f"taxonomy 指纹不一致:src={str(src_tax)[:12]}… vs "
                    f"当前={TAXONOMY_SHA256[:12]}… —— 重写 prompt 会引入场景措辞"
                    f"差异。要么用建库时的 taxonomy,要么重建数据集。")
            for k in keys:
                g = data[k]
                sc = canonical_scene(g.attrs.get("source_scene_category", "general"))
                nseg = int(g.attrs.get("num_segments", 2))
                wp = g["waypoints"][:] if "waypoints" in g else None
                sa = g["segment_angles"][:] if "segment_angles" in g else []
                new_prompt = _make_prompt(sc, nseg, sa, waypoints=wp,
                                          include_direction=False)
                assert "toward the" not in new_prompt and "ward" not in \
                    new_prompt.replace("toward", ""), f"仍含方向词: {new_prompt}"
                g.attrs["prompt_original"] = g.attrs.get("prompt", "")
                g.attrs["prompt"] = new_prompt
                g.attrs["prompt_schema"] = "stripped"   # prompt 里没有方向词
                n_mod += 1
            f["metadata"].attrs["prompt_schema"] = "stripped"
            _src_ptv = f["metadata"].attrs.get(
                "prompt_template_version", TAXONOMY["prompt_template_version"])
            f["metadata"].attrs["prompt_template_version"] = (
                str(_src_ptv) + "+direction_stripped")

        f["metadata"].attrs["variant"] = args.mode
        f["metadata"].attrs["variant_of_sha256"] = src_sha

    print(f"[done] {n_mod}/{n} 样本已改写 → {dst}")
    print("[note] 评估注意:direction_stripped 的 Track A 方向目标由 "
          "waypoints 回退推导(evaluate v2.5 已支持,日志会打 "
          "'方向目标来源: waypoints=N');shuffled_pose 训评同用本 H5。")


if __name__ == "__main__":
    main()
