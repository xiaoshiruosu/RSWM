#!/usr/bin/env python3
# =============================================================================
# run_baseline.py — 按名跑 **一个** baseline,吃共享 manifest,吐聚合结果
#
# 共享协议(所有方法必须一致,否则表格不可比):同一份 H5、同一份 split、同一批
# sample key、同一分辨率、同一帧数、同一 gen_seed、同一套打分函数。这些全部由
# manifest 钉死,本脚本对每条 record 逐项核对,不符 **直接非零退出**。
#
# 用法:
#   python server_run/baselines/run_baseline.py \
#       --baseline persistence \
#       --manifest $WORK/baselines/manifest.json \
#       --out      $WORK/baselines/persistence
#   # 只查依赖/文件、不跑:
#   python server_run/baselines/run_baseline.py --baseline mvsplat360 \
#       --manifest ... --out ... --dryrun
#
# 产物(每个 baseline 只有 2-3 个文件 —— qk73 磁盘紧、rx40 inode 紧,绝不散落
# 成千上万小文件):
#   <out>/scores.json            聚合 + 逐样本 Track B / SSIM / PSNR
#   <out>/resolved_manifest.json 实际用到的 sample key 列表 + 协议参数(审计用)
#   <out>/frames.npz             --save-frames 时才写,一个压缩包装全部生成帧
#
# 打分刻意复用既有代码,保证与既有 Track B 表逐字节可比:
#   - lora_remotesense/new/trajectory_metrics.py  score_trajectory
#   - server_run/run_trajectory_eval.py           _flatten/_agg/TRACKB_KEYS
#   - lora_remotesense/new/metrics.py             compute_ssim/compute_psnr
# =============================================================================
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parents[1]
_NEW = ROOT / "lora_remotesense" / "new"
_SERVER_RUN = ROOT / "server_run"

MANIFEST_REQUIRED = ("h5", "split", "num_frames", "resolution", "gen_seed")


def _fatal(msg: str) -> None:
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(2)


def load_manifest(path: Path) -> Dict[str, Any]:
    """读共享 manifest;缺 key / 缺文件一律响亮失败。"""
    if not path.is_file():
        _fatal(f"manifest 不存在: {path}(见 README_BASELINES.md 的 manifest 模板)")
    doc = json.loads(path.read_text())
    # ``num_frames=null`` and ``resolution=null`` are intentional for the v2
    # benchmark manifest: formal samples can contain T={17,25,33} and heterogeneous
    # native condition sizes.  Presence is mandatory; a single global value is not.
    missing = [k for k in MANIFEST_REQUIRED if k not in doc]
    if missing:
        _fatal(f"manifest 缺字段 {missing};必需: {list(MANIFEST_REQUIRED)}")
    res = doc["resolution"]
    if res is not None and (not isinstance(res, (list, tuple)) or len(res) != 2):
        _fatal(f"manifest.resolution 必须是 [H,W],得到 {res!r}")
    for k in ("h5", "split"):
        if not doc[k] or not Path(doc[k]).is_file():
            _fatal(f"manifest.{k} 指向的文件不存在: {doc[k]}")
    return doc


def resolve_keys(h5_path: str, split_path: str, manifest: Dict[str, Any]) -> List[str]:
    """定出这一轮要跑哪些 sample key。

    规则与 server_run/run_trajectory_eval.py:70-74 完全一致:
    sorted(f['data'].keys()) 后按 split 的 val_indices 取 —— 确定性,所有方法一致。
    manifest 显式给了 sample_keys 就用它(更强的可复现保证)。
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        if "data" not in f:
            _fatal(f"{h5_path} 里没有 'data' 组")
        all_keys = sorted(f["data"].keys())

    # A frozen explicit ID set is the strongest fairness contract and must take
    # precedence over any legacy split parser.  The old code read val_indices before
    # this branch, which made a perfectly valid v2 test manifest fail even though all
    # sample IDs were already pinned.
    explicit = manifest.get("sample_keys")
    if explicit:
        missing = [k for k in explicit if k not in set(all_keys)]
        if missing:
            _fatal(f"manifest.sample_keys 里有 {len(missing)} 个 key 不在 H5 中,例: {missing[:3]}")
        return list(explicit)

    split_doc = json.load(open(split_path))
    split_name = str(manifest.get("source_split") or "val")
    old_key = f"{split_name}_indices"
    if old_key in split_doc:
        split_idx = split_doc[old_key]
    else:
        try:
            split_idx = split_doc["splits"][split_name]["indices"]
        except (KeyError, TypeError):
            _fatal(f"split 缺 {old_key} 或 splits.{split_name}.indices: {split_path}")
    keys = [all_keys[i] for i in split_idx if i < len(all_keys)]
    n = manifest.get("max_samples") or 0
    if n:
        keys = keys[:int(n)]
    if not keys:
        _fatal("解析出的 sample key 为空(split 的 val_indices 与 H5 对不上?)")
    return keys


def load_sample(f, key: str, gen_seed: int, window) -> Dict[str, Any]:
    """从 H5 取一条样本,拼成适配器要的 sample dict。"""
    g = f["data"][key]
    if "waypoints" not in g or "segment_angles" not in g:
        _fatal(f"样本 {key} 没有 waypoints/segment_angles —— 该数据集是旧 builder 产的,"
               f"必须用 lora_remotesense/build_dataset_from_benchmark.py 重建。")
    if "prompt" not in g.attrs:
        _fatal(f"样本 {key} 缺 prompt attr")
    # [v2.5 审查修复] scene_image 必须是【真·整幅场景】full_scene(512²),
    # 不是 g["image"](首帧窗口上采样到 480×832)。旧写法让 det/random 基线
    # 拿 scene 像素坐标的 waypoints 在一张放大的窗口图上滑窗 → 垃圾视频+
    # 垃圾分数;而 1.5× 场景/窗口守卫抓不住(480/128=3.75 ≥ 1.5 照样过)。
    # 这正是 builder 注释里点名过的服务器事故(两基线 valid_pct=0%)。
    # 缺 full_scene 时置 None —— run_baselines 会自动省略 det/random,
    # persistence/gt_oracle 照常(与 run_trajectory_eval.py 同口径)。
    _scene = g["full_scene"][:] if "full_scene" in g else None
    if _scene is None:
        print(f"[warn] {key}: 无 full_scene → deterministic/random 基线将被跳过"
              f"(persistence/gt_oracle 不受影响)", flush=True)
    return dict(
        sample_id=key,
        prompt=str(g.attrs["prompt"]),
        scene_image=_scene,                        # 真·整幅静态场景(512²)
        source_frame=g["image"][:],                # 首帧窗口(基线各自按需用)
        gt_frames=g["video_frames"][:],            # (T,H,W,3) GT
        waypoints=g["waypoints"][:],               # [K+1,2] (cy,cx) 相机路径
        segment_angles=g["segment_angles"][:],     # [K] 相机段角(度)
        window_size=window,
        T=int(g.attrs.get("num_frames", g["video_frames"].shape[0])),
        gen_seed=int(gen_seed),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="按名跑一个 RSWM baseline(共享 manifest 协议)")
    # 这三个除 --list 外都必填,但不用 argparse 的 required=True:那样 `--list`
    # 也得凑齐一堆假路径才能列表。改成手工检查。
    ap.add_argument("--baseline", help="注册名;`--list` 可查")
    ap.add_argument("--manifest")
    ap.add_argument("--out", help="输出目录(建议 $WORK/baselines/<name>)")
    ap.add_argument("--save-frames", action="store_true",
                    help="额外写一个 frames.npz(所有生成帧打包成单文件)")
    ap.add_argument("--dryrun", action="store_true",
                    help="只查适配器/依赖/文件与样本解析,不生成不打分")
    ap.add_argument("--skip-failed", action="store_true",
                    help="[v2.6] 单样本 SampleUnavailable 时跳过并显式计数,"
                         "而非整跑中止(外部基线的部分失败协议;失败样本绝不进均值)")
    ap.add_argument("--min-success", type=float, default=1.0,
                    help="[v2.6] 成功样本占比门槛(0-1;与 --skip-failed 连用,"
                         "低于此值非零退出。MVSplat360 冒烟/全量用 0.8)")
    ap.add_argument("--list", action="store_true", help="列出所有 baseline 后退出")
    args = ap.parse_args()
    if not (0.0 < args.min_success <= 1.0):
        _fatal(f"--min-success 必须在 (0,1],得到 {args.min_success}")

    sys.path.insert(0, str(_HERE))
    import registry  # noqa: E402  (纯 stdlib,不需要 numpy)

    if args.list:
        for a in registry.list_baselines():
            print(f"{a.name:<30} {a.kind:<20} {a.status:<13} gpu={a.requires_gpu}"
                  f"{'  [UPPER BOUND, 非方法]' if a.upper_bound else ''}")
        return 0

    for flag in ("baseline", "manifest", "out"):
        if not getattr(args, flag):
            _fatal(f"缺 --{flag}(除 --list 外必填)")

    try:
        adapter = registry.get(args.baseline)
    except KeyError as e:
        _fatal(str(e))

    manifest = load_manifest(Path(args.manifest))
    out_dir = Path(args.out)
    _resolution = manifest.get("resolution")
    if _resolution is None:
        H = W = None
        resolution_label = "per-sample"
    else:
        H, W = int(_resolution[0]), int(_resolution[1])
        resolution_label = f"{H}x{W}"
    gen_seed = int(manifest["gen_seed"])
    # [v2.5] manifest 的 num_frames 允许为 null:canonical 数据集帧数按样本
    # 变化(v2.8 起 T=17/25/33),单一值无法硬比;为 null 时只做逐样本 GT 对齐校验。
    _nf = manifest.get("num_frames")
    n_frames = None if _nf in (None, "", "null") else int(_nf)

    print(f"[info] baseline={adapter.name} kind={adapter.kind} status={adapter.status} "
          f"gpu={adapter.requires_gpu}")
    if adapter.upper_bound:
        print(f"[warn] {adapter.name} 是 **指标上界**,不是可部署方法。"
              f"论文里必须标注为 upper bound,不得与真实方法并列排名。")
    if adapter.kind == "external_generative":
        missing = getattr(adapter, "missing_dependencies", lambda: [])()
        for m in missing:
            print(f"[dep-missing] {m}")

    keys = resolve_keys(manifest["h5"], manifest["split"], manifest)
    print(f"[info] 共享协议: h5={manifest['h5']}")
    print(f"[info] split={manifest['split']}  样本 {len(keys)} 条  "
          f"分辨率 {resolution_label}  帧数 {n_frames or 'per-sample'}  "
          f"gen_seed {gen_seed}")

    if args.dryrun:
        print(f"[dryrun] 前 3 个 key: {keys[:3]}")
        print(f"[dryrun] 会写到: {out_dir}/scores.json (+resolved_manifest.json"
              f"{', frames.npz' if args.save_frames else ''})")
        if adapter.status != "implemented":
            print(f"[dryrun] 注意: status={adapter.status} —— 真跑会抛 NotImplementedError。")
        print("[dryrun] OK(没有生成、没有打分、没有落盘)")
        return 0

    # ---- 真跑:此处才需要 numpy/h5py/cv2 ----
    import h5py
    import numpy as np

    sys.path.insert(0, str(_NEW))
    sys.path.insert(0, str(_SERVER_RUN))
    import metrics as MET                # noqa: E402
    import trajectory_metrics as TM      # noqa: E402
    # 复用既有 Track B 聚合逻辑,保证与 run_trajectory_eval.py 的表口径一致
    import run_trajectory_eval as RTE    # noqa: E402

    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample: List[Dict[str, Any]] = []
    frames_out: Dict[str, Any] = {}

    with h5py.File(manifest["h5"], "r") as f:
        window = None
        if "metadata" in f and "pose_window_size" in f["metadata"].attrs:
            window = tuple(int(v) for v in f["metadata"].attrs["pose_window_size"])
        if window is None:
            _fatal("H5 的 metadata 组没有 pose_window_size attr —— "
                   "deterministic_translation/random_direction 需要窗口尺寸才能裁窗。")

        failed_keys: List[str] = []
        for i, key in enumerate(keys):
            sample = load_sample(f, key, gen_seed, window)
            try:
                record = adapter.run(sample)  # 适配器内部已 validate_record
            except registry.SampleUnavailable as e:
                # [v2.6] 单样本级失败:--skip-failed 时显式计数跳过(失败样本
                # 绝不进均值 —— 与静默平均黑帧是两个世界);未开启则保持
                # 原 fail-fast 行为。
                if not args.skip_failed:
                    _fatal(f"{key}: {e}\n(外部基线的部分失败请用 "
                           f"--skip-failed --min-success 0.8 显式处理)")
                print(f"[skip-failed] {key}: {e}", flush=True)
                failed_keys.append(key)
                continue

            rt = record["runtime_metadata"]
            # [v2.5 审查修复] 旧写法拿 manifest 的单一 num_frames/resolution 硬比
            # 每条 record,与 canonical 数据集互斥:①帧数按样本 T=18/27/36 变化
            # (段数 2-4 × 9 步),第一条 T 不等的样本就 _fatal —— 对 outpaint 是
            # 烧完 GPU 才判死、零产出;②det/random 原生输出 128×222,永远 ≠
            # manifest 的 480×832。改为【per-sample 自洽】:生成帧数/分辨率必须
            # 与该样本自己的 GT 对齐(这才是"可比"的真定义),manifest 的值仅在
            # 非 None 时作为软提示记录。
            gt_T = int(len(sample["gt_frames"]))
            if rt["num_frames"] != gt_T:
                _fatal(f"{key}: 生成帧数 {rt['num_frames']} != 本样本 GT 帧数 {gt_T} "
                       f"—— 逐样本对齐被破坏,指标不可比。")
            if n_frames is not None and rt["num_frames"] != n_frames:
                print(f"[note] {key}: 帧数 {rt['num_frames']} 与 manifest 记录的 "
                      f"{n_frames} 不同(canonical 数据集帧数按样本变化,属正常)",
                      flush=True)
            gen_hw = tuple(rt["resolution"])
            gt_hw = (int(sample["gt_frames"].shape[1]), int(sample["gt_frames"].shape[2]))
            if gen_hw != gt_hw:
                # 诊断基线原生窗口分辨率(128×222)是合法的:打分前会统一 resize,
                # 但必须在 record 里如实记录、并与 manifest 的差异显式说明。
                print(f"[note] {key}: 生成分辨率 {gen_hw} != GT {gt_hw}"
                      f"(基线原生口径;打分按 window_size 统一)", flush=True)

            gen = record["generated_frames"]
            tgt = record["target_frames"]
            score = TM.score_trajectory(
                gen, record["route_metadata"]["waypoints"],
                record["route_metadata"]["segment_angles"],
                num_frames=rt["num_frames"],
                # [v2.5 审查修复] 必须传 window_size:不传则 Track B 光流在
                # 480×832 上算 —— 仓库自己的标定结论是该分辨率下位移 ~34px/帧
                # 超 Farnebäck 量程(连 gt_oracle 都只拿到 80.67°),
                # run_trajectory_eval.py 对缺该参数是硬拒绝的。window 已在
                # :198-201 从 H5 metadata 读出并对缺失 _fatal。
                window_size=window,
                # 诊断基线不吃文本条件(condition_type='none'),与
                # run_trajectory_eval.py:143 一致用 text_only=False。
                text_only=(record["route_metadata"]["condition_type"] == "text_only"),
            )
            row = RTE._flatten(score)
            row["sample_id"] = key
            # Paper protocol: t=0 is the provided condition frame and must never
            # inflate future rollout quality.  Trajectory recovery still sees the
            # full clip because its displacement origin is the condition frame.
            if len(gen) <= 1:
                _fatal(f"{key}: 没有 future frame,无法按 future-only 协议评分")
            future = range(1, len(gen))
            row["ssim"] = float(np.mean([
                MET.compute_ssim(gen[t], tgt[t]) for t in future]))
            row["psnr"] = float(np.mean([
                MET.compute_psnr(gen[t], tgt[t]) for t in future]))
            row["condition_frames_excluded"] = 1
            row["scored_future_frames"] = len(gen) - 1
            row["wall_seconds"] = rt["wall_seconds"]
            per_sample.append(row)
            if args.save_frames:
                frames_out[key] = gen
            if (i + 1) % 10 == 0 or i + 1 == len(keys):
                print(f"[info] {i + 1}/{len(keys)}")

    # [v2.6] 成功率结算:失败样本已被排除出一切均值;这里把排除本身摆到明面。
    n_ok = len(keys) - len(failed_keys)
    success_rate = n_ok / len(keys) if keys else 0.0
    if failed_keys:
        print(f"[warn] {len(failed_keys)}/{len(keys)} 样本失败被跳过: "
              f"{failed_keys[:5]}{'…' if len(failed_keys) > 5 else ''}")
    if not per_sample:
        _fatal("没有任何成功样本 —— 无可聚合。")

    agg = RTE._agg(per_sample, RTE.TRACKB_KEYS)
    for k in ("ssim", "psnr"):
        vals = [r[k] for r in per_sample if r.get(k) is not None and np.isfinite(r[k])]
        agg[k] = float(np.mean(vals)) if vals else None

    doc = {
        "baseline": adapter.name,
        "kind": adapter.kind,
        "status": adapter.status,
        "upper_bound": adapter.upper_bound,
        "tested": False,   # 本仓从未对 baseline 做过 runtime 测试
        "protocol": {"h5": manifest["h5"], "split": manifest["split"],
                     "resolution": _resolution, "num_frames": n_frames,
                     "gen_seed": gen_seed, "n_samples": len(keys)},
        "scoring_scope": "future_frames_only",
        "condition_frames_excluded": 1,
        # [v2.6] 失败样本显式入档:表格里的 n 必须以 n_success 为准并注明
        # 缺失(REVIEWER_STYLE O6 的幸存者偏差教训 —— 失败不是随机缺失)。
        "n_success": n_ok,
        "n_failed": len(failed_keys),
        "failed_keys": failed_keys,
        "success_rate": success_rate,
        "aggregate": agg,
        "per_sample": per_sample,
    }
    (out_dir / "scores.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False))
    (out_dir / "resolved_manifest.json").write_text(json.dumps(
        {"baseline": adapter.name, "sample_keys": keys, "protocol": doc["protocol"]},
        indent=2, ensure_ascii=False))
    if args.save_frames:
        np.savez_compressed(out_dir / "frames.npz", **frames_out)

    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"\n[done] → {out_dir}/scores.json"
          f"(成功 {n_ok}/{len(keys)} = {success_rate:.0%})")
    if adapter.upper_bound:
        print("[warn] 再说一遍:这是 upper bound,不是方法。")
    if success_rate < args.min_success:
        print(f"[FATAL] 成功率 {success_rate:.0%} < --min-success "
              f"{args.min_success:.0%} —— 失败样本非随机缺失(往往是最难的那批),"
              f"聚合数字不可入表。", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
