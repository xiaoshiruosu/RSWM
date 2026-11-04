#!/usr/bin/env python3
# =============================================================================
# validate_direction_metric.py — 方向指标校验 + 廉价基线(纯 CPU,存量数据)
#
# 检查项(全部通过才有资格烧 GPU 跑多种子):
#   A.  GT 地板:在 ground-truth 滑窗视频上跑同一 Farnebäck 管线,
#       报告 vs 内容角(=方向词+180°,修正约定)与 vs 相机角 两种目标。
#       预期:vs 内容角 ≈ 22.5°(四方位量化地板),vs 相机角 ≈ 157.5°。
#       同时按场景分组(重点看 flood/rural 低纹理是否失稳)。
#       [v2.2 修复 2026-07-23] A 与各配置角误差、流幅度的光流一律先降到
#       原生窗口分辨率(metadata pose_window_size)再算 —— 旧版直接在
#       480x832 原始帧上跑,GT 位移 ~34px/帧 × step=3 采样 ≈102px/对,
#       超 Farnebäck 量程,真实 GT 地板实测 82.2°(量程垃圾,非指标语义
#       问题;与 Track B v1 gt_oracle 80.67° 同一事故)。--fullres 可复现
#       旧口径供对照。另报 step=1 逐帧对地板:窗口口径下 step=3 每对
#       ~27px 仍处量程边缘,两读数分离 → 指标定义须改 step=1。
#   A2. 人工平移校验:把 GT 源帧按 8 个已知方向合成平移视频,跑估计器,
#       报告恢复误差(预期 ≈ 0°,纯粹校验估计器本身,与量化无关)。
#       [v2 标定] 默认位移改为 6 px/frame:服务器实测 480x832 下 34 px/frame
#       超出 Farnebäck (winsize=15, levels=3) 量程,恢复误差 ~75° 垃圾值;
#       小位移下为 0.020°。并支持先缩到原生窗口分辨率再校验(window_size)。
#   B.  流幅度:GT 与各 eval 配置生成视频的平均光流幅度(px/帧对),
#       区分"方向错"与"根本没动";按场景分组。
#   C.  配对检验:共享轨迹的两两配置逐轨迹角误差配对 t + Wilcoxon +
#       bootstrap 95% CI;并报告每配置 mean/median/10%截尾均值 +
#       top-5 异常轨迹(检查 103° 是否少数轨迹拉动)。
#   D.  persistence 基线:所有帧=源帧 的 SSIM/PSNR(无模型基线,校准
#       "只保持外观"能拿多少分);按场景分组。
#
# 用法(激活 rswm 环境,包根执行;eval 目录 = evaluate*.py 的 output_dir,
# 其下有 generated/*.npy):
#   python server_run/validate_direction_metric.py \
#       --gt-h5 $WORK/data/train_semantic.h5 \
#       --split $WORK/train_main/train_val_split.json \
#       --eval-dirs textonly=$WORK/eval_r128_sem textpluecker=$WORK/eval_r128_geo base=$WORK/eval_base \
#       -o $WORK/direction_validation.json
# =============================================================================
import argparse, json, sys
from pathlib import Path

import numpy as np
import h5py
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lora_remotesense" / "new"))
from metrics import estimate_motion_direction, compute_angular_error, compute_ssim, compute_psnr  # noqa: E402
from evaluate import (  # noqa: E402
    canonical_scene,
    direction_target_from_waypoints,
    parse_direction_angle_from_prompt,
    parse_scene_from_prompt,
)


def fold(a, b):
    return compute_angular_error(a, b)


def maybe_downscale(frames, win):
    """把 (T,H,W,C) 帧堆栈 INTER_AREA 降到窗口分辨率(只降不升)。

    [v2.2 标定修复] A 部分(真实 GT 地板)此前直接在 480x832 原始帧上跑
    估计器,GT 速度级位移 ~34 px/帧 × step=3 采样 ≈ 102 px/对,超出
    Farnebäck 量程 → 地板实测 82.2°(2026-07-23,与 Track B v1 gt_oracle
    80.67° 同一事故)。修复 = 与 A2/Track B v2 相同口径:一切光流先降到
    原生窗口分辨率再算。win=None 时原样返回(调用方需大声警告)。
    """
    if win is None:
        return frames
    win_h, win_w = int(win[0]), int(win[1])
    if frames.shape[1] <= win_h and frames.shape[2] <= win_w:
        return frames
    out = np.stack([
        cv2.resize(np.clip(f, 0, 255).astype(np.uint8), (win_w, win_h),
                   interpolation=cv2.INTER_AREA)
        for f in frames
    ])
    return out


def flow_magnitude(frames, max_pairs=8):
    """平均光流幅度(px/帧,与 estimate_motion_direction 同采样)。"""
    step = max(1, len(frames) // max_pairs)
    mags = []
    for i in range(0, len(frames) - step, step):
        g1 = cv2.cvtColor(np.clip(frames[i], 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(np.clip(frames[i + step], 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        f = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.sqrt(f[..., 0] ** 2 + f[..., 1] ** 2).mean()) / step)
    return float(np.mean(mags)) if mags else None


def bootstrap_ci(x, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if len(x) == 0:
        return [None, None]
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def stats_block(errs):
    e = np.asarray(sorted(errs.values() if isinstance(errs, dict) else errs), float)
    if len(e) == 0:
        return None
    k = max(1, int(0.05 * len(e)))
    return {"n": len(e), "mean": float(e.mean()), "median": float(np.median(e)),
            "trimmed10_mean": float(e[k:-k].mean()) if len(e) > 2 * k else float(e.mean()),
            "ci95": bootstrap_ci(e)}


def synthetic_translation_check(src_frames, n_frames=17, px_per_frame=6, window_size=None):
    """A2: 已知方向的人工平移视频 → 估计器恢复误差(预期≈0°)。

    用 GT 首帧作纹理源,向 8 个方向平移采样(镜像 pad 防黑边),
    合成"内容沿 θ 移动"的视频;估计器应恢复 θ 本身。

    px_per_frame 默认 6(小位移,Farnebäck winsize=15/levels=3 量程内)。
    [服务器标定] 旧默认 34 px/frame(480x832 生成分辨率的真实内容位移)超出
    Farnebäck 可靠量程,8 方向恢复误差 ~75° 全是垃圾值;减小逐帧位移后恢复
    误差降到 0.020°。大位移检查请配合 window_size 在原生分辨率下做。

    window_size=(win_h, win_w) 给定时,先把源帧 INTER_AREA 降到原生窗口分辨率
    (H5 f['metadata'].attrs['pose_window_size'])再合成/估计 —— 与 Track B v2
    的 native-window-resolution 光流口径一致。只降不升(帧已不大于窗口则原样)。
    """
    out = {
        "_px_per_frame": float(px_per_frame),
        "_window_size": [int(v) for v in window_size] if window_size is not None else None,
    }
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    errs_all = []
    for name, src in src_frames.items():
        if window_size is not None:
            win_h, win_w = int(window_size[0]), int(window_size[1])
            if win_h <= 0 or win_w <= 0:
                raise ValueError(f"window_size 必须为正: {window_size!r}")
            if src.shape[0] > win_h or src.shape[1] > win_w:
                src = cv2.resize(
                    np.clip(src, 0, 255).astype(np.uint8),
                    (win_w, win_h), interpolation=cv2.INTER_AREA,
                )
        h, w = src.shape[:2]
        pad = px_per_frame * n_frames + 8
        big = cv2.copyMakeBorder(src, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        per_angle = {}
        for th in angles:
            # 内容运动方向 θ(数学系,y 向上):内容向 θ 动 = 采样窗向 -θ 移
            dx = np.cos(np.radians(th)) * px_per_frame
            dy = -np.sin(np.radians(th)) * px_per_frame  # 像素 y 向下
            frames = []
            for t in range(n_frames):
                x0 = int(round(pad - dx * t)); y0 = int(round(pad - dy * t))
                frames.append(big[y0:y0 + h, x0:x0 + w])
            est = estimate_motion_direction(np.stack(frames))
            err = None if est is None else fold(est, float(th))
            per_angle[th] = err
            if err is not None:
                errs_all.append(err)
        out[name] = per_angle
    out["_mean_recovery_error"] = float(np.mean(errs_all)) if errs_all else None
    out["_expectation"] = "≈0-5°;若≈180° 说明估计器/合成之一方向定义又拧了"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--eval-dirs", nargs="*", default=[], help="name=dir(evaluate 输出目录,含 generated/*.npy)")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("-o", "--output", default="direction_validation.json")
    ap.add_argument("--fullres", action="store_true",
                    help="[legacy] 光流在原始分辨率上跑(复现 2026-07-22 的 "
                         "82.2° 量程事故口径,仅供对照;正式校准勿用)")
    args = ap.parse_args()

    val_idx = json.load(open(args.split))["val_indices"]
    if args.max_samples:
        val_idx = val_idx[: args.max_samples]

    out = {"n_val": len(val_idx)}
    # [v2.5] flood 更名 water;旧数据的 flood 经 canonical_scene 归一进 water 桶
    by_scene = lambda: {"urban": [], "rural": [], "general": [], "water": [], "?": []}

    gt_errC, gt_errCam = by_scene(), by_scene()
    gt_errC1 = by_scene()   # step=1 逐帧对口径(量程敏感性对照)
    gt_mag, p_ssim, p_psnr = by_scene(), by_scene(), by_scene()
    gt_targets, gt_scenes, syn_srcs = {}, {}, {}

    win = None  # 原生窗口尺寸 (win_h, win_w),供 A2 在窗口分辨率下校验
    with h5py.File(args.gt_h5, "r") as f:
        # [布局修复] canonical final H5 把样本放在 f['data'] 组
        # (build_dataset_from_benchmark.py 写入;evaluate.py:124-131 读的也是
        # f['data'])。旧代码只认 'samples'/根布局,在服务器上对 canonical H5
        # 直接崩,当场热修成先查 'data' —— 这里是该服务器热修的仓库正式版;
        # 'samples'/根布局仅作历史数据集的回退。
        if "data" in f:
            samples = f["data"]
        elif "samples" in f:
            samples = f["samples"]
        else:
            samples = f
        if "metadata" in f and "pose_window_size" in f["metadata"].attrs:
            win = tuple(int(v) for v in f["metadata"].attrs["pose_window_size"])
        # [v2.2] 一切光流的分辨率口径:默认原生窗口分辨率(与 A2/Track B v2
        # 一致);--fullres 仅供复现旧事故口径。窗口尺寸缺失又没显式 --fullres
        # 时大声警告 —— 全分辨率地板读数是已知垃圾(82.2°),不许静默产出。
        FS = None if args.fullres else win
        if FS is None and not args.fullres:
            print(
                "[WARN] H5 metadata 缺 pose_window_size —— 光流被迫在原始分辨率"
                "上跑,GT 地板读数不可信(已知 ~82° 量程垃圾)。该数据集需用新 "
                "builder 重建或先补 metadata。",
                file=sys.stderr,
            )
        out["_flow_resolution"] = (
            list(FS) if FS is not None else "native-full(量程外,读数不可信)"
        )
        keys = sorted(samples.keys(), key=lambda k: int(k.split("_")[-1]))
        for i in val_idx:
            s = samples[keys[i]]
            frames = s["video_frames"][:]
            prompt = s.attrs.get("prompt", "")
            scene = canonical_scene(
                s.attrs.get("source_scene_category", "")
                or parse_scene_from_prompt(prompt) or "?")
            scene = scene if scene in gt_errC else "?"
            tgtC = parse_direction_angle_from_prompt(prompt)          # 内容角(修正约定)
            if tgtC is None and "waypoints" in s:
                # [v2.5] direction-stripped 数据:目标从 waypoints 推导(与建库量化一致)
                tgtC = direction_target_from_waypoints(s["waypoints"][:])
            tgtCam = None if tgtC is None else (tgtC + 180.0) % 360.0  # 旧约定对照
            fr_flow = maybe_downscale(frames, FS)   # [v2.2] 光流一律窗口口径
            est = estimate_motion_direction(fr_flow)
            if est is not None and tgtC is not None:
                gt_errC[scene].append(fold(est, tgtC))
                gt_errCam[scene].append(fold(est, tgtCam))
            # step=1 逐帧对读数:窗口口径下默认 step=3 每对 ~27 px 已达
            # Farnebäck 量程边缘;两个读数若显著分离 → 说明 step 采样本身
            # 在啃量程,指标定义须改 step=1 而非再降分辨率。
            est1 = estimate_motion_direction(fr_flow, step=1)
            if est1 is not None and tgtC is not None:
                gt_errC1[scene].append(fold(est1, tgtC))
            m = flow_magnitude(fr_flow)
            if m is not None:
                gt_mag[scene].append(m)
            src = frames[0]
            p_ssim[scene].append(float(np.mean([compute_ssim(src, fr) for fr in frames[1:]])))
            p_psnr[scene].append(float(np.mean([compute_psnr(src, fr) for fr in frames[1:]])))
            gt_targets[keys[i]] = tgtC
            gt_scenes[keys[i]] = scene
            if scene not in syn_srcs and scene != "?":
                syn_srcs[scene] = src

    flat = lambda d: [v for vs in d.values() for v in vs]
    out["A_gt_floor"] = {
        "_flow_resolution": out["_flow_resolution"],
        "vs_content_target": stats_block(flat(gt_errC)),
        "vs_content_target_step1": stats_block(flat(gt_errC1)),
        "vs_camera_target_mean": float(np.mean(flat(gt_errCam))) if flat(gt_errCam) else None,
        "per_scene_vs_content": {k: stats_block(v) for k, v in gt_errC.items() if v},
        "per_scene_vs_content_step1": {k: stats_block(v) for k, v in gt_errC1.items() if v},
        "expectation": "content≈22.5°(量化地板), camera≈157.5°;某场景显著偏离→该场景光流不可靠;"
                       "step1 与默认读数显著分离→step 采样在啃量程,指标定义应改 step=1",
    }
    # A2 在原生窗口分辨率(若 metadata 有 pose_window_size)+ 小位移(6 px/frame)
    # 下校验估计器 —— 与 Track B v2 的 native-res flow 口径一致。
    out["A2_synthetic_translation"] = synthetic_translation_check(syn_srcs, window_size=win)
    out["B_flow_mag_gt"] = {"_flow_resolution": out["_flow_resolution"],
                            "overall": float(np.mean(flat(gt_mag))),
                            "per_scene": {k: float(np.mean(v)) for k, v in gt_mag.items() if v}}
    out["D_persistence_baseline"] = {
        "ssim_mean": float(np.mean(flat(p_ssim))), "psnr_mean": float(np.mean(flat(p_psnr))),
        "per_scene_ssim": {k: float(np.mean(v)) for k, v in p_ssim.items() if v},
        "note": "所有帧=源帧;与 tab:main SSIM 对照",
    }

    # ---- 各 eval 配置:角误差(默认 step + step=1)+ 流幅度 + 异常轨迹 ----
    # [v2.3] step=1 是正式口径:GT 地板实测 默认step(=3,每对~27px)下
    # 82.67° / step=1 下 35.93°(median 24.45°≈量化地板)—— 默认 step 在窗口
    # 分辨率仍啃量程,只留作对照;方向表/显著性以 *_step1 为准。
    per_cfg, per_cfg1 = {}, {}
    for spec in args.eval_dirs:
        name, d = spec.split("=", 1)
        gen = Path(d) / "generated"
        errs, errs1, mags = {}, {}, []
        for npy in sorted(gen.glob("*.npy")):
            key = npy.stem
            if key not in gt_targets or gt_targets[key] is None:
                continue
            frames = maybe_downscale(np.load(npy), FS)  # [v2.2] 与 GT 地板同口径
            est = estimate_motion_direction(frames)
            if est is not None:
                errs[key] = fold(est, gt_targets[key])
            est1 = estimate_motion_direction(frames, step=1)
            if est1 is not None:
                errs1[key] = fold(est1, gt_targets[key])
            m = flow_magnitude(frames)
            if m is not None:
                mags.append(m)
        if not errs and not errs1:
            print(f"[warn] {name}: {gen} 无可用样本,跳过", file=sys.stderr)
            continue
        per_cfg[name] = errs
        per_cfg1[name] = errs1
        flow_mean = float(np.mean(mags)) if mags else None
        for label, e in (("angular_by_config", errs),
                         ("angular_by_config_step1", errs1)):
            if not e:
                continue
            top = sorted(e.items(), key=lambda kv: -kv[1])[:5]
            scene_errs = {}
            for k, v in e.items():
                scene_errs.setdefault(gt_scenes.get(k, "?"), []).append(v)
            out.setdefault(label, {})[name] = {
                "_flow_resolution": out["_flow_resolution"],
                **stats_block(e),
                "flow_mag_mean": flow_mean,
                "per_scene": {s: stats_block(v) for s, v in scene_errs.items() if v},
                "top5_outliers": [{"key": k, "err": round(v, 1)} for k, v in top],
            }

    # ---- C: 两两配对检验(默认 step 作对照 + step=1 正式)----
    try:
        from scipy import stats as sps
    except ImportError:
        sps = None
    from itertools import combinations
    for label, cfgs in (("C_paired_tests", per_cfg),
                        ("C_paired_tests_step1", per_cfg1)):
        for a, b in combinations(cfgs, 2):
            shared = sorted(set(cfgs[a]) & set(cfgs[b]))
            if len(shared) < 10:
                continue
            da = np.array([cfgs[a][s] for s in shared])
            db = np.array([cfgs[b][s] for s in shared])
            rec = {"n_paired": len(shared), "mean_diff": float((da - db).mean()),
                   "diff_ci95": bootstrap_ci(da - db)}
            if sps:
                t, p = sps.ttest_rel(da, db)
                rec.update({"paired_t": float(t), "p": float(p),
                            "wilcoxon_p": float(sps.wilcoxon(da, db).pvalue)})
            out.setdefault(label, {})[f"{a}_vs_{b}"] = rec

    json.dump(out, open(args.output, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[done] → {args.output}")


if __name__ == "__main__":
    main()
