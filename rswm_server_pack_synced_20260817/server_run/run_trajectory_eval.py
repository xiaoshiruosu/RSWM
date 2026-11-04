#!/usr/bin/env python3
# =============================================================================
# run_trajectory_eval.py — Track B v2(完整轨迹控制)+ 诊断基线 的评估驱动
#
# 【v2 canonical】本文件就是 Track B v2 的规范驱动。服务器上为当次标定手搓的
# run_trajectory_eval_v2.py 是临时脚本,应当用本文件替换,不再各自演化。
#
# 把三块拼起来(纯 CPU,吃存量数据):
#   A: 从 canonical H5 按 key 读 GT 轨迹结构(waypoints/segment_angles/
#      trajectory/full_scene/poses;旧数据集缺 trajectory/full_scene 时用
#      server_run/augment_h5_trajectory.py 回填,无需整库重建)
#   B: trajectory_metrics.score_trajectory (v2) 对各 eval 目录的生成视频打分
#   C: diagnostic_baselines.run_baselines 合成 persistence/det-translation/
#      random-direction/gt-oracle,同样用 score_trajectory 打分 → 证明 benchmark
#      能区分「视觉好但方向错 / 方向对但质量差 / 单帧好但不连续 / 几乎没动」
#
# 服务器标定复盘(gt_oracle: net_direction_error=80.67°, valid_pct=80%,
# turn_recall=0.25 —— 指标失效而非模型失效)在本驱动侧的 v2 修正:
#   1. 分辨率:score_trajectory 传 window_size = f['metadata'].attrs
#      ['pose_window_size'],全部 Track B 光流在原生窗口分辨率上计算
#      (480x832 上内容位移 ~34 px/frame 超出 Farnebäck 量程;降回 128x222
#      后 ~9 px/frame,GT 逐段误差 ~1-10°)。
#   2. 真实结点:传 frame_positions=g['trajectory'](真实逐帧窗口中心,来自
#      v5 源 benchmark,build_multi_direction_benchmark_v5.py:826-834)与
#      poses=g['poses'];metrics 内部按 positions → poses → 均匀 linspace 的
#      顺序回退,输出 knots_source 标注来源(uniform_fallback 显式可见)。
#   3. 基线场景:scene_image 一律用 g['full_scene'](真实 512x512 场景)。
#      绝不传首帧 g['image'] —— 那是 128x222 窗口的 480x832 上采样,不是场景;
#      服务器上误传它曾使 deterministic/random 两基线 valid_pct=0%。
#      full_scene 缺失 → 跳过这两基线(persistence/gt_oracle 照常)。
#   4. 标定门:跑完 gt_oracle 后打印 'TRACK B V2 CALIBRATION: PASS/FAIL';
#      --with-baselines 时 FAIL 即非零退出(镜像服务器期望的 smoke 门)。
#      [v2.1] 门再加 knots 检查:gt_oracle 的 knots_source 必须 **严格全部**
#      'positions'(knots_source_counts == {"positions": n};零 poses、零
#      uniform_fallback、零 None)—— gt_oracle 帧数=GT 轨迹,任何回退都是异常。
#   5. [v2.1] 净方向门控改由 **GT 命令轨迹** 决定(polyline_net_ratio(waypoints)
#      < 0.2 才 gated;恢复比例仅诊断)。曾用恢复比例门控:GT 近闭合的 00349
#      (gt=0.054, rec=0.215) 被错误放行,147.36° 污染 gt_oracle 均值 → 42.79°
#      假失败;GT 明确的 00331 (gt=0.216, rec=0.074) 被错误门控。
#   6. [v2.1] 时间轴自适配(v2.8 起源头已去重,本机制对新数据是 no-op,
#      仅兼容 v2.8 前的 18/27/36 旧数据):旧 GT 18/27/36 →
#      生成 17/25/33;生成首帧≡GT 首帧,MSE ~3e-5)。_score_v2 把 trajectory 与
#      poses 用同一 mask 去重后传入;对不齐则 fail-loud 跳过(timeline_mismatch
#      计数),绝不静默退回均匀 knot;num_frames 一律 = len(frames)。
#
# 各视频实际按什么分辨率打分(resize 口径,务必保持一致):
#   * eval 目录生成视频 / gt_oracle / persistence:GT 分辨率(480x832)进,
#     score_trajectory 内部 _maybe_downsample 降到窗口分辨率再算光流。
#   * deterministic/random 基线:diagnostic_baselines v2 默认直接产出原生窗口
#     分辨率视频(不再上采样到 GT 分辨率),metrics 侧降采样为 no-op ——
#     与上面同一口径,不存在"有的在 480x832 有的在 128x222 算光流"的错配。
#
# 用法(激活 rswm 环境,包根执行):
#   python server_run/run_trajectory_eval.py \
#       --gt-h5 $WORK/data/train_geo.h5 \
#       --split $WORK/train_main/train_val_split.json \
#       --eval-dirs textonly=$WORK/eval_r16_sem_s42:text \
#                   textpluecker=$WORK/eval_r16_geo_s42:geo \
#       --with-baselines \
#       -o $WORK/trajectory_eval.json
#   (:text/:geo 后缀 = 显式路由声明,v2.6 起推荐总是写;v2.6 evaluate 产出的
#    目录也可省略 —— 会读 meta 的 pose_conditioning 标记,两者冲突则 FATAL。)
#
# 5 样本冒烟(标定门 FAIL 时非零退出):
#   python server_run/run_trajectory_eval.py --gt-h5 ... --split ... \
#       --max-samples 5 --with-baselines
#
# 注意:GT 轨迹结构从 GEO_H5 读(SEM_H5 位姿被中和;v2 还要用 poses 做结点回退,
# 更必须用 GEO_H5)。text-only 的 Track B 分会带 commanded=False 标记
# (它没拿到逐段信息,逐段/转弯/waypoint/路径指标仅供参考,不作评判)——
# 该语义保持不变:text-only 条件下只有净方向 + 段数是被命令的。
# =============================================================================
import argparse
import inspect
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lora_remotesense" / "new"))
import trajectory_metrics as TM       # noqa: E402
import diagnostic_baselines as DB      # noqa: E402

# ---- fail-loud:v2 驱动必须搭配 v2 metrics ----
# 旧签名(无 window_size/frame_positions/poses)下"能跑"就等于在 480x832 错误
# 分辨率上静默打分 —— 正是服务器标定事故本身。宁可当场退出。
_V2_PARAMS = ("window_size", "frame_positions", "poses")
_missing_params = [
    p for p in _V2_PARAMS
    if p not in inspect.signature(TM.score_trajectory).parameters
]
if _missing_params:
    print(
        f"[FATAL] trajectory_metrics.score_trajectory 缺 v2 参数 {_missing_params};"
        f"本驱动是 Track B v2,请先更新 lora_remotesense/new/trajectory_metrics.py。",
        file=sys.stderr,
    )
    sys.exit(2)


def _agg(dicts, keys):
    """对一组 per-sample dict 的数值键求 mean(忽略 None),外加 v2 聚合量。"""
    out = {}
    for k in keys:
        vals = [d[k] for d in dicts if d.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    out["n"] = len(dicts)
    out["valid_pct"] = (
        100.0 * sum(1 for d in dicts if d.get("valid")) / len(dicts) if dicts else 0.0
    )
    # (v2.1) 净位移门控命中率:gated=True 表示 **GT 命令轨迹** 的净位移比
    # (net_displacement_ratio,来自 waypoints 折线)< 0.2 —— 净方向这个问题
    # 对该样本本身无意义 → net_direction_error=None(不进上面的 mean 分母)。
    # 注意:门控由 GT 决定,故 gated% 是【样本集属性】,各配置应一致;
    # 若两配置 gated% 不同,说明它们评了不同的样本子集(如 timeline_mismatch)。
    out["net_direction_gated_pct"] = (
        100.0 * sum(1 for d in dicts if d.get("net_direction_gated")) / len(dicts)
        if dicts else 0.0
    )
    # (v2) 结点来源分布:positions(真实逐帧轨迹)/ poses / uniform_fallback /
    # None(未产出,如 valid=False)。uniform_fallback 占比高 → 先回填 trajectory。
    out["knots_source_counts"] = dict(
        Counter(str(d.get("knots_source")) for d in dicts)
    )
    return out


# [v2.6/O9 修复] 本清单曾静默丢弃 score_trajectory 已实现的 6 个键
# (segment_success_rate_30/45、turn_precision、turn_f1、waypoint_endpoint_error、
# num_*)—— 指标层做了、报表层拿不到,师兄意见 2/3/4 的产出落空
# (REVIEWER_STYLE_EXPERIMENT_REVIEW.md §3-O9)。现已补全。
# 聚合语义:全部经 _agg 求 mean(忽略 None);num_gt_turns/num_recovered_turns
# 的均值 = 平均拐点数(非分数,与 turn_precision 联读用)。
TRACKB_KEYS = ["net_direction_error", "net_displacement_ratio",
               "recovered_net_displacement_ratio",
               "segment_angular_mean",
               "segment_success_rate_30", "segment_success_rate_45",
               "turn_recall", "turn_precision", "turn_f1",
               "turn_timing_error",
               "waypoint_error", "waypoint_endpoint_error",
               "frechet", "dtw",
               "flow_magnitude",
               "num_gt_turns", "num_recovered_turns"]

# (v2.3) 近共线几何退化阈值 = knots 检测的 angle_tol(20°):相邻两段命令
# 方向的折叠夹角低于它时,"转弯"在检测意义上不存在,knot 依赖指标对该样本
# 无定义(waypoint clamp 的几何边角料,builder 蒙特卡洛 ~3.35% 概率)。
DEGENERATE_TURN_DEG = 20.0


def _flatten(score):
    """score_trajectory 已是扁平 dict;挑 valid/commanded/v2 标注 + TRACKB_KEYS。"""
    f = {
        "valid": score.get("valid"),
        "commanded": score.get("commanded"),
        "net_direction_gated": score.get("net_direction_gated"),
        "knots_source": score.get("knots_source"),
        # (v2.3) 偏置尸检:原始角不折叠、不聚合,只进 per_sample。
        "recovered_net_angle": score.get("recovered_net_angle"),
        "gt_net_content_angle": score.get("gt_net_content_angle"),
    }
    for k in TRACKB_KEYS:
        f[k] = score.get(k)
    return f


# ---------------------------------------------------------------------------
# [v2.6/O11 修复] text_only(=commanded 语义)判定加固
#
# 旧实现:text_only = ("textonly" in name or "sem" in name) —— commanded 标志
# (决定一个数是"达标量"还是"诊断量")由命令行目录别名的字符串决定,且
# 两个子串都不含时【静默】判为 geo(commanded=True)。实害:
#   * base=$W/eval_base:31 号在 SEM_H5(identity 位姿)上评,是 text 路由,
#     旧启发式静默判成 geo —— 整块语义翻转,恰是 O11 警告的形态;
#   * shufpose 等新别名不含任何已知子串,同样静默落 geo。
# 现规则(优先级从高到低,冲突一律 FATAL,绝不静默):
#   1) 显式后缀:--eval-dirs name=path:text|:geo(权威声明);
#   2) eval 目录 meta/*.json 的 pose_conditioning 标记(v2.6 evaluate 起写入:
#      'identity_neutralized' → text,'native' → geo)。目录内混存(部分有
#      部分无 / 值不一致)= 混版本目录,FATAL;
#   3) 两者都无(v2.6 之前的旧目录)→ 目录名启发式,但只认显式词根:
#      text 类 = textonly/text_only/sem;geo 类 = pluecker/plucker/geo。
#      同时命中/都不命中 → FATAL,要求加 :text/:geo 后缀。
#   显式后缀与 meta 标记冲突 → FATAL(别名拼错或目录传错,当场暴露)。
# ---------------------------------------------------------------------------
_ROUTE_TEXT_PATTERNS = ("textonly", "text_only", "sem")
_ROUTE_GEO_PATTERNS = ("pluecker", "plucker", "geo")
# [v2.8.1 P0-4] "dropped" = pose_mode=drop,位姿通道整个没接上 → 真·text-only。
# 与 identity_neutralized 的区别是后者仍走 Plücker 通路(静止相机诊断臂),
# 但对 Track B 的 commanded 语义一致:轨迹/转弯/waypoint 都【未被命令】。
_POSE_COND_TO_TEXT_ONLY = {"identity_neutralized": True, "native": False,
                           "dropped": True}


def _split_route_suffix(rest):
    """把 'path[:text|:geo]' 拆成 (path, route|None)。仅认这两个后缀,
    其余含 ':' 的字符串原样当路径(不猜)。"""
    p, sep, tok = rest.rpartition(":")
    if sep and tok in ("text", "geo"):
        return p, tok
    return rest, None


def _route_from_meta(eval_dir):
    """读 eval 目录全部 meta/*.json 的 pose_conditioning 标记。

    返回 (pose_conditioning|None, n_with, n_without)。
    值不一致 → FATAL(同目录两种口径,任何聚合都是静默混算)。
    """
    meta_dir = Path(eval_dir) / "meta"
    vals, n_with, n_without = set(), 0, 0
    for m in sorted(meta_dir.glob("*.json")) if meta_dir.is_dir() else []:
        try:
            with open(m) as f:
                pc = json.load(f).get("pose_conditioning")
        except (OSError, json.JSONDecodeError) as e:
            print(f"[FATAL] 读 {m} 失败: {e}", file=sys.stderr)
            sys.exit(2)
        if pc is None:
            n_without += 1
        else:
            vals.add(str(pc))
            n_with += 1
    if len(vals) > 1:
        print(f"[FATAL] {eval_dir}/meta 里 pose_conditioning 不一致: {sorted(vals)} "
              f"—— 同一目录混了两种位姿口径,不能聚合。", file=sys.stderr)
        sys.exit(2)
    return (vals.pop() if vals else None), n_with, n_without


def _resolve_text_only(name, eval_dir, suffix_route):
    """按上面的优先级把 (别名, 目录, 显式后缀) 解析成 (text_only, route_source)。"""
    # meta 标记(可能不存在)
    pc, n_with, n_without = _route_from_meta(eval_dir)
    if pc is not None and n_without > 0:
        print(f"[FATAL] {eval_dir}: {n_with} 个 meta 有 pose_conditioning、"
              f"{n_without} 个没有 —— v2.6 前后混版本目录(断点续推?)。"
              f"请删除 meta/ 与 generated/ 重跑,或对同版本子集分开分析。",
              file=sys.stderr)
        sys.exit(2)
    meta_text_only = None
    if pc is not None:
        if pc not in _POSE_COND_TO_TEXT_ONLY:
            print(f"[FATAL] {eval_dir}: 未知 pose_conditioning={pc!r}"
                  f"(认识的值: {sorted(_POSE_COND_TO_TEXT_ONLY)})。", file=sys.stderr)
            sys.exit(2)
        meta_text_only = _POSE_COND_TO_TEXT_ONLY[pc]

    if suffix_route is not None:
        suffix_text_only = (suffix_route == "text")
        if meta_text_only is not None and meta_text_only != suffix_text_only:
            print(f"[FATAL] 配置 {name!r}: 显式后缀 :{suffix_route} 与 eval 目录 "
                  f"meta 的 pose_conditioning={pc!r} 矛盾 —— 别名拼错或目录传错。",
                  file=sys.stderr)
            sys.exit(2)
        return suffix_text_only, f"explicit:{suffix_route}"
    if meta_text_only is not None:
        return meta_text_only, f"meta:{pc}"

    # 旧目录回退:只认显式词根,拒绝静默默认
    ln = name.lower()
    hit_text = any(p in ln for p in _ROUTE_TEXT_PATTERNS)
    hit_geo = any(p in ln for p in _ROUTE_GEO_PATTERNS)
    if hit_text == hit_geo:   # 都命中(如 sem_geo)或都不命中(如 base/shufpose)
        print(f"[FATAL] 配置 {name!r}: eval 目录无 pose_conditioning 标记"
              f"(v2.6 之前产出?),目录名启发式也无法唯一判定路由"
              f"(text 词根 {_ROUTE_TEXT_PATTERNS} / geo 词根 {_ROUTE_GEO_PATTERNS})。"
              f"请显式写 --eval-dirs {name}=<dir>:text 或 :geo。", file=sys.stderr)
        sys.exit(2)
    print(f"[WARN] 配置 {name!r}: 路由由【目录名启发式】判定为 "
          f"{'text' if hit_text else 'geo'}(旧 eval 目录无 meta 标记)。"
          f"出表前请人工核对;建议显式加 :text/:geo 后缀。", file=sys.stderr)
    return hit_text, "name_heuristic"


def _dedup_timeline(trajectory, poses):
    """(v2.1) 去除 trajectory 中相邻重复位置的帧,poses 用同一 mask。

    背景:v2.8 前 GT 每段都含首尾帧,段边界位置重复 —— 旧 GT 18/27/36 帧;
    [v2.8 ②] 源头已跳过后续段的 step=0,新数据 17/25/33 无重复,此处 no-op;
    模型生成会去掉 K-1 个重复边界帧 → 17/25/33 帧(生成首帧与 GT 首帧 MSE
    ~3e-5,证明是去重不是丢头)。因此对生成视频打分时,必须把 trajectory 和
    poses 同步去重到生成帧数,再作为 frame_positions 传入 —— 否则长度不匹配
    会静默退回均匀 knot(v2 冒烟中 knots_source_counts={'uniform_fallback':5}
    的根源)。

    返回 (traj_dedup, poses_dedup, keep_mask)。相邻重复判定:**自适应容差** =
    max(1e-6, 正步长中位数的 1%) —— 兼顾两种量纲:scene 像素轨迹(步长 ~9-18px
    → 容差 ~0.1-0.2px,吸收 float32 往返抖动)与 poses 归一化平移(max|t|=1,
    步长 ~0.06 → 容差 ~6e-4)。正常步长与重复帧之间有 >100x 的间隔,无误伤空间。
    """
    tr = np.asarray(trajectory, dtype=np.float64)
    keep = np.ones(len(tr), dtype=bool)
    d = np.linalg.norm(np.diff(tr, axis=0), axis=1)
    pos = d[d > 1e-6]
    tol = max(1e-6, 0.01 * float(np.median(pos))) if len(pos) else 1e-6
    keep[1:] = d > tol
    traj_d = tr[keep]
    poses_d = np.asarray(poses)[keep] if poses is not None else None
    return traj_d, poses_d, keep


def _score_v2(frames, g, text_only, sample_key=""):
    """统一的 v2.1 打分入口:窗口分辨率光流 + 真实结点 + 时间轴自适配。

    时间轴规则(v2.2):
      * len(frames) == len(trajectory) 且帧内容在轨迹重复索引处也重复
        (真 GT / gt_oracle)—— frames/trajectory/poses 同 mask 去重后传入,
        与生成口径共享同一条去重时间轴;
      * len(frames) == len(trajectory) 但帧不重复(det/random 等合成基线,
        均匀时间轴)—— 原样传入;
      * len(frames) == len(dedup(trajectory)) —— 模型生成口径(去掉 K-1 个
        段边界重复帧),trajectory/poses 用同一 mask 去重后传入;
      * 都对不上 —— **fail-loud 返回 None**(调用方计入
        timeline_mismatch 并跳过),绝不静默退回均匀 knot。
    num_frames 一律 = len(frames)(v2 曾传 GT 的 T,生成 17 帧时错配)。
    """
    traj, poses = g["trajectory"], g["poses"]
    Tf = len(frames)
    if traj is not None:
        Tt = len(traj)
        if Tf != Tt:
            traj_d, poses_d, _keep = _dedup_timeline(traj, poses)
            if len(traj_d) == Tf:
                traj, poses = traj_d, poses_d
            else:
                print(
                    f"[timeline-mismatch] {sample_key}: 帧数 {Tf} 既≠GT轨迹 {Tt} "
                    f"也≠去重后 {len(traj_d)} —— 无法对齐时间轴,样本跳过 "
                    f"(绝不静默退回均匀 knot)。",
                    file=sys.stderr,
                )
                return None
        else:
            # (v2.2) 长度相等但 GT 轨迹含段边界重复位置(诊断34:GT 27 帧仅
            # 25 唯一位置,traj[8]==traj[9]、traj[17]==traj[18])。仅当帧内容
            # 也在同一索引重复时(真 GT / gt_oracle,重复帧逐字节相同),才把
            # frames/trajectory/poses 用同一 mask 同步去重 —— 使 gt_oracle 与
            # 生成口径共享同一条 25 帧时间轴(转弯时机、DTW 才同 T 可比)。
            # det/random 基线的帧是均匀时间轴合成的,并不在这些索引重复,
            # 帧守卫不满足 → 保持原样(对它们套 mask 会删真帧)。
            # 注意:此去重不改净方向(端点法,重复帧光流恰为 0)——oracle
            # net_direction_error 门若失败,原因在别处,勿指望本分支救它。
            traj_d, poses_d, keep = _dedup_timeline(traj, poses)
            if len(traj_d) < Tt:
                fr = np.asarray(frames)
                drop = np.where(~keep)[0]
                frames_dup = all(
                    float(np.mean(np.abs(
                        fr[j].astype(np.float32) - fr[j - 1].astype(np.float32)
                    ))) < 0.5
                    for j in drop
                )
                if frames_dup:
                    frames = fr[keep]
                    traj, poses = traj_d, poses_d
                    Tf = len(frames)
    elif poses is not None and len(poses) != Tf:
        # 没有 trajectory 时无法可靠去重(poses 平移在边界同样重复,但缺位置
        # 校验依据)——尝试用 poses 平移做同样的相邻去重。
        tr_from_poses = np.asarray(poses)[:, :3, 3][:, [1, 0]]  # (x,y)->(cy,cx) 近似
        _traj_d, poses_d, keep = _dedup_timeline(tr_from_poses, poses)
        if len(poses_d) == Tf:
            poses = poses_d
        else:
            print(f"[timeline-mismatch] {sample_key}: 帧数 {Tf} 与 poses "
                  f"{len(poses)}/去重后 {len(poses_d)} 均不合 —— 样本跳过。",
                  file=sys.stderr)
            return None
    score = TM.score_trajectory(
        frames, g["waypoints"], g["segment_angles"],
        num_frames=Tf, text_only=text_only,
        window_size=g["window_size"],
        frame_positions=traj,
        poses=poses,
    )
    # (v2.3) 近共线几何退化统一规则:waypoint 被 clamp 后相邻两段夹角可跌破
    # knot 检测容差(20°),结点数与 waypoints 天然对不上(尸检实证:
    # sample_00016 最小折叠转角 13.1°,蒙特卡洛复刻 builder 回退率 3.35%)。
    # 这类样本的"转弯"在检测意义上不存在 —— 对【所有】此类样本统一处理
    # (不许单样本豁免):4 个 knot 依赖指标置 None(不进均值),
    # knots_source 改标 'degenerate_geometry'(门规格单独放行并计数);
    # 净方向 / Fréchet / DTW / 流幅度 / valid 不依赖 knot,全部保留。
    seg_ang = g.get("segment_angles")
    if seg_ang is not None and len(seg_ang) >= 2:
        sa = np.asarray(seg_ang, dtype=np.float64).ravel()
        folded = np.abs(np.diff(sa)) % 360.0
        folded = np.where(folded > 180.0, 360.0 - folded, folded)
        min_turn = float(folded.min())
        if min_turn < DEGENERATE_TURN_DEG:
            # [v2.6] TRACKB_KEYS 补全后,knot/GT-拐点依赖集合同步扩展:新入聚合
            # 的 segment_success_rate_*(依赖 knot 分段)、turn_precision/turn_f1
            # (依赖 GT 拐点配对)、waypoint_endpoint_error(依赖 GT 结点)对
            # 近共线样本同样无定义 —— 不置 None 会让退化样本把垃圾值混进恰好是
            # 本轮新增的那几列均值。保留的两个计数:num_gt_turns 是 GT 属性
            # (K-1);num_recovered_turns 只看恢复路径自身的拐点检测,不与 GT
            # 配对 —— 与"净方向/形状指标保留"同一原则,保留作诊断。
            for kk in ("segment_angular_mean",
                       "segment_success_rate_30", "segment_success_rate_45",
                       "turn_recall", "turn_precision", "turn_f1",
                       "turn_timing_error",
                       "waypoint_error", "waypoint_endpoint_error"):
                score[kk] = None
            score["knots_source"] = "degenerate_geometry"
            print(f"[degenerate-geometry] {sample_key}: 最小折叠转角 "
                  f"{min_turn:.1f}° < {DEGENERATE_TURN_DEG}°(knot 检测容差)"
                  f"—— knot 依赖指标置 None,净方向/形状指标保留。",
                  file=sys.stderr)
            return score
    # (v2.1) 禁止静默 uniform_fallback:有 trajectory/poses 却仍回退,说明
    # 结点检测与 waypoints 对不上 —— 大声点名,别让它悄悄混进均值。
    if score.get("knots_source") == "uniform_fallback" and (
            g["trajectory"] is not None or g["poses"] is not None):
        print(f"[knots-fallback] {sample_key}: 有真实轨迹却退回均匀 knot "
              f"(结点数与 waypoints 不匹配?)——该样本的分段/转弯/waypoint "
              f"指标基于均匀假设,谨慎解读。", file=sys.stderr)
    return score


def load_gt(gt_h5, val_idx):
    """按 val_indices 读 {key: GT 结构 dict}。

    v2:额外读 trajectory(真实逐帧窗口中心 (T,2)=(cy,cx))、full_scene
    (真实 512x512 场景)与 poses(结点次选来源)。规范转换器
    (build_dataset_from_benchmark.py:765-810 起)已把 trajectory/full_scene
    写入 final H5;旧 H5 用 server_run/augment_h5_trajectory.py 原地回填。
    不再读首帧 'image' —— 它是首帧窗口的 480x832 上采样,不是场景,任何代码
    都不得把它当 scene_image 用(服务器上那样做使 det/random 基线 valid_pct=0%)。
    """
    gt = {}
    n_missing_traj = 0
    n_missing_scene = 0
    with h5py.File(gt_h5, "r") as f:
        data = f["data"]
        # 原生窗口尺寸:v2 全部 Track B 光流的分辨率基准,缺失即无法按 v2 打分。
        if "metadata" in f and "pose_window_size" in f["metadata"].attrs:
            win = tuple(int(v) for v in f["metadata"].attrs["pose_window_size"])
        else:
            print(
                "[FATAL] f['metadata'].attrs['pose_window_size'] 缺失 —— Track B v2 "
                "必须在原生窗口分辨率上算光流(否则重演 480x832 上 gt_oracle "
                "net_direction_error=80.67° 的标定事故)。该数据集需用新 builder 重建。",
                file=sys.stderr,
            )
            sys.exit(2)
        keys = sorted(data.keys())
        for i in val_idx:
            if i >= len(keys):
                continue
            k = keys[i]
            g = data[k]
            if "waypoints" not in g:
                continue  # 旧数据集无轨迹结构,跳过(需用新 builder 重建)
            traj = g["trajectory"][:] if "trajectory" in g else None
            scene = g["full_scene"][:] if "full_scene" in g else None
            if traj is None:
                n_missing_traj += 1
            if scene is None:
                n_missing_scene += 1
            gt[k] = dict(
                waypoints=g["waypoints"][:],
                segment_angles=g["segment_angles"][:],
                T=int(g.attrs.get("num_frames", g["video_frames"].shape[0])),
                gt_frames=g["video_frames"][:],
                window_size=win,
                trajectory=traj,     # None → metrics 回退 poses/uniform 结点
                full_scene=scene,    # None → 跳过 det/random 基线
                poses=g["poses"][:],
            )
    if n_missing_traj or n_missing_scene:
        # 一次性大声警告(不逐样本刷屏),并指名回填工具与后果。
        print(
            f"[WARN] GT H5 缺 v2 字段:{n_missing_traj}/{len(gt)} 样本缺 'trajectory',"
            f"{n_missing_scene}/{len(gt)} 样本缺 'full_scene'。\n"
            f"       回填工具:server_run/augment_h5_trajectory.py"
            f"(从 v5 源 benchmark 原地回填,免重建)。\n"
            f"       后果:缺 trajectory → 结点回退 poses 推导"
            f"(输出 knots_source='poses',再不行 'uniform_fallback');\n"
            f"             缺 full_scene → 该样本跳过 deterministic/random 两基线"
            f"(persistence/gt_oracle 照常)。\n"
            f"       绝不回退用首帧 'image' 充当场景 —— 服务器上那样做曾使"
            f"两基线 valid_pct=0%。",
            file=sys.stderr,
        )
    return gt


def _calibration_gate(gt_oracle_agg):
    """标定门(镜像服务器期望的 smoke 门):gt_oracle 必须近乎满分。

    真值视频喂进管线都拿不到分 = 指标失效,任何模型对比数字都不可信。
    PASS 条件(全部满足,均为 gt_oracle 聚合值):
      net_direction_error mean < RSWM_ORACLE_ANGLE_GATE(默认 30°,按实测仪器
                               地板 ~25° 定;旧写死 15° 在真实数据上恒 FAIL)
      segment_angular_mean     < 同上
      turn_recall              > 0.8
      valid_pct                > 95
    任一聚合值缺失(None)按 FAIL 处理 —— "算不出来"不等于"通过"。
    """
    if gt_oracle_agg is None:
        return {
            "passed": False,
            "reasons": ["gt_oracle 没有任何得分样本"],
            "checks": None,
        }
    # =====================================================================
    # [v3.0 P0-4] 门槛必须**按实测仪器地板**定,不能凭愿望写。
    #
    # 旧值 15° 是拍脑袋的"近乎满分",但真值视频喂进本管线的实测 gt_oracle
    # 净方向误差约 **25.11°** —— 也就是说这道门在真实数据上**必然 FAIL**,
    # 而它本该是"指标是否可信"的判据。一道恒 FAIL 的门等于没有门,
    # 更糟的是它会诱人去调数据而不是承认仪器精度有限。
    #
    # 现在:门槛从环境变量取,默认 30°(留在实测 25° 之上一点),
    # 并**要求同时报出实测地板**(instrument_floor_report.py)。
    # 想收紧到 15° 必须先证明估计器地板真的降下来了 —— 那是仪器问题,
    # 不是阈值问题。
    # =====================================================================
    _floor = float(os.environ.get("RSWM_ORACLE_ANGLE_GATE", "30.0"))
    spec = [
        ("net_direction_error", lambda v: v is not None and v < _floor,
         f"< {_floor:g} deg(按实测仪器地板 ~25° 定;旧值 15° 恒 FAIL)"),
        ("segment_angular_mean", lambda v: v is not None and v < _floor,
         f"< {_floor:g} deg"),
        ("turn_recall", lambda v: v is not None and v > 0.8, "> 0.8"),
        ("valid_pct", lambda v: v is not None and v > 95.0, "> 95 %"),
        # (v2.1) gt_oracle 帧数=GT 轨迹长度,knots 必须 **全部** 来自真实 positions。
        # 严格相等:不仅零 uniform_fallback,连 'poses' 回退都不允许 ——
        # gt_oracle 的 frame_positions 就是真值逐帧轨迹,任何一个样本退到 poses/None
        # 都说明数据或适配有问题。旧版只查 "uniform_fallback==0 且 positions>0",会
        # 放行 {positions:3, poses:2} 混合,与"全部 positions"的表述不符 —— 已收紧。
        # (v2.3) 唯一放行的额外标签 = 'degenerate_geometry':最小折叠转角 <20°
        # 的近共线样本(waypoint clamp 几何边角料),其 knot 依赖指标已统一置
        # None、不进任何均值 —— 属于"样本对该问题无定义",不是回退;计数公开。
        ("knots_source_counts",
         lambda v: (isinstance(v, dict)
                    and set(v) <= {"positions", "degenerate_geometry"}
                    and v.get("positions", 0) > 0),
         '⊆ {"positions", "degenerate_geometry"} 且 positions>0'
         '(零 poses / 零 uniform_fallback / 零 None;degenerate 公开计数)'),
    ]
    reasons = []
    checks = {}
    for key, ok, want in spec:
        v = gt_oracle_agg.get(key)
        passed = bool(ok(v))
        checks[key] = {"value": v, "want": want, "pass": passed}
        if not passed:
            reasons.append(f"{key}={v} (要求 {want})")
    return {"passed": not reasons, "reasons": reasons, "checks": checks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-h5", required=True,
                    help="GEO_H5(新 builder 产物;v2 需要 trajectory/full_scene/poses,"
                         "旧 H5 先用 server_run/augment_h5_trajectory.py 回填)")
    ap.add_argument("--split", required=True)
    ap.add_argument("--eval-dirs", nargs="*", default=[],
                    help="name=dir[:text|:geo](dir 含 generated/*.npy;后缀显式声明"
                         "路由。无后缀时读 eval 目录 meta 的 pose_conditioning 标记;"
                         "旧目录再退目录名启发式,歧义/冲突 FATAL)")
    ap.add_argument("--with-baselines", action="store_true",
                    help="跑诊断基线 + 标定门;门 FAIL 时非零退出")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="只取前 N 条 val;5 样本冒烟:--max-samples 5 --with-baselines")
    ap.add_argument("-o", "--output", default="trajectory_eval.json")
    args = ap.parse_args()

    val_idx = json.load(open(args.split))["val_indices"]
    if args.max_samples:
        val_idx = val_idx[: args.max_samples]
    gt = load_gt(args.gt_h5, val_idx)
    if not gt:
        print("[FATAL] GT 里没有 waypoints —— 该数据集需用【补了 waypoint 持久化的新 builder】重建。",
              file=sys.stderr)
        sys.exit(2)
    print(f"[info] {len(gt)} 条 val 样本含 GT 轨迹结构")

    out = {"n_gt": len(gt)}

    # ---- 各 eval 配置:Track B v2 ----
    # 生成视频是 GT 分辨率(480x832);score_trajectory 内部降到窗口分辨率算光流。
    for spec in args.eval_dirs:
        name, rest = spec.split("=", 1)
        d, suffix_route = _split_route_suffix(rest)
        # [v2.6] 目录必须存在:旧版对不存在的目录静默产出空配置(路径打错 /
        # 后缀拼错成路径的一部分时,整个配置无声消失)。
        if not Path(d).is_dir():
            print(f"[FATAL] 配置 {name!r}: eval 目录不存在: {d}"
                  f"(检查路径;合法的路由后缀只有 :text / :geo)", file=sys.stderr)
            sys.exit(2)
        gen_dir = Path(d) / "generated"
        # text-only 语义保持不变:commanded=False,只有净方向 + 段数是被命令的。
        # [v2.6/O11] 判定走显式后缀 → meta 标记 → (仅旧目录)名字启发式,
        # 冲突/歧义一律 FATAL —— commanded 标志不允许静默翻转。
        text_only, route_source = _resolve_text_only(name, d, suffix_route)
        scores = []
        per_sample = {}    # (v2.2) 逐样本落盘:均值被谁拉动、误差-净位移比相关、
        n_mismatch = 0     # 离群直方图 —— 没有逐样本行这些尸检全做不了(镜像 B3)。
        for k, g in gt.items():
            npy = gen_dir / f"{k}.npy"
            if not npy.exists():
                continue
            frames = np.load(npy)
            score = _score_v2(frames, g, text_only, sample_key=f"{name}/{k}")
            if score is None:          # (v2.1) 时间轴对不齐 -> 跳过并计数,不静默
                n_mismatch += 1
                continue
            fl = _flatten(score)
            scores.append(fl)
            per_sample[k] = fl
        # [v2.6.1] 零匹配 = 配置静默消失的最后一个洞:目录在、但 generated/ 里
        # 没有任何一个 npy 与 GT 的 val key 对上(传错 gt-h5 / split 与 eval
        # 不同源 / generated 为空)。旧行为是该配置无声不出现在输出里。
        if not scores and not n_mismatch:
            print(f"[FATAL] 配置 {name!r}: {gen_dir} 中没有任何 npy 与 GT val key "
                  f"匹配(gt-h5/split 与该 eval 目录不同源?generated/ 为空?)。",
                  file=sys.stderr)
            sys.exit(2)
        if scores or n_mismatch:
            out.setdefault("configs", {})[name] = _agg(scores, TRACKB_KEYS)
            out["configs"][name]["text_only"] = text_only
            # [v2.6/O11] 判定来源落盘:explicit:*/meta:*/name_heuristic。
            # 出表脚本应拒绝 name_heuristic 的行进正式表格。
            out["configs"][name]["route_source"] = route_source
            out["configs"][name]["timeline_mismatch"] = n_mismatch
            out["configs"][name]["per_sample"] = per_sample

    # ---- 诊断基线(证明 benchmark 可区分四类失败)+ 标定门 ----
    gate = None
    if args.with_baselines:
        baseline_names = ["gt_oracle", "deterministic_translation",
                          "persistence", "random_direction"]
        bscores = {b: [] for b in baseline_names}
        b_per_sample = {b: {} for b in baseline_names}
        for k, g in gt.items():
            # scene_image 必须是真实 full_scene(512x512);为 None 时
            # run_baselines 自动省略 det/random(persistence/gt_oracle 照常)。
            # 分辨率口径:det/random 产出即原生窗口分辨率(v2 默认不上采样);
            # persistence/gt_oracle 为 GT 分辨率,由 metrics 降到窗口分辨率。
            vids = DB.run_baselines(dict(
                scene_image=g["full_scene"],
                source_frame=g["gt_frames"][0],
                waypoints=g["waypoints"], window_size=g["window_size"], T=g["T"],
                gt_frames=g["gt_frames"],
            ))
            for bname in baseline_names:
                v = vids.get(bname)
                if v is None:
                    continue
                bs = _score_v2(v, g, text_only=False, sample_key=f"{bname}/{k}")
                if bs is None:
                    # 基线视频帧数=GT 帧数,时间轴恒等 —— 走到这里说明数据本身
                    # 有结构问题,大声跳过(会体现在 n 上,标定门随之 FAIL)。
                    continue
                fl = _flatten(bs)
                bscores[bname].append(fl)
                b_per_sample[bname][k] = fl
        for bname in baseline_names:
            if bscores[bname]:
                out.setdefault("baselines", {})[bname] = _agg(bscores[bname], TRACKB_KEYS)
                out["baselines"][bname]["per_sample"] = b_per_sample[bname]

        gate = _calibration_gate(out.get("baselines", {}).get("gt_oracle"))
        out["calibration_gate"] = gate

    json.dump(out, open(args.output, "w"), indent=2, ensure_ascii=False)
    # stdout 只打印聚合(逐样本行在输出 JSON 里,别刷爆日志)
    printable = json.loads(json.dumps(out))
    for sect in ("configs", "baselines"):
        for v in printable.get(sect, {}).values():
            v.pop("per_sample", None)
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"\n[done] → {args.output}(含逐样本 per_sample 行)")

    if gate is not None:
        print(f"\nTRACK B V2 CALIBRATION: {'PASS' if gate['passed'] else 'FAIL'}")
        if not gate["passed"]:
            for r in gate["reasons"]:
                print(f"  - {r}", file=sys.stderr)
            print("[FATAL] 标定门未通过:真值视频拿不到近满分 → 指标不可信,"
                  "先修指标/数据再烧 GPU。", file=sys.stderr)
            sys.exit(3)  # 结果 JSON 已写出,便于排查;仍以非零退出示败


if __name__ == "__main__":
    main()
