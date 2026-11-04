#!/usr/bin/env python3
# =============================================================================
# score_submission.py — RSWM-Bench 隐藏 test 的官方评分器(v1.0)
#
# 位置:参赛者交 generated/*.npy(或一个 tar),组织者在**有真值**的机器上跑本
# 脚本。它做两件事,顺序不可颠倒:
#   ① 合同校验(纯形状级,便宜):样本集合、逐样本帧数、分辨率、dtype/取值域。
#      任何一条不满足 → **整份提交拒收**,不出任何指标。
#      为什么不"能算多少算多少":部分评分等于允许挑样本 —— 只交容易的 200 条
#      也能得一个漂亮的均值。benchmark 的分数只有在"全集"上才有意义。
#   ② 评分:Track A(保真 + 净方向)与 Track B(轨迹控制),两轨的口径与
#      论文主表完全同源 —— 分别复用 lora_remotesense/new/metrics.py 与
#      server_run/run_trajectory_eval.py 的函数,不在这里重写任何指标。
#
# 三处**必须**复用而不是重写的东西(重写=两套口径迟早漂移,分数不可比):
#   * Track A 角误差口径 = flow_size=metadata['pose_window_size'] + step=1。
#     全分辨率 480×832 跑 Farnebäck 是已知量程事故(GT 地板会读出 82°),
#     见 evaluate.py 的 _ANGKW 与 instrument_floor_report.py 文件头。
#   * 方向目标 = **waypoints** 端点净位移量化后 +180°(内容角),prompt 只在
#     无 waypoints 时回退 —— 直接调 evaluate.py 的同名函数。
#   * Track B = run_trajectory_eval._score_v2(窗口分辨率光流 + 真实结点
#     + 时间轴自适配),连同 _flatten/_agg/TRACKB_KEYS/退化几何规则一起复用。
#
# 逐样本流式:GT 视频是 float32 (33,480,832,3) ≈ 158 MB/条,test 有几百条。
# 绝不像 evaluate.compute_metrics 那样把整批读进 list —— 那在 test 规模上必炸。
# 代价是 FID/FVD 不能直接调 compute_fid/compute_fvd(它们只吃整表),改为
# 逐样本抽特征后套同一条 Fréchet 公式(见 _frechet_distance 的注释)。
#
# 仪器地板(必读,否则分数会被过度解读):
#   * 四方向 prompt 的**量化地板** ≈ 22.5°,仅当库是连续角(angle_quantize=0)
#     时存在;判门库(angle_quantize=4)轨迹角本身就是四基向,该项为 0。
#     两者不可混用 —— 本脚本从 H5 metadata 读 angle_quantize 自动判定。
#   * 随机方向的期望角误差 = 90°(可用量程的上端)。
#   * 光流估计器在**真值视频**上的实测地板(≈25° 量级)= 纯仪器噪声,
#     由 --floor measured 当场实测,不写死数字。
#
# 用法:
#   python3 server_run/score_submission.py \
#       --gt-h5 $WORK/data/train_geo.h5 --splits $WORK/official_splits.json \
#       --submission /path/to/team_x.tar.gz -o $WORK/score_team_x.json
#
# 退出码:0 通过并出分 / 2 用法或环境错误 / 3 仪器地板门 FAIL(--require-floor-gate)
#         / 4 提交不合规(已拒收)
# =============================================================================
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

# [v3.0] 共享评分契约(唯一权威)。评分器与指标计算必须对同一违规给同一判决。
_CONTRACT_DIR = Path(__file__).resolve().parent.parent / "lora_remotesense" / "new"
if str(_CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTRACT_DIR))
try:
    from submission_contract import (check_id_set as _contract_check_id_set,
                                     CONDITION_FRAMES as _CONTRACT_COND_FRAMES,
                                     MIN_SAMPLES_FOR_DISTRIBUTION as _CONTRACT_MIN_DIST,
                                     strip_condition_frames as _strip_condition_frames,
                                     assert_enough_for_distribution as _enough_for_distribution,
                                     contract_summary as _contract_summary)
except Exception as _e:      # 契约模块必须在 —— 缺了就不该评分
    raise SystemExit(
        f"[FATAL] 无法导入 submission_contract({_e})—— "
        f"评分器必须与指标计算共用同一契约,不允许各写一套。") from _e

import h5py

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "lora_remotesense" / "new"))

try:
    import trajectory_metrics as TM        # noqa: E402  (纯 numpy/cv2)
    import metrics as MET                  # noqa: E402  (Track A 指标真源)
    import evaluate as EV                  # noqa: E402  (方向目标口径真源)
    import run_trajectory_eval as RTE      # noqa: E402  (Track B 打分真源)
except ImportError as _e:
    # metrics/evaluate 依赖 torch(SSIM/LPIPS/FID/FVD 都在那条链上)。裸
    # ImportError 的 traceback 对组织者没有信息量,这里换成一句能照做的话。
    print(f"[FATAL] 评分依赖导入失败: {_e}\n"
          f"        本脚本必须在 rswm 环境里跑(需要 numpy/h5py/opencv/scipy/"
          f"scikit-image/torch;lpips 与 torchvision 缺失只会让 LPIPS/FID/FVD "
          f"记 null,不影响其余指标)。", file=sys.stderr)
    sys.exit(2)

SCHEMA = "rswm_submission_score/v1"
REQUIRED_META_KEYS = ("team", "method", "track")


# --------------------------------------------------------------------------
def sha256_file(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def h5_str(v):
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def json_sanitize(obj, path="", found=None):
    """把 NaN/Inf 换成 null,并记下出现位置。

    Python 的 json 默认会写出裸 `Infinity`/`NaN` —— 那**不是合法 JSON**,
    jq、浏览器、多数排行榜后端都会解析失败。而这两个值在这里是会真实出现的:
    逐帧完全一致的提交 PSNR = inf,拿不到权重的 FID = nan。换成 null 并在
    warnings 里点名,比写出一个没人能解析的报告好。
    """
    found = [] if found is None else found

    def walk(o, p):
        if isinstance(o, dict):
            return {k: walk(v, f"{p}.{k}") for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [walk(v, f"{p}[{i}]") for i, v in enumerate(o)]
        if isinstance(o, np.ndarray):
            return walk(o.tolist(), p)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, (float, np.floating)):
            o = float(o)
            if not np.isfinite(o):
                found.append(f"{p}={o}")
                return None
            return o
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, bytes):
            return o.decode("utf-8", "replace")
        return o

    return walk(obj, path), found


class Rejection:
    """收集拒收理由。每条带机器可读 code,参赛者能按 code 自查(见 SUBMISSION.md)。"""

    def __init__(self):
        self.errors, self.warnings = [], []

    def err(self, code, detail, examples=None):
        self.errors.append({"code": code, "detail": detail,
                            "examples": list(examples or [])[:20],
                            "n_affected": len(examples) if examples else None})

    def warn(self, code, detail, examples=None):
        self.warnings.append({"code": code, "detail": detail,
                              "examples": list(examples or [])[:20]})


# --------------------------------------------------------------------------
# 提交包定位与解包
# --------------------------------------------------------------------------
def unpack_submission(path, workdir):
    """→ (root_dir, tmp_dir_to_cleanup)。支持目录 / .tar / .tar.gz / .tgz。"""
    p = Path(path)
    if p.is_dir():
        return p, None
    if not p.is_file():
        print(f"[FATAL] 提交路径不存在: {p}", file=sys.stderr)
        sys.exit(2)
    tmp = Path(tempfile.mkdtemp(prefix="rswm_sub_", dir=workdir))
    try:
        _extract_tar(p, tmp)
    except Exception as e:
        # 解包失败(损坏 / 路径穿越 / 非普通文件)一律当**用法错误**处理并给一句
        # 人话,而不是把 tarfile 的 traceback 甩给组织者 —— 这条路径上出现的
        # 异常有一半是恶意构造的,报错信息本身就是审计记录。
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[FATAL] 提交包解包失败: {type(e).__name__}: {e}\n"
              f"        (tar 里出现 ../ 绝对路径或软链会被直接拒绝)", file=sys.stderr)
        sys.exit(2)
    return tmp, tmp


def _extract_tar(p, tmp):
    with tarfile.open(p, "r:*") as tf:
        # tar 里的 ../ 与绝对路径能覆盖组织者机器上的任意文件。Python 3.12 的
        # data 过滤器专治这个;老版本手工拦一遍,绝不裸 extractall。
        if hasattr(tarfile, "data_filter"):
            tf.extractall(tmp, filter="data")
        else:
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in Path(m.name).parts:
                    raise ValueError(f"tar 成员路径不安全: {m.name}")
                if not (m.isfile() or m.isdir()):
                    raise ValueError(f"tar 含非普通文件成员(软链/设备?): {m.name}")
            tf.extractall(tmp)


def locate_generated(root):
    """找 generated/ 目录。**定位**上宽容(打包多一层是最常见的失误),
    但内容校验一步不让 —— 宽容的边界就到"找得到"为止。"""
    cands = [root / "generated"]
    cands += sorted(root.glob("*/generated"))
    for c in cands:
        if c.is_dir():
            return c
    return None


def load_submission_meta(root, rej, required=True):
    f = root / "submission.json"
    if not f.is_file():
        cand = sorted(root.glob("*/submission.json"))
        f = cand[0] if cand else f
    if not f.is_file():
        if required:
            rej.err("NO_SUBMISSION_JSON",
                    "提交包根目录缺 submission.json(见 SUBMISSION.md §目录结构)")
        return {}
    try:
        meta = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        rej.err("BAD_SUBMISSION_JSON", f"submission.json 解析失败: {e}")
        return {}
    if required:
        missing = [k for k in REQUIRED_META_KEYS if not meta.get(k)]
        if missing:
            rej.err("SUBMISSION_META_INCOMPLETE",
                    f"submission.json 缺必填字段 {missing}")
        if str(meta.get("track", "")).upper() not in ("A", "B"):
            rej.err("BAD_TRACK",
                    f"track={meta.get('track')!r} 非法,只能是 'A' 或 'B'")
    return meta


# --------------------------------------------------------------------------
# 合同校验(pass 1:只看形状,不算指标)
# --------------------------------------------------------------------------
def validate_contract(gen_dir, expected, rej, target_hw, all_test_keys=None):
    """expected: {key: T}。→ {key: path}(全部合规时)。

    这里用 mmap 只读 header:几百条 × 40 MB 的全量读盘留给评分那一趟,
    形状类错误没必要为此等十分钟。
    """
    files = {}
    bad_ext = []
    for p in sorted(gen_dir.iterdir()):
        if p.is_dir():
            continue
        if p.suffix != ".npy":
            bad_ext.append(p.name)
            continue
        files[p.stem] = p
    if bad_ext:
        rej.warn("UNEXPECTED_FILES",
                 f"generated/ 里有 {len(bad_ext)} 个非 .npy 文件,已忽略", bad_ext)

    # [v3.0] ID 集规则复用 lora_remotesense/new/submission_contract.py 的
    # check_id_set —— 它同时被 metrics/evaluate 用。上几轮的 P0 全部源自
    # "孪生入口各写一份必然漂移";这里能对齐接口就直接调,对齐不了的
    # (逐样本 T / 分辨率 / dtype)由 test_submission_contract.py 的
    # 等价性用例保证两条路径对同一违规给出同样的判决。
    _cid = _contract_check_id_set(list(files), list(expected), what="提交")
    missing = sorted(set(expected) - set(files))
    extra = sorted(set(files) - set(expected))
    # --max-samples 冒烟模式:被截掉的那部分 test 样本交上来不算"多余"——
    # 它们本来就属于 test,只是这次不评。真正的多余是 test 之外的 id。
    if all_test_keys is not None:
        ignored = [k for k in extra if k in all_test_keys]
        extra = [k for k in extra if k not in all_test_keys]
        if ignored:
            rej.warn("IGNORED_BY_MAX_SAMPLES",
                     f"{len(ignored)} 个样本因 --max-samples 未参与评分", ignored)
    if missing:
        rej.err("MISSING_SAMPLES",
                f"缺 {len(missing)}/{len(expected)} 个 test 样本 —— 必须交齐,"
                f"不接受部分提交", missing)
    if extra:
        rej.err("EXTRA_SAMPLES",
                f"多出 {len(extra)} 个不属于 test 的样本(train/val 的 id?"
                f"改名?)—— 提交必须与 test 集合**恰好**相等", extra)

    H, W = target_hw
    wrong_T, wrong_res, wrong_dtype, unreadable, bad_range = [], [], [], [], []
    for k in sorted(set(expected) & set(files)):
        try:
            a = np.load(files[k], mmap_mode="r")
        except Exception as e:
            unreadable.append(f"{k}: {e}")
            continue
        if a.ndim != 4 or a.shape[1:] != (H, W, 3):
            wrong_res.append(f"{k}: shape={tuple(a.shape)} 期望 (T,{H},{W},3)")
            continue
        if a.shape[0] != expected[k]:
            wrong_T.append(f"{k}: T={a.shape[0]} 期望 {expected[k]}")
        if not np.issubdtype(a.dtype, np.number):
            wrong_dtype.append(f"{k}: dtype={a.dtype}")
            continue
        if a.dtype != np.uint8:
            # 抽样看量纲:把 [0,1] 归一化的浮点当 [0,255] 存是最隐蔽的错 ——
            # clip 之后整段视频几乎全黑,SSIM/光流全废,却不会报任何错。
            sub = np.asarray(a[:: max(1, a.shape[0] // 4)], dtype=np.float64)
            if not np.all(np.isfinite(sub)):
                bad_range.append(f"{k}: 含 NaN/Inf")
            elif float(np.nanmax(np.abs(sub))) <= 1.5:
                bad_range.append(
                    f"{k}: dtype={a.dtype} 且 max|v|={np.nanmax(np.abs(sub)):.4f} "
                    f"—— 看起来是 [0,1] 归一化;本 benchmark 要求 [0,255]")
            elif float(np.nanmin(sub)) < -1.0 or float(np.nanmax(sub)) > 260.0:
                bad_range.append(
                    f"{k}: 取值域 [{np.nanmin(sub):.2f},{np.nanmax(sub):.2f}] "
                    f"超出 [0,255]")
    if unreadable:
        rej.err("UNREADABLE_NPY", f"{len(unreadable)} 个 npy 读不出来", unreadable)
    if wrong_res:
        rej.err("RESOLUTION_MISMATCH",
                f"{len(wrong_res)} 个样本分辨率/维度不对(必须 (T,{H},{W},3) RGB)",
                wrong_res)
    if wrong_T:
        rej.err("FRAME_COUNT_MISMATCH",
                f"{len(wrong_T)} 个样本帧数不对(逐样本 T 见公开 manifest;"
                f"契约 T = 9+(K-1)*8)", wrong_T)
    if wrong_dtype:
        rej.err("BAD_DTYPE", f"{len(wrong_dtype)} 个样本不是数值类型", wrong_dtype)
    if bad_range:
        rej.err("SUSPECT_VALUE_RANGE",
                f"{len(bad_range)} 个样本的取值域不像 [0,255] 的 RGB", bad_range)
    return files


# --------------------------------------------------------------------------
# GT 逐样本读取(流式;字段语义与 run_trajectory_eval.load_gt 一致)
# --------------------------------------------------------------------------
def open_gt(gt_h5):
    f = h5py.File(gt_h5, "r")
    meta = dict(f["metadata"].attrs) if "metadata" in f else {}
    if "pose_window_size" not in meta:
        print("[FATAL] H5 metadata 缺 pose_window_size —— Track A/B 的光流都必须"
              "在原生窗口分辨率上算(全分辨率是已知的量程事故),拒绝评分。",
              file=sys.stderr)
        f.close()
        sys.exit(2)
    win = tuple(int(v) for v in meta["pose_window_size"])
    return f, meta, win


def read_gt_sample(f, key, win):
    """→ 与 RTE._score_v2 约定一致的 dict(只读它真正用到的字段 + 基线所需)。

    不复用 RTE.load_gt:那个函数一次把所有 val 的 gt_frames 读进内存
    (float32 158 MB/条),test 规模下直接 OOM。字段含义与它逐字对齐。
    """
    g = f["data"][key]
    return {
        "key": key,
        "waypoints": g["waypoints"][:] if "waypoints" in g else None,
        "segment_angles": g["segment_angles"][:] if "segment_angles" in g else None,
        "trajectory": g["trajectory"][:] if "trajectory" in g else None,
        "poses": g["poses"][:] if "poses" in g else None,
        "full_scene": g["full_scene"][:] if "full_scene" in g else None,
        "window_size": win,
        "T": int(g.attrs.get("num_frames", g["video_frames"].shape[0])),
        "prompt": h5_str(g.attrs.get("prompt", "")),
        "scene": EV.canonical_scene(h5_str(g.attrs.get("source_scene_category", "")) or ""),
        "num_segments": int(g.attrs["num_segments"]) if "num_segments" in g.attrs else None,
    }


def direction_target(gt, key):
    """方向目标角(内容角)+ 方向词。规则与 evaluate.load_val_samples 同源:
    **waypoints 优先**(数据级真源),prompt 只在无 waypoints 时回退。
    prompt 与 waypoints 矛盾 = 图像/轨迹/措辞对错位,方向指标全不可信 → 硬失败。
    """
    wp = gt["waypoints"]
    p_ang = EV.parse_direction_angle_from_prompt(gt["prompt"])
    p_dw = EV.parse_direction_word_from_prompt(gt["prompt"])
    if wp is not None and len(wp) >= 2:
        tgt = EV.direction_target_from_waypoints(wp)
        dw = EV.direction_word_from_waypoints(wp)
        if p_ang is not None and abs(((p_ang - tgt + 180.0) % 360.0) - 180.0) > 1e-6:
            raise RuntimeError(
                f"GT 样本 {key}: prompt 净方向({p_dw},{p_ang}°)与 waypoints "
                f"净位移({dw},{tgt}°)矛盾 —— 真值本身不自洽,拒绝评分。")
        return tgt, dw, "waypoints"
    if p_ang is not None:
        return p_ang, p_dw, "prompt"
    return None, None, None


def _frechet_distance(fa, fb):
    """Fréchet 距离(FID/FVD 共用)。

    ⚠️ 本函数必须与 metrics.compute_fid / compute_fvd 里的四行公式**逐字一致**。
    之所以在这里再写一遍:那两个函数只接受"整批视频的 list",而 test 规模下
    必须逐样本抽完特征就把视频丢掉。改动指标口径时两处一起改。
    """
    from scipy.linalg import sqrtm
    mu_a, sig_a = fa.mean(axis=0), np.cov(fa, rowvar=False)
    mu_b, sig_b = fb.mean(axis=0), np.cov(fb, rowvar=False)
    diff = mu_a - mu_b
    covmean = sqrtm(sig_a @ sig_b)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_a + sig_b - 2 * covmean))


def fidelity_views(gen_u8, gt_u8):
    """保真/分布指标只看预测帧；轨迹指标的调用方继续使用完整视频。"""
    return _strip_condition_frames(gen_u8, gt_u8)


# --------------------------------------------------------------------------
def score(gt_h5, keys, files, win, meta, track, device, rej,
          dist_metrics=True, verbose=True):
    """逐样本流式打分 → (track_a, track_b)。"""
    f, _meta, _win = open_gt(gt_h5)
    text_only = (track == "A")
    ssim, psnr, lpips_v = [], [], []
    ang_err, ang_wellposed, net_ratios = [], [], []
    per_sample, tb_scores, tb_per_sample = {}, [], {}
    frame_feat_gen, frame_feat_gt = [], []
    video_feat_gen, video_feat_gt = [], []
    n_mismatch, n_lpips_fail = 0, 0
    frame_model = vid_model = None
    if dist_metrics:
        frame_model, _ = MET._load_frame_feature_model(device)
        if frame_model is None:
            rej.warn("FID_UNAVAILABLE", "InceptionV3 权重加载失败(离线?)→ FID 记 null")
        vid_model, _ = MET._load_video_feature_model(device)
        if vid_model is None:
            rej.warn("FVD_UNAVAILABLE", "r3d_18 权重加载失败(离线?)→ FVD 记 null")
    dup_hash = Counter()
    try:
        for n, k in enumerate(keys, 1):
            gt = read_gt_sample(f, k, win)
            gen = np.load(files[k])
            # 合同校验那趟只抽样查了 NaN/Inf(几百条 × 40 MB 全读两遍不值),
            # 这里做全量检查:NaN 经 np.clip 会变成 0,整段视频静默变黑。
            if gen.dtype != np.uint8 and not np.all(np.isfinite(gen)):
                rej.err("NON_FINITE_VALUES", f"{k} 含 NaN/Inf", [k])
                break
            gen_u8 = gen if gen.dtype == np.uint8 else np.clip(gen, 0, 255).astype(np.uint8)
            del gen
            gt_u8 = np.clip(f["data"][k]["video_frames"][:], 0, 255).astype(np.uint8)
            dup_hash[hashlib.md5(gen_u8[:: max(1, len(gen_u8) // 3)].tobytes()).hexdigest()] += 1

            # ---- Track A:保真（首帧是 I2V 条件输入，不是预测） ----
            gen_pred, gt_pred = fidelity_views(gen_u8, gt_u8)
            s_ = MET.compute_video_ssim(gen_pred, gt_pred)
            p_ = MET.compute_video_psnr(gen_pred, gt_pred)
            ssim.append(s_)
            psnr.append(p_)
            l_ = None
            try:
                l_ = MET.compute_video_lpips(gen_pred, gt_pred, device)
                lpips_v.append(l_)
            except Exception:
                n_lpips_fail += 1
            # ---- Track A:净方向(口径 = 窗口分辨率 + step=1,同 evaluate._ANGKW)
            tgt, dw, tsrc = direction_target(gt, k)
            a_ = MET.compute_batch_angular_error(
                [gen_u8], [tgt], flow_size=win, step=1)["per_sample"][0]
            if a_ is not None:
                ang_err.append(a_)
            nr = (float(TM.polyline_net_ratio(gt["waypoints"]))
                  if gt["waypoints"] is not None and len(gt["waypoints"]) >= 2 else None)
            net_ratios.append(nr)
            # well-posed 双栏:GT 命令轨迹近闭合时"净方向"这个问题本身无定义,
            # 门限与 Track B 的净方向门控同源(evaluate.WELL_POSED_MIN_NET_RATIO)。
            if a_ is not None and nr is not None and nr >= EV.WELL_POSED_MIN_NET_RATIO:
                ang_wellposed.append(a_)
            if dist_metrics:
                if frame_model is not None:
                    frame_feat_gen.append(MET._extract_frame_features(
                        list(gen_pred), frame_model, device))
                    frame_feat_gt.append(MET._extract_frame_features(
                        list(gt_pred), frame_model, device))
                if vid_model is not None:
                    video_feat_gen.append(MET._extract_video_features(
                        [gen_pred], vid_model, device)[0])
                    video_feat_gt.append(MET._extract_video_features(
                        [gt_pred], vid_model, device)[0])

            # ---- Track B:完整轨迹控制(复用 run_trajectory_eval 的打分入口)
            tb = RTE._score_v2(gen_u8, gt, text_only, sample_key=f"submission/{k}")
            if tb is None:
                n_mismatch += 1
            else:
                fl = RTE._flatten(tb)
                tb_scores.append(fl)
                tb_per_sample[k] = fl
            per_sample[k] = {
                "SSIM": float(s_), "PSNR": float(p_),
                "LPIPS": (float(l_) if l_ is not None else None),
                "Angular_Error": a_, "direction_word": dw,
                "direction_target_deg": tgt, "direction_target_source": tsrc,
                "gt_net_displacement_ratio": nr, "scene": gt["scene"],
                "num_frames": int(len(gen_u8)),
                "condition_frames_excluded": int(_CONTRACT_COND_FRAMES),
                "fidelity_frames": int(len(gen_pred)),
            }
            del gen_pred, gt_pred, gen_u8, gt_u8
            if verbose and n % 25 == 0:
                print(f"  … 已评 {n}/{len(keys)}")
    finally:
        f.close()

    dups = {h: c for h, c in dup_hash.items() if c > 1}
    if dups:
        rej.warn("DUPLICATE_VIDEOS",
                 f"{sum(dups.values())} 个样本的视频指纹重复({len(dups)} 组)——"
                 f"同一段视频交给多个样本?(仅告警,不拒收;Track B 的 valid/"
                 f"净方向会如实反映)")

    ta = {
        "num_samples": len(ssim),
        "SSIM": float(np.mean(ssim)) if ssim else None,
        "PSNR": float(np.mean(psnr)) if psnr else None,
        "LPIPS": float(np.mean(lpips_v)) if lpips_v else None,
        "LPIPS_unavailable_n": n_lpips_fail,
        "Angular_Error_Mean": float(np.mean(ang_err)) if ang_err else None,
        "Angular_Error_Median": float(np.median(ang_err)) if ang_err else None,
        "Angular_Error_Std": float(np.std(ang_err)) if ang_err else None,
        "Angular_Error_Count": len(ang_err),
        "Angular_Error_Mean_wellposed": (float(np.mean(ang_wellposed))
                                         if ang_wellposed else None),
        "Angular_Error_wellposed_Count": len(ang_wellposed),
        "wellposed_min_net_ratio": EV.WELL_POSED_MIN_NET_RATIO,
        "angular_caliber": {"flow_size": list(win), "step": 1,
                            "target": "waypoints 端点净位移 → 四基向量化 → +180°(内容角)"},
        "Condition_Frames_Excluded": int(_CONTRACT_COND_FRAMES),
        "FID_Frame_Selection": "all_predicted_frames",
        "FID": None, "FVD": None, "FVD_Extractor": MET.FVD_EXTRACTOR_NAME,
        "FVD_Is_Standard_I3D": False,
        "Contract": _contract_summary(),
        "per_sample": per_sample,
    }
    enough_dist = _enough_for_distribution(
        len(ssim), metric="FID/FVD", group="official test")
    if dist_metrics and enough_dist:
        if frame_feat_gen:
            try:
                fg, ft = np.concatenate(frame_feat_gen), np.concatenate(frame_feat_gt)
                ta["FID"] = _frechet_distance(fg, ft)
                ta["FID_Frames"] = int(len(fg))
            except Exception as e:
                rej.warn("FID_UNAVAILABLE", f"FID 计算失败: {e}")
        if video_feat_gen:
            try:
                ta["FVD"] = _frechet_distance(
                    np.stack(video_feat_gen), np.stack(video_feat_gt))
            except Exception as e:
                rej.warn("FVD_UNAVAILABLE", f"FVD 计算失败: {e}")
    elif dist_metrics:
        why = f"n={len(ssim)} < {_CONTRACT_MIN_DIST}"
        ta["FID_Skipped_Reason"] = why
        ta["FVD_Skipped_Reason"] = why

    tb_agg = RTE._agg(tb_scores, RTE.TRACKB_KEYS) if tb_scores else {"n": 0}
    tb_agg["text_only"] = text_only
    tb_agg["commanded_note"] = (
        "Track A(pose drop):只有净方向与段数是被命令的,逐段/转弯/waypoint/"
        "形状指标 commanded=False,仅供诊断,不作达标量"
        if text_only else
        "Track B(pose native):Plücker 位姿即完整折线命令,全部指标均为达标量")
    tb_agg["timeline_mismatch"] = n_mismatch
    tb_agg["per_sample"] = tb_per_sample
    return ta, tb_agg


# --------------------------------------------------------------------------
def measure_floor(gt_h5, test_idx, win, meta, device, max_samples, seed, workdir):
    """仪器地板:解析项(量化/随机)+ 在**真值视频**上实测的两轨地板。

    Track A 直接调 instrument_floor_report.measure_gt_floor(与论文地板同一函数);
    Track B 跑 gt_oracle 基线并套 run_trajectory_eval 的标定门 —— 真值视频喂进
    管线都拿不到分,说明是指标失效而不是模型失效,这时任何提交分数都不该被解读。
    """
    import instrument_floor_report as IFR
    import diagnostic_baselines as DB

    aq = int(meta.get("angle_quantize", 0) or 0)
    out = {
        "quantization_floor_deg": 22.5 if aq == 0 else 0.0,
        "quantization_floor_note": (
            "轨迹净方向是连续角、prompt 只说 E/N/W/S → 量化残差期望 |offset|=22.5°;"
            "本库 angle_quantize=%d %s" % (
                aq, "(连续角,该项存在)" if aq == 0
                else "(轨迹角本身即四基向,该项为 0;判门库的地板是纯仪器噪声,"
                     "不可与主库地板混用)")),
        "random_expectation_deg": 90.0,
        "usable_range_note": "可用量程 = [实测地板, 90°];>90° 的读数比随机还差",
    }
    idx = list(test_idx)[:max_samples] if max_samples else list(test_idx)
    tmp = Path(workdir) / "_floor_split.json"
    # measure_gt_floor 按 split 文件的 val_indices 取样本 —— 给它一份"val=test"
    # 的临时视图,比复制它一整套读盘逻辑安全。
    tmp.write_text(json.dumps({"val_indices": [int(i) for i in idx]}))
    try:
        out["track_a_gt_floor"] = IFR.measure_gt_floor(
            gt_h5, str(tmp), max_samples=0, n_boot=1000, alpha=0.05,
            seed=seed, force_window=tuple(win))
    except Exception as e:
        out["track_a_gt_floor"] = {"error": f"{type(e).__name__}: {e}"}
    finally:
        tmp.unlink(missing_ok=True)

    scores = []
    with h5py.File(gt_h5, "r") as f:
        keys = sorted(f["data"].keys())
        for i in idx:
            k = keys[i]
            gt = read_gt_sample(f, k, win)
            gt_frames = f["data"][k]["video_frames"][:]
            vids = DB.run_baselines(dict(
                scene_image=gt["full_scene"], source_frame=gt_frames[0],
                waypoints=gt["waypoints"], window_size=win, T=gt["T"],
                gt_frames=gt_frames))
            v = vids.get("gt_oracle")
            if v is None:
                continue
            s = RTE._score_v2(v, gt, text_only=False, sample_key=f"gt_oracle/{k}")
            if s is not None:
                scores.append(RTE._flatten(s))
    agg = RTE._agg(scores, RTE.TRACKB_KEYS) if scores else None
    if agg is not None:
        agg.pop("per_sample", None)
    out["track_b_gt_oracle"] = agg
    out["track_b_calibration_gate"] = RTE._calibration_gate(agg)
    return out


# --------------------------------------------------------------------------
def check_external_sources(gt_h5, test_idx, declared, rej):
    """参赛者声明的外部训练语料 vs test 源图 —— 只有组织者能做这道核查。

    为什么协议是"声明 + 组织者核验"而不是"公开 test 源图黑名单":AID/RESISC45
    是公开语料,一旦公布 test 用了哪些源图,配上公开的窗口尺寸就能反推出 GT
    窗口(Track B 更是连轨迹都是给定的)—— 黑名单本身就是作弊说明书。
    """
    if not declared:
        return None
    ids = set()
    with h5py.File(gt_h5, "r") as f:
        keys = sorted(f["data"].keys())
        for i in test_idx:
            a = f["data"][keys[i]].attrs
            if "source_index" in a:
                ids.add(f"{h5_str(a.get('source_dataset',''))}/"
                        f"{h5_str(a.get('source_split',''))}/{int(a['source_index'])}")
            p = h5_str(a.get("source_path", "") or "")
            if p:
                ids.add(p)
                ids.add(Path(p).name)
    hit = sorted(ids & {str(x) for x in declared})
    if hit:
        rej.err("EXTERNAL_SOURCE_OVERLAP",
                f"声明的外部训练语料里有 {len(hit)} 项与 test 源图重合 —— "
                f"禁止在 test 源图上训练", hit)
    return {"n_declared": len(declared), "n_test_sources": len(ids),
            "n_overlap": len(hit), "overlap_examples": hit[:20]}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RSWM-Bench 隐藏 test 提交评分器")
    ap.add_argument("--gt-h5", required=True, help="含真值的私有 H5")
    ap.add_argument("--splits", required=True, help="make_official_splits.py 的产物")
    ap.add_argument("--submission", required=True, help="提交目录 或 tar/tar.gz")
    ap.add_argument("-o", "--output", default="score_report.json")
    ap.add_argument("--track", default=None, choices=["A", "B"],
                    help="覆盖 submission.json 里的 track(默认读提交声明)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-dist-metrics", action="store_true",
                    help="跳过 FID/FVD(需要预训练权重,离线环境拿不到)")
    ap.add_argument("--no-require-metadata", action="store_true",
                    help="不强制 submission.json(组织者内部复算用)")
    ap.add_argument("--floor", default="measured", choices=["measured", "none"],
                    help="是否在真值视频上实测仪器地板(默认测;测了才敢解读分数)")
    ap.add_argument("--floor-max-samples", type=int, default=100)
    ap.add_argument("--require-floor-gate", action="store_true",
                    help="Track B 标定门 FAIL 时 exit 3(真值都拿不到分 = 指标失效)")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="[仅冒烟] 只要求/只评 test 的前 N 条;报告会标记 partial")
    ap.add_argument("--allow-h5-sha-mismatch", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.require_floor_gate and args.floor == "none":
        ap.error("--require-floor-gate 需要 --floor measured(不实测就没有门可判)")

    t0 = time.time()
    rej = Rejection()
    splits_doc = json.loads(Path(args.splits).read_text())
    if splits_doc.get("schema") != "rswm_official_splits/v1":
        print(f"[FATAL] {args.splits} 不是 official_splits(schema="
              f"{splits_doc.get('schema')!r})", file=sys.stderr)
        sys.exit(2)
    gt_sha = sha256_file(args.gt_h5)
    rec_sha = splits_doc.get("dataset", {}).get("sha256")
    if rec_sha and rec_sha != gt_sha:
        msg = (f"--gt-h5 的 SHA {gt_sha[:16]}… 与 official_splits 记录的 "
               f"{str(rec_sha)[:16]}… 不同 —— 换了数据集,'test' 指的就不是同一批"
               f"样本,分数不可比。")
        if args.allow_h5_sha_mismatch:
            rej.warn("GT_H5_SHA_MISMATCH", msg)
        else:
            print(f"[FATAL] {msg} 确要继续加 --allow-h5-sha-mismatch。", file=sys.stderr)
            sys.exit(2)

    test_idx = list(splits_doc["splits"]["test"]["indices"])
    test_keys = list(splits_doc["splits"]["test"]["keys"])
    all_test_keys = set(test_keys)
    partial = False
    if args.max_samples:
        test_idx, test_keys = test_idx[:args.max_samples], test_keys[:args.max_samples]
        partial = True

    f, meta, win = open_gt(args.gt_h5)
    tgt_hw = tuple(int(v) for v in meta.get("target_size", (480, 832)))
    expected = {}
    for i, k in zip(test_idx, test_keys):
        g = f["data"][k]
        expected[k] = int(g.attrs.get("num_frames", g["video_frames"].shape[0]))
    f.close()

    workdir = tempfile.mkdtemp(prefix="rswm_score_")
    root, cleanup = unpack_submission(args.submission, workdir)
    sub_meta = load_submission_meta(root, rej, required=not args.no_require_metadata)
    gen_dir = locate_generated(root)
    if gen_dir is None:
        rej.err("NO_GENERATED_DIR",
                f"提交包里找不到 generated/ 目录(在 {root} 下最多找一层)")
        files = {}
    else:
        files = validate_contract(gen_dir, expected, rej, tgt_hw,
                                  all_test_keys=all_test_keys if partial else None)

    track = args.track or str(sub_meta.get("track", "")).upper()
    if track not in ("A", "B"):
        if args.track is None and args.no_require_metadata:
            track = "B"
            rej.warn("TRACK_DEFAULTED", "未声明 track 且免除了元数据校验 → 按 B 评"
                                        "(Track B 的 commanded 语义更严)")
        else:
            rej.err("BAD_TRACK", f"无法确定 track(submission.json={sub_meta.get('track')!r},"
                                 f"--track={args.track!r})")

    ext = check_external_sources(args.gt_h5, test_idx,
                                 sub_meta.get("external_source_ids") or [], rej)
    if sub_meta.get("used_external_data") and not sub_meta.get("external_source_ids"):
        rej.warn("EXTERNAL_DATA_UNDECLARED",
                 "声明用了外部数据但没给 external_source_ids —— 无法核验它与 test "
                 "源图不重叠;组织者可据此要求补充。")

    report = {
        "schema": SCHEMA,
        "accepted": False,
        "partial": partial,
        "errors": rej.errors, "warnings": rej.warnings,
        "submission": {
            "path": str(args.submission), "root": str(root),
            "metadata": sub_meta, "track_scored": track,
            "n_files": len(files),
            "external_source_check": ext,
        },
        "dataset": {
            "gt_h5": str(args.gt_h5), "gt_h5_sha256": gt_sha,
            "splits_file": str(args.splits),
            "splits_sha256": splits_doc.get("splits_sha256"),
            "n_test": len(test_keys), "target_size": list(tgt_hw),
            "pose_window_size": list(win),
            "angle_quantize": int(meta.get("angle_quantize", 0) or 0),
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__, "h5py": h5py.__version__,
            "device": args.device,
        },
    }
    try:
        from run_manifest import build_manifest
        report["run_manifest"] = build_manifest(
            role=os.environ.get("RSWM_RUN_ROLE", "submission_scoring"),
            data_h5=args.gt_h5, split_file=args.splits,
            checkpoint_id=f"submission:{sub_meta.get('team', '?')}")
    except Exception as e:      # manifest 缺失不该挡住评分,但要留痕
        report["run_manifest"] = {"error": f"{type(e).__name__}: {e}"}

    if rej.errors:
        report["verdict"] = "REJECTED"
        _doc, _ = json_sanitize(report)
        Path(args.output).write_text(
            json.dumps(_doc, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nSUBMISSION: REJECTED")
        for e in rej.errors:
            print(f"  [{e['code']}] {e['detail']}", file=sys.stderr)
            for x in e["examples"][:5]:
                print(f"      · {x}", file=sys.stderr)
        print(f"[done] → {args.output}")
        if cleanup:
            shutil.rmtree(cleanup, ignore_errors=True)
        shutil.rmtree(workdir, ignore_errors=True)
        sys.exit(4)

    print(f"[info] 合同校验通过:{len(files)} 个样本,track={track};开始评分…")
    ta, tb = score(args.gt_h5, test_keys, files, win, meta, track, args.device,
                   rej, dist_metrics=not args.no_dist_metrics)
    report["track_a"] = ta
    report["track_b"] = tb
    if args.floor == "measured":
        print("[info] 实测仪器地板(真值视频过同一条管线)…")
        report["instrument_floor"] = measure_floor(
            args.gt_h5, test_idx, win, meta, args.device,
            args.floor_max_samples, args.seed, workdir)
    else:
        report["instrument_floor"] = {
            "quantization_floor_deg": 22.5 if int(meta.get("angle_quantize", 0) or 0) == 0 else 0.0,
            "random_expectation_deg": 90.0,
            "measured": None,
            "note": "--floor none:未实测地板,分数缺少下端参照,不得直接写进论文表",
        }
    report["accepted"] = not rej.errors
    report["verdict"] = "ACCEPTED" if report["accepted"] else "REJECTED"
    report["elapsed_seconds"] = round(time.time() - t0, 2)

    doc, nonfinite = json_sanitize(report)
    if nonfinite:
        # 只在报告里留痕,不改判:PSNR=inf 是"逐帧完全一致"的合法结果,
        # FID=nan 是"权重取不到"的合法缺失 —— 两者都不该让提交被拒。
        doc["warnings"].append({
            "code": "NON_FINITE_METRIC",
            "detail": f"{len(nonfinite)} 个指标值是 NaN/Inf,已写成 null 以保证 JSON 合法",
            "examples": nonfinite[:20]})
    Path(args.output).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

    # stdout 只打聚合(逐样本行在 JSON 里)
    brief = {k: v for k, v in ta.items() if k != "per_sample"}
    print("\n===== Track A =====")
    print(json.dumps(brief, indent=2, ensure_ascii=False))
    print("===== Track B =====")
    print(json.dumps({k: v for k, v in tb.items() if k != "per_sample"},
                     indent=2, ensure_ascii=False))
    fl = report["instrument_floor"]
    print(f"===== 仪器地板 ===== 量化 {fl.get('quantization_floor_deg')}° / "
          f"随机 {fl.get('random_expectation_deg')}°")
    gate = fl.get("track_b_calibration_gate")
    if gate:
        print(f"TRACK B CALIBRATION: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"\nSUBMISSION: {report['verdict']}  → {args.output}")

    if cleanup:
        shutil.rmtree(cleanup, ignore_errors=True)
    shutil.rmtree(workdir, ignore_errors=True)
    if rej.errors:
        # 评分途中才发现的硬错(全量 NaN 检查等)—— 合同校验那趟看不到,
        # 但结论一样:拒收。指标已落盘供排查,退出码仍是 4。
        for e in rej.errors:
            print(f"  [{e['code']}] {e['detail']}", file=sys.stderr)
        sys.exit(4)
    if args.require_floor_gate and gate and not gate["passed"]:
        for r in gate["reasons"]:
            print(f"  - {r}", file=sys.stderr)
        print("[FATAL] 真值视频在本管线上都拿不到近满分 → 指标失效,提交分数不可解读。",
              file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
