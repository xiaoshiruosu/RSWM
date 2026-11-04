#!/usr/bin/env python3
# =============================================================================
# verify_full_scene_provenance.py — full_scene 血统【一致性检查】(v2.7.11)
#
# 保真比(pres_ratio)的分母是 full_scene 的声明池质量 —— 本脚本验证
# full_scene 与 video_frames 的生成关系一致(不是首帧/均值/缩略图冒充)。
# 措辞口径(审查方定):这是【一致性检查/强佐证】,不是严格证明。
#
# 验证顺序(审查方定稿):
#   ① 坐标重建(优先):H5 存有 trajectory(逐帧窗口中心 (cy,cx))与
#      metadata.pose_window_size —— 按 v5 _get_bounds 同款取整裁 full_scene,
#      用生成链同款插值(cv2.INTER_LINEAR,见 _resize_frame)放大到帧尺寸,
#      与存储帧逐像素比。这是最强的一致性证据。
#   ② 模板匹配(回退/交叉验证):坐标缺失时,把帧降采样回窗口尺寸,在
#      full_scene 里 matchTemplate 定位;检查唯一性(最优 vs 次优差距)。
#
# 输出:全部抽查样本的误差分布(min/median/p90/p95/max)+ 定位稳定性。
# MAE 阈值 12 为【暂定值】—— 以真实数据分布为准,过宽/过严都按分布重定。
#
# 退出码:0=一致(或 WARN 供判读);10=硬伤(定位不稳/超阈值/属性缺失或非法/
# dtype 异常/多位置同优)。
#
# 用法:
#   python server_run/verify_full_scene_provenance.py --gt-h5 <smoke_geo.h5> \
#       [--keys k1,k2] [--frames-per-sample 3] [--mae-tol 12]
# =============================================================================
import argparse
import sys
from pathlib import Path

import numpy as np
import h5py


def _bounds(cy, cx, win_h, win_w, H, W):
    """与 v5 _get_bounds 逐字节同款的取整规则。"""
    return (max(0, cy - win_h // 2), min(H, cy + win_h // 2),
            max(0, cx - win_w // 2), min(W, cx + win_w // 2))


def _resize_linear(img, w, h):
    import cv2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _match_fallback(full, frame_small):
    """模板匹配回退:返回 (loc, sqdiff, mae, second_gap)。"""
    import cv2
    res = cv2.matchTemplate(full, frame_small, cv2.TM_SQDIFF_NORMED)
    flat = res.flatten()
    order = np.argsort(flat)
    min_val = float(flat[order[0]])
    # 次优:与最优位置至少隔半个窗口,避免相邻像素伪次优
    h, w = frame_small.shape[:2]
    y0, x0 = np.unravel_index(order[0], res.shape)
    second = None
    for idx in order[1:]:
        y, x = np.unravel_index(idx, res.shape)
        if abs(int(y) - int(y0)) > h // 2 or abs(int(x) - int(x0)) > w // 2:
            second = float(flat[idx])
            break
    patch = full[y0:y0 + h, x0:x0 + w]
    mae = float(np.abs(patch.astype(np.float32)
                       - frame_small.astype(np.float32)).mean())
    gap = (second - min_val) if second is not None else float("inf")
    return (int(x0), int(y0)), min_val, mae, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("--keys", default=None)
    ap.add_argument("--frames-per-sample", type=int, default=3,
                    help="每样本抽查帧数(首/中/末)")
    ap.add_argument("--mae-tol", type=float, default=12.0,
                    help="暂定阈值;以真实分布重定,不是生产硬阈值")
    ap.add_argument("--gap-min", type=float, default=0.02,
                    help="模板匹配最优 vs 次优的最小差距(唯一性)")
    args = ap.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("[STOP] 需要 cv2(坐标重建与模板匹配都依赖)", file=sys.stderr)
        sys.exit(10)

    all_maes, n_stop, n_warn, rows = [], 0, 0, []
    with h5py.File(args.gt_h5, "r") as f:
        data = f["data"]
        meta = f["metadata"].attrs if "metadata" in f else {}
        if "pose_window_size" not in meta:
            print("[STOP] metadata 无 pose_window_size —— 属性缺失", file=sys.stderr)
            sys.exit(10)
        wh, ww = (int(v) for v in meta["pose_window_size"])
        if wh <= 0 or ww <= 0:
            print("[STOP] pose_window_size 非法", file=sys.stderr)
            sys.exit(10)

        if args.keys:
            keys = [k.strip() for k in args.keys.split(",")]
        else:
            seen, keys = set(), []
            for k in sorted(data.keys()):
                sc = str(data[k].attrs.get("source_scene_category", "?"))
                if sc not in seen:
                    seen.add(sc)
                    keys.append(k)

        for k in keys:
            if k not in data:
                print(f"[STOP] {k}: H5 中不存在该样本键")
                n_stop += 1
                continue
            g = data[k]
            if "full_scene" not in g:
                print(f"[STOP] {k}: 无 full_scene")
                n_stop += 1
                continue
            full = g["full_scene"][:]
            vf = g["video_frames"]
            # [v2.7.11/A3-C1 修复] 全链实际是 float32(0-255):v5 场景 float32、
            # 转换器不做 uint8 化 —— 旧 uint8 硬守卫会把一切合法生产数据判 STOP。
            # 现接受 uint8 与 float32(值域 0-255);float 值域 ≤1.5 = 归一化图,
            # 不是本链产物,仍 STOP。匹配统一转 float32。
            def _img_ok(a):
                if a.dtype == np.uint8:
                    return True
                if a.dtype in (np.float32, np.float64):
                    return float(np.nanmax(a)) > 1.5 if a.size else False
                return False
            if (not _img_ok(full) or full.ndim != 3 or full.shape[2] != 3
                    or vf.ndim != 4 or vf.shape[0] == 0 or vf.shape[3] != 3
                    or not _img_ok(vf[0])):
                print(f"[STOP] {k}: dtype/结构异常 full={full.dtype}{full.shape} "
                      f"frames={vf.dtype}{vf.shape}")
                n_stop += 1
                continue
            full = full.astype(np.float32)
            T = vf.shape[0]
            fidx = sorted(set([0, T // 2, T - 1][:args.frames_per_sample]))
            traj = g["trajectory"][:] if "trajectory" in g else None
            H, W = full.shape[:2]

            for i in fidx:
                frame = vf[i]
                fh, fw = frame.shape[:2]
                if traj is not None and i < len(traj):
                    # ① 坐标重建(优先):裁【实际 bounds】(含边界钳制情形)→
                    # 生成链同款 INTER_LINEAR resize 到帧尺寸 → 逐像素比。
                    # [A2-N2] 钳制窗口不再退回模板匹配(拉伸帧对完整场景匹配
                    # 必然高 MAE+低唯一性 → 合法样本假 STOP);同链重建对钳制
                    # 情形同样成立,因为转换器 resize 的就是这个钳制后的 crop。
                    cy, cx = int(round(float(traj[i][0]))), int(round(float(traj[i][1])))
                    y0, y1, x0, x1 = _bounds(cy, cx, wh, ww, H, W)
                    crop = full[y0:y1, x0:x1]
                    clamp_note = ("(窗口被边界钳制)"
                                  if (crop.shape[0] != wh or crop.shape[1] != ww) else "")
                    recon = _resize_linear(crop, fw, fh)
                    mae = float(np.abs(recon.astype(np.float32)
                                       - frame.astype(np.float32)).mean())
                    method = "coord" + clamp_note
                    loc, sqd, gap, uniq_ok = (x0, y0), 0.0, float("inf"), True
                else:
                    # ② 模板匹配回退
                    small = _resize_linear(frame, ww, wh)
                    loc, sqd, mae, gap = _match_fallback(full, small)
                    method = "template"
                    uniq_ok = gap >= args.gap_min
                all_maes.append(mae)
                ok = mae <= args.mae_tol and uniq_ok
                rows.append((k, i, method, loc, mae, gap, ok))
                if not ok:
                    n_stop += 1 if (mae > 3 * args.mae_tol or not uniq_ok) else 0
                    n_warn += 0 if (mae > 3 * args.mae_tol or not uniq_ok) else 1

    print(f"{'样本':<16}{'帧':>4}  {'方法':<26}{'loc':<12}{'MAE':>8}{'次优差距':>10}  判定")
    for k, i, m, loc, mae, gap, ok in rows:
        gs = f"{gap:.3f}" if gap != float("inf") else "  n/a"
        print(f"{k:<16}{i:>4}  {m:<26}{str(loc):<12}{mae:>8.3f}{gs:>10}  "
              f"{'OK' if ok else '★超界'}")
    if all_maes:
        a = np.array(all_maes)
        print(f"\nMAE 分布: min={a.min():.3f} median={np.median(a):.3f} "
              f"p90={np.percentile(a, 90):.3f} p95={np.percentile(a, 95):.3f} "
              f"max={a.max():.3f}(n={len(a)};暂定阈值 {args.mae_tol},"
              f"以本分布为准重定)")
    if not rows and not n_stop:
        # [A2-C2] 零样本受检不得出具 PASS(与 checker 空转守卫同构)
        print("[STOP] 0 个样本受检 —— 空转不算通过(data 组为空或 keys 全无效)",
              file=sys.stderr)
        return 10
    if n_stop:
        print(f"\n[STOP] {n_stop} 处硬伤(定位不稳/严重超阈值/属性异常)—— "
              f"血统一致性检查失败,保真比判据作废前不得使用", file=sys.stderr)
        return 10
    if n_warn:
        print(f"\n[WARN] {n_warn} 处轻度超阈值 —— 把分布发回判读(可能需要按真实"
              f"分布调 --mae-tol)")
        return 0
    print("\n[PASS] 血统一致性检查通过(坐标重建/模板匹配口径见上表)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
