#!/usr/bin/env python3
# =============================================================================
# visualize_benchmark.py — 最终 H5 的 benchmark 可视化(v2.4.5,纯 CPU)
#
# 动机:旧可视化(wan/remotesensebenchmark/visualize_multi_direction_*.py)只认
# 中间产物 benchmark.h5(sliding_frames 格式),而 v2.4.4 起中间产物建完即删;
# 最终交付 H5(data/sample_XXXXX: video_frames/full_scene/trajectory/waypoints
# + source_* attrs)此前没有任何查看工具。本脚本补上:
#   1. 样张网格(默认每张大图 8 个样本,inode 友好):
#      左 = full_scene(512²)+ 轨迹折线(黄)+ waypoints(红点编号)
#          + 起点(绿)/终点(蓝)+ 起始窗口框(青,按 pose_window_size);
#      右 = video_frames 首/中/末三帧缩略;
#      注 = key/场景/方向词/段数/净位移比/帧数/源图 class@dataset。
#   2. summary.json:每场景计数、方向词分布、段数分布、净位移比统计、
#      源 class 多样性、metadata 回显、缺字段计数。
# 依赖:numpy + h5py + Pillow(不需要 cv2/matplotlib/torch)。
#
# 用法(任意 tier 的最终 H5 通用):
#   python server_run/visualize_benchmark.py --gt-h5 $WORK/data/train_geo.h5 \
#       -o $WORK/viz --max 24
#   python server_run/visualize_benchmark.py --gt-h5 ... --scene water --max 8
#   python server_run/visualize_benchmark.py --gt-h5 ... --keys sample_00015,sample_00140
#   python server_run/visualize_benchmark.py --gt-h5 ... --summary-only
# =============================================================================
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import h5py
from PIL import Image, ImageDraw, ImageFont

SCENE_TH = 256          # 场景缩略边长(512² → 256²,坐标等比缩)
FRAME_W = 256           # 帧缩略宽
CAP_H = 58              # 注释条高
PAD = 8

_SCENE_ALIASES = {"flood": "water"}   # [v2.5] 旧 H5 读侧归一


def _canon_scene(x):
    return _SCENE_ALIASES.get(str(x), str(x))


_DIR_RE = re.compile(r"toward the (east|north|west|south)|"
                     r"\b(east|north|west|south)ward\b")


def parse_dir(prompt):
    m = _DIR_RE.search(str(prompt))
    return (m.group(1) or m.group(2)) if m else "?"


def net_ratio(wp):
    if wp is None or len(wp) < 2:
        return None
    wp = np.asarray(wp, dtype=np.float64)
    arc = float(np.sum(np.linalg.norm(np.diff(wp, axis=0), axis=1)))
    return float(np.linalg.norm(wp[-1] - wp[0]) / arc) if arc > 0 else 0.0


def to_img(arr):
    a = np.asarray(arr)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return Image.fromarray(a[..., :3])


def draw_panel(g, key, win_hw, font):
    """单样本面板 → PIL Image。坐标约定:trajectory/waypoints = (cy, cx) 场景系。"""
    # ---- 左:场景 + 轨迹叠加 ----
    if "full_scene" in g:
        scene = to_img(g["full_scene"][:])
        scene_note = None
    else:
        scene = Image.new("RGB", (512, 512), (60, 60, 60))
        scene_note = "no full_scene (旧 H5)"
    sw, sh = scene.size
    k = SCENE_TH / max(sw, sh)
    scene = scene.resize((int(sw * k), int(sh * k)))
    d = ImageDraw.Draw(scene)

    def pt(p):        # (cy, cx) → 画布 (x, y)
        return (float(p[1]) * k, float(p[0]) * k)

    traj = g["trajectory"][:] if "trajectory" in g else None
    wp = g["waypoints"][:] if "waypoints" in g else None
    if traj is not None and len(traj) >= 2:
        d.line([pt(p) for p in traj], fill=(255, 220, 0), width=2)
    if wp is not None:
        for i, p in enumerate(wp):
            x, y = pt(p)
            d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=(255, 60, 60), width=2)
            d.text((x + 5, y - 6), str(i), fill=(255, 60, 60), font=font)
    ref = traj if (traj is not None and len(traj)) else wp
    if ref is not None and len(ref):
        x0, y0 = pt(ref[0]); x1, y1 = pt(ref[-1])
        d.ellipse([x0 - 5, y0 - 5, x0 + 5, y0 + 5], fill=(0, 220, 0))
        d.ellipse([x1 - 5, y1 - 5, x1 + 5, y1 + 5], fill=(60, 120, 255))
        if win_hw:  # 起始窗口框(尺度感)
            wh, ww = win_hw[0] * k / 2, win_hw[1] * k / 2
            d.rectangle([x0 - ww, y0 - wh, x0 + ww, y0 + wh],
                        outline=(0, 230, 230), width=1)
    if scene_note:
        d.text((6, 6), scene_note, fill=(255, 120, 120), font=font)

    # ---- 右:首/中/末帧 ----
    vf = g["video_frames"]
    T = vf.shape[0]
    idxs = [0, T // 2, T - 1]
    thumbs = []
    for i in idxs:
        im = to_img(vf[i])
        fw, fh = im.size
        th = max(1, int(fh * FRAME_W / fw))
        thumbs.append((i, im.resize((FRAME_W, th))))
    fr_h = sum(t.size[1] for _, t in thumbs) + PAD * (len(thumbs) - 1)

    # ---- 注释 ----
    prompt = str(g.attrs.get("prompt", ""))
    nseg = g.attrs.get("num_segments", "?")
    nr = net_ratio(wp)
    src = ""
    if "source_index" in g.attrs:
        src = (f"{g.attrs.get('source_class', '?')}@"
               f"{str(g.attrs.get('source_dataset', '?')).split('/')[-1]}"
               f"#{g.attrs.get('source_index', '?')}")
    nr_s = f"{nr:.2f}" if nr is not None else "n/a"
    cap1 = (f"{key}  [{g.attrs.get('source_scene_category', '?')}]  "
            f"dir={parse_dir(prompt)}  K={nseg}  net_ratio={nr_s}")
    cap2 = f"T={T}  frame={vf.shape[1]}x{vf.shape[2]}  src={src or 'n/a'}"

    W = SCENE_TH + PAD + FRAME_W + PAD * 2
    H = max(SCENE_TH, fr_h) + CAP_H + PAD * 2
    panel = Image.new("RGB", (W, H), (24, 24, 28))
    panel.paste(scene, (PAD, PAD))
    y = PAD
    dd = ImageDraw.Draw(panel)
    for i, t in thumbs:
        panel.paste(t, (SCENE_TH + PAD * 2, y))
        dd.text((SCENE_TH + PAD * 2 + 3, y + 2), f"f{i}",
                fill=(255, 255, 0), font=font)
        y += t.size[1] + PAD
    ytxt = max(SCENE_TH, fr_h) + PAD + 4
    dd.text((PAD, ytxt), cap1, fill=(230, 230, 230), font=font)
    dd.text((PAD, ytxt + 16), cap2, fill=(160, 200, 255), font=font)
    dd.text((PAD, ytxt + 32), prompt[:96], fill=(150, 150, 150), font=font)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("-o", "--out-dir", default=None,
                    help="默认 <H5 目录>/viz")
    ap.add_argument("--max", type=int, default=24, help="最多可视化样本数")
    ap.add_argument("--per-sheet", type=int, default=8, help="每张大图样本数")
    ap.add_argument("--scene", default=None,
                    help="只看某场景 (general|water|urban|rural;旧数据 flood 自动归一)")
    ap.add_argument("--keys", default=None, help="逗号分隔的样本 key 列表")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    h5p = Path(args.gt_h5)
    out = Path(args.out_dir) if args.out_dir else h5p.parent / "viz"
    out.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    with h5py.File(h5p, "r") as f:
        data = f["data"] if "data" in f else f
        meta = dict(f["metadata"].attrs) if "metadata" in f else {}
        win_hw = tuple(int(v) for v in meta.get("pose_window_size", ())) or None
        keys = sorted(data.keys())

        # ---- 统计摘要(全量,不受 --max 影响)----
        sc_cnt, dir_cnt, k_cnt, cls_cnt = Counter(), Counter(), Counter(), Counter()
        ratios, miss_traj, miss_scene, miss_src = [], 0, 0, 0
        for kk in keys:
            g = data[kk]
            sc_cnt[_canon_scene(g.attrs.get("source_scene_category", "?"))] += 1
            dir_cnt[parse_dir(g.attrs.get("prompt", ""))] += 1
            k_cnt[int(g.attrs.get("num_segments", -1))] += 1
            if "source_class" in g.attrs:
                cls_cnt[str(g.attrs["source_class"])] += 1
            else:
                miss_src += 1
            r = net_ratio(g["waypoints"][:] if "waypoints" in g else None)
            if r is not None:
                ratios.append(r)
            miss_traj += 0 if "trajectory" in g else 1
            miss_scene += 0 if "full_scene" in g else 1
        summary = {
            "h5": str(h5p), "n_samples": len(keys),
            "metadata": {k: (v.tolist() if hasattr(v, "tolist") else str(v))
                         for k, v in meta.items()},
            "per_scene": dict(sc_cnt), "per_direction": dict(dir_cnt),
            "num_segments": {str(k): v for k, v in sorted(k_cnt.items())},
            "net_ratio": ({"mean": float(np.mean(ratios)),
                           "min": float(np.min(ratios)),
                           "max": float(np.max(ratios)),
                           "pct_below_0.2": float(np.mean(np.array(ratios) < 0.2))}
                          if ratios else None),
            "unique_source_classes": len(cls_cnt),
            "source_class_top10": dict(cls_cnt.most_common(10)),
            "missing": {"trajectory": miss_traj, "full_scene": miss_scene,
                        "source_attrs": miss_src},
        }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        if args.summary_only:
            print(f"\n[done] summary → {out/'summary.json'}")
            return

        # ---- 选样(确定性:排序后均匀抽,不用随机)----
        if args.keys:
            sel = [k for k in args.keys.split(",") if k in data]
            missing = [k for k in args.keys.split(",") if k and k not in data]
            if missing:
                print(f"[warn] 不存在的 key: {missing}", file=sys.stderr)
        else:
            pool = [k for k in keys if not args.scene or
                    _canon_scene(data[k].attrs.get("source_scene_category", "")) == _canon_scene(args.scene)]
            step = max(1, len(pool) // max(1, args.max))
            sel = pool[::step][: args.max]
        if not sel:
            sys.exit("[FATAL] 没有可视化样本(--scene/--keys 过滤后为空)")

        # ---- 出图 ----
        sheets = [sel[i:i + args.per_sheet]
                  for i in range(0, len(sel), args.per_sheet)]
        for si, batch in enumerate(sheets):
            panels = [draw_panel(data[kk], kk, win_hw, font) for kk in batch]
            cols = 2
            rows = (len(panels) + cols - 1) // cols
            pw = max(p.size[0] for p in panels)
            ph = max(p.size[1] for p in panels)
            sheet = Image.new("RGB", (cols * pw + PAD * (cols + 1),
                                      rows * ph + PAD * (rows + 1)), (10, 10, 12))
            for i, p in enumerate(panels):
                r, c = divmod(i, cols)
                sheet.paste(p, (PAD + c * (pw + PAD), PAD + r * (ph + PAD)))
            fp = out / f"sheet_{si:02d}.png"
            sheet.save(fp)
            print(f"[sheet] {fp}  ({len(batch)} samples, {sheet.size[0]}x{sheet.size[1]})")
    print(f"\n[done] {len(sel)} 样本 → {out}(每 {args.per_sheet} 样本一张大图 + summary.json)")


if __name__ == "__main__":
    main()
