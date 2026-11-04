#!/usr/bin/env python3
# =============================================================================
# semgate_build_report.py — 受控 P0 语义门建库统计(v2.7.13,拍板验收要求)
#
# 产出(拍板验收清单原文):
#   * 各 source_class 的尝试数、拒绝数、最终接受数;
#   * Port/BareLand/Farmland 型(已知误拒类)的重采次数单列;
#   * 是否接近源池耗尽(最大 tries 对照 RSWM_SEMGATE_MAX_TRIES);
#   * 全部最终语义样本的 semgate_frac 列表与分布。
# 信息性报告(不判门),JSON + 可读表双输出。
#
# 用法:
#   python server_run/semgate_build_report.py --gt-h5 $WORK/data/train_geo.h5 \
#       [--attempts $WORK/tmp_bench_*/semgate_attempts.json] -o $WORK/semgate_build_report.json
#   attempts 文件缺失时仍出接受侧统计(拒绝侧标 unavailable)。
# =============================================================================
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import h5py

KNOWN_FALSE_REJECT_CLASSES = ("port", "bareland", "farmland")   # 归一化名


def _norm(c):
    return "".join(ch.lower() for ch in str(c) if ch.isalnum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("--attempts", default=None,
                    help="semgate_attempts.json(建库 wrapper 落盘;缺省自动在 H5 同级/上级找)")
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    h5p = Path(args.gt_h5)
    att_path = Path(args.attempts) if args.attempts else None
    if att_path is None:
        for cand in [h5p.parent / "semgate_attempts.json",
                     h5p.parent.parent / "semgate_attempts.json"]:
            if cand.is_file():
                att_path = cand
                break
    attempts = None
    if att_path and att_path.is_file():
        attempts = json.loads(att_path.read_text())["attempts"]

    accepted = defaultdict(lambda: {"n": 0, "tries": [], "fracs": []})
    accepted_sids = set()
    fracs_all, max_tries_seen = [], 0
    with h5py.File(args.gt_h5, "r") as f:
        data = f["data"]
        enabled = (bool(f["metadata"].attrs["semgate_enabled"])
                   if "metadata" in f and "semgate_enabled" in f["metadata"].attrs
                   else None)
        for k in sorted(data.keys()):
            g = data[k]
            sc = str(g.attrs.get("source_scene_category", "?"))
            cls = _norm(g.attrs.get("source_class", "?"))
            if "source_scene_id" in g.attrs:
                accepted_sids.add((sc, int(g.attrs["source_scene_id"])))
            if "semgate_frac" not in g.attrs:
                continue
            fr = float(g.attrs["semgate_frac"])
            tr = int(g.attrs.get("semgate_tries", 1))
            a = accepted[(sc, cls)]
            a["n"] += 1
            a["tries"].append(tr)
            a["fracs"].append(fr)
            fracs_all.append(fr)
            max_tries_seen = max(max_tries_seen, tr)

    # [v2.8.2 N5] overlap 耗尽消耗掉的源图必须从"语义门拒绝"里扣掉。
    # 本报告的口径是"registry 里有、accepted 里没有 = 被语义门拒了",但 v2.8 起
    # overlap enforce 也会 continue 掉源图(见 v5 的 n_source_exhausted),而那些
    # sid 同样已经过 _hd_load_scene_from_hf 登记 → 会被按 AID class 计进语义门
    # 拒绝率。量级不大(canonical 500 样本约 0.4 次),但它污染的正是 P0 人检的
    # 验收证据表 —— 让人以为某个类被语义门系统性拒了。
    _ov_exhausted = set()
    _ov_legacy_int = set()
    _semantic_fallbacks = []
    _sampling_retry_cap = None
    _ovp = (att_path.parent / "overlap_audit.json") if att_path is not None else None
    if _ovp is not None and _ovp.is_file():
        try:
            _ov_doc = json.loads(_ovp.read_text())
            _raw = _ov_doc.get("overlap_exhausted_sids", [])
            _semantic_fallbacks = list(_ov_doc.get("semantic_fallbacks", []))
            if _ov_doc.get("max_retry_per_sample") is not None:
                _sampling_retry_cap = int(_ov_doc["max_retry_per_sample"])
            # [v2.9.1 C3] 新格式 "scene|sid";兼容旧的裸 int(那时会跨场景误扣,
            # 只能退化成"任意场景的该 sid",并在下面提示)。
            _ov_exhausted = {str(x) for x in _raw}
            _ov_legacy_int = {str(x) for x in _raw if str(x).isdigit()}
            if _ov_legacy_int:
                print(f"[warn] overlap_exhausted_sids 含 {len(_ov_legacy_int)} 个旧格式"
                      f"裸 sid(无场景维度),扣除会跨场景误伤;请用 v2.9.1+ 重建")
        except Exception as _e:
            print(f"[warn] 读取 overlap_audit.json 失败,未扣除 overlap 耗尽源图: {_e}")
    _n_ov_ded = 0
    rejected = defaultdict(int)
    if attempts is not None:
        for key, rec in attempts.items():
            sc, sid = key.rsplit("|", 1)
            if (sc, int(sid)) in accepted_sids:
                continue
            if f"{sc}|{int(sid)}" in _ov_exhausted or str(int(sid)) in _ov_legacy_int:
                _n_ov_ded += 1          # overlap 耗尽,不是语义门的锅
                continue
            rejected[(sc, _norm(rec.get("class")))] += 1
    if _n_ov_ded:
        print(f"[注] 已从语义门拒绝中扣除 {_n_ov_ded} 张被 overlap enforce 耗尽的源图")
    if _semantic_fallbacks:
        print(f"[WARN] {len(_semantic_fallbacks)} 个样本使用了有界 "
              "best-semantic-candidate fallback;详见 overlap_audit.json")

    max_cfg = (_sampling_retry_cap if _sampling_retry_cap is not None
               else min(int(os.environ.get("RSWM_SEMGATE_MAX_TRIES", "6")), 20))
    print(f"semgate_enabled={enabled}  语义样本 {len(fracs_all)} 个  "
          f"最大重采 {max_tries_seen}/{max_cfg}"
          + ("  ⚠️接近池耗尽路径" if max_tries_seen >= max_cfg - 1 else ""))
    print(f"{'场景':<8}{'source_class':<20}{'接受':>5}{'拒绝':>5}{'尝试':>5}"
          f"{'均tries':>8}{'均frac':>8}  备注")
    rows = []
    all_keys = sorted(set(list(accepted) + list(rejected)))
    for sc, cls in all_keys:
        a = accepted.get((sc, cls), {"n": 0, "tries": [], "fracs": []})
        rej = rejected.get((sc, cls), None if attempts is None else 0)
        n_att = (a["n"] + rej) if rej is not None else None
        note = "★已知误拒类" if cls in KNOWN_FALSE_REJECT_CLASSES else ""
        mt = f"{np.mean(a['tries']):.2f}" if a["tries"] else "  -"
        mf = f"{np.mean(a['fracs']):.2f}" if a["fracs"] else "  -"
        print(f"{sc:<8}{cls:<20}{a['n']:>5}"
              f"{rej if rej is not None else '  ?':>5}"
              f"{n_att if n_att is not None else '  ?':>5}{mt:>8}{mf:>8}  {note}")
        rows.append({"scene": sc, "class": cls, "accepted": a["n"],
                     "rejected": rej, "attempts": n_att,
                     "mean_tries": (float(np.mean(a["tries"])) if a["tries"] else None),
                     "mean_frac": (float(np.mean(a["fracs"])) if a["fracs"] else None),
                     "known_false_reject_class": cls in KNOWN_FALSE_REJECT_CLASSES})
    if fracs_all:
        arr = np.array(fracs_all)
        print(f"\nsemgate_frac 分布: min={arr.min():.2f} median={np.median(arr):.2f} "
              f"mean={arr.mean():.2f} max={arr.max():.2f}(n={len(arr)})")
    if attempts is None:
        print("[注意] 无 semgate_attempts.json —— 拒绝侧统计 unavailable"
              "(接受侧 tries 仍能反映重采强度)")
    if args.output:
        Path(args.output).write_text(json.dumps({
            "semgate_enabled": enabled, "n_semantic": len(fracs_all),
            "max_tries_seen": max_tries_seen, "max_tries_cfg": max_cfg,
            "semantic_fallback_count": len(_semantic_fallbacks),
            "semantic_fallbacks": _semantic_fallbacks,
            "attempts_available": attempts is not None,
            "per_class": rows,
            "semgate_frac_all": fracs_all,
        }, indent=2, ensure_ascii=False))
        print(f"报告 → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
