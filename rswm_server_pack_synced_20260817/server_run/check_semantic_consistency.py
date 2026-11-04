#!/usr/bin/env python3
# =============================================================================
# check_semantic_consistency.py — 第四道门:裁剪后语义一致性(v2.7)
#
# 职责:独立复核最终 H5 里每个语义类样本(water/urban/rural)的 video_frames
# 是否仍符合 scene_category 声明 —— 与建库期语义门同一评分器、独立跑一遍
# (建库门决定数据集,本门复核并留证据,与重复/泄漏门同构)。
#
# 2026-07-27 P0 人检坐实的失败模式,本门要能抓住:
#   sample_00009 (water/River): 12 帧 10 帧纯草地 → 同意帧占比应显著低
#   sample_00011 (urban/StorageTanks): 4 帧纯河水 → 同上
#
# 【先校准再当门用】(部署顺序,DEPLOY_D4.md §3b):
#   python server_run/check_semantic_consistency.py \
#       --gt-h5 $OLD_SMOKE/smoke/smoke_geo.h5 --report-only
#   → 表按占比升序;人检已定的 00009/00011 必须排最前且与其余样本明显分离。
#     分不开 = 评分器不可靠,调 prompt 库/阈值,不许带着假门去建库。
#
# 门模式(建库/冒烟链自动调):
#   python server_run/check_semantic_consistency.py --gt-h5 train_geo.h5 \
#       -o semantic_report.json
#   PASS=exit 0;任一样本低于 --min-frac → exit 8(不放行下游)。
#   默认 min-frac = 建库门阈值 - 0.1(复核比建库宽 0.1:同一评分器在
#   边界样本上的浮点抖动不该判死刑;差距大于 0.1 说明建库门没起作用)。
#
# general 类无语义声明,不评(计入 skipped)。
# =============================================================================
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lora_remotesense.semantic_gate import (  # noqa: E402
    SEMGATE_VERSION, WEAK_CLS_DENOM_WARN, WEAK_DENOM_WARN,
    SemanticGate, default_thresholds)

_ALIASES = {"flood": "water"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-h5", required=True)
    ap.add_argument("--min-frac", type=float, default=None,
                    help="门阈值(默认 = 建库阈值-0.1,按类)")
    ap.add_argument("--frames", type=int, default=9, help="每样本抽帧数(均匀含首尾)")
    ap.add_argument("--keys", default=None, help="逗号分隔,只查这些样本")
    ap.add_argument("--max", type=int, default=0, help="每场景最多查 N 个(0=全量)")
    ap.add_argument("--dump-frames", action="store_true",
                    help="[v2.7.7] JSON 里附逐帧诊断(pos/neg/margin/胜出词)——"
                         "校准判读与 prompt 调优的依据")
    ap.add_argument("--report-only", action="store_true",
                    help="校准模式:只打表(按占比升序),不判门、恒 exit 0")
    ap.add_argument("--device", default=None)
    ap.add_argument("-o", "--output", default=None, help="JSON 报告路径")
    args = ap.parse_args()
    # [v2.7.11/A2-C3] 门模式禁部分抽查:--max/--keys 截掉的样本不进任何统计,
    # 留下的都过就发无条件 PASS —— 部分抽查不得签发放行章。抽查请 --report-only。
    if not args.report_only and (args.max or args.keys):
        sys.exit("[FATAL] 门模式禁用 --max/--keys(部分抽查不得签发 PASS);"
                 "抽查请加 --report-only。")

    build_th = default_thresholds()
    gate_th = {c: (args.min_frac if args.min_frac is not None else max(0.0, t - 0.1))
               for c, t in build_th.items()}

    gate = SemanticGate(frames_per_sample=args.frames, device=args.device)
    gate.ensure_ready()

    only = set(k.strip() for k in args.keys.split(",")) if args.keys else None
    rows, skipped, per_scene_cnt = [], 0, {}
    bad_attr = []   # [F5] (key, 原始值|None):类别缺失或不在合法集
    enabled_mismatch = []   # [v2.7.11] semgate_enabled=true 但 per-sample 字段缺失
    n_missing_scene = 0     # [C4] 无 full_scene → 保真比轨不可用的样本数
    with h5py.File(args.gt_h5, "r") as f:
        data = f["data"] if "data" in f else f
        # [v2.7.11 兼容矩阵] metadata.semgate_enabled:
        #   * 旧 v261 H5 无此 attr(None)→ stored_*=missing 合法,不判缺陷;
        #   * True → 语义类样本必须有 semgate_frac(建库声称开门却无证据=硬伤);
        #   * False → 字段缺失合法(关门构建,数据本身不得用于论文另有大声警告)。
        _sg_enabled = None
        if "metadata" in f and "semgate_enabled" in f["metadata"].attrs:
            _sg_enabled = bool(f["metadata"].attrs["semgate_enabled"])
        for key in sorted(data.keys()):
            if only and key not in only:
                continue
            g = data[key]
            # [v2.7.1 F5/P1 修复] 不再把缺失/拼错的类别默认成 general 静默跳过:
            # general 有意不设门,但「attr 丢失」「watre 这类拼错」是数据缺陷,
            # 收集后按门模式硬失败(report-only 只大声警告,旧 H5 校准仍可跑)。
            _raw = g.attrs.get("source_scene_category", None)
            sc = _ALIASES.get(str(_raw), str(_raw)) if _raw is not None else None
            if sc == "general":
                skipped += 1
                continue
            if sc is None or sc not in build_th:
                bad_attr.append((key, None if _raw is None else str(_raw)))
                continue
            if args.max and per_scene_cnt.get(sc, 0) >= args.max:
                continue
            # [v2.7.7] 双轨:source_class 保持度 + 按建库数据集选 prompt 池
            _src_cls = (str(g.attrs["source_class"])
                        if "source_class" in g.attrs else None)
            # [v2.7.8 R3] source_dataset 缺失 = 数据缺陷,不许静默回退 union 库
            # (跨数据集类污染正类)。v2.4+ 构建必有该 attr。
            if "source_dataset" not in g.attrs:
                bad_attr.append((key, "<missing source_dataset>"))
                continue
            _src_ds = str(g.attrs["source_dataset"])
            if _sg_enabled is True and "semgate_frac" not in g.attrs:
                enabled_mismatch.append(key)
            # [A2-N5②] 场景配额在全部缺陷检查后才占用,缺陷样本不挤掉合法样本
            per_scene_cnt[sc] = per_scene_cnt.get(sc, 0) + 1
            # [v2.7.9] full_scene 参照:保真比 = 声明池质量(帧)/质量(源场景),
            # 零样本固有混淆(船排≈停车场)在比值里抵消
            _scene = g["full_scene"][:] if "full_scene" in g else None
            if _scene is None:
                n_missing_scene += 1   # [C4] 保真比轨静默全空的唯一入口,必须显式计数
            v = gate.check_frames(g["video_frames"][:], sc,
                                  source_class=_src_cls, dataset=_src_ds,
                                  full_scene=_scene)
            # [v2.7 自查修复] 复核阈值优先用【该样本建库时记录的阈值】-0.1:
            # 建库跑在 PBS(可能带 RSWM_SEMGATE_MIN_FRAC_* 覆盖),复核可能在
            # 登录节点(env 不同)—— 只认当前 env 会让两道门口径漂移。
            # 旧 H5 无该 attr(校准场景)才回退 env 推导值;--min-frac 显式指定
            # 时永远最优先。
            _bt = (float(g.attrs["semgate_threshold"])
                   if "semgate_threshold" in g.attrs else None)
            _th = (args.min_frac if args.min_frac is not None
                   else max(0.0, _bt - 0.1) if _bt is not None
                   else gate_th[sc])
            rows.append({
                "key": key, "scene": sc, "frac": v.frac, "n_scored": v.n_scored,
                "source_class": str(g.attrs.get("source_class", "?")),
                "build_frac": (float(g.attrs["semgate_frac"])
                               if "semgate_frac" in g.attrs else None),
                "build_threshold": _bt,
                "gate_threshold": _th,
                "ok": bool(v.frac is not None and v.frac >= _th),
                # [v2.7.7 双轨+诊断]
                "mean_margin": v.mean_margin,
                "retention_frac": v.retention_frac,
                # [v2.7.11] 重算值与建库值分开报,drift 显式 —— 模型/taxonomy/
                # TAU/transformers 版本变化都会漂,不许用重算值冒充建库证据
                "recomputed_pres_ratio": v.pres_ratio,
                "recomputed_pres_ratio_raw": v.pres_ratio_raw,
                "recomputed_scene_pool_mass": v.scene_pool_mass,
                "stored_pres_ratio": (float(g.attrs["semgate_pres_ratio"])
                                      if "semgate_pres_ratio" in g.attrs else None),
                "stored_pres_ratio_raw": (float(g.attrs["semgate_pres_ratio_raw"])
                                          if "semgate_pres_ratio_raw" in g.attrs else None),
                "stored_scene_pool_mass": (float(g.attrs["semgate_scene_pool_mass"])
                                           if "semgate_scene_pool_mass" in g.attrs else None),
                "pres_drift": (round(v.pres_ratio - float(g.attrs["semgate_pres_ratio"]), 4)
                               if (v.pres_ratio is not None
                                   and "semgate_pres_ratio" in g.attrs) else None),
                # 向后兼容旧键名(校准器读 pres_ratio)
                "pres_ratio": v.pres_ratio,
                "pres_ratio_raw": v.pres_ratio_raw,
                "scene_pool_mass": v.scene_pool_mass,
                # [v2.7.12] 类别保真三键(四轮校准的新特征)
                "cls_pres_ratio": v.cls_pres_ratio,
                "cls_pres_ratio_raw": v.cls_pres_ratio_raw,
                "scene_cls_mass": v.scene_cls_mass,
                "bank_dataset": v.bank_dataset,
                **({"frames": v.details} if args.dump_frames else {}),
            })

    rows.sort(key=lambda r: (r["frac"] if r["frac"] is not None else 2.0))
    print(f"\n=== 语义一致性({SEMGATE_VERSION};复核阈值=样本建库阈值-0.1,"
          f"无记录时回退 {gate_th};当前 env 建库阈值 {build_th})===")
    print(f"{'样本':<16}{'场景':<8}{'源类':<20}{'同意占比':>8}{'保真比':>8}"
          f"{'类保真':>8}{'margin':>8}{'保持度':>8}{'建库占比':>10}  判定")
    for r in rows:
        bf = f"{r['build_frac']:.2f}" if r["build_frac"] is not None else "  - "
        mark = "PASS" if r["ok"] else "★FAIL"
        mm = f"{r['mean_margin']:+.3f}" if r.get("mean_margin") is not None else "   -"
        rf = f"{r['retention_frac']:.2f}" if r.get("retention_frac") is not None else "  -"
        pr = f"{r['pres_ratio']:.2f}" if r.get("pres_ratio") is not None else "  -"
        cp = f"{r['cls_pres_ratio']:.2f}" if r.get("cls_pres_ratio") is not None else "  -"
        print(f"{r['key']:<16}{r['scene']:<8}{r['source_class']:<20}"
              f"{r['frac']:>8.2f}{pr:>8}{cp:>8}{mm:>8}{rf:>8}{bf:>10}  {mark}")
    fails = [r for r in rows if not r["ok"]]
    n_sem = len(rows)
    print(f"\n合计:语义类 {n_sem} 个(general/未知 跳过 {skipped}),"
          f"FAIL {len(fails)} 个")

    if args.output:
        Path(args.output).write_text(json.dumps({
            "semgate_version": SEMGATE_VERSION, "gate_thresholds": gate_th,
            "build_thresholds": build_th, "n_semantic": n_sem,
            "n_skipped_general": skipped, "n_fail": len(fails),
            "report_only": bool(args.report_only),
            "bad_attr": [list(x) for x in bad_attr],
            "enabled_mismatch": enabled_mismatch,
            "scope": {"max": args.max, "keys": args.keys,
                      "n_examined": n_sem, "n_bad_attr": len(bad_attr),
                      "n_missing_scene": n_missing_scene},
            "rows": rows,
        }, indent=2, ensure_ascii=False))
        print(f"报告 → {args.output}")

    # [v2.7.10 ⑤] 分母弱警示:场景本身声明池质量小 → 高保真比不可轻信
    weak_den = [r for r in rows
                if r.get("scene_pool_mass") is not None and r["scene_pool_mass"] < 0.05]
    if weak_den:
        print(f"\n[⚠️ 分母弱] {len(weak_den)} 个样本的源场景声明池质量 <{WEAK_DENOM_WARN}"
              f"(如 {[(r['key'], round(r['scene_pool_mass'], 3)) for r in weak_den[:5]]})"
              f" —— 这些样本的保真比≈1 可能是分母太小,不是真保留;校准判读时单看。",
              file=sys.stderr)
    if bad_attr:
        print(f"\n[F5] {len(bad_attr)} 个样本的 source_scene_category 缺失或非法"
              f"(如 {bad_attr[:5]})—— 这不是 general,是数据缺陷。", file=sys.stderr)
    if n_missing_scene:
        print(f"\n[⚠️ C4] {n_missing_scene}/{n_sem} 个语义类样本无 full_scene —— "
              f"这些样本的保真比轨【整条为空】(不是 0,是没算)。占比高时 pres 族"
              f"规则不可用;旧版源 H5 属预期,新建 H5 出现即建库缺陷。", file=sys.stderr)
    # [v2.7.12/C3] 类保真的分母塌陷独立警示:源类在 full_scene 里就没被认出
    # (scene_cls_mass 极小)时,噪声帧也能拿"类保真=1.00"满分 —— raw 一并打出
    weak_cls = [r for r in rows
                if r.get("scene_cls_mass") is not None
                and r["scene_cls_mass"] < WEAK_CLS_DENOM_WARN]
    if weak_cls:
        print(f"\n[⚠️ 类分母弱] {len(weak_cls)} 个样本 scene_cls_mass<"
              f"{WEAK_CLS_DENOM_WARN}(如 "
              f"{[(r['key'], round(r['scene_cls_mass'], 4), 'raw=' + str(r.get('cls_pres_ratio_raw'))) for r in weak_cls[:4]]}"
              f")—— 类保真≈1 可能是源类本就没被认出,判读时单看。", file=sys.stderr)
    if enabled_mismatch:
        print(f"\n[v2.7.11] {len(enabled_mismatch)} 个语义类样本:metadata 声称 "
              f"semgate_enabled=true 但无 semgate_frac 建库证据"
              f"(如 {enabled_mismatch[:5]})—— 证据链断裂。", file=sys.stderr)
    if args.report_only:
        print("\n[校准模式] 不判门。核对:人检 FAIL 的样本(如 00009/00011)是否"
              "排在最前且与其余样本明显分离;分不开就调 prompt 库/阈值。")
        return 0
    if bad_attr:
        print("SEMANTIC GATE: FAIL —— 类别 attr 缺失/非法,见上。", file=sys.stderr)
        return 8
    if enabled_mismatch:
        print("SEMANTIC GATE: FAIL —— semgate_enabled=true 但建库证据缺失,见上。",
              file=sys.stderr)
        return 8
    # [F5] 空转守卫:0 个语义类样本受检 = 门没有检查任何东西,不许打 PASS
    if n_sem == 0:
        print("SEMANTIC GATE: FAIL —— 0 个语义类样本受检(attr 全丢或 H5 结构"
              "异构),门空转不算通过。", file=sys.stderr)
        return 8
    if fails:
        print(f"\nSEMANTIC GATE: FAIL —— {len(fails)} 个样本裁剪后语义与声明不符"
              f"(如 {[r['key'] for r in fails[:5]]})。该数据集不放行:\n"
              f"  * 建库门开着还漏到这里 → 评分器/阈值两处口径漂了,先对齐;\n"
              f"  * 建库门关着(RSWM_SEMGATE=0)→ 这就是后果,开门重建。",
              file=sys.stderr)
        return 8
    print("SEMANTIC GATE: PASS(仅第四道门;前三道见 duplicate/split 报告)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
