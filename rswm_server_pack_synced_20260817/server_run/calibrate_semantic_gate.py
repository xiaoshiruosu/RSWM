#!/usr/bin/env python3
# =============================================================================
# calibrate_semantic_gate.py — 用人检真值给语义门选组合规则(v2.7.7)
#
# 背景:v1 手写 prompt 库在 20 样本真值上分不开(7 误拒,BareLand=0.00 比真坏
# 样本还低)。v2 库(taxonomy 派生)重跑后,本脚本把逐样本特征
# (scene_frac / mean_margin / retention_frac)与人检真值对表,网格搜索能
# 【完美分离】的判定规则;找不到就非零退出并给最小错误方案 —— 规则由人拍板,
# 本脚本只提供证据,不擅自启用。
#
# 用法(服务器,先跑 checker 生成带诊断的 scores):
#   python server_run/check_semantic_consistency.py \
#       --gt-h5 <v261smoke>/smoke/smoke_geo.h5 --report-only --dump-frames \
#       -o "$TMPDIR/semcal_v2.json"
#   python server_run/calibrate_semantic_gate.py \
#       --scores "$TMPDIR/semcal_v2.json" \
#       --labels server_run/calibration_labels_smoke_v261.json
#
# 退出码:0=存在完美分离规则(已打印全部+推荐);9=不存在(打印最小错误方案
# 与逐样本特征表,把输出发回继续调 prompt 库)。
# =============================================================================
import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from lora_remotesense.semantic_gate import (WEAK_CLS_DENOM_WARN,
                                                WEAK_DENOM_WARN)
except ImportError:          # 独立拷走本脚本时的回退(与 semantic_gate 同值)
    WEAK_DENOM_WARN = 0.05
    WEAK_CLS_DENOM_WARN = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="checker 的 -o JSON")
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()

    rep = json.loads(Path(args.scores).read_text())
    lab = json.loads(Path(args.labels).read_text())["samples"]
    rows = []
    for r in rep["rows"]:
        k = r["key"]
        if k not in lab:
            continue
        if lab[k]["category"] == "general":
            continue
        rows.append({
            "key": k, "cat": r["scene"],
            "frac": r.get("frac"),
            "margin": r.get("mean_margin"),
            "ret": r.get("retention_frac"),
            "pres": r.get("pres_ratio"),        # [v2.7.9] 场景参照保真比
            "pres_raw": r.get("pres_ratio_raw"),
            "scene_mass": r.get("scene_pool_mass"),
            "clspres": r.get("cls_pres_ratio"),  # [v2.7.12] 类别保真比
            "clspres_raw": r.get("cls_pres_ratio_raw"),
            "scene_cls_mass": r.get("scene_cls_mass"),
            "valid": bool(lab[k]["valid"]),
        })
    if not rows:
        sys.exit("[FATAL] scores 与 labels 无交集(检查 H5/键名)")
    n_bad = sum(1 for r in rows if not r["valid"])
    n_good = len(rows) - n_bad
    # [v2.7.11/A2-C2] 单类样本下任何足够严/松的规则都"完美"(实测 13 全 valid 时
    # 排第一的推荐是 frac>=0.0 = 不设门,还带 [稳定✓] 与 RC=0)。不可校准,硬错。
    if n_bad == 0 or n_good == 0:
        sys.exit(f"[FATAL] 交集里 valid={n_good}/invalid={n_bad} —— 单类样本下"
                 f"任何规则都'完美',不可校准。检查 checker 是否带了 --max/--keys,"
                 f"或 labels 与 scores 是否同一批样本。")
    print(f"语义类样本 {len(rows)} 个,其中人检无效 {n_bad} 个\n")
    print(f"{'样本':<16}{'类':<7}{'frac':>6}{'保真比':>7}{'类保真':>7}{'raw':>7}"
          f"{'场景质量':>7}{'margin':>9}{'保持度':>7}  人检")
    for r in sorted(rows, key=lambda x: (x['frac'] if x['frac'] is not None else 9)):
        mg = f"{r['margin']:+.3f}" if r["margin"] is not None else "    -"
        rt = f"{r['ret']:.2f}" if r["ret"] is not None else "  -"
        pr = f"{r['pres']:.2f}" if r["pres"] is not None else "  -"
        cp = f"{r['clspres']:.2f}" if r.get("clspres") is not None else "  -"
        rw = f"{r['pres_raw']:.2f}" if r.get("pres_raw") is not None else "  -"
        sm_ = f"{r['scene_mass']:.3f}" if r.get("scene_mass") is not None else "  -"
        wk = " ⚠分母弱" if (r.get("scene_mass") is not None and r["scene_mass"] < WEAK_DENOM_WARN) else ""
        print(f"{r['key']:<16}{r['cat']:<7}{r['frac']:>6.2f}{pr:>7}{cp:>7}{rw:>7}"
              f"{sm_:>7}{mg:>9}{rt:>7}  {'有效' if r['valid'] else '★无效'}{wk}")

    def apply(rule, r):
        kind = rule[0]
        if kind == "frac":
            return r["frac"] is not None and r["frac"] >= rule[1]
        if kind == "frac+margin":
            return (r["frac"] is not None and r["frac"] >= rule[1]
                    and r["margin"] is not None and r["margin"] >= rule[2])
        if kind == "frac+ret":       # 场景过 且 保持度过(保持度缺失=只看场景)
            sc = r["frac"] is not None and r["frac"] >= rule[1]
            if r["ret"] is None:
                return sc
            return sc and r["ret"] >= rule[2]
        if kind == "frac|ret":       # 场景或保持度任一达标即过(救回纹理类)
            sc = r["frac"] is not None and r["frac"] >= rule[1]
            rt = r["ret"] is not None and r["ret"] >= rule[2]
            return sc or rt
        if kind == "pres":           # [v2.7.9] 场景参照保真比单特征
            return r["pres"] is not None and r["pres"] >= rule[1]
        if kind == "pres+frac":      # 保真比 且 场景占比(双紧)
            return (r["pres"] is not None and r["pres"] >= rule[1]
                    and r["frac"] is not None and r["frac"] >= rule[2])
        if kind == "pres|frac":      # 放宽型:任一达标(警示同 frac|ret)
            pv = r["pres"] is not None and r["pres"] >= rule[1]
            sc = r["frac"] is not None and r["frac"] >= rule[2]
            return pv or sc
        if kind == "clspres":        # [v2.7.12] 类别保真单特征
            return r["clspres"] is not None and r["clspres"] >= rule[1]
        if kind == "clspres+frac":
            return (r["clspres"] is not None and r["clspres"] >= rule[1]
                    and r["frac"] is not None and r["frac"] >= rule[2])
        if kind == "clspres|frac":   # 放宽型,同警示
            cv = r["clspres"] is not None and r["clspres"] >= rule[1]
            sc = r["frac"] is not None and r["frac"] >= rule[2]
            return cv or sc
        raise ValueError(rule)

    fr_grid = [i / 18 for i in range(0, 19)]
    mg_grid = [-0.05, -0.02, 0.0, 0.01, 0.02, 0.03, 0.05]
    rt_grid = [i / 9 for i in range(0, 10)]
    pr_grid = [i / 20 for i in range(0, 21)]
    # [A2-C1] 退化族剔除:pres 全 None 时 pres 族规则会退化成纯 frac(或恒 False)
    # 却顶着 pres 之名进推荐 —— 整族剔除,并打一行说明
    _has_pres = any(r["pres"] is not None for r in rows)
    _has_ret = any(r["ret"] is not None for r in rows)
    _has_cls = any(r["clspres"] is not None for r in rows)
    _n_cls = sum(1 for r in rows if r["clspres"] is not None)
    if not _has_cls:
        print("[注意] 全部样本 cls_pres_ratio 为空(旧 scores 或无 source_class)"
              "—— clspres 族规则整族跳过")
    elif _n_cls < len(rows):
        print(f"[注意] clspres 覆盖 {_n_cls}/{len(rows)}(stitched/缺 full_scene "
              f"样本按拒绝计)—— 该族结构上无法'完美',判读时把覆盖数算进去")
    if not _has_pres:
        print("[注意] 全部样本 pres_ratio 为空(H5 缺 full_scene 或旧 scores)——"
              "pres 族规则整族跳过")
    candidates = ([("frac", t) for t in fr_grid]
                  + ([("pres", t) for t in pr_grid] if _has_pres else [])
                  + [("frac+margin", t, m) for t in fr_grid for m in mg_grid]
                  + ([("frac+ret", t, rr) for t in fr_grid for rr in rt_grid]
                     if _has_ret else [])
                  + ([("pres+frac", tp, tf) for tp in pr_grid for tf in fr_grid]
                     if _has_pres else [])
                  + ([("frac|ret", t, rr) for t in fr_grid for rr in rt_grid]
                     if _has_ret else [])
                  + ([("pres|frac", tp, tf) for tp in pr_grid for tf in fr_grid]
                     if _has_pres else [])
                  + ([("clspres", t) for t in pr_grid] if _has_cls else [])
                  + ([("clspres+frac", tc, tf) for tc in pr_grid for tf in fr_grid]
                     if _has_cls else [])
                  + ([("clspres|frac", tc, tf) for tc in pr_grid for tf in fr_grid]
                     if _has_cls else []))
    perfect, best = [], (10**9, None)
    for rule in candidates:
        fp = sum(1 for r in rows if not r["valid"] and apply(rule, r))   # 放走坏样本
        fn = sum(1 for r in rows if r["valid"] and not apply(rule, r))   # 误拒好样本
        err = fp * 10 + fn        # 放走坏样本代价 10 倍(门的天职)
        if fp == 0 and fn == 0:
            perfect.append(rule)
        if err < best[0]:
            best = (err, rule, fp, fn)

    print()
    if perfect:
        # [v2.7.8 R5] 13 正/2 负的小样本上"完美分离"很廉价,防过拟合三重排序:
        #   ①家族简单优先(纯 frac > +margin > 与保持度 > 或保持度);
        #   ②±1/9 扰动稳定(每样本 frac/ret 各挪一帧、margin ±0.01 仍完美);
        #   ③分离带宽大者优先(阈值到两侧最近样本的最小距离)。
        # OR 家族是放宽型(frac 低靠保持度放行)—— 正是可能重新放过 00009 型
        # 的方向,单独警示,默认不推荐。
        def _stable(rule):
            for r in rows:
                # [A2-C5] pres 用独立步长 0.05(=其网格粒度):它是连续比值均值,
                # 不是 k/9 计数,借用 frac 的 1/9 会要求 >0.22 间隔,把 pres 族
                # 系统性打成 [±1帧即翻] 降级 —— 恰好压制了为抵消零样本混淆而
                # 引入的那一维。margin 扰动方向随本轮扰动同号。
                for df, dr, dp in ((1/9, 0, 0.05), (-1/9, 0, -0.05),
                                   (0, 1/9, 0.05), (0, -1/9, -0.05)):
                    rr = dict(r)
                    if rr["frac"] is not None: rr["frac"] = min(1, max(0, rr["frac"] + df))
                    if rr["ret"] is not None: rr["ret"] = min(1, max(0, rr["ret"] + dr))
                    if rr["pres"] is not None: rr["pres"] = min(1, max(0, rr["pres"] + dp))
                    if rr["clspres"] is not None: rr["clspres"] = min(1, max(0, rr["clspres"] + dp))
                    if rr["margin"] is not None: rr["margin"] += (0.01 if dp >= 0 else -0.01)
                    ok = apply(rule, rr)
                    if r["valid"] and not ok and (df < 0 or dr < 0 or dp < 0):
                        return False        # 好样本向下扰动就被拒 → 不稳
                    if (not r["valid"]) and ok and (df > 0 or dr > 0 or dp > 0):
                        return False        # 坏样本向上扰动就放行 → 不稳
            return True
        def _sep(rule):
            # [v2.7.9 修] 分离带宽按规则的【主特征】算(pres 族用 pres 维,
            # 否则 frac 维)—— 旧版对 pres 规则错用 frac 维,sep 恒为负
            feat = ("clspres" if rule[0].startswith("clspres")
                    else "pres" if rule[0].startswith("pres") else "frac")
            t = rule[1]
            good = [r[feat] for r in rows if r["valid"] and r[feat] is not None]
            bad = [r[feat] for r in rows if not r["valid"] and r[feat] is not None]
            return min(min(good) - t, t - max(bad)) if good and bad else 0.0
        order = {"frac": 0, "pres": 0, "clspres": 0,
                 "frac+margin": 1, "pres+frac": 1, "clspres+frac": 1,
                 "frac+ret": 2, "frac|ret": 3, "pres|frac": 3, "clspres|frac": 3}
        # [A2-N6] 同族折叠:完美规则常是整段阈值区间(外加退化克隆),只打 8 条
        # 看不到区间端点,"取中点"无法执行 —— 先打每族的完美主阈值区间
        fam = {}
        for r in perfect:
            key = (r[0],) + tuple(r[2:])
            fam.setdefault(key, []).append(r[1])
        print("完美规则族阈值区间(主阈值 min~max,拍板取区间中点):")
        for key in sorted(fam, key=lambda kk: (order[kk[0]], kk)):
            ts = sorted(fam[key])
            sub = f", 副参={key[1:]}" if len(key) > 1 else ""
            print(f"    {key[0]:<12} 主阈值 [{ts[0]:.2f}, {ts[-1]:.2f}]"
                  f"({len(ts)} 格点{sub})")
        ranked = sorted(perfect,
                        key=lambda r: (order[r[0]], not _stable(r), -_sep(r)))
        print(f"✅ 完美分离规则 {len(perfect)} 个(⚠️ 仅拟合 {len(rows)} 个人检样本,"
              f"不证明泛化);按 简单性→扰动稳定→分离带宽 排序:")
        def _key_dist(rule):
            feat = ("clspres" if rule[0].startswith("clspres")
                    else "pres" if rule[0].startswith("pres") else "frac")
            t = rule[1]
            bad = {r["key"]: t - r[feat] for r in rows
                   if not r["valid"] and r[feat] is not None}
            cand = [(r[feat] - t, r["key"]) for r in rows
                    if r["valid"] and r[feat] is not None]
            return bad, (min(cand) if cand else None)   # [A2-C1] 空序列不炸
        def _weak_pass_report(rule):
            """[v2.7.11] 弱分母放行样本报告 —— 只陈述事实(该规则放行了哪些
            scene_pool_mass<警戒线的样本),不断言"规则依赖弱分母"(那是推断,
            留给人看着明细判)。按人工标签分组;pres 族才有意义。"""
            if not (rule[0].startswith("pres") or rule[0].startswith("clspres")):
                return [], []
            t = rule[1]
            # [v2.7.12 自查修] 明细按规则主特征取值:clspres 族用 clspres/
            # scene_cls_mass,pres 族用 pres/scene_mass —— 旧 lambda 写死 pres,
            # clspres 规则会打错特征,且 pres=None 时格式化直接崩。
            _is_cls = rule[0].startswith("clspres")
            WEAK_DEN = WEAK_CLS_DENOM_WARN if _is_cls else WEAK_DENOM_WARN
            _mkey = "scene_cls_mass" if _is_cls else "scene_mass"
            _fkey = "clspres" if _is_cls else "pres"
            _rkey = "clspres_raw" if _is_cls else "pres_raw"
            wk = [r for r in rows
                  if r.get(_mkey) is not None and r[_mkey] < WEAK_DEN
                  and r.get(_fkey) is not None
                  and apply(rule, r)]
            fmt = lambda r: (f"{r['key']} {_fkey}={r[_fkey]:.2f} "
                             f"raw={r[_rkey] if r.get(_rkey) is not None else '-'} "
                             f"{_mkey}={r[_mkey]:.4f} "
                             f"距阈值{(r[_fkey] - t):+.2f} frac={r['frac']:.2f}")
            _weak_pass_report._mkey = _mkey       # 供组标题打印
            return ([fmt(r) for r in wk if r["valid"]],
                    [fmt(r) for r in wk if not r["valid"]])

        for r in ranked[:8]:
            tag = ("  [稳定✓]" if _stable(r) else "  [±1帧即翻,慎]")
            warn = ("  ⚠️放宽型(OR),须单独论证不放走 00009 型"
                    if r[0] in ("frac|ret", "pres|frac", "clspres|frac") else "")
            bad_d, near_v = _key_dist(r)
            print(f"    {r}  sep={_sep(r):.2f}{tag}{warn}")
            _nv = (f"{near_v[1]} +{near_v[0]:.2f}" if near_v is not None
                   else "(该特征无有效样本取值)")
            _bd = (", ".join(f"{k} -{d:.2f}" for k, d in bad_d.items())
                   or "(该特征无无效样本取值)")
            print(f"        坏样本距阈值: {_bd};最近有效样本: {_nv}")
            wv, wb = _weak_pass_report(r)
            if wv or wb:
                _mk_show = getattr(_weak_pass_report, "_mkey", "scene_mass")
                print(f"        弱分母放行样本({_mk_show}<阈,共 "
                      f"{len(wv) + len(wb)} 个;明细供判读,不自动降级为断言):")
                for line in wv:
                    print(f"          WEAK_DENOM_ALLOWED_VALID   {line}")
                # [A2-C4] perfect 分支里 fp==0 ⇒ 无效样本不可能被放行,wb 恒空
                # (数学同义反复,打印会造成"已检查无坏样本"的假保证)——不打。
        print("\n把候选发回拍板(建议取最简且[稳定✓]、sep 最大者的阈值区间中点);"
              "拍板后写进门配置再重建。")
        return 0
    _, rule, fp, fn = best
    print(f"❌ 不存在完美分离规则。最小错误方案: {rule}"
          f"(放走坏样本 {fp},误拒好样本 {fn})")
    # [A2-C4] 弱分母放行报告真正的用武之地:fp>0 的最小错误方案 —— 被它放行的
    # 坏样本若靠弱分母高保真比,是直接阻塞证据
    if rule[0].startswith("pres") or rule[0].startswith("clspres"):
        # [v2.7.12/C1] clspres 族同样要报 —— 旧 startswith("pres") 在该分支
        # 唯一的用武之地上把 clspres 静默漏掉(实跑:放走 clspres=0.99/
        # scene_cls_mass=0.0007 的坏样本,一行不打)
        _is_cls9 = rule[0].startswith("clspres")
        WEAK_DEN9 = WEAK_CLS_DENOM_WARN if _is_cls9 else WEAK_DENOM_WARN
        _mk9 = "scene_cls_mass" if _is_cls9 else "scene_mass"
        _fk9 = "clspres" if _is_cls9 else "pres"
        _rk9 = "clspres_raw" if _is_cls9 else "pres_raw"
        t9 = rule[1]
        wk9 = [r for r in rows
               if r.get(_mk9) is not None and r[_mk9] < WEAK_DEN9
               and r.get(_fk9) is not None and r[_fk9] >= t9]
        for r in wk9:
            tagv = "WEAK_DENOM_ALLOWED_VALID  " if r["valid"] else                    "WEAK_DENOM_ALLOWED_INVALID"
            extra = "   ← 弱分母上放行坏样本 = 直接阻塞" if not r["valid"] else ""
            rawv = r.get(_rk9)
            print(f"    {tagv} {r['key']} {_fk9}={r[_fk9]:.2f} "
                  f"raw={rawv if rawv is not None else '-'} "
                  f"{_mk9}={r[_mk9]:.4f}{extra}")
    print("   把本表+带 --dump-frames 的 JSON 发回,继续调 prompt 库。")
    return 9


if __name__ == "__main__":
    sys.exit(main())
