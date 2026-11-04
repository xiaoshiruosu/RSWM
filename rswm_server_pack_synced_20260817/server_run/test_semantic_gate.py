#!/usr/bin/env python3
# =============================================================================
# test_semantic_gate.py — 语义保持门单测 + v5 重采循环端到端(v2.7)
#
# 三层覆盖(全部注入桩评分器,不需要 CLIP/GPU;登录节点可跑):
#   A. SemanticGate 本体:抽帧确定性、阈值判定、general 跳过、
#      评分器返回数错硬错、阈值 env 覆盖。
#   B. v5 _process_scene 重采循环【真跑】(合成场景路径):
#      拒绝 → 游标推进换源图;接受样本的 source_scene_id = 实际消耗的池索引
#      (registry 对账正确性的根);连续拒绝 → 有界 best-candidate fallback;
#      overlap/轨迹无合法候选仍硬错;无门 → 行为同旧版。
#   C. check_semantic_consistency 检查器端到端:造 fixture H5,
#      report-only 恒 0;门模式 坏样本 exit 8 / 全好 exit 0;general 跳过。
#
#   python server_run/test_semantic_gate.py
# =============================================================================
import importlib.util
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lora_remotesense.semantic_gate import SemanticGate  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


# ---------------------------------------------------------------- A. 本体
def test_gate_unit():
    print("== A. SemanticGate 本体 ==")
    idx = SemanticGate.sample_frame_indices(36, 9)
    check("sample_idx_bounds", idx[0] == 0 and idx[-1] == 35 and len(idx) == 9, str(idx))
    check("sample_idx_short", SemanticGate.sample_frame_indices(5, 9) == [0, 1, 2, 3, 4])
    check("sample_idx_deterministic",
          SemanticGate.sample_frame_indices(27, 9) == SemanticGate.sample_frame_indices(27, 9))

    frames = np.zeros((18, 8, 8, 3), np.uint8)
    # 桩:前 nθ 帧同意
    def fn_frac(n_agree):
        return lambda picked, cat: [i < n_agree for i in range(len(picked))]

    g = SemanticGate(score_fn=fn_frac(9), thresholds={"water": 0.6, "urban": 0.6, "rural": 0.6})
    v = g.check_frames(frames, "water")
    check("all_agree_pass", v.passed and v.frac == 1.0)
    g = SemanticGate(score_fn=fn_frac(3), thresholds={"water": 0.6, "urban": 0.6, "rural": 0.6})
    v = g.check_frames(frames, "water")
    check("low_frac_fail", (not v.passed) and abs(v.frac - 3 / 9) < 1e-9, str(v.frac))
    v = g.check_frames(frames, "general")
    check("general_skipped", v.skipped and v.passed and v.frac is None)
    # 00009 模式:12 帧 10 帧草地 → frac≈0.17,必须 FAIL
    g = SemanticGate(score_fn=lambda p, c: [False] * (len(p) - 2) + [True] * 2,
                     thresholds={"water": 0.6, "urban": 0.6, "rural": 0.6})
    v = g.check_frames(frames, "water")
    check("p0_00009_pattern_fails", not v.passed, f"frac={v.frac}")

    # [F5] 未知类别必须硬错(拼错/新类未登记),general 仍有意跳过
    g5 = SemanticGate(score_fn=lambda p, c: [True] * len(p))
    try:
        g5.check_frames(frames, "watre")
        check("unknown_category_raises", False, "没抛")
    except ValueError as e:
        check("unknown_category_raises", "未知 scene_category" in str(e))

    bad = SemanticGate(score_fn=lambda p, c: [True])   # 返回数不匹配
    try:
        bad.check_frames(frames, "water")
        check("scorer_count_mismatch_raises", False, "没抛")
    except RuntimeError:
        check("scorer_count_mismatch_raises", True)

    # [2026-07-29 拍板] 默认阈值 = 冻结值(全精度),env 覆盖仅供校准
    from lora_remotesense.semantic_gate import (FROZEN_P0_THRESHOLD,
                                                default_thresholds)
    check("frozen_full_precision",
          repr(FROZEN_P0_THRESHOLD) == "0.2777777777777778",
          repr(FROZEN_P0_THRESHOLD))
    th0 = default_thresholds()
    check("default_is_frozen",
          all(abs(v - FROZEN_P0_THRESHOLD) < 1e-15 for v in th0.values()), str(th0))
    os.environ["RSWM_SEMGATE_MIN_FRAC_WATER"] = "0.9"
    try:
        th = default_thresholds()
        check("env_override",
              th["water"] == 0.9 and abs(th["urban"] - FROZEN_P0_THRESHOLD) < 1e-15,
              str(th))
    finally:
        del os.environ["RSWM_SEMGATE_MIN_FRAC_WATER"]


# ---------------------------------------- A2. transformers 返回类型兼容
def test_coerce_features():
    """[v2.7.1/v2.7.6] 钉死两代 transformers helper 返回契约:
    Tensor / 已投影 pooler(512) / 未投影 pooler(vision 768),不需要 torch。"""
    print("== A2. _coerce_features(transformers API 漂移防护) ==")

    class FakeTensor:
        def __init__(self, shape): self.shape = tuple(shape)
        def norm(self, **k): return self
        def dim(self): return len(self.shape)

    class Out:                     # ModelOutput 风格的假对象
        def __init__(self, **kw): self.__dict__.update(kw)

    class FakeProjection:
        def __init__(self, in_features, out_features):
            self.in_features = in_features
            self.out_features = out_features
            self.calls = []

        def __call__(self, x):
            if x.shape[-1] != self.in_features:
                raise RuntimeError(
                    f"mat1 and mat2 shapes cannot be multiplied "
                    f"({x.shape[0]}x{x.shape[-1]} and "
                    f"{self.in_features}x{self.out_features})")
            self.calls.append(x)
            return FakeTensor((x.shape[0], self.out_features))

    co = SemanticGate._coerce_features
    visual_projection = FakeProjection(768, 512)
    # ① 直接张量(旧版 transformers)
    t = co(FakeTensor((5, 512)), visual_projection, "image")
    check("tensor_passthrough", t.shape == (5, 512) and not visual_projection.calls)
    # ② CLIPOutput 风格 text_embeds(已投影,不得再过投影头)
    text_projection = FakeProjection(512, 512)
    t = co(Out(text_embeds=FakeTensor((5, 512))), text_projection, "text")
    check("clipoutput_embeds", t.shape == (5, 512) and not text_projection.calls)
    # ③ 旧/第三方 BaseModelOutputWithPooling:vision 768 → 必须投影一次
    t = co(Out(pooler_output=FakeTensor((5, 768)), last_hidden_state=FakeTensor((5, 7, 768))),
           visual_projection, "image")
    check("pooling_768_projected_once",
          t.shape == (5, 512) and len(visual_projection.calls) == 1)
    # ④ Transformers v5/Gadi 实炸场景:pooler 已是 512 → 禁止重复投影
    before = len(visual_projection.calls)
    projected_pooler = FakeTensor((9, 512))
    t = co(Out(pooler_output=projected_pooler,
               last_hidden_state=FakeTensor((9, 50, 768))),
           visual_projection, "image")
    check("pooling_512_passthrough",
          t is projected_pooler and len(visual_projection.calls) == before)
    # ④b 普通 dict 包装也遵守同一契约;embeds 字段优先于 raw pooler
    dict_projected = FakeTensor((9, 512))
    t = co({"pooler_output": dict_projected}, visual_projection, "image")
    check("dict_pooling_512_passthrough",
          t is dict_projected and len(visual_projection.calls) == before)
    dict_embeds = FakeTensor((9, 512))
    t = co({"image_embeds": dict_embeds,
            "pooler_output": FakeTensor((9, 768))},
           visual_projection, "image")
    check("dict_embeds_precede_pooler",
          t is dict_embeds and len(visual_projection.calls) == before)
    # ⑤ text projection 为 512→512 时维度有歧义;helper 契约优先,不得重复投影
    t = co(Out(pooler_output=FakeTensor((7, 512))), text_projection, "text")
    check("square_projection_pooler_passthrough",
          t.shape == (7, 512) and not text_projection.calls)
    # ⑥ pooler 既非输入维也非输出维 → 硬错,不猜
    try:
        co(Out(pooler_output=FakeTensor((5, 640))), visual_projection, "image")
        check("pooling_unknown_dim_raises", False, "没抛")
    except RuntimeError as e:
        check("pooling_unknown_dim_raises", "既不等于" in str(e))
    # ⑦ pooler 本身若是 3D → 在调用 projection 前硬错
    try:
        co(Out(pooler_output=FakeTensor((5, 7, 768))), visual_projection, "image")
        check("pooling_3d_rejected", False, "没抛")
    except RuntimeError as e:
        check("pooling_3d_rejected", "2D" in str(e))
    # ⑧ 未知对象 → 硬错不猜
    try:
        co(Out(logits=FakeTensor((5, 10, 3))), text_projection, "text")
        check("unknown_raises", False, "没抛")
    except RuntimeError as e:
        check("unknown_raises", "无法" in str(e) or "coerce" in str(e).lower())
    # ⑨ 直接 Tensor 为 3D(误取 last_hidden_state)→ 硬错
    try:
        co(FakeTensor((5, 7, 512)), text_projection, "text")
        check("3d_rejected", False, "没抛")
    except RuntimeError as e:
        check("3d_rejected", "2D" in str(e))


# ---------------------------------------- A3. v2 库构造与双轨评分(纯 numpy)
def test_bank_and_dual_track():
    """[v2.7.7] 校准失败(BareLand=0.00 比坏样本低)的根治验证:
    ①prompt 库从 taxonomy 派生,冲突按构造消灭;②双轨评分逐帧诊断可测。"""
    print("== A3. taxonomy 派生库 + 双轨评分 ==")
    from lora_remotesense.semantic_gate import build_prompt_bank

    b = build_prompt_bank("blanchon/AID")
    rural = [p for p, c in zip(b["prompts"], b["cat_of"]) if c == "rural"]
    nonrural = [p for p, c in zip(b["prompts"], b["cat_of"]) if c != "rural"]
    check("bareland_is_rural_pos", any("bare land" in p for p in rural))
    check("bareland_not_neg", not any("bare land" in p for p in nonrural))
    check("meadow_is_rural_pos", any("meadow" in p for p in rural))
    check("no_grassland_distractor", not any("grassland" in p for p in nonrural))
    check("aid_bank_no_wetland", not any("wetland" in p for p in b["prompts"]))
    check("union_has_wetland",
          any("wetland" in p for p in build_prompt_bank()["prompts"]))
    try:
        build_prompt_bank("nonexistent/DS")
        check("unknown_dataset_raises", False, "没抛")
    except ValueError:
        check("unknown_dataset_raises", True)

    # 双轨评分:合成相似度矩阵(3 帧 × 全库),纯 numpy 路径
    g = SemanticGate(score_fn=lambda p, c: [True] * len(p))   # 占位,不走 CLIP
    bank = g._bank("blanchon/AID")
    n = len(bank["prompts"])
    cats = bank["cat_of"]; classes = bank["class_of"]
    i_river = next(i for i in range(n) if classes[i] == "river")
    i_meadow = next(i for i in range(n) if classes[i] == "meadow")
    sims = np.full((3, n), 0.1, np.float32)
    sims[0, i_river] = 0.9      # 帧0:河 → water 同意
    sims[1, i_meadow] = 0.9     # 帧1:草地 → water 不同意(rural 赢)
    sims[2, i_river] = 0.6      # 帧2:河弱赢
    per, det, mm, ret, _pres, _praw, _sm, _cp, _cpr, _scm = g._score_from_sims(
        sims, "water", "blanchon/AID", "River")
    check("dual_per_frame", per == [True, False, True], str(per))
    check("detail_fields", all(k in det[0] for k in
          ("pos_max", "neg_max", "margin", "top_pos", "top_neg", "agree")))
    check("detail_top_neg_is_meadow",
          det[1]["top_neg_class"] == "meadow" and det[1]["top_neg_cat"] == "rural",
          str(det[1]))
    check("retention_two_thirds", abs(ret - 2/3) < 1e-9, str(ret))
    check("mean_margin_present", mm is not None)

    # [v2.7.9] 场景参照保真比:同分布→1.0;帧池衰减→按比例;封顶 1
    gg = SemanticGate(score_fn=lambda p, c: [True] * len(p))
    bank = gg._bank("blanchon/AID")
    groups = bank["groups"]
    n = len(bank["prompts"])
    i_river = next(j for j, g in enumerate(groups) if g["cls"] == "river")
    scene = np.full(n, 0.10, np.float32)
    for k in groups[i_river]["idx"]: scene[k] = 0.30      # 场景:河信号强
    f_same = scene.copy()                                  # 帧0 = 场景同分布
    f_weak = scene.copy()
    for k in groups[i_river]["idx"]: f_weak[k] = 0.12      # 帧1:河信号大幅衰减
    # (TAU=100 下 0.30→0.12 的衰减:河组与本底差从 +20σ 缩到 +2σ,
    #  softmax 质量从 ≈1.0 掉到 ≈0.3 —— 期望值按同一公式手推)
    sims = np.stack([f_same, f_weak])
    per, det, mm, ret, pres, pres_raw, sm, _cp0, _cpr0, _scm0 = gg._score_from_sims(
        sims, "water", "blanchon/AID", None, scene_row=scene)
    check("pres_scene_mass_present", sm is not None and sm > 0.5, str(sm))
    check("pres_same_frame_is_1", abs(det[0]["pool_mass"] / sm - 1.0) < 1e-6)
    check("pres_weak_frame_below", det[1]["pool_mass"] < 0.6 * sm,
          f"{det[1]['pool_mass']:.3f} vs scene {sm:.3f}")
    check("pres_ratio_range", 0.5 < pres < 1.0, str(pres))
    check("pres_monotonic", det[1]["pool_mass"] < det[0]["pool_mass"])
    # [v2.7.10] raw 未截断落盘;帧强于场景 → raw>1 但 capped=1
    check("pres_raw_recorded", all("pres_raw" in d for d in det))
    # 场景中等强度(0.13→质量≈0.47,softmax 未饱和),帧强(0.30→≈1.0)→ raw≈2
    scene_mid = np.full(n, 0.10, np.float32)
    for k in groups[i_river]["idx"]: scene_mid[k] = 0.13
    f_strong = np.full(n, 0.10, np.float32)
    for k in groups[i_river]["idx"]: f_strong[k] = 0.30
    _, det2, _, _, p2, p2raw, sm2, _c1, _c2, _c3 = gg._score_from_sims(
        np.stack([f_strong]), "water", "blanchon/AID", None, scene_row=scene_mid)
    check("pres_capped_at_1", abs(p2 - 1.0) < 1e-9, str(p2))
    check("pres_raw_exceeds_1", p2raw > 1.5 and det2[0]["pres_raw"] > 1.5,
          f"raw={p2raw} scene_mass={sm2}")
    v_np = gg.check_frames(np.zeros((3, 4, 4, 3), np.uint8), "water")
    check("pres_none_without_scene", v_np.pres_ratio is None)

    # [v2.7.11/C2] pad 方形:矩形边缘复制、方形原样、内容居中
    sq = SemanticGate._pad_square
    rect = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    out = sq(rect)
    check("pad_square_shape", out.shape == (4, 4, 3), str(out.shape))
    check("pad_square_center", np.array_equal(out[1:3], rect))
    check("pad_square_edge_replicated", np.array_equal(out[0], rect[0]))
    sq_in = np.zeros((5, 5, 3), np.uint8)
    check("pad_square_passthrough", sq(sq_in) is sq_in)

    # [v2.7.11/C1] retention:stitched 哨兵 → None;未知类名 → 硬错
    gg1 = SemanticGate(score_fn=lambda p, c: [True] * len(p))
    _, _, _, ret_s, _, _, _, _cA, _cB, _cC = gg1._score_from_sims(
        np.full((2, len(gg1._bank("blanchon/AID")["prompts"])), 0.1, np.float32),
        "water", "blanchon/AID", "stitched", scene_row=None)
    check("retention_stitched_none", ret_s is None, str(ret_s))
    try:
        gg1._score_from_sims(
            np.full((1, len(gg1._bank("blanchon/AID")["prompts"])), 0.1, np.float32),
            "water", "blanchon/AID", "NotARealClass", scene_row=None)
        check("retention_unknown_raises", False, "没抛")
    except ValueError as e:
        check("retention_unknown_raises", "错配" in str(e))

    # [v2.7.12] 类别保真比:源类在帧中保留 → ≈1;消失 → 塌
    ggc = SemanticGate(score_fn=lambda p, c: [True] * len(p))
    bankc = ggc._bank("blanchon/AID")
    nC = len(bankc["prompts"])
    gC = bankc["groups"]
    i_riv = next(j for j, g in enumerate(gC) if g["cls"] == "river")
    i_mea = next(j for j, g in enumerate(gC) if g["cls"] == "meadow")
    sceneC = np.full(nC, 0.10, np.float32)
    for k in gC[i_riv]["idx"]: sceneC[k] = 0.14      # 场景:河中等强度
    f_keep = sceneC.copy()                            # 帧0:河保留
    f_gone = np.full(nC, 0.10, np.float32)
    for k in gC[i_mea]["idx"]: f_gone[k] = 0.14      # 帧1:河没了,草地占上风
    _, detC, _, _, _, _, _, cp, cpr, scm = ggc._score_from_sims(
        np.stack([f_keep, f_gone]), "water", "blanchon/AID", "River",
        scene_row=sceneC)
    check("clspres_scene_mass", scm is not None and scm > 0.1, str(scm))
    check("clspres_keep_frame_1", abs(detC[0]["cls_pres_raw"] - 1.0) < 1e-6,
          str(detC[0].get("cls_pres_raw")))
    check("clspres_gone_frame_low", detC[1]["cls_mass"] < 0.3 * scm,
          f"{detC[1].get('cls_mass')} vs {scm}")
    check("clspres_mean_between", 0.4 < cp < 0.9, str(cp))
    check("clspres_none_without_class",
          gg.check_frames(np.zeros((2, 4, 4, 3), np.uint8), "general").cls_pres_ratio is None)

    # [v2.7.12/C4] stitched 哨兵 + 给了场景:cls 三值必须是 None 不是 0
    _, _, _, _, _, _, _, cpS, cprS, scmS = ggc._score_from_sims(
        np.stack([f_keep]), "water", "blanchon/AID", "stitched",
        scene_row=sceneC)
    check("clspres_stitched_all_none",
          cpS is None and cprS is None and scmS is None,
          str((cpS, cprS, scmS)))
    # source_class=None + 场景:同样 None(修正旧空断言)
    _, _, _, _, _, _, _, cpN, _, _ = ggc._score_from_sims(
        np.stack([f_keep]), "water", "blanchon/AID", None, scene_row=sceneC)
    check("clspres_none_class_none", cpN is None)

    # [v2.7.12/C5] 同 softmax 空间的可加性:池质量 == Σ 池内单类质量
    # (这是"混淆在分子分母抵消"与 clspres/pres 可比性的全部数学依据;
    #  故障注入实证:改 cls_ids 构成/聚合方式/TAU 任一处,本断言当场红)
    row_t = np.random.RandomState(7).uniform(0.05, 0.3, nC).astype(np.float32)
    water_cls = {g["cls"] for g in gC if g["cat"] == "water" and g["cls"] is not None}
    pm = ggc._pool_mass(row_t, gC, "water")
    cm_sum = sum(ggc._class_mass(row_t, gC, c) for c in water_cls)
    check("pool_mass_additivity", abs(pm - cm_sum) < 1e-9, f"{pm} vs {cm_sum}")

    # 扩展桩契约:(bools, details) 元组
    det_stub = [{"margin": 0.05, "class_retained": True},
                {"margin": -0.02, "class_retained": False}]
    g2 = SemanticGate(score_fn=lambda p, c: ([True, False] + [True] * (len(p) - 2),
                                             det_stub))
    v = g2.check_frames(np.zeros((2, 4, 4, 3), np.uint8), "water",
                        source_class="River", dataset="blanchon/AID")
    check("ext_stub_margin", abs(v.mean_margin - 0.015) < 1e-9, str(v.mean_margin))
    check("ext_stub_retention", v.retention_frac == 0.5, str(v.retention_frac))


# ------------------------------------------------- B. v5 重采循环(真跑)
def test_v5_retry_loop():
    print("== B. v5 _process_scene 重采循环(合成场景真跑) ==")
    spec = importlib.util.spec_from_file_location(
        "v5", ROOT / "wan" / "remotesensebenchmark" / "build_multi_direction_benchmark_v5.py")
    v5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v5)

    tmp = Path(tempfile.mkdtemp(prefix="semv5_"))
    try:
        cfg = v5.MultiDirectionConfig(
            output_dir=str(tmp), instances_per_scene=3,
            scene_categories=["water"], full_scene_size=(256, 256),
            window_size=(64, 64), random_seed=42)
        b = v5.MultiDirectionBenchmarkBuilder(cfg, use_hf=False)

        # 桩门:池索引 0/2/3 被拒,其余通过 —— 断言游标跳号且 source_scene_id 正确
        rejected_sids = {0, 2, 3}
        calls = []
        class StubGate:
            def check_frames(self, frames, cat):
                sid = b._semgate_cursors[cat] - 1     # 刚消耗的索引
                calls.append(sid)
                from lora_remotesense.semantic_gate import GateVerdict
                ok = sid not in rejected_sids
                return GateVerdict(category=cat, passed=ok,
                                   frac=1.0 if ok else 0.1, n_scored=9, threshold=0.6)
        b.semantic_gate = StubGate()
        b.semantic_gate_max_tries = 4

        samples, _ = b._process_scene("water", tmp, 0)
        sids = [s["source_scene_id"] for s in samples]
        # 期望:槽0 尝试0(拒)→1(收);槽1 尝试2(拒)3(拒)4(收);槽2 尝试5(收)
        check("cursor_skips_rejected", sids == [1, 4, 5], str(sids))
        check("gate_called_per_attempt", calls == [0, 1, 2, 3, 4, 5], str(calls))
        check("semgate_recorded",
              samples[0]["semgate"]["tries"] == 2 and samples[1]["semgate"]["tries"] == 3
              and samples[2]["semgate"]["tries"] == 1,
              str([s["semgate"] for s in samples]))

        # [v2.7.11 (a)] 拒后重采:最终样本只落【被接受尝试】的 pres,不残留被拒值
        class TwoPhase:
            def __init__(self): self.n = 0
            def check_frames(self, frames, cat):
                from lora_remotesense.semantic_gate import GateVerdict
                self.n += 1
                if self.n == 1:      # 第一张源图:拒,pres=0.1
                    return GateVerdict(category=cat, passed=False, frac=0.2,
                                       n_scored=9, threshold=0.6, pres_ratio=0.1)
                return GateVerdict(category=cat, passed=True, frac=0.9,
                                   n_scored=9, threshold=0.6, pres_ratio=0.9)
        b2p = v5.MultiDirectionBenchmarkBuilder(cfg, use_hf=False)
        b2p.semantic_gate = TwoPhase()
        b2p.semantic_gate_max_tries = 3
        s2p, _ = b2p._process_scene("water", tmp, 0)
        check("accepted_attempt_only",
              s2p[0]["semgate"]["pres_ratio"] == 0.9
              and s2p[0]["semgate"]["frac"] == 0.9,
              str(s2p[0]["semgate"]))

        # [v2.7.11] 桩 verdict 带 pres 时,sample["semgate"] 应落盘三个新键
        class StubGateP:
            def check_frames(self, frames, cat):
                from lora_remotesense.semantic_gate import GateVerdict
                return GateVerdict(category=cat, passed=True, frac=1.0,
                                   n_scored=9, threshold=0.6,
                                   pres_ratio=0.8, pres_ratio_raw=1.2,
                                   scene_pool_mass=0.4)
        bp = v5.MultiDirectionBenchmarkBuilder(cfg, use_hf=False)
        bp.semantic_gate = StubGateP()
        bp.semantic_gate_max_tries = 2
        sp, _ = bp._process_scene("water", tmp, 0)
        check("semgate_pres_persisted",
              sp[0]["semgate"].get("pres_ratio") == 0.8
              and sp[0]["semgate"].get("scene_pool_mass") == 0.4,
              str(sp[0]["semgate"]))

        # 连续拒绝 → 在严格预算内选择分数最高且几何合法的 candidate。
        class AlwaysReject:
            def __init__(self):
                self.calls = 0

            def check_frames(self, frames, cat):
                from lora_remotesense.semantic_gate import GateVerdict
                scores = (0.1, 0.4, 0.2)
                frac = scores[self.calls % len(scores)]
                self.calls += 1
                return GateVerdict(category=cat, passed=False, frac=frac,
                                   n_scored=9, threshold=0.6)
        b2 = v5.MultiDirectionBenchmarkBuilder(cfg, use_hf=False)
        b2.semantic_gate = AlwaysReject()
        b2.semantic_gate_max_tries = 3
        b2.max_retry_per_sample = 3
        b2.enforce_overlap = False
        fallback_samples, _ = b2._process_scene("water", tmp, 0)
        check("exhaust_accepts_best",
              fallback_samples[0]["source_scene_id"] == 1
              and fallback_samples[0]["semgate"]["frac"] == 0.4,
              str(fallback_samples[0]["semgate"]))
        check("fallback_effective_threshold",
              fallback_samples[0]["semgate"]["threshold"] == 0.4
              and fallback_samples[0]["semgate"]["tries"] == 3,
              str(fallback_samples[0]["semgate"]))
        check("fallback_audited",
              b2._overlap_audit["n_semantic_fallback"] == 3
              and len(b2._overlap_audit["semantic_fallbacks"]) == 3,
              str(b2._overlap_audit.get("semantic_fallbacks")))

        # 即使服务器遗留 retry=100，生产硬上限仍为 20；retry>10 必须打印 debug。
        cap_cfg = v5.MultiDirectionConfig(
            output_dir=str(tmp), instances_per_scene=1,
            scene_categories=["water"], full_scene_size=(256, 256),
            window_size=(64, 64), random_seed=42)
        cap_gate = AlwaysReject()
        bcap = v5.MultiDirectionBenchmarkBuilder(cap_cfg, use_hf=False)
        bcap.semantic_gate = cap_gate
        bcap.semantic_gate_max_tries = 100
        bcap.max_retry_per_sample = 100
        bcap.enforce_overlap = False
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            capped_samples, _ = bcap._process_scene("water", tmp, 0)
        cap_log = capture.getvalue()
        check("hard_cap_20", cap_gate.calls == 20,
              f"calls={cap_gate.calls}")
        check("retry_debug_fields",
              "scene=water" in cap_log and "sample_id=0" in cap_log
              and "candidate_count=" in cap_log and "best_score=" in cap_log,
              cap_log[-400:])
        check("hard_cap_fallback",
              len(capped_samples) == 1
              and bcap._overlap_audit["max_retry_per_sample"] == 20,
              str(bcap._overlap_audit))

        # 无门 → 行为同旧版(source_scene_id == 块内序号)
        b3 = v5.MultiDirectionBenchmarkBuilder(cfg, use_hf=False)
        samples3, _ = b3._process_scene("water", tmp, 0)
        check("no_gate_identity",
              [s["source_scene_id"] for s in samples3] == [0, 1, 2]
              and all(s["semgate"] is None for s in samples3),
              str([s["source_scene_id"] for s in samples3]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_four_scene_100_bounded_completion():
    """Data-free 100-slot integration: every scene completes with bounded retry."""
    print("== B1. 四场景 100 样本有界重试集成(桩门,无 CLIP/AID) ==")
    spec = importlib.util.spec_from_file_location(
        "v5_100", ROOT / "wan" / "remotesensebenchmark" /
        "build_multi_direction_benchmark_v5.py")
    v5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v5)
    from lora_remotesense.semantic_gate import GateVerdict

    class NoopVideoWriter:
        def __init__(self, *args, **kwargs):
            pass

        def write_frames(self, frames):
            return True

    class BoundedStubGate:
        def __init__(self):
            self.calls = {}

        def check_frames(self, frames, cat):
            n = self.calls.get(cat, 0)
            self.calls[cat] = n + 1
            if cat == "general":
                return GateVerdict(category=cat, passed=True, frac=None,
                                   n_scored=0, threshold=None, skipped=True)
            passed = (n % 3) != 0       # deterministic reject then recovery
            return GateVerdict(category=cat, passed=passed,
                               frac=1.0 if passed else 0.2,
                               n_scored=9, threshold=0.6)

    tmp = Path(tempfile.mkdtemp(prefix="sem100_"))
    old_writer = v5.VideoWriter
    try:
        v5.VideoWriter = NoopVideoWriter
        cfg = v5.MultiDirectionConfig(
            output_dir=str(tmp), instances_per_scene=25,
            scene_categories=["general", "water", "urban", "rural"],
            full_scene_size=(256, 256), window_size=(64, 64),
            min_segments=2, max_segments=4, steps_per_segment=9,
            random_seed=42)
        builder = v5.MultiDirectionBenchmarkBuilder(cfg, use_hf=False)
        builder.semantic_gate = BoundedStubGate()
        builder.semantic_gate_max_tries = 6
        global_idx = 0
        counts = {}
        for scene in cfg.scene_categories:
            samples, global_idx = builder._process_scene(scene, tmp, global_idx)
            counts[scene] = len(samples)
        check("four_scene_100_counts",
              counts == {"general": 25, "water": 25,
                         "urban": 25, "rural": 25}, str(counts))
        check("four_scene_100_total", global_idx == 100, str(global_idx))
        check("four_scene_retry_cap_respected",
              builder._overlap_audit["max_retry_per_sample"] == 20,
              str(builder._overlap_audit.get("max_retry_per_sample")))
    finally:
        v5.VideoWriter = old_writer
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------- B2. v5 门调用分发专项(5 用例)
def test_dispatch():
    """[v2.7.11] 签名分发定稿回归:显式新签名/旧签名/**kwargs 代理/内部
    TypeError 原样传播(调用计数断言不二次调用)/签名不可解析硬错。"""
    print("== B2. v5 门调用分发 ==")
    spec = importlib.util.spec_from_file_location(
        "v5d", ROOT / "wan" / "remotesensebenchmark" / "build_multi_direction_benchmark_v5.py")
    v5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v5)
    from lora_remotesense.semantic_gate import GateVerdict

    tmp = Path(tempfile.mkdtemp(prefix="disp_"))
    try:
        cfg = v5.MultiDirectionConfig(
            output_dir=str(tmp), instances_per_scene=1,
            scene_categories=["water"], full_scene_size=(256, 256),
            window_size=(64, 64), random_seed=42)

        def run_with(gate):
            b = v5.MultiDirectionBenchmarkBuilder(cfg, use_hf=False)
            b.semantic_gate = gate
            b.semantic_gate_max_tries = 2
            return b._process_scene("water", tmp, 0)

        ok = GateVerdict(category="water", passed=True, frac=1.0,
                         n_scored=9, threshold=0.6)

        class NewSig:                      # 显式新签名
            def __init__(self): self.got = None
            def check_frames(self, frames, cat, source_class=None,
                             dataset=None, full_scene=None):
                self.got = (dataset, full_scene is not None)
                return ok
        g = NewSig(); run_with(g)
        check("dispatch_new_sig", g.got == ("standalone_demo", True), str(g.got))

        class OldSig:                      # 旧签名:不得收到新字段
            def __init__(self): self.calls = 0
            def check_frames(self, frames, cat):
                self.calls += 1
                return ok
        g = OldSig(); run_with(g)
        check("dispatch_old_sig", g.calls == 1)

        class KwProxy:                     # **kwargs 代理:新字段均收到
            def __init__(self): self.got = None
            def check_frames(self, frames, cat, **kwargs):
                self.got = sorted(kwargs)
                return ok
        g = KwProxy(); run_with(g)
        check("dispatch_var_kwargs", g.got == ["dataset", "full_scene"], str(g.got))

        class RaisesTE:                    # 内部 TypeError:原样传播,不二次调用
            def __init__(self): self.calls = 0
            def check_frames(self, frames, cat, dataset=None, full_scene=None):
                self.calls += 1
                raise TypeError("内部真 bug")
        g = RaisesTE()
        try:
            run_with(g)
            check("dispatch_te_propagates", False, "没抛")
        except TypeError as e:
            check("dispatch_te_propagates",
                  "内部真 bug" in str(e) and g.calls == 1,
                  f"calls={g.calls}")

        class NoSig:                       # 签名不可解析:硬错不猜
            @property
            def check_frames(self):
                class _C:
                    def __call__(self, *a, **k): return ok
                    @property
                    def __signature__(self):
                        raise ValueError("no signature")
                return _C()
        try:
            run_with(NoSig())
            check("dispatch_unparsable_raises", False, "没抛")
        except RuntimeError as e:
            check("dispatch_unparsable_raises", "签名" in str(e), str(e)[:60])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------- C. 检查器端到端
def test_checker():
    print("== C. check_semantic_consistency 端到端(桩评分器) ==")
    spec = importlib.util.spec_from_file_location(
        "csc", ROOT / "server_run" / "check_semantic_consistency.py")
    csc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(csc)

    # 桩:亮帧=不同意(草地代理),暗帧=同意
    class StubGateCls(SemanticGate):
        def __init__(self, **kw):
            kw.pop("device", None)
            super().__init__(score_fn=lambda picked, cat:
                             [float(np.asarray(f).mean()) < 100 for f in picked], **kw)
        def ensure_ready(self):
            pass
    csc.SemanticGate = StubGateCls

    import h5py
    tmp = Path(tempfile.mkdtemp(prefix="semchk_"))
    try:
        h5p = tmp / "t.h5"
        with h5py.File(h5p, "w") as f:
            d = f.create_group("data")
            def put(key, sc, bright):
                g = d.create_group(key)
                val = 200 if bright else 30
                g.create_dataset("video_frames",
                                 data=np.full((12, 8, 8, 3), val, np.uint8))
                g.attrs["source_scene_category"] = sc
                g.attrs["source_class"] = "River"
                g.attrs["source_dataset"] = "blanchon/AID"   # [R3] 缺它会被判数据缺陷
                g.attrs["semgate_frac"] = 1.0
            put("sample_00000", "water", bright=False)   # 好
            put("sample_00001", "water", bright=True)    # 坏(00009 模式)
            put("sample_00002", "general", bright=True)  # 无声明,跳过
            put("sample_00003", "urban", bright=False)   # 好

        def run(argv):
            old = sys.argv
            sys.argv = ["csc"] + argv
            try:
                return csc.main()
            except SystemExit as e:
                if e.code is None:
                    return 0
                return e.code if isinstance(e.code, int) else 1   # sys.exit(str)→1
            finally:
                sys.argv = old

        rc = run(["--gt-h5", str(h5p), "--report-only",
                  "-o", str(tmp / "rep1.json")])
        check("report_only_exit0", rc == 0, str(rc))
        rep = json.loads((tmp / "rep1.json").read_text())
        worst = rep["rows"][0]
        check("worst_first", worst["key"] == "sample_00001" and not worst["ok"], str(worst))
        check("general_skipped_count", rep["n_skipped_general"] == 1)

        rc = run(["--gt-h5", str(h5p), "-o", str(tmp / "rep2.json")])
        check("gate_fail_exit8", rc == 8, str(rc))

        with h5py.File(h5p, "a") as f:      # 修掉坏样本 → PASS
            f["data"]["sample_00001"]["video_frames"][...] = 30
        rc = run(["--gt-h5", str(h5p), "-o", str(tmp / "rep3.json")])
        check("gate_pass_exit0", rc == 0, str(rc))

        # [F5] 拼错类别 → 门模式 exit 8(不再默认成 general 静默跳过)
        h5b = tmp / "bad.h5"
        with h5py.File(h5b, "w") as f:
            d = f.create_group("data")
            g = d.create_group("sample_00000")
            g.create_dataset("video_frames", data=np.full((12, 8, 8, 3), 30, np.uint8))
            g.attrs["source_scene_category"] = "watre"     # 拼错
            g.attrs["source_dataset"] = "blanchon/AID"
        rc = run(["--gt-h5", str(h5b), "-o", str(tmp / "rep4.json")])
        check("typo_category_exit8", rc == 8, str(rc))
        rc = run(["--gt-h5", str(h5b), "--report-only"])
        check("typo_report_only_warns_exit0", rc == 0, str(rc))

        # [v2.7.11 兼容矩阵] 四例:旧 H5 缺 stored 字段合法 / 新 H5 算 drift /
        # enabled=true 无证据硬失败 / enabled=false 缺字段合法
        from lora_remotesense.semantic_gate import GateVerdict as GV
        class PresGate(SemanticGate):
            def __init__(self, **kw):
                kw.pop("device", None); super().__init__(score_fn=lambda p, c: [True]*len(p), **kw)
            def ensure_ready(self): pass
            def check_frames(self, frames, cat, source_class=None,
                             dataset=None, full_scene=None):
                return GV(category=cat, passed=True, frac=1.0, n_scored=9,
                          threshold=0.6, pres_ratio=0.7, pres_ratio_raw=0.9,
                          scene_pool_mass=0.4)
        csc.SemanticGate = PresGate

        def mk_h5(path, enabled, with_stored):
            with h5py.File(path, "w") as f:
                if enabled is not None:
                    f.create_group("metadata").attrs["semgate_enabled"] = enabled
                d2 = f.create_group("data")
                g = d2.create_group("sample_00000")
                g.create_dataset("video_frames", data=np.full((12, 8, 8, 3), 30, np.uint8))
                g.attrs["source_scene_category"] = "water"
                g.attrs["source_class"] = "River"
                g.attrs["source_dataset"] = "blanchon/AID"
                if with_stored:
                    g.attrs["semgate_frac"] = 1.0
                    g.attrs["semgate_threshold"] = 0.6
                    g.attrs["semgate_pres_ratio"] = 0.5
                    g.attrs["semgate_scene_pool_mass"] = 0.3
            return path

        h_old = mk_h5(tmp / "old261.h5", None, False)        # 旧 H5:无 flag 无字段
        rc = run(["--gt-h5", str(h_old), "--report-only", "-o", str(tmp / "o1.json")])
        rep = json.loads((tmp / "o1.json").read_text())["rows"][0]
        check("old_h5_report_only_ok", rc == 0)
        check("old_h5_stored_missing",
              rep["stored_pres_ratio"] is None and rep["pres_drift"] is None
              and rep["recomputed_pres_ratio"] == 0.7, str({k: rep[k] for k in
              ("stored_pres_ratio", "pres_drift", "recomputed_pres_ratio")}))

        h_new = mk_h5(tmp / "new.h5", True, True)            # 新 H5:算 drift
        rc = run(["--gt-h5", str(h_new), "-o", str(tmp / "o2.json")])
        rep = json.loads((tmp / "o2.json").read_text())["rows"][0]
        check("new_h5_drift", rc == 0 and abs(rep["pres_drift"] - 0.2) < 1e-9,
              str(rep.get("pres_drift")))

        h_bad = mk_h5(tmp / "bad_enabled.h5", True, False)   # 声称开门却无证据(A3-N5:独立路径)
        rc = run(["--gt-h5", str(h_bad), "-o", str(tmp / "o3.json")])
        check("enabled_no_evidence_exit8", rc == 8, str(rc))

        h_off = mk_h5(tmp / "off.h5", False, False)          # 关门:缺字段合法
        rc = run(["--gt-h5", str(h_off), "-o", str(tmp / "o4.json")])
        check("disabled_missing_legal", rc == 0, str(rc))
        csc.SemanticGate = StubGateCls    # 还原,后续用例不受影响

        # [v2.7.11/A2-C3] 门模式带 --max → FATAL(部分抽查不得签发 PASS)
        rc = run(["--gt-h5", str(h5p), "--max", "2", "-o", str(tmp / "om.json")])
        check("gate_mode_max_forbidden", rc not in (0, 8) and rc != 0, str(rc))
        rc = run(["--gt-h5", str(h5p), "--max", "2", "--report-only"])
        check("report_only_max_allowed", rc == 0, str(rc))

        # [R3] 缺 source_dataset → 数据缺陷 exit 8(不许静默回退 union 库)
        h5d = tmp / "nods.h5"
        with h5py.File(h5d, "w") as f:
            d = f.create_group("data")
            g = d.create_group("sample_00000")
            g.create_dataset("video_frames", data=np.full((12, 8, 8, 3), 30, np.uint8))
            g.attrs["source_scene_category"] = "water"     # 无 source_dataset
        rc = run(["--gt-h5", str(h5d), "-o", str(tmp / "rep6.json")])
        check("missing_dataset_exit8", rc == 8, str(rc))

        # [F5] 语义类样本数为 0(attr 全丢)→ 空转 exit 8
        h5e = tmp / "empty.h5"
        with h5py.File(h5e, "w") as f:
            d = f.create_group("data")
            g = d.create_group("sample_00000")
            g.create_dataset("video_frames", data=np.full((12, 8, 8, 3), 30, np.uint8))
            # 不写 source_scene_category
        rc = run(["--gt-h5", str(h5e), "-o", str(tmp / "rep5.json")])
        check("missing_attr_exit8", rc == 8, str(rc))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_gate_unit()
    test_coerce_features()
    test_bank_and_dual_track()
    test_v5_retry_loop()
    test_four_scene_100_bounded_completion()
    test_dispatch()
    test_checker()
    print(f"\n汇总: {len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        print("结果: FAIL —— " + ", ".join(FAIL))
        return 4
    print("结果: 全绿 ✓(语义门:本体 + v5 重采 + 第四道门)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
