#!/usr/bin/env python3
# =============================================================================
# test_taxonomy_prompt.py — v2.5 统一分类学 + prompt 的单元/回归测试(硬门 C)
#
# 覆盖:
#   T1 taxonomy JSON 结构:四类无 flood、别名 flood→water、模板无灾害语义
#   T2 四数据集组互斥(builder 导入期断言真跑一遍)
#   T3 用户拍板规则逐条:beach→water、forest→rural、pond/river/port/harbor→water、
#      swimming_pool→urban、shipping_yard→urban、resort→general(逐数据集验证)
#   T4 _make_prompt:模板正确、方向独立成句、include_direction=False 零方向词、
#      输出零 "flood";方向词可被 parse_direction_word_from_prompt 解析
#   T5 evaluate 的 SCENE_CATEGORIES/_SCENE_ALIASES 与 JSON 一致;
#      parse_scene_from_prompt(新模板) 四类全对;canonical_scene 别名生效
#   T6 direction_target_from_waypoints 与 _make_prompt 的四基向量化逐位一致
#      (0-359 整数角全扫);目标=量化相机角+180
#
# 登录节点/沙箱皆可跑(缺 torch/cv2 自动打桩,只测纯逻辑层):
#   python server_run/test_taxonomy_prompt.py
# =============================================================================
import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lora_remotesense" / "new"))

# 重依赖打桩(登录节点有真环境时不触发)
for m in ("torch", "cv2", "lpips"):
    if m not in sys.modules:
        try:
            __import__(m)
        except Exception:
            sys.modules[m] = types.ModuleType(m)
if "lora_training" not in sys.modules:
    try:
        import lora_training.camera_trajectory  # noqa: F401
    except Exception:
        pkg = types.ModuleType("lora_training")
        ct = types.ModuleType("lora_training.camera_trajectory")
        ct.make_poses_from_waypoints = lambda *a, **k: None
        sys.modules["lora_training"] = pkg
        sys.modules["lora_training.camera_trajectory"] = ct

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "bdb", ROOT / "lora_remotesense" / "build_dataset_from_benchmark.py")
bdb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(bdb)          # T2:导入期互斥断言在此真跑

import evaluate as ev  # noqa: E402  (new/ 路径)

TAX = json.loads((ROOT / "lora_remotesense" / "scene_taxonomy.json")
                 .read_text(encoding="utf-8"))
PASS = []
FAIL = []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


print("== T1 taxonomy 结构 ==")
check("categories", TAX["scene_categories"] == ["general", "water", "urban", "rural"],
      str(TAX["scene_categories"]))
check("no_flood_category", "flood" not in TAX["scene_categories"])
check("alias", TAX["legacy_aliases"].get("flood") == "water")
check("templates_no_disaster",
      all("flood" not in t.lower() for t in TAX["prompt_templates"].values()))
check("water_template", "water-dominated" in TAX["prompt_templates"]["water"])

print("== T2 组互斥(builder 断言已在导入期执行) ==")
check("import_disjoint_assert", True)   # 走到这里=没抛
check("builder_categories", bdb.SCENE_CATEGORIES == TAX["scene_categories"])

print("== T3 拍板规则逐条 ==")
RULES = {  # class → 期望场景(存在于该数据集才检查)
    "beach": "water", "forest": "rural", "pond": "water", "river": "water",
    "port": "water", "harbor": "water", "swimming_pool": "urban",
    "shipping_yard": "urban", "resort": "general",
}
for ds, groups in TAX["groups"].items():
    loc = {c: sc for sc, cs in groups.items() for c in cs}
    for cls, want in RULES.items():
        if cls in loc:
            check(f"{ds.split('/')[-1]}:{cls}->{want}", loc[cls] == want,
                  f"实际 {loc[cls]}")

print("== T4 _make_prompt ==")
wp = np.array([[100, 100], [100, 300]], np.float32)   # 向东(dx>0, dy=0)
for sc in TAX["scene_categories"]:
    pr = bdb._make_prompt(sc, 3, [0.0], waypoints=wp)
    check(f"prompt_{sc}_template", TAX["prompt_templates"][sc] in pr, pr)
    check(f"prompt_{sc}_no_flood", "flood" not in pr.lower(), pr)
pr = bdb._make_prompt("water", 3, [0.0], waypoints=wp)
check("direction_sentence", "toward the east" in pr, pr)
check("direction_parse", ev.parse_direction_word_from_prompt(pr) == "E", pr)
pr_ns = bdb._make_prompt("water", 3, [0.0], waypoints=wp, include_direction=False)
check("stripped_no_dirword", ev.parse_direction_word_from_prompt(pr_ns) is None, pr_ns)
check("stripped_keeps_K", "3-segment" in pr_ns, pr_ns)
check("legacy_alias_prompt",
      TAX["prompt_templates"]["water"] in bdb._make_prompt("flood", 2, [0.0], waypoints=wp))

print("== T5 evaluate 一致性 ==")
check("ev_categories", ev.SCENE_CATEGORIES == TAX["scene_categories"])
check("ev_aliases", ev._SCENE_ALIASES == TAX["legacy_aliases"])
check("ev_canonical", ev.canonical_scene("flood") == "water"
      and ev.canonical_scene("water") == "water")
for sc in TAX["scene_categories"]:
    pr = bdb._make_prompt(sc, 2, [0.0], waypoints=wp)
    check(f"ev_scene_parse_{sc}", ev.parse_scene_from_prompt(pr) == sc,
          f"{pr} → {ev.parse_scene_from_prompt(pr)}")

print("== T6 waypoints 目标量化 与 建库量化 逐位一致(0-359 全扫) ==")
mismatch = []
for deg in range(360):
    r = np.radians(deg)
    # 构造端点净位移角=deg 的 waypoints((cy,cx),y 向下 → dy=-sin)
    wpd = np.array([[200.0, 200.0],
                    [200.0 - 100.0 * np.sin(r), 200.0 + 100.0 * np.cos(r)]],
                   np.float32)
    pr = bdb._make_prompt("general", 2, [], waypoints=wpd)
    word = ev.parse_direction_word_from_prompt(pr)          # E/N/W/S
    tgt_prompt = ev.parse_direction_angle_from_prompt(pr)   # 内容角
    tgt_wp = ev.direction_target_from_waypoints(wpd)
    if tgt_prompt is None or tgt_wp is None or abs(tgt_prompt - tgt_wp) > 1e-6:
        mismatch.append((deg, word, tgt_prompt, tgt_wp))
check("quantization_identity_360", not mismatch, str(mismatch[:5]))

print(f"\n汇总: {len(PASS)} PASS / {len(FAIL)} FAIL")
if FAIL:
    print("结果: FAIL —— " + ", ".join(FAIL))
    sys.exit(4)
print("结果: 全绿 ✓(硬门 C 的 taxonomy/prompt 部分)")
