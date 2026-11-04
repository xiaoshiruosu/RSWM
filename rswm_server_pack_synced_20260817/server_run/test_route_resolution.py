#!/usr/bin/env python3
# =============================================================================
# test_route_resolution.py — Track B 路由(text_only/commanded)判定单测
#
# 为什么必须钉住:_resolve_text_only 决定 commanded 标志 —— 错了【整表语义
# 就反了】(text-only 下逐段/转弯/waypoint 指标只作诊断,geo 下才是被命令的
# 合规分)。v2.6 把它从"目录名字符串启发式"改成
#   显式 :text/:geo 后缀  >  meta 的 pose_conditioning  >  目录名(且必须唯一命中)
# 并对四类歧义硬失败。这份测试把这些行为逐条固定,防止以后回归。
#
# 覆盖:
#   T1 meta 判定:identity_neutralized→text_only、native→geo
#   T2 显式后缀优先于 meta;后缀与 meta 矛盾 → FATAL
#   T3 混版本目录(部分 meta 有 pose_conditioning、部分没有)→ FATAL
#   T4 同目录两种 pose_conditioning 值 → FATAL
#   T5 未知 pose_conditioning → FATAL
#   T6 无 meta 时目录名启发式:唯一命中 → 通过并 WARN
#   T7 无 meta 且启发式歧义(sem_geo / base / shufpose)→ FATAL(不静默默认)
#
#   python server_run/test_route_resolution.py
# =============================================================================
import contextlib
import io
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _m in ("numpy", "h5py", "cv2", "torch"):
    if _m not in sys.modules:
        try:
            __import__(_m)
        except Exception:
            sys.modules[_m] = types.ModuleType(_m)

# 只提取被测函数与其依赖常量,避免 import 整个驱动(顶层要 numpy/h5py/cv2)
_src = (ROOT / "server_run" / "run_trajectory_eval.py").read_text()
_start = _src.index("_ROUTE_TEXT_PATTERNS")
_end = _src.index("def _dedup_timeline")
_ns = {"json": json, "sys": sys, "Path": Path}
exec(_src[_start:_end], _ns)          # noqa: S102
resolve = _ns["_resolve_text_only"]

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


def mkdir_with(root, dname, metas):
    d = root / dname
    (d / "meta").mkdir(parents=True)
    for i, mm in enumerate(metas):
        (d / "meta" / f"{i:05d}.json").write_text(json.dumps(mm))
    return d


def expect_ok(tag, d, name, suffix, want_text_only, want_source_prefix):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            to, src = resolve(name, d, suffix)
    except SystemExit as e:
        check(tag, False, f"意外 FATAL(exit {e.code}): {err.getvalue()[:90]}")
        return
    check(tag, to is want_text_only and src.startswith(want_source_prefix),
          f"got ({to}, {src})")


def expect_fatal(tag, d, name, suffix, keyword=""):
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            resolve(name, d, suffix)
        check(tag, False, "没有 FATAL")
    except SystemExit as e:
        msg = err.getvalue()
        check(tag, e.code == 2 and (keyword in msg if keyword else True),
              f"exit={e.code} msg={msg[:100]}")


def main():
    root = Path(tempfile.mkdtemp(prefix="route_"))
    try:
        print("== T1 meta 判定 ==")
        d = mkdir_with(root, "eval_r16_sem_s42", [{"pose_conditioning": "identity_neutralized"}])
        expect_ok("meta_identity_text_only", d, "textonly", None, True, "meta:")
        d = mkdir_with(root, "eval_r16_geo_s42", [{"pose_conditioning": "native"}])
        expect_ok("meta_native_geo", d, "textpluecker", None, False, "meta:")
        # meta 压过误导性目录名/别名
        d = mkdir_with(root, "eval_semantic_like", [{"pose_conditioning": "native"}])
        expect_ok("meta_beats_name", d, "sem_looking_name", None, False, "meta:")

        print("== T2 显式后缀 ==")
        d = mkdir_with(root, "eval_no_meta_1", [])
        expect_ok("suffix_text", d, "anything", "text", True, "explicit:")
        expect_ok("suffix_geo", d, "anything", "geo", False, "explicit:")
        d = mkdir_with(root, "eval_conflict", [{"pose_conditioning": "native"}])
        expect_fatal("suffix_vs_meta_conflict", d, "cfg", "text", "矛盾")

        print("== T3/T4/T5 meta 异常 ==")
        d = mkdir_with(root, "eval_mixed_version",
                       [{"pose_conditioning": "native"}, {}, {}])
        expect_fatal("mixed_version_dir", d, "cfg", None, "混版本")
        d = mkdir_with(root, "eval_two_values",
                       [{"pose_conditioning": "native"},
                        {"pose_conditioning": "identity_neutralized"}])
        expect_fatal("two_pose_values", d, "cfg", None, "不一致")
        d = mkdir_with(root, "eval_unknown_pc", [{"pose_conditioning": "zeroed_poses"}])
        expect_fatal("unknown_pose_conditioning", d, "cfg", None, "未知")

        print("== T6 目录名启发式(唯一命中 → 通过 + WARN) ==")
        d = mkdir_with(root, "eval_legacy_sem", [])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            to, src = resolve("r16_sem_s42", d, None)
        check("heuristic_text", to is True and src == "name_heuristic", f"({to},{src})")
        check("heuristic_warns", "WARN" in err.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            to, src = resolve("r16_geo_s42", d, None)
        check("heuristic_geo", to is False and src == "name_heuristic", f"({to},{src})")

        print("== T7 启发式歧义 → FATAL(不静默默认) ==")
        for nm, tag in (("r16_sem_geo", "both_patterns"),
                        ("base", "no_pattern"),
                        ("r16_shufpose_s42", "shufpose_ambiguous")):
            expect_fatal(f"ambiguous:{tag}", d, nm, None, "无法唯一判定")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n汇总: {len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        print("结果: FAIL —— " + ", ".join(FAIL))
        return 4
    print("结果: 全绿 ✓(路由判定;它决定整表 commanded 语义)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
