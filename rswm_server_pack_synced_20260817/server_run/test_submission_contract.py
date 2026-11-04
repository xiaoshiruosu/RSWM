#!/usr/bin/env python3
# =============================================================================
# test_submission_contract.py — 不可作弊契约的回归(全部行为级,无字符串匹配)
#
# 钉住的是外部审查(2026-07-30)动态复现的四种作弊。它们全部通过**静默截断**
# 实现,而且旧代码下每一种都能拿到 PSNR=inf:
#     完整 17 帧 → inf / 只交第 1 帧 → inf / 只交左上角 1/4 → inf
#
# 这个文件的每条用例都**真调**被测函数并要求它抛,不是查源码里有没有某个字符串。
# 上几轮的教训是:字符串断言可以全绿而运行时路径是坏的。
#
#   python server_run/test_submission_contract.py
# =============================================================================
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lora_remotesense" / "new"))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f" — {detail}" if detail and not cond else ""))


def blocks(fn):
    """fn 是否以 SubmissionError 拒绝。返回 (是否拒绝, 第一行原因)。"""
    from submission_contract import SubmissionError
    try:
        fn()
        return False, ""
    except SubmissionError as e:
        return True, str(e).splitlines()[0]


def main():
    import subprocess as _sp, tempfile as _tp, json as _js2
    import submission_contract as sc
    from metrics import (compute_psnr, compute_ssim,
                         compute_video_psnr, compute_video_ssim)

    rng = np.random.default_rng(0)
    gt = (rng.random((17, 64, 64, 3)) * 255).astype(np.uint8)

    print("== T1 严格模式:四种作弊必须全部被拒 ==")
    os.environ.pop("RSWM_LENIENT_METRICS", None)
    sc.reset_taint()
    check("strict_is_default", sc.strict_mode())

    CHEATS = [
        ("合法:完整 17 帧", lambda: compute_video_psnr(gt, gt), False),
        ("时间轴:只交第 1 帧", lambda: compute_video_psnr(gt[:1], gt), True),
        ("时间轴:只交前 8 帧", lambda: compute_video_psnr(gt[:8], gt), True),
        ("时间轴:多交 5 帧", lambda: compute_video_psnr(
            np.concatenate([gt, gt[:5]]), gt), True),
        ("空间:只交左上角 1/4", lambda: compute_video_psnr(gt[:, :32, :32], gt), True),
        ("空间:只交上半", lambda: compute_video_psnr(gt[:, :32], gt), True),
        ("SSIM 同样吃时间轴", lambda: compute_video_ssim(gt[:1], gt), True),
        ("SSIM 同样吃空间", lambda: compute_video_ssim(gt[:, :32, :32], gt), True),
        ("图像级 PSNR:半张", lambda: compute_psnr(gt[0][:32], gt[0]), True),
        ("图像级 SSIM:半张", lambda: compute_ssim(gt[0][:32], gt[0]), True),
    ]
    for name, fn, should_block in CHEATS:
        got, why = blocks(fn)
        check(f"cheat:{name}", got == should_block,
              f"{'应拒未拒' if should_block else '应放行却拒了'} — {why[:60]}")

    print("== T2 样本 ID 集必须完全一致 ==")
    req = [f"s{i:05d}" for i in range(10)]
    check("id_exact_ok", not blocks(lambda: sc.assert_id_set(req, req))[0])
    for name, sub in (("缺 1 个", req[:-1]),
                      ("多 1 个", req + ["s99999"]),
                      ("换 1 个", req[:-1] + ["s99999"]),
                      ("空提交", [])):
        got, why = blocks(lambda s=sub: sc.assert_id_set(s, req))
        check(f"id:{name}", got, f"未被拒 — {why[:60]}")

    print("== T3 零失败容忍(旧版容忍 20%)==")
    check("fail_0_ok", not blocks(lambda: sc.assert_no_failures(0, 100))[0])
    for f in (1, 5, 19, 20, 21):
        got, _ = blocks(lambda k=f: sc.assert_no_failures(k, 100))
        check(f"fail:{f}/100 被拒", got,
              f"{f}/100 未被拒 —— 旧版 ≤20% 放行,那等于允许放弃最难的样本")

    print("== T4 条件帧必须从保真指标里排除 ==")
    ge, gte = sc.strip_condition_frames(gt, gt)
    check("cond_frame_stripped", len(ge) == 16 and len(gte) == 16, f"{len(ge)}")
    check("cond_frame_is_first",
          np.array_equal(ge[0], gt[1]), "排除的应是首帧而不是末帧")
    got, _ = blocks(lambda: sc.strip_condition_frames(gt[:1], gt[:1]))
    check("cond_frame_too_short_fails", got, "只有 1 帧时应拒(排除后无剩余)")
    check("cond_frame_count_is_1", sc.CONDITION_FRAMES == 1)

    print("== T5 分布型指标的样本量下限(旧版 n>=2 就算)==")
    check("dist_min_is_32", sc.MIN_SAMPLES_FOR_DISTRIBUTION == 32)
    for n, want in ((2, False), (10, False), (31, False), (32, True), (400, True)):
        check(f"dist:n={n}", sc.assert_enough_for_distribution(n, metric="FID") is want)

    print("== T6 宽松模式:放行但必须污染,且不可撤销 ==")
    os.environ["RSWM_LENIENT_METRICS"] = "1"
    sc.reset_taint()
    check("lenient_off_strict", not sc.strict_mode())
    check("clean_before", not sc.is_tainted())
    v = compute_video_psnr(gt[:1], gt)          # 宽松下放行
    check("lenient_allows", np.isfinite(v) or np.isinf(v))
    check("lenient_taints", sc.is_tainted(), "用过宽松却没留痕 = 把作弊做成默认")
    check("taint_has_reason", len(sc.taint_reasons()) >= 1)
    summ = sc.contract_summary()
    check("summary_reports_taint",
          summ["tainted"] is True and summ["strict"] is False)
    check("summary_reports_cond_frames", summ["condition_frames_excluded"] == 1)
    # 宽松开关的取值解析:空串按 default(严格),不能靠 `export X=` 静默放松
    for val, want_strict in (("0", True), ("false", True), ("no", True),
                             ("", True), ("1", False), ("yes", False)):
        os.environ["RSWM_LENIENT_METRICS"] = val
        check(f"lenient_parse:{val or '<空>'}", sc.strict_mode() is want_strict)
    os.environ.pop("RSWM_LENIENT_METRICS", None)
    sc.reset_taint()

    print("== T7 evaluate/metrics 真的接上了契约(而不是只 import)==")
    msrc = (ROOT / "lora_remotesense" / "new" / "metrics.py").read_text()
    esrc = (ROOT / "lora_remotesense" / "new" / "evaluate.py").read_text()
    # 每个 min() 站点前面都必须有一次 assert_same_shape —— 用 AST 数,不数字符串
    import ast
    tree = ast.parse(msrc)
    n_assert = sum(1 for n in ast.walk(tree)
                   if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "assert_same_shape")
    # 计数按【函数】而不是按行:每个 min(...) 站点占 h/w 两行,按行数会算重。
    # 需要 guard 的函数 = 任何函数体里出现 min(...shape) 或 min(len(...)) 的。
    need, have = set(), set()
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        body = ast.get_source_segment(msrc, fn) or ""
        if ("min(img1.shape" in body or "min(len(gen_frames)" in body
                or "min(g.shape" in body):
            need.add(fn.name)
        if "assert_same_shape(" in body:
            have.add(fn.name)
    check("every_min_site_guarded", need <= have,
          f"需要 guard 的函数 {sorted(need)},实际有 guard 的 {sorted(have)},"
          f"缺 {sorted(need - have)}")
    check("no_duplicate_guard_in_ssim",
          (ast.get_source_segment(msrc, next(
              n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "compute_ssim")) or "").count("assert_same_shape(") == 1,
          "compute_ssim 里有重复/错标的断言")
    check("eval_requires_full_id_set", "assert_id_set(" in esrc)
    check("eval_zero_failure_tolerance",
          "assert_no_failures(" in esrc and "attempted * 0.2" not in esrc,
          "20% 容忍仍在")
    check("metrics_excludes_cond_frame", "strip_condition_frames(" in msrc)
    check("metrics_gates_distribution", "assert_enough_for_distribution(" in msrc)
    check("contract_lands_in_results", 'results["Contract"]' in msrc)

    print("== T8 评分器与指标计算必须同源(不许孪生入口各写一份)==")
    ssrc = (ROOT / "server_run" / "score_submission.py").read_text()
    check("scorer_imports_contract", "from submission_contract import" in ssrc)
    check("scorer_uses_shared_id_rule", "_contract_check_id_set(" in ssrc)
    check("scorer_fails_without_contract", "不允许各写一套" in ssrc)
    # 等价性:同一违规,check_* 与评分器的规则必须都判"拒"
    import importlib.util as _iu2
    _ss = _iu2.spec_from_file_location("scr", str(ROOT / "server_run" / "score_submission.py"))
    _sm = _iu2.module_from_spec(_ss)
    try:
        _ss.loader.exec_module(_sm)
        _loaded = True
    except SystemExit:
        _loaded = False
    check("scorer_module_loads", _loaded, "评分器导入失败(契约缺失?)")
    if _loaded:
        # 缺样本 / 多样本:两条路径都必须给出非空违规
        _req = ["a", "b", "c"]
        for nm, sub in (("缺", ["a", "b"]), ("多", ["a", "b", "c", "d"])):
            _v = sc.check_id_set(sub, _req)
            check(f"equiv_id_{nm}", len(_v) >= 1,
                  "共享规则未报违规 —— 评分器与指标计算会分叉")
        check("equiv_id_clean", sc.check_id_set(_req, _req) == [])
        # 契约常量必须是同一份(评分器不许自己定义另一套)
        check("equiv_const_cond_frames",
              getattr(_sm, "_CONTRACT_COND_FRAMES", None) == sc.CONDITION_FRAMES)
        check("equiv_const_min_dist",
              getattr(_sm, "_CONTRACT_MIN_DIST", None) == sc.MIN_SAMPLES_FOR_DISTRIBUTION)
        # 行为级钉住官方评分入口：不是只检查它 import 了契约，而是真正验证
        # SSIM/PSNR/LPIPS/FID/FVD 共用的输入已经去掉首个条件帧。
        _sg = np.arange(17 * 3, dtype=np.uint8).reshape(17, 1, 1, 3)
        _st = _sg.copy()
        _fg, _ft = _sm.fidelity_views(_sg, _st)
        check("scorer_future_only_len", len(_fg) == 16 and len(_ft) == 16,
              f"评分器保真帧数应为 16，实得 {len(_fg)}/{len(_ft)}")
        check("scorer_future_only_starts_at_t1",
              np.array_equal(_fg[0], _sg[1]) and np.array_equal(_ft[0], _st[1]),
              "评分器没有从 t=1 开始，可能仍把条件帧计入分数")

    print("== T9 阶梯的解析仪器地板必须按段数分(aq=4 只保证逐段轴向)==")
    _lp = ROOT / "server_run" / "ladder_report.py"
    _lspec = ROOT / "server_run" / "ladder_spec.json"
    if _lp.is_file() and _lspec.is_file():
        import json as _j3
        _lsrc = _lp.read_text()
        check("ladder_floor_is_K_aware",
              'if aq == 4 and _K == 1:' in _lsrc,
              "解析地板对 aq=4 一律给 0° —— K≥2 时低估约 20°")
        check("ladder_floor_reads_spec_fields",
              'rung.get("min_segments")' in _lsrc,
              "取不到段数会一律落到兜底值")
        check("ladder_prefers_measured",
              'waypoint_quantization_deg' in _lsrc,
              "必须优先用该档 H5 waypoints 实测的残差")
        # 逐档核对:解析值与实测(离线量过)必须一致到 3° 内
        _MEAS = {(4, 1): 0.0, (4, 2): 21.27, (4, 4): 19.77,
                 (0, 1): 21.48, (0, 2): 23.17, (0, 4): 23.14}
        _bad = []
        for r in _j3.loads(_lspec.read_text())["rungs"]:
            _aq = int(r["angle_quantize"])
            _km, _kx = int(r["min_segments"]), int(r["max_segments"])
            _k = _km if (_km == _kx and _km > 0) else 0
            _analytic = 0.0 if (_aq == 4 and _k == 1) else 22.5
            _m = _MEAS.get((_aq, _k if _k in (1, 2, 4) else _kx))
            if _m is not None and abs(_analytic - _m) > 3.0:
                _bad.append(f"{r['id']}: 解析 {_analytic}° vs 实测 {_m}°")
        check("ladder_floor_matches_measurement", not _bad, "; ".join(_bad))

    print("== T10 光流量程:默认 step 必须落在安全区 ==")
    from metrics import estimate_motion_direction as _emd
    import inspect as _insp
    _msrc2 = (ROOT / "lora_remotesense" / "new" / "metrics.py").read_text()
    check("default_step_is_1", "step = 1" in _msrc2 and "len(frames) // 8" not in _msrc2,
          "默认仍是 len//8 —— T=17→18px、T=33→36px,实测 24px 处误差已 128°(近乎反向)")
    # 行为级:合成纯平移,step=1 必须准;并证明大 step 会劣化
    def _shift_video(dx, dy, T=17, hw=(128, 222)):
        """纯平移合成视频。pad 必须 ≥ 最大位移,否则切片越界 → 形状不一。"""
        h, w = hw
        pad = int(max(abs(dx), abs(dy)) * (T - 1)) + 8
        big = (np.random.default_rng(5).random((h + 2 * pad, w + 2 * pad)) * 255
               ).astype(np.uint8)
        for _ in range(3):        # 低通,避免高频让光流本身不稳
            big = ((big.astype(np.float32)
                    + np.roll(big, 1, 0) + np.roll(big, -1, 0)
                    + np.roll(big, 1, 1) + np.roll(big, -1, 1)) / 5).astype(np.uint8)
        fr = []
        for t in range(T):
            y0, x0 = int(pad - dy * t), int(pad - dx * t)
            win = big[y0:y0 + h, x0:x0 + w]
            assert win.shape == (h, w), f"pad 不足: t={t} 得到 {win.shape}"
            fr.append(np.repeat(win[:, :, None], 3, axis=2))
        return np.stack(fr)
    _v9 = _shift_video(9.0, 0.0)      # 每帧 9px → step=1 安全;step=2 就 18px
    _e1 = _emd(_v9, flow_size=(128, 222), step=1)
    check("step1_accurate", _e1 is not None and min(abs(_e1 - 0), abs(_e1 - 360)) < 8.0,
          f"step=1 下 9px/帧的东向被估成 {_e1}")
    _e_big = _emd(_shift_video(24.0, 0.0), flow_size=(128, 222), step=1)
    check("large_disp_degrades",
          _e_big is None or min(abs(_e_big - 0), abs(_e_big - 360)) > 20.0,
          f"24px/帧对本应严重劣化,却估成 {_e_big} —— 那说明合成夹具没造出混叠")
    # Track B 的调用点必须显式传 step
    _tmsrc = (ROOT / "lora_remotesense" / "new" / "trajectory_metrics.py").read_text()
    check("trackb_passes_step", "estimate_motion_direction(frames, step=1)" in _tmsrc)

    print("== T11 MDE 必须是 80% 功效口径,不是 50% 检出阈 ==")
    _mss2 = ROOT / "server_run" / "metric_sample_size_study.py"
    if _mss2.is_file():
        _s2 = _mss2.read_text()
        check("mde_has_power_term", "_Z_B" in _s2 and "mde_alpha05_power80" in _s2,
              "MDE 仍是 1.96·√2·SE —— 那是 50% 功效的检出阈,会低估所需差异")
        check("mde_keeps_old_key_renamed", "detect_threshold_50pct_power" in _s2)
        # 数值:MDE/检出阈 之比必须 ≈ (z_a+z_b)/z_a ≈ 1.43
        _d3 = Path(_tp.mkdtemp()) / "e"; _d3.mkdir(parents=True)
        _vv = np.random.default_rng(3).normal(77.5, 54.0, 400)
        (_d3 / "scores.json").write_text(_js2.dumps(
            {"per_sample": [{"Angular_Error": float(x)} for x in _vv]}))
        _r3 = _sp.run([sys.executable, str(_mss2), "--eval-dirs", f"a={_d3}",
                       "--sizes", "100", "--repeats", "5", "-o", str(_d3 / "s.json")],
                      capture_output=True, text=True)
        if _r3.returncode == 0:
            _mm = _js2.loads((_d3 / "s.json").read_text())["dirs"]["a"][
                "per_sample_curves"]["Angular_Error"]["_mde"][0]
            _ratio = _mm["mde_alpha05"] / _mm["detect_threshold_50pct_power"]
            check("mde_ratio_correct", abs(_ratio - 1.4295) < 0.01,
                  f"MDE/检出阈 = {_ratio:.4f},应为 (z.975+z.80)/z.975 ≈ 1.4295")
            check("mde_has_three_rho",
                  set(_mm["mde_alpha05_power80"]) == {"rho_0", "rho_0.3", "rho_0.6"})

    print("== T12 条件帧剔除必须覆盖【全部】保真指标(不只 FID/FVD)==")
    import metrics as _M
    _gt2 = (rng.random((17, 64, 64, 3)) * 255).astype(np.uint8)
    _gen2 = (rng.random((17, 64, 64, 3)) * 255).astype(np.uint8)
    _gen2[0] = _gt2[0]          # i2v 的结构性必然:首帧就是输入,逐位相同
    # 未剔除时:逐帧 PSNR 取平均,MSE=0 那帧给 inf → 整段 inf
    check("raw_video_psnr_is_inf_by_construction",
          not np.isfinite(_M.compute_video_psnr(_gen2, _gt2)),
          "首帧相同却没给出 inf —— 那说明夹具没复现该结构")
    _r = _M.compute_all_metrics([_gen2], [_gt2], device="cpu")
    check("aggregate_psnr_is_finite", np.isfinite(_r["PSNR"]),
          f"PSNR={_r['PSNR']} —— 条件帧没被剔除,任何提交的 PSNR 都会是 inf")
    check("cond_frames_reported", _r.get("Condition_Frames_Excluded") == 1)
    check("fidelity_frames_reported", _r.get("Fidelity_Frames_Per_Sample") == 16,
          f"保真用帧数={_r.get('Fidelity_Frames_Per_Sample')},应为 17-1=16")
    # 只能剔一帧:若被剔两次(FID 段再剔一次)会变 15
    check("stripped_exactly_once", _r.get("Fidelity_Frames_Per_Sample") == 16)
    # 合法满分仍应是 inf
    _r2 = _M.compute_all_metrics([_gt2], [_gt2], device="cpu")
    check("perfect_submission_still_inf", not np.isfinite(_r2["PSNR"]),
          "gen==gt 的合法满分被误改成有限值")
    # 剔除发生在逐帧循环之前(源码顺序)
    _msrc3 = (ROOT / "lora_remotesense" / "new" / "metrics.py").read_text()
    check("strip_before_fidelity_loop",
          _msrc3.index("_pairs_all = [strip_condition_frames")
          < _msrc3.index("ssim_scores.append(compute_video_ssim"),
          "剔除仍在保真循环之后 —— SSIM/PSNR/LPIPS 不受保护")

    print("== T13 两份 wan 必须不漂移(canonical 训练用 new/wan,但根目录那份是陷阱)==")
    import filecmp as _fc
    _a, _b = ROOT / "wan", ROOT / "lora_remotesense" / "new" / "wan"
    _diff, _only_a, _only_b = [], [], []
    def _walk(da, db, rel=""):
        cmp = _fc.dircmp(da, db)
        _diff.extend(f"{rel}{x}" for x in cmp.diff_files)
        _only_a.extend(f"{rel}{x}" for x in cmp.left_only)
        _only_b.extend(f"{rel}{x}" for x in cmp.right_only)
        for d in cmp.common_dirs:
            _walk(da / d, db / d, f"{rel}{d}/")
    if _a.is_dir() and _b.is_dir():
        _walk(_a, _b)
        check("wan_copies_no_content_drift", not _diff,
              f"内容漂移: {_diff[:5]} —— get_loraconfig 曾在两份间不同,"
              f"决定 LoRA 打在哪些模块上,直接改变可训参数表")
        check("wan_no_backup_remnants",
              not [x for x in _only_a + _only_b
                   if ".bak" in x or ".backup" in x],
              f"残留: {[x for x in _only_a + _only_b if '.bak' in x or '.backup' in x]}")
        # get_loraconfig 的签名必须一致(旧版没有 targets 参数,train.py 会 TypeError)
        import ast as _ast2
        _sigs = []
        for _r in (_a, _b):
            _f = _r / "modules" / "animate" / "animate_utils.py"
            if not _f.is_file():
                continue
            for _n in _ast2.walk(_ast2.parse(_f.read_text())):
                if isinstance(_n, _ast2.FunctionDef) and _n.name == "get_loraconfig":
                    _sigs.append([a.arg for a in _n.args.args])
        check("get_loraconfig_signature_same", len(_sigs) == 2 and _sigs[0] == _sigs[1],
              f"签名不同 {_sigs} —— train.py 传 targets=,旧签名会 TypeError")
        check("get_loraconfig_has_targets",
              bool(_sigs) and "targets" in _sigs[0])

    print("== T14 无 torch 环境(CI)下契约必须可用 ==")
    _msrc4 = (ROOT / "lora_remotesense" / "new" / "metrics.py").read_text()
    _esrc4 = (ROOT / "lora_remotesense" / "new" / "evaluate.py").read_text()
    # torch 必须是惰性的:模块级 `import torch` 会让整个 CI 覆盖不到本文件
    check("metrics_torch_lazy",
          "\nimport torch\n" not in _msrc4 and "def _torch():" in _msrc4,
          "metrics.py 仍在模块级 import torch —— CI 无 torch,本文件跑不起来")
    check("evaluate_torch_lazy",
          "\nimport torch\n" not in _esrc4 and "def _torch():" in _esrc4)
    check("metrics_all_torch_via_lazy", _msrc4.count("_torch().") >= 12,
          f"_torch(). 只出现 {_msrc4.count('_torch().')} 次,应 ≥12")
    # 缺 torch 时深度指标必须给出**能照做的**错误,不是裸 ImportError
    import importlib.util as _iu4
    _mspec = _iu4.spec_from_file_location(
        "m_probe", str(ROOT / "lora_remotesense" / "new" / "metrics.py"))
    _mm4 = _iu4.module_from_spec(_mspec); _mspec.loader.exec_module(_mm4)
    check("lazy_torch_helper_exists", callable(getattr(_mm4, "_torch", None)))
    # CI 依赖清单必须含 scikit-image / scipy(SSIM 与统计量要用)
    _ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    check("ci_installs_skimage", "scikit-image" in _ci,
          "CI 不装 scikit-image → SSIM 相关用例全部跑不了")
    check("ci_installs_scipy", "scipy" in _ci)
    check("ci_globs_tests", "test_*.py" in _ci,
          "CI 写死测试清单 → 新增的测试文件会被静默漏掉")

    print(f"\n汇总: {len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        print("结果: FAIL —— " + ", ".join(FAIL))
        return 4
    print("结果: 全绿 ✓(四种作弊全部被拒;宽松模式必留痕)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
