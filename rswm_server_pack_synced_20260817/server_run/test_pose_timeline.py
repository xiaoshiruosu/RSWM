#!/usr/bin/env python3
# =============================================================================
# test_pose_timeline.py — 训练/推理 pose 时间轴一致性(v2.8 P0 回归)
#
# 2026-07-29 审查发现的 P0:train.py 用 `c2ws_i[:lat_f]` 截断 pose,而
# inference.py 用 interpolate_camera_poses 把【完整】轨迹重采样到 lat_f。
# 实测三种帧数配置下,训练 pose 全部落在第一段(0..8)内、**一个转弯点都没有**,
# 只覆盖轨迹前 27-29%;推理却给 100%。训练/推理条件分布根本不一致 ——
# "几何通道零增益且损保真""模型忽略转弯"都可由此平凡解释,20-33 全部结果
# 因此不能作为论文 null 证据。
#
# 本测试用纯 numpy 钉住索引语义(不需要 torch/scipy/GPU):
#   T1 覆盖率:旧截断只覆盖 [0, lat_f-1];新重采样覆盖 [0, T-1](含终点)
#   T2 转弯可见性:三种配置下旧行为均不含首个转弯点(第 steps 帧),新行为含
#   T3 训练/推理同源:两侧 tgt_indices 公式逐位相同
#   T4 lat_f 公式与 4n+1 对齐:17/25/33 → 5/7/9
#
#   python server_run/test_pose_timeline.py
# =============================================================================
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PASS, FAIL = [], []

STEPS_PER_SEGMENT = 9          # v5 config;首个转弯点落在第 STEPS 帧
# [v2.8.2 N8] 基线换成【去重后的真实输入】。旧值 (18,27,36) 是 v2.8 之前的
# 帧数,aligned_frames(18)=17 让断言照样绿,但那套"36 项全绿"对真实新输入零覆盖
# —— 验证的是一条不再存在的路径。旧路径的覆盖率/转弯用例仍保留(见 LEGACY),
# 因为它们钉的是"当初为什么必须修"的证据。
CONFIGS = ((17, 2), (25, 3), (33, 4))     # v2.8 段界去重后的真实帧数
LEGACY_CONFIGS = ((18, 2), (27, 3), (36, 4))   # v2.8 之前(段界含重复帧)


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


def aligned_frames(T):
    """_collate_fn 的 4n+1 对齐。"""
    return ((T - 1) // 4) * 4 + 1


def lat_frames(train_T, vae_stride_t=4):
    """train.py:537 / inference.py:207 的 lat_f。"""
    return (train_T - 1) // vae_stride_t + 1


def main():
    print("== T0 [v2.8.2 N8] 新帧数下 4n+1 对齐必须是 no-op ==")
    for T, _ in CONFIGS:
        check(f"T{T}:aligned_is_noop", aligned_frames(T) == T,
              f"aligned_frames({T})={aligned_frames(T)} —— 新数据不该再被对齐动到")
    check("new_configs_are_4n1", all((T - 1) % 4 == 0 for T, _ in CONFIGS))
    check("legacy_configs_were_not_4n1",
          all((T - 1) % 4 != 0 for T, _ in LEGACY_CONFIGS),
          "旧帧数本该都不是 4n+1(这正是当初被静默截尾的原因)")

    print("== T1/T2 覆盖率与转弯可见性(旧路径证据,钉住'当初为什么必须修')==")
    for T, nseg in LEGACY_CONFIGS:
        tT = aligned_frames(T)
        lf = lat_frames(tT)
        old_cover = np.arange(lf)                       # [:lat_f] 截断
        new_cover = np.linspace(0, tT - 1, lf)          # 重采样(与推理同式)
        check(f"T{T}:old_truncation_misses_end",
              old_cover[-1] < tT - 1,
              f"旧覆盖到帧 {old_cover[-1]},训练序列末帧 {tT - 1}")
        check(f"T{T}:new_resample_reaches_end",
              abs(new_cover[-1] - (tT - 1)) < 1e-9,
              f"{new_cover[-1]}")
        # 首个转弯点:第 STEPS 帧(第一段 0..STEPS-1)
        turn = STEPS_PER_SEGMENT
        check(f"T{T}:old_excludes_first_turn",
              old_cover[-1] < turn,
              f"旧覆盖 [0,{old_cover[-1]}] 未含转弯点 {turn}(本条 FAIL 说明 bug 已不复现)")
        check(f"T{T}:new_includes_first_turn",
              new_cover[-1] >= turn and nseg >= 2,
              f"新覆盖 [0,{new_cover[-1]:.0f}] 应含转弯点 {turn}")
        frac = (old_cover[-1] + 1) / tT
        print(f"      T={T} nseg={nseg} train_T={tT} lat_f={lf} "
              f"旧覆盖率={frac:.0%} 新覆盖率=100%")

    print("== T3 训练侧已改为与推理同源(源码断言) ==")
    train_src = (ROOT / "lora_remotesense" / "new" / "train.py").read_text()
    infer_src = (ROOT / "lora_remotesense" / "new" / "inference.py").read_text()
    # 训练侧必须调用 interpolate_camera_poses,且不得再出现裸截断
    check("train_calls_interpolate",
          "interpolate_camera_poses(" in train_src)
    check("train_no_bare_pose_truncation",
          not re.search(r"c2ws_i\s*=\s*c2ws_i\[:lat_f\]", train_src),
          "仍存在 c2ws_i[:lat_f] 裸截断")
    # 两侧 tgt_indices 公式一致(np.linspace(0, n-1, lat_f))
    pat = r"tgt_indices\s*=\s*np\.linspace\(\s*0\s*,\s*[A-Za-z_0-9]+\s*-\s*1\s*,\s*lat_f\s*\)"
    check("train_tgt_indices_formula", re.search(pat, train_src) is not None)
    check("infer_tgt_indices_formula", re.search(pat, infer_src) is not None)

    print("== T4 lat_f 公式 ==")
    check("lat_f_17_25_33",
          [lat_frames(t) for t, _ in CONFIGS] == [5, 7, 9],
          str([lat_frames(t) for t, _ in CONFIGS]))
    check("configs_are_17_25_33", [t for t, _ in CONFIGS] == [17, 25, 33])
    check("legacy_aligned_to_same",
          [aligned_frames(t) for t, _ in LEGACY_CONFIGS] == [17, 25, 33],
          "旧帧数被对齐后恰好等于新帧数 —— 这正是旧测试假绿的原因")
    check("probe_K1_lat_f", lat_frames(9) == 3, str(lat_frames(9)))

    print("== T5 [v2.8 ④] pose_mode 三态接线 ==")
    tr = (ROOT / "lora_remotesense" / "new" / "train.py").read_text()
    inf = (ROOT / "lora_remotesense" / "new" / "inference.py").read_text()
    ev = (ROOT / "lora_remotesense" / "new" / "evaluate.py").read_text()
    cm = (ROOT / "server_run" / "common.sh").read_text()
    check("train_has_pose_mode_arg", '"--pose_mode"' in tr)
    check("train_drop_skips_cond",
          'if _pm == "drop":' in tr and "dit_cond_dict = None" in tr)
    check("train_asserts_h5_consistency", "_assert_pose_mode" in tr)
    check("train_ckpt_records_mode", '"pose_mode":' in tr)
    check("infer_drop_disables_pose",
          'RSWM_POSE_MODE' in inf and 'action_path = None' in inf)
    check("eval_meta_records_mode", '"pose_mode": os.environ.get' in ev)
    # [v2.9.2] 改为 train_lora 里先算 _TRAIN_PM 再传(资格要按该值逐次重算)
    check("common_passes_mode", '--pose_mode "$_TRAIN_PM"' in cm
                                and '_TRAIN_PM="$(_infer_pose_mode "$data")"' in cm)
    check("common_cross_checks_ckpt", "pose_mode 冲突" in cm)

    print("== T6 [v2.8 ②⑤] 源头去重与 overlap enforce ==")
    v5 = (ROOT / "wan" / "remotesensebenchmark"
          / "build_multi_direction_benchmark_v5.py").read_text()
    v5b = (ROOT / "lora_remotesense" / "new" / "wan" / "remotesensebenchmark"
           / "build_multi_direction_benchmark_v5.py").read_text()
    check("v5_copies_identical", v5 == v5b)
    check("v5_skips_seam_frame", "_step0 = 0 if seg_idx == 0 else 1" in v5)
    check("v5_4n1_contract_warns", "不满足 4n+1" in v5)
    check("v5_overlap_enforced",
          "enforce_overlap" in v5 and "n_reject_overlap" in v5)
    check("v5_resamples_within_source",
          "RSWM_TRAJ_MAX_TRIES" in v5 and "不消耗源图" in v5)
    check("v5_final_overlap_assertion", "最终断言失败" in v5)
    check("v5_audit_json", "overlap_audit.json" in v5)
    # 帧数契约:steps=9 去重后必为 4n+1
    for K in (2, 3, 4):
        T = 9 + (K - 1) * 8
        check(f"frames_K{K}_is_4n1", (T - 1) % 4 == 0 and T in (17, 25, 33), str(T))

    R = lambda *a: (ROOT.joinpath(*a)).read_text()
    print("== T7 [v2.8 自查修] 34 判门作业的两处断裂 ==")
    v5s = (ROOT / "wan" / "remotesensebenchmark"
           / "build_multi_direction_benchmark_v5.py").read_text()
    bd = (ROOT / "lora_remotesense" / "build_dataset_from_benchmark.py").read_text()
    p34 = R("server_run", "pbs", "34a_learnability_drop.pbs")
    p34b = R("server_run", "pbs", "34b_learnability_native.pbs")
    # ① 34 用环境变量关 enforce,v5 必须读得到(旧代码只认 self 属性 → 34 的
    #    高重叠数据会被 enforce 全拒直至源池耗尽,判门作业跑不起来)
    check("enforce_reads_env", '_envbool("RSWM_ENFORCE_OVERLAP", True)' in v5s)
    # [v2.9] 34 不再豁免 overlap,改为预注册区间选择
    check("probe34_no_longer_exempts", "RSWM_ENFORCE_OVERLAP=0" not in p34)
    check("enforce_off_warns", "enforce 已关闭" in v5s)
    # ② 34 传的 argparse 参数必须都存在,否则建库 unrecognized 当场失败
    for a in ("--min-segments", "--max-segments", "--window-height",
              "--window-width", "--temp-dir"):
        used = a in p34
        exists = f'"{a}"' in bd
        check(f"arg{a}_exists", exists or not used, f"34 用了但 builder 没有")
    # 段数范围必须作用在【活路径】上。教训:上一版把解析写进了 v1 死分支,
    # 而实跑走 v5 分支的硬编码 2/4 —— 字符串断言全绿,--min-segments 1 却
    # 静默无效,34 号照样产出 K=2..4。所以这里检查的是 v5 构造器本身。
    import re as _re
    _v5blk = bd[bd.index('if version == "v5":'):bd.index('elif version == "v1":')]
    check("seg_range_in_v5_branch",
          "min_segments=_min_seg" in _v5blk and "max_segments=_max_seg" in _v5blk,
          "v5 分支仍是硬编码 2/4 —— 改在了死分支上")
    check("no_hardcoded_seg_in_v5",
          not _re.search(r"min_segments=\s*\d", _v5blk))
    check("seg_range_parsed_before_branch",
          bd.index("_min_seg = int(getattr(args") < bd.index('if version == "v5":'))
    check("seg_range_validated", "段数范围非法" in bd)
    check("build_rejects_non_4n1", "不满足 4n+1,训练侧会拒收整个数据集" in bd)
    check("pose_injection_backfilled_after_loop",
          bd.index("_pose_sources_seen.add(pose_source)")
          < bd.index("meta.attrs['pose_injection'] = _inj[0]"),
          "文件级 pose_injection 仍写在样本循环之前 → 恒为空集回退")
    # ③ 单段探针的帧数契约:K=1,steps=9 → T=9,仍是 4n+1
    check("probe_K1_frames_4n1", (9 - 1) % 4 == 0)

    print("== T8 [v2.8.1] 外部审查 6 条(P0×5 + P1)==")
    ev = R("lora_remotesense", "new", "evaluate.py")
    tr = R("lora_remotesense", "new", "train.py")
    rt = R("server_run", "run_trajectory_eval.py")
    mv = R("server_run", "make_variant_h5.py")
    bd = R("lora_remotesense", "build_dataset_from_benchmark.py")
    cr = R("server_run", "collect_results.py")
    p34 = R("server_run", "pbs", "34a_learnability_drop.pbs")
    p34b = R("server_run", "pbs", "34b_learnability_native.pbs")
    orch = R("server_run", "pbs", "qsub_corrected_rerun.sh")
    envf = R("server_run", "pbs", "_pbs_env.sh")

    # P0-2 每个训练作业都必须显式 POSE_MODE(漏设 → native 配 SEM_H5 硬失败)
    # [v2.9] 开关从 _pbs_env.sh 上移到 common.sh 的 profile 段(单一来源)
    check("env_has_sem_arm_switch",
          'export SEM_ARM_MODE="${SEM_ARM_MODE:-$_RSWM_DEFAULT_SEM_ARM}"'
          in R("server_run", "common.sh"))
    for j in ("20_r16_sem_s42", "21_r16_sem_s43", "22_r16_sem_s44",
              "23_r16_geo_s42", "24_r16_geo_s43", "25_r16_geo_s44",
              "26_r128_sem_s42", "27_r128_geo_s42", "28_r16_sem_attncam",
              "29_r16_geo_attncam", "30_r8_sem_s42",
              "32_r16_shufpose_s42", "33_r16_geostrip_s42"):
        t = R("server_run", "pbs", j + ".pbs")
        want = "$SEM_ARM_MODE" if "_sem" in j else "native"
        check(f"posemode_{j}", f"export POSE_MODE={want}" in t)

    # P0-3 rich prompt 保留净方向句;方向真值以 waypoints 为准
    check("rich_keeps_net_direction", "include_direction=True)" in mv)
    check("variant_tags_prompt_schema", 'prompt_schema"] = "rich_persegment"' in mv
                                        and 'prompt_schema"] = "stripped"' in mv)
    check("target_prefers_waypoints",
          '"direction_target_source"] = "waypoints"' in ev
          and ev.index("if _has_wp:") < ev.index('elif _p_ang is not None:'))
    check("prompt_wp_crosscheck", "与 waypoints" in ev and "拒绝继续" in ev)

    # P0-4 pose_mode 决定位姿口径;dropped 路由为 text-only
    check("posecond_honors_mode", 'if _rt_pose_mode == "drop":' in ev
                                  and '"dropped"' in ev)
    check("route_knows_dropped", '"dropped": True' in rt)
    check("meta_records_prompt_schema", '"prompt_schema": sample.get' in ev)
    check("native_x_identity_hard_fail", "把中和过的 SEM_H5 当几何臂在跑" in ev)

    # P1 非 4n+1 / 批内混长 硬错;v5 默认 steps=9
    check("collate_rejects_non4n1", "不满足 4n+1(VAE 时间下采样 4×)" in tr)
    check("collate_rejects_mixed_len", "批内帧数不一致" in tr)
    check("no_silent_align_left",
          "min_frames = min(aligned, raw_min)" not in tr)
    v5 = R("wan", "remotesensebenchmark", "build_multi_direction_benchmark_v5.py")
    check("v5_default_steps_9", "steps_per_segment: int = 9" in v5)

    # 其余阻断项
    check("collect_has_34_35_36",
          "probe34" in cr and "eval_r16_richtext_s42" in cr and "eval_r16_nopose_s42" in cr)
    check("audit_json_rescued", "overlap_audit.json" in R("server_run", "pbs", "01_build_data.pbs"))
    check("file_pose_injection_derived", "_pose_sources_seen" in bd)
    # [v2.9.1] 建库参数移到 01(34a/34b 共享同一份探针库)
    check("probe34_trainset_overfit", "train_as_val.json" in p34
                                      and "PROBE_N:-20" in R("server_run","pbs","01_build_data.pbs"))
    check("probe34_five_gates", all(k in p34 for k in
          ("PROBE34_MIN_VALID", "PROBE34_MIN_PER_DIR", "PROBE34_MIN_MOTION")))
    check("orchestrator_approval_gate", "APPROVED_01_DATA" in orch)

    print("== T9 [v2.8.2] 对抗审查 5 个 P0 + 一批 P1 ==")
    tr2 = R("lora_remotesense", "new", "train.py")
    cm2 = R("server_run", "common.sh")
    ev2 = R("lora_remotesense", "new", "evaluate.py")
    inf2 = R("lora_remotesense", "new", "inference.py")
    v52 = R("wan", "remotesensebenchmark", "build_multi_direction_benchmark_v5.py")
    bd2 = R("lora_remotesense", "build_dataset_from_benchmark.py")
    p342 = p34
    cr2 = R("server_run", "collect_results.py")
    vh = R("lora_remotesense", "new", "verify_remotesense_h5.py")
    sbr = R("server_run", "semgate_build_report.py")

    # C1 drop 下 dit_cond_dict=None 不得被下标(旧写法第一个 batch 就 TypeError)
    check("c1_sub_plucker_none_safe",
          "sub_plucker = None" in tr2 and "if dit_cond_dict is not None:" in tr2)
    check("c1_no_bare_subscript",
          'sub_plucker = {\n                "c2ws_plucker_emb": [dit_cond_dict' not in tr2)
    # C2 eval_base 必须传 pose 口径
    # [v2.9.3] 同一行现在还带 RSWM_RUN_ROLE
    check("c2_evalbase_passes_mode",
          'RSWM_POSE_MODE="$_pm" RSWM_RUN_ROLE="$_role" $PY evaluate_baseline.py' in cm2)
    # 只查【语句】,不查注释里提到的字样(旧断言把说明文字也算成违规)
    check("c2_no_global_export",
          not any(l.strip().startswith("export RSWM_POSE_MODE")
                  for l in cm2.splitlines()))
    # C3 缺省 pose_mode 从 H5 推断(老入口 p0_smoke/p1_main/p2_ranks/02-10 不再必死)
    check("c3_infer_helper", "_infer_pose_mode() {" in cm2)
    check("c3_evalbase_infers", '_pm="$(_infer_pose_mode "$data")"' in cm2)
    check("c3_gate_before_model_load",
          tr2.index("self._assert_pose_mode(getattr(args") < tr2.index("self._init_vae()"),
          "pose 校验仍在 14B 加载之后 → 每次误配白烧十几分钟")
    check("c3_eval_falls_back_to_ckpt", 'POSE_MODE 未设,采用 checkpoint' in cm2)
    # C4 续训跨 pose_mode 硬错
    check("c4_resume_checks_mode", "_assert_resume_pose_mode" in tr2)
    check("c4_resume_called", "self._assert_resume_pose_mode(args.resume_from)" in tr2)
    # 34 目录与 collect_results 对齐
    check("probe34_dir_matches_collect",
          "eval_probe34a_drop" in p342 and "probe34/eval_probe34a_drop" in cr2)
    check("probe34_or_zero_safe", 'ang = None' in p342 and 'for _k in (' in p342)
    check("probe34_four_directions",
          "--angle-quantize 4" in R("server_run","pbs","01_build_data.pbs"))
    # 门不再 fail-open / 不再只信 attr
    check("gate_fail_closed", "校验不了不许开训" in tr2)
    check("gate_checks_real_poses", "首样本位姿 identity" in tr2)
    # 训推各一次插值、同式
    check("single_interp_train", tr2.count("interpolate_camera_poses(") == 1)
    check("single_interp_infer", inf2.count("interpolate_camera_poses(") == 1)
    # v5 稳健性
    check("v5_envbool", "def _envbool(" in v52)
    check("v5_src_budget_decoupled", "src_max_tries" in v52 and "RSWM_SRC_MAX_TRIES" in v52)
    check("v5_overlap_error_distinct",
          "硬约束不允许 fallback" in v52 and "best_semantic is not None" in v52)
    check("v5_audit_on_failure", "assert_final=False" in v52)
    check("v5_unlink_on_reject", "避免被当成正常产物复用" in v52)
    check("v5_4n1_raises", "RSWM_ALLOW_NON_4N1" in v52)
    check("v5_angle_quantize", "angle_quantize" in v52 and "def _envbool" in v52)
    check("builder_angle_quantize", '"--angle-quantize"' in bd2)
    check("v5_accept_rate_source_level", 'accept_rate_denominator' in v52)
    check("v5_records_exhausted_sids", "overlap_exhausted_sids" in v52)
    check("report_deducts_exhausted", "overlap_exhausted_sids" in sbr)
    # 陈旧帧数门
    check("verify_h5_frames_updated", "{17, 25, 33}" in vh and "{18, 27, 36}" not in vh)
    # pose 口径进聚合层
    check("overall_has_pose_mode", 'results["overall"]["pose_mode"]' in ev2)
    check("overall_rejects_mixed", "同一 eval 目录混了多种口径" in ev2)

    print("== T10 [v2.9 拍板] official profile / drop 主线 / 三 baseline / 预注册 overlap ==")
    cm3 = R("server_run", "common.sh")
    cr3 = R("server_run", "collect_results.py")
    p31 = R("server_run", "pbs", "31_eval_base.pbs")
    p36 = R("server_run", "pbs", "36_r16_nopose_s42.pbs")
    p343 = p34
    orch3 = R("server_run", "pbs", "qsub_corrected_rerun.sh")
    v53 = R("wan", "remotesensebenchmark", "build_multi_direction_benchmark_v5.py")
    bd3 = R("lora_remotesense", "build_dataset_from_benchmark.py")
    tr3 = R("lora_remotesense", "new", "train.py")
    ev3 = R("lora_remotesense", "new", "evaluate.py")
    envf3 = R("server_run", "pbs", "_pbs_env.sh")

    # profile 写进编排,不靠用户 export
    check("profile_in_common", "_RSWM_DEFAULT_SEM_ARM=drop" in cm3
                               and "_RSWM_DEFAULT_SEM_ARM=identity" in cm3)
    check("profile_single_source", "SEM_ARM_MODE:-identity" not in envf3,
          "_pbs_env.sh 仍自设默认 → 两处默认必然漂移")
    check("profile_warns_on_override", "不得作为 headline 数字" in cm3)
    check("orchestrator_requires_official", "正式矩阵要求 RSWM_PROFILE=official" in orch3)

    # 正式矩阵:20-22/26/28/30 = drop + GEO_H5;36 = 显式 identity + SEM_H5
    for j in ("20_r16_sem_s42", "21_r16_sem_s43", "22_r16_sem_s44",
              "26_r128_sem_s42", "28_r16_sem_attncam", "30_r8_sem_s42"):
        t = R("server_run", "pbs", j + ".pbs")
        # [v2.9.1 C6] 数据跟着口径走:drop→GEO_H5,identity(legacy)→SEM_H5。
        # 训练/评估调用点必须用 $ARM_H5,不得直接写死任一个。
        check(f"official_{j}_h5_follows_mode",
              'ARM_H5="$SEM_H5"; else ARM_H5="$GEO_H5"' in t
              and '"$ARM_H5"' in t
              and 'train_lora' in t and '"$GEO_H5"' not in t.split("ARM_H5=", 1)[1].split("\n", 1)[1],
              "drop 臂的数据选择没跟着 SEM_ARM_MODE 走")
        check(f"official_{j}_drop", "export POSE_MODE=$SEM_ARM_MODE" in t)
    check("arm36_explicit_identity", "export POSE_MODE=identity" in p36)
    check("arm36_uses_sem_h5", '"$SEM_H5"' in p36 and '"$GEO_H5"' not in p36)
    check("arm36_not_auto_inferred", "不得依赖 _infer_pose_mode 的自动推断" in p36)

    # 三 baseline + 按模式配对 Δ,禁止 text-only 叫法
    check("base31_per_mode_loop", 'POSE_MODE="$_m" eval_base' in p31)
    check("base31_no_single_mode", "export POSE_MODE=$SEM_ARM_MODE" not in p31)
    check("base_three_modes", all(f'eval_base_{m}' in p31
                                  for m in ("drop", "native", "identity")))
    check("collect_three_bases", all(f'eval_base_{m}/eval_results.json' in cr3
                                     for m in ("drop", "native", "identity")))
    check("collect_delta_by_mode", "_delta_by_mode" in cr3 and "ARM_MODE = {" in cr3)
    check("label_headline_drop", "Text/no-pose (drop, headline)" in cr3)
    check("label_headline_native", "Native Plücker (headline)" in cr3)
    check("label_stationary_diag", "Stationary-pose diagnostic" in cr3)
    check("no_text_only_for_identity", "禁止把 identity 臂称作 text-only" in cr3)

    # 预注册 overlap 区间
    check("v5_overlap_is_interval", "min_overlap_ratio" in v53
                                    and "min_ov <= _ov < max_ov" in v53)
    check("v5_asserts_both_bounds", "超出预注册区间" in v53)
    check("builder_overlap_cli", '"--min-overlap-ratio"' in bd3)
    _p01 = R("server_run", "pbs", "01_build_data.pbs")
    check("probe34_preregistered_band", "PROBE34_OV_LO" in _p01
                                        and "--min-overlap-ratio" in _p01)
    check("probe34_gate_asserts_band", "越出预注册区间" in p343
                                       and "建库参数与判门口径分叉" in p343)
    check("probe34_gate_needs_enforce", "区间没有被真正强制" in p343)
    check("probe34_uses_geo", 'probe_geo.h5' in p343 and 'neutralize_poses' not in p343)

    # profile 进 checkpoint / resume / eval meta / 结果表
    check("ckpt_records_profile", '"rswm_profile": os.environ.get' in tr3)
    check("resume_checks_profile", "RSWM_PROFILE 冲突" in tr3)
    check("evalmeta_records_profile", '"rswm_profile": os.environ.get' in ev3)
    check("overall_records_profile", 'results["overall"]["rswm_profile"]' in ev3)

    print("== T11 [v2.9.1] 对抗审查 C1-C9 + nitpicks ==")
    v54 = R("wan", "remotesensebenchmark", "build_multi_direction_benchmark_v5.py")
    cm4 = R("server_run", "common.sh")
    cr4 = R("server_run", "collect_results.py")
    dl4 = R("lora_remotesense", "new", "dataset_loader.py")
    eb4 = R("lora_remotesense", "new", "evaluate_baseline.py")
    tr4 = R("lora_remotesense", "new", "train.py")
    ev4 = R("lora_remotesense", "new", "evaluate.py")
    sbr4 = R("server_run", "semgate_build_report.py")
    p01 = R("server_run", "pbs", "01_build_data.pbs")
    orch4 = R("server_run", "pbs", "qsub_corrected_rerun.sh")

    # C1 四方向位移从 overlap 带反解(不再靠拒绝采样)
    check("c1_band_solves_displacement", "位移【从带反解】" in v54
                                         and "d_lo = (1.0 - hi) * extent" in v54)
    check("c1_round_robin_dirs", "_dir_rr" in v54)
    # C2 判门只认四个基本方向键
    for f, t in (("34a", p34), ("34b", p34b)):
        check(f"c2_{f}_cardinal_only", '_CARD = ("E", "N", "W", "S")' in t
                                       and 'k[4:] in _CARD' in t)
    # C3 耗尽 sid 带场景维度
    check("c3_sid_has_scene", 'f"{scene_category}|{int(sid)}"' in v54)
    check("c3_report_uses_scene", 'f"{sc}|{int(sid)}" in _ov_exhausted' in sbr4)
    # C4 checkpoint 值需与数据相容
    check("c4_ckpt_vs_data", "_data_pm" in cm4 and "以【数据】为准取" in cm4)
    # C5 baseline meta 补齐 + 交叉校验
    check("c5_baseline_meta", '"pose_mode": os.environ.get("RSWM_POSE_MODE"' in eb4
                              and '"prompt_schema": sample.get' in eb4)
    check("c5_collect_crosschecks", "_mode_conflicts" in cr4 and "口径冲突" in cr4)
    # C6 legacy 下数据跟着口径走
    for j in ("20_r16_sem_s42", "26_r128_sem_s42", "30_r8_sem_s42"):
        t = R("server_run", "pbs", j + ".pbs")
        check(f"c6_{j}_h5_follows_mode",
              'ARM_H5="$SEM_H5"; else ARM_H5="$GEO_H5"' in t and '"$ARM_H5"' in t)
    # C7 真正的截断点硬错
    check("c7_loader_raises", "_truncation_allowed()" in dl4
                              and "RSWM_ALLOW_FRAME_TRUNCATION" in dl4)
    # C8 报错按落在哪一侧给建议
    check("c8_side_aware_msg", "_side_lo" in v54 and "偏低" in v54)
    # C9 推断器 fail-closed
    check("c9_infer_fail_closed", "推断器本身没跑起来" in cm4)
    check("c9_py_probe", "既没有 python 也没有 python3" in cm4)
    # nitpicks
    check("n2_no_loop_var_leak",
          "if accepted is None:" in v54 and "达到严格上限" in v54)
    check("n3_resume_gate_early",
          tr4.index("self._assert_resume_pose_mode(args.resume_from)")
          < tr4.index("self._init_vae()"))
    check("n4_intrinsics_assert", "内参不再逐帧恒定" in tr4)
    check("n5_schema_mixed_raises", "prompt_schema={_pss}" in ev4)
    check("n6_missing_key_raises", "个 meta 没有 pose_mode" in ev4)
    check("n7_envbool_empty_is_default", "空串按 default 处理" in v54)
    check("n9_audit_deepcopy", "_copy.deepcopy" in v54)
    check("n10_audit_has_min", '"min": float(ov.min())' in v54)
    # 34a/34b 拆分与共享探针库
    check("gate_split_two_arms",
          (ROOT / "server_run" / "pbs" / "34a_learnability_drop.pbs").is_file()
          and (ROOT / "server_run" / "pbs" / "34b_learnability_native.pbs").is_file())
    check("gate_shared_probe_built_in_01", "p1_probe_build" in p01)
    check("gate_both_must_pass", 'j34="$j34a:$j34b"' in orch4)
    check("gate34b_is_native", "POSE_MODE=native train_lora probe34b_native" in p34b)
    check("gate34a_is_drop", "POSE_MODE=drop train_lora probe34a_drop" in p34)
    # headline 资格 + 指标方向
    # [v2.9.2] profile 只给基值,真资格逐作业按 pose_mode 重算
    check("headline_flag_exported", "RSWM_PROFILE_ELIGIBLE=1" in cm4
                                    and 'export RSWM_HEADLINE_ELIGIBLE=0' in cm4)
    check("collect_auto_rejects", "rejected_rows" in cr4 and "已拒收" in cr4)
    check("metric_direction_marked", "METRIC_DIR = {" in cr4 and "def improved(" in cr4)
    check("zero_delta_is_same", '"same"' in cr4)

    print("== T12 [v2.9.2] run manifest 统一身份 / headline fail-closed / 双判门 ==")
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "lora_remotesense" / "new"))
    import run_manifest as _rm
    tr5 = R("lora_remotesense", "new", "train.py")
    ev5 = R("lora_remotesense", "new", "evaluate.py")
    eb5 = R("lora_remotesense", "new", "evaluate_baseline.py")
    cm5 = R("server_run", "common.sh")
    cr5 = R("server_run", "collect_results.py")
    p015 = R("server_run", "pbs", "01_build_data.pbs")
    or5 = R("server_run", "pbs", "qsub_corrected_rerun.sh")
    v55 = R("wan", "remotesensebenchmark", "build_multi_direction_benchmark_v5.py")
    bd5 = R("lora_remotesense", "build_dataset_from_benchmark.py")
    p34a = R("server_run", "pbs", "34a_learnability_drop.pbs")
    p34b2 = R("server_run", "pbs", "34b_learnability_native.pbs")

    # manifest 规则(行为级,非字符串)
    for pr, sa, pm, role, exp in (
            ("official","drop","drop","headline_text",True),
            ("official","drop","native","headline_native",True),
            ("official","drop","identity","diagnostic_stationary",False),
            ("official","drop","identity","baseline",False),
            ("official","identity","drop","headline_text",False),
            ("legacy","identity","identity","legacy",False),
            ("official","drop","drop","probe_gate",False)):
        got, _ = _rm.headline_eligible(
            pr, sa, pm, role,
            prompt_schema=("net_continuous" if pr == "official" else "net"),
            path_estimator=("registration" if pr == "official" else "flow"))
        check(f"manifest_elig_{pr}_{sa}_{pm}_{role}", got == exp)
    # fail-closed
    _m = _rm.build_manifest(role="headline_text", pose_mode="drop")
    _ok = False
    try:
        _rm.assert_compatible(None, _m, ctx="test")
    except RuntimeError:
        _ok = True
    check("manifest_missing_is_fatal", _ok, "缺 manifest 被放行 = fail-open")
    _m2 = dict(_m); _m2["pose_mode"] = "identity"
    _ok2 = False
    try:
        _rm.assert_compatible(_m, _m2, ctx="test")
    except RuntimeError:
        _ok2 = True
    check("manifest_mismatch_is_fatal", _ok2)

    # P0-1 读对层级 + 缺字段拒
    check("p01_resume_reads_data_split", '_ds.get("rswm_profile")' in tr5)
    check("p01_resume_missing_is_fatal", "没有记录 rswm_profile" in tr5)
    check("p01_resume_uses_manifest", "ctx=\"续训\"" in tr5)
    check("ckpt_writes_manifest", "write_manifest(save_dir" in tr5)
    # P0-2 评测复用身份门
    check("p02_eval_gate", 'ctx="评测断点续推"' in ev5)
    check("p02_base_gate", 'ctx="baseline 断点续推"' in eb5)
    check("p02_gate_before_loop",
          ev5.index('ctx="评测断点续推"') < ev5.index("existing_meta.exists()"))
    # P0-3 fail-closed + 逐作业资格
    check("p03_headline_failclosed", "缺失恰恰是旧产物和被篡改产物的共同特征" in cr5)
    # [v2.9.4] 两档改三档,判据从 elig is not True 变成 classify() 的 tier
    check("p03_requires_true", 'tier == "rejected"' in cr5
                               and 'if o.get("headline_eligible") is True:' in cr5)
    check("p03_per_job_eligibility", "_headline_eligible_for()" in cm5)
    check("p03_identity_never_headline", '"$_pm" != "identity"' in cm5)
    check("p03_recomputed_in_train", 'RSWM_HEADLINE_ELIGIBLE="$(_headline_eligible_for' in cm5)
    # P1 baseline 自证
    check("p1_baseline_selfcheck", "_baseline_mode_conflicts" in cr5)
    # 其余 P1
    check("approval_bound_to_sha", "approval_token" in p015 and "approval_token" in or5)
    check("probe_build_sha_stamp", 'p1_probe_build.sha256' in p015)
    check("h5_persists_band", "meta.attrs['min_overlap_ratio']" in v55
                              and "'angle_quantize'" in v55)
    check("h5_persists_per_sample_ov", "g.attrs['overlap_ratio']" in bd5)
    check("gate_emits_json", "gate_result.json" in p34a and "gate_result.json" in p34b2)
    check("gate_selfchecks_h5", "H5 自记口径" in p34a and "H5 自记口径" in p34b2)
    # 文档
    check("docs_no_stale_gate_ref",
          "34_learnability_probe" not in R("DEPLOY_D4.md"))
    # [v2.9.3] 版本号钉在一处:以 CHANGELOG 顶部为准,其余文档必须与之一致。
    # (写死版本字面量会让每次发版都要改测试,反而催生"改测试让它绿"的习惯。)
    _ver = _re.search(r"v(\d+\.\d+(?:\.\d+)?)", R("CHANGELOG_vs_0710.md")[:200])
    check("docs_have_version", _ver is not None)
    if _ver:
        _v = "v" + _ver.group(1)
        for _d in ("DEPLOY_D4.md", "README_RUN.md", f"README_{_v.replace('.', '')}.md"):
            check(f"docs_version_{_d}", _v in R(_d), f"应含 {_v}")

    print("== T13 [v2.9.3] 对抗审查 2 个 P0 + 3 条未复核项 ==")
    import re as _re2
    cm6 = R("server_run", "common.sh")
    or6 = R("server_run", "pbs", "qsub_corrected_rerun.sh")
    eb6 = R("lora_remotesense", "new", "evaluate_baseline.py")
    ev6 = R("lora_remotesense", "new", "evaluate.py")
    rm6 = R("lora_remotesense", "new", "run_manifest.py")
    p016 = R("server_run", "pbs", "01_build_data.pbs")

    # P0-B 编排器不得用未定义的 $PY
    check("orch_no_bare_PY", "$PY -c" not in or6 and "_PYBIN" in or6)
    # P0-A 两个入口共用同一道 --metrics_only 门
    check("metrics_only_shared_fn", "def assert_metrics_only_ok(" in rm6)
    check("metrics_only_in_evaluate", "assert_metrics_only_ok(args.output_dir" in ev6)
    check("metrics_only_in_baseline", "assert_metrics_only_ok(args.output_dir" in eb6)
    check("metrics_only_checks_code_sha", '"code_sha256": code_sha256()' in rm6)
    # role 不泄漏 + 参与资格
    check("role_not_exported", "export RSWM_RUN_ROLE" not in cm6)
    check("role_per_command", 'RSWM_RUN_ROLE="$_role" $PY' in cm6)
    check("role_in_eligibility", '_role_never_headline' in cm6)
    # shell 与 python 的名单必须一致
    _sh = _re2.search(r'_RSWM_NEVER_HEADLINE_ROLES="([^"]+)"', cm6).group(1).split()
    check("never_headline_lists_match", sorted(_sh) == sorted(_rm._NEVER_HEADLINE_ROLES),
          f"shell={sorted(_sh)} python={sorted(_rm._NEVER_HEADLINE_ROLES)}")
    # 各作业显式声明 role
    for j, want in (("20_r16_sem_s42", "headline_text"),
                    ("23_r16_geo_s42", "headline_native"),
                    ("34a_learnability_drop", "probe_gate"),
                    ("34b_learnability_native", "probe_gate"),
                    ("36_r16_nopose_s42", "diagnostic_stationary"),
                    ("32_r16_shufpose_s42", "variant_attribution"),
                    ("02_train_r128_sem", "legacy")):
        check(f"role_{j}", f"export RSWM_RUN_ROLE={want}" in R("server_run","pbs",j+".pbs"))
    # 判门产物不得被标成 headline 合格(行为级)
    _e, _ = _rm.headline_eligible(
        "official", "drop", "drop", "probe_gate",
        prompt_schema="net_continuous", path_estimator="registration")
    check("probe_never_headline", _e is False)
    # approval_token 必须在抢救之后算
    check("approval_after_rescue",
          p016.index('cp "$WORK/tmp_bench_main/semgate_attempts.json"')
          < p016.index("approval_target.json"))
    check("approval_before_cleanup",
          p016.index("approval_target.json") < p016.index("清理中间目录"))

    print("== T14 [v2.9.4] 审查剩余 6 类:逃生口留痕 / 三档分类 / fail-open / 其它出口 ==")
    rm7 = R("lora_remotesense", "new", "run_manifest.py")
    cr7 = R("server_run", "collect_results.py")
    tr7 = R("lora_remotesense", "new", "train.py")
    g34 = R("server_run", "pbs", "34a_learnability_drop.pbs")
    ag7 = R("server_run", "aggregate_seeds.py")
    fv7 = R("server_run", "recompute_fvd_i3d.py")

    # 逃生口必须留痕且自动降级(行为级)
    import os as _os, tempfile as _tf, json as _js
    _old = dict(_os.environ)
    _os.environ.update(RSWM_PROFILE="official", SEM_ARM_MODE="drop",
                       RSWM_POSE_MODE="drop", RSWM_ALLOW_MANIFEST_MISMATCH="1",
                       RSWM_PROMPT_SCHEMA="net_continuous",
                       RSWM_PATH_ESTIMATOR="registration")
    _prev = _rm.build_manifest(role="legacy", pose_mode="identity")
    _cur = _rm.build_manifest(role="headline_text", pose_mode="drop")
    _rm.assert_compatible(_prev, _cur, ctx="t")
    check("escape_leaves_trace", bool(_cur.get("manifest_override")))
    check("escape_downgrades_headline", _cur.get("headline_eligible") is False)
    _os.environ.clear(); _os.environ.update(_old)

    # 三档分类
    check("collect_three_tiers", 'def classify(' in cr7 and '"diagnostic"' in cr7)
    check("collect_reads_manifest", "read_run_manifest" in cr7
                                    and "run_manifest.json" in cr7)
    check("collect_rejects_override", "manifest_override" in cr7)
    check("collect_diagnostic_section", "🔬 诊断臂" in cr7)
    # 三处 never-headline 名单一致
    import re as _re3
    _sh2 = _re3.search(r'_RSWM_NEVER_HEADLINE_ROLES="([^"]+)"',
                       R("server_run", "common.sh")).group(1).split()
    _py2 = sorted(_rm._NEVER_HEADLINE_ROLES)
    # [v3.1 修正] 用 **import 真实值**,不用正则解析源码。
    # frozenset 里加了中文注释后正则会把注释行当成成员 —— 断言假红。
    # 这一节的教训贯穿本项目:字符串/正则断言在两个方向上都会骗人,
    # 能 import 就 import。
    import importlib.util as _iu3
    _crspec = _iu3.spec_from_file_location(
        "cr_probe", str(ROOT / "server_run" / "collect_results.py"))
    _crmod = _iu3.module_from_spec(_crspec)
    _crspec.loader.exec_module(_crmod)
    _cl2 = sorted(_crmod.DIAGNOSTIC_ROLES)
    # ladder 必须在三处都登记,且在 collect 里是 diagnostic(报告)而非 rejected(故障)
    check("ladder_registered_python", "ladder" in _rm._NEVER_HEADLINE_ROLES)
    check("ladder_registered_shell", "ladder" in _sh2)
    check("ladder_is_diagnostic_not_rejected", "ladder" in _cl2,
          "阶梯若不在 DIAGNOSTIC_ROLES 会被当成来路不明产物整个拒收 —— 语义错了")
    check("never_headline_three_way",
          sorted(_sh2) == _py2 and set(_cl2) <= set(_py2),
          f"shell={sorted(_sh2)} py={_py2} collect={_cl2}")
    # Δ 表的 baseline 与臂都过门,None 不再放行
    check("delta_base_gated", "不得用作 Δ 的分母" in cr7)
    check("delta_none_not_failopen", "没有自报 pose_mode(旧产物或 meta 被删)" in cr7)
    check("delta_arm_none_flagged", "无法交叉核对映射表" in cr7)
    # 判门两处 fail-open
    check("gate_missing_angle_fails", "有样本却拿不到角误差" in g34)
    check("gate_missing_h5_ov_fails", "没有逐样本 overlap_ratio 属性" in g34)
    # manifest 字段用真实属性 + split_seed 进身份
    check("manifest_uses_real_attrs", 'getattr(a, "split_seed"' in tr7
                                      and 'train_val_split.json' in tr7)
    check("identity_has_split_seed", '"split_seed",' in rm7)
    # 其它论文数字出口有门
    check("aggregate_gated", "_assert_headline_eligible(eval_dir" in ag7)
    check("fvd_gated", "_assert_headline_eligible(eval_dir" in fv7)
    check("gate_has_escape", "RSWM_SKIP_HEADLINE_GATE" in ag7 and "RSWM_SKIP_HEADLINE_GATE" in fv7)
    # 所有用 train/eval 的入口都声明了 role
    import pathlib as _pl
    _missing = []
    for _f in sorted((ROOT / "server_run").rglob("*")):
        if _f.suffix not in (".pbs", ".sh") or not _f.is_file():
            continue
        _t = _f.read_text(errors="ignore")
        # 只认【真正的调用行】(行首命令位),排除注释、作业名清单、echo
        if _re3.search(r"(?m)^\s*(?:if\s.*?;\s*then\s+)?(train_lora|eval_lora|eval_base)\s+\S", _t) \
                and "RSWM_RUN_ROLE" not in _t:
            _missing.append(_f.name)
    check("all_entries_declare_role", not _missing, f"缺 role: {_missing}")

    print("== T15 [v2.10] ICLR 补齐:分层 N / 样本量敏感性 / 泛化间隙 ==")
    import subprocess as _sp, tempfile as _tp, json as _js2, math as _mh
    cm8 = R("server_run", "common.sh")

    # 分层 N 接线
    check("tiered_n_defined", "N_EVAL_HEADLINE" in cm8 and "N_EVAL_ABLATION" in cm8)
    check("n_main_raised", 'N_MAIN="${N_MAIN:-4000}"' in cm8)
    _hl = ("20_r16_sem_s42", "21_r16_sem_s43", "22_r16_sem_s44",
           "23_r16_geo_s42", "24_r16_geo_s43", "25_r16_geo_s44")
    for j in _hl:
        check(f"n_headline_{j}", '"$N_EVAL_HEADLINE"' in R("server_run", "pbs", j + ".pbs"))
    for j in ("26_r128_sem_s42", "30_r8_sem_s42", "35_r16_richtext_s42",
              "36_r16_nopose_s42", "32_r16_shufpose_s42"):
        check(f"n_ablation_{j}", '"$N_EVAL_ABLATION"' in R("server_run", "pbs", j + ".pbs"))
    # 没有作业还在用写死的 0
    _hard = [f.name for f in (ROOT / "server_run" / "pbs").glob("[23]*.pbs")
             if _re.search(r'"\$WORK/eval_[A-Za-z0-9_]+"\s+0\b', f.read_text(errors="ignore"))]
    check("no_hardcoded_eval_n", not _hard, f"仍写死 0: {_hard}")

    # 样本量敏感性:行为级 —— MDE 必须等于闭式 1.96·√2·s/√n
    _mss = ROOT / "server_run" / "metric_sample_size_study.py"
    check("sample_size_study_exists", _mss.is_file())
    if _mss.is_file():
        _d = Path(_tp.mkdtemp()) / "e"; _d.mkdir(parents=True)
        _rng = np.random.default_rng(11)
        _v = _rng.normal(30, 12, 400)
        (_d / "scores.json").write_text(_js2.dumps(
            {"per_sample": [{"Angular_Error": float(x)} for x in _v]}))
        _r = _sp.run([sys.executable, str(_mss), "--eval-dirs", f"a={_d}",
                      "--sizes", "100,400", "--repeats", "5",
                      "-o", str(_d / "s.json")], capture_output=True, text=True)
        check("sample_size_study_runs", _r.returncode == 0, _r.stderr[-160:])
        if _r.returncode == 0:
            _j = _js2.loads((_d / "s.json").read_text())
            _m = _j["dirs"]["a"]["per_sample_curves"]["Angular_Error"]["_mde"]
            _sd = float(np.std(_v, ddof=1))
            # [v3.0 更正] 1.96·√2·SE 是**50% 功效的检出阈**,不是 MDE。
            # MDE 要加 z_{1-β}:(z_.975 + z_.80)·s·√(2(1−ρ))/√n。
            # 旧断言钉的是被更正掉的那个口径 —— 保留它会把"低估所需差异"锁死。
            _ZA, _ZB = 1.959963984540054, 0.8416212335729143
            _ok_thr = all(abs(x["detect_threshold_50pct_power"]
                              - _ZA * _mh.sqrt(2) * _sd / _mh.sqrt(x["n"])) < 1e-9
                          for x in _m)
            _ok_mde = all(abs(x["mde_alpha05"]
                              - (_ZA + _ZB) * _mh.sqrt(2) * _sd / _mh.sqrt(x["n"])) < 1e-9
                          for x in _m)
            check("detect_threshold_matches_closed_form", _ok_thr,
                  "50% 功效检出阈与 1.96·√2·s/√n 不符(子采样经验 SE 含 FPC 会偏小)")
            check("mde_is_80pct_power", _ok_mde,
                  "MDE 不等于 (z.975+z.80)·√2·s/√n —— 用 50% 阈当 MDE 会低估所需差异")
            check("mde_gt_threshold",
                  all(x["mde_alpha05"] > x["detect_threshold_50pct_power"] for x in _m))
            check("mde_uses_analytic_se",
                  all(x.get("se_analytic") is not None for x in _m))
            # 逐样本指标不得报 drift(子采样对均值无偏,报了是同义反复)
            check("no_vacuous_drift_for_per_sample",
                  "_note" in _j["dirs"]["a"]["per_sample_curves"]["Angular_Error"])

    # 泛化间隙:行为级 —— _Std 查找不得误把指标值当标准差
    _ggr = ROOT / "server_run" / "generalization_gap_report.py"
    check("gap_report_exists", _ggr.is_file())
    if _ggr.is_file():
        _w = Path(_tp.mkdtemp())
        def _mkeval(work, arm, o):
            dd = work / f"eval_{arm}"; dd.mkdir(parents=True)
            (dd / "eval_results.json").write_text(_js2.dumps({"overall": o}))
            (dd / "run_manifest.json").write_text(_js2.dumps(
                {"rswm_profile": "official", "pose_mode": o.get("pose_mode", "drop"),
                 "role": "generalization", "headline_eligible": True}))
        _b = {"SSIM": 0.60, "SSIM_Std": 0.09, "PSNR": 15.0, "LPIPS": 0.5,
              "FID": 180.0, "FVD": 33.0, "Angular_Error_Mean": 78.0,
              "Angular_Error_Std": 11.0, "num_samples": 400, "pose_mode": "drop"}
        _mkeval(_w / "src", "a", _b)
        _mkeval(_w / "t1", "a", {**_b, "SSIM": 0.55})
        _r2 = _sp.run([sys.executable, str(_ggr), "--source-work", str(_w / "src"),
                       "--tier-works", f"t1={_w / 't1'}", "--arms", "a:drop",
                       "-o", str(_w / "g.json")], capture_output=True, text=True)
        check("gap_report_runs", _r2.returncode == 0, _r2.stderr[-160:])
        if _r2.returncode == 0:
            _g = _js2.loads((_w / "g.json").read_text())["arms"]["a"]["tiers"]["t1"]["gaps"]
            check("gap_std_lookup_correct",
                  abs(_g["SSIM"]["mde"] - 1.96 * _mh.sqrt(2) * 0.09 / 20.0) < 1e-9,
                  "SSIM 的 MDE 不等于用 SSIM_Std 算的值 —— 可能把指标值当成了 std")
            check("gap_setlevel_unknowable",
                  _g["FVD"]["mde"] is None and _g["FVD"]["resolvable"] is None,
                  "集合级指标没有 std,可分辨性应为『无法判定』而不是『可分辨』")
            check("gap_direction_aware", _g["SSIM"]["worse"] is True)

    # 集合级子采样:不许是空开关(传了 --set-level 却只打印说明 = 用户以为跑过了)
    if _mss.is_file():
        import importlib.util as _iu
        _sp2 = _iu.spec_from_file_location("mss_t", str(_mss))
        _mm = _iu.module_from_spec(_sp2); _sp2.loader.exec_module(_mm)
        _calls = []
        def _fake(sub):
            _calls.append(len(sub)); return 100.0 / len(sub)
        _o = _mm.subsample_set_metric(_fake, [f"k{i}" for i in range(40)],
                                      [10, 20, 40], 5, np.random.default_rng(0))
        check("setlevel_really_recomputes", len(_calls) == 15,
              f"期望 3 档×5 次=15 次真算,实际 {len(_calls)}")
        check("setlevel_values_exact",
              all(abs(_o[n]["mean"] - 100.0 / n) < 1e-9 for n in (10, 20, 40)))
        check("setlevel_drift_detects_nonconvergence",
              "仍在漂移" in _mm.drift_report(_o)["verdict"])
        _e2 = Path(_tp.mkdtemp()) / "e"; _e2.mkdir(parents=True)
        (_e2 / "scores.json").write_text(_js2.dumps({"per_sample": [{"SSIM": 0.5}, {"SSIM": 0.6}]}))
        _rc_hard = _sp.run([sys.executable, str(_mss), "--eval-dirs", f"a={_e2}",
                            "--sizes", "2", "--set-level"],
                           capture_output=True, text=True).returncode
        _rc_soft = _sp.run([sys.executable, str(_mss), "--eval-dirs", f"a={_e2}",
                            "--sizes", "2", "--set-level", "--set-level-soft"],
                           capture_output=True, text=True).returncode
        check("setlevel_fails_loud", _rc_hard != 0, "依赖不满足却静默通过")
        check("setlevel_soft_degrades", _rc_soft == 0)

    print(f"\n汇总: {len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        print("结果: FAIL —— " + ", ".join(FAIL))
        return 4
    print("结果: 全绿 ✓(pose 时间轴:训练=推理,覆盖完整轨迹含转弯)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
