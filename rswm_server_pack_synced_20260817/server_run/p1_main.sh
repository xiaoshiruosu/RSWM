#!/bin/bash
# =============================================================================
# P1 — 主表(~32-40h):build 500 → 训 r128(语义) → 评 baseline(全val) →
#      评 LoRA(全val) → 对齐对比
# 产出论文 headline:baseline vs LoRA r=128,per-scene 全指标
# =============================================================================
# [v2.9] 本脚本属【已退役的 legacy 链】(SEM_H5 = identity 中和位姿)。
# 声明 legacy profile 以保持 SEM_ARM_MODE=identity,复现旧数字;
# 正式 ICLR 矩阵走 official profile(SEM_ARM_MODE=drop),见 common.sh。
export RSWM_PROFILE=legacy
# [v2.9.4] 产物角色 —— 进 run_manifest,并参与 headline 资格判定
export RSWM_RUN_ROLE=legacy
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# ---- P1.1 canonical 数据集(几何版 + 语义中和版) ----
if ! done_p p1_build; then
    echo "[P1.1] 构建 $N_MAIN 样本 canonical 数据集(tier=$DATASET_TIER/128x222/steps9,~1h,首次含 HF 下载)"
    # [v2.4.3 修 P0-2] 重建前清全部下游 stamp(陈旧门/训练 stamp 旁路)
    rm -f "$STAMPS"/p1_dupgate* "$STAMPS"/p1_splitgate* "$STAMPS"/p1_neutralize* \
          "$STAMPS"/p1_train* "$STAMPS"/p1_eval* "$STAMPS"/p1_aligned*
    ( cd "$ROOT" && $PY lora_remotesense/build_dataset_from_benchmark.py \
        -o "$GEO_H5" -n "$N_MAIN" --dataset-tier "$DATASET_TIER" \
        --temp-dir "$WORK/tmp_bench_main" ) 2>&1 | tee "$LOGS/p1_build.log"
    stamp p1_build
fi

# [v2.4.3] 三道门与 pbs/01_build_data.pbs 完全同构(sha 绑定,H5 变必重跑):
# 门1 重复/源身份(--expect-zero)→ 门2 train/val 泄漏(预生成确定性 split)
# → neutralize。任何 FAIL 当场停,不进训练(§九:旧/坏数据不得被自动复用)。
GEO_SHA=$(h5_sha "$GEO_H5")
if ! sha_stamp_ok "$STAMPS/p1_dupgate.sha256" "$GEO_SHA"; then
    _rc=0
    $PY "$ROOT/server_run/check_benchmark_duplicates.py" \
        --gt-h5 "$GEO_H5" --expect-zero --expect-n "$N_MAIN" \
        -o "$WORK/duplicate_report_build.json" \
        2>&1 | tee "$LOGS/p1_dupcheck.log" || _rc=$?
    [ "$_rc" -eq 0 ] || { echo "FATAL: DUPLICATE GATE FAIL (exit $_rc) —— 不放行训练;看 $WORK/duplicate_report_build.json"; exit "$_rc"; }
    sha_stamp "$STAMPS/p1_dupgate.sha256" "$GEO_SHA"
fi
# --- [v2.7] 语义一致性门 ---
if ! sha_stamp_ok "$STAMPS/p1_semgate.sha256" "$GEO_SHA"; then
    _rc=0
    $PY "$ROOT/server_run/check_semantic_consistency.py" \
        --gt-h5 "$GEO_H5" -o "$WORK/semantic_report.json" \
        2>&1 | tee "$LOGS/p1_semcheck.log" || _rc=$?
    [ "$_rc" -eq 0 ] || { echo "FATAL: SEMANTIC GATE FAIL (exit $_rc)"; exit "$_rc"; }
    sha_stamp "$STAMPS/p1_semgate.sha256" "$GEO_SHA"
fi
if ! sha_stamp_ok "$STAMPS/p1_splitgate.sha256" "$GEO_SHA"; then
    _rc=0
    $PY "$ROOT/server_run/make_precheck_split.py" --gt-h5 "$GEO_H5" \
        -o "$WORK/train_val_split_precheck.json" 2>&1 | tee "$LOGS/p1_splitgen.log" || _rc=$?
    [ "$_rc" -eq 0 ] && { $PY "$ROOT/server_run/check_benchmark_duplicates.py" \
        --gt-h5 "$GEO_H5" --split "$WORK/train_val_split_precheck.json" \
        -o "$WORK/duplicate_report_split.json" 2>&1 | tee "$LOGS/p1_splitcheck.log" || _rc=$?; }
    [ "$_rc" -eq 0 ] || { echo "FATAL: SPLIT LEAKAGE GATE FAIL (exit $_rc) —— 看 $WORK/duplicate_report_split.json"; exit "$_rc"; }
    sha_stamp "$STAMPS/p1_splitgate.sha256" "$GEO_SHA"
fi
if ! sha_stamp_ok "$STAMPS/p1_neutralize.sha256" "$GEO_SHA"; then
    $PY "$ROOT/server_run/neutralize_poses.py" "$GEO_H5" "$SEM_H5"
    sha_stamp "$STAMPS/p1_neutralize.sha256" "$GEO_SHA"
fi
SPLIT="$T_MAIN/train_val_split.json"

# ---- P1.2 主训练 r16/a16(语义通道)[v2.7.1 F10: r128→r16,对齐 D3 主线拍板] ----
if ! done_p p1_train; then
    train_lora r16_sem 16 16 "$SEM_H5" "$T_MAIN"      # alpha=rank;r128 移入 p2 rank 环
    stamp p1_train
fi

# ---- P1.3 baseline 评估(全 val≈100;整个 sweep 只评这一次) ----
if ! done_p p1_evalbase; then
    eval_base main "$SEM_H5" "$SPLIT" "$E_BASE" 0
    stamp p1_evalbase
fi

# ---- P1.4 LoRA 评估(全 val) ----
if ! done_p p1_evallora; then
    # [v2.7.3 ①/P0 修复] v2.7.2 用"前缀替换+行内注释"改本行,注释吞掉了后面
    # 4 个参数 → common.sh 的 eval_lora 需 6 参,set -u 下 P1 在训练+baseline
    # 跑完后当场中止。教训:改带续参的调用行必须整行替换。
    eval_lora r16_sem "$T_MAIN/best_checkpoint" \
        "$SEM_H5" "$SPLIT" "$E_MAIN" 0
    stamp p1_evallora
fi

# ---- P1.5 对齐对比 ----
if ! done_p p1_aligned; then
    aligned main "$E_MAIN" "$E_BASE" "$WORK/aligned_main"
    stamp p1_aligned
fi

echo
echo "=========================================================="
echo " P1 完成 ✓  主表数字: $WORK/aligned_main/comparison.json"
echo " 下一步: bash server_run/p2_ranks.sh"
echo "=========================================================="
