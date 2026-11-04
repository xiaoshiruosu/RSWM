#!/bin/bash
# =============================================================================
# P0 — Smoke test(~1-2h):20 样本 / 1 epoch / 评 4 样本,端到端验证
# 通不过就不要烧后面 6 天。
# =============================================================================
# [v2.9] 本脚本属【已退役的 legacy 链】(SEM_H5 = identity 中和位姿)。
# 声明 legacy profile 以保持 SEM_ARM_MODE=identity,复现旧数字;
# 正式 ICLR 矩阵走 official profile(SEM_ARM_MODE=drop),见 common.sh。
export RSWM_PROFILE=legacy
# [v2.9.4] 产物角色 —— 进 run_manifest,并参与 headline 资格判定
export RSWM_RUN_ROLE=legacy
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SMK="$WORK/smoke"
SMK_GEO="$SMK/smoke_geo.h5"
SMK_SEM="$SMK/smoke_sem.h5"
mkdir -p "$SMK"

if ! done_p smoke_build; then
    echo "[P0.1] 构建 20 样本 canonical 数据集"
    # [v2.4.3 修 P0-2] 重建前清下游 stamp(陈旧门/训练 stamp 旁路)
    rm -f "$STAMPS"/smoke_dupgate* "$STAMPS"/smoke_splitgate* \
          "$STAMPS"/smoke_neutralize* "$STAMPS"/smoke_train* "$STAMPS"/smoke_eval*
    ( cd "$ROOT" && $PY lora_remotesense/build_dataset_from_benchmark.py \
        -o "$SMK_GEO" -n 20 --dataset-tier aid \
        --temp-dir "$SMK/tmp_bench" ) 2>&1 | tee "$LOGS/smoke_build.log"
    stamp smoke_build
fi

# [v2.4.2/v2.4.3] D3"20 样本冒烟 → 门"落地处,与全量三道门同构(sha 绑定):
# 门1 重复/源身份 → 门2 split 泄漏(预生成) → neutralize。冒烟即验全部门。
SMK_SHA=$(h5_sha "$SMK_GEO")
if ! sha_stamp_ok "$STAMPS/smoke_dupgate.sha256" "$SMK_SHA"; then
    _rc=0
    $PY "$ROOT/server_run/check_benchmark_duplicates.py" \
        --gt-h5 "$SMK_GEO" --expect-zero --expect-n 20 \
        -o "$SMK/duplicate_report_smoke.json" \
        2>&1 | tee "$LOGS/smoke_dupcheck.log" || _rc=$?
    [ "$_rc" -eq 0 ] || { echo "FATAL: SMOKE DUP GATE FAIL (exit $_rc) —— 修好再谈全量。"; exit "$_rc"; }
    sha_stamp "$STAMPS/smoke_dupgate.sha256" "$SMK_SHA"
fi
# --- [v2.7.13] 受控 P0 验收埋点:attempts 抢救 + 逐类重采统计 ---
cp "$SMK/tmp_bench/semgate_attempts.json" "$SMK/" 2>/dev/null || true
$PY "$ROOT/server_run/semgate_build_report.py" --gt-h5 "$SMK_GEO"     --attempts "$SMK/semgate_attempts.json"     -o "$SMK/semgate_build_report.json" 2>&1 | tee "$LOGS/smoke_semgate_report.log" ||     echo "[P0][注意] 语义门统计报告失败(不阻塞)"

# --- [v2.7] 门 1.5:裁剪后语义一致性(P0 人检失败的自动化) ---
if ! sha_stamp_ok "$STAMPS/smoke_semgate.sha256" "$SMK_SHA"; then
    _rc=0
    $PY "$ROOT/server_run/check_semantic_consistency.py" \
        --gt-h5 "$SMK_GEO" -o "$SMK/semantic_report_smoke.json" \
        2>&1 | tee "$LOGS/smoke_semcheck.log" || _rc=$?
    [ "$_rc" -eq 0 ] || { echo "FATAL: SMOKE SEMANTIC GATE FAIL (exit $_rc) —— 裁剪后语义与声明不符(00009/00011 模式)。"; exit "$_rc"; }
    sha_stamp "$STAMPS/smoke_semgate.sha256" "$SMK_SHA"
fi
if ! sha_stamp_ok "$STAMPS/smoke_splitgate.sha256" "$SMK_SHA"; then
    _rc=0
    $PY "$ROOT/server_run/make_precheck_split.py" --gt-h5 "$SMK_GEO" \
        -o "$SMK/train_val_split_precheck.json" 2>&1 | tee "$LOGS/smoke_splitgen.log" || _rc=$?
    [ "$_rc" -eq 0 ] && { $PY "$ROOT/server_run/check_benchmark_duplicates.py" \
        --gt-h5 "$SMK_GEO" --split "$SMK/train_val_split_precheck.json" \
        -o "$SMK/duplicate_report_smoke_split.json" 2>&1 | tee "$LOGS/smoke_splitcheck.log" || _rc=$?; }
    [ "$_rc" -eq 0 ] || { echo "FATAL: SMOKE SPLIT GATE FAIL (exit $_rc) —— 修好再谈全量。"; exit "$_rc"; }
    sha_stamp "$STAMPS/smoke_splitgate.sha256" "$SMK_SHA"
fi
if ! sha_stamp_ok "$STAMPS/smoke_neutralize.sha256" "$SMK_SHA"; then
    $PY "$ROOT/server_run/neutralize_poses.py" "$SMK_GEO" "$SMK_SEM"
    sha_stamp "$STAMPS/smoke_neutralize.sha256" "$SMK_SHA"
fi

if ! done_p smoke_train; then
    echo "[P0.2] 1-epoch r16 训练(盯 nvidia-smi,应 <70GB)"
    EPOCHS=1 ACCUM=2 train_lora smoke 16 16 "$SMK_SEM" "$SMK/train"
    stamp smoke_train
fi

if ! done_p smoke_eval; then
    echo "[P0.3] 评 4 样本(LoRA + baseline)"
    eval_lora smoke "$SMK/train/best_checkpoint" "$SMK_SEM" "$SMK/train/train_val_split.json" "$SMK/eval_lora" 4
    eval_base smoke "$SMK_SEM" "$SMK/train/train_val_split.json" "$SMK/eval_base" 4
    aligned smoke "$SMK/eval_lora" "$SMK/eval_base" "$SMK/eval_aligned"
    # [v2.7.1 F1/P0] 产物门:comparison.json 必须存在且含 overall 块。
    # 旧行为:三个评估脚本任何一个空转(全失败/无共同样本)都 exit 0,
    # 这里无条件盖 stamp、打"通过"横幅 —— GPU 一条没成功也永久标记完成,
    # 且 stamp 写入后重跑永远跳过评估段。
    if [ ! -s "$SMK/eval_aligned/comparison.json" ] || \
       ! grep -q '"overall"' "$SMK/eval_aligned/comparison.json"; then
        echo "FATAL: SMOKE EVAL GATE FAIL —— comparison.json 缺失或无 overall 块。"
        echo "       评估空转(推理全失败/无共同样本)不许盖章;查 eval_* 日志。"
        exit 1
    fi
    stamp smoke_eval
fi

echo
echo "=========================================================="
echo " P0 SMOKE 通过 ✓"
echo "   核对清单:"
echo "   1. $SMK/eval_aligned/comparison.json 存在且指标非全 NaN"
echo "      (n=4 时 FID/FVD 为 NaN 是正常的,看 SSIM/PSNR 即可)"
echo "   2. 训练日志无 OOM;峰值显存 <75GB"
echo "   3. generated npy 打开非全黑"
echo " 通过后运行: bash server_run/p1_main.sh"
echo "=========================================================="
