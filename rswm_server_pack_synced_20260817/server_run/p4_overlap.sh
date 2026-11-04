#!/bin/bash
# =============================================================================
# P4 — Overlap 消融(~15-20h @ OVERLAP_N=50):
#   build_overlap_benchmark.py 拒绝采样 4 档(trivial/easy/medium/hard)
#   → convert 480x832 → LoRA(P1 ckpt) + baseline 各档评估 → re-bin 重分箱
# 这次 baseline 4 档全跑(补上当年 walltime 被杀的 medium/hard)。
# 流程逐行对齐 ablation_overlap_v2_eval.pbs / ablation_overlap_v2_rebin.pbs。
# =============================================================================
# [v2.9.4] 产物角色 —— 进 run_manifest,并参与 headline 资格判定
export RSWM_PROFILE=legacy
export RSWM_RUN_ROLE=legacy
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

LORA_CKPT="$T_MAIN/best_checkpoint"
[ -d "$LORA_CKPT" ] || { echo "FATAL: 缺 P1 checkpoint $LORA_CKPT"; exit 1; }
mkdir -p "$OVBASE"

LEVELS=(trivial easy medium hard)

for LEVEL in "${LEVELS[@]}"; do
    LDIR="$OVBASE/$LEVEL"
    BENCH_H5="$LDIR/benchmark/benchmark.h5"
    DATA_H5="$LDIR/eval_data.h5"
    SPLIT="$LDIR/split.json"
    mkdir -p "$LDIR"

    # ---- 4.1 构建该档 benchmark(拒绝采样, 25/scene = 100/档) ----
    if ! done_p "p4_build_$LEVEL"; then
        echo "[P4.1] build $LEVEL benchmark"
        ( cd "$ROOT" && $PY lora_remotesense/build_overlap_benchmark.py \
            --output-base "$OVBASE" \
            --instances-per-scene 25 \
            --seed "$SEED" \
            --dataset-tier aid \
            --levels "$LEVEL" \
            ${LOCAL_AID_DIR:+--local-aid-dir "$LOCAL_AID_DIR"} \
        ) 2>&1 | tee "$LOGS/p4_build_$LEVEL.log"
        stamp "p4_build_$LEVEL"
    fi

    # ---- 4.2 转 LingBot 格式(与 pbs 完全一致的调用) ----
    if ! done_p "p4_convert_$LEVEL"; then
        ( cd "$ROOT" && $PY -c "
import sys; sys.path.insert(0, '.')
from lora_remotesense.build_dataset_from_benchmark import convert_benchmark_to_lingbot
convert_benchmark_to_lingbot('$BENCH_H5', '$DATA_H5', target_size=(480, 832))
" ) 2>&1 | tee "$LOGS/p4_convert_$LEVEL.log"
        NSAMP=$($PY -c "import h5py; f=h5py.File('$DATA_H5','r'); print(len(f['data'].keys())); f.close()")
        $PY -c "import json; json.dump({'train_indices':[], 'val_indices':list(range($NSAMP))}, open('$SPLIT','w'))"
        echo "$LEVEL: $NSAMP samples"
        stamp "p4_convert_$LEVEL"
    fi

    # ---- 4.3 LoRA 评估 ----
    if ! done_p "p4_lora_$LEVEL"; then
        eval_lora "ov_${LEVEL}_lora" "$LORA_CKPT" "$DATA_H5" "$SPLIT" "$LDIR/eval_lora" "$OVERLAP_N"
        stamp "p4_lora_$LEVEL"
    fi

    # ---- 4.4 baseline 评估(本次 4 档全补齐) ----
    if ! done_p "p4_base_$LEVEL"; then
        eval_base "ov_${LEVEL}" "$DATA_H5" "$SPLIT" "$LDIR/eval_baseline" "$OVERLAP_N"
        stamp "p4_base_$LEVEL"
    fi
done

# ---- 4.5 re-bin 重分箱(池化 4 档按真实 overlap_ratio 重算) ----
for METHOD in lora baseline; do
    for STRAT in quartile two_level; do
        TAG="p4_rebin_${METHOD}_${STRAT}"
        if ! done_p "$TAG"; then
            ( cd "$ROOT" && $PY lora_remotesense/rebin_overlap_metrics.py \
                --base-dir "$OVBASE" \
                --method "$METHOD" \
                --strategy "$STRAT" \
                --output-dir "$OVBASE/rebin_${METHOD}_${STRAT}" \
                --device "$DEVICE" \
            ) 2>&1 | tee "$LOGS/$TAG.log"
            stamp "$TAG"
        fi
    done
done

echo
echo "=========================================================="
echo " P4 完成 ✓  overlap 表数字:"
echo "   各档原始   -> $OVBASE/<level>/eval_{lora,baseline}/eval_results.json"
echo "   重分箱汇总 -> $OVBASE/rebin_{lora,baseline}_{quartile,two_level}/"
echo " 下一步: bash server_run/p5_quals.sh"
echo "=========================================================="
