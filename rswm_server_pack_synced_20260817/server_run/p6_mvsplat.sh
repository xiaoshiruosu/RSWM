#!/bin/bash
# =============================================================================
# P6(可选,~4-8h + 半天调试缓冲)— MVSplat360 外部 baseline(20 样本)
# 支撑论文 domain-gap claim:通用稀疏视角 NVS 方法不能直接处理遥感 rollout。
#
# ⚠️ 环境隔离(必读):
#   setup_mvsplat360_autodl.sh 会安装 torch==2.2.2,与主训练环境
#   (torch 2.4/2.6 + flash-attn)不兼容。必须新建独立 conda env:
#       conda create -n mvsplat python=3.10 -y && conda activate mvsplat
#   本脚本检测到 flash_attn 可导入(=主环境)会直接拒跑。
#
# ⚠️ 黑帧陷阱:
#   evaluate_mvsplat360.py 单样本失败会以黑帧填充继续算指标。本脚本结束时
#   强制核查 generation_success 比例,<80% 直接判失败,防止把假的
#   domain-gap 数字写进论文。
#
# 前置:P1 已完成(需要 SEM_H5 + 主 split);SVD 权重是 gated,先在 HF
#   网页 accept license 并 huggingface-cli login。
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MVSPLAT_DIR="${MVSPLAT_DIR:-$HOME/mvsplat360}"
# [v2.6.1] 子集文件带 _geo 后缀:与 v2.5 时期 SEM 派生的旧 eval_subset_20.h5
# 物理隔离(旧文件如在,属陈旧产物,不会再被引用)。
SUBSET_H5="$DATA_DIR/eval_subset_20_geo.h5"
SUBSET_SPLIT="$DATA_DIR/eval_subset_20_geo_split.json"
OUT="$WORK/eval_mvsplat360"
N="${MVSPLAT_N:-20}"

# [v2.6 前置门] 全量前必须先过 2-5 样本冒烟(拍板②档B)。
# [v2.6.1 审查修复 F1] 不能只查文件存在:run_baseline 刻意「先写 scores.json
# 再判 min-success」(保留失败现场供排查,exit 4),所以部分失败的冒烟也会
# 留下 scores.json —— 存在性检查会被它永久旁路。改为解析内容判决。
SMOKE_SCORES="$WORK/mvsplat_smoke/bridge/scores.json"
[ -f "$SMOKE_SCORES" ] || {
    echo "FATAL: 未见冒烟产物 $SMOKE_SCORES。"
    echo "       先跑 bash server_run/mvsplat_smoke.sh(2-5 样本 + 黑帧门),PASS 再投全量。"
    exit 1; }
$PY - "$SMOKE_SCORES" <<'EOF' || exit 1
import json, sys
doc = json.load(open(sys.argv[1]))
rate, n_ok = doc.get("success_rate"), doc.get("n_success")
if not isinstance(rate, (int, float)) or not isinstance(n_ok, int):
    sys.exit(f"FATAL: {sys.argv[1]} 缺 success_rate/n_success(v2.6 之前的产物?)——重跑冒烟。")
if rate < 0.8 or n_ok < 2:
    sys.exit(f"FATAL: 冒烟成功率 {rate:.0%}(n_success={n_ok})未达 80%/2 —— "
             f"冒烟没 PASS,不许投全量。修好后重跑 mvsplat_smoke.sh。")
print(f"[gate] 冒烟判决通过: success_rate={rate:.0%}, n_success={n_ok}"
      f"(注意:GEO_H5 重建后冒烟需重跑,勿拿旧冒烟放行新数据)")
EOF

# ---- 0. 环境守卫:严禁在主训练环境里跑 ----
if $PY -c "import flash_attn" 2>/dev/null; then
    echo "FATAL: 当前是主训练环境(检测到 flash_attn)。"
    echo "       请先: conda create -n mvsplat python=3.10 -y && conda activate mvsplat"
    exit 1
fi

# [v2.6 修复] 子集源改 **GEO_H5**:MVSplat360 是位姿条件 NVS,旧写法用
# SEM_H5(neutralize 后全 identity 位姿)→ 5 个 context 相机完全重合,
# 渲染必然退化 —— 位姿基线拿中和位姿属既存 bug(从未跑过所以从未暴露);
# evaluate_mvsplat360.py 现已内置"全 identity 即 FATAL"守卫兜底。
[ -f "$GEO_H5" ] || { echo "FATAL: 缺 $GEO_H5,先跑 p1_main.sh"; exit 1; }
[ -f "$T_MAIN/train_val_split.json" ] || { echo "FATAL: 缺主 split,先跑 p1_main.sh"; exit 1; }

# ---- 1. 抽 20 样本评估子集(与主表同一 val 分布) ----
# [v2.6.1 审查修复 F3] 旧布尔 stamp(done_p p6_subset)会旁路 SEM→GEO 修复:
# v2.5 时期留下的 stamp + SEM 派生子集命中后不重抽 → identity 守卫每次 FATAL
# 且病因(陈旧子集)无人点名。改 sha 绑定 stamp(common.sh v2.4.3 机制:
# GEO_H5 内容一变必重抽;文件名已在顶部改 *_geo 物理隔离旧 SEM 子集)。
GEO_SHA=$(h5_sha "$GEO_H5")
if ! sha_stamp_ok "$STAMPS/p6_subset.sha256" "$GEO_SHA"; then
    # 子集变 → 下游 eval 产物与 stamp 一并清(evaluate 断点续推只看 npy 存在,
    # 不清会拿旧数据的产物给新子集打分 —— v2.4.3「重建前清下游」同一原则)
    rm -f "$SUBSET_H5" "$SUBSET_SPLIT" "$STAMPS/p6_eval.done"
    rm -rf "$OUT"
    $PY "$NEW/extract_eval_subset.py" \
        --src "$GEO_H5" --split "$T_MAIN/train_val_split.json" \
        --dst "$SUBSET_H5" --n "$N"
    $PY -c "import json; json.dump({'train_indices':[], 'val_indices':list(range($N))}, open('$SUBSET_SPLIT','w'))"
    sha_stamp "$STAMPS/p6_subset.sha256" "$GEO_SHA"
fi

# ---- 2. 装 MVSplat360 环境 + 权重(首次 ~20min + 9.5GB gated SVD) ----
if ! done_p p6_setup; then
    if [ ! -d "$MVSPLAT_DIR" ]; then
        bash "$ROOT/lora_remotesense/setup_mvsplat360_autodl.sh" 2>&1 | tee "$LOGS/p6_setup.log"
    else
        echo "[skip] $MVSPLAT_DIR 已存在"
    fi
    stamp p6_setup
fi

# ---- 3. 推理 + 指标 ----
if ! done_p p6_eval; then
    mkdir -p "$OUT"
    ( cd "$ROOT" && $PY lora_remotesense/evaluate_mvsplat360.py \
        --mvsplat360_dir "$MVSPLAT_DIR" \
        --checkpoint "$MVSPLAT_DIR/checkpoints/dl3dv_480p.ckpt" \
        --data_path "$SUBSET_H5" \
        --split_file "$SUBSET_SPLIT" \
        --output_dir "$OUT" \
        --max_samples "$N" --num_context 5 --device "${DEVICE:-cuda}" \
    ) 2>&1 | tee "$LOGS/p6_eval.log"
    stamp p6_eval
fi

# ---- 4. 黑帧自检:generation_success 必须 >= 80% ----
$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
metas = sorted(Path(sys.argv[1], "meta").glob("*.json"))
if not metas:
    sys.exit("FATAL: 无 meta 输出,推理未产出任何样本")
# [v2.6] 缺 key 默认 **False**:旧默认 True 会把「没写 flag 的 meta」当成功,
# 门形同虚设(v2.6 起 meta 恒有该字段,此处只是兜底)。
ok = sum(1 for m in metas if json.loads(m.read_text()).get("generation_success", False))
rate = ok / len(metas)
print(f"generation_success: {ok}/{len(metas)} ({rate:.0%})")
if rate < 0.8:
    sys.exit(f"FATAL: 成功率 {rate:.0%} < 80% —— 指标含大量黑帧填充,不可用于论文!"
             "\n排查 meta/*.json 的 error 字段与 p6_eval.log(常见:c2w/w2c 约定、ModelWrapper 签名)")
print("[PASS] 指标可信,可写入论文 domain-gap 对比行:", sys.argv[1] + "/eval_results.json")
EOF

echo
echo "=========================================================="
echo " P6 完成 ✓  MVSplat360 数字: $OUT/eval_results.json"
echo "=========================================================="
