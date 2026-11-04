#!/bin/bash
# =============================================================================
# mvsplat_smoke.sh — MVSplat360 档B 冒烟门(2-5 样本;v2.6 新增)
#
# 拍板②(2026-07-26):MVSplat360 属档B —— **先 2-5 样本冒烟过黑帧门,再投**
# 全量(~8-12 GPU-h)。本脚本就是那道门:
#   0. 环境守卫(独立 mvsplat env / repo / 三权重)
#   1. 从 **GEO_H5** 抽 N=2-5 样本子集
#      ⚠️ 必须 GEO:MVSplat360 是位姿条件 NVS,旧 p6 用 SEM_H5(位姿被中和成
#      identity → 5 个 context 相机重合)是既存 bug,v2.6 已连同 p6 一起修正。
#   2. evaluate_mvsplat360.py --no_metrics 推理(逐样本 gen_seed,可复现)
#   3. generation_success >= 80% 黑帧门(meta 扫描)
#   4. 桥接回收:run_baseline.py --baseline mvsplat360(record 契约 + Track B
#      窗口口径打分;target 槽黑帧兜底在桥接层,失败样本显式计数)
#   全绿 → 打印 MVSPLAT360 SMOKE: PASS,才许投全量。
#
# 冒烟要回答的两个问题(FAIL 时先查这两处):
#   a. ModelWrapper 调用契约:evaluate_mvsplat360.run_mvsplat360_single 的
#      model(model_input) 输入/输出键是按 MVSplat360 公开接口写的,从未在真
#      repo 上运行过 —— 契约不符会表现为 100% 失败 + meta.error 同一条异常。
#   b. c2w/w2c 与 intrinsics 归一化约定(meta.error 无异常但输出全黑/糊)。
#
# 用法(独立 mvsplat env,包根执行;DRYRUN=1 只打印计划):
#   conda activate mvsplat
#   export CHECKPOINT_DIR=...(common.sh 需要;本脚本不用基模)
#   MVSPLAT_DIR=/path/mvsplat360 [MVSPLAT_SMOKE_N=3] bash server_run/mvsplat_smoke.sh
# =============================================================================
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MVSPLAT_DIR="${MVSPLAT_DIR:-$HOME/mvsplat360}"
N="${MVSPLAT_SMOKE_N:-3}"
OUT="$WORK/mvsplat_smoke"
SUBSET_H5="$DATA_DIR/mvsplat_smoke_${N}.h5"
SUBSET_SPLIT="${SUBSET_H5%.h5}.json"
SPLIT_MAIN="$T_MAIN/train_val_split.json"
GEN_SEED="${GEN_SEED:-42}"

# ---- N 合法性:拍板②说的是 2-5 样本 ----
if [ "$N" -lt 2 ] || [ "$N" -gt 5 ]; then
    echo "FATAL: MVSPLAT_SMOKE_N=$N 不在 [2,5](冒烟门定义;全量走 p6_mvsplat.sh)"; exit 1
fi

# ---- 0. 环境守卫 ----
if $PY -c "import flash_attn" 2>/dev/null; then
    echo "FATAL: 当前是主训练环境(检测到 flash_attn)。MVSplat360 需要独立 env:"
    echo "       conda create -n mvsplat python=3.10 -y && conda activate mvsplat"
    echo "       (离线 wheelhouse: offline_prefetch.sh MVSPLAT=1 → wheelhouse_mvsplat/)"
    exit 1
fi
for ck in re10k.ckpt dl3dv_480p.ckpt svd.safetensors; do
    [ -f "$MVSPLAT_DIR/checkpoints/$ck" ] || {
        echo "FATAL: 缺权重 $MVSPLAT_DIR/checkpoints/$ck"
        echo "       (联网机: MVSPLAT=1 bash server_run/offline_prefetch.sh;svd 是 gated 须先 login)"
        exit 1; }
done
[ -d "$MVSPLAT_DIR/src" ] || [ -f "$MVSPLAT_DIR/config/main.yaml" ] || [ -d "$MVSPLAT_DIR/config" ] || {
    echo "FATAL: $MVSPLAT_DIR 下看不到 MVSplat360 代码(src/ 或 config/)"; exit 1; }
[ -f "$GEO_H5" ] || { echo "FATAL: 缺 $GEO_H5(先跑 01 建库)"; exit 1; }
[ -f "$SPLIT_MAIN" ] || { echo "FATAL: 缺主 split $SPLIT_MAIN(job20 产出)"; exit 1; }

if [ "${DRYRUN:-0}" = "1" ]; then
    echo "[DRYRUN] N=$N  GEO_H5=$GEO_H5  MVSPLAT_DIR=$MVSPLAT_DIR"
    echo "[DRYRUN] 会写: $SUBSET_H5 / $OUT/{generated,gt,meta} / $OUT/bridge/scores.json"
    echo "[DRYRUN] 不提交不生成。"; exit 0
fi

mkdir -p "$OUT"

# ---- 1. GEO 子集(⚠️ 不是 SEM —— 见文件头) ----
# [v2.6.1 审查修复 F3 同类] 子集与 GEO_H5 内容 sha 绑定:GEO 重建后陈旧子集
# 必重抽,绝不拿旧数据冒烟放行新数据(布尔存在性检查是 v2.4.3 已知旁路形态)。
GEO_SHA=$(h5_sha "$GEO_H5")
if ! sha_stamp_ok "$STAMPS/mvsplat_smoke_subset.sha256" "$GEO_SHA"; then
    # 子集变 → 下游产物一并清(evaluate 断点续推只看 npy 是否存在,
    # 不清会拿旧 GEO 的产物给新子集打分 —— 与 v2.4.3「重建前清下游」同一原则)
    rm -f "$SUBSET_H5" "$SUBSET_SPLIT"
    rm -rf "$OUT"
    mkdir -p "$OUT"
    $PY "$NEW/extract_eval_subset.py" \
        --src "$GEO_H5" --split "$SPLIT_MAIN" --dst "$SUBSET_H5" --n "$N" \
        2>&1 | tee "$LOGS/mvsplat_smoke_subset.log"
    sha_stamp "$STAMPS/mvsplat_smoke_subset.sha256" "$GEO_SHA"
fi
[ -f "$SUBSET_SPLIT" ] || { echo "FATAL: 子集 split $SUBSET_SPLIT 未产出"; exit 1; }

# ---- 2. 推理(--no_metrics:打分统一走 record 契约,不用 legacy 口径) ----
( cd "$ROOT" && $PY lora_remotesense/evaluate_mvsplat360.py \
    --mvsplat360_dir "$MVSPLAT_DIR" \
    --checkpoint "$MVSPLAT_DIR/checkpoints/dl3dv_480p.ckpt" \
    --data_path "$SUBSET_H5" --split_file "$SUBSET_SPLIT" \
    --output_dir "$OUT" --max_samples "$N" --num_context 5 \
    --gen_seed "$GEN_SEED" --device "${DEVICE:-cuda}" --no_metrics \
) 2>&1 | tee "$LOGS/mvsplat_smoke_eval.log"

# ---- 3. 黑帧门:generation_success >= 80% ----
$PY - "$OUT" <<'EOF'
import json, sys
from pathlib import Path
metas = sorted(Path(sys.argv[1], "meta").glob("*.json"))
if not metas:
    sys.exit("FATAL: 无 meta 输出 —— 推理一个样本都没跑出来(查 mvsplat_smoke_eval.log)")
docs = [json.loads(m.read_text()) for m in metas]
ok = sum(1 for d in docs if d.get("generation_success", False))
rate = ok / len(docs)
print(f"generation_success: {ok}/{len(docs)} ({rate:.0%})")
for d in docs:
    if not d.get("generation_success", False):
        print(f"  [fail] {d.get('key')}: {d.get('error')}")
if rate < 0.8:
    sys.exit("FATAL: 成功率 < 80% —— 先修 ModelWrapper 契约 / 位姿约定,再谈全量(见脚本头 a/b)")
EOF

# ---- 4. 桥接回收 + record 契约打分(Track B 窗口口径) ----
MANIFEST="$OUT/smoke_manifest.json"
$PY - "$SUBSET_H5" "$SUBSET_SPLIT" "$GEN_SEED" "$MANIFEST" <<'EOF'
import json, sys
h5, split, seed, out = sys.argv[1:5]
json.dump({"h5": h5, "split": split, "num_frames": None,
           "resolution": [480, 832], "gen_seed": int(seed)}, open(out, "w"), indent=2)
print(f"[manifest] {out}")
EOF
RSWM_MVSPLAT_EVAL_DIR="$OUT" MVSPLAT_DIR="$MVSPLAT_DIR" \
    $PY "$ROOT/server_run/baselines/run_baseline.py" \
    --baseline mvsplat360 --manifest "$MANIFEST" \
    --out "$OUT/bridge" --skip-failed --min-success 0.8 \
    2>&1 | tee "$LOGS/mvsplat_smoke_bridge.log"

echo
echo "=========================================================="
echo " MVSPLAT360 SMOKE: PASS"
echo "   产物: $OUT/bridge/scores.json(record 契约 + Track B 窗口口径)"
echo "   下一步(才许投全量): p6_mvsplat.sh(20 样本;它现在同样走 GEO_H5)"
echo "   论文脚注义务: 5 context views + poses,条件多于 RSWM;"
echo "                 context 槽位按协议填 GT 帧,SSIM 类指标有天然抬高。"
echo "=========================================================="
