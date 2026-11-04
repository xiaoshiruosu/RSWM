#!/usr/bin/env bash
# =============================================================================
# qsub_ladder.sh — 难度阶梯编排(50 建库 → 51 array 训评 → 登录节点出报告)
#
# 依赖链:
#   50_ladder_build.pbs         纯 CPU(normal 队列),按 ladder_spec.json 建 N 档数据集
#     → 51_ladder_train_eval.pbs  GPU array 作业 -J 0-(N-1),每元素训评一档
#     → (登录节点,纯 CPU) server_run/ladder_report.py 出难度曲线表
#
# 用法(包根):
#   DRYRUN=1 bash server_run/pbs/qsub_ladder.sh                      # 只打印,不提交
#   bash server_run/pbs/qsub_ladder.sh                               # 真提交
#   LADDER_ROOT=/scratch/rx40/$USER/rswm_ladder bash ...             # 自定阶梯根目录
#   bash server_run/pbs/qsub_ladder.sh --pose-mode drop              # 换 drop 臂重跑整条阶梯
#   bash server_run/pbs/qsub_ladder.sh --only-build                  # 只投 50
#   bash server_run/pbs/qsub_ladder.sh --only-train                  # 只投 51(50 已绿)
#   bash server_run/pbs/qsub_ladder.sh --report                      # 不提交,直接出报告
#
# ⚠ DRYRUN 必须走【和真提交一样的检查】。qsub_corrected_rerun.sh 踩过的坑是
#   DRYRUN 绕过了唯一实质性的闸,于是"文档让人先干跑、干跑全绿"与"真提交会死"
#   同时成立。本脚本的三道预检(spec 自洽 / -J 与档数一致 / headline 接线)
#   在 DRYRUN 下【照常执行】,只有 qsub 本身被替换成打印。
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PBS="$HERE"
ROOT="$(cd "$PBS/../.." && pwd)"
DRYRUN="${DRYRUN:-0}"

SPEC="${LADDER_SPEC:-$ROOT/server_run/ladder_spec.json}"
POSE_MODE_ARG=""
ONLY=""
DO_REPORT=0

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}"; }
while [ $# -gt 0 ]; do
    case "$1" in
        --spec)       SPEC="${2:?--spec 需要参数}"; shift 2 ;;
        --pose-mode)  POSE_MODE_ARG="${2:?--pose-mode 需要参数}"; shift 2 ;;
        --only-build) ONLY="build"; shift ;;
        --only-train) ONLY="train"; shift ;;
        --report)     DO_REPORT=1; shift ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "FATAL: 未知参数 '$1'" >&2; usage >&2; exit 1 ;;
    esac
done
case "${POSE_MODE_ARG:-native}" in
    native|drop|identity) ;;
    *) echo "FATAL: --pose-mode 必须是 native|drop|identity" >&2; exit 1 ;;
esac

# 本脚本【不】source common.sh —— 它需要 CHECKPOINT_DIR 等一堆只有计算节点才对的
# 变量,而且 set -u 下缺一个就把编排器整个杀掉(qsub_corrected_rerun.sh 的 $PY
# unbound 事故)。这里只用系统 python3。
PYBIN="${PY:-$(command -v python3 || command -v python || true)}"
[ -n "$PYBIN" ] || { echo "FATAL: 找不到 python3/python" >&2; exit 1; }

[ -f "$SPEC" ] || { echo "FATAL: 缺 spec $SPEC" >&2; exit 1; }
for f in 50_ladder_build 51_ladder_train_eval; do
    [ -f "$PBS/${f}.pbs" ] || { echo "FATAL: 缺 $PBS/${f}.pbs" >&2; exit 1; }
done

# ---- 阶梯根目录:绝不可等于主 WORK 或主 WORK/data ----
WORK_DEFAULT="$ROOT/server_run/work"
LADDER_ROOT="${LADDER_ROOT:-${WORK:-$WORK_DEFAULT}/ladder}"
_abs() { "$PYBIN" -c "import os,sys;print(os.path.realpath(sys.argv[1]))" "$1" 2>/dev/null || echo "$1"; }
for _forbid in "${WORK:-$WORK_DEFAULT}" "${WORK:-$WORK_DEFAULT}/data"; do
    if [ "$(_abs "$LADDER_ROOT")" = "$(_abs "$_forbid")" ]; then
        echo "FATAL: LADDER_ROOT=$LADDER_ROOT 指向 $_forbid —— 拒绝执行。" >&2
        echo "       阶梯的高重叠/四方向数据一旦混进主库,主表就在一个更容易的分布上" >&2
        echo "       被评了,而且完全无症状。请另指一个目录。" >&2
        exit 1
    fi
done

echo "== spec        : $SPEC"
echo "== LADDER_ROOT : $LADDER_ROOT"
echo "== pose_mode   : ${POSE_MODE_ARG:-native(spec 默认)}"
echo "== DRYRUN      : $DRYRUN"
echo

# =============================================================================
# 预检 1 —— spec 自洽(与 50 号作业里跑的是同一套判据:帧数契约 / overlap 区间 /
# index 对齐 / n 是 4 的倍数)。放在登录节点跑,不用等排队。
# =============================================================================
NRUNGS="$("$PYBIN" - "$SPEC" <<'PYCHK'
import json, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
steps = int(spec["protocol"]["steps"])
bad = []
ids = set()
for i, r in enumerate(spec["rungs"]):
    rid = r["id"]
    if rid in ids:
        bad.append(f"{rid}: id 重复 —— 两档会写进同一个目录、互相覆盖")
    ids.add(rid)
    if int(r["index"]) != i:
        bad.append(f"{rid}: index={r['index']} ≠ 数组位置 {i}")
    lo, hi = float(r["min_overlap_ratio"]), float(r["max_overlap_ratio"])
    if not (0.0 <= lo < hi <= 1.0):
        bad.append(f"{rid}: overlap 区间非法 [{lo},{hi})")
    kmin, kmax = int(r["min_segments"]), int(r["max_segments"])
    if kmin < 1 or kmin > kmax:
        bad.append(f"{rid}: 段数范围非法 [{kmin},{kmax}]")
    T = [steps + (k - 1) * (steps - 1) for k in range(kmin, kmax + 1)]
    off = [t for t in T if (t - 1) % 4 != 0]
    if off:
        bad.append(f"{rid}: 帧数 {off} 不满足 4n+1(T = steps + (K-1)(steps-1))")
    if list(r.get("frames") or []) != T:
        bad.append(f"{rid}: frames={r.get('frames')} 与推导的 {T} 不符")
    n = int(r["n_build"])
    if n <= 0 or n % 4 != 0:
        bad.append(f"{rid}: n_build={n} 必须是正的 4 的倍数")
# chains 里引用的 id 必须存在(写简称会让报告的表 3 整块静默为空)
for cname, c in (spec.get("chains") or {}).items():
    if cname.startswith("_") or not isinstance(c, dict):
        continue
    for x in c.get("rungs") or []:
        if x not in ids:
            bad.append(f"chains.{cname} 引用了不存在的档 id {x!r}")
if bad:
    sys.stderr.write("FATAL: spec 校验失败:\n" + "".join(f"   - {b}\n" for b in bad))
    sys.exit(1)
print(len(spec["rungs"]))
PYCHK
)" || exit 1
echo "[预检 1] spec 自洽:$NRUNGS 档,帧数契约/overlap 区间/index/chains 引用 全绿"

# =============================================================================
# 预检 2 —— 51 号 PBS 头里的 `-J` 默认上界必须等于 spec 档数。
# 提交时会用命令行 -J 覆盖它,但【手动 qsub】走的是头里那一行:spec 加减了档
# 而头没跟着改,手动提交会静默漏跑最后几档(或整批越界失败)。两边都得对。
# =============================================================================
_HDR_J="$(grep -m1 '^#PBS -J ' "$PBS/51_ladder_train_eval.pbs" | awk '{print $3}')"
_WANT_J="0-$((NRUNGS - 1))"
if [ "$_HDR_J" != "$_WANT_J" ]; then
    echo "FATAL: 51_ladder_train_eval.pbs 头里是 '#PBS -J $_HDR_J',而 spec 有 $NRUNGS 档" >&2
    echo "       (应为 '#PBS -J $_WANT_J')。改 spec 档数时必须同步改那一行 ——" >&2
    echo "       否则手动 qsub 会静默漏跑或越界。" >&2
    exit 1
fi
echo "[预检 2] 51 号 -J 默认上界 $_HDR_J 与 spec 档数一致"

# =============================================================================
# 预检 3 —— role=ladder 必须已被登记为"永不入 headline"。
# 这一条在登录节点先跑,是因为它是【纯代码检查】:没必要让 11 个 GPU 作业排完队、
# 跑到 51 里的同一道闸才发现接线没做。DRYRUN 下同样执行。
# =============================================================================
"$PYBIN" - "$ROOT" <<'PYWIRE'
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "lora_remotesense" / "new"))
from run_manifest import headline_eligible

ok, why = headline_eligible("official", "drop", "native", "ladder")
if ok:
    print("FATAL: role=ladder 会被判成【合格 headline】—— 接线未完成,拒绝提交。")
    print("       阶梯各档故意改过 K/角度粒度/overlap,数字进主表会让 benchmark")
    print("       看起来比它实际定义的容易,而主表没有任何字段提示这一点。")
    print("       请先在这三处登记 \"ladder\":")
    print("         1. lora_remotesense/new/run_manifest.py :: _NEVER_HEADLINE_ROLES")
    print("         2. server_run/common.sh :: _RSWM_NEVER_HEADLINE_ROLES")
    print("         3. server_run/collect_results.py :: DIAGNOSTIC_ROLES")
    print("       (server_run/test_pose_timeline.py 有用例钉住这几处一致)")
    sys.exit(2)
print(f"[预检 3] role=ladder 已登记为永不入 headline({why})")
PYWIRE

# ---- 报告模式:不提交任何作业,直接出表 ----
if [ "$DO_REPORT" = "1" ]; then
    echo
    echo "== 只出报告(纯 CPU)"
    exec "$PYBIN" "$ROOT/server_run/ladder_report.py" \
        --ladder-root "$LADDER_ROOT" --spec "$SPEC"
fi

# ---- 成本预告(从 spec 的 cost_model 直接读,不在这里另算一份)----
"$PYBIN" - "$SPEC" <<'PYCOST'
import json, sys
from pathlib import Path
spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
t = spec["cost_model"]["totals"]
print()
print(f"== 成本预告:{t['n_training_jobs']} 个训练配置")
print(f"   按任务口径的 ~10 GPU-h/配置 上界 : {t['flat_rate_ceiling_gpu_hours']} GPU-h")
print(f"   按帧数加权的成本模型估计         : {t['frame_weighted_estimate_gpu_hours']} GPU-h")
print(f"   建库(纯 CPU)                    : {t['build_job_cpu_hours_estimate']}")
print(f"   已砍掉 {len(spec.get('cut_cells') or [])} 类格子(理由见 spec.cut_cells,不是静默截断)")
PYCOST
echo

Q() { if [ "$DRYRUN" = "1" ]; then echo "qsub $*" >&2; echo "dryrun_$RANDOM"; else qsub "$@"; fi; }

# -v 传给作业的变量。WORK 只在调用者显式设了才传 —— 不传时由 _pbs_env.sh 给默认值。
V="LADDER_ROOT=$LADDER_ROOT,LADDER_SPEC=$SPEC"
[ -n "${WORK:-}" ] && V="$V,WORK=$WORK"
[ -n "${LADDER_N:-}" ] && V="$V,LADDER_N=$LADDER_N"
[ -n "${LADDER_EVAL_N:-}" ] && V="$V,LADDER_EVAL_N=$LADDER_EVAL_N"
[ -n "${DATASET_TIER:-}" ] && V="$V,DATASET_TIER=$DATASET_TIER"
[ -n "${LOCAL_AID_DIR:-}" ] && V="$V,LOCAL_AID_DIR=$LOCAL_AID_DIR"
V51="$V"
[ -n "$POSE_MODE_ARG" ] && V51="$V51,LADDER_POSE_MODE=$POSE_MODE_ARG"

J50=""
if [ "$ONLY" != "train" ]; then
    echo "===== 提交 50 建库(纯 CPU,normal 队列)====="
    J50="$(Q -v "$V" "$PBS/50_ladder_build.pbs")"
    echo "50_ladder_build = $J50"
fi

if [ "$ONLY" != "build" ]; then
    echo "===== 提交 51 训评 array($_WANT_J,共 $NRUNGS 档)====="
    if [ -n "$J50" ]; then
        # array 整体 afterok 于 50:任何一档数据没建成,一个 GPU 作业都不许起。
        J51="$(Q -W "depend=afterok:$J50" -J "$_WANT_J" -v "$V51" "$PBS/51_ladder_train_eval.pbs")"
    else
        echo "[注意] --only-train:不加依赖,请自行确认 $LADDER_ROOT 下各档 H5 已就绪"
        J51="$(Q -J "$_WANT_J" -v "$V51" "$PBS/51_ladder_train_eval.pbs")"
    fi
    echo "51_ladder_train_eval = $J51"
fi

cat <<EOM

===== 提交完毕 =====
阶梯根目录: $LADDER_ROOT

全部 array 元素跑完后,在【登录节点】出报告(纯 CPU,不占卡):
  python3 $ROOT/server_run/ladder_report.py --ladder-root $LADDER_ROOT
  # 或: bash $PBS/qsub_ladder.sh --report

单档排障(不用 array):
  qsub -v "$V51,LADDER_INDEX=<i>" $PBS/51_ladder_train_eval.pbs

只重建某一档(多档请每档投一次 —— qsub 的 -v 用逗号分隔变量,
写 LADDER_ONLY=A,B 会被 PBS 当成两个变量):
  qsub -v "$V,LADDER_ONLY=R4_k2_q4_ovH" $PBS/50_ladder_build.pbs
EOM
