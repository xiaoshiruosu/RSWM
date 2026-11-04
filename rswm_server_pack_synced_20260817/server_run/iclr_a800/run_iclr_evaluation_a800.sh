#!/usr/bin/env bash
# Future-only unified evaluation and table export for completed standardized runs.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

MODE="plan"
ACTION="both"
DATASET="aid"
GROUP="lower_bound"
ONLY_METHOD=""
METRICS="core"
RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
export RSWM_BASE
RESULT_ROOT="$RSWM_BASE/work/rsroll_iclr_baselines"
TABLE_ROOT="$RSWM_BASE/work/rsroll_iclr_paper_results"

usage() {
    cat <<'EOF'
Usage: bash server_run/iclr_a800/run_iclr_evaluation_a800.sh [options]

  --plan                 Print exact evaluation/export commands (default).
  --execute              Update real result artifacts. Requires
                         RSWM_CONFIRM_EVALUATION_RUN=YES.
  --evaluation-only      Update metrics but defer table export to a later step.
  --export-only          Export tables from existing metrics; do not re-evaluate.
  --dataset NAME         aid | whurs19 | resisc45
  --group NAME           lower_bound | primary | main
  --method NAME          Evaluate one method from the selected group.
  --metrics NAME         core | paper (default: core)
  --result-root PATH     Standardized input result root.
  --table-root PATH      CSV/export-manifest output root.

core = CPU SSIM/PSNR. paper additionally requests LPIPS, FID, R3D-FVD proxy and
standard I3D-FVD; it requires one visible A800 and local RSWM_I3D_CHECKPOINT.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --plan) MODE="plan"; shift ;;
        --execute) MODE="execute"; shift ;;
        --evaluation-only)
            [[ "$ACTION" == "both" ]] || iclr_die "select only one action mode"
            ACTION="evaluation"; shift ;;
        --export-only)
            [[ "$ACTION" == "both" ]] || iclr_die "select only one action mode"
            ACTION="export"; shift ;;
        --dataset) DATASET="${2:-}"; shift 2 ;;
        --group) GROUP="${2:-}"; shift 2 ;;
        --method) ONLY_METHOD="${2:-}"; shift 2 ;;
        --metrics) METRICS="${2:-}"; shift 2 ;;
        --result-root) RESULT_ROOT="${2:-}"; shift 2 ;;
        --table-root) TABLE_ROOT="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) iclr_die "unknown argument: $1" ;;
    esac
done
case "$DATASET" in aid|whurs19|resisc45) ;; *) iclr_die "invalid dataset: $DATASET" ;; esac
case "$GROUP" in lower_bound|primary|main) ;; *) iclr_die "invalid group: $GROUP" ;; esac
case "$METRICS" in core|paper) ;; *) iclr_die "invalid metrics mode: $METRICS" ;; esac

iclr_require_control_python
iclr_validate_plan
LIST_ARGS=(--plan "$ICLR_PLAN_FILE" --rswm-base "$RSWM_BASE" list-baselines
           --group "$GROUP" --format tsv)
[[ -n "$ONLY_METHOD" ]] && LIST_ARGS+=(--method "$ONLY_METHOD")
BASELINE_ROWS="$("$ICLR_CONTROL_PYTHON" "$ICLR_PLAN_TOOL" "${LIST_ARGS[@]}")" || \
    iclr_die "failed to materialize baseline group"

echo "================================"
echo "ICLR A800 EVALUATION PLAN"
echo "================================"
echo "MODE=$MODE ACTION=$ACTION DATASET=$DATASET GROUP=$GROUP METRICS=$METRICS"
echo "RESULT_ROOT=$RESULT_ROOT TABLE_ROOT=$TABLE_ROOT"
echo "CONDITION_FRAMES_EXCLUDED=1"

if [[ "$MODE" == "execute" ]]; then
    iclr_require_confirmation RSWM_CONFIRM_EVALUATION_RUN YES "evaluation"
    [[ -x "$ICLR_WORKLOAD_PYTHON" ]] || \
        iclr_die "workload Python is not executable: $ICLR_WORKLOAD_PYTHON"
    export RSWM_PYTHON="$ICLR_WORKLOAD_PYTHON"
    if [[ "$ACTION" != "export" && "$METRICS" == "paper" ]]; then
        iclr_require_single_visible_gpu
        iclr_require_local_asset RSWM_I3D_CHECKPOINT
        # shellcheck disable=SC1091
        source "$ICLR_PROJECT_ROOT/server_run/a800_env.sh"
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        "$ICLR_WORKLOAD_PYTHON" "$ICLR_PROJECT_ROOT/server_run/preflight_a800.py" \
            --strict --require-single-visible --min-free-gib 70
    fi
fi

RESULT_DIRS=()
while IFS=$'\t' read -r ALIAS METHOD _CONFIG _ENV _ADAPTER_ENV _GPU; do
    RESULT_DIR="$RESULT_ROOT/$DATASET/$ALIAS"
    RESULT_DIRS+=("$RESULT_DIR")
    if [[ "$ACTION" != "export" ]]; then
        COMMAND=("$ICLR_WORKLOAD_PYTHON"
                 "$ICLR_PROJECT_ROOT/server_run/evaluate_baseline_results.py"
                 --result-dir "$RESULT_DIR" --update)
        if [[ "$METRICS" == "paper" ]]; then
            COMMAND+=(--device cuda --lpips --fid --r3d-proxy
                      --i3d-checkpoint "${RSWM_I3D_CHECKPOINT:-<SET_RSWM_I3D_CHECKPOINT>}")
        else
            COMMAND+=(--device cpu)
        fi
        echo "EVALUATE=$ALIAS method=$METHOD"
        if [[ "$MODE" == "plan" ]]; then
            iclr_print_command "${COMMAND[@]}"
        else
            [[ -f "$RESULT_DIR/aggregate_metrics.json" && -f "$RESULT_DIR/frames.npz" ]] || \
                iclr_die "standardized result is incomplete: $RESULT_DIR"
            "${COMMAND[@]}"
        fi
    fi
done <<< "$BASELINE_ROWS"

EXPORT_COMMAND=("$ICLR_WORKLOAD_PYTHON" "$ICLR_PROJECT_ROOT/tools/export_paper_tables.py"
                --results "${RESULT_DIRS[@]}" --output "$TABLE_ROOT/$DATASET")
if [[ "$MODE" == "plan" ]]; then
    [[ "$ACTION" != "evaluation" ]] && iclr_print_command "${EXPORT_COMMAND[@]}"
    echo "GPU_ACCESSED=0"
    echo "RESULTS_UPDATED=0"
else
    if [[ "$ACTION" != "evaluation" ]]; then
        "${EXPORT_COMMAND[@]}"
        echo "TABLE_EXPORT=PASS"
    fi
    if [[ "$ACTION" == "export" ]]; then
        echo "RESULTS_UPDATED=0"
    else
        echo "RESULTS_UPDATED=${#RESULT_DIRS[@]}"
    fi
fi
echo "TRAINING_STARTED=0"
echo "FINAL_RESULT=PASS"
