#!/usr/bin/env bash
# Generate/ingest unified baseline rollouts on frozen benchmark manifests.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

MODE="plan"
DATASET="aid"
GROUP="lower_bound"
ONLY_METHOD=""
RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
export RSWM_BASE
OUTPUT_ROOT="${RSWM_BASE}/work/rsroll_iclr_baselines"

usage() {
    cat <<'EOF'
Usage: bash server_run/iclr_a800/run_iclr_baselines_a800.sh [options]

  --plan                 Print dry-run commands only (default).
  --server-dry-run       Validate real frozen inputs/adapters; no generation.
  --execute              Generate/ingest baseline results. Requires
                         RSWM_CONFIRM_BASELINE_RUN=YES.
  --dataset NAME         aid | whurs19 | resisc45 (default: aid)
  --group NAME           lower_bound | primary | main (default: lower_bound)
  --method NAME          Run one method from the selected group.
  --output-root PATH     Standardized result root.

The main group includes configured bridges whose upstream products/checkpoints are
not bundled. Execute mode fails closed unless every declared local asset is staged.
It never substitutes persistence/black frames for a missing external method.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --plan) MODE="plan"; shift ;;
        --server-dry-run) MODE="server-dry-run"; shift ;;
        --execute) MODE="execute"; shift ;;
        --dataset) DATASET="${2:-}"; shift 2 ;;
        --group) GROUP="${2:-}"; shift 2 ;;
        --method) ONLY_METHOD="${2:-}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) iclr_die "unknown argument: $1" ;;
    esac
done
case "$DATASET" in aid|whurs19|resisc45) ;; *) iclr_die "invalid dataset: $DATASET" ;; esac
case "$GROUP" in lower_bound|primary|main) ;; *) iclr_die "invalid baseline group: $GROUP" ;; esac

iclr_require_control_python
iclr_validate_plan
MANIFEST="$($ICLR_CONTROL_PYTHON "$ICLR_PLAN_TOOL" --plan "$ICLR_PLAN_FILE" \
    --rswm-base "$RSWM_BASE" manifest-path --dataset "$DATASET")" || \
    iclr_die "failed to resolve $DATASET manifest"
LIST_ARGS=(--plan "$ICLR_PLAN_FILE" --rswm-base "$RSWM_BASE" list-baselines
           --group "$GROUP" --format tsv)
[[ -n "$ONLY_METHOD" ]] && LIST_ARGS+=(--method "$ONLY_METHOD")
BASELINE_ROWS="$("$ICLR_CONTROL_PYTHON" "$ICLR_PLAN_TOOL" "${LIST_ARGS[@]}")" || \
    iclr_die "failed to materialize baseline group $GROUP"
[[ -n "$BASELINE_ROWS" ]] || iclr_die "selected baseline set is empty"

echo "================================"
echo "ICLR A800 BASELINE PLAN"
echo "================================"
echo "MODE=$MODE DATASET=$DATASET GROUP=$GROUP"
echo "MANIFEST=$MANIFEST"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "SCORED_FRAMES=1...T-1"

if [[ "$MODE" == "execute" ]]; then
    iclr_require_confirmation RSWM_CONFIRM_BASELINE_RUN YES "baseline execution"
    [[ -x "$ICLR_WORKLOAD_PYTHON" ]] || \
        iclr_die "workload Python is not executable: $ICLR_WORKLOAD_PYTHON"
    [[ -f "$MANIFEST" ]] || iclr_die "frozen manifest is missing: $MANIFEST"
    export RSWM_PYTHON="$ICLR_WORKLOAD_PYTHON"
    # Check every external dependency before producing the first partial result.
    while IFS=$'\t' read -r _ALIAS _METHOD _CONFIG REQUIRED_ENV _ADAPTER_ENV _REQUIRES_GPU; do
        [[ "$REQUIRED_ENV" == "-" ]] || iclr_require_local_asset "$REQUIRED_ENV"
        if [[ "$_REQUIRES_GPU" == "1" ]]; then
            iclr_require_single_visible_gpu
        fi
    done <<< "$BASELINE_ROWS"
    if [[ "$BASELINE_ROWS" == *$'\t1'* ]]; then
        # shellcheck disable=SC1091
        source "$ICLR_PROJECT_ROOT/server_run/a800_env.sh"
        export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        "$ICLR_WORKLOAD_PYTHON" "$ICLR_PROJECT_ROOT/server_run/preflight_a800.py" \
            --strict --require-single-visible --min-free-gib 70
    fi
fi

RUN_COUNT=0
while IFS=$'\t' read -r ALIAS METHOD CONFIG_PATH REQUIRED_ENV ADAPTER_ENV REQUIRES_GPU; do
    OUTPUT_DIR="$OUTPUT_ROOT/$DATASET/$ALIAS"
    if [[ "$MODE" == "execute" && "$REQUIRED_ENV" != "-" ]]; then
        printf -v ADAPTER_VALUE '%s' "${!REQUIRED_ENV}"
        export "$ADAPTER_ENV=$ADAPTER_VALUE"
    fi
    COMMAND=("$ICLR_WORKLOAD_PYTHON" "$ICLR_PROJECT_ROOT/server_run/run_baseline.py"
             --method "$METHOD" --manifest "$MANIFEST" --output "$OUTPUT_DIR"
             --experiment-id "iclr_${DATASET}_${ALIAS}")
    [[ "$CONFIG_PATH" == "-" ]] || COMMAND+=(--config "$CONFIG_PATH")
    RUN_COUNT=$((RUN_COUNT + 1))
    echo "BASELINE[$RUN_COUNT]=$ALIAS method=$METHOD required_env=$REQUIRED_ENV generation_gpu=$REQUIRES_GPU"
    if [[ "$MODE" == "plan" ]]; then
        iclr_print_command "${COMMAND[@]}" --dry-run
    elif [[ "$MODE" == "server-dry-run" ]]; then
        "${COMMAND[@]}" --dry-run
    else
        echo "BASELINE_COMMAND=$(iclr_quote_command "${COMMAND[@]}")"
        "${COMMAND[@]}"
    fi
done <<< "$BASELINE_ROWS"

echo "RUN_COUNT=$RUN_COUNT"
echo "TRAINING_STARTED=0"
[[ "$MODE" == "plan" || "$MODE" == "server-dry-run" ]] && echo "GPU_ACCESSED=0"
echo "FINAL_RESULT=PASS"
