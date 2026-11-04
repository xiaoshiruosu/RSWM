#!/usr/bin/env bash
# Safe A800 entry for the preregistered LoRA training matrix.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"

MODE="plan"
GROUP="primary"
ONLY_RUN=""
GPU=""
RESUME=0
RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
export RSWM_BASE
EXPERIMENTS_ROOT="${RSWM_EXPERIMENTS_ROOT:-$RSWM_BASE/work/rsroll_iclr_experiments}"

usage() {
    cat <<'EOF'
Usage: bash server_run/iclr_a800/run_iclr_main_a800.sh [options]

Modes (default is safe plan-only):
  --plan                 Validate and print runner dry-run commands only.
  --server-dry-run       Execute rswm_train_runner.sh --dry-run; no CUDA/training.
  --execute              Start the selected formal runs sequentially. Requires
                         RSWM_CONFIRM_FORMAL_RUN=YES.

Selection:
  --group NAME           primary | matched | target_all | rank | all
  --run EXPERIMENT_ID    Select one canonical run from the chosen group.
  --gpu PHYSICAL_ID      Optional explicit physical GPU id passed to the runner.
  --resume               Resume the selected single run from its latest complete
                         checkpoint (requires --run).
  --experiments-root P   Override formal experiment output root.
  -h, --help             Show help.

The existing READY_FOR_TRAINING scheduler gates, A800 preflight, lock, smoke prerequisites,
offline mode, manifest creation, training, and evaluation remain owned by the formal
rswm_train_runner.sh. This wrapper does not bypass any of them.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --plan) MODE="plan"; shift ;;
        --server-dry-run) MODE="server-dry-run"; shift ;;
        --execute) MODE="execute"; shift ;;
        --group) GROUP="${2:-}"; shift 2 ;;
        --run) ONLY_RUN="${2:-}"; shift 2 ;;
        --gpu) GPU="${2:-}"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --experiments-root) EXPERIMENTS_ROOT="${2:-}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) iclr_die "unknown argument: $1" ;;
    esac
done

case "$GROUP" in primary|matched|target_all|rank|all) ;; *) iclr_die "invalid group: $GROUP" ;; esac
if [[ -n "$GPU" && ! "$GPU" =~ ^[0-9]+$ ]]; then
    iclr_die "--gpu must be a non-negative numeric physical id"
fi
if (( RESUME == 1 )) && [[ -z "$ONLY_RUN" ]]; then
    iclr_die "--resume requires --run so exactly one experiment is resumed"
fi

iclr_require_control_python
iclr_validate_plan

LIST_ARGS=(--plan "$ICLR_PLAN_FILE" --rswm-base "$RSWM_BASE" list-runs
           --group "$GROUP" --format tsv)
[[ -n "$ONLY_RUN" ]] && LIST_ARGS+=(--run "$ONLY_RUN")
RUN_ROWS="$("$ICLR_CONTROL_PYTHON" "$ICLR_PLAN_TOOL" "${LIST_ARGS[@]}")" || \
    iclr_die "failed to materialize run group $GROUP"
[[ -n "$RUN_ROWS" ]] || iclr_die "selected run set is empty"

echo "================================"
echo "ICLR A800 MAIN EXPERIMENT PLAN"
echo "================================"
echo "MODE=$MODE"
echo "GROUP=$GROUP"
echo "RSWM_BASE=$RSWM_BASE"
echo "EXPERIMENTS_ROOT=$EXPERIMENTS_ROOT"
echo "PROTOCOL=single-image + continuous trajectory -> 17/25/33-frame rollout"
echo "SCORED_FRAMES=1...T-1"
echo "TRAIN_SEEDS=42,43,44 SPLIT_SEED=42 GEN_SEED=42"
echo "GPU_PROFILE=A800-SXM4-80GB PRECISION=bf16 OFFLOAD=1"
echo "RESUME=$RESUME"

if [[ "$MODE" == "execute" ]]; then
    iclr_require_confirmation RSWM_CONFIRM_FORMAL_RUN YES "formal training"
    [[ -x "$ICLR_WORKLOAD_PYTHON" ]] || \
        iclr_die "formal Python is not executable: $ICLR_WORKLOAD_PYTHON"
    export RSWM_PYTHON="$ICLR_WORKLOAD_PYTHON"
    # a800_env configures caches/storage but deliberately preserves GPU visibility.
    # shellcheck disable=SC1091
    source "$ICLR_PROJECT_ROOT/server_run/a800_env.sh"
    iclr_validate_plan --check-runtime --experiments-root "$EXPERIMENTS_ROOT" --datasets aid
fi

RUN_COUNT=0
while IFS=$'\t' read -r EXPERIMENT_ID CONFIG_PATH; do
    [[ -n "$EXPERIMENT_ID" && -n "$CONFIG_PATH" ]] || \
        iclr_die "invalid planner row: $EXPERIMENT_ID $CONFIG_PATH"
    COMMAND=(bash "$ICLR_PROJECT_ROOT/server_run/rswm_train_runner.sh"
             --config "$CONFIG_PATH" --experiment-id "$EXPERIMENT_ID"
             --experiments-root "$EXPERIMENTS_ROOT")
    [[ -n "$GPU" ]] && COMMAND+=(--gpu "$GPU")
    (( RESUME == 1 )) && COMMAND+=(--resume)
    RUN_COUNT=$((RUN_COUNT + 1))
    echo "RUN[$RUN_COUNT]=$EXPERIMENT_ID"
    if [[ "$MODE" == "plan" ]]; then
        iclr_print_command "${COMMAND[@]}" --dry-run
    elif [[ "$MODE" == "server-dry-run" ]]; then
        "${COMMAND[@]}" --dry-run
    else
        echo "FORMAL_COMMAND=$(iclr_quote_command "${COMMAND[@]}")"
        "${COMMAND[@]}"
    fi
done <<< "$RUN_ROWS"

echo "RUN_COUNT=$RUN_COUNT"
if [[ "$MODE" == "execute" ]]; then
    echo "FORMAL_RUNS_COMPLETED=$RUN_COUNT"
else
    echo "GPU_ACCESSED=0"
    echo "TRAINING_STARTED=0"
fi
echo "FINAL_RESULT=PASS"
