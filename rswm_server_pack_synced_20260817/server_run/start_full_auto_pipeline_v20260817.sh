#!/usr/bin/env bash
# Wait for one stably idle physical GPU, then start the resumable ICLR pipeline.
# Intended for tmux/nohup use; all launcher output is appended to one log.
set -Eeuo pipefail
umask 077

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

RSWM_BASE="${RSWM_BASE:-/nix/rsroll_25828}"
case "$RSWM_BASE" in
    /nix/rsroll_25828|/nix/rsroll_25828/*) ;;
    *) echo "FATAL: RSWM_BASE must stay under /nix/rsroll_25828: $RSWM_BASE" >&2; exit 2 ;;
esac
export RSWM_BASE

readonly OFFICIAL_ROOT="${RSWM_OFFICIAL_ROOT:-$RSWM_BASE/work/rsroll_official}"
readonly STATE_DIR="$OFFICIAL_ROOT/pipeline_state"
readonly LOG_DIR="$OFFICIAL_ROOT/logs/full_pipeline"
readonly LOG_FILE="${RSWM_FULL_AUTO_LOG:-$LOG_DIR/start_full_auto_pipeline_v20260817.log}"
readonly GPU_RESULT_FILE="$STATE_DIR/full_auto_gpu.env"
readonly GPU_STATE_FILE="$STATE_DIR/full_auto_gpu_poll.env"
readonly PYTHON="${RSWM_PYTHON:-$RSWM_BASE/envs/rswm_compat_test/bin/python}"

# The server-validated launcher polls every 30 seconds by default. A candidate
# must remain idle for the confirmation window before the pipeline may start.
readonly GPU_POLL_SECONDS="${RSWM_GPU_POLL_SECONDS:-30}"
readonly GPU_IDLE_CONFIRM_SECONDS="${RSWM_GPU_IDLE_CONFIRM_SECONDS:-600}"
readonly GPU_MAX_UTIL="${RSWM_GPU_MAX_UTIL:-10}"
readonly GPU_MAX_USED_MB="${RSWM_GPU_MAX_USED_MB:-5000}"

fatal() {
    echo "FATAL: $*" >&2
    exit 2
}

quote_command() {
    local quoted
    printf -v quoted '%q ' "$@"
    printf '%s' "${quoted% }"
}

[[ "$GPU_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    fatal "RSWM_GPU_POLL_SECONDS must be a positive integer"
[[ "$GPU_IDLE_CONFIRM_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
    fatal "RSWM_GPU_IDLE_CONFIRM_SECONDS must be a positive integer"
[[ "${RSWM_SEMGATE:-1}" == "1" ]] || \
    fatal "RSWM_SEMGATE must remain 1 for the paper pipeline"
export RSWM_SEMGATE=1

for confirmation in RSWM_CONFIRM_FULL_PIPELINE RSWM_CONFIRM_FORMAL_RUN \
        RSWM_CONFIRM_BASELINE_RUN RSWM_CONFIRM_EVALUATION_RUN; do
    [[ "${!confirmation:-}" == "YES" ]] || \
        fatal "export $confirmation=YES before starting the full auto pipeline"
done

for command in bash flock nvidia-smi tee; do
    command -v "$command" >/dev/null 2>&1 || fatal "required command is unavailable: $command"
done
[[ -x "$PYTHON" ]] || fatal "Python is not executable: $PYTHON"
[[ -f "$SCRIPT_DIR/rswm_gpu_selector.py" ]] || fatal "GPU selector is missing"
[[ -f "$SCRIPT_DIR/run_iclr_full_pipeline.sh" ]] || fatal "full pipeline entry is missing"

mkdir -p -- "$STATE_DIR" "$LOG_DIR"
exec 9>"$STATE_DIR/full_auto_pipeline.lock"
flock -n 9 || fatal "another full-auto launcher owns $STATE_DIR/full_auto_pipeline.lock"
exec > >(tee -a "$LOG_FILE") 2>&1

CURRENT_PHASE="gpu_wait"
on_exit() {
    local rc="$?"
    echo "FULL_AUTO_EXIT=$(date -u +%Y-%m-%dT%H:%M:%SZ) phase=$CURRENT_PHASE exit_code=$rc"
}
trap on_exit EXIT

echo "FULL_AUTO_START=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "LOG_FILE=$LOG_FILE"
echo "GPU_POLL_SECONDS=$GPU_POLL_SECONDS"
echo "GPU_IDLE_CONFIRM_SECONDS=$GPU_IDLE_CONFIRM_SECONDS"
echo "GPU_THRESHOLDS=util<$GPU_MAX_UTIL used_mb<$GPU_MAX_USED_MB"
echo "RSWM_SEMGATE=1"
echo "RSWM_SEMGATE_MIN_FRAC=${RSWM_SEMGATE_MIN_FRAC:-<frozen-default>}"
echo "RSWM_SEMGATE_MIN_FRAC_WATER=${RSWM_SEMGATE_MIN_FRAC_WATER:-<inherit>}"
echo "RSWM_SEMGATE_MIN_FRAC_URBAN=${RSWM_SEMGATE_MIN_FRAC_URBAN:-<inherit>}"
echo "RSWM_SEMGATE_MIN_FRAC_RURAL=${RSWM_SEMGATE_MIN_FRAC_RURAL:-<inherit>}"

ORIGINAL_CVD_WAS_SET=0
ORIGINAL_CVD=""
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    ORIGINAL_CVD_WAS_SET=1
    ORIGINAL_CVD="$CUDA_VISIBLE_DEVICES"
fi
readonly ORIGINAL_CVD_WAS_SET ORIGINAL_CVD

while true; do
    selector_command=(
        "$PYTHON" "$SCRIPT_DIR/rswm_gpu_selector.py"
        --confirm-seconds "$GPU_IDLE_CONFIRM_SECONDS"
        --poll-seconds "$GPU_POLL_SECONDS"
        --max-util "$GPU_MAX_UTIL"
        --max-used-mb "$GPU_MAX_USED_MB"
        --result-file "$GPU_RESULT_FILE"
        --state-file "$GPU_STATE_FILE"
    )
    if (( ORIGINAL_CVD_WAS_SET == 1 )); then
        selector_command+=(--visible-devices "$ORIGINAL_CVD")
    fi
    echo "[gpu-wait] $(quote_command "${selector_command[@]}")"
    "${selector_command[@]}"

    unset PHYSICAL_GPU_ID BASELINE_COMPUTE_PIDS
    # Written atomically with mode 0600 by rswm_gpu_selector.py.
    # shellcheck disable=SC1090
    source "$GPU_RESULT_FILE"
    GPU="${PHYSICAL_GPU_ID:-}"
    [[ "$GPU" =~ ^[0-9]+$ ]] || \
        fatal "GPU selector returned a non-numeric physical id: ${GPU:-<empty>}"
    export CUDA_VISIBLE_DEVICES="$GPU"
    echo "GPU=$GPU"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    if "$PYTHON" "$SCRIPT_DIR/rswm_gpu_selector.py" \
            --check-now "$GPU" \
            --baseline-pids "${BASELINE_COMPUTE_PIDS:-}" \
            --max-util "$GPU_MAX_UTIL" \
            --max-used-mb "$GPU_MAX_USED_MB"; then
        break
    else
        selector_exit="$?"
        if [[ "$selector_exit" == "3" ]]; then
            echo "[gpu-wait] selected GPU changed before launch; polling again"
            continue
        fi
        exit "$selector_exit"
    fi
done

CURRENT_PHASE="full_pipeline"
pipeline_command=(
    bash "$SCRIPT_DIR/run_iclr_full_pipeline.sh"
    --execute
    --gpu "$GPU"
)
echo "[launch] $(quote_command "${pipeline_command[@]}")"
"${pipeline_command[@]}"

CURRENT_PHASE="complete"
echo "FINAL_RESULT=PASS"
