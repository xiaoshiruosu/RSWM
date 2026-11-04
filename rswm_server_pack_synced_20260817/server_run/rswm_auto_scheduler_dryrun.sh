#!/usr/bin/env bash
# Dry-run the scheduler state, logging, lock, and selector paths without CUDA.
set -Eeuo pipefail
umask 077

RSWM_BASE="${RSWM_DRYRUN_BASE:-/nix/rsroll_25828}"
readonly RSWM_BASE
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly WORK_DIR="$RSWM_BASE/work"
readonly LOG_DIR="$RSWM_BASE/logs"
readonly LOCK_FILE="$WORK_DIR/rswm_auto_scheduler_dryrun.lock"
readonly STATUS_FILE="$WORK_DIR/rswm_scheduler_dryrun_status.env"
readonly RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
readonly LOG_FILE="$LOG_DIR/auto_scheduler_dryrun_${RUN_ID}.log"
readonly DRYRUN_DIR="$WORK_DIR/rswm_scheduler_dryrun_${RUN_ID}"
readonly GPU_RESULT_FILE="$DRYRUN_DIR/mock_gpu_selection.env"
readonly SELECTOR_STATE_FILE="$DRYRUN_DIR/mock_gpu_selector_status.env"
readonly MOCK_NVIDIA_SMI="$DRYRUN_DIR/mock_nvidia_smi.sh"
PYTHON="${RSWM_PYTHON:-$RSWM_BASE/envs/rswm_compat_test/bin/python}"
readonly PYTHON

# Fail closed against any accidental CUDA use.  The selector receives only a
# generated nvidia-smi mock and therefore never reads or allocates a real GPU.
export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE

mkdir -p -- "$WORK_DIR" "$LOG_DIR" "$DRYRUN_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

STATE="STARTING"
CURRENT_STEP="initialization"
SELECTED_GPU=""
DETAIL="dry_run=1 no_gpu_used=1 formal_training_started=0"
CURRENT_COMMAND=""

timestamp() {
    date '+%Y-%m-%d %H:%M:%S%z'
}

write_status() {
    local temporary="${STATUS_FILE}.tmp.$$"
    {
        printf 'MODE=DRY_RUN\n'
        printf 'STATE=%q\n' "$STATE"
        printf 'SELECTED_GPU=%q\n' "$SELECTED_GPU"
        printf 'CURRENT_STEP=%q\n' "$CURRENT_STEP"
        printf 'LAST_UPDATE=%q\n' "$(timestamp)"
        printf 'LOG_FILE=%q\n' "$LOG_FILE"
        printf 'PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
        printf 'DETAIL=%q\n' "$DETAIL"
        printf 'NO_GPU_USED=1\n'
        printf 'FORMAL_TRAINING_STARTED=0\n'
    } >"$temporary"
    mv -f -- "$temporary" "$STATUS_FILE"
}

set_state() {
    STATE="$1"
    CURRENT_STEP="$2"
    DETAIL="${3:-dry_run=1 no_gpu_used=1 formal_training_started=0}"
    write_status
    echo "[DRYRUN STATE $(timestamp)] $STATE step=$CURRENT_STEP detail=$DETAIL"
}

on_error() {
    local exit_code="$1" command="${2:-unknown}"
    trap - ERR INT TERM
    set +e
    STATE="FAILED"
    DETAIL="dry_run=1 exit_code=$exit_code no_gpu_used=1 formal_training_started=0"
    write_status
    echo "DRYRUN_RESULT=FAIL"
    echo "FAILED_STEP=$CURRENT_STEP"
    echo "command=${CURRENT_COMMAND:-$command}"
    echo "log_file=$LOG_FILE"
    exit "$exit_code"
}
trap 'on_error $? "$BASH_COMMAND"' ERR
trap 'on_error 130 "SIGINT"' INT
trap 'on_error 143 "SIGTERM"' TERM

run_step() {
    local step="$1" exit_code
    shift
    CURRENT_STEP="$step"
    printf -v CURRENT_COMMAND '%q ' "$@"
    CURRENT_COMMAND="${CURRENT_COMMAND% }"
    write_status
    echo "[DRYRUN RUN $(timestamp)] step=$step"
    echo "command=$CURRENT_COMMAND"
    set +e
    "$@"
    exit_code=$?
    set -e
    echo "[DRYRUN EXIT $(timestamp)] step=$step exit_code=$exit_code"
    (( exit_code == 0 )) || return "$exit_code"
}

create_mock_nvidia_smi() {
    cat >"$MOCK_NVIDIA_SMI" <<'MOCK'
#!/usr/bin/env bash
set -euo pipefail
case " $* " in
    *" --query-compute-apps=pid "*)
        # No compute process exists on the simulated device.
        exit 0
        ;;
    *" --query-gpu="*)
        printf '%s\n' '0, NVIDIA A800-SXM4-80GB, 81920, 256, 81664, 0'
        ;;
    *)
        echo "mock_nvidia_smi: unsupported arguments: $*" >&2
        exit 2
        ;;
esac
MOCK
    chmod 700 "$MOCK_NVIDIA_SMI"
}

main() {
    local physical_gpu_id
    command -v flock >/dev/null 2>&1 || {
        echo "FATAL: flock is required for dry-run lock validation"
        return 2
    }
    command -v tee >/dev/null 2>&1 || {
        echo "FATAL: tee is required for dry-run logging"
        return 2
    }
    [[ -x "$PYTHON" ]] || {
        echo "FATAL: Python is not executable: $PYTHON"
        return 2
    }
    [[ -f "$SCRIPT_DIR/rswm_gpu_selector.py" ]] || {
        echo "FATAL: GPU selector is missing"
        return 2
    }

    exec 9>"$LOCK_FILE"
    flock -n 9 || {
        echo "FATAL: another scheduler dry-run owns $LOCK_FILE"
        return 75
    }
    if flock -n "$LOCK_FILE" true; then
        echo "FATAL: dry-run lock did not reject a second owner"
        return 2
    fi
    echo "[DRYRUN CHECK] lock=PASS"

    set_state "STARTING" "dryrun_initialization"
    echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE-<unset>}"
    echo "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE-<unset>}"
    echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE-<unset>}"

    set_state "WAITING_DATA" "simulate_resisc45_wait" \
        "dry_run=1 simulated_data_wait=1 offline_mode=unset no_gpu_used=1"
    echo "[DRYRUN CHECK] WAITING_DATA=PASS"

    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "[MODE $(timestamp)] OFFLINE_MODE_ENABLED"

    set_state "CPU_TEST" "simulate_cpu_gate" \
        "dry_run=1 simulated_cpu_test=1 no_gpu_used=1"
    echo "[DRYRUN CHECK] CPU_TEST=PASS (simulated)"

    create_mock_nvidia_smi
    run_step "GPU selector help" "$PYTHON" \
        "$SCRIPT_DIR/rswm_gpu_selector.py" --help
    set_state "WAITING_GPU" "simulate_gpu_wait" \
        "dry_run=1 selector_uses_mock=1 no_gpu_used=1"
    echo "[DRYRUN CHECK] WAITING_GPU=PASS"
    run_step "mocked stable idle GPU selection" "$PYTHON" \
        "$SCRIPT_DIR/rswm_gpu_selector.py" \
        --confirm-seconds 0.02 --poll-seconds 0.01 \
        --max-util 10 --max-used-mb 5000 \
        --visible-devices 0 \
        --result-file "$GPU_RESULT_FILE" \
        --state-file "$SELECTOR_STATE_FILE" \
        --nvidia-smi "$MOCK_NVIDIA_SMI"

    # The trusted helper writes shell-escaped scalar values atomically.
    # shellcheck disable=SC1090
    source "$GPU_RESULT_FILE"
    physical_gpu_id="${PHYSICAL_GPU_ID:-}"
    [[ "$physical_gpu_id" == "0" ]] || {
        echo "FATAL: mocked selector returned unexpected GPU '$physical_gpu_id'"
        return 2
    }
    SELECTED_GPU="MOCK_$physical_gpu_id"
    set_state "GPU_SELECTED" "mock_gpu_selected" \
        "dry_run=1 selected_gpu=MOCK_$physical_gpu_id no_gpu_used=1"
    echo "[DRYRUN CHECK] GPU_SELECTED=PASS selected=MOCK_$physical_gpu_id"

    set_state "DRYRUN_COMPLETE" "state_machine_verified" \
        "dry_run=1 lock=pass logging=pass selector=pass no_gpu_used=1 formal_training_started=0"
    echo "DRYRUN_RESULT=PASS"
    echo "NO_GPU_USED=1"
    echo "FORMAL_TRAINING_STARTED=0"
    echo "status_file=$STATUS_FILE"
    echo "log_file=$LOG_FILE"
}

main "$@"
